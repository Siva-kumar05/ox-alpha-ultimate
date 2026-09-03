"""Order-book ingestion and conservative order-flow admission checks.

This module deliberately works from an actual market-by-price book.  It does
not relabel OHLCV candle volume as order flow, and it never claims queue
position, individual-order priority, or aggressor-side trade classification
when the feed does not provide those fields.
"""

from __future__ import annotations

import math
import struct
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from .core import iso


class DepthParseError(ValueError):
    """Raised when a depth packet cannot be safely decoded."""


@dataclass(frozen=True)
class BookLevel:
    price: float
    quantity: int
    order_count: int


@dataclass(frozen=True)
class DepthPacket:
    security_id: str
    side: str
    sequence: int
    levels: tuple[BookLevel, ...]


@dataclass(frozen=True)
class OrderFlowAssessment:
    symbol: str
    source: str
    observed_at: float
    observations: int
    bid_price: float
    ask_price: float
    mid_price: float
    microprice: float
    spread_bps: float
    book_imbalance: float
    flow_imbalance: float
    pressure_ema: float
    positive_streak: int
    liquidity_score: float
    book_state: str
    microprice_edge_bps: float
    bid_notional: float
    ask_notional: float
    ready: bool
    long_entry: bool
    long_exit: bool
    reason: str

    def details(self) -> dict[str, float | int | str | bool]:
        return {
            "source": self.source,
            "observations": self.observations,
            "bid": round(self.bid_price, 2),
            "ask": round(self.ask_price, 2),
            "spread_bps": round(self.spread_bps, 3),
            "book_imbalance": round(self.book_imbalance, 4),
            "flow_imbalance": round(self.flow_imbalance, 4),
            "pressure_ema": round(self.pressure_ema, 4),
            "positive_streak": self.positive_streak,
            "liquidity_score": round(self.liquidity_score, 4),
            "book_state": self.book_state,
            "microprice_edge_bps": round(self.microprice_edge_bps, 3),
            "bid_notional": round(self.bid_notional, 2),
            "ask_notional": round(self.ask_notional, 2),
            "ready": self.ready,
            "reason": self.reason,
        }


class DhanDepthParser:
    """Parse documented Dhan 20-level full-market-depth packets.

    Dhan sends a 12-byte header followed by 20 * 16-byte entries for each
    bid or ask packet.  Packets may be stacked in one WebSocket message.
    """

    HEADER_SIZE = 12
    LEVEL_SIZE = 16
    LEVELS = 20
    PACKET_SIZE = HEADER_SIZE + LEVEL_SIZE * LEVELS
    BID_CODE = 41
    ASK_CODE = 51

    @classmethod
    def parse(cls, message: bytes) -> list[DepthPacket]:
        if not isinstance(message, (bytes, bytearray, memoryview)):
            raise DepthParseError("depth message must be binary")
        payload = bytes(message)
        packets: list[DepthPacket] = []
        cursor = 0
        while cursor < len(payload):
            remaining = len(payload) - cursor
            if remaining < cls.HEADER_SIZE:
                raise DepthParseError("truncated Dhan depth header")
            message_length, response_code, _segment, security_id, sequence = struct.unpack_from("<HBBII", payload, cursor)
            # Dhan documents the message length as the entire 332-byte 20-L2
            # packet. Strict parsing avoids silently accepting a mismatched
            # payload layout as a tradable market state.
            if message_length != cls.PACKET_SIZE:
                raise DepthParseError(f"unexpected Dhan depth packet length: {message_length}")
            if remaining < cls.PACKET_SIZE:
                raise DepthParseError("truncated Dhan depth packet")
            if response_code in {cls.BID_CODE, cls.ASK_CODE}:
                levels: list[BookLevel] = []
                for index in range(cls.LEVELS):
                    offset = cursor + cls.HEADER_SIZE + index * cls.LEVEL_SIZE
                    price, quantity, order_count = struct.unpack_from("<dII", payload, offset)
                    if not math.isfinite(price) or price <= 0 or quantity <= 0:
                        continue
                    levels.append(BookLevel(float(price), int(quantity), int(order_count)))
                if not levels:
                    raise DepthParseError("Dhan depth packet contains no usable levels")
                side = "BID" if response_code == cls.BID_CODE else "ASK"
                packets.append(DepthPacket(str(security_id), side, int(sequence), tuple(levels)))
            cursor += cls.PACKET_SIZE
        return packets


class OrderFlowReplayValidator:
    """Validate the primary L2 gate against retained real-depth snapshots.

    Dhan's depth feed does not provide historical executions.  This validator
    therefore measures the *admission gate* only: whether real, recorded
    ``DHAN_DEPTH20`` entry snapshots were followed by a favourable recorded
    candle move.  It is deliberately not labelled a full execution backtest.
    """

    def __init__(self, cfg, db):
        self.cfg = cfg
        self.db = db
        self.rules = cfg["order_flow"]

    @staticmethod
    def _epoch(value: object) -> int | None:
        try:
            parsed = datetime.fromisoformat(str(value))
            return int(parsed.timestamp())
        except (TypeError, ValueError, OverflowError):
            return None

    def evaluate(self) -> dict[str, float | int | bool | str]:
        horizon = int(self.rules["replay_horizon_candles"])
        records = self.db.q(
            "SELECT sym,mid,ts FROM orderflow "
            "WHERE source='DHAN_DEPTH20' AND entry_signal=1 "
            "ORDER BY ofid DESC LIMIT ?",
            (int(self.rules["replay_max_records"]),),
        )
        returns_bps: list[float] = []
        seen: set[tuple[str, int]] = set()
        timeframe = int(self.cfg["timeframe_sec"])
        for symbol, midpoint, timestamp in records:
            epoch = self._epoch(timestamp)
            try:
                entry = float(midpoint)
            except (TypeError, ValueError):
                continue
            if epoch is None or not math.isfinite(entry) or entry <= 0:
                continue
            key = (str(symbol).upper(), epoch // timeframe)
            if key in seen:
                continue
            seen.add(key)
            candles = self.db.q(
                "SELECT ts,c FROM candles WHERE sym=? AND ts>? ORDER BY ts LIMIT ?",
                (key[0], epoch, horizon),
            )
            if len(candles) < horizon:
                continue
            try:
                candle_times = [int(row[0]) for row in candles]
                exit_price = float(candles[-1][1])
            except (TypeError, ValueError):
                continue
            expected_last = epoch + horizon * timeframe
            # Never turn an overnight, holiday, or large feed gap into a
            # favourable short-horizon replay observation.
            if candle_times[-1] > expected_last + 2 * timeframe:
                continue
            if any(later - earlier > 2 * timeframe for earlier, later in zip(candle_times, candle_times[1:])):
                continue
            if math.isfinite(exit_price) and exit_price > 0:
                returns_bps.append((exit_price / entry - 1.0) * 10_000.0)

        samples = len(returns_bps)
        hit_rate = float(sum(value > 0 for value in returns_bps) / samples) if samples else 0.0
        mean_return = float(sum(returns_bps) / samples) if samples else 0.0
        passed = (
            samples >= int(self.rules["replay_min_signals"])
            and hit_rate >= float(self.rules["replay_min_hit_rate"])
            and mean_return >= float(self.rules["replay_min_mean_return_bps"])
        )
        return {
            "kind": "L2_GATE_REPLAY_NOT_EXECUTION_BACKTEST",
            "source": "DHAN_DEPTH20",
            "samples": samples,
            "hit_rate": round(hit_rate, 4),
            "mean_return_bps": round(mean_return, 4),
            "horizon_candles": horizon,
            "passed": passed,
        }


class OrderFlowEngine:
    """Hold the current L2 book and derive bounded, observable flow metrics."""

    def __init__(self, cfg, db):
        self.cfg = cfg
        self.db = db
        self.rules = cfg["order_flow"]
        self._lock = threading.RLock()
        self._books: dict[str, dict[str, tuple[BookLevel, ...]]] = {}
        self._current: dict[str, OrderFlowAssessment] = {}
        self._last_signed_size: dict[str, float] = {}
        self._pressure_ema: dict[str, float] = {}
        self._positive_streak: dict[str, int] = {}
        self._observations: dict[str, int] = {}
        self._last_persisted: dict[str, float] = {}

    @staticmethod
    def _levels(levels: Iterable[BookLevel], side: str, count: int) -> tuple[BookLevel, ...]:
        valid = [
            level for level in levels
            if math.isfinite(float(level.price)) and float(level.price) > 0 and int(level.quantity) > 0
        ]
        valid.sort(key=lambda level: float(level.price), reverse=side == "BID")
        return tuple(valid[:count])

    def ingest(self, symbol: str, bids: Iterable[BookLevel], asks: Iterable[BookLevel], source: str) -> OrderFlowAssessment | None:
        """Atomically replace the public L2 snapshot for an instrument."""
        depth = int(self.rules["depth_levels"])
        bid_levels = self._levels(bids, "BID", depth)
        ask_levels = self._levels(asks, "ASK", depth)
        if not bid_levels or not ask_levels or bid_levels[0].price >= ask_levels[0].price:
            return None

        bid_notional = sum(level.price * level.quantity for level in bid_levels)
        ask_notional = sum(level.price * level.quantity for level in ask_levels)
        total_notional = bid_notional + ask_notional
        if total_notional <= 0:
            return None
        bid_size = sum(level.quantity for level in bid_levels)
        ask_size = sum(level.quantity for level in ask_levels)
        best_bid, best_ask = bid_levels[0], ask_levels[0]
        mid = (best_bid.price + best_ask.price) / 2.0
        microprice = (best_ask.price * best_bid.quantity + best_bid.price * best_ask.quantity) / max(best_bid.quantity + best_ask.quantity, 1)
        book_imbalance = (bid_notional - ask_notional) / total_notional
        signed_size = float(bid_size - ask_size)
        observed_at = time.monotonic()

        with self._lock:
            key = symbol.upper()
            previous_signed_size = self._last_signed_size.get(key)
            flow_imbalance = 0.0
            if previous_signed_size is not None:
                change = signed_size - previous_signed_size
                flow_imbalance = change / max(abs(signed_size) + abs(previous_signed_size), 1.0)
            self._last_signed_size[key] = signed_size
            observations = self._observations.get(key, 0) + 1
            self._observations[key] = observations
            self._books[key] = {"bids": bid_levels, "asks": ask_levels}

            spread_bps = (best_ask.price - best_bid.price) / mid * 10_000.0
            microprice_edge_bps = (microprice - mid) / mid * 10_000.0
            previous_pressure = self._pressure_ema.get(symbol.upper(), book_imbalance)
            alpha = float(self.rules["pressure_ema_alpha"])
            pressure_ema = alpha * book_imbalance + (1.0 - alpha) * previous_pressure
            self._pressure_ema[symbol.upper()] = pressure_ema
            supportive_snapshot = (
                book_imbalance >= float(self.rules["min_book_imbalance"])
                and microprice_edge_bps >= float(self.rules["min_microprice_edge_bps"])
            )
            positive_streak = self._positive_streak.get(symbol.upper(), 0) + 1 if supportive_snapshot else 0
            self._positive_streak[symbol.upper()] = positive_streak

            minimum_side_notional = float(self.rules["min_side_notional"])
            max_spread_bps = float(self.rules["max_spread_bps"])
            depth_quality = min(1.0, min(bid_notional, ask_notional) / (minimum_side_notional * 2.0))
            spread_quality = max(0.0, 1.0 - spread_bps / max_spread_bps)
            liquidity_score = 0.65 * depth_quality + 0.35 * spread_quality
            if book_imbalance >= float(self.rules["min_book_imbalance"]):
                book_state = "BUY_SUPPORT"
            elif book_imbalance <= -float(self.rules["min_book_imbalance"]):
                book_state = "SELL_PRESSURE"
            else:
                book_state = "BALANCED"

            fresh = True  # A snapshot received in this call is necessarily fresh.
            min_observations = int(self.rules["min_observations"])
            enough_depth = min(bid_notional, ask_notional) >= minimum_side_notional
            live_source = source == "DHAN_DEPTH20"
            source_allowed = self.cfg["mode"] != "live" or live_source
            ready = (
                fresh and source_allowed and observations >= min_observations and enough_depth
                and spread_bps <= max_spread_bps
                and liquidity_score >= float(self.rules["min_liquidity_score"])
            )
            long_entry = (
                ready
                and supportive_snapshot
                and pressure_ema >= float(self.rules["min_pressure_ema"])
                and positive_streak >= int(self.rules["min_positive_streak"])
                and flow_imbalance >= float(self.rules["min_flow_imbalance"])
            )
            long_exit = (
                ready
                and book_imbalance <= -float(self.rules["min_book_imbalance"])
                and pressure_ema <= -float(self.rules["min_pressure_ema"])
                and flow_imbalance <= -float(self.rules["min_flow_imbalance"])
            )
            reason = "FLOW_CONFIRMATION" if long_entry else self._reason(
                observations, enough_depth, spread_bps, source_allowed, liquidity_score,
                book_imbalance, flow_imbalance, pressure_ema, positive_streak, microprice_edge_bps,
            )
            assessment = OrderFlowAssessment(
                symbol=symbol.upper(), source=source, observed_at=observed_at, observations=observations,
                bid_price=best_bid.price, ask_price=best_ask.price, mid_price=mid, microprice=microprice,
                spread_bps=spread_bps, book_imbalance=book_imbalance, flow_imbalance=flow_imbalance,
                pressure_ema=pressure_ema, positive_streak=positive_streak, liquidity_score=liquidity_score,
                book_state=book_state,
                microprice_edge_bps=microprice_edge_bps, bid_notional=bid_notional, ask_notional=ask_notional,
                ready=ready, long_entry=long_entry, long_exit=long_exit, reason=reason,
            )
            self._current[symbol.upper()] = assessment
            if observed_at - self._last_persisted.get(symbol.upper(), 0.0) >= 1.0:
                self._persist(assessment)
                self._last_persisted[symbol.upper()] = observed_at
            return assessment

    def _reason(self, observations: int, enough_depth: bool, spread_bps: float, source_allowed: bool, liquidity_score: float, book: float, flow: float, pressure_ema: float, positive_streak: int, edge: float) -> str:
        if not source_allowed:
            return "LIVE_REQUIRES_DHAN_DEPTH20"
        if observations < int(self.rules["min_observations"]):
            return "FLOW_WARMUP"
        if not enough_depth:
            return "INSUFFICIENT_BOOK_NOTIONAL"
        if spread_bps > float(self.rules["max_spread_bps"]):
            return "SPREAD_TOO_WIDE"
        if liquidity_score < float(self.rules["min_liquidity_score"]):
            return "LIQUIDITY_QUALITY_LOW"
        if book < float(self.rules["min_book_imbalance"]):
            return "BID_LIQUIDITY_NOT_DOMINANT"
        if edge < float(self.rules["min_microprice_edge_bps"]):
            return "MICROPRICE_NOT_SUPPORTIVE"
        if pressure_ema < float(self.rules["min_pressure_ema"]):
            return "BOOK_PRESSURE_NOT_PERSISTENT"
        if positive_streak < int(self.rules["min_positive_streak"]):
            return "BOOK_SUPPORT_NOT_PERSISTENT"
        if flow < float(self.rules["min_flow_imbalance"]):
            return "BUY_FLOW_NOT_CONFIRMING"
        return "FLOW_NOT_CONFIRMED"

    def _persist(self, assessment: OrderFlowAssessment) -> None:
        self.db.ex(
            "INSERT INTO orderflow(sym,source,bid,ask,mid,microprice,spread_bps,book_imbalance,flow_imbalance,pressure_ema,positive_streak,liquidity_score,book_state,microprice_edge_bps,bid_notional,ask_notional,observations,ready,entry_signal,exit_signal,reason,ts)VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                assessment.symbol, assessment.source, assessment.bid_price, assessment.ask_price, assessment.mid_price,
                assessment.microprice, assessment.spread_bps, assessment.book_imbalance, assessment.flow_imbalance,
                assessment.pressure_ema, assessment.positive_streak, assessment.liquidity_score, assessment.book_state,
                assessment.microprice_edge_bps, assessment.bid_notional, assessment.ask_notional, assessment.observations,
                int(assessment.ready), int(assessment.long_entry), int(assessment.long_exit), assessment.reason, iso(),
            ),
        )
        self.db.kv_set(f"orderflow_current:{assessment.symbol}", assessment.details())
        book = self._books.get(assessment.symbol, {})
        self.db.kv_set(
            f"orderflow_book:{assessment.symbol}",
            {
                "source": assessment.source,
                "observed_at": iso(),
                "bids": [[round(level.price, 2), level.quantity, level.order_count] for level in book.get("bids", ())],
                "asks": [[round(level.price, 2), level.quantity, level.order_count] for level in book.get("asks", ())],
            },
        )

    def assessment(self, symbol: str) -> OrderFlowAssessment | None:
        with self._lock:
            assessment = self._current.get(symbol.upper())
            if assessment is None:
                return None
            age = time.monotonic() - assessment.observed_at
            if age <= float(self.rules["max_staleness_seconds"]):
                return assessment
            return OrderFlowAssessment(
                **{**assessment.__dict__, "ready": False, "long_entry": False, "long_exit": False, "reason": "STALE_DEPTH"}
            )

    def status(self, symbol: str) -> dict[str, object]:
        assessment = self.assessment(symbol)
        if assessment is None:
            return {"state": "WAITING_FOR_DEPTH", "ready": False}
        age = max(0.0, time.monotonic() - assessment.observed_at)
        return {"state": assessment.reason, "ready": assessment.ready, "source": assessment.source, "age_seconds": round(age, 2), **assessment.details()}

    def book(self, symbol: str) -> dict[str, tuple[BookLevel, ...]]:
        with self._lock:
            book = self._books.get(symbol.upper(), {})
            return {side: tuple(levels) for side, levels in book.items()}
