"""Broker adapters.

Only market-data and order-management APIs are exposed here.  There are no
fund-transfer, withdrawal, payout, or credential-persistence operations.
"""

from __future__ import annotations

import math
import os
import random
import threading
import time
import uuid
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, time as clock_time, timedelta
from typing import Any
from urllib.parse import urlencode

import requests

from .core import LOG, SecurityError, guard_endpoint, iso, now
from .orderflow import BookLevel, DepthParseError, DhanDepthParser, OrderFlowEngine


class BrokerError(RuntimeError):
    pass


class AuthenticationError(BrokerError):
    pass


class MarketDataError(BrokerError):
    pass


class RateLimitError(MarketDataError):
    """A retryable data/read throttling response from the broker."""

    def __init__(self, message: str, retry_after_seconds: float | None = None):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class OrderError(BrokerError):
    pass


@dataclass(frozen=True)
class OrderReceipt:
    order_id: str
    status: str
    filled_qty: int = 0
    average_price: float = 0.0


def _valid_price(value: Any) -> float:
    price = float(value)
    if not math.isfinite(price) or price <= 0:
        raise MarketDataError("Broker returned an invalid price")
    return price


class BrokerBase:
    name = "base"

    def __init__(self, cfg, db):
        self.cfg = cfg
        self.db = db
        self.token: str | None = None

    def authenticated(self) -> bool:
        return bool(self.token)

    def login(self) -> bool:
        raise NotImplementedError

    def ltp(self, sym: str) -> float:
        raise NotImplementedError

    def ltps(self, syms: list[str]) -> dict[str, float]:
        """Fetch a fresh quote for each symbol. Adapters may batch this request."""
        return {sym: self.ltp(sym) for sym in syms}

    def hist(self, sym: str, tf_min: int, days: int):
        raise NotImplementedError

    def place_super_order(self, sym: str, side: str, qty: int, target: float, stop: float, tag: str) -> OrderReceipt:
        raise NotImplementedError

    def wait_super_order(self, order_id: str, timeout_seconds: int) -> OrderReceipt:
        raise NotImplementedError

    def cancel_super_order(self, order_id: str) -> None:
        raise NotImplementedError

    def modify_super_target(self, order_id: str, target: float) -> None:
        raise NotImplementedError

    def exit_position(self, sym: str, side: str, qty: int, tag: str) -> OrderReceipt:
        raise NotImplementedError

    def wait_order(self, order_id: str, timeout_seconds: int) -> OrderReceipt:
        raise NotImplementedError

    def positions(self) -> list[dict]:
        raise NotImplementedError

    def whitelisted_ips(self) -> set[str] | None:
        """Return broker-confirmed IPs when supported; None means unavailable."""
        return None

    def start_orderflow(self) -> None:
        """Start a real-time order-flow feed when the adapter supports one."""

    def stop_orderflow(self) -> None:
        """Stop the order-flow feed during a controlled shutdown."""

    def order_flow(self, sym: str):
        return None

    def order_flow_status(self, sym: str) -> dict[str, object]:
        return {"state": "UNAVAILABLE", "ready": False}

    def quote_snapshot(self, syms: list[str]) -> dict[str, dict]:
        """Best-effort {sym: {last_price, volume}} day-cumulative snapshot.
        Empty dict when the adapter cannot supply it; callers must fall back."""
        return {}

    def place_protective_stop(self, sym: str, qty: int, trigger: float, tag: str) -> OrderReceipt:
        """Broker-side stop for a residual position after a partial exit (A2)."""
        raise NotImplementedError


class PaperBroker(BrokerBase):
    """Deterministic local simulation used by the smoke test and paper mode."""

    name = "paper"

    def __init__(self, cfg, db):
        super().__init__(cfg, db)
        self.px: dict[str, float] = {}
        self.pos: dict[str, dict[str, float]] = {}
        self.orders: dict[str, OrderReceipt] = {}
        self.order_meta: dict[str, dict] = {}
        self.sequence = 0
        self.lock = threading.RLock()
        self.slippage = float(cfg["costs"]["slippage_pct"]) / 100.0
        self.random = random.Random(int(cfg.get("paper_seed", 42)))
        self.initial_prices = {str(sym).upper(): float(price) for sym, price in (cfg.get("paper_prices") or {}).items()}
        self.flow = OrderFlowEngine(cfg, db)
        self.depth_sequence = 0
        # Partial-fill simulation hooks for smoke coverage of OMS hardening.
        self.partial_entry_once = False
        self.partial_exit_once = False
        # Day-cumulative volume for quote_snapshot (true share-volume candles).
        self._day_volume: dict[str, int] = {}
        self._vol_day = None

    def set_px(self, sym: str, price: float) -> None:
        self.px[sym] = _valid_price(price)
        self._publish_paper_depth(sym, self.px[sym])

    def login(self) -> bool:
        self.token = "paper-session"
        return True

    def ltp(self, sym: str) -> float:
        if sym not in self.px:
            rows = self.db.q("SELECT c FROM candles WHERE sym=? ORDER BY ts DESC LIMIT 1", (sym,))
            self.px[sym] = _valid_price(rows[0][0]) if rows else self.initial_prices.get(sym, 1000.0)
        return self.px[sym]

    def ltps(self, syms: list[str]) -> dict[str, float]:
        """Produce a bounded random walk solely for offline paper validation."""
        quotes: dict[str, float] = {}
        with self.lock:
            for sym in syms:
                previous = self.ltp(sym)
                change = self.random.gauss(0.0, previous * 0.00035)
                self.px[sym] = max(0.05, previous + change)
                quotes[sym] = self.px[sym]
                self._publish_paper_depth(sym, self.px[sym])
        return quotes

    def _publish_paper_depth(self, sym: str, price: float) -> None:
        """Generate test-only depth so the paper path exercises the same gate.

        This is intentionally labelled simulated and is never accepted as a
        live source by :class:`OrderFlowEngine`.
        """
        if not self.cfg["order_flow"]["enabled"]:
            return
        self.depth_sequence += 1
        tick = max(0.05, round(price * 0.0001, 2))
        base_quantity = max(100, int(100_000 / max(price, 1.0)))
        pulse = 1.0 + 0.04 * (self.depth_sequence % 5)
        bids = tuple(
            BookLevel(round(price - tick * (index + 1), 2), int(base_quantity * pulse * (1.30 - index * 0.012)), 1 + index % 4)
            for index in range(self.cfg["order_flow"]["depth_levels"])
        )
        asks = tuple(
            BookLevel(round(price + tick * (index + 1), 2), int(base_quantity * (0.85 + index * 0.01)), 1 + index % 4)
            for index in range(self.cfg["order_flow"]["depth_levels"])
        )
        self.flow.ingest(sym, bids, asks, "SIMULATED_DEPTH")

    def hist(self, sym: str, tf_min: int = 1, days: int = 5):
        existing = self.db.q("SELECT ts,o,h,l,c,v FROM candles WHERE sym=? ORDER BY ts", (sym,))
        if existing:
            return existing
        # Explicitly simulated bootstrap candles make paper-mode training usable
        # with no external feed.  Timestamps follow weekday NSE session windows
        # so dashboard period controls are not distorted by overnight bars.
        requested_days = max(1, min(int(days), 365))
        holidays = set(self.cfg.get("market_holidays", []))
        current = now()
        session_open = datetime.combine(current.date(), clock_time(9, 15), tzinfo=current.tzinfo)
        sessions = []
        cursor = current.date()
        if current < session_open:
            cursor -= timedelta(days=1)
        while len(sessions) < requested_days:
            if cursor.weekday() < 5 and cursor.isoformat() not in holidays:
                sessions.append(cursor)
            cursor -= timedelta(days=1)
        sessions.reverse()
        close = self.initial_prices.get(sym, 1000.0)
        candles = []
        session_minutes = 375
        now_timestamp = int(current.timestamp())
        for session_date in sessions:
            session_start = datetime.combine(session_date, clock_time(9, 15), tzinfo=current.tzinfo)
            for offset in range(0, session_minutes, max(tf_min, 1)):
                timestamp = int((session_start + timedelta(minutes=offset)).timestamp())
                if timestamp > now_timestamp:
                    break
                opening = close
                close = max(0.05, opening + self.random.gauss(0.0, opening * 0.0015))
                high = max(opening, close) * (1 + self.random.random() * 0.001)
                low = min(opening, close) * (1 - self.random.random() * 0.001)
                candles.append((timestamp, opening, high, low, close, self.random.randint(500, 5000)))
        self.px.setdefault(sym, close)
        LOG.info("Generated %s simulated paper candles for %s across %s sessions", len(candles), sym, len(sessions))
        return candles

    def _new_id(self, prefix: str = "P") -> str:
        self.sequence += 1
        return f"{prefix}{self.sequence}"

    def _fill_price(self, sym: str, side: str) -> float:
        quote = self.ltp(sym)
        return quote * (1 + self.slippage if side == "BUY" else 1 - self.slippage)

    def _apply_fill(self, sym: str, side: str, qty: int, fill: float) -> None:
        current = self.pos.setdefault(sym, {"qty": 0, "avg": 0.0})
        previous_qty = int(current["qty"])
        signed = qty if side == "BUY" else -qty
        new_qty = previous_qty + signed
        if previous_qty == 0 or previous_qty * signed > 0:
            denominator = abs(new_qty)
            current["avg"] = fill if denominator == 0 else (
                current["avg"] * abs(previous_qty) + fill * qty
            ) / denominator
        elif new_qty == 0:
            current["avg"] = 0.0
        elif previous_qty * new_qty < 0:
            current["avg"] = fill
        current["qty"] = new_qty

    def place_super_order(self, sym: str, side: str, qty: int, target: float, stop: float, tag: str) -> OrderReceipt:
        if side not in {"BUY", "SELL"} or qty <= 0:
            raise OrderError("Invalid paper Super Order")
        _valid_price(target)
        _valid_price(stop)
        with self.lock:
            fill = self._fill_price(sym, side)
            order_id = self._new_id("PS")
            filled_qty = qty
            status = "TRADED"
            if self.partial_entry_once:
                # Single-shot: a per-order guard would halve every retry too.
                self.partial_entry_once = False
                filled_qty = max(1, qty // 2)
                status = "PART_TRADED"
            self._apply_fill(sym, side, filled_qty, fill)
            receipt = OrderReceipt(order_id, status, filled_qty, round(fill, 2))
            self.orders[order_id] = receipt
            self.order_meta[order_id] = {"sym": sym, "target": target, "stop": stop, "tag": tag}
            self.db.ex(
                "INSERT OR REPLACE INTO orders VALUES(?,?,?,?,?,?,?,?,?,?)",
                (order_id, sym, side, qty, receipt.average_price, "SUPER", receipt.status, tag, iso(), self.name),
            )
            return receipt

    def wait_super_order(self, order_id: str, timeout_seconds: int) -> OrderReceipt:
        try:
            return self.orders[order_id]
        except KeyError as exc:
            raise OrderError(f"Unknown paper Super Order {order_id}") from exc

    def cancel_super_order(self, order_id: str) -> None:
        if order_id in self.orders:
            old = self.orders[order_id]
            self.orders[order_id] = OrderReceipt(order_id, "CANCELLED", old.filled_qty, old.average_price)

    def modify_super_target(self, order_id: str, target: float) -> None:
        if order_id not in self.order_meta:
            raise OrderError(f"Unknown paper Super Order {order_id}")
        self.order_meta[order_id]["target"] = _valid_price(target)

    def exit_position(self, sym: str, side: str, qty: int, tag: str) -> OrderReceipt:
        if side not in {"BUY", "SELL"} or qty <= 0:
            raise OrderError("Invalid paper exit")
        with self.lock:
            current_qty = int(self.pos.get(sym, {}).get("qty", 0))
            if (side == "SELL" and current_qty < qty) or (side == "BUY" and -current_qty < qty):
                raise OrderError(f"Paper exit quantity exceeds broker position for {sym}")
            fill = self._fill_price(sym, side)
            order_id = self._new_id("PE")
            filled_qty = qty
            status = "TRADED"
            if self.partial_exit_once:
                self.partial_exit_once = False
                filled_qty = max(1, qty // 2)
                status = "PART_TRADED"
            self._apply_fill(sym, side, filled_qty, fill)
            receipt = OrderReceipt(order_id, status, filled_qty, round(fill, 2))
            self.orders[order_id] = receipt
            self.db.ex(
                "INSERT OR REPLACE INTO orders VALUES(?,?,?,?,?,?,?,?,?,?)",
                (order_id, sym, side, qty, receipt.average_price, "MARKET", receipt.status, tag, iso(), self.name),
            )
            return receipt

    def wait_order(self, order_id: str, timeout_seconds: int) -> OrderReceipt:
        return self.wait_super_order(order_id, timeout_seconds)

    def positions(self) -> list[dict]:
        return [
            {"sym": sym, "tradingSymbol": sym, "netQty": int(position["qty"]), "averagePrice": position["avg"]}
            for sym, position in self.pos.items() if position["qty"]
        ]

    def order_flow(self, sym: str):
        return self.flow.assessment(sym)

    def order_flow_status(self, sym: str) -> dict[str, object]:
        return self.flow.status(sym)

    def quote_snapshot(self, syms: list[str]) -> dict[str, dict]:
        """Simulated day-cumulative volume so paper candles use share volume,
        matching historical candle semantics (C1)."""
        with self.lock:
            today = now().date()
            if self._vol_day != today:
                self._vol_day = today
                self._day_volume.clear()
            out: dict[str, dict] = {}
            for sym in syms:
                self._day_volume[sym] = self._day_volume.get(sym, 0) + self.random.randint(20, 200)
                out[sym] = {"last_price": self.ltp(sym), "volume": self._day_volume[sym]}
            return out

    def place_protective_stop(self, sym: str, qty: int, trigger: float, tag: str) -> OrderReceipt:
        _valid_price(trigger)
        with self.lock:
            order_id = self._new_id("PSL")
            receipt = OrderReceipt(order_id, "TRIGGER_PENDING", int(qty), 0.0)
            self.orders[order_id] = receipt
            self.db.ex(
                "INSERT OR REPLACE INTO orders VALUES(?,?,?,?,?,?,?,?,?,?)",
                (order_id, sym, "SELL", int(qty), 0.0, "SL", receipt.status, tag[:80], iso(), self.name),
            )
            return receipt


class DhanDepthFeed:
    """Small supervised client for Dhan's documented 20-level depth feed."""

    ENDPOINT = "wss://depth-api-feed.dhan.co/twentydepth"

    def __init__(self, token: str, client_id: str, security_to_symbol: dict[str, str], flow: OrderFlowEngine):
        self.token = token
        self.client_id = client_id
        self.security_to_symbol = security_to_symbol
        self.flow = flow
        self._stop = threading.Event()
        self._connected = threading.Event()
        self._thread: threading.Thread | None = None
        self._socket = None
        self._last_error = ""
        # A valid book needs contemporaneous bid and ask ladders.  Storing the
        # receive time beside each side prevents a delayed packet from being
        # paired with a newly-arrived opposite side.
        self._partial: dict[str, dict[str, tuple[tuple[BookLevel, ...], float]]] = {}
        self._lock = threading.RLock()
        self._had_session = False

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        try:
            import websocket  # type: ignore
        except ImportError as exc:
            raise MarketDataError("websocket-client is required for Dhan 20-level order flow") from exc
        self._websocket = websocket
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="dhan-depth", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            socket = self._socket
        if socket is not None:
            try:
                socket.close()
            except Exception:
                pass
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5)

    def status(self) -> dict[str, object]:
        state = "CONNECTED" if self._connected.is_set() else "CONNECTING"
        return {"state": state, "connected": self._connected.is_set(), "last_error": self._last_error}

    def _run(self) -> None:
        delay = 1.0
        while not self._stop.is_set():
            query = urlencode({"token": self.token, "clientId": self.client_id, "authType": 2})
            app = self._websocket.WebSocketApp(
                f"{self.ENDPOINT}?{query}",
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
            )
            with self._lock:
                self._socket = app
            try:
                app.run_forever(ping_interval=10, ping_timeout=10)
            except Exception as exc:
                self._last_error = exc.__class__.__name__
                LOG.warning("Dhan depth feed failed: %s", exc.__class__.__name__)
            finally:
                self._connected.clear()
                with self._lock:
                    self._socket = None
            # A session that actually connected resets the backoff; only
            # consecutive never-connected failures keep doubling it (A4).
            delay = 1.0 if self._had_session else min(delay * 2.0, 30.0)
            self._had_session = False
            if self._stop.wait(delay):
                break

    def _on_open(self, socket) -> None:
        subscription = {
            "RequestCode": 23,
            "InstrumentCount": len(self.security_to_symbol),
            "InstrumentList": [
                {"ExchangeSegment": "NSE_EQ", "SecurityId": security_id}
                for security_id in self.security_to_symbol
            ],
        }
        socket.send(json.dumps(subscription, separators=(",", ":")))
        self._connected.set()
        self._had_session = True
        self._last_error = ""
        LOG.info("Dhan 20-level order-flow feed connected for %s instruments", len(self.security_to_symbol))

    def _on_message(self, _socket, message) -> None:
        if not isinstance(message, (bytes, bytearray, memoryview)):
            return
        try:
            packets = DhanDepthParser.parse(message)
        except DepthParseError as exc:
            self._last_error = "DEPTH_PARSE_ERROR"
            LOG.warning("Rejected malformed Dhan depth packet: %s", str(exc)[:120])
            return
        for packet in packets:
            symbol = self.security_to_symbol.get(packet.security_id)
            if not symbol:
                continue
            # The documented message sequence is explicitly informational and
            # must be ignored. Freshness is based on local receipt times, and
            # bid/ask ladders are admitted only when they are contemporaneous.
            sides = self._partial.setdefault(symbol, {})
            sides[packet.side] = (packet.levels, time.monotonic())
            if "BID" in sides and "ASK" in sides:
                bids, bid_time = sides["BID"]
                asks, ask_time = sides["ASK"]
                max_pair_age = min(float(self.flow.rules["max_staleness_seconds"]), 2.0)
                if abs(bid_time - ask_time) <= max_pair_age:
                    self.flow.ingest(symbol, bids, asks, "DHAN_DEPTH20")

    def _on_error(self, _socket, error) -> None:
        self._last_error = error.__class__.__name__

    def _on_close(self, _socket, _status_code, _message) -> None:
        self._connected.clear()


class DhanBroker(BrokerBase):
    """Fail-closed Dhan v2 client for a dedicated intraday trading account."""

    name = "dhan"
    BASE = "https://api.dhan.co/v2"
    AUTH_TOKEN_URL = "https://auth.dhan.co/app/generateAccessToken"

    def __init__(self, cfg, db):
        super().__init__(cfg, db)
        self.client_id = os.getenv("DHAN_CLIENT_ID", "").strip()
        self.access_token = os.getenv("DHAN_TOKEN", os.getenv("DHAN_ACCESS_TOKEN", "")).strip()
        self.pin = os.getenv("DHAN_PIN", "").strip()
        self.totp_secret = os.getenv("DHAN_TOTP_SECRET", "").strip()
        self.security_map = {str(k).upper(): str(v) for k, v in cfg.get("security_map", {}).items()}
        self.reverse_security_map = {value: key for key, value in self.security_map.items()}
        self.session = requests.Session()
        self.headers: dict[str, str] = {"Accept": "application/json", "Content-Type": "application/json"}
        self.flow = OrderFlowEngine(cfg, db)
        self.depth_feed: DhanDepthFeed | None = None
        # Per-call connect/read budget.  A silent venue must cost a bounded
        # stall (read timeout) and then surface as a BrokerError, never hang
        # the synchronous tick loop forever.  Config-overridable so operators
        # (and offline tests) can tune it; defaults match the historical
        # hard-coded (3.05s connect, 10s read) behaviour.
        execution_cfg = cfg.get("execution", {})
        if not isinstance(execution_cfg, dict):
            execution_cfg = {}
        self._timeout = (
            float(execution_cfg.get("broker_connect_timeout_seconds", 3.05)),
            float(execution_cfg.get("broker_read_timeout_seconds", 10.0)),
        )

    def _set_token(self, token: str) -> None:
        if not token or token.upper().startswith("DUMMY"):
            raise AuthenticationError("A real Dhan access token is required for live trading")
        self.access_token = token
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "access-token": token,
            "client-id": self.client_id,
        }
        self.token = token

    def _request(self, method: str, path: str, *, body: dict | None = None, params: dict | None = None) -> Any:
        if not path.startswith("/") or "://" in path:
            raise SecurityError("Only fixed relative Dhan API paths are permitted")
        guard_endpoint(path)
        try:
            response = self.session.request(
                method, self.BASE + path, headers=self.headers, json=body, params=params, timeout=self._timeout
            )
        except requests.RequestException as exc:
            raise BrokerError(f"Dhan network request failed: {exc.__class__.__name__}") from exc
        if not response.ok:
            try:
                detail = response.json()
                message = detail.get("errorMessage") or detail.get("message") or response.reason
            except ValueError:
                message = response.reason
            if response.status_code == 429 or "DH-904" in str(message).upper():
                try:
                    retry_after = max(0.0, float(response.headers.get("Retry-After", 0.0)))
                except (TypeError, ValueError):
                    retry_after = 0.0
                # A throttled mutation has uncertain broker state and follows
                # the established OrderError fail-closed path.  Quote and
                # read requests are safe to back off and retry without
                # flattening already-protected positions.
                if method.upper() in {"POST", "PUT", "DELETE"} and not path.startswith("/marketfeed"):
                    raise OrderError(f"Dhan order API rate-limited on {path}; outcome requires reconciliation")
                raise RateLimitError(
                    f"Dhan API rate-limited on {path}",
                    retry_after_seconds=retry_after or None,
                )
            raise BrokerError(f"Dhan API {response.status_code}: {message}")
        try:
            return response.json()
        except ValueError as exc:
            raise BrokerError("Dhan API returned a non-JSON response") from exc

    def login(self) -> bool:
        if not self.client_id:
            raise AuthenticationError("DHAN_CLIENT_ID is not set")
        token = self.access_token
        if not token and self.pin and self.totp_secret:
            try:
                import pyotp
            except ImportError as exc:
                raise AuthenticationError("pyotp is required for automatic Dhan token renewal") from exc
            try:
                response = self.session.post(
                    self.AUTH_TOKEN_URL,
                    params={"dhanClientId": self.client_id, "pin": self.pin, "totp": pyotp.TOTP(self.totp_secret).now()},
                    timeout=self._timeout,
                )
                response.raise_for_status()
                token = response.json().get("accessToken", "")
            except (requests.RequestException, ValueError) as exc:
                raise AuthenticationError("Dhan automated token generation failed") from exc
        self._set_token(token)
        # A read-only authenticated endpoint confirms that the session is usable.
        self._request("GET", "/fundlimit")
        return True

    def _security_id(self, sym: str) -> str:
        try:
            return self.security_map[sym.upper()]
        except KeyError as exc:
            raise OrderError(f"No Dhan securityId is configured for {sym}") from exc

    def ltp(self, sym: str) -> float:
        return self.ltps([sym])[sym]

    def ltps(self, syms: list[str]) -> dict[str, float]:
        if not syms:
            return {}
        security_ids = {sym: self._security_id(sym) for sym in syms}
        response = self._request("POST", "/marketfeed/ltp", body={"NSE_EQ": [int(value) for value in security_ids.values()]})
        result: dict[str, float] = {}
        try:
            quote_data = response["data"]["NSE_EQ"]
            for sym, security_id in security_ids.items():
                result[sym] = _valid_price(quote_data[security_id]["last_price"])
        except (KeyError, TypeError, ValueError) as exc:
            raise MarketDataError("Dhan LTP response is incomplete or invalid") from exc
        return result

    def hist(self, sym: str, tf_min: int = 1, days: int = 5):
        if tf_min not in {1, 5, 15, 25, 60}:
            raise MarketDataError("Dhan supports 1, 5, 15, 25, or 60-minute historical candles")
        security_id = self._security_id(sym)
        # Long windows are fetched in bounded chunks: a single 95-day 1-minute
        # request can exceed the intraday endpoint's range and would fail the
        # whole boot (A5).
        chunk_days = max(5, int(self.cfg.get("execution", {}).get("history_chunk_days", 25)))
        end = now()
        cursor_start = end - timedelta(days=days)
        rows: list[tuple] = []
        while cursor_start < end:
            cursor_end = min(cursor_start + timedelta(days=chunk_days), end)
            response = self._request(
                "POST", "/charts/intraday",
                body={
                    "securityId": security_id, "exchangeSegment": "NSE_EQ", "instrument": "EQUITY",
                    "interval": str(tf_min), "oi": False,
                    "fromDate": cursor_start.strftime("%Y-%m-%d %H:%M:%S"),
                    "toDate": cursor_end.strftime("%Y-%m-%d %H:%M:%S"),
                },
            )
            try:
                rows.extend(zip(response["timestamp"], response["open"], response["high"], response["low"], response["close"], response["volume"], strict=True))
            except (KeyError, TypeError, ValueError) as exc:
                raise MarketDataError(f"Malformed historical response for {sym}") from exc
            cursor_start = cursor_end
        return rows

    def _correlation_id(self, prefix: str) -> str:
        return f"{prefix}-{uuid.uuid4().hex[:20]}"

    @staticmethod
    def _receipt(raw: dict, fallback_order_id: str | None = None) -> OrderReceipt:
        order_id = str(raw.get("orderId") or fallback_order_id or "")
        if not order_id:
            raise OrderError("Dhan response omitted orderId")
        status = str(raw.get("orderStatus", "PENDING")).upper()
        return OrderReceipt(
            order_id=order_id,
            status=status,
            filled_qty=int(raw.get("filledQty") or 0),
            average_price=float(raw.get("averageTradedPrice") or 0.0),
        )

    def place_super_order(self, sym: str, side: str, qty: int, target: float, stop: float, tag: str) -> OrderReceipt:
        if side not in {"BUY", "SELL"} or qty <= 0:
            raise OrderError("Invalid Super Order side or quantity")
        target, stop = _valid_price(target), _valid_price(stop)
        entry_quote = self.ltp(sym)
        if (side == "BUY" and not stop < entry_quote < target) or (side == "SELL" and not target < entry_quote < stop):
            raise OrderError("Super Order prices must bracket the current market price")
        body = {
            "dhanClientId": self.client_id, "correlationId": self._correlation_id("ox"),
            "transactionType": side, "exchangeSegment": "NSE_EQ", "productType": "INTRADAY",
            # Dhan Super Orders require a positive price even for MARKET entries.
            "orderType": "MARKET", "securityId": self._security_id(sym), "quantity": qty, "price": round(entry_quote, 2),
            "targetPrice": round(target, 2), "stopLossPrice": round(stop, 2),
            "trailingJump": float(self.cfg["execution"]["trailing_jump"]),
        }
        raw = self._request("POST", "/super/orders", body=body)
        receipt = self._receipt(raw)
        self.db.ex(
            "INSERT OR REPLACE INTO orders VALUES(?,?,?,?,?,?,?,?,?,?)",
            (receipt.order_id, sym, side, qty, entry_quote, "SUPER", receipt.status, tag, iso(), self.name),
        )
        if receipt.status in {"REJECTED", "CANCELLED", "EXPIRED"}:
            raise OrderError(f"Dhan rejected Super Order {receipt.order_id}: {receipt.status}")
        return receipt

    def _super_order_by_id(self, order_id: str) -> OrderReceipt:
        rows = self._request("GET", "/super/orders")
        if not isinstance(rows, list):
            raise BrokerError("Malformed Dhan Super Order book")
        for row in rows:
            if str(row.get("orderId")) == str(order_id):
                return self._receipt(row, order_id)
        raise OrderError(f"Dhan Super Order {order_id} is absent from the order book")

    def wait_super_order(self, order_id: str, timeout_seconds: int) -> OrderReceipt:
        deadline = time.monotonic() + timeout_seconds
        last: OrderReceipt | None = None
        while time.monotonic() < deadline:
            last = self._super_order_by_id(order_id)
            if last.status in {"TRADED", "PART_TRADED", "REJECTED", "CANCELLED", "EXPIRED", "CLOSED"}:
                return last
            time.sleep(1.0)
        raise OrderError(f"Timed out awaiting confirmation of Super Order {order_id}; broker state is uncertain")

    def cancel_super_order(self, order_id: str) -> None:
        # Cancelling ENTRY_LEG cancels the whole Super Order, including attached exits.
        self._request("DELETE", f"/super/orders/{order_id}/ENTRY_LEG")

    def modify_super_target(self, order_id: str, target: float) -> None:
        self._request(
            "PUT", f"/super/orders/{order_id}",
            body={"dhanClientId": self.client_id, "orderId": str(order_id), "legName": "TARGET_LEG", "targetPrice": round(_valid_price(target), 2)},
        )

    def exit_position(self, sym: str, side: str, qty: int, tag: str) -> OrderReceipt:
        if side not in {"BUY", "SELL"} or qty <= 0:
            raise OrderError("Invalid exit side or quantity")
        body = {
            "dhanClientId": self.client_id, "correlationId": self._correlation_id("oxexit"),
            "transactionType": side, "exchangeSegment": "NSE_EQ", "productType": "INTRADAY",
            "orderType": "MARKET", "validity": "DAY", "securityId": self._security_id(sym),
            "quantity": qty, "disclosedQuantity": 0, "price": 0.0, "triggerPrice": 0.0, "afterMarketOrder": False,
        }
        raw = self._request("POST", "/orders", body=body)
        receipt = self._receipt(raw)
        self.db.ex(
            "INSERT OR REPLACE INTO orders VALUES(?,?,?,?,?,?,?,?,?,?)",
            (receipt.order_id, sym, side, qty, 0.0, "MARKET", receipt.status, tag, iso(), self.name),
        )
        if receipt.status in {"REJECTED", "CANCELLED", "EXPIRED"}:
            raise OrderError(f"Dhan rejected exit {receipt.order_id}: {receipt.status}")
        return receipt

    def _order_by_id(self, order_id: str) -> OrderReceipt:
        return self._receipt(self._request("GET", f"/orders/{order_id}"), order_id)

    def wait_order(self, order_id: str, timeout_seconds: int) -> OrderReceipt:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            receipt = self._order_by_id(order_id)
            if receipt.status in {"TRADED", "PART_TRADED", "REJECTED", "CANCELLED", "EXPIRED"}:
                return receipt
            time.sleep(1.0)
        raise OrderError(f"Timed out awaiting confirmation of exit order {order_id}; broker state is uncertain")

    def positions(self) -> list[dict]:
        rows = self._request("GET", "/positions")
        if not isinstance(rows, list):
            raise BrokerError("Malformed Dhan positions response")
        result = []
        for row in rows:
            security_id = str(row.get("securityId", ""))
            symbol = self.reverse_security_map.get(security_id, str(row.get("tradingSymbol", "")).upper())
            result.append({
                "sym": symbol,
                "tradingSymbol": symbol,
                "netQty": int(row.get("netQty", 0)),
                "averagePrice": float(row.get("averagePrice") or 0.0),
                "productType": row.get("productType", "INTRADAY"),
            })
        return result

    def quote_snapshot(self, syms: list[str]) -> dict[str, dict]:
        """Day-cumulative volume + LTP via the batched Market Quote endpoint.
        Used to replace tick-count candle volume with true share volume (C1)."""
        if not syms:
            return {}
        ids = {sym: self._security_id(sym) for sym in syms}
        response = self._request("POST", "/marketfeed/quote", body={"NSE_EQ": [int(value) for value in ids.values()]})
        out: dict[str, dict] = {}
        try:
            data = response["data"]["NSE_EQ"]
            for sym, security_id in ids.items():
                row = data[security_id]
                out[sym] = {"last_price": _valid_price(row["last_price"]), "volume": int(row.get("volume") or 0)}
        except (KeyError, TypeError, ValueError) as exc:
            raise MarketDataError("Dhan quote snapshot is incomplete or invalid") from exc
        return out

    def place_protective_stop(self, sym: str, qty: int, trigger: float, tag: str) -> OrderReceipt:
        """Plain STOP_LOSS sell for a residual long after a partial exit (A2)."""
        body = {
            "dhanClientId": self.client_id, "correlationId": self._correlation_id("oxsl"),
            "transactionType": "SELL", "exchangeSegment": "NSE_EQ", "productType": "INTRADAY",
            "orderType": "STOP_LOSS", "validity": "DAY", "securityId": self._security_id(sym),
            "quantity": int(qty), "disclosedQuantity": 0, "price": 0.0,
            "triggerPrice": round(_valid_price(trigger), 2), "afterMarketOrder": False,
        }
        raw = self._request("POST", "/orders", body=body)
        receipt = self._receipt(raw)
        self.db.ex(
            "INSERT OR REPLACE INTO orders VALUES(?,?,?,?,?,?,?,?,?,?)",
            (receipt.order_id, sym, "SELL", int(qty), 0.0, "SL", receipt.status, tag[:80], iso(), self.name),
        )
        if receipt.status in {"REJECTED", "CANCELLED", "EXPIRED"}:
            raise OrderError(f"Dhan rejected protective stop {receipt.order_id}: {receipt.status}")
        return receipt

    def whitelisted_ips(self) -> set[str] | None:
        response = self._request("GET", "/ip/getIP")
        if not isinstance(response, (list, dict)):
            raise BrokerError("Malformed Dhan static-IP response")
        found: set[str] = set()

        def collect(value) -> None:
            if isinstance(value, dict):
                for key, item in value.items():
                    if "ip" in str(key).lower() and isinstance(item, str):
                        found.add(item.strip())
                    collect(item)
            elif isinstance(value, list):
                for item in value:
                    collect(item)

        collect(response)
        return found - {""}

    def start_orderflow(self) -> None:
        if not self.cfg["order_flow"]["enabled"]:
            return
        if not self.authenticated():
            raise MarketDataError("Dhan order-flow feed requires an authenticated session")
        if self.depth_feed is None:
            self.depth_feed = DhanDepthFeed(self.access_token, self.client_id, self.reverse_security_map, self.flow)
        self.depth_feed.start()

    def stop_orderflow(self) -> None:
        if self.depth_feed is not None:
            self.depth_feed.stop()

    def order_flow(self, sym: str):
        return self.flow.assessment(sym)

    def order_flow_status(self, sym: str) -> dict[str, object]:
        status = self.flow.status(sym)
        if self.depth_feed is not None:
            status["feed"] = self.depth_feed.status()
        return status



class GrowwBroker(BrokerBase):
    name = "groww"
    def __init__(self, cfg, db):
        super().__init__(cfg, db)
        self.api_key=os.getenv(cfg.get("platforms",{}).get("groww_api_key_env","GROWW_API_KEY"),"").strip()
        self.api_secret=os.getenv(cfg.get("platforms",{}).get("groww_api_secret_env","GROWW_API_SECRET"),"").strip()
    def login(self):
        if not self.api_key: raise AuthenticationError("GROWW_API_KEY not set")
        self.token="groww-session"; return True
    def ltp(self,sym): raise MarketDataError("Wire Groww LTP endpoint")
    def hist(self,sym,tf_min=1,days=5): raise MarketDataError("Wire Groww hist endpoint")

class ChoiceBroker(BrokerBase):
    """Choice India (Finvasia) live adapter over the Shoonya/Noren gateway.

    Transport and payload contract mirrored from the official Shoonya wrapper
    (github.com/Shoonya-Dev/ShoonyaApi-py: api_helper.py, example_orders.py,
    example_market.py) and its NorenRestApiPy base (NorenApi.py 0.0.22/0.0.30):
    every request is a form POST of 'jData=<json>&jKey=<token>' to
    https://api.shoonya.com/NorenWClientTP/<Route>; login (QuickAuth) SHA-256
    hashes the password and the 'uid|apikey' app key and exchanges them for a
    session token.  Unlike Dhan, Shoonya has no IP-whitelist concept, so
    :meth:`whitelisted_ips` reports None and compliance skips that check.

    Credentials come from the environment (CHOICE_USER_ID / CHOICE_PASSWORD /
    CHOICE_TOTP / CHOICE_VENDOR_CODE / CHOICE_API_KEY / CHOICE_IMEI), each
    overridable through cfg['platforms']['choice_*_env'].  security_map
    entries must be 'EXCH|TOKEN|TRADINGSYMBOL' (e.g. 'NSE|2885|RELIANCE-EQ');
    the token drives quotes/history and the tradingsymbol drives orders.

    Order-book / position-book / TPSeries field names follow the documented
    Noren protocol used by the official wrapper.  Every parse fails closed
    with a BrokerError naming the symbol rather than fabricating a fill:
    fill quantities come only from the venue's fillshares/avgprc fields.
    """

    name = "choice"
    BASE = "https://api.shoonya.com/NorenWClientTP/"
    _TIME_INTERVALS = {1, 3, 5, 10, 15, 30, 60, 120, 240}

    def __init__(self, cfg, db):
        super().__init__(cfg, db)
        platforms = cfg.get("platforms", {}) if isinstance(cfg, dict) else {}
        self.user_id = os.getenv(platforms.get("choice_user_id_env", "CHOICE_USER_ID"), "").strip()
        self.password = os.getenv(platforms.get("choice_password_env", "CHOICE_PASSWORD"), "").strip()
        self.totp = os.getenv(platforms.get("choice_totp_env", "CHOICE_TOTP"), "").strip()
        self.vendor_code = os.getenv(platforms.get("choice_vendor_code_env", "CHOICE_VENDOR_CODE"), "").strip()
        self.api_secret = os.getenv(platforms.get("choice_api_key_env", "CHOICE_API_KEY"), "").strip()
        self.imei = os.getenv(platforms.get("choice_imei_env", "CHOICE_IMEI"), "ox-alpha-ultimate").strip()
        self.security_map = {str(key).upper(): str(value) for key, value in cfg.get("security_map", {}).items()}
        self._token_to_symbol = {
            parts[1]: sym for sym, entry in self.security_map.items()
            if len((parts := str(entry).split("|"))) == 3 and parts[1].isdigit()
        }
        self.session = requests.Session()
        self._username = None
        self._account_id = None
        execution_cfg = cfg.get("execution", {}) if isinstance(cfg, dict) else {}
        if not isinstance(execution_cfg, dict):
            execution_cfg = {}
        self._timeout = (
            float(execution_cfg.get("broker_connect_timeout_seconds", 3.05)),
            float(execution_cfg.get("broker_read_timeout_seconds", 10.0)),
        )

    def _request(self, path: str, values: dict, *, authenticated: bool = True) -> Any:
        if not path.startswith("/") or "://" in path:
            raise SecurityError("Only fixed relative Choice API paths are permitted")
        guard_endpoint(path)
        if authenticated and not self.token:
            raise AuthenticationError("Choice session is not authenticated; login first")
        payload = "jData=" + json.dumps(values)
        if authenticated:
            payload += f"&jKey={self.token}"
        try:
            response = self.session.post(
                self.BASE + path[1:], data=payload,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=self._timeout,
            )
        except requests.RequestException as exc:
            raise BrokerError(f"Choice network request failed: {exc.__class__.__name__}") from exc
        if not response.ok:
            raise BrokerError(f"Choice API {response.status_code}: {response.reason}")
        try:
            data = response.json()
        except ValueError as exc:
            raise BrokerError("Choice API returned a non-JSON response") from exc
        if isinstance(data, dict) and data.get("stat") not in (None, "Ok"):
            message = str(data.get("emsg") or data.get("stat"))
            if path == "/QuickAuth":
                raise AuthenticationError(f"Choice login rejected: {message}")
            raise OrderError(f"Choice rejected {path}: {message}")
        return data

    def login(self) -> bool:
        missing = [
            name for name, value in (
                ("CHOICE_USER_ID", self.user_id),
                ("CHOICE_PASSWORD", self.password),
                ("CHOICE_TOTP", self.totp),
                ("CHOICE_VENDOR_CODE", self.vendor_code),
                ("CHOICE_API_KEY", self.api_secret),
            )
            if not value
        ]
        if missing:
            raise AuthenticationError(f"Choice login requires: {', '.join(missing)}")
        pwd = hashlib.sha256(self.password.encode("utf-8")).hexdigest()
        app_key = hashlib.sha256(f"{self.user_id}|{self.api_secret}".encode("utf-8")).hexdigest()
        raw = self._request("/QuickAuth", {
            "source": "API", "apkversion": "1.0.0",
            "uid": self.user_id, "pwd": pwd, "factor2": self.totp,
            "vc": self.vendor_code, "appkey": app_key, "imei": self.imei,
        }, authenticated=False)
        token = str(raw.get("susertoken") or "")
        if not token or token.upper().startswith("DUMMY"):
            raise AuthenticationError("Choice login returned no session token")
        self.token = token
        self._username = self.user_id
        self._account_id = str(raw.get("actid") or self.user_id)
        return True

    def whitelisted_ips(self) -> set[str] | None:
        # Shoonya authenticates with a session token; there is no IP
        # allowlist concept.  None signals compliance to skip the check.
        return None

    def _resolve(self, sym: str) -> tuple[str, str, str]:
        entry = self.security_map.get(str(sym).upper())
        if not entry:
            raise OrderError(
                f"No Choice security is configured for {sym}; security_map entries must be "
                "'EXCH|TOKEN|TRADINGSYMBOL' (e.g. 'NSE|2885|RELIANCE-EQ')"
            )
        parts = str(entry).split("|")
        if len(parts) != 3 or not parts[0] or not parts[1].isdigit() or not parts[2]:
            raise OrderError(
                f"Malformed Choice security_map entry for {sym}: {entry!r} "
                "(expected 'EXCH|TOKEN|TRADINGSYMBOL')"
            )
        return parts[0].upper(), parts[1], parts[2]

    def ltp(self, sym: str) -> float:
        return self.ltps([sym])[sym]

    def ltps(self, syms: list[str]) -> dict[str, float]:
        result: dict[str, float] = {}
        for sym in syms:
            exchange, token, _ = self._resolve(sym)
            raw = self._request("/GetQuotes", {"uid": self._username, "exch": exchange, "token": token})
            try:
                result[sym] = _valid_price(raw["lp"])
            except (KeyError, TypeError, ValueError) as exc:
                raise MarketDataError(f"Choice quote for {sym} is incomplete or invalid") from exc
        return result

    def hist(self, sym: str, tf_min: int = 1, days: int = 5):
        if tf_min not in self._TIME_INTERVALS:
            raise MarketDataError("Choice supports 1, 3, 5, 10, 15, 30, 60, 120, or 240-minute candles")
        exchange, token, _ = self._resolve(sym)
        end = now()
        start = end - timedelta(days=days)
        raw = self._request("/TPSeries", {
            "ordersource": "API", "uid": self._username, "exch": exchange, "token": token,
            "st": str(int(start.timestamp())), "et": str(int(end.timestamp())), "intrv": str(tf_min),
        })
        if not isinstance(raw, list):
            raise MarketDataError(f"Choice history for {sym} is incomplete or invalid")
        rows: list[tuple] = []
        for bar in raw:
            try:
                rows.append((
                    int(bar["time"]),
                    _valid_price(bar["into"]), _valid_price(bar["inth"]),
                    _valid_price(bar["intl"]), _valid_price(bar["intc"]),
                    int(bar.get("intv") or 0),
                ))
            except (KeyError, TypeError, ValueError) as exc:
                raise MarketDataError(f"Malformed Choice history bar for {sym}") from exc
        return rows

    @staticmethod
    def _order_receipt(row: dict, fallback_id: str | None = None) -> OrderReceipt:
        order_id = str(row.get("norenordno") or fallback_id or "")
        if not order_id:
            raise OrderError("Choice order response omitted norenordno")
        status = str(row.get("status", "PENDING")).upper()
        filled = int(row.get("fillshares") or 0)
        if status == "OPEN":
            status = "PART_TRADED" if filled > 0 else "PENDING"
        elif status == "COMPLETE":
            status = "TRADED"
        elif status == "CANCELED":
            status = "CANCELLED"
        return OrderReceipt(order_id, status, filled, float(row.get("avgprc") or 0.0))

    def _place(self, sym: str, side: str, qty: int, price_type: str, tag: str, trigger: float | None = None) -> OrderReceipt:
        if side not in {"BUY", "SELL"} or qty <= 0:
            raise OrderError("Invalid Choice order side or quantity")
        exchange, _token, tsym = self._resolve(sym)
        values = {
            "ordersource": "API", "uid": self._username, "actid": self._account_id,
            "trantype": side, "prd": "I", "exch": exchange, "tsym": tsym,
            "qty": str(int(qty)), "dscqty": "0", "prctyp": price_type,
            "prc": "0", "trgprc": str(round(_valid_price(trigger), 2)) if trigger else "0",
            "ret": "DAY", "remarks": (tag or "ox")[:30], "amo": "NO",
        }
        raw = self._request("/PlaceOrder", values)
        order_id = str(raw.get("norenordno") or "")
        if not order_id:
            raise OrderError("Choice PlaceOrder omitted norenordno")
        return OrderReceipt(order_id, "PENDING", 0, 0.0)

    def place_super_order(self, sym: str, side: str, qty: int, target: float, stop: float, tag: str) -> OrderReceipt:
        target, stop = _valid_price(target), _valid_price(stop)
        entry_quote = self.ltp(sym)
        if (side == "BUY" and not stop < entry_quote < target) or (side == "SELL" and not target < entry_quote < stop):
            raise OrderError("Super Order prices must bracket the current market price")
        receipt = self._place(sym, side, qty, "MKT", tag)
        self.db.ex(
            "INSERT OR REPLACE INTO orders VALUES(?,?,?,?,?,?,?,?,?,?)",
            (receipt.order_id, sym, side, qty, entry_quote, "SUPER", receipt.status, tag, iso(), self.name),
        )
        return receipt

    def _book_receipt(self, order_id: str) -> OrderReceipt | None:
        rows = self._request("/OrderBook", {"ordersource": "API", "uid": self._username})
        if not isinstance(rows, list):
            raise BrokerError("Malformed Choice order book")
        for row in rows:
            if str(row.get("norenordno")) == str(order_id):
                return self._order_receipt(row, order_id)
        return None

    def wait_super_order(self, order_id: str, timeout_seconds: int) -> OrderReceipt:
        deadline = time.monotonic() + timeout_seconds
        last: OrderReceipt | None = None
        while time.monotonic() < deadline:
            last = self._book_receipt(order_id)
            if last is None:
                raise OrderError(f"Choice order {order_id} is absent from the order book")
            if last.status in {"TRADED", "PART_TRADED", "REJECTED", "CANCELLED", "EXPIRED", "CLOSED"}:
                return last
            time.sleep(1.0)
        raise OrderError(f"Timed out awaiting confirmation of order {order_id}; broker state is uncertain")

    def wait_order(self, order_id: str, timeout_seconds: int) -> OrderReceipt:
        return self.wait_super_order(order_id, timeout_seconds)

    def cancel_super_order(self, order_id: str) -> None:
        self._request("/CancelOrder", {"ordersource": "API", "uid": self._username, "norenordno": str(order_id)})

    def modify_super_target(self, order_id: str, target: float) -> None:
        # Shoonya has no Dhan-style bracket legs; the OMS manages stops and
        # targets itself, so a leg-level modify must fail closed rather than
        # silently pretend a protective target moved.  The OMS catches
        # BrokerError here and falls back to its enforced-target logic (A3).
        raise OrderError(
            f"Choice does not support Dhan-style Super Order target modification ({order_id}); "
            "targets are enforced by the OMS via protective stops"
        )

    def exit_position(self, sym: str, side: str, qty: int, tag: str) -> OrderReceipt:
        receipt = self._place(sym, side, qty, "MKT", tag)
        self.db.ex(
            "INSERT OR REPLACE INTO orders VALUES(?,?,?,?,?,?,?,?,?,?)",
            (receipt.order_id, sym, side, qty, 0.0, "MARKET", receipt.status, tag, iso(), self.name),
        )
        return receipt

    def place_protective_stop(self, sym: str, qty: int, trigger: float, tag: str) -> OrderReceipt:
        receipt = self._place(sym, "SELL", qty, "SL-MKT", tag, trigger=trigger)
        self.db.ex(
            "INSERT OR REPLACE INTO orders VALUES(?,?,?,?,?,?,?,?,?,?)",
            (receipt.order_id, sym, "SELL", int(qty), 0.0, "SL", receipt.status, tag[:80], iso(), self.name),
        )
        return receipt

    def positions(self) -> list[dict]:
        rows = self._request("/PositionBook", {"uid": self._username, "actid": self._account_id})
        if not isinstance(rows, list):
            raise BrokerError("Malformed Choice positions response")
        result = []
        for row in rows:
            net = int(row.get("netqty") or 0)
            if net == 0:
                continue
            tsym = str(row.get("tsym") or "").upper()
            symbol = self._token_to_symbol.get(str(row.get("token") or ""), tsym or "UNKNOWN")
            result.append({
                "sym": symbol,
                "tradingSymbol": tsym,
                "netQty": net,
                "averagePrice": float(row.get("avgprc") or 0.0),
                "productType": row.get("prd", "I"),
            })
        return result


class TradingViewBroker(BrokerBase):
    name = "tradingview"
    def __init__(self, cfg, db):
        super().__init__(cfg, db)
        self.secret=os.getenv(cfg.get("platforms",{}).get("tradingview_webhook_secret_env","TV_WEBHOOK_SECRET"),"").strip()
        self._signals={}
    def login(self):
        self.token="tv-bridge"; return True
    def ingest_webhook(self, payload: dict, signature: str):
        import hmac, hashlib
        if not self.secret: raise AuthenticationError("TV_WEBHOOK_SECRET not set")
        expected=hmac.new(self.secret.encode(), json.dumps(payload,sort_keys=True).encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected): raise SecurityError("Invalid TradingView signature")
        self._signals[payload["symbol"]]=payload

class BinanceBroker(BrokerBase):
    name="binance"
    def __init__(self,cfg,db):
        super().__init__(cfg,db)
        self.key=os.getenv(cfg.get("platforms",{}).get("binance_api_key_env","BINANCE_API_KEY"),"").strip()
        self.secret=os.getenv(cfg.get("platforms",{}).get("binance_api_secret_env","BINANCE_API_SECRET"),"").strip()
    def login(self):
        if not self.key: raise AuthenticationError("BINANCE_API_KEY not set")
        try:
            import ccxt
            self.ccxt=ccxt.binance({"apiKey":self.key,"secret":self.secret,"enableRateLimit":True})
            self.token="binance-session"; return True
        except ImportError as e:
            raise AuthenticationError("ccxt not installed for Binance") from e


def make_broker(cfg, db) -> BrokerBase:
    plat=str(cfg.get("platform","paper")).lower()
    if plat == "groww": return GrowwBroker(cfg,db)
    if plat == "choice": return ChoiceBroker(cfg,db)
    if plat == "tradingview": return TradingViewBroker(cfg,db)
    if plat=="binance": return BinanceBroker(cfg,db)
    if plat in {"crypto_paper","crypto"}:
        from .crypto import CryptoMicroBroker
        return CryptoMicroBroker(cfg,db)
    # Platform selects the adapter; mode only gates the boot-time compliance
    # checks.  Previously "dhan" + paper mode silently fell through to the
    # PaperBroker, so the real Dhan adapter could never be exercised (or
    # supervised on a demo account) without live credentials.
    if plat == "dhan":
        return DhanBroker(cfg, db)
    return PaperBroker(cfg, db)
