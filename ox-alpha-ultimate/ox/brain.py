"""Constrained strategy selection.

Strategies are parameter sets for audited, in-process templates.  The database
never stores or executes Python source code; this prevents a tampered database
or news payload from becoming code execution in the trading process.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import random

import numpy as np
import pandas as pd

from .charges import ChargesCalculator
from .core import LOG, iso
from .features import REG
from .risk import Metrics


def core_signal(df: pd.DataFrame, p: dict) -> dict:
    o, h, l, c, v = [df[k].to_numpy(dtype=float) for k in ("o", "h", "l", "c", "v")]
    ef = REG["ema"](c, p.get("ema_fast", 9))
    es = REG["ema"](c, p.get("ema_slow", 21))
    rsi = REG["rsi"](c, p.get("rsi_len", 14))
    atr = REG["atr"](h, l, c)
    sw, _ = REG["bos_choch"](h, l, c, p.get("k_swing", 3))
    sweep = REG["liquidity_sweep"](h, l, c, p.get("k_swing", 3))
    delta, _ = REG["delta"](o, h, l, c, v)
    ultra_delta = REG["ultra_delta"](delta)
    signal = np.zeros(len(c), dtype=int)
    for index in range(2, len(c)):
        long_signal = (
            ef[index] > es[index]
            and rsi[index] < p.get("rsi_os", 30) + 15
            and (sw[index] >= 0.5 or sweep[index] == -1)
            and ultra_delta[index] > -1.0
        )
        short_signal = (
            ef[index] < es[index]
            and rsi[index] > p.get("rsi_ob", 70) - 15
            and (sw[index] <= -0.5 or sweep[index] == 1)
            and ultra_delta[index] < 1.0
        )
        signal[index] = 1 if long_signal else -1 if short_signal else 0
    return {"signal": signal, "atr": atr}


def scalp_signal(df: pd.DataFrame, p: dict) -> dict:
    o, h, l, c, v = [df[k].to_numpy(dtype=float) for k in ("o", "h", "l", "c", "v")]
    ef = REG["ema"](c, 5)
    es = REG["ema"](c, 13)
    delta, _ = REG["delta"](o, h, l, c, v)
    ultra_delta = REG["ultra_delta"](delta)
    big_trades = REG["big_trades"](v)
    bullish_fvg, bearish_fvg = REG["fvg"](o, h, l, c)
    signal = np.zeros(len(c), dtype=int)
    for index in range(2, len(c)):
        up = ef[index] > es[index] and ultra_delta[index] > 0.5 and (big_trades[index] > 0 or bullish_fvg[index] > 0)
        down = ef[index] < es[index] and ultra_delta[index] < -0.5 and (big_trades[index] > 0 or bearish_fvg[index] > 0)
        signal[index] = 1 if up else -1 if down else 0
    return {"signal": signal}


def _session_vwap(df: pd.DataFrame, h: np.ndarray, l: np.ndarray, c: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Compute an intraday VWAP that restarts each exchange day."""
    dates = pd.to_datetime(df["ts"], unit="s", utc=True, errors="coerce").dt.tz_convert("Asia/Kolkata").dt.date
    typical = (h + l + c) / 3.0
    weighted = pd.Series(typical * v).groupby(dates, sort=False).cumsum().to_numpy(dtype=float)
    volume = pd.Series(v).groupby(dates, sort=False).cumsum().to_numpy(dtype=float)
    return weighted / np.maximum(volume, 1e-9)


def breakout_signal(df: pd.DataFrame, p: dict) -> dict:
    """Long-only trend breakout with volume and volatility regime confirmation.

    This is intentionally transparent rather than a fitted black-box model.
    Its parameters remain constrained to Refiner.GRID and it must pass the
    same walk-forward backtest, cost model, and signal-stability checks as the
    existing templates before it can execute.
    """
    o, h, l, c, v = [df[key].to_numpy(dtype=float) for key in ("o", "h", "l", "c", "v")]
    fast = REG["ema"](c, p["ema_fast"])
    slow = REG["ema"](c, p["ema_slow"])
    rsi = REG["rsi"](c, p["rsi_len"])
    atr = REG["atr"](h, l, c)
    session_vwap = _session_vwap(df, h, l, c, v)
    baseline = pd.Series(v).rolling(p["volume_lookback"], min_periods=max(5, p["volume_lookback"] // 2)).median()
    volume_ratio = (pd.Series(v) / baseline.replace(0, np.nan)).fillna(0.0).to_numpy(dtype=float)
    prior_high = pd.Series(h).rolling(p["breakout_lookback"], min_periods=p["breakout_lookback"]).max().shift(1).to_numpy(dtype=float)
    atr_pct = atr / np.maximum(c, 1e-9) * 100.0
    signal = np.zeros(len(c), dtype=int)
    first = max(p["breakout_lookback"], p["volume_lookback"] // 2, p["ema_slow"])
    for index in range(first, len(c)):
        trend = fast[index] > slow[index] and c[index] > slow[index] and c[index] >= session_vwap[index]
        momentum = 50.0 <= rsi[index] <= p["rsi_ob"]
        liquid_breakout = c[index] > prior_high[index] and volume_ratio[index] >= p["min_volume_ratio"]
        tradable_volatility = p["min_atr_pct"] <= atr_pct[index] <= p["max_atr_pct"]
        if trend and momentum and liquid_breakout and tradable_volatility:
            signal[index] = 1
        elif c[index] < fast[index] or rsi[index] < p["rsi_os"]:
            signal[index] = -1
    return {"signal": signal, "atr": atr}


TEMPLATES = {"core": core_signal, "scalp": scalp_signal, "breakout": breakout_signal}
STRATEGY_SCHEMA_VERSION = 5

# 100x upgrade: register the expanded live-eligible template library so the
# genetic search can mutate and promote any of them. Research-only templates
# (options premium, carry, arb) stay excluded from autonomous entry.
try:
    from .strategies100 import live_templates as _live_templates
    TEMPLATES.update(_live_templates())
except Exception as _exc:  # noqa: BLE001 - expanded library must never block boot
    LOG.warning("strategies100 live template library could not load (%d templates active): %s",
                len(TEMPLATES), _exc.__class__.__name__)


class Backtester:
    def __init__(self, cfg):
        self.cfg = cfg
        self.cost = cfg.get("costs", {})
        self.calculator = ChargesCalculator(self.cost)

    @staticmethod
    def _empty_stats() -> dict:
        return {
            "sharpe": 0.0, "sortino": 0.0, "maxdd": 0.0, "trades": 0,
            "pf": 0.0, "ret": 0.0, "icir": 0.0, "frame_ic": float("nan"),
            "win_rate": 0.0, "cost_drag": 0.0, "trade_details": [],
            "execution": "NEXT_CANDLE_OPEN_LONG_ONLY",
        }

    @staticmethod
    def _validated_frame(df: pd.DataFrame) -> pd.DataFrame:
        required = ("ts", "o", "h", "l", "c", "v")
        if not isinstance(df, pd.DataFrame) or any(column not in df.columns for column in required):
            raise ValueError("Backtest frame must contain ts, o, h, l, c, and v")
        result = df.loc[:, required].copy().sort_values("ts").drop_duplicates(subset=["ts"], keep="last")
        for column in ("o", "h", "l", "c", "v"):
            result[column] = pd.to_numeric(result[column], errors="coerce")
        result = result.replace([np.inf, -np.inf], np.nan).dropna()
        if (result[["o", "h", "l", "c"]] <= 0).any().any() or (result["v"] < 0).any():
            raise ValueError("Backtest frame contains an invalid OHLCV value")
        if (result["h"] < result[["o", "c", "l"]].max(axis=1)).any() or (result["l"] > result[["o", "c", "h"]].min(axis=1)).any():
            raise ValueError("Backtest frame contains inconsistent OHLC values")
        return result.reset_index(drop=True)

    def run(self, df: pd.DataFrame, signal_builder, params: dict) -> tuple[dict, list[float]]:
        frame = self._validated_frame(df)
        if len(frame) < 4:
            return self._empty_stats(), []
        opening = frame["o"].to_numpy(dtype=float)
        high = frame["h"].to_numpy(dtype=float)
        low = frame["l"].to_numpy(dtype=float)
        close = frame["c"].to_numpy(dtype=float)
        result = signal_builder(frame, params)
        signal = np.asarray(result.get("signal"), dtype=int)
        atr = np.asarray(result.get("atr", np.maximum(high - low, close * 0.005)), dtype=float)
        if len(signal) != len(close) or len(atr) != len(close):
            raise ValueError("Strategy returned signal array with invalid length")
        slippage = float(self.cost.get("slippage_pct", 0.03)) / 100.0
        position: dict | None = None
        equity = [1.0]
        bar_equity = [1.0]
        trades: list[float] = []
        trade_details: list[dict] = []
        total_cost_ratio = 0.0
        capital_base = max(float(self.cfg.get("capital", 100000.0)), 1e-9)
        entry_index = -1

        def exit_position(exit_price: float, exit_i: int) -> None:
            nonlocal position, total_cost_ratio
            if position is None:
                return
            sell = max(float(exit_price) * (1.0 - slippage), 0.01)
            charges = self.calculator.compute_charges(position["entry"], sell, position["qty"])
            denominator = max(position["entry"] * position["qty"], 1e-9)
            net_return = float(charges["net_pnl"]) / denominator
            cost_ratio = float(charges["total_charges"]) / denominator
            trades.append(net_return)
            total_cost_ratio += cost_ratio
            equity.append(equity[-1] * (1.0 + net_return))
            trade_details.append({"entry_i": entry_index, "exit_i": exit_i, "ret": net_return})
            position = None

        # Signals use a completed candle. Entries and signal exits occur only at
        # the *next* candle open, so OOS scores cannot benefit from close-price
        # look-ahead. The production engine is long-only, so a negative signal
        # is a long exit rather than a synthetic short.
        for index in range(len(close) - 1):
            next_index = index + 1
            next_open = opening[next_index]

            if position is not None:
                if next_open <= position["stop"] or next_open >= position["target"] or signal[index] < 0:
                    exit_position(next_open, next_index)
                elif low[next_index] <= position["stop"]:
                    exit_position(position["stop"], next_index)
                elif high[next_index] >= position["target"]:
                    exit_position(position["target"], next_index)

            if position is None and signal[index] > 0:
                entry = next_open * (1.0 + slippage)
                atr_distance = max(float(atr[index]), entry * 0.005)
                stop = entry - max(atr_distance * float(params.get("sl_atr", 1.5)), entry * 0.002)
                target = entry + max(atr_distance * float(params.get("tp_atr", 2.0)), entry * 0.004)
                risk_amount = float(self.cfg["capital"]) * float(self.cfg["risk"]["risk_per_trade_pct"]) / 100.0
                quantity = min(int(risk_amount / max(entry - stop, 1e-9)), int(float(self.cfg["risk"]["max_notional_per_trade"]) / entry))
                if quantity > 0 and stop < entry < target:
                    position = {"entry": entry, "stop": stop, "target": target, "qty": quantity}
                    entry_index = next_index

            # Bar-by-bar mark-to-market so maxdd sees intratrade drawdown (C5).
            if position is not None:
                weight = min(position["qty"] * position["entry"] / capital_base, 1.0)
                bar_equity.append(bar_equity[-1] * (1.0 + weight * (close[next_index] / position["entry"] - 1.0)))
            else:
                bar_equity.append(bar_equity[-1])

        if position is not None:
            exit_position(close[-1], len(close) - 1)

        # Per-frame Information Coefficient: correlation between the long
        # signal and next-bar forward returns (Grinold-Kahn style), replacing
        # the old signal-weighted-return sum that was mislabeled ICIR (C3).
        forward = close[1:] / np.maximum(close[:-1], 1e-9) - 1.0
        long_signal = np.maximum(signal[:-1], 0)
        if long_signal.std() > 0 and forward.std() > 0:
            frame_ic = float(np.corrcoef(long_signal, forward)[0, 1])
        else:
            frame_ic = float("nan")

        returns = np.asarray(trades, dtype=float) if trades else np.array([0.0])
        gross_profit = sum(value for value in trades if value > 0)
        gross_loss = -sum(value for value in trades if value < 0)
        return {
            "sharpe": Metrics.sharpe(returns, annualisation=1),
            "sortino": Metrics.sortino(returns, annualisation=1),
            "maxdd": Metrics.maxdd(np.asarray(bar_equity)),
            "trades": len(trades),
            "pf": float(gross_profit / max(gross_loss, 1e-9)),
            "ret": float(equity[-1] - 1.0),
            # ICIR is only meaningful after aggregate() pools several frames'
            # ICs; report NaN (not a misleading 0.0) for a single frame so no
            # consumer can mistake run() output for a real ICIR.
            "icir": float("nan"),
            "frame_ic": frame_ic,
            "win_rate": float(sum(value > 0 for value in trades) / len(trades)) if trades else 0.0,
            "cost_drag": float(total_cost_ratio),
            "execution": "NEXT_CANDLE_OPEN_LONG_ONLY",
            "trade_details": trade_details[:500],
        }, trades

    @classmethod
    def aggregate(cls, results: list[tuple[dict, list[float]]]) -> tuple[dict, list[float]]:
        if not results:
            return cls._empty_stats(), []
        stats = [item[0] for item in results]
        trades = [trade for _, values in results for trade in values]
        ics = [float(item["frame_ic"]) for item in stats if np.isfinite(float(item.get("frame_ic", float("nan"))))]
        icir = float(np.mean(ics) / np.std(ics)) if len(ics) > 1 and float(np.std(ics)) > 0 else 0.0
        returns = np.asarray(trades, dtype=float) if trades else np.array([0.0])
        gross_profit = sum(value for value in trades if value > 0)
        gross_loss = -sum(value for value in trades if value < 0)
        return {
            "sharpe": Metrics.sharpe(returns, annualisation=1),
            "sortino": Metrics.sortino(returns, annualisation=1),
            "maxdd": min(float(item["maxdd"]) for item in stats),
            "trades": len(trades),
            "pf": float(gross_profit / max(gross_loss, 1e-9)),
            "ret": float(np.mean([float(item["ret"]) for item in stats])),
            "icir": icir,
            "win_rate": float(sum(value > 0 for value in trades) / len(trades)) if trades else 0.0,
            "cost_drag": float(sum(float(item["cost_drag"]) for item in stats)),
            "symbols": len(results),
            "execution": "NEXT_CANDLE_OPEN_LONG_ONLY",
        }, trades

    @classmethod
    def frame_consistency(cls, results: list[tuple[dict, list[float]]]) -> dict:
        """Report how many individual walk-forward OOS frames were themselves profitable.

        Pooling every fold/symbol frame into one OOS score (see ``aggregate``)
        can hide a result that is really carried by a single strong fold while
        the rest lose. This looks at each frame's own stats before pooling, so
        that evidence stays visible next to the pooled score rather than
        silently averaged away. A frame with zero trades is excluded from the
        ratio rather than counted as a win or a loss.
        """
        total = len(results)
        traded = [stats for stats, _ in results if int(stats.get("trades", 0)) > 0]
        if not traded:
            return {"ratio": 0.0, "traded": 0, "total": total}
        profitable = sum(1 for stats in traded if float(stats.get("ret", 0.0)) > 0 and float(stats.get("pf", 0.0)) > 1.0)
        return {"ratio": round(profitable / len(traded), 4), "traded": len(traded), "total": total}


class Scorer:
    WEIGHTS = dict(sharpe=0.35, sortino=0.15, pf=0.20, icir=0.15, dd=0.15)

    @staticmethod
    def score(stats: dict, min_trades: int) -> float:
        if stats["trades"] < min_trades or stats["ret"] <= 0 or stats["pf"] <= 1.0:
            return -9.0
        metrics = ("sharpe", "sortino", "pf", "icir", "maxdd", "ret")
        if any(not np.isfinite(float(stats.get(key, float("nan")))) for key in metrics):
            return -9.0
        sharpe = float(np.clip(stats["sharpe"], -5.0, 5.0))
        sortino = float(np.clip(stats["sortino"], -5.0, 5.0))
        icir = float(np.clip(stats["icir"], -5.0, 5.0))
        profit_factor = float(np.clip(stats["pf"], 0.0, 3.0))
        drawdown = float(np.clip(stats["maxdd"], -1.0, 0.0))
        return round(
            Scorer.WEIGHTS["sharpe"] * sharpe
            + Scorer.WEIGHTS["sortino"] * sortino
            + Scorer.WEIGHTS["icir"] * icir
            + Scorer.WEIGHTS["pf"] * profit_factor
            + Scorer.WEIGHTS["dd"] * drawdown * 10.0,
            4,
        )


class FailureAnalyzer:
    """Regime forensics over actual trade entry points, split by outcome.

    Losers get their own shares so Refiner.biased_choice reacts to where the
    strategy loses money, not to the whole window's weather (C4).
    """

    @staticmethod
    def analyze(df: pd.DataFrame, trade_details: list[dict]) -> dict:
        close = df["c"].to_numpy(dtype=float)
        volatility = pd.Series(close).pct_change().rolling(20, min_periods=2).std().fillna(0).to_numpy()
        trend = (pd.Series(close) / pd.Series(close).shift(50).fillna(close[0]) - 1.0).fillna(0).to_numpy()
        report = {
            "n": len(trade_details),
            "high_vol_share": 0.0, "down_trend_share": 0.0,
            "loser_high_vol_share": 0.0, "loser_down_trend_share": 0.0,
        }
        if not len(close) or not trade_details:
            return report
        median_volatility = np.median(volatility[volatility > 0]) if (volatility > 0).any() else 1.0

        def regime(entry_i: object) -> tuple[bool, bool]:
            i = min(max(int(entry_i), 0), len(close) - 1)
            return bool(volatility[i] > median_volatility), bool(trend[i] < 0)

        losers = [d for d in trade_details if float(d.get("ret", 0.0)) <= 0]
        if trade_details:
            flags = [regime(d.get("entry_i", 0)) for d in trade_details]
            report["high_vol_share"] = float(np.mean([f[0] for f in flags]))
            report["down_trend_share"] = float(np.mean([f[1] for f in flags]))
        if losers:
            flags = [regime(d.get("entry_i", 0)) for d in losers]
            report["loser_high_vol_share"] = float(np.mean([f[0] for f in flags]))
            report["loser_down_trend_share"] = float(np.mean([f[1] for f in flags]))
        return report


class Refiner:
    GRID = {
        "ema_fast": [5, 8, 9, 12], "ema_slow": [21, 26, 34, 50], "rsi_len": [7, 14, 21],
        "rsi_ob": [65, 70, 75], "rsi_os": [25, 30, 35], "sl_atr": [1.0, 1.5, 2.0],
        "tp_atr": [1.5, 2.0, 3.0], "k_swing": [3, 5],
        "breakout_lookback": [10, 20, 30], "volume_lookback": [20, 50],
        "min_volume_ratio": [1.0, 1.25, 1.5], "min_atr_pct": [0.10, 0.20],
        "max_atr_pct": [2.0, 3.0],
    }

    @classmethod
    def valid_params(cls, params: dict) -> bool:
        return (
            isinstance(params, dict)
            and set(params) == set(cls.GRID)
            and all(value in cls.GRID[key] for key, value in params.items())
        )

    @classmethod
    def random_params(cls, rng: random.Random) -> dict:
        return {key: rng.choice(values) for key, values in cls.GRID.items()}

    @classmethod
    def biased_choice(cls, report: dict, rng: random.Random) -> dict:
        params = cls.random_params(rng)
        # Prefer loser-specific regimes; fall back to window regimes when a
        # frame produced no losing trades.
        down = report.get("loser_down_trend_share", report.get("down_trend_share", 0))
        high_vol = report.get("loser_high_vol_share", report.get("high_vol_share", 0))
        if down > 0.6:
            params["ema_slow"], params["rsi_os"] = 50, 25
        if high_vol > 0.6:
            params["sl_atr"], params["tp_atr"], params["min_volume_ratio"] = 2.0, 3.0, 1.25
        return params


class Brain:
    def __init__(self, cfg, db):
        self.cfg = cfg
        self.db = db
        self.backtester = Backtester(cfg)
        # Keep exploration local to this Brain instance.  Mutating Python's
        # global RNG made every restart repeat the same candidate sequence.
        seed = cfg["training"].get("random_seed")
        self.rng = random.Random(seed)

    def _strategy(self, encoded: str, expected_hash: str | None = None):
        data = json.loads(encoded)
        if not isinstance(data, dict):
            raise ValueError("strategy payload must be an object")
        template = data.get("template")
        params = data.get("params", {})
        if data.get("schema_version") != STRATEGY_SCHEMA_VERSION or template not in TEMPLATES or not Refiner.valid_params(params):
            raise ValueError("untrusted or malformed strategy record")
        digest = hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
        if expected_hash is not None and not hmac.compare_digest(str(expected_hash), digest):
            raise ValueError("strategy payload hash mismatch")
        return template, TEMPLATES[template], params

    def quarantine_legacy_strategies(self) -> None:
        for sid, encoded, digest in self.db.q("SELECT sid,json,hash FROM strategies WHERE status IN ('CANDIDATE','ELITE','LIVE_APPROVED','PENDING_APPROVAL','APPROVED')"):
            try:
                self._strategy(encoded, digest)
            except (ValueError, TypeError, json.JSONDecodeError):
                self.db.ex("UPDATE strategies SET status='QUARANTINED' WHERE sid=?", (sid,))
                self.db.ex("INSERT INTO events(kind,msg,ts)VALUES('STRATEGY_QUARANTINED',?,?)", (sid, iso()))
                self.db.audit("STRATEGY_QUARANTINED", {"sid": sid})
                LOG.warning("Quarantined non-template strategy %s; stored source is never executed", sid)

    def _insert_strategy(self, sid: str, template: str, params: dict, generation: int, parent: str | None) -> None:
        payload = {"schema_version": STRATEGY_SCHEMA_VERSION, "template": template, "params": params}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
        self.db.ex(
            "INSERT OR REPLACE INTO strategies(sid,json,score,status,gen,parent,hash,created)VALUES(?,?,?,?,?,?,?,?)",
            (sid, json.dumps(payload, sort_keys=True), -99.0, "CANDIDATE", generation, parent, digest, iso()),
        )

    def seed_population(self) -> None:
        population = self.cfg["training"]["population"]
        # Seed across the full live-eligible template library so the genetic
        # search explores the expanded 100-strategy space from generation 0.
        template_names = [name for name in TEMPLATES if isinstance(name, str)]
        for index in range(population):
            template = template_names[index % len(template_names)]
            self._insert_strategy(f"s_gen0_{index}", template, Refiner.random_params(self.rng), 0, None)

    def _signal_stability(self, frames_full: list[pd.DataFrame], builder, params: dict) -> float:
        """Compare full-history and live-window signals to expose warm-up drift."""
        window = int(self.cfg["execution"]["signal_history_candles"])
        matches = 0
        tested = 0
        for frame in frames_full:
            if len(frame) < window:
                continue
            full_signal = np.asarray(builder(frame, params).get("signal"), dtype=int)
            live_signal = np.asarray(builder(frame.tail(window).reset_index(drop=True), params).get("signal"), dtype=int)
            if not len(full_signal) or not len(live_signal):
                continue
            tested += 1
            # Compare the last 20 bars, not just one: a single warm-up bar can
            # agree by luck, which let borderline signals pass the 0.95 gate.
            tail = min(20, len(full_signal), len(live_signal))
            matches += int(np.array_equal(full_signal[-tail:], live_signal[-tail:]))
        return float(matches / tested) if tested else 0.0

    def evaluate(self, sid: str, frames_is: list[pd.DataFrame], frames_oos: list[pd.DataFrame]) -> tuple[float, dict]:
        rows = self.db.q("SELECT json,hash FROM strategies WHERE sid=?", (sid,))
        if not rows:
            raise ValueError(f"Unknown strategy {sid}")
        _, builder, params = self._strategy(rows[0][0], rows[0][1])
        is_results = [self.backtester.run(frame, builder, params) for frame in frames_is]
        oos_results = [self.backtester.run(frame, builder, params) for frame in frames_oos]
        stats_is, _ = self.backtester.aggregate(is_results)
        score_is = Scorer.score(stats_is, self.cfg["training"]["min_trades"])
        stats_oos, trades_oos = self.backtester.aggregate(oos_results)
        consistency = self.backtester.frame_consistency(oos_results)
        stats_oos["oos_frame_consistency"] = consistency["ratio"]
        stats_oos["oos_frames_traded"] = consistency["traded"]
        stats_oos["oos_frames_total"] = consistency["total"]
        score_oos = Scorer.score(stats_oos, self.cfg["training"]["min_trades"])
        full_frames = [pd.concat((frame_is, frame_oos), ignore_index=True) for frame_is, frame_oos in zip(frames_is, frames_oos, strict=True)]
        stability = self._signal_stability(full_frames, builder, params)
        stats_is["signal_stability"] = stability
        stats_oos["signal_stability"] = stability
        if stability < self.cfg["training"]["min_signal_stability"]:
            score_oos = -9.0
            stats_oos["promotion_eligible"] = False
        else:
            stats_oos["promotion_eligible"] = score_oos > -9.0
        self.db.ex("INSERT INTO backtests(sid,is_oos,score,stats,ts)VALUES(?,?,?,?,?)", (sid, "IS", score_is, json.dumps(stats_is), iso()))
        self.db.ex("INSERT INTO backtests(sid,is_oos,score,stats,ts)VALUES(?,?,?,?,?)", (sid, "OOS", score_oos, json.dumps(stats_oos), iso()))
        # Never concatenate different instruments into one synthetic price series:
        # an instrument boundary would create a false return and distort refinements.
        symbol_reports = [
            FailureAnalyzer.analyze(frame, stats_i.get("trade_details", []))
            for frame, (stats_i, _) in zip(frames_oos, oos_results, strict=True)
        ]

        def _mean(key: str) -> float:
            values = [float(item[key]) for item in symbol_reports]
            return float(np.mean(values)) if values else 0.0

        report = {
            "n": len(trades_oos),
            "high_vol_share": _mean("high_vol_share"),
            "down_trend_share": _mean("down_trend_share"),
            "loser_high_vol_share": _mean("loser_high_vol_share"),
            "loser_down_trend_share": _mean("loser_down_trend_share"),
            "signal_stability": stability,
        }
        self.db.ex("INSERT INTO failures(sid,report,ts)VALUES(?,?,?)", (sid, json.dumps(report), iso()))
        # Promotion always requires the out-of-sample result. In-sample figures
        # are retained for audit only and can never substitute for OOS evidence.
        self.db.ex("UPDATE strategies SET score=? WHERE sid=?", (score_oos, sid))
        return score_oos, report

    def iterate(self, frames_is: list[pd.DataFrame], frames_oos: list[pd.DataFrame], generations: int | None = None) -> None:
        if not frames_is or len(frames_is) != len(frames_oos):
            raise ValueError("Training requires matched in-sample and out-of-sample frames")
        self.quarantine_legacy_strategies()
        training = self.cfg["training"]
        generations = generations or training["iterations"]
        if self.db.q("SELECT COUNT(*) FROM strategies WHERE status='CANDIDATE'")[0][0] == 0:
            self.seed_population()
        last_report: dict = {}
        for generation in range(generations):
            candidates = self.db.q("SELECT sid FROM strategies WHERE status='CANDIDATE'")
            if not candidates:
                break
            scored: list[tuple[float, str]] = []
            for (sid,) in candidates:
                try:
                    score, report = self.evaluate(sid, frames_is, frames_oos)
                except Exception as exc:
                    self.db.ex("UPDATE strategies SET status='QUARANTINED' WHERE sid=?", (sid,))
                    LOG.warning("Quarantined strategy %s after evaluation error: %s", sid, exc)
                    continue
                last_report = report
                scored.append((score, sid))
            scored.sort(reverse=True)
            for score, sid in scored[: training["elite_k"]]:
                if score > -9.0:
                    self.db.ex("UPDATE strategies SET status='ELITE' WHERE sid=?", (sid,))
            for _, sid in scored[training["elite_k"] :]:
                self.db.ex("DELETE FROM strategies WHERE sid=?", (sid,))
            parents = self.db.q("SELECT sid,json,hash FROM strategies WHERE status='ELITE'")
            if not parents:
                LOG.warning("No strategies passed the minimum trade-count gate; waiting for more data")
                break
            for index in range(training["population"] // 2):
                parent_sid, encoded, digest = self.rng.choice(parents)
                parent_template, _, _ = self._strategy(encoded, digest)
                self._insert_strategy(
                    f"s_gen{generation + 1}_{index}",
                    parent_template,
                    Refiner.biased_choice(last_report, self.rng),
                    generation + 1,
                    parent_sid,
                )
            LOG.info("Training generation %s completed; candidates=%s", generation, len(scored))

        # Human-approval gate: with require_human_approval (default true),
        # validated strategies stop at PENDING_APPROVAL and an operator must
        # run `python run.py approve <sid>` before they can trade live
        # capital. This restores the approval-before-live requirement.
        require_approval = bool(training.get("require_human_approval", True))
        for sid, score in self.db.q("SELECT sid,score FROM strategies WHERE status='ELITE' ORDER BY score DESC LIMIT 3"):
            if score >= training["promote_score"]:
                if require_approval:
                    self.db.ex(
                        "UPDATE strategies SET status='PENDING_APPROVAL', approved_by='awaiting_operator', approved_at=? WHERE sid=? AND status='ELITE'",
                        (iso(), sid),
                    )
                    self.db.audit("STRATEGY_PENDING_APPROVAL", {"sid": sid, "oos_score": round(float(score), 4)})
                    LOG.warning("Validated strategy %s (OOS %.3f) is PENDING_APPROVAL; run: python run.py approve %s", sid, score, sid)
                else:
                    self.db.ex(
                        "UPDATE strategies SET status='LIVE_APPROVED', approved_by='autonomous_validation', approved_at=? WHERE sid=? AND status='ELITE'",
                        (iso(), sid),
                    )
                    self.db.audit("STRATEGY_PROMOTED", {"sid": sid, "oos_score": round(float(score), 4)})
                    LOG.warning("Autonomously promoted validated strategy %s (OOS score %.3f)", sid, score)

    def approve(self, sid: str, operator: str = "operator") -> bool:
        """Operator promotion of a PENDING_APPROVAL strategy to LIVE_APPROVED."""
        rows = self.db.q("SELECT score FROM strategies WHERE sid=? AND status='PENDING_APPROVAL'", (sid,))
        if not rows:
            return False
        self.db.ex(
            "UPDATE strategies SET status='LIVE_APPROVED', approved_by=?, approved_at=? WHERE sid=?",
            (operator[:80], iso(), sid),
        )
        self.db.audit("STRATEGY_APPROVED_BY_OPERATOR", {"sid": sid, "operator": operator[:80], "score": float(rows[0][0])})
        LOG.warning("Operator %s approved strategy %s", operator[:80], sid)
        return True

    def approved_strategies(self):
        self.quarantine_legacy_strategies()
        approved = []
        for sid, encoded, digest, score in self.db.q("SELECT sid,json,hash,score FROM strategies WHERE status='LIVE_APPROVED' ORDER BY score DESC, sid"):
            try:
                _, builder, params = self._strategy(encoded, digest)
            except (ValueError, TypeError, json.JSONDecodeError):
                self.db.ex("UPDATE strategies SET status='QUARANTINED' WHERE sid=?", (sid,))
                continue
            approved.append((sid, builder, params, float(score)))
        return approved
