"""
Bull/Bear Debate Panel — adversarial pre-execution review (no LLM required).
============================================================================

Ported from the TradingAgents design (arXiv:2412.20138): before capital is
deployed, a *bull* case and a *bear* case are argued from independent
evidence and a verdict decides.  Here the debaters are deterministic
indicator ensembles from the shared ``ox/indicators.py`` library (no API
keys, no hallucination, reproducible), plus a per-agent "memory of past
mistakes" journal in the TradingAgents spirit: closed trades are recorded
with their regime features and outcomes, and a losing streak in a similar
regime raises the bear's weight — the reflection loop, done cheaply.

Verdict semantics (returned to the caller):
  verdict in [-1, +1]; > threshold → buy passes with strength scaled down
  to |verdict|; <= threshold → buy is vetoed (a *non-event*: it simply is
  not published; nothing is queued).  Sells never enter the debate.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .. import indicators as I

DEFAULT_THRESHOLD = 0.15


def _last(x: np.ndarray, default: float = float("nan")) -> float:
    if x is None or len(x) == 0:
        return default
    value = x[-1]
    if isinstance(value, float) and np.isnan(value):
        valid = x[~np.isnan(x)]
        value = valid[-1] if len(valid) else default
    return float(value)


class TradeMemory:
    """Per-agent journal of closed trades with regime features + outcome.

    Mirrors TradingAgents' TradingMemoryLog at small scale: decisions are
    stored, retrieved as context for future debates, and updated with
    outcomes.  Persisted as JSON under ``<state_dir>/<agent>_memory.json``
    so lessons survive restarts.
    """

    def __init__(self, agent_id: str, state_dir: Path | str, max_entries: int = 200):
        self.agent_id = agent_id
        self.path = Path(state_dir) / f"{agent_id}_memory.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_entries = max_entries
        self._lock = threading.RLock()
        self._entries: List[Dict[str, Any]] = self._load()

    def _load(self) -> List[Dict[str, Any]]:
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, list):
                return data[-self.max_entries:]
        except (OSError, ValueError):
            pass
        return []

    def _flush(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._entries, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    def record(self, symbol: str, features: Dict[str, float], verdict: float) -> None:
        with self._lock:
            self._entries.append({
                "symbol": symbol, "features": features, "verdict": verdict,
                "outcome": None, "pnl": None,
            })
            self._entries = self._entries[-self.max_entries:]
            self._flush()

    def update_outcome(self, symbol: str, pnl: float, max_lookback: int = 10) -> bool:
        """Attach an outcome to the most recent undecided entry for symbol."""
        with self._lock:
            for entry in reversed(self._entries[-max_lookback:]):
                if entry["symbol"] == symbol and entry["outcome"] is None:
                    entry["outcome"] = "win" if pnl > 0 else "loss"
                    entry["pnl"] = round(float(pnl), 4)
                    self._flush()
                    return True
            return False

    def loss_streak(self, symbol: str, lookback: int = 8) -> int:
        """Consecutive recent losses on the same symbol (0 if none)."""
        with self._lock:
            streak = 0
            for entry in reversed([e for e in self._entries if e["symbol"] == symbol][-lookback:]):
                if entry["outcome"] == "loss":
                    streak += 1
                elif entry["outcome"] == "win":
                    break
            return streak

    def regime_loss_rate(self, features: Dict[str, float], tolerance: float = 0.5,
                         min_samples: int = 4) -> Optional[float]:
        """Loss rate in historically similar regimes (simple NN on features)."""
        with self._lock:
            decided = [e for e in self._entries if e["outcome"] is not None
                       and e.get("features")]
            if len(decided) < min_samples:
                return None
            keys = sorted(set(features) & set(decided[0]["features"]))
            if not keys:
                return None
            query = np.array([features[k] for k in keys], dtype=float)
            losses = same = 0
            for entry in decided:
                ref = np.array([entry["features"].get(k, 0.0) for k in keys], dtype=float)
                denom = np.linalg.norm(query) * np.linalg.norm(ref)
                if denom == 0:
                    continue
                if 1.0 - float(query @ ref) / denom <= tolerance:  # cosine similarity
                    same += 1
                    losses += entry["outcome"] == "loss"
            return losses / same if same >= min_samples else None


class DebatePanel:
    """Deterministic bull/bear debate over a price series before buys."""

    def __init__(self, threshold: float = DEFAULT_THRESHOLD, state_dir: Path | str = "state/promax"):
        self.threshold = float(threshold)
        self.state_dir = Path(state_dir)
        self._memories: Dict[str, TradeMemory] = {}
        self._lock = threading.RLock()

    def memory(self, agent_id: str) -> TradeMemory:
        with self._lock:
            if agent_id not in self._memories:
                self._memories[agent_id] = TradeMemory(agent_id, self.state_dir)
            return self._memories[agent_id]

    # ── feature extraction (the analysts) ─────────────────────────────
    @staticmethod
    def features(closes: np.ndarray) -> Dict[str, float]:
        """Regime features shared by both debaters and the memory journal."""
        c = np.asarray(closes, dtype=float)
        n = len(c)
        if n < 60:
            return {}
        rsi = _last(I.rsi(c, 14), 50.0)
        ema20 = I.ema(c, 20)
        ema50 = I.ema(c, 50)
        atr = _last(I.atr(c, c, c, 14), 0.0)  # close-only proxy: TR == |Δc|
        slope = float(c[-1] / c[-21] - 1.0) if n >= 21 else 0.0
        return {
            "rsi14": round(rsi, 2),
            "trend": round(float(_last(ema20) - _last(ema50)) / max(_last(ema50), 1e-9), 5),
            "vol": round(_last(I.rsi(I.ema(c, 20), 14), 50.0), 2),  # variability proxy
            "atr_pct": round(atr / max(c[-1], 1e-9), 5),
            "slope20": round(slope, 5),
        }

    # ── the debate ────────────────────────────────────────────────────
    def debate(self, agent_id: str, symbol: str, closes: np.ndarray,
               side_hint: str = "buy") -> Dict[str, Any]:
        """Run bull vs bear on the series. Returns verdict + rationale."""
        c = np.asarray(closes, dtype=float)
        feats = self.features(c)
        if not feats or len(c) < 60:
            return {"verdict": 0.0, "pass": False, "reason": "insufficient history",
                    "features": {}, "bull": 0.0, "bear": 0.0}

        bull_score = 0.0
        bear_score = 0.0
        notes: List[str] = []

        # ── bull researcher ──
        if feats["trend"] > 0:
            bull_score += 1.0
            notes.append(f"bull: ema20>ema50 (+{feats['trend']:.1%})")
        if 40 < feats["rsi14"] < 68:
            bull_score += 0.6
            notes.append(f"bull: rsi {feats['rsi14']:.0f} in trend band")
        if feats["slope20"] > 0.01:
            bull_score += 0.6
            notes.append(f"bull: 20-bar slope +{feats['slope20']:.1%}")
        donch_up, donch_mid, donch_low = I.donchian(c, c, 20)
        if not np.isnan(_last(donch_mid)) and c[-1] > _last(donch_mid):
            bull_score += 0.4
            notes.append("bull: above 20-bar midline")

        # ── bear researcher ──
        if feats["trend"] < 0:
            bear_score += 1.0
            notes.append(f"bear: ema20<ema50 ({feats['trend']:.1%})")
        if feats["rsi14"] > 72:
            bear_score += 0.8
            notes.append(f"bear: rsi {feats['rsi14']:.0f} overbought")
        if feats["rsi14"] < 32:
            bear_score += 0.4
            notes.append(f"bear: rsi {feats['rsi14']:.0f} breaking down")
        if feats["atr_pct"] > 0.03:
            bear_score += 0.5
            notes.append(f"bear: vol elevated atr={feats['atr_pct']:.1%}")
        if c[-1] < _last(donch_low, c[-1]):
            bear_score += 0.5
            notes.append("bear: below 20-bar low")

        # ── memory of past mistakes (reflection) ──
        mem = self.memory(agent_id)
        streak = mem.loss_streak(symbol)
        if streak >= 2:
            bear_score += 0.3 * streak
            notes.append(f"bear: {streak} straight losses on {symbol} (memory)")
        regime_loss = mem.regime_loss_rate(feats)
        if regime_loss is not None and regime_loss > 0.6:
            bear_score += 0.6
            notes.append(f"bear: {regime_loss:.0%} loss rate in similar regimes (memory)")

        total = bull_score + bear_score
        verdict = (bull_score - bear_score) / total if total > 0 else 0.0
        passed = verdict > self.threshold if side_hint in ("buy", "open", "add") else True

        result = {
            "verdict": round(verdict, 3),
            "pass": bool(passed),
            "reason": "; ".join(notes) if notes else "no strong case either way",
            "features": feats,
            "bull": round(bull_score, 2),
            "bear": round(bear_score, 2),
        }
        if passed:
            mem.record(symbol, feats, verdict)
        return result
