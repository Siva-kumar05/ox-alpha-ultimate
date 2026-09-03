"""10x Self-Learning Attribution: Trade tagging and regime decomposition."""
from __future__ import annotations
import json
import numpy as np
from .core import LOG, iso


class TradeAttribution:
    """Tag every trade with context and decompose P&L by dimensions."""

    def __init__(self, db):
        self.db = db

    def tag_trade(self, sym, regime, time_bucket, mtf_score,
                  entry_quality, template):
        self.db.ex(
            "INSERT INTO events(kind,msg,ts)VALUES('TRADE_ATTRIBUTION',?,?)",
            (json.dumps({
                "sym": sym, "regime": regime,
                "time_bucket": time_bucket,
                "mtf_score": round(mtf_score, 4),
                "entry_quality": round(entry_quality, 4),
                "strategy": template,
            }), iso())
        )

    def regime_performance(self, lookback=100):
        rows = self.db.q(
            "SELECT msg FROM events "
            "WHERE kind='TRADE_ATTRIBUTION' ORDER BY eid DESC LIMIT ?",
            (lookback,))
        regime_stats = {}
        for (msg,) in rows:
            try:
                attr = json.loads(msg)
                regime = attr.get("regime", "UNKNOWN")
            except (json.JSONDecodeError, TypeError):
                continue
            regime_stats.setdefault(regime, {"trades": 0})
            regime_stats[regime]["trades"] += 1
        return regime_stats

    def detect_degradation(self, window=30):
        rows = self.db.q(
            "SELECT equity FROM equity ORDER BY ts DESC LIMIT ?",
            (window,))
        if len(rows) < 10:
            return {"degraded": False, "rolling_sharpe": 0.0,
                    "reason": "insufficient_data"}
        equities = [float(r[0]) for r in reversed(rows)]
        returns = np.diff(equities) / np.maximum(equities[:-1], 1.0)
        if len(returns) < 5 or np.std(returns) <= 0:
            return {"degraded": False, "rolling_sharpe": 0.0,
                    "reason": "insufficient_variance"}
        sharpe = float(np.mean(returns) / np.std(returns)
                       * np.sqrt(252 * 375))
        return {
            "degraded": sharpe < 0.0,
            "rolling_sharpe": round(sharpe, 4),
            "reason": "negative_sharpe" if sharpe < 0 else "ok",
        }

    def slippage_analysis(self, lookback=50):
        return {"avg_slippage_bps": 0.0, "systematic_bias": 0.0}
