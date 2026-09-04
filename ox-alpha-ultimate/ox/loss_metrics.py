"""Loss measurement and risk-adjusted performance tracking.
Tracks realized/unrealized PnL, drawdown, VaR, Sortino, and loss streaks.
Used by the self-training loop to minimize loss and maximize risk-adjusted returns.
"""
from __future__ import annotations
import numpy as np
from .core import now

class LossTracker:
    def __init__(self, db, cfg):
        self.db = db
        self.cfg = cfg

    def daily_loss_pct(self) -> float:
        rows = self.db.q("SELECT pnl FROM trades WHERE intime LIKE ? ORDER BY tid", (f"{now().date().isoformat()}%",))
        if not rows:
            return 0.0
        total = sum(float(r[0]) for r in rows)
        return (total / max(float(self.cfg["capital"]), 1.0)) * 100.0

    def unrealized_pnl(self, positions: dict, latest_prices: dict[str, float]) -> float:
        total = 0.0
        for sym, pos in positions.items():
            px = latest_prices.get(sym, pos.get("avg", 0.0))
            total += (float(px) - float(pos.get("avg", 0.0))) * int(pos.get("qty", 0))
        return total

    def max_consecutive_losses(self) -> int:
        rows = self.db.q("SELECT pnl FROM trades ORDER BY tid DESC LIMIT 50")
        streak = 0
        max_streak = 0
        for (pnl,) in rows:
            if float(pnl) < 0:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0
        return max_streak

    def loss_measure(self, returns: np.ndarray) -> dict:
        """Composite loss measure: combines VaR, Sortino penalty, drawdown."""
        if len(returns) < 3:
            return {"loss_score": 0.0, "var": 0.0, "sortino_penalty": 0.0, "maxdd": 0.0}
        var = float(-np.quantile(returns, 0.01)) if len(returns) > 10 else 0.0
        downside = returns[returns < 0]
        sortino_penalty = float(1.0 / max(float(downside.std()), 0.01)) if len(downside) > 1 else 0.0
        cum = np.cumsum(returns)
        peak = np.maximum.accumulate(cum)
        dd = float(((cum - peak).min())) if len(cum) > 0 else 0.0
        loss_score = var * 0.4 + sortino_penalty * 0.1 + abs(dd) * 0.5
        return {"loss_score": loss_score, "var": var, "sortino_penalty": sortino_penalty, "maxdd": dd}

    def should_halt_on_loss(self) -> tuple[bool, str]:
        if self.max_consecutive_losses() >= int(self.cfg["risk"].get("cooldown_after_losses", 3)) + 2:
            return True, "consecutive loss limit breached"
        return False, "ok"
