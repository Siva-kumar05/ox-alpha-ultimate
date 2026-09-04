"""
Risk Coordinator - Centralized Risk Management
===============================================
Coordinates risk across all agents, enforces global limits,
and provides portfolio-level risk management.

This module also owns the LeverageLadder: leverage is *earned*, not
assumed.  Every agent starts with a conservative fraction of its platform
leverage cap and may only climb after its closed-trade record proves an
edge (profit factor, win rate, bounded drawdown) AND a Monte-Carlo ruin
estimate at the next level stays below the configured threshold.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np

from .base import SharedDataBus, Signal, Position

LOG = logging.getLogger("promax.risk")


@dataclass
class RiskLimits:
    """Global risk limits."""
    max_total_leverage: float = 5.0
    max_portfolio_var_pct: float = 0.05
    max_correlation_exposure: float = 0.7
    max_sector_concentration: float = 0.3
    max_single_position_pct: float = 0.15
    max_daily_loss_pct: float = 0.03
    max_drawdown_pct: float = 0.15
    max_total_positions: int = 50


@dataclass
class AgentRiskState:
    """Risk state for a single agent."""
    agent_id: str
    current_leverage: float = 0.0
    current_positions: int = 0
    daily_pnl: float = 0.0
    daily_loss_pct: float = 0.0
    drawdown_pct: float = 0.0
    var_pct: float = 0.0
    last_update: datetime = field(default_factory=datetime.now)
    blocked: bool = False
    block_reason: str = ""


class RiskCoordinator:
    """
    Centralized risk management across all agents.
    Enforces global limits and coordinates risk across agents.
    """

    def __init__(self, data_bus: SharedDataBus, config: Optional[Dict] = None):
        self.data_bus = data_bus
        self.config = config or {}

        # Limits
        self.limits = RiskLimits(**self.config.get('global_limits', {}))

        # Per-agent risk state
        self.agent_states: Dict[str, AgentRiskState] = {}
        # Per-agent configured risk params (stored at registration so limit
        # checks use each agent's own maxima instead of hardcoded guesses).
        self.agent_params: Dict[str, Dict] = {}
        # Optional performance-gated leverage ladder (attach_ladder).
        self.ladder: Optional["LeverageLadder"] = None
        self._lock = threading.RLock()

        # Portfolio-level tracking
        self.portfolio_positions: Dict[str, Position] = {}
        self.portfolio_pnl: float = 0.0
        self.peak_equity: float = 0.0
        self.current_equity: float = 0.0

        # Correlation tracking
        self.correlation_matrix: Optional[np.ndarray] = None
        self.symbol_list: List[str] = []

        # Subscribe to data
        self._setup_subscriptions()

        # Start monitoring
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None

    def attach_ladder(self, capital_allocator, ladder_config: Optional[Dict] = None) -> None:
        """Enable the performance-gated leverage ladder for all agents."""
        self.ladder = LeverageLadder(capital_allocator, ladder_config)
        LOG.info("Leverage ladder attached: agents start at %.0f%% of their "
                 "platform leverage cap and earn more with evidence",
                 self.ladder.config.start_fraction * 100)

    def _setup_subscriptions(self) -> None:
        """Subscribe to relevant data topics."""
        pass  # Will be set up when orchestrator provides data bus

    def register_agent(self, agent_id: str, risk_params: Dict) -> None:
        """Register an agent for risk management (params kept for limit checks)."""
        with self._lock:
            self.agent_states[agent_id] = AgentRiskState(
                agent_id=agent_id,
                blocked=False
            )
            self.agent_params[agent_id] = dict(risk_params or {})
        LOG.info(f"Registered agent for risk management: {agent_id}")

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister agent."""
        with self._lock:
            self.agent_states.pop(agent_id, None)
            self.agent_params.pop(agent_id, None)

    async def approve_signal(self, signal: Signal) -> bool:
        """
        Approve or reject a trading signal based on risk limits.
        Returns True if approved, False if rejected.
        """
        with self._lock:
            agent_state = self.agent_states.get(signal.agent_id)
            if not agent_state:
                LOG.warning(f"Signal from unregistered agent: {signal.agent_id}")
                return False

            if agent_state.blocked:
                LOG.warning(f"Signal rejected: Agent {signal.agent_id} is blocked: {agent_state.block_reason}")
                return False

            # Check agent-level limits
            if not self._check_agent_limits(agent_state, signal):
                return False

            # Check portfolio-level limits
            if not self._check_portfolio_limits(signal):
                return False

            return True

    def _check_agent_limits(self, agent_state: AgentRiskState, signal: Signal) -> bool:
        """Check agent-specific risk limits (configured params + ladder)."""
        params = self.agent_params.get(signal.agent_id, {})
        max_positions = int(params.get("max_concurrent_positions", 10))
        max_daily_loss = float(params.get("max_daily_loss_pct", 0.05))
        # The ladder scales how much of the platform leverage cap the agent
        # may actually use; it starts conservative and is earned with evidence.
        platform_cap = float(params.get("max_leverage", 1.0))
        if self.ladder is not None:
            max_leverage = self.ladder.allowed_leverage(signal.agent_id, platform_cap)
        else:
            max_leverage = platform_cap

        if agent_state.current_positions >= max_positions:
            LOG.warning(f"Agent {signal.agent_id} max positions reached")
            return False

        if signal.action.lower() in ("buy", "open", "add") and signal.leverage > max_leverage + 1e-9:
            LOG.warning(
                f"Signal leverage {signal.leverage:.2f}x exceeds ladder-allowed "
                f"{max_leverage:.2f}x for {signal.agent_id} "
                f"(platform cap {platform_cap:.2f}x, ladder level "
                f"{self.ladder.level(signal.agent_id) if self.ladder else 'n/a'})"
            )
            return False

        if agent_state.daily_loss_pct >= max_daily_loss:
            LOG.warning(f"Agent {signal.agent_id} daily loss limit reached")
            if self.ladder is not None:
                self.ladder.demote(signal.agent_id, "daily loss limit breached")
            return False

        return True

    def _check_portfolio_limits(self, signal: Signal) -> bool:
        """Check portfolio-level risk limits."""
        # Check total leverage
        total_leverage = self._calculate_total_leverage()
        if total_leverage + signal.leverage > self.limits.max_total_leverage:
            LOG.warning("Portfolio max leverage would be exceeded")
            return False

        # Check correlation exposure
        if not self._check_correlation_exposure(signal):
            return False

        # Check sector concentration
        if not self._check_sector_concentration(signal):
            return False

        # Check single position size
        position_value = signal.price * signal.quantity * signal.leverage
        portfolio_value = self.current_equity or 100000
        if position_value / portfolio_value > self.limits.max_single_position_pct:
            LOG.warning("Position size exceeds limit")
            return False

        return True

    def _check_correlation_exposure(self, signal: Signal) -> bool:
        """Check if adding this position would exceed correlation limits."""
        # Simplified - would use actual correlation matrix
        return True

    def _check_sector_concentration(self, signal: Signal) -> bool:
        """Check sector concentration limits."""
        # Simplified - would map symbols to sectors
        return True

    def _calculate_total_leverage(self) -> float:
        """Calculate current portfolio leverage."""
        if self.current_equity <= 0:
            return 0.0
        total_exposure = sum(
            abs(p.quantity * p.current_price * p.leverage)
            for p in self.portfolio_positions.values()
        )
        return total_exposure / self.current_equity

    async def update_agent_state(self, agent_id: str, positions: Dict[str, Position]) -> None:
        """Update agent risk state from current positions."""
        with self._lock:
            if agent_id not in self.agent_states:
                return

            agent_state = self.agent_states[agent_id]
            agent_state.current_positions = len(positions)

            # Calculate leverage
            total_exposure = sum(
                abs(p.quantity * p.current_price * p.leverage)
                for p in positions.values()
            )
            equity = self.current_equity or 100000
            agent_state.current_leverage = total_exposure / equity if equity > 0 else 0

            # Calculate daily P&L
            daily_pnl = sum(p.unrealized_pnl + p.realized_pnl for p in positions.values())
            agent_state.daily_pnl = daily_pnl
            agent_state.daily_loss_pct = abs(min(0, daily_pnl)) / (self.current_equity or 100000)

            # Update portfolio positions
            for symbol, pos in positions.items():
                self.portfolio_positions[f"{agent_id}:{symbol}"] = pos

            agent_state.last_update = datetime.now()

    async def check_agent_limits(self, agent_id: str, positions: Dict[str, Position]) -> bool:
        """Check if agent is within risk limits."""
        with self._lock:
            agent_state = self.agent_states.get(agent_id)
            if not agent_state:
                return False

            await self.update_agent_state(agent_id, positions)

            # Check if blocked
            if agent_state.blocked:
                return False

            # Check drawdown
            if agent_state.drawdown_pct > 0.15:  # 15% max drawdown
                agent_state.blocked = True
                agent_state.block_reason = "Max drawdown exceeded"
                LOG.error(f"Agent {agent_id} blocked: {agent_state.block_reason}")
                return False

            return True

    async def update_portfolio_equity(self, equity: float) -> None:
        """Update portfolio equity for risk calculations."""
        with self._lock:
            self.current_equity = equity
            if equity > self.peak_equity:
                self.peak_equity = equity

            # Calculate drawdown
            if self.peak_equity > 0:
                drawdown = (self.peak_equity - equity) / self.peak_equity
                for agent_state in self.agent_states.values():
                    agent_state.drawdown_pct = drawdown

                    # Auto-block on excessive drawdown
                    if drawdown > self.limits.max_drawdown_pct:
                        agent_state.blocked = True
                        agent_state.block_reason = f"Portfolio drawdown {drawdown:.1%} exceeds limit"

    async def get_risk_report(self) -> Dict:
        """Get comprehensive risk report."""
        with self._lock:
            return {
                "portfolio": {
                    "equity": self.current_equity,
                    "peak_equity": self.peak_equity,
                    "drawdown_pct": (self.peak_equity - self.current_equity) / self.peak_equity if self.peak_equity > 0 else 0,
                    "total_leverage": self._calculate_total_leverage(),
                    "total_positions": len(self.portfolio_positions),
                    "daily_pnl": sum(s.daily_pnl for s in self.agent_states.values()),
                },
                "limits": {
                    "max_total_leverage": self.limits.max_total_leverage,
                    "max_drawdown_pct": self.limits.max_drawdown_pct,
                    "max_daily_loss_pct": self.limits.max_daily_loss_pct,
                },
                "agents": {
                    agent_id: {
                        "leverage": state.current_leverage,
                        "positions": state.current_positions,
                        "daily_pnl": state.daily_pnl,
                        "daily_loss_pct": state.daily_loss_pct,
                        "drawdown_pct": state.drawdown_pct,
                        "blocked": state.blocked,
                        "block_reason": state.block_reason
                    }
                    for agent_id, state in self.agent_states.items()
                },
                "ladder": self.ladder.report() if self.ladder else None,
                "timestamp": datetime.now().isoformat()
            }

    def emergency_stop_all(self, reason: str = "Emergency stop") -> None:
        """Emergency stop all agents."""
        LOG.critical(f"EMERGENCY STOP: {reason}")
        with self._lock:
            for agent_state in self.agent_states.values():
                agent_state.blocked = True
                agent_state.block_reason = reason


# ────────────────────────────────────────────────────────────────────────────
# Performance-gated leverage ladder
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class LadderConfig:
    """Evidence thresholds an agent must meet to climb a leverage level."""
    min_trades: int = 20                 # closed trades in the lookback window
    min_profit_factor: float = 1.3
    min_win_rate: float = 0.5
    max_recent_drawdown: float = 0.15    # fraction of agent budget
    start_fraction: float = 0.25         # level 1 = 25% of the platform cap
    step_multiple: float = 2.0           # each level doubles the allowed fraction
    max_level: int = 3                   # level 3 == 100% of the platform cap
    ruin_prob_max: float = 0.05          # Monte-Carlo gate for promotion
    ruin_drawdown: float = 0.5           # "ruin" = 50% peak-to-trough
    lookback_trades: int = 60
    mc_paths: int = 4000
    mc_horizon_trades: int = 100
    seed: int = 7


def monte_carlo_survival(
    win_rate: float,
    avg_win_fraction: float,
    avg_loss_fraction: float,
    n_trades: int,
    paths: int = 4000,
    ruin_drawdown: float = 0.5,
    seed: int = 7,
) -> Dict[str, float]:
    """Simulate equity paths of fixed-fraction returns and estimate ruin odds.

    ``avg_win_fraction``/``avg_loss_fraction`` are per-trade equity return
    fractions (leverage already baked in).  Returns the probability of ever
    touching the ruin drawdown, drawdown quantiles, and the median final
    equity multiple.  This is a risk-gating estimate, not a forecast.
    """
    if n_trades <= 0 or paths <= 0:
        return {"p_ruin": 0.0, "median_max_dd": 0.0, "p90_max_dd": 0.0,
                "median_final_multiple": 1.0}
    rng = np.random.default_rng(seed)
    wins = rng.random((paths, n_trades)) < win_rate
    per_trade = np.where(wins, avg_win_fraction, -avg_loss_fraction)
    equity = np.cumprod(1.0 + per_trade, axis=1)
    peak = np.maximum.accumulate(np.maximum(equity, 1e-12), axis=1)
    dd = 1.0 - equity / peak
    max_dd = dd.max(axis=1)
    return {
        "p_ruin": float(np.mean(max_dd >= ruin_drawdown)),
        "median_max_dd": float(np.median(max_dd)),
        "p90_max_dd": float(np.quantile(max_dd, 0.9)),
        "median_final_multiple": float(np.median(equity[:, -1])),
    }


class LeverageLadder:
    """Earned leverage: each agent climbs toward its platform cap on evidence.

    Level 1 allows ``start_fraction`` of the platform cap; every promotion
    multiplies the allowed fraction by ``step_multiple``; the top level
    unlocks the full cap.  Promotion requires, over the lookback window:
    enough closed trades, profit factor >= threshold, win rate >= threshold,
    recent drawdown <= threshold, and a Monte-Carlo ruin probability at the
    NEXT level below ``ruin_prob_max``.  Demotion is immediate when a daily
    loss limit is breached or the evidence degrades.
    """

    def __init__(self, capital_allocator, config: Optional[Dict] = None):
        self.allocator = capital_allocator
        cfg = dict(config or {})
        for k in list(cfg.keys()):
            if k not in LadderConfig.__dataclass_fields__:
                cfg.pop(k)
        self.config = LadderConfig(**cfg)
        self._levels: Dict[str, int] = {}
        self._history: Dict[str, List[str]] = defaultdict(list)
        self._load_persisted_levels()

    # ── persistence ───────────────────────────────────────────────────
    def _load_persisted_levels(self) -> None:
        db = getattr(self.allocator, "db", None)
        if db is None:
            return
        try:
            rows = db.q("SELECT agent, level FROM ladder_levels")
            for agent_id, level in rows:
                self._levels[agent_id] = int(level)
        except Exception:
            # Table absent in older databases; create it and start at 1.
            try:
                db.ex(
                    "CREATE TABLE IF NOT EXISTS ladder_levels("
                    "agent TEXT PRIMARY KEY, level INTEGER NOT NULL, updated TEXT)"
                )
            except Exception:
                pass

    def _persist_level(self, agent_id: str) -> None:
        db = getattr(self.allocator, "db", None)
        if db is None:
            return
        try:
            db.ex(
                "INSERT INTO ladder_levels(agent,level,updated)VALUES(?,?,?) "
                "ON CONFLICT(agent) DO UPDATE SET level=excluded.level, updated=excluded.updated",
                (agent_id, self._levels.get(agent_id, 1), datetime.now().isoformat()),
            )
        except Exception as exc:
            LOG.warning(f"Ladder persistence failed for {agent_id}: {exc}")

    # ── level math ────────────────────────────────────────────────────
    def level(self, agent_id: str) -> int:
        return int(self._levels.get(agent_id, 1))

    def fraction_for_level(self, level: int) -> float:
        return min(1.0, self.config.start_fraction * (self.config.step_multiple ** (level - 1)))

    def allowed_leverage(self, agent_id: str, platform_cap: float) -> float:
        return float(platform_cap) * self.fraction_for_level(self.level(agent_id))

    def _log(self, agent_id: str, message: str) -> None:
        self._history[agent_id].append(f"{datetime.now().isoformat()} {message}")
        self._history[agent_id] = self._history[agent_id][-50:]

    # ── evaluation ────────────────────────────────────────────────────
    def evaluate(self, agent_id: str) -> tuple[int, str, str]:
        """Re-evaluate one agent. Returns (level, action, reason)."""
        cfg = self.config
        level = self.level(agent_id)
        trades = self.allocator.closed_trades(agent_id, limit=cfg.lookback_trades)
        if len(trades) < cfg.min_trades:
            return level, "hold", f"evidence {len(trades)}/{cfg.min_trades} trades"

        chronological = list(reversed(trades))
        pnls = [float(t["pnl"]) for t in chronological]
        gross_win = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        profit_factor = gross_win / gross_loss if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)
        win_rate = sum(1 for p in pnls if p > 0) / len(pnls)

        # Recent drawdown measured against the agent's capital budget.
        budget = float(self.allocator.budget(agent_id)) or 1.0
        cum = np.cumsum(pnls)
        peak = np.maximum.accumulate(cum)
        drawdown_abs = float(np.max(peak - cum)) if len(cum) else 0.0
        drawdown_pct = max(0.0, drawdown_abs) / budget

        # Monte-Carlo ruin estimate at the NEXT level: per-trade return
        # fractions observed at the current level scale with the leverage
        # fraction the next level would allow.
        ret_fracs = []
        for trade in chronological:
            notional = abs(float(trade.get("entry_price", 0.0)) * float(trade.get("qty", 0.0)))
            if notional <= 0:
                notional = budget
            ret_fracs.append(float(trade["pnl"]) / notional)
        wins = [r for r in ret_fracs if r > 0]
        losses = [abs(r) for r in ret_fracs if r < 0]
        avg_win = float(np.median(wins)) if wins else 0.005
        avg_loss = float(np.median(losses)) if losses else 0.005

        current_frac = self.fraction_for_level(level)
        next_frac = self.fraction_for_level(level + 1)
        scale = next_frac / max(current_frac, 1e-9)
        mc = monte_carlo_survival(
            win_rate=win_rate,
            avg_win_fraction=avg_win * scale,
            avg_loss_fraction=avg_loss * scale,
            n_trades=cfg.mc_horizon_trades,
            paths=cfg.mc_paths,
            ruin_drawdown=cfg.ruin_drawdown,
            seed=cfg.seed,
        )

        if level < cfg.max_level:
            if mc["p_ruin"] > cfg.ruin_prob_max:
                reason = (f"hold: ruin probability {mc['p_ruin']:.1%} at next level "
                          f"> limit {cfg.ruin_prob_max:.0%}")
                self._log(agent_id, reason)
                return level, "hold", reason
            if (profit_factor >= cfg.min_profit_factor and win_rate >= cfg.min_win_rate
                    and drawdown_pct <= cfg.max_recent_drawdown):
                level += 1
                self._levels[agent_id] = level
                self._persist_level(agent_id)
                reason = (f"promoted to {level}: PF {profit_factor:.2f}, WR {win_rate:.0%}, "
                          f"DD {drawdown_pct:.1%}, ruin@next {mc['p_ruin']:.1%}")
                self._log(agent_id, reason)
                LOG.info(f"Ladder promotion {agent_id}: {reason}")
                return level, "promoted", reason

        if (profit_factor < cfg.min_profit_factor * 0.8 or win_rate < cfg.min_win_rate * 0.8
                or drawdown_pct > cfg.max_recent_drawdown):
            if level > 1:
                level -= 1
                self._levels[agent_id] = level
                self._persist_level(agent_id)
                reason = (f"demoted to {level}: PF {profit_factor:.2f}, WR {win_rate:.0%}, "
                          f"DD {drawdown_pct:.1%}")
                self._log(agent_id, reason)
                LOG.warning(f"Ladder demotion {agent_id}: {reason}")
                return level, "demoted", reason

        reason = (f"hold: PF {profit_factor:.2f}, WR {win_rate:.0%}, DD {drawdown_pct:.1%}, "
                  f"ruin@next {mc['p_ruin']:.1%}")
        return level, "hold", reason

    def demote(self, agent_id: str, reason: str) -> None:
        level = self.level(agent_id)
        if level <= 1:
            return
        self._levels[agent_id] = level - 1
        self._persist_level(agent_id)
        self._log(agent_id, f"demoted to {level - 1}: {reason}")
        LOG.warning(f"Ladder demotion {agent_id}: {reason} (now level {level - 1})")

    def report(self) -> Dict[str, Any]:
        agents = set(self._levels) | set(self._history)
        return {
            agent_id: {
                "level": self.level(agent_id),
                "allowed_fraction_of_cap": self.fraction_for_level(self.level(agent_id)),
                "history": list(self._history.get(agent_id, []))[-5:],
            }
            for agent_id in sorted(agents)
        }
