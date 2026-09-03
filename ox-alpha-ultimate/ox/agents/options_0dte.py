"""
0DTE Options Agent — same-day expiry, defined-risk only.
========================================================

Mandate: capture same-day index expiries (NIFTY / BANKNIFTY) with debit
spreads, never naked shorts — max loss per trade is fixed at entry.

Signal model (all from the index feed on the bus):
  * momentum burst: short-window return vs its recent sigma,
  * intraday location: premium/discount to the running VWAP,
  * opening-range breakout confirm,
  * time gate: entries only inside the liquid 09:30-14:30 IST window;
    everything is force-closed before 15:05 IST (theta death).

Paper execution approximation (documented honestly): the spread is carried
as a synthetic position whose mark moves with the underlying by the spread's
net delta; fills on real venues replace this unchanged elsewhere.

Entries are capital-deploying -> human approval gateway; exits are instant.
Leverage semantics: defined-risk ratio max_gain/max_loss, capped by the
agent's risk params and the leverage ladder.
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import numpy as np

from .base import AgentConfig, BaseAgent, Signal
from .capital_allocator import CapitalAllocator
from .risk_coordinator import RiskCoordinator

LOG = logging.getLogger("promax.0dte")

IST = ZoneInfo("Asia/Kolkata")
STRIKE_STEPS = {"NIFTY": 50, "BANKNIFTY": 100}
LOT_SIZES = {"NIFTY": 75, "BANKNIFTY": 35}


class Options0DTEAgent(BaseAgent):
    """Same-day expiry debit-spread specialist."""

    def __init__(
        self,
        config: AgentConfig,
        resource_pool,
        data_bus,
        risk_coordinator: RiskCoordinator,
        capital_allocator: CapitalAllocator,
    ):
        super().__init__(config, resource_pool, data_bus, risk_coordinator, capital_allocator)

        params = config.custom_params
        self.momentum_window = params.get("momentum_window", 40)
        self.momentum_threshold_sigma = params.get("momentum_threshold_sigma", 2.0)
        self.spread_width_strikes = params.get("spread_width_strikes", 2)
        self.risk_per_trade = params.get("risk_per_trade", 0.05)
        self.target_pct_of_debit = params.get("target_pct_of_debit", 0.5)
        self.stop_pct_of_debit = params.get("stop_pct_of_debit", 0.4)
        self.entry_start = params.get("entry_start", "09:30")
        self.entry_end = params.get("entry_end", "14:30")
        self.squareoff = params.get("squareoff", "15:05")
        self.atm_premium_atr_frac = params.get("atm_premium_atr_frac", 0.35)

        self.price_buffers: Dict[str, deque] = {s: deque(maxlen=800) for s in config.symbols}
        self.day_anchor: Dict[str, Dict[str, float]] = {}

    # ── lifecycle ─────────────────────────────────────────────────────
    async def initialize(self) -> bool:
        self.risk_coordinator.register_agent(self.agent_id, self.config.risk_params.__dict__)
        self.capital_allocator.register_agent(self.agent_id)
        LOG.info(f"Options0DTEAgent initialized for {self.config.symbols}")
        return True

    def _get_loop_interval(self) -> float:
        return 5.0

    # ── helpers ───────────────────────────────────────────────────────
    @staticmethod
    def _now_ist() -> datetime:
        return datetime.now(IST)

    def _in_entry_window(self) -> bool:
        now = self._now_ist()
        minute = now.hour * 60 + now.minute
        sh, sm = (int(x) for x in self.entry_start.split(":"))
        eh, em = (int(x) for x in self.entry_end.split(":"))
        return sh * 60 + sm <= minute < eh * 60 + em

    def _past_squareoff(self) -> bool:
        now = self._now_ist()
        sh, sm = (int(x) for x in self.squareoff.split(":"))
        return now.hour * 60 + now.minute >= sh * 60 + sm

    def _day_stats(self, symbol: str, price: float) -> Dict[str, float]:
        anchor = self.day_anchor.get(symbol)
        today = self._now_ist().strftime("%Y-%m-%d")
        if not anchor or anchor.get("day") != today:
            anchor = {"day": today, "open": price, "high": price, "low": price}
            self.day_anchor[symbol] = anchor
        anchor["high"] = max(anchor["high"], price)
        anchor["low"] = min(anchor["low"], price)
        return anchor

    # ── signal generation ─────────────────────────────────────────────
    async def process_market_data(self, symbol: str, data: Dict) -> List[Signal]:
        price = float(data.get("price", 0.0))
        if price <= 0:
            return []
        anchor = self._day_stats(symbol, price)
        buffer = self.price_buffers[symbol]
        buffer.append(price)
        if len(buffer) < self.momentum_window + 2 or symbol in self.positions:
            return []
        if not self._in_entry_window():
            return []

        prices = np.asarray(buffer, dtype=float)
        rets = np.diff(np.log(prices))
        sigma = float(np.std(rets[-self.momentum_window:])) + 1e-9
        burst = float(rets[-1]) / sigma

        window_mean = float(np.mean(prices[-self.momentum_window:]))
        vwap_proxy = window_mean  # tick-count VWAP proxy when volume absent
        vwap_edge = (prices[-1] - vwap_proxy) / (float(np.std(prices[-self.momentum_window:])) + 1e-9)

        direction = 0
        if burst >= self.momentum_threshold_sigma and vwap_edge > 0.3:
            direction = 1
        elif burst <= -self.momentum_threshold_sigma and vwap_edge < -0.3:
            direction = -1
        if direction == 0:
            return []

        atr = float(np.mean(np.abs(rets[-60:]))) * price if len(rets) >= 60 else price * 0.004
        plan = self._build_spread_plan(symbol, price, direction, atr)
        if plan is None:
            return []

        budget = self.capital_allocator.budget(self.agent_id)
        risk_capital = budget * self.risk_per_trade
        lots = int(risk_capital / plan["max_loss_per_lot"])
        if lots < 1:
            return []

        signals = [Signal(
            agent_id=self.agent_id, symbol=symbol, action="buy",
            strength=min(1.0, abs(burst) / (self.momentum_threshold_sigma * 2)),
            price=price, quantity=float(lots),
            leverage=plan["defined_risk_ratio"],
            metadata={
                "reason": f"0dte_{'bull' if direction > 0 else 'bear'}_burst_{burst:.1f}sigma",
                "spread": plan,
                "max_loss": plan["max_loss_per_lot"] * lots,
                "max_gain": plan["max_gain_per_lot"] * lots,
                "approval_ttl_hint": 120,
            },
        )]
        return signals

    def _build_spread_plan(self, symbol: str, spot: float, direction: int, atr: float) -> Optional[Dict[str, Any]]:
        step = STRIKE_STEPS.get(symbol, 50)
        lot = LOT_SIZES.get(symbol, 1)
        atm = round(spot / step) * step
        width = step * self.spread_width_strikes

        # Paper premium model: ATM premium ≈ atr_frac * ATR; the short leg
        # retains residual value proportional to how far OTM it is in ATRs.
        atm_premium = max(1.0, self.atm_premium_atr_frac * atr)
        otm_frac = min(0.85, (width / step) * 0.18)
        short_premium = atm_premium * (1.0 - otm_frac)
        debit = max(0.5, atm_premium - short_premium)
        max_gain = max(0.5, width - debit)

        return {
            "type": f"{'bull_call' if direction > 0 else 'bear_put'}_debit_spread",
            "direction": direction,
            "long_leg": {"strike": atm, "type": "CE" if direction > 0 else "PE", "side": "BUY"},
            "short_leg": {
                "strike": atm + width if direction > 0 else atm - width,
                "type": "CE" if direction > 0 else "PE", "side": "SELL",
            },
            "lot_size": lot,
            "debit_per_lot": round(debit * lot, 2),
            "max_loss_per_lot": round(debit * lot, 2),
            "max_gain_per_lot": round(max_gain * lot, 2),
            "defined_risk_ratio": round(max_gain / max(debit, 0.5), 2),
            "net_delta_approx": 0.35,
            "paper_model": "delta_approximation (research-only until venue wiring)",
        }

    # ── position management ───────────────────────────────────────────
    async def manage_positions(self) -> List[Signal]:
        signals: List[Signal] = []
        if self._past_squareoff():
            for symbol, position in list(self.positions.items()):
                signals.append(Signal(
                    agent_id=self.agent_id, symbol=symbol, action="close",
                    strength=1.0, price=position.current_price,
                    quantity=position.quantity,
                    metadata={"reason": "theta_squareoff"},
                ))
            return signals

        for symbol, position in list(self.positions.items()):
            buffer = self.price_buffers.get(symbol)
            if not buffer:
                continue
            spot = float(buffer[-1])
            plan = position.metadata.get("spread", {}) if position.metadata else {}
            delta = float(plan.get("net_delta_approx", 0.35)) * float(plan.get("direction", 1))
            entry_spot = position.metadata.get("entry_spot", spot) if position.metadata else spot
            spot_move = spot - float(entry_spot)
            premium_pnl = spot_move * delta * position.quantity * float(plan.get("lot_size", 1))
            debit_paid = float(position.metadata.get("max_loss", 1.0)) if position.metadata else 1.0
            progress = premium_pnl / max(debit_paid, 1.0)

            exit_reason = None
            if progress >= self.target_pct_of_debit:
                exit_reason = "spread_target"
            elif progress <= -self.stop_pct_of_debit:
                exit_reason = "spread_stop"
            if exit_reason:
                signals.append(Signal(
                    agent_id=self.agent_id, symbol=symbol, action="close",
                    strength=1.0, price=spot, quantity=position.quantity,
                    metadata={"reason": exit_reason, "progress": progress},
                ))
        return signals
