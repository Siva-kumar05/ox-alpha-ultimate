"""
Market Maker Agent — simulated two-sided quoting with inventory skew.
=====================================================================

Mandate: earn the spread instead of paying it.  This is a *paper* market
maker for retail-scale capital: it quotes both sides around a fair-value
EWMA, tilts quotes against inventory, captures the spread when the mid
crosses its quotes, and forces itself flat by end of session.

Quote fills are approval-exempt by design (``approval_exempt`` metadata):
every buy is one half of a two-sided quote, inventory is capped at a couple
of units, and exposure never exceeds a small fraction of the agent budget.
Set ``approval_required: true`` in the agent config to route maker buys
through the human gateway anyway (each fill will then wait for approval).

When the compiled C++ kernel (cpp/) is available it replaces the Python
quoting loop for latency; the strategy interface here is the contract.
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

LOG = logging.getLogger("promax.mm")

IST = ZoneInfo("Asia/Kolkata")


class MarketMakerAgent(BaseAgent):
    """Inventory-capped two-sided quoter (paper fills on mid crossing)."""

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
        self.half_spread_bps = params.get("half_spread_bps", 8.0)
        self.fair_ewma_alpha = params.get("fair_ewma_alpha", 0.2)
        self.max_inventory = params.get("max_inventory", 2)
        self.unit_notional = params.get("unit_notional", 0.05)  # fraction of budget per unit
        self.vol_half_spread_mult = params.get("vol_half_spread_mult", 0.5)
        self.squareoff = params.get("squareoff", "15:15")

        self.fair: Dict[str, float] = {}
        self.returns: Dict[str, deque] = {s: deque(maxlen=120) for s in config.symbols}
        self.last_mid: Dict[str, float] = {}
        self.inventory: Dict[str, int] = {s: 0 for s in config.symbols}
        self.cash_pnl: Dict[str, float] = {s: 0.0 for s in config.symbols}
        self.open_quotes: Dict[str, Dict[str, float]] = {}

    # ── lifecycle ─────────────────────────────────────────────────────
    async def initialize(self) -> bool:
        self.risk_coordinator.register_agent(self.agent_id, self.config.risk_params.__dict__)
        self.capital_allocator.register_agent(self.agent_id)
        LOG.info(f"MarketMakerAgent initialized on {self.config.symbols}")
        return True

    def _get_loop_interval(self) -> float:
        return 2.0

    # ── quoting ───────────────────────────────────────────────────────
    async def process_market_data(self, symbol: str, data: Dict) -> List[Signal]:
        mid = float(data.get("price", 0.0))
        if mid <= 0:
            return []

        signals: List[Signal] = []
        previous_mid = self.last_mid.get(symbol)
        self.last_mid[symbol] = mid

        # Fair value EWMA + realized vol for the spread width.
        if symbol not in self.fair:
            self.fair[symbol] = mid
        self.fair[symbol] += self.fair_ewma_alpha * (mid - self.fair[symbol])
        if previous_mid:
            self.returns[symbol].append(np.log(mid / previous_mid))
        vols = np.asarray(self.returns[symbol], dtype=float)
        vol = float(np.std(vols)) if len(vols) >= 20 else 0.0005

        fair = self.fair[symbol]
        half = (self.half_spread_bps / 1e4) * fair + self.vol_half_spread_mult * vol * fair
        skew = self.inventory.get(symbol, 0) / max(1, self.max_inventory)
        bid = fair - half * (1.0 + skew)
        ask = fair + half * (1.0 - skew)
        self.open_quotes[symbol] = {"bid": bid, "ask": ask, "fair": fair}

        # Paper fills: the mid traded through our quotes.
        if previous_mid is not None:
            if previous_mid <= bid < mid or mid <= bid:
                signals.extend(self._fill(symbol, "buy", bid, mid))
            if previous_mid >= ask > mid or mid >= ask:
                signals.extend(self._fill(symbol, "sell", ask, mid))
        return signals

    def _fill(self, symbol: str, side: str, quote: float, mid: float) -> List[Signal]:
        signals: List[Signal] = []
        budget = self.capital_allocator.budget(self.agent_id)
        unit_value = budget * self.unit_notional
        quantity = max(1.0, unit_value / mid)

        if side == "buy":
            if self.inventory.get(symbol, 0) >= self.max_inventory:
                return []  # inventory cap: never accumulate direction
            self.inventory[symbol] = self.inventory.get(symbol, 0) + 1
            self.cash_pnl[symbol] -= quantity * quote
            signals.append(Signal(
                agent_id=self.agent_id, symbol=symbol, action="buy",
                strength=0.4, price=quote, quantity=quantity,
                leverage=1.0,
                metadata={
                    "reason": "maker_bid_fill",
                    "approval_exempt": True,  # two-sided quote, inventory-capped
                    "inventory": self.inventory[symbol],
                },
            ))
        else:
            if self.inventory.get(symbol, 0) <= 0:
                return []  # never naked short in paper equity mode
            self.inventory[symbol] = self.inventory.get(symbol, 0) - 1
            self.cash_pnl[symbol] += quantity * quote
            signals.append(Signal(
                agent_id=self.agent_id, symbol=symbol, action="close",
                strength=1.0, price=quote, quantity=quantity,
                metadata={"reason": "maker_ask_fill", "inventory": self.inventory[symbol]},
            ))
        return signals

    # ── position management ───────────────────────────────────────────
    async def manage_positions(self) -> List[Signal]:
        now = datetime.now(IST)
        sh, sm = (int(x) for x in self.squareoff.split(":"))
        eod = now.hour * 60 + now.minute >= sh * 60 + sm
        signals: List[Signal] = []
        for symbol, position in list(self.positions.items()):
            mid = self.last_mid.get(symbol, position.current_price)
            position.current_price = mid
            if eod and self.inventory.get(symbol, 0) > 0:
                signals.append(Signal(
                    agent_id=self.agent_id, symbol=symbol, action="close",
                    strength=1.0, price=mid, quantity=position.quantity,
                    metadata={"reason": "mm_squareoff"},
                ))
        return signals

    def get_status(self) -> Dict[str, Any]:
        status = super().get_status()
        status["inventory"] = dict(self.inventory)
        status["quotes"] = {k: dict(v) for k, v in self.open_quotes.items()}
        status["cash_pnl"] = dict(self.cash_pnl)
        return status
