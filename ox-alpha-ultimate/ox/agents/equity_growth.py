"""
Equity Growth Agent — short-term and long-term high-growth stocks.
==================================================================

Mandate: find and hold growth leaders instead of churning large-cap index
names.  Two horizons run from the same engine (custom_params.horizon):

* "short" (days): momentum + volume confirmation, 3-4% stops, trails fast.
* "long"  (weeks): broader lookback, wide 8% stops, small size, patience.

Conviction inputs, all computed from the data-bus feed:
  * multi-window price momentum vs the symbol's own volatility,
  * trend quality (price above its running mean and mean rising),
  * volume confirmation (volume z-score when the feed supplies volume),
  * news optimism from the news-intel agent (bus topic ``news:<sym>``);
    strongly negative news vetoes a new entry (never creates one).

Every entry is capital-deploying and therefore parks in the human approval
gateway; exits are autonomous and immediate.
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from .base import AgentConfig, BaseAgent, Position, Signal
from .capital_allocator import CapitalAllocator
from .risk_coordinator import RiskCoordinator

LOG = logging.getLogger("promax.growth")


class EquityGrowthAgent(BaseAgent):
    """Growth-stock specialist with short/long horizons."""

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
        self.horizon: str = params.get("horizon", "short")
        if self.horizon == "long":
            self.momentum_windows = params.get("momentum_windows", [120, 360])
            self.stop_pct = params.get("stop_pct", 0.08)
            self.take_profit_pct = params.get("take_profit_pct", 0.25)
            self.max_hold_minutes = params.get("max_hold_minutes", 60 * 24 * 30)
            self.min_momentum = params.get("min_momentum", 0.03)
        else:
            self.momentum_windows = params.get("momentum_windows", [30, 90])
            self.stop_pct = params.get("stop_pct", 0.035)
            self.take_profit_pct = params.get("take_profit_pct", 0.09)
            self.max_hold_minutes = params.get("max_hold_minutes", 60 * 24 * 5)
            self.min_momentum = params.get("min_momentum", 0.012)

        self.trend_window = params.get("trend_window", 60)
        self.volume_window = params.get("volume_window", 30)
        self.min_volume_z = params.get("min_volume_z", 0.8)
        self.news_veto_below = params.get("news_veto_below", -0.3)
        self.risk_per_trade = params.get("risk_per_trade", 0.08)

        self.price_buffers: Dict[str, deque] = {s: deque(maxlen=600) for s in config.symbols}
        self.volume_buffers: Dict[str, deque] = {s: deque(maxlen=120) for s in config.symbols}
        self.news_scores: Dict[str, float] = {}
        self.data_bus.subscribe("news:sentiment", self._on_news_sentiment)

    # ── lifecycle ─────────────────────────────────────────────────────
    async def initialize(self) -> bool:
        self.risk_coordinator.register_agent(self.agent_id, self.config.risk_params.__dict__)
        self.capital_allocator.register_agent(self.agent_id)
        LOG.info(f"EquityGrowthAgent ({self.horizon}-horizon) initialized for "
                 f"{len(self.config.symbols)} symbols")
        return True

    def _get_loop_interval(self) -> float:
        return 20.0 if self.horizon == "short" else 60.0

    # ── data ──────────────────────────────────────────────────────────
    def _on_news_sentiment(self, payload: Dict[str, Any]) -> None:
        symbol = str(payload.get("symbol", ""))
        if symbol in self.price_buffers:
            self.news_scores[symbol] = float(payload.get("avg_score", 0.0))

    async def process_market_data(self, symbol: str, data: Dict) -> List[Signal]:
        price = float(data.get("price", 0.0))
        if price <= 0:
            return []
        self.price_buffers[symbol].append(price)
        if data.get("volume") is not None:
            self.volume_buffers[symbol].append(float(data["volume"]))

        signals: List[Signal] = []
        if symbol in self.positions:
            return signals  # one position per symbol; exits handled in manage_positions
        if len(self.price_buffers[symbol]) < max(self.momentum_windows) + 2:
            return signals

        # News veto: fresh strongly-negative coverage blocks a new entry.
        news_score = self.news_scores.get(symbol, 0.0)
        if news_score < self.news_veto_below:
            return signals

        conviction = self._growth_conviction(symbol)
        if conviction is None or conviction < 1.0:
            return signals

        quantity = self._position_size(symbol, price)
        if quantity <= 0:
            return signals

        stop = price * (1 - self.stop_pct)
        target = price * (1 + self.take_profit_pct)
        signals.append(Signal(
            agent_id=self.agent_id, symbol=symbol, action="buy",
            strength=min(1.5, conviction) / 1.5, price=price, quantity=quantity,
            stop_loss=stop, take_profit=target, leverage=1.0,
            metadata={
                "reason": f"growth_{self.horizon}_conviction_{conviction:.2f}",
                "news_score": news_score,
                "approval_ttl_hint": 86400 if self.horizon == "long" else 3600,
            },
        ))
        return signals

    # ── conviction model ──────────────────────────────────────────────
    def _growth_conviction(self, symbol: str) -> Optional[float]:
        prices = np.asarray(self.price_buffers[symbol], dtype=float)
        score = 0.0
        hits = 0
        for window in self.momentum_windows:
            if len(prices) <= window:
                continue
            momentum = prices[-1] / prices[-1 - window] - 1.0
            if momentum >= self.min_momentum:
                score += 1.0
            elif momentum <= 0:
                score -= 1.0
            hits += 1

        # Trend quality: above rising mean.
        trend_window = min(self.trend_window, len(prices) - 1)
        mean = float(np.mean(prices[-trend_window:]))
        prev_mean = float(np.mean(prices[-trend_window * 2:-trend_window])) if len(prices) >= trend_window * 2 else mean
        if prices[-1] > mean and mean >= prev_mean:
            score += 1.0
            hits += 1
        elif prices[-1] < mean:
            score -= 0.5
            hits += 1

        # Volume confirmation when the feed supplies it.
        volumes = self.volume_buffers.get(symbol)
        if volumes and len(volumes) >= self.volume_window:
            vols = np.asarray(volumes, dtype=float)
            z = (vols[-1] - vols.mean()) / (vols.std() + 1e-9)
            if z >= self.min_volume_z:
                score += 0.5
                hits += 1

        # Fresh positive news nudges conviction up (never down-vetoed here;
        # the veto already ran in process_market_data).
        news_score = self.news_scores.get(symbol, 0.0)
        if news_score > 0.2:
            score += 0.5
            hits += 1

        if hits == 0:
            return None
        return score / hits * 2.0  # conviction in ~[-2, 2]; >=1 means trade

    def _position_size(self, symbol: str, price: float) -> float:
        budget = self.capital_allocator.budget(self.agent_id)
        allocation = budget * self.risk_per_trade
        if price <= 0 or allocation <= 0:
            return 0.0
        return max(0.0, allocation / price)

    # ── position management ───────────────────────────────────────────
    async def manage_positions(self) -> List[Signal]:
        signals: List[Signal] = []
        for symbol, position in list(self.positions.items()):
            prices = self.price_buffers.get(symbol)
            if prices:
                position.current_price = float(prices[-1])
            pnl_pct = position.current_price / position.entry_price - 1.0
            hold_minutes = (datetime.now() - position.entry_time).total_seconds() / 60

            exit_reason = None
            if pnl_pct >= self.take_profit_pct:
                exit_reason = "target_hit"
            elif pnl_pct <= -self.stop_pct:
                exit_reason = "stop_hit"
            elif hold_minutes >= self.max_hold_minutes:
                exit_reason = "time_stop"
            elif self.horizon == "short" and pnl_pct > 0.04:
                # Trail: hand back only what a wide-breath trade needs.
                position.stop_loss = max(position.stop_loss or 0.0, position.current_price * (1 - self.stop_pct / 2))
            elif self.news_scores.get(symbol, 0.0) < -0.6:
                exit_reason = "news_deteriorated"

            if exit_reason:
                signals.append(Signal(
                    agent_id=self.agent_id, symbol=symbol, action="close",
                    strength=1.0, price=position.current_price,
                    quantity=position.quantity,
                    metadata={"reason": exit_reason, "pnl_pct": pnl_pct},
                ))
        return signals
