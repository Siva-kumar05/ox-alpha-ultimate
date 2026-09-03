"""
Crypto Perpetual Futures Agent
==============================
High-leverage perpetual futures trading on major crypto assets.
Uses funding rates, basis, order flow, and momentum.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .base import BaseAgent, AgentConfig, RiskParams, Signal, Position, AgentType
from .risk_coordinator import RiskCoordinator
from .capital_allocator import CapitalAllocator

LOG = logging.getLogger("promax.crypto_perp")


@dataclass
class CryptoPerpState:
    """State for perpetual futures per symbol."""
    symbol: str
    mark_price: float = 0.0
    index_price: float = 0.0
    funding_rate: float = 0.0
    funding_rate_8h: float = 0.0
    basis: float = 0.0
    basis_pct: float = 0.0
    open_interest: float = 0.0
    oi_change_24h: float = 0.0
    funding_apr: float = 0.0
    long_short_ratio: float = 1.0
    liquidation_price: float = 0.0
    mark_price_history: List[float] = None
    funding_history: List[float] = None


class CryptoPerpAgent(BaseAgent):
    """
    Crypto Perpetual Futures Trading Agent.
    
    Strategies:
    - Funding rate arbitrage (long spot + short perp when funding > threshold)
    - Basis trading (long spot + short perp when basis > threshold)
    - Momentum with leverage (trend following on perp)
    - Mean reversion on extreme funding/basis
    - Liquidation hunting (trade towards liquidation clusters)
    - Delta-neutral strategies
    """
    
    def __init__(
        self,
        config: AgentConfig,
        resource_pool,
        data_bus,
        risk_coordinator: RiskCoordinator,
        capital_allocator: CapitalAllocator
    ):
        super().__init__(config, resource_pool, data_bus, risk_coordinator, capital_allocator)
        
        # Strategy parameters
        self.funding_threshold = config.custom_params.get('funding_threshold', 0.0001)  # 0.01% per 8h
        self.basis_threshold = config.custom_params.get('basis_threshold', 0.005)  # 0.5%
        self.momentum_threshold = config.custom_params.get('momentum_threshold', 0.02)
        self.max_leverage = config.risk_params.max_leverage
        self.default_leverage = config.custom_params.get('default_leverage', 10.0)
        
        # Risk parameters
        self.max_position_pct = config.risk_params.max_position_pct
        self.max_daily_loss = config.risk_params.max_daily_loss_pct
        self.liquidation_buffer = config.custom_params.get('liquidation_buffer', 0.05)  # 5% buffer
        
        # Execution
        self.max_slippage_bps = config.custom_params.get('max_slippage_bps', 10)
        self.use_post_only = config.custom_params.get('use_post_only', True)
        
        # State
        self.states: Dict[str, CryptoPerpState] = {}
        self.funding_schedule: Dict[str, List[datetime]] = {}
        self.last_funding_time: Dict[str, datetime] = {}
        
        # Exchange connections (would be injected)
        self.exchange_clients: Dict[str, Any] = {}
    
    async def initialize(self) -> bool:
        try:
            for symbol in self.config.symbols:
                self.states[symbol] = CryptoPerpState(
                    symbol=symbol,
                    mark_price_history=[],
                    funding_history=[]
                )
            
            self.risk_coordinator.register_agent(self.agent_id, self.config.risk_params.__dict__)
            self.capital_allocator.register_agent(self.agent_id)
            
            # Subscribe to funding rate updates
            self.data_bus.subscribe("crypto:funding", self._on_funding_update)
            self.data_bus.subscribe("crypto:oi", self._on_oi_update)
            self.data_bus.subscribe("crypto:liquidations", self._on_liquidation_update)
            
            LOG.info(f"CryptoPerpAgent initialized for {len(self.config.symbols)} symbols with {self.max_leverage}x max leverage")
            return True
        except Exception as e:
            LOG.error(f"Failed to initialize CryptoPerpAgent: {e}")
            return False
    
    def _get_loop_interval(self) -> float:
        return 10.0  # 10-second loop for crypto
    
    async def process_market_data(self, symbol: str, data: Dict) -> List[Signal]:
        """Process perpetual futures market data."""
        if symbol not in self.config.symbols:
            return []
        
        signals = []
        
        try:
            state = self.states.get(symbol)
            if not state:
                return []
            
            # Update state from market data
            self._update_state(symbol, data)
            
            # Generate signals
            signals.extend(self._check_funding_arb(symbol, state))
            signals.extend(self._check_basis_trade(symbol, state))
            signals.extend(self._check_momentum(symbol, state))
            signals.extend(self._check_mean_reversion(symbol, state))
            signals.extend(self._check_liquidation_hunt(symbol, state))
            
        except Exception as e:
            LOG.error(f"Error processing {symbol}: {e}")
        
        return signals
    
    def _update_state(self, symbol: str, data: Dict) -> None:
        """Update state from market data."""
        state = self.states[symbol]
        
        state.mark_price = float(data.get('mark_price', data.get('last_price', 0)))
        state.index_price = float(data.get('index_price', data.get('mark_price', state.mark_price)))
        state.funding_rate = float(data.get('funding_rate', 0))
        state.funding_rate_8h = state.funding_rate * 3  # Annualize
        state.funding_apr = state.funding_rate_8h * 365 * 3
        state.open_interest = float(data.get('open_interest', 0))
        state.oi_change_24h = float(data.get('oi_change_24h', 0))
        state.long_short_ratio = float(data.get('long_short_ratio', 1.0))
        
        # Basis calculations
        state.basis = state.mark_price - state.index_price
        state.basis_pct = state.basis / state.index_price if state.index_price > 0 else 0
        
        # Track history
        state.mark_price_history.append(state.mark_price)
        state.funding_history.append(state.funding_rate)
        
        max_hist = 1000
        if len(state.mark_price_history) > max_hist:
            state.mark_price_history = state.mark_price_history[-max_hist:]
            state.funding_history = state.funding_history[-max_hist:]
        
        # Liquidation price estimation
        state.liquidation_price = self._estimate_liquidation_price(state)
    
    def _estimate_liquidation_price(self, state: CryptoPerpState) -> float:
        """Estimate liquidation price for current positions."""
        # Simplified: maintenance margin ~ 0.5% for 100x, 1% for 50x, 2.5% for 20x, 5% for 10x
        pos = self.positions.get(getattr(state, "symbol", ""))
        if pos is not None:
            if pos.side == 'long':
                return pos.entry_price * (1 - 0.9 * (1 / pos.leverage))
            else:
                return pos.entry_price * (1 + 0.9 * (1 / pos.leverage))
        return state.mark_price * 0.5  # Fallback
    
    def _check_funding_arb(self, symbol: str, state: CryptoPerpState) -> List[Signal]:
        """Check for funding rate arbitrage opportunities."""
        signals = []
        
        if symbol in self.positions:
            return signals
        
        # High positive funding = shorts pay longs
        # Strategy: Long spot + Short perp = collect funding
        if state.funding_rate > self.funding_threshold:
            # Check if basis is reasonable
            if state.basis_pct < 0.02:  # Basis not too wide
                # This would be a delta-neutral position
                # Long spot (not handled here) + Short perp
                signals.append(Signal(
                    agent_id=self.agent_id,
                    symbol=symbol,
                    action="sell",  # Short perp
                    strength=min(1.0, state.funding_rate / self.funding_threshold * 0.5),
                    price=state.mark_price,
                    quantity=0,  # Calculated in execution
                    leverage=self.default_leverage,
                    metadata={
                        "strategy": "funding_arb",
                        "funding_rate": state.funding_rate,
                        "funding_apr": state.funding_apr,
                        "basis_pct": state.basis_pct,
                        "action": "short_perp_long_spot"
                    }
                ))
        
        # Negative funding = longs pay shorts
        # Strategy: Short spot + Long perp
        elif state.funding_rate < -self.funding_threshold:
            signals.append(Signal(
                agent_id=self.agent_id,
                symbol=symbol,
                action="buy",  # Long perp
                strength=min(1.0, abs(state.funding_rate) / self.funding_threshold * 0.5),
                price=state.mark_price,
                quantity=0,
                leverage=self.default_leverage,
                metadata={
                    "strategy": "funding_arb",
                    "funding_rate": state.funding_rate,
                    "funding_apr": state.funding_apr,
                    "action": "long_perp_short_spot"
                }
            ))
        
        return []
    
    def _check_basis_trade(self, symbol: str, state: CryptoPerpState) -> List[Signal]:
        """Check for basis trading opportunities."""
        signals = []
        
        if symbol in self.positions:
            return signals
        
        # Wide positive basis = perp rich, spot cheap
        if state.basis_pct > self.basis_threshold:
            # Short perp, long spot
            signals.append(Signal(
                agent_id=self.agent_id,
                symbol=symbol,
                action="sell",
                strength=min(1.0, state.basis_pct / self.basis_threshold * 0.5),
                price=state.mark_price,
                quantity=0,
                leverage=self.default_leverage,
                metadata={
                    "strategy": "basis_trade",
                    "basis_pct": state.basis_pct,
                    "basis": state.basis,
                    "action": "short_perp_long_spot"
                }
            ))
        
        # Negative basis (backwardation) - rare but possible
        elif state.basis_pct < -self.basis_threshold:
            signals.append(Signal(
                agent_id=self.agent_id,
                symbol=symbol,
                action="buy",
                strength=min(1.0, abs(state.basis_pct) / self.basis_threshold * 0.5),
                price=state.mark_price,
                quantity=0,
                leverage=self.default_leverage,
                metadata={
                    "strategy": "basis_trade_reverse",
                    "basis_pct": state.basis_pct,
                    "action": "long_perp_short_spot"
                }
            ))
        
        return []
    
    def _check_momentum(self, symbol: str, state: CryptoPerpState) -> List[Signal]:
        """Momentum trading with leverage."""
        signals = []
        
        if symbol in self.positions:
            return []
        
        # Need price history
        if len(state.mark_price_history) < 100:
            return []
        
        prices = np.array(state.mark_price_history[-100:])
        
        # Calculate momentum
        mom_20 = (prices[-1] / prices[-20] - 1) * 100
        mom_50 = (prices[-1] / prices[-50] - 1) * 100
        
        # RSI
        rsi = self._calc_rsi(np.array(state.mark_price_history[-50:]))
        
        # Volume/OI confirmation
        oi_increasing = True  # Would check OI change
        
        # Long momentum
        if mom_20 > 3 and rsi < 70:
            signals.append(Signal(
                agent_id=self.agent_id,
                symbol=symbol,
                action="buy",
                strength=min(1.0, mom_20 / 10),
                price=state.mark_price,
                quantity=0,
                leverage=min(self.max_leverage, 5.0),
                metadata={
                    "strategy": "momentum",
                    "momentum_20": mom_20,
                    "rsi": rsi,
                    "leverage": 5.0
                }
            ))
        
        # Short momentum (if allowed)
        elif mom_20 < -3 and rsi > 30:
            signals.append(Signal(
                agent_id=self.agent_id,
                symbol=symbol,
                action="sell",
                strength=min(1.0, abs(mom_20) / 10),
                price=state.mark_price,
                quantity=0,
                leverage=min(self.max_leverage, 3.0),
                metadata={
                    "strategy": "momentum_short",
                    "momentum_20": mom_20,
                    "rsi": rsi,
                    "leverage": 3.0
                }
            ))
        
        return []
    
    def _check_mean_reversion(self, symbol: str, state: CryptoPerpState) -> List[Signal]:
        """Mean reversion on extreme funding/basis."""
        signals = []
        
        if symbol in self.positions:
            return []
        
        # Extreme funding mean reversion
        if state.funding_rate > 0.001:  # 0.1% per 8h = extremely high
            # Expect funding to revert, short perp
            signals.append(Signal(
                agent_id=self.agent_id,
                symbol=symbol,
                action="sell",
                strength=0.9,
                price=state.mark_price,
                quantity=0,
                leverage=min(self.max_leverage, 3.0),
                metadata={
                    "strategy": "funding_mean_reversion",
                    "funding_rate": state.funding_rate,
                    "reason": "extreme_positive_funding"
                }
            ))
        elif state.funding_rate < -0.001:
            signals.append(Signal(
                agent_id=self.agent_id,
                symbol=symbol,
                action="buy",
                strength=0.9,
                price=state.mark_price,
                quantity=0,
                leverage=min(self.max_leverage, 3.0),
                metadata={
                    "strategy": "funding_mean_reversion",
                    "funding_rate": state.funding_rate,
                    "reason": "extreme_negative_funding"
                }
            ))
        
        return []
    
    def _check_liquidation_hunt(self, symbol: str, state: CryptoPerpState) -> List[Signal]:
        """Trade towards liquidation clusters."""
        signals = []
        
        # Would need liquidation data from exchange
        # Simplified: if OI dropping fast + price moving, hunt direction
        oi_change = state.oi_change_24h
        price_change = 0
        
        if len(state.mark_price_history) > 2:
            price_change = (state.mark_price_history[-1] / state.mark_price_history[-2] - 1) * 100
        
        # OI dropping + price moving = liquidations
        if state.oi_change_24h < -10 and abs(price_change) > 2:
            if price_change > 0:
                # Long liquidations, potential bounce
                signals.append(Signal(
                    agent_id=self.agent_id,
                    symbol=symbol,
                    action="buy",
                    strength=0.7,
                    price=state.mark_price,
                    quantity=0,
                    leverage=min(self.max_leverage, 3.0),
                    metadata={
                        "strategy": "liquidation_hunt",
                        "oi_change": state.oi_change_24h,
                        "price_change": price_change,
                        "direction": "long_liquidations"
                    }
                ))
            else:
                # Short liquidations, potential bounce down
                signals.append(Signal(
                    agent_id=self.agent_id,
                    symbol=symbol,
                    action="sell",
                    strength=0.7,
                    price=state.mark_price,
                    quantity=0,
                    leverage=min(self.max_leverage, 3.0),
                    metadata={
                        "strategy": "liquidation_hunt",
                        "oi_change": state.oi_change_24h,
                        "direction": "short_liquidations"
                    }
                ))
        
        return []
    
    def _calc_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        if len(prices) < period + 1:
            return 50.0
        deltas = np.diff(prices[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    def _can_open_position(self, symbol: str) -> bool:
        if len(self.positions) >= self.config.risk_params.max_concurrent_positions:
            return False
        
        capital = self.capital_allocator.get_allocation(self.agent_id)
        used = sum(p.quantity * p.current_price for p in self.positions.values())
        if used > self.capital_allocator.get_allocation(self.agent_id) * 0.8:
            return False
        
        return True
    
    async def manage_positions(self) -> List[Signal]:
        """Manage crypto perp positions."""
        signals = []
        
        for symbol, position in list(self.positions.items()):
            state = self.states.get(symbol)
            if not state:
                continue
            
            current_price = state.mark_price
            entry_price = position.entry_price
            leverage = position.leverage
            
            # Liquidation check
            liq_price = state.liquidation_price
            if position.side == 'long' and state.mark_price < liq_price * 1.1:
                # Close to liquidation, reduce or close
                signals.append(Signal(
                    agent_id=self.agent_id,
                    symbol=symbol,
                    action="close",
                    strength=1.0,
                    price=state.mark_price,
                    quantity=position.quantity,
                    metadata={"reason": "near_liquidation", "liq_price": liq_price}
                ))
            elif position.side == 'short' and state.mark_price > liq_price * 0.9:
                signals.append(Signal(
                    agent_id=self.agent_id,
                    symbol=symbol,
                    action="close",
                    strength=1.0,
                    price=state.mark_price,
                    quantity=position.quantity,
                    metadata={"reason": "near_liquidation"}
                ))
            
            # Funding cost check for long-held positions
            hold_hours = (datetime.now() - position.entry_time).total_seconds() / 3600
            if hold_hours > 24:
                # Check if funding costs exceed profits
                signals.append(Signal(
                    agent_id=self.agent_id,
                    symbol=symbol,
                    action="close",
                    strength=0.5,
                    price=state.mark_price,
                    quantity=position.quantity,
                    metadata={"reason": "funding_cost_review", "hold_hours": hold_hours}
                ))
        
        return []
    
    def _on_funding_update(self, data: Dict) -> None:
        symbol = data.get('symbol')
        if symbol in self.states:
            self.states[symbol].funding_rate = float(data.get('funding_rate', 0))
    
    def _on_oi_update(self, data: Dict) -> None:
        symbol = data.get('symbol')
        if symbol in self.states:
            self.states[symbol].open_interest = float(data.get('open_interest', 0))
            self.states[symbol].oi_change_24h = float(data.get('oi_change_24h', 0))
    
    def _on_liquidation_update(self, data: Dict) -> None:
        # Process liquidation data for hunting
        pass