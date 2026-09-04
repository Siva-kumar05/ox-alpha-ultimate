"""
Intraday Scalper Agent
======================
High-frequency intraday scalping on liquid large-cap stocks.
Uses order flow, VWAP deviation, and micro-structure patterns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from .base import BaseAgent, AgentConfig, Signal

LOG = logging.getLogger("promax.intraday_scalper")


@dataclass
class ScalperState:
    """State for scalping per symbol."""
    symbol: str
    vwap: float = 0.0
    vwap_upper: float = 0.0
    vwap_lower: float = 0.0
    vwap_dev: float = 0.0
    order_flow_imbalance: float = 0.0
    cumulative_delta: float = 0.0
    last_trade_price: float = 0.0
    last_trade_size: int = 0
    vwap_initialized: bool = False
    session_high: float = 0.0
    session_low: float = 0.0


class IntradayScalperAgent(BaseAgent):
    """
    High-frequency intraday scalper.
    
    Strategy:
    - VWAP mean reversion with bands
    - Order flow imbalance detection
    - Cumulative volume delta (CVD) tracking
    - Opening range breakout/fade
    - Micro-structure patterns (absorption, icebergs)
    - Tight stops, quick targets
    """
    
    def __init__(
        self,
        config: AgentConfig,
        resource_pool,
        data_bus,
        risk_coordinator,
        capital_allocator
    ):
        super().__init__(config, resource_pool, data_bus, risk_coordinator, capital_allocator)
        
        # Scalper parameters
        self.vwap_band_mult = config.custom_params.get('vwap_band_mult', 1.5)
        self.min_flow_imbalance = config.custom_params.get('min_flow_imbalance', 0.3)
        self.min_delta = config.custom_params.get('min_delta', 1000)
        self.tick_size = config.custom_params.get('tick_size', 0.05)
        
        # Execution parameters
        self.target_ticks = config.custom_params.get('target_ticks', 8)
        self.stop_ticks = config.custom_params.get('stop_ticks', 4)
        self.max_hold_minutes = config.custom_params.get('max_hold_minutes', 15)
        
        # Session parameters
        self.or_minutes = config.custom_params.get('or_minutes', 15)
        self.fade_or = config.custom_params.get('fade_or', True)
        
        # State
        self.states: Dict[str, ScalperState] = {}
        self.tick_buffers: Dict[str, List[Dict]] = {}
        self.vwap_numerator: Dict[str, float] = {}
        self.vwap_denominator: Dict[str, float] = {}
        
        # Order flow tracking
        self.cvd: Dict[str, float] = {}
        self.last_trade_side: Dict[str, int] = {}
    
    async def initialize(self) -> bool:
        try:
            for symbol in self.config.symbols:
                self.states[symbol] = ScalperState(symbol=symbol)
                self.tick_buffers[symbol] = []
                self.vwap_numerator[symbol] = 0.0
                self.vwap_denominator[symbol] = 0.0
                self.cvd[symbol] = 0.0
                self.last_trade_side[symbol] = 0
            
            self.risk_coordinator.register_agent(self.agent_id, self.config.risk_params.__dict__)
            self.capital_allocator.register_agent(self.agent_id)
            
            LOG.info(f"IntradayScalperAgent initialized for {len(self.config.symbols)} symbols")
            return True
        except Exception as e:
            LOG.error(f"Failed to initialize IntradayScalperAgent: {e}")
            return False
    
    def _get_loop_interval(self) -> float:
        return 5.0  # 5-second loop for scalping
    
    async def process_market_data(self, symbol: str, data: Dict) -> List[Signal]:
        """Process tick data for scalping signals."""
        if symbol not in self.config.symbols:
            return []
        
        signals = []
        
        try:
            # Extract tick data
            price = float(data.get('last_price', data.get('close', 0)))
            volume = int(data.get('volume', data.get('size', 0)))
            side = int(data.get('side', data.get('aggressor', 0)))  # +1 buy, -1 sell
            
            if price <= 0 or volume <= 0:
                return []
            
            state = self.states[symbol]
            
            # Update VWAP
            self._update_vwap(symbol, price, volume)
            
            # Update CVD
            self._update_cvd(symbol, price, volume, side)
            
            # Update order flow imbalance
            self._update_flow_imbalance(symbol, side, volume)
            
            # Update session high/low
            state.session_high = max(state.session_high, price)
            state.session_low = min(state.session_low, price) if state.session_low > 0 else price
            
            # Track opening range
            self._update_opening_range(symbol, price, volume)
            
            # Generate signals
            entry_signal = self._check_scalp_entry(symbol, state, price, volume, side)
            if entry_signal:
                signals.append(entry_signal)
            
            state.last_trade_price = price
            state.last_trade_size = volume
            
        except Exception as e:
            LOG.error(f"Error processing {symbol}: {e}")
        
        return signals
    
    def _update_vwap(self, symbol: str, price: float, volume: int) -> None:
        """Update running VWAP."""
        self.vwap_numerator[symbol] += price * volume
        self.vwap_denominator[symbol] += volume
        
        state = self.states[symbol]
        if self.vwap_denominator[symbol] > 0:
            state.vwap = self.vwap_numerator[symbol] / self.vwap_denominator[symbol]
            
            # Calculate VWAP bands
            # Simplified: use rolling std dev of price around VWAP
            if hasattr(self, f'_price_history_{symbol}'):
                prices = getattr(self, f'_price_history_{symbol}')
                prices.append(price)
                if len(prices) > 20:
                    prices.pop(0)
                    std = np.std(prices)
                    state.vwap_upper = state.vwap + std * self.vwap_band_mult
                    state.vwap_lower = state.vwap - std * self.vwap_band_mult
                    state.vwap_dev = (price - state.vwap) / (std + 1e-10)
            else:
                setattr(self, f'_price_history_{symbol}', [price])
                state.vwap_upper = state.vwap * 1.002
                state.vwap_lower = state.vwap * 0.998
                state.vwap_dev = 0
        
        state.vwap_initialized = True
    
    def _update_cvd(self, symbol: str, price: float, volume: int, side: int) -> None:
        """Update Cumulative Volume Delta."""
        if side > 0:
            self.cvd[symbol] += volume
        elif side < 0:
            self.cvd[symbol] -= volume
        
        state = self.states[symbol]
        state.cumulative_delta = self.cvd[symbol]
        self.last_trade_side[symbol] = side
    
    def _update_flow_imbalance(self, symbol: str, side: int, volume: int) -> None:
        """Update order flow imbalance using rolling window."""
        state = self.states[symbol]
        
        # Simple exponential moving average of flow
        alpha = 0.1
        flow = volume if side > 0 else -volume
        state.order_flow_imbalance = alpha * flow + (1 - alpha) * state.order_flow_imbalance
    
    def _update_opening_range(self, symbol: str, price: float, volume: int) -> None:
        """Track opening range high/low."""
        state = self.states[symbol]
        now = datetime.now()
        session_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
        minutes_elapsed = (now - session_start).total_seconds() / 60
        
        if minutes_elapsed <= self.or_minutes:
            state.session_high = max(state.session_high, price)
            state.session_low = min(state.session_low, price) if state.session_low > 0 else price
    
    def _check_scalp_entry(self, symbol: str, state: ScalperState, 
                          price: float, volume: int, side: int) -> Optional[Signal]:
        """Check for scalping entry signals."""
        # Already have position?
        if symbol in self.positions:
            return None
        
        if not state.vwap_initialized:
            return None
        
        # Check risk limits
        if not self._can_open_position(symbol):
            return None
        
        # Check if we're in opening range
        now = datetime.now()
        session_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
        minutes_elapsed = (now - session_start).total_seconds() / 60
        
        signals = []
        
        # 1. VWAP Mean Reversion
        if state.vwap_dev < -1.5 and state.order_flow_imbalance > 0.2:
            signals.append(("vwap_reversion_long", 0.8))
        elif state.vwap_dev > 1.5 and state.order_flow_imbalance < -0.2:
            signals.append(("vwap_reversion_short", 0.8))
        
        # 2. Opening Range Breakout/Fade
        if minutes_elapsed == self.or_minutes and state.session_high > 0:
            or_high = state.session_high
            or_low = state.session_low
            or_range = or_high - or_low
            
            if price > or_high + 0.1 * or_range:
                signals.append(("or_breakout_long", 0.9))
            elif price < or_low - 0.1 * or_range:
                signals.append(("or_breakout_short", 0.9))
            elif self.fade_or and price > or_high - 0.2 * or_range and price < or_high:
                signals.append(("or_fade_short", 0.7))
            elif self.fade_or and price < or_low + 0.2 * or_range and price > or_low:
                signals.append(("or_fade_long", 0.7))
        
        # 3. CVD Divergence
        if state.cumulative_delta > self.min_delta and price < state.vwap:
            signals.append(("cvd_long", 0.75))
        elif state.cumulative_delta < -self.min_delta and price > state.vwap:
            signals.append(("cvd_short", 0.75))
        
        # Select best signal
        if not signals:
            return None
        
        signals.sort(key=lambda x: x[1], reverse=True)
        best_signal_type, strength = signals[0]
        
        # Determine action
        action = "buy" if "long" in best_signal_type else "sell"
        
        # Check if we can trade this direction
        if action == "sell" and self.config.risk_params.max_leverage == 1.0:
            # Long-only mode
            if "short" in best_signal_type:
                return None
        
        # Calculate position
        capital = self.capital_allocator.get_allocation(self.agent_id)
        risk_per_trade = capital * 0.005  # 0.5% risk per scalp
        stop_distance = self.stop_ticks * self.tick_size
        quantity = int(risk_per_trade / stop_distance)
        
        if quantity <= 0:
            return None
        
        if action == "buy":
            stop_loss = price - stop_distance
            take_profit = price + self.target_ticks * self.tick_size
        else:
            stop_loss = price + stop_distance
            take_profit = price - self.target_ticks * self.tick_size
        
        return Signal(
            agent_id=self.agent_id,
            symbol=symbol,
            action=action,
            strength=strength,
            price=price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            leverage=self.config.risk_params.max_leverage,
            metadata={
                "strategy": "scalp",
                "signal_type": best_signal_type,
                "vwap_dev": state.vwap_dev,
                "flow_imbalance": state.order_flow_imbalance,
                "cvd": state.cumulative_delta
            }
        )
    
    def _can_open_position(self, symbol: str) -> bool:
        if len(self.positions) >= self.config.risk_params.max_concurrent_positions:
            return False
        
        capital = self.capital_allocator.get_allocation(self.agent_id)
        used = sum(p.quantity * p.current_price for p in self.positions.values())
        if used > capital * 0.8:
            return False
        
        return True
    
    async def manage_positions(self) -> List[Signal]:
        """Manage scalping positions - very tight management."""
        signals = []
        
        for symbol, position in list(self.positions.items()):
            state = self.states.get(symbol)
            if not state or not state.vwap_initialized:
                continue
            
            current_price = position.current_price
            entry_price = position.entry_price
            hold_minutes = (datetime.now() - position.entry_time).total_seconds() / 60
            
            # Quick target hit
            if position.side == "long":
                if current_price >= position.take_profit:
                    signals.append(Signal(
                        agent_id=self.agent_id, symbol=symbol, action="close",
                        strength=1.0, price=current_price, quantity=position.quantity,
                        metadata={"reason": "target_hit"}
                    ))
                elif current_price <= position.stop_loss:
                    signals.append(Signal(
                        agent_id=self.agent_id, symbol=symbol, action="close",
                        strength=1.0, price=current_price, quantity=position.quantity,
                        metadata={"reason": "stop_hit"}
                    ))
                # Trail stop
                elif current_price > entry_price + 4 * self.tick_size:
                    new_stop = current_price - 3 * self.tick_size
                    if new_stop > position.stop_loss:
                        signals.append(Signal(
                            agent_id=self.agent_id, symbol=symbol, action="modify",
                            strength=0.5, price=current_price, quantity=position.quantity,
                            stop_loss=new_stop, metadata={"reason": "trail_stop"}
                        ))
            else:  # short
                if current_price <= position.take_profit:
                    signals.append(Signal(
                        agent_id=self.agent_id, symbol=symbol, action="close",
                        strength=1.0, price=current_price, quantity=position.quantity,
                        metadata={"reason": "target_hit"}
                    ))
                elif current_price >= position.stop_loss:
                    signals.append(Signal(
                        agent_id=self.agent_id, symbol=symbol, action="close",
                        strength=1.0, price=current_price, quantity=position.quantity,
                        metadata={"reason": "stop_hit"}
                    ))
                elif current_price < entry_price - 4 * self.tick_size:
                    new_stop = current_price + 3 * self.tick_size
                    if new_stop < position.stop_loss:
                        signals.append(Signal(
                            agent_id=self.agent_id, symbol=symbol, action="modify",
                            strength=0.5, price=current_price, quantity=position.quantity,
                            stop_loss=new_stop, metadata={"reason": "trail_stop"}
                        ))
            
            # Time stop
            if hold_minutes >= self.max_hold_minutes:
                signals.append(Signal(
                    agent_id=self.agent_id, symbol=symbol, action="close",
                    strength=1.0, price=current_price, quantity=position.quantity,
                    metadata={"reason": "time_stop", "hold_minutes": hold_minutes}
                ))
        
        return signals