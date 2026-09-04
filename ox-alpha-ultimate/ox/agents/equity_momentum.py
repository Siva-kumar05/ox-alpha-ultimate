"""
Equity Momentum Agent
=====================
High-growth momentum trading on liquid large-cap equities.
Uses multi-timeframe momentum, relative strength, and volume confirmation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from .base import BaseAgent, AgentConfig, Signal
from .risk_coordinator import RiskCoordinator
from .capital_allocator import CapitalAllocator

LOG = logging.getLogger("promax.equity_momentum")


@dataclass
class MomentumState:
    """State for momentum tracking per symbol."""
    symbol: str
    momentum_20: float = 0.0
    momentum_50: float = 0.0
    momentum_100: float = 0.0
    rsi_14: float = 50.0
    volume_ratio: float = 1.0
    rs_rank: int = 0
    trend_strength: float = 0.0
    last_update: datetime = None


class EquityMomentumAgent(BaseAgent):
    """
    Momentum trading agent for large-cap equities.
    
    Strategy:
    - Multi-timeframe momentum (20, 50, 100 periods)
    - Relative strength vs sector/benchmark
    - Volume confirmation on breakouts
    - RSI filter for overbought/oversold
    - Trailing stops with ATR
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
        
        # Momentum tracking
        self.momentum_states: Dict[str, MomentumState] = {}
        self.lookback_periods = config.custom_params.get('lookback_periods', [20, 50, 100])
        self.rsi_period = config.custom_params.get('rsi_period', 14)
        self.volume_lookback = config.custom_params.get('volume_lookback', 20)
        self.rs_lookback = config.custom_params.get('rs_lookback', 63)
        
        # Signal thresholds
        self.momentum_threshold = config.custom_params.get('momentum_threshold', 0.02)
        self.rs_threshold = config.custom_params.get('rs_threshold', 0.6)  # Top 40%
        self.rsi_oversold = config.custom_params.get('rsi_oversold', 35)
        self.rsi_overbought = config.custom_params.get('rsi_overbought', 70)
        self.min_volume_ratio = config.custom_params.get('min_volume_ratio', 1.2)
        
        # Position management
        self.trailing_atr_mult = config.custom_params.get('trailing_atr_mult', 2.5)
        self.max_hold_days = config.custom_params.get('max_hold_days', 10)
        
        # Data buffers
        self.price_buffers: Dict[str, List[float]] = {}
        self.volume_buffers: Dict[str, List[float]] = {}
        self.high_buffers: Dict[str, List[float]] = {}
        self.low_buffers: Dict[str, List[float]] = {}
        
        # Benchmark for relative strength
        self.benchmark_symbol = config.custom_params.get('benchmark', 'NIFTY50')
        self.benchmark_buffer: List[float] = []
    
    async def initialize(self) -> bool:
        """Initialize momentum agent."""
        try:
            # Initialize buffers for each symbol
            for symbol in self.config.symbols:
                self.momentum_states[symbol] = MomentumState(symbol=symbol)
                self.price_buffers[symbol] = []
                self.volume_buffers[symbol] = []
                self.high_buffers[symbol] = []
                self.low_buffers[symbol] = []
            
            # Register with risk coordinator
            self.risk_coordinator.register_agent(
                self.agent_id,
                self.config.risk_params.__dict__
            )
            
            # Register with capital allocator
            self.capital_allocator.register_agent(self.agent_id)
            
            LOG.info(f"EquityMomentumAgent initialized for {len(self.config.symbols)} symbols")
            return True
            
        except Exception as e:
            LOG.error(f"Failed to initialize EquityMomentumAgent: {e}")
            return False
    
    def _get_loop_interval(self) -> float:
        return 30.0  # 30-second loop for momentum
    
    async def process_market_data(self, symbol: str, data: Dict) -> List[Signal]:
        """Process market data and generate momentum signals."""
        if symbol not in self.config.symbols:
            return []
        
        signals = []
        
        try:
            # Update buffers
            price = float(data.get('last_price', data.get('close', 0)))
            volume = float(data.get('volume', 0))
            high = float(data.get('high', price))
            low = float(data.get('low', price))
            
            self._update_buffers(symbol, price, volume, high, low)
            
            # Calculate indicators
            state = self._calculate_indicators(symbol)
            if not state:
                return []
            
            # Generate entry signals
            entry_signal = self._check_entry(symbol, state, price)
            if entry_signal:
                signals.append(entry_signal)
            
            # Update benchmark
            if symbol == self.benchmark_symbol:
                self.benchmark_buffer.append(price)
                if len(self.benchmark_buffer) > 200:
                    self.benchmark_buffer.pop(0)
            
        except Exception as e:
            LOG.error(f"Error processing {symbol}: {e}")
        
        return signals
    
    def _update_buffers(self, symbol: str, price: float, volume: float, high: float, low: float) -> None:
        """Update price/volume buffers."""
        max_len = max(self.lookback_periods) + 50
        
        self.price_buffers[symbol].append(price)
        self.volume_buffers[symbol].append(volume)
        self.high_buffers[symbol].append(high)
        self.low_buffers[symbol].append(low)
        
        for buf in [self.price_buffers[symbol], self.volume_buffers[symbol], 
                    self.high_buffers[symbol], self.low_buffers[symbol]]:
            if len(buf) > max_len:
                buf.pop(0)
    
    def _calculate_indicators(self, symbol: str) -> Optional[MomentumState]:
        """Calculate momentum indicators for a symbol."""
        state = self.momentum_states.get(symbol)
        if not state:
            return None
        
        prices = self.price_buffers[symbol]
        volumes = self.volume_buffers[symbol]
        highs = self.high_buffers[symbol]
        lows = self.low_buffers[symbol]
        
        if len(prices) < max(self.lookback_periods) + 20:
            return None
        
        prices_arr = np.array(prices)
        volumes_arr = np.array(volumes)
        highs_arr = np.array(highs)
        lows_arr = np.array(lows)

        # Multi-timeframe momentum
        for period in self.lookback_periods:
            if len(prices_arr) >= period + 1:
                mom = (prices_arr[-1] / prices_arr[-period-1] - 1) * 100
                if period == 20:
                    state.momentum_20 = mom
                elif period == 50:
                    state.momentum_50 = mom
                elif period == 100:
                    state.momentum_100 = mom
        
        # RSI
        state.rsi_14 = self._calculate_rsi(prices_arr, self.rsi_period)
        
        # Volume ratio
        if len(volumes_arr) >= self.volume_lookback + 1:
            avg_vol = np.mean(volumes_arr[-self.volume_lookback-1:-1])
            state.volume_ratio = volumes_arr[-1] / avg_vol if avg_vol > 0 else 1.0
        
        # Relative strength vs benchmark
        state.rs_rank = self._calculate_rs_rank(symbol, prices_arr)
        
        # Trend strength (ADX-like)
        state.trend_strength = self._calculate_trend_strength(highs_arr, lows_arr, prices_arr)
        
        # ATR for stops
        atr = self._calculate_atr(highs_arr, lows_arr, prices_arr, 14)
        
        state.last_update = datetime.now()
        state.atr = atr
        
        return state
    
    def _calculate_rsi(self, prices: np.ndarray, period: int) -> float:
        """Calculate RSI."""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_rs_rank(self, symbol: str, prices: np.ndarray) -> int:
        """Calculate relative strength rank vs benchmark."""
        if len(self.benchmark_buffer) < self.rs_lookback or len(prices) < self.rs_lookback:
            return 50
        
        try:
            sym_ret = (prices[-1] / prices[-self.rs_lookback] - 1) * 100
            bench_prices = np.array(self.benchmark_buffer[-self.rs_lookback:])
            bench_ret = (bench_prices[-1] / bench_prices[0] - 1) * 100
            
            rs = sym_ret - bench_ret
            # Simplified rank: positive RS = good
            return min(99, max(1, int(50 + rs * 2)))
        except Exception:
            return 50
    
    def _calculate_trend_strength(self, highs: np.ndarray, lows: np.ndarray, prices: np.ndarray) -> float:
        """Calculate ADX-like trend strength."""
        if len(prices) < 20:
            return 0.0
        
        highs = highs[-20:]
        lows = lows[-20:]
        prices = prices[-20:]
        
        tr = np.maximum(highs - lows, 
                       np.maximum(np.abs(highs - np.roll(prices, 1)), 
                                 np.abs(lows - np.roll(prices, 1))))
        tr = tr[1:]  # Remove first NaN
        
        plus_dm = np.diff(highs)
        minus_dm = -np.diff(lows)
        plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
        minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
        
        atr_14 = np.mean(tr[-14:])
        plus_di = 100 * np.mean(plus_dm[-14:]) / (atr_14 + 1e-10)
        minus_di = 100 * np.mean(minus_dm[-14:]) / (atr_14 + 1e-10)
        
        dx = np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100
        return float(dx)
    
    def _calculate_atr(self, highs: np.ndarray, lows: np.ndarray, prices: np.ndarray, period: int) -> float:
        """Calculate ATR."""
        if len(prices) < period + 1:
            return prices[-1] * 0.02
        
        highs = highs[-period-1:]
        lows = lows[-period-1:]
        prices = prices[-period-1:]
        
        tr = np.maximum(highs[1:] - lows[1:],
                       np.maximum(np.abs(highs[1:] - prices[:-1]),
                                 np.abs(lows[1:] - prices[:-1])))
        return float(np.mean(tr[-period:]))
    
    def _check_entry(self, symbol: str, state: MomentumState, price: float) -> Optional[Signal]:
        """Check for entry signals."""
        # Already have position?
        if symbol in self.positions:
            return None
        
        # Check risk limits
        if not self._can_open_position(symbol):
            return None
        
        # Multi-timeframe momentum alignment
        mom_aligned = (
            state.momentum_20 > self.momentum_threshold and
            state.momentum_50 > self.momentum_threshold * 0.5 and
            state.momentum_100 > 0
        )
        
        # Relative strength
        rs_strong = state.rs_rank > (self.rs_threshold * 100)
        
        # RSI filter
        rsi_ok = self.rsi_oversold < state.rsi_14 < self.rsi_overbought
        
        # Volume confirmation
        vol_confirmed = state.volume_ratio > self.min_volume_ratio
        
        # Trend strength
        trend_ok = state.trend_strength > 20
        
        if mom_aligned and rs_strong and rsi_ok and vol_confirmed and trend_ok:
            # Calculate position size
            capital = self.capital_allocator.get_allocation(self.agent_id)
            risk_per_trade = capital * 0.01  # 1% risk per trade
            atr = getattr(state, 'atr', price * 0.02)
            stop_distance = atr * self.trailing_atr_mult
            quantity = int(risk_per_trade / stop_distance)
            
            if quantity <= 0:
                return None
            
            stop_loss = price - stop_distance
            take_profit = price + (stop_distance * 2)  # 2:1 reward:risk
            
            return Signal(
                agent_id=self.agent_id,
                symbol=symbol,
                action="buy",
                strength=min(1.0, (state.momentum_20 / 5 + state.rs_rank / 100) / 2),
                price=price,
                quantity=quantity,
                stop_loss=stop_loss,
                take_profit=take_profit,
                leverage=1.0,
                metadata={
                    "strategy": "momentum",
                    "momentum_20": state.momentum_20,
                    "momentum_50": state.momentum_50,
                    "rs_rank": state.rs_rank,
                    "rsi": state.rsi_14,
                    "volume_ratio": state.volume_ratio
                }
            )
        
        return None
    
    def _can_open_position(self, symbol: str) -> bool:
        """Check if we can open a new position."""
        # Check portfolio limits
        if len(self.positions) >= self.config.risk_params.max_concurrent_positions:
            return False
        
        # Check capital allocation
        capital = self.capital_allocator.get_allocation(self.agent_id)
        used_capital = sum(p.quantity * p.current_price for p in self.positions.values())
        if used_capital > capital * 0.9:
            return False
        
        return True
    
    async def manage_positions(self) -> List[Signal]:
        """Manage existing positions - trailing stops, time stops, scaling."""
        signals = []
        
        for symbol, position in list(self.positions.items()):
            state = self.momentum_states.get(symbol)
            if not state:
                continue
            
            current_price = position.current_price
            atr = getattr(self.momentum_states.get(symbol), 'atr', current_price * 0.02)
            
            # Trailing stop
            if position.side == "long":
                trail_stop = current_price - (atr * self.trailing_atr_mult)
                if trail_stop > position.stop_loss:
                    signals.append(Signal(
                        agent_id=self.agent_id,
                        symbol=symbol,
                        action="modify",
                        strength=0.5,
                        price=current_price,
                        quantity=position.quantity,
                        stop_loss=trail_stop,
                        take_profit=position.take_profit,
                        metadata={"reason": "trailing_stop", "new_stop": trail_stop}
                    ))
            
            # Time stop
            hold_days = (datetime.now() - position.entry_time).days
            if hold_days >= self.max_hold_days:
                signals.append(Signal(
                    agent_id=self.agent_id,
                    symbol=symbol,
                    action="close",
                    strength=1.0,
                    price=current_price,
                    quantity=position.quantity,
                    metadata={"reason": "time_stop", "hold_days": hold_days}
                ))
            
            # Momentum breakdown exit
            if state.momentum_20 < -self.momentum_threshold:
                signals.append(Signal(
                    agent_id=self.agent_id,
                    symbol=symbol,
                    action="close",
                    strength=0.8,
                    price=current_price,
                    quantity=position.quantity,
                    metadata={"reason": "momentum_breakdown"}
                ))
        
        return signals