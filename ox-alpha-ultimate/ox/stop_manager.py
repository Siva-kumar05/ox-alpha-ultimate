"""Advanced Stop Management: Trailing stops, time stops, breakeven, chandelier exits."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List
import numpy as np
import pandas as pd
from .indicators import ind
from .core import iso


class StopType(Enum):
    FIXED = "fixed"
    TRAILING = "trailing"
    TIME = "time"
    BREAKEVEN = "breakeven"
    CHANDELIER = "chandelier"
    ATR_TRAILING = "atr_trailing"
    PERCENTAGE_TRAILING = "percentage_trailing"
    VOLATILITY_ADJUSTED = "volatility_adjusted"


@dataclass
class StopConfig:
    """Configuration for a stop type."""
    stop_type: StopType
    atr_multiple: float = 2.0
    trail_percent: float = 0.02
    time_limit_bars: int = 50
    breakeven_trigger_pct: float = 0.01
    chandelier_mult: float = 3.0
    chandelier_lookback: int = 22
    volatility_lookback: int = 20
    min_trail_distance: float = 0.005


@dataclass
class StopState:
    """Current state of a stop."""
    stop_type: StopType
    initial_stop: float
    current_stop: float
    highest_price: float
    lowest_price: float
    entry_price: float
    entry_time: str
    bars_held: int = 0
    breakeven_triggered: bool = False
    trail_activated: bool = False
    metadata: Dict = field(default_factory=dict)


class StopManager:
    """Manages multiple stop types for positions."""
    
    def __init__(self, cfg):
        self.cfg = cfg
        stop_cfg = cfg.get("stop_management", {})
        self.default_stop_type = StopType(stop_cfg.get("default_type", "atr_trailing"))
        self.default_atr_mult = stop_cfg.get("default_atr_mult", 2.0)
        self.default_trail_pct = stop_cfg.get("default_trail_pct", 0.02)
        self.default_time_bars = stop_cfg.get("default_time_bars", 100)
        self.breakeven_trigger = stop_cfg.get("breakeven_trigger_pct", 0.01)
        self.chandelier_mult = stop_cfg.get("chandelier_mult", 3.0)
        self.chandelier_lookback = stop_cfg.get("chandelier_lookback", 22)
        
        # Per-symbol stop states
        self._stop_states: Dict[str, StopState] = {}
    
    def initialize_stop(
        self,
        symbol: str,
        entry_price: float,
        initial_stop: float,
        stop_type: Optional[StopType] = None,
        config: Optional[StopConfig] = None,
        entry_time: Optional[str] = None
    ) -> StopState:
        """Initialize stop for a new position."""
        stop_type = stop_type or self.default_stop_type
        
        if config is None:
            config = StopConfig(
                stop_type=stop_type,
                atr_multiple=self.default_atr_mult,
                trail_percent=self.default_trail_pct,
                time_limit_bars=self.default_time_bars,
                breakeven_trigger_pct=self.breakeven_trigger,
                chandelier_mult=self.chandelier_mult,
                chandelier_lookback=self.chandelier_lookback
            )
        
        state = StopState(
            stop_type=stop_type,
            initial_stop=initial_stop,
            current_stop=initial_stop,
            highest_price=entry_price,
            lowest_price=entry_price,
            entry_price=entry_price,
            entry_time=entry_time or iso(),
            metadata={"config": config}
        )
        
        self._stop_states[symbol] = state
        return state
    
    def update_stop(
        self,
        symbol: str,
        current_price: float,
        high: float,
        low: float,
        close: float,
        atr: float,
        bar_count: int = 1
    ) -> Optional[float]:
        """Update stop based on price action. Returns new stop price if changed."""
        state = self._stop_states.get(symbol)
        if not state:
            return None
        
        config = state.metadata.get("config", StopConfig(stop_type=state.stop_type))
        state.bars_held += bar_count
        state.highest_price = max(state.highest_price, high)
        state.lowest_price = min(state.lowest_price, low)
        
        new_stop = state.current_stop
        
        # Apply stop logic based on type
        if state.stop_type == StopType.TRAILING:
            new_stop = self._update_trailing_stop(state, current_price, config)
        elif state.stop_type == StopType.ATR_TRAILING:
            new_stop = self._update_atr_trailing_stop(state, current_price, atr, config)
        elif state.stop_type == StopType.PERCENTAGE_TRAILING:
            new_stop = self._update_percentage_trailing_stop(state, current_price, config)
        elif state.stop_type == StopType.CHANDELIER:
            new_stop = self._update_chandelier_stop(state, high, low, close, config)
        elif state.stop_type == StopType.VOLATILITY_ADJUSTED:
            new_stop = self._update_volatility_adjusted_stop(state, current_price, atr, config)
        elif state.stop_type == StopType.BREAKEVEN:
            new_stop = self._update_breakeven_stop(state, current_price, config)
        elif state.stop_type == StopType.TIME:
            new_stop = self._update_time_stop(state, current_price, config)
        
        # Time stop always applies as secondary
        time_stop = self._update_time_stop(state, current_price, config)
        if time_stop is not None:
            new_stop = max(new_stop, time_stop) if state.stop_type in [StopType.TRAILING, StopType.ATR_TRAILING] else time_stop
        
        # Breakeven stop as secondary (if not primary)
        if state.stop_type != StopType.BREAKEVEN:
            be_stop = self._update_breakeven_stop(state, current_price, config)
            if be_stop is not None and be_stop > new_stop:
                new_stop = be_stop
        
        if new_stop != state.current_stop:
            state.current_stop = new_stop
            return new_stop
        
        return None
    
    def _update_trailing_stop(self, state: StopState, current_price: float, config: StopConfig) -> float:
        """Simple trailing stop: trails highest price by fixed percentage."""
        trail_distance = state.highest_price * config.trail_percent
        new_stop = state.highest_price - trail_distance
        return max(new_stop, state.current_stop)  # Stop only moves up
    
    def _update_atr_trailing_stop(self, state: StopState, current_price: float, atr: float, config: StopConfig) -> float:
        """ATR-based trailing stop."""
        if atr <= 0:
            return state.current_stop
        trail_distance = atr * config.atr_multiple
        new_stop = state.highest_price - trail_distance
        return max(new_stop, state.current_stop)
    
    def _update_percentage_trailing_stop(self, state: StopState, current_price: float, config: StopConfig) -> float:
        """Percentage-based trailing stop with minimum distance."""
        trail_distance = state.highest_price * config.trail_percent
        min_distance = state.highest_price * config.min_trail_distance
        trail_distance = max(trail_distance, min_distance)
        new_stop = state.highest_price - trail_distance
        return max(new_stop, state.current_stop)
    
    def _update_chandelier_stop(self, state: StopState, high: float, low: float, close: float, config: StopConfig) -> float:
        """Chandelier exit: trails highest high by ATR multiple."""
        # Chandelier long stop: Highest High - (ATR * multiplier)
        trail_distance = ind("atr")(high, low, close, config.chandelier_lookback)[-1] * config.chandelier_mult
        if np.isnan(trail_distance) or trail_distance <= 0:
            return state.current_stop
        new_stop = state.highest_price - trail_distance
        return max(new_stop, state.current_stop)
    
    def _update_volatility_adjusted_stop(self, state: StopState, current_price: float, atr: float, config: StopConfig) -> float:
        """Volatility-adjusted trailing stop."""
        if atr <= 0:
            return state.current_stop
        
        # Adjust ATR multiplier based on recent volatility
        vol_series = ind("historical_volatility")(np.array([state.entry_price] * config.volatility_lookback), config.volatility_lookback)
        current_vol = vol_series[-1] if not np.isnan(vol_series[-1]) else atr / current_price * 100
        
        # Higher vol = wider stop
        vol_factor = max(0.5, min(2.0, current_vol / 20.0))  # Normalize around 20% annual vol
        adjusted_mult = config.atr_multiple * vol_factor
        
        trail_distance = atr * adjusted_mult
        new_stop = state.highest_price - trail_distance
        return max(new_stop, state.current_stop)
    
    def _update_breakeven_stop(self, state: StopState, current_price: float, config: StopConfig) -> Optional[float]:
        """Move stop to breakeven once profit target reached."""
        if state.breakeven_triggered:
            return state.entry_price
        
        profit_pct = (current_price - state.entry_price) / state.entry_price
        if profit_pct >= config.breakeven_trigger_pct:
            state.breakeven_triggered = True
            return state.entry_price
        
        return None
    
    def _update_time_stop(self, state: StopState, current_price: float, config: StopConfig) -> Optional[float]:
        """Time-based stop: exit after max bars held."""
        if state.bars_held >= config.time_limit_bars:
            # Force exit by setting stop to current price (will trigger on next update)
            return current_price
        return None
    
    def get_stop(self, symbol: str) -> Optional[float]:
        """Get current stop price for symbol."""
        state = self._stop_states.get(symbol)
        return state.current_stop if state else None
    
    def get_stop_state(self, symbol: str) -> Optional[StopState]:
        """Get full stop state."""
        return self._stop_states.get(symbol)
    
    def should_exit(self, symbol: str, current_price: float) -> tuple[bool, str]:
        """Check if position should be exited."""
        state = self._stop_states.get(symbol)
        if not state:
            return False, "no_stop"
        
        if current_price <= state.current_stop:
            reason = f"{state.stop_type.value}_stop_hit"
            if state.bars_held >= state.metadata.get("config", StopConfig(stop_type=state.stop_type)).time_limit_bars:
                reason = "time_stop"
            return True, reason
        
        return False, "holding"
    
    def remove_stop(self, symbol: str) -> bool:
        """Remove stop for closed position."""
        if symbol in self._stop_states:
            del self._stop_states[symbol]
            return True
        return False
    
    def get_all_stops(self) -> Dict[str, StopState]:
        """Get all active stops."""
        return dict(self._stop_states)


class MultiStopManager:
    """Manages multiple concurrent stops for a position (e.g., trailing + time + breakeven)."""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.stop_manager = StopManager(cfg)
        self.multi_stop_config = cfg.get("multi_stop", {
            "primary": "atr_trailing",
            "secondary": ["time", "breakeven"]
        })
    
    def initialize_position(
        self,
        symbol: str,
        entry_price: float,
        initial_stop: float,
        atr: float,
        entry_time: Optional[str] = None
    ) -> Dict[StopType, StopState]:
        """Initialize multiple stops for a position."""
        stops = {}
        
        # Primary stop
        primary_type = StopType(self.multi_stop_config["primary"])
        config = StopConfig(
            stop_type=primary_type,
            atr_multiple=self.stop_manager.default_atr_mult,
            trail_percent=self.stop_manager.default_trail_pct,
            time_limit_bars=self.stop_manager.default_time_bars,
            breakeven_trigger_pct=self.stop_manager.breakeven_trigger,
            chandelier_mult=self.stop_manager.chandelier_mult,
            chandelier_lookback=self.stop_manager.chandelier_lookback
        )
        stops[primary_type] = self.stop_manager.initialize_stop(
            symbol, entry_price, initial_stop, primary_type, config, entry_time
        )
        
        # Secondary stops
        for sec_type_str in self.multi_stop_config.get("secondary", []):
            sec_type = StopType(sec_type_str)
            sec_config = StopConfig(
                stop_type=sec_type,
                time_limit_bars=self.stop_manager.default_time_bars,
                breakeven_trigger_pct=self.stop_manager.breakeven_trigger
            )
            stops[sec_type] = self.stop_manager.initialize_stop(
                symbol, entry_price, initial_stop, sec_type, sec_config, entry_time
            )
        
        return stops
    
    def update_all_stops(
        self,
        symbol: str,
        current_price: float,
        high: float,
        low: float,
        close: float,
        atr: float
    ) -> Dict[StopType, float]:
        """Update all stops and return the tightest (most protective) stop."""
        results = {}
        tightest_stop = None
        
        for stop_type, state in self.stop_manager._stop_states.items():
            if state.metadata.get("symbol") == symbol:
                new_stop = self.stop_manager.update_stop(symbol, current_price, high, low, close, atr)
                if new_stop is not None:
                    results[stop_type] = new_stop
                    if tightest_stop is None or new_stop > tightest_stop:
                        tightest_stop = new_stop
        
        return results
    
    def check_exit(self, symbol: str, current_price: float) -> tuple[bool, str, StopType]:
        """Check if any stop triggers exit."""
        for stop_type, state in self.stop_manager._stop_states.items():
            if state.metadata.get("symbol") == symbol:
                should_exit, reason = self.stop_manager.should_exit(symbol, current_price)
                if should_exit:
                    return True, reason, stop_type
        return False, "holding", StopType.FIXED