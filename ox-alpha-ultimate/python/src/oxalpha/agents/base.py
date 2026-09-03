"""
Base Agent Class
=================
Abstract base class for all trading agents.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Awaitable

from pydantic import BaseModel, Field

from ...config import AgentConfig, RiskLimits
from ...events import EventBus, Event, EventType

logger = logging.getLogger(__name__)


class AgentState(str, Enum):
    """Agent lifecycle states"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class Position:
    """Trading position"""
    symbol: str
    side: str  # "long" or "short"
    quantity: float
    avg_entry_price: float
    current_price: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    leverage: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def unrealized_pnl_pct(self) -> float:
        if self.avg_entry_price == 0:
            return 0.0
        if self.side == "long":
            return (self.current_price - self.avg_entry_price) / self.avg_entry_price
        else:
            return (self.avg_entry_price - self.current_price) / self.avg_entry_price
    
    def update_price(self, price: float) -> None:
        self.current_price = price
        if self.side == "long":
            self.unrealized_pnl = (price - self.avg_entry_price) * self.quantity
        else:
            self.unrealized_pnl = (self.avg_entry_price - price) * self.quantity


@dataclass
class Signal:
    """Trading signal"""
    signal_id: str
    agent_id: str
    symbol: str
    action: str  # buy, sell, close, modify
    side: str  # long, short
    quantity: float
    price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    strength: float = 1.0  # 0.0 to 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """
    Abstract base class for all trading agents.
    
    Each agent is independent with its own:
    - Risk parameters
    - Capital allocation
    - Data subscriptions
    - Signal generation logic
    - Position management
    """
    
    def __init__(self, config: "AgentConfig"):
        self.config = config
        self.agent_id = config.agent_id
        self.name = config.name
        self.symbols = config.symbols
        self.risk_limits = config.risk_limits
        self.params = config.params
        
        # State
        self.state = AgentState.INITIALIZING
        self.positions: Dict[str, Position] = {}
        self.signals: List[Signal] = []
        
        # Risk & Capital
        self.risk_limit_override: Optional[RiskLimits] = None
        self.allocated_capital: float = 0.0
        self.used_capital: float = 0.0
        
        # Performance
        self.daily_pnl: float = 0.0
        self.total_pnl: float = 0.0
        self.daily_trades: int = 0
        self.win_rate: float = 0.0
        
        # Event bus
        self.event_bus: Optional[asyncio.Queue] = None
        self._event_subscriptions: List[str] = []
        
        # State management
        self._lock = asyncio.Lock()
        self._running = False
        self._shutdown_event = asyncio.Event()
    
    @property
    def is_running(self) -> bool:
        return self.state == AgentState.RUNNING
    
    @property
    def is_paused(self) -> bool:
        return self.state == AgentState.PAUSED
    
    @property
    def available_capital(self) -> float:
        return max(0, self.allocated_capital - self.used_capital)
    
    @property
    def total_exposure(self) -> float:
        return sum(abs(p.quantity * p.current_price) for p in self.positions.values())
    
    @property
    def current_leverage(self) -> float:
        if self.allocated_capital <= 0:
            return 0.0
        return self.total_exposure / self.allocated_capital
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize agent resources (connections, data feeds, etc.)"""
        pass
    
    @abstractmethod
    async def start(self) -> bool:
        """Start the agent"""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the agent gracefully"""
        pass
    
    async def pause(self) -> None:
        """Pause the agent (keep positions, stop new signals)"""
        async with self._lock:
            if self.state == AgentState.RUNNING:
                self.state = AgentState.PAUSED
                logger.info(f"Agent {self.agent_id} paused")
    
    async def resume(self) -> None:
        """Resume a paused agent"""
        async with self._lock:
            if self.state == AgentState.PAUSED:
                self.state = AgentState.RUNNING
                logger.info(f"Agent {self.agent_id} resumed")
    
    async def health_check(self) -> bool:
        """Health check - override in subclasses"""
        return self.state == AgentState.RUNNING
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "state": self.state.value,
            "symbols": self.symbols,
            "positions": len(self.positions),
            "daily_pnl": self.daily_pnl,
            "total_pnl": self.total_pnl,
            "allocated_capital": self.allocated_capital,
            "used_capital": self.used_capital,
            "leverage": self.current_leverage,
            "daily_trades": self.daily_trades,
            "win_rate": self.win_rate,
        }
    
    # Event handling
    def set_event_bus(self, event_bus: asyncio.Queue) -> None:
        self.event_bus = event_bus
    
    def subscribe(self, event_type: str) -> None:
        """Subscribe to event type"""
        if event_type not in self._event_subscriptions:
            self._event_subscriptions.append(event_type)
    
    async def publish_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Publish event to event bus"""
        if self.event_bus:
            event = Event(
                event_type=event_type,
                source_agent=self.agent_id,
                data=data,
                timestamp=datetime.now()
            )
            try:
                self.event_bus.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning(f"{self.agent_id}: Event bus full")
    
    # Signal management
    async def emit_signal(self, signal: Signal) -> None:
        """Emit a trading signal"""
        if self.state != AgentState.RUNNING:
            logger.warning(f"{self.agent_id}: Cannot emit signal, not running")
            return
        
        # Validate signal
        if not await self._validate_signal(signal):
            return
        
        # Check risk limits
        if not await self._check_risk_limits(signal):
            return
        
        # Emit signal
        self.signals.append(signal)
        await self.publish_event("signal", {
            "signal": signal.model_dump() if hasattr(signal, 'model_dump') else signal.__dict__,
            "source_agent": self.agent_id,
        })
    
    async def _validate_signal(self, signal: Signal) -> bool:
        """Validate signal before emission"""
        if signal.quantity <= 0:
            logger.warning(f"{self.agent_id}: Invalid quantity")
            return False
        
        if signal.price <= 0:
            logger.warning(f"{self.agent_id}: Invalid price")
            return False
        
        if signal.strength < 0 or signal.strength > 1:
            logger.warning(f"{self.agent_id}: Invalid strength")
            return False
        
        return True
    
    async def _check_risk_limits(self, signal: Signal) -> bool:
        """Check signal against risk limits"""
        limits = self.risk_limit_override or self.risk_limits
        
        # Position count
        if len(self.positions) >= limits.max_positions:
            logger.warning(f"{self.agent_id}: Max positions reached")
            return False
        
        # Leverage
        estimated_exposure = signal.quantity * signal.price
        new_leverage = (self.total_exposure + estimated_exposure) / max(self.allocated_capital, 1)
        if new_leverage > limits.max_leverage:
            logger.warning(f"{self.agent_id}: Leverage limit exceeded")
            return False
        
        # Daily loss
        if self.daily_pnl < -self.allocated_capital * limits.max_daily_loss_pct:
            logger.warning(f"{self.agent_id}: Daily loss limit reached")
            return False
        
        return True
    
    # Position management
    async def open_position(self, signal: Signal) -> bool:
        """Open a new position from signal"""
        async with self._lock:
            if signal.symbol in self.positions:
                logger.warning(f"{self.agent_id}: Position already exists for {signal.symbol}")
                return False
            
            position = Position(
                symbol=signal.symbol,
                side=signal.side,
                quantity=signal.quantity,
                avg_entry_price=signal.price,
                current_price=signal.price,
                entry_time=datetime.now(),
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                leverage=signal.leverage if signal.leverage > 0 else 1.0,
            )
            
            self.positions[signal.symbol] = position
            self.used_capital += signal.quantity * signal.price
            self.daily_trades += 1
            
            logger.info(f"{self.agent_id}: Opened {signal.side} position in {signal.symbol}: "
                       f"{signal.quantity} @ {signal.price}")
            return True
    
    async def close_position(self, symbol: str, reason: str = "") -> bool:
        """Close an existing position"""
        async with self._lock:
            position = self.positions.pop(symbol, None)
            if not position:
                logger.warning(f"{self.agent_id}: No position to close for {symbol}")
                return False
            
            # Calculate realized PnL
            pnl = position.unrealized_pnl
            self.daily_pnl += pnl
            self.total_pnl += pnl
            self.used_capital -= position.quantity * position.current_price
            
            logger.info(f"{self.agent_id}: Closed {position.side} position in {symbol}: "
                       f"PnL={pnl:.2f} ({reason})")
            return True
    
    async def update_position_price(self, symbol: str, price: float) -> None:
        """Update position with current price"""
        if symbol in self.positions:
            self.positions[symbol].update_price(price)
    
    async def manage_positions(self) -> List[Signal]:
        """Manage existing positions - returns exit signals if any"""
        signals = []
        
        for symbol, position in list(self.positions.items()):
            # Check stop loss
            if position.stop_loss:
                if position.side == "long" and position.current_price <= position.stop_loss:
                    signals.append(Signal(
                        signal_id=f"{self.agent_id}_{symbol}_sl",
                        agent_id=self.agent_id,
                        symbol=symbol,
                        action="close",
                        side=position.side,
                        quantity=position.quantity,
                        price=position.current_price,
                        metadata={"reason": "stop_loss"}
                    ))
                elif position.side == "short" and position.current_price >= position.stop_loss:
                    signals.append(Signal(
                        signal_id=f"{self.agent_id}_{symbol}_sl",
                        agent_id=self.agent_id,
                        symbol=symbol,
                        action="close",
                        side=position.side,
                        quantity=position.quantity,
                        price=position.current_price,
                        metadata={"reason": "stop_loss"}
                    ))
            
            # Check take profit
            if position.take_profit:
                if position.side == "long" and position.current_price >= position.take_profit:
                    signals.append(Signal(
                        signal_id=f"{self.agent_id}_{symbol}_tp",
                        agent_id=self.agent_id,
                        symbol=symbol,
                        action="close",
                        side=position.side,
                        quantity=position.quantity,
                        price=position.current_price,
                        metadata={"reason": "take_profit"}
                    ))
                elif position.side == "short" and position.current_price <= position.take_profit:
                    signals.append(Signal(
                        signal_id=f"{self.agent_id}_{symbol}_tp",
                        agent_id=self.agent_id,
                        symbol=symbol,
                        action="close",
                        side=position.side,
                        quantity=position.quantity,
                        price=position.current_price,
                        metadata={"reason": "take_profit"}
                    ))
            
            # Time-based exit
            hold_time = (datetime.now() - position.entry_time).total_seconds() / 3600
            max_hold_hours = self.params.get("max_hold_hours", 24)
            if hold_time > max_hold_hours:
                signals.append(Signal(
                    signal_id=f"{self.agent_id}_{symbol}_time",
                    agent_id=self.agent_id,
                    symbol=symbol,
                    action="close",
                    side=position.side,
                    quantity=position.quantity,
                    price=position.current_price,
                    metadata={"reason": "time_stop", "hold_hours": hold_time}
                ))
        
        return signals
    
    # Capital management
    def set_allocated_capital(self, amount: float) -> None:
        self.allocated_capital = amount
        logger.info(f"{self.agent_id}: Capital allocated: {amount:,.2f}")
    
    def get_capital_usage(self) -> Dict[str, float]:
        return {
            "allocated": self.allocated_capital,
            "used": self.used_capital,
            "available": self.available_capital,
            "exposure": self.total_exposure,
            "leverage": self.current_leverage,
        }
    
    # Performance tracking
    def record_trade(self, pnl: float, commission: float = 0.0) -> None:
        """Record completed trade"""
        net_pnl = pnl - commission
        self.daily_pnl += net_pnl
        self.total_pnl += net_pnl
        self.daily_trades += 1
        
        # Update win rate (simplified)
        if self.daily_trades > 0:
            wins = sum(1 for _ in range(self.daily_trades) if net_pnl > 0)  # Simplified
            self.win_rate = wins / self.daily_trades
    
    def reset_daily(self) -> None:
        """Reset daily counters (call at market open)"""
        self.daily_pnl = 0.0
        self.daily_trades = 0
    
    def get_performance_summary(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "total_pnl": self.total_pnl,
            "daily_pnl": self.daily_pnl,
            "total_trades": self.daily_trades,  # Simplified
            "win_rate": self.win_rate,
            "current_leverage": self.current_leverage,
            "positions": len(self.positions),
            "allocated_capital": self.allocated_capital,
        }