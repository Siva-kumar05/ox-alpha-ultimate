"""
Core Multi-Agent Infrastructure
================================
Base classes, orchestration, resource pooling, and shared data bus.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor
import threading

import numpy as np

if TYPE_CHECKING:
    from .risk_coordinator import RiskCoordinator
    from .capital_allocator import CapitalAllocator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s'
)
LOG = logging.getLogger("promax")


class AgentState(Enum):
    """Agent lifecycle states."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class AgentType(Enum):
    """Agent specialization types."""
    EQUITY_MOMENTUM = "equity_momentum"
    EQUITY_GROWTH = "equity_growth"
    INTRADAY_SCALPER = "intraday_scalper"
    CRYPTO_PERP = "crypto_perp"
    CRYPTO_FUNDING = "crypto_funding"
    CRYPTO_MEME_SWING = "crypto_meme_swing"
    OPTIONS_0DTE = "options_0dte"
    MARKET_MAKER = "market_maker"
    NEWS_INTEL = "news_intel"
    SOCIAL_MONITOR = "social_monitor"
    RISK_COORDINATOR = "risk_coordinator"
    CAPITAL_ALLOCATOR = "capital_allocator"


@dataclass
class RiskParams:
    """Risk parameters for an agent."""
    max_leverage: float = 1.0
    max_position_pct: float = 0.10
    max_daily_loss_pct: float = 0.02
    max_drawdown_pct: float = 0.10
    kelly_fraction: float = 0.25
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.05
    max_concurrent_positions: int = 5
    correlation_limit: float = 0.7


@dataclass
class AgentConfig:
    """Configuration for an agent."""
    agent_id: str
    agent_type: AgentType
    name: str
    symbols: List[str]
    risk_params: RiskParams
    enabled: bool = True
    priority: int = 1  # 1=highest, 10=lowest
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """Unified position representation."""
    agent_id: str
    symbol: str
    side: str  # long, short
    quantity: float
    entry_price: float
    current_price: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    leverage: float = 1.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class Signal:
    """Trading signal from an agent."""
    agent_id: str
    symbol: str
    action: str  # buy, sell, close
    strength: float  # 0-1
    price: float
    quantity: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    leverage: float = 1.0
    metadata: Dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    signal_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


class SharedDataBus:
    """Thread-safe shared data bus for inter-agent communication."""
    
    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._locks: Dict[str, threading.RLock] = defaultdict(threading.RLock)
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._global_lock = threading.RLock()
    
    def publish(self, topic: str, data: Any) -> None:
        """Publish data to a topic."""
        with self._locks[topic]:
            self._data[topic] = data
            for callback in self._subscribers[topic]:
                try:
                    callback(data)
                except Exception as e:
                    LOG.error(f"Subscriber error on {topic}: {e}")
    
    def subscribe(self, topic: str, callback: Callable) -> None:
        """Subscribe to a topic."""
        with self._locks[topic]:
            self._subscribers[topic].append(callback)
    
    def get(self, topic: str, default: Any = None) -> Any:
        """Get latest data for a topic."""
        with self._locks[topic]:
            return self._data.get(topic, default)
    
    def get_all(self, prefix: str = "") -> Dict[str, Any]:
        """Get all data matching prefix."""
        with self._global_lock:
            return {k: v for k, v in self._data.items() if k.startswith(prefix)}


class ResourcePool:
    """Shared resource pool to minimize duplication."""
    
    def __init__(self):
        self._resources: Dict[str, Any] = {}
        self._ref_counts: Dict[str, int] = {}
        self._lock = threading.RLock()
    
    def acquire(self, resource_key: str, factory: Callable) -> Any:
        """Acquire a shared resource, creating if needed."""
        with self._lock:
            if resource_key not in self._resources:
                self._resources[resource_key] = factory()
                self._ref_counts[resource_key] = 0
            self._ref_counts[resource_key] += 1
            return self._resources[resource_key]
    
    def release(self, resource_key: str) -> None:
        """Release a shared resource."""
        with self._lock:
            if resource_key in self._ref_counts:
                self._ref_counts[resource_key] -= 1
                if self._ref_counts[resource_key] <= 0:
                    resource = self._resources.pop(resource_key, None)
                    if hasattr(resource, 'close'):
                        try:
                            resource.close()
                        except Exception:
                            pass
                    self._ref_counts.pop(resource_key, None)
    
    def get(self, resource_key: str) -> Optional[Any]:
        """Get resource without incrementing ref count."""
        with self._lock:
            return self._resources.get(resource_key)


class BaseAgent(ABC):
    """
    Abstract base class for all trading agents.
    Each agent is independent with its own risk, data, and execution.
    """
    
    def __init__(
        self,
        config: AgentConfig,
        resource_pool: ResourcePool,
        data_bus: SharedDataBus,
        risk_coordinator: 'RiskCoordinator',
        capital_allocator: 'CapitalAllocator'
    ):
        self.config = config
        self.resource_pool = resource_pool
        self.data_bus = data_bus
        self.risk_coordinator = risk_coordinator
        self.capital_allocator = capital_allocator
        
        self.state = AgentState.INITIALIZING
        self.agent_id = config.agent_id
        self.agent_type = config.agent_type
        
        # State
        self.positions: Dict[str, Position] = {}
        self.signals: List[Signal] = []
        self.performance_metrics: Dict[str, float] = {}
        
        # Resources
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._running = False
        self._lock = threading.RLock()
        
        # Subscribe to relevant data
        self._setup_subscriptions()
    
    def _setup_subscriptions(self) -> None:
        """Subscribe to relevant data topics."""
        # Market data for our symbols
        for symbol in self.config.symbols:
            self.data_bus.subscribe(f"market:{symbol}", self._on_market_data)
        
        # Risk updates
        self.data_bus.subscribe("risk:limits", self._on_risk_update)
        self.data_bus.subscribe("capital:allocation", self._on_capital_update)
        
        # News/sentiment
        self.data_bus.subscribe("news:sentiment", self._on_news)
        self.data_bus.subscribe("social:sentiment", self._on_social)
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize agent resources. Return True if successful."""
        pass
    
    @abstractmethod
    async def process_market_data(self, symbol: str, data: Dict) -> List[Signal]:
        """Process market data and generate signals."""
        pass
    
    @abstractmethod
    async def manage_positions(self) -> List[Signal]:
        """Manage existing positions (stops, targets, scaling)."""
        pass
    
    async def start(self) -> bool:
        """Start the agent."""
        try:
            self.state = AgentState.INITIALIZING
            success = await self.initialize()
            if not success:
                self.state = AgentState.ERROR
                return False
            
            self.state = AgentState.RUNNING
            self._running = True
            LOG.info(f"Agent {self.config.name} ({self.agent_id}) started")
            
            # Start main loop
            asyncio.create_task(self._main_loop())
            return True
        except Exception as e:
            LOG.error(f"Failed to start agent {self.agent_id}: {e}")
            self.state = AgentState.ERROR
            return False
    
    async def pause(self) -> None:
        """Pause the agent (keep positions, stop new signals)."""
        self.state = AgentState.PAUSED
        LOG.info(f"Agent {self.agent_id} paused")
    
    async def resume(self) -> None:
        """Resume the agent."""
        if self.state == AgentState.PAUSED:
            self.state = AgentState.RUNNING
            LOG.info(f"Agent {self.agent_id} resumed")
    
    async def stop(self) -> None:
        """Stop the agent gracefully."""
        self.state = AgentState.STOPPING
        self._running = False
        
        # Close positions if needed
        await self._close_all_positions()
        
        self._executor.shutdown(wait=True)
        self.state = AgentState.STOPPED
        LOG.info(f"Agent {self.agent_id} stopped")
    
    async def _main_loop(self) -> None:
        """Main agent loop."""
        while self._running and self.state == AgentState.RUNNING:
            try:
                # Check risk limits
                if not await self._check_risk_limits():
                    await self.pause()
                    await asyncio.sleep(60)
                    continue
                
                # Generate signals
                signals = await self.manage_positions()
                for signal in signals:
                    await self._emit_signal(signal)
                
                # Update metrics
                await self._update_metrics()
                
                # Sleep based on agent type
                await asyncio.sleep(self._get_loop_interval())
                
            except Exception as e:
                LOG.error(f"Error in agent {self.agent_id} main loop: {e}")
                await asyncio.sleep(5)
    
    @abstractmethod
    def _get_loop_interval(self) -> float:
        """Get loop interval in seconds based on agent type."""
        pass
    
    async def _check_risk_limits(self) -> bool:
        """Check if agent is within risk limits."""
        return await self.risk_coordinator.check_agent_limits(self.agent_id, self.positions)
    
    async def _emit_signal(self, signal: Signal) -> None:
        """Emit signal after the risk gate, the debate gate and — for
        capital-deploying actions — the human approval gate.

        Order of gates for a buy: risk limits → bull/bear debate (deterministic
        indicator panel + per-agent mistake memory; vetoes weak setups) →
        human approval (PENDING intent until decided).  Sells and closes
        pass straight through: exiting a position must never wait.
        """
        # Validate with risk coordinator
        approved = await self.risk_coordinator.approve_signal(signal)
        if not approved:
            return

        action = str(signal.action).lower()
        if action in ("buy", "open", "add"):
            panel = self.resource_pool.get("debate_panel")
            if panel is not None:
                series = self._price_series_for_debate(signal.symbol)
                if series is not None and len(series) >= 60:
                    verdict = panel.debate(self.agent_id, signal.symbol, series, side_hint="buy")
                    if not verdict["pass"]:
                        self.data_bus.publish("debate:veto", {
                            "agent_id": self.agent_id, "symbol": signal.symbol,
                            "verdict": verdict["verdict"], "reason": verdict["reason"],
                        })
                        return
                    signal.strength = max(0.05, signal.strength * abs(verdict["verdict"]))
                    signal.metadata = dict(signal.metadata or {})
                    signal.metadata["debate_verdict"] = verdict["verdict"]

        gateway = self.resource_pool.get("approval_gateway")
        exempt = bool(signal.metadata.get("approval_exempt")) if signal.metadata else False
        if gateway is not None and not exempt and gateway.needs_human(signal.action):
            intent = gateway.submit(self.agent_id, signal)
            if intent.get("status") != "APPROVED":
                # Capital-deploying action parked until a human decides.
                self.data_bus.publish("approvals:pending", intent)
                return

        self.data_bus.publish(f"signals:{signal.agent_id}", signal)
        self.signals.append(signal)

    def _price_series_for_debate(self, symbol: str) -> Optional[np.ndarray]:
        """Best-effort recent closes for the debate panel.

        Agents keep buffers under different names (price_buffers, buffers,
        price_histories...).  This probes the common ones so the debate gate
        works for every agent without each one implementing a hook.
        """
        for attr in ("price_buffers", "buffers", "price_histories"):
            store = getattr(self, attr, None)
            if isinstance(store, dict):
                buf = store.get(symbol)
                if buf is not None and len(buf) > 0:
                    return np.asarray(list(buf), dtype=float)
        state = getattr(self, "states", None)
        if isinstance(state, dict):
            st = state.get(symbol)
            if st is not None:
                for attr in ("price_history", "mark_price_history", "prices", "price_buffer"):
                    hist = getattr(st, attr, None)
                    if hist is not None and len(hist) > 0:
                        return np.asarray(list(hist), dtype=float)
        # Attribute-per-symbol pattern (intraday_scalper: _price_history_<sym>).
        hist = getattr(self, f"_price_history_{symbol}", None)
        if hist is not None and len(hist) > 0:
            return np.asarray(list(hist), dtype=float)
        return None
    
    async def _update_metrics(self) -> None:
        """Update performance metrics."""
        total_pnl = sum(p.unrealized_pnl + p.realized_pnl for p in self.positions.values())
        self.performance_metrics = {
            "total_pnl": total_pnl,
            "position_count": len(self.positions),
            "capital_used": self.capital_allocator.get_agent_usage(self.agent_id),
            "timestamp": datetime.now().isoformat()
        }
        self.data_bus.publish(f"metrics:{self.agent_id}", self.performance_metrics)
    
    async def _close_all_positions(self) -> None:
        """Close all positions on stop."""
        for symbol, position in self.positions.items():
            close_signal = Signal(
                agent_id=self.agent_id,
                symbol=symbol,
                action="close",
                strength=1.0,
                price=position.current_price,
                quantity=position.quantity,
                metadata={"reason": "agent_stop"}
            )
            self.data_bus.publish(f"signals:{self.agent_id}", close_signal)
    
    # Event handlers
    def _on_market_data(self, data: Dict) -> None:
        """Handle incoming market data."""
        asyncio.create_task(self._handle_market_data(data))
    
    async def _handle_market_data(self, data: Dict) -> None:
        if self.state != AgentState.RUNNING:
            return
        symbol = data.get("symbol")
        if symbol in self.config.symbols:
            signals = await self.process_market_data(symbol, data)
            for signal in signals:
                await self._emit_signal(signal)
    
    def _on_risk_update(self, data: Dict) -> None:
        """Handle risk limit updates."""
        pass
    
    def _on_capital_update(self, data: Dict) -> None:
        """Handle capital allocation updates."""
        pass
    
    def _on_news(self, data: Dict) -> None:
        """Handle news sentiment."""
        pass
    
    def _on_social(self, data: Dict) -> None:
        """Handle social sentiment."""
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status for monitoring."""
        return {
            "agent_id": self.agent_id,
            "name": self.config.name,
            "type": self.config.agent_type.value,
            "state": self.state.value,
            "symbols": self.config.symbols,
            "positions": len(self.positions),
            "metrics": self.performance_metrics,
            "enabled": self.config.enabled
        }
