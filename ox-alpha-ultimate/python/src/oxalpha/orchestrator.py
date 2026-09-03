"""
Agent Orchestrator - Manages lifecycle and coordination of all agents
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable, Awaitable

from pydantic import BaseModel

from .config import Settings, AgentConfig, get_settings
from .agents.base import BaseAgent, AgentState

logger = logging.getLogger(__name__)


@dataclass
class AgentMetrics:
    """Runtime metrics for an agent"""
    agent_id: str
    state: AgentState
    signals_generated: int = 0
    signals_executed: int = 0
    pnl: float = 0.0
    positions: int = 0
    last_update: datetime = field(default_factory=datetime.now)
    errors: int = 0


class AgentOrchestrator:
    """
    Orchestrates all agents - manages lifecycle, coordination, and monitoring.
    
    Responsibilities:
    - Agent lifecycle (start, stop, pause, resume)
    - Inter-agent communication via event bus
    - Capital allocation coordination
    - Risk limit enforcement
    - Health monitoring
    - Graceful shutdown
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_configs: Dict[str, AgentConfig] = {}
        self.metrics: Dict[str, AgentMetrics] = {}
        
        # Shared infrastructure
        self.event_bus: asyncio.Queue = asyncio.Queue(maxsize=10000)
        self.capital_allocator = CapitalAllocator()
        self.risk_manager = RiskManager()
        
        # State
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._tasks: List[asyncio.Task] = []
        
        # Callbacks
        self._state_change_callbacks: List[Callable[[str, AgentState], Awaitable[None]]] = []
    
    def register_agent(self, config: AgentConfig, agent_factory: Callable[[AgentConfig], BaseAgent]) -> None:
        """Register an agent with its factory function"""
        self.agent_configs[config.agent_id] = config
        self._agent_factories[config.agent_id] = agent_factory
        logger.info(f"Registered agent: {config.agent_id} ({config.name})")
    
    async def start(self) -> None:
        """Start all enabled agents in priority order"""
        if self._running:
            logger.warning("Orchestrator already running")
            return
        
        logger.info("Starting Agent Orchestrator...")
        self._running = True
        
        # Initialize capital allocator
        await self.capital_allocator.initialize(self.settings.total_capital)
        
        # Initialize risk manager
        await self.risk_manager.initialize()
        
        # Sort agents by priority
        sorted_agents = sorted(
            self.settings.get_enabled_agents(),
            key=lambda c: c.priority
        )
        
        # Start agents in priority order
        for config in sorted_agents:
            await self._start_agent(config)
            await asyncio.sleep(0.5)  # Stagger startup
        
        # Start background tasks
        self._tasks = [
            asyncio.create_task(self._event_loop()),
            asyncio.create_task(self._monitor_loop()),
            asyncio.create_task(self._health_check_loop()),
            asyncio.create_task(self._rebalance_loop()),
        ]
        
        logger.info(f"Started {len(self.agents)} agents")
    
    async def _start_agent(self, config: AgentConfig) -> bool:
        """Start a single agent"""
        try:
            factory = self._agent_factories.get(config.agent_id)
            if not factory:
                logger.error(f"No factory for agent: {config.agent_id}")
                return False
            
            agent = factory(config)
            await agent.initialize()
            
            self.agents[config.agent_id] = agent
            self.metrics[config.agent_id] = AgentMetrics(
                agent_id=config.agent_id,
                state=AgentState.INITIALIZING
            )
            
            success = await agent.start()
            if success:
                await self._on_state_change(config.agent_id, AgentState.RUNNING)
                logger.info(f"Started agent: {config.name}")
                return True
            else:
                logger.error(f"Failed to start agent: {config.agent_id}")
                return False
                
        except Exception as e:
            logger.exception(f"Error starting agent {config.agent_id}: {e}")
            return False
    
    async def stop(self, timeout: float = 30.0) -> None:
        """Gracefully stop all agents"""
        logger.info("Stopping orchestrator...")
        self._running = False
        
        # Cancel background tasks
        for task in self._tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        
        # Stop agents in reverse priority order
        sorted_agents = sorted(
            self.agents.items(),
            key=lambda x: self.agent_configs[x[0]].priority,
            reverse=True
        )
        
        for agent_id, agent in sorted_agents:
            try:
                await asyncio.wait_for(agent.stop(), timeout=timeout)
                logger.info(f"Stopped agent: {agent_id}")
            except asyncio.TimeoutError:
                logger.warning(f"Timeout stopping agent: {agent_id}")
            except Exception as e:
                logger.exception(f"Error stopping agent {agent_id}: {e}")
        
        self._shutdown_event.set()
        logger.info("Orchestrator stopped")
    
    async def pause_agent(self, agent_id: str) -> bool:
        """Pause a specific agent"""
        agent = self.agents.get(agent_id)
        if not agent:
            return False
        
        await agent.pause()
        await self._on_state_change(agent_id, AgentState.PAUSED)
        return True
    
    async def resume_agent(self, agent_id: str) -> bool:
        """Resume a paused agent"""
        agent = self.agents.get(agent_id)
        if not agent:
            return False
        
        await agent.resume()
        await self._on_state_change(agent_id, AgentState.RUNNING)
        return True
    
    async def stop_agent(self, agent_id: str) -> bool:
        """Stop a specific agent"""
        agent = self.agents.pop(agent_id, None)
        if not agent:
            return False
        
        await agent.stop()
        await self._on_state_change(agent_id, AgentState.STOPPED)
        return True
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        return self.agents.get(agent_id)
    
    def get_metrics(self, agent_id: str) -> Optional[AgentMetrics]:
        return self.metrics.get(agent_id)
    
    def get_all_metrics(self) -> Dict[str, AgentMetrics]:
        return self.metrics.copy()
    
    def get_system_status(self) -> Dict[str, Any]:
        return {
            "running": self._running,
            "agents": {
                aid: {
                    "state": metrics.state.value,
                    "signals_generated": metrics.signals_generated,
                    "signals_executed": metrics.signals_executed,
                    "pnl": metrics.pnl,
                    "positions": metrics.positions,
                    "errors": metrics.errors,
                }
                for aid, metrics in self.metrics.items()
            },
            "total_agents": len(self.agents),
            "running_agents": sum(1 for m in self.metrics.values() if m.state == AgentState.RUNNING),
        }
    
    # Event loop - processes inter-agent events
    async def _event_loop(self) -> None:
        while True:
            try:
                event = await asyncio.wait_for(self.event_bus.get(), timeout=1.0)
                await self._process_event(event)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.exception(f"Event loop error: {e}")
    
    async def _process_event(self, event: Dict[str, Any]) -> None:
        """Process inter-agent events"""
        event_type = event.get("type")
        source_agent = event.get("source_agent")
        
        if event_type == "signal":
            # Route signal to risk manager
            await self.risk_manager.validate_signal(event)
        
        elif event_type == "fill":
            # Update metrics
            if source_agent in self.metrics:
                self.metrics[source_agent].signals_executed += 1
        
        elif event_type == "risk_alert":
            # Handle risk alerts
            await self._handle_risk_alert(event)
        
        elif event_type == "capital_request":
            # Handle capital allocation requests
            await self.capital_allocator.process_request(event)
    
    async def _handle_risk_alert(self, event: Dict[str, Any]) -> None:
        """Handle risk alerts - may pause/stop agents"""
        severity = event.get("severity", "warning")
        agent_id = event.get("agent_id")
        message = event.get("message", "")
        
        logger.warning(f"Risk alert [{severity}] for {agent_id}: {message}")
        
        if severity == "critical" and agent_id:
            await self.pause_agent(agent_id)
    
    async def _monitor_loop(self) -> None:
        """Monitor agent health and performance"""
        while True:
            try:
                await asyncio.sleep(10)
                
                for agent_id, agent in self.agents.items():
                    try:
                        status = await agent.get_status()
                        metrics = self.metrics[agent_id]
                        metrics.state = status.get("state", AgentState.UNKNOWN)
                        metrics.positions = status.get("positions", 0)
                        metrics.pnl = status.get("pnl", 0.0)
                        metrics.last_update = datetime.now()
                        
                        # Check for anomalies
                        if metrics.errors > 10:
                            logger.warning(f"Agent {agent_id} has high error count: {metrics.errors}")
                            
                    except Exception as e:
                        logger.error(f"Error monitoring {agent_id}: {e}")
                        if agent_id in self.metrics:
                            self.metrics[agent_id].errors += 1
                            
            except Exception as e:
                logger.exception(f"Monitor loop error: {e}")
    
    async def _health_check_loop(self) -> None:
        """Periodic health checks"""
        while True:
            try:
                await asyncio.sleep(30)
                
                # Check system health
                for agent_id, agent in self.agents.items():
                    if hasattr(agent, 'health_check'):
                        healthy = await agent.health_check()
                        if not healthy:
                            logger.error(f"Agent {agent_id} health check failed")
                            
            except Exception as e:
                logger.exception(f"Health check error: {e}")
    
    async def _rebalance_loop(self) -> None:
        """Periodic capital rebalancing"""
        while True:
            try:
                await asyncio.sleep(300)  # 5 minutes
                
                # Rebalance capital based on performance
                await self.capital_allocator.rebalance()
                
            except Exception as e:
                logger.exception(f"Rebalance error: {e}")
    
    async def _on_state_change(self, agent_id: str, new_state: AgentState) -> None:
        """Handle agent state changes"""
        if agent_id in self.metrics:
            self.metrics[agent_id].state = new_state
        
        for callback in self._state_change_callbacks:
            try:
                await callback(agent_id, new_state)
            except Exception as e:
                logger.error(f"State change callback error: {e}")
    
    def on_state_change(self, callback: Callable[[str, AgentState], Awaitable[None]]) -> None:
        """Register state change callback"""
        self._state_change_callbacks.append(callback)
    
    def publish_event(self, event: Dict[str, Any]) -> None:
        """Publish event to event bus (non-blocking)"""
        try:
            self.event_bus.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning("Event bus full, dropping event")


# Singleton
_orchestrator: Optional[AgentOrchestrator] = None


def get_orchestrator() -> AgentOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator()
    return _orchestrator


async def run_orchestrator() -> None:
    """Run the orchestrator with signal handling"""
    orchestrator = get_orchestrator()
    
    # Setup signal handlers
    loop = asyncio.get_running_loop()
    
    def shutdown():
        logger.info("Shutdown signal received")
        asyncio.create_task(orchestrator.stop())
    
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, shutdown)
        except NotImplementedError:
            # Windows
            pass
    
    try:
        await orchestrator.start()
        await orchestrator._shutdown_event.wait()
    except Exception as e:
        logger.exception(f"Orchestrator error: {e}")
        raise
    finally:
        await orchestrator.stop()