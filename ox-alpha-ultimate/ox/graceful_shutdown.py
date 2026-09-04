"""Graceful Shutdown Handling."""
from __future__ import annotations
import signal
import sys
import threading
import time
import atexit
from typing import Callable, List, Optional
from dataclasses import dataclass
from enum import Enum
from .core import LOG


class ShutdownPhase(Enum):
    RUNNING = "running"
    SHUTTING_DOWN = "shutting_down"
    STOPPING_SERVICES = "stopping_services"
    FLUSHING_DATA = "flushing_data"
    FINALIZING = "finalizing"
    STOPPED = "stopped"


@dataclass
class ShutdownHook:
    """Shutdown hook with priority."""
    name: str
    callback: Callable[[], None]
    priority: int  # Lower = runs first
    timeout_seconds: float = 30.0
    critical: bool = False  # If True, failure aborts shutdown


class GracefulShutdownManager:
    """Manages graceful shutdown with proper cleanup ordering."""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.shutdown_cfg = cfg.get("graceful_shutdown", {})
        self.enabled = self.shutdown_cfg.get("enabled", True)
        self.shutdown_timeout = self.shutdown_cfg.get("timeout_seconds", 60)
        self.force_timeout = self.shutdown_cfg.get("force_timeout_seconds", 10)
        
        self._phase = ShutdownPhase.RUNNING
        self._hooks: List[ShutdownHook] = []
        self._shutdown_requested = False
        self._shutdown_complete = False
        self._lock = threading.RLock()
        self._shutdown_thread: Optional[threading.Thread] = None
        
        # Register signal handlers
        self._register_signals()
        atexit.register(self._atexit_handler)
    
    def _register_signals(self):
        """Register signal handlers."""
        if not self.enabled:
            return
        
        def signal_handler(signum, frame):
            signal_name = signal.Signals(signum).name
            LOG.info(f"Received signal {signal_name}, initiating graceful shutdown")
            self.request_shutdown()
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        # Handle SIGHUP for config reload (separate)
        try:
            signal.signal(signal.SIGHUP, signal_handler)
        except AttributeError:
            pass  # Windows doesn't have SIGHUP
    
    def _atexit_handler(self):
        """Handle exit."""
        if not self._shutdown_complete:
            self.shutdown()
    
    def register_hook(
        self,
        name: str,
        callback: Callable[[], None],
        priority: int = 50,
        timeout_seconds: float = 30.0,
        critical: bool = False
    ):
        """Register a shutdown hook."""
        with self._lock:
            hook = ShutdownHook(
                name=name,
                callback=callback,
                priority=priority,
                timeout_seconds=timeout_seconds,
                critical=critical
            )
            self._hooks.append(hook)
            # Sort by priority (lower first)
            self._hooks.sort(key=lambda h: h.priority)
    
    def request_shutdown(self, reason: str = "signal"):
        """Request graceful shutdown."""
        with self._lock:
            if self._shutdown_requested:
                return
            self._shutdown_requested = True
            self._phase = ShutdownPhase.SHUTTING_DOWN
        
        LOG.critical(f"SHUTDOWN REQUESTED: {reason}")
        
        # Run shutdown in separate thread to avoid blocking signal handler
        self._shutdown_thread = threading.Thread(target=self._execute_shutdown, daemon=False)
        self._shutdown_thread.start()
    
    def _execute_shutdown(self):
        """Execute all shutdown hooks."""
        start_time = time.time()
        
        try:
            # Phase 1: Stopping services
            self._phase = ShutdownPhase.STOPPING_SERVICES
            self._run_hooks("stopping_services")
            
            # Phase 2: Flushing data
            self._phase = ShutdownPhase.FLUSHING_DATA
            self._run_hooks("flushing_data")
            
            # Phase 3: Finalizing
            self._phase = ShutdownPhase.FINALIZING
            self._run_hooks("finalizing")
            
            self._phase = ShutdownPhase.STOPPED
            self._shutdown_complete = True
            
            elapsed = time.time() - start_time
            LOG.info(f"Graceful shutdown completed in {elapsed:.1f}s")
            
        except Exception as e:
            LOG.critical(f"Shutdown error: {e}")
            self._phase = ShutdownPhase.STOPPED
            self._shutdown_complete = True
    
    def _run_hooks(self, phase: str):
        """Run hooks for a specific phase."""
        # Filter hooks for this phase (based on name or priority ranges)
        phase_hooks = []
        for hook in self._hooks:
            if phase in hook.name.lower() or hook.name.lower() == "all":
                phase_hooks.append(hook)
        
        # If no phase-specific hooks, run all non-critical hooks
        if not phase_hooks:
            phase_hooks = [h for h in self._hooks if not h.critical]
        
        for hook in phase_hooks:
            try:
                LOG.info(f"Running shutdown hook: {hook.name}")
                start = time.time()
                
                # Run with timeout
                self._run_with_timeout(hook)

                elapsed = time.time() - start
                LOG.info(f"Hook {hook.name} completed in {elapsed:.1f}s")
                
            except TimeoutError:
                LOG.error(f"Hook {hook.name} timed out after {hook.timeout_seconds}s")
                if hook.critical:
                    raise
            except Exception as e:
                LOG.error(f"Hook {hook.name} failed: {e}")
                if hook.critical:
                    raise
    
    def _run_with_timeout(self, hook: ShutdownHook):
        """Run hook with timeout."""
        result_container = [None]
        exception_container = [None]
        
        def target():
            try:
                result_container[0] = hook.callback()
            except Exception as e:
                exception_container[0] = e
        
        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout=hook.timeout_seconds)
        
        if thread.is_alive():
            raise TimeoutError(f"Hook {hook.name} exceeded {hook.timeout_seconds}s timeout")
        
        if exception_container[0]:
            raise exception_container[0]
        
        return result_container[0]
    
    def wait_for_shutdown(self, timeout: Optional[float] = None) -> bool:
        """Wait for shutdown to complete."""
        if self._shutdown_thread:
            self._shutdown_thread.join(timeout=timeout or self.shutdown_timeout)
        return self._shutdown_complete
    
    def is_shutting_down(self) -> bool:
        return self._shutdown_requested
    
    def get_phase(self) -> ShutdownPhase:
        return self._phase
    
    def force_shutdown(self):
        """Force immediate shutdown."""
        LOG.critical("FORCE SHUTDOWN")
        self._phase = ShutdownPhase.STOPPED
        self._shutdown_complete = True
        sys.exit(1)
    
    def shutdown(self) -> None:
        """Execute all shutdown hooks in priority order."""
        if self._shutdown_complete:
            return
        self._phase = ShutdownPhase.SHUTTING_DOWN
        self._shutdown_requested = True
        
        # Sort hooks by priority (lower = runs first)
        sorted_hooks = sorted(self._hooks, key=lambda h: h.priority)
        
        for hook in sorted_hooks:
            try:
                LOG.info(f"Running shutdown hook: {hook.name}")
                hook.callback()
            except Exception as e:
                LOG.error(f"Shutdown hook {hook.name} failed: {e}")
                if hook.critical:
                    raise
        
        self._phase = ShutdownPhase.STOPPED
        self._shutdown_complete = True
        LOG.info("Graceful shutdown completed")

def create_shutdown_manager(cfg, agent=None, db=None, broker=None, oms=None) -> GracefulShutdownManager:
    """Create shutdown manager with standard hooks."""
    manager = GracefulShutdownManager(cfg)
    
    # Register standard hooks
    if oms:
        manager.register_hook(
            "stopping_services:oms_kill",
            lambda: oms.kill_switch("graceful_shutdown"),
            priority=10,
            critical=True
        )
    
    if broker:
        manager.register_hook(
            "stopping_services:broker_stop",
            lambda: broker.stop_orderflow() if hasattr(broker, 'stop_orderflow') else None,
            priority=20
        )
    
    if db:
        manager.register_hook(
            "flushing_data:db_sync",
            lambda: db.c.commit() if hasattr(db, 'c') else None,
            priority=30
        )
        manager.register_hook(
            "finalizing:db_close",
            lambda: db.close() if hasattr(db, 'close') else None,
            priority=90,
            critical=True
        )
    
    if agent:
        manager.register_hook(
            "flushing_data:agent_eod",
            lambda: agent.eod() if hasattr(agent, 'eod') else None,
            priority=40
        )
    
    return manager