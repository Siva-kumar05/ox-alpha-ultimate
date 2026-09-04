"""Failover Broker Support with Multi-Broker Manager."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Callable
import time
import threading
from .brokers import BrokerBase, BrokerError, MarketDataError, OrderError, RateLimitError, make_broker, PaperBroker, DhanBroker
from .core import LOG, iso


class BrokerState(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    RECOVERING = "recovering"


@dataclass
class BrokerHealth:
    """Broker health status."""
    name: str
    state: BrokerState
    last_success: float
    last_failure: float
    consecutive_failures: int
    consecutive_successes: int
    avg_latency_ms: float
    error_rate: float
    circuit_open: bool


class CircuitBreaker:
    """Circuit breaker for broker API calls."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        success_threshold: int = 3,
        timeout_seconds: float = 60.0,
        half_open_max_calls: int = 3
    ):
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout_seconds = timeout_seconds
        self.half_open_max_calls = half_open_max_calls
        
        self._state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0
        self._lock = threading.RLock()
    
    @property
    def state(self) -> str:
        with self._lock:
            if self._state == "OPEN":
                if time.time() - self._last_failure_time > self.timeout_seconds:
                    self._state = "HALF_OPEN"
                    self._half_open_calls = 0
            return self._state
    
    def record_success(self):
        with self._lock:
            self._failure_count = 0
            if self._state == "HALF_OPEN":
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    self._state = "CLOSED"
                    self._success_count = 0
            elif self._state == "CLOSED":
                self._success_count = 0
    
    def record_failure(self):
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            self._success_count = 0
            
            if self._state == "HALF_OPEN":
                self._state = "OPEN"
            elif self._state == "CLOSED" and self._failure_count >= self.failure_threshold:
                self._state = "OPEN"
    
    def can_execute(self) -> bool:
        return self.state != "OPEN"
    
    def reset(self):
        with self._lock:
            self._state = "CLOSED"
            self._failure_count = 0
            self._success_count = 0


class FailoverBrokerManager:
    """Manages multiple brokers with automatic failover."""
    
    def __init__(self, cfg, db):
        self.cfg = cfg
        self.db = db
        self.failover_cfg = cfg.get("failover", {})
        self.enabled = self.failover_cfg.get("enabled", True)
        self.primary_broker_name = self.failover_cfg.get("primary", "dhan")
        self.failover_brokers = self.failover_cfg.get("failover", ["paper"])
        self.health_check_interval = self.failover_cfg.get("health_check_interval", 30)
        self.max_failover_attempts = self.failover_cfg.get("max_failover_attempts", 3)
        
        self._brokers: Dict[str, BrokerBase] = {}
        self._health: Dict[str, BrokerHealth] = {}
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._current_broker: Optional[str] = None
        self._lock = threading.RLock()
        self._failover_count = 0
        self._last_health_check = 0.0
        
        # Callbacks
        self._on_failover: List[Callable[[str, str], None]] = []
    
    def register_broker(self, name: str, broker: BrokerBase):
        """Register a broker."""
        with self._lock:
            self._brokers[name] = broker
            self._health[name] = BrokerHealth(
                name=name,
                state=BrokerState.HEALTHY,
                last_success=time.time(),
                last_failure=0.0,
                consecutive_failures=0,
                consecutive_successes=0,
                avg_latency_ms=0.0,
                error_rate=0.0,
                circuit_open=False
            )
            self._circuit_breakers[name] = CircuitBreaker()
    
    def initialize(self) -> bool:
        """Initialize all brokers and select primary."""
        with self._lock:
            # Initialize all brokers
            for name, broker in self._brokers.items():
                try:
                    if broker.login():
                        self._health[name].state = BrokerState.HEALTHY
                        self._health[name].last_success = time.time()
                        LOG.info(f"Broker {name} initialized successfully")
                    else:
                        self._health[name].state = BrokerState.FAILED
                        LOG.error(f"Broker {name} login failed")
                except Exception as e:
                    self._health[name].state = BrokerState.FAILED
                    LOG.error(f"Broker {name} initialization error: {e}")
            
            # Select primary
            if self.primary_broker_name in self._brokers:
                if self._health[self.primary_broker_name].state == BrokerState.HEALTHY:
                    self._current_broker = self.primary_broker_name
                else:
                    self._failover_to_next()
            
            return self._current_broker is not None
    
    def get_broker(self) -> Optional[BrokerBase]:
        """Get current active broker."""
        with self._lock:
            if self._current_broker and self._current_broker in self._brokers:
                return self._brokers[self._current_broker]
            return None
    
    def get_current_broker_name(self) -> Optional[str]:
        with self._lock:
            return self._current_broker
    
    def execute_with_failover(self, operation: Callable, *args, **kwargs):
        """Execute operation with automatic failover."""
        with self._lock:
            attempts = 0
            last_error = None
            
            while attempts < self.max_failover_attempts:
                broker_name = self._current_broker
                if not broker_name:
                    self._failover_to_next()
                    broker_name = self._current_broker
                    if not broker_name:
                        raise BrokerError("No available brokers")
                
                circuit = self._circuit_breakers.get(broker_name)
                if circuit and not circuit.can_execute():
                    LOG.warning(f"Circuit open for {broker_name}, failing over")
                    self._failover_to_next()
                    attempts += 1
                    continue
                
                broker = self._brokers[broker_name]
                start_time = time.time()
                
                try:
                    result = operation(broker, *args, **kwargs)
                    latency_ms = (time.time() - start_time) * 1000
                    self._record_success(broker_name, latency_ms)
                    if circuit:
                        circuit.record_success()
                    return result
                except (BrokerError, MarketDataError, OrderError, RateLimitError) as e:
                    latency_ms = (time.time() - start_time) * 1000
                    self._record_failure(broker_name, latency_ms, str(e))
                    if circuit:
                        circuit.record_failure()
                    last_error = e
                    LOG.warning(f"Broker {broker_name} error: {e}, attempting failover")
                    self._failover_to_next()
                    attempts += 1
                except Exception as e:
                    latency_ms = (time.time() - start_time) * 1000
                    self._record_failure(broker_name, latency_ms, str(e))
                    if circuit:
                        circuit.record_failure()
                    last_error = e
                    LOG.error(f"Unexpected error from {broker_name}: {e}")
                    self._failover_to_next()
                    attempts += 1
            
            raise BrokerError(f"All failover attempts exhausted. Last error: {last_error}")
    
    def _record_success(self, broker_name: str, latency_ms: float):
        with self._lock:
            health = self._health.get(broker_name)
            if health:
                health.last_success = time.time()
                health.consecutive_failures = 0
                health.consecutive_successes += 1
                # Exponential moving average for latency
                health.avg_latency_ms = 0.9 * health.avg_latency_ms + 0.1 * latency_ms
                health.error_rate = max(0.0, health.error_rate * 0.95)
                
                if health.state == BrokerState.DEGRADED and health.consecutive_successes >= 5:
                    health.state = BrokerState.HEALTHY
    
    def _record_failure(self, broker_name: str, latency_ms: float, error: str):
        with self._lock:
            health = self._health.get(broker_name)
            if health:
                health.last_failure = time.time()
                health.consecutive_failures += 1
                health.consecutive_successes = 0
                health.error_rate = min(1.0, health.error_rate + 0.1)
                
                if health.consecutive_failures >= 3:
                    health.state = BrokerState.DEGRADED
                if health.consecutive_failures >= 5:
                    health.state = BrokerState.FAILED
    
    def _failover_to_next(self):
        """Failover to next available broker."""
        with self._lock:
            # Try failover brokers in order
            for name in self.failover_brokers:
                if name in self._brokers and self._health[name].state in [BrokerState.HEALTHY, BrokerState.DEGRADED]:
                    if name != self._current_broker:
                        old_broker = self._current_broker
                        self._current_broker = name
                        self._failover_count += 1
                        
                        LOG.critical(f"FAILOVER: {old_broker} -> {name}")
                        self.db.ex("INSERT INTO events(kind,msg,ts)VALUES('FAILOVER',?,?)",
                                  (f"{old_broker} -> {name}", iso()))
                        
                        for callback in self._on_failover:
                            try:
                                callback(old_broker or "none", name)
                            except Exception:
                                pass
                        return
            
            # If no failover available, try primary again if it recovered
            if self.primary_broker_name in self._brokers:
                if self._health[self.primary_broker_name].state in [BrokerState.HEALTHY, BrokerState.DEGRADED]:
                    if self.primary_broker_name != self._current_broker:
                        old_broker = self._current_broker
                        self._current_broker = self.primary_broker_name
                        LOG.critical(f"FAILOVER: {old_broker} -> {self.primary_broker_name} (primary recovered)")
                        return
            
            LOG.critical("No available brokers for failover!")
    
    def on_failover(self, callback: Callable[[str, str], None]):
        """Register failover callback."""
        self._on_failover.append(callback)
    
    def health_check(self) -> Dict[str, BrokerHealth]:
        """Perform health check on all brokers."""
        with self._lock:
            now = time.time()
            if now - self._last_health_check < self.health_check_interval:
                return dict(self._health)
            
            self._last_health_check = now
            
            for name, broker in self._brokers.items():
                if name == self._current_broker:
                    continue  # Skip current broker during active trading
                
                try:
                    # Quick health check
                    start = time.time()
                    broker.ltps(self.cfg["symbols"][:1])  # Test with one symbol
                    latency = (time.time() - start) * 1000
                    
                    health = self._health[name]
                    health.avg_latency_ms = 0.9 * health.avg_latency_ms + 0.1 * latency
                    health.consecutive_successes += 1
                    health.consecutive_failures = 0
                    health.last_success = time.time()
                    
                    if health.state == BrokerState.FAILED:
                        health.state = BrokerState.RECOVERING
                    elif health.state == BrokerState.RECOVERING and health.consecutive_successes >= 3:
                        health.state = BrokerState.HEALTHY
                        
                except Exception:
                    health = self._health[name]
                    health.consecutive_failures += 1
                    health.consecutive_successes = 0
                    health.last_failure = time.time()
                    if health.state in [BrokerState.HEALTHY, BrokerState.RECOVERING]:
                        health.state = BrokerState.DEGRADED
    
    def get_health_summary(self) -> Dict:
        with self._lock:
            return {
                "current_broker": self._current_broker,
                "failover_count": self._failover_count,
                "brokers": {
                    name: {
                        "state": health.state.value,
                        "latency_ms": round(health.avg_latency_ms, 2),
                        "error_rate": round(health.error_rate, 4),
                        "circuit_open": self._circuit_breakers[name].state == "OPEN",
                        "consecutive_failures": health.consecutive_failures,
                        "consecutive_successes": health.consecutive_successes
                    }
                    for name, health in self._health.items()
                }
            }
    
    def force_failover(self, target_broker: str) -> bool:
        """Force failover to specific broker."""
        with self._lock:
            if target_broker in self._brokers:
                if self._health[target_broker].state in [BrokerState.HEALTHY, BrokerState.DEGRADED]:
                    old = self._current_broker
                    self._current_broker = target_broker
                    self._failover_count += 1
                    LOG.warning(f"MANUAL FAILOVER: {old} -> {target_broker}")
                    return True
            return False


def create_failover_manager(cfg, db) -> FailoverBrokerManager:
    """Create and initialize failover broker manager."""
    manager = FailoverBrokerManager(cfg, db)
    
    # Register primary broker
    primary_name = cfg.get("failover", {}).get("primary", "dhan")
    if primary_name == "dhan":
        manager.register_broker("dhan", DhanBroker(cfg, db))
    else:
        # Use make_broker for other types
        manager.register_broker(primary_name, make_broker(cfg, db))
    
    # Register failover brokers
    for name in cfg.get("failover", {}).get("failover", ["paper"]):
        if name == "paper":
            manager.register_broker("paper", PaperBroker(cfg, db))
        else:
            manager.register_broker(name, make_broker(cfg, db))
    
    return manager