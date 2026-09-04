"""Chaos Engineering Framework."""
from __future__ import annotations
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from enum import Enum
from .core import LOG, iso


class ChaosType(Enum):
    LATENCY = "latency"
    ERROR = "error"
    TIMEOUT = "timeout"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    NETWORK_PARTITION = "network_partition"
    CLOCK_DRIFT = "clock_drift"
    DATA_CORRUPTION = "data_corruption"


class ChaosSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ChaosExperiment:
    """Chaos experiment definition."""
    name: str
    chaos_type: ChaosType
    severity: ChaosSeverity
    target: str  # Component to target
    duration_seconds: int
    probability: float = 1.0  # Probability of injection
    parameters: Dict = field(default_factory=dict)
    enabled: bool = True


@dataclass
class ChaosResult:
    """Chaos experiment result."""
    experiment: ChaosExperiment
    start_time: str
    end_time: str
    success: bool
    observations: List[str]
    metrics_before: Dict
    metrics_after: Dict
    error: Optional[str] = None


class ChaosEngine:
    """Chaos engineering engine."""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.chaos_cfg = cfg.get("chaos_engineering", {})
        self.enabled = self.chaos_cfg.get("enabled", False)
        self.experiments: Dict[str, ChaosExperiment] = {}
        self._active_experiments: Dict[str, threading.Thread] = {}
        self._results: List[ChaosResult] = []
        self._lock = threading.RLock()
        
        # Injectors
        self._injectors: Dict[ChaosType, Callable] = {
            ChaosType.LATENCY: self._inject_latency,
            ChaosType.ERROR: self._inject_error,
            ChaosType.TIMEOUT: self._inject_timeout,
            ChaosType.RESOURCE_EXHAUSTION: self._inject_resource_exhaustion,
            ChaosType.NETWORK_PARTITION: self._inject_network_partition,
            ChaosType.CLOCK_DRIFT: self._inject_clock_drift,
            ChaosType.DATA_CORRUPTION: self._inject_data_corruption,
        }
    
    def register_experiment(self, experiment: ChaosExperiment):
        """Register a chaos experiment."""
        with self._lock:
            self.experiments[experiment.name] = experiment
    
    def run_experiment(self, name: str) -> ChaosResult:
        """Run a chaos experiment."""
        with self._lock:
            experiment = self.experiments.get(name)
            if not experiment:
                raise ValueError(f"Experiment not found: {name}")
            
            if not experiment.enabled:
                raise ValueError(f"Experiment disabled: {name}")
            
            if name in self._active_experiments:
                raise ValueError(f"Experiment already running: {name}")
        
        LOG.warning(f"CHAOS EXPERIMENT STARTED: {name}")
        
        # Collect metrics before
        metrics_before = self._collect_metrics()
        
        result = ChaosResult(
            experiment=experiment,
            start_time=iso(),
            end_time="",
            success=False,
            observations=[],
            metrics_before=metrics_before,
            metrics_after={}
        )
        
        # Run injection in thread
        stop_event = threading.Event()
        injection_thread = threading.Thread(
            target=self._run_injection,
            args=(experiment, stop_event, result),
            daemon=True
        )
        
        with self._lock:
            self._active_experiments[name] = injection_thread
        
        injection_thread.start()
        injection_thread.join(timeout=experiment.duration_seconds + 10)
        
        stop_event.set()
        
        with self._lock:
            self._active_experiments.pop(name, None)
        
        # Collect metrics after
        result.metrics_after = self._collect_metrics()
        result.end_time = iso()
        
        with self._lock:
            self._results.append(result)
        
        LOG.warning(f"CHAOS EXPERIMENT COMPLETED: {name}, success={result.success}")
        return result
    
    def _run_injection(self, experiment: ChaosExperiment, stop_event: threading.Event, result: ChaosResult):
        """Run the chaos injection."""
        injector = self._injectors.get(experiment.chaos_type)
        if not injector:
            result.error = f"No injector for {experiment.chaos_type}"
            return
        
        end_time = time.time() + experiment.duration_seconds
        
        try:
            while time.time() < end_time and not stop_event.is_set():
                # Check probability
                if random.random() < experiment.probability:
                    injector(experiment, result)
                
                # Wait before next injection
                stop_event.wait(min(1.0, experiment.duration_seconds / 10))
            
            result.success = True
            result.observations.append("Experiment completed normally")
            
        except Exception as e:
            result.success = False
            result.error = str(e)
            result.observations.append(f"Experiment failed: {e}")
    
    def _inject_latency(self, experiment: ChaosExperiment, result: ChaosResult):
        """Inject latency."""
        latency_ms = experiment.parameters.get("latency_ms", 100)
        jitter_ms = experiment.parameters.get("jitter_ms", 0)
        
        actual_latency = latency_ms + random.uniform(-jitter_ms, jitter_ms)
        time.sleep(actual_latency / 1000)
        
        result.observations.append(f"Injected latency: {actual_latency:.1f}ms")
    
    def _inject_error(self, experiment: ChaosExperiment, result: ChaosResult):
        """Inject error."""
        error_type = experiment.parameters.get("error_type", "Exception")
        error_message = experiment.parameters.get("error_message", "Chaos injected error")
        
        error_class = getattr(__builtins__, error_type, Exception)
        raise error_class(error_message)
    
    def _inject_timeout(self, experiment: ChaosExperiment, result: ChaosResult):
        """Inject timeout."""
        timeout_seconds = experiment.parameters.get("timeout_seconds", 30)
        time.sleep(timeout_seconds + 1)
        raise TimeoutError(f"Chaos injected timeout after {timeout_seconds}s")
    
    def _inject_resource_exhaustion(self, experiment: ChaosExperiment, result: ChaosResult):
        """Inject resource exhaustion."""
        resource = experiment.parameters.get("resource", "memory")
        amount = experiment.parameters.get("amount_mb", 100)
        
        if resource == "memory":
            # Allocate memory
            data = bytearray(amount * 1024 * 1024)
            time.sleep(1)
            del data
            result.observations.append(f"Allocated {amount}MB memory")
        elif resource == "cpu":
            # Busy loop
            end = time.time() + 1
            while time.time() < end:
                _ = sum(range(1000))
            result.observations.append("Consumed CPU")
    
    def _inject_network_partition(self, experiment: ChaosExperiment, result: ChaosResult):
        """Inject network partition (simulated)."""
        # This would require actual network manipulation
        # For simulation, just raise connection error
        raise ConnectionError("Chaos injected network partition")
    
    def _inject_clock_drift(self, experiment: ChaosExperiment, result: ChaosResult):
        """Inject clock drift (simulated)."""
        drift_seconds = experiment.parameters.get("drift_seconds", 1)
        # Would require actual clock manipulation
        result.observations.append(f"Simulated clock drift: {drift_seconds}s")
    
    def _inject_data_corruption(self, experiment: ChaosExperiment, result: ChaosResult):
        """Inject data corruption."""
        raise ValueError("Chaos injected data corruption")
    
    def _collect_metrics(self) -> Dict:
        """Collect system metrics."""
        # Would integrate with actual metrics
        return {
            "timestamp": iso(),
            "simulated": True
        }
    
    def stop_experiment(self, name: str) -> bool:
        """Stop a running experiment."""
        with self._lock:
            # Can't easily stop thread, but can mark for stop
            return name in self._active_experiments
    
    def get_results(self) -> List[ChaosResult]:
        with self._lock:
            return list(self._results)


# Predefined experiments for trading system
def create_trading_chaos_experiments() -> List[ChaosExperiment]:
    """Create standard chaos experiments for trading system."""
    return [
        ChaosExperiment(
            name="broker_latency",
            chaos_type=ChaosType.LATENCY,
            severity=ChaosSeverity.MEDIUM,
            target="broker",
            duration_seconds=60,
            probability=0.3,
            parameters={"latency_ms": 500, "jitter_ms": 200}
        ),
        ChaosExperiment(
            name="broker_errors",
            chaos_type=ChaosType.ERROR,
            severity=ChaosSeverity.HIGH,
            target="broker",
            duration_seconds=60,
            probability=0.1,
            parameters={"error_type": "ConnectionError", "error_message": "Broker unavailable"}
        ),
        ChaosExperiment(
            name="database_slow",
            chaos_type=ChaosType.LATENCY,
            severity=ChaosSeverity.MEDIUM,
            target="database",
            duration_seconds=120,
            probability=0.2,
            parameters={"latency_ms": 2000, "jitter_ms": 500}
        ),
        ChaosExperiment(
            name="market_data_stale",
            chaos_type=ChaosType.ERROR,
            severity=ChaosSeverity.HIGH,
            target="market_data",
            duration_seconds=60,
            probability=0.15,
            parameters={"error_type": "MarketDataError", "error_message": "Stale market data"}
        ),
        ChaosExperiment(
            name="order_timeout",
            chaos_type=ChaosType.TIMEOUT,
            severity=ChaosSeverity.CRITICAL,
            target="order_execution",
            duration_seconds=30,
            probability=0.05,
            parameters={"timeout_seconds": 10}
        ),
    ]