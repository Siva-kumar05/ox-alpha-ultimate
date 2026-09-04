"""Load Testing and Benchmarking."""
from __future__ import annotations
import threading
import time
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from .core import LOG, iso


@dataclass
class BenchmarkResult:
    """Benchmark result."""
    name: str
    operations: int
    duration_seconds: float
    ops_per_second: float
    latency_ms: Dict[str, float]  # min, max, mean, p50, p95, p99
    errors: int
    error_rate: float
    timestamp: str


@dataclass
class LoadTestConfig:
    """Load test configuration."""
    name: str
    target_func: Callable
    concurrency: int
    duration_seconds: int
    ramp_up_seconds: int = 0
    args: tuple = ()
    kwargs: Dict = field(default_factory=dict)


class LoadTester:
    """Load testing framework."""
    
    def __init__(self):
        self._results: List[BenchmarkResult] = []
        self._lock = threading.RLock()
    
    def run_load_test(self, config: LoadTestConfig) -> BenchmarkResult:
        """Run a load test."""
        latencies = []
        errors = 0
        completed = 0
        start_time = time.time()
        end_time = start_time + config.duration_seconds
        
        # Ramp up
        if config.ramp_up_seconds > 0:
            ramp_start = time.time()
            while time.time() - ramp_start < config.ramp_up_seconds:
                time.sleep(0.1)
        
        def worker():
            nonlocal completed, errors
            while time.time() < end_time:
                op_start = time.time()
                try:
                    config.target_func(*config.args, **config.kwargs)
                    latency_ms = (time.time() - op_start) * 1000
                    latencies.append(latency_ms)
                except Exception:
                    errors += 1
                finally:
                    completed += 1
        
        # Run workers
        with ThreadPoolExecutor(max_workers=config.concurrency) as executor:
            futures = [executor.submit(worker) for _ in range(config.concurrency)]
            for future in as_completed(futures):
                future.result()  # Propagate exceptions
        
        actual_duration = time.time() - start_time
        
        if latencies:
            latencies.sort()
            latency_stats = {
                "min": latencies[0],
                "max": latencies[-1],
                "mean": statistics.mean(latencies),
                "p50": latencies[len(latencies) // 2],
                "p95": latencies[int(len(latencies) * 0.95)],
                "p99": latencies[int(len(latencies) * 0.99)],
            }
        else:
            latency_stats = {"min": 0, "max": 0, "mean": 0, "p50": 0, "p95": 0, "p99": 0}
        
        result = BenchmarkResult(
            name=config.name,
            operations=completed,
            duration_seconds=actual_duration,
            ops_per_second=completed / actual_duration if actual_duration > 0 else 0,
            latency_ms=latency_stats,
            errors=errors,
            error_rate=errors / completed if completed > 0 else 0,
            timestamp=iso()
        )
        
        with self._lock:
            self._results.append(result)
        
        return result
    
    def run_benchmark_suite(self, configs: List[LoadTestConfig]) -> List[BenchmarkResult]:
        """Run multiple load tests."""
        results = []
        for config in configs:
            LOG.info(f"Running load test: {config.name}")
            result = self.run_load_test(config)
            results.append(result)
            LOG.info(f"  {result.ops_per_second:.1f} ops/sec, p95: {result.latency_ms['p95']:.1f}ms")
        return results
    
    def get_results(self) -> List[BenchmarkResult]:
        with self._lock:
            return list(self._results)


class BenchmarkRunner:
    """Runs predefined benchmarks for the trading system."""
    
    def __init__(self, agent):
        self.agent = agent
        self.load_tester = LoadTester()
    
    def benchmark_ltp_fetch(self, symbols: List[str], duration: int = 30) -> BenchmarkResult:
        """Benchmark LTP fetching."""
        def fetch_ltps():
            return self.agent.broker.ltps(symbols)
        
        config = LoadTestConfig(
            name="ltp_fetch",
            target_func=fetch_ltps,
            concurrency=5,
            duration_seconds=duration
        )
        return self.load_tester.run_load_test(config)
    
    def benchmark_order_flow_assessment(self, symbols: List[str], duration: int = 30) -> BenchmarkResult:
        """Benchmark order flow assessment."""
        def assess_flow():
            results = {}
            for sym in symbols:
                results[sym] = self.agent.broker.order_flow(sym)
            return results
        
        config = LoadTestConfig(
            name="order_flow_assessment",
            target_func=assess_flow,
            concurrency=3,
            duration_seconds=duration
        )
        return self.load_tester.run_load_test(config)
    
    def benchmark_strategy_evaluation(self, symbols: List[str], duration: int = 30) -> BenchmarkResult:
        """Benchmark strategy evaluation."""
        def eval_strategies():
            for sym in symbols:
                frame = self.agent.frame(sym, self.agent.cfg["execution"]["signal_history_candles"])
                if frame is not None:
                    self.agent._vote_details(frame)
        
        config = LoadTestConfig(
            name="strategy_evaluation",
            target_func=eval_strategies,
            concurrency=2,
            duration_seconds=duration
        )
        return self.load_tester.run_load_test(config)
    
    def benchmark_full_tick(self, duration: int = 60) -> BenchmarkResult:
        """Benchmark full tick cycle."""
        def full_tick():
            self.agent.tick_once()
        
        config = LoadTestConfig(
            name="full_tick_cycle",
            target_func=full_tick,
            concurrency=1,  # Sequential
            duration_seconds=duration
        )
        return self.load_tester.run_load_test(config)
    
    def run_full_benchmark(self) -> List[BenchmarkResult]:
        """Run full benchmark suite."""
        symbols = self.agent.cfg["symbols"]
        
        configs = [
            LoadTestConfig(
                name="ltp_fetch",
                target_func=lambda: self.agent.broker.ltps(symbols),
                concurrency=10,
                duration_seconds=30
            ),
            LoadTestConfig(
                name="order_flow",
                target_func=lambda: {s: self.agent.broker.order_flow(s) for s in symbols},
                concurrency=5,
                duration_seconds=30
            ),
            LoadTestConfig(
                name="strategy_voting",
                target_func=lambda: [
                    self.agent._vote_details(
                        self.agent.frame(s, self.agent.cfg["execution"]["signal_history_candles"])
                    ) for s in symbols if self.agent.frame(s, self.agent.cfg["execution"]["signal_history_candles"]) is not None
                ],
                concurrency=2,
                duration_seconds=30
            ),
            LoadTestConfig(
                name="risk_check",
                target_func=lambda: self.agent.risk.approve(
                    symbols[0], "BUY", 100, 1000.0, [], 1.0
                ),
                concurrency=10,
                duration_seconds=30
            ),
        ]
        
        return self.load_tester.run_benchmark_suite(configs)


def create_benchmark_runner(agent) -> BenchmarkRunner:
    return BenchmarkRunner(agent)