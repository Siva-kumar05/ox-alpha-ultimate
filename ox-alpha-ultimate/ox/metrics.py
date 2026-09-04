"""10x Observability: Metrics, alerts, and performance profiling."""
from __future__ import annotations
import time
import threading
import numpy as np
from typing import Dict, List, Any, Optional
from .core import iso


class MetricsCollector:
    """Collect and export trading metrics for monitoring."""

    def __init__(self, cfg=None):
        self.cfg = cfg or {}
        self._counters: Dict[str, float] = {}
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = {}
        self._alerts: List[Dict] = []
        self._lock = threading.RLock()
        self._thresholds = {
            "rolling_sharpe_low": -0.5,
            "daily_loss_pct": -2.0,
            "position_count_high": 5,
            "broker_error_count": 5,
        }

    def counter(self, name: str, value: float = 1.0) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + value

    def gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = value

    def histogram(self, name: str, value: float) -> None:
        with self._lock:
            self._histograms.setdefault(name, []).append(value)
            if len(self._histograms[name]) > 1000:
                self._histograms[name] = self._histograms[name][-1000:]

    def check_alerts(self, state: Dict[str, Any]) -> List[Dict]:
        alerts = []
        if state.get("rolling_sharpe", 0) < self._thresholds["rolling_sharpe_low"]:
            alerts.append({"level": "WARNING",
                          "msg": "Rolling Sharpe below threshold"})
        if state.get("daily_loss_pct", 0) < self._thresholds["daily_loss_pct"]:
            alerts.append({"level": "CRITICAL",
                          "msg": "Daily loss limit breached"})
        if state.get("position_count", 0) > self._thresholds["position_count_high"]:
            alerts.append({"level": "WARNING",
                          "msg": "Position count elevated"})
        with self._lock:
            self._alerts.extend(alerts)
        return alerts

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            hist = {}
            for name, values in self._histograms.items():
                arr = np.array(values[-100:]) if values else np.array([0.0])
                hist[name] = {
                    "count": len(values),
                    "mean": float(np.mean(arr)),
                    "p50": float(np.percentile(arr, 50)),
                    "p95": float(np.percentile(arr, 95)),
                }
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": hist,
                "recent_alerts": self._alerts[-20:],
            }

    def export_prometheus(self) -> str:
        lines = []
        with self._lock:
            for name, value in self._counters.items():
                lines.append("ox_counter_" + name + " " + str(value))
            for name, value in self._gauges.items():
                lines.append("ox_gauge_" + name + " " + str(value))
        return "\n".join(lines)


class AlertManager:
    """Configurable alert system with escalation levels."""

    def __init__(self, cfg=None):
        self.cfg = cfg or {}
        self._alerts: List[Dict] = []
        self._lock = threading.RLock()

    def check_and_alert(self, metric_name: str, value: float,
                        thresholds: Dict[str, float]) -> Optional[Dict]:
        level = None
        for tname, tval in sorted(thresholds.items(),
                                  key=lambda x: abs(x[1])):
            if value < tval:
                level = "CRITICAL"
                break
            elif value > tval * 0.5:
                level = "WARNING"
        if level:
            alert = {"metric": metric_name, "value": value,
                     "level": level, "threshold": thresholds, "ts": iso()}
            with self._lock:
                self._alerts.append(alert)
                if len(self._alerts) > 200:
                    self._alerts = self._alerts[-200:]
            return alert
        return None

    def get_recent_alerts(self, limit=20) -> List[Dict]:
        with self._lock:
            return list(self._alerts[-limit:])


class PerformanceProfiler:
    """Profile critical path execution times."""

    def __init__(self):
        self._timings: Dict[str, List[float]] = {}
        self._lock = threading.RLock()

    def record(self, operation: str, duration_ms: float):
        with self._lock:
            self._timings.setdefault(operation, []).append(duration_ms)
            if len(self._timings[operation]) > 500:
                self._timings[operation] = self._timings[operation][-500:]

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        stats = {}
        with self._lock:
            for op, times in self._timings.items():
                arr = np.array(times)
                stats[op] = {
                    "count": len(times),
                    "mean_ms": float(np.mean(arr)),
                    "p50_ms": float(np.percentile(arr, 50)),
                    "p95_ms": float(np.percentile(arr, 95)),
                    "p99_ms": float(np.percentile(arr, 99)),
                }
        return stats


class CircuitBreaker:
    """3-level circuit breaker: L1 (reduce), L2 (stop), L3 (halt)."""

    def __init__(self, cfg=None):
        cfg = cfg or {}
        shc = cfg.get("self_healing", {})
        self.l1_threshold = float(shc.get("l1_sharpe_threshold", 0.3))
        self.l2_threshold = float(shc.get("l2_sharpe_threshold", 0.0))
        self.l3_threshold = float(shc.get("l3_sharpe_threshold", -0.5))
        self.size_multiplier = float(
            shc.get("degraded_size_multiplier", 0.5))
        self.cooldown_seconds = int(
            shc.get("observation_duration_seconds", 300))
        self._state = "NORMAL"
        self._cooldown_until = 0.0

    @property
    def state(self) -> str:
        # Only L1 is transient (reduces size for an observation window then
        # recovers).  L2/HALT are sticky until a fresh evaluate() moves the
        # state, so a terminal halt must never read as NORMAL.
        if (self._state == "L1_REDUCE_SIZE"
                and time.monotonic() >= self._cooldown_until):
            return "NORMAL"
        return self._state

    def evaluate(self, rolling_sharpe: float) -> str:
        if rolling_sharpe < self.l3_threshold:
            self._state = "HALT"
            return "HALT"
        elif rolling_sharpe < self.l2_threshold:
            self._state = "L2_STOP_ENTRIES"
            return "L2_STOP_ENTRIES"
        elif rolling_sharpe < self.l1_threshold:
            self._state = "L1_REDUCE_SIZE"
            self._cooldown_until = (time.monotonic()
                                    + self.cooldown_seconds)
            return "L1_REDUCE_SIZE"
        self._state = "NORMAL"
        return "NORMAL"

    def get_size_multiplier(self) -> float:
        if self._state == "L1_REDUCE_SIZE":
            return self.size_multiplier
        elif self._state in ("L2_STOP_ENTRIES", "HALT"):
            return 0.0
        return 1.0

    def should_halt(self) -> bool:
        return self._state == "HALT"

    def should_block_entries(self) -> bool:
        return self._state in ("L2_STOP_ENTRIES", "HALT")


__all__ = ["MetricsCollector", "AlertManager",
           "PerformanceProfiler", "CircuitBreaker"]
