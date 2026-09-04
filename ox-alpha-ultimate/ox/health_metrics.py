"""Health Checks, Metrics Export, and Observability Endpoints."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import time
import threading
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from .core import LOG, iso
from .metrics import MetricsCollector


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Individual health check result."""
    name: str
    status: HealthStatus
    message: str
    latency_ms: float
    timestamp: str
    details: Dict = field(default_factory=dict)


@dataclass
class SystemHealth:
    """Overall system health."""
    status: HealthStatus
    checks: List[HealthCheck]
    timestamp: str
    uptime_seconds: float
    version: str = "1.0.0"


class HealthChecker:
    """Manages health checks for the system."""
    
    def __init__(self, cfg, db, agent=None):
        self.cfg = cfg
        self.db = db
        self.agent = agent
        self.health_cfg = cfg.get("health_checks", {})
        self.enabled = self.health_cfg.get("enabled", True)
        self.check_interval = self.health_cfg.get("interval_seconds", 30)
        self.timeout_seconds = self.health_cfg.get("timeout_seconds", 5)
        
        self._checks: Dict[str, Callable[[], HealthCheck]] = {}
        self._last_results: List[HealthCheck] = []
        self._last_check_time = 0.0
        self._start_time = time.time()
        self._lock = threading.RLock()
        
        # Register default checks
        self._register_default_checks()
    
    def _register_default_checks(self):
        """Register default health checks."""
        self.register_check("database", self._check_database)
        self.register_check("broker_connection", self._check_broker)
        self.register_check("data_freshness", self._check_data_freshness)
        self.register_check("disk_space", self._check_disk_space)
        self.register_check("memory", self._check_memory)
        self.register_check("circuit_breaker", self._check_circuit_breaker)
        self.register_check("kill_switch", self._check_kill_switch)
    
    def register_check(self, name: str, check_fn: Callable[[], HealthCheck]):
        """Register a health check function."""
        with self._lock:
            self._checks[name] = check_fn
    
    def run_checks(self) -> SystemHealth:
        """Run all health checks."""
        if not self.enabled:
            return SystemHealth(
                status=HealthStatus.UNKNOWN,
                checks=[],
                timestamp=iso(),
                uptime_seconds=time.time() - self._start_time
            )
        
        now = time.time()
        if now - self._last_check_time < self.check_interval and self._last_results:
            # Return cached results
            return SystemHealth(
                status=self._compute_overall_status(self._last_results),
                checks=self._last_results,
                timestamp=iso(),
                uptime_seconds=time.time() - self._start_time
            )
        
        self._last_check_time = now
        results = []
        
        for name, check_fn in self._checks.items():
            try:
                start = time.time()
                result = check_fn()
                result.latency_ms = (time.time() - start) * 1000
                result.timestamp = iso()
                results.append(result)
            except Exception as e:
                results.append(HealthCheck(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check failed: {e}",
                    latency_ms=0.0,
                    timestamp=iso(),
                    details={"error": str(e)}
                ))
        
        self._last_results = results
        return SystemHealth(
            status=self._compute_overall_status(results),
            checks=results,
            timestamp=iso(),
            uptime_seconds=time.time() - self._start_time
        )
    
    def _compute_overall_status(self, checks: List[HealthCheck]) -> HealthStatus:
        if not checks:
            return HealthStatus.UNKNOWN
        
        if any(c.status == HealthStatus.UNHEALTHY for c in checks):
            return HealthStatus.UNHEALTHY
        if any(c.status == HealthStatus.DEGRADED for c in checks):
            return HealthStatus.DEGRADED
        return HealthStatus.HEALTHY
    
    def _check_database(self) -> HealthCheck:
        """Check database connectivity."""
        try:
            start = time.time()
            self.db.q("SELECT 1")
            latency = (time.time() - start) * 1000
            return HealthCheck(
                name="database",
                status=HealthStatus.HEALTHY if latency < 100 else HealthStatus.DEGRADED,
                message=f"Database responsive in {latency:.1f}ms",
                latency_ms=latency,
                timestamp=iso(),
                details={"latency_ms": latency}
            )
        except Exception as e:
            return HealthCheck(
                name="database",
                status=HealthStatus.UNHEALTHY,
                message=f"Database error: {e}",
                latency_ms=0.0,
                timestamp=iso(),
                details={"error": str(e)}
            )
    
    def _check_broker(self) -> HealthCheck:
        """Check broker connectivity."""
        if not self.agent or not hasattr(self.agent, 'broker'):
            return HealthCheck(
                name="broker_connection",
                status=HealthStatus.UNKNOWN,
                message="Agent not available",
                latency_ms=0.0,
                timestamp=iso()
            )
        
        try:
            start = time.time()
            # Quick LTP check
            symbols = self.cfg["symbols"][:1]
            self.agent.broker.ltps(symbols)
            latency = (time.time() - start) * 1000
            
            return HealthCheck(
                name="broker_connection",
                status=HealthStatus.HEALTHY if latency < 500 else HealthStatus.DEGRADED,
                message=f"Broker responsive in {latency:.1f}ms",
                latency_ms=latency,
                timestamp=iso(),
                details={"latency_ms": latency, "broker": self.agent.broker.name}
            )
        except Exception as e:
            return HealthCheck(
                name="broker_connection",
                status=HealthStatus.UNHEALTHY,
                message=f"Broker error: {e}",
                latency_ms=0.0,
                timestamp=iso(),
                details={"error": str(e)}
            )
    
    def _check_data_freshness(self) -> HealthCheck:
        """Check data freshness."""
        try:
            max_age = self.cfg["execution"].get("max_data_staleness_seconds", 10)
            stale_count = 0
            
            for symbol in self.cfg["symbols"]:
                rows = self.db.q(
                    "SELECT ts FROM candles WHERE sym=? ORDER BY ts DESC LIMIT 1",
                    (symbol,)
                )
                if rows:
                    last_ts = rows[0][0]
                    age = time.time() - last_ts
                    if age > max_age:
                        stale_count += 1
            
            if stale_count == 0:
                return HealthCheck(
                    name="data_freshness",
                    status=HealthStatus.HEALTHY,
                    message="All data fresh",
                    latency_ms=0.0,
                    timestamp=iso()
                )
            elif stale_count < len(self.cfg["symbols"]) / 2:
                return HealthCheck(
                    name="data_freshness",
                    status=HealthStatus.DEGRADED,
                    message=f"{stale_count} symbols have stale data",
                    latency_ms=0.0,
                    timestamp=iso()
                )
            else:
                return HealthCheck(
                    name="data_freshness",
                    status=HealthStatus.UNHEALTHY,
                    message="Majority of symbols have stale data",
                    latency_ms=0.0,
                    timestamp=iso()
                )
        except Exception as e:
            return HealthCheck(
                name="data_freshness",
                status=HealthStatus.UNHEALTHY,
                message=f"Data freshness check failed: {e}",
                latency_ms=0.0,
                timestamp=iso()
            )
    
    def _check_disk_space(self) -> HealthCheck:
        """Check disk space."""
        import shutil
        try:
            db_path = self.cfg.get("db_path", "oxalpha.db")
            total, used, free = shutil.disk_usage(db_path)
            free_pct = free / total * 100
            
            if free_pct > 20:
                status = HealthStatus.HEALTHY
            elif free_pct > 10:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.UNHEALTHY
            
            return HealthCheck(
                name="disk_space",
                status=status,
                message=f"Disk free: {free_pct:.1f}%",
                latency_ms=0.0,
                timestamp=iso(),
                details={"free_gb": free / 1e9, "total_gb": total / 1e9, "free_pct": free_pct}
            )
        except Exception as e:
            return HealthCheck(
                name="disk_space",
                status=HealthStatus.UNHEALTHY,
                message=f"Disk check failed: {e}",
                latency_ms=0.0,
                timestamp=iso()
            )
    
    def _check_memory(self) -> HealthCheck:
        """Check memory usage."""
        try:
            import psutil
            process = psutil.Process()
            mem = process.memory_info()
            mem_mb = mem.rss / 1e6
            
            # Also check system memory
            sys_mem = psutil.virtual_memory()
            sys_free_pct = 100 - sys_mem.percent
            
            if sys_free_pct > 20 and mem_mb < 500:
                status = HealthStatus.HEALTHY
            elif sys_free_pct > 10 and mem_mb < 1000:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.UNHEALTHY
            
            return HealthCheck(
                name="memory",
                status=status,
                message=f"Process: {mem_mb:.0f}MB, System free: {sys_free_pct:.1f}%",
                latency_ms=0.0,
                timestamp=iso(),
                details={"process_mb": mem_mb, "system_free_pct": sys_free_pct}
            )
        except Exception as e:
            return HealthCheck(
                name="memory",
                status=HealthStatus.UNKNOWN,
                message=f"Memory check unavailable: {e}",
                latency_ms=0.0,
                timestamp=iso()
            )
    
    def _check_circuit_breaker(self) -> HealthCheck:
        """Check circuit breaker status."""
        if not self.agent or not hasattr(self.agent, 'circuit_breaker'):
            return HealthCheck(
                name="circuit_breaker",
                status=HealthStatus.UNKNOWN,
                message="Circuit breaker not available",
                latency_ms=0.0,
                timestamp=iso()
            )
        
        cb = self.agent.circuit_breaker
        state = cb.state
        
        if state == "NORMAL":
            status = HealthStatus.HEALTHY
        elif state == "L1_REDUCE_SIZE":
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.UNHEALTHY
        
        return HealthCheck(
            name="circuit_breaker",
            status=status,
            message=f"Circuit breaker: {state}",
            latency_ms=0.0,
            timestamp=iso(),
            details={"state": state, "size_multiplier": cb.get_size_multiplier()}
        )
    
    def _check_kill_switch(self) -> HealthCheck:
        """Check kill switch status."""
        from .core import Path
        kill_path = Path(self.cfg.root) / "KILL.flag"
        
        if kill_path.exists():
            return HealthCheck(
                name="kill_switch",
                status=HealthStatus.UNHEALTHY,
                message="KILL.flag present",
                latency_ms=0.0,
                timestamp=iso()
            )
        
        return HealthCheck(
            name="kill_switch",
            status=HealthStatus.HEALTHY,
            message="No kill switch",
            latency_ms=0.0,
            timestamp=iso()
        )


class MetricsExporter:
    """Exports metrics in various formats."""
    
    def __init__(self, metrics_collector: MetricsCollector, cfg):
        self.metrics = metrics_collector
        self.cfg = cfg
        self.export_cfg = cfg.get("metrics_export", {})
        self.enabled = self.export_cfg.get("enabled", True)
        self.prometheus_enabled = self.export_cfg.get("prometheus", True)
        self.json_enabled = self.export_cfg.get("json", True)
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        return self.metrics.export_prometheus()
    
    def export_json(self) -> Dict[str, Any]:
        """Export metrics as JSON."""
        snapshot = self.metrics.snapshot()
        snapshot["timestamp"] = iso()
        return snapshot
    
    def export_all(self) -> Dict[str, Any]:
        """Export all metrics in all formats."""
        return {
            "prometheus": self.export_prometheus() if self.prometheus_enabled else "",
            "json": self.export_json() if self.json_enabled else {},
            "timestamp": iso()
        }


class HealthCheckServer:
    """HTTP server for health checks and metrics."""
    
    def __init__(self, health_checker: HealthChecker, metrics_exporter: MetricsExporter, port: int = 8080):
        self.health_checker = health_checker
        self.metrics_exporter = metrics_exporter
        self.port = port
        self.server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
    
    def start(self):
        """Start the HTTP server."""
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/health":
                    self._handle_health()
                elif self.path == "/health/ready":
                    self._handle_ready()
                elif self.path == "/health/live":
                    self._handle_live()
                elif self.path == "/metrics":
                    self._handle_metrics()
                elif self.path == "/metrics/prometheus":
                    self._handle_prometheus()
                else:
                    self.send_response(404)
                    self.end_headers()
            
            def _handle_health(self):
                health = self.server.health_checker.run_checks()
                self.send_response(200 if health.status != HealthStatus.UNHEALTHY else 503)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "status": health.status.value,
                    "checks": [
                        {
                            "name": c.name,
                            "status": c.status.value,
                            "message": c.message,
                            "latency_ms": c.latency_ms,
                            "timestamp": c.timestamp,
                            "details": c.details
                        }
                        for c in health.checks
                    ],
                    "uptime_seconds": health.uptime_seconds,
                    "timestamp": health.timestamp
                }).encode())
            
            def _handle_ready(self):
                health = self.server.health_checker.run_checks()
                ready = health.status != HealthStatus.UNHEALTHY
                self.send_response(200 if ready else 503)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"ready": ready}).encode())
            
            def _handle_live(self):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"alive": True}).encode())
            
            def _handle_metrics(self):
                metrics = self.server.metrics_exporter.export_json()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(metrics).encode())
            
            def _handle_prometheus(self):
                prom = self.server.metrics_exporter.export_prometheus()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.end_headers()
                self.wfile.write(prom.encode())
            
            def log_message(self, format, *args):
                pass  # Suppress default logging
        
        Handler.health_checker = self.health_checker
        Handler.metrics_exporter = self.metrics_exporter
        
        self.server = HTTPServer(("0.0.0.0", self.port), Handler)
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self._thread.start()
        LOG.info(f"Health check server started on port {self.port}")
    
    def stop(self):
        """Stop the HTTP server."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        if self._thread:
            self._thread.join(timeout=5)