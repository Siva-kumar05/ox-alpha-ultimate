"""Structured JSON Logging and Observability."""
from __future__ import annotations
import json
import logging
import sys
import time
import threading
from datetime import datetime
from typing import Dict, Any, Optional


class JSONFormatter(logging.Formatter):
    """JSON log formatter with structured fields."""
    
    def __init__(self, service_name: str = "ox-alpha"):
        super().__init__()
        self.service_name = service_name
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "service": self.service_name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add extra fields
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)
        
        # Add exception info
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add correlation ID if present
        if hasattr(record, "correlation_id"):
            log_entry["correlation_id"] = record.correlation_id
        
        # Add trace ID if present
        if hasattr(record, "trace_id"):
            log_entry["trace_id"] = record.trace_id
        
        return json.dumps(log_entry, default=str)


class StructuredLogger:
    """Structured logger with context support."""
    
    def __init__(self, name: str, service_name: str = "ox-alpha"):
        self.logger = logging.getLogger(name)
        self.service_name = service_name
        self._context: Dict[str, Any] = {}
        self._lock = threading.RLock()
    
    def bind(self, **kwargs) -> "StructuredLogger":
        """Create a new logger with additional context."""
        new_logger = StructuredLogger(self.logger.name, self.service_name)
        with self._lock:
            new_logger._context = {**self._context, **kwargs}
        return new_logger
    
    def _log(self, level: int, message: str, **kwargs):
        extra = {"extra_fields": {**self._context, **kwargs}}
        self.logger.log(level, message, extra=extra)
    
    def debug(self, message: str, **kwargs):
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        self._log(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        self._log(logging.CRITICAL, message, **kwargs)
    
    def exception(self, message: str, **kwargs):
        extra = {"extra_fields": {**self._context, **kwargs}}
        self.logger.exception(message, extra=extra)


def setup_structured_logging(
    log_path: str = "oxalpha.log",
    service_name: str = "ox-alpha",
    level: int = logging.INFO,
    json_output: bool = True
) -> logging.Logger:
    """Setup structured JSON logging."""
    logger = logging.getLogger("ox")
    logger.setLevel(level)
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    if json_output:
        file_handler.setFormatter(JSONFormatter(service_name))
    else:
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s"
        ))
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    if json_output:
        console_handler.setFormatter(JSONFormatter(service_name))
    else:
        console_handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s"
        ))
    logger.addHandler(console_handler)
    
    logger.propagate = False
    
    return logger


class LatencyTracker:
    """Tracks latency for operations."""
    
    def __init__(self):
        self._timings: Dict[str, list] = {}
        self._lock = threading.RLock()
    
    def record(self, operation: str, duration_ms: float):
        with self._lock:
            if operation not in self._timings:
                self._timings[operation] = []
            self._timings[operation].append(duration_ms)
            # Keep last 1000 measurements
            if len(self._timings[operation]) > 1000:
                self._timings[operation] = self._timings[operation][-1000:]
    
    def get_stats(self, operation: str) -> Optional[Dict[str, float]]:
        with self._lock:
            if operation not in self._timings or not self._timings[operation]:
                return None
            values = self._timings[operation]
            return {
                "count": len(values),
                "mean_ms": sum(values) / len(values),
                "min_ms": min(values),
                "max_ms": max(values),
                "p50_ms": sorted(values)[len(values) // 2],
                "p95_ms": sorted(values)[int(len(values) * 0.95)],
                "p99_ms": sorted(values)[int(len(values) * 0.99)],
            }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            return {op: self.get_stats(op) for op in self._timings}


class LatencyContext:
    """Context manager for tracking operation latency."""
    
    def __init__(self, tracker: LatencyTracker, operation: str):
        self.tracker = tracker
        self.operation = operation
        self.start_time = 0.0
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.perf_counter() - self.start_time) * 1000
        self.tracker.record(self.operation, duration_ms)


# Global instances
_latency_tracker = LatencyTracker()
_structured_logger = None


def get_latency_tracker() -> LatencyTracker:
    return _latency_tracker


def get_structured_logger(name: str = "ox") -> StructuredLogger:
    global _structured_logger
    if _structured_logger is None:
        _structured_logger = StructuredLogger(name)
    return _structured_logger


def track_latency(operation: str):
    """Decorator for tracking function latency."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with LatencyContext(_latency_tracker, operation):
                return func(*args, **kwargs)
        return wrapper
    return decorator