"""Request Tracing and Distributed Tracing."""
from __future__ import annotations
import threading
import time
import uuid
from contextvars import ContextVar
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


# Context variables for trace propagation
trace_id_var: ContextVar[Optional[str]] = ContextVar("trace_id", default=None)
span_id_var: ContextVar[Optional[str]] = ContextVar("span_id", default=None)
parent_span_id_var: ContextVar[Optional[str]] = ContextVar("parent_span_id", default=None)


@dataclass
class Span:
    """Single trace span."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "ok"
    error: Optional[str] = None
    
    def finish(self, status: str = "ok", error: Optional[str] = None):
        """Finish the span."""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.status = status
        self.error = error
    
    def set_tag(self, key: str, value: Any):
        self.tags[key] = value
    
    def log(self, message: str, **fields):
        self.logs.append({
            "timestamp": time.time(),
            "message": message,
            **fields
        })


class Tracer:
    """Distributed tracer."""
    
    def __init__(self, service_name: str = "ox-alpha"):
        self.service_name = service_name
        self._spans: Dict[str, Span] = {}
        self._completed_spans: List[Span] = []
        self._lock = threading.RLock()
        self._max_spans = 10000
    
    def start_span(
        self,
        operation: str,
        trace_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None
    ) -> Span:
        """Start a new span."""
        # Generate or use provided trace ID
        if trace_id is None:
            trace_id = trace_id_var.get() or str(uuid.uuid4())[:16]
            trace_id_var.set(trace_id)
        
        # Generate span ID
        span_id = str(uuid.uuid4())[:16]
        
        # Set parent span ID
        if parent_span_id is None:
            parent_span_id = span_id_var.get()
        
        # Set current span ID
        span_id_var.set(span_id)
        parent_span_id_var.set(parent_span_id)
        
        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation=operation,
            start_time=time.time(),
            tags=tags or {}
        )
        span.set_tag("service", self.service_name)
        
        with self._lock:
            self._spans[span_id] = span
        
        return span
    
    def finish_span(self, span: Span, status: str = "ok", error: Optional[str] = None):
        """Finish a span."""
        span.finish(status, error)
        
        with self._lock:
            self._spans.pop(span.span_id, None)
            self._completed_spans.append(span)
            
            # Trim completed spans
            if len(self._completed_spans) > self._max_spans:
                self._completed_spans = self._completed_spans[-self._max_spans:]
        
        # Restore parent span context
        if span.parent_span_id:
            span_id_var.set(span.parent_span_id)
        else:
            span_id_var.set(None)
    
    def get_trace(self, trace_id: str) -> List[Span]:
        """Get all spans for a trace."""
        with self._lock:
            return [
                s for s in self._completed_spans
                if s.trace_id == trace_id
            ]
    
    def get_recent_spans(self, limit: int = 100) -> List[Span]:
        """Get recent completed spans."""
        with self._lock:
            return self._completed_spans[-limit:]
    
    def get_active_spans(self) -> List[Span]:
        """Get currently active spans."""
        with self._lock:
            return list(self._spans.values())
    
    def export_json(self) -> List[Dict]:
        """Export spans as JSON."""
        with self._lock:
            return [
                {
                    "trace_id": s.trace_id,
                    "span_id": s.span_id,
                    "parent_span_id": s.parent_span_id,
                    "operation": s.operation,
                    "start_time": s.start_time,
                    "end_time": s.end_time,
                    "duration_ms": s.duration_ms,
                    "tags": s.tags,
                    "logs": s.logs,
                    "status": s.status,
                    "error": s.error
                }
                for s in self._completed_spans[-1000:]
            ]


class TraceContext:
    """Context manager for automatic span management."""
    
    def __init__(self, tracer: Tracer, operation: str, trace_id: Optional[str] = None, **tags):
        self.tracer = tracer
        self.operation = operation
        self.trace_id = trace_id
        self.tags = tags
        self.span: Optional[Span] = None
    
    def __enter__(self) -> Span:
        self.span = self.tracer.start_span(self.operation, trace_id=self.trace_id, tags=self.tags)
        return self.span
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.span:
            if exc_type:
                self.tracer.finish_span(self.span, status="error", error=str(exc_val))
            else:
                self.tracer.finish_span(self.span, status="ok")


# Global tracer
_global_tracer: Optional[Tracer] = None


def get_tracer(service_name: str = "ox-alpha") -> Tracer:
    """Get global tracer."""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = Tracer(service_name)
    return _global_tracer


def trace(operation: str, **tags):
    """Decorator for tracing function calls."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            with TraceContext(tracer, operation, **tags) as span:
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    span.set_tag("error", True)
                    span.log("exception", error=str(e))
                    raise
        return wrapper
    return decorator


class RequestTracer:
    """HTTP request tracer for API endpoints."""
    
    def __init__(self, tracer: Tracer):
        self.tracer = tracer
    
    def trace_request(self, method: str, path: str, request_id: Optional[str] = None) -> TraceContext:
        """Trace an HTTP request."""
        return TraceContext(
            self.tracer,
            f"HTTP {method} {path}",
            trace_id=request_id or trace_id_var.get(),
            http_method=method,
            http_path=path,
            http_request_id=request_id
        )
    
    def trace_broker_call(self, broker: str, method: str, symbol: Optional[str] = None) -> TraceContext:
        """Trace a broker API call."""
        return TraceContext(
            self.tracer,
            f"Broker {broker}.{method}",
            broker=broker,
            broker_method=method,
            symbol=symbol
        )
    
    def trace_db_query(self, query: str, table: Optional[str] = None) -> TraceContext:
        """Trace a database query."""
        # Truncate query for tag
        query_tag = query[:100] if len(query) > 100 else query
        return TraceContext(
            self.tracer,
            "DB Query",
            db_query=query_tag,
            db_table=table
        )
    
    def trace_strategy_eval(self, strategy_id: str) -> TraceContext:
        """Trace strategy evaluation."""
        return TraceContext(
            self.tracer,
            f"Strategy Eval: {strategy_id}",
            strategy_id=strategy_id
        )