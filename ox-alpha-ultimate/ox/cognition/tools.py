"""Dynamic tool ecosystem: synthesis, registry, escalation chains, DAG execution.

Tools are synthesized from natural-language specs into sandboxed pure-Python
source, statically validated (AST allowlist — no imports, no I/O, no
dunders), then registered and reused within and across sessions. Execution
failures walk an escalation chain informed by failure memory ("if grep
fails, try find"), and batch work runs as a dependency graph with
fan-out/fan-in rather than sequential calls.
"""

from __future__ import annotations

import ast
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

_FORBIDDEN_NODES = (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal)
_FORBIDDEN_CALLS = {"open", "eval", "exec", "compile", "__import__", "input", "globals", "locals"}
_ALLOWED_BUILTINS = {"abs", "min", "max", "sum", "len", "round", "sorted", "range", "enumerate",
                     "zip", "float", "int", "str", "bool", "list", "dict", "set", "tuple", "any", "all"}
_SAFE_BUILTINS = {name: __builtins__[name] if not isinstance(__builtins__, dict) else __builtins__[name]
                  for name in _ALLOWED_BUILTINS if name in (
                      __builtins__ if isinstance(__builtins__, dict) else vars(__builtins__))}

import math
import statistics

_SAFE_GLOBALS = {"__builtins__": _SAFE_BUILTINS, "math": math, "statistics": statistics}


class UnsafeToolError(ValueError):
    """Raised when synthesized tool source violates the sandbox policy."""


def validate_source(source: str) -> None:
    """Static allowlist check: single pure function, no imports, no I/O, no escape hatches."""
    tree = ast.parse(source)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(functions) != 1:
        raise UnsafeToolError("synthesized tool must define exactly one function")
    for node in ast.walk(tree):
        if isinstance(node, _FORBIDDEN_NODES):
            raise UnsafeToolError(f"forbidden construct in synthesized tool: {type(node).__name__}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise UnsafeToolError("dunder attribute access is not allowed")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_CALLS:
            raise UnsafeToolError(f"forbidden call: {node.func.id}")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise UnsafeToolError("dunder name access is not allowed")


@dataclass
class SynthTool:
    name: str
    description: str
    fn: Callable
    source: str
    uses: int = 0
    successes: int = 0


class ToolSynthesizer:
    """Compile a spec like 'sum_of_squares: sum the squares of a list of numbers'
    into a registered, sandboxed callable. The function body is expected in the
    spec as Python source after a '->' marker; description precedes it."""

    def __init__(self, store_dir: str | Path = ".ox-alpha/tools"):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.registry: dict[str, SynthTool] = {}

    def synthesize(self, spec: str) -> SynthTool:
        if "->" not in spec:
            raise UnsafeToolError("spec must be 'name: description -> def fn(...): ...'")
        head, source = spec.split("->", 1)
        name, _, description = head.strip().partition(":")
        name = name.strip()
        if not name.isidentifier():
            raise UnsafeToolError(f"tool name {name!r} is not a valid identifier")
        source = textwrap.dedent(source).strip()
        validate_source(source)
        namespace: dict[str, Any] = dict(_SAFE_GLOBALS)
        exec(compile(source, f"<synth:{name}>", "exec"), namespace)  # noqa: S102 - validated allowlist
        fn = next(v for k, v in namespace.items() if callable(v) and not k.startswith("_"))
        tool = SynthTool(name=name, description=description.strip() or name, fn=fn, source=source)
        self.registry[name] = tool
        (self.store_dir / f"synth_{name}.py").write_text(source, encoding="utf-8")
        return tool

    def get(self, name: str) -> SynthTool | None:
        return self.registry.get(name)

    def call(self, name: str, *args, **kwargs) -> Any:
        tool = self.registry.get(name)
        if tool is None:
            raise KeyError(f"tool {name} is not registered")
        try:
            result = tool.fn(*args, **kwargs)
            tool.uses += 1
            tool.successes += 1
            return result
        except Exception:
            tool.uses += 1
            raise

    def list(self) -> list[dict]:
        return [{"name": t.name, "description": t.description, "uses": t.uses,
                 "success_rate": round(t.successes / t.uses, 3) if t.uses else None}
                for t in self.registry.values()]


DEFAULT_ESCALATIONS: dict[str, list[str]] = {
    "grep": ["rg --no-heading", "findstr /s /i", "python_re_scan"],
    "pip_install": ["pip install --user", "pip install --index-url alt", "manual_wheel"],
    "read_file": ["read_text_utf8", "read_bytes_decode", "read_chunked"],
    "http_get": ["retry_backoff", "alternate_mirror", "cache_fallback"],
}


class EscalationChain:
    """Try fallbacks in order when a primary strategy fails, consulting memory
    for a fix that worked before."""

    def __init__(self, strategies: dict[str, list[Callable[[], Any]]] | None = None, memory=None):
        self.strategies = strategies or {}
        self.memory = memory

    def register(self, name: str, fns: list[Callable[[], Any]]) -> None:
        self.strategies[name] = fns

    def run(self, name: str) -> tuple[Any, int]:
        fns = self.strategies.get(name) or []
        last_error: Exception | None = None
        for index, fn in enumerate(fns):
            try:
                return fn(), index
            except Exception as exc:  # noqa: BLE001 - chain must try every fallback
                last_error = exc
                if self.memory is not None:
                    self.memory.record_failure(name, str(exc), {"attempt": index, "strategy": index})
        if self.memory is not None and last_error is not None:
            raise last_error
        raise RuntimeError(f"no strategy registered for {name!r}")


@dataclass
class DAGNode:
    name: str
    fn: Callable[[dict], Any]
    depends_on: list[str] = field(default_factory=list)
    retries: int = 0
    fallback: Callable[[dict], Any] | None = None


class DAGExecutor:
    """Dependency-graph execution with parallel fan-out and barrier fan-in.

    Nodes whose dependencies are satisfied run concurrently on a thread pool;
    a node that exhausts retries falls back to its declared fallback before
    the graph is declared failed. Cycles are rejected up front."""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers

    def _validate(self, nodes: dict[str, DAGNode]) -> list[str]:
        state: dict[str, int] = {}

        def visit(name: str) -> None:
            match state.get(name, 0):
                case 1:
                    raise ValueError(f"dependency cycle at {name}")
                case 2:
                    return
            state[name] = 1
            for dep in nodes[name].depends_on:
                if dep not in nodes:
                    raise ValueError(f"unknown dependency {dep} of {name}")
                visit(dep)
            state[name] = 2

        for name in nodes:
            visit(name)
        ready = [n for n in nodes if not nodes[n].depends_on]
        if not ready:
            raise ValueError("DAG has no root nodes")
        return ready

    def execute(self, nodes: list[DAGNode]) -> dict:
        graph = {n.name: n for n in nodes}
        self._validate(graph)
        results: dict[str, Any] = {}
        failures: dict[str, str] = {}
        remaining = set(graph)
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            while remaining:
                runnable = [n for name, n in graph.items()
                            if name in remaining and all(d in results for d in n.depends_on)]
                if not runnable:
                    return {"ok": False, "results": results, "failures": failures,
                            "error": "deadlocked or unsatisfiable nodes", "completed": sorted(results)}
                futures = {}
                for node in runnable:
                    futures[pool.submit(self._run_node, node, results)] = node
                for future in as_completed(futures):
                    node = futures[future]
                    outcome, error = future.result()
                    if outcome is _FAILED:
                        failures[node.name] = error or "unknown"
                        remaining.discard(node.name)
                    else:
                        results[node.name] = outcome
                        remaining.discard(node.name)
        return {"ok": not failures, "results": results, "failures": failures,
                "completed": sorted(results)}

    def _run_node(self, node: DAGNode, results: dict) -> tuple[Any, str | None]:
        attempts = [node.fn] * (node.retries + 1) + ([node.fallback] if node.fallback else [])
        last_error = ""
        for fn in attempts:
            try:
                return fn(results), None
            except Exception as exc:  # noqa: BLE001
                last_error = f"{exc.__class__.__name__}: {exc}"
        return _FAILED, last_error


class _Failed:
    pass


_FAILED = _Failed()


def map_reduce(items: list[Any], mapper: Callable[[Any], Any], reducer: Callable[[list[Any]], Any],
               max_workers: int = 4) -> Any:
    """Classic fan-out/fan-in over a batch: map in parallel, reduce once."""
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        mapped = list(pool.map(mapper, items))
    return reducer(mapped)
