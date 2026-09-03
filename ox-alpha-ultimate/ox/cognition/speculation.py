"""Speculative execution: parallel what-if branches with best-branch promotion.

Instead of a single chain of thought, the engine launches N independent
branches in sandboxed sub-contexts (gap #4), evaluates each against
objectives, and promotes only the highest-scoring branch to the main thread.
This is the structural cure for zero-lookahead decisions.
"""

from __future__ import annotations

import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class BranchResult:
    branch_id: int
    name: str
    ok: bool
    output: Any = None
    error: str | None = None
    score: float = 0.0
    elapsed: float = 0.0
    trace: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "branch": self.branch_id, "name": self.name, "ok": self.ok,
            "output": str(self.output)[:2000] if self.output is not None else None,
            "error": self.error, "score": self.score, "elapsed": round(self.elapsed, 4),
            "trace": self.trace[:50],
        }


@dataclass
class SpeculativeBranch:
    branch_id: int
    name: str
    fn: Callable[[dict], tuple[Any, float]]  # (context) -> (output, score)
    context: dict
    timeout: float = 30.0


class SpeculativeEngine:
    """Run N candidate branches in parallel, pick the best.

    Each branch runs in its own sandboxed context; exceptions are caught and
    scored negatively so a crashing branch never blocks selection. The
    winner is promoted atomically back to the caller.
    """

    def __init__(self, max_workers: int = 4, timeout_seconds: float = 60.0):
        self.max_workers = max_workers
        self.timeout = timeout_seconds
        self.history: list[list[BranchResult]] = []

    def run(self, branches: list[SpeculativeBranch]) -> dict:
        """Execute all branches concurrently. Return winner + full ranking."""
        if not branches:
            return {"ok": False, "error": "no branches supplied"}
        results: dict[int, BranchResult] = {}
        lock = threading.Lock()
        completed = threading.Event()
        pending = len(branches)

        def _runner(branch: SpeculativeBranch) -> BranchResult:
            start = time.monotonic()
            try:
                output, score = branch.fn(dict(branch.context))
                elapsed = time.monotonic() - start
                return BranchResult(
                    branch_id=branch.branch_id, name=branch.name, ok=True,
                    output=output, score=float(score), elapsed=elapsed)
            except Exception as exc:  # noqa: BLE001
                elapsed = time.monotonic() - start
                tb = traceback.format_exc(limit=3).splitlines()
                return BranchResult(
                    branch_id=branch.branch_id, name=branch.name, ok=False,
                    error=f"{exc.__class__.__name__}: {exc}",
                    score=-10.0, elapsed=elapsed, trace=tb)

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures: dict[Future, SpeculativeBranch] = {}
            for branch in branches:
                f = pool.submit(_runner, branch)
                futures[f] = branch
            for future in futures:
                branch = futures[future]
                try:
                    result = future.result(timeout=branch.timeout)
                except TimeoutError:
                    result = BranchResult(
                        branch_id=branch.branch_id, name=branch.name, ok=False,
                        error=f"timeout after {branch.timeout:.1f}s",
                        score=-5.0, elapsed=branch.timeout)
                except Exception as exc:  # noqa: BLE001
                    result = BranchResult(
                        branch_id=branch.branch_id, name=branch.name, ok=False,
                        error=f"executor: {exc.__class__.__name__}: {exc}",
                        score=-8.0, elapsed=0.0)
                with lock:
                    results[branch.branch_id] = result
                    pending -= 1
                    if pending == 0:
                        completed.set()

        ranked = sorted(results.values(), key=lambda r: r.score, reverse=True)
        self.history.append(ranked)
        if len(self.history) > 20:
            self.history.pop(0)
        winner = ranked[0]
        return {
            "ok": True,
            "winner": winner.to_dict(),
            "ranking": [r.to_dict() for r in ranked],
            "branches_run": len(ranked),
            "successful_branches": sum(1 for r in ranked if r.ok),
        }

    def what_if(self, state: dict, candidates: list[tuple[str, Callable[[dict], tuple[Any, float]]]],
                *, score_threshold: float = 0.0) -> dict:
        """Convenience: build branches from (name, fn) pairs and run them."""
        branches = [
            SpeculativeBranch(branch_id=i, name=name, fn=fn, context=state, timeout=self.timeout)
            for i, (name, fn) in enumerate(candidates)
        ]
        result = self.run(branches)
        if result.get("ok") and result["winner"]["score"] < score_threshold:
            result["winner_promoted"] = False
            result["warning"] = f"no branch exceeded score threshold {score_threshold}"
        else:
            result["winner_promoted"] = result.get("ok", False)
        return result

    def best_strategy_histogram(self) -> dict[str, int]:
        """Aggregate which branch names have won most often — feeds procedural learning."""
        counts: dict[str, int] = {}
        for run in self.history:
            if run:
                counts[run[0].name] = counts.get(run[0].name, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
