"""Planning: MCTS over action sequences plus a backtracking checkpointing executor.

The planner explores candidate action sequences with UCT tree search and a
caller-supplied cheap simulator, so a robust plan is chosen *before* anything
touches the real world. The executor then runs the chosen plan step by step,
trying per-step alternatives on failure (backtracking) instead of abandoning
the whole plan, and journals checkpoints so an interrupted task resumes
instead of restarting.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

PlanStepFn = Callable[[dict], dict]          # state -> {'ok': bool, 'reward': float, 'state': dict}
Simulator = Callable[[list[str], dict], float]  # (actions, state) -> simulated reward


@dataclass
class Step:
    name: str
    fn: PlanStepFn
    alternatives: list[PlanStepFn] = field(default_factory=list)
    retries: int = 1
    depends_on: list[str] = field(default_factory=list)


class Plan:
    def __init__(self, name: str, steps: list[Step], initial_state: dict | None = None):
        self.name = name
        self.steps = {s.name: s for s in steps}
        self.order = [s.name for s in steps]
        self.initial_state = dict(initial_state or {})

    def topological_order(self) -> list[str]:
        ordered, seen, temp = [], set(), set()

        def visit(node: str) -> None:
            if node in seen:
                return
            if node in temp:
                raise ValueError(f"Plan dependency cycle at {node}")
            temp.add(node)
            for dep in self.steps[node].depends_on:
                visit(dep)
            temp.discard(node)
            seen.add(node)
            ordered.append(node)

        for name in self.order:
            visit(name)
        return ordered


@dataclass
class _Node:
    actions: tuple[str, ...]
    parent: "_Node | None" = None
    children: list["_Node"] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0


class MCTSPlanner:
    """UCT search over action sequences scored by a simulator.

    ``action_space`` maps a state to candidate next actions; ``simulator``
    returns a reward for a full action sequence. Search picks robust plans
    (mean reward over rollouts), not lucky single paths.
    """

    def __init__(self, action_space: Callable[[dict], list[str]], simulator: Simulator,
                 exploration: float = 1.41, rollout_depth: int = 6, seed: int | None = None):
        self.action_space = action_space
        self.simulator = simulator
        self.exploration = exploration
        self.rollout_depth = rollout_depth
        self.rng = random.Random(seed)

    def search(self, state: dict, iterations: int = 200) -> tuple[list[str], float]:
        root = _Node(actions=())
        for _ in range(iterations):
            node, sim_state = self._select(root, dict(state))
            node = self._expand(node, sim_state)
            reward = self._rollout(list(node.actions), sim_state)
            self._backprop(node, reward)
        best = max(root.children, key=lambda ch: (ch.value, ch.visits)) if root.children else root
        return list(best.actions), round(best.value / max(best.visits, 1), 4)

    def _select(self, node: _Node, state: dict) -> tuple[_Node, dict]:
        while node.children and all(ch.visits > 0 for ch in node.children):
            total = node.visits
            node = max(node.children, key=lambda ch: (
                ch.value / ch.visits + self.exploration * math.sqrt(math.log(total + 1) / ch.visits)))
            state = self._apply(state, node.actions[-1])
        return node, state

    def _expand(self, node: _Node, state: dict) -> _Node:
        if not node.actions and node.parent is None or node.children:
            return node
        for action in self.action_space(state):
            child = _Node(actions=node.actions + (action,), parent=node)
            node.children.append(child)
        return self.rng.choice(node.children) if node.children else node

    def _rollout(self, actions: list[str], state: dict) -> float:
        sim_state = dict(state)
        sequence = list(actions)
        while len(sequence) < self.rollout_depth:
            options = self.action_space(sim_state)
            if not options:
                break
            pick = self.rng.choice(options)
            sequence.append(pick)
            sim_state = self._apply(sim_state, pick)
        return float(self.simulator(sequence, sim_state))

    def _apply(self, state: dict, action: str) -> dict:
        state = dict(state)
        state.setdefault("history", []).append(action)  # type: ignore[union-attr]
        return state

    def _backprop(self, node: _Node, reward: float) -> None:
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent


class StalePlanError(RuntimeError):
    """Raised when resuming a checkpoint whose plan definition changed."""


class BacktrackingExecutor:
    """Executes a Plan with per-step alternatives, retry, checkpoint, resume.

    On failure the executor first retries, then tries each declared
    alternative for the step, and only then fails the plan — recording which
    step and which recovery succeeded so the pattern is learnable.
    """

    def __init__(self, checkpoint_dir: str | Path = ".ox-alpha/checkpoints", on_failure=None):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.on_failure = on_failure  # callback(step, error) -> None

    def _checkpoint_path(self, plan: Plan) -> Path:
        return self.checkpoint_dir / f"{plan.name}.json"

    def save(self, plan: Plan, state: dict, completed: list[str]) -> None:
        payload = {"plan": plan.name, "order": plan.order, "completed": completed,
                   "state": state, "ts": time.time()}
        self._checkpoint_path(plan).write_text(json.dumps(payload, default=str), encoding="utf-8")

    def load(self, plan: Plan) -> dict | None:
        path = self._checkpoint_path(plan)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("plan") != plan.name or payload.get("order") != plan.order:
            raise StalePlanError(f"Checkpoint for {plan.name} does not match current plan definition")
        return payload

    def clear(self, plan: Plan) -> None:
        self._checkpoint_path(plan).unlink(missing_ok=True)

    def execute(self, plan: Plan, *, resume: bool = True) -> dict:
        state = dict(plan.initial_state)
        completed: list[str] = []
        if resume:
            saved = self.load(plan)
            if saved:
                completed = list(saved["completed"])
                state.update(saved.get("state") or {})
        started = time.monotonic()
        for name in plan.topological_order():
            if name in completed:
                continue
            step = plan.steps[name]
            attempts = [(name, fn) for fn in [step.fn, *step.alternatives] for _ in range(step.retries)]
            outcome = None
            last_error: str | None = None
            for label, fn in attempts:
                try:
                    result = fn(state)
                    if isinstance(result, dict) and result.get("ok"):
                        outcome = result
                        break
                    last_error = str((result or {}).get("error", "step reported failure"))
                except Exception as exc:  # noqa: BLE001 - executor must survive step crashes
                    last_error = f"{exc.__class__.__name__}: {exc}"
                if self.on_failure:
                    self.on_failure(label, last_error or "unknown")
            if outcome is None:
                self.save(plan, state, completed)
                return {"ok": False, "failed_at": name, "error": last_error,
                        "completed": completed, "checkpointed": True}
            state.update(outcome.get("state", {}))
            completed.append(name)
            self.save(plan, state, completed)
        self.clear(plan)
        return {"ok": True, "completed": completed, "state": state,
                "elapsed": round(time.monotonic() - started, 4)}
