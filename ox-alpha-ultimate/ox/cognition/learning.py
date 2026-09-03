"""Continuous self-improvement: failure autopsy, preference learning, skill extraction.

Every error is classified, matched against past failures, and — once a fix is
observed to work — promoted into procedural memory and an importable skill
file. Preferences are learned implicitly from (produced, corrected) pairs
(e.g. the user re-indenting 4-space output to 2 spaces), so the next session
starts personalised instead of from zero.
"""

from __future__ import annotations

import difflib
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .memory import MemoryStore

FAILURE_TAXONOMY = {
    "network": ("timeout", "connection", "reset", "unreachable", "dns", "refused"),
    "permission": ("permission", "forbidden", "unauthorized", "403", "401", "access denied"),
    "schema": ("schema", "unexpected key", "missing key", "malformed", "parse", "decode"),
    "notfound": ("no such file", "not found", "404", "missing"),
    "logic": ("valueerror", "assertion", "invalid state", "off by"),
    "ratelimit": ("rate limit", "429", "too many requests"),
    "dependency": ("modulenotfound", "importerror", "no module", "version"),
}


def classify_failure(error: str) -> str:
    lowered = str(error).lower()
    for category, markers in FAILURE_TAXONOMY.items():
        if any(marker in lowered for marker in markers):
            return category
    return "unknown"


@dataclass
class Autopsy:
    tool: str
    error: str
    category: str
    similar_past: list[dict]
    recommended_fix: str | None


class FailureAutopsy:
    def __init__(self, memory: MemoryStore):
        self.memory = memory

    def perform(self, tool: str, error: str, context: dict | None = None) -> Autopsy:
        category = classify_failure(error)
        similar = self.memory.similar_failures(tool, error)
        past_fix = next((f["fix"] for f in similar if f.get("resolved") and f.get("fix")), None)
        self.memory.record_failure(tool, error, context or {})
        recommendation = past_fix or _heuristic_fix(category, tool, error)
        return Autopsy(tool=tool, error=str(error), category=category,
                       similar_past=similar, recommended_fix=recommendation)

    def confirm_fix(self, tool: str, error: str, fix: str, pattern_name: str | None = None) -> None:
        """A fix worked: mark matching failures resolved and learn the pattern."""
        for failure in self.memory.similar_failures(tool, error, limit=10):
            if not failure["resolved"]:
                self.memory.resolve_failure(failure["id"], fix)
        name = pattern_name or f"fix:{tool}:{classify_failure(error)}"
        self.memory.learn_pattern(name, {"tool": tool, "fix": fix}, success=True)


def _heuristic_fix(category: str, tool: str, error: str) -> str:
    table = {
        "network": "retry with exponential backoff, then fail over to a mirror/cache",
        "permission": "check credentials scope and path ACLs before retrying",
        "schema": "pin the parser to the observed shape; validate before use",
        "notfound": "verify the path/identifier exists before the next attempt",
        "logic": "add a failing unit test reproducing the error, then fix the logic",
        "ratelimit": "back off exponentially and cache the response",
        "dependency": "pin a known-good version and reinstall into the project env",
    }
    return table.get(category, f"inspect {tool} failure: {error}")


class PreferenceLearner:
    """Implicit preference mining from (agent-produced, user-corrected) pairs."""

    INDENT = re.compile(r"^( +)\S")
    QUOTE = re.compile(r"(['\"])")

    def __init__(self, memory: MemoryStore, min_observations: int = 2):
        self.memory = memory
        self.min_observations = min_observations

    def observe_edit(self, produced: str, corrected: str) -> dict:
        diffs = list(difflib.unified_diff(produced.splitlines(), corrected.splitlines(), n=0))
        signals: Counter = Counter()
        for line in diffs:
            if line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
                body = line[1:]
                if not body.strip():
                    continue
                added = line.startswith("+")
                match = self.INDENT.match(body)
                if match:
                    width = len(match.group(1))
                    if width in (2, 4, 8):
                        signals[f"indent:{'2-space' if width == 2 else f'{width}-space'}" if added else "indent:remove"] += 1
                if self.QUOTE.search(body):
                    signals[f"quotes:{'single' if chr(39) in body else 'double'}" if added else "quotes:strip"] += 1
        learned = {}
        for signal, count in signals.items():
            if count >= max(self.min_observations, 2):
                key = f"pref:{signal.split(':')[0]}"
                self.memory.set_fact(key, signal.split(":", 1)[1], confidence=min(1.0, count / 5.0))
                learned[key] = signal
        return learned

    def snapshot(self) -> dict:
        return {key: self.memory.get_fact(key) for key in
                ("pref:indent", "pref:quotes") if self.memory.get_fact(key) is not None}


class SkillExtractor:
    """Generalise a successful workflow into a reusable SKILL.md file."""

    TEMPLATE = """---
name: {name}
description: {description}
source: auto-extracted
confidence: {confidence}
---

# {name}

## When to use
{trigger}

## Procedure
{steps}

## Evidence
{evidence}
"""

    def __init__(self, skills_dir: str | Path = ".ox-alpha/skills"):
        self.skills_dir = Path(skills_dir)
        self.skills_dir.mkdir(parents=True, exist_ok=True)

    def extract(self, name: str, trigger: str, steps: list[str], evidence: dict | None = None,
                confidence: float = 0.7, memory: MemoryStore | None = None) -> Path:
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        body = self.TEMPLATE.format(
            name=name,
            description=trigger[:200],
            confidence=round(confidence, 2),
            trigger=trigger,
            steps="\n".join(f"{i}. {step}" for i, step in enumerate(steps, 1)),
            evidence="\n".join(f"- {k}: {v}" for k, v in (evidence or {"source": "observed success"}).items()),
        )
        path = self.skills_dir / f"{slug}.md"
        path.write_text(body, encoding="utf-8")
        if memory is not None:
            memory.learn_pattern(f"skill:{slug}", {"trigger": trigger, "steps": steps,
                                                   "path": str(path)}, success=True)
            memory.record_episodic("skill_extracted", {"name": name, "path": str(path)},
                                   tags=["skill", "procedural"], importance=0.7)
        return path
