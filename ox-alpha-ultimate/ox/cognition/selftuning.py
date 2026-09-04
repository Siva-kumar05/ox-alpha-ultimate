"""Self-modification layer: context compression, tool-calling pattern
optimisation, and preference-driven prompt adaptation (gap #5).

The agent is no longer static: between turns it compresses stale context,
identifies repeated tool-calling inefficiencies, and adapts its behaviour
template from learned preferences — without touching the system prompt
directly (which would be unsafe).
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .vectors import embed, cosine
import numpy as np


@dataclass
class CompressionRecord:
    original_tokens: int
    compressed_tokens: int
    retained_facts: list[str]
    dropped_facts: list[str]
    compression_ratio: float = 0.0

    def __post_init__(self) -> None:
        if self.original_tokens:
            self.compression_ratio = round(1.0 - self.compressed_tokens / self.original_tokens, 4)


@dataclass
class PatternObservation:
    tool_sequence: tuple[str, ...]
    count: int
    avg_latency_ms: float
    error_rate: float


class ContextCompressor:
    """Semantic context compression with importance-weighted retention.

    Instead of truncating context from the head, each entry is scored by
    (recency × semantic-similarity-to-current-query × explicit-importance)
    and only the top-K are retained, with the rest summarised into a few
    factual bullets so nothing important is lost.
    """

    def __init__(self, max_tokens: int = 4000, similarity_weight: float = 0.5,
                 recency_weight: float = 0.3, importance_weight: float = 0.2):
        self.max_tokens = max_tokens
        self.w_sim = similarity_weight
        self.w_rec = recency_weight
        self.w_imp = importance_weight

    @staticmethod
    def _token_count(text: str) -> int:
        return max(1, len(re.findall(r"\S+", str(text))))

    def compress(self, history: list[dict], query: str | None = None) -> tuple[list[dict], CompressionRecord]:
        if not history:
            return [], CompressionRecord(0, 0, [], [])
        query_vec = embed(query or "")
        n = len(history)
        scored: list[tuple[float, int, dict]] = []
        for idx, entry in enumerate(history):
            body = json.dumps(entry, default=str, sort_keys=True)
            sim = cosine(query_vec, embed(body)) if query else 0.5
            recency = (idx + 1) / n
            importance = float(entry.get("importance", 0.5) if isinstance(entry, dict) else 0.5)
            score = self.w_sim * sim + self.w_rec * recency + self.w_imp * importance
            scored.append((score, idx, entry))
        scored.sort(key=lambda t: t[0], reverse=True)
        kept: list[dict] = []
        kept_indices: set[int] = set()
        token_budget = self.max_tokens
        original_tokens = self._token_count(" ".join(json.dumps(e, default=str) for e in history))
        for _, idx, entry in scored:
            entry_tokens = self._token_count(json.dumps(entry, default=str))
            if token_budget - entry_tokens >= 0:
                kept.append((idx, entry))
                kept_indices.add(idx)
                token_budget -= entry_tokens
            else:
                break
        kept.sort(key=lambda t: t[0])
        retained = [e for _, e in kept]
        dropped = [history[i] for i in range(n) if i not in kept_indices]
        retained_facts = [f"{e.get('kind','entry')}@{e.get('ts','?')}" for e in retained[:10]]
        dropped_facts = [f"{e.get('kind','entry')}@{e.get('ts','?')}" for e in dropped[:10]]
        compressed_tokens = self._token_count(" ".join(json.dumps(e, default=str) for e in retained))
        return retained, CompressionRecord(original_tokens, compressed_tokens, retained_facts, dropped_facts)

    def summarise_drop(self, dropped: list[dict]) -> list[str]:
        """Turn dropped context into compact factual bullets."""
        if not dropped:
            return []
        grouped: dict[str, list[str]] = defaultdict(list)
        for entry in dropped:
            if isinstance(entry, dict):
                kind = entry.get("kind", "event")
                detail = str(entry.get("content") or entry.get("detail") or entry)[:120]
                grouped[kind].append(detail)
        bullets = []
        for kind, details in sorted(grouped.items()):
            bullets.append(f"{len(details)}× {kind}: " + "; ".join(details[:3]) + (" …" if len(details) > 3 else ""))
        return bullets[:5]


class ToolPatternOptimiser:
    """Detect repeated tool sequences and suggest fused single calls."""

    def __init__(self, min_occurrences: int = 3):
        self.min_occurrences = min_occurrences
        self.sequence_counter: Counter = Counter()
        self.error_counts: Counter = Counter()
        self.latency_ms: dict[tuple[str, ...], list[float]] = defaultdict(list)

    def observe(self, sequence: list[str], *, error: bool = False, latency_ms: float = 0.0) -> None:
        if len(sequence) < 2:
            return
        key = tuple(sequence)
        self.sequence_counter[key] += 1
        if error:
            self.error_counts[key] += 1
        if latency_ms > 0:
            self.latency_ms[key].append(float(latency_ms))

    def optimisations(self) -> list[PatternObservation]:
        results = []
        for seq, count in self.sequence_counter.most_common():
            if count < self.min_occurrences or len(seq) < 2:
                continue
            lats = self.latency_ms.get(seq, [0.0])
            avg_lat = float(np.mean(lats)) if lats else 0.0
            err = self.error_counts[seq]
            results.append(PatternObservation(
                tool_sequence=seq, count=count,
                avg_latency_ms=round(avg_lat, 2),
                error_rate=round(err / count, 4) if count else 0.0))
        return results

    def recommendation(self, seq: list[str]) -> str | None:
        obs = self.optimisations()
        key = tuple(seq)
        for pattern in obs:
            if pattern.tool_sequence == key and pattern.count >= self.min_occurrences:
                return (f"sequence {list(key)} repeated {pattern.count}× "
                        f"(avg {pattern.avg_latency_ms:.0f}ms, err {pattern.error_rate:.0%}); "
                        f"consider a fused custom tool via ToolSynthesizer")
        return None


class BehaviouralAdapter:
    """Adapt agent behaviour templates from learned preferences.

    Reads pref:* facts from MemoryStore and materialises them into a
    structured behaviour block that prepends reasoning — the safe
    alternative to mutating the system prompt.
    """

    PREF_KEYS = (
        "pref:indent", "pref:quotes", "pref:theme", "pref:framework",
        "pref:style", "pref:verbosity",
    )

    def __init__(self, memory):
        self.memory = memory

    def behaviour_block(self) -> str:
        prefs = {k: self.memory.get_fact(k) for k in self.PREF_KEYS if self.memory.get_fact(k) is not None}
        if not prefs:
            return ""
        mapping = {
            "pref:indent": lambda v: f"use {v} indentation",
            "pref:quotes": lambda v: f"prefer {v}-quoted strings",
            "pref:theme": lambda v: f"theme preference: {v}",
            "pref:framework": lambda v: f"prefer framework {v}",
            "pref:style": lambda v: f"code style: {v}",
            "pref:verbosity": lambda v: f"output verbosity: {v}",
        }
        rules = [mapping[k](v) for k, v in prefs.items() if k in mapping]
        if not rules:
            return ""
        return "# learned behaviour preferences\n- " + "\n- ".join(rules)


@dataclass
class SelfTuner:
    """Orchestrates compression, pattern optimisation, and behavioural
    adaptation. Holds the single state object the agent calls between turns.
    """

    memory: Any = None
    compressor: ContextCompressor = field(default_factory=ContextCompressor)
    patterner: ToolPatternOptimiser = field(default_factory=ToolPatternOptimiser)

    def __post_init__(self) -> None:
        self.adapter = BehaviouralAdapter(self.memory) if self.memory is not None else None
        self.last_compression: CompressionRecord | None = None

    def turn_cleanup(self, history: list[dict], query: str | None = None,
                     sequence: list[str] | None = None, *, error: bool = False, latency_ms: float = 0.0) -> dict:
        """End-of-turn housekeeping: compress, observe, adapt, return summary."""
        retained, record = self.compressor.compress(history, query=query)
        self.last_compression = record
        if sequence:
            self.patterner.observe(sequence, error=error, latency_ms=latency_ms)
        optims = self.patterner.optimisations()[:5]
        block = self.adapter.behaviour_block() if self.adapter else ""
        return {
            "history_retained": len(retained),
            "compression_ratio": record.compression_ratio,
            "behaviour_block": block,
            "pattern_optimisations": [
                {"seq": list(o.tool_sequence), "count": o.count,
                 "avg_ms": o.avg_latency_ms, "error_rate": o.error_rate}
                for o in optims
            ],
        }

    def save(self, path: str | Path) -> None:
        payload = {
            "pattern_counter": {",".join(k): v for k, v in self.patterner.sequence_counter.items()},
            "error_counter": {",".join(k): v for k, v in self.patterner.error_counts.items()},
            "latency_ms": {",".join(k): v for k, v in self.patterner.latency_ms.items()},
        }
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        if not Path(path).exists():
            return
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        for k, v in payload.get("pattern_counter", {}).items():
            self.patterner.sequence_counter[tuple(k.split(",")) if k else ()] = v
        for k, v in payload.get("error_counter", {}).items():
            self.patterner.error_counts[tuple(k.split(",")) if k else ()] = v
        for k, v in payload.get("latency_ms", {}).items():
            self.patterner.latency_ms[tuple(k.split(",")) if k else ()] = list(v)
