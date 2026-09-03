"""Explainable reasoning: evidence chains behind every decision.

Each decision records what evidence supported it, how confident the agent
was (calibrated), which alternatives were considered and why they lost, and
what would have falsified the choice. Rendered as markdown for the user and
stored as structured data for post-mortems.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class Evidence:
    source: str
    statement: str
    weight: float = 1.0
    supports: bool = True

    def render(self) -> str:
        arrow = "supports" if self.supports else "contradicts"
        return f"- [{self.source}] {self.statement} ({arrow}, weight {self.weight:.2f})"


@dataclass
class Considered:
    option: str
    score: float
    rejected_because: str = ""


@dataclass
class DecisionRecord:
    decision: str
    action: str
    confidence: float
    calibrated_confidence: float | None = None
    evidence: list[Evidence] = field(default_factory=list)
    alternatives: list[Considered] = field(default_factory=list)
    falsifiers: list[str] = field(default_factory=list)
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    outcome: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_evidence(self, source: str, statement: str, weight: float = 1.0, supports: bool = True) -> None:
        self.evidence.append(Evidence(source, statement, weight, supports))

    def add_alternative(self, option: str, score: float, rejected_because: str = "") -> None:
        self.alternatives.append(Considered(option, score, rejected_because))

    def to_dict(self) -> dict:
        return {
            "decision": self.decision, "action": self.action,
            "confidence": self.confidence,
            "calibrated_confidence": self.calibrated_confidence,
            "evidence": [{"source": e.source, "statement": e.statement,
                          "weight": e.weight, "supports": e.supports} for e in self.evidence],
            "alternatives": [{"option": a.option, "score": a.score,
                              "rejected_because": a.rejected_because} for a in self.alternatives],
            "falsifiers": self.falsifiers, "ts": self.ts,
            "outcome": self.outcome, "metadata": self.metadata,
        }

    def render(self) -> str:
        lines = [
            f"## Decision: {self.decision}",
            f"**Action:** {self.action}",
            f"**Confidence:** {self.confidence:.0%}"
            + (f" (calibrated: {self.calibrated_confidence:.0%})" if self.calibrated_confidence is not None else ""),
            f"**When:** {self.ts}",
            "",
            "### Evidence chain",
        ]
        lines += [e.render() for e in self.evidence] or ["- (none recorded)"]
        if self.alternatives:
            lines += ["", "### Alternatives considered"]
            lines += [f"- {a.option} (score {a.score:.2f}) — {a.rejected_because or 'not selected'}"
                      for a in sorted(self.alternatives, key=lambda a: -a.score)]
        if self.falsifiers:
            lines += ["", "### What would have changed my mind"]
            lines += [f"- {f}" for f in self.falsifiers]
        if self.outcome:
            lines += ["", f"### Outcome: {self.outcome}"]
        return "\n".join(lines)


class DecisionLedger:
    """Collects DecisionRecords for a session; supports outcome feedback loops."""

    def __init__(self, on_record=None):
        self.records: list[DecisionRecord] = []
        self.on_record = on_record

    def log(self, record: DecisionRecord) -> DecisionRecord:
        self.records.append(record)
        if self.on_record:
            self.on_record(record)
        return record

    def resolve(self, decision: str, outcome: str) -> DecisionRecord | None:
        for record in reversed(self.records):
            if record.decision == decision and record.outcome is None:
                record.outcome = outcome
                return record
        return None

    def unresolved(self) -> list[DecisionRecord]:
        return [r for r in self.records if r.outcome is None]

    def render_all(self) -> str:
        return "\n\n".join(r.render() for r in self.records)
