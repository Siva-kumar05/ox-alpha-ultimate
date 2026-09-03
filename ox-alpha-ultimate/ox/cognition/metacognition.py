"""Metacognition: calibrated confidence, uncertainty gating, ask-when-unsure.

The agent states how sure it is, and those statements are scored against
outcomes (Brier score) so raw self-reports get corrected by measured
reliability. Decisions whose calibrated confidence falls below a threshold
raise an explicit "ask the user" trigger instead of guessing silently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class CalibrationRecord:
    confidence: float
    outcome: float  # 1.0 = assertion was right, 0.0 = wrong


class ConfidenceTracker:
    """Learns a per-domain bias correction from scored assertions."""

    def __init__(self, buckets: int = 5):
        self.buckets = buckets
        self.records: list[CalibrationRecord] = []

    def observe(self, confidence: float, outcome: bool | float) -> None:
        conf = min(max(float(confidence), 0.0), 1.0)
        self.records.append(CalibrationRecord(conf, 1.0 if outcome else 0.0))

    def brier_score(self) -> float:
        if not self.records:
            return 0.25
        squared = sum((r.confidence - r.outcome) ** 2 for r in self.records)
        return squared / len(self.records)

    def calibration_curve(self) -> list[dict]:
        curve = []
        for b in range(self.buckets):
            lo, hi = b / self.buckets, (b + 1) / self.buckets
            bucket = [r for r in self.records if lo <= r.confidence <= hi or (b == self.buckets - 1 and r.confidence == 1.0)]
            if bucket:
                curve.append({
                    "bucket": f"{lo:.1f}-{hi:.1f}",
                    "mean_confidence": round(sum(r.confidence for r in bucket) / len(bucket), 3),
                    "observed_accuracy": round(sum(r.outcome for r in bucket) / len(bucket), 3),
                    "n": len(bucket),
                })
        return curve

    def reliability(self) -> float:
        """1 - Brier, bounded to [0,1]: how trustworthy raw confidences are."""
        return max(0.0, 1.0 - 2.0 * self.brier_score())

    def calibrated(self, confidence: float) -> float:
        """Shrink extreme self-reports toward the historically observed rate.

        An agent that is right 60% of the time when it says "90% sure" should
        not act as if 90% were the true probability."""
        if not self.records:
            return min(max(float(confidence), 0.0), 1.0)
        base = sum(r.outcome for r in self.records) / len(self.records)
        shrinkage = min(1.0, len(self.records) / 30.0) * self.reliability()
        corrected = base + shrinkage * (float(confidence) - base)
        return min(max(corrected, 0.01), 0.99)


class UncertaintyGate:
    """Threshold policy: confident -> act, uncertain -> ask, ambiguous -> verify."""

    def __init__(self, act_threshold: float = 0.75, ask_threshold: float = 0.45):
        self.act_threshold = act_threshold
        self.ask_threshold = ask_threshold
        self.tracker = ConfidenceTracker()

    def evaluate(self, confidence: float, evidence_count: int = 0, conflict: float = 0.0) -> "Disposition":
        calibrated = self.tracker.calibrated(confidence)
        # Thin or conflicting evidence lowers usable confidence further.
        penalty = min(0.25, 0.05 * max(0, 3 - evidence_count)) + min(0.25, max(0.0, conflict) * 0.5)
        usable = max(0.0, calibrated - penalty)
        if usable >= self.act_threshold:
            action = "ACT"
        elif usable >= self.ask_threshold:
            action = "VERIFY"
        else:
            action = "ASK_USER"
        return Disposition(action=action, raw_confidence=round(float(confidence), 3),
                           calibrated_confidence=round(calibrated, 3),
                           usable_confidence=round(usable, 3),
                           evidence_count=evidence_count, conflict=round(float(conflict), 3))

    def feedback(self, confidence: float, outcome: bool | float) -> None:
        self.tracker.observe(confidence, outcome)


@dataclass
class Disposition:
    action: str
    raw_confidence: float
    calibrated_confidence: float
    usable_confidence: float
    evidence_count: int
    conflict: float
    notes: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (f"{self.action}(usable={self.usable_confidence:.2f}, "
                f"calibrated={self.calibrated_confidence:.2f}, evidence={self.evidence_count})")


def combine_independent(confidences: list[float]) -> float:
    """Noisy-OR combination of independent evidence confidences."""
    if not confidences:
        return 0.0
    prob = 1.0
    for c in confidences:
        prob *= (1.0 - min(max(float(c), 0.0), 0.999))
    return 1.0 - prob


def entropy(probability: float) -> float:
    p = min(max(float(probability), 1e-9), 1.0 - 1e-9)
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
