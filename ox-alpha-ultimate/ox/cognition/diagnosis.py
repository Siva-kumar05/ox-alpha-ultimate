"""Differential diagnosis: hypothesis generation, evidence scoring, falsification.

Instead of fixing the first plausible explanation, the engine keeps a live
differential: hypotheses compete as log-odds posteriors updated by evidence,
and the next test chosen is the one with maximum expected information gain —
explicitly rewarding tests that can *falsify* the leading hypothesis, which
is the structural cure for confirmation bias.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .metacognition import entropy


@dataclass
class Hypothesis:
    name: str
    prior: float = 0.5
    likelihoods: dict[str, float] = field(default_factory=dict)  # evidence name -> P(e | H)
    tested_by: set[str] = field(default_factory=set)

    def posterior(self) -> float:
        """Normalised posterior over the differential given observed evidence."""
        return self._logodds  # replaced by engine; kept for introspection

    def __post_init__(self) -> None:
        self._logodds = math.log(min(max(self.prior, 1e-6), 1 - 1e-6) /
                                 (1 - min(max(self.prior, 1e-6), 1 - 1e-6)))


@dataclass
class Evidence:
    name: str
    observation: str          # which outcome was seen
    false_positive_rate: dict[str, float]  # outcome -> P(e | not H), shared across hypotheses


class DifferentialEngine:
    def __init__(self):
        self.hypotheses: dict[str, Hypothesis] = {}
        self.observed: list[tuple[str, str]] = []

    def add(self, name: str, prior: float = 0.5) -> Hypothesis:
        h = Hypothesis(name=name, prior=prior)
        self.hypotheses[name] = h
        return h

    def define_test(self, hypothesis: str, evidence_name: str,
                    likelihoods: dict[str, float], false_positive_rate: dict[str, float]) -> None:
        """Register P(outcome | hypothesis) for a test, and the shared
        P(outcome | not hypothesis) used as the denominator of the Bayes update."""
        h = self.hypotheses[hypothesis]
        h.likelihoods[evidence_name] = likelihoods
        h.tested_by.add(evidence_name)
        if not hasattr(self, "tests"):
            self.tests: dict[str, dict[str, float]] = {}
        self.tests[evidence_name] = false_positive_rate

    def observe(self, evidence_name: str, outcome: str) -> None:
        if evidence_name not in getattr(self, "tests", {}):
            raise ValueError(f"Evidence {evidence_name} has no defined test")
        self.observed.append((evidence_name, outcome))
        for h in self.hypotheses.values():
            if evidence_name in h.likelihoods:
                p_e_h = h.likelihoods[evidence_name].get(outcome, 1e-3)
                p_e_not = self.tests[evidence_name].get(outcome, 1e-3)
                h._logodds += math.log(max(p_e_h, 1e-6) / max(p_e_not, 1e-6))

    def posteriors(self) -> dict[str, float]:
        odds = {name: math.exp(h._logodds) for name, h in self.hypotheses.items()}
        total = sum(odds.values()) or 1.0
        return {name: round(o / total, 4) for name, o in odds.items()}

    def leader(self) -> tuple[str, float] | None:
        posteriors = self.posteriors()
        if not posteriors:
            return None
        name = max(posteriors, key=posteriors.get)
        return name, posteriors[name]

    def next_test(self, candidate_tests: dict[str, list[str]]) -> str | None:
        """Pick the test with maximum expected entropy reduction.

        ``candidate_tests`` maps an unobserved test name to its possible
        outcomes. A test that cannot distinguish the leader from the runner-up
        carries no information and is never selected — this is what forces
        falsifying evidence to be gathered before commitment."""
        if not candidate_tests:
            return None
        leader = self.leader()
        if leader is None:
            return None
        posteriors = self.posteriors()
        best_name, best_gain = None, -1.0
        for test, outcomes in candidate_tests.items():
            if any(test == observed_name for observed_name, _ in self.observed):
                continue
            expected_final_entropy = 0.0
            for outcome in outcomes:
                # P(outcome) under current beliefs, assuming the test is
                # informative for whichever hypothesis assigns it likelihood.
                p_out = 0.0
                for h in self.hypotheses.values():
                    if test in h.likelihoods:
                        p_h = posteriors[h.name]
                        p_out += p_h * h.likelihoods[test].get(outcome, 0.0)
                p_out = min(max(p_out, 1e-6), 1 - 1e-6)
                # Entropy after a Bayesian update that scales each hypothesis
                # by the outcome likelihood it assigns.
                weights = {}
                for h in self.hypotheses.values():
                    lh = h.likelihoods.get(test, {}).get(outcome, 0.5)
                    weights[h.name] = posteriors[h.name] * max(lh, 1e-6)
                z = sum(weights.values()) or 1.0
                probs = [w / z for w in weights.values()]
                h_out = sum(-p * math.log(max(p, 1e-9)) for p in probs if p > 0)
                expected_final_entropy += p_out * h_out
            gain = entropy(leader[1]) - expected_final_entropy
            if gain > best_gain:
                best_name, best_gain = test, gain
        return best_name

    def ready_to_commit(self, min_posterior: float = 0.85, require_falsifier: bool = True) -> bool:
        leader = self.leader()
        if leader is None:
            return False
        name, posterior = leader
        if posterior < min_posterior:
            return False
        if require_falsifier:
            # The leading hypothesis must have survived at least one test that
            # could have dropped its posterior below the commit threshold.
            h = self.hypotheses[name]
            return len(h.tested_by & {t for t, _ in self.observed}) >= 1
        return True

    def report(self) -> dict:
        leader = self.leader()
        return {
            "posteriors": self.posteriors(),
            "leader": leader,
            "observed": [{"test": t, "outcome": o} for t, o in self.observed],
            "ready_to_commit": self.ready_to_commit(),
        }
