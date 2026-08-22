"""Agent Search SLO and observability/agent metrics
(docs/milestones.md): detection recall, false alarm rate, rollback
precision, recovery rate, plus the harm/cost/latency constraint check.
"""
from __future__ import annotations

from dataclasses import dataclass


def detection_recall(detected_and_failed: int, failed: int) -> float:
    """P(detected | failure)."""
    return detected_and_failed / failed if failed else float("nan")


def false_alarm_rate(intervened_and_healthy: int, healthy: int) -> float:
    """P(intervention | healthy)."""
    return intervened_and_healthy / healthy if healthy else float("nan")


def rollback_precision(ranking_improved_after_rollback: int, rollbacks: int) -> float:
    """P(R_{t-1} > R_t | ROLLBACK)."""
    return ranking_improved_after_rollback / rollbacks if rollbacks else float("nan")


def recovery_rate(restored: int, recoverable_failed: int) -> float:
    return restored / recoverable_failed if recoverable_failed else float("nan")


@dataclass(frozen=True)
class AgentSearchSLO:
    """Target range, not a single optimized metric (Google SRE framing).
    Freeze from ~100 local dev episodes before main-test evaluation --
    see docs/milestones.md. maximize retrieval quality subject to:
    """

    max_harm_rate: float
    max_expected_llm_calls: float
    max_p95_latency_ms: float

    def is_met(self, harm_rate: float, expected_llm_calls: float, p95_latency_ms: float) -> bool:
        return (
            harm_rate <= self.max_harm_rate
            and expected_llm_calls <= self.max_expected_llm_calls
            and p95_latency_ms <= self.max_p95_latency_ms
        )
