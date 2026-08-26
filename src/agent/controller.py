"""Action selection and recovery control. Deliberately not the LLM:

- Initial-action selection conditions on retrieval telemetry (O_t at
  q_0's initial ranking) to pick an operator (or NO_EXPAND).
- Recovery control (accept / ROLLBACK / REPLAN / STOP after observing an
  expansion's consequences) is a deterministic finite-state controller
  driven by verifier score + budget, per docs/failure_taxonomy.md.

v0 implementation: both are fixed, documented threshold rules rather than
a trained classifier. The brief's target is a small learned logistic
regression / GBDT controller -- that needs labeled failure/healthy
episodes to train on, which this v0 rule-based controller is what
generates (running it + fault injection produces the training set for a
v1 learned controller). Thresholds here are placeholders to be frozen
against ~100 local dev episodes per the go/no-go checkpoints in
docs/milestones.md, not tuned per-dataset.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.agent.actions import Action
from src.agent.state import AgentState
from src.observability.telemetry import ObservabilityState


@dataclass(frozen=True)
class SelectorThresholds:
    min_entity_overlap: float = 1.0  # below this (i.e. any missing capitalized entity) -> ENTITY_NORMALIZE
    min_term_coverage: float = 0.5  # below this -> VOCABULARY
    max_query_tokens_for_context: int = 2  # at/below this length -> CONTEXT
    max_entropy_for_context: float = 0.8  # above this (dispersed results) -> CONTEXT
    max_coherence_for_ambiguity: float = 0.15  # below this -> AMBIGUITY


@dataclass(frozen=True)
class RecoveryThresholds:
    # Recalibrated 2026-08-26 against verify()'s v1 fitted-weight score
    # (src/agent/verify.py), which has a much smaller scale (~-0.1 to
    # +0.03) than the v0 hand-picked weights these were previously tuned
    # against (~-5 to +30) -- the old 0.05/-1.0 values are meaningless on
    # the new scale and would have effectively disabled ACCEPT/ROLLBACK
    # entirely. Chosen from the same 1,168-sample TripClick TAIL
    # calibration verify() itself was fit on
    # (results/tripclick-tail_verifier_calibration.json), by checking
    # which candidate cutoffs actually separated true-harmed episodes
    # from the rest in that data (not just score percentiles blind to
    # outcome, unlike the v0 calibration). accept=-0.036 is the sample
    # median; harmful=-0.06 was the best precision/recall trade-off found
    # in that grid (catches 34% of true-harmed episodes at an 18% false
    # -rollback rate on true-helped ones) -- still a weak, not strong,
    # signal (verify()'s docstring has the full caveat). Gives a roughly
    # even three-way split on that data: ~50% ACCEPT, ~22% ROLLBACK,
    # ~28% REPLAN.
    accept: float = -0.036  # verifier_score >= this -> accept
    harmful: float = -0.06  # verifier_score <= this -> ROLLBACK


class ActionSelector:
    """Chooses the initial action from O_t (pre-expansion telemetry)."""

    def __init__(self, thresholds: SelectorThresholds | None = None) -> None:
        self.thresholds = thresholds or SelectorThresholds()

    def select(self, query: str, observation: ObservabilityState) -> Action:
        t = self.thresholds
        r = observation.retrieval

        if r.entity_overlap < t.min_entity_overlap:
            return Action.ENTITY_NORMALIZE
        if r.query_term_coverage < t.min_term_coverage:
            return Action.VOCABULARY
        if len(query.split()) <= t.max_query_tokens_for_context or r.score_entropy > t.max_entropy_for_context:
            return Action.CONTEXT
        if r.result_coherence < t.max_coherence_for_ambiguity:
            return Action.AMBIGUITY
        return Action.NO_EXPAND


class RecoveryController:
    """Finite-state controller: given post-action O_t and the verifier
    score, decide ACCEPT (None) / ROLLBACK / REPLAN / STOP.
    """

    def __init__(self, thresholds: RecoveryThresholds | None = None) -> None:
        self.thresholds = thresholds or RecoveryThresholds()

    def decide(self, state: AgentState, verifier_score: float) -> Action | None:
        t = self.thresholds
        if state.exhausted:
            return Action.STOP
        if verifier_score >= t.accept:
            return None  # accept
        if verifier_score <= t.harmful:
            return Action.ROLLBACK
        return Action.REPLAN
