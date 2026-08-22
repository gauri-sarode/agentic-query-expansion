"""verify(previous_state, new_state) -> score. Qrel-free: judges whether
the new retrieval state is healthier than the previous one using
telemetry only (score margin/entropy gain, coverage gain, coherence,
stability, drift) -- this is what makes the verifier usable at inference
time, when qrels aren't available.

v0 implementation: a fixed-weight linear combination of telemetry deltas,
frozen before evaluation rather than tuned per dataset. This is the
interpretable baseline the brief calls for; the learned verifier the
system-design doc describes is the natural v1 replacement once labeled
episodes (from fault injection + this heuristic's own trajectories)
accumulate to train it on.
"""
from __future__ import annotations

from src.agent.state import AgentState

# Frozen weights: positive terms reward healthier retrieval, negative
# terms penalize signs of a harmful rewrite (drift, incoherence,
# instability, introduced/unsupported entities).
_WEIGHTS = {
    "score_margin_gain": 1.0,
    "coverage_gain": 1.0,
    "coherence_gain": 0.5,
    "stability": 0.5,
    "drift_penalty": -1.0,
    "disagreement_penalty": -0.5,
}


def verify(previous_state: AgentState, new_state: AgentState) -> float:
    """Returns a verifier score; higher means the new state is more
    likely to be a retrieval-quality improvement over previous_state.
    Requires both states' O_t to be populated.
    """
    prev_o, new_o = previous_state.O_t, new_state.O_t
    if prev_o is None or new_o is None:
        raise ValueError("verify() requires both states to have O_t computed")

    score_margin_gain = new_o.retrieval.score_margin - prev_o.retrieval.score_margin
    coverage_gain = new_o.retrieval.query_term_coverage - prev_o.retrieval.query_term_coverage
    coherence_gain = new_o.retrieval.result_coherence - prev_o.retrieval.result_coherence
    stability = new_o.retrieval.ranking_stability
    drift = new_o.agent.rewrite_semantic_drift
    disagreement = new_o.retrieval.reranker_bm25_disagreement

    return (
        _WEIGHTS["score_margin_gain"] * score_margin_gain
        + _WEIGHTS["coverage_gain"] * coverage_gain
        + _WEIGHTS["coherence_gain"] * coherence_gain
        + _WEIGHTS["stability"] * stability
        + _WEIGHTS["drift_penalty"] * drift
        + _WEIGHTS["disagreement_penalty"] * disagreement
    )
