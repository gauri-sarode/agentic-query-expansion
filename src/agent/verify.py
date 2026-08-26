"""verify(previous_state, new_state) -> score. Qrel-free: judges whether
the new retrieval state is healthier than the previous one using
telemetry only -- this is what makes the verifier usable at inference
time, when qrels aren't available.

v1 implementation: a fixed-weight linear combination of 10 telemetry-delta
features, fitted (LinearRegression, no intercept) against true NDCG@10
delta on 1,168 TripClick TAIL episodes (scripts/08_calibrate_verifier.py,
2026-08-26) -- see docs/milestones.md and the project's calibration
history. This replaces a v0 hand-picked-weight baseline that was tuned by
intuition, not data, and turned out to be actively uninformative on
TripClick (Pearson r=-0.064 vs. true NDCG delta). The fitted weights
here score r=+0.134 on the same held-out (5-fold CV) data -- a real,
statistically meaningful improvement, but still a WEAK signal in absolute
terms (R^2 ~ 1.8% of variance explained). Two GBDT (non-linear) fits were
also tried on both NFCorpus and TripClick data and did not beat this
linear fit on TripClick specifically -- see results/tripclick-tail_verifier_calibration.json
for the raw calibration data this was fit from.

This is real, useful signal-detection improvement over v0, not a solved
problem: treat the verifier as weakly-but-genuinely informative, not
authoritative, and expect it to need recalibration if the feature set,
generator, or corpus changes materially.
"""
from __future__ import annotations

from src.agent.state import AgentState

# Fitted via LinearRegression(fit_intercept=False) on
# (X=10 telemetry-delta features, y=true NDCG@10 delta), n=1168,
# TripClick TAIL, 2026-08-26. Order matches the feature computation below.
_WEIGHTS = {
    "score_margin_gain": -0.0019,
    "coverage_gain": 0.2787,
    "coherence_gain": 0.2934,
    "stability": -0.0048,
    "drift": -0.0709,
    "disagreement": -0.0974,
    "topk_overlap": 0.0520,
    "reranker_top_score": 0.0024,
    "reranker_score_margin": 0.0005,
    "query_length_ratio": -0.0026,
}


def verify(previous_state: AgentState, new_state: AgentState) -> float:
    """Returns a verifier score; higher means the new state is more
    likely to be a retrieval-quality improvement over previous_state.
    Requires both states' O_t to be populated. Score scale is small
    (roughly -0.1 to +0.03 in the calibration sample) -- do not reuse the
    v0 thresholds (~-1 to +0.05) against this verifier; see
    src/agent/controller.py's RecoveryThresholds, recalibrated to match.
    """
    prev_o, new_o = previous_state.O_t, new_state.O_t
    if prev_o is None or new_o is None:
        raise ValueError("verify() requires both states to have O_t computed")

    features = {
        "score_margin_gain": new_o.retrieval.score_margin - prev_o.retrieval.score_margin,
        "coverage_gain": new_o.retrieval.query_term_coverage - prev_o.retrieval.query_term_coverage,
        "coherence_gain": new_o.retrieval.result_coherence - prev_o.retrieval.result_coherence,
        "stability": new_o.retrieval.ranking_stability,
        "drift": new_o.agent.rewrite_semantic_drift,
        "disagreement": new_o.retrieval.reranker_bm25_disagreement,
        "topk_overlap": new_o.retrieval.topk_overlap,
        "reranker_top_score": new_o.retrieval.reranker_top_score,
        "reranker_score_margin": new_o.retrieval.reranker_score_margin,
        "query_length_ratio": new_o.agent.query_length_ratio,
    }

    return sum(_WEIGHTS[name] * value for name, value in features.items())
