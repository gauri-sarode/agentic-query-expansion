#!/usr/bin/env python3
"""Calibrates the qrel-free verifier against ground truth: this is
RQ-Detection (README) made concrete -- can retrieval/agent telemetry
actually predict whether an expansion action helped or hurt, measured
against real qrels? verify() currently answers that with hand-picked
fixed weights (src/agent/verify.py); this script checks how well those
weights track the true NDCG@10 delta, and fits a data-driven alternative
using the exact same features.

For each query: retrieve original, select the initial operator, run one
expansion action, then compute (a) the CURRENT verify() score, (b) the
same 6 telemetry-delta features verify() uses, and (c) the ground-truth
NDCG@10 delta (candidate ranking vs. original, against real qrels) --
something only available offline/in calibration, never at inference
time. Correlates (a) and a fitted linear model of (b) against (c).

Usage: python scripts/08_calibrate_verifier.py [dataset] [n_queries]
"""
from __future__ import annotations

import sys

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict

from src.agent.actions import EXPANSION_ACTIONS
from src.agent.controller import ActionSelector
from src.agent.state import AgentState
from src.agent.steps import act_once, build_evidence, retrieve_and_observe_initial
from src.agent.verify import verify
from src.data import load_qrels, load_queries
from src.eval.metrics import evaluate
from src.llm_client import LLMUnavailableError

_DB_PATHS = {
    "nfcorpus": "data/nfcorpus_bm25.sqlite3",
    "tripclick-head": "data/tripclick_bm25.sqlite3",
    "tripclick-torso": "data/tripclick_bm25.sqlite3",
    "tripclick-tail": "data/tripclick_bm25.sqlite3",
}

_FEATURE_NAMES = [
    "score_margin_gain",
    "coverage_gain",
    "coherence_gain",
    "stability",
    "drift",
    "disagreement",
]


def _rank_to_scores(ranking: list[str]) -> dict[str, float]:
    return {d: 1.0 / (i + 1) for i, d in enumerate(ranking)}


def _ndcg10(qid: str, ranking: list[str], qrels: dict) -> float:
    result = evaluate({qid: qrels[qid]}, {qid: _rank_to_scores(ranking)})
    return result[qid]["ndcg_cut_10"]


def _features(prev_o, new_o) -> list[float]:
    return [
        new_o.retrieval.score_margin - prev_o.retrieval.score_margin,
        new_o.retrieval.query_term_coverage - prev_o.retrieval.query_term_coverage,
        new_o.retrieval.result_coherence - prev_o.retrieval.result_coherence,
        new_o.retrieval.ranking_stability,
        new_o.agent.rewrite_semantic_drift,
        new_o.retrieval.reranker_bm25_disagreement,
    ]


def main() -> None:
    dataset = sys.argv[1] if len(sys.argv) > 1 else "nfcorpus"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 80
    db_path = _DB_PATHS[dataset]

    queries = load_queries(dataset)
    qrels = load_qrels(dataset)
    query_ids = [qid for qid in queries if qid in qrels][:n]

    selector = ActionSelector()
    X, y_delta_ndcg, y_current_verify = [], [], []
    skipped_no_expand = 0
    skipped_llm_error = 0

    for qid in query_ids:
        text = queries[qid]
        init = retrieve_and_observe_initial(text, db_path, k=100)
        action = selector.select(text, init.o0)
        if action not in EXPANSION_ACTIONS:
            skipped_no_expand += 1
            continue

        state = AgentState(q0=text, q_t=text, R_t=init.r0_ids, O_t=init.o0, B_t=1)
        evidence = build_evidence(init.r0_ids, db_path)
        try:
            result = act_once(action, state, init.r0, db_path, evidence, k=100, use_reranker=True)
        except LLMUnavailableError:
            skipped_llm_error += 1
            continue

        original_ndcg = _ndcg10(qid, init.r0_ids, qrels)
        candidate_ndcg = _ndcg10(qid, result.candidate.R_t, qrels)
        delta_ndcg = candidate_ndcg - original_ndcg

        X.append(_features(init.o0, result.candidate.O_t))
        y_delta_ndcg.append(delta_ndcg)
        y_current_verify.append(result.verifier_score)

        print(f"{qid:>8}  action={action.value:<16} verify={result.verifier_score:+8.3f}  delta_ndcg={delta_ndcg:+.4f}")

    print(f"\nn_samples={len(X)}  (skipped: {skipped_no_expand} NO_EXPAND, {skipped_llm_error} LLM errors)")
    X = np.array(X)
    y_delta_ndcg = np.array(y_delta_ndcg)
    y_current_verify = np.array(y_current_verify)

    current_corr = np.corrcoef(y_current_verify, y_delta_ndcg)[0, 1]
    print(f"\ncurrent fixed-weight verify() vs. true NDCG@10 delta: Pearson r = {current_corr:.3f}")

    # Cross-validated fit so the reported correlation isn't just training-set overfit.
    model = LinearRegression(fit_intercept=False)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_pred = cross_val_predict(model, X, y_delta_ndcg, cv=kf)
    fitted_corr = np.corrcoef(cv_pred, y_delta_ndcg)[0, 1]
    print(f"fitted linear model (5-fold CV) vs. true NDCG@10 delta: Pearson r = {fitted_corr:.3f}")

    model.fit(X, y_delta_ndcg)
    print("\nfitted weights (full-data fit, for reference):")
    for name, coef in zip(_FEATURE_NAMES, model.coef_):
        print(f"  {name:<20} {coef:+.4f}")

    if fitted_corr > current_corr + 0.05:
        print(
            "\nFitted model meaningfully improves on the current fixed weights -- "
            "worth adopting into src/agent/verify.py."
        )
    else:
        print(
            "\nFitted model does NOT meaningfully improve on the current fixed weights. "
            "This is itself a real RQ-Detection finding: with this feature set, "
            "linear combination of these 6 telemetry signals has limited power to predict "
            "true relevance-quality change -- either different/more features are needed, or "
            "a non-linear model, or this is a genuine ceiling on qrel-free detection here."
        )


if __name__ == "__main__":
    main()
