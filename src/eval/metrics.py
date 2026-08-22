"""Retrieval effectiveness metrics via pytrec_eval: NDCG@10, Recall@100,
MRR. TripClick results are reported separately per HEAD/TORSO/TAIL slice
by filtering qrels/runs before calling evaluate().
"""
from __future__ import annotations

import pytrec_eval

MEASURES = {"ndcg_cut_10", "recall_100", "recip_rank"}


def evaluate(qrels: dict[str, dict[str, int]], run: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """qrels: query_id -> {doc_id: relevance}. run: query_id -> {doc_id: score}.
    Returns query_id -> {measure: value}.
    """
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, MEASURES)
    return evaluator.evaluate(run)


def mean_metrics(per_query: dict[str, dict[str, float]]) -> dict[str, float]:
    if not per_query:
        return {}
    measures = next(iter(per_query.values())).keys()
    return {
        m: sum(scores[m] for scores in per_query.values()) / len(per_query)
        for m in measures
    }


def helped_unchanged_harmed(
    baseline: dict[str, dict[str, float]],
    treatment: dict[str, dict[str, float]],
    measure: str = "ndcg_cut_10",
    eps: float = 1e-9,
) -> dict[str, float]:
    """Query-level helped/unchanged/harmed fractions -- mean NDCG can
    conceal intervention regressions the paper needs to report.
    """
    n = len(baseline)
    helped = sum(1 for q in baseline if treatment[q][measure] > baseline[q][measure] + eps)
    harmed = sum(1 for q in baseline if treatment[q][measure] < baseline[q][measure] - eps)
    unchanged = n - helped - harmed
    return {"helped": helped / n, "unchanged": unchanged / n, "harmed": harmed / n}
