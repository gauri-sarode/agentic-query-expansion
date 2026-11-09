#!/usr/bin/env python3
"""Reranker-matched action-utility labels on TripClick BM25, exactly as
pre-committed in configs/reranker_matched_precommit.json (EXPLORATORY).

The stored label compares the reranked expanded candidate with the
unreranked BM25 original, so it mixes expansion with reranking. Here the
original side is the frozen BM25+rerank arm instead, so both sides share
the reranker. No generator calls, retrieval, reranking or refitting: the
detection and action-utility scores are scripts/46's frozen predictions.
Writes results/reranker_matched_labels.json (local, like results/).

Usage: PYTHONPATH=. python3 scripts/47_reranker_matched_labels.py
"""
from __future__ import annotations

import importlib.util
import json

import numpy as np

from src.data import load_qrels
from src.eval.metrics import evaluate

spec = importlib.util.spec_from_file_location("ra", "scripts/46_revision_analyses.py")
ra = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ra)
EPS = ra.EPS


def main() -> None:
    S = ra.tripclick_bm25()
    ro = ra._load("results/tripclick-tail_bm25_reranker_only.json")
    qids = S["test_qids"]
    acted = S["acted"]
    acted_qids = [q for q, a in zip(qids, acted) if a]

    cand = S["orig_acted"] + S["delta"]  # nDCG@10 of the reranked expanded candidate
    rr_orig = np.array([ro["reranked_ndcg"][q] for q in acted_qids])
    d = cand - rr_orig
    g = S["au_gain"]

    tie, helped, harmed = np.abs(d) <= EPS, d > EPS, d < -EPS
    nz = ~tie
    possible = rr_orig > EPS
    yp, ds = S["y_poor_all"][acted], S["det_all"][acted]

    qrels = load_qrels("tripclick-tail")
    ev = evaluate(qrels, S["ck"]["static_run"])
    static = np.array([ev[q]["ndcg_cut_10"] for q in qids])
    rr_all = np.array([ro["reranked_ndcg"][q] for q in qids])

    out = {
        "_note": "EXPLORATORY (configs/reranker_matched_precommit.json). TripClick BM25 test, acted-upon queries; label = candidate minus BM25+rerank of the original.",
        "M1_label_rates": {"n": int(len(d)), "tied": float(tie.mean()), "helped": float(helped.mean()),
                           "harmed": float(harmed.mean()), "harm_possible": float(possible.mean()),
                           "original_label_harm_possible": float((S["orig_acted"] > EPS).mean())},
        "M2_action_utility_auc": {
            "rule_a_harmed_vs_rest": ra.auc_ci(harmed.astype(int), -g),
            "rule_b_helped_vs_rest": ra.auc_ci(helped.astype(int), g),
            "rule_c_helped_vs_harmed": ra.auc_ci(helped[nz].astype(int), g[nz]),
            "harm_possible_only_rule_a": ra.auc_ci(harmed[possible].astype(int), -g[possible]),
        },
        "M3_paired_gap": {"detection": ra.auc_ci(yp, ds),
                          "detection_minus_matched_rule_a": ra.auc_diff(yp, ds, harmed.astype(int), -g)},
        "M4_policy_contrast": {"static_qe_mean": float(static.mean()), "bm25_rerank_mean": float(rr_all.mean()),
                               "static_minus_bm25_rerank": ra.mean_diff(rr_all, static)},
    }
    out = ra._clean(out)
    with open("results/reranker_matched_labels.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
