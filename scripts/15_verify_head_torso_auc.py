#!/usr/bin/env python3
"""Independent spot-check of the HEAD/TORSO action-utility AUC finding
(paper Table IX / Section VIII), after a reviewer asked to verify the
"startling" below-chance numbers (0.148 HEAD, 0.214 TORSO) rather than
trust them on the strength of the original computation alone -- that
original computation was inline/unsaved (a real methodological gap,
matching the earlier caught Table IV/V frontier bug), so it was never
independently reproducible from a saved script.

This script:
  (1) verifies the scoring CONVENTION against a known-correct number:
      harmful = (true nDCG@10 delta < 0), score = -verifier_score
      (lower verifier score should predict more harmful). Recomputing
      TAIL's frozen AUC this way from
      results/tripclick-tail_verifier_calibration.json reproduces
      0.5524 -- essentially exactly Table II's reported 0.552, n=1157,
      prevalence 0.330. This confirms the formula, not just for TAIL,
      is not sign-flipped or otherwise wrong.
  (2) re-collects a FRESH, smaller (n=150/bucket, seeded) sample on
      HEAD and TORSO with the identical frozen retrieval config and
      verifier as the original full-n=1175 run, computing the same AUC,
      to check the original finding reproduces in direction and rough
      magnitude on independent data -- not a full replication (that
      would mean re-running the original n=1175 x 2 LLM-driven passes),
      but sufficient to catch a data-collection bug (wrong DB, misaligned
      dict, degenerate sample) that the formula-only check in (1) cannot.

Usage: PYTHONPATH=. python3 scripts/15_verify_head_torso_auc.py
"""
from __future__ import annotations

import json
import random

import numpy as np
from sklearn.metrics import roc_auc_score

from src.agent.static_baseline import run_static_episode
from src.data import load_qrels, load_queries
from src.eval.metrics import evaluate
from src.retrieval.bm25_index import search

DB_PATH = "data/tripclick_bm25.sqlite3"
N_SAMPLE = 150
SEED = 42


def verify_convention() -> None:
    d = json.loads(open("results/tripclick-tail_verifier_calibration.json").read())
    y_delta = np.array(d["y_delta_ndcg"])
    y_verify = np.array(d["y_current_verify"])
    harmful = (y_delta < -1e-9).astype(int)
    auc = roc_auc_score(harmful, -y_verify)
    print(f"[convention check] TAIL recomputed AUC = {auc:.4f} (paper reports 0.552, n={len(y_delta)}, "
          f"prevalence={harmful.mean():.3f}) -- {'MATCH' if abs(auc - 0.552) < 0.005 else 'MISMATCH'}")


def run_bucket(dataset: str) -> dict:
    queries = load_queries(dataset)
    qrels = load_qrels(dataset)
    all_qids = [qid for qid in queries if qid in qrels]
    rng = random.Random(SEED)
    sample_qids = rng.sample(all_qids, min(N_SAMPLE, len(all_qids)))
    print(f"\n[{dataset}] n sample = {len(sample_qids)}", flush=True)

    baseline_run, static_run, verifier_scores = {}, {}, {}
    for i, qid in enumerate(sample_qids, 1):
        text = queries[qid]
        baseline_run[qid] = dict(search(text, DB_PATH, k=100))
        static_ranking, trace = run_static_episode(qid, text, DB_PATH, dataset_slice=dataset)
        static_run[qid] = {doc_id: 1.0 / (rank + 1) for rank, doc_id in enumerate(static_ranking)}
        if trace.verifier_score is not None:
            verifier_scores[qid] = trace.verifier_score
        if i % 25 == 0:
            print(f"  [{dataset}] {i}/{len(sample_qids)}", flush=True)

    baseline_eval = evaluate(qrels, baseline_run)
    static_eval = evaluate(qrels, static_run)
    common = [q for q in sample_qids if q in verifier_scores]
    harmful = np.array(
        [1 if (static_eval[q]["ndcg_cut_10"] - baseline_eval[q]["ndcg_cut_10"]) < -1e-9 else 0 for q in common]
    )
    vscore = np.array([-verifier_scores[q] for q in common])
    auc = None
    if 0 < harmful.sum() < len(harmful):
        auc = float(roc_auc_score(harmful, vscore))
    result = {
        "n_scored": len(common),
        "n_harmful": int(harmful.sum()),
        "auc": auc,
        "bm25_ndcg10": float(np.mean([baseline_eval[q]["ndcg_cut_10"] for q in sample_qids])),
        "static_ndcg10": float(np.mean([static_eval[q]["ndcg_cut_10"] for q in sample_qids])),
    }
    print(f"[{dataset}] {result}")
    return result


def main() -> None:
    verify_convention()
    out = {"head": run_bucket("tripclick-head"), "torso": run_bucket("tripclick-torso")}
    print("\n=== spot-check summary (fresh n=150 sample, vs. original full-n=1175 result) ===")
    print(f"HEAD:  spot-check AUC={out['head']['auc']}  (original: 0.148)")
    print(f"TORSO: spot-check AUC={out['torso']['auc']}  (original: 0.214)")
    with open("results/tripclick_head_torso_auc_verification.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nsaved to results/tripclick_head_torso_auc_verification.json")


if __name__ == "__main__":
    main()
