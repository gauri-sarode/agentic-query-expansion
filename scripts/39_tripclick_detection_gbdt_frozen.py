#!/usr/bin/env python3
"""Single confirmatory test touch for the GBDT detection-model swap
(scripts/37 found a replicating, seed-stable +0.02-0.025 AUC validation-CV
advantage for GradientBoostingClassifier over the deployed LogisticRegression,
on both TripClick and NQ validation -- see results/detection_nonlinear_check.json).
This freezes that model class (same hyperparameters as scripts/08's
action-utility GBDT check: n_estimators=30, max_depth=2, learning_rate=0.1,
random_state=42) and touches TripClick TAIL frozen test with it exactly
once, reproducing BOTH numbers the paper needs:

  (A) Full-population (n=1175) test AUC, mirroring scripts/17's
      methodology exactly -- this is what would replace Table 1's 0.649.
  (B) Acted-upon-subset (n=1157) AUC and its paired bootstrap difference
      against action-utility AUC, mirroring scripts/28's methodology and
      alignment logic exactly (same qid-keyed reduction via the frozen
      agent_vs_static checkpoint) -- this is what would replace the
      "+0.091*" paired-difference row.

Nothing upstream is touched or re-selected based on this script's output:
features, val/test split, and the action-utility side are all the
already-frozen artifacts. This is the one allowed test look for the
new model class, per scripts/37's own stated protocol.

Usage: PYTHONPATH=. python3 scripts/39_tripclick_detection_gbdt_frozen.py
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

N_BOOTSTRAP = 10000
SEED = 42
GBDT_PARAMS = dict(n_estimators=30, max_depth=2, learning_rate=0.1, random_state=SEED)


def bootstrap_ci(harmful: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    n = len(harmful)
    rng = np.random.RandomState(SEED)
    aucs = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.randint(0, n, n)
        h, s = harmful[idx], score[idx]
        if 0 < h.sum() < len(h):
            aucs.append(roc_auc_score(h, s))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def main() -> None:
    fd = json.loads(open("results/tripclick-tail_failure_detection.json").read())
    X_val, y_val_ndcg = np.array(fd["X_val"]), np.array(fd["y_val_ndcg"])
    X_test, y_test_ndcg = np.array(fd["X_test"]), np.array(fd["y_test_ndcg"])
    y_val_poor = (y_val_ndcg <= 1e-9).astype(int)
    y_test_poor = (y_test_ndcg <= 1e-9).astype(int)

    model = GradientBoostingClassifier(**GBDT_PARAMS).fit(X_val, y_val_poor)
    test_scores = model.predict_proba(X_test)[:, 1]

    point_a = roc_auc_score(y_test_poor, test_scores)
    lo_a, hi_a = bootstrap_ci(y_test_poor, test_scores)
    print(f"(A) full-population (n={len(y_test_poor)}) GBDT test AUC: {point_a:.4f}  "
          f"95% CI=[{lo_a:.4f},{hi_a:.4f}]")
    print(f"    (existing logistic-regression number, Table 1: 0.6493)")

    # --- (B) acted-upon subset + paired difference vs. action-utility, mirroring scripts/28 ---
    vc = json.loads(open("results/tripclick-tail_verifier_calibration.json").read())
    ck = json.loads(open("results/tripclick-tail_agent_vs_static_checkpoint.json").read())

    action_by_qid = {t["query_id"]: t["planner_action"] for t in ck["static_traces"]}
    assert set(action_by_qid) == set(fd["test_qids"]), "population mismatch"

    acted_qids_in_order = [q for q in fd["test_qids"] if action_by_qid[q] != "NO_EXPAND"]
    assert len(acted_qids_in_order) == len(vc["y_delta_ndcg"]) == 1157

    idx_by_qid = {q: i for i, q in enumerate(fd["test_qids"])}
    reduced_idx = [idx_by_qid[q] for q in acted_qids_in_order]
    y_poor_reduced = y_test_poor[reduced_idx]
    gbdt_proba_reduced = test_scores[reduced_idx]

    failure_auc_reduced = roc_auc_score(y_poor_reduced, gbdt_proba_reduced)
    print(f"\n(B) acted-upon subset (n={len(y_poor_reduced)}) GBDT AUC: {failure_auc_reduced:.4f}")

    y_delta_ndcg = np.array(vc["y_delta_ndcg"])
    y_current_verify = np.array(vc["y_current_verify"])
    harmful = (y_delta_ndcg < -1e-9).astype(int)
    action_score = -y_current_verify
    action_auc = roc_auc_score(harmful, action_score)
    print(f"    action-utility AUC (n={len(harmful)}, unchanged): {action_auc:.4f}")

    point_diff = failure_auc_reduced - action_auc
    n = len(acted_qids_in_order)
    rng = np.random.RandomState(SEED)
    diffs = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.randint(0, n, n)
        yp, pp, h, s = y_poor_reduced[idx], gbdt_proba_reduced[idx], harmful[idx], action_score[idx]
        if not (0 < yp.sum() < n and 0 < h.sum() < n):
            continue
        diffs.append(roc_auc_score(yp, pp) - roc_auc_score(h, s))
    diffs = np.array(diffs)
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    print(f"\n    paired bootstrap difference (GBDT detection - action-utility): "
          f"{point_diff:+.4f}  95% CI=[{ci_lo:+.4f},{ci_hi:+.4f}]")
    print(f"    excludes zero: {ci_lo > 0 or ci_hi < 0}")
    print(f"    (existing logistic-based paired difference, Table 1: +0.091, CI=[+0.045,+0.136])")

    out = {
        "full_population": {"n": len(y_test_poor), "auc": float(point_a), "ci_lo": float(lo_a), "ci_hi": float(hi_a)},
        "acted_upon_subset": {
            "n": n, "detection_auc": float(failure_auc_reduced), "action_utility_auc": float(action_auc),
            "paired_diff": float(point_diff), "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
        },
        "gbdt_params": GBDT_PARAMS,
    }
    with open("results/tripclick_detection_gbdt_frozen.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nsaved to results/tripclick_detection_gbdt_frozen.json")


if __name__ == "__main__":
    main()
