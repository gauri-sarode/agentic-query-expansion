#!/usr/bin/env python3
"""Review request: is TripClick's failure-detection AUC (0.649) actually
significantly HIGHER than its action-utility AUC (0.552), or just
suggestively so via non-overlapping individual CIs? Needs a paired
bootstrap on the same query population -- mirrors scripts/24's NQ
version, adapted to TripClick's two separately-computed artifacts.

Alignment problem this script solves: the failure-detection artifact
(results/tripclick-tail_failure_detection.json) has explicit test_qids
for its full n=1175 population, but the action-utility artifact
(results/tripclick-tail_verifier_calibration.json, from scripts/08) has
no stored qids -- only feature/label arrays in whatever order scripts/08
iterated the frozen query pool. Verified before trusting any pairing:

  1. A fresh enumeration of tripclick-tail's [qid for qid in queries if
     qid in qrels] matches failure_detection.json's test_qids EXACTLY,
     position for position (deterministic, reproducible query ordering
     -- no raw-file reshuffling between runs).
  2. verifier_calibration.json's n=1157 = 1175 - 18, matching its own
     stated skipped_no_expand=18 and Table II's reported n=1157 exactly.
  3. Independent cross-check: results/tripclick-tail_agent_vs_static_checkpoint.json
     (the RQ2 frozen-test run, a DIFFERENT script than scripts/08's
     calibration) finds the identical count -- 18 NO_EXPAND, 1157 acted
     -- over the identical 1175-qid population (set-verified equal to
     failure_detection.json's test_qids). Since operator selection is a
     deterministic function of each query's initial telemetry (no
     randomness), two independent runs agreeing on which 18 queries get
     NO_EXPAND is strong convergent evidence scripts/08's 1157-row order
     is fd's test_qids with exactly those 18 positions dropped, in the
     same relative order.

Given that, this script: filters failure_detection.json's (test_qids,
X_test, y_test_ndcg) down to the "acted" subset using the checkpoint's
qid->planner_action map (not position -- qid-keyed, so this step needs
no positional trust at all), then pairs the resulting reduced-1157
ordering positionally with verifier_calibration.json's 1157 rows.

Usage: PYTHONPATH=. python3 scripts/28_tripclick_auc_difference_ci.py
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

N_BOOTSTRAP = 10000
SEED = 42


def main() -> None:
    fd = json.loads(open("results/tripclick-tail_failure_detection.json").read())
    vc = json.loads(open("results/tripclick-tail_verifier_calibration.json").read())
    ck = json.loads(open("results/tripclick-tail_agent_vs_static_checkpoint.json").read())

    action_by_qid = {t["query_id"]: t["planner_action"] for t in ck["static_traces"]}
    assert set(action_by_qid) == set(fd["test_qids"]), "population mismatch between checkpoint and failure-detection artifact"

    acted_qids_in_order = [q for q in fd["test_qids"] if action_by_qid[q] != "NO_EXPAND"]
    print(f"acted-upon (non-NO_EXPAND) queries: {len(acted_qids_in_order)}")
    assert len(acted_qids_in_order) == len(vc["y_delta_ndcg"]) == 1157, "reduced population size must match verifier_calibration.json's n=1157"

    idx_by_qid = {q: i for i, q in enumerate(fd["test_qids"])}
    reduced_idx = [idx_by_qid[q] for q in acted_qids_in_order]
    X_test_reduced = np.array(fd["X_test"])[reduced_idx]
    y_test_ndcg_reduced = np.array(fd["y_test_ndcg"])[reduced_idx]
    y_poor_reduced = (y_test_ndcg_reduced <= 1e-9).astype(int)

    # Failure detector: same refit-on-val, score-on-test methodology as scripts/17,
    # scored here on the acted-upon subset (not all 1175) so it's paired with action-utility.
    X_val, y_val_ndcg = np.array(fd["X_val"]), np.array(fd["y_val_ndcg"])
    y_val_poor = (y_val_ndcg <= 1e-9).astype(int)
    failure_model = LogisticRegression(max_iter=1000).fit(X_val, y_val_poor)
    poor_proba_reduced = failure_model.predict_proba(X_test_reduced)[:, 1]

    failure_auc_reduced = roc_auc_score(y_poor_reduced, poor_proba_reduced)
    print(f"failure-detection AUC on acted-upon subset (n={len(y_poor_reduced)}): {failure_auc_reduced:.4f}")

    # Action-utility: exact frozen artifact/convention from scripts/17 (harmful=1 scores higher via -y_current_verify).
    y_delta_ndcg = np.array(vc["y_delta_ndcg"])
    y_current_verify = np.array(vc["y_current_verify"])
    harmful = (y_delta_ndcg < -1e-9).astype(int)
    action_score = -y_current_verify
    action_auc = roc_auc_score(harmful, action_score)
    print(f"action-utility AUC (n={len(harmful)}): {action_auc:.4f}")

    point_diff = failure_auc_reduced - action_auc
    print(f"\npoint difference (failure - action-utility): {point_diff:+.4f}")

    n = len(acted_qids_in_order)
    rng = np.random.RandomState(SEED)
    diffs = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.randint(0, n, n)
        yp, pp, h, s = y_poor_reduced[idx], poor_proba_reduced[idx], harmful[idx], action_score[idx]
        if not (0 < yp.sum() < n and 0 < h.sum() < n):
            continue
        diffs.append(roc_auc_score(yp, pp) - roc_auc_score(h, s))
    diffs = np.array(diffs)
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    print(f"\npaired bootstrap difference: {point_diff:+.4f}  95% CI=[{ci_lo:+.4f},{ci_hi:+.4f}]  (n_resamples={len(diffs)})")
    print(f"excludes zero: {ci_lo > 0 or ci_hi < 0}")

    out = {
        "n_acted": n,
        "failure_auc_on_acted_subset": float(failure_auc_reduced),
        "action_utility_auc": float(action_auc),
        "point_diff": float(point_diff),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
    }
    with open("results/tripclick-tail_auc_difference.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nsaved to results/tripclick-tail_auc_difference.json")


if __name__ == "__main__":
    main()
