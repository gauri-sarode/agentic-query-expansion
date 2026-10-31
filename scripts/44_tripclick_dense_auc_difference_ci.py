#!/usr/bin/env python3
"""Recomputes the TripClick dense-retrieval check's detection-minus-
action-utility AUC difference CI with a true paired bootstrap,
mirroring scripts/28's BM25 methodology.

scripts/36's "paired" bootstrap held detection AUC fixed at its
full-test point value (n=1175) and resampled only the action-utility
side -- not paired, and it understates the difference's variance. Here
both AUCs are scored on the same acted-upon test subset, and every
resample recomputes BOTH on the same resampled queries, exactly as
scripts/28 does (and as configs/nq_dense_precommit.json fixes for the
NQ dense replication, so both dense rows use one procedure).

No new test touch: reuses results/dense_retrieval_full_scale_raw.json's
existing episode records and refits the identical val-only
LogisticRegression models scripts/36 fit (deterministic, so the
scripts/36 point estimates are reproduced first as a sanity check).

Usage: PYTHONPATH=. python3 scripts/44_tripclick_dense_auc_difference_ci.py
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

SEED = 42
N_BOOTSTRAP = 10000
RAW_PATH = "results/dense_retrieval_full_scale_raw.json"
OLD_RESULT_PATH = "results/dense_retrieval_full_scale_result.json"
OUT_PATH = "results/dense_retrieval_auc_difference_paired.json"


def main() -> None:
    raw = json.loads(open(RAW_PATH).read())
    old = json.loads(open(OLD_RESULT_PATH).read())
    val, test = raw["val"], raw["test"]

    # Detection: refit on all val, as scripts/36.
    X_val = np.array([r["t0_features"] for r in val])
    y_val = np.array([r["y_poor"] for r in val])
    det_clf = LogisticRegression(max_iter=1000).fit(X_val, y_val)

    X_test = np.array([r["t0_features"] for r in test])
    y_test = np.array([r["y_poor"] for r in test])
    det_auc_full = roc_auc_score(y_test, det_clf.predict_proba(X_test)[:, 1])

    # Action-utility: refit on val's acted-upon subset, as scripts/36.
    val_acted = [r for r in val if "au_features" in r]
    Xa_val = np.array([r["au_features"] for r in val_acted])
    ya_val = (np.array([r["delta_ndcg"] for r in val_acted]) < -1e-9).astype(int)
    au_clf = LogisticRegression(max_iter=1000).fit(Xa_val, ya_val)

    test_acted = [r for r in test if "au_features" in r]
    Xa_test = np.array([r["au_features"] for r in test_acted])
    harmful = (np.array([r["delta_ndcg"] for r in test_acted]) < -1e-9).astype(int)
    au_score = au_clf.predict_proba(Xa_test)[:, 1]
    au_auc = roc_auc_score(harmful, au_score)

    print(f"sanity vs scripts/36: detection AUC (full test, n={len(y_test)}) "
          f"{det_auc_full:.4f} vs {old['detection_auc']['point']:.4f}; "
          f"action-utility AUC (n={len(harmful)}) {au_auc:.4f} vs {old['action_utility_auc']['point']:.4f}")
    assert abs(det_auc_full - old["detection_auc"]["point"]) < 1e-9
    assert abs(au_auc - old["action_utility_auc"]["point"]) < 1e-9

    # Detection scored on the acted-upon subset (not all 1175) so it's
    # paired with action-utility query for query, as scripts/28.
    y_poor_acted = np.array([r["y_poor"] for r in test_acted])
    det_score_acted = det_clf.predict_proba(np.array([r["t0_features"] for r in test_acted]))[:, 1]
    det_auc_acted = roc_auc_score(y_poor_acted, det_score_acted)
    print(f"detection AUC on acted-upon subset (n={len(y_poor_acted)}): {det_auc_acted:.4f}")

    point_diff = det_auc_acted - au_auc
    print(f"\npoint difference (detection - action-utility): {point_diff:+.4f}")

    n = len(test_acted)
    rng = np.random.RandomState(SEED)
    diffs = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.randint(0, n, n)
        yp, pp, h, s = y_poor_acted[idx], det_score_acted[idx], harmful[idx], au_score[idx]
        if not (0 < yp.sum() < n and 0 < h.sum() < n):
            continue
        diffs.append(roc_auc_score(yp, pp) - roc_auc_score(h, s))
    diffs = np.array(diffs)
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    print(f"paired bootstrap difference: {point_diff:+.4f}  95% CI=[{ci_lo:+.4f},{ci_hi:+.4f}]  (n_resamples={len(diffs)})")
    print(f"excludes zero: {ci_lo > 0 or ci_hi < 0}")
    print(f"previous (scripts/36, fixed-detection) CI: "
          f"[{old['paired_difference']['ci_lo']:+.4f},{old['paired_difference']['ci_hi']:+.4f}]")

    out = {
        "n_acted": n,
        "detection_auc_full_test": float(det_auc_full),
        "detection_auc_on_acted_subset": float(det_auc_acted),
        "action_utility_auc": float(au_auc),
        "point_diff": float(point_diff),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "n_resamples": len(diffs),
        "superseded_scripts36_ci": old["paired_difference"],
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
