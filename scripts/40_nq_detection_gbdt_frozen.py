#!/usr/bin/env python3
"""NQ counterpart to scripts/39: single confirmatory test touch of the
GBDT detection model (scripts/37's non-linear check) on the frozen
n=2,052 NQ extension test set, using t0 features recomputed by
scripts/38 (deterministic retrieval telemetry, not a new experiment --
see that script's docstring). Mirrors scripts/29's exact methodology
(full-population AUC + acted-upon-subset paired difference vs.
action-utility) so the two numbers it produces are directly comparable
to the paper's existing 0.680 / "+0.119*" NQ numbers.

Usage: PYTHONPATH=. python3 scripts/40_nq_detection_gbdt_frozen.py
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
    val = json.loads(open("results/nq_val_calibration.json").read())
    X_val = np.array(val["X_t0"])
    y_val_poor = (np.array(val["y_ndcg"]) <= 1e-9).astype(int)

    ext = json.loads(open("results/nq_test_extension.json").read())["records"]
    t0 = json.loads(open("results/nq_extension_t0_features.json").read())["t0_features"]
    assert all(r["qid"] in t0 for r in ext), "t0-feature recompute missing some extension qids"

    X_test = np.array([t0[r["qid"]] for r in ext])
    y_test_poor = np.array([r["y_poor"] for r in ext])
    print(f"NQ extension: n={len(ext)}, val n={len(X_val)}")

    model = GradientBoostingClassifier(**GBDT_PARAMS).fit(X_val, y_val_poor)
    test_scores = model.predict_proba(X_test)[:, 1]

    point_a = roc_auc_score(y_test_poor, test_scores)
    lo_a, hi_a = bootstrap_ci(y_test_poor, test_scores)
    print(f"(A) full-population (n={len(y_test_poor)}) GBDT test AUC: {point_a:.4f}  "
          f"95% CI=[{lo_a:.4f},{hi_a:.4f}]")
    print(f"    (existing logistic-regression number, Table 1: 0.680)")

    # --- (B) acted-upon subset + paired difference vs. action-utility, mirroring scripts/29 ---
    acted = [r for r in ext if r["delta_ndcg"] is not None]
    acted_qids = {r["qid"] for r in acted}
    idx_by_qid = {r["qid"]: i for i, r in enumerate(ext)}
    acted_idx = [idx_by_qid[q] for q in [r["qid"] for r in acted]]
    y_poor_acted = y_test_poor[acted_idx]
    gbdt_proba_acted = test_scores[acted_idx]

    failure_auc_acted = roc_auc_score(y_poor_acted, gbdt_proba_acted)
    print(f"\n(B) acted-upon subset (n={len(acted)}) GBDT AUC: {failure_auc_acted:.4f}")

    harmful = np.array([1 if r["delta_ndcg"] < -1e-9 else 0 for r in acted])
    action_score = -np.array([r["predicted_delta"] for r in acted])
    action_auc = roc_auc_score(harmful, action_score)
    print(f"    action-utility AUC (n={len(harmful)}, unchanged): {action_auc:.4f}")

    point_diff = failure_auc_acted - action_auc
    n = len(acted)
    rng = np.random.RandomState(SEED)
    diffs = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.randint(0, n, n)
        yp, pp, h, s = y_poor_acted[idx], gbdt_proba_acted[idx], harmful[idx], action_score[idx]
        if not (0 < yp.sum() < n and 0 < h.sum() < n):
            continue
        diffs.append(roc_auc_score(yp, pp) - roc_auc_score(h, s))
    diffs = np.array(diffs)
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    print(f"\n    paired bootstrap difference (GBDT detection - action-utility): "
          f"{point_diff:+.4f}  95% CI=[{ci_lo:+.4f},{ci_hi:+.4f}]")
    print(f"    excludes zero: {ci_lo > 0 or ci_hi < 0}")
    print(f"    (existing logistic-based paired difference, Table 1: +0.119, CI likely similar magnitude)")

    out = {
        "full_population": {"n": len(y_test_poor), "auc": float(point_a), "ci_lo": float(lo_a), "ci_hi": float(hi_a)},
        "acted_upon_subset": {
            "n": n, "detection_auc": float(failure_auc_acted), "action_utility_auc": float(action_auc),
            "paired_diff": float(point_diff), "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
        },
        "gbdt_params": GBDT_PARAMS,
    }
    with open("results/nq_detection_gbdt_frozen.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nsaved to results/nq_detection_gbdt_frozen.json")


if __name__ == "__main__":
    main()
