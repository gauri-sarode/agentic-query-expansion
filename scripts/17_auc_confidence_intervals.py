#!/usr/bin/env python3
"""Bootstrap 95% CIs for RQ1's two frozen-test AUCs (Table II), added
after review: point estimates alone invite the question of how much
sampling noise they carry. Both CIs are computed on the exact frozen
artifacts already used to report the point estimates -- no new frozen-test
touch, purely a re-derivation.

Action-utility AUC (0.552): resample rows of
results/tripclick-tail_verifier_calibration.json (the frozen verifier's
score vs. true nDCG@10 delta on TAIL test, n=1157).

Failure-detection AUC (0.649): refit the frozen logistic model on
X_val/y_val_poor from results/tripclick-tail_failure_detection.json
(matching scripts/... Step 2's exact fitting procedure), score X_test
once, then bootstrap-resample the TEST rows only (the model itself is
not refit per resample -- standard practice for a CI on a frozen
classifier's test-set AUC).

Usage: PYTHONPATH=. python3 scripts/17_auc_confidence_intervals.py
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

N_BOOTSTRAP = 10000
SEED = 42


def bootstrap_ci(harmful: np.ndarray, score: np.ndarray, n_bootstrap: int = N_BOOTSTRAP) -> tuple[float, float]:
    n = len(harmful)
    rng = np.random.RandomState(SEED)
    aucs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        h, s = harmful[idx], score[idx]
        if 0 < h.sum() < len(h):
            aucs.append(roc_auc_score(h, s))
    aucs = np.array(aucs)
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def action_utility_ci() -> None:
    d = json.loads(open("results/tripclick-tail_verifier_calibration.json").read())
    y_delta = np.array(d["y_delta_ndcg"])
    y_verify = np.array(d["y_current_verify"])
    harmful = (y_delta < -1e-9).astype(int)
    score = -y_verify
    point = roc_auc_score(harmful, score)
    ci_lo, ci_hi = bootstrap_ci(harmful, score)
    print(f"action-utility AUC: {point:.4f}  95% CI=[{ci_lo:.4f}, {ci_hi:.4f}]  n={len(harmful)}")


def failure_detection_ci() -> None:
    d = json.loads(open("results/tripclick-tail_failure_detection.json").read())
    X_val, y_val_ndcg = np.array(d["X_val"]), np.array(d["y_val_ndcg"])
    X_test, y_test_ndcg = np.array(d["X_test"]), np.array(d["y_test_ndcg"])
    y_val_poor = (y_val_ndcg <= 1e-9).astype(int)
    y_test_poor = (y_test_ndcg <= 1e-9).astype(int)

    model = LogisticRegression(max_iter=1000).fit(X_val, y_val_poor)
    test_scores = model.predict_proba(X_test)[:, 1]
    point = roc_auc_score(y_test_poor, test_scores)
    ci_lo, ci_hi = bootstrap_ci(y_test_poor, test_scores)
    print(f"failure-detection AUC: {point:.4f}  95% CI=[{ci_lo:.4f}, {ci_hi:.4f}]  n={len(y_test_poor)}")


if __name__ == "__main__":
    action_utility_ci()
    failure_detection_ci()
