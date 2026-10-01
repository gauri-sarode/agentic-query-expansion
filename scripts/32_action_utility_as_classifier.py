#!/usr/bin/env python3
"""Reviewer question (WSDM draft): was direct binary classification
tested for the action-utility model, rather than fitting a linear
regression against the continuous nDCG change and using its sign-
flipped score for the reported AUC? The deployed verifier is a
LinearRegression (scripts/10) for reasons unrelated to this paper
(it's also used as a continuous score for the FSM's accept/rollback/
replan thresholds, not just for ranking) -- but the AUC comparison in
Table 1 only needs a score that ranks queries, so it's fair to ask
whether a model fit DIRECTLY as a classifier on the binary harmful
label would score differently.

This script answers that directly rather than arguing it away: fits a
LogisticRegression on the identical 10 post-action features
(TripClick TAIL), 5-fold CV on validation for an honest estimate
(mirroring the failure detector's own methodology exactly), then
scores once on frozen test -- same population, same protocol, as
scripts/17/28's existing 0.552 action-utility AUC.

Scope note: this check is TripClick-only. The NQ confirmatory set's
raw 10-feature vectors were not persisted in results/nq_test_extension.json
(only the deployed LinearRegression's final scalar output was) -- so
this cannot be replicated on NQ without a new feature-extraction pass
over already-frozen NQ test data, which was not done here.

Usage: PYTHONPATH=. python3 scripts/32_action_utility_as_classifier.py
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

SEED = 42
N_BOOTSTRAP = 10000


def main() -> None:
    val = json.loads(open("results/tripclick-tail-val_verifier_calibration.json").read())
    test = json.loads(open("results/tripclick-tail_verifier_calibration.json").read())
    assert val["feature_names"] == test["feature_names"]

    X_val = np.array(val["X"])
    y_val = (np.array(val["y_delta_ndcg"]) < -1e-9).astype(int)
    X_test = np.array(test["X"])
    y_test = (np.array(test["y_delta_ndcg"]) < -1e-9).astype(int)

    print(f"val n={len(X_val)} (harmful rate {y_val.mean():.3f}), "
          f"test n={len(X_test)} (harmful rate {y_test.mean():.3f})")

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    clf = LogisticRegression(max_iter=1000)
    cv_proba = cross_val_predict(clf, X_val, y_val, cv=kf, method="predict_proba")[:, 1]
    cv_auc = roc_auc_score(y_val, cv_proba)
    print(f"5-fold CV AUC on validation: {cv_auc:.4f}")

    clf.fit(X_val, y_val)
    test_proba = clf.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_proba)
    print(f"Frozen test AUC (logistic classifier, direct fit): {test_auc:.4f}")
    print(f"Existing reported action-utility AUC (linear regression, scripts/17): 0.5524 (rounds to 0.552)")

    rng = np.random.RandomState(SEED)
    n = len(y_test)
    boots = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.randint(0, n, n)
        yt, pp = y_test[idx], test_proba[idx]
        if 0 < yt.sum() < n:
            boots.append(roc_auc_score(yt, pp))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"95% CI: [{lo:.4f},{hi:.4f}]")

    out = {
        "cv_auc_val": float(cv_auc),
        "test_auc_logistic": float(test_auc),
        "test_auc_ci_lo": float(lo),
        "test_auc_ci_hi": float(hi),
        "existing_linear_regression_auc": 0.5524,
        "n_val": len(X_val),
        "n_test": len(X_test),
    }
    with open("results/action_utility_as_classifier.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nsaved to results/action_utility_as_classifier.json")


if __name__ == "__main__":
    main()
