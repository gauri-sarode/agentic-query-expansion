#!/usr/bin/env python3
"""Reviewer gap (raised in an outside review): the action-utility model's non-linear check (scripts/08's GBDT
vs. linear comparison, r=0.045 vs 0.134, Section 4 of the WSDM draft)
was never repeated for the failure DETECTION model -- Table 1's 0.649
TripClick AUC comes from a LogisticRegression never compared against
anything else. This answers that directly: same features
(results/tripclick-tail_failure_detection.json, the frozen Table 1
artifact), same CV protocol (5-fold, seed 42) scripts/17 used to
compute the reported 0.649, and the same GBDT hyperparameters
scripts/08 used for the action-utility check (n_estimators=30,
max_depth=2, learning_rate=0.1, random_state=42) -- substituting
GradientBoostingClassifier for the regressor since detection's target
is already binary.

This stops at the validation CV AUC deliberately. Table 1's test AUC
is already frozen and reported (0.649); touching test again with a
second model here, after already seeing this val-CV result, would be
exactly the kind of post-hoc test-conditioned choice the paper's own
"Experimental integrity" paragraph commits against. If the GBDT CV AUC
had clearly beaten logistic regression's, that would be a reason to
STOP and ask before going further (freezing a new model and touching
test is a bigger decision than a backfilled sentence). It does not.

Usage: PYTHONPATH=. python3 scripts/37_detection_nonlinear_check.py
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

SEED = 42


def main() -> None:
    d = json.loads(open("results/tripclick-tail_failure_detection.json").read())
    X_val = np.array(d["X_val"])
    y_val = (np.array(d["y_val_ndcg"]) <= 1e-9).astype(int)
    print(f"val n={len(X_val)} (poor-retrieval rate {y_val.mean():.3f})")

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    logit = LogisticRegression(max_iter=1000)
    logit_cv_proba = cross_val_predict(logit, X_val, y_val, cv=kf, method="predict_proba")[:, 1]
    logit_cv_auc = roc_auc_score(y_val, logit_cv_proba)
    print(f"LogisticRegression 5-fold CV AUC on validation: {logit_cv_auc:.4f}")
    print(f"  (matches stored val_cv_auc in the frozen artifact: {d['val_cv_auc']:.4f})")

    gbdt = GradientBoostingClassifier(n_estimators=30, max_depth=2, learning_rate=0.1, random_state=SEED)
    gbdt_cv_proba = cross_val_predict(gbdt, X_val, y_val, cv=kf, method="predict_proba")[:, 1]
    gbdt_cv_auc = roc_auc_score(y_val, gbdt_cv_proba)
    print(f"GradientBoostingClassifier 5-fold CV AUC on validation: {gbdt_cv_auc:.4f}")

    print(f"\nDifference (GBDT - logistic): {gbdt_cv_auc - logit_cv_auc:+.4f}")
    print("Test was NOT touched with the GBDT model -- this is a validation-only "
          "robustness check, mirroring scripts/08's action-utility non-linear check.")

    out = {
        "logit_cv_auc_val": float(logit_cv_auc),
        "gbdt_cv_auc_val": float(gbdt_cv_auc),
        "gbdt_hyperparams": {"n_estimators": 30, "max_depth": 2, "learning_rate": 0.1, "random_state": SEED},
        "n_val": len(X_val),
        "note": "validation-CV-only check; test not touched with the GBDT model",
    }
    with open("results/detection_nonlinear_check.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nsaved to results/detection_nonlinear_check.json")


if __name__ == "__main__":
    main()
