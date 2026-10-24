#!/usr/bin/env python3
"""Robustness check on scripts/37's finding: is the detection model's
validation-CV advantage over logistic regression specific to one GBDT
hyperparameter choice, or general across non-linear model classes?
Sweeps 8 non-linear classifiers (3 GBDT configs, HistGradientBoosting,
RandomForest, ExtraTrees, SVM-RBF, a small MLP) against the same 8
pre-action features, 5-fold CV on validation, both corpora.

Deliberately validation-only, like scripts/37: scripts/39-40 already
spent this paper's one allowed test touch on the specific GBDT config
chosen before any of this sweeping happened (for parity with the
action-utility non-linear check, scripts/08). Picking whichever model
scores best here and then touching test with IT would be exactly the
test-conditioned, post-hoc model selection the paper's "Experimental
integrity" paragraph commits against -- so this script reports CV
numbers only and does not touch test with any of these models.

Usage: PYTHONPATH=. python3 scripts/42_detection_model_sweep.py
"""
from __future__ import annotations

import json
import warnings

import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
SEED = 42


def _models() -> dict:
    return {
        "logistic_deployed": LogisticRegression(max_iter=1000),
        "gbdt_existing_n30_d2_lr0.1": GradientBoostingClassifier(n_estimators=30, max_depth=2, learning_rate=0.1, random_state=SEED),
        "gbdt_n100_d3_lr0.05": GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=SEED),
        "gbdt_n50_d2_lr0.05": GradientBoostingClassifier(n_estimators=50, max_depth=2, learning_rate=0.05, random_state=SEED),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=SEED, max_iter=100),
        "random_forest_n200_d4": RandomForestClassifier(n_estimators=200, max_depth=4, random_state=SEED),
        "extra_trees_n200_d4": ExtraTreesClassifier(n_estimators=200, max_depth=4, random_state=SEED),
        "svm_rbf_scaled": make_pipeline(StandardScaler(), SVC(probability=True, random_state=SEED)),
        "mlp_16_scaled": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(16,), max_iter=2000, random_state=SEED)),
    }


def main() -> None:
    fd = json.loads(open("results/tripclick-tail_failure_detection.json").read())
    nq = json.loads(open("results/nq_val_calibration.json").read())
    datasets = {
        "tripclick": (np.array(fd["X_val"]), (np.array(fd["y_val_ndcg"]) <= 1e-9).astype(int)),
        "nq": (np.array(nq["X_t0"]), (np.array(nq["y_ndcg"]) <= 1e-9).astype(int)),
    }

    out = {}
    for dname, (X, y) in datasets.items():
        print(f"=== {dname} (n={len(X)}, poor rate {y.mean():.3f}) ===")
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        out[dname] = {}
        for mname, model in _models().items():
            proba = cross_val_predict(model, X, y, cv=kf, method="predict_proba")[:, 1]
            auc = roc_auc_score(y, proba)
            out[dname][mname] = float(auc)
            print(f"  {mname:<32} CV AUC = {auc:.4f}")
        print()

    print("No test touch performed with any of these models (validation-CV-only robustness check).")
    with open("results/detection_model_sweep.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved to results/detection_model_sweep.json")


if __name__ == "__main__":
    main()
