#!/usr/bin/env python3
"""Trains and persists the verifier's scoring model as a real artifact,
replacing src/agent/verify.py's previous hardcoded-constant weights
(copy-pasted from a one-off calibration run's printed output) with a
proper train-script -> artifact -> load-at-inference pattern: retraining
means re-running this script, not hand-editing source.

Fits LinearRegression(fit_intercept=False) on the persisted calibration
data (scripts/08_calibrate_verifier.py's output) -- linear is not a
default choice here, it's the validated one: GBDT was tried on this same
TripClick TAIL data and scored worse under 5-fold CV (r=0.045 vs 0.134,
see git history / docs/milestones.md). Only refit with a different model
class if a future calibration run finds one that actually wins.

Usage: python scripts/10_train_verifier_model.py [calibration_json] [model_out]
  defaults: results/tripclick-tail_verifier_calibration.json -> models/verifier_v1.joblib
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict

_DEFAULT_CALIBRATION = "results/tripclick-tail_verifier_calibration.json"
_DEFAULT_MODEL_OUT = "models/verifier_v1.joblib"


def main() -> None:
    calibration_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(_DEFAULT_CALIBRATION)
    model_out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(_DEFAULT_MODEL_OUT)

    data = json.loads(calibration_path.read_text())
    X = np.array(data["X"])
    y = np.array(data["y_delta_ndcg"])
    feature_names = data["feature_names"]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    model = LinearRegression(fit_intercept=False)
    cv_pred = cross_val_predict(model, X, y, cv=kf)
    cv_corr = float(np.corrcoef(cv_pred, y)[0, 1])

    model.fit(X, y)

    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_names": feature_names,
            "trained_on": str(calibration_path),
            "n_samples": len(X),
            "cv_pearson_r": cv_corr,
        },
        model_out,
    )

    print(f"Trained on {calibration_path} (n={len(X)})")
    print(f"5-fold CV Pearson r vs. true NDCG@10 delta: {cv_corr:.4f}")
    print("Coefficients:")
    for name, coef in zip(feature_names, model.coef_):
        print(f"  {name:<22} {coef:+.4f}")
    print(f"\nSaved to {model_out}")


if __name__ == "__main__":
    main()
