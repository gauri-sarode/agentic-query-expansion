#!/usr/bin/env python3
"""Review request: is NQ's failure-detection AUC (0.638) actually
significantly HIGHER than its action-utility AUC (0.522), or just
suggestively so? Both AUCs individually have their own CIs (Table X),
but that doesn't establish the *difference* is non-zero -- needs a
paired bootstrap on the same query population.

Restricted to the n=586 test queries where both quantities are defined
(the 14 NO_EXPAND-skipped queries have no action-utility label), so the
resample draws query indices jointly for both AUCs each iteration --
this also means the failure-detection AUC recomputed here on n=586 may
differ trivially from Table X's n=600 point estimate; reported
separately for transparency.

Usage: PYTHONPATH=. python3 scripts/24_nq_auc_difference_ci.py
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import roc_auc_score

N_BOOTSTRAP = 10000
SEED = 42


def main() -> None:
    d = json.loads(open("results/nq_frozen_test.json").read())
    acted = [r for r in d["records"] if r["delta_ndcg"] is not None]
    print(f"n acted-upon (both AUCs defined): {len(acted)}")

    y_poor = np.array([r["y_poor"] for r in acted])
    poor_proba = np.array([r["poor_proba"] for r in acted])
    harmful = np.array([1 if r["delta_ndcg"] < -1e-9 else 0 for r in acted])
    action_score = -np.array([r["predicted_delta"] for r in acted])

    failure_auc = roc_auc_score(y_poor, poor_proba)
    action_auc = roc_auc_score(harmful, action_score)
    point_diff = failure_auc - action_auc
    print(f"failure-detection AUC (n={len(acted)} subset): {failure_auc:.4f}")
    print(f"action-utility AUC: {action_auc:.4f}")
    print(f"point difference: {point_diff:+.4f}")

    n = len(acted)
    rng = np.random.RandomState(SEED)
    diffs = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.randint(0, n, n)
        yp, pp, h, s = y_poor[idx], poor_proba[idx], harmful[idx], action_score[idx]
        if not (0 < yp.sum() < n and 0 < h.sum() < n):
            continue
        diffs.append(roc_auc_score(yp, pp) - roc_auc_score(h, s))
    diffs = np.array(diffs)
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    print(f"\npaired bootstrap difference: {point_diff:+.4f}  95% CI=[{ci_lo:+.4f},{ci_hi:+.4f}]  (n_resamples={len(diffs)})")
    print(f"excludes zero: {ci_lo > 0 or ci_hi < 0}")


if __name__ == "__main__":
    main()
