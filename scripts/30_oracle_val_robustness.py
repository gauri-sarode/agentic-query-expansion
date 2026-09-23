#!/usr/bin/env python3
"""Reviewer concern: the retrospective oracle (mean_q[max(0, benefit(q))],
scripts/14) selects the better system separately for every test query
using the same sparse click labels it is then evaluated on -- a form of
per-query selection optimism, worsened by click-label noise. If the
+0.0246 nDCG@10 TripClick TAIL oracle gain is substantially inflated by
this effect, an oracle computed on a DIFFERENT split under the identical
definition should look meaningfully different (systematically larger,
since validation was used more heavily during development) or wildly
unstable.

This script computes the same oracle definition --
benefit(q) = treatment_ndcg10(q) - baseline_ndcg10(q),
oracle gain = mean_q[max(0, benefit(q))]
-- on TAIL VALIDATION (never used to select the oracle itself; it was
only used to calibrate the verifier and BM25 config) for both TripClick
and the NQ replication, as an out-of-sample cross-check of the oracle
ceiling's stability. This is a robustness check on a quantity already
computed elsewhere, not a new frozen-test touch: validation was always
allowed to be used for any decision (Section IV-B), and no decision is
conditioned on this result.

Usage: PYTHONPATH=. python3 scripts/30_oracle_val_robustness.py
"""
from __future__ import annotations

import json

import numpy as np

N_BOOTSTRAP = 10000
SEED = 42


def oracle_gain_ci(delta: np.ndarray) -> tuple[float, float, float]:
    benefit = np.maximum(0.0, delta)
    point = float(benefit.mean())
    n = len(benefit)
    rng = np.random.RandomState(SEED)
    boots = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        idx = rng.randint(0, n, n)
        boots[i] = benefit[idx].mean()
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main() -> None:
    out = {}

    tc = json.loads(open("results/tripclick-tail-val_verifier_calibration.json").read())
    tc_delta = np.array(tc["y_delta_ndcg"])
    tc_point, tc_lo, tc_hi = oracle_gain_ci(tc_delta)
    print(f"TripClick TAIL validation oracle gain (n={len(tc_delta)}): "
          f"{tc_point:+.4f}  95% CI=[{tc_lo:+.4f},{tc_hi:+.4f}]")
    out["tripclick_tail_val"] = {"n": len(tc_delta), "point": tc_point, "ci_lo": tc_lo, "ci_hi": tc_hi}

    nq = json.loads(open("results/nq_val_calibration.json").read())
    nq_delta = np.array(nq["y_delta_ndcg"])
    nq_point, nq_lo, nq_hi = oracle_gain_ci(nq_delta)
    print(f"NQ validation oracle gain (n={len(nq_delta)}): "
          f"{nq_point:+.4f}  95% CI=[{nq_lo:+.4f},{nq_hi:+.4f}]")
    out["nq_val"] = {"n": len(nq_delta), "point": nq_point, "ci_lo": nq_lo, "ci_hi": nq_hi}

    print("\nFor comparison, the frozen TEST oracle numbers already reported in the paper:")
    print("  TripClick TAIL test: +0.0246 [0.0197,0.0296]  (Table IV, scripts/14)")
    print("  NQ test (n=2,052 extension): +0.0415  (Table X, scripts/29)")

    with open("results/oracle_val_robustness.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nsaved to results/oracle_val_robustness.json")


if __name__ == "__main__":
    main()
