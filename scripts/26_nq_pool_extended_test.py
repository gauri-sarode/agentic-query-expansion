#!/usr/bin/env python3
"""Pools scripts/23's original frozen NQ test (n=600) with scripts/25's
pre-committed extension (n=2052, configs/nq_test_extension_precommit.json)
into the final n=2652 result -- same frozen retrieval config, verifier,
and failure detector throughout, nothing retuned between the two runs.
Recomputes the same five numbers as scripts/23, same bootstrap
conventions (10,000 resamples, seed 42).

Usage: PYTHONPATH=. python3 scripts/26_nq_pool_extended_test.py
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import roc_auc_score

N_BOOTSTRAP = 10000
SEED = 42


def paired_bootstrap_diff(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    point = float(b.mean() - a.mean())
    n = len(a)
    rng = np.random.RandomState(SEED)
    diffs = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        idx = rng.randint(0, n, n)
        diffs[i] = b[idx].mean() - a[idx].mean()
    return point, float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def auc_ci(label: np.ndarray, score: np.ndarray) -> tuple[float, float, float]:
    point = float(roc_auc_score(label, score))
    n = len(label)
    rng = np.random.RandomState(SEED)
    aucs = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.randint(0, n, n)
        h, s = label[idx], score[idx]
        if 0 < h.sum() < len(h):
            aucs.append(roc_auc_score(h, s))
    return point, float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def main() -> None:
    original = json.loads(open("results/nq_frozen_test.json").read())["records"]
    extension = json.loads(open("results/nq_test_extension.json").read())["records"]
    records = original + extension
    print(f"pooled n={len(records)} (original {len(original)} + extension {len(extension)})")

    original_ndcg = np.array([r["original_ndcg"] for r in records])
    static_qe = np.array([r["accepted_ndcg"] for r in records])
    oracle = np.array([max(r["original_ndcg"], r["candidate_ndcg"] or r["original_ndcg"]) for r in records])

    static_point, static_lo, static_hi = paired_bootstrap_diff(original_ndcg, static_qe)
    oracle_point, oracle_lo, oracle_hi = paired_bootstrap_diff(original_ndcg, oracle)
    print(f"\nStatic QE gain: {static_point:+.4f}  95% CI=[{static_lo:+.4f},{static_hi:+.4f}]")
    print(f"Oracle gain:    {oracle_point:+.4f}  95% CI=[{oracle_lo:+.4f},{oracle_hi:+.4f}]")

    y_poor = np.array([r["y_poor"] for r in records])
    poor_proba = np.array([r["poor_proba"] for r in records])
    fpoint, flo, fhi = auc_ci(y_poor, poor_proba)
    print(f"Failure-detection AUC: {fpoint:.4f}  95% CI=[{flo:.4f},{fhi:.4f}]  n={len(y_poor)}")

    acted = [r for r in records if r["delta_ndcg"] is not None]
    harmful = np.array([1 if r["delta_ndcg"] < -1e-9 else 0 for r in acted])
    score = -np.array([r["predicted_delta"] for r in acted])
    apoint, alo, ahi = auc_ci(harmful, score)
    print(f"Action-utility AUC: {apoint:.4f}  95% CI=[{alo:.4f},{ahi:.4f}]  n={len(harmful)}")
    print(f"(n acted on: {len(acted)}/{len(records)}, {100 * len(acted) / len(records):.1f}%)")

    # Paired bootstrap of the AUC difference (review request, cf. scripts/24),
    # restricted to the acted-upon subset where both quantities are defined.
    acted_y_poor = np.array([r["y_poor"] for r in acted])
    acted_poor_proba = np.array([r["poor_proba"] for r in acted])
    n_acted = len(acted)
    rng = np.random.RandomState(SEED)
    diffs = []
    point_diff = roc_auc_score(acted_y_poor, acted_poor_proba) - apoint
    for _ in range(N_BOOTSTRAP):
        idx = rng.randint(0, n_acted, n_acted)
        yp, pp, h, s = acted_y_poor[idx], acted_poor_proba[idx], harmful[idx], score[idx]
        if not (0 < yp.sum() < n_acted and 0 < h.sum() < n_acted):
            continue
        diffs.append(roc_auc_score(yp, pp) - roc_auc_score(h, s))
    diffs = np.array(diffs)
    dlo, dhi = np.percentile(diffs, [2.5, 97.5])
    print(f"\nAUC difference (failure - action-utility, acted-upon subset n={n_acted}): "
          f"{point_diff:+.4f}  95% CI=[{dlo:+.4f},{dhi:+.4f}]  excludes zero: {dlo > 0 or dhi < 0}")

    out = {
        "n_pooled": len(records),
        "n_original": len(original),
        "n_extension": len(extension),
        "static_qe_gain": {"point": static_point, "ci_lo": static_lo, "ci_hi": static_hi},
        "oracle_gain": {"point": oracle_point, "ci_lo": oracle_lo, "ci_hi": oracle_hi},
        "failure_auc": {"point": fpoint, "ci_lo": flo, "ci_hi": fhi, "n": len(y_poor)},
        "action_utility_auc": {"point": apoint, "ci_lo": alo, "ci_hi": ahi, "n": len(harmful)},
        "auc_difference": {"point": float(point_diff), "ci_lo": float(dlo), "ci_hi": float(dhi), "n": n_acted},
    }
    with open("results/nq_pooled_test.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nsaved to results/nq_pooled_test.json")


if __name__ == "__main__":
    main()
