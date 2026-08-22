"""Paired bootstrap significance testing, used at every go/no-go
checkpoint in docs/milestones.md (expansion headroom, agency headroom,
recovery headroom).
"""
from __future__ import annotations

import numpy as np


def paired_bootstrap(
    a: list[float],
    b: list[float],
    n_resamples: int = 10_000,
    seed: int = 42,
) -> dict[str, float]:
    """a, b: paired per-query scores (e.g. baseline vs. treatment NDCG@10).
    Returns the observed mean difference (b - a) and a 95% CI via the
    percentile bootstrap over query-level pairs.
    """
    a_arr, b_arr = np.asarray(a), np.asarray(b)
    if len(a_arr) != len(b_arr):
        raise ValueError("a and b must be paired (same length)")
    rng = np.random.default_rng(seed)
    n = len(a_arr)
    diffs = b_arr - a_arr
    observed = diffs.mean()

    idx = rng.integers(0, n, size=(n_resamples, n))
    resampled_means = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(resampled_means, [2.5, 97.5])

    return {
        "mean_diff": float(observed),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "significant": bool(lo > 0 or hi < 0),
    }
