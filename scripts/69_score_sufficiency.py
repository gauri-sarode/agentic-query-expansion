"""EXPLORATORY QPP-score sufficiency analysis, as frozen in
configs/score_sufficiency_protocol.json (committed before computing).

One information source, the learned failure gate's score S, two decision
rules: A ranks queries by S (failure routing); B ranks them by a
cross-fitted estimate of E[Delta | S] over 10 equal-frequency bins and
acts only where it exceeds kappa. B_star is the optimistic in-sample bin
oracle; random and the per-query oracle V* bracket them.

Writes rounded aggregates only to docs/score_sufficiency.json.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, ".")


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


EPS = 1e-9
BUDGETS = (0.10, 0.25, 0.50, 0.75, 1.00)
AVG_BUDGETS = (0.10, 0.25, 0.50, 0.75)
KAPPAS = (0.0, 0.02)
K, FOLDS, REPEATS = 10, 5, 20
NBOOT = 10000
SEED = 42


def cap(b: float, n: int) -> int:
    return int(np.floor(b * n + 1e-9))


def select_gain(g: np.ndarray, d: np.ndarray, b: float, k: float) -> float:
    """Intervene in descending g where g > k, up to floor(b*n); ties filled fractionally."""
    n, m = len(d), cap(b, len(d))
    vals, inv = np.unique(g, return_inverse=True)
    sums = np.bincount(inv, weights=d - k, minlength=len(vals))
    cnts = np.bincount(inv, minlength=len(vals))
    gain, left = 0.0, m
    for j in np.argsort(-vals):
        if vals[j] <= k or left <= 0:
            break
        take = min(left, cnts[j])
        gain += sums[j] * take / cnts[j]
        left -= take
    return gain / n


def failure_rank_gain(s: np.ndarray, d: np.ndarray, b: float, k: float) -> float:
    pick = np.argsort(-s, kind="stable")[:cap(b, len(d))]
    return float((d[pick] - k).sum() / len(d))


def bin_means(s_fit: np.ndarray, d_fit: np.ndarray, s_apply: np.ndarray, nbins: int) -> np.ndarray:
    edges = np.quantile(s_fit, np.linspace(0, 1, nbins + 1)[1:-1])
    b_fit = np.searchsorted(edges, s_fit, side="right")
    sums = np.bincount(b_fit, weights=d_fit, minlength=nbins)
    cnts = np.bincount(b_fit, minlength=nbins)
    g = np.where(cnts > 0, sums / np.maximum(cnts, 1), d_fit.mean())
    return g[np.searchsorted(edges, s_apply, side="right")]


def crossfit(s: np.ndarray, d: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    fold = rng.permutation(len(d)) % FOLDS
    g = np.empty(len(d))
    for f in range(FOLDS):
        te = fold == f
        g[te] = bin_means(s[~te], d[~te], s[te], K)
    return g


def oracle_gain(d: np.ndarray, b: float, k: float) -> float:
    return float(np.sort(np.maximum(d - k, 0))[::-1][:cap(b, len(d))].sum() / len(d))


def analyse(s: np.ndarray, d: np.ndarray, F: np.ndarray) -> dict:
    rng = np.random.default_rng(SEED)
    gs = [crossfit(s, d, rng) for _ in range(REPEATS)]
    out = {"n": int(len(d)), "mean_delta": float(d.mean()), "policies": {}}
    for k in KAPPAS:
        for b in BUDGETS:
            v = oracle_gain(d, b, k)
            row = {"oracle": v,
                   "A_failure_ranking": failure_rank_gain(s, d, b, k),
                   "B_calibrated": float(np.mean([select_gain(g, d, b, k) for g in gs])),
                   "random": b * float((d - k).mean())}
            for nb in (5, 10, 20):
                row[f"B_star_K{nb}"] = select_gain(bin_means(s, d, s, nb), d, b, k)
            row["shares_of_oracle"] = {p: (row[p] / v if v > 0 else None) for p in list(row) if p != "oracle"}
            out["policies"][f"k={k},b={b}"] = row

    # Primary contrast: B - A averaged over 10-75% at kappa = 0, bootstrap with refit.
    def contrast(si, di, r):
        g = crossfit(si, di, r)
        return np.mean([select_gain(g, di, b, 0.0) - failure_rank_gain(si, di, b, 0.0) for b in AVG_BUDGETS])

    point = float(np.mean([out["policies"][f"k=0.0,b={b}"]["B_calibrated"] - out["policies"][f"k=0.0,b={b}"]["A_failure_ranking"]
                           for b in AVG_BUDGETS]))
    brng = np.random.default_rng(SEED)
    boot = np.empty(NBOOT)
    for i in range(NBOOT):
        idx = brng.integers(0, len(d), len(d))
        boot[i] = contrast(s[idx], d[idx], brng)
    out["B_minus_A_avg"] = {"estimate": point, "ci": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
                            "p": float(min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean())))}

    # Binned profile (candidate figure).
    edges = np.quantile(s, np.linspace(0, 1, K + 1)[1:-1])
    bins = np.searchsorted(edges, s, side="right")
    prof = []
    prng = np.random.default_rng(SEED)
    for j in range(K):
        m = bins == j
        dj = d[m]
        bm = [dj[prng.integers(0, len(dj), len(dj))].mean() for _ in range(2000)] if len(dj) else [np.nan]
        prof.append({"n": int(m.sum()), "p_fail": float(F[m].mean()) if m.any() else None,
                     "p_helped": float((dj > EPS).mean()) if m.any() else None,
                     "mean_delta": float(dj.mean()) if m.any() else None,
                     "mean_delta_ci": [float(np.quantile(bm, .025)), float(np.quantile(bm, .975))]})
    out["binned_profile"] = prof
    out["spearman_S_delta"] = {"failing": float(spearmanr(s[F], d[F]).statistic) if F.sum() > 2 else None,
                               "healthy": float(spearmanr(s[~F], d[~F]).statistic) if (~F).sum() > 2 else None}
    return out


def settings() -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    da = _load("da50", "scripts/50_decision_analyses.py")
    out = [(S["name"], S["det"], S["delta"], S["fail"].astype(bool))
           for S in da.llm_qe_settings() + da.intervention_settings() + da.dense_settings()]
    for corpus in ("arguana", "fiqa"):
        pred = pd.read_csv(f"results/{corpus}/runner/test_predictions.csv")
        for action, g in pred.groupby("action"):
            out.append((f"{action}_{corpus}", g["failure"].to_numpy(), g["delta"].to_numpy(),
                        g["original_ndcg"].to_numpy() <= EPS))
    return out


def main() -> None:
    da = _load("da50", "scripts/50_decision_analyses.py")
    res = {"_note": "EXPLORATORY (configs/score_sufficiency_protocol.json, committed before computing). "
                    "S = learned failure gate score; A ranks by S, B by cross-fitted E[Delta|S] (10 bins, 5 folds, "
                    "20 repeats); B_star_K = in-sample bin oracle (optimistic). Gains in nDCG@10 per query."}
    names = []
    for nm, s, d, F in settings():
        res[nm] = analyse(s, d, F)
        names.append(nm)
        P = res[nm]["policies"]
        print(f"{nm:24s} B-A avg {res[nm]['B_minus_A_avg']['estimate']:+.4f} {res[nm]['B_minus_A_avg']['ci']}  "
              + " ".join(f"b={b}: A {P[f'k=0.0,b={b}']['shares_of_oracle']['A_failure_ranking']:.2f} "
                         f"B {P[f'k=0.0,b={b}']['shares_of_oracle']['B_calibrated']:.2f} "
                         f"B* {P[f'k=0.0,b={b}']['shares_of_oracle']['B_star_K10']:.2f}" for b in (0.1, 0.25)), flush=True)
    res["holm_B_minus_A"] = da.holm({nm: res[nm]["B_minus_A_avg"]["p"] for nm in names})

    def rnd(x):
        if isinstance(x, (float, np.floating)):
            return None if np.isnan(x) else round(float(x), 4)
        if isinstance(x, dict):
            return {k: rnd(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [rnd(v) for v in x]
        return x
    Path("docs/score_sufficiency.json").write_text(json.dumps(rnd(res), indent=2) + "\n")
    print("done")


if __name__ == "__main__":
    main()
