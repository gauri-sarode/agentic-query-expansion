"""EXPLORATORY second amendment of configs/score_sufficiency_protocol.json
(added after the primary and reference results; committed before computing).

1. B_calibrated minus random (the registered condition never computed).
2. Exact-budget vs abstention: B_exact spends the full budget in g order.
3. Decision-value gap at b=1: V_full, V_0, and V_T bracketed between an
   out-of-sample cross-fitted value and the in-sample 10-bin value.

Writes rounded aggregates only to docs/score_sufficiency_decision_value.json.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ss = _load("ss69", "scripts/69_score_sufficiency.py")
ref = _load("ref70", "scripts/70_score_sufficiency_reference.py")
EPS, AVG, K, NBOOT, SEED, KAPPAS = ss.EPS, ss.AVG_BUDGETS, ss.K, ss.NBOOT, ss.SEED, ss.KAPPAS


def exact_gain(g: np.ndarray, d: np.ndarray, b: float, k: float) -> float:
    """Spend exactly floor(b*n) in descending g (no abstention), paying k each; ties filled fractionally."""
    n, m = len(d), ss.cap(b, len(d))
    vals, inv = np.unique(g, return_inverse=True)
    sums = np.bincount(inv, weights=d - k, minlength=len(vals))
    cnts = np.bincount(inv, minlength=len(vals))
    gain, left = 0.0, m
    for j in np.argsort(-vals):
        if left <= 0:
            break
        take = min(left, cnts[j])
        gain += sums[j] * take / cnts[j]
        left -= take
    return gain / n


def policy_set(s, d, g, k):
    """Budget-averaged (10-75%) gains of A, B_calibrated, B_exact and random."""
    A = np.mean([ss.failure_rank_gain(s, d, b, k) for b in AVG])
    B = np.mean([ss.select_gain(g, d, b, k) for b in AVG])
    E = np.mean([exact_gain(g, d, b, k) for b in AVG])
    R = np.mean([b * (d - k).mean() for b in AVG])
    return A, B, E, R


def contrasts(s: np.ndarray, d: np.ndarray) -> dict:
    out = {}
    rng = np.random.default_rng(SEED)
    gs = [ss.crossfit(s, d, rng) for _ in range(ss.REPEATS)]
    brng = np.random.default_rng(SEED)
    idx = [brng.integers(0, len(d), len(d)) for _ in range(NBOOT)]
    for k in KAPPAS:
        pts = np.mean([policy_set(s, d, g, k) for g in gs], axis=0)
        boot = np.array([policy_set(s[i], d[i], ss.crossfit(s[i], d[i], brng), k) for i in idx])
        def summ(j1, j2):
            v = boot[:, j1] - boot[:, j2]
            return {"estimate": float(pts[j1] - pts[j2]), "ci": [float(np.quantile(v, .025)), float(np.quantile(v, .975))],
                    "p": float(min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean())))}
        out[f"k={k}"] = {"B_minus_random": summ(1, 3), "B_exact_minus_A": summ(2, 0), "abstention_B_minus_B_exact": summ(1, 2)}
    return out


def decision_value(d: np.ndarray, sc: dict) -> dict:
    out = {}
    for k in KAPPAS:
        v_full = float(np.maximum(d - k, 0).mean())
        v0 = max(float((d - k).mean()), 0.0)
        row = {"V_full": v_full, "V_0": v0, "scores": {}}
        for m, s in sc.items():
            rng = np.random.default_rng(SEED)
            lo = float(np.mean([((ss.crossfit(s, d, rng) > k) * (d - k)).mean() for _ in range(ss.REPEATS)]))
            hi = float(np.maximum(ss.bin_means(s, d, s, K) - k, 0).mean())
            brng = np.random.default_rng(SEED)
            boot = []
            for _ in range(NBOOT):
                i = brng.integers(0, len(d), len(d))
                di = d[i]
                boot.append(((ss.crossfit(s[i], di, brng) > k) * (di - k)).mean() - max((di - k).mean(), 0.0))
            denom = v_full - v0
            row["scores"][m] = {"V_T_lo": lo, "V_T_hi": hi,
                                "missing_lo_hi": [v_full - hi, v_full - lo],
                                "share_of_info_value": [(lo - v0) / denom if denom > 0 else None,
                                                        (hi - v0) / denom if denom > 0 else None],
                                "V_T_lo_minus_V_0": {"estimate": lo - v0,
                                                     "ci": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]}}
        out[f"k={k}"] = row
    return out


def main() -> None:
    da = _load("da50", "scripts/50_decision_analyses.py")
    res = {"_note": "EXPLORATORY second amendment (post-results, committed before computing). Contrasts are "
                    "budget-averaged over 10-75%. Decision value at b=1: V_T_lo is an achievable out-of-sample "
                    "value, V_T_hi the optimistic in-sample 10-bin value; neither is the population V_T."}
    pb = {}
    for nm, d, F, sc in ref.settings():
        res[nm] = {"contrasts": contrasts(sc["failure"], d), "decision_value": decision_value(d, sc)}
        pb[nm] = res[nm]["contrasts"]["k=0.0"]["B_minus_random"]["p"]
        c, dv = res[nm]["contrasts"]["k=0.0"], res[nm]["decision_value"]["k=0.0"]
        print(f"{nm:24s} B-R {c['B_minus_random']['estimate']:+.4f} {c['B_minus_random']['ci']} | "
              f"Bx-A {c['B_exact_minus_A']['estimate']:+.4f} {c['B_exact_minus_A']['ci']} | "
              f"abst {c['abstention_B_minus_B_exact']['estimate']:+.4f} || Vfull {dv['V_full']:.4f} V0 {dv['V_0']:.4f} "
              + " ".join(f"{m}:[{v['share_of_info_value'][0]:.2f},{v['share_of_info_value'][1]:.2f}]"
                         for m, v in dv["scores"].items() if v["share_of_info_value"][0] is not None), flush=True)
    res["holm_B_minus_random"] = da.holm(pb)

    def rnd(x):
        if isinstance(x, (float, np.floating)):
            return None if np.isnan(x) else round(float(x), 4)
        if isinstance(x, dict):
            return {k: rnd(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [rnd(v) for v in x]
        return x
    Path("docs/score_sufficiency_decision_value.json").write_text(json.dumps(rnd(res), indent=2) + "\n")
    print("done")


if __name__ == "__main__":
    main()
