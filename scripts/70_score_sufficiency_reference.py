"""EXPLORATORY reference comparison for the score-sufficiency analysis
(amendment of configs/score_sufficiency_protocol.json, added after the
primary results and committed before computing).

Applies the same in-sample 10-bin oracle (B_star_K10) and cross-fitted
calibrated rule (B_calibrated) to each stored benefit-oriented score and
to NQC, so the failure score's share of V* has a reference. Also writes
per-bin P(F) and P(Delta>0) with bootstrap CIs for the candidate figure.

Writes rounded aggregates only to docs/score_sufficiency_reference.json.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ss = _load("ss69", "scripts/69_score_sufficiency.py")
EPS, BUDGETS, AVG, K, NBOOT, SEED = ss.EPS, ss.BUDGETS, ss.AVG_BUDGETS, ss.K, ss.NBOOT, ss.SEED
FIG_BOOT = 2000


def settings() -> list[tuple[str, np.ndarray, np.ndarray, dict]]:
    """(name, delta, F, scores) with scores oriented higher = route; 'failure' is S."""
    da = _load("da50", "scripts/50_decision_analyses.py")
    pp = _load("pp60", "scripts/60_partial_probe.py")
    b1_nq = {a: np.load(f"results/nq_stronger_baselines/{a}_test_scores.npy")[0] for a in ("qe", "rerank", "prf")}
    probes = {}
    for corpus in ("nq", "tripclick", "arguana"):
        T = pp.corpus_data(corpus)["test"]
        probes[corpus] = (T["delta"], np.load(pp.PROBES / f"{corpus}_test_probe_scores.npy")[pp.SIZES.index(pp.PRIMARY)])
    out = []
    for S in da.llm_qe_settings() + da.intervention_settings() + da.dense_settings():
        nm, sc = S["name"], {"failure": S["det"]}
        if S.get("nqc") is not None:
            sc["nqc"] = -S["nqc"]
        if S.get("benefit_pre") is not None:
            sc["benefit_linear"] = S["benefit_pre"]
        if nm in ("llm_qe_nq", "rerank_nq", "prf_nq"):
            sc["B1_gbdt"] = b1_nq[{"llm_qe_nq": "qe", "rerank_nq": "rerank", "prf_nq": "prf"}[nm]]
        if nm in ("rerank_nq", "rerank_tripclick"):
            pd_, ps = probes[nm.split("_")[1]]
            assert np.allclose(pd_, S["delta"]), f"probe order mismatch {nm}"
            sc["probe_10"] = ps
        out.append((nm, S["delta"], S["fail"].astype(bool), sc))
    for corpus in ("arguana", "fiqa"):
        pred = pd.read_csv(f"results/{corpus}/runner/test_predictions.csv")
        for action, g in pred.groupby("action"):
            d = g["delta"].to_numpy()
            sc = {"failure": g["failure"].to_numpy(), "nqc": g["nqc"].to_numpy(), "B1_gbdt": g["B1_gbdt"].to_numpy()}
            if corpus == "arguana" and action == "rerank":
                pd_, ps = probes["arguana"]
                assert np.allclose(pd_, d), "probe order mismatch arguana"
                sc["probe_10"] = ps
            out.append((f"{action}_{corpus}", d, g["original_ndcg"].to_numpy() <= EPS, sc))
    return out


def bstar(s: np.ndarray, d: np.ndarray, b: float) -> float:
    return ss.select_gain(ss.bin_means(s, d, s, K), d, b, 0.0)


def analyse(d: np.ndarray, F: np.ndarray, sc: dict) -> dict:
    rng = np.random.default_rng(SEED)
    out = {"scores": {}}
    for m, s in sc.items():
        gs = [ss.crossfit(s, d, rng) for _ in range(ss.REPEATS)]
        row = {}
        for b in BUDGETS:
            v = ss.oracle_gain(d, b, 0.0)
            row[f"b={b}"] = {"B_star_K10": bstar(s, d, b) / v if v > 0 else None,
                             "B_calibrated": float(np.mean([ss.select_gain(g, d, b, 0.0) for g in gs])) / v if v > 0 else None,
                             "rank": ss.failure_rank_gain(s, d, b, 0.0) / v if v > 0 else None}
        out["scores"][m] = row
    brng = np.random.default_rng(SEED)
    idx = [brng.integers(0, len(d), len(d)) for _ in range(NBOOT)]
    for m, s in sc.items():
        if m == "failure":
            continue
        point = np.mean([bstar(s, d, b) - bstar(sc["failure"], d, b) for b in AVG])
        boot = np.array([np.mean([bstar(s[i], d[i], b) - bstar(sc["failure"][i], d[i], b) for b in AVG]) for i in idx])
        out["scores"][m]["B_star_minus_failure_avg"] = {"estimate": float(point),
                                                         "ci": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]}
    # Figure data: deciles of S, P(F) and P(Delta>0) with bootstrap CIs.
    s = sc["failure"]
    bins = np.searchsorted(np.quantile(s, np.linspace(0, 1, K + 1)[1:-1]), s, side="right")
    frng = np.random.default_rng(SEED)
    prof = []
    for j in range(K):
        f, h = F[bins == j].astype(float), (d[bins == j] > EPS).astype(float)
        bi = [frng.integers(0, len(f), len(f)) for _ in range(FIG_BOOT)]
        prof.append({"n": int(len(f)), "p_fail": float(f.mean()), "p_helped": float(h.mean()),
                     "p_fail_ci": [float(np.quantile([f[i].mean() for i in bi], q)) for q in (.025, .975)],
                     "p_helped_ci": [float(np.quantile([h[i].mean() for i in bi], q)) for q in (.025, .975)]})
    out["figure_profile"] = prof
    return out


def main() -> None:
    res = {"_note": "EXPLORATORY amendment (post-results, committed before computing). Shares of V* at kappa=0 for "
                    "each score under the in-sample 10-bin oracle (B_star_K10), the cross-fitted calibrated rule, and "
                    "plain ranking; B_star(T) - B_star(failure) averaged over 10-75%."}
    for nm, d, F, sc in settings():
        res[nm] = analyse(d, F, sc)
        r = res[nm]["scores"]
        print(f"{nm:24s} " + " | ".join(
            f"{m}: B* {r[m]['b=0.25']['B_star_K10']:.2f} rank {r[m]['b=0.25']['rank']:+.2f}"
            + (f" diff {r[m]['B_star_minus_failure_avg']['estimate']:+.4f} [{r[m]['B_star_minus_failure_avg']['ci'][0]:+.4f},"
               f"{r[m]['B_star_minus_failure_avg']['ci'][1]:+.4f}]" if m != "failure" else "")
            for m in r), flush=True)

    def rnd(x):
        if isinstance(x, (float, np.floating)):
            return None if np.isnan(x) else round(float(x), 4)
        if isinstance(x, dict):
            return {k: rnd(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [rnd(v) for v in x]
        return x
    Path("docs/score_sufficiency_reference.json").write_text(json.dumps(rnd(res), indent=2) + "\n")
    print("done")


if __name__ == "__main__":
    main()
