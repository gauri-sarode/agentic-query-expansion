"""EXPLORATORY post-review core-claim audit on the frozen, already inspected
test traces (no fitting, no new labels). Two analyses:

A. Same-target evaluation. Every routing score (failure detector, NQC,
   benefit models, probe-10 where it exists) is scored against one
   outcome, Y = 1[Delta > 0]: help-vs-rest AUC, help-vs-harm AUC among
   changed queries, and Spearman with Delta. Paired bootstrap CIs for
   each score minus the failure detector.
B. Budgeted failure-restricted oracle. With at most b*n interventions
   at cost kappa, V*(b,k) = (1/n) sum of the top floor(b*n) positive
   (Delta-kappa); V_F*(b,k) is the same restricted to failing queries.
   Reports V_F*/V* and H = V* - V_F* with bootstrap CIs. b = 1 recovers
   docs/restricted_oracle.json. Also reports the failure gate's realised
   gain at each cap and the share of its picks that are failing.

Writes rounded aggregates only to docs/core_claim_audit.json.
"""

from __future__ import annotations

import importlib.util
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, ".")


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


EPS = 1e-9
BUDGETS = (0.10, 0.25, 0.50, 0.75, 1.00)
NBOOT = 2000
SEED = 42


def auc(y: np.ndarray, s: np.ndarray) -> float:
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def target_metrics(d: np.ndarray, s: np.ndarray) -> dict:
    ch = np.abs(d) > EPS
    return {"auc_help_vs_rest": auc(d > EPS, s),
            "auc_help_vs_harm": auc(d[ch] > EPS, s[ch]),
            "spearman_delta": float(spearmanr(s, d).statistic)}


def same_target(d: np.ndarray, F: np.ndarray, scorers: dict[str, np.ndarray]) -> dict:
    rng = np.random.default_rng(SEED)
    idx = [rng.integers(0, len(d), len(d)) for _ in range(NBOOT)]
    keys = ("auc_help_vs_rest", "auc_help_vs_harm", "spearman_delta")
    point = {m: target_metrics(d, s) for m, s in scorers.items()}
    boot = {m: np.array([[target_metrics(d[i], s[i])[k] for k in keys] for i in idx]) for m, s in scorers.items()}
    # Within failing queries: does the failure score rank the rescued ones higher?
    fs = scorers["failure"]
    wf = [auc(d[i][F[i]] > EPS, fs[i][F[i]]) for i in idx]
    within = {"estimate": auc(d[F] > EPS, fs[F]), "n_failing": int(F.sum()), "n_rescued": int(((d > EPS) & F).sum()),
              "ci": [float(np.nanquantile(wf, .025)), float(np.nanquantile(wf, .975))]}
    out = {"n": int(len(d)), "p_helped": float((d > EPS).mean()), "n_changed": int((np.abs(d) > EPS).sum()),
           "failure_auc_helped_within_failing": within,
           "failure_auc_of_failure_score": auc(F, scorers["failure"]), "scores": {}}
    for m in scorers:
        out["scores"][m] = {k: {"estimate": point[m][k],
                                "ci": [float(np.nanquantile(boot[m][:, j], .025)), float(np.nanquantile(boot[m][:, j], .975))]}
                            for j, k in enumerate(keys)}
        if m != "failure":
            diff = boot[m] - boot["failure"]
            out["scores"][m]["minus_failure"] = {
                k: {"estimate": point[m][k] - point["failure"][k],
                    "ci": [float(np.nanquantile(diff[:, j], .025)), float(np.nanquantile(diff[:, j], .975))]}
                for j, k in enumerate(keys)}
    return out


def budget_oracle(d: np.ndarray, F: np.ndarray, b: float, k: float) -> tuple[float, float]:
    n, m = len(d), int(np.floor(b * len(d) + 1e-9))
    g = np.maximum(d - k, 0)
    v = np.sort(g)[::-1][:m].sum() / n
    vf = np.sort(g[F])[::-1][:m].sum() / n
    return float(v), float(vf)


def budgeted(d: np.ndarray, F: np.ndarray, kappas, fail_score: np.ndarray) -> dict:
    rng = np.random.default_rng(SEED)
    idx = [rng.integers(0, len(d), len(d)) for _ in range(NBOOT)]
    out = {}
    for k in kappas:
        for b in BUDGETS:
            v, vf = budget_oracle(d, F, b, k)
            bs = np.array([budget_oracle(d[i], F[i], b, k) for i in idx])
            ok = bs[:, 0] > 0
            sh = bs[ok, 1] / bs[ok, 0]
            out[f"k={k},b={b}"] = {"V_star": v, "V_F_star": vf, "H": v - vf,
                                   "share": vf / v if v > 0 else None,
                                   "share_ci": [float(np.quantile(sh, .025)), float(np.quantile(sh, .975))] if ok.any() else None,
                                   "H_ci": [float(np.quantile(bs[:, 0] - bs[:, 1], .025)),
                                            float(np.quantile(bs[:, 0] - bs[:, 1], .975))]}
    # The deployed failure gate at the same caps (top-b by failure score, kappa = 0).
    order = np.argsort(-fail_score, kind="stable")
    for b in BUDGETS:
        pick = order[:int(round(b * len(d)))]
        v, vf = budget_oracle(d, F, b, 0.0)
        out[f"failure_gate,b={b}"] = {"gain": float(d[pick].sum() / len(d)),
                                      "share_of_V_star": float(d[pick].sum() / len(d) / v) if v > 0 else None,
                                      "share_of_V_F_star": float(d[pick].sum() / len(d) / vf) if vf > 0 else None,
                                      "picks_failing": float(F[pick].mean())}
    return out


def main() -> None:
    da = _load("da50", "scripts/50_decision_analyses.py")
    pp = _load("pp60", "scripts/60_partial_probe.py")
    res = {"_note": "EXPLORATORY post-review audit of frozen, inspected test traces (ArguAna: pre-registered test, "
                    "re-used); no fitting; not independent confirmation. Scores are oriented as used for routing "
                    "(higher = route). Y = 1[Delta>0]."}
    b1_nq = {a: np.load(f"results/nq_stronger_baselines/{a}_test_scores.npy")[0] for a in ("qe", "rerank", "prf")}
    probes = {}
    for corpus in ("nq", "tripclick", "arguana"):
        T = pp.corpus_data(corpus)["test"]
        probes[corpus] = (T["delta"], np.load(pp.PROBES / f"{corpus}_test_probe_scores.npy")[pp.SIZES.index(pp.PRIMARY)])

    settings = []
    for S in da.llm_qe_settings() + da.intervention_settings() + da.dense_settings():
        sc = {"failure": S["det"]}
        if S.get("nqc") is not None:
            sc["nqc"] = -S["nqc"]
        if S.get("benefit_pre") is not None:
            sc["benefit_linear"] = S["benefit_pre"]
        nm = S["name"]
        if nm in ("llm_qe_nq", "rerank_nq", "prf_nq"):
            sc["B1_gbdt"] = b1_nq[{"llm_qe_nq": "qe", "rerank_nq": "rerank", "prf_nq": "prf"}[nm]]
        if nm in ("rerank_nq", "rerank_tripclick"):
            pd_, ps = probes[nm.split("_")[1]]
            assert np.allclose(pd_, S["delta"]), f"probe order mismatch {nm}"
            sc["probe_10"] = ps
        settings.append((nm, S["delta"], S["fail"].astype(bool), sc))
    pred = pd.read_csv("results/arguana/runner/test_predictions.csv")
    for action, g in pred.groupby("action"):
        d = g["delta"].to_numpy()
        sc = {"failure": g["failure"].to_numpy(), "nqc": g["nqc"].to_numpy(), "B1_gbdt": g["B1_gbdt"].to_numpy()}
        if action == "rerank":
            pd_, ps = probes["arguana"]
            assert np.allclose(pd_, d), "probe order mismatch arguana"
            sc["probe_10"] = ps
        settings.append((f"{action}_arguana", d, g["original_ndcg"].to_numpy() <= EPS, sc))

    for nm, d, F, sc in settings:
        res[nm] = {"same_target": same_target(d, F, sc), "budgeted_oracle": budgeted(d, F, da.KAPPAS, sc["failure"])}
        st = res[nm]["same_target"]["scores"]
        print(f"{nm:26s} pH={np.mean(d > EPS):.3f} " + " | ".join(
            f"{m}: {v['auc_help_vs_rest']['estimate']:.3f}/{v['auc_help_vs_harm']['estimate']:.3f}/{v['spearman_delta']['estimate']:+.3f}"
            for m, v in st.items()))
        bo = res[nm]["budgeted_oracle"]
        print(" " * 27 + "share k=0: " + " ".join(f"{bo[f'k=0.0,b={b}']['share']:.2f}" for b in BUDGETS)
              + "  k=.02: " + " ".join(f"{(bo[f'k=0.02,b={b}']['share'] or float('nan')):.2f}" for b in BUDGETS))

    def rnd(x):
        if isinstance(x, (float, np.floating)):
            return None if np.isnan(x) else round(float(x), 4)
        if isinstance(x, dict):
            return {k: rnd(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [rnd(v) for v in x]
        return x
    Path("docs/core_claim_audit.json").write_text(json.dumps(rnd(res), indent=2) + "\n")
    print("done")


if __name__ == "__main__":
    main()
