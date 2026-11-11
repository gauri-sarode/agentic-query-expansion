"""EXPLORATORY selection diagnostics (configs/partial_probe_protocol.json,
'diagnostics'): which queries failure routing, benefit routing and the
oracle select, by prespecified Delta categories, from existing traces.
No fitting. Writes rounded aggregates to docs/selection_diagnostics.json;
query examples only from public benchmarks (NQ, ArguAna).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.data import load_queries  # noqa: E402


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


EPS = 1e-9
BIG = 0.10
BUDGET = 0.25
CATS = ("rescue", "failing_unchanged", "healthy_large_gain", "healthy_small_gain",
        "healthy_unchanged", "small_harm", "large_harm")


def categorize(d: np.ndarray, F: np.ndarray) -> np.ndarray:
    c = np.empty(len(d), dtype=object)
    c[F & (d > EPS)] = "rescue"
    c[F & (np.abs(d) <= EPS)] = "failing_unchanged"
    h = ~F
    c[h & (d >= BIG)] = "healthy_large_gain"
    c[h & (d > EPS) & (d < BIG)] = "healthy_small_gain"
    c[h & (np.abs(d) <= EPS)] = "healthy_unchanged"
    c[h & (d < -EPS) & (d > -BIG)] = "small_harm"
    c[h & (d <= -BIG)] = "large_harm"
    assert all(x in CATS for x in c), "uncategorised query"
    return c


def select(score: np.ndarray, frac: float) -> np.ndarray:
    k = int(round(frac * len(score)))
    m = np.zeros(len(score), bool)
    m[np.argsort(-score, kind="stable")[:k]] = True
    return m


def diagnose(d: np.ndarray, F: np.ndarray, scorers: dict[str, np.ndarray]) -> dict:
    c = categorize(d, F)
    dp = np.maximum(d, 0)
    out = {"n": int(len(d)),
           "population": {k: float((c == k).mean()) for k in CATS},
           "oracle_gain_share": {k: float(dp[c == k].sum() / dp.sum()) if dp.sum() > 0 else None for k in CATS},
           "budget": BUDGET, "selected": {}}
    sel = {m: select(s, BUDGET) for m, s in scorers.items()}
    for m, mask in sel.items():
        out["selected"][m] = {"composition": {k: float((c[mask] == k).mean()) for k in CATS},
                              "inactive_fraction": float((np.abs(d[mask]) <= EPS).mean()),
                              "mean_delta": float(d[mask].mean()),
                              "jaccard_with_oracle": float((mask & sel["oracle"]).sum() / (mask | sel["oracle"]).sum())}
    return out


def examples(qids, queries, d, F, fail_score, n=2) -> dict:
    """Highest-scored failing queries the intervention leaves unchanged, and the
    largest healthy gains failure routing does not select at the budget."""
    chosen = select(fail_score, BUDGET)
    a = [i for i in np.argsort(-fail_score, kind="stable") if F[i] and abs(d[i]) <= EPS][:n]
    b = [i for i in np.argsort(-d, kind="stable") if not F[i] and not chosen[i]][:n]
    fmt = lambda i: {"query": queries[qids[i]], "delta": round(float(d[i]), 3)}  # noqa: E731
    return {"failure_selected_but_inactive": [fmt(i) for i in a],
            "large_healthy_gain_missed": [fmt(i) for i in b]}


def main() -> None:
    da = _load("da50", "scripts/50_decision_analyses.py")
    iv = da.iv
    res = {"_note": "EXPLORATORY; prespecified categories (configs/partial_probe_protocol.json). "
                    "Selection at a 25% budget; benefit = Table 3 linear model (TripClick/NQ), B1 where available."}
    nq_q = load_queries("nq")
    nq_test = iv.corpus_inputs("nq")["test"]["qids"]
    b1_nq = {a: np.load(f"results/nq_stronger_baselines/{a}_test_scores.npy")[0] for a in ("qe", "rerank", "prf")}
    for S in da.llm_qe_settings() + da.intervention_settings():
        name, d, F = S["name"], S["delta"], S["fail"].astype(bool)
        scorers = {"failure": S["det"], "nqc": -S["nqc"], "benefit_linear": S["benefit_pre"], "oracle": d}
        if name.endswith("_nq"):
            scorers["B1_gbdt"] = b1_nq[{"llm_qe_nq": "qe", "rerank_nq": "rerank", "prf_nq": "prf"}[name]]
        res[name] = diagnose(d, F, scorers)
        if name.endswith("_nq"):
            res[name]["examples"] = examples(nq_test, nq_q, d, F, S["det"])
    pred = pd.read_csv("results/arguana/runner/test_predictions.csv")
    arg_q = load_queries("arguana")
    for action, g in pred.groupby("action"):
        d, F = g["delta"].to_numpy(), (g["original_ndcg"].to_numpy() <= EPS)
        scorers = {"failure": g["failure"].to_numpy(), "nqc": g["nqc"].to_numpy(),
                   "B1_gbdt": g["B1_gbdt"].to_numpy(), "oracle": d}
        res[f"{action}_arguana"] = diagnose(d, F, scorers)
        res[f"{action}_arguana"]["examples"] = examples(g["query_id"].tolist(), arg_q, d, F, g["failure"].to_numpy())

    def rnd(x):
        if isinstance(x, (float, np.floating)):
            return round(float(x), 4)
        if isinstance(x, dict):
            return {k: rnd(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [rnd(v) for v in x]
        return x
    Path("docs/selection_diagnostics.json").write_text(json.dumps(rnd(res), indent=2) + "\n")
    print("done")


if __name__ == "__main__":
    main()
