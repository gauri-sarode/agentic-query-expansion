#!/usr/bin/env python3
"""Independent information-value test (DBPedia-Entity, HotpotQA), as
pre-committed in configs/info_value_precommit.json. This file is committed
before any test label of either corpus is read; test mode runs once per corpus.

Modes:
  time            label-free latency on 50 val queries (scripts/68 procedure)
  select-prf      PRF setting by val mean nDCG@10 (val qrels only)
  val-prediction  mean reranking Delta on val -> the P4 prediction
                  (docs/{corpus}_val_prediction.json; val qrels only)
  export          {corpus}_frozen[_fold{k}].csv in the scripts/54 input schema
                  (reads test qrels: part of the single test pass)
  test            runner per fold + probes + P1-P5 (one Holm family, reranking)
                  + descriptive -> docs/{corpus}_summary.json (aggregates only)

DBPedia has no training split: the 400 test queries are cross-fitted in the
5 folds of configs/info_value_split.json (fit on folds k+1..k+3, select on
k+4, score fold k); every test query is scored once by models that never saw it.

Usage: PYTHONPATH=. python3 scripts/76_iv_analysis.py CORPUS MODE
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.data import load_qrels  # noqa: E402


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


CORPUS = sys.argv[1]
da = _load("da50", "scripts/50_decision_analyses.py")
iv = da.iv
r75 = _load("r75", "scripts/75_iv_runs.py")
ss = _load("ss69", "scripts/69_score_sufficiency.py")
dv72 = _load("dv72", "scripts/72_decision_value.py")
mech = _load("mech73", "scripts/73_mechanism.py")
RUNS = Path(f"results/{CORPUS}")
SPLIT = r75.SPLIT
QRELS_NAME = {"dbpedia": {"val": "dbpedia-dev", "test": "dbpedia-test"},
              "hotpotqa": {"train": "hotpotqa-train", "val": "hotpotqa-dev", "test": "hotpotqa-test"}}[CORPUS]
ACTIONS = ("rerank", "prf")
EPS = 1e-9
NBOOT = 10000
SEED = 42
KAPPAS = (0.0, 0.01, 0.02)
AVG = (0.10, 0.25, 0.50, 0.75)
FEATS = ["top_score", "score_margin", "score_entropy", "query_term_coverage", "mean_rare_term_idf",
         "entity_overlap", "result_coherence", "ranking_stability", "nqc", "qlen"]


def _runs(step: str) -> dict:
    return json.loads((RUNS / f"{step}.json").read_text())


def _ndcg(qids: list[str], rankings: dict, qrels: dict) -> np.ndarray:
    return iv.ndcg(qids, {q: [d for d, _ in rankings[q]] for q in qids}, qrels)


def _prf() -> tuple[int, int, float]:
    p = json.loads((RUNS / "prf_selected.json").read_text())
    return p["fb_docs"], p["fb_terms"], p["lam"]


def roles() -> list[dict[str, list[str]]]:
    """Train/val/test role assignments: one for HotpotQA, five folds for DBPedia."""
    if CORPUS == "hotpotqa":
        return [{"train": SPLIT["train"], "val": SPLIT["val"], "test": SPLIT["test"]}]
    fold = SPLIT["test_fold"]
    ids = lambda ks: [q for q in SPLIT["test"] if fold[q] in ks]  # noqa: E731
    return [{"train": ids({(k + 1) % 5, (k + 2) % 5, (k + 3) % 5}), "val": ids({(k + 4) % 5}), "test": ids({k})}
            for k in range(5)]


def qrels_for(role_split: str) -> dict:
    """Qrels for a role. DBPedia's roles are all test-split queries."""
    return load_qrels(QRELS_NAME["test"] if CORPUS == "dbpedia" else QRELS_NAME[role_split])


# ---------------------------------------------------------------- label-free / val only

def time_val() -> None:
    r68 = _load("r68", "scripts/68_fiqa_analysis.py")
    r68.RUNS, r68.SPLIT, r68.r67 = RUNS, SPLIT, r75
    r68.time_val()


def select_prf() -> None:
    qrels = load_qrels(QRELS_NAME["val"])
    bm25 = {q: v["ranking"] for q, v in _runs("bm25").items()}
    res = {}
    for d, t in iv.PRF_RETRIEVAL:
        exp = _runs(f"prf_d{d}_t{t}")
        for lam in sorted({g[2] for g in iv.PRF_GRID}):
            fused = {q: iv.prf_fuse(bm25[q], exp[q]["ranking"], lam) for q in SPLIT["val"]}
            res[(d, t, lam)] = float(_ndcg(SPLIT["val"], fused, qrels).mean())
    best = max(res, key=res.get)
    out = {"fb_docs": best[0], "fb_terms": best[1], "lam": best[2], "val_ndcg": res[best],
           "val_bm25": float(_ndcg(SPLIT["val"], bm25, qrels).mean()), "n_settings": len(res)}
    (RUNS / "prf_selected.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def val_prediction() -> None:
    qrels = load_qrels(QRELS_NAME["val"])
    bm25 = {q: v["ranking"] for q, v in _runs("bm25").items()}
    rr = {q: v["ranking"] for q, v in _runs("rerank").items()}
    d = _ndcg(SPLIT["val"], rr, qrels) - _ndcg(SPLIT["val"], bm25, qrels)
    out = {"_note": "configs/info_value_precommit.json P4, recorded before test labels are read.",
           "val_mean_rerank_delta": round(float(d.mean()), 4), "n_val": len(d),
           "prediction": "P4 positive" if d.mean() > 0 else "P4 no significant difference"}
    Path(f"docs/{CORPUS}_val_prediction.json").write_text(json.dumps(out, indent=2) + "\n")
    print(out)


# ---------------------------------------------------------------- test pass

def _rankings() -> dict:
    bm25 = _runs("bm25")
    fd, ft, lam = _prf()
    exp = _runs(f"prf_d{fd}_t{ft}")
    return {"rerank": {q: v["ranking"] for q, v in _runs("rerank").items()},
            "prf": {q: iv.prf_fuse(bm25[q]["ranking"], exp[q]["ranking"], lam) for q in exp}}


def export() -> None:
    bm25, rk = _runs("bm25"), _rankings()
    emb = np.load(RUNS / "embed.npy")
    row = {q: i for i, q in enumerate(json.loads((RUNS / "embed_ids.json").read_text()))}
    for k, R in enumerate(roles()):
        path = RUNS / (f"{CORPUS}_frozen.csv" if CORPUS == "hotpotqa" else f"{CORPUS}_frozen_fold{k}.csv")
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["query_id", "split", "baseline_ndcg", *[f"{a}_ndcg" for a in ACTIONS],
                        *[f"z_{n}" for n in FEATS], *[f"emb_{j:03d}" for j in range(emb.shape[1])], "nqc_score"])
            for sp in ("train", "val", "test"):
                qrels = qrels_for(sp)
                ids = R[sp]
                b = _ndcg(ids, {q: bm25[q]["ranking"] for q in ids}, qrels)
                n = {a: _ndcg(ids, rk[a], qrels) for a in ACTIONS}
                for i, q in enumerate(ids):
                    r = bm25[q]
                    w.writerow([q, sp, b[i], *[n[a][i] for a in ACTIONS], *r["t0"], r["nqc"], r["qlen"],
                                *emb[row[q]], r["nqc"]])
        print(path, hashlib.sha256(path.read_bytes()).hexdigest())


def _p(draws: np.ndarray) -> float:
    draws = draws[~np.isnan(draws)]
    return float(min(1.0, 2 * min((draws <= 0).mean(), (draws >= 0).mean())))


def _exact_avg(s: np.ndarray, d: np.ndarray, rng) -> float:
    g = ss.crossfit(s, d, rng)
    return float(np.mean([dv72.exact_gain(g, d, b, 0.0) for b in AVG]))


def _exact_point(s: np.ndarray, d: np.ndarray) -> float:
    rng = np.random.default_rng(SEED)
    return float(np.mean([_exact_avg(s, d, rng) for _ in range(ss.REPEATS)]))


def _predictions() -> pd.DataFrame:
    """Runner test predictions (all folds concatenated) plus probe scores."""
    pp = _load("pp60", "scripts/60_partial_probe.py")
    rn = pp._load("rn54", "scripts/54_baseline_runner.py")
    bm25 = _runs("bm25")
    probes = json.loads(Path(f"results/probes/{CORPUS}_all.json").read_text())
    parts, hp = [], {}
    for k, R in enumerate(roles()):
        tag = "" if CORPUS == "hotpotqa" else f"_fold{k}"
        out_dir = RUNS / f"runner{tag}"
        subprocess.run([sys.executable, "scripts/54_baseline_runner.py", "--input", str(RUNS / f"{CORPUS}_frozen{tag}.csv"),
                        "--output", str(out_dir), "--actions", ",".join(ACTIONS), "--bootstrap", str(NBOOT),
                        "--cost", "0"], check=True)
        pred = pd.read_csv(out_dir / "test_predictions.csv")
        pred["query_id"] = pred["query_id"].astype(str)
        fr = pd.read_csv(RUNS / f"{CORPUS}_frozen{tag}.csv")
        fr["query_id"] = fr["query_id"].astype(str)
        fr = fr.set_index("query_id")
        X = {}
        for sp in ("train", "val", "test"):
            ids = [str(q) for q in R[sp]]
            X[sp] = {kk: np.array([pp.probe_features(bm25[q]["ranking"], probes[q], kk, bm25[q]["nqc"], bm25[q]["qlen"])
                                   for q in ids]) for kk in pp.SIZES}
            X[sp]["delta"] = (fr.loc[ids, "rerank_ndcg"] - fr.loc[ids, "baseline_ndcg"]).to_numpy()
        test_ids = [str(q) for q in R["test"]]
        for kk in pp.SIZES:
            sc, hp[f"fold{k}_probe_{kk}"] = rn.cat_gbdt(X["train"][kk], X["train"]["delta"], X["val"][kk],
                                                        X["val"]["delta"], X["test"][kk])
            for a in ACTIONS:
                m = pred.action == a
                pred.loc[m, f"probe_{kk}"] = pd.Series(sc, index=test_ids).loc[pred.loc[m, "query_id"]].to_numpy()
        pred["fold"] = k
        parts.append(pred)
    out = pd.concat(parts, ignore_index=True)
    out.attrs["probe_hyperparameters"] = {kk: str(v) for kk, v in hp.items()}
    return out


def test() -> None:
    a65 = _load("a65", "scripts/65_core_claim_audit.py")
    pred = _predictions()
    bm25 = _runs("bm25")
    qrels = load_qrels(QRELS_NAME["test"])
    test_ids = [str(q) for q in SPLIT["test"]]
    lat = json.loads((RUNS / "latency.json").read_text()) if (RUNS / "latency.json").exists() else None
    res = {"_note": "configs/info_value_precommit.json; untouched benchmark, single test pass; aggregates only.",
           "n": {sp: len(SPLIT[sp]) for sp in ("train", "val", "test") if sp in SPLIT},
           "prf_selected": json.loads((RUNS / "prf_selected.json").read_text()),
           "val_prediction": json.loads(Path(f"docs/{CORPUS}_val_prediction.json").read_text()),
           "probe_hyperparameters": pred.attrs["probe_hyperparameters"], "latency_s": lat}
    ps, per = {}, {}
    for a in ACTIONS:
        g = pred[pred.action == a].set_index("query_id").loc[test_ids]
        d = g["delta"].to_numpy()
        F = g["original_ndcg"].to_numpy() <= 1e-12
        S = g["failure"].to_numpy()
        sc = {"failure": S, "nqc": g["nqc"].to_numpy(), "B1_gbdt": g["B1_gbdt"].to_numpy(),
              "B2_semantic_knn": g["B2_semantic_knn"].to_numpy(), "probe_10": g["probe_10"].to_numpy()}
        rng = np.random.default_rng(SEED)
        idx = [rng.integers(0, len(d), len(d)) for _ in range(NBOOT)]
        r = {}
        # P1
        p1 = a65.auc(F, S) - a65.auc(d > EPS, S)
        b1 = np.array([a65.auc(F[i], S[i]) - a65.auc(d[i] > EPS, S[i]) for i in idx])
        r["P1"] = {"estimate": p1, "ci": [float(np.nanquantile(b1, .025)), float(np.nanquantile(b1, .975))],
                   "p": _p(b1), "auc_failure": a65.auc(F, S), "auc_helped": a65.auc(d > EPS, S)}
        # P2-P4: exact-budget calibrated routing, budget-averaged 10-75%, kappa 0
        rnd_avg = lambda dd: float(np.mean([b * dd.mean() for b in AVG]))  # noqa: E731
        brng = np.random.default_rng(SEED)
        pt = {m: _exact_point(s, d) for m, s in sc.items()}
        boot = {"P2": [], "P3": [], "P4": []}
        for i in idx:
            e = {m: _exact_avg(sc[m][i], d[i], brng) for m in ("failure", "B1_gbdt", "probe_10")}
            boot["P2"].append(e["failure"] - rnd_avg(d[i]))
            boot["P3"].append(e["B1_gbdt"] - e["failure"])
            boot["P4"].append(e["probe_10"] - e["failure"])
        est = {"P2": pt["failure"] - rnd_avg(d), "P3": pt["B1_gbdt"] - pt["failure"], "P4": pt["probe_10"] - pt["failure"]}
        for k in ("P2", "P3", "P4"):
            v = np.array(boot[k])
            r[k] = {"estimate": est[k], "ci": [float(np.quantile(v, .025)), float(np.quantile(v, .975))], "p": _p(v)}
        r["exact_budget_gain_avg"] = pt
        # P5 and mechanism
        rk = [[dd for dd, _ in bm25[q]["ranking"]][:100] for q in test_ids]
        rel = [{dd for dd, x in qrels.get(q, {}).items() if x > 0} for q in test_ids]
        r["mechanism"] = mech.analyse(S, d, F, rk, rel)
        first = np.array([next((j + 1 for j, x in enumerate(rr) if x in R), 0) for rr, R in zip(rk, rel)])
        reach = F & (first >= 11)
        n_r, n_u = int(reach.sum()), int((F & ~reach).sum())
        r["P5"] = {"auc": a65.auc(reach[F], S[F]) if n_r and n_u else None, "n_failing": int(F.sum()),
                   "n_reachable": n_r, "n_unreachable": n_u, "evaluable": F.sum() >= 30 and n_r >= 20 and n_u >= 20}
        if r["P5"]["evaluable"]:
            b5 = np.array([a65.auc(reach[i][F[i]], S[i][F[i]]) - 0.5 for i in idx])
            r["P5"].update({"ci": [float(np.nanquantile(b5, .025)) + .5, float(np.nanquantile(b5, .975)) + .5],
                            "p": _p(b5)})
        if a == "rerank":
            ps.update({"P1": r["P1"]["p"], "P2": r["P2"]["p"], "P3": r["P3"]["p"], "P4": r["P4"]["p"]})
            if r["P5"]["evaluable"]:
                ps["P5"] = r["P5"]["p"]
        # descriptive
        r["mean_delta"] = float(d.mean())
        r["rates"] = {"p_fail": float(F.mean()), "rescue": float((d[F] > EPS).mean()) if F.any() else None,
                      "healthy_helped": float((d[~F] > EPS).mean()), "healthy_harmed": float((d[~F] < -EPS).mean())}
        r["same_target"] = a65.same_target(d, F, sc) if F.any() and (~F).any() else None
        r["budgeted_oracle"] = a65.budgeted(d, F, KAPPAS, S)
        r["decision_value"] = dv72.decision_value(d, sc)
        bins = np.searchsorted(np.quantile(S, np.linspace(0, 1, 11)[1:-1]), S, side="right")
        r["profile"] = [{"n": int((bins == j).sum()), "p_fail": float(F[bins == j].mean()),
                         "p_helped": float((d[bins == j] > EPS).mean()), "mean_delta": float(d[bins == j].mean())}
                        for j in range(10)]
        per[a] = r
    res["primary_holm"] = da.holm(ps)
    res["per_action"] = per
    res["runner_primary_by_fold"] = {
        f: pd.read_csv(RUNS / f"runner{'' if CORPUS == 'hotpotqa' else f'_fold{f}'}" / "primary_comparisons.csv").to_dict("records")
        for f in range(len(roles()))}

    def rnd(x):
        if isinstance(x, (float, np.floating)):
            return None if np.isnan(x) else round(float(x), 4)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (bool, np.bool_)):
            return bool(x)
        if isinstance(x, dict):
            return {str(k): rnd(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [rnd(v) for v in x]
        return x
    Path(f"docs/{CORPUS}_summary.json").write_text(json.dumps(rnd(res), indent=2) + "\n")
    for k, v in res["primary_holm"].items():
        print(k, v)
    r = per["rerank"]
    print("P1", rnd(r["P1"]), "\nP2", rnd(r["P2"]), "\nP3", rnd(r["P3"]), "\nP4", rnd(r["P4"]), "\nP5", rnd(r["P5"]))


if __name__ == "__main__":
    {"time": time_val, "select-prf": select_prf, "val-prediction": val_prediction,
     "export": export, "test": test}[sys.argv[2]]()
