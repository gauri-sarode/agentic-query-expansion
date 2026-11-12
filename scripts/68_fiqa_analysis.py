#!/usr/bin/env python3
"""FiQA replication, as pre-committed in configs/fiqa_precommit.json. This file
is committed before any FiQA test label is read; test mode runs once.

Modes:
  time            label-free latency on 50 val queries: probe_k, rest_k, full
                  reranking (scripts/60 procedure) -> results/fiqa/latency.json
  select-prf      PRF setting by val mean nDCG@10 (val qrels only)
  val-prediction  mean reranking Delta on val -> the P3/P4 prediction
                  (docs/fiqa_val_prediction.json; val qrels only)
  export          results/fiqa/fiqa_frozen.csv in the scripts/54 input schema
                  (reads test qrels: part of the single test pass)
  test            runner + probe + P1-P4 (one Holm family) + descriptive
                  -> docs/fiqa_summary.json (rounded aggregates only)

Usage: PYTHONPATH=. python3 scripts/68_fiqa_analysis.py MODE
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
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")
from src.data import load_qrels  # noqa: E402


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


da = _load("da50", "scripts/50_decision_analyses.py")
iv = da.iv
r67 = _load("r67", "scripts/67_fiqa_runs.py")
RUNS = Path("results/fiqa")
SPLIT = r67.SPLIT
QRELS_NAME = {"train": "fiqa-train", "val": "fiqa-dev", "test": "fiqa-test"}
ACTIONS = ("rerank", "prf")
EPS = 1e-9
NBOOT = 10000
SEED = 42
KAPPAS = (0.0, 0.01, 0.02)


def _runs(step: str) -> dict:
    return json.loads((RUNS / f"{step}.json").read_text())


def _ndcg(sp: str, rankings: dict, qrels: dict) -> np.ndarray:
    qids = SPLIT[sp]
    return iv.ndcg(qids, {q: [d for d, _ in rankings[q]] for q in qids}, qrels)


def _prf() -> tuple[int, int, float]:
    p = json.loads((RUNS / "prf_selected.json").read_text())
    return p["fb_docs"], p["fb_terms"], p["lam"]


# ---------------------------------------------------------------- label-free

def time_val() -> None:
    from src.retrieval import bm25_index
    from src.retrieval.rerank import _model

    pp = _load("pp60", "scripts/60_partial_probe.py")
    bm, texts, model = {q: v["ranking"] for q, v in _runs("bm25").items()}, r67.queries(), _model()
    sample = sorted(SPLIT["val"], key=pp._h)[:50]

    def ce(q, ids):
        tx = bm25_index.get_texts(ids, r67.DB)
        model.predict([(texts[q], tx[d]) for d in ids if d in tx])

    pipes = {f"probe_{k}": (lambda q, k=k: ce(q, [d for d, _ in bm[q][:k]])) for k in pp.SIZES}
    pipes |= {f"rest_{k}": (lambda q, k=k: ce(q, [d for d, _ in bm[q][k:pp.FULL]])) for k in pp.SIZES}
    pipes["full"] = lambda q: ce(q, [d for d, _ in bm[q][:pp.FULL]])
    for q in sample[:3]:
        for f in pipes.values():
            f(q)
    names, t = list(pipes), {n: [] for n in pipes}
    for i, q in enumerate(sample):
        for n in names[i % len(names):] + names[: i % len(names)]:
            s = time.perf_counter()
            pipes[n](q)
            t[n].append(time.perf_counter() - s)
    out = {n: {"median_s": float(np.median(v)), "p95_s": float(np.percentile(v, 95))} for n, v in t.items()}
    tel = [v["ms"] / 1000 for q, v in _runs("bm25").items() if q in set(SPLIT["val"])]
    out["telemetry"] = {"median_s": float(np.median(tel)), "p95_s": float(np.percentile(tel, 95)),
                        "source": "bm25 step wall clock incl. retrieval (val)"}
    (RUNS / "latency.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


# ---------------------------------------------------------------- val only

def select_prf() -> None:
    qrels = load_qrels(QRELS_NAME["val"])
    bm25 = {q: v["ranking"] for q, v in _runs("bm25").items()}
    b = _ndcg("val", bm25, qrels)
    res = {}
    for d, t in iv.PRF_RETRIEVAL:
        exp = _runs(f"prf_d{d}_t{t}")
        for lam in sorted({g[2] for g in iv.PRF_GRID}):
            fused = {q: iv.prf_fuse(bm25[q], exp[q]["ranking"], lam) for q in SPLIT["val"]}
            res[(d, t, lam)] = float(_ndcg("val", fused, qrels).mean())
    best = max(res, key=res.get)
    out = {"fb_docs": best[0], "fb_terms": best[1], "lam": best[2], "val_ndcg": res[best],
           "val_bm25": float(b.mean()), "n_settings": len(res)}
    (RUNS / "prf_selected.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def val_prediction() -> None:
    qrels = load_qrels(QRELS_NAME["val"])
    bm25 = {q: v["ranking"] for q, v in _runs("bm25").items()}
    rr = {q: v["ranking"] for q, v in _runs("rerank").items()}
    d = _ndcg("val", rr, qrels) - _ndcg("val", bm25, qrels)
    out = {"_note": "configs/fiqa_precommit.json P3_P4_prediction, recorded before test labels are read.",
           "val_mean_rerank_delta": round(float(d.mean()), 4), "n_val": len(d),
           "prediction": "P3 and P4 positive" if d.mean() > 0 else "no significant difference"}
    Path("docs/fiqa_val_prediction.json").write_text(json.dumps(out, indent=2) + "\n")
    print(out)


# ---------------------------------------------------------------- test pass

def export() -> None:
    bm25 = _runs("bm25")
    rk = {"rerank": {q: v["ranking"] for q, v in _runs("rerank").items()}}
    fd, ft, lam = _prf()
    exp = _runs(f"prf_d{fd}_t{ft}")
    rk["prf"] = {q: iv.prf_fuse(bm25[q]["ranking"], exp[q]["ranking"], lam) for q in r67.QIDS}
    emb = np.load(RUNS / "embed.npy")
    row = {q: i for i, q in enumerate(json.loads((RUNS / "embed_ids.json").read_text()))}
    names = ["top_score", "score_margin", "score_entropy", "query_term_coverage", "mean_rare_term_idf",
             "entity_overlap", "result_coherence", "ranking_stability", "nqc", "qlen"]
    path = RUNS / "fiqa_frozen.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "split", "baseline_ndcg", *[f"{a}_ndcg" for a in ACTIONS],
                    *[f"z_{n}" for n in names], *[f"emb_{j:03d}" for j in range(emb.shape[1])], "nqc_score"])
        for sp in ("train", "val", "test"):
            qrels = load_qrels(QRELS_NAME[sp])
            b = _ndcg(sp, {q: bm25[q]["ranking"] for q in SPLIT[sp]}, qrels)
            n = {a: _ndcg(sp, rk[a], qrels) for a in ACTIONS}
            for i, q in enumerate(SPLIT[sp]):
                r = bm25[q]
                w.writerow([q, sp, b[i], *[n[a][i] for a in ACTIONS], *r["t0"], r["nqc"], r["qlen"],
                            *emb[row[q]], r["nqc"]])
    print(path, hashlib.sha256(path.read_bytes()).hexdigest())


def _p_two_sided(draws: np.ndarray, null: float = 0.0) -> float:
    draws = draws[~np.isnan(draws)]
    return float(min(1.0, 2 * min((draws <= null).mean(), (draws >= null).mean())))


def test() -> None:
    a65 = _load("a65", "scripts/65_core_claim_audit.py")
    pp = _load("pp60", "scripts/60_partial_probe.py")
    rn = pp._load("rn54", "scripts/54_baseline_runner.py")
    out_dir = RUNS / "runner"
    subprocess.run([sys.executable, "scripts/54_baseline_runner.py", "--input", str(RUNS / "fiqa_frozen.csv"),
                    "--output", str(out_dir), "--actions", ",".join(ACTIONS), "--bootstrap", str(NBOOT),
                    "--cost", "0"], check=True)
    pred = pd.read_csv(out_dir / "test_predictions.csv")
    fr = pd.read_csv(RUNS / "fiqa_frozen.csv").set_index("query_id")
    fr.index = fr.index.astype(str)
    bm25 = _runs("bm25")
    probes = json.loads(Path("results/probes/fiqa_all.json").read_text())
    lat = json.loads((RUNS / "latency.json").read_text())
    res = {"_note": "configs/fiqa_precommit.json; untouched benchmark, single test pass; aggregates only.",
           "n": {sp: len(SPLIT[sp]) for sp in SPLIT if sp in ("train", "val", "test")},
           "prf_selected": json.loads((RUNS / "prf_selected.json").read_text()),
           "val_prediction": json.loads(Path("docs/fiqa_val_prediction.json").read_text()),
           "runner_primary": pd.read_csv(out_dir / "primary_comparisons.csv").to_dict("records")}
    rng = np.random.default_rng(SEED)
    ps, per = {}, {}
    for a in ACTIONS:
        g = pred[pred.action == a].copy()
        g["query_id"] = g["query_id"].astype(str)
        assert list(g["query_id"]) == [str(q) for q in SPLIT["test"]] or set(g["query_id"]) == set(map(str, SPLIT["test"]))
        g = g.set_index("query_id").loc[[str(q) for q in SPLIT["test"]]]
        d = g["delta"].to_numpy()
        F = g["original_ndcg"].to_numpy() <= 1e-12
        fail = g["failure"].to_numpy()
        sc = {"failure": fail, "nqc": g["nqc"].to_numpy(), "B1_gbdt": g["B1_gbdt"].to_numpy(),
              "B2_semantic_knn": g["B2_semantic_knn"].to_numpy(), "B3_quantile_LCB": g["B3_quantile_LCB"].to_numpy()}
        if a == "rerank":
            X = {}
            for sp in ("train", "val", "test"):
                ids = [str(q) for q in SPLIT[sp]]
                X[sp] = {k: np.array([pp.probe_features(bm25[q]["ranking"], probes[q], k, bm25[q]["nqc"], bm25[q]["qlen"])
                                      for q in ids]) for k in pp.SIZES}
                X[sp]["delta"] = (fr.loc[ids, "rerank_ndcg"] - fr.loc[ids, "baseline_ndcg"]).to_numpy()
            hp = {}
            for k in pp.SIZES:
                sc[f"probe_{k}"], hp[k] = rn.cat_gbdt(X["train"][k], X["train"]["delta"], X["val"][k],
                                                     X["val"]["delta"], X["test"][k])
            assert np.allclose(X["test"]["delta"], d)
            res["probe_hyperparameters"] = {str(k): str(v) for k, v in hp.items()}
        idx = [rng.integers(0, len(d), len(d)) for _ in range(NBOOT)]
        # P1: AUC(F) - AUC(helped), same score, same queries
        p1 = a65.auc(F, fail) - a65.auc(d > EPS, fail)
        b1 = np.array([a65.auc(F[i], fail[i]) - a65.auc(d[i] > EPS, fail[i]) for i in idx])
        # P2: AUC(helped | F) - 0.5
        resc = int(((d > EPS) & F).sum())
        p2 = a65.auc(d[F] > EPS, fail[F]) - 0.5 if resc else float("nan")
        b2 = np.array([a65.auc(d[i][F[i]] > EPS, fail[i][F[i]]) - 0.5 for i in idx]) if resc else np.array([np.nan])
        per[a] = {"P1": {"estimate": p1, "ci": [float(np.nanquantile(b1, .025)), float(np.nanquantile(b1, .975))],
                         "p": _p_two_sided(b1), "auc_failure": a65.auc(F, fail), "auc_helped": a65.auc(d > EPS, fail)},
                  "P2": {"auc_within_failing": p2 + 0.5, "n_failing": int(F.sum()), "n_rescued": resc,
                         "evaluable": resc >= 20,
                         "ci": [float(np.nanquantile(b2, .025)) + .5, float(np.nanquantile(b2, .975)) + .5]
                         if resc else None,
                         "p": _p_two_sided(b2) if resc else None}}
        ps[f"P1_{a}"] = per[a]["P1"]["p"]
        if resc >= 20:
            ps[f"P2_{a}"] = per[a]["P2"]["p"]
        if a == "rerank":
            tb = list(pp.TEST_BUDGETS)
            per[a]["P3"] = rn.bootstrap_pair(d, sc["probe_10"], sc["failure"], budgets=tb, nboot=NBOOT)
            per[a]["P4"] = rn.bootstrap_pair(d, sc["probe_10"], sc["B1_gbdt"], budgets=tb, nboot=NBOOT)
            ps["P3"], ps["P4"] = per[a]["P3"]["signflip_p"], per[a]["P4"]["signflip_p"]
            for k in (5, 20):
                per[a][f"probe_{k}_minus_failure"] = rn.bootstrap_pair(d, sc[f"probe_{k}"], sc["failure"], budgets=tb, nboot=NBOOT)
                per[a][f"probe_{k}_minus_B1"] = rn.bootstrap_pair(d, sc[f"probe_{k}"], sc["B1_gbdt"], budgets=tb, nboot=NBOOT)
            per[a]["probe_minus_nqc_matched"] = rn.bootstrap_pair(d, sc["probe_10"], sc["nqc"], budgets=tb, nboot=NBOOT)
            for stat in ("median_s", "p95_s"):
                L = {n: v[stat] for n, v in lat.items() if isinstance(v, dict) and stat in v}
                eq = {b: min(1.0, (L["probe_10"] + b * L["rest_10"]) / L["full"]) for b in tb}

                def eq_diff(ix):
                    return np.mean([rn.metrics(d[ix], sc["probe_10"][ix], b)["gain"]
                                    - rn.metrics(d[ix], sc["nqc"][ix], eq[b])["gain"] for b in tb])
                dr = np.array([eq_diff(i) for i in idx[:2000]])
                per[a][f"probe_minus_nqc_equal_cost_{stat}"] = {
                    "estimate": float(eq_diff(np.arange(len(d)))), "ci": [float(np.quantile(dr, .025)), float(np.quantile(dr, .975))],
                    "bootstrap_p": _p_two_sided(dr), "nqc_fraction": eq}
            per[a]["latency_s"] = lat
            per[a]["routing_curves"] = {m: [rn.metrics(d, s, b)["gain"] for b in (0.1, 0.25, 0.5, 0.75, 1.0)]
                                        for m, s in {**sc, "oracle": d}.items()}
        # descriptive
        per[a]["mean_delta"] = float(d.mean())
        per[a]["rates"] = {"p_fail": float(F.mean()), "rescue": float((d[F] > EPS).mean()),
                           "healthy_helped": float((d[~F] > EPS).mean()), "healthy_harmed": float((d[~F] < -EPS).mean())}
        per[a]["same_target"] = a65.same_target(d, F, sc)
        per[a]["budgeted_oracle"] = a65.budgeted(d, F, KAPPAS, fail)
        step = "rerank" if a == "rerank" else "prf_d{}_t{}".format(*_prf()[:2])
        test_ids = set(map(str, SPLIT["test"]))
        ms = np.array([v["ms"] for q, v in _runs(step).items() if q in test_ids])
        per[a]["intervention_latency_ms"] = {"median": float(np.median(ms)), "p95": float(np.percentile(ms, 95))}
    res["primary_holm"] = da.holm(ps)
    res["per_action"] = per
    res["table2_secondary"] = table2(fr)

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
    Path("docs/fiqa_summary.json").write_text(json.dumps(rnd(res), indent=2) + "\n")
    for k, v in res["primary_holm"].items():
        print(k, v)
    for a in ACTIONS:
        print(a, "P1", rnd(per[a]["P1"]), "P2", rnd(per[a]["P2"]))
    print("P3", rnd(per["rerank"]["P3"]), "P4", rnd(per["rerank"]["P4"]))


def table2(fr: pd.DataFrame) -> dict:
    """The paper's Table 2 quantity (secondary): failure-detector AUC for F
    (logistic on the 8 t0 features, train+val) minus the harm AUC of the
    action-utility model g (linear on pre + post-action features, train+val)."""
    a65 = _load("a65b", "scripts/65_core_claim_audit.py")
    bm25 = _runs("bm25")
    fd, ft, lam = _prf()
    exp = _runs(f"prf_d{fd}_t{ft}")
    rk = {"rerank": {q: v["ranking"] for q, v in _runs("rerank").items()},
          "prf": {q: iv.prf_fuse(bm25[q]["ranking"], exp[q]["ranking"], lam) for q in r67.QIDS}}
    tv = [str(q) for q in SPLIT["train"] + SPLIT["val"]]
    te = [str(q) for q in SPLIT["test"]]
    t0 = lambda ids: np.array([bm25[q]["t0"] for q in ids])  # noqa: E731
    pre = lambda ids: np.array([bm25[q]["t0"] + [bm25[q]["nqc"], bm25[q]["qlen"]] for q in ids])  # noqa: E731
    F = lambda ids: (fr.loc[ids, "baseline_ndcg"] <= 1e-12).to_numpy()  # noqa: E731
    det = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)).fit(t0(tv), F(tv))
    p_det = det.predict_proba(t0(te))[:, 1]
    out = {}
    for a in ACTIONS:
        post = lambda ids: np.array([iv.post_features(bm25[q]["ranking"], rk[a][q]) for q in ids])  # noqa: E731
        dl = lambda ids: (fr.loc[ids, f"{a}_ndcg"] - fr.loc[ids, "baseline_ndcg"]).to_numpy()  # noqa: E731
        g = make_pipeline(StandardScaler(), LinearRegression()).fit(np.hstack([pre(tv), post(tv)]), dl(tv))
        gs = g.predict(np.hstack([pre(te), post(te)]))
        d, f = dl(te), F(te)
        rng = np.random.default_rng(SEED)
        harm = d < -EPS
        diff = a65.auc(f, p_det) - a65.auc(harm, -gs)
        bs = np.array([a65.auc(f[i], p_det[i]) - a65.auc(harm[i], -gs[i])
                       for i in (rng.integers(0, len(d), len(d)) for _ in range(NBOOT))])
        out[a] = {"detection_auc": a65.auc(f, p_det), "harm_auc": a65.auc(harm, -gs), "diff": diff,
                  "ci": [float(np.nanquantile(bs, .025)), float(np.nanquantile(bs, .975))], "p": _p_two_sided(bs)}
    return out


if __name__ == "__main__":
    {"time": time_val, "select-prf": select_prf, "val-prediction": val_prediction,
     "export": export, "test": test}[sys.argv[1]]()
