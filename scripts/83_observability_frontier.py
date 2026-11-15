"""EXPLORATORY analyses frozen in configs/observability_frontier_protocol.json
(committed before computing): (A) a score-only spline policy calibrated on
training data, (B) the value of pre-action, probe and post-action
information for reranking at matched budgets, (C) sensitivity of the core
quantities to the failure threshold.

Writes rounded aggregates only to docs/observability_frontier.json.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

sys.path.insert(0, ".")
from src.data import load_qrels  # noqa: E402


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


a81 = _load("a81", "scripts/81_actionability.py")
pp = _load("pp60", "scripts/60_partial_probe.py")
iv = _load("iv49", "scripts/49_intervention_analysis.py")
auc, route, pval, LGB = a81.auc, a81.route, a81.pval, a81.LGB
EPS, NBOOT, SEED, AVG = 1e-9, 10000, 42, a81.AVG
TAUS = (0.0, 0.1, 0.2, 0.3)
PROBE = [f"probe_{i}" for i in range(9)]
POST = [f"post_{i}" for i in range(6)]


def hfold(q: str) -> int:
    return int(hashlib.sha256(f"{q}:oof".encode()).hexdigest(), 16) % 5


def gate():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(max_iter=2000, random_state=SEED))


# ---------------------------------------------------------------- data

def extend(df: pd.DataFrame, bm: dict, rr: dict, probes: dict, b: np.ndarray) -> pd.DataFrame:
    df = df.copy()
    df["b"] = b
    df[PROBE] = np.array([pp.probe_features(bm[q], probes[q], 10, df.loc[q, "nqc"], df.loc[q, "qlen"]) for q in df.index])
    df[POST] = np.array([iv.post_features(bm[q], rr[q]) for q in df.index])
    return df


def nq_tripclick(corpus: str):
    da = _load("da50", "scripts/50_decision_analyses.py")
    inp = da.iv.corpus_inputs(corpus)
    (tr, te), = a81.nq_tripclick(corpus)
    sets = {"val": tr, "test": te}
    out = {}
    for sp, df in sets.items():
        qids, qrels = list(df.index), inp[sp]["qrels"]
        bm, rr = da.iv._runs(corpus, sp, "bm25"), da.iv._runs(corpus, sp, "rerank")
        b = da.iv.ndcg(qids, {q: [x for x, _ in bm[q]] for q in qids}, qrels)
        probes = json.loads(Path(f"results/probes/{corpus}_{sp}.json").read_text())
        out[sp] = extend(df, bm, rr, probes, b)
    prf = {s["name"]: s for s in da.intervention_settings()}[f"prf_{corpus}"]["delta"]
    out["test"]["delta_prf"] = prf
    return [(out["val"], out["test"])], 8


def beir(corpus: str):
    folds = a81.beir(corpus)
    bm = {q: v["ranking"] for q, v in json.loads(Path(f"results/{corpus}/bm25.json").read_text()).items()}
    rr = {q: v["ranking"] for q, v in json.loads(Path(f"results/{corpus}/rerank.json").read_text()).items()}
    probes = json.loads(Path(f"results/probes/{corpus}_all.json").read_text())
    csvs = ([f"results/dbpedia/dbpedia_frozen_fold{k}.csv" for k in range(5)] if corpus == "dbpedia"
            else [f"results/{corpus}/{corpus}_frozen.csv"])
    out = []
    for (tr, te), c in zip(folds, csvs):
        fr = pd.read_csv(c, usecols=["query_id", "baseline_ndcg", "prf_ndcg"]).assign(
            query_id=lambda x: x.query_id.astype(str)).set_index("query_id")
        tr = extend(tr, bm, rr, probes, fr.loc[tr.index, "baseline_ndcg"].to_numpy())
        te = extend(te, bm, rr, probes, fr.loc[te.index, "baseline_ndcg"].to_numpy())
        te["delta_prf"] = (fr.loc[te.index, "prf_ndcg"] - fr.loc[te.index, "baseline_ndcg"]).to_numpy()
        out.append((tr, te))
    return out, 10


# ---------------------------------------------------------------- A

def oof_scores(tr: pd.DataFrame, feats: list[str]) -> np.ndarray:
    fold = np.array([hfold(q) for q in tr.index])
    X, y, s = tr[feats].to_numpy(), tr["F"].to_numpy().astype(int), np.empty(len(tr))
    for k in range(5):
        m = fold == k
        if len(np.unique(y[~m])) < 2:
            s[m] = y[~m].mean()
            continue
        s[m] = gate().fit(X[~m], y[~m]).predict_proba(X[m])[:, 1]
    return s


def spline_fit(s: np.ndarray, d: np.ndarray):
    model = make_pipeline(SplineTransformer(n_knots=6, degree=3, knots="quantile", extrapolation="linear"),
                          Ridge(alpha=1.0)).fit(s.reshape(-1, 1), d)
    return lambda z: model.predict(np.asarray(z).reshape(-1, 1))


def route_abstain(score: np.ndarray, d: np.ndarray) -> float:
    n, order = len(d), np.argsort(-score, kind="stable")
    vals = []
    for b in AVG:
        top = order[: int(np.floor(b * n))]
        top = top[score[top] > 0]
        vals.append(d[top].sum() / n)
    return float(np.mean(vals))


def boot(fns: dict, d: np.ndarray, pairs) -> tuple[dict, dict]:
    rng = np.random.default_rng(SEED)
    n = len(d)
    draws = {p: np.empty(NBOOT) for p in pairs}
    for j in range(NBOOT):
        i = rng.integers(0, n, n)
        v = {k: f(i) for k, f in fns.items()}
        for a, b in pairs:
            draws[(a, b)][j] = v[a] - v[b]
    full = np.arange(n)
    pt = {k: f(full) for k, f in fns.items()}
    return {f"{a} - {b}": {"estimate": pt[a] - pt[b], "ci": [float(np.quantile(draws[(a, b)], q)) for q in (.025, .975)],
                           "p": pval(draws[(a, b)])} for a, b in pairs}, pt


# ---------------------------------------------------------------- main analysis

def analyse(corpus: str, folds, nfeat: int, ref: dict) -> dict:
    gate_feats = a81.BASE[:nfeat]
    preds = {k: [] for k in ("spline", "Z0_cheap", "Z0_full", "Zp", "Z1")}
    refit = {t: [] for t in TAUS}
    tests = []
    for tr, te in folds:
        s_tr = oof_scores(tr, gate_feats)
        preds["spline"].append(spline_fit(s_tr, tr["delta"].to_numpy())(te["S"].to_numpy()))
        sets = {"Z0_cheap": a81.CHEAP_BASE + a81.DEEP, "Z0_full": a81.BASE + a81.DEEP,
                "Zp": a81.CHEAP_BASE + a81.DEEP + PROBE, "Z1": a81.CHEAP_BASE + a81.DEEP + PROBE + POST}
        for k, f in sets.items():
            preds[k].append(lgb.LGBMRegressor(**LGB).fit(tr[f].to_numpy(), tr["delta"].to_numpy()).predict(te[f].to_numpy()))
        for t in TAUS:
            y = (tr["b"].to_numpy() <= t + EPS).astype(int)
            refit[t].append(gate().fit(tr[gate_feats].to_numpy(), y).predict_proba(te[gate_feats].to_numpy())[:, 1]
                            if 0 < y.sum() < len(y) else np.full(len(te), y.mean()))
        tests.append(te)
    te = pd.concat(tests)
    P = {k: np.concatenate(v) for k, v in preds.items()}
    d, S, b = te["delta"].to_numpy(), te["S"].to_numpy(), te["b"].to_numpy()
    n = len(d)
    orc = float(np.mean([a81.ss.oracle_gain(d, x, 0.0) for x in AVG]))
    rnd = float(np.mean(AVG) * d.mean())
    res = {"n_test": n}

    # A
    fns = {"Spline_S_abstain": lambda i: route_abstain(P["spline"][i], d[i]),
           "Spline_S_exact": lambda i: route(P["spline"][i], d[i]),
           "Raw_S": lambda i: route(S[i], d[i]),
           "Mono_full": lambda i: route(P["Z0_full"][i], d[i]),
           "Random": lambda i: float(np.mean(AVG) * d[i].mean())}
    con, pt = boot(fns, d, [("Spline_S_abstain", "Raw_S"), ("Spline_S_exact", "Raw_S"),
                            ("Mono_full", "Spline_S_abstain"), ("Spline_S_abstain", "Random")])
    share = {k: (v - rnd) / (orc - rnd) if orc > rnd else None for k, v in pt.items()}
    res["A"] = {"value": pt, "oracle": orc, "TenBin_xfit_reference": ref["regret_S"]["V_S_lo"],
                "share_of_oracle_minus_random": share, "contrasts": con}

    # B
    keep = lambda p, i: float(d[i][p[i] > 0].sum() / len(i))  # noqa: E731
    fnsB = {k: (lambda i, k=k: route(P[k][i], d[i])) for k in ("Z0_cheap", "Z0_full", "Zp", "Z1")}
    fnsB["Z1_keep_unconstrained"] = lambda i: keep(P["Z1"], i)
    fnsB["Always"] = lambda i: float(d[i].mean())
    conB, ptB = boot(fnsB, d, [("Zp", "Z0_cheap"), ("Z1", "Zp"), ("Z0_full", "Z0_cheap"),
                               ("Z1_keep_unconstrained", "Always")])
    res["B"] = {"value_budget_avg": ptB, "oracle": orc, "random": rnd, "increments": conB}

    # C
    out = {}
    for name, dd in (("rerank", d), ("prf", te["delta_prf"].to_numpy())):
        rows = {}
        for k, t in enumerate(TAUS):
            Ft = b <= t + EPS
            rng = np.random.default_rng(SEED)
            idx = [rng.integers(0, n, n) for _ in range(NBOOT)]
            h = dd > EPS
            gap = auc(Ft, S) - auc(h, S)
            gb = np.array([auc(Ft[i], S[i]) - auc(h[i], S[i]) for i in idx])
            pos = np.maximum(dd, 0)
            sh = pos[Ft].sum() / pos.sum() if pos.sum() > 0 else np.nan
            sb = np.array([np.maximum(dd[i], 0)[Ft[i]].sum() / max(np.maximum(dd[i], 0).sum(), 1e-12) for i in idx])
            hw = h[Ft]
            rf = np.concatenate(refit[t])
            rows[f"tau={t}"] = {
                "P_F": float(Ft.mean()),
                "AUC_F_S": auc(Ft, S), "AUC_helped_S": auc(h, S),
                "gap": {"estimate": gap, "ci": [float(np.nanquantile(gb, q)) for q in (.025, .975)]},
                "within_F_helped_AUC": auc(hw, S[Ft]) if min(hw.sum(), (~hw).sum()) >= 20 else None,
                "ceiling_share": {"estimate": sh, "ci": [float(np.nanquantile(sb, q)) for q in (.025, .975)]},
                "P_harmed_given_F": float((dd[Ft] < -EPS).mean()) if Ft.any() else None,
                "refit_gate": {"AUC_F": auc(Ft, rf), "AUC_helped": auc(h, rf)}}
        out[name] = rows
    res["C"] = out
    return res


def costs() -> dict:
    pl = json.loads(Path("results/probes/latency.json").read_text())
    c = {k: {"rerank": pl[k]["full"]["median_s"], "probe_10": pl[k]["probe_10"]["median_s"]}
         for k in ("nq", "tripclick", "arguana") if k in pl}
    for k in ("fiqa", "dbpedia", "hotpotqa"):
        l = json.loads(Path(f"results/{k}/latency.json").read_text())
        c[k] = {"rerank": l["full"]["median_s"], "probe_10": l["probe_10"]["median_s"]}
    c["hotpotqa"]["telemetry_incremental"] = json.loads(
        Path("docs/hotpotqa_latency_split.json").read_text())["incremental_telemetry_paired"]["median_s"]
    return c


def main() -> None:
    global NBOOT
    smoke = "--smoke" in sys.argv
    if smoke:
        NBOOT = 50
    ref = json.loads(Path("docs/actionability.json").read_text())
    res = {"_note": "EXPLORATORY (configs/observability_frontier_protocol.json, committed before computing). "
                    "Aggregates only.", "cost_median_s": costs()}
    ps = {}
    for corpus in ("nq", "tripclick", "arguana", "fiqa", "dbpedia", "hotpotqa"):
        folds, nfeat = nq_tripclick(corpus) if corpus in ("nq", "tripclick") else beir(corpus)
        r = analyse(corpus, folds, nfeat, ref[corpus])
        res[corpus] = r
        ps[corpus] = r["A"]["contrasts"]["Spline_S_abstain - Raw_S"]["p"]
        A, B = r["A"], r["B"]
        print(f"{corpus:9s} A spline-raw {A['contrasts']['Spline_S_abstain - Raw_S']['estimate']:+.4f} "
              f"{np.round(A['contrasts']['Spline_S_abstain - Raw_S']['ci'], 4)} share spline "
              f"{A['share_of_oracle_minus_random']['Spline_S_abstain']} raw {A['share_of_oracle_minus_random']['Raw_S']} "
              f"| B Zp-Z0c {B['increments']['Zp - Z0_cheap']['estimate']:+.4f} Z1-Zp {B['increments']['Z1 - Zp']['estimate']:+.4f} "
              f"| C rerank gaps {[round(v['gap']['estimate'], 3) for v in r['C']['rerank'].values()]}", flush=True)
    res["A_primary_holm"] = _load("da50", "scripts/50_decision_analyses.py").holm(ps)

    def rnd(x):
        if isinstance(x, (float, np.floating)):
            return None if np.isnan(x) else round(float(x), 4)
        if isinstance(x, (bool, np.bool_)):
            return bool(x)
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, dict):
            return {k: rnd(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [rnd(v) for v in x]
        return x
    if smoke:
        print("smoke ok")
        return
    Path("docs/observability_frontier.json").write_text(json.dumps(rnd(res), indent=2) + "\n")
    print("done")


if __name__ == "__main__":
    main()
