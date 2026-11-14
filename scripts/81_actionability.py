"""EXPLORATORY actionability analysis, as frozen in
configs/actionability_protocol.json (committed before computing):
(A) information vs policy regret of failure-score routing, (B)
reachability-conditioned benefit routing (RCBR) against a matched
monolithic benefit model, (C) attribution of how mean reranking gain
varies with the failure score. Fixed-pool reranking only.

Writes rounded aggregates only to docs/actionability.json.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, ".")
from src.data import load_qrels, load_queries  # noqa: E402


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ss = _load("ss69", "scripts/69_score_sufficiency.py")
auc = _load("cc65", "scripts/65_core_claim_audit.py").auc
EPS, NBOOT, SEED = 1e-9, 10000, 42
AVG = (0.10, 0.25, 0.50, 0.75)
BASE = ["top_score", "score_margin", "score_entropy", "query_term_coverage", "mean_rare_term_idf",
        "entity_overlap", "result_coherence", "ranking_stability", "nqc", "qlen"]
CHEAP_BASE = ["top_score", "score_margin", "score_entropy", "nqc", "qlen"]
DEEP = ["s11", "s20", "s50", "s100", "mean_top10", "mean_deep", "std_deep", "gap_10_11", "slope_deep", "n_ret"]
LGB = dict(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=20, subsample=0.8,
           subsample_freq=1, colsample_bytree=0.8, random_state=42, verbose=-1)
RERANK_S = {"nq": 0.48, "tripclick": 1.11, "arguana": 1.26, "fiqa": 0.84}  # stored medians (paper, Section 9)


def deep_features(ranking: list) -> list[float]:
    s = np.array([x for _, x in ranking[:100]], dtype=float)
    n = len(s)
    s = np.pad(s, (0, 100 - n))
    top = s[0] if s[0] > 0 else 1.0
    deep = s[10:100][: max(n - 10, 0)] if n > 10 else np.zeros(1)
    return [s[10] / top, s[19] / top, s[49] / top, s[99] / top, s[:10].mean() / top, deep.mean() / top,
            deep.std() / top, (s[9] - s[10]) / top, (s[10] - s[99]) / top, float(n)]


def first_rel(ranking: list, rel: set) -> int:
    return next((i + 1 for i, (d, _) in enumerate(ranking[:100]) if d in rel), 0)


def frame(qids, base: np.ndarray, rankings: dict, b: np.ndarray, r: np.ndarray, qrels: dict) -> pd.DataFrame:
    df = pd.DataFrame(base, columns=BASE, index=qids)
    df[DEEP] = np.array([deep_features(rankings[q]) for q in qids])
    df["delta"] = r - b
    df["F"] = b <= EPS
    first = np.array([first_rel(rankings[q], {d for d, g in qrels.get(q, {}).items() if g > 0}) for q in qids])
    assert np.array_equal(~((first >= 1) & (first <= 10)), df["F"].to_numpy()), "F != no relevant in top 10"
    df["R"] = df["F"] & (first >= 11)
    return df


# ---------------------------------------------------------------- data

def nq_tripclick(corpus: str) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    da = _load("da50", "scripts/50_decision_analyses.py")
    iv, iv48 = da.iv, _load("iv48", "scripts/48_intervention_runs.py")
    inp = iv.corpus_inputs(corpus)
    det = {s["name"]: s for s in da.intervention_settings()}[f"rerank_{corpus}"]
    out = {}
    for sp in ("val", "test"):
        qids, qrels = inp[sp]["qids"], inp[sp]["qrels"]
        texts = iv48.setup(corpus, sp)[1]
        bm, rr = iv._runs(corpus, sp, "bm25"), iv._runs(corpus, sp, "rerank")
        b = iv.ndcg(qids, {q: [x for x, _ in bm[q]] for q in qids}, qrels)
        r = iv.ndcg(qids, {q: [x for x, _ in rr[q]] for q in qids}, qrels)
        base = np.hstack([inp[sp]["pre"], np.array([[da.nqc(bm[q]), len(texts[q].split())] for q in qids])])
        out[sp] = frame(qids, base, bm, b, r, qrels)
    assert np.allclose(out["test"]["delta"].to_numpy(), det["delta"]), "test Delta mismatch"
    out["test"]["S"] = det["det"]
    return [(out["val"], out["test"])]


def beir(corpus: str) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    names = {"arguana": ["arguana"], "fiqa": ["fiqa-train", "fiqa-dev", "fiqa-test"],
             "hotpotqa": ["hotpotqa-train", "hotpotqa-dev", "hotpotqa-test"], "dbpedia": ["dbpedia-test"]}[corpus]
    qrels = {}
    for n in names:
        qrels.update(load_qrels(n))
    bm = {q: v["ranking"] for q, v in json.loads(Path(f"results/{corpus}/bm25.json").read_text()).items()}
    if corpus == "dbpedia":
        csvs = [f"results/dbpedia/dbpedia_frozen_fold{k}.csv" for k in range(5)]
        preds = [f"results/dbpedia/runner_fold{k}/test_predictions.csv" for k in range(5)]
    else:
        csvs, preds = [f"results/{corpus}/{corpus}_frozen.csv"], [f"results/{corpus}/runner/test_predictions.csv"]
    out = []
    for c, pth in zip(csvs, preds):
        fr = pd.read_csv(c, usecols=["query_id", "split", "baseline_ndcg", "rerank_ndcg"] + [f"z_{f}" for f in BASE])
        fr["query_id"] = fr["query_id"].astype(str)
        parts = {}
        for role, mask in (("train", fr.split != "test"), ("test", fr.split == "test")):
            g = fr[mask]
            parts[role] = frame(list(g.query_id), g[[f"z_{f}" for f in BASE]].to_numpy(), bm,
                                g.baseline_ndcg.to_numpy(), g.rerank_ndcg.to_numpy(), qrels)
        p = pd.read_csv(pth)
        p = p[p.action == "rerank"].assign(query_id=lambda x: x.query_id.astype(str)).set_index("query_id")
        parts["test"]["S"] = p.loc[parts["test"].index, "failure"].to_numpy()
        assert np.allclose(parts["test"]["delta"].to_numpy(), p.loc[parts["test"].index, "delta"].to_numpy())
        out.append((parts["train"], parts["test"]))
    return out


# ---------------------------------------------------------------- models

def _clf(X, y, note, name):
    if len(y) < 30 or min(y.sum(), (~y).sum()) < 10:
        note.append(f"{name}: training mean ({len(y)} rows, {int(y.sum())} positive)")
        m = float(y.mean()) if len(y) else 0.0
        return lambda Z: np.full(len(Z), m)
    model = lgb.LGBMClassifier(**LGB).fit(X, y.astype(int))
    return lambda Z: model.predict_proba(Z)[:, 1]


def _reg(X, y, note, name):
    if len(y) < 30:
        note.append(f"{name}: training mean ({len(y)} rows)")
        m = float(y.mean()) if len(y) else 0.0
        return lambda Z: np.full(len(Z), m)
    model = lgb.LGBMRegressor(**LGB).fit(X, y)
    return lambda Z: model.predict(Z)


def fit_predict(tr: pd.DataFrame, te: pd.DataFrame, feats: list[str], note: list) -> dict:
    X, Z = tr[feats].to_numpy(), te[feats].to_numpy()
    F, R, d = tr["F"].to_numpy(), tr["R"].to_numpy(), tr["delta"].to_numpy()
    pF = _clf(X, F, note, "failure")(Z)
    pR = _clf(X[F], R[F], note, "reachability")(Z)
    muR = _reg(X[R], d[R], note, "rescue_gain")(Z)
    muH = _reg(X[~F], d[~F], note, "healthy_gain")(Z)
    rbar, mbar = float(R[F].mean()) if F.any() else 0.0, float(d[R].mean()) if R.any() else 0.0
    return {"rcbr": pF * pR * muR + (1 - pF) * muH,
            "noR": pF * rbar * muR + (1 - pF) * muH,
            "noMu": pF * pR * mbar + (1 - pF) * muH,
            "mono": lgb.LGBMRegressor(**LGB).fit(X, d).predict(Z),
            "pR": pR}


# ---------------------------------------------------------------- evaluation

def route(score: np.ndarray, d: np.ndarray) -> float:
    n, order = len(d), np.argsort(-score, kind="stable")
    cs = np.concatenate([[0.0], np.cumsum(d[order])])
    return float(np.mean([cs[int(np.floor(b * n))] for b in AVG]) / n)


def route_at(score, d, b):
    n, order = len(d), np.argsort(-score, kind="stable")
    return float(d[order[: int(np.floor(b * n))]].sum() / n)


def pval(boot: np.ndarray) -> float:
    return float(max(min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean())), 1 / len(boot)))


def boot_contrasts(scores: dict, d: np.ndarray, pairs: list[tuple[str, str]]) -> dict:
    rng = np.random.default_rng(SEED)
    n = len(d)
    draws = {p: np.empty(NBOOT) for p in pairs}
    for j in range(NBOOT):
        i = rng.integers(0, n, n)
        di = d[i]
        v = {k: (np.mean(AVG) * di.mean() if k == "random" else route(s[i], di)) for k, s in scores.items()}
        for a, b in pairs:
            draws[(a, b)][j] = v[a] - v[b]
    pt = {k: (np.mean(AVG) * d.mean() if k == "random" else route(s, d)) for k, s in scores.items()}
    return {f"{a} - {b}": {"estimate": pt[a] - pt[b], "ci": [float(np.quantile(draws[(a, b)], q)) for q in (.025, .975)],
                           "p": pval(draws[(a, b)])} for a, b in pairs}, pt


def regret(s: np.ndarray, d: np.ndarray) -> dict:
    rng = np.random.default_rng(SEED)
    orc = np.mean([ss.oracle_gain(d, b, 0.0) for b in AVG])
    lo = np.mean([np.mean([ss.select_gain(ss.crossfit(s, d, rng), d, b, 0.0) for b in AVG]) for _ in range(ss.REPEATS)])
    hi = np.mean([ss.select_gain(ss.bin_means(s, d, s, ss.K), d, b, 0.0) for b in AVG])
    raw, rnd = route(s, d), np.mean(AVG) * d.mean()
    brng = np.random.default_rng(SEED)
    a_draw, b_draw = np.empty(NBOOT), np.empty(NBOOT)
    for j in range(NBOOT):
        i = brng.integers(0, len(d), len(d))
        si, di = s[i], d[i]
        o, r = np.mean([ss.oracle_gain(di, b, 0.0) for b in AVG]), route(si, di)
        g = ss.crossfit(si, di, brng)
        a_draw[j] = o - r
        b_draw[j] = np.mean([ss.select_gain(g, di, b, 0.0) for b in AVG]) - r
    return {"V_oracle": orc, "V_S_lo": lo, "V_S_hi": hi, "V_raw": raw, "V_random": rnd,
            "information_regret": [orc - hi, orc - lo], "policy_regret_raw": [lo - raw, hi - raw],
            "information_regret_share_of_oracle_minus_random": [(orc - hi) / (orc - rnd), (orc - lo) / (orc - rnd)]
            if orc > rnd else None,
            "oracle_minus_raw": {"estimate": orc - raw, "ci": [float(np.quantile(a_draw, q)) for q in (.025, .975)]},
            "S_lo_minus_raw": {"estimate": lo - raw, "ci": [float(np.quantile(b_draw, q)) for q in (.025, .975)]}}


def components(s, d, F, R, cuts=None) -> list[dict]:
    if cuts is None:
        cuts = np.quantile(s, [1 / 3, 2 / 3])
    t = np.searchsorted(cuts, s, side="right")
    out = []
    for j in range(len(cuts) + 1):
        m = t == j
        f, r, h = F[m], R[m], ~F[m]
        pF = f.mean()
        pR = r[f].mean() if f.any() else 0.0
        muR = d[m][r].mean() if r.any() else 0.0
        muH = d[m][h].mean() if h.any() else 0.0
        out.append({"n": int(m.sum()), "p_F": pF, "p_R_given_F": pR, "mu_R": muR, "mu_H": muH,
                    "m_reconstructed": pF * pR * muR + (1 - pF) * muH, "m_observed": d[m].mean()})
    return out


def shapley(x0: dict, x1: dict) -> dict:
    keys = ["p_F", "p_R_given_F", "mu_R", "mu_H"]
    f = lambda x: x["p_F"] * x["p_R_given_F"] * x["mu_R"] + (1 - x["p_F"]) * x["mu_H"]  # noqa: E731
    contrib = dict.fromkeys(keys, 0.0)
    perms = list(itertools.permutations(keys))
    for perm in perms:
        cur = {k: x0[k] for k in keys}
        for k in perm:
            before = f(cur)
            cur[k] = x1[k]
            contrib[k] += (f(cur) - before) / len(perms)
    return contrib


def heterogeneity(s, d, F, R) -> dict:
    tert = components(s, d, F, R)
    assert all(abs(c["m_reconstructed"] - c["m_observed"]) < 1e-9 for c in tert), "decomposition not exact"
    point = shapley(tert[0], tert[-1])
    rng = np.random.default_rng(SEED)
    draws = {k: np.empty(NBOOT) for k in point}
    for j in range(NBOOT):
        i = rng.integers(0, len(d), len(d))
        c = components(s[i], d[i], F[i], R[i])
        for k, v in shapley(c[0], c[-1]).items():
            draws[k][j] = v
    dec = components(s, d, F, R, cuts=np.quantile(s, np.linspace(0, 1, 11)[1:-1]))
    return {"tertiles": tert, "m_change_bottom_to_top": tert[-1]["m_observed"] - tert[0]["m_observed"],
            "shapley": {k: {"estimate": v, "ci": [float(np.quantile(draws[k], q)) for q in (.025, .975)]}
                        for k, v in point.items()}, "deciles": dec}


def analyse(corpus: str, folds: list[tuple[pd.DataFrame, pd.DataFrame]], rerank_s: float) -> dict:
    note: list[str] = []
    preds = {k: [] for k in ("RCBR_full", "RCBR_cheap", "Mono_full", "Mono_cheap", "Mono_base",
                             "ablation_noR", "ablation_noMu", "pR_full")}
    tests = []
    for tr, te in folds:
        full, cheap = fit_predict(tr, te, BASE + DEEP, note), fit_predict(tr, te, CHEAP_BASE + DEEP, note)
        base_mono = lgb.LGBMRegressor(**LGB).fit(tr[BASE].to_numpy(), tr["delta"].to_numpy()).predict(te[BASE].to_numpy())
        for k, v in (("RCBR_full", full["rcbr"]), ("RCBR_cheap", cheap["rcbr"]), ("Mono_full", full["mono"]),
                     ("Mono_cheap", cheap["mono"]), ("Mono_base", base_mono), ("ablation_noR", full["noR"]),
                     ("ablation_noMu", full["noMu"]), ("pR_full", full["pR"])):
            preds[k].append(v)
        tests.append(te)
    te = pd.concat(tests)
    d, F, R, S = te["delta"].to_numpy(), te["F"].to_numpy(), te["R"].to_numpy(), te["S"].to_numpy()
    sc = {k: np.concatenate(v) for k, v in preds.items() if k != "pR_full"}
    sc["S"], sc["random"] = S, None
    pR = np.concatenate(preds["pR_full"])
    pairs = [("RCBR_full", "Mono_full"), ("RCBR_full", "S"), ("RCBR_cheap", "Mono_cheap"),
             ("RCBR_full", "ablation_noR"), ("RCBR_full", "ablation_noMu"), ("Mono_full", "Mono_base")]
    pairs += [(k, "random") for k in sc if k != "random"]
    contrasts, pt = boot_contrasts({k: v for k, v in sc.items()}, d, pairs)
    res = {"n_test": int(len(d)), "n_failing": int(F.sum()), "n_reachable": int(R.sum()), "fallbacks": sorted(set(note)),
           "routing_gain": pt, "contrasts": contrasts,
           "reachability_auc": {"head_full": auc(R[F], pR[F]) if 0 < R[F].sum() < F.sum() else None,
                                "S": auc(R[F], S[F]) if 0 < R[F].sum() < F.sum() else None},
           "spearman_delta": {k: float(spearmanr(v, d).statistic) for k, v in sc.items() if v is not None},
           "helped_auc": {k: auc(d > EPS, v) for k, v in sc.items() if v is not None}}
    always = float(d.mean())
    res["cost"] = {"rerank_median_s": rerank_s, "always_gain": always, "always_cost_s": rerank_s,
                   **{f"b{int(b * 100)}": {"cost_s": b * rerank_s,
                                            **{k: route_at(sc[k], d, b) for k in ("RCBR_cheap", "Mono_cheap", "S")},
                                            "random": b * always} for b in (0.50, 0.75)}}
    res["regret_S"] = regret(S, d)
    res["heterogeneity"] = heterogeneity(S, d, F, R)
    return res


def main() -> None:
    global NBOOT
    if "--smoke" in sys.argv:  # pipeline check only; never written to docs
        NBOOT = 50
    lat = {c: json.loads(Path(f"results/{c}/latency.json").read_text())["full"]["median_s"] for c in ("dbpedia", "hotpotqa")}
    res = {"_note": "EXPLORATORY (configs/actionability_protocol.json, committed before computing). Fixed-pool "
                    "reranking. Aggregates only."}
    ps = {}
    for corpus in ("nq", "tripclick", "arguana", "fiqa", "dbpedia", "hotpotqa"):
        folds = nq_tripclick(corpus) if corpus in ("nq", "tripclick") else beir(corpus)
        r = analyse(corpus, folds, RERANK_S.get(corpus, lat.get(corpus)))
        res[corpus] = r
        ps[corpus] = r["contrasts"]["RCBR_full - Mono_full"]["p"]
        c = r["contrasts"]
        h = r["heterogeneity"]["shapley"]
        print(f"{corpus:9s} RCBR-Mono {c['RCBR_full - Mono_full']['estimate']:+.4f} {np.round(c['RCBR_full - Mono_full']['ci'], 4)} "
              f"RCBR-S {c['RCBR_full - S']['estimate']:+.4f} reachAUC head {r['reachability_auc']['head_full']} S {r['reachability_auc']['S']} "
              f"| regret info {np.round(r['regret_S']['information_regret'], 4)} policy {np.round(r['regret_S']['policy_regret_raw'], 4)} "
              f"| dm {r['heterogeneity']['m_change_bottom_to_top']:+.3f} shap {{{', '.join(f'{k}: {v['estimate']:+.3f}' for k, v in h.items())}}}",
              flush=True)
    da = _load("da50", "scripts/50_decision_analyses.py")
    res["primary_holm_RCBR_minus_Mono"] = da.holm(ps)

    def rnd(x):
        if isinstance(x, (float, np.floating)):
            return None if np.isnan(x) else round(float(x), 4)
        if isinstance(x, (bool, np.bool_)):
            return bool(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, dict):
            return {k: rnd(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [rnd(v) for v in x]
        return x
    if "--smoke" in sys.argv:
        print("smoke ok")
        return
    Path("docs/actionability.json").write_text(json.dumps(rnd(res), indent=2) + "\n")
    print("done")


if __name__ == "__main__":
    main()
