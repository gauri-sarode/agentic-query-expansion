#!/usr/bin/env python3
"""ArguAna generalization analysis, as pre-committed in
configs/arguana_precommit.json.

  select-prf  validation only: choose the PRF setting from the 108-setting grid
  select      train + validation only: choose each learned scorer's
              hyperparameters per intervention (validation budget-averaged gain)
  test        refit on train + val, score the untouched test split once
  export      write results/arguana/arguana_frozen.csv in the input schema of
              scripts/54_baseline_runner.py (the external protocol's primary
              analysis); part of the single test pass

`select-prf` and `select` never compute a test label. Outputs:
results/arguana/{prf_selected,selection}.json, results/arguana_analysis.json
(local), docs/arguana_summary.json (rounded aggregates).

Usage: PYTHONPATH=. python3 scripts/53_arguana_analysis.py {select-prf,select,test,export}
"""
from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import load_qrels


def _module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


da = _module("da", "scripts/50_decision_analyses.py")
ra, iv = da.ra, da.iv
EPS = ra.EPS
BUDGETS = da.BUDGETS
RUNS = Path("results/arguana")
SPLIT = json.loads(Path("configs/arguana_split.json").read_text())
INTERVENTIONS = ("rerank", "prf", "qe")
HGB_GRID = list(itertools.product((0.03, 0.1), (2, 3, None), (100, 300), (20, 50)))
KNN_GRID = (5, 10, 20, 50)
LEARNED = ("benefit_linear", "B1_gbdt", "B2_semantic_knn", "B3_quantile")


def _runs(step: str) -> dict:
    return json.loads((RUNS / f"{step}.json").read_text())


class Data:
    """Per-split arrays. Labels are computed only for the splits asked for."""

    def __init__(self, splits: tuple[str, ...], prf: tuple[int, int, float] | None):
        qrels = load_qrels("arguana")
        bm25, rr = _runs("bm25"), _runs("rerank")
        qe = _runs("qe") if (RUNS / "qe.json").exists() else None
        prf_exp = _runs(f"prf_d{prf[0]}_t{prf[1]}") if prf else None
        emb = np.load(RUNS / "embed.npy")
        emb_row = {q: i for i, q in enumerate(json.loads((RUNS / "embed_ids.json").read_text()))}
        self.S = {}
        for sp in splits:
            qids = SPLIT[sp]
            b = iv.ndcg(qids, {q: [d for d, _ in bm25[q]["ranking"]] for q in qids}, qrels)
            s = {"qids": qids, "b": b, "fail": (b <= EPS).astype(int),
                 "t0": np.array([bm25[q]["t0"] for q in qids]),
                 "pre": np.array([bm25[q]["t0"] + [bm25[q]["nqc"], bm25[q]["qlen"]] for q in qids]),
                 "nqc": np.array([bm25[q]["nqc"] for q in qids]),
                 "emb": emb[[emb_row[q] for q in qids]], "iv": {}}
            new = {"rerank": {q: rr[q]["ranking"] for q in qids}}
            if prf_exp:
                new["prf"] = {q: iv.prf_fuse(bm25[q]["ranking"], prf_exp[q]["ranking"], prf[2]) for q in qids}
            if qe:
                new["qe"] = {q: [[d, None] for d in qe[q]["ranking"]] if qe[q]["ranking"] else bm25[q]["ranking"]
                             for q in qids}
            for a, rk in new.items():
                n = iv.ndcg(qids, {q: [d for d, _ in rk[q]] for q in qids}, qrels)
                acted = np.array([qe[q]["ranking"] is not None for q in qids]) if a == "qe" else np.ones(len(qids), bool)
                runs = {"rerank": rr, "prf": prf_exp, "qe": qe}[a]
                s["iv"][a] = {"delta": n - b, "acted": acted,
                              "post": np.array([iv.post_features(bm25[q]["ranking"], rk[q]) for q in qids]),
                              "ms": np.array([runs[q].get("ms", 0.0) for q in qids])}
            self.S[sp] = s

    def merged(self, a: str, *splits: str) -> dict:
        S = [self.S[sp] for sp in splits]
        cat = lambda key, src: np.concatenate([x[key] if src is None else x["iv"][a][key] for x in S])  # noqa: E731
        return {"t0": cat("t0", None), "pre": cat("pre", None), "emb": cat("emb", None), "fail": cat("fail", None),
                "delta": cat("delta", a), "post": cat("post", a), "acted": cat("acted", a)}


# ---------------------------------------------------------------- scorers

def fit_predict(kind: str, hp, tr: dict, te: dict) -> np.ndarray:
    if kind == "failure":
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)).fit(tr["t0"], tr["fail"])
        return m.predict_proba(te["t0"])[:, 1]
    if kind == "benefit_linear":
        return make_pipeline(StandardScaler(), LinearRegression()).fit(tr["pre"], tr["delta"]).predict(te["pre"])
    if kind in ("B1_gbdt", "B3_quantile", "verify_gbdt"):
        lr, depth, iters, leaf = hp
        loss = {"loss": "quantile", "quantile": 0.25} if kind == "B3_quantile" else {}
        X = (lambda d: np.hstack([d["pre"], d["post"]])) if kind == "verify_gbdt" else (lambda d: d["pre"])
        m = HistGradientBoostingRegressor(learning_rate=lr, max_depth=depth, max_iter=iters,
                                          min_samples_leaf=leaf, random_state=42, **loss)
        return m.fit(X(tr), tr["delta"]).predict(X(te))
    if kind == "B2_semantic_knn":
        sim = te["emb"] @ tr["emb"].T
        nn = np.argsort(-sim, axis=1, kind="stable")[:, :hp]
        return tr["delta"][nn].mean(1)
    raise ValueError(kind)


def grid(kind: str):
    if kind in ("B1_gbdt", "B3_quantile", "verify_gbdt"):
        return HGB_GRID
    return KNN_GRID if kind == "B2_semantic_knn" else [None]


def val_criterion(kind: str, score: np.ndarray, d: np.ndarray) -> float:
    if kind == "verify_gbdt":  # verification keeps queries with predicted Delta > 0
        return float(np.where(score > 0, d, 0).mean())
    return float(da._gains(score, d).mean())


# ---------------------------------------------------------------- modes

def select_prf() -> None:
    qrels = load_qrels("arguana")
    qids = SPLIT["val"]
    bm25 = _runs("bm25")
    b = iv.ndcg(qids, {q: [d for d, _ in bm25[q]["ranking"]] for q in qids}, qrels)
    res = {}
    for d, t in iv.PRF_RETRIEVAL:
        exp = _runs(f"prf_d{d}_t{t}")
        for lam in sorted({g[2] for g in iv.PRF_GRID}):
            fused = {q: iv.prf_fuse(bm25[q]["ranking"], exp[q]["ranking"], lam) for q in qids}
            res[(d, t, lam)] = float(iv.ndcg(qids, {q: [x for x, _ in fused[q]] for q in qids}, qrels).mean())
    best = max(res, key=res.get)
    out = {"fb_docs": best[0], "fb_terms": best[1], "lam": best[2], "val_ndcg": res[best],
           "val_bm25": float(b.mean()), "n_settings": len(res)}
    (RUNS / "prf_selected.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def _prf() -> tuple[int, int, float]:
    p = json.loads((RUNS / "prf_selected.json").read_text())
    return p["fb_docs"], p["fb_terms"], p["lam"]


def select() -> None:
    D = Data(("train", "val"), _prf())
    sel = {}
    for a in INTERVENTIONS:
        if a not in D.S["val"]["iv"]:
            continue
        tr, va = D.merged(a, "train"), D.merged(a, "val")
        sel[a] = {}
        for kind in (*LEARNED, "verify_gbdt"):
            scores = {str(hp): val_criterion(kind, fit_predict(kind, hp, tr, va), va["delta"]) for hp in grid(kind)}
            best = max(grid(kind), key=lambda hp: scores[str(hp)])
            sel[a][kind] = {"hp": best, "val_criterion": scores[str(best)], "grid": scores}
        sel[a]["val_reference"] = {
            "failure": val_criterion("failure", fit_predict("failure", None, tr, va), va["delta"]),
            "oracle": val_criterion("oracle", va["delta"], va["delta"]), "mean_delta": float(va["delta"].mean())}
        print(a, {k: (v["hp"], round(v["val_criterion"], 4)) for k, v in sel[a].items() if "hp" in v},
              sel[a]["val_reference"], flush=True)
    (RUNS / "selection.json").write_text(json.dumps(sel, indent=2, default=str))


def _hp(v):
    return tuple(v) if isinstance(v, list) else v


def routing(d: np.ndarray, scorers: dict[str, np.ndarray]) -> dict:
    curves = {k: da._gains(v, d) for k, v in scorers.items()}
    curves["random"] = np.array([b * d.mean() for b in BUDGETS])
    boots = {k: [] for k in scorers}
    for i in ra._boot_idx(len(d)):
        for k, v in scorers.items():
            boots[k].append(da._gains(v[i], d[i]))
    boots = {k: np.array(v) for k, v in boots.items()}
    out = {"curves": {k: c.tolist() for k, c in curves.items()},
           "budget_averaged": {k: float(c.mean()) for k, c in curves.items()},
           "per_budget_ci": {k: [list(ra._ci(v[:, j])) for j in range(len(BUDGETS))] for k, v in boots.items()},
           "vs_failure": {}}
    for k in scorers:
        if k in ("failure", "oracle"):
            continue
        diff = boots[k].mean(1) - boots["failure"].mean(1)
        point = float(curves[k].mean() - curves["failure"].mean())
        headroom = float(curves["oracle"].mean() - curves["failure"].mean())
        out["vs_failure"][k] = {"point": point, **dict(zip(("ci_lo", "ci_hi"), ra._ci(diff))), "p": da._p(diff),
                                "share_of_oracle_minus_failure_headroom": point / headroom if headroom > 0 else None}
    b3_b1 = boots["B3_quantile"] - boots["B1_gbdt"]
    out["B3_minus_B1_per_budget"] = [{"point": float(curves["B3_quantile"][j] - curves["B1_gbdt"][j]),
                                      **dict(zip(("ci_lo", "ci_hi"), ra._ci(b3_b1[:, j])))} for j in range(len(BUDGETS))]
    return out


def harmed_among_routed(score: np.ndarray, d: np.ndarray) -> list[float]:
    order = np.argsort(-score, kind="stable")
    return [float((d[order[: int(round(b * len(d)))]] < -EPS).mean()) for b in BUDGETS]


def abstaining(score: np.ndarray, d: np.ndarray) -> list[dict]:
    n, order = len(d), np.argsort(-score, kind="stable")
    pos = order[score[order] > 0]
    out = []
    for b in BUDGETS:
        take = pos[: int(round(b * n))]
        out.append({"budget_cap": b, "routed": len(take) / n, "gain": float(d[take].sum() / n)})
    return out


def h1(fail, det, d, g) -> dict:
    harmed = (d < -EPS).astype(int)
    res = {"detection_auc": ra.auc_ci(fail, det), "harm_auc": ra.auc_ci(harmed, -g),
           "primary": da.auc_diff_p(fail, det, harmed, -g), "effect_threshold": {}}
    for t in (0.05, 0.10):
        ht = (d < -t).astype(int)
        res["effect_threshold"][str(t)] = {"harm_auc": ra.auc_ci(ht, -g), "detection_minus_harm": ra.auc_diff(fail, det, ht, -g)}
    nz = np.abs(d) > EPS
    res["untied"] = {"n": int(nz.sum()), "detection_untied": ra.auc_ci(fail[nz], det[nz]),
                     "direction_untied": ra.auc_ci((d[nz] > EPS).astype(int), g[nz]),
                     "difference_untied": ra.auc_diff(fail[nz], det[nz], (d[nz] > EPS).astype(int), g[nz])}
    return res


def test() -> None:
    sel = json.loads((RUNS / "selection.json").read_text())
    D = Data(("train", "val", "test"), _prf())
    out = {"_note": "configs/arguana_precommit.json; test scored once with train+val-fit, validation-selected scorers.",
           "n": {sp: len(SPLIT[sp]) for sp in ("train", "val", "test")},
           "bm25_test_mean": float(D.S["test"]["b"].mean()), "prf_selected": dict(zip(("fb_docs", "fb_terms", "lam"), _prf())),
           "selection": {a: {k: v["hp"] for k, v in s.items() if "hp" in v} for a, s in sel.items()}}
    pA, pB = {}, {}
    for a in INTERVENTIONS:
        if a not in sel:
            continue
        tr, te = D.merged(a, "train", "val"), D.merged(a, "test")
        d, acted = te["delta"], te["acted"]
        det = fit_predict("failure", None, tr, te)
        scorers = {"failure": det, "nqc": -D.S["test"]["nqc"], "oracle": d}
        for kind in LEARNED:
            scorers[kind] = fit_predict(kind, _hp(sel[a][kind]["hp"]), tr, te)
        # action-utility model g on pre + post features (acted-upon queries for LLM QE)
        m_tr = tr["acted"]
        g_model = make_pipeline(StandardScaler(), LinearRegression()).fit(
            np.hstack([tr["pre"], tr["post"]])[m_tr], tr["delta"][m_tr])
        g = g_model.predict(np.hstack([te["pre"], te["post"]]))
        verify = fit_predict("verify_gbdt", _hp(sel[a]["verify_gbdt"]["hp"]), tr, te)
        ms = D.S["test"]["iv"][a]["ms"][acted]
        S1 = {"fail": te["fail"], "delta": d}
        res = {
            "label_rates": {"n": int(len(d)), "acted": float(acted.mean()), "mean_delta": float(d.mean()),
                            "helped": float((d > EPS).mean()), "harmed": float((d < -EPS).mean()),
                            "tied": float((np.abs(d) <= EPS).mean()), "failing": float(te["fail"].mean()),
                            "always_vs_original": ra.mean_diff(np.zeros_like(d), d)},
            "latency_ms": {"median": float(np.median(ms)) if len(ms) else None, "p95": float(np.percentile(ms, 95)) if len(ms) else None,
                           "mean": float(ms.mean()) if len(ms) else None, "calls": int(len(ms))},
            "H1": h1(te["fail"][acted], det[acted], d[acted], g[acted]),
            "H2": da.d1(S1),
            "H3": routing(d, scorers),
            "harmed_among_routed": {k: harmed_among_routed(scorers[k], d) for k in ("failure", *LEARNED)},
            "abstaining": {k: abstaining(scorers[k], d) for k in LEARNED},
            "verification": {"routed_fraction": float((verify > 0).mean()), "gain": ra.mean_diff(np.zeros_like(d), np.where(verify > 0, d, 0)),
                             "note": "post-action: the intervention is computed for every query; compute is not saved"},
        }
        pA[a] = res["H1"]["primary"]["p"]
        for k in ("B1_gbdt", "B2_semantic_knn", "B3_quantile"):
            pB[f"{a}:{k}"] = res["H3"]["vs_failure"][k]["p"]
        out[a] = res
    out["sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in [Path("configs/arguana_split.json"), *sorted(RUNS.glob("*.json")), RUNS / "embed.npy"]}
    out["holm_family_A_H1"] = da.holm(pA)
    out["holm_family_B_H3"] = da.holm(pB)
    for a in INTERVENTIONS:
        if a not in out:
            continue
        vf = out[a]["H3"]["vs_failure"]
        out[a]["H3_closes_gap"] = any(
            out["holm_family_B_H3"][f"{a}:{k}"]["reject"] and vf[k]["point"] > 0
            and (vf[k]["share_of_oracle_minus_failure_headroom"] or 0) >= 0.25
            for k in ("B1_gbdt", "B2_semantic_knn", "B3_quantile"))
        out[a]["H1_replicates"] = bool(out["holm_family_A_H1"][a]["reject"] and out[a]["H1"]["primary"]["point"] > 0)
    out = ra._clean(out)
    Path("results/arguana_analysis.json").write_text(json.dumps(out, indent=2))
    Path("docs/arguana_summary.json").write_text(json.dumps(summary(out), indent=2))
    print(json.dumps(summary(out), indent=2))


def summary(out: dict) -> dict:
    r = lambda x: None if x is None else round(x, 4)  # noqa: E731
    s = {"_note": out["_note"], "n": out["n"], "split_sha256": out["sha256"]["arguana_split.json"], "bm25_test_mean": r(out["bm25_test_mean"]),
         "prf_selected": out["prf_selected"], "selection": out["selection"],
         "holm_family_A_H1": out["holm_family_A_H1"], "holm_family_B_H3": out["holm_family_B_H3"]}
    for a in INTERVENTIONS:
        if a not in out:
            continue
        x = out[a]
        s[a] = {"label_rates": {k: r(v) for k, v in x["label_rates"].items() if not isinstance(v, dict)},
                "always_ci": [r(x["label_rates"]["always_vs_original"][k]) for k in ("ci_lo", "ci_hi")],
                "latency_ms": {k: r(v) for k, v in x["latency_ms"].items()},
                "H1": {k: {kk: r(vv) for kk, vv in x["H1"][k].items() if kk in ("point", "ci_lo", "ci_hi", "p")}
                       for k in ("detection_auc", "harm_auc", "primary")},
                "H1_effect_threshold": {t: {kk: r(vv) for kk, vv in v["detection_minus_harm"].items() if kk != "n"}
                                        for t, v in x["H1"]["effect_threshold"].items()},
                "H1_untied_difference": {kk: r(vv) for kk, vv in x["H1"]["untied"]["difference_untied"].items()},
                "H2": {k: r(x["H2"][k]) for k in ("failing", "rescue_rate", "healthy_helped", "healthy_harmed",
                                                  "healthy_tied", "oracle", "share_failing")} | {"share_ci": [r(v) for v in x["H2"]["share_ci"]]},
                "H2_cost_aware": {k: {kk: r(vv) for kk, vv in v.items()} for k, v in x["H2"]["cost_aware"].items()},
                "H3_budget_averaged": {k: r(v) for k, v in x["H3"]["budget_averaged"].items()},
                "H3_curves": {k: [r(v) for v in c] for k, c in x["H3"]["curves"].items()},
                "H3_vs_failure": {k: {kk: r(vv) for kk, vv in v.items()} for k, v in x["H3"]["vs_failure"].items()},
                "B3_minus_B1_per_budget": [{k: r(v) for k, v in e.items()} for e in x["H3"]["B3_minus_B1_per_budget"]],
                "harmed_among_routed": {k: [r(v) for v in c] for k, c in x["harmed_among_routed"].items()},
                "abstaining": {k: [{kk: r(vv) for kk, vv in e.items()} for e in c] for k, c in x["abstaining"].items()},
                "verification": {"routed_fraction": r(x["verification"]["routed_fraction"]),
                                 **{k: r(x["verification"]["gain"][k]) for k in ("point", "ci_lo", "ci_hi")}},
                "H1_replicates": x["H1_replicates"], "H3_closes_gap": x["H3_closes_gap"]}
    return s


def export() -> None:
    """Input CSV for scripts/54_baseline_runner.py: z_* = the 10 pre-action
    features, emb_* = the frozen embedding, nqc_score = NQC (larger = better)."""
    import csv

    D = Data(("train", "val", "test"), _prf())
    names = ["top_score", "score_margin", "score_entropy", "query_term_coverage", "mean_rare_term_idf",
             "entity_overlap", "result_coherence", "ranking_stability", "nqc", "qlen"]
    dim = D.S["train"]["emb"].shape[1]
    path = RUNS / "arguana_frozen.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "split", "baseline_ndcg", *[f"{a}_ndcg" for a in INTERVENTIONS],
                    *[f"z_{n}" for n in names], *[f"emb_{j:03d}" for j in range(dim)], "nqc_score"])
        for sp in ("train", "val", "test"):
            s = D.S[sp]
            for i, q in enumerate(s["qids"]):
                w.writerow([q, sp, s["b"][i], *[s["b"][i] + s["iv"][a]["delta"][i] for a in INTERVENTIONS],
                            *s["pre"][i], *s["emb"][i], s["nqc"][i]])
    print(path, hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    {"select-prf": select_prf, "select": select, "test": test, "export": export}[sys.argv[1]]()
