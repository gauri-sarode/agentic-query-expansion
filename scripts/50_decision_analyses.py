#!/usr/bin/env python3
"""Post-hoc decision analyses on frozen test traces, as pre-committed in
configs/decision_analyses_precommit.json: the failure-only headroom
decomposition (D1), budget-matched pre-action routing (D2), post-action
verification (D3), and Holm / label / effect-threshold statistics (D4).
No new retrieval run, generator call, or relevance judgment.

Writes results/decision_analyses.json (local) and the aggregate export
docs/decision_analyses_summary.json.

Usage: PYTHONPATH=. python3 scripts/50_decision_analyses.py
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.agent.controller import SelectorThresholds
from src.data import load_queries


def _module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ra = _module("ra", "scripts/46_revision_analyses.py")
iv = _module("iv", "scripts/49_intervention_analysis.py")
EPS = ra.EPS
BUDGETS = (0.10, 0.25, 0.50, 0.75, 1.00)
KAPPAS = (0.0, 0.01, 0.02)
THRESHOLDS = (0.0, 0.05, 0.10)


def _p(vals: np.ndarray) -> float:
    vals = np.asarray(vals)
    return float(min(1.0, 2 * min((vals <= 0).mean(), (vals >= 0).mean())))


def auc_diff_p(y1, s1, y2, s2) -> dict:
    y1, s1, y2, s2 = map(np.asarray, (y1, s1, y2, s2))
    vals = np.array([roc_auc_score(y1[i], s1[i]) - roc_auc_score(y2[i], s2[i])
                     for i in ra._boot_idx(len(y1)) if ra._ok(y1[i]) and ra._ok(y2[i])])
    lo, hi = ra._ci(vals)
    return {"point": float(roc_auc_score(y1, s1) - roc_auc_score(y2, s2)), "ci_lo": lo, "ci_hi": hi, "p": _p(vals)}


def holm(ps: dict[str, float], alpha: float = 0.05) -> dict:
    order = sorted(ps, key=ps.get)
    out, running, m = {}, 0.0, len(ps)
    for k, name in enumerate(order):
        running = max(running, min(1.0, (m - k) * ps[name]))
        out[name] = {"p": ps[name], "p_holm": running, "reject": running < alpha}
    return out


def nqc(ranking: list[list]) -> float:
    sc = np.array(sorted((s for _, s in ranking[:100]), reverse=True), dtype=float)
    return float(sc.std() / sc.mean()) if len(sc) and sc.mean() > 0 else 0.0


def replay_acted(X: np.ndarray, names: list[str], qids: list[str], queries: dict) -> np.ndarray:
    """The planner's deterministic NO_EXPAND rule, as scripts/46 A5."""
    t, col = SelectorThresholds(), {n: names.index(n) for n in names}
    return np.array([
        x[col["entity_overlap"]] < t.min_entity_overlap
        or x[col["query_term_coverage"]] < t.min_term_coverage
        or len(queries[q].split()) <= t.max_query_tokens_for_context
        or x[col["score_entropy"]] > t.max_entropy_for_context
        or x[col["result_coherence"]] < t.max_coherence_for_ambiguity
        for q, x in zip(qids, X)])


def _benefit_pre(X_val, d_val, X_test) -> np.ndarray:
    return make_pipeline(StandardScaler(), LinearRegression()).fit(X_val, d_val).predict(X_test)


# ---------------------------------------------------------------- settings

def llm_qe_settings() -> list[dict]:
    T, N = ra.tripclick_bm25(), ra.nq_bm25()
    out = []

    fd = ra._load("results/tripclick-tail_failure_detection.json")
    a4 = ra.a4_tripclick(T)
    b, s = a4["per_query"]["b"], a4["per_query"]["s"]
    bm = iv._runs("tripclick", "test", "bm25")
    benefit, note = None, "TripClick: validation action-utility label (no fused validation ranking)"
    if T["pre_val_acted"] is not None:
        benefit = _benefit_pre(T["pre_val_acted"], T["delta_val"], np.array(fd["X_test"]))
    out.append({"name": "llm_qe_tripclick", "delta": s - b, "fail": T["y_poor_all"], "det": T["det_all"],
                "nqc": np.array([nqc(bm[q]) for q in fd["test_qids"]]), "benefit_pre": benefit, "benefit_note": note,
                "acted": T["acted"], "au_delta": T["delta"], "au_score": T["au_gain"]})

    recs = N["recs"]
    b = np.array([r["original_ndcg"] for r in recs])
    s = np.array([r["candidate_ndcg"] if r["candidate_ndcg"] is not None else r["original_ndcg"] for r in recs])
    cal = ra._load("results/nq_val_calibration.json")
    X_val = np.array(cal["X_t0"])
    acted_val = replay_acted(X_val, cal["t0_feature_names"], cal["val_qids"], load_queries("nq"))
    aligned = int((~acted_val).sum()) == cal["skipped_no_expand"] and int(acted_val.sum()) == len(cal["y_delta_ndcg"])
    t0 = ra._load("results/nq_extension_t0_features.json")["t0_features"]
    benefit = _benefit_pre(X_val[acted_val], np.array(cal["y_delta_ndcg"]), np.array([t0[r["qid"]] for r in recs])) if aligned else None
    bm = iv._runs("nq", "test", "bm25")
    out.append({"name": "llm_qe_nq", "delta": s - b, "fail": N["y_poor_all"], "det": N["det_all"],
                "nqc": np.array([nqc(bm[r["qid"]]) for r in recs]), "benefit_pre": benefit,
                "benefit_note": f"NQ planner replay aligned: {aligned} ({int((~acted_val).sum())} skipped vs {cal['skipped_no_expand']} recorded)",
                "acted": N["acted"], "au_delta": N["delta"], "au_score": N["au_gain"]})
    return out


def intervention_settings() -> list[dict]:
    sel = json.loads((iv.RUNS / "prf_selected.json").read_text())
    out = []
    for corpus in ("tripclick", "nq"):
        inp = iv.corpus_inputs(corpus)
        d, t, lam = (sel[corpus][k] for k in ("fb_docs", "fb_terms", "lam"))
        split = {}
        for sp in ("val", "test"):
            qids, qrels = inp[sp]["qids"], inp[sp]["qrels"]
            bm = iv._runs(corpus, sp, "bm25")
            rr = iv._runs(corpus, sp, "rerank")
            ex = iv._runs(corpus, sp, f"prf_d{d}_t{t}")
            fused = {q: iv.prf_fuse(bm[q], ex[q]["ranking"], lam) for q in qids}
            b = iv.ndcg(qids, {q: [x for x, _ in bm[q]] for q in qids}, qrels)
            split[sp] = {"b": b, "bm": bm, "qids": qids,
                         "rerank": (iv.ndcg(qids, {q: [x for x, _ in rr[q]] for q in qids}, qrels),
                                    np.array([iv.post_features(bm[q], rr[q]) for q in qids])),
                         "prf": (iv.ndcg(qids, {q: [x for x, _ in fused[q]] for q in qids}, qrels),
                                 np.array([iv.post_features(bm[q], fused[q]) for q in qids]))}
        V, T = split["val"], split["test"]
        fail = (T["b"] <= EPS).astype(int)
        q_nqc = np.array([nqc(T["bm"][q]) for q in T["qids"]])
        for name in ("rerank", "prf"):
            dv, dt = V[name][0] - V["b"], T[name][0] - T["b"]
            post = make_pipeline(StandardScaler(), LinearRegression()).fit(
                np.hstack([inp["val"]["pre"], V[name][1]]), dv).predict(np.hstack([inp["test"]["pre"], T[name][1]]))
            out.append({"name": f"{name}_{corpus}", "delta": dt, "fail": fail, "det": inp["test"]["det"], "nqc": q_nqc,
                        "benefit_pre": _benefit_pre(inp["val"]["pre"], dv, inp["test"]["pre"]), "benefit_note": "",
                        "acted": np.ones(len(dt), bool), "au_delta": dt, "au_score": post, "tau": inp["tau"]})
    return out


def dense_settings() -> list[dict]:
    out = []
    for path, name in (("results/dense_retrieval_full_scale_raw.json", "llm_qe_tripclick_dense"),
                       ("results/nq_dense_retrieval_raw.json", "llm_qe_nq_dense")):
        S = ra.dense(path, name)
        delta = np.zeros(len(S["acted"]))
        delta[S["acted"]] = S["delta"]
        out.append({"name": name, "delta": delta, "fail": S["y_poor_all"], "det": S["det_all"],
                    "acted": S["acted"], "au_delta": S["delta"], "au_score": S["au_gain"]})
    return out


# ---------------------------------------------------------------- analyses

def d1(S: dict) -> dict:
    d, F = S["delta"], S["fail"].astype(bool)
    dp = np.maximum(d, 0)
    assert (d[F] >= -EPS).all(), "Delta < 0 on a failing query: the identity's premise fails"
    share = [dp[i][F[i]].sum() / dp[i].sum() for i in ra._boot_idx(len(d)) if dp[i].sum() > 0]
    h = ~F
    res = {
        "n": int(len(d)), "failing": float(F.mean()),
        "rescue_rate": float((d[F] > EPS).mean()),
        "healthy_helped": float((d[h] > EPS).mean()), "healthy_harmed": float((d[h] < -EPS).mean()),
        "healthy_tied": float((np.abs(d[h]) <= EPS).mean()),
        "gain_failing": float((dp * F).mean()), "gain_healthy": float((dp * h).mean()), "oracle": float(dp.mean()),
        "share_failing": float((dp * F).sum() / dp.sum()) if dp.sum() > 0 else None,
        "share_ci": list(ra._ci(share)),
        "cost_aware": {},
    }
    for k in KAPPAS:
        orc, fail = np.maximum(d - k, 0).mean(), ((d - k) * F).mean()
        t1, t2 = (np.maximum(d - k, 0) * h).mean(), (np.maximum(k - d, 0) * F).mean()
        assert abs((orc - fail) - (t1 + t2)) < 1e-12
        res["cost_aware"][str(k)] = {"oracle": float(orc), "perfect_failure_gate": float(fail),
                                     "missed_healthy_gain": float(t1), "wasted_cost_on_unrescued_failures": float(t2)}
    return res


def _gains(order_scores: np.ndarray, d: np.ndarray) -> np.ndarray:
    n = len(d)
    cs = np.concatenate([[0.0], np.cumsum(d[np.argsort(-order_scores, kind="stable")])])
    return np.array([cs[int(round(b * n))] / n for b in BUDGETS])


def d2(S: dict) -> dict:
    d = S["delta"]
    scorers = {"failure": S["det"], "nqc": -S["nqc"], "oracle": d}
    if S["benefit_pre"] is not None:
        scorers["benefit_pre"] = S["benefit_pre"]
    curves = {k: _gains(v, d).tolist() for k, v in scorers.items()}
    curves["random"] = [b * d.mean() for b in BUDGETS]
    boots = {k: [] for k in scorers}
    for i in ra._boot_idx(len(d)):
        for k, v in scorers.items():
            boots[k].append(_gains(v[i], d[i]))
    boots = {k: np.array(v) for k, v in boots.items()}
    out = {"budgets": list(BUDGETS), "curves": curves, "benefit_note": S["benefit_note"],
           "per_budget_ci": {k: [list(ra._ci(v[:, j])) for j in range(len(BUDGETS))] for k, v in boots.items()},
           "budget_averaged": {k: float(np.mean(c)) for k, c in curves.items()}}
    for other in ("benefit_pre", "nqc"):
        if other in boots:
            diff = boots[other].mean(1) - boots["failure"].mean(1)
            out[f"{other}_minus_failure"] = {"point": float(np.mean(curves[other]) - np.mean(curves["failure"])),
                                             **dict(zip(("ci_lo", "ci_hi"), ra._ci(diff))), "p": _p(diff)}
    return out


def d4(S: dict) -> dict:
    acted = S["acted"]
    yp, det = S["fail"][acted], S["det"][acted]
    d, g = S["au_delta"], S["au_score"]
    res = {"primary": auc_diff_p(yp, det, (d < -EPS).astype(int), -g), "effect_threshold": {}}
    for t in THRESHOLDS[1:]:
        harm = (d < -t).astype(int)
        res["effect_threshold"][str(t)] = {"harm_auc": ra.auc_ci(harm, -g), "detection_minus_harm": ra.auc_diff(yp, det, harm, -g)}
    res["effect_threshold"]["0.0"] = {"harm_auc": ra.auc_ci((d < -EPS).astype(int), -g), "detection_minus_harm": {k: res["primary"][k] for k in ("point", "ci_lo", "ci_hi")}}
    return res


def d4_untied(S: dict) -> dict:
    nz = np.abs(S["au_delta"]) > EPS
    yp, det = S["fail"][nz], S["det"][nz]
    helped, g = (S["au_delta"][nz] > EPS).astype(int), S["au_score"][nz]
    return {"n": int(nz.sum()), "detection_untied": ra.auc_ci(yp, det), "direction_untied": ra.auc_ci(helped, g),
            "difference_untied": ra.auc_diff(yp, det, helped, g)}


def main() -> None:
    llm, ivs, dense = llm_qe_settings(), intervention_settings(), dense_settings()
    bm25_six = llm + ivs
    out = {"_note": "POST HOC on frozen, previously inspected test traces (configs/decision_analyses_precommit.json); not independent confirmation."}
    out["D1"] = {S["name"]: d1(S) for S in bm25_six + dense}
    print("D1 done", flush=True)
    out["D2"] = {S["name"]: d2(S) for S in bm25_six}
    for other in ("benefit_pre", "nqc"):
        ps = {k: v[f"{other}_minus_failure"]["p"] for k, v in out["D2"].items() if f"{other}_minus_failure" in v}
        out["D2"][f"holm_{other}_minus_failure"] = holm(ps)
    print("D2 done", flush=True)
    ia = json.loads(open("results/intervention_analysis.json").read())
    out["D3"] = {f"{iv_}_{c}": {"compute_fraction": 1.0, "retained_fraction": ia[c][iv_]["S4_policies"]["benefit_selective"]["routed"],
                                "gain": {k: ia[c][iv_]["S4_policies"]["benefit_selective"][k] for k in ("point", "ci_lo", "ci_hi")}}
                 for c in ("tripclick", "nq") for iv_ in ("rerank", "prf")}
    out["D4"] = {S["name"]: d4(S) for S in bm25_six + dense}
    out["D4"]["holm_primary"] = holm({k: v["primary"]["p"] for k, v in out["D4"].items()})
    out["D4_untied"] = {S["name"]: d4_untied(S) for S in ivs}
    print("D4 done", flush=True)
    out = ra._clean(out)
    Path("results/decision_analyses.json").write_text(json.dumps(out, indent=2))

    def rnd(x):
        if isinstance(x, float):
            return round(x, 4)
        if isinstance(x, dict):
            return {k: rnd(v) for k, v in x.items()}
        if isinstance(x, list):
            return [rnd(v) for v in x]
        return x
    Path("docs/decision_analyses_summary.json").write_text(json.dumps(rnd(out), indent=2) + "\n")
    print(json.dumps(rnd({k: out[k] for k in ("D1", "D3")}), indent=1)[:200])


if __name__ == "__main__":
    main()
