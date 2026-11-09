#!/usr/bin/env python3
"""Detection vs. benefit prediction for reranking and PRF, as
pre-committed in configs/intervention_scope_precommit.json. Reads the
rankings scripts/48 stored, labels them, tunes PRF on validation only,
fits the benefit models on validation only, and scores test once.
Writes results/intervention_analysis.json (local).

Usage: PYTHONPATH=. python3 scripts/49_intervention_analysis.py select   (validation only)
       PYTHONPATH=. python3 scripts/49_intervention_analysis.py          (after test runs)
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import load_qrels
from src.eval.metrics import evaluate
from src.observability.telemetry import _entropy

spec = importlib.util.spec_from_file_location("ra", "scripts/46_revision_analyses.py")
ra = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ra)
EPS = ra.EPS
RUNS = Path("results/interventions")
PRF_RETRIEVAL = [(d, t) for d in (5, 10, 20) for t in (5, 10, 20, 30)]
PRF_GRID = [(d, t, lam) for d, t in PRF_RETRIEVAL for lam in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)]


def _runs(corpus: str, split: str, step: str) -> dict:
    return json.loads((RUNS / f"{corpus}_{split}_{step}.json").read_text())


def ndcg(qids: list[str], rankings: dict[str, list[str]], qrels: dict) -> np.ndarray:
    run = {q: {d: 1.0 / (i + 1) for i, d in enumerate(rankings[q])} for q in qids}
    ev = evaluate({q: qrels[q] for q in qids}, run)
    return np.array([ev[q]["ndcg_cut_10"] if q in ev else 0.0 for q in qids])


def _minmax(pairs: list[list]) -> dict[str, float]:
    if not pairs:
        return {}
    s = np.array([p[1] for p in pairs], dtype=float)
    lo, hi = s.min(), s.max()
    return {d: (v - lo) / (hi - lo) if hi > lo else 1.0 for d, v in zip((p[0] for p in pairs), s)}


def prf_fuse(bm25: list[list], exp: list[list], lam: float) -> list[list]:
    a, b = _minmax(bm25), _minmax(exp)
    fused = {d: lam * a.get(d, 0.0) + (1 - lam) * b.get(d, 0.0) for d in set(a) | set(b)}
    return [[d, s] for d, s in sorted(fused.items(), key=lambda x: (-x[1], x[0]))[:100]]


def rbo(a: list[str], b: list[str], p: float = 0.9, depth: int = 20) -> float:
    return (1 - p) * sum(p ** (k - 1) * len(set(a[:k]) & set(b[:k])) / k for k in range(1, depth + 1))


def post_features(bm25: list[list], new: list[list]) -> list[float]:
    b_ids, n_ids = [d for d, _ in bm25], [d for d, _ in new]
    sc = [s for _, s in new[:10] if s is not None]
    return [
        len(set(b_ids[:10]) & set(n_ids[:10])) / 10.0,
        float(bool(b_ids) and bool(n_ids) and b_ids[0] != n_ids[0]),
        rbo(b_ids, n_ids),
        sc[0] if sc else 0.0,
        sc[0] - sc[1] if len(sc) >= 2 else 0.0,
        _entropy(sc) if sc else 0.0,
    ]


def corpus_inputs(corpus: str) -> dict:
    """qids, qrels, pre-action features and detector outputs per split."""
    if corpus == "tripclick":
        fd = ra._load("results/tripclick-tail_failure_detection.json")
        X_val, X_test = np.array(fd["X_val"]), np.array(fd["X_test"])
        det = LogisticRegression(max_iter=1000).fit(X_val, (np.array(fd["y_val_ndcg"]) <= EPS).astype(int))
        return {"val": {"qids": fd["val_qids"], "qrels": load_qrels("tripclick-tail-val"), "pre": X_val,
                        "stored_bm25_ndcg": np.array(fd["y_val_ndcg"])},
                "test": {"qids": fd["test_qids"], "qrels": load_qrels("tripclick-tail"), "pre": X_test,
                         "det": det.predict_proba(X_test)[:, 1], "stored_bm25_ndcg": np.array(fd["y_test_ndcg"])},
                "tau": float(np.median(det.predict_proba(X_val)[:, 1]))}
    cal = ra._load("results/nq_val_calibration.json")
    recs = ra._load("results/nq_test_extension.json")["records"]
    t0 = ra._load("results/nq_extension_t0_features.json")["t0_features"]
    det = joblib.load("models/nq_failure_detector_val_v1.joblib")["model"]
    qrels = load_qrels("nq")
    return {"val": {"qids": cal["val_qids"], "qrels": qrels, "pre": np.array(cal["X_t0"]),
                    "stored_bm25_ndcg": np.array(cal["y_ndcg"])},
            "test": {"qids": [r["qid"] for r in recs], "qrels": qrels, "pre": np.array([t0[r["qid"]] for r in recs]),
                     "det": np.array([r["poor_proba"] for r in recs]),
                     "stored_bm25_ndcg": np.array([r["original_ndcg"] for r in recs])},
            "tau": float(np.median(det.predict_proba(np.array(cal["X_t0"]))[:, 1]))}


def build(corpus: str, split: str, inp: dict, prf: tuple[int, int, float] | None) -> dict:
    qids, qrels = inp["qids"], inp["qrels"]
    bm25 = _runs(corpus, split, "bm25")
    b = ndcg(qids, {q: [d for d, _ in bm25[q]] for q in qids}, qrels)
    out = {"bm25_ndcg": b, "bm25_reproduces_max_abs": float(np.abs(b - inp["stored_bm25_ndcg"]).max())}
    rr = _runs(corpus, split, "rerank")
    out["rerank"] = {"ndcg": ndcg(qids, {q: [d for d, _ in rr[q]] for q in qids}, qrels),
                     "post": np.array([post_features(bm25[q], rr[q]) for q in qids])}
    if prf is None:  # validation: evaluate the full grid
        exp = {(d, t): _runs(corpus, split, f"prf_d{d}_t{t}") for d, t in PRF_RETRIEVAL}
        out["prf_grid"] = {}
        for d, t, lam in PRF_GRID:
            fused = {q: prf_fuse(bm25[q], exp[(d, t)][q]["ranking"], lam) for q in qids}
            out["prf_grid"][(d, t, lam)] = ndcg(qids, {q: [x for x, _ in fused[q]] for q in qids}, qrels)
        out["prf_exp"] = exp
    else:
        d, t, lam = prf
        exp = _runs(corpus, split, f"prf_d{d}_t{t}")
        fused = {q: prf_fuse(bm25[q], exp[q]["ranking"], lam) for q in qids}
        out["prf"] = {"ndcg": ndcg(qids, {q: [d for d, _ in fused[q]] for q in qids}, qrels),
                      "post": np.array([post_features(bm25[q], fused[q]) for q in qids])}
    out["bm25"] = bm25
    return out


def analyse(b: np.ndarray, new: np.ndarray, g: np.ndarray, det: np.ndarray, tau: float) -> dict:
    d = new - b
    tie, helped, harmed = np.abs(d) <= EPS, d > EPS, d < -EPS
    fail = (b <= EPS).astype(int)
    possible = b > EPS
    gc = g - g.mean()
    blp = [LinearRegression().fit(gc[i, None], d[i]).coef_[0] for i in ra._boot_idx(len(d))]
    route = lambda mask: np.where(mask, new, b)  # noqa: E731
    perfect, oracle = route(fail.astype(bool)), np.maximum(b, new)
    return {
        "S1_label_rates": {"n": int(len(d)), "tied": float(tie.mean()), "helped": float(helped.mean()),
                           "harmed": float(harmed.mean()), "harm_possible": float(possible.mean()),
                           "mean_delta": float(d.mean())},
        "S2_auc": {
            "detection": ra.auc_ci(fail, det),
            "rule_a_harmed_vs_rest": ra.auc_ci(harmed.astype(int), -g),
            "rule_b_helped_vs_rest": ra.auc_ci(helped.astype(int), g),
            "harm_possible_only_rule_a": ra.auc_ci(harmed[possible].astype(int), -g[possible]),
            "detection_minus_rule_a": ra.auc_diff(fail, det, harmed.astype(int), -g),
            "detection_minus_rule_b": ra.auc_diff(fail, det, helped.astype(int), g),
        },
        "S3_blp": {"point": float(LinearRegression().fit(gc[:, None], d).coef_[0]), **dict(zip(("ci_lo", "ci_hi"), ra._ci(blp)))},
        "S4_policies": {
            "always": ra.mean_diff(b, new),
            "failure_selective": {**ra.mean_diff(b, route(det > tau)), "routed": float((det > tau).mean())},
            "benefit_selective": {**ra.mean_diff(b, route(g > 0)), "routed": float((g > 0).mean())},
            "perfect_detector": ra.mean_diff(b, perfect),
            "oracle": ra.mean_diff(b, oracle),
            "perfect_detector_share": float((perfect - b).mean() / (oracle - b).mean()) if (oracle - b).mean() > 0 else None,
        },
    }


def select() -> None:
    """Validation only: pick each corpus's PRF setting and record it before
    any test PRF run exists."""
    sel = {}
    for corpus in ("tripclick", "nq"):
        inp = corpus_inputs(corpus)
        V = build(corpus, "val", inp["val"], None)
        best = max(PRF_GRID, key=lambda k: V["prf_grid"][k].mean())
        sel[corpus] = {"fb_docs": best[0], "fb_terms": best[1], "lam": best[2],
                       "val_ndcg": float(V["prf_grid"][best].mean()), "val_bm25": float(V["bm25_ndcg"].mean()),
                       "bm25_reproduces_max_abs": V["bm25_reproduces_max_abs"]}
    (RUNS / "prf_selected.json").write_text(json.dumps(sel, indent=2))
    print(json.dumps(sel, indent=2))


def main() -> None:
    import sys
    if sys.argv[1:] == ["select"]:
        return select()
    selected = json.loads((RUNS / "prf_selected.json").read_text())
    out = {"_note": "configs/intervention_scope_precommit.json; test scored once with validation-frozen settings."}
    for corpus in ("tripclick", "nq"):
        inp = corpus_inputs(corpus)
        V = build(corpus, "val", inp["val"], None)
        grid = {f"d{d}_t{t}_lam{lam}": float(v.mean()) for (d, t, lam), v in V["prf_grid"].items()}
        best = max(PRF_GRID, key=lambda k: V["prf_grid"][k].mean())
        prf_val = V["prf_grid"][best]
        fused_val = {q: prf_fuse(V["bm25"][q], V["prf_exp"][best[:2]][q]["ranking"], best[2]) for q in inp["val"]["qids"]}
        V["prf"] = {"ndcg": prf_val, "post": np.array([post_features(V["bm25"][q], fused_val[q]) for q in inp["val"]["qids"]])}
        assert dict(zip(("fb_docs", "fb_terms", "lam"), best)) == {k: selected[corpus][k] for k in ("fb_docs", "fb_terms", "lam")}
        T = build(corpus, "test", inp["test"], best)
        res = {"bm25_reproduces_max_abs": {"val": V["bm25_reproduces_max_abs"], "test": T["bm25_reproduces_max_abs"]},
               "prf_val_grid": grid, "prf_selected": dict(zip(("fb_docs", "fb_terms", "lam"), best)),
               "bm25_test_mean": float(T["bm25_ndcg"].mean())}
        for iv in ("rerank", "prf"):
            Xv = np.hstack([inp["val"]["pre"], V[iv]["post"]])
            Xt = np.hstack([inp["test"]["pre"], T[iv]["post"]])
            m = make_pipeline(StandardScaler(), LinearRegression()).fit(Xv, V[iv]["ndcg"] - V["bm25_ndcg"])
            res[iv] = {"val_mean_ndcg": float(V[iv]["ndcg"].mean()), "test_mean_ndcg": float(T[iv]["ndcg"].mean()),
                       **analyse(T["bm25_ndcg"], T[iv]["ndcg"], m.predict(Xt), inp["test"]["det"], inp["tau"])}
        out[corpus] = res
    out = ra._clean(out)
    Path("results/intervention_analysis.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
