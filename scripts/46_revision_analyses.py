#!/usr/bin/env python3
"""Revision analyses A1-A8, exactly as pre-committed in
configs/revision_analyses_precommit.json (written before any of these
quantities was computed on test). No generator calls, no new episodes;
every model is fit on validation and scored once on the frozen test
records. Writes results/revision_analyses.json (gitignored: it holds
aggregates only, but results/ stays local by convention).

A6 for NQ needs results/nq_bm25_score_lists.json from scripts/45; if
that file is missing or incomplete, NQ's A6 entry is left null.

Usage: PYTHONPATH=. python3 scripts/46_revision_analyses.py
"""
from __future__ import annotations

import json
import os

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.agent.controller import SelectorThresholds
from src.data import load_qrels, load_queries
from src.eval.metrics import evaluate

N_BOOTSTRAP = 10000
SEED = 42
EPS = 1e-9


def _load(path: str):
    return json.loads(open(path).read())


def _boot_idx(n: int):
    rng = np.random.RandomState(SEED)
    for _ in range(N_BOOTSTRAP):
        yield rng.randint(0, n, n)


def _ci(vals) -> tuple[float, float]:
    lo, hi = np.percentile(np.asarray(vals), [2.5, 97.5])
    return float(lo), float(hi)


def _ok(y: np.ndarray) -> bool:
    return 0 < y.sum() < len(y)


def auc_ci(y, s) -> dict:
    y, s = np.asarray(y), np.asarray(s)
    if len(y) < 30 or not _ok(y):
        return {"point": None, "n": int(len(y))}
    vals = [roc_auc_score(y[i], s[i]) for i in _boot_idx(len(y)) if _ok(y[i])]
    lo, hi = _ci(vals)
    return {"point": float(roc_auc_score(y, s)), "ci_lo": lo, "ci_hi": hi, "n": int(len(y)), "pos_rate": float(y.mean())}


def auc_diff(y1, s1, y2, s2) -> dict:
    """Paired bootstrap of AUC(y1,s1) - AUC(y2,s2) over the same queries."""
    y1, s1, y2, s2 = map(np.asarray, (y1, s1, y2, s2))
    vals = []
    for i in _boot_idx(len(y1)):
        if _ok(y1[i]) and _ok(y2[i]):
            vals.append(roc_auc_score(y1[i], s1[i]) - roc_auc_score(y2[i], s2[i]))
    lo, hi = _ci(vals)
    return {"point": float(roc_auc_score(y1, s1) - roc_auc_score(y2, s2)), "ci_lo": lo, "ci_hi": hi, "n": int(len(y1))}


def mean_diff(a, b) -> dict:
    """Paired bootstrap of mean(b) - mean(a); se gives the MDE at 80% power."""
    a, b = np.asarray(a), np.asarray(b)
    vals = np.array([b[i].mean() - a[i].mean() for i in _boot_idx(len(a))])
    lo, hi = _ci(vals)
    se = float(vals.std(ddof=1))
    return {"point": float(b.mean() - a.mean()), "ci_lo": lo, "ci_hi": hi, "se": se, "mde80": 2.80 * se, "n": int(len(a))}


# ---------------------------------------------------------------- settings

def tripclick_bm25() -> dict:
    fd = _load("results/tripclick-tail_failure_detection.json")
    vc = _load("results/tripclick-tail_verifier_calibration.json")
    vcv = _load("results/tripclick-tail-val_verifier_calibration.json")
    ck = _load("results/tripclick-tail_agent_vs_static_checkpoint.json")

    X_val, X_test = np.array(fd["X_val"]), np.array(fd["X_test"])
    y_val_poor = (np.array(fd["y_val_ndcg"]) <= EPS).astype(int)
    y_test_ndcg = np.array(fd["y_test_ndcg"])
    det = LogisticRegression(max_iter=1000).fit(X_val, y_val_poor)
    det_test = det.predict_proba(X_test)[:, 1]
    tau = float(np.median(det.predict_proba(X_val)[:, 1]))

    action = {t["query_id"]: t["planner_action"] for t in ck["static_traces"]}
    acted = np.array([action[q] != "NO_EXPAND" for q in fd["test_qids"]])
    assert acted.sum() == len(vc["y_delta_ndcg"]) == 1157

    # A5 alignment: replay the planner's deterministic operator rule on val.
    names = fd["T0_FEATURES"]
    t = SelectorThresholds()
    val_queries = load_queries("tripclick-tail-val")
    col = {n: names.index(n) for n in names}
    val_acted = []
    for q, x in zip(fd["val_qids"], X_val):
        no_expand = not (
            x[col["entity_overlap"]] < t.min_entity_overlap
            or x[col["query_term_coverage"]] < t.min_term_coverage
            or len(val_queries[q].split()) <= t.max_query_tokens_for_context
            or x[col["score_entropy"]] > t.max_entropy_for_context
            or x[col["result_coherence"]] < t.max_coherence_for_ambiguity
        )
        val_acted.append(not no_expand)
    val_acted = np.array(val_acted)
    aligned = int((~val_acted).sum()) == vcv["skipped_no_expand"] and int(val_acted.sum()) == len(vcv["X"])

    return {
        "name": "tripclick_bm25",
        "y_poor_all": (y_test_ndcg <= EPS).astype(int), "det_all": det_test,
        "acted": acted,
        "delta": np.array(vc["y_delta_ndcg"]), "au_gain": np.array(vc["y_current_verify"]),
        "orig_acted": y_test_ndcg[acted],
        "pre_test_acted": X_test[acted], "post_test": np.array(vc["X"]),
        "pre_val_acted": X_val[val_acted] if aligned else None,
        "post_val": np.array(vcv["X"]), "delta_val": np.array(vcv["y_delta_ndcg"]),
        "a5_alignment": {"replayed_no_expand": int((~val_acted).sum()), "recorded": vcv["skipped_no_expand"], "aligned": aligned},
        "tau": tau, "test_qids": fd["test_qids"], "ck": ck,
    }


def nq_bm25() -> dict:
    recs = _load("results/nq_test_extension.json")["records"]
    cal = _load("results/nq_val_calibration.json")
    t0 = _load("results/nq_extension_t0_features.json")["t0_features"]
    X_val = np.array(cal["X_t0"])
    det = joblib.load("models/nq_failure_detector_val_v1.joblib")["model"]  # the frozen val-fit detector scripts/25 used
    recomputed = det.predict_proba(np.array([t0[r["qid"]] for r in recs]))[:, 1]
    frozen = np.array([r["poor_proba"] for r in recs])
    # scripts/38's recomputed t0 features match the originals only to float
    # noise (max |dp| 0.008), so the frozen poor_proba is used throughout and
    # the recompute only checks we hold the same detector.
    assert np.corrcoef(recomputed, frozen)[0, 1] > 0.9999, "frozen NQ detector does not reproduce poor_proba"
    acted = np.array([r["delta_ndcg"] is not None for r in recs])
    ar = [r for r in recs if r["delta_ndcg"] is not None]
    return {
        "name": "nq_bm25",
        "y_poor_all": np.array([r["y_poor"] for r in recs]), "det_all": frozen,
        "acted": acted,
        "delta": np.array([r["delta_ndcg"] for r in ar]), "au_gain": np.array([r["predicted_delta"] for r in ar]),
        "orig_acted": np.array([r["original_ndcg"] for r in ar]),
        "pre_test_acted": None, "post_test": None,
        "a5_alignment": {"aligned": False, "reason": "post-action features were not stored for the NQ BM25 test episodes (scripts/25 kept only the frozen prediction)"},
        "tau": float(np.median(det.predict_proba(X_val)[:, 1])),
        "recs": recs,
    }


def dense(path: str, name: str) -> dict:
    raw = _load(path)
    val, test = raw["val"], raw["test"]
    X_val = np.array([r["t0_features"] for r in val])
    det = LogisticRegression(max_iter=1000).fit(X_val, np.array([r["y_poor"] for r in val]))
    det_test = det.predict_proba(np.array([r["t0_features"] for r in test]))[:, 1]
    va = [r for r in val if "au_features" in r]
    ta = [r for r in test if "au_features" in r]
    harm_val = (np.array([r["delta_ndcg"] for r in va]) < -EPS).astype(int)
    au = LogisticRegression(max_iter=1000).fit(np.array([r["au_features"] for r in va]), harm_val)
    p_harm = au.predict_proba(np.array([r["au_features"] for r in ta]))[:, 1]
    return {
        "name": name,
        "y_poor_all": np.array([r["y_poor"] for r in test]), "det_all": det_test,
        "acted": np.array(["au_features" in r for r in test]),
        "delta": np.array([r["delta_ndcg"] for r in ta]), "au_gain": -p_harm,
        "orig_acted": np.array([r["orig_ndcg"] for r in ta]),
        "pre_test_acted": np.array([r["t0_features"] for r in ta]), "post_test": np.array([r["au_features"] for r in ta]),
        "pre_val_acted": np.array([r["t0_features"] for r in va]), "post_val": np.array([r["au_features"] for r in va]),
        "delta_val": np.array([r["delta_ndcg"] for r in va]),
        "a5_alignment": {"aligned": True, "reason": "records carry both feature sets per query"},
    }


# ---------------------------------------------------------------- analyses

def a1(S: dict) -> dict:
    yp, ds = S["y_poor_all"][S["acted"]], S["det_all"][S["acted"]]
    harm = (S["delta"] < -EPS).astype(int)
    return {
        "detection_full_population": auc_ci(S["y_poor_all"], S["det_all"]),
        "detection_acted": auc_ci(yp, ds),
        "action_utility_acted": auc_ci(harm, -S["au_gain"]),
        "difference_acted": auc_diff(yp, ds, harm, -S["au_gain"]),
    }


def a2_a3(S: dict) -> dict:
    d, g = S["delta"], S["au_gain"]
    tie, helped, harmed = np.abs(d) <= EPS, d > EPS, d < -EPS
    nz = ~tie
    big = nz & (np.abs(d) >= np.median(np.abs(d[nz])))
    possible = S["orig_acted"] > EPS
    gc = g - g.mean()
    blp = [LinearRegression().fit(gc[i, None], d[i]).coef_[0] for i in _boot_idx(len(d))]
    q1, q3 = np.percentile(g, [25, 75])
    top, bot = g >= q3, g <= q1
    gates = [d[i][top[i]].mean() - d[i][bot[i]].mean() for i in _boot_idx(len(d))]
    return {
        "label_rates": {"n": int(len(d)), "tied": float(tie.mean()), "helped": float(helped.mean()), "harmed": float(harmed.mean())},
        "rule_a_harmed_vs_rest": auc_ci(harmed.astype(int), -g),
        "rule_b_helped_vs_rest": auc_ci(helped.astype(int), g),
        "rule_c_helped_vs_harmed": auc_ci(helped[nz].astype(int), g[nz]),
        "noise_large_effects_rule_c": {**auc_ci(helped[big].astype(int), g[big]), "abs_delta_threshold": float(np.median(np.abs(d[nz])))},
        "harm_possible_only_rule_a": auc_ci(harmed[possible].astype(int), -g[possible]),
        "blp_slope": {"point": float(LinearRegression().fit(gc[:, None], d).coef_[0]), **dict(zip(("ci_lo", "ci_hi"), _ci(blp)))},
        "gates_top_minus_bottom_quartile": {"point": float(d[top].mean() - d[bot].mean()), **dict(zip(("ci_lo", "ci_hi"), _ci(gates)))},
    }


def post_hoc_untied(S: dict) -> dict:
    """NOT pre-committed (added after seeing A2/A3): paired difference of
    detection vs. direction-only action utility (rule c) on the same
    untied acted-upon queries. Reported as exploratory."""
    d = S["delta"]
    nz = np.abs(d) > EPS
    yp, ds = S["y_poor_all"][S["acted"]][nz], S["det_all"][S["acted"]][nz]
    helped, g = (d[nz] > EPS).astype(int), S["au_gain"][nz]
    return {"detection_untied": auc_ci(yp, ds), "direction_untied": auc_ci(helped, g),
            "difference_untied": auc_diff(yp, ds, helped, g)}


def a5(S: dict) -> dict:
    if not S["a5_alignment"]["aligned"]:
        return {"alignment": S["a5_alignment"]}
    hv = (S["delta_val"] < -EPS).astype(int)
    ht = (S["delta"] < -EPS).astype(int)

    def fit(Xv, Xt):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Xv, hv)
        return m.predict_proba(Xt)[:, 1]

    pre = fit(S["pre_val_acted"], S["pre_test_acted"])
    post = fit(S["post_val"], S["post_test"])
    both = fit(np.hstack([S["pre_val_acted"], S["post_val"]]), np.hstack([S["pre_test_acted"], S["post_test"]]))
    return {
        "alignment": S["a5_alignment"],
        "n_pre_features": int(S["pre_test_acted"].shape[1]), "n_post_features": int(S["post_test"].shape[1]),
        "pre_only": auc_ci(ht, pre), "post_only": auc_ci(ht, post), "pre_plus_post": auc_ci(ht, both),
        "pre_plus_post_minus_post": auc_diff(ht, both, ht, post),
        "detection_acted_minus_pre_plus_post": auc_diff(S["y_poor_all"][S["acted"]], S["det_all"][S["acted"]], ht, both),
    }


def a4_tripclick(S: dict) -> dict:
    ck = S["ck"]
    qrels = load_qrels("tripclick-tail")
    qids = S["test_qids"]
    ev = {k: evaluate(qrels, ck[k]) for k in ("baseline_run", "static_run", "agent_run")}
    b, s, a = (np.array([ev[k][q]["ndcg_cut_10"] for q in qids]) for k in ("baseline_run", "static_run", "agent_run"))
    route = S["det_all"] > S["tau"]
    fs = np.where(route, s, b)
    fail = S["y_poor_all"].astype(bool)
    perfect = np.where(fail, s, b)
    pol = {"bm25": b, "static": s, "failure_selective": fs, "closed_loop": a}
    keys = list(pol)
    pairs = {f"{keys[j]}_minus_{keys[i]}": mean_diff(pol[keys[i]], pol[keys[j]]) for i in range(4) for j in range(i + 1, 4)}
    gain = s - b
    acted_delta_matches = bool(np.allclose(gain[S["acted"]], S["delta"], atol=1e-6))
    return {
        "routed_fraction": float(route.mean()),
        "pairwise": pairs,
        "perfect_detector_minus_bm25": mean_diff(b, perfect),
        "static_gain_by_true_failure": {"failing": mean_diff(b[fail], s[fail]), "healthy": mean_diff(b[~fail], s[~fail]),
                                         "failing_fraction": float(fail.mean())},
        "static_gain_by_routing": {"routed": mean_diff(b[route], s[route]), "not_routed": mean_diff(b[~route], s[~route])},
        "check_static_minus_bm25_equals_action_utility_delta_on_acted": acted_delta_matches,
        "per_query": {"b": b, "s": s},
    }


def a4_nq(S: dict) -> dict:
    recs = S["recs"]
    # Correction found while running A4: the extension records' accepted_ndcg
    # is the verifier-GATED policy (accepted iff predicted gain >= the frozen
    # accept threshold; ~40% accepted), not unconditional QE. Unconditional QE
    # on NQ is the unfused rewritten ranking (candidate_ndcg) on every acted
    # query. No fused ranking was stored for NQ, so NQ arms are unfused,
    # unlike TripClick's RRF-fused Static QE.
    b = np.array([r["original_ndcg"] for r in recs])
    s = np.array([r["candidate_ndcg"] if r["candidate_ndcg"] is not None else r["original_ndcg"] for r in recs])
    gated = np.array([r["accepted_ndcg"] for r in recs])
    route = S["det_all"] > S["tau"]
    fs = np.where(route, s, b)
    fail = S["y_poor_all"].astype(bool)
    return {
        "routed_fraction": float(route.mean()),
        "bm25_mean": float(b.mean()),
        "note": "NQ arms use the unfused rewritten ranking; accepted_ndcg is a verifier-gated policy, reported separately",
        "static_unfused_minus_bm25": mean_diff(b, s),
        "verifier_gated_minus_bm25": mean_diff(b, gated),
        "oracle_unfused_minus_bm25": mean_diff(b, np.maximum(b, s)),
        "failure_selective_minus_bm25": mean_diff(b, fs),
        "failure_selective_minus_static": mean_diff(s, fs),
        "perfect_detector_minus_bm25": mean_diff(b, np.where(fail, s, b)),
        "static_gain_by_true_failure": {"failing": mean_diff(b[fail], s[fail]), "healthy": mean_diff(b[~fail], s[~fail]),
                                         "failing_fraction": float(fail.mean())},
    }


def qpp(scores: list[list[float]], qlens: list[int]) -> tuple[np.ndarray, np.ndarray]:
    wig, nqc = [], []
    for sc, ql in zip(scores, qlens):
        sc = np.array(sorted(sc, reverse=True), dtype=float)
        if len(sc) == 0:
            wig.append(0.0)
            nqc.append(0.0)
            continue
        wig.append(sc[:5].mean() / np.sqrt(max(ql, 1)))
        top = sc[:100]
        nqc.append(top.std() / top.mean() if top.mean() > 0 else 0.0)
    return np.array(wig), np.array(nqc)


def a6(S: dict, scores: list[list[float]] | None, qtexts: list[str]) -> dict | None:
    if scores is None:
        return None
    wig, nqc = qpp(scores, [len(t.split()) for t in qtexts])
    y, det = S["y_poor_all"], S["det_all"]
    return {
        "wig": auc_ci(y, -wig), "nqc": auc_ci(y, -nqc),
        "detector_minus_wig": auc_diff(y, det, y, -wig), "detector_minus_nqc": auc_diff(y, det, y, -nqc),
    }


def a7(S: dict, nrel: np.ndarray) -> dict:
    y, det = S["y_poor_all"], S["det_all"]
    strata = {"1": nrel == 1, "2-4": (nrel >= 2) & (nrel <= 4), ">=5": nrel >= 5}
    per = {k: auc_ci(y[m], det[m]) for k, m in strata.items()}
    w = [(per[k]["point"], per[k]["n"]) for k in per if per[k]["point"] is not None]
    X = np.column_stack([det, np.log1p(nrel)])
    coefs = [LogisticRegression(max_iter=1000).fit(X[i], y[i]).coef_[0][0] for i in _boot_idx(len(y)) if _ok(y[i])]
    return {
        "nrel_distribution": {"min": int(nrel.min()), "median": float(np.median(nrel)), "max": int(nrel.max())},
        "count_alone_auc": auc_ci(y, -nrel),
        "within_stratum": per,
        "weighted_within_stratum_auc": float(sum(p * n for p, n in w) / sum(n for _, n in w)),
        "detector_coef_controlling_log_nrel": {"point": float(LogisticRegression(max_iter=1000).fit(X, y).coef_[0][0]),
                                               **dict(zip(("ci_lo", "ci_hi"), _ci(coefs)))},
    }


def nrel_for(dataset: str, qids: list[str]) -> np.ndarray:
    qrels = load_qrels(dataset)
    return np.array([sum(1 for v in qrels.get(q, {}).values() if v > 0) for q in qids])


def a8(T: dict) -> dict:
    vcv = _load("results/tripclick-tail-val_verifier_calibration.json")
    vc = _load("results/tripclick-tail_verifier_calibration.json")
    lr = LogisticRegression(max_iter=1000).fit(np.array(vcv["X"]), (np.array(vcv["y_delta_ndcg"]) < -EPS).astype(int))
    p = lr.predict_proba(np.array(vc["X"]))[:, 1]
    harm = (T["delta"] < -EPS).astype(int)
    pilot = _load("results/nq_frozen_test.json")["records"]
    pa = [r for r in pilot if r["delta_ndcg"] is not None]
    return {
        "tripclick_logistic_refit_minus_linear": auc_diff(harm, p, harm, -T["au_gain"]),
        "oracle_val": _load("results/oracle_val_robustness.json"),
        "oracle_test": {"tripclick": _load("results/tripclick-tail_oracle_and_selectivity_v2.json")["oracle_ceiling"]["oracle_static_vs_bm25"],
                        "nq": _load("results/nq_extension_only_result.json")["oracle_gain"]},
        "nq_pilot_n600": {"detection": auc_ci([r["y_poor"] for r in pilot], [r["poor_proba"] for r in pilot]),
                          "action_utility": auc_ci([int(r["delta_ndcg"] < -EPS) for r in pa], [-r["predicted_delta"] for r in pa])},
    }


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items() if k != "per_query"}
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return o


def main() -> None:
    T, N = tripclick_bm25(), nq_bm25()
    TD = dense("results/dense_retrieval_full_scale_raw.json", "tripclick_dense")
    ND = dense("results/nq_dense_retrieval_raw.json", "nq_dense")
    out: dict = {"precommit": "configs/revision_analyses_precommit.json", "settings": {}}
    for S in (T, N, TD, ND):
        out["settings"][S["name"]] = {"A1": a1(S), "A2_A3": a2_a3(S), "A5": a5(S),
                                      "post_hoc_untied_not_precommitted": post_hoc_untied(S)}
        print(f"{S['name']}: A1/A2/A3/A5 done", flush=True)

    out["A4"] = {"tripclick_bm25": a4_tripclick(T), "nq_bm25": a4_nq(N)}
    print("A4 done", flush=True)

    tq = load_queries("tripclick-tail")
    ck = T["ck"]
    t_scores = [list(ck["baseline_run"][q].values()) for q in T["test_qids"]]
    nq_queries = load_queries("nq")
    nq_qids = [r["qid"] for r in N["recs"]]
    nq_scores = None
    if os.path.exists("results/nq_bm25_score_lists.json"):
        sl = _load("results/nq_bm25_score_lists.json")
        if all(q in sl for q in nq_qids):
            nq_scores = [sl[q] for q in nq_qids]
    out["A6"] = {"tripclick_bm25": a6(T, t_scores, [tq[q] for q in T["test_qids"]]),
                 "nq_bm25": a6(N, nq_scores, [nq_queries[q] for q in nq_qids])}
    print("A6 done", flush=True)

    out["A7"] = {"tripclick_bm25": a7(T, nrel_for("tripclick-tail", T["test_qids"])),
                 "nq_bm25": a7(N, nrel_for("nq", nq_qids))}
    print("A7 done", flush=True)

    out["A8"] = a8(T)
    with open("results/revision_analyses.json", "w") as f:
        json.dump(_clean(out), f, indent=2)
    print("saved to results/revision_analyses.json")


if __name__ == "__main__":
    main()
