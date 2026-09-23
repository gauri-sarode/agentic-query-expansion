#!/usr/bin/env python3
"""Reviewer request: the paper's central finding is that failure
detection (AUC 0.649) is much easier than action-utility prediction
(AUC 0.552) -- but the only selective-expansion baseline compared
against BM25/Static QE/Agent is the closed-loop agent itself. A much
simpler controller, motivated directly by that finding, is missing:
route on the failure detector ALONE, before any action, with no
post-action verifier, no rollback, no agent loop at all --

    if P(failure | pre-action telemetry) > tau: Static QE
    else: BM25

This is the cheapest possible test of whether failure detection by
itself is a useful control signal, isolated from the weaker
action-utility signal RQ1 already showed is the bottleneck.

No new touches of anything: this re-combines three ALREADY-FROZEN,
already-reported artifacts under a new, pre-specified (not
outcome-tuned) selection rule -- the RQ2 checkpoint's per-query BM25
and Static QE rankings (results/tripclick-tail_agent_vs_static_checkpoint.json),
and the RQ1 failure detector's exact fit (LogisticRegression on TAIL
val's 8 retrieval features, scripts/17's methodology, reused verbatim
here). The threshold tau is the median predicted failure probability
on VALIDATION (mirroring Section III-D's existing accept/rollback
threshold convention: chosen on validation, before touching this
test-set combination) -- never swept or chosen to make this comparison
look better.

Usage: PYTHONPATH=. python3 scripts/31_failure_selective_qe.py
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.data import load_qrels
from src.eval.bootstrap import paired_bootstrap
from src.eval.metrics import evaluate

N_BOOTSTRAP = 10000
SEED = 42


def main() -> None:
    ck = json.loads(open("results/tripclick-tail_agent_vs_static_checkpoint.json").read())
    baseline_run, static_run, agent_run = ck["baseline_run"], ck["static_run"], ck["agent_run"]
    query_ids = sorted(baseline_run.keys())
    assert set(query_ids) == set(static_run.keys()) == set(agent_run.keys())

    fd = json.loads(open("results/tripclick-tail_failure_detection.json").read())
    assert set(fd["test_qids"]) == set(query_ids), "population mismatch vs RQ2 checkpoint"

    qrels = load_qrels("tripclick-tail")
    baseline_eval = evaluate(qrels, baseline_run)
    static_eval = evaluate(qrels, static_run)
    agent_eval = evaluate(qrels, agent_run)

    b = np.array([baseline_eval[q]["ndcg_cut_10"] for q in query_ids])
    s = np.array([static_eval[q]["ndcg_cut_10"] for q in query_ids])
    a = np.array([agent_eval[q]["ndcg_cut_10"] for q in query_ids])

    # Refit the failure detector exactly as scripts/17/28 do: val-fit, test-scored.
    X_val, y_val_ndcg = np.array(fd["X_val"]), np.array(fd["y_val_ndcg"])
    y_val_poor = (y_val_ndcg <= 1e-9).astype(int)
    X_test = np.array(fd["X_test"])
    failure_model = LogisticRegression(max_iter=1000).fit(X_val, y_val_poor)

    val_proba = failure_model.predict_proba(X_val)[:, 1]
    tau = float(np.median(val_proba))  # pre-specified on validation, mirrors Section III-D
    print(f"threshold tau (median predicted failure prob. on validation): {tau:.4f}")

    # Align test_qids -> X_test row order with query_ids -> b/s/a order (by qid, not position).
    qid_to_row = {q: i for i, q in enumerate(fd["test_qids"])}
    test_proba = failure_model.predict_proba(X_test)[:, 1]
    proba_aligned = np.array([test_proba[qid_to_row[q]] for q in query_ids])

    route_to_qe = proba_aligned > tau
    print(f"routed to Static QE: {route_to_qe.sum()}/{len(query_ids)} ({100 * route_to_qe.mean():.1f}%)")

    hybrid = np.where(route_to_qe, s, b)

    def per_query(measure: str) -> tuple[np.ndarray, np.ndarray]:
        bb = np.array([baseline_eval[q][measure] for q in query_ids])
        ss = np.array([static_eval[q][measure] for q in query_ids])
        return bb, ss

    r_b, r_s = per_query("recall_100")
    m_b, m_s = per_query("recip_rank")
    hybrid_recall = np.where(route_to_qe, r_s, r_b)
    hybrid_mrr = np.where(route_to_qe, m_s, m_b)
    print(f"Failure-selective QE Recall@100={hybrid_recall.mean():.4f}  MRR={hybrid_mrr.mean():.4f}")

    def report(name: str, treatment: np.ndarray) -> None:
        r = paired_bootstrap(b.tolist(), treatment.tolist())
        print(f"{name}: mean nDCG@10={treatment.mean():.4f}  gain vs BM25={r['mean_diff']:+.4f}  "
              f"95% CI=[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]  significant={r['significant']}  n={len(treatment)}")

    print(f"\nBM25 only: mean nDCG@10={b.mean():.4f}")
    report("Unconditional Static QE", s)
    report("Failure-selective QE", hybrid)
    report("Agent", a)
    report("Oracle-static", np.maximum(b, s))

    # Failure-selective QE vs unconditional Static QE, and vs Agent, directly.
    fs_vs_static = paired_bootstrap(s.tolist(), hybrid.tolist())
    fs_vs_agent = paired_bootstrap(a.tolist(), hybrid.tolist())
    print(f"\nFailure-selective QE vs unconditional Static QE: "
          f"{fs_vs_static['mean_diff']:+.4f} [{fs_vs_static['ci_lo']:+.4f},{fs_vs_static['ci_hi']:+.4f}] "
          f"significant={fs_vs_static['significant']}")
    print(f"Failure-selective QE vs Agent: "
          f"{fs_vs_agent['mean_diff']:+.4f} [{fs_vs_agent['ci_lo']:+.4f},{fs_vs_agent['ci_hi']:+.4f}] "
          f"significant={fs_vs_agent['significant']}")

    out = {
        "tau": tau,
        "n_routed_to_qe": int(route_to_qe.sum()),
        "n_total": len(query_ids),
        "bm25_mean": float(b.mean()),
        "static_qe_mean": float(s.mean()),
        "failure_selective_qe_mean": float(hybrid.mean()),
        "failure_selective_qe_recall100": float(hybrid_recall.mean()),
        "failure_selective_qe_mrr": float(hybrid_mrr.mean()),
        "agent_mean": float(a.mean()),
        "oracle_static_mean": float(np.maximum(b, s).mean()),
    }
    with open("results/failure_selective_qe.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nsaved to results/failure_selective_qe.json")


if __name__ == "__main__":
    main()
