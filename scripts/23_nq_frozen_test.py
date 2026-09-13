#!/usr/bin/env python3
"""Second-corpus replication (RQ1 only), FROZEN TEST touch: the single,
single-touch evaluation this whole NQ side-study exists to produce.
Mirrors configs/experimental_protocol.yaml's rq1_test_eval /
oracle_ceiling_analysis stages, generalized to NQ.

Frozen inputs (nothing below may be changed based on what this script
prints -- see configs/nq_protocol_frozen.json and
configs/nq_verifier_frozen.json, both written before this script ever
touched test data):
  - retrieval config: porter stemming, title_weight=50 (data/nq_bm25.sqlite3)
  - action-utility verifier: models/nq_verifier_val_v1.joblib
  - failure detector: models/nq_failure_detector_val_v1.joblib
  - accept threshold: predicted delta_ndcg >= 0 (see rationale below)

For each of the 600 frozen test queries: BM25 retrieve -> ActionSelector
picks one operator (or NO_EXPAND) -> if an expansion action, apply it
once (no reranker, matching val) -> compute the candidate ranking's true
NDCG@10 delta -> the frozen verifier scores the episode's 10 delta
features; ACCEPT (use candidate ranking) iff predicted delta_ndcg >= 0,
else REJECT (keep original ranking). This defines "Static QE" for NQ,
methodologically parallel to TripClick's own verifier-gated Static QE
(Table III).

Accept-threshold rationale: TripClick's RecoveryThresholds.accept was
calibrated via an ad hoc median/25th-percentile convention over the
verifier's own raw score distribution (no natural zero point on that
scale). NQ's verifier is a LinearRegression *directly* predicting
delta_ndcg -- a threshold of 0.0 has a natural, principled meaning
("accept iff predicted to help") and needs no empirical recalibration.
Simpler and more defensible than reproducing TripClick's percentile
exercise, appropriate given this replication's confirmatory scope.

Reports the five numbers the review's minimal protocol asked for:
  1. Static QE gain: mean(accepted-ranking NDCG@10) - mean(original NDCG@10),
     paired bootstrap CI vs. BM25-only.
  2. Oracle gain: retrospective ceiling (accept only if truly helps, using
     test qrels post hoc) -- an upper bound, not a deployable method.
  3. Failure-detection AUC: frozen failure detector's predicted
     P(poor) vs. y_test_poor = (original_ndcg <= 1e-9).
  4. Action-utility AUC: frozen verifier's raw predicted delta_ndcg
     (sign-flipped, harmful=1 should score higher -- same convention as
     scripts/17), vs. true harmful = (delta_ndcg < 0).
  5. n and its 95% CIs (bootstrap, 10000 resamples, seed=42).

Usage:
  PYTHONPATH=. python3 scripts/23_nq_frozen_test.py [n_queries] [--resume]
  (n_queries defaults to all 600 test queries; pass a small number for a smoke test)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score

from src.agent.actions import EXPANSION_ACTIONS
from src.agent.controller import ActionSelector
from src.agent.state import AgentState
from src.agent.steps import act_once, build_evidence, retrieve_and_observe_initial
from src.agent.verify import _features as verifier_features
from src.data import load_qrels, load_queries
from src.eval.metrics import evaluate
from src.llm_client import LLMUnavailableError
from src.retrieval import bm25_index

FROZEN = json.loads(open("configs/nq_protocol_frozen.json").read())
DB_PATH = FROZEN["db_path"]
SPLIT = json.loads(open("configs/nq_split.json").read())
VERIFIER_ARTIFACT = joblib.load("models/nq_verifier_val_v1.joblib")
FAILURE_ARTIFACT = joblib.load("models/nq_failure_detector_val_v1.joblib")
ACCEPT_THRESHOLD = 0.0
N_BOOTSTRAP = 10000
SEED = 42

bm25_index.TITLE_WEIGHT = FROZEN["title_weight"]
bm25_index.BODY_WEIGHT = FROZEN["body_weight"]

CHECKPOINT_PATH = Path("results/nq_frozen_test.json")


def _rank_to_scores(ranking: list[str]) -> dict[str, float]:
    return {d: 1.0 / (i + 1) for i, d in enumerate(ranking)}


def _ndcg10(qid: str, ranking: list[str], qrels: dict) -> float:
    return evaluate({qid: qrels[qid]}, {qid: _rank_to_scores(ranking)})[qid]["ndcg_cut_10"]


def _t0_features(o0) -> list[float]:
    r = o0.retrieval
    return [
        r.top_score, r.score_margin, r.score_entropy, r.query_term_coverage,
        r.mean_rare_term_idf, r.entity_overlap, r.result_coherence, r.ranking_stability,
    ]


def predict_delta(feats: dict[str, float]) -> float:
    names = VERIFIER_ARTIFACT["feature_names"]
    x = np.array([[feats[n] for n in names]])
    return float(VERIFIER_ARTIFACT["model"].predict(x)[0])


def predict_poor_proba(t0_feats: list[float]) -> float:
    x = np.array([t0_feats])
    return float(FAILURE_ARTIFACT["model"].predict_proba(x)[0, 1])


def paired_bootstrap_diff(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """point, ci_lo, ci_hi for mean(b) - mean(a), paired resampling."""
    point = float(b.mean() - a.mean())
    n = len(a)
    rng = np.random.RandomState(SEED)
    diffs = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        idx = rng.randint(0, n, n)
        diffs[i] = b[idx].mean() - a[idx].mean()
    return point, float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def auc_ci(label: np.ndarray, score: np.ndarray) -> tuple[float, float, float]:
    point = float(roc_auc_score(label, score))
    n = len(label)
    rng = np.random.RandomState(SEED)
    aucs = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.randint(0, n, n)
        h, s = label[idx], score[idx]
        if 0 < h.sum() < len(h):
            aucs.append(roc_auc_score(h, s))
    return point, float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    resume = "--resume" in sys.argv
    n = int(args[0]) if args else len(SPLIT["test"])
    test_qids = SPLIT["test"][:n]

    queries = load_queries("nq")
    qrels = load_qrels("nq")
    selector = ActionSelector()

    records: list[dict] = []
    start_index = 0
    if resume and CHECKPOINT_PATH.exists():
        saved = json.loads(CHECKPOINT_PATH.read_text())
        records = saved["records"]
        start_index = len(records)
        print(f"Resuming: {start_index}/{len(test_qids)} already processed.", flush=True)

    t_start = time.time()
    for i, qid in enumerate(test_qids[start_index:], start_index + 1):
        text = queries[qid]
        init = retrieve_and_observe_initial(text, DB_PATH, k=100)
        original_ndcg = _ndcg10(qid, init.r0_ids, qrels)
        t0_feats = _t0_features(init.o0)
        poor_proba = predict_poor_proba(t0_feats)
        y_poor = int(original_ndcg <= 1e-9)

        action = selector.select(text, init.o0)
        record = {
            "qid": qid, "original_ndcg": original_ndcg, "y_poor": y_poor,
            "poor_proba": poor_proba, "action": action.value,
            "candidate_ndcg": None, "delta_ndcg": None, "predicted_delta": None,
            "accepted": False, "accepted_ndcg": original_ndcg,
        }

        if action in EXPANSION_ACTIONS:
            state = AgentState(q0=text, q_t=text, R_t=init.r0_ids, O_t=init.o0, B_t=1)
            evidence = build_evidence(init.r0_ids, DB_PATH)
            try:
                result = act_once(action, state, init.r0, DB_PATH, evidence, k=100, use_reranker=False)
            except LLMUnavailableError:
                records.append(record)
                continue

            candidate_ndcg = _ndcg10(qid, result.candidate.R_t, qrels)
            delta_ndcg = candidate_ndcg - original_ndcg
            feats = verifier_features(init.o0, result.candidate.O_t)
            predicted_delta = predict_delta(feats)
            accepted = predicted_delta >= ACCEPT_THRESHOLD

            record.update({
                "candidate_ndcg": candidate_ndcg, "delta_ndcg": delta_ndcg,
                "predicted_delta": predicted_delta, "accepted": accepted,
                "accepted_ndcg": candidate_ndcg if accepted else original_ndcg,
            })

        records.append(record)
        elapsed = time.time() - t_start
        rate = elapsed / (i - start_index)
        eta_min = rate * (len(test_qids) - i) / 60
        print(
            f"  [{i}/{len(test_qids)}] {qid} action={record['action']:<16} "
            f"orig={original_ndcg:.4f} delta={record['delta_ndcg']}  accepted={record['accepted']}  "
            f"({rate:.1f}s/q, eta={eta_min:.1f}min)",
            flush=True,
        )
        if i % 50 == 0:
            CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
            CHECKPOINT_PATH.write_text(json.dumps({"records": records}, indent=2))
            print(f"  [checkpoint saved, n={len(records)}]", flush=True)

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.write_text(json.dumps({"records": records}, indent=2))
    print(f"\nn={len(records)} saved to {CHECKPOINT_PATH}")

    if len(records) < 20:
        print("\n(smoke test -- too few records for meaningful summary stats)")
        return

    original = np.array([r["original_ndcg"] for r in records])
    static_qe = np.array([r["accepted_ndcg"] for r in records])
    oracle = np.array([max(r["original_ndcg"], r["candidate_ndcg"] or r["original_ndcg"]) for r in records])

    static_point, static_lo, static_hi = paired_bootstrap_diff(original, static_qe)
    oracle_point, oracle_lo, oracle_hi = paired_bootstrap_diff(original, oracle)
    print(f"\nStatic QE gain: {static_point:+.4f}  95% CI=[{static_lo:+.4f},{static_hi:+.4f}]")
    print(f"Oracle gain:    {oracle_point:+.4f}  95% CI=[{oracle_lo:+.4f},{oracle_hi:+.4f}]")

    y_poor = np.array([r["y_poor"] for r in records])
    poor_proba = np.array([r["poor_proba"] for r in records])
    if 0 < y_poor.sum() < len(y_poor):
        fpoint, flo, fhi = auc_ci(y_poor, poor_proba)
        print(f"Failure-detection AUC: {fpoint:.4f}  95% CI=[{flo:.4f},{fhi:.4f}]  n={len(y_poor)}")

    acted = [r for r in records if r["delta_ndcg"] is not None]
    if acted:
        harmful = np.array([1 if r["delta_ndcg"] < -1e-9 else 0 for r in acted])
        score = -np.array([r["predicted_delta"] for r in acted])
        if 0 < harmful.sum() < len(harmful):
            apoint, alo, ahi = auc_ci(harmful, score)
            print(f"Action-utility AUC: {apoint:.4f}  95% CI=[{alo:.4f},{ahi:.4f}]  n={len(harmful)}")
        print(f"(n acted on: {len(acted)}/{len(records)}, {100*len(acted)/len(records):.1f}%)")


if __name__ == "__main__":
    main()
