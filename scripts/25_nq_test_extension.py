#!/usr/bin/env python3
"""Second, pre-committed frozen-test touch (configs/nq_test_extension_precommit.json,
written and committed BEFORE this script ever ran, before any of these
queries' data was seen): extends scripts/23's n=600 frozen NQ test with
ALL 2,052 remaining, never-touched NQ queries (the full 3,452-query pool
minus configs/nq_split.json's val[800] and test[600]).

Motivation: review feedback asked for a statistically stronger NQ
replication. The original n=600 result left action-utility AUC's CI
straddling chance ([0.442,0.602]) -- too imprecise to say whether the
gap versus failure-detection AUC is real or just underpowered. More
frozen-test queries increases precision without touching anything
upstream: same frozen retrieval config, same frozen verifier, same
frozen failure detector, same accept threshold as scripts/23 -- nothing
tuned, nothing re-selected. This is not a second bite at model
selection; it is the same frozen pipeline evaluated on more never-seen
data, pre-committed as a single all-or-nothing run (no early stopping,
no peek-then-decide) to avoid any appearance of favorable-result fishing.

scripts/26 pools this extension's records with scripts/23's original 600
into the final n=2,652 result reported in the paper.

Usage:
  PYTHONPATH=. python3 scripts/25_nq_test_extension.py [n_queries] [--resume]
  (n_queries defaults to all 2052 extension queries; pass a small number for a smoke test)
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
SEED = 42

bm25_index.TITLE_WEIGHT = FROZEN["title_weight"]
bm25_index.BODY_WEIGHT = FROZEN["body_weight"]

CHECKPOINT_PATH = Path("results/nq_test_extension.json")


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


def extension_qids() -> list[str]:
    queries = load_queries("nq")
    qrels = load_qrels("nq")
    all_qids = sorted(q for q in queries if q in qrels)
    used = set(SPLIT["val"]) | set(SPLIT["test"])
    remaining = [q for q in all_qids if q not in used]
    return remaining


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    resume = "--resume" in sys.argv
    ext_qids = extension_qids()
    print(f"extension pool: {len(ext_qids)} never-touched queries", flush=True)
    n = int(args[0]) if args else len(ext_qids)
    ext_qids = ext_qids[:n]

    queries = load_queries("nq")
    qrels = load_qrels("nq")
    selector = ActionSelector()

    records: list[dict] = []
    start_index = 0
    if resume and CHECKPOINT_PATH.exists():
        saved = json.loads(CHECKPOINT_PATH.read_text())
        records = saved["records"]
        start_index = len(records)
        print(f"Resuming: {start_index}/{len(ext_qids)} already processed.", flush=True)

    t_start = time.time()
    for i, qid in enumerate(ext_qids[start_index:], start_index + 1):
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
        eta_min = rate * (len(ext_qids) - i) / 60
        print(
            f"  [{i}/{len(ext_qids)}] {qid} action={record['action']:<16} "
            f"orig={original_ndcg:.4f} delta={record['delta_ndcg']}  accepted={record['accepted']}  "
            f"({rate:.1f}s/q, eta={eta_min:.1f}min)",
            flush=True,
        )
        if i % 100 == 0:
            CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
            CHECKPOINT_PATH.write_text(json.dumps({"records": records}, indent=2))
            print(f"  [checkpoint saved, n={len(records)}]", flush=True)

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.write_text(json.dumps({"records": records}, indent=2))
    print(f"\nn={len(records)} saved to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
