#!/usr/bin/env python3
"""Second-corpus replication (RQ1 only), val-set calibration step: for
each NQ val query, run one Static QE episode (no reranker -- excluded by
the review's scoped-down protocol) and collect the raw (features, true
NDCG@10 delta) samples needed to fit BOTH detectors, mirroring
scripts/08_calibrate_verifier.py (action-utility verifier) and the
failure-detector recipe in scripts/17_auc_confidence_intervals.py
(T0-feature LogisticRegression on y_poor = nDCG@10 <= 1e-9), generalized
to a fresh corpus.

Does NOT call src/agent/verify.py's verify() for anything meaningful --
that function is hardcoded to models/verifier_tripclick_tail_val_v1.joblib
(TripClick-fit) and must not silently score NQ episodes. act_once() still
calls it internally as a side effect (harmless, ignored here); this
script fits its OWN NQ-specific models from the raw features instead.

Reuses the frozen retrieval config from configs/nq_protocol_frozen.json
(porter stemming, title_weight=50, data/nq_bm25.sqlite3) and the frozen
val/test split from configs/nq_split.json (val=800, seed=42).

Checkpoints every 50 queries (same pattern as scripts/08) so a killed run
resumes with --resume instead of restarting.

Usage:
  PYTHONPATH=. python3 scripts/22_nq_val_calibration.py [n_queries] [--resume]
  (n_queries defaults to all 800 val queries; pass a small number for a smoke test)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

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

# bm25_index.search() reads these as module-level globals at call time, not
# import time -- reassigning here overrides TripClick's frozen TITLE_WEIGHT=10
# (configs/experimental_protocol.yaml) with NQ's own frozen value for the
# lifetime of this process only. TripClick's separately-run scripts/frozen
# artifacts are untouched since they never share a process with this script.
bm25_index.TITLE_WEIGHT = FROZEN["title_weight"]
bm25_index.BODY_WEIGHT = FROZEN["body_weight"]

T0_FEATURE_NAMES = [
    "top_score",
    "score_margin",
    "score_entropy",
    "query_term_coverage",
    "mean_rare_term_idf",
    "entity_overlap",
    "result_coherence",
    "ranking_stability",
]

VERIFIER_FEATURE_NAMES = [
    "score_margin_gain",
    "coverage_gain",
    "coherence_gain",
    "stability",
    "drift",
    "disagreement",
    "topk_overlap",
    "reranker_top_score",
    "reranker_score_margin",
    "query_length_ratio",
]

CHECKPOINT_PATH = Path("results/nq_val_calibration.json")


def _rank_to_scores(ranking: list[str]) -> dict[str, float]:
    return {d: 1.0 / (i + 1) for i, d in enumerate(ranking)}


def _ndcg10(qid: str, ranking: list[str], qrels: dict) -> float:
    return evaluate({qid: qrels[qid]}, {qid: _rank_to_scores(ranking)})[qid]["ndcg_cut_10"]


def _t0_features(o0) -> list[float]:
    r = o0.retrieval
    return [
        r.top_score,
        r.score_margin,
        r.score_entropy,
        r.query_term_coverage,
        r.mean_rare_term_idf,
        r.entity_overlap,
        r.result_coherence,
        r.ranking_stability,
    ]


def _save(val_qids, X_t0, y_ndcg, X_verifier, y_delta_ndcg, last_index, skipped_no_expand, skipped_llm_error):
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.write_text(
        json.dumps(
            {
                "dataset": "nq",
                "db_path": DB_PATH,
                "val_qids": val_qids,
                "t0_feature_names": T0_FEATURE_NAMES,
                "X_t0": X_t0,
                "y_ndcg": y_ndcg,
                "verifier_feature_names": VERIFIER_FEATURE_NAMES,
                "X_verifier": X_verifier,
                "y_delta_ndcg": y_delta_ndcg,
                "last_index": last_index,
                "skipped_no_expand": skipped_no_expand,
                "skipped_llm_error": skipped_llm_error,
            },
            indent=2,
        )
    )


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    resume = "--resume" in sys.argv
    n = int(args[0]) if args else len(SPLIT["val"])
    val_qids = SPLIT["val"][:n]

    queries = load_queries("nq")
    qrels = load_qrels("nq")

    selector = ActionSelector()
    X_t0, y_ndcg, X_verifier, y_delta_ndcg = [], [], [], []
    processed_qids: list[str] = []
    skipped_no_expand = 0
    skipped_llm_error = 0
    start_index = 0

    if resume and CHECKPOINT_PATH.exists():
        saved = json.loads(CHECKPOINT_PATH.read_text())
        X_t0, y_ndcg = saved["X_t0"], saved["y_ndcg"]
        X_verifier, y_delta_ndcg = saved["X_verifier"], saved["y_delta_ndcg"]
        processed_qids = saved["val_qids"]
        start_index = saved.get("last_index", 0)
        skipped_no_expand = saved.get("skipped_no_expand", 0)
        skipped_llm_error = saved.get("skipped_llm_error", 0)
        print(f"Resuming: {len(X_verifier)} samples, {start_index}/{len(val_qids)} processed.", flush=True)

    t_start = time.time()
    for i, qid in enumerate(val_qids[start_index:], start_index + 1):
        text = queries[qid]
        init = retrieve_and_observe_initial(text, DB_PATH, k=100)
        original_ndcg = _ndcg10(qid, init.r0_ids, qrels)

        X_t0.append(_t0_features(init.o0))
        y_ndcg.append(original_ndcg)
        processed_qids.append(qid)

        action = selector.select(text, init.o0)
        if action not in EXPANSION_ACTIONS:
            skipped_no_expand += 1
            if i % 10 == 0:
                print(f"  [{i}/{len(val_qids)}] {qid} NO_EXPAND", flush=True)
            continue

        state = AgentState(q0=text, q_t=text, R_t=init.r0_ids, O_t=init.o0, B_t=1)
        evidence = build_evidence(init.r0_ids, DB_PATH)
        try:
            result = act_once(action, state, init.r0, DB_PATH, evidence, k=100, use_reranker=False)
        except LLMUnavailableError:
            skipped_llm_error += 1
            continue

        candidate_ndcg = _ndcg10(qid, result.candidate.R_t, qrels)
        delta_ndcg = candidate_ndcg - original_ndcg
        feats = verifier_features(init.o0, result.candidate.O_t)
        X_verifier.append([feats[name] for name in VERIFIER_FEATURE_NAMES])
        y_delta_ndcg.append(delta_ndcg)

        elapsed = time.time() - t_start
        rate = elapsed / (i - start_index)
        eta_min = rate * (len(val_qids) - i) / 60
        print(
            f"  [{i}/{len(val_qids)}] {qid} action={action.value:<16} orig_ndcg={original_ndcg:.4f} "
            f"delta_ndcg={delta_ndcg:+.4f}  ({rate:.1f}s/q, eta={eta_min:.1f}min)",
            flush=True,
        )

        if i % 50 == 0:
            _save(processed_qids, X_t0, y_ndcg, X_verifier, y_delta_ndcg, i, skipped_no_expand, skipped_llm_error)
            print(f"  [checkpoint saved, last_index={i}]", flush=True)

    _save(processed_qids, X_t0, y_ndcg, X_verifier, y_delta_ndcg, len(val_qids), skipped_no_expand, skipped_llm_error)
    print(
        f"\ndone: n_t0={len(X_t0)}  n_verifier={len(X_verifier)}  "
        f"(skipped: {skipped_no_expand} NO_EXPAND, {skipped_llm_error} LLM errors)",
        flush=True,
    )
    print(f"saved to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
