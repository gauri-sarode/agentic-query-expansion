#!/usr/bin/env python3
"""Recomputes the 8 pre-action (t0) retrieval-telemetry features for
every query in the frozen NQ extension test set (results/nq_test_extension.json,
n=2,052), which were NOT persisted when that frozen touch ran (only the
frozen logistic detector's scalar poor_proba was kept, scripts/25).

This is a pure recomputation, not a new experiment: t0 features are a
deterministic function of BM25 retrieval on the ORIGINAL query only
(src/agent/steps.py's retrieve_and_observe_initial -> _t0_features),
computed before any expansion action and independent of it -- the same
frozen retrieval config already used throughout (configs/nq_protocol_frozen.json:
porter stemming, title_weight=50, body_weight=1, k=100, data/nq_bm25.sqlite3).
No LLM calls, no action selection, nothing stochastic. Needed to score a
non-linear (GBDT) detection-model alternative on this frozen test
population without re-running anything that could change y_poor or
poor_proba for the already-reported logistic-regression result.

Checkpointed every 100 queries (--resume to continue); each query costs
~9s here (dominated by repeated sqlite connects against the 4GB index,
not model inference), so the full n=2,052 run is multi-hour.

Usage: PYTHONPATH=. python3 scripts/38_nq_extension_t0_recompute.py [--resume]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from src.agent.steps import retrieve_and_observe_initial
from src.data import load_queries, load_qrels
from src.retrieval import bm25_index

FROZEN = json.loads(open("configs/nq_protocol_frozen.json").read())
DB_PATH = FROZEN["db_path"]
SPLIT = json.loads(open("configs/nq_split.json").read())

bm25_index.TITLE_WEIGHT = FROZEN["title_weight"]
bm25_index.BODY_WEIGHT = FROZEN["body_weight"]

CHECKPOINT_PATH = Path("results/nq_extension_t0_features.json")
_CHECKPOINT_EVERY = 100


def _t0_features(o0) -> list[float]:
    r = o0.retrieval
    return [
        r.top_score, r.score_margin, r.score_entropy, r.query_term_coverage,
        r.mean_rare_term_idf, r.entity_overlap, r.result_coherence, r.ranking_stability,
    ]


def extension_qids() -> list[str]:
    queries = load_queries("nq")
    qrels = load_qrels("nq")
    all_qids = sorted(q for q in queries if q in qrels)
    used = set(SPLIT["val"]) | set(SPLIT["test"])
    remaining = [q for q in all_qids if q not in used]
    return remaining


def main() -> None:
    resume = "--resume" in sys.argv
    ext = json.loads(open("results/nq_test_extension.json").read())["records"]
    ext_qids = extension_qids()
    assert [r["qid"] for r in ext] == ext_qids, "qid order mismatch vs. frozen extension records"
    print(f"n={len(ext_qids)} extension queries (order-verified against frozen records)", flush=True)

    queries = load_queries("nq")
    results: dict[str, list[float]] = {}
    start = 0
    if resume:
        try:
            saved = json.loads(open(CHECKPOINT_PATH).read())
            results = saved["t0_features"]
            start = saved["n_done"]
            print(f"resuming at {start}/{len(ext_qids)}", flush=True)
        except FileNotFoundError:
            print("--resume requested but no checkpoint found -- starting fresh.", flush=True)

    t0 = time.time()
    for i in range(start, len(ext_qids)):
        qid = ext_qids[i]
        init = retrieve_and_observe_initial(queries[qid], DB_PATH, k=100)
        results[qid] = _t0_features(init.o0)
        if (i + 1) % _CHECKPOINT_EVERY == 0:
            CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
            CHECKPOINT_PATH.write_text(json.dumps({"t0_features": results, "n_done": i + 1}, indent=2))
            elapsed = time.time() - t0
            rate = elapsed / (i + 1 - start)
            eta_min = rate * (len(ext_qids) - (i + 1)) / 60
            print(f"[{i + 1}/{len(ext_qids)}] ({rate:.1f}s/q, eta={eta_min:.1f}min)", flush=True)

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.write_text(json.dumps({"t0_features": results, "n_done": len(ext_qids)}, indent=2))
    print(f"\ndone: {len(results)}/{len(ext_qids)} saved to {CHECKPOINT_PATH}", flush=True)


if __name__ == "__main__":
    main()
