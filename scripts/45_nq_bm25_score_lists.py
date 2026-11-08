#!/usr/bin/env python3
"""Revision analysis A6 (configs/revision_analyses_precommit.json): the
unsupervised QPP baselines (WIG/NQC-style) need each query's first-stage
BM25 score list, which the NQ extension run (scripts/25) never stored.
Re-runs the frozen NQ BM25 config (configs/nq_protocol_frozen.json) over
the 2,052-query confirmatory set -- retrieval only, no generator, no
qrels -- and stores the top-100 scores per query.

Resumable. Usage: PYTHONPATH=. python3 scripts/45_nq_bm25_score_lists.py
"""
from __future__ import annotations

import json
import os

from src.data import load_queries
from src.retrieval import bm25_index

FROZEN = json.loads(open("configs/nq_protocol_frozen.json").read())
bm25_index.TITLE_WEIGHT = FROZEN["title_weight"]
bm25_index.BODY_WEIGHT = FROZEN["body_weight"]
OUT = "results/nq_bm25_score_lists.json"


def main() -> None:
    qids = [r["qid"] for r in json.loads(open("results/nq_test_extension.json").read())["records"]]
    queries = load_queries("nq")
    done = json.loads(open(OUT).read()) if os.path.exists(OUT) else {}
    for i, qid in enumerate(qids, 1):
        if qid in done:
            continue
        done[qid] = [s for _, s in bm25_index.search(queries[qid], FROZEN["db_path"], k=100)]
        if i % 100 == 0:
            json.dump(done, open(OUT, "w"))
            print(f"{i}/{len(qids)}", flush=True)
    json.dump(done, open(OUT, "w"))
    print(f"done: {len(done)} queries -> {OUT}")


if __name__ == "__main__":
    main()
