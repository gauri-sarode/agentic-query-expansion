#!/usr/bin/env python3
"""BM25-only retrieval floor over TripClick, reported separately per
HEAD/TORSO/TAIL test split (docs/milestones.md experiment matrix's "BM25"
row). Run after scripts/03_build_tripclick_index.sh.
"""
from __future__ import annotations

from src import tripclick
from src.eval.metrics import evaluate, mean_metrics
from src.retrieval.bm25_index import search

DB_PATH = "data/tripclick_bm25.sqlite3"


def run_slice(slice_: str) -> None:
    queries = tripclick.load_topics(slice_, "test")
    qrels = tripclick.load_qrels(slice_, "test", label_type="raw")
    run = {qid: dict(search(text, DB_PATH, k=100)) for qid, text in queries.items()}
    metrics = mean_metrics(evaluate(qrels, run))
    print(f"{slice_:>6}  n_queries={len(queries):5d}  {metrics}")


if __name__ == "__main__":
    for slice_ in ("head", "torso", "tail"):
        run_slice(slice_)
