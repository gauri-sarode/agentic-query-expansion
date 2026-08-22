#!/usr/bin/env python3
"""End-to-end closed-loop agent run over a dataset slice: for each query,
runs the full OBSERVE -> DIAGNOSE -> ACT -> RETRIEVE -> OBSERVE -> VERIFY
-> {ACCEPT, ROLLBACK, REPLAN, STOP} episode, logs one trace per query to
the trace store, and reports agent NDCG@10/Recall@100/MRR against the
BM25-only floor for the same queries.

Usage: python scripts/05_run_agent_episode.py [dataset] [n_queries]
  dataset: nfcorpus (default) | tripclick-head | tripclick-torso | tripclick-tail
"""
from __future__ import annotations

import sys
from collections import Counter

from src.agent.loop import run_episode
from src.data import load_qrels, load_queries
from src.eval.metrics import evaluate, helped_unchanged_harmed, mean_metrics
from src.observability.store import JsonlTraceStore
from src.retrieval.bm25_index import search

_DB_PATHS = {
    "nfcorpus": "data/nfcorpus_bm25.sqlite3",
    "tripclick-head": "data/tripclick_bm25.sqlite3",
    "tripclick-torso": "data/tripclick_bm25.sqlite3",
    "tripclick-tail": "data/tripclick_bm25.sqlite3",
}


def main() -> None:
    dataset = sys.argv[1] if len(sys.argv) > 1 else "nfcorpus"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    db_path = _DB_PATHS[dataset]

    queries = load_queries(dataset)
    qrels = load_qrels(dataset)
    query_ids = [qid for qid in queries if qid in qrels][:n]

    store = JsonlTraceStore(f"results/{dataset}_agent_trace.jsonl")
    decisions = Counter()

    baseline_run, agent_run = {}, {}
    for qid in query_ids:
        text = queries[qid]
        baseline_run[qid] = dict(search(text, db_path, k=100))

        final_ranking, trace = run_episode(qid, text, db_path, dataset_slice=dataset)
        agent_run[qid] = {doc_id: 1.0 / (rank + 1) for rank, doc_id in enumerate(final_ranking)}
        store.append(trace)
        decisions[trace.controller_decision] += 1

        print(
            f"{qid:>8}  action={trace.planner_action:<16} decision={trace.controller_decision:<10} "
            f"verifier={trace.verifier_score}"
        )

    q_baseline = {qid: qrels[qid] for qid in query_ids}
    baseline_metrics = mean_metrics(evaluate(q_baseline, baseline_run))
    agent_metrics = mean_metrics(evaluate(q_baseline, agent_run))
    hh = helped_unchanged_harmed(
        evaluate(q_baseline, baseline_run), evaluate(q_baseline, agent_run)
    )

    print(f"\ncontroller decisions: {dict(decisions)}")
    print(f"BM25 only:      {baseline_metrics}")
    print(f"Observable agent: {agent_metrics}")
    print(f"helped/unchanged/harmed: {hh}")
    print(f"\ntrace written to results/{dataset}_agent_trace.jsonl")


if __name__ == "__main__":
    main()
