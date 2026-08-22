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

import numpy as np

from src.agent.loop import run_episode
from src.data import load_qrels, load_queries
from src.eval.metrics import evaluate, helped_unchanged_harmed, mean_metrics
from src.eval.slo import AgentSearchSLO
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
    traces = []

    baseline_run, agent_run = {}, {}
    for qid in query_ids:
        text = queries[qid]
        baseline_run[qid] = dict(search(text, db_path, k=100))

        final_ranking, trace = run_episode(qid, text, db_path, dataset_slice=dataset)
        agent_run[qid] = {doc_id: 1.0 / (rank + 1) for rank, doc_id in enumerate(final_ranking)}
        store.append(trace)
        traces.append(trace)
        decisions[trace.controller_decision] += 1

        print(
            f"{qid:>8}  action={trace.planner_action:<16} decision={trace.controller_decision:<10} "
            f"verifier={trace.verifier_score}"
        )

    q_baseline = {qid: qrels[qid] for qid in query_ids}
    baseline_eval = evaluate(q_baseline, baseline_run)
    agent_eval = evaluate(q_baseline, agent_run)
    baseline_metrics = mean_metrics(baseline_eval)
    agent_metrics = mean_metrics(agent_eval)
    hh = helped_unchanged_harmed(baseline_eval, agent_eval)

    # Agent Search SLO calibration inputs (docs/milestones.md): measure
    # these on a representative dev sample and freeze h/c/L before
    # main-test evaluation, rather than optimizing a single metric.
    harm_rate = hh["harmed"]
    mean_llm_calls = sum(t.llm_calls for t in traces) / len(traces)
    latencies = [t.episode_latency_ms for t in traces]
    p95_latency = float(np.percentile(latencies, 95))

    print(f"\ncontroller decisions: {dict(decisions)}")
    print(f"BM25 only:      {baseline_metrics}")
    print(f"Observable agent: {agent_metrics}")
    print(f"helped/unchanged/harmed: {hh}")
    print(
        f"\nAgent Search SLO inputs (n={len(traces)}): "
        f"harm_rate={harm_rate:.3f}  mean_llm_calls/query={mean_llm_calls:.2f}  "
        f"p95_latency_ms={p95_latency:.1f}"
    )
    if len(traces) < 100:
        print(
            "NOTE: docs/milestones.md calls for freezing h/c/L against ~100 "
            f"representative episodes before main-test eval -- this run used only {len(traces)}."
        )
    else:
        slo = AgentSearchSLO(
            max_harm_rate=harm_rate, max_expected_llm_calls=mean_llm_calls, max_p95_latency_ms=p95_latency
        )
        print(f"Candidate frozen SLO (this run's own measurements as the bar): {slo}")
    print(f"\ntrace written to results/{dataset}_agent_trace.jsonl")


if __name__ == "__main__":
    main()
