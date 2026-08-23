#!/usr/bin/env python3
"""The critical comparison (README): observable agent vs. the strong
failure-conditioned static QE comparator, using the same retriever,
grounding, generator, and fusion mechanism -- isolating whether the
closed-loop recovery behavior (ROLLBACK/REPLAN) adds anything beyond a
single-shot "diagnose once, expand once, verify once" pipeline.

Runs BM25-only, static QE, and the full agent over the same query set and
reports NDCG/Recall/MRR for all three, cost (LLM calls/query, latency),
and paired bootstrap significance on the agent-vs-static NDCG@10 delta --
the strongest-hypothesis test isn't "agent beats BM25", it's "agent moves
the quality/harm/cost Pareto frontier past the static comparator".

Usage: python scripts/06_agent_vs_static.py [dataset] [n_queries]
"""
from __future__ import annotations

import sys

from src.agent.loop import run_episode
from src.agent.static_baseline import run_static_episode
from src.data import load_qrels, load_queries
from src.eval.bootstrap import paired_bootstrap
from src.eval.metrics import evaluate, helped_unchanged_harmed, mean_metrics
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

    baseline_run, static_run, agent_run = {}, {}, {}
    static_traces, agent_traces = [], []

    for qid in query_ids:
        text = queries[qid]
        baseline_run[qid] = dict(search(text, db_path, k=100))

        static_ranking, static_trace = run_static_episode(qid, text, db_path, dataset_slice=dataset)
        static_run[qid] = {doc_id: 1.0 / (rank + 1) for rank, doc_id in enumerate(static_ranking)}
        static_traces.append(static_trace)

        agent_ranking, agent_trace = run_episode(qid, text, db_path, dataset_slice=dataset)
        agent_run[qid] = {doc_id: 1.0 / (rank + 1) for rank, doc_id in enumerate(agent_ranking)}
        agent_traces.append(agent_trace)

        print(
            f"{qid:>8}  static={static_trace.controller_decision:<8} "
            f"agent={agent_trace.controller_decision:<8}"
        )

    q = {qid: qrels[qid] for qid in query_ids}
    baseline_eval = evaluate(q, baseline_run)
    static_eval = evaluate(q, static_run)
    agent_eval = evaluate(q, agent_run)

    print(f"\nn_queries={len(query_ids)}")
    print(f"BM25 only:  {mean_metrics(baseline_eval)}")
    print(f"Static QE:  {mean_metrics(static_eval)}")
    print(f"Agent:      {mean_metrics(agent_eval)}")

    print(f"\nagent vs static, helped/unchanged/harmed: {helped_unchanged_harmed(static_eval, agent_eval)}")
    print(f"agent vs BM25,   helped/unchanged/harmed: {helped_unchanged_harmed(baseline_eval, agent_eval)}")
    print(f"static vs BM25,  helped/unchanged/harmed: {helped_unchanged_harmed(baseline_eval, static_eval)}")

    static_ndcg = [static_eval[qid]["ndcg_cut_10"] for qid in query_ids]
    agent_ndcg = [agent_eval[qid]["ndcg_cut_10"] for qid in query_ids]
    bootstrap = paired_bootstrap(static_ndcg, agent_ndcg)
    print(
        f"\npaired bootstrap, agent - static NDCG@10: mean_diff={bootstrap['mean_diff']:.4f} "
        f"95% CI [{bootstrap['ci_lo']:.4f}, {bootstrap['ci_hi']:.4f}]  significant={bootstrap['significant']}"
    )

    static_calls = sum(t.llm_calls for t in static_traces) / len(static_traces)
    agent_calls = sum(t.llm_calls for t in agent_traces) / len(agent_traces)
    static_latency = sum(t.episode_latency_ms for t in static_traces) / len(static_traces)
    agent_latency = sum(t.episode_latency_ms for t in agent_traces) / len(agent_traces)
    print(
        f"\ncost -- static: {static_calls:.2f} llm_calls/query, {static_latency:.0f}ms/query mean\n"
        f"        agent:  {agent_calls:.2f} llm_calls/query, {agent_latency:.0f}ms/query mean"
    )

    if len(query_ids) < 50:
        print(
            "\nNOTE: this is a small dev sample -- the bootstrap CI above is illustrative, "
            "not a claim of statistical significance for the paper."
        )


if __name__ == "__main__":
    main()
