#!/usr/bin/env python3
"""Isolated latency benchmark for static vs. agent, designed to avoid the
two confounds found in scripts/06_agent_vs_static.py's cost numbers:

1. Prompt-adjacency KV-cache reuse (running static then agent on the
   *same* prompt back-to-back let one pipeline's call ride the other's
   warm cache) -- fixed by running each pipeline as a full batch over all
   queries, not interleaved per query.
2. Any monotonic drift over a long single-process run (thermal
   throttling, OS warmup) aliasing with pipeline identity if one
   pipeline always runs first -- fixed by alternating which pipeline's
   batch goes first across repetitions, and discarding one warm-up call
   at the start of each batch.

Samples are paired by (query, repetition) regardless of which pipeline
ran first in that repetition, so paired_bootstrap gives a clean
significance read on the latency delta.

Usage: python scripts/07_latency_benchmark.py [dataset] [n_queries] [n_repeats]
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import numpy as np

from src.agent.loop import run_episode
from src.agent.static_baseline import run_static_episode
from src.data import load_qrels, load_queries
from src.eval.bootstrap import paired_bootstrap

_DB_PATHS = {
    "nfcorpus": "data/nfcorpus_bm25.sqlite3",
    "tripclick-head": "data/tripclick_bm25.sqlite3",
    "tripclick-torso": "data/tripclick_bm25.sqlite3",
    "tripclick-tail": "data/tripclick_bm25.sqlite3",
}


def _run_batch(pipeline_fn, query_ids, queries, db_path, dataset) -> dict[str, float]:
    if query_ids:
        # Discard one warm-up call so steady-state (not cold-start) latency is measured.
        pipeline_fn(query_ids[0], queries[query_ids[0]], db_path, dataset_slice=dataset)
    latencies = {}
    for qid in query_ids:
        _, trace = pipeline_fn(qid, queries[qid], db_path, dataset_slice=dataset)
        latencies[qid] = trace.episode_latency_ms
    return latencies


def main() -> None:
    dataset = sys.argv[1] if len(sys.argv) > 1 else "nfcorpus"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    repeats = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    db_path = _DB_PATHS[dataset]

    queries = load_queries(dataset)
    qrels = load_qrels(dataset)
    query_ids = [qid for qid in queries if qid in qrels][:n]

    static_samples: list[float] = []
    agent_samples: list[float] = []

    for rep in range(repeats):
        static_first = rep % 2 == 0
        order = ["static", "agent"] if static_first else ["agent", "static"]
        print(f"repetition {rep + 1}/{repeats}, order={order}")

        results: dict[str, dict[str, float]] = {}
        for pipeline in order:
            fn = run_static_episode if pipeline == "static" else run_episode
            results[pipeline] = _run_batch(fn, query_ids, queries, db_path, dataset)

        for qid in query_ids:
            static_samples.append(results["static"][qid])
            agent_samples.append(results["agent"][qid])

    def summarize(name: str, samples: list[float]) -> None:
        print(
            f"{name:>7}: median={statistics.median(samples):.0f}ms  mean={statistics.mean(samples):.0f}ms  "
            f"stdev={statistics.stdev(samples):.0f}ms  p95={np.percentile(samples, 95):.0f}ms  n={len(samples)}"
        )

    print(f"\nn_queries={len(query_ids)}  repeats={repeats}  total_samples_per_pipeline={len(static_samples)}")
    summarize("static", static_samples)
    summarize("agent", agent_samples)

    bootstrap = paired_bootstrap(static_samples, agent_samples)
    print(
        f"\npaired bootstrap, agent - static latency_ms: mean_diff={bootstrap['mean_diff']:.1f} "
        f"95% CI [{bootstrap['ci_lo']:.1f}, {bootstrap['ci_hi']:.1f}]  significant={bootstrap['significant']}"
    )

    out_path = Path(f"results/{dataset}_latency_benchmark.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "dataset": dataset,
                "n_queries": len(query_ids),
                "repeats": repeats,
                "static_samples_ms": static_samples,
                "agent_samples_ms": agent_samples,
            },
            indent=2,
        )
    )
    print(f"raw samples written to {out_path}")


if __name__ == "__main__":
    main()
