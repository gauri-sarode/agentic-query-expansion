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

Usage: python scripts/06_agent_vs_static.py [dataset] [n_queries] [--freeze-slo] [--resume]

--freeze-slo writes this run's agent harm rate / mean LLM calls / p95
latency into configs/default.yaml's slo: block (only when n_queries >= 100,
per the docs/milestones.md go/no-go calibration step -- smaller samples
refuse to freeze rather than silently locking in a noisy estimate).

--resume continues from results/{dataset}_agent_vs_static_checkpoint.json
if present, rather than reprocessing from scratch -- a full two-pass
TripClick-scale run can take a day+, and this is checkpointed every 50
queries per pass so an interruption doesn't discard everything already
computed.
"""
from __future__ import annotations

import dataclasses
import json
import re
import sys
from pathlib import Path

from src.agent.loop import run_episode
from src.agent.static_baseline import run_static_episode
from src.data import load_qrels, load_queries
from src.eval.bootstrap import paired_bootstrap
from src.eval.metrics import evaluate, helped_unchanged_harmed, mean_metrics
from src.observability.trace_schema import TraceRecord
from src.retrieval.bm25_index import search

_DB_PATHS = {
    "nfcorpus": "data/nfcorpus_bm25.sqlite3",
    "tripclick-head": "data/tripclick_bm25.sqlite3",
    "tripclick-torso": "data/tripclick_bm25.sqlite3",
    "tripclick-tail": "data/tripclick_bm25.sqlite3",
    "tripclick-tail-val": "data/tripclick_bm25.sqlite3",
}

_CONFIG_PATH = Path("configs/default.yaml")
_CHECKPOINT_EVERY = 50


def freeze_slo(harm_rate: float, mean_llm_calls: float, p95_latency_ms: float, path: Path = _CONFIG_PATH) -> None:
    """Targeted substitution of the three slo: fields, rather than a full
    YAML round-trip, so the file's comments/formatting survive.
    """
    text = path.read_text()
    text = re.sub(r"max_harm_rate:\s*\S+", f"max_harm_rate: {harm_rate:.4f}", text)
    text = re.sub(r"max_expected_llm_calls:\s*\S+", f"max_expected_llm_calls: {mean_llm_calls:.4f}", text)
    text = re.sub(r"max_p95_latency_ms:\s*\S+", f"max_p95_latency_ms: {p95_latency_ms:.1f}", text)
    path.write_text(text)


def _checkpoint_path(dataset: str) -> Path:
    return Path(f"results/{dataset}_agent_vs_static_checkpoint.json")


def _save_checkpoint(
    dataset: str,
    baseline_run: dict,
    static_run: dict,
    static_traces: list[TraceRecord],
    static_done: bool,
    agent_run: dict,
    agent_traces: list[TraceRecord],
) -> None:
    path = _checkpoint_path(dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "dataset": dataset,
                "baseline_run": baseline_run,
                "static_run": static_run,
                "static_traces": [dataclasses.asdict(t) for t in static_traces],
                "static_done": static_done,
                "agent_run": agent_run,
                "agent_traces": [dataclasses.asdict(t) for t in agent_traces],
            }
        )
    )


def _load_checkpoint(dataset: str) -> dict | None:
    path = _checkpoint_path(dataset)
    if not path.exists():
        return None
    saved = json.loads(path.read_text())
    saved["static_traces"] = [TraceRecord(**d) for d in saved["static_traces"]]
    saved["agent_traces"] = [TraceRecord(**d) for d in saved["agent_traces"]]
    return saved


def main() -> None:
    args = [a for a in sys.argv[1:] if a not in ("--freeze-slo", "--resume")]
    do_freeze = "--freeze-slo" in sys.argv
    resume = "--resume" in sys.argv
    dataset = args[0] if len(args) > 0 else "nfcorpus"
    n = int(args[1]) if len(args) > 1 else 15
    db_path = _DB_PATHS[dataset]

    queries = load_queries(dataset)
    qrels = load_qrels(dataset)
    query_ids = [qid for qid in queries if qid in qrels][:n]

    baseline_run, static_run, agent_run = {}, {}, {}
    static_traces, agent_traces = [], []
    static_done = False

    if resume:
        saved = _load_checkpoint(dataset)
        if saved is not None and saved["dataset"] == dataset:
            baseline_run = saved["baseline_run"]
            static_run = saved["static_run"]
            static_traces = saved["static_traces"]
            static_done = saved["static_done"]
            agent_run = saved["agent_run"]
            agent_traces = saved["agent_traces"]
            print(
                f"Resuming: static {len(static_run)}/{len(query_ids)} "
                f"(done={static_done}), agent {len(agent_run)}/{len(query_ids)}",
                flush=True,
            )
        else:
            print("--resume requested but no matching checkpoint found -- starting fresh.", flush=True)

    # Two fully separate passes, not interleaved per query: running static
    # and agent back-to-back on the *same* prompt let Ollama's backend
    # reuse a warm KV-cache prefill for the second (agent) call, making
    # its latency look artificially ~6x faster in an earlier run of this
    # script. Separating the passes so ~100 unrelated prompts sit between
    # any repeat gives each pipeline an honest, independent latency read.
    if not static_done:
        print("--- static pass ---", flush=True)
        for i, qid in enumerate(query_ids[len(static_run) :], len(static_run) + 1):
            text = queries[qid]
            baseline_run[qid] = dict(search(text, db_path, k=100))
            static_ranking, static_trace = run_static_episode(qid, text, db_path, dataset_slice=dataset)
            static_run[qid] = {doc_id: 1.0 / (rank + 1) for rank, doc_id in enumerate(static_ranking)}
            static_traces.append(static_trace)
            print(f"{qid:>8}  static={static_trace.controller_decision:<8}", flush=True)
            if i % _CHECKPOINT_EVERY == 0:
                _save_checkpoint(dataset, baseline_run, static_run, static_traces, False, agent_run, agent_traces)
                print(f"  [checkpoint saved, static {i}/{len(query_ids)}]", flush=True)
        static_done = True
        _save_checkpoint(dataset, baseline_run, static_run, static_traces, True, agent_run, agent_traces)

    print("\n--- agent pass ---", flush=True)
    for i, qid in enumerate(query_ids[len(agent_run) :], len(agent_run) + 1):
        text = queries[qid]
        agent_ranking, agent_trace = run_episode(qid, text, db_path, dataset_slice=dataset)
        agent_run[qid] = {doc_id: 1.0 / (rank + 1) for rank, doc_id in enumerate(agent_ranking)}
        agent_traces.append(agent_trace)
        print(f"{qid:>8}  agent={agent_trace.controller_decision:<8}", flush=True)
        if i % _CHECKPOINT_EVERY == 0:
            _save_checkpoint(dataset, baseline_run, static_run, static_traces, True, agent_run, agent_traces)
            print(f"  [checkpoint saved, agent {i}/{len(query_ids)}]", flush=True)

    q = {qid: qrels[qid] for qid in query_ids}
    baseline_eval = evaluate(q, baseline_run)
    static_eval = evaluate(q, static_run)
    agent_eval = evaluate(q, agent_run)

    print(f"\nn_queries={len(query_ids)}", flush=True)
    print(f"BM25 only:  {mean_metrics(baseline_eval)}", flush=True)
    print(f"Static QE:  {mean_metrics(static_eval)}", flush=True)
    print(f"Agent:      {mean_metrics(agent_eval)}", flush=True)

    print(f"\nagent vs static, helped/unchanged/harmed: {helped_unchanged_harmed(static_eval, agent_eval)}", flush=True)
    print(f"agent vs BM25,   helped/unchanged/harmed: {helped_unchanged_harmed(baseline_eval, agent_eval)}", flush=True)
    print(f"static vs BM25,  helped/unchanged/harmed: {helped_unchanged_harmed(baseline_eval, static_eval)}", flush=True)

    static_ndcg = [static_eval[qid]["ndcg_cut_10"] for qid in query_ids]
    agent_ndcg = [agent_eval[qid]["ndcg_cut_10"] for qid in query_ids]
    bootstrap = paired_bootstrap(static_ndcg, agent_ndcg)
    print(
        f"\npaired bootstrap, agent - static NDCG@10: mean_diff={bootstrap['mean_diff']:.4f} "
        f"95% CI [{bootstrap['ci_lo']:.4f}, {bootstrap['ci_hi']:.4f}]  significant={bootstrap['significant']}",
        flush=True,
    )

    static_calls = sum(t.llm_calls for t in static_traces) / len(static_traces)
    agent_calls = sum(t.llm_calls for t in agent_traces) / len(agent_traces)
    static_latency = sum(t.episode_latency_ms for t in static_traces) / len(static_traces)
    agent_latency = sum(t.episode_latency_ms for t in agent_traces) / len(agent_traces)
    print(
        f"\ncost -- static: {static_calls:.2f} llm_calls/query, {static_latency:.0f}ms/query mean\n"
        f"        agent:  {agent_calls:.2f} llm_calls/query, {agent_latency:.0f}ms/query mean",
        flush=True,
    )

    if len(query_ids) < 50:
        print(
            "\nNOTE: this is a small dev sample -- the bootstrap CI above is illustrative, "
            "not a claim of statistical significance for the paper.",
            flush=True,
        )

    if do_freeze:
        harm_rate = helped_unchanged_harmed(baseline_eval, agent_eval)["harmed"]
        latencies = sorted(t.episode_latency_ms for t in agent_traces)
        p95_latency = latencies[int(0.95 * (len(latencies) - 1))]
        if len(query_ids) < 100:
            print(
                f"\n--freeze-slo requested but n_queries={len(query_ids)} < 100 -- "
                "refusing to freeze on an under-sized sample (docs/milestones.md). Not writing configs/default.yaml.",
                flush=True,
            )
        else:
            freeze_slo(harm_rate, agent_calls, p95_latency)
            print(
                f"\nFroze SLO into {_CONFIG_PATH}: max_harm_rate={harm_rate:.4f} "
                f"max_expected_llm_calls={agent_calls:.4f} max_p95_latency_ms={p95_latency:.1f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
