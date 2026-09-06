#!/usr/bin/env python3
"""Fourth cell of the BM25 x {QE, rerank} factorial requested in review:
does expansion's apparent "canceling" of reranker harm (Sections VI,
RQ2) actually hold, or is it just two roughly-neutral numbers landing
near each other by coincidence? Table III already has three of the four
cells (BM25 only 0.3152; BM25+rerank, no QE 0.2850; BM25+QE+rerank
"Static QE" 0.3159); this script fills the fourth: BM25+QE, no rerank.

Reuses the FROZEN RQ2 checkpoint's baseline_run and static_run
(results/tripclick-tail_agent_vs_static_checkpoint.json) and the
reranker-only checkpoint (results/tripclick-tail_bm25_reranker_only.json)
rather than recomputing them -- those are single-touch frozen-test
results (configs/experimental_protocol.yaml) and must not be rerun.
Only the new no-reranker variant is computed here, on the identical
query population, mirroring scripts/12's precedent of one additional
frozen-test diagnostic touch beyond the three original RQ1-RQ3 touches.

Also computes the paired bootstrap CI for the QE x reranking
interaction (review request): with BM25/QE/rerank as a 2x2 factorial,
  I = mean(QE+rerank) - mean(QE) - mean(rerank) + mean(BM25)
resampled jointly over query indices (paired across all four arms,
same qid drawn together each resample) since all four conditions share
the identical query population.

Per-query no-rerank nDCG values are now persisted (they were not in
the original run this script replaces), so this is the authoritative,
reproducible source for both the no-rerank cell and the interaction
term.

Usage: PYTHONPATH=. python3 scripts/13_static_qe_no_reranker.py
"""
from __future__ import annotations

import json

import numpy as np

from src.agent.static_baseline import run_static_episode
from src.data import load_qrels, load_queries
from src.eval.bootstrap import paired_bootstrap
from src.eval.metrics import evaluate

DB_PATH = "data/tripclick_bm25.sqlite3"
DATASET = "tripclick-tail"
CHECKPOINT_PATH = "results/tripclick-tail_agent_vs_static_checkpoint.json"
RERANKER_ONLY_PATH = "results/tripclick-tail_bm25_reranker_only.json"
OUT_PATH = "results/tripclick-tail_static_qe_no_reranker.json"
N_BOOTSTRAP = 10000
SEED = 42


def interaction_ci(b: np.ndarray, rerank_only: np.ndarray, qe_only: np.ndarray, qe_rerank: np.ndarray):
    point = qe_rerank.mean() - qe_only.mean() - rerank_only.mean() + b.mean()
    n = len(b)
    rng = np.random.RandomState(SEED)
    samples = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.randint(0, n, n)
        samples.append(qe_rerank[idx].mean() - qe_only[idx].mean() - rerank_only[idx].mean() + b[idx].mean())
    samples = np.array(samples)
    return float(point), float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def main() -> None:
    frozen = json.loads(open(CHECKPOINT_PATH).read())
    baseline_run = frozen["baseline_run"]
    static_run_reranked = frozen["static_run"]
    reranker_only = json.loads(open(RERANKER_ONLY_PATH).read())

    queries = load_queries(DATASET)
    qrels = load_qrels(DATASET)
    query_ids = [qid for qid in queries if qid in qrels]
    assert set(query_ids) == set(baseline_run.keys()), "population must match the frozen RQ2 checkpoint exactly"
    print(f"n queries: {len(query_ids)}", flush=True)

    static_run_no_rerank = {}
    for i, qid in enumerate(query_ids, 1):
        ranking, _trace = run_static_episode(qid, queries[qid], DB_PATH, dataset_slice=DATASET, use_reranker=False)
        static_run_no_rerank[qid] = {doc_id: 1.0 / (rank + 1) for rank, doc_id in enumerate(ranking)}
        if i % 100 == 0:
            print(f"  [{i}/{len(query_ids)}]", flush=True)

    baseline_eval = evaluate(qrels, baseline_run)
    static_reranked_eval = evaluate(qrels, static_run_reranked)
    static_no_rerank_eval = evaluate(qrels, static_run_no_rerank)
    reranker_only_eval = reranker_only["reranked_ndcg"]

    b = [baseline_eval[q]["ndcg_cut_10"] for q in query_ids]
    s_rerank = [static_reranked_eval[q]["ndcg_cut_10"] for q in query_ids]
    s_no_rerank = [static_no_rerank_eval[q]["ndcg_cut_10"] for q in query_ids]
    r_no_rerank = [static_no_rerank_eval[q]["recall_100"] for q in query_ids]
    m_no_rerank = [static_no_rerank_eval[q]["recip_rank"] for q in query_ids]
    rerank_only_vals = [reranker_only_eval[q] for q in query_ids]

    mean_ndcg_no_rerank = sum(s_no_rerank) / len(s_no_rerank)
    mean_ndcg_rerank = sum(s_rerank) / len(s_rerank)

    vs_bm25 = paired_bootstrap(b, s_no_rerank)
    vs_static_reranked = paired_bootstrap(s_no_rerank, s_rerank)  # does adding rerank back help or hurt, given QE?

    print("\n=== BM25 + QE, no reranker (frozen TAIL test, n=%d) ===" % len(query_ids))
    print("mean nDCG@10:", mean_ndcg_no_rerank)
    print("Recall@100:", sum(r_no_rerank) / len(r_no_rerank))
    print("MRR:", sum(m_no_rerank) / len(m_no_rerank))
    print("\nvs BM25 only:", vs_bm25)
    print("\nvs Static QE (reranked) -- effect of ADDING reranker given QE already applied:", vs_static_reranked)

    reranker_penalty_no_qe = 0.2850 - 0.3152  # from Table III's existing rows
    reranker_penalty_with_qe = mean_ndcg_rerank - mean_ndcg_no_rerank
    print(f"\nreranker penalty WITHOUT QE (BM25+rerank - BM25): {reranker_penalty_no_qe:+.4f}")
    print(f"reranker penalty WITH QE (StaticQE+rerank - StaticQE+no-rerank): {reranker_penalty_with_qe:+.4f}")

    point, ci_lo, ci_hi = interaction_ci(
        np.array(b), np.array(rerank_only_vals), np.array(s_no_rerank), np.array(s_rerank)
    )
    print(f"\nQE x reranking interaction I = mean(QE+RR)-mean(QE)-mean(RR)+mean(BM25):")
    print(f"  I = {point:+.4f}  95% CI = [{ci_lo:+.4f}, {ci_hi:+.4f}]")

    out = {
        "n": len(query_ids),
        "mean_ndcg10_no_rerank": mean_ndcg_no_rerank,
        "mean_recall100_no_rerank": sum(r_no_rerank) / len(r_no_rerank),
        "mean_mrr_no_rerank": sum(m_no_rerank) / len(m_no_rerank),
        "vs_bm25": vs_bm25,
        "vs_static_reranked": vs_static_reranked,
        "reranker_penalty_no_qe": reranker_penalty_no_qe,
        "reranker_penalty_with_qe": reranker_penalty_with_qe,
        "per_query_ndcg10_no_rerank": {q: v for q, v in zip(query_ids, s_no_rerank)},
        "interaction": {"point": point, "ci_lo": ci_lo, "ci_hi": ci_hi},
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
