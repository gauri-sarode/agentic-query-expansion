#!/usr/bin/env python3
"""Tests whether embedding-based (dense semantic) features raise the
verifier's predictive ceiling beyond the lexical/scalar features tried so
far (scripts/08_calibrate_verifier.py: r=0.127 hand-picked, r=0.148/0.161
fitted on NFCorpus, r=0.134 fitted on TripClick TAIL -- all lexical/BM25/
reranker-score based).

Three new features, using the local mxbai-embed-large model
(src/observability/embeddings.py):
  - embedding_drift: 1 - cosine_similarity(embed(q0), embed(q_t)) -- a
    real semantic-distance measure, vs. the existing rewrite_semantic_drift's
    crude token-Jaccard proxy.
  - embedding_coherence: mean pairwise cosine similarity among the new
    ranking's top-5 documents -- semantic coherence, vs. the existing
    lexical-Jaccard result_coherence.
  - embedding_groundedness: max cosine similarity between the generated
    expansion and any evidence passage -- does the corpus-grounded
    rewrite actually align with the evidence it was grounded in? No
    lexical analogue exists for this at all.

Unlike scripts/08, this ALSO persists the raw text (q0, generated
expansion, evidence, candidate doc texts) alongside the numeric features,
so future feature experiments can be computed from the saved data without
re-running the LLM-call-expensive collection loop.

Usage: python scripts/09_calibrate_embedding_features.py [dataset] [n_queries]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict

from src.agent.actions import EXPANSION_ACTIONS
from src.agent.controller import ActionSelector
from src.agent.state import AgentState
from src.agent.steps import act_once, build_evidence, retrieve_and_observe_initial
from src.data import load_qrels, load_queries
from src.eval.metrics import evaluate
from src.llm_client import LLMUnavailableError
from src.observability.embeddings import EmbeddingUnavailableError, cosine_similarity, embed, mean_pairwise_similarity
from src.retrieval.bm25_index import get_texts

_DB_PATHS = {
    "nfcorpus": "data/nfcorpus_bm25.sqlite3",
    "tripclick-head": "data/tripclick_bm25.sqlite3",
    "tripclick-torso": "data/tripclick_bm25.sqlite3",
    "tripclick-tail": "data/tripclick_bm25.sqlite3",
}

_LEXICAL_FEATURE_NAMES = [
    "score_margin_gain", "coverage_gain", "coherence_gain", "stability", "drift",
    "disagreement", "topk_overlap", "reranker_top_score", "reranker_score_margin",
    "query_length_ratio",
]
_EMBEDDING_FEATURE_NAMES = ["embedding_drift", "embedding_coherence", "embedding_groundedness"]
_ALL_FEATURE_NAMES = _LEXICAL_FEATURE_NAMES + _EMBEDDING_FEATURE_NAMES


def _rank_to_scores(ranking: list[str]) -> dict[str, float]:
    return {d: 1.0 / (i + 1) for i, d in enumerate(ranking)}


def _ndcg10(qid: str, ranking: list[str], qrels: dict) -> float:
    result = evaluate({qid: qrels[qid]}, {qid: _rank_to_scores(ranking)})
    return result[qid]["ndcg_cut_10"]


def _lexical_features(prev_o, new_o) -> list[float]:
    return [
        new_o.retrieval.score_margin - prev_o.retrieval.score_margin,
        new_o.retrieval.query_term_coverage - prev_o.retrieval.query_term_coverage,
        new_o.retrieval.result_coherence - prev_o.retrieval.result_coherence,
        new_o.retrieval.ranking_stability,
        new_o.agent.rewrite_semantic_drift,
        new_o.retrieval.reranker_bm25_disagreement,
        new_o.retrieval.topk_overlap,
        new_o.retrieval.reranker_top_score,
        new_o.retrieval.reranker_score_margin,
        new_o.agent.query_length_ratio,
    ]


def _embedding_features(q0: str, q_t: str, evidence: list[str], candidate_texts: list[str]) -> list[float]:
    q0_vec = embed(q0)
    qt_vec = embed(q_t)
    embedding_drift = 1.0 - cosine_similarity(q0_vec, qt_vec)
    embedding_coherence = mean_pairwise_similarity(candidate_texts[:5])
    groundedness_sims = [cosine_similarity(qt_vec, embed(e)) for e in evidence] if evidence else []
    embedding_groundedness = max(groundedness_sims) if groundedness_sims else 0.0
    return [embedding_drift, embedding_coherence, embedding_groundedness]


def _save(dataset: str, samples: list[dict]) -> Path:
    out_path = Path(f"results/{dataset}_embedding_calibration.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"dataset": dataset, "feature_names": _ALL_FEATURE_NAMES, "samples": samples}, indent=2))
    return out_path


def main() -> None:
    dataset = sys.argv[1] if len(sys.argv) > 1 else "nfcorpus"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    db_path = _DB_PATHS[dataset]

    queries = load_queries(dataset)
    qrels = load_qrels(dataset)
    query_ids = [qid for qid in queries if qid in qrels][:n]

    selector = ActionSelector()
    samples: list[dict] = []
    skipped_no_expand = 0
    skipped_llm_error = 0
    t_start = time.time()

    for i, qid in enumerate(query_ids, 1):
        text = queries[qid]
        init = retrieve_and_observe_initial(text, db_path, k=100)
        action = selector.select(text, init.o0)
        if action not in EXPANSION_ACTIONS:
            skipped_no_expand += 1
            continue

        state = AgentState(q0=text, q_t=text, R_t=init.r0_ids, O_t=init.o0, B_t=1)
        evidence = build_evidence(init.r0_ids, db_path)
        try:
            result = act_once(action, state, init.r0, db_path, evidence, k=100, use_reranker=True)
        except LLMUnavailableError:
            skipped_llm_error += 1
            continue

        candidate_top5_texts = list(get_texts(result.candidate.R_t[:5], db_path).values())
        try:
            embedding = _embedding_features(text, result.generated_expansion, evidence, candidate_top5_texts)
        except EmbeddingUnavailableError as e:
            print(f"{qid:>8}  SKIPPED (embedding error: {e})", flush=True)
            skipped_llm_error += 1  # reused counter: any tool failure, not just LLM
            continue

        original_ndcg = _ndcg10(qid, init.r0_ids, qrels)
        candidate_ndcg = _ndcg10(qid, result.candidate.R_t, qrels)
        delta_ndcg = candidate_ndcg - original_ndcg

        lexical = _lexical_features(init.o0, result.candidate.O_t)

        samples.append(
            {
                "qid": qid,
                "q0": text,
                "action": action.value,
                "generated_expansion": result.generated_expansion,
                "evidence": evidence,
                "candidate_top5_texts": candidate_top5_texts,
                "features": lexical + embedding,
                "delta_ndcg": delta_ndcg,
                "current_verify_score": result.verifier_score,
            }
        )

        print(
            f"{qid:>8}  action={action.value:<16} verify={result.verifier_score:+8.3f}  delta_ndcg={delta_ndcg:+.4f}",
            flush=True,
        )

        if i % 10 == 0 or i == len(query_ids):
            elapsed = time.time() - t_start
            rate = elapsed / i
            eta_min = rate * (len(query_ids) - i) / 60
            print(f"  [progress {i}/{len(query_ids)}  {rate:.1f}s/query  elapsed={elapsed / 60:.1f}min  eta={eta_min:.1f}min]", flush=True)
        if i % 25 == 0:
            _save(dataset, samples)
            print(f"  [checkpoint saved, n={len(samples)}]", flush=True)

    print(f"\nn_samples={len(samples)}  (skipped: {skipped_no_expand} NO_EXPAND, {skipped_llm_error} LLM errors)", flush=True)
    out_path = _save(dataset, samples)
    print(f"raw samples written to {out_path}", flush=True)

    X = np.array([s["features"] for s in samples])
    y = np.array([s["delta_ndcg"] for s in samples])
    y_current = np.array([s["current_verify_score"] for s in samples])

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    current_corr = np.corrcoef(y_current, y)[0, 1]
    print(f"\ncurrent fixed-weight verify() vs. true NDCG@10 delta: Pearson r = {current_corr:.3f}")

    lexical_only = LinearRegression(fit_intercept=False)
    X_lex = X[:, : len(_LEXICAL_FEATURE_NAMES)]
    lex_pred = cross_val_predict(lexical_only, X_lex, y, cv=kf)
    lex_corr = np.corrcoef(lex_pred, y)[0, 1]
    print(f"lexical-only fit (10 features, 5-fold CV): Pearson r = {lex_corr:.3f}")

    all_features = LinearRegression(fit_intercept=False)
    all_pred = cross_val_predict(all_features, X, y, cv=kf)
    all_corr = np.corrcoef(all_pred, y)[0, 1]
    print(f"lexical + embedding fit (13 features, 5-fold CV): Pearson r = {all_corr:.3f}")

    all_features.fit(X, y)
    print("\nfitted weights (full-data fit, lexical+embedding):")
    for name, coef in zip(_ALL_FEATURE_NAMES, all_features.coef_):
        print(f"  {name:<22} {coef:+.4f}")

    if all_corr > lex_corr + 0.05:
        print(f"\nEmbedding features meaningfully improve on lexical-only (r={all_corr:.3f} vs {lex_corr:.3f}) -- worth adopting.")
    else:
        print(
            f"\nEmbedding features do NOT meaningfully improve on lexical-only "
            f"(r={all_corr:.3f} vs {lex_corr:.3f}). This narrows the verifier's ceiling down to a "
            "genuine detection-difficulty finding rather than a missing-feature-family one."
        )


if __name__ == "__main__":
    main()
