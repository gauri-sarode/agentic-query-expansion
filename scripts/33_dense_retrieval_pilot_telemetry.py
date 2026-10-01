"""Self-contained, reduced-feature telemetry for the dense-retrieval
pilot (WSDM reviewer question: does the detection-vs-actionability gap
hold under dense first-stage retrieval, not just BM25?).

Deliberately NOT importing from src/observability/telemetry.py: that
module's compute_retrieval_telemetry() calls bm25_index.search() and
bm25_index.idf() directly inside two of its eight features
(mean_rare_term_idf, and ranking_stability -- which re-queries BM25
with the lowest-IDF query term dropped) -- i.e. it is not retriever-
agnostic despite taking a ranking as a parameter. Making it pluggable
would mean editing a module every existing frozen result in this
project depends on, for a single bounded pilot -- not worth that risk.
This file reimplements the retriever-agnostic subset of the same
features from scratch, computable from any (ranking, query, doc texts)
triple regardless of what produced the ranking.

Dropped from the original 8 detection features: mean_rare_term_idf,
ranking_stability (both require corpus document-frequency/IDF, which
this module's lean dense_index.py meta table does not compute --
IDF needs the FTS5 structure bm25_index.py builds, deliberately not
built here, see dense_index.py's module docstring). The remaining 6
are reported honestly as a reduced set, not padded to look like parity
with the main paper's 8.

Dropped from the original 10 action-utility features: "stability"
only (the same IDF-coupled ranking_stability feature, reused as a
post-action signal in the main pipeline) -- reduced set here is 9.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return {t.lower() for t in _WORD_RE.findall(text)}


def _entropy(scores: list[float]) -> float:
    if len(scores) < 2:
        return 0.0
    shifted = [s - min(scores) + 1e-9 for s in scores]
    total = sum(shifted)
    probs = [s / total for s in shifted]
    h = -sum(p * math.log(p + 1e-12) for p in probs)
    return h / math.log(len(scores))


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass(frozen=True)
class ReducedRetrievalTelemetry:
    top_score: float
    score_margin: float
    score_entropy: float
    query_term_coverage: float
    entity_overlap: float
    result_coherence: float


def compute_reduced_retrieval_telemetry(
    query: str,
    ranking: list[tuple[str, float]],
    texts: dict[str, str],
    coherence_sample: int = 10,
) -> ReducedRetrievalTelemetry:
    """ranking: (doc_id, score) pairs, any retriever. texts: doc_id->text
    for at least the top coherence_sample docs (caller fetches via
    dense_index.get_texts or bm25_index.get_texts, either works -- this
    function only reads the dict)."""
    scores = [s for _, s in ranking]
    top_score = scores[0] if scores else 0.0
    score_margin = (scores[0] - scores[1]) if len(scores) >= 2 else 0.0
    score_entropy = _entropy(scores[:10]) if scores else 0.0

    query_terms = _tokenize(query)
    top_ids = [d for d, _ in ranking[:coherence_sample]]
    top_texts = [texts.get(d, "") for d in top_ids]
    top_token_sets = [_tokenize(t) for t in top_texts]
    all_top_tokens: set[str] = set().union(*top_token_sets) if top_token_sets else set()

    covered = sum(1 for t in query_terms if t in all_top_tokens)
    query_term_coverage = covered / len(query_terms) if query_terms else 0.0

    cap_terms = {w.lower() for w in re.findall(r"\b[A-Z][a-zA-Z]+\b", query)}
    entity_overlap = (
        sum(1 for t in cap_terms if t in all_top_tokens) / len(cap_terms) if cap_terms else 1.0
    )

    if len(top_token_sets) >= 2:
        pairs = [
            _jaccard(top_token_sets[i], top_token_sets[j])
            for i in range(len(top_token_sets))
            for j in range(i + 1, len(top_token_sets))
        ]
        result_coherence = sum(pairs) / len(pairs)
    else:
        result_coherence = 0.0

    return ReducedRetrievalTelemetry(
        top_score=top_score,
        score_margin=score_margin,
        score_entropy=score_entropy,
        query_term_coverage=query_term_coverage,
        entity_overlap=entity_overlap,
        result_coherence=result_coherence,
    )


DETECTION_FEATURE_NAMES = [
    "top_score", "score_margin", "score_entropy",
    "query_term_coverage", "entity_overlap", "result_coherence",
]


def detection_feature_vector(t: ReducedRetrievalTelemetry) -> list[float]:
    return [t.top_score, t.score_margin, t.score_entropy,
            t.query_term_coverage, t.entity_overlap, t.result_coherence]


ACTION_UTILITY_FEATURE_NAMES = [
    "score_margin_gain", "coverage_gain", "coherence_gain",
    "drift", "disagreement", "topk_overlap",
    "reranker_top_score", "reranker_score_margin", "query_length_ratio",
]


def action_utility_feature_vector(
    t0: ReducedRetrievalTelemetry,
    t1: ReducedRetrievalTelemetry,
    q0: str,
    q_t: str,
    pre_rerank_ids: list[str],
    post_rerank_ids: list[str],
    r0_ids: list[str],
    r1_ids: list[str],
    reranker_top_score: float,
    reranker_score_margin: float,
) -> list[float]:
    score_margin_gain = t1.score_margin - t0.score_margin
    coverage_gain = t1.query_term_coverage - t0.query_term_coverage
    coherence_gain = t1.result_coherence - t0.result_coherence

    q0_tokens, qt_tokens = _tokenize(q0), _tokenize(q_t)
    drift = 1.0 - _jaccard(q0_tokens, qt_tokens)
    query_length_ratio = (len(qt_tokens) / len(q0_tokens)) if q0_tokens else 1.0

    disagreement = 1.0 - _jaccard(set(pre_rerank_ids), set(post_rerank_ids))
    topk_overlap = _jaccard(set(r1_ids), set(r0_ids))

    return [
        score_margin_gain, coverage_gain, coherence_gain,
        drift, disagreement, topk_overlap,
        reranker_top_score, reranker_score_margin, query_length_ratio,
    ]
