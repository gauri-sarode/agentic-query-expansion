"""Observation vector O_t: three telemetry families computed from the
current retrieval state, agent trajectory, and system runtime. These
features are what the agent's controller and verifier condition on --
telemetry as causal input to policy, not passive logging. See
docs/observability_model.md.

v0 implementation note: retrieval telemetry is computed from BM25 (lexical)
signals only, plus the reranker's cross-encoder scores where available --
there is no separate dense/semantic retriever in this experiment (see
docs/milestones.md), so features like "entity overlap" and "result
coherence" are lexical proxies, not embedding-based. That's a deliberate
scope choice for the v0 controller/verifier, not a placeholder.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from src.retrieval import bm25_index

_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return {t.lower() for t in _WORD_RE.findall(text)}


def _entropy(scores: list[float]) -> float:
    """Normalized entropy of a score distribution -- 0 (all mass on one
    result) to 1 (uniform), robust to negative/unnormalized BM25 scores.
    """
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
class RetrievalTelemetry:
    top_score: float
    score_margin: float  # top_1 - top_2
    score_entropy: float  # normalized entropy of top-k score distribution
    query_term_coverage: float  # fraction of query terms with lexical support in top-k
    mean_rare_term_idf: float
    entity_overlap: float  # capitalized query tokens present verbatim in top-k docs
    topk_overlap: float  # Jaccard overlap of top-k doc IDs vs. previous state (1.0 if no previous)
    reranker_bm25_disagreement: float  # 1 - top-k overlap between BM25 order and reranked order
    result_coherence: float  # mean pairwise lexical Jaccard among top-k docs
    ranking_stability: float  # top-k overlap after dropping the lowest-IDF query term
    reranker_top_score: float  # cross-encoder score of the top reranked doc (0.0 if reranker unavailable)
    reranker_score_margin: float  # cross-encoder top_1 - top_2 (0.0 if reranker unavailable/<2 candidates)


@dataclass(frozen=True)
class AgentTelemetry:
    selected_operator: str
    trajectory_length: int
    previous_verifier_score: float | None
    rewrite_semantic_drift: float  # 1 - token Jaccard, q_t vs. q_0 (lexical proxy)
    introduced_entities: int  # capitalized tokens in q_t absent from q_0
    query_length_ratio: float  # len(q_t tokens) / len(q_0 tokens)
    repeated_actions: int
    remaining_budget: int
    accepted: bool | None
    rolled_back: bool


@dataclass(frozen=True)
class SystemTelemetry:
    bm25_latency_ms: float
    reranker_latency_ms: float
    llm_latency_ms: float
    prompt_tokens: int
    output_tokens: int
    tool_errors: int
    timed_out: bool


@dataclass(frozen=True)
class ObservabilityState:
    """O_t: the full observation the controller and verifier see at step t."""

    retrieval: RetrievalTelemetry
    agent: AgentTelemetry
    system: SystemTelemetry


def compute_retrieval_telemetry(
    query: str,
    ranking: list[tuple[str, float]],  # (doc_id, score), BM25 order, already top-k
    db_path: str,
    previous_ranking: list[tuple[str, float]] | None = None,
    reranked_order: list[str] | None = None,
    reranked_scores: list[tuple[str, float]] | None = None,
    coherence_sample: int = 10,
) -> RetrievalTelemetry:
    doc_ids = [d for d, _ in ranking]
    scores = [s for _, s in ranking]
    top_score = scores[0] if scores else 0.0
    score_margin = (scores[0] - scores[1]) if len(scores) >= 2 else 0.0
    score_entropy = _entropy(scores)

    query_terms = _tokenize(query)
    texts = bm25_index.get_texts(doc_ids[:coherence_sample], db_path)
    doc_token_sets = {doc_id: _tokenize(text) for doc_id, text in texts.items()}
    covered_terms = set()
    for tokens in doc_token_sets.values():
        covered_terms |= (tokens & query_terms)
    query_term_coverage = len(covered_terms) / len(query_terms) if query_terms else 1.0

    n = bm25_index.doc_count(db_path)
    idfs = sorted((bm25_index.idf(t, db_path, total_docs=n) for t in query_terms), reverse=True)
    mean_rare_term_idf = sum(idfs) / len(idfs) if idfs else 0.0

    capitalized = {t for t in re.findall(r"\b[A-Z][a-zA-Z0-9]*\b", query)}
    lower_doc_tokens = {tok for toks in doc_token_sets.values() for tok in toks}
    entity_overlap = (
        sum(1 for e in capitalized if e.lower() in lower_doc_tokens) / len(capitalized)
        if capitalized
        else 1.0
    )

    if previous_ranking is not None:
        topk_overlap = _jaccard(set(doc_ids), {d for d, _ in previous_ranking})
    else:
        topk_overlap = 1.0

    if reranked_order:
        reranker_bm25_disagreement = 1.0 - _jaccard(set(doc_ids), set(reranked_order))
    else:
        reranker_bm25_disagreement = 0.0

    sample_ids = list(doc_token_sets.keys())
    if len(sample_ids) >= 2:
        pairs = [
            _jaccard(doc_token_sets[a], doc_token_sets[b])
            for i, a in enumerate(sample_ids)
            for b in sample_ids[i + 1 :]
        ]
        result_coherence = sum(pairs) / len(pairs)
    else:
        result_coherence = 1.0

    if query_terms and idfs:
        # Drop the single lowest-IDF (least informative) query term and
        # re-search; a healthy trajectory shouldn't change much.
        rarest_first = sorted(query_terms, key=lambda t: bm25_index.idf(t, db_path, total_docs=n))
        perturbed_query = " ".join(t for t in query.split() if t.lower() != rarest_first[0]) or query
        perturbed = bm25_index.search(perturbed_query, db_path, k=len(doc_ids) or 10)
        ranking_stability = _jaccard(set(doc_ids), {d for d, _ in perturbed})
    else:
        ranking_stability = 1.0

    return RetrievalTelemetry(
        top_score=top_score,
        score_margin=score_margin,
        score_entropy=score_entropy,
        query_term_coverage=query_term_coverage,
        mean_rare_term_idf=mean_rare_term_idf,
        entity_overlap=entity_overlap,
        topk_overlap=topk_overlap,
        reranker_bm25_disagreement=reranker_bm25_disagreement,
        result_coherence=result_coherence,
        ranking_stability=ranking_stability,
        reranker_top_score=reranked_scores[0][1] if reranked_scores else 0.0,
        reranker_score_margin=(reranked_scores[0][1] - reranked_scores[1][1]) if reranked_scores and len(reranked_scores) >= 2 else 0.0,
    )


def compute_agent_telemetry(
    q0: str,
    q_t: str,
    selected_operator: str,
    history: list,  # list[HistoryStep], kept untyped here to avoid a src.agent import cycle
    remaining_budget: int,
    previous_verifier_score: float | None,
    accepted: bool | None,
    rolled_back: bool,
) -> AgentTelemetry:
    q0_tokens = _tokenize(q0)
    qt_tokens = _tokenize(q_t)
    rewrite_semantic_drift = 1.0 - _jaccard(q0_tokens, qt_tokens)

    capitalized_qt = {t for t in re.findall(r"\b[A-Z][a-zA-Z0-9]*\b", q_t)}
    capitalized_q0 = {t for t in re.findall(r"\b[A-Z][a-zA-Z0-9]*\b", q0)}
    introduced_entities = len(capitalized_qt - capitalized_q0)

    actions_taken = [step.action for step in history]
    repeated_actions = len(actions_taken) - len(set(actions_taken))
    query_length_ratio = (len(q_t.split()) / len(q0.split())) if q0.split() else 1.0

    return AgentTelemetry(
        selected_operator=selected_operator,
        trajectory_length=len(history),
        previous_verifier_score=previous_verifier_score,
        rewrite_semantic_drift=rewrite_semantic_drift,
        introduced_entities=introduced_entities,
        query_length_ratio=query_length_ratio,
        repeated_actions=repeated_actions,
        remaining_budget=remaining_budget,
        accepted=accepted,
        rolled_back=rolled_back,
    )


def compute_system_telemetry(
    bm25_latency_ms: float = 0.0,
    reranker_latency_ms: float = 0.0,
    llm_latency_ms: float = 0.0,
    prompt_tokens: int = 0,
    output_tokens: int = 0,
    tool_errors: int = 0,
    timed_out: bool = False,
) -> SystemTelemetry:
    return SystemTelemetry(
        bm25_latency_ms=bm25_latency_ms,
        reranker_latency_ms=reranker_latency_ms,
        llm_latency_ms=llm_latency_ms,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        tool_errors=tool_errors,
        timed_out=timed_out,
    )
