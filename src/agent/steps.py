"""Shared episode steps used by both the closed-loop agent
(src/agent/loop.py) and the static failure-conditioned QE comparator
(src/agent/static_baseline.py).

This isn't just deduplication: the experiment matrix's critical
comparison ("observable agent vs strong static retrieval-aware QE using
the same components", README) is only valid if both pipelines actually
run the same retriever, grounding, generator, reranker, and fusion code
-- sharing this module is what enforces that, rather than two
independently-written implementations silently drifting apart.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from src.agent.actions import Action
from src.agent.state import AgentState
from src.agent.verify import verify
from src.config import CONFIG
from src.expansion.operators import generate_expansion
from src.observability.telemetry import (
    ObservabilityState,
    compute_agent_telemetry,
    compute_retrieval_telemetry,
    compute_system_telemetry,
)
from src.retrieval import bm25_index


def try_rerank(
    query: str, ranking: list[tuple[str, float]], db_path: str
) -> tuple[list[tuple[str, float]] | None, float, int]:
    """Best-effort cross-encoder rerank of the top candidates. Returns
    scored (doc_id, score) pairs, not just an order -- the raw
    cross-encoder scores are a real relevance-model signal (unlike the
    lexical BM25-only features), worth keeping for telemetry rather than
    discarding down to just a reordering.

    A missing reranker (e.g. sentence-transformers not installed) or a
    runtime failure degrades gracefully to "no reranker signal" rather
    than aborting the episode -- this *is* the tool_degradation failure
    mode in docs/failure_taxonomy.md, not a bug to hide.
    """
    t0 = time.time()
    try:
        from src.retrieval.rerank import rerank as cross_encoder_rerank

        candidate_ids = [d for d, _ in ranking[: CONFIG.reranker_max_candidates]]
        texts = bm25_index.get_texts(candidate_ids, db_path)
        pairs = [(d, texts[d]) for d in candidate_ids if d in texts]
        reranked = cross_encoder_rerank(query, pairs, top_k=CONFIG.reranker_top_k)
        return [(d, float(s)) for d, s in reranked], (time.time() - t0) * 1000, 0
    except Exception:
        return None, (time.time() - t0) * 1000, 1


@dataclass
class InitialObservation:
    r0: list[tuple[str, float]]
    r0_ids: list[str]
    o0: ObservabilityState
    bm25_latency_ms: float


def retrieve_and_observe_initial(q0: str, db_path: str, k: int) -> InitialObservation:
    """retrieve_original -> observe (t0). Shared first step of every episode."""
    bm25_t0 = time.time()
    r0 = bm25_index.search(q0, db_path, k=k)
    bm25_latency_ms = (time.time() - bm25_t0) * 1000
    r0_ids = [d for d, _ in r0]

    o0_retrieval = compute_retrieval_telemetry(q0, r0, db_path)
    o0_agent = compute_agent_telemetry(q0, q0, "", [], CONFIG.action_budget, None, None, False)
    o0_system = compute_system_telemetry(bm25_latency_ms=bm25_latency_ms)
    o0 = ObservabilityState(retrieval=o0_retrieval, agent=o0_agent, system=o0_system)

    return InitialObservation(r0=r0, r0_ids=r0_ids, o0=o0, bm25_latency_ms=bm25_latency_ms)


def build_evidence(r0_ids: list[str], db_path: str, n: int = 5) -> list[str]:
    """extract_evidence: top-n original-ranking passages ground the expansion."""
    evidence_ids = r0_ids[:n]
    evidence_map = bm25_index.get_texts(evidence_ids, db_path)
    return [evidence_map[d] for d in evidence_ids if d in evidence_map]


@dataclass
class ActStepResult:
    candidate: AgentState
    verifier_score: float
    generated_expansion: str
    bm25_latency_ms: float
    reranker_latency_ms: float
    llm_latency_ms: float
    tool_errors: int


def act_once(
    action: Action,
    state: AgentState,
    r0: list[tuple[str, float]],
    db_path: str,
    evidence: list[str],
    k: int,
    use_reranker: bool,
) -> ActStepResult:
    """generate_expansion -> retrieve_expanded -> rerank -> observe (t1) ->
    verify. One full act/observe/verify cycle for a single expansion
    action. Raises LLMUnavailableError (uncaught) if generation fails --
    callers decide what that means for their own control flow (agent:
    STOP; static baseline: REJECT).
    """
    llm_t0 = time.time()
    q_t = generate_expansion(action, state.q0, evidence)
    llm_latency_ms = (time.time() - llm_t0) * 1000

    bm25_t1 = time.time()
    r1 = bm25_index.search(q_t, db_path, k=k)
    bm25_latency_ms = (time.time() - bm25_t1) * 1000
    r1_ids = [d for d, _ in r1]

    reranked, reranker_latency_ms, tool_errors = (None, 0.0, 0)
    if use_reranker:
        reranked, reranker_latency_ms, tool_errors = try_rerank(q_t, r1, db_path)
    reranked_order = [d for d, _ in reranked] if reranked else None

    # The reranked top-k should actually lead the accepted ranking, not
    # just feed telemetry -- previously R_t was unconditionally raw BM25
    # order even when reranking succeeded, so the reranker never
    # influenced anything but its own disagreement/score telemetry (see
    # git history: this was found while investigating why a
    # disable_reranker fault-injection test showed almost no effect --
    # disabling it changed nothing because it was never wired into the
    # outcome even when enabled). Remaining BM25-order candidates beyond
    # the reranked top-k are appended after, to preserve full Recall@100
    # depth rather than truncating to just the reranked subset.
    if reranked_order:
        reranked_set = set(reranked_order)
        final_ids = reranked_order + [d for d in r1_ids if d not in reranked_set]
    else:
        final_ids = r1_ids

    r0_ids = [d for d, _ in r0]
    o1_retrieval = compute_retrieval_telemetry(
        q_t, r1, db_path, previous_ranking=r0, reranked_order=reranked_order, reranked_scores=reranked
    )
    o1_agent = compute_agent_telemetry(state.q0, q_t, action.value, state.H_t, state.B_t - 1, None, None, False)
    o1_system = compute_system_telemetry(
        bm25_latency_ms=bm25_latency_ms,
        reranker_latency_ms=reranker_latency_ms,
        llm_latency_ms=llm_latency_ms,
        tool_errors=tool_errors,
    )
    candidate = AgentState(
        q0=state.q0,
        q_t=q_t,
        R_t=final_ids,
        O_t=ObservabilityState(retrieval=o1_retrieval, agent=o1_agent, system=o1_system),
        H_t=list(state.H_t),
        B_t=state.B_t - 1,
    )

    verifier_score = verify(state, candidate)

    return ActStepResult(
        candidate=candidate,
        verifier_score=verifier_score,
        generated_expansion=q_t,
        bm25_latency_ms=bm25_latency_ms,
        reranker_latency_ms=reranker_latency_ms,
        llm_latency_ms=llm_latency_ms,
        tool_errors=tool_errors,
    )
