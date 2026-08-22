"""The closed-loop control episode:

OBSERVE -> DIAGNOSE -> ACT -> RETRIEVE -> OBSERVE -> VERIFY
        -> {ACCEPT, ROLLBACK, REPLAN, STOP}

Only typed tools are called (retrieve, rerank, extract_evidence,
generate_expansion, verify, fuse) -- see README System design and
docs/observability_model.md for the corresponding trace spans.
"""
from __future__ import annotations

import dataclasses
import time
import uuid

from src.agent.actions import EXPANSION_ACTIONS, Action
from src.agent.controller import ActionSelector, RecoveryController
from src.agent.fuse import fuse
from src.agent.state import AgentState, HistoryStep
from src.agent.verify import verify
from src.config import CONFIG
from src.expansion.operators import generate_expansion
from src.llm_client import LLMUnavailableError
from src.observability.telemetry import (
    ObservabilityState,
    compute_agent_telemetry,
    compute_retrieval_telemetry,
    compute_system_telemetry,
)
from src.observability.trace_schema import TraceRecord
from src.retrieval import bm25_index

# Order REPLAN cycles through when the current operator's rewrite doesn't verify.
_REPLAN_ORDER = [Action.VOCABULARY, Action.CONTEXT, Action.AMBIGUITY, Action.ENTITY_NORMALIZE]


def _try_rerank(query: str, ranking: list[tuple[str, float]], db_path: str) -> tuple[list[str] | None, float, int]:
    """Best-effort cross-encoder rerank of the top candidates. A missing
    reranker (e.g. sentence-transformers not installed) or a runtime
    failure degrades gracefully to "no reranker signal" rather than
    aborting the episode -- this *is* the tool_degradation failure mode
    in docs/failure_taxonomy.md, not a bug to hide.
    """
    t0 = time.time()
    try:
        from src.retrieval.rerank import rerank as cross_encoder_rerank

        candidate_ids = [d for d, _ in ranking[: CONFIG.reranker_max_candidates]]
        texts = bm25_index.get_texts(candidate_ids, db_path)
        pairs = [(d, texts[d]) for d in candidate_ids if d in texts]
        reranked = cross_encoder_rerank(query, pairs, top_k=CONFIG.reranker_top_k)
        return [d for d, _ in reranked], (time.time() - t0) * 1000, 0
    except Exception:
        return None, (time.time() - t0) * 1000, 1


def run_episode(
    query_id: str,
    q0: str,
    db_path: str,
    action_selector: ActionSelector | None = None,
    recovery_controller: RecoveryController | None = None,
    dataset_slice: str = "dev",
    k: int = 100,
    use_reranker: bool = True,
) -> tuple[list[str], TraceRecord]:
    action_selector = action_selector or ActionSelector()
    recovery_controller = recovery_controller or RecoveryController()
    episode_t0 = time.time()

    # retrieve_original
    bm25_t0 = time.time()
    r0 = bm25_index.search(q0, db_path, k=k)
    bm25_latency_ms = (time.time() - bm25_t0) * 1000
    r0_ids = [d for d, _ in r0]

    # observe (t0)
    o0_retrieval = compute_retrieval_telemetry(q0, r0, db_path)
    o0_agent = compute_agent_telemetry(q0, q0, "", [], CONFIG.action_budget, None, None, False)
    o0_system = compute_system_telemetry(bm25_latency_ms=bm25_latency_ms)
    o0 = ObservabilityState(retrieval=o0_retrieval, agent=o0_agent, system=o0_system)

    state = AgentState(q0=q0, q_t=q0, R_t=r0_ids, O_t=o0, B_t=CONFIG.action_budget)

    # diagnose -> act (initial operator selection)
    action = action_selector.select(q0, o0)

    final_ranking = r0_ids
    generated_expansion: str | None = None
    verifier_score: float | None = None
    controller_decision = "NO_EXPAND"
    rollback_or_retry = False
    llm_latency_ms = 0.0
    reranker_latency_ms = 0.0
    tool_errors = 0
    llm_calls = 0

    if action in EXPANSION_ACTIONS:
        # build_evidence: top-5 original-ranking passages ground the expansion.
        evidence_ids = r0_ids[:5]
        evidence_map = bm25_index.get_texts(evidence_ids, db_path)
        evidence = [evidence_map[d] for d in evidence_ids if d in evidence_map]

        tried: set[Action] = set()
        while action in EXPANSION_ACTIONS and llm_calls < CONFIG.max_llm_calls_per_query and not state.exhausted:
            tried.add(action)
            llm_calls += 1

            llm_t0 = time.time()
            try:
                q_t = generate_expansion(action, state.q0, evidence)
            except LLMUnavailableError:
                tool_errors += 1
                controller_decision = "STOP"
                rollback_or_retry = False
                break
            llm_latency_ms += (time.time() - llm_t0) * 1000
            generated_expansion = q_t

            # retrieve_expanded
            bm25_t1 = time.time()
            r1 = bm25_index.search(q_t, db_path, k=k)
            bm25_latency_ms += (time.time() - bm25_t1) * 1000
            r1_ids = [d for d, _ in r1]

            # rerank
            reranked_order = None
            if use_reranker:
                reranked_order, rr_ms, rr_errors = _try_rerank(q_t, r1, db_path)
                reranker_latency_ms += rr_ms
                tool_errors += rr_errors

            # observe (t1)
            o1_retrieval = compute_retrieval_telemetry(
                q_t, r1, db_path, previous_ranking=r0, reranked_order=reranked_order
            )
            o1_agent = compute_agent_telemetry(
                state.q0, q_t, action.value, state.H_t, state.B_t - 1, verifier_score, None, False
            )
            o1_system = compute_system_telemetry(
                bm25_latency_ms=bm25_latency_ms,
                reranker_latency_ms=reranker_latency_ms,
                llm_latency_ms=llm_latency_ms,
                tool_errors=tool_errors,
            )
            candidate = AgentState(
                q0=state.q0,
                q_t=q_t,
                R_t=r1_ids,
                O_t=ObservabilityState(retrieval=o1_retrieval, agent=o1_agent, system=o1_system),
                H_t=list(state.H_t),
                B_t=state.B_t - 1,
            )

            # verify
            verifier_score = verify(state, candidate)
            decision = recovery_controller.decide(candidate, verifier_score)
            candidate.H_t.append(HistoryStep(action=action, observation=candidate.O_t, accepted=decision is None))

            if decision is None:
                # accept -> fuse
                final_ranking = fuse(r0_ids, r1_ids)
                controller_decision = "ACCEPT"
                state = candidate
                break
            if decision is Action.ROLLBACK:
                rollback_or_retry = True
                final_ranking = r0_ids
                controller_decision = "ROLLBACK"
                state = candidate
                break
            if decision is Action.STOP:
                final_ranking = r0_ids  # abstain: fall back to the verified-safe original
                controller_decision = "STOP"
                state = candidate
                break
            # REPLAN: try the next untried operator, if any remain within budget.
            rollback_or_retry = True
            state = candidate
            remaining = [a for a in _REPLAN_ORDER if a not in tried]
            if not remaining:
                final_ranking = r0_ids
                controller_decision = "STOP"
                break
            action = remaining[0]
            controller_decision = "REPLAN"

    trace = TraceRecord(
        query_id=query_id,
        dataset_slice=dataset_slice,
        state_features_t0=dataclasses.asdict(o0),
        retrieval_span_ms=bm25_latency_ms,
        planner_action=action.value,
        evidence_doc_ids=r0_ids[:5],
        prompt_hash=None,
        model_id=CONFIG.expansion_model if llm_calls else None,
        generated_expansion=generated_expansion,
        prompt_tokens=0,
        output_tokens=0,
        state_features_t1=dataclasses.asdict(state.O_t) if state.O_t else dataclasses.asdict(o0),
        reranker_span_ms=reranker_latency_ms,
        verifier_score=verifier_score,
        fault_type=None,
        controller_decision=controller_decision,
        rollback_or_retry=rollback_or_retry,
        final_run_id=str(uuid.uuid4()),
        episode_latency_ms=(time.time() - episode_t0) * 1000,
    )
    return final_ranking, trace
