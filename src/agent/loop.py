"""The closed-loop control episode:

OBSERVE -> DIAGNOSE -> ACT -> RETRIEVE -> OBSERVE -> VERIFY
        -> {ACCEPT, ROLLBACK, REPLAN, STOP}

Only typed tools are called (retrieve, rerank, extract_evidence,
generate_expansion, verify, fuse) -- see README System design and
docs/observability_model.md for the corresponding trace spans. Shared
steps (retrieve/observe/act/verify) live in src/agent/steps.py so this
and the static comparator (src/agent/static_baseline.py) run identical
component code -- see that module's docstring for why that matters.
"""
from __future__ import annotations

import dataclasses
import time
import uuid

from src.agent.actions import EXPANSION_ACTIONS, Action
from src.agent.controller import ActionSelector, RecoveryController
from src.agent.fuse import fuse
from src.agent.state import AgentState, HistoryStep
from src.agent.steps import act_once, build_evidence, retrieve_and_observe_initial
from src.config import CONFIG
from src.llm_client import LLMUnavailableError
from src.observability.trace_schema import TraceRecord

# Order REPLAN cycles through when the current operator's rewrite doesn't verify.
_REPLAN_ORDER = [Action.VOCABULARY, Action.CONTEXT, Action.AMBIGUITY, Action.ENTITY_NORMALIZE]


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

    init = retrieve_and_observe_initial(q0, db_path, k)
    state = AgentState(q0=q0, q_t=q0, R_t=init.r0_ids, O_t=init.o0, B_t=CONFIG.action_budget)

    # diagnose -> act (initial operator selection)
    action = action_selector.select(q0, init.o0)

    final_ranking = init.r0_ids
    generated_expansion: str | None = None
    verifier_score: float | None = None
    controller_decision = "NO_EXPAND"
    rollback_or_retry = False
    bm25_latency_ms = init.bm25_latency_ms
    llm_latency_ms = 0.0
    reranker_latency_ms = 0.0
    tool_errors = 0
    llm_calls = 0

    if action in EXPANSION_ACTIONS:
        evidence = build_evidence(init.r0_ids, db_path)
        tried: set[Action] = set()

        while action in EXPANSION_ACTIONS and llm_calls < CONFIG.max_llm_calls_per_query and not state.exhausted:
            tried.add(action)
            llm_calls += 1

            try:
                result = act_once(action, state, init.r0, db_path, evidence, k, use_reranker)
            except LLMUnavailableError:
                tool_errors += 1
                controller_decision = "STOP"
                rollback_or_retry = False
                break

            generated_expansion = result.generated_expansion
            bm25_latency_ms += result.bm25_latency_ms
            reranker_latency_ms += result.reranker_latency_ms
            llm_latency_ms += result.llm_latency_ms
            tool_errors += result.tool_errors
            verifier_score = result.verifier_score
            candidate = result.candidate

            decision = recovery_controller.decide(candidate, verifier_score)
            candidate.H_t.append(HistoryStep(action=action, observation=candidate.O_t, accepted=decision is None))

            if decision is None:
                # accept -> fuse
                final_ranking = fuse(init.r0_ids, candidate.R_t)
                controller_decision = "ACCEPT"
                state = candidate
                break
            if decision is Action.ROLLBACK:
                rollback_or_retry = True
                final_ranking = init.r0_ids
                controller_decision = "ROLLBACK"
                state = candidate
                break
            if decision is Action.STOP:
                final_ranking = init.r0_ids  # abstain: fall back to the verified-safe original
                controller_decision = "STOP"
                state = candidate
                break
            # REPLAN: try the next untried operator, if any remain within budget.
            rollback_or_retry = True
            state = candidate
            remaining = [a for a in _REPLAN_ORDER if a not in tried]
            if not remaining:
                final_ranking = init.r0_ids
                controller_decision = "STOP"
                break
            action = remaining[0]
            controller_decision = "REPLAN"

    trace = TraceRecord(
        query_id=query_id,
        dataset_slice=dataset_slice,
        state_features_t0=dataclasses.asdict(init.o0),
        retrieval_span_ms=bm25_latency_ms,
        planner_action=action.value,
        evidence_doc_ids=init.r0_ids[:5],
        prompt_hash=None,
        model_id=CONFIG.expansion_model if llm_calls else None,
        generated_expansion=generated_expansion,
        prompt_tokens=0,
        output_tokens=0,
        state_features_t1=dataclasses.asdict(state.O_t) if state.O_t else dataclasses.asdict(init.o0),
        reranker_span_ms=reranker_latency_ms,
        verifier_score=verifier_score,
        fault_type=None,
        controller_decision=controller_decision,
        rollback_or_retry=rollback_or_retry,
        llm_calls=llm_calls,
        final_run_id=str(uuid.uuid4()),
        episode_latency_ms=(time.time() - episode_t0) * 1000,
    )
    return final_ranking, trace
