"""Failure-conditioned static QE: the strong non-agentic comparator from
the experiment matrix (README). Same operator-selection policy, expansion
generator, verifier, and fusion mechanism as the closed-loop agent
(src/agent/loop.py, both built on src/agent/steps.py) -- the only
difference under test is agency itself:

  - initial retrieval state selects ONE operator (or NO_EXPAND), exactly
    like the agent's first action;
  - the verifier gates a single accept/reject decision;
  - there is no ROLLBACK/REPLAN loop, no multi-step recovery, and never
    more than one LLM call.

This isolates whether the *closed-loop* part of the design (observing an
action's consequences and recovering from a bad one) adds anything beyond
a single-shot "diagnose once, expand once, verify once" pipeline.
"""
from __future__ import annotations

import dataclasses
import time
import uuid

from src.agent.actions import EXPANSION_ACTIONS
from src.agent.controller import ActionSelector, RecoveryThresholds
from src.agent.fuse import fuse
from src.agent.state import AgentState
from src.agent.steps import act_once, build_evidence, retrieve_and_observe_initial
from src.config import CONFIG
from src.llm_client import LLMUnavailableError
from src.observability.trace_schema import TraceRecord


def run_static_episode(
    query_id: str,
    q0: str,
    db_path: str,
    action_selector: ActionSelector | None = None,
    accept_threshold: float = RecoveryThresholds().accept,
    dataset_slice: str = "dev",
    k: int = 100,
    use_reranker: bool = True,
) -> tuple[list[str], TraceRecord]:
    action_selector = action_selector or ActionSelector()
    episode_t0 = time.time()

    init = retrieve_and_observe_initial(q0, db_path, k)
    state = AgentState(q0=q0, q_t=q0, R_t=init.r0_ids, O_t=init.o0, B_t=1)

    action = action_selector.select(q0, init.o0)

    final_ranking = init.r0_ids
    generated_expansion: str | None = None
    verifier_score: float | None = None
    controller_decision = "NO_EXPAND"
    bm25_latency_ms = init.bm25_latency_ms
    llm_latency_ms = 0.0
    reranker_latency_ms = 0.0
    tool_errors = 0
    llm_calls = 0
    final_o = init.o0

    if action in EXPANSION_ACTIONS:
        evidence = build_evidence(init.r0_ids, db_path)
        llm_calls = 1
        try:
            result = act_once(action, state, init.r0, db_path, evidence, k, use_reranker)
        except LLMUnavailableError:
            tool_errors += 1
            controller_decision = "REJECT"
        else:
            generated_expansion = result.generated_expansion
            bm25_latency_ms += result.bm25_latency_ms
            reranker_latency_ms += result.reranker_latency_ms
            llm_latency_ms += result.llm_latency_ms
            tool_errors += result.tool_errors
            verifier_score = result.verifier_score
            final_o = result.candidate.O_t

            if verifier_score >= accept_threshold:
                final_ranking = fuse(init.r0_ids, result.candidate.R_t)
                controller_decision = "ACCEPT"
            else:
                controller_decision = "REJECT"

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
        state_features_t1=dataclasses.asdict(final_o),
        reranker_span_ms=reranker_latency_ms,
        verifier_score=verifier_score,
        fault_type=None,
        controller_decision=controller_decision,
        rollback_or_retry=False,  # no retry mechanism exists in the static pipeline
        llm_calls=llm_calls,
        final_run_id=str(uuid.uuid4()),
        episode_latency_ms=(time.time() - episode_t0) * 1000,
    )
    return final_ranking, trace
