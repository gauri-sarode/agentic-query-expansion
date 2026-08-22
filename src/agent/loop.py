"""The closed-loop control episode:

OBSERVE -> DIAGNOSE -> ACT -> RETRIEVE -> OBSERVE -> VERIFY
        -> {ACCEPT, ROLLBACK, REPLAN, STOP}

Only typed tools are called (retrieve, rerank, extract_evidence,
generate_expansion, verify, fuse) -- see README System design and
docs/observability_model.md for the corresponding trace spans.
"""
from __future__ import annotations

from src.agent.actions import Action
from src.agent.controller import ActionSelector, RecoveryController
from src.agent.state import AgentState
from src.observability.trace_schema import TraceRecord


def run_episode(
    query_id: str,
    q0: str,
    action_selector: ActionSelector,
    recovery_controller: RecoveryController,
) -> tuple[AgentState, list[TraceRecord]]:
    raise NotImplementedError(
        "TODO: orchestrate retrieve_original -> build_evidence -> plan_action "
        "-> generate_expansion -> retrieve_expanded -> rerank -> verify -> "
        "recover_or_accept -> fuse, emitting one TraceRecord per episode"
    )
