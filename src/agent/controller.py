"""Action selection and recovery control. Deliberately not the LLM:

- Initial-action selection is a small learned classifier (logistic
  regression / GBDT) over retrieval-state telemetry -- given O_t at
  q_0's initial ranking, which operator (or NO_EXPAND) is likely to help.
- Recovery control (accept / ROLLBACK / REPLAN / STOP after observing an
  expansion's consequences) is a deterministic finite-state controller
  driven by verifier score + budget + fault signals, per
  docs/failure_taxonomy.md, plus a small learned acceptance threshold.

Keeping both outside the LLM keeps the policy inspectable and keeps the
LLM confined to a single typed tool (generate_expansion).
"""
from __future__ import annotations

from src.agent.actions import Action
from src.agent.state import AgentState
from src.observability.telemetry import ObservabilityState


class ActionSelector:
    """Chooses the initial action from O_t (pre-expansion telemetry)."""

    def select(self, state: AgentState) -> Action:
        raise NotImplementedError(
            "TODO: logistic regression / GBDT over retrieval telemetry, "
            "trained against the failure taxonomy labels"
        )


class RecoveryController:
    """Finite-state controller: given post-action O_t and the verifier
    score, decide ACCEPT / ROLLBACK / REPLAN / STOP.
    """

    def decide(
        self, state: AgentState, observation: ObservabilityState, verifier_score: float
    ) -> Action:
        if state.exhausted:
            return Action.STOP
        raise NotImplementedError(
            "TODO: deterministic rules (docs/failure_taxonomy.md) + learned "
            "acceptance threshold on verifier_score"
        )
