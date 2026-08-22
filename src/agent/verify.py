"""verify(previous_state, new_state) -> score. Qrel-free: judges whether
the new retrieval state is healthier than the previous one using
telemetry only (score margin/entropy gain, coherence, stability, drift,
disagreement) -- this is what makes the verifier usable at inference
time, when qrels aren't available.
"""
from __future__ import annotations

from src.agent.state import AgentState


def verify(previous_state: AgentState, new_state: AgentState) -> float:
    """Returns a verifier score; higher means the new state is more
    likely to be a retrieval-quality improvement over previous_state.
    """
    raise NotImplementedError(
        "TODO: telemetry-only comparison of previous_state.O_t vs. new_state.O_t"
    )
