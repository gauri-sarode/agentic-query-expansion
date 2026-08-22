"""The agent's action space. Deliberately small and interpretable: the LLM
is one typed tool the agent can invoke (via EXPANSION_ACTIONS), not the
planner, retriever, judge, and memory manager at once.
"""
from __future__ import annotations

from enum import Enum


class Action(str, Enum):
    NO_EXPAND = "NO_EXPAND"
    VOCABULARY = "VOCABULARY"
    CONTEXT = "CONTEXT"
    AMBIGUITY = "AMBIGUITY"
    ENTITY_NORMALIZE = "ENTITY_NORMALIZE"
    ROLLBACK = "ROLLBACK"
    REPLAN = "REPLAN"
    STOP = "STOP"


# Actions that invoke generate_expansion (an LLM call against the corpus
# evidence). NO_EXPAND, ROLLBACK, REPLAN, STOP are control decisions and
# never call the LLM.
EXPANSION_ACTIONS = frozenset(
    {Action.VOCABULARY, Action.CONTEXT, Action.AMBIGUITY, Action.ENTITY_NORMALIZE}
)

# Terminal recovery decisions the verifier can choose after observing the
# consequences of an expansion action.
RECOVERY_ACTIONS = frozenset({Action.ROLLBACK, Action.REPLAN, Action.STOP})
