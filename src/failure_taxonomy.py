"""The compact failure taxonomy as code (docs/failure_taxonomy.md), so the
controller and evaluation scripts share one source of truth for
labels/recoveries instead of duplicating the table in prose.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.agent.actions import Action


class FailureLabel(str, Enum):
    VOCABULARY_MISMATCH = "vocabulary_mismatch"
    UNDERSPECIFICATION = "underspecification"
    AMBIGUITY = "ambiguity"
    ENTITY_MISMATCH = "entity_mismatch"
    INTENT_DRIFT = "intent_drift"
    CONFIRMATION_BIAS = "confirmation_bias"
    RETRIEVER_DISAGREEMENT = "retriever_disagreement"
    TOOL_DEGRADATION = "tool_degradation"
    BUDGET_EXHAUSTION = "budget_exhaustion"
    HEALTHY = "healthy"


@dataclass(frozen=True)
class FailureProfile:
    label: FailureLabel
    signature: str
    preferred_recovery: tuple[Action, ...]


TAXONOMY: dict[FailureLabel, FailureProfile] = {
    FailureLabel.VOCABULARY_MISMATCH: FailureProfile(
        FailureLabel.VOCABULARY_MISMATCH,
        "low lexical support; semantic evidence stronger",
        (Action.VOCABULARY,),
    ),
    FailureLabel.UNDERSPECIFICATION: FailureProfile(
        FailureLabel.UNDERSPECIFICATION,
        "weak separation, dispersed results, very short query",
        (Action.CONTEXT,),
    ),
    FailureLabel.AMBIGUITY: FailureProfile(
        FailureLabel.AMBIGUITY,
        "multiple coherent but divergent result clusters",
        (Action.AMBIGUITY,),
    ),
    FailureLabel.ENTITY_MISMATCH: FailureProfile(
        FailureLabel.ENTITY_MISMATCH,
        "alias/acronym/canonical-name disagreement",
        (Action.ENTITY_NORMALIZE,),
    ),
    FailureLabel.INTENT_DRIFT: FailureProfile(
        FailureLabel.INTENT_DRIFT,
        "expansion diverges from original; ranking shifts sharply",
        (Action.ROLLBACK,),
    ),
    FailureLabel.CONFIRMATION_BIAS: FailureProfile(
        FailureLabel.CONFIRMATION_BIAS,
        "expansion amplifies evidence from a weak initial ranking",
        (Action.ROLLBACK, Action.REPLAN),
    ),
    FailureLabel.RETRIEVER_DISAGREEMENT: FailureProfile(
        FailureLabel.RETRIEVER_DISAGREEMENT,
        "sparse and semantic rankings strongly disagree",
        (Action.REPLAN,),
    ),
    FailureLabel.TOOL_DEGRADATION: FailureProfile(
        FailureLabel.TOOL_DEGRADATION,
        "timeout, truncated candidates, missing reranking",
        (Action.STOP,),
    ),
    FailureLabel.BUDGET_EXHAUSTION: FailureProfile(
        FailureLabel.BUDGET_EXHAUSTION,
        "additional action yields little state improvement",
        (Action.STOP,),
    ),
    FailureLabel.HEALTHY: FailureProfile(
        FailureLabel.HEALTHY,
        "stable, supported, high-confidence state",
        (Action.NO_EXPAND,),
    ),
}
