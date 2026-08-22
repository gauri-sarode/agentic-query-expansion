"""Agent state s_t = (q0, q_t, R_t, O_t, H_t, B_t). See README System design."""
from __future__ import annotations

from dataclasses import dataclass, field

from src.agent.actions import Action
from src.observability.telemetry import ObservabilityState

Ranking = list[str]  # ordered doc IDs


@dataclass(frozen=True)
class HistoryStep:
    action: Action
    observation: ObservabilityState | None
    accepted: bool


@dataclass
class AgentState:
    q0: str  # immutable original query
    q_t: str  # active search expression
    R_t: Ranking
    O_t: ObservabilityState | None
    H_t: list[HistoryStep] = field(default_factory=list)
    B_t: int = 3  # remaining compute/action budget

    def record(self, action: Action, observation: ObservabilityState, accepted: bool) -> None:
        self.H_t.append(HistoryStep(action=action, observation=observation, accepted=accepted))
        self.B_t -= 1

    @property
    def exhausted(self) -> bool:
        return self.B_t <= 0
