from src.agent.actions import Action, EXPANSION_ACTIONS, RECOVERY_ACTIONS
from src.agent.state import AgentState


def test_expansion_and_recovery_actions_partition_llm_vs_control():
    assert EXPANSION_ACTIONS.isdisjoint(RECOVERY_ACTIONS)
    assert Action.NO_EXPAND not in EXPANSION_ACTIONS
    assert Action.NO_EXPAND not in RECOVERY_ACTIONS


def test_agent_state_budget_decrements_and_exhausts():
    state = AgentState(q0="q", q_t="q", R_t=[], O_t=None, B_t=1)
    assert not state.exhausted
    state.record(Action.VOCABULARY, observation=None, accepted=True)
    assert state.exhausted
    assert len(state.H_t) == 1
