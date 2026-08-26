from src.agent.actions import Action
from src.agent.controller import ActionSelector, RecoveryController, RecoveryThresholds, SelectorThresholds
from src.agent.fuse import fuse
from src.agent.state import AgentState
from src.agent.verify import verify
from src.observability.telemetry import (
    AgentTelemetry,
    ObservabilityState,
    RetrievalTelemetry,
    SystemTelemetry,
)


def _retrieval(**overrides):
    defaults = dict(
        top_score=10.0,
        score_margin=1.0,
        score_entropy=0.5,
        query_term_coverage=0.8,
        mean_rare_term_idf=2.0,
        entity_overlap=1.0,
        topk_overlap=1.0,
        reranker_bm25_disagreement=0.0,
        result_coherence=0.5,
        ranking_stability=1.0,
        reranker_top_score=0.0,
        reranker_score_margin=0.0,
    )
    defaults.update(overrides)
    return RetrievalTelemetry(**defaults)


def _agent(**overrides):
    defaults = dict(
        selected_operator="",
        trajectory_length=0,
        previous_verifier_score=None,
        rewrite_semantic_drift=0.0,
        introduced_entities=0,
        query_length_ratio=1.0,
        repeated_actions=0,
        remaining_budget=3,
        accepted=None,
        rolled_back=False,
    )
    defaults.update(overrides)
    return AgentTelemetry(**defaults)


def _system(**overrides):
    defaults = dict(
        bm25_latency_ms=1.0,
        reranker_latency_ms=0.0,
        llm_latency_ms=0.0,
        prompt_tokens=0,
        output_tokens=0,
        tool_errors=0,
        timed_out=False,
    )
    defaults.update(overrides)
    return SystemTelemetry(**defaults)


def _obs(retrieval=None, agent=None, system=None):
    return ObservabilityState(
        retrieval=retrieval or _retrieval(), agent=agent or _agent(), system=system or _system()
    )


# --- ActionSelector -----------------------------------------------------


def test_selector_picks_entity_normalize_when_entity_missing():
    selector = ActionSelector()
    obs = _obs(retrieval=_retrieval(entity_overlap=0.0))
    assert selector.select("WHO guidelines on diabetes", obs) == Action.ENTITY_NORMALIZE


def test_selector_picks_vocabulary_when_coverage_low():
    selector = ActionSelector()
    obs = _obs(retrieval=_retrieval(query_term_coverage=0.1))
    assert selector.select("informal phrasing here", obs) == Action.VOCABULARY


def test_selector_picks_context_for_short_query():
    selector = ActionSelector()
    obs = _obs(retrieval=_retrieval(query_term_coverage=0.9))
    assert selector.select("diabetes", obs) == Action.CONTEXT


def test_selector_picks_ambiguity_for_low_coherence():
    selector = ActionSelector()
    obs = _obs(retrieval=_retrieval(query_term_coverage=0.9, result_coherence=0.05))
    assert selector.select("jaguar speed comparison chart", obs) == Action.AMBIGUITY


def test_selector_picks_no_expand_when_healthy():
    selector = ActionSelector()
    obs = _obs(retrieval=_retrieval(query_term_coverage=0.9, result_coherence=0.8, score_entropy=0.2))
    assert selector.select("statin drugs cholesterol treatment", obs) == Action.NO_EXPAND


def test_selector_respects_custom_thresholds():
    selector = ActionSelector(SelectorThresholds(min_term_coverage=0.95))
    obs = _obs(retrieval=_retrieval(query_term_coverage=0.9, result_coherence=0.8, score_entropy=0.2))
    assert selector.select("statin drugs cholesterol treatment", obs) == Action.VOCABULARY


# --- RecoveryController --------------------------------------------------


def test_recovery_stops_when_budget_exhausted():
    controller = RecoveryController()
    state = AgentState(q0="q", q_t="q", R_t=[], O_t=None, B_t=0)
    assert controller.decide(state, verifier_score=1.0) == Action.STOP


def test_recovery_accepts_above_threshold():
    controller = RecoveryController()
    state = AgentState(q0="q", q_t="q", R_t=[], O_t=None, B_t=2)
    assert controller.decide(state, verifier_score=0.5) is None


def test_recovery_rolls_back_below_harmful_threshold():
    controller = RecoveryController()
    state = AgentState(q0="q", q_t="q", R_t=[], O_t=None, B_t=2)
    assert controller.decide(state, verifier_score=-1.5) == Action.ROLLBACK


def test_recovery_replans_on_mildly_negative_score():
    # v1 verify() scale is small (~-0.1 to +0.03); the calibrated band
    # between harmful (-0.06) and accept (-0.036) should get a REPLAN
    # chance rather than an immediate ROLLBACK.
    controller = RecoveryController()
    state = AgentState(q0="q", q_t="q", R_t=[], O_t=None, B_t=2)
    assert controller.decide(state, verifier_score=-0.05) == Action.REPLAN


def test_recovery_replans_in_between():
    controller = RecoveryController(RecoveryThresholds(accept=0.5, harmful=-0.5))
    state = AgentState(q0="q", q_t="q", R_t=[], O_t=None, B_t=2)
    assert controller.decide(state, verifier_score=0.0) == Action.REPLAN


# --- verify ---------------------------------------------------------------


def test_verify_rewards_improvement_and_penalizes_drift():
    prev = AgentState(q0="q", q_t="q", R_t=[], O_t=_obs(), B_t=3)

    better = _obs(
        retrieval=_retrieval(score_margin=2.0, query_term_coverage=0.95, result_coherence=0.7),
        agent=_agent(rewrite_semantic_drift=0.0),
    )
    worse = _obs(
        retrieval=_retrieval(score_margin=0.0, query_term_coverage=0.1, result_coherence=0.1),
        agent=_agent(rewrite_semantic_drift=0.9),
    )

    better_state = AgentState(q0="q", q_t="q2", R_t=[], O_t=better, B_t=2)
    worse_state = AgentState(q0="q", q_t="q3", R_t=[], O_t=worse, B_t=2)

    assert verify(prev, better_state) > verify(prev, worse_state)


# --- fuse -------------------------------------------------------------


def test_fuse_ranks_docs_present_in_both_lists_highest():
    original = ["a", "b", "c"]
    accepted = ["b", "d", "a"]
    fused = fuse(original, accepted)
    # "a" and "b" both appear near the top of both lists -> should lead the fused ranking.
    assert set(fused[:2]) == {"a", "b"}
