from unittest.mock import patch

from src.agent.actions import Action
from src.agent.state import AgentState
from src.agent.steps import act_once
from src.observability.telemetry import ObservabilityState
from src.retrieval.bm25_index import build_index

_DOCS = [
    ("d1", "cholesterol statin drugs breast cancer risk study"),
    ("d2", "cholesterol statin drugs cardiovascular disease outcomes"),
    ("d3", "unrelated topic about gardening and plants"),
    ("d4", "breast cancer treatment options and outcomes"),
    ("d5", "cholesterol levels and diet in older adults"),
]


def _build(tmp_path):
    db_path = str(tmp_path / "test.sqlite3")
    build_index(_DOCS, db_path)
    return db_path


def _fake_o0():
    from tests.test_agent_control import _agent, _obs, _retrieval

    return _obs(retrieval=_retrieval(), agent=_agent())


def test_act_once_reranked_order_leads_the_final_ranking(tmp_path):
    """Regression test: R_t used to always be raw BM25 order even when
    reranking succeeded, so the reranker never actually influenced the
    accepted ranking -- only its own telemetry. Found via a
    disable_reranker fault-injection test showing near-zero effect.
    """
    db_path = _build(tmp_path)
    state = AgentState(q0="cholesterol", q_t="cholesterol", R_t=["d1", "d2", "d5"], O_t=_fake_o0(), B_t=3)

    # BM25 order for "cholesterol" over this corpus would rank d1/d2/d5
    # (all contain the term) ahead of d3/d4. Force the reranker to
    # invert that order completely, and confirm R_t follows the reranker,
    # not raw BM25.
    fake_reranked = [("d5", 9.0), ("d2", 5.0), ("d1", 1.0)]
    with (
        patch("src.agent.steps.generate_expansion", return_value="cholesterol"),
        patch("src.agent.steps.try_rerank", return_value=(fake_reranked, 1.0, 0)),
    ):
        result = act_once(
            Action.CONTEXT,
            state,
            r0=[("d1", 3.0), ("d2", 2.0), ("d5", 1.0)],
            db_path=db_path,
            evidence=["cholesterol statin"],
            k=10,
            use_reranker=True,
        )

    assert result.candidate.R_t[:3] == ["d5", "d2", "d1"]
    # remaining BM25 candidates beyond the reranked set still appear, preserving recall depth
    assert set(result.candidate.R_t) >= {"d5", "d2", "d1"}


def test_act_once_falls_back_to_bm25_order_when_reranker_unavailable(tmp_path):
    db_path = _build(tmp_path)
    state = AgentState(q0="cholesterol", q_t="cholesterol", R_t=["d1", "d2"], O_t=_fake_o0(), B_t=3)

    with (
        patch("src.agent.steps.generate_expansion", return_value="cholesterol"),
        patch("src.agent.steps.try_rerank", return_value=(None, 1.0, 1)),
    ):
        result = act_once(
            Action.CONTEXT,
            state,
            r0=[("d1", 3.0), ("d2", 2.0)],
            db_path=db_path,
            evidence=["cholesterol statin"],
            k=10,
            use_reranker=True,
        )

    from src.retrieval.bm25_index import search

    expected_bm25_order = [d for d, _ in search("cholesterol", db_path, k=10)]
    assert result.candidate.R_t == expected_bm25_order
