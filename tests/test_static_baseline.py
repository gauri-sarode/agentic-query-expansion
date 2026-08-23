from unittest.mock import patch

from src.agent.static_baseline import run_static_episode
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


def test_static_episode_never_makes_more_than_one_llm_call(tmp_path):
    db_path = _build(tmp_path)
    with patch("src.agent.steps.generate_expansion", return_value="cholesterol statin outcomes"):
        _, trace = run_static_episode("q1", "cholesterol", db_path, use_reranker=False)
    assert trace.llm_calls <= 1
    assert trace.rollback_or_retry is False
    assert trace.controller_decision in {"NO_EXPAND", "ACCEPT", "REJECT"}


def test_static_episode_rejects_below_accept_threshold(tmp_path):
    db_path = _build(tmp_path)
    with patch("src.agent.steps.generate_expansion", return_value="unrelated gardening plants"):
        final_ranking, trace = run_static_episode(
            "q1", "cholesterol statin cardiovascular", db_path, accept_threshold=1e9, use_reranker=False
        )
    assert trace.controller_decision in {"REJECT", "NO_EXPAND"}
    if trace.controller_decision == "REJECT":
        from src.retrieval.bm25_index import search

        original_ids = [d for d, _ in search("cholesterol statin cardiovascular", db_path, k=100)]
        assert final_ranking == original_ids
