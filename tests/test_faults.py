from src.faults import injection
from src.retrieval.bm25_index import build_index

_DOCS = [
    ("d1", "cholesterol statin drugs breast cancer risk study"),
    ("d2", "cholesterol statin drugs cardiovascular disease outcomes"),
    ("d3", "unrelated topic about gardening and plants"),
    ("d4", "breast cancer treatment options and outcomes"),
    ("d5", "cholesterol levels and diet in older adults"),
    ("d6", "cholesterol screening guidelines for primary care"),
]


def _build(tmp_path):
    db_path = str(tmp_path / "test.sqlite3")
    build_index(_DOCS, db_path)
    return db_path


def test_remove_high_idf_term_drops_a_token_deterministically(tmp_path):
    db_path = _build(tmp_path)
    query = "cholesterol statin breast cancer"
    result = injection.remove_high_idf_term(query, db_path, seed=1)
    assert result != query
    assert len(result.split()) == len(query.split()) - 1
    # deterministic given the same seed
    assert result == injection.remove_high_idf_term(query, db_path, seed=1)


def test_remove_high_idf_term_noop_on_single_token(tmp_path):
    db_path = _build(tmp_path)
    assert injection.remove_high_idf_term("cholesterol", db_path, seed=1) == "cholesterol"


def test_append_irrelevant_term_is_deterministic_and_extends_query():
    a = injection.append_irrelevant_term("statin drugs", seed=42)
    b = injection.append_irrelevant_term("statin drugs", seed=42)
    assert a == b
    assert a.startswith("statin drugs ")
    assert len(a.split()) == 3


def test_replace_grounding_passage_swaps_one_doc(tmp_path):
    db_path = _build(tmp_path)
    evidence_ids = ["d1", "d2"]
    result = injection.replace_grounding_passage(evidence_ids, db_path, seed=7)
    assert len(result) == 2
    # exactly one of the two texts should differ from the originals
    from src.retrieval.bm25_index import get_texts

    originals = get_texts(evidence_ids, db_path)
    unchanged = sum(1 for doc_id, text in zip(evidence_ids, result) if originals[doc_id] == text)
    assert unchanged == 1


def test_inject_unsupported_entity_is_deterministic():
    a = injection.inject_unsupported_entity("statin drugs cholesterol", seed=3)
    b = injection.inject_unsupported_entity("statin drugs cholesterol", seed=3)
    assert a == b
    assert a.startswith("statin drugs cholesterol ")


def test_truncate_and_disable_reranker_passthrough():
    candidates = [("d1", 1.0), ("d2", 0.5), ("d3", 0.1)]
    assert injection.truncate_retrieval_pool(candidates, 2) == candidates[:2]
    assert injection.disable_reranker(candidates) == candidates


def test_simulate_timeout_raises():
    import pytest

    with pytest.raises(TimeoutError):
        injection.simulate_timeout()
