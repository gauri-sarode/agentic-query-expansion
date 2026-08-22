from src.observability.telemetry import compute_retrieval_telemetry
from src.retrieval.bm25_index import build_index, search

_DOCS = [
    ("d1", "cholesterol statin drugs breast cancer risk study"),
    ("d2", "cholesterol statin drugs cardiovascular disease"),
    ("d3", "unrelated topic about gardening and plants"),
    ("d4", "breast cancer treatment options and outcomes"),
]


def _build(tmp_path):
    db_path = str(tmp_path / "test.sqlite3")
    build_index(_DOCS, db_path)
    return db_path


def test_retrieval_telemetry_is_bounded_and_sane(tmp_path):
    db_path = _build(tmp_path)
    query = "cholesterol statin drugs breast cancer"
    ranking = search(query, db_path, k=4)
    assert ranking, "expected non-empty BM25 hits"

    telemetry = compute_retrieval_telemetry(query, ranking, db_path)

    assert 0.0 <= telemetry.query_term_coverage <= 1.0
    assert 0.0 <= telemetry.score_entropy <= 1.0
    assert 0.0 <= telemetry.topk_overlap <= 1.0
    assert 0.0 <= telemetry.result_coherence <= 1.0
    assert 0.0 <= telemetry.ranking_stability <= 1.0
    assert telemetry.top_score >= ranking[0][1] - 1e-6


def test_topk_overlap_is_one_without_previous_ranking(tmp_path):
    db_path = _build(tmp_path)
    ranking = search("cholesterol statin", db_path, k=4)
    telemetry = compute_retrieval_telemetry("cholesterol statin", ranking, db_path)
    assert telemetry.topk_overlap == 1.0


def test_topk_overlap_reflects_previous_ranking_change(tmp_path):
    db_path = _build(tmp_path)
    ranking_a = search("cholesterol statin", db_path, k=4)
    ranking_b = search("gardening plants", db_path, k=4)
    telemetry = compute_retrieval_telemetry(
        "gardening plants", ranking_b, db_path, previous_ranking=ranking_a
    )
    assert telemetry.topk_overlap < 1.0
