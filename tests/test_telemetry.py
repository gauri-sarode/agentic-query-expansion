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


def test_top_score_reflects_the_actual_ranking_passed_in(tmp_path):
    """Regression test: telemetry must be computed from the ranking that
    is actually being evaluated/accepted, not always raw BM25 -- src/agent/steps.py
    now reorders BM25-scored tuples to match the post-rerank accepted
    order before calling this function, precisely so top_score/margin
    reflect what's really in first place, not whatever BM25 alone
    preferred. Simulate that here directly: same doc set, deliberately
    reordered scores.
    """
    db_path = _build(tmp_path)
    query = "cholesterol statin"
    bm25_ranking = search(query, db_path, k=4)  # e.g. [(d1, 5.0), (d2, 3.0), ...]

    reordered = list(reversed(bm25_ranking))
    telemetry = compute_retrieval_telemetry(query, reordered, db_path)

    assert telemetry.top_score == reordered[0][1]
    assert telemetry.top_score != bm25_ranking[0][1]  # genuinely different from the un-reordered case


def test_reranker_bm25_disagreement_is_set_based_regardless_of_order(tmp_path):
    """reranker_bm25_disagreement is a Jaccard (set) comparison, so it's
    insensitive to the order of `ranking` itself -- src/agent/steps.py's
    final ranking is always a full reordering (not a truncation) of the
    raw BM25 candidates, so the doc *set* used for this comparison is
    identical either way. bm25_ids exists for callers that pass a
    truncated ranking, where the sets genuinely would differ.
    """
    db_path = _build(tmp_path)
    query = "cholesterol statin"
    bm25_ranking = search(query, db_path, k=4)
    bm25_ids = [d for d, _ in bm25_ranking]
    reranked_order = list(reversed(bm25_ids))
    reordered_ranking = list(reversed(bm25_ranking))

    with_explicit_bm25_ids = compute_retrieval_telemetry(
        query, reordered_ranking, db_path, reranked_order=reranked_order, bm25_ids=bm25_ids
    )
    without_explicit_bm25_ids = compute_retrieval_telemetry(
        query, reordered_ranking, db_path, reranked_order=reranked_order
    )
    assert with_explicit_bm25_ids.reranker_bm25_disagreement == without_explicit_bm25_ids.reranker_bm25_disagreement


def test_reranker_bm25_disagreement_uses_bm25_ids_when_ranking_is_truncated(tmp_path):
    """When `ranking` covers a genuinely different (e.g. truncated) doc
    set than the raw BM25 candidates, bm25_ids anchors the disagreement
    measure to the real BM25 pool rather than whatever subset happens to
    be in `ranking`.
    """
    db_path = _build(tmp_path)
    query = "cholesterol statin"
    bm25_ranking = search(query, db_path, k=4)
    bm25_ids = [d for d, _ in bm25_ranking]
    reranked_order = [bm25_ids[0]]  # reranker "kept" only the top doc

    truncated_ranking = bm25_ranking[:1]  # caller passes just the top-1, not the full pool
    telemetry_default = compute_retrieval_telemetry(
        query, truncated_ranking, db_path, reranked_order=reranked_order
    )
    telemetry_with_bm25_ids = compute_retrieval_telemetry(
        query, truncated_ranking, db_path, reranked_order=reranked_order, bm25_ids=bm25_ids
    )

    assert telemetry_default.reranker_bm25_disagreement == 0.0  # truncated ranking == reranked_order exactly
    assert telemetry_with_bm25_ids.reranker_bm25_disagreement > 0.0  # full BM25 pool disagrees with a 1-doc reranked set
