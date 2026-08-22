"""Cross-encoder reranker: applied only to the top-k BM25 candidates
(default 50, max 100) -- never a full-corpus dense pass. See
docs/milestones.md for the memory-budget rationale.
"""
from __future__ import annotations

from functools import lru_cache

from src.config import CONFIG


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(CONFIG.reranker_model)


def rerank(query: str, candidates: list[tuple[str, str]], top_k: int | None = None) -> list[tuple[str, float]]:
    """candidates: (doc_id, text) pairs, already truncated to
    CONFIG.reranker_max_candidates by the caller. Returns (doc_id, score)
    sorted descending.
    """
    top_k = top_k or CONFIG.reranker_top_k
    if not candidates:
        return []
    pairs = [(query, text) for _, text in candidates]
    scores = _model().predict(pairs)
    ranked = sorted(zip((doc_id for doc_id, _ in candidates), scores), key=lambda x: -x[1])
    return ranked[:top_k]
