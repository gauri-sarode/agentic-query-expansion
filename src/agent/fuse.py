"""fuse(original_ranking, accepted_ranking) -> final ranking. Combines the
original query's ranking with the ranking produced by an accepted
expansion action via reciprocal rank fusion, rather than discarding the
original signal outright.
"""
from __future__ import annotations

from src.agent.state import Ranking

_RRF_K = 60  # standard RRF damping constant


def fuse(original_ranking: Ranking, accepted_ranking: Ranking) -> Ranking:
    scores: dict[str, float] = {}
    for rank, doc_id in enumerate(original_ranking):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (_RRF_K + rank + 1)
    for rank, doc_id in enumerate(accepted_ranking):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (_RRF_K + rank + 1)
    return sorted(scores, key=lambda d: -scores[d])
