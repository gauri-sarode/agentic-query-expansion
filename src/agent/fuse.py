"""fuse(original_ranking, accepted_ranking) -> final ranking. Combines the
original query's ranking with the ranking produced by an accepted
expansion action, rather than discarding the original signal outright.
"""
from __future__ import annotations

from src.agent.state import Ranking


def fuse(original_ranking: Ranking, accepted_ranking: Ranking) -> Ranking:
    raise NotImplementedError("TODO: reciprocal-rank or score-based fusion")
