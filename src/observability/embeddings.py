"""Embedding client via Ollama's local mxbai-embed-large model (already
pulled, previously unused). All telemetry tried so far
(src/observability/telemetry.py) is lexical (BM25 term overlap) or a
scalar reranker score -- this is a qualitatively different signal family
(dense semantic similarity), used to test whether it raises the
verifier's weak lexical-only ceiling (r=0.134,
scripts/08_calibrate_verifier.py) rather than just re-weighting the same
features again.
"""
from __future__ import annotations

import time
from functools import lru_cache

import numpy as np
import requests

_EMBED_MODEL = "mxbai-embed-large"
_EMBED_URL = "http://localhost:11434/api/embeddings"
_MAX_CHARS = 2000  # defensive first truncation -- see embed()'s docstring for why this alone isn't enough
_MIN_CHARS = 200  # below this, stop halving and just retry-with-backoff (likely a real server issue, not length)
_MAX_ATTEMPTS = 5
_RETRY_BACKOFF_SECONDS = 2.0


class EmbeddingUnavailableError(RuntimeError):
    """Raised when embed() fails after retries; callers decide fallback behavior."""


@lru_cache(maxsize=4096)
def embed(text: str) -> tuple[float, ...]:
    """Cached: the same doc/query text often recurs across an episode
    (evidence passages, repeated queries) and across episodes.

    mxbai-embed-large is a BERT-architecture model with a hard 512-token
    context limit (`ollama show mxbai-embed-large`), and the server 500s
    rather than truncating server-side when a prompt exceeds it. The
    fixed _MAX_CHARS cap is a poor proxy for token count on this corpus:
    dense scientific/medical vocabulary (e.g. "nonsporulating",
    "facultative anaerobic") produces more subword tokens per character
    than typical English, so some texts well under _MAX_CHARS chars
    still overflow 512 tokens while others near the cap don't -- e.g. a
    1,831-char TripClick abstract 500'd while the same text truncated to
    1,600 chars succeeded (empirically found investigating a ~20% skip
    rate in scripts/09's calibration run, 2026-08-30). Retrying the same
    truncated text on a 500 just repeats the same failure, so this
    instead halves the length on each retry until it fits or a floor is
    hit.
    """
    truncated = text[:_MAX_CHARS]
    last_error: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            response = requests.post(_EMBED_URL, json={"model": _EMBED_MODEL, "prompt": truncated}, timeout=30)
            response.raise_for_status()
            return tuple(response.json()["embedding"])
        except requests.RequestException as e:
            last_error = e
            if attempt < _MAX_ATTEMPTS - 1:
                if len(truncated) > _MIN_CHARS:
                    truncated = truncated[: len(truncated) // 2]
                else:
                    time.sleep(_RETRY_BACKOFF_SECONDS * (attempt + 1))
    raise EmbeddingUnavailableError(str(last_error)) from last_error


def cosine_similarity(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    a_arr, b_arr = np.array(a), np.array(b)
    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    return float(np.dot(a_arr, b_arr) / denom) if denom else 0.0


def mean_pairwise_similarity(texts: list[str]) -> float:
    if len(texts) < 2:
        return 1.0
    vecs = [embed(t) for t in texts]
    sims = [cosine_similarity(vecs[i], vecs[j]) for i in range(len(vecs)) for j in range(i + 1, len(vecs))]
    return sum(sims) / len(sims)
