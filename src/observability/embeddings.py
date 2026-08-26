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
_MAX_CHARS = 2000  # defensive truncation -- long passages can exceed the model's context
_MAX_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = 2.0


class EmbeddingUnavailableError(RuntimeError):
    """Raised when embed() fails after retries; callers decide fallback behavior."""


@lru_cache(maxsize=4096)
def embed(text: str) -> tuple[float, ...]:
    """Cached: the same doc/query text often recurs across an episode
    (evidence passages, repeated queries) and across episodes. Retries a
    couple of times on transient server errors (observed empirically:
    Ollama's embedding endpoint occasionally 500s, seemingly independent
    of text length) before giving up.
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
