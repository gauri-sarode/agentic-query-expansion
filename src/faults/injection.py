"""Seeded fault injection for the recovery study (RQ-Recovery,
docs/failure_taxonomy.md). Each fault is deterministic given a seed so a
clean paired episode for the same query can be compared against the
corrupted one -- run the same query through both and diff the agent's
behavior.
"""
from __future__ import annotations

import random
import re
from enum import Enum

from src.retrieval import bm25_index

_WORD_RE = re.compile(r"[A-Za-z0-9]+")

# Fixed, topic-neutral vocabulary for injected noise -- deliberately far
# from typical corpus domains (health/medical for TripClick, science for
# NFCorpus) so it reads as unambiguously irrelevant/unsupported rather
# than a plausible corpus term.
_IRRELEVANT_TERMS = [
    "kayak", "turbine", "opera", "volcano", "marmalade", "trombone",
    "glacier", "spreadsheet", "carnival", "lighthouse", "quicksand",
    "origami", "tapestry", "meridian", "cobblestone",
]
_UNSUPPORTED_ENTITIES = [
    "Zentara Corp", "Dr. Halvorsen", "the Kestrel Accord", "Meridian Labs",
    "Senator Voss", "the Thistlewood Institute", "Novak Dynamics",
]


class FaultType(str, Enum):
    REMOVE_HIGH_IDF_TERM = "remove_high_idf_term"
    APPEND_IRRELEVANT_TERM = "append_irrelevant_term"
    REPLACE_GROUNDING_PASSAGE = "replace_grounding_passage"
    INJECT_UNSUPPORTED_ENTITY = "inject_unsupported_entity"
    TRUNCATE_RETRIEVAL_POOL = "truncate_retrieval_pool"
    DISABLE_RERANKER = "disable_reranker"
    SIMULATE_TIMEOUT = "simulate_timeout"


def remove_high_idf_term(query: str, db_path: str, seed: int) -> str:
    """Drop the single highest-IDF (most discriminative) query term --
    simulates a vocabulary-mismatch-style failure by removing the term
    most responsible for precision.
    """
    tokens = _WORD_RE.findall(query)
    if len(tokens) <= 1:
        return query
    n = bm25_index.doc_count(db_path)
    rarest = max(tokens, key=lambda t: bm25_index.idf(t, db_path, total_docs=n))
    rng = random.Random(seed)
    rarest_idf = bm25_index.idf(rarest, db_path, total_docs=n)
    # If multiple tokens tie for highest IDF, break ties deterministically via seed.
    tied = [t for t in tokens if bm25_index.idf(t, db_path, total_docs=n) == rarest_idf]
    drop = rng.choice(tied)

    result = re.sub(rf"\b{re.escape(drop)}\b", "", query, count=1, flags=re.IGNORECASE)
    result = " ".join(result.split())
    return result or query


def append_irrelevant_term(query: str, seed: int) -> str:
    rng = random.Random(seed)
    return f"{query} {rng.choice(_IRRELEVANT_TERMS)}"


def replace_grounding_passage(
    evidence_doc_ids: list[str], db_path: str, seed: int
) -> list[str]:
    """Swaps one evidence doc for a "topically similar but non-supporting"
    one: a document that shares the single lowest-IDF (least specific,
    most generic) term with the evidence but is otherwise an unrelated
    random draw from that loosely-matching pool -- plausible-looking
    grounding that doesn't actually support the query. Returns the
    evidence *texts* (not doc IDs), in evidence_doc_ids order.
    """
    texts = bm25_index.get_texts(evidence_doc_ids, db_path)
    original = [texts.get(d, "") for d in evidence_doc_ids]
    if not evidence_doc_ids or not texts:
        return original

    rng = random.Random(seed)
    victim_id = rng.choice(list(texts.keys()))
    victim_tokens = _WORD_RE.findall(texts[victim_id])
    if not victim_tokens:
        return original
    n = bm25_index.doc_count(db_path)
    generic_term = min(victim_tokens, key=lambda t: bm25_index.idf(t, db_path, total_docs=n))

    candidates = bm25_index.search(generic_term, db_path, k=50)
    pool = [d for d, _ in candidates if d not in evidence_doc_ids]
    if not pool:
        return original
    replacement_id = rng.choice(pool)
    replacement_text = bm25_index.get_texts([replacement_id], db_path).get(replacement_id)
    if replacement_text is None:
        return original

    return [replacement_text if doc_id == victim_id else texts.get(doc_id, "") for doc_id in evidence_doc_ids]


def inject_unsupported_entity(expansion: str, seed: int) -> str:
    rng = random.Random(seed)
    return f"{expansion} {rng.choice(_UNSUPPORTED_ENTITIES)}"


def truncate_retrieval_pool(candidates: list, max_n: int) -> list:
    return candidates[:max_n]


def disable_reranker(candidates: list) -> list:
    return candidates  # pass sparse ranking through untouched


def simulate_timeout() -> None:
    raise TimeoutError("simulated tool timeout")
