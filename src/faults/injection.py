"""Seeded fault injection for the recovery study (RQ-Recovery,
docs/failure_taxonomy.md). Each fault is deterministic given a seed so a
clean paired episode for the same query can be compared against the
corrupted one.
"""
from __future__ import annotations

from enum import Enum


class FaultType(str, Enum):
    REMOVE_HIGH_IDF_TERM = "remove_high_idf_term"
    APPEND_IRRELEVANT_TERM = "append_irrelevant_term"
    REPLACE_GROUNDING_PASSAGE = "replace_grounding_passage"
    INJECT_UNSUPPORTED_ENTITY = "inject_unsupported_entity"
    TRUNCATE_RETRIEVAL_POOL = "truncate_retrieval_pool"
    DISABLE_RERANKER = "disable_reranker"
    SIMULATE_TIMEOUT = "simulate_timeout"


def remove_high_idf_term(query: str, seed: int) -> str:
    raise NotImplementedError("TODO: drop the highest-IDF query term, seeded for reproducibility")


def append_irrelevant_term(query: str, seed: int) -> str:
    raise NotImplementedError("TODO: append a corpus term unrelated to the query topic")


def replace_grounding_passage(evidence: list[str], corpus, seed: int) -> list[str]:
    raise NotImplementedError(
        "TODO: swap one grounding passage for a topically similar but non-supporting one"
    )


def inject_unsupported_entity(expansion: str, seed: int) -> str:
    raise NotImplementedError("TODO: insert an entity not present in query or evidence")


def truncate_retrieval_pool(candidates: list, max_n: int) -> list:
    return candidates[:max_n]


def disable_reranker(candidates: list) -> list:
    return candidates  # pass sparse ranking through untouched


def simulate_timeout() -> None:
    raise TimeoutError("simulated tool timeout")
