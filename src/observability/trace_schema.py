"""The episode-level trace record. One row per query episode, written to
the trace store (src/observability/store.py). Field set follows the
brief's query_episode span sequence: retrieve_original -> build_evidence
-> plan_action -> generate_expansion -> retrieve_expanded -> rerank ->
verify -> recover_or_accept -> fuse.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TraceRecord:
    query_id: str
    dataset_slice: str  # e.g. "tripclick/head", "tripclick/torso", "tripclick/tail"

    state_features_t0: dict
    retrieval_span_ms: float

    planner_action: str
    evidence_doc_ids: list[str]

    prompt_hash: str | None
    model_id: str | None
    generated_expansion: str | None
    prompt_tokens: int
    output_tokens: int

    state_features_t1: dict
    reranker_span_ms: float

    verifier_score: float | None
    fault_type: str | None
    controller_decision: str
    rollback_or_retry: bool
    llm_calls: int

    final_run_id: str
    episode_latency_ms: float

    def to_dict(self) -> dict:
        return asdict(self)
