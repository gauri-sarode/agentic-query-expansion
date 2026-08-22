"""Observation vector O_t: three telemetry families computed from the
current retrieval state, agent trajectory, and system runtime. These
features are what the agent's controller and verifier condition on --
telemetry as causal input to policy, not passive logging. See
docs/observability_model.md.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalTelemetry:
    top_score: float
    score_margin: float  # top_1 - top_2
    score_entropy: float  # normalized entropy of top-k score distribution
    query_term_coverage: float  # fraction of query terms with lexical support in top-k
    mean_rare_term_idf: float
    entity_overlap: float  # query entities present in top-k docs
    topk_overlap: float  # Jaccard overlap of top-k doc IDs vs. previous state
    reranker_bm25_disagreement: float  # rank correlation distance, sparse vs. reranked
    result_coherence: float  # inter-document similarity within top-k
    ranking_stability: float  # top-k overlap after a small controlled perturbation


@dataclass(frozen=True)
class AgentTelemetry:
    selected_operator: str
    trajectory_length: int
    previous_verifier_score: float | None
    rewrite_semantic_drift: float  # embedding distance, q_t vs. q_0
    introduced_entities: int  # entities in q_t/expansion absent from q_0 and evidence
    repeated_actions: int
    remaining_budget: int
    accepted: bool | None
    rolled_back: bool


@dataclass(frozen=True)
class SystemTelemetry:
    bm25_latency_ms: float
    reranker_latency_ms: float
    llm_latency_ms: float
    prompt_tokens: int
    output_tokens: int
    tool_errors: int
    timed_out: bool


@dataclass(frozen=True)
class ObservabilityState:
    """O_t: the full observation the controller and verifier see at step t."""

    retrieval: RetrievalTelemetry
    agent: AgentTelemetry
    system: SystemTelemetry


def compute_retrieval_telemetry(*args, **kwargs) -> RetrievalTelemetry:
    raise NotImplementedError("TODO: derive from ranking + reranker + BM25 outputs")


def compute_agent_telemetry(*args, **kwargs) -> AgentTelemetry:
    raise NotImplementedError("TODO: derive from episode history H_t")


def compute_system_telemetry(*args, **kwargs) -> SystemTelemetry:
    raise NotImplementedError("TODO: derive from tool call timing/errors")
