"""Central, config-driven knobs for the agentic query expansion experiment.

Datasets: NFCorpus (dev loop), MS MARCO Chameleons (hard-query stress
test), TREC DL 2019/2020 (deep judgments), TripClick (primary long-tail
benchmark, HEAD/TORSO/TAIL). Runtime is constrained to a 16GB unified
memory machine -- see docs/milestones.md for why BM25 is disk-backed and
non-JVM, generation is a small 4-bit local model with a hard call budget,
and there is no dense index for the main experiment.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    dataset: str = os.getenv("AQE_DATASET", "nfcorpus")
    data_dir: str = os.getenv("AQE_DATA_DIR", "data")
    results_dir: str = os.getenv("AQE_RESULTS_DIR", "results")
    seed: int = int(os.getenv("AQE_SEED", "42"))
    k_values: tuple[int, ...] = field(default_factory=lambda: (10, 100))

    # Sparse retrieval: disk-backed, no JVM (Pyserini/Lucene has crashed
    # the JVM natively on this machine in a related project -- see
    # docs/milestones.md).
    bm25_db_path: str = os.getenv("AQE_BM25_DB_PATH", "data/bm25_index.sqlite3")

    reranker_model: str = os.getenv(
        "AQE_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2"
    )
    reranker_top_k: int = int(os.getenv("AQE_RERANKER_TOP_K", "50"))
    reranker_max_candidates: int = int(os.getenv("AQE_RERANKER_MAX_CANDIDATES", "100"))

    # Generation.
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama")
    # Brief recommends Qwen3.5-2B (4-bit); not pulled locally yet -- default
    # to the closest available local Ollama model (qwen2.5:3b) so the
    # pipeline runs today, swap via EXPANSION_MODEL once 3.5-2B is pulled.
    expansion_model: str = os.getenv("EXPANSION_MODEL", "qwen2.5:3b")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    hf_api_token: str | None = os.getenv("HF_API_TOKEN")
    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY")

    llm_context_tokens: int = int(os.getenv("AQE_LLM_CONTEXT_TOKENS", "4096"))
    llm_output_tokens: int = int(os.getenv("AQE_LLM_OUTPUT_TOKENS", "64"))
    max_llm_calls_per_query: int = int(os.getenv("AQE_MAX_LLM_CALLS", "2"))

    # Agent action budget (remaining actions in B_t).
    action_budget: int = int(os.getenv("AQE_ACTION_BUDGET", "3"))

    # Agent Search SLO thresholds -- frozen from ~100 local dev episodes
    # before main-test evaluation (docs/milestones.md); placeholders until
    # that calibration run.
    slo_harm_rate: float | None = None
    slo_max_llm_calls: float | None = None
    slo_p95_latency_ms: float | None = None


CONFIG = Config()
