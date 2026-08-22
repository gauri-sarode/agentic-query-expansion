# Observability model

One end-to-end trace per query episode, spans:

```
query_episode
  |-- retrieve_original
  |-- build_evidence
  |-- plan_action
  |-- generate_expansion
  |-- retrieve_expanded
  |-- rerank
  |-- verify
  |-- recover_or_accept
  `-- fuse
```

Traces, not just logs: OpenTelemetry's trace/metric/log split is the
implementation model, and its GenAI semantic conventions already name the
right concepts (operations, model interactions, data sources, evaluation
scores). The point of the observability layer is not a dashboard around a
search experiment — it is that telemetry becomes **causal input to the
agent's policy and verifier**. The question the agent asks is not "are
these documents relevant-looking?" but "is this search trajectory
healthy?"

## Telemetry families

**Retrieval telemetry** — top score, score margins, normalized score
entropy, query-term coverage, mean/rare-term IDF, entity overlap, top-`k`
ranking overlap, reranker/BM25 disagreement, result coherence, ranking
stability after a small perturbation.

**Agent telemetry** — selected operator, trajectory length, previous
verifier score, rewrite-to-original semantic drift, introduced entities,
repeated actions, remaining budget, accepted/rolled-back status.

**System telemetry** — BM25 latency, reranking latency, LLM latency,
prompt/output token counts, tool errors/timeouts, memory/resource
observations from the local process.

Capture only attributes with clear diagnostic value, not everything
observable — especially appropriate for a memory-constrained local
experiment.

## Trace record

```
query_id
dataset_slice
state_features_t0
retrieval_span_ms
planner_action
evidence_doc_ids
prompt_hash
model_id
generated_expansion
prompt_tokens
output_tokens
state_features_t1
reranker_span_ms
verifier_score
fault_type
controller_decision
rollback_or_retry
final_run_id
episode_latency_ms
```

Store: JSONL/Parquet or DuckDB, with an optional OTel export. No
requirement to run a commercial observability platform for the
experiment itself.
