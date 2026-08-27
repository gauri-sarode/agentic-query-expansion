# Execution plan and go/no-go checkpoints

## Hardware/runtime constraints

M4 MacBook Air, 16GB unified memory shared across OS, Python/JVM,
index buffers, reranker, LLM weights, and KV cache. A full dense index
plus a large local LLM is unnecessary risk — keep this lightweight:

| Component | Configuration |
|---|---|
| Sparse retrieval | BM25 over the full corpus, disk-backed (no JVM — see below) |
| Semantic tool | Small cross-encoder reranker, top-50 (max 100) candidates only |
| Generation | Small local LLM, 4-bit quantized; 2K-4K token context, not the model max |
| LLM output | 32-64 tokens, structured search action only |
| Generation budget | 1 call normally; hard maximum 2 calls/query |
| Controller | Logistic regression / GBDT + finite-state controller |
| Trace store | JSONL/Parquet or DuckDB; optional OTel export |
| Dense index | None for the main experiment |
| Caching | Prompts, outputs, rankings, verifier features, and timing all cached |

**Known local constraint:** Pyserini/Lucene BM25 has previously crashed
the JVM natively on this machine's JDK in a related project
(`llm-query-expansion`). Prefer a disk-backed, non-JVM BM25 implementation
(e.g. SQLite FTS5) unless/until a working JVM setup is confirmed here —
don't rediscover that failure mode from scratch.

## Sequencing

1. **Immediate, no dataset blocker:** NFCorpus builds the whole agent +
   telemetry stack (index, retrieval, reranker, expansion operators,
   agent loop, trace schema, verifier, fault injection, evaluation
   scripts) end to end on a cheap corpus.
2. **MS MARCO Chameleons** — public, hard-query stress test; start once
   the pipeline works.
3. **TREC DL 2019/2020** — public, deep NIST judgments; intent-drift and
   recovery evaluation.
4. **TripClick** — access granted. Ingest via the provided Google Drive
   package once the pipeline is validated on the public datasets;
   HEAD/TORSO/TAIL frequency-tail analysis is TripClick-exclusive.

## Go/no-go checkpoints

- **Expansion headroom** — proceed only if the best targeted
  expansion/operator materially beats generic unconditional expansion on
  development queries, with a paired bootstrap interval supporting a real
  advantage.
- **Observability headroom** — proceed with the "observability-guided"
  claim only if the full telemetry model predicts harmful/healthy
  trajectories better than a retrieval-only detector and better than the
  relevant class-prevalence baseline.
- **Agency headroom** — the feedback-enabled agent must improve
  retrieval, harm, or cost relative to the static failure-conditioned
  pipeline without meaningful deterioration in the other dimensions.
  **NOT MET** (2026-08-26, full TripClick TAIL test set, n=1175, v1
  fitted verifier): agent vs. static NDCG@10 mean_diff=-0.0007 (95% CI
  [-0.0024, 0.0007], not significant), and cost is meaningfully *worse*
  (+27% LLM calls, +45% latency) — see README's Core result section. This
  was tested rigorously, not skipped: the verifier was fit against real
  ground truth on this exact corpus before this run. Traces to the
  verifier's own weak predictive ceiling (r=0.134) rather than a plumbing
  or calibration-methodology gap.
- **Recovery headroom** — rollback/replan must measurably recover
  injected or naturally harmful trajectories; otherwise recovery should be
  demoted from a headline contribution.
  **MET, for content-level faults** (2026-08-27, `scripts/11_fault_injection_recovery_study.py`,
  n=149-150/fault type, TripClick TAIL): entity injection (Detection
  Recall=1.000, recovery gain +0.0238 NDCG@10) and grounding-passage
  corruption (Recall=0.773, discriminating 46.5% false-alarm rate, largest
  gain +0.0278) are both caught and recovered reliably — an order of
  magnitude larger than any effect seen on natural queries. **NOT met for
  system-level faults**: reranker-disabled degradation is caught only
  18.5% of the time (the fitted verifier's weights are almost entirely
  lexical/content features with near-zero reranker-specific weight).
  This is now the paper's strongest result: promote recovery to a
  headline contribution, scoped honestly to content-level faults.

## Metrics, SLOs, fault injection

Retrieval effectiveness: NDCG@10, Recall@100, MRR — TripClick reported
separately for HEAD/TORSO/TAIL. Also report query-level
helped/unchanged/harmed fractions (mean NDCG can conceal intervention
regressions).

Observability/agent metrics:

```
Detection Recall  = P(detected | failure)
False Alarm Rate  = P(intervention | healthy)
Rollback Precision = P(R_{t-1} > R_t | ROLLBACK)
Recovery Rate = (# failed episodes restored) / (# recoverable failed episodes)
```

Plus: actions-to-recovery, action/oracle regret, intervention rate,
calls/query, generated tokens/query, p50/p95 latency, retrieval gain per
LLM call.

**Agent Search SLO** (target range, not a single optimized metric):

```
maximize retrieval quality
subject to   harm rate <= h,   E[LLM calls/query] <= c,   p95(latency) <= L
```

Do not invent `L` from an online benchmark — measure ~100 representative
episodes locally first and freeze `h`, `c`, `L` before main-test
evaluation.

## Deliverables

Reproducible codebase with typed retrieval/expansion/verification tools;
episode-level telemetry schema; cached baseline and agent runs;
deterministic fault-injection scripts; paired-bootstrap evaluation
scripts; configs that pin model, prompts, budgets, and thresholds; and
four paper-ready outputs: the agent architecture figure, the
quality-harm-cost Pareto plot, the failure-detection/recovery plot, and
HEAD/TORSO/TAIL TripClick results.

For TripClick specifically: publish code and aggregated results, never
the corpus itself (non-commercial research license, no redistribution).
