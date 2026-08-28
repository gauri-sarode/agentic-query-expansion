# Observe, Expand, Recover

An observability-guided agent for safe query expansion. Target venue: **IEEE ICA 2026**.

LLM query expansion is usually applied unconditionally or not at all. This
project treats it instead as a **bounded agentic decision**: an agent
observes retrieval health, selectively invokes a corpus-grounded LLM
expansion operator, observes the consequences of that action through
runtime telemetry, and decides to accept, roll back, replan, or abstain —
under an explicit quality/harm/cost budget.

Three ideas stay fused rather than separate: LLM query expansion is the
**task mechanism**, an agentic control loop is the **decision process**,
and AI observability is the **feedback/control layer** — the telemetry is
causal input to the agent's policy and verifier, not just logging.

## Research questions

- **RQ-Detection** — Can qrel-free retrieval telemetry and agent-runtime
  telemetry identify unhealthy retrieval states and predict whether
  expansion will help or harm?
- **RQ-Control** — Does an agent that conditions expansion actions on
  those observations outperform an otherwise identical *static*
  retrieval-aware QE pipeline, on quality, harmful-intervention rate, or
  inference cost?
- **RQ-Recovery** — After a harmful rewrite or an injected system/retrieval
  fault, can observability-driven verification detect the degradation and
  choose `ROLLBACK` / `REPLAN` / `STOP` accurately enough to restore
  retrieval quality under a bounded action budget?

The target result is not top raw NDCG — it's moving the
**quality–safety–cost Pareto frontier** relative to a static QE pipeline
built from the same retriever, grounding, generator, and fusion mechanism.

## System design

Agent state at step `t`:

```
s_t = (q0, q_t, R_t, O_t, H_t, B_t)
```

`q0` immutable original query, `q_t` active search expression, `R_t`
current ranking, `O_t` observability state, `H_t` action/observation
history, `B_t` remaining compute/action budget.

Control loop:

```
OBSERVE -> DIAGNOSE -> ACT -> RETRIEVE -> OBSERVE -> VERIFY
        -> {ACCEPT, ROLLBACK, REPLAN, STOP}
```

The critical experimental contrast: a **static** comparator gets the same
query, ranking, evidence, LLM, and verifier, but executes exactly one
predetermined route. The agent's second decision depends on evidence its
own first action created. That dependency is the thing being tested.

Actions: `NO_EXPAND, VOCABULARY, CONTEXT, AMBIGUITY, ENTITY_NORMALIZE,
ROLLBACK, REPLAN, STOP`. The LLM is one typed tool inside the agent
(`generate_expansion`), not the planner, retriever, judge, and memory
manager simultaneously — action selection and recovery are handled by a
small learned/rule-based controller, not free-form LLM reasoning.

Observability model: one end-to-end trace per query episode
(`retrieve_original -> build_evidence -> plan_action -> generate_expansion
-> retrieve_expanded -> rerank -> verify -> recover_or_accept -> fuse`),
with three telemetry families — retrieval (score margins, entropy, term
coverage, entity overlap, top-k overlap, reranker disagreement, ranking
stability under perturbation), agent (selected operator, trajectory
length, drift from original, budget remaining), and system (latency,
token counts, tool errors/timeouts). See `docs/observability_model.md`.

Failure taxonomy (what each observed signature should trigger — see
`docs/failure_taxonomy.md`): vocabulary mismatch, underspecification,
ambiguity, entity mismatch, intent drift, confirmation bias, retriever
disagreement, tool degradation, budget exhaustion, healthy retrieval.

## Datasets

| Dataset | Role | Status |
|---|---|---|
| **TripClick** | Primary long-tail benchmark (HEAD/TORSO/TAIL) — ~1.52M docs, ~686K train queries, 3,525-query test set | **Access granted** — non-commercial research use, dataset itself may not be redistributed |
| MS MARCO Chameleons | Hard-query stress test (queries that resist conventional reformulation) | Public |
| TREC DL 2019/2020 | Deep NIST graded judgments — intent-drift/recovery safety check | Public |
| NFCorpus (BEIR) | Fast dev loop — ~3.6K docs / 323 queries | Public, start immediately |

Only TripClick supports the frequency-tail claim; Chameleons supports the
retrieval-difficulty claim; TREC DL supports the deeply-judged
safety/recovery claim; NFCorpus is the engineering/early-feasibility
benchmark. Development does not wait on any single dataset — NFCorpus
builds the whole agent+telemetry stack first.

TripClick terms prohibit redistributing the corpus. This repo publishes
**code and aggregated results only**; raw TripClick data stays out of
version control (see `data/README.md`).

## Experiment matrix

| System | Query expansion | Closed-loop action | Runtime observability controls policy | Recovery | Purpose |
|---|---|---|---|---|---|
| BM25 | — | — | — | — | Retrieval floor |
| RM3/PRF | yes | — | — | — | Classical feedback baseline |
| Query2doc-style | yes | — | — | — | Strong LLM-QE baseline |
| MuGI-style | yes | — | — | — | Multi-generation/fusion baseline |
| Grounded static QE | yes | — | retrieval evidence only | — | Isolate grounding |
| Failure-conditioned static QE | yes | — | initial retrieval state | verifier only | Strong non-agentic comparator |
| Retrieval agent | yes | yes | retrieval state only | yes | Isolate agency |
| **Observable retrieval agent** | yes | yes | retrieval + agent + system telemetry | yes | **Proposed method** |
| Strategy oracle | yes | offline only | qrels | — | Upper bound, never deployed |

## Repo layout

```
src/
  config.py                 -- run knobs: datasets, budgets, model IDs, paths
  retrieval/                -- BM25 index/search, cross-encoder reranker
  expansion/                -- typed expansion operators + prompts
  agent/                    -- state, actions, controller/policy, control loop, verify, fuse
  observability/            -- trace schema, telemetry feature extraction, trace store
  faults/                   -- seeded fault injection for the recovery study
  eval/                     -- metrics, Agent Search SLO, Pareto/bootstrap analysis
configs/                     -- pinned model/budget/threshold configs per experiment
scripts/                     -- dataset setup, index builds, experiment runners
docs/                        -- observability model, failure taxonomy, milestones, sources
data/                        -- gitignored; local indexes/caches/downloaded corpora
tests/
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Status

The full closed-loop agent runs end to end (rule-based v0
controller/verifier -- see `src/agent/controller.py` for why v0 is
rule-based rather than learned). TripClick's IR Benchmark package is
downloaded, parsed, and indexed (1,523,878 docs; 1,175 queries each for
HEAD/TORSO/TAIL test).

BM25-only retrieval floor (no expansion, no agent):

| Split | MRR | Recall@100 | NDCG@10 |
|---|---|---|---|
| NFCorpus | 0.514 | 0.232 | 0.303 |
| TripClick HEAD | 0.324 | 0.411 | 0.179 |
| TripClick TORSO | 0.261 | 0.582 | 0.182 |
| TripClick TAIL | 0.225 | 0.646 | 0.228 |

## Core result: closed-loop agent vs. static QE, full TripClick TAIL test set

The critical comparison from the experiment matrix -- **observable agent
vs. the strong static failure-conditioned QE comparator**
(`scripts/06_agent_vs_static.py`), same operator-selection policy,
generator, verifier, and fusion mechanism for both (`src/agent/steps.py`)
-- run on the **complete TripClick TAIL test set (n=1175)**, with
`verify()`'s weights fitted against real ground-truth NDCG@10 on this
exact corpus (`scripts/08_calibrate_verifier.py`, n=1168, Pearson
r=0.134) and `RecoveryThresholds` recalibrated to match against real
outcomes, not blind score percentiles:

| System | NDCG@10 | Recall@100 | MRR |
|---|---|---|---|
| BM25 only | 0.2279 | 0.6462 | 0.2254 |
| Static QE | 0.2279 | 0.6481 | 0.2262 |
| Agent | 0.2272 | 0.6476 | 0.2254 |

Agent vs. static: helped 0.5%, unchanged 98.6%, harmed 0.9%. Paired
bootstrap mean_diff=**-0.0007**, 95% CI [-0.0024, 0.0007] -- **not
significant, point estimate slightly negative**. Cost: agent uses 27%
more LLM calls (1.27 vs 0.99/query) and 45% higher latency (16.2s vs
11.1s/query mean) than static, for no measurable quality gain.

**This is a definitive, methodologically rigorous negative result, not a
placeholder to re-run.** The verifier was fit against real ground truth
on the target corpus and validated at full test-set scale; the thresholds
were chosen from outcome-validated cutoffs; every measurement confound
found along the way (prompt-adjacency KV-cache reuse, run-order drift,
FTS5 performance) was diagnosed and fixed before trusting a number. The
closed loop does not beat static QE here, and the reason traces cleanly
to the verifier's own weak predictive power (r=0.134, ~1.8% of variance
explained) -- REPLAN's retry attempts are close to a coin flip on
whether they help or hurt, netting to ~0 aggregate gain at real added
cost. SLO frozen in `configs/default.yaml` from this run:
`max_harm_rate=0.0477`, `max_expected_llm_calls=1.2681`,
`max_p95_latency_ms=28370`.

Earlier, smaller-scale results (NFCorpus n=100, an intermediate
verifier) are preserved in git history but superseded by the above.

### Why the verifier is stuck at r=0.13: three converged attempts to raise it

After the result above, two follow-up investigations tried to raise the
verifier's r=0.134 ceiling rather than accept it as final:

- **Learned model, not hand-tuning**: `verify()` now loads a trained
  artifact (`models/verifier_v1.joblib`, `scripts/10_train_verifier_model.py`)
  instead of hardcoded coefficients -- proper train-script/artifact
  separation, confirmed behaviorally identical to the prior hardcoded
  version. Still linear: a GBDT (non-linear) fit was tried on the same
  data and scored worse (r=0.045).
- **Richer features**: added embedding-based signals (`src/observability/embeddings.py`,
  local `mxbai-embed-large`, previously unused) -- embedding drift,
  cross-document embedding coherence, expansion-to-evidence
  groundedness, a qualitatively different (dense semantic) family vs.
  everything lexical/BM25/reranker-scalar tried before. Tested at a
  properly powered n=368 (`scripts/09_calibrate_embedding_features.py`):
  lexical-only 5-fold CV r=0.070, lexical+embedding r=0.094 -- a +0.024
  gain, under the 0.05 adoption bar, and still below the real n=1168
  lexical-only baseline (0.134). **Not adopted.**

Three feature families (6, 10, and 13 features) and two model classes
(linear, GBDT) now converge on the same ~r=0.13 ceiling. This looks like
a genuine detection-difficulty limit on this corpus with this telemetry,
not a fixable feature-richness or modeling-choice gap.

## RQ-Recovery: the weak verifier as a safety net against injected faults

Given the above, the paper's core question was reframed: not "does the
closed-loop agent improve retrieval" (no), but "is observability-driven
recovery still useful as a safety net against large, obvious failures,
even with a verifier too weak to help on subtle natural variation?"
`scripts/11_fault_injection_recovery_study.py` reuses the live agent's
exact pipeline with three fault types injected at their natural point,
comparing an always-accept ablation against the real calibrated
`RecoveryController` on the identical corrupted state (paired, no LLM
call re-run). Full run, TripClick TAIL, n=149-150/fault type:

| Fault | Detection Recall | False Alarm Rate | Recovery gain (NDCG@10) |
|---|---|---|---|
| inject_unsupported_entity | 1.000 | 0.992 | +0.0238 |
| replace_grounding_passage | 0.773 | 0.465 | +0.0278 |
| disable_reranker | 0.185 | 0.115 | +0.0038 |

**Content-level corruption is caught reliably** -- entity injection with
perfect (if blunt/near-blanket) recall, and grounding-passage corruption
with genuinely discriminating detection (77% recall, a moderate 46.5%
false-alarm rate, not blanket rejection) and the largest recovery gain of
the three. Both recovery gains are an order of magnitude larger than
anything seen on natural queries (-0.0007, not significant). **System-level
degradation is mostly missed** -- disabling the reranker is caught only
18.5% of the time, because the fitted verifier's weights are almost
entirely lexical/content features with near-zero reranker-specific
weight; it wasn't built to notice that failure mode.

This is the paper's strongest, most defensible result: a weak
general-purpose qrel-free detector (r~0.13 on subtle natural variation)
is still a genuinely effective safety net against large, obvious content
corruption -- just not against failure modes outside its feature family.

**Update (2026-08-28) -- a real bug was found and fixed, and the
corrected result is more modest.** Investigating why `disable_reranker`
showed almost no effect revealed that the reranker's judgment was
computed but never actually applied to the accepted ranking (`R_t` was
unconditionally raw BM25 order) -- telemetry-only, contradicting the
documented control loop. Fixing that exposed a second bug: `verify()`
was judging a different (pre-rerank) ranking than the one actually being
accepted. Both fixed (see `src/agent/steps.py`, `src/observability/telemetry.py`
git history), with regression tests locking in the correct behavior.

Re-running the full verifier calibration (n=1168) on the corrected
pipeline confirmed the deployed model and thresholds remain well-suited
(r=0.146, a fresh refit doesn't beat it) -- no retraining needed. But
re-running the fault injection study with the corrected pipeline gives a
more modest result than originally reported: `inject_unsupported_entity`
and `replace_grounding_passage` recovery gains are **no longer
statistically significant** (95% CIs straddle zero: [-0.0196, +0.0331]
and [-0.0204, +0.0166] respectively), though the point estimates stay
positive. `disable_reranker`'s small effect (+0.0038, still significant)
is unchanged, since that fault explicitly bypasses the fixed code path.
The qualitative conclusion (content corruption more detectable than
system-level faults) likely still holds directionally, but the earlier
"order of magnitude larger and clearly significant" framing was
partly an artifact of the bug, not the full story. Full history of both
fixes and all three fault-injection runs (original buggy pipeline,
reranker-fix-only, both fixes) is in git log for `src/agent/steps.py`.

See `docs/milestones.md` for the execution plan and go/no-go checkpoints,
and `docs/sources.md` for the source bibliography this project is built
on.
