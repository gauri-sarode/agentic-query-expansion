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

The critical comparison from the experiment matrix -- **observable agent
vs. the strong static failure-conditioned QE comparator**
(`scripts/06_agent_vs_static.py`), same operator-selection policy,
generator, verifier, and fusion mechanism for both (`src/agent/steps.py`)
-- run at n=100 on NFCorpus:

| System | NDCG@10 | Recall@100 | MRR |
|---|---|---|---|
| BM25 only | 0.332 | 0.252 | 0.542 |
| Static QE | 0.340 | 0.264 | 0.532 |
| Agent | 0.341 | 0.265 | 0.537 |

Both expansion pipelines beat plain BM25. But **agent vs. static is not
yet significant** (paired bootstrap on NDCG@10, mean diff 0.0014, 95% CI
[0.0000, 0.0042]) -- with the current rule-based v0 controller, REPLAN
fires on roughly 1% of queries, so the agent's ROLLBACK decisions land on
essentially the same fallback-to-original outcome as the static
comparator's REJECT. The closed loop isn't yet exercised enough to show
the paper's core claim; the next lever is the controller/verifier
thresholds, not the plumbing.

Cost is measurably different, though: an isolated, order-alternated
latency benchmark (`scripts/07_latency_benchmark.py`, n=50 paired
samples/pipeline, avoiding two real measurement confounds found along
the way -- see its docstring and git history) shows agent p95 latency
(4976ms) meaningfully above static's (3141ms), fully explained by the
agent's occasional second LLM call on REPLAN (mean diff 213ms, 95% CI
[18, 450], significant). SLO frozen in `configs/default.yaml`:
`max_harm_rate=0.19`, `max_expected_llm_calls=1.03`,
`max_p95_latency_ms=4976`.

See `docs/milestones.md` for the execution plan and go/no-go checkpoints,
and `docs/sources.md` for the source bibliography this project is built
on.
