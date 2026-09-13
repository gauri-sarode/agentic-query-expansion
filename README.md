# Observe, Expand, Recover

Observability-guided control for LLM query expansion.

LLM query expansion is usually applied unconditionally or not at all. This
project treats it instead as a **bounded, observable agentic decision**: an
agent observes retrieval telemetry, selectively invokes a corpus-grounded
LLM expansion operator, observes the consequences of that action through
runtime telemetry, and decides to accept, roll back, replan, or abstain —
under an explicit budget, with no relevance judgments (qrels) available at
inference time.

Three ideas stay fused rather than separate: LLM query expansion is the
**task mechanism**, an agentic control loop is the **decision process**,
and AI observability is the **feedback/control layer** — telemetry is
causal input to the agent's policy and verifier, not just logging.

## Central finding

Qrel-free telemetry recorded *before* any action is reasonably predictive
of whether retrieval has already failed (AUC 0.649), but the same
vocabulary, recorded *after* observing a specific action's consequences,
is markedly weaker at telling whether that action helped (AUC 0.552).
**Detection is not actionability.** Neither a static expansion pipeline
nor a closed-loop recovery agent improves on a properly-tuned retriever on
ordinary queries, despite a retrospective oracle showing substantial,
mostly-uncaptured control headroom. The same ordering — pre-action
telemetry beats post-action telemetry at predicting intervention utility —
replicates on an independent second corpus (BEIR Natural Questions),
ruling out a TripClick-specific artifact.

## Research questions

- **RQ1 (Detection vs. Actionability)** — Can qrel-free retrieval and
  agent-runtime telemetry (a) identify that unexpanded retrieval is
  already unhealthy, and (b) predict whether a *specific* expansion action
  is likely to help or harm — and, if both are possible, is the second any
  easier than the first once the first is solved?
- **RQ2 (Control)** — Does an agent that conditions expansion actions on
  that telemetry outperform an otherwise identical *static*,
  failure-conditioned pipeline — same retriever, grounding, generator, and
  fusion, differing only in whether a closed loop can observe an action's
  consequences and recover from a bad one?
- **RQ3 (Recovery)** — After a harmful rewrite or a simulated system-level
  fault, can observability-driven verification detect the degradation and
  select a recovery action to restore retrieval quality, under a bounded
  budget — even for a verifier whose general-purpose detection power (RQ1)
  is only weakly informative?

## System design

Agent state at step `t`:

```
s_t = (q0, q_t, R_t, O_t, H_t, B_t)
```

`q0` immutable original query, `q_t` active search expression, `R_t`
current ranking, `O_t` observability state, `H_t` action/observation
history, `B_t` remaining action budget.

Control loop:

```
OBSERVE -> DIAGNOSE -> ACT -> RETRIEVE -> OBSERVE -> VERIFY
        -> {ACCEPT, ROLLBACK, REPLAN, STOP}
```

The critical experimental contrast: a **static** comparator gets the same
query, initial ranking, evidence, generator, and verifier, but executes
exactly one predetermined route. The agent's second decision depends on
evidence its own first action created. Isolating that dependency — not
simply comparing against unmodified BM25 — is the paper's central
experimental contrast.

Actions: `NO_EXPAND, VOCABULARY, CONTEXT, AMBIGUITY, ENTITY_NORMALIZE,
ROLLBACK, REPLAN, STOP`. The LLM is invoked through exactly one typed tool
(`generate_expansion`); it is never the sole arbiter of its own output's
use — that judgment is a separate, non-generative controller.

Observability model: one end-to-end trace per query episode
(`retrieve_original -> build_evidence -> plan_action -> generate_expansion
-> retrieve_expanded -> rerank -> verify -> recover_or_accept -> fuse`),
with three telemetry families — retrieval (score margins, entropy, term
coverage, entity overlap, top-k overlap, reranker disagreement, ranking
stability under perturbation), agent (selected operator, trajectory
length, drift from original, budget remaining), and system (latency,
token counts, tool errors/timeouts). Every feature is computable without
relevance judgments. See `docs/observability_model.md`.

Failure taxonomy (see `docs/failure_taxonomy.md`): vocabulary mismatch,
underspecification, ambiguity, entity mismatch, intent drift, confirmation
bias, retriever disagreement, tool degradation, budget exhaustion, healthy
retrieval.

## Datasets

| Dataset | Role | Status |
|---|---|---|
| **TripClick** | Primary benchmark — 1,523,878 docs, head/torso/tail frequency buckets, 1,175 queries/split/bucket | Access granted — non-commercial research use, dataset itself may not be redistributed |
| **BEIR Natural Questions** | Independent second-corpus replication of the RQ1 core finding | Public — no native train/val/test split; we construct a fixed-seed (42) disjoint split ourselves |
| NFCorpus (BEIR) | Fast dev loop / engineering iteration | Public |

TripClick terms prohibit redistributing the corpus. This repo publishes
**code, configs, and aggregated results only**; raw TripClick data stays
out of version control (see `data/README.md`).

## Frozen experimental protocol

Every reported test-set number follows a committed protocol
(`configs/experimental_protocol.yaml`, and its NQ analogue
`configs/nq_protocol_frozen.json` / `configs/nq_verifier_frozen.json`):
validation data may be used for any retrieval-config, verifier, or
threshold decision; test data may be touched only after every upstream
decision is frozen, with no decision afterward conditioned on a test-set
outcome. The NQ replication's statistical-power extension
(`configs/nq_test_extension_precommit.json`) follows the same discipline:
pre-committed to a fixed additional query pool *before* touching any of
it, no early stopping, no retuning.

## Repo layout

```
src/
  config.py                 -- run knobs: datasets, budgets, model IDs, paths
  retrieval/                -- BM25 index/search, cross-encoder reranker
  expansion/                -- typed expansion operators + prompts
  agent/                    -- state, actions, controller/policy, control loop, verify, fuse
  observability/            -- trace schema, telemetry feature extraction, trace store
  faults/                   -- seeded fault injection for the recovery study
  eval/                     -- metrics, SLO thresholds, bootstrap/CI analysis
configs/                     -- pinned model/budget/threshold configs, frozen protocols
scripts/                     -- dataset setup, index builds, experiment runners (numbered, chronological)
docs/                        -- observability model, failure taxonomy, milestones, sources
data/                        -- gitignored; local indexes/caches/downloaded corpora
tests/
paper/                       -- LaTeX source and compiled PDF
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Results

### A corrected retrieval baseline

BM25 over an unstemmed FTS5 index scored nDCG@10 0.228 on TAIL test — 15%
below the 0.267 published reference for this exact split and label type.
Selecting entirely on validation, porter stemming and a 10:1 title:body
field weight close the gap (val nDCG@10 0.235 → 0.327); a single,
one-time sanity check against withheld test gives nDCG@10 0.315, above
the published reference. Every downstream conclusion in this paper
changed once this fix was in place — it is a precondition for the rest of
the study, not an implementation footnote.

### RQ1: Detection is not actionability

Frozen TAIL test (n=1157), both models touched once:

| Metric | Test value |
|---|---|
| Failure-detection AUC (is retrieval already unhealthy?) | **0.649** [0.618, 0.680] |
| Action-utility AUC (will this specific action help?) | 0.552 [0.518, 0.586] |
| Action-utility Pearson r vs. true ΔnDCG@10 | 0.114 (p < 0.001) |

Pre-action telemetry is moderately predictive of *whether* retrieval is
failing; the same vocabulary, recorded *after* observing a specific
action's consequences, is markedly weaker at predicting *whether that
action helped*.

### RQ2: Control

Full TripClick TAIL test (n=1175), corrected baseline:

| System | nDCG@10 | Recall@100 | MRR |
|---|---|---|---|
| BM25 only | 0.3152 | 0.7750 | 0.3014 |
| BM25 + rerank, no QE | 0.2850 | 0.7750 | 0.2701 |
| BM25 + QE, no rerank | 0.2992 | 0.7716 | 0.2856 |
| Static QE (QE + rerank) | 0.3159 | 0.7759 | 0.2993 |
| Agent | 0.3161 | 0.7770 | 0.3015 |

Neither Static QE nor Agent significantly outperforms BM25 alone, and
Agent does not significantly outperform Static QE. The QE × reranking
interaction is real and significant (paired bootstrap, `scripts/13`):
+0.0468 [+0.0303, +0.0630], excluding zero — reranking alone and
expansion alone are each independently harmful, but combined they roughly
cancel out, a genuine interaction rather than one component offsetting a
fixed cost from the other.

A retrospective oracle shows the null result is not "no opportunity
exists": oracle-static beats BM25 by +0.0246 nDCG@10 (p < 0.001), while
the qrel-free verifier captures only 0.5–11.8% of that headroom across
selective-intervention quotas (`scripts/14`).

### RQ3: Which failures are recoverable

Six fault types tested (three used to derive a diagnostic rule, three
held out and never used to derive it): sorting all six by a closed-form
net-contribution statistic (`Σ_j w_j·Δx_j`, computed from paired
clean-vs-faulty telemetry, no relevance labels needed) reproduces the
observed detection-recall ordering with no exceptions. Human intuition
about the three held-out faults, tested prospectively, went 1-for-3 — the
statistic is a post-probe characterization tool, not a zero-shot
predictor: it needs a fault's own telemetry shift to compute.

### Distribution shift: does the verifier generalize beyond TAIL?

The action-utility verifier, calibrated on TAIL validation only, is
tested unmodified on HEAD and TORSO test splits: AUC falls from 0.552 on
TAIL to 0.234 on TORSO and 0.143 on HEAD — both CIs entirely below
chance, i.e. anti-correlated with true utility, not merely uninformative.
A verifier calibrated on one query-frequency regime does not just fail to
add value in another; its ranking inverts.

### Independent second-corpus replication: BEIR Natural Questions

To check whether the RQ1 finding is specific to TripClick, the core
detection-vs-actionability comparison is replicated on BEIR Natural
Questions — an open-domain corpus unrelated to TripClick's clinical
queries, with its own fresh, fixed-seed val/test split, fresh BM25
config, and fresh failure/action-utility detectors (never reusing
TripClick's fitted weights). Frozen test, pooled to n=2,652 after a
pre-committed statistical-power extension:

| Corpus | Static QE gain | Oracle gain | Failure AUC | Action-utility AUC |
|---|---|---|---|---|
| TripClick | +0.0007 (NS) | +0.0246* | 0.649 [0.618, 0.680] | 0.552 [0.518, 0.586] |
| BEIR NQ | +0.0033 (NS) | +0.0409* | 0.670 [0.649, 0.690] | 0.550 [0.515, 0.585] |

The same ordering replicates, decisively: a paired bootstrap of the AUC
difference on the acted-upon subset (n=2,599) excludes zero (+0.115, 95%
CI [+0.075, +0.154]). This is an independent second-corpus replication,
not evidence of broad cross-domain generalization — one further corpus,
one further query population.

## Two real defects, found by distrusting convenient results

Both surfaced only from refusing to accept a result that looked coherent
rather than broken:

- **Under-tuned BM25 baseline** — every downstream comparison was
  inflated until corrected on validation alone (see above).
- **Reranker-wiring defect** — the reranker's judgment was computed but
  never actually applied to the accepted ranking (`R_t` stayed
  unconditionally raw BM25 order); fixing it exposed a second bug where
  `verify()` was judging a different (pre-rerank) ranking than the one
  actually accepted. Both fixed, with regression tests locking in correct
  behavior — see `src/agent/steps.py` git history.

See `docs/milestones.md` for the execution plan and go/no-go checkpoints,
and `docs/sources.md` for the source bibliography this project is built
on.
