# Handoff — 2026-10-30

Status snapshot for picking this back up cold. Everything below "Done"
is committed locally on `main`; nothing has been pushed to `origin`
since `fee4389` (8 commits ahead as of this doc).

## Done (committed, not pushed)

All in `paper_wsdm/` unless noted. Commit hashes oldest → newest:

- `879cbcc` — Tested a non-linear (GBDT) model for the detection task
  (the failure-selective-QE side), mirroring the existing action-utility
  non-linear check. Reported in Section 4: GBDT beats logistic
  regression on validation CV, consistently across 6 seeds on both
  corpora (+0.023 TripClick, +0.025 NQ), but that advantage does **not**
  survive a single frozen-test touch on either corpus (TripClick 0.655
  vs. 0.649; NQ 0.680 vs. 0.680). Reported as a finding, not adopted —
  Table 1's numbers are unchanged.
- `253e236` — The dense-retrieval section used to hedge ("consistent
  with, not proof of" the bi-encoder being a weaker retriever). Computed
  the actual number from already-frozen data: mean nDCG@10 0.217
  (dense) vs. 0.315 (tuned BM25) on TripClick TAIL test. Confound now
  stated as confirmed, not assumed.
- `5d7a019` — Verified all 12 `references.bib` entries against source
  (arXiv/ACL Anthology/DOI). Fixed two: `bge-embedding`'s title had two
  wrong words and silently dropped 2 of 6 real authors; `beir`'s title
  now matches the paper's actual (non-standard) spelling, "Heterogenous."
- `f11c96d` — Added `scripts/41` (NQ dense-corpus build, see "Running
  now" below) and `scripts/42` (sweep of 8 non-linear classifiers for
  the detection task, validation-CV only, no test touch — tree
  ensembles consistently beat logistic ~0.02-0.025 AUC on both corpora;
  other non-linear families don't. Deliberately not acted on further —
  picking a "best" model from this sweep and touching test with it
  would be the exact post-hoc model selection the paper's own
  "Experimental integrity" paragraph argues against.)
- `70ccc85` — Real bug fix in `src/retrieval/dense_index.py`:
  `build_embeddings` used to `fetchall()` the entire corpus table into
  Python memory before embedding anything. Fine at TripClick's 1.52M
  docs; drove this 16GB machine into severe swap thrashing at BEIR NQ's
  2.68M-doc scale. Now streams via a cursor (`fetchmany` per batch).
- `40e036d`, `9985550`, `bc7b8fc` — Title/framing pass. Final WSDM paper
  title: **"Observability-Guided LLM Query Expansion in Search: The
  Actionability Gap"** (observability + ML-adjacent + LLM + search, per
  explicit request; "machine learning" was tried and then explicitly
  dropped). README retitled and reframed to match verbatim, including
  restoring the three-pillar framing (QE task / agentic control loop /
  observability feedback layer) after an earlier misreading briefly
  stripped the observability angle out entirely.

Also earlier in this session (already committed before this doc existed):
Figure 1/2 redesigned as forest plots (fixed a real pgfplots bar/error-bar
misalignment bug), Figure 3 added (oracle-vs-verifier quota-sweep line
chart, replacing a table), the verified-current anonymized-repo link
added to the paper, and the final title change to "The Actionability
Gap: When Should Search Agents Expand a Query?" (since superseded by the
observability-inclusive title above).

Page budget throughout: holds at 4 content pages + references/Ethical
Considerations overflow onto page 5 (WSDM's limit excludes those two
sections) — verified after every edit in this stack, no regressions.

## Running now

**`scripts/41_nq_dense_build_full_corpus.py`** — building the full BEIR
NQ dense corpus (2,681,468 docs, `BAAI/bge-small-en-v1.5`, float16
memmap) at the same scale as the existing TripClick dense-retrieval
check, for a second-corpus dense-retrieval replication (currently the
paper's dense-retrieval check is TripClick-only).

- Log: `/tmp/nq_dense_build2.log`. Checkpointed/resumable
  (`data/nq_dense_embeddings.f16` + `data/nq_dense_ids.txt`).
- **Reliability history, read before trusting a silent "it's running":**
  died once from resource contention with an unrelated job (`~/dev/o-drift`,
  a different repo) running concurrently on this machine — paused/resumed
  manually around that a few times. Died a **second** time with no error
  in the log at all (just a benign semaphore-cleanup warning), most
  likely an OS-level kill under memory pressure — restarted from
  checkpoint, currently healthy (~43-70 docs/s observed post-restart).
- A `Monitor` task is currently watching: fires on either the `[dense_index]
  done` completion line OR the process disappearing without one (the
  second death produced no error text, so a pure error-grep watcher
  would have stayed silent forever — this one polls process liveness
  directly instead).
- Standalone `caffeinate -d -i -s` is running independent of the job's
  own process tree, so system sleep isn't a risk even if the job's own
  wrapper dies again.
- The script's own printed `docs/s` and `ETA` fields go garbage after
  any pause/resume cycle (wall-clock `t0` never resets, so paused time
  gets counted as near-zero throughput) — don't trust them. Measure real
  throughput from raw doc-count deltas over fresh wall-clock samples
  instead.
- Rough ETA as of this doc: **2-4 days**, bursty (7-70 docs/s observed
  across samples), not a single confident number.

## Next (not started yet)

The embedding build (`scripts/41`) only produces the dense index.
**The actual NQ dense-retrieval episode pipeline does not exist yet** —
scripts 33-36 (TripClick's pilot telemetry / corpus-scoping / pilot
runner / full-scale runner) are all hardcoded to TripClick's data paths,
configs, and query splits. Once `scripts/41` finishes, still need:

1. An NQ-equivalent of `scripts/35`/`36`: run episodes over NQ's val
   (n=800) and test (n=600, or possibly the n=2,052 extension pool for
   parity with the paper's main NQ numbers — not yet decided) query
   populations against the new dense index, computing detection AUC and
   action-utility AUC the same way the TripClick dense check does.
2. Decide the frozen population size/scope for this touch *before*
   running it, same discipline as every other result in this paper —
   this hasn't been written down yet and should be before any code runs
   against frozen NQ test data.
3. Once results land: likely a new subsection or extension to the
   existing "Does this hold under dense retrieval?" part of Section 4,
   same page-budget constraints as everything else (currently zero
   slack — expect to need trims elsewhere, same pattern as every other
   addition in this stack).

## Also unresolved, not blocking

- Two untracked zip files in repo root (`paper.zip`, `paper 5.33.28 PM.zip`)
  — flagged early in this session, never explained or cleaned up. Not
  touched.
- Nothing has been pushed to `origin`, and the anonymized
  `anonymous.4open.science` mirror has not been resynced since before
  this entire commit stack — both need an explicit go-ahead.
