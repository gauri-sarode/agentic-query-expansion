# Handoff — 2026-11-08

Status snapshot for picking this back up cold. Everything listed below is
committed and pushed up to `648cf73`; the 2026-11-08 revision commits
are local until pushed.

**Read first: history was rewritten on 2026-11-03** (see "TripClick
license scrub" below). Any clone older than that must re-clone or
`git reset --hard origin/main`; every commit hash before `648cf73`
changed, and hashes in this doc are the post-rewrite ones.

## Paper state

WSDM short paper, **"Query Performance Prediction Is Not Benefit
Prediction: Observability for Selective Retrieval Interventions"**
(retitled 2026-11-09 when reranking and PRF were added; before that
"…for Selective LLM Query Expansion", and originally "…: The
Actionability Gap") (`paper_wsdm/main.tex`, `main.pdf`). Page budget:
4 content pages, references + Ethical Considerations on page 5 (WSDM's
limit excludes both). Holds after every edit below.

Core claim (C1-C3): qrel-free retrieval telemetry predicts *whether a
query is failing* (detection AUC) significantly better than *whether
acting on it will help* (action-utility AUC) — the "actionability gap" —
on two unrelated corpora and under both sparse and dense retrieval.

## Done since the 2026-10-30 handoff

- **NQ dense-retrieval replication — complete, reproduces.**
  - Frozen scope written *before* any test touch:
    `configs/nq_dense_precommit.json` (`0b343ca`). Fit = NQ val (n=800);
    frozen test = the 2,052-query extension pool (sha256-pinned); the
    n=600 pilot excluded; LogisticRegression only; scripts/28 paired
    bootstrap; "reproduces" iff the difference CI's lower bound > 0.
  - Index: `scripts/41` built the full 2,681,468-doc bge-small index
    (finished 2026-11-01 07:59). `be5dcee` fixed an MPS caching-allocator
    leak in `build_embeddings` (driver memory 4→17 GB, swap thrash) with
    `torch.mps.empty_cache()` per batch.
  - Episodes: `scripts/43_nq_dense_retrieval_full_scale.py` (`f8947da`),
    800 val + 2,052 test, 0 LLM errors (737 / 1,932 acted).
  - Result (`results/nq_dense_retrieval_result.json`): detection 0.688
    [0.662, 0.713] vs action-utility 0.611 [0.583, 0.639]; paired
    difference +0.084 [+0.045, +0.123] on n=1,932 → **reproduces**.
    Dense nDCG@10 0.500 vs frozen BM25 0.308; dense failure rate 26%.
- **TripClick dense CI corrected.** scripts/36's "paired" bootstrap held
  detection AUC fixed — not paired. `scripts/44` (`ead6a93`) recomputes it
  the scripts/28 way on the acted subset (n=1,050): +0.022 [-0.028,
  +0.072] — *not* significant. The paper says so (`cfd8fc0`).
- **Paper integration** (`35c87d5`, `120e3ec`): dense paragraph covers
  both corpora with a retriever-strength explanation (NQ dense > BM25,
  TripClick dense < BM25 → gap resolves where dense is strong); Figure 1
  is a 4-row forest plot (TripClick, NQ, Dense TripClick, Dense NQ);
  abstract, C3, related work, limitations, conclusion, experimental
  integrity all updated. Did **not** claim BGE was/wasn't trained on NQ
  (unverified).
- **Anonymized mirror** (`anonymous.4open.science/r/agentic-query-expansion-8939/`)
  resynced by the user 2026-11-02 and verified byte-identical for the
  new scripts/configs/paper. Content is unchanged by the scrub below, so
  it needs no resync for that. Expires 2027-03-14.

## TripClick license scrub (2026-11-03)

TripClick terms (tripdatabase.github.io/tripclick): non-commercial
research only; no part of the dataset may be publicly shared except
aggregated findings/visualizations; **statistical models or resources
created from it may not be shared publicly without the data owners'
permission**; a violation ends usage rights automatically.

Audit: no raw docs/queries/qrels/IDs tracked anywhere in history;
`models/*.joblib` gitignored (the August `verifier_v1.joblib` leak was
already scrubbed). One remaining violation: old commit `1676e2d`
hardcoded the 10 TripClick-fitted verifier coefficients in
`src/agent/verify.py`. Fixed with `git filter-repo --replace-text`
(values → `None  # redacted ...`, now commit `f943c20`), force-pushed;
cited hashes in docs/configs/scripts remapped in `648cf73`. Current tree
byte-identical to before. Local backup bundle deleted.

**Purge requested (2026-11-08):** GitHub still served five pre-scrub
commits by direct URL, all with the 10 coefficients: `1676e2d`,
`f5f82c4`, `c819cc1`, `e8aacc2`, `86681ec` (found by checking every old
SHA in `.git/filter-repo/commit-map`). A GitHub Support ticket asks to
remove cached views and garbage-collect them, plus all unreachable
objects. No forks, no PRs. Once Support confirms, check that
`gh api repos/<owner>/agentic-query-expansion/commits/1676e2d`
returns 404.

Rule going forward: never commit anything fit on TripClick (weights,
coefficients, joblib, per-query outputs). Aggregate metrics and scalar
thresholds are fine. NQ (BEIR) artifacts are not under this restriction.

## Machine notes (16 GB M-series Mac)

- `caffeinate` does **not** stop lid-close sleep on battery — an
  overnight run stalled 8h this way. Keep the lid open / on AC, or
  `sudo pmset -a disablesleep 1`.
- One pool job at a time; other repos (e.g. `~/dev/o-drift`) also run
  long jobs here — check `ps` before starting anything heavy.
- A standalone `caffeinate -d -i -s` (PID 2777) from the NQ build is
  still running; safe to `kill 2777`.

## Loose ends, not blocking

- Untracked `logs/` (NQ build/episode logs, launcher script) and
  `paper 5.33.28 PM.zip` in the repo root — never explained, untouched.
- `scripts/41`'s printed docs/s and ETA are wrong after any pause
  (wall-clock `t0` never resets).

## Revision after the simulated WSDM review (2026-11-08)

Simulated 5-seat review (ARS academic-paper-reviewer, scratchpad only):
Major Revision. All must-fix items addressed in `paper_wsdm/main.tex`.

- Pre-committed the analyses first: `configs/revision_analyses_precommit.json`
  (`fbe56ac`), then `scripts/46_revision_analyses.py` ->
  `results/revision_analyses.json` (local, gitignored). `scripts/45` re-ran
  frozen NQ BM25 retrieval to store score lists for the QPP baseline.
- **The central claim changed.** The harm-label AUC gap holds in 3/4
  settings, but it is largely structural: 84-94% of failing queries stay
  at nDCG 0 after expansion (ties, "not harmed"), so the two labels are
  coupled. On queries whose nDCG changed (post hoc), the BM25 gaps vanish;
  only dense NQ keeps one (+0.122 [+0.047,+0.197]).
- New robust finding: a perfect failure detector recovers only 42%
  (TripClick) / 33% (NQ) of the oracle's gain; the rest is on healthy
  queries and needs action-utility prediction.
- NQC alone matches our detector (TripClick 0.647 vs 0.649).
- Correction: NQ's stored `accepted_ndcg` is the verifier-GATED policy,
  not unconditional QE. Unconditional (unfused) QE on NQ hurts (-0.022,
  significant); failure-selective beats it (+0.015). The NQ BM25 run
  never reranked (`use_reranker=False`); the paper now says so.
- Pre-revision paper kept only in the session scratchpad; git has it.

## External review response (2026-11-09)

An outside reviewer-style read (Borderline Accept) raised: reranker
confound, AUC != intrinsic difficulty, benefit evaluated as harm, fused
vs unfused policies, TripClick test history, a missing related-work citation.
- Reranker: original rankings are never reranked, expanded ones are
  (TripClick, dense). Pre-committed (`configs/reranker_matched_precommit.json`)
  then ran `scripts/47` on frozen records only: TripClick BM25 labels
  against BM25+rerank of the original. Harm AUC 0.552 -> 0.583; gap to
  detection +0.091 -> +0.059 [+0.012, +0.107], still significant;
  helped-vs-rest falls to 0.504. Static QE vs BM25+rerank +0.031.
  Aggregates in `docs/reranker_matched_summary.json`. Dense has no
  reranked-original arm (would need a new test touch).
- Related-work citation: added, then removed at the author's request.
- Text: claim reworded to "a failure detector is not a substitute for a
  benefit predictor"; helped-vs-rest range in abstract; TripClick called
  partly exploratory, NQ the confirmation; pilot stated disjoint;
  per-corpus policy comparisons; operators named.
- Not changed: BEIR "Heterogenous" is the original paper's spelling.
- Content still 4 pages; references now spill to page 5.

## Scope extension: reranking and PRF (2026-11-09)

Added to broaden the paper beyond LLM QE.
- Pre-commit `configs/intervention_scope_precommit.json` (`57107ea`),
  amended twice: PRF grid widened to 3x4x9 = 108 settings before any PRF
  run; the failed rerank check (below) recorded in `8f54afe` before any
  test label was read.
- `scripts/48_intervention_runs.py` (rankings, no labels) and
  `scripts/49_intervention_analysis.py` (`select` on validation, then one
  test pass). Rankings live in `results/interventions/` (gitignored).
  Aggregates: `docs/intervention_summary.json`.
- `data/tripclick_bm25.sqlite3` had become a 0-byte file; rebuilt from
  `data/tripclick_raw` (17 min). BM25 reproduces stored nDCG exactly (max
  diff 0.0) on val and test of both corpora.
- PRF selected on validation: TripClick (d=20, t=10, lam=0.9), where no
  setting beat BM25 on val; NQ (20, 30, 0.6).
- Rerank check FAILED: 17/1175 TripClick test top-10 identical to the
  stored arm. Cause: the stored arm (2026-09-03) predates `8831fbf`
  (2026-09-14), which removed the literal `<eot>` from reranker input;
  re-scoring 50 queries with the old text matched 50/50. The current text
  is primary; the paper discloses 0.285 (old) vs 0.287 (new).
- Results (test): detection beats harm AUC in all four new settings
  (+0.075 to +0.179, CIs exclude 0); perfect-detector share rerank 52%/57%,
  PRF 8%/32%; NQ rerank always +0.225 vs failure-gated +0.114 vs
  benefit-gated +0.223; TripClick rerank always -0.028, failure-gated
  -0.010 (CI excludes 0), benefit-gated -0.001.
- Paper: retitled, abstract/C1-C3/conclusion rewritten, new Setup
  paragraph, Section 6 "Result 3: Beyond LLM QE" with Table 3; the Qini
  figure was dropped for space (its 12% number is inline). Still 4
  content pages.

## Decision analyses after the second external review (2026-11-09)

Pre-commit `configs/decision_analyses_precommit.json` (`0d98fb3`), written
before computing; `scripts/50_decision_analyses.py`; aggregates in
`docs/decision_analyses_summary.json`. All POST HOC on already-inspected
test traces; no new runs (the dense reranker-matched rerun was dropped
to avoid another test touch).
- D1 failure-only ceiling (identity + cost term): share 8% (PRF TC) to
  57% (rerank NQ); tracks rescue rate (0.8-46%) vs healthy-help rate.
  At kappa=0.02 the perfect failure gate goes negative for PRF (both)
  and dense TC QE.
- D2 budget-matched pre-action routing: failure/NQC/benefit all near
  random, far below oracle; benefit beats failure only on rerank NQ
  (+0.0088 [+0.0033, +0.0150], Holm-significant). NQ LLM QE planner
  replay aligned (13/13).
- D3 post-action verification reported separately (100% compute).
- D4 Holm: 7/8 detection-minus-harm gaps survive (dense TC does not);
  6/8 persist at |Delta| thresholds 0.05/0.10; changed-only gaps vanish
  in all four new settings.
- Paper restructured: Framework section (pi*(Z), identity), Section 5
  ceiling table, merged 8-row AUC table, routing table + budget figure,
  evidence ledger. Content ends on page 3. Same title.
- Not done: CODS citation (author's decision: not archived yet);
  independent held-out confirmation (no unused queries remain).
- Mirror blocker: `paper/main.tex` (an older draft) shows the
  author email and coauthor name on the anonymized mirror; add those
  as masked terms or remove `paper/` before submission.

## Possible next steps

- Re-review the revised draft (ARS re-review mode).
- Resync the anonymized mirror after pushing.
- If page budget allows: nothing else is pending experimentally. Any new
  test touch needs a precommit first, same discipline as above.
