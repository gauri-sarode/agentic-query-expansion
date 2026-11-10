# WSDM 2027: three new pre-action baselines and an untouched-benchmark experiment

## Objective and falsification
The paper currently claims that observed query-failure telemetry is not sufficient to route retrieval interventions effectively. We will challenge that claim with stronger pre-action benefit predictors, including learned nonlinear interactions, semantic similarity, and lower-tail benefit predictions. All outcomes and confidence intervals, including unfavorable ones, will be reported.

## Mandatory separation of three decision problems
1. **Pre-action routing**: Before computing reranking/PRF/LLM-QE, decide whether to pay for it. Available signals: original-query features and frozen query embedding, and no intervention-result features. Primary comparison uses identical intervention budgets.
2. **Post-action verification**: After computing the intervention, decide whether to keep its ranking. Report retention percentage separately from actual interventions computed (100% for a full-query verification pass).
3. **Perfect-failure / oracle analyses**: Use qrels to diagnose theoretical headroom, never as a feasible policy input.

## New baselines (pre-specified)
- **B1 nonlinear direct benefit**: LightGBM regression on pre-action telemetry, trained on per-query delta nDCG@10, one model per intervention. Only validation RMSE chooses between two pre-specified depth/regularization options.
- **B2 semantic nearest-neighbor uplift**: Frozen, pre-specified query encoder (recommend `intfloat/e5-base-v2` or `BAAI/bge-small-en-v1.5`, with the exact model revision/hash frozen); cosine k-NN of *training* queries' observed gains. k = 5,15,30,60 chosen on validation MSE. No test outcomes enter neighbor means.
- **B3 conservative lower-tail benefit**: LightGBM quantile regression on original-query telemetry; quantile 0.10/0.20/0.30 chosen using validation pinball loss. Rank by predicted conservative gain; at a budget cap, abstain if predicted gain does not exceed the cost threshold. **Do not assert conditional coverage or conformal guarantees**.

Existing comparators: logistic failure probability, NQC (negative score when larger NQC implies better retrieval), random, top-delta hindsight oracle. Optional extra baseline: cheap semantic-neighbor classifier trained to predict improvement. Do not claim hindsight oracle as deployable.

## Primary benchmark
**BEIR ArguAna**: 1,406 argument queries and 8,674 passages (counterargument retrieval). Unseen for this project if true. Important: queries also appear in the corpus; **exclude the query's own document from every retrieval and reranking run** before computing pre-action telemetry, PRF terms, ground-truth results, or fusion. Check duplicates, leakage, and near-duplicate argument pairs. Use standard qrel IDs; do not replace judgment semantics with relevance guessed by an LLM.

**Pre-specify train/validation/test partition before calculating relevance metrics or inspecting results.** Suggested deterministic split: 50% train, 20% validation, 30% test, group by argument topic/pair if reliable metadata exists; otherwise hash query IDs and audit duplicate topics. Aim approximately 703/281/422. Record exact IDs and SHA256 of the manifest. No changes to thresholds, models, target metric, or corpus rules based on test outputs.

**Secondary/optional**: BEIR SCIDOCS (1,000 queries, 25K docs) to assess a very different retrieval task, but only after the ArguAna protocol is sealed; the larger scientific value is an independent, meaningful finding rather than an extra benchmark count.

## Retrieval outputs and comparison
Compute BM25 original on full corpus and identical consistent intervention arms. Focus **reranking** as primary: a frozen cross-encoder with query+doc inputs, top-100 input, top-50 reordered, and same relevant original baseline for the intervention. Secondary: PRF with frozen parameters, LLM QE with frozen qwen2.5:3b and prompts. Avoid mixing query-expansion and reranker effects by adding a reranker-matched original; report pure QE increment and combined-pipeline increment separately.

For all action arms, cache complete rankings and per-query nDCG@10 in a single immutable output with SHA256. Label construction uses original and intervention outcomes. Cost: report actual GPU/CPU runtime and calls (median and P95) instead of treating the illustrative kappa as measured latency. Measure warm model-cache and hardware fairly. Define costs for query embeddings as pre-action overhead.

## Primary estimand and plots
At each exact fraction b in {10%,25%,50%,75%,100%} of the test queries, score every query with pre-action signals, intervene on the top floor(bN) queries, and report the gain in mean nDCG@10 over never intervening. Equal budgets make action costs comparable for one intervention. Report Area Under the Budget Curve (average of these five gains), per-intervention harm rate, and regret to per-query hindsight oracle.

**Budget-cap setting:** may intervene on <= bN queries but abstains on predicted gains <= cost threshold. Report selection rate, actual compute, net gain, and effect on quality; do not call retained post-action results an inference-cost saving.

## Primary comparisons (lock before unseen test)
- B1 vs learned failure router at 25% budget and average over budget grid, reranking on ArguAna.
- B2 vs learned failure router on identical grid, reranking on ArguAna.
- B3 vs B1 on capped budgets, report both retrieval gain and harm avoidance.
- All three vs NQC, random and hindsight oracle; PRF/LLM-QE are secondary.
Use paired query bootstrap (at least 10,000 resamples for final analysis), resampling queries and re-selecting the top bN inside each resample. Use the three primary contrasts in `primary_comparisons.csv` with Holm adjustment. The sign-flip test assumes approximate symmetry of per-query policy differences; report paired bootstrap confidence intervals and abstain from strong claims when the assumptions or repeated-test access undermine interpretation. Do not pick a winning baseline after seeing the test outcomes and call that pre-registered.

## Headroom identity (precise scope)
With F=1 when original nDCG@10 is zero, delta=after-before, and cost kappa >= 0:
G_oracle(kappa)=E[(delta-kappa)+]; G_failure(kappa)=E[(delta-kappa)F].
Their difference is E[(delta-kappa)+(1-F)]+E[(kappa-delta)+F].
This is an exact regret decomposition for the policy that intervenes on *every* failed query. At kappa=0 it is also a ceiling for policies restricted to intervening only on failed queries. It is **not** a universal ceiling for every learned failure-predictor policy or the optimal positive-cost policy.

## Interpretation rule
If all new baselines perform near random and far below oracle, the claim is strongly stress-tested; a new-benchmark result makes this more credible than reused-data analysis. If a stronger model reliably beats the failure gate, the scientific conclusion changes to: the gap is model- and signal-dependent, with nonlinear or semantic information helping in some regimes. Both outcomes are publishable, provided all results and protocol deviations are reported.

## Seven-day sprint
- **Nov 10** Freeze questions, dataset/split, encoder, reranker, prompts and evaluation; implement self-exclusion and unit tests.
- **Nov 11** Build original and consistent rerank/PRF rankings, store hashed caches; verify qrel alignment and no self retrieval.
- **Nov 12** Build B1 and failure/NQC comparators; train only on train, tune only on validation.
- **Nov 13** Build B2 (frozen embeddings + kNN) and B3 (quantile regression); run leakage and determinism tests.
- **Nov 14** Evaluate sealed ArguAna holdout once; retain every result and failure; if time, start SCIDOCS as declared secondary.
- **Nov 15** Statistical intervals, cost audit, cohort analyses by F, rescue, help, harm, and matching reranker conditions.
- **Nov 16** Rewrite short paper, fit four content pages, sanity-check mathematics and references, generate anonymous supplement.
- **Nov 17** Verify final CMT-uploaded PDF and submit by official 11:59pm AoE deadline.

## Existing datasets
All TripClick TAIL test and all 3,452 NQ queries were already used; no additional analysis of those queries is independently confirmatory. A new ArguAna evaluation is confirmatory only if its test labels/results have not been previously used to choose models, metrics, methods, or claims.

## Input schema for baseline_runner.py
One row per query, columns `query_id`, `split` in {train,val,test}, `baseline_ndcg`, `rerank_ndcg`, `prf_ndcg`, `qe_ndcg`, `z_*` for pre-action feature columns, `emb_*` for fixed query embedding dimensions, and optional `nqc_score` (larger=better original retrieval). Missing `z_*` or `emb_*` features are errors. Data producer must guarantee each intervention's ranking is properly aligned and uses no test-relevance signal to construct outputs. If an intervention was not run, use `--actions rerank` rather than leaving its outcome null.

Example: `python baseline_runner.py --input arguana_frozen.csv --output results_arguana --actions rerank,prf,qe --bootstrap 10000 --cost 0`

The runner ranks by pre-action predicted benefit, not by any post-action feature; its output includes predictions and paired bootstrap results. Report large-scale run hashes and the equipment details separately. Treat `bootstrap_sign_tail` as descriptive rather than a formal permutation p-value.
