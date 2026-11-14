# Headline claims → code and outputs

Every number in the WSDM draft (`paper_wsdm/main.tex`) comes from a committed script; outputs in `docs/` are rounded aggregates only (TripClick terms). "Exploratory" means the test set had been inspected; "pre-registered" means the protocol was committed before download.

| Claim (paper section) | Status | Script | Output |
|---|---|---|---|
| Failure-only ceiling 8–57%, Table 1 (§5) | pre-committed / pre-registered rows | `scripts/65_core_claim_audit.py`, `scripts/76_iv_analysis.py` | `docs/core_claim_audit.json`, `docs/{dbpedia,hotpotqa}_summary.json` |
| Failure vs helped AUC, Table 2 (§6) | exploratory; last four rows pre-registered | `scripts/65_core_claim_audit.py`, `scripts/68_fiqa_analysis.py`, `scripts/76_iv_analysis.py` | as above |
| Figure 1 decile profiles | exploratory (NQ, TripClick); pre-registered runs (DBPedia, HotpotQA) | `scripts/70_score_sufficiency_reference.py`, `scripts/78_iv_profile_bands.py`, `scripts/71_fig_score_profile.py` | `docs/score_sufficiency_reference.json`, `docs/iv_profile.json` |
| Reachability: rescue rates, AUC of S for reachable (§6) | exploratory (4 corpora), pre-registered P5 (2) | `scripts/73_mechanism.py`, `scripts/76_iv_analysis.py` | `docs/mechanism.json`, `docs/*_summary.json` |
| P1–P5, Table 3 (§8); bootstrap redoes the cross-fit per resample | pre-registered (`configs/info_value_precommit.json`) | `scripts/76_iv_analysis.py` (`_exact_avg` inside the resample loop) | `docs/dbpedia_summary.json`, `docs/hotpotqa_summary.json` |
| Raw failure-score routing minus random (§8, §10) | exploratory | `scripts/79_raw_routing_check.py` | `docs/raw_routing_check.json` |
| HotpotQA telemetry 2.8 s beyond BM25, reranking 0.22 s (§8) | exploratory | `scripts/80_hotpotqa_latency_split.py`; `scripts/77_iv_retime_telemetry.py` | `docs/hotpotqa_latency_split.json` |
| Shapley attribution of m = p_F p_R mu_R + (1 - p_F) mu_H (§8); exact sum, 24 orderings, bootstrap CIs; decile endpoints | exploratory (`configs/actionability_protocol.json`) | `scripts/81_actionability.py` (`components`, `shapley`, `heterogeneity`) | `docs/actionability.json` (`heterogeneity.tertiles`, `.shapley`, `.deciles`) |
| Information vs policy regret, ten-bin class, 65–93% (in-sample) / 65–99% (cross-fitted) (§3, §10) | exploratory | `scripts/81_actionability.py` (`regret`) | `docs/actionability.json` (`regret_S`) |
| Best trained pre-action predictor closes 5–15% (41% HotpotQA) of the oracle–random gap (§10) | exploratory | `scripts/81_actionability.py` | `docs/actionability.json` (`routing_gain`, `regret_S.V_oracle`, `V_random`) |
| RCBR vs matched single model, Holm p = 1.0; reachability head AUC 0.59–0.70 (§10) | exploratory | `scripts/81_actionability.py` | `docs/actionability.json` (`primary_holm_RCBR_minus_Mono`, `reachability_auc`) |
| Cheap benefit gate vs S, paired +0.001 [−0.0002, 0.003], 0.3 ms/query; 75% of always-rerank gain at 50% (§8) | exploratory | `scripts/82_cheap_gate_check.py`; `scripts/81_actionability.py` (`cost`) | `docs/cheap_gate_check.json`, `docs/actionability.json` |
| Probe routing (§7) | exploratory (`configs/partial_probe_protocol.json`) | `scripts/58_probe_scores.py`, `scripts/60_partial_probe.py` | repository docs |
| Routing table and cost curves (moved out of the paper) | pre-committed / exploratory | — | `docs/routing_table.tex`, `docs/probe_cost_figure.tex` |
