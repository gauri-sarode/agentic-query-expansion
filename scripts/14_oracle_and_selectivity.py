#!/usr/bin/env python3
"""Recomputes RQ2's oracle-ceiling table AND the selectivity-frontier
table from the SAME frozen per-query data in one place, with a single
shared definition of "oracle," after a reviewer-caught inconsistency:
the oracle ceiling (max(BM25, treatment) per query, averaged) reported
+0.0246 nDCG@10, but the selectivity frontier's "oracle" column reported
+0.0250 at 20-40% budgets -- impossible if both are a budget-constrained
special case of the same unconstrained maximum, since a budget-capped
oracle can never beat the uncapped one. Both tables were originally
produced by separate, unsaved ad hoc scripts; this script is the single
saved source of truth for both, going forward.

Definitions (fixed here, both tables use these):
  benefit(q) = static_ndcg10(q) - baseline_ndcg10(q)
  oracle ceiling ("Oracle-static vs. BM25") = mean_q[ max(0, benefit(q)) ]
    -- the unconstrained best a per-query oracle can do, averaged over
    ALL n queries (queries where treatment doesn't help just keep BM25).
  selectivity frontier at budget X%: sort queries by a POLICY's ranking
  (oracle: by true benefit descending; verifier: by verifier score
  descending; random: shuffled), treat only the top X% with Static QE,
  leave the rest at BM25, report mean gain over always-BM25 across ALL
  n queries. By construction this must be monotonically bounded above
  by the unconstrained oracle ceiling at every budget, with equality
  only once the budget reaches (or exceeds, then plateaus at) the
  fraction of queries with positive benefit.

Reuses the FROZEN RQ2 checkpoint (baseline_run, static_run, agent_run)
and the frozen verifier's saved per-query scores from the fault-free
calibration run -- no new frozen-test touches, only a re-derivation.

Usage: PYTHONPATH=. python3 scripts/14_oracle_and_selectivity.py
"""
from __future__ import annotations

import json
import random

import numpy as np

from src.eval.bootstrap import paired_bootstrap
from src.eval.metrics import evaluate

CHECKPOINT_PATH = "results/tripclick-tail_agent_vs_static_checkpoint.json"
OUT_PATH = "results/tripclick-tail_oracle_and_selectivity_v2.json"
BUDGETS = [0.05, 0.10, 0.20, 0.40, 1.00]
SEED = 42


def main() -> None:
    frozen = json.loads(open(CHECKPOINT_PATH).read())
    baseline_run, static_run, agent_run = frozen["baseline_run"], frozen["static_run"], frozen["agent_run"]
    query_ids = sorted(baseline_run.keys())
    assert set(query_ids) == set(static_run.keys()) == set(agent_run.keys())
    print(f"n queries: {len(query_ids)}", flush=True)

    qrels_path_dataset = "tripclick-tail"
    from src.data import load_qrels

    qrels = load_qrels(qrels_path_dataset)

    baseline_eval = evaluate(qrels, baseline_run)
    static_eval = evaluate(qrels, static_run)
    agent_eval = evaluate(qrels, agent_run)

    b = np.array([baseline_eval[q]["ndcg_cut_10"] for q in query_ids])
    s = np.array([static_eval[q]["ndcg_cut_10"] for q in query_ids])
    a = np.array([agent_eval[q]["ndcg_cut_10"] for q in query_ids])

    # --- Table IV: unconstrained oracle ceiling ---
    oracle_static = np.maximum(b, s)
    oracle_agent = np.maximum(b, a)
    ceiling_vs_bm25 = paired_bootstrap(b.tolist(), oracle_static.tolist())
    ceiling_vs_static = paired_bootstrap(s.tolist(), oracle_static.tolist())
    ceiling_agent_vs_agent = paired_bootstrap(a.tolist(), oracle_agent.tolist())
    print("\n=== Table IV: oracle ceiling ===")
    print("Oracle-static vs BM25:", ceiling_vs_bm25)
    print("Oracle-static vs Static QE:", ceiling_vs_static)
    print("Oracle-agent vs Agent:", ceiling_agent_vs_agent)

    # --- Table V: selectivity frontier, using the SAME benefit definition ---
    benefit = s - b  # treatment = Static QE throughout, matching "Oracle-static"
    rng = random.Random(SEED)
    shuffled_order = list(range(len(query_ids)))
    rng.shuffle(shuffled_order)

    # Verifier scores from the static pass's own traces (the score that
    # gated real Static QE's accept/reject decision) -- NaN (no expansion
    # attempted, or LLM error) sorts last, contributing 0 benefit either way.
    qid_to_verifier_score = {}
    for trace in frozen["static_traces"]:
        vs = trace.get("verifier_score")
        qid_to_verifier_score[trace["query_id"]] = vs if vs is not None else float("-inf")
    verifier_score_arr = np.array([qid_to_verifier_score.get(q, float("-inf")) for q in query_ids])

    oracle_order = np.argsort(-benefit)  # descending true benefit
    verifier_order = np.argsort(-verifier_score_arr)  # descending verifier score
    random_order = np.array(shuffled_order)

    print("\n=== Table V: selectivity frontier ===")
    frontier = {}
    n = len(query_ids)
    for budget in BUDGETS:
        k = max(1, round(budget * n))

        def gain_for(order):
            top = set(order[:k].tolist() if hasattr(order, "tolist") else order[:k])
            treated_gain = sum(benefit[i] for i in top)
            return treated_gain / n

        oracle_gain = gain_for(oracle_order)
        verifier_gain = gain_for(verifier_order)
        random_gain = gain_for(random_order)
        frontier[f"{budget:.2f}"] = {
            "oracle": oracle_gain, "verifier": verifier_gain, "random": random_gain, "k": k,
        }
        print(f"budget={budget:.0%}  k={k}  oracle_gain={oracle_gain:+.4f}  "
              f"verifier_gain={verifier_gain:+.4f}  random_gain={random_gain:+.4f}")

    print(f"\nunconstrained oracle ceiling (Table IV) = {oracle_static.mean() - b.mean():+.4f}")
    print("sanity check: max over budgets of oracle_gain must be <= unconstrained ceiling")
    max_frontier_oracle = max(v["oracle"] for v in frontier.values())
    ceiling = oracle_static.mean() - b.mean()
    print(f"  max frontier oracle = {max_frontier_oracle:+.4f}, ceiling = {ceiling:+.4f}, "
          f"{'OK' if max_frontier_oracle <= ceiling + 1e-9 else 'STILL INCONSISTENT'}")

    out = {
        "n": n,
        "oracle_ceiling": {
            "oracle_static_vs_bm25": ceiling_vs_bm25,
            "oracle_static_vs_static": ceiling_vs_static,
            "oracle_agent_vs_agent": ceiling_agent_vs_agent,
        },
        "frontier": frontier,
        "consistency_check": {
            "max_frontier_oracle": max_frontier_oracle,
            "unconstrained_ceiling": ceiling,
            "consistent": bool(max_frontier_oracle <= ceiling + 1e-9),
        },
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
