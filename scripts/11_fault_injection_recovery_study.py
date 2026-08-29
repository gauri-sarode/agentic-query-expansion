#!/usr/bin/env python3
"""RQ-Recovery experiment (docs/failure_taxonomy.md fault injection
protocol): does observability-driven recovery (the calibrated
RecoveryController, src/agent/controller.py) actually catch and recover
from LARGE, INJECTED failures -- even though the verifier can't reliably
detect subtle natural variation (r~0.13, scripts/08_calibrate_verifier.py)?
This is the pivot from RQ-Control (agent vs. static on natural queries --
not significant, see README) toward the one RQ pillar never validated
end to end. Detecting an obvious, deliberately-injected corruption is a
different (plausibly much easier) regime than detecting whether a normal
LLM rewrite subtly helped or hurt.

For each query and fault type: run ONE fault-injected expansion attempt
through the real retrieve/rerank/telemetry/verify pipeline
(src/agent/steps.act_once, same code the live agent uses), with the fault
applied at the point that type of fault actually corrupts (evidence
before generation, the generated expansion after it, or the reranker
tool itself). Then compare two decisions on that SAME corrupted state,
for an exactly paired comparison (no repeat LLM call, no extra noise):

  - "without recovery": an ablation RecoveryController with an
    unreachable accept threshold -- always ACCEPT, i.e. no
    observability-driven intervention at all.
  - "with recovery": the real calibrated RecoveryController.

Ground truth for whether the fault was actually harmful: does the
without-recovery (always-accept) outcome's NDCG@10 fall below the
original (pre-expansion) ranking's NDCG@10? This is the brief's "clean
paired episode for the same query" -- the original ranking is that clean
reference. Metrics: Detection Recall, False Alarm Rate, and the core
comparison -- mean NDCG with vs. without observability-driven recovery
(the brief's "strongest test": full observable agent > same agent
without observability-driven recovery).

Usage: python scripts/11_fault_injection_recovery_study.py [dataset] [n_queries] [--resume]

Checkpoints every 50 queries to
results/{dataset}_fault_injection_checkpoint.json (raw per-episode NDCG
triples per fault type, for both aggregate stats and later paired
bootstrap significance testing) so a multi-hour run at large n survives
an interruption via --resume.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import src.agent.steps as steps_module
from src.agent.actions import EXPANSION_ACTIONS
from src.agent.controller import ActionSelector, RecoveryController
from src.agent.fuse import fuse
from src.agent.state import AgentState
from src.data import load_qrels, load_queries
from src.eval.metrics import evaluate
from src.eval.slo import detection_recall, false_alarm_rate
from src.faults import injection as faults
from src.llm_client import LLMUnavailableError

_DB_PATHS = {
    "nfcorpus": "data/nfcorpus_bm25.sqlite3",
    "tripclick-head": "data/tripclick_bm25.sqlite3",
    "tripclick-torso": "data/tripclick_bm25.sqlite3",
    "tripclick-tail": "data/tripclick_bm25.sqlite3",
    "tripclick-tail-val": "data/tripclick_bm25.sqlite3",
}

_FAULT_TYPES = ["inject_unsupported_entity", "replace_grounding_passage", "disable_reranker"]
_EPS = 1e-9
_CHECKPOINT_EVERY = 50


def _rank_to_scores(ranking: list[str]) -> dict[str, float]:
    return {d: 1.0 / (i + 1) for i, d in enumerate(ranking)}


def _ndcg10(qid: str, ranking: list[str], qrels: dict) -> float:
    return evaluate({qid: qrels[qid]}, {qid: _rank_to_scores(ranking)})[qid]["ndcg_cut_10"]


def _run_faulty_attempt(action, state, r0, db_path, evidence, seed, fault_type, k=100):
    """One act_once call with the given fault applied at its natural
    injection point. Monkeypatches src.agent.steps.generate_expansion for
    the post-generation fault, restoring it afterward regardless of
    outcome.
    """
    use_reranker = fault_type != "disable_reranker"
    faulty_evidence = evidence
    if fault_type == "replace_grounding_passage":
        faulty_evidence = faults.replace_grounding_passage([d for d, _ in r0][:len(evidence)], db_path, seed)

    original_generate = steps_module.generate_expansion
    if fault_type == "inject_unsupported_entity":

        def _corrupted_generate(action, q0, evidence):
            return faults.inject_unsupported_entity(original_generate(action, q0, evidence), seed)

        steps_module.generate_expansion = _corrupted_generate

    try:
        return steps_module.act_once(action, state, r0, db_path, faulty_evidence, k, use_reranker)
    finally:
        steps_module.generate_expansion = original_generate


def _empty_results() -> dict:
    return {
        ft: {
            "harmful": 0,
            "healthy": 0,
            "detected_and_harmful": 0,
            "detected_and_healthy": 0,
            "ndcg_original": [],
            "ndcg_without_recovery": [],
            "ndcg_with_recovery": [],
        }
        for ft in _FAULT_TYPES
    }


def _checkpoint_path(dataset: str) -> Path:
    return Path(f"results/{dataset}_fault_injection_checkpoint.json")


def _save_checkpoint(dataset: str, results: dict, last_index: int, skipped_no_expand: int, skipped_llm_error: int) -> None:
    path = _checkpoint_path(dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "dataset": dataset,
                "results": results,
                "last_index": last_index,
                "skipped_no_expand": skipped_no_expand,
                "skipped_llm_error": skipped_llm_error,
            }
        )
    )


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--resume"]
    resume = "--resume" in sys.argv
    dataset = args[0] if len(args) > 0 else "nfcorpus"
    n = int(args[1]) if len(args) > 1 else 100
    db_path = _DB_PATHS[dataset]

    queries = load_queries(dataset)
    qrels = load_qrels(dataset)
    query_ids = [qid for qid in queries if qid in qrels][:n]

    selector = ActionSelector()
    recovery_controller = RecoveryController()

    results = _empty_results()
    skipped_no_expand = 0
    skipped_llm_error = 0
    start_index = 0

    if resume:
        checkpoint_path = _checkpoint_path(dataset)
        if checkpoint_path.exists():
            saved = json.loads(checkpoint_path.read_text())
            if saved.get("dataset") == dataset:
                results = saved["results"]
                start_index = saved.get("last_index", 0)
                skipped_no_expand = saved.get("skipped_no_expand", 0)
                skipped_llm_error = saved.get("skipped_llm_error", 0)
                print(f"Resuming from checkpoint: {start_index}/{len(query_ids)} query_ids already processed.", flush=True)
            else:
                print("WARNING: checkpoint dataset differs -- starting fresh.", flush=True)
        else:
            print("--resume requested but no checkpoint found -- starting fresh.", flush=True)

    t_start = time.time()

    for i, qid in enumerate(query_ids[start_index:], start_index + 1):
        text = queries[qid]
        init = steps_module.retrieve_and_observe_initial(text, db_path, k=100)
        action = selector.select(text, init.o0)
        if action not in EXPANSION_ACTIONS:
            skipped_no_expand += 1
            continue

        original_ndcg = _ndcg10(qid, init.r0_ids, qrels)
        evidence = steps_module.build_evidence(init.r0_ids, db_path)

        for fault_type in _FAULT_TYPES:
            state = AgentState(q0=text, q_t=text, R_t=init.r0_ids, O_t=init.o0, B_t=3)
            seed = hash((qid, fault_type)) % (2**31)
            try:
                result = _run_faulty_attempt(action, state, init.r0, db_path, evidence, seed, fault_type)
            except LLMUnavailableError:
                skipped_llm_error += 1
                continue

            without_recovery_ranking = fuse(init.r0_ids, result.candidate.R_t)  # always accept
            decision = recovery_controller.decide(result.candidate, result.verifier_score)
            with_recovery_ranking = (
                fuse(init.r0_ids, result.candidate.R_t) if decision is None else init.r0_ids
            )

            ndcg_without = _ndcg10(qid, without_recovery_ranking, qrels)
            ndcg_with = _ndcg10(qid, with_recovery_ranking, qrels)

            actually_harmful = ndcg_without < original_ndcg - _EPS
            detected = decision is not None  # any intervention (ROLLBACK/REPLAN/STOP), not just ACCEPT

            r = results[fault_type]
            r["ndcg_original"].append(original_ndcg)
            r["ndcg_without_recovery"].append(ndcg_without)
            r["ndcg_with_recovery"].append(ndcg_with)
            if actually_harmful:
                r["harmful"] += 1
                if detected:
                    r["detected_and_harmful"] += 1
            else:
                r["healthy"] += 1
                if detected:
                    r["detected_and_healthy"] += 1

            print(
                f"{qid:>8}  fault={fault_type:<26} harmful={actually_harmful!s:<5} "
                f"detected={detected!s:<5} ndcg orig={original_ndcg:.3f} "
                f"no-rec={ndcg_without:.3f} with-rec={ndcg_with:.3f}",
                flush=True,
            )

        if i % 25 == 0 or i == len(query_ids):
            elapsed = time.time() - t_start
            done_this_session = i - start_index
            rate = elapsed / done_this_session
            eta_min = rate * (len(query_ids) - i) / 60
            print(
                f"  [progress {i}/{len(query_ids)}  {rate:.1f}s/query  "
                f"elapsed={elapsed / 60:.1f}min  eta={eta_min:.1f}min]",
                flush=True,
            )
        if i % _CHECKPOINT_EVERY == 0:
            _save_checkpoint(dataset, results, i, skipped_no_expand, skipped_llm_error)
            print(f"  [checkpoint saved, last_index={i}]", flush=True)

    _save_checkpoint(dataset, results, len(query_ids), skipped_no_expand, skipped_llm_error)

    print(f"\nn_queries={len(query_ids)}  (skipped: {skipped_no_expand} NO_EXPAND, {skipped_llm_error} LLM errors)")
    for fault_type in _FAULT_TYPES:
        r = results[fault_type]
        n_total = r["harmful"] + r["healthy"]
        if n_total == 0:
            print(f"\n{fault_type}: no samples")
            continue
        mean_orig = sum(r["ndcg_original"]) / n_total
        mean_without = sum(r["ndcg_without_recovery"]) / n_total
        mean_with = sum(r["ndcg_with_recovery"]) / n_total
        print(f"\n=== {fault_type} (n={n_total}, harmful={r['harmful']}, healthy={r['healthy']}) ===")
        print(f"  Detection Recall   = {detection_recall(r['detected_and_harmful'], r['harmful']):.3f}")
        print(f"  False Alarm Rate   = {false_alarm_rate(r['detected_and_healthy'], r['healthy']):.3f}")
        print(f"  Mean NDCG@10: original={mean_orig:.4f}  without_recovery={mean_without:.4f}  with_recovery={mean_with:.4f}")
        print(f"  Recovery gain (with - without): {mean_with - mean_without:+.4f}")


if __name__ == "__main__":
    main()
