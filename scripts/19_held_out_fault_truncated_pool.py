#!/usr/bin/env python3
"""Third held-out fault-coverage prediction test (second of the
follow-up pair extending scripts/12's n=1 to n=2 additional held-out
faults, alongside scripts/18).

Fault: truncate_retrieval_pool (src/faults/injection.py) -- implemented
from the start but never wired into a study script. Simulates a
retrieval backend returning fewer candidates than configured (e.g. a
misconfigured page size). Implemented here as a smaller-$k$ retrieval
rather than a post-hoc slice of a $k=100$ pool: BM25 is a deterministic
ranking function, so truncate_retrieval_pool(search(q, k=100), 20) and
search(q, k=20) return identical top-20 results -- the smaller-$k$
retrieval is the simpler, self-consistent way to get telemetry that is
natively computed over the truncated pool (rather than reusing $k=100$
telemetry and hoping it still applies), not a different fault.

This is mechanistically distinct from all four faults tested so far
(scripts/12 corrupts the query pre-retrieval by removing a term,
scripts/18 corrupts it by adding one; the original three corrupt the
expansion output, the evidence, or disable a component): here nothing
about the query or the pipeline's logic changes, only how much of the
BM25 ranking the rest of the pipeline is allowed to see.

PREDICTION (frozen weights, no data from this fault seen yet; lower
confidence than scripts/12 or scripts/18, stated honestly): unlike
disable_reranker, this fault does not sever a specific wiring
connection the verifier's dominant feature (disagreement) depends on,
and unlike the query-corruption faults it does not directly damage
vocabulary-coverage features. We predict a SMALL-MAGNITUDE net
contribution (closer to grounding corruption's near-zero regime than
to the two query-corruption faults' clearly negative one), because the
reranker and coverage features are computed natively over the smaller
pool on both sides of the comparison rather than being pushed sharply
in one direction. Correspondingly we predict recall closer to a
coin-flip than to entity-injection's near-ceiling recall.

Usage: PYTHONPATH=. python3 scripts/19_held_out_fault_truncated_pool.py
"""
from __future__ import annotations

import json
import random

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score

from src.agent.actions import EXPANSION_ACTIONS
from src.agent.controller import ActionSelector, RecoveryController
from src.agent.fuse import fuse
from src.agent.state import AgentState
from src.agent.steps import act_once, build_evidence, retrieve_and_observe_initial
from src.agent.verify import _features
from src.data import load_qrels, load_queries
from src.eval.metrics import evaluate
from src.eval.slo import detection_recall, false_alarm_rate
from src.llm_client import LLMUnavailableError

DB_PATH = "data/tripclick_bm25.sqlite3"
N_SAMPLE = 80
SEED = 42
TRUNCATED_K = 20
_EPS = 1e-9
_PREDICTION = "small-magnitude net contribution (near-zero regime like grounding corruption); recall closer to a coin-flip than to entity injection's near-ceiling recall"


def _rank_to_scores(ranking: list[str]) -> dict[str, float]:
    return {d: 1.0 / (i + 1) for i, d in enumerate(ranking)}


def _ndcg10(qid: str, ranking: list[str], qrels: dict) -> float:
    return evaluate({qid: qrels[qid]}, {qid: _rank_to_scores(ranking)})[qid]["ndcg_cut_10"]


def main() -> None:
    queries = load_queries("tripclick-tail")
    qrels = load_qrels("tripclick-tail")
    all_qids = [qid for qid in queries if qid in qrels]
    rng = random.Random(SEED)
    sample_qids = rng.sample(all_qids, N_SAMPLE)
    print(f"n sample queries: {len(sample_qids)}", flush=True)

    selector = ActionSelector()
    recovery_controller = RecoveryController()

    feature_deltas = []
    harmful = healthy = detected_and_harmful = detected_and_healthy = 0
    ndcg_original, ndcg_without, ndcg_with = [], [], []
    verifier_scores, harmful_labels = [], []
    skipped = 0

    for i, qid in enumerate(sample_qids, 1):
        text = queries[qid]
        init_clean = retrieve_and_observe_initial(text, DB_PATH, k=100)
        action = selector.select(text, init_clean.o0)
        if action not in EXPANSION_ACTIONS:
            skipped += 1
            continue
        evidence_clean = build_evidence(init_clean.r0_ids, DB_PATH)
        state_clean = AgentState(q0=text, q_t=text, R_t=init_clean.r0_ids, O_t=init_clean.o0, B_t=2)

        # Faulty: same query, only the retrieval pool is truncated.
        init_faulty = retrieve_and_observe_initial(text, DB_PATH, k=TRUNCATED_K)
        evidence_faulty = build_evidence(init_faulty.r0_ids, DB_PATH)
        state_faulty = AgentState(
            q0=text, q_t=text, R_t=init_faulty.r0_ids, O_t=init_faulty.o0, B_t=2
        )

        try:
            clean_result = act_once(action, state_clean, init_clean.r0, DB_PATH, evidence_clean, 100, True)
            faulty_result = act_once(action, state_faulty, init_faulty.r0, DB_PATH, evidence_faulty, TRUNCATED_K, True)
        except LLMUnavailableError:
            skipped += 1
            continue

        clean_features = _features(init_clean.o0, clean_result.candidate.O_t)
        faulty_features = _features(init_faulty.o0, faulty_result.candidate.O_t)
        feature_deltas.append({k: faulty_features[k] - clean_features[k] for k in clean_features})

        original_ndcg = _ndcg10(qid, init_clean.r0_ids, qrels)
        without_recovery_ranking = fuse(init_faulty.r0_ids, faulty_result.candidate.R_t)
        decision = recovery_controller.decide(faulty_result.candidate, faulty_result.verifier_score)
        with_recovery_ranking = (
            fuse(init_faulty.r0_ids, faulty_result.candidate.R_t) if decision is None else init_faulty.r0_ids
        )

        n_without = _ndcg10(qid, without_recovery_ranking, qrels)
        n_with = _ndcg10(qid, with_recovery_ranking, qrels)
        actually_harmful = n_without < original_ndcg - _EPS
        detected = decision is not None

        ndcg_original.append(original_ndcg)
        ndcg_without.append(n_without)
        ndcg_with.append(n_with)
        verifier_scores.append(faulty_result.verifier_score)
        harmful_labels.append(1 if actually_harmful else 0)
        if actually_harmful:
            harmful += 1
            if detected:
                detected_and_harmful += 1
        else:
            healthy += 1
            if detected:
                detected_and_healthy += 1

        if i % 10 == 0:
            print(f"  [{i}/{len(sample_qids)}]", flush=True)

    print(f"\nskipped: {skipped}", flush=True)
    n = harmful + healthy

    artifact = joblib.load("models/verifier_tripclick_tail_val_v1.joblib")
    feature_names = artifact["feature_names"]
    weights = dict(zip(feature_names, artifact["model"].coef_))
    mean_delta = {k: float(np.mean([d[k] for d in feature_deltas])) for k in feature_names}
    contribution = {k: weights[k] * mean_delta[k] for k in feature_names}
    net_contribution = sum(contribution.values())
    ranked = sorted(contribution.items(), key=lambda kv: -abs(kv[1]))

    print(f"\n=== truncate_retrieval_pool (k={TRUNCATED_K}) causal attribution (n={len(feature_deltas)}) ===")
    print(f"{'feature':22s} {'w_j':>9s} {'Δx_j':>9s} {'w_j·Δx_j':>10s}")
    for name, contrib in ranked:
        print(f"{name:22s} {weights[name]:+9.4f} {mean_delta[name]:+9.4f} {contrib:+10.4f}")
    print(f"\nnet contribution (sum over all 10 features): {net_contribution:+.4f}")
    print(f"PREDICTED: {_PREDICTION}")

    print(f"\n=== recovery study (n={n}, harmful={harmful}, healthy={healthy}) ===")
    recall = detection_recall(detected_and_harmful, harmful)
    fa = false_alarm_rate(detected_and_healthy, healthy)
    mean_orig = sum(ndcg_original) / n
    mean_without = sum(ndcg_without) / n
    mean_with = sum(ndcg_with) / n
    print(f"  Detection Recall = {recall:.3f}")
    print(f"  False Alarm Rate = {fa:.3f}")
    print(
        f"  Mean NDCG@10: original={mean_orig:.4f}  without_recovery={mean_without:.4f}  "
        f"with_recovery={mean_with:.4f}"
    )
    print(f"  Recovery gain (with - without): {mean_with - mean_without:+.4f}")

    auc = None
    if 0 < sum(harmful_labels) < len(harmful_labels):
        auc = roc_auc_score(harmful_labels, [-s for s in verifier_scores])
        print(f"  Verifier action-utility AUC on this fault alone: {auc:.3f}")

    report = {
        "fault_type": "truncate_retrieval_pool",
        "truncated_k": TRUNCATED_K,
        "n": n,
        "skipped": skipped,
        "harmful": harmful,
        "healthy": healthy,
        "detection_recall": recall,
        "false_alarm_rate": fa,
        "mean_ndcg_original": mean_orig,
        "mean_ndcg_without_recovery": mean_without,
        "mean_ndcg_with_recovery": mean_with,
        "recovery_gain": mean_with - mean_without,
        "verifier_auc": auc,
        "mean_delta": mean_delta,
        "contribution": contribution,
        "net_contribution": net_contribution,
        "prediction": _PREDICTION,
    }
    with open("results/tripclick-tail_held_out_fault_truncated_pool.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nsaved to results/tripclick-tail_held_out_fault_truncated_pool.json")


if __name__ == "__main__":
    main()
