#!/usr/bin/env python3
"""Held-out fault-coverage prediction test: can RQ3's causal-attribution
mechanism (scripts that produced results/tripclick-tail_feature_attribution.json,
Delta x_j = E[x_j^fault - x_j^clean], Contribution_j = w_j * Delta x_j)
predict in advance whether a brand-new, not-yet-used fault type will be
recoverable -- not just explain the three faults it was built to explain?

PROTOCOL: the prediction below was written down BEFORE this script was
run, using only the frozen verifier's fitted weights
(models/verifier_tripclick_tail_val_v1.joblib) and domain reasoning
about the fault's mechanism -- no data from this fault type was looked
at first.

Fault: remove_high_idf_term (src/faults/injection.py) -- implemented
from the start of the project but never wired into a study script.
Drops the single most discriminative (highest-IDF) term from the QUERY
itself, before retrieval: a vocabulary-mismatch/underspecification-style
corruption, mechanistically distinct from the three faults RQ3 already
covers (all of which corrupt mid-pipeline -- the expansion output, the
evidence, or the reranker component -- never the retrieval input
itself).

PREDICTION (frozen weights, no data from this fault seen yet):
  score_margin_gain +0.0004   coverage_gain      +0.2051
  coherence_gain    -0.0049   stability          -0.0001
  drift             +0.0023   disagreement       -0.2528
  topk_overlap      +0.0748   reranker_top_score -0.0009
  reranker_score_margin +0.0008   query_length_ratio +0.0055
  Removing the query's most discriminative term starves the LLM of the
  cue that most determines what a corpus-supported expansion should
  cover, so the resulting expansion should cover less of the (true,
  clean) query's vocabulary than a clean expansion would -- coverage_gain
  should drop sharply relative to the clean run. Since coverage_gain
  carries the largest positive weight, this predicts a negative net
  contribution: the fault should be DETECTABLE, and (unlike
  disable_reranker) not actively misleading -- i.e. closer in profile to
  inject_unsupported_entity (high recall, but false-alarm-swamped net
  effect) than to disable_reranker (actively disguised as healthy).

RESULT (n=79, one run, reported honestly either way): net contribution
was negative (-0.0236) as predicted, and detection recall was high
(0.800) -- the qualitative direction and the "closer to
inject_unsupported_entity than disable_reranker" profile call were both
right (recall 0.800, false alarm 0.694, recovery gain +0.0012 -- a
near-zero net effect from high recall swamped by high false alarms,
the same pattern as inject_unsupported_entity, not a clean win like
replace_grounding_passage). The specific MECHANISM predicted was only
partly right, though: coverage_gain contributed in the predicted
direction but was not the dominant feature -- topk_overlap dominated
instead (-0.0212 vs coverage_gain's -0.0067), because truncating the
query before retrieval changes which documents get retrieved as
candidates in the first place, not just how well the final expansion
covers the query's vocabulary. Reported in the paper's Limitations
section as a partial-credit result: the causal-attribution
methodology's qualitative predictions generalized to a held-out fault,
its specific single-feature story did not.

Because this fault corrupts the query before retrieval (unlike the
other three, which corrupt mid-episode), the faulty pass reruns
retrieve_and_observe_initial on the corrupted query -- its own initial
observation o0 is the correct "before" baseline for that run's telemetry
deltas, exactly mirroring how the clean run's o0 is its own baseline.
"original_ndcg" (the harm/help reference point) is always the TRUE
clean query's initial retrieval, since that is what the fault denies
the user.

Reduced sample (n=80, seed=42, same TAIL test population and seed as
the three-fault attribution run) -- a diagnostic/prediction-check pass,
not a new inferential claim on the frozen protocol
(configs/experimental_protocol.yaml); not part of RQ1-RQ3's frozen
touches of test.

Usage: PYTHONPATH=. python3 scripts/12_held_out_fault_prediction.py
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
from src.faults import injection as faults
from src.llm_client import LLMUnavailableError

DB_PATH = "data/tripclick_bm25.sqlite3"
N_SAMPLE = 80
SEED = 42
_EPS = 1e-9
_PREDICTION = "negative net contribution, dominated by coverage_gain -> detectable, entity-injection-like profile"


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

        corrupted_text = faults.remove_high_idf_term(text, DB_PATH, hash(qid) % (2**31))
        init_faulty = retrieve_and_observe_initial(corrupted_text, DB_PATH, k=100)
        evidence_faulty = build_evidence(init_faulty.r0_ids, DB_PATH)
        state_faulty = AgentState(
            q0=corrupted_text, q_t=corrupted_text, R_t=init_faulty.r0_ids, O_t=init_faulty.o0, B_t=2
        )

        try:
            clean_result = act_once(action, state_clean, init_clean.r0, DB_PATH, evidence_clean, 100, True)
            faulty_result = act_once(action, state_faulty, init_faulty.r0, DB_PATH, evidence_faulty, 100, True)
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

    print(f"\n=== remove_high_idf_term causal attribution (n={len(feature_deltas)}) ===")
    print(f"{'feature':22s} {'w_j':>9s} {'Δx_j':>9s} {'w_j·Δx_j':>10s}")
    for name, contrib in ranked:
        print(f"{name:22s} {weights[name]:+9.4f} {mean_delta[name]:+9.4f} {contrib:+10.4f}")
    print(f"\nnet contribution (sum over all 10 features): {net_contribution:+.4f}")
    print(f"PREDICTED: {_PREDICTION}")
    print(f"ACTUAL sign: {'negative (direction CORRECT)' if net_contribution < 0 else 'positive (direction WRONG)'}")

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
        "fault_type": "remove_high_idf_term",
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
        "prediction_direction_correct": bool(net_contribution < 0),
    }
    with open("results/tripclick-tail_held_out_fault_prediction.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nsaved to results/tripclick-tail_held_out_fault_prediction.json")


if __name__ == "__main__":
    main()
