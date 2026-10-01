#!/usr/bin/env python3
"""Dense-retrieval pilot: does the detection-vs-actionability AUC gap
(Table 1 of the WSDM draft, BM25: 0.649 vs 0.552) hold when first-stage
retrieval is dense (BAAI/bge-small-en-v1.5) instead of BM25?

Scope, documented honestly rather than silently: a 300-query sample
(150 val, 150 test, fixed seed 42) against a SCOPED corpus (all
qrel-relevant docs for exactly these queries + 150k random distractors,
not the full 1.52M-doc TripClick collection -- see scripts/34), and a
REDUCED 6/9-feature telemetry set (scripts/33 -- drops the two features
that need BM25/FTS5's IDF machinery, which this pilot's lean index
deliberately doesn't build). This is a bounded pilot, not a full
replication of Table 1 under dense retrieval at the original scale.

Action selection reuses the exact same 4-threshold rule as the main
pipeline's ActionSelector (src/agent/controller.py), inlined here since
that class expects the full ObservabilityState dataclass this pilot's
reduced telemetry doesn't populate -- same decision logic, not a
divergent one.

Usage: PYTHONPATH=. python3 scripts/35_dense_retrieval_pilot.py
"""
from __future__ import annotations

import json
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.agent.actions import Action
from src.eval.metrics import evaluate
from src.expansion.operators import generate_expansion
from src.retrieval import dense_index
from src.retrieval.rerank import rerank

from importlib import import_module

tel = import_module("scripts.33_dense_retrieval_pilot_telemetry")

META_DB = "data/dense_pilot_meta.sqlite3"
EMBED_PATH = "data/dense_pilot_embeddings.f16"
IDS_PATH = "data/dense_pilot_ids.txt"
SEED = 42
N_BOOTSTRAP = 10000


def select_action(query: str, t: "tel.ReducedRetrievalTelemetry") -> Action:
    """Mirrors ActionSelector.select()'s 4-threshold rule exactly
    (src/agent/controller.py), using only the telemetry fields this
    pilot's reduced set actually has."""
    if t.entity_overlap < 1.0:
        return Action.ENTITY_NORMALIZE
    if t.query_term_coverage < 0.5:
        return Action.VOCABULARY
    if len(query.split()) <= 2 or t.score_entropy > 0.8:
        return Action.CONTEXT
    if t.result_coherence < 0.15:
        return Action.AMBIGUITY
    return Action.NO_EXPAND


def run_episode(qid: str, query: str, qrels_for_q: dict[str, int], index: dense_index.DenseIndex):
    r0 = index.search(query, k=100)
    r0_ids = [d for d, _ in r0]
    texts0 = dense_index.get_texts(r0_ids[:10], META_DB)
    t0 = tel.compute_reduced_retrieval_telemetry(query, r0, texts0)

    run0 = {qid: dict(r0)}
    orig_ndcg = evaluate({qid: qrels_for_q}, run0)[qid]["ndcg_cut_10"]
    y_poor = 1 if orig_ndcg <= 1e-9 else 0

    action = select_action(query, t0)
    record = {
        "qid": qid, "y_poor": y_poor, "action": action.value,
        "t0_features": tel.detection_feature_vector(t0),
        "orig_ndcg": orig_ndcg,
    }
    if action == Action.NO_EXPAND:
        return record

    evidence_ids = r0_ids[:5]
    evidence_texts = dense_index.get_texts(evidence_ids, META_DB)
    evidence = [evidence_texts[d] for d in evidence_ids if d in evidence_texts]

    try:
        q_t = generate_expansion(action, query, evidence)
    except Exception as e:
        record["llm_error"] = str(e)
        return record

    r1 = index.search(q_t, k=100)
    r1_ids = [d for d, _ in r1]
    candidate_texts = dense_index.get_texts(r1_ids[:50], META_DB)
    candidates = [(d, candidate_texts[d]) for d in r1_ids[:50] if d in candidate_texts]
    reranked = rerank(q_t, candidates)
    reranked_ids = [d for d, _ in reranked]
    reranked_set = set(reranked_ids)
    final_ids = reranked_ids + [d for d in r1_ids if d not in reranked_set]
    final_scores = dict(r1)
    final_ranking = [(d, final_scores.get(d, 0.0)) for d in final_ids]

    texts1 = dense_index.get_texts(final_ids[:10], META_DB)
    t1 = tel.compute_reduced_retrieval_telemetry(q_t, final_ranking, texts1)

    run1 = {qid: dict(final_ranking)}
    cand_ndcg = evaluate({qid: qrels_for_q}, run1)[qid]["ndcg_cut_10"]

    reranker_top_score = float(reranked[0][1]) if reranked else 0.0
    reranker_score_margin = float(reranked[0][1] - reranked[1][1]) if len(reranked) >= 2 else 0.0

    au_features = tel.action_utility_feature_vector(
        t0, t1, query, q_t,
        pre_rerank_ids=r1_ids[:50], post_rerank_ids=reranked_ids,
        r0_ids=r0_ids, r1_ids=r1_ids,
        reranker_top_score=reranker_top_score, reranker_score_margin=reranker_score_margin,
    )
    record.update({
        "candidate_ndcg": cand_ndcg,
        "delta_ndcg": cand_ndcg - orig_ndcg,
        "au_features": au_features,
    })
    return record


def main() -> None:
    sample = json.loads(open("results/dense_pilot_query_sample.json").read())
    index = dense_index.DenseIndex(EMBED_PATH, IDS_PATH)

    results = {"val": [], "test": []}
    for split, qids_key, queries_key, qrels_key in [
        ("val", "val_qids", "val_queries", "val_qrels"),
        ("test", "test_qids", "test_queries", "test_qrels"),
    ]:
        qids = sample[qids_key]
        queries = sample[queries_key]
        qrels = sample[qrels_key]
        t0 = time.time()
        for i, qid in enumerate(qids):
            rec = run_episode(qid, queries[qid], qrels[qid], index)
            results[split].append(rec)
            if (i + 1) % 25 == 0:
                print(f"[{split}] {i + 1}/{len(qids)} ({(time.time() - t0):.0f}s elapsed)", flush=True)
        print(f"[{split}] done: {len(results[split])} episodes in {time.time() - t0:.0f}s", flush=True)

    with open("results/dense_retrieval_pilot_raw.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda o: float(o) if hasattr(o, "item") else str(o))
    print("saved results/dense_retrieval_pilot_raw.json")

    # --- Detection AUC: val-fit (5-fold CV), test-score-once ---
    X_val = np.array([r["t0_features"] for r in results["val"]])
    y_val = np.array([r["y_poor"] for r in results["val"]])
    X_test = np.array([r["t0_features"] for r in results["test"]])
    y_test = np.array([r["y_poor"] for r in results["test"]])

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    det_clf = LogisticRegression(max_iter=1000)
    cv_proba = cross_val_predict(det_clf, X_val, y_val, cv=kf, method="predict_proba")[:, 1]
    cv_auc = roc_auc_score(y_val, cv_proba)
    det_clf.fit(X_val, y_val)
    test_proba = det_clf.predict_proba(X_test)[:, 1]
    det_auc = roc_auc_score(y_test, test_proba)
    print(f"\nDetection: val CV AUC={cv_auc:.4f}, test AUC={det_auc:.4f} (n={len(y_test)})")

    # --- Action-utility AUC: acted-upon subset only ---
    val_acted = [r for r in results["val"] if "au_features" in r]
    test_acted = [r for r in results["test"] if "au_features" in r]
    Xa_val = np.array([r["au_features"] for r in val_acted])
    ya_val = (np.array([r["delta_ndcg"] for r in val_acted]) < -1e-9).astype(int)
    Xa_test = np.array([r["au_features"] for r in test_acted])
    ya_test = (np.array([r["delta_ndcg"] for r in test_acted]) < -1e-9).astype(int)
    print(f"acted-upon: val={len(val_acted)}/{len(results['val'])}, test={len(test_acted)}/{len(results['test'])}")

    au_clf = LogisticRegression(max_iter=1000)
    au_clf.fit(Xa_val, ya_val)
    au_test_proba = au_clf.predict_proba(Xa_test)[:, 1]
    au_auc = roc_auc_score(ya_test, au_test_proba)
    print(f"Action-utility: test AUC={au_auc:.4f} (n={len(ya_test)})")

    # --- Bootstrap CIs ---
    def boot_ci(y, p):
        rng = np.random.RandomState(SEED)
        n = len(y)
        vals = []
        for _ in range(N_BOOTSTRAP):
            idx = rng.randint(0, n, n)
            yy, pp = y[idx], p[idx]
            if 0 < yy.sum() < n:
                vals.append(roc_auc_score(yy, pp))
        return np.percentile(vals, [2.5, 97.5])

    det_lo, det_hi = boot_ci(y_test, test_proba)
    au_lo, au_hi = boot_ci(ya_test, au_test_proba)
    print(f"\nDetection AUC: {det_auc:.4f} [{det_lo:.4f},{det_hi:.4f}]")
    print(f"Action-utility AUC: {au_auc:.4f} [{au_lo:.4f},{au_hi:.4f}]")
    print(f"Difference: {det_auc - au_auc:+.4f}")
    print("\nFor comparison, BM25 TripClick TAIL (full scale, main result): 0.649 vs 0.552")

    out = {
        "n_val": len(results["val"]), "n_test": len(results["test"]),
        "n_acted_val": len(val_acted), "n_acted_test": len(test_acted),
        "detection_auc": {"point": float(det_auc), "ci_lo": float(det_lo), "ci_hi": float(det_hi)},
        "action_utility_auc": {"point": float(au_auc), "ci_lo": float(au_lo), "ci_hi": float(au_hi)},
        "corpus_size": len(index.doc_ids),
    }
    with open("results/dense_retrieval_pilot_result.json", "w") as f:
        json.dump(out, f, indent=2, default=lambda o: float(o) if hasattr(o, "item") else str(o))
    print("saved results/dense_retrieval_pilot_result.json")


if __name__ == "__main__":
    main()
