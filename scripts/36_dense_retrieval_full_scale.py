#!/usr/bin/env python3
"""Full-scale version of scripts/35's dense-retrieval pilot: same
telemetry (scripts/33), same episode logic (reuses scripts/35's
run_episode/select_action directly rather than duplicating them), but
against the FULL 1.52M-document TripClick corpus and the FULL TAIL
val+test query populations (1175 + 1175 = 2350 episodes), not the
n=300 scoped-corpus pilot. This is what makes it a true parity
comparison against the main BM25 result (also n=1157/2013 on frozen
TAIL test) rather than a bounded pilot.

Checkpointed every 50 episodes per split (--resume to continue) --
this is realistically a 6+ hour run (2350 episodes at ~9.5s/episode
observed from the pilot), too long to risk losing to an interruption.

Usage: PYTHONPATH=. python3 scripts/36_dense_retrieval_full_scale.py [--resume]
"""
from __future__ import annotations

import json
import sys
import time
from importlib import import_module

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src import tripclick
from src.retrieval import dense_index

pilot = import_module("scripts.35_dense_retrieval_pilot")

META_DB = "data/dense_full_meta.sqlite3"
EMBED_PATH = "data/dense_full_embeddings.f16"
IDS_PATH = "data/dense_full_ids.txt"
CHECKPOINT_PATH = "results/dense_retrieval_full_scale_checkpoint.json"
SEED = 42
N_BOOTSTRAP = 10000
_CHECKPOINT_EVERY = 50

# scripts/35's run_episode() reads META_DB as a module-level constant;
# point it at the full-scale meta db for this run.
pilot.META_DB = META_DB


def _save_checkpoint(results: dict, progress: dict) -> None:
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump({"results": results, "progress": progress}, f,
                   default=lambda o: float(o) if hasattr(o, "item") else str(o))


def main() -> None:
    resume = "--resume" in sys.argv

    val_topics = tripclick.load_topics("tail", "val")
    test_topics = tripclick.load_topics("tail", "test")
    val_qrels = tripclick.load_qrels("tail", "val", label_type="raw")
    test_qrels = tripclick.load_qrels("tail", "test", label_type="raw")
    val_qids = sorted(q for q in val_topics if q in val_qrels)
    test_qids = sorted(q for q in test_topics if q in test_qrels)
    print(f"full TAIL population: val={len(val_qids)}, test={len(test_qids)}", flush=True)

    results = {"val": [], "test": []}
    start = {"val": 0, "test": 0}
    if resume:
        try:
            saved = json.loads(open(CHECKPOINT_PATH).read())
            results = saved["results"]
            start = saved["progress"]
            print(f"resuming: val={start['val']}/{len(val_qids)}, test={start['test']}/{len(test_qids)}", flush=True)
        except FileNotFoundError:
            print("--resume requested but no checkpoint found -- starting fresh.", flush=True)

    print("loading dense index (full corpus)...", flush=True)
    t_load = time.time()
    index = dense_index.DenseIndex(EMBED_PATH, IDS_PATH)
    print(f"index loaded in {time.time() - t_load:.0f}s, {len(index.doc_ids)} docs", flush=True)

    for split, qids, topics, qrels in [
        ("val", val_qids, val_topics, val_qrels),
        ("test", test_qids, test_topics, test_qrels),
    ]:
        t0 = time.time()
        for i in range(start[split], len(qids)):
            qid = qids[i]
            rec = pilot.run_episode(qid, topics[qid], qrels[qid], index)
            results[split].append(rec)
            if (i + 1) % _CHECKPOINT_EVERY == 0:
                start[split] = i + 1
                _save_checkpoint(results, start)
                print(f"[{split}] {i + 1}/{len(qids)} ({time.time() - t0:.0f}s this run)", flush=True)
        start[split] = len(qids)
        _save_checkpoint(results, start)
        print(f"[{split}] done: {len(results[split])} episodes", flush=True)

    with open("results/dense_retrieval_full_scale_raw.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda o: float(o) if hasattr(o, "item") else str(o))
    print("saved results/dense_retrieval_full_scale_raw.json", flush=True)

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
    print("\nFor comparison, BM25 TripClick TAIL (main result): 0.649 vs 0.552")

    # Paired bootstrap of the AUC difference, mirroring scripts/28's methodology.
    rng = np.random.RandomState(SEED)
    n_acted = len(ya_test)
    diffs = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.randint(0, n_acted, n_acted)
        yy, pp = ya_test[idx], au_test_proba[idx]
        if 0 < yy.sum() < n_acted:
            diffs.append(det_auc - roc_auc_score(yy, pp))
    diff_lo, diff_hi = np.percentile(diffs, [2.5, 97.5])
    print(f"Paired difference 95% CI: [{diff_lo:+.4f},{diff_hi:+.4f}], "
          f"excludes zero: {diff_lo > 0 or diff_hi < 0}")

    out = {
        "n_val": len(results["val"]), "n_test": len(results["test"]),
        "n_acted_val": len(val_acted), "n_acted_test": len(test_acted),
        "detection_auc": {"point": float(det_auc), "ci_lo": float(det_lo), "ci_hi": float(det_hi)},
        "action_utility_auc": {"point": float(au_auc), "ci_lo": float(au_lo), "ci_hi": float(au_hi)},
        "paired_difference": {"ci_lo": float(diff_lo), "ci_hi": float(diff_hi)},
        "corpus_size": len(index.doc_ids),
    }
    with open("results/dense_retrieval_full_scale_result.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved results/dense_retrieval_full_scale_result.json")


if __name__ == "__main__":
    main()
