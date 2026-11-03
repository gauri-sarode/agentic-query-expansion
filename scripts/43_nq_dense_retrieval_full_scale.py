#!/usr/bin/env python3
"""NQ analogue of scripts/36: the dense-retrieval check run on BEIR NQ,
under the scope, pipeline, statistics and run rules pre-committed in
configs/nq_dense_precommit.json (commit 0b343ca). Same telemetry
(scripts/33), same episode logic (reuses scripts/35's run_episode/
select_action directly), against the FULL 2,681,468-doc NQ dense index
(scripts/41).

Populations, both checked against the precommit's sha256 before any
episode runs:
  val  -- configs/nq_split.json val (n=800): every model fit
  test -- scripts/25's 2,052-query extension pool: single touch
The n=600 pilot (nq_split.json test) is never run here.

Differences from scripts/36, all fixed by the precommit:
  - AUC-difference CI is scripts/28's true paired bootstrap (both AUCs
    on the acted-upon test subset, both recomputed per resample), not
    scripts/36's fixed-detection variant -- see scripts/44.
  - Also reports the dense pre-action failure rate and mean nDCG@10 vs.
    frozen BM25 on the same 2,052 queries (the retriever-strength
    confound the TripClick paragraph checks).
  - Progress logging is episode counts only; no test outcome is printed
    until every test episode has run.
  - Test episodes refuse to start unless val is complete and the dense
    index is complete (all 2,681,468 docs).

Checkpointed every 50 episodes per split (--resume to continue).
~7.5h for 2,852 episodes at TripClick's observed ~9.5s/episode.

--smoke N: runs the first N val queries against the first 100k docs of
whatever index exists so far (works mid-build), writes only to
results/nq_dense_smoke.json, and never loads the test population.
Code-path check only -- its numbers mean nothing.

Usage: PYTHONPATH=. python3 scripts/43_nq_dense_retrieval_full_scale.py [--resume]
       PYTHONPATH=. python3 scripts/43_nq_dense_retrieval_full_scale.py --smoke 5
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from importlib import import_module
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.data import load_qrels, load_queries
from src.retrieval import dense_index

pilot = import_module("scripts.35_dense_retrieval_pilot")
extension = import_module("scripts.25_nq_test_extension")

PRECOMMIT = json.loads(open("configs/nq_dense_precommit.json").read())
SPLIT = json.loads(open("configs/nq_split.json").read())
META_DB = "data/nq_dense_meta.sqlite3"
EMBED_PATH = "data/nq_dense_embeddings.f16"
IDS_PATH = "data/nq_dense_ids.txt"
CORPUS_SIZE = 2_681_468
CHECKPOINT_PATH = "results/nq_dense_retrieval_checkpoint.json"
RAW_PATH = "results/nq_dense_retrieval_raw.json"
RESULT_PATH = "results/nq_dense_retrieval_result.json"
BM25_TEST_PATH = "results/nq_test_extension.json"
SMOKE_PATH = "results/nq_dense_smoke.json"
SMOKE_IDS_PATH = "results/nq_dense_smoke_ids.txt"
SMOKE_DOCS = 100_000
SEED = 42
N_BOOTSTRAP = 10000
_CHECKPOINT_EVERY = 50

# scripts/35's run_episode() reads META_DB as a module-level constant;
# point it at the NQ meta db for this run.
pilot.META_DB = META_DB


def _sha256(qids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(qids)).encode()).hexdigest()


def _check_population(name: str, qids: list[str]) -> None:
    want = PRECOMMIT["populations"][name]
    got = (len(qids), _sha256(qids))
    assert got == (want["n"], want["sha256_sorted_qids"]), \
        f"{name} population does not match configs/nq_dense_precommit.json: {got}"


def _save_checkpoint(results: dict, progress: dict) -> None:
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump({"results": results, "progress": progress}, f,
                   default=lambda o: float(o) if hasattr(o, "item") else str(o))


def _boot_ci(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    rng = np.random.RandomState(SEED)
    n = len(y)
    vals = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.randint(0, n, n)
        yy, pp = y[idx], p[idx]
        if 0 < yy.sum() < n:
            vals.append(roc_auc_score(yy, pp))
    return np.percentile(vals, [2.5, 97.5])


def smoke(n: int) -> None:
    val_qids = list(SPLIT["val"])
    _check_population("fit", val_qids)
    queries = load_queries("nq")
    qrels = load_qrels("nq")

    with open(IDS_PATH) as f:
        head = [line for _, line in zip(range(SMOKE_DOCS), f)]
    assert len(head) == SMOKE_DOCS, f"index has fewer than {SMOKE_DOCS} docs so far"
    Path(SMOKE_IDS_PATH).write_text("".join(head))
    index = dense_index.DenseIndex(EMBED_PATH, SMOKE_IDS_PATH)
    print(f"[smoke] {n} val queries vs. first {len(index.doc_ids)} docs (code-path check only)", flush=True)

    records = []
    for qid in val_qids[:n]:
        t = time.time()
        rec = pilot.run_episode(qid, queries[qid], qrels[qid], index)
        records.append(rec)
        print(f"[smoke] {qid}: action={rec['action']} acted={'au_features' in rec} "
              f"llm_error={'llm_error' in rec} n_t0={len(rec['t0_features'])} "
              f"n_au={len(rec.get('au_features', []))} ({time.time() - t:.1f}s)", flush=True)
    with open(SMOKE_PATH, "w") as f:
        json.dump(records, f, indent=2, default=lambda o: float(o) if hasattr(o, "item") else str(o))
    print(f"[smoke] saved {SMOKE_PATH}", flush=True)


def main() -> None:
    if "--smoke" in sys.argv:
        smoke(int(sys.argv[sys.argv.index("--smoke") + 1]))
        return
    resume = "--resume" in sys.argv

    val_qids = list(SPLIT["val"])
    test_qids = extension.extension_qids()
    _check_population("fit", val_qids)
    _check_population("frozen_test", test_qids)
    assert not set(test_qids) & set(SPLIT["test"]), "n=600 pilot must never run here"
    queries = load_queries("nq")
    qrels = load_qrels("nq")
    print(f"NQ dense: val={len(val_qids)}, test={len(test_qids)}", flush=True)

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

    with open(IDS_PATH) as f:
        n_embedded = sum(1 for _ in f)
    assert n_embedded == CORPUS_SIZE, \
        f"dense index incomplete ({n_embedded}/{CORPUS_SIZE}) -- finish scripts/41 first"

    print("loading dense index (full corpus)...", flush=True)
    t_load = time.time()
    index = dense_index.DenseIndex(EMBED_PATH, IDS_PATH)
    print(f"index loaded in {time.time() - t_load:.0f}s, {len(index.doc_ids)} docs", flush=True)

    for split, qids in [("val", val_qids), ("test", test_qids)]:
        if split == "test":
            assert start["val"] == len(val_qids), "val must complete before any test episode"
        t0 = time.time()
        for i in range(start[split], len(qids)):
            qid = qids[i]
            rec = pilot.run_episode(qid, queries[qid], qrels[qid], index)
            results[split].append(rec)
            if (i + 1) % _CHECKPOINT_EVERY == 0:
                start[split] = i + 1
                _save_checkpoint(results, start)
                print(f"[{split}] {i + 1}/{len(qids)} ({time.time() - t0:.0f}s this run)", flush=True)
        start[split] = len(qids)
        _save_checkpoint(results, start)
        print(f"[{split}] done: {len(results[split])} episodes", flush=True)

    with open(RAW_PATH, "w") as f:
        json.dump(results, f, indent=2, default=lambda o: float(o) if hasattr(o, "item") else str(o))
    print(f"saved {RAW_PATH}", flush=True)
    analyze(results, test_qids, len(index.doc_ids))


def analyze(results: dict, test_qids: list[str], corpus_size: int) -> None:
    val, test = results["val"], results["test"]

    # --- Detection: val-fit, test-score-once (val CV is diagnostic only) ---
    X_val = np.array([r["t0_features"] for r in val])
    y_val = np.array([r["y_poor"] for r in val])
    X_test = np.array([r["t0_features"] for r in test])
    y_test = np.array([r["y_poor"] for r in test])

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_proba = cross_val_predict(LogisticRegression(max_iter=1000), X_val, y_val, cv=kf, method="predict_proba")[:, 1]
    cv_auc = roc_auc_score(y_val, cv_proba)
    det_clf = LogisticRegression(max_iter=1000).fit(X_val, y_val)
    test_proba = det_clf.predict_proba(X_test)[:, 1]
    det_auc = roc_auc_score(y_test, test_proba)
    det_lo, det_hi = _boot_ci(y_test, test_proba)
    print(f"\nDetection: val CV AUC={cv_auc:.4f} (validation-only diagnostic)")
    print(f"Detection: test AUC={det_auc:.4f} [{det_lo:.4f},{det_hi:.4f}] (n={len(y_test)})")

    # --- Action-utility: acted-upon subset only ---
    val_acted = [r for r in val if "au_features" in r]
    test_acted = [r for r in test if "au_features" in r]
    Xa_val = np.array([r["au_features"] for r in val_acted])
    ya_val = (np.array([r["delta_ndcg"] for r in val_acted]) < -1e-9).astype(int)
    au_clf = LogisticRegression(max_iter=1000).fit(Xa_val, ya_val)
    Xa_test = np.array([r["au_features"] for r in test_acted])
    harmful = (np.array([r["delta_ndcg"] for r in test_acted]) < -1e-9).astype(int)
    au_score = au_clf.predict_proba(Xa_test)[:, 1]
    au_auc = roc_auc_score(harmful, au_score)
    au_lo, au_hi = _boot_ci(harmful, au_score)
    n_llm_err = {s: sum("llm_error" in r for r in results[s]) for s in results}
    print(f"acted-upon: val={len(val_acted)}/{len(val)}, test={len(test_acted)}/{len(test)} "
          f"(llm errors: val={n_llm_err['val']}, test={n_llm_err['test']})")
    print(f"Action-utility: test AUC={au_auc:.4f} [{au_lo:.4f},{au_hi:.4f}] (n={len(harmful)})")

    # --- Paired difference, scripts/28 method (see scripts/44) ---
    y_poor_acted = np.array([r["y_poor"] for r in test_acted])
    det_score_acted = det_clf.predict_proba(np.array([r["t0_features"] for r in test_acted]))[:, 1]
    det_auc_acted = roc_auc_score(y_poor_acted, det_score_acted)
    point_diff = det_auc_acted - au_auc
    n = len(test_acted)
    rng = np.random.RandomState(SEED)
    diffs = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.randint(0, n, n)
        yp, pp, h, s = y_poor_acted[idx], det_score_acted[idx], harmful[idx], au_score[idx]
        if not (0 < yp.sum() < n and 0 < h.sum() < n):
            continue
        diffs.append(roc_auc_score(yp, pp) - roc_auc_score(h, s))
    diff_lo, diff_hi = np.percentile(diffs, [2.5, 97.5])
    reproduces = bool(diff_lo > 0)
    print(f"\nDetection AUC on acted-upon subset (n={n}): {det_auc_acted:.4f}")
    print(f"Paired difference: {point_diff:+.4f}  95% CI=[{diff_lo:+.4f},{diff_hi:+.4f}]  (n_resamples={len(diffs)})")
    print(f"Reproduces the asymmetry (precommit rule: CI lower bound > 0): {reproduces}")

    # --- Retriever-strength confound: dense vs. frozen BM25, same 2,052 queries ---
    bm25 = {r["qid"]: r["original_ndcg"] for r in json.loads(open(BM25_TEST_PATH).read())["records"]}
    assert set(bm25) == set(test_qids)
    dense_ndcg = float(np.mean([r["orig_ndcg"] for r in test]))
    bm25_ndcg = float(np.mean([bm25[q] for q in test_qids]))
    fail_rate = float(y_test.mean())
    print(f"\nDense pre-action failure rate (test): {fail_rate:.3f}")
    print(f"Mean nDCG@10 on test: dense {dense_ndcg:.4f} vs. frozen BM25 {bm25_ndcg:.4f}")

    out = {
        "precommit": "configs/nq_dense_precommit.json",
        "n_val": len(val), "n_test": len(test),
        "n_acted_val": len(val_acted), "n_acted_test": len(test_acted),
        "n_llm_error": n_llm_err,
        "detection_val_cv_auc": float(cv_auc),
        "detection_auc": {"point": float(det_auc), "ci_lo": float(det_lo), "ci_hi": float(det_hi)},
        "action_utility_auc": {"point": float(au_auc), "ci_lo": float(au_lo), "ci_hi": float(au_hi)},
        "detection_auc_on_acted_subset": float(det_auc_acted),
        "paired_difference": {"point": float(point_diff), "ci_lo": float(diff_lo), "ci_hi": float(diff_hi),
                              "n_resamples": len(diffs)},
        "reproduces_asymmetry": reproduces,
        "dense_failure_rate_test": fail_rate,
        "mean_ndcg10_test": {"dense": dense_ndcg, "bm25_frozen": bm25_ndcg},
        "corpus_size": corpus_size,
    }
    with open(RESULT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {RESULT_PATH}")


if __name__ == "__main__":
    main()
