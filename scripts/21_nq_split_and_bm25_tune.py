#!/usr/bin/env python3
"""Second-corpus replication (RQ1 only, scoped-down per review request):
Step 1 of the NQ protocol -- construct a fixed-seed disjoint val/test
query split (BEIR NQ ships as a single flat 3,452-query test set, no
pre-existing train/val/test partition) and tune/freeze the BM25 config
on val ONLY, mirroring configs/experimental_protocol.yaml's TripClick
methodology and freeze discipline.

Sweep order matches the TripClick precedent (see
configs/experimental_protocol.yaml retrieval_config_frozen note):
  1. stemming (porter vs none), title_weight=1 (unweighted)
  2. title_weight sweep at the winning stemming, since NQ docs -- like
     TripClick's -- have a distinct title field (BEIR Wikipedia
     passages).

Does NOT touch bm25_index.py's module-level TITLE_WEIGHT/BODY_WEIGHT or
search() -- those are frozen constants backing already-reported TripClick
results. This script queries the SQLite FTS5 tables directly with a
parameterized title weight instead, for both the stemmed and unstemmed
NQ indexes it builds.

Usage: PYTHONPATH=. python3 scripts/21_nq_split_and_bm25_tune.py
"""
from __future__ import annotations

import json
import random
import sqlite3

import ir_datasets

from src.data import load_qrels, load_queries
from src.eval.metrics import evaluate
from src.retrieval.bm25_index import build_index

N_VAL = 800
N_TEST = 600
SEED = 42
_DOC_SEP = "<eot>"
STEMMED_DB = "data/nq_bm25.sqlite3"
UNSTEMMED_DB = "data/nq_bm25_unstemmed.sqlite3"
SPLIT_PATH = "configs/nq_split.json"


def build_unstemmed() -> None:
    def iter_docs():
        ds = ir_datasets.load("beir/nq")
        for doc in ds.docs_iter():
            title = (doc.title or "").strip()
            body = (doc.text or "").strip()
            yield doc.doc_id, f"{title}{_DOC_SEP}{body}"

    # Same build_index() as scripts/20, but tokenize='unicode61' (no porter)
    # -- build_index always uses 'porter unicode61', so replicate its body
    # here rather than parameterize the frozen TripClick-serving function.
    import time
    from pathlib import Path

    Path(UNSTEMMED_DB).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(UNSTEMMED_DB)
    con.execute("PRAGMA synchronous=OFF")
    con.execute("PRAGMA journal_mode=MEMORY")
    con.execute("DROP TABLE IF EXISTS docs")
    con.execute("CREATE VIRTUAL TABLE docs USING fts5(doc_id UNINDEXED, title, body, tokenize='unicode61')")
    batch = []
    n = 0
    t0 = time.time()
    for doc_id, text in iter_docs():
        title, body = text.split(_DOC_SEP, 1)
        batch.append((doc_id, title.strip(), body.strip()))
        if len(batch) >= 50_000:
            con.executemany("INSERT INTO docs(doc_id, title, body) VALUES (?, ?, ?)", batch)
            con.commit()
            n += len(batch)
            batch = []
    if batch:
        con.executemany("INSERT INTO docs(doc_id, title, body) VALUES (?, ?, ?)", batch)
        con.commit()
        n += len(batch)
    con.close()
    print(f"[unstemmed] {n} docs in {time.time() - t0:.1f}s -> {UNSTEMMED_DB}")


def _fts5_escape(query: str) -> str:
    tokens = query.split()
    if not tokens:
        return ""
    return " OR ".join('"' + t.replace('"', '""') + '"' for t in tokens)


def search_weighted(query: str, db_path: str, title_weight: float, body_weight: float, k: int = 100):
    escaped = _fts5_escape(query)
    if not escaped:
        return []
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(
            "SELECT doc_id, bm25(docs, 0, ?, ?) AS score FROM docs WHERE docs MATCH ? ORDER BY score LIMIT ?",
            (title_weight, body_weight, escaped, k),
        ).fetchall()
    finally:
        con.close()
    return [(doc_id, -score) for doc_id, score in rows]


def eval_config(qids, queries, qrels, db_path, title_weight, body_weight=1.0):
    run = {}
    for qid in qids:
        hits = search_weighted(queries[qid], db_path, title_weight, body_weight, k=100)
        run[qid] = {d: s for d, s in hits}
    res = evaluate({q: qrels[q] for q in qids}, run)
    ndcg = sum(r["ndcg_cut_10"] for r in res.values()) / len(res)
    mrr = sum(r["recip_rank"] for r in res.values()) / len(res)
    recall = sum(r["recall_100"] for r in res.values()) / len(res)
    return {"ndcg10": ndcg, "mrr": mrr, "recall100": recall}


def main() -> None:
    queries = load_queries("nq")
    qrels = load_qrels("nq")
    qids = sorted(q for q in queries if q in qrels)
    print(f"n queries w/ qrels: {len(qids)}")

    rng = random.Random(SEED)
    shuffled = qids[:]
    rng.shuffle(shuffled)
    val_qids = shuffled[:N_VAL]
    test_qids = shuffled[N_VAL:N_VAL + N_TEST]
    assert set(val_qids).isdisjoint(test_qids)
    with open(SPLIT_PATH, "w") as f:
        json.dump({"seed": SEED, "n_val": N_VAL, "n_test": N_TEST, "val": val_qids, "test": test_qids}, f)
    print(f"split saved to {SPLIT_PATH}: val={len(val_qids)} test={len(test_qids)}")

    print("\nbuilding unstemmed comparison index...")
    build_unstemmed()

    print("\n=== step 1: stemming (val, title_weight=1) ===")
    porter_r = eval_config(val_qids, queries, qrels, STEMMED_DB, title_weight=1.0)
    none_r = eval_config(val_qids, queries, qrels, UNSTEMMED_DB, title_weight=1.0)
    print("porter:", porter_r)
    print("none:  ", none_r)
    best_stemming = "porter" if porter_r["ndcg10"] >= none_r["ndcg10"] else "none"
    best_db = STEMMED_DB if best_stemming == "porter" else UNSTEMMED_DB
    print(f"-> selected stemming: {best_stemming}")

    print(f"\n=== step 2: title_weight sweep (val, stemming={best_stemming}) ===")
    sweep_results = {}
    for tw in [1.0, 5.0, 10.0, 20.0]:
        r = eval_config(val_qids, queries, qrels, best_db, title_weight=tw)
        sweep_results[tw] = r
        print(f"title_weight={tw:5.1f}  nDCG@10={r['ndcg10']:.4f}  MRR={r['mrr']:.4f}  Recall@100={r['recall100']:.4f}")
    best_tw = max(sweep_results, key=lambda tw: sweep_results[tw]["ndcg10"])
    print(f"-> selected title_weight: {best_tw}")

    frozen = {
        "seed": SEED,
        "split_path": SPLIT_PATH,
        "n_val": len(val_qids),
        "n_test": len(test_qids),
        "stemming": best_stemming,
        "db_path": best_db,
        "title_weight": best_tw,
        "body_weight": 1.0,
        "val_selection": {
            "stemming_sweep": {"porter": porter_r, "none": none_r},
            "title_weight_sweep": {str(k): v for k, v in sweep_results.items()},
        },
    }
    with open("configs/nq_protocol_frozen.json", "w") as f:
        json.dump(frozen, f, indent=2)
    print("\nfrozen config saved to configs/nq_protocol_frozen.json")


if __name__ == "__main__":
    main()
