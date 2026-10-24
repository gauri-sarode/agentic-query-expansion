#!/usr/bin/env python3
"""Builds the full BEIR NQ dense corpus (2,681,468 docs) for a
second-corpus dense-retrieval replication, mirroring the TripClick
dense-retrieval check (scripts/34-36) but at the SAME full-corpus scale
used there rather than the pilot's qrel-scoped subset -- this is the
"same scale as the BM25 result" standard the TripClick dense check
already set, applied to NQ.

Two-stage, same as src/retrieval/dense_index.py's own split:
  1. build_meta_db: doc_id -> "title<eot>body" text, same join scripts/20
     used for NQ's BM25 index (so this corpus is identical in content to
     the one data/nq_bm25.sqlite3 already indexes, just a different
     storage/retrieval mechanism).
  2. build_embeddings: BAAI/bge-small-en-v1.5, float16 memmap, resumable.
     TripClick's full embed benchmarked at ~75 docs/s (5.7h for 1.52M
     docs, scripts/34's docstring); NQ's 2.68M docs is proportionally
     ~10h on this hardware. Both stages are checkpointed/resumable by
     dense_index.py itself -- safe to interrupt and rerun this script.

Usage: PYTHONPATH=. python3 scripts/41_nq_dense_build_full_corpus.py
"""
from __future__ import annotations

import ir_datasets

from src.retrieval import dense_index

META_DB = "data/nq_dense_meta.sqlite3"
EMBED_PATH = "data/nq_dense_embeddings.f16"
IDS_PATH = "data/nq_dense_ids.txt"
_DOC_SEP = "<eot>"


def iter_docs():
    ds = ir_datasets.load("beir/nq")
    for doc in ds.docs_iter():
        title = (doc.title or "").strip()
        body = (doc.text or "").strip()
        yield doc.doc_id, f"{title}{_DOC_SEP}{body}"


def main() -> None:
    import sqlite3
    from pathlib import Path

    if not Path(META_DB).exists():
        print("building meta db...", flush=True)
        dense_index.build_meta_db(iter_docs(), META_DB)
    else:
        con = sqlite3.connect(META_DB)
        (n,) = con.execute("SELECT COUNT(*) FROM meta").fetchone()
        con.close()
        print(f"meta db already exists ({n} docs), skipping build", flush=True)

    print("building embeddings...", flush=True)
    dense_index.build_embeddings(META_DB, EMBED_PATH, IDS_PATH, resume=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
