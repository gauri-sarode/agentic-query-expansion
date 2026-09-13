#!/usr/bin/env python3
"""Build the BM25 index over BEIR Natural Questions (~2.68M docs), for
the scoped-down second-corpus replication of the detection-vs-actionability
finding (RQ1). Mirrors scripts/03_build_tripclick_index.sh's approach:
SQLite FTS5, no JVM (see src/retrieval/bm25_index.py, docs/milestones.md).

Docs are joined as "title<eot>body" -- the same separator TripClick's
loader uses -- so build_index's title/body split fires and the
TITLE_WEIGHT=10 config (tuned on TripClick TAIL-val) applies here too.
Going through src.data.load_docs would silently skip this: its generic
ir_datasets branch joins title+text with a plain space, never the
"<eot>" separator build_index actually looks for, so NFCorpus's existing
index has always indexed title text as unweighted body. Not touching
that here -- fixing it retroactively would change an already-used
artifact for no benefit to this replication -- but NQ's index is built
directly in this script instead of through load_docs, specifically to
avoid inheriting the same gap.

Usage: PYTHONPATH=. python3 scripts/20_build_nq_index.py
"""
from __future__ import annotations

import ir_datasets

from src.retrieval.bm25_index import build_index

DB_PATH = "data/nq_bm25.sqlite3"
_DOC_SEP = "<eot>"


def iter_docs():
    ds = ir_datasets.load("beir/nq")
    for doc in ds.docs_iter():
        title = (doc.title or "").strip()
        body = (doc.text or "").strip()
        yield doc.doc_id, f"{title}{_DOC_SEP}{body}"


def main() -> None:
    build_index(iter_docs(), DB_PATH)


if __name__ == "__main__":
    main()
