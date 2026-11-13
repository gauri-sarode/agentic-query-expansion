#!/usr/bin/env python3
"""Split and BM25 index for the independent information-value test, as
pre-committed in configs/info_value_precommit.json. Reads query ids, query
texts and documents only -- no relevance judgment is opened here.

dbpedia:  val = all 67 dev queries (PRF selection only); test = all 400 test
          queries, with 5 cross-fitting folds by sha256(query_id + ':42') mod 5.
hotpotqa: train = first 1,500 train queries by sha256(query_id + ':42');
          val = first 500 dev queries by the same order; test = all test queries.
Also reports exact-duplicate query texts across splits (label-free audit).

Outputs: configs/info_value_split.json (committed; one entry per corpus),
data/{corpus}_bm25.sqlite3 (local).

Usage: PYTHONPATH=. python3 scripts/74_iv_split_and_index.py {dbpedia|hotpotqa}
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

from src.data import load_docs, load_queries
from src.retrieval.bm25_index import build_index

SPLIT_PATH = Path("configs/info_value_split.json")


def h(x: str) -> str:
    return hashlib.sha256(f"{x}:42".encode()).hexdigest()


def norm(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def main() -> None:
    corpus = sys.argv[1]
    if corpus == "dbpedia":
        q = {"dev": load_queries("dbpedia-dev"), "test": load_queries("dbpedia-test")}
        parts = {"val": sorted(q["dev"]), "test": sorted(q["test"])}
        folds = {i: int(h(i), 16) % 5 for i in parts["test"]}
        extra = {"test_fold": folds, "fold_sizes": dict(Counter(folds.values()))}
    else:
        q = {sp: load_queries(f"hotpotqa-{sp}") for sp in ("train", "dev", "test")}
        parts = {"train": sorted(sorted(q["train"], key=h)[:1500]), "val": sorted(sorted(q["dev"], key=h)[:500]),
                 "test": sorted(q["test"])}
        extra = {}
    texts = {k: v for d in q.values() for k, v in d.items()}
    seen: dict[str, set[str]] = {}
    for sp, ids in parts.items():
        for i in ids:
            seen.setdefault(norm(texts[i]), set()).add(sp)
    dup = Counter("+".join(sorted(s)) for s in seen.values() if len(s) > 1)
    allsplits = json.loads(SPLIT_PATH.read_text()) if SPLIT_PATH.exists() else {}
    allsplits[corpus] = {"rule": "configs/info_value_precommit.json splits", "n": {k: len(v) for k, v in parts.items()},
                         "duplicate_texts_across_splits": dict(dup), **parts, **extra}
    SPLIT_PATH.write_text(json.dumps(allsplits, indent=1))
    print(corpus, {k: len(v) for k, v in parts.items()}, "dups", dict(dup), extra.get("fold_sizes", ""))
    db = f"data/{corpus}_bm25.sqlite3"
    if "--split-only" not in sys.argv and not Path(db).exists():
        build_index(load_docs(corpus), db)
        print("index built", db)


if __name__ == "__main__":
    main()
