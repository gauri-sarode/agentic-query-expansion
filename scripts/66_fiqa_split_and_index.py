#!/usr/bin/env python3
"""FiQA split and BM25 index, as pre-committed in configs/fiqa_precommit.json.
Reads query ids, query texts and documents only -- no relevance judgment is
opened here.

Split: official BEIR splits; train capped at the first 1,500 train queries by
sha256(query_id + ':42') (fitting only), all 500 dev queries as val, all 648
test queries as test. Also reports exact-duplicate query texts across splits
(label-free audit).

Outputs: configs/fiqa_split.json (committed), data/fiqa_bm25.sqlite3 (local).

Usage: PYTHONPATH=. python3 scripts/66_fiqa_split_and_index.py
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from src.data import load_docs, load_queries
from src.retrieval.bm25_index import build_index

DB_PATH = "data/fiqa_bm25.sqlite3"
SPLIT_PATH = Path("configs/fiqa_split.json")
N_TRAIN = 1500


def norm(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def main() -> None:
    q = {sp: load_queries(f"fiqa-{sp}") for sp in ("train", "dev", "test")}
    train = sorted(q["train"], key=lambda x: hashlib.sha256(f"{x}:42".encode()).hexdigest())[:N_TRAIN]
    parts = {"train": sorted(train), "val": sorted(q["dev"]), "test": sorted(q["test"])}
    texts = {**q["train"], **q["dev"], **q["test"]}
    owner = {i: sp for sp, ids in parts.items() for i in ids}
    by_text = Counter()
    seen: dict[str, set[str]] = {}
    for i, sp in owner.items():
        seen.setdefault(norm(texts[i]), set()).add(sp)
    for sps in seen.values():
        if len(sps) > 1:
            by_text["+".join(sorted(sps))] += 1
    SPLIT_PATH.write_text(json.dumps({
        "dataset": "beir/fiqa", "rule": "configs/fiqa_precommit.json data.split",
        "n": {k: len(v) for k, v in parts.items()}, "duplicate_texts_across_splits": dict(by_text), **parts,
    }, indent=1))
    print({k: len(v) for k, v in parts.items()}, "cross-split duplicate texts", dict(by_text))
    if not Path(DB_PATH).exists():
        build_index(load_docs("fiqa"), DB_PATH)
        print("index built")


if __name__ == "__main__":
    main()
