#!/usr/bin/env python3
"""ArguAna split and BM25 index, as pre-committed in
configs/arguana_precommit.json. Reads query ids and documents only --
no relevance judgment is opened here.

Split: grouped by debate (query id minus its final -pro../-con.. segment),
groups ordered by sha256(group + ':42') and filled greedily into
train/val/test up to 50/20/30% of the queries.

Outputs: configs/arguana_split.json (committed), data/arguana_bm25.sqlite3
(local). The BEIR loader puts title and text into the body column
(src/data.py load_docs), so the title weight plays no role.

Usage: PYTHONPATH=. python3 scripts/51_arguana_split_and_index.py
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from src.data import load_docs, load_queries
from src.retrieval.bm25_index import build_index

DB_PATH = "data/arguana_bm25.sqlite3"
SPLIT_PATH = Path("configs/arguana_split.json")
GROUP = re.compile(r"^(.*)-(pro|con)\d+[a-z]$")
SHARES = (("train", 0.5), ("val", 0.2), ("test", 0.3))


def split(qids: list[str]) -> tuple[dict[str, list[str]], int]:
    groups: dict[str, list[str]] = defaultdict(list)
    unmatched = 0
    for q in qids:
        m = GROUP.match(q)
        unmatched += m is None
        groups[m.group(1) if m else q].append(q)
    order = sorted(groups, key=lambda g: hashlib.sha256(f"{g}:42".encode()).hexdigest())
    n = len(qids)
    targets = [(name, round(share * n)) for name, share in SHARES]
    out: dict[str, list[str]] = {name: [] for name, _ in SHARES}
    i = 0
    for name, target in targets[:-1]:
        while i < len(order) and len(out[name]) + len(groups[order[i]]) <= target:
            out[name] += groups[order[i]]
            i += 1
    for g in order[i:]:
        out["test"] += groups[g]
    return {k: sorted(v) for k, v in out.items()}, unmatched


def main() -> None:
    qids = sorted(load_queries("arguana"))
    parts, unmatched = split(qids)
    assert sum(len(v) for v in parts.values()) == len(qids)
    SPLIT_PATH.write_text(json.dumps({
        "dataset": "beir/arguana", "rule": "configs/arguana_precommit.json data.split",
        "n": {k: len(v) for k, v in parts.items()}, "unmatched_group_ids": unmatched, **parts,
    }, indent=1))
    print({k: len(v) for k, v in parts.items()}, "unmatched", unmatched)
    if not Path(DB_PATH).exists():
        build_index(load_docs("arguana"), DB_PATH)


if __name__ == "__main__":
    main()
