"""Direct TSV parsing for the local TripClick IR Benchmark package
(data/tripclick_raw/benchmark_tsv/), rather than going through
ir_datasets' tripclick catalog -- the files are already in the exact
format the official package ships (doc_id\ttitle <eot> body for
documents; standard 4-column TREC qrels; qid\ttext topics), so parsing
them directly avoids a local ir_datasets registration step.

TripClick may not be redistributed (non-commercial research license) --
data/ is gitignored; only code and aggregated results are committed.
"""
from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path

RAW_DIR = Path("data/tripclick_raw/benchmark_tsv")

Slice = str  # "head" | "torso" | "tail"
Split = str  # "train" | "val" | "test"


def iter_docs(raw_dir: Path = RAW_DIR) -> Iterator[tuple[str, str]]:
    """Yields (doc_id, text). The title/body separator '<eot>' is kept as
    plain text -- fine for lexical BM25 indexing, which doesn't need the
    structure.
    """
    path = raw_dir / "documents" / "docs.tsv"
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            if len(row) < 2:
                continue
            doc_id, text = row[0], row[1]
            yield doc_id, text


def load_topics(slice_: Slice, split: Split, raw_dir: Path = RAW_DIR) -> dict[str, str]:
    path = raw_dir / "topics" / f"topics.{slice_}.{split}.tsv"
    queries: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            if len(row) < 2:
                continue
            queries[row[0]] = row[1]
    return queries


def load_qrels(
    slice_: Slice, split: Split, label_type: str = "raw", raw_dir: Path = RAW_DIR
) -> dict[str, dict[str, int]]:
    """label_type: 'raw' (binary click labels, all slices/splits) or
    'dctr' (graded click-model labels, head only in this package).
    """
    path = raw_dir / "qrels" / f"qrels.{label_type}.{slice_}.{split}.tsv"
    qrels: dict[str, dict[str, int]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            if len(row) < 4:
                continue
            qid, _, doc_id, rel = row[0], row[1], row[2], row[3]
            qrels.setdefault(qid, {})[doc_id] = int(rel)
    return qrels


def available_splits(raw_dir: Path = RAW_DIR) -> dict[str, list[str]]:
    """Introspects what's actually on disk, e.g. {'topics': [...], 'qrels': [...]}."""
    out: dict[str, list[str]] = {}
    for kind in ("topics", "qrels"):
        d = raw_dir / kind
        out[kind] = sorted(p.name for p in d.glob("*.tsv")) if d.exists() else []
    return out
