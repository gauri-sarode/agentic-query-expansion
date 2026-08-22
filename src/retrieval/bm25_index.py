"""BM25 retrieval via SQLite FTS5: disk-backed, no JVM. Scales to
TripClick's ~1.52M documents without a JVM dependency -- Pyserini/Lucene
has previously crashed natively on this machine's JDK in a related
project, so this avoids re-hitting that wall (see docs/milestones.md).
"""
from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterable
from pathlib import Path

Hit = tuple[str, float]


def build_index(docs: Iterable[tuple[str, str]], db_path: str, batch_size: int = 50_000) -> None:
    """docs: iterable of (doc_id, text) pairs."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA synchronous=OFF")
    con.execute("PRAGMA journal_mode=MEMORY")
    con.execute("DROP TABLE IF EXISTS docs")
    con.execute("CREATE VIRTUAL TABLE docs USING fts5(doc_id UNINDEXED, text)")

    batch: list[tuple[str, str]] = []
    n = 0
    t0 = time.time()
    for doc_id, text in docs:
        batch.append((doc_id, text))
        if len(batch) >= batch_size:
            con.executemany("INSERT INTO docs(doc_id, text) VALUES (?, ?)", batch)
            con.commit()
            n += len(batch)
            batch = []
            elapsed = time.time() - t0
            print(f"[bm25_index] {n} docs indexed in {elapsed:.1f}s ({n / elapsed:.0f} docs/s)", flush=True)
    if batch:
        con.executemany("INSERT INTO docs(doc_id, text) VALUES (?, ?)", batch)
        con.commit()
        n += len(batch)
    con.close()
    print(f"[bm25_index] done: {n} docs in {time.time() - t0:.1f}s -> {db_path}")


def search(query: str, db_path: str, k: int = 100) -> list[Hit]:
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(
            "SELECT doc_id, bm25(docs) AS score FROM docs WHERE docs MATCH ? "
            "ORDER BY score LIMIT ?",
            (query, k),
        ).fetchall()
    finally:
        con.close()
    # sqlite's bm25() is lower-is-better; flip sign so higher = more relevant.
    return [(doc_id, -score) for doc_id, score in rows]
