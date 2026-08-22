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


def _fts5_escape(query: str) -> str:
    """FTS5 MATCH has its own query syntax (column filters via ':', NOT via
    '-', phrase/prefix operators, etc.) -- raw user/query text must be
    escaped or it can be misparsed as a query operator rather than search
    terms (e.g. a query containing "area:" raises "no such column: area").
    Quoting each token as a literal string sidesteps that.

    FTS5's default combining operator between space-separated tokens is
    AND, not OR -- that would require every query term to co-occur in a
    document, which is boolean search, not the ranked best-match
    retrieval BM25 is supposed to provide (and returns zero hits for any
    multi-term query where no single document contains every term).
    Joining with explicit OR restores normal ranked-retrieval behavior;
    bm25() still scores AND-satisfying documents higher via term
    coverage, it just no longer excludes partial matches entirely.
    """
    tokens = query.split()
    if not tokens:
        return ""
    return " OR ".join('"' + t.replace('"', '""') + '"' for t in tokens)


def search(query: str, db_path: str, k: int = 100) -> list[Hit]:
    escaped = _fts5_escape(query)
    if not escaped:
        return []
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(
            "SELECT doc_id, bm25(docs) AS score FROM docs WHERE docs MATCH ? "
            "ORDER BY score LIMIT ?",
            (escaped, k),
        ).fetchall()
    finally:
        con.close()
    # sqlite's bm25() is lower-is-better; flip sign so higher = more relevant.
    return [(doc_id, -score) for doc_id, score in rows]


def get_texts(doc_ids: Iterable[str], db_path: str) -> dict[str, str]:
    doc_ids = list(doc_ids)
    if not doc_ids:
        return {}
    con = sqlite3.connect(db_path)
    try:
        placeholders = ",".join("?" for _ in doc_ids)
        rows = con.execute(
            f"SELECT doc_id, text FROM docs WHERE doc_id IN ({placeholders})", doc_ids
        ).fetchall()
    finally:
        con.close()
    return dict(rows)


def doc_count(db_path: str) -> int:
    con = sqlite3.connect(db_path)
    try:
        (n,) = con.execute("SELECT count(*) FROM docs").fetchone()
    finally:
        con.close()
    return n


def doc_frequency(term: str, db_path: str) -> int:
    """Number of documents containing `term` (single token, no MATCH operators)."""
    escaped = _fts5_escape(term)
    if not escaped:
        return 0
    con = sqlite3.connect(db_path)
    try:
        (n,) = con.execute(
            "SELECT count(*) FROM docs WHERE docs MATCH ?", (escaped,)
        ).fetchone()
    finally:
        con.close()
    return n


def idf(term: str, db_path: str, total_docs: int | None = None) -> float:
    """Inductive-free IDF over the indexed corpus: log((N+1)/(df+1)) + 1,
    a smoothed variant that stays finite and positive for df in [0, N].
    """
    import math

    n = total_docs if total_docs is not None else doc_count(db_path)
    df = doc_frequency(term, db_path)
    return math.log((n + 1) / (df + 1)) + 1.0
