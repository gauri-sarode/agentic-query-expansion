"""BM25 retrieval via SQLite FTS5: disk-backed, no JVM. Scales to
TripClick's ~1.52M documents without a JVM dependency -- Pyserini/Lucene
has previously crashed natively on this machine's JDK in a related
project, so this avoids re-hitting that wall (see docs/milestones.md).

FTS5 virtual tables only index the column(s) used in MATCH -- `doc_id`
here is UNINDEXED, so any WHERE/COUNT(*) touching it degrades to a full
table scan. That's invisible at NFCorpus scale (~3.6K docs) but brutal at
TripClick scale (~1.52M docs): get_texts() for 10 doc_ids and doc_count()
each took ~13s in profiling, entirely from that scan -- one full episode's
telemetry computation calls both several times. A companion regular
table (`meta`, doc_id PRIMARY KEY) plus a precomputed doc-count row give
both a real index instead, while `docs` (FTS5) stays dedicated to what it
does well: MATCH-based search and term document-frequency lookups.

Tokenizer and title/body field weighting were selected entirely on
TripClick TAIL-val (configs/experimental_protocol.yaml), never on the
test split: 'porter unicode61' (stemming) took val nDCG@10 from 0.2353
(unstemmed) to 0.2695; splitting title/body into separately-weighted
columns and sweeping the title:body ratio on val found a plateau at
title_weight=10 (val nDCG@10 0.2695 -> 0.3266, MRR peaks at 10 then
declines at 20) -- see git history around 2026-08-29 for the sweep.
TripClick documents ship as "title <eot> body" in a single field; we
split on that delimiter at index time.
"""
from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterable
from pathlib import Path

Hit = tuple[str, float]

# Selected on TAIL-val only (see module docstring). bm25()'s weight args
# map to ALL declared columns in table order including UNINDEXED ones --
# doc_id consumes the first slot even though it carries no searchable
# text, so the call is bm25(docs, 0, TITLE_WEIGHT, BODY_WEIGHT).
TITLE_WEIGHT = 10.0
BODY_WEIGHT = 1.0
_DOC_SEP = "<eot>"


def build_index(docs: Iterable[tuple[str, str]], db_path: str, batch_size: int = 50_000) -> None:
    """docs: iterable of (doc_id, text) pairs, text = "title <eot> body"."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA synchronous=OFF")
    con.execute("PRAGMA journal_mode=MEMORY")
    con.execute("DROP TABLE IF EXISTS docs")
    con.execute(
        "CREATE VIRTUAL TABLE docs USING fts5(doc_id UNINDEXED, title, body, tokenize='porter unicode61')"
    )
    con.execute("DROP TABLE IF EXISTS meta")
    con.execute("CREATE TABLE meta (doc_id TEXT PRIMARY KEY, text TEXT)")
    con.execute("DROP TABLE IF EXISTS stats")
    con.execute("CREATE TABLE stats (key TEXT PRIMARY KEY, value INTEGER)")

    doc_batch: list[tuple[str, str, str]] = []
    meta_batch: list[tuple[str, str]] = []
    n = 0
    t0 = time.time()
    for doc_id, text in docs:
        if _DOC_SEP in text:
            title, body = text.split(_DOC_SEP, 1)
            title, body = title.strip(), body.strip()
        else:
            title, body = "", text
        doc_batch.append((doc_id, title, body))
        meta_batch.append((doc_id, text))
        if len(doc_batch) >= batch_size:
            con.executemany("INSERT INTO docs(doc_id, title, body) VALUES (?, ?, ?)", doc_batch)
            con.executemany("INSERT INTO meta(doc_id, text) VALUES (?, ?)", meta_batch)
            con.commit()
            n += len(doc_batch)
            doc_batch, meta_batch = [], []
            elapsed = time.time() - t0
            print(f"[bm25_index] {n} docs indexed in {elapsed:.1f}s ({n / elapsed:.0f} docs/s)", flush=True)
    if doc_batch:
        con.executemany("INSERT INTO docs(doc_id, title, body) VALUES (?, ?, ?)", doc_batch)
        con.executemany("INSERT INTO meta(doc_id, text) VALUES (?, ?)", meta_batch)
        con.commit()
        n += len(doc_batch)
    con.execute("INSERT OR REPLACE INTO stats(key, value) VALUES ('doc_count', ?)", (n,))
    con.commit()
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
            "SELECT doc_id, bm25(docs, 0, ?, ?) AS score FROM docs WHERE docs MATCH ? "
            "ORDER BY score LIMIT ?",
            (TITLE_WEIGHT, BODY_WEIGHT, escaped, k),
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
            f"SELECT doc_id, text FROM meta WHERE doc_id IN ({placeholders})", doc_ids
        ).fetchall()
    finally:
        con.close()
    return dict(rows)


def doc_count(db_path: str) -> int:
    con = sqlite3.connect(db_path)
    try:
        row = con.execute("SELECT value FROM stats WHERE key = 'doc_count'").fetchone()
        if row is not None:
            return row[0]
        (n,) = con.execute("SELECT count(*) FROM meta").fetchone()
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
