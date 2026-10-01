"""Dense first-stage retrieval: a drop-in alternative to bm25_index with
a matching search()/get_texts() interface, built for a disk/compute-
constrained pilot (WSDM reviewer question: does the detection-vs-
actionability gap hold under dense retrieval, not just BM25?).

Design choices forced by this machine's constraints (M4 MacBook Air,
disk down to ~5GB free after the corpus TSV copy-in): embeddings are
stored float16 (not float32) to roughly halve the ~2.3GB a 1.52M x 384
matrix would otherwise need; no FAISS/ANN index -- at this corpus size
a single matrix-multiply brute-force cosine search fits comfortably in
16GB RAM and needs no extra dependency; no FTS5 virtual table (unlike
bm25_index) since this module never does lexical search, only a plain
doc_id->text lookup table, to avoid FTS5's posting-list storage
overhead entirely.

Model: BAAI/bge-small-en-v1.5 (384-dim, ~130MB), chosen over a general
sentence-similarity model because it's trained specifically for
asymmetric query/passage retrieval (with the query-side instruction
prefix BGE's own model card specifies), making it a more honest "dense
retriever" comparison than repurposing the reranker's cross-encoder
family for a bi-encoder role it wasn't trained for.
"""
from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np

Hit = tuple[str, float]

MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_DOC_SEP = "<eot>"


def _get_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


def build_meta_db(docs: Iterable[tuple[str, str]], db_path: str, batch_size: int = 50_000) -> None:
    """Plain doc_id->text lookup table, no FTS5 -- this module never does
    lexical search, so the FTS5 posting-list overhead bm25_index.py pays
    for MATCH-based search would be pure waste here."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA synchronous=OFF")
    con.execute("PRAGMA journal_mode=MEMORY")
    con.execute("DROP TABLE IF EXISTS meta")
    con.execute("CREATE TABLE meta (doc_id TEXT PRIMARY KEY, text TEXT)")
    batch: list[tuple[str, str]] = []
    n = 0
    t0 = time.time()
    for doc_id, text in docs:
        batch.append((doc_id, text))
        if len(batch) >= batch_size:
            con.executemany("INSERT INTO meta(doc_id, text) VALUES (?, ?)", batch)
            con.commit()
            n += len(batch)
            batch = []
            elapsed = time.time() - t0
            print(f"[dense_index] meta: {n} docs in {elapsed:.1f}s ({n / elapsed:.0f} docs/s)", flush=True)
    if batch:
        con.executemany("INSERT INTO meta(doc_id, text) VALUES (?, ?)", batch)
        con.commit()
        n += len(batch)
    con.close()
    print(f"[dense_index] meta done: {n} docs -> {db_path}")


def build_embeddings(
    db_path: str,
    embed_path: str,
    ids_path: str,
    batch_size: int = 512,
    resume: bool = True,
) -> None:
    """Embeds every doc in meta (db_path) in id order, writes a float16
    memmap (embed_path) + a parallel doc_id list (ids_path, one per
    line). Resumable: if embed_path/ids_path already have N complete
    rows, continues from row N rather than re-embedding from scratch --
    this is a multi-hour job on this hardware and the only realistic way
    to run it is as an interruptible background process.
    """
    model = _get_model()

    con = sqlite3.connect(db_path)
    try:
        (total,) = con.execute("SELECT COUNT(*) FROM meta").fetchone()
        rows = con.execute("SELECT doc_id, text FROM meta ORDER BY doc_id").fetchall()
    finally:
        con.close()
    print(f"[dense_index] {total} docs to embed", flush=True)

    start = 0
    if resume and Path(ids_path).exists():
        with open(ids_path) as f:
            start = sum(1 for _ in f)
        print(f"[dense_index] resuming from doc {start}/{total}", flush=True)

    mode = "r+" if (resume and Path(embed_path).exists()) else "w+"
    mat = np.memmap(embed_path, dtype=np.float16, mode=mode, shape=(total, EMBED_DIM))

    ids_f = open(ids_path, "a" if start else "w")
    t0 = time.time()
    for i in range(start, total, batch_size):
        batch = rows[i : i + batch_size]
        doc_ids = [d for d, _ in batch]
        texts = [t.replace(_DOC_SEP, " ") for _, t in batch]
        emb = model.encode(texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
        mat[i : i + len(batch)] = emb.astype(np.float16)
        for d in doc_ids:
            ids_f.write(d + "\n")
        ids_f.flush()
        mat.flush()
        n_done = i + len(batch)
        elapsed = time.time() - t0
        rate = (n_done - start) / elapsed if elapsed > 0 else 0
        eta_min = (total - n_done) / rate / 60 if rate > 0 else float("inf")
        print(f"[dense_index] embedded {n_done}/{total} ({rate:.1f} docs/s, ETA {eta_min:.0f}min)", flush=True)
    ids_f.close()
    print(f"[dense_index] done: {total} docs -> {embed_path}")


class DenseIndex:
    """Loaded once per process: holds the memmap + id list + model in
    memory for repeated search() calls (embedding the query fresh each
    call is cheap; reloading the 1.15GB corpus matrix is not)."""

    def __init__(self, embed_path: str, ids_path: str):
        with open(ids_path) as f:
            self.doc_ids = [line.strip() for line in f]
        n = len(self.doc_ids)
        self.mat = np.memmap(embed_path, dtype=np.float16, mode="r", shape=(n, EMBED_DIM))
        self.model = _get_model()

    def search(self, query: str, k: int = 100) -> list[Hit]:
        q_emb = self.model.encode(
            [_QUERY_PREFIX + query], normalize_embeddings=True, show_progress_bar=False
        )[0].astype(np.float32)
        scores = self.mat.astype(np.float32) @ q_emb
        top_idx = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [(self.doc_ids[i], float(scores[i])) for i in top_idx]


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
    return {doc_id: text.replace(_DOC_SEP, " ") for doc_id, text in rows}
