#!/usr/bin/env python3
"""Rankings for the reranking and PRF interventions, as pre-committed in
configs/intervention_scope_precommit.json. Reads no relevance labels:
it stores BM25, reranked and PRF rankings (ids + scores) per query, and
scripts/49 does all labelling and analysis.

Outputs (local; results/ is gitignored):
  results/interventions/{corpus}_{split}_bm25.json     top-100 BM25
  results/interventions/{corpus}_{split}_rerank.json   reranked top-100
  results/interventions/{corpus}_{split}_prf_d{D}_t{T}.json  expansion-term top-100 for fb_docs=D, fb_terms=T
PRF fusion with lam happens in scripts/49, since lam needs no new retrieval.

Usage: PYTHONPATH=. python3 scripts/48_intervention_runs.py CORPUS SPLIT STEP
  CORPUS in {tripclick, nq}; SPLIT in {val, test}; STEP in {bm25, rerank, prf_d{D}_t{T}}
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.config import CONFIG
from src.data import load_queries
from src.retrieval import bm25_index

OUT_DIR = Path("results/interventions")
K = 100
TOKEN = re.compile(r"[a-z]{3,}")


def setup(corpus: str, split: str) -> tuple[list[str], dict[str, str], str]:
    if corpus == "tripclick":
        fd = json.loads(open("results/tripclick-tail_failure_detection.json").read())
        qids = fd["val_qids"] if split == "val" else fd["test_qids"]
        queries = load_queries("tripclick-tail-val" if split == "val" else "tripclick-tail")
        return qids, queries, "data/tripclick_bm25.sqlite3"
    frozen = json.loads(open("configs/nq_protocol_frozen.json").read())
    bm25_index.TITLE_WEIGHT = frozen["title_weight"]
    bm25_index.BODY_WEIGHT = frozen["body_weight"]
    if split == "val":
        qids = json.loads(open("results/nq_val_calibration.json").read())["val_qids"]
    else:
        qids = [r["qid"] for r in json.loads(open("results/nq_test_extension.json").read())["records"]]
    return qids, load_queries("nq"), frozen["db_path"]


def rerank(query: str, bm25: list[list], db: str) -> list[list]:
    """The agent's composition: cross-encoder top-50 first, then the rest
    of the BM25 order (src/agent/steps.py)."""
    from src.retrieval.rerank import rerank as ce

    ids = [d for d, _ in bm25[: CONFIG.reranker_max_candidates]]
    texts = bm25_index.get_texts(ids, db)
    top = ce(query, [(d, texts[d]) for d in ids if d in texts], top_k=CONFIG.reranker_top_k)
    seen = {d for d, _ in top}
    return [[d, float(s)] for d, s in top] + [[d, None] for d, _ in bm25 if d not in seen]


def prf_terms(query: str, bm25: list[list], db: str, fb_docs: int, n_terms: int) -> list[str]:
    fb = bm25[:fb_docs]
    if not fb:
        return []
    texts = bm25_index.get_texts([d for d, _ in fb], db)
    m = max(s for _, s in fb)
    w = {d: math.exp(s - m) for d, s in fb}
    z = sum(w.values())
    qterms = set(TOKEN.findall(query.lower()))
    p: Counter = Counter()
    for d, _ in fb:
        toks = TOKEN.findall(texts.get(d, "").lower())
        if not toks:
            continue
        for t, c in Counter(toks).items():
            if t not in ENGLISH_STOP_WORDS and t not in qterms:
                p[t] += (w[d] / z) * c / len(toks)
    return [t for t, _ in p.most_common(n_terms)]


def main() -> None:
    corpus, split, step = sys.argv[1:4]
    qids, queries, db = setup(corpus, split)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{corpus}_{split}_{step}.json"
    done = json.loads(out_path.read_text()) if out_path.exists() else {}
    bm25 = None if step == "bm25" else json.loads((OUT_DIR / f"{corpus}_{split}_bm25.json").read_text())

    for i, q in enumerate(qids, 1):
        if q in done:
            continue
        text = queries[q]
        if step == "bm25":
            done[q] = [[d, s] for d, s in bm25_index.search(text, db, k=K)]
        elif step == "rerank":
            done[q] = rerank(text, bm25[q], db)
        else:
            _, d, t = step.split("_")
            terms = prf_terms(text, bm25[q], db, int(d[1:]), int(t[1:]))
            hits = bm25_index.search(" ".join(terms), db, k=K) if terms else []
            done[q] = {"terms": terms, "ranking": [[d, s] for d, s in hits]}
        if i % 100 == 0 or i == len(qids):
            out_path.write_text(json.dumps(done))
            if step == "rerank":
                import torch
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            print(f"{corpus} {split} {step}: {i}/{len(qids)}", flush=True)
    out_path.write_text(json.dumps(done))


if __name__ == "__main__":
    main()
