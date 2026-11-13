"""EXPLORATORY (configs/partial_probe_protocol.json): cross-encoder scores
of the first 20 BM25 candidates per query, same model and document text
as the stored reranking runs. Reads no relevance labels. Audits the
recomputed scores against the stored run where it has them.

Outputs (results/ is gitignored): results/probes/{corpus}_{split}.json

Usage: PYTHONPATH=. python scripts/58_probe_scores.py CORPUS SPLIT
  CORPUS in {tripclick, nq, arguana, fiqa}; SPLIT in {val, test} (arguana, fiqa: all)
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from src.retrieval import bm25_index  # noqa: E402
from src.retrieval.rerank import _model  # noqa: E402

K = 20
OUT = Path("results/probes")


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def inputs(corpus: str, split: str) -> tuple[list[str], dict, dict, dict, str]:
    """qids, query texts, BM25 rankings, stored rerank rankings, db path."""
    if corpus in ("dbpedia", "hotpotqa"):
        r75 = _load("r75", "scripts/75_iv_runs.py")  # reads the corpus from sys.argv[1]
        bm = {q: v["ranking"] for q, v in json.loads(Path(f"results/{corpus}/bm25.json").read_text()).items()}
        rr = {q: v["ranking"] for q, v in json.loads(Path(f"results/{corpus}/rerank.json").read_text()).items()}
        return r75.QIDS, r75.queries(), bm, rr, r75.DB
    if corpus == "fiqa":
        r67 = _load("r67", "scripts/67_fiqa_runs.py")
        bm = {q: v["ranking"] for q, v in json.loads(Path("results/fiqa/bm25.json").read_text()).items()}
        rr = {q: v["ranking"] for q, v in json.loads(Path("results/fiqa/rerank.json").read_text()).items()}
        return r67.QIDS, r67.queries(), bm, rr, r67.DB
    if corpus == "arguana":
        a52 = _load("a52", "scripts/52_arguana_runs.py")
        bm = {q: v["ranking"] for q, v in json.loads(Path("results/arguana/bm25.json").read_text()).items()}
        rr = {q: v["ranking"] for q, v in json.loads(Path("results/arguana/rerank.json").read_text()).items()}
        from src.data import load_queries
        return sorted(bm), load_queries("arguana"), bm, rr, a52.DB
    iv48 = _load("iv48", "scripts/48_intervention_runs.py")
    qids, queries, db = iv48.setup(corpus, split)  # sets NQ's frozen BM25 weights
    runs = Path("results/interventions")
    bm = json.loads((runs / f"{corpus}_{split}_bm25.json").read_text())
    rr = json.loads((runs / f"{corpus}_{split}_rerank.json").read_text())
    return qids, queries, bm, rr, db


def main() -> None:
    corpus, split = sys.argv[1], sys.argv[2]
    qids, queries, bm, rr, db = inputs(corpus, split)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{corpus}_{split}.json"
    done = json.loads(path.read_text()) if path.exists() else {}
    for i, q in enumerate(qids, 1):
        if q in done:
            continue
        ids = [d for d, _ in bm[q][:K]]
        texts = bm25_index.get_texts(ids, db)
        keep = [d for d in ids if d in texts]
        scores = _model().predict([(queries[q], texts[d]) for d in keep]) if keep else []
        done[q] = [[d, float(s)] for d, s in zip(keep, scores)]
        if i % 100 == 0 or i == len(qids):
            path.write_text(json.dumps(done))
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            print(f"{corpus} {split} {i}/{len(qids)}", flush=True)
    path.write_text(json.dumps(done))

    diffs, missing = [], 0
    for q in qids:
        stored = {d: s for d, s in rr[q] if s is not None}
        for d, s in done[q]:
            if d in stored:
                diffs.append(abs(stored[d] - s))
        missing += K - len(done[q])
    diffs = np.array(diffs)
    print(f"AUDIT {corpus} {split}: compared {len(diffs)}, max |diff| {diffs.max():.2e}, "
          f"over 1e-3: {(diffs > 1e-3).sum()}, missing texts {missing}", flush=True)


if __name__ == "__main__":
    main()
