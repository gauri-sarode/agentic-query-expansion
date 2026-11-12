#!/usr/bin/env python3
"""FiQA rankings, telemetry and embeddings, as pre-committed in
configs/fiqa_precommit.json (the scripts/52 ArguAna procedure without the
self-match filter, which FiQA does not need, and without LLM QE). Reads no
relevance judgment: scripts/69 does all labelling and analysis.

Outputs (local; results/ is gitignored):
  results/fiqa/{step}.json   per query id
Steps: bm25 (top-100 + 8 t0 features + NQC + query length), rerank,
  prf_d{D}_t{T} (expansion-term ranking; val only unless --all), embed.
Each intervention record carries its wall-clock latency in ms.

Usage: PYTHONPATH=. python3 scripts/67_fiqa_runs.py STEP [--all] [--limit N]
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

from src.data import load_queries
from src.retrieval import bm25_index

DB = "data/fiqa_bm25.sqlite3"
OUT_DIR = Path("results/fiqa")
K = 100
SPLIT = json.loads(Path("configs/fiqa_split.json").read_text())
QIDS = SPLIT["train"] + SPLIT["val"] + SPLIT["test"]

spec = importlib.util.spec_from_file_location("iv48", "scripts/48_intervention_runs.py")
iv48 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(iv48)


def queries() -> dict[str, str]:
    out: dict[str, str] = {}
    for sp in ("train", "dev", "test"):
        out.update(load_queries(f"fiqa-{sp}"))
    return out


def nqc(scores: list[float]) -> float:
    s = np.array(scores, dtype=float)
    return float(s.std() / s.mean()) if len(s) and s.mean() else 0.0


def step_bm25(q: str, text: str, _bm25) -> dict:
    from src.agent.steps import retrieve_and_observe_initial

    t = time.time()
    init = retrieve_and_observe_initial(text, DB, k=K)
    r = init.o0.retrieval
    t0 = [r.top_score, r.score_margin, r.score_entropy, r.query_term_coverage,
          r.mean_rare_term_idf, r.entity_overlap, r.result_coherence, r.ranking_stability]
    return {"ranking": [[d, s] for d, s in init.r0], "t0": t0, "ms": (time.time() - t) * 1000,
            "nqc": nqc([s for _, s in init.r0]), "qlen": len(text.split())}


def step_rerank(q: str, text: str, bm25) -> dict:
    t = time.time()
    ranking = iv48.rerank(text, bm25[q]["ranking"], DB)
    return {"ranking": ranking, "ms": (time.time() - t) * 1000}


def step_prf(q: str, text: str, bm25, fb_docs: int, n_terms: int) -> dict:
    t = time.time()
    terms = iv48.prf_terms(text, bm25[q]["ranking"], DB, fb_docs, n_terms)
    hits = bm25_index.search(" ".join(terms), DB, k=K) if terms else []
    return {"terms": terms, "ranking": [[d, s] for d, s in hits], "ms": (time.time() - t) * 1000}


def main() -> None:
    step = sys.argv[1]
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    texts = queries()
    qids = SPLIT["val"] if step.startswith("prf") and "--all" not in sys.argv else QIDS
    if step == "embed":
        from src.retrieval.dense_index import _get_model

        emb = _get_model().encode([texts[q] for q in qids], batch_size=32, normalize_embeddings=True,
                                  show_progress_bar=False)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        np.save(OUT_DIR / "embed.npy", emb.astype(np.float32))
        (OUT_DIR / "embed_ids.json").write_text(json.dumps(qids))
        print("embed", emb.shape)
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{step}.json"
    done = json.loads(out_path.read_text()) if out_path.exists() else {}
    bm25 = None if step == "bm25" else json.loads((OUT_DIR / "bm25.json").read_text())
    if step.startswith("prf"):
        _, d, t = step.split("_")
        fn = lambda q, text, b: step_prf(q, text, b, int(d[1:]), int(t[1:]))  # noqa: E731
    else:
        fn = {"bm25": step_bm25, "rerank": step_rerank}[step]
    t_start, n_new = time.time(), 0
    for i, q in enumerate(qids, 1):
        if limit is not None and n_new >= limit:
            break
        if q in done:
            continue
        done[q] = fn(q, texts[q], bm25)
        n_new += 1
        if i % 50 == 0 or i == len(qids):
            out_path.write_text(json.dumps(done))
            if step == "rerank":
                import torch
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            rate = (time.time() - t_start) / n_new
            print(f"fiqa {step}: {i}/{len(qids)} ({rate:.2f}s/q, eta {rate * (len(qids) - i) / 60:.1f}min)",
                  flush=True)
    out_path.write_text(json.dumps(done))
    print(f"fiqa {step}: {len(done)} done, {(time.time() - t_start) / max(n_new, 1):.2f}s/q", flush=True)


if __name__ == "__main__":
    main()
