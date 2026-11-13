#!/usr/bin/env python3
"""Rankings, telemetry and embeddings for the independent information-value
test (DBPedia-Entity, HotpotQA), as pre-committed in
configs/info_value_precommit.json: the scripts/67 FiQA procedure unchanged
apart from the corpus. Reads no relevance judgment.

Outputs (local; results/ is gitignored):
  results/{corpus}/{step}.json   per query id
Steps: bm25 (top-100 + 8 t0 features + NQC + query length), rerank,
  prf_d{D}_t{T} (expansion-term ranking; val only unless --all), embed.
Each intervention record carries its wall-clock latency in ms.

Usage: PYTHONPATH=. python3 scripts/75_iv_runs.py CORPUS STEP [--all] [--limit N] [--shard I/N]
       PYTHONPATH=. python3 scripts/75_iv_runs.py CORPUS STEP [--all] --merge
--shard I/N runs the queries at positions = I mod N (in parallel processes) into
{step}.shard{I}.json; --merge folds the shard files into {step}.json in query
order. Outputs are identical to a single-process run; per-record "ms" is not
used for reported latency (scripts/76 time mode measures it in isolation).
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

CORPUS = sys.argv[1]
DB = f"data/{CORPUS}_bm25.sqlite3"
OUT_DIR = Path(f"results/{CORPUS}")
K = 100
SPLIT = json.loads(Path("configs/info_value_split.json").read_text())[CORPUS]
QIDS = SPLIT.get("train", []) + SPLIT["val"] + SPLIT["test"]
SOURCES = {"dbpedia": ("dev", "test"), "hotpotqa": ("train", "dev", "test")}[CORPUS]

spec = importlib.util.spec_from_file_location("iv48", "scripts/48_intervention_runs.py")
iv48 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(iv48)


def queries() -> dict[str, str]:
    out: dict[str, str] = {}
    for sp in SOURCES:
        out.update(load_queries(f"{CORPUS}-{sp}"))
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


def _write(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj))
    tmp.replace(path)


def merge(step: str, qids: list[str]) -> None:
    out_path = OUT_DIR / f"{step}.json"
    done = json.loads(out_path.read_text()) if out_path.exists() else {}
    shards = sorted(OUT_DIR.glob(f"{step}.shard*.json"))
    for sp in shards:
        done.update(json.loads(sp.read_text()))
    _write(out_path, {q: done[q] for q in qids if q in done} | {q: v for q, v in done.items() if q not in set(qids)})
    for sp in shards:
        sp.unlink()
    print(f"{CORPUS} {step}: merged {len(shards)} shards, {sum(q in done for q in qids)}/{len(qids)} done", flush=True)


def main() -> None:
    step = sys.argv[2]
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
    if "--merge" in sys.argv:
        merge(step, qids)
        return
    out_path = OUT_DIR / f"{step}.json"
    done = json.loads(out_path.read_text()) if out_path.exists() else {}
    if "--shard" in sys.argv:
        si, sn = map(int, sys.argv[sys.argv.index("--shard") + 1].split("/"))
        qids = [q for j, q in enumerate(qids) if j % sn == si]
        out_path = OUT_DIR / f"{step}.shard{si}.json"
        if out_path.exists():
            done.update(json.loads(out_path.read_text()))
        done = {q: done[q] for q in qids if q in done}
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
            _write(out_path, done)
            if step == "rerank":
                import torch
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            rate = (time.time() - t_start) / n_new
            print(f"{CORPUS} {step}: {i}/{len(qids)} ({rate:.2f}s/q, eta {rate * (len(qids) - i) / 60:.1f}min)",
                  flush=True)
    _write(out_path, done)
    print(f"{CORPUS} {step}: {len(done)} done, {(time.time() - t_start) / max(n_new, 1):.2f}s/q", flush=True)


if __name__ == "__main__":
    main()
