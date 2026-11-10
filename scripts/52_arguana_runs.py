#!/usr/bin/env python3
"""ArguAna rankings, telemetry and embeddings, as pre-committed in
configs/arguana_precommit.json. Reads no relevance judgment: scripts/53
does all labelling and analysis.

Every BM25 search goes through a wrapper that asks for k+1 hits and drops
the current query's own id (each ArguAna query is also a corpus document),
so rankings, t0/t1 telemetry and the expansion retrieval never see the
self-match.

Outputs (local; results/ is gitignored):
  results/arguana/{step}.json   per query id
Steps: bm25 (top-100 + 8 t0 features + NQC + query length), rerank,
  prf_d{D}_t{T} (expansion-term ranking; val only unless --all),
  qe (planner-selected LLM expansion, unfused, no reranker), embed.
Each record carries the wall-clock latency of its intervention in ms.

Usage: PYTHONPATH=. python3 scripts/52_arguana_runs.py STEP [--all]
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

DB = "data/arguana_bm25.sqlite3"
OUT_DIR = Path("results/arguana")
K = 100
SPLIT = json.loads(Path("configs/arguana_split.json").read_text())

_search = bm25_index.search
_exclude: list[str | None] = [None]


def _search_without_self(query: str, db_path: str, k: int = 100):
    return [h for h in _search(query, db_path, k=k + 1) if h[0] != _exclude[0]][:k]


bm25_index.search = _search_without_self

spec = importlib.util.spec_from_file_location("iv48", "scripts/48_intervention_runs.py")
iv48 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(iv48)


def nqc(scores: list[float]) -> float:
    s = np.array(scores, dtype=float)
    return float(s.std() / s.mean()) if len(s) and s.mean() else 0.0


def step_bm25(q: str, text: str, _bm25) -> dict:
    from src.agent.steps import retrieve_and_observe_initial

    init = retrieve_and_observe_initial(text, DB, k=K)
    r = init.o0.retrieval
    t0 = [r.top_score, r.score_margin, r.score_entropy, r.query_term_coverage,
          r.mean_rare_term_idf, r.entity_overlap, r.result_coherence, r.ranking_stability]
    return {"ranking": [[d, s] for d, s in init.r0], "t0": t0,
            "nqc": nqc([s for _, s in init.r0]), "qlen": len(text.split()),
            "self_in_ranking": any(d == q for d, _ in init.r0)}


def step_rerank(q: str, text: str, bm25) -> dict:
    t = time.time()
    ranking = iv48.rerank(text, bm25[q]["ranking"], DB)
    return {"ranking": ranking, "ms": (time.time() - t) * 1000,
            "self_in_ranking": any(d == q for d, _ in ranking)}


def step_prf(q: str, text: str, bm25, fb_docs: int, n_terms: int) -> dict:
    t = time.time()
    terms = iv48.prf_terms(text, bm25[q]["ranking"], DB, fb_docs, n_terms)
    hits = bm25_index.search(" ".join(terms), DB, k=K) if terms else []
    return {"terms": terms, "ranking": [[d, s] for d, s in hits], "ms": (time.time() - t) * 1000,
            "self_in_ranking": any(d == q for d, _ in hits)}


def step_qe(q: str, text: str, _bm25) -> dict:
    from src.agent.actions import EXPANSION_ACTIONS
    from src.agent.controller import ActionSelector
    from src.agent.state import AgentState
    from src.agent.steps import act_once, build_evidence, retrieve_and_observe_initial
    from src.llm_client import LLMUnavailableError

    init = retrieve_and_observe_initial(text, DB, k=K)
    action = ActionSelector().select(text, init.o0)
    rec = {"action": action.value, "ranking": None, "ms": 0.0, "llm_error": False}
    if action not in EXPANSION_ACTIONS:
        return rec
    t = time.time()
    state = AgentState(q0=text, q_t=text, R_t=init.r0_ids, O_t=init.o0, B_t=1)
    try:
        res = act_once(action, state, init.r0, DB, build_evidence(init.r0_ids, DB), k=K, use_reranker=False)
    except LLMUnavailableError:
        rec["llm_error"] = True
        return rec
    rec.update({"ranking": res.candidate.R_t, "expansion": res.generated_expansion,
                "ms": (time.time() - t) * 1000, "llm_ms": res.llm_latency_ms,
                "self_in_ranking": q in res.candidate.R_t})
    return rec


def main() -> None:
    step = sys.argv[1]
    queries = load_queries("arguana")
    if step.startswith("prf") and "--all" not in sys.argv:
        qids = SPLIT["val"]
    else:
        qids = sorted(queries)
    if step == "embed":
        from src.retrieval.dense_index import _get_model

        emb = _get_model().encode([queries[q] for q in qids], batch_size=32, normalize_embeddings=True,
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
        fn = {"bm25": step_bm25, "rerank": step_rerank, "qe": step_qe}[step]

    t_start, n_new = time.time(), 0
    for i, q in enumerate(qids, 1):
        if q in done:
            continue
        _exclude[0] = q
        done[q] = fn(q, queries[q], bm25)
        n_new += 1
        if i % 50 == 0 or i == len(qids):
            out_path.write_text(json.dumps(done))
            if step == "rerank":
                import torch
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            rate = (time.time() - t_start) / n_new
            print(f"arguana {step}: {i}/{len(qids)} ({rate:.2f}s/q, eta {rate * (len(qids) - i) / 60:.1f}min)",
                  flush=True)
    out_path.write_text(json.dumps(done))
    print(f"self-match in rankings: {sum(bool(r.get('self_in_ranking')) for r in done.values())}", flush=True)


if __name__ == "__main__":
    main()
