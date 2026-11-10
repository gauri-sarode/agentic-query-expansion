"""Per-query latency of the ArguAna pre-action features (BM25 search, t0 telemetry incl. the perturbed search, bge-small query embedding), test split, warm caches. Label-free; cited in paper Section 8."""
import json, time, sys, numpy as np
sys.path.insert(0, '.')
from src.data import load_queries
from src.retrieval import bm25_index
from src.observability.telemetry import compute_retrieval_telemetry
from src.retrieval.dense_index import _get_model
DB = "data/arguana_bm25.sqlite3"
split = json.load(open("configs/arguana_split.json"))
test = split["test"] if isinstance(split.get("test"), list) else [q for q, s in split["assignment"].items() if s == "test"]
queries = load_queries("arguana")
qs = [queries[q] for q in test]
m = _get_model(); m.encode(["warm up"], normalize_embeddings=True)
for q in qs[:5]: bm25_index.search(q, DB, k=102)
t_bm, t_tel, t_emb = [], [], []
for q in qs:
    t = time.perf_counter(); r = bm25_index.search(q, DB, k=102)[:100]; t_bm.append(time.perf_counter() - t)
    t = time.perf_counter(); compute_retrieval_telemetry(q, r, DB); t_tel.append(time.perf_counter() - t)
    t = time.perf_counter(); m.encode([q], normalize_embeddings=True); t_emb.append(time.perf_counter() - t)
for n, x in [("bm25", t_bm), ("telemetry", t_tel), ("embed", t_emb)]:
    x = np.array(x) * 1000; print(n, "median %.1f ms  p95 %.1f ms" % (np.median(x), np.percentile(x, 95)))
