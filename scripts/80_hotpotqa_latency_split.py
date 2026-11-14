"""EXPLORATORY (post-review). Split HotpotQA pre-action latency into BM25 retrieval (shared by every
policy) and the remaining telemetry (incremental for gated policies), on the
same 50 val queries as scripts/68 time_val. Single process, alternating order."""
import importlib.util, json, sys, time
import numpy as np
sys.argv = ["x", "hotpotqa"]
sys.path.insert(0, ".")
def load(n, p):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
r75 = load("r75", "scripts/75_iv_runs.py"); pp = load("pp60", "scripts/60_partial_probe.py")
from src.retrieval import bm25_index
from src.agent.steps import retrieve_and_observe_initial
texts = r75.queries()
sample = sorted(r75.SPLIT["val"], key=pp._h)[:50]
bm, full = [], []
for i, q in enumerate(sample):
    order = ("bm", "full") if i % 2 == 0 else ("full", "bm")
    for o in order:
        t = time.perf_counter()
        if o == "bm":
            bm25_index.search(texts[q], r75.DB, k=100); bm.append(time.perf_counter() - t)
        else:
            retrieve_and_observe_initial(texts[q], r75.DB, k=100); full.append(time.perf_counter() - t)
bm, full = np.array(bm), np.array(full)
out = {"bm25_only": {"median_s": float(np.median(bm)), "p95_s": float(np.percentile(bm, 95))},
       "bm25_plus_telemetry": {"median_s": float(np.median(full)), "p95_s": float(np.percentile(full, 95))},
       "incremental_telemetry_paired": {"median_s": float(np.median(full - bm)), "p95_s": float(np.percentile(full - bm, 95))}}
print(json.dumps(out, indent=1))
