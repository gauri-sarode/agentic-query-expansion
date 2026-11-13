#!/usr/bin/env python3
"""Re-time the BM25 + telemetry step on validation queries in a single
process, for the reported telemetry latency (scripts/68 time_val reads the
per-record "ms" of bm25.json on val). Needed when bm25.json was produced by
parallel shards (scripts/75 --shard), whose wall clock is inflated by
contention. Asserts the recomputed ranking, t0, NQC and query length equal
the stored ones, then replaces only "ms" on the val records. Reads no
relevance judgment. Run with nothing else heavy running.

Usage: PYTHONPATH=. python3 scripts/77_iv_retime_telemetry.py CORPUS
"""
from __future__ import annotations

import importlib.util
import json
import sys

spec = importlib.util.spec_from_file_location("r75", "scripts/75_iv_runs.py")
r75 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r75)  # reads the corpus from sys.argv[1]


def main() -> None:
    path = r75.OUT_DIR / "bm25.json"
    stored = json.loads(path.read_text())
    texts = r75.queries()
    val = r75.SPLIT["val"]
    for i, q in enumerate(val, 1):
        new = r75.step_bm25(q, texts[q], None)
        old = stored[q]
        assert all(new[k] == old[k] for k in ("ranking", "t0", "nqc", "qlen")), q
        old["ms_parallel"], old["ms"] = old["ms"], new["ms"]
        if i % 50 == 0:
            print(f"{r75.CORPUS} retime: {i}/{len(val)}", flush=True)
    r75._write(path, stored)
    print(f"{r75.CORPUS} retime: {len(val)} val records re-timed, outputs identical", flush=True)


if __name__ == "__main__":
    main()
