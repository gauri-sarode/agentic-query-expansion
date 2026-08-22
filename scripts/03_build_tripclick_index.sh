#!/usr/bin/env bash
# Build the BM25 index over the full TripClick corpus (~1.52M docs) from
# the locally extracted IR Benchmark TSV package
# (data/tripclick_raw/benchmark_tsv/). No JVM (see docs/milestones.md).
set -euo pipefail
cd "$(dirname "$0")/.."

python -c "
from src import tripclick
from src.retrieval.bm25_index import build_index

build_index(tripclick.iter_docs(), 'data/tripclick_bm25.sqlite3')
"
