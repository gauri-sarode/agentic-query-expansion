#!/usr/bin/env bash
# Build the BM25 index over NFCorpus and run a sanity retrieval + eval
# pass. No dataset access blockers -- this is the immediate dev loop
# (docs/milestones.md).
set -euo pipefail
cd "$(dirname "$0")/.."

python -c "
from src.data import load_docs
from src.retrieval.bm25_index import build_index

build_index(load_docs('nfcorpus'), 'data/nfcorpus_bm25.sqlite3')
"

python -c "
from src.data import load_queries, load_qrels
from src.retrieval.bm25_index import search
from src.eval.metrics import evaluate, mean_metrics

queries = load_queries('nfcorpus')
qrels = load_qrels('nfcorpus')
run = {qid: dict(search(text, 'data/nfcorpus_bm25.sqlite3', k=100)) for qid, text in queries.items()}
print(mean_metrics(evaluate(qrels, run)))
"
