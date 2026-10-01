#!/usr/bin/env python3
"""Builds a scoped corpus subset for the dense-retrieval pilot, rather
than embedding the full 1,523,878-doc TripClick collection for a
bounded query sample -- a full-corpus embed benchmarked at ~75 docs/s
(~5.7h) is disproportionate to a ~300-query pilot.

Scope: all qrel-relevant documents for the SAMPLED validation+test
queries specifically (not all of TAIL), plus a large random distractor
pool, so the retrieval task stays real (genuine competition among many
non-relevant docs) rather than artificially easy. This is a documented
experimental-design choice, reported as such in the paper -- not a
shortcut that silently changes what's being measured.

Usage: PYTHONPATH=. python3 scripts/34_dense_pilot_build_corpus.py
"""
from __future__ import annotations

import json
import random

from src import tripclick

SEED = 42
N_VAL_QUERIES = 150
N_TEST_QUERIES = 150
N_DISTRACTORS = 150_000


def main() -> None:
    val_topics = tripclick.load_topics("tail", "val")
    test_topics = tripclick.load_topics("tail", "test")
    val_qrels = tripclick.load_qrels("tail", "val", label_type="raw")
    test_qrels = tripclick.load_qrels("tail", "test", label_type="raw")

    rng = random.Random(SEED)
    val_qids = sorted(q for q in val_topics if q in val_qrels)
    test_qids = sorted(q for q in test_topics if q in test_qrels)
    val_sample = rng.sample(val_qids, min(N_VAL_QUERIES, len(val_qids)))
    test_sample = rng.sample(test_qids, min(N_TEST_QUERIES, len(test_qids)))
    print(f"sampled {len(val_sample)} val queries, {len(test_sample)} test queries")

    relevant_docs: set[str] = set()
    for qid in val_sample:
        relevant_docs.update(d for d, r in val_qrels.get(qid, {}).items() if r > 0)
    for qid in test_sample:
        relevant_docs.update(d for d, r in test_qrels.get(qid, {}).items() if r > 0)
    print(f"{len(relevant_docs)} unique relevant docs for the sampled queries")

    out = {
        "val_qids": val_sample,
        "test_qids": test_sample,
        "val_queries": {q: val_topics[q] for q in val_sample},
        "test_queries": {q: test_topics[q] for q in test_sample},
        "val_qrels": {q: val_qrels[q] for q in val_sample},
        "test_qrels": {q: test_qrels[q] for q in test_sample},
    }
    with open("results/dense_pilot_query_sample.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved results/dense_pilot_query_sample.json")

    print(f"scanning full corpus for relevant docs + {N_DISTRACTORS} random distractors...")
    corpus_subset: dict[str, str] = {}
    all_ids_seen = []
    for doc_id, text in tripclick.iter_docs():
        all_ids_seen.append(doc_id)
        if doc_id in relevant_docs:
            corpus_subset[doc_id] = text
    print(f"found {len(corpus_subset)}/{len(relevant_docs)} relevant docs in corpus "
          f"(some qrel doc_ids may be stale/removed)")

    rng2 = random.Random(SEED + 1)
    remaining_needed = N_DISTRACTORS
    distractor_ids = set(rng2.sample(all_ids_seen, min(N_DISTRACTORS, len(all_ids_seen))))
    distractor_ids -= set(corpus_subset.keys())
    print(f"sampled {len(distractor_ids)} distractor ids, re-scanning corpus to pull their text...")

    for doc_id, text in tripclick.iter_docs():
        if doc_id in distractor_ids and doc_id not in corpus_subset:
            corpus_subset[doc_id] = text

    print(f"final scoped corpus: {len(corpus_subset)} docs "
          f"({len(relevant_docs & corpus_subset.keys())} relevant + "
          f"{len(corpus_subset) - len(relevant_docs & corpus_subset.keys())} distractor)")

    with open("results/dense_pilot_corpus_ids.json", "w") as f:
        json.dump(sorted(corpus_subset.keys()), f)

    from src.retrieval import dense_index
    dense_index.build_meta_db(corpus_subset.items(), "data/dense_pilot_meta.sqlite3")


if __name__ == "__main__":
    main()
