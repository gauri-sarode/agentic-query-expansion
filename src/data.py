"""Query/qrel/corpus loading. NFCorpus/MS MARCO Chameleons/TREC DL via
ir_datasets; TripClick via ir_datasets' tripclick catalog once the local
package is set up (scripts/download_tripclick.py). TripClick documents
are never committed to this repo -- see data/README.md.
"""
from __future__ import annotations

from collections.abc import Iterator

DATASET_IDS = {
    "nfcorpus": "beir/nfcorpus/test",
    "trec-dl-2019": "msmarco-passage/trec-dl-2019/judged",
    "trec-dl-2020": "msmarco-passage/trec-dl-2020/judged",
    # TripClick slices populated once local ir_datasets registration is
    # confirmed against the downloaded package (docs/milestones.md).
    "tripclick-head": "tripclick/head/test",
    "tripclick-torso": "tripclick/torso/test",
    "tripclick-tail": "tripclick/tail/test",
}


def load_docs(dataset: str) -> Iterator[tuple[str, str]]:
    """Yields (doc_id, text) for build_index. text = title + body where the
    dataset provides a title field.
    """
    import ir_datasets

    ds = ir_datasets.load(DATASET_IDS[dataset])
    for doc in ds.docs_iter():
        title = getattr(doc, "title", "") or ""
        text = getattr(doc, "text", "") or getattr(doc, "body", "")
        yield doc.doc_id, f"{title} {text}".strip()


def load_queries(dataset: str) -> dict[str, str]:
    import ir_datasets

    ds = ir_datasets.load(DATASET_IDS[dataset])
    return {q.query_id: q.text for q in ds.queries_iter()}


def load_qrels(dataset: str) -> dict[str, dict[str, int]]:
    import ir_datasets

    ds = ir_datasets.load(DATASET_IDS[dataset])
    qrels: dict[str, dict[str, int]] = {}
    for qrel in ds.qrels_iter():
        qrels.setdefault(qrel.query_id, {})[qrel.doc_id] = qrel.relevance
    return qrels
