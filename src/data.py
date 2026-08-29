"""Query/qrel/corpus loading. NFCorpus/MS MARCO Chameleons/TREC DL via
ir_datasets. TripClick is parsed directly from the local IR Benchmark
package (src/tripclick.py, data/tripclick_raw/) rather than through
ir_datasets -- see docs/milestones.md. TripClick documents are never
committed to this repo -- see data/README.md.
"""
from __future__ import annotations

from collections.abc import Iterator

from src import tripclick

DATASET_IDS = {
    "nfcorpus": "beir/nfcorpus/test",
    "trec-dl-2019": "msmarco-passage/trec-dl-2019/judged",
    "trec-dl-2020": "msmarco-passage/trec-dl-2020/judged",
}

# dataset name -> (tripclick slice, split). "-test" is reserved for the
# frozen, single-touch generalization evaluation (RQ1/RQ2/RQ3) -- all
# config selection, verifier fitting, and threshold calibration must use
# "-val" instead. See configs/experimental_protocol.yaml.
TRIPCLICK_SLICES = {
    "tripclick-head": ("head", "test"),
    "tripclick-torso": ("torso", "test"),
    "tripclick-tail": ("tail", "test"),
    "tripclick-head-train": ("head", "train"),
    "tripclick-torso-train": ("torso", "train"),
    "tripclick-tail-train": ("tail", "train"),
    "tripclick-head-val": ("head", "val"),
    "tripclick-torso-val": ("torso", "val"),
    "tripclick-tail-val": ("tail", "val"),
}


def load_docs(dataset: str) -> Iterator[tuple[str, str]]:
    """Yields (doc_id, text) for build_index. For ir_datasets-backed
    datasets, text = title + body where the dataset provides a title
    field. TripClick's corpus is shared across all slices/splits, so any
    "tripclick-*" dataset name yields the same full document stream.
    """
    if dataset in TRIPCLICK_SLICES or dataset == "tripclick":
        yield from tripclick.iter_docs()
        return

    import ir_datasets

    ds = ir_datasets.load(DATASET_IDS[dataset])
    for doc in ds.docs_iter():
        title = getattr(doc, "title", "") or ""
        text = getattr(doc, "text", "") or getattr(doc, "body", "")
        yield doc.doc_id, f"{title} {text}".strip()


def load_queries(dataset: str) -> dict[str, str]:
    if dataset in TRIPCLICK_SLICES:
        slice_, split = TRIPCLICK_SLICES[dataset]
        return tripclick.load_topics(slice_, split)

    import ir_datasets

    ds = ir_datasets.load(DATASET_IDS[dataset])
    return {q.query_id: q.text for q in ds.queries_iter()}


def load_qrels(dataset: str, label_type: str = "raw") -> dict[str, dict[str, int]]:
    if dataset in TRIPCLICK_SLICES:
        slice_, split = TRIPCLICK_SLICES[dataset]
        return tripclick.load_qrels(slice_, split, label_type=label_type)

    import ir_datasets

    ds = ir_datasets.load(DATASET_IDS[dataset])
    qrels: dict[str, dict[str, int]] = {}
    for qrel in ds.qrels_iter():
        qrels.setdefault(qrel.query_id, {})[qrel.doc_id] = qrel.relevance
    return qrels
