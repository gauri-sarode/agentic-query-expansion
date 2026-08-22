"""Parser tests against synthetic files matching the real package's
format -- not the real download (gitignored, not present in CI).
"""
from __future__ import annotations

from src import tripclick


def _write_package(root):
    (root / "documents").mkdir(parents=True)
    (root / "topics").mkdir()
    (root / "qrels").mkdir()

    (root / "documents" / "docs.tsv").write_text(
        "1\tTitle one <eot> Body text one.\n"
        "2\tTitle two <eot> Body text two.\n"
    )
    (root / "topics" / "topics.head.test.tsv").write_text("100\tpain management\n")
    (root / "qrels" / "qrels.raw.head.test.tsv").write_text("100\t0\t1\t1\n100\t0\t2\t0\n")
    (root / "qrels" / "qrels.dctr.head.test.tsv").write_text("100\t0\t1\t2\n")


def test_iter_docs_parses_doc_id_and_text(tmp_path):
    _write_package(tmp_path)
    docs = list(tripclick.iter_docs(tmp_path))
    assert docs == [
        ("1", "Title one <eot> Body text one."),
        ("2", "Title two <eot> Body text two."),
    ]


def test_load_topics_and_qrels(tmp_path):
    _write_package(tmp_path)
    topics = tripclick.load_topics("head", "test", raw_dir=tmp_path)
    assert topics == {"100": "pain management"}

    qrels_raw = tripclick.load_qrels("head", "test", label_type="raw", raw_dir=tmp_path)
    assert qrels_raw == {"100": {"1": 1, "2": 0}}

    qrels_dctr = tripclick.load_qrels("head", "test", label_type="dctr", raw_dir=tmp_path)
    assert qrels_dctr == {"100": {"1": 2}}


def test_available_splits_reflects_disk_contents(tmp_path):
    _write_package(tmp_path)
    splits = tripclick.available_splits(tmp_path)
    assert splits["topics"] == ["topics.head.test.tsv"]
    assert splits["qrels"] == ["qrels.dctr.head.test.tsv", "qrels.raw.head.test.tsv"]
