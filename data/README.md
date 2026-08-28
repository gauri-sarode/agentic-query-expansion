# data/

Gitignored. Local indexes, caches, and downloaded corpora live here.

- `nfcorpus_bm25.sqlite3` — built by `scripts/01_nfcorpus_quickstart.sh`.
- `tripclick_bm25.sqlite3` — built by `scripts/03_build_tripclick_index.sh`.
- `tripclick_raw/benchmark_tsv/` — the TripClick **IR Benchmark** package
  (documents, topics, qrels for HEAD/TORSO/TAIL × train/val/test),
  extracted from the official Google Drive share. `scripts/02_download_tripclick.py`
  can fetch it directly if you don't already have it locally; parsed by
  `src/tripclick.py`. Access was granted through the official TripClick
  request process (non-commercial research use).
  **TripClick may not be redistributed** — only code and aggregated
  results derived from it are published in this repo, never the raw
  corpus, queries, or qrels themselves.
  - Not needed for this project: the **logs** package (raw click logs —
    already distilled into the qrels we use) and the **deep learning
    training** package (neural-retriever training triples — irrelevant
    since the design uses off-the-shelf BM25 + a pretrained cross-encoder,
    no retriever fine-tuning).

**The same restriction applies to `models/*.joblib`** (see
`../models/`, also gitignored): those are statistical models fit on
TripClick data, and TripClick's terms prohibit publicly sharing
"statistical models or other resources created from the dataset"
without permission. This was caught after `models/verifier_v1.joblib`
had already been committed and pushed to the public repo (see git
history around 2026-08-28) — the file was scrubbed from history via
`git filter-repo` and force-push. Regenerate locally via
`python scripts/10_train_verifier_model.py`; never `git add` a
`.joblib` file.
