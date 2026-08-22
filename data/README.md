# data/

Gitignored. Local indexes, caches, and downloaded corpora live here.

- `nfcorpus_bm25.sqlite3`, `bm25_index.sqlite3` — built by
  `scripts/01_nfcorpus_quickstart.sh` / index build scripts.
- `tripclick_raw/` — populated by `scripts/02_download_tripclick.py`.
  **TripClick may not be redistributed** (non-commercial research license,
  access requested and granted through the official TripClick site). Only
  code and aggregated results derived from it are published in this repo
  — never the raw corpus, queries, or qrels themselves.
