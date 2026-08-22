#!/usr/bin/env python3
"""Download the TripClick IR Benchmark package from Google Drive.

TripClick access is granted for non-commercial research use; the dataset
itself must never be redistributed (see data/README.md), so this script
only pulls it into the local, gitignored data/ directory.

Usage:
    python scripts/02_download_tripclick.py --folder-id <GOOGLE_DRIVE_FOLDER_ID>

or set TRIPCLICK_DRIVE_FOLDER_ID in .env. Requires `gdown`
(pip install gdown, already in requirements.txt).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DEST = Path("data/tripclick_raw")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder-id", default=os.getenv("TRIPCLICK_DRIVE_FOLDER_ID"))
    args = parser.parse_args()

    if not args.folder_id:
        print(
            "No Google Drive folder ID given. Pass --folder-id or set "
            "TRIPCLICK_DRIVE_FOLDER_ID in .env.",
            file=sys.stderr,
        )
        return 1

    import gdown

    DEST.mkdir(parents=True, exist_ok=True)
    gdown.download_folder(
        id=args.folder_id, output=str(DEST), quiet=False, use_cookies=False
    )
    print(f"Downloaded TripClick package to {DEST} (gitignored, not redistributed).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
