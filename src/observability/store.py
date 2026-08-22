"""Trace store: append-only JSONL by default (cheap, human-inspectable,
fine at this experiment's scale), with an optional DuckDB/Parquet load
path for analysis. OTel export is a naming-convention/interop concern,
not a requirement to run a commercial observability backend.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.observability.trace_schema import TraceRecord


class JsonlTraceStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: TraceRecord) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(record.to_dict()) + "\n")

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open() as f:
            return [json.loads(line) for line in f if line.strip()]
