"""results.jsonl readers. Large run outputs are committed gzip-compressed (results.jsonl.gz, see
runs/README.md); readers use the plain file if present, else the compressed one."""
from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path


def results_path(run: Path) -> Path | None:
    run = Path(run)
    for p in (run / "results.jsonl", run / "results.jsonl.gz"):
        if p.exists():
            return p
    return None


def read_results(run: Path) -> list[dict]:
    p = results_path(run)
    if p is None:
        raise FileNotFoundError(f"no results.jsonl(.gz) in {run}")
    with (gzip.open(p, "rt") if p.suffix == ".gz" else open(p)) as f:
        return [json.loads(l) for l in f]


def ensure_plain(run: Path) -> None:
    """Before appending to a run (resume), restore results.jsonl from the committed .gz."""
    run = Path(run)
    if not (run / "results.jsonl").exists() and (run / "results.jsonl.gz").exists():
        with gzip.open(run / "results.jsonl.gz", "rb") as src, open(run / "results.jsonl", "wb") as dst:
            shutil.copyfileobj(src, dst)
