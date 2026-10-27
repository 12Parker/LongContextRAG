"""
API spend ledger. Every real (non-cached) call is stored once in the LLM cache, so summing the
cache's recorded costs gives total spend. Writes runs/COSTS.md.

    python -m experiments.conflict.costs [--budget 100]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import date
from pathlib import Path

from .llm import CACHE_PATH, PRICES_CHECKED

RUNS = Path(__file__).resolve().parents[2] / "runs"


def total_spend() -> float:
    db = sqlite3.connect(CACHE_PATH)
    return sum(json.loads(v)["cost_usd"] for (v,) in db.execute("SELECT v FROM c"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=100.0)
    args = ap.parse_args(argv)
    db = sqlite3.connect(CACHE_PATH)
    vals = [json.loads(v) for (v,) in db.execute("SELECT v FROM c")]
    total = sum(v["cost_usd"] for v in vals)
    pt = sum(v["prompt_tokens"] for v in vals); ct = sum(v["completion_tokens"] for v in vals)
    lines = [f"# API spend ledger ({date.today().isoformat()})", "",
             f"- Real API calls: {len(vals):,}",
             f"- Tokens: {pt:,} prompt, {ct:,} completion",
             f"- **Total spend: ${total:.2f}** of a ${args.budget:.0f} budget",
             f"- Prices: {PRICES_CHECKED}", "",
             "Computed from runs/llm_cache.sqlite (one entry per real call; cache hits cost nothing).",
             "Calls priced before cached-token pricing was added are counted at the full input rate.", ""]
    (RUNS / "COSTS.md").write_text("\n".join(lines))
    print("\n".join(lines[2:5]))
    if total > args.budget:
        raise SystemExit("OVER BUDGET")


if __name__ == "__main__":
    main()
