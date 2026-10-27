"""
Recompute score fields in a run's results.jsonl from the saved responses (no API calls), e.g.
after a fix in metrics.py. The previous file stays in git history.

    python -m experiments.conflict.rescore runs/ramdocs_madam_check
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from . import data
from .jsonl import ensure_plain
from .metrics import score


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    args = ap.parse_args(argv)
    for rd in map(Path, args.run_dirs):
        cfg = json.loads((rd / "config.json").read_text())
        kw = {"label_fix": cfg.get("ramdocs_labels", "none")} if cfg["dataset"] == "ramdocs" else {}
        exs = {e.id: e for e in data.filter_ids(data.load(cfg["dataset"], n=cfg.get("n"),
                                                           seed=cfg.get("sample_seed", 0), **kw), cfg.get("ids_file"))}
        ensure_plain(rd)
        rows = [json.loads(l) for l in open(rd / "results.jsonl")]
        before, after, changed = defaultdict(float), defaultdict(float), 0
        for r in rows:
            new = score(exs[r["id"]], r["response"])
            before[r["condition"]] += r.get("primary") or 0
            changed += new.get("primary") != r.get("primary")
            r.update(new)
            after[r["condition"]] += r.get("primary") or 0
        with open(rd / "results.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        n = defaultdict(int)
        for r in rows:
            n[r["condition"]] += 1
        print(f"{rd}: {changed} of {len(rows)} primary scores changed")
        for c in sorted(n):
            if before[c] != after[c]:
                print(f"  {c:28s} mean primary {before[c]/n[c]:.4f} -> {after[c]/n[c]:.4f}")


if __name__ == "__main__":
    main()
