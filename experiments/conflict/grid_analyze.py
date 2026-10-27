"""
Precision x recall (x direction) view of a grid run (see `grid` in run.py).

    python -m experiments.conflict.grid_analyze runs/ramdocs_grid_pilot --reference instruct

Cells are grouped by (prompt mode, false-flag pool, direction accuracy). For each group and
subset (all / conflict / no_conflict) it prints a table: rows are target pair precision,
columns target recall, cells the paired difference in the primary score from the reference
with a 95% cluster-bootstrap CI (not corrected across cells). It also reports the break-even
precision: the lowest precision at and above which no cell is significantly below the
reference. Achieved precision, recall and direction accuracy are in <run>/grid_analysis.json.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .analyze import per_question
from .detectors import parse_synth_name
from .stats import paired_bootstrap

SUBSETS = (("all", None), ("conflict", True), ("no_conflict", False))


def _fmt(t):
    mark = "*" if t["ci_high"] < 0 or t["ci_low"] > 0 else " "
    return f"{t['diff']:+.3f} [{t['ci_low']:+.2f},{t['ci_high']:+.2f}]{mark}".rjust(24)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--reference", default="instruct")
    args = ap.parse_args(argv)
    run = Path(args.run_dir)
    rows = [json.loads(l) for l in open(run / "results.jsonl")]
    achieved = json.loads((run / "synthetic_achieved.json").read_text())
    groups = {r["id"]: r.get("group", r["id"]) for r in rows}
    cells = defaultdict(dict)  # (mode, pool, direction) -> {(p, r): condition}
    for c in {r["condition"] for r in rows}:
        det, _, mode = c.partition("+")
        if det.startswith("synth-"):
            p, r, pool, d = parse_synth_name(det)
            cells[(mode, pool, d)][(p, r)] = c

    report = {}
    for (mode, pool, d), cs in sorted(cells.items(), key=lambda kv: (kv[0][0], kv[0][1], -(kv[0][2] or 0))):
        ps, rs = sorted({p for p, _ in cs}), sorted({r for _, r in cs})
        key = f"{mode}|{pool}|d={d}"
        report[key] = {}
        for subset_name, subset in SUBSETS:
            ref = per_question(rows, args.reference, subset)
            tab = {}
            for (p, r), c in cs.items():
                a = achieved[c.split("+")[0]]
                tab[f"{p:g},{r:g}"] = {**paired_bootstrap(ref, per_question(rows, c, subset), groups),
                                       "mean": sum(per_question(rows, c, subset).values()) / max(len(per_question(rows, c, subset)), 1),
                                       "achieved_precision": a["pair_level"]["precision"],
                                       "achieved_recall": a["pair_level"]["recall"],
                                       "achieved_direction": a.get("direction_accuracy"),
                                       "clean_question_fpr": a["question_level"]["false_positive_rate"]}
            harmful = {p for p in ps if any(tab[f"{p:g},{r:g}"]["ci_high"] < 0 for r in rs)}
            safe = [p for p in ps if not any(q in harmful for q in ps if q >= p)]
            be = min(safe) if safe else None
            report[key][subset_name] = {"cells": tab, "break_even_precision": be}
            print(f"\n## mode={mode} pool={pool} direction={d} subset={subset_name}  diff vs {args.reference}"
                  f"  (rows: precision, cols: recall)")
            print("P \\ R   " + "".join(f"{r:>24g}" for r in rs))
            for p in ps:
                print(f"{p:<8g}" + "".join(_fmt(tab[f"{p:g},{r:g}"]) for r in rs))
            print(f"break-even precision: {be}")

    # direction view: at the highest recall, how does the effect move with direction accuracy?
    dirs = sorted({k[2] for k in cells if k[2] is not None}, reverse=True)
    if dirs:
        for mode in sorted({k[0] for k in cells}):
            for pool in sorted({k[1] for k in cells}):
                r_max = max(r for (m, pl, _), cs in cells.items() if m == mode and pl == pool for _, r in cs)
                for subset_name, _ in SUBSETS:
                    print(f"\n## direction view: mode={mode} pool={pool} recall={r_max:g} subset={subset_name}"
                          f"  diff vs {args.reference}  (rows: precision, cols: direction accuracy)")
                    ps = sorted({p for (m, pl, _), cs in cells.items() if m == mode and pl == pool for p, _ in cs})
                    print("P \\ D   " + "".join(f"{d:>24g}" for d in dirs))
                    for p in ps:
                        print(f"{p:<8g}" + "".join(
                            _fmt(report[f"{mode}|{pool}|d={d}"][subset_name]["cells"][f"{p:g},{r_max:g}"]) for d in dirs))
    (run / "grid_analysis.json").write_text(json.dumps({"reference": args.reference, "groups": report}, indent=1))
    print("\n* = 95% CI excludes 0 (uncorrected). Rows/cols are targets; achieved values in grid_analysis.json.")


if __name__ == "__main__":
    main()
