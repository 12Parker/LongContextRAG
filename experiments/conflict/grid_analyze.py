"""
Precision x recall (x direction) view of a grid run (see `grid` in run.py).

    python -m experiments.conflict.grid_analyze runs/ramdocs_grid_pilot --reference instruct

Cells are grouped by (prompt mode, false-flag pool, direction accuracy). For each group and
subset (all / conflict / no_conflict) it prints a table: rows are target pair precision,
columns target recall, cells the paired difference in the primary score from the reference
with a 95% cluster-bootstrap CI, and a sign-flip p-value Holm-adjusted across the cells of that
table. "+" / "-" mark a significant gain / harm after that adjustment.

Per recall column it reports
  break_even_precision  the lowest precision p such that every cell at precision >= p is a
                        significant GAIN (Holm). None if no such p. This is the headline number.
  harm_free_precision   the lowest p such that no cell at precision >= p is a significant harm.
                        Weaker: "not shown harmful" is not "safe"; report it only as secondary.
Achieved precision, recall and direction accuracy are in <run>/grid_analysis.json.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .analyze import per_question
from .detectors import parse_synth_name
from .jsonl import read_results
from .stats import holm, paired_bootstrap, sign_flip_p

SUBSETS = (("all", None), ("conflict", True), ("no_conflict", False))


def _sig(t, alpha):
    return t["p_holm"] < alpha and t["diff"] != 0


def _fmt(t, alpha=0.05):
    mark = ("+" if t["diff"] > 0 else "-") if _sig(t, alpha) else " "
    return f"{t['diff']:+.3f} [{t['ci_low']:+.2f},{t['ci_high']:+.2f}]{mark}".rjust(24)


def thresholds(tab: dict, ps: list[float], rs: list[float], alpha: float = 0.05) -> dict:
    """Per recall column: break-even precision (all cells at >= p are significant gains) and
    harm-free precision (no cell at >= p is a significant harm). `tab` keys are "p,r"."""
    out = {}
    for r in rs:
        col = [tab[f"{p:g},{r:g}"] for p in ps]
        gain = [_sig(t, alpha) and t["diff"] > 0 for t in col]
        harm = [_sig(t, alpha) and t["diff"] < 0 for t in col]
        be = next((ps[i] for i in range(len(ps)) if all(gain[i:])), None)
        hf = next((ps[i] for i in range(len(ps)) if not any(harm[i:])), None)
        out[f"{r:g}"] = {"break_even_precision": be, "harm_free_precision": hf}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--reference", default="instruct")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--metric", default="primary")
    ap.add_argument("--n-perm", type=int, default=5000)
    args = ap.parse_args(argv)
    run = Path(args.run_dir)
    rows = read_results(run)
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
            ref = per_question(rows, args.reference, subset, args.metric)
            tab, pv = {}, {}
            for (p, r), c in cs.items():
                a = achieved[c.split("+")[0]]
                cur = per_question(rows, c, subset, args.metric)
                pv[f"{p:g},{r:g}"] = sign_flip_p(ref, cur, n_perm=args.n_perm)
                tab[f"{p:g},{r:g}"] = {**paired_bootstrap(ref, cur, groups), "p": pv[f"{p:g},{r:g}"],
                                       "mean": sum(cur.values()) / max(len(cur), 1),
                                       "achieved_precision": a["pair_level"]["precision"],
                                       "achieved_recall": a["pair_level"]["recall"],
                                       "achieved_direction": a.get("direction_accuracy"),
                                       "clean_question_fpr": a["question_level"]["false_positive_rate"]}
            for k, ph in holm(pv).items():
                tab[k]["p_holm"] = ph
            th = thresholds(tab, ps, rs, args.alpha)
            report[key][subset_name] = {"cells": tab, "thresholds_by_recall": th}
            print(f"\n## mode={mode} pool={pool} direction={d} subset={subset_name}  diff vs {args.reference}"
                  f"  (rows: precision, cols: recall)")
            print("P \\ R   " + "".join(f"{r:>24g}" for r in rs))
            for p in ps:
                print(f"{p:<8g}" + "".join(_fmt(tab[f"{p:g},{r:g}"], args.alpha) for r in rs))
            print("break-even precision by recall: "
                  + ", ".join(f"r={r}: {v['break_even_precision']}" for r, v in th.items())
                  + "   (harm-free: " + ", ".join(f"r={r}: {v['harm_free_precision']}" for r, v in th.items()) + ")")

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
                            _fmt(report[f"{mode}|{pool}|d={d}"][subset_name]["cells"][f"{p:g},{r_max:g}"], args.alpha)
                            for d in dirs))
    name = "grid_analysis.json" if args.metric == "primary" else f"grid_analysis.{args.metric}.json"
    (run / name).write_text(json.dumps({"reference": args.reference, "alpha": args.alpha, "metric": args.metric,
                                        "groups": report}, indent=1))
    print(f"\n+/- = significant gain/harm, sign-flip p Holm-adjusted within each table, alpha={args.alpha}. "
          "Rows/cols are targets; achieved values in grid_analysis.json.")


if __name__ == "__main__":
    main()
