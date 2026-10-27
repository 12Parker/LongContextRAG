"""
Place a real detector on a synthetic grid.

    python -m experiments.conflict.place_detector runs/ramdocs_dirgrid_pilot runs/ramdocs_llmdet_pilot \
        --detector llm_dir --modes flag_directional,remove_suspect

Reads the detector's measured pair precision, recall and direction accuracy from
<det_run>/detector_quality.json, and its actual effect (paired diff vs --reference, by subset)
from <det_run>/results.jsonl. From the grid run it takes every synthetic cell in the same mode,
at its *achieved* (precision, recall, direction), and predicts the detector's effect by
inverse-distance weighting of the k nearest cells. Agreement between actual and predicted is
evidence that the grid summarizes detectors by those three numbers alone. Both runs must use
the same questions and seeds. Writes <det_run>/placement.<detector>.json.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .analyze import per_question
from .detectors import parse_synth_name
from .grid_analyze import SUBSETS
from .stats import paired_bootstrap


def _rows(run: Path):
    return [json.loads(l) for l in open(run / "results.jsonl")]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("grid_run")
    ap.add_argument("det_run")
    ap.add_argument("--detector", default="llm_dir")
    ap.add_argument("--modes", default="flag_directional,remove_suspect")
    ap.add_argument("--reference", default="plain")
    ap.add_argument("--k", type=int, default=4)
    args = ap.parse_args(argv)
    grid, detr = Path(args.grid_run), Path(args.det_run)
    g_rows, d_rows = _rows(grid), _rows(detr)
    achieved = json.loads((grid / "synthetic_achieved.json").read_text())
    q = json.loads((detr / "detector_quality.json").read_text())[args.detector]
    P, R = q["pair_level"]["precision"], q["pair_level"]["recall"]
    D = q["direction_accuracy"] if q["direction_accuracy"] is not None else 0.5
    groups = {r["id"]: r.get("group", r["id"]) for r in d_rows}
    if {r["id"] for r in g_rows} != {r["id"] for r in d_rows}:
        raise SystemExit("grid and detector runs cover different questions; use the same --n/--sample-seed")

    print(f"{args.detector}: pair precision={P:.3f} recall={R:.3f} direction accuracy={D:.3f} "
          f"(direction given on {q['direction_coverage'] or 0:.0%} of true flags; "
          f"clean-question FPR={q['question_level']['false_positive_rate']:.3f})")
    out = {"detector": args.detector, "quality": q, "reference": args.reference, "modes": {}}
    for mode in args.modes.split(","):
        cells = []
        for c in {r["condition"] for r in g_rows}:
            det, _, m = c.partition("+")
            if m == mode and det.startswith("synth-") and parse_synth_name(det)[3] is not None:
                a = achieved[det]
                cells.append((c, a["pair_level"]["precision"], a["pair_level"]["recall"], a["direction_accuracy"] or 0.5))
        if not cells:
            raise SystemExit(f"no directional grid cells in mode {mode}")
        near = sorted(cells, key=lambda t: math.dist((t[1], t[2], t[3]), (P, R, D)))[: args.k]
        out["modes"][mode] = {"nearest_cells": [n[0] for n in near]}
        print(f"\n## mode={mode}  diff vs {args.reference}; nearest cells: "
              + ", ".join(f"{n[0].split('+')[0]}" for n in near))
        print(f"{'subset':12s} {'actual':>8s} {'95% CI':>17s} {'grid-predicted':>15s} {'nearest cell':>13s}")
        for subset_name, subset in SUBSETS:
            ref_d = per_question(d_rows, args.reference, subset)
            act = paired_bootstrap(ref_d, per_question(d_rows, f"{args.detector}+{mode}", subset), groups)
            ref_g = per_question(g_rows, args.reference, subset)
            diffs, ws = [], []
            for c, p, r, d in near:
                diffs.append(paired_bootstrap(ref_g, per_question(g_rows, c, subset), groups, n_boot=200)["diff"])
                ws.append(1 / max(math.dist((p, r, d), (P, R, D)), 1e-6))
            pred = sum(w * x for w, x in zip(ws, diffs)) / sum(ws)
            out["modes"][mode][subset_name] = {"actual": act, "grid_predicted": pred, "nearest_diff": diffs[0]}
            print(f"{subset_name:12s} {act['diff']:+8.3f} [{act['ci_low']:+.3f},{act['ci_high']:+.3f}] "
                  f"{pred:+15.3f} {diffs[0]:+13.3f}")
    (detr / f"placement.{args.detector}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
