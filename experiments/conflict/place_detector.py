"""
Place a real detector on a synthetic grid.

    python -m experiments.conflict.place_detector runs/ramdocs_dirgrid_pilot runs/ramdocs_llmdet_pilot \
        --detector llm_dir --modes flag_directional,remove_suspect

Default `--space doc`: each subset's effect is predicted from grid cells that do the same thing
to that subset's documents in that mode (metrics.doc_level):

  conflict     (conflict questions with all misinformation marked,
                conflict questions with a correct document marked)
  no_conflict  (clean questions with a correct document marked,)
  all          the conflict and no_conflict predictions, weighted by their question counts
               (questions with has_conflict=None are left out of this weighting)

Each prediction is an inverse-distance-weighted mean of the k nearest directional cells. A
single shared location was tried first and failed in the pilot: matching on pooled document
rates missed the clean-question harm, and matching on clean-question harm picked low-precision
cells whose misdirected flags also hurt conflict questions. `--space pair` locates by pair
precision, recall and direction accuracy, kept for comparison (it also mispredicted).

Coordinates come from the runs' detections.*.jsonl and config.json, so older runs work. Both
runs must cover the same questions and seeds with the same RAMDocs labels.
Writes <det_run>/placement.<detector>.<space>.json.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from . import data
from .analyze import per_question
from .detectors import Detection, DetectorOutput, parse_synth_name
from .metrics import direction_accuracy, doc_level, pair_level
from .jsonl import read_results
from .stats import paired_bootstrap

DOC_AXES = {"conflict": ("conflict_q_all_misinfo_marked", "conflict_q_any_correct_marked"),
            "no_conflict": ("clean_q_any_correct_marked",)}
PAIR_AXES = ("precision", "recall", "direction")
SUBSET_FLAG = {"all": None, "conflict": True, "no_conflict": False}


def _rows(run: Path):
    return read_results(run)


def _examples(run: Path):
    cfg = json.loads((run / "config.json").read_text())
    kw = {"label_fix": cfg.get("ramdocs_labels", "none")} if cfg["dataset"] == "ramdocs" else {}
    xs = data.load(cfg["dataset"], n=cfg.get("n"), seed=cfg.get("sample_seed", 0), **kw)
    return data.filter_ids(xs, cfg.get("ids_file")), kw


def _dets(run: Path, name: str) -> dict:
    return {r["id"]: DetectorOutput([Detection(**p) for p in r["pairs"]])
            for r in map(json.loads, open(run / f"detections.{name}.jsonl"))}


def locate(space: str, mode: str, examples, dets) -> dict[str, tuple]:
    """Coordinates per subset ("conflict", "no_conflict") in the given space."""
    if space == "doc":
        d = doc_level(examples, dets, mode)
        return {s: tuple(d[a] or 0.0 for a in axes) for s, axes in DOC_AXES.items()}
    fl = {e: {(x.i, x.j) for x in o.pairs} for e, o in dets.items()}
    p = pair_level(examples, fl)
    xyz = (p["precision"], p["recall"], direction_accuracy(examples, dets) or 0.5)
    return {"conflict": xyz, "no_conflict": xyz}


def idw(here: tuple, cells: list[tuple[tuple, float]], k: int) -> tuple[float, list[int]]:
    """Inverse-distance-weighted mean of the k nearest (coords, value) cells; exact hit wins."""
    order = sorted(range(len(cells)), key=lambda n: math.dist(cells[n][0], here))[:k]
    ws = [1 / max(math.dist(cells[n][0], here), 1e-6) for n in order]
    return sum(w * cells[n][1] for w, n in zip(ws, order)) / sum(ws), order


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("grid_run")
    ap.add_argument("det_run")
    ap.add_argument("--detector", default="llm_dir")
    ap.add_argument("--modes", default="flag_directional,remove_suspect")
    ap.add_argument("--reference", default="plain")
    ap.add_argument("--space", default="doc", choices=("doc", "pair"))
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--metric", default="primary")
    args = ap.parse_args(argv)
    grid, detr = Path(args.grid_run), Path(args.det_run)
    g_rows, d_rows = _rows(grid), _rows(detr)
    if {r["id"] for r in g_rows} != {r["id"] for r in d_rows}:
        raise SystemExit("grid and detector runs cover different questions; use the same --n/--sample-seed")
    g_ex, g_kw = _examples(grid)
    d_ex, d_kw = _examples(detr)
    if g_kw != d_kw:
        raise SystemExit(f"runs use different RAMDocs labels: grid {g_kw}, detector {d_kw}")
    groups = {r["id"]: r.get("group", r["id"]) for r in d_rows}
    det_out = _dets(detr, args.detector)
    n_sub = {s: sum(e.has_conflict is f for e in d_ex) for s, f in SUBSET_FLAG.items() if f is not None}

    out = {"detector": args.detector, "space": args.space, "reference": args.reference,
           "metric": args.metric, "modes": {}}
    for mode in args.modes.split(","):
        here = locate(args.space, mode, d_ex, det_out)
        cells = [c for c in sorted({r["condition"] for r in g_rows})
                 if c.partition("+")[2] == mode and c.startswith("synth-")
                 and parse_synth_name(c.partition("+")[0])[3] is not None]
        if not cells:
            raise SystemExit(f"no directional grid cells in mode {mode}")
        where = {c: locate(args.space, mode, g_ex, _dets(grid, c.partition("+")[0])) for c in cells}
        effect = {}
        for s, flag in SUBSET_FLAG.items():
            ref = per_question(g_rows, args.reference, flag, args.metric)
            effect[s] = {c: paired_bootstrap(ref, per_question(g_rows, c, flag, args.metric), groups, n_boot=200)["diff"]
                         for c in cells}
        print(f"\n## {args.detector} mode={mode} space={args.space}")
        res = {}
        for s in ("conflict", "no_conflict"):
            pred, order = idw(here[s], [(where[c][s], effect[s][c]) for c in cells], args.k)
            res[s] = {"location": here[s], "grid_predicted": pred, "nearest_cells": [cells[n] for n in order]}
            print(f"  {s} location {tuple(round(v, 3) for v in here[s])}; nearest: "
                  + ", ".join(cells[n].split("+")[0] for n in order))
        res["all"] = {"grid_predicted": sum(res[s]["grid_predicted"] * n_sub[s] for s in n_sub) / sum(n_sub.values())}
        print(f"{'subset':12s} {'actual':>8s} {'95% CI':>17s} {'grid-predicted':>15s}")
        for s, flag in SUBSET_FLAG.items():
            act = paired_bootstrap(per_question(d_rows, args.reference, flag, args.metric),
                                   per_question(d_rows, f"{args.detector}+{mode}", flag, args.metric), groups)
            res[s]["actual"] = act
            print(f"{s:12s} {act['diff']:+8.3f} [{act['ci_low']:+.3f},{act['ci_high']:+.3f}] "
                  f"{res[s]['grid_predicted']:+15.3f}")
        out["modes"][mode] = res
    suffix = "" if args.metric == "primary" else f".{args.metric}"
    (detr / f"placement.{args.detector}.{args.space}{suffix}.json").write_text(json.dumps(out, indent=1, default=list))


if __name__ == "__main__":
    main()
