"""
Freeze the §19 predictions (PREREGISTRATION.md §19.1, §19.3) from stage-1 files only: each cell's detections and
its generator's plain answers. No flagged answer is read.

    python -m experiments.conflict.predict19 [--out experiments/conflict/splits/predictions19.json]

The file is committed before stage 2 (the flagged condition) runs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .revision10 import FOLLOW19, cells19, vectors19

ROOT = Path(__file__).resolve().parents[2]


def predict(v, gen):
    fr, fw = FOLLOW19[gen]
    return float(np.mean(v["right"] * (1 - v["g"]) * fr - v["wrong"] * v["g"] * fw)) if len(v["g"]) else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "experiments/conflict/splits/predictions19.json"))
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    args = ap.parse_args(argv)
    out = {}
    for name, (ds, gen, det, run) in cells19(Path(args.runs)).items():
        if not (run / "results.jsonl").exists() or not (run / "detections.llm_dir.jsonl").exists():
            print(f"{name}: stage 1 incomplete, skipped")
            continue
        V = vectors19(run, ds, flagged=False)
        out[name] = {"gen": gen, "det": det, "dataset": ds,
                     **{part: {"pred": 100 * predict(v, gen) if predict(v, gen) is not None else None, "n": len(v["g"]),
                               "right": float(v["right"].mean()) if len(v["g"]) else None,
                               "wrong": float(v["wrong"].mean()) if len(v["g"]) else None,
                               "plain": float(v["g"].mean()) if len(v["g"]) else None}
                        for part, v in V.items()}}
        print(f"{name}: conflict pred {out[name]['conflict']['pred']:+.1f} (n {out[name]['conflict']['n']})")
    Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True))
    print(f"{len(out)} cells -> {args.out}")


if __name__ == "__main__":
    main()
