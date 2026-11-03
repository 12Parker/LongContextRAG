"""
Freeze the §21.2 predictions (PREREGISTRATION.md §21.2, C4) from stage-1 files only: each randomised-direction run's
synthetic detections and its generator's plain answers. No flagged answer is read.

    python -m experiments.conflict.predict21 [--out experiments/conflict/splits/predictions21.json]

The file is committed and pushed before stage 2 (the flagged condition) runs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .revision10 import FOLLOW19, GEN19, vectors19

ROOT = Path(__file__).resolve().parents[2]
RAND = "synth-p1-r1-all-d0.5"
GENS21 = ("luna", "haiku", "terra")


def predict(v, gen, ones=False):
    fr, fw = (1.0, 1.0) if ones else FOLLOW19[gen]
    return float(np.mean(v["right"] * (1 - v["g"]) * fr - v["wrong"] * v["g"] * fw)) if len(v["g"]) else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "experiments/conflict/splits/predictions21.json"))
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    args = ap.parse_args(argv)
    out = {}
    for g in GENS21:
        run, gen = Path(args.runs) / f"randdir_fresh_{g}", GEN19[g]
        if not (run / "results.jsonl").exists() or not (run / f"detections.{RAND}.jsonl").exists():
            print(f"{run.name}: stage 1 incomplete, skipped")
            continue
        flagged = {json.loads(l)["condition"] for l in open(run / "results.jsonl")} - {"plain"}
        if flagged:
            raise SystemExit(f"{run.name} already has flagged rows ({flagged}); the freeze must precede stage 2")
        v = vectors19(run, "conflictbank_fresh", flagged=False, det_name=RAND)["conflict"]
        out[run.name] = {"gen": gen, "n": len(v["g"]), "pred": 100 * predict(v, gen),
                         "pred_ones": 100 * predict(v, gen, ones=True), "follow": list(FOLLOW19[gen]),
                         "right": float(v["right"].mean()), "wrong": float(v["wrong"].mean()),
                         "plain": float(v["g"].mean())}
        print(f"{run.name}: conflict pred {out[run.name]['pred']:+.1f} (n {out[run.name]['n']}, "
              f"right {out[run.name]['right']:.2f}, plain {100 * out[run.name]['plain']:.1f})")
    Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True))
    print(f"{len(out)} runs -> {args.out}")


if __name__ == "__main__":
    main()
