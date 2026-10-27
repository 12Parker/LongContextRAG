"""
Phase 1: detector-only evaluation (no generation cost except the LLM detector).

    python -m experiments.conflict.detect_eval --dataset ramdocs --detectors caag,nli,llm \
        --out runs/detect_ramdocs

Writes <out>/summary.json and <out>/caag_sweep.csv, and prints CAAG's per-pair flag
rate, which is the --random-rate to use for the placebo condition in run.py.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from . import data
from .detectors import CAAGDetector, STEmbedder
from .metrics import false_flags_by_relation, pair_level, question_level
from .run import make_detector, run_detection


def _flagsets(dets):
    return {eid: {(d.i, d.j) for d in o.pairs} for eid, o in dets.items()}


def caag_sweep(out: Path, examples, thetas_s, thetas_c) -> list[dict]:
    rows = [json.loads(l) for l in open(out / "detections.caag.jsonl")]
    by_id = {e.id: e for e in examples}
    table = []
    for ts in thetas_s:
        for tc in thetas_c:
            flagged = {}
            for r in rows:
                cr = r["extra"].get("caag_rows", {})
                flagged[r["id"]] = {tuple(map(int, k.split(","))) for k, v in cr.items()
                                    if v["sim"] >= ts and v["final"] >= tc}
            exs = [by_id[r["id"]] for r in rows]
            q = question_level(exs, flagged)
            p = pair_level(exs, flagged)
            n_pairs = sum(len(e.passages) * (len(e.passages) - 1) // 2 for e in exs)
            table.append({"theta_s": ts, "theta_c": tc,
                          "pair_flag_rate": sum(map(len, flagged.values())) / max(n_pairs, 1),
                          "q_precision": q["precision"], "q_recall": q["recall"], "q_f1": q["f1"],
                          "q_fpr": q["false_positive_rate"],
                          "pair_precision": p["precision"], "pair_recall": p["recall"], "pair_f1": p["f1"]})
    with open(out / "caag_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader(); w.writerows(table)
    return table


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--detectors", default="caag,nli")
    ap.add_argument("--signal-ablation", action="store_true",
                    help="also run CAAG with each signal alone")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--theta-s", type=float, default=0.65)
    ap.add_argument("--theta-c", type=float, default=0.35)
    ap.add_argument("--nli-model", default="cross-encoder/nli-deberta-v3-base")
    ap.add_argument("--nli-threshold", type=float, default=0.5)
    ap.add_argument("--random-rate", type=float, default=None)
    ap.add_argument("--ramdocs-labels", default="exclude", choices=data.RAMDOCS_LABEL_FIXES,
                    help="how to treat documents typed correct that contain a wrong answer (data.load_ramdocs)")
    ap.add_argument("--synth-seed", type=int, default=0)
    args = ap.parse_args(argv)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    kw = {"label_fix": args.ramdocs_labels} if args.dataset == "ramdocs" else {}
    examples = data.load(args.dataset, n=args.n, seed=args.sample_seed, **kw)
    llm = None
    if "llm" in args.detectors:
        from .llm import OpenAILLM
        llm = OpenAILLM(args.model, temperature=0.0)

    summary = {}
    n_pairs = sum(len(e.passages) * (len(e.passages) - 1) // 2 for e in examples)
    for name in args.detectors.split(","):
        det = make_detector(name, args, llm, examples)
        dets = run_detection(det, examples, out)
        fl = _flagsets(dets)
        summary[name] = {
            "question_level": question_level(examples, fl),
            "pair_level": pair_level(examples, fl),
            "false_flags_by_gold_relation": false_flags_by_relation(examples, fl),
            "pair_flag_rate": sum(map(len, fl.values())) / max(n_pairs, 1),
            "mean_latency_s": float(np.mean([o.latency_s for o in dets.values()])),
            "total_cost_usd": float(sum(o.cost_usd for o in dets.values())),
        }

    if args.signal_ablation:
        emb = STEmbedder(args.embedder)
        for sig in ("negation", "entity", "temporal"):
            kw = {f"use_{s}": s == sig for s in ("negation", "entity", "temporal")}
            det = CAAGDetector(emb, args.theta_s, args.theta_c, name=f"caag_{sig}_only", **kw)
            fl = _flagsets(run_detection(det, examples, out))
            summary[det.name] = {"question_level": question_level(examples, fl),
                                 "pair_level": pair_level(examples, fl),
                                 "pair_flag_rate": sum(map(len, fl.values())) / max(n_pairs, 1)}

    if "caag" in args.detectors.split(","):
        caag_sweep(out, examples, np.round(np.arange(0.40, 0.91, 0.05), 2).tolist(),
                   np.round(np.arange(0.10, 0.61, 0.05), 2).tolist())
        print(f"CAAG per-pair flag rate at defaults: {summary['caag']['pair_flag_rate']:.4f} "
              f"(use as --random-rate for the placebo)")

    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
