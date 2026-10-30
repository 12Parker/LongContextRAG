"""
Summarize a run directory.

    python -m experiments.conflict.analyze runs/ramdocs_pilot --reference plain

Per condition: mean primary score (averaged over seeds per question first), split into
conflict / no-conflict questions (H2 vs H3), the paired difference from the reference
with a 95% bootstrap CI, a sign-flip p-value, Holm-adjusted across conditions, plus
mean tokens, cost and per-stage latency.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .jsonl import read_results
from .stats import holm, paired_bootstrap, sign_flip_p


def per_question(rows, cond, subset=None, metric: str = "primary"):
    """Per-question mean of `metric` over seeds, for one condition and subset (None = all)."""
    acc = defaultdict(list)
    for r in rows:
        if r["condition"] != cond or r.get(metric) is None:
            continue
        if subset is not None and r.get("has_conflict") != subset:
            continue
        acc[r["id"]].append(r[metric])
    return {k: float(np.mean(v)) for k, v in acc.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--reference", default="plain")
    ap.add_argument("--metric", default="primary", help="score field, e.g. ramdocs_strict_oneway")
    args = ap.parse_args(argv)
    rows = read_results(Path(args.run_dir))
    groups = {r["id"]: r.get("group", r["id"]) for r in rows}
    conds = sorted({r["condition"] for r in rows}, key=lambda c: (c != args.reference, c))

    report = {}
    for subset_name, subset in (("all", None), ("conflict", True), ("no_conflict", False)):
        ref = per_question(rows, args.reference, subset, args.metric)
        stats, pvals = {}, {}
        for c in conds:
            cur = per_question(rows, c, subset, args.metric)
            if not cur:
                continue
            s = {"mean": float(np.mean(list(cur.values()))), "n": len(cur)}
            if c != args.reference and set(cur) & set(ref):
                s.update(paired_bootstrap(ref, cur, groups))
                pvals[c] = sign_flip_p(ref, cur)
            stats[c] = s
        for c, p in holm(pvals).items():
            stats[c]["p_holm"] = p
        report[subset_name] = stats

    cost = {}
    for c in conds:
        rs = [r for r in rows if r["condition"] == c]
        cost[c] = {k: float(np.mean([r.get(k, 0) or 0 for r in rs]))
                   for k in ("prompt_tokens", "completion_tokens", "cost_usd", "detect_s",
                             "generate_s", "detect_cost_usd", "n_flags")}
    report["cost_latency"] = cost
    report["metric"] = args.metric
    name = "analysis.json" if args.metric == "primary" else f"analysis.{args.metric}.json"
    (Path(args.run_dir) / name).write_text(json.dumps(report, indent=1))

    for subset_name in ("all", "conflict", "no_conflict"):
        print(f"\n## {subset_name}")
        print(f"{'condition':22s} {'n':>5s} {'mean':>7s} {'diff':>7s} {'95% CI':>17s} {'p_holm':>7s}")
        for c, s in report[subset_name].items():
            ci = f"[{s['ci_low']:+.3f},{s['ci_high']:+.3f}]" if "ci_low" in s else ""
            d = f"{s['diff']:+.3f}" if "diff" in s else ""
            p = f"{s['p_holm']:.3f}" if "p_holm" in s else ""
            print(f"{c:22s} {s['n']:5d} {s['mean']:7.3f} {d:>7s} {ci:>17s} {p:>7s}")
    print("\n## cost / latency (means per question)")
    for c, v in cost.items():
        print(f"{c:22s} in={v['prompt_tokens']:7.0f} out={v['completion_tokens']:5.0f} "
              f"${v['cost_usd']+v['detect_cost_usd']:.6f} detect={v['detect_s']:.3f}s "
              f"gen={v['generate_s']:.2f}s flags={v['n_flags']:.2f}")


if __name__ == "__main__":
    main()
