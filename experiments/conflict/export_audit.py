"""
Human audit of the string matcher (PREREGISTRATION.md §15.4.7). Not run by us: the labels are the authors'.

    python -m experiments.conflict.export_audit            # writes runs/labelling/audit_200.csv
    python -m experiments.conflict.export_audit --score    # agreement and kappa once the human columns are filled

Samples 200 outputs stratified over (run, condition), conflict and clean questions alike, with the same strata as
judge_matcher plus Claude Sonnet, and a different seed. Fill human_gives_all_correct, human_gives_wrong and
human_abstains with 1/0; the human strict verdict is all-correct and not wrong, compared with the one-way matcher.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict

from .judge_matcher import CONDS, ROOT, RUNS
from .jsonl import read_results

OUT = ROOT / "runs/labelling/audit_200.csv"
HUMAN = ("human_gives_all_correct", "human_gives_wrong", "human_abstains")


def export(n: int = 200):
    from . import data
    ex = {e.id: e for name in ("conflictbank", "ramdocs", "raguard") for e in data.load(name)}
    strata = defaultdict(list)
    for run in RUNS + ["cb_single_sonnet", "holdout_sonnet"]:
        p = ROOT / "runs" / run
        if (p / "results.jsonl").exists() or (p / "results.jsonl.gz").exists():
            for r in read_results(p):
                if r["condition"] in CONDS and r["id"] in ex and r.get("has_conflict") is not None:
                    strata[(run, r["condition"])].append(r)
    rng = random.Random(1)
    keys = sorted(strata)
    per = max(1, -(-n // len(keys)))
    sample = [(k, r) for k in keys for r in rng.sample(strata[k], min(per, len(strata[k])))]
    rng.shuffle(sample)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "id", "run", "condition", "has_conflict", "question", "gold", "wrong", "response_tail",
                    "matcher_strict_oneway", *HUMAN])
        for k, ((run, cond), r) in enumerate(sample[:n]):
            e = ex[r["id"]]
            w.writerow([k, r["id"], run, cond, r["has_conflict"], e.question, " | ".join(e.gold_answers),
                        " | ".join(e.wrong_answers), r["response"][-1500:], r["ramdocs_strict_oneway"], "", "", ""])
    print(f"wrote {min(n, len(sample))} rows to {OUT}")


def score():
    rows = [r for r in csv.DictReader(open(OUT)) if all(r[h] in ("0", "1") for h in HUMAN)]
    if not rows:
        raise SystemExit("no fully labelled rows yet")
    a = [float(r["matcher_strict_oneway"]) for r in rows]
    b = [float(r["human_gives_all_correct"] == "1" and r["human_gives_wrong"] == "0") for r in rows]
    agree = sum(x == y for x, y in zip(a, b)) / len(rows)
    pa, pb = sum(a) / len(a), sum(b) / len(b)
    pe = pa * pb + (1 - pa) * (1 - pb)
    out = {"n": len(rows), "agreement": agree, "kappa": (agree - pe) / (1 - pe) if pe < 1 else 1.0,
           "matcher_only": sum(x > y for x, y in zip(a, b)), "human_only": sum(y > x for x, y in zip(a, b))}
    (ROOT / "runs/labelling/audit_200_score.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    score() if a.score else export(a.n)
