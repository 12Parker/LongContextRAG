"""
Check the strict string matcher against an LLM judge (PREREGISTRATION.md §13.4, exploratory).

    python -m experiments.conflict.judge_matcher [--n 200] [--judge gpt-6.1-terra]

Samples conflict-question outputs stratified over (run, condition), asks the judge whether the response's final
answer gives every gold answer, gives a wrong answer, or abstains, and compares the judge's strict verdict
(gold given and no wrong answer) with the matcher's one-way strict score. Writes runs/judge_matcher.json.
No human labels: this measures agreement with a strong model, not correctness.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from .jsonl import read_results
from .llm import OpenAILLM

ROOT = Path(__file__).resolve().parents[2]
RUNS = ["cb_single", "cb_single_luna", "cb_single_terra", "cb_single_haiku", "holdout_main", "holdout_luna",
        "holdout_terra", "holdout_haiku", "rg_mini", "rg_haiku", "rg_terra"]
CONDS = ("plain", "oracle+flag", "oracle+flag_onewrong", "oracle+flag_directional", "llm_dir+flag_directional")

PROMPT = """You are checking how a model's answer was scored. Read the question, the correct answer(s), the wrong
(misinformation) answer(s), and the model's full response. Judge only the model's final answer.

Question: {question}
Correct answer(s): {gold}
Wrong answer(s): {wrong}

Model response:
<<<
{response}
>>>

Reply with JSON only: {{"gives_all_correct": true/false, "gives_wrong": true/false, "abstains": true/false}}
- gives_all_correct: the final answer states every correct answer (paraphrases and aliases count).
- gives_wrong: the final answer states any wrong answer as an answer (mentioning it only to reject it does not count).
- abstains: the final answer is "unknown" or declines to answer."""


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--judge", default="gpt-6.1-terra")
    args = ap.parse_args(argv)
    from . import data
    ex = {}
    for name in ("conflictbank", "ramdocs", "raguard"):
        for e in data.load(name):
            ex[e.id] = e
    strata = defaultdict(list)
    for run in RUNS:
        p = ROOT / "runs" / run
        if not (p / "results.jsonl").exists() and not (p / "results.jsonl.gz").exists():
            continue
        for r in read_results(p):
            if r["condition"] in CONDS and r.get("has_conflict") and r["id"] in ex:
                strata[(run, r["condition"])].append(r)
    rng = random.Random(0)
    keys = sorted(strata)
    per = max(1, args.n // len(keys))
    sample = [r for k in keys for r in rng.sample(strata[k], min(per, len(strata[k])))][:args.n]
    judge = OpenAILLM(args.judge, temperature=0.0, max_tokens=512)
    out = []
    for r in sample:
        e = ex[r["id"]]
        res = judge(PROMPT.format(question=e.question, gold=" | ".join(e.gold_answers), wrong=" | ".join(e.wrong_answers),
                                  response=r["response"][-3000:]), seed=0, json_mode=True)
        m = re.search(r"\{.*\}", res.text, re.S)
        try:
            j = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            j = {}
        judge_strict = float(bool(j.get("gives_all_correct")) and not j.get("gives_wrong"))
        out.append({"id": r["id"], "condition": r["condition"], "matcher": r["ramdocs_strict_oneway"],
                    "judge": judge_strict, "judge_raw": j, "cost_usd": res.cost_usd})
    a = [o["matcher"] for o in out]; b = [o["judge"] for o in out]
    agree = sum(x == y for x, y in zip(a, b)) / len(out)
    pa, pb = sum(a) / len(a), sum(b) / len(b)
    pe = pa * pb + (1 - pa) * (1 - pb)
    kappa = (agree - pe) / (1 - pe) if pe < 1 else 1.0
    summary = {"n": len(out), "agreement": agree, "kappa": kappa, "matcher_acc": pa, "judge_acc": pb,
               "matcher_only": sum(x > y for x, y in zip(a, b)), "judge_only": sum(y > x for x, y in zip(a, b)),
               "judge": args.judge, "cost_usd": sum(o["cost_usd"] for o in out)}
    (ROOT / "runs/judge_matcher.json").write_text(json.dumps({"summary": summary, "items": out}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
