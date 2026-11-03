"""
LLM judge for answers the automatic metrics cannot score (CONFLICTS opinion/complementary items)
and for checking those metrics elsewhere. Use a model from a DIFFERENT family than the generator.

    python -m experiments.conflict.judge runs/conflicts_pilot --model <judge-model> \
        --base-url <endpoint> --api-key-env <VAR> --price-input X --price-output Y \
        --per-stratum 20 --human 100
    python -m experiments.conflict.judge runs/conflicts_pilot --agreement runs/conflicts_pilot/human_labels.csv

Samples up to --per-stratum results per (condition, has_conflict, conflict_type), judges them, and
writes <run>/judgments.jsonl. It also exports <run>/human_labels.csv with --human of the judged
items (judge verdicts hidden) for the ~100-label agreement check; --agreement reports Cohen's kappa.

RUBRIC IS A DRAFT: align it with the CONFLICTS paper's evaluation (Cattan et al. 2025) before use.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from . import data
from .jsonl import read_results

JUDGE_PROMPT = """You are grading an answer to a question that was answered from retrieved passages.

Question: {question}
Reference answer: {reference}
The passages {conflict_note}

Answer to grade:
{response}

Grade it on two criteria:
1. correct: 1 if the answer's final answer agrees with the reference answer (same entity, value or
   claim; wording may differ), 0 if not. If there is no reference answer, use null.
2. handles_conflict: 1 if the answer explicitly notes the disagreement between sources and either
   resolves it with a stated reason or presents the competing views; 0 if it ignores a real
   disagreement or presents one side as settled. If the passages do not disagree, use null.

Return JSON only: {{"correct": 0|1|null, "handles_conflict": 0|1|null, "reason": "<one sentence>"}}"""

CONFLICT_NOTE = {True: "disagree with each other ({ctype}).", False: "do not disagree.",
                 None: "may or may not disagree."}


def _sample(rows, per_stratum: int, seed: int):
    strata = defaultdict(list)
    for r in rows:
        strata[(r["condition"], r.get("has_conflict"), r.get("conflict_type"))].append(r)
    rng = random.Random(seed)
    picked = []
    for k in sorted(strata, key=str):
        xs = sorted(strata[k], key=lambda r: (r["id"], r["seed"]))
        picked += rng.sample(xs, min(per_stratum, len(xs)))
    return picked


def judge_rows(rows, examples: dict, llm, seed: int = 0) -> list[dict]:
    out = []
    for r in rows:
        ex = examples[r["id"]]
        prompt = JUDGE_PROMPT.format(
            question=ex.question, reference=" | ".join(ex.gold_answers) or "(none)",
            conflict_note=CONFLICT_NOTE[ex.has_conflict].format(ctype=ex.conflict_type or "unspecified"),
            response=r["response"])
        res = llm(prompt, seed=seed, json_mode=True, max_tokens=200)
        try:
            v = json.loads(res.text)
        except json.JSONDecodeError:
            v = {"correct": None, "handles_conflict": None, "reason": "unparseable: " + res.text[:200]}
        out.append({"id": r["id"], "condition": r["condition"], "seed": r["seed"],
                    "has_conflict": r.get("has_conflict"), "conflict_type": r.get("conflict_type"),
                    "correct": v.get("correct"), "handles_conflict": v.get("handles_conflict"),
                    "reason": v.get("reason", ""), "judge_model": getattr(llm, "model", "?"),
                    "cost_usd": res.cost_usd, "prompt_tokens": res.prompt_tokens,
                    "completion_tokens": res.completion_tokens})
    return out


def cohen_kappa(a: list, b: list) -> float | None:
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return None
    cats = sorted({v for p in pairs for v in p})
    n = len(pairs)
    po = sum(x == y for x, y in pairs) / n
    pe = sum((sum(x == c for x, _ in pairs) / n) * (sum(y == c for _, y in pairs) / n) for c in cats)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def _int_or_none(v):
    v = str(v).strip().lower()
    return None if v in ("", "none", "null", "na") else int(float(v))


def agreement(run: Path, csv_path: Path) -> dict:
    j = {(r["id"], r["condition"], r["seed"]): r for r in map(json.loads, open(run / "judgments.jsonl"))}
    rep = {}
    hum = list(csv.DictReader(open(csv_path)))
    for crit in ("correct", "handles_conflict"):
        a, b = [], []
        for h in hum:
            k = (h["id"], h["condition"], int(h["seed"]))
            if k in j and h.get(f"human_{crit}", "").strip():
                a.append(_int_or_none(h[f"human_{crit}"])); b.append(j[k][crit])
        rep[crit] = {"n": sum(x is not None and y is not None for x, y in zip(a, b)), "kappa": cohen_kappa(a, b)}
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--price-input", type=float, default=None)
    ap.add_argument("--price-output", type=float, default=None)
    ap.add_argument("--per-stratum", type=int, default=20)
    ap.add_argument("--human", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--agreement", default=None, help="filled human_labels.csv: report kappa and exit")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    run = Path(args.run_dir)
    if args.agreement:
        rep = agreement(run, Path(args.agreement))
        (run / "judge_agreement.json").write_text(json.dumps(rep, indent=1))
        print(json.dumps(rep, indent=1))
        return
    cfg = json.loads((run / "config.json").read_text())
    if not args.dry_run and args.model in (None, cfg.get("model")):
        raise SystemExit("--model must name a judge from a different family than the generator "
                         f"({cfg.get('model')})")
    kw = {"label_fix": cfg.get("ramdocs_labels", "exclude")} if cfg["dataset"] == "ramdocs" else {}
    exs = {e.id: e for e in data.filter_ids(data.load(cfg["dataset"], n=cfg.get("n"),
                                                       seed=cfg.get("sample_seed", 0), **kw), cfg.get("ids_file"))}
    rows = read_results(run)
    if args.dry_run:
        from .llm import FakeLLM
        llm = FakeLLM(lambda p: '{"correct": 1, "handles_conflict": null, "reason": "dry run"}')
    else:
        from .llm import OpenAILLM, register_price
        if args.price_input is not None:
            register_price(args.model, args.price_input, args.price_output)
        llm = OpenAILLM(args.model, temperature=0.0, base_url=args.base_url, api_key_env=args.api_key_env)
    picked = _sample(rows, args.per_stratum, args.seed)
    js = judge_rows(picked, exs, llm, args.seed)
    with open(run / "judgments.jsonl", "w") as f:
        f.writelines(json.dumps(x) + "\n" for x in js)
    rng = random.Random(args.seed)
    hum = rng.sample(picked, min(args.human, len(picked)))
    with open(run / "human_labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "condition", "seed", "question", "reference", "response",
                                          "human_correct", "human_handles_conflict", "notes"])
        w.writeheader()
        for r in hum:
            e = exs[r["id"]]
            w.writerow({"id": r["id"], "condition": r["condition"], "seed": r["seed"], "question": e.question,
                        "reference": " | ".join(e.gold_answers), "response": r["response"],
                        "human_correct": "", "human_handles_conflict": "", "notes": ""})
    print(f"judged {len(js)} results (cost ${sum(x['cost_usd'] for x in js):.4f}) -> {run/'judgments.jsonl'}; "
          f"{len(hum)} for human labels -> {run/'human_labels.csv'}")


if __name__ == "__main__":
    main()
