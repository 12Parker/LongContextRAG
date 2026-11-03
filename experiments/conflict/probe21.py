"""
Artefact probes on ConflictBank (PREREGISTRATION.md §21.1, C1): answer-only and single-passage.

    python -m experiments.conflict.probe21 --part answer_only|single_passage|all --models mini,haiku,luna,terra [--dry-run]

answer_only: no passage; the question and the two candidate answers (gold and substituted object) as A and B, in
  both orders, 300 conflict questions x 2 calls per model.
single_passage: one passage alone (first 1,500 characters, as the detector sees it) with the question; the model
  gives the probability (0-100) that what it says about the question is false. Correct and misinformation passages
  of the 300 conflict questions and the correct passage of the 100 clean ones: 700 calls per model.
Seed 0, requested temperature 0 (Claude and the reasoning models run at their default). API token counts and cost
are logged per row. Writes runs/probe21/{answer_only,single_passage}.<model>.jsonl; resumes, skipping done rows.
The object swap (§21.1c) runs through run.py --dataset conflictbank_swap --detect-only.
"""
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from . import data

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/probe21"
IDS = ROOT / "experiments/conflict/splits/conflictbank_eval.txt"
MODELS = {"mini": "gpt-4o-mini", "haiku": "claude-haiku-5-5", "luna": "gpt-6-luna", "terra": "gpt-6.1-terra"}
MAX_CHARS = 1500

ANSWER_ONLY_PROMPT = """Question: {question}

Two candidate answers:
A. {a}
B. {b}

Exactly one of them is correct. Which one? End with one line in exactly this format:
Answer: <A or B>"""

SINGLE_PASSAGE_PROMPT = """Below is a passage retrieved for a question. Judge how likely it is that what the passage says \
about the question is false (misinformation), using the passage and anything you know.

Question: {question}

Passage:
<<<
{passage}
>>>

End with one line in exactly this format:
Probability false: <integer from 0 to 100>"""


def parse_choice(text: str) -> str | None:
    m = re.findall(r"Answer:\s*\**\s*\(?([AB])\b", text or "")
    return m[-1] if m else None


def parse_prob(text: str) -> int | None:
    m = re.findall(r"Probability false:\s*\**\s*(\d{1,3})", text or "", re.I)
    return min(100, int(m[-1])) if m else None


def answer_only_jobs(examples):
    for e in examples:
        if not e.has_conflict:
            continue
        g, w = e.gold_answers[0], e.wrong_answers[0]
        for order, (a, b) in (("gold_first", (g, w)), ("wrong_first", (w, g))):
            yield ({"id": e.id, "order": order, "gold_letter": "A" if order == "gold_first" else "B"},
                   ANSWER_ONLY_PROMPT.format(question=e.question, a=a, b=b))


def single_passage_jobs(examples):
    for e in examples:
        for k, t in enumerate(e.meta["doc_types"]):
            if t in ("correct", "misinfo"):
                yield ({"id": e.id, "passage": k, "type": t, "has_conflict": e.has_conflict},
                       SINGLE_PASSAGE_PROMPT.format(question=e.question, passage=e.passages[k][:MAX_CHARS]))


def run_part(part: str, model: str, llm, examples, workers: int):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{part}.{model}.jsonl"
    keyf = (lambda m: (m["id"], m["order"])) if part == "answer_only" else (lambda m: (m["id"], m["passage"]))
    done = {keyf(json.loads(l)) for l in open(path)} if path.exists() else set()
    jobs = [j for j in (answer_only_jobs if part == "answer_only" else single_passage_jobs)(examples)
            if keyf(j[0]) not in done]

    def work(job):
        meta, prompt = job
        r = llm(prompt, seed=0)
        out = {**meta, "model": llm.model, "raw": r.text, "prompt_tokens": r.prompt_tokens,
               "completion_tokens": r.completion_tokens, "cost_usd": r.cost_usd, "cached": r.cached,
               "served_model": getattr(r, "served_model", None)}
        if part == "answer_only":
            c = parse_choice(r.text)
            out.update(choice=c, correct=None if c is None else c == meta["gold_letter"])
        else:
            out["prob_false"] = parse_prob(r.text)
        return out

    with open(path, "a") as f, ThreadPoolExecutor(workers) as pool:
        for row in tqdm(pool.map(work, jobs), total=len(jobs), desc=f"{part}[{model}]"):
            f.write(json.dumps(row) + "\n")
            f.flush()
    return path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=("answer_only", "single_passage", "all"))
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true", help="FakeLLM; no API calls; writes under runs/probe21_dry")
    args = ap.parse_args(argv)
    global OUT
    examples = data.filter_ids(data.load("conflictbank"), IDS)
    parts = ("answer_only", "single_passage") if args.part == "all" else (args.part,)
    for short in args.models.split(","):
        model = MODELS[short]
        if args.dry_run:
            from .llm import FakeLLM
            OUT = ROOT / "runs/probe21_dry"
            llm = FakeLLM(lambda p: "Answer: A" if "candidate answers" in p else "Probability false: 50")
            llm.model = model
        else:
            from .llm import NO_TEMPERATURE_MODELS, OpenAILLM
            llm = OpenAILLM(model, temperature=0.0)
        workers = min(args.workers, 6) if (not args.dry_run and model in NO_TEMPERATURE_MODELS) else args.workers
        for part in parts:
            print(f"{part}[{short}] -> {run_part(part, short, llm, examples, workers)}")


if __name__ == "__main__":
    main()
