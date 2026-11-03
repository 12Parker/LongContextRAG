"""
Build the §15 artefact-probe variants of the ConflictBank evaluation set (PREREGISTRATION.md §15).

    python -m experiments.conflict.prepare_cb_probe            # real API (gpt-4o-mini paraphrases)
    python -m experiments.conflict.prepare_cb_probe --dry-run  # identity "paraphrase", no API

conflictbank_rev.jsonl: every item with its passage order reversed (types reversed with it), so a detector
that keys on position is caught. conflictbank_para.jsonl: every passage, correct, misinformation and
distractor alike, rewritten by one model with one prompt, so style differences between ConflictBank's
default and conflict evidence are evened out. Each paraphrase is checked for whether the gold and wrong
answers still appear wherever they appeared in the original (`para_check`); failures are kept and counted.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .data import DATA_DIR

SRC = DATA_DIR / "conflictbank" / "conflictbank_eval.jsonl"
PARA_PROMPT = """Paraphrase the passage below. Keep every fact, name, number, date and place exactly as stated, \
including any that seem wrong; do not add, remove or correct information, and keep about the same length. \
Output only the paraphrased passage.

Passage:
{passage}"""


def reverse_item(r: dict) -> dict:
    return {**r, "passages": r["passages"][::-1], "types": r["types"][::-1]}


def check(orig: str, para: str, answers: list[str]) -> bool:
    """True if every answer string present in the original is still present in the paraphrase."""
    o, p = orig.lower(), para.lower()
    return all(a.lower() in p for a in answers if a and a.lower() in o)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    rows = [json.loads(l) for l in open(SRC)]
    out_dir = SRC.parent
    with open(out_dir / "conflictbank_rev.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(reverse_item(r)) + "\n")
    if args.dry_run:
        para = lambda p: (p, 0.0)
    else:
        from .llm import OpenAILLM
        llm = OpenAILLM(args.model, temperature=0.0, max_tokens=1024)
        para = lambda p: (lambda r: (r.text.strip(), r.cost_usd))(llm(PARA_PROMPT.format(passage=p)))
    jobs = [(k, n, p) for k, r in enumerate(rows) for n, p in enumerate(r["passages"])]
    with ThreadPoolExecutor(args.workers) as pool:
        res = dict(zip([(k, n) for k, n, _ in jobs], pool.map(lambda j: para(j[2]), jobs)))
    cost, fails = sum(c for _, c in res.values()), 0
    with open(out_dir / "conflictbank_para.jsonl", "w") as f:
        for k, r in enumerate(rows):
            ps = [res[(k, n)][0] for n in range(len(r["passages"]))]
            ok = [check(o, p, [r["gold"], r["wrong"]]) for o, p in zip(r["passages"], ps)]
            fails += ok.count(False)
            f.write(json.dumps({**r, "passages": ps, "para_check": ok}) + "\n")
    print(f"{len(rows)} items, {len(jobs)} passages paraphrased, {fails} failed the answer check, ${cost:.3f}")


if __name__ == "__main__":
    main()
