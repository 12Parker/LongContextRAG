"""
Semantic re-scoring with an outside-family judge (PREREGISTRATION.md §17.3, exploratory).

    python -m experiments.conflict.judge8 [--part contrasts|wiki|raguard|all] [--n 120] [--dry-run]

contrasts: for each registered contrast, 120 conflict questions (seed 0, fixed sample) judged under both of its
  conditions; the judge reads the question, the gold and wrong answers and the response, and classifies the final
  answer (gives gold, asserts wrong, rejects wrong, abstains, adds another answer, acknowledges a conflict).
wiki: WikiContradict under its original rubric (Hou et al. 2024, §3.4): both answers, the incompatibility
  acknowledged, no unsupported preference. Plain vs LLM directional flag, seed 0, all questions.
raguard: 150 sampled supporting x misleading post pairs; does the pair contradict on the claim, are the posts
  temporally compatible, and is the misleading post false about the claim.
Writes runs/judge8/{contrasts,wiki,raguard}.jsonl. Judge: gemini-3.5-flash-lite, temperature 0, thinking off.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .jsonl import read_results

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/judge8"
JUDGE = "gemini-3.5-flash-lite"   # §17.3 named 2.5 Flash, closed to new users; same price
RAND = "synth-p1-r1-all-d0.5+flag_directional"
# name: (dataset, (run, condition) for arm a, (run, condition) for arm b)
CONTRASTS = {
    "neutral|RD|gpt-4o-mini": ("ramdocs", ("holdout_main", "plain"), ("holdout_main", "oracle+flag")),
    "neutral|CBs|gpt-4o-mini": ("conflictbank", ("cb_single", "plain"), ("cb_single", "oracle+flag")),
    "neutral|CBs|claude-haiku-5-5": ("conflictbank", ("cb_single_haiku", "plain"), ("cb_single_haiku", "oracle+flag")),
    "H23|RD|gpt-4o-mini": ("ramdocs", ("ctl_rd_mini", "answer_key"), ("holdout_main", "oracle+flag_directional")),
    "H23|CBs|gpt-4o-mini": ("conflictbank", ("ctl_cb_mini", "answer_key"), ("cb_single", "oracle+flag_directional")),
    "H23|CBs|gpt-6-luna": ("conflictbank", ("ctl_cb_luna", "answer_key"), ("cb_single_luna", "oracle+flag_directional")),
    "H30|CBs|gpt-4o-mini": ("conflictbank", ("vkey_cb_mini_mini", "llm_dir+verdict_key"), ("cb_single", "llm_dir+flag_directional")),
    "H30|CBs|gpt-6-luna": ("conflictbank", ("vkey_cb_luna_mini", "llm_dir+verdict_key"), ("cb_single_luna", "llm_dir+flag_directional")),
    "H31|CBs|gpt-4o-mini": ("conflictbank", ("randdir_cb_mini", "plain"), ("randdir_cb_mini", RAND)),
    "H31|CBs|claude-haiku-5-5": ("conflictbank", ("randdir_cb_haiku", "plain"), ("randdir_cb_haiku", RAND)),
}
WIKI_RUNS = {"gpt-4o-mini": "wiki_llm_mini", "gpt-6-luna": "wiki_llm_luna", "claude-haiku-5-5": "wiki_llm_haiku"}

ANSWER_PROMPT = """You are grading a model's answer to a question. Read the question, the reference answer(s), the
known wrong (misinformation) answer(s), and the model's full response. Judge what the response finally asserts.

Question: {question}
Reference answer(s): {gold}
Known wrong answer(s): {wrong}

Model response:
<<<
{response}
>>>

Reply with JSON only:
{{"gives_gold": true/false, "asserts_wrong": true/false, "rejects_wrong": true/false, "abstains": true/false,
  "other_answer": true/false, "acknowledges_conflict": true/false}}
- gives_gold: the final answer states every reference answer (paraphrases, aliases and equivalent forms count).
- asserts_wrong: the final answer states a known wrong answer as true. Mentioning it only to reject it does not count.
- rejects_wrong: the response explicitly says a known wrong answer is false or not the answer.
- abstains: the final answer is "unknown", or declines or fails to give an answer.
- other_answer: the final answer also asserts an answer that is in neither list.
- acknowledges_conflict: the response says that the sources disagree."""

WIKI_PROMPT = """A question has two conflicting answers, each supported by a different passage of the same Wikipedia
article. Read the question, the two answers and the model's response.

Question: {question}
Answer 1: {a1}
Answer 2: {a2}

Model response:
<<<
{response}
>>>

Reply with JSON only:
{{"mentions_1": true/false, "mentions_2": true/false, "acknowledges_conflict": true/false, "unsupported_preference": true/false}}
- mentions_1 / mentions_2: the response gives that answer (paraphrases count).
- acknowledges_conflict: the response says the sources conflict or that the answers are incompatible.
- unsupported_preference: the response picks one answer as right without evidence from the passages for doing so."""

RAGUARD_PROMPT = """Two Reddit posts were retrieved for a claim. A benchmark labels Post A as supporting the correct
verdict on the claim and Post B as misleading. Judge the posts yourself.

Claim: {claim}
Correct verdict (per the benchmark): {verdict}

Post A:
<<<
{a}
>>>

Post B:
<<<
{b}
>>>

Reply with JSON only:
{{"contradict": true/false, "temporally_compatible": true/false, "b_false_about_claim": "yes"|"no"|"unclear",
  "a_supports_verdict": "yes"|"no"|"unclear"}}
- contradict: the posts cannot both be right about the claim.
- temporally_compatible: the posts could both be right because they describe different times.
- b_false_about_claim: Post B asserts or implies something false about the claim.
- a_supports_verdict: Post A supports the stated correct verdict."""


def _json(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}


def _examples(name):
    from . import data
    return {e.id: e for e in data.load(name)}


def _rows(run, cond, seed=0):
    return {r["id"]: r for r in read_results(ROOT / "runs" / run)
            if r["condition"] == cond and r["seed"] == seed and r.get("has_conflict")}


def contrasts_jobs(n):
    jobs, exs = [], {}
    for name, (ds, (ra, ca), (rb, cb)) in CONTRASTS.items():
        ex = exs.setdefault(ds, _examples(ds))
        A, B = _rows(ra, ca), _rows(rb, cb)
        ids = sorted(set(A) & set(B) & set(ex))
        ids = random.Random(f"judge8|{name}").sample(ids, min(n, len(ids)))
        for i in ids:
            for arm, r in (("a", A[i]), ("b", B[i])):
                e = ex[i]
                jobs.append(({"contrast": name, "arm": arm, "id": i, "condition": r["condition"],
                              "matcher": r["ramdocs_strict_oneway"], "wrong_rate": r["wrong_rate"]},
                             ANSWER_PROMPT.format(question=e.question, gold=" | ".join(e.gold_answers),
                                                  wrong=" | ".join(e.wrong_answers), response=r["response"][-3000:])))
    return jobs


def wiki_jobs():
    ex = _examples("wikicontradict"); jobs = []
    for g, run in WIKI_RUNS.items():
        for cond in ("plain", "llm_dir+flag_directional"):
            for i, r in sorted(_rows(run, cond).items()):
                e = ex.get(i)
                if e is None or len(e.gold_answers) < 2:
                    continue
                jobs.append(({"gen": g, "condition": cond, "id": i, "matcher": r.get("primary")},
                             WIKI_PROMPT.format(question=e.question, a1=e.gold_answers[0], a2=e.gold_answers[1],
                                                response=r["response"][-3000:])))
    return jobs


def raguard_jobs(n=150):
    ex = _examples("raguard"); pairs = []
    for i, e in sorted(ex.items()):
        t = e.meta["doc_types"]
        pairs += [(i, a, b) for a in range(len(t)) if t[a] == "correct" for b in range(len(t)) if t[b] == "misinfo"]
    pairs = random.Random("judge8|raguard").sample(pairs, min(n, len(pairs)))
    claim = lambda q: q.split("Claim:", 1)[-1].strip()
    return [({"id": i, "a": a, "b": b},
             RAGUARD_PROMPT.format(claim=claim(ex[i].question), verdict=ex[i].gold_answers[0],
                                   a=ex[i].passages[a][:2500], b=ex[i].passages[b][:2500])) for i, a, b in pairs]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=("contrasts", "wiki", "raguard", "all"))
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    parts = ("contrasts", "wiki", "raguard") if args.part == "all" else (args.part,)
    if args.dry_run:
        from .llm import FakeLLM
        judge = FakeLLM(lambda p: "{}")
    else:
        from .llm import OpenAILLM
        judge = OpenAILLM(JUDGE, temperature=0.0, max_tokens=256)
    OUT.mkdir(parents=True, exist_ok=True)
    for part in parts:
        jobs = {"contrasts": lambda: contrasts_jobs(args.n), "wiki": wiki_jobs, "raguard": raguard_jobs}[part]()

        def one(job):
            meta, prompt = job
            r = judge(prompt, seed=0, json_mode=True)
            return {**meta, "judge": _json(r.text), "cost_usd": r.cost_usd,
                    "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens}
        with ThreadPoolExecutor(args.workers) as pool:
            out = list(pool.map(one, jobs))
        path = (OUT if not args.dry_run else OUT / "dry") / f"{part}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(o) + "\n" for o in out))
        print(f"{part}: {len(out)} judged, ${sum(o['cost_usd'] for o in out):.3f}, "
              f"unparsed {sum(not o['judge'] for o in out)}")


if __name__ == "__main__":
    main()
