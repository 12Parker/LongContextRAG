"""
Outside-family judge for PREREGISTRATION.md §19 (exploratory scoring; the judge, prompt and judge-strict score of §17.3).

    python -m experiments.conflict.judge10 [--part pooled|hc_labels|all] [--dry-run]

pooled: every conflict response of the conditions in the pooled H30a / H23 tests (§19.2), judged once per distinct
  (dataset, question, response). Rows are matched back by `response_key`.
hc_labels: per HealthContradict item, which answer each page supports (§19.3 label audit).
Writes runs/judge10/{pooled,hc_labels}.jsonl. Resumes: rows already judged are skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .jsonl import read_results
from .judge8 import ANSWER_PROMPT, JUDGE, _json

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/judge10"
# run family -> conditions that feed the §16.4 pooled H30a / H23 tests
FAMILIES = ((r"^vkey_cb_", {"llm_dir+verdict_key"}), (r"^cross_cb_", {"llm_dir+flag_directional"}),
            (r"^cb_single", {"llm_dir+flag_directional", "oracle+flag_directional"}), (r"^ctl_(rd|cb|mj)_", {"answer_key"}),
            (r"^holdout_", {"oracle+flag_directional"}), (r"^cbmaj_single", {"oracle+flag_directional"}))

HC_PROMPT = """A yes/no health question and two web pages retrieved for it. For each page, say which answer to the
question the page supports.

Question: {question}

Page 1:
<<<
{p1}
>>>

Page 2:
<<<
{p2}
>>>

Reply with JSON only: {{"page_1": "yes"|"no"|"unclear", "page_2": "yes"|"no"|"unclear"}}"""


def response_key(text: str) -> str:
    return hashlib.sha1((text or "").encode()).hexdigest()[:16]


def _examples(run: Path, cache: dict):
    from . import data
    cfg = json.loads((run / "config.json").read_text())
    kw = {"label_fix": cfg.get("ramdocs_labels", "none")} if cfg["dataset"] == "ramdocs" else {}
    k = (cfg["dataset"], tuple(sorted(kw.items())))
    if k not in cache:
        cache[k] = {e.id: e for e in data.load(cfg["dataset"], **kw)}
    return cache[k]


def pooled_jobs(runs: Path = ROOT / "runs", done=frozenset()):
    jobs, seen, cache = [], set(done), {}
    for res in sorted(runs.glob("*/results.jsonl")):
        run = res.parent
        conds = set().union(*[c for p, c in FAMILIES if re.match(p, run.name)] or [set()])
        if not conds:
            continue
        ex = None
        for r in read_results(run):
            if r["condition"] not in conds or not r.get("has_conflict"):
                continue
            key = (r["dataset"], r["id"], response_key(r["response"]))
            if key in seen:
                continue
            seen.add(key)
            ex = ex or _examples(run, cache)
            e = ex[r["id"]]
            jobs.append(({"dataset": r["dataset"], "id": r["id"], "key": key[2], "run": run.name, "condition": r["condition"]},
                         ANSWER_PROMPT.format(question=e.question, gold=" | ".join(e.gold_answers),
                                              wrong=" | ".join(e.wrong_answers), response=r["response"][-3000:])))
    return jobs


def hc_jobs(done=frozenset()):
    from . import data
    jobs = []
    for e in data.filter_ids(data.load("healthcontradict"), str(ROOT / "experiments/conflict/splits/healthcontradict_eval.txt")):
        if e.id in done:
            continue
        jobs.append(({"id": e.id, "types": e.meta["doc_types"], "gold": e.gold_answers[0], "wrong": e.wrong_answers[0]},
                     HC_PROMPT.format(question=e.question, p1=e.passages[0], p2=e.passages[1])))
    return jobs


def hc_fields(meta, j):
    """Map page stances to the labels: does the page typed correct support the gold answer, the misinformation page
    the wrong one."""
    if not j:
        return {}
    st = [str(j.get("page_1", "")).lower(), str(j.get("page_2", "")).lower()]
    c, m = meta["types"].index("correct"), meta["types"].index("misinfo")
    yn = lambda s, want: "yes" if s == want else "unclear" if s not in ("yes", "no") else "no"
    return {**j, "correct_supports_gold": yn(st[c], meta["gold"]), "misinfo_supports_wrong": yn(st[m], meta["wrong"])}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=("pooled", "hc_labels", "all"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="judge at most this many new rows (smoke test)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if args.dry_run:
        from .llm import FakeLLM
        judge = FakeLLM(lambda p: '{"page_1": "yes", "page_2": "no"}' if "Page 1" in p else
                        '{"gives_gold": true, "asserts_wrong": false, "abstains": false}')
        out_dir = OUT / "dry"
    else:
        from .llm import OpenAILLM
        judge = OpenAILLM(JUDGE, temperature=0.0, max_tokens=256)
        out_dir = OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    for part in (("pooled", "hc_labels") if args.part == "all" else (args.part,)):
        path = out_dir / f"{part}.jsonl"
        prev = [json.loads(l) for l in open(path)] if path.exists() else []
        if part == "pooled":
            jobs = pooled_jobs(done={(x["dataset"], x["id"], x["key"]) for x in prev if x["judge"]})
        else:
            jobs = hc_jobs(done={x["id"] for x in prev if x["judge"]})
        prev = [x for x in prev if x["judge"]]   # unparsed rows are retried
        jobs = jobs[: args.limit] if args.limit else jobs

        def one(job):
            meta, prompt = job
            r = judge(prompt, seed=0, json_mode=True)
            j = _json(r.text)
            if part == "hc_labels":
                j = hc_fields(meta, j)
            return {**meta, "judge": j, "cost_usd": r.cost_usd,
                    "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens}
        path.write_text("".join(json.dumps(o) + "\n" for o in prev))
        new = []
        with ThreadPoolExecutor(args.workers) as pool, open(path, "a") as f:
            for o in pool.map(one, jobs):   # written as they arrive, so an interrupted run resumes
                new.append(o); f.write(json.dumps(o) + "\n"); f.flush()
        print(f"{part}: {len(new)} judged (${sum(o['cost_usd'] for o in new):.3f}), "
              f"unparsed {sum(not o['judge'] for o in new)}, total {len(prev) + len(new)}")


if __name__ == "__main__":
    main()
