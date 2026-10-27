"""
Run the condition matrix (Phase 2).

    python -m experiments.conflict.run --dataset ramdocs --n 200 \
        --conditions plain,instruct,caag+flag,nli+flag,llm+flag,random+flag,oracle+flag,madam \
        --seeds 0,1,2 --out runs/ramdocs_pilot

A condition is "<detector>+<mode>" or one of: plain, instruct, madam.
Detectors: caag, nli, llm, random, oracle. Modes: see prompts.py.

Output (append-only, safe to interrupt and rerun with the same --out):
    <out>/config.json
    <out>/detections.<detector>.jsonl   one line per example
    <out>/results.jsonl                 one line per (example, condition, seed)
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from . import data
from .detectors import (CAAGDetector, DetectorOutput, Detection, LLMDetector, NLIDetector,
                        NoDetector, OracleDetector, RandomDetector, STEmbedder)
from .madam import madam_rag
from .metrics import score
from .prompts import build_prompt


def parse_condition(c: str) -> tuple[str, str]:
    if c in ("plain", "instruct"):
        return "none", c
    if c == "madam":
        return "none", "madam"
    det, mode = c.split("+")
    return det, mode


def make_detector(name: str, args, llm):
    if name == "none":
        return NoDetector()
    if name == "caag":
        return CAAGDetector(STEmbedder(args.embedder), theta_s=args.theta_s, theta_c=args.theta_c)
    if name == "nli":
        return NLIDetector(args.nli_model, threshold=args.nli_threshold)
    if name == "llm":
        return LLMDetector(llm)
    if name == "random":
        if args.random_rate is None:
            raise SystemExit("--random-rate is required for the random detector "
                             "(use CAAG's per-pair flag rate from detect_eval).")
        return RandomDetector(args.random_rate)
    if name == "oracle":
        return OracleDetector()
    raise SystemExit(f"unknown detector {name}")


def _read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p)] if p.exists() else []


def run_detection(det, examples, out: Path) -> dict[str, DetectorOutput]:
    path = out / f"detections.{det.name}.jsonl"
    done = {r["id"]: r for r in _read_jsonl(path)}
    res = {}
    with open(path, "a") as f:
        for ex in tqdm(examples, desc=f"detect[{det.name}]"):
            if ex.id in done:
                r = done[ex.id]
                res[ex.id] = DetectorOutput([Detection(**d) for d in r["pairs"]], {},
                                            r["latency_s"], r["cost_usd"])
                continue
            o = det.detect(ex)
            res[ex.id] = o
            f.write(json.dumps({"id": ex.id, "pairs": [asdict(d) for d in o.pairs],
                                "scores": {f"{i},{j}": s for (i, j), s in o.scores.items()},
                                "latency_s": o.latency_s, "cost_usd": o.cost_usd,
                                "extra": o.extra}) + "\n")
            f.flush()
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--conditions", required=True)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--theta-s", type=float, default=0.65)
    ap.add_argument("--theta-c", type=float, default=0.35)
    ap.add_argument("--nli-model", default="cross-encoder/nli-deberta-v3-base")
    ap.add_argument("--nli-threshold", type=float, default=0.5)
    ap.add_argument("--random-rate", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true", help="FakeLLM; no API calls, no cost")
    args = ap.parse_args(argv)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=1))
    examples = data.load(args.dataset, n=args.n, seed=args.sample_seed)
    by_id = {e.id: e for e in examples}
    seeds = [int(s) for s in args.seeds.split(",")]
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]

    if args.dry_run:
        from .llm import FakeLLM
        llm = madam_llm = FakeLLM(lambda p: "All Correct Answers: [\"unknown\"]" if "aggregator" in p
                                  else "Answer: unknown. Explanation: dry run.\nFinal answer(s): unknown")
    else:
        from .llm import OpenAILLM
        llm = OpenAILLM(args.model, temperature=args.temperature)
        madam_llm = OpenAILLM(args.model, temperature=0.0, max_tokens=128)  # matches public MADAM script

    dets_needed = sorted({parse_condition(c)[0] for c in conditions})
    detections = {}
    for dn in dets_needed:
        detections[dn] = run_detection(make_detector(dn, args, llm), examples, out)

    res_path = out / "results.jsonl"
    done = {(r["id"], r["condition"], r["seed"]) for r in _read_jsonl(res_path)}
    jobs = [(ex.id, c, s) for c in conditions for s in seeds for ex in examples if (ex.id, c, s) not in done]

    def work(job):
        eid, cond, seed = job
        ex = by_id[eid]
        dname, mode = parse_condition(cond)
        dout = detections[dname][eid]
        t0 = time.perf_counter()
        if mode == "madam":
            m = madam_rag(ex.question, ex.passages, madam_llm, seed=seed)
            text, usage, kept = m["final"], m["usage"], list(range(len(ex.passages)))
        else:
            prompt, kept = build_prompt(ex.question, ex.passages, mode, dout.pairs)
            r = llm(prompt, seed=seed)
            text = r.text
            usage = {"prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
                     "cost_usd": r.cost_usd, "calls": 1, "cached": r.cached}
        gen_s = time.perf_counter() - t0
        return {"id": eid, "dataset": ex.dataset, "condition": cond, "seed": seed,
                "has_conflict": ex.has_conflict, "conflict_type": ex.conflict_type, "group": ex.group,
                "n_flags": len(dout.pairs), "kept": kept, "response": text,
                **score(ex, text), **usage,
                "detect_s": dout.latency_s, "detect_cost_usd": dout.cost_usd, "generate_s": gen_s}

    with open(res_path, "a") as f, ThreadPoolExecutor(args.workers) as pool:
        futs = [pool.submit(work, j) for j in jobs]
        for fu in tqdm(as_completed(futs), total=len(futs), desc="generate"):
            f.write(json.dumps(fu.result()) + "\n")
            f.flush()
    print(f"done: {len(jobs)} new results -> {res_path}")


if __name__ == "__main__":
    main()
