"""
Phase 0: how often does CAAG flag passages from a single coherent story, and are the
flags real? No OpenAI calls; needs Hugging Face access for NarrativeQA + MiniLM.

    python -m experiments.conflict.phase0_audit --n 100 --out runs/phase0

Questions are sampled at random across stories (max 2 per story), fixing the original
loader that took the first rows of the split. Retrieval here is BM25 top-k over
1,500-char chunks with 200 overlap (CAAG's chunking). That isolates the detector from
the neural re-ranker; note the difference when you report it.

Outputs
  <out>/flags_to_label.csv  one row per flagged pair; fill in `label` with one of:
        true_contradiction | time_or_plot_change | dialogue_or_perspective |
        different_entities | no_relation
  <out>/audit_summary.json  flag rates and signal breakdown
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from rank_bm25 import BM25Okapi

from .data import Example, load_narrativeqa_questions
from .detectors import CAAGDetector, STEmbedder


def chunk(text: str, size: int = 1500, overlap: int = 200) -> list[str]:
    step = size - overlap
    return [text[i:i + size] for i in range(0, max(len(text) - overlap, 1), step)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--chunk-size", type=int, default=1500)
    ap.add_argument("--out", default="runs/phase0")
    ap.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args(argv)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    qs = load_narrativeqa_questions(args.n, seed=args.seed)
    det = CAAGDetector(STEmbedder(args.embedder))
    rows, sig, types, n_flagged_q, n_pairs_total = [], Counter(), Counter(), 0, 0
    bm25_cache = {}
    for q in qs:
        if q["story_id"] not in bm25_cache:
            ch = chunk(q["story"], args.chunk_size)
            bm25_cache[q["story_id"]] = (ch, BM25Okapi([c.lower().split() for c in ch]))
        ch, bm = bm25_cache[q["story_id"]]
        scores = bm.get_scores(q["question"].lower().split())
        top = sorted(range(len(ch)), key=lambda i: -scores[i])[: args.top_k]
        passages = [ch[i] for i in top]
        ex = Example(id=q["id"], dataset="narrativeqa", question=q["question"], passages=passages,
                     gold_answers=q["answers"], has_conflict=False, group=q["story_id"])
        o = det.detect(ex)
        n_pairs_total += len(passages) * (len(passages) - 1) // 2
        n_flagged_q += bool(o.pairs)
        for d in o.pairs:
            info = o.extra["caag_rows"][f"{d.i},{d.j}"]
            sig.update(info["signals"]); types[d.type] += 1
            rows.append({"question_id": q["id"], "story_id": q["story_id"], "question": q["question"],
                         "i": d.i, "j": d.j, "sim": round(info["sim"], 3), "score": round(d.score, 3),
                         "type": d.type, "signals": "+".join(info["signals"]),
                         "passage_i": passages[d.i], "passage_j": passages[d.j],
                         "label": "", "notes": ""})

    with open(out / "flags_to_label.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["question_id"])
        w.writeheader(); w.writerows(rows)
    summary = {
        "questions": len(qs), "stories": len({q["story_id"] for q in qs}),
        "questions_with_any_flag": n_flagged_q, "flagged_pairs": len(rows),
        "pair_flag_rate": len(rows) / max(n_pairs_total, 1),
        "signal_counts": dict(sig), "type_counts": dict(types),
        "retrieval": f"BM25 top-{args.top_k}, {args.chunk_size}-char chunks, 200 overlap",
    }
    (out / "audit_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print(f"\nLabel the pairs in {out/'flags_to_label.csv'} (column `label`).")


if __name__ == "__main__":
    main()
