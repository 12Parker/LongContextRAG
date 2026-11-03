"""
Build the RAGuard evaluation set (PREREGISTRATION.md §14) from the pinned release.

    # Hugging Face dataset UCSC-IRKM/RAGuard @ dae344a2edc90373dd22e563c6e26e25cbbfc6ce (MIT on the card)
    # claims.csv and documents.csv into data/external/raguard/
    python -m experiments.conflict.prepare_raguard

RAGuard (Zeng et al., NeurIPS 2025) pairs PolitiFact claims (binary verdict) with Reddit posts that an
LLM labelled supporting (pushes toward the correct verdict), misleading (toward the wrong one) or unrelated.
Unlike RAMDocs and ConflictBank, nothing here is written for the benchmark: the misleading side is real
online discussion. A conflict item is a claim with at least one usable supporting and one usable misleading
post; a clean item has supporting posts and no misleading one in the release. Usable: at least MIN_WORDS words
after dropping the "[Link Post]" placeholder, deduplicated within the claim. Per item we keep up to
N_SIDE supporting, N_SIDE misleading and N_NOISE unrelated posts, each cut to MAX_WORDS words, in an order
shuffled by claim id. Clean items are a seed-0 sample of N_CLEAN.

Writes data/external/raguard/raguard_eval.jsonl (not committed) and the item ids to
experiments/conflict/splits/raguard_eval.txt.
"""
from __future__ import annotations

import ast
import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/external/raguard"
OUT = SRC / "raguard_eval.jsonl"
IDS = Path(__file__).resolve().parent / "splits/raguard_eval.txt"
MIN_WORDS, MAX_WORDS, N_SIDE, N_NOISE, N_CLEAN = 15, 300, 2, 2, 100


def _text(doc: dict) -> str:
    body = doc["Full Text"].replace("[Link Post]", "").strip()
    return f"{doc['Title'].strip()}\n{body}".strip()


def main():
    csv.field_size_limit(10 ** 8)
    claims = list(csv.DictReader(open(SRC / "claims.csv", encoding="utf-8")))
    docs = {d["Document ID"]: d for d in csv.DictReader(open(SRC / "documents.csv", encoding="utf-8"))}
    conflict, clean_pool = [], []
    for c in claims:
        ids, labs = ast.literal_eval(c["Document IDs"]), ast.literal_eval(c["Document Labels"])
        by, seen = {"supporting": [], "misleading": [], "unrelated": []}, set()
        for i, lab in zip(ids, labs):
            t = _text(docs[str(i)])
            if len(t.split()) < MIN_WORDS or t in seen:
                continue
            seen.add(t)
            by[lab].append(" ".join(t.split()[:MAX_WORDS]))
        item = (c, by)
        if by["supporting"] and by["misleading"]:
            conflict.append(item)
        elif by["supporting"] and "misleading" not in labs:
            clean_pool.append(item)
    clean = random.Random(0).sample(clean_pool, min(N_CLEAN, len(clean_pool)))
    out = []
    for c, by in conflict + clean:
        rng = random.Random(int(c["Claim ID"]))
        docs_ = [(t, "correct") for t in by["supporting"][:N_SIDE]] + \
                [(t, "misinfo") for t in by["misleading"][:N_SIDE]] + \
                [(t, "noise") for t in by["unrelated"][:N_NOISE]]
        rng.shuffle(docs_)
        gold = c["Verdict"].strip()
        out.append({"id": f"rg-{c['Claim ID']}", "question": f"Is the following claim true or false?\nClaim: {c['Claim']}",
                    "gold": gold, "wrong": "False" if gold == "True" else "True",
                    "passages": [t for t, _ in docs_], "types": [k for _, k in docs_],
                    "has_conflict": bool(by["misleading"]), "original_verdict": c["Original Verdict"]})
    OUT.write_text("".join(json.dumps(o) + "\n" for o in out))
    IDS.write_text("".join(o["id"] + "\n" for o in out))
    print(f"{len(out)} items: {len(conflict)} conflict, {len(clean)} clean (pool {len(clean_pool)}); "
          f"mean passages {sum(len(o['passages']) for o in out) / len(out):.1f} -> {OUT}")


if __name__ == "__main__":
    main()
