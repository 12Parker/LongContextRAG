"""
Build the ConflictBank evaluation set (PREREGISTRATION.md §10) from the pinned QA file.

    curl -L -o data/external/conflictbank/QA_dataset.json \
      https://huggingface.co/datasets/Warrieryes/CB_qa_v2/resolve/92866a09920a4234cc10a497de8084c16725a111/QA_dataset.json
    python -m experiments.conflict.prepare_conflictbank

ConflictBank (Su et al., NeurIPS 2024 D&B) gives, per Wikidata fact, an LLM-written evidence passage
for the true object (`default_evidence`) and one for a substituted object
(`misinformation_conflict_evidence`). RAMDocs edits real documents; here both sides are generated, and
every question has exactly one correct answer. We sample N_CONFLICT + N_CLEAN rows (seed 0,
round-robin over relations so no relation dominates). A conflict item gets the correct passage, the
misinformation passage and two distractors; a clean item gets the correct passage and three
distractors. Distractors are other sampled rows' correct passages (a different subject), so they are
fluent, on-topic-looking noise. Passage order is shuffled per item with a fixed seed.

Also writes conflictbank_majority.jsonl (PREREGISTRATION.md §11): the same 400 items, but each conflict item
gets the correct passage and all three of ConflictBank's wrong-object passages for that fact
(misinformation, temporal and semantic conflict evidence, which all support the substituted object) and no
distractors, so the misinformation is the majority (3 to 1). Clean items are unchanged.

Writes data/external/conflictbank/conflictbank_eval.jsonl (not committed: the source licence is
stated as CC BY-SA 4.0 in the repo README but no licence file exists) and the item ids to
experiments/conflict/splits/conflictbank_eval.txt.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/external/conflictbank/QA_dataset.json"
OUT = ROOT / "data/external/conflictbank/conflictbank_eval.jsonl"
OUT_MAJ = ROOT / "data/external/conflictbank/conflictbank_majority.jsonl"
WRONG_SIDE = ("misinformation_conflict_evidence", "temporal_conflict_evidence", "semantic_conflict_evidence")
IDS = Path(__file__).resolve().parent / "splits/conflictbank_eval.txt"
N_CONFLICT, N_CLEAN, N_DISTRACT = 300, 100, 2


def main():
    rows = []
    with open(SRC) as f:
        for idx, line in enumerate(f):
            r = json.loads(line)
            ok = all(isinstance(r.get(k), str) and r[k].strip() for k in
                     ("question", "object", "replaced_object", "default_evidence", "misinformation_conflict_evidence"))
            if ok and r["object"].strip().lower() != r["replaced_object"].strip().lower():
                rows.append((idx, r))
    rng = random.Random(0)
    by_rel = defaultdict(list)
    for idx, r in rows:
        by_rel[r["relation"]].append((idx, r))
    for v in by_rel.values():
        rng.shuffle(v)
    rels = sorted(by_rel)
    rng.shuffle(rels)
    picked, seen_subj = [], set()
    while len(picked) < N_CONFLICT + N_CLEAN + N_CONFLICT:   # extra rows serve as distractor sources
        for rel in rels:
            while by_rel[rel]:
                idx, r = by_rel[rel].pop()
                if r["subject"] not in seen_subj:
                    seen_subj.add(r["subject"]); picked.append((idx, r))
                    break
            if len(picked) >= 2 * N_CONFLICT + N_CLEAN:
                break
    items, pool = picked[:N_CONFLICT + N_CLEAN], picked[N_CONFLICT + N_CLEAN:]
    out, maj = [], []
    for n, (idx, r) in enumerate(items):
        conflict = n < N_CONFLICT
        k = N_DISTRACT if conflict else N_DISTRACT + 1
        distract = [pool[(n * 3 + j) % len(pool)][1]["default_evidence"] for j in range(k)]
        docs = [(r["default_evidence"], "correct")] + ([(r["misinformation_conflict_evidence"], "misinfo")] if conflict else [])
        docs += [(d, "noise") for d in distract]
        random.Random(idx).shuffle(docs)
        out.append({"id": f"cb-{idx}", "relation": r["relation"], "question": r["question"],
                    "gold": r["object"], "wrong": r["replaced_object"] if conflict else None,
                    "passages": [d for d, _ in docs], "types": [t for _, t in docs], "has_conflict": conflict})
        if conflict and all(isinstance(r.get(k), str) and r[k].strip() for k in WRONG_SIDE):
            mdocs = [(r["default_evidence"], "correct")] + [(r[k], "misinfo") for k in WRONG_SIDE]
            random.Random(idx).shuffle(mdocs)
            maj.append({**out[-1], "passages": [d for d, _ in mdocs], "types": [t for _, t in mdocs]})
        else:
            maj.append(out[-1])
    OUT.write_text("".join(json.dumps(o) + "\n" for o in out))
    OUT_MAJ.write_text("".join(json.dumps(o) + "\n" for o in maj))
    print(f"majority variant: {sum(o['types'].count('misinfo') == 3 for o in maj)} items with 3 wrong-side passages")
    IDS.write_text("".join(o["id"] + "\n" for o in out))
    print(f"{len(out)} items ({N_CONFLICT} conflict, {N_CLEAN} clean) from {len(rows)} usable rows, "
          f"{len({o['relation'] for o in out})} relations -> {OUT}")


if __name__ == "__main__":
    main()
