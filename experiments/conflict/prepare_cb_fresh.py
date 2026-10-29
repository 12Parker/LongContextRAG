"""
Build a fresh ConflictBank evaluation set for PREREGISTRATION.md §17, disjoint from the §10 set.

    python -m experiments.conflict.prepare_cb_fresh

Same construction as prepare_conflictbank.py (300 conflict + 100 clean items, round-robin over relations, two
or three distractors from other sampled rows, passage order shuffled per item), with sampling seed 1. Every
subject that the §10 build used, as an item or as a distractor source, is excluded, so no fact or passage
recurs. `pick` with seed 0 and no exclusions reproduces the §10 selection; main() checks that before
building, so the exclusion list is exactly the subjects the §10 set used.

Writes data/external/conflictbank/conflictbank_fresh.jsonl (not committed) and the item ids to
experiments/conflict/splits/conflictbank_fresh.txt.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from .prepare_conflictbank import N_CLEAN, N_CONFLICT, N_DISTRACT, SRC

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/external/conflictbank/conflictbank_fresh.jsonl"
IDS = Path(__file__).resolve().parent / "splits/conflictbank_fresh.txt"
OLD_IDS = Path(__file__).resolve().parent / "splits/conflictbank_eval.txt"
SEED = 1


def usable_rows():
    rows = []
    with open(SRC) as f:
        for idx, line in enumerate(f):
            r = json.loads(line)
            ok = all(isinstance(r.get(k), str) and r[k].strip() for k in
                     ("question", "object", "replaced_object", "default_evidence", "misinformation_conflict_evidence"))
            if ok and r["object"].strip().lower() != r["replaced_object"].strip().lower():
                rows.append((idx, r))
    return rows


def pick(rows, seed: int, exclude_subjects=frozenset()):
    """prepare_conflictbank.main's selection, with the seed and excluded subjects as parameters."""
    rng = random.Random(seed)
    by_rel = defaultdict(list)
    for idx, r in rows:
        if r["subject"] not in exclude_subjects:
            by_rel[r["relation"]].append((idx, r))
    for v in by_rel.values():
        rng.shuffle(v)
    rels = sorted(by_rel)
    rng.shuffle(rels)
    picked, seen_subj = [], set()
    while len(picked) < N_CONFLICT + N_CLEAN + N_CONFLICT:
        for rel in rels:
            while by_rel[rel]:
                idx, r = by_rel[rel].pop()
                if r["subject"] not in seen_subj:
                    seen_subj.add(r["subject"]); picked.append((idx, r))
                    break
            if len(picked) >= 2 * N_CONFLICT + N_CLEAN:
                break
    return picked[:N_CONFLICT + N_CLEAN], picked[N_CONFLICT + N_CLEAN:]


def main():
    rows = usable_rows()
    old_items, old_pool = pick(rows, 0)
    old_ids = [l.strip() for l in open(OLD_IDS) if l.strip()]
    assert [f"cb-{i}" for i, _ in old_items] == old_ids, "pick(seed=0) no longer reproduces the §10 set"
    used = {r["subject"] for _, r in old_items + old_pool}
    items, pool = pick(rows, SEED, frozenset(used))
    assert not {r["subject"] for _, r in items + pool} & used
    out = []
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
    OUT.write_text("".join(json.dumps(o) + "\n" for o in out))
    IDS.write_text("".join(o["id"] + "\n" for o in out))
    print(f"{len(out)} items ({N_CONFLICT} conflict, {N_CLEAN} clean), {len({o['relation'] for o in out})} relations, "
          f"{len(used)} subjects excluded -> {OUT}")


if __name__ == "__main__":
    main()
