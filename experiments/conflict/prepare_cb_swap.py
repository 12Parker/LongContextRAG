"""
Build the §21.1(c) object-swap variant of the ConflictBank evaluation set (PREREGISTRATION.md §21.1).

    python -m experiments.conflict.prepare_cb_swap

In each usable conflict item every whole-word, case-insensitive mention of the gold answer in the correct passage
becomes the substituted answer, and every mention of the substituted answer in the misinformation passage becomes
the gold answer: each passage keeps its own style but now argues the other side. Types are re-labelled by content
(the passage that now states the gold answer is `correct`); `style_types` keeps the original labels. An item is
usable only if each answer occurs, as a whole word, in its own passage, neither occurs in the other passage, and
neither occurs in either passage as the prefix of a longer word. Clean items are not written.

Writes data/external/conflictbank/conflictbank_swap.jsonl (not committed, as the eval set) and the usable ids to
experiments/conflict/splits/conflictbank_swap.txt.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .data import DATA_DIR

SRC = DATA_DIR / "conflictbank" / "conflictbank_eval.jsonl"
OUT = DATA_DIR / "conflictbank" / "conflictbank_swap.jsonl"
IDS = Path(__file__).resolve().parent / "splits/conflictbank_swap.txt"


def _word(s: str) -> re.Pattern:
    return re.compile(r"(?<!\w)" + re.escape(s) + r"(?!\w)", re.I)


def _prefix(s: str) -> re.Pattern:
    return re.compile(r"(?<!\w)" + re.escape(s) + r"\w", re.I)


def usable(correct: str, misinfo: str, gold: str, wrong: str) -> bool:
    return bool(_word(gold).search(correct) and _word(wrong).search(misinfo)
                and not _word(wrong).search(correct) and not _word(gold).search(misinfo)
                and not any(_prefix(a).search(p) for a in (gold, wrong) for p in (correct, misinfo)))


def swap_item(r: dict) -> dict | None:
    t = r["types"]
    if not r["has_conflict"]:
        return None
    ic, im = t.index("correct"), t.index("misinfo")
    c, m, g, w = r["passages"][ic], r["passages"][im], r["gold"], r["wrong"]
    if not usable(c, m, g, w):
        return None
    passages, types = list(r["passages"]), list(t)
    passages[ic] = _word(g).sub(lambda _: w, c)     # correct style, now states the substituted answer
    passages[im] = _word(w).sub(lambda _: g, m)     # misinformation style, now states the gold answer
    types[ic], types[im] = "misinfo", "correct"
    return {**r, "passages": passages, "types": types, "style_types": t}


def main():
    rows = [json.loads(l) for l in open(SRC)]
    out = [s for s in map(swap_item, rows) if s]
    with open(OUT, "w") as f:
        for s in out:
            f.write(json.dumps(s) + "\n")
    IDS.write_text("".join(s["id"] + "\n" for s in out))
    print(f"{len(out)} usable of {sum(r['has_conflict'] for r in rows)} conflict items -> {OUT}")


if __name__ == "__main__":
    main()
