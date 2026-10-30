"""
Export the RAMDocs suspect documents for hand-labelling.

    python -m experiments.conflict.export_labelling --out runs/labelling

A suspect is a document not typed "misinfo" whose text contains a wrong answer
(data._ramdocs_suspects). Each row shows the match in context. Fill in `label` with:

    misinformation   the document asserts the wrong answer as the answer to the question
    incidental       the wrong answer appears for another reason (menu, footer, another entity)
    unclear

Then load the labels with `data.load_ramdocs(label_fix="hand", hand_labels=<csv>)`.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from .data import DATA_DIR, _ramdocs_suspects

LABELS = ("misinformation", "incidental", "unclear")


def _snippet(text: str, needle: str, width: int = 220) -> str:
    m = re.search(re.escape(needle), text, re.I)
    if not m:
        return text[: 2 * width]
    a, b = max(0, m.start() - width), min(len(text), m.end() + width)
    return ("..." if a else "") + text[a:m.start()] + ">>>" + text[m.start():m.end()] + "<<<" + text[m.end():b] \
        + ("..." if b < len(text) else "")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/labelling")
    args = ap.parse_args(argv)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(DATA_DIR / "RAMDocs" / "RAMDocs_test.jsonl")]
    pilot = {l.strip() for l in open(Path(__file__).parent / "splits" / "ramdocs_pilot.txt")}
    recs = []
    for k, r in enumerate(rows):
        for i, w in sorted(_ramdocs_suspects(r).items()):
            d = r["documents"][i]
            recs.append({"question_id": f"ramdocs-{k}", "split": "pilot" if f"ramdocs-{k}" in pilot else "holdout",
                         "doc_index": i, "question": r["question"], "gold_answers": " | ".join(r["gold_answers"]),
                         "wrong_answer_found": w, "released_type": d["type"], "released_answer": d["answer"],
                         "context": _snippet(d["text"], w), "full_text": d["text"], "label": "", "notes": ""})
    path = out / "ramdocs_suspect_docs.csv"
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(recs[0]))
        wr.writeheader(); wr.writerows(recs)
    print(f"{len(recs)} suspect documents in {len({r['question_id'] for r in recs})} questions -> {path}")
    print(f"label column values: {', '.join(LABELS)}")


if __name__ == "__main__":
    main()
