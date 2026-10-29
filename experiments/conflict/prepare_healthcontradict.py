"""
Build the HealthContradict evaluation set (PREREGISTRATION.md §17) from the pinned repository.

    git clone https://github.com/tinaboya/HealthContradict data/external/healthcontradict_src/repo
    git -C data/external/healthcontradict_src/repo checkout 97925bfb8680721be9d36e43397f988a6d499634
    python -m experiments.conflict.prepare_healthcontradict

HealthContradict (Zhang et al., npj Digital Medicine 2025) pairs, for a TREC Health Misinformation topic
(a yes/no health question with an evidence-based answer), one real web page that supports "yes" and one that
supports "no" (`doc_a` is always the "yes" page). The page whose stance matches the topic's answer is typed
correct, the other misinfo, so every item is a genuine misinformation conflict and there are no clean items.

Topics are unbalanced (5 pairs per topic at the median, 77 at most), so we keep at most CAP pairs per topic
(seed 0). Pages are long and noisy (median 7.5k characters), so each is cut to its first MAX_CHARS characters,
for the generator and the detectors alike. Passage order is shuffled per item with a fixed seed, so the
correct page comes first in about half the items. The question gets the suffix "Answer yes or no." so that
the gold answer appears as a word in the final answer.

Writes data/external/healthcontradict/hc_eval.jsonl (not committed) and the item ids to
experiments/conflict/splits/healthcontradict_eval.txt.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/external/healthcontradict_src/repo/dataset/dataset_ready.jsonl"
OUT = ROOT / "data/external/healthcontradict/hc_eval.jsonl"
IDS = Path(__file__).resolve().parent / "splits/healthcontradict_eval.txt"
CAP, MAX_CHARS = 4, 4000
SUFFIX = " Answer yes or no."


def main():
    by_topic = defaultdict(list)
    for r in map(json.loads, open(SRC)):
        by_topic[r["topic_id"]].append(r)
    rng = random.Random(0)
    out = []
    for topic in sorted(by_topic):
        rows = by_topic[topic][:]
        rng.shuffle(rows)
        for r in rows[:CAP]:
            gold = r["query_stance"].strip().lower()
            assert gold in ("yes", "no"), gold
            docs = [(r["doc_a"][:MAX_CHARS], "correct" if gold == "yes" else "misinfo"),
                    (r["doc_b"][:MAX_CHARS], "correct" if gold == "no" else "misinfo")]
            random.Random(r["instance_id"]).shuffle(docs)
            out.append({"id": f"hc-{r['instance_id']}", "topic": topic, "question": r["query"].strip() + SUFFIX,
                        "gold": gold, "wrong": "no" if gold == "yes" else "yes",
                        "passages": [d for d, _ in docs], "types": [t for _, t in docs], "has_conflict": True})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(o) + "\n" for o in out))
    IDS.write_text("".join(o["id"] + "\n" for o in sorted(out, key=lambda o: o["id"])))
    first = sum(o["types"][0] == "correct" for o in out)
    print(f"{len(out)} items from {len(by_topic)} topics (cap {CAP}); gold yes {sum(o['gold'] == 'yes' for o in out)}, "
          f"correct page first in {first} -> {OUT}")


if __name__ == "__main__":
    main()
