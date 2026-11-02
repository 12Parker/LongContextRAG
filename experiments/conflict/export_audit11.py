"""
Blind human-audit sheet A1 (PREREGISTRATION.md §20.3): matcher-judge disagreements in the pooled H23 / H30a arms.

    python -m experiments.conflict.export_audit11

Frame: the §19.2 re-score, i.e. the conflict responses of the arms oracle+flag_directional and answer_key (pooled H23)
and llm_dir+flag_directional and llm_dir+verdict_key (pooled H30a) that the judge scored, joined to
runs/judge10/pooled.jsonl by (dataset, id, response key) (rev11_scoring.a1_population). Per arm: 50 responses on which
the one-way matcher and judge-strict disagree and 25 on which they agree, drawn with seed 0, at most one response per
(dataset, question) per arm; 300 rows, shuffled. Writes runs/labelling/
  audit11_disagreements.csv   the blind sheet: row, question, gold and wrong answers, response; gives_gold,
                              asserts_wrong, abstains (yes/no) and notes left empty
  audit11_key.csv             row -> dataset, question id, run, condition, seed, response key, matcher, judge-strict,
                              sampling stratum (keep it away from the labellers)
  audit11_INSTRUCTIONS.txt
Nothing is labelled here. Two labellers fill copies named audit11_disagreements_<initials>.csv independently.
"""
from __future__ import annotations

import csv
import random
from collections import Counter

from .export_audit8 import redact
from .rev11_scoring import A1_ARMS, ROOT, _examples, a1_population

OUT = ROOT / "runs/labelling"
N_DIS, N_AGR, SEED = 50, 25, 0
LABELS = ("gives_gold", "asserts_wrong", "abstains")
INSTRUCTIONS = """A1 audit sheet (PREREGISTRATION.md section 20.3). audit11_disagreements.csv has 300 rows in random order.
Each row is one model response to a question, shown with the reference answer(s) and the known wrong
(misinformation) answer(s). Answers are separated by " | ". Judge what the response FINALLY asserts (its final answer,
read in the light of the whole response), against the answers given on the row; do not look the question up.

Labels, one per column, written yes or no:
- gives_gold: the final answer states every reference answer (paraphrases, aliases and equivalent forms count).
- asserts_wrong: the final answer states a known wrong answer as true. Mentioning it only to reject it does not count.
- abstains: the final answer is "unknown", or the response declines or fails to give an answer.
Leave a cell blank only if you cannot decide, and say why in notes. Do not edit the row, question, answer or response
columns. Human strict (computed later, not labelled) = gives_gold yes, asserts_wrong no and abstains no.

Procedure:
1. Two annotators label independently, each in their own copy named audit11_disagreements_<initials>.csv, without
   discussing rows and without opening audit11_key.csv.
2. After both copies are complete, the two meet and adjudicate every row and label on which they differ (blanks
   included), and record the agreed labels in audit11_consensus.csv (same columns; notes give the reason).
3. Keep both individual copies unchanged; agreement is computed from them, the analysis from the consensus.
"""


def draw(pop, rng, arms=A1_ARMS, n_dis=N_DIS, n_agr=N_AGR):
    """Per arm: its disagreements, then its agreements, each in a random order, taking a response only while its
    (dataset, question) is unused in that arm, until n_dis and n_agr are taken; the result shuffled."""
    out = []
    for arm in arms:
        P = sorted((p for p in pop if p["condition"] == arm),
                   key=lambda p: (p["dataset"], p["id"], p["run"], p["seed"], p["response_key"]))
        used = set()
        for stratum, dis, k in (("disagree", True, n_dis), ("agree", False, n_agr)):
            pool = [p for p in P if (p["matcher"] != p["judge_strict"]) == dis]
            rng.shuffle(pool)
            took = 0
            for p in pool:
                if took == k:
                    break
                if (p["dataset"], p["id"]) not in used:
                    used.add((p["dataset"], p["id"])); out.append({**p, "stratum": stratum}); took += 1
    rng.shuffle(out)
    return out


def main(argv=None):
    rows = draw(a1_population(), random.Random(SEED))
    sheet, key = [], []
    for n, p in enumerate(rows, 1):
        e = _examples(p["run"])[p["id"]]
        clean = redact if p["dataset"] == "raguard" else (lambda t: t)
        sheet.append({"row": n, "question": clean(e.question), "gold_answers": " | ".join(e.gold_answers),
                      "wrong_answers": " | ".join(e.wrong_answers), "response": clean(p["response"]),
                      **{l: "" for l in LABELS}, "notes": ""})
        key.append({"row": n, "dataset": p["dataset"], "id": p["id"], "run": p["run"], "condition": p["condition"],
                    "seed": p["seed"], "response_key": p["response_key"], "matcher": int(p["matcher"]),
                    "judge_strict": int(p["judge_strict"]), "stratum": p["stratum"]})
    OUT.mkdir(parents=True, exist_ok=True)
    for name, data in (("audit11_disagreements.csv", sheet), ("audit11_key.csv", key)):
        with open(OUT / name, "w", newline="") as f:
            w = csv.DictWriter(f, list(data[0]))
            w.writeheader(); w.writerows(data)
    (OUT / "audit11_INSTRUCTIONS.txt").write_text(INSTRUCTIONS)
    c = Counter((k["condition"], k["stratum"]) for k in key)
    sub = Counter((k["condition"], "judge-only" if k["judge_strict"] else "matcher-only") for k in key if k["stratum"] == "disagree")
    print(f"{len(key)} rows -> {OUT}")
    for arm in A1_ARMS:
        ds = Counter(k["dataset"] for k in key if k["condition"] == arm)
        print(f"  {arm:26s} disagree {c[arm, 'disagree']:3d} (judge-only {sub[arm, 'judge-only']}, matcher-only "
              f"{sub[arm, 'matcher-only']})  agree {c[arm, 'agree']:3d}  datasets {dict(ds)}")


if __name__ == "__main__":
    main()
