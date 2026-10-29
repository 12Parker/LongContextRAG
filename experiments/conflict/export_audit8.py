"""
Blinded, paired human-audit sheet (PREREGISTRATION.md §17.4).

    python -m experiments.conflict.export_audit8 [--per-contrast 15] [--n-raguard 50]

Pairs: for each §17.3 contrast, the first `per-contrast` questions of the judge's sample, each with the response
under both conditions, in random order with the condition hidden. RAGuard: the first `n-raguard` judged post pairs.
Writes runs/labelling/audit8_pairs.csv and audit8_raguard.csv for the labellers, and audit8_key.csv (which condition
each response is; keep it away from the labellers). Two labellers fill the label columns independently, in copies
named audit8_pairs_<initials>.csv.
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from . import judge8 as J

OUT = Path(__file__).resolve().parents[2] / "runs/labelling"
LABELS = ("gives_gold", "asserts_wrong", "rejects_wrong", "abstains", "other_answer")
INSTRUCTIONS = ("Label each response's FINAL answer: gives_gold = states every reference answer (aliases count); "
                "asserts_wrong = states a known wrong answer as true (mentioning it to reject it does not count); "
                "rejects_wrong = explicitly says a wrong answer is false; abstains = unknown/declines; "
                "other_answer = asserts an answer in neither list. Use 1/0; leave blank if unsure and add a note.")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-contrast", type=int, default=15)
    ap.add_argument("--n-raguard", type=int, default=50)
    args = ap.parse_args(argv)
    rng = random.Random("audit8")
    exs = {}
    pairs, key = [], []
    for name, (ds, (ra, ca), (rb, cb)) in J.CONTRASTS.items():
        ex = exs.setdefault(ds, J._examples(ds))
        A, B = J._rows(ra, ca), J._rows(rb, cb)
        ids = sorted(set(A) & set(B) & set(ex))
        ids = random.Random(f"judge8|{name}").sample(ids, min(120, len(ids)))[: args.per_contrast]
        for i in ids:
            arms = [("a", A[i]), ("b", B[i])]
            rng.shuffle(arms)
            e = ex[i]
            pairs.append({"item": len(pairs), "question": e.question, "reference": " | ".join(e.gold_answers),
                          "known_wrong": " | ".join(e.wrong_answers),
                          "response_1": arms[0][1]["response"], "response_2": arms[1][1]["response"],
                          **{f"r1_{l}": "" for l in LABELS}, **{f"r2_{l}": "" for l in LABELS}, "notes": ""})
            key.append({"item": len(pairs) - 1, "contrast": name, "id": i,
                        "response_1": arms[0][1]["condition"], "response_2": arms[1][1]["condition"],
                        "matcher_1": arms[0][1]["ramdocs_strict_oneway"], "matcher_2": arms[1][1]["ramdocs_strict_oneway"]})
    rng.shuffle(pairs)   # contrasts interleaved; the key is joined on `item`
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "audit8_pairs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(pairs[0]))
        w.writeheader(); w.writerows(pairs)
    with open(OUT / "audit8_key.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(key[0]))
        w.writeheader(); w.writerows(key)
    rg = J.raguard_jobs()[: args.n_raguard]
    ex = exs.setdefault("raguard", J._examples("raguard"))
    with open(OUT / "audit8_raguard.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "claim", "correct_verdict", "post_A", "post_B", "contradict", "temporally_compatible",
                    "b_false_about_claim", "a_supports_verdict", "notes"])
        for k, (m, _) in enumerate(rg):
            e = ex[m["id"]]
            w.writerow([k, e.question.split("Claim:", 1)[-1].strip(), e.gold_answers[0], e.passages[m["a"]],
                        e.passages[m["b"]], "", "", "", "", ""])
    (OUT / "audit8_INSTRUCTIONS.txt").write_text(
        INSTRUCTIONS + "\nRAGuard sheet: contradict / temporally_compatible = 1/0; b_false_about_claim and "
        "a_supports_verdict = yes/no/unclear. Label independently; do not open audit8_key.csv.\n")
    print(f"{len(pairs)} pairs ({len(pairs) * 2} responses), {len(rg)} RAGuard pairs -> {OUT}")


if __name__ == "__main__":
    main()
