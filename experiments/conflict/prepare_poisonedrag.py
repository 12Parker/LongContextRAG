"""
Build the PoisonedRAG evaluation set: real retrieval with injected attack passages (planned §17, pipeline v3).

Sources (downloaded to data/external/poisonedrag_src/, not committed):
- PoisonedRAG (Zou et al., USENIX Security 2025), github.com/sleeepeer/PoisonedRAG at commit f660d72 (MIT):
  results/adv_targeted_results/{nq,hotpotqa}.json (100 target questions each, with a correct answer, the
  attacker's target answer and 5 attack passages) and results/beir_results/{nq,hotpotqa}-contriever.json
  (Contriever top-100 corpus scores for every BEIR test query).
- BEIR nq.zip and hotpotqa.zip (corpus, queries) from public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets.
- NQ-open validation answers (google-research-datasets/nq_open) for clean NQ questions, matched by question text;
  BEIR's HotpotQA queries carry their answer in `metadata.answer`.
- facebook/contriever, run locally.

Attacked items: the first DOSE attack passages of each target question are scored against the question with
Contriever (mean pooling, dot product, as PoisonedRAG) and merged with the released corpus scores; the generator gets
the top K. Clean items: other test queries with a gold answer, top K from the released scores, no attack.
Passage types: attack passages are misinfo; a corpus passage containing a gold answer (normalised, whole word) is
correct; anything else is noise. Every attacked item is a conflict item (has_conflict) whether or not a correct
passage was retrieved; meta records both.

Writes data/external/poisonedrag/pr_eval.jsonl and splits/poisonedrag_pilot.txt (20 attacked + 20 clean, excluded
from evaluation) and splits/poisonedrag_eval.txt (the rest).
"""
from __future__ import annotations

import io
import json
import random
import re
import string
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/external/poisonedrag_src"
OUT = ROOT / "data/external/poisonedrag/pr_eval.jsonl"
SPLITS = Path(__file__).resolve().parent / "splits"
DOSE, K, N_CLEAN = 2, 5, 150
N_PILOT_ATT, N_PILOT_CLEAN = 10, 10   # per dataset


def norm(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def contains(text, ans):
    a = norm(ans)
    return bool(a) and re.search(rf"(^| ){re.escape(a)}( |$)", norm(text)) is not None


def read_zip(path, name):
    with zipfile.ZipFile(path) as z:
        member = next(m for m in z.namelist() if m.endswith(name))
        with z.open(member) as f:
            for line in io.TextIOWrapper(f, encoding="utf-8"):
                yield json.loads(line)


class Contriever:
    def __init__(self):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained("facebook/contriever")
        self.model = AutoModel.from_pretrained("facebook/contriever").eval()

    def __call__(self, texts):
        with self.torch.no_grad():
            b = self.tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
            h = self.model(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).float()
            return ((h * m).sum(1) / m.sum(1)).numpy()


def main():
    import numpy as np
    import pandas as pd
    enc = Contriever()
    nq_open = pd.read_parquet(SRC / "answers/nq_open/validation-00000-of-00001.parquet")
    nq_ans = {norm(q): list(a) for q, a in zip(nq_open["question"], nq_open["answer"])}
    rng = random.Random(0)
    items, checks = [], []
    for ds in ("nq", "hotpotqa"):
        adv = json.load(open(SRC / f"results/adv_targeted_results/{ds}.json"))
        top = json.load(open(SRC / f"results/beir_results/{ds}-contriever.json"))
        queries = {q["_id"]: q for q in read_zip(SRC / f"beir/{ds}.zip", "queries.jsonl")}
        def gold_of(qid):
            q = queries[qid]
            if ds == "hotpotqa":
                a = q.get("metadata", {}).get("answer")
                return [a] if a else []
            return nq_ans.get(norm(q["text"]), [])
        targets = [v["id"] for v in adv.values()]
        pool = [q for q in top if q not in set(targets) and gold_of(q) and q in queries]
        rng.shuffle(pool)
        clean = sorted(pool[:N_CLEAN])
        need = {d for q in targets + clean for d, _ in sorted(top[q].items(), key=lambda kv: -kv[1])[:K]}
        corpus = {}
        for doc in read_zip(SRC / f"beir/{ds}.zip", "corpus.jsonl"):
            if doc["_id"] in need:
                corpus[doc["_id"]] = (doc.get("title", "") + ". " + doc["text"]).strip(". ").strip() if doc.get("title") else doc["text"]
        # alignment check: our Contriever reproduces the released scores
        for q in targets[:5]:
            d, s = max(top[q].items(), key=lambda kv: kv[1])
            qe, de = enc([queries[q]["text"]]), enc([corpus[d]])
            checks.append((ds, q, round(float(s), 3), round(float((qe @ de.T)[0, 0]), 3)))
        for v in adv.values():
            q = v["id"]; qe = enc([v["question"]])[0]
            atk = v["adv_texts"][:DOSE]
            ae = enc(atk)
            cand = [(float(s), "corpus", d) for d, s in top[q].items()] + [(float(ae[k] @ qe), "attack", k) for k in range(len(atk))]
            cand = sorted(cand, key=lambda t: -t[0])[:K]
            gold = [v["correct answer"]] + [a for a in gold_of(q) if norm(a) != norm(v["correct answer"])]
            passages, types = [], []
            for _, kind, ref in cand:
                text = atk[ref] if kind == "attack" else corpus[ref]
                passages.append(text)
                types.append("misinfo" if kind == "attack" else ("correct" if any(contains(text, a) for a in gold) else "noise"))
            items.append({"id": f"pr-{ds}-{q}", "dataset_src": ds, "question": v["question"], "gold": gold,
                          "wrong": v["incorrect answer"], "passages": passages, "types": types, "has_conflict": True,
                          "n_attack_retrieved": types.count("misinfo"), "correct_retrieved": "correct" in types})
        for q in clean:
            gold = gold_of(q)
            ids = [d for d, _ in sorted(top[q].items(), key=lambda kv: -kv[1])[:K]]
            passages = [corpus[d] for d in ids]
            types = ["correct" if any(contains(p, a) for a in gold) else "noise" for p in passages]
            items.append({"id": f"pr-{ds}-{q}", "dataset_src": ds, "question": queries[q]["text"], "gold": gold,
                          "wrong": None, "passages": passages, "types": types, "has_conflict": False,
                          "n_attack_retrieved": 0, "correct_retrieved": "correct" in types})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(o) + "\n" for o in items))
    pilot = []
    for ds in ("nq", "hotpotqa"):
        for att, n in ((True, N_PILOT_ATT), (False, N_PILOT_CLEAN)):
            ids = sorted(o["id"] for o in items if o["dataset_src"] == ds and o["has_conflict"] is att)
            pilot += random.Random(1).sample(ids, n)
    (SPLITS / "poisonedrag_pilot.txt").write_text("".join(i + "\n" for i in sorted(pilot)))
    (SPLITS / "poisonedrag_eval.txt").write_text("".join(o["id"] + "\n" for o in sorted(items, key=lambda o: o["id"]) if o["id"] not in set(pilot)))
    att = [o for o in items if o["has_conflict"]]
    print("score check (released, ours):", checks)
    print(f"{len(items)} items: {len(att)} attacked, {len(items) - len(att)} clean; attack passages in top {K}: "
          f"{sum(o['n_attack_retrieved'] for o in att) / len(att):.2f} mean; correct passage retrieved: "
          f"attacked {sum(o['correct_retrieved'] for o in att)}/{len(att)}, clean "
          f"{sum(o['correct_retrieved'] for o in items if not o['has_conflict'])}/{len(items) - len(att)}")


if __name__ == "__main__":
    main()
