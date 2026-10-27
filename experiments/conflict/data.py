"""
Benchmark loaders. Every loader returns a list of `Example` with the same schema,
so detectors, prompts and metrics never need dataset-specific code.

Verified against the real files (Oct 2026):
  - RAMDocs   : github.com/HanNight/RAMDocs  (RAMDocs_test.jsonl, 500 questions)
  - CONFLICTS : github.com/google-research-datasets/rag_conflicts (conflicts.jsonl, 458)
  - NarrativeQA: HF "narrativeqa" (same loader fields the repo already uses)

Not yet verified (no Hugging Face access when this was written): FaithEval,
ConfRAG, WikiContradict. Run `python -m experiments.conflict.inspect_data <name>`
and fill in the loader before using them.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "external"

# CONFLICTS taxonomy (Cattan et al. 2025). Only these are true contradictions;
# "Complementary information" means sources differ without contradicting.
CONFLICTS_CONTRADICTORY = {
    "Conflicting opinions and research outcomes",
    "Conflict due to outdated information",
    "Conflict due to misinformation",
}


@dataclass
class Example:
    id: str
    dataset: str
    question: str
    passages: list[str]
    gold_answers: list[str] = field(default_factory=list)
    wrong_answers: list[str] = field(default_factory=list)
    has_conflict: Optional[bool] = None          # question-level gold label
    conflict_type: Optional[str] = None
    # Pair-level gold: {(i, j): label} with i < j. None if the dataset has no pair labels.
    gold_pairs: Optional[dict[tuple[int, int], str]] = None
    group: str = ""                               # cluster id for bootstrap (e.g. story id)
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- RAMDocs
def _ramdocs_pair_label(a: dict, b: dict) -> str:
    """Gold relation between two RAMDocs documents.

    contradiction : one supports a gold answer, the other a misinformation answer
                    (or two different misinformation answers)
    ambiguity     : both correct but for different entities (different valid answers)
    consistent    : same answer
    unrelated     : at least one is noise
    """
    if a["type"] == "noise" or b["type"] == "noise":
        return "unrelated"
    if a["answer"].strip().lower() == b["answer"].strip().lower():
        return "consistent"
    if a["type"] == "correct" and b["type"] == "correct":
        return "ambiguity"
    return "contradiction"


def load_ramdocs(path: Path | None = None, n: int | None = None, seed: int = 0) -> list[Example]:
    path = path or DATA_DIR / "RAMDocs" / "RAMDocs_test.jsonl"
    rows = [json.loads(l) for l in open(path)]
    out = []
    for k, r in enumerate(rows):
        docs = r["documents"]
        pairs = {(i, j): _ramdocs_pair_label(docs[i], docs[j])
                 for i in range(len(docs)) for j in range(i + 1, len(docs))}
        out.append(Example(
            id=f"ramdocs-{k}", dataset="ramdocs", question=r["question"],
            passages=[d["text"] for d in docs],
            gold_answers=r["gold_answers"], wrong_answers=r["wrong_answers"],
            has_conflict=any(v == "contradiction" for v in pairs.values()),
            conflict_type="misinfo" if r["wrong_answers"] else ("ambiguity" if len(r["gold_answers"]) > 1 else "none"),
            gold_pairs=pairs, group=f"ramdocs-{k}",
            meta={"doc_types": [d["type"] for d in docs], "doc_answers": [d["answer"] for d in docs]},
        ))
    return _subsample(out, n, seed)


# --------------------------------------------------------------------------- CONFLICTS
def load_conflicts(path: Path | None = None, n: int | None = None, seed: int = 0,
                   text_field: str = "short_text") -> list[Example]:
    """`short_text` is the dataset's 512-token window; `response_str` is the full page."""
    path = path or DATA_DIR / "rag_conflicts" / "conflicts.jsonl"
    rows = [json.loads(l) for l in open(path)]
    out = []
    for k, r in enumerate(rows):
        ans = r["correct_answer"].strip()
        out.append(Example(
            id=f"conflicts-{k}", dataset="conflicts", question=r["question"],
            passages=[s[text_field] for s in r["search_results"]],
            gold_answers=[ans] if ans else [],
            has_conflict=r["conflict_type"] in CONFLICTS_CONTRADICTORY,
            conflict_type=r["conflict_type"], gold_pairs=None, group=f"conflicts-{k}",
            meta={"source": r["source"], "dates": [s.get("date") for s in r["search_results"]]},
        ))
    return _subsample(out, n, seed)


# --------------------------------------------------------------------------- NarrativeQA
def load_narrativeqa_questions(n: int, seed: int = 42, split: str = "test",
                               max_per_story: int = 2) -> list[dict]:
    """Random questions spread across stories.

    Fixes the original loader, which took the first N rows of the split; because rows
    are grouped by story, the 50-question run in the paper came from only 2 stories.
    Returns raw dicts with the full story; retrieval happens later.
    """
    from datasets import load_dataset  # heavy, import lazily
    ds = load_dataset("narrativeqa", split=split)
    by_story = defaultdict(list)
    for i, doc_id in enumerate(ds["document"]):
        by_story[doc_id["id"]].append(i)
    rng = random.Random(seed)
    stories = list(by_story)
    rng.shuffle(stories)
    picked = []
    for sid in stories:
        idx = by_story[sid][:]
        rng.shuffle(idx)
        picked.extend(idx[:max_per_story])
        if len(picked) >= n:
            break
    picked = picked[:n]
    out = []
    for i in picked:
        ex = ds[i]
        out.append({
            "id": f"nqa-{ex['document']['id']}-{i}", "story_id": ex["document"]["id"],
            "question": ex["question"]["text"], "answers": [a["text"] for a in ex["answers"]],
            "story": ex["document"]["text"],
        })
    return out


# --------------------------------------------------------------------------- helpers
def _subsample(xs: list[Example], n: int | None, seed: int) -> list[Example]:
    if n is None or n >= len(xs):
        return xs
    rng = random.Random(seed)
    return sorted(rng.sample(xs, n), key=lambda e: e.id)


LOADERS = {"ramdocs": load_ramdocs, "conflicts": load_conflicts}


def load(name: str, **kw) -> list[Example]:
    if name not in LOADERS:
        raise NotImplementedError(
            f"No verified loader for '{name}'. Inspect it with "
            f"`python -m experiments.conflict.inspect_data {name}` and add one to data.py.")
    return LOADERS[name](**kw)
