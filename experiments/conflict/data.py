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


RAMDOCS_LABEL_FIXES = ("exclude", "contains", "hand", "none")
HAND_LABELS = Path(__file__).resolve().parents[2] / "runs" / "labelling" / "ramdocs_suspect_docs.csv"


def _read_hand_labels(path: Path) -> dict[tuple[str, int], str]:
    import csv
    labs = {}
    for r in csv.DictReader(open(path)):
        lab = r.get("label", "").strip().lower()
        if lab and lab not in ("misinformation", "incidental", "unclear"):
            raise ValueError(f"{path}: bad label {lab!r} for {r['question_id']} doc {r['doc_index']}")
        labs[(r["question_id"], int(r["doc_index"]))] = lab
    return labs


def _ramdocs_suspects(r: dict) -> dict[int, str]:
    """Non-misinfo documents whose text contains a wrong answer -> the wrong answer found.

    In the released file, some injected misinformation sits in documents typed "correct"
    (e.g. a "Football" document saying Ronald Powell is "a renowned chess player"). But many
    hits are incidental mentions in genuine documents (a navigation menu listing "Soccer", a
    footer naming "John Coltrane"): 151 of the 163 hits also contain a gold answer. A string
    match cannot separate the two, so these documents are *suspects*, not misinformation.
    """
    from .metrics import _contains, normalize_answer  # metrics imports Example from here
    out = {}
    for i, d in enumerate(r["documents"]):
        if d["type"] == "misinfo":
            continue
        t = normalize_answer(d["text"])
        hit = next((w for w in r["wrong_answers"] if _contains(t, w)), None)
        if hit is not None:
            out[i] = hit
    return out


def load_ramdocs(path: Path | None = None, n: int | None = None, seed: int = 0,
                 label_fix: str = "exclude", hand_labels: Path | None = None) -> list[Example]:
    """`label_fix` handles suspect documents (see _ramdocs_suspects):
      exclude  : pairs touching a suspect get label "uncertain" (left out of pair metrics and
                 synthetic pools); a question whose only sign of conflict is a suspect gets
                 has_conflict=None, so it leaves the clean split. Conflict questions are unchanged.
      contains : treat every suspect as misinformation for the wrong answer it contains.
      hand     : use the team's labels (export_labelling.py; default file HAND_LABELS):
                 "misinformation" -> misinfo for the wrong answer found; "incidental" -> keep
                 the released type; "unclear" or blank -> handled as under exclude.
      none     : the released document types as they are.
    """
    if label_fix not in RAMDOCS_LABEL_FIXES:
        raise ValueError(f"label_fix must be one of {RAMDOCS_LABEL_FIXES}")
    path = path or DATA_DIR / "RAMDocs" / "RAMDocs_test.jsonl"
    rows = [json.loads(l) for l in open(path)]
    hand = _read_hand_labels(hand_labels or HAND_LABELS) if label_fix == "hand" else {}
    out = []
    for k, r in enumerate(rows):
        docs = [dict(d) for d in r["documents"]]
        suspects = _ramdocs_suspects(r) if label_fix != "none" else {}
        if label_fix == "contains":
            for i, w in suspects.items():
                docs[i]["type"], docs[i]["answer"] = "misinfo", w
        if label_fix == "hand":
            for i, w in list(suspects.items()):
                lab = hand.get((f"ramdocs-{k}", i), "")
                if lab == "misinformation":
                    docs[i]["type"], docs[i]["answer"] = "misinfo", w
                if lab in ("misinformation", "incidental"):
                    del suspects[i]       # resolved; only unclear/unlabelled stay suspect
        pairs = {(i, j): _ramdocs_pair_label(docs[i], docs[j])
                 for i in range(len(docs)) for j in range(i + 1, len(docs))}
        has_conflict = any(v == "contradiction" for v in pairs.values())
        if label_fix in ("exclude", "hand") and suspects:
            pairs = {pr: ("uncertain" if (pr[0] in suspects or pr[1] in suspects) and v != "contradiction" else v)
                     for pr, v in pairs.items()}
            has_conflict = True if has_conflict else None
        out.append(Example(
            id=f"ramdocs-{k}", dataset="ramdocs", question=r["question"],
            passages=[d["text"] for d in docs],
            gold_answers=r["gold_answers"], wrong_answers=r["wrong_answers"],
            has_conflict=has_conflict,
            conflict_type="misinfo" if r["wrong_answers"] else ("ambiguity" if len(r["gold_answers"]) > 1 else "none"),
            gold_pairs=pairs, group=f"ramdocs-{k}",
            meta={"doc_types": [d["type"] for d in docs], "doc_answers": [d["answer"] for d in docs],
                  "doc_types_released": [d["type"] for d in r["documents"]],
                  "suspect_docs": sorted(suspects), "label_fix": label_fix},
        ))
    return _subsample(out, n, seed)


# --------------------------------------------------------------------------- ConflictBank
def load_conflictbank(path: Path | None = None, n: int | None = None, seed: int = 0) -> list[Example]:
    """ConflictBank misinformation subset as built by prepare_conflictbank.py: one correct passage,
    one misinformation passage (conflict items only) and distractors; one gold answer. Scored with
    the RAMDocs strict metrics (metrics.score). Pair labels: contradiction (correct vs misinformation),
    consistent (two wrong-side passages, majority variant only), unrelated (any pair with a distractor)."""
    path = path or DATA_DIR / "conflictbank" / "conflictbank_eval.jsonl"
    out = []
    for r in map(json.loads, open(path)):
        t = r["types"]
        lab = lambda a, b: ("contradiction" if {a, b} == {"correct", "misinfo"} else
                            "consistent" if a == b == "misinfo" else "unrelated")   # misinfo-misinfo: majority variant
        pairs = {(i, j): lab(t[i], t[j]) for i in range(len(t)) for j in range(i + 1, len(t))}
        out.append(Example(
            id=r["id"], dataset="conflictbank", question=r["question"], passages=r["passages"],
            gold_answers=[r["gold"]], wrong_answers=[r["wrong"]] if r["wrong"] else [],
            has_conflict=r["has_conflict"], conflict_type="misinfo" if r["has_conflict"] else "none",
            gold_pairs=pairs, group=r["id"],
            meta={"doc_types": t, "relation": r["relation"], "suspect_docs": [], "label_fix": "none"}))
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
    ds = load_dataset("deepmind/narrativeqa", split=split)  # namespaced id required by datasets>=4
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


# --------------------------------------------------------------------------- WikiContradict
WIKICONTRADICT_REVISION = "c20e361f985ed480a659b35d98b49f2311fcd174"   # HF commit, checked 2026-10-27


def load_wikicontradict(n: int | None = None, seed: int = 0) -> list[Example]:
    """WikiContradict (Hou et al. 2024; HF ibm/Wikipedia_contradict_benchmark).

    253 questions, each with two Wikipedia passages that contradict each other about the answer
    (the HF train split has every question twice; the first row per question_ID is kept). The
    reference answer is BOTH answers ("answer1|answer2"): these are real-world knowledge conflicts
    with no misinformation side, so there is no direction, and no conflict-free questions.
    That makes it the contrast case to RAMDocs, where listing both sides is wrong.
    """
    from datasets import load_dataset  # heavy, import lazily
    ds = load_dataset("ibm/Wikipedia_contradict_benchmark", revision=WIKICONTRADICT_REVISION)["train"]
    seen, out = set(), []
    for r in ds:
        if r["question_ID"] in seen:
            continue
        seen.add(r["question_ID"])
        out.append(Example(
            id=f"wikicontradict-{r['question_ID']}", dataset="wikicontradict", question=r["question"],
            passages=[r["context1"], r["context2"]],
            gold_answers=[r["answer1"].strip(), r["answer2"].strip()],
            has_conflict=True, conflict_type=r["contradictType"],
            gold_pairs={(0, 1): "contradiction"}, group=f"wikicontradict-{r['question_ID']}",
            meta={"doc_types": ["correct", "correct"], "samepassage": r["samepassage"],
                  "title": r["WikipediaArticleTitle"], "url": r["url"]},
        ))
    return _subsample(out, n, seed)


# --------------------------------------------------------------------------- helpers
def _subsample(xs: list[Example], n: int | None, seed: int) -> list[Example]:
    if n is None or n >= len(xs):
        return xs
    rng = random.Random(seed)
    return sorted(rng.sample(xs, n), key=lambda e: e.id)


SPLITS_DIR = Path(__file__).resolve().parent / "splits"


def filter_ids(xs: list[Example], ids_file: str | Path | None) -> list[Example]:
    """Keep the examples listed in `ids_file` (one id per line), e.g. splits/ramdocs_holdout.txt.
    Fails if an id is missing, so a split can never silently shrink."""
    if not ids_file:
        return xs
    want = [l.strip() for l in open(ids_file) if l.strip()]
    by_id = {x.id: x for x in xs}
    missing = [i for i in want if i not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} ids from {ids_file} not in the loaded data, e.g. {missing[:3]}")
    return [by_id[i] for i in sorted(want)]


def load_conflictbank_majority(path: Path | None = None, **kw) -> list[Example]:
    """§11: the same items with three wrong-side passages per conflict item (prepare_conflictbank.py)."""
    return load_conflictbank(path or DATA_DIR / "conflictbank" / "conflictbank_majority.jsonl", **kw)


LOADERS = {"conflictbank": load_conflictbank, "conflictbank_majority": load_conflictbank_majority, "ramdocs": load_ramdocs, "conflicts": load_conflicts, "wikicontradict": load_wikicontradict}


def load(name: str, **kw) -> list[Example]:
    if name not in LOADERS:
        raise NotImplementedError(
            f"No verified loader for '{name}'. Inspect it with "
            f"`python -m experiments.conflict.inspect_data {name}` and add one to data.py.")
    return LOADERS[name](**kw)
