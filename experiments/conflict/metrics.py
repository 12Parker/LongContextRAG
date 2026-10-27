"""
Scoring.

RAMDocs: the public repo ships `normalize_answer` but no scorer. `ramdocs_strict` below is
our reconstruction of the paper's exact-match criterion (every gold answer present and no
wrong answer present). VERIFY against Wang et al. 2025, Sec. 4, before reporting numbers
next to theirs; also report `gold_recall` and `wrong_rate` so the strict metric is not the
only view.

CONFLICTS: automatic accuracy is defined only where `correct_answer` exists (all 161
"No conflict" items and the outdated/misinformation items). Opinion and complementary
items need a judge (their paper uses one); they get `None` here.
"""
from __future__ import annotations

import re
import string

from .data import Example


def normalize_answer(s: str) -> str:
    """Identical to RAMDocs/run_madam_rag.py."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


_FINAL = re.compile(r"final answer\(s\)\s*:\s*(.*)", re.I)
_MADAM = re.compile(r"all correct answers\s*:\s*\[(.*?)\]", re.I | re.S)


def extract_answers(text: str) -> list[str]:
    m = _FINAL.search(text) or None
    if m:
        return [a.strip() for a in m.group(1).split("|") if a.strip()]
    m = _MADAM.search(text)
    if m:
        return [a.strip().strip('"\'') for a in m.group(1).split(",") if a.strip()]
    return [text.strip()]


def _contains(pred_norm: str, ans: str) -> bool:
    a = normalize_answer(ans)
    return bool(a) and re.search(rf"(?<!\w){re.escape(a)}(?!\w)", pred_norm) is not None


def score(ex: Example, response: str) -> dict:
    answers = extract_answers(response)
    pred = normalize_answer(" ; ".join(answers))
    out: dict = {"pred_answers": answers}
    if ex.dataset == "ramdocs":
        g = [_contains(pred, a) for a in ex.gold_answers]
        w = [_contains(pred, a) for a in ex.wrong_answers]
        out.update(ramdocs_strict=float(all(g) and not any(w)),
                   gold_recall=sum(g) / max(len(g), 1),
                   wrong_rate=float(any(w)))
        out["primary"] = out["ramdocs_strict"]
    elif ex.dataset == "conflicts":
        out["primary"] = float(_contains(pred, ex.gold_answers[0])) if ex.gold_answers else None
    else:
        out["primary"] = float(any(_contains(pred, a) for a in ex.gold_answers)) if ex.gold_answers else None
    return out


# --------------------------------------------------------------------------- detection
def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}


def pair_level(examples: list[Example], flagged: dict[str, set[tuple[int, int]]],
               positive: str = "contradiction") -> dict:
    tp = fp = fn = 0
    for ex in examples:
        if ex.gold_pairs is None:
            continue
        gold = {k for k, v in ex.gold_pairs.items() if v == positive}
        f = flagged.get(ex.id, set())
        tp += len(f & gold); fp += len(f - gold); fn += len(gold - f)
    return prf(tp, fp, fn)


def question_level(examples: list[Example], flagged: dict[str, set[tuple[int, int]]]) -> dict:
    tp = fp = fn = tn = 0
    for ex in examples:
        if ex.has_conflict is None:
            continue
        pred = bool(flagged.get(ex.id))
        tp += pred and ex.has_conflict; fp += pred and not ex.has_conflict
        fn += (not pred) and ex.has_conflict; tn += (not pred) and not ex.has_conflict
    d = prf(tp, fp, fn)
    d["tn"] = tn
    d["false_positive_rate"] = fp / (fp + tn) if fp + tn else 0.0
    return d


def false_flags_by_relation(examples: list[Example], flagged: dict[str, set[tuple[int, int]]]) -> dict:
    """For RAMDocs: what gold relation do the false flags actually have?"""
    from collections import Counter
    c = Counter()
    for ex in examples:
        if ex.gold_pairs is None:
            continue
        for pr in flagged.get(ex.id, set()):
            c[ex.gold_pairs.get(pr, "?")] += 1
    return dict(c)
