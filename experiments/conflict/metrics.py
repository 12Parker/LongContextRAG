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


_TAGGED = re.compile(r"<([A-Za-z_]+)>(.*?)</\1>", re.S)


def extract_answers(text: str) -> list[str]:
    m = _FINAL.search(text) or None
    if m:
        # some models copy the answer-line template literally: "Final answer(s): <answer>Paris</answer>"
        line = _TAGGED.sub(lambda t: t.group(2), m.group(1))
        return [a.strip() for a in line.split("|") if a.strip()]
    m = _MADAM.search(text)
    if m:
        # items are quoted strings that may themselves contain commas ("July 10, 1850");
        # splitting on commas turned one answer into fragments that then matched gold answers
        quoted = re.findall(r'"([^"]*)"|\'([^\']*)\'', m.group(1))
        if quoted:
            return [(a or b).strip() for a, b in quoted if (a or b).strip()]
        return [a.strip() for a in m.group(1).split(",") if a.strip()]
    return [text.strip()]


def _contains(pred_norm: str, ans: str) -> bool:
    a = normalize_answer(ans)
    return bool(a) and re.search(rf"(?<!\w){re.escape(a)}(?!\w)", pred_norm) is not None


def _matches(answers: list[str], ref: str) -> bool:
    """Some predicted answer matches `ref` as a whole-word span, in either direction:
    "1877" matches gold "January 1, 1877", and "5,000" matches wrong answer "5,000 people".
    One-way matching (ref inside the prediction) missed both, so it credited wrong answers
    and penalised right ones whenever RAMDocs answers carry extra words."""
    r = normalize_answer(ref)
    return any(_contains(normalize_answer(a), ref) or _contains(r, a) for a in answers)


def score(ex: Example, response: str) -> dict:
    answers = extract_answers(response)
    pred = normalize_answer(" ; ".join(answers))
    out: dict = {"pred_answers": answers}
    if ex.dataset in ("ramdocs", "conflictbank", "raguard", "healthcontradict", "poisonedrag"):
        g = [_matches(answers, a) for a in ex.gold_answers]
        w = [_matches(answers, a) for a in ex.wrong_answers]
        g1 = [_contains(pred, a) for a in ex.gold_answers]   # one-way, kept for comparison
        w1 = [_contains(pred, a) for a in ex.wrong_answers]
        out.update(ramdocs_strict=float(all(g) and not any(w)),
                   gold_recall=sum(g) / max(len(g), 1),
                   wrong_rate=float(any(w)),
                   ramdocs_strict_oneway=float(all(g1) and not any(w1)))
        out["primary"] = out["ramdocs_strict"]
    elif ex.dataset == "wikicontradict":
        # the reference is both sides of a genuine conflict; reporting only one is incomplete
        g = [_matches(answers, a) for a in ex.gold_answers]
        out.update(both_answers=float(all(g)), gold_recall=sum(g) / max(len(g), 1))
        out["primary"] = out["both_answers"]
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
        f = {pr for pr in flagged.get(ex.id, set()) if ex.gold_pairs.get(pr) != "uncertain"}
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


def direction_accuracy(examples: list[Example], dets: dict) -> float | None:
    """Share of flagged gold-contradiction pairs whose `suspects` are exactly the gold
    misinformation side(s). None if no flagged true pair carries a direction."""
    right = total = 0
    for ex in examples:
        types = ex.meta.get("doc_types", [])
        for d in dets[ex.id].pairs:
            if (ex.gold_pairs or {}).get((d.i, d.j)) != "contradiction" or not d.suspects:
                continue
            total += 1
            right += sorted(d.suspects) == [k for k in (d.i, d.j) if types[k] == "misinfo"]
    return right / total if total else None


def direction_coverage(examples: list[Example], dets: dict) -> float | None:
    """Share of flagged gold-contradiction pairs that carry any direction (`suspects`)."""
    flagged = with_dir = 0
    for ex in examples:
        for d in dets[ex.id].pairs:
            if (ex.gold_pairs or {}).get((d.i, d.j)) == "contradiction":
                flagged += 1
                with_dir += bool(d.suspects)
    return with_dir / flagged if flagged else None


# --------------------------------------------------------------------------- document level
DOC_MODES = ("remove", "remove_suspect", "flag_directional")
# modes whose documents are marked like flag_directional
_MARK_AS = {"flag_directional_instruct_v2": "flag_directional"}


def marked_docs(dets, mode: str) -> set[int]:
    """Passages a prompt mode acts on: removed (remove, remove_suspect) or headed
    LIKELY MISINFORMATION (flag_directional). Mirrors prompts.build_prompt."""
    mode = _MARK_AS.get(mode, mode)
    if mode == "remove":
        return {d.j for d in dets}
    if mode == "remove_suspect":
        return {k for d in dets for k in (d.suspects or [d.j])}
    if mode == "flag_directional":
        return {k for d in dets for k in d.suspects}
    raise ValueError(f"no document-level reading for mode {mode}")


def doc_level(examples: list[Example], dets: dict, mode: str) -> dict:
    """What a detector does to documents in a given mode.

    misinfo_marked  share of misinformation documents marked (benefit side)
    correct_marked  share of correct documents marked (harm side)
    conflict_q_all_misinfo_marked  conflict questions whose every misinfo document is marked
    conflict_q_any_correct_marked  conflict questions with at least one correct document marked
    clean_q_any_correct_marked     clean questions with at least one correct document marked
    Pair-level P/R miss this: one correctly directed pair is enough to remove a misinfo
    document, and false flags concentrated on correct-correct pairs remove valid answers.
    Under label_fix="exclude" (and "hand", for unresolved suspects), suspect documents are left
    out of both counts.
    """
    mis_n = mis_m = cor_n = cor_m = cq = cq_all = cq_cor = kq = kq_any = 0
    for ex in examples:
        types = ex.meta.get("doc_types", [])
        skip = set(ex.meta.get("suspect_docs", [])) if ex.meta.get("label_fix") in ("exclude", "hand") else set()
        marked = marked_docs(dets[ex.id].pairs, mode)
        mis = {k for k, t in enumerate(types) if t == "misinfo" and k not in skip}
        cor = {k for k, t in enumerate(types) if t == "correct" and k not in skip}
        mis_n += len(mis); mis_m += len(mis & marked); cor_n += len(cor); cor_m += len(cor & marked)
        if ex.has_conflict and mis:
            cq += 1; cq_all += mis <= marked; cq_cor += bool(cor & marked)
        if ex.has_conflict is False:
            kq += 1; kq_any += bool(cor & marked)
    r = lambda a, b: a / b if b else None
    return {"misinfo_marked": r(mis_m, mis_n), "correct_marked": r(cor_m, cor_n),
            "conflict_q_all_misinfo_marked": r(cq_all, cq), "conflict_q_any_correct_marked": r(cq_cor, cq),
            "clean_q_any_correct_marked": r(kq_any, kq)}
