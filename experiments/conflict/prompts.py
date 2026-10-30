"""
Prompt construction. Every condition shares the same skeleton; only the conflict
information differs, so differences in outcome are attributable to that information.

Modes
  plain      C0  passages + question
  instruct   C1  + generic "check for contradictions" instruction, no detector (v1 wording)
  instruct_v2 C1b + a second wording (INSTRUCTION_V2). v1 ends "if a disagreement reflects
                 genuinely different valid answers, report each one", which in the pilots licensed
                 listing both sides of a contradiction; v2 asks for a reliability judgement instead.
                 Report both so the prompt-only baseline does not hinge on one wording.
  flag           passage headers + "POTENTIAL CONFLICTS DETECTED" list, in the original
                 CAAG format (_build_context_with_contradictions/_build_prompt)
  flag_instruct  flag + the C1 instruction (v1); flag_instruct_v2 uses v2
  arbitrate      original ARBITRATE format (only the heading differs from flag)
  remove         original REMOVE: drop the second passage of each flagged pair
                 (the original drops the lower retrieval score; with no scores it drops
                  doc_idx_2, which is what this reproduces)
  flag_directional  flag, but passages the detector marks as `suspects` are headed
                 "LIKELY MISINFORMATION" and the conflict list says which side is suspect.
                 Needs a detector that knows direction (oracle; an LLM detector could).
  flag_directional_instruct_v2  flag_directional + the v2 reliability instruction (does direction
                 add anything once the model is told to resolve the conflict?)
  remove_suspect drop the suspect passage(s) of each flagged pair; falls back to doc_idx_2
                 (as remove) for pairs with no suspect
  flag_onewrong  neutral flag that also says exactly one passage of each flagged pair contains
                 misinformation, without saying which (§11: separates "one side is false" from
                 "this side is false"; same headers' position and imperative force as flag_directional)
  self_identify  no detector: one call that first names the passages it judges to be misinformation,
                 then answers from the rest (§11: compute-matched single-call baseline)
  closedbook     no passages at all (§11: what the model knows without retrieval)
  flag_pick      neutral flag + "decide which side is better supported and answer with it" (§13: a hedged
                 neutral wording that asks the model to commit)
  flag_nopairs, flag_onewrong_nopairs, flag_directional_nopairs
                 the same flags without pair structure (§13): flagged passages are marked, but no header or
                 list says which passage conflicts with which, so a passage that appears in every flagged pair
                 cannot be singled out from the flag layout
  answer_key     plain + "the answer X is false" for each wrong answer (§13: the directional flag's
                 information given as an answer key, with no passage marked)
  verdict_key    plain + "the answer X is false" for each answer a suspect passage of the detector supports, no
                 passage marked (§16: the detector's judgement passed as an answer key; run.py supplies the answers)
  hint           plain + 'Another model that read these passages answered: "X"', X being the detector
                 model's own plain answer (§15: is a detector's gain just a stronger model's answer forwarded?)

Skeletons (`skeleton`, run.py --skeleton)
  default    BASE: tells the model to give every valid answer when the passages support more than
             one (RAMDocs has same-name ambiguity questions). Used for every pre-registered run.
  minimal    BASE_MINIMAL: the same prompt without any answer policy (no "give every valid answer"),
             and an answer line that allows but does not invite several answers. Revision ablation:
             does the neutral-flag harm depend on the default skeleton's policy?
  single     BASE_SINGLE: says the question has exactly one correct answer and asks for one
             (§11; for ConflictBank, where that is true by construction)
  forced     BASE_FORCED: as single, but "unknown" is not allowed (§13: is the neutral-flag harm only
             abstention?)

Note for the paper: in the original code SEPARATE renders exactly the same prompt as FLAG,
so it is not a separate condition here.

Flag styles (`style`, run.py --flag-style)
  uniform    every flag renders as "- Passages i and j: possible contradiction", whatever the
             detector, so detectors differ only in WHICH pairs they flag (default; use for any
             comparison across detectors)
  original   CAAG's "- Passages i and j: <type> conflict (confidence: <score>)"; detectors then
             also differ in type labels and score scale (CAAG ~0.4, LLM/oracle 1.00). Use only to
             reproduce the original CAAG format (C2).
"""
from __future__ import annotations

from .detectors import Detection

BASE = """Answer the question using the passages below.
If the passages support more than one valid answer (for example, different people or things share the same name), give every valid answer.
If no passage answers the question, the answer is "unknown".
{instruction}
{context}
{conflicts}
Question: {question}

Give a brief explanation, then end with one line in exactly this format:
Final answer(s): <answer 1> | <answer 2> | ..."""

BASE_MINIMAL = """Answer the question using the passages below.
If no passage answers the question, the answer is "unknown".
{instruction}
{context}
{conflicts}
Question: {question}

Give a brief explanation, then end with one line in exactly this format:
Final answer(s): <your answer; if there is more than one, separate them with |>"""

BASE_SINGLE = """Answer the question using the passages below. The question has exactly one correct answer.
If no passage answers the question, the answer is "unknown".
{instruction}
{context}
{conflicts}
Question: {question}

Give a brief explanation, then end with one line in exactly this format:
Final answer(s): <your one answer>"""

BASE_FORCED = """Answer the question using the passages below. The question has exactly one correct answer.
You must give your best answer; "unknown" is not allowed.
{instruction}
{context}
{conflicts}
Question: {question}

Give a brief explanation, then end with one line in exactly this format:
Final answer(s): <your one answer>"""

SKELETONS = {"default": BASE, "minimal": BASE_MINIMAL, "single": BASE_SINGLE, "forced": BASE_FORCED}

# closed-book: same answer line as each skeleton, no passages
CLOSED_BOOK = {
    "default": """Answer the question from your own knowledge.
If the question has more than one valid answer (for example, different people or things share the same name), give every valid answer.
Question: {question}

Give a brief explanation, then end with one line in exactly this format:
Final answer(s): <answer 1> | <answer 2> | ...""",
    "minimal": """Answer the question from your own knowledge.
Question: {question}

Give a brief explanation, then end with one line in exactly this format:
Final answer(s): <your answer; if there is more than one, separate them with |>""",
    "forced": """Answer the question from your own knowledge. The question has exactly one correct answer.
You must give your best answer; "unknown" is not allowed.
Question: {question}

Give a brief explanation, then end with one line in exactly this format:
Final answer(s): <your one answer>""",
    "single": """Answer the question from your own knowledge. The question has exactly one correct answer.
Question: {question}

Give a brief explanation, then end with one line in exactly this format:
Final answer(s): <your one answer>"""}

INSTRUCTION = """Some passages may contradict each other. Before answering, check whether they do.
Ignore passages that appear to contain misinformation, and if a disagreement reflects genuinely different valid answers, report each one.
"""

INSTRUCTION_V2 = """Some passages may contain false information. Before answering, decide which passages are reliable by checking them against each other.
Do not include an answer that is supported only by a passage you judge to be unreliable.
"""

INSTRUCTION_SELF = """Some passages may contain misinformation. First decide which passages, if any, contain misinformation, by checking them against each other and against what you know, and write one line in this format:
Suspect passages: <passage numbers, or none>
Then answer the question using only the other passages.
"""

INSTRUCTION_PICK = """When flagged passages conflict, decide which side is better supported by the other passages and by what you know, and answer with that side only.
"""

INSTRUCTIONS = {"flag_pick": INSTRUCTION_PICK, "instruct": INSTRUCTION, "instruct_v2": INSTRUCTION_V2, "self_identify": INSTRUCTION_SELF,
                "flag_instruct": INSTRUCTION, "flag_instruct_v2": INSTRUCTION_V2,
                "flag_directional_instruct_v2": INSTRUCTION_V2}
PROMPT_ONLY_MODES = ("plain", "instruct", "instruct_v2", "self_identify", "closedbook")


def _flag_context(passages: list[str], dets: list[Detection], directional: bool = False,
                  onewrong: bool = False) -> str:
    partners: dict[int, list[int]] = {}
    for d in dets:
        partners.setdefault(d.i, []).append(d.j)
        partners.setdefault(d.j, []).append(d.i)
    suspects = {k for d in dets for k in d.suspects} if directional else set()
    parts = []
    for k, p in enumerate(passages):
        header = f"Passage {k}"
        if k in partners:
            with_ = ", ".join(map(str, sorted(partners[k])))
            if k in suspects:
                header += f" [LIKELY MISINFORMATION; conflicts with passage(s): {with_}]"
            elif onewrong:
                header += f" [CONFLICT with passage(s): {with_}; one side contains misinformation]"
            else:
                header += f" [POTENTIAL CONFLICT with passage(s): {with_}]"
        parts.append(f"{header}:\n{p}")
    return "\n\n".join(parts)


FLAG_STYLES = ("uniform", "original")


def _conflict_list(dets: list[Detection], heading: str, directional: bool = False,
                   style: str = "uniform", onewrong: bool = False) -> str:
    if not dets:
        return ""
    lines = []
    for d in dets:
        line = (f"- Passages {d.i} and {d.j}: {d.type} conflict (confidence: {d.score:.2f})" if style == "original"
                else f"- Passages {d.i} and {d.j}: possible contradiction")
        if directional and d.suspects:
            line += f"; passage {' and '.join(map(str, d.suspects))} likely contains misinformation"
        elif onewrong:
            line += "; exactly one of the two contains misinformation (it is not known which)"
        lines.append(line)
    return f"\n{heading}\n" + "\n".join(lines) + "\n"


def _marked_context(passages: list[str], dets: list[Detection], kind: str) -> tuple[str, str]:
    """Flags without pair structure (§13): which passages are involved, not which pairs."""
    flagged = {k for d in dets for k in (d.i, d.j)}
    suspects = {k for d in dets for k in d.suspects}
    parts = []
    for k, p in enumerate(passages):
        header = f"Passage {k}"
        if kind == "directional" and k in suspects:
            header += " [LIKELY MISINFORMATION]"
        elif kind == "onewrong" and k in flagged:
            header += " [IN CONFLICT; some marked passages contain misinformation]"
        elif kind == "neutral" and k in flagged:
            header += " [POTENTIAL CONFLICT]"
        parts.append(f"{header}:\n{p}")
    note = {"neutral": "Some of the marked passages conflict with each other.",
            "onewrong": "Some of the marked passages contain misinformation (it is not known which).",
            "directional": "Passages marked LIKELY MISINFORMATION probably contain false information."}[kind]
    conflicts = f"\nPOTENTIAL CONFLICTS DETECTED:\n{note}\n" if (flagged if kind != "directional" else suspects) else ""
    return "\n\n".join(parts), conflicts


def build_prompt(question: str, passages: list[str], mode: str,
                 dets: list[Detection] | None = None, style: str = "uniform",
                 skeleton: str = "default", wrong_answers: list[str] | None = None,
                 hint: list[str] | None = None) -> tuple[str, list[int]]:
    """Returns (prompt, kept_passage_indices)."""
    if style not in FLAG_STYLES:
        raise ValueError(f"style must be one of {FLAG_STYLES}")
    if skeleton not in SKELETONS:
        raise ValueError(f"skeleton must be one of {tuple(SKELETONS)}")
    dets = dets or []
    kept = list(range(len(passages)))
    instruction, conflicts = "", ""

    if mode == "closedbook":
        return CLOSED_BOOK[skeleton].format(question=question), []
    if mode == "verdict_key":   # same rendering as answer_key; the answers come from the detector's suspects
        mode = "answer_key"
    if mode == "answer_key":
        context = "\n\n".join(f"Passage {k}:\n{p}" for k, p in enumerate(passages))
        if wrong_answers:
            conflicts = "\nNOTE: " + " ".join(f'The answer "{w}" is false.' for w in wrong_answers) + "\n"
        prompt = SKELETONS[skeleton].format(instruction="", context=context, conflicts=conflicts, question=question)
        return prompt, kept
    if mode == "hint":
        if hint is None:
            raise ValueError("mode hint needs the other model's answer (run.py --hint-run)")
        context = "\n\n".join(f"Passage {k}:\n{p}" for k, p in enumerate(passages))
        conflicts = f'\nNOTE: Another model that read these passages answered: "{" | ".join(hint) or "unknown"}".\n'
        return SKELETONS[skeleton].format(instruction="", context=context, conflicts=conflicts, question=question), kept
    if mode in ("flag_nopairs", "flag_onewrong_nopairs", "flag_directional_nopairs"):
        kind = {"flag_nopairs": "neutral", "flag_onewrong_nopairs": "onewrong", "flag_directional_nopairs": "directional"}[mode]
        context, conflicts = _marked_context(passages, dets, kind)
        prompt = SKELETONS[skeleton].format(instruction="", context=context, conflicts=conflicts, question=question)
        return prompt, kept
    if mode in PROMPT_ONLY_MODES:
        context = "\n\n".join(f"Passage {k}:\n{p}" for k, p in enumerate(passages))
        instruction = INSTRUCTIONS.get(mode, "")
    elif mode in ("flag", "flag_instruct", "flag_instruct_v2", "arbitrate", "flag_pick"):
        context = _flag_context(passages, dets)
        heading = "CONFLICTING INFORMATION DETECTED:" if mode == "arbitrate" else "POTENTIAL CONFLICTS DETECTED:"
        conflicts = _conflict_list(dets, heading, style=style)
        instruction = INSTRUCTIONS.get(mode, "")
    elif mode == "flag_onewrong":
        context = _flag_context(passages, dets, onewrong=True)
        conflicts = _conflict_list(dets, "POTENTIAL CONFLICTS DETECTED:", style=style, onewrong=True)
    elif mode in ("flag_directional", "flag_directional_instruct_v2"):
        context = _flag_context(passages, dets, directional=True)
        conflicts = _conflict_list(dets, "POTENTIAL CONFLICTS DETECTED:", directional=True, style=style)
        instruction = INSTRUCTIONS.get(mode, "")
    elif mode in ("remove", "remove_suspect"):
        drop = ({k for d in dets for k in (d.suspects or [d.j])} if mode == "remove_suspect"
                else {d.j for d in dets})
        kept = [k for k in kept if k not in drop]
        context = "\n\n".join(f"Passage {n}:\n{passages[k]}" for n, k in enumerate(kept))
    else:
        raise ValueError(f"unknown mode {mode}")

    prompt = SKELETONS[skeleton].format(instruction=instruction, context=context, conflicts=conflicts, question=question)
    return prompt, kept
