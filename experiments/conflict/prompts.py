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

Skeletons (`skeleton`, run.py --skeleton)
  default    BASE: tells the model to give every valid answer when the passages support more than
             one (RAMDocs has same-name ambiguity questions). Used for every pre-registered run.
  minimal    BASE_MINIMAL: the same prompt without any answer policy (no "give every valid answer"),
             and an answer line that allows but does not invite several answers. Revision ablation:
             does the neutral-flag harm depend on the default skeleton's policy?
  single     BASE_SINGLE: says the question has exactly one correct answer and asks for one
             (§11; for ConflictBank, where that is true by construction)

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

SKELETONS = {"default": BASE, "minimal": BASE_MINIMAL, "single": BASE_SINGLE}

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

INSTRUCTIONS = {"instruct": INSTRUCTION, "instruct_v2": INSTRUCTION_V2, "self_identify": INSTRUCTION_SELF,
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


def build_prompt(question: str, passages: list[str], mode: str,
                 dets: list[Detection] | None = None, style: str = "uniform",
                 skeleton: str = "default") -> tuple[str, list[int]]:
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
    if mode in PROMPT_ONLY_MODES:
        context = "\n\n".join(f"Passage {k}:\n{p}" for k, p in enumerate(passages))
        instruction = INSTRUCTIONS.get(mode, "")
    elif mode in ("flag", "flag_instruct", "flag_instruct_v2", "arbitrate"):
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
