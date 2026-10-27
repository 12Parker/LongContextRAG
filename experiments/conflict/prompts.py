"""
Prompt construction. Every condition shares the same skeleton; only the conflict
information differs, so differences in outcome are attributable to that information.

Modes
  plain      C0  passages + question
  instruct   C1  + generic "check for contradictions" instruction, no detector
  flag           passage headers + "POTENTIAL CONFLICTS DETECTED" list, in the original
                 CAAG format (_build_context_with_contradictions/_build_prompt)
  flag_instruct  flag + the C1 instruction
  arbitrate      original ARBITRATE format (only the heading differs from flag)
  remove         original REMOVE: drop the second passage of each flagged pair
                 (the original drops the lower retrieval score; with no scores it drops
                  doc_idx_2, which is what this reproduces)
  flag_directional  flag, but passages the detector marks as `suspects` are headed
                 "LIKELY MISINFORMATION" and the conflict list says which side is suspect.
                 Needs a detector that knows direction (oracle; an LLM detector could).
  remove_suspect drop the suspect passage(s) of each flagged pair; falls back to doc_idx_2
                 (as remove) for pairs with no suspect

Note for the paper: in the original code SEPARATE renders exactly the same prompt as FLAG,
so it is not a separate condition here.
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

INSTRUCTION = """Some passages may contradict each other. Before answering, check whether they do.
Ignore passages that appear to contain misinformation, and if a disagreement reflects genuinely different valid answers, report each one.
"""


def _flag_context(passages: list[str], dets: list[Detection], directional: bool = False) -> str:
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
            header += (f" [LIKELY MISINFORMATION; conflicts with passage(s): {with_}]" if k in suspects
                       else f" [POTENTIAL CONFLICT with passage(s): {with_}]")
        parts.append(f"{header}:\n{p}")
    return "\n\n".join(parts)


def _conflict_list(dets: list[Detection], heading: str, directional: bool = False) -> str:
    if not dets:
        return ""
    lines = []
    for d in dets:
        line = f"- Passages {d.i} and {d.j}: {d.type} conflict (confidence: {d.score:.2f})"
        if directional and d.suspects:
            line += f"; passage {' and '.join(map(str, d.suspects))} likely contains misinformation"
        lines.append(line)
    return f"\n{heading}\n" + "\n".join(lines) + "\n"


def build_prompt(question: str, passages: list[str], mode: str,
                 dets: list[Detection] | None = None) -> tuple[str, list[int]]:
    """Returns (prompt, kept_passage_indices)."""
    dets = dets or []
    kept = list(range(len(passages)))
    instruction, conflicts = "", ""

    if mode in ("plain", "instruct"):
        context = "\n\n".join(f"Passage {k}:\n{p}" for k, p in enumerate(passages))
        instruction = INSTRUCTION if mode == "instruct" else ""
    elif mode in ("flag", "flag_instruct", "arbitrate"):
        context = _flag_context(passages, dets)
        heading = "CONFLICTING INFORMATION DETECTED:" if mode == "arbitrate" else "POTENTIAL CONFLICTS DETECTED:"
        conflicts = _conflict_list(dets, heading)
        instruction = INSTRUCTION if mode == "flag_instruct" else ""
    elif mode == "flag_directional":
        context = _flag_context(passages, dets, directional=True)
        conflicts = _conflict_list(dets, "POTENTIAL CONFLICTS DETECTED:", directional=True)
    elif mode in ("remove", "remove_suspect"):
        drop = ({k for d in dets for k in (d.suspects or [d.j])} if mode == "remove_suspect"
                else {d.j for d in dets})
        kept = [k for k in kept if k not in drop]
        context = "\n\n".join(f"Passage {n}:\n{passages[k]}" for n, k in enumerate(kept))
    else:
        raise ValueError(f"unknown mode {mode}")

    prompt = BASE.format(instruction=instruction, context=context, conflicts=conflicts, question=question)
    return prompt, kept
