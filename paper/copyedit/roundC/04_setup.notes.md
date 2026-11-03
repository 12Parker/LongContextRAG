# 04_setup — Round C notes

## Main kinds of change
- Merged restating sentences: the ConflictBank majority-variant passage description folded into a parenthesis; RAGuard's "labels do not verify" and "oracle is a label transfer" joined into one sentence; the WikiContradict sentence "We score whether an answer gives both." deleted (Metrics states the same scoring rule).
- Used defined terms instead of paraphrase: "own-answer hint" (H27 control), "misinformation share", "abstention share"; "conflict"/"clean" question definitions tightened.
- Removed needless words: "at all", "current", "own", "evidence", "Anthropic's", "the original rubric is", "; Qwen ran locally" (restates "(4-bit, local)"), "since they reason before answering".
- Registration statements (pilot/confirmatory split, H1–H9 core, eight addenda, last two addenda's re-analyses, unregistered final round, ninth addendum §17, "All of these are exploratory", commit history, deviations pointer) kept in meaning and scope; only "that were not registered" -> "Unregistered", "at all" dropped, "its own runs" -> "its runs", "these hypotheses" -> "these".

## Word count
Prose 1202 -> 1141 (-5.1%). Both checks pass (check_preservation, check_latex).

## Issues for the authors
- "we also run it on Qwen2.5-7B with gpt-4o-mini generating." — "it" (the LLM detector) is only clear on second reading, and the benchmark is not stated.
- "We rerun the oracle and LLM-detector conditions with the minimal skeleton." — sits after the ConflictBank sentence; unclear whether this is ConflictBank only or all benchmarks.
- "The reasoning and Claude models run at their default temperature with up to 4,096 output tokens." — the dropped rationale ("since they reason before answering") was the only justification for the 4,096 budget; restore if reviewers would want it.

## Coherence issues across sections
- Setup "Pre-registration and splits" ("The repository's commit history is the only timestamp.") and App. reproducibility (10_app_repro, "The repository's commit history is the only timestamp; we did not use a third-party registry") say the same thing; one could be a pointer, but both are registration statements so I left them.
