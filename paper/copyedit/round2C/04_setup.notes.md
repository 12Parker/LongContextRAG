# 04_setup (Experimental setup): Round 2C notes

## Main kinds of change

- **Restatements cut.**
  - Majority variant: cut ", so the false side outnumbers the truth three to one" and "for that fact". The same sentence already gives three wrong-object passages against the one correct passage.
  - Pre-registration: "These hypotheses, H1--H9, are the confirmatory core, registered before any confirmatory data" is folded into the previous sentence as "...corrections and the confirmatory hypotheses, H1--H9". "Before running anything on the remaining 300 questions" already gives the timing.
- **Sentences tightened, content kept.**
  - "Every RAMDocs result in the main text is on this confirmatory split" → "Main-text RAMDocs results all use this split".
  - RAGuard: "behavioural: supporting or misleading according to whether GPT-4 judged" → "behavioural, set by whether GPT-4 judged"; "so on RAGuard the ``oracle''" → "so its ``oracle''". "not a factual oracle" is kept.
  - Statistics: the definition of *family* and the Holm sentence are merged into one sentence. Both the definition and the citation are kept.
  - WikiContradict: "not prefer a side" → "favour no side".
- **Clarity.** "All three (its misinformation, temporal and semantic conflicts)" now reads "ConflictBank's misinformation, temporal and semantic conflicts all". After the cut, "All three" would have followed "three wrong-object passages" and read as if it referred to them.

No numbers, macros, hypothesis IDs, labels, citations, hedges or table rows were changed. Fixes from rounds A, B, C, 2A and 2B are kept, including Round 2B's merged "re-analyses ... as are unregistered comparisons" sentence.

## Word count

- Whole file: 1318 → 1281.
- Prose, by the checker: 1227 → 1190 (−3.0%).
- Prose without the table environment: 1194 → 1157 (−37 words).
- Both checks (`check_preservation.py`, `check_latex.py`) pass.

## Issues for the authors

1. Inconsistent seed notation: "\texttt{gpt-6.1-terra} (seed 0; one sample)" against "Claude Sonnet 5.5 (seed 0)". Either both need "one sample" or neither does.
2. Ambiguous "it": "we also run it on Qwen2.5-7B with gpt-4o-mini generating". The antecedent is "The LLM detector", but "it" could also be read as the crossed design.
3. Table~\ref{tab:evidence} (body not editable): the ConflictBank row cites only "prospective test (H32)", but the prospective test is H32–H36 (glossary, Pre-registration paragraph).
4. "registered before any confirmatory data" was folded into the pre-registration sentence (see above). Restore the explicit phrase if the authors want it stated in exactly those words.

## Coherence issues across sections

- None new. The judge naming in Setup (gpt-6.1-terra agreement judge, Metrics paragraph) against Results (the outside-family judge of H38) is already on Round 2A's list, so it is not repeated here.
