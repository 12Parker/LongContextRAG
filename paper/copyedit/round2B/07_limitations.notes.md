# 07_limitations (Limitations and Ethics Statement): Round 2B notes (sentence and paragraph craft)

## Main kinds of change
- **Pronouns given their noun** where the antecedent was unclear (Data and power):
  - "so its oracle is a label transfer" -> "so RAGuard's oracle is a label transfer" (the sentence's own subject is
    the LLM audit, so "its" could read as the audit's);
  - "so they make a weak retrieval setting" -> "so the posts make a weak retrieval setting" ("they" followed
    "Every generator", so it read as the generators).
- **Dangling modifier fixed (new H38 text, Metric):** "Re-scored by that judge over all \jdN\ responses, neither
  pooled equivalence holds" (the equivalence was not re-scored; the responses were) -> "Under that judge's re-score
  of all \jdN\ responses, neither pooled equivalence holds". Same words, same macros, "within $\pm$3", "H38" and
  "under the matcher only" unchanged.
- **Ethics:** "the RAGuard audit sheet we release replaces" -> "our released RAGuard audit sheet replaces" (one word
  shorter, pays for "the posts"; "RAGuard" kept so it is not confused with the judge audit sheet of the Metric
  paragraph). Round 2A's coordinator fix (handles replaced with placeholders) is kept.
- No limitation cut or softened; every hedge, registration label, macro, number and hypothesis ID kept. Checked
  `\zcRgMiniConflict` (64.9, closed-book) before `\zpRgMiniConflict` (36.7, with posts): the order matches
  "more accurate closed-book than with RAGuard's posts".

## Word count
Prose 1241 -> 1241. check_preservation and check_latex: OK.

## Issues for the authors
1. "Position, surface style and deeper regularities of generated text are therefore not ruled out as sources of
   their direction." "their" must reach back past "equivalence" and "four changes" to "the detectors"; "the
   detectors' direction" is clearer but costs a word, which I could not find without cutting a limitation.
2. "The two audit sheets": Limitations calls one "the paired, blinded audit sheet we release" (Metric) and the
   other "our released RAGuard audit sheet" (Ethics). A reader may take them for one sheet; a distinguishing name
   for the judge/human-audit sheet (setup.tex calls it "an unlabelled 200-output human-audit sheet") would help,
   but adding the digit-bearing name is not allowed here.

## Coherence issues across sections
- **Audit-sheet naming.** limitations.tex Metric paragraph ("the paired, blinded audit sheet we release"), setup.tex
  l.94 ("an unlabelled 200-output human-audit sheet") and appendix.tex l.217 ("A blinded, paired human-audit sheet")
  use three wordings for the scorer audit sheet, while Ethics names a separate RAGuard audit sheet. One name for
  each sheet would avoid confusion.
