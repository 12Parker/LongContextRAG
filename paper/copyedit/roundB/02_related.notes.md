# 02_related: Round B notes

## Main kinds of change
- Split two sentences that joined unrelated works with "and" (huang2025trust / zhou2023context; CONFLICTS / WikiContradict), so each sentence tells one work's story.
- Replaced ambiguous pronouns with the noun: "which makes it more robust" -> "making the generator more robust"; "handle them without an explicit contradiction detector" -> "handle conflicts without ...".
- Williams: cut empty opener "What we add is" -> "We add"; passive "how a detector's output should be passed" -> "how to pass a detector's output"; "models conflicts at the level of facts" -> "models fact-level conflicts".
- Contribution list: the dangling second "and why it can misjudge" -> "showing why it can misjudge" (list stays parallel; no item dropped).
- "and accuracy still falls" -> "yet accuracy still falls" (marks the contrast; claim and "because" unchanged, see Round A issue 1).
- "a warning, and some vary" -> "a warning; some vary" (two points, one sentence boundary).
- Every citation kept, each with the same finding attributed.

## Word count
644 -> 633 (prose 644 -> 633). Both checks pass.

## Issues for the authors
1. "Advice improves a decision-maker only when its errors fall where the decision-maker is right less often \citep{bansal2021accurate,mozannar2020defer}, the condition our break-even makes quantitative." "our break-even" uses the defined label as a bare noun; the glossary term is "break-even direction precision", and the paper frames it as a calibration for randomised flags on true pairs, not a decision rule. "makes quantitative" may read as stronger than that framing.
2. "answer-key controls that separate the verdict from the marking, its cost on clean questions" -- "its" still has no clear antecedent (verdict, direction or flag); Round A issue 3, unresolved because naming it adds words.
3. "TACS \citep{yu2024tacs} detects untruthful context and masks it, close to our removal format." calls it the "removal format", while Method heads the format "Remove (suspect)" (see 03_method notes, Round A issue 5).

## Coherence issues across sections
- None new beyond Round A's (neutral-flag "accuracy still falls" vs intro "does not reliably help"; break-even framing). Related: issue 1 above vs intro.tex third findings bullet ("It calibrates flags on true pairs. For real detectors precision alone can mislead").
