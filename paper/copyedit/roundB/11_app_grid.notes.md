# 11_app_grid — Round B notes

## Main kinds of change
- One point per paragraph: the "Gains rise with direction accuracy; precision matters mainly for removal." paragraph is split at "Precision matters mainly for removal." The first part covers direction accuracy and the second covers precision and removal (an appendix, so the break costs no body space; the head is unchanged and still covers both).
- Actor as subject: "Each synthetic detector ... is built from the gold labels" -> "We build each synthetic detector ... from the gold labels".
- Bare "this"/"that" given nouns: "That rate is dataset-wide, so" -> "Because the rate is dataset-wide,"; "This keeps comparisons ... paired" -> "The nesting keeps ..."; "so this is a dose-response" -> "so the pattern is a dose-response".
- Parallel form, shorter: "Locating detectors at the document level puts X of N effects inside; locating them at the pair level puts Y inside" -> "Document-level locations put X of N effects inside, and pair-level locations Y."
- No paragraph head or caption claim changed.

Word count: 789 before, 782 after (prose 789 -> 782). Both checks pass (check_preservation.py, check_latex.py).

## Issues for the authors
Round A's seven issues (r < 1 cells not located, "precision matters mainly" before its numbers, unmacro'd "at most two points", "twice", position of the post hoc axes disclosure, far-away fig:placement, which detectors give no direction) still apply and are not repeated. New ones:
1. "Only the cells with perfect direction are significant on all questions (directional flag \gDirOneDOneAll, removal \gRemOneDOneAll\ points at perfect precision)." The parenthesis gives only the perfect-precision cells, so a reader cannot tell whether perfect-direction cells at lower precision are also significant.
2. "Neighbouring cells at direction accuracy 0.75 have similar point estimates on conflict questions" — no number backs "similar"; a pointer to fig:grid's values would help.
3. "If the grid only described synthetic detectors, it would be of limited use." An empty motivating opener; it could be cut (about ten words) if the authors want the paragraph to open on H6.

## Coherence issues across sections
- "matched cells" (H4c, here) vs "distinct comparisons" (H4a, same paragraph) vs "distinct cells" (H4b): three counts with three nouns; method.tex/results.tex should use the same terms if they report H4.
- None new otherwise; Round A's (Holm family of 36 vs the 12 cells shown in fig:grid; "one-way strict accuracy" not in the glossary) stand.
