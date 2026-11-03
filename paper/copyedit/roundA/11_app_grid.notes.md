# 11_app_grid — Round A notes

## Main kinds of change
- Paragraph heads state the finding: "Results." -> "Gains rise with direction accuracy; precision matters mainly for removal."; "Does the grid predict real detectors?" -> "Both summaries place most real effects inside the prediction interval." (and removed the stray blank line after that head).
- Split the dose-response sentence and the H4a sentence into shorter sentences.
- Removed repetition: the H6 paragraph re-defined the document-level and pair-level axes already defined in "Placing real detectors on the grid"; now "at the document level" / "at the pair level".
- Terminology: "mode" -> "format" (fig:grid caption), "interface" -> "format", "conflict-free questions" -> "clean questions", "(Remove)"/"(Directional flag)" -> lowercase format names; "Removal hurt" -> "hurts" (tense).
- Wording fix: "false flags make the model delete correct passages" -> "Under removal, false flags delete correct passages" (removal is done by the pipeline, not the generator, per the format definition).

Word count: 803 before, 789 after (prose 803 -> 789). Both checks pass.

## Issues for the authors
1. "Figure~\ref{fig:grid} replaces the oracle with synthetic detectors of known quality." The figure shows recall 1 only; the text does not say where the r = 0.25 / 0.5 cells are reported.
2. "Precision matters mainly for removal." This claim is stated without a number; the supporting numbers come later (clean-question costs). Fine, but could be tied explicitly.
3. "while synthetic false directional flags cost at most two points" — spelled-out number with no macro; check it is generated or consistent with the table.
4. "Removal is the more powerful and the more fragile interface [now: format]: its best cell gains twice the directional flag's on all questions" — "twice" is derived from \gRemOneDOneAll (+13) vs \gDirOneDOneAll (+6), which is slightly more than twice; acceptable but authors may prefer "about twice" (qualifier change, not made).
5. "We chose the document-level axes after seeing the LLM detector on the pilot split." This post hoc disclosure comes after the H6 results; consider moving it to the placement-method paragraph, where the axes are defined (not moved, as it changes where the hedge sits).
6. "Figure~\ref{fig:placement} (Appendix~\ref{app:extra})" — the figure is defined in another appendix section, far from this one; readers must jump.
7. fig:tradeoff caption: "Without direction, removal deletes the second passage of each pair." The caption does not say which detectors give no direction (CAAG, NLI, Random).

## Coherence issues across sections
- method.tex (sec:synthetic) gives the grid levels p in {0.25,0.5,0.75,1}, r in {0.25,0.5,1}, d in {0.5,0.75,1}; this appendix's Construction paragraph refers to Section sec:synthetic for them, consistent. But fig:grid's Holm family is "36 cells of that format and subset" while the figure shows recall 1 only (12 cells); worth stating in the caption.
- limitations.tex ("Direction and the grid"): "The grid predicts real detectors only roughly"; this appendix closes with "a useful first estimate". Consistent in substance; heads now say "most real effects inside the prediction interval", in line with \hSixDocIn/\hSixPairIn of \hSixN.
- "one-way strict accuracy" (both captions here) vs "accuracy" / "one-way metric" in setup.tex; "strict" is not in the glossary.
