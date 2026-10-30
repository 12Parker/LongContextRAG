# 03_method: Round A notes

## Main kinds of change
- Split the opening paragraph: the detector/format abstraction and the prompt skeletons are now two paragraphs (no wording change).
- Format head "Flag." -> "Flag (neutral)." so the glossary term "neutral flag" (used later in the same list, "Pick: the neutral flag plus ...") is defined where the format is defined. "As Flag" references kept.
- One-wrong flag: colon clause split into its own sentence; removed the repeated "but" ("It conveys that the conflict is misinformation, not its direction.").
- Random: "a placebo that matches" -> "a placebo matching".
- Synthetic detectors: "(36 cells)" -> "(36 grid cells)" so the term "grid" used in Appendix app:grid is introduced here.
- All quoted prompt strings, \texttt{} headers, model names, math and definitional details are unchanged.

## Word count
651 -> 651 (prose 651 -> 651). Both checks pass.

## Issues for the authors
1. Long sentence (about 40 words) not split because a split adds words: "\item[LLM] is a single call that lists contradicting pairs, telling them apart from ambiguity, complementary and temporal relations, and names the side more likely to be misinformation (or ``both'' or ``unclear''), as JSON."
2. Undefined forward reference: "in the crossed design gpt-4.1, gpt-6-luna, gpt-6.1-terra, Claude Haiku 5.5 and Claude Sonnet 5.5 run the same prompt." The crossed design is defined only in Setup; a pointer "(H24, \S\ref{sec:setup})" contains a digit, so it is listed here instead.
3. Directional flag with a non-directional detector: the Remove item says what happens "For a pair without a suspect", but the Directional flag item does not say how a flagged pair with no suspect (CAAG, NLI, or LLM "both"/"unclear") is rendered. Worth one clause if space allows.
4. "\emph{Answer key}: plain plus ``The answer $X$ is false'' for each wrong answer" -- the source of the wrong answers (gold labels here; the detector's verdict in the H30 verdict key) is not stated. The verdict key (H30) and own-answer hint (H27), both glossary terms, are not defined in this section; the own-answer hint is described in Setup, the verdict key apparently only in Results.
5. Label inconsistency: the format is headed "Remove (suspect)" but the paper's prose (intro, Synthetic detectors, results, app_grid) calls it "removal". Heads left as is, since the appendix prompts may use the head.
6. "Single call ... at half the calls of a detector pipeline" relies on Setup's later definition "A detector pipeline counts as one detector call plus one generator call."

## Coherence issues across sections
- "forced choice" (intro.tex line 25 "under forced choice the effect is small"; results.tex lines 36, 48, 58) names what this section calls the "forced skeleton"; the phrase is never tied to the skeleton here, and the Pick control is also a choice-forcing control. Suggest "forced skeleton" throughout or a defining phrase here.
- Intro.tex line 12 describes the neutral flag as a ``these passages may conflict'' flag, while this section's actual rendered text is \texttt{[POTENTIAL CONFLICT with passage(s): j]} / "possible contradiction". A reader may take the intro's quote as the prompt text.
- Setup ("six detectors and six generators form \hTwentyFourCells\ pairs") vs this section, which lists the crossed-design LLM detectors (gpt-4o-mini plus five) but not that the six detectors are all LLM detectors; consistent, but only if the reader counts.
