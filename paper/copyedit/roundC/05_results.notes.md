# 05_results — Round C notes

## Main kinds of change
- **Defined terms instead of paraphrases.**
  - "the share of gpt-4o-mini's answers that contain the misinformation" → "gpt-4o-mini's share of misinformed answers" (the H21 sentence already says "misinformed answers").
  - "Telling the model ``the answer $X$ is false'', with no passage marked" → "An answer key (``the answer $X$ is false'', no passage marked)" (method defines *answer key*).
  - "When ``unknown'' is not allowed (H20)" → "Under forced choice (H20)" (the term the section uses elsewhere).
- **Restated definitions cut.** Only definitions method.tex already gives:
  - "which states no answer policy" (minimal prompt);
  - "that names the correct side" (inverted oracle);
  - "that uses each cell's coverage and no-flag accuracy", which the follow-rate account's variable list then defines again.
- **Restated heading.** "Its mechanism depends on the answer policy (Table~\ref{tab:risk})" was folded into the H2/H9 sentence as "(H9; Table~\ref{tab:risk} splits it by answer policy)", because the subsection heading already says it.
- **Needless words.**
  - "the neutral flag" → "the flag" inside the neutral-flag subsection;
  - "still", "the" and "detector--generator" (H24's cells, already named in the caption and Setup);
  - "with a bootstrap that resamples" → "(bootstrapping … jointly across them)";
  - the parallel "under X prompts" list compressed;
  - "carries over to … only if that detector" → "valid for … only if it".
- **Figure 1 caption.** Two opening sentences became one. "Table~\ref{tab:cells} lists the cells" became a parenthetical, and the follow-rate wording was shortened ("own rates", "mean rates").
- **Kept as they were:**
  - every macro, with its metric and sign verb ("lowers … \ahThirtyOneBMax", "costs … \aadvDirConf", "lowers … \ahTwentyEightWc*", "change … by \selHaikuMiniWrongGain" were not touched);
  - every exploratory / retrospective / descriptive / registered label;
  - "codings chosen after seeing data" and "defined after the cells were seen".
- Round A/B fixes intact (H29 "does not significantly change", "Across settings." head).

## Word count
Prose 1361 → 1292 (−69, −5.1%). Both checks pass.

## Issues for the authors
1. "The flag changes accuracy insignificantly for gpt-6.1-terra, …" — Round B changed the same construction for H29 to "does not significantly change" (glossary: "not significant"); this one was left as is because the fix adds words.
2. "Tables~\ref{tab:cb} and \ref{tab:main} give every condition and the hypothesis tables every registered test." — "the hypothesis tables" has no `\ref`; a reader cannot find them from here.
3. The H30 sentence repeats the answer key's quoted wording ("``the answer $X$ is false'' for the answer its suspect passage supports") one subsection after H23 defines it; the inline-math check stops the repeat from being dropped. Consider "passed as an answer key on the answer its suspect passage supports".
4. "Every OpenAI and Qwen generator moves the same way; both Claude models the other way" — no significance statement for the misinformation-share shift (cf. Round A's significance list; this instance was not listed).

## Coherence issues across sections
- Results, H31 paragraph, "break-even direction precision $p^\star$" vs Discussion, "Such advice pays only above a break-even precision" (the glossary term is *break-even direction precision*).
