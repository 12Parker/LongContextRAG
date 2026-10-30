# 10_app_fresh (Appendix §19): Round 2C notes

## Main kinds of change
- Shorter connectives and clauses: "Detections and no-flag answers ran first. We committed ..." → "After detections and
  no-flag answers, we committed ..."; "sharing no subject with it" → "with no shared subject".
- Leaner phrasing with no change in content: "3 points or more" → "at least 3 points"; "the smallest margins that would
  pass" → "the smallest passing margins"; "judged rows" → "rows"; "every response feeding" → "every response in".
- Cause stated first in one clause: "The matcher misses more correct answers under the keys, so it understates ..." →
  "Missing more correct answers under the keys, the matcher understates ...".
- Defined term used in place of a paraphrase: "accuracy with the pages" → "plain accuracy" (the macro is
  `\hcPlainMini`, the no-flag condition with the pages).
- Selection sentence: "Claude Haiku is the detector with the highest predicted effect and the best observed" → "the
  Claude Haiku detector has the highest predicted and observed effect".
- Kept unchanged: H32 holds, H33 fails, H38 "Neither is equivalent within ±3 under the judge", "point estimates",
  "effects over all 400 questions", the "as released" framing, and the trailing "so the" (the text continues in the next part).

## Word count
490 → 475 prose words (−3.1%). check_preservation and check_latex both OK.

## Issues for the authors
1. Table~\ref{tab:hc} labels the column "No flag", and the text now says "plain accuracy" (the glossary's term for
   the no-flag format). Consider using one name in both places.
2. Comparatives are mixed within the section: "\frSelMiniObs\ vs \frMiniDetMiniObs\ points" against
   "gpt-4o-mini \hcCbMini\% against \hcPlainMini\%". Pick one form.
3. "The neutral flag changes accuracy by at most \ahcNeuMini\ points (H37b; within $\pm$3 for gpt-6-luna and Claude
   Haiku)." This could read as an equivalence claim for gpt-6-luna and Claude Haiku. If a TOST passed, say
   "equivalent within ±3". If not, "within ±3" here is looser than the glossary's sense.

## Coherence issues across sections
- Generator names: this section uses "gpt-6-luna" and "gpt-6.1-terra". The brief's model list names only gpt-6.1-terra.
  Check that gpt-6-luna is introduced in setup.tex (or app_repro) before it appears here.
