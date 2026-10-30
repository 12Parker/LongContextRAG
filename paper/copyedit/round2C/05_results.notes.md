# 05_results — Round 2C notes

## Main kinds of change

- **Restatement cut.** The opening registration sentence no longer repeats Setup's definitions ("later", "of data already seen"; Setup says "Re-analyses of seen data"). "told that two passages conflict" (the flag's own content) cut; "the model reports both sides" keeps the mechanism.
- **Compressed lists and parentheticals.** The by-prompt pooled effects are now a parenthetical list ("By prompt, it is X (minimal), Y (multi-answer) ..."); bootstrapping, judge re-score ("re-scoring all \jdN\ pooled-test responses") and caption ("open: mean rates, no H31 run"; "own rates"; "a wrong one (the correct side)") shortened.
- **Round 2B fixes applied.** Round 2B's TRIAGE records two Results fixes that were not in this round's copy. Both are now in:
  - H30: "equivalent within $\pm$3 under the matcher but not under the judge (\jdThirtyJ\ \jdThirtyJCI; H38a)";
  - "does not significantly change accuracy" in place of "changes accuracy insignificantly".
  The first is why both checks report one missing `$\pm$` / "3,". Round 2B's coordinator accepted that difference, and the prompt asks to keep this wording.
- **Answer-key sentence (2B issue 4).** "The key says ``the answer $X$ is false'' for the answer the benchmark labels its suspect passage as supporting" → "The key's $X$ is the answer the detector's suspect passage supports, by the benchmark's labels". The quote is still given verbatim in the H23 paragraph, and the "its" antecedent problem is fixed.
- **Small edits:**
  - H30 subject: "Passed as an answer key, the verdict matches" (the head already says "the detector's verdict");
  - H32: "with effects predicted from detections and no-flag answers before any flagged answer existed" (the head says "Frozen in advance");
  - "the string matcher" → "the matcher";
  - "where giving both sides is correct";
  - the redundant "also", "fitted", "or for" and "the two" are cut.
- **Kept, as instructed:** "(H33 fails)", the H19 sentence after the H24 account, and "under the matcher but not under the judge". Every hedge, registration label, macro and hypothesis ID is unchanged.

## Word count

Prose 1277 → 1238 (−39, −3.1%; target 3%). `check_latex.py` and `check_preservation.py` pass, except for the accepted H38a fold noted above (one `$\pm$3` folded).

## Issues for the authors

1. "matches the directional flag (H30, no cell significant): pooled ... equivalent within $\pm$3 under the matcher but not under the judge". The head says the verdict "carries most of its contribution", and "matches" is unqualified, but under the judge the pooled equivalence fails. Consider "matches ... under the matcher".
2. "naming the false side for a random \nRandRight\, the correct side for the other \nRandWrong." "a random N" is terse: it does not say questions. "for a random \nRandRight\ questions" would add a word.
3. "The sign is right in all \frBigN\ cells predicted at $\ge$3 points". This still reads as gains only (2B issue 2, unresolved).
4. "On 400 unseen ConflictBank questions". The name ConflictBank-fresh is still never used in Results (2B issue 6).
5. "\paragraph{Genuine conflicts and baselines.}" The baselines get only a pointer ("Baselines: Appendix~\ref{app:extra}."). The head could name the H28/H29 findings instead.
6. "(and of answers correct under the benchmark's own rubric by \awikiOrigMiniEffPt\ to \awikiOrigLunaEffPt;" This has no "points" unit after the macros; the earlier range in the sentence has one.

## Coherence issues across sections

- Results (H30 paragraph) now says "the verdict" for the detector's judgement passed as an answer key. The appendix (10_app_fresh, H38 paragraph) says "the verdict key". Results still never defines the term "verdict key" (2B note).
- Results, H23: "Equivalence rests on the matcher". The abstract and intro say "string metric" (2A left item); Results now uses only "matcher".
