# 09_app_further_semantic: Round 2A notes

## Main kinds of change
- **Moved a caveat.** "Given the gold and wrong answers, the judge tests the scorer, not the benchmark labels." now sits right after the judge is introduced (and after the Gemini 2.5 Flash deviation). It qualifies the whole re-score. Before, it fell between the H38 pointer and the audit-sheet sentence. This was Round B's suggestion.
- **Split long sentences.**
  - The fig:answers caption's last sentence is now two.
  - "Almost all disagreements … and depend on condition" is now two.
  - "One small contrast changes sign (…), and two whose matcher intervals …" is now two.
  - The §18 per-question agreement sentence is now two. Its last parenthesis "(X% and Y% on those cells)" did not say whose rate it gave. It now reads "where flag and key disagree on X% and Y%".
- **Parallel wording.** "RAMDocs oracle flag vs answer key" now reads "against answer key", to match "LLM flag against its verdict key" in the same sentence. "judged over all their responses, both fail" now reads "re-scored on all their responses, neither holds", which is the wording of App. fresh (H38) and Limitations. In the WikiContradict rubric, "acknowledges that they conflict" now reads "acknowledges the conflict", which matches the paragraph's last sentence.
- **Left unchanged.**
  - Every paragraph head.
  - Every number and macro.
  - All hedges ("unadjusted", "exploratory", "One untested reading", "significant after Holm").
  - Round A/B/C fixes, including "Given the gold and wrong answers" and "human-audit sheet".

## Word count
Prose 1256 → 1256 (total 1256 → 1256). Both checks pass (`check_preservation.py`, `check_latex.py`).

## Issues for the authors
This list leaves out Rounds A–C's items, which still apply (the addendum pointer, "unlabelled" vs "excluded", the unnamed RAGuard judge, the undefined R/W in the selection paragraph, and others).
1. "and gpt-4o-mini's LLM flag against its verdict key (\semThirtyCBsMiniM\ to \semThirtyCBsMiniJ, outside $\pm$3)". The judge interval is [+0.8, +12.5], so only the point estimate (+6.7) lies outside ±3, not the interval. "outside $\pm$3" can be read as a claim about the interval. Consider "point estimate outside $\pm$3".
2. "Claude Haiku's neutral flag (\semNeuCBsHaikuJ)". The macro is a ConflictBank contrast, but unlike the RAMDocs contrast in the previous sentence, the prose does not name the benchmark. The gpt-4o-mini verdict-key contrast has the same gap. Naming them adds words.
3. "these intervals are too wide to test the $\pm$3-point equivalence claims; re-scored on all their responses, neither holds (Appendix~\ref{app:fresh})." The prose never names the two claims (H30, H23) or H38. Adding the IDs would put new digits in the text, so I didn't.
4. "Per question, the flag and its key disagree on …". This §18 per-question agreement result has nothing to do with the paragraph head "The neutral-flag cost holds without cross-entity pairs (\S18)". Consider giving it its own paragraph or head (that adds words).
5. Paragraph head "Selection in the crossed cells." still does not state the finding. A finding head would be something like "The follow-rate account tracks crossed cells better than the marginal calculation" (\predEightMae\ vs \selMargMae). But the recall-scaled marginal error (\selMargCovMae) lies inside the account's CI \predEightMaeCI, so "better" may overstate. Left for the authors.
6. "Its directional flag gains … and \zQDetLlmDirFlagDirectionalClean\ on clean ones". The value is 0.0 (Round B's item 1). This is repeated only because "gains" is still there.

## Coherence issues across sections
- Judge-score wording: this appendix says "neither holds" (now). App. fresh (H38) says "Neither is equivalent within $\pm$3 under the judge", and Limitations says "neither pooled equivalence holds within $\pm$3". These are consistent. Results line 56 tags H38b and line 111 tags H38a; this appendix does not mention H38.
- "verdict key" appears here and in App. fresh (H38). Round A's triage noted that "verdict key" was not used in the paper. It now is, but Results calls the same control "the detector's judgement, passed as an answer key". Check that Results or Method defines "verdict key" once (results.tex, "In mean accuracy, the detector's verdict carries most of its contribution").
