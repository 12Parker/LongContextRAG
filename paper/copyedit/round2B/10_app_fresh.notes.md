# 10_app_fresh (Appendix §19) — Round 2B notes

## Main kinds of change
- Active voice with "we" as subject for the commitment step ("We committed the follow-rate account's predictions ... before running the flagged condition").
- H35: removed the repeated "on a clean question" ("there, a flag changes fewer correct answers ...").
- Selection sentence: Claude Haiku is now the subject; "highest prediction" → "highest predicted effect"; the parenthetical names the comparison once ("against the gpt-4o-mini detector") and puts each compared pair side by side ("X vs Y").
- H38: added the missing verb in the parallel clause ("the oracle flag minus the answer key is ..."); "the point estimates" → "both estimates".
- H37: moved "For every generator" to the topic position; parallel form for the detector's direction ("names the false page as suspect on X% ... and the correct page on Y%", replacing "names ... and marks ..."); replaced the unclear "it" with "the judge".
- The file's last words ("so the") continue into 11_app_repro and are unchanged.
- No numbers, macros, hypothesis IDs, hedges, labels or table inputs changed. H32 holds, H33 fails, H38 equivalence fails under the judge, and "labels as released" are all unchanged.

## Word count
Prose 489 → 489. Both checks pass (check_preservation, check_latex).

## Issues for the authors
1. "so the dataset's oracle, like RAGuard's, is a label transfer." The causal "so" reads as if the label transfer follows from the judge's unclear stances. It follows from using labels as released; the audit only fails to verify them. Consider "and the dataset's oracle ... remains a label transfer".
2. "and both for \hcLabBoth\%." "Both" is ambiguous: both pages supporting their labelled answers, or one page supporting both answers?
3. "the mean error is \frMae\ $\le$ 3": "mean error" is used for a mean absolute error. The signed reading would also make "$\le$ 3" trivially true for negative values. Consider "mean absolute error" if space allows (here and in Results).
4. "All three H32 criteria hold: the slope's CI \frSlopeCI\ excludes 0 and includes 1 (intercept \frIntercept)". The CI's upper bound is exactly 1.00, so "includes 1" rests on rounding display. Please check that the unrounded bound is $\ge$ 1.
5. "A cell-level marginal estimate scores \frMaeMarg; predicting zero, \frMaeZero." These two extra baselines are not tied to an H32 criterion. Say whether they are descriptive.
6. "the smallest margins that would pass are \jdThirtyJMargin\ and \jdTwentyThreeJMargin": no unit ("points") is given.

## Coherence issues across sections
- Baseline names: here "generator-mean ... baselines" and "A cell-level marginal estimate". Results (results.tex, H32 paragraph) says "for generator means (cell-level rates alone: \frMaeMarg)". Use one name for the marginal baseline.
- Here the HealthContradict harm example uses the gpt-4o-mini detector rates (\hcLlmRightMini, \hcLlmWrongMini). Results (H32 paragraph) cites gpt-6-luna (\hcLlmPredLuna\ vs \hcLlmLuna). The appendix never gives the gpt-6-luna prediction in prose, so a reader looking for it must go to Table tab:hc.
- Here the paragraph head is "Frozen, the account predicts unseen cells". Results says "Frozen in advance, the account predicts new cells", and Setup says "prospective test". This matches round2A's "frozen vs prospective" item; listed only to note that it now covers this paragraph head too.
