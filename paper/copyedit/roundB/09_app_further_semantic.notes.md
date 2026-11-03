# 09_app_further_semantic — Round B notes

## Main kinds of change
- Stress position: the reasoning-model counts on RAMDocs now end on the misinformation share ("... recover X% and Y% of the gold answers, but A% and B% of their answers include the misinformation"); "Removal, which does not depend on the model acting on a warning, gains in both groups" ends on the result.
- Hedge kept, moved earlier: "One reading, which we did not test, is that ..." (was a trailing "; we did not test this").
- Active voice with the actor as subject: "The addendum registered the directional-versus-neutral comparison ... as exploratory" ("registered" and "exploratory" kept).
- Ambiguous references fixed: "The two agree" (after a sentence naming two Gemini models) -> "Judge and matcher agree"; NLI paragraph "its passages" (could read as the cross-encoder's) -> "the benchmark's passages".
- Cut words: fig:answers caption joins two sentences with "so" in place of "therefore"; "Most pairs are posts that discuss" -> "Most pairs discuss".
- No paragraph head changed. "Reasoning models on RAMDocs." (reverted in Round A) and "Detector parsing." / "Selection in the crossed cells." left as they are.
- Tried putting gpt-4o-mini's multi-answer numbers next to the "changes little" clause, but that dropped one "gpt-4.1" (the number check), so the sentence stays as it was.

Word count: 1109 before, 1104 after (prose 1109 -> 1104). Both checks pass (check_preservation.py, check_latex.py).

## Issues for the authors
Round A's eight issues (addendum pointer, "unlabelled" vs "excluded", neutral-flag subset, "do not depend on the detector's family", unnamed RAGuard judge, "slightly conservative" next to a negative shift, "these intervals" antecedent, missing pointer for "marginal break-even calculation" / "follow-rate account") still apply and are not repeated here. New ones:
1. "Its directional flag gains \zQDetLlmDirFlagDirectionalConflict\ points on conflict questions and \zQDetLlmDirFlagDirectionalClean\ on clean ones" — the clean-question value is 0.0, so "gains" reads oddly for that half; "changes accuracy by" would be neutral (verb change near a macro, so not made).
2. "On multi-answer questions it changes little except with gpt-4.1 (gpt-4o-mini \yMiniMultiOracleDir, and \yMiniMinMultiOracleDir\ with the minimal skeleton; gpt-4.1 \yFourMultiOracleDir)." The parenthesis lists only gpt-4o-mini and gpt-4.1, so "changes little" over the other settings is not shown here; and the compared numbers would read better split as "changes little (gpt-4o-mini ...) except with gpt-4.1 (...)" if the authors accept one "gpt-4.1" fewer.
3. "Asked to name suspect passages before answering, the reasoning models and Claude Haiku distrust most of ConflictBank's generated passages and abstain" — "most" has no number or pointer to where the distrust share is measured.
4. "The judge is given the answer keys, so it tests the scorer, not the benchmark labels." This caveat comes at the end of the paragraph; it qualifies the whole re-score and might sit better right after the judge is introduced (not moved: hedges stay in place).

## Coherence issues across sections
- None new beyond Round A's (35 "unlabelled" here vs "35 excluded" in setup.tex; tab:risk's matcher; judge naming across setup.tex, this appendix and limitations.tex).
- "the addendum" (Real detectors subsection, this file) vs "eight later addenda" (setup/limitations): the definite article implies a single addendum.
