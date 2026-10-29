# 05_results: Round B notes

## Main kinds of change
- Subject first, action in the verb: "Flagging exactly the gold contradictions in the neutral format lowers" -> "A neutral flag on exactly the gold contradictions lowers"; "Passing the detector's judgement as an answer key ... matches" -> "The detector's judgement, passed as an answer key (...), matches".
- Cut redundancy: the RAMDocs default prompt's "which asks for every valid answer" (the paragraph head already says it); "With a prompt stating that the question has one answer" -> "Under the one-answer prompt" (the term used elsewhere in the section).
- Bare "It" openers given their noun: "The flag changes accuracy insignificantly", "The effect is \neuPolMinimal ... under the minimal prompt", "The net effect is significantly negative", "The account tracks ...".
- Stress position: the break-even sentence now ends on $p^\star$ right before its values ("For right- and wrong-flag effects $R$ and $W$, randomised flags on true pairs do net harm below a break-even direction precision $p^\star=-W/(R-W)$: \beMini ..."), so the colon no longer follows "do net harm".
- Follow-rate account: "is a retrospective decomposition, defined after the cells were seen" moved into the defining sentence, so the label comes with the account; the other-seed check ($r=\predXfR$) moved next to the $r$ it is compared with; "It still misses the Haiku cell" joined to the tracking sentence. Four "It" openers become one sentence per point.
- Parallelism: WikiContradict clause ("where both sides are valid and listing both is correct, the flag raises accuracy"); "helps under forced choice (H20) or majority misinformation, and on RAGuard (H26)".
- Figure 1 caption: right panel now names its quantity first ("the flag's effect in the \predAllN\ crossed detector--generator cells, predicted from H31's follow rates, against the observed effect").
- Negative-valued macros checked in numbers.tex (\neuPolMinimal, \neuMlMu, \PoolTwentyThree, \selHaikuMiniObs, \selHaikuMiniWrongGain, \zSgTerra..., \hTwentyFc*, \blHFourteenMj, \realThirtyOneLuna, \hThirtyAMin, \keyDiffMin); no verb next to a signed or a-prefixed macro was changed. Paragraph heads unchanged. No qualifier moved out of its sentence.

## Word count
Prose words 1366 -> 1359. check_preservation and check_latex: OK.

## Issues for the authors
1. "Tables~\ref{tab:cb} and \ref{tab:main} give every condition and the hypothesis tables every registered test." "The hypothesis tables" has no \ref; the reader cannot find them.
2. "mostly through the cell-level rates (\predMargMae\ with cell means alone)". It is unclear what "cell means alone" are and why a 1.6 error (vs 1.5 for the full account) shows the fit comes "mostly through the cell-level rates". Name the reduced model.
3. "the LLM directional flag beats TrustRAG by \blHFourteenCbs\ points on ConflictBank": the macro includes its CI, so it renders "by +14.4 [+9.3, +19.4] points"; the unit falls after the interval. Same for \blHFourteenRd and \blHFourteenMj.
4. "It also helps under forced choice (H20) or majority misinformation, and on RAGuard (H26)." Only the forced-choice and RAGuard parts carry an ID; the majority-misinformation oracle result has no hypothesis ID or significance statement.
(Round A's items 2-12 in roundA/05_results.notes.md still apply and are not repeated.)

## Coherence issues across sections
- Intro bullet 3, "the follow rates, combined with what each detector flags, account for the effects (retrospective; ...)", vs Results §\ref{sec:real}, "The account tracks the \predEightN\ cells ... but still misses the Haiku cell (\predHaikuMini)". The intro omits that it covers only the cells with an H31 generator and misses one.
- Results §\ref{sec:real}, "Reordering \citep{liu2024lost} or paraphrasing passages changes the detectors' hit rate insignificantly (H29)", vs Limitations (Data and power), "Reversing and paraphrasing the passages change the detectors' hit rate insignificantly (H29)". "Reordering" vs "reversing": use one term.
- Intro bullet 1, "under forced choice the effect is small", vs Results H20: no significant effect for three generators (equivalence not established) and "a small one for Claude Haiku". The intro reads as an established small effect.
