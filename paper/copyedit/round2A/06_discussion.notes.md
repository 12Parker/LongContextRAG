# 06_discussion: Round 2A notes (coherence and clarity)

## Main kinds of change
- Term made consistent with Results and the glossary: "a break-even precision" -> "a break-even direction precision" (left for the authors in Rounds A–C for lack of a spare word; paid for by "whether right or wrong" -> ", right or wrong").
- The new last sentence (frozen follow-rate account) was split off from the break-even sentence at its semicolon, so the calibration caveat and the H32 result are two sentences: "That threshold calibrates randomised flags on true pairs. For selective detectors, the frozen follow-rate account predicted unseen cells but not real web pages."
- Hedging check against Results: "in our settings", "only" and "frozen" kept. No claim is stronger than Results', with the two caveats below.

## Word count
Prose 82 -> 82. check_preservation and check_latex: OK.

## Issues for the authors
1. "For selective detectors, the frozen follow-rate account predicted unseen cells but not real web pages." Results (H32 paragraph) says the account *overstates harm* on HealthContradict (gpt-6-luna \hcLlmPredLuna vs \hcLlmLuna, same sign) and also on clean questions (\frCleanPred vs \frCleanObs). "but not real web pages" reads as a flat miss and omits the clean-question overstatement; "overstated harm on real web pages" would match Results but costs two words.
2. Same sentence: "unseen cells" are unseen questions on the same benchmark, prompt and generators (Limitations, Models paragraph). The Discussion does not carry that restriction; Limitations does.
3. "A directional flag is advice generators follow, right or wrong" is unqualified; Results says "Follow rates overlap except gpt-6.1-terra's".
4. Authors' §18/§19 rewrite dropped "only" from "it suits only a one-answer policy" and replaced the limitation "a selective detector needs its own test, which we did not run". Both changes are consistent with H32 existing; noting only so the hedge change is deliberate.

## Coherence issues across sections
- Discussion "predicted unseen cells but not real web pages" vs abstract "overstate its harm on real health web pages" vs Results §\ref{sec:real} "It overstates harm on clean questions ... and on HealthContradict": three wordings of one finding.
