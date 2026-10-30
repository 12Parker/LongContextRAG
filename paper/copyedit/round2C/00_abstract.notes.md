# 00_abstract — Round 2C notes

## Main kinds of change
- Merged a doubled verb: "detect contradictions ... and flag them to the generator" → "flag contradictions ... to the generator" (flagging implies detection).
- Dropped redundant determiners: "on the benchmarks' gold pairs" → "on gold pairs" (gold pairs are the benchmarks' by definition); "built from the benchmark's labels" → "built from benchmark labels"; "matches its flag in mean accuracy" → "matches its flag's mean accuracy".
- Cut a connective that adds nothing: "Randomised flags on true pairs therefore help" → "... help".
- "predict the flag's effect" → "predict flag effects"; "The predictions overstate its harm" → "They overstate the harm".
- Kept every macro, "under our string metric", "under our metric", "(exploratory)", "pre-registered", "on average", "only", "alone", and all H-IDs.

## Word count
308 → 298 prose words (−3.2%). Both checks pass (check_preservation, check_latex).

## Issues for the authors
- "Frozen before any flagged answer existed, the test's follow rates, with each cell's detections and no-flag accuracy, predict flag effects ..." — the predictions were frozen, not the follow rates (already raised in round 2B; a fix would add words, so not made).
- "naming the correct side lowers it by \ahThirtyOneBMax\ to \ahThirtyOneBMin." — still no "points" (raised in round B).
- "matches its flag's mean accuracy under our metric" — "our metric" vs "our string metric" two sentences earlier; the H30 judge result (H38a) is still not given beside it, while the H23 judge result is (raised in round 2A/2B).
- "They overstate the harm on real health web pages" — "They" refers to the frozen predictions/follow-rate account; reads acceptably but authors may prefer an explicit subject if space allows.

## Coherence issues across sections
None new beyond those listed in round 2A/2B (string metric vs matcher naming, abstract vs intro/discussion wording on health web pages).
