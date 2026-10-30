# 00_abstract — Round 2A notes

## Changes
- **Prospective-test sentence (H32, HealthContradict)** split in two. The original (~60 words) ran the prediction, its freezing, its accuracy and the health-page overstatement into one sentence, and "frozen before any flagged answer existed" dangled after "on unseen questions". It now opens with the freezing, ends the first sentence on the fit statistics, and starts a new sentence with "They overstate its harm on real health web pages ...".
- "This test's follow rates" → "the test's follow rates". Two sentences now sit between it and "In a randomised test", so "This" no longer clearly pointed back to it.
- "each cell's coverage" → "each cell's detections". "Coverage" is not defined anywhere in the paper. Results (H32 paragraph) says "predictions from detections and no-flag answers".
- H23/judge sentence: checked and kept. At 36 words it reads clearly. Folding the judge clause into the main sentence ("..., though an outside-family judge favours the flag by ... points") added 3 words or more, so I left it as it was.

Word count (prose, checker): 308 → 308. Numbers, macros and LaTeX checks: OK.

## Issues for the authors
1. "an outside-family judge favours the flag by \jdTwentyThreeJ)." The judge figure has no "points", unlike the matcher figure just before it ("\PoolTwentyThree\ points"). Adding the word would cost +1.
2. "This test's follow rates" (now "the test's follow rates"): *follow rate* is a defined label, but the abstract never says what it means (the rate at which generators follow right and wrong flags). A reader who meets it only in the abstract has to guess. Defining it here would add words.
3. "Frozen before any flagged answer existed, the test's follow rates, ...": strictly, what was frozen is the predictions, not the follow rates. The wording that fits the word budget makes the follow rates the subject. If you can spare about 3 words, "predictions from the test's follow rates, ..., frozen before ..." is more exact.
4. "A detector's verdict, passed as an answer key built from the benchmark's labels, matches its flag in mean accuracy under our metric." The H38a judge result for this comparison (flag ahead by \jdThirtyJ, not within ±3) is not in the abstract. The H23 sentence does give its judge result, so the two sentences are now handled unevenly. The brief's "one idea" mentions a judge gap of about 2.5 for the verdict. Decide whether the abstract should say it for both.
5. "matches its flag" (H30) vs "no more than an answer key" (H23): both rest on a pooled TOST under the matcher, but neither says "equivalent within ±3". After roundA's coordinator fix 1 the H30 wording read "(pooled, equivalent within ±3)", and that qualifier has since gone. Check that this was intended.

## Coherence issues across sections
- The abstract says "each cell's detections and no-flag accuracy" (now). The intro (bullet 3) says "with what each detector flags", and results.tex (H32 paragraph) says "predictions from detections and no-flag answers". All three are compatible, but the wording differs.
- The abstract says the account overstates harm "on real health web pages". Results (§sec:real, H32 paragraph) and the appendix row "it overstates harm on clean questions and HealthContradict" also list clean ConflictBank questions. The abstract omits that case.
