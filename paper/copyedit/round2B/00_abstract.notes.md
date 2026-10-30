# 00_abstract — Round 2B notes

## Changes
- **Design sentence:** subject and verb first. "On fixed passages from six benchmarks, with seven generators from three families, we vary detector and flag format, in hypotheses pre-registered (...)" → "We vary detector and flag format on fixed passages from six benchmarks, with seven generators from three families, testing hypotheses pre-registered (...)". "pre-registered" and its parenthetical stay together.
- **H31 sentence:** "a flag naming the false side raises accuracy …; naming the correct side lowers it …" → "naming the false side raises accuracy …; naming the correct side lowers it …". The two halves are now parallel and echo "Naming the false side helps" in the sentence before (−2 words).
- **Break-even sentence:** the dangling "rising with generator accuracy" now attaches to its noun: "a break-even direction precision that rises with generator accuracy from \beMin\ to \beMax" (+1).
- **HealthContradict sentence:** bare "They overstate its harm" → "The predictions overstate its harm". "They" grammatically pointed at "the test's follow rates"; the overstating is done by the predictions (Results: "The account overstates harm"), which the preceding verb "predict" sets up (+1).
- Round 2A's split of the prospective-test sentence and its "detections" wording are kept.

Word count (prose, checker): 308 → 308. Preservation and LaTeX checks: OK.

## Issues for the authors
1. "Frozen before any flagged answer existed, the test's follow rates, with each cell's detections and no-flag accuracy, predict the flag's effect in \frN\ new detector--generator cells on unseen questions (mean error \frMae\ points, slope \frSlope)." The opening participle still modifies "the test's follow rates", but what was frozen is the predictions; the subject also arrives only after a long opener. A fix ("In predictions frozen before any flagged answer existed, …" or "Predictions from …, frozen before …, …") costs about two words, which the page-1 budget did not allow. (Round 2A raised the same point as its issue 3; restated here because this round's "The predictions overstate" makes the mismatch more visible.)
2. "A detector's verdict, passed as an answer key built from the benchmark's labels, matches its flag in mean accuracy under our metric." Says "our metric", while the H23 sentence two lines earlier says "our string metric". Same concept, two names (+1 word to unify).

## Coherence issues across sections
- Abstract: "The predictions overstate its harm on real health web pages". Results (H32 paragraph, results.tex line ~103): "The account overstates harm on clean questions (...) and" HealthContradict. The abstract names the predictions, Results names the account; the abstract also omits the clean-question case (noted in 2A; not re-reported as new).
