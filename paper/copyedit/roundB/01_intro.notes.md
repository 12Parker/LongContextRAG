# 01_intro: Round B notes

## Main kinds of change
- Put the action in the verb: "A natural fix is to detect the contradiction and tell the generator" -> "A natural fix detects the contradiction and tells the generator."
- Kept the verb next to its object: "vary, independently, the detector" -> "independently vary the detector".
- Removed a bare "It" at a sentence start (bullet 3): "The break-even direction precision rises from \beMin ... It calibrates flags on true pairs." -> "The break-even direction precision, which calibrates flags on true pairs, rises from \beMin\ for gpt-4o-mini to \beMax\ for gpt-6.1-terra." The sentence now ends on the numbers.
- Replaced conjunctions with semicolons to save words, in bullet 1 ("Claude models move less; the minimal prompt ...") and bullet 4 ("(H28); at a 5\% share ...").
- Removed a comma before "but not how the two connect".
- Kept unchanged: "coded after seeing data", "retrospective", "exploratory", "on average ... equivalent", "significantly", every H-ID, macro and citation, and all of Round A's fixes 3 and 4.

## Word count
Prose words 565 -> 561. Both checks pass.

## Issues for the authors
1. "Controls separate what a flag conveys: forced choice, an answer key, randomised direction, the detector's verdict as an answer key and its model's own answer as a hint, and six detectors crossed with six generators." The crossed design is not a control of what a flag conveys, so the list mixes two kinds of item. Also, "six detectors crossed with six generators" reads as a full 6x6 design (see the coherence issue below).

## Coherence issues across sections
1. Intro P2 says "six detectors crossed with six generators", but Setup (after Round A coordinator fix 5) says the six detectors and six generators "form 30 of the possible pairs". The abstract says "\predEightN\ crossed detector--generator cells" (22). A reader may expect 36 cells. Locations: `01_intro.tex` P2, and `04_setup.tex` (crossed-design sentence).
