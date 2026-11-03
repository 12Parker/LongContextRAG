# 08_app_prompts_additional: Round 2C notes

## Main kinds of change

- Merged short sentences that continue one point ("...system message; sending it", "keeps both answers, but the pair",
  "Retuning shows why F1 misleads: over a ... sweep ..., but it flags", pilot "showed the same pattern: oracle neutral
  flags ...", "removes correct documents, and led us to add").
- Folded the restating sentence "The original format is no different from the uniform one." into the preceding
  sentence ("..., no different from the uniform format"); claim kept.
- Removed needless words: "below", "The" in the claims caption, "their", "of these", "of its ... false flags" (the
  subject already says false flags), "in general ... about" → "generally", "would be" → "is",
  "Rendering CAAG's flags ... changes" → "CAAG's flags ... change", "so the prompt is unchanged" → "(prompt unchanged)",
  "CAAG flags a pair whose score reaches" → "a pair is flagged at".
- Kept "The pilots also showed" (Round C restored it from "They"); the next sentence is now joined to it, so no bare
  "They" follows the flag numbers.
- Prompts (`\input{generated/app_prompts}`), table bodies, labels, citations, macros and hedges untouched.

## Word count

All words 1,307 → 1,281; prose 829 → 803 (−26, −3.1%; target 3%). Both checks pass.

## Issues for the authors

1. "The original format is no different from the uniform one." (now "..., no different from the uniform format") gives
   no test or CI; if it means "not significantly different", say so with the test.
2. "Its false flags fall mostly on pairs that should both be reported" paraphrases the defined term *ambiguity*; the
   same sentence then says "ambiguity pairs".
3. Pilot effects ("$-$10.5", "$+$11.8", "$+$28.4"), the NarrativeQA figures (100, 50, 54\%, 166) and CAAG's "6.22\%",
   "134 of its 221" are hard-coded rather than macros from `generated/numbers.tex`.
4. "compared our pipeline under the multi-answer prompt with the baselines' own one-answer prompts" (now "our
   pipeline's multi-answer prompt"): the authors may prefer to name the comparison as systems, not prompts.

## Coherence issues across sections

- None new. (Round 2A's "with policies matched" / Holm-over-four item and the tab:claims caption covering only two
  status labels remain open; not re-reported.)
