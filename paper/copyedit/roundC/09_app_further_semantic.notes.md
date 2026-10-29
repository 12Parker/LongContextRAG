# 09_app_further_semantic: Round C notes

## Main kinds of change
- Wordy constructions tightened: "One reading, which we did not test" → "One untested reading"; "The failure we were most concerned about" → "The failure of most concern"; "gpt-4.1 as the generator" → "gpt-4.1 generating"; "Judged that way" → "Under it"; "are a transfer of behavioural labels" → "transfer behavioural labels"; "which does not depend on the model acting on a warning" → "which needs no warning to be acted on".
- Semantic re-score opener: the stand-alone sentence "The primary metric matches answer strings." merged into the purpose sentence ("whether the primary metric's string-matching errors differ by condition"); "a family none of our generators belongs to" → "outside our generators' families".
- Figure caption last sentence shortened (same content, same pointer to Table~\ref{tab:risk}).
- Defined-term hygiene: "The judge is given the answer keys" → "Given the gold answers, the judge ..." so *answer key* keeps its sense as the control format (the same paragraph uses "answer-key outputs" in that sense). "evaluation" → "rubric" in the WikiContradict paragraph to match its head.
- Small deletions: "to it", "the" before "gold answers", "that", "which then"; two ", and" joins made semicolons.
- No paragraph heads, numbers, macros, hedges ("exploratory", "significant after Holm", "after seeing", "post hoc"-type wording) or pointers changed. First and last lines' structure untouched.

## Word count
Prose 1105 → 1060 (−45, −4.1%). Both checks pass.

## Issues for the authors
- "The conclusions do not depend on the detector's family." One outside-family detector (Qwen2.5-7B), with a very different flag rate (\qdetFpr\% of clean questions), supports a narrower statement ("hold with a detector from another family"); the general wording may overreach.
- "but its direction accuracy is \qdetDir\%." The contrast "much less selective ... but" is clear, but the paragraph never says whether this direction accuracy is higher or lower than the gpt-4o-mini detector's, which is what the comparison needs.
- "its gain over all 300 questions (\gptLlmDirAll) is significant after Holm" — "after Holm" is redundant with the paper's definition of significant; harmless, left.

## Coherence issues across sections
- "Answer key" also used for gold answers: this part (fixed here, semantic-re-score paragraph) vs. glossary/method (answer key = the control format). Worth a grep for other loose uses in 10_app_repro.tex and limitations.tex.
