# 09_app_further_semantic: Round 2C notes

## Main kinds of change
- **Caption, two sentences into one.** "Categories are exclusive: an answer the primary metric scores correct counts as correct. The misinformation share can therefore be slightly lower …" → "Categories are exclusive, correct (under the primary metric) taking precedence, so the misinformation share can be slightly lower …". The answer-policy sentence was tightened ("told to give every valid answer, a model adds the misinformation; told there is one answer, it abstains").
- **Defined term instead of paraphrase.** "the primary metric's string-matching errors" → "the matcher's errors"; "re-scored on all their responses" → "re-scored in full" (glossary: H38 is the full judge re-score).
- **Restatement cut.** WikiContradict: ", so the H28 harm holds" repeats the paragraph head; the H28 ID moved into the parenthesis "(H28; seed 0, all 253 questions)". Semantic: "both arms of each contrast gain" → "both arms gain" (the contrast is named in the main clause).
- **Merged sentences.** RAGuard audit: the three "judge finds" results and "all three hold" are now one sentence with one "the judge finds".
- **Needless words.** "number of gold answers" → "gold-answer count"; "reading is that" → "reading:"; "alternative answers" → "alternatives"; "(they gain …)" → ", gaining …"; "which was closed" → "closed"; "which uses" → "using"; "it flags … but its direction accuracy is" → "flagging … but with direction accuracy"; "what the flag does on each question" → "the flag's per-question effect"; "rather than assert" → "without asserting"; "not with verified" → "not verified"; "and by \entNeuMulti" → "and \entNeuMulti".
- Unchanged: every number, macro, hedge ("exploratory", "One untested reading", "unadjusted", "significant after Holm"), paragraph head, label/ref/cite, and all Round A/B/C/2A/2B fixes.

## Word count
Prose 1256 → 1218 (−3.0%). Both checks pass (`check_preservation.py`, `check_latex.py`).

## Issues for the authors
Earlier rounds' items (addendum pointer, unlabelled vs excluded, unnamed RAGuard judge, "outside ±3" for an interval that crosses 3, unnamed benchmarks for the Haiku/mini contrasts, missing H23/H30/H38 IDs, per-question agreement under an unrelated head, R/W undefined here, the neutral-flag split in the Qwen paragraph, bare "They"/"it", "these pairs") still apply and are not repeated. New:
1. "The failure of most concern, a final answer such as ``$Y$, not $X$'' counted as asserting the wrong answer $X$, occurs in \semRejected\% of outputs." Says how often the matcher would make this error, but not whether those outputs were then judged correct; a reviewer may ask which way it biases the contrasts.
2. "RAGuard's oracle-flag rows therefore test what a generator does with these labels, not verified contradictions; we draw no factual-oracle conclusion from them." The second clause restates the first; kept because it reads as a limitation.
3. "Removal, which needs no warning to be acted on, gains in both groups." Gives no numbers or pointer, unlike the flag results in the same paragraph.

## Coherence issues across sections
- None new beyond Rounds 2A/2B (verdict-key definition in results.tex vs this appendix and App. fresh; H38 not named here though results.tex and App. fresh use it).
