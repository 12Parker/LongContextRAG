# 09_app_further_semantic — Round A notes

## Main kinds of change
- Paragraph heads now state the finding: "Stronger models abstain in the single call", "Reasoning models list nearly every candidate", "An outside-family judge changes no contrast's sign (\S17.3)", "The WikiContradict harm holds under its original rubric", "RAGuard pairs rarely contradict". "Detector parsing" and "Selection in the crossed cells" (last line of the cut) left as they were.
- Split long sentences at semicolons/colons (fig:answers caption; gpt-4.1 labelled-question result; RAMDocs reasoning-model counts; judge re-score paragraph, three splits; RAGuard audit percentages).
- Moved the Gemini 2.5 Flash deviation out of the middle of the re-score sentence into its own sentence.
- Removed repetition: "under the original rubric" (now in the head), second "ConflictBank" in the NLI sentence, "RAMDocs" kept once.
- Terminology: "human audit ... is released unlabelled" -> "human-audit sheet ... is released unlabelled", matching "released human sheet" in the next paragraph and setup.tex.

Word count: 1110 before, 1110 after (prose 1110 -> 1110). Both checks pass.

## Issues for the authors
1. "The directional-versus-neutral comparison with the LLM detector's pairs was registered as exploratory in the addendum." Which addendum? Eight are mentioned elsewhere; a pointer (contains a digit) would help.
2. "but the 35 unlabelled questions contribute disproportionately to it ... Over the labelled questions it is ..." The rest of the paper (setup.tex, glossary) calls these 35 "excluded"; "unlabelled" also clashes with "released unlabelled" (audit sheet) a page later. Suggest "excluded" / "included".
3. "Its directional flag gains \zQDetLlmDirFlagDirectionalConflict\ points on conflict questions and \zQDetLlmDirFlagDirectionalClean\ on clean ones, and its neutral flag \zQDetLlmDirFlagConflict." The neutral-flag number's subset (conflict questions, by the macro name) is not stated; fixing it adds words.
4. "The conclusions do not depend on the detector's family." Based on one alternative family (Qwen2.5-7B) and RAMDocs only; authors may want to check whether this generalisation is as intended.
5. RAGuard paragraph: "the judge finds that the two posts cannot both be right..." does not say which judge (the gemini-3.5-flash-lite judge of the previous paragraph, or another). "This audit uses one LLM judge" at the end does not name it.
6. "the matcher-scored differences are slightly conservative" sits next to a range whose lower end is negative (\semDidMin = -1.7); a reader may ask how a negative shift is conservative. Worth one clarifying clause if space allows.
7. "With \semNq\ questions per contrast these intervals are too wide" — "these intervals" has no antecedent in the prose (they are the judge-scored intervals in Table tab:semantic).
8. Last paragraph ("Selection in the crossed cells") uses "marginal break-even calculation" and "follow-rate account" without pointing to where they are defined (results.tex, "Real detectors are selective"); a \S\ref would help.

## Coherence issues across sections
- 35 "unlabelled" questions (appendix.tex, Real detectors subsection) vs "35 excluded" (setup.tex, data paragraph).
- fig:answers caption says Table tab:risk counts wrong answers "under the bidirectional matcher"; setup.tex Metrics names one-way as primary. Check that tab:risk (generated) indeed uses bidirectional and says so in its caption.
- Judges: setup.tex Metrics uses gpt-6.1-terra as the LLM judge; this section uses gemini-3.5-flash-lite (outside-family) and an unnamed judge for the RAGuard audit; limitations.tex says "Gemini serves only as a judge". Naming the judge in each place would avoid confusion.
