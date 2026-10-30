# 11_app_repro (Appendix: Reproducibility), Round 2C notes

## Main kinds of change
- Word-level tightening across the deviations list. Every item is still there, and so are all revisions, dates, prices and identifiers. Examples: "the dataset's Hugging Face release" → "the Hugging Face release"; "in place of" → "not"; "The two models returned" → "They returned"; "to stay within budget" → "to save budget"; "our client did not record the served snapshot identifiers" → "served snapshot identifiers were not recorded".
- Ninth registration: the sentence "\S17.6 records two deviations, both registered before the runs they describe" is folded into the item's first sentence ("…, as were its two deviations (\S17.6)"). The last two sentences are merged.
- Run interruptions: the two sentences are merged into one. "no row was re-sampled" is kept.
- Model configuration: "at their only allowed (default) temperature and default reasoning effort" → "at default temperature and reasoning effort". The "only allowed" fact is still stated in the Reasoning-models deviation item.
- CONFLICTS judge: "a judge from a different model family" → "an outside-family judge" (the paper's term).
- Closing sentence: "before the runs they cover" → "before their runs".
- No hedge, registration label ("exploratory", "post hoc", "not significant", "about", "at most", "only"), number or macro was changed. Both checks pass.

## Word count
Prose 1747 → 1693 (−54, −3.1%; target 3%).

## Issues for the authors
1. The Reasoning-models deviation item and the Model-configuration paragraph both give the gpt-6 models' temperature and the 4,096-token budget: "we used their defaults with a 4,096-token budget" and "ran at default temperature and reasoning effort, with 4,096 completion tokens including reasoning". You could keep the detail in one place and point to it from the other.
2. "One ConflictBank gpt-4o-mini run was resumed from the response cache after a transient API error" (Second registration) covers one case of the later "Run interruptions" item. Consider whether both are needed.
3. "The addenda (H10--H38 and the re-analyses above) and the Qwen seed change were each committed before their runs." The Qwen item itself says only "a change made before inspecting any Qwen result", which is a different claim (made before inspecting results, not committed before runs). Check that both hold.
4. Metric choice: "our port of MADAM-RAG scores 28.5\% on the pilot split" hard-codes the number, while the neighbouring values are macros (\madamOneWay, \madamBidir). Check it against the generated numbers.
5. "The question-level model adds statsmodels 0.14.5." (was "adds one dependency (statsmodels 0.14.5)"). If you want to keep the point that it is the only new dependency, restore "one dependency".

## Coherence issues across sections
- None new. Round 2A's left items still apply: the release and licence details for ConflictBank-fresh and HealthContradict are missing from this appendix, and the pre-registration "§" numbers can be mistaken for paper sections (this appendix vs Setup). They are not re-reported here.
