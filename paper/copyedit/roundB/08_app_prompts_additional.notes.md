# 08_app_prompts_additional: Round B notes

## Main kinds of change

- H14 paragraph: folded "We take the matched comparison as the principal one" into the sentence that reports the matched result, so the result and its status sit together (one sentence fewer).
- Heuristic detector: moved "CAAG is the detector we set out to study" into the paragraph's opening sentence as an appositive (context first), and gave the following sentence CAAG as its subject.
- Replaced bare sentence-initial "This"/"They" with a noun ("The original format is no different from the uniform one"; "The pilots led us to add ..."; "NarrativeQA is the setting where ...").
- Table bodies, captions' content, `\input{generated/...}` lines, the first and last lines' structure left untouched.

## Word count

- Prose words: 904 -> 903 (all words 1437 -> 1436). Both checks OK.

## Issues for the authors

- Phase-0 paragraph repeats itself: "A single story rarely contradicts itself, so most of these flags are likely false." is followed by "NarrativeQA is the setting where flags are most likely to be false; the flagged pairs are released for annotation." (originally "This is the setting ..."). The second clause either restates the first or makes an unsupported comparative claim (most likely of all settings); consider cutting it.
- H14 paragraph covers four topics (H14 baselines, the TrustRAG system-message fix, the NLI window, H18). It could be split before "NLI on ConflictBank first saw passage prefixes ..." if appendix space allows.
- "Original CAAG format" paragraph: "The original format is no different from the uniform one." does not say whether this is a test result (not significant / equivalent) or a descriptive reading; neither word is used, so it is unclear which category applies.

## Coherence issues across sections

- None new beyond those already in roundA/TRIAGE.md ("uniform format" is already listed there).
