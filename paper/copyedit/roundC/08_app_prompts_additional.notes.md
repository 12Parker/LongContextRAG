# 08_app_prompts_additional — Round C notes

## Main kinds of change
- Merged sentence pairs joined by "But"/"Sending it"/"Retuning illustrates ... Over" into single sentences (TrustRAG system message, NLI 1,500-character window, CAAG F1 sweep, robustness tables).
- Removed needless words: "copied verbatim" -> "verbatim"; "committed to the repository before" -> "committed before" (the caption's last sentence still names the commit history); "rather than as a problem with" -> "not about"; "The pilots" -> "They" (twice).
- Deleted one restating sentence in the NarrativeQA paragraph ("NarrativeQA is the setting where flags are most likely to be false"), which repeats the preceding "most of these flags are likely false"; kept the release pointer.
- Evidence-table caption: dropped "what each benchmark can and cannot support", which the column headers already say.
- No numbers, macros, hedges, labels, refs, cites or table bodies changed; \input lines and first/last line structure untouched.

## Word count
- Prose: 903 -> 865 (-38, -4.2%). Whole file: 1436 -> 1398.
- Both checks (check_preservation.py, check_latex.py) report OK.

## Issues for the authors
1. "It is no different from the uniform format." (Original CAAG format paragraph; was "The original format is no different from the uniform one.") "No different" is informal: is this "not significant" or a passed TOST ("equivalent within ±3")? State which.
2. "Astute RAG ... and TrustRAG ... lose same-name answers on RAMDocs (H14)." The paragraph head already says H14; the inline "(H14)" (and "the registered H14" next sentence) is redundant, but hypothesis IDs are frozen.
3. Pilot paragraph and heuristic paragraph hard-code numbers ("$-$10.5", "$+$11.8", "$+$28.4", "$6.22\%$", "134 of its 221", "54\%", "166 pairs") rather than using macros from generated/numbers.tex; worth checking they match the released runs.
4. "\S9.5", "\S17.2", "\S17.6" in this appendix refer to PREREGISTRATION sections, not paper sections, with no marker distinguishing them from paper §-references (the claims table uses the same style).

## Coherence issues across sections
- None new found beyond Round A/B lists.
