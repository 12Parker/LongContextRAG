# 03_method: Round C notes

## Main kinds of change
- Deleted one restating sentence: One-wrong flag's "The flag conveys that the conflict is misinformation, not its direction." (the preceding sentence already says exactly one passage is misinformation "without saying which").
- No-pairs control: dropped the paraphrase "each flag without its pair structure" and kept its definition "flagged passages are marked, but not which conflicts with which".
- Random: "a placebo matching CAAG's flag volume" -> "a placebo" (the same per-pair rate already fixes the volume).
- Tightened wording: "to leave out" -> "omit"; "This format is CAAG's original one ... are rendered" -> "The format is CAAG's original ... render"; "The suspect passages of every flagged pair" -> "Each flagged pair's suspect passages"; "telling them apart from" -> "distinguishing them from"; "from the gold labels" -> "from gold labels"; Instruction "a generic instruction ..., with no detector" -> "a generic, detector-free instruction"; Single call "first writes ... and then" -> "writes ... and then"; Closed book "No passages, measuring what" -> "No passages: what".
- Kept verbatim: every \texttt{} header, quoted prompt string (v1 ending, Pick, Answer key, ``unknown'', ``both''/``unclear''), model names, math, item heads, skeleton definitions, NLI threshold, synthetic grid values, "used with its code unchanged".

## Word count
650 -> 617 prose words (-5.1%). Both checks pass.

## Issues for the authors
1. "It is gpt-4o-mini (temperature 0) unless stated; in the crossed design gpt-4.1, gpt-6-luna, gpt-6.1-terra, Claude Haiku 5.5 and Claude Sonnet 5.5 run the same prompt." The crossed design is still undefined here (Round A issue 2).
2. "Each flagged passage gets the header \texttt{[POTENTIAL CONFLICT with passage(s): j]}, followed by a list \texttt{Passages i and j: possible contradiction}." Where the list sits in the prompt is not stated (Round B issue 2).
3. "No passages: what the generator knows without retrieval." is now a fragment, as are other item bodies ("Passages and question only."); acceptable in a description list, but check the style is wanted.
4. Round A issues 3-6 (directional rendering of suspect-less pairs, answer-key source, "Remove (suspect)" vs "removal", "detector pipeline" defined only in Setup) still stand.

## Coherence issues across sections
- None new. Round A/B items still stand: "forced choice" (intro.tex, results.tex) vs "forced skeleton" here; intro.tex's ``these passages may conflict'' and abstract.tex's ``these passages conflict'' vs the rendered \texttt{possible contradiction} here.
