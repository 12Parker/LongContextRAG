# 03_method: Round B notes

## Main kinds of change
- Bare sentence-initial pronouns given their noun: "This is the original CAAG format" -> "This format is CAAG's original one"; "It conveys" -> "The flag conveys" (One-wrong flag); "It scores similar pairs" -> "CAAG scores similar pairs"; Closed book "No passages; it measures" -> "No passages, measuring".
- NLI: "scored in both directions" -> "scored in both orders", so "direction" is not used in a looser sense one clause before "It gives no direction" (direction = which side is misinformation).
- LLM detector: moved "as JSON" next to the verb it qualifies ("lists, as JSON, contradicting pairs"), so the sentence ends on the suspect-side options rather than the output format.
- Minimal skeleton: ", and its answer line" -> "; its answer line" (two points).
- All quoted prompt strings, \texttt{} headers, model names, math, item heads and definitional details unchanged.

## Word count
651 -> 650 (prose 651 -> 650). Both checks pass.

## Issues for the authors
1. "It is gpt-4o-mini (temperature 0) unless stated; in the crossed design gpt-4.1, gpt-6-luna, gpt-6.1-terra, Claude Haiku 5.5 and Claude Sonnet 5.5 run the same prompt." Sentence-initial "It" kept (the noun would add words); the crossed design is still undefined here (Round A issue 2).
2. "Each flagged passage gets the header \texttt{[POTENTIAL CONFLICT with passage(s): j]}, followed by a list \texttt{Passages i and j: possible contradiction}." "followed by" reads as if the list follows the header inside the passage; where the list is placed in the prompt is not stated.
3. "Plain plus a generic instruction to check the passages for contradictions, with no detector. v1 ends ..." The sentence starting "v1" begins with a lowercase identifier; harmless but noticeable.
4. Round A issues 3-6 (directional rendering of suspect-less pairs, answer-key source, "Remove (suspect)" vs "removal", "detector pipeline" defined only in Setup) still stand.

## Coherence issues across sections
- None new beyond Round A's ("forced choice" in intro.tex/results.tex vs "forced skeleton" here; intro.tex's quoted ``these passages may conflict'' vs the rendered \texttt{possible contradiction} here; abstract.tex quotes ``these passages conflict'').
