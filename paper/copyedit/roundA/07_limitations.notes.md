# 07_limitations: notes (Round A)

## Main kinds of change
- **Split long sentences.** Models: the seed / temperature / endpoint sentence is now three sentences, and the H24 and follow-rate sentences were split at their semicolons. Data: the majority-variant, RAGuard-audit and H29 sentences were split. Direction: the direction-precision definition moved out of a parenthesis into its own sentence. Pre-registration: the addenda / timestamp sentence was split. Ethics: the dual-use sentence was split before the inverted-oracle costs.
- **Reordered within paragraphs** so related points sit together. Limitations were reordered only; none was cut or softened.
  - *Models and detectors*: generators first, then detectors and baselines (LLM prompt, reimplementations, NLI window, ConflictRAG), then the crossed design (H24 power, the registered gap, concurrent-run drift, the follow-rate account, break-even).
  - *Data and power*: synthetic benchmarks, then their consequences (reasoning models distrust generated passages, H29 surface cues, majority variant), then RAMDocs labels, then all RAGuard points together (the closed-book sentence was moved up from the middle of the paragraph), then WikiContradict and CONFLICTS, with power last.
  - *Metric*: the sentence about the lenient variant now follows the string-matching sentence rather than the human-label sentence.
  - *Direction and the grid*: the LLM-detector points come first and the three grid sentences come together at the end. The sentence "In the synthetic grid ..." had been separated from the other grid sentences.
- **Paragraph head**: "Models." became "Models and detectors.", because the paragraph also covers detectors, baselines and the crossed design.
- **Terms**:
  - "a more lenient variant" became "the more lenient bidirectional variant" (glossary: the bidirectional metric is the robustness metric).
  - "conflict-free questions" in the Ethics Statement became "clean questions" (glossary label; it also matches "clean questions" in the Data paragraph).
  - The LLM judge now "agrees with the matcher", replacing "it", which became ambiguous after the reorder.

## Word count
1199 before, 1199 after (prose 1199 -> 1199). check_preservation.py OK, check_latex.py OK.

## Issues for the authors
1. "The registered gap turned out to be dominated by the generator's own accuracy." "Registered gap" is not defined here. Results calls this the "registered H24 correlation" (results.tex, para. "Real detectors are selective."). Name it the same way in both places, or say what the gap is between.
2. "We report the more lenient bidirectional variant throughout and mark any conclusion that depends on the choice." I added "bidirectional" from the glossary. Please confirm the lenient variant meant here is the bidirectional metric.
3. "NLI sees only a sentence-sized window". Commit 21d916c and §17.6 describe a 300-character window. I could not add the number. Consider "a 300-character window" for precision.
4. "The grid predicts real detectors only roughly, and three detectors cannot show whether document- or pair-level summaries predict better." The crossed design has six detectors, so a reader may ask why only three appear here. A short pointer to app_grid (line 71, "with three detectors we cannot claim ...") would help, but I could not add it without adding words.
5. "Reversing and paraphrasing the passages change the detectors' hit rate insignificantly (H29)". The brief reserves "significant" for Holm. "Insignificantly" is presumably "not significant under Holm". I kept the wording as written; consider "do not significantly change".
6. "Its output is identical across its cells for at least \frozenShareMin\% of questions" ("it" = a detector). This holds per detector; "each detector's output" would be clearer, but I did not change it because that edges toward changing the claim.

## Coherence issues across sections
- **Pre-registration count.** Limitations says "H10--H31 were registered in eight addenda". setup.tex (line 39) also describes "A ninth addendum (\S17)" registering re-analyses, corrected baselines and a semantic re-score, and the glossary lists §17 as a ninth addendum. The Limitations pre-registration paragraph never mentions the ninth addendum or that it came after a mock review. Locations: limitations.tex, Pre-registration paragraph; setup.tex line 39.
- **"Conflict-free" vs "clean".** related.tex line 40 and app_grid.tex line 26 use "conflict-free questions", while the glossary and the rest of Limitations use "clean questions". The Ethics Statement now says "clean"; the other files' editors may want to align theirs.
- **H24 naming.** "registered gap" (limitations.tex, Models paragraph) vs "registered H24 correlation" (results.tex line 103).
