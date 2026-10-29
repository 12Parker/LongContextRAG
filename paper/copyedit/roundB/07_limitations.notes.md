# 07_limitations — Round B notes

## Main kinds of change
- Active voice with the main character as subject: "The registered H24 correlation turned out to be dominated by the generator's own accuracy" → "The generator's own accuracy dominated the registered H24 correlation."
- Fixed bare-pronoun sentence openers by joining sentences:
  - "…in different cells. Its output is identical across its cells…" → "…in different cells, but its output is identical across them…";
  - "…which a real attacker would vary. They also include ConflictBank's…" → "…which a real attacker would vary, and include ConflictBank's…". The subject is still the majority variant's extra wrong-side passages (setup.tex l.17–18).
- Direction and the grid: the definitions of direction precision and pair precision now come before the LLM detector's numbers. "Its hit rate per question is lower" now sits right after the precision it is compared with, and "its flags" in the definition became "a detector's flags".
- "change the detectors' hit rate insignificantly (H29)" → "do not significantly change the detectors' hit rate (H29)". This uses the paper's "not significant" qualifier; the claim is the same.
- "known though before their own runs" → "known but before their own runs".
- I removed no limitation and softened no hedge. I tried cutting "Our results show that" in the Ethics Statement, then reverted it, because the phrase ties the claim to the evidence.

## Word count
Prose words 1200 → 1197. Both checks (check_preservation.py, check_latex.py) report OK.

## Issues for the authors
1. **The ±5 margin differs from the paper's ±3.** The sentence reads: "but equivalence within $\pm$5 points is shown for only one of four changes." Everywhere else, "equivalent" means a TOST at ±3. Please confirm that H29 registered a ±5 margin, and consider saying so ("H29's registered ±5 margin").
2. **A macro carries its own sign.** `\semDidMax` expands to "$+$3.3", so the sentence reads "move no contrast by more than +3.3 points". The sentence is: "but move no contrast by more than \semDidMax\ points". An unsigned macro would read better here.
3. **"pair precision" is defined but not used in the paragraph.** The sentence is: "pair precision is the share of its flagged pairs that are true contradictions." No pair-precision number follows. Either drop the definition or keep it as the setup for the last sentence ("pair-level summaries"); the authors should choose.
4. **"which is why" is a causal claim.** The sentence is: "generators abstain on \rgAbstMin--\rgAbstMax\% of conflict claims, which is why accuracy on this true/false task falls below chance". Check that the abstention share alone accounts for the below-chance accuracy.
5. **"Claude ran through Anthropic's OpenAI-compatible endpoint."** The sentence names no consequence. If the endpoint matters (for example, unsupported parameters), say how.

## Coherence issues across sections
1. **H29 equivalence.**
   - limitations.tex l.25–26: "equivalence within $\pm$5 points is shown for only one of four changes".
   - appendix.tex l.53 (claims table): "registered addendum; equivalence not shown".
   - The table says equivalence is not shown at all; Limitations says it is shown for one of four changes. One of them should be qualified.
2. **H29 wording.** results.tex l.120 still says "change the detectors' hit rate insignificantly (H29)"; Limitations now says "do not significantly change". The meaning is the same. The results file could adopt the same wording.
