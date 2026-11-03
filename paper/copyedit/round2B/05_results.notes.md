# 05_results — Round 2B notes

## Changes (all meaning-preserving)

- **No bare "This"/"It" openers in new text.** H23 paragraph: "This rests on the string matcher" → "Equivalence rests on the string matcher" (the subject is the pooled equivalence just stated). H31 paragraph: "It is a calibration" → "The threshold is a calibration".
- **Compared numbers side by side (H30/H38a).** "…equivalent within $\pm$3 under the matcher. The judge gives \jdThirtyJ …" → "…under the matcher; under the judge, \jdThirtyJ\ \jdThirtyJCI\ (not within $\pm$3, H38a)": the matcher and judge estimates now sit in one parallel sentence, with the same ±3 qualifier.
- **Cut a redundant word** to keep length: "It also helps under forced choice" → "It helps under forced choice".
- Kept Round 2A's "(H33 fails)" and the H19 sentence after the H24 account; kept all earlier-round fixes.

Both checks pass (`check_preservation.py`, `check_latex.py`).

## Word count

1277 → 1277 (prose 1277 → 1277). No growth was possible beyond these edits without cutting content.

## Issues for the authors

1. "points (H37a); the neutral flag, nothing significant (H37b; Table~\ref{tab:hc})." The gapped verb ("gains") is implied; the appendix phrases H37b as a change of at most \ahcNeuMini\ points. A fuller verb ("changes nothing significantly") would cost words.
2. "The sign is right in all \frBigN\ cells predicted at $\ge$3 points" — the appendix uses $|\text{pred}|\ge3$; "predicted at ≥3 points" can read as gains only. "predicted to change by ≥3 points" would be clearer (+2 words).
3. "but only \frHarmConf\ of \frHarmN\ predicted harms is significant (H33 fails)" — Results does not say "predicted harm" means a predicted loss of 3 points or more (appendix: "cells predicted to lose 3 points or more"). Adding it requires a digit, so not done here.
4. "The key says ``the answer $X$ is false'' for the answer the benchmark labels its suspect passage as supporting" — hard to parse, and "its" (the detector's) has no nearby antecedent. A clearer version ("for the answer that, by the benchmark's labels, the detector's suspect passage supports") is longer.
5. "It helps under forced choice (H20) or majority misinformation" and "It helps all \nRealMjTests\ generators under majority misinformation" still start with a bare "It" (the directional flag); left to avoid growth.
6. "On 400 unseen ConflictBank questions" — the glossary and appendix call this set ConflictBank-fresh; Results never uses the name, so readers following the pointer to Appendix~\ref{app:fresh} meet a new term.
7. "five detectors (one new) crossed four generators" — the new detector is not named in Results (Table~\ref{tab:fresh} is the only pointer).

## Coherence issues across sections

- Results, H30 paragraph uses "the detector's judgement, passed as an answer key"; the appendix (app:fresh, H38 paragraph) says "the verdict key" ("the LLM flag minus the verdict key"), and the glossary's term is *verdict key*. Results never introduces the term, so the appendix uses it undefined. Locations: results H30 paragraph; 10_app_fresh "Under the judge, neither pooled equivalence holds (H38)".
- Results says the neutral flag on HealthContradict gives "nothing significant (H37b)"; the abstract says "a neutral flag does not" help there. Consistent, but the appendix adds "within ±3 for gpt-6-luna and Claude Haiku" (equivalence for two of three), which neither the abstract nor Results mentions. Locations: 00_abstract; results §Naming the false side; 10_app_fresh H37 paragraph.
