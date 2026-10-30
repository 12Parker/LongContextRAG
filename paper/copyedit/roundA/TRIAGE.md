# Round A triage: coherence and clarity

**Principles:** one idea per sentence (≤ ~35 words); each paragraph opens with its point; terms match the glossary; no section gets longer; no number, macro, citation, reference, hedge or registration label changes.

## Per-part results (both checks: `check_preservation.py` and `tools/check_latex.py`)

| Part | Prose words | Checks |
|---|---|---|
| 00_abstract | 286 → 284 | OK |
| 01_intro | 557 → 556 | OK |
| 02_related | 648 → 644 | OK |
| 03_method | 651 → 651 | OK |
| 04_setup | 1204 → 1203 | OK |
| 05_results | 1369 → 1369 | OK |
| 06_discussion | 75 → 75 | OK |
| 07_limitations | 1199 → 1199 | OK |
| 08_app_prompts_additional | 903 → 902 | OK |
| 09_app_further_semantic | 1110 → 1110 | OK |
| 10_app_repro | 1699 → 1699 | OK |
| 11_app_grid | 803 → 789 | OK |

Whole-paper check after the coordinator fixes (`--paper`, HEAD vs joined): no number gone, no occurrence removed. The
reported "unresolved" `\ref{tab:*}` are labels defined in `generated/` tables (the script does not read `\input`
files); the tectonic build reports no undefined reference.

## Reverted after review

- 09: head "Reasoning models list nearly every candidate." → restored "Reasoning models on RAMDocs." (2.8–3.0 answers at 77–79% recall is not "nearly every"; a new claim).
- 05: head "Pooled effects vary by policy." → restored "Across settings." (states a post hoc split as a finding; no test of the difference).
- 03: the new paragraph break before "Every prompt shares a skeleton" (merged back for page fit).

Checked and kept: "An outside-family judge changes no contrast's sign" (no sign changes; `judge8_stats`), "The WikiContradict harm holds under its original rubric" (`\wikiOrig*Eff` all negative), "RAGuard pairs rarely contradict" (`\rgAudContradict` = 6), "Both summaries place most real effects inside the prediction interval" (`\hSixDocIn`=7, `\hSixPairIn`=6 of `\hSixN`=8), "Registered results differ by generator", "With rare conflicts, false flags can outweigh gains" (abstract wording), "Stronger models abstain in the single call", "Gains rise with direction accuracy; precision matters mainly for removal" (paragraph's own content).

## Fixed after the round (coordinator), with source

1. **Abstract, H30 "matches the flag within 3 points"** → "matches the flag (pooled, equivalent within ±3)". Source: per-cell range `\hThirtyAMin`..`\hThirtyAMax` = −2.0..+3.3; equivalence is the pooled TOST (results.tex, H30 paragraph).
2. **Abstract, pooled neutral flag** now tagged "(exploratory)", as in results.tex ("Table~\ref{tab:pooled}; exploratory").
3. **Intro bullet 4, "lowers accuracy in 22 of 31 settings"** → "has a negative estimated effect in … (exploratory)". Source: results.tex ("its flag's estimate is negative in … significantly in \prevSigNeg"); `prevalence8` is an unregistered re-analysis.
4. **Intro, "the detector's verdict and its model's own answer as hints"** → "the detector's verdict as an answer key and its model's own answer as a hint". Source: method/results (H30 is an answer key; H27 is the hint).
5. **Setup, "six detectors and six generators form 30 pairs"** → "form 30 of the possible pairs". Source: PREREGISTRATION §13/§14.1 (cells added piecemeal; not a full 6×6).
6. **Limitations, "the registered gap"** → "the registered H24 correlation" (results.tex's name for it).
7. **App. reproducibility, one-way metric "reproduces MADAM-RAG's published score"** → "is closer to", matching setup.tex and the Metric-choice paragraph ("not a reproduction").
8. **App. reproducibility, "Gemini was planned but not run"** → "planned as a generator … then" (Gemini is used as the §17.3 judge).
9. **App. reproducibility, Cost paragraph "The judge, gemini-3.5-flash-lite"** → "The §17.3 judge" (setup.tex's judge is gpt-6.1-terra; two different judges).
10. **App. reproducibility, truncation shares** given their missing `\%`. Source: `assets8.py` lines 93–94 build them with `pct()`.
11. **App. reproducibility, deviations list** gains a "Ninth registration" item (judge swap; 300-character NLI window chosen on detection output only), from PREREGISTRATION §17.6; setup.tex says this appendix "lists every deviation".
12. **App. reproducibility, TrustRAG "system message is not sent"** → not sent in the registered runs; the §17 rerun restores it (pointer to app:extra, where `\blTrustSys*` are reported).
13. **App. extra, H14 "the LLM directional flag gains \blHFourteenCbs … \blHFourteenMj"** → "changes accuracy by" (`\blHFourteenMj` = −0.5).
14. **App. extra, CAAG sweep pointer** `Appendix~\ref{app:extra}` (self-reference) → `\ref{app:grid}`, where the grid's removal harm is shown.

## Rejected

- Intro "Claude models move less" vs results "move the other way": the bullet is about accuracy (Claude's neutral-flag changes are not significant, H25); "the other way" is about the misinformation share. Not a conflict.
- "Limitations omits the ninth addendum": Limitations' count ("H10–H31 in eight addenda") is correct; §17 registers no hypotheses.
- "35 unlabelled vs 35 excluded": both true (excluded because unlabelled); the appendix paragraph contrasts labelled with unlabelled questions.

## Left for the authors

- **Page fit.** After this round the body ran about four lines onto page 9: page 1 now ends about three lines earlier because of the abstract and intro edits and coordinator fixes 1–4. Round C's cut target is meant to absorb this; if it doesn't, cut by hand.
- **Exploratory labels in the summaries.** The abstract and intro still state the break-even range and the H23 equivalence without "exploratory" (results.tex labels both). Adding them costs words on page 1.
- **Intro bullet 3's "because" clause** drops results' "except gpt-6.1-terra's", "descriptive" and "mainly".
- **Related Work's "accuracy still falls"** for the neutral flag: sourced (H2 on RAMDocs), kept. Its "break-even account of when a real detector's verdict helps" was fixed after the round (now "a break-even direction precision per generator, and why it can misjudge selective detectors"; results.tex "This threshold is a calibration").
- **Terms.** These need defining, unifying or naming:
  - "follow rate" is not defined in the main text;
  - "forced choice" vs the "forced skeleton";
  - "uniform format" (app. extra);
  - the neutral flag's quoted wording differs across abstract, intro and method;
  - "verdict key" is not used in the paper;
  - "correlated settings";
  - "the two spaces' errors" (H6 deviation item).
- **Significance not stated.** Results doesn't say whether these are significant:
  - the majority-misinformation and RAGuard-Claude counts;
  - H27;
  - the H20 Claude Haiku effect.
- **Two precision macros for one detector:** `\selHaikuMiniPrec` (69) and `\dirPrecMini` (68.8).
- **Captions in generated tables** (`tab:claims` "Re-analysis" variants, `tab:risk` matcher) are not editable here; the source is `assets*.py`.
- Per-part detail is in each `*.notes.md`.
