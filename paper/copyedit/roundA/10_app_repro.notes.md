# 10_app_repro: notes (Round A)

## Main kinds of change
- Split long run-on sentences: H21(c) redefinition (Seventh registration); pooled/moderators/utility changes (Eighth registration); RAMDocs labels; the Model configuration model list (semicolon chain turned into one sentence per model group).
- Replaced the awkward "ratio of correct to answered answers" with the glossary form "correct over answered".
- Cost paragraph: moved the price list out of a long parenthetical and gave the judge price its own sentence (all prices and both dates unchanged).
- RAGuard: rewrote the gapped "Supporting × misleading pairs are the contradictions and misleading posts the misinformation side" as one pair-level statement.
- Placement modes: "these" replaced with "the duplicates" (unclear referent).
- No change to deviations, prices, dates, model settings, counts, macros, labels or paragraph heads.

## Word count
1699 -> 1699 (prose 1699 -> 1699). Both checks pass.

## Issues for the authors
1. Item 1 vs Metric choice (same section) and setup: item 1 says one-way matching was chosen "because it reproduces MADAM-RAG's published score on the pilot split", but Metric choice says "we do not claim to reproduce the published number" (28.5% vs 28.00), and setup.tex says "closer to MADAM-RAG's published score". Suggest "because it comes closest to" in item 1.
2. "Across all settings and conditions at most \truncMaxEmpty\ of responses are empty or aborted; the share with no final-answer line has median \truncMedNoFinal\ and maximum \truncMaxNoFinal\" — the macros (2.1, 0.0, 20.0) carry no unit; a `\%` (or "points") appears to be missing.
3. H6 item: "a paired test of the two spaces' errors" — "spaces" is not defined here (it means the two prediction spaces of App. grid, app_grid.tex line 67). Needs a pointer or a word naming them.
4. Item "Reasoning models": "\Nempty\ responses from the two models (almost all closed-book) were empty because reasoning used the whole budget" reads as near-tautological; consider "returned no text because…".
5. "The answer parser unwraps paired tags (\texttt{<answer>X</answer>}), which no earlier result contained." — "which" can attach to the parser; perhaps "a format no earlier result contained".
6. The deviations list ends at the eighth registration; the glossary/commit history mention a ninth addendum (§17), and §17 deviations are disclosed elsewhere in appendix.tex but not listed here (see cross-section issues).

## Coherence issues across sections
1. Judge model: setup.tex (Metrics paragraph) names "An LLM judge (gpt-6.1-terra)"; this section's Cost paragraph says "the judge, gemini-3.5-flash-lite"; appendix.tex line ~224 (outside-family re-score, §17.3) uses gemini-3.5-flash-lite. Two different judges are both called "the judge"; the Cost sentence should say which one (e.g. "the outside-family judge").
2. Gemini: item "Fifth and sixth registrations" says "Gemini was planned but not run, because no API key was available", while the Cost paragraph and appendix.tex line ~224 report gemini-3.5-flash-lite as a judge. Reconcile (Gemini not run as a generator/detector, later used as a judge).
3. §17 deviations missing from this list: appendix.tex line 13 (TrustRAG system message, §17.2), line 21 (300-character NLI window, "a disclosed deviation, §17.6"), line 225 (judge swapped from Gemini 2.5 Flash). The Deviations list here and the closing sentence ("The addenda (H10--H31 and the re-analyses above)…") do not cover §17. Relatedly, Baselines here says TrustRAG's "system message is not sent", while appendix.tex line 13 reports a run that sends it (§17.2).
4. Model configuration says Qwen ran at Q4_K_M; setup.tex says "4-bit, local" — consistent, fine; Qwen "one seed" (item 2) vs setup.tex, which gives no seed count for Qwen — fine but could be stated once.
