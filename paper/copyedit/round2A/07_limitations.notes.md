# 07_limitations (Limitations and Ethics Statement): Round 2A notes

## Main kinds of change
- **New follow-rate text (Models and detectors).** Split "Its prospective test (H32) uses ... generators, and the account overstated harm on clean questions and on HealthContradict" into two sentences (one idea each), and put the second in the present tense, as in Results ("It overstates harm").
- **Terms.** "The break-even precision is a calibration" -> "The break-even direction precision" (glossary and Results); paid for by the "and" removed in the split.
- **HealthContradict sentence.** "the misinformation page the wrong one" -> "the wrong answer" (the elliptical "one" read as "the wrong page"; wording now matches Appendix~\ref{app:fresh}'s label audit).
- **Judge re-score sentence (H38)** (about 45 words) split before "No human labels check either scorer".
- **Ethics.** The new RAGuard-usernames sentence made "Its main risk" refer to the audit sheet; changed to "The main risk" (this work's).
- No limitation cut or softened; every hedge, macro and hypothesis ID kept.

## Word count
Prose 1237 -> 1236. check_preservation and check_latex: OK.

## Issues for the authors
1. **Ethics, possibly now false:** "we use them as released, and the RAGuard audit sheet quotes them verbatim." HEAD commit 5133341 ("Redact private users' handles in the RAGuard audit sheet") replaces Reddit u/ names, permalink ids and a TikTok URL with placeholders, so the sheet no longer quotes the posts verbatim. Suggest e.g. "quotes them with private users' handles redacted".
2. "Re-scored by that judge over all \jdN\ responses, neither pooled equivalence holds within $\pm$3". "Neither pooled equivalence" does not say which two tests; the parenthesis implies H30a and H23 only by macro order. Naming them ("(H23, H30)") would add digits, so it is left here.
3. "Oracle direction is close to an answer key (H23), so the oracle rows are ceilings". After H38, "close to" holds under the matcher (pooled TOST) but the judge puts the flag ahead by \jdTwentyThreeJ; the Metric paragraph says so, but this sentence carries no "under the matcher".
4. "an LLM judge finds the correct page supporting the gold answer for \hcLabCorrect\% of items": "an LLM judge" is presumably the same outside-family judge (gemini-3.5-flash-lite) as in the Metric paragraph; "the judge" would tie them, but this paragraph comes first.
5. "Its prospective test (H32) uses unseen questions but the same benchmark, prompt and generators": the fresh design adds one new detector (Results, H32 paragraph); "same generators" is right (four of the six), but readers may read "same" as all of them.
6. "HealthContradict's pages are real, but its labels are as released": the sentence does not say that its oracle is therefore a label transfer, as the RAGuard sentence and Appendix~\ref{app:fresh} do ("so the dataset's oracle is a label transfer too"). Adding it costs words.

## Coherence issues across sections
- **Addenda count.** Limitations (Pre-registration) and abstract/intro say "H10--H38 ... nine addenda"; setup.tex line 58 still says "Later addenda registered H10--H31", and appendix.tex line 371 "The addenda (H10--H31 and the re-analyses above)". Round A noted setup.tex calls §17 "a ninth addendum"; with §19 added, check that "nine" counts the same addenda everywhere.
- **Follow-rate harm wording.** Limitations "overstates harm on clean questions and on HealthContradict" (matches Results); Discussion "predicted unseen cells but not real web pages"; abstract "overstate its harm on real health web pages" (no clean-question part).
- **H23 status.** Limitations "Oracle direction is close to an answer key (H23)" and appendix.tex line 42 (claims table, same wording) vs intro "equivalent within ±3 under our string metric, not under an LLM judge". The claims table is generated.
