# 11_app_repro — Round 2A notes (coherence and clarity)

The file is unchanged since Round C (the §18/§19 revisions added no text here), so the edits are light.

## Word count
Prose 1713 -> 1712. `check_preservation.py` and `check_latex.py`: OK.

## Main kinds of change
- Split long sentences:
  - the Ninth-registration item (two deviations, now one sentence each);
  - the Third/fourth item (seed-0 consequence);
  - TrustRAG's system-message sentence;
  - the truncation sentence in Model configuration.
- Paragraph head now states the point: "RAMDocs labels." -> "RAMDocs document types are imperfect." The body opens "In the release, 163 documents typed correct ..." so the claim isn't stated twice.
- Glossary wording: "now non-significant" -> "now not significant" (Significance item).
- Clarity: "the response cache served the duplicates entirely" -> "served every duplicate" (Placement modes).

## Issues for the authors
1. **The closing sentence of the Deviations list is out of date:** "The addenda (H10--H31 and the re-analyses above) and the Qwen seed change were each committed before the runs they cover." The abstract, intro and Limitations all say H10--H38 in nine addenda. This can't be fixed here because the fix changes a number.
2. **The Deviations list has no item for §18 or §19 (H32--H38).** Setup says this appendix "lists every deviation". If §19 had none, a short "§19: no deviations" item would close the list. Otherwise the deviations need adding. (Content, with digits, so left alone.)
3. **Ordinals:** the items count registrations ("Second registration" ... "Ninth registration" = §17), but the rest of the paper counts addenda ("nine addenda", with §19 the last). A reader may take "Ninth registration" for the ninth addendum, which is §19. Consider renaming the items or adding one line that explains the count.
4. **Model configuration dates:** "Models were called by these names between 2026-10-26 and 2026-10-29". Please confirm the window covers the §19 runs (ConflictBank-fresh, HealthContradict, the H38 full judge re-score).
5. **Data configuration for the §19 datasets:** ConflictBank and RAGuard each get a paragraph here (release revision, licence, sampling). ConflictBank-fresh's sampling rule and HealthContradict's release, revision and licence have none, and the §19 appendix (`10_app_fresh`) gives none either.
6. **Cost:** \totalSpend and the judge price line should cover the §19 runs, including the H38 full re-score by the same §17.3 judge. Please confirm that the macro was regenerated after §19.

## Coherence issues across sections
- `sections/setup.tex` (Pre-registration and splits) says "Later addenda registered H10--H31 ... The last (\S19) registered H32--H38". Read together, that is consistent. But this appendix's closing sentence (issue 1) says only H10--H31, and `limitations.tex` says "H10--H38 were registered in nine addenda".
- `sections/setup.tex` ends that paragraph with "Appendix~\ref{app:repro} lists every deviation", but this appendix's list stops at §17 (issue 2).
