# Round 2A triage: coherence and clarity (second pass, after §18–§19)

**Principles:** define before use; one idea per sentence; paragraph heads state the finding; glossary terms; no growth.
Parts: related.tex and app_grid.tex are unchanged since the first pass and were not edited.

## Per-part results

Editors' parts all passed both checks with no growth (prose 9,564 → 9,560). After the coordinator fixes below:
9,616 prose words; the only check differences are those fixes' deliberate additions.

## Reviewed (meaning-bearing edits)

- **Results:** the H19 sentence ("least precise detector … (H19)") had been left at the end of the new H32 paragraph,
  where it read as a fresh-cell result; moved back after the H24 account (source: results.tex at dac102a).
- **Results:** H27's subject was the verdict key; it is now "The flag beats the detector model's own-answer hint".
  Source: tab_hypotheses (H27 = LLM directional flag vs hint) and PREREGISTRATION H27a.
- **Intro:** the crossed design is no longer listed among the controls; it is its own sentence. Not a claim change.
- **Abstract:** "coverage" → "detections" (the predictor uses detections, `predict19.py`).

## Fixed after the round (coordinator)

1. **Limitations, ethics:** the audit sheet no longer quotes usernames verbatim; now "replaces private users' handles
   with placeholders". Source: commit 5133341 (`export_audit8.redact`).
2. **Results:** H33's failure was stated only in the appendix; added "but only \frHarmConf\ of \frHarmN\ predicted
   harms is significant (H33 fails)". Source: `fresh19["H33"]`.
3. **App. fresh, H38 shift:** absolute-value macros were followed by negative CIs; the signed matcher-minus-judge
   values now carry the CIs. Source: `judge19[...]["matcher_minus_judge"]`.
4. **App. fresh, selection:** the selection effects are over all 400 questions (revision10.py: "selection over all
   questions"), not the table's conflict split; now said.
5. **App. repro:** "The addenda (H10--H31 …)" → H10--H38; deviations list gains §18/§19 (§18.5 timing note; §19
   records none). Source: PREREGISTRATION §18.5, §19; CONTEXT Round 13.
6. Checked, no change needed: the model-call window (2026-10-26 to 2026-10-29) covers the §19 runs (run files dated
   2026-10-29).

## Left for the authors

- Data-configuration details (release, revision, licence) for ConflictBank-fresh and HealthContradict in app:repro.
- Setup never names the outside-family judge used in H38; Results calls it "outside-family judge" without a pointer.
- "frozen" vs "prospective" naming of the H32 test across sections; "string metric" (abstract/intro) vs "matcher".
- Intro/Discussion say the account fails "on real health web pages"; Results says it overstates harm there.
- The abstract omits the H30 judge result (H38a) while giving the H23 one.
- App. part 1: "with policies matched" covers four Holm comparisons, one of which is mismatched; tab:claims caption
  defines two of several status labels.
- `\selHaikuMiniPrec` (69) vs `\dirPrecMini` (68.8): two precision values for one detector, now in one paragraph.
