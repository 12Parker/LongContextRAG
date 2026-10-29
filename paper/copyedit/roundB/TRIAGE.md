# Round B triage: sentence and paragraph craft

**Principles:** Gopen & Swan (old information first, new information in the stress position, subject then verb); Williams (characters as subjects, actions as verbs, active voice); Mensh & Kording (one message per paragraph, one term per concept); no bare "this"/"it"; no section gets longer.

## Per-part results

Both checks (`check_preservation.py`, `tools/check_latex.py`) pass for every part. Total prose words 10584 → 10535.
- **Body:** abstract −2, intro −4, related −11, method −1, setup −4, results −7, discussion 0, limitations −3.
- **Appendix:** −1, −5, −4 and −7.

## Reviewed

- I read the word diffs of all body parts and saw no change of claim, hedge, sign wording or registration label.
- I checked by hand the sentences that moved "retrospective decomposition, defined after the cells were seen" and the other-seed r (results), and the direction-precision definition (limitations). Each keeps its content and label.
- No paragraph head changed. Round A's fixes are intact.

## Fixed after the round (coordinator)

1. **Intro, "six detectors crossed with six generators"** → "\hTwentyFourCells cells crossing six detectors with six generators". Source: setup.tex (30 of the possible pairs).
2. **Results, H29** "changes the detectors' hit rate insignificantly" → "does not significantly change". This matches Limitations and the glossary's "not significant".

## Left for the authors (new this round; Round A's list still stands)

- **Page fit:** the body still ends about three lines onto page 9 (Round C's cut target should absorb this).
- **Abstract:** the second H31 range ("lowers it by \ahThirtyOneBMax to \ahThirtyOneBMin") has no "points".
- **Intro's "Controls" list** includes the crossed design, which is not a control.
- **Setup:**
  - the ConflictBank primary-skeleton sentence sits under "Generators and baselines";
  - WikiContradict's "gives both" appears in Data and in Metrics;
  - RAGuard's labeller is "an LLM" in one sentence and "GPT-4" in another.
- **Limitations:**
  - H29's ±5 margin vs ±3 elsewhere;
  - `\semDidMax` prints "+3.3" after "more than";
  - pair precision is defined but no number follows;
  - "which is why" (RAGuard below chance) is causal.
- **Claims table:** H29 "equivalence not shown" vs Limitations "shown for only one of four changes". These are consistent, but the table is generated.
- **App. further results:** "the addendum" doesn't say which one. H4a–c count "distinct comparisons", "distinct cells" and "matched cells" (three nouns for one idea).
- **App. reproducibility:** "those not abstained in every seed"; "so it is post hoc" (the margin or the TOST?).
