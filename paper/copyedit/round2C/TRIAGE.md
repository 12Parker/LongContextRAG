# Round 2C triage: the reader's view and concision (second pass)

**Principles:** Peyton Jones (one idea; cut restatement); Mensh & Kording (each section does its own job); omit
needless words; use the defined term. Cut target 3% per part; no whole paragraph deleted.

## Per-part results

Prose 9,608 → 9,315 (−3.0%). Every part passes both checks except Results: its one difference is the Round 2B fold
("equivalent within $\pm$3 under the matcher but not under the judge (\jdThirtyJ\ \jdThirtyJCI; H38a)"), which the
Results editor applied from an early draft of the 2B triage. Same claim; accepted. The same editor also applied
"does not significantly change accuracy" for "changes accuracy insignificantly" (same meaning).

| Part | Change |
|---|---|
| abstract | −3.2% |
| intro | −1.5% (after restorations below) |
| method | −3.2% |
| setup | −3.0% |
| results | −3.1% |
| discussion | 0 |
| limitations | −3.1% |
| app. part 1 | −3.1% |
| app. part 2 | −3.0% |
| app. §19 | −3.1% |
| reproducibility | −3.1% |

## Reviewed

- **Setup:** "These hypotheses, H1--H9, are the confirmatory core, registered before any confirmatory data" folded into
  the pre-registration sentence ("Before running anything on the remaining 300 questions … the confirmatory
  hypotheses, H1--H9"). Timing and label kept. Accepted.
- **Limitations:** "not ruled out as sources of their direction" → "remain possible sources of the detectors'
  direction". Same strength. Accepted.
- **App. §19:** H37b "within $\pm$3" is the registered TOST at ±3 on the 90% CI (`revision10.py`, `equivalent3`;
  PREREGISTRATION §19.3). Kept.

## Fixed after the round (coordinator)

1. **Intro:** "We release all materials." restored to "We release the harness, pre-registrations and every run."
   (a contribution statement; matches the abstract).
2. **Intro:** "what must the detector get right?" restored to "… right for it to help?" (the question's point).
3. **main.tex:** `\clearpage` before Limitations. The cuts moved Limitations onto page 8; the standing rule puts it at
   the top of page 9. About a fifth of page 8 is now free.

## Whole paper vs HEAD (5133341)

- `check_preservation.py --paper`: words 10,133 → 9,892 (parts edited this pass); no number gone; the two removed
  occurrences are the H10--H31 → H10--H38 fix and the ±3 fold.
- Build: no errors, no undefined references; overfull-box lines unchanged (116); 56 pages (55 before).
- Tests pass.

## Left for the authors

- Free space at the bottom of page 8 (about a fifth of a page): use it, or keep the page break.
- See round2A and round2B TRIAGE for the remaining items (data configuration for ConflictBank-fresh and
  HealthContradict; naming of the judge, "frozen"/"prospective", "string metric"/"matcher"; the abstract's H30 judge
  result; "Frozen …" modifying the follow rates).
