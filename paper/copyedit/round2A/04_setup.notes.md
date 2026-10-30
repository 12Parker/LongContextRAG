# 04_setup: Round 2A notes (coherence and clarity)

## Changes

- **Data order.** HealthContradict was tacked onto the end of the WikiContradict paragraph. It now has its own
  paragraph, placed between RAGuard and WikiContradict to match the row order of Table `tab:evidence` and to sit
  next to the other "labels as released / label transfer" benchmark.
- **Registration paragraph (new §19 text).**
  - The H10–H31 sentence registered "H10--H31 ... and re-analyses of seen data, some unregistered", which read as
    registering unregistered analyses. The re-analysis clause is now merged with the later "Re-analyses are
    exploratory" sentence: "Re-analyses of seen data, some unregistered (...), are exploratory". This removes the
    repetition of "re-analyses" and keeps the list, "some unregistered" and "exploratory".
  - "The last (\S19)" → "The last addendum (\S19)" (the noun was missing).
  - The §19 list mixed a "with ..." clause into its items; the predictions clause is now a parenthesis on the
    prospective test.
  - "full semantic re-score" → "full judge re-score" (glossary term; Results and App. `app:fresh` call it the judge
    re-score).
- **ConflictBank majority variant.** The 45-word sentence is split: the "three to one" point comes first, then the
  identity of the three passages.
- **Generators.** The replication list mixed commas at two levels ("two reasoning models, X and Y, and Claude ...");
  top-level items are now separated by semicolons.
- **Table caption.** "break-even precision" → "break-even direction precision" (glossary; Results and abstract).

No numbers, macros, hedges, labels, refs or citations changed; both checks pass.

## Word count

Prose words 1228 → 1228 (total 1319 → 1319).

## Issues for the authors

1. The §19 item "a full judge re-score" does not say what is re-scored, and "the judge" here is not the judge
   Setup introduces. Metrics introduces only "An LLM judge (gpt-6.1-terra) agrees with the primary metric on
   \judgeAgree\% ..."; the H38 judge is the outside-family gemini-3.5-flash-lite of §17.3, which Setup never names.
   A reviewer reading the abstract's "an outside-family judge" and Results' "under the judge" has no Setup
   definition. Naming it (and "of the pooled H30a/H23 equivalence tests") would add words, so not done.
2. WikiContradict: "The original evaluation also asks that an answer acknowledge the incompatibility and not prefer a
   side without support ..., so we measure coverage of the competing sources". The "so" does not follow: the
   intended sense seems to be that we measure *only* coverage (a subset of the original rubric). Adding "only"
   would change the hedge, so left.
3. RAMDocs: "questions whose only sign of conflict is such a document count only in ``All''": "All" is a column/
   subset name not defined in Setup; a pointer to the table that has it would help.
4. HealthContradict counts "247 items" where every other benchmark counts "questions"; App. `app:fresh` also uses
   "items". Consider one term.
5. The prospective test's 20 fresh cells (five detectors × four generators) are not mentioned in Generators and
   baselines, which describes only the crossed design's cells; Results and App. `app:fresh` use "fresh cells".
   Adding them costs words and digits.

## Coherence issues across sections

- Setup Metrics names the agreement judge as gpt-6.1-terra; abstract ("an outside-family judge"), Results
  §sec:direction ("an outside-family judge scoring all \jdN\ responses") and App. `app:fresh` ("The \S17.3 judge")
  refer to a different judge (gemini-3.5-flash-lite). Setup is the natural place to name both.
- Setup registration paragraph says the last addendum is §19 and registered H32–H38; App. `app:repro` closing line
  still says "The addenda (H10--H31 and the re-analyses above) ... were each committed before the runs they
  cover", omitting H32–H38 (appendix.tex, end of Deviations list).
