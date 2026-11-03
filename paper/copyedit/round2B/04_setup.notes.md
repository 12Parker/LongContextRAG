# 04_setup: Round 2B notes (sentence and paragraph craft)

The section was already tight after Rounds A–C and 2A, so the changes are few and are mostly about order.

## Changes

- **ConflictBank paragraph order.** "ConflictBank had no pilot: its registered tests come from the addenda ..., and
  everything else on it is exploratory." sat between the sampling sentence and the *majority* variant and broke up
  the data description. It now closes the paragraph, so the paragraph runs from data to variant to registration status.
- **Registration paragraph order.** "Unregistered comparisons, including every real-detector row of Table main, are
  exploratory." sat between the confirmatory core (H1–H9) and the addenda (H10–H31). It is now attached to the
  closing "Re-analyses ... are exploratory (Table claims)" sentence as "..., as are unregistered comparisons,
  including every real-detector row of Table~\ref{tab:main}." The paragraph now goes core → addenda → §19 → what is
  exploratory. The "exploratory" label still applies to the same comparisons.
- **CONFLICTS:** "We also run" → "We run" (the redundant "also").

Unchanged: all numbers, macros, hedges, registration labels, refs, citations and table rows. Both checks pass.

## Word count

Prose words 1228 → 1227 (total 1319 → 1318).

## Issues for the authors

1. The §19 list isn't parallel: "a prospective test on 400 unseen ConflictBank questions (...), a full judge
   re-score and HealthContradict." The last item is a dataset name, not a test. "and a HealthContradict test" would
   fix it, but it adds a word, so I left it.
2. Metrics names scoring for WikiContradict, ConflictBank and RAGuard but not for HealthContradict (yes/no) or CONFLICTS:
   "On WikiContradict an answer is correct if it contains both sides (bidirectional matching); on ConflictBank and
   RAGuard we use the RAMDocs metrics."
3. "We rerun the oracle and LLM-detector conditions with the minimal skeleton." This does not say on which benchmark.
   It follows the ConflictBank skeleton sentence but could be read as RAMDocs (Results reports minimal-prompt
   gpt-4o-mini on RAMDocs).
4. Round 2A items 1–5 still stand. The outside-family H38 judge is not named in Setup; WikiContradict has "so we
   measure coverage"; "All" is undefined; HealthContradict uses "items" where the other benchmarks use "questions"; the
   20 fresh cells are not described.

## Coherence issues across sections

- None new beyond Round 2A's two items: the gpt-6.1-terra agreement judge in Setup Metrics vs the gemini-3.5-flash-lite
  judge in the abstract, Results and app:fresh, and the H32–H38 omission in app:repro's closing deviation line.
