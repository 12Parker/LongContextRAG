# 07_limitations (Limitations and Ethics Statement): Round 2C notes (concision)

## Main kinds of change
- **Wordy phrases shortened:** "so a test of a detector-level property has little power" -> "so detector-level tests
  have little power"; "ran only at default temperature and so sample nondeterministically" -> ", sampling
  nondeterministically"; "on the claims they answer" -> "on answered claims"; "when asked to judge them" -> "when
  judging them"; "that depend on it" -> "depending on it"; "that name the false side" -> "naming the false side";
  "controls which pairs are flagged" -> "chooses the flagged pairs"; "what a detector does at the document level" ->
  "a detector's document-level behaviour" (Ethics).
- **Repeated words dropped:** "400 sampled questions" -> "400 sampled" (parallel to "300 questions" just before);
  "ConflictBank's generated passages" -> "ConflictBank's passages" (the previous sentence says they are LLM-written);
  "the passages" -> "passages"; "of items" after \hcLabCorrect (the second figure already omits it); "the oracle rows"
  -> "oracle rows"; "over all of them" -> "over all tests"; "not significant after correction" -> "not significant"
  (in this paper "significant" already means Holm-corrected).
- **Sentences joined:** H1--H9 / H10--H38 registration (one "were registered" removed); "Several analyses, labelled as
  such, were chosen after seeing data"; the concurrent-runs sentence ("but identical ones" -> parenthesis); the
  RAMDocs metric sentence ("but ..., so this" -> "; as ..., this").
- **H29 sentence (Round 2B issue 1 resolved):** "... one of four changes. Position, surface style and deeper
  regularities of generated text are therefore not ruled out as sources of their direction." -> "... one of four
  changes, so position, surface style and deeper regularities of generated text remain possible sources of the
  detectors' direction." "their" now names its noun; "not ruled out" and "remain possible" say the same, so the
  limitation is not softened.
- No limitation, hedge, registration label, macro, number, citation or hypothesis ID cut. Round 2A's audit-sheet
  redaction wording and Round 2B's pronoun fixes are kept.

## Word count
Prose 1241 -> 1203 (-3.1%). check_preservation and check_latex: OK.

## Issues for the authors
1. "Direction precision is the share of a detector's flags on true contradictions naming the false side; pair
   precision, the share of its flagged pairs that are true contradictions." Pair precision is still defined with no
   number after it (Round B item); if no number is added, the clause could go.
2. "An inverted oracle that names the correct side as misinformation costs ..." (Ethics) repeats the definition of a
   term Setup defines; kept because Ethics may be read alone.
3. "Every generator is more accurate closed-book than with RAGuard's posts (...), so the posts make a weak retrieval
   setting." "posts" twice in one sentence; "which therefore make" would avoid it but Round 2B chose the noun for
   clarity.

## Coherence issues across sections
- None new. Round 2A's item on "overstates harm on ... HealthContradict" (here and Results) vs Discussion's "but not
  real web pages" still stands.
