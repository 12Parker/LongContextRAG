# 01_intro — Round 2B notes

## Changes (light; the section had been edited three times already)

- **Paragraph 1:** "They report detection F1 …" began with a bare "They". Joined it to the previous sentence so "Recent
  systems" is the subject of both verbs ("pair … and report …, but not how the two connect"). The cattan citation
  sentence now stands alone.
- **Bullet 3, H32 sentence:** "Frozen in advance, these follow rates and what each detector flags predict …" opened
  on a participle phrase before its subject. Now the subject comes first: "These follow rates and each detector's
  flags, frozen in advance, predict …". Hedge ("though not on real health web pages") unchanged.
- **Punctuation:** added a comma after the introductory phrase in "For real detectors, precision alone can mislead"
  and "At a 5\% share of conflict questions, the LLM …".
- No numbers, macros, citations, hypothesis IDs or contribution claims were changed or cut.

## Word count

Prose 530 → 528 (−0.4%). Both checks (check_preservation.py, check_latex.py) pass.

## Issues for the authors

1. Release statement vs footnote: "We release the harness, pre-registrations and every run." vs the footnote
   "Code, the pre-registration and all run outputs are in the anonymised supplementary material and will be released
   publicly on acceptance." The body uses present-tense "release" and plural "pre-registrations"; the footnote says
   "will be released" and singular "the pre-registration".
2. Bullet 4: "the LLM detector's flag has a negative estimated effect in \prevNegFive\ of \prevN\ settings
   (exploratory)" does not say which flag format (neutral or directional) this is. Results ("its flag's estimate is
   negative in \prevNegFive\ of \prevN\ correlated settings") does not say either.
3. Bullet 3: "A directional flag pays only above a break-even direction precision." The bullet then says "For real
   detectors, precision alone can mislead". There, "precision" could mean detection precision or the break-even
   direction precision. Results calls the threshold "a calibration" that can misjudge selective detectors; consider
   naming which precision is meant.

## Coherence issues across sections

- Intro bullet 4 says "settings"; Results (results.tex, "With rare conflicts…" paragraph) says "correlated settings"
  for the same \prevNegFive\ of \prevN\ count. (Round A listed "correlated settings" as an undefined term. This note
  is only about the two sections using different wording.)
