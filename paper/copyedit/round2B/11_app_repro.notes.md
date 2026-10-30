# 11_app_repro (Appendix: Reproducibility), Round 2B notes

## Main kinds of change

The section had already been through three rounds and reads cleanly, so the edits are few and local:
- Removed the bare "This" opener in the Permutation-resolution item: "This changes only $p$-values at their floor, and
  only lowers them." became "Only $p$-values at their floor change, and only downward." (Both "only"s are kept.)
- Made "we" the subject and put the action in the verb where the passive hid the actor (Williams). This also matches
  the "We port MADAM-RAG … we use …" pattern in the Baselines paragraph:
  - "The registered window's results are reported unchanged." became "We report the registered window's results unchanged."
  - "A generated passage … is dropped" became "We drop a generated passage …"
  - "TrustRAG is ported from its repository" became "We port TrustRAG from its repository"
  - "the unmodified variants are also reported" became "we also report the unmodified variants"
  - "Questions are sampled round-robin …" became "We sample questions round-robin …"
- Joined the two TrustRAG system-message sentences with a semicolon, because the second only completes the first.

No numbers, macros, revisions, dates, prices, identifiers, hedges, labels or citations were changed. Both checks pass.

## Word count

Prose words: 1749 before, 1747 after.

## Issues for the authors

1. Cost paragraph: "The \S17.3 judge, gemini-3.5-flash-lite, cost \$0.30 input and \$2.50 output (checked on
   2026-10-29)." The sentence does not repeat "per million … tokens", so "cost" reads as a total spend and not a price.
   A suggested wording is "is priced at \$0.30 and \$2.50 per million input and output tokens". It is not applied
   because it lengthens the section.
2. RAMDocs paragraph: "Treating them as misinformation is a robustness check." "them" could refer to the 163 documents
   or to the 55/35 questions in the sentence before. The documents are probably meant.
3. Tenth and eleventh registrations: "The \S18 pair-rule counts were computed during review triage, before \S18 was
   written (\S18.5); its other analyses were not." The words "were not" leave it unclear what the other analyses were
   not: computed during triage, or computed before \S18 was written.
4. Seventh registration: "To finish within the session, the \S15 runs ran in parallel tracks, not strictly in
   priority order". The phrase "runs ran" is awkward, and the purpose clause has no agent. "we ran the \S15 runs …"
   would fix both but adds a word.
5. Significance item: "so some exploratory differences once bold are now not significant". "once bold" is shorthand
   for "once shown in bold in a table".

## Coherence issues across sections

- In the deviations list, \S11–\S19, \S17.3, \S17.6 and \S18.5 are sections of the pre-registration document. The
  paper uses "Section~\ref{…}" for its own sections, and readers may take "\S" numbers for paper sections. The
  locations are app:repro (the whole deviations list and the Cost paragraph's "\S17.3 judge") and setup.tex line 61
  ("addendum (\S19) registered H32--H38"), which is the only place where the convention is implied. One sentence
  saying that "\S" numbers refer to the pre-registration's addenda would remove the ambiguity.
