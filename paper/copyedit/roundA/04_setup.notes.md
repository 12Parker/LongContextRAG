# 04_setup notes (Round A)

## Main kinds of change
- Split long sentences: ConflictBank majority variant (parenthetical moved into its own sentence), RAGuard label caveat, WikiContradict scoring, the Statistics "family" sentence, and the Pre-registration addenda / re-analyses sentences.
- Pre-registration paragraph: the H1--H9 statement now follows the sentence that commits the pre-registration ("These hypotheses, H1--H9, are the confirmatory core, registered before any confirmatory data"). Every other status statement keeps its order and wording ("each committed before its own runs", "not registered at all", "after a mock review and before their runs", "All of these are exploratory"). No status moved; "we had already seen" became "already seen".
- ConflictBank: "its registered tests are addenda" became "its registered tests come from the addenda". Same meaning, grammatical.
- Generators and baselines: regrouped so that the prompt-skeleton sentences come first, then the detector variants (crossed design, Qwen2.5-7B detector, H27 controls, H29 probe), then the baselines (MADAM-RAG, Astute RAG, TrustRAG, call counting). Sentences are unchanged except that "We also run the LLM detector on Qwen2.5-7B" is now "we also run it on Qwen2.5-7B", joined to the crossed-design sentence.
- The *robust* definition moved from Statistics to Metrics, next to the bidirectional metric it depends on.
- Removed the redundant "WikiContradict" in "our WikiContradict measure".

## Word count
Prose words 1204 -> 1203. Both checks (check_preservation, check_latex) pass.

## Issues for the authors
1. "questions whose only sign of conflict is such a document count only in ``All'' (Appendix~\ref{app:repro})." The ``All'' split is not defined here. A reviewer cannot tell that it is a table column or split.
2. "where six detectors and six generators form \hTwentyFourCells\ pairs on ConflictBank" -- \hTwentyFourCells = 30, but six by six is 36. If self-pairs (or some pairs) are excluded, say so. Otherwise the count reads as an error. Glossary also says "six detectors × six generators".
3. "A final round of re-analyses (...) was not registered at all. A ninth addendum (\S17) registered further re-analyses, ..." Calling the unregistered round "final" and then describing a later ninth addendum is confusing. Also, does "All of these are exploratory" cover the corrected baselines and the semantic re-score of §17, or only the re-analyses? The scope is ambiguous, and only the authors can settle it.
4. "We rerun the oracle and LLM-detector conditions with the minimal skeleton." The benchmark is not named, and the RAMDocs primary skeleton (presumably default) is never stated in this section. Only ConflictBank's is.
5. "we also run it on Qwen2.5-7B with gpt-4o-mini generating" (originally "We also run the LLM detector on Qwen2.5-7B with gpt-4o-mini generating"). "on" could mean the detector model or the data. Presumably Qwen2.5-7B is the detector model.
6. "\texttt{gpt-4o-mini}" and "\texttt{gpt-4.1}" are in \texttt here, but later in the same paragraph (and in Method) model names are plain text ("The LLM detector is gpt-4o-mini"). The formatting is inconsistent. Not changed, because \texttt content is protected.
7. "two current reasoning models, \texttt{gpt-6-luna} (3 seeds) and \texttt{gpt-6.1-terra} (seed 0; one sample)". "seed 0; one sample" is redundant next to Claude Sonnet 5.5's plain "(seed 0)". Consider making them uniform.

## Coherence issues across sections
- Limitations (sections/limitations.tex, \paragraph{Pre-registration}) mentions only H1--H9 and the eight addenda. It does not mention the ninth addendum (§17) or the unregistered final round that Setup's Pre-registration paragraph describes. Results opening (sections/results.tex, lines 4--6) lists three statuses, but Setup has four (adding §17).
- Intro (sections/intro.tex, line 15) says the eight addenda "extended the study to ConflictBank". Setup's ConflictBank paragraph says only that "its registered tests come from the addenda". This is consistent, but the intro does not mention the ninth addendum either.
