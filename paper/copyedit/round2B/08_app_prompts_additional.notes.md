# 08_app_prompts_additional: Round 2B notes (sentence craft)

## Changes (light; the part has had four editing rounds and has no §18/§19 prose of its own)
- H14 paragraph: "Under the matched one-answer prompt, which we take as the principal comparison," → "Under the matched one-answer prompt (our principal comparison),"; "NLI on ConflictBank first saw …" → "On ConflictBank, NLI first saw …" (setting in topic position, parallel with the paragraph's earlier "On ConflictBank, …").
- Heuristic detector: passive "A pair is flagged if the score reaches $\theta_c$" → "CAAG flags a pair whose score reaches $\theta_c$" (CAAG as subject, as in the surrounding sentences).
- Original CAAG format: bare "It is no different from the uniform format." → "The original format is no different from the uniform one." (noun added after "It"; claim unchanged, "no different" kept).
- Pilot paragraph: subject and verb moved forward ("Before any confirmatory run, the 200-question pilot split, scored with the bidirectional metric, showed the same pattern."); the pattern is now in the stress position and leads into the numbers.
- Prompts, table bodies, captions, numbers, macros, hedges ("exploratory", "registered", "significantly", "likely", "about") untouched. Round A–C and 2A fixes kept ("The pilots" sentences, the split NLI and CAAG-retuning sentences).

## Word count
- words 1311 → 1307; prose 833 → 829. Preservation and LaTeX checks pass.

## Issues for the authors (new; Round 2A's list for this part still stands)
1. **Which of the CAAG results is "it".** "But it flags \caagBestRate\% of all pairs and \caagBestCleanFpr\% of clean questions, which our grid predicts would be harmful under removal (Appendix~\ref{app:grid})." "it" is CAAG at the best-F1 setting; "But at that setting it flags" would be clearer (+3 words).
2. **Pilot numbers are hard-coded.** "Oracle neutral flags changed conflict-question accuracy by $-$10.5 points, directional flags by $+$11.8, and removing the suspect by $+$28.4." These, the NarrativeQA counts ("54\%", "166 pairs"), CAAG's "$6.22\%$" and "134 of its 221 false flags" are literals, not macros from numbers.tex; check they match the released runs.
3. **Inverted-oracle table** (tab:inverted) reports CIs only for the RAMDocs conflict rows; the ConflictBank row and clean columns have none, and the caption does not say why.
4. **"(exploratory)" placement in H14.** "our pipeline beats them by \hFourteenMatchedAst\ and \hFourteenMatchedTr\ points (exploratory)" gives no significance statement for these two numbers, while the next comparisons have "Holm over the four"; readers may wonder whether "beats" is a tested claim.

## Coherence issues across sections
- tab:claims row "Under the judge, neither pooled equivalence holds (\S\ref{sec:direction}, \S\ref{sec:real}) | H38" (appendix.tex) vs the brief/abstract framing that an outside-family judge "favours the flag by about 2.5 points" (abstract.tex): the claims table has no row for the H38a direction or size, only the failure of equivalence.
- The fig:placement caption (appendix.tex) and App. grid both describe "document-level" placement; the pilot paragraph here says the pilots led to "document-level placement" without a pointer to Appendix~\ref{app:grid} (adding one is cheap but adds a word).
