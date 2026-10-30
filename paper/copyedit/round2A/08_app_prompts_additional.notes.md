# 08_app_prompts_additional: Round 2A notes

## Changes (light; the section was already edited in Round A–C)
- Split three long sentences: TrustRAG system-message sentence (semicolon → full stop); NLI windowing sentence ("includes both answers, but …" → "keeps both answers. But …"; "and NLI flags no pair" kept, no causal "so" added); CAAG retuning sentence ("misleads: over …, but by flagging" → "misleads. Over …. But it flags").
- "its sentences closest to the question" → "the sentences closest to the question".
- No numbers, macros, hedges, labels, table rows or prompts touched.

## Word count
- words 1311 → 1311; prose 833 → 833. Both checks (preservation, LaTeX) pass.

## Issues for the authors
1. **"our pipeline" is undefined.** "the registered H14 compared our pipeline under the multi-answer prompt with the baselines' own one-answer prompts" / "our pipeline beats them by …" / "does not differ significantly from the pipeline (H18 …)". Setup defines "A detector pipeline"; the next sentence of this paragraph says "the LLM directional flag". Say which detector/format "our pipeline" is (adding "detector" costs a word per use).
2. **Scope of "with policies matched".** "Against the corrected TrustRAG, with policies matched (Holm over the four), the LLM directional flag changes accuracy by \blHFourteenCbs\ on ConflictBank (one-answer), \blHFourteenRd\ on RAMDocs (multi-answer), \blHFourteenMj\ under majority misinformation, and \blHFourteenCb\ under the registered mismatched prompts." The fourth comparison is explicitly mismatched, so "with policies matched" cannot cover all four. Also no "points" unit on these four macros (other sentences in the paragraph say "points"), and no statement of which of the four is significant under that Holm.
3. **"Its flag changes conflict accuracy by \blNliSentFlag\ points (removal: \blNliSentRem)."** "Its flag" = the 300-character NLI neutral flag, presumably; consider "Its neutral flag".
4. **"It is no different from the uniform format."** (Original CAAG format) — "uniform format" undefined (already on Round A's list) and "no different" has no test named (significant? equivalent?).
5. **Tense:** the CAAG paragraph mixes past ("CAAG did not beat random flags … only deleted correct passages") with present elsewhere; kept past because present "only deletes" could read as a general claim.
6. **Bold heads not restated as findings** ("Published baselines (H14).", "Robustness to the metric and the labels.", "Original CAAG format.") because a finding-head would drop the "exploratory" tag or mix the paragraph's several results. The H14 paragraph also covers the NLI window fix and H18, which the head does not announce; a separate \paragraph would add words.
7. **tab:claims caption** defines "Registered addendum" and "Re-analysis", but rows also use "unregistered re-analysis", "re-analysis of data already seen", "unregistered; retrospective" and "registered (\S19)"; the caption could map these to its two definitions. Rows not edited.

## Coherence issues across sections
- "our pipeline" (appendix.tex, Published baselines paragraph) vs "A detector pipeline counts as one detector call plus one generator call" (setup.tex, Generators and baselines).
- "uniform format" (appendix.tex, Original CAAG format) has no definition in method.tex's format list.
