# 08_app_prompts_additional — Round A notes

## Main kinds of change
- Split long sentences in the H14 paragraph: the ConflictBank prompt-mismatch sentence, the TrustRAG system-message sentence, and the NLI window sentence (about 70 words cut into three sentences). The ambiguous "it" became "our pipeline" / "NLI".
- Caption of `tab:inverted`: made it grammatical ("Adversarial control, gpt-4o-mini: change from no flags, in points") and replaced the unclear "it flags nothing" with "the inverted oracle flags nothing".
- Phase-0 audit: put the two "likely false" sentences next to each other and moved the release note to the end of the paragraph.
- CAAG paragraph: "sets the rate of the random detector" became "sets the random detector's rate" (to keep the word count down).
- Inline table bodies, `\input{generated/...}` lines, labels, refs, macros and numbers are unchanged. Both checks pass.

## Word count
- Total: 1436 -> 1435; prose: 903 -> 902.

## Issues for the authors
1. Sign wording with a negative macro: "the LLM directional flag gains \blHFourteenCbs\ on ConflictBank (one-answer), \blHFourteenRd\ on RAMDocs (multi-answer), \blHFourteenMj\ under majority misinformation, and \blHFourteenCb\ under the registered mismatched prompts." `\blHFourteenMj` is −0.5, so "gains −0.5" reads oddly. Consider "changes accuracy by", or splitting the sentence. I left it unchanged because the verb covers all four macros.
2. Self-reference: "which our grid predicts would be harmful under removal (Appendix~\ref{app:extra})". This sentence is itself in `app:extra`. The grid is in `app:grid`, so the ref should probably point there.
3. Undefined term: "This is no different from the uniform format." "Uniform format" is not defined anywhere else in the paper. The glossary calls it the (neutral) *flag* format. Also, "no different" has no test attached (is it a test, or descriptive?).
4. The lead-in "Published baselines (H14)." covers three things: H14, the NLI window deviation, and H18 (single call). The NLI material is a detector diagnostic, not a published-baseline result. Consider a separate paragraph, or a head that states the finding (e.g. that the matched comparison favours our pipeline).
5. "NLI on ConflictBank first saw passage prefixes that usually omit the changed fact." The window length of that first run is not given here, and no pointer says where it is.
6. "Its mechanism depends on the answer policy: added misinformation, abstention, little effect" (table `tab:claims`, row 2). The three outcomes are not tied to the three policies, so a reader cannot tell which goes with which. This is a table body, so I left it frozen.
7. Pilot paragraph: "Oracle neutral flags changed conflict-question accuracy by $-$10.5 points ..." These numbers are hard-coded rather than macros (unlike the rest of the section). Probably intentional, since they are pilot numbers, but worth a check.
8. The `tab:claims` caption defines "Re-analysis" as "registered after the data were seen, or exploratory". The table body, however, uses several variants ("unregistered re-analysis", "re-analysis of data already seen", "re-analysis (\S17)", "unregistered; retrospective"). The caption does not say how these differ.

## Coherence issues across sections
- results.tex line ~125 ("the registered H14, with mismatched policies, shows no win") against the H14 paragraph here, which reports \blHFourteenCb\ (+1.8, not significant) under the registered mismatched prompts. The two are consistent, but "shows no win" is looser than "not significant". Consider matching the wording.
- results.tex line ~58 cites Table~\ref{tab:inverted} for "costs \aadvDirConf\ points on RAMDocs". The table caption here calls it "the adversarial control". method.tex line 53 introduces it as "inverted oracle, an adversarial control". Consistent; no action needed beyond noting that the caption now names the inverted oracle.
