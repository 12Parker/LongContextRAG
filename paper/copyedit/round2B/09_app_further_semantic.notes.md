# 09_app_further_semantic: Round 2B notes

## Main kinds of change
- **Neutral verb for signed macros (Qwen detector).** "Its directional flag gains \zQDet…DirectionalConflict … and \zQDet…DirectionalClean (= 0.0) on clean ones, and its neutral flag \zQDet…FlagConflict (= −2.6)" → "changes accuracy by … ; its neutral flag, by …". The macros are signed, so the sign reading is unchanged; "gains 0.0" and "gains −2.6" no longer appear (Round B item 1, Round 2A item 6).
- **Glossary term, shorter.** "an LLM from outside our generators' families" → "an outside-family LLM" (glossary: outside-family judge; matches the paragraph head).
- **No bare pronoun openers.** "They depend on condition:" → "Their share depends on condition:"; RAGuard "It finds the misleading post false" → "The judge finds …".
- **Antecedent.** "these intervals are too wide" → "the re-score's intervals are too wide" (Round A's "these intervals" antecedent issue).
- **Compared numbers side by side.** "Two no-flag seeds disagree on A% and B% on the cells that have them, where flag and key disagree on C% and D%" → "On the cells with two no-flag seeds, the seeds disagree on A% and B%, and flag and key on C% and D%." The cell restriction is now in the topic position and the two rates follow each other.
- Unchanged: every paragraph head, number, macro, hedge ("unadjusted", "exploratory", "One untested reading", "significant after Holm"), and the Round A/B/C/2A fixes.

## Word count
Prose 1256 → 1256 (total 1256 → 1256). Both checks pass (`check_preservation.py`, `check_latex.py`).

## Issues for the authors
Items from Rounds A–C and 2A (addendum pointer, unlabelled vs excluded, unnamed RAGuard judge, "outside ±3" for an interval that crosses 3, unnamed benchmarks for the two Haiku/mini contrasts, missing H23/H30/H38 IDs, per-question agreement under an unrelated head, "Selection" head, R/W undefined here) still apply and are not repeated. New:
1. "Its directional flag changes accuracy by … on conflict questions and … on clean ones; its neutral flag, by \zQDetLlmDirFlagConflict." The neutral-flag value does not say which questions it covers (the macro name says conflict). Adding "on conflict questions" costs three words.
2. "They give \mixRdLunaNAnswers\ and \mixRdTerraNAnswers\ answers on average …" starts with a bare "They" (the two reasoning models); "The two models give" adds words.
3. "On labelled questions it is \gptlabLlmDir\ \gptlabLlmDirCI." Bare "it" (the gain); "the gain is" adds a word.
4. "(a Claude Sonnet output), counted as flagging nothing." The participle has no clear subject (the failed parses); "; we count these as flagging nothing" adds words.
5. "A blinded, paired human-audit sheet of 150 of these pairs …" "these pairs" (contrast-arm output pairs) is not named in the paragraph; consider "150 of these output pairs".

## Coherence issues across sections
- None new beyond Round 2A's (verdict-key definition in results.tex vs this appendix and App. fresh; H38 not named here though results.tex and App. fresh use it).
