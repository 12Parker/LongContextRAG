# 05_results — Round 2A notes (coherence and clarity)

## Main changes

- **Stranded sentence moved back.** "The least precise detector (gpt-4o-mini, \dirPrecMini\%) lowers accuracy for the four most accurate generators, significantly for gpt-6.1-terra (H19)." had ended up at the end of the new H32 paragraph, where it read as a result on the fresh cells. In the pre-§19 text (commit 067e5bc) it closed "Real detectors are selective" (the H24 cells). It is back there, after the H24-correlation sentence.
- **H32 paragraph.** The long results sentence is split into two. The slope now comes first, so "against \frMaeGen\ for generator means (cell-level rates alone: …)" attaches to the mean error it is compared with. The sign sentence stands alone. "It overstates harm" → "The account overstates harm", because "It" followed "the sign".
- **H30/H38a/H27 paragraph.** One sentence of about 70 words with nested parentheses is now four sentences:
  1. the verdict key matches the flag (H30, no cell significant), pooled and equivalent under the matcher;
  2. the judge's estimate (not within ±3, H38a);
  3. what the key says, and that it is a control, not a deployable format;
  4. H27.
- **H27 subject.** As written, the subject of "beats the detector model's own answer as a hint" was the verdict key. H27 actually tests the LLM directional flag against the hint (`tab_hypotheses.tex`: "LLM dir. flag vs hint"), so the sentence now reads "The flag beats the detector model's own-answer hint …" ("own-answer hint" is the glossary term). Please confirm this is the intended reading.
- **Wording.** Minor: "rather than" → "not".
- **H23/H38b and HealthContradict sentences.** Checked and left unchanged: moving the H38b sentence or rewording the H37b gapping cost words I could not recover. The H38b sentence still follows the pooled equivalence it qualifies ("This rests on the string matcher …").

## Checks

- **Word count:** prose 1266 → 1266.
- `check_preservation.py --only 05_results`: OK.
- `check_latex.py --only 05_results`: OK.

## Issues for the authors

1. **H33's failure is not stated in Results.** It appears only in App. `app:fresh` ("of the \frHarmN\ cells predicted to lose 3 points or more, \frHarmConf\ does so significantly, so H33 fails"). The Results paragraph gives only the positive sign result: "The sign is right in all \frBigN\ cells predicted at $\ge$3 points." I could not add a clause, since it would add words and the digits of the ID. Suggested addition: "; of the \frHarmN\ predicted to lose $\ge$3, only \frHarmConf\ does so significantly (H33 fails)".
2. **H35 is stated but not labelled or explained.** "The account overstates harm on clean questions (\frCleanPred\ vs \frCleanObs) and on HealthContradict (gpt-6-luna \hcLlmPredLuna\ vs \hcLlmLuna; …)".
   - No hypothesis ID is given (H35).
   - The parentheses do not say which number is predicted and which observed. I tried "predicted …, observed …", but it cost a word I could not recover.
   - The HealthContradict figures are for the LLM directional flag with the gpt-4o-mini detector, which the sentence does not say.
   - Claude Haiku's miss (predicted −16.6, observed +1.1 in `tab:hc`) is larger than gpt-6-luna's and is not mentioned. Report it if the choice of gpt-6-luna is meant as an example.
3. **"the neutral flag, nothing significant (H37b; …)".** The gapped verb reads awkwardly. "the neutral flag changes nothing significantly" costs one word.
4. **The judge is not introduced in Results.** It first appears as "an outside-family judge" (H23 paragraph). Later it is "The judge gives \jdThirtyJ …" (H30 paragraph), with no pointer to App. `app:semantic`/`app:fresh` or its name. See also cross-section item A.
5. **"five detectors (one new)".** The new detector is not named, here or in the sentence.
6. **Two precision macros now share one paragraph.** Moving the H19 sentence back puts `\selHaikuMiniPrec` (69) and `\dirPrecMini` (68.8) for the same gpt-4o-mini detector in the same paragraph. Round A already listed this item; the move only makes it more visible.

## Coherence issues across sections

- **A. Two different LLM judges.** Setup (04_setup.tex, Metrics: "An LLM judge (gpt-6.1-terra) agrees with the primary metric …") uses a different judge from Results' "outside-family judge" (gemini-3.5-flash-lite, §17.3; 05_results.tex, H23 and H30 paragraphs). A reader of Results may take "the judge" to be Setup's.
- **B. The judge has three names.** The intro (01_intro.tex, bullet 2) says "not under an LLM judge". The abstract and Results say "an outside-family judge". Limitations says "that judge"/"an outside-family LLM judge". These are consistent in substance but differ in name.
- **C. Results gives no reason for H35's overstatement.** The H35 explanation in App. fresh ("a flag on a clean question changes fewer correct answers than H31's wrong-flag follow rate implies") has no counterpart in Results. Limitations (07_limitations.tex: "the account overstated harm on clean questions and on HealthContradict") states the fact without the reason. This is consistent, so it is flagged only in case the reason should reach the main text.
