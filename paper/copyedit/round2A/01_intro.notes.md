# 01_intro: Round 2A notes (coherence and clarity)

## Main kinds of change
- **Paragraph 1, referent of "They".** The two prior-work clauses are now in the other order ("Prompting a model about conflicts can help \citep{cattan…}, and recent systems pair contradiction detectors with prompts flagging conflicting passages \citep{wang2026…}. They report…"). "They" now follows "recent systems" directly instead of the Cattan clause. Both citations are kept.
- **Format list (paragraph 2).** The list now uses semicolons, because "a flag saying one side is false, not which, a directional flag…" read as nested commas. "not which" became "but not which".
- **Addenda sentence (new §19 text).** "extended the study to X, to Y, to Z, and to W" became "added X; Y; Z; and W". The semicolons keep RAGuard's relative clause from merging with the next list item. "whose misleading side is real Reddit posts" became "whose misleading passages are real Reddit posts", which fixes the number agreement that Round A noted.
- **Controls sentence.** The crossed design is no longer listed as a control. It is now its own sentence ("A further \hTwentyFourCells\ cells cross six detectors with six generators."). This resolves Round A item 8 / Round B item 1 without changing content.
- **Bullet 2.** "an LLM judge" became "an outside-family judge", the abstract's and glossary's name for the §17.3/H38 judge.
- **Bullet 3 (new H32 text).**
  - The head now says "break-even direction precision" (the glossary term, and the term used in the bullet body and in Results).
  - The long semicolon sentence is split. "the follow rates, with what each detector flags, predict the effects … when frozen in advance" became "Frozen in advance, these follow rates and what each detector flags predict the flag's effects…". "these" ties "follow rates" back to "generators follow both at similar rates", since the term is not otherwise defined in the intro.
  - "real web pages" became "real health web pages", as in the abstract.
- **Bullet 4.** The semicolon became a full stop: two findings, two sentences.
- Unchanged: every number and macro, every hedge or tag ("exploratory", "on average", "equivalent within $\pm$3", "significantly", "each committed before its runs"), every H-ID and citation, and the footnote.

## Word count
Prose 532 → 530. `check_preservation.py --only 01_intro`: OK. `check_latex.py --only 01_intro`: OK.

## Issues for the authors
1. **The "coded after seeing data" tag is gone from bullet 1.** Round C's intro had "These policy and family splits were coded after seeing data." after bullet 1. The §18/§19 revision dropped it. Bullet 1 still states the splits ("Claude models move less; the minimal prompt is most harmful."), and Results tags them "codings chosen after seeing data". Restoring the tag costs about 8 words (2 are free now).
2. **The verdict finding has left the contributions list.** Round C's bullet 3 ended "The detector's verdict as an answer key matches its flag (H30); its model's own answer as a hint does worse (H27)." That sentence is gone. Yet the paper's one idea, and the abstract ("A detector's verdict, passed as an answer key …, matches its flag in mean accuracy under our metric"), present the verdict as a main finding. Its H38a caveat (the judge puts the flag ahead) appears nowhere in the intro. The intro now names the verdict only as a control.
3. **Bullet 3 may overstate the H32 miss.** Bullet 3 says "predict the flag's effects on unseen questions (H32), though not on real health web pages." Results and the claims table say the account "overstates harm on clean questions (\frCleanPred\ vs \frCleanObs) and on HealthContradict". "Not on" reads as no prediction. The source gives a directional bias, and it also applies to clean questions within ConflictBank-fresh, which the intro does not mention.
4. **Bullet 2 is worded differently from the abstract.** Intro: "on average it is close to an answer key naming the wrong answer". Abstract: "helps, but on average no more than an answer key naming the wrong answer". Under the judge, the flag is ahead by \jdTwentyThreeJ. So "no more than" (abstract) is harder to square with H38b than "close to" (intro).

## Coherence issues across sections
- **The prospective test is named differently.** The intro has "frozen in advance (H32)" and no "prospective". The glossary and Results' head "Frozen in advance, the account predicts new cells (H32)" match the intro. Check that §19/appendix `app:fresh` does not use "prospective" as the only name. Locations: `01_intro.tex` bullet 3; `results.tex` sec:real; appendix `app:fresh`.
- **Judge naming.** The intro now has "outside-family judge", and the abstract has "an outside-family judge". Results says "the judge" and "under the matcher" (sec:direction, H30 paragraph). The intro says "our string metric". All are glossary terms, but the abstract/intro use "string metric" while Results uses "matcher". Locations: `01_intro.tex` bullet 2; `results.tex` H30 paragraph.
