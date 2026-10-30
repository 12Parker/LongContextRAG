# 01_intro: Round C notes

## Main kinds of change
- Deleted the sentence "A natural fix detects the contradiction and tells the generator." It has no number, citation or claim, and the next sentence says the same thing with a citation ("Recent systems pair contradiction detectors with prompts flagging conflicting passages").
- Shortened relative clauses to participles: "prompts that flag" -> "prompts flagging"; "the format that passes" -> "the format passing"; "a flag that says one side is false without saying which" -> "a flag saying one side is false, not which"; "flag that names" -> "flag naming"; "answer key that names" -> "answer key naming".
- Used defined terms: "(including a variant where the misinformation outnumbers the truth)" -> "(including a majority-misinformation variant)". This is the glossary's *majority* variant, and results uses the phrase "majority misinformation".
- Cut empty words: "These systems" -> "They"; "our hypotheses (H1--H9)" -> "hypotheses H1--H9"; "the retrieved passages" -> "the passages"; "add the misinformation to their answer" -> "also give the misinformation"; "the most harmful" -> "most harmful"; "combined with" -> "with"; "the detector model's own answer" -> "its model's own answer", matching the controls sentence.
- No contribution bullet, hedge or tag was cut. "coded after seeing data", "(retrospective; ...)", "(exploratory)", "on average", "equivalent", "not reliably", "significantly", "each committed before its runs" and all H-IDs are kept. The footnote is unchanged (check_latex requires it verbatim).

## Word count
Prose 563 -> 534 (-29 words, -5.2%). Both checks pass (check_preservation, check_latex).

## Issues for the authors
1. This sentence may overstate the result: "Naming the false side helps on every benchmark that labels one, but on average it is equivalent to an answer key naming the wrong answer". Results (sec:direction) reports exceptions: gpt-6.1-terra on RAGuard is not significant, and on RAMDocs there is no significant gain "for Qwen2.5-7B or for gpt-4o-mini under the minimal prompt (H10)". "helps on every benchmark" is true only per benchmark, not per setting; the authors may want "on every benchmark, for most generators" or similar. Not changed: it is a claim.

## Coherence issues across sections
1. The intro bullet head is "A correct neutral flag does not reliably help." and the abstract says "A perfectly accurate neutral flag (``these passages conflict'') does not reliably help." These are two wordings for the oracle neutral flag; using one would be clearer.
