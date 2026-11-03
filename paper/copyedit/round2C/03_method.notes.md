# 03_method — Round 2C notes

## Main kinds of change
- Dropped repeated nouns where the subject is already clear: "The \emph{minimal} skeleton" → "\emph{Minimal}", "The \emph{single} skeleton" → "\emph{Single}"; "the second passage is deleted" → "its second passage" (Remove), joined into one sentence.
- Removed filler words: "but" (two places), "with its", "that ... and" in Single call, the serial comma before "and an instruction".
- Shorter phrasing with the same content: "the side more likely to be misinformation" → "the likelier misinformation side"; "detectors from gold labels" → "gold-label detectors"; "so they are nested and comparisons across cells are paired" → "so they nest and cross-cell comparisons are paired"; "an adversarial control," → "(adversarial control)".
- Quoted prompt and flag text (`\texttt{}`, quoted instructions), every number, label, ref, citation and model name are unchanged.

## Word count
623 → 603 prose words (−3.2%; target 3%). Both checks (`check_preservation.py`, `check_latex.py`) pass.

## Issues for the authors
- Detectors, Random: "is a placebo flagging each pair independently at CAAG's per-pair flag rate." It does not say whether the rate is matched per benchmark or pooled. Add the scope if it is not given in the setup or appendix.

## Coherence issues across sections
- Model-name formatting: method (03, LLM detector item) writes gpt-6-luna and gpt-6.1-terra as plain text ("in the crossed design gpt-4.1, gpt-6-luna, gpt-6.1-terra, Claude Haiku 5.5"), but setup (04, line 68) writes them as `\texttt{gpt-6-luna}` and `\texttt{gpt-6.1-terra}` (and `\texttt{gpt-4o-mini}` at line 66). Results and the appendices use plain text. Pick one style.
