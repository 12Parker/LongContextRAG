# 03_method: Round 2B notes

This section has no new §18/§19 text other than the Oracle parenthetical. Round 2A and the first pass already edited it, so the edits are few and small. Every \texttt{} header, quoted prompt string, model name, math, item head, label and pointer is unchanged.

## Main kinds of change
- Instruction: "...for contradictions. v1 ends ..." became "...for contradictions: v1 ends ...". The lowercase "v1" no longer opens a sentence (a first-pass item), and the colon introduces the two versions. No words added.
- CAAG: "used with its code unchanged" became "with its code unchanged" (−1 word).
- LLM: "lists, as JSON, contradicting pairs, distinguishing them ..." became "lists contradicting pairs as JSON, distinguishing them ...". The verb now sits next to its object and the comma run is gone.
- LLM: "It is gpt-4o-mini (temperature 0) unless stated" became "Its model is gpt-4o-mini ...". This removes the second bare "It" in a row (Round 2A issue 1) and says what is gpt-4o-mini (+1 word, paid for by the CAAG cut).

## Word count
623 -> 623 (prose 623 -> 623). `check_preservation.py` and `check_latex.py` both pass.

## Issues for the authors
1. "\item[NLI] ... a pair is flagged if the larger contradiction probability exceeds $0.5$. It gives no direction." The item's second sentence opens with a bare "It". It is unambiguous here, and the CAAG item says the same thing inside one sentence ("runs locally and gives no direction"). Merging the two would read more parallel.
2. "(36 grid cells), each rendered as a directional flag and as removal." The sentence is still about 35 words with its math (Round 2A issue 2). "removal" here is the format called "Remove (suspect)" in the list above.
3. Earlier rounds' items still stand and are not repeated in full: the crossed design is undefined here; where the neutral flag's pair list sits in the prompt; how the directional flag renders suspect-less pairs; the source of the answer key's wrong answers; "detector pipeline" is defined only in Setup; the Oracle's "gold" plus "(as the benchmark's labels define them)".

## Coherence issues across sections
- Nothing new. The earlier items still stand: "forced choice" (intro.tex, results.tex) vs "forced skeleton" here; the neutral flag is quoted as ``these passages conflict'' in abstract.tex, while the text rendered here is \texttt{possible contradiction}.
