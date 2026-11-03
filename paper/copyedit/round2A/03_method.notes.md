# 03_method: Round 2A notes

The only text changed since Round C is the Oracle parenthetical "(as the benchmark's labels define them)" (§18 pair-rule audit). It reads cleanly and was kept as written. The section otherwise needed little, so the edits are small.

## Main kinds of change
- LLM detector: split the ~40-word sentence (Round A issue 1). "..., and names the side more likely ..." became "... temporal relations. It names the side more likely ...". No net words added.
- Random: the trailing appositive "..., a placebo." now leads: "is a placebo flagging each pair independently at CAAG's per-pair flag rate." The item now opens with its role.
- Flag (neutral): "CAAG's original, except that every detector's flags render identically" became "CAAG's original, but every detector's flags render identically" (one word shorter, which pays for the Random change).
- Unchanged: every \texttt{} header, quoted prompt string, model name, math, item head, label and pointer.

## Word count
623 -> 623 (prose 623 -> 623). `check_preservation.py` and `check_latex.py` both pass.

## Issues for the authors
1. "\item[LLM] ... It names the side more likely to be misinformation (or ``both'' or ``unclear''). It is gpt-4o-mini (temperature 0) unless stated; ..." Two sentences in a row now start with "It". Both refer to the LLM detector, so they are unambiguous, but naming the detector costs a word.
2. "To separate a detector's accuracy from its identity, we build detectors ... (36 grid cells), each rendered as a directional flag and as removal." The sentence is about 35 words with its math. Splitting it ("Each cell is rendered ...") adds two words.
3. "flags exactly the gold contradiction pairs (as the benchmark's labels define them)": "gold" and the parenthetical both say "from the benchmark's labels". This is kept because the parenthetical was added in review. RAMDocs' pair rule (Setup) is what the parenthetical means, and a pointer to it would help, but it costs words.
4. Earlier rounds' items still stand and are not repeated in full:
   - the crossed design is undefined here;
   - where the neutral flag's pair list sits in the prompt;
   - how the directional flag renders suspect-less pairs;
   - the source of the answer key's wrong answers;
   - "Remove (suspect)" vs "removal";
   - "detector pipeline" is defined only in Setup;
   - the lowercase "v1" opens a sentence.

## Coherence issues across sections
- Nothing new. The earlier items still stand:
  - "forced choice" (intro.tex, results.tex) vs "forced skeleton" here;
  - intro.tex quotes the neutral flag as ``these passages may conflict'' and abstract.tex as ``these passages conflict'', but the text rendered here is \texttt{possible contradiction}.
