# 07_limitations (Limitations and Ethics Statement), Round C notes

## Main kinds of change
- Dropped empty words and fillers: "The original hypotheses (H1--H9)" became "H1--H9", "The confirmatory evidence" became "Confirmatory evidence", "Our results show that" was removed in Ethics (the claim and its numbers stay), and I dropped "we" where it was repeated, "the earlier", "measured", "any", "explicit", "shown" in "(both shown in ...)", and "but" in "but not an attacker".
- Merged or tightened sentences without losing content. The concurrent-runs sentence, the MADAM-RAG reproduction pair (now one sentence joined by "but"), "Position, surface style and deeper regularities ... are therefore not ruled out", "so accuracy ... falls below chance" (was "which is why"), "Synthetic false flags are spread uniformly with a random suspect side", "may trade off differently", "ran only at default temperature and so sample nondeterministically".
- Used the shorter defined term where the context already fixes it: "In the grid" (the paragraph head is "Direction and the grid"), "its oracle" and "they" for RAGuard's posts, "posts in context", "139 with conflicts" (matches ConflictBank's "300 with conflicts" in the same sentence), and "a 4-bit Qwen2.5-7B".
- No limitation was cut or softened. Every hedge stays: "not significant", "may", "suggests", "only", "roughly", "retrospective", "post-hoc", "registered". No paragraph was deleted and no line breaks were added.

## Word count
- 1197 -> 1149 words (prose, per `check_preservation.py`): 4.0% cut. Both checks pass.

## Issues for the authors
1. "Its directional flag's gain over all ConflictBank questions with gpt-4o-mini is not significant after correction (H15)". The glossary says "significant" already means significant under Holm, so "after correction" either repeats that or quietly signals that the uncorrected p was below 0.05. If the second reading is meant, say so outright, for example "(uncorrected p < 0.05)". Otherwise the phrase could go. I left it as it was.
2. "An inverted oracle that names the correct side as misinformation costs ...". The relative clause repeats the glossary definition of *inverted oracle*. I kept it because Ethics may be read on its own.
3. "and we compare against no trained detector such as ConflictRAG's". This was reworded from "we do not compare against a trained detector such as ConflictRAG's" and the meaning is the same. Please check that the register reads well.

## Coherence issues across sections
- Precision for Qwen: Limitations says "a 4-bit Qwen2.5-7B" (was "at 4-bit precision") and setup.tex line 45 says "Qwen2.5-7B-Instruct ... (4-bit, local)". The two are consistent. Limitations does not say "-Instruct", and setup.tex does.
- Definition of direction precision: it is defined only in Limitations (limitations.tex lines 54-56). app_grid.tex line 54 and appendix.tex line 117 use "direction accuracy" for the pair-level summary. If these are the same quantity, two terms are in use (Limitations vs app_grid.tex:54).
