# 02_related: Round A notes

## Main kinds of change
- Split long sentences at semicolons/coordinators (huang2025trust sentence; cattan/hou sentence; LLM-conflict-behaviour sentence; TCR ... InstructRAG list split into "learn or decode around" vs "handle without a detector"; turpin/sharma vs bansal/mozannar).
- Reordered for coherence: the RGB "warning" sentence moved from the "which passage is wrong" paragraph to the "Telling the model about conflicts" paragraph, right after our neutral-flag sentence it relates to. The "What we add ..." contribution sentence moved to the end of its paragraph, after all the related work it contrasts with (contribution text unchanged except the term below).
- Grammar/parallelism: "as in X, who find" -> "as do X, who find"; "Of the conflict types of" -> "Among the conflict types of"; "which of context and memory to trust" -> "whether to trust context or memory"; "the more frequent side, the one matching their prior and LLM-generated passages" -> "the more frequent side, the side matching their prior, and LLM-generated passages" (three-item list made explicit); "which is the condition" -> "the condition".
- Terminology: "conflict-free questions" -> "clean questions" (glossary).
- All citations kept, each with the same finding attributed.

## Word count
648 -> 644 (prose 648 -> 644). Both checks pass.

## Issues for the authors
1. Claim strength vs intro: "In our neutral-flag result the hint is true and the input contradictory, and accuracy still falls, because the hint does not say which side to trust." The intro/abstract say the neutral flag "does not reliably help" (and that under forced choice "the effect is small"); "accuracy still falls" plus an unhedged "because" is stronger. Not changed.
2. Break-even framing: "... and a break-even account of when a real detector's verdict helps a given generator." The intro and glossary say break-even precision calibrates randomised flags on true pairs and is not a decision rule; for real detectors the (retrospective) follow-rate account applies and "precision alone can mislead". Consider aligning. Not changed.
3. Ambiguous antecedent: "answer-key controls that separate the verdict from the marking, its cost on clean questions" -- "its" could be the verdict, the direction or the flag. Naming it would add words.
4. RAGuard (used as a benchmark in Setup, cited there as zeng2025raguard) is not mentioned in the Benchmarks paragraph; adding it would add words and a citation here.
5. "\citet{pan2023risk}'s finding that vigilant prompts give limited protection" is cited twice in the section (paragraphs 3 and 4); kept, as both carry content.

## Coherence issues across sections
- "conflict-free questions" vs glossary "clean questions": limitations.tex line 82 ("We therefore report harm on conflict-free questions alongside every gain") and app_grid.tex line 26 ("so conflict-free questions receive"); this section now says "clean questions".
- Related ("accuracy still falls") vs intro first findings bullet ("A correct neutral flag does not reliably help"): see issue 1.
- Related ("break-even account of when a real detector's verdict helps") vs intro third bullet ("It calibrates flags on true pairs; for real detectors, the follow rates ... account for the effects (retrospective ...)"): see issue 2.
