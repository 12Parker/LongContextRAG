# 02_related: Round C notes

## Main kinds of change
- Needless words cut: "who find that" -> "who find" (x2); "finds that" -> "finds"; "already show that" -> "already show"; "Language models" -> "Models"; "including on" -> "including".
- Phrases tightened, same content: "to a degree that depends on the strength of their prior" -> "depending on their prior's strength"; "weigh a passage's relevance more than its credibility cues" -> "weigh relevance over credibility cues"; "each with its own desired behaviour" -> "and each type's desired behaviour"; "shape how models resolve conflicts" -> "shape conflict resolution"; "to the generator in the prompt" -> "into the generator's prompt"; "the cost of a wrong answer" -> "a wrong answer's cost"; "is right less often" -> "is less accurate"; "the pair-specific version of that warning" -> "that warning, made pair-specific"; "a randomised manipulation of direction" -> "a randomised direction manipulation".
- Doubled contrast collapsed: "yet accuracy still falls, because" -> "yet accuracy falls:" (colon keeps the causal link); "and they still rarely" -> "yet they rarely".
- "our removal format" -> "our removal" (matches "the analogue of our removal" later in the section); "explicit contradiction detector" -> "explicit detector".
- Every citation kept, each with the finding attributed to it; contribution sentence ("We add ...") keeps all four items; no paragraph removed.

## Word count
633 -> 600 prose words (-5.2%). Both checks pass.

## Issues for the authors
1. "Advice improves a decision-maker only when its errors fall where the decision-maker is less accurate \citep{bansal2021accurate,mozannar2020defer}, the condition our break-even makes quantitative." Still uses "break-even" as a bare noun and "makes quantitative" may overstate a calibration (Round B issue 1, unresolved).
2. "answer-key controls that separate the verdict from the marking, its cost on clean questions" -- "its" still lacks a clear antecedent (Round A/B issue).
3. "With our neutral flag the hint is true and the input contradictory, yet accuracy falls: the hint does not say which side to trust." The causal reading ("the hint does not say which side") is the authors' interpretation, not a tested result; consider "likely because" or a pointer to the H2/pick-control result.
4. "\citet{pan2023risk} test detection and ``vigilant'' prompting ..." and later "with \citet{pan2023risk}'s finding that vigilant prompts give limited protection" -- the same work is introduced in two paragraphs; merging would save a line but moves content between paragraphs.

## Coherence issues across sections
- Related "With our neutral flag ... accuracy falls" vs intro.tex first findings bullet "A correct neutral flag does not reliably help" (and the pooled estimate being exploratory in results.tex): Related states a uniform fall. Round A noted this; still stands.
