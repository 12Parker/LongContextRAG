# 04_setup notes (Round B)

## Main kinds of change
- Light-touch only; Round A's restructuring (pre-registration order, Generators regrouping, *robust* moved to Metrics) is kept.
- Bare pronoun openers given nouns: "It had no pilot" -> "ConflictBank had no pilot"; "These are its misinformation, temporal and semantic conflicts" -> "These passages are ..."; RAGuard "They do not verify" -> "The labels do not verify".
- Glossary term at first use: Metrics now says "selective accuracy (correct over answered)" instead of "accuracy among answered questions"; the duplicate gloss in Statistics ("a ratio (correct over answered)") is cut.
- Agent as subject: "A 200-output sheet for a human audit is released, but it has not been labelled" -> "We release a 200-output human-audit sheet; it has not been labelled."
- Comma added before "with bootstrap 95% CIs" so the CIs attach to the reported differences, not to "the plain format".
- Pre-registration and splits paragraph: unchanged. Every registration-status statement (H1--H9 before any confirmatory data; H10--H31 in eight addenda, each before its own runs; last two addenda re-analyses of seen data; final round "not registered at all"; ninth addendum §17 after a mock review and before its runs; "All of these are exploratory"; ConflictBank "registered tests come from the addenda ... everything else on it is exploratory") keeps its wording and scope.

## Word count
Prose words 1206 -> 1202. check_preservation.py and check_latex.py: OK.

## Issues for the authors
(Round A items 1-7 still stand and are not repeated.)
1. "On ConflictBank the primary prompt is the single skeleton; the default and minimal skeletons are robustness checks." This sits in "Generators and baselines" between generator settings and the LLM-detector sentence; it belongs with prompts (Method) or Metrics. Not moved, to keep the paragraph's content fixed.
2. "The reasoning and Claude models run at their default temperature with up to 4,096 output tokens, since they reason before answering." Whether Claude Haiku 5.5 / Sonnet 5.5 run with extended thinking is not stated; "since they reason" implies it for all four.
3. "We score whether an answer gives both." (Data) and "On WikiContradict an answer is correct if it contains both sides (bidirectional matching)" (Metrics) say the same thing twice; one could go to save a line.
4. "Its post labels are behavioural: ... whether GPT-4 judged the claim correctly with it in context" vs the opening "Reddit posts that an LLM labelled supporting, misleading or unrelated". Name the LLM once (GPT-4) or say "an LLM (GPT-4)" in the first sentence so readers do not wonder whether these are two labellers.

## Coherence issues across sections
- Glossary/term "selective accuracy": now defined in Setup Metrics (04_setup); check that Results (05_results) uses "selective accuracy", not "accuracy among answered questions".
- Round A's cross-section notes (Limitations and Results opening vs the four registration statuses in Setup's Pre-registration paragraph) still apply.
