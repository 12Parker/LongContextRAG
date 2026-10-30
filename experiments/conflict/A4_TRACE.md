# A4 trace: where do the paper's CAAG numbers come from?

Checked 2026-10-26 against the paper PDF (Downloads, versions of 2026-05-18 and 2026-06-01, identical
text), the repo at `89e545d`, all git history (`git log --all -S`), and `../LongContextRAG.zip`
(2025-12-01 snapshot).

## Paper Table 2, CAAG column

| Metric | CAAG in paper | Source found |
|---|---|---|
| Avg. overall | 5.01 | none |
| Correctness / completeness / relevance / clarity | 4.38 / 4.26 / 6.78 / 7.46 | none |
| Avg. input tokens | 3,621.0 | none |
| Cost/query | $0.000571 | none |
| Resp. time | 2.02 s | none |
| Cost / latency efficiency | 8,774.08 / 2.48 | none |

**Result: no file in the repo, its history, or the zip snapshot contains any CAAG column value.**

## Why they can't have come from committed code

- `scripts/comparison/compare_systems_narrativeqa.py` never imports
  `hybrid/narrativeqa_hybrid_rag_contradiction_aware.py`. Its systems are base LLM, BM25 RAG,
  neural hybrid, improved hybrid and optimized hybrid.
- The CAAG script itself only prints to stdout (single query or interactive mode). It writes no results
  file and has no batch evaluation path.
- The only CAAG result file, `results/llm_judge_evaluations/llm_judge_eval_contradict.json`, is
  labelled "Expected/mock results" and disagrees with the paper: 4.36 overall (−0.19 vs. its own
  baseline of 4.55), 11 questions from a single story.

## The other three columns, for comparison

- **Quality rows (overall, the four dimensions, median) for Base, BM25 and NQA** match
  `results/llm_judge_evaluations/llm_judge_evaluation_20251201_004022.json` → `system_averages`
  exactly (5.18 / 4.595 / 4.895). "Overall" is the judge's own `overall_score` field averaged; it is
  not a formula over the four dimensions.
- **The token, cost and efficiency rows do not come from that run.**
  - The judge file's mean `context_tokens` is 9,072.82 for Base, 1,599.62 for BM25 and 3,295.74 for NQA.
  - The paper's BM25 (2,330.5 tokens, $0.000383, cost efficiency 10,965.64, latency efficiency 2.20)
    and NQA cost ($0.001222) come from `results/cost_benefit_analysis/cost_benefit_analysis_20251126_232522.json`,
    an earlier run with different quality scores (BM25 4.2).
  - The paper's NQA tokens (5,650.0) match neither file.

## Consequence

The CAAG column must be treated as untraceable. Either it was produced by uncommitted code or output
that no longer exists, or it was not measured. It can't be cited in a resubmission. If a CAAG-on-
NarrativeQA number is wanted, re-run it with the new harness and commit the output under `runs/`
(working rule 1).
