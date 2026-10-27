# Run outputs

Every number reported from the conflict experiments must trace to a file here (CONTEXT.md, rule 1).
The LLM response cache (`llm_cache.sqlite`) and progress logs are not committed.
All runs use GPT-4o-mini on RAMDocs; pilots are **exploratory**, since they shaped the design.

| Run | What | Labels | Status |
|---|---|---|---|
| `smoke_ramdocs` | 6 questions, plumbing check | released (`none`) | superseded |
| `ramdocs_grid_pilot` | 200 q, 3 seeds; neutral-flag precision x recall grid | released (`none`) | **stale**: pre-label-fix; flag format before unification |
| `ramdocs_format_pilot` | 200 q, 3 seeds; oracle flags in 5 formats | `exclude` | exploratory |
| `ramdocs_dirgrid_pilot` | 200 q, 3 seeds; directional grid (P x R x direction), flag_directional and remove_suspect | `exclude` | exploratory |
| `ramdocs_llmdet_pilot` | 200 q, 3 seeds; directional LLM detector in 3 formats | `exclude` | exploratory |
| `ramdocs_caagnli_pilot` | 200 q, 3 seeds; CAAG and NLI with flag_directional and remove_suspect, plus instruct_v2 | `exclude` | exploratory |
| `ramdocs_madam_check` | all 500 q, seed 0; MADAM-RAG, to compare our metric with the published 28.00 | `exclude` | metric check only |
| `phase0` | Phase 0: CAAG on 100 NarrativeQA questions (50 stories), BM25 top-10; `flags_to_label.csv` for hand labels | n/a | to label |
| `labelling` | `ramdocs_suspect_docs.csv`: 163 RAMDocs documents to hand-label (`export_labelling.py`) | n/a | to label |

Runs made before the cached-latency fix (all runs listed above) record `generate_s` as wall-clock time, ~0 for
cache hits; their latency figures should not be reported.

`ramdocs_dirgrid_pilot/grid_vs_plain.txt` and `grid_analysis.json` were regenerated with the
Holm-based break-even definition; `ramdocs_llmdet_pilot/placement.*.{doc,pair}.json` with
document-level placement.

All runs were rescored (`rescore.py`) after the MADAM answer-parsing fix of 2026-10-27; only
`ramdocs_madam_check` changed (38.4%, up from 35.8%).
