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
| `holdout_detect` | Confirmatory: detector-only evaluation on the 300 hold-out questions (H1), θ sweep, signal ablation | `exclude` | **confirmatory** |
| `holdout_main` | Confirmatory: all pre-registered conditions + grid, gpt-4o-mini, 3 seeds (`results.jsonl.gz`) | `exclude` | **confirmatory** |
| `holdout_caag_original` | Confirmatory: CAAG in its original typed/scored flag format | `exclude` | **confirmatory** |
| `holdout_gpt41` | Confirmatory: gpt-4.1 replication (H9) | `exclude` | **confirmatory** |
| `holdout_contains` | Confirmatory robustness: oracle + grid with `contains` labels (`results.jsonl.gz`) | `contains` | **confirmatory** |
| `wikicontradict_main`, `wikicontradict_gpt41` | WikiContradict (H8), all 253 questions | n/a | **confirmatory** |
| `conflicts_auto` | CONFLICTS, 237 auto-scorable questions, 1 seed | n/a | exploratory |
| `holdout_minimal` | Revision H10: minimal skeleton, gpt-4o-mini, 3 seeds; oracle and LLM-detector conditions | `exclude` | **registered (§9.1)** |
| `holdout_gpt41_minimal` | Revision H10: minimal skeleton, gpt-4.1, 3 seeds; oracle conditions | `exclude` | **registered (§9.1)** |
| `holdout_qwen` | Revision H11: Qwen2.5-7B-Instruct (Ollama, 4-bit), default skeleton, seed 0 | `exclude` | **registered (§9.2, §9.4)** |
| `holdout_qwen_minimal` | Revision H10: Qwen2.5-7B-Instruct, minimal skeleton, seed 0 | `exclude` | **registered (§9.2, §9.4)** |
| `holdout_adversarial` | Revision §9.5: inverted oracle (names the correct side), gpt-4o-mini, 3 seeds | `exclude` | exploratory (registered) |
| `labelling` | `ramdocs_suspect_docs.csv`: 163 RAMDocs documents to hand-label (`export_labelling.py`) | n/a | to label |

Runs made before the cached-latency fix (all runs listed above) record `generate_s` as wall-clock time, ~0 for
cache hits; their latency figures should not be reported.

`ramdocs_dirgrid_pilot/grid_vs_plain.txt` and `grid_analysis.json` were regenerated with the
Holm-based break-even definition; `ramdocs_llmdet_pilot/placement.*.{doc,pair}.json` with
document-level placement.

All runs were rescored (`rescore.py`) after the MADAM answer-parsing fix of 2026-10-27; only
`ramdocs_madam_check` changed (38.4%, up from 35.8%).

Runs from `holdout_*` onward record `generate_s` as API latency (also on cache hits). The paper's
numbers, tables and figures are all produced by `python -m experiments.conflict.paper_stats` from
these directories (written to `paper/generated/`). Total API spend is in `COSTS.md`.
