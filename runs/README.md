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

Runs made before the cached-latency fix (all runs listed above) record `generate_s` as wall-clock time, ~0 for
cache hits; their latency figures should not be reported.

`ramdocs_dirgrid_pilot/grid_vs_plain.txt` and `grid_analysis.json` were regenerated with the
Holm-based break-even definition; `ramdocs_llmdet_pilot/placement.*.{doc,pair}.json` with
document-level placement.
