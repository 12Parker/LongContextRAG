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
| `holdout_round2` | Round 2 (§10): neutral/directional flag + v2 instruction, Astute RAG, TrustRAG (+multi), gpt-4o-mini, 3 seeds | `exclude` | **registered (§10, H13, H14)** |
| `holdout_gpt41_round2` | Round 2: same with gpt-4.1, seed 0 | `exclude` | **registered (§10)** |
| `holdout_qwendet` | Round 2: Qwen2.5-7B as LLM detector, gpt-4o-mini generator, 3 seeds | `exclude` | exploratory (registered) |
| `cb_main` | Round 2: ConflictBank (400 items, `prepare_conflictbank.py`), gpt-4o-mini, 3 seeds, all formats and baselines | n/a | **registered (§10, H12)** |
| `cb_gpt41` | Round 2: ConflictBank, gpt-4.1, seed 0 | n/a | **registered (H12)** |
| `cb_qwen` | Round 2: ConflictBank, Qwen2.5-7B (Ollama), seed 0 | n/a | **registered (H12)** |
| `cb_single` | Round 3 (§11, H15): ConflictBank, single-answer skeleton, gpt-4o-mini, 3 seeds (incl. closed-book, one-wrong, self-identify) | n/a | **registered (§11)** |
| `cb_minimal` | Round 3: ConflictBank, minimal skeleton, gpt-4o-mini, 3 seeds | n/a | exploratory (registered) |
| `cb_single_gpt41` | Round 3 (H15, H16): ConflictBank single, gpt-4.1, seed 0 | n/a | **registered (§11)** |
| `cb_single_qwen` | Round 3 (H15): ConflictBank single, Qwen2.5-7B (Ollama), seed 0 | n/a | **registered (§11)** |
| `cbmaj_single` | Round 3 (H17, H18): ConflictBank majority variant (3 wrong : 1 correct), single, gpt-4o-mini, 3 seeds, incl. Astute RAG / TrustRAG | n/a | **registered (§11)** |
| `holdout_extra` | Round 3 (H16, H18): RAMDocs closed-book, self-identify, one-wrong flag, gpt-4o-mini, 3 seeds | `exclude` | **registered (§11)** |
| `cb_single_luna`, `holdout_luna`, `cbmaj_single_luna` | Round 4 (§12, H19): gpt-6-luna, 3 seeds, default temperature | as above | **registered (§12)** |
| `cb_single_terra`, `holdout_terra`, `cbmaj_single_terra` | Round 4 (H19): gpt-6.1-terra, seed 0, default temperature | as above | **registered (§12)** |
| `cb_forced_{mini,luna,terra,haiku}` | Round 7 (§13 H20, H21; §14): ConflictBank, `forced` skeleton (no "unknown"); gpt-4o-mini, gpt-6-luna and Claude Haiku 3 seeds, gpt-6.1-terra seed 0 | n/a | **registered (§13, §14)** |
| `ctl_{rd,cb,mj}_{mini,luna,terra}` | Round 7 (H22, H23): answer key, `flag_pick`, no-pairs flags on RAMDocs, ConflictBank single and CB majority | as base set | **registered (§13)** |
| `cross_cb_<gen>_<det>`, `cross_mj_<gen>_<det>` | Round 7 (H24): LLM detector run with gpt-4.1, gpt-6-luna, gpt-6.1-terra, Claude Haiku/Sonnet for each generator; `plain` and `llm_dir+flag_directional`; CB single (registered) and CB majority (exploratory) | n/a | **registered (§13, §14)** |
| `cb_single_{haiku,sonnet}`, `holdout_{haiku,sonnet}`, `cbmaj_single_haiku` | Round 7 (§14, H25): Claude Haiku 5.5 (3 seeds) and Sonnet 5.5 (seed 0) | as above | **registered (§14)** |
| `rg_{mini,luna,terra,haiku,sonnet}` | Round 7 (§14, H26): RAGuard (294 conflict + 100 clean claims), single skeleton | n/a | **registered (§14)** |
| `conflicts_auto` | CONFLICTS, now 3 seeds (§13.4, exploratory) | n/a | exploratory |
| `judge_matcher.json` | LLM-judge (gpt-6.1-terra) check of the string matcher on 162 stratified conflict outputs (§13.4) | n/a | exploratory |
| `hint_cb_<gen>_<det>` | Round 8 (§15, H27): `plain` and `hint` (the detector model's own plain seed-0 answer, no passage marked) for every H24 cell; seeds as the cell | n/a | **registered (§15)** |
| `probe_{rev,para}_{mini,lunadet}` | Round 8 (§15, H29): ConflictBank with passage order reversed / every passage paraphrased by gpt-4o-mini (`prepare_cb_probe.py`; 11 of 1,600 paraphrases fail the answer-string check); gpt-4o-mini generator | n/a | **registered (§15)** |
| `wiki_llm_{mini,luna,haiku}`, `holdout_rem_{luna,haiku,terra}` | Round 8 (§15, H28): LLM directional flag and LLM removal on WikiContradict and on RAMDocs (analysed on ambiguity-only questions) | n/a | **registered (§15)** |
| `randdir_cb_{mini,luna,haiku,terra}` | Round 9 (§16, H31): ConflictBank single; gold pair flagged on every conflict question, suspect = misinformation on a random half (`synth-p1-r1-all-d0.5+flag_directional`) | n/a | **registered (§16)** |
| `vkey_cb_<gen>_<det>` | Round 9 (§16, H30): `llm_dir+verdict_key` (the detector's suspect passages turned into "the answer X is false", no passage marked) for the H24 cells with gpt-4o-mini, gpt-6-luna, Qwen2.5-7B and Claude Haiku generators | n/a | **registered (§16)** |
| `pr_pilot_{mini,mini_lunadet,haiku}` | Pilot for a planned PoisonedRAG registration: 40 pilot questions (20 attacked, 20 clean; `splits/poisonedrag_pilot.txt`, excluded from any evaluation), single prompt; gpt-4o-mini (3 seeds) with the gpt-4o-mini or gpt-6-luna detector, Claude Haiku (seed 0) | n/a | pilot (not reported as results) |
| `hoh_smoke_{mini,mini_dated,haiku}` | Smoke test for a planned HoH-QA registration: 40 items (20 stale-vs-current conflicts, 20 clean; `splits/hoh_smoke.txt`), real Wikipedia revision paragraphs, single prompt; gpt-4o-mini (3 seeds, undated and dated passages), Claude Haiku (seed 0); passage-level `llm_dir` flags and the answer-level `llm_ans` verdict | n/a | smoke (not reported as results) |
| `cb_single_gpt41` (added `self_identify`) | Round 8 (§15): single-call mode for gpt-4.1, seed 0 | n/a | registered (§15) |
| `labelling` | `ramdocs_suspect_docs.csv`: 163 RAMDocs documents to hand-label (`export_labelling.py`); `audit_200.csv`: 200 outputs for a human audit of the matcher (`export_audit.py`, §15.4.7), not yet labelled | n/a | to label |

Runs made before the cached-latency fix (all runs listed above) record `generate_s` as wall-clock time, ~0 for
cache hits; their latency figures should not be reported.

`ramdocs_dirgrid_pilot/grid_vs_plain.txt` and `grid_analysis.json` were regenerated with the
Holm-based break-even definition; `ramdocs_llmdet_pilot/placement.*.{doc,pair}.json` with
document-level placement.

All runs were rescored (`rescore.py`) after the MADAM answer-parsing fix of 2026-10-27; only
`ramdocs_madam_check` changed (38.4%, up from 35.8%).

Runs from `holdout_*` onward record `generate_s` as API latency (also on cache hits). The paper's
numbers, tables and figures are all produced by `python -m experiments.conflict.paper_stats` from
these directories (written to `paper/generated/`). `cb_main` stopped once on a transient HTTP 403 and was
resumed (completed rows kept; finished calls served from the cache). Total API spend is in `COSTS.md`.
