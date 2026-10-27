# Pre-registration: confirmatory RAMDocs run

Draft of 2026-10-27, written **before** any confirmatory run. Items marked **TEAM** need the team's
decision before this is frozen. Once frozen, commit it, and treat anything not listed here as
exploratory in the paper.

## 1. Data and splits

- **RAMDocs** test file at commit `d44454c` (500 questions).
- **Pilot split:** [`splits/ramdocs_pilot.txt`](splits/ramdocs_pilot.txt), 200 ids. These shaped the
  design (flag formats, direction, document-level placement), so **results on them are exploratory**.
- **Confirmatory split:** [`splits/ramdocs_holdout.txt`](splits/ramdocs_holdout.txt), the other 300 ids.
  Under the default labels that is 139 conflict, 126 clean and 35 excluded questions.
  Nothing in our conditions has been run on these. MADAM-RAG was run on all 500, but only to
  validate the metric against its published score.
- **Labels:** `--ramdocs-labels exclude`. **TEAM:** confirm, or switch to the hand-checked labels
  once the 163 suspect documents are labelled (`runs/labelling/ramdocs_suspect_docs.csv`).
- **Second dataset:** see §7.

## 2. Fixed settings

| Setting | Value |
|---|---|
| Generator | `gpt-4o-mini`, temperature 0.1, max 256 output tokens, seeds 0, 1, 2 |
| Detector LLM (`llm_dir`) | `gpt-4o-mini`, temperature 0, seed 0 |
| CAAG | `all-MiniLM-L6-v2`, θ_s = 0.65, θ_c = 0.35 (original defaults) |
| NLI | `cross-encoder/nli-deberta-v3-base`, threshold 0.5, both directions |
| Flag text | `--flag-style uniform` |
| Primary metric | `ramdocs_strict`: every gold answer matched and no wrong answer matched, bidirectional whole-word matching. **Pending** the MADAM-RAG check (§6). |
| Secondary metrics | `gold_recall`, `wrong_rate`, `ramdocs_strict_oneway` |
| Price | see `llm.PRICES`, `llm.PRICES_CHECKED` |

## 3. Conditions

- **Prompt-only:** `plain`, `instruct`, `instruct_v2`.
- **Oracle:** `oracle+flag`, `oracle+flag_directional`, `oracle+remove_suspect`.
- **Real detectors:** `caag+flag`, `caag+remove`, `nli+flag`, `nli+remove`, `llm_dir+flag`, `llm_dir+flag_directional`, `llm_dir+remove_suspect`.
- **Placebo:** `random+flag` at CAAG's per-pair flag rate on the confirmatory split, taken from its detections before any generation.
- **Grid:** `grid` with precision {0.25, 0.5, 0.75, 1}, recall {0.25, 0.5, 1}, direction {0.5, 0.75, 1}, pool `all`, modes `flag_directional,remove_suspect`.
- **MADAM-RAG:** `madam` (temperature 0, 128 tokens, as in the public script).

**TEAM:** confirm which of these go in the main paper and which in the appendix. The analyses
below don't change.

## 4. Hypotheses (directional unless stated)

All on the confirmatory split; Δ is the paired difference in `ramdocs_strict`.

- **H1, detection.** Pair-level F1: `llm_dir` > `nli` > `caag`. `llm_dir` direction accuracy > 0.5. Cost and latency are reported for each; there is no test on them.
- **H2, neutral flags hurt.** On conflict questions, Δ(`oracle+flag`, `plain`) < 0.
- **H3, direction helps.** On conflict questions:
  - Δ(`oracle+flag_directional`, `plain`) > 0;
  - Δ(`oracle+flag_directional`, `oracle+flag`) > 0;
  - Δ(`oracle+remove_suspect`, `plain`) > 0.
- **H4, false flags hurt clean questions.**
  - On clean questions, grid cells with false flags score below `plain`.
  - The harm grows with the share of clean questions that have a correct document marked: Spearman ρ < 0 across cells.
  - At matched precision and recall, the harm is larger under `remove_suspect` than under `flag_directional`.
- **H5, content vs. template.** Δ(`caag+flag`, `random+flag`) on all questions. Two-sided; we expect ≈ 0 given CAAG's precision.
- **H6, document-level placement predicts real detectors.**
  - For `caag`, `nli` and `llm_dir`, in each mode and subset, the actual Δ lies within the 95% CI around the grid prediction from `place_detector --space doc`.
  - Its absolute error is smaller than with `--space pair`.
  - These axes were chosen after seeing `llm_dir` on the pilot split. The pilot CAAG and NLI runs are a first out-of-sample check, but the confirmatory test is on the hold-out.
- **H7, prompt-only baselines.** Δ(`instruct`, `plain`) and Δ(`instruct_v2`, `plain`), on conflict and clean questions. Two-sided, with no direction predicted.

## 5. Analysis

- Average each question's score over its 3 seeds first. Then compute the paired cluster-bootstrap 95% CI (10,000 resamples) and the sign-flip permutation p-value (10,000 permutations).
- **Multiplicity:** Holm within each hypothesis family (H2–H7 separately, across their listed comparisons and subsets). For the grid, Holm runs within each table (`grid_analyze`).
- **Break-even precision:** the lowest precision at and above which every cell is a significant gain (`grid_analyze`). `harm_free_precision` is secondary only.
- **Splits:** always report all, conflict and clean questions.
- **Exclusions:** none after the run. Truncated responses (256 tokens) and missing `Final answer(s)` lines are scored as they are, and their counts are reported.
- **Failures:** if an API error kills a call, rerun only that call (the runner resumes). There are no other reruns.

## 6. Checks to finish before freezing

1. **Metric (done; needs a TEAM decision).** Wang et al. count an output correct "if and only if it includes *all* gold answers" and no answers supported only by misinformation. That's our definition, but they publish no scorer, normalization or question subset. MADAM-RAG on GPT-4o-mini, `runs/ramdocs_madam_check`, seed 0:

   | | bidirectional (`ramdocs_strict`) | one-way (`ramdocs_strict_oneway`) |
   |---|---|---|
   | all 500 | 38.4% | 32.8% |
   | pilot 200 | 36.0% | 28.5% |
   | published | 28.00 | |

   We do **not** reproduce 28.00 under our primary metric. Bidirectional matching is more lenient: it credits partial answers such as "1884" for "1884–1886", as well as real paraphrases. Every published GPT-4o-mini score is a multiple of 0.5, which suggests a 200-question subset. The model snapshot also differs.

   **TEAM:** choose one of:
   - (a) keep bidirectional as primary, and use one-way whenever we put a number next to Wang et al.'s;
   - (b) make one-way primary;
   - (c) ask the authors for their scorer and question ids.

   Either way, report both.
2. **Price (done).** `llm.PRICES_CHECKED`: 2026-10-27, $0.15 / $0.075 cached / $0.60 per 1M tokens. Cached prompt tokens are now priced at the cached rate.
3. **TEAM:** the label rule (§1), the main-paper conditions (§3), the second dataset and model (§7).

## 7. Second model, dataset and judge (TEAM)

- **Second generator:** one open-weights model through an OpenAI-compatible endpoint (`--base-url`, `--api-key-env`). The same pre-registered analysis applies; the model's name and price go here before its run.
- **Second dataset: WikiContradict** (`--dataset wikicontradict`, HF revision pinned in `data.py`). It has 253 questions, each with two Wikipedia passages that contradict each other; the reference answer is *both* sides. There's no misinformation side and no clean questions, so it can't test direction or false-flag harm. It is the contrast case: on RAMDocs, listing both sides is wrong, while here it is the correct answer. Proposed hypothesis **H8:** Δ(`oracle+flag`, `plain`) > 0 on WikiContradict (primary `both_answers`), the opposite sign from H2.
- **CONFLICTS judge:** a model from a different family, run only on a stratified subset, with about 100 human labels to check agreement.
