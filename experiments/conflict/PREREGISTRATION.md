# Pre-registration: confirmatory RAMDocs run

**Frozen 2026-10-27, before any confirmatory run.** The TEAM items below were resolved by the
decisions in §8, made on the evidence available then; §8 overrides any earlier text it conflicts
with. Anything not listed here is exploratory in the paper.

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

## 8. Decisions (frozen 2026-10-27)

These were made without the team, on the evidence below. A change after this date counts as a
deviation and gets reported as one.

1. **Primary metric: `ramdocs_strict_oneway`.** Wang et al. define a correct output as one that
   "includes all gold answers" and no misinformation answer. One-way matching reads that literally:
   each gold string must appear inside the output. On the pilot split, one-way matching scores
   MADAM-RAG at 28.5%, against the published 28.00; bidirectional scores it at 36.0%.
   `ramdocs_strict` (bidirectional) is the pre-registered **robustness** metric. A hypothesis
   counts as *robustly supported* only if its Holm-adjusted test passes under both metrics.
2. **Labels: `exclude`** for the main analysis. It makes no judgement calls on the 163 suspect
   documents. `contains` is a robustness check for H2–H4. Hand labels aren't used, because no human
   labelling has been done.
3. **Conditions:** every condition in §3 runs on the confirmatory split with `gpt-4o-mini`. The main
   paper reports the prompt-only, oracle, real-detector and placebo conditions and the grid; MADAM-RAG
   is a reference point. The original-CAAG-format reproduction (`caag+flag` with
   `--flag-style original`) goes in the appendix.
4. **Second generator: `gpt-4.1`**, the strongest OpenAI non-reasoning model that supports `temperature`
   and `seed` ($2.00 / $8.00 per 1M tokens, checked 2026-10-27). No open-weights endpoint was available;
   that is a stated limitation. It runs on the confirmatory split, 3 seeds, with:
   - `plain`, `instruct_v2`;
   - `oracle+flag`, `oracle+flag_directional`, `oracle+remove_suspect`;
   - `llm_dir+flag_directional`, `llm_dir+remove_suspect`;
   - `nli+remove_suspect`.

   The detector stays `gpt-4o-mini` (`--detector-model`). **H9 (replication):** H2 and H3 hold with
   the same sign under `gpt-4.1` (Holm within H9).
5. **WikiContradict (H8):** all 253 questions, 3 seeds:
   - `plain`, `instruct`, `instruct_v2`;
   - `oracle+flag`;
   - `caag+flag`, `nli+flag`, `llm_dir+flag`.

   Primary metric `both_answers`. H8 is two-sided, but the predicted sign is positive for `oracle+flag`.
   It runs with `gpt-4o-mini`, plus `gpt-4.1` for `plain` and `oracle+flag`.
6. **CONFLICTS (exploratory, secondary):** auto-scorable items only (`primary` not None), 1 seed:
   - `plain`, `instruct_v2`;
   - `caag+flag`, `nli+flag`, `llm_dir+flag`, `llm_dir+flag_directional`.

   Plus question-level detection metrics for CAAG, NLI and `llm_dir`. No judge: none from a different
   model family was available. CONFLICTS' opinion and complementary items are left out, and that is
   a stated limitation.
7. **Budget:** at most $100 in API spend across everything, tracked in `runs/COSTS.md`.

## 9. Revision addendum (frozen 2026-10-27, after a mock review, before any run below)

Three reviewers read the draft. This section registers the runs and re-analyses that answer them.
It is committed before any of the runs in 9.1–9.2 start. §1–§8 are unchanged. The re-analyses in 9.3
use data that already exists, so the paper reports them as **re-analyses**, not as pre-registered
tests of new data.

### 9.1 Skeleton ablation (H10)

The default skeleton (`prompts.BASE`) tells the model to "give every valid answer" when the passages
support more than one. The reviewers' concern is that a neutral flag may hurt only because of that
policy combined with the strict metric. `--skeleton minimal` (`prompts.BASE_MINIMAL`) removes the
policy sentence, and its answer line allows several answers without inviting them. Everything else
is unchanged: hold-out split, `exclude` labels, both metrics, analysis as in §5.

- **gpt-4o-mini**, 3 seeds, `--skeleton minimal`:
  - `plain`;
  - `oracle+flag`, `oracle+flag_directional`, `oracle+remove_suspect`;
  - `llm_dir+flag`, `llm_dir+flag_directional`, `llm_dir+remove_suspect`.

  Output: `runs/holdout_minimal`.
- **gpt-4.1**, 3 seeds, `--skeleton minimal`, detector `gpt-4o-mini`:
  - `plain`;
  - `oracle+flag`, `oracle+flag_directional`, `oracle+remove_suspect`.

  Output: `runs/holdout_gpt41_minimal`.
- **H10a.** Under the minimal skeleton, Δ(`oracle+flag`, `plain`) on conflict questions, for each
  generator. Two-sided, because we do not predict a sign. A negative, significant effect means the
  harm does not depend on the "every valid answer" policy. A null or positive effect means it does,
  and the paper says so.
- **H10b.** The three H3 comparisons hold, with the same sign, under the minimal skeleton, for each
  generator.
- **Multiplicity:** Holm within H10, across generators and comparisons, per metric.

### 9.2 Open-weights generator (H11)

This fulfils §7. No hosted endpoint is available, so the model runs locally.

- **Model:** `qwen2.5:7b-instruct` (Qwen2.5-7B-Instruct, Q4_K_M quantization) through Ollama's
  OpenAI-compatible endpoint. Ollama model digest `845dbda0ea48`; 4 parallel slots; context length
  8192. Temperature 0.1, 256 output tokens, seeds 0, 1, 2. There is no API cost. Batched local
  inference is not bit-for-bit deterministic, so this is stated as a limitation.
- **Detector:** `llm_dir` stays `gpt-4o-mini` (`--detector-model`). Its outputs are the same as in
  `holdout_main`.
- **Default skeleton** (`runs/holdout_qwen`):
  - `plain`;
  - `oracle+flag`, `oracle+flag_directional`, `oracle+remove_suspect`;
  - `llm_dir+flag`, `llm_dir+flag_directional`, `llm_dir+remove_suspect`.
- **Minimal skeleton** (`runs/holdout_qwen_minimal`):
  - `plain`;
  - `oracle+flag`, `oracle+flag_directional`, `oracle+remove_suspect`.

  These enter H10 as a third generator.
- **H11 (replication).** H2 and H3 hold with the same sign under Qwen2.5-7B, with the default
  skeleton. Holm within H11.
- **Real detector (exploratory, all generators):** Δ(`llm_dir+flag_directional`, `llm_dir+flag`) on
  conflict questions. This is the direct test of whether direction helps with a real detector.

### 9.3 Re-analyses of existing data (reported as such)

1. **H4a, by mode.** Count false-flag cells separately for `flag_directional` and `remove_suspect`.
   On clean questions, direction accuracy cannot change the prompt, so cells that differ only in
   direction are one comparison. We collapse them, giving 18 distinct comparisons. Holm runs within
   each mode. H4c is reported as a count, with no "supported" label.
2. **Dose-response.** Spearman ρ over the distinct clean-question cells (the collapse above), with a
   95% CI from a cluster bootstrap over questions. The cell-permutation p-value is dropped.
3. **H6 as pre-registered.** For each placement space, a 95% interval for the grid prediction,
   built by a cluster bootstrap over questions: resample questions, recompute the cell effects and
   the detector's coordinates, then interpolate. We count how often the actual effect falls inside
   that interval. Doc- and pair-level absolute errors are compared with an exact paired sign-flip
   test over the cases. The lenient converse test is kept, labelled as the converse.
4. **H5 equivalence.** TOST with a margin of ±3 points (half the H2 effect, fixed now), α = 0.05.
5. **"All" over labelled questions.** Every "All" effect is also reported over the 265 labelled
   questions (conflict + clean), with the 35 excluded questions reported separately.
6. **H1 direction accuracy.** A cluster bootstrap CI over questions replaces the binomial CI.
7. **Real-detector rows.** In every table, these are labelled exploratory. Significance is Holm
   across that table.

### 9.4 Change before any Qwen result was looked at (2026-10-27)

Running Qwen2.5-7B locally takes about 3 s per call. At 3 seeds the two Qwen runs would take about
9 hours, so both now use **seed 0 only**. With temperature 0.1, seeds change little: under `plain`,
only 4.7% of gpt-4o-mini's hold-out questions score differently across seeds.

- 113 `plain` rows had already been generated, all at seed 0. They are kept.
- No Qwen score had been inspected when this change was made.
- All other §9.2 settings are unchanged.
- The paper reports this as a deviation.
