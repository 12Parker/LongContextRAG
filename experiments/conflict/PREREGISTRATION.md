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

### 9.5 Adversarial direction (exploratory; registered 2026-10-27, before its run)

A reviewer asked what happens if the directional signal is wrong. `oracle_inv` flags exactly the gold
contradiction pairs but names the *correct* side as the suspect. We run it with **gpt-4o-mini**, the
default skeleton, 3 seeds, on the hold-out split, in two modes: `oracle_inv+flag_directional` and
`oracle_inv+remove_suspect`. Output goes to `runs/holdout_adversarial`, together with a `plain`
reference (served from the cache).

We expect both to lower accuracy on conflict questions, removal more than the flag. This is
exploratory and reported with a raw p-value. It covers only a detector with the wrong direction; an
attacker who also chooses which pairs to flag is not tested.

### 9.6 Degenerate Qwen generations (2026-10-27, during the Qwen run)

Ollama sometimes aborts a generation that keeps repeating one token, with the error "token repeat
limit reached". The call fails the same way on every retry, which stopped the run. This is a
degenerate output, like a truncated one (§5), so it is scored as it is: the stored response is a
marker with no `Final answer(s)` line. We report how many there are. This rule was set when the
first one occurred, without looking at any scores.

## 10. Second revision addendum (frozen 2026-10-27, after a second mock review, before any run below)

A second, independent mock review asked for five things: stronger published baselines, a neutral
flag paired with a resolution instruction, a second misinformation benchmark, a detector from a
different model family than the generator, and a single significance scheme. Nothing below has
been run yet.

### 10.1 Conditions and runs

**New prompt mode.** `flag_directional_instruct_v2` is the directional flag plus the v2 reliability
instruction. The existing `flag_instruct_v2` is the neutral flag plus the same instruction.

**Baselines** (`baselines.py`):
- **Astute RAG** (ACL 2025). Paper prompts, t=1, m=1.
- **TrustRAG** (repo commit `11dcea0`, MIT). K-means plus n-gram filter, then 3-call conflict resolution.

Both produce one answer. `*_multi` variants ask for every valid answer, because RAMDocs has
multi-answer questions. Both the plain and the `_multi` variants are reported.

**ConflictBank** (`prepare_conflictbank.py`; `Warrieryes/CB_qa_v2` at `92866a0`):
- 400 items, sampled round-robin over relations with seed 0, ids in `splits/conflictbank_eval.txt`.
- 300 conflict items: correct passage, misinformation passage and 2 distractors.
- 100 clean items: correct passage and 3 distractors.
- Each item has a single gold answer, scored with the same strict metrics as RAMDocs.
- No pilot: every ConflictBank result is confirmatory for the tests below.

**Runs:**

| Output | Data | Generator | Seeds | Conditions |
|---|---|---|---|---|
| `runs/holdout_round2` | RAMDocs hold-out | gpt-4o-mini | 0, 1, 2 | `plain`, `oracle+flag_instruct_v2`, `oracle+flag_directional_instruct_v2`, `llm_dir+flag_instruct_v2`, `llm_dir+flag_directional_instruct_v2`, `astute`, `astute_multi`, `trustrag`, `trustrag_multi` |
| `runs/holdout_gpt41_round2` | RAMDocs hold-out | gpt-4.1 | 0 | `plain`, `oracle+flag_instruct_v2`, `oracle+flag_directional_instruct_v2`, `astute_multi`, `trustrag_multi`, `llm_dir+flag_directional` |
| `runs/holdout_qwendet` | RAMDocs hold-out | gpt-4o-mini, with `llm_dir` run on Qwen2.5-7B (`--detector-model qwen2.5:7b-instruct --detector-base-url`) | 0, 1, 2 | `plain`, `llm_dir+flag`, `llm_dir+flag_directional`, `llm_dir+remove_suspect` |
| `runs/cb_main` | ConflictBank | gpt-4o-mini | 0, 1, 2 | `plain`, `instruct_v2`, `oracle+flag`, `oracle+flag_directional`, `oracle+remove_suspect`, `oracle+flag_instruct_v2`, `oracle+flag_directional_instruct_v2`, `oracle_inv+flag_directional`, `nli+flag`, `nli+remove`, `llm_dir+flag`, `llm_dir+flag_directional`, `llm_dir+remove_suspect`, `astute`, `trustrag` |
| `runs/cb_gpt41` | ConflictBank | gpt-4.1 | 0 | `plain`, `oracle+flag`, `oracle+flag_directional`, `oracle+remove_suspect`, `llm_dir+flag`, `llm_dir+flag_directional`, `llm_dir+remove_suspect` |
| `runs/cb_qwen` | ConflictBank | Qwen2.5-7B (local) | 0 | same as `cb_gpt41` |

Wherever the generator is not gpt-4o-mini, the LLM detector is gpt-4o-mini; the exception is
`holdout_qwendet`, where Qwen2.5-7B is the detector.

### 10.2 Hypotheses

Primary metric: one-way strict accuracy. Holm runs within each family, per metric.

- **H12, ConflictBank replication**, for each of the 3 generators, on conflict items:
  - Δ(`oracle+flag`, `plain`) is two-sided. In the RAMDocs exploratory split, the neutral-flag harm
    was small on single-answer questions, so we do not predict its sign.
  - The three H3 comparisons are directional (+).
- **H13, resolution instruction**, on conflict questions:
  - Δ(`oracle+flag_instruct_v2`, `plain`) is two-sided.
  - Δ(`oracle+flag_directional_instruct_v2`, `oracle+flag_instruct_v2`) > 0: direction adds something
    beyond the instruction.
  - Settings: RAMDocs with gpt-4o-mini, RAMDocs with gpt-4.1, and ConflictBank with gpt-4o-mini.
- **H14, against published baselines**, on labelled questions (conflict plus clean), all two-sided:
  - Δ(`llm_dir+flag_directional`, `astute_multi`);
  - Δ(`llm_dir+flag_directional`, `trustrag_multi`);
  - on ConflictBank, Δ(`llm_dir+flag_directional`, `astute`) and Δ(`llm_dir+flag_directional`, `trustrag`).

  Settings are gpt-4o-mini on both datasets, and gpt-4.1 on RAMDocs.

### 10.3 Exploratory analyses and the significance scheme

**Exploratory analyses:**
- the detector from a different model family (`holdout_qwendet`);
- the inverted oracle on ConflictBank;
- every comparison against `plain` that is not listed above.

**One significance scheme for the whole paper.** A comparison that is a registered test (H2–H14)
takes that test's Holm-adjusted p within its family. Every other comparison the paper reports
belongs to one exploratory family, Holm-corrected together, per metric. A table cell is bold if
and only if it is significant under this scheme. Text that calls a result "significant" uses the
same scheme.

**Analyses specified after the data were seen.** The ±3-point TOST margin (§9.3) was fixed after H2
was known. The single- vs multi-answer split (§9) was chosen after the minimal-skeleton run. The
paper labels both as post hoc.

## 11. Third revision addendum (frozen 2026-10-27, after a third mock review, before any run below)

A third mock review (three reviewers plus a discussion round, all 2.5) found one decisive problem.
ConflictBank was run only under the default skeleton, which asks for "every valid answer", although
every ConflictBank question has exactly one answer. That could explain the neutral-flag harm, the
low no-flag accuracy (21.6% on conflict questions against 89.0% on clean ones), and the large
removal gains.

The reviewers also asked for:
- a closed-book control;
- a neutral flag that says one side is false without saying which;
- a compute-matched single-call baseline;
- a condition where the misinformation is the majority;
- seed-matched gpt-4.1 numbers for H14.

Nothing below has been run.

### 11.1 Conditions and runs

**New skeleton.** `single` (`prompts.BASE_SINGLE`) says the question has exactly one correct
answer and asks for one.

**New modes.**
- `flag_onewrong`: the neutral flag, plus "exactly one of the two contains misinformation (it is
  not known which)" in the header and conflict list.
- `self_identify`: no detector. One call that first writes `Suspect passages: …` and then answers
  from the remaining passages.
- `closedbook`: no passages, with the skeleton's answer line.

**Majority variant.** `conflictbank_majority` (`prepare_conflictbank.py`) uses the same 400 items.
Each conflict item has the correct passage plus all three of ConflictBank's wrong-object passages
(misinformation, temporal and semantic evidence) and no distractors, so the split is 3 wrong to
1 correct. Clean items are unchanged. Pairs of two wrong-side passages are labelled `consistent`.

**Runs** (the LLM detector is gpt-4o-mini throughout):

| Output | Data | Generator | Seeds | Skeleton | Conditions |
|---|---|---|---|---|---|
| `runs/cb_single` | ConflictBank | gpt-4o-mini | 0, 1, 2 | single | `plain`, `closedbook`, `self_identify`, `oracle+flag`, `oracle+flag_onewrong`, `oracle+flag_directional`, `oracle+remove_suspect`, `llm_dir+flag`, `llm_dir+flag_directional`, `llm_dir+remove_suspect` |
| `runs/cb_minimal` | ConflictBank | gpt-4o-mini | 0, 1, 2 | minimal | `plain`, `oracle+flag`, `oracle+flag_directional`, `oracle+remove_suspect` |
| `runs/cb_single_gpt41` | ConflictBank | gpt-4.1 | 0 | single | `plain`, `closedbook`, `oracle+flag`, `oracle+flag_onewrong`, `oracle+flag_directional`, `oracle+remove_suspect`, `llm_dir+flag_directional` |
| `runs/cb_single_qwen` | ConflictBank | Qwen2.5-7B (local) | 0 | single | as `cb_single_gpt41`, plus `llm_dir+flag` |
| `runs/cbmaj_single` | ConflictBank majority | gpt-4o-mini | 0, 1, 2 | single | `plain`, `self_identify`, `oracle+flag`, `oracle+flag_directional`, `oracle+remove_suspect`, `llm_dir+flag`, `llm_dir+flag_directional`, `llm_dir+remove_suspect`, `astute`, `trustrag` |
| `runs/cbmaj_single_gpt41` | ConflictBank majority | gpt-4.1 | 0 | single | `plain`, `oracle+flag`, `oracle+flag_directional`, `llm_dir+flag_directional` |
| `runs/holdout_extra` | RAMDocs hold-out | gpt-4o-mini | 0, 1, 2 | default | `closedbook`, `self_identify`, `oracle+flag_onewrong` |
| `runs/holdout_gpt41_round2` | RAMDocs hold-out | gpt-4.1 | adds 1, 2 | default | `plain`, `llm_dir+flag_directional`, `astute_multi`, `trustrag_multi` |
| `runs/holdout_gpt41_closedbook` | RAMDocs hold-out | gpt-4.1 | 0 | default | `closedbook` |

**Budget rule.** Runs go in the order listed. If total spend reaches $95 before a gpt-4.1 run
starts, that run is dropped and the paper says so.

### 11.2 Hypotheses

Primary metric: one-way strict accuracy. Holm runs within each family, per metric.

- **H15: the ConflictBank result without the multi-answer policy.** Skeleton `single`. Run for
  each of gpt-4o-mini, gpt-4.1 and Qwen2.5-7B.
  - On conflict questions:
    - Δ(`oracle+flag`, `plain`), two-sided.
    - Δ(`oracle+flag_directional`, `oracle+flag`) > 0.
    - Δ(`oracle+flag_directional`, `plain`) > 0.
  - On all 400 questions:
    - Δ(`llm_dir+flag_directional`, `plain`) > 0. This makes the real detector's directional
      flag a confirmatory test.

  If the neutral-flag harm is absent here, the paper will say that on ConflictBank it depends
  on the answer policy.
- **H16: information versus direction.** On conflict questions:
  - Δ(`oracle+flag_directional`, `oracle+flag_onewrong`) > 0;
  - Δ(`oracle+flag_onewrong`, `plain`), two-sided.

  Settings: ConflictBank `single` with gpt-4o-mini, ConflictBank `single` with gpt-4.1, and
  RAMDocs default with gpt-4o-mini.
- **H17: majority misinformation.** On `conflictbank_majority` with skeleton `single`, using
  gpt-4o-mini (3 seeds) and gpt-4.1 (seed 0), on conflict questions:
  - Δ(`oracle+flag`, `plain`), two-sided;
  - Δ(`oracle+flag_directional`, `plain`) > 0;
  - Δ(`llm_dir+flag_directional`, `plain`), two-sided, because a majority may mislead the
    detector.
- **H18: compute-matched single call.** Two-sided, on all labelled questions:
  - Δ(`llm_dir+flag_directional`, `self_identify`).

  Settings: RAMDocs gpt-4o-mini, ConflictBank `single` gpt-4o-mini, and ConflictBank majority
  gpt-4o-mini. Detector-plus-generator counts as two calls everywhere in the paper.

### 11.3 Exploratory analyses and changes to reporting

**Closed-book stratification (exploratory, registered).** A question counts as "known" if
gpt-4o-mini's closed-book answer is strictly correct on at least 2 of 3 seeds. We will report:
- closed-book accuracy for each generator and dataset;
- the LLM detector's direction accuracy split by known vs unknown;
- the effect of `llm_dir+flag_directional` split the same way.

**Majority variant (exploratory).** We will report the LLM detector's direction accuracy on the
majority variant against the 1:1 set.

**ConflictBank minimal skeleton (exploratory).** We will report `cb_minimal` as exploratory.

**H14 with gpt-4.1.** H14 was registered with gpt-4.1 seed 0 (§10). We add seeds 1 and 2 and, decided
now and before those seeds exist, report the 3-seed estimates for H14 with gpt-4.1. The seed-0
values are kept in the appendix.

**"No harm" and "tie" claims.** Any such claim is reported with its 90% CI and a TOST at the
±3-point margin fixed in §9.3. Without equivalence, the paper says "not significant", not "no
harm" or "tie".

**One significance scheme.** §10.3 still applies. H15–H18 are registered families, and every
other comparison is in the single exploratory family.

## 12. Current models (frozen 2026-10-27, before any run below)

gpt-4o-mini and gpt-4.1 are no longer current. We add two current OpenAI models:
- `gpt-6-luna` (the current small tier);
- `gpt-6.1-terra` (a current frontier tier).

Prices on 2026-10-27, per 1M input / cached input / output tokens:
- `gpt-6-luna`: $0.10 / $0.01 / $0.50;
- `gpt-6.1-terra`: $1.00 / $0.05 / $5.00.

To stay within budget, two §11 runs are dropped before they start:
- `cbmaj_single_gpt41`. In H17 it is replaced by `gpt-6.1-terra`.
- gpt-4.1 seeds 1–2 for H14. H14 with gpt-4.1 stays as registered in §10, on seed 0. Every comparison
  involving gpt-4.1 baselines in the paper uses seed-0-matched numbers. This reverses the §11.3 plan,
  and no seed 1–2 data exists.

**Decoding.** Both models are reasoning models. A smoke test on 2026-10-27 established this:
- They reject `max_tokens` and any temperature other than their default (1).
- They therefore run at their default temperature and default reasoning effort.
- They get `max_completion_tokens=4096`, so reasoning cannot crowd out the answer. In the smoke test a
  call used 50–300 completion tokens.
- The `seed` parameter is sent but does not make sampling deterministic.

Everything else is unchanged:
- the LLM detector stays gpt-4o-mini;
- the prompts, the scorer and the significance scheme are unchanged.

### 12.1 Runs

| Output | Data | Generator | Seeds | Skeleton | Conditions |
|---|---|---|---|---|---|
| `runs/cb_single_luna` | ConflictBank | gpt-6-luna | 0, 1, 2 | single | `plain`, `closedbook`, `self_identify`, `oracle+flag`, `oracle+flag_onewrong`, `oracle+flag_directional`, `oracle+remove_suspect`, `llm_dir+flag`, `llm_dir+flag_directional` |
| `runs/cb_single_terra` | ConflictBank | gpt-6.1-terra | 0 | single | same as `cb_single_luna` |
| `runs/holdout_luna` | RAMDocs hold-out | gpt-6-luna | 0, 1, 2 | default | `plain`, `oracle+flag`, `oracle+flag_onewrong`, `oracle+flag_directional`, `oracle+remove_suspect`, `llm_dir+flag`, `llm_dir+flag_directional` |
| `runs/holdout_terra` | RAMDocs hold-out | gpt-6.1-terra | 0 | default | same as `holdout_luna` |
| `runs/cbmaj_single_luna` | ConflictBank majority | gpt-6-luna | 0, 1, 2 | single | `plain`, `oracle+flag`, `oracle+flag_directional`, `llm_dir+flag_directional` |
| `runs/cbmaj_single_terra` | ConflictBank majority | gpt-6.1-terra | 0 | single | same as `cbmaj_single_luna` |

The §11 budget rule applies: a run that would start with total spend at or above $95 is dropped.

### 12.2 Hypotheses

**H19. The core findings with current models.** Each test is run separately for `gpt-6-luna` and for
`gpt-6.1-terra`, and Holm runs within H19, per metric.

| Setting | Comparison | Questions | Prediction |
|---|---|---|---|
| ConflictBank `single` | Δ(`oracle+flag`, `plain`) | conflict | two-sided |
| ConflictBank `single` | Δ(`oracle+flag_directional`, `oracle+flag`) | conflict | > 0 |
| ConflictBank `single` | Δ(`oracle+flag_directional`, `oracle+flag_onewrong`) | conflict | > 0 |
| ConflictBank `single` | Δ(`llm_dir+flag_directional`, `plain`) | all | > 0 |
| RAMDocs default | Δ(`oracle+flag`, `plain`) | conflict | two-sided |
| RAMDocs default | Δ(`oracle+flag_directional`, `oracle+flag`) | conflict | > 0 |
| ConflictBank majority | Δ(`oracle+flag_directional`, `plain`) | conflict | > 0 |
| ConflictBank majority | Δ(`llm_dir+flag_directional`, `plain`) | conflict | two-sided |

If a reasoning model resolves conflicts unaided, we expect three things: a high no-flag baseline, a
neutral-flag effect near zero, and a smaller gain from direction. The paper reports whichever pattern
appears.

## 13. Fourth revision addendum (frozen 2026-10-27, after a fourth mock review, before any run below)

A fourth mock review, with three reviewers and a discussion round, scored the paper 2.5. The
reviewers agreed on five things to address:

1. Strict accuracy scores abstention as an error. They asked for misinformation rate, abstention and
   selective accuracy, plus a forced-choice prompt.
2. The pair structure of a flag may leak which side is false. They asked for flags without pair
   information.
3. Part of the directional flag's gain may be answer-key leakage. They asked for an answer-key control.
4. "The detector must beat the generator" was post hoc. They asked for a crossed detector × generator
   test on a common scale.
5. The paper lacks realistic data and models from other providers.

The budget is raised to $200 and the budget rule becomes: drop any run that would start at or above
$195.

### 13.1 New conditions

- **`forced` skeleton** (`prompts.BASE_FORCED`). Like `single`, but "unknown" is not allowed.
- **`flag_pick`.** The neutral flag plus "decide which side is better supported ... and answer with
  that side only". This is a hedged neutral wording that asks the model to commit to one side.
- **`flag_nopairs`, `flag_onewrong_nopairs`, `flag_directional_nopairs`.** The same flags with no pair
  structure: flagged passages are marked, but nothing says which passage conflicts with which.
- **`answer_key`.** Plain, plus "The answer X is false" for each wrong answer. No passage is marked.
- **Crossed detectors.** The LLM detector runs with `--detector-model` set to gpt-4.1, gpt-6-luna or
  gpt-6.1-terra, in addition to gpt-4o-mini.

### 13.2 Runs

Runs are listed in priority order.

| Output | Data | Generators (seeds) | Skeleton | Conditions |
|---|---|---|---|---|
| `runs/cb_forced_{mini,luna,terra}` | ConflictBank | gpt-4o-mini (3), gpt-6-luna (3), gpt-6.1-terra (1) | forced | `plain`, `oracle+flag`, `oracle+flag_onewrong`, `oracle+flag_directional`, `oracle+flag_pick`, `llm_dir+flag_directional` |
| `runs/ctl_{rd,cb,mj}_{mini,luna,terra}` | RAMDocs (default), ConflictBank (single), CB majority (single) | same | as listed | `answer_key`, `oracle+flag_pick`, `oracle+flag_nopairs`, `oracle+flag_onewrong_nopairs`, `llm_dir+flag_directional_nopairs`, and on CB majority also `oracle+flag_onewrong` |
| `runs/cross_cb_<gen>_<det>` | ConflictBank | gen ∈ {gpt-4o-mini (3), gpt-6-luna (3), gpt-4.1 (1), gpt-6.1-terra (1), Qwen2.5-7B (1, local)}; det ∈ {gpt-4.1, gpt-6-luna, gpt-6.1-terra} | single | `plain`, `llm_dir+flag_directional` |
| `runs/cross_mj_<gen>_<det>` | CB majority | same | single | same |
| `runs/conflicts_auto` | CONFLICTS | gpt-4o-mini, adds seeds 1, 2 | as registered | existing conditions |

The `cross_mj` runs come last and are the first to be dropped under the budget rule.

### 13.3 Hypotheses

Primary metric: one-way strict accuracy, except where noted. Holm runs within each family. The
families are run per generator unless stated.

- **H20: forced choice.** ConflictBank `forced`, conflict questions:
  - Δ(`oracle+flag`, `plain`), two-sided;
  - Δ(`oracle+flag_onewrong`, `plain`), two-sided;
  - Δ(`oracle+flag_directional`, `plain`) > 0.
- **H21: risk and coverage.** ConflictBank `single` and `forced`, conflict questions. For
  `oracle+flag` and `oracle+flag_directional` against `plain`, two-sided, on:
  - (a) misinformation rate (the answer contains the wrong answer);
  - (b) abstention rate;
  - (c) selective accuracy (strict accuracy among non-abstained answers, per question and averaged
    over seeds).
- **H22: pair topology.** Conflict questions, all two-sided:
  - Δ(`oracle+flag_onewrong_nopairs`, `oracle+flag_onewrong`) on RAMDocs;
  - Δ(`oracle+flag_nopairs`, `oracle+flag`) on CB majority;
  - Δ(`llm_dir+flag_directional_nopairs`, `llm_dir+flag_directional`) on CB majority.
- **H23: answer key.** Δ(`oracle+flag_directional`, `answer_key`), conflict questions, two-sided.
  Settings: RAMDocs, ConflictBank `single` and CB majority.
- **H24: reliability gap (crossed).**
  - **Cells.** A cell is a (detector, generator) pair on ConflictBank `single`, conflict questions.
    This includes the existing gpt-4o-mini-detector cells.
  - **Gain.** Gain = Δ(`llm_dir+flag_directional`, `plain`).
  - **Common-scale predictor.** d_q = 1 if the detector flags the gold contradiction pair with
    exactly the misinformation passage as suspect, else 0. g_q is the generator's plain strict
    accuracy on q, averaged over seeds. The cell gap is mean(d_q) − mean(g_q).
  - **H24a.** Spearman(gap, gain) across cells > 0. One-sided permutation over cell labels, 10,000
    permutations.
  - **H24b.** At question level, pooled over cells with cell fixed effects, the slope of the
    per-question gain on d_q − g_q is > 0. Cluster bootstrap over questions.
  - The same analysis on CB majority is exploratory.

### 13.4 Exploratory analyses and reporting changes

**Exploratory analyses (registered):**
- CONFLICTS on 3 seeds, by conflict type.
- An LLM-judge check of the string matcher on 200 outputs, stratified by condition and generator. The
  judge is gpt-6.1-terra; we report agreement and kappa with the matcher. No human labels.
- Per-question reasons the LLM detector's directional flag gains under majority misinformation:
  whether it marks the correct passage, and how many wrong passages it marks.
- A random-effects pooled estimate (DerSimonian–Laird) of the oracle neutral flag's effect across all
  settings, replacing the vote count. Settings share questions, so the pooled CI is optimistic, and
  we say so.

**Reporting changes:**
- Every "as good as" or "same picture" claim gets a TOST at ±3 points or is reworded.
- H13 is also analysed as flag × instruction, using `instruct_v2` alone as the control.

**Other providers and RAGuard.** Claude and Gemini runs, and RAGuard, will be registered in a further
addendum before they run, once API access and the data are confirmed.

## 14. Other providers and realistic misinformation (frozen 2026-10-27, before any run below)

**Access.** An Anthropic API key was made available. No Gemini key was available, so Gemini is not run;
if one arrives before §14's runs end, it is added in a further addendum.

**Claude models.** Claude runs through Anthropic's OpenAI-compatible endpoint.
- `claude-haiku-5-5`: $0.10 input / $0.01 cached input / $0.50 output per 1M tokens.
- `claude-sonnet-5-5`: $2.00 / $0.10 / $10.00.
- A smoke test on 2026-10-27 established the decoding settings:
  - The endpoint rejects `temperature`, so Claude runs at its default.
  - Claude thinks before answering, so it gets `max_tokens=4096`, as the reasoning models do. With 256
    tokens, answers were truncated or empty.
  - JSON response mode is not accepted. When Claude is the detector, the detector's existing
    regex fallback parses the JSON from the text.
- **Parser change.** Claude Haiku sometimes copies the answer-line template literally
  (`Final answer(s): <answer>Paris</answer>`). The answer parser now unwraps paired tags. No existing
  result contains a paired tag, so nothing earlier is rescored.

**RAGuard** (Zeng et al., NeurIPS 2025; Hugging Face `UCSC-IRKM/RAGuard` at `dae344a2`, MIT on the
card; `documents.csv` sha256 `ef1f5931…`).
- What it contains: PolitiFact claims with a binary verdict, plus Reddit posts that an LLM labelled
  supporting, misleading or unrelated. The misleading side is real online text, not written for a
  benchmark. The labels are LLM-assigned stances, not human-verified facts.
- What we build (`prepare_raguard.py`):
  - 294 conflict items: claims with at least one usable supporting post and one usable misleading post.
  - 100 clean items: a seed-0 sample of claims with supporting posts and no misleading post.
  - A post is usable if it has at least 15 words, ignoring the `[Link Post]` placeholder, and is not a
    duplicate within its claim.
  - Each item keeps up to 2 supporting, 2 misleading and 2 unrelated posts, each cut to 300 words.
- The question is "Is the following claim true or false?". The gold answer is the verdict; the wrong
  answer is its opposite. Scoring uses the strict metrics. Supporting × misleading pairs are the
  contradictions, and misleading posts are the misinformation side.
- No pilot was run, so every result is confirmatory for the tests below.

### 14.1 Runs, in priority order

The budget rule is the same as before: a run that would start at or above $195 total is dropped. The
`cross_mj` runs of §13 rank below all §14 runs and are stopped first if spend approaches the limit.

1. **RAGuard**, `single` skeleton, gpt-4o-mini detector.
   - Generators: gpt-4o-mini (3 seeds), gpt-6-luna (3), claude-haiku-5-5 (3) and gpt-6.1-terra (1).
   - Conditions: `plain`, `closedbook`, `oracle+flag`, `oracle+flag_onewrong`, `oracle+flag_directional`,
     `oracle+remove_suspect`, `llm_dir+flag_directional`, `answer_key`.
2. **claude-haiku-5-5**, 3 seeds:
   - ConflictBank `single` and RAMDocs: the conditions of `cb_single_luna` and `holdout_luna`.
   - ConflictBank majority: the conditions of `cbmaj_single_luna`.
   - ConflictBank `forced`: the conditions of `cb_forced_*`.
3. **claude-sonnet-5-5**, seed 0:
   - ConflictBank `single`: `plain`, `oracle+flag`, `oracle+flag_onewrong`, `oracle+flag_directional`,
     `llm_dir+flag_directional`.
   - RAMDocs and RAGuard: `plain`, `oracle+flag`, `oracle+flag_directional`, `llm_dir+flag_directional`.
4. **Crossed cells added to H24** (ConflictBank `single`):
   - Claude as detector: claude-haiku-5-5 and claude-sonnet-5-5, each with gpt-4o-mini, gpt-6-luna
     and claude-haiku-5-5 as generators.
   - Claude as generator: claude-haiku-5-5, with gpt-4.1, gpt-6-luna and gpt-6.1-terra as detectors.

### 14.2 Hypotheses

Holm runs within each family, per metric. All comparisons are on conflict questions unless noted.

**H25: Claude.** Each Claude model is tested separately, on the H19 set where the conditions were run:

| Benchmark | Comparison | Prediction |
|---|---|---|
| ConflictBank | Δ(`oracle+flag`, `plain`) | two-sided |
| ConflictBank | Δ(`oracle+flag_directional`, `oracle+flag`) | > 0 |
| ConflictBank | Δ(`oracle+flag_directional`, `oracle+flag_onewrong`) | > 0 |
| ConflictBank | Δ(`llm_dir+flag_directional`, `plain`), all questions | > 0 |
| RAMDocs | Δ(`oracle+flag`, `plain`) | two-sided |
| RAMDocs | Δ(`oracle+flag_directional`, `oracle+flag`) | > 0 |
| CB majority (Haiku only) | Δ(`oracle+flag_directional`, `plain`) | > 0 |
| CB majority (Haiku only) | Δ(`llm_dir+flag_directional`, `plain`) | two-sided |

**H26: RAGuard.** Each generator is tested separately:

| Comparison | Prediction |
|---|---|
| Δ(`oracle+flag`, `plain`) | two-sided |
| Δ(`oracle+flag_directional`, `plain`) | > 0 |
| Δ(`oracle+flag_directional`, `oracle+flag`) | > 0 |
| Δ(`oracle+flag_onewrong`, `plain`) | two-sided |
| Δ(`llm_dir+flag_directional`, `plain`) | two-sided |

**H24, extended.** The primary H24 analysis uses every cell run under §13 and §14. The §13-only set is a
sensitivity check.

## 15. Fifth revision addendum (frozen 2026-10-28, after a fifth mock review, before any run below)

A fifth mock review, with three reviewers and a consensus round, scored the paper 3.0 (Findings). Its
must-fix list asks for:
- a detector-answer control for the crossed cells;
- the LLM directional flag and LLM removal where conflicts are genuine;
- a probe of whether detector direction rests on surface artefacts of ConflictBank's generated passages;
- re-analyses of existing data: H24 without the shared −G coupling, TOST for the null claims of H20, H21
  and H23, a dependence-aware pooled estimate, a cost-sensitive view of abstention, and a valid H21
  selective-accuracy estimand.

Spend before this section: $169.23. The budget rule is unchanged: a run that would start at or above
$195 is dropped. Runs are listed in priority order, and the lowest-priority runs are dropped first.

### 15.1 New conditions and data

- **`hint`** (`prompts.py`; `run.py --hint-run <dir>`). The plain prompt plus
  `NOTE: Another model that read these passages answered: "X".`
  - X is the detector model's own parsed answer under `plain`, seed 0, in its ConflictBank `single` run.
    Several answers are joined with " | ".
  - No passage is marked.
  - Detector model → source run: gpt-4o-mini → `cb_single`; gpt-4.1 → `cb_single_gpt41`; gpt-6-luna →
    `cb_single_luna`; gpt-6.1-terra → `cb_single_terra`; claude-haiku-5-5 → `cb_single_haiku`;
    claude-sonnet-5-5 → `cb_single_sonnet`.
  - The question it answers: is a crossed cell's gain just a stronger model's answer being forwarded?
- **Artefact probe** (`prepare_cb_probe.py`; datasets `conflictbank_rev` and `conflictbank_para`). Both
  hold the same 400 ConflictBank items.
  - `rev`: the passage order is reversed.
  - `para`: every passage (correct, misinformation and distractor alike) is paraphrased by gpt-4o-mini
    at temperature 0, with one prompt that keeps every fact unchanged.
  - The `para_check` field records, per passage, whether the gold and wrong answer strings that appeared
    in the original still appear in the paraphrase. Failures are kept and counted.

### 15.2 Runs, in priority order

1. **Probe.**
   - Build `rev` and `para`.
   - Runs `runs/probe_{rev,para}_mini`: gpt-4o-mini detector and generator, 3 seeds, `single` skeleton;
     conditions `plain`, `oracle+flag_directional`, `llm_dir+flag_directional`.
   - Runs `runs/probe_{rev,para}_lunadet`: gpt-6-luna detector, gpt-4o-mini generator, 3 seeds;
     conditions `plain`, `llm_dir+flag_directional`.
2. **Hint, cheap generators.** `runs/hint_cb_<gen>_<det>` for every H24 cell whose generator is
   gpt-4o-mini, gpt-6-luna, claude-haiku-5-5 (3 seeds each) or Qwen2.5-7B (seed 0, local). Conditions:
   `plain` and `hint`. `plain` is a cache hit, so the cost is the `hint` calls alone.
3. **Genuine conflicts.**
   - WikiContradict, `runs/wiki_llm_{mini,luna,haiku}`: 3 seeds, default skeleton, gpt-4o-mini detector;
     conditions `plain`, `oracle+flag`, `llm_dir+flag_directional`, `llm_dir+remove_suspect`.
   - RAMDocs hold-out, `runs/holdout_rem_{luna,haiku,terra}`: the same generators and seeds as
     `holdout_*`, gpt-4o-mini detector; conditions `plain`, `llm_dir+flag_directional`,
     `llm_dir+remove_suspect`. The detections are cache hits.
4. **Hint for the gpt-6.1-terra generator cells**, seed 0.
5. **Hint for the gpt-4.1 generator cells**, seed 0.
6. **Single-call mode for gpt-4.1**: `self_identify` on ConflictBank `single`, seed 0, added to
   `cb_single_gpt41`.

### 15.3 Hypotheses

Primary metric: one-way strict accuracy, except on WikiContradict, which uses `both_answers`. Holm runs
within each family. Conflict questions unless stated, and clean questions are reported alongside.

**H27: detector answer as hint.** The cells are those of H24 (ConflictBank `single`). Each comparison
forms one family across cells:
- H27a: Δ(`llm_dir+flag_directional`, `hint`), two-sided.
- H27b: Δ(`hint`, `plain`), two-sided.
- H27c: Δ(cell `llm_dir+flag_directional`, the detector model's own `plain` accuracy), two-sided.
  - The comparison is paired by question.
  - It is the cost-matched alternative: use the detector model as the generator.
  - Only cells whose detector model has a ConflictBank `single` run are included; that is all of them.

If the gain were only a stronger model's answer being forwarded, the hint would match the flag and the
detector model alone would match the cell.

**H28: genuine conflicts.** One family per generator and benchmark, all two-sided:
- WikiContradict: Δ(`llm_dir+flag_directional`, `plain`) and Δ(`llm_dir+remove_suspect`, `plain`).
- RAMDocs, questions whose conflict type is `ambiguity` (several valid answers, no misinformation): the
  same two comparisons, on strict accuracy and on gold recall.
- The gpt-4o-mini RAMDocs rows already exist (`holdout_main`). They have been reported only pooled over
  clean questions, never on this subset. We say so.

**H29: artefact probe.** Detectors gpt-4o-mini and gpt-6-luna; ConflictBank conflict questions.
- H29a: Δ direction accuracy d_q (as in H24) for `rev` − original and `para` − original. Two-sided,
  sign-flip over questions; Holm over the four tests.
- H29b: TOST at ±5 points on the same differences.
- H29c: Δ(`llm_dir+flag_directional`, `plain`) with the gpt-4o-mini generator on `rev` and on `para`,
  compared with the original. A difference-in-differences, two-sided.
- The `para` tests are also reported on items whose every passage passes `para_check`.

### 15.4 Re-analyses of existing data

These analyses are registered here but run on data already seen, and are reported as such.

1. **H24 without the shared −G coupling.**
   - (a) Cell-level regression of the flag's accuracy A on the detector's direction accuracy D and the
     generator's no-flag accuracy G, with and without generator fixed effects.
     - Inference: a question bootstrap that resamples questions jointly across cells.
     - A structure-respecting exact permutation: the D values are permuted across the six detectors,
       consistently in every generator.
   - (b) A question-level linear probability model of Y1 (correct with the flag) on d_q, g_q and a
     wrong-side indicator w_q, with crossed random intercepts for detector, generator and question
     (MixedLM variance components).
   - (c) Leave-one-detector-out prediction of the cell gains by four models: G only; D + G additive; the
     registered gap D − G; the mechanism score.
   - (d) H24a on the cells that did not exist when §13 was committed. Five cells existed then: the
     gpt-4o-mini detector with gpt-4o-mini, gpt-6-luna, gpt-6.1-terra, gpt-4.1 and Qwen2.5-7B.
   - (e) A compliance decomposition per generator:
     - P(Y1 = 1 | d_q = 1, g_q = 0): the generator follows a correct flag it needed;
     - P(Y1 = 0 | w_q = 1, g_q = 1): the generator follows a wrong-side flag;
     - the calibration slope of the cell gains on the mechanism score.
   - (f) Direction accuracy is d_q throughout, computed on the detections each cell's generator
     actually saw. Detections can differ slightly between cells of the same detector, because concurrent
     runs made independent uncached calls; we report how often they differ.
2. **TOST at ±3 points**, as §13.4 already requires, for every null claim:
   - H20's neutral and one-wrong comparisons;
   - H21's selective accuracy;
   - H23's comparisons.
3. **Pooled neutral-flag estimate** (registered as exploratory in §13.4).
   - List the settings.
   - Replace the CI by a question-cluster bootstrap of the DerSimonian–Laird estimate: question ids are
     resampled within each benchmark, jointly across all settings that share them, with 2,000
     replicates.
   - Report a prediction interval.
   - Run a mixed-effects meta-regression with three moderators: model family (OpenAI, Anthropic, Qwen);
     whether "unknown" is allowed; synthetic (ConflictBank, CB majority) vs natural (RAMDocs, RAGuard,
     CONFLICTS) data.
4. **Abstention-aware utility.**
   - Per answer: +1 if correct; −c if it contains the misinformation; 0 if it abstains or is otherwise
     wrong. Here c ∈ {0, 1, 2, 4}.
   - Δ vs `plain` for the oracle neutral, one-wrong and directional flags, in every setting with an
     oracle.
   - Also per-question transition matrices from `plain` to each flag (correct, misinformation, abstain,
     other), per seed.
5. **H21(c) as a ratio estimand.**
   - Selective accuracy = Σ correct / Σ answered over question × seed.
   - The CI comes from a question bootstrap. The p-value comes from a permutation that swaps the two
     conditions within each question.
   - This replaces the per-question mean, which drops questions abstained in every seed, so the two
     conditions were averaged over different question sets. Both are reported, and the change is listed
     as a deviation.
6. **Descriptive checks.**
   - JSON parse-failure rates of each LLM detector.
   - The answer mix of the reasoning models' `plain` answers on RAMDocs, to explain their low strict
     accuracy.
7. **Human audit.** 200 outputs, stratified by condition and generator, are exported for author
   labelling (`runs/labelling/audit_200.csv`). Agreement with the matcher is reported only if the labels
   are complete before submission; this is not done by us.

## 16. Sixth revision addendum (frozen 2026-10-28, after a sixth mock review, before any run below)

A sixth mock review, with three reviewers and a consensus round, scored the paper 3.0 (Findings). Its must-fix list
asks for:
- scoping the neutral-flag claim to the answer policy;
- reconciling inconsistencies;
- separating confirmatory from post hoc claims;
- recasting the "where errors fall" analysis, which is observational and has an asymmetric definition of following;
- a detector-judgement answer key as the control for H27;
- a dependence-aware meta-analysis and seed-to-seed nulls;
- the missing prompts.

Spend before this section: $185.14. The budget rule is unchanged: a run that would start at or above $195 is
dropped. Runs are listed in priority order.

### 16.1 New conditions

- **`verdict_key`** (with an LLM detector: `llm_dir+verdict_key`). The plain prompt plus
  `NOTE: The answer "X" is false.` for each answer supported by a passage the detector marks as suspect.
  - On ConflictBank a correct passage supports the gold answer and a misinformation passage supports the wrong answer,
    so a wrong-side mark yields "the [gold answer] is false".
  - No passage is marked.
  - Suspect distractors support neither answer and are ignored.
  - This passes the detector's judgement of which side is false, as an answer key; `hint` passed the detector
    model's own answer.
- **Randomised direction**, `synth-p1-r1-all-d0.5+flag_directional`.
  - The gold contradiction pair of every conflict question is flagged.
  - The suspect is the misinformation passage with probability 0.5, drawn independently per question (seed 0);
    otherwise it is the correct passage.
  - Clean questions receive no flag.
  - Whether the flag is right is therefore independent of the question.

### 16.2 Runs, in priority order

1. **Randomised direction**, `runs/randdir_cb_{mini,luna,haiku,terra}`. ConflictBank `single`; gpt-4o-mini,
   gpt-6-luna and claude-haiku-5-5 with 3 seeds, gpt-6.1-terra with seed 0. Conditions: `plain` and the randomised
   condition.
2. **Verdict key**, `runs/vkey_cb_<gen>_<det>`. Every H24 cell whose generator is gpt-4o-mini, gpt-6-luna or
   Qwen2.5-7B, with the cell's seeds and its own detector. Conditions: `plain`, `llm_dir+verdict_key`. The detections
   are cache hits.
3. **Verdict key for the claude-haiku-5-5 generator cells.**

### 16.3 Hypotheses

Conflict questions; one-way strict accuracy; Holm within each family.

**H30: the detector's verdict as an answer key.** One family per comparison, across cells, all two-sided:
- H30a: Δ(`llm_dir+flag_directional`, `llm_dir+verdict_key`), with a TOST at ±3 points.
- H30b: Δ(`llm_dir+verdict_key`, `hint`).

**H31: randomised direction.** One family per estimand, across generators:
- H31a: the right-flag effect, the mean over right-assigned questions of the flagged minus the plain accuracy, > 0.
- H31b: the wrong-flag effect, the same over wrong-assigned questions, < 0.
- H31c: (a) + (b), two-sided. Is the harm of a wrong flag larger than the gain of a right one?
  - Inference: the arms are different questions, so the p-value is a permutation of arm labels across questions
    (10,000 permutations) and the CI a question bootstrap within arms.
- Reported with it: follow rates with one definition, conditioned on the plain outcome.
  - A right flag is followed when the flagged answer is correct.
  - A wrong flag is followed when the flagged answer contains the answer the flag implies, here the misinformation.
  - Abstentions are reported separately.

### 16.4 Re-analyses of existing data (registered here, run on data already seen, reported as such)

1. **Answer policy.**
   - Risk and coverage (correct, misinformation, abstain, selective) for the multi-answer settings: RAMDocs and
     ConflictBank `default`.
   - The pooled neutral-flag estimate and the utility analysis split by answer policy (multi-answer, one answer
     with "unknown", forced).
   - The meta-regression moderators recoded: answer policy, model family, and natural misinformation (RAGuard
     only; RAMDocs' misinformation is synthetic edits).
2. **Dependence-aware meta-analysis.**
   - A multilevel random-effects model with random effects for generator, benchmark and setting, fitted by REML with
     the settings' sampling variances. The pooled mean and a prediction interval are reported.
   - The utility and pooled analyses use the same set of settings.
3. **H24 recast.**
   - Direction recall R (the detector flags the gold pair) and direction precision P (among those flags, the suspect
     is exactly the misinformation side) enter separately: A ~ P + R + generator FE.
   - P gets the same exact permutation across detectors, and leave-one-detector-out R² is reported for P + R.
   - The question-level model is refitted with question, detector and generator fixed effects. CIs come from a
     question bootstrap, with the leave-one-detector-out range of the coefficients.
   - Follow rates use the §16.3 definition.
   - A regression-to-the-mean null: in 3-seed generators, the plain outcome on seed 2 given the plain outcome on
     seeds 0 and 1.
4. **Seed-to-seed null for the transition tables.** Plain seed s → plain seed s′ transitions, with the same
   categories.
5. **H14 under a matched prompt** (exploratory). The registered H14 on ConflictBank compared the pipeline under the
   multi-answer prompt (the only one registered at §10) with the baselines' own single-answer prompts. Add the
   comparison with the pipeline under the `single` prompt (`cb_single`), from existing runs.
6. **H29 equivalence** at ±5 points, as registered in §15, reported in the text. Every remaining "unchanged" claim
   gets a TOST or is reworded.
7. **H21(c) in the hypothesis tables** uses the §15.4.5 ratio estimand, with the per-question version in the
   released statistics.

## 17. Eighth revision addendum (frozen 2026-10-29, after an eighth mock review, before any run below)

An eighth (multi-agent) mock review found that the abstract's pooled interval ignores shared questions, that the
H31 break-even is transferred to selective detectors without its assumptions, that the follow-rate prediction is
retrospective, that RAGuard labels and WikiContradict scoring do not support factual-oracle language, that the
matcher lacks a condition-sensitive semantic check, and that two baselines were implemented with gaps. The HoH-QA
plan drafted after the seventh review is shelved; nothing from it is registered. Everything below is exploratory.

### 17.1 Re-analyses of existing data (registered here, run on data already seen, reported as such)

1. **Pooled neutral flag under shared questions.** The §16.4.2 multilevel model refitted inside a joint question
   bootstrap (1,000 replicates): each replicate resamples question ids once per question pool (ConflictBank, which
   its majority and forced variants share; RAMDocs; RAGuard) and applies the same draw to every setting, recomputes
   each setting's mean and sampling variance, and refits the intercept-only model, the moderator model and the
   per-policy models. Percentile CIs. The prediction interval is the 2.5–97.5% range of μ_b + z·√(Στ_b), z ~ N(0,1).
   Also reported: the equal-weight mean over settings with the same bootstrap (inference about this design only).
2. **H31 at the realised split.** Per generator, the mean flag effect over all randomised conflict questions (the
   159/141 mixture as run), paired bootstrap and sign-flip p, Holm over the generators, beside the registered 50/50
   H31c.
3. **Selection-aware decomposition of the crossed cells.** For each H24 cell: coverage, baseline accuracy and mean
   gain on questions whose flag names the false side, on those whose flag marks a correct passage, and on the rest;
   the marginal break-even prediction (precision × R + (1 − precision) × W) against the observed gain.
4. **The follow-rate prediction as a retrospective account.** Against baselines (zero; marginal rates without the
   question-level joint; the generator's mean over its other cells), with MAE and sign agreement; a joint bootstrap
   of H31 questions and cell questions for the CI of r and MAE; and, for cells with three seeds, a cross-fitted
   version whose prediction uses the plain answers of seed 0 and whose observed gain uses seeds 1–2.
5. **H23 heterogeneity.** Per-comparison 90% CIs and TOST at ±3, and the between-comparison τ (REML).
6. **Prevalence uncertainty.** A joint bootstrap (conflict and clean questions resampled within each pool) of the
   5% and 20% reweighted effects: per-setting CIs and the distribution of the number of settings below zero.

### 17.2 Corrected baselines (new runs, gpt-4o-mini, the original splits, seeds and prompts)

- `trustrag_sys`, `trustrag_sys_multi`: TrustRAG with the repository's system message ("You are a helpful
  assistant.") sent on all three calls. Runs: `rev8_cb_main` (ConflictBank eval, default skeleton, with `plain`),
  `rev8_cbmaj_single` (majority, single skeleton), `rev8_holdout_round2` (RAMDocs hold-out, both variants).
- `nli_win`: the same cross-encoder and threshold, but each passage is reduced to its sentences that best match the
  question (token overlap, ties by position), in original order, up to 1,500 characters, so the changed fact is in
  the input. With `flag` and `remove` in `rev8_cb_main`. Detection recall on the gold pairs is reported (no API).
- Reported: each corrected baseline against its original and in the H14 comparisons (the matched-policy version as
  the principal comparison). No new hypothesis tests beyond paired comparisons with Holm over the H14 family.

### 17.3 Semantic re-scoring (outside-family judge)

- Judge: gemini-2.5-flash (Google; no generator is from this family), temperature 0, thinking off, JSON output.
- Rubric, per response: final answer gives the gold answer (aliases count); asserts a wrong answer (mentioning it
  only to reject it does not count); explicitly rejects a wrong candidate; abstains; adds an answer outside both
  lists; acknowledges a conflict. Judge-strict = gives gold, asserts no wrong answer, does not abstain.
- Sample (seed 0, conflict questions, both arms of each pair on the same questions, 120 questions per contrast,
  fixed seed): neutral flag vs plain (RAMDocs gpt-4o-mini, ConflictBank single gpt-4o-mini and Haiku); oracle
  directional flag vs answer key (ConflictBank single gpt-4o-mini, gpt-6-luna, Haiku); LLM directional flag vs
  verdict key (ConflictBank gpt-4o-mini detector, generators gpt-4o-mini and gpt-6-luna); H31 right and wrong arms
  vs plain (gpt-4o-mini, Haiku).
- Reported: matcher-vs-judge disagreement per condition, the contrasts re-estimated under judge-strict with a
  question bootstrap, and whether the ±3 equivalence conclusions survive.
- WikiContradict under its original rubric (acknowledges the incompatibility, gives both answers, no unsupported
  preference): plain vs LLM directional flag, for the three H28 generators, all questions of seed 0.
- RAGuard pairs: 150 sampled supporting × misleading pairs checked by the judge for a real contradiction, temporal
  compatibility and whether the misleading post is false about the claim. Reported as a share, not a test.

### 17.4 Human audit

A blinded, paired sheet (the same question under two conditions, condition hidden, order randomised) over the
contrasts above and RAGuard pairs, for two labellers. Results are reported only if the labels are completed.

### 17.5 Budget

Estimated $4.5 (reruns about $2.3, judge about $2.2). Spend before this section: $188.34; the $195 gate stands.

### 17.6 Deviations (appended 2026-10-29, before the runs they describe)

- **Judge.** gemini-2.5-flash is closed to new users; the judge is gemini-3.5-flash-lite (same price, Google family),
  with minimal thinking (its endpoint does not accept "none") and no seed parameter (not accepted).
- **Windowed NLI.** With the registered 1,500-character window, the cross-encoder flags no ConflictBank pair (as
  truncated NLI): two such windows exceed its 512-token input and the scores stay near zero although both answers are
  now in the windows. After seeing this (on detection only, no generator output), we add `nli_win300`, the same rule
  with a 300-character window (about one sentence), which found the gold pair in 16 of 60 checked conflict items.
  It runs with `flag` and `remove` in `rev8_cb_main`; the registered `nli_win` results are reported as they are.

## 18. Ninth revision addendum (frozen 2026-10-29, after a ninth mock review, before any analysis below)

A ninth (multi-agent) mock review asks whether the RAMDocs oracle's contradiction pairs are semantically aligned,
whether the flag and the verdict key answer the same questions correctly despite matching mean accuracy, and notes
that the semantic re-score changes one contrast's sign. Checking the code after the review: `_ramdocs_pair_label`
labels a pair a contradiction whenever one side is misinformation and the answers differ, with no entity alignment,
and `verdict_answers` takes the false answer from the benchmark's per-passage labels. No new API calls. Everything
below is exploratory, run on data already seen, and outside the FDR count.

### 18.1 RAMDocs pair-rule audit (no model outputs)

On the hold-out (`splits/ramdocs_holdout.txt`, `label_fix="exclude"`): the number of conflict questions; those whose
correct passages give two or more distinct answers (*multi-entity*); the oracle contradiction pairs; and a lower bound
on cross-entity pairs, assuming each misinformation passage targets one entity: for each misinformation passage, its
contradiction pairs with correct passages minus the largest number that share one answer.

### 18.2 Neutral and directional oracle flags on single-entity questions

For every RAMDocs setting in the §16.4 pooled set, the oracle neutral flag minus plain (and the oracle directional flag
minus plain where that arm exists) on (a) *single-entity* conflict questions, whose correct passages all give one
answer, so no oracle pair can cross entities; and (b) the multi-entity rest. Per setting: mean, question-bootstrap 95%
CI (5,000 resamples), and the difference (a) − (b). Over the RAMDocs settings: the equal-weight mean of (a) and of
(b), with a joint question bootstrap (the same draw for every setting). Primary metric, seed means within questions.

### 18.3 Per-question agreement between interfaces

For each H30a cell (LLM directional flag vs verdict key) and each H23 comparison (oracle directional flag vs answer
key), over (question, seed) pairs of conflict questions scored in both arms: the share where exactly one arm is
correct (discordance), split into flag-only and key-only. As a null, the same share between plain seeds s and s′ on the
same questions (generators with at least two plain seeds). Pooled as equal-weight means over cells, with a joint
question bootstrap within benchmarks.

### 18.4 Reporting changes (no analysis)

- The §17.3 paragraph: one contrast changes sign (−0.8 to +0.8, both intervals across zero), and two contrasts whose
  matcher intervals include zero exclude it under the judge (unadjusted bootstrap CIs), one of them the gpt-4o-mini
  H30 comparison at +6.7.
- H30 is described as an information-content control: the verdict key uses the benchmark's per-passage labels to
  turn a marked passage into an answer.
- "Equivalent" claims are qualified as mean accuracy under the primary metric; the H23 "SD" is labelled τ (REML).
- The RAMDocs pair rule and the §18.1 counts are described in Setup and Limitations; "perfectly accurate" becomes
  accurate against the benchmark's labels.
- Figure 1 is redrawn (layout only); the evidence-status table for the benchmarks moves into the main text if it fits.

### 18.5 Note on timing (appended 2026-10-29)

The §18.1 counts were first computed during the review triage, before this section was written (hold-out: 139
conflict questions, 98 multi-entity, 709 oracle contradiction pairs, at least 261 cross-entity). §18.2–18.3 were not.

## 19. Prospective test, full-sample semantic re-score and a natural-document check (frozen 2026-10-29, before any run below)

The ninth mock review's top requests that need new data: (a) test the follow-rate account on unseen questions, with
the procedure frozen, against simple baselines, including clean questions and an unseen detector; (b) check the two
pooled equivalence claims under semantic scoring on more than a 120-question sample; (c) replicate a small frozen set
of core conditions on natural documents. A human audit remains outside what we can run. The budget is raised to $250
by the authors (2026-10-29); the gate before every run is $245; spend before this section is $190.55. The code below
(`run.py --detect-only`, `predict19.py`, `judge10.py`, `revision10.py`) is committed with this section.

### 19.1 Prospective follow-rate test on ConflictBank-fresh (H32–H36)

**Data.** `splits/conflictbank_fresh.txt`: 400 items (300 conflict, 100 clean), built like the §10 set and sharing no
subject with it (79737eb); no generator or flagged output on it exists yet.

**Cells.** Generators gpt-4o-mini (T 0.1, seeds 0–2), gpt-6-luna (default temperature, seeds 0–2), claude-haiku-5-5
(T 0.1, seeds 0–2), gpt-6.1-terra (default temperature, seed 0), as in the round-10 crossed cells. Detectors (`llm_dir`,
greedy): gpt-4o-mini, gpt-6-luna, claude-haiku-5-5, gpt-6.1-terra, and Qwen2.5-7B (local, 4-bit; never a ConflictBank
`llm_dir` detector in any earlier cell). Prompt skeleton `single`; conditions `plain` and `llm_dir+flag_directional`.
20 cells, runs `fresh_cb_{gen}_{det}`.

**Order (the freeze).** Stage 1: every detector's detections (`--detect-only`) and every generator's `plain` answers.
Then `predict19.py` computes each cell's prediction from stage-1 files only and writes
`splits/predictions19.json`, committed before stage 2. Stage 2: the flagged condition. No flagged output is read
before that commit.

**Predictor (frozen).** For cell (G, D) and question i: g_i is G's plain primary-metric score, the mean over seeds;
right_i = 1 if D flags a gold contradiction pair and its suspects are exactly that pair's misinformation member(s) (as
H24's d_q); wrong_i = 1 if D's suspects include a correct passage. Then

    pred(G, D) = mean_i [ right_i (1 − g_i) fR_G − wrong_i g_i fW_G ]

with the round-10 follow rates, not refitted: gpt-4o-mini 0.5658 / 0.5143; gpt-6-luna 0.7202 / 0.7719;
claude-haiku-5-5 0.6288 / 0.6392; gpt-6.1-terra 0.6667 / 0.9474 (fR / fW). The observed effect is
obs(G, D) = mean_i (flag_i − g_i) over the same questions, flag_i the mean over seeds.

**H32 (primary; the 20 cells, conflict questions).** All three must hold:
1. the OLS calibration slope of obs on pred has a 95% CI (bootstrap over questions, one draw shared by all cells,
   5,000 resamples) that excludes 0 and includes 1;
2. the mean absolute error |pred − obs| over cells is at most 3 points;
3. pred has lower MAE than each of two baselines, by a paired bootstrap of the MAE difference (same draws), one-sided
   α = 0.025 each, Holm over the two: (a) the generator's mean observed effect over its round-10 crossed cells
   (`H24` cells, all detectors); (b) the same formula with fR = fW = 1.

Reported alongside, with no success criterion: (c) the cell-level marginal estimate
fR · mean(right) · (1 − mean g) − fW · mean(wrong) · mean(g), which the retrospective account barely beat (MAE 1.6 vs
1.5), so we expect a near tie; (d) zero. Also r, the within-generator r, sign errors and per-cell values.

**H33 (sign).** Among cells with |pred| ≥ 3 points, obs has the same sign as pred in at least 80%. For each cell with
pred ≤ −3, obs < 0 by a one-sided paired sign-flip test over questions at α = 0.05, Holm across those cells.

**H34 (within generators).** The correlation of pred and obs after subtracting generator means is above 0: one-sided
permutation test (detector labels permuted within generator, 10,000 draws), α = 0.05.

**H35 (clean questions; secondary).** The same formula on the 100 clean questions (right_i = 0, so pred ≤ 0): MAE and
sign agreement per cell, reported without a success criterion. fW was measured on conflict questions, where following
a wrong flag means adopting the misinformation; on clean questions it can only mean dropping or changing a correct
answer, so the formula may misestimate this harm.

**H36 (follow-rate stability).** Randomised direction (`synth-p1-r1-all-d0.5+flag_directional`, with `plain`) for
gpt-4o-mini on ConflictBank-fresh, seeds 0–2 (`randdir_fresh_mini`). Holds if the fresh fR and fW each lie inside the
frozen 95% CIs ([0.46, 0.67] and [0.40, 0.62]). Not used for prediction.

**Selection (descriptive).** Per generator, the detector with the highest prediction over all 400 questions (conflict
and clean, as sampled), or no flag if every prediction is negative; its observed effect against the best observed
detector's and against the gpt-4o-mini detector's.

### 19.2 Full-sample semantic re-score of the pooled equivalence tests (H38)

Judge, prompt and judge-strict score as §17.3 (gemini-3.5-flash-lite). Every conflict response of the conditions
`answer_key`, `oracle+flag_directional`, `llm_dir+verdict_key` and `llm_dir+flag_directional` in the run families that
feed the §16.4 pooled tests (`vkey_cb_*`, `cross_cb_*`, `cb_single*`, `ctl_{rd,cb,mj}_*`, `holdout_*`,
`cbmaj_single*`), judged once per distinct (dataset, question, response), about 22,000 calls.
- **H38a.** Pooled H30a (LLM directional flag minus verdict key) re-estimated with judge-strict in place of the
  matcher, the same cells, pooling and joint bootstrap (`revision7._pool`); equivalence within ±3 by its 90% CI.
- **H38b.** The same for pooled H23 (oracle directional flag minus answer key).
- If either fails, the paper states that equivalence holds under the matcher only. The ±3 margin is not re-chosen (it
  was set after seeing data, §16); we also report the smallest symmetric margin each 90% CI fits inside under both
  scorers, and the matcher-minus-judge difference of each pooled contrast with a bootstrap CI.
- Matcher–judge agreement per condition on the full sample is reported.

### 19.3 HealthContradict: a natural-document check (H37)

**Data.** `splits/healthcontradict_eval.txt`: 247 yes/no health questions over 81 TREC Health Misinformation topics,
each with one real web page per side; the page agreeing with the topic's evidence-based answer is typed correct, the
other misinformation (labels as released; not re-verified by humans). Every item is a conflict. Seen before this
section: a 30-item gpt-4o-mini smoke in §17 preparation (plain 70%, closed-book 77%, LLM directional flag 70%; detector
direction accuracy 0.46). A judge audit (descriptive) asks, per item, whether each page supports its labelled answer
(247 calls; `judge10.py --part hc_labels`).

**Runs.** Generators gpt-4o-mini, gpt-6-luna, claude-haiku-5-5, seeds 0–2 (temperatures as above); skeleton `single`;
conditions `plain`, `closedbook`, `oracle+flag`, `oracle+flag_directional`, `llm_dir+flag_directional` (gpt-4o-mini
detector, `--detector-max-chars 4000`); runs `hc_{gen}`. Same freeze as 19.1: detections and `plain` first; the
`llm_dir` predictions (generator's own frozen follow rates) go into `predictions19.json` before any flagged output.

**Tests** (primary metric; per-question seed means; paired, with the topic as the cluster for bootstrap CIs and for
sign-flip permutation, 10,000 draws; Holm over the three generators within each):
- **H37a.** Oracle directional flag minus plain > 0 (one-sided).
- **H37b.** Oracle neutral flag minus plain ≠ 0 (two-sided); TOST at ±3 reported.
- **H37c.** Oracle directional minus oracle neutral > 0 (one-sided).
- **H37d (descriptive).** The LLM directional flag: obs against the frozen prediction, per generator (sign and error).

### 19.4 Budget and order

Estimated: 19.1 about $14, 19.2 about $6, 19.3 about $9, label audit under $0.5; about $30, $38 with 25% contingency.
Order: 19.1 stage 1; 19.3 stage 1 and its label audit; commit `predictions19.json`; 19.1 stage 2; 19.3 stage 2;
`randdir_fresh_mini`; 19.2. A run that would start at or above the $245 gate is skipped and every hypothesis that needs
it is reported as not run. §19 tests join the programme-wide BH count.
