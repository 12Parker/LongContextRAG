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
