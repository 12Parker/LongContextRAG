# Contradiction-Aware RAG — Project Context for Claude Code

Written 2026-10-26 from a claude.ai planning conversation. Read this before working in
`experiments/conflict/`. Repo: https://github.com/12Parker/LongContextRAG, base commit `89e545d`.

---

## 1. Situation

**Paper:** *Efficient Long-Context Language Models with Retrieval-Augmented Generation*. It was submitted
twice to a top-tier IEEE conference and rejected both times. It compares four systems on 50 NarrativeQA
questions:

- a GPT-4o-mini baseline given the full story;
- BM25 RAG;
- "NQA Hybrid", which adds neural re-ranking;
- CAAG ("Contradiction-Aware Attention Gating"), a heuristic contradiction detector layered on top of
  NQA Hybrid that annotates flagged passages in the prompt (FLAG strategy).

**Latest review scores:** Evaluation "Very Questionable"; Originality "Complete lack of original ideas";
Presentation "Several Flaws".

**Reviewer comments (summarized):**

1. 50 questions cannot support conclusions at the architecture level. Report CIs or expand.
2. There is no statistical testing; 5.01 vs 4.90 may be noise.
3. CAAG uses 1,500-char chunks while the others use 1,000, so its token savings are confounded.
4. The ">32K tokens" motivation is never tested. Reframe as a cost study or use a genuinely long benchmark.
5. GPT-4o-mini is both generator and judge. Add a second judge or human validation.
6. There is no CAAG ablation: contradiction counts, strategies, and sensitivity to θ_s and θ_c are missing.
7. Latency is blamed on "retrieval overhead" without timing each component separately.
8. The cost per query relies on an unstated price. State the price and date and show the arithmetic.
9. The writing needs a copyedit and has promotional phrasing ("remarkable efficiency", "3.1× superior").
10. "Near-baseline" and "preserves quality" overclaim.

**Decision:** reframe the paper around contradiction-aware retrieval. The new question is when
contradiction flagging helps, when it hurts, and how that depends on detector precision and cost. CAAG
becomes the zero-API-cost end of a range of detectors rather than the headline. The name "attention
gating" should go, since nothing in the method gates attention.

---

## 2. Repo audit (verified against code and committed results at 89e545d)

| # | Finding | Evidence |
|---|---|---|
| A1 | **The 50 questions come from only 2 stories (29 + 21).** The loader takes the first N rows of the split, and those rows are grouped by story. The paper says the questions were randomly sampled. | `scripts/comparison/compare_systems_narrativeqa.py` loader; `results/system_comparisons/*.json` |
| A2 | **The baseline truncates every story to 50,000 characters.** Every baseline context is exactly 50,010 characters. | `baselines/narrativeqa_base_llm.py` (`max_story_length=50000`) |
| A3 | **"Input tokens" are whitespace word counts** (`len(context.split())`), and costs are computed from them. | base LLM + `scripts/analysis/analyze_cost_benefit.py` |
| A4 | **The only committed CAAG result is labelled mock data.** Its metadata reads "Expected/mock results…". It shows CAAG *below* the baseline: −0.19 overall, 3 flags across 11 questions, 67% false-positive rate. The paper's CAAG numbers (5.01, 3,621 tokens, $0.000571) appear in no committed file. **These must be traced or re-run before any resubmission.** | `results/llm_judge_evaluations/llm_judge_eval_contradict.json` |
| A5 | **The paper mixes numbers from different runs.** Its "NQA Hybrid" cost ($0.001222) matches the `hybrid_bm25_only` row of the cost file (BM25, k=10). The cost file's quality scores (4.88 baseline, 4.2 BM25) don't match the paper's (5.18, 4.60). | `results/cost_benefit_analysis/*.json` |
| A6 | **The BM25 system uses 600-char chunks**, not the 1,000 the paper reports. | compare script `hybrid_bm25_optimized` config |
| A7 | **The neural re-ranker is untrained unless `--retriever-checkpoint` is passed**, and no checkpoint is committed. Confirm what the reported run used. | `hybrid/narrativeqa_hybrid_rag_neural_retriever.py` |
| A8 | **There is no summary leak.** The loader passes an empty summary. | compare script |
| A9 | **The CAAG module can't be imported from a fresh install.** It needs `chunking_evaluation` and `langchain.text_splitter`, and neither is in requirements. | imports in the CAAG file |
| A10 | **The detector's design over-flags.** Contrast and temporal cues match substrings ("but" inside "about", "now" inside "know", "after" inside "afternoon"). Any two chunks with differing DATE/CARDINAL/etc. entities add 0.3. Temporal (0.4) plus entity (0.3) clears θ_c=0.35 on most similar pairs. | `ContradictionDetector` |
| A11 | **SEPARATE renders the same prompt as FLAG**, and ARBITRATE differs only in its heading. | `_build_prompt` |
| A12 | **The public MADAM-RAG code runs only local HF models (4-bit, GPU) and ships no scorer.** It has been ported here. | github.com/HanNight/RAMDocs |

---

## 3. Other paper issues from the review

- **Token and cost figures contradict each other.** Sections III and V disagree on whether NQA sends 10 or
  5 chunks to the generator. CAAG reports fewer tokens than NQA despite larger chunks and extra
  annotations. NQA's cost implies about 600 output tokens per answer, against about 40–55 for the other
  systems.
- **Metrics are undefined or inconsistent.**
  - The weights in the overall score aren't given; a plain mean of the four dimensions gives 5.94, not 5.18.
  - "Cost efficiency" and "latency efficiency" don't equal score ÷ cost or score ÷ time.
  - The judge prompt isn't shown, and it's unclear whether the judge saw the reference answers.
- **The median is misread.** A median of 4.00 for every system points to a lumpy score distribution. It
  doesn't show that quality gaps narrow for moderate questions.
- **Several references look wrong or are missing; verify every entry.**
  - RankRAG is Yu et al. (NVIDIA).
  - RAGChecker is Ru et al. (Amazon).
  - HELMET is Yen, Gao, …, Chen.
  - The LongBench co-author list is wrong.
  - NarrativeQA is missing its venue (TACL).
  - OP-RAG and the "6–9 chunks" claim have no citation at all.
- **Unsupported claims:** "the multi-dimensional framework is more informative than overlap metrics"
  (no overlap metrics are reported) and "intelligent gating offsets overhead" (based on a 0.08s difference).
- **Missing baselines:** dense-only retrieval, BM25+dense with RRF, an off-the-shelf cross-encoder
  re-ranker, and Self-Route.

---

## 4. Literature grounding

- **ConflictRAG** (arXiv 2605.17301, v2 June 2026). **This is the closest prior work; position against it
  explicitly.**
  - Design: pairwise conflict detection with all-MiniLM-L6-v2 and an MLP, an LLM fallback, type-adaptive
    resolution, and a GPT-4o-mini generator.
  - Baselines include an NLI-Filter (DeBERTa-v3) and a single-call chain-of-thought detection prompt.
    The prompt baseline beat NLI on all three of its benchmarks.
  - Methodology: GPT-4o as a separate judge, human agreement checks, 3 seeds, paired bootstrap.
  - Its code is "released upon acceptance", so reimplement its two baselines.
- **MADAM-RAG / RAMDocs** (Wang et al. 2025, arXiv 2504.13079). A multi-agent debate method. Reports
  **28.00 on RAMDocs with GPT-4o-mini**, a direct comparison point. Data and code are public.
- **CONFLICTS** (Cattan et al., Google, arXiv 2506.08500). 458 queries labelled with expert conflict
  types. Prompting the model to reason about conflicts significantly improves responses, which makes the
  prompt-only condition (C1) mandatory.
- **ConfRAG** (Yuan et al., ACL 2026). 1,814 real web questions, about 9.6 paragraphs each, 57.2% with
  explicit contradictions. HF `OracleY/ConfRAG`.
- **WikiContradict** (Hou et al., NeurIPS 2024 D&B). 253 human-annotated instances; implicit conflicts
  are the hardest. HF `ibm/Wikipedia_contradict_benchmark`.
- **FaithEval** (Ming et al., ICLR 2025). Includes an inconsistent-context task. HF
  `Salesforce/FaithEval-inconsistent-v1.0`.
- **Ragability** (Gross et al., LREC 2026). Hinting that a contradiction exists improves detection on
  contradictory contexts but **significantly hurts on non-contradictory ones**. This motivates H3.
- **Gururangan et al.** (NAACL 2018). Negation is a known annotation artifact correlated with NLI labels,
  so counting negation words is a shortcut signal.
- **Gokul et al.** (arXiv 2504.00180). Validating retrieved context for contradictions is hard even for
  strong LLMs, and difficulty varies by contradiction type.
- Also cite **TCR** (AAAI 2026, arXiv 2601.06842) and **EvidentialRAG** (arXiv 2607.10491).

---

## 5. Research question and hypotheses (pre-register before running)

**Research question:** when does contradiction flagging help, when does it hurt, and how does the
trade-off depend on detector precision and cost?

- **H1 (detection):** CAAG's heuristics have lower F1 than an NLI cross-encoder or an LLM detector, at
  much lower cost and latency.
- **H2 (benefit):** on conflict queries, accurate flags improve answers over plain RAG *and* over the
  prompt-only baseline.
- **H3 (harm):** on conflict-free queries, flags reduce correctness, and the drop grows with the
  false-positive rate.
- **H4 (content):** real flags beat random flags at the same rate. If they don't, any gain comes from the
  prompt template, not the detection.

---

## 6. Benchmarks

- **Primary:**
  - **RAMDocs:** 500 questions with pair-level gold labels derived from document types.
  - **CONFLICTS:** 458 questions, including 161 "No conflict" items as a built-in clean subset for H3.
  - **ConfRAG** and **WikiContradict:** loaders still need to be written.
- **Secondary:** FaithEval-inconsistent.
- **Control:** NarrativeQA (one coherent story, so almost no true contradictions).
- **Retrieval:** use each benchmark's provided passages for the main results. This holds retrieval fixed
  across conditions and also removes the chunk-size confound (reviewer comment 3).

---

## 7. Conditions (harness condition strings)

| ID | Condition | String |
|---|---|---|
| C0 | Plain concatenation | `plain` |
| C1 | Conflict-aware instruction, no detector | `instruct` |
| C2 | CAAG + FLAG (original format) | `caag+flag` |
| C3 | NLI cross-encoder + FLAG | `nli+flag` |
| C4 | Single-call LLM detector + FLAG | `llm+flag` |
| C5 | Random flags at CAAG's per-pair rate (placebo) | `random+flag` (needs `--random-rate`) |
| C6 | Oracle flags (RAMDocs only) | `oracle+flag` |
| C7 | CAAG with the other strategies | `caag+remove`, `caag+arbitrate`, `caag+flag_instruct` |
| C8 | MADAM-RAG on GPT-4o-mini | `madam` |

---

## 8. Metrics and statistics

- **Detection:** precision, recall and F1 at pair level (RAMDocs) and question level (all datasets), the
  false-positive rate on clean questions, the gold relation behind each false flag, latency, and cost.
- **QA:** each benchmark's own metric first.
  - RAMDocs strict: every gold answer present and no wrong answer present. This is reconstructed; see §11.
  - CONFLICTS: contains-match accuracy wherever `correct_answer` exists.
  - An LLM judge from a different model family, used only on a stratified subset, with about 100 human
    labels to check agreement.
- **Statistics:**
  - 3 generation seeds, averaged per question.
  - Paired cluster-bootstrap 95% CIs (clustered by story for NarrativeQA).
  - Sign-flip permutation p-values with Holm correction.
  - Always report results split into conflict and no-conflict questions.
- **Power warning:** a 0.1-point gap on a 0–10 judge scale needs on the order of 3,000 questions to detect.
- **Cost:** report real API usage tokens and state the price and the date it was checked.

---

## 9. Execution plan

### Phase 0 — flag audit (no API cost)

1. Run `phase0_audit` on about 100 NarrativeQA questions sampled across stories.
2. Hand-label the CSV with one of: `true_contradiction | time_or_plot_change | dialogue_or_perspective |
   different_entities | no_relation`.
3. Trace the source of the paper's CAAG numbers (A4).

**Gate 1:** if fewer than about half of the flags are real contradictions, treat the heuristics as a cheap
prefilter rather than the detector.

### Phase 1 — detector study

1. Run `detect_eval` on RAMDocs and CONFLICTS with `caag,nli` (add `llm` later), using `--signal-ablation`.
2. Use the θ sweep CSV for reviewer comment 6.

**Gate 2:** if a CAAG→NLI cascade matches NLI's F1 at lower cost, that cascade is the method contribution.

### Phase 2 — end-to-end QA

1. Run the condition matrix, piloting with 200 questions per dataset.
2. Then run the full sets with 3 seeds.

**Gate 3:** if C2 and C3 don't beat C1, detection adds nothing over prompting. That is still publishable
as a negative result if paired with the H3 harm analysis.

### Phase 3 — analysis

The main figure plots Δ correctness on conflict questions against Δ on clean questions, with one point
per detector, sized by cost.

### Budget

About 4,000 questions × 8 conditions × 3 seeds × ~3k input tokens comes to roughly $45 at $0.15 per
million tokens. MADAM-RAG costs several times more per question. Judging is the expensive part, so the
official metrics carry the main results.

---

## 10. Harness: `experiments/conflict/`

| File | Purpose |
|---|---|
| `caag_core.py` | Original detector, **verbatim** (lines 56–423). Imports without langchain or chroma. Do not edit; a parity test enforces this. |
| `data.py` | Common `Example` schema. Verified loaders for RAMDocs and CONFLICTS. NarrativeQA sampled randomly across stories (fixes A1). |
| `detectors.py` | `CAAGDetector`, `NLIDetector`, `LLMDetector`, `RandomDetector`, `OracleDetector`, `NoDetector`. CAAG runs once with its gates open, so θ sweeps are computed offline. NLI scores both directions and reads the label index from `config.id2label`. |
| `prompts.py` | Prompt modes `plain`, `instruct`, `flag`, `flag_instruct`, `arbitrate`, `remove`, reproducing the original FLAG and REMOVE formats. |
| `llm.py` | OpenAI client: real usage tokens, a `PRICES` table (fill in `PRICES_CHECKED`), SQLite cache, retries, and a `FakeLLM` for dry runs. |
| `madam.py` | MADAM-RAG port. Prompts and early-stopping rule are verbatim; runs at temperature 0 with 128 tokens to match the public script. |
| `metrics.py` | Answer extraction and scoring, plus pair- and question-level detection metrics. |
| `stats.py` | Cluster bootstrap, sign-flip test, Holm correction. |
| `run.py` | Resumable condition-matrix runner. Writes `config.json`, `detections.<det>.jsonl` and `results.jsonl`. |
| `detect_eval.py` | Phase 1 detector evaluation, signal ablation, and θ sweep. |
| `phase0_audit.py` | Phase 0 NarrativeQA flag export for hand-labelling. |
| `analyze.py` | Summary tables with CIs and Holm-adjusted p-values, plus cost and latency. |
| `check_env.py` | Checks packages, the spaCy model, data, the API key, model downloads, and NLI labels; `--live` adds a tiny real API call. |
| `inspect_data.py` | Prints schemas for FaithEval, ConfRAG and WikiContradict so their loaders can be written. |

Supporting files:

- `requirements-conflict.txt`: pinned to releases at least 14 days old as of 2026-10-26, for Python
  3.10–3.13 (3.12 recommended). Key pins: openai 3.29.0, sentence-transformers 6.1.0, transformers
  5.19.0, torch 2.14.1, spaCy 3.8.16 with en_core_web_sm 3.8.0.
- `setup_env.sh`: creates `.venv-conflict`, installs dependencies, clones the data at pinned commits
  (RAMDocs `d44454c`, rag_conflicts `81ba921`), creates `.env`, and runs the tests and `check_env`.
- `tests/`: 13 tests, all passing in the sandbox. They cover verbatim source parity and **identical
  flags** to the original detector across four (θ_s, θ_c) settings, the data loaders, prompts, metrics,
  stats, and an end-to-end dry run with resume.

---

## 11. Caveats and TODO

- **Not yet run on real models.** Nothing has been run with MiniLM, NLI or GPT-4o-mini; the sandbox
  couldn't reach Hugging Face or OpenAI. Run `bash setup_env.sh --live` first.
- **RAMDocs metric unverified.** `ramdocs_strict` is a reconstruction. Check it against Wang et al. 2025
  before putting numbers next to their 28.00. `gold_recall` and `wrong_rate` are also reported.
- **Missing loaders.** FaithEval, ConfRAG and WikiContradict have no loaders yet; use `inspect_data.py`
  first.
- **Phase 0 retrieval.** The audit uses BM25 top-10, not the repo's hybrid plus neural re-ranker. Say so
  when reporting it.
- **NLI truncation.** The NLI detector truncates passages to 1,500 characters, and NLI models are trained
  on sentence pairs. A sentence-level variant is a possible improvement.
- **Price not yet verified.** Fill in `PRICES_CHECKED` in `llm.py` after confirming GPT-4o-mini pricing.
- **MADAM-RAG with one round.** With `num_rounds=1` the original returns `None`; the port returns the
  round-1 aggregation.

---

## 12. Commands

```bash
bash setup_env.sh --live                     # env, data, tests, checks
python -m experiments.conflict.phase0_audit --n 100 --out runs/phase0
python -m experiments.conflict.detect_eval --dataset ramdocs   --detectors caag,nli --signal-ablation --out runs/detect_ramdocs
python -m experiments.conflict.detect_eval --dataset conflicts --detectors caag,nli --out runs/detect_conflicts
# pilot (use CAAG's pair_flag_rate from detect_eval as --random-rate)
python -m experiments.conflict.run --dataset ramdocs --n 200 --seeds 0 --out runs/ramdocs_pilot \
  --conditions plain,instruct,caag+flag,nli+flag,random+flag,oracle+flag --random-rate <RATE>
python -m experiments.conflict.analyze runs/ramdocs_pilot --reference plain
# add --dry-run to any run.py command to test plumbing with no API calls
```

---

## 13. Working rules for this project

1. **Every number in the paper must trace to a committed file in `runs/` or `results/`.** Never write
   expected or mock numbers into results files, and keep anything illustrative out of `results/`.
2. **Don't change the logic in `caag_core.py`.** Improvements go in new detector classes so the original
   stays reproducible.
3. **Use API-reported token counts only**, never word counts.
4. **Sample across stories or items with a fixed seed**, and report how many clusters there are.
5. **Keep `--dry-run` working** and keep the tests passing.
6. **Report the conflict and no-conflict splits** for every QA result.

---

## 14. Changes since this document was written (pilots of 2026-10-26/27)

Pilot outputs and what each run is are in `runs/README.md`. All pilots are exploratory.

- **RAMDocs labels:** some misinformation sits in documents typed "correct". `--ramdocs-labels`
  defaults to `exclude` (55 questions leave the clean split; pairs touching those documents are
  "uncertain"). `contains` treats them as misinformation; `none` keeps the released types.
  The 163 suspect documents still need hand-checking.
- **Scoring:** RAMDocs answer matching is bidirectional; `ramdocs_strict_oneway` keeps the old
  one-way metric.
- **Flag format:** `--flag-style uniform` (default) renders every detector's flags identically;
  `original` reproduces CAAG's typed, scored lines (use only for C2). Control and LLM detectors use
  the neutral type "contradiction" (the old "oracle"/"random" types leaked into the prompt).
- **New conditions:** `instruct_v2` / `flag_instruct_v2` (second C1 wording; v1 licensed listing
  both sides), `flag_directional`, `remove_suspect`, `llm_dir`, synthetic `grid` cells with
  precision, recall and direction accuracy.
- **Latency:** `generate_s` is now the API latency of the original call, including on cache hits;
  `wall_s` is wall clock. Pilot runs before this fix have unusable latency.
- **Analysis:** `grid_analyze` break-even = lowest precision at and above which every cell is a
  significant gain (sign-flip, Holm within table); `harm_free_precision` is the weaker secondary.
  `place_detector` predicts each subset from grid cells matched on that subset's document-level
  behaviour (`metrics.doc_level`); these axes were chosen after seeing one detector (llm_dir), so
  confirm them on another detector before relying on them.
- **Pilot findings (exploratory):** neutral flags hurt even when perfect; direction is what pays
  off; removal has the largest gain and the largest harm; pair-level P/R does not predict a real
  detector's effect, document-level behaviour does.

### Round 2 (2026-10-27): open issues before the full run

- **Splits:** `splits/ramdocs_pilot.txt` (200 questions, exploratory) and `splits/ramdocs_holdout.txt`
  (300, confirmatory). Use `--ids-file`. The plan is in `PREREGISTRATION.md`, which still needs
  TEAM decisions before it is frozen.
- **Metric check:** MADAM-RAG scores 38.4% (bidirectional) / 32.8% (one-way) against the published
  28.00. They don't match, and the team has to decide what to do (PREREGISTRATION §6). A MADAM
  answer-parsing bug (commas inside quoted answers) was fixed, and all runs were rescored.
- **Pricing:** verified 2026-10-27; cached prompt tokens are priced at the cached rate.
- **CAAG and NLI on the pilot:** neither helps. Pair F1: CAAG 0.07, NLI 0.28, llm_dir 0.33.
  Document-level placement predicted CAAG and NLI better than pair-level placement (5/6 vs 4/6
  inside the 95% CIs; mean error 0.030 vs 0.043).
- **Phase 0:** CAAG flags 54% of NarrativeQA questions (166 pairs from single coherent stories).
  To label: `runs/phase0/flags_to_label.csv`.
- **Labelling:** `runs/labelling/ramdocs_suspect_docs.csv`, then `--ramdocs-labels hand`.
- **Second model / judge:** `--base-url`, `--api-key-env` and `--price-*` support any
  OpenAI-compatible endpoint. `judge.py` has a DRAFT rubric (align it with Cattan et al.), a
  human-label export and Cohen's kappa. Neither has been run yet: we need the model choice and API key.
- **WikiContradict:** loader and `both_answers` metric added. It is the contrast dataset (H8).

### Round 3 (2026-10-27): confirmatory runs and paper

- **Pre-registration frozen** (commit `a6f7124`, §8). Primary metric: one-way strict (it matches MADAM-RAG's
  published score on the pilot split); bidirectional is the robustness metric.
- **Confirmatory results** (RAMDocs hold-out, 300 questions): H2, H3, H4, H6, H8 and H9 supported. H1 is supported
  in part (LLM vs NLI F1 is n.s.). H5: CAAG is no better than random. H7: no instruction effect after Holm.
  Labels: everything holds under `contains`.
- **Paper:** `paper/` (ACL review format; see `paper/README.md` for the build and the remaining human steps).
  Every number comes from `paper_stats.py`. Total API spend $27.42.

### Round 4 (2026-10-27): mock review and revision

- **Mock review:** three blind reviewers scored the first draft 2.5 out of 5 (reject for main, borderline Findings).
  Their main points were: the prompt skeleton may cause the neutral-flag harm; there was no open-weights model; H4,
  H6 and the dose-response statistics were overstated; real-detector gains are not significant; and the title
  overclaimed.
- **Addendum** `PREREGISTRATION.md` §9 was committed before every revision run:
  - H10: minimal skeleton (`--skeleton minimal`);
  - H11: Qwen2.5-7B through Ollama (`--base-url http://localhost:11434/v1`), seed 0 only (§9.4);
  - re-analyses (§9.3);
  - inverted-oracle adversarial control (§9.5);
  - Ollama repeat-limit aborts scored as degenerate (§9.6; 1 occurred).
- **Results:**
  - The neutral-flag effect is negative in all 6 generator × skeleton settings, significant in 4.
  - Removal of the oracle suspect is significant in all 6.
  - Directional flag > neutral flag in all 6. Directional flag > plain only for gpt-4.1 (both skeletons) and
    gpt-4o-mini (default skeleton). Exploratory split: it fails on multi-answer conflict questions.
  - Real detector: directional vs neutral flag +16.5 on conflict questions (gpt-4o-mini).
  - H6 re-analysis: document-level is not significantly better than pair-level (7/8 vs 6/8 inside the prediction
    intervals).
  - H5 TOST: not equivalent.
- **Code:** `revision_stats.py` (analyses), `InvertedOracleDetector`, and `prompts.BASE_MINIMAL`.
- **Paper:** retitled "Which Side Is Wrong? Neutral Contradiction Flags Do Not Help Retrieval-Augmented Generation
  Against Misinformation". The body still fits in 8 pages.

### Round 5 (2026-10-27): second mock review, round-2 experiments

- **Second mock review:** three fresh reviewers, all 2.5. Required:
  - rescope to synthetic misinformation and strict scoring;
  - one significance scheme;
  - a non-RAMDocs misinformation benchmark;
  - stronger baselines;
  - a neutral flag + resolution-instruction condition;
  - a cross-family detector;
  - the related work on telling the model which passage is wrong.
- **Registration:** `PREREGISTRATION.md` §10 (commit a76ff84), committed before the runs.
- **Code:**
  - `baselines.py`: Astute RAG, TrustRAG and the `*_multi` variants;
  - `prepare_conflictbank.py` and `data.load_conflictbank`;
  - `flag_directional_instruct_v2`;
  - `--detector-base-url`;
  - `revision_stats.assign_significance` (one scheme: registered tests use their family's Holm; everything else
    goes into one exploratory Holm family per metric).
- **Results:**
  - H12 on ConflictBank: all 12 tests are supported. Neutral flag −6 to −9; directional flag +18 to +28; removal
    +64 to +78 (3 generators).
  - H13: neutral flag + instruction ≈ plain; directional + instruction beats it everywhere.
  - H14:
    - On RAMDocs, the LLM directional flag beats Astute RAG and TrustRAG (3 of 4 tests significant).
    - On ConflictBank it ties both.
    - The baselines hurt RAMDocs clean questions by 11 to 25 points.
  - NLI flags nothing on ConflictBank.
  - Qwen detector: flags 97% of clean questions; direction accuracy 89%.
- **Spend:** about $52 of $100.

### Round 6 (2026-10-27): third mock review, round-3/4 experiments

- **Third mock review.** All three reviewers gave 2.5. The decisive issue was ConflictBank run only
  under the "every valid answer" skeleton.
- **Pre-registration.** §11 (e58249f) and §12 (886e5c1), each committed before its runs.
- **Code:**
  - `prompts`: `single` skeleton; new modes `flag_onewrong`, `self_identify`, `closedbook`.
  - `prepare_conflictbank` writes the majority variant; `data.load_conflictbank_majority`.
  - `llm.REASONING_MODELS`: `max_completion_tokens`, default temperature.
  - `paper_stats.revision3`: H15–H19, closed-book stratification, TOST claims, answer mix.
  - `fig_answers`; rebuilt `tab_cb` and `tab_baselines`; hypotheses tables split, with family header rows.
- **Results:**
  - **Single skeleton on ConflictBank.** The plain baseline rises from 21.6 to 51.1 (gpt-4o-mini).
    - The neutral flag still hurts (−18.7). The model abstains, with "unknown" on 41% of answers.
    - The one-wrong flag is worse (−35.7).
    - The directional flag gains +30.1.
  - **Majority misinformation (3:1).** Plain accuracy is 21.6.
    - Neutral flag: n.s.
    - Oracle directional flag: +32.7.
    - LLM detector direction accuracy falls to 49%.
    - Astute RAG gains only +1.4.
  - **Closed-book.** gpt-4o-mini knows 28/300 ConflictBank facts. Detector direction accuracy is
    86% on known facts and 67% on unknown ones.
  - **gpt-6.1-terra:**
    - Neutral flag: n.s. everywhere.
    - Directional flag: +17 on ConflictBank, +40 on RAMDocs, +66 under majority misinformation.
    - The LLM (gpt-4o-mini) directional flag *hurts* on ConflictBank (−11.7). A detector must be more
      reliable than the generator (exploratory).
    - Self-identify: −21 / −51, because the model distrusts every generated passage.
  - **Spend.** About $80+ of $100.


### Round 7 (2026-10-28): fourth mock review, §13–§14 experiments

- **Fourth mock review.** The consensus score was 2.5. The reviewers asked for:
  - a risk/coverage view and a forced-choice condition;
  - pair-topology and answer-key controls;
  - a crossed, pre-registered test of the reliability claim;
  - realistic data and models from other providers.
- **Budget.** Raised to $200 (gate $195). Spend after all runs is about $169.
- **Pre-registration.** Both sections were committed before their runs:
  - §13 (108e41d): H20–H24;
  - §14 (eca911b): Claude (H25) and RAGuard (H26).
  - Gemini was not run: no key was available.
- **Code:**
  - `prompts`: the `forced` skeleton; the modes `flag_pick`, `*_nopairs` and `answer_key`.
  - `llm`: the Claude provider, via the OpenAI-compatible endpoint. Claude takes no temperature and
    no JSON mode, and gets `max_tokens=4096`. The cache key includes `max_tokens` for Claude only.
  - `metrics`: unwrapping of paired tags (`<answer>X</answer>`).
  - `prepare_raguard`, `data.load_raguard`, and `judge_matcher`.
  - `paper_stats.revision4`:
    - H20–H26;
    - H24 with its §13-only sensitivity, the exploratory majority version, and an exploratory
      mechanism score;
    - the DerSimonian–Laird pooled neutral-flag estimate.
  - `stats.sign_flip_p(max_perm=...)`: p-values at the 1e-4 floor are refined up to 200k permutations,
    because otherwise no exploratory result could survive Holm over 633 tests. This is reported as a
    deviation.
  - `paper_assets`: `tab_risk`, `fig_reliability` (two panels), `tab_cells`, and the hypotheses tables
    chunked into 38-row parts.
- **Results:**
  - **Neutral flag.** The pooled effect is −2.8 [−4.5, −1.0]. It significantly harms 12 of 34 settings
    and helps 2.
    - Under forced choice it is null for three of four generators (H20).
    - Selective accuracy is unchanged (H21). The flag acts only through abstention.
  - **Directional flag.** It helps on every benchmark and generator except Qwen on RAMDocs, Claude
    included, and on RAGuard by +7 to +20.
    - The answer key does as well: H23 finds no significant difference in any of 9 tests.
  - **H24.** 30 cells: ρ = 0.64, p < 0.001. The §13-only cells give ρ = 0.60; under majority
    misinformation ρ = 0.30 (n.s.).
    - The gain is positive in 27 of 30 cells.
    - The exploratory score (right where the generator errs, minus a correct passage marked where it
      is right) gives ρ = 0.95.
    - The gpt-4o-mini detector marks a correct passage on 30% of questions. The other detectors do so
      on 4–11%, and the reasoning-model detectors rarely flag at all.
  - **RAGuard.** With the posts, every model is below closed-book. The stronger models abstain on
    about 55–60% of claims.
  - **Matcher.** An LLM judge agrees with the matcher 92.6% of the time (κ = 0.85). The matcher is
    slightly conservative.
- **Lesson.** A patch script used `str.index` on a string that occurs twice in `paper_assets.py` and
  deleted about 800 lines. They were recovered by replaying the session's patches onto the last
  file-history snapshot. Always assert `count == 1` before replacing.
