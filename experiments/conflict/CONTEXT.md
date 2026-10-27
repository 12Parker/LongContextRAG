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
