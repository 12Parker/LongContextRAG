# Restructure brief (revision 11, after the professor's review)

The authors accepted the triage's recommendations (`experiments/conflict/TRIAGE_REVIEW10.md`, §5 defaults): rebuild the
paper around three claims, one paper, pooled estimates out of the body, HealthContradict and RAGuard as label-transfer
stress tests, family claims dropped, registration described in three tiers. Read, in this order:
1. the review: its text is summarised point by point in TRIAGE_REVIEW10.md §2 (the original is not in the repo);
2. TRIAGE_REVIEW10.md §3 (claims, outline, decision rule, abstract plan) and §4.D (writing fixes);
3. PREREGISTRATION.md §20 (what the new re-analyses R1–R19 are) and the macro catalogue `paper/revision11/MACROS.md`.

## What changed since the triage was written (new results, PR§20; all exploratory)

- **R1: ConflictBank's false side is readable from passage style alone.** A bag-of-words logistic regression trained
  on other ConflictBank facts picks the misinformation passage on every eval and fresh conflict question (pairwise
  accuracy 1.000), also with the answer strings masked. Its heaviest words are genre words ("study", "exclusive
  interview", "journal of", "dr" for the false side; "evidence", "breaking news", "confirms" for the correct side).
  So on ConflictBank a detector's or generator's direction can come from style, not from knowledge or contradiction
  reasoning. This settles the reviewer's issue 2 more strongly than they argued: say it plainly, in the abstract, the
  setup, claim 2 and Limitations. It does not undo the randomised H31 design (the synthetic detector's direction is
  assigned by coin flip, so follow rates measure deference to the flag), but the generators' no-flag accuracy, the
  real detectors' direction precision and everything built on them live in a regime where the truth is legible from
  style. Deference despite legible truth is itself worth one sentence: generators follow a flag that points at the
  passage whose style marks it as correct.
- **R2:** every detector names the false side more often on facts its model knows closed-book (single-side
  definition; differences +3 to +19 points). **R3:** counting only one-side verdicts, direction precision is 0.69–0.97.
- **R4:** conditioning follow rates on the generator's prior is uninformative on ConflictBank (known facts are rare;
  pooled f_W odds ratio 0.85 [0.38, 1.85]); **R5:** conditioning moves the crossed-cell account by at most 0.7 points.
- **R6:** the account's edge over the cell-level marginal is +0.37 [−0.25, +1.22] MAE points (not significant); a
  common-rate baseline (one f_R, f_W for all generators) scores 1.63. Clean questions: in the 8 cells whose detector
  flags any clean question, MAE 8.9. The two predicted-harm cells (terra × gpt-4o-mini, terra × Qwen detector).
- **R7:** terra's other plain draw moves its p* from 0.79 to 0.81.
- **R9:** 0.15% of responses lack a final-answer line; 49 of 939 condition cells exceed 1% (gpt-4.1, Qwen, one luna);
  the worst is gpt-4.1 neutral + instruction (20%). **R10:** without those questions H13's gpt-4.1 neutral +
  instruction test becomes significant (+17.2); the registered result keeps its status.
- **R11:** seed-by-seed, pooled H30a ranges +0.5 to +1.4 and H23 −0.4 to +1.2. **R12:** one-way misinformation
  matching changes 6 of 210 exploratory decisions and one bold mark in Table 2 (RAMDocs Haiku).
- **R13–R19:** HealthContradict stance shares; RAGuard key minus flag per generator; single-entity RAMDocs
  leave-one-out; risk–coverage operating points (figure `fig_riskcov.pdf`); RAMDocs minimal answer mix; prevalence
  break-even shares by clean-set group; H4b label-robust rho −0.08, permutation p 0.75, BCa [−0.32, +0.01].

## Hard rules

- Numbers quoted in this brief are for orientation only and some are rounded differently from the macros; always cite
  the macro (`MACROS.md` names it, or the existing macro it duplicates). R16 and R18 "significant" counts are
  unadjusted CI exclusions: write "CI excludes zero", not "significant".
- Labels fixed in advance: the risk–coverage figure is `\label{fig:riskcov}`; the new appendix section is
  `\label{app:rev11}` (generated captions already point to both); `tab_deference` carries `tab:deference`.

- Every number is a macro from `paper/generated/numbers.tex` (grep it; `paper/revision11/MACROS.md` lists the new
  ones). Never type a number that the data produced. Counts of things in the design (300 questions, seven generators)
  may be typed if they already appear typed in the paper.
- Never edit `paper/generated/`. Pre-registration sections are written `\prs{20.3}` (renders PR§20.3); paper sections
  use `\S\ref{...}`.
- Body (abstract through Discussion) ends on page 8; `\clearpage` before Limitations keeps it at the top of page 9.
  Build: `cd paper && tectonic -X compile main.tex` and check the exit status (a failed build leaves the old PDF).
  Page check: `pdftotext -f 9 -l 9 main.pdf - | head` must start with Limitations.
- Status words: "registered" (with its tier), "exploratory", "prospective". "Significant" only under Holm. No "tie" or
  "no harm" without a TOST. Say "under the string matcher" where a claim is matcher-only and near zero.
- Terms (define once in §3, then use only these): *hit rate*, *direction precision* (counting one-side verdicts;
  say which definition when a number uses the other), *pair precision/recall*, *follow rates* f_R and f_W (adoption:
  a right flag is followed when the flagged answer is correct where the no-flag answer was wrong; a wrong flag when
  the flagged answer is wrong where the no-flag answer was right), *break-even direction precision* p*, *answer key*,
  *verdict key*. Never bare "precision". Drop "share of headroom". Rename the H31 effects Δ_R and Δ_W.
- Style: plain, direct sentences; one idea per sentence; no hype; no process jargon (session, triage, draft, budget,
  tracks). Match the existing LaTeX conventions (`\citep`, `\emph`, `\paragraph{...}`).

## Body outline (target columns in brackets; about 16 in all)

- **Abstract** (≤ 230 words; at most five numbers: 95%, 0.52, 0.79, 1.4, 1.8). Use TRIAGE §3.4 v3 and add R1 without a
  number ("on ConflictBank, whose false side a bag-of-words classifier identifies from style alone"). Registration
  not in the lead; break-even marked exploratory; matcher-scored; no HealthContradict; no family claims.
- **§1 Introduction** [~1.4]: problem and question; the design in three sentences; three findings (the claims), each
  with its status; one paragraph of scope (ConflictBank readable from style; stress tests; key equivalence unresolved
  under the judge; no family claims); release sentence and footnote (keep).
- **§2 Related work** [~1]: trim `related.tex` to about 450 words; add Chen et al. (2025) "Reasoning Models Don't Always
  Say What They Think" beside sycophancy/advice-following (verify the bib entry against arXiv before adding: authors,
  year, arXiv id); keep the synthetic-context citations; keep ConflictRAG and say we did not test it as a detector.
- **§3 Framework** [~2]: skeletons (one paragraph); formats compressed (plain, neutral, directional, remove, controls
  in one line: one-wrong, pick, answer key, verdict key, hint; single call; closed book); detectors: LLM detector and
  oracle in full, the synthetic detector with direction precision d (H31 uses d = 0.5) in two sentences; CAAG, NLI,
  random and the grid in one sentence pointing to the appendix. A small terms table (≤ 7 rows) and **Equation (1)**,
  the follow-rate account, numbered, with the definition of following stated once and the note that unflagged
  questions keep the no-flag answer.
- **§4 Setup** [~2]: Table 1 (evidence; rows updated: ConflictBank "side readable from style (R1)"; RAGuard and
  HealthContradict "label-transfer stress test"). Data paragraphs shortened. Generators: one sentence on setting
  differences (Claude and reasoning models at default temperature; Claude via an OpenAI-compatible endpoint, no JSON
  mode). **Scorers** paragraph: one-way string matcher primary; the outside-family judge (gemini-3.5-flash-lite)
  re-scored all pooled-test responses (agreement and the per-condition miss rates: the matcher misses more correct
  answers under the flags than under the keys); human audit sheets released unlabelled; the within-family terra check
  moves to the appendix. **Registration** paragraph: three tiers (H1–H9 before any confirmatory data; H10–H31 and
  H37–H38 before their runs but after earlier results, with the PR§20.1 exceptions; H32–H36 frozen predictions), that
  "PR§N" means the pre-registration file, the timeline table (Appendix), and that PR§20 re-analyses are exploratory.
  Statistics compressed.
- **§5.1 Claim 1: a neutral flag shifts the answer policy** [~2]: `fig_riskcov.pdf` replaces Table 2 in the body
  (`\input{generated/tab_risk}` moves to the appendix with a caption note on matchers). One-answer prompt: abstention
  rises, selective accuracy shows no detectable change for gpt-4o-mini (ratio estimand; name it). Multi-answer
  prompt: misinformation added. Exceptions stated (Haiku, Sonnet, terra barely move on the one-answer prompt; Haiku's
  misinformation share falls on RAMDocs). WikiContradict: the same compliance is rewarded (H8). One utility sentence
  (gpt-4o-mini turns positive between c = 1 and 2, significant only at c = 4). No pooled estimate in the body.
- **§5.2 Claim 2: generators defer to directional flags** [~2.5]: oracle direction helps (brief, with the RAMDocs and
  ConflictBank ranges); H31: right and wrong effects, `tab_deference` (main text), follow rates (three of four
  similar; terra 67% right vs 95% wrong), p* 0.52–0.79 (exploratory), terra's two draws (R7) in a clause; RAMDocs grid
  corroboration if a macro exists (d = 0.5/0.75/1); scope: R1, R2, the majority-variant drop (68.8 → 49.4), PR§11.3
  split; R4 in one sentence. Figure 1 left panel may stay (it shows H31) or go to the appendix if space is short.
- **§5.3 Claim 3: a frozen account predicts within a benchmark** [~1.5]: Equation (1) is bookkeeping plus one claim,
  that follow rates transfer; H32 (MAE 1.4, slope), baselines (marginal 1.8, the difference not significant; common
  rates 1.63; generator means 4.1); H33 fails (16 predicted gains, 2 predicted harms; name both cells); H36 holds;
  clean questions: the 8 flagging cells (MAE 8.9); HealthContradict misses for all three generators, label noise an
  untested candidate.
- **§5.4 What does not hold, or is not settled** [~1]: answer-key and verdict-key equivalence (H23, H30, H38) depend on
  the scorer; seeds (R11) in a clause. Stress tests: HealthContradict's oracle flag is effectively an answer key;
  RAGuard's key carries more than the flag except for gpt-4o-mini (R14); CONFLICTS underpowered. Genuine conflicts:
  direction and removal remove valid answers (H28). Prevalence: break-even conflict shares (R18), not the 5% count.
- **§6 Discussion and a decision rule** [~1.3]: TRIAGE §3.3 steps 0–5, compressed, with its caveats (derived from these
  runs, not tested end to end); single call; links to sycophancy, advice-taking (Bansal, Mozannar), abstention (Wen),
  Chen et al.
- **Limitations** (page 9, outside the limit; rewrite to match): ConflictBank regime and R1; scorer and audit status;
  HealthContradict labels; registration timing (PR§20.1, timeline); snapshots not recorded; terra single draw;
  family confounds (two Claude models; one 4-bit open model); detectors tested (no trained detector such as
  ConflictRAG); the NLI window wording; Ethics paragraph kept.

## Appendix changes (separate writer)

- New section "Re-analyses after the external review (PR§20)" with `tab_prior`, `tab_missing`, `tab_riskcov`,
  `tab_prev11`, and short paragraphs for R1–R19 not covered in the body (R5, R6, R8, R10, R11, R12, R13, R15, R17, R19).
- Move `tab_risk` here (caption: correct and selective one-way; misinformation bidirectional, R12 for one-way), and
  any body table the lead writer drops (`tab_pooled` stays in the appendix with "describes these 34 settings only").
- Rewrite the claims table (`tab:claims`): one row per body claim, statuses in the three tiers; remove rows the body no
  longer makes; mark R-numbers exploratory.
- Factual fixes from TRIAGE §4.D that sit in the appendix: "up to 51 points" → the luna macro (61.0) and a table that
  carries it; "distrust generated passages" wording (appendix.tex and limitations.tex); Appendix E's matcher direction
  (misses more under the flags); CONFLICTS item description; "does not depend on the detector's family" scoped to one
  Qwen detector on RAMDocs; the judge-sample paragraph labelled within-family; H4b caveat (`app_grid.tex`) with R19;
  H24 status "supported (mechanical)"; `audit_200.csv` sentence replaced by the A1/A2 status.
