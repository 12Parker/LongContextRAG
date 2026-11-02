# Review 10 triage and revision plan

Date: 2026-11-02 (version 2). Branch: `conflict-experiments`. Review: the professor's review of "Which Side Is Wrong?" (draft PDF of 2026-10-30).

How we checked: each review point was checked against the PDF, `paper/sections/*.tex`, `paper/generated/*`, `hypotheses.json`, the run files and the git history. A sceptical reviewer then checked it a second time, and a completeness check a third time. This version applies the completeness check. We made no API calls and computed no new analysis. A few counts were first tabulated on 2026-11-02, while we wrote this plan. They are listed in Section 4B.3. None enters the paper unless a registration covers it, and the paper then discloses it as "first tabulated on 2026-11-02, before PR§20" (the PR§18.5 precedent, without its "review triage" wording).

In this document, "PR§N" means section N of `experiments/conflict/PREREGISTRATION.md`, and "§N" means section N of the paper.

---

## 1. Verdict in brief

- **The reviewer is right on all five overall points.** The paper reads as a catalogue. Its quantitative core rests on ConflictBank. The scorer question is open and no human has labelled anything. The real-document evidence cannot be read as evidence about misinformation. The pre-registration is overweighted. This needs a restructure, not an edit.
- **Issue 2 (artefacts) is a serious, plausible concern, and our own data, absent from the reviewed draft, add to it.** We concede the possibility, not the conclusion, until the probes run.
  - Our registered PR§11.3 split is not in the reviewed draft. An earlier draft printed its ConflictBank half (added in 8e0e261, removed in 1bd1e11); the RAMDocs half was never printed. On ConflictBank, 272 conflict questions are ones gpt-4o-mini cannot answer closed-book. On the 264 of them where its detector flags the gold pair and names a side, it names the false side 67.0% of the time (177 of 264). On the 28 known questions the rate is 85.7% (24 of 28).
  - The RAMDocs half of the same split shows no gap (88.3% on 132 unknown, 87.5% on 7 known). There, agreement between documents is available.
  - The registered majority-variant result points the same way. With more misinformation passages than correct ones, direction accuracy falls from 68.8% to 49.4%.
  - On the one-answer prompt, every generator gives the correct answer more often than the misinformation. The ratio runs from 1.4:1 (Qwen2.5-7B, 50.3 vs 35.3) to 4.8:1 (Sonnet, 75.7 vs 15.7). The six generators run closed-book score 4.3–20.0%; Sonnet has no closed-book run.
  - In sampled items, the substituted object often clashes visibly with the subject.
  - Recognition-level knowledge remains the alternative. The probes test it (about $8, plus a free text classifier). Limit p*, the follow rates and H32 to synthetic, low-prior misinformation.
- **Issue 3 is right, and on clean questions the reviewer is generous.** The account is expected-value bookkeeping plus one empirical claim: that rates transfer. 12 of the 20 clean cells are 0/0 by construction, because unflagged questions keep the cached plain answer. Our pre-registration already said the account "barely beat" the cell-level marginal. The reviewer's "18 of 20 positive" is really 16 predicted gains and 2 predicted harms, and our own wording invited the misreading.
- **Issue 4 is right, and our text adds an error.** Appendix E says the matcher misses more under the keys. The data show the opposite. The sheet the reviewer names (`audit_200.csv`) contains no answer-key or verdict-key outputs and is not blind, so labelling it cannot settle H23/H30. The sheet that can (A1) takes about 3.5–4 hours per annotator, so the reviewer's "a few hours" is right. Agreement statistics (A2) add 3–4 hours per annotator, plus about 1 hour of joint adjudication. A1 settles H23/H30 only if the humans side clearly with one scorer. Human labelling is the top author task.
- **Issues 5 and 6 are right.**
  - On HealthContradict the oracle directional flag is effectively an answer key. Label noise could explain the prediction's overshoot, and we cannot test that until humans label pages (A4).
  - We ran a RAGuard answer-key arm and never reported it. For luna, terra and Haiku the flag carries a fraction of what the key carries (+8.8 vs +42.1, +6.8 vs +19.0, +12.7 vs +19.8). For gpt-4o-mini the flag carries more (+9.9 vs +1.4).
  - For most generators the neutral flag is a policy shift: towards listing both sides under "give every valid answer", and towards abstaining otherwise. On the one-answer prompt, Haiku, Sonnet and terra barely move (terra's abstention 1.7→4.7). Under the multi-answer prompt on RAMDocs, Haiku moves the other way.
  - The pooled −1.4 and its meta-regression leave the body. The registered summaries go to one appendix table that describes these 34 settings only.
- **Issue 7 is right, and worse in two places.** Our claim that each addendum was "committed before its runs" is false for PR§9.4, PR§9.6 and PR§17.6. The PR§9.4 Qwen seed change was committed about 5 minutes after the Qwen run started. The PR§17.6 judge substitution was committed about 10 minutes after the judge outputs were written. The claim appears in seven places. The supplement also excludes the git history, which is the paper's only timestamp. Under double-blind review the history cannot simply be added (A7).
- **Where the reviewer overstates (neither point changes the verdict):**
  - INC.10: for H23 the flag–key discordance (11.7%) is clearly above the seed null (6.7%; CIs do not overlap). Discordance is also not the gap.
  - H18 does not show the single call "matches" the pipeline. Its ±3 TOSTs were computed under PR§13.4, are uncited, and show no equivalence. H18 covers gpt-4o-mini only. The paper says "does not differ significantly".
- **Six registered or run results are missing from the paper.** Report all six; they cost nothing.
  - The PR§11.3 closed-book split, both its ConflictBank and its RAMDocs half.
  - The PR§11.3 majority-variant direction accuracy (49.4% against 68.8% on the 1:1 set).
  - The PR§19.2 per-condition matcher–judge agreement.
  - The RAGuard answer-key arm.
  - The H18 TOSTs at ±3 (none equivalent; M-d).
  - The PR§17.1.1 equal-weight pooled mean (−3.0 [−3.8, −2.2]).
  - The first five bear on the reviewer's points. The equal-weight mean does not: its CI excludes 0, and it describes only this set of settings. It goes into one appendix table with that label (6.4).
- **Cost.** Every must-fix except C1 ($7.8, the artefact probes) is writing or author work, at $0. The recommended API package is $22.3 now. Another $2.8 follows the HealthContradict human labels (A4), and $7.9 more is spent only if the humans in A1 favour the judge, or are split. That is $33.0 in all, against the $40.82 left before the $245 gate. The worst case is about $38.7: a Sonnet cache miss plus 25% overruns on C5 and C9, which carry no contingency.

---

## 2. Point-by-point

Verdicts: **Accurate**, **Partly accurate**, **Inaccurate**, **Recommendation** (what we think of the reviewer's suggested fix). Priority sits in the response column: must, should, could, decline.

### Issue 1: no central contribution; main text not self-contained

| # | Reviewer claim | Verdict | Evidence | Planned response |
|---|---|---|---|---|
| 1.1 | Catalogue, not argument: about 38 hypotheses, 30 cells, a 36-cell grid, a prospective test and a meta-analysis in 8 pages. | Accurate | H1–H38; 30 H24 cells; 36 grid cells; 20 prospective cells; 34 pooled settings; 532 registered tests (`\fdrN`); 636 exploratory comparisons (`\nExploratory`). The ninth mock review said the same (CONTEXT Round 12). The body already fills page 8. | **Must.** Rebuild around three claims (Section 3). Cut before adding anything. Remove running H-number lists, e.g. results.tex:38. |
| 1.2 | The abstract carries about 15 numerical results and scorer qualifiers. | Accurate | About 297 words; 15 numerals besides hypothesis labels (about 11 distinct results); six qualifiers. Two abstract claims fail under the judge: H38b +2.5 [90%: 1.1, 3.9], H38a +2.4 [1.3, 3.5]. The v3 numbers (95%, 0.52–0.79, 1.4, 1.8) are matcher-only (X4). terra's 95% was never judge-checked: PR§17.3 re-scored H31 for gpt-4o-mini and Haiku only. | **Must.** Replace with the v3 abstract (Section 3.4). It has five numbers and says they come from string matching. It drops the claims the judge contradicts (H23, H30) and drops HealthContradict. |
| 1.3 | Core claims depend on appendix Tables 25, 42, 43, 26 and 3. | Accurate | The main text holds only Table 1, Table 2 and Figure 1, and cites 12 appendix tables. Follow rate, hit rate and direction precision are defined only in appendix captions (Tables 25, 36). | **Must.** One compact main-text table for claims 2–3 (Table 25's four rows plus a prospective summary). Claim status goes into Table 1. Definitions move to §3 (INC.7). |
| 1.4 | Pick two or three claims: (a) policy shift; (b) follow rates that do not depend on the detector, giving a break-even; (c) prediction within a benchmark but not across corpora. | Recommendation: agree, with changes | (a) holds without harm only for one-answer prompts, and not for every generator (claim 1's exceptions, Section 3.1). Under the multi-answer prompt the policy shifts towards listing both sides, and selective accuracy falls significantly (ConflictBank gpt-4o-mini 21.6→12.9). (b) mixes registered effects (H31a–c) and registered, untested follow rates (PR§16.3) with an exploratory p*. The RAMDocs grid was registered before any confirmatory data. It gives −1 / +4 / +13* at d = 0.5 / 0.75 / 1 for gpt-4o-mini. (c) "across corpora" rests on HealthContradict, whose labels are unreliable. | **Must.** Adopt the three claims in Section 3.1. State registration status inside claim 2. Use the grid as corroboration. Treat HealthContradict as a stress test. |
| 1.5 | Move CAAG, the grid, the baselines and the H22 tests out of the narrative. | Partly accurate | H6 is not supported (−1.2 [−3.6, +0.4]). CAAG and the grid sit in §3. But the baselines are already one paragraph plus an appendix, and H22 is not in §5. The grid is partly load-bearing: H31's randomised detector is the grid's d = 0.5 synthetic detector. | **Should.** Move the grid results, H5/H6, CAAG, the baseline setup and the control definitions to the appendix. Keep the synthetic-detector definition and one grid sentence. Most space comes from other cuts (Section 3.2). |
| 1.6 | Consider two papers: formats and key equivalence; randomised direction and the follow-rate account. | Recommendation: decline | The first paper's distinctive claim (key equivalence) fails under the judge and has no human labels. Both papers would share the harness, every run and one pre-registration. | **Decline.** The restructure already centres the paper on the second paper's argument. Each half alone would be thin. |
| 1.7 | The Discussion is one paragraph. Table 3 lists recommendations that §6 does not contain. | Accurate | 5 sentences, about 80 words; Limitations runs to 1,203 words. The PR§19.1 selection (registered, descriptive) picked the Haiku detector for every generator; for terra it gave +5.3 against −3.5 with the gpt-4o-mini detector. | **Must.** Expand the Discussion to about half a column and put the decision rule there. Split the Table 3 row: "detector choice by frozen account, checked descriptively at a 75% conflict share (PR§19.1)" and "prevalence threshold and full rule: not tested end to end". |
| 1.8 | Practitioners need an explicit decision rule. | Recommendation: agree | All inputs exist. Our earlier argument against a precision veto does not hold. It said the Haiku detector's precision, 78.1% [74.9, 80.9], is below terra's p* of 0.79, yet the selection picked it and it gave +5.3. But 78.1 counts "both" and "unclear" verdicts as misses, while p* refers to flags that name one side. 0.781 also lies inside terra's p* CI [0.72, 0.85]. On fresh, the Haiku detector named the false side on 69% of conflict questions and marked a correct passage on 7% (tab_fresh). The data do show that precision above p* is not sufficient. Haiku generator × gpt-4o-mini detector: at least 68.8% against 0.67, observed −4.1 on conflict questions, through flags on correct passages. | **Must.** Write the rule in Section 3.3: the frozen account picks the detector. p* is a screen. It is not sufficient on its own, and whether it should veto is untested. Optional: C7 and B10 (in-sample only). |

### Issue 2: ConflictBank's direction signal is probably an artefact

| # | Reviewer claim | Verdict | Evidence | Planned response |
|---|---|---|---|---|
| 2.0 | Headline: the direction signal is probably an artefact, and the main quantitative results depend on it. | Accurate as a concern; the conclusion is untested until C1 | Both sides are LLM-written; the wrong side substitutes the object (prepare_conflictbank.py). In a triage look at every 20th eval item, many substitutions clash with the subject (an Indian bishop said to be Slovenian; an Australian power station said to be owned by Iwate Prefecture). Some misinformation passages state the clash outright. Detectors pick the false side 69–83% of the time at 4–20% closed-book accuracy. On HealthContradict (real pages, yes/no answers, closed-book 76–91%), the same detector names the false page on 36% of items and the correct page on 48% (released labels; see 5.1). Our related work already says LLMs favour LLM-generated passages and the side matching their prior (related.tex:51–52: jin2024tugofwar, kortukov2024realistic, tan2024blinded). Limitations already says generated benchmarks may not measure how such models treat real documents (limitations.tex:20–22). Recognition-level knowledge remains a live alternative (2.6). | **Must.** In the response and in Limitations, concede that the signal may be an artefact and that the main numbers depend on ConflictBank. Do not concede that it is one. Show one or two examples drawn by a fixed seed. Measure the signal with B1, C1 and B2. Scope every ConflictBank number. |
| 2.1 | Table 1 admits break-even rests on ConflictBank; H31, H24, H30 and H32 use it too. | Accurate | Run configs: randdir_cb_*, cross_cb_* and vkey_cb_* use `conflictbank`; fresh_cb_* uses `conflictbank_fresh`, built the same way. The only off-ConflictBank test of the account is H37, and it missed. | Covered by 2.0, Fix c and the probes. |
| 2.2 | Closed-book accuracy is 4–20%. | Accurate | Recomputed from cb_single*: gpt-4o-mini 10.0, gpt-4.1 20.0, Qwen 4.3, luna 16.9, terra 20.0, Haiku 11.8. Sonnet and the fresh set have no closed-book runs. | **Should.** Emit min/max macros for the scoping sentence. |
| 2.3 | LLM detectors pick the false side 69–83% of the time. | Accurate | Table 36: 68.8 / 81.8 / 82.2 / 82.6 / 78.1 / 77.7, recomputed. That precision counts "unclear" and "both" verdicts as misses. Restricted to one-side verdicts the rate is higher (tabulated 2026-11-02 only). | **Should.** Put the HealthContradict contrast (36% right, 48% wrong) next to ConflictBank's 67.0 / 30.3, with the label caveat. |
| 2.4 | Generators reach 51–76% with passages. | Accurate | Plain accuracy 50.3–75.7 (one-answer prompt). Every generator gives the correct answer more often than the misinformation. The ratio runs from 1.4:1 (Qwen2.5-7B, 50.3 vs 35.3) to 4.8:1 (Sonnet, 75.7 vs 15.7). gpt-4o-mini is 1.65:1 (51.1 vs 31.0). The closed-book range (4.3–20.0%) covers the six generators run closed-book. Sonnet has no closed-book run. | Do not argue that weaker generators are "at chance": they are not. |
| 2.5 | With exactly two on-topic passages, agreement cannot help. | Accurate | 300/300 conflict items have one correct passage, one misinformation passage and two off-subject distractors; the fresh set is the same. In the majority variant, where agreement exists, the gpt-4o-mini detector's direction accuracy falls to 49.4% from 68.8% on the 1:1 set. This is registered in PR§11.3 and not in the reviewed draft (X10). | **Must.** Add one sentence to §4: the detector cannot use agreement here, so its verdict comes from knowledge, plausibility of the answer, or cues in the text. Report the majority-variant contrast beside it. |
| 2.6 | The answer must come from cues inside the text, not from memory. | Accurate (slightly overstated) | Our registered PR§11.3 split, ConflictBank half: 24/28 known (85.7%) against 177/264 unknown (67.0%). 272 questions are unknown; 264 is the number whose gold pair the detector flagged with a side. RAMDocs half (also registered, also unreported): 87.5% on 7 known, 88.3% on 132 unknown. An earlier draft printed the same reading ("it mostly compares passages rather than recalling the fact"; removed in 1bd1e11). Implausible random substitutions are themselves a construction artefact. The one non-artefact alternative left is recognition-level knowledge that strict recall misses. | **Must.** Report both halves of the PR§11.3 results with denominators (272 unknown, 264 flagged; 28 known); this restores the ConflictBank half. Present the reviewer's reading as the leading one and recognition as the alternative the probes test. |
| 2.7 | Reasoning models and Haiku reject most passages and abstain, losing up to 51 points on clean questions. | Partly accurate (the paper's number is wrong, and its inference is not supported as stated) | The paper's maximum is wrong: luna loses 61.0 [−69.7, −52.0], terra 51.0, Haiku 37.7. The table pointer (Table 35) shows only gpt-4o-mini, and no table carries these losses. The reviewer took the inference from our own wording ("distrust most of ConflictBank's generated passages", appendix.tex:176–177; "distrust ConflictBank's passages when judging them", limitations.tex:20–21). The data do not support that wording. The loss belongs to the single-call format. On clean questions in that format, luna and terra name about 2.4 and 2.1 of the 4 passages as suspect, and Haiku about 1.1 (tabulated 2026-11-02). They abstain on 65%, 56% and 39% of those questions. For Haiku, "most" fails on either count. Used as two-call detectors, luna, terra, Haiku and Sonnet flag 0 of 100 clean eval questions. Luna, terra and Haiku also flag 0 of 100 on fresh. We have no single-call run of luna, terra or Haiku outside ConflictBank; the only other single call is gpt-4o-mini on the RAMDocs hold-out (holdout_extra; Table 35). So nothing shows that generated passages are distrusted more than real ones. | **Must.** Change to "up to 61.0 points (gpt-6-luna)". Add a clean-question single-call column to Table 5 (tab_cb). Rewrite both sentences: in one call that asks them to name suspect passages, these models name many passages as suspect even on clean questions and abstain. As two-call detectors they flag no clean question. In the response, say our wording invited the reading, and leave the artefact question to C1. |
| 2.8 | H29 does not rule out content artefacts; equivalence held for 1 of 4 probes. | Accurate | H29b TOST ±5: equivalent only for gpt-4o-mini paraphrase. Paraphrase keeps subject and object, so a clash survives. Limitations concedes 1 of 4; Results (results.tex:115–116) does not. | **Should.** Make Results match Limitations and say H29 tests position and style, not content. The object-swap probe in C1 is the content test, with three registered outcomes (survive / invert / fall to chance). |
| 2.9 | Follow rates, break-even values and the prospective test were calibrated (i) at low prior and (ii) where direction may be readable from surface cues. | Accurate | (i) Closed-book 4.3–20.0%. (ii) Randomisation keeps detector artefacts out of f_R and f_W. But results.tex:82 says p* "rises mainly because an accurate generator has more to lose". Generator accuracy here is reached through readable cues, so the cues feed p*, R, W and the account's a_i. | **Must.** Say which quantity each concern reaches. Do not argue that randomisation protects p*. |
| Fix a | Single-passage plausibility probe. | Recommendation: agree; raise to must | The detector prompt is pair-based. The answer-only two-choice probe, with both answer orders, is the most diagnostic arm. A free passage-only text classifier trained on non-eval ConflictBank rows gives a lower bound on surface readability. | **Must.** B1 (free), then C1 ($7.8, the only must-fix that costs money). Extension C-x ($19.5) not funded. |
| Fix b | Split direction accuracy by closed-book correctness. | Recommendation: agree; partly done | Registered (PR§11.3) and computed for gpt-4o-mini, on ConflictBank and RAMDocs; macros at numbers.tex:4431–4466; not in the reviewed draft (the ConflictBank half was printed from 8e0e261 until 1bd1e11). The definition of "known" depends on the metric (28 with one-way strict, 31 with the rows' `primary` field). | **Must.** Report both halves of the gpt-4o-mini split now. Register and compute it for the other detectors (B2). **Should:** closed-book on fresh and for Sonnet (C9). Read it only together with the answer-only probe, because "known" is partly guessable from the subject's name. |
| Fix c | State that the break-even values belong to the low-prior regime. | Recommendation: agree | No sentence ties p*, the follow rates or H32 to prior knowledge or synthetic misinformation. | **Must.** Add scoping to the abstract, intro, results, Table 1 and Limitations. Cover both the low prior and the synthetic substitution. Name label noise as the rival explanation of the HealthContradict miss. |

### Issue 3: the follow-rate account

| # | Reviewer claim | Verdict | Evidence | Planned response |
|---|---|---|---|---|
| 3.1 | Given d_i, w_i and a_i, the account is close to an identity; its real claim is that f_R and f_W are constant within a generator. | Accurate | Every term except the 8 rates is observed before any flagged answer exists. Unflagged questions get the cached plain answer (run.py:318; `cached=True` on every unflagged row checked), so they contribute exactly 0 on both sides. With Table 25's share-of-headroom rates the account reproduces H31 exactly. Rate-level transfer evidence: H36 (gpt-4o-mini only; the fresh rates, 50 [40, 61] and 50 [38, 60], lie inside the frozen 95% CIs [46, 67] and [40, 62] around 57/51). H34 r 0.97 is scale-free. The unit-rate baseline is a straw man. Round 11 notes already say "retrospective decomposition". | **Must.** Adopt the reviewer's framing: bookkeeping plus a rate-transfer claim. Add a list of the channels the account omits (right flags on correct answers, wrong flags on wrong answers, abstention, flags that are neither right nor wrong). Optional B4: common-rate baseline. |
| 3.2 | It barely beats the cell-level marginal (1.8 vs 1.4). | Accurate | Exact (1.8048 vs 1.4355). PR§19.1 says the retrospective account "barely beat (MAE 1.6 vs 1.5), so we expect a near tie". The appendix also has break-even × recall at 1.9 vs 1.5. The 1.8 lies inside the account's own CI [1.2, 2.4]. | **Should.** Concede in §5 and in the abstract (v3 gives both numbers): per-question pairing adds 0.4 points, and the gains over generator means come from the transferred rates. Optional B5 (paired CI, descriptive). |
| 3.3 | The prospective test is narrow: same benchmark, prompt and generators. | Accurate | Limitations already says so. The abstract does not name the benchmark. The only new detector is Qwen. | **Should.** Abstract and §5: "on unseen ConflictBank questions, same prompt and generators". |
| 3.4 | 18 of 20 cells predicted at ≥ +3; only two test harm; H33 fails. | Partly accurate | Of the 20 cells, 16 are predicted at ≥ +3 and 2 at ≤ −3; the 18 counts absolute values of 3 or more. Our results.tex:99–100 says "≥ 3 points", which invited the misreading. Both harm cells are terra at one draw. Sign and size were right (−10.1 → −8.3, p 0.016; −3.1 → −4.0, p 0.105), so H33 fails on power. | **Should.** Fix the wording (16 gains, 2 harms). Concede that the test has only two real-detector harm cells. No funded run adds more: C4's cells are randomised-direction cells, so they replicate H31c, not harm from selective real detectors. C-u6 (unfunded) would add real-detector harm cells. Decline extra terra seeds (Section 6). |
| 3.5 | The account fails on clean questions; the mean error is 3.6. | Accurate (understated) | Qwen cells: −16.4/−8.3, −12.2/−5.3, −19.6/−2.7, −25.6/0.0. In 12 of 20 cells the detector flags no clean question, so prediction and observation are 0.0 by construction. All of the error comes from 8 cells. | **Must.** Report the clean MAE over the 8 informative cells beside the 20-cell figure (B6). Explain the 0/0 mechanism in Table 43's caption. State that f_W does not transfer to clean questions. |
| 3.6 | The account fails on HealthContradict (luna −19.4 predicted, −8.6 observed). | Accurate | Also Haiku −16.6 vs +1.1 (wrong sign) and gpt-4o-mini −8.2 vs −2.7. The main text cites only luna. Exposures use labels the judge confirms for only 37% of items. | **Should.** Give all three misses, or at least Haiku's. Add the label clause. Re-estimate on human-confirmed items once A4 exists (B18). |
| 3.7 | Fix: model why follow rates vary, starting with prior; condition f_W on closed-book correctness. | Recommendation: partly agree | Feasibility is thin on ConflictBank. Under the PR§11.3 rule (closed-book correct on at least 2 of 3 seeds), the known H31 f_W pools are 5/11/10/22 (48 pooled; 259 not known). Requiring every seed gives 5/7/9/22 (43). The f_R pools hold at most 5 per generator under either rule. Fresh has no closed-book run. HealthContradict is richer than we first said. hc_mini, hc_luna and hc_haiku all carry closed-book arms. All three also carry `llm_dir+flag_directional` with the same gpt-4o-mini detections, which mark the correct page on 48% of items. So f_R (from oracle rows) and observational f_W can be conditioned for all three generators. But every HealthContradict rate rests on labels the judge confirms for only 37% of items. Where a label is reversed, C2's inverted-oracle "wrong" flags are right flags. That is the objection at 5.3. | **Should.** B7(a) and (c) now (ConflictBank). B7(b), C2 (HealthContradict inverted oracle, $0.8) and optional C10 ($2) wait for A4. They are analysed on human-confirmed items only; without A4 they do not run. Do not promise to re-score the fresh cells without C9. |
| 3.8 | Fix: report error separately for negative cells and clean questions. | Partly accurate | Clean error is in Appendix E but diluted. Only 2–3 cells are predicted negative, so an MAE over them would be anecdotal. | **Must.** B6: list the negative cells individually in the main text; clean MAE over the 8 informative cells; conflict and clean error split for the selection analysis. |

### Issue 4: metric validity

| # | Reviewer claim | Verdict | Evidence | Planned response |
|---|---|---|---|---|
| 4.1 | The matcher's misses depend on the condition (10.3% / 11.7% vs 5.2%). | Accurate | Full re-score, judge-only / matcher-only: oracle-directional 8.71/0.86%, answer key 6.66/1.36%, LLM flag 2.93/0.54%, verdict key 2.39/0.76%. What biases a contrast is the gap between its two arms. The judge is not ground truth either. | **Should.** One §4 sentence: the misses are one-sided and depend on the condition, so near-zero contrasts need both scorers. |
| 4.2 | H23 and H30 hold only under the matcher. | Accurate | H38 judge: H30a +2.4 [1.3, 3.5], H23 +2.5 [1.1, 3.9], neither within ±3. Even under the matcher, the registered per-comparison H23 TOST holds in only 2 of 9; the pooled TOST that passes is unregistered. | **Must.** Lead with H38. Matcher equivalence becomes a secondary, unregistered observation until A1 decides between the scorers. |
| 4.3 | Both abstract claims depend on the scorer. | Accurate | abstract.tex:8–9 says "no more than an answer key", while its own parenthesis gives +2.5. Its 95% CI, [+0.9, +4.2], excludes 0 but is not printed in the abstract. abstract.tex:16–17 gives the verdict key with no judge figure. | **Must.** Remove both from the abstract (v3 does). |
| 4.4 | No human has labelled anything; a few hours would fix it. | Accurate | All label columns are empty (audit_200, audit8_pairs, audit8_raguard, ramdocs_suspect_docs, phase0). `audit_200.csv` is registered (PR§15.4.7) and mentioned at setup.tex:92. It has no answer-key or verdict-key outputs, and it shows the condition and the matcher score. The paired audit8 sheet has 19 matcher–judge disagreements, 0 of them in H30 arms. A1, the sheet that can decide H23/H30, takes about 3.5–4 h per annotator, which matches "a few hours". | **Must; top author task.** A1 (targeted disagreement sheet) can decide H23/H30 if the humans side clearly with one scorer. A2 (audit8 pairs) gives agreement statistics in 3–4 h more each, plus about 1 h joint adjudication. Supersede `audit_200.csv` with A1/A2. Record the change in PR§20. PR§15.4.7 reports agreement only if labels are complete, so no registered result is lost. Rewrite setup.tex:92. |
| 4.5 | Fix: choose the primary scorer from the human agreement. | Recommendation: partly decline | The registered primary is the one-way matcher (PR§8). Overall κ ignores the per-arm error difference that biases a contrast. A full judge switch costs about $41.5 for conflict-question responses only, or about $51 with clean questions (186,214 or 228,379 distinct unjudged non-closed-book responses × $0.000223). Either way it exceeds the $40.82 headroom. | **Should.** Register now: the matcher stays primary; near-zero claims must hold under both scorers; the reference scorer for H23/H30 is the one with the smaller human-estimated per-arm error difference. C12/C13 run only if humans favour the judge, or are split. |
| 4.6 | The §4 judge (terra) is in the same family as systems it scores. | Accurate | 45 of the 162 checked outputs (28%) were terra's own. The outside-family Gemini judge's agreement rates (92.8% of 2,400; 94.3% of 23,204) appear only in Appendices D/E and Limitations. Its H38 re-score is already cited in the abstract, the intro and §5 (results.tex:52–54). | **Should.** §4 cites the Gemini judge (94.3% of 23,204, plus per-condition rates). Move the terra check to the appendix, labelled within-family. |
| 4.7 | Up to 20% of responses lack a final-answer line, under the 256-token cap, and are scored wrong. Fix: report the share per condition in the main tables. | Accurate | gpt-4.1 neutral+instruction cell: 60 of 300 rows lack the line, all at exactly 256 completion tokens; 49 are conflict questions; 53 of the 60 score wrong. It affects both registered H13 gpt-4.1 tests, with most of the damage in the neutral arm. The statistic covers 42 of 202 runs (gpt-4.1 round 2: seed 0 only); H9's holdout_gpt41, H10 and H8 are not covered. PR§5 says these counts "are reported", and scores the responses as they are, with no exclusions or reruns. | **Must.** B11 (all 202 runs) gives the counts PR§5 promises. Report them in full, per run and condition, in an appendix table. In the main tables, mark cells above 1% and give each table's range in its caption. That is a partial decline of "in the main tables", for space (Section 6). B12 is a sensitivity analysis only. H13 keeps its registered status, and nothing is rerun. |

### Issue 5: real-document evidence

| # | Reviewer claim | Verdict | Evidence | Planned response |
|---|---|---|---|---|
| 5.1 | HealthContradict: the judge confirms both labels for only 37%. | Accurate | 68% correct page, 55% misinformation page, 37% both (92 of 247). The judge saw the same 4,000-character pages as the generators; 357 of 494 pages hit that cap. Only 60 of 494 pages (12%) read as reversed; most unconfirmed pages are "unclear". | **Must.** Report the scale precisely: 63% of items lack a confirmed pair, 12% of pages read reversed. Disclose the breakdown as first tabulated on 2026-11-02, before PR§20 (B16). A4 is the real check. |
| 5.2 | Yes/no with one page per side, so the oracle flag gives the answer away; H37a is close to trivial. | Accurate | The flag heads the typed-misinformation page; the generator answers the opposite of its stance. Staying below closed-book accuracy is not evidence against this: on RAGuard the explicit answer key also leaves terra and Haiku 20–30 points below closed-book. The majority class ("no", 62.3%) is above gpt-4o-mini's no-flag 60.5%. | **Must.** Say H37a measures deference to released labels. Add the majority rate to tab_hc's caption. Retitle "Direction helps on real web pages". Optional C3 (HealthContradict answer-key arm, $0.7). |
| 5.3 | If half the labels are wrong, wrong flags are really right flags, which alone would explain the overshoot. | Accurate as a possibility; untested until A4 | We tabulated a bound on 2026-11-02. Judge-detected reversals (17 correct-typed pages) could move a prediction by up to 3.9 points for gpt-4o-mini, 5.3 for luna and 4.4 for Haiku (17/247 × the larger frozen follow rate). None of the three exceeds its gap (gpt-4o-mini 5.5: −8.2 predicted, −2.7 observed; luna 10.8; Haiku 17.7). The rate-free bound (17/247 = 6.9) exceeds only gpt-4o-mini's gap. But the bound counts only reversals the LLM judge detected. 63% of items lack a confirmed pair, mostly "unclear". And a judge cannot stand in for the human check A4 provides. The 43 misinformation-typed pages read as supporting gold push the other way. Priors and the clean-question overshoot seen on fresh are rival explanations. | **Must.** Concede that label noise could explain the overshoot and that we cannot test it until A4. Report the miss as unexplained, with label noise, priors and transferred rates as candidates. Remove "but not real web pages" as an explanation. Use the judge-based bound in neither the response nor the paper. B18 after A4; B17 only beside it. |
| 5.4 | RAGuard: 6% of pairs contradict; abstention 32–61%; every generator is better closed-book; H26 still listed as support. | Accurate | 9 of 150 pairs contradict (6.0%). Closed-book is better for the 4 of 5 generators that ran it (Sonnet has no closed-book arm). H26 is cited as support in Table 3 (3 rows), results.tex:24/38/42/63, and the pooled moderator. | **Must.** Tag H26 "label transfer" everywhere. Rename or drop the "natural misinformation" moderator. Fix "every generator". A3: label the 50 RAGuard pairs. |
| 5.4b | (Not raised) The RAGuard answer-key arm was run and never reported. | Accurate | Answer key minus no flag: +1.4 / +42.1 / +19.0 / +19.8. Oracle directional: +9.9 / +8.8 / +6.8 / +12.7 (gpt-4o-mini / luna / terra / Haiku). The key carries more than the flag for luna, terra and Haiku; for gpt-4o-mini it carries less. Macros exist; nothing cites them. | **Must.** Add a key column to tab_cb's RAGuard rows. Limit "close to an answer key" to the synthetic benchmarks. State the gpt-4o-mini exception. Any test of key minus flag must be registered first. |
| 5.5 | CONFLICTS gives only nulls. | Accurate | 69 conflict items, only 5 of them misinformation. The appendix description of the 237 items leaves out 9. The intro calls it "the one benchmark of real search results", but HealthContradict pages are also retrieved pages. | **Should.** Remove or qualify the intro sentence. Fix the appendix description. |
| 5.6 | All confirmatory results use synthetic misinformation; the abstract and intro overstate. | Accurate | abstract.tex:15–16, intro.tex:14–15/24/31 and discussion.tex:8. Limitations already says confirmatory evidence is synthetic. | **Must.** Rewrite the abstract and intro (v3 drops HealthContradict). Make Table 3 and Limitations agree. |
| 5.7 | Fix: re-label a HealthContradict subset, or present these benchmarks as label-transfer stress tests and take them out of the abstract. | Recommendation: do both if possible | 247 items, 81 topics; about 590 words per cut page; about 3–5 minutes per item. The pinned repo has no licence file. | **Must:** the reframe, now. **Should:** A4 (100 items) if the authors hold a labelling session anyway. A4 also gates C2, C10 and B7(b). |

### Issue 6: neutral flag, policy shift vs harm

| # | Reviewer claim | Verdict | Evidence | Planned response |
|---|---|---|---|---|
| 6.1 | Multi-answer "harm" is partly compliance with "give every valid answer"; the same behaviour is rewarded on WikiContradict (+6.5). | Accurate | WikiContradict ran under the default (multi-answer) skeleton. On ConflictBank, gpt-4o-mini lists both sides under that skeleton (answers containing the misinformation 72.8→83.2) and abstains under the minimal one (abstention 3.2→32.4; answers containing the misinformation 51.8→45.9). H16's misinformation rise is not significant; its cost runs through abstention. Claude models move the other way. | **Should.** Rewrite the §5.1 lead as a policy shift whose direction the prompt clause decides, with claim 1's exceptions. Add the ConflictBank-minimal panel to fig_answers. Optional B22 (RAMDocs minimal answer mix). |
| 6.2 | Under the one-answer prompt, selective accuracy does not change (54.8 → 55.0). | Accurate | Ratio estimand +0.2 [−4.8, +5.2], not equivalent at ±3. This ratio replaced the registered per-question mean after the data were seen; the registered estimand gives −1.0 [−4.9, +2.9]. Abstention rises by +34.3. Abstentions hit misinformed and correct answers alike (45% vs 37%). Under majority misinformation, the stored exploratory numbers show selective accuracy rising for gpt-4o-mini (+8.1) and luna (+8.3), +4.3 [+1.7, +7.0] for Haiku (not significant after correction), and −3.7 [−8.4, +1.0] for terra. | **Must.** Make this the main neutral-flag statement: the flag trades coverage for risk. Use "no detectable change" wording, name both estimands, and show the majority cells. |
| 6.3 | At c = 2 the neutral flag turns positive (−18.7 → +8.0). | Accurate | For gpt-4o-mini, +8.0 is not significant; +34.7 at c = 4 is. luna turns positive too but is never significant (+18.1 at c = 4). Multi-answer settings mostly get worse as c rises (gpt-4o-mini −35.0 at c = 2). The exception is RAMDocs Claude Haiku, which improves (+9.1, then +17.0 and +32.9, both significant). | **Should.** One sentence giving both directions and the Haiku exception. |
| 6.4 | The pooled −1.4 [−5.1, +2.3] has no clear meaning; drop it. | Accurate | 19 of 34 settings share one 300-question ConflictBank pool. PR§17.1.1 registered an equal-weight mean, −3.0 [−3.8, −2.2], "inference about this design only". It was never reported. results.tex:30 also prints a design-conditional CI for −1.4 ([−2.2, −0.6]). Both of these CIs exclude 0. Both describe only this set of settings, which over-represents ConflictBank prompt variants. Neither supports the reviewer, and neither may become a headline. | **Must.** Drop every pooled summary from the abstract, intro and §5.1, including the design-conditional CI at results.tex:30. Describe each policy by its per-setting range. Put the per-setting effects, the REML mean (with its CI, the design-conditional CI and the prediction interval), the moderator and per-policy models, and the equal-weight mean in one appendix table. Label the table: "summaries of these 34 settings, not estimates for any wider set; no claim rests on them". PR§17.1.1 registers all of these, so only the full table meets its reporting commitment. Leaving any of them out, or dropping them altogether, would be a deviation to list. |
| 6.5 | REML variance components are poorly identified. | Accurate | 7 generators and 4 benchmarks; the RAGuard and Qwen moderators coincide with single random-effect levels; the tau values are not in the PDF; 14 of 34 settings use one seed. | **Should.** Drop the meta-regression block and the "+4.6 Anthropic" clause from the body. The moderator model appears only in the appendix table (6.4), with its tau values. |
| 6.6 | The moderator codings were chosen after seeing the data. | Accurate | PR§15.4.3 → PR§16.4.1 recoding happened with per-setting effects in view; this is disclosed. | Resolved by 6.5. |
| 6.7 | Fix: report the neutral flag as risk–coverage, with strict accuracy as one point. | Recommendation: agree | 28 settings have stored risk dictionaries; many conditions per setting are stored. No logprobs exist. Table 2 mixes matchers (INC.4). | **Must.** B21: an operating-point figure re-presenting stored numbers. Fix INC.4 first. Decline the logprob curve. |

### Issue 7: pre-registration weight

| # | Reviewer claim | Verdict | Evidence | Planned response |
|---|---|---|---|---|
| 7.1 | "Pre-registered" opens the abstract. | Accurate | It is the first methods sentence, ahead of any result. Neither the abstract nor the intro says the addenda followed earlier results; only Setup, Limitations and Table 3 do. | **Should.** v3 moves it out of the lead. Add "after earlier results were known" to intro.tex:13. |
| 7.2 | The distinctive results are exploratory. | Accurate | Table 3 marks break-even, the account and prevalence as unregistered, and the pooled effects as "re-analysis of data already seen; dependence: §17" (appendix.tex:40). Break-even direction precision and the account were first computed in Round 10 (5e5ab30). A conflict-prevalence net effect was already computed and stored on 2026-10-27 (07d7419, the RAMDocs generality settings); Round 10 only reweighted it across settings. None of the three had an addendum. PR§17.1 later registered re-analyses of the account and prevalence on data already seen. The pooled effects were registered as exploratory re-analyses in PR§13.4 (DerSimonian–Laird), PR§16.4.2 (multilevel) and PR§17.1.1, each after per-setting effects were seen. The abstract labels only the pooled −1.4 as exploratory; the intro labels only prevalence. | **Must.** Label every unregistered result where it appears in the abstract and intro. Where a re-analysis was registered after the data were seen, say so. |
| 7.3 | Ten addenda followed earlier results. | Partly accurate (a slip) | Eleven exist (PR§9–§19). The reviewer's own INC.1 gives eleven. Appendix F starts at "Second registration", which invites the count of ten. | **Could.** Settled by INC.1. Not listed as a reviewer error. |
| 7.4 | Registering just before running constrains how, not which; forking paths remain (532 + 636). | Accurate | Registrations preceded their runs by 4 s to 8 min. Pooled TOSTs were chosen after the per-comparison TOSTs (2 of 9). Shelved pilots exist (PoisonedRAG 40 questions, HoH 40 items, both in runs/). PR§19.3 already publishes the numbers of the 30-item HealthContradict smoke (plain 70%, closed-book 77%, LLM flag 70%, detector direction 0.46). Its outputs exist only in the LLM cache, which the supplement excludes. | **Should.** In Limitations and Appendix F, say each addendum followed a review of earlier results. List the pooled-after-per-comparison choice and the shelved pilots. Cite the smoke numbers only as "seen before registration (PR§19.3)", never as results. A9 decides whether the smoke outputs are exported from the cache into runs/. |
| 7.5 | H24's admitted uninformativeness shows an ill-posed hypothesis. | Accurate | −G sits on both sides of both registered tests (PR§13.3). Table 16 still says "supported" (tab_hypotheses.tex:406–408). | **Should.** Status becomes "supported (mechanical: −G on both sides)". Revise the existing Table 3 H24 row (appendix.tex:52) and add one Limitations sentence calling H24 mis-specified. |
| 7.6 | A dozen registrations in four days, timestamped only by own commits. | Accurate (worse than stated) | 18 PR commits (2/9/2/5 per day). Seven addenda were pushed before their runs; the base, PR§9, PR§10, PR§17 and PR§18 were pushed after. "Committed before its runs" is false for PR§9.4 (11:18:21, after holdout_qwen started at 11:13:10; PR§9.4 itself says 113 `plain` rows existed), PR§9.6 (during the Qwen run) and PR§17.6 (10:04:42, after judge outputs at 09:51–09:54). The claim appears at intro.tex:13 ("each committed before its runs"), setup.tex:58 ("each before its runs"), results.tex:5–6 ("in an addendum before its runs (H10--H38)"), the Table 3 caption (appendix.tex:62), appendix.tex:364–365, appendix.tex:374 (which names the Qwen seed change), and the addendum hypothesis-table captions, which the shared template at paper_assets.py:413 generates. The supplement excludes git history; the `supplement.zip` on disk stops at PR§12. | **Must.** Correct the claim in all seven places. Regenerate the hypothesis tables. Add all three cases to the deviation list. Append a dated correction note to the PR; never edit frozen text. Add a 12-row registration-timeline table: commit hashes and times, no author fields, local-only timestamps marked. Rebuild the supplement (A7). |
| 7.7 | Process jargon ("to finish within the session", "computed during review triage"). | Accurate (wider than quoted) | About 12 more phrases in Appendix F: "earlier/first draft" ×5, budget/"$195 limit" ×5, "parallel tracks" ×2. | **Must.** Rewrite all of Appendix F's process wording; keep every disclosure. Disclose late-computed numbers with dates, e.g. "first tabulated on 2026-10-29, before PR§18 was written". |
| 7.8 | The TOST margin was set post hoc. | Accurate | ±3 is half the observed H2 effect, fixed after the confirmatory run (PR§9.3) but before the H23/H30/H37/H38 runs. Smallest passing margins: judge 3.9 / 3.5; matcher 1.3 / 2.2. | **Should.** State the margin's history and report the smallest passing margins under both scorers. |
| 7.9 | The NLI window was changed after seeing detection output. | Accurate | Disclosed. NLI-300's results appear only in Appendix B prose (appendix.tex:19–23). The window is also described in the Appendix F deviation list (appendix.tex:366–368) and in Limitations ("NLI sees only a sentence-sized window", limitations.tex:8). | **Could.** Add a "post hoc window" tag (see M-b1). |
| 7.10 | Fix: use a third-party registry for future addenda. | Recommendation: agree | Past addenda cannot be backdated. | **Should.** Deposit the current PR and hashes on OSF now; register PR§20 there first (A6). |
| 7.11 | Fix: name H32–H38 as the confirmatory component beyond H1–H9. | Recommendation: partly agree | H32–H36 had frozen numeric predictions, but their criteria were set with the retrospective fit known (MAE ≤ 3 against a retrospective 1.5; the near-tying baseline was given no criterion). H37 followed a smoke test; H38 re-scores outputs already seen. | **Should.** Three tiers: (i) H1–H9; (ii) H10–H31 and H37–H38, registered before their runs but after earlier results; (iii) H32–H36, prospective. Use "confirmatory" only for (i) and (iii). |

### Moderate issues

| # | Reviewer claim | Verdict | Evidence | Planned response |
|---|---|---|---|---|
| M-a1 | ConflictBank clean questions are trivially conflict-free. | Accurate | Correct passage plus three off-subject passages. The luna, terra and Haiku detectors flag 0/100 (Sonnet too, on eval). The gpt-4o-mini detector names a correct passage as suspect on 7% (eval) and 9% (fresh) of clean questions; Qwen2.5-7B (fresh) does so on 29%. gpt-4.1 flags 4 of 100 on eval but names a correct passage on none. | **Should.** Say in Setup and Limitations that these questions test off-topic false flags only. Report each detector's clean flag rate. |
| M-a2 | RAMDocs clean questions include same-name ambiguity; that makes the detector fire on 76.2%. | Partly accurate | 104 of 126 clean questions have several gold answers. The causal "makes" is untested. The share of clean questions where the suspect is a correct document is 65.9% on RAMDocs vs 7–9% on ConflictBank. | **Should.** Describe them as multi-answer questions. Optional B23. |
| M-a3 | Prevalence (22 of 31 negative at 5%) inherits both problems. | Accurate | 21 of the 22 are ConflictBank or RAMDocs. At 5% the effect is 0.95 × clean + 0.05 × conflict. Only the gpt-4o-mini detector is used. | **Should.** Group Table 26 by clean construction; drop the single count. Optional B24. |
| M-a4 | The 5% is arbitrary. | Accurate | Unjustified at every use. Break-even is undefined in 13 of 31 settings. | **Must.** Replace with break-even where defined, plus a stated 1–20% range. Cite CONFLICTS' composition with caveats; note that poisoning prevalence is adversarial. |
| M-a5 | Add on-topic, non-contradicting clean questions. | Recommendation: defer | ConflictBank has no second passage for the same fact. Same-subject other-fact passages need author checks. A run matching Table 26 costs about $5.4 for its 14 distinct clean settings; the 4 CB-majority cells reuse the one-answer clean questions. | **Should (writing):** cite CONFLICTS, RAGuard and RAMDocs "none" as existing on-topic evidence. Run C-u2 only if prevalence stays in the main text. |
| M-b1 | NLI recall is 0.0% under the registered window; the 300-char fix is post hoc. | Accurate | Registered windows flag nothing. NLI-300: recall 21.0, precision 24.7, removal +7.6. limitations.tex:8 ("sentence-sized window") is wrong for the registered runs. | **Should.** One main-text sentence; label NLI-on-ConflictBank rows exploratory; fix limitations.tex:8. |
| M-b2 | CAAG is the authors' own heuristic and mainly a cautionary tale. | Accurate | F1 11.6; H5 n.s. The paper already says as much. | **Could.** Move it to an appendix section; one sentence stays in §3. |
| M-b3 | No trained detector (for example ConflictRAG), claim-decomposition or long-context NLI baseline. | Accurate | None exists. The intro (intro.tex:5) and related work (related.tex:49) cite ConflictRAG as the motivating system. It is a trained MiniLM+MLP query–document classifier with an LLM fallback. Its resolution stage scores source credibility (Entropy-TOPSIS), which picks a side (related_work_notes.md:33). Whether its code or weights are released is unchecked (A15). A trained classifier would bear on the neutral-flag, removal and clean-question false-flag results, whether or not it names a side. The oracle neutral flag is not a ceiling when it hurts (−18.7 vs 0.0 for a detector that flags nothing). Only removal has an oracle ceiling. | **Should.** Scope every detector claim to "the detectors tested". Name ConflictRAG in the response and in Limitations. If A15 finds released code and weights, add it as C-u7. If not, decline: re-training needs labelled conflict pairs we do not have, and a re-implementation would not be ConflictRAG. C11 (sentence-pair NLI, $0.5). Decline long-context NLI (needs a download; C11 covers the sentence-level case). |
| M-c | Direction precision is fact-checking from memory or artefacts, not contradiction detection; frame H27/H30 accordingly. | Partly accurate | True for the two-passage benchmarks (ConflictBank, HealthContradict), which the reviewer names. There, "knowledge from outside the passages" includes general plausibility, not only recall of the fact (Issue 2). RAMDocs allows agreement: its PR§11.3 half shows 88.3% direction accuracy on 132 unknown facts. RAGuard is not one post per side (54 of 294 claims are 1:1). | **Must.** Reframe §3.2 and the H27/H30 paragraph for the two-passage benchmarks, pending C1/B2. |
| M-d | The single call "matches" the pipeline (H18) and belongs in the main text. | Partly accurate | H18 covers gpt-4o-mini only: −1.3, −0.2, −1.8, all n.s.; the paper says "does not differ significantly". Its ±3 TOSTs already exist under PR§13.4's "as good as" rule, and none shows equivalence (90% CIs [−4.2, +1.6], [−3.6, +3.1], [−4.1, +0.6]; `\tostHEighteen*`). The paper never cites them. luna, terra and Haiku lose 37.7–61.0 points on clean questions with a single call. gpt-4.1 has a single-call arm outside H18 (cb_single_gpt41, seed 0, Table 5): +7.3 [+2.7, +12.0] on conflict questions and −5.0 [−10.0, 0.0] on clean ones. | **Should.** Two sentences in the Discussion and rule. For gpt-4o-mini, no significant difference at half the calls. For gpt-4.1, a gain on conflict questions at a small clean cost (one seed, exploratory). It backfires with the reasoning models and Haiku. Report the existing H18 TOSTs (B.1; B14 moves there). None is equivalent, so the paper cannot say "matches". |
| M-e | Confidence-bearing flags are never tested. | Accurate | Only holdout_caag_original (CAAG, RAMDocs). No detector in the directional pipeline emits a confidence. | **Should.** C5 (synthetic stated confidence on the H31 design, $2.9). Real verbalized confidence (C-u5) only if C5 moves the rates. |
| M-f1 | "Claude models move less" rests on two models, one single-seed. | Accurate (contradicted by our own numbers) | Sonnet uses seed 0 everywhere. Haiku's +3.9 under forced choice is the only significant neutral effect on accuracy. On RAMDocs, Haiku's misinformation share moves −7.9 (significant), more than luna or terra. No Claude setting uses the minimal or ConflictBank multi-answer prompt. | **Must.** Drop "and model family" from the abstract. Rewrite the intro line as an observation about two models. Label the moderator post hoc. |
| M-f2 | Claude via an OpenAI-compatible endpoint, default temperature, regex fallback. | Accurate | Disclosed. Parse failures: 1 of 800 Claude-detector outputs on the ConflictBank eval split (1 of 1,200 with Haiku on fresh; the paper's figure is 1 of 2,400 across the six eval detectors, appendix.tex:186), and it gave the same outcome as a clean parse. The regex fallback parsed 105 (Haiku eval), 118 (Haiku fresh) and 198 (Sonnet eval) of 400 outputs. `config.json` records temperature 0.1 for 36 Claude, 15 luna and 6 terra runs, but none was sent. PR§19 lists Haiku at "T 0.1". | **Should.** Name the setting differences in one sentence. Document the effective temperature in runs/README and Appendix F; add a PR§20 note on PR§19's "T 0.1". |
| M-f3 | Family is confounded with age and capability. | Accurate | All 8 Claude settings use current models; 16 of 26 non-Claude settings use 2024–25 models, including all minimal-prompt settings. terra is the most accurate OpenAI model only on ConflictBank one-answer (73.0 vs gpt-4.1 64.7) and forced choice (77.7), where it sits close to Claude. On RAMDocs (multi-answer prompt) it is the least accurate OpenAI model (10.8 vs 14.4–22.1); it trails luna under majority misinformation (26.0 vs 29.8) and gpt-4o-mini on RAGuard (24.8 vs 36.7). | **Should.** One sentence naming the confounds. Decline an accuracy-moderator regression. |
| M-f4 | The only open model is a 4-bit 7B with one seed. | Accurate | At T = 0.1 seeds differ little (PR§9.4). Qwen has no H31 arm. | **Could.** C8 (Qwen-7B H31 arm, local, $0). 32B only with download approval (A12). No extra 7B seeds. |
| M-g1 | Served snapshot ids were not recorded. | Accurate | `llm.py` discards `resp.model` and `system_fingerprint`; the cache has no such field. Cannot be recovered. | **Should.** Concede in Limitations. Record ids from PR§20 on (code prerequisite; `LLMResult` must gain the fields). |
| M-g2 | terra provides the top break-even and the strongest harms from one default-temperature sample. | Accurate | 25 of 25 runs use seed 0. Undocumented: terra's plain condition was sampled twice (80 of 400 identical texts) because the cache key holds the command-line temperature. H31 uses one draw; Table 2 and H19 use the other. | **Should.** B8 (two-draw sensitivity, free). C4 gives a fresh-question replication. One-draw caveat on Table 25. |
| M-h1 | The RAMDocs "oracle" is not an oracle: at least 37% of pairs cross entities. | Accurate | 261 of 709 pairs; already disclosed. | **Could.** Define "label oracle" once and use it for every benchmark. |
| M-h2 | The CI on −4.4 pools settings that share questions; was the dependence modelled? | Accurate (the paper does not say) | Yes: one question resample shared by all 10 settings per replicate (PR§18.2), conditional on those settings. Table 41 has no pooled row. | **Should.** Add the pooled row and one clause. Optional B19. |
| M-i | terra follows 95% of wrong flags: promote it, examine explanations, link to Chen et al. 2025. | Partly accurate | 95 [89, 99] (90 of 95); no overlap with the other generators. "Most capable" holds among the four H31 generators only; Sonnet is more accurate and untested. The pattern is asymmetric within terra (67 [53, 80] right vs 95 wrong). Its right-flag rate overlaps luna's (72 [60, 83]). One draw, matcher-scored, no judge check. No reasoning text is returned. The visible explanations are one sentence citing a passage, then the final-answer line. They cannot show why terra defers. Chen et al. 2025 is not yet in references.bib. | **Should.** One sentence plus terra's row in the new main-text table, with caveats. Cite Chen et al. after adding the entry and checking it (A11). B9 (API completion tokens, which include hidden reasoning tokens) is the only, indirect, signal. A5 is dropped. C4 and C6 test the pattern. No reasoning summaries. |

### Inconsistencies and presentation

| # | Reviewer claim | Verdict | Evidence | Planned response |
|---|---|---|---|---|
| INC.1 | "Nine later addenda" vs Appendix F's "tenth and eleventh". | Accurate | Eleven addenda; nine carry hypotheses. A third numbering scheme sits in the PR headings. PR§17 registered runs and a re-score; only PR§18 is analysis-only. | **Must.** One Setup sentence with the counts. Replace ordinals with PR§ numbers in Appendix F and `stage_txt` (paper_assets.py:392–394). |
| INC.2 | §9.5, §13.4 etc. point nowhere. | Accurate | 77 references to 24 PR targets: 41 in sections, 36 in generated captions. | **Must.** Add a `\prs{}` macro; replace all 77; regenerate; add one definition sentence. |
| INC.3 | The Table 3 pointer says Appendix B; the table prints in Appendix C. | Accurate | All of Appendix B's floats print on pp. 18–49 (Figure 2 on p. 18, where Appendix C begins). A blanket `\FloatBarrier` would push every later appendix about 30 pages back. | **Should.** Fix the pointer; place Table 3 inside B; move the hypothesis tables to their own final appendix; check the layout. |
| INC.4 | Table 2 mixes matchers. | Accurate | Correct = one-way strict; misinformation = bidirectional `wrong_rate`. The bold marks come from H21(a) only in the ConflictBank one-answer and forced panels; the RAMDocs and ConflictBank multi-answer panels take theirs from the exploratory family, on the same bidirectional value (tab_risk caption; paper_assets.py:525–533). | **Should.** Name each column's matcher in the caption now. Use one matcher for the risk–coverage figure, either via B15 (registered one-way rescore plus an H21(a) rerun) or via C12. |
| INC.5 | Two judges, three agreement figures. | Accurate | terra 92.6% of 162; Gemini 92.8% of 2,400; Gemini 94.3% of 23,204. | **Should.** One "Scorers" paragraph in §4 (with 4.6). |
| INC.6a | The follow-rate equation is inline with poor notation. | Accurate | R and W mean H31 effects in the main text but R is recall in Table 37. a_i is no-flag accuracy while A in Table 37 is accuracy with the flag. Code and PR use other symbols. | **Must.** One numbered equation in §3; rename the H31 effects (Δ_R, Δ_W) and Table 37's A. |
| INC.6b | "Following" is defined three ways. | Partly accurate | Tables 25 and 37 compute the same adoption rate; Deviation 17 describes a superseded draft. But the equation and Appendix E treat f_W as lost accuracy, and Table 25's share-of-headroom column is that loss rate (up to 10 points away). So the paper does use more than one definition; the reviewer's locations are off. | **Must.** One boxed definition: f_W is measured as adoption and enters the equation as lost accuracy. Fix Appendix E l.270–271. Relabel or drop share of headroom. Never redefine the frozen H32 rates. |
| INC.7 | Too many terms. | Accurate | All eight appear ("verdict key" 45 times). "Direction accuracy d" and "direction precision" name the same thing. Bare "precision" is used in two senses. | **Should.** Six-row terminology box in §3. Never write bare "precision". Drop "share of headroom". |
| INC.8 | The abstract's long sentences; at most five numbers. | Accurate | The quoted sentence exists. | **Must.** v3 abstract (Section 3.4). |
| INC.9 | H4b "supported" but ρ = −0.08 under label robustness, with the point estimate at the edge of its interval. | Accurate | The prose gives only −0.39. The test itself was re-specified in PR§9.3. Table 30 prints ρ = −0.08 [−0.56, −0.02]. Its caption already says the bootstrap distribution is skewed, but not why. The question bootstrap of a Spearman correlation over 18 cells, with small differences between cell means and ties, is skewed and lies mostly below the full-sample value. So the percentile interval cannot be read as support. | **Should.** Caveat in app_grid.tex:45; status "supported (label-sensitive)". Say in Table 30's caption why the interval is unreliable. Register a second form of the label-robust test in PR§20 (B26), as a sensitivity analysis only. |
| INC.10 | Seeds look nominal; seed disagreement (6.7%) is about the same as the flag–key gap (11.7%). | Partly accurate | gpt-4o-mini seeds differ on 2–3% of answers; luna up to 16%. H23: 11.7% [10.2, 13.2] vs 6.7% [5.7, 7.7], no overlap, and 11.7% is discordance, not the gap (net −0.1). H30a: 8.3% vs 9.5%, so the reviewer is right there. Seed variation sits inside the pooled CIs. | **Should.** Reword results.tex:55, which set up the misreading. List the single-seed cells. Optional B13. |

### Problems we found that the reviewer did not raise

| # | Problem | Evidence | Planned response |
|---|---|---|---|
| X1 | Appendix E says the matcher misses more under the keys. The data show it misses more under the flags. | Judge-only 8.71% (flag) vs 6.66% (key); 2.93% vs 2.39%. CONTEXT Round 13 repeats the error. | **Must.** Fix both. |
| X2 | PR§19.2 registered per-condition agreement; the paper gives only the overall figure. | rev.judge19.agreement holds the four rows. | **Should.** Add them to Appendix E and cite them in §4. |
| X3 | Appendix F's "across settings and conditions" truncation figures cover only the r2 settings. | 42 of 202 runs (32 settings; gpt-4.1 round 2 seed 0 only). | **Should.** Scope the sentence now; replace it after B11. |
| X4 | Follow rates, p* and the H32 error use the matcher only. | The only judge check is the 120-question H31 re-score, for gpt-4o-mini and Haiku. terra's 95% has none. | **Should.** Say "under the string matcher" in the abstract (v3), §5 and Limitations. C13 is conditional. |
| X5 | The registered PR§11.3 closed-book split is not in the reviewed draft. An earlier draft printed the ConflictBank half (added in 8e0e261, removed in 1bd1e11); the RAMDocs half was never printed. | ConflictBank: 85.7% (24 of 28 known) vs 67.0% (177 of the 264 flagged among 272 unknown). RAMDocs: 87.5% (7 known) vs 88.3% (132 unknown). Macros exist, uncited. | **Must** (see Fix b). |
| X6 | The PR§17.1.1 equal-weight pooled mean was never reported. | `\neuMlEqual` −3.0 [−3.8, −2.2], uncited. results.tex:30 also prints a design-conditional CI for −1.4 ([−2.2, −0.6]). Both exclude 0 and describe these 34 settings only. | **Should** (see 6.4): one appendix table, design-only label, no body mention. |
| X7 | "Committed before its runs" is false for PR§9.4, PR§9.6 and PR§17.6. | See 7.6: seven places. | **Must.** |
| X8 | The supplement excludes git history; `supplement.zip` is stale. | make_supplement.sh line 3; the zip stops at PR§12. The remote and commit author fields identify the authors, and earlier commits hold the unredacted RAGuard sheet (CONTEXT Round 14). | **Must.** Rebuild; timing evidence per A7. |
| X9 | terra and luna plain conditions were sampled twice; configs misrecord the effective temperature. | See M-f2 and M-g2. PR§15.2 item 2 says the hint-run plain "is a cache hit" for gpt-4o-mini, luna, Haiku and Qwen; for luna it was not (175 of 1,200 plain texts identical between cb_single_luna and hint_cb_luna_mini). PR§15.2 item 4 (terra) makes no such claim, but terra's hint plain was also re-sampled (80 of 400 identical). | **Should.** Document in runs/README and Appendix F; B8. |
| X10 | The registered PR§11.3 majority-variant direction accuracy is not in the reviewed draft. | `\cbmajDirAccLlm` 49.4 vs `\cbDirAccLlm` 68.8. It was printed from 8e0e261 until 876c6a8; neither `sections/` nor any table cites it now. With more misinformation passages than correct ones, the gpt-4o-mini detector's verdict moves with passage counts. | **Must** (writing). Report it beside 2.5 and in the Issue 2 response. |

---

## 3. Proposed restructure

### 3.1 The three claims

We would build the paper around these three claims. They are the parts of the evidence that are registered or tested prospectively and that the reviewer named. All three are scored by the string matcher; the scorer caveat goes with each.

1. **A neutral flag shifts the answer policy; it does not reliably improve accuracy.**
   - Under a one-answer prompt that allows "unknown", it trades coverage for risk for gpt-4o-mini and luna. gpt-4o-mini's selective accuracy on ConflictBank shows no detectable change (54.8 → 55.0).
   - Under "give every valid answer", it adds the misinformation (ConflictBank gpt-4o-mini selective 21.6 → 12.9; misinformation also rises significantly for gpt-4.1 and Qwen). The same compliance is rewarded on WikiContradict (+6.5).
   - Exceptions, stated with the claim. Under the one-answer prompt, abstention barely moves for Haiku (1.2 → 0.9), Sonnet (1.0 → 1.7) and terra (1.7 → 4.7, not significant). On RAMDocs under the multi-answer prompt, Haiku's misinformation share falls (55.2 → 47.2, significant).
   - Status: H21 and H8 registered. The 54.8 → 55.0 figure is the PR§15.4.5 ratio estimand, which replaced the registered per-question mean after the data were seen; the registered H21(c) gives −1.0 [−4.9, +2.9]. The risk–coverage view (PR§16.4.1) is a re-analysis registered and run on data already seen. Table 2's multi-answer significance marks (including gpt-4.1, Qwen and RAMDocs Haiku 55.2 → 47.2) come from the exploratory family. Scope: four of the seven generators, with the exceptions above.
   - Why it holds: it reorganises existing numbers, most of them registered. Scorer: on the PR§17.3 sample the judge moves these contrasts by up to 3.3 points. It also makes Haiku's ConflictBank neutral effect significant: +6.7 [+0.8, +13.3], against +3.3 under the matcher. So the claim is about direction, not size, until Table 2 uses one matcher (INC.4) and A1 is done.
2. **Generators defer to directional flags; for three of the four tested, wrong flags cost more than right ones gain.**
   - On ConflictBank the randomised right and wrong effects are registered (H31a–c); H31c is significant for three of the four generators (gpt-4o-mini −1.9 [−12.2, +8.6]). The follow rates are registered descriptives with no test (PR§16.3). The break-even (0.52–0.79) is exploratory.
   - Three of the four H31 generators follow wrong flags about as often as right ones. terra follows 95% of wrong flags against 67% of right ones.
   - The RAMDocs grid, registered before any confirmatory data, gives support from a second benchmark: −1 / +4 / +13* at d = 0.5 / 0.75 / 1.
   - Scope: synthetic misinformation about facts the generators rarely know (closed-book 4.3–20.0%), with the PR§11.3 split and the C1 probes. Matcher-scored; terra's rates have no judge check (X4).
   - Why it holds: the right-flag effects (+17.6 to +33.5) and wrong-flag effects (−31.2 to −64.5) are much larger than the scorer shifts seen so far (−1.7 to +3.3).
3. **The frozen account predicts within a benchmark and shows where it fails.**
   - It predicted 20 cells on unseen ConflictBank questions: MAE 1.4 [1.2, 2.4], slope 0.88. Most of that comes from the transferred rates: the cell-level marginal scores 1.8.
   - It overstates harm on clean questions (H35). It missed on HealthContradict, whose labels are unreliable; label noise may explain that miss.
   - Status: H32 and H34–H36 prospective (tier iii); H33 failed. Matcher-scored. This feeds the decision rule.

What is demoted:
- Answer-key and verdict-key equivalence (H23/H30/H38) shrinks to one paragraph: it depends on the scorer and is unresolved until A1.
- Prevalence becomes an input to the rule.
- The family claims are dropped.
- RAGuard and HealthContradict become label-transfer stress tests. CONFLICTS stays an underpowered null (it has no passage labels or oracle pairs).
- Every pooled summary moves to one appendix table that describes these 34 settings only (6.4).

### 3.2 Section-by-section outline (8 pages, about 16 columns)

The column budget is a rough guide. The body already fills page 8, so the cuts must land before the additions.

| Section | Columns | Content | Cut or moved |
|---|---|---|---|
| Abstract | — | v3 (Section 3.4). | Old abstract. |
| §1 Introduction | ~1.5 | Problem; question; three claim bullets, each with a status and scope tag; one sentence on stress tests; one on what we do not claim (key equivalence, family). | Running H-number lists; the CONFLICTS sentence; "Claude models move less"; "every benchmark that labels one". |
| §2 Related work | ~1 | Trim to ~450 words. Add Chen et al. 2025 beside Turpin 2023 and Sharma 2024; keep the citations on LLMs favouring generated passages (jin2024tugofwar, kortukov2024realistic, tan2024blinded). Keep ConflictRAG, and say we did not test it as a detector. | — |
| §3 Framework | ~2 | 3.1 Flags (neutral, directional; answer and verdict keys as controls). 3.2 LLM detector, and what a verdict means on two-passage benchmarks. 3.3 Synthetic detector definition (H31 uses it) plus one grid sentence. 3.4 Terminology box (six rows) and the numbered equation with one definition of following. | CAAG, NLI and baselines, grid results, H5/H6, control definitions → appendix; one sentence each stays. |
| §4 Setup | ~2 | Core benchmarks (RAMDocs; ConflictBank with its construction consequence and closed-book range). RAGuard and HealthContradict as label-transfer stress tests, one line each; CONFLICTS as an underpowered null. WikiContradict for genuine conflicts. Generators, with one sentence on setting differences. Scorers paragraph. Registration tiers in 2–3 sentences. Table 1 with a claim-status column. | Long benchmark descriptions; pre-registration paragraph → 2–3 sentences plus an appendix pointer. |
| §5.1 Claim 1 | ~2 | Risk–coverage figure (B21), or Table 2 trimmed to ConflictBank and RAMDocs under one matcher. One-answer vs multi-answer vs minimal route. The generators that do not move. WikiContradict. One utility sentence. | Pooled −1.4 and every other pooled summary, REML moderators, per-policy means; utility table stays in the appendix. |
| §5.2 Claim 2 | ~2.5 | Oracle direction helps (brief). H31 randomised. Follow rates (exploratory). terra asymmetry. p*. Grid corroboration. PR§11.3 split (both halves), the majority-variant contrast, and scope. New compact table (Table 25 rows plus prospective summary); Figure 1 left panel. | H24 registered correlation → one "ill-posed" sentence. |
| §5.3 Claim 3 | ~1.5 | Bookkeeping plus rate transfer. H32 and baselines (marginal 1.8). Two harm cells named. Clean overshoot over the 8 informative cells. HealthContradict miss, with label noise as an untested explanation. H36 (plus C4 if run). | Figure 1 right panel can move to the appendix if space is short. |
| §5.4 What does not hold | ~1 | One paragraph: H23/H30/H38 depend on the scorer. One paragraph: stress tests (HealthContradict oracle as an answer-key ceiling; RAGuard key vs flag, with the gpt-4o-mini exception; CONFLICTS power). Prevalence as an input. | Detail → Appendices D and E. |
| §6 Discussion and rule | ~1.5 | Decision rule (3.3), single call, what is untested. | — |
| Limitations | outside the page limit | ConflictBank regime; scorer and audit status; label noise on HealthContradict; pre-registration timing; snapshots; terra single draw; family confounds; trained detectors such as ConflictRAG untested; NLI wording fix. | — |

Space: the larger sinks (pre-registration paragraph, benchmark descriptions, pooled estimate, H23 paragraph, verdict-key paragraph) come to about 114 PDF lines. The reviewer's four items add some more. The additions (terminology box, equation, compact table, Discussion, scoping sentences) come to about 60–70 lines. Do the cuts first, then rebuild and check that Limitations still starts on page 9.

### 3.3 Decision rule for practitioners (draft for §6)

The rule is derived from these runs. It has not been tested end to end. Its detector-choice step was illustrated, in sample, on the fresh cells (PR§19.1).

0. **Scope.** It applies to misinformation conflicts in a low-prior regime like ConflictBank. Where genuine multi-answer conflicts are common (same-name entities, contested facts), use neither direction nor removal (H28). No tested detector tells genuine conflicts from planted ones.
1. **Measure the generator.** Estimate f_R and f_W with a small randomised-direction run. In our runs that cost about $0.2–1.8 per generator per seed (400 questions, plain and flagged arms, full price). p* is the direction precision a detector needs on the true pair, counting only flags that name one side. It is not sufficient. A detector above p* can still hurt through flags on correct passages and on clean questions. Example of the first route: Haiku generator × gpt-4o-mini detector, at least 68.8% against 0.67, observed −4.1 on conflict questions. Whether p* should veto a detector is untested.
2. **Choose the detector, or none, with the frozen account.** Use a labelled sample that includes clean questions, reweighted to the deployment's conflict share. Flag only if the predicted effect is positive. This is the PR§19.1 procedure. On the fresh cells at a 75% conflict share it picked the Haiku detector for every generator; for terra that gave +5.3 against −3.5 with the gpt-4o-mini detector. We wrote the rule after seeing these outcomes, so this is an illustration, not a test.
3. **Prevalence.** Flag only if the conflict share exceeds −C/(K − C), where K is the effect on conflict questions and C the effect on clean questions. Measure C directly; the account overstates it (H35). So far C has been measured on non-trivial clean questions only for the gpt-4o-mini and Qwen detectors (C7 adds Haiku).
4. **Neutral flag.** Under a one-answer prompt it acts as a costly abstention trigger for gpt-4o-mini and luna. Their utility change turns positive between c = 1 and c = 2; it is significant only for gpt-4o-mini, at c = 4. terra, Haiku and Sonnet show no significant change. Under multi-answer prompts it hurt gpt-4o-mini at every c, and luna at c = 0. It helped Claude Haiku on RAMDocs, significantly at c = 2 and c = 4.
5. **Single call.** For gpt-4o-mini it does not differ significantly from the two-call pipeline (H18; its ±3 TOSTs do not show equivalence); on clean questions it gives −1.0 [−4.0, +1.7]. For gpt-4.1 (one seed, outside H18) it gives +7.3 [+2.7, +12.0] on conflict questions and −5.0 [−10.0, 0.0] on clean ones. Do not use it with reasoning models or Claude Haiku: they lose 37.7–61.0 points on clean questions.

### 3.4 Abstract plan

Five numbers: **95%, 0.52, 0.79, 1.4, 1.8**. The abstract says the numbers come from string matching. It makes no claim the judge re-score contradicts, and it leaves out HealthContradict. Registration is not in the lead, and the break-even is marked exploratory. terra's f_W already has a macro (`\rfolWrongTerra`, 95 [89, 99]); the 1.8 has one (`\frMaeMarg`).

Draft v3 (about 270 words). If the venue limit is 200 words, cut the third sentence, the sentence that begins "A flag that names the false side", and the release clause, then shorten further as needed. Never cut the exceptions, the scorer clause or the 1.8 baseline.

> Retrieval-augmented generation systems increasingly tell the generator when retrieved passages contradict each other. We ask what such a flag must say, and how reliable its detector must be, to help against misinformation. We vary the detector and the flag's wording on fixed passages across seven generators. For four of them, a neutral flag ("these passages may conflict") moves answers along the policy the prompt sets: towards abstaining when one answer is asked for, and towards listing both sides when the prompt asks for every valid answer. The other three barely move or move the other way. A flag that names the false side helps when it is right, but generators also follow it when it is wrong. On ConflictBank, three of the four generators tested this way followed wrong flags about as often as right ones; the most accurate followed 95% of wrong flags, more than right ones. For three of the four, a wrong flag costs more than a right one gains: in an exploratory analysis, direction on true contradictions pays only above a break-even precision that rises with generator accuracy, from 0.52 to 0.79. Assuming follow rates transfer, an expected-value account frozen in advance predicted detector-generator cells on unseen ConflictBank questions with a mean error of 1.4 points, against 1.8 for a cell-level version that ignores which questions are flagged. It overstated the harm on questions without a conflict. These values come from synthetic misinformation about facts the generators rarely know, scored by string matching. We derive a rule for choosing a detector, or none, not yet tested end to end, and release the harness, pre-registrations and all runs.

Changes from the earlier draft:
- **95% clause.** "The most accurate generator we tested" was false, because Sonnet is more accurate. It now reads "three of the four generators tested this way …; the most accurate followed 95% of wrong flags".
- **Follow rates.** "Followed it about as often when it was wrong as when it was right" contradicted terra's 67% vs 95% and results.tex:81. It now covers three of the four generators, and terra's asymmetry is stated.
- **Policy sentence.** It now says the shift holds for four of seven generators, and that the other three barely move or move the other way (claim 1's exceptions).
- **Multi-answer clause.** "When several are allowed" becomes "when the prompt asks for every valid answer". The minimal prompt also allows several answers, but it leads to abstention (6.1).
- **Account.** It now states its one assumption (follow rates transfer), and the 1.8 cell-level baseline sits beside the 1.4 (3.1, 3.2). "20 new pairs" is dropped to keep five numbers, and "new" with it: 15 of the 20 detector-generator pairs were already crossed cells (only the four Qwen-detector cells and terra × Haiku detector are new).
- **Cost asymmetry.** "A wrong flag costs more than a right one gains" now covers three of the four generators; H31c is not significant for gpt-4o-mini.
- **Flag wording.** The neutral flag is quoted as "these passages may conflict", as in intro.tex:10 and the prompt.
- **Scorer.** The numbers are matcher-only (X4), and the regime sentence now says so.
- **Break-even.** It is marked exploratory.
- **Regime.** A sentence on the regime is added (Fix c).
- **Families.** "Three model families" is dropped.

Rewrite the abstract last, after the claims and the rule are settled.

### 3.5 Two papers?

Not advisable. The first paper's distinctive claim fails under the judge and has no human labels. The second would be the restructured paper minus claim 1. Both would reuse every run and one pre-registration, so they would overlap heavily as submissions. Revisit a short separate paper on deference only if prior conditioning (B7, C2 and C10, after A4) or C6 (Sonnet) produces a clear finding.

---

## 4. Work packages

The reviewer's own plan maps onto our packages as follows:
1. Label the audit → A1, A2.
2. Artefact probes and prior conditioning → B1, B2, C1, C9; B7, C2, C10 (the HealthContradict parts after A4).
3. Rebuild around claims → Section 3, B21, D.
4. HealthContradict → D (reframe now), A4.
5. Pre-registration framing and cross-references → D.

Order of work:
- Writing fixes for factual errors can start now.
- PR§20 must be written, committed, pushed and deposited on OSF before any B analysis, C run, or human-label export or analysis.
- C4 and C10 need predictions frozen and committed before their flagged runs, as in PR§19.
- B7(b), C2, C10 and the HealthContradict part of B2 wait for the A4 labels. Their HealthContradict estimands use human-confirmed items only. Without A4 they do not run.
- C12 and C13 wait for the A1 result.
- Any number first tabulated on 2026-11-02 is disclosed with that date, never as a "triage" number.

### A. Author-only

Human labels come only from the authors. No model stands in for them, and a judge-selected subset is never "verified".

| ID | Task | Effort | Depends on | Notes |
|---|---|---|---|---|
| A1 | Targeted disagreement sheet. Per arm (oracle-directional, answer key, LLM flag, verdict key): 50 random matcher–judge disagreements plus 25 agreed outputs, so 300 outputs. Blinded (condition and both scorers hidden), shuffled. | 3.5–4 h per annotator | PR§20 audit plan; export script (Claude can write it, mirroring export_audit8.py) | Can decide H23/H30, but only if the humans side clearly with one scorer. It gives human-calibrated per-arm shifts. 76–91% of each arm's disagreements are judge-only (oracle-directional 91%, answer key 83%, LLM flag 84%, verdict key 76%; judge-only rates 8.7%, 6.7%, 2.9% and 2.4%). For H23, the SE of a calibrated arm shift is then about 0.4–0.65 points, depending on the arm and on how often humans side with the judge. The SE of the flag-minus-key contrast is about √2 larger (0.65–0.85). For H30a (LLM flag vs verdict key) the contrast's SE is about 0.3 or less. The 25 agreed outputs per arm can bound the rate at which both scorers are wrong, but cannot estimate it. |
| A2 | `audit8_pairs.csv` (150 pairs, registered PR§17.4). Two annotators, after a ~10-item calibration from outside the sheet. Adjudicate. | 3–4 h each + ~1 h joint | Labelling can start now. PR§20 (B25) before any label is read or adjudicated: PR§17.4 registers the sheet, not an analysis. | κ per label and for "strict"; PABAK or AC1 for rare labels; each scorer's agreement with the consensus; McNemar on the 19 discordant items. |
| A3 | `audit8_raguard.csv` (50 pairs), same sitting. | ~2 h each | Labelling can start now. PR§20 (B25) before any label is read. | Human–judge agreement on RAGuard contradiction. |
| A4 | HealthContradict stance sheet: 100 items, fixed seed, at most 2 per topic, pages exactly as the generators saw them, labels hidden. Timed 10-item pilot first. | 5–8 h each | PR§20; export script; A9 (licence) | Then B18 on the human-confirmed subset. Gates B7(b), C2, C10 and the HealthContradict part of B2. |
| A5 | (Dropped.) Code terra's visible explanations. | — | — | terra's flagged responses are one sentence citing a passage, then the final-answer line (M-i). Coding them cannot show why terra defers. B9 stays. |
| A6 | OSF: deposit the current PR plus a SHA-256 per PR version; register PR§20 and later addenda there. | <1 h | — | Anonymised view link for reviewers. |
| A7 | Rebuild the supplement and decide how reviewers can check timing under double-blind review. For review, give only the anonymised OSF view (A6) and a table of PR commit hashes with commit and push times, without author fields. Keep git history, GitHub push records and provider usage dashboards out of the supplement. The remote (github.com/12Parker/…) and every commit's author name and email identify the authors. Earlier commits also hold the unredacted RAGuard sheet (CONTEXT Round 14). Offer the full history after review, or to the chairs through a confidential channel if the venue allows it. | 1–2 h | A6 | — |
| A8 | Decide how to disclose that addenda followed (multi-agent) mock reviews, and any AI assistance, per ACL policy. | Decision | — | The PR already says it in PR§18. |
| A9 | HealthContradict licence and stance-label provenance. The 30-item smoke outputs are in the LLM cache, which the supplement excludes, while PR§19.3 already states their numbers. Decide whether to export them into runs/ (a local cache read, no API call). | <1 h | — | — |
| A10 | Look up which dated snapshot each model alias pointed to on 2026-10-26 to 2026-10-29, where providers document it. | <1 h | — | State it as documented, not recorded. |
| A11 | Add the Chen et al. 2025 entry ("Reasoning Models Don't Always Say What They Think") to references.bib and check it against the primary source. | Minutes | — | paper/README rule. |
| A12 | Approve or decline the Qwen-32B download (~20 GB). | Decision | — | Costs time (~7 h per seed, estimated), not money. |
| A13 | If C-u2 runs: check each on-topic clean pair for incidental contradictions. | ~1–2 h | C-u2 | — |
| A14 | (Optional.) `ramdocs_suspect_docs.csv`: 163 documents typed correct that contain a wrong answer (101 hold-out, 62 pilot), all unlabelled. PR§1 tied the RAMDocs labels to this sheet, but PR§8 item 2 settled the question: `exclude` for the main analysis, `contains` as robustness, no hand labels. | ~2–3 h each (estimate) | PR§20 if used | Hand labels would be a new registered robustness check. Say in the response that the sheet is released unlabelled. |
| A15 | Check whether ConflictRAG's code or trained weights are public (arXiv 2605.17301). | Minutes | — | Decides M-b3: C-u7 if public, a stated decline if not. |

Minimum labelling if time is short: A1 alone, about 3.5–4 h per annotator. A1 + A2: about 7–9 h. Everything (A1–A4): about 14–19 h per annotator.

### B. Zero-cost re-analyses (register as PR§20 before computing)

**B.1 Already computed, report only (writing, no registration needed):**
- the PR§11.3 closed-book split, both halves, with denominators (numbers.tex:4431–4466);
- the PR§11.3 majority-variant direction accuracy (`\cbmajDirAccLlm` vs `\cbDirAccLlm`);
- the PR§17.1.1 equal-weight mean, with the REML mean, the design-conditional CI, the prediction interval and the moderator and per-policy models, in one appendix table labelled as describing these 34 settings only (6.4);
- the PR§19.2 per-condition agreement;
- the RAGuard answer-key macros (descriptive);
- the H18 TOSTs at ±3 (`\tostHEighteen*`, PR§13.4; all three "not equivalent"; formerly B14);
- the tau values, if any REML stays.

**B.2 New analyses on existing runs (register first):**

| ID | Analysis | Answers | Data |
|---|---|---|---|
| B1 | Passage-only text classifier (TF-IDF logistic regression) trained on non-eval, non-fresh ConflictBank rows; pairwise accuracy on eval and fresh. | 2.0, Fix a (lower bound on surface readability) | QA_dataset.json (local); exclusion as in prepare_cb_fresh.py; sklearn in .venv-conflict |
| B2 | Closed-book direction split for gpt-4.1, luna, terra and Haiku (eval) and gpt-4o-mini (HealthContradict, after A4, human-confirmed items only). Pin the metric (one-way strict), seed rule, direction definition and pooling. | Fix b | cb_single_* closed-book; cross_cb_* detections; hc_mini |
| B3 | (Optional) single-side direction precision. First tabulated on 2026-11-02, before PR§20; disclose as such (B.3). | 2.3, 1.8 | cross_cb_* detections |
| B4 | Common-rate baseline (one f_R and one f_W for all four generators) on the 20 fresh cells. | 3.1 | randdir_cb_*, fresh_cb_* |
| B5 | Paired-bootstrap CI on MAE(marginal) − MAE(account); descriptive. | 3.2 | predictions19.json; revision10 bootstrap |
| B6 | Clean MAE over the 8 informative cells; predicted-negative cells listed individually; conflict and clean error split for the selection analysis. | 3.5, 3.8 | Table 43; rev.fresh19 |
| B7 | Prior conditioning. (a) ConflictBank H31 f_W by closed-book correctness, pooled with generator effects (307 questions: 48 known vs 259 not under the PR§11.3 rule; by seed count, 43 correct on every seed, 242 on none, 22 in between; PR§20 fixes the rule). (b) After A4, on human-confirmed items only: HealthContradict f_R by closed-book from oracle rows, and observational f_W, for all three generators. (c) Conditioned account on the 30 crossed cells. State in advance that (c) can move only the 4–20% of known questions. | 3.7 | randdir_cb_*, cb_single_*, hc_* |
| B8 | terra two-plain-draw sensitivity for H31 (break-even, follow rates) and H19. | M-g2 | cb_single_terra vs randdir_cb_terra; holdout_terra vs holdout_rem_terra |
| B9 | API completion_tokens, plain vs flagged, by arm and by followed or not, for terra, luna and Haiku; descriptive. Reasoning models' and Claude's counts include hidden reasoning tokens. terra's plain vs flagged counts were first tabulated on 2026-11-02, before PR§20; disclose as such (B.3). | M-i | randdir_cb_* |
| B10 | In-sample check of the decision rule's choices on the 20 fresh cells. Step 2 is the PR§19.1 selection, already computed on these cells, and step 1 was revised with their outcomes in view. So this describes, it does not test. The 30 crossed cells cannot test the rule either, because they were used to derive it. | 1.8 | predictions19.json; fresh_cb_* |
| B11 | Missing-final-answer and at-cap shares for all 202 runs, using the scorer's own regex. Reported in full per run and condition (PR§5). | 4.7, X3 | Every row has response and completion_tokens |
| B12 | H13 gpt-4.1 sensitivity analysis excluding questions with no final-answer line in either arm. Reported beside the registered result, which keeps its status. | 4.7 | holdout_gpt41_round2 |
| B13 | Per-seed estimates (range across seeds) for pooled H23/H30 and the Table 2 contrasts. | INC.10 | 3-seed cells |
| B14 | (Moved to B.1.) H18 TOST at ±3: already computed (`\tostHEighteen*`); report only. | M-d | — |
| B15 | One-way misinformation rescore (on a copy, since rescore.py writes in place); rerun H21(a); regenerate Table 2 and Figure 3. | INC.4 | pred_answers in every row |
| B16 | Report the HealthContradict page-stance breakdown, disclosed as first tabulated on 2026-11-02, before PR§20. | 5.1 | runs/judge10/hc_labels.jsonl |
| B17 | (Optional, after A4.) H37a–c and H37d on the judge-confirmed subset (92 items, 45 topics), labelled "judge-confirmed" and reported only beside B18. It is not evidence about label noise. | 5.3 | hc_*; judge10 |
| B18 | Agreement of the human labels with the released labels and the judge; H37a–c and H37d on the human-confirmed subset. | 5.3, 5.7 | Needs A4 |
| B19 | Setting-level sensitivity for the single-entity −4.4 (leave one setting out, or random effects). | M-h2 | entity9 inputs |
| B20 | (Optional) CI on RAGuard key minus flag. | 5.4b | rg_* |
| B21 | Risk–coverage operating-point figure. It re-presents the stored risk numbers for 28 settings. The ConflictBank-minimal selective point and the 5 RAMDocs generality settings are new numbers. | 6.7 | rev.r2; holdout_* |
| B22 | RAMDocs minimal answer mix (3 settings). | 6.1 | holdout_minimal, _gpt41_minimal, _qwen_minimal |
| B23 | RAMDocs clean flag rate and effect, multi-gold (104) vs "none" (22). | M-a2 | holdout_*; conflict_type |
| B24 | (Optional) prevalence curves with CIs; RAMDocs three-way mixture. | M-a3, M-a4 | rev.prevalence7/8 |
| B25 | Human-audit analysis plan: κ, PABAK/AC1, McNemar, per-arm calibrated shifts, the reference-scorer rule (4.5). | 4.4, 4.5 | Before any label is read |
| B26 | H4b label-robust correlation in a second form (a permutation p-value or a bias-corrected interval, chosen in PR§20 before computing); sensitivity analysis only. | INC.9 | holdout_contains; revision_stats H4 code |

PR§20 also registers every C run, with its estimand and, for C4 and C10, the two-stage freeze.

**B.3 Numbers first tabulated on 2026-11-02, while writing this plan.** None enters the paper unless a registration covers it, and the paper then discloses it as "first tabulated on 2026-11-02, before PR§20":
- the HealthContradict page-stance joint breakdown and the 357/494 at-cap count;
- the HealthContradict oracle-directional "unknown" rows (161 of 741, gpt-4o-mini);
- single-side direction precision (about 88–97%);
- clean-question abstention under the single call (about 65 / 56 / 39%), and the passages named suspect there (about 2.4 / 2.1 / 1.1 of 4 for luna / terra / Haiku);
- terra's plain vs flagged completion tokens, and the length of its visible responses (one sentence; the paper describes it in words, not word counts, per the token rule);
- the HealthContradict overshoot bound (3.9 / 5.3 / 4.4 with the frozen rates; 6.9 rate-free). It must stay out of the paper and the response: it rests on judge labels and does not answer the label-noise objection (5.3);
- the terra share (45 of 162) in the old judge sample;
- the extra gpt-4.1 truncation detail;
- the sampled ConflictBank clash examples (use a fixed-seed sample in the paper);
- the colleague's counterfactual terra p* (about 0.68): keep it out.

The plain-draw identity counts (80/400 etc.), the data-composition counts (104/126 multi-gold; CONFLICTS types; 163 suspect documents) and the response counts behind the C13 cost basis describe the data, not outcomes. They can be reported as documentation.

### C. New API runs, ranked by value per dollar

Basis, unless a row says otherwise: summed `cost_usd` from comparable rows in existing runs (`results.jsonl`, `detections.*.jsonl`), scaled by row counts. C1's single-passage part and C9's Sonnet part are token-based, and C8 is measured in time. Cached rows keep their original cost, so these are full-price upper bounds. Each row says whether it carries contingency. Headroom is $40.82 before the $245 gate.

Code prerequisites (no API; `caag_core.py` untouched; keep `--dry-run` working):
- record `resp.model`, `system_fingerprint`, `id` and `created` (add optional fields to `LLMResult` and to the cache value, not the key);
- a probe module on the judge10 pattern, logging API token counts;
- a swap prepare script and loader;
- confidence rendering in prompts.py;
- `predict19.py` support for synthetic detections;
- launch every cached-plain run with the same `--temperature` and `base_url` as the original, or the plain rows will not hit the cache.

**Funded package (run after PR§20).**

| Rank | Run | Answers | Cost | Cumulative | Basis | Depends on |
|---|---|---|---|---|---|---|
| C1 | Artefact probes, lean: answer-only (both answer orders), single-passage (700 calls per model), object swap (about 254–266 usable items). Detectors gpt-4o-mini, Haiku, luna, terra; ConflictBank eval; one seed. | Issue 2 (decisive) | $7.8 | $7.8 | single-passage ~450 input tokens per call ($3.0, conservative); answer-only from cb_single* closed-book per-row costs × 600 ($2.26); swap from the four eval detectors' measured cost per conflict item ($0.0027) × 254–266 items ($0.72); ×1.3 contingency | PR§20; probe code |
| C3 | HealthContradict answer-key arm, 3 generators × 3 seeds. | 5.2: how much of the key the flag carries | $0.7 | $8.5 | hc_* plain arms: $0.58 in total for the three generators (3 × 741 rows); one extra sentence; about +20% | PR§20 |
| C4 | Randomised direction on ConflictBank-fresh for luna, Haiku and terra; predictions frozen first from synthetic detections and existing fresh plain answers. | 3.1 (H36 extended to 4 generators), H31c on new questions, M-i (95% on new questions). Not 3.4: these are randomised-direction cells, not selective real detectors. | $2.1 | $10.6 | randdir_cb_{luna,haiku,terra} synthetic-arm sums $1.66; +25% | PR§20; predict19 change; two-stage freeze |
| C5 | Synthetic stated confidence (55% vs 95%) on the H31 design, same direction draws; gpt-4o-mini, Haiku, luna ×3 seeds, terra ×1. | M-e (title question) | $2.9 | $13.5 | randdir_cb_* per-row costs × 2 arms × 300 ($2.93); no contingency (+25% would add $0.7) | PR§20; prompts.py rendering |
| C6 | Sonnet H31 arm, seed 0, ConflictBank one-answer. | M-i ("most capable most deferential"), M-f1 | $3.0 (up to $6.75 if plain misses the cache) | $16.5 | cb_single_sonnet flagged rows $0.0083–0.0105 × 300; a plain cache miss adds $3.75 (cb_single_sonnet plain: $3.7508 for 400 rows); no contingency | PR§20; same --temperature 0.1 as cb_single_sonnet |
| C7 | Haiku detector on the RAMDocs hold-out; gpt-4o-mini generator, seeds 0–2. | 1.8: clean cost C for the detector the rule selects | $0.4 | $16.9 | fresh Haiku detection $0.164 per 400; RAMDocs prompts about half as long; holdout_qwendet flagged $0.16 per 900 rows; about $0.22 at these rates, rounded up to $0.4 | PR§20 |
| C8 | Qwen-7B H31 arm, local. | M-f4: open-model follow rate | $0 (~30 min) | $16.9 | about 6 s per real flagged call (147 uncached hint calls in 14:57, hint_cb_qwen_luna.log) × 300 flagged calls; the 3.4 s/it in cb_single_qwen.log averages over cached and closed-book rows | PR§20 |
| C9 | Closed-book on ConflictBank-fresh (gpt-4o-mini ×3, luna ×3, Haiku ×3, terra ×1; Qwen local), plus Sonnet closed-book on eval. | Fix b on fresh and for Sonnet (should); B7 on fresh | $4.9 | $21.8 | cb_single* closed-book costs $2.25; Sonnet from Haiku's closed-book tokens ($0.13 per seed) at Sonnet prices, $2.6; no contingency (+25% would add $1.2) | PR§20 |
| C11 | Sentence-pair NLI baseline (cached nli-deberta-v3-base, local detection); flag and remove with gpt-4o-mini ×3. | M-b3 | $0.5 | $22.3 | ≤2,400 rows × $0.00022; no contingency | PR§20; new detector class |

**Conditional on A4** (HealthContradict human labels; analysed on human-confirmed items only; not run without A4):

| Rank | Run | Answers | Cost | Cumulative | Basis | Depends on |
|---|---|---|---|---|---|---|
| C2 | HealthContradict inverted oracle (oracle_inv+flag_directional), 3 generators × 3 seeds. Where a released label is reversed, its "wrong" flags are right flags, hence the A4 restriction. | 3.7: f_R and f_W at high prior | $0.8 | $23.1 | hc_* oracle+flag_directional: $0.164 + $0.103 + $0.358 per 741 rows; +25% | PR§20; A4 |
| C10 | Frozen HealthContradict cells: luna and Haiku detectors × 3 generators, conditioned rates frozen before any flagged answer. | 3.7 confirmatory part | $2.0 | $25.1 | hc_* llm_dir flagged $0.65 per detector; detection about $0.2; about $1.5 at these rates, so about +33% | B7, C2, A4 |

**Conditional on A1** (run only if humans favour the judge, or are split):

| Rank | Run | Cost | Cumulative | Basis |
|---|---|---|---|---|
| C12 | Judge re-score of Table 2's conflict cells (also resolves INC.4). | $4.3 | $29.4 | 19,072 distinct unjudged responses × $0.000223 (judge10 mean); no contingency |
| C13 | Judge re-score of randdir_* and fresh_cb_* (X4), conflict and clean questions (H35 needs clean). | $3.6 | $33.0 | 16,064 distinct responses × $0.000223; no contingency. Conflict questions only: 13,407, $3.0. Reusing the 3,229 already judged: 12,835, $2.9. |

The reserve is about $7.8 ($40.82 − $33.0). It covers a Sonnet cache miss (+$3.75) and 25% overruns on C5 and C9 (+$1.9). That worst case comes to about $38.7 and leaves about $2.

**Unfunded unless money remains** (in this order; most fit only if C12/C13 do not run):
- C-u1, terra extra draw on eval flagged arms ($2.4): C4 already replicates the 95%.
- C-u6, new real-detector harm cells: terra (one draw) with the Qwen, gpt-4o-mini and luna detectors on a second fresh sample, predictions frozen first ($4.7 with 25% contingency; fresh_cb_terra_{qwen,mini,luna}: plain $0.92, flagged $0.86–0.89 each, detections $0.21). This is the run that answers 3.4.
- C-u2, ConflictBank on-topic clean set for Table 26's 14 distinct clean settings ($5.4; the 4 CB-majority cells reuse the one-answer clean questions; plus A13): only if prevalence stays in the main text.
- C-u7, ConflictRAG as a detector, only if A15 finds released code and weights (cost to estimate then; gpt-4o-mini fallback calls, of the order of C7 and C11).
- C-u4, claim-decomposition + NLI ($0.8).
- C-u5, real verbalized-confidence detector ($4 per detector, including a matched no-confidence arm): only if C5 moves the follow rates.
- C-x, probe extension to gpt-4.1, Sonnet and fresh ($19.5).

The gpt-4.1 H13 rerun (formerly C-u3) is declined (Section 6).

**If the budget is tight, cut in this order:** C12/C13 (they are conditional anyway), then C9's Sonnet part ($2.6), then C10, C11, C6 and C5. Never cut C1, C3 or C4 (core $10.6). C2 runs whenever A4 exists.

### D. Writing fixes ($0)

**Factual errors (must, now):**
- "Up to 51 points" → `\azSgLunaSelfIdentifyClean` (61.0, luna); the table ref must point to a table that carries it (appendix.tex:176–178).
- "Distrust most of ConflictBank's generated passages" (appendix.tex:176–177) and "distrust ConflictBank's passages when judging them" (limitations.tex:20–21). Rewrite both: in one call that asks for suspect passages, these models name many passages as suspect even on clean questions and abstain. As two-call detectors they flag no clean question. Whether they distrust generated text more than real text is untested.
- Appendix E, appendix.tex:282–283 ("Missing more correct answers under the keys"): the matcher misses more under the flags; also fix CONTEXT Round 13 (X1).
- Every "committed before its runs" claim (X7): intro.tex:13, setup.tex:58, results.tex:5–6, the Table 3 caption (appendix.tex:62), appendix.tex:364–365, appendix.tex:374, and the addendum hypothesis-table captions (template at paper_assets.py:413; regenerate). Add PR§9.4, PR§9.6 and PR§17.6 to the deviation list. Append a dated correction note to the PR. The per-registration sentences at appendix.tex:334, 342, 353 and 359 cover PR§11–§16; check each against the timeline table.
- results.tex:99–100: 16 predicted gains and 2 predicted harms, not "18 at ≥ 3".
- limitations.tex:8: the NLI window wording.
- "Claude models move less" (intro).
- CONFLICTS: the appendix description of the 237 items, and "the one benchmark of real search results" (intro.tex:36).
- RAGuard: "every generator" becomes "all four generators run closed-book".
- appendix.tex:189–195: limit "does not depend on the detector's family" to one Qwen detector on RAMDocs.
- results.tex:81: reword "follow rates overlap except terra's, so p* rises mainly because..." qualitatively.
- results.tex:5–6: Table 3 pointer (INC.3).
- results.tex:30: remove the design-conditional CI with the pooled estimate (6.4).
- setup.tex:92: replace the `audit_200.csv` sentence with the A1/A2 status (4.4).
- runs/README and Appendix F: effective temperature for Claude and reasoning models. PR§20 note on PR§19's "T 0.1" for Haiku.

**Pre-registration framing:**
- Three tiers (7.11), used in Setup, Table 3 and the abstract.
- "After earlier results were known" in the intro.
- Label every unregistered result where it appears; say which re-analyses were registered after the data were seen (7.2).
- State the TOST margin's history plus the smallest passing margins under both scorers.
- Say addenda followed reviews (per A8).
- Pooling came after the per-comparison TOSTs.
- Shelved pilots; the HealthContradict smoke cited only as seen before registration (PR§19.3).
- Registration-timeline table (12 rows; commit hashes and times, no author fields; local-only timestamps marked).
- One OSF sentence.
- PR§20 records: `audit_200.csv` superseded by A1/A2; the dated correction note for PR§9.4, PR§9.6 and PR§17.6.

**Cross-references and counts:**
- `\prs{}` macro; all 77 PR§ references replaced in sections and asset scripts (paper_assets.py, assets5/6/8/9/10.py); regenerate the tables.
- One sentence: eleven addenda, nine with hypotheses; PR§17 registers runs and a re-score, PR§18 analyses only.
- Replace ordinals in Appendix F and `stage_txt`.

**Terminology and equation:**
- Six-row terminology box: hit rate; direction precision (rename the synthetic d to match); pair precision and recall; f_R and f_W; p*; answer key and verdict key.
- One numbered equation, Δ̂(G,D) = (1/n) Σ_i [ r_i^D (1 − a_i^G) f_R^G − w_i^D a_i^G f_W^G ], noting that unflagged questions keep the plain answer.
- Rename the H31 effects (Δ_R, Δ_W) and Table 37's A.
- One definition of following (adoption measured, lost accuracy applied).
- Never write bare "precision". Drop "share of headroom".

**One metric per table:**
- Table 2 caption names each column's matcher and says the bold marks are H21(a) (ConflictBank one-answer and forced) or the exploratory family (RAMDocs and ConflictBank multi-answer), all on the bidirectional matcher.
- Full fix via B15 or C12.

**Judge roles:**
- One §4 "Scorers" paragraph: matcher primary; the Gemini judge's two roles; per-condition rates (X2); the terra check moved to the appendix and labelled within-family; human audit status.

**H4b and H24:**
- H4b: in-text caveat, status "supported (label-sensitive)", and a Table 30 caption that says why the interval is unreliable (INC.9).
- H24: status "supported (mechanical)" in Table 16; revise the existing Table 3 H24 row (appendix.tex:52); one Limitations sentence calling it ill-posed.

**Process jargon:**
- Rewrite all ~14 phrases in Appendix F: session, triage, earlier or first draft, budget, $195 limit, parallel tracks.
- Disclose late-computed numbers with dates: appendix.tex:369 becomes "first tabulated on 2026-10-29, before PR§18 was written"; numbers from this plan become "first tabulated on 2026-11-02, before PR§20".

**Scoping and reframing:**
- ConflictBank construction sentence (2.5), with the majority-variant contrast (X10).
- Regime sentences (Fix c, 2.9).
- Stress-test reframe (5.x); label noise as an untested explanation of the HealthContradict miss (5.3).
- §3.2 detector framing (M-c).
- Detector claims scoped to "the detectors tested"; ConflictRAG named (M-b3).
- Claim 1's exceptions (Haiku, Sonnet, terra).
- Clean-question descriptions (M-a1, M-a2).
- Table 26 grouped by clean construction.
- Break-even prevalence instead of 5%.
- Table 41 pooled row.
- Majority-class rate in tab_hc.
- Single-call clean column in tab_cb.
- RAGuard key column, with the gpt-4o-mini exception.
- Closed-book min/max macros.
- Matcher caveat on follow rates, p* and H32 (X4).
- Missing-final-answer marks in the main tables and the full appendix table (4.7, B11).
- Seed sentence (INC.10).
- terra one-draw caveat in the Table 25 caption.

---

## 5. Decisions needed from the authors

1. **Core claims.** Policy shift; deference and break-even; prediction and its limits. *Recommend:* adopt them as in Section 3.1, with claim 1's exceptions.
2. **One paper or two.** *Recommend:* one.
3. **Labelling.** *Recommend:* A1 first (about 3.5–4 h per annotator, the reviewer's "few hours"), then A2 (another 3–4 h each, plus about 1 h joint), with A3 in the same sitting. Add A4 if a second session is possible; it also gates C2, C10 and B7(b). Supersede `audit_200.csv` and record the change in PR§20.
4. **Scorer rule.** *Recommend:* keep the one-way matcher primary; near-zero claims must hold under both scorers; the reference scorer for H23/H30 is chosen by human-estimated per-arm error difference. Register this before any label is read.
5. **HealthContradict.** *Recommend:* reframe now as a label-transfer stress test and take it out of the abstract. Concede that label noise may explain the overshoot. Re-label only alongside A1–A3.
6. **Pooled estimate.** *Recommend:* drop every pooled summary from the body, including the design-conditional CI. Keep one appendix table with the per-setting effects, the REML mean (with its CI, the design-conditional CI and the prediction interval), the moderator and per-policy models, and the equal-weight mean, labelled as describing these 34 settings only. This meets PR§17.1.1. *Alternative:* drop them all and list a deviation from PR§17.1.1.
7. **API spend.** *Recommend:* C1, C3–C9 and C11 (about $22.3) after PR§20; C2 and C10 ($2.8) after A4; C12/C13 ($7.9) only if the humans in A1 favour the judge, or are split. That leaves a reserve of about $7.8; the worst case is about $38.7.
8. **OSF.** *Recommend:* deposit now, and register PR§20 there before anything runs.
9. **Process disclosure.** Mock reviews and AI assistance. *Recommend:* a short factual statement consistent with ACL policy and the PR's own wording.
10. **Family claims.** *Recommend:* drop them from the abstract; keep one scoped observation about two models only if C6 runs.
11. **H13 for gpt-4.1.** *Recommend:* B12 as a sensitivity analysis only. H13 keeps its registered status, and nothing is rerun (PR§5).
12. **Table 2 form.** *Recommend:* replace it in the main text with the risk–coverage figure (B21) under one matcher, and move the full table to the appendix.
13. **Qwen-32B.** *Recommend:* only if the download is approved and there is machine time; it is not needed for the core claims.
14. **Timing evidence under double-blind review.** *Recommend:* an anonymised OSF view and a commit-hash table only; full history after review or through the chairs (A7).
15. **ConflictRAG.** *Recommend:* check for released code and weights (A15). Add it as C-u7 if they exist; otherwise decline with the reason in Section 6.

## 6. What we will not do, and why

| Request or option | Decision | Reason |
|---|---|---|
| Split into two papers (1.6) | Decline | See Section 3.5. |
| Label `audit_200.csv` as named (4.4) | Replace with A1/A2 | It has no key arms and is not blind, so it cannot test H23/H30. It is registered (PR§15.4.7), so PR§20 records the change. PR§15.4.7 reports agreement only if labels are complete, so no registered result is lost. |
| Report the missing-final-answer share for every condition in the main tables (4.7) | Partly decline | PR§5's counts go in full into an appendix table (B11). The main tables mark cells above 1% and give the range in each caption, for space. |
| Switch the primary scorer wholesale (4.5) | Decline | About $41.5 for conflict-question responses only, or $51 with clean questions; either way more than the $40.82 headroom. It is also a deviation from frozen PR§8. Use the both-scorers rule and C12/C13 instead. |
| Use the LLM judge in place of human labels, or call a judge subset "verified" | Never | Project rule; B17 is labelled "judge-confirmed" and is not evidence about label noise. |
| Fetch reasoning summaries or traces for terra (M-i) | Decline | The API returns none on this path. A summary would need a new client path, would change the call being explained, and is not faithful (the Chen et al. point). Use B9. |
| Code terra's visible explanations (A5) | Drop | The one-answer prompt gets one sentence citing a passage, then the final-answer line. Nearly every response would fall in "cites passage content only", so coding cannot show why terra defers. |
| Re-test H33 with extra terra seeds ($5.3) | Decline | Adds power to the same two cells and looks like rescuing a failed test. C-u6 would add new real-detector harm cells instead. |
| Rerun gpt-4.1 H13 at 1,024 tokens (formerly C-u3, $3.5–4.5) | Decline | PR§5 scores truncated and missing-final-line responses as they are and allows no reruns. Rerunning so that H13 stays "supported" would be an outcome-contingent rescue, the same logic we reject for H33. B12 is a sensitivity analysis only. |
| Sonnet seeds 1–2 ($13.4) | Decline | Does not fix n = 2 models. |
| More Qwen-7B seeds | Decline | Near-duplicates at T = 0.1. |
| Gemini as a third generator family | Decline | It is the outside-family judge; generating with it breaks the judge's independence. |
| ConflictRAG as a trained detector (M-b3) | Depends on A15 | A trained classifier would bear on the neutral-flag, removal and clean-question false-flag results, and its credibility stage picks a side. If code and weights are public, run it (C-u7). If not, decline: re-training needs labelled conflict pairs we do not have, and a re-implementation would not be ConflictRAG. Scope the claims to the detectors tested either way. |
| Long-context NLI | Defer | Needs a download. C11 covers the sentence-level NLI case for the neutral-flag, removal and clean false-flag results. |
| Accuracy-moderator meta-regression | Decline | With 7 generators, accuracy and family are nearly collinear, and the pooled models move out of the body into one appendix table (6.4). |
| Logprob risk–coverage curves | Decline | Needs harness changes and covers OpenAI non-reasoning models only. Stored operating points suffice. |
| End-to-end prospective test of the rule | Defer | Needs on-topic clean questions at a realistic mix, which no current benchmark provides. |
| Full-page LLM re-audit of HealthContradict ($0.45) | Decline | Still an LLM judge, and the generators saw the cut pages. |
| Rebut the label-noise objection with judge labels (5.3) | Decline | A judge cannot stand in for the human check. The bound covers only the 17 judge-detected reversals, while 63% of items lack a confirmed pair, so a bound below each gap does not show that label noise is small. Wait for A4. |
| Conditioned account re-scored on the 20 fresh cells | Only with C9 | Fresh has no closed-book run; without C9 it cannot be done. |
| Git history, GitHub push records or provider dashboards as timing evidence during review | Decline during review | They identify the authors (remote, author fields), and earlier commits hold the unredacted RAGuard sheet. Use the anonymised OSF view and a commit-hash table (A7). |
| Backdating registrations | Not possible | We say that past registrations carry local timestamps only, and register on OSF from PR§20 on. |
| CONFLICTS extension, PoisonedRAG main run, probe extension | Defer | Low value per dollar against the funded package. |
