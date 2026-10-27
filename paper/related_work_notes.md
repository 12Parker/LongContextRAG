# Related-work verification notes

Paper: *When do contradiction flags help retrieval-augmented generation?*
Checked 2026-10-27 against primary sources: the arXiv API and abstract pages, ACL Anthology `.bib` exports, the OpenReview API (venue records and DBLP imports), NeurIPS proceedings `.bib` exports, Crossref for the AAAI DOI, the USENIX presentation page, and the Hugging Face dataset card.

Status legend:
- **VERIFIED**: the planning note's details match the primary source.
- **CORRECTED**: the paper exists, but the planning note had a wrong or incomplete title, authors or venue. The bib holds the corrected version.
- **NOT FOUND**: the paper could not be confirmed. It is kept out of the bib.

All summaries below are my own paraphrases.

---

## Core: contradictions and conflicting evidence in RAG

### wang2025madam: MADAM-RAG / RAMDocs. VERIFIED
- URL: https://arxiv.org/abs/2504.13079 (v2). Venue record: https://openreview.net/forum?id=z1MHB2m3V9 (OpenReview lists the venue as "COLM 2025"; the arXiv comment also says COLM 2025).
- Authors: Han Wang, Archiki Prasad, Elias Stengel-Eskin, Mohit Bansal. The title matches the note.
- Claims: The paper introduces RAMDocs, a dataset that mixes ambiguity (several valid answers), misinformation and noise across the retrieved documents. It also proposes MADAM-RAG, where each document gets its own LLM agent, the agents debate over several rounds, and an aggregator keeps the answers for disambiguated entities and drops misinformation. It reports gains on AmbigDocs (up to about 11 points) and FaithEval (up to about 16 points with Llama-3.3-70B). RAMDocs stays hard: the baseline gets about 33 EM, and performance falls as misinformation outweighs supporting evidence.
- Relation: This is our dataset and one of our four detectors. MADAM-RAG handles conflicts implicitly through per-document debate and never emits explicit flags. We test whether explicit flags (neutral, directional, or removal) do better or worse than this. Their result that performance degrades as misinformation grows fits our finding that neutral flags make the model repeat misinformation.

### cattan2025conflicts: CONFLICTS. CORRECTED (title)
- URL: https://arxiv.org/abs/2506.08500 (v2). No peer-reviewed venue found: OpenReview has only a CoRR record and an untitled-venue record, and nothing turned up in the ACL Anthology. Cite as arXiv.
- Correct title: "DRAGged into Conflicts: Detecting and Addressing Conflicting Sources in Search-Augmented LLMs". The benchmark inside it is called CONFLICTS.
- Authors (Google): Arie Cattan, Alon Jacovi, Ori Ram, Jonathan Herzig, Roee Aharoni, Sasha Goldshtein, Eran Ofek, Idan Szpektor, Avi Caciularu.
- Claims: The paper proposes a taxonomy of conflict types in RAG, each with its own desired model behaviour: no conflict, complementary information, conflicting opinions or research outcomes, outdated information, and misinformation. CONFLICTS has 458 expert-annotated queries. Prompting models to reason about conflicts explicitly helps: predicting the conflict type and then answering gains about 9 points. Giving the model the gold conflict type helps much more (about 24 points), which leaves large headroom.
- Relation: This is the closest prior work on telling the model about conflicts. Their oracle result says that knowing the right conflict type helps a lot. Our result refines it: a perfect but neutral "these passages contradict" flag can hurt on misinformation. What helps is directional information, saying which passage is wrong, which is close to what their "misinformation" category plus desired behaviour encodes. Their taxonomy explains why a type-agnostic flag is not enough.

### wang2026conflictrag: ConflictRAG. CORRECTED (the paper exists; authors and status added)
- URL: https://arxiv.org/abs/2605.17301 (v1 2026-05-17, v2 2026-06-08). The id is correct. The arXiv comment says "submitted to IEEE SMC 2026". No acceptance was found, so cite as arXiv.
- Authors: Chenyu Wang, Yueyuan Li, Yingmin Liu, Yang Shu.
- Claims: ConflictRAG detects, classifies and resolves conflicts before generation. The detector is two-stage: a frozen all-MiniLM-L6-v2 encoder with MLP heads classifies query-document pairs (binary conflict, plus factual, temporal or opinion type), and pairs below a confidence of 0.7 go to an LLM. This cuts API cost by about 62% at about 91% detection accuracy. Source credibility is scored with Entropy-TOPSIS, and the paper proposes a "CARS" diagnostic metric. It reports 88.7% detection F1 and 5 to 6 points of correctness gain over the strongest baseline. The six baselines are Standard RAG, RAG+Reranking, Self-RAG, CRAG, NLI-Filter and CoT Detection. Benchmarks are ConflictQA, a self-built NQ-Conflict and AmbigQA. The backbone is mainly GPT-4o-mini, with DeepSeek-V3 and Claude-3.5-Sonnet also tested.
- Relation: ConflictRAG optimises and reports pair-level detection F1 and assumes that better detection gives better answers. Its ablation shows that removing detection costs about 17 points, but it never tests whether F1 differences between detectors predict downstream gains. Our finding that pair-level F1 does not predict downstream effect, while document-level behaviour does, speaks directly to this design. Its NLI-Filter baseline (generate only from a consistent subset) is the same idea as our "remove the suspect passage" condition, and we show that this condition is high-risk under false flags. It uses the same generator family as us (GPT-4o-mini).

### yuan2026confrag: ConfRAG. CORRECTED (title)
- URL: https://aclanthology.org/2026.acl-long.11/. Dataset: https://huggingface.co/datasets/OracleY/ConfRAG.
- Correct title: "Benchmarking LLM's Capability in Reasoning over Conflicting Web References", ACL 2026 (long), pages 303–322. Authors: Yizhen Yuan, Rui Kong, Dongze Li, Yuanchun Li, Yunxin Liu.
- Name clash: an unrelated arXiv paper 2506.07309 by Huang et al. is also titled "ConfRAG" (Confidence-Guided RAG). Do not mix them up.
- Claims: The benchmark has 1,814 real questions, each with about 9.6 retrieved web paragraphs, and about 57% involve explicit contradictions. It scores answer clustering, answer coverage and reason coverage. Strong models, including GPT-4.1 and Claude-3.7-Sonnet, show large gaps.
- Relation: This is a realistic web-scale benchmark for contradiction-aware QA and a possible extra evaluation set. Its focus is organising conflicting viewpoints, whereas ours is suppressing misinformation.

### hou2024wikicontradict: WikiContradict. VERIFIED
- URL: https://proceedings.neurips.cc/paper_files/paper/2024/hash/c63819755591ea972f8570beffca6b1b-Abstract-Datasets_and_Benchmarks_Track.html (NeurIPS 2024 D&B, vol. 37, pp. 109701–109747). arXiv: 2406.13805.
- Authors: Yufang Hou, Alessandra Pascale, Javier Carnerero-Cano, Tigran Tchrakian, Radu Marinescu, Elizabeth Daly, Inkit Padhi, Prasanna Sattigeri. These follow the proceedings; OpenReview gives the middle initials "Tigran T." and "Elizabeth M.".
- Claims: The benchmark is built from real Wikipedia passages that contradict each other while coming from equally trustworthy sources. LLMs often fail to acknowledge both sides unless the prompt explicitly says the passages may conflict.
- Relation: This is earlier evidence that telling the model about a conflict changes its behaviour. In their setting both sides are legitimate. In ours one side is misinformation, which is why a neutral flag can backfire for us.

### ming2025faitheval: FaithEval. VERIFIED
- URL: https://openreview.net/forum?id=UeVx6L59fg (ICLR 2025 poster). arXiv: 2410.03727.
- Authors: Yifei Ming, Senthil Purushwalkam, Shrey Pandit, Zixuan Ke, Xuan-Phi Nguyen, Caiming Xiong, Shafiq Joty.
- Claims: The benchmark tests context faithfulness under unanswerable, inconsistent and counterfactual contexts. Even strong LLMs often do not stay faithful to the context, and model size does not reliably fix this.
- Relation: MADAM-RAG reports results on it. Its inconsistent-context subset is close to our setting.

### gross2026ragability: Ragability. VERIFIED (full title added)
- URL: https://aclanthology.org/2026.lrec-1.182/ (LREC 2026, pp. 2323–2333).
- Title: "Ragability Benchmark: A Dataset and Library to Test LLMs on Inter-context Conflicts". Authors: Stephanie Gross, Johann Petrak, Brigitte Krenn.
- Claims: The benchmark tests 7 LLMs on inter-context conflicts that need implicit reasoning, using fantasy entities so prior knowledge cannot help. Models detect conflicts reasonably well but struggle to answer content questions. Hinting that a contradiction may exist improves conflict identification on contradictory contexts but significantly hurts it on non-contradictory ones.
- Relation: This is the direct precedent for our false-flag analysis. A contradiction hint has costs when it is wrong. We extend this from "identifying the conflict" to downstream answer accuracy. We also separate neutral hints from directional hints and from removal, and show that the neutral hint can hurt even when it is correct.

### gokul2025contradiction: context validation. VERIFIED
- URL: https://arxiv.org/abs/2504.00180. arXiv only; OpenReview shows an ACL ARR Feb 2025 submission but no accepted venue.
- Title: "Contradiction Detection in RAG Systems: Evaluating LLMs as Context Validators for Improved Information Consistency". Authors: Vignesh Gokul, Srikanth Tenneti, Alwarappan Nakkiran.
- Claims: The paper synthesises several contradiction types in retrieved sets and tests LLMs as detectors. Detection stays hard and varies by contradiction type. Chain-of-thought helps some models and hurts others.
- Relation: This motivates our single-call LLM detector baseline and the point that detector quality is imperfect and type-dependent.

### ye2026tcr: TCR. VERIFIED (title found; AAAI 2026 confirmed)
- URL: https://doi.org/10.1609/aaai.v40i40.40740 (Crossref: Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, no. 40, pp. 34423–34431). arXiv: 2601.06842.
- Title: "Seeing through the Conflict: Transparent Knowledge Conflict Handling in Retrieval-Augmented Generation". Authors: Hua Ye, Siyuan Chen, Ziqi Zhong, Canran Xiao, Haoliang Zhang, Yuhan Wu, Fei Shen.
- Claims: TCR is a plug-in. Dual contrastive encoders separate semantic match from factual consistency, a self-answerability score estimates confidence in internal memory, and these scalar signals reach the generator through a small soft prompt. It reports better conflict detection (+5 to 18 F1) and fewer cases where misleading context overrides the answer.
- Relation: This is another way to give the generator conflict information, through continuous signals in a soft prompt rather than text flags. It needs training, and our flags do not.

### hossain2026evidentialrag: EvidentialRAG. VERIFIED (title found)
- URL: https://arxiv.org/abs/2607.10491 (v1, 2026-07-11, cs.LG). arXiv only.
- Title: "EvidentialRAG: Quantifying and Mitigating Information Conflict in Multi-Source Retrieval-Augmented Generation via Evidential Deep Learning". Authors: S M Asif Hossain, Ruksat Khan Shayoni, M. F. Mridha. The abstract calls the method "ERAG".
- Claims: The method maps chunk-level claim support to Dirichlet evidence and fuses it with a conflict-preserving Dempster–Shafer rule. The fused uncertainty decides whether to answer directly, answer in conflict-aware mode, or abstain. It reports lower hallucination and better calibration than Corrective RAG on CRAG, ConflictQA and MuSiQue.
- Relation: This is an alternative to binary flags that routes on uncertainty. It is useful context for the "what to do with the detector output" question.

---

## Knowledge conflicts (foundational)

### longpre2021entity. VERIFIED
- https://aclanthology.org/2021.emnlp-main.565/ (EMNLP 2021).
- Claims: Substituting entities in context shows that QA models often fall back on memorised answers when the context conflicts with what they learned in training.
- Relation: This is the starting point for knowledge-conflict research. Ours is an inter-context conflict, not a context-memory one.

### chen2022rich. VERIFIED
- https://aclanthology.org/2022.emnlp-main.146/ (EMNLP 2022). Full title: "Rich Knowledge Sources Bring Complex Knowledge Conflicts: Recalibrating Models to Reflect Conflicting Evidence". Authors: Hung-Ting Chen, Michael Zhang, Eunsol Choi.
- Claims: The paper studies conflicts among multiple retrieved passages. Models tend to rely on a single passage, and the authors propose calibrating models to abstain when the evidence conflicts.
- Relation: This is an early study of inter-passage conflict, the same type we study.

### xie2024chameleon. VERIFIED
- https://arxiv.org/abs/2305.13300. The arXiv comment says "ICLR 2024 (Spotlight)". Authors: Jian Xie, Kai Zhang, Jiangjie Chen, Renze Lou, Yu Su.
- Claims: LLMs readily accept coherent, convincing external evidence even when it contradicts their memory. When given both supporting and conflicting evidence, they show confirmation bias toward their prior.
- Relation: This explains why coherent misinformation passages are persuasive and why only telling the model that a conflict exists may not be enough.

### xu2024survey. VERIFIED
- https://aclanthology.org/2024.emnlp-main.486/ (EMNLP 2024).
- Claims: The survey sorts knowledge conflicts into context-memory, inter-context and intra-memory types, and reviews causes, behaviours and mitigations.
- Relation: Use it for terminology. Our setting is inter-context conflict.

### lee2024ambigdocs. VERIFIED
- https://openreview.net/forum?id=mkYCfO822n (venue "COLM", the 2024 conference). arXiv: 2404.12447. Authors: Yoonsang Lee, Xi Ye, Eunsol Choi.
- Claims: The benchmark is built from Wikipedia disambiguation pages, with documents about different entities that share a name. LMs often merge entities or give ambiguous answers.
- Relation: This is the ambiguity component behind RAMDocs. Ambiguity-type "contradictions" are not misinformation, which is one reason flags can mislead.

### wang2025astute. CORRECTED (venue)
- https://aclanthology.org/2025.acl-long.1476/ (ACL 2025 main, long). arXiv: 2410.07176 (Oct 2024). Use the 2025 ACL entry, not "2024". Authors: Fei Wang, Xingchen Wan, Ruoxi Sun, Jiefeng Chen, Sercan Ö. Arık (the Anthology spells it "Arik, Sercan O").
- Claims: Astute RAG has the LLM elicit its own internal knowledge, then consolidate it with retrieved passages in a source-aware way and answer based on reliability. This improves robustness to imperfect retrieval.
- Relation: This is a source-reliability approach that uses no explicit detector. It contrasts with our detector-plus-flag pipeline.

---

## Additional related work (2024–2026)

### zou2025poisonedrag: PoisonedRAG. VERIFIED
- https://www.usenix.org/conference/usenixsecurity25/presentation/zou-poisonedrag (USENIX Security 2025, pp. 3827–3844). arXiv: 2402.07867.
- Claims: Injecting a handful of crafted texts (five per target question) into a corpus of millions makes RAG output attacker-chosen answers about 90% of the time. The defenses the authors evaluated were insufficient.
- Relation: This is the adversarial version of RAMDocs-style misinformation. It motivates suspect-passage removal as a defense, and it shows why false flags matter: an attacker could also trigger them.

### xiang2024robustrag: RobustRAG. VERIFIED (arXiv; workshop only)
- https://arxiv.org/abs/2405.15556. Authors: Chong Xiang, Tong Wu, Zexuan Zhong, David Wagner, Danqi Chen, Prateek Mittal. It appeared at the ICML 2024 NextGenAISafety workshop. An ICLR 2025 submission exists on OpenReview, but no acceptance was found, so cite as arXiv.
- Claims: An isolate-then-aggregate scheme answers from each passage separately and then securely aggregates the answers. This gives certifiable robustness when only a few passages are corrupted.
- Relation: Like MADAM-RAG, it isolates passages per document. It offers a principled alternative to removing passages based on a possibly wrong detector.

### wu2024clasheval: ClashEval. VERIFIED
- https://proceedings.neurips.cc/paper_files/paper/2024/hash/3aa291abc426d7a29fb08418c1244177-Abstract-Datasets_and_Benchmarks_Track.html (NeurIPS 2024 D&B, vol. 37, pp. 33402–33422). Authors: Kevin Wu, Eric Wu, James Zou.
- Claims: The benchmark has over 1,200 questions with perturbed retrieved content. LLMs often adopt incorrect context over a correct prior, and how much they do so depends on how strong the prior is and how large the perturbation is.
- Relation: This is context-memory conflict, and it shows that models repeat wrong context. That fits our finding that the model repeats misinformation under neutral flags.

### pan2024credibility: Credibility-aware Generation. VERIFIED
- https://aclanthology.org/2024.emnlp-main.1109/ (EMNLP 2024).
- Claims: The method attaches credibility scores to retrieved contexts and trains models to weigh them, which improves robustness to flawed retrieval.
- Relation: This is the closest analogue to our directional flags. Both tell the model which source to trust, but they fine-tune and we only prompt.

### su2024conflictbank: ConflictBank. VERIFIED
- https://proceedings.neurips.cc/paper_files/paper/2024/hash/baf4b960d118f838ad0b2c08247a9ebe-Abstract-Datasets_and_Benchmarks_Track.html (NeurIPS 2024 D&B, vol. 37, pp. 103242–103268). Note the final "s": the proceedings title ends "…in LLMs", while the arXiv title ends "…in LLM".
- Claims: This is a large benchmark of conflicts in retrieved knowledge, in model knowledge, and between the two, covering misinformation, temporal and semantic causes.
- Relation: It is an alternative evaluation resource, and its misinformation-type conflicts are close to RAMDocs.

### chen2026contextcompliance: Context compliance. VERIFIED (arXiv preprint)
- https://arxiv.org/abs/2605.14473 (v4, July 2026). Authors: Yihang Chen, Pin Qian, Su Wang, Sipeng Zhang, Huan Xu, Shuhuai Lin, Xinpeng Wei.
- Claims: The paper argues that answer accuracy hides whether the model followed the context or its prior. It proposes an inference-time decomposition that elicits the contextual answer and the prior answer separately and isolates the conflicting premise.
- Relation: This supports our argument that the evaluation should look at behaviour (whether the model repeats misinformation), not only at accuracy or detector F1. It is an optional cite and only a preprint.

---

## RAG systems

### lewis2020rag. VERIFIED
- https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html (NeurIPS 2020, vol. 33, pp. 9459–9474). This is the original RAG paper.

### asai2024selfrag. VERIFIED
- https://openreview.net/forum?id=hSyW5go0v8 (ICLR 2024 oral).
- Claims: The model is trained to decide when to retrieve and to critique the passages it retrieves and its own outputs using reflection tokens.
- Relation: This is a baseline family that filters passages implicitly. ConflictRAG also compares against it.

### yan2024crag: Corrective RAG. VERIFIED (arXiv only)
- https://arxiv.org/abs/2401.15884. OpenReview shows a withdrawn ICLR 2025 submission and an ARR 2026 submission, with no accepted venue, so cite as arXiv.
- Claims: A lightweight retrieval evaluator scores the quality of the retrieved documents. Depending on the score, the system uses, refines, or replaces them with web search.
- Relation: This is document-level quality gating, which is similar in spirit to suspect-passage removal.

---

## Models, datasets, NLI

- **kocisky2018narrativeqa**: VERIFIED. https://aclanthology.org/Q18-1023/ (TACL vol. 6, 2018, pp. 317–328). A reading-comprehension benchmark over books and scripts.
- **reimers2019sbert**: VERIFIED. https://aclanthology.org/D19-1410/ (EMNLP-IJCNLP 2019). Siamese BERT sentence embeddings; cite for MiniLM/sentence-transformers encoders.
- **he2023debertav3**: VERIFIED. https://arxiv.org/abs/2111.09543. The arXiv comment says it was published at ICLR 2023. Cite as the backbone of the NLI cross-encoder.
- **gururangan2018annotation**: VERIFIED. https://aclanthology.org/N18-2017/ (NAACL 2018 short). Shows that hypothesis-only models can solve NLI datasets well above chance because of annotation artifacts. Relation: this is why an NLI cross-encoder's pair-level F1 may not transfer to RAG passages.
- **openai2024gpt4o**: VERIFIED. https://arxiv.org/abs/2410.21276 ("GPT-4o System Card", OpenAI, Oct 2024). arXiv lists "OpenAI" followed by hundreds of named authors. The bib truncates the list with "and others", which is standard practice.

## Evaluation and statistics

- **zheng2023judging**: VERIFIED. NeurIPS 2023 D&B track, vol. 36, pp. 46595–46623. https://proceedings.neurips.cc/paper_files/paper/2023/hash/91f18a1287b398d378ef22505bf41832-Abstract-Datasets_and_Benchmarks.html. Cite for the LLM-judge methodology and its known biases.
- **holm1979simple**: VERIFIED. Sture Holm, "A Simple Sequentially Rejective Multiple Test Procedure", Scandinavian Journal of Statistics 6(2):65–70, 1979. JSTOR 4615733 (https://www.jstor.org/stable/4615733). Confirmed through the JSTOR stable id quoted in several catalog sources; I did not read the full JSTOR text.
- **efron1993bootstrap**: VERIFIED. Bradley Efron and Robert J. Tibshirani, *An Introduction to the Bootstrap*, Chapman & Hall, New York, 1993. Monographs on Statistics and Applied Probability 57, ISBN 0412042312. Library catalogs confirm the 1993 Chapman & Hall edition. Crossref also has a Springer-US 1993 DOI (10.1007/978-1-4899-4541-9) and a later CRC DOI (10.1201/9780429246593); I left both DOIs out to avoid mixing up editions.
- **dror2018hitchhiker**: VERIFIED. https://aclanthology.org/P18-1128/ (ACL 2018).

---

## Not verified — do not cite

Every item on the planning list was found, so nothing is in this section. Unconfirmed venue claims are stated in the entries themselves; those papers are cited as arXiv only:
- CONFLICTS (Cattan et al.): no venue found.
- ConflictRAG: "submitted to IEEE SMC 2026", acceptance not confirmed.
- Gokul et al.: ARR submission only.
- Corrective RAG: no accepted venue.
- RobustRAG: workshop paper only.

Do not describe any of these as conference papers unless a proceedings record appears later.
