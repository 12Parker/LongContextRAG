# Paper: "Which Side Is Wrong? Contradiction Flags in RAG Need Direction, Not Just Detection"

ACL-format long paper (review mode, anonymous). The body is about 7.3 pages; Limitations, Ethics, References and the
appendix follow.

## Build

```bash
python -m experiments.conflict.paper_stats --out paper/generated   # numbers, tables, figures (from runs/)
cd paper && tectonic -X compile main.tex                           # -> main.pdf
bash paper/make_supplement.sh                                      # -> paper/supplement.zip (anonymised)
```

`paper/generated/` is written only by `paper_stats.py` and `paper_assets.py`; never edit it by hand. Every number in
the text is a macro from `generated/numbers.tex`.

## Layout

- `main.tex`, `sections/*.tex`: the paper. Results text cites `\macro`s from `generated/numbers.tex`.
- `references.bib`: every entry was checked against a primary source (see `related_work_notes.md`). Papers with
  no confirmed venue are cited as arXiv.
- `acl.sty`, `acl_natbib.bst`: official ACL style files (acl-org/acl-style-files, master).
  `acl_template_reference.tex` is the upstream template, kept for reference.

## Before submitting (needs a human)

1. **Read the whole paper and take responsibility for it.** The experiments, analysis and draft were produced with
   an AI assistant (Claude Code). ACL and ARR policy requires authors to disclose AI assistance in the Responsible
   NLP checklist and to take full responsibility for the content. Check every claim against `generated/hypotheses.json`.
2. **Fill in the ARR Responsible NLP checklist** in the submission form. Relevant facts:
   - artifacts and licences: RAMDocs (MIT), CONFLICTS (Apache-2.0), WikiContradict (MIT, per its Hugging Face card);
   - compute: CPU only, plus $27.42 of API spend (`runs/COSTS.md`);
   - no human subjects;
   - AI assistance: yes, see item 1.
3. **Venue and deadline.** Check the current ARR cycle dates and the target venue's page limit (8 pages for long
   papers in review; this draft fits).
4. **Upload `supplement.zip`** (42 MB) as software/data. The build script refuses to write it if any identifying
   string remains.
5. **Optional, but strengthens the paper** (see Limitations):
   - hand-label the 163 suspect RAMDocs documents (`runs/labelling/ramdocs_suspect_docs.csv`, then
     `--ramdocs-labels hand`);
   - label the 166 Phase-0 CAAG flags (`runs/phase0/flags_to_label.csv`);
   - add an open-weights generator through `--base-url`;
   - add a different-family judge for CONFLICTS' opinion items (`judge.py`).
6. **Camera-ready:** de-anonymise (author block, the code-release footnote in `sections/intro.tex`) and switch
   `\usepackage[review]{acl}` to `\usepackage[final]{acl}`.

## Where the claims come from

| Claim | Source |
|---|---|
| Hypothesis tests H1–H9 | `generated/hypotheses.json`, `generated/tab_hypotheses.tex` |
| Main RAMDocs table | `runs/holdout_main/results.jsonl.gz` |
| gpt-4.1 replication | `runs/holdout_gpt41/` |
| WikiContradict | `runs/wikicontradict_main/`, `runs/wikicontradict_gpt41/` |
| Label robustness | `runs/holdout_contains/` |
| Detector quality, θ sweep | `runs/holdout_detect/` |
| Pre-registration (frozen before confirmatory runs) | `experiments/conflict/PREREGISTRATION.md`, commit `a6f7124` |
