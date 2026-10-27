#!/usr/bin/env bash
# Build the anonymised supplementary-material archive for review: code, tests, pre-registration,
# run outputs and paper sources. Excludes git history, logs, the LLM response cache, and project
# notes that name the authors' repository. Fails if anything identifying slips through.
#   bash paper/make_supplement.sh        -> paper/supplement.zip
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=paper/supplement.zip
TMP=$(mktemp -d)
DEST="$TMP/supplement"
mkdir -p "$DEST"

rsync -a --exclude '__pycache__' --exclude 'CONTEXT.md' --exclude 'A4_TRACE.md' \
  experiments/ "$DEST/experiments/"
rsync -a --exclude '__pycache__' tests/ "$DEST/tests/"
rsync -a --exclude '*.log' --exclude 'llm_cache.sqlite' \
  --exclude 'holdout_main/results.jsonl' --exclude 'holdout_contains/results.jsonl' \
  runs/ "$DEST/runs/"
mkdir -p "$DEST/paper"
rsync -a paper/main.tex paper/references.bib paper/acl.sty paper/acl_natbib.bst \
  paper/sections paper/generated "$DEST/paper/"
cp requirements-conflict.txt setup_env.sh "$DEST/"
cat > "$DEST/README.md" <<'EOF'
# Supplementary material

- `experiments/conflict/`: the harness (detectors, prompts, runner, metrics, statistics).
  `PREREGISTRATION.md` was committed before any confirmatory run; `splits/` holds the pilot and
  hold-out question ids.
- `runs/`: every run used in the paper (per-call responses, API token counts, detector outputs,
  configs). Large results are `results.jsonl.gz`. `runs/README.md` says what each run is.
- `paper/`: LaTeX sources. Every number, table and figure in `paper/generated/` is produced by
  `python -m experiments.conflict.paper_stats --out paper/generated` from `runs/`.
- `tests/`: `python -m pytest -q tests` (data-dependent tests need `bash setup_env.sh`).
  `test_caag_parity.py::test_source_is_verbatim` compares `caag_core.py` with the original system's
  source, which is withheld for anonymity, so that one test is expected to fail here.

Benchmarks are fetched at pinned commits by `setup_env.sh` (RAMDocs, CONFLICTS); WikiContradict
is loaded from Hugging Face at the revision pinned in `experiments/conflict/data.py`. ConflictBank is
not redistributed (no licence file in its release): download the pinned QA file with the command in
`experiments/conflict/prepare_conflictbank.py`, then run that script to rebuild the 400 evaluation items.
EOF

# anonymity check: no author names, usernames, repo names or home paths
if grep -rIlE '12Parker|12camparker|cameronparker|/Users/|LongContextRAG' "$DEST"; then
  echo "ANONYMITY CHECK FAILED: the files above contain identifying strings" >&2
  exit 1
fi
rm -f "$OUT"
(cd "$TMP" && zip -qr - supplement) > "$OUT"
rm -rf "$TMP"
echo "wrote $OUT ($(du -h "$OUT" | cut -f1))"
