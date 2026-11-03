#!/usr/bin/env bash
# One-shot setup for experiments/conflict. Safe to re-run.
#   bash setup_env.sh            # venv + deps + data + checks
#   bash setup_env.sh --live     # also makes one tiny real API call (~$0.00001)
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3.12}
command -v "$PY" >/dev/null || PY=python3
"$PY" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,13), sys.version' \
  || { echo "Need Python 3.10-3.13 (set PYTHON=/path/to/python)"; exit 1; }

[ -d .venv-conflict ] || "$PY" -m venv .venv-conflict
source .venv-conflict/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements-conflict.txt

# Benchmarks, pinned to the commits the harness was written against.
mkdir -p data/external
fetch () {  # name url commit
  if [ ! -d "data/external/$1" ]; then
    git clone -q "$2" "data/external/$1"
    git -C "data/external/$1" checkout -q "$3"
  fi
}
fetch RAMDocs        https://github.com/HanNight/RAMDocs.git                       d44454c9ebb0bf513d8236a03decc9fb58704cad
fetch rag_conflicts  https://github.com/google-research-datasets/rag_conflicts.git 81ba921dd684a93db41a7e9dda6b6a7c67348a88

[ -f .env ] || { cp env.template .env; echo ">> Created .env from env.template: add your OPENAI_API_KEY"; }

python -m pytest -q tests
python -m experiments.conflict.check_env "$@"
