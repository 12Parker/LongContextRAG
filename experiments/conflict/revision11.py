"""
Tenth revision (PREREGISTRATION §20): re-analyses of outputs already seen, after an external review. No API calls.

  rev11_prior     R1-R5, R8   text classifier on ConflictBank, direction and follow rates by prior, response length
  rev11_account   R6, R7      prospective-test summaries; gpt-6.1-terra's two plain draws
  rev11_scoring   R9-R12      missing final answers, H13 sensitivity, seed dependence, one-way misinformation
  rev11_policy    R13-R19     HealthContradict stance, RAGuard key vs flag, single-entity, risk-coverage, answer mix,
                              prevalence, H4b second form

Each module also runs alone (python -m experiments.conflict.rev11_<name>) and writes runs/rev11/rev11_<name>.json.
Every result is exploratory and outside the FDR count (§20).
"""
from __future__ import annotations

import json

MODULES = ("prior", "account", "scoring", "policy")


def revision11(R):
    import importlib
    plain = json.loads(json.dumps(R, default=float))   # the modules take R as hypotheses.json holds it
    out = {}
    for m in MODULES:
        out.update(importlib.import_module(f".rev11_{m}", __package__).compute(plain))
    R["rev"]["r11"] = out
