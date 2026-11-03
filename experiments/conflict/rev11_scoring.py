"""
Eleventh revision, scoring (PREREGISTRATION.md §20.3, "PR§20"): R9-R12. Exploratory re-analyses of stored outputs, run
after §20 was committed and pushed; no API call, outside the FDR count. The A1 audit sheet is exported by
export_audit11.py from a1_population() below.

  R9   per run and condition: responses with no final-answer line (no metrics._FINAL match, no MADAM list), responses
       at the completion cap, and both
  R10  H13's two gpt-4.1 tests without the questions on which either arm lacks a final-answer line
  R11  seed dependence: pooled H23 / H30a (matcher) and Table 2's neutral-flag contrasts from one seed at a time
  R12  Table 2 under one matcher: the misinformation share with one-way matching from the stored pred_answers, and
       H21(a) and the exploratory risk family recomputed on it
  A1   the sampling frame of the human audit (judge19's re-scored responses), with its per-arm disagreement rates

    python -m experiments.conflict.rev11_scoring        # writes runs/rev11/rev11_scoring.json
"""
from __future__ import annotations

import inspect
import json
import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

from .analyze import per_question
from .baselines import BASELINE_MODES
from .jsonl import read_results, results_path
from .llm import NO_TEMPERATURE_MODELS, REASONING_MAX_TOKENS, REASONING_MODELS, OpenAILLM
from .metrics import _FINAL, _MADAM, _contains, _matches, normalize_answer

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
OUT = RUNS / "rev11/rev11_scoring.json"
PRIMARY, ROBUST = "ramdocs_strict_oneway", "ramdocs_strict"
G4 = ("gpt-4o-mini", "gpt-6-luna", "gpt-6.1-terra", "claude-haiku-5-5")
SEEDS = (0, 1, 2)
# the generator client's output cap (config.json does not record it): llm.OpenAILLM's default, raised to
# REASONING_MAX_TOKENS for reasoning models and Claude
DEFAULT_CAP = inspect.signature(OpenAILLM.__init__).parameters["max_tokens"].default
BIG_CAP_MODELS = REASONING_MODELS | NO_TEMPERATURE_MODELS
# paper_stats.revision2's sets, in its insertion order (revision7._pool draws benchmarks in that order)
R2_RUNS = (("gpt-4.1|seed0", "holdout_gpt41_round2"), ("gpt-4o-mini|qwen-detector", "holdout_qwendet"),
           ("CB:gpt-4o-mini", "cb_main"), ("CB:gpt-4.1", "cb_gpt41"), ("CB:Qwen2.5-7B", "cb_qwen"),
           ("CBs:gpt-4o-mini", "cb_single"), ("CBs:gpt-4.1", "cb_single_gpt41"), ("CBs:Qwen2.5-7B", "cb_single_qwen"),
           ("CBn:gpt-4o-mini", "cb_minimal"), ("CBmaj:gpt-4o-mini", "cbmaj_single"),
           ("CBmaj:gpt-4.1", "cbmaj_single_gpt41"), ("CBs:gpt-6-luna", "cb_single_luna"),
           ("CBs:gpt-6.1-terra", "cb_single_terra"), ("RD:gpt-6-luna", "holdout_luna"),
           ("RD:gpt-6.1-terra", "holdout_terra"), ("CBmaj:gpt-6-luna", "cbmaj_single_luna"),
           ("CBmaj:gpt-6.1-terra", "cbmaj_single_terra"), ("CBs:claude-haiku-5-5", "cb_single_haiku"),
           ("CBs:claude-sonnet-5-5", "cb_single_sonnet"), ("RD:claude-haiku-5-5", "holdout_haiku"),
           ("RD:claude-sonnet-5-5", "holdout_sonnet"), ("CBmaj:claude-haiku-5-5", "cbmaj_single_haiku"),
           ("CBf:gpt-4o-mini", "cb_forced_mini"), ("CBf:gpt-6-luna", "cb_forced_luna"),
           ("CBf:gpt-6.1-terra", "cb_forced_terra"), ("CBf:claude-haiku-5-5", "cb_forced_haiku"),
           ("RG:gpt-4o-mini", "rg_mini"), ("RG:gpt-6-luna", "rg_luna"), ("RG:claude-haiku-5-5", "rg_haiku"),
           ("RG:gpt-6.1-terra", "rg_terra"), ("RG:claude-sonnet-5-5", "rg_sonnet"),
           ("RD:gpt-4o-mini", "holdout_main"))   # main rows + holdout_extra; added after the loop in revision2
CTL = {f"{pre}:{gk}": f"ctl_{d}_{g}" for g, gk in (("mini", "gpt-4o-mini"), ("luna", "gpt-6-luna"), ("terra", "gpt-6.1-terra"))
       for d, pre in (("rd", "RD"), ("cb", "CBs"), ("mj", "CBmaj"))}
# Table 2 (paper_assets.tab_risk): panels and generators
TAB2_PRE = ("RD", "CB", "CBs", "CBf")
TAB2_GENS = ("gpt-4o-mini", "gpt-6-luna", "claude-haiku-5-5", "gpt-6.1-terra", "claude-sonnet-5-5", "gpt-4.1", "Qwen2.5-7B")
FLAG, VKEY, KEY, ODIR = "llm_dir+flag_directional", "llm_dir+verdict_key", "answer_key", "oracle+flag_directional"


def _ps():
    from . import paper_stats as ps
    return ps


# --------------------------------------------------------------------------- per-row fields
def missing(text) -> bool:
    """No final-answer line: no match for metrics._FINAL and no MADAM list (what extract_answers falls back from)."""
    text = text or ""
    return not (_FINAL.search(text) or _MADAM.search(text))


def cap(model: str, cond: str, calls: int = 1):
    """Effective output cap of the call that wrote the response; None for multi-call conditions (MADAM, Astute,
    TrustRAG), whose completion_tokens sum over calls."""
    if calls > 1 or cond in ("madam",) + BASELINE_MODES:
        return None
    return max(DEFAULT_CAP, REASONING_MAX_TOKENS) if model in BIG_CAP_MODELS else DEFAULT_CAP


def unknown(pred_answers) -> bool:
    return any(a.strip().lower().rstrip(".") == "unknown" for a in pred_answers or [])


def oneway_wrong(pred_answers, wrong_answers) -> float:
    """R12: some wrong answer is a whole-word span of the joined prediction (metrics._contains, as the one-way strict
    score does), never the reverse."""
    pred = normalize_answer(" ; ".join(pred_answers or []))
    return float(any(_contains(pred, w) for w in wrong_answers))


_EX: dict = {}


@lru_cache(maxsize=None)
def _examples(run: str) -> dict:
    from .judge10 import _examples as ex
    return ex(RUNS / run, _EX)


_ROWS: dict = {}


def run_rows(run: str) -> tuple:
    """A run's rows (cached), tagged with the run, with the derived fields: _missing, misinfo (= wrong_rate,
    bidirectional), misinfo_oneway, abstain, selective (as revision4 for H21), and _chk (the stored one-way strict
    score and wrong_rate reproduced from pred_answers)."""
    if run in _ROWS:
        return _ROWS[run]
    rows = read_results(RUNS / run)
    ex = _examples(run)
    for r in rows:
        r["_run"] = run
        r["_missing"] = missing(r.get("response"))
        e = ex.get(r["id"])
        if e is None or "wrong_rate" not in r:
            continue
        ab = unknown(r.get("pred_answers"))
        r["misinfo"], r["abstain"], r["selective"] = r["wrong_rate"], float(ab), None if ab else r[PRIMARY]
        r["misinfo_oneway"] = oneway_wrong(r["pred_answers"], e.wrong_answers)
        pred = normalize_answer(" ; ".join(r["pred_answers"]))
        need = any if e.dataset == "poisonedrag" else all
        s1 = float(need([_contains(pred, a) for a in e.gold_answers]) and not r["misinfo_oneway"])
        w2 = float(any(_matches(r["pred_answers"], a) for a in e.wrong_answers))
        r["_chk"] = (s1 == r[PRIMARY], w2 == r["wrong_rate"])
    _ROWS[run] = tuple(rows)
    return _ROWS[run]


@lru_cache(maxsize=None)
def set_rows(key: str) -> tuple:
    """Rows of one paper_stats.revision2 set (same runs, merges and filters)."""
    run = dict(R2_RUNS)[key]
    rows = list(run_rows(run)) + (list(run_rows("holdout_extra")) if key == "RD:gpt-4o-mini" else [])
    if key == "gpt-4.1|seed0":
        rows = [r for r in rows if r["seed"] == 0]
    if key == "CBs:gpt-4o-mini":   # baselines ignore the skeleton
        rows += [r for r in run_rows("cb_main") if r["condition"] in ("astute", "trustrag")]
    if key in CTL and results_path(RUNS / CTL[key]):
        rows += [r for r in run_rows(CTL[key]) if r["condition"] != "plain"]
    return tuple(rows)


def set_keys() -> list[str]:
    return [k for k, run in R2_RUNS if results_path(RUNS / run)]


def _conds(rows) -> set:
    return {r["condition"] for r in rows}


# --------------------------------------------------------------------------- R9
def tally(rows, model: str) -> dict:
    """Per condition: n, missing, at_cap and both (counts and shares); missing split by conflict / clean."""
    acc = defaultdict(lambda: defaultdict(int))
    for r in rows:
        c = acc[r["condition"]]
        m = r["_missing"] if "_missing" in r else missing(r.get("response"))
        cp = cap(model, r["condition"], r.get("calls", 1))
        c["n"] += 1; c["missing"] += m
        if cp is not None:
            at = r.get("completion_tokens", 0) >= cp
            c["n_single"] += 1; c["missing_single"] += m; c["at_cap"] += at; c["both"] += m and at
        sub = {True: "conflict", False: "clean"}.get(r.get("has_conflict"))
        if sub:
            c[f"n_{sub}"] += 1; c[f"missing_{sub}"] += m
    out = {}
    for cond, c in acc.items():
        o = {k: int(v) for k, v in c.items()}
        o["missing_share"] = c["missing"] / c["n"]
        o["at_cap_share"] = c["at_cap"] / c["n_single"] if c["n_single"] else None
        o["both_share"] = c["both"] / c["n_single"] if c["n_single"] else None
        for sub in ("conflict", "clean"):
            if c[f"n_{sub}"]:
                o[f"missing_{sub}_share"] = c[f"missing_{sub}"] / c[f"n_{sub}"]
        out[cond] = o
    return out


def r9() -> dict:
    runs, by_model = {}, defaultdict(lambda: defaultdict(int))
    for d in sorted(p for p in RUNS.iterdir() if p.is_dir()):
        if not results_path(d):
            continue
        model = json.loads((d / "config.json").read_text())["model"]
        rows = _ROWS.get(d.name) or read_results(d)   # runs outside the sets are read once, not cached
        t = tally(rows, model)
        cp = sorted({cap(model, c, 1) for c in t if cap(model, c, 1)})
        runs[d.name] = {"model": model, "cap": cp[0] if cp else None, "file": results_path(d).name, "conditions": t}
        for v in t.values():
            for k in ("n", "missing", "n_single", "missing_single", "at_cap", "both"):
                by_model[model][k] += v.get(k, 0)
    cells = [(run, c, v) for run, x in runs.items() for c, v in x["conditions"].items()]
    above = sorted(([run, c, v["n"], v["missing_share"], v["at_cap_share"]] for run, c, v in cells
                    if v["missing_share"] > 0.01), key=lambda x: -x[3])
    tot = {k: sum(v.get(k, 0) for _, _, v in cells) for k in ("n", "missing", "n_single", "missing_single", "at_cap", "both")}
    by_set = {}
    for key in set_keys():   # the main tables' cells (sets as paper_stats builds them)
        model = runs[dict(R2_RUNS)[key]]["model"]
        t = tally(set_rows(key), model)
        by_set[key] = {c: {k: v.get(k) for k in ("n", "missing_share", "missing_conflict_share", "at_cap_share")}
                       for c, v in t.items()}
    return {"cap_rule": f"config.json has no max_tokens; generator calls use llm.OpenAILLM's default {DEFAULT_CAP}, "
                        f"raised to {REASONING_MAX_TOKENS} for {sorted(BIG_CAP_MODELS)}; MADAM, Astute and TrustRAG "
                        "rows sum several calls, so their at-cap share is not defined (null)",
            "n_runs": len(runs), "totals": {**tot, "missing_share": tot["missing"] / tot["n"],
                                            "at_cap_share_of_missing_single_call": tot["both"] / max(tot["missing_single"], 1)},
            "by_model": {m: {**{k: int(x) for k, x in v.items()}, "missing_share": v["missing"] / v["n"],
                             "at_cap_share": v["at_cap"] / v["n_single"] if v["n_single"] else None}
                         for m, v in by_model.items()},
            "n_cells": len(cells), "n_cells_above_1pct": len(above), "above_1pct": above,
            "by_set": by_set, "runs": runs}


# --------------------------------------------------------------------------- R10
H13_41 = (("RAMDocs, gpt-4.1: neutral flag + instruction vs plain (conflict)", "plain", "oracle+flag_instruct_v2", 0),
          ("RAMDocs, gpt-4.1: dir. flag + instruction > neutral flag + instruction (conflict)",
           "oracle+flag_instruct_v2", "oracle+flag_directional_instruct_v2", +1))


def r10(R) -> dict:
    from .stats import holm
    ps = _ps()
    rows = set_rows("gpt-4.1|seed0")
    out = {"seeds": sorted({r["seed"] for r in rows})}
    for m in (PRIMARY, ROBUST):
        fam = R["rev"]["H13"][m]; res = {}
        for tk, a, b, _ in H13_41:
            drop = {r["id"] for r in rows if r["condition"] in (a, b) and r["_missing"]}
            conf = {r["id"] for r in rows if r.get("has_conflict")}
            full = ps.compare(list(rows), a, b, True, m)
            sub = ps.compare([r for r in rows if r["id"] not in drop], a, b, True, m)
            reg = fam[tk]
            res[tk] = {"registered": {k: reg[k] for k in ("diff", "ci_low", "ci_high", "p", "p_holm", "supported", "n")},
                       "reproduced": bool(abs(full["diff"] - reg["diff"]) < 1e-12 and full["p"] == reg["p"]),
                       "n_dropped_conflict": len(drop & conf),
                       "n_missing_rows_conflict": {c: sum(r["_missing"] for r in rows if r["condition"] == c and r.get("has_conflict"))
                                                   for c in (a, b)},
                       **sub}
        p = {k: v["p"] for k, v in fam.items()}
        p.update({tk: res[tk]["p"] for tk in res})
        adj = holm(p)   # the H13 family with the two sensitivity p-values in place of the registered ones
        for tk, _, _, want in H13_41:
            res[tk]["p_holm"] = adj[tk]
            res[tk]["supported"] = bool(adj[tk] < 0.05 and (want == 0 or np.sign(res[tk]["diff"]) == want))
        out[m] = res
    return out


# --------------------------------------------------------------------------- R11 (and the A1 frame)
def pooled_cells() -> dict:
    """revision7.pooled's cells, (label, bench, rows, a, b): H30a = LLM directional flag minus verdict key over the H24
    cells with a vkey run (sorted as H24's cells); H23 = oracle directional flag minus answer key over the RD / CBs /
    CBmaj sets with an answer-key arm (revision2's order)."""
    from .revision5 import SHORT_GEN
    full = {v: k for k, v in SHORT_GEN.items()}
    h30 = []
    for d in RUNS.glob("vkey_cb_*"):
        if not results_path(d):
            continue
        gs, ds = d.name[len("vkey_cb_"):].split("_")
        g, det = full[gs], full.get(ds, "gpt-4o-mini")
        src = set_rows(f"CBs:{g}") if ds == "mini" else (run_rows(f"cross_cb_{gs}_{ds}") if results_path(RUNS / f"cross_cb_{gs}_{ds}") else ())
        rows = [r for r in src if r["condition"] == FLAG] + [r for r in run_rows(d.name) if r["condition"] == VKEY]
        if {FLAG, VKEY} <= _conds(rows):
            h30.append(((g, det), f"{g}|{det}", "CB", rows, VKEY, FLAG))
    h30 = [c[1:] for c in sorted(h30, key=lambda c: c[0])]
    h23 = []
    for k in set_keys():
        if k.startswith(("RD:", "CBs:", "CBmaj:")):
            rows = set_rows(k)
            if KEY in _conds(rows):
                h23.append((k, k.split(":")[0], [r for r in rows if r["condition"] in (KEY, ODIR)], KEY, ODIR))
    return {"H30a": h30, "H23": h23}


def _arm_seeds(rows, a, b) -> list:
    return sorted({r["seed"] for r in rows if r["condition"] == a} & {r["seed"] for r in rows if r["condition"] == b})


def seed_cells(cells, s, fixed=False):
    """(bench, {qid: b - a}) per cell from seed s alone; cells without seed s are left out, or with fixed=True a
    single-seed cell keeps its one seed."""
    from .revision7 import _diff
    out = []
    for _, bench, rows, a, b in cells:
        have = _arm_seeds(rows, a, b)
        use = s if s in have else (have[0] if fixed and len(have) == 1 else None)
        if use is None:
            continue
        d = _diff([r for r in rows if r["seed"] == use], a, b, PRIMARY)
        if d:
            out.append((bench, d))
    return out


def _brief(p):
    return {k: p[k] for k in ("diff", "ci_low", "ci_high", "ci90_low", "ci90_high", "p", "equivalent", "n_cells")}


def _range(xs):
    xs = [x for x in xs if x is not None]
    return [float(min(xs)), float(max(xs))] if xs else None


def r11_pooled(R, cells) -> dict:
    from .revision7 import _diff, _pool
    names = {"H30a": "H30a: LLM dir. flag minus verdict key", "H23": "H23: oracle dir. flag minus answer key"}
    out = {}
    for fam, cs in cells.items():
        allc = [(bench, _diff(rows, a, b, PRIMARY)) for _, bench, rows, a, b in cs]
        full = _pool(allc)
        reg = R["rev"]["pooled7"][names[fam]]
        o = {"all": _brief(full), "reproduced": bool(abs(full["diff"] - reg["diff"]) < 1e-9 and full["ci_low"] == reg["ci_low"]),
             "cell_seeds": {lab: _arm_seeds(rows, a, b) for lab, _, rows, a, b in cs},
             "by_seed": {}, "by_seed_fixed": {}}
        o["single_seed_cells"] = sorted(k for k, v in o["cell_seeds"].items() if len(v) == 1)
        for s in SEEDS:
            for tag, fixed in (("by_seed", False), ("by_seed_fixed", True)):
                c = seed_cells(cs, s, fixed)
                if c:
                    o[tag][str(s)] = _brief(_pool(c))
        for tag in ("by_seed", "by_seed_fixed"):
            o[tag.replace("by_seed", "range")] = _range([v["diff"] for v in o[tag].values()])
        out[fam] = o
    return out


def contrast(rows, a: str, b: str, seed=None) -> dict | None:
    """b minus a on conflict questions, in points: correct (one-way strict), misinformation (bidirectional, as Table 2;
    and one-way, R12), abstention (paired means over shared questions) and selective accuracy (ratio, as H21(c))."""
    rs = [r for r in rows if r.get("has_conflict") and r["condition"] in (a, b) and (seed is None or r["seed"] == seed)]
    out = {}
    for name, m in (("correct", PRIMARY), ("misinfo", "wrong_rate"), ("misinfo_oneway", "misinfo_oneway"), ("abstain", "abstain")):
        A, B = per_question(rs, a, True, m), per_question(rs, b, True, m)
        ids = sorted(set(A) & set(B))
        if not ids:
            return None
        out[name] = 100 * float(np.mean([B[i] - A[i] for i in ids]))
    keep = set(ids)
    sel = {}
    for c in (a, b):
        ans = [r for r in rs if r["condition"] == c and r["id"] in keep and not r["abstain"]]
        sel[c] = sum(r[PRIMARY] for r in ans) / len(ans) if ans else None
    out["selective"] = 100 * (sel[b] - sel[a]) if None not in sel.values() else None
    out["n"] = len(ids)
    return out


def tab2_keys() -> list[str]:
    ks = set(set_keys())
    return [f"{p}:{g}" for p in TAB2_PRE for g in TAB2_GENS if f"{p}:{g}" in ks]


def r11_table2() -> dict:
    out, single = {}, []
    for key in tab2_keys():
        rows = set_rows(key)
        seeds = _arm_seeds(rows, "plain", "oracle+flag")
        if not seeds:
            continue
        o = {"seeds": seeds, "all": contrast(rows, "plain", "oracle+flag"),
             "by_seed": {str(s): contrast(rows, "plain", "oracle+flag", s) for s in seeds}}
        o["range"] = {m: _range([v[m] for v in o["by_seed"].values()]) for m in o["all"] if m != "n"}
        out[key] = o
        if len(seeds) == 1:
            single.append(key)
    return {"settings": out, "single_seed_settings": single}


# --------------------------------------------------------------------------- R12
def _mix(rows, cond):
    rr = [r for r in rows if r["condition"] == cond and r.get("has_conflict")]
    if not rr:
        return None
    n = len(rr)
    corr = sum(r[PRIMARY] == 1 for r in rr); unk = sum(r["abstain"] for r in rr)
    mis1 = sum(r[PRIMARY] != 1 and r["misinfo_oneway"] > 0 for r in rr)
    return {"n": n, "correct": corr / n, "misinfo": float(np.mean([r["wrong_rate"] for r in rr])),
            "misinfo_oneway": float(np.mean([r["misinfo_oneway"] for r in rr])), "abstain": unk / n,
            "selective": corr / (n - unk) if n > unk else None,
            "out_misinfo_oneway": mis1 / n,
            "out_abstain_oneway": sum(r[PRIMARY] != 1 and not r["misinfo_oneway"] and r["abstain"] for r in rr) / n}


def r12(R) -> dict:
    ps = _ps()
    r2 = R["rev"]["r2"]
    # check: the one-way strict score and the bidirectional wrong_rate, recomputed from pred_answers, equal the stored fields
    chk, seen = defaultdict(int), set()
    for k in set_keys():
        for r in set_rows(k):
            if id(r) in seen:   # cb_main's Astute / TrustRAG rows sit in two sets; count each row once
                continue
            seen.add(id(r))
            if "wrong_rate" in r and "misinfo_oneway" not in r:
                chk["rows_without_oneway"] += 1
            if "_chk" in r:
                chk["rows"] += 1; chk["strict_oneway_mismatch"] += not r["_chk"][0]; chk["wrong_rate_mismatch"] += not r["_chk"][1]
    # H21(a) on the one-way share (revision4's family: CBs and CBf, G4, neutral and directional flag vs plain)
    tests = {}
    for key in [f"{p}:{g}" for p in ("CBs", "CBf") for g in G4]:
        if key not in set_keys():
            continue
        rows = list(set_rows(key))
        for c, nm in (("oracle+flag", "neutral flag"), (ODIR, "dir. flag")):
            if c in _conds(rows):
                tests[f"{ps.set_label(key)}: {nm} vs plain (conflict)"] = ps.compare(rows, "plain", c, True, "misinfo_oneway")
    h21 = ps.family(tests, {k: 0 for k in tests})
    reg = R["rev"]["H21"]["misinfo"]
    for tk, v in h21.items():
        v["registered"] = {k: reg[tk][k] for k in ("diff", "ci_low", "ci_high", "p_holm", "supported")}
    # the exploratory risk family (assign_significance: every r2 risk/misinfo comparison, one Holm family) on the one-way share
    from .stats import holm
    explo = {}
    for key, v in r2.items():
        if key not in set_keys():
            continue
        rows = list(set_rows(key))
        for c, old in v.get("risk", {}).get("misinfo", {}).items():
            t = ps.compare(rows, "plain", c, True, "misinfo_oneway")
            explo[f"{key}|{c}"] = {**{k: t[k] for k in ("diff", "ci_low", "ci_high", "p", "n")},
                                   "bidir": {k: old.get(k) for k in ("diff", "p_adj", "sig")}}
    fam = {k: t["p"] for k, t in explo.items() if not (t["ci_low"] == t["ci_high"] == 0) and not math.isnan(t["p"])}
    adj = holm(fam)
    for k, t in explo.items():
        t["p_adj"] = adj.get(k, float("nan")); t["sig"] = bool(k in adj and adj[k] < 0.05)
    # Table 2 with one matcher: levels per setting and condition; bold as tab_risk (H21 for CBs / CBf, else the family)
    table, mix_err = {}, 0.0
    for key in tab2_keys():
        rows = set_rows(key); table[key] = {}
        for c in ("plain", "oracle+flag", ODIR):
            m = _mix(rows, c)
            if m is None:
                continue
            old = r2[key]["answer_mix"].get(c, {})
            mix_err = max(mix_err, abs(m["misinfo"] - old.get("wrong", m["misinfo"])),
                          abs(m["correct"] - old.get("out_correct", m["correct"])), abs(m["abstain"] - old.get("unknown", m["abstain"])))
            if c != "plain":
                pre = key.split(":")[0]
                if pre in ("CBs", "CBf"):
                    tk = f"{ps.set_label(key)}: {'neutral flag' if c == 'oracle+flag' else 'dir. flag'} vs plain (conflict)"
                    m["bold_misinfo_oneway"] = bool(h21.get(tk, {}).get("supported"))
                    m["bold_misinfo_bidir"] = bool(reg.get(tk, {}).get("supported"))
                else:
                    m["bold_misinfo_oneway"] = bool(explo.get(f"{key}|{c}", {}).get("sig"))
                    m["bold_misinfo_bidir"] = bool(r2[key].get("risk", {}).get("misinfo", {}).get(c, {}).get("sig"))
            table[key][c] = m
    flips = [k for k, t in explo.items() if t["sig"] != bool(t["bidir"]["sig"])]
    return {"check": {**chk, "table2_max_abs_diff_vs_stored_answer_mix": mix_err},
            "H21a_oneway": h21,
            "H21a_changed": [tk for tk, v in h21.items() if v["supported"] != v["registered"]["supported"]],
            "table2": table, "explo_oneway": explo, "explo_n": len(fam), "explo_sig_changed": flips}


# --------------------------------------------------------------------------- A1 sampling frame
A1_ARMS = (ODIR, KEY, FLAG, VKEY)


def a1_population(cells=None) -> list[dict]:
    """judge19's re-scored responses: conflict rows of the pooled H30a / H23 arms with a parsed judge verdict, one per
    (dataset, id, condition, seed, response key), first occurrence kept (judge19's order)."""
    from .judge10 import response_key
    from .revision10 import _judge_map
    J = _judge_map()
    cells = cells or pooled_cells()
    out, seen = [], set()
    for fam in ("H30a", "H23"):
        for lab, bench, rows, a, b in cells[fam]:
            for r in rows:
                if r["condition"] not in (a, b) or not r.get("has_conflict"):
                    continue
                rk = response_key(r["response"])
                key = (r["dataset"], r["id"], r["condition"], r["seed"], rk)
                j = J.get((r["dataset"], r["id"], rk))
                if key in seen or j is None:
                    continue
                seen.add(key)
                out.append({"dataset": r["dataset"], "id": r["id"], "run": r["_run"], "cell": lab, "condition": r["condition"],
                            "seed": r["seed"], "response_key": rk, "matcher": float(r[PRIMARY] >= 1), "judge_strict": float(j),
                            "response": r["response"]})
    return out


def a1_frame(R, pop) -> dict:
    agree = R["rev"].get("judge19", {}).get("agreement", {})
    out = {}
    for arm in A1_ARMS:
        P = [p for p in pop if p["condition"] == arm]
        n = len(P)
        jo = int(sum(p["judge_strict"] and not p["matcher"] for p in P)); mo = int(sum(p["matcher"] and not p["judge_strict"] for p in P))
        dq = {(p["dataset"], p["id"]) for p in P if p["matcher"] != p["judge_strict"]}
        out[arm] = {"n": n, "m_k": mo / n, "j_k": jo / n, "n_disagree": jo + mo, "n_judge_only": jo, "n_matcher_only": mo,
                    "n_disagree_questions": len(dq), "n_questions": len({(p["dataset"], p["id"]) for p in P}),
                    "matches_judge19": bool(arm in agree and agree[arm]["n"] == n and abs(agree[arm]["judge_only"] - jo / n) < 1e-12)}
    return out


# --------------------------------------------------------------------------- entry points
def _plain(x):
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return None if math.isnan(float(x)) else float(x)
    return x


def compute(R: dict) -> dict:
    cells = pooled_cells()
    pop = a1_population(cells)
    out = {"R12": r12(R), "R11": {"pooled": r11_pooled(R, cells), "table2": r11_table2()}, "R10": r10(R),
           "A1": a1_frame(R, pop), "R9": r9()}
    return _plain(out)


def main(argv=None):
    R = json.loads((ROOT / "paper/generated/hypotheses.json").read_text())
    out = compute(R)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True))
    r9_, r10_, r11_, r12_ = out["R9"], out["R10"], out["R11"], out["R12"]
    t = r9_["totals"]
    print(f"R9: {r9_['n_runs']} runs, {t['n']} rows; no final-answer line {t['missing']} ({100 * t['missing_share']:.2f}%), "
          f"of the single-call ones at the cap {100 * t['at_cap_share_of_missing_single_call']:.1f}%; {r9_['n_cells_above_1pct']} of "
          f"{r9_['n_cells']} run x condition cells above 1%")
    for m, v in sorted(r9_["by_model"].items()):
        cs = "n/a" if v["at_cap_share"] is None else f"{100 * v['at_cap_share']:.2f}%"
        print(f"    {m:22s} rows {v['n']:7d}  missing {100 * v['missing_share']:5.2f}%  at cap {cs}")
    for run, c, n, ms, cs in r9_["above_1pct"][:8]:
        print(f"    {run:28s} {c:40s} n {n:5d} missing {100 * ms:5.1f}%  at cap {'n/a' if cs is None else f'{100 * cs:.1f}%'}")
    for tk, v in r10_[PRIMARY].items():
        rg = v["registered"]
        print(f"R10 {tk}\n    registered {100 * rg['diff']:+.1f} [{100 * rg['ci_low']:+.1f}, {100 * rg['ci_high']:+.1f}] "
              f"p_holm {rg['p_holm']:.3g} (reproduced {v['reproduced']}); without {v['n_dropped_conflict']} conflict q.: "
              f"{100 * v['diff']:+.1f} [{100 * v['ci_low']:+.1f}, {100 * v['ci_high']:+.1f}] n {v['n']} p_holm {v['p_holm']:.3g}")
    for fam, v in r11_["pooled"].items():
        bs = ", ".join(f"s{s} {x['diff']:+.2f} ({x['n_cells']} cells)" for s, x in sorted(v["by_seed"].items()))
        bf = ", ".join(f"s{s} {x['diff']:+.2f}" for s, x in sorted(v["by_seed_fixed"].items()))
        print(f"R11 {fam}: all {v['all']['diff']:+.2f} [{v['all']['ci_low']:+.2f}, {v['all']['ci_high']:+.2f}] "
              f"(reproduced {v['reproduced']}); by seed {bs}; range {_fmt(v['range'], 2)}; single-seed cells kept: {bf}; "
              f"single-seed cells {len(v['single_seed_cells'])}")
    for key, v in r11_["table2"]["settings"].items():
        if len(v["seeds"]) > 1:
            print(f"R11 Table 2 {key:24s} neutral-flag correct {v['all']['correct']:+.1f} range {_fmt(v['range']['correct'])}; "
                  f"misinfo {v['all']['misinfo']:+.1f} range {_fmt(v['range']['misinfo'])}; abstain {v['all']['abstain']:+.1f} "
                  f"range {_fmt(v['range']['abstain'])}")
    print(f"    single-seed settings: {', '.join(r11_['table2']['single_seed_settings'])}")
    print(f"R12 check: {r12_['check']}")
    for tk, v in r12_["H21a_oneway"].items():
        rg = v["registered"]
        print(f"R12 H21(a) {tk:62s} one-way {100 * v['diff']:+.1f} [{100 * v['ci_low']:+.1f}, {100 * v['ci_high']:+.1f}] "
              f"p_holm {v['p_holm']:.3g} {'S' if v['supported'] else '-'} | registered {100 * rg['diff']:+.1f} "
              f"{'S' if rg['supported'] else '-'}")
    print(f"R12: H21(a) decisions changed: {r12_['H21a_changed'] or 'none'}; exploratory family ({r12_['explo_n']} tests) "
          f"significance changed: {len(r12_['explo_sig_changed'])} {r12_['explo_sig_changed'][:6]}")
    for arm, v in out["A1"].items():
        print(f"A1 frame {arm:26s} n {v['n']:5d} matcher-only {100 * v['m_k']:.2f}% judge-only {100 * v['j_k']:.2f}% "
              f"disagreements {v['n_disagree']} on {v['n_disagree_questions']} questions (matches judge19 {v['matches_judge19']})")


def _fmt(r, nd=1):
    return "n/a" if r is None else f"[{r[0]:+.{nd}f}, {r[1]:+.{nd}f}]"


if __name__ == "__main__":
    main()
