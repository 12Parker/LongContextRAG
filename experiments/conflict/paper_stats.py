"""
Pre-registered hypothesis tests (PREREGISTRATION.md) and every number, table and figure in the paper.

    python -m experiments.conflict.paper_stats --out paper/generated

Reads the confirmatory runs listed in RUNS, writes
  <out>/hypotheses.json      every test: estimate, 95% CI, raw and Holm p, both metrics
  <out>/numbers.tex          \\newcommand macros used in the paper text
  <out>/tab_*.tex            tables
  <out>/fig_*.pdf            figures
Holm correction is applied within each hypothesis family, separately per metric, as pre-registered.
A hypothesis is "robustly supported" only if it passes under both RAMDocs metrics.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import data
from .analyze import per_question
from .detectors import Detection, DetectorOutput, parse_synth_name
from .metrics import direction_accuracy, doc_level
from .jsonl import read_results, results_path
from .stats import holm, paired_bootstrap, sign_flip_p

ROOT = Path(__file__).resolve().parents[2]
RUNS = {
    "main": ROOT / "runs/holdout_main",
    "detect": ROOT / "runs/holdout_detect",
    "orig": ROOT / "runs/holdout_caag_original",
    "gpt41": ROOT / "runs/holdout_gpt41",
    "wiki": ROOT / "runs/wikicontradict_main",
    "wiki_gpt41": ROOT / "runs/wikicontradict_gpt41",
    "conflicts": ROOT / "runs/conflicts_auto",
    "madam_all": ROOT / "runs/ramdocs_madam_check",
    "contains": ROOT / "runs/holdout_contains",
    # revision runs (PREREGISTRATION.md §9)
    "minimal": ROOT / "runs/holdout_minimal",
    "gpt41_minimal": ROOT / "runs/holdout_gpt41_minimal",
    "qwen": ROOT / "runs/holdout_qwen",
    "qwen_minimal": ROOT / "runs/holdout_qwen_minimal",
    "adversarial": ROOT / "runs/holdout_adversarial",
    # second revision (PREREGISTRATION.md §10)
    "round2": ROOT / "runs/holdout_round2",
    "gpt41_round2": ROOT / "runs/holdout_gpt41_round2",
    "qwendet": ROOT / "runs/holdout_qwendet",
    "cb_main": ROOT / "runs/cb_main",
    "cb_gpt41": ROOT / "runs/cb_gpt41",
    "cb_qwen": ROOT / "runs/cb_qwen",
    # third revision (PREREGISTRATION.md §11)
    "cb_single": ROOT / "runs/cb_single",
    "cb_minimal": ROOT / "runs/cb_minimal",
    "cb_single_gpt41": ROOT / "runs/cb_single_gpt41",
    "cb_single_qwen": ROOT / "runs/cb_single_qwen",
    "cbmaj_single": ROOT / "runs/cbmaj_single",
    "cbmaj_single_gpt41": ROOT / "runs/cbmaj_single_gpt41",
    "holdout_extra": ROOT / "runs/holdout_extra",
    "gpt41_closedbook": ROOT / "runs/holdout_gpt41_closedbook",
    # current reasoning models (PREREGISTRATION.md §12)
    "cb_single_luna": ROOT / "runs/cb_single_luna", "cb_single_terra": ROOT / "runs/cb_single_terra",
    "holdout_luna": ROOT / "runs/holdout_luna", "holdout_terra": ROOT / "runs/holdout_terra",
    "cbmaj_single_luna": ROOT / "runs/cbmaj_single_luna", "cbmaj_single_terra": ROOT / "runs/cbmaj_single_terra",
    # fourth revision (PREREGISTRATION.md §13, §14)
    "cb_forced_mini": ROOT / "runs/cb_forced_mini",
    "cb_forced_luna": ROOT / "runs/cb_forced_luna",
    "cb_forced_terra": ROOT / "runs/cb_forced_terra",
    "cb_forced_haiku": ROOT / "runs/cb_forced_haiku",
    "ctl_rd_mini": ROOT / "runs/ctl_rd_mini",
    "ctl_cb_mini": ROOT / "runs/ctl_cb_mini",
    "ctl_mj_mini": ROOT / "runs/ctl_mj_mini",
    "ctl_rd_luna": ROOT / "runs/ctl_rd_luna",
    "ctl_cb_luna": ROOT / "runs/ctl_cb_luna",
    "ctl_mj_luna": ROOT / "runs/ctl_mj_luna",
    "ctl_rd_terra": ROOT / "runs/ctl_rd_terra",
    "ctl_cb_terra": ROOT / "runs/ctl_cb_terra",
    "ctl_mj_terra": ROOT / "runs/ctl_mj_terra",
    "rg_mini": ROOT / "runs/rg_mini",
    "rg_luna": ROOT / "runs/rg_luna",
    "rg_haiku": ROOT / "runs/rg_haiku",
    "rg_terra": ROOT / "runs/rg_terra",
    "rg_sonnet": ROOT / "runs/rg_sonnet",
    "cb_single_haiku": ROOT / "runs/cb_single_haiku",
    "cb_single_sonnet": ROOT / "runs/cb_single_sonnet",
    "holdout_haiku": ROOT / "runs/holdout_haiku",
    "holdout_sonnet": ROOT / "runs/holdout_sonnet",
    "cbmaj_single_haiku": ROOT / "runs/cbmaj_single_haiku",
}
GENERATORS = {"gpt-4o-mini": ("main", "minimal"), "gpt-4.1": ("gpt41", "gpt41_minimal"),
              "Qwen2.5-7B": ("qwen", "qwen_minimal")}
PRIMARY, ROBUST = "ramdocs_strict_oneway", "ramdocs_strict"
SUB = {"all": None, "conflict": True, "clean": False}
LABELLED = "labelled"   # conflict + clean questions, without the ones the label rule excludes
N_BOOT, N_PERM = 10_000, 10_000
MAX_PERM = 200_000   # floor p-values are refined up to this many permutations (see stats.sign_flip_p)


# --------------------------------------------------------------------------- loading
def rows_of(run: Path) -> list[dict]:
    return read_results(run)


def examples_of(run: Path):
    cfg = json.loads((run / "config.json").read_text())
    kw = {"label_fix": cfg.get("ramdocs_labels", "none")} if cfg["dataset"] == "ramdocs" else {}
    xs = data.filter_ids(data.load(cfg["dataset"], n=cfg.get("n"), seed=cfg.get("sample_seed", 0), **kw),
                         cfg.get("ids_file"))
    return xs


def dets_of(run: Path, name: str) -> dict:
    return {r["id"]: DetectorOutput([Detection(**p) for p in r["pairs"]], {}, r.get("latency_s", 0.0),
                                    r.get("cost_usd", 0.0))
            for r in map(json.loads, open(run / f"detections.{name}.jsonl"))}


def groups_of(rows):
    return {r["id"]: r.get("group", r["id"]) for r in rows}


# --------------------------------------------------------------------------- tests
def compare(rows, a: str, b: str, subset, metric: str) -> dict:
    """Paired effect of condition b over a (b - a), cluster bootstrap CI, two-sided sign-flip p.
    subset LABELLED: all questions that have a conflict label (§9.3.5)."""
    if subset == LABELLED:
        rows, subset = [r for r in rows if r.get("has_conflict") is not None], None
    ra, rb = per_question(rows, a, subset, metric), per_question(rows, b, subset, metric)
    shared = set(ra) & set(rb)
    if not shared:
        return {"diff": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "p": float("nan"), "n": 0}
    bs = paired_bootstrap(ra, rb, groups_of(rows), n_boot=N_BOOT)
    return {**bs, "p": sign_flip_p(ra, rb, n_perm=N_PERM, max_perm=MAX_PERM),
            "mean_a": float(np.mean([ra[i] for i in shared])), "mean_b": float(np.mean([rb[i] for i in shared]))}


def family(tests: dict[str, dict], direction: dict[str, int] | None = None, alpha: float = 0.05) -> dict:
    """Holm within a family; `direction` gives the predicted sign (+1/-1) or 0 for two-sided."""
    adj = holm({k: v["p"] for k, v in tests.items() if not math.isnan(v["p"])})
    for k, v in tests.items():
        v["p_holm"] = adj.get(k, float("nan"))
        want = (direction or {}).get(k, 0)
        v["supported"] = bool(v["p_holm"] < alpha and (want == 0 or np.sign(v["diff"]) == want))
    return tests


def both_metrics(fn) -> dict:
    return {m: fn(m) for m in (PRIMARY, ROBUST)}


def robust(fam_by_metric: dict, key: str) -> bool:
    return all(fam_by_metric[m][key]["supported"] for m in fam_by_metric)


# --------------------------------------------------------------------------- detection (H1)
def _prf_counts(ex, flagged):
    gold = {k for k, v in ex.gold_pairs.items() if v == "contradiction"}
    f = {p for p in flagged if ex.gold_pairs.get(p) != "uncertain"}
    return len(f & gold), len(f - gold), len(gold - f)


def f1_from(c):
    tp, fp, fn = c.sum(0)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def h1(examples, run: Path) -> dict:
    names = ("caag", "nli", "llm_dir")
    dets = {n: dets_of(run, n) for n in names}
    ids = [e.id for e in examples]
    counts = {n: np.array([_prf_counts(e, {(d.i, d.j) for d in dets[n][e.id].pairs}) for e in examples], float)
              for n in names}
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(ids), size=(N_BOOT, len(ids)))
    f1 = {n: f1_from(counts[n]) for n in names}
    boot = {n: np.array([f1_from(counts[n][b]) for b in idx]) for n in names}
    tests = {}
    for hi, lo in (("llm_dir", "nli"), ("nli", "caag")):
        d = boot[hi] - boot[lo]
        p = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))   # two-sided bootstrap p
        tests[f"F1 {hi} > {lo}"] = {"diff": f1[hi] - f1[lo], "ci_low": float(np.quantile(d, .025)),
                                    "ci_high": float(np.quantile(d, .975)), "p": max(p, 1 / N_BOOT)}
    # direction accuracy of llm_dir on flagged true pairs vs 0.5 (exact binomial, two-sided)
    from scipy.stats import binomtest
    right = total = 0
    for e in examples:
        types = e.meta["doc_types"]
        for d in dets["llm_dir"][e.id].pairs:
            if e.gold_pairs.get((d.i, d.j)) == "contradiction" and d.suspects:
                total += 1
                right += sorted(d.suspects) == [k for k in (d.i, d.j) if types[k] == "misinfo"]
    bt = binomtest(right, total, 0.5)
    ci = bt.proportion_ci()
    tests["direction accuracy llm_dir > 0.5"] = {"diff": right / total - 0.5, "estimate": right / total, "n": total,
                                                 "ci_low": ci.low - 0.5, "ci_high": ci.high - 0.5, "p": bt.pvalue}
    family(tests, {k: +1 for k in tests})
    quality = {}
    for n in names:
        fl = {e.id: {(d.i, d.j) for d in dets[n][e.id].pairs} for e in examples}
        tp, fp, fn = counts[n].sum(0)
        clean = [e for e in examples if e.has_conflict is False]
        conf = [e for e in examples if e.has_conflict]
        quality[n] = {
            "pair_precision": tp / (tp + fp) if tp + fp else 0.0, "pair_recall": tp / (tp + fn) if tp + fn else 0.0,
            "pair_f1": f1[n],
            "q_recall": float(np.mean([bool(fl[e.id]) for e in conf])),
            "clean_fpr": float(np.mean([bool(fl[e.id]) for e in clean])),
            "direction_accuracy": direction_accuracy(examples, dets[n]),
            "doc_remove": doc_level(examples, dets[n], "remove_suspect"),
            "latency_s": float(np.mean([o.latency_s for o in dets[n].values()])),
            "cost_per_1k_q": 1000 * float(np.mean([o.cost_usd for o in dets[n].values()])),
        }
    return {"tests": tests, "quality": quality}


# --------------------------------------------------------------------------- QA hypotheses
def qa_family(rows, spec: list[tuple[str, str, str, str]], dirs: dict[str, int]) -> dict:
    """spec: (key, reference, condition, subset name)."""
    return both_metrics(lambda m: family(
        {k: compare(rows, a, b, SUB[s], m) for k, a, b, s in spec}, dirs))


def h4(rows, examples, run: Path) -> dict:
    """False flags on clean questions: cell-wise harm, dose-response, remove vs flag."""
    out = {}
    cells = sorted({r["condition"] for r in rows if r["condition"].startswith("synth-")})
    false_cells = [c for c in cells if parse_synth_name(c.split("+")[0])[0] < 1]
    for m in (PRIMARY, ROBUST):
        t = family({c: compare(rows, "plain", c, False, m) for c in false_cells},
                   {c: -1 for c in false_cells})
        harm = sum(v["supported"] for v in t.values())
        # dose-response: per cell, clean-question exposure vs clean-question effect
        exp, eff = [], []
        for c in cells:
            det, _, mode = c.partition("+")
            exp.append(doc_level(examples, dets_of(run, det), mode)["clean_q_any_correct_marked"])
            eff.append(compare(rows, "plain", c, False, m)["diff"])
        from scipy.stats import spearmanr
        rho = spearmanr(exp, eff)
        perm = np.random.default_rng(0)
        null = [spearmanr(exp, perm.permutation(eff)).statistic for _ in range(5000)]
        p_rho = float((1 + sum(abs(x) >= abs(rho.statistic) for x in null)) / 5001)
        # remove_suspect vs flag_directional at matched (P, R, D), clean questions
        pairs = {}
        for c in false_cells:
            det, _, mode = c.partition("+")
            if mode == "remove_suspect":
                fc = f"{det}+flag_directional"
                if fc in cells:
                    pairs[det] = compare(rows, fc, c, False, m)
        pt = family(pairs, {k: -1 for k in pairs})
        out[m] = {"cells": t, "n_cells": len(false_cells), "n_harmful": harm,
                  "dose_response": {"spearman_rho": float(rho.statistic), "p_perm": p_rho, "n_cells": len(cells)},
                  "remove_vs_flag": pt, "n_remove_worse": sum(v["supported"] for v in pt.values()),
                  "exposure_effect": list(zip(cells, exp, eff))}
    return out


def h6(run_grid: Path, run_det: Path, metric: str) -> dict:
    """Document- vs pair-level placement of real detectors (same split, same runs)."""
    from . import place_detector as pd
    g_rows, d_rows = rows_of(run_grid), rows_of(run_det)
    g_ex, _ = pd._examples(run_grid)
    groups = groups_of(d_rows)
    res = {}
    for det, modes in (("caag", ["remove_suspect"]), ("nli", ["remove_suspect"]),
                       ("llm_dir", ["flag_directional", "remove_suspect"])):
        det_out = dets_of(run_det, det)
        for mode in modes:
            cond = f"{det}+{mode}"
            if cond not in {r["condition"] for r in d_rows}:
                continue
            cells = [c for c in sorted({r["condition"] for r in g_rows})
                     if c.partition("+")[2] == mode and c.startswith("synth-")
                     and parse_synth_name(c.partition("+")[0])[3] is not None]
            for space in ("doc", "pair"):
                here = pd.locate(space, mode, g_ex, det_out)
                where = {c: pd.locate(space, mode, g_ex, dets_of(run_grid, c.partition("+")[0])) for c in cells}
                for s in ("conflict", "clean"):
                    flag = SUB[s]
                    key = "no_conflict" if s == "clean" else s
                    ref = per_question(g_rows, "plain", flag, metric)
                    eff = {c: paired_bootstrap(ref, per_question(g_rows, c, flag, metric), groups, n_boot=500)["diff"]
                           for c in cells}
                    pred, _ = pd.idw(here[key], [(where[c][key], eff[c]) for c in cells], 4)
                    act = paired_bootstrap(per_question(d_rows, "plain", flag, metric),
                                           per_question(d_rows, cond, flag, metric), groups, n_boot=N_BOOT)
                    res[f"{cond}|{s}|{space}"] = {"actual": act["diff"], "ci_low": act["ci_low"],
                                                  "ci_high": act["ci_high"], "predicted": pred,
                                                  "inside": act["ci_low"] <= pred <= act["ci_high"],
                                                  "abs_error": abs(pred - act["diff"])}
    summ = {}
    for space in ("doc", "pair"):
        v = [x for k, x in res.items() if k.endswith("|" + space)]
        summ[space] = {"n": len(v), "inside": sum(x["inside"] for x in v),
                       "mae": float(np.mean([x["abs_error"] for x in v])) if v else None}
    return {"cases": res, "summary": summ}


# --------------------------------------------------------------------------- revision (§9)
def revision(R, main_rows, main_ex, h3spec):
    from . import revision_stats as rv
    rev = R["rev"] = {}
    groups = groups_of(main_rows)
    rev["H4"] = {m: rv.h4_revised(main_rows, main_ex, dets_of, RUNS["main"], m, compare) for m in (PRIMARY, ROBUST)}
    rev["H6"] = {m: rv.h6_prereg(main_rows, main_ex, dets_of, RUNS["main"], m) for m in (PRIMARY, ROBUST)}
    rev["H5_tost"] = {m: rv.tost(main_rows, "random+flag", "caag+flag", None, m, groups) for m in (PRIMARY, ROBUST)}
    rev["direction_ci"] = rv.direction_ci(main_ex, dets_of(RUNS["detect"], "llm_dir"))
    # "All" over labelled questions, and the excluded questions on their own
    rev["labelled"] = {}
    for key, run in (("gpt-4o-mini", "main"), ("gpt-4.1", "gpt41")):
        if not results_path(RUNS[run]):
            continue
        rows = main_rows if run == "main" else rows_of(RUNS[run])
        conds = sorted({r["condition"] for r in rows} - {"plain"})
        rev["labelled"][key] = {m: {c: {LABELLED: compare(rows, "plain", c, LABELLED, m),
                                        "excluded": compare(rv.excluded(rows), "plain", c, None, m)}
                                    for c in conds if not c.startswith("synth-")} for m in (PRIMARY, ROBUST)}
    # minimum detectable effect (80% power, alpha 0.05 two-sided) on conflict questions: 2.8 x the
    # bootstrap SE of each main-table comparison; the paper reports the range
    ses = [(v["conflict"]["ci_high"] - v["conflict"]["ci_low"]) / (2 * 1.96)
           for c, v in R["main_table"][PRIMARY].items() if c != "madam"]
    rev["mde_conflict"] = {"min": 2.8 * min(ses), "median": 2.8 * float(np.median(ses)), "max": 2.8 * max(ses)}
    rev["plain_labelled"] = {m: float(np.mean(list(per_question(rv.labelled(main_rows), "plain", None, m).values())))
                             for m in (PRIMARY, ROBUST)}
    if results_path(RUNS["gpt41"]):
        g41 = rv.labelled(rows_of(RUNS["gpt41"]))
        rev["plain_labelled_gpt41"] = float(np.mean(list(per_question(g41, "plain", None, PRIMARY).values())))
    # generator x skeleton: H10 (minimal skeleton), H11 (open weights, default skeleton)
    gen_rows = {}
    for g, (d, mn) in GENERATORS.items():
        for sk, run in (("default", d), ("minimal", mn)):
            if results_path(RUNS[run]):
                gen_rows[(g, sk)] = main_rows if run == "main" else rows_of(RUNS[run])
    rev["generality"] = {}
    for (g, sk), rows in gen_rows.items():
        conds = [c for c in ("oracle+flag", "oracle+flag_directional", "oracle+remove_suspect", "llm_dir+flag",
                             "llm_dir+flag_directional", "llm_dir+remove_suspect") if c in {r["condition"] for r in rows}]
        from .llm import ABORTED_PREFIX
        n_abort = sum(r["response"].startswith(ABORTED_PREFIX) for r in rows)
        rev["generality"][f"{g}|{sk}"] = {"n_aborted": n_abort, "n_rows": len(rows),
            "plain": {m: {s: float(np.mean(list(per_question(rows, "plain", SUB[s], m).values()))) for s in SUB}
                      for m in (PRIMARY, ROBUST)},
            "effects": {m: {c: {s: compare(rows, "plain", c, sub, m)
                                for s, sub in (("conflict", True), ("clean", False), (LABELLED, LABELLED))}
                            for c in conds} for m in (PRIMARY, ROBUST)},
            "mechanism": {c: {k: float(np.mean([r[k] for r in rows if r["condition"] == c and r["has_conflict"]]))
                              for k in ("wrong_rate", "gold_recall")}
                          for c in ("plain", "oracle+flag", "oracle+flag_directional", "oracle+remove_suspect")
                          if c in {r["condition"] for r in rows}}}
    # exploratory (post hoc, after seeing the minimal-skeleton run): conflict questions with one gold
    # answer vs several (ambiguity + misinformation); a strict answer must list every gold answer
    multi = {e.id for e in main_ex if e.has_conflict and len(e.gold_answers) > 1}
    rev["n_conflict_multi"] = len(multi)
    rev["n_conflict_single"] = sum(1 for e in main_ex if e.has_conflict) - len(multi)
    rev["by_answers"] = {}
    for (g, sk), rows in gen_rows.items():
        conds = {r["condition"] for r in rows}
        for name, keep in (("single", lambda i: i not in multi), ("multi", lambda i: i in multi)):
            rr = [r for r in rows if keep(r["id"])]
            rev["by_answers"][f"{g}|{sk}|{name}"] = {
                "plain": float(np.mean(list(per_question(rr, "plain", True, PRIMARY).values()))),
                **{c: compare(rr, "plain", c, True, PRIMARY) for c in ("oracle+flag", "oracle+flag_directional",
                                                                        "oracle+remove_suspect") if c in conds}}
    h2h3 = [("oracle+flag vs plain (conflict)", "plain", "oracle+flag", "conflict")] + list(h3spec)
    dirs = lambda g, k: 0 if k.startswith("oracle+flag vs") else +1
    h10 = {(g, k, a, b, s): rows for (g, sk), rows in gen_rows.items() if sk == "minimal" for k, a, b, s in h2h3}
    if h10:
        rev["H10"] = {m: family({f"{g}: {k}": compare(rows, a, b, SUB[s], m) for (g, k, a, b, s), rows in h10.items()},
                                {f"{g}: {k}": dirs(g, k) for (g, k, a, b, s) in h10}) for m in (PRIMARY, ROBUST)}
    if ("Qwen2.5-7B", "default") in gen_rows:
        q = gen_rows[("Qwen2.5-7B", "default")]
        spec = [("oracle+flag < plain (conflict)", "plain", "oracle+flag", "conflict")] + list(h3spec)
        rev["H11"] = qa_family(q, [(f"Qwen2.5-7B: {k}", a, b, s) for k, a, b, s in spec],
                               {f"Qwen2.5-7B: {k}": (-1 if "<" in k else +1) for k, *_ in spec})
    # exploratory (§9.5): directional signal that always names the correct side
    if results_path(RUNS["adversarial"]):
        ar = rows_of(RUNS["adversarial"])
        rev["adversarial"] = {c: {s: compare(ar, "plain", c, SUB[s], PRIMARY) for s in SUB}
                              for c in ("oracle_inv+flag_directional", "oracle_inv+remove_suspect")}
        rev["adversarial_wrong"] = {c: float(np.mean([r["wrong_rate"] for r in ar if r["condition"] == c and r["has_conflict"]]))
                                    for c in ("oracle_inv+flag_directional", "oracle_inv+remove_suspect")}
    # break-even precision (pre-registered, §5) and conflict prevalence
    rev["break_even"] = {f"{mode}|r{r:g}|d{d:g}": rv.break_even(R["grid_cells"], mode, r, d)
                         for mode in ("flag_directional", "remove_suspect") for r in (0.25, 0.5, 1.0)
                         for d in (0.5, 0.75, 1.0)}
    rev["prevalence"] = {}
    for gk, v in rev["generality"].items():
        for c in ("llm_dir+flag_directional", "llm_dir+remove_suspect"):
            e = v["effects"][PRIMARY].get(c)
            if e:
                rev["prevalence"][f"{gk}|{c}"] = rv.prevalence(e["conflict"]["diff"], e["clean"]["diff"])
    # exploratory: does direction help with the real detector? (directional vs neutral flag, same detector)
    rev["real_direction"] = {m: {f"{g}|{sk}": compare(rows, "llm_dir+flag", "llm_dir+flag_directional", True, m)
                                 for (g, sk), rows in gen_rows.items()
                                 if {"llm_dir+flag", "llm_dir+flag_directional"} <= {r["condition"] for r in rows}}
                             for m in (PRIMARY, ROBUST)}


# --------------------------------------------------------------------------- second revision (§10)
R2_SUBSETS = (("conflict", True), ("clean", False), (LABELLED, LABELLED))


def revision2(R, main_rows, main_ex=None):
    """Baselines, resolution instruction, ConflictBank, cross-family detector (§10)."""
    rev = R["rev"]
    sets = {}
    if results_path(RUNS["round2"]):
        sets["gpt-4o-mini"] = main_rows + [r for r in rows_of(RUNS["round2"]) if r["condition"] != "plain"]
    for key, run in (("gpt-4.1|seed0", "gpt41_round2"), ("gpt-4o-mini|qwen-detector", "qwendet"),
                     ("CB:gpt-4o-mini", "cb_main"), ("CB:gpt-4.1", "cb_gpt41"), ("CB:Qwen2.5-7B", "cb_qwen"),
                     ("CBs:gpt-4o-mini", "cb_single"), ("CBs:gpt-4.1", "cb_single_gpt41"),
                     ("CBs:Qwen2.5-7B", "cb_single_qwen"), ("CBn:gpt-4o-mini", "cb_minimal"),
                     ("CBmaj:gpt-4o-mini", "cbmaj_single"), ("CBmaj:gpt-4.1", "cbmaj_single_gpt41"),
                     ("CBs:gpt-6-luna", "cb_single_luna"), ("CBs:gpt-6.1-terra", "cb_single_terra"),
                     ("RD:gpt-6-luna", "holdout_luna"), ("RD:gpt-6.1-terra", "holdout_terra"),
                     ("CBmaj:gpt-6-luna", "cbmaj_single_luna"), ("CBmaj:gpt-6.1-terra", "cbmaj_single_terra"),
                     ("CBs:claude-haiku-5-5", "cb_single_haiku"), ("CBs:claude-sonnet-5-5", "cb_single_sonnet"),
                     ("RD:claude-haiku-5-5", "holdout_haiku"), ("RD:claude-sonnet-5-5", "holdout_sonnet"),
                     ("CBmaj:claude-haiku-5-5", "cbmaj_single_haiku"),
                     ("CBf:gpt-4o-mini", "cb_forced_mini"), ("CBf:gpt-6-luna", "cb_forced_luna"),
                     ("CBf:gpt-6.1-terra", "cb_forced_terra"), ("CBf:claude-haiku-5-5", "cb_forced_haiku"),
                     ("RG:gpt-4o-mini", "rg_mini"), ("RG:gpt-6-luna", "rg_luna"), ("RG:claude-haiku-5-5", "rg_haiku"),
                     ("RG:gpt-6.1-terra", "rg_terra"), ("RG:claude-sonnet-5-5", "rg_sonnet")):
        if results_path(RUNS[run]):
            sets[key] = rows_of(RUNS[run])
    if "gpt-4.1|seed0" in sets:
        allseeds = sets["gpt-4.1|seed0"]
        sets["gpt-4.1|seed0"] = [r for r in allseeds if r["seed"] == 0]
        if len({r["seed"] for r in allseeds}) > 1:   # §11.3: H14 with gpt-4.1 on 3 seeds
            h14c = ("plain", "llm_dir+flag_directional", "astute_multi", "trustrag_multi")
            sets["gpt-4.1|all"] = [r for r in allseeds if r["condition"] in h14c]
    if "CBs:gpt-4o-mini" in sets and results_path(RUNS["cb_main"]):   # baselines ignore our skeleton: same prompts
        sets["CBs:gpt-4o-mini"] = sets["CBs:gpt-4o-mini"] + [r for r in rows_of(RUNS["cb_main"])
                                                             if r["condition"] in ("astute", "trustrag")]
    if results_path(RUNS["holdout_extra"]):
        sets["RD:gpt-4o-mini"] = main_rows + rows_of(RUNS["holdout_extra"])
    # §13 controls join the base set of the same generator and data (same items and seeds; their plain is a cache copy)
    for g, gk in (("mini", "gpt-4o-mini"), ("luna", "gpt-6-luna"), ("terra", "gpt-6.1-terra")):
        for d, pre in (("rd", "RD"), ("cb", "CBs"), ("mj", "CBmaj")):
            run = RUNS[f"ctl_{d}_{g}"]
            if results_path(run) and f"{pre}:{gk}" in sets:
                sets[f"{pre}:{gk}"] = sets[f"{pre}:{gk}"] + [r for r in rows_of(run) if r["condition"] != "plain"]
    r2 = rev["r2"] = {}
    for key, rows in sets.items():
        conds = sorted({r["condition"] for r in rows} - {"plain", "closedbook"})
        if key == "gpt-4o-mini":   # only what round 2 added on RAMDocs; the rest is in main_table
            conds = [c for c in conds if "instruct_v2" in c and c.startswith(("oracle", "llm")) or
                     c.split("+")[0] in ("astute", "astute_multi", "trustrag", "trustrag_multi") or
                     c in ("llm_dir+flag_directional",)]
        if key == "RD:gpt-4o-mini":   # main-table conditions are reported from main_table
            conds = [c for c in conds if c in ("self_identify", "oracle+flag_onewrong", "oracle+flag", "oracle+flag_directional",
                                               "llm_dir+flag_directional", "answer_key", "oracle+flag_pick",
                                               "oracle+flag_nopairs", "oracle+flag_onewrong_nopairs",
                                               "llm_dir+flag_directional_nopairs")]
        r2[key] = {
            "plain": {m: {s: float(np.mean(list(per_question(rr, "plain", f, m).values())))
                          for s, f, rr in (("all", None, rows), ("conflict", True, rows), ("clean", False, rows),
                                           ("labelled", None, [r for r in rows if r.get("has_conflict") is not None]))}
                      for m in (PRIMARY, ROBUST)},
            "effects": {m: {c: {s: compare(rows, "plain", c, sub, m) for s, sub in R2_SUBSETS} for c in conds}
                        for m in (PRIMARY, ROBUST)},
            "cost": {c: {"prompt_tokens": float(np.mean([r["prompt_tokens"] for r in rows if r["condition"] == c])),
                         "calls": float(np.mean([r.get("calls", 1) for r in rows if r["condition"] == c])),
                         "cost_usd": float(np.mean([r["cost_usd"] + r.get("detect_cost_usd", 0) for r in rows
                                                    if r["condition"] == c]))}
                     for c in ["plain"] + conds},
            "n": {s: len(per_question(rows, "plain", f, PRIMARY)) for s, f in (("conflict", True), ("clean", False))}}
        if any(r["condition"] == "closedbook" for r in rows):
            r2[key]["closedbook"] = {s: float(np.mean(list(per_question(rows, "closedbook", f, PRIMARY).values())))
                                     for s, f in (("all", None), ("conflict", True), ("clean", False))}
        if key.startswith("CB"):
            r2[key]["wrong_rate"] = {c: float(np.mean([r["wrong_rate"] for r in rows if r["condition"] == c and r["has_conflict"]]))
                                     for c in ("plain", "oracle+flag", "oracle+flag_onewrong", "oracle+flag_directional",
                                               "oracle_inv+flag_directional", "self_identify")
                                     if c in conds + ["plain"]}
        # §11: answer mix on conflict questions (abstention, gold recall, wrong answers), per condition
        unk = lambda r: any(a.strip().lower().rstrip(".") == "unknown" for a in r.get("pred_answers", []))
        def outcome(r):   # mutually exclusive, for the answer-mix figure
            if r[PRIMARY] == 1:
                return "correct"
            if r["wrong_rate"] > 0:
                return "misinfo"
            return "abstain" if unk(r) else "other"
        mix = {}
        for c in ["plain"] + conds:
            rr = [r for r in rows if r["condition"] == c and r["has_conflict"]]
            if rr:
                oc = [outcome(r) for r in rr]
                mix[c] = {"unknown": float(np.mean([unk(r) for r in rr])),
                          "gold_recall": float(np.mean([r["gold_recall"] for r in rr])),
                          "wrong": float(np.mean([r["wrong_rate"] for r in rr])),
                          **{f"out_{k}": oc.count(k) / len(oc) for k in ("correct", "misinfo", "abstain", "other")}}
        r2[key]["answer_mix"] = mix
        r2[key]["_rows"] = rows   # dropped before writing
    regmap = rev["registered_map"] = {}

    def fam(name, spec, dirs):
        out = {}
        for m in (PRIMARY, ROBUST):
            tests = {}
            for tk, key, a, b, sub in spec:
                rows = r2[key]["_rows"]
                tests[tk] = compare(rows, a, b, sub, m)
            out[m] = family(tests, {tk: dirs.get(tk, 0) for tk, *_ in spec})
        for tk, key, a, b, sub in spec:
            sname = {True: "conflict", False: "clean", LABELLED: "labelled"}[sub]
            regmap["\t".join((key, "default", b, a, sname))] = (name, tk)
        return out

    h3 = [("oracle+flag", "plain", 0, "neutral flag vs plain"),
          ("oracle+flag_directional", "plain", +1, "dir. flag > plain"),
          ("oracle+flag_directional", "oracle+flag", +1, "dir. flag > neutral flag"),
          ("oracle+remove_suspect", "plain", +1, "removal > plain")]
    spec, dirs = [], {}
    for key in ("CB:gpt-4o-mini", "CB:gpt-4.1", "CB:Qwen2.5-7B"):
        if key in r2:
            for b, a, d, name in h3:
                tk = f"{key[3:]}: {name} (conflict)"
                spec.append((tk, key, a, b, True)); dirs[tk] = d
    if spec:
        rev["H12"] = fam("H12", spec, dirs)
    spec, dirs = [], {}
    for key, label in (("gpt-4o-mini", "RAMDocs, gpt-4o-mini"), ("gpt-4.1|seed0", "RAMDocs, gpt-4.1"),
                       ("CB:gpt-4o-mini", "ConflictBank, gpt-4o-mini")):
        if key in r2:
            spec.append((f"{label}: neutral flag + instruction vs plain (conflict)", key, "plain",
                         "oracle+flag_instruct_v2", True))
            tk = f"{label}: dir. flag + instruction > neutral flag + instruction (conflict)"
            spec.append((tk, key, "oracle+flag_instruct_v2", "oracle+flag_directional_instruct_v2", True)); dirs[tk] = +1
    if spec:
        rev["H13"] = fam("H13", spec, dirs)
    spec = []
    for key, label, bases in (("gpt-4o-mini", "RAMDocs, gpt-4o-mini", ("astute_multi", "trustrag_multi")),
                              ("gpt-4.1|all" if "gpt-4.1|all" in r2 else "gpt-4.1|seed0", "RAMDocs, gpt-4.1",
                               ("astute_multi", "trustrag_multi")),
                              ("CB:gpt-4o-mini", "ConflictBank, gpt-4o-mini", ("astute", "trustrag"))):
        if key in r2:
            for bname in bases:
                spec.append((f"{label}: LLM dir. flag vs {bname.replace('_', ' ')} (labelled)", key, bname,
                             "llm_dir+flag_directional", LABELLED))
    if spec:
        rev["H14"] = fam("H14", spec, {})
    revision3(R, r2, fam, main_ex)
    revision4(R, r2, fam, main_ex)
    # exploratory (after seeing H14): do the baselines lose valid answers on multi-answer clean questions?
    if "gpt-4o-mini" in r2 and main_ex is not None:
        n_gold = {e.id: len(e.gold_answers) for e in main_ex}
        rows = r2["gpt-4o-mini"]["_rows"]
        rev["baseline_recall"] = {
            grp: {c: float(np.mean([r["gold_recall"] for r in rows if r["condition"] == c and r["has_conflict"] is False
                                    and (n_gold.get(r["id"], 0) > 1) == (grp == "multi")]))
                  for c in ("plain", "astute_multi", "trustrag_multi")}
            for grp in ("single", "multi")}
    for v in r2.values():
        v.pop("_rows", None)


SET_PREFIX = {"CBs": "ConflictBank", "RD": "RAMDocs", "CBmaj": "CB majority", "CBf": "ConflictBank forced",
              "RG": "RAGuard"}


def set_label(key: str) -> str:
    pre, gen = key.split(":", 1)
    return f"{SET_PREFIX[pre]}, {gen}"


def revision3(R, r2, fam, main_ex):
    """Third revision (§11): H15-H18, closed-book stratification, TOST for "tie" / "no harm" claims."""
    from . import revision_stats as rv
    rev = R["rev"]

    def family_of(name, settings, tests=None):
        groups = [(settings, tests)] if tests is not None else settings
        spec, dirs = [], {}
        for key, tests in ((k, t) for ss, t in groups for k in ss):
            if key not in r2:
                continue
            label = set_label(key)
            have = {r["condition"] for r in r2[key]["_rows"]}
            for a, b, d, what, sub in tests:
                if a in have and b in have:
                    tk = f"{label}: {what}"
                    spec.append((tk, key, a, b, sub)); dirs[tk] = d
        if spec:
            rev[name] = fam(name, spec, dirs)

    cb_settings = ("CBs:gpt-4o-mini", "CBs:gpt-4.1", "CBs:Qwen2.5-7B")
    family_of("H15", cb_settings, [
        ("plain", "oracle+flag", 0, "neutral flag vs plain (conflict)", True),
        ("oracle+flag", "oracle+flag_directional", +1, "dir. flag > neutral flag (conflict)", True),
        ("plain", "oracle+flag_directional", +1, "dir. flag > plain (conflict)", True),
        ("plain", "llm_dir+flag_directional", +1, "LLM dir. flag > plain (all q.)", LABELLED)])
    family_of("H16", ("CBs:gpt-4o-mini", "CBs:gpt-4.1", "RD:gpt-4o-mini"), [
        ("oracle+flag_onewrong", "oracle+flag_directional", +1, "dir. flag > one-wrong flag (conflict)", True),
        ("plain", "oracle+flag_onewrong", 0, "one-wrong flag vs plain (conflict)", True)])
    family_of("H17", ("CBmaj:gpt-4o-mini", "CBmaj:gpt-4.1"), [
        ("plain", "oracle+flag", 0, "neutral flag vs plain (conflict)", True),
        ("plain", "oracle+flag_directional", +1, "dir. flag > plain (conflict)", True),
        ("plain", "llm_dir+flag_directional", 0, "LLM dir. flag vs plain (conflict)", True)])
    family_of("H18", ("RD:gpt-4o-mini", "CBs:gpt-4o-mini", "CBmaj:gpt-4o-mini"), [
        ("self_identify", "llm_dir+flag_directional", 0, "LLM dir. flag vs self-identify (labelled)", LABELLED)])

    new = ("gpt-6-luna", "gpt-6.1-terra")
    family_of("H19", [
        ([f"CBs:{g}" for g in new], [
            ("plain", "oracle+flag", 0, "neutral flag vs plain (conflict)", True),
            ("oracle+flag", "oracle+flag_directional", +1, "dir. flag > neutral flag (conflict)", True),
            ("oracle+flag_onewrong", "oracle+flag_directional", +1, "dir. flag > one-wrong flag (conflict)", True),
            ("plain", "llm_dir+flag_directional", +1, "LLM dir. flag > plain (all q.)", LABELLED)]),
        ([f"RD:{g}" for g in new], [
            ("plain", "oracle+flag", 0, "neutral flag vs plain (conflict)", True),
            ("oracle+flag", "oracle+flag_directional", +1, "dir. flag > neutral flag (conflict)", True)]),
        ([f"CBmaj:{g}" for g in new], [
            ("plain", "oracle+flag_directional", +1, "dir. flag > plain (conflict)", True),
            ("plain", "llm_dir+flag_directional", 0, "LLM dir. flag vs plain (conflict)", True)])])

    # degenerate reasoning-model outputs (§12): empty responses, all runs of the two current models
    rev["n_empty_reasoning"] = sum(not r["response"].strip() for k in r2 if k.endswith(("gpt-6-luna", "gpt-6.1-terra"))
                                   for r in r2[k]["_rows"])

    # closed-book stratification (§11.3): "known" = strictly correct closed-book on >= 2 of 3 seeds
    def known_ids(rows):
        by = defaultdict(list)
        for r in rows:
            if r["condition"] == "closedbook":
                by[r["id"]].append(r[PRIMARY])
        return {i for i, v in by.items() if sum(v) >= min(2, len(v))}

    strat = rev["closedbook"] = {}
    for name, key, run in (("ConflictBank", "CBs:gpt-4o-mini", "cb_main"), ("RAMDocs", "RD:gpt-4o-mini", "detect")):
        if key not in r2 or "closedbook" not in r2[key]:
            continue
        rows = r2[key]["_rows"]
        kn = known_ids(rows)
        ex = [e for e in (main_ex if name == "RAMDocs" else examples_of(RUNS[run])) if e.has_conflict]
        dets = dets_of(RUNS[run], "llm_dir")
        d = {"n_conflict": len(ex), "n_known": sum(e.id in kn for e in ex)}
        for grp, keep in (("known", lambda i: i in kn), ("unknown", lambda i: i not in kn)):
            sub_ex = [e for e in ex if keep(e.id)]
            sub_rows = [r for r in rows if keep(r["id"])]
            d[grp] = {"n": len(sub_ex), "direction": direction_accuracy(sub_ex, dets),
                      **{c: compare(sub_rows, "plain", c, True, PRIMARY)
                         for c in ("oracle+flag", "oracle+flag_directional", "llm_dir+flag_directional")
                         if any(r["condition"] == c for r in sub_rows)}}
        strat[name] = d

    # TOST for every "no harm" / "tie" style claim (§11.3), ±3 points
    tt = rev["tost_claims"] = {}
    for tk, key, a, b, sub in (
            ("cb_llmdir_vs_astute", "CB:gpt-4o-mini", "astute", "llm_dir+flag_directional", None),
            ("cb_llmdir_vs_trustrag", "CB:gpt-4o-mini", "trustrag", "llm_dir+flag_directional", None),
            ("cb_llmdir_clean_mini", "CB:gpt-4o-mini", "plain", "llm_dir+flag_directional", False),
            ("cb_llmdir_clean_four", "CB:gpt-4.1", "plain", "llm_dir+flag_directional", False),
            ("cbs_llmdir_clean_mini", "CBs:gpt-4o-mini", "plain", "llm_dir+flag_directional", False),
            ("rd_llmdir_clean_mini", "gpt-4o-mini", "plain", "llm_dir+flag_directional", False)):
        if key in r2:
            rows = r2[key]["_rows"]
            if any(r["condition"] == b for r in rows):
                tt[tk] = rv.tost(rows, a, b, sub, PRIMARY, groups_of(rows))


def revision4(R, r2, fam, main_ex):
    """Fourth revision (§13, §14): forced choice, risk/coverage, pair topology, answer key, crossed
    detector x generator reliability, Claude, RAGuard, pooled neutral-flag estimate, majority breakdown."""
    from scipy.stats import spearmanr
    rev = R["rev"]
    unk = lambda r: any(a.strip().lower().rstrip(".") == "unknown" for a in r.get("pred_answers", []))
    for v in r2.values():            # derived per-row outcomes for H21
        for r in v["_rows"]:
            ab = unk(r)
            r["misinfo"], r["abstain"] = r["wrong_rate"], float(ab)
            r["selective"] = None if ab else r[PRIMARY]

    def fam_of(name, groups):
        spec, dirs = [], {}
        for keys, tests in groups:
            for key in keys:
                if key not in r2:
                    continue
                have = {r["condition"] for r in r2[key]["_rows"]}
                for a, b, d, what, sub in tests:
                    if a in have and b in have:
                        tk = f"{set_label(key)}: {what}"
                        spec.append((tk, key, a, b, sub)); dirs[tk] = d
        if spec:
            rev[name] = fam(name, spec, dirs)

    G4 = ("gpt-4o-mini", "gpt-6-luna", "gpt-6.1-terra", "claude-haiku-5-5")
    fam_of("H20", [([f"CBf:{g}" for g in G4], [
        ("plain", "oracle+flag", 0, "neutral flag vs plain (conflict)", True),
        ("plain", "oracle+flag_onewrong", 0, "one-wrong flag vs plain (conflict)", True),
        ("plain", "oracle+flag_directional", +1, "dir. flag > plain (conflict)", True)])])
    fam_of("H22", [
        ([f"RD:{g}" for g in G4], [("oracle+flag_onewrong", "oracle+flag_onewrong_nopairs", 0,
                                    "one-wrong without pairs vs with pairs (conflict)", True)]),
        ([f"CBmaj:{g}" for g in G4], [
            ("oracle+flag", "oracle+flag_nopairs", 0, "neutral without pairs vs with pairs (conflict)", True),
            ("llm_dir+flag_directional", "llm_dir+flag_directional_nopairs", 0,
             "LLM dir. flag without pairs vs with pairs (conflict)", True)])])
    fam_of("H23", [([f"{p}:{g}" for p in ("RD", "CBs", "CBmaj") for g in G4], [
        ("answer_key", "oracle+flag_directional", 0, "dir. flag vs answer key (conflict)", True)])])
    claude = ("claude-haiku-5-5", "claude-sonnet-5-5")
    fam_of("H25", [
        ([f"CBs:{g}" for g in claude], [
            ("plain", "oracle+flag", 0, "neutral flag vs plain (conflict)", True),
            ("oracle+flag", "oracle+flag_directional", +1, "dir. flag > neutral flag (conflict)", True),
            ("oracle+flag_onewrong", "oracle+flag_directional", +1, "dir. flag > one-wrong flag (conflict)", True),
            ("plain", "llm_dir+flag_directional", +1, "LLM dir. flag > plain (all q.)", LABELLED)]),
        ([f"RD:{g}" for g in claude], [
            ("plain", "oracle+flag", 0, "neutral flag vs plain (conflict)", True),
            ("oracle+flag", "oracle+flag_directional", +1, "dir. flag > neutral flag (conflict)", True)]),
        ([f"CBmaj:{g}" for g in claude], [
            ("plain", "oracle+flag_directional", +1, "dir. flag > plain (conflict)", True),
            ("plain", "llm_dir+flag_directional", 0, "LLM dir. flag vs plain (conflict)", True)])])
    fam_of("H26", [([f"RG:{g}" for g in G4 + ("claude-sonnet-5-5",)], [
        ("plain", "oracle+flag", 0, "neutral flag vs plain (conflict)", True),
        ("plain", "oracle+flag_directional", +1, "dir. flag > plain (conflict)", True),
        ("oracle+flag", "oracle+flag_directional", +1, "dir. flag > neutral flag (conflict)", True),
        ("plain", "oracle+flag_onewrong", 0, "one-wrong flag vs plain (conflict)", True),
        ("plain", "llm_dir+flag_directional", 0, "LLM dir. flag vs plain (conflict)", True)])])

    # H21: risk and coverage (single and forced prompts), registered as one family per metric
    h21 = {}
    for m_ in ("misinfo", "abstain", "selective"):
        tests, dirs = {}, {}
        for key in [f"{p}:{g}" for p in ("CBs", "CBf") for g in G4]:
            if key not in r2:
                continue
            rows = r2[key]["_rows"]
            for c, nm in (("oracle+flag", "neutral flag"), ("oracle+flag_directional", "dir. flag")):
                if any(r["condition"] == c for r in rows):
                    tk = f"{set_label(key)}: {nm} vs plain (conflict)"
                    tests[tk] = compare(rows, "plain", c, True, m_); dirs[tk] = 0
        if tests:
            h21[m_] = family(tests, dirs)
    rev["H21"] = h21
    for k, v in r2.items():   # risk/coverage effects of every condition, for the tables (exploratory)
        if not k.startswith(("CBs", "CBf", "CBmaj", "RG", "CB:")):
            continue
        rows = v["_rows"]
        v["risk"] = {m_: {c: compare(rows, "plain", c, True, m_) for c in v["effects"][PRIMARY]}
                     for m_ in ("misinfo", "abstain", "selective")}

    # H24: crossed detector x generator reliability (ConflictBank single, conflict questions)
    gen_key = {"mini": "gpt-4o-mini", "luna": "gpt-6-luna", "four": "gpt-4.1", "terra": "gpt-6.1-terra",
               "haiku": "claude-haiku-5-5", "qwen": "Qwen2.5-7B"}
    det_key = {"luna": "gpt-6-luna", "four": "gpt-4.1", "terra": "gpt-6.1-terra", "haiku": "claude-haiku-5-5",
               "sonnet": "claude-sonnet-5-5"}

    def crossed(ex_run, base_run, set_prefix, cross_prefix, keep=lambda g, d: True):
        ex_ = {e.id: e for e in examples_of(RUNS[ex_run]) if e.has_conflict}
        # d_q: the detector flags a gold contradiction pair and names exactly its misinformation member(s)
        contra = {i: {p: {k for k in p if e.meta["doc_types"][k] == "misinfo"}
                      for p, lab in e.gold_pairs.items() if lab == "contradiction"} for i, e in ex_.items()}
        correct = {i: {k for k, t in enumerate(e.meta["doc_types"]) if t == "correct"} for i, e in ex_.items()}
        cells = {}
        for gname in gen_key.values():          # gpt-4o-mini detector: the base single-prompt runs
            if f"{set_prefix}{gname}" in r2 and keep(gname, "gpt-4o-mini"):
                cells[(gname, "gpt-4o-mini")] = (r2[f"{set_prefix}{gname}"]["_rows"], RUNS[base_run])
        for gk, gname in gen_key.items():
            for dk, dname in det_key.items():
                run = ROOT / f"runs/{cross_prefix}_{gk}_{dk}"
                if results_path(run) and keep(gname, dname):
                    cells[(gname, dname)] = (rows_of(run), run)
        cell_out, pooled = {}, []
        for (g, d), (rows, run) in cells.items():
            det = dets_of(run, "llm_dir")
            dq = {i: float(any(contra[i].get((x.i, x.j)) == set(x.suspects) for x in det[i].pairs))
                  for i in ex_ if i in det}
            gp = per_question(rows, "plain", True, PRIMARY)
            gf = per_question(rows, "llm_dir+flag_directional", True, PRIMARY)
            ids = sorted(set(dq) & set(gp) & set(gf))
            if not ids:
                continue
            gain = {i: gf[i] - gp[i] for i in ids}
            # exploratory (post hoc): right where the generator is wrong, minus pointing at a correct passage where it is right
            wrong_dir = {i: float(bool({k for x in det[i].pairs for k in x.suspects} & correct[i])) for i in ids}
            mech = float(np.mean([dq[i] * (1 - gp[i]) - wrong_dir[i] * gp[i] for i in ids]))
            cell_out[f"{g}|{d}"] = {"mech": mech, "wrong_dir": float(np.mean(list(wrong_dir.values()))),
                                    "any_flag": float(np.mean([bool(det[i].pairs) for i in ids])),"gap": float(np.mean([dq[i] for i in ids]) - np.mean([gp[i] for i in ids])),
                                    "det_acc": float(np.mean([dq[i] for i in ids])), "gen_acc": float(np.mean([gp[i] for i in ids])),
                                    "gain": float(np.mean(list(gain.values()))), "n": len(ids),
                                    "effect": compare(rows, "plain", "llm_dir+flag_directional", True, PRIMARY)}
            pooled += [(f"{g}|{d}", i, dq[i] - gp[i], gain[i]) for i in ids]
        h = {"cells": cell_out}
        if len(cell_out) < 4:
            return h
        gaps = np.array([v["gap"] for v in cell_out.values()]); gains = np.array([v["gain"] for v in cell_out.values()])
        rho = float(spearmanr(gaps, gains).statistic)
        rng = np.random.default_rng(0)
        perm = np.array([spearmanr(gaps, rng.permutation(gains)).statistic for _ in range(10_000)])
        h["a"] = {"rho": rho, "p_one_sided": float((np.sum(perm >= rho) + 1) / (len(perm) + 1)), "n_cells": len(cell_out)}
        # H24b: within-cell slope of per-question gain on (d_q - g_q), cluster bootstrap over questions
        cells_ = sorted({c for c, *_ in pooled})
        def slope(data):
            x, y = [], []
            for c in cells_:
                xs = np.array([a for cc, _, a, _ in data if cc == c]); ys = np.array([b for cc, _, _, b in data if cc == c])
                if len(xs) > 1:
                    x.append(xs - xs.mean()); y.append(ys - ys.mean())
            x, y = np.concatenate(x), np.concatenate(y)
            return float((x * y).sum() / (x * x).sum())
        b0 = slope(pooled)
        qids = sorted({i for _, i, _, _ in pooled}); by_q = defaultdict(list)
        for t in pooled:
            by_q[t[1]].append(t)
        boots = []
        for _ in range(2000):
            samp = rng.choice(qids, len(qids))
            boots.append(slope([t for q in samp for t in by_q[q]]))
        lo, hi = np.quantile(boots, [0.025, 0.975])
        h["b"] = {"slope": b0, "ci_low": float(lo), "ci_high": float(hi), "p_one_sided": float(np.mean(np.array(boots) <= 0))}
        # where the cross-cell least-squares line of gain on gap crosses zero
        mech = np.array([v["mech"] for v in cell_out.values()])
        h["mech"] = {"rho": float(spearmanr(mech, gains).statistic), "r": float(np.corrcoef(mech, gains)[0, 1]),
                     "rho_gap": rho, "r_gap": float(np.corrcoef(gaps, gains)[0, 1])}
        k, c0 = np.polyfit(gaps, gains, 1)
        h["break_even_gap"] = float(-c0 / k) if k > 0 else None
        h["n_positive"] = int(sum(v["effect"]["diff"] > 0 for v in cell_out.values()))
        return h

    rev["H24"] = crossed("cb_single", "cb_main", "CBs:", "cross_cb")
    # sensitivity (§14): the cells registered under §13 only, i.e. no Claude detector or generator
    rev["H24_s13"] = crossed("cb_single", "cb_main", "CBs:", "cross_cb", lambda g, d: "claude" not in g + d)
    if "CBmaj:gpt-4o-mini" in r2:   # exploratory: the same analysis under majority misinformation
        rev["H24_maj"] = crossed("cbmaj_single", "cbmaj_single", "CBmaj:", "cross_mj")

    # pooled random-effects estimate of the oracle neutral flag across settings (DerSimonian-Laird)
    neu = [(k, v["effects"][PRIMARY]["oracle+flag"]["conflict"]) for k, v in r2.items()
           if k.startswith(("CB", "RD:gpt-6", "RD:claude", "RG")) and "oracle+flag" in v["effects"][PRIMARY]]
    neu += [(k, v["effects"][PRIMARY]["oracle+flag"]["conflict"]) for k, v in rev.get("generality", {}).items()
            if "oracle+flag" in v["effects"][PRIMARY]]
    y = np.array([t["diff"] for _, t in neu]); se = np.array([(t["ci_high"] - t["ci_low"]) / (2 * 1.96) for _, t in neu])
    w = 1 / se ** 2; ybar = (w * y).sum() / w.sum(); Q = (w * (y - ybar) ** 2).sum()
    tau2 = max(0.0, (Q - (len(y) - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    ws = 1 / (se ** 2 + tau2); mu = (ws * y).sum() / ws.sum(); se_mu = (1 / ws.sum()) ** 0.5
    rev["neutral_pooled"] = {"mu": float(mu), "ci_low": float(mu - 1.96 * se_mu), "ci_high": float(mu + 1.96 * se_mu),
                             "tau": float(tau2 ** 0.5), "k": len(y), "_effects": [t for _, t in neu]}

    # majority misinformation: what the gpt-4o-mini detector's directional flag marks, per question
    if "CBmaj:gpt-4o-mini" in r2:
        mj_ex = {e.id: e for e in examples_of(RUNS["cbmaj_single"]) if e.has_conflict}
        det = dets_of(RUNS["cbmaj_single"], "llm_dir")
        rows = r2["CBmaj:gpt-4o-mini"]["_rows"]
        gp = per_question(rows, "plain", True, PRIMARY); gf = per_question(rows, "llm_dir+flag_directional", True, PRIMARY)
        groups = defaultdict(list)
        for i, e in mj_ex.items():
            t = e.meta["doc_types"]; sus = {k for x in det[i].pairs for k in x.suspects}
            marks_correct = any(t[k] == "correct" for k in sus); n_mis = sum(t[k] == "misinfo" for k in sus)
            grp = "correct marked" if marks_correct else (f"{min(n_mis, 3)} wrong marked" if n_mis else "nothing marked")
            if i in gp and i in gf:
                groups[grp].append(gf[i] - gp[i])
        rev["majority_breakdown"] = {g: {"n": len(v), "gain": float(np.mean(v))} for g, v in groups.items()}

    # H13 as flag x instruction (§13.4): the neutral flag's effect given the instruction
    from . import revision_stats as rv
    h13x = {}
    for key, label in (("gpt-4o-mini", "RAMDocs, gpt-4o-mini"), ("gpt-4.1|seed0", "RAMDocs, gpt-4.1 (seed 0)"),
                       ("CB:gpt-4o-mini", "ConflictBank (multi-answer prompt), gpt-4o-mini")):
        if key in r2:
            rows = r2[key]["_rows"] + ([r for r in R.get("_main_rows", []) if r["condition"] == "instruct_v2"] if key == "gpt-4.1|seed0" else [])
            if {"instruct_v2", "oracle+flag_instruct_v2"} <= {r["condition"] for r in rows}:
                h13x[label] = compare(rows, "instruct_v2", "oracle+flag_instruct_v2", True, PRIMARY)
    rev["H13_2x2"] = h13x
    # TOST (±3 points) for "as good as" claims (§13.4): single call vs detector pipeline, Qwen vs gpt-4o-mini detector
    tt = rev.setdefault("tost_claims", {})
    for tk, key in (("h18_rd", "RD:gpt-4o-mini"), ("h18_cb", "CBs:gpt-4o-mini"), ("h18_mj", "CBmaj:gpt-4o-mini")):
        if key in r2:
            rows = [r for r in r2[key]["_rows"] if r.get("has_conflict") is not None]
            if {"self_identify", "llm_dir+flag_directional"} <= {r["condition"] for r in rows}:
                tt[tk] = rv.tost(rows, "self_identify", "llm_dir+flag_directional", None, PRIMARY, groups_of(rows))

    # CONFLICTS by conflict type, 3 seeds (exploratory)
    if results_path(RUNS["conflicts"]):
        cr = rows_of(RUNS["conflicts"])
        out = {}
        for ct in sorted({r["conflict_type"] for r in cr}):
            sub = [r for r in cr if r["conflict_type"] == ct]
            out[ct] = {"n": len({r["id"] for r in sub}), "plain": float(np.mean(list(per_question(sub, "plain", None, "primary").values()))),
                       **{c: compare(sub, "plain", c, None, "primary") for c in ("llm_dir+flag", "llm_dir+flag_directional", "instruct_v2")}}
        rev["conflicts_by_type"] = out


# --------------------------------------------------------------------------- output helpers
def fmt(x, nd=3, sign=True):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def fp(p):
    if p is None or math.isnan(p):
        return "--"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def macro(name: str, value) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "paper/generated"))
    args = ap.parse_args(argv)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    R = {}

    main_rows = rows_of(RUNS["main"]); main_ex = examples_of(RUNS["main"])
    R["H1"] = h1(main_ex, RUNS["detect"])
    R["H2"] = qa_family(main_rows, [("oracle+flag < plain (conflict)", "plain", "oracle+flag", "conflict")],
                        {"oracle+flag < plain (conflict)": -1})
    h3spec = [("oracle+flag_directional > plain (conflict)", "plain", "oracle+flag_directional", "conflict"),
              ("oracle+flag_directional > oracle+flag (conflict)", "oracle+flag", "oracle+flag_directional", "conflict"),
              ("oracle+remove_suspect > plain (conflict)", "plain", "oracle+remove_suspect", "conflict")]
    R["H3"] = qa_family(main_rows, h3spec, {k: +1 for k, *_ in h3spec})
    R["H4"] = h4(main_rows, main_ex, RUNS["main"])
    # every grid cell in every subset, Holm within each (mode, subset) table, for the heatmap
    gc = {}
    cells = sorted({r["condition"] for r in main_rows if r["condition"].startswith("synth-")})
    for mode in sorted({c.partition("+")[2] for c in cells}):
        for s in SUB:
            fam = {c: compare(main_rows, "plain", c, SUB[s], PRIMARY) for c in cells if c.endswith("+" + mode)}
            family(fam)
            for c, v in fam.items():
                pp, rr, _, dd = parse_synth_name(c.partition("+")[0])
                gc[f"{mode}|{pp:g}|{rr:g}|{dd:g}|{s}"] = v
    R["grid_cells"] = gc
    R["H5"] = qa_family(main_rows, [("caag+flag vs random+flag (all)", "random+flag", "caag+flag", "all")], {})
    R["H6"] = {m: h6(RUNS["main"], RUNS["main"], m) for m in (PRIMARY, ROBUST)}
    h7spec = [(f"{c} vs plain ({s})", "plain", c, s) for c in ("instruct", "instruct_v2") for s in ("conflict", "clean")]
    R["H7"] = qa_family(main_rows, h7spec, {})
    # all conditions vs plain, for the main table
    conds = ["instruct", "instruct_v2", "oracle+flag", "oracle+flag_directional", "oracle+remove_suspect",
             "random+flag", "caag+flag", "caag+remove", "nli+flag", "nli+remove", "llm_dir+flag",
             "llm_dir+flag_directional", "llm_dir+remove_suspect", "madam"]
    R["main_table"] = {m: {c: {s: compare(main_rows, "plain", c, SUB[s], m) for s in SUB} for c in conds}
                       for m in (PRIMARY, ROBUST)}
    R["plain_means"] = {m: {s: float(np.mean(list(per_question(main_rows, "plain", SUB[s], m).values())))
                            for s in SUB} for m in (PRIMARY, ROBUST)}
    # mechanism: on conflict questions, does a flag add the misinformation answer or drop the gold one?
    R["mechanism"] = {c: {k: float(np.mean([r[k] for r in main_rows if r["condition"] == c and r["has_conflict"]]))
                          for k in ("wrong_rate", "gold_recall")}
                      for c in ("plain", "oracle+flag", "oracle+flag_directional", "oracle+remove_suspect")}
    R["main_costs"] = {c: {"prompt_tokens": float(np.mean([r["prompt_tokens"] for r in main_rows if r["condition"] == c])),
                           "cost_usd": float(np.mean([r["cost_usd"] + r.get("detect_cost_usd", 0) for r in main_rows
                                                      if r["condition"] == c])),
                           "generate_s": float(np.mean([r["generate_s"] for r in main_rows if r["condition"] == c])),
                           "detect_s": float(np.mean([r.get("detect_s", 0) for r in main_rows if r["condition"] == c]))}
                       for c in ["plain"] + conds}
    if results_path(RUNS["orig"]):
        o = rows_of(RUNS["orig"])
        R["orig_format"] = {m: {s: compare(o, "plain", "caag+flag", SUB[s], m) for s in SUB} for m in (PRIMARY, ROBUST)}
    if results_path(RUNS["gpt41"]):
        g = rows_of(RUNS["gpt41"])
        h9spec = [("gpt-4.1: oracle+flag < plain (conflict)", "plain", "oracle+flag", "conflict")] + \
                 [(f"gpt-4.1: {k}", a, b, s) for k, a, b, s in h3spec]
        R["H9"] = qa_family(g, h9spec, {k: (-1 if "<" in k else +1) for k, *_ in h9spec})
        g_conds = ["instruct_v2", "oracle+flag", "oracle+flag_directional", "oracle+remove_suspect",
                   "llm_dir+flag_directional", "llm_dir+remove_suspect", "nli+remove_suspect"]
        R["gpt41_table"] = {m: {c: {s: compare(g, "plain", c, SUB[s], m) for s in SUB} for c in g_conds}
                            for m in (PRIMARY, ROBUST)}
        R["gpt41_plain"] = {m: {s: float(np.mean(list(per_question(g, "plain", SUB[s], m).values()))) for s in SUB}
                            for m in (PRIMARY, ROBUST)}
    if results_path(RUNS["wiki"]):
        w = rows_of(RUNS["wiki"])
        wc = ["instruct", "instruct_v2", "oracle+flag", "caag+flag", "nli+flag", "llm_dir+flag"]
        R["H8"] = {"both_answers": family({f"{c} vs plain": compare(w, "plain", c, None, "both_answers") for c in wc},
                                          {"oracle+flag vs plain": +1})}
        R["wiki_plain"] = float(np.mean(list(per_question(w, "plain", None, "both_answers").values())))
        R["wiki_by_type"] = {t: {c: compare([r for r in w if r["conflict_type"] == t], "plain", c, None, "both_answers")
                                 for c in wc} for t in sorted({r["conflict_type"] for r in w})}
    if results_path(RUNS["wiki_gpt41"]):
        w4 = rows_of(RUNS["wiki_gpt41"])
        R["wiki_gpt41"] = {"plain": float(np.mean(list(per_question(w4, "plain", None, "both_answers").values()))),
                           "oracle+flag": compare(w4, "plain", "oracle+flag", None, "both_answers")}
    if results_path(RUNS["conflicts"]):
        cr = rows_of(RUNS["conflicts"])
        cc = ["instruct_v2", "caag+flag", "nli+flag", "llm_dir+flag", "llm_dir+flag_directional"]
        R["conflicts"] = {c: {s: compare(cr, "plain", c, SUB[s], "primary") for s in SUB} for c in cc}
        R["conflicts_plain"] = {s: float(np.mean(list(per_question(cr, "plain", SUB[s], "primary").values())))
                                for s in SUB}
        cq = json.loads((RUNS["conflicts"] / "detector_quality.json").read_text()) \
            if (RUNS["conflicts"] / "detector_quality.json").exists() else {}
        cx = examples_of(RUNS["conflicts"])
        R["conflicts_detect"] = {}
        for n in ("caag", "nli", "llm_dir"):
            if (RUNS["conflicts"] / f"detections.{n}.jsonl").exists():
                d = dets_of(RUNS["conflicts"], n)
                conf = [e for e in cx if e.has_conflict]; clean = [e for e in cx if e.has_conflict is False]
                R["conflicts_detect"][n] = {"q_recall": float(np.mean([bool(d[e.id].pairs) for e in conf])),
                                            "clean_fpr": float(np.mean([bool(d[e.id].pairs) for e in clean]))}
    if results_path(RUNS["madam_all"]):
        mr = rows_of(RUNS["madam_all"])
        R["madam_all500"] = {m: float(np.mean([r[m] for r in mr])) for m in (PRIMARY, ROBUST)}

    if results_path(RUNS["contains"]):
        cr = rows_of(RUNS["contains"]); cx = examples_of(RUNS["contains"])
        R["contains"] = {
            "H2": qa_family(cr, [("oracle+flag < plain (conflict)", "plain", "oracle+flag", "conflict")],
                            {"oracle+flag < plain (conflict)": -1}),
            "H3": qa_family(cr, h3spec, {k: +1 for k, *_ in h3spec}),
            "n": {s: sum(e.has_conflict is f for e in cx) for s, f in (("conflict", True), ("clean", False))}}
        if any(r["condition"].startswith("synth-") for r in cr):
            from . import revision_stats as rv
            R["contains"]["H4"] = {m: rv.h4_revised(cr, cx, dets_of, RUNS["contains"], m, compare)
                                   for m in (PRIMARY, ROBUST)}
    revision(R, main_rows, main_ex, h3spec)
    revision2(R, main_rows, main_ex)
    import csv
    sweep = list(csv.DictReader(open(RUNS["detect"] / "caag_sweep.csv")))
    for r in sweep:
        for k in r:
            r[k] = float(r[k])
    summ = json.loads((RUNS["detect"] / "summary.json").read_text())
    R["caag_sweep"] = {"default": next(r for r in sweep if r["theta_s"] == 0.65 and r["theta_c"] == 0.35),
                       "best_pair_f1": max(sweep, key=lambda r: r["pair_f1"]),
                       "best_q_f1": max(sweep, key=lambda r: r["q_f1"]), "n_settings": len(sweep)}
    R["caag_ablation"] = {k: {"pair_f1": v["pair_level"]["f1"], "pair_flag_rate": v["pair_flag_rate"],
                              "q_fpr": v["question_level"]["false_positive_rate"]}
                          for k, v in summ.items() if k.startswith("caag")}
    from .revision_stats import assign_significance
    assign_significance(R)
    np_ = R["rev"].get("neutral_pooled")
    if np_:   # harm/help counts use the adjusted significance, so they are taken after it is assigned
        eff = np_.pop("_effects")
        np_["n_harm"] = int(sum(t["sig"] and t["diff"] < 0 for t in eff))
        np_["n_help"] = int(sum(t["sig"] and t["diff"] > 0 for t in eff))
    (out / "hypotheses.json").write_text(json.dumps(R, indent=1, default=float))
    from . import paper_assets
    paper_assets.write_all(R, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
