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
}
PRIMARY, ROBUST = "ramdocs_strict_oneway", "ramdocs_strict"
SUB = {"all": None, "conflict": True, "clean": False}
N_BOOT, N_PERM = 10_000, 10_000


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
    """Paired effect of condition b over a (b - a), cluster bootstrap CI, two-sided sign-flip p."""
    ra, rb = per_question(rows, a, subset, metric), per_question(rows, b, subset, metric)
    shared = set(ra) & set(rb)
    if not shared:
        return {"diff": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "p": float("nan"), "n": 0}
    bs = paired_bootstrap(ra, rb, groups_of(rows), n_boot=N_BOOT)
    return {**bs, "p": sign_flip_p(ra, rb, n_perm=N_PERM),
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
            h4c = h4(cr, cx, RUNS["contains"])
            R["contains"]["H4"] = {m: {k: h4c[m][k] for k in ("n_cells", "n_harmful", "dose_response",
                                                              "n_remove_worse")} | {"n_matched": len(h4c[m]["remove_vs_flag"])}
                                   for m in h4c}
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
    (out / "hypotheses.json").write_text(json.dumps(R, indent=1, default=float))
    from . import paper_assets
    paper_assets.write_all(R, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
