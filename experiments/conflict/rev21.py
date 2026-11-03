"""
Analyses of the PR§21 runs (PREREGISTRATION.md §21; C1, C3-C9, C11). Reads only finished run files; no API calls.

    python -m experiments.conflict.rev21 [--only c1,c3,c4,c5,c6,c7,c9,c11]

Writes runs/rev21/rev21.json and prints a summary. A part whose runs are missing is reported as not run.
Conventions (§21): primary metric one-way strict, seed means within questions, question-bootstrap 95% CIs with 5,000
resamples (seed 0), conflict and clean questions separately; follow rates as §16.3 (`revision10._follow`).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .analyze import per_question
from .jsonl import read_results, results_path
from .rev11_prior import _q, _ratio_split, _split_rates, gold_flag_counts, known_ids
from .revision10 import FOLLOW19, _examples
from .revision6 import _cell_outcomes
from .stats import holm, paired_bootstrap

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
OUT = RUNS / "rev21/rev21.json"
PRIMARY = "ramdocs_strict_oneway"
N_BOOT, N_PERM = 5000, 10_000
SYN = "synth-p1-r1-all-d0.5"
RAND = f"{SYN}+flag_directional"
GEN = {"mini": "gpt-4o-mini", "luna": "gpt-6-luna", "haiku": "claude-haiku-5-5", "terra": "gpt-6.1-terra",
       "sonnet": "claude-sonnet-5-5", "qwen": "qwen2.5:7b-instruct"}
# §21.2 H40a: the frozen eval 95% CIs (registered values)
FROZEN_CI = {"luna": ((0.607, 0.827), (0.673, 0.860)), "haiku": ((0.492, 0.758), (0.545, 0.729)),
             "terra": ((0.533, 0.800), (0.895, 0.990))}


def _have(run: str, cond: str | None = None) -> bool:
    p = RUNS / run
    if not results_path(p):
        return False
    return cond is None or any(r["condition"] == cond for r in read_results(p))


def _dets(run: str, name: str) -> dict:
    return {r["id"]: r["pairs"] for r in map(json.loads, open(RUNS / run / f"detections.{name}.jsonl"))}


def _boot_mean(x, seed=0):
    x = np.asarray(x, float)
    if not len(x):
        return {"est": None, "ci": [None, None], "n": 0}
    rng = np.random.default_rng(seed)
    return {"est": float(x.mean()), "ci": _q(x[rng.integers(0, len(x), (N_BOOT, len(x)))].mean(1)), "n": len(x)}


def _ratio(num, den, seed=0):
    num, den = np.asarray(num, float), np.asarray(den, float)
    if not den.sum():
        return {"est": None, "ci": [None, None], "n": 0}
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(num), (N_BOOT, len(num)))
    with np.errstate(invalid="ignore", divide="ignore"):
        b = num[ix].sum(1) / den[ix].sum(1)
    return {"est": float(num.sum() / den.sum()), "ci": _q(b), "n": int(den.sum())}


def _sf1(d, direction=1, seed=0):
    """One-sided paired sign-flip p for mean(d) in `direction` (+1: > 0, -1: < 0)."""
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    stats = (rng.choice([-1, 1], size=(N_PERM, len(d))) * d).mean(1) * direction
    return float((np.sum(stats >= d.mean() * direction - 1e-12) + 1) / (N_PERM + 1))


# --------------------------------------------------------------------------- H31-style arms
def arms(run: str, cond: str = RAND) -> dict:
    p = RUNS / run
    ex = {i: e for i, e in _examples(p).items() if e.has_conflict}
    det = _dets(run, cond.split("+")[0])
    rows = read_results(p)
    right = {i: all(ex[i].meta["doc_types"][k] == "misinfo" for x in det[i] for k in x["suspects"]) for i in ex if i in det}
    pl, fl = per_question(rows, "plain", True, PRIMARY), per_question(rows, cond, True, PRIMARY)
    ids = sorted(set(pl) & set(fl) & set(right))
    return {"R": [i for i in ids if right[i]], "W": [i for i in ids if not right[i]], "pl": pl, "fl": fl,
            "oc": _cell_outcomes(rows, cond, PRIMARY), "seeds": sorted({r["seed"] for r in rows if r["condition"] == cond})}


def h31_summary(a: dict, seed=0) -> dict:
    """Δ_R, Δ_W (flagged minus plain, in points), their sum (H31c), follow rates, break-even p*."""
    R, W, pl, fl, oc = a["R"], a["W"], a["pl"], a["fl"], a["oc"]
    gR = np.array([fl[i] - pl[i] for i in R]); gW = np.array([fl[i] - pl[i] for i in W])
    rng = np.random.default_rng(seed)
    bR = np.array([rng.choice(gR, len(gR)).mean() for _ in range(N_BOOT)])
    bW = np.array([rng.choice(gW, len(gW)).mean() for _ in range(N_BOOT)])
    x = np.r_[gR, -gW]; lab = np.r_[np.ones(len(gR)), np.zeros(len(gW))].astype(bool)
    obs = gR.mean() + gW.mean()
    perm = np.array([(lambda lp: x[lp].mean() - x[~lp].mean())(rng.permutation(lab)) for _ in range(N_PERM)])
    with np.errstate(invalid="ignore", divide="ignore"):
        be_b = -bW / (bR - bW)
    fr = [oc[i][0] for i in R if pl[i] == 0 and i in oc]
    fw = [oc[i][1] for i in W if pl[i] == 1 and i in oc]
    fa = [oc[i][2] for i in W if pl[i] == 1 and i in oc]
    return {"seeds": a["seeds"], "n_R": len(R), "n_W": len(W),
            "dR": 100 * gR.mean(), "dR_ci": [100 * v for v in _q(bR)], "dR_p1": _sf1(gR, +1),
            "dW": 100 * gW.mean(), "dW_ci": [100 * v for v in _q(bW)], "dW_p1": _sf1(gW, -1),
            "sum": 100 * obs, "sum_ci": [100 * v for v in _q(bR + bW)],
            "sum_p": float((np.sum(np.abs(perm) >= abs(obs) - 1e-12) + 1) / (N_PERM + 1)),
            "net_even": 50 * obs,
            "breakeven": float(-gW.mean() / (gR.mean() - gW.mean())) if gR.mean() != gW.mean() else None,
            "breakeven_ci": _q(be_b),
            "f_R": _boot_mean(fr, seed), "f_W": _boot_mean(fw, seed), "abstain_W": _boot_mean(fa, seed),
            "plain_R": 100 * float(np.mean([pl[i] for i in R])), "plain_W": 100 * float(np.mean([pl[i] for i in W]))}


# --------------------------------------------------------------------------- C1 probes
def _probe_rows(part, model):
    p = RUNS / "probe21" / f"{part}.{model}.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else []


def _eval_direction(short):
    """The detector's ConflictBank eval direction measures (single-side and §11.3) over the 300 conflict items."""
    run = {"mini": "cb_single", "haiku": "cross_cb_mini_haiku", "luna": "cross_cb_mini_luna",
           "terra": "cross_cb_mini_terra"}[short]
    ex = {i: e for i, e in _examples(RUNS / run).items() if e.has_conflict}
    det = _dets(run, "llm_dir")
    return ex, det, run


def c1() -> dict:
    out = {}
    for short in ("mini", "haiku", "luna", "terra"):
        o = {}
        ao = _probe_rows("answer_only", short)
        if ao:
            by = defaultdict(dict)
            for r in ao:
                by[r["id"]][r["order"]] = r
            items = [v for v in by.values() if len(v) == 2]
            score = lambda r: 0.5 if r["choice"] is None else float(r["correct"])
            acc = [np.mean([score(v["gold_first"]), score(v["wrong_first"])]) for v in items]
            chose_gold = lambda r: None if r["choice"] is None else r["correct"]
            cons = [chose_gold(v["gold_first"]) == chose_gold(v["wrong_first"]) for v in items
                    if None not in (chose_gold(v["gold_first"]), chose_gold(v["wrong_first"]))]
            o["answer_only"] = {"accuracy": _boot_mean(acc), "n_items": len(items),
                                "unparsed": float(np.mean([r["choice"] is None for r in ao])),
                                "same_both_orders": float(np.mean(cons)) if cons else None,
                                "chose_A": float(np.mean([r["choice"] == "A" for r in ao if r["choice"]])),
                                "cost_usd": float(sum(r["cost_usd"] for r in ao if not r["cached"]))}
        sp = _probe_rows("single_passage", short)
        if sp:
            by = defaultdict(dict)
            for r in sp:
                by[r["id"]][r["type"]] = (50 if r["prob_false"] is None else r["prob_false"], r["has_conflict"])
            conf = [v for v in by.values() if "misinfo" in v]
            win = [1.0 if v["misinfo"][0] > v["correct"][0] else 0.5 if v["misinfo"][0] == v["correct"][0] else 0.0
                   for v in conf]
            hi = lambda typ, cflag: [float(v[typ][0] >= 50) for v in by.values() if typ in v and v[typ][1] == cflag]
            o["single_passage"] = {"pairwise": _boot_mean(win), "n_items": len(conf),
                                   "share_false_misinfo": _boot_mean(hi("misinfo", True)),
                                   "share_false_correct_conflict": _boot_mean(hi("correct", True)),
                                   "share_false_correct_clean": _boot_mean(hi("correct", False)),
                                   "unparsed": float(np.mean([r["prob_false"] is None for r in sp])),
                                   "cost_usd": float(sum(r["cost_usd"] for r in sp if not r["cached"]))}
        run = f"swap_cb_{short}"
        if (RUNS / run / "detections.llm_dir.jsonl").exists():
            sw_ex = {i: e for i, e in _examples(RUNS / run).items()}
            sw_det = _dets(run, "llm_dir")
            ids = sorted(set(sw_ex) & set(sw_det))
            ev_ex, ev_det, ev_run = _eval_direction(short)
            c_sw = [gold_flag_counts(sw_ex[i], sw_det[i]) for i in ids]
            c_ev = [gold_flag_counts(ev_ex[i], ev_det.get(i, [])) for i in ids]
            content = _ratio([c["one_right"] for c in c_sw], [c["one_total"] for c in c_sw])
            lo, hi_ = content["ci"]
            outcome = (None if lo is None else "survives" if lo > 0.5 else "inverts" if hi_ < 0.5 else "chance")
            o["swap"] = {"n_items": len(ids), "eval_detections": ev_run,
                         "content_precision": content,
                         "direction_11_3": _ratio([c["dir_right"] for c in c_sw], [c["dir_total"] for c in c_sw]),
                         "before_single_side": _ratio([c["one_right"] for c in c_ev], [c["one_total"] for c in c_ev]),
                         "before_direction_11_3": _ratio([c["dir_right"] for c in c_ev], [c["dir_total"] for c in c_ev]),
                         "gold_pair_flagged_after": float(np.mean([c["one_total"] + c["both"] + c["unclear"] > 0 for c in c_sw])),
                         "gold_pair_flagged_before": float(np.mean([c["one_total"] + c["both"] + c["unclear"] > 0 for c in c_ev])),
                         "outcome": outcome}
        # reading rule (a): answer-only accuracy against the detector's eval direction accuracy on all 300 items
        ev_ex, ev_det, ev_run = _eval_direction(short)
        ce = [gold_flag_counts(ev_ex[i], ev_det.get(i, [])) for i in sorted(ev_ex)]
        o["eval_direction_11_3"] = _ratio([c["dir_right"] for c in ce], [c["dir_total"] for c in ce])
        closed = {"mini": "cb_single", "haiku": "cb_single_haiku", "luna": "cb_single_luna", "terra": "cb_single_terra"}[short]
        acc = _closed(closed, True)[1]
        o["eval_closedbook"] = _boot_mean([acc[i] for i in sorted(acc)])
        o["eval_single_side"] = _ratio([c["one_right"] for c in ce], [c["one_total"] for c in ce])
        if "answer_only" in o and o["eval_direction_11_3"]["est"] is not None:
            o["rule_a_no_passage_needed"] = bool(o["answer_only"]["accuracy"]["est"] >= o["eval_direction_11_3"]["est"] - 0.05)
        if "single_passage" in o:
            o["rule_b_recognisable_alone"] = bool(o["single_passage"]["pairwise"]["est"] >= 0.9)
        out[short] = o
    return out


# --------------------------------------------------------------------------- C4: H40
def c4() -> dict:
    frozen = json.loads((ROOT / "experiments/conflict/splits/predictions21.json").read_text())
    out, fam = {}, {"a": {}, "b": {}, "c": {}}
    errs, errs0, errs1 = [], [], []
    for g in ("luna", "haiku", "terra", "mini"):
        run = f"randdir_fresh_{g}"
        if not _have(run, RAND):
            out[g] = "not run"; continue
        a = arms(run)
        s = h31_summary(a)
        ids = a["R"] + a["W"]
        eff = np.array([a["fl"][i] - a["pl"][i] for i in ids])
        s["obs_all"] = _boot_mean(100 * eff)
        if g in FROZEN_CI:
            (rl, rh), (wl, wh) = FROZEN_CI[g]
            s["H40a"] = {"f_R_inside": bool(rl <= s["f_R"]["est"] <= rh), "f_W_inside": bool(wl <= s["f_W"]["est"] <= wh),
                         "frozen_ci_R": [rl, rh], "frozen_ci_W": [wl, wh]}
            s["H40a"]["holds"] = s["H40a"]["f_R_inside"] and s["H40a"]["f_W_inside"]
            f = frozen[run]
            s["H40c"] = {"pred": f["pred"], "pred_ones": f["pred_ones"], "obs": s["obs_all"]["est"],
                         "error": s["obs_all"]["est"] - f["pred"],
                         "pred_in_obs_ci": bool(s["obs_all"]["ci"][0] <= f["pred"] <= s["obs_all"]["ci"][1])}
            errs.append(abs(s["obs_all"]["est"] - f["pred"])); errs0.append(abs(s["obs_all"]["est"]))
            errs1.append(abs(s["obs_all"]["est"] - f["pred_ones"]))
            fam["a"][g] = {"p": s["dR_p1"], "diff": s["dR"]}
            fam["b"][g] = {"p": s["dW_p1"], "diff": s["dW"]}
            fam["c"][g] = {"p": s["sum_p"], "diff": s["sum"]}
        else:
            s["note"] = "outcome known before PR§21 (H36); reported, untested"
        out[g] = s
    for key, direction in (("a", +1), ("b", -1), ("c", 0)):
        adj = holm({k: v["p"] for k, v in fam[key].items()})
        for k, v in fam[key].items():
            v["p_holm"] = adj[k]
            v["supported"] = bool(adj[k] < 0.05 and (direction == 0 or np.sign(v["diff"]) == direction))
    out["H40b"] = fam
    if errs:
        out["H40c"] = {"mae": float(np.mean(errs)), "mae_zero": float(np.mean(errs0)), "mae_ones": float(np.mean(errs1)),
                       "n": len(errs), "holds": bool(np.mean(errs) <= 3)}
    return out


# --------------------------------------------------------------------------- C5: H41
def c5() -> dict:
    out, fa, fb = {}, {}, {}
    for g in ("mini", "haiku", "luna", "terra"):
        run = f"conf_cb_{g}"
        c55, c95 = f"{SYN}+flag_directional_c55", f"{SYN}+flag_directional_c95"
        if not (_have(run, c55) and _have(run, c95)):
            out[g] = "not run"; continue
        a55, a95 = arms(run, c55), arms(run, c95)
        o = {"c55": h31_summary(a55), "c95": h31_summary(a95)}
        if _have(f"randdir_cb_{g}", RAND):
            o["no_confidence"] = h31_summary(arms(f"randdir_cb_{g}"))
        W = sorted(set(a55["W"]) & set(a95["W"])); R = sorted(set(a55["R"]) & set(a95["R"]))
        dW = np.array([a55["fl"][i] - a95["fl"][i] for i in W])     # H41a: > 0
        dR = np.array([a95["fl"][i] - a55["fl"][i] for i in R])     # H41b: > 0
        o["H41a"] = {"diff": 100 * dW.mean(), "ci": [100 * v for v in _boot_mean(dW)["ci"]], "p": _sf1(dW, +1), "n": len(W)}
        o["H41b"] = {"diff": 100 * dR.mean(), "ci": [100 * v for v in _boot_mean(dR)["ci"]], "p": _sf1(dR, +1), "n": len(R)}
        fa[g], fb[g] = o["H41a"], o["H41b"]
        out[g] = o
    for fam in (fa, fb):
        adj = holm({k: v["p"] for k, v in fam.items()})
        for k, v in fam.items():
            v["p_holm"] = adj[k]; v["supported"] = bool(adj[k] < 0.05 and v["diff"] > 0)
    return out


# --------------------------------------------------------------------------- C6, C8
def c6_c8() -> dict:
    out = {}
    for g, run in (("sonnet", "randdir_cb_sonnet"), ("qwen", "randdir_cb_qwen"), ("terra", "randdir_cb_terra")):
        out[g] = h31_summary(arms(run)) if _have(run, RAND) else "not run"
    return out


# --------------------------------------------------------------------------- C9
def _closed(run, subset=None):
    rows = [r for r in read_results(RUNS / run) if r["condition"] == "closedbook"]
    acc = per_question(rows, "closedbook", subset, PRIMARY) if rows else {}
    return rows, acc


def _follow_split(run, cond, known):
    a = arms(run, cond)
    fr = [(i, i in known, a["oc"][i][0]) for i in a["R"] if a["pl"][i] == 0 and i in a["oc"]]
    fw = [(i, i in known, a["oc"][i][1]) for i in a["W"] if a["pl"][i] == 1 and i in a["oc"]]
    return {"f_R": _split_rates(fr), "f_W": _split_rates(fw), "n_known": len(known & set(a["R"] + a["W"])),
            "n_conflict": len(a["R"] + a["W"])}


def c9() -> dict:
    out = {"closedbook": {}, "R2_fresh": {}, "R4_fresh": {}}
    runs = {g: f"fresh_closed_{g}" for g in ("mini", "luna", "haiku", "terra", "qwen")}
    runs["sonnet_eval"] = "cb_closed_sonnet"
    known = {}
    for g, run in runs.items():
        if not _have(run, "closedbook"):
            out["closedbook"][g] = "not run"; continue
        rows, acc = _closed(run, True)
        clean = _closed(run, False)[1]
        known[g] = known_ids(rows)
        conflict = {r["id"] for r in rows if r["has_conflict"]}
        out["closedbook"][g] = {"run": run, "accuracy": _boot_mean([acc[i] for i in sorted(acc)]),
                                "accuracy_clean": _boot_mean([clean[i] for i in sorted(clean)]),
                                "n_known_conflict": len(known[g] & conflict), "n_conflict": len(conflict),
                                "seeds": sorted({r["seed"] for r in rows})}
    for d in ("mini", "luna", "haiku", "terra", "qwen"):
        det_run = f"fresh_cb_mini_{d}"
        if d not in known or not (RUNS / det_run / "detections.llm_dir.jsonl").exists():
            out["R2_fresh"][d] = "not run"; continue
        ex = {i: e for i, e in _examples(RUNS / det_run).items() if e.has_conflict}
        det = _dets(det_run, "llm_dir")
        ids = sorted(ex)
        c = [gold_flag_counts(ex[i], det.get(i, [])) for i in ids]
        grp = [i in known[d] for i in ids]
        out["R2_fresh"][d] = {"detections": det_run, "n_known": int(sum(grp)), "n_conflict": len(ids),
                              "single_side": _ratio_split([x["one_right"] for x in c], [x["one_total"] for x in c], grp),
                              "direction": _ratio_split([x["dir_right"] for x in c], [x["dir_total"] for x in c], grp)}
    if "sonnet_eval" in known and (RUNS / "cross_cb_mini_sonnet/detections.llm_dir.jsonl").exists():
        ex = {i: e for i, e in _examples(RUNS / "cross_cb_mini_sonnet").items() if e.has_conflict}
        det = _dets("cross_cb_mini_sonnet", "llm_dir")
        ids = sorted(ex)
        c = [gold_flag_counts(ex[i], det.get(i, [])) for i in ids]
        grp = [i in known["sonnet_eval"] for i in ids]
        out["R2_sonnet_eval"] = {"n_known": int(sum(grp)), "n_conflict": len(ids),
                                 "single_side": _ratio_split([x["one_right"] for x in c], [x["one_total"] for x in c], grp),
                                 "direction": _ratio_split([x["dir_right"] for x in c], [x["dir_total"] for x in c], grp)}
    for g in ("mini", "luna", "haiku", "terra"):
        run = f"randdir_fresh_{g}"
        out["R4_fresh"][g] = _follow_split(run, RAND, known[g]) if g in known and _have(run, RAND) else "not run"
    if "sonnet_eval" in known and _have("randdir_cb_sonnet", RAND):
        out["R4_sonnet"] = _follow_split("randdir_cb_sonnet", RAND, known["sonnet_eval"])
    if _have("randdir_cb_qwen", RAND):
        out["R4_qwen"] = _follow_split("randdir_cb_qwen", RAND, known_ids(_closed("cb_single_qwen")[0]))
    return out


# --------------------------------------------------------------------------- C3
def c3() -> dict:
    out = {}
    for g in ("mini", "luna", "haiku"):
        run, hc = f"hckey_{g}", f"hc_{g}"
        if not _have(run, "answer_key"):
            out[g] = "not run"; continue
        ex = _examples(RUNS / run)
        topic = {i: e.group for i, e in ex.items()}
        rows, hrows = read_results(RUNS / run), read_results(RUNS / hc)
        key, pl = per_question(rows, "answer_key", True, PRIMARY), per_question(rows, "plain", True, PRIMARY)
        odir = per_question(hrows, "oracle+flag_directional", True, PRIMARY)
        o = {"answer_key": 100 * float(np.mean(list(key.values()))), "plain": 100 * float(np.mean(list(pl.values()))),
             "oracle_dir": 100 * float(np.mean(list(odir.values())))}
        for name, base in (("key_minus_plain", pl), ("key_minus_oracle_dir", odir)):
            b = paired_bootstrap(base, key, groups=topic, n_boot=N_BOOT)
            o[name] = {k: (100 * v if isinstance(v, float) else v) for k, v in b.items()}
        out[g] = o
    return out


# --------------------------------------------------------------------------- C7, C11
def _effects(run, conds, base_run=None):
    rows = read_results(RUNS / run)
    out = {}
    for part, sub in (("conflict", True), ("clean", False)):
        pl = per_question(rows, "plain", sub, PRIMARY)
        for c in conds:
            fl = per_question(rows, c, sub, PRIMARY)
            ids = sorted(set(pl) & set(fl))
            if not ids:
                continue
            d = np.array([fl[i] - pl[i] for i in ids])
            b = _boot_mean(100 * d)
            out[f"{c}|{part}"] = {"effect": b["est"], "ci": b["ci"], "n": len(ids),
                                  "plain": 100 * float(np.mean([pl[i] for i in ids]))}
    return out


def _flag_rates(run, det):
    ex = _examples(RUNS / run)
    d = _dets(run, det)
    rate = lambda sub: float(np.mean([bool(d[i]) for i, e in ex.items() if e.has_conflict == sub and i in d]))
    return {"flag_rate_conflict": rate(True), "flag_rate_clean": rate(False)}


def _quality(run, det):
    """Pair- and question-level detection quality and direction accuracy (metrics.py) from a run's detections."""
    from .detectors import Detection, DetectorOutput
    from .metrics import direction_accuracy, pair_level, question_level
    ex = list(_examples(RUNS / run).values())
    d = _dets(run, det)
    fl = {e.id: {(x["i"], x["j"]) for x in d.get(e.id, [])} for e in ex}
    outs = {e.id: DetectorOutput([Detection(**x) for x in d.get(e.id, [])], {}) for e in ex}
    return {"pair_level": pair_level(ex, fl), "question_level": question_level(ex, fl),
            "direction_accuracy": direction_accuracy(ex, outs)}


def _breakeven(K, C):
    return -C / (K - C) if K is not None and C is not None and K > 0 > C else None


def c7() -> dict:
    if not _have("holdout_haikudet", "llm_dir+flag_directional"):
        return {"status": "not run"}
    conds = ["llm_dir+flag_directional", "llm_dir+remove_suspect"]
    out = {"haiku_detector": _effects("holdout_haikudet", conds), "mini_detector": _effects("holdout_main", conds)}
    # computed from the detections themselves: holdout_main's detector_quality.json was rewritten by a later
    # invocation that ran only caag and nli, so it no longer holds the llm_dir entry
    out["quality"] = {"haiku": _quality("holdout_haikudet", "llm_dir"), "mini": _quality("holdout_main", "llm_dir")}
    out["flags"] = {"haiku": _flag_rates("holdout_haikudet", "llm_dir"), "mini": _flag_rates("holdout_main", "llm_dir")}
    for k in ("haiku_detector", "mini_detector"):
        for c in conds:
            K = out[k].get(f"{c}|conflict", {}).get("effect"); C = out[k].get(f"{c}|clean", {}).get("effect")
            out[k][f"{c}|breakeven"] = _breakeven(K, C)
    return out


def c11() -> dict:
    if not _have("nlisent_cb", "nli_sent+flag"):
        return {"status": "not run"}
    out = {"nli_sent": _effects("nlisent_cb", ["nli_sent+flag", "nli_sent+remove"]),
           "nli_win300": _effects("rev8_cb_main", ["nli_win300+flag", "nli_win300+remove"]),
           "nli": _effects("cb_main", ["nli+flag", "nli+remove"])}
    q = json.loads((RUNS / "nlisent_cb/detector_quality.json").read_text())["nli_sent"]
    out["quality"] = {k: q.get(k) for k in ("pair_level", "question_level")}
    out["flags"] = {"nli_sent": _flag_rates("nlisent_cb", "nli_sent"), "nli_win300": _flag_rates("rev8_cb_main", "nli_win300"),
                    "nli": _flag_rates("cb_main", "nli")}
    return out


def registered21(rev: dict) -> None:
    """Called by revision7 before the FDR count: the PR§21 analyses go to rev["r21"], and the H40b and H41 tests join
    the programme-wide BH count as rev["H40b"] and rev["H41"] (revision7.fdr reads nodes with p and p_holm)."""
    r = compute()
    rev["r21"] = r
    c4, c5 = r.get("c4", {}), r.get("c5", {})
    if isinstance(c4.get("H40b"), dict) and any(c4["H40b"].values()):
        rev["H40b"] = {k: {g: {"p": v["p"], "p_holm": v["p_holm"]} for g, v in fam.items()} for k, fam in c4["H40b"].items()}
    t = {f"{g}|{k}": {"p": v[k]["p"], "p_holm": v[k]["p_holm"]} for g, v in c5.items() if isinstance(v, dict)
         for k in ("H41a", "H41b") if k in v}
    if t:
        rev["H41"] = t


PARTS = {"c1": c1, "c3": c3, "c4": c4, "c5": c5, "c6": c6_c8, "c7": c7, "c9": c9, "c11": c11}


def compute(only=None) -> dict:
    old = json.loads(OUT.read_text()) if OUT.exists() else {}
    for k, fn in PARTS.items():
        if only and k not in only:
            continue
        try:
            old[k] = fn()
        except FileNotFoundError as e:
            old[k] = {"status": f"not run ({e.filename})"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(old, indent=1, sort_keys=True, default=float))
    return old


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma list of parts: " + ",".join(PARTS))
    args = ap.parse_args(argv)
    o = compute(args.only.split(",") if args.only else None)
    print(json.dumps({k: (v if isinstance(v, str) else "computed") for k, v in o.items()}, indent=1))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
