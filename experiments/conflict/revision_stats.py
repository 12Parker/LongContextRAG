"""
Revision analyses (PREREGISTRATION.md §9): the skeleton ablation (H10), the open-weights
replication (H11), and the re-analyses of existing data in §9.3.

Called from paper_stats.main; every function takes rows/examples already loaded there.
Re-analyses of data that existed before §9 was frozen are labelled as such in the paper.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .analyze import per_question
from .detectors import parse_synth_name
from .metrics import marked_docs
from .place_detector import idw
from .stats import holm, paired_bootstrap

N_BOOT_PRED = 1000
TOST_MARGIN = 0.03   # §9.3.4: half the H2 effect, fixed before this analysis


# --------------------------------------------------------------------------- per-question helpers
def _skip(ex):
    return set(ex.meta.get("suspect_docs", [])) if ex.meta.get("label_fix") in ("exclude", "hand") else set()


def q_doc(ex, pairs, mode) -> tuple[float, float] | None:
    """Per-question document-level indicators (metrics.doc_level, one question at a time).
    Conflict question: (all misinfo marked, any correct marked); clean: (any correct marked,)."""
    types = ex.meta.get("doc_types", [])
    skip = _skip(ex)
    marked = marked_docs(pairs, mode)
    mis = {k for k, t in enumerate(types) if t == "misinfo" and k not in skip}
    cor = {k for k, t in enumerate(types) if t == "correct" and k not in skip}
    if ex.has_conflict and mis:
        return float(mis <= marked), float(bool(cor & marked))
    if ex.has_conflict is False:
        return (float(bool(cor & marked)),)
    return None


def q_pair(ex, pairs) -> np.ndarray:
    """Per-question (tp, fp, fn, direction right, direction total), as metrics.pair_level and
    metrics.direction_accuracy count them."""
    if ex.gold_pairs is None:
        return np.zeros(5)
    gold = {k for k, v in ex.gold_pairs.items() if v == "contradiction"}
    f = {(d.i, d.j) for d in pairs if ex.gold_pairs.get((d.i, d.j)) != "uncertain"}
    types = ex.meta.get("doc_types", [])
    right = total = 0
    for d in pairs:
        if ex.gold_pairs.get((d.i, d.j)) == "contradiction" and d.suspects:
            total += 1
            right += sorted(d.suspects) == [k for k in (d.i, d.j) if types[k] == "misinfo"]
    return np.array([len(f & gold), len(f - gold), len(gold - f), right, total], float)


def _pair_coords(c: np.ndarray) -> tuple:
    tp, fp, fn, r, t = c
    return (tp / (tp + fp) if tp + fp else 0.0, tp / (tp + fn) if tp + fn else 0.0, r / t if t else 0.5)


# --------------------------------------------------------------------------- generic tests
def family(tests: dict, direction: dict | None = None, alpha: float = 0.05) -> dict:
    adj = holm({k: v["p"] for k, v in tests.items() if not math.isnan(v["p"])})
    for k, v in tests.items():
        v["p_holm"] = adj.get(k, float("nan"))
        want = (direction or {}).get(k, 0)
        v["supported"] = bool(v["p_holm"] < alpha and (want == 0 or np.sign(v["diff"]) == want))
    return tests


def labelled(rows):
    """Rows of questions with a conflict label (drops the questions excluded by the label rule)."""
    return [r for r in rows if r.get("has_conflict") is not None]


def excluded(rows):
    return [r for r in rows if r.get("has_conflict") is None]


# --------------------------------------------------------------------------- H4 (§9.3.1-2)
def h4_revised(rows, examples, dets_of, run, metric, compare) -> dict:
    """False-flag harm on clean questions, by mode, with cells that differ only in direction
    accuracy collapsed (on clean questions they render the same prompts)."""
    cells = sorted({r["condition"] for r in rows if r["condition"].startswith("synth-")})
    by_key = defaultdict(list)            # (mode, p, r) -> cells over direction
    for c in cells:
        det, _, mode = c.partition("+")
        p, r, _, d = parse_synth_name(det)
        by_key[(mode, p, r)].append(c)
    # check the collapse: clean-question scores must not depend on direction accuracy
    max_gap = 0.0
    for cs in by_key.values():
        qs = [per_question(rows, c, False, metric) for c in cs]
        for q in qs[1:]:
            max_gap = max(max_gap, max(abs(q[i] - qs[0][i]) for i in q))
    rep = {k: sorted(v)[-1] for k, v in by_key.items()}   # one representative cell per key
    out = {"max_clean_gap_across_direction": max_gap, "by_mode": {}}
    for mode in sorted({k[0] for k in rep}):
        keys = [k for k in rep if k[0] == mode and k[1] < 1]
        t = family({f"p{k[1]:g}-r{k[2]:g}": compare(rows, "plain", rep[k], False, metric) for k in keys},
                   {f"p{k[1]:g}-r{k[2]:g}": -1 for k in keys})
        out["by_mode"][mode] = {"cells": t, "n": len(t), "n_harmful": sum(v["supported"] for v in t.values())}
    # H4c: remove_suspect vs flag_directional at matched (P, R), clean questions
    pairs = {}
    for (mode, p, r), c in rep.items():
        if mode == "remove_suspect" and p < 1 and ("flag_directional", p, r) in rep:
            pairs[f"p{p:g}-r{r:g}"] = compare(rows, rep[("flag_directional", p, r)], c, False, metric)
    family(pairs, {k: -1 for k in pairs})
    out["remove_vs_flag"] = {"cells": pairs, "n": len(pairs), "n_remove_worse": sum(v["supported"] for v in pairs.values())}

    # dose-response over the distinct cells, cluster bootstrap over clean questions
    ex_by = {e.id: e for e in examples if e.has_conflict is False}
    plain = per_question(rows, "plain", False, metric)
    ids = sorted(set(plain) & set(ex_by))
    expo, eff = [], []
    # only cells that contain false flags (precision < 1); precision-1 cells sit at (0, 0) by
    # construction and would inflate the correlation
    for k, c in sorted((k, c) for k, c in rep.items() if k[1] < 1):
        det = dets_of(run, c.partition("+")[0])
        sc = per_question(rows, c, False, metric)
        expo.append([q_doc(ex_by[i], det[i].pairs, k[0])[0] for i in ids])
        eff.append([sc[i] - plain[i] for i in ids])
    expo, eff = np.array(expo), np.array(eff)           # cells x questions
    from scipy.stats import spearmanr
    rho = float(spearmanr(expo.mean(1), eff.mean(1)).statistic)
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(2000):
        b = rng.integers(0, len(ids), len(ids))
        boots.append(spearmanr(expo[:, b].mean(1), eff[:, b].mean(1)).statistic)
    lo, hi = np.nanquantile(boots, [0.025, 0.975])
    out["dose_response"] = {"spearman_rho": rho, "ci_low": float(lo), "ci_high": float(hi), "n_cells": len(rep),
                            "n_questions": len(ids)}
    return out


# --------------------------------------------------------------------------- H6 (§9.3.3)
def h6_prereg(rows, examples, dets_of, run, metric) -> dict:
    """Pre-registered H6: is the actual effect inside a 95% interval around the grid prediction?
    The interval comes from a cluster bootstrap over questions that recomputes the cell effects,
    every location (cells and detector) and the interpolation. Also: exact paired sign-flip test
    of document- vs pair-level absolute errors over the cases."""
    ex_by = {e.id: e for e in examples}
    plain = {s: per_question(rows, "plain", f, metric) for s, f in (("conflict", True), ("clean", False))}
    ids = {s: sorted(plain[s]) for s in plain}
    all_ids = sorted(set(ids["conflict"]) | set(ids["clean"]) | {e.id for e in examples})
    pos = {i: n for n, i in enumerate(all_ids)}
    conds = {r["condition"] for r in rows}
    rng = np.random.default_rng(1)
    draws = rng.integers(0, len(all_ids), size=(N_BOOT_PRED, len(all_ids)))

    def unit(det_name, mode, cond):
        """Per-question arrays over all_ids for one detector/condition."""
        det = dets_of(run, det_name)
        doc_c = np.full((len(all_ids), 2), np.nan); doc_k = np.full(len(all_ids), np.nan)
        pair = np.zeros((len(all_ids), 5))
        for i in all_ids:
            if i not in det:
                continue
            q = q_doc(ex_by[i], det[i].pairs, mode)
            if q is not None and len(q) == 2:
                doc_c[pos[i]] = q
            elif q is not None:
                doc_k[pos[i]] = q[0]
            pair[pos[i]] = q_pair(ex_by[i], det[i].pairs)
        eff = {}
        for s, f in (("conflict", True), ("clean", False)):
            sc = per_question(rows, cond, f, metric)
            e = np.full(len(all_ids), np.nan)
            for i in ids[s]:
                if i in sc:
                    e[pos[i]] = sc[i] - plain[s][i]
            eff[s] = e
        return {"doc_c": doc_c, "doc_k": doc_k, "pair": pair, "eff": eff}

    def coords(u, b, space, s):
        if space == "pair":
            return _pair_coords(u["pair"][b].sum(0))
        if s == "conflict":
            return tuple(np.nanmean(u["doc_c"][b], 0))
        return (float(np.nanmean(u["doc_k"][b])),)

    cases = {}
    for det, modes in (("caag", ["remove_suspect"]), ("nli", ["remove_suspect"]),
                       ("llm_dir", ["flag_directional", "remove_suspect"])):
        for mode in modes:
            cond = f"{det}+{mode}"
            if cond not in conds:
                continue
            cells = [c for c in sorted(conds) if c.startswith("synth-") and c.partition("+")[2] == mode
                     and parse_synth_name(c.partition("+")[0])[3] is not None]
            U = {c: unit(c.partition("+")[0], mode, c) for c in cells}
            D = unit(det, mode, cond)
            full = np.arange(len(all_ids))
            for s in ("conflict", "clean"):
                for space in ("doc", "pair"):
                    def predict(b):
                        here = coords(D, b, space, s)
                        return idw(here, [(coords(U[c], b, space, s), float(np.nanmean(U[c]["eff"][s][b])))
                                          for c in cells], 4)[0]
                    pred = predict(full)
                    actual = float(np.nanmean(D["eff"][s]))
                    bp = np.array([predict(b) for b in draws])
                    ba = np.array([np.nanmean(D["eff"][s][b]) for b in draws])
                    lo, hi = np.quantile(bp, [0.025, 0.975])
                    cases[f"{cond}|{s}|{space}"] = {
                        "predicted": pred, "pred_low": float(lo), "pred_high": float(hi), "actual": actual,
                        "inside_pred_interval": bool(lo <= actual <= hi), "abs_error": abs(pred - actual),
                        "boot_abs_error": np.abs(bp - ba)}
    # summaries and the paired error comparison
    summ = {}
    for space in ("doc", "pair"):
        v = [x for k, x in cases.items() if k.endswith("|" + space)]
        summ[space] = {"n": len(v), "inside": sum(x["inside_pred_interval"] for x in v),
                       "mae": float(np.mean([x["abs_error"] for x in v]))}
    keys = sorted({k.rsplit("|", 1)[0] for k in cases})
    d = np.array([cases[k + "|doc"]["abs_error"] - cases[k + "|pair"]["abs_error"] for k in keys])
    import itertools
    null = [abs(np.mean(d * np.array(sg))) for sg in itertools.product((-1, 1), repeat=len(d))]
    p_exact = float(np.mean([x >= abs(d.mean()) - 1e-12 for x in null]))
    bd = np.mean([cases[k + "|doc"]["boot_abs_error"] - cases[k + "|pair"]["boot_abs_error"] for k in keys], 0)
    summ["doc_minus_pair_mae"] = {"diff": float(d.mean()), "p_exact_signflip": p_exact,
                                  "ci_low": float(np.quantile(bd, 0.025)), "ci_high": float(np.quantile(bd, 0.975)),
                                  "n_doc_better": int((d < 0).sum()), "n": len(d)}
    for v in cases.values():
        v.pop("boot_abs_error")
    return {"cases": cases, "summary": summ}


# --------------------------------------------------------------------------- H5 TOST (§9.3.4)
def tost(rows, a, b, subset, metric, groups, margin=TOST_MARGIN) -> dict:
    """Equivalence at alpha=0.05: the 90% cluster-bootstrap CI of (b - a) lies inside +/- margin."""
    ra, rb = per_question(rows, a, subset, metric), per_question(rows, b, subset, metric)
    bs = paired_bootstrap(ra, rb, groups, n_boot=10_000, alpha=0.10)
    return {"diff": bs["diff"], "ci90_low": bs["ci_low"], "ci90_high": bs["ci_high"], "margin": margin,
            "equivalent": bool(-margin < bs["ci_low"] and bs["ci_high"] < margin)}


# --------------------------------------------------------------------------- H1 direction (§9.3.6)
def direction_ci(examples, dets) -> dict:
    per = np.array([q_pair(e, dets[e.id].pairs)[3:] for e in examples if e.id in dets])
    per = per[per[:, 1] > 0]
    est = per[:, 0].sum() / per[:, 1].sum()
    rng = np.random.default_rng(0)
    b = rng.integers(0, len(per), size=(10_000, len(per)))
    boots = per[b, 0].sum(1) / per[b, 1].sum(1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"estimate": float(est), "ci_low": float(lo), "ci_high": float(hi), "n_pairs": int(per[:, 1].sum()),
            "n_questions": len(per)}


# --------------------------------------------------------------------------- one significance scheme (§10)
REGISTERED = {
    # (generator, skeleton, condition, reference, subset) -> hypothesis family in R / R["rev"]
}


def _registered_keys():
    out = {}
    h3 = [("oracle+flag_directional", "plain"), ("oracle+flag_directional", "oracle+flag"),
          ("oracle+remove_suspect", "plain")]
    fams = {("gpt-4o-mini", "default"): ("H2", "H3"), ("gpt-4.1", "default"): ("H9", "H9"),
            ("gpt-4o-mini", "minimal"): ("H10", "H10"), ("gpt-4.1", "minimal"): ("H10", "H10"),
            ("Qwen2.5-7B", "minimal"): ("H10", "H10"), ("Qwen2.5-7B", "default"): ("H11", "H11")}
    for (g, sk), (f2, f3) in fams.items():
        out[(g, sk, "oracle+flag", "plain", "conflict")] = f2
        for c, ref in h3:
            out[(g, sk, c, ref, "conflict")] = f3
    for c in ("instruct", "instruct_v2"):
        for s in ("conflict", "clean"):
            out[("gpt-4o-mini", "default", c, "plain", s)] = "H7"
    return out


def _family_lookup(R, fam, key, metric):
    """Find the registered test for `key` in family `fam` (its keys are free text)."""
    g, sk, c, ref, s = key
    src = R["rev"][fam] if fam in ("H10", "H11") else R[fam]
    tests = src[metric]
    for k, v in tests.items():
        if c not in k or f"({s})" not in k:
            continue
        if fam in ("H9", "H10", "H11") and not k.startswith(g):
            continue
        is_vs_flag = "> oracle+flag (" in k
        if (ref == "oracle+flag") != is_vs_flag:
            continue
        if c == "oracle+flag" and ("directional" in k or "remove" in k):
            continue
        return v
    return None


def comparisons(R):
    """Every paired comparison the paper reports, as (key, metric, result dict). Keys identify the
    same comparison wherever it appears, so it gets one decision everywhere."""
    rev = R.get("rev", {})
    out = []
    for m in ("ramdocs_strict_oneway", "ramdocs_strict"):
        for gen, tk in (("gpt-4o-mini", "main_table"), ("gpt-4.1", "gpt41_table")):
            for c, d in R.get(tk, {}).get(m, {}).items():
                for s, v in d.items():
                    out.append(((gen, "default", c, "plain", s), m, v))
        for gk, v in rev.get("generality", {}).items():
            g, sk = gk.split("|")
            for c, d in v["effects"][m].items():
                for s, t in d.items():
                    out.append(((g, sk, c, "plain", s), m, t))
        for gen, d in rev.get("labelled", {}).items():
            for c, e in d.get(m, {}).items():
                out.append(((gen, "default", c, "plain", "labelled"), m, e["labelled"]))
                out.append(((gen, "default", c, "plain", "excluded"), m, e["excluded"]))
        for gk, t in rev.get("real_direction", {}).get(m, {}).items():
            g, sk = gk.split("|")
            out.append(((g, sk, "llm_dir+flag_directional", "llm_dir+flag", "conflict"), m, t))
        for gk, v in rev.get("r2", {}).items():               # second revision (§10)
            for c, d in v["effects"][m].items():
                for s, t in d.items():
                    out.append(((gk, "default", c, "plain", s), m, t))
        for fam in ("H12", "H13", "H14"):
            for tk, t in rev.get(fam, {}).get(m, {}).items():
                out.append((("registered", fam, tk, "", ""), m, t))
    for c, d in rev.get("adversarial", {}).items():
        for s, v in d.items():
            out.append((("gpt-4o-mini", "default", c, "plain", s), "ramdocs_strict_oneway", v))
    for key, d in rev.get("by_answers", {}).items():
        g, sk, part = key.split("|")
        for c, v in d.items():
            if isinstance(v, dict):
                out.append(((g, sk, c, "plain", f"conflict-{part}"), "ramdocs_strict_oneway", v))
    return out


def assign_significance(R, alpha=0.05):
    """One scheme for the whole paper. A comparison that is a registered test takes that test's
    Holm-adjusted p (Holm within its family). Every other comparison is exploratory, and all of them
    together form ONE Holm family per metric. Writes p_adj / sig / registered into each result."""
    reg = _registered_keys()
    regmap = R.get("rev", {}).get("registered_map", {})
    for k, (fam, tk) in regmap.items():
        g, sk, c, ref, s = k.split("\t")
        reg[(g, sk, c, ref, s)] = (fam, tk)
    rows = comparisons(R)
    explo = {}
    for key, m, v in rows:
        if key in reg or key[0] == "registered" or v.get("p") is None or (isinstance(v.get("p"), float) and math.isnan(v["p"])):
            continue
        if v.get("ci_low") == v.get("ci_high") == 0:     # identical prompts (e.g. oracle on clean q.)
            continue
        explo.setdefault(m, {})[key] = v["p"]
    adj = {m: holm(ps) for m, ps in explo.items()}
    for key, m, v in rows:
        if key[0] == "registered":
            v["registered"] = key[1]; v["p_adj"] = v["p_holm"]; v["sig"] = bool(v["p_holm"] < alpha)
            continue
        if key in reg:
            t = (R["rev"][reg[key][0]][m].get(reg[key][1]) if isinstance(reg[key], tuple)
                 else _family_lookup(R, reg[key], key, m))
            v["registered"] = reg[key] if isinstance(reg[key], str) else reg[key][0]
            v["p_adj"] = t["p_holm"] if t else float("nan")
            v["sig"] = bool(t and t["p_holm"] < alpha)
        else:
            v["registered"] = None
            v["p_adj"] = adj.get(m, {}).get(key, float("nan"))
            v["sig"] = bool(v["p_adj"] < alpha) if not math.isnan(v["p_adj"]) else False
    R.setdefault("rev", {})["n_exploratory"] = {m: len(p) for m, p in explo.items()}
    return R


# --------------------------------------------------------------------------- break-even precision (§5)
def break_even(grid_cells: dict, mode: str, recall: float, direction: float, subset: str = "all"):
    """Pre-registered break-even precision: the lowest precision at and above which every cell
    (this mode, recall, direction) is a significant gain (Holm within the mode x subset table).
    None if not even precision 1 qualifies."""
    best = None
    for p in (1.0, 0.75, 0.5, 0.25):
        v = grid_cells.get(f"{mode}|{p:g}|{recall:g}|{direction:g}|{subset}")
        if not v or not (v["p_holm"] < 0.05 and v["diff"] > 0):
            break
        best = p
    return best


# --------------------------------------------------------------------------- prevalence
def prevalence(d_conf: float, d_clean: float) -> dict:
    """Net effect as a function of the share pi of conflict questions: pi*d_conf + (1-pi)*d_clean.
    break_even_pi: the conflict share above which the net effect is positive (None if it never
    or always is)."""
    if d_conf > 0 > d_clean:
        pi = -d_clean / (d_conf - d_clean)
    else:
        pi = None
    return {"d_conf": d_conf, "d_clean": d_clean, "break_even_pi": pi,
            "net_at": {f"{q:.2f}": q * d_conf + (1 - q) * d_clean for q in (0.1, 0.25, 0.5)}}
