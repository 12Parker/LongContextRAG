"""
Fifth revision (PREREGISTRATION.md §15): H27 (detector answer as hint), H28 (LLM flags on genuine conflicts),
H29 (artefact probe), and the §15.4 re-analyses of existing data:
H24 without the shared -G coupling, TOST for the null claims, a dependence-aware pooled neutral-flag estimate,
an abstention-aware utility, transition matrices, H21(c) as a ratio estimand, and descriptive checks.

Called from paper_stats.revision2 after revision4, while r2 still holds the per-set rows.
"""
from __future__ import annotations

import itertools
import json
import math
import re
import warnings
from collections import defaultdict

import numpy as np

from .analyze import per_question
from .jsonl import results_path
from .stats import paired_bootstrap, sign_flip_p

SHORT_GEN = {"gpt-4o-mini": "mini", "gpt-6-luna": "luna", "gpt-4.1": "four", "gpt-6.1-terra": "terra",
             "claude-haiku-5-5": "haiku", "Qwen2.5-7B": "qwen", "claude-sonnet-5-5": "sonnet"}
FAMILY = {"gpt-4o-mini": "OpenAI", "gpt-6-luna": "OpenAI", "gpt-4.1": "OpenAI", "gpt-6.1-terra": "OpenAI",
          "claude-haiku-5-5": "Anthropic", "claude-sonnet-5-5": "Anthropic", "Qwen2.5-7B": "Qwen"}
# the detector model's own ConflictBank single-prompt run (§15.1): its plain seed-0 answer is the hint
DET_RUN = {"gpt-4o-mini": "cb_single", "gpt-4.1": "cb_single_gpt41", "gpt-6-luna": "cb_single_luna",
           "gpt-6.1-terra": "cb_single_terra", "claude-haiku-5-5": "cb_single_haiku",
           "claude-sonnet-5-5": "cb_single_sonnet"}
# cells that existed when §13 (H24) was committed (§15.4.1d)
PRE_REG_CELLS = {(g, "gpt-4o-mini") for g in ("gpt-4o-mini", "gpt-6-luna", "gpt-6.1-terra", "gpt-4.1", "Qwen2.5-7B")}
N_BOOT5 = 2000
TOST_MARGIN = 0.03


def _ps():
    from . import paper_stats as ps   # late import: paper_stats imports this module
    return ps


def _family(tests, dirs=None):
    return _ps().family(tests, dirs or {})


def _unk(r):
    return any(a.strip().lower().rstrip(".") == "unknown" for a in r.get("pred_answers", []))


def _outcome(r, metric):
    if r[metric] == 1:
        return "correct"
    if r["wrong_rate"] > 0:
        return "misinfo"
    return "abstain" if _unk(r) else "other"


# --------------------------------------------------------------------------- H24 re-analysis (§15.4.1)
def _ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def h24_redo(h, label):
    """h: rev['H24'] with its '_q' per-question data. Returns the §15.4.1 analyses."""
    from scipy.stats import spearmanr
    q = h.get("_q", {})
    if len(q) < 4:
        return None
    cells = sorted(q)
    qids = sorted({i for c in cells for i in q[c]["ids"]})
    pos = {i: k for k, i in enumerate(qids)}
    nC, nQ = len(cells), len(qids)
    M = {v: np.full((nC, nQ), np.nan) for v in ("d", "g", "y1", "w")}
    for ci, c in enumerate(cells):
        for v in M:
            M[v][ci, [pos[i] for i in q[c]["ids"]]] = q[c][v]
    gens = sorted({g for g, _ in cells}); dets = sorted({d for _, d in cells})
    gi = np.array([gens.index(g) for g, _ in cells]); di = np.array([dets.index(d) for _, d in cells])
    Gfe = np.eye(len(gens))[gi]

    def cell_stats(idx):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            A, D, G, W = (np.nanmean(M[v][:, idx], 1) for v in ("y1", "d", "g", "w"))
        mech = np.nanmean(M["d"][:, idx] * (1 - M["g"][:, idx]) - M["w"][:, idx] * M["g"][:, idx], 1)
        return A, D, G, W, mech

    def fits(idx):
        A, D, G, W, mech = cell_stats(idx)
        b_add = _ols(np.c_[np.ones(nC), D, G], A)                    # A ~ 1 + D + G
        b_fe = _ols(np.c_[D, Gfe], A)                                 # A ~ D + generator FE
        b_cal = _ols(np.c_[np.ones(nC), mech], A - G)                 # gain ~ 1 + mech (calibration)
        return np.r_[b_add[1:3], b_fe[0], b_cal]

    full = np.arange(nQ)
    est = fits(full)
    rng = np.random.default_rng(0)
    boots = np.array([fits(rng.integers(0, nQ, nQ)) for _ in range(N_BOOT5)])
    lo, hi = np.quantile(boots, [0.025, 0.975], axis=0)
    names = ("add_D", "add_G", "fe_D", "cal_intercept", "cal_slope")
    out = {"regression": {n: {"est": float(est[k]), "ci_low": float(lo[k]), "ci_high": float(hi[k])}
                          for k, n in enumerate(names)}}
    # structure-respecting exact permutation: the detectors' mean direction accuracies are permuted
    # across detectors, consistently in every generator; A ~ D_perm + generator FE
    A, D, G, W, mech = cell_stats(full)
    Dbar = np.array([D[di == k].mean() for k in range(len(dets))])
    stat = lambda dv: _ols(np.c_[dv[di], Gfe], A)[0]
    obs = stat(Dbar)
    perm = np.array([stat(Dbar[list(p)]) for p in itertools.permutations(range(len(dets)))])
    out["perm_fe_D"] = {"slope": float(obs), "p_one_sided": float(np.mean(perm >= obs - 1e-12)),
                        "n_perm": len(perm), "D_by_detector": dict(zip(dets, map(float, Dbar)))}
    # leave-one-detector-out prediction of the cell gains
    gain = A - G
    feats = {"G only": np.c_[np.ones(nC), G], "D + G": np.c_[np.ones(nC), D, G],
             "gap D - G (registered)": np.c_[np.ones(nC), D - G], "mechanism score": np.c_[np.ones(nC), mech]}
    cv = {}
    for nm, X in feats.items():
        pred = np.empty(nC)
        for k in range(len(dets)):
            te = di == k
            pred[te] = X[te] @ _ols(X[~te], gain[~te])
        sse = float(((gain - pred) ** 2).sum())
        cv[nm] = {"rmse": math.sqrt(sse / nC), "r2_cv": 1 - sse / float(((gain - gain.mean()) ** 2).sum())}
    cv["mechanism score, no fit"] = {"rmse": float(np.sqrt(((gain - mech) ** 2).mean())),
                                    "r2_cv": 1 - float(((gain - mech) ** 2).sum()) / float(((gain - gain.mean()) ** 2).sum())}
    out["lodo"] = cv
    # question level: linear probability model, detector and generator fixed, question random intercept;
    # and all three as crossed variance components (one group)
    out["mixed"] = _mixed(M, cells)
    # H24a on the cells that did not exist when §13 was committed
    post = [k for k, c in enumerate(cells) if c not in PRE_REG_CELLS]
    if len(post) >= 4:
        gp, gn = (D - G)[post], gain[post]
        rho = float(spearmanr(gp, gn).statistic)
        r2 = np.random.default_rng(0)
        pr = np.array([spearmanr(gp, r2.permutation(gn)).statistic for _ in range(10_000)])
        out["post_registration"] = {"n_cells": len(post), "rho": rho,
                                    "p_one_sided": float((np.sum(pr >= rho) + 1) / (len(pr) + 1)),
                                    "pre_cells": sorted(f"{g}|{d}" for g, d in PRE_REG_CELLS if (g, d) in q)}
    # compliance decomposition per generator (pooled over its cells)
    comp = {}
    for g in gens:
        rows = [k for k, (gg, _) in enumerate(cells) if gg == g]
        d_, g_, y_, w_ = (M[v][rows].ravel() for v in ("d", "g", "y1", "w"))
        ok = ~np.isnan(y_)
        need = ok & (d_ == 1) & (g_ == 0); harm = ok & (w_ == 1) & (g_ == 1)
        comp[g] = {"follow_right": float(y_[need].mean()) if need.any() else None, "n_right": int(need.sum()),
                   "follow_wrong": float(1 - y_[harm].mean()) if harm.any() else None, "n_wrong": int(harm.sum())}
    out["compliance"] = comp
    out["cells"] = {f"{g}|{d}": {"A": float(A[k]), "D": float(D[k]), "G": float(G[k]), "W": float(W[k]),
                                 "mech": float(mech[k]), "gain": float(gain[k])} for k, (g, d) in enumerate(cells)}
    out["detection_differences"] = _det_diffs(q)
    out["label"] = label
    return out


def _mixed(M, cells):
    try:
        import pandas as pd
        import statsmodels.formula.api as smf
    except ImportError:
        return None
    recs = []
    for ci, (g, d) in enumerate(cells):
        for qi in range(M["y1"].shape[1]):
            if not np.isnan(M["y1"][ci, qi]):
                recs.append((M["y1"][ci, qi], M["d"][ci, qi], M["g"][ci, qi], M["w"][ci, qi], g, d, qi))
    df = pd.DataFrame(recs, columns=["y1", "d", "g", "w", "gen", "det", "q"])
    res = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = smf.mixedlm("y1 ~ d + g + w + C(det) + C(gen)", df, groups="q").fit(reml=True)
        res["fe_det_gen_re_q"] = {k: {"est": float(m.params[k]), "se": float(m.bse[k]),
                                      "ci_low": float(m.params[k] - 1.96 * m.bse[k]),
                                      "ci_high": float(m.params[k] + 1.96 * m.bse[k]), "p": float(m.pvalues[k])}
                                  for k in ("d", "g", "w")}
        df["one"] = 1
        try:
            m2 = smf.mixedlm("y1 ~ d + g + w", df, groups="one",
                             vc_formula={"det": "0+C(det)", "gen": "0+C(gen)", "q": "0+C(q)"}).fit(reml=True)
            res["crossed_re"] = {k: {"est": float(m2.params[k]), "se": float(m2.bse[k]),
                                     "ci_low": float(m2.params[k] - 1.96 * m2.bse[k]),
                                     "ci_high": float(m2.params[k] + 1.96 * m2.bse[k])}
                                 for k in ("d", "g", "w")}
            res["crossed_re"]["converged"] = bool(m2.converged)
        except Exception as e:   # noqa: BLE001 - reported, not fatal
            res["crossed_re"] = {"error": str(e)[:200]}
    res["n_rows"] = len(df)
    return res


def _det_diffs(q):
    """How often the same detector's detections differ between the cells it serves (§15.4.1f)."""
    by_det = defaultdict(list)
    for (g, d), v in q.items():
        p = v["run"] / "detections.llm_dir.jsonl"
        by_det[d].append({json.loads(l)["id"]: json.dumps(json.loads(l)["pairs"], sort_keys=True) for l in open(p)})
    out = {}
    for d, maps in by_det.items():
        if len(maps) < 2:
            continue
        ids = set.intersection(*(set(m) for m in maps))
        n_diff = sum(len({m[i] for m in maps}) > 1 for i in ids)
        out[d] = {"n_cells": len(maps), "n_questions": len(ids), "n_differ": n_diff}
    return out


# --------------------------------------------------------------------------- H27: detector answer as hint
def h27(rev, PRIMARY, ROBUST):
    ps = _ps()
    q = rev.get("H24", {}).get("_q", {})
    det_rows = {}
    for dname, run in DET_RUN.items():
        p = ps.ROOT / "runs" / run
        if results_path(p):
            det_rows[dname] = [{**r, "condition": "det_alone"} for r in ps.rows_of(p) if r["condition"] == "plain"]
    specs = {"a": ("hint", "llm_dir+flag_directional", "LLM dir. flag vs hint"),
             "b": ("plain", "hint", "hint vs plain"),
             "c": ("det_alone", "llm_dir+flag_directional", "cell vs detector model alone")}
    out = {k: {m: {} for m in (PRIMARY, ROBUST)} for k in specs}
    clean = {k: {} for k in specs}
    for (g, d), v in sorted(q.items()):
        run = ps.ROOT / f"runs/hint_cb_{SHORT_GEN[g]}_{SHORT_GEN[d]}"
        if not results_path(run):
            continue
        rows = [r for r in v["rows"] if r["condition"] in ("plain", "llm_dir+flag_directional")]   # the cell's own rows
        rows += [r for r in ps.rows_of(run) if r["condition"] == "hint"] + det_rows.get(d, [])
        for k, (a, b, nm) in specs.items():
            tk = f"{g}, detector {d}: {nm} (conflict)"
            for m in (PRIMARY, ROBUST):
                out[k][m][tk] = ps.compare(rows, a, b, True, m)
            clean[k][tk.replace("(conflict)", "(clean)")] = ps.compare(rows, a, b, False, PRIMARY)
    for k in specs:
        if out[k][PRIMARY]:
            rev[f"H27{k}"] = {m: _family(out[k][m]) for m in (PRIMARY, ROBUST)}
            rev[f"H27{k}_clean"] = clean[k]


# --------------------------------------------------------------------------- H28: genuine conflicts
def h28(rev, main_rows, PRIMARY):
    ps = _ps()
    out = {}
    conds = (("llm_dir+flag_directional", "LLM dir. flag vs plain"), ("llm_dir+remove_suspect", "LLM removal vs plain"))
    for g, run in (("gpt-4o-mini", "wiki_llm_mini"), ("gpt-6-luna", "wiki_llm_luna"), ("claude-haiku-5-5", "wiki_llm_haiku")):
        p = ps.ROOT / "runs" / run
        if results_path(p):
            rows = ps.rows_of(p)
            fam = {f"WikiContradict, {g}: {nm}": ps.compare(rows, "plain", c, None, "primary") for c, nm in conds}
            out[f"WC:{g}"] = {"both_answers": _family(fam),
                              "plain": float(np.mean(list(per_question(rows, "plain", None, "primary").values()))),
                              "oracle_flag": ps.compare(rows, "plain", "oracle+flag", None, "primary")}
    for g, rows in (("gpt-4o-mini", main_rows),) + tuple(
            (g, ps.rows_of(ps.ROOT / "runs" / run)) for g, run in
            (("gpt-6-luna", "holdout_rem_luna"), ("claude-haiku-5-5", "holdout_rem_haiku"),
             ("gpt-6.1-terra", "holdout_rem_terra")) if results_path(ps.ROOT / "runs" / run)):
        amb = [r for r in rows if r["conflict_type"] == "ambiguity" and r["condition"] in
               ("plain", "llm_dir+flag_directional", "llm_dir+remove_suspect")]
        if not {r["condition"] for r in amb} >= {"plain", "llm_dir+remove_suspect"}:
            continue
        res = {}
        for m in (PRIMARY, "gold_recall"):
            res[m] = _family({f"RAMDocs ambiguity, {g}: {nm}": ps.compare(amb, "plain", c, None, m) for c, nm in conds})
        res["n"] = len(per_question(amb, "plain", None, PRIMARY))
        res["plain"] = {m: float(np.mean(list(per_question(amb, "plain", None, m).values()))) for m in (PRIMARY, "gold_recall")}
        out[f"RDamb:{g}"] = res
    rev["H28"] = out


# --------------------------------------------------------------------------- H29: artefact probe
def _dq(examples, dets):
    out, wrong = {}, {}
    for e in examples:
        if not e.has_conflict or e.id not in dets:
            continue
        t = e.meta["doc_types"]
        contra = {p: {k for k in p if t[k] == "misinfo"} for p, lab in e.gold_pairs.items() if lab == "contradiction"}
        out[e.id] = float(any(contra.get((x.i, x.j)) == set(x.suspects) for x in dets[e.id].pairs))
        wrong[e.id] = float(any(t[k] == "correct" for x in dets[e.id].pairs for k in x.suspects))
    return out, wrong


def h29(rev, PRIMARY):
    ps = _ps()
    orig = {"gpt-4o-mini": ps.ROOT / "runs/cb_single", "gpt-6-luna": ps.ROOT / "runs/cross_cb_mini_luna"}
    probe = {("gpt-4o-mini", v): ps.ROOT / f"runs/probe_{v}_mini" for v in ("rev", "para")}
    probe.update({("gpt-6-luna", v): ps.ROOT / f"runs/probe_{v}_lunadet" for v in ("rev", "para")})
    para_ok = set()
    pf = ps.data.DATA_DIR / "conflictbank" / "conflictbank_para.jsonl"
    n_fail = None
    if pf.exists():
        rows = [json.loads(l) for l in open(pf)]
        para_ok = {r["id"] for r in rows if all(r.get("para_check", [True]))}
        n_fail = {"passages": sum(c.count(False) for c in (r.get("para_check", []) for r in rows)),
                  "items_all_pass": len(para_ok), "items": len(rows)}
    base = {}
    for det, run in orig.items():
        if results_path(run):
            base[det] = (_dq(ps.examples_of(run), ps.dets_of(run, "llm_dir")), ps.rows_of(run))
    a_tests, b_tests, c_tests, sub, desc = {}, {}, {}, {}, {}
    for (det, v), run in probe.items():
        if det not in base or not results_path(run):
            continue
        (d0, w0), rows0 = base[det]
        d1, w1 = _dq(ps.examples_of(run), ps.dets_of(run, "llm_dir"))
        tk = f"{det} detector, {'reversed order' if v == 'rev' else 'paraphrased'}"
        ids = sorted(set(d0) & set(d1))
        a = {i: d0[i] for i in ids}; b = {i: d1[i] for i in ids}
        bs = paired_bootstrap(a, b, n_boot=10_000)
        a_tests[tk] = {**bs, "p": sign_flip_p(a, b, n_perm=10_000, max_perm=200_000),
                       "mean_a": float(np.mean(list(a.values()))), "mean_b": float(np.mean(list(b.values())))}
        b90 = paired_bootstrap(a, b, n_boot=10_000, alpha=0.10)
        b_tests[tk] = {"diff": b90["diff"], "ci90_low": b90["ci_low"], "ci90_high": b90["ci_high"], "margin": 0.05,
                       "equivalent": bool(-0.05 < b90["ci_low"] and b90["ci_high"] < 0.05)}
        desc[tk] = {"wrong_side_orig": float(np.mean([w0[i] for i in ids])), "wrong_side_probe": float(np.mean([w1[i] for i in ids])),
                    "dir_orig": a_tests[tk]["mean_a"], "dir_probe": a_tests[tk]["mean_b"], "n": len(ids)}
        if v == "para" and para_ok:
            ids2 = [i for i in ids if i in para_ok]
            a2 = {i: d0[i] for i in ids2}; b2 = {i: d1[i] for i in ids2}
            sub[tk] = {**paired_bootstrap(a2, b2, n_boot=10_000), "n": len(ids2)}
        # H29c: difference in differences of the generator's gain (gpt-4o-mini generator)
        rows1 = ps.rows_of(run)
        if not {r["condition"] for r in rows1} >= {"plain", "llm_dir+flag_directional"}:
            continue
        g0 = {i: x - y for i, x, y in _pair(per_question(rows0, "llm_dir+flag_directional", True, PRIMARY),
                                             per_question(rows0, "plain", True, PRIMARY))}
        g1 = {i: x - y for i, x, y in _pair(per_question(rows1, "llm_dir+flag_directional", True, PRIMARY),
                                             per_question(rows1, "plain", True, PRIMARY))}
        if not set(g0) & set(g1):
            continue
        bs = paired_bootstrap(g0, g1, n_boot=10_000)
        c_tests[tk] = {**bs, "p": sign_flip_p(g0, g1, n_perm=10_000, max_perm=200_000),
                       "gain_orig": float(np.mean([g0[i] for i in set(g0) & set(g1)])),
                       "gain_probe": float(np.mean([g1[i] for i in set(g0) & set(g1)])),
                       "plain_probe": float(np.mean(list(per_question(rows1, "plain", True, PRIMARY).values())))}
    if a_tests:
        rev["H29"] = {"a": _family(a_tests), "b_tost": b_tests, "c": _family(c_tests) if c_tests else {},
                      "para_subset": sub, "desc": desc, "para_check": n_fail}


def _pair(x, y):
    return [(i, x[i], y[i]) for i in sorted(set(x) & set(y))]


# --------------------------------------------------------------------------- TOST for null claims (§15.4.2)
def tost5(rev, r2, PRIMARY):
    from . import revision_stats as rv
    ps = _ps()
    out = {}
    for key, v in r2.items():
        rows = v["_rows"]; have = {r["condition"] for r in rows}
        pairs = []
        if key.startswith("CBf:"):
            pairs += [("plain", "oracle+flag", "H20 neutral flag vs plain"), ("plain", "oracle+flag_onewrong", "H20 one-wrong flag vs plain")]
        if key.startswith(("RD:", "CBs:", "CBmaj:")) and "answer_key" in have:
            pairs += [("answer_key", "oracle+flag_directional", "H23 dir. flag vs answer key")]
        for a, b, nm in pairs:
            if a in have and b in have:
                out[f"{ps.set_label(key)}: {nm}"] = rv.tost(rows, a, b, True, PRIMARY, ps.groups_of(rows), margin=TOST_MARGIN)
    rev["tost5"] = out


# --------------------------------------------------------------------------- H21(c) as a ratio (§15.4.5)
def _sel_arrays(rows, a, b, metric):
    acc = defaultdict(lambda: np.zeros(4))   # correct_a, answered_a, correct_b, answered_b
    for r in rows:
        if not r.get("has_conflict") or r["condition"] not in (a, b):
            continue
        k = 0 if r["condition"] == a else 2
        ans = not _unk(r)
        acc[r["id"]][k] += r[metric] if ans else 0
        acc[r["id"]][k + 1] += ans
    X = np.array([v for v in acc.values()])
    return X


def selective_ratio(X, n_boot=10_000, n_perm=10_000, seed=0, alpha=0.05):
    def est(Z):
        return Z[:, 2].sum() / Z[:, 3].sum() - Z[:, 0].sum() / Z[:, 1].sum()
    if X[:, 1].sum() == 0 or X[:, 3].sum() == 0:
        return None
    d = est(X)
    rng = np.random.default_rng(seed)
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.array([est(X[rng.integers(0, len(X), len(X))]) for _ in range(n_boot)])
        swap = rng.random((n_perm, len(X))) < 0.5
        ca = np.where(swap, X[:, 2], X[:, 0]); na = np.where(swap, X[:, 3], X[:, 1])
        cb = np.where(swap, X[:, 0], X[:, 2]); nb = np.where(swap, X[:, 1], X[:, 3])
        perm = cb.sum(1) / nb.sum(1) - ca.sum(1) / na.sum(1)
    b = b[~np.isnan(b)]
    lo, hi = np.quantile(b, [alpha / 2, 1 - alpha / 2]); lo90, hi90 = np.quantile(b, [0.05, 0.95])
    p = float((np.sum(np.abs(perm) >= abs(d) - 1e-12) + 1) / (n_perm + 1))
    return {"diff": float(d), "ci_low": float(lo), "ci_high": float(hi), "p": p,
            "sel_a": float(X[:, 0].sum() / X[:, 1].sum()), "sel_b": float(X[:, 2].sum() / X[:, 3].sum()),
            "answered_a": float(X[:, 1].sum()), "answered_b": float(X[:, 3].sum()),
            "tost": {"ci90_low": float(lo90), "ci90_high": float(hi90),
                     "equivalent": bool(-TOST_MARGIN < lo90 and hi90 < TOST_MARGIN)}}


def h21_ratio(rev, r2, PRIMARY):
    tests = {}
    ps = _ps()
    for key in [f"{p}:{g}" for p in ("CBs", "CBf") for g in ("gpt-4o-mini", "gpt-6-luna", "gpt-6.1-terra", "claude-haiku-5-5")]:
        if key not in r2:
            continue
        rows = r2[key]["_rows"]
        for c, nm in (("oracle+flag", "neutral flag"), ("oracle+flag_directional", "dir. flag")):
            if any(r["condition"] == c for r in rows):
                t = selective_ratio(_sel_arrays(rows, "plain", c, PRIMARY))
                if t:
                    tests[f"{ps.set_label(key)}: {nm} vs plain (conflict)"] = t
    rev["H21c_ratio"] = _family(tests) if tests else {}


# --------------------------------------------------------------------------- utility and transitions (§15.4.4)
UTIL_C = (0, 1, 2, 4)
FLAGS = (("oracle+flag", "neutral"), ("oracle+flag_onewrong", "one-wrong"), ("oracle+flag_directional", "directional"))


def utility(rev, r2, PRIMARY):
    ps = _ps()
    out, trans = {}, {}
    for key, v in r2.items():
        if not key.startswith(("CBs:", "CBf:", "CBmaj:", "RG:", "RD:", "CB:")):
            continue
        rows = v["_rows"]
        for r in rows:
            mis = float(r["wrong_rate"] > 0)
            for c in UTIL_C:
                r[f"util{c}"] = r[PRIMARY] - c * mis
        have = {r["condition"] for r in rows}
        for cond, nm in FLAGS:
            if cond not in have:
                continue
            out.setdefault(key, {})[nm] = {f"c{c}": ps.compare(rows, "plain", cond, True, f"util{c}") for c in UTIL_C}
            # transitions plain -> flag, per question and seed
            by = {(r["id"], r["seed"], r["condition"]): _outcome(r, PRIMARY) for r in rows
                  if r.get("has_conflict") and r["condition"] in ("plain", cond)}
            cats = ("correct", "misinfo", "abstain", "other")
            Tm = {a: dict.fromkeys(cats, 0) for a in cats}
            for (i, s, c), o in by.items():
                if c == "plain" and (i, s, cond) in by:
                    Tm[o][by[(i, s, cond)]] += 1
            trans.setdefault(key, {})[nm] = Tm
    rev["utility"] = out
    rev["transitions"] = trans
    # one exploratory Holm family per cost c
    from .stats import holm
    for c in UTIL_C:
        ps_ = {(k, nm): t[f"c{c}"]["p"] for k, d in out.items() for nm, t in d.items() if not math.isnan(t[f"c{c}"]["p"])}
        adj = holm({f"{k}|{nm}": p for (k, nm), p in ps_.items()})
        for (k, nm) in ps_:
            t = out[k][nm][f"c{c}"]; t["p_holm"] = adj[f"{k}|{nm}"]; t["sig"] = bool(t["p_holm"] < 0.05)


# --------------------------------------------------------------------------- pooled neutral flag (§15.4.3)
def pooled5(rev, r2, main_rows, PRIMARY):
    ps = _ps()
    settings = []   # (name, benchmark pool, per-question diff dict, family, forced, natural)
    def add(name, rows, pool, gen, forced, natural):
        a = per_question(rows, "plain", True, PRIMARY); b = per_question(rows, "oracle+flag", True, PRIMARY)
        ids = sorted(set(a) & set(b))
        if ids:
            settings.append((name, pool, {i: b[i] - a[i] for i in ids}, FAMILY.get(gen, "OpenAI"), forced, natural))
    for k, v in r2.items():
        if k.startswith(("CB", "RD:gpt-6", "RD:claude", "RG")) and "oracle+flag" in v["effects"][PRIMARY]:
            pre, gen = k.split(":", 1)
            pool = {"RD": "RAMDocs", "RG": "RAGuard"}.get(pre, "ConflictBank")
            add(k, v["_rows"], pool, gen, pre == "CBf", pre in ("RD", "RG"))
    gen_runs = {("gpt-4o-mini", "default"): "main", ("gpt-4o-mini", "minimal"): "minimal", ("gpt-4.1", "default"): "gpt41",
                ("gpt-4.1", "minimal"): "gpt41_minimal", ("Qwen2.5-7B", "default"): "qwen", ("Qwen2.5-7B", "minimal"): "qwen_minimal"}
    for (g, sk), run in gen_runs.items():
        if f"{g}|{sk}" in rev.get("generality", {}) and "oracle+flag" in rev["generality"][f"{g}|{sk}"]["effects"][PRIMARY]:
            rows = main_rows if run == "main" else ps.rows_of(ps.RUNS[run])
            add(f"RD:{g}|{sk}", rows, "RAMDocs", g, False, True)
    pools = sorted({s[1] for s in settings})
    pool_ids = {p: sorted({i for s in settings if s[1] == p for i in s[2]}) for p in pools}
    vec = []
    for name, pool, dd, *_ in settings:
        idx = {i: k for k, i in enumerate(pool_ids[pool])}
        x = np.full(len(idx), np.nan); x[[idx[i] for i in dd]] = list(dd.values()); vec.append(x)
    Xmod = np.array([[1, s[3] == "Anthropic", s[3] == "Qwen", s[4], s[5]] for s in settings], float)

    def dl(y, se):
        w = 1 / se ** 2; ybar = (w * y).sum() / w.sum(); Q = (w * (y - ybar) ** 2).sum()
        tau2 = max(0.0, (Q - (len(y) - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
        ws = 1 / (se ** 2 + tau2)
        mu = (ws * y).sum() / ws.sum()
        beta = np.linalg.lstsq(Xmod * np.sqrt(ws)[:, None], y * np.sqrt(ws), rcond=None)[0]
        return mu, (1 / ws.sum()) ** 0.5, tau2, beta

    def stats_for(draw):
        y, se = [], []
        for (name, pool, *_), x in zip(settings, vec):
            xs = x[draw[pool]]; xs = xs[~np.isnan(xs)]
            y.append(xs.mean()); se.append(max(xs.std(ddof=1) / math.sqrt(len(xs)), 1e-6))
        return dl(np.array(y), np.array(se))

    full = {p: np.arange(len(pool_ids[p])) for p in pools}
    mu, se_mu, tau2, beta = stats_for(full)
    rng = np.random.default_rng(0)
    B = [stats_for({p: rng.integers(0, len(pool_ids[p]), len(pool_ids[p])) for p in pools}) for _ in range(N_BOOT5)]
    mus = np.array([b[0] for b in B]); betas = np.array([b[3] for b in B])
    from scipy.stats import t as tdist
    k = len(settings)
    half = tdist.ppf(0.975, k - 2) * math.sqrt(tau2 + se_mu ** 2)
    mods = ("intercept (OpenAI, unknown allowed, synthetic)", "Anthropic", "Qwen", "forced (no unknown)", "natural data")
    rev["neutral_pooled5"] = {
        "k": k, "mu": float(mu), "tau": math.sqrt(tau2),
        "ci_model": [float(mu - 1.96 * se_mu), float(mu + 1.96 * se_mu)],
        "ci_cluster_boot": [float(np.quantile(mus, 0.025)), float(np.quantile(mus, 0.975))],
        "pred_interval": [float(mu - half), float(mu + half)],
        "moderators": {m: {"est": float(beta[j]), "ci_low": float(np.quantile(betas[:, j], 0.025)),
                           "ci_high": float(np.quantile(betas[:, j], 0.975))} for j, m in enumerate(mods)},
        "settings": [{"name": s[0], "pool": s[1], "n": len(s[2]), "diff": float(np.mean(list(s[2].values()))),
                      "family": s[3], "forced": bool(s[4]), "natural": bool(s[5])} for s in settings]}


# --------------------------------------------------------------------------- descriptive checks (§15.4.6)
def descriptives(rev, r2, PRIMARY):
    ps = _ps()
    parse = {}
    for d in sorted(ps.ROOT.glob("runs/cross_cb_*_*")):
        if not d.is_dir():
            continue
        det = d.name.split("_")[-1]
        f = d / "detections.llm_dir.jsonl"
        if det in parse or not f.exists():
            continue
        n = bad = 0
        for l in open(f):
            raw = json.loads(l).get("extra", {}).get("raw", "")
            n += 1
            try:
                json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                m = re.search(r"\{.*\}", raw or "", re.S)
                try:
                    json.loads(m.group(0)) if m else (_ for _ in ()).throw(ValueError)
                except (json.JSONDecodeError, ValueError):
                    bad += 1
        parse[det] = {"n": n, "failed": bad}
    f = ps.RUNS["cb_single"] / "detections.llm_dir.jsonl"
    n = bad = 0
    for l in open(f):
        raw = json.loads(l).get("extra", {}).get("raw", ""); n += 1
        try:
            json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            bad += 1
    parse["mini"] = {"n": n, "failed": bad}
    rev["parse_failures"] = parse
    mix = {}
    for key in ("RD:gpt-6-luna", "RD:gpt-6.1-terra", "RD:claude-haiku-5-5", "RD:claude-sonnet-5-5", "RD:gpt-4o-mini"):
        if key in r2:
            rr = [r for r in r2[key]["_rows"] if r["condition"] == "plain" and r.get("has_conflict")]
            oc = [_outcome(r, PRIMARY) for r in rr]
            mix[key] = {k: oc.count(k) / len(oc) for k in ("correct", "misinfo", "abstain", "other")}
            mix[key]["gold_recall"] = float(np.mean([r["gold_recall"] for r in rr]))
            mix[key]["n_answers"] = float(np.mean([len(r.get("pred_answers", [])) for r in rr]))
    rev["rd_plain_mix"] = mix


def revision5(R, r2, fam, main_rows):
    ps = _ps()
    PRIMARY, ROBUST = ps.PRIMARY, ps.ROBUST
    rev = R["rev"]
    for k in ("H24", "H24_s13", "H24_maj"):
        if k in rev:
            red = h24_redo(rev[k], k)
            if red:
                rev[f"{k}_redo"] = red
    h27(rev, PRIMARY, ROBUST)
    h28(rev, main_rows, PRIMARY)
    h29(rev, PRIMARY)
    tost5(rev, r2, PRIMARY)
    h21_ratio(rev, r2, PRIMARY)
    utility(rev, r2, PRIMARY)
    pooled5(rev, r2, main_rows, PRIMARY)
    descriptives(rev, r2, PRIMARY)
