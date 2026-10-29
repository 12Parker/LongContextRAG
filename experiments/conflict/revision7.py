"""
Seventh revision: re-analyses of existing data asked for by the seventh mock review (no new runs). Every result
here uses data already seen and is reported as exploratory.

- pooled: H30a/b and H23 pooled over cells, with a joint question bootstrap (cells share questions) and a TOST.
- h31x: break-even direction accuracy per generator, the net effect at several direction accuracies, follow
  rates with CIs, headroom-normalised effects.
- predict: each crossed cell's flag effect predicted from H31's follow rates and the cell's own detections and
  plain outcomes, against the observed effect.
- h24_frozen: how much the crossed cells' detections differ across generators, and the H24 recast refitted on
  the questions where a detector's output is identical in all its cells.
- fdr: Benjamini-Hochberg q-values over every registered test, next to the Holm decisions.
- prevalence: flag effects reweighted to a 5% and 20% share of conflict questions.
- raguard_risk: correct / misinformation / abstention on RAGuard.

Called from revision6 before it drops H24's per-question data.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict

import numpy as np

from .analyze import per_question
from .jsonl import results_path
from .revision5 import SHORT_GEN, _ols, _unk

N_BOOT7 = 5000
TOST7 = 3.0
PREV = (0.05, 0.20)


def _ps():
    from . import paper_stats as ps
    return ps


def _wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


# --------------------------------------------------------------------------- pooled H30 / H23 with a joint bootstrap
def _pool(cells, seed=0):
    """cells: list of (bench, {qid: b - a}). The pooled estimate is the mean over cells of the cell means. Questions
    are resampled once per replicate within each benchmark and the same draw is applied to every cell of that
    benchmark, so the CI respects shared questions; the p-value flips the sign of a question's differences in
    every cell at once."""
    rng = np.random.default_rng(seed)
    by_b = defaultdict(list)
    for b, d in cells:
        by_b[b].append(d)
    qs = {b: sorted(set().union(*[set(d) for d in ds])) for b, ds in by_b.items()}
    mats = {}
    for b, ds in by_b.items():
        M = np.full((len(ds), len(qs[b])), np.nan)
        pos = {q: k for k, q in enumerate(qs[b])}
        for ci, d in enumerate(ds):
            M[ci, [pos[q] for q in d]] = list(d.values())
        mats[b] = M

    def est(idx):
        return float(np.mean(np.concatenate([np.nanmean(mats[b][:, idx[b]], 1) for b in mats])))
    full = {b: np.arange(len(qs[b])) for b in mats}
    obs = est(full)
    boots = np.array([est({b: rng.integers(0, len(qs[b]), len(qs[b])) for b in mats}) for _ in range(N_BOOT7)])
    flips = []
    for _ in range(N_BOOT7):
        cell_means = []
        for b, M in mats.items():
            s = rng.choice([-1.0, 1.0], M.shape[1])
            cell_means.append(np.nanmean(M * s, 1))
        flips.append(np.mean(np.concatenate(cell_means)))
    flips = np.abs(np.array(flips))
    lo90, hi90 = np.quantile(boots, [0.05, 0.95])
    return {"diff": 100 * obs, "ci_low": 100 * float(np.quantile(boots, 0.025)), "ci_high": 100 * float(np.quantile(boots, 0.975)),
            "ci90_low": 100 * float(lo90), "ci90_high": 100 * float(hi90),
            "p": float((np.sum(flips >= abs(obs) - 1e-12) + 1) / (len(flips) + 1)),
            "equivalent": bool(-TOST7 < 100 * lo90 and 100 * hi90 < TOST7), "n_cells": len(cells),
            "n_ahead": int(sum(np.nanmean(list(d.values())) > 0 for _, d in cells)),
            "n_behind": int(sum(np.nanmean(list(d.values())) < 0 for _, d in cells))}


def _diff(rows, a, b, metric):
    ra, rb = per_question(rows, a, True, metric), per_question(rows, b, True, metric)
    return {i: rb[i] - ra[i] for i in set(ra) & set(rb)}


def pooled(rev, r2, PRIMARY):
    ps = _ps()
    q = rev.get("H24", {}).get("_q", {})
    c30a, c30b = [], []
    for (g, d), v in sorted(q.items()):
        run = ps.ROOT / f"runs/vkey_cb_{SHORT_GEN[g]}_{SHORT_GEN[d]}"
        hrun = ps.ROOT / f"runs/hint_cb_{SHORT_GEN[g]}_{SHORT_GEN[d]}"
        if not results_path(run):
            continue
        rows = [r for r in v["rows"] if r["condition"] in ("plain", "llm_dir+flag_directional")]
        rows += [r for r in ps.rows_of(run) if r["condition"] == "llm_dir+verdict_key"]
        c30a.append(("CB", _diff(rows, "llm_dir+verdict_key", "llm_dir+flag_directional", PRIMARY)))
        if results_path(hrun):
            rows += [r for r in ps.rows_of(hrun) if r["condition"] == "hint"]
            c30b.append(("CB", _diff(rows, "hint", "llm_dir+verdict_key", PRIMARY)))
    c23 = []
    for k, v in r2.items():
        if k.startswith(("RD:", "CBs:", "CBmaj:")) and "answer_key" in {r["condition"] for r in v["_rows"]}:
            c23.append((k.split(":")[0], _diff(v["_rows"], "answer_key", "oracle+flag_directional", PRIMARY)))
    out = {}
    if c30a:
        out["H30a: LLM dir. flag minus verdict key"] = _pool(c30a)
    if c30b:
        out["H30b: verdict key minus hint"] = _pool(c30b)
    if c23:
        out["H23: oracle dir. flag minus answer key"] = _pool(c23)
    rev["pooled7"] = out


# --------------------------------------------------------------------------- H31: break-even, nets, follow rates
H31_RUNS = (("gpt-4o-mini", "randdir_cb_mini"), ("gpt-6-luna", "randdir_cb_luna"),
            ("claude-haiku-5-5", "randdir_cb_haiku"), ("gpt-6.1-terra", "randdir_cb_terra"))


def _h31_arms(PRIMARY):
    from .revision6 import RAND, _cell_outcomes
    ps = _ps()
    arms = {}
    for g, run in H31_RUNS:
        p = ps.ROOT / "runs" / run
        if not results_path(p):
            continue
        rows = ps.rows_of(p)
        ex = {e.id: e for e in ps.examples_of(p) if e.has_conflict}
        det = ps.dets_of(p, RAND.split("+")[0])
        right = {i: all(ex[i].meta["doc_types"][k] == "misinfo" for x in det[i].pairs for k in x.suspects) for i in ex}
        pl = per_question(rows, "plain", True, PRIMARY); fl = per_question(rows, RAND, True, PRIMARY)
        ids = sorted(set(pl) & set(fl) & set(right))
        oc = _cell_outcomes(rows, RAND, PRIMARY)
        arms[g] = {"R": [i for i in ids if right[i]], "W": [i for i in ids if not right[i]], "pl": pl, "fl": fl, "oc": oc}
    return arms


def h31x(rev, PRIMARY):
    arms = _h31_arms(PRIMARY)
    rng = np.random.default_rng(0)
    precision = rev.get("detector_precision7", {})
    out = {}
    for g, a in arms.items():
        gR = np.array([a["fl"][i] - a["pl"][i] for i in a["R"]]); gW = np.array([a["fl"][i] - a["pl"][i] for i in a["W"]])
        pR = np.array([a["pl"][i] for i in a["R"]]); pW = np.array([a["pl"][i] for i in a["W"]])
        def be(x, y):
            return -y.mean() / (x.mean() - y.mean())
        bs = np.array([be(rng.choice(gR, len(gR)), rng.choice(gW, len(gW))) for _ in range(N_BOOT7)])
        d_obs = len(a["R"]) / (len(a["R"]) + len(a["W"]))
        nets = {f"{d:.2f}": 100 * (d * gR.mean() + (1 - d) * gW.mean()) for d in (0.5, d_obs, 0.7, 0.8, 0.9)}
        for det, p in precision.items():
            nets[f"at {det} ({p:.3f})"] = 100 * (p * gR.mean() + (1 - p) * gW.mean())
        fr = [a["oc"][i][0] for i in a["R"] if a["pl"][i] == 0 and i in a["oc"]]
        fw = [a["oc"][i][1] for i in a["W"] if a["pl"][i] == 1 and i in a["oc"]]
        # follow rates over questions where following is possible; CI: bootstrap over questions
        bfr = [np.mean(rng.choice(fr, len(fr))) for _ in range(N_BOOT7)] if fr else [np.nan]
        bfw = [np.mean(rng.choice(fw, len(fw))) for _ in range(N_BOOT7)] if fw else [np.nan]
        out[g] = {"breakeven": float(be(gR, gW)), "be_low": float(np.quantile(bs, 0.025)), "be_high": float(np.quantile(bs, 0.975)),
                  "d_obs": d_obs, "net": nets,
                  "right": 100 * gR.mean(), "wrong": 100 * gW.mean(),
                  "headroom_right": float(gR.mean() / max(1 - pR.mean(), 1e-9)),
                  "headroom_wrong": float(-gW.mean() / max(pW.mean(), 1e-9)),
                  "plain_right_arm": 100 * pR.mean(), "plain_wrong_arm": 100 * pW.mean(),
                  "follow_right": float(np.mean(fr)) if fr else None,
                  "follow_right_ci": [float(np.quantile(bfr, 0.025)), float(np.quantile(bfr, 0.975))], "n_fr": len(fr),
                  "follow_wrong": float(np.mean(fw)) if fw else None,
                  "follow_wrong_ci": [float(np.quantile(bfw, 0.025)), float(np.quantile(bfw, 0.975))], "n_fw": len(fw)}
    rev["H31x"] = out
    return arms


# --------------------------------------------------------------------------- detector direction precision on ConflictBank
def detector_precision(rev):
    """Direction precision (among flags of the gold pair, the share naming exactly the false side), pooled over a
    detector's crossed cells, with a Wilson CI over flagged question-cells. Kept distinct from pair precision."""
    q = rev.get("H24", {}).get("_q", {})
    k, n = defaultdict(float), defaultdict(float)
    for (g, d), v in q.items():
        k[d] += float(np.sum(v["d"])); n[d] += float(np.sum(v["r"]))
    out = {d: k[d] / n[d] for d in k if n[d]}
    rev["detector_precision7"] = out
    rev["detector_precision7_ci"] = {d: _wilson(k[d], n[d]) for d in out}


# --------------------------------------------------------------------------- predicting crossed cells from H31
def _fit_stats(rows):
    P = np.array([r["pred"] for r in rows]); O = np.array([r["obs"] for r in rows])
    b, *_ = np.linalg.lstsq(np.c_[np.ones(len(P)), P], O, rcond=None)
    rng = np.random.default_rng(0)
    rs = [np.corrcoef(P[i], O[i])[0, 1] for i in (rng.integers(0, len(P), len(P)) for _ in range(N_BOOT7))]
    gens = sorted({r["gen"] for r in rows})
    gm = {g: (np.mean([r["pred"] for r in rows if r["gen"] == g]), np.mean([r["obs"] for r in rows if r["gen"] == g])) for g in gens}
    pw = np.array([r["pred"] - gm[r["gen"]][0] for r in rows]); ow = np.array([r["obs"] - gm[r["gen"]][1] for r in rows])
    return {"n": len(rows), "r": float(np.corrcoef(P, O)[0, 1]),
            "r_ci": [float(np.nanquantile(rs, 0.025)), float(np.nanquantile(rs, 0.975))],
            "r_within": float(np.corrcoef(pw, ow)[0, 1]) if len(gens) > 1 else None,
            "intercept": float(b[0]), "slope": float(b[1]), "mae": float(np.mean(np.abs(P - O))),
            "sign_agree": int(np.sum(np.sign(P) == np.sign(O))),
            "harm_called": int(np.sum((P < 0) & (O < 0))), "harm_obs": int(np.sum(O < 0)), "harm_pred": int(np.sum(P < 0))}


def predict(rev, arms):
    """Predicted flag effect of a crossed cell, from H31's follow rates (estimated on other runs, with the oracle
    pair) and the cell's own detections and plain outcomes: mean over conflict questions of
    right_i (1 - plain_i) f_right - wrongside_i plain_i f_wrong. Compared with the observed effect. Generators with
    an H31 run use their own follow rates ("own"); the others use the mean of the four ("transfer")."""
    q = rev.get("H24", {}).get("_q", {})
    f = {g: v for g, v in rev.get("H31x", {}).items() if v["follow_right"] is not None}
    if not f:
        return
    mean_f = (float(np.mean([v["follow_right"] for v in f.values()])), float(np.mean([v["follow_wrong"] for v in f.values()])))
    rows = []
    for (g, d), v in sorted(q.items()):
        fr, fw = (f[g]["follow_right"], f[g]["follow_wrong"]) if g in f else mean_f
        D, Gp, W, Y = (np.array(v[x], float) for x in ("d", "g", "w", "y1"))
        pred = np.mean(D * (1 - Gp) * fr - W * Gp * fw)
        rows.append({"gen": g, "det": d, "pred": 100 * float(pred), "obs": 100 * float(np.mean(Y - Gp)),
                     "kind": "own" if g in f else "transfer"})
    own = [r for r in rows if r["kind"] == "own"]; tr = [r for r in rows if r["kind"] == "transfer"]
    if len(own) < 3:
        return
    rev["predict7"] = {"cells": rows, **_fit_stats(own), "transfer": _fit_stats(tr) if len(tr) >= 3 else None,
                       "all": _fit_stats(rows), "mean_follow": mean_f}


# --------------------------------------------------------------------------- H24 on identical detections
def h24_frozen(rev):
    ps = _ps()
    q = rev.get("H24", {}).get("_q", {})
    if len(q) < 4:
        return
    sig = {}
    for c, v in q.items():
        det = ps.dets_of(v["run"], "llm_dir")
        sig[c] = {i: tuple(sorted((x.i, x.j, tuple(sorted(x.suspects))) for x in det[i].pairs if x.type == "contradiction"))
                  for i in v["ids"] if i in det}
    by_det = defaultdict(list)
    for (g, d) in q:
        by_det[d].append((g, d))
    same = {}
    for d, cs in by_det.items():
        ids = set.intersection(*[set(sig[c]) for c in cs])
        same[d] = {i for i in ids if len({sig[c][i] for c in cs}) == 1}
    share = {d: len(same[d]) / len(set.intersection(*[set(sig[c]) for c in by_det[d]])) for d in by_det}
    # refit the §16.4.3 model A ~ P + R + generator FE on each cell's identical-detection questions
    cells = sorted(q)
    gens = sorted({g for g, _ in cells})
    rows = []
    for c in cells:
        v = q[c]; keep = [k for k, i in enumerate(v["ids"]) if i in same[c[1]]]
        if not keep:
            continue
        D, R_, A, G = (np.array(v[x], float)[keep] for x in ("d", "r", "y1", "g"))
        rows.append((c, D.mean() / max(R_.mean(), 1e-9), R_.mean(), (A - G).mean(), (np.array(v["y1"]) - np.array(v["g"])).mean()))
    X = np.c_[[r[1] for r in rows], [r[2] for r in rows], np.eye(len(gens))[[gens.index(r[0][0]) for r in rows]]]
    y = np.array([r[3] for r in rows])
    coef = _ols(X, y)
    gain_full = np.array([r[4] for r in rows]); gain_same = y
    rev["H24_frozen"] = {"share_identical": share, "n_cells": len(rows),
                         "coef_P": 100 * float(coef[0]), "coef_R": 100 * float(coef[1]),
                         "max_gain_shift": 100 * float(np.max(np.abs(gain_full - gain_same))),
                         "r_gain_full_same": float(np.corrcoef(gain_full, gain_same)[0, 1])}


# --------------------------------------------------------------------------- Benjamini-Hochberg over registered tests
def _bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n); out[o] = np.minimum(q, 1)
    return out


def fdr(R):
    tests = []

    def walk(node, path):
        if isinstance(node, dict):
            if "p" in node and "p_holm" in node and isinstance(node.get("p"), float) and not math.isnan(node["p"]):
                tests.append((path, node)); return
            for k, v in node.items():
                if not str(k).startswith("_"):
                    walk(v, path + (str(k),))
    for k, v in R.items():
        if re.match(r"^H\d", str(k)):
            walk(v, (k,))
    for k, v in R.get("rev", {}).items():
        if re.match(r"^H\d", str(k)):
            walk(v, ("rev", k))
    # primary metric only (the robustness metric repeats each test)
    tests = [(p, n) for p, n in tests if "ramdocs_strict" not in p and "robust" not in "/".join(p)]
    if not tests:
        return
    q = _bh([n["p"] for _, n in tests])
    for (path, n), qq in zip(tests, q):
        n["q_bh"] = float(qq)
    holm_sig = sum(n["p_holm"] < 0.05 for _, n in tests); bh_sig = sum(n["q_bh"] < 0.05 for _, n in tests)
    R["rev"]["fdr7"] = {"n_tests": len(tests), "holm_sig": int(holm_sig), "bh_sig": int(bh_sig),
                        "holm_not_bh": int(sum(n["p_holm"] < 0.05 and n["q_bh"] >= 0.05 for _, n in tests)),
                        "bh_not_holm": int(sum(n["q_bh"] < 0.05 and n["p_holm"] >= 0.05 for _, n in tests))}


# --------------------------------------------------------------------------- prevalence reweighting
def prevalence(rev, S, PRIMARY):
    out = {}
    for s in S:
        rows = s["rows"]; have = {r["condition"] for r in rows}
        if not any(r.get("has_conflict") is False for r in rows):
            continue
        for cond in ("oracle+flag", "oracle+flag_directional", "llm_dir+flag_directional"):
            if cond not in have:
                continue
            dc = _diff(rows, "plain", cond, PRIMARY)
            ra = per_question(rows, "plain", False, PRIMARY); rb = per_question(rows, cond, False, PRIMARY)
            dn = [rb[i] - ra[i] for i in set(ra) & set(rb)]
            if not dc or not dn:
                continue
            c, n = float(np.mean(list(dc.values()))), float(np.mean(dn))
            out[f"{s['name']}|{cond}"] = {"conflict": 100 * c, "clean": 100 * n,
                                          **{f"pi{p:.2f}": 100 * (p * c + (1 - p) * n) for p in PREV},
                                          "breakeven_pi": (-n / (c - n)) if (c - n) and -n / (c - n) > 0 else None}
    rev["prevalence7"] = out


# --------------------------------------------------------------------------- RAGuard risk
def raguard_risk(rev, r2, PRIMARY):
    out = {}
    for k, v in r2.items():
        if not k.startswith("RG:"):
            continue
        for cond in ("closedbook", "plain", "oracle+flag", "oracle+flag_directional", "llm_dir+flag_directional"):
            rs = [r for r in v["_rows"] if r["condition"] == cond and r.get("has_conflict")]
            if not rs:
                continue
            n = len(rs); ans = [r for r in rs if not _unk(r)]
            out[f"{k}|{cond}"] = {"correct": 100 * sum(r[PRIMARY] for r in rs) / n,
                                  "misinfo": 100 * sum(r["wrong_rate"] > 0 for r in rs) / n,
                                  "abstain": 100 * (n - len(ans)) / n,
                                  "answered_acc": 100 * sum(r[PRIMARY] for r in ans) / max(len(ans), 1)}
    rev["raguard7"] = out


def revision7(R, r2, S, PRIMARY):
    rev = R["rev"]
    detector_precision(rev)
    pooled(rev, r2, PRIMARY)
    arms = h31x(rev, PRIMARY)
    predict(rev, arms)
    h24_frozen(rev)
    prevalence(rev, S, PRIMARY)
    raguard_risk(rev, r2, PRIMARY)
    from .revision10 import revision10
    revision10(R, r2, S, PRIMARY)   # §19 (prospective test, full judge, HealthContradict); before the FDR count
    fdr(R)
    from .revision8 import revision8
    revision8(R, r2, S, PRIMARY, arms)   # eighth-review re-analyses (§17.1); exploratory, outside the FDR count
    from .revision9 import revision9
    revision9(R, r2, S, PRIMARY)   # ninth-review re-analyses (§18); exploratory, outside the FDR count
