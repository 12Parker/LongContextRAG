"""
Sixth revision (PREREGISTRATION.md §16): H30 (the detector's verdict as an answer key), H31 (randomised direction),
and the §16.4 re-analyses: answer policy, a multilevel meta-analysis, utility over the same settings with a
seed-to-seed null, H24 with direction precision and recall, a question fixed-effects model, one definition of
following a flag with a regression-to-the-mean null, and H14 under a matched prompt.

Called from paper_stats.revision2 after revision5, while r2 and H24's per-question data still hold rows.
"""
from __future__ import annotations

import itertools
import json
import math
from collections import defaultdict

import numpy as np

from .analyze import per_question
from .jsonl import results_path
from .revision5 import FAMILY, SHORT_GEN, _ols, _outcome, _unk
from .stats import holm, paired_bootstrap, sign_flip_p

N_BOOT6 = 2000
UTIL_C = (0, 1, 2, 4)
RAND = "synth-p1-r1-all-d0.5+flag_directional"


def _ps():
    from . import paper_stats as ps
    return ps


# --------------------------------------------------------------------------- settings shared by pooling and utility
def settings(rev, r2, main_rows):
    """Every setting with an oracle neutral flag (§13.4 pooled set), with its rows and moderators (§16.4.1)."""
    ps = _ps()
    out = []
    bench = {"RD": "RAMDocs", "RG": "RAGuard", "CBmaj": "CB majority"}
    policy = {"CB": "multi", "RD": "multi", "CBn": "minimal", "CBs": "single", "CBmaj": "single", "RG": "single",
              "CBf": "forced"}
    for k, v in r2.items():
        if k.startswith(("CB", "RD:gpt-6", "RD:claude", "RG")) and "oracle+flag" in v["effects"][ps.PRIMARY]:
            pre, gen = k.split(":", 1)
            out.append({"name": k, "rows": v["_rows"], "gen": gen, "bench": bench.get(pre, "ConflictBank"),
                        "policy": policy[pre]})
    gen_runs = {("gpt-4o-mini", "default"): "main", ("gpt-4o-mini", "minimal"): "minimal", ("gpt-4.1", "default"): "gpt41",
                ("gpt-4.1", "minimal"): "gpt41_minimal", ("Qwen2.5-7B", "default"): "qwen", ("Qwen2.5-7B", "minimal"): "qwen_minimal"}
    for (g, sk), run in gen_runs.items():
        gk = f"{g}|{sk}"
        if gk in rev.get("generality", {}) and "oracle+flag" in rev["generality"][gk]["effects"][ps.PRIMARY]:
            rows = main_rows if run == "main" else ps.rows_of(ps.RUNS[run])
            out.append({"name": f"RD:{gk}", "rows": rows, "gen": g, "bench": "RAMDocs",
                        "policy": "multi" if sk == "default" else "minimal"})
    return out


def _reml(y, v, X, groups):
    """Multilevel random-effects meta-regression: y = Xb + sum_k Z_k u_k + e, Var(e) = v + tau_res^2, by REML."""
    from scipy.optimize import minimize
    Zs = [np.eye(int(g.max()) + 1)[g] for g in groups]

    def V(theta):
        t = np.exp(theta)
        M = np.diag(v + t[-1])
        for tk, Z in zip(t[:-1], Zs):
            M = M + tk * Z @ Z.T
        return M

    def nll(theta):
        M = V(theta); Mi = np.linalg.inv(M); A = X.T @ Mi @ X
        b = np.linalg.solve(A, X.T @ Mi @ y); r = y - X @ b
        return 0.5 * (np.linalg.slogdet(M)[1] + np.linalg.slogdet(A)[1] + r @ Mi @ r)
    start = np.full(len(groups) + 1, math.log(max(np.var(y), 1e-6) / (len(groups) + 1)))
    res = minimize(nll, start, method="Nelder-Mead", options={"maxiter": 20000, "xatol": 1e-7, "fatol": 1e-10})
    M = V(res.x); Mi = np.linalg.inv(M); A = X.T @ Mi @ X
    b = np.linalg.solve(A, X.T @ Mi @ y)
    return b, np.linalg.inv(A), np.exp(res.x)


def meta6(rev, S, PRIMARY):
    y, v = [], []
    for s in S:
        a = per_question(s["rows"], "plain", True, PRIMARY); f = per_question(s["rows"], "oracle+flag", True, PRIMARY)
        d = np.array([f[i] - a[i] for i in sorted(set(a) & set(f))])
        y.append(d.mean()); v.append(d.var(ddof=1) / len(d))
    y, v = np.array(y), np.array(v)
    lv = lambda key: np.unique([s[key] for s in S], return_inverse=True)[1]
    groups = [lv("gen"), lv("bench")]
    b0, c0, t0 = _reml(y, v, np.ones((len(y), 1)), groups)
    tau_sum = float(t0.sum())
    half = 1.96 * math.sqrt(c0[0, 0] + tau_sum)
    out = {"k": len(y), "mu": float(b0[0]), "ci": [float(b0[0] - 1.96 * math.sqrt(c0[0, 0])), float(b0[0] + 1.96 * math.sqrt(c0[0, 0]))],
           "pred_interval": [float(b0[0] - half), float(b0[0] + half)],
           "tau": {"generator": float(math.sqrt(t0[0])), "benchmark": float(math.sqrt(t0[1])), "setting": float(math.sqrt(t0[2]))}}
    # moderators: answer policy (reference: one answer with "unknown"), family (reference OpenAI), natural misinformation
    cols = {"several answers allowed": [s["policy"] in ("multi", "minimal") for s in S],
            "forced (no unknown)": [s["policy"] == "forced" for s in S],
            "Anthropic": [FAMILY.get(s["gen"], "OpenAI") == "Anthropic" for s in S],
            "Qwen": [FAMILY.get(s["gen"], "OpenAI") == "Qwen" for s in S],
            "natural misinformation (RAGuard)": [s["bench"] == "RAGuard" for s in S]}
    X = np.c_[np.ones(len(y)), np.array(list(cols.values()), float).T]
    b, c, t = _reml(y, v, X, groups)
    se = np.sqrt(np.diag(c))
    names = ["intercept (OpenAI, one answer, generated)"] + list(cols)
    out["moderators"] = {n: {"est": float(b[j]), "ci_low": float(b[j] - 1.96 * se[j]), "ci_high": float(b[j] + 1.96 * se[j])}
                         for j, n in enumerate(names)}
    # pooled mean per answer policy (intercept-only within the policy)
    out["by_policy"] = {}
    for pol in ("multi", "minimal", "single", "forced"):
        idx = [k for k, s in enumerate(S) if s["policy"] == pol]
        if len(idx) >= 3:
            gg = [np.unique(g[idx], return_inverse=True)[1] for g in groups]
            bp, cp, tp = _reml(y[idx], v[idx], np.ones((len(idx), 1)), gg)
            out["by_policy"][pol] = {"k": len(idx), "mu": float(bp[0]),
                                     "ci_low": float(bp[0] - 1.96 * math.sqrt(cp[0, 0])),
                                     "ci_high": float(bp[0] + 1.96 * math.sqrt(cp[0, 0]))}
        elif idx:
            out["by_policy"][pol] = {"k": len(idx), "mu": float(np.mean(y[idx])), "ci_low": float("nan"), "ci_high": float("nan")}
    out["settings"] = [{"name": s["name"], "policy": s["policy"], "bench": s["bench"], "diff": float(yy)} for s, yy in zip(S, y)]
    rev["neutral_pooled6"] = out


def utility6(rev, S, PRIMARY):
    """Utility and transitions over the same settings as the pooled estimate; plain-to-plain seed transitions as the null."""
    ps = _ps()
    cats = ("correct", "misinfo", "abstain", "other")
    out, trans, null = {}, {}, {}
    for s in S:
        rows = s["rows"]
        for r in rows:
            mis = float(r["wrong_rate"] > 0)
            for c in UTIL_C:
                r[f"util{c}"] = r[PRIMARY] - c * mis
        have = {r["condition"] for r in rows}
        by = {(r["id"], r["seed"], r["condition"]): _outcome(r, PRIMARY) for r in rows
              if r.get("has_conflict") and r["condition"] in ("plain", "oracle+flag", "oracle+flag_directional")}
        for cond, nm in (("oracle+flag", "neutral"), ("oracle+flag_directional", "directional")):
            if cond not in have:
                continue
            out.setdefault(s["name"], {})[nm] = {f"c{c}": ps.compare(rows, "plain", cond, True, f"util{c}") for c in UTIL_C}
            T = {a: dict.fromkeys(cats, 0) for a in cats}
            for (i, sd, c), o in by.items():
                if c == "plain" and (i, sd, cond) in by:
                    T[o][by[(i, sd, cond)]] += 1
            trans.setdefault(s["name"], {})[nm] = T
        seeds = sorted({sd for (_, sd, c) in by if c == "plain"})
        if len(seeds) > 1:
            T = {a: dict.fromkeys(cats, 0) for a in cats}
            for s0, s1 in itertools.permutations(seeds, 2):
                for (i, sd, c), o in by.items():
                    if c == "plain" and sd == s0 and (i, s1, "plain") in by:
                        T[o][by[(i, s1, "plain")]] += 1
            null[s["name"]] = T
    for c in UTIL_C:
        ps_ = {f"{k}|{nm}": t[f"c{c}"]["p"] for k, d in out.items() for nm, t in d.items() if not math.isnan(t[f"c{c}"]["p"])}
        adj = holm(ps_)
        for key, p in ps_.items():
            k, nm = key.split("|", 1) if key.count("|") == 1 else key.rsplit("|", 1)
            t = out[k][nm][f"c{c}"]; t["p_holm"] = adj[key]; t["sig"] = bool(adj[key] < 0.05)
    rev["utility6"], rev["transitions6"], rev["transitions_null"] = out, trans, null


# --------------------------------------------------------------------------- H24 recast (§16.4.3)
def _cell_outcomes(rows, cond, metric):
    acc = defaultdict(lambda: np.zeros(4))
    for r in rows:
        if r["condition"] != cond or not r.get("has_conflict"):
            continue
        o = _outcome(r, metric)
        acc[r["id"]] += np.array([o == "correct", r["wrong_rate"] > 0, o == "abstain", 1.0])
    return {i: v[:3] / v[3] for i, v in acc.items()}


def h24_recast(rev, PRIMARY):
    from scipy.stats import spearmanr  # noqa: F401
    q = rev.get("H24", {}).get("_q", {})
    if len(q) < 4:
        return
    cells = sorted(q)
    qids = sorted({i for c in cells for i in q[c]["ids"]}); pos = {i: k for k, i in enumerate(qids)}
    nC, nQ = len(cells), len(qids)
    M = {v: np.full((nC, nQ), np.nan) for v in ("d", "r", "g", "y1", "w")}
    for ci, c in enumerate(cells):
        for v in M:
            M[v][ci, [pos[i] for i in q[c]["ids"]]] = q[c][v]
    gens = sorted({g for g, _ in cells}); dets = sorted({d for _, d in cells})
    gi = np.array([gens.index(g) for g, _ in cells]); di = np.array([dets.index(d) for _, d in cells])
    Gfe = np.eye(len(gens))[gi]

    def stats(idx):
        D, Rr, G, A, W = (np.nanmean(M[v][:, idx], 1) for v in ("d", "r", "g", "y1", "w"))
        return D, Rr, np.where(Rr > 0, D / np.maximum(Rr, 1e-9), np.nan), G, A, W

    def fit(idx):
        D, Rr, P, G, A, W = stats(idx)
        return _ols(np.c_[P, Rr, Gfe], A)[:2]
    full = np.arange(nQ)
    est = fit(full)
    rng = np.random.default_rng(0)
    boots = np.array([fit(rng.integers(0, nQ, nQ)) for _ in range(N_BOOT6)])
    lo, hi = np.quantile(boots, [0.025, 0.975], axis=0)
    D, Rr, P, G, A, W = stats(full)
    out = {"P_R_fe": {n: {"est": float(est[k]), "ci_low": float(lo[k]), "ci_high": float(hi[k])} for k, n in enumerate(("P", "R"))}}
    for nm, X in (("P", P), ("R", Rr)):
        bar = np.array([X[di == k].mean() for k in range(len(dets))])
        other = Rr if nm == "P" else P
        stat = lambda b: _ols(np.c_[b[di], other, Gfe], A)[0]
        obs = stat(bar)
        perm = np.array([stat(bar[list(p)]) for p in itertools.permutations(range(len(dets)))])
        out[f"perm_{nm}"] = {"est": float(obs), "p_one_sided": float(np.mean(perm >= obs - 1e-12)), "n_perm": len(perm),
                             "by_detector": dict(zip(dets, map(float, bar)))}
    gain = A - G
    cv = {}
    for nm, X in (("G only", np.c_[np.ones(nC), G]), ("P + R + G", np.c_[np.ones(nC), P, Rr, G]),
                  ("P + G", np.c_[np.ones(nC), P, G])):
        pred = np.empty(nC)
        for k in range(len(dets)):
            te = di == k
            pred[te] = X[te] @ _ols(X[~te], gain[~te])
        cv[nm] = 1 - float(((gain - pred) ** 2).sum()) / float(((gain - gain.mean()) ** 2).sum())
    out["lodo_r2"] = cv
    out["cells"] = {f"{g}|{d}": {"P": float(P[k]), "R": float(Rr[k]), "D": float(D[k])} for k, (g, d) in enumerate(cells)}
    # question, detector and generator fixed effects; question bootstrap and leave-one-detector-out range
    rec = [(ci, qi) for ci in range(nC) for qi in range(nQ) if not np.isnan(M["y1"][ci, qi])]
    ci_, qi_ = np.array([a for a, _ in rec]), np.array([b for _, b in rec])
    y = M["y1"][ci_, qi_]; dd = M["d"][ci_, qi_]; ww = M["w"][ci_, qi_]

    def qfe(mask=None, qmap=None):
        sel = np.ones(len(y), bool) if mask is None else mask
        qq = qi_ if qmap is None else qmap
        Xq = np.eye(nQ)[qq[sel]]
        X = np.c_[dd[sel], ww[sel], Xq, np.eye(len(dets))[di[ci_[sel]]][:, 1:], np.eye(len(gens))[gi[ci_[sel]]][:, 1:]]
        return _ols(X, y[sel])[:2]
    b_full = qfe()
    bq = []
    rng = np.random.default_rng(1)
    rows_by_q = defaultdict(list)
    for n, qq in enumerate(qi_):
        rows_by_q[qq].append(n)
    for _ in range(300):
        samp = rng.integers(0, nQ, nQ)
        idx = np.concatenate([rows_by_q[s] for s in samp]); newq = np.concatenate([[k] * len(rows_by_q[s]) for k, s in enumerate(samp)])
        Xq = np.eye(nQ)[newq]
        X = np.c_[dd[idx], ww[idx], Xq, np.eye(len(dets))[di[ci_[idx]]][:, 1:], np.eye(len(gens))[gi[ci_[idx]]][:, 1:]]
        bq.append(_ols(X, y[idx])[:2])
    bq = np.array(bq); lo, hi = np.quantile(bq, [0.025, 0.975], axis=0)
    lodo = np.array([qfe(di[ci_] != k) for k in range(len(dets))])
    out["qfe"] = {n: {"est": float(b_full[j]), "ci_low": float(lo[j]), "ci_high": float(hi[j]),
                      "lodo_min": float(lodo[:, j].min()), "lodo_max": float(lodo[:, j].max())}
                  for j, n in enumerate(("d", "w"))}
    # following, one definition (§16.3): right flag followed = flagged answer correct; wrong flag followed = flagged
    # answer contains the misinformation; abstentions separately. Null: plain seed 2 given plain seeds 0 and 1.
    follow = {}
    for g in gens:
        fr, fw, fa = [], [], []
        null_r, null_w = [], []
        for (gg, d), v in q.items():
            if gg != g:
                continue
            oc = _cell_outcomes(v["rows"], "llm_dir+flag_directional", PRIMARY)
            for i, dq, gq, wq in zip(v["ids"], v["d"], v["g"], v["w"]):
                if i not in oc:
                    continue
                if dq == 1 and gq == 0:
                    fr.append(oc[i][0])
                if wq == 1 and gq == 1:
                    fw.append(oc[i][1]); fa.append(oc[i][2])
        # regression to the mean, from the generator's plain rows (3-seed generators only)
        rows = next(iter(v["rows"] for (gg, _), v in q.items() if gg == g))
        ps_ = defaultdict(dict)
        for r in rows:
            if r["condition"] == "plain" and r.get("has_conflict"):
                ps_[r["id"]][r["seed"]] = r
        for i, sd in ps_.items():
            if {0, 1, 2} <= set(sd):
                c01 = [sd[0][PRIMARY], sd[1][PRIMARY]]
                if c01 == [0, 0]:
                    null_r.append(sd[2][PRIMARY])
                if c01 == [1, 1]:
                    null_w.append(float(sd[2]["wrong_rate"] > 0))
        follow[g] = {"follow_right": float(np.mean(fr)) if fr else None, "n_right": len(fr),
                     "follow_wrong": float(np.mean(fw)) if fw else None, "abstain_wrong": float(np.mean(fa)) if fa else None,
                     "n_wrong": len(fw), "null_right": float(np.mean(null_r)) if null_r else None,
                     "null_wrong": float(np.mean(null_w)) if null_w else None}
    out["follow"] = follow
    rev["H24_recast"] = out


# --------------------------------------------------------------------------- H30, H31, H14 matched
def h30(rev, PRIMARY, ROBUST):
    from . import revision_stats as rv
    ps = _ps()
    q = rev.get("H24", {}).get("_q", {})
    fa, fb, tost = {m: {} for m in (PRIMARY, ROBUST)}, {m: {} for m in (PRIMARY, ROBUST)}, {}
    for (g, d), v in sorted(q.items()):
        run = ps.ROOT / f"runs/vkey_cb_{SHORT_GEN[g]}_{SHORT_GEN[d]}"
        hrun = ps.ROOT / f"runs/hint_cb_{SHORT_GEN[g]}_{SHORT_GEN[d]}"
        if not results_path(run):
            continue
        rows = [r for r in v["rows"] if r["condition"] in ("plain", "llm_dir+flag_directional")]
        rows += [r for r in ps.rows_of(run) if r["condition"] == "llm_dir+verdict_key"]
        if results_path(hrun):
            rows += [r for r in ps.rows_of(hrun) if r["condition"] == "hint"]
        tk = f"{g}, detector {d}"
        for m in (PRIMARY, ROBUST):
            fa[m][f"{tk}: LLM dir. flag vs verdict key (conflict)"] = ps.compare(rows, "llm_dir+verdict_key", "llm_dir+flag_directional", True, m)
            fb[m][f"{tk}: verdict key vs hint (conflict)"] = ps.compare(rows, "hint", "llm_dir+verdict_key", True, m)
        tost[tk] = rv.tost(rows, "llm_dir+verdict_key", "llm_dir+flag_directional", True, PRIMARY, ps.groups_of(rows), margin=0.03)
        fa[PRIMARY][f"{tk}: LLM dir. flag vs verdict key (conflict)"]["verdict"] = float(np.mean(list(
            per_question(rows, "llm_dir+verdict_key", True, PRIMARY).values())))
    if fa[PRIMARY]:
        rev["H30a"] = {m: ps.family(fa[m], {}) for m in (PRIMARY, ROBUST)}
        rev["H30b"] = {m: ps.family(fb[m], {}) for m in (PRIMARY, ROBUST)}
        rev["H30a_tost"] = tost


def h31(rev, PRIMARY):
    ps = _ps()
    res = {"a": {}, "b": {}, "c": {}}
    follow = {}
    for g, run in (("gpt-4o-mini", "randdir_cb_mini"), ("gpt-6-luna", "randdir_cb_luna"),
                   ("claude-haiku-5-5", "randdir_cb_haiku"), ("gpt-6.1-terra", "randdir_cb_terra")):
        p = ps.ROOT / "runs" / run
        if not results_path(p):
            continue
        rows = ps.rows_of(p)
        if RAND not in {r["condition"] for r in rows}:
            continue
        ex = {e.id: e for e in ps.examples_of(p) if e.has_conflict}
        det = ps.dets_of(p, RAND.split("+")[0])
        right = {i: all(ex[i].meta["doc_types"][k] == "misinfo" for x in det[i].pairs for k in x.suspects) for i in ex}
        pl = per_question(rows, "plain", True, PRIMARY); fl = per_question(rows, RAND, True, PRIMARY)
        ids = sorted(set(pl) & set(fl) & set(right))
        R_ = [i for i in ids if right[i]]; W_ = [i for i in ids if not right[i]]
        gR = np.array([fl[i] - pl[i] for i in R_]); gW = np.array([fl[i] - pl[i] for i in W_])
        for key, sub in (("a", R_), ("b", W_)):
            aa = {i: pl[i] for i in sub}; bb = {i: fl[i] for i in sub}
            res[key][f"{g}: {'right' if key == 'a' else 'wrong'}-flag effect (conflict)"] = {
                **paired_bootstrap(aa, bb, n_boot=10_000), "p": sign_flip_p(aa, bb, n_perm=10_000, max_perm=200_000), "n": len(sub)}
        x = np.r_[gR, -gW]; lab = np.r_[np.ones(len(gR)), np.zeros(len(gW))].astype(bool)
        obs = gR.mean() + gW.mean()   # = mean(x[lab]) - mean(x[~lab])
        rng = np.random.default_rng(0)
        perm = []
        for _ in range(10_000):
            lp = rng.permutation(lab)
            perm.append(x[lp].mean() - x[~lp].mean())
        boots = [rng.choice(gR, len(gR)).mean() + rng.choice(gW, len(gW)).mean() for _ in range(10_000)]
        res["c"][f"{g}: right-flag effect + wrong-flag effect (conflict)"] = {
            "diff": float(obs), "ci_low": float(np.quantile(boots, 0.025)), "ci_high": float(np.quantile(boots, 0.975)),
            "p": float((np.sum(np.abs(perm) >= abs(obs) - 1e-12) + 1) / (len(perm) + 1)), "n": len(ids)}
        # following with one definition, conditioned on the plain outcome over all seeds
        oc = _cell_outcomes(rows, RAND, PRIMARY)
        fr = [oc[i][0] for i in R_ if pl[i] == 0 and i in oc]
        fw = [oc[i][1] for i in W_ if pl[i] == 1 and i in oc]
        fa = [oc[i][2] for i in W_ if pl[i] == 1 and i in oc]
        follow[g] = {"follow_right": float(np.mean(fr)) if fr else None, "n_right": len(fr),
                     "follow_wrong": float(np.mean(fw)) if fw else None, "abstain_wrong": float(np.mean(fa)) if fa else None,
                     "n_wrong": len(fw), "n_right_q": len(R_), "n_wrong_q": len(W_)}
    if res["a"]:
        rev["H31a"] = ps.family(res["a"], {k: +1 for k in res["a"]})
        rev["H31b"] = ps.family(res["b"], {k: -1 for k in res["b"]})
        rev["H31c"] = ps.family(res["c"], {})
        rev["H31_follow"] = follow


def h14_matched(rev, r2, PRIMARY):
    ps = _ps()
    if "CBs:gpt-4o-mini" not in r2:
        return
    rows = r2["CBs:gpt-4o-mini"]["_rows"]
    have = {r["condition"] for r in rows}
    out = {}
    for b in ("astute", "trustrag"):
        if b in have:
            out[f"ConflictBank (one-answer prompt), gpt-4o-mini: LLM dir. flag vs {b} (all q.)"] = \
                ps.compare(rows, b, "llm_dir+flag_directional", ps.LABELLED, PRIMARY)
    rev["H14_matched"] = out


def revision6(R, r2, main_rows):
    ps = _ps()
    PRIMARY, ROBUST = ps.PRIMARY, ps.ROBUST
    rev = R["rev"]
    S = settings(rev, r2, main_rows)
    meta6(rev, S, PRIMARY)
    utility6(rev, S, PRIMARY)
    h24_recast(rev, PRIMARY)
    h30(rev, PRIMARY, ROBUST)
    h31(rev, PRIMARY)
    h14_matched(rev, r2, PRIMARY)
    for k in ("H24", "H24_s13", "H24_maj"):
        if k in rev:
            for v in rev[k].get("_q", {}).values():
                v.pop("run", None); v.pop("rows", None)
            rev[k].pop("_q", None)
