"""
Eighth revision (PREREGISTRATION.md §17.1): re-analyses of existing data asked for by the eighth mock review. Every
result here uses data already seen and is reported as exploratory.

- meta8: the §16.4.2 multilevel model refitted inside a joint question bootstrap (settings share questions).
- realised31: H31's net flag effect at the realised right/wrong split, beside the registered 50/50 H31c.
- selection8: each crossed cell split into questions whose flag names the false side, marks a correct passage, or
  neither, with coverage, baseline accuracy and gain; the marginal break-even prediction against the observed gain.
- predict8: the follow-rate prediction against simple baselines, with calibration uncertainty propagated and a
  cross-fitted no-flag baseline.
- h23_het: per-comparison equivalence and between-comparison heterogeneity for H23.
- prevalence8: joint-bootstrap uncertainty for the prevalence-reweighted effects and the count of settings below 0.

Called from revision6 after revision7, before H24's per-question data is dropped.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict

import numpy as np

from .analyze import per_question
from .revision5 import FAMILY
from .stats import paired_bootstrap, sign_flip_p

N_BOOT8 = 2000
TOST8 = 3.0


def _ps():
    from . import paper_stats as ps
    return ps


def _pool_of(qid):
    return qid.split("-")[0]   # cb (shared by the majority and forced variants), ramdocs, rg


def _vectors(diffs):
    """diffs: list of {qid: value}. Returns the question list of each pool and, per diff, (pool, array over it)."""
    qs = defaultdict(set)
    for d in diffs:
        for i in d:
            qs[_pool_of(i)].add(i)
    qs = {p: sorted(v) for p, v in qs.items()}
    pos = {p: {q: k for k, q in enumerate(v)} for p, v in qs.items()}
    out = []
    for d in diffs:
        p = _pool_of(next(iter(d)))
        v = np.full(len(qs[p]), np.nan)
        v[[pos[p][q] for q in d]] = list(d.values())
        out.append((p, v))
    return qs, out


def _draw(rng, qs):
    return {p: rng.integers(0, len(v), len(v)) for p, v in qs.items()}


def _mv(vec, idx):
    y, v = [], []
    for p, a in vec:
        x = a[idx[p]]; x = x[~np.isnan(x)]
        var = x.var(ddof=1) / len(x) if len(x) > 1 else 0.0
        y.append(x.mean()); v.append(max(var, 1e-6))   # floor: a draw can have no variance
    return np.array(y), np.array(v)


# --------------------------------------------------------------------------- pooled neutral flag, joint bootstrap
def _reml_cov(y, V0, X, groups):
    """As revision6._reml, but with a full sampling covariance V0 between settings (they share questions)."""
    from scipy.optimize import minimize
    Zs = [np.eye(int(g.max()) + 1)[g] for g in groups]

    def V(theta):
        t = np.exp(theta)
        M = V0 + np.eye(len(y)) * t[-1]
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
    C = np.linalg.inv(A)
    return C @ X.T @ Mi @ y, C, np.exp(res.x), C @ X.T @ Mi   # last: the GLS weights


def meta8(rev, S, PRIMARY):
    """Each replicate resamples each pool's questions once and recomputes every setting's mean; their covariance is
    the settings' sampling covariance. (a) The multilevel model refitted with that covariance in place of the
    independent diagonal: CIs and prediction interval for generalising across settings like these. (b) The same GLS
    weights applied to each replicate: a CI conditional on these settings (question sampling only)."""
    diffs = []
    for s in S:
        a = per_question(s["rows"], "plain", True, PRIMARY); f = per_question(s["rows"], "oracle+flag", True, PRIMARY)
        diffs.append({i: f[i] - a[i] for i in set(a) & set(f)})
    qs, vec = _vectors(diffs)
    lv = lambda key: np.unique([s[key] for s in S], return_inverse=True)[1]
    groups = [lv("gen"), lv("bench")]
    cols = {"several answers allowed": [s["policy"] in ("multi", "minimal") for s in S],
            "forced (no unknown)": [s["policy"] == "forced" for s in S],
            "Anthropic": [FAMILY.get(s["gen"], "OpenAI") == "Anthropic" for s in S],
            "Qwen": [FAMILY.get(s["gen"], "OpenAI") == "Qwen" for s in S],
            "natural misinformation (RAGuard)": [s["bench"] == "RAGuard" for s in S]}
    names = ["intercept (OpenAI, one answer, generated)"] + list(cols)
    X = np.c_[np.ones(len(S)), np.array(list(cols.values()), float).T]
    one = np.ones((len(S), 1))
    y = _mv(vec, {p: np.arange(len(v)) for p, v in qs.items()})[0]
    rng = np.random.default_rng(0)
    Yb = np.array([_mv(vec, _draw(rng, qs))[0] for _ in range(N_BOOT8)])
    V0 = np.cov(Yb.T)
    q = lambda x: [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))]
    ci = lambda b, C, j=0: [float(b[j] - 1.96 * math.sqrt(C[j, j])), float(b[j] + 1.96 * math.sqrt(C[j, j]))]
    b0, c0, t0, w0 = _reml_cov(y, V0, one, groups)
    half = 1.96 * math.sqrt(c0[0, 0] + t0.sum())
    bm, cm, tm, wm = _reml_cov(y, V0, X, groups)
    design = Yb @ w0[0]
    out = {"k": len(S), "pools": {p: len(v) for p, v in qs.items()}, "n_boot": N_BOOT8, "mu": float(b0[0]), "ci": ci(b0, c0),
           "pred_interval": [float(b0[0] - half), float(b0[0] + half)], "ci_design": q(design),
           "tau": {"generator": float(math.sqrt(t0[0])), "benchmark": float(math.sqrt(t0[1])), "setting": float(math.sqrt(t0[2]))},
           "mean_corr_same_pool": float(np.mean([np.corrcoef(Yb.T)[i, j] for i in range(len(S)) for j in range(len(S))
                                                  if i != j and vec[i][0] == vec[j][0]])),
           "equal_weight": float(y.mean()), "equal_weight_ci": q(Yb.mean(1)),
           "moderators": {n: {"est": float(bm[j]), "ci_low": ci(bm, cm, j)[0], "ci_high": ci(bm, cm, j)[1],
                              "ci_design": q(Yb @ wm[j])} for j, n in enumerate(names)},
           "by_policy": {}}
    for pol in ("multi", "minimal", "single", "forced"):
        ix = np.array([k for k, s in enumerate(S) if s["policy"] == pol])
        if len(ix) >= 3:
            gg = [np.unique(g[ix], return_inverse=True)[1] for g in groups]
            bp, cp, tp, wp = _reml_cov(y[ix], V0[np.ix_(ix, ix)], np.ones((len(ix), 1)), gg)
            out["by_policy"][pol] = {"k": len(ix), "mu": float(bp[0]), "ci_low": ci(bp, cp)[0], "ci_high": ci(bp, cp)[1],
                                     "ci_design": q(Yb[:, ix] @ wp[0])}
    rev["neutral_pooled8"] = out
    # these intervals replace the model-based ones downstream; the model-based CI is kept for comparison
    m6 = rev.get("neutral_pooled6")
    if m6:
        m6.update(ci_model=m6["ci"], pred_interval_model=m6["pred_interval"], mu_model=m6["mu"], mu=out["mu"], ci=out["ci"],
                  pred_interval=out["pred_interval"], tau=out["tau"], ci_method="REML with the joint-bootstrap sampling covariance (§17.1)")
        for n, t in out["moderators"].items():
            m6["moderators"][n].update(est=t["est"], ci_low=t["ci_low"], ci_high=t["ci_high"])
        for pol, t in out["by_policy"].items():
            m6["by_policy"][pol].update(mu=t["mu"], ci_low=t["ci_low"], ci_high=t["ci_high"])


# --------------------------------------------------------------------------- H31 at the realised split
def realised31(rev, arms):
    ps = _ps()
    res = {}
    for g, a in arms.items():
        ids = a["R"] + a["W"]
        aa = {i: a["pl"][i] for i in ids}; bb = {i: a["fl"][i] for i in ids}
        res[g] = {**paired_bootstrap(aa, bb, n_boot=10_000), "p": sign_flip_p(aa, bb, n_perm=10_000),
                  "n": len(ids), "share_right": len(a["R"]) / len(ids)}
    rev["realised31"] = ps.family(res, {})


# --------------------------------------------------------------------------- selection-aware decomposition
def selection8(rev):
    q = rev.get("H24", {}).get("_q", {})
    h = rev.get("H31x", {})
    out = {}
    for (g, d), v in sorted(q.items()):
        D, W, G, Y, Rr = (np.array(v[x], float) for x in ("d", "w", "g", "y1", "r"))
        right = D > 0; wrong = (W > 0) & ~right; rest = ~right & ~wrong
        st = {}
        for nm, m in (("right", right), ("wrong", wrong), ("rest", rest)):
            st[nm] = {"share": float(m.mean()), "base_acc": float(G[m].mean()) if m.any() else None,
                      "gain": float((Y - G)[m].mean()) if m.any() else None,
                      "contrib": float(((Y - G) * m).mean())}
        prec = float(D.sum() / Rr.sum()) if Rr.sum() else None
        o = {"strata": st, "obs": float((Y - G).mean()), "precision": prec, "recall": float(Rr.mean()),
             "base_acc": float(G.mean())}
        if g in h and prec is not None:
            R_, W_ = h[g]["right"] / 100, h[g]["wrong"] / 100
            o["marginal_net"] = prec * R_ + (1 - prec) * W_            # every question flagged on its true pair
            o["marginal_net_cov"] = Rr.mean() * o["marginal_net"]      # scaled by the cell's recall
            o["breakeven"] = h[g]["breakeven"]
        out[f"{g}|{d}"] = o
    rev["selection8"] = out


# --------------------------------------------------------------------------- prediction: baselines and uncertainty
def _stats(P, O):
    P, O = np.asarray(P, float), np.asarray(O, float)
    r = float(np.corrcoef(P, O)[0, 1]) if np.std(P) > 0 else float("nan")
    return {"r": r, "mae": float(np.mean(np.abs(P - O))), "sign_agree": int(np.sum(np.sign(P) == np.sign(O))),
            "n": len(P)}


def predict8(rev, arms, PRIMARY):
    q = rev.get("H24", {}).get("_q", {}); p7 = rev.get("predict7")
    f = {g: v for g, v in rev.get("H31x", {}).items() if v["follow_right"] is not None}
    if not p7 or not f:
        return
    cells = sorted(q)
    own = [c for c in cells if c[0] in f]
    obs = {c: float(np.mean(np.array(q[c]["y1"]) - np.array(q[c]["g"]))) for c in cells}
    pred = {(r["gen"], r["det"]): r["pred"] / 100 for r in p7["cells"]}
    out = {"follow_rate": _stats([pred[c] for c in own], [obs[c] for c in own])}
    out["zero"] = {"mae": float(np.mean([abs(obs[c]) for c in own])), "n": len(own)}
    marg = {}
    for c in own:
        D, G, W = (np.array(q[c][x], float) for x in ("d", "g", "w"))
        marg[c] = f[c[0]]["follow_right"] * D.mean() * (1 - G.mean()) - f[c[0]]["follow_wrong"] * W.mean() * G.mean()
    out["marginal"] = _stats([marg[c] for c in own], [obs[c] for c in own])
    loo = {c: np.mean([obs[o] for o in own if o[0] == c[0] and o != c]) for c in own}
    out["generator_mean"] = _stats([loo[c] for c in own], [obs[c] for c in own])
    # joint bootstrap: H31 questions (follow rates) and the cells' shared ConflictBank questions
    fl = {}
    for g, a in arms.items():
        fl[g] = (np.array([a["oc"][i][0] for i in a["R"] if a["pl"][i] == 0 and i in a["oc"]]),
                 np.array([a["oc"][i][1] for i in a["W"] if a["pl"][i] == 1 and i in a["oc"]]))
    qids = sorted({i for c in own for i in q[c]["ids"]}); pos = {i: k for k, i in enumerate(qids)}
    M = {}
    for c in own:
        ix = np.array([pos[i] for i in q[c]["ids"]])
        M[c] = {x: np.full(len(qids), np.nan) for x in ("d", "g", "w", "y1")}
        for x in M[c]:
            M[c][x][ix] = q[c][x]
    rng = np.random.default_rng(0)
    rs, maes = [], []
    for _ in range(N_BOOT8):
        fb = {g: (rng.choice(a, len(a)).mean(), rng.choice(b, len(b)).mean()) for g, (a, b) in fl.items()}
        idx = rng.integers(0, len(qids), len(qids))
        P, O = [], []
        for c in own:
            D, G, W, Y = (M[c][x][idx] for x in ("d", "g", "w", "y1"))
            ok = ~np.isnan(Y)
            P.append(np.mean(D[ok] * (1 - G[ok]) * fb[c[0]][0] - W[ok] * G[ok] * fb[c[0]][1]))
            O.append(np.mean(Y[ok] - G[ok]))
        rs.append(np.corrcoef(P, O)[0, 1]); maes.append(np.mean(np.abs(np.array(P) - np.array(O))))
    out["follow_rate"]["r_ci"] = [float(np.nanquantile(rs, 0.025)), float(np.nanquantile(rs, 0.975))]
    out["follow_rate"]["mae_ci"] = [float(np.quantile(maes, 0.025)), float(np.quantile(maes, 0.975))]
    # cross-fitted: prediction from seed-0 plain answers, observed gain on seeds 1-2
    P, O = [], []
    for c in own:
        rows = q[c]["rows"]
        if {r["seed"] for r in rows if r["condition"] == "plain"} < {0, 1, 2}:
            continue
        by = defaultdict(lambda: defaultdict(list))
        for r in rows:
            if r.get("has_conflict") and r["condition"] in ("plain", "llm_dir+flag_directional") and r.get(PRIMARY) is not None:
                by[(r["condition"], r["seed"] == 0)][r["id"]].append(r[PRIMARY])
        ids = q[c]["ids"]; D = np.array(q[c]["d"], float); W = np.array(q[c]["w"], float)
        try:
            G0 = np.array([np.mean(by[("plain", True)][i]) for i in ids])
            G12 = np.array([np.mean(by[("plain", False)][i]) for i in ids])
            Y12 = np.array([np.mean(by[("llm_dir+flag_directional", False)][i]) for i in ids])
        except Exception:
            continue
        fr, fw = f[c[0]]["follow_right"], f[c[0]]["follow_wrong"]
        P.append(np.mean(D * (1 - G0) * fr - W * G0 * fw)); O.append(np.mean(Y12 - G12))
    if len(P) >= 3:
        out["cross_fit"] = _stats(P, O)
    rev["predict8"] = out


# --------------------------------------------------------------------------- H23 heterogeneity
def h23_het(rev, r2, PRIMARY):
    from .revision6 import _reml
    from .revision7 import _diff
    cells = {k: _diff(v["_rows"], "answer_key", "oracle+flag_directional", PRIMARY) for k, v in r2.items()
             if k.startswith(("RD:", "CBs:", "CBmaj:")) and "answer_key" in {r["condition"] for r in v["_rows"]}}
    if len(cells) < 3:
        return
    rng = np.random.default_rng(0)
    per = {}
    for k, d in cells.items():
        x = np.array(list(d.values()))
        b = np.array([rng.choice(x, len(x)).mean() for _ in range(5000)])
        lo, hi = np.quantile(b, [0.05, 0.95])
        per[k] = {"diff": 100 * x.mean(), "ci90": [100 * lo, 100 * hi], "equivalent": bool(-TOST8 < 100 * lo and 100 * hi < TOST8),
                  "n": len(x)}
    y = np.array([np.mean(list(d.values())) for d in cells.values()])
    v = np.array([np.var(list(d.values()), ddof=1) / len(d) for d in cells.values()])
    b, c, t = _reml(y, v, np.ones((len(y), 1)), [])
    Q = float(np.sum((y - np.sum(y / v) / np.sum(1 / v)) ** 2 / v))
    rev["h23_het"] = {"cells": per, "n_equivalent": int(sum(p["equivalent"] for p in per.values())), "k": len(per),
                      "tau": 100 * float(math.sqrt(t[-1])), "Q": Q, "I2": max(0.0, (Q - (len(y) - 1)) / Q) if Q > 0 else 0.0,
                      "range": [float(min(p["diff"] for p in per.values())), float(max(p["diff"] for p in per.values()))]}


# --------------------------------------------------------------------------- prevalence with uncertainty
def prevalence8(rev, S, PRIMARY, prev=0.05):
    from .revision7 import _diff
    keys, dc, dn = [], [], []
    for s in S:
        rows = s["rows"]
        if "llm_dir+flag_directional" not in {r["condition"] for r in rows} or not any(r.get("has_conflict") is False for r in rows):
            continue
        c = _diff(rows, "plain", "llm_dir+flag_directional", PRIMARY)
        ra = per_question(rows, "plain", False, PRIMARY); rb = per_question(rows, "llm_dir+flag_directional", False, PRIMARY)
        n = {i: rb[i] - ra[i] for i in set(ra) & set(rb)}
        if c and n:
            keys.append(s["name"]); dc.append(c); dn.append(n)
    if not keys:
        return
    qc, vc = _vectors(dc); qn, vn = _vectors(dn)

    def est(ic, inn):
        C = np.array([np.nanmean(a[ic[p]]) for p, a in vc]); N = np.array([np.nanmean(a[inn[p]]) for p, a in vn])
        return prev * C + (1 - prev) * N
    full = est({p: np.arange(len(v)) for p, v in qc.items()}, {p: np.arange(len(v)) for p, v in qn.items()})
    rng = np.random.default_rng(0)
    B = np.array([est(_draw(rng, qc), _draw(rng, qn)) for _ in range(N_BOOT8)])
    lo, hi = np.quantile(B, [0.025, 0.975], axis=0)
    neg = (B < 0).sum(1)
    rev["prevalence8"] = {"prev": prev, "n": len(keys), "n_neg": int((full < 0).sum()),
                          "n_neg_ci": [int(np.quantile(neg, 0.025)), int(np.quantile(neg, 0.975))],
                          "n_sig_neg": int((hi < 0).sum()), "n_sig_pos": int((lo > 0).sum()),
                          "settings": {k: {"est": 100 * float(e), "ci_low": 100 * float(a), "ci_high": 100 * float(b)}
                                       for k, e, a, b in zip(keys, full, lo, hi)}}


# --------------------------------------------------------------------------- §17.2 corrected baselines (own runs)
def baselines8(rev, PRIMARY):
    ps = _ps()
    from .jsonl import results_path
    run = lambda n: ps.ROOT / "runs" / n
    if not results_path(run("rev8_cb_main")):
        return
    rows = lambda n, conds: [r for r in ps.rows_of(run(n)) if r["condition"] in conds]
    cbm = rows("cb_main", {"plain", "trustrag", "nli+flag", "nli+remove", "llm_dir+flag_directional"})
    new = rows("rev8_cb_main", {"trustrag_sys", "nli_win+flag", "nli_win+remove", "nli_win300+flag", "nli_win300+remove"})
    L = ps.LABELLED
    t = {}
    t["CB multi: TrustRAG with minus without system message (labelled)"] = ps.compare(cbm + new, "trustrag", "trustrag_sys", L, PRIMARY)
    for m in ("flag", "remove"):
        t[f"CB multi: windowed NLI {m} vs plain (conflict)"] = ps.compare(cbm + new, "plain", f"nli_win+{m}", True, PRIMARY)
        if any(r["condition"] == f"nli_win300+{m}" for r in new):
            t[f"CB multi: sentence-window NLI {m} vs plain (conflict)"] = ps.compare(cbm + new, "plain", f"nli_win300+{m}", True, PRIMARY)
        t[f"CB multi: truncated NLI {m} vs plain (conflict)"] = ps.compare(cbm + new, "plain", f"nli+{m}", True, PRIMARY)
    fam = {"CB multi (registered H14 prompt): LLM dir. flag vs TrustRAG+sys (labelled)":
           ps.compare(cbm + new, "trustrag_sys", "llm_dir+flag_directional", L, PRIMARY)}
    cbs = rows("cb_single", {"llm_dir+flag_directional"})
    fam["CB one-answer (matched): LLM dir. flag vs TrustRAG+sys (labelled)"] = ps.compare(cbs + new, "trustrag_sys", "llm_dir+flag_directional", L, PRIMARY)
    if results_path(run("rev8_cbmaj_single")):
        mj = rows("cbmaj_single", {"trustrag", "llm_dir+flag_directional"}) + rows("rev8_cbmaj_single", {"trustrag_sys"})
        t["CB majority: TrustRAG with minus without system message (labelled)"] = ps.compare(mj, "trustrag", "trustrag_sys", L, PRIMARY)
        fam["CB majority (one-answer): LLM dir. flag vs TrustRAG+sys (labelled)"] = ps.compare(mj, "trustrag_sys", "llm_dir+flag_directional", L, PRIMARY)
    if results_path(run("rev8_holdout_round2")):
        ho = rows("holdout_round2", {"trustrag", "trustrag_multi"}) + rows("rev8_holdout_round2", {"trustrag_sys", "trustrag_sys_multi"})
        ho += rows("holdout_main", {"llm_dir+flag_directional"})
        for v in ("", "_multi"):
            t[f"RAMDocs: TrustRAG{v.replace('_', ' ')} with minus without system message (labelled)"] = \
                ps.compare(ho, f"trustrag{v}", f"trustrag_sys{v}", L, PRIMARY)
        fam["RAMDocs (multi-answer): LLM dir. flag vs TrustRAG multi+sys (labelled)"] = ps.compare(ho, "trustrag_sys_multi", "llm_dir+flag_directional", L, PRIMARY)
    det = {}
    for sp in (ps.ROOT / "runs/rev8_detect_cb_nli/summary.json", ps.ROOT / "runs/rev8_detect_cb_nli300/summary.json"):
        if sp.exists():
            d = json.loads(sp.read_text())
            det.update({k: {"q": v["question_level"], "pair": v["pair_level"], "flag_rate": v["pair_flag_rate"]} for k, v in d.items()})
    rev["baselines8"] = {"tests": t, "H14_corrected": ps.family(fam, {}), "detect": det}


# --------------------------------------------------------------------------- §17.3 semantic re-score (own runs)
def _boot_mean(x, rng, n=5000):
    x = np.asarray(x, float)
    b = np.array([rng.choice(x, len(x)).mean() for _ in range(n)])
    return float(x.mean()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def judge8_stats(rev):
    ps = _ps()
    d = ps.ROOT / "runs/judge8"
    if not (d / "contrasts.jsonl").exists():
        return
    rng = np.random.default_rng(0)
    rows = [json.loads(l) for l in open(d / "contrasts.jsonl")]
    js = lambda j: float(bool(j.get("gives_gold")) and not j.get("asserts_wrong") and not j.get("abstains"))
    out = {"n_unparsed": sum(not r["judge"] for r in rows), "n": len(rows), "contrasts": {}, "by_condition": {}}
    byc = defaultdict(list)
    for r in rows:
        if r["judge"]:
            byc[r["condition"]].append((r["matcher"], js(r["judge"]), r))
    for c, v in byc.items():
        m = np.array([a for a, _, _ in v]); j = np.array([b for _, b, _ in v])
        rej = [r for _, _, r in v if r["wrong_rate"] > 0 and not r["judge"].get("asserts_wrong") and r["judge"].get("rejects_wrong")]
        out["by_condition"][c] = {"n": len(v), "agree": float(np.mean(m == j)), "matcher": float(m.mean()), "judge": float(j.mean()),
                                  "matcher_only": float(np.mean((m == 1) & (j == 0))), "judge_only": float(np.mean((m == 0) & (j == 1))),
                                  "rejected_but_counted_wrong": len(rej) / len(v)}
    by = defaultdict(dict)
    for r in rows:
        if r["judge"]:
            by[r["contrast"]].setdefault(r["id"], {})[r["arm"]] = (r["matcher"], js(r["judge"]))
    for name, q in by.items():
        ids = [i for i, a in q.items() if "a" in a and "b" in a]
        dm = [q[i]["b"][0] - q[i]["a"][0] for i in ids]; dj = [q[i]["b"][1] - q[i]["a"][1] for i in ids]
        dd = [y - x for x, y in zip(dm, dj)]
        out["contrasts"][name] = {"n": len(ids), "matcher": _boot_mean(dm, rng), "judge": _boot_mean(dj, rng),
                                  "judge_minus_matcher": _boot_mean(dd, rng)}
    if (d / "wiki.jsonl").exists():
        w = defaultdict(list)
        for l in open(d / "wiki.jsonl"):
            r = json.loads(l); j = r["judge"]
            if not j:
                continue
            both = bool(j.get("mentions_1")) and bool(j.get("mentions_2"))
            w[(r["gen"], r["condition"])].append((r["id"], float(both), float(both and j.get("acknowledges_conflict") and not j.get("unsupported_preference")),
                                                 float(r["matcher"] or 0)))
        wo = {}
        for (g, c), v in w.items():
            wo[f"{g}|{c}"] = {"n": len(v), "both": float(np.mean([x[1] for x in v])), "original": float(np.mean([x[2] for x in v])),
                              "matcher": float(np.mean([x[3] for x in v]))}
        for g in sorted({g for g, _ in w}):   # fixed order: the bootstrap shares one rng
            a = {x[0]: x for x in w.get((g, "plain"), [])}; b = {x[0]: x for x in w.get((g, "llm_dir+flag_directional"), [])}
            ids = sorted(set(a) & set(b))
            if ids:
                wo[f"{g}|effect"] = {k: _boot_mean([b[i][n] - a[i][n] for i in ids], rng) for k, n in (("both", 1), ("original", 2), ("matcher", 3))}
        out["wiki"] = wo
    if (d / "raguard.jsonl").exists():
        rg = [json.loads(l)["judge"] for l in open(d / "raguard.jsonl")]
        rg = [j for j in rg if j]
        out["raguard"] = {"n": len(rg), "contradict": float(np.mean([bool(j.get("contradict")) for j in rg])),
                          "temporal": float(np.mean([bool(j.get("temporally_compatible")) for j in rg])),
                          **{f"b_false_{k}": float(np.mean([j.get("b_false_about_claim") == k for j in rg])) for k in ("yes", "no", "unclear")},
                          **{f"a_supports_{k}": float(np.mean([j.get("a_supports_verdict") == k for j in rg])) for k in ("yes", "no", "unclear")},
                          "verified": float(np.mean([bool(j.get("contradict")) and j.get("b_false_about_claim") == "yes" and j.get("a_supports_verdict") == "yes" for j in rg]))}
    rev["judge8"] = out


# --------------------------------------------------------------------------- empty and truncated responses
def truncation8(rev, r2):
    """Per setting and condition, the share of responses that are empty, aborted by the server, or have no
    final-answer line (most of these hit the output budget)."""
    from .llm import ABORTED_PREFIX
    out = {}
    for k, v in r2.items():
        cnt = defaultdict(lambda: [0, 0, 0])
        for r in v["_rows"]:
            if r["condition"].startswith(("madam", "astute", "trustrag")):   # own output formats, no final-answer line
                continue
            t = (r.get("response") or "").strip()
            c = cnt[r["condition"]]
            c[0] += 1; c[1] += (not t) or t.startswith(ABORTED_PREFIX); c[2] += "final answer" not in t.lower()
        out[k] = {c: {"n": n, "empty_or_aborted": e / n, "no_final_line": f / n} for c, (n, e, f) in cnt.items()}
    flat = [(k, c, t) for k, d in out.items() for c, t in d.items()]
    worst = max(flat, key=lambda x: x[2]["no_final_line"]) if flat else None
    rev["truncation8"] = {"by_setting": out, "max_empty": max((t["empty_or_aborted"] for _, _, t in flat), default=0.0),
                          "max_no_final": worst[2]["no_final_line"] if worst else 0.0,
                          "max_no_final_at": f"{worst[0]}|{worst[1]}" if worst else None,
                          "median_no_final": float(np.median([t["no_final_line"] for _, _, t in flat])) if flat else 0.0}


def revision8(R, r2, S, PRIMARY, arms):
    rev = R["rev"]
    realised31(rev, arms)
    selection8(rev)
    predict8(rev, arms, PRIMARY)
    h23_het(rev, r2, PRIMARY)
    prevalence8(rev, S, PRIMARY)
    truncation8(rev, r2)
    baselines8(rev, PRIMARY)
    judge8_stats(rev)
    meta8(rev, S, PRIMARY)
