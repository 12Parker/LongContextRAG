"""
Tenth revision (PREREGISTRATION §19): the prospective follow-rate test, the full-sample semantic re-score of the
pooled equivalence tests, and HealthContradict.

  cells19 / vectors19   per-question vectors of a §19 cell (shared with predict19.py, which freezes predictions)
  fresh19               H32-H36 and the selection summary on ConflictBank-fresh
  hc19                  H37 on HealthContradict
  judge19               H38: pooled H30a / H23 under the judge
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .analyze import per_question
from .jsonl import read_results, results_path

ROOT = Path(__file__).resolve().parents[2]
PRIMARY19 = "ramdocs_strict_oneway"
FLAG19 = "llm_dir+flag_directional"
RAND19 = "synth-p1-r1-all-d0.5+flag_directional"
N_BOOT19 = 5000
# frozen in §19.1 (round-10 H31x): generator -> (follow right, follow wrong), and the 95% CIs H36 checks against
FOLLOW19 = {"gpt-4o-mini": (0.5658, 0.5143), "gpt-6-luna": (0.7202, 0.7719),
            "claude-haiku-5-5": (0.6288, 0.6392), "gpt-6.1-terra": (0.6667, 0.9474)}
FOLLOW19_CI = {"gpt-4o-mini": ((0.46, 0.67), (0.40, 0.62))}
GEN19 = {"mini": "gpt-4o-mini", "luna": "gpt-6-luna", "haiku": "claude-haiku-5-5", "terra": "gpt-6.1-terra"}
DET19 = {**GEN19, "qwen": "Qwen2.5-7B"}
HC_GENS = ("mini", "luna", "haiku")
PRED_PATH = ROOT / "experiments/conflict/splits/predictions19.json"


def cells19(runs: Path = ROOT / "runs") -> dict:
    out = {f"fresh_cb_{g}_{d}": ("conflictbank_fresh", GEN19[g], DET19[d], runs / f"fresh_cb_{g}_{d}")
           for g in GEN19 for d in DET19}
    out.update({f"hc_{g}": ("healthcontradict", GEN19[g], "gpt-4o-mini", runs / f"hc_{g}") for g in HC_GENS})
    return out


def _examples(run: Path):
    from . import data
    cfg = json.loads((run / "config.json").read_text())
    return {e.id: e for e in data.filter_ids(data.load(cfg["dataset"]), cfg.get("ids_file"))}


def _dets(run: Path) -> dict:
    return {r["id"]: r["pairs"] for r in map(json.loads, open(run / "detections.llm_dir.jsonl"))}


def vectors19(run: Path, ds: str, flagged: bool = True, metric: str = PRIMARY19, flag_cond: str = FLAG19) -> dict:
    """{"conflict"|"clean": {"ids", "g", "right", "wrong"[, "y"]}}: plain score (seed mean), whether the detector names
    exactly the misinformation side of a gold contradiction pair (as H24's d_q), whether its suspects include a correct
    passage, and (flagged) the flagged score. With flagged=False no flagged row is read."""
    ex, det = _examples(run), _dets(run)
    rows = [r for r in read_results(run) if r["condition"] == "plain" or (flagged and r["condition"] == flag_cond)]
    out = {}
    for part, sub in (("conflict", True), ("clean", False)):
        g = per_question(rows, "plain", sub, metric)
        y = per_question(rows, flag_cond, sub, metric) if flagged else None
        ids = sorted(i for i in g if i in det and i in ex and (y is None or i in y))
        right, wrong = [], []
        for i in ids:
            e = ex[i]; t = e.meta["doc_types"]
            contra = {tuple(p): {k for k in p if t[k] == "misinfo"} for p, lab in (e.gold_pairs or {}).items()
                      if lab == "contradiction"}
            right.append(float(any(contra.get((x["i"], x["j"])) == set(x["suspects"]) for x in det[i])))
            wrong.append(float(bool({k for x in det[i] for k in x["suspects"]} & {k for k, z in enumerate(t) if z == "correct"})))
        v = {"ids": ids, "g": np.array([g[i] for i in ids]), "right": np.array(right), "wrong": np.array(wrong)}
        if flagged:
            v["y"] = np.array([y[i] for i in ids])
        out[part] = v
    return out


def _pred(V, gen, ones=False):
    fr, fw = (1.0, 1.0) if ones else FOLLOW19[gen]
    return V["right"] * (1 - V["g"]) * fr - V["wrong"] * V["g"] * fw


def _ols_slope(P, O):
    P, O = np.asarray(P), np.asarray(O)
    return float(np.polyfit(P, O, 1)[0]) if np.std(P) > 0 else float("nan")


def _sign_flip_one(d, rng, n=10_000, less=True):
    """One-sided paired sign-flip p for mean(d) < 0 (less) or > 0."""
    d = np.asarray(d, float); obs = d.mean()
    s = rng.choice([-1.0, 1.0], size=(n, len(d)))
    m = (s * d).mean(1)
    hits = np.sum(m <= obs + 1e-12) if less else np.sum(m >= obs - 1e-12)
    return float((hits + 1) / (n + 1))


def _baseline_gen_mean(rev):
    q = rev.get("H24", {}).get("_q", {})
    by = defaultdict(list)
    for (g, _d), v in q.items():
        by[g].append(float(np.mean(np.array(v["y1"]) - np.array(v["g"]))))
    return {g: float(np.mean(x)) for g, x in by.items()}


# --------------------------------------------------------------------------- 19.1
def _follow(run: Path, metric=PRIMARY19):
    """H31-style follow rates on a randomised-direction run (as revision7._h31_arms / h31x)."""
    from .revision6 import _cell_outcomes
    ex = {i: e for i, e in _examples(run).items() if e.has_conflict}
    det = {r["id"]: r["pairs"] for r in map(json.loads, open(run / f"detections.{RAND19.split('+')[0]}.jsonl"))}
    rows = read_results(run)
    right = {i: all(ex[i].meta["doc_types"][k] == "misinfo" for x in det[i] for k in x["suspects"]) for i in ex if i in det}
    pl, fl = per_question(rows, "plain", True, metric), per_question(rows, RAND19, True, metric)
    ids = sorted(set(pl) & set(fl) & set(right))
    oc = _cell_outcomes(rows, RAND19, metric)
    fr = [oc[i][0] for i in ids if right[i] and pl[i] == 0 and i in oc]
    fw = [oc[i][1] for i in ids if not right[i] and pl[i] == 1 and i in oc]
    rng = np.random.default_rng(0)
    ci = lambda x: [float(np.quantile([np.mean(rng.choice(x, len(x))) for _ in range(N_BOOT19)], q)) for q in (0.025, 0.975)]
    return {"follow_right": float(np.mean(fr)), "follow_right_ci": ci(fr), "n_fr": len(fr),
            "follow_wrong": float(np.mean(fw)), "follow_wrong_ci": ci(fw), "n_fw": len(fw)}


def fresh19(rev):
    frozen = json.loads(PRED_PATH.read_text()) if PRED_PATH.exists() else {}
    cells = {k: v for k, v in cells19().items() if v[0] == "conflictbank_fresh" and results_path(v[3])}
    V = {}
    for name, (ds, gen, det, run) in cells.items():
        vv = vectors19(run, ds)
        if len(vv["conflict"]["ids"]) and FLAG19 in {r["condition"] for r in read_results(run)}:
            V[name] = (gen, det, vv)
    if len(V) < 3:
        return
    base_a = _baseline_gen_mean(rev)
    names = sorted(V)
    out = {"cells": {}, "n_cells": len(names)}
    for name in names:
        gen, det, vv = V[name]
        c, k = vv["conflict"], vv["clean"]
        o = {"gen": gen, "det": det, "n": len(c["ids"]),
             "pred": 100 * float(_pred(c, gen).mean()), "obs": 100 * float((c["y"] - c["g"]).mean()),
             "pred_ones": 100 * float(_pred(c, gen, True).mean()), "base_gen": 100 * base_a.get(gen, float("nan")),
             "marginal": 100 * (FOLLOW19[gen][0] * c["right"].mean() * (1 - c["g"].mean())
                                - FOLLOW19[gen][1] * c["wrong"].mean() * c["g"].mean()),
             "right": float(c["right"].mean()), "wrong": float(c["wrong"].mean()), "plain": 100 * float(c["g"].mean()),
             "frozen_pred": frozen.get(name, {}).get("conflict", {}).get("pred")}
        if len(k["ids"]):
            o["clean"] = {"pred": 100 * float(_pred(k, gen).mean()), "obs": 100 * float((k["y"] - k["g"]).mean()),
                          "wrong": float(k["wrong"].mean()), "n": len(k["ids"])}
        out["cells"][name] = o
    C = out["cells"]
    P = np.array([C[n]["pred"] for n in names]); O = np.array([C[n]["obs"] for n in names])
    if any(C[n]["frozen_pred"] is not None for n in names):
        out["max_frozen_drift"] = float(max(abs(C[n]["pred"] - C[n]["frozen_pred"]) for n in names
                                            if C[n]["frozen_pred"] is not None))
    # joint question bootstrap: one draw over the shared conflict questions for every cell
    qids = sorted({i for n in names for i in V[n][2]["conflict"]["ids"]}); pos = {i: k for k, i in enumerate(qids)}
    M = {}
    for n in names:
        gen, _, vv = V[n]; c = vv["conflict"]
        arr = {x: np.full(len(qids), np.nan) for x in ("p", "p1", "d")}
        ix = np.array([pos[i] for i in c["ids"]])
        arr["p"][ix] = _pred(c, gen); arr["p1"][ix] = _pred(c, gen, True); arr["d"][ix] = c["y"] - c["g"]
        M[n] = arr
    rng = np.random.default_rng(0)
    bs = defaultdict(list)
    for _ in range(N_BOOT19):
        idx = rng.integers(0, len(qids), len(qids))
        p = np.array([np.nanmean(M[n]["p"][idx]) for n in names]) * 100
        p1 = np.array([np.nanmean(M[n]["p1"][idx]) for n in names]) * 100
        o = np.array([np.nanmean(M[n]["d"][idx]) for n in names]) * 100
        ba = np.array([C[n]["base_gen"] for n in names])
        bs["slope"].append(_ols_slope(p, o)); bs["mae"].append(np.mean(np.abs(p - o)))
        bs["d_a"].append(np.mean(np.abs(ba - o)) - np.mean(np.abs(p - o)))
        bs["d_b"].append(np.mean(np.abs(p1 - o)) - np.mean(np.abs(p - o)))
    q = lambda x: [float(np.nanquantile(x, 0.025)), float(np.nanquantile(x, 0.975))]
    mae = float(np.mean(np.abs(P - O)))
    pa = float((np.sum(np.array(bs["d_a"]) <= 0) + 1) / (N_BOOT19 + 1))
    pb = float((np.sum(np.array(bs["d_b"]) <= 0) + 1) / (N_BOOT19 + 1))
    from .stats import holm
    hp = holm({"a": pa, "b": pb})
    slope = _ols_slope(P, O); sci = q(bs["slope"])
    gm = {g: (np.mean([C[n]["pred"] for n in names if C[n]["gen"] == g]), np.mean([C[n]["obs"] for n in names if C[n]["gen"] == g]))
          for g in {C[n]["gen"] for n in names}}
    pw = np.array([C[n]["pred"] - gm[C[n]["gen"]][0] for n in names]); ow = np.array([C[n]["obs"] - gm[C[n]["gen"]][1] for n in names])
    base = {"gen_mean": float(np.mean(np.abs(np.array([C[n]["base_gen"] for n in names]) - O))),
            "ones": float(np.mean(np.abs(np.array([C[n]["pred_ones"] for n in names]) - O))),
            "marginal": float(np.mean(np.abs(np.array([C[n]["marginal"] for n in names]) - O))),
            "zero": float(np.mean(np.abs(O)))}
    out["H32"] = {"slope": slope, "slope_ci": sci, "intercept": float(np.polyfit(P, O, 1)[1]),
                  "mae": mae, "mae_ci": q(bs["mae"]), "r": float(np.corrcoef(P, O)[0, 1]),
                  "r_within": float(np.corrcoef(pw, ow)[0, 1]), "baseline_mae": base,
                  "p_vs_gen_mean": pa, "p_vs_ones": pb, "p_holm_vs_gen_mean": hp["a"], "p_holm_vs_ones": hp["b"],
                  "mae_gain_vs_gen_mean_ci": q(bs["d_a"]), "mae_gain_vs_ones_ci": q(bs["d_b"]),
                  "sign_agree": int(np.sum(np.sign(P) == np.sign(O))),
                  "c1": bool(sci[0] > 0 and sci[0] <= 1 <= sci[1]), "c2": bool(mae <= 3),
                  "c3": bool(hp["a"] < 0.025 and hp["b"] < 0.025)}
    out["H32"]["supported"] = bool(out["H32"]["c1"] and out["H32"]["c2"] and out["H32"]["c3"])
    # H33: sign among |pred| >= 3; one-sided harm tests where pred <= -3
    big = [n for n in names if abs(C[n]["pred"]) >= 3]
    harm = {}
    for n in names:
        if C[n]["pred"] <= -3:
            c = V[n][2]["conflict"]
            harm[n] = {"p": _sign_flip_one(c["y"] - c["g"], rng), "diff": C[n]["obs"]}
    hh = holm({n: v["p"] for n, v in harm.items()}) if harm else {}
    for n in harm:
        harm[n]["p_holm"] = hh[n]
    agree = sum(np.sign(C[n]["pred"]) == np.sign(C[n]["obs"]) for n in big)
    out["H33"] = {"n_big": len(big), "agree": int(agree), "share": agree / len(big) if big else None,
                  "harm_tests": harm, "harm_confirmed": sum(v["p_holm"] < 0.05 for v in harm.values()),
                  "supported": bool(big and agree / len(big) >= 0.8 and all(v["p_holm"] < 0.05 for v in harm.values()))}
    # H34: within-generator correlation, detector labels permuted within generator
    r0 = float(np.corrcoef(pw, ow)[0, 1])
    gens = np.array([C[n]["gen"] for n in names]); perm = []
    for _ in range(10_000):
        o2 = ow.copy()
        for g in set(gens):
            m = np.where(gens == g)[0]; o2[m] = ow[rng.permutation(m)]
        perm.append(np.corrcoef(pw, o2)[0, 1])
    p34 = float((np.sum(np.array(perm) >= r0 - 1e-12) + 1) / (len(perm) + 1))
    out["H34"] = {"r_within": r0, "p": p34, "supported": bool(p34 < 0.05)}
    # H35: clean questions
    cl = [n for n in names if "clean" in C[n]]
    if cl:
        Pc = np.array([C[n]["clean"]["pred"] for n in cl]); Oc = np.array([C[n]["clean"]["obs"] for n in cl])
        out["H35"] = {"mae": float(np.mean(np.abs(Pc - Oc))), "sign_agree": int(np.sum(np.sign(Pc) == np.sign(Oc))),
                      "n": len(cl), "obs_mean": float(Oc.mean()), "pred_mean": float(Pc.mean()),
                      "r": float(np.corrcoef(Pc, Oc)[0, 1]) if np.std(Pc) > 0 else None}
    # selection over all questions (conflict and clean as sampled)
    sel = {}
    for g in sorted(set(gens)):
        ns = [n for n in names if C[n]["gen"] == g]
        def allq(n, key):
            c = C[n]; k = c.get("clean", {"n": 0, key: 0.0})
            return (c[key] * c["n"] + k[key] * k["n"]) / (c["n"] + k["n"])
        pr = {n: allq(n, "pred") for n in ns}; ob = {n: allq(n, "obs") for n in ns}
        pick = max(pr, key=pr.get)
        chosen = pick if pr[pick] > 0 else None
        mini = next((n for n in ns if C[n]["det"] == "gpt-4o-mini"), None)
        sel[g] = {"chosen": C[chosen]["det"] if chosen else "none", "chosen_obs": ob[chosen] if chosen else 0.0,
                  "best": C[max(ob, key=ob.get)]["det"], "best_obs": max(ob.values()),
                  "mini_obs": ob[mini] if mini else None, "pred_chosen": pr[chosen] if chosen else None}
    out["selection"] = sel
    # H36: follow-rate stability
    rd = ROOT / "runs/randdir_fresh_mini"
    if results_path(rd):
        f = _follow(rd)
        (lr, hr), (lw, hw) = FOLLOW19_CI["gpt-4o-mini"]
        f["supported"] = bool(lr <= f["follow_right"] <= hr and lw <= f["follow_wrong"] <= hw)
        out["H36"] = f
    rev["fresh19"] = out


# --------------------------------------------------------------------------- 19.3
def _cluster_flip(d: dict, groups: dict, rng, n=10_000, side=0):
    """Sign-flip permutation with the cluster (topic) as the unit; side +1/-1 one-sided, 0 two-sided."""
    by = defaultdict(list)
    for i, x in d.items():
        by[groups[i]].append(x)
    S = np.array([sum(v) for v in by.values()]); N = sum(len(v) for v in by.values())
    obs = S.sum() / N
    m = (rng.choice([-1.0, 1.0], size=(n, len(S))) * S).sum(1) / N
    hits = np.sum(np.abs(m) >= abs(obs) - 1e-12) if side == 0 else np.sum(side * m >= side * obs - 1e-12)
    return float((hits + 1) / (n + 1))


def hc19(rev):
    from .stats import holm, paired_bootstrap
    out = {}
    rng = np.random.default_rng(0)
    for g in HC_GENS:
        run = ROOT / f"runs/hc_{g}"
        if not results_path(run):
            continue
        rows = read_results(run)
        groups = {r["id"]: r.get("group", r["id"]) for r in rows}
        have = {r["condition"] for r in rows}
        pq = {c: per_question(rows, c, True, PRIMARY19) for c in have}
        o = {"plain": 100 * float(np.mean(list(pq["plain"].values()))), "n": len(pq["plain"])}
        if "closedbook" in pq:
            o["closedbook"] = 100 * float(np.mean(list(pq["closedbook"].values())))
        for key, a, b, side in (("H37a", "plain", "oracle+flag_directional", 1), ("H37b", "plain", "oracle+flag", 0),
                                ("H37c", "oracle+flag", "oracle+flag_directional", 1), ("llm", "plain", FLAG19, 0)):
            if a in pq and b in pq:
                ids = sorted(set(pq[a]) & set(pq[b]))
                d = {i: pq[b][i] - pq[a][i] for i in ids}
                bt = paired_bootstrap({i: pq[a][i] for i in ids}, {i: pq[b][i] for i in ids}, groups, n_boot=N_BOOT19)
                b90 = paired_bootstrap({i: pq[a][i] for i in ids}, {i: pq[b][i] for i in ids}, groups, n_boot=N_BOOT19, alpha=0.10)
                o[key] = {"diff": 100 * bt["diff"], "ci_low": 100 * bt["ci_low"], "ci_high": 100 * bt["ci_high"],
                          "ci90": [100 * b90["ci_low"], 100 * b90["ci_high"]], "p": _cluster_flip(d, groups, rng, side=side),
                          "n": len(ids), "n_topics": bt["n_clusters"]}
                o[key]["equivalent3"] = bool(-3 < o[key]["ci90"][0] and o[key]["ci90"][1] < 3)
        if FLAG19 in have and (run / "detections.llm_dir.jsonl").exists():
            vv = vectors19(run, "healthcontradict")["conflict"]
            o["llm"]["pred"] = 100 * float(_pred(vv, GEN19[g]).mean())
            o["llm"]["right"] = float(vv["right"].mean()); o["llm"]["wrong"] = float(vv["wrong"].mean())
        out[g] = o
    for key in ("H37a", "H37b", "H37c"):
        ps_ = {g: v[key]["p"] for g, v in out.items() if key in v}
        for g, p in holm(ps_).items():
            out[g][key]["p_holm"] = p
            want = {"H37a": 1, "H37c": 1, "H37b": 0}[key]
            out[g][key]["supported"] = bool(p < 0.05 and (want == 0 or np.sign(out[g][key]["diff"]) == want))
    lab = ROOT / "runs/judge10/hc_labels.jsonl"
    if lab.exists():
        L = [json.loads(l) for l in open(lab)]
        J = [x["judge"] for x in L if x["judge"]]
        yes = lambda k: float(np.mean([j.get(k) == "yes" for j in J])) if J else None
        rev["hc19_labels"] = {"n": len(L), "parsed": len(J), "correct_supports": yes("correct_supports_gold"),
                              "misinfo_supports_wrong": yes("misinfo_supports_wrong"),
                              "both": float(np.mean([j.get("correct_supports_gold") == "yes" and j.get("misinfo_supports_wrong") == "yes" for j in J])) if J else None}
    rev["hc19"] = out


# --------------------------------------------------------------------------- 19.2
def judge_strict(j: dict):
    return float(bool(j.get("gives_gold")) and not j.get("asserts_wrong") and not j.get("abstains")) if j else None


def _judge_map():
    p = ROOT / "runs/judge10/pooled.jsonl"
    if not p.exists():
        return None
    return {(x["dataset"], x["id"], x["key"]): judge_strict(x["judge"]) for x in map(json.loads, open(p))}


def _pq(rows, cond, score):
    acc = defaultdict(list)
    for r in rows:
        if r["condition"] == cond and r.get("has_conflict"):
            s = score(r)
            if s is not None:
                acc[r["id"]].append(s)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def judge19(rev, r2):
    from .judge10 import response_key
    from .revision5 import SHORT_GEN
    from .revision7 import _pool
    J = _judge_map()
    if J is None:
        return
    js = lambda r: J.get((r["dataset"], r["id"], response_key(r["response"])))
    both = lambda r: r[PRIMARY19] if js(r) is not None else None   # matcher on the judged rows only
    q = rev.get("H24", {}).get("_q", {})
    fams = {"H30a": [], "H23": []}
    for (g, d), v in sorted(q.items()):
        run = ROOT / f"runs/vkey_cb_{SHORT_GEN[g]}_{SHORT_GEN[d]}"
        if not results_path(run):
            continue
        rows = [r for r in v["rows"] if r["condition"] == FLAG19] + \
               [r for r in read_results(run) if r["condition"] == "llm_dir+verdict_key"]
        fams["H30a"].append(("CB", rows, "llm_dir+verdict_key", FLAG19))
    for k, v in r2.items():
        if k.startswith(("RD:", "CBs:", "CBmaj:")) and "answer_key" in {r["condition"] for r in v["_rows"]}:
            fams["H23"].append((k.split(":")[0], v["_rows"], "answer_key", "oracle+flag_directional"))
    out = {}
    for fam, cells in fams.items():
        res = {}
        for tag, score in (("judge", js), ("matcher", both)):
            cs = []
            for bench, rows, a, b in cells:
                A, B = _pq(rows, a, score), _pq(rows, b, score)
                d = {i: B[i] - A[i] for i in set(A) & set(B)}
                if d:
                    cs.append((bench, d))
            res[tag] = _pool(cs) if cs else None
            res[tag + "_cells"] = cs
        if res["judge"] and res["matcher"]:
            dd = []
            for (b1, dj), (_, dm) in zip(res["judge_cells"], res["matcher_cells"]):
                dd.append((b1, {i: dm[i] - dj[i] for i in set(dj) & set(dm)}))
            res["matcher_minus_judge"] = _pool(dd)
            for tag in ("judge", "matcher"):
                res[tag]["margin"] = max(abs(res[tag]["ci90_low"]), abs(res[tag]["ci90_high"]))
        res.pop("judge_cells"); res.pop("matcher_cells")
        out[fam] = res
    # coverage and agreement per condition
    agree = defaultdict(lambda: [0, 0, 0, 0])   # n, agree, judge-only correct, matcher-only correct
    seen = set()
    for fam, cells in fams.items():
        for _, rows, a, b in cells:
            for r in rows:
                if r["condition"] in (a, b) and r.get("has_conflict"):
                    key = (r["dataset"], r["id"], r["condition"], r["seed"], response_key(r["response"]))
                    j = js(r)
                    if key in seen or j is None:
                        continue
                    seen.add(key); m = r[PRIMARY19] >= 1
                    s = agree[r["condition"]]; s[0] += 1; s[1] += int(bool(j) == m); s[2] += int(j and not m); s[3] += int(m and not j)
    out["agreement"] = {c: {"n": v[0], "agree": v[1] / v[0], "judge_only": v[2] / v[0], "matcher_only": v[3] / v[0]}
                        for c, v in agree.items() if v[0]}
    rev["judge19"] = out


def _registered(rev):
    """§19.4: the §19 tests join the programme-wide BH count (revision7.fdr reads rev["H.."] nodes with p and p_holm)."""
    f = rev.get("fresh19", {})
    if "H32" in f:
        h = f["H32"]
        rev["H32"] = {"vs_gen_mean": {"p": h["p_vs_gen_mean"], "p_holm": h["p_holm_vs_gen_mean"]},
                      "vs_ones": {"p": h["p_vs_ones"], "p_holm": h["p_holm_vs_ones"]}}
    if f.get("H33", {}).get("harm_tests"):
        rev["H33"] = {n: {"p": v["p"], "p_holm": v["p_holm"]} for n, v in f["H33"]["harm_tests"].items()}
    if "H34" in f:
        rev["H34"] = {"within": {"p": f["H34"]["p"], "p_holm": f["H34"]["p"]}}
    hc = rev.get("hc19", {})
    t = {f"{g}|{k}": {"p": v[k]["p"], "p_holm": v[k]["p_holm"]} for g, v in hc.items() for k in ("H37a", "H37b", "H37c")
         if k in v and "p_holm" in v[k]}
    if t:
        rev["H37"] = t


def revision10(R, r2, S, PRIMARY):
    rev = R["rev"]
    fresh19(rev)
    hc19(rev)
    judge19(rev, r2)
    _registered(rev)
