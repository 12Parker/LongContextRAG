"""
Eleventh revision, the follow-rate account (PREREGISTRATION §20.2: R6, R7). Exploratory re-analyses of outputs
already seen, outside the FDR count; no API call.

  r6   R6 prospective-test summaries on the 20 ConflictBank-fresh cells, on H32's own bootstrap draws (the first
       5,000 question resamples of revision10.fresh19's rng, seed 0, same order), so each MAE difference is paired
       with H32: (a) MAE(cell-level marginal) - MAE(account); (b) a common-rate baseline (one f_R and one f_W for
       every generator); (c) clean questions over the cells whose detector flags a clean question, and the cells
       predicted at <= -3 on conflict questions; (d) the selection analysis' error split into conflict and clean parts
  r7   R7 gpt-6.1-terra's two plain draws: H31x (follow rates, break-even) with cb_single_terra's plain in place of
       randdir_cb_terra's; H19 with holdout_rem_terra's plain in place of holdout_terra's

    python -m experiments.conflict.rev11_account      # -> runs/rev11/rev11_account.json
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import numpy as np

from .analyze import per_question
from .jsonl import read_results, results_path
from .revision10 import FOLLOW19, N_BOOT19, PRIMARY19, _dets, _ols_slope, _pred, cells19, vectors19

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/rev11/rev11_account.json"
TERRA = "gpt-6.1-terra"
q95 = lambda x: [float(np.nanquantile(x, 0.025)), float(np.nanquantile(x, 0.975))]


def _ps():
    from . import paper_stats as ps
    return ps


# --------------------------------------------------------------------------- pure helpers
def common_rates(follow=FOLLOW19):
    """R6(b): one (f_R, f_W) for every generator, the equal-weight mean of the frozen pairs."""
    return tuple(float(np.mean([v[k] for v in follow.values()])) for k in (0, 1))


def pred_rates(V, fr, fw):
    """The frozen per-question formula (revision10._pred) with the given rates."""
    return V["right"] * (1 - V["g"]) * fr - V["wrong"] * V["g"] * fw


def marginal(fr, fw, right, wrong, g):
    """fresh19's cell-level marginal: the formula on the cell means of right, wrong and plain."""
    return fr * right * (1 - g) - fw * wrong * g


def error_split(c, k):
    """R6(d): the selection analysis' all-question error (pred - obs over conflict and clean questions as sampled) and
    its conflict and clean parts, each weighted by its share of questions; the two parts sum to the error."""
    n = c["n"] + k["n"]
    parts = {p: v["n"] / n * (v["pred"] - v["obs"]) for p, v in (("conflict", c), ("clean", k))}
    return {"pred": (c["pred"] * c["n"] + k["pred"] * k["n"]) / n, "obs": (c["obs"] * c["n"] + k["obs"] * k["n"]) / n,
            "err": parts["conflict"] + parts["clean"], **parts}


def h32_draws(n_q, n_boot=N_BOOT19, seed=0):
    """The question resamples of fresh19's H32 bootstrap, in its order (its rng's first n_boot calls)."""
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        yield rng.integers(0, n_q, n_q)


def flags_any(ids, dets):
    """Whether the detector flags (returns at least one pair for) any of these questions."""
    return any(bool(dets.get(i)) for i in ids)


def swap_plain(rows, other):
    """rows with their plain rows replaced by `other`'s (the other draw of the same plain prompts)."""
    key = lambda r: (r["id"], r["seed"])
    a = {key(r) for r in rows if r["condition"] == "plain"}; b = [r for r in other if r["condition"] == "plain"]
    if a != {key(r) for r in b}:
        raise ValueError("the two plain draws cover different (question, seed) keys")
    return [r for r in rows if r["condition"] != "plain"] + b


def _stack(V, names, part, fields):
    """Cells x questions arrays (NaN where a cell lacks a question) over the union of the cells' ids, sorted."""
    qids = sorted({i for n in names for i in V[n][2][part]["ids"]}); pos = {i: k for k, i in enumerate(qids)}
    A = {x: np.full((len(names), len(qids)), np.nan) for x in fields}
    for ci, n in enumerate(names):
        gen, _, vv, _ = V[n]; v = vv[part]; ix = np.array([pos[i] for i in v["ids"]], int)
        for x, f in fields.items():
            A[x][ci, ix] = f(v, gen)
    return qids, A


# --------------------------------------------------------------------------- R6
def r6(R):
    V = {}
    for name, (ds, gen, det, run) in cells19().items():
        if ds == "conflictbank_fresh" and results_path(run):
            vv = vectors19(run, ds)
            if len(vv["conflict"]["ids"]):
                V[name] = (gen, det, vv, run)
    if len(V) < 3:
        return {"not_run": "the ConflictBank-fresh cells are missing"}
    names = sorted(V); gens = [V[n][0] for n in names]
    stored = R.get("rev", {}).get("fresh19", {})
    fc = common_rates()
    FR = np.array([FOLLOW19[g][0] for g in gens]); FW = np.array([FOLLOW19[g][1] for g in gens])
    qids, A = _stack(V, names, "conflict", {
        "p": lambda v, g: _pred(v, g), "p1": lambda v, g: _pred(v, g, True), "pc": lambda v, g: pred_rates(v, *fc),
        "d": lambda v, g: v["y"] - v["g"], "r": lambda v, g: v["right"], "w": lambda v, g: v["wrong"],
        "g": lambda v, g: v["g"]})

    def stats(idx):
        m = {x: np.nanmean(a[:, idx], 1) for x, a in A.items()}
        return {"P": 100 * m["p"], "P1": 100 * m["p1"], "Pc": 100 * m["pc"], "O": 100 * m["d"],
                "Mg": 100 * marginal(FR, FW, m["r"], m["w"], m["g"])}
    full = stats(np.arange(len(qids)))
    mae = lambda s, k: float(np.mean(np.abs(s[k] - s["O"])))
    bs = {k: [] for k in ("mae", "slope", "d_ones", "d_marg", "d_common", "mae_common", "mae_marg")}
    cellP, cellO = [], []
    for idx in h32_draws(len(qids)):
        s = stats(idx)
        bs["mae"].append(mae(s, "P")); bs["slope"].append(_ols_slope(s["P"], s["O"]))
        bs["d_ones"].append(mae(s, "P1") - mae(s, "P"))
        bs["d_marg"].append(mae(s, "Mg") - mae(s, "P")); bs["mae_marg"].append(mae(s, "Mg"))
        bs["d_common"].append(mae(s, "Pc") - mae(s, "P")); bs["mae_common"].append(mae(s, "Pc"))
        cellP.append(s["P"]); cellO.append(s["O"])
    cellP, cellO = np.array(cellP), np.array(cellO)
    h = stored.get("H32", {})
    check = {"mae_ci": q95(bs["mae"]), "slope_ci": q95(bs["slope"]), "ones_gain_ci": q95(bs["d_ones"])}
    check["same_draws_as_H32"] = bool(h) and all(np.allclose(check[a], h[b]) for a, b in (
        ("mae_ci", "mae_ci"), ("slope_ci", "slope_ci"), ("ones_gain_ci", "mae_gain_vs_ones_ci")))
    C = stored.get("cells", {})
    check["max_abs_diff_vs_fresh19"] = float(max(abs(full[k][i] - C[n][f]) for i, n in enumerate(names)
                                                 for k, f in (("P", "pred"), ("O", "obs"), ("Mg", "marginal")))) if C else None
    out = {"n_cells": len(names), "n_q": len(qids), "n_boot": N_BOOT19, "draws": check}
    out["a"] = {"mae_account": mae(full, "P"), "mae_marginal": mae(full, "Mg"), "mae_marginal_ci": q95(bs["mae_marg"]),
                "diff": mae(full, "Mg") - mae(full, "P"), "diff_ci": q95(bs["d_marg"])}
    out["b"] = {"f_right": fc[0], "f_wrong": fc[1], "mae": mae(full, "Pc"), "mae_ci": q95(bs["mae_common"]),
                "diff": mae(full, "Pc") - mae(full, "P"), "diff_ci": q95(bs["d_common"]),
                "slope": _ols_slope(full["Pc"], full["O"]), "r": float(np.corrcoef(full["Pc"], full["O"])[0, 1]),
                "sign_agree": int(np.sum(np.sign(full["Pc"]) == np.sign(full["O"]))),
                "cells": {n: {"pred": float(full["Pc"][i]), "obs": float(full["O"][i])} for i, n in enumerate(names)}}
    # (c) clean questions; CI: a joint bootstrap over the shared clean questions (seed 0), separate from H32's draws
    kq, K = _stack(V, names, "clean", {"p": lambda v, g: _pred(v, g), "d": lambda v, g: v["y"] - v["g"]})
    flag = np.array([flags_any(V[n][2]["clean"]["ids"], _dets(V[n][3])) for n in names])
    def cmae(idx):
        e = np.abs(np.nanmean(K["p"][:, idx], 1) - np.nanmean(K["d"][:, idx], 1)) * 100
        return float(e.mean()), float(e[flag].mean()) if flag.any() else float("nan")
    rng = np.random.default_rng(0)
    cb = np.array([cmae(rng.integers(0, len(kq), len(kq))) for _ in range(N_BOOT19)])
    a20, af = cmae(np.arange(len(kq)))
    cells_c = {}
    for i, n in enumerate(names):
        k = V[n][2]["clean"]; d = _dets(V[n][3])
        cells_c[n] = {"pred": 100 * float(_pred(k, gens[i]).mean()), "obs": 100 * float((k["y"] - k["g"]).mean()),
                      "n": len(k["ids"]), "n_flagged": sum(bool(d.get(j)) for j in k["ids"]), "n_wrong": int(k["wrong"].sum())}
    neg = {n: {"pred": float(full["P"][i]), "pred_ci": q95(cellP[:, i]), "obs": float(full["O"][i]),
               "obs_ci": q95(cellO[:, i]), "obs_minus_pred_ci": q95(cellO[:, i] - cellP[:, i]),
               "n": len(V[n][2]["conflict"]["ids"])} for i, n in enumerate(names) if full["P"][i] <= -3}
    out["c"] = {"mae_20": a20, "mae_20_ci": q95(cb[:, 0]), "n_flagging": int(flag.sum()),
                "flagging_cells": [n for n, f in zip(names, flag) if f], "mae_flagging": af, "mae_flagging_ci": q95(cb[:, 1]),
                "cells": cells_c, "negative_cells": neg}
    # (d) the selection analysis (fresh19's rule: per generator, the detector with the highest all-question prediction)
    split = {}
    for i, n in enumerate(names):
        c = {"n": len(V[n][2]["conflict"]["ids"]), "pred": float(full["P"][i]), "obs": float(full["O"][i])}
        split[n] = {"gen": gens[i], "det": V[n][1], **error_split(c, cells_c[n])}
    sel = {}
    for g in sorted(set(gens)):
        ns = [n for n in names if split[n]["gen"] == g]
        pick = max(ns, key=lambda n: split[n]["pred"])
        sel[g] = {"chosen": "none"} if split[pick]["pred"] <= 0 else \
            {"chosen": split[pick]["det"], **{k: split[pick][k] for k in ("pred", "obs", "err", "conflict", "clean")}}
    E = {p: np.array([split[n][p] for n in names]) for p in ("err", "conflict", "clean")}
    out["d"] = {"cells": split, "chosen": sel,
                "summary": {"mae": float(np.abs(E["err"]).mean()), "mean_abs_conflict": float(np.abs(E["conflict"]).mean()),
                            "mean_abs_clean": float(np.abs(E["clean"]).mean()), "mean_conflict": float(E["conflict"].mean()),
                            "mean_clean": float(E["clean"].mean()),
                            "mae_flagging": float(np.abs(E["err"][flag]).mean()) if flag.any() else None,
                            "mae_not_flagging": float(np.abs(E["err"][~flag]).mean()) if (~flag).any() else None}}
    return out


# --------------------------------------------------------------------------- R7
H31_RUN, H31_OTHER = "randdir_cb_terra", "cb_single_terra"
H19_SETS = {"CBs:gpt-6-luna": ("cb_single_luna", "ctl_cb_luna"), "CBs:gpt-6.1-terra": ("cb_single_terra", "ctl_cb_terra"),
            "RD:gpt-6-luna": ("holdout_luna", "ctl_rd_luna"), "RD:gpt-6.1-terra": ("holdout_terra", "ctl_rd_terra"),
            "CBmaj:gpt-6-luna": ("cbmaj_single_luna", "ctl_mj_luna"),
            "CBmaj:gpt-6.1-terra": ("cbmaj_single_terra", "ctl_mj_terra")}
H19_KEEP = ("diff", "ci_low", "ci_high", "p", "p_holm", "supported", "mean_a", "mean_b", "n")


def _h31x(arms, rev):
    """revision7.h31x run on the given arms: same code, rng (seed 0) and draw order (mini, luna, Haiku, terra)."""
    from . import revision7
    stub = {"detector_precision7": rev.get("detector_precision7", {})}
    with mock.patch.object(revision7, "_h31_arms", lambda _p: arms):
        revision7.h31x(stub, _ps().PRIMARY)
    return stub["H31x"]


def _h19(sets):
    """paper_stats.revision3's H19 family (its spec; compare and Holm per metric, as paper_stats' fam) on these rows."""
    ps = _ps()
    r2 = {k: {"_rows": v} for k, v in sets.items()}
    def fam(name, spec, dirs):
        return {m: ps.family({tk: ps.compare(r2[key]["_rows"], a, b, sub, m) for tk, key, a, b, sub in spec},
                             {tk: dirs.get(tk, 0) for tk, *_ in spec}) for m in (ps.PRIMARY, ps.ROBUST)}
    stub = {"rev": {}}
    ps.revision3(stub, r2, fam, None)   # with only these six settings in r2, revision3 computes H19 alone
    # compare's mean_a / mean_b average over a set, so their last bit varies with the hash seed: rounded
    rd = lambda k, x: round(x, 12) if k in ("mean_a", "mean_b") else x
    return {m: {tk: {k: rd(k, t[k]) for k in H19_KEEP} for tk, t in v.items()} for m, v in stub["rev"]["H19"].items()}


def _plain_summary(a, b):
    """Two plain draws of the same prompts: identical texts; per metric and subset, the two plain means and the number
    of questions whose score differs."""
    ps = _ps()
    A = {(r["id"], r["seed"]): r for r in a if r["condition"] == "plain"}
    B = {(r["id"], r["seed"]): r for r in b if r["condition"] == "plain"}
    out = {"n_rows": len(A), "identical_texts": sum(A[k]["response"] == B[k]["response"] for k in set(A) & set(B))}
    for m in (ps.PRIMARY, ps.ROBUST):
        for sub, f in (("conflict", True), ("clean", False)):
            pa, pb = per_question(a, "plain", f, m), per_question(b, "plain", f, m)
            sh = sorted(set(pa) & set(pb))
            out[f"{m}|{sub}"] = {"n": len(sh), "mean": [100 * float(np.mean([x[i] for i in sh])) for x in (pa, pb)],
                                 "score_differs": sum(pa[i] != pb[i] for i in sh)}
    return out


def r7(R):
    from .revision7 import _h31_arms
    ps = _ps(); rev = R.get("rev", {})
    need = [H31_RUN, H31_OTHER, "holdout_rem_terra", *(r for v in H19_SETS.values() for r in v)]
    miss = [r for r in need if not results_path(ROOT / "runs" / r)]
    if miss:
        return {"not_run": f"missing runs: {', '.join(miss)}"}
    rows = lambda r: read_results(ROOT / "runs" / r)
    temp = {r: json.loads((ROOT / "runs" / r / "config.json").read_text())["temperature"]
            for r in (H31_RUN, H31_OTHER, "holdout_terra", "holdout_rem_terra")}
    out = {"draw_used": {"H31 (H31a-c, H31_follow, H31x; frozen FOLLOW19)": H31_RUN,
                         "H19 ConflictBank tests (and Table 2)": "cb_single_terra",
                         "H19 RAMDocs tests (and Table 2)": "holdout_terra"},
           "other_draw": {"H31": H31_OTHER, "H19 RAMDocs": "holdout_rem_terra"},
           "requested_temperature": temp,
           "plain": {b: {"runs": [x, y], **_plain_summary(rows(x), rows(y))}
                     for b, x, y in (("ConflictBank", H31_RUN, H31_OTHER), ("RAMDocs", "holdout_terra", "holdout_rem_terra"))}}
    # H31: same flagged rows and right/wrong assignment; only terra's plain changes
    arms = _h31_arms(ps.PRIMARY)
    reg = _h31x(arms, rev)
    a = arms[TERRA]; pl = per_question(rows(H31_OTHER), "plain", True, ps.PRIMARY)
    arms2 = {**arms, TERRA: {**a, "pl": pl, "R": [i for i in a["R"] if i in pl], "W": [i for i in a["W"] if i in pl]}}
    oth = _h31x(arms2, rev)
    st = rev.get("H31x", {})
    keys = ("breakeven", "be_low", "be_high", "follow_right", "follow_wrong", "right", "wrong")
    out["H31"] = {"registered": reg[TERRA], "other_draw": oth[TERRA],
                  "same_arms": arms2[TERRA]["R"] == a["R"] and arms2[TERRA]["W"] == a["W"],
                  "reproduces_stored": bool(st) and all(np.allclose(reg[g][k], st[g][k]) for g in st for k in keys),
                  "other_generators_unchanged": all(reg[g] == oth[g] for g in reg if g != TERRA)}
    # H19: the RAMDocs terra plain swapped (registered); ConflictBank terra plain as well (supplementary)
    sets = {k: rows(b) + [r for r in rows(c) if r["condition"] != "plain"] for k, (b, c) in H19_SETS.items()}
    v_reg = _h19(sets)
    sets_rd = {**sets, "RD:gpt-6.1-terra": swap_plain(sets["RD:gpt-6.1-terra"], rows("holdout_rem_terra"))}
    v_rd = _h19(sets_rd)
    v_both = _h19({**sets_rd, "CBs:gpt-6.1-terra": swap_plain(sets["CBs:gpt-6.1-terra"], rows(H31_RUN))})
    st = rev.get("H19", {})
    changed = lambda v: {m: [tk for tk in v[m] if v[m][tk]["supported"] != v_reg[m][tk]["supported"]] for m in v}
    out["H19"] = {"registered": v_reg, "other_draw": v_rd, "changed_status": changed(v_rd),
                  "reproduces_stored": bool(st) and all(np.allclose([v_reg[m][tk][k] for k in ("diff", "p", "p_holm")],
                                                                    [st[m][tk][k] for k in ("diff", "p", "p_holm")])
                                                        for m in st for tk in st[m]),
                  "supplementary_cb_too": {"note": "not registered in PR§20: H19's ConflictBank terra plain also swapped, "
                                                   "to randdir_cb_terra's draw", "results": v_both,
                                           "changed_status": changed(v_both)}}
    return out


# --------------------------------------------------------------------------- entry points
def _js(x):
    if isinstance(x, dict):
        return {str(k): _js(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_js(v) for v in x]
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return float(x)
    return x


def compute(R: dict) -> dict:
    return _js({"R6": r6(R), "R7": r7(R)})


def _summary(o):
    f = lambda c: f"[{c[0]:.2f}, {c[1]:.2f}]"
    s = o["R6"]
    if "not_run" in s:
        print("R6 not run:", s["not_run"])
    else:
        print(f"R6 ({s['n_cells']} cells, {s['n_q']} conflict q., same draws as H32: {s['draws']['same_draws_as_H32']})")
        a, b, c, d = s["a"], s["b"], s["c"], s["d"]
        print(f"  (a) MAE account {a['mae_account']:.2f}, marginal {a['mae_marginal']:.2f}; "
              f"marginal - account {a['diff']:+.2f} {f(a['diff_ci'])}")
        print(f"  (b) common rates f_R {b['f_right']:.4f}, f_W {b['f_wrong']:.4f}: MAE {b['mae']:.2f} {f(b['mae_ci'])}; "
              f"common - account {b['diff']:+.2f} {f(b['diff_ci'])}; slope {b['slope']:.2f}, r {b['r']:.3f}")
        print(f"  (c) clean MAE, 20 cells {c['mae_20']:.2f} {f(c['mae_20_ci'])}; {c['n_flagging']} flagging cells "
              f"{c['mae_flagging']:.2f} {f(c['mae_flagging_ci'])}")
        for n, v in c["negative_cells"].items():
            print(f"      {n}: pred {v['pred']:+.1f} {f(v['pred_ci'])}, obs {v['obs']:+.1f} {f(v['obs_ci'])}")
        m = d["summary"]
        print(f"  (d) all-question MAE {m['mae']:.2f}: mean |conflict part| {m['mean_abs_conflict']:.2f}, "
              f"|clean part| {m['mean_abs_clean']:.2f} (signed {m['mean_conflict']:+.2f} / {m['mean_clean']:+.2f})")
        for g, v in d["chosen"].items():
            if "err" in v:
                print(f"      {g} -> {v['chosen']}: pred {v['pred']:+.2f}, obs {v['obs']:+.2f}, "
                      f"error {v['err']:+.2f} = conflict {v['conflict']:+.2f} + clean {v['clean']:+.2f}")
    t = o["R7"]
    if "not_run" in t:
        print("R7 not run:", t["not_run"]); return
    for bench, v in t["plain"].items():
        c = v[f"{_ps().PRIMARY}|conflict"]
        print(f"R7 terra plain draws, {bench}: {v['identical_texts']}/{v['n_rows']} identical texts; conflict plain "
              f"{c['mean'][0]:.1f} ({v['runs'][0]}) vs {c['mean'][1]:.1f} ({v['runs'][1]}), "
              f"score differs on {c['score_differs']}/{c['n']}")
    for nm, v in (("H31 " + H31_RUN, t["H31"]["registered"]), ("H31 " + H31_OTHER, t["H31"]["other_draw"])):
        print(f"  {nm}: f_R {v['follow_right']:.3f} {f(v['follow_right_ci'])} (n {v['n_fr']}), f_W {v['follow_wrong']:.3f} "
              f"{f(v['follow_wrong_ci'])} (n {v['n_fw']}), break-even {v['breakeven']:.3f} [{v['be_low']:.3f}, {v['be_high']:.3f}]")
    print(f"  H31 reproduces stored: {t['H31']['reproduces_stored']} (same right/wrong arms: {t['H31']['same_arms']}); "
          f"H19 reproduces stored: {t['H19']['reproduces_stored']}")
    P = _ps().PRIMARY
    for tk, v in t["H19"]["other_draw"][P].items():
        if "gpt-6.1-terra" in tk and "RAMDocs" in tk:
            r = t["H19"]["registered"][P][tk]
            print(f"  H19 {tk}: {100 * r['diff']:+.2f} (p_holm {r['p_holm']:.3f}) -> {100 * v['diff']:+.2f} "
                  f"[{100 * v['ci_low']:.2f}, {100 * v['ci_high']:.2f}] (p_holm {v['p_holm']:.3f})")
    print(f"  H19 status changes: {t['H19']['changed_status']}; with CB too (supplementary): "
          f"{t['H19']['supplementary_cb_too']['changed_status']}")
    for m in (P, _ps().ROBUST):   # supplementary, not registered: H19's ConflictBank terra plain swapped as well
        for tk, v in t["H19"]["supplementary_cb_too"]["results"][m].items():
            r = t["H19"]["registered"][m][tk]
            if "gpt-6.1-terra" in tk and tk.startswith("ConflictBank") and v["diff"] != r["diff"]:
                print(f"  (suppl., {m}) H19 {tk}: {100 * r['diff']:+.2f} (p_holm {r['p_holm']:.4f}) -> "
                      f"{100 * v['diff']:+.2f} (p_holm {v['p_holm']:.4f})")


def main():
    R = json.loads((ROOT / "paper/generated/hypotheses.json").read_text())
    out = compute(R)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True))
    _summary(out)
    print(f"-> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
