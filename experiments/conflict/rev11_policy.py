"""
Tenth-revision re-analyses, part 4 (PREREGISTRATION §20.4: real documents, answer policy and prevalence). Exploratory:
outputs already seen, outside the FDR count, no model call.

  R13  HealthContradict page stance: the judge's joint verdicts on the two pages (§19.3 audit), pages at the 4,000-char cut
  R14  RAGuard: answer key minus the oracle directional flag on conflict claims, per generator
  R15  RAMDocs single-entity questions: the §18.2 equal-weight mean leaving out one setting at a time
  R16  risk-coverage operating points (plain, oracle neutral, oracle directional) under the one-way matcher (R12)
  R17  RAMDocs answer mix under the minimal skeleton; clean questions with several gold answers vs one
  R18  prevalence curves (pi 0.01-0.50) with joint-bootstrap bands; break-even shares with CIs; clean-set groups
  R19  H4b label-robust correlation (holdout_contains): permutation p-value and BCa interval

Settings are revision6.settings' 34 (the §16.4 pooled set), rebuilt from each setting's base run: the §13 control runs,
holdout_extra and the baselines that paper_stats merges in add only answer_key, oracle+flag_pick / _onewrong / _nopairs,
self_identify, closedbook, astute and trustrag rows, which R15-R18 do not read.
Selective accuracy (R16) is the §15.4.5 ratio estimand (sum correct / sum answered over question x seed), as in H21(c)
and Table 2's Sel. column; the per-question mean it replaced is kept beside it as `selective_perq`.

    python -m experiments.conflict.rev11_policy     # writes runs/rev11/rev11_policy.json
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

from .analyze import per_question
from .jsonl import read_results, results_path
from .metrics import _contains, normalize_answer
from .stats import paired_bootstrap

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/rev11/rev11_policy.json"
PRIMARY = "ramdocs_strict_oneway"
N_BOOT = 5000
NEUTRAL, DIRECTIONAL, LLM_FLAG, LLM_DIR = "oracle+flag", "oracle+flag_directional", "llm_dir+flag", "llm_dir+flag_directional"
PI = [round(0.01 * k, 2) for k in range(1, 51)]
HC_EVAL = ROOT / "data/external/healthcontradict/hc_eval.jsonl"
HC_LABELS = ROOT / "runs/judge10/hc_labels.jsonl"
HC_CUT = 4000   # prepare_healthcontradict.MAX_CHARS
RG_RUNS = {"gpt-4o-mini": "rg_mini", "gpt-6-luna": "rg_luna", "gpt-6.1-terra": "rg_terra",
           "claude-haiku-5-5": "rg_haiku", "claude-sonnet-5-5": "rg_sonnet"}
MINIMAL_RD = ("RD:gpt-4o-mini|minimal", "RD:gpt-4.1|minimal", "RD:Qwen2.5-7B|minimal")
# setting -> base run (paper_stats.RUNS / revision2 sets; RAMDocs generality settings as revision6.settings)
RUN_OF = {"CB:gpt-4o-mini": "cb_main", "CB:gpt-4.1": "cb_gpt41", "CB:Qwen2.5-7B": "cb_qwen",
          "CBs:gpt-4o-mini": "cb_single", "CBs:gpt-4.1": "cb_single_gpt41", "CBs:Qwen2.5-7B": "cb_single_qwen",
          "CBn:gpt-4o-mini": "cb_minimal", "CBmaj:gpt-4o-mini": "cbmaj_single",
          "CBs:gpt-6-luna": "cb_single_luna", "CBs:gpt-6.1-terra": "cb_single_terra",
          "RD:gpt-6-luna": "holdout_luna", "RD:gpt-6.1-terra": "holdout_terra",
          "CBmaj:gpt-6-luna": "cbmaj_single_luna", "CBmaj:gpt-6.1-terra": "cbmaj_single_terra",
          "CBs:claude-haiku-5-5": "cb_single_haiku", "CBs:claude-sonnet-5-5": "cb_single_sonnet",
          "RD:claude-haiku-5-5": "holdout_haiku", "RD:claude-sonnet-5-5": "holdout_sonnet",
          "CBmaj:claude-haiku-5-5": "cbmaj_single_haiku",
          "CBf:gpt-4o-mini": "cb_forced_mini", "CBf:gpt-6-luna": "cb_forced_luna",
          "CBf:gpt-6.1-terra": "cb_forced_terra", "CBf:claude-haiku-5-5": "cb_forced_haiku",
          **{f"RG:{g}": r for g, r in RG_RUNS.items()},
          "RD:gpt-4o-mini|default": "holdout_main", "RD:gpt-4o-mini|minimal": "holdout_minimal",
          "RD:gpt-4.1|default": "holdout_gpt41", "RD:gpt-4.1|minimal": "holdout_gpt41_minimal",
          "RD:Qwen2.5-7B|default": "holdout_qwen", "RD:Qwen2.5-7B|minimal": "holdout_qwen_minimal"}
BENCH = {"RD": "RAMDocs", "RG": "RAGuard", "CBmaj": "CB majority"}
POLICY = {"CB": "multi", "RD": "multi", "CBn": "minimal", "CBs": "single", "CBmaj": "single", "RG": "single", "CBf": "forced"}
# R18: how each benchmark's clean questions were built
CLEAN_GROUP = {"ConflictBank": "cb_distractors", "CB majority": "cb_distractors", "RAMDocs": "ramdocs_multi",
               "RAGuard": "raguard"}
CLEAN_BUILD = {"cb_distractors": "ConflictBank: the correct passage plus three off-subject distractors (the majority "
                                 "variant reuses the same clean items)",
               "ramdocs_multi": "RAMDocs: questions without misinformation, mostly with several valid answers",
               "raguard": "RAGuard: claims with supporting posts and no misleading post",
               "other": "none of the round-10 settings"}
KEEP_CONDS = ("plain", NEUTRAL, DIRECTIONAL, LLM_FLAG, LLM_DIR, "answer_key")
KEEP_KEYS = ("id", "condition", "seed", "has_conflict", PRIMARY, "pred_answers", "n_flags")


# --------------------------------------------------------------------------- loading
@lru_cache(maxsize=None)
def _rows(run: str) -> list:
    """Slim rows of the conditions used here."""
    return [{k: r.get(k) for k in KEEP_KEYS} for r in read_results(ROOT / "runs" / run) if r["condition"] in KEEP_CONDS]


@lru_cache(maxsize=None)
def _examples(run: str) -> dict:
    from .paper_stats import examples_of
    return {e.id: e for e in examples_of(ROOT / "runs" / run)}


def _cfg(run: str) -> dict:
    return json.loads((ROOT / "runs" / run / "config.json").read_text())


def settings(R: dict) -> list[dict]:
    """revision6.settings, by name: r2 settings with an oracle neutral flag, then the RAMDocs generality settings."""
    rev, out = R["rev"], []
    for k, v in rev["r2"].items():
        if k.startswith(("CB", "RD:gpt-6", "RD:claude", "RG")) and NEUTRAL in v["effects"][PRIMARY]:
            pre, gen = k.split(":", 1)
            out.append({"name": k, "gen": gen, "bench": BENCH.get(pre, "ConflictBank"), "policy": POLICY[pre]})
    for gk, v in rev["generality"].items():
        if NEUTRAL in v["effects"][PRIMARY]:
            g, sk = gk.split("|")
            out.append({"name": f"RD:{gk}", "gen": g, "bench": "RAMDocs", "policy": "multi" if sk == "default" else "minimal"})
    for s in out:
        s["run"] = RUN_OF[s["name"]]
    return out


def _stored_plain(R, name):
    """Stored plain one-way strict accuracy on conflict questions (fraction), for the reconstruction check."""
    rev = R["rev"]
    if name in rev["r2"]:
        return rev["r2"][name]["plain"][PRIMARY]["conflict"]
    return rev["generality"][name[3:]]["plain"][PRIMARY]["conflict"]


# --------------------------------------------------------------------------- helpers
def _unk(r) -> bool:
    """revision5._unk: an answer is the literal "unknown"."""
    return any(a.strip().lower().rstrip(".") == "unknown" for a in r.get("pred_answers") or [])


def misinfo_oneway(pred_answers, wrong_answers) -> float:
    """R12: the answer contains a wrong answer as a whole-word span (metrics._contains on the joined, normalised
    prediction, exactly as the one-way strict score does)."""
    pred = normalize_answer(" ; ".join(pred_answers or []))
    return float(any(_contains(pred, w) for w in wrong_answers))


def outcome_oneway(r, wrong_answers) -> str:
    """Mutually exclusive outcome (paper_stats' answer mix) with one-way matching throughout."""
    if r[PRIMARY] == 1:
        return "correct"
    if misinfo_oneway(r.get("pred_answers"), wrong_answers):
        return "misinfo"
    return "abstain" if _unk(r) else "other"


def metric_fns(wrong: dict) -> dict:
    """Per-row scores; None drops the row (the per-question selective mean counts answered rows only)."""
    return {"strict": lambda r: r[PRIMARY],
            "misinfo": lambda r: misinfo_oneway(r.get("pred_answers"), wrong.get(r["id"], [])),
            "abstain": lambda r: float(_unk(r)),
            "coverage": lambda r: 1.0 - float(_unk(r)),
            "selective_perq": lambda r: None if _unk(r) else r[PRIMARY]}


def sel_arrays(rows, a, b, sub) -> np.ndarray:
    """revision5._sel_arrays for conflict (True) or clean (False) questions, ids sorted: per question, correct and
    answered counts over seeds in a, then in b."""
    acc = defaultdict(lambda: np.zeros(4))
    for r in rows:
        if r.get("has_conflict") != sub or r["condition"] not in (a, b) or r.get(PRIMARY) is None:
            continue
        k = 0 if r["condition"] == a else 2
        ans = not _unk(r)
        acc[r["id"]][k] += r[PRIMARY] if ans else 0
        acc[r["id"]][k + 1] += ans
    return np.array([acc[i] for i in sorted(acc)]).reshape(-1, 4)


def sel_ratio(X: np.ndarray, n: int = N_BOOT, seed: int = 0):
    """§15.4.5 selective accuracy of one arm (X: correct, answered per question), points, question-bootstrap CI."""
    if not len(X) or X[:, 1].sum() == 0:
        return None
    idx = np.random.default_rng(seed).integers(0, len(X), (n, len(X)))
    c, m = X[idx, 0].sum(1), X[idx, 1].sum(1)
    b = c[m > 0] / m[m > 0]
    return {"est": 100 * float(X[:, 0].sum() / X[:, 1].sum()), "ci": [100 * float(np.quantile(b, 0.025)),
            100 * float(np.quantile(b, 0.975))], "n": int(len(X)), "answered": int(X[:, 1].sum())}


def sel_diff(X: np.ndarray):
    """b - a in the §15.4.5 ratio (revision5.selective_ratio: question bootstrap, within-question swap p), points."""
    from .revision5 import selective_ratio
    t = selective_ratio(X, n_boot=N_BOOT) if len(X) else None
    return None if t is None else {"diff": 100 * t["diff"], "ci": [100 * t["ci_low"], 100 * t["ci_high"]],
                                   "p": t["p"], "n": int(len(X))}


def per_q(rows, cond, sub, f) -> dict:
    """Seed mean within questions of f(row), one condition, conflict (True) or clean (False) questions."""
    acc = defaultdict(list)
    for r in rows:
        if r["condition"] == cond and r.get("has_conflict") == sub and r.get(PRIMARY) is not None:
            v = f(r)
            if v is not None:
                acc[r["id"]].append(float(v))
    return {i: float(np.mean(v)) for i, v in acc.items()}


def mean_ci(d: dict, n: int = N_BOOT, seed: int = 0):
    """Mean over questions (points), with a question-bootstrap percentile 95% CI."""
    if not d:
        return None
    x = np.array([d[i] for i in sorted(d)])
    m = x[np.random.default_rng(seed).integers(0, len(x), (n, len(x)))].mean(1)
    return {"est": 100 * float(x.mean()), "ci": [100 * float(np.quantile(m, 0.025)), 100 * float(np.quantile(m, 0.975))],
            "n": int(len(x))}


def diff_ci(a: dict, b: dict):
    """b - a over shared questions (points), stats.paired_bootstrap (questions as clusters, seed 0)."""
    if not set(a) & set(b):
        return None
    bt = paired_bootstrap(a, b, n_boot=N_BOOT)
    return {"diff": 100 * bt["diff"], "ci": [100 * bt["ci_low"], 100 * bt["ci_high"]], "n": bt["n"]}


def _sub(d: dict, ids) -> dict:
    return {i: d[i] for i in d if i in ids}


def regime(K: float, C: float) -> str:
    """Sign pattern of effect(pi) = pi K + (1 - pi) C on 0 < pi < 1."""
    if K > 0 > C:
        return "crossing"                     # helps above -C/(K - C)
    if K >= 0 and C >= 0:
        return "zero" if K == C == 0 else "helps_all"
    if K <= 0 and C <= 0:
        return "hurts_all"
    return "reverse"                          # K < 0 < C: helps only below C/(C - K)


def breakeven(K, C):
    """Conflict share above which the flag helps: -C/(K - C) where K > 0 > C; 0 where it helps at every share; 1 where
    it helps at none; nan in the reverse and all-zero cases."""
    K, C = np.atleast_1d(np.asarray(K, float)), np.atleast_1d(np.asarray(C, float))
    out = np.full(K.shape, np.nan)
    cross = (K > 0) & (C < 0)
    out[cross] = -C[cross] / (K[cross] - C[cross])
    out[(K >= 0) & (C >= 0) & ((K > 0) | (C > 0))] = 0.0
    out[(K <= 0) & (C <= 0) & ((K < 0) | (C < 0))] = 1.0
    return out


def bca(theta: float, boots, jack, alpha: float = 0.05) -> list:
    """BCa interval (Efron 1987): bias correction from the share of replicates below theta, acceleration from the
    jackknife. nan replicates are dropped; the share is clipped to (0, 1) by one replicate."""
    from scipy.stats import norm
    b = np.asarray(boots, float); b = b[~np.isnan(b)]
    j = np.asarray(jack, float); j = j[~np.isnan(j)]
    p = np.clip(np.mean(b < theta), 1 / (len(b) + 1), len(b) / (len(b) + 1))
    z0 = norm.ppf(p)
    d = j.mean() - j
    s2 = float((d ** 2).sum())
    a = float((d ** 3).sum() / (6 * s2 ** 1.5)) if s2 > 0 else 0.0
    z = norm.ppf([alpha / 2, 1 - alpha / 2])
    q = norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
    return [float(x) for x in np.quantile(b, q)] + [float(z0), a]


def perm_p(x, y, n: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """Spearman permutation p-values over cells (y permuted): two-sided, and one-sided for rho < 0."""
    from scipy.stats import spearmanr
    x, y = np.asarray(x, float), np.asarray(y, float)
    rho = spearmanr(x, y).statistic
    rng = np.random.default_rng(seed)
    null = np.array([spearmanr(x, rng.permutation(y)).statistic for _ in range(n)])
    two = (1 + np.sum(np.abs(null) >= abs(rho) - 1e-12)) / (n + 1)
    less = (1 + np.sum(null <= rho + 1e-12)) / (n + 1)
    return float(two), float(less)


def stance_table(labels: list, items: dict, cut: int = HC_CUT) -> dict:
    """R13: joint judge verdicts (correct page supports gold x misinformation page supports the wrong answer), page
    verdicts by whether the page sits at the character cut, and the share of pages at the cut."""
    V = ("yes", "no", "unclear")
    J = [x for x in labels if x.get("judge")]
    joint = Counter((x["judge"]["correct_supports_gold"], x["judge"]["misinfo_supports_wrong"]) for x in J)
    n = len(J)
    pages, by_cut = [], {"cut": Counter(), "uncut": Counter()}
    for x in J:
        it = items[x["id"]]
        for k, t in enumerate(x["types"]):
            v = x["judge"]["correct_supports_gold" if t == "correct" else "misinfo_supports_wrong"]
            at = len(it["passages"][k]) >= cut
            pages.append((t, v, at)); by_cut["cut" if at else "uncut"][v] += 1
    npg = len(pages)
    at_t = {t: (sum(a for tt, _, a in pages if tt == t), sum(tt == t for tt, _, _ in pages)) for t in ("correct", "misinfo")}
    return {"n_items": len(labels), "n_parsed": n,
            "joint": {c: {m: joint[(c, m)] for m in V} for c in V},
            "joint_share": {c: {m: joint[(c, m)] / n for m in V} for c in V},
            "correct_page": {c: sum(joint[(c, m)] for m in V) / n for c in V},
            "misinfo_page": {m: sum(joint[(c, m)] for c in V) / n for m in V},
            "both_confirmed": joint[("yes", "yes")], "both_confirmed_share": joint[("yes", "yes")] / n,
            "no_confirmed_pair_share": 1 - joint[("yes", "yes")] / n,
            "reversed_pages": sum(v == "no" for _, v, _ in pages), "n_pages": npg,
            "reversed_share": sum(v == "no" for _, v, _ in pages) / npg,
            "at_cut": {"n": sum(a for *_, a in pages), "share": sum(a for *_, a in pages) / npg, "cut_chars": cut,
                       "by_type": {t: {"n": a, "of": m, "share": a / m} for t, (a, m) in at_t.items()}},
            "verdict_by_cut": {k: {v: c[v] for v in V} for k, c in by_cut.items()}}


# --------------------------------------------------------------------------- R13
def r13(R: dict) -> dict:
    if not HC_LABELS.exists() or not HC_EVAL.exists():
        return {"not_run": f"missing {HC_LABELS.name if not HC_LABELS.exists() else HC_EVAL}"}
    items = {o["id"]: o for o in map(json.loads, open(HC_EVAL))}
    out = stance_table([json.loads(l) for l in open(HC_LABELS)], items)
    src = ROOT / "data/external/healthcontradict_src/repo/dataset/dataset_ready.jsonl"
    if src.exists():   # pages longer than the cut in the source (the rest at the cut are exactly 4,000 characters)
        L = {f"hc-{r['instance_id']}": (len(r["doc_a"]), len(r["doc_b"])) for r in map(json.loads, open(src))}
        out["at_cut"]["truncated_in_source"] = sum(x > HC_CUT for i in items for x in L[i])
    st = R["rev"].get("hc19_labels", {})
    out["check_vs_hc19_labels"] = {"correct_supports": out["correct_page"]["yes"] - st.get("correct_supports", float("nan")),
                                   "misinfo_supports_wrong": out["misinfo_page"]["yes"] - st.get("misinfo_supports_wrong", float("nan")),
                                   "both": out["both_confirmed_share"] - st.get("both", float("nan"))}
    out["note"] = "first tabulated on 2026-11-02 (TRIAGE_REVIEW10 B.3); descriptive, does not test label noise"
    return out


# --------------------------------------------------------------------------- R14
def r14() -> dict:
    out = {}
    for g, run in RG_RUNS.items():
        if not results_path(ROOT / "runs" / run):
            out[g] = {"not_run": f"no results in runs/{run}"}; continue
        rows = _rows(run)
        pq = {c: per_question(rows, c, True, PRIMARY) for c in ("plain", DIRECTIONAL, "answer_key")}
        if not pq["answer_key"]:
            out[g] = {"not_run": f"runs/{run} has no answer_key arm"}; continue
        out[g] = {"key_minus_flag": diff_ci(pq[DIRECTIONAL], pq["answer_key"]),
                  "key_minus_plain": diff_ci(pq["plain"], pq["answer_key"]),
                  "flag_minus_plain": diff_ci(pq["plain"], pq[DIRECTIONAL]),
                  "seeds": sorted({r["seed"] for r in rows})}
    return out


# --------------------------------------------------------------------------- R15
def r15(R: dict, S: list) -> dict:
    from .revision9 import _diffs, _joint, pair_audit9
    single, _multi = pair_audit9({})
    stored = R["rev"].get("entity9", {}).get("pooled", {})
    out = {"n_single": len(single)}
    for arm, cond in (("neutral", NEUTRAL), ("directional", DIRECTIONAL)):
        names, ds = [], []
        for s in S:
            rows = _rows(s["run"])
            if s["bench"] == "RAMDocs" and any(r["condition"] == cond for r in rows):
                names.append(s["name"]); ds.append(_diffs(rows, "plain", cond, PRIMARY))
        if len(ds) < 2:
            out[arm] = {"not_run": "fewer than two RAMDocs settings"}; continue
        loo = {n: _joint(ds[:k] + ds[k + 1:], single, np.random.default_rng(0)) for k, n in enumerate(names)}
        est = {n: v["diff"] for n, v in loo.items()}
        out[arm] = {"pooled": _joint(ds, single, np.random.default_rng(0)), "pooled_stored": stored.get(arm, {}).get("single"),
                    "loo": loo, "range": [min(est.values()), max(est.values())],
                    "min_when_dropping": min(est, key=est.get), "max_when_dropping": max(est, key=est.get),
                    "ci_envelope": [min(v["ci"][0] for v in loo.values()), max(v["ci"][1] for v in loo.values())],
                    "n_ci_excludes_0": sum(v["ci"][1] < 0 or v["ci"][0] > 0 for v in loo.values()), "k": len(ds)}
    return out


# --------------------------------------------------------------------------- R16
def _setting_points(s: dict, conds=("plain", NEUTRAL, DIRECTIONAL)) -> dict:
    rows = _rows(s["run"]); ex = _examples(s["run"])
    F = metric_fns({i: e.wrong_answers for i, e in ex.items()})
    have = {r["condition"] for r in rows}
    conds = [c for c in conds if c in have]
    pts, vs = {}, {}
    for sub, name, mets in ((True, "conflict", ("coverage", "selective_perq", "misinfo", "strict")),
                            (False, "clean", ("coverage", "selective_perq", "strict"))):
        Q = {c: {m: per_q(rows, c, sub, F[m]) for m in mets} for c in conds}
        if not Q["plain"]["strict"]:
            continue
        for c in conds:
            pts.setdefault(c, {})[name] = {m: mean_ci(Q[c][m]) for m in mets}
            pts[c][name]["selective"] = sel_ratio(sel_arrays(rows, c, c, sub)[:, :2])
            if c != "plain":
                vs.setdefault(c, {})[name] = {m: diff_ci(Q["plain"][m], Q[c][m]) for m in mets}
                vs[c][name]["selective"] = sel_diff(sel_arrays(rows, "plain", c, sub))
    # consistency of the R12 helper with the stored one-way score: a strict-correct row never contains a wrong answer
    bad = sum(r[PRIMARY] == 1 and F["misinfo"](r) == 1 for r in rows if r["condition"] in conds and r.get("has_conflict"))
    return {"points": pts, "vs_plain": vs, "n_strict_and_misinfo": int(bad)}


def r16(R: dict, S: list) -> dict:
    risk = {k for k, v in R["rev"]["r2"].items() if "risk" in v}
    out, dev = {"settings": {}}, []
    for s in S:
        o = _setting_points(s)
        src = ("§16.4.1" if s["name"] in risk else "§16.4.1 (as RD:gpt-4o-mini)" if s["name"] == "RD:gpt-4o-mini|default"
               else "ConflictBank minimal" if s["name"].startswith("CBn:") else "RAMDocs generality")
        out["settings"][s["name"]] = {**{k: s[k] for k in ("gen", "bench", "policy", "run")}, "source": src, **o}
        dev.append(abs(o["points"]["plain"]["conflict"]["strict"]["est"] / 100 - _stored_plain(R, s["name"])))
    out["n_settings"] = len(S)
    out["check_plain_strict_max_abs_diff"] = float(max(dev))
    from .paper_stats import set_label   # the ratio against the stored H21(c) ratio estimand (CBs / CBf settings)
    sd = []
    for k, t in R["rev"].get("H21c_ratio", {}).items():
        for name in out["settings"]:
            for c, nm in ((NEUTRAL, "neutral flag"), (DIRECTIONAL, "dir. flag")):
                if name.startswith(("CBs:", "CBf:")) and k == f"{set_label(name)}: {nm} vs plain (conflict)":
                    p = out["settings"][name]["points"]
                    sd += [abs(p["plain"]["conflict"]["selective"]["est"] - 100 * t["sel_a"]),
                           abs(p[c]["conflict"]["selective"]["est"] - 100 * t["sel_b"]),
                           abs(out["settings"][name]["vs_plain"][c]["conflict"]["selective"]["diff"] - 100 * t["diff"])]
    out["check_selective_vs_H21c_ratio"] = {"max_abs_diff": float(max(sd)) if sd else None, "n": len(sd)}
    out["n_strict_and_misinfo_rows"] = int(sum(v["n_strict_and_misinfo"] for v in out["settings"].values()))
    out["note"] = ("conflict: coverage (1 - abstention), selective accuracy, one-way misinformation share (R12), one-way "
                   "strict accuracy; clean: the same without misinformation; points with question-bootstrap CIs, "
                   "flag minus plain with paired question-bootstrap CIs; selective = §15.4.5 ratio (sum correct / sum "
                   "answered over question x seed, as H21c_ratio and Table 2), selective_perq = the per-question mean")
    return out


# --------------------------------------------------------------------------- R17
def _mix(rows, cond, wrong) -> dict:
    acc = defaultdict(list)
    for r in rows:
        if r["condition"] == cond and r.get("has_conflict") is True:
            acc[r["id"]].append(outcome_oneway(r, wrong.get(r["id"], [])))
    return {k: 100 * float(np.mean([v.count(k) / len(v) for v in acc.values()])) for k in ("correct", "misinfo", "abstain", "other")}


def r17(R: dict, S: list) -> dict:
    from .revision9 import _joint
    byname = {s["name"]: s for s in S}
    a = {}
    for name in MINIMAL_RD:
        if name not in byname:
            a[name] = {"not_run": "setting missing"}; continue
        s = byname[name]; rows = _rows(s["run"]); ex = _examples(s["run"])
        W = {i: e.wrong_answers for i, e in ex.items()}; F = metric_fns(W)
        o = {"run": s["run"], "gen": s["gen"]}
        for sub, part in ((True, "conflict"), (False, "clean")):
            mets = ("strict", "misinfo", "abstain") if sub else ("strict", "abstain")
            Q = {c: {m: per_q(rows, c, sub, F[m]) for m in mets} for c in ("plain", NEUTRAL)}
            o[part] = {c: {m: mean_ci(Q[c][m]) for m in mets} for c in Q}
            o[part]["flag_minus_plain"] = {m: diff_ci(Q["plain"][m], Q[NEUTRAL][m]) for m in mets}
        o["conflict_mix"] = {c: _mix(rows, c, W) for c in ("plain", NEUTRAL)}
        a[name] = o
    # (b) RAMDocs clean questions: several gold answers vs one
    b = {"settings": {}}
    ds, types = [], None
    for s in S:
        rows = _rows(s["run"])
        if s["bench"] != "RAMDocs" or not any(r["condition"] == LLM_FLAG for r in rows):
            continue
        ex = _examples(s["run"])
        types = {"several": {i for i, e in ex.items() if e.has_conflict is False and len(e.gold_answers) > 1},
                 "one": {i for i, e in ex.items() if e.has_conflict is False and len(e.gold_answers) == 1}}
        fl = per_q(rows, LLM_FLAG, False, lambda r: float((r.get("n_flags") or 0) > 0))
        nf = per_q(rows, LLM_FLAG, False, lambda r: float(r.get("n_flags") or 0))
        pl, fg = per_question(rows, "plain", False, PRIMARY), per_question(rows, LLM_FLAG, False, PRIMARY)
        cfg = _cfg(s["run"])
        o = {"run": s["run"], "detector": cfg.get("detector_model") or cfg["model"]}
        for t, ids in types.items():
            o[t] = {"flag_rate": mean_ci(_sub(fl, ids)), "mean_flags": float(np.mean(list(_sub(nf, ids).values()))),
                    "effect": diff_ci(_sub(pl, ids), _sub(fg, ids))}
        o["gap_several_minus_one"] = o["several"]["effect"]["diff"] - o["one"]["effect"]["diff"]
        b["settings"][s["name"]] = o
        ds.append({i: fg[i] - pl[i] for i in set(pl) & set(fg)})
    if ds:
        b["n"] = {t: len(v) for t, v in types.items()}
        b["pooled"] = {t: _joint(ds, ids, np.random.default_rng(0)) for t, ids in types.items()}
        b["flag_rate_spread"] = {t: [min(v[t]["flag_rate"]["est"] for v in b["settings"].values()),
                                     max(v[t]["flag_rate"]["est"] for v in b["settings"].values())] for t in types}
        b["k"] = len(ds)
    else:
        b = {"not_run": "no RAMDocs setting with llm_dir+flag"}
    return {"a": a, "b": b}


# --------------------------------------------------------------------------- R18
def _pool_means(vec, idx) -> np.ndarray:
    return np.array([np.nanmean(a[idx[p]]) for p, a in vec])


def r18(R: dict, S: list) -> dict:
    from .revision7 import _diff
    from .revision8 import _draw, _vectors
    keys, meta, dc, dn = [], [], [], []
    for s in S:
        rows = _rows(s["run"])
        if not any(r["condition"] == LLM_DIR for r in rows) or not any(r.get("has_conflict") is False for r in rows):
            continue
        c = _diff(rows, "plain", LLM_DIR, PRIMARY)
        ra, rb = per_question(rows, "plain", False, PRIMARY), per_question(rows, LLM_DIR, False, PRIMARY)
        n = {i: rb[i] - ra[i] for i in set(ra) & set(rb)}
        if c and n:
            keys.append(s["name"]); meta.append(s); dc.append(c); dn.append(n)
    if not keys:
        return {"not_run": "no setting with clean questions and llm_dir+flag_directional"}
    qc, vc = _vectors(dc); qn, vn = _vectors(dn)
    K0, C0 = _pool_means(vc, {p: np.arange(len(v)) for p, v in qc.items()}), _pool_means(vn, {p: np.arange(len(v)) for p, v in qn.items()})
    rng = np.random.default_rng(0)
    Kb, Cb = np.empty((N_BOOT, len(keys))), np.empty((N_BOOT, len(keys)))
    for b in range(N_BOOT):   # one draw per pool, shared by every setting (revision8.prevalence8)
        Kb[b] = _pool_means(vc, _draw(rng, qc)); Cb[b] = _pool_means(vn, _draw(rng, qn))
    # seed means of 0/1 scores carry float noise; round so exact zeros classify as zero
    K0, C0, Kb, Cb = (np.round(x, 12) for x in (K0, C0, Kb, Cb))
    pi = np.array(PI)
    q = lambda x: [100 * float(np.quantile(x, 0.025)), 100 * float(np.quantile(x, 0.975))]
    stored = R["rev"].get("prevalence7", {})
    out = {"pi": PI, "settings": {}, "n_boot": N_BOOT, "pools": {p: len(v) for p, v in qc.items()}}
    lo_all, hi_all, est_all, dev = [], [], [], []
    for j, (k, s) in enumerate(zip(keys, meta)):
        E = Kb[:, j, None] * pi + Cb[:, j, None] * (1 - pi)
        est = K0[j] * pi + C0[j] * (1 - pi)
        lo, hi = np.quantile(E, 0.025, axis=0), np.quantile(E, 0.975, axis=0)
        lo_all.append(lo); hi_all.append(hi); est_all.append(est)
        be_b = breakeven(Kb[:, j], Cb[:, j])
        reg = regime(K0[j], C0[j])
        o = {"gen": s["gen"], "bench": s["bench"], "policy": s["policy"], "group": CLEAN_GROUP.get(s["bench"], "other"),
             "K": {"est": 100 * float(K0[j]), "ci": q(Kb[:, j]), "n": int(np.sum(~np.isnan(vc[j][1])))},
             "C": {"est": 100 * float(C0[j]), "ci": q(Cb[:, j]), "n": int(np.sum(~np.isnan(vn[j][1])))},
             "est": [100 * float(x) for x in est], "lo": [100 * float(x) for x in lo], "hi": [100 * float(x) for x in hi],
             "regime": reg}
        regs = Counter(regime(a_, b_) for a_, b_ in zip(Kb[:, j], Cb[:, j]))
        o["regimes_boot"] = {r_: regs[r_] / N_BOOT for r_ in ("crossing", "helps_all", "hurts_all", "reverse", "zero")}
        if reg == "crossing":   # replicates in the reverse / all-zero cases have no break-even and are dropped
            fin = be_b[~np.isnan(be_b)]
            o["breakeven"] = float(breakeven(K0[j], C0[j])[0])
            o["breakeven_ci"] = [float(np.quantile(fin, 0.025)), float(np.quantile(fin, 0.975))]
            o["breakeven_ci_upper_capped"] = bool(o["breakeven_ci"][1] >= 1.0)   # >= 2.5% of replicates help at no share
        sp = stored.get(f"{k}|{LLM_DIR}")
        if sp:
            dev.append(max(abs(100 * K0[j] - sp["conflict"]), abs(100 * C0[j] - sp["clean"])))
        out["settings"][k] = o
    L, H, Es = np.array(lo_all), np.array(hi_all), np.array(est_all)
    grp = np.array([o["group"] for o in out["settings"].values()])

    def count(mask):
        return {"n": int(mask.sum()), "n_neg": [int(x) for x in (Es[mask] < 0).sum(0)],
                "n_sig_neg": [int(x) for x in (H[mask] < 0).sum(0)], "n_sig_pos": [int(x) for x in (L[mask] > 0).sum(0)]}
    out["summary"] = {"all": count(np.ones(len(keys), bool)), **{g: count(grp == g) for g in CLEAN_BUILD if (grp == g).any()}}
    rd = [s for s in meta if s["bench"] == "RAMDocs"]
    groups = {g: {"build": CLEAN_BUILD[g], "settings": [k for k, o in out["settings"].items() if o["group"] == g]}
              for g in CLEAN_BUILD}
    if rd:
        ex = _examples(rd[0]["run"])
        groups["ramdocs_multi"]["n_clean"] = sum(e.has_conflict is False for e in ex.values())
        groups["ramdocs_multi"]["n_clean_several_gold"] = sum(e.has_conflict is False and len(e.gold_answers) > 1 for e in ex.values())
    out["groups"] = groups
    out["check_max_abs_diff_vs_prevalence7"] = float(max(dev)) if dev else None
    out["note"] = ("gpt-4o-mini LLM detector, directional flag; effect(pi) = pi K + (1 - pi) C in points; bands: 2.5-97.5% "
                   "of a joint question bootstrap (conflict and clean questions resampled within each pool, one draw shared "
                   "by every setting); break-even CI over replicates with helps-at-every-share = 0 and helps-at-none = 1")
    return out


# --------------------------------------------------------------------------- R19
def _h4_matrices(metric: str = PRIMARY):
    """revision_stats.h4_revised's dose-response inputs on holdout_contains: per false-flag cell (one representative per
    mode, precision, recall), clean-question exposure and effect over the clean questions (cells x questions)."""
    from .detectors import parse_synth_name
    from .paper_stats import dets_of, examples_of, rows_of
    from .revision_stats import q_doc
    run = ROOT / "runs/holdout_contains"
    rows, examples = rows_of(run), examples_of(run)
    by_key = defaultdict(list)
    for c in sorted({r["condition"] for r in rows if r["condition"].startswith("synth-")}):
        det, _, mode = c.partition("+")
        p, r, _, d = parse_synth_name(det)
        by_key[(mode, p, r)].append(c)
    rep = {k: sorted(v)[-1] for k, v in by_key.items()}
    ex_by = {e.id: e for e in examples if e.has_conflict is False}
    plain = per_question(rows, "plain", False, metric)
    ids = sorted(set(plain) & set(ex_by))
    expo, eff = [], []
    for k, c in sorted((k, c) for k, c in rep.items() if k[1] < 1):
        det = dets_of(run, c.partition("+")[0])
        sc = per_question(rows, c, False, metric)
        expo.append([q_doc(ex_by[i], det[i].pairs, k[0])[0] for i in ids])
        eff.append([sc[i] - plain[i] for i in ids])
    return np.array(expo), np.array(eff), ids


def r19(R: dict) -> dict:
    from scipy.stats import spearmanr
    if not results_path(ROOT / "runs/holdout_contains"):
        return {"not_run": "no results in runs/holdout_contains"}
    expo, eff, ids = _h4_matrices()
    rho_of = lambda cols: spearmanr(expo[:, cols].mean(1), eff[:, cols].mean(1)).statistic
    rho = float(rho_of(np.arange(len(ids))))
    stored = R.get("contains", {}).get("H4", {}).get(PRIMARY, {}).get("dose_response", {})
    rng = np.random.default_rng(0)   # the stored percentile interval: 2,000 question draws, seed 0
    pct = np.nanquantile([rho_of(rng.integers(0, len(ids), len(ids))) for _ in range(2000)], [0.025, 0.975])
    rng = np.random.default_rng(0)
    boots = np.array([rho_of(rng.integers(0, len(ids), len(ids))) for _ in range(N_BOOT)])
    jack = np.array([rho_of(np.delete(np.arange(len(ids)), k)) for k in range(len(ids))])
    lo, hi, z0, a = bca(rho, boots, jack)
    two, less = perm_p(expo.mean(1), eff.mean(1))
    return {"rho": rho, "rho_stored": stored.get("spearman_rho"), "n_cells": int(expo.shape[0]), "n_questions": len(ids),
            "percentile_ci_reproduced": [float(pct[0]), float(pct[1])],
            "percentile_ci_stored": [stored.get("ci_low"), stored.get("ci_high")],
            "p_perm": two, "p_perm_one_sided_negative": less, "n_perm": 10_000,
            "bca_ci": [lo, hi], "bca_z0": z0, "bca_accel": a, "n_boot": N_BOOT,
            "n_boot_nan": int(np.isnan(boots).sum()),
            "percentile_ci_5000": [float(x) for x in np.nanquantile(boots, [0.025, 0.975])]}


# --------------------------------------------------------------------------- entry points
def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if math.isnan(x) or math.isinf(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def compute(R: dict) -> dict:
    S = settings(R)
    return _clean({"R13": r13(R), "R14": r14(), "R15": r15(R, S), "R16": r16(R, S), "R17": r17(R, S),
                   "R18": r18(R, S), "R19": r19(R)})


def _f(x, nd=1):
    return "--" if x is None else f"{x:+.{nd}f}"


def summary(o: dict) -> str:
    L = []
    t = o["R13"]
    if "not_run" not in t:
        L.append(f"R13 HealthContradict: {t['n_parsed']} items; both pages confirmed {t['both_confirmed']} "
                 f"({100 * t['both_confirmed_share']:.1f}%); correct page yes/no/unclear "
                 + "/".join(f"{100 * t['correct_page'][v]:.1f}" for v in ("yes", "no", "unclear"))
                 + "; misinfo page " + "/".join(f"{100 * t['misinfo_page'][v]:.1f}" for v in ("yes", "no", "unclear"))
                 + f"; reversed pages {t['reversed_pages']}/{t['n_pages']}; at cut {t['at_cut']['n']}/{t['n_pages']}")
    for g, v in o["R14"].items():
        if "not_run" in v:
            L.append(f"R14 {g}: {v['not_run']}")
        else:
            k = v["key_minus_flag"]
            L.append(f"R14 {g}: key - oracle dir. flag {_f(k['diff'])} [{_f(k['ci'][0])}, {_f(k['ci'][1])}] (n {k['n']}); "
                     f"key - plain {_f(v['key_minus_plain']['diff'])}, flag - plain {_f(v['flag_minus_plain']['diff'])}")
    for arm in ("neutral", "directional"):
        v = o["R15"][arm]
        p = v["pooled"]
        L.append(f"R15 {arm}: pooled single-entity {_f(p['diff'])} [{_f(p['ci'][0])}, {_f(p['ci'][1])}] (k {v['k']}); "
                 f"leave-one-out {_f(v['range'][0])} to {_f(v['range'][1])}; CIs excluding 0: {v['n_ci_excludes_0']}/{v['k']}")
    r16 = o["R16"]
    L.append(f"R16: {r16['n_settings']} settings; plain check max |diff| {r16['check_plain_strict_max_abs_diff']:.2e}; "
             f"strict-and-misinfo rows {r16['n_strict_and_misinfo_rows']}; selective vs H21c_ratio max |diff| "
             f"{r16['check_selective_vs_H21c_ratio']['max_abs_diff']:.2e} ({r16['check_selective_vs_H21c_ratio']['n']} values)")
    for name, v in r16["settings"].items():
        pts = v["points"]
        row = []
        for c in ("plain", NEUTRAL, DIRECTIONAL):
            if c in pts and "conflict" in pts[c]:
                p = pts[c]["conflict"]
                row.append(f"{({NEUTRAL: 'neutral', DIRECTIONAL: 'dir'}).get(c, c)} cov {p['coverage']['est']:.1f} sel {p['selective']['est']:.1f} "
                           f"mis {p['misinfo']['est']:.1f} acc {p['strict']['est']:.1f}")
        L.append(f"  {name}: " + " | ".join(row))
    for name, v in o["R17"]["a"].items():
        if "not_run" in v:
            continue
        c = v["conflict"]; d = c["flag_minus_plain"]
        L.append(f"R17a {name}: correct {c['plain']['strict']['est']:.1f}->{c[NEUTRAL]['strict']['est']:.1f} "
                 f"({_f(d['strict']['diff'])}), misinfo {c['plain']['misinfo']['est']:.1f}->{c[NEUTRAL]['misinfo']['est']:.1f} "
                 f"({_f(d['misinfo']['diff'])}), abstain {c['plain']['abstain']['est']:.1f}->{c[NEUTRAL]['abstain']['est']:.1f} "
                 f"({_f(d['abstain']['diff'])} [{_f(d['abstain']['ci'][0])}, {_f(d['abstain']['ci'][1])}])")
    b = o["R17"]["b"]
    if "not_run" not in b:
        for t in ("several", "one"):
            p = b["pooled"][t]
            L.append(f"R17b clean {t} gold (n {b['n'][t]}): flag rate {b['flag_rate_spread'][t][0]:.1f}-{b['flag_rate_spread'][t][1]:.1f}%; "
                     f"LLM flag effect pooled {_f(p['diff'])} [{_f(p['ci'][0])}, {_f(p['ci'][1])}] (k {p['k']})")
    r18 = o["R18"]
    if "not_run" not in r18:
        s = r18["summary"]["all"]
        L.append(f"R18: {s['n']} settings; check vs prevalence7 {r18['check_max_abs_diff_vs_prevalence7']:.2e}; "
                 + "; ".join(f"pi {p:.2f}: <0 {s['n_neg'][k]}, sig<0 {s['n_sig_neg'][k]}, sig>0 {s['n_sig_pos'][k]}"
                             for k, p in enumerate(r18["pi"]) if p in (0.01, 0.05, 0.2, 0.5)))
        for g in ("cb_distractors", "ramdocs_multi", "raguard"):
            if g in r18["summary"]:
                s = r18["summary"][g]; k5 = r18["pi"].index(0.05)
                L.append(f"  {g}: {s['n']} settings, at 5% <0 {s['n_neg'][k5]}, sig<0 {s['n_sig_neg'][k5]}")
        for name, v in r18["settings"].items():
            be = (f"break-even {100 * v['breakeven']:.1f}% [{100 * v['breakeven_ci'][0]:.1f}, {100 * v['breakeven_ci'][1]:.1f}]"
                  if "breakeven" in v else v["regime"])
            L.append(f"  {name} [{v['group']}]: K {_f(v['K']['est'])} C {_f(v['C']['est'])}; {be}")
    t = o["R19"]
    if "not_run" not in t:
        L.append(f"R19 H4b (contains): rho {t['rho']:+.3f} (stored {t['rho_stored']:+.3f}); percentile reproduced "
                 f"[{t['percentile_ci_reproduced'][0]:+.3f}, {t['percentile_ci_reproduced'][1]:+.3f}]; permutation p "
                 f"{t['p_perm']:.3f} (one-sided {t['p_perm_one_sided_negative']:.3f}); BCa [{t['bca_ci'][0]:+.3f}, {t['bca_ci'][1]:+.3f}]")
    return "\n".join(L)


def main(argv=None):
    R = json.loads((ROOT / "paper/generated/hypotheses.json").read_text())
    out = compute(R)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True))
    print(summary(out))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
