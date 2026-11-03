"""
Eleventh revision, part 1 (PREREGISTRATION §20.2): artefacts, prior knowledge and the follow-rate account. Exploratory
re-analyses of outputs already seen (registered in PR§20 before they ran); no API call, outside the FDR count.

  R1  passage-only text classifier on ConflictBank (TF-IDF 1-2 grams + L2 logistic), full and object-masked passages
  R2  detector direction accuracy (single-side as registered; §11.3's coding beside) split by the detector's own
      closed-book knowledge (§11.3 rule)
  R3  single-side direction precision beside the current definition ("both" / "unclear" count as misses)
  R4  H31 follow rates split by the generator's closed-book knowledge; pooled logistic with generator fixed effects
  R5  the round-10 retrospective account of the 30 crossed cells, with known / unknown follow rates per question
  R8  completion tokens under the flag in the H31 runs (terra, luna, Haiku)

    python -m experiments.conflict.rev11_prior [--only R2,R4]

R1 reads the 1.3 GB QA file and trains two classifiers (minutes); compute() reuses the R1 block of
runs/rev11/rev11_prior.json when its fingerprint (QA file hash, split ids, parameters) matches.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from .analyze import per_question
from .jsonl import read_results, results_path
from .revision10 import RAND19 as RAND, _examples
from .revision6 import _cell_outcomes

ROOT = Path(__file__).resolve().parents[2]
PRIMARY = "ramdocs_strict_oneway"
N_BOOT = 5000
OUT = ROOT / "runs/rev11/rev11_prior.json"
CB = ROOT / "data/external/conflictbank"
SPLITS = ROOT / "experiments/conflict/splits"
# closed-book runs on the ConflictBank eval set (seeds: mini, luna, haiku 0-2; gpt-4.1, terra, Qwen 0 only)
CLOSED = {"gpt-4o-mini": "cb_single", "gpt-4.1": "cb_single_gpt41", "gpt-6-luna": "cb_single_luna",
          "gpt-6.1-terra": "cb_single_terra", "claude-haiku-5-5": "cb_single_haiku", "Qwen2.5-7B": "cb_single_qwen"}
SHORT = {"gpt-4o-mini": "mini", "gpt-4.1": "four", "gpt-6-luna": "luna", "gpt-6.1-terra": "terra",
         "claude-haiku-5-5": "haiku", "claude-sonnet-5-5": "sonnet", "Qwen2.5-7B": "qwen"}
H31 = {"gpt-4o-mini": "randdir_cb_mini", "gpt-6-luna": "randdir_cb_luna", "claude-haiku-5-5": "randdir_cb_haiku",
       "gpt-6.1-terra": "randdir_cb_terra"}
R1_PARAMS = {"version": 2, "ngram_range": [1, 2], "lowercase": True, "min_df": 2, "sublinear_tf": True, "C": 1.0, "max_iter": 2000,
             "penalty": "l2", "solver": "lbfgs", "mask": "OBJSLOT", "mask_match": "case-insensitive, word-bounded"}


# --------------------------------------------------------------------------- shared helpers
def _q(x):
    x = np.asarray(x, float)
    return [float(np.nanquantile(x, 0.025)), float(np.nanquantile(x, 0.975))] if np.isfinite(x).any() else [None, None]


def _f(x):
    return None if x is None or not np.isfinite(x) else float(x)


def _mean_ci(x, seed=0):
    """Mean and question-bootstrap 95% CI (5,000 resamples)."""
    x = np.asarray(x, float)
    if not len(x):
        return {"rate": None, "ci": [None, None], "n": 0}
    rng = np.random.default_rng(seed)
    return {"rate": float(x.mean()), "ci": _q(x[rng.integers(0, len(x), (N_BOOT, len(x)))].mean(1)), "n": len(x)}


def _wilson(k, n, z=1.96):
    from .revision7 import _wilson as w
    return [float(v) for v in w(k, n, z)]


def known_ids(rows, metric=PRIMARY):
    """§11.3 rule: closed-book one-way strict correct on at least 2 of 3 seeds, or on the one seed where only one exists."""
    by = defaultdict(list)
    for r in rows:
        if r["condition"] == "closedbook":
            by[r["id"]].append(r[metric])
    return {i for i, v in by.items() if sum(v) >= min(2, len(v))}


def _known(gen):
    """(known ids, closed-book seeds, ids with a closed-book answer) from the generator's cb_single* run."""
    rows = [r for r in read_results(ROOT / "runs" / CLOSED[gen]) if r["condition"] == "closedbook"]
    return known_ids(rows), sorted({r["seed"] for r in rows}), {r["id"] for r in rows}


def _dets(run: Path, name="llm_dir"):
    return {r["id"]: r["pairs"] for r in map(json.loads, open(run / f"detections.{name}.jsonl"))}


def gold_flag_counts(e, pairs):
    """Flags on a gold contradiction pair of one question: (dir_right, dir_total) as metrics.direction_accuracy
    (any non-empty `suspects`; "both" is a miss, "unclear" is skipped), (one_right, one_total) over flags naming exactly
    one side, and the verdict mix (misinfo, correct, both, unclear)."""
    t = e.meta["doc_types"]
    c = dict.fromkeys(("dir_right", "dir_total", "one_right", "one_total", "misinfo", "correct", "both", "unclear"), 0)
    for x in pairs:
        if (e.gold_pairs or {}).get((x["i"], x["j"])) != "contradiction":
            continue
        s = x["suspects"]
        if s:
            c["dir_total"] += 1
            c["dir_right"] += sorted(s) == [k for k in (x["i"], x["j"]) if t[k] == "misinfo"]
        if len(s) == 1:
            c["one_total"] += 1; c["one_right"] += t[s[0]] == "misinfo"
        c["unclear" if not s else "both" if len(s) > 1 else "misinfo" if t[s[0]] == "misinfo" else "correct"] += 1
    return c


def _ratio_split(num, den, grp, seed=0):
    """Pooled ratio sum(num)/sum(den) in groups True / False and their difference, with a joint question bootstrap."""
    num, den, grp = (np.asarray(v, float) for v in (num, den, grp))
    rat = lambda m: num[m].sum() / den[m].sum() if den[m].sum() else np.nan
    est = {"known": rat(grp == 1), "unknown": rat(grp == 0)}
    rng = np.random.default_rng(seed)
    bk, bu = [], []
    for _ in range(N_BOOT):
        ix = rng.integers(0, len(num), len(num)); g = grp[ix]
        n, d = num[ix], den[ix]
        bk.append(n[g == 1].sum() / d[g == 1].sum() if d[g == 1].sum() else np.nan)
        bu.append(n[g == 0].sum() / d[g == 0].sum() if d[g == 0].sum() else np.nan)
    bk, bu = np.array(bk), np.array(bu)
    return {"known": _f(est["known"]), "known_ci": _q(bk), "unknown": _f(est["unknown"]), "unknown_ci": _q(bu),
            "diff": _f(est["known"] - est["unknown"]), "diff_ci": _q(bk - bu),
            "n_known_flags": int(den[grp == 1].sum()), "n_unknown_flags": int(den[grp == 0].sum())}


# --------------------------------------------------------------------------- R1
def mask_object(text: str, obj: str, token: str = R1_PARAMS["mask"]) -> str:
    """Every word-bounded, case-insensitive occurrence of the passage's own object string -> one placeholder token."""
    obj = (obj or "").strip()
    return re.sub(r"(?<!\w)" + re.escape(obj) + r"(?!\w)", f" {token} ", text, flags=re.I) if obj else text


def pairwise_accuracy(p_mis, p_cor):
    """Share of items whose misinformation passage gets the higher "false" probability (ties count 1/2)."""
    p_mis, p_cor = np.asarray(p_mis, float), np.asarray(p_cor, float)
    return (p_mis > p_cor) + 0.5 * (p_mis == p_cor)


def _sha(path: Path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def _r1_fingerprint():
    src = CB / "QA_dataset.json"
    return {"qa_bytes": src.stat().st_size, "qa_sha256": _sha(src),
            "eval_ids": _sha(SPLITS / "conflictbank_eval.txt"), "fresh_ids": _sha(SPLITS / "conflictbank_fresh.txt"),
            "params": R1_PARAMS}


def _load_qa():
    """One pass over QA_dataset.json: the usable rows (prepare_conflictbank's filter) with only subject, relation,
    the two passages and the two object strings."""
    keep = ("question", "object", "replaced_object", "default_evidence", "misinformation_conflict_evidence")
    rows, n_all = [], 0
    with open(CB / "QA_dataset.json") as f:
        for idx, line in enumerate(f):
            r = json.loads(line); n_all += 1
            ok = all(isinstance(r.get(k), str) and r[k].strip() for k in keep)
            if ok and r["object"].strip().lower() != r["replaced_object"].strip().lower():
                rows.append((idx, {k: r[k] for k in ("subject", "relation", "object", "replaced_object",
                                                     "default_evidence", "misinformation_conflict_evidence")}))
            del r
    return rows, n_all


def _eval_items(name):
    out = []
    for o in map(json.loads, open(CB / name)):
        if o["has_conflict"]:
            t = o["types"]
            out.append({"id": o["id"], "cor": o["passages"][t.index("correct")], "mis": o["passages"][t.index("misinfo")],
                        "gold": o["gold"], "wrong": o["wrong"]})
    return out


def r1():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from .prepare_cb_fresh import SEED, pick
    rows, n_all = _load_qa()
    gc.collect()
    slim = [(i, {"subject": r["subject"], "relation": r["relation"]}) for i, r in rows]
    ev_items, ev_pool = pick(slim, 0)
    used = {r["subject"] for _, r in ev_items + ev_pool}
    fr_items, fr_pool = pick(slim, SEED, frozenset(used))
    ids = lambda f: [l.strip() for l in open(SPLITS / f) if l.strip()]
    assert [f"cb-{i}" for i, _ in ev_items] == ids("conflictbank_eval.txt"), "eval selection not reproduced"
    assert [f"cb-{i}" for i, _ in fr_items] == ids("conflictbank_fresh.txt"), "fresh selection not reproduced"
    excl = used | {r["subject"] for _, r in fr_items + fr_pool}
    train = [r for _, r in rows if r["subject"] not in excl]
    assert not {r["subject"] for r in train} & excl
    n_usable = len(rows); del rows, slim; gc.collect()
    sets = {"eval": _eval_items("conflictbank_eval.jsonl"), "fresh": _eval_items("conflictbank_fresh.jsonl")}
    eval_text = {h for v in sets.values() for it in v for h in (hash(it["cor"]), hash(it["mis"]))}
    out = {"n_qa_rows": n_all, "n_usable": n_usable, "n_excluded_subjects": len(excl), "n_train_rows": len(train),
           "n_train_passages": 2 * len(train),
           "n_train_passages_identical_to_eval": sum(hash(r[k]) in eval_text for r in train
                                                     for k in ("default_evidence", "misinformation_conflict_evidence"))}
    y = np.r_[np.zeros(len(train)), np.ones(len(train))]   # 1 = misinformation ("false")
    for var in ("full", "masked"):
        if var == "full":
            X_txt = [r["default_evidence"] for r in train] + [r["misinformation_conflict_evidence"] for r in train]
        else:
            X_txt = [mask_object(r["default_evidence"], r["object"]) for r in train] + \
                    [mask_object(r["misinformation_conflict_evidence"], r["replaced_object"]) for r in train]
        vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        X = vec.fit_transform(X_txt); del X_txt; gc.collect()
        clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs").fit(X, y)
        coef, names = clf.coef_[0], vec.get_feature_names_out()
        o = {"n_features": int(X.shape[1]), "n_iter": int(clf.n_iter_[0]), "train_acc": float(clf.score(X, y)),
             "top_false": [[str(names[j]), float(coef[j])] for j in np.argsort(-coef)[:15]],
             "top_correct": [[str(names[j]), float(coef[j])] for j in np.argsort(coef)[:15]]}
        XT = X.T.tocsr(); del X; gc.collect()
        for s, items in sets.items():
            prep = (lambda t, _o: t) if var == "full" else mask_object
            Ec = vec.transform([prep(it["cor"], it["gold"]) for it in items])
            Em = vec.transform([prep(it["mis"], it["wrong"]) for it in items])
            pc, pm = clf.predict_proba(Ec)[:, 1], clf.predict_proba(Em)[:, 1]
            acc = pairwise_accuracy(pm, pc)
            # leakage check: each eval passage's highest TF-IDF cosine to any training passage (rows are L2-normalised)
            nn = np.concatenate([(E[a:a + 50] @ XT).max(1).toarray().ravel() for E in (Ec, Em) for a in range(0, E.shape[0], 50)])
            o[s] = {**{k: v for k, v in _mean_ci(acc).items() if k != "rate"}, "pairwise_acc": float(acc.mean()),
                    "n_ties": int(np.sum(pm == pc)),
                    "pointwise_acc": float(np.mean(np.r_[pc < 0.5, pm >= 0.5])),
                    "mean_p_false_correct": float(pc.mean()), "mean_p_false_misinfo": float(pm.mean()),
                    "nn_cosine_median": float(np.median(nn)), "nn_cosine_max": float(nn.max()),
                    "n_nn_cosine_ge_0.8": int(np.sum(nn >= 0.8))}
        out[var] = o
        del vec, clf, XT; gc.collect()
    return out


def _r1_cached():
    if not (CB / "QA_dataset.json").exists():
        return {"not_run": "data/external/conflictbank/QA_dataset.json is missing"}
    fp = _r1_fingerprint()
    try:
        old = json.loads(OUT.read_text()).get("R1", {})
        if old.get("fingerprint") == json.loads(json.dumps(fp)):
            return old
    except (OSError, ValueError):
        pass
    return {**r1(), "fingerprint": fp}


# --------------------------------------------------------------------------- crossed cells (H24's per-question data)
def crossed_cells():
    """The 30 H24 cells (paper_stats.crossed on cb_single / cross_cb): {(gen, det): vectors}. gpt-4o-mini-detector cells
    use the generator's own cb_single* run and its detections (§15.4.1f); the others the cross_cb_{g}_{d} run."""
    ex = {i: e for i, e in _examples(ROOT / "runs/cb_single").items() if e.has_conflict}
    contra = {i: {p: {k for k in p if e.meta["doc_types"][k] == "misinfo"} for p, lab in e.gold_pairs.items()
                  if lab == "contradiction"} for i, e in ex.items()}
    correct = {i: {k for k, t in enumerate(e.meta["doc_types"]) if t == "correct"} for i, e in ex.items()}
    runs = {(g, "gpt-4o-mini"): (ROOT / "runs" / CLOSED[g], ROOT / "runs" / CLOSED[g]) for g in CLOSED}
    dets = {"gpt-4.1", "gpt-6-luna", "gpt-6.1-terra", "claude-haiku-5-5", "claude-sonnet-5-5"}
    for g in CLOSED:
        for d in dets:
            run = ROOT / f"runs/cross_cb_{SHORT[g]}_{SHORT[d]}"
            if results_path(run):
                runs[(g, d)] = (run, run)
    out = {}
    for (g, d), (run, drun) in sorted(runs.items()):
        if not (drun / "detections.llm_dir.jsonl").exists():
            drun = ROOT / "runs/cb_main"
        det = _dets(drun)
        rows = read_results(run)
        gp = per_question(rows, "plain", True, PRIMARY); gf = per_question(rows, "llm_dir+flag_directional", True, PRIMARY)
        ids = sorted(i for i in ex if i in det and i in gp and i in gf)
        if not ids:
            continue
        cnt = {i: gold_flag_counts(ex[i], det[i]) for i in ids}
        out[(g, d)] = {
            "ids": ids, "g": np.array([gp[i] for i in ids]), "y1": np.array([gf[i] for i in ids]),
            "d": np.array([float(any(contra[i].get((x["i"], x["j"])) == set(x["suspects"]) for x in det[i])) for i in ids]),
            "r": np.array([float(any((x["i"], x["j"]) in contra[i] for x in det[i])) for i in ids]),
            "w": np.array([float(bool({k for x in det[i] for k in x["suspects"]} & correct[i])) for i in ids]),
            "cnt": cnt}
    return out


# --------------------------------------------------------------------------- R2
def r2(R):
    ex = {i: e for i, e in _examples(ROOT / "runs/cb_single").items() if e.has_conflict}
    ids = sorted(ex)
    out = {}
    for d, run in (("gpt-4o-mini", "cb_main"), ("gpt-4.1", "cross_cb_mini_four"), ("gpt-6-luna", "cross_cb_mini_luna"),
                   ("gpt-6.1-terra", "cross_cb_mini_terra"), ("claude-haiku-5-5", "cross_cb_mini_haiku")):
        det = _dets(ROOT / "runs" / run)
        kn, seeds, have = _known(d)
        assert set(ids) <= have
        c = [gold_flag_counts(ex[i], det.get(i, [])) for i in ids]
        grp = [i in kn for i in ids]
        o = {"detections": run, "closedbook_run": CLOSED[d], "closedbook_seeds": seeds, "n_conflict": len(ids),
             "n_known": int(sum(grp)), "n_unknown": int(len(ids) - sum(grp)),
             "direction": _ratio_split([x["dir_right"] for x in c], [x["dir_total"] for x in c], grp),
             "single_side": _ratio_split([x["one_right"] for x in c], [x["one_total"] for x in c], grp),
             "max_gold_flags_per_question": int(max(x["misinfo"] + x["correct"] + x["both"] + x["unclear"] for x in c))}
        if d == "gpt-4o-mini":   # §11.3, already computed; recomputed here only to add CIs
            cbk = R["rev"].get("closedbook", {}).get("ConflictBank", {})
            o["registered_11_3"] = {g: {"n": cbk[g]["n"], "direction": cbk[g]["direction"]} for g in ("known", "unknown")
                                    if g in cbk}
        out[d] = o
    out["registered"] = "single_side"
    out["note"] = ("single_side is PR§20.2's R2 estimand (questions whose flag on the gold pair names one side; at most "
                   "one such flag per question, so flag and question counts agree); direction = metrics.direction_accuracy "
                   "as §11.3 coded it ('both' a miss, 'unclear' skipped), reported beside it. The two coincide for "
                   "gpt-4o-mini, which never answers 'both'. Sonnet: no closed-book run. The HealthContradict part waits "
                   "for the PR§20.3 human labels.")
    return out


# --------------------------------------------------------------------------- R3
def r3(R, cells):
    """Pooled over each detector's crossed cells, as detector_precision7 (the current definition: among question-cells
    whose flags include the gold pair, the share naming exactly the misinformation side)."""
    dp7 = R["rev"].get("detector_precision7", {})
    by = defaultdict(list)
    for (g, d), v in cells.items():
        by[d].append(v)
    out = {}
    for d, vs in sorted(by.items()):
        k, n = sum(float(v["d"].sum()) for v in vs), sum(float(v["r"].sum()) for v in vs)
        one = [(c["one_right"], c["one_total"]) for v in vs for c in (v["cnt"][i] for i in v["ids"])]
        mix = {m: sum(v["cnt"][i][m] for v in vs for i in v["ids"]) for m in ("misinfo", "correct", "both", "unclear")}
        # question-cluster bootstrap: one draw over the shared questions, applied to every cell of the detector
        qids = sorted({i for v in vs for i in v["ids"]}); pos = {q: j for j, q in enumerate(qids)}
        A = np.zeros((4, len(qids)))   # d, r, one_right, one_total summed over cells per question
        for v in vs:
            for j, i in enumerate(v["ids"]):
                A[:, pos[i]] += [v["d"][j], v["r"][j], v["cnt"][i]["one_right"], v["cnt"][i]["one_total"]]
        rng = np.random.default_rng(0)
        bc, bo = [], []
        for _ in range(N_BOOT):
            s = A[:, rng.integers(0, len(qids), len(qids))].sum(1)
            bc.append(s[0] / s[1] if s[1] else np.nan); bo.append(s[2] / s[3] if s[3] else np.nan)
        k1, n1 = sum(a for a, _ in one), sum(b for _, b in one)
        out[d] = {"n_cells": len(vs), "current": k / n, "current_k": int(k), "current_n": int(n),
                  "current_wilson": _wilson(k, n), "current_ci": _q(bc),
                  "single_side": k1 / n1 if n1 else None, "single_k": int(k1), "single_n": int(n1),
                  "single_wilson": _wilson(k1, n1), "single_ci": _q(bo), "verdict_mix": mix,
                  "matches_detector_precision7": bool(np.isclose(k / n, dp7.get(d, np.nan))),
                  "max_gold_flags_per_question": int(max(c["misinfo"] + c["correct"] + c["both"] + c["unclear"]
                                                         for v in vs for c in v["cnt"].values()))}
    out["note"] = "single-side precision first tabulated on 2026-11-02 while writing the triage (disclosed in PR§20.2)"
    return out


# --------------------------------------------------------------------------- R4
def _logit(X, y, iters=100, tol=1e-9):
    """Logistic MLE (fractional y allowed) by Newton-Raphson; returns (beta, converged)."""
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ b, -35, 35)))
        H = X.T @ (X * (p * (1 - p))[:, None]); g = X.T @ (y - p)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            return b, False
        b = b + step
        if np.max(np.abs(step)) < tol:
            return b, True
    return b, False


def logit_known(obs, seed=0):
    """obs: list of (qid, gen, known, followed share). Log-odds ratio of following for known questions, generator fixed
    effects; 95% CI by question bootstrap (all of a question's generator rows move together). A generator whose
    outcomes are all 0 or all 1 carries no information on the known effect and is dropped from that fit; a fit that is
    still separated on `known` gives an infinite log-OR (reported as an unbounded CI end)."""
    def fit(ob):
        gens = sorted({o[1] for o in ob})
        y_by = {g: [o[3] for o in ob if o[1] == g] for g in gens}
        gens = [g for g in gens if 0 < np.mean(y_by[g]) < 1]
        ob = [o for o in ob if o[1] in gens]
        k = np.array([float(o[2]) for o in ob])
        if not len(ob) or k.sum() == 0 or k.sum() == len(k):
            return np.nan, True
        X = np.array([[1.0, float(o[2])] + [float(o[1] == g) for g in gens[1:]] for o in ob])
        b, ok = _logit(X, np.array([o[3] for o in ob], float))
        return (b[1] if ok else np.sign(b[1]) * np.inf), ok
    b0, ok0 = fit(obs)
    byq = defaultdict(list)
    for o in obs:
        byq[o[0]].append(o)
    qs = sorted(byq); rng = np.random.default_rng(seed)
    bs, bad = [], 0
    for _ in range(N_BOOT):
        b, ok = fit([o for j in rng.integers(0, len(qs), len(qs)) for o in byq[qs[j]]])
        bs.append(b); bad += not ok
    bs = np.array(bs); bs = bs[~np.isnan(bs)]
    meth = "inverted_cdf" if np.isinf(bs).any() else "linear"   # no interpolation across an infinite end
    lo, hi = (float(np.quantile(bs, q, method=meth)) for q in (0.025, 0.975))
    ex = lambda v: None if not np.isfinite(v) else float(np.exp(v))
    return {"log_or": _f(b0), "or": ex(b0) if np.isfinite(b0) else None, "or_ci": [ex(lo), ex(hi)],
            "ci_unbounded": [bool(np.isinf(lo)), bool(np.isinf(hi))], "converged": bool(ok0),
            "n_boot_separated": int(bad), "n_boot_undefined": int(N_BOOT - len(bs)), "n_obs": len(obs),
            "n_questions": len(qs), "n_known_obs": int(sum(o[2] for o in obs))}


def follow_split(gen):
    """revision10._follow's per-question follow outcomes on G's H31 run, with G's known set."""
    run = ROOT / "runs" / H31[gen]
    ex = {i: e for i, e in _examples(run).items() if e.has_conflict}
    det = _dets(run, RAND.split("+")[0])
    rows = read_results(run)
    right = {i: all(ex[i].meta["doc_types"][k] == "misinfo" for x in det[i] for k in x["suspects"]) for i in ex if i in det}
    pl, fl = per_question(rows, "plain", True, PRIMARY), per_question(rows, RAND, True, PRIMARY)
    ids = sorted(set(pl) & set(fl) & set(right))
    oc = _cell_outcomes(rows, RAND, PRIMARY)
    kn, seeds, have = _known(gen)
    assert set(ex) <= have, f"{gen}: conflict questions without a closed-book answer"
    fr = [(i, i in kn, oc[i][0]) for i in ids if right[i] and pl[i] == 0 and i in oc]
    fw = [(i, i in kn, oc[i][1]) for i in ids if not right[i] and pl[i] == 1 and i in oc]
    return {"fr": fr, "fw": fw, "known": kn, "seeds": seeds, "n_known_all": sum(i in kn for i in ex), "n_all": len(ex)}


def _split_rates(lst):
    x = np.array([v for _, _, v in lst]); k = np.array([kk for _, kk, _ in lst], bool)
    o = {"all": _mean_ci(x), "known": _mean_ci(x[k]), "unknown": _mean_ci(x[~k])}
    rng = np.random.default_rng(0); dd = []
    for _ in range(N_BOOT):
        ix = rng.integers(0, len(x), len(x)); kk = k[ix]
        dd.append(x[ix][kk].mean() - x[ix][~kk].mean() if kk.any() and (~kk).any() else np.nan)
    o["diff"] = _f(x[k].mean() - x[~k].mean()) if k.any() and (~k).any() else None
    o["diff_ci"] = _q(dd)
    return o


def r4(R):
    out, F = {}, {}
    for g in H31:
        if not results_path(ROOT / "runs" / H31[g]):
            continue
        s = F[g] = follow_split(g)
        h = R["rev"].get("H31x", {}).get(g, {})
        out[g] = {"run": H31[g], "closedbook_run": CLOSED[g], "closedbook_seeds": s["seeds"],
                  "n_known": s["n_known_all"], "n_conflict": s["n_all"],
                  "f_R": _split_rates(s["fr"]), "f_W": _split_rates(s["fw"]),
                  "matches_H31x": bool(np.isclose(np.mean([v for *_, v in s["fr"]]), h.get("follow_right", np.nan))
                                       and np.isclose(np.mean([v for *_, v in s["fw"]]), h.get("follow_wrong", np.nan)))}
    pooled = {}
    for arm, key in (("f_R", "fr"), ("f_W", "fw")):
        pooled[arm] = logit_known([(i, g, k, v) for g, s in F.items() for i, k, v in s[key]])
    out["pooled_logit"] = pooled
    out["note"] = ("followed = per-question share of flagged seeds following the flag (revision10._follow); logistic "
                   "with fractional outcomes, one observation per question and generator, generator fixed effects")
    return out, F


# --------------------------------------------------------------------------- R5
def cond_pred(D, G, W, K, rates):
    """Per-question account with follow rates chosen by the generator's knowledge K (1 known, 0 unknown).
    rates = (fR_known, fR_unknown, fW_known, fW_unknown)."""
    D, G, W, K = (np.asarray(v, float) for v in (D, G, W, K))
    frk, fru, fwk, fwu = rates
    fr = np.where(K == 1, frk, fru); fw = np.where(K == 1, fwk, fwu)
    return D * (1 - G) * fr - W * G * fw


def r5(R, cells, F):
    from .revision7 import _fit_stats
    if len(F) < 4:
        return {"not_run": "not all four H31 runs are present"}
    rates, fallback = {}, []
    for g, s in F.items():
        allr = (np.mean([v for *_, v in s["fr"]]), np.mean([v for *_, v in s["fw"]]))
        def grp(lst, k, a):
            x = [v for _, kk, v in lst if kk == k]
            if not x:
                fallback.append(f"{g}:{'known' if k else 'unknown'}")
            return float(np.mean(x)) if x else float(a)
        rates[g] = (grp(s["fr"], True, allr[0]), grp(s["fr"], False, allr[0]),
                    grp(s["fw"], True, allr[1]), grp(s["fw"], False, allr[1]))
    uncond = {g: (np.mean([v for *_, v in s["fr"]]), np.mean([v for *_, v in s["fw"]])) for g, s in F.items()}
    mean_u = tuple(float(np.mean([u[j] for u in uncond.values()])) for j in (0, 1))
    mean_c = tuple(float(np.mean([r[j] for r in rates.values()])) for j in range(4))
    known = {g: _known(g)[0] for g in CLOSED}
    rows_u, rows_c = [], []
    for (g, d), v in sorted(cells.items()):
        own = g in rates
        fr, fw = uncond[g] if own else mean_u
        K = np.array([i in known[g] for i in v["ids"]], float)
        pu = np.mean(v["d"] * (1 - v["g"]) * fr - v["w"] * v["g"] * fw)
        pc = np.mean(cond_pred(v["d"], v["g"], v["w"], K, rates[g] if own else mean_c))
        base = {"gen": g, "det": d, "obs": 100 * float(np.mean(v["y1"] - v["g"])), "kind": "own" if own else "transfer",
                "known_share": float(K.mean())}
        rows_u.append({**base, "pred": 100 * float(pu)}); rows_c.append({**base, "pred": 100 * float(pc)})
    old = {(c["gen"], c["det"]): c for c in R["rev"].get("predict7", {}).get("cells", [])}
    drift = max((abs(r["pred"] - old[(r["gen"], r["det"])]["pred"]) + abs(r["obs"] - old[(r["gen"], r["det"])]["obs"])
                 for r in rows_u if (r["gen"], r["det"]) in old), default=None)
    def fits(rows):
        own = [r for r in rows if r["kind"] == "own"]; tr = [r for r in rows if r["kind"] == "transfer"]
        return {"own": _fit_stats(own), "transfer": _fit_stats(tr) if len(tr) >= 3 else None, "all": _fit_stats(rows)}
    fu, fc = fits(rows_u), fits(rows_c)
    return {"n_cells": len(rows_u), "unconditioned": fu, "conditioned": fc,
            "mae_change": {k: fc[k]["mae"] - fu[k]["mae"] for k in fu if fu[k] and fc[k]},
            "cells": [{**u, "pred_unconditioned": u["pred"], "pred_conditioned": c["pred"]} for u, c in zip(rows_u, rows_c)],
            "rates": {g: dict(zip(("fR_known", "fR_unknown", "fW_known", "fW_unknown"), r)) for g, r in rates.items()},
            "transfer_rates": dict(zip(("fR_known", "fR_unknown", "fW_known", "fW_unknown"), mean_c)),
            "fallback_to_overall": fallback, "max_drift_vs_predict7": _f(drift),
            "max_abs_pred_change": float(max(abs(u["pred"] - c["pred"]) for u, c in zip(rows_u, rows_c)))}


# --------------------------------------------------------------------------- R8
def _paired(a, b, seed=0):
    ids = sorted(set(a) & set(b)); d = np.array([b[i] - a[i] for i in ids])
    rng = np.random.default_rng(seed)
    return {"plain": float(np.mean([a[i] for i in ids])), "flagged": float(np.mean([b[i] for i in ids])),
            "diff": float(d.mean()), "diff_ci": _q(d[rng.integers(0, len(d), (N_BOOT, len(d)))].mean(1)), "n": len(ids)}


def r8():
    out = {}
    for g in ("gpt-6.1-terra", "gpt-6-luna", "claude-haiku-5-5"):
        run = ROOT / "runs" / H31[g]
        if not results_path(run):
            out[g] = {"not_run": f"{H31[g]} missing"}; continue
        ex = {i: e for i, e in _examples(run).items() if e.has_conflict}
        det = _dets(run, RAND.split("+")[0])
        right = {i: all(ex[i].meta["doc_types"][k] == "misinfo" for x in det[i] for k in x["suspects"]) for i in ex if i in det}
        rows = [r for r in read_results(run) if r.get("has_conflict") and r.get("completion_tokens") is not None]
        tp, tf = per_question(rows, "plain", True, "completion_tokens"), per_question(rows, RAND, True, "completion_tokens")
        pl = per_question(rows, "plain", True, PRIMARY)
        o = {"run": H31[g], "seeds": sorted({r["seed"] for r in rows}),
             "all": _paired(tp, tf),
             "right_arm": _paired({i: v for i, v in tp.items() if right.get(i)}, tf),
             "wrong_arm": _paired({i: v for i, v in tp.items() if i in right and not right[i]}, tf)}
        for arm, elig, fol in (("right_followed", lambda i: right.get(i) and pl.get(i) == 0, lambda r: r[PRIMARY] == 1),
                               ("wrong_followed", lambda i: i in right and not right[i] and pl.get(i) == 1,
                                lambda r: r["wrong_rate"] > 0)):
            fr_ = [r for r in rows if r["condition"] == RAND and elig(r["id"])]
            byq = defaultdict(list)
            for r in fr_:
                byq[r["id"]].append((float(fol(r)), r["completion_tokens"]))
            qs = sorted(byq)
            def stat(sel):
                a = [t for q in sel for f, t in byq[q] if f == 1]; b = [t for q in sel for f, t in byq[q] if f == 0]
                return (np.mean(a) if a else np.nan, np.mean(b) if b else np.nan, len(a), len(b))
            fy, fn, na, nb = stat(qs)
            rng = np.random.default_rng(0)
            bd = [(lambda s: s[0] - s[1])(stat([qs[j] for j in rng.integers(0, len(qs), len(qs))])) for _ in range(N_BOOT)]
            o[arm] = {"followed": _f(fy), "not_followed": _f(fn), "n_rows_followed": na, "n_rows_not": nb,
                      "diff": _f(fy - fn), "diff_ci": _q(bd), "n_questions": len(qs),
                      "plain_same_questions": float(np.mean([tp[q] for q in qs if q in tp])) if qs else None}
        out[g] = o
    out["note"] = "API completion_tokens, conflict questions, seed means within questions; for luna, terra and Claude they include hidden reasoning tokens"
    return out


# --------------------------------------------------------------------------- entry points
def compute(R: dict, only=None) -> dict:
    want = lambda k: only is None or k in only
    out = {}
    if want("R1"):
        out["R1"] = _r1_cached()
    cells = crossed_cells() if want("R3") or want("R5") else None
    if want("R2"):
        out["R2"] = r2(R)
    if want("R3"):
        out["R3"] = r3(R, cells)
    if want("R4") or want("R5"):
        r4o, F = r4(R)
        if want("R4"):
            out["R4"] = r4o
        if want("R5"):
            out["R5"] = r5(R, cells, F)
    if want("R8"):
        out["R8"] = r8()
    return json.loads(json.dumps(out, default=float))


def _p(x, k=3):
    return "NA" if x is None else f"{x:.{k}f}"


def _ci(c, k=3):
    return f"[{_p(c[0], k)}, {_p(c[1], k)}]"


def summary(o: dict) -> str:
    L = []
    if "R1" in o:
        a = o["R1"]
        if "not_run" in a:
            L.append(f"R1 not run: {a['not_run']}")
        else:
            L.append(f"R1 train rows {a['n_train_rows']} (excluded subjects {a['n_excluded_subjects']}, "
                     f"identical-to-eval passages {a['n_train_passages_identical_to_eval']})")
            for v in ("full", "masked"):
                L.append(f"  {v:6s} " + "  ".join(f"{s} pairwise {a[v][s]['pairwise_acc']:.3f} {_ci(a[v][s]['ci'])} (n {a[v][s]['n']})"
                                                  for s in ("eval", "fresh")))
    if "R2" in o:
        L.append("R2 direction accuracy known / unknown / diff: single-side (registered), then §11.3 coding ('both' = miss):")
        for d, v in o["R2"].items():
            if d in ("note", "registered"):
                continue
            for lab, key in (("single", "single_side"), ("§11.3 ", "direction")):
                x = v[key]
                L.append(f"  {d:17s} {lab} seeds {v['closedbook_seeds']} known {v['n_known']:3d}: {_p(x['known'])} "
                         f"{_ci(x['known_ci'])} ({x['n_known_flags']} flags)  unknown {_p(x['unknown'])} "
                         f"{_ci(x['unknown_ci'])} ({x['n_unknown_flags']})  diff {_p(x['diff'])} {_ci(x['diff_ci'])}")
    if "R3" in o:
        L.append("R3 direction precision, current (both/unclear = miss) vs single-side:")
        for d, v in o["R3"].items():
            if d == "note":
                continue
            L.append(f"  {d:17s} cells {v['n_cells']:2d} current {v['current']:.3f} ({v['current_k']}/{v['current_n']})"
                     f"  single {_p(v['single_side'])} ({v['single_k']}/{v['single_n']}) {_ci(v['single_ci'])}  mix {v['verdict_mix']}")
    if "R4" in o:
        L.append("R4 follow rates known / unknown (n):")
        for g, v in o["R4"].items():
            if g in ("pooled_logit", "note"):
                continue
            s = lambda a: f"{_p(v[a]['known']['rate'])} ({v[a]['known']['n']}) / {_p(v[a]['unknown']['rate'])} ({v[a]['unknown']['n']})"
            L.append(f"  {g:17s} known {v['n_known']}/{v['n_conflict']}  f_R {s('f_R')}  f_W {s('f_W')}  H31x match {v['matches_H31x']}")
        for arm, v in o["R4"]["pooled_logit"].items():
            L.append(f"  pooled {arm}: OR {_p(v['or'])} {_ci(v['or_ci'])} (known obs {v['n_known_obs']}/{v['n_obs']}, "
                     f"separated boots {v['n_boot_separated']}, unbounded {v['ci_unbounded']})")
    if "R5" in o and "not_run" not in o["R5"]:
        v = o["R5"]
        L.append(f"R5 {v['n_cells']} cells, drift vs predict7 {_p(v['max_drift_vs_predict7'], 6)}, max |pred change| {v['max_abs_pred_change']:.2f}")
        for k in ("own", "transfer", "all"):
            u, c = v["unconditioned"][k], v["conditioned"][k]
            if u:
                L.append(f"  {k:8s} MAE {u['mae']:.3f} -> {c['mae']:.3f}  slope {u['slope']:.3f} -> {c['slope']:.3f}"
                         f"  r {u['r']:.3f} -> {c['r']:.3f}  (n {u['n']})")
    if "R8" in o:
        L.append("R8 completion tokens plain -> flagged (diff):")
        for g, v in o["R8"].items():
            if g == "note" or "not_run" in v:
                continue
            L.append(f"  {g:17s} all {v['all']['plain']:.0f} -> {v['all']['flagged']:.0f} ({v['all']['diff']:+.0f} {_ci(v['all']['diff_ci'], 0)})"
                     f"  right {v['right_arm']['diff']:+.0f}  wrong {v['wrong_arm']['diff']:+.0f}"
                     f"  | followed vs not: R {_p(v['right_followed']['followed'], 0)} vs {_p(v['right_followed']['not_followed'], 0)}"
                     f", W {_p(v['wrong_followed']['followed'], 0)} vs {_p(v['wrong_followed']['not_followed'], 0)}")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma-separated IDs, e.g. R2,R4 (default: all)")
    args = ap.parse_args(argv)
    R = json.loads((ROOT / "paper/generated/hypotheses.json").read_text())
    only = set(args.only.split(",")) if args.only else None
    out = compute(R, only)
    if only and OUT.exists():   # keep the other IDs already on disk
        out = {**json.loads(OUT.read_text()), **out}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True))
    print(summary(out))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
