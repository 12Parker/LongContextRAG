"""
Ninth-review re-analyses (PREREGISTRATION §18; exploratory, existing outputs only, outside the FDR count).

  pair_audit9   §18.1 RAMDocs pair rule: multi-entity questions and a lower bound on cross-entity oracle pairs
  entity9       §18.2 oracle neutral / directional flag minus plain on single- vs multi-entity conflict questions
  agree9        §18.3 per-question discordance between the flag and the key (H30a, H23), against plain seed-to-seed
"""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations

import numpy as np

from .analyze import per_question
from .jsonl import results_path
from .revision5 import SHORT_GEN

N_BOOT9 = 5000


def _ps():
    from . import paper_stats as ps
    return ps


def _holdout():
    from .data import load_ramdocs
    ps = _ps()
    ids = set((ps.ROOT / "experiments/conflict/splits/ramdocs_holdout.txt").read_text().split())
    return [e for e in load_ramdocs(label_fix="exclude") if e.id in ids]


def _entities(e):
    t, a = e.meta["doc_types"], [x.strip().lower() for x in e.meta["doc_answers"]]
    return Counter(a[k] for k in range(len(t)) if t[k] == "correct"), t, a


def pair_audit9(rev):
    ex = _holdout()
    conf = [e for e in ex if e.has_conflict]
    n_pairs = lb = n_multi = n_mc = 0
    single = []
    for e in conf:
        ents, t, a = _entities(e)
        contra = [p for p, v in e.gold_pairs.items() if v == "contradiction"]
        n_pairs += len(contra)
        if len(ents) < 2:
            single.append(e.id)
            continue
        n_multi += 1
        for m in [k for k in range(len(t)) if t[k] == "misinfo"]:
            part = Counter(a[k] for (i, j) in contra if m in (i, j) for k in (i, j) if k != m and t[k] == "correct")
            n_mc += sum(part.values())
            lb += sum(part.values()) - (max(part.values()) if part else 0)
    rev["pair_audit9"] = {"n_q": len(ex), "n_conflict": len(conf), "n_multi": n_multi, "n_single": len(single),
                          "n_pairs": n_pairs, "n_misinfo_correct_multi": n_mc, "cross_lb": lb, "cross_lb_share": lb / n_pairs}
    return set(single), {e.id for e in conf} - set(single)


def _diffs(rows, a, b, metric):
    ra, rb = per_question(rows, a, True, metric), per_question(rows, b, True, metric)
    return {i: rb[i] - ra[i] for i in set(ra) & set(rb)}


def _mean_ci(d, ids, rng):
    x = np.array([d[i] for i in sorted(ids) if i in d])
    if len(x) == 0:
        return None
    b = np.array([rng.choice(x, len(x)).mean() for _ in range(N_BOOT9)])
    return {"diff": 100 * float(x.mean()), "ci": [100 * float(np.quantile(b, 0.025)), 100 * float(np.quantile(b, 0.975))],
            "n": int(len(x))}


def _joint(diffs, ids, rng):
    """Equal-weight mean over settings of each setting's mean on `ids`, with one question draw shared by all settings."""
    qs = sorted(set(ids) & set().union(*[set(d) for d in diffs]))
    M = np.array([[d.get(q, np.nan) for q in qs] for d in diffs])
    est = lambda idx: float(np.mean(np.nanmean(M[:, idx], 1)))
    b = np.array([est(rng.integers(0, len(qs), len(qs))) for _ in range(N_BOOT9)])
    return {"diff": 100 * est(np.arange(len(qs))), "ci": [100 * float(np.quantile(b, 0.025)), 100 * float(np.quantile(b, 0.975))],
            "n": len(qs), "k": len(diffs)}


def entity9(rev, S, PRIMARY, single, multi):
    rng = np.random.default_rng(0)
    out = {"settings": {}, "pooled": {}}
    for arm, cond in (("neutral", "oracle+flag"), ("directional", "oracle+flag_directional")):
        ds = []
        for s in S:
            if s["bench"] != "RAMDocs" or cond not in {r["condition"] for r in s["rows"]}:
                continue
            d = _diffs(s["rows"], "plain", cond, PRIMARY)
            ds.append(d)
            a, b = _mean_ci(d, single, rng), _mean_ci(d, multi, rng)
            out["settings"].setdefault(s["name"], {})[arm] = {"single": a, "multi": b,
                                                              "gap": a["diff"] - b["diff"] if a and b else None}
        if ds:
            out["pooled"][arm] = {"single": _joint(ds, single, rng), "multi": _joint(ds, multi, rng)}
    rev["entity9"] = out


def _seed_scores(rows, cond, metric):
    """{(id, seed): score} over conflict questions."""
    return {(r["id"], r.get("seed", 0)): r[metric] for r in rows
            if r["condition"] == cond and r.get("has_conflict") is True and r.get(metric) is not None}


def _discord(rows, a, b, metric):
    """Per question: mean over shared seeds of [exactly one arm correct], and of [flag only], [key only]."""
    A, B = _seed_scores(rows, a, metric), _seed_scores(rows, b, metric)
    acc = defaultdict(lambda: [[], [], []])
    for k in set(A) & set(B):
        x, y = A[k] >= 1, B[k] >= 1
        acc[k[0]][0].append(float(x != y)); acc[k[0]][1].append(float(y and not x)); acc[k[0]][2].append(float(x and not y))
    return [{q: float(np.mean(v[j])) for q, v in acc.items()} for j in range(3)]


def _null(rows, metric):
    P = _seed_scores(rows, "plain", metric)
    seeds = sorted({s for _, s in P})
    acc = defaultdict(list)
    for s, t in combinations(seeds, 2):
        for q in {i for i, z in P if z == s}:
            if (q, t) in P:
                acc[q].append(float((P[(q, s)] >= 1) != (P[(q, t)] >= 1)))
    return {q: float(np.mean(v)) for q, v in acc.items()} if acc else None


def _pool_share(cells):
    from .revision7 import _pool
    if not cells:
        return None
    p = _pool(cells)
    return {"share": p["diff"], "ci": [p["ci_low"], p["ci_high"]], "k": p["n_cells"]}


def agree9(rev, r2, PRIMARY):
    ps = _ps()
    fams = {"H30a": [], "H23": []}
    q = rev.get("H24", {}).get("_q", {})
    for (g, d), v in sorted(q.items()):
        run = ps.ROOT / f"runs/vkey_cb_{SHORT_GEN[g]}_{SHORT_GEN[d]}"
        if not results_path(run):
            continue
        rows = [r for r in v["rows"] if r["condition"] in ("plain", "llm_dir+flag_directional")]
        rows += [r for r in ps.rows_of(run) if r["condition"] == "llm_dir+verdict_key"]
        fams["H30a"].append((f"{g}|{d}", "CB", rows, "llm_dir+verdict_key", "llm_dir+flag_directional"))
    for k, v in r2.items():
        if k.startswith(("RD:", "CBs:", "CBmaj:")) and "answer_key" in {r["condition"] for r in v["_rows"]}:
            fams["H23"].append((k, k.split(":")[0], v["_rows"], "answer_key", "oracle+flag_directional"))
    out = {}
    for fam, cells in fams.items():
        dis, fo, ko, nul, per = [], [], [], [], {}
        for name, bench, rows, a, b in cells:
            d, f, kk = _discord(rows, a, b, PRIMARY)
            if not d:
                continue
            dis.append((bench, d)); fo.append((bench, f)); ko.append((bench, kk))
            n = _null(rows, PRIMARY)
            if n:
                nul.append((bench, n))
            per[name] = {"discord": 100 * float(np.mean(list(d.values()))), "flag_only": 100 * float(np.mean(list(f.values()))),
                         "key_only": 100 * float(np.mean(list(kk.values()))),
                         "null": 100 * float(np.mean(list(n.values()))) if n else None, "n": len(d)}
        if dis:
            has = {name for name, v in per.items() if v["null"] is not None}
            matched = [c for (name, *_), c in zip([x for x in cells if x[0] in per], dis) if name in has]
            out[fam] = {"cells": per, "discord": _pool_share(dis), "flag_only": _pool_share(fo), "key_only": _pool_share(ko),
                        "null": _pool_share(nul), "k_null": len(nul),
                        "discord_matched": _pool_share(matched)}   # the same cells as the null
    rev["agree9"] = out


def revision9(R, r2, S, PRIMARY):
    rev = R["rev"]
    single, multi = pair_audit9(rev)
    entity9(rev, S, PRIMARY, single, multi)
    agree9(rev, r2, PRIMARY)
