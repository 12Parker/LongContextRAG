"""
Paired statistics for per-question scores (reviewer comments 1 and 2).

Bootstrap resamples *clusters* (e.g. stories for NarrativeQA), because questions about
the same story are not independent. For datasets with one question per item, the
cluster is the item itself and this reduces to the ordinary paired bootstrap.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def paired_bootstrap(a: dict[str, float], b: dict[str, float], groups: dict[str, str] | None = None,
                     n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Mean of (b - a) over shared ids, with a cluster-bootstrap CI."""
    ids = sorted(set(a) & set(b))
    if not ids:
        raise ValueError("no shared ids")
    groups = groups or {i: i for i in ids}
    clusters = defaultdict(list)
    for i in ids:
        clusters[groups[i]].append(b[i] - a[i])
    keys = list(clusters)
    sums = np.array([sum(clusters[k]) for k in keys])
    counts = np.array([len(clusters[k]) for k in keys])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    boots = sums[draws].sum(1) / counts[draws].sum(1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"diff": float(sums.sum() / counts.sum()), "ci_low": float(lo), "ci_high": float(hi),
            "n": len(ids), "n_clusters": len(keys)}


def sign_flip_p(a: dict[str, float], b: dict[str, float], n_perm: int = 10_000, seed: int = 0,
                max_perm: int | None = None) -> float:
    """Two-sided paired permutation test on the mean difference. With max_perm, a p-value at its floor (no
    permuted statistic as extreme as the observed one) is refined with further permutations, in blocks, up to
    max_perm in total, so that it can survive a large Holm family. Other p-values are unchanged."""
    ids = sorted(set(a) & set(b))
    d = np.array([b[i] - a[i] for i in ids])
    obs = abs(d.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1, 1], size=(n_perm, len(d)))
    hits, total = int((np.abs((signs * d).mean(1)) >= obs - 1e-12).sum()), n_perm
    while max_perm and hits == 0 and total < max_perm:
        k = min(100_000, max_perm - total)
        signs = rng.choice(np.array([-1, 1], dtype=np.int8), size=(k, len(d)))
        hits += int((np.abs((signs * d).mean(1)) >= obs - 1e-12).sum()); total += k
    return float((1 + hits) / (total + 1))


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, running, out = len(items), 0.0, {}
    for rank, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - rank) * p))
        out[k] = running
    return out
