"""Eleventh revision, part 1 (PR§20.2): the known rule, direction counting, object masking, pairwise accuracy, the
pooled logistic and the conditioned account."""
from types import SimpleNamespace

import numpy as np

from experiments.conflict import rev11_prior as m


def _row(i, seed, ok, cond="closedbook"):
    return {"id": i, "seed": seed, "condition": cond, "ramdocs_strict_oneway": float(ok)}


def test_known_rule_two_of_three_or_the_single_seed():
    rows = [_row("a", s, ok) for s, ok in enumerate((1, 1, 0))] + [_row("b", s, ok) for s, ok in enumerate((1, 0, 0))]
    rows += [_row("c", 0, 1), _row("d", 0, 0), _row("e", 0, 1, cond="plain")]
    assert m.known_ids(rows) == {"a", "c"}


def test_gold_flag_counts_definitions():
    e = SimpleNamespace(meta={"doc_types": ["correct", "misinfo", "noise"]},
                        gold_pairs={(0, 1): "contradiction", (0, 2): "consistent"})
    f = lambda s, i=0, j=1: {"i": i, "j": j, "suspects": s}
    c = m.gold_flag_counts(e, [f([1])])
    assert (c["dir_right"], c["dir_total"], c["one_right"], c["one_total"], c["misinfo"]) == (1, 1, 1, 1, 1)
    c = m.gold_flag_counts(e, [f([0, 1])])           # "both": a miss for §11.3, not single-side
    assert (c["dir_right"], c["dir_total"], c["one_total"], c["both"]) == (0, 1, 0, 1)
    c = m.gold_flag_counts(e, [f([])])               # "unclear": skipped by §11.3
    assert (c["dir_total"], c["one_total"], c["unclear"]) == (0, 0, 1)
    c = m.gold_flag_counts(e, [f([0]), f([2], 0, 2)])   # wrong side; the non-gold pair is ignored
    assert (c["dir_right"], c["dir_total"], c["one_right"], c["one_total"], c["correct"]) == (0, 1, 0, 1, 1)


def test_ratio_split_estimates():
    o = m._ratio_split([1, 1, 0, 1], [1, 1, 1, 2], [1, 1, 0, 0])
    assert o["known"] == 1.0 and o["unknown"] == 1 / 3 and abs(o["diff"] - 2 / 3) < 1e-12
    assert o["n_known_flags"] == 2 and o["n_unknown_flags"] == 3


def test_mask_object_is_word_bounded_and_case_insensitive():
    t = m.mask_object("Paris is big. paris, not Parisian.", "Paris")
    assert t.count("OBJSLOT") == 2 and "Parisian" in t
    assert m.mask_object("a (b) c", "(b)").count("OBJSLOT") == 1
    assert m.mask_object("text", "") == "text"


def test_pairwise_accuracy_counts_ties_half():
    assert list(m.pairwise_accuracy([0.9, 0.2, 0.5], [0.1, 0.8, 0.5])) == [1.0, 0.0, 0.5]


def test_logit_recovers_coefficients():
    rng = np.random.default_rng(1)
    X = np.c_[np.ones(4000), rng.integers(0, 2, 4000)]
    y = (rng.random(4000) < 1 / (1 + np.exp(-(X @ [-0.5, 1.2])))).astype(float)
    b, ok = m._logit(X, y)
    assert ok and abs(b[1] - 1.2) < 0.2


def test_logit_known_drops_constant_generators_and_flags_separation():
    obs = [(f"q{i}", "g1", i % 2 == 0, 0.5 + 0.25 * (i % 2 == 0) * (i % 4 == 0)) for i in range(40)]
    obs += [(f"q{i}", "g2", i % 2 == 0, 1.0) for i in range(40)]   # all followed: no information, dropped
    o = m.logit_known(obs)
    assert o["or"] is not None and o["or"] > 1
    sep = [(f"q{i}", "g1", i < 5, 1.0 if i < 5 else 0.5) for i in range(30)]   # every known question followed
    o = m.logit_known(sep)
    assert o["or"] is None and o["log_or"] is None and o["converged"] is False


def test_conditioned_account_formula():
    D, G, W, K = [1, 0, 1, 0], [0, 1, 0.5, 1], [0, 1, 0, 1], [1, 1, 0, 0]
    rates = (0.9, 0.5, 0.8, 0.4)
    assert np.allclose(m.cond_pred(D, G, W, K, rates), [0.9, -0.8, 0.25, -0.4])
    same = m.cond_pred(D, G, W, K, (0.6, 0.6, 0.7, 0.7))
    assert np.allclose(same, np.array(D) * (1 - np.array(G)) * 0.6 - np.array(W) * np.array(G) * 0.7)


def test_summary_prints_registered_single_side_and_11_3_coding():
    x = {"known": 1.0, "known_ci": [1.0, 1.0], "unknown": 0.9, "unknown_ci": [0.8, 1.0], "diff": 0.1, "diff_ci": [0.0, 0.2],
         "n_known_flags": 3, "n_unknown_flags": 10}
    r2 = {"registered": "single_side", "note": "n", "d": {"closedbook_seeds": [0], "n_known": 3, "single_side": x,
                                                         "direction": {**x, "unknown": 0.7}}}
    s = m.summary({"R2": r2})
    assert "single" in s and "§11.3" in s and "0.900" in s and "0.700" in s


def test_mean_ci_is_deterministic():
    a, b = m._mean_ci([0, 1, 1, 0, 1]), m._mean_ci([0, 1, 1, 0, 1])
    assert a == b and a["rate"] == 0.6 and a["ci"][0] <= 0.6 <= a["ci"][1] and m._mean_ci([])["n"] == 0
