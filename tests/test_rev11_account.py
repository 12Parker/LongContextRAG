"""Eleventh revision (PR§20.2, R6-R7): the common-rate baseline, the cell-level marginal, the selection error split,
H32's draw order, the plain-draw swap and h31x on substituted arms."""
import numpy as np
import pytest

from experiments.conflict import rev11_account as ra
from experiments.conflict import revision10


def test_common_rates_are_the_equal_weight_mean():
    assert ra.common_rates({"a": (0.2, 0.4), "b": (0.6, 0.8)}) == pytest.approx((0.4, 0.6))
    fr, fw = ra.common_rates()
    assert fr == pytest.approx(np.mean([v[0] for v in revision10.FOLLOW19.values()]))
    assert fw == pytest.approx(np.mean([v[1] for v in revision10.FOLLOW19.values()]))


def test_pred_rates_matches_the_frozen_formula():
    V = {"right": np.array([1.0, 0.0, 1.0]), "wrong": np.array([0.0, 1.0, 1.0]), "g": np.array([0.0, 1.0, 0.5])}
    g = "gpt-6.1-terra"
    assert np.allclose(ra.pred_rates(V, *revision10.FOLLOW19[g]), revision10._pred(V, g))


def test_marginal_uses_cell_means():
    assert ra.marginal(0.5, 1.0, 0.4, 0.2, 0.25) == pytest.approx(0.5 * 0.4 * 0.75 - 0.2 * 0.25)
    fr, fw = np.array([0.5, 1.0]), np.array([1.0, 0.5])
    assert np.allclose(ra.marginal(fr, fw, np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.0, 1.0])), [0.5, -0.5])


def test_error_split_parts_sum_to_the_error():
    c, k = {"n": 300, "pred": 6.0, "obs": 4.0}, {"n": 100, "pred": -8.0, "obs": 0.0}
    s = ra.error_split(c, k)
    assert s["pred"] == pytest.approx((6 * 300 - 8 * 100) / 400) and s["obs"] == pytest.approx(3.0)
    assert s["conflict"] == pytest.approx(0.75 * 2.0) and s["clean"] == pytest.approx(0.25 * -8.0)
    assert s["err"] == pytest.approx(s["pred"] - s["obs"]) == pytest.approx(s["conflict"] + s["clean"])


def test_h32_draws_follow_the_seed0_stream():
    rng = np.random.default_rng(0)
    want = [rng.integers(0, 7, 7) for _ in range(3)]
    got = list(ra.h32_draws(7, n_boot=3))
    assert len(got) == 3 and all(np.array_equal(a, b) for a, b in zip(got, want))


def test_flags_any():
    assert ra.flags_any(["a", "b"], {"a": [], "b": [{"i": 0, "j": 1}]})
    assert not ra.flags_any(["a", "c"], {"a": [], "b": [{"i": 0, "j": 1}]})


def test_swap_plain_replaces_only_plain_rows():
    rows = [{"id": "q", "seed": 0, "condition": "plain", "x": 1}, {"id": "q", "seed": 0, "condition": "flag", "x": 2}]
    other = [{"id": "q", "seed": 0, "condition": "plain", "x": 3}, {"id": "q", "seed": 0, "condition": "flag", "x": 4}]
    out = ra.swap_plain(rows, other)
    assert sorted((r["condition"], r["x"]) for r in out) == [("flag", 2), ("plain", 3)]
    with pytest.raises(ValueError):
        ra.swap_plain(rows, [{"id": "z", "seed": 0, "condition": "plain", "x": 5}])


@pytest.mark.filterwarnings("ignore::RuntimeWarning")   # tiny toy arms: some resamples tie
def test_h31x_runs_on_substituted_arms():
    R, W = ["r1", "r2", "r3"], ["w1", "w2", "w3"]
    pl = {"r1": 0.0, "r2": 0.0, "r3": 1.0, "w1": 1.0, "w2": 1.0, "w3": 0.0}
    fl = {"r1": 1.0, "r2": 0.0, "r3": 1.0, "w1": 0.0, "w2": 1.0, "w3": 0.0}
    oc = {i: np.array([fl[i], 1.0 - fl[i], 0.0]) for i in pl}
    out = ra._h31x({"g": {"R": R, "W": W, "pl": pl, "fl": fl, "oc": oc}}, {})
    gR, gW = np.mean([fl[i] - pl[i] for i in R]), np.mean([fl[i] - pl[i] for i in W])
    assert out["g"]["breakeven"] == pytest.approx(-gW / (gR - gW))
    assert out["g"]["follow_right"] == pytest.approx(0.5) and out["g"]["follow_wrong"] == pytest.approx(0.5)
