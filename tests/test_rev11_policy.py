"""PR§20.4 (rev11_policy): the one-way misinformation helper, break-even regimes, BCa, the permutation p, the stance
table and the setting map. Pure helpers only; no run is read."""
import math

import numpy as np

from experiments.conflict import rev11_policy as rp


def test_misinfo_oneway_is_whole_word_and_one_way():
    assert rp.misinfo_oneway(["Paris", "Lyon"], ["lyon"]) == 1.0
    assert rp.misinfo_oneway(["The Lyon."], ["Lyon"]) == 1.0          # normalised: article and punctuation dropped
    assert rp.misinfo_oneway(["Lyons"], ["Lyon"]) == 0.0              # whole words only
    assert rp.misinfo_oneway(["5,000"], ["5,000 people"]) == 0.0      # one-way: the wrong answer must be inside
    assert rp.misinfo_oneway([], ["x"]) == 0.0 and rp.misinfo_oneway(["x"], []) == 0.0


def test_outcome_oneway_order():
    P = rp.PRIMARY
    assert rp.outcome_oneway({P: 1.0, "pred_answers": ["Paris"]}, ["Lyon"]) == "correct"
    assert rp.outcome_oneway({P: 0.0, "pred_answers": ["unknown", "Lyon"]}, ["Lyon"]) == "misinfo"
    assert rp.outcome_oneway({P: 0.0, "pred_answers": ["Unknown."]}, ["Lyon"]) == "abstain"
    assert rp.outcome_oneway({P: 0.0, "pred_answers": ["Nice"]}, ["Lyon"]) == "other"


def test_per_q_seed_means_and_subsets():
    P = rp.PRIMARY
    rows = [{"id": "a", "condition": "plain", "has_conflict": True, P: 1.0},
            {"id": "a", "condition": "plain", "has_conflict": True, P: 0.0},
            {"id": "b", "condition": "plain", "has_conflict": None, P: 1.0},     # excluded by the label rule
            {"id": "c", "condition": "plain", "has_conflict": False, P: 1.0},
            {"id": "a", "condition": "oracle+flag", "has_conflict": True, P: 1.0}]
    assert rp.per_q(rows, "plain", True, lambda r: r[P]) == {"a": 0.5}
    assert rp.per_q(rows, "plain", False, lambda r: r[P]) == {"c": 1.0}
    assert rp.per_q(rows, "plain", True, lambda r: None) == {}


def test_selective_is_the_ratio_estimand():
    P = rp.PRIMARY
    row = lambda i, c, s, a, sub=True: {"id": i, "condition": c, "has_conflict": sub, P: s, "pred_answers": [a]}
    rows = [row("a", "plain", 1.0, "x"), row("a", "plain", 0.0, "unknown"), row("b", "plain", 0.0, "y"),
            row("b", "plain", 0.0, "y"), row("c", "plain", 1.0, "z", False),
            row("a", "oracle+flag", 0.0, "Unknown."), row("b", "oracle+flag", 1.0, "y")]
    X = rp.sel_arrays(rows, "plain", "plain", True)
    assert X[:, :2].tolist() == [[1, 1], [0, 2]]                   # a: 1 correct of 1 answered; b: 0 of 2
    s = rp.sel_ratio(X[:, :2])
    assert math.isclose(s["est"], 100 / 3) and s["answered"] == 3   # sum ratio 1/3, not the per-question mean 1/2
    Y = rp.sel_arrays(rows, "plain", "oracle+flag", True)
    assert Y.tolist() == [[1, 1, 0, 0], [0, 2, 1, 1]]
    assert math.isclose(rp.sel_diff(Y)["diff"], 100 * (1 - 1 / 3))
    assert rp.sel_arrays(rows, "plain", "plain", False)[:, :2].tolist() == [[1, 1]]
    assert rp.sel_ratio(np.zeros((0, 2))) is None and rp.sel_ratio(np.array([[0.0, 0.0]])) is None


def test_mean_and_diff_ci():
    d = {f"q{i}": float(i % 2) for i in range(40)}
    m = rp.mean_ci(d)
    assert m["est"] == 50.0 and m["ci"][0] < 50 < m["ci"][1] and m["n"] == 40
    z = rp.diff_ci(d, d)
    assert z["diff"] == 0.0 and z["ci"] == [0.0, 0.0]
    assert rp.mean_ci({}) is None and rp.diff_ci({"a": 1.0}, {"b": 1.0}) is None


def test_regimes_and_breakeven():
    assert rp.regime(0.2, -0.01) == "crossing"
    assert math.isclose(rp.breakeven(0.2, -0.01)[0], 0.01 / 0.21)
    assert rp.regime(0.1, 0.0) == "helps_all" and rp.breakeven(0.1, 0.0)[0] == 0.0
    assert rp.regime(-0.1, 0.0) == "hurts_all" and rp.breakeven(-0.1, 0.0)[0] == 1.0
    assert rp.regime(-0.1, 0.1) == "reverse" and np.isnan(rp.breakeven(-0.1, 0.1)[0])
    assert rp.regime(0.0, 0.0) == "zero" and np.isnan(rp.breakeven(0.0, 0.0)[0])
    K, C = np.array([0.2, 0.1, -0.1]), np.array([-0.2, 0.05, -0.05])
    assert np.allclose(rp.breakeven(K, C), [0.5, 0.0, 1.0])
    # at the break-even share the effect is zero
    k, c = 0.3, -0.06
    pi = rp.breakeven(k, c)[0]
    assert math.isclose(pi * k + (1 - pi) * c, 0.0, abs_tol=1e-12)


def test_bca_reduces_to_percentile_without_bias_or_skew():
    theta = 0.0
    boots = np.linspace(-1, 1, 2001)
    jack = np.array([-0.1, 0.1] * 20)                      # symmetric: no acceleration
    lo, hi, z0, a = rp.bca(theta, boots, jack)
    assert abs(a) < 1e-12 and abs(z0) < 0.01
    plo, phi = np.quantile(boots, [0.025, 0.975])
    assert abs(lo - plo) < 0.01 and abs(hi - phi) < 0.01


def test_bca_moves_against_a_biased_bootstrap():
    theta = 0.0
    boots = np.linspace(-0.5, 1.5, 2001)                   # centred above theta
    jack = np.array([-0.1, 0.1] * 20)
    lo, hi, z0, _ = rp.bca(theta, boots, jack)
    plo, phi = np.quantile(boots, [0.025, 0.975])
    assert z0 < 0 and lo < plo and hi < phi


def test_perm_p_detects_a_monotone_relation():
    x = np.arange(12.0)
    two, less = rp.perm_p(x, -x, n=2000)
    assert two < 0.01 and less < 0.01
    two, less = rp.perm_p(x, x, n=2000)
    assert two < 0.01 and less > 0.99


def test_stance_table():
    items = {"hc-1": {"passages": ["x" * 4000, "short"]}, "hc-2": {"passages": ["y" * 10, "z" * 4000]}}
    L = [{"id": "hc-1", "types": ["correct", "misinfo"],
          "judge": {"correct_supports_gold": "yes", "misinfo_supports_wrong": "no"}},
         {"id": "hc-2", "types": ["misinfo", "correct"],
          "judge": {"correct_supports_gold": "yes", "misinfo_supports_wrong": "yes"}}]
    t = rp.stance_table(L, items)
    assert t["joint"]["yes"]["no"] == 1 and t["joint"]["yes"]["yes"] == 1 and t["both_confirmed"] == 1
    assert t["correct_page"]["yes"] == 1.0 and t["misinfo_page"]["no"] == 0.5
    assert t["reversed_pages"] == 1 and t["n_pages"] == 4
    assert t["at_cut"]["n"] == 2 and t["at_cut"]["by_type"]["correct"]["n"] == 2
    assert t["verdict_by_cut"]["cut"]["yes"] == 2 and t["verdict_by_cut"]["uncut"]["no"] == 1


def test_settings_follow_revision6():
    eff = lambda *c: {"effects": {rp.PRIMARY: {k: {} for k in c}}}
    R = {"rev": {"r2": {"CBs:gpt-4o-mini": eff("oracle+flag"), "RD:gpt-4o-mini": eff("oracle+flag"),
                        "RD:gpt-6-luna": eff("oracle+flag"), "RG:gpt-6-luna": eff("oracle+flag"),
                        "CBs:gpt-4.1": eff("llm_dir+flag_directional"), "gpt-4o-mini": eff("oracle+flag")},
                 "generality": {"gpt-4o-mini|minimal": eff("oracle+flag"), "gpt-4.1|default": eff("llm_dir+flag")}}}
    S = rp.settings(R)
    names = [s["name"] for s in S]
    assert names == ["CBs:gpt-4o-mini", "RD:gpt-6-luna", "RG:gpt-6-luna", "RD:gpt-4o-mini|minimal"]
    by = {s["name"]: s for s in S}
    assert by["RD:gpt-4o-mini|minimal"]["policy"] == "minimal" and by["RG:gpt-6-luna"]["bench"] == "RAGuard"
    assert by["RD:gpt-6-luna"]["run"] == "holdout_luna"
    assert all(n.split(":")[0] in ("CB", "CBs", "CBn", "CBmaj", "CBf", "RD", "RG") for n in rp.RUN_OF)


def test_clean_json():
    o = rp._clean({"a": float("nan"), "b": np.float64(1.5), "c": [np.int64(2), np.bool_(True)], 3: float("inf")})
    assert o == {"a": None, "b": 1.5, "c": [2, True], "3": None}
