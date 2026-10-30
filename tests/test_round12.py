"""Tenth revision (§19): the frozen predictor, the topic-cluster sign flip, the HealthContradict label mapping and the
judge's response key."""
import numpy as np

from experiments.conflict import export_audit8, judge10, revision10


def test_frozen_predictor_formula():
    V = {"right": np.array([1.0, 0.0, 1.0]), "wrong": np.array([0.0, 1.0, 0.0]), "g": np.array([0.0, 1.0, 0.5])}
    fr, fw = revision10.FOLLOW19["gpt-4o-mini"]
    want = [fr, -fw, 0.5 * fr]
    assert np.allclose(revision10._pred(V, "gpt-4o-mini"), want)
    assert np.allclose(revision10._pred(V, "gpt-4o-mini", ones=True), [1.0, -1.0, 0.5])


def test_follow_rates_are_the_registered_values():
    assert revision10.FOLLOW19["gpt-6.1-terra"] == (0.6667, 0.9474)
    assert set(revision10.FOLLOW19) == set(revision10.GEN19.values())


def test_cluster_flip_uses_topics_as_units():
    rng = np.random.default_rng(0)
    d = {f"q{i}": 1.0 for i in range(40)}
    one_topic = {i: "t" for i in d}            # one cluster: the sign flip cannot reject
    many = {i: i for i in d}
    assert revision10._cluster_flip(d, one_topic, rng, n=2000, side=1) > 0.4
    assert revision10._cluster_flip(d, many, rng, n=2000, side=1) < 0.01


def test_hc_label_mapping():
    meta = {"types": ["misinfo", "correct"], "gold": "no", "wrong": "yes"}
    j = judge10.hc_fields(meta, {"page_1": "yes", "page_2": "no"})
    assert j["correct_supports_gold"] == "yes" and j["misinfo_supports_wrong"] == "yes"
    j = judge10.hc_fields(meta, {"page_1": "unclear", "page_2": "yes"})
    assert j["correct_supports_gold"] == "no" and j["misinfo_supports_wrong"] == "unclear"
    assert judge10.hc_fields(meta, {}) == {}


def test_response_key_is_stable():
    assert judge10.response_key("abc") == judge10.response_key("abc") != judge10.response_key("abd")


def test_audit_sheet_redacts_user_handles():
    r = export_audit8.redact
    assert r("as /u/someone said") == "as u/[user] said" and r("thanks u/abc_12)") == "thanks u/[user])"
    assert r("https://www.reddit.com/user/someone/") == "https://www.reddit.com/user/[user]/"
    assert r("reddit.com/r/x/comments/ab12/title/fbeovl8/?context=3") == "reddit.com/r/x/comments/ab12/title/[comment]/?context=3"
    assert r("[this](https://www.tiktok.com/@name/video/123?x=1) more") == "[this](https://www.tiktok.com/@[user]) more"
    assert r("Secretary of State @MikePompeo4USA") == "Secretary of State @MikePompeo4USA"
    assert r("reddit.com/r/newzealand/comments/fsoice/coronavirus_47/") == "reddit.com/r/newzealand/comments/fsoice/coronavirus_47/"
