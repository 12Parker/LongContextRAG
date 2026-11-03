"""Eighth-review revision (§17): joint-bootstrap helpers, the covariance REML, windowed NLI, TrustRAG's system
message and the judge's JSON parsing."""
from types import SimpleNamespace

import numpy as np

from experiments.conflict import judge8, revision8
from experiments.conflict.detectors import NLIWindowDetector
from experiments.conflict.revision6 import _reml


def test_vectors_align_questions_within_pool():
    qs, vec = revision8._vectors([{"cb-1": 1.0, "cb-2": 0.0}, {"cb-2": -1.0, "cb-3": 1.0}, {"rg-9": 0.5}])
    assert qs == {"cb": ["cb-1", "cb-2", "cb-3"], "rg": ["rg-9"]}
    assert vec[0][0] == "cb" and np.isnan(vec[0][1][2]) and vec[1][1][1] == -1.0
    y, v = revision8._mv(vec, {"cb": np.arange(3), "rg": np.arange(1)})
    assert np.allclose(y, [0.5, 0.0, 0.5]) and (v > 0).all()   # one-question setting: variance floored


def test_reml_cov_matches_reml_with_diagonal_covariance():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 0.05, 12); v = np.full(12, 0.001)
    g = [np.repeat(np.arange(3), 4), np.tile(np.arange(2), 6)]
    X = np.ones((12, 1))
    b1, c1, _ = _reml(y, v, X, g)
    b2, c2, _, w = revision8._reml_cov(y, np.diag(v), X, g)
    assert np.allclose(b1, b2, atol=1e-6) and np.allclose(c1, c2, atol=1e-6)
    assert np.isclose(float(w[0] @ y), float(b2[0]))   # GLS weights reproduce the estimate


def test_nli_window_keeps_question_relevant_sentences():
    p = ("The town has a long history. It was founded by settlers in 1820. " * 5
         + "The mayor of Springfield is Jane Doe. The weather is mild.")
    out = NLIWindowDetector._prep(SimpleNamespace(max_chars=80), p, "Who is the mayor of Springfield?")
    assert "mayor of Springfield is Jane Doe" in out and len(out) <= 80
    short = "Short passage about the mayor."
    assert NLIWindowDetector._prep(SimpleNamespace(max_chars=500), short, "Who is the mayor?") == short


def test_trustrag_sends_system_message_only_when_asked():
    from experiments.conflict.baselines import TRUST_SYSTEM, trustrag
    seen = []

    class L:
        def __call__(self, prompt, seed=0, json_mode=False, max_tokens=None, system=None):
            seen.append(system)
            from experiments.conflict.llm import LLMResult
            return LLMResult("x", 1, 1, 0.0, 0.0, False)
    emb = lambda t: np.eye(len(t))
    trustrag("q?", ["a"], L(), emb)
    trustrag("q?", ["a"], L(), emb, system=TRUST_SYSTEM)
    assert seen == [None] * 3 + [TRUST_SYSTEM] * 3


def test_judge_json_parse():
    assert judge8._json('noise {"gives_gold": true} tail') == {"gives_gold": True}
    assert judge8._json("not json") == {}
