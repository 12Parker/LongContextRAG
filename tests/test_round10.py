"""§17: the agreement gate and the fresh ConflictBank / HealthContradict sets."""
import json

import pytest

from experiments.conflict import data
from experiments.conflict.detectors import AgreeDetector, Detection, DetectorOutput, suspect_set


class Fixed:
    def __init__(self, pairs, cost=0.1):
        self.pairs, self.cost = pairs, cost

    def detect(self, ex):
        return DetectorOutput(self.pairs, {}, 0.5, self.cost)


A = [Detection(0, 1, 1.0, "contradiction", [1])]
B = [Detection(0, 1, 1.0, "contradiction", [0])]


def test_agree_passes_primary_only_on_same_suspects():
    out = AgreeDetector(Fixed(A), Fixed([Detection(0, 1, 1.0, "contradiction", [1])])).detect(None)
    assert out.pairs == A and out.extra["agree"] and out.cost_usd == pytest.approx(0.2)
    assert AgreeDetector(Fixed(A), Fixed(B)).detect(None).pairs == []      # opposite direction
    assert AgreeDetector(Fixed(A), Fixed([])).detect(None).pairs == []     # second detector flags nothing
    assert AgreeDetector(Fixed([]), Fixed([])).detect(None).pairs == []    # neither flags: no agreement
    neutral = [Detection(0, 1, 1.0, "contradiction", [])]
    assert AgreeDetector(Fixed(neutral), Fixed(neutral)).detect(None).pairs == []   # no direction to agree on
    assert suspect_set(DetectorOutput(A + [Detection(1, 2, 1.0, "unrelated", [2])])) == {1}


@pytest.mark.skipif(not (data.DATA_DIR / "healthcontradict/hc_eval.jsonl").exists(), reason="data not built")
def test_healthcontradict_items():
    ex = data.load("healthcontradict")
    assert len(ex) == 247 and all(e.has_conflict for e in ex)
    for e in ex:
        assert sorted(e.meta["doc_types"]) == ["correct", "misinfo"] and len(e.passages) == 2
        assert {e.gold_answers[0], e.wrong_answers[0]} == {"yes", "no"} and e.question.endswith("Answer yes or no.")
        assert all(len(p) <= 4000 for p in e.passages) and e.group.startswith("hc-topic-")


@pytest.mark.skipif(not (data.DATA_DIR / "conflictbank/conflictbank_fresh.jsonl").exists(), reason="data not built")
def test_fresh_conflictbank_is_disjoint():
    old, new = data.load("conflictbank"), data.load("conflictbank_fresh")
    assert len(new) == 400 and sum(e.has_conflict for e in new) == 300
    assert not {e.id for e in old} & {e.id for e in new}
    assert not {e.question for e in old} & {e.question for e in new}


def test_revision7_helpers():
    import numpy as np
    from experiments.conflict.revision7 import _bh, _pool, _wilson
    q = _bh([0.01, 0.04, 0.03, 0.5])
    assert np.allclose(q, [0.04, 0.16 / 3, 0.16 / 3, 0.5])
    lo, hi = _wilson(5, 10)
    assert 0.23 < lo < 0.24 and 0.76 < hi < 0.77
    # two cells sharing questions: pooled mean of cell means, identical cells give zero spread across cells
    d = {f"q{i}": (0.1 if i % 2 else -0.1) for i in range(40)}
    out = _pool([("CB", d), ("CB", dict(d))])
    assert abs(out["diff"]) < 1e-9 and out["n_cells"] == 2 and out["equivalent"]
    big = _pool([("CB", {k: 0.5 for k in d})])
    assert big["diff"] == 50 and big["p"] < 0.01 and not big["equivalent"]


@pytest.mark.skipif(not (data.DATA_DIR / "poisonedrag/pr_eval.jsonl").exists(), reason="data not built")
def test_poisonedrag_items():
    ex = data.load("poisonedrag")
    att = [e for e in ex if e.has_conflict]
    assert len(ex) == 500 and len(att) == 200
    assert all(len(e.passages) == 5 and e.gold_answers for e in ex)
    assert all(e.wrong_answers and "misinfo" in e.meta["doc_types"] or e.meta["n_attack_retrieved"] == 0 for e in att)
    assert not any("misinfo" in e.meta["doc_types"] for e in ex if not e.has_conflict)


def test_answer_judge_and_verdict_key():
    from experiments.conflict.detectors import LLMAnswerJudge
    from experiments.conflict.llm import FakeLLM
    from experiments.conflict.prompts import build_prompt
    ex = data.Example(id="x", dataset="hoh", question="Q?", passages=["a 24", "a 25"], gold_answers=["25"],
                      wrong_answers=["24"], has_conflict=True)
    out = LLMAnswerJudge(FakeLLM(lambda p: '{"answers": ["24", "25"], "false": ["24"]}')).detect(ex)
    assert out.pairs == [] and out.extra["false_answers"] == ["24"]
    p, _ = build_prompt("Q?", ex.passages, "verdict_key", skeleton="single", wrong_answers=out.extra["false_answers"])
    assert 'The answer "24" is false.' in p and "MISINFORMATION" not in p
    bad = LLMAnswerJudge(FakeLLM(lambda p: "not json")).detect(ex)
    assert bad.extra["false_answers"] == []
