"""§11–§12: single-answer skeleton, one-wrong flag, single-call and closed-book modes, the majority
variant's pair labels, and reasoning-model request parameters."""
import json

from experiments.conflict import llm as llm_mod
from experiments.conflict.data import load_conflictbank
from experiments.conflict.detectors import Detection, OracleDetector
from experiments.conflict.prompts import BASE_SINGLE, build_prompt
from experiments.conflict.run import parse_condition

P = ["a", "b", "c"]
D = [Detection(0, 1, 1.0, "contradiction", [1])]


def test_single_skeleton_and_answer_line():
    p, kept = build_prompt("Q?", P, "plain", skeleton="single")
    assert "exactly one correct answer" in p and p.rstrip().endswith("Final answer(s): <your one answer>")
    assert kept == [0, 1, 2]


def test_onewrong_is_neutral_about_direction():
    p, _ = build_prompt("Q?", P, "flag_onewrong", D)
    assert "exactly one of the two contains misinformation (it is not known which)" in p
    assert "LIKELY MISINFORMATION" not in p
    assert p.count("one side contains misinformation") == 2          # both passages of the pair, same header


def test_self_identify_and_closedbook():
    p, _ = build_prompt("Q?", P, "self_identify")
    assert "Suspect passages:" in p and "Passage 2:" in p
    p, kept = build_prompt("Q?", P, "closedbook", skeleton="single")
    assert kept == [] and "Passage" not in p and "own knowledge" in p
    assert parse_condition("self_identify") == ("none", "self_identify")
    assert parse_condition("closedbook") == ("none", "closedbook")


def test_majority_variant_labels(tmp_path):
    row = {"id": "cb-1", "relation": "P1", "question": "q", "gold": "A", "wrong": "B", "has_conflict": True,
           "passages": ["t", "f1", "f2", "f3"], "types": ["correct", "misinfo", "misinfo", "misinfo"]}
    f = tmp_path / "maj.jsonl"; f.write_text(json.dumps(row) + "\n")
    ex = load_conflictbank(f)[0]
    assert ex.gold_pairs[(0, 1)] == "contradiction" and ex.gold_pairs[(1, 2)] == "consistent"
    dets = OracleDetector().detect(ex).pairs
    assert sorted((d.i, d.j, tuple(d.suspects)) for d in dets) == [(0, 1, (1,)), (0, 2, (2,)), (0, 3, (3,))]


def test_reasoning_models_use_max_completion_tokens(monkeypatch, tmp_path):
    sent = {}

    class Resp:
        class _U: prompt_tokens, completion_tokens, prompt_tokens_details = 10, 5, None
        usage = _U()
        choices = [type("C", (), {"message": type("M", (), {"content": "Final answer(s): x"})()})()]

    class Client:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    sent.update(kw); return Resp()
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    m = llm_mod.OpenAILLM("gpt-6-luna", temperature=1.0, cache_path=tmp_path / "c.sqlite")
    m.client = Client()
    m("hello", seed=0)
    assert "max_tokens" not in sent and "temperature" not in sent
    assert sent["max_completion_tokens"] == llm_mod.REASONING_MAX_TOKENS
    assert BASE_SINGLE  # imported for the skeleton test above
