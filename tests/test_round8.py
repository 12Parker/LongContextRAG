"""§15: the detector-answer hint condition and the artefact-probe variants of ConflictBank."""
import pytest

from experiments.conflict.prepare_cb_probe import check, reverse_item
from experiments.conflict.prompts import build_prompt
from experiments.conflict.run import parse_condition

P = ["a", "b", "c"]


def test_hint_prompt_marks_no_passage():
    p, kept = build_prompt("Q?", P, "hint", skeleton="single", hint=["Paris"])
    assert 'Another model that read these passages answered: "Paris".' in p
    assert "MISINFORMATION" not in p and "CONFLICT" not in p and kept == [0, 1, 2]
    assert parse_condition("hint") == ("none", "hint")
    with pytest.raises(ValueError):
        build_prompt("Q?", P, "hint")


def test_reverse_keeps_types_aligned_and_check():
    r = reverse_item({"passages": ["x", "y", "z"], "types": ["noise", "misinfo", "correct"]})
    assert r["passages"] == ["z", "y", "x"] and r["types"] == ["correct", "misinfo", "noise"]
    assert check("Born in India.", "India was his birthplace.", ["India", "Slovenia"])
    assert not check("Born in India.", "Born in Asia.", ["India", "Slovenia"])
