from experiments.conflict.detectors import Detection
from experiments.conflict.prompts import BASE, BASE_MINIMAL, build_prompt


def test_minimal_skeleton_drops_answer_policy_only():
    dets = [Detection(0, 1, 1.0, suspects=[1])]
    for mode in ("plain", "flag", "flag_directional", "remove_suspect"):
        d, _ = build_prompt("Q?", ["a", "b"], mode, dets)
        m, _ = build_prompt("Q?", ["a", "b"], mode, dets, skeleton="minimal")
        assert "give every valid answer" in d and "give every valid answer" not in m
        assert "Final answer(s):" in m
        # same conflict information in both skeletons
        assert d.split("Question:")[0].split("unknown\".")[1] == m.split("Question:")[0].split("unknown\".")[1]


def test_default_skeleton_unchanged():
    p, _ = build_prompt("Q?", ["a"], "plain")
    assert p == BASE.format(instruction="", context="Passage 0:\na", conflicts="", question="Q?")
    assert BASE_MINIMAL != BASE


def test_inverted_oracle_names_the_correct_side():
    from experiments.conflict import data
    from experiments.conflict.detectors import InvertedOracleDetector, OracleDetector
    xs = [e for e in data.load("ramdocs", label_fix="exclude") if e.has_conflict][:20]
    for e in xs:
        o, inv = OracleDetector().detect(e), InvertedOracleDetector().detect(e)
        assert [(d.i, d.j) for d in o.pairs] == [(d.i, d.j) for d in inv.pairs]
        for a, b in zip(o.pairs, inv.pairs):
            assert set(a.suspects) | set(b.suspects) == {a.i, a.j} and not set(a.suspects) & set(b.suspects)
