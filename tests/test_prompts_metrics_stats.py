import numpy as np

from experiments.conflict.data import Example
from experiments.conflict.detectors import Detection
from experiments.conflict.metrics import extract_answers, normalize_answer, question_level, score
from experiments.conflict.prompts import build_prompt
from experiments.conflict.stats import holm, paired_bootstrap, sign_flip_p

PS = ["alpha text", "beta text", "gamma text"]


def test_prompt_modes():
    d = [Detection(0, 2, 0.5, "factual")]
    plain, _ = build_prompt("q?", PS, "plain")
    flag, _ = build_prompt("q?", PS, "flag", d, style="original")
    instr, _ = build_prompt("q?", PS, "instruct")
    rem, kept = build_prompt("q?", PS, "remove", d)
    assert "CONFLICT" not in plain and "contradict" not in plain
    assert "POTENTIAL CONFLICT with passage(s): 2" in flag and "Passages 0 and 2: factual conflict" in flag
    assert "contradict each other" in instr
    assert kept == [0, 1] and "gamma" not in rem
    # flag with no detections renders the same as plain: only detections change the prompt
    assert build_prompt("q?", PS, "flag", [])[0] == plain


def test_metrics():
    assert normalize_answer("The A-League!") == "aleague"
    assert extract_answers("blah\nFinal answer(s): AFL | NBL") == ["AFL", "NBL"]
    assert extract_answers('All Correct Answers: ["1963", "1956"]. Explanation: x') == ["1963", "1956"]
    ex = Example("r", "ramdocs", "q", PS, gold_answers=["AFL", "NBL"], wrong_answers=["NFL"])
    assert score(ex, "Final answer(s): AFL | NBL")["ramdocs_strict"] == 1.0
    assert score(ex, "Final answer(s): AFL")["ramdocs_strict"] == 0.0
    assert score(ex, "Final answer(s): AFL | NBL | NFL")["ramdocs_strict"] == 0.0
    # no partial-word matches
    assert score(ex, "Final answer(s): AFLX | NBL")["gold_recall"] == 0.5


def test_question_level_fpr():
    xs = [Example(str(i), "t", "q", PS, has_conflict=(i < 2)) for i in range(4)]
    flagged = {"0": {(0, 1)}, "2": {(0, 1)}}
    m = question_level(xs, flagged)
    assert m["tp"] == 1 and m["fp"] == 1 and m["fn"] == 1 and m["false_positive_rate"] == 0.5


def test_stats():
    rng = np.random.default_rng(0)
    a = {str(i): float(rng.random() < 0.5) for i in range(400)}
    b = {k: min(1.0, v + (rng.random() < 0.2)) for k, v in a.items()}
    r = paired_bootstrap(a, b, n_boot=2000)
    assert r["ci_low"] < r["diff"] < r["ci_high"] and r["ci_low"] > 0
    assert sign_flip_p(a, b, n_perm=2000) < 0.01
    assert sign_flip_p(a, a, n_perm=500) == 1.0
    h = holm({"x": 0.01, "y": 0.04, "z": 0.03})
    assert h["x"] == 0.03 and h["z"] == 0.06 and h["y"] == 0.06


def test_ramdocs_matching_is_bidirectional():
    ex = Example("r", "ramdocs", "q", PS, gold_answers=["January 1, 1877", "1,664 people"],
                 wrong_answers=["5,000 people"])
    s = score(ex, "Final answer(s): 1877 | 1,664")
    assert s["ramdocs_strict"] == 1.0 and s["ramdocs_strict_oneway"] == 0.0
    s = score(ex, "Final answer(s): 1877 | 1,664 | 5,000")
    assert s["wrong_rate"] == 1.0 and s["ramdocs_strict"] == 0.0
    assert score(ex, "Final answer(s): 1878 | 1,664")["gold_recall"] == 0.5


def test_directional_and_remove_suspect():
    d = [Detection(0, 2, 1.0, "contradiction", suspects=[0])]
    dirp, _ = build_prompt("q?", PS, "flag_directional", d)
    assert "Passage 0 [LIKELY MISINFORMATION; conflicts with passage(s): 2]" in dirp
    assert "Passage 2 [POTENTIAL CONFLICT with passage(s): 0]" in dirp
    assert "passage 0 likely contains misinformation" in dirp
    # plain flag ignores direction
    assert "MISINFORMATION" not in build_prompt("q?", PS, "flag", d)[0]
    rem, kept = build_prompt("q?", PS, "remove_suspect", d)
    assert kept == [1, 2] and "alpha" not in rem
    # no suspect -> falls back to original remove (drops doc_idx_2)
    assert build_prompt("q?", PS, "remove_suspect", [Detection(0, 2, 1.0)])[1] == [0, 1]
    assert build_prompt("q?", PS, "remove", d)[1] == [0, 1]


def test_pair_level_skips_uncertain():
    from experiments.conflict.metrics import pair_level
    ex = Example("u", "ramdocs", "q", PS, gold_pairs={(0, 1): "contradiction", (0, 2): "uncertain", (1, 2): "ambiguity"})
    m = pair_level([ex], {"u": {(0, 1), (0, 2), (1, 2)}})
    assert m["tp"] == 1 and m["fp"] == 1
