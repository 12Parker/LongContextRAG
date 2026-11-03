"""PR§21 code: stated-confidence flags, the object swap, sentence-pair NLI selection, probe parsers."""
from experiments.conflict.detectors import Detection, NLISentenceDetector
from experiments.conflict.prepare_cb_swap import swap_item, usable
from experiments.conflict.probe21 import parse_choice, parse_prob
from experiments.conflict.prompts import build_prompt


def test_confidence_flag_renders_on_suspect_only():
    d = [Detection(0, 2, 1.0, "contradiction", [2])]
    p, kept = build_prompt("Q?", ["p0", "p1", "p2"], "flag_directional_c55", d, skeleton="single")
    assert "Passage 2 [LIKELY MISINFORMATION (detector confidence: 55%); conflicts with passage(s): 0]" in p
    assert "passage 2 likely contains misinformation (detector confidence: 55%)" in p
    assert "Passage 0 [POTENTIAL CONFLICT with passage(s): 2]" in p
    # without the confidence the text is flag_directional's
    base, _ = build_prompt("Q?", ["p0", "p1", "p2"], "flag_directional", d, skeleton="single")
    assert p.replace(" (detector confidence: 55%)", "") == base
    # no flag: identical to plain, so clean questions are cache hits
    assert (build_prompt("Q?", ["p0"], "flag_directional_c95", [], skeleton="single")[0]
            == build_prompt("Q?", ["p0"], "plain", [], skeleton="single")[0])


def test_swap_rules():
    assert usable("He was born in India.", "He was born in Slovenia.", "India", "Slovenia")
    assert not usable("An Indian bishop born in India.", "He was born in Slovenia.", "India", "Slovenia")  # prefix
    assert not usable("Born in India.", "Not India but Slovenia.", "India", "Slovenia")                    # gold in misinfo
    r = {"id": "x", "has_conflict": True, "gold": "India", "wrong": "Slovenia", "types": ["noise", "correct", "misinfo"],
         "passages": ["n", "Born in india, he lived in India.", "Born in Slovenia."]}
    s = swap_item(r)
    assert s["passages"][1] == "Born in Slovenia, he lived in Slovenia." and s["passages"][2] == "Born in India."
    assert s["types"] == ["noise", "misinfo", "correct"] and s["style_types"] == r["types"]
    assert swap_item({**r, "has_conflict": False}) is None


def test_sentence_selection_without_model():
    d = NLISentenceDetector.__new__(NLISentenceDetector)
    d.k, d.sent_chars = 2, 10
    sents = d._sentences("Alpha beta. Who founded Acme? Acme was founded by Bob!\nUnrelated line here.", "Who founded Acme")
    assert sents == ["Who founde", "Acme was f"]   # top-2 by overlap, original order, cut to 10 characters


def test_probe_parsers():
    assert parse_choice("Reasoning...\nAnswer: B") == "B"
    assert parse_choice("Answer: **A**") == "A"
    assert parse_choice("I cannot tell") is None
    assert parse_prob("...\nProbability false: 85") == 85
    assert parse_prob("Probability false: 140") == 100
    assert parse_prob("no number") is None
