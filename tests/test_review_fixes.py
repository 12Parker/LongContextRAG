"""Latency recording, flag styles, the second instruct wording, document-level metrics,
document-level placement and the Holm-based break-even definition."""
import json

import pytest

from experiments.conflict import llm as llm_mod
from experiments.conflict import place_detector, run
from experiments.conflict.data import Example
from experiments.conflict.detectors import Detection, DetectorOutput
from experiments.conflict.grid_analyze import thresholds
from experiments.conflict.metrics import doc_level, marked_docs
from experiments.conflict.prompts import INSTRUCTION, INSTRUCTION_V2, build_prompt
from tests.conftest import needs_data

PS = ["alpha text", "beta text", "gamma text"]


@needs_data
def test_generate_s_is_api_latency(tmp_path, monkeypatch):
    real = llm_mod.FakeLLM
    monkeypatch.setattr(llm_mod, "FakeLLM", lambda responder=None: real(responder, latency_s=1.5))
    run.main(["--dataset", "ramdocs", "--n", "4", "--seeds", "0", "--out", str(tmp_path), "--dry-run",
              "--conditions", "plain,oracle+flag,madam"])
    rows = [json.loads(l) for l in open(tmp_path / "results.jsonl")]
    for r in rows:
        assert r["generate_s"] == pytest.approx(1.5 * r["calls"])   # API latency, summed over MADAM calls
        assert "wall_s" in r and "latency_s" not in r and r["flag_style"] == "uniform"


def test_uniform_style_hides_detector_identity():
    a = [Detection(0, 2, 0.41, "factual")]           # CAAG-like
    b = [Detection(0, 2, 1.0, "contradiction")]      # oracle/LLM-like
    assert build_prompt("q?", PS, "flag", a)[0] == build_prompt("q?", PS, "flag", b)[0]
    assert "Passages 0 and 2: possible contradiction" in build_prompt("q?", PS, "flag", a)[0]
    orig = build_prompt("q?", PS, "flag", a, style="original")[0]
    assert "factual conflict (confidence: 0.41)" in orig
    d = [Detection(0, 2, 0.41, "factual", suspects=[2])]
    assert build_prompt("q?", PS, "flag_directional", d)[0] == \
        build_prompt("q?", PS, "flag_directional", [Detection(0, 2, 1.0, "contradiction", [2])])[0]
    with pytest.raises(ValueError):
        build_prompt("q?", PS, "flag", a, style="fancy")


def test_instruct_v2():
    v1, v2 = build_prompt("q?", PS, "instruct")[0], build_prompt("q?", PS, "instruct_v2")[0]
    assert INSTRUCTION in v1 and INSTRUCTION_V2 in v2 and INSTRUCTION not in v2
    assert "report each one" not in v2
    f2 = build_prompt("q?", PS, "flag_instruct_v2", [Detection(0, 1, 1.0)])[0]
    assert INSTRUCTION_V2 in f2 and "POTENTIAL CONFLICT" in f2
    assert run.parse_condition("instruct_v2") == ("none", "instruct_v2")


def _doc_examples():
    conf = Example("c", "ramdocs", "q", PS, has_conflict=True, gold_pairs={},
                   meta={"doc_types": ["correct", "misinfo", "correct"]})
    clean = Example("k", "ramdocs", "q", PS, has_conflict=False, gold_pairs={},
                    meta={"doc_types": ["correct", "correct", "noise"]})
    return [conf, clean]


def test_doc_level_and_marked_docs():
    xs = _doc_examples()
    dets = {"c": DetectorOutput([Detection(0, 1, 1, suspects=[1]), Detection(1, 2, 1)]),
            "k": DetectorOutput([Detection(0, 1, 1, suspects=[0])])}
    assert marked_docs(dets["c"].pairs, "remove_suspect") == {1, 2}     # no suspect -> doc_idx_2
    assert marked_docs(dets["c"].pairs, "flag_directional") == {1}
    assert marked_docs(dets["c"].pairs, "remove") == {1, 2}
    d = doc_level(xs, dets, "remove_suspect")
    assert d["misinfo_marked"] == 1.0 and d["correct_marked"] == pytest.approx(2 / 4)
    assert d["conflict_q_all_misinfo_marked"] == 1.0 and d["conflict_q_any_correct_marked"] == 1.0
    assert d["clean_q_any_correct_marked"] == 1.0
    assert doc_level(xs, dets, "flag_directional")["conflict_q_any_correct_marked"] == 0.0
    with pytest.raises(ValueError):
        marked_docs([], "flag")


def test_doc_level_skips_excluded_suspects():
    x = Example("s", "ramdocs", "q", PS, has_conflict=None, gold_pairs={},
                meta={"doc_types": ["correct", "correct", "correct"], "suspect_docs": [1], "label_fix": "exclude"})
    d = doc_level([x], {"s": DetectorOutput([Detection(0, 1, 1, suspects=[1])])}, "remove_suspect")
    assert d["correct_marked"] == 0.0


def test_idw_exact_hit_and_weighting():
    pred, order = place_detector.idw((0.5, 0.5), [((0.5, 0.5), 0.3), ((0, 0), -1.0), ((1, 1), 1.0)], k=3)
    assert pred == pytest.approx(0.3, abs=1e-4) and order[0] == 0
    pred, _ = place_detector.idw((0.25,), [((0.0,), 0.0), ((0.5,), 1.0)], k=2)
    assert pred == pytest.approx(0.5)


def _cell(diff, p_holm):
    return {"diff": diff, "p_holm": p_holm}


def test_break_even_requires_significant_gain_at_and_above():
    ps, rs = [0.25, 0.5, 0.75, 1.0], [1.0]
    tab = {"0.25,1": _cell(-0.2, 0.001), "0.5,1": _cell(+0.05, 0.3),
           "0.75,1": _cell(+0.07, 0.01), "1,1": _cell(+0.1, 0.001)}
    th = thresholds(tab, ps, rs)["1"]
    assert th["break_even_precision"] == 0.75        # 0.5 is positive but not significant
    assert th["harm_free_precision"] == 0.5          # the weaker criterion would say 0.5
    tab["1,1"] = _cell(+0.1, 0.2)                    # top cell not significant -> no break-even
    assert thresholds(tab, ps, rs)["1"]["break_even_precision"] is None


@needs_data
def test_placement_end_to_end(tmp_path):
    g, d = tmp_path / "g", tmp_path / "d"
    run.main(["--dataset", "ramdocs", "--n", "10", "--seeds", "0", "--dry-run", "--out", str(g),
              "--conditions", "plain,grid", "--grid-precision", "0.5,1", "--grid-recall", "0.5,1",
              "--grid-direction", "1,0.5", "--grid-mode", "flag_directional,remove_suspect"])
    run.main(["--dataset", "ramdocs", "--n", "10", "--seeds", "0", "--dry-run", "--out", str(d),
              "--conditions", "plain,oracle+flag_directional,oracle+remove_suspect"])
    q = json.loads((d / "detector_quality.json").read_text())["oracle"]
    assert q["doc_level"]["remove_suspect"]["correct_marked"] == 0.0
    place_detector.main([str(g), str(d), "--detector", "oracle"])
    out = json.loads((d / "placement.oracle.doc.json").read_text())
    assert set(out["modes"]) == {"flag_directional", "remove_suspect"}
    assert out["modes"]["remove_suspect"]["conflict"]["location"][1] == 0.0
