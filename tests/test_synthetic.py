"""SyntheticDetector hits its target precision/recall, nests across grid cells, and the
grid condition runs end to end."""
import json
import random

import pytest

from experiments.conflict import run
from experiments.conflict.data import Example
from experiments.conflict.detectors import (OracleDetector, RandomDetector, SyntheticDetector,
                                            parse_synth_name, synth_name)
from experiments.conflict.metrics import pair_level, question_level
from experiments.conflict.prompts import build_prompt
from tests.conftest import needs_data

RELS = ["contradiction", "ambiguity", "unrelated", "consistent"]


def _examples(n=600, seed=0):
    """Fake RAMDocs-like examples; about a third have no contradiction at all."""
    rng = random.Random(seed)
    xs = []
    for k in range(n):
        m = rng.randint(3, 7)
        clean = rng.random() < 0.35
        gp = {(i, j): rng.choice(RELS[1:] if clean else RELS) for i in range(m) for j in range(i + 1, m)}
        xs.append(Example(f"e{k}", "ramdocs", "q", ["p"] * m, gold_pairs=gp,
                          has_conflict=any(v == "contradiction" for v in gp.values())))
    return xs


def _flags(det, xs):
    return {x.id: {(d.i, d.j) for d in det.detect(x).pairs} for x in xs}


@pytest.mark.parametrize("p,r", [(0.25, 1.0), (0.5, 0.5), (0.75, 0.25), (1.0, 1.0), (1.0, 0.5)])
def test_hits_targets(p, r):
    xs = _examples()
    m = pair_level(xs, _flags(SyntheticDetector(xs, p, r), xs))
    assert abs(m["recall"] - r) < 0.04
    assert abs(m["precision"] - p) < 0.04
    if p == 1.0:
        assert m["fp"] == 0


def test_clean_questions_get_false_flags():
    xs = _examples()
    q = question_level(xs, _flags(SyntheticDetector(xs, 0.5, 1.0), xs))
    assert q["false_positive_rate"] > 0.3          # H3 needs flags on conflict-free questions
    assert question_level(xs, _flags(SyntheticDetector(xs, 1.0, 1.0), xs))["fp"] == 0


def test_pool_restricts_false_flags():
    xs = _examples()
    fl = _flags(SyntheticDetector(xs, 0.5, 1.0, pool="ambiguity"), xs)
    by_id = {x.id: x for x in xs}
    labs = {by_id[e].gold_pairs[pr] for e, s in fl.items() for pr in s}
    assert labs == {"contradiction", "ambiguity"}


def test_cells_are_nested():
    xs = _examples()
    lo_r, hi_r = _flags(SyntheticDetector(xs, 1.0, 0.25), xs), _flags(SyntheticDetector(xs, 1.0, 1.0), xs)
    hi_p, lo_p = _flags(SyntheticDetector(xs, 0.75, 1.0), xs), _flags(SyntheticDetector(xs, 0.25, 1.0), xs)
    assert all(lo_r[e] <= hi_r[e] for e in lo_r)
    assert all(hi_p[e] <= lo_p[e] for e in hi_p)


def test_capped_when_pool_too_small():
    xs = _examples()
    d = SyntheticDetector(xs, 0.01, 1.0, pool="consistent")
    assert d.capped and d.fp_rate == 1.0
    assert pair_level(xs, _flags(d, xs))["precision"] > 0.01


def test_deterministic_and_validates():
    xs = _examples(50)
    assert _flags(SyntheticDetector(xs, 0.5, 0.5), xs) == _flags(SyntheticDetector(xs, 0.5, 0.5), xs)
    assert _flags(SyntheticDetector(xs, 0.5, 0.5), xs) != _flags(SyntheticDetector(xs, 0.5, 0.5, seed=1), xs)
    with pytest.raises(ValueError):
        SyntheticDetector([Example("x", "conflicts", "q", ["a", "b"])], 0.5, 0.5)
    with pytest.raises(ValueError):
        SyntheticDetector(xs, 0.0, 0.5)
    with pytest.raises(ValueError):
        SyntheticDetector(xs, 0.5, 0.5, pool="noise")
    assert parse_synth_name(synth_name(0.25, 1.0, "ambiguity")) == (0.25, 1.0, "ambiguity", None)


def test_control_flags_do_not_name_themselves_in_prompt():
    xs = _examples(5)
    for det in (OracleDetector(), RandomDetector(1.0), SyntheticDetector(xs, 0.5, 1.0)):
        x = next(x for x in xs if det.detect(x).pairs)
        prompt, _ = build_prompt("q?", x.passages, "flag", det.detect(x).pairs)
        assert "oracle" not in prompt and "random" not in prompt and "synth" not in prompt


@needs_data
def test_grid_dry_run(tmp_path):
    args = ["--dataset", "ramdocs", "--n", "8", "--seeds", "0", "--out", str(tmp_path), "--dry-run",
            "--conditions", "plain,grid", "--grid-precision", "0.5,1.0", "--grid-recall", "1.0",
            "--grid-pool", "all,ambiguity"]
    run.main(args)
    cfg = json.loads((tmp_path / "config.json").read_text())
    assert cfg["expanded_conditions"] == ["plain", "synth-p0.5-r1-all+flag", "synth-p1-r1-all+flag",
                                          "synth-p0.5-r1-ambiguity+flag", "synth-p1-r1-ambiguity+flag"]
    rows = [json.loads(l) for l in open(tmp_path / "results.jsonl")]
    assert len(rows) == 8 * 5
    ach = json.loads((tmp_path / "synthetic_achieved.json").read_text())
    assert set(ach) == {"synth-p0.5-r1-all", "synth-p1-r1-all", "synth-p0.5-r1-ambiguity", "synth-p1-r1-ambiguity"}
    assert ach["synth-p1-r1-all"]["pair_level"]["fp"] == 0


def test_oracle_suspects_point_at_misinfo():
    ex = Example("o", "ramdocs", "q", ["a", "b", "c"],
                 gold_pairs={(0, 1): "contradiction", (0, 2): "ambiguity", (1, 2): "contradiction"},
                 meta={"doc_types": ["correct", "misinfo", "correct"]})
    assert [d.suspects for d in OracleDetector().detect(ex).pairs] == [[1], [1]]


def _directed_examples(n=600, seed=1):
    """Like _examples, plus doc_types consistent with the pair labels: each contradiction
    pair has exactly one misinfo side."""
    rng = random.Random(seed)
    xs = []
    for k in range(n):
        m = rng.randint(3, 7)
        types = ["correct"] * m
        bad = rng.randrange(m) if rng.random() < 0.6 else None
        if bad is not None:
            types[bad] = "misinfo"
        gp = {(i, j): ("contradiction" if bad in (i, j) else rng.choice(RELS[1:]))
              for i in range(m) for j in range(i + 1, m)}
        xs.append(Example(f"d{k}", "ramdocs", "q", ["p"] * m, gold_pairs=gp, has_conflict=bad is not None,
                          meta={"doc_types": types}))
    return xs


@pytest.mark.parametrize("a", [1.0, 0.75, 0.5, 0.0])
def test_direction_accuracy_hits_target(a):
    from experiments.conflict.metrics import direction_accuracy
    xs = _directed_examples()
    det = SyntheticDetector(xs, 0.5, 1.0, direction=a)
    acc = direction_accuracy(xs, {x.id: det.detect(x) for x in xs})
    assert abs(acc - a) < 0.04


def test_direction_does_not_change_which_pairs_are_flagged():
    xs = _directed_examples()
    base = _flags(SyntheticDetector(xs, 0.5, 0.5), xs)
    assert base == _flags(SyntheticDetector(xs, 0.5, 0.5, direction=0.5), xs)
    assert all(not d.suspects for x in xs for d in SyntheticDetector(xs, 0.5, 0.5).detect(x).pairs)


def test_direction_draws_nest_and_false_flags_pick_one_side():
    xs = _directed_examples()
    hi, lo = SyntheticDetector(xs, 0.5, 1.0, direction=0.9), SyntheticDetector(xs, 0.5, 1.0, direction=0.6)
    for x in xs:
        types = x.meta["doc_types"]
        for dh, dl in zip(hi.detect(x).pairs, lo.detect(x).pairs):
            true = [k for k in (dh.i, dh.j) if types[k] == "misinfo"]
            if x.gold_pairs[(dh.i, dh.j)] == "contradiction":
                assert dl.suspects != true or dh.suspects == true      # right at 0.6 => right at 0.9
            else:
                assert len(dh.suspects) == 1 and dh.suspects == dl.suspects


def test_synth_name_with_direction():
    assert parse_synth_name(synth_name(0.5, 1.0, "all", 0.75)) == (0.5, 1.0, "all", 0.75)
    assert parse_synth_name(synth_name(0.5, 1.0, "all")) == (0.5, 1.0, "all", None)


@needs_data
def test_directional_grid_dry_run(tmp_path):
    args = ["--dataset", "ramdocs", "--n", "8", "--seeds", "0", "--out", str(tmp_path), "--dry-run",
            "--conditions", "plain,instruct,grid", "--grid-precision", "0.5,1.0", "--grid-recall", "1.0",
            "--grid-direction", "1,0.5", "--grid-mode", "flag_directional,remove_suspect"]
    run.main(args)
    cfg = json.loads((tmp_path / "config.json").read_text())
    assert len(cfg["expanded_conditions"]) == 2 + 2 * 2 * 2
    assert "synth-p0.5-r1-all-d0.5+remove_suspect" in cfg["expanded_conditions"]
    ach = json.loads((tmp_path / "synthetic_achieved.json").read_text())
    assert ach["synth-p1-r1-all-d1"]["direction_accuracy"] in (1.0, None)
    from experiments.conflict import grid_analyze
    grid_analyze.main([str(tmp_path)])
    assert (tmp_path / "grid_analysis.json").exists()


def test_llm_directional_parsing():
    from experiments.conflict.detectors import LLMDetector
    from experiments.conflict.llm import FakeLLM
    reply = json.dumps({"pairs": [
        {"i": 2, "j": 0, "relation": "contradiction", "misinformation": 2},
        {"i": 1, "j": 3, "relation": "contradiction", "misinformation": "both"},
        {"i": 0, "j": 1, "relation": "contradiction", "misinformation": "unclear"},
        {"i": 1, "j": 2, "relation": "ambiguity"},
        {"i": 0, "j": 3, "relation": "contradiction", "misinformation": 7},     # not in the pair
        {"i": 0, "j": 9, "relation": "contradiction"}]})                       # out of range
    ex = Example("x", "ramdocs", "q", ["a", "b", "c", "d"])
    dets = LLMDetector(FakeLLM(lambda p: reply), directional=True).detect(ex).pairs
    assert [(d.i, d.j, d.suspects) for d in dets] == [(0, 2, [2]), (1, 3, [1, 3]), (0, 1, []), (0, 3, [])]
    assert all(d.type == "contradiction" for d in dets)
    plain = LLMDetector(FakeLLM(lambda p: reply)).detect(ex).pairs
    assert plain and all(not d.suspects for d in plain)
    assert LLMDetector(FakeLLM(lambda p: "not json {")).detect(ex).pairs == []


def test_direction_coverage():
    from experiments.conflict.detectors import DetectorOutput, Detection
    from experiments.conflict.metrics import direction_accuracy, direction_coverage
    ex = Example("c", "ramdocs", "q", ["a", "b", "c"], gold_pairs={(0, 1): "contradiction", (0, 2): "contradiction",
                                                                (1, 2): "ambiguity"},
                 meta={"doc_types": ["misinfo", "correct", "correct"]})
    o = {"c": DetectorOutput([Detection(0, 1, 1, suspects=[0]), Detection(0, 2, 1), Detection(1, 2, 1, suspects=[1])])}
    assert direction_coverage([ex], o) == 0.5 and direction_accuracy([ex], o) == 1.0
