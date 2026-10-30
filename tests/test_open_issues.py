"""Splits, hand labels, pricing with cached tokens, OpenAI-compatible endpoints, the judge,
WikiContradict scoring and the labelling export."""
import csv
import json
from pathlib import Path

import pytest

from experiments.conflict import data, judge, llm as llm_mod
from experiments.conflict.data import Example
from experiments.conflict.metrics import score
from tests.conftest import needs_data

SPLITS = Path(__file__).resolve().parents[1] / "experiments" / "conflict" / "splits"


@needs_data
def test_splits_partition_ramdocs_and_match_pilot_sample():
    pilot = [l.strip() for l in open(SPLITS / "ramdocs_pilot.txt") if l.strip()]
    hold = [l.strip() for l in open(SPLITS / "ramdocs_holdout.txt") if l.strip()]
    assert len(pilot) == 200 and len(hold) == 300 and not set(pilot) & set(hold)
    assert set(pilot) | set(hold) == {x.id for x in data.load_ramdocs()}
    # the pilot file is exactly the n=200, seed=0 sample the pilots ran on
    assert pilot == [x.id for x in data.load_ramdocs(n=200, seed=0)]
    assert [x.id for x in data.filter_ids(data.load_ramdocs(), SPLITS / "ramdocs_holdout.txt")] == sorted(hold)


def test_filter_ids_refuses_missing(tmp_path):
    f = tmp_path / "ids.txt"; f.write_text("a\nzzz\n")
    with pytest.raises(ValueError):
        data.filter_ids([Example("a", "t", "q", ["p"])], f)


@needs_data
def test_hand_labels(tmp_path):
    from experiments.conflict import export_labelling
    export_labelling.main(["--out", str(tmp_path)])
    rows = list(csv.DictReader(open(tmp_path / "ramdocs_suspect_docs.csv")))
    assert len(rows) == 163
    rp = next(r for r in rows if r["question"].startswith("What sport is Ronald Powell"))
    rp["label"] = "misinformation"
    other = next(r for r in rows if r["question_id"] != rp["question_id"])
    other["label"] = "incidental"
    with open(tmp_path / "lab.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    hand = {x.id: x for x in data.load_ramdocs(label_fix="hand", hand_labels=tmp_path / "lab.csv")}
    x = hand[rp["question_id"]]
    assert x.has_conflict is True and x.meta["doc_types"][int(rp["doc_index"])] == "misinfo"
    assert int(other["doc_index"]) not in hand[other["question_id"]].meta["suspect_docs"]
    rows[0]["label"] = "maybe"
    with open(tmp_path / "bad.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with pytest.raises(ValueError):
        data.load_ramdocs(label_fix="hand", hand_labels=tmp_path / "bad.csv")


def test_price_with_cached_tokens_and_registered_models():
    assert llm_mod.price("gpt-4o-mini", 1_000_000, 0) == pytest.approx(0.15)
    assert llm_mod.price("gpt-4o-mini", 1_000_000, 1_000_000, cached_pt=1_000_000) == pytest.approx(0.075 + 0.60)
    assert "UNVERIFIED" not in llm_mod.PRICES_CHECKED
    llm_mod.register_price("open/model", 0.2, 0.4)
    assert llm_mod.price("open/model", 1_000_000, 1_000_000) == pytest.approx(0.6)
    with pytest.raises(KeyError):
        llm_mod.price("unknown/model", 1, 1)


def test_base_url_needs_key_and_changes_cache_key(monkeypatch, tmp_path):
    monkeypatch.delenv("MY_KEY", raising=False)
    llm_mod.register_price("open/model", 0.2, 0.4)
    with pytest.raises(RuntimeError):
        llm_mod.OpenAILLM("open/model", base_url="http://localhost:9/v1", api_key_env="MY_KEY",
                          cache_path=tmp_path / "c.sqlite")
    with pytest.raises(KeyError):        # no price -> fail at construction, not after a call
        llm_mod.OpenAILLM("unpriced/model", cache_path=tmp_path / "c.sqlite")


def test_wikicontradict_requires_both_answers():
    ex = Example("w", "wikicontradict", "q", ["a", "b"], gold_answers=["Apomorphine", "Aporphine"], has_conflict=True)
    assert score(ex, "Final answer(s): Apomorphine | Aporphine")["primary"] == 1.0
    s = score(ex, "Final answer(s): Aporphine")
    assert s["primary"] == 0.0 and s["gold_recall"] == 0.5


def test_judge_kappa_and_rows():
    assert judge.cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert judge.cohen_kappa([1, 1, 0, 0], [0, 0, 1, 1]) == -1.0
    assert judge.cohen_kappa([None], [1]) is None
    ex = {"e": Example("e", "conflicts", "Is X good?", ["p"], gold_answers=[], has_conflict=True,
                       conflict_type="Conflicting opinions and research outcomes")}
    fake = llm_mod.FakeLLM(lambda p: '{"correct": null, "handles_conflict": 1, "reason": "ok"}')
    out = judge.judge_rows([{"id": "e", "condition": "plain", "seed": 0, "response": "Some say yes, some no."}], ex, fake)
    assert out[0]["handles_conflict"] == 1 and out[0]["correct"] is None
    bad = judge.judge_rows([{"id": "e", "condition": "plain", "seed": 0, "response": "x"}], ex,
                           llm_mod.FakeLLM(lambda p: "not json"))
    assert bad[0]["correct"] is None and bad[0]["reason"].startswith("unparseable")


@needs_data
def test_judge_end_to_end_dry_run(tmp_path):
    from experiments.conflict import run
    run.main(["--dataset", "ramdocs", "--n", "6", "--seeds", "0", "--out", str(tmp_path), "--dry-run",
              "--conditions", "plain,instruct_v2"])
    judge.main([str(tmp_path), "--dry-run", "--per-stratum", "2", "--human", "3"])
    js = [json.loads(l) for l in open(tmp_path / "judgments.jsonl")]
    assert js and all(j["correct"] == 1 for j in js)
    hum = list(csv.DictReader(open(tmp_path / "human_labels.csv")))
    assert len(hum) == 3
    for h in hum:
        h["human_correct"] = "1"
    with open(tmp_path / "filled.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hum[0])); w.writeheader(); w.writerows(hum)
    judge.main([str(tmp_path), "--agreement", str(tmp_path / "filled.csv")])
    rep = json.loads((tmp_path / "judge_agreement.json").read_text())
    assert rep["correct"]["n"] == 3 and rep["correct"]["kappa"] == 1.0


@needs_data
def test_judge_refuses_same_model(tmp_path):
    from experiments.conflict import run
    run.main(["--dataset", "ramdocs", "--n", "2", "--seeds", "0", "--out", str(tmp_path), "--dry-run",
              "--conditions", "plain"])
    with pytest.raises(SystemExit):
        judge.main([str(tmp_path), "--model", "gpt-4o-mini"])


def test_run_price_flags_must_pair():
    from experiments.conflict import run
    with pytest.raises(SystemExit):
        run.main(["--dataset", "ramdocs", "--out", "/tmp/never", "--conditions", "plain", "--price-input", "0.1"])


def test_madam_answers_keep_inner_commas():
    from experiments.conflict.metrics import extract_answers
    assert extract_answers('All Correct Answers: ["1902", "July 10, 1850"]. Explanation: x') == ["1902", "July 10, 1850"]
    assert extract_answers("All Correct Answers: ['a, b', 'c']") == ["a, b", "c"]
    assert extract_answers("All Correct Answers: [1963, 1956]") == ["1963", "1956"]
    ex = Example("m", "ramdocs", "q", ["p"], gold_answers=["1902"], wrong_answers=["1850"])
    # the fragment "1850" used to match the wrong answer; the whole date does not
    assert score(ex, 'All Correct Answers: ["1902", "July 10, 1850"].')["wrong_rate"] == 1.0
    assert score(ex, 'All Correct Answers: ["1902", "July 10, 1851"].')["wrong_rate"] == 0.0
