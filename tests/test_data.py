from collections import Counter

from experiments.conflict import data
from tests.conftest import needs_data


@needs_data
def test_ramdocs():
    xs = data.load_ramdocs()
    assert len(xs) == 500
    labs = Counter(v for x in xs for v in x.gold_pairs.values())
    assert set(labs) <= {"contradiction", "ambiguity", "consistent", "unrelated", "uncertain"}
    assert labs["contradiction"] > 0
    # every question with a wrong answer has at least one contradiction pair
    for x in xs:
        if any(t == "misinfo" for t in x.meta["doc_types"]) and any(t == "correct" for t in x.meta["doc_types"]):
            assert x.has_conflict


@needs_data
def test_conflicts():
    xs = data.load_conflicts()
    assert len(xs) == 458
    c = Counter(x.conflict_type for x in xs)
    assert c["No conflict"] == 161
    assert sum(x.has_conflict for x in xs) == 115 + 62 + 5
    assert all(x.gold_answers for x in xs if x.conflict_type == "No conflict")


@needs_data
def test_subsample_is_deterministic():
    a = [x.id for x in data.load_ramdocs(n=20, seed=3)]
    assert a == [x.id for x in data.load_ramdocs(n=20, seed=3)] and len(a) == 20


@needs_data
def test_ramdocs_label_fixes():
    by = {f: {x.id: x for x in data.load_ramdocs(label_fix=f)} for f in data.RAMDOCS_LABEL_FIXES}
    none, exc, con = by["none"], by["exclude"], by["contains"]
    # Ronald Powell: a document typed "correct" says he is "a renowned chess player"
    rp = next(x for x in none.values() if x.question.startswith("What sport is Ronald Powell"))
    assert rp.has_conflict is False
    assert exc[rp.id].has_conflict is None and exc[rp.id].meta["suspect_docs"]
    assert con[rp.id].has_conflict is True
    k = exc[rp.id].meta["suspect_docs"][0]
    assert con[rp.id].meta["doc_types"][k] == "misinfo" and none[rp.id].meta["doc_types"][k] == "correct"
    # exclude never changes a question that already had a contradiction, nor its contradiction pairs
    for i, x in none.items():
        if x.has_conflict:
            assert exc[i].has_conflict is True
            assert {p for p, v in x.gold_pairs.items() if v == "contradiction"} == \
                   {p for p, v in exc[i].gold_pairs.items() if v == "contradiction"}
    assert sum(x.has_conflict is None for x in exc.values()) == 55
