from collections import Counter

from experiments.conflict import data
from tests.conftest import needs_data


@needs_data
def test_ramdocs():
    xs = data.load_ramdocs()
    assert len(xs) == 500
    labs = Counter(v for x in xs for v in x.gold_pairs.values())
    assert set(labs) <= {"contradiction", "ambiguity", "consistent", "unrelated"}
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
