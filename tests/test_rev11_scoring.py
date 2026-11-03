"""Eleventh revision, scoring (§20.3): the final-line and cap rules (R9), one-way misinformation (R12), per-seed cells
and contrasts (R11), and the A1 draw."""
import random

from experiments.conflict import export_audit11, rev11_scoring as S


def test_missing_final_line():
    assert not S.missing("Passage 2 says so.\n\nFinal answer(s): Paris")
    assert not S.missing("final answer(s) : unknown")
    assert not S.missing('All Correct Answers: ["July 10, 1850"]')     # MADAM list
    assert S.missing("Passage 3 is reliable, so the primary medium is painting.\n\nFinal answer")   # cut at the cap
    assert S.missing("") and S.missing(None)


def test_effective_cap():
    assert S.DEFAULT_CAP == 256
    assert S.cap("gpt-4o-mini", "plain") == 256 and S.cap("qwen2.5:7b-instruct", "oracle+flag") == 256
    assert S.cap("gpt-6-luna", "plain") == S.cap("claude-haiku-5-5", "hint") == 4096
    assert S.cap("gpt-4o-mini", "madam", 11) is None and S.cap("gpt-4.1", "astute_multi", 2) is None


def test_tally_counts_missing_and_cap():
    rows = [{"condition": "plain", "response": "x", "completion_tokens": 256, "has_conflict": True, "calls": 1},
            {"condition": "plain", "response": "Final answer(s): a", "completion_tokens": 40, "has_conflict": False, "calls": 1},
            {"condition": "plain", "response": "y", "completion_tokens": 100, "has_conflict": None, "calls": 1},
            {"condition": "madam", "response": "z", "completion_tokens": 900, "has_conflict": True, "calls": 9}]
    t = S.tally(rows, "gpt-4o-mini")
    p = t["plain"]
    assert (p["n"], p["missing"], p["at_cap"], p["both"]) == (3, 2, 1, 1)
    assert p["missing_conflict_share"] == 1.0 and p["missing_clean_share"] == 0.0
    assert t["madam"]["at_cap_share"] is None and t["madam"]["missing_share"] == 1.0


def test_oneway_misinformation():
    assert S.oneway_wrong(["January 1, 1877"], ["1877"]) == 1.0
    assert S.oneway_wrong(["1877"], ["January 1, 1877"]) == 0.0      # the bidirectional matcher counts this one
    assert S.oneway_wrong(["Vigo University"], ["Vigo"]) == 1.0
    assert S.oneway_wrong(["Vigorous"], ["Vigo"]) == 0.0             # whole-word spans only
    assert S.oneway_wrong([], ["x"]) == 0.0


def _row(i, cond, seed, correct, wrong=0.0, abstain=0.0):
    return {"id": i, "condition": cond, "seed": seed, "has_conflict": True, "ramdocs_strict_oneway": correct,
            "wrong_rate": wrong, "misinfo_oneway": wrong, "abstain": abstain}


def test_seed_cells_and_contrast():
    three = [_row(q, c, s, float(c == "b" and s == 1)) for q in ("q1", "q2") for c in ("a", "b") for s in (0, 1, 2)]
    one = [_row(q, c, 0, 1.0 if c == "b" else 0.0) for q in ("q1", "q2") for c in ("a", "b")]
    cells = [("three", "CB", three, "a", "b"), ("one", "CB", one, "a", "b")]
    assert S._arm_seeds(three, "a", "b") == [0, 1, 2] and S._arm_seeds(one, "a", "b") == [0]
    s1 = S.seed_cells(cells, 1)
    assert len(s1) == 1 and s1[0][1] == {"q1": 1.0, "q2": 1.0}
    assert len(S.seed_cells(cells, 1, fixed=True)) == 2 and len(S.seed_cells(cells, 0)) == 2
    c = S.contrast(three, "a", "b", seed=1)
    assert c["correct"] == 100.0 and c["n"] == 2 and S.contrast(three, "a", "b", seed=0)["correct"] == 0.0
    assert abs(S.contrast(three, "a", "b")["correct"] - 100 / 3) < 1e-9


def test_selective_contrast_is_a_ratio():
    rows = [_row("q1", "a", 0, 1.0), _row("q2", "a", 0, 0.0),
            _row("q1", "b", 0, 1.0), _row("q2", "b", 0, 0.0, abstain=1.0)]
    c = S.contrast(rows, "a", "b")
    assert c["selective"] == 50.0 and c["abstain"] == 50.0      # 1/1 answered vs 1/2


def test_a1_draw_strata_and_one_per_question():
    pop = []
    for arm in ("x", "y"):
        for q in range(30):
            for seed in range(3):
                dis = q < 12
                pop.append({"dataset": "d", "id": f"q{q}", "run": "r", "condition": arm, "seed": seed,
                            "response_key": f"{arm}{q}{seed}", "matcher": 1.0, "judge_strict": 0.0 if dis else 1.0})
    out = export_audit11.draw(pop, random.Random(0), arms=("x", "y"), n_dis=5, n_agr=4)
    assert len(out) == 18
    for arm in ("x", "y"):
        mine = [p for p in out if p["condition"] == arm]
        assert len({p["id"] for p in mine}) == len(mine) == 9
        assert sum(p["stratum"] == "disagree" for p in mine) == 5
        assert all((p["matcher"] != p["judge_strict"]) == (p["stratum"] == "disagree") for p in mine)
    again = export_audit11.draw(pop, random.Random(0), arms=("x", "y"), n_dis=5, n_agr=4)
    assert [p["response_key"] for p in again] == [p["response_key"] for p in out]


def test_a1_draw_short_pool_takes_what_exists():
    pop = [{"dataset": "d", "id": f"q{q}", "run": "r", "condition": "x", "seed": 0, "response_key": str(q),
            "matcher": 1.0, "judge_strict": float(q > 1)} for q in range(6)]
    out = export_audit11.draw(pop, random.Random(0), arms=("x",), n_dis=5, n_agr=2)
    assert sum(p["stratum"] == "disagree" for p in out) == 2 and len(out) == 4
