"""assets11 (PR§20 macros, tables, figure): macro names are letters only and unique, write11 writes its files from a
minimal synthetic R, and the full numbers.tex stays free of duplicate names on the real data (skipped without it).
The numbers below are synthetic fixtures for the formatter, not results."""
import json
import re
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

from experiments.conflict import assets11 as a11  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _pt(est, lo, hi, **kw):
    return {"est": est, "ci": [lo, hi], **kw}


def _setting(policy, bench, gen, cov0, sel0, cov1, sel1):
    def side(cov, sel):
        return {"conflict": {"coverage": _pt(cov, cov - 2, cov + 2), "selective": _pt(sel, sel - 3, sel + 3),
                             "misinfo": _pt(30.0, 25.0, 35.0), "strict": _pt(sel * cov / 100, 1.0, 99.0)}}
    d = lambda x: {"diff": x, "ci": [x - 1, x + 1]}
    return {"policy": policy, "bench": bench, "gen": gen, "run": "r",
            "points": {"plain": side(cov0, sel0), "oracle+flag": side(cov1, sel1)},
            "vs_plain": {"oracle+flag": {"conflict": {"coverage": d(cov1 - cov0), "selective": d(sel1 - sel0),
                                                      "misinfo": d(2.0), "strict": d(-0.5)}}}}


def minimal_R():
    r2 = {"gpt-4o-mini": {"n_known": 28, "n_unknown": 272, "n_conflict": 300, "closedbook_seeds": [0, 1, 2],
                          **{b: {"n_known_flags": 28, "n_unknown_flags": 264, "known": 0.85, "known_ci": [0.7, 0.97],
                                 "unknown": 0.67, "unknown_ci": [0.61, 0.73], "diff": 0.18, "diff_ci": [0.03, 0.32]}
                             for b in ("single_side", "direction")}},
          "note": "x", "registered": "single_side"}
    r3 = {"gpt-4o-mini": {"single_side": 0.69, "single_ci": [0.63, 0.74], "single_n": 1752, "single_k": 1206,
                          "current": 0.69, "current_ci": [0.63, 0.74], "current_n": 1752,
                          "verdict_mix": {"both": 0, "unclear": 0}},
          "claude-sonnet-5-5": {"single_side": 0.94, "single_ci": [0.9, 0.97], "single_n": 624, "single_k": 585,
                                "current": 0.78, "current_ci": [0.73, 0.83], "current_n": 753,
                                "verdict_mix": {"both": 42, "unclear": 87}}}
    r9 = {"n_runs": 2, "n_cells": 3, "n_cells_above_1pct": 2,
          "totals": {"n": 1000, "missing": 30, "missing_share": 0.03, "at_cap_share_of_missing_single_call": 0.5},
          "above_1pct": [["run_a", "oracle+flag_instruct_v2", 300, 0.2, 0.25], ["run_b", "plain", 400, 0.02, None]],
          "runs": {"run_a": {"model": "gpt-4.1", "cap": 256}, "run_b": {"model": "qwen2.5:7b-instruct", "cap": 256}},
          "by_model": {"gpt-4.1": {"missing_share": 0.024}}}
    r16 = {"settings": {"CB:gpt-4o-mini": _setting("multi", "ConflictBank", "gpt-4o-mini", 99.0, 21.0, 99.0, 13.0),
                        "CBn:gpt-4o-mini": _setting("minimal", "ConflictBank", "gpt-4o-mini", 97.0, 36.0, 68.0, 25.0),
                        "CBs:gpt-4o-mini": _setting("single", "ConflictBank", "gpt-4o-mini", 93.0, 55.0, 59.0, 55.0),
                        "RG:gpt-4o-mini": _setting("single", "RAGuard", "gpt-4o-mini", 68.0, 54.0, 68.0, 59.0),
                        "CBf:gpt-4o-mini": _setting("forced", "ConflictBank", "gpt-4o-mini", 100.0, 55.0, 100.0, 55.0)}}
    pi = [round(0.01 * k, 2) for k in range(1, 51)]
    cnt = {"n": 2, "n_neg": [1] * 50, "n_sig_neg": [1] * 50, "n_sig_pos": [0] * 50}
    r18 = {"pi": pi, "summary": {"all": cnt, "cb_distractors": cnt},
           "groups": {"cb_distractors": {"settings": ["CB:gpt-4o-mini", "CBs:gpt-4.1"]}},
           "settings": {"CB:gpt-4o-mini": {"group": "cb_distractors", "regime": "crossing", "K": {"est": 20.0},
                                           "C": {"est": -1.0}, "breakeven": 0.05, "breakeven_ci": [0.0, 0.18],
                                           "breakeven_ci_upper_capped": False, "bench": "ConflictBank",
                                           "policy": "multi", "gen": "gpt-4o-mini"},
                        "CBs:gpt-4.1": {"group": "cb_distractors", "regime": "hurts_all", "K": {"est": -2.3},
                                        "C": {"est": -3.0}, "bench": "ConflictBank", "policy": "single",
                                        "gen": "gpt-4.1"}}}
    rate = lambda r, n: {"rate": r, "n": n, "ci": [r - 0.1, r + 0.1]}
    r4 = {"gpt-4o-mini": {"n_known": 28, "n_conflict": 300,
                          **{f: {"known": rate(0.8, 5), "unknown": rate(0.5, 70), "all": rate(0.57, 75), "diff": 0.3,
                                 "diff_ci": [0.0, 0.5]} for f in ("f_R", "f_W")}}}
    r13 = {"n_parsed": 247, "n_pages": 494, "reversed_pages": 60, "reversed_share": 60 / 494,
           "both_confirmed": 92, "both_confirmed_share": 92 / 247, "no_confirmed_pair_share": 155 / 247,
           "correct_page": {"yes": 0.684, "no": 0.069, "unclear": 0.247},
           "misinfo_page": {"yes": 0.551, "no": 0.174, "unclear": 0.275}}
    r19 = {"rho": -0.081, "p_perm": 0.75, "p_perm_one_sided_negative": 0.38, "n_perm": 10000, "n_cells": 18,
           "n_questions": 127, "bca_ci": [-0.32, 0.014], "bca_z0": 0.52, "percentile_ci_stored": [-0.558, -0.017],
           "percentile_ci_5000": [-0.571, -0.017]}
    h31x = {"gpt-4o-mini": {"breakeven": 0.52, "be_low": 0.43, "be_high": 0.6, "follow_right": 0.57,
                            "follow_right_ci": [0.46, 0.67], "follow_wrong": 0.51, "follow_wrong_ci": [0.4, 0.62],
                            "right": 29.4, "wrong": -31.2, "plain_right_arm": 49.9, "plain_wrong_arm": 52.5}}
    return {"rev": {"r11": {"R2": r2, "R3": r3, "R4": r4, "R9": r9, "R13": r13, "R16": r16, "R18": r18, "R19": r19},
                    "H31x": h31x,
                    "H31a": {"gpt-4o-mini: right-flag effect (conflict)": {"n": 159}},
                    "H31b": {"gpt-4o-mini: wrong-flag effect (conflict)": {"n": 141}},
                    "r2": {"CBs:gpt-4o-mini": {"closedbook": {"conflict": 0.1}},
                           "CBs:gpt-4.1": {"closedbook": {"conflict": 0.2}}},
                    "judge19": {"agreement": {"answer_key": {"n": 10, "agree": 0.9, "judge_only": 0.08,
                                                             "matcher_only": 0.02}}}}}


def _collect(R):
    out = []
    a11.numbers11(R, lambda n, v: out.append((n, v)))
    return out


def test_numbers11_names_are_letters_only_and_unique():
    m = _collect(minimal_R())
    names = [n for n, _ in m]
    assert names and all(re.fullmatch(r"[A-Za-z]+", n) for n in names)
    assert len(names) == len(set(names))
    d = dict(m)
    assert d["priorSsMiniDiff"] == "$+$18.0" and d["priorSsMiniDiffCI"] == "[$+$3.0, $+$32.0]"
    assert d["ssPrecMaxDet"] == "Claude Sonnet" and d["ssPrecOtherMax"] == "94.0"
    assert d["cbClosedMax"] == "20.0" and d["cbClosedMaxGen"] == "gpt-4.1"
    assert d["deferPlainMini"] == f"{(49.9 * 159 + 52.5 * 141) / 300:.1f}"
    assert d["missTopRun"] == r"\texttt{run\_a}" and d["missAboveQwen"] == 1
    assert d["rcSingleAbstUp"] == 1 and d["rcMultiSelDown"] == 1 and d["rcForcedSel"] == 0
    assert d["pvBeN"] == 1 and d["pvHurtsAll"] == 1 and d["pvFiveSigNeg"] == 1
    # review fixes: no second CI for the all-question follow rates (the paper's \rfolRight<G>CI); R13 in whole
    # percentages like \hcLab*; the H4b percentile interval the paper already prints (2,000 draws)
    assert "prFolRAllMini" not in d and "prFolWAllMiniCI" not in d and d["prFolRAllMiniN"] == 75
    assert d["stanceCorrectYes"] == "68" and d["stanceBoth"] == "37" and d["stanceReversed"] == "12"
    assert d["hFourBPctCI"] == "[$-$0.56, $-$0.02]" and d["hFourBPctSameCI"] == "[$-$0.57, $-$0.02]"
    assert d["hFourBBelow"] == "70"


def test_deference_caption_names_generators_with_their_own_plain_draw(tmp_path):
    R = minimal_R()
    R["rev"]["r2"]["CBs:gpt-4o-mini"]["plain"] = {a11.PRIMARY: {"conflict": 0.55}}   # differs from the H31 run's 51.1
    a11.tab_deference11(R, tmp_path)
    t = (tmp_path / "tab_deference.tex").read_text()
    assert "For gpt-4o-mini the H31 run drew its own no-flag answers" in t
    R["rev"]["r2"]["CBs:gpt-4o-mini"]["plain"] = {a11.PRIMARY: {"conflict": 0.511}}
    a11.tab_deference11(R, tmp_path)
    assert "drew its own" not in (tmp_path / "tab_deference.tex").read_text()


def test_numbers11_tolerates_missing_results():
    assert _collect({"rev": {}}) == []
    assert _collect({}) == []


def test_adder_rejects_bad_and_repeated_names():
    A = a11._Adder(lambda n, v: None)
    A("okName", 1)
    with pytest.raises(ValueError):
        A("bad1", 1)
    with pytest.raises(ValueError):
        A("okName", 2)


def test_catalogue_lists_every_macro_once():
    R = minimal_R()
    cat = a11.catalogue11(R)
    names = [n for n, _ in _collect(R)]
    listed = re.findall(r"^- `\\([A-Za-z]+)` = ", cat, flags=re.M)
    assert listed == names


def test_write11_writes_expected_files(tmp_path):
    a11.write11(minimal_R(), tmp_path)
    for f, label in (("tab_deference.tex", "tab:deference"), ("tab_prior.tex", "tab:prior"),
                     ("tab_missing.tex", "tab:missing"), ("tab_riskcov.tex", "tab:riskcov"),
                     ("tab_prev11.tex", "tab:prev11")):
        t = (tmp_path / f).read_text()
        assert r"\toprule" in t and r"\bottomrule" in t and rf"\label{{{label}}}" in t
        assert t.count(r"\begin{tabular}") == t.count(r"\end{tabular}") == 1
    assert (tmp_path / "fig_riskcov.pdf").stat().st_size > 1000
    assert "gpt-4o-mini & 10.0 & 51.1 & 57 & 51" in (tmp_path / "tab_deference.tex").read_text()
    prev = (tmp_path / "tab_prev11.tex").read_text()
    assert "$\\downarrow$" in prev and "one-answer, gpt-4.1 &" in prev and "multi-answer, gpt-4o-mini &" in prev
    prior = (tmp_path / "tab_prior.tex").read_text()
    assert "Claude Sonnet has no closed-book run." in prior and "R1:" not in prior


def test_write11_skips_without_results(tmp_path):
    a11.write11({"rev": {}}, tmp_path)
    assert list(tmp_path.iterdir()) == []


_HYP = ROOT / "paper/generated/hypotheses.json"
_R11 = [ROOT / f"runs/rev11/rev11_{m}.json" for m in ("prior", "account", "scoring", "policy")]


@pytest.mark.skipif(not _HYP.exists() or not all(p.exists() for p in _R11), reason="needs hypotheses.json and runs/rev11")
def test_full_numbers_tex_has_no_duplicate_names(tmp_path):
    from experiments.conflict import paper_assets
    R = json.loads(_HYP.read_text())
    R["rev"]["r11"] = {k: v for p in _R11 for k, v in json.loads(p.read_text()).items()}
    paper_assets.numbers(R, tmp_path)
    names = re.findall(r"\\newcommand\{\\([^}]*)\}", (tmp_path / "numbers.tex").read_text())
    assert len(names) == len(set(names))
    assert all(re.fullmatch(r"[A-Za-z]+", n) for n in names)
    new = {n for n, _ in _collect(R)}
    assert new <= set(names) and len(new) > 500
    # the same quantity carries the same value in its old and its new macro
    v = dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}\n", (tmp_path / "numbers.tex").read_text()))
    same = [("stanceCorrectYes", "hcLabCorrect"), ("stanceMisYes", "hcLabMis"), ("stanceBoth", "hcLabBoth"),
            ("acctMae", "frMae"), ("acctMaeMarg", "frMaeMarg"), ("acctCleanMaeAll", "frCleanMae"),
            ("sensDirReg", "hThirteenDirFour"), ("sensDirRegCI", "hThirteenDirFourCI"),
            ("terraDrawRegBe", "beTerra"), ("terraDrawRegFRCI", "rfolRightTerraCI"),
            ("priorSsMiniKnown", "cbkCbKnownDir"), ("priorSsMiniUnknown", "cbkCbUnknownDir")]
    same += [(f"ssPrecAll{G}", f"dirPrec{G}") for G in ("Mini", "Four", "Luna", "Terra", "Haiku", "Sonnet")]
    for a, b in same:
        assert a in v and b in v, (a, b)
        assert v[a] == v[b], (a, v[a], b, v[b])
