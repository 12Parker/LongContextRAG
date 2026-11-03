"""
Paper macros, tables and the risk-coverage figure for PR§20 (revision11.py: R["rev"]["r11"] holds the union of
rev11_prior, rev11_account, rev11_scoring and rev11_policy, keys R1-R19 and A1). numbers11 runs inside
paper_assets.numbers after numbers10; write11 after write10. Every result here is exploratory (PR§20), outside the
FDR count. Scales as the modules store them: rates, shares and accuracies of R1-R4, R9, R10, R12, R13 are fractions;
effects, MAEs and levels of R5-R8, R11, R14-R18 are points (x100); break-even shares and p* are fractions.

catalogue11(R) renders the macro list (name, generated value, meaning) as Markdown for the paper writers.
"""
from __future__ import annotations

import math
import re
import statistics
from pathlib import Path

from .assets5 import GEN_A, NICE
from .assets6 import _lab
from .assets7 import H31_ORDER, _pt
from .paper_assets import _nan, pct

PRIMARY = "ramdocs_strict_oneway"
R2_ORDER = ("gpt-4o-mini", "gpt-4.1", "gpt-6-luna", "gpt-6.1-terra", "claude-haiku-5-5")
R3_ORDER = R2_ORDER + ("claude-sonnet-5-5",)
FRESH = {"mini": "gpt-4o-mini", "luna": "gpt-6-luna", "haiku": "claude-haiku-5-5", "terra": "gpt-6.1-terra",
         "qwen": "Qwen2.5-7B"}
MODEL = {"qwen2.5:7b-instruct": "Qwen2.5-7B"}
POLICY = (("multi", "Multi", "multi-answer"), ("minimal", "Minimal", "minimal skeleton (no stated policy)"),
          ("single", "Single", "one answer, ``unknown'' allowed"), ("forced", "Forced", "one answer, forced"))
# figure: three colours (the paper's validated categorical slots 1-3); the minimal skeleton allows several answers
# (revision6's moderator groups it with multi-answer), so it takes that colour with a dashed arrow
POL_COL = {"multi": "#2a78d6", "minimal": "#2a78d6", "single": "#eb6834", "forced": "#1baf7a"}
PREV_POL = {"multi": "multi-answer", "minimal": "minimal", "single": "one-answer", "forced": "forced"}
CLEAN_GROUP = (("cb_distractors", "Cb", "ConflictBank: correct passage plus three off-subject distractors"),
               ("ramdocs_multi", "Rd", "RAMDocs: questions without misinformation, mostly several valid answers"),
               ("raguard", "Rg", "RAGuard: claims with supporting posts and no misleading post"))
JUDGE_COND = (("oracle+flag_directional", "OracleDir", "oracle directional flag"), ("answer_key", "AnswerKey", "answer key"),
              ("llm_dir+flag_directional", "LlmDir", "LLM directional flag"), ("llm_dir+verdict_key", "VerdictKey", "verdict key"))


# --------------------------------------------------------------------------- formatting
def _ok(x):
    return x is not None and not _nan(x) and not (isinstance(x, float) and math.isinf(x))


def _fr(x, nd=1, sign=False):
    """A fraction as a percentage (or signed points)."""
    return pct(x, sign, nd) if _ok(x) else "--"


def _frci(ci, nd=1, sign=False):
    return "--" if not ci or not all(_ok(c) for c in ci) else f"[{_fr(ci[0], nd, sign)}, {_fr(ci[1], nd, sign)}]"


def _p(x, nd=1, sign=False):
    """Points (x100 already)."""
    return _pt(x, sign, nd) if _ok(x) else "--"


def _pci(ci, nd=1, sign=True):
    return "--" if not ci or not all(_ok(c) for c in ci) else f"[{_p(ci[0], nd, sign)}, {_p(ci[1], nd, sign)}]"


def _num(x, nd=2, sign=False):
    """A plain number with a real minus sign (and plus, if signed)."""
    if not _ok(x):
        return "--"
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    if float(s) == 0:
        return f"{0:.{nd}f}"
    return ("$-$" + s[1:] if s[0] == "-" else "$+$" + s[1:] if s[0] == "+" else s)


def _numci(ci, nd=2, sign=False):
    return "--" if not ci or not all(_ok(c) for c in ci) else f"[{_num(ci[0], nd, sign)}, {_num(ci[1], nd, sign)}]"


def _pv(p):
    """A p-value as assets7 writes it."""
    return "--" if not _ok(p) else "$<$0.001" if p < 0.001 else f"{p:.3f}"


def _int(n):
    return f"{int(n):,}" if _ok(n) else "--"


def _ncdf(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def _esc(s: str) -> str:
    return re.sub(r"([&%$#_{}])", r"\\\1", str(s)).replace("~", r"\textasciitilde{}").replace("^", r"\textasciicircum{}")


def _yn(b):
    return "yes" if b else "no"


def _mid(name):
    """A run's config model name as the model id the rest of the code uses."""
    return MODEL.get(name, name)


def _g(name):
    return NICE.get(_mid(name), _mid(name))


def _sig(ci):
    """+1 / -1 when a 95% CI excludes zero above / below, else 0."""
    if not ci or not all(_ok(c) for c in ci):
        return 0
    return 1 if ci[0] > 0 else -1 if ci[1] < 0 else 0


def _cond(c: str) -> str:
    special = {"plain": "no flag", "closedbook": "closed-book", "hint": "hint", "instruct_v2": "instruction (v2)",
               "instruct": "instruction (v1)", "answer_key": "answer key", "madam": "MADAM-RAG"}
    if c in special:
        return special[c]
    det, _, fmt = c.partition("+")
    if not fmt:
        return _esc(c)
    instr = fmt.endswith("_instruct_v2")
    fmt = fmt[: -len("_instruct_v2")] if instr else fmt
    base = {"flag": "neutral flag", "flag_directional": "dir.\\ flag", "flag_onewrong": "one-wrong flag",
            "remove_suspect": "remove suspect", "remove": "remove", "verdict_key": "verdict key"}.get(fmt, _esc(fmt))
    d = {"oracle": "oracle", "llm_dir": "LLM", "nli": "NLI", "caag": "CAAG", "random": "random"}.get(det, _esc(det))
    return f"{d} {base}" + (" + instr.\\ (v2)" if instr else "")


class _Adder:
    """add(name, value) plus an optional describe(section, name, value, meaning) for the catalogue. Names are checked
    to be letters only (LaTeX macro names) and unique within numbers11."""

    def __init__(self, add, describe=None):
        self._add, self._describe, self.section, self.seen = add, describe, "", set()

    def __call__(self, name, value, meaning=""):
        if not re.fullmatch(r"[A-Za-z]+", name):
            raise ValueError(f"macro name {name!r} is not letters only")
        if name in self.seen:
            raise ValueError(f"macro {name!r} emitted twice")
        self.seen.add(name)
        self._add(name, value)
        if self._describe:
            self._describe(self.section, name, value, meaning)


# --------------------------------------------------------------------------- macros
def numbers11(R, add, describe=None):
    A = _Adder(add, describe)
    rev = R.get("rev", {})
    r11 = rev.get("r11", {})
    _r1(A, r11.get("R1")); _r2(A, r11.get("R2")); _r3(A, r11.get("R3")); _r4(A, r11.get("R4"))
    _r5(A, r11.get("R5")); _r6(A, r11.get("R6")); _r7(A, r11.get("R7")); _r8(A, r11.get("R8"))
    _r9(A, r11.get("R9")); _r10(A, r11.get("R10")); _r11(A, r11.get("R11")); _r12(A, r11.get("R12"))
    _r13(A, r11.get("R13")); _r14(A, r11.get("R14")); _r15(A, r11.get("R15")); _r16(A, r11.get("R16"))
    _r17(A, r11.get("R17")); _r18(A, r11.get("R18")); _r19(A, r11.get("R19"))
    _uncited(A, rev)
    _deference(A, rev)


def _r1(A, r1):
    if not r1 or "full" not in r1:
        return
    A.section = "R1 passage-only text classifier (ConflictBank)"
    for var, vt, vd in (("full", "Full", "full passages"), ("masked", "Mask", "each passage's own object masked")):
        for split, st, sd in (("eval", "Eval", "eval set"), ("fresh", "Fresh", "ConflictBank-fresh")):
            e = r1[var].get(split)
            if not e:
                continue
            A(f"clf{vt}{st}", f"{e['pairwise_acc']:.3f}", f"pairwise accuracy ({vd}, {sd} conflict items): share where the "
              "misinformation passage gets the higher P(false)")
            A(f"clf{vt}{st}CI", _numci(e["ci"], 3), f"95% item-bootstrap CI of clf{vt}{st}")
            A(f"clf{vt}{st}Point", f"{e['pointwise_acc']:.3f}", f"pointwise accuracy (each passage classified alone; {vd}, {sd})")
            A(f"clf{vt}{st}PMis", f"{e['mean_p_false_misinfo']:.2f}", f"mean P(false) on misinformation passages ({vd}, {sd})")
            A(f"clf{vt}{st}PCor", f"{e['mean_p_false_correct']:.2f}", f"mean P(false) on correct passages ({vd}, {sd})")
            A(f"clf{vt}{st}NnMax", f"{e['nn_cosine_max']:.2f}", f"max TF-IDF cosine of a {sd} passage to its nearest training "
              f"passage ({vd})")
            A(f"clf{vt}{st}NnHigh", e["n_nn_cosine_ge_0.8"], f"{sd} items with a nearest-training cosine >= 0.8 ({vd})")
        A(f"clf{vt}TrainAcc", f"{r1[var]['train_acc']:.3f}", f"training accuracy ({vd})")
    A("clfN", r1["full"]["eval"]["n"], "eval conflict items scored (pairs)")
    if "fresh" in r1["full"]:
        A("clfNFresh", r1["full"]["fresh"]["n"], "ConflictBank-fresh conflict items scored (pairs)")
    A("clfTrainRows", _int(r1["n_train_rows"]), "QA_dataset rows used for training (subjects used by neither eval nor fresh)")
    A("clfTrainPassages", _int(r1["n_train_passages"]), "training passages (two per row)")
    A("clfExclSubjects", _int(r1["n_excluded_subjects"]), "subjects excluded from training (used by eval or fresh)")
    A("clfIdentical", r1["n_train_passages_identical_to_eval"], "training passages identical to an eval passage")
    q = lambda xs: ", ".join(f"``{_esc(w)}''" for w, _ in xs[:3])
    A("clfTopFalse", q(r1["full"]["top_false"]), "the three n-grams with the largest weight towards 'false' (full model)")
    A("clfTopCorrect", q(r1["full"]["top_correct"]), "the three n-grams with the largest weight towards 'correct' (full model)")


def _r2(A, r2):
    if not r2:
        return
    A.section = ("R2 direction precision by the detector's own closed-book knowledge (ConflictBank eval; PR§20.2 calls it "
                 "direction accuracy, as §11.3)")
    dets = [d for d in R2_ORDER if isinstance(r2.get(d), dict) and "single_side" in r2[d]]
    for d in dets:
        v, D = r2[d], GEN_A[d]
        A(f"priorKnownQ{D}", v["n_known"], f"{NICE[d]}: conflict questions it knows closed-book (PR§11.3 rule)")
        A(f"priorUnknownQ{D}", v["n_unknown"], f"{NICE[d]}: conflict questions it does not know")
        A(f"priorSeeds{D}", len(v.get("closedbook_seeds", [])), f"{NICE[d]}: closed-book seeds behind the known rule")
        for blk, bt, bd in (("single_side", "Ss", "registered single-side estimand (flags naming exactly one side)"),
                            ("direction", "Dir", "PR§11.3 coding ('both' a miss, 'unclear' skipped)")):
            b = v[blk]
            A(f"prior{bt}{D}KnownN", b["n_known_flags"], f"{NICE[d]}, {bd}: flags on known questions (denominator)")
            A(f"prior{bt}{D}UnknownN", b["n_unknown_flags"], f"{NICE[d]}, {bd}: flags on unknown questions")
            A(f"prior{bt}{D}Known", _fr(b["known"]), f"{NICE[d]}, {bd}: direction precision on known questions (%)")
            A(f"prior{bt}{D}KnownCI", _frci(b["known_ci"]), "95% question-bootstrap CI")
            A(f"prior{bt}{D}Unknown", _fr(b["unknown"]), f"{NICE[d]}, {bd}: direction precision on unknown questions (%)")
            A(f"prior{bt}{D}UnknownCI", _frci(b["unknown_ci"]), "95% question-bootstrap CI")
            A(f"prior{bt}{D}Diff", _fr(b["diff"], sign=True), f"{NICE[d]}, {bd}: known minus unknown (points)")
            A(f"prior{bt}{D}DiffCI", _frci(b["diff_ci"], sign=True), "95% question-bootstrap CI")
    for bt, blk, bd in (("Ss", "single_side", "single-side"), ("Dir", "direction", "PR§11.3-coded")):
        diffs = {d: r2[d][blk]["diff"] for d in dets}
        if not diffs:
            continue
        lo, hi = min(diffs, key=diffs.get), max(diffs, key=diffs.get)
        A(f"prior{bt}DiffMin", _fr(diffs[lo], sign=True), f"smallest {bd} known-minus-unknown difference over the "
          f"{len(diffs)} detectors (points)")
        A(f"prior{bt}DiffMinDet", NICE[lo], "detector with that smallest difference")
        A(f"prior{bt}DiffMax", _fr(diffs[hi], sign=True), f"largest {bd} known-minus-unknown difference (points)")
        A(f"prior{bt}DiffMaxDet", NICE[hi], "detector with that largest difference")
        A(f"prior{bt}DiffPos", sum(_sig(r2[d][blk]["diff_ci"]) > 0 for d in dets),
          f"detectors whose {bd} difference has a 95% CI above zero")
    A("priorN", len(dets), "detectors with a closed-book run (R2)")


def _r3(A, r3):
    if not r3:
        return
    A.section = "R3 single-side direction precision (ConflictBank eval, every detector)"
    dets = [d for d in R3_ORDER if isinstance(r3.get(d), dict) and "single_side" in r3[d]]
    for d in dets:
        v, D = r3[d], GEN_A[d]
        A(f"ssPrec{D}", _fr(v["single_side"]), f"{NICE[d]}: share of its gold-pair flags naming exactly one side that "
          "name the misinformation passage (%)")
        A(f"ssPrec{D}CI", _frci(v["single_ci"]), "95% question-cluster bootstrap CI (not Wilson)")
        A(f"ssPrec{D}N", _int(v["single_n"]), f"{NICE[d]}: single-side flags (denominator)")
        A(f"ssPrecAll{D}", _fr(v["current"]), f"{NICE[d]}: the earlier definition ('both'/'unclear' count as misses, %; "
          f"equals \\dirPrec{D})")
        A(f"ssPrecAll{D}CI", _frci(v["current_ci"]), f"95% question-cluster bootstrap CI (wider than \\dirPrec{D}CI, a "
          "Wilson interval over flags; use this one beside ssPrec)")
        A(f"ssPrecAll{D}N", _int(v["current_n"]), f"{NICE[d]}: all gold-pair flags (denominator)")
        mix = v.get("verdict_mix", {})
        A(f"ssBoth{D}", _int(mix.get("both", 0)), f"{NICE[d]}: gold-pair flags naming both sides")
        A(f"ssUnclear{D}", _int(mix.get("unclear", 0)), f"{NICE[d]}: gold-pair flags with an unclear side")
    if dets:
        p = {d: r3[d]["single_side"] for d in dets}
        lo, hi = min(p, key=p.get), max(p, key=p.get)
        A("ssPrecMin", _fr(p[lo]), f"lowest single-side precision over the {len(dets)} detectors (%)")
        A("ssPrecMinDet", NICE[lo], "detector with the lowest")
        A("ssPrecMax", _fr(p[hi]), "highest single-side precision (%)")
        A("ssPrecMaxDet", NICE[hi], "detector with the highest")
        o = {d: x for d, x in p.items() if d != "gpt-4o-mini"}
        if o:
            A("ssPrecOtherMin", _fr(min(o.values())), "lowest single-side precision excluding gpt-4o-mini (%)")
            A("ssPrecOtherMax", _fr(max(o.values())), "highest single-side precision excluding gpt-4o-mini (%)")
        A("ssPrecN", len(dets), "detectors in R3")


def _r4(A, r4):
    if not r4:
        return
    A.section = ("R4 follow rates by the generator's own closed-book knowledge (H31 runs). The all-question rates and "
                 "CIs are the paper's existing \\rfolRight<G> / \\rfolWrong<G> (R4 reproduces the rates; its own "
                 "bootstrap CI for them would differ by a point, so it is not emitted)")
    gens = [g for g in H31_ORDER if isinstance(r4.get(g), dict) and "f_R" in r4[g]]
    for g in gens:
        v, G = r4[g], GEN_A[g]
        A(f"prFolKnownQ{G}", v["n_known"], f"{NICE[g]}: conflict questions it knows closed-book")
        for fk, ft, fd in (("f_R", "R", "right flag followed"), ("f_W", "W", "wrong flag followed")):
            x = v[fk]
            for part, pt_, pd in (("known", "Known", "known"), ("unknown", "Unknown", "unknown")):
                A(f"prFol{ft}{pt_}{G}", _fr(x[part]["rate"], 0), f"{NICE[g]}: f_{ft} ({fd}) on {pd} questions (%)")
                A(f"prFol{ft}{pt_}{G}N", x[part]["n"], f"{NICE[g]}: questions behind f_{ft} {pd}")
                A(f"prFol{ft}{pt_}{G}CI", _frci(x[part]["ci"], 0), "95% question-bootstrap CI")
            A(f"prFol{ft}All{G}N", x["all"]["n"], f"{NICE[g]}: questions behind f_{ft} on all questions "
              f"(the rate is \\rfol{'Right' if ft == 'R' else 'Wrong'}{G})")
            A(f"prFol{ft}Diff{G}", _fr(x["diff"], 0, True), f"{NICE[g]}: f_{ft} known minus unknown (points)")
            A(f"prFol{ft}Diff{G}CI", _frci(x["diff_ci"], 0, True), "95% question-bootstrap CI")
    if gens:
        sh = [r4[g]["n_known"] / r4[g]["n_conflict"] for g in gens]
        A("prFolKnownShareMin", _fr(min(sh), 0), "smallest share of known conflict questions over H31 generators (%)")
        A("prFolKnownShareMax", _fr(max(sh), 0), "largest share of known conflict questions (%)")
    pl = r4.get("pooled_logit", {})
    w = pl.get("f_W")
    if w:
        A("prFolOrW", _num(w["or"]), "pooled f_W odds ratio, known vs unknown (logit, generator fixed effects)")
        A("prFolOrWCI", _numci(w["or_ci"]), "95% question-bootstrap CI of prFolOrW")
        A("prFolOrWN", w["n_obs"], "question x generator observations behind prFolOrW")
        A("prFolOrWQ", w["n_questions"], "distinct questions behind prFolOrW")
        A("prFolOrWKnownN", w["n_known_obs"], "known observations behind prFolOrW")
    r = pl.get("f_R")
    if r:   # separated resamples make the f_R upper bound infinite: no point estimate or CI is emitted
        A("prFolOrRUnbounded", _yn(r.get("ci_unbounded", [False, False])[1]),
          "whether the f_R odds ratio's bootstrap upper bound is infinite (separation); its OR is not reported")
        A("prFolOrRSeparated", _int(r["n_boot_separated"]), "bootstrap resamples with complete separation (f_R)")
        A("prFolOrRKnownN", r["n_known_obs"], "known observations behind the f_R logit")


def _r5(A, r5):
    if not r5 or "conditioned" not in r5:
        return
    A.section = ("R5 follow-rate account on the 30 crossed cells, conditioned on the generator's prior. MAEs and the MAE "
                 "change in 2 decimals because the change is below 0.05; the unconditioned MAEs are \\predAllMae, "
                 "\\predMae and \\predTrMae in 1 decimal")
    for tag, T, d in (("all", "All", "all crossed cells"), ("own", "Own", "cells with the generator's own H31 rates"),
                      ("transfer", "Transfer", "transfer cells (generators without an H31 run)")):
        c, u = r5["conditioned"][tag], r5["unconditioned"][tag]
        A(f"condMae{T}", _num(c["mae"]), f"MAE (points) of the conditioned account, {d}")
        A(f"uncondMae{T}", _num(u["mae"]), f"MAE (points) of the unconditioned account on the same cells, {d}")
        A(f"condMaeChange{T}", _num(r5["mae_change"][tag], 2, True), f"conditioned minus unconditioned MAE, {d}")
        A(f"condN{T}", c["n"], f"cells, {d}")
        A(f"condR{T}", _num(c["r"]), f"r, conditioned, {d}"); A(f"uncondR{T}", _num(u["r"]), f"r, unconditioned, {d}")
        A(f"condSlope{T}", _num(c["slope"]), f"slope, conditioned, {d}")
        A(f"uncondSlope{T}", _num(u["slope"]), f"slope, unconditioned, {d}")
    A("condMaxShift", _num(r5["max_abs_pred_change"], 1), "largest change in any cell's prediction from conditioning (points)")


def _cellname(key):
    g, d = key.split("_")[-2:]
    return FRESH[g], FRESH[d]


def _r6(A, r6):
    if not r6 or "a" not in r6:
        return
    A.section = "R6 prospective-test summaries (ConflictBank-fresh, 20 cells)"
    a = r6["a"]
    # one decimal throughout, as the H32 macros beside them (frMae, frMaeMarg, frMaeGen, frCleanMae)
    A("acctN", r6["n_cells"], "fresh cells")
    A("acctMae", _num(a["mae_account"], 1), "MAE of the frozen account (points; equals frMae)")
    A("acctMaeMarg", _num(a["mae_marginal"], 1), "MAE of the cell-level marginal baseline (points; equals frMaeMarg)")
    A("acctMaeMargCI", _numci(a["mae_marginal_ci"], 1), "95% CI of acctMaeMarg (H32's draws)")
    A("acctMargDiff", _num(a["diff"], 1, True), "(a) MAE(marginal) minus MAE(account), points")
    A("acctMargDiffCI", _numci(a["diff_ci"], 1, True), "paired 95% bootstrap CI (H32's draws)")
    b = r6["b"]
    A("acctCommonMae", _num(b["mae"], 1), "(b) MAE of the common-rate baseline (one f_R, f_W for every generator)")
    A("acctCommonMaeCI", _numci(b["mae_ci"], 1), "95% CI of acctCommonMae")
    A("acctCommonDiff", _num(b["diff"], 1, True), "MAE(common rates) minus MAE(account), points")
    A("acctCommonDiffCI", _numci(b["diff_ci"], 1, True), "paired 95% CI")
    A("acctCommonFR", _fr(b["f_right"], 1), "common f_R: equal-weight mean of the four frozen generators' f_R (%)")
    A("acctCommonFW", _fr(b["f_wrong"], 1), "common f_W (%)")
    A("acctCommonR", _num(b["r"]), "r of the common-rate predictions"); A("acctCommonSlope", _num(b["slope"]), "slope")
    A("acctCommonSign", b["sign_agree"], "cells whose common-rate prediction has the observed sign")
    c = r6["c"]
    A("acctCleanMaeAll", _p(c["mae_20"]), "(c) clean-question MAE over all cells (points)")
    A("acctCleanMaeAllCI", _pci(c["mae_20_ci"], sign=False), "95% CI")
    A("acctCleanMaeFlag", _p(c["mae_flagging"]), "clean-question MAE over the cells whose detector flags a clean question")
    A("acctCleanMaeFlagCI", _pci(c["mae_flagging_ci"], sign=False), "95% CI")
    A("acctCleanNFlag", c["n_flagging"], "cells whose detector flags at least one clean question")
    A("acctCleanNAll", len(c.get("cells", {})) or r6["n_cells"], "cells in the clean-question MAE")
    neg = c.get("negative_cells", {})
    A("acctHarmN", len(neg), "cells with a predicted conflict effect <= -3 (the predicted harms)")
    for k, v in neg.items():
        g, d = _cellname(k)
        t = f"{GEN_A[g]}{GEN_A[d]}"
        A(f"acctHarm{t}", f"{NICE[g]} with the {NICE[d]} detector", "a predicted-harm cell (generator, detector)")
        A(f"acctHarm{t}Pred", _p(v["pred"], sign=True), f"{NICE[g]} x {NICE[d]}: predicted conflict effect (points)")
        A(f"acctHarm{t}PredCI", _pci(v["pred_ci"]), "95% CI of the prediction")
        A(f"acctHarm{t}Obs", _p(v["obs"], sign=True), f"{NICE[g]} x {NICE[d]}: observed conflict effect (points)")
        A(f"acctHarm{t}ObsCI", _pci(v["obs_ci"]), "95% CI of the observed effect")
        A(f"acctHarm{t}ErrCI", _pci(v["obs_minus_pred_ci"]), "95% CI of observed minus predicted")
    s = r6.get("d", {}).get("summary")
    if s:
        A("acctSelMae", _num(s["mae"], 1), "(d) selection analysis: MAE of the all-question prediction over the "
          f"{r6['n_cells']} cells (points)")
        A("acctSelMaeFlag", _num(s["mae_flagging"], 1), "the same over cells whose detector flags clean questions")
        A("acctSelMaeNoFlag", _num(s["mae_not_flagging"], 1), "the same over the other cells")
        A("acctSelConflict", _num(s["mean_conflict"], 1, True), "mean signed error from the conflict part (points)")
        A("acctSelClean", _num(s["mean_clean"], 1, True), "mean signed error from the clean part (points)")
        A("acctSelAbsConflict", _num(s["mean_abs_conflict"], 1), "mean absolute conflict part")
        A("acctSelAbsClean", _num(s["mean_abs_clean"], 1), "mean absolute clean part")
        cells = r6["d"].get("cells", {})
        if cells:
            k = max(cells, key=lambda x: abs(cells[x]["err"]))
            v = cells[k]
            A("acctSelMaxErr", _num(v["err"], 1, True), "largest selection-prediction error (points)")
            A("acctSelMaxErrClean", _num(v["clean"], 1, True), "its clean part")
            A("acctSelMaxErrCell", f"{NICE[v['gen']]} with the {NICE[v['det']]} detector", "the cell with that error")


def _r7(A, r7):
    if not r7 or "H31" not in r7:
        return
    A.section = "R7 gpt-6.1-terra's two plain draws"
    h = r7["H31"]
    for tag, blk, d in (("Reg", "registered", "plain draw the paper uses (the H31 run's own)"),
                        ("Alt", "other_draw", "the other plain draw (the one-answer run's)")):
        t = h[blk]
        A(f"terraDraw{tag}Be", f"{t['breakeven']:.2f}", f"terra break-even p*, {d}")
        A(f"terraDraw{tag}BeCI", f"[{t['be_low']:.2f}, {t['be_high']:.2f}]", "bootstrap 95% CI")
        A(f"terraDraw{tag}FR", _fr(t["follow_right"], 0), f"terra f_R, {d} (%)")
        A(f"terraDraw{tag}FRCI", _frci(t["follow_right_ci"], 0), "95% CI")
        A(f"terraDraw{tag}FW", _fr(t["follow_wrong"], 0), f"terra f_W, {d} (%)")
        A(f"terraDraw{tag}FWCI", _frci(t["follow_wrong_ci"], 0), "95% CI")
        A(f"terraDraw{tag}Right", _p(t["right"], sign=True), f"terra right-flag effect, {d} (points)")
        A(f"terraDraw{tag}Wrong", _p(t["wrong"], sign=True), f"terra wrong-flag effect, {d} (points)")
    du = r7.get("draw_used", {})
    reg = next((v for k, v in du.items() if k.startswith("H31")), None)
    for bench, bt in (("ConflictBank", ""), ("RAMDocs", "Rd")):
        p = r7.get("plain", {}).get(bench)
        if not p:
            continue
        runs = p["runs"]
        i = runs.index(reg) if bench == "ConflictBank" and reg in runs else 0
        for sub, st in (("conflict", "Conf"), ("clean", "Clean")):
            x = p.get(f"{PRIMARY}|{sub}")
            if not x:
                continue
            A(f"terraDraw{bt}Plain{st}A", _p(x["mean"][i]), f"{bench}, terra plain accuracy on {sub} questions, draw "
              f"{runs[i]} (%)")
            A(f"terraDraw{bt}Plain{st}B", _p(x["mean"][1 - i]), f"the same, draw {runs[1 - i]} (%)")
            A(f"terraDraw{bt}Differ{st}", x["score_differs"], f"{bench} {sub} questions whose one-way score differs "
              "between the two draws")
        A(f"terraDraw{bt}Differ", sum(p.get(f"{PRIMARY}|{s}", {}).get("score_differs", 0) for s in ("conflict", "clean")),
          f"{bench} questions whose one-way score differs between the draws (conflict + clean)")
        A(f"terraDraw{bt}Identical", p["identical_texts"], f"{bench} responses with identical text in both draws")
        A(f"terraDraw{bt}Rows", p["n_rows"], f"{bench} responses compared")
    hn = r7.get("H19", {})
    if hn:
        A("terraDrawHNineteenChanged", len(hn["changed_status"].get(PRIMARY, [])),
          "H19 tests whose status changes with the other RAMDocs draw (registered swap)")
        sup = hn.get("supplementary_cb_too", {}).get("changed_status", {})
        A("terraDrawHNineteenChangedCb", len(sup.get(PRIMARY, [])),
          "the same with the ConflictBank draw swapped too (supplementary, not registered)")


def _r8(A, r8):
    if not r8:
        return
    A.section = "R8 completion tokens under the flag (H31 runs; include hidden reasoning tokens)"
    for g in ("gpt-6.1-terra", "gpt-6-luna", "claude-haiku-5-5"):
        v = r8.get(g)
        if not v:
            continue
        G = GEN_A[g]
        for blk, bt, d in (("all", "", "all conflict questions"), ("right_arm", "Right", "right-flag arm"),
                           ("wrong_arm", "Wrong", "wrong-flag arm")):
            x = v[blk]
            A(f"tok{G}{bt}Plain", _num(x["plain"], 0), f"{NICE[g]}: mean completion tokens without flags, {d}")
            A(f"tok{G}{bt}Flag", _num(x["flagged"], 0), f"{NICE[g]}: mean completion tokens with the flag, {d}")
            A(f"tok{G}{bt}Diff", _num(x["diff"], 0, True), f"{NICE[g]}: flagged minus plain, {d}")
            A(f"tok{G}{bt}DiffCI", _numci(x["diff_ci"], 0, True), "95% question-bootstrap CI")
        # PR§20.2 R8 also splits by whether the flag was followed (the f_R / f_W questions, flagged responses only)
        for blk, bt, d in (("right_followed", "Right", "right flag, on the f_R questions; follows = one-way strict correct"),
                           ("wrong_followed", "Wrong", "wrong flag, on the f_W questions; follows = the response contains "
                            "the wrong answer")):
            x = v.get(blk)
            if not x:
                continue
            A(f"tok{G}{bt}Fol", _num(x["followed"], 0), f"{NICE[g]}: mean completion tokens of flagged responses that "
              f"follow the flag ({d})")
            A(f"tok{G}{bt}NotFol", _num(x["not_followed"], 0), f"{NICE[g]}: the same for responses that do not follow it")
            A(f"tok{G}{bt}FolDiff", _num(x["diff"], 0, True), f"{NICE[g]}: followed minus not followed ({d})")
            A(f"tok{G}{bt}FolDiffCI", _numci(x["diff_ci"], 0, True), "95% question-bootstrap CI")
            A(f"tok{G}{bt}FolRows", x["n_rows_followed"], f"{NICE[g]}: flagged responses that follow ({d})")
            A(f"tok{G}{bt}NotFolRows", x["n_rows_not"], f"{NICE[g]}: flagged responses that do not follow ({d})")
            A(f"tok{G}{bt}FolQ", x["n_questions"], f"{NICE[g]}: questions behind the split ({d})")


def _r9(A, r9):
    if not r9 or "totals" not in r9:
        return
    A.section = "R9 missing final answers (every run)"
    t = r9["totals"]
    A("missRuns", r9["n_runs"], "runs with results")
    A("missRows", _int(t["n"]), "responses checked")
    A("missN", _int(t["missing"]), "responses with no final-answer line")
    A("missShare", f"{100 * t['missing_share']:.2f}", "share of responses with no final-answer line (%)")
    A("missCapShare", _fr(t["at_cap_share_of_missing_single_call"]), "share of the single-call missing responses that "
      "stopped at the completion cap (%)")
    A("missCells", r9["n_cells"], "run x condition cells")
    A("missCellsAbove", r9["n_cells_above_1pct"], "cells with more than 1% missing")
    ab = r9.get("above_1pct", [])
    if ab:
        run, c, n, ms, cs = ab[0]
        A("missTopRun", rf"\texttt{{{_esc(run)}}}", "run of the worst cell")
        A("missTopModel", _g(r9["runs"][run]["model"]), "generator of the worst cell")
        A("missTopCond", _cond(c), "condition of the worst cell")
        A("missTopShare", _fr(ms), "missing share of the worst cell (%)")
        A("missTopCap", _fr(cs), "share of the worst cell's responses at the cap (%)")
        A("missTopN", n, "responses in the worst cell")
        from collections import Counter
        cnt = Counter(_mid(r9["runs"][r]["model"]) for r, *_ in ab)
        for m, k in cnt.items():
            tag = GEN_A.get(m, re.sub(r"[^A-Za-z]", "", m).title())
            A(f"missAbove{tag}", k, f"cells above 1% with {_g(m)}")
    for m, v in r9.get("by_model", {}).items():
        if _mid(m) in GEN_A:
            A(f"missShare{GEN_A[_mid(m)]}", f"{100 * v['missing_share']:.2f}",
              f"{_g(m)}: share of responses missing a final answer (%; 0.00 can hide a few, see the count)")
            if "missing" in v and "n" in v:
                A(f"missN{GEN_A[_mid(m)]}", _int(v["missing"]), f"{_g(m)}: responses missing a final answer")
                A(f"missRows{GEN_A[_mid(m)]}", _int(v["n"]), f"{_g(m)}: responses checked")


def _r10(A, r10):
    t = (r10 or {}).get(PRIMARY)
    if not t:
        return
    A.section = "R10 H13 with gpt-4.1 without questions missing a final answer"
    for tk, v in t.items():
        T = "Neu" if "neutral flag + instruction vs plain" in tk else "Dir" if "dir. flag + instruction" in tk else None
        if T is None:
            continue
        g = v["registered"]
        d = ("neutral flag + instruction vs plain" if T == "Neu"
             else "directional flag + instruction vs neutral flag + instruction")
        A(f"sens{T}Reg", _fr(g["diff"], sign=True), f"registered H13 ({d}, RAMDocs gpt-4.1, conflict; points)")
        A(f"sens{T}RegCI", _frci([g["ci_low"], g["ci_high"]], sign=True), "95% CI")
        A(f"sens{T}RegP", _pv(g["p_holm"]), "Holm p (registered family)")
        A(f"sens{T}RegN", g["n"], "questions (registered)")
        A(f"sens{T}RegSupported", _yn(g["supported"]), "registered status")
        A(f"sens{T}", _fr(v["diff"], sign=True), f"the same without the dropped questions ({d}; points)")
        A(f"sens{T}CI", _frci([v["ci_low"], v["ci_high"]], sign=True), "95% CI")
        A(f"sens{T}P", _pv(v["p_holm"]), "Holm p")
        A(f"sens{T}N", v["n"], "questions kept")
        A(f"sens{T}Dropped", v["n_dropped_conflict"], "conflict questions dropped (any seed of either arm lacks a final answer)")
        A(f"sens{T}Supported", _yn(v["supported"]), "status without the dropped questions (sensitivity only)")


def _r11(A, r11):
    if not r11:
        return
    A.section = "R11 seed dependence"
    for fam, T in (("H23", "TwentyThree"), ("H30a", "ThirtyA")):
        v = r11.get("pooled", {}).get(fam)
        if not v:
            continue
        A(f"seed{T}All", _p(v["all"]["diff"], sign=True), f"pooled {fam} (matcher), all seeds (points)")
        A(f"seed{T}K", v["all"]["n_cells"], f"cells in pooled {fam}")
        A(f"seed{T}Lo", _p(v["range"][0], sign=True), f"pooled {fam}: lowest estimate from one seed alone (points)")
        A(f"seed{T}Hi", _p(v["range"][1], sign=True), f"pooled {fam}: highest estimate from one seed alone (points)")
        A(f"seed{T}FixedLo", _p(v["range_fixed"][0], sign=True), f"{fam}: lowest, single-seed cells kept in every seed")
        A(f"seed{T}FixedHi", _p(v["range_fixed"][1], sign=True), f"{fam}: highest, single-seed cells kept in every seed")
        A(f"seed{T}Single", len(v["single_seed_cells"]), f"single-seed cells in pooled {fam}")
    t = r11.get("table2")
    if t:
        S = t["settings"]
        A("seedTabSettings", len(S), "Table 2 settings")
        A("seedTabSingle", len(t["single_seed_settings"]), "Table 2 settings with one seed")
        multi = {k: v for k, v in S.items() if len(v["seeds"]) > 1}
        for m, M, d in (("correct", "Correct", "accuracy"), ("misinfo", "Misinfo", "misinformation share"),
                        ("abstain", "Abstain", "abstention")):
            if multi:
                k = max(multi, key=lambda x: multi[x]["range"][m][1] - multi[x]["range"][m][0])
                A(f"seedTab{M}Spread", _p(multi[k]["range"][m][1] - multi[k]["range"][m][0]),
                  f"largest seed-to-seed spread of a neutral-flag {d} contrast over multi-seed Table 2 settings (points)")
                A(f"seedTab{M}SpreadSet", _lab(k), "setting with that spread")


def _r12(A, r12):
    if not r12:
        return
    A.section = "R12 one matcher per table (one-way misinformation)"
    ex = r12.get("explo_oneway", {})
    ch = r12.get("explo_sig_changed", [])
    A("oneWayExploN", r12["explo_n"], "exploratory misinformation decisions (one Holm family)")
    A("oneWayExploChanged", len(ch), "decisions that change with the one-way matcher")
    A("oneWayExploLost", sum(1 for k in ch if ex[k]["bidir"]["sig"] and not ex[k]["sig"]), "of these, significant -> not")
    A("oneWayExploGained", sum(1 for k in ch if ex[k]["sig"] and not ex[k]["bidir"]["sig"]), "of these, not -> significant")
    A("oneWayHTwentyOneN", len(r12.get("H21a_oneway", {})), "registered H21(a) tests recomputed")
    A("oneWayHTwentyOneChanged", len(r12.get("H21a_changed", [])), "H21(a) tests whose status would change")
    lost, gained = [], []
    for key, conds in r12.get("table2", {}).items():
        for c, m in conds.items():
            if c == "plain" or "bold_misinfo_oneway" not in m:
                continue
            if m["bold_misinfo_bidir"] and not m["bold_misinfo_oneway"]:
                lost.append((key, c))
            elif m["bold_misinfo_oneway"] and not m["bold_misinfo_bidir"]:
                gained.append((key, c))
    nm = lambda kc: f"{_lab(kc[0])}, {_cond(kc[1])}"
    A("oneWayBoldLost", len(lost), "Table 2 misinformation bold marks lost under one-way matching")
    A("oneWayBoldGained", len(gained), "Table 2 misinformation bold marks gained")
    if lost:
        A("oneWayBoldLostCell", "; ".join(nm(x) for x in lost), "the Table 2 cell(s) that lose bold")
        x = ex.get(f"{lost[0][0]}|{lost[0][1]}")
        if x:
            A("oneWayBoldLostDiff", _fr(x["diff"], sign=True), "its misinformation change, one-way (points)")
            A("oneWayBoldLostDiffBidir", _fr(x["bidir"]["diff"], sign=True), "its misinformation change, bidirectional")
            A("oneWayBoldLostP", _pv(x["p_adj"]), "its Holm p, one-way")
            A("oneWayBoldLostPBidir", _pv(x["bidir"]["p_adj"]), "its Holm p, bidirectional")


def _r13(A, r13):
    if not r13 or "correct_page" not in r13:
        return
    # whole percentages, as the same shares already in the paper (hcLabCorrect, hcLabMis, hcLabBoth)
    A.section = ("R13 HealthContradict page stance (judge verdicts, §19.3 audit; stanceCorrectYes, stanceMisYes and "
                 "stanceBoth equal \\hcLabCorrect, \\hcLabMis and \\hcLabBoth)")
    A("stanceN", r13["n_parsed"], "items with a parsed judge verdict")
    for blk, bt, d in (("correct_page", "Correct", "correct-typed page supports the gold answer"),
                       ("misinfo_page", "Mis", "misinformation-typed page supports the wrong answer")):
        for v, V in (("yes", "Yes"), ("no", "No"), ("unclear", "Unclear")):
            A(f"stance{bt}{V}", _fr(r13[blk][v], 0), f"share of items: {d} = {v} (%)")
    A("stanceBothN", r13["both_confirmed"], "items whose two pages both support their labelled answer")
    A("stanceBoth", _fr(r13["both_confirmed_share"], 0), "the same as % of items")
    A("stanceNotBoth", _fr(r13["no_confirmed_pair_share"], 0), "items without a confirmed pair (%)")
    A("stancePages", r13["n_pages"], "pages judged")
    A("stanceReversedN", r13["reversed_pages"], "pages the judge says do NOT support their labelled side ('no')")
    A("stanceReversed", _fr(r13["reversed_share"], 0), "the same as % of pages")
    cut = r13.get("at_cut", {})
    if cut:
        A("stanceCutN", cut["n"], "pages at the character cut")
        A("stanceCut", _fr(cut["share"], 0), "pages at the cut (%)")
        A("stanceCutChars", _int(cut["cut_chars"]), "the cut (characters)")
        if "truncated_in_source" in cut:
            A("stanceTruncSrc", cut["truncated_in_source"], "pages longer than the cut in the source (truncated)")


def _r14(A, r14):
    if not r14:
        return
    A.section = ("R14 RAGuard answer key against the oracle directional flag (conflict claims). Each arm against no flag "
                 "is already in the paper: \\zRg<G>AnswerKeyConflict and \\zRg<G>OracleFlagDirectionalConflict (R14's "
                 "own CIs for them would differ by 0.1, so they are not emitted)")
    ok = [g for g in r14 if isinstance(r14[g], dict) and "key_minus_flag" in r14[g]]
    for g in ok:
        v, G = r14[g], GEN_A[g]
        for k, T, d in (("key_minus_flag", "KeyFlag", "answer key minus oracle directional flag"),):
            A(f"rgKey{T}{G}", _p(v[k]["diff"], sign=True), f"{NICE[g]}: {d} (points)")
            A(f"rgKey{T}{G}CI", _pci(v[k]["ci"]), "95% question-bootstrap CI")
    if ok:
        A("rgKeyN", r14[ok[0]]["key_minus_flag"]["n"], "conflict claims")
        A("rgKeyGens", len(ok), "generators with an answer-key arm")
        A("rgKeyAbove", sum(_sig(r14[g]["key_minus_flag"]["ci"]) > 0 for g in ok), "generators whose key beats the flag (CI > 0)")
        A("rgKeyBelow", sum(_sig(r14[g]["key_minus_flag"]["ci"]) < 0 for g in ok), "generators whose flag beats the key (CI < 0)")
    nr = [NICE[g] for g in r14 if isinstance(r14[g], dict) and "not_run" in r14[g]]
    if nr:
        A("rgKeyNotRun", ", ".join(nr), "generators without an answer-key arm")


def _r15(A, r15):
    if not r15 or "neutral" not in r15:
        return
    A.section = ("R15 single-entity RAMDocs questions, leave one setting out. The pooled row (all settings) is the "
                 "paper's \\entNeuSingle / \\entDirSingle with their CIs and \\entNeuSingleN questions (R15's own CI "
                 "for it would differ by 0.1, so it is not emitted)")
    for k, T, d in (("neutral", "Neu", "oracle neutral flag"), ("directional", "Dir", "oracle directional flag")):
        v = r15[k]
        A(f"seLoo{T}K", v["k"], "settings")
        A(f"seLoo{T}Lo", _p(v["range"][0], sign=True), f"{d}: lowest leave-one-out mean")
        A(f"seLoo{T}Hi", _p(v["range"][1], sign=True), f"{d}: highest leave-one-out mean")
        A(f"seLoo{T}LoDrop", _lab(v["min_when_dropping"]), "the setting whose removal gives the lowest")
        A(f"seLoo{T}HiDrop", _lab(v["max_when_dropping"]), "the setting whose removal gives the highest")
        A(f"seLoo{T}Excl", v["n_ci_excludes_0"], "leave-one-out means whose 95% CI excludes zero")


def _r16(A, r16):
    if not r16 or "settings" not in r16:
        return
    A.section = ("R16 risk-coverage: settings whose oracle neutral flag changes a conflict-question metric "
                 "(paired 95% CI excludes zero; unadjusted, not Holm)")
    S = r16["settings"]
    groups = [(p, P, d, [v for v in S.values() if v["policy"] == p]) for p, P, d in POLICY]
    groups.append(("all", "All", "all settings", list(S.values())))
    for p, P, d, vs in groups:
        if not vs:
            continue
        A(f"rc{P}N", len(vs), f"settings, {d}")
        dd = [v["vs_plain"]["oracle+flag"]["conflict"] for v in vs]
        for m, M, md, flip in (("selective", "Sel", "selective accuracy", 1), ("coverage", "Abst", "abstention", -1),
                               ("misinfo", "Mis", "misinformation share", 1), ("strict", "Strict", "strict accuracy", 1)):
            s = [flip * _sig(x[m]["ci"]) for x in dd if m in x]
            A(f"rc{P}{M}Up", sum(z > 0 for z in s), f"{d}: neutral flag raises {md}")
            A(f"rc{P}{M}Down", sum(z < 0 for z in s), f"{d}: neutral flag lowers {md}")
            A(f"rc{P}{M}", sum(z != 0 for z in s), f"{d}: neutral flag changes {md} (either way)")


def _r17(A, r17):
    if not r17:
        return
    A.section = ("R17 RAMDocs answer mix (minimal skeleton) and clean questions by type. The neutral flag's "
                 "strict-accuracy change is the paper's \\x<G>MinOracleFlagConf (G = Mini, Four, Qwen)")
    for k, v in (r17.get("a") or {}).items():
        G = GEN_A[v["gen"]]
        for arm, AT in (("plain", "Plain"), ("oracle+flag", "Flag")):
            mix = v["conflict_mix"][arm]
            for cat, C in (("correct", "Correct"), ("misinfo", "Misinfo"), ("abstain", "Abstain"), ("other", "Other")):
                A(f"mixMin{G}{AT}{C}", _p(mix[cat]), f"{NICE[v['gen']]}, minimal skeleton, conflict questions, "
                  f"{'no flag' if arm == 'plain' else 'oracle neutral flag'}: {cat} share (%; exclusive outcomes)")
        fm = v["conflict"]["flag_minus_plain"]
        # the strict change is the paper's \\x<G>MinOracleFlagConf (with its CI); only the one-way misinformation and
        # the abstention changes are new
        for m, M in (("misinfo", "Misinfo"), ("abstain", "Abstain")):
            if m in fm:
                A(f"mixMin{G}{M}Diff", _p(fm[m]["diff"], sign=True), f"{NICE[v['gen']]}, minimal: neutral flag minus "
                  f"no flag, {m} (points)")
                A(f"mixMin{G}{M}DiffCI", _pci(fm[m]["ci"]), "95% CI")
    b = r17.get("b")
    if b and "pooled" in b:
        first = next(iter(b["settings"].values()))
        A("cleanTypeDet", NICE.get(first["detector"], first["detector"]), "the LLM detector behind R17(b)")
        A("cleanTypeK", b["k"], "settings pooled")
        for t, T, d in (("several", "Several", "several gold answers"), ("one", "One", "one gold answer")):
            lo, hi = b["flag_rate_spread"][t]
            A(f"cleanTypeRate{T}", _p(lo) if abs(hi - lo) < 1e-9 else f"{_p(lo)}--{_p(hi)}",
              f"clean questions with {d}: share the detector flags (%)")
            A(f"cleanTypeRate{T}CI", _pci(first[t]["flag_rate"]["ci"], sign=False), "95% CI")
            A(f"cleanTypeN{T}", b["n"][t], f"clean questions with {d}")
            A(f"cleanTypeFlags{T}", _num(first[t]["mean_flags"]), f"mean flags per clean question with {d}")
            A(f"cleanTypeEff{T}", _p(b["pooled"][t]["diff"], sign=True), f"llm_dir+flag effect on clean questions with {d}, "
              "equal-weight mean over settings (points)")
            A(f"cleanTypeEff{T}CI", _pci(b["pooled"][t]["ci"]), "95% CI")


def _r18(A, r18):
    if not r18 or "summary" not in r18:
        return
    A.section = ("R18 prevalence (gpt-4o-mini LLM directional flag): effect at conflict share pi; 'significantly' = "
                 "joint-bootstrap 95% band excludes zero")
    pi = r18["pi"]
    idx = {P: next((i for i, x in enumerate(pi) if abs(x - p) < 1e-9), None)
           for p, P in ((0.01, "One"), (0.05, "Five"), (0.2, "Twenty"), (0.5, "Fifty"))}
    s = r18["summary"]["all"]
    A("pvN", s["n"], "settings")
    for P, i in idx.items():
        if i is None:
            continue
        A(f"pv{P}Neg", s["n_neg"][i], f"settings with a negative point estimate at pi = {pi[i]:.2f}")
        A(f"pv{P}SigNeg", s["n_sig_neg"][i], f"settings significantly below zero at pi = {pi[i]:.2f}")
        A(f"pv{P}SigPos", s["n_sig_pos"][i], f"settings significantly above zero at pi = {pi[i]:.2f}")
    i5 = idx["Five"]
    for g, G, d in CLEAN_GROUP:
        x = r18["summary"].get(g)
        if not x:
            continue
        A(f"pvFive{G}N", x["n"], f"settings in group {d}")
        if i5 is not None:
            A(f"pvFive{G}Neg", x["n_neg"][i5], f"{d}: negative at 5%")
            A(f"pvFive{G}SigNeg", x["n_sig_neg"][i5], f"{d}: significantly negative at 5%")
            A(f"pvFive{G}SigPos", x["n_sig_pos"][i5], f"{d}: significantly positive at 5%")
    S = r18["settings"]
    be = {k: v["breakeven"] for k, v in S.items() if v.get("breakeven") is not None}
    A("pvBeN", len(be), "settings with a break-even share (K > 0 > C)")
    for reg, Rg, d in (("helps_all", "HelpsAll", "helps at every share (K >= 0 and C >= 0, not both zero)"),
                       ("hurts_all", "HurtsAll", "hurts at every share (K <= 0 and C <= 0, not both zero)")):
        A(f"pv{Rg}", sum(v["regime"] == reg for v in S.values()), f"settings whose flag {d}")
    if be:
        lo, hi = min(be, key=be.get), max(be, key=be.get)
        A("pvBeMin", _fr(be[lo], 0), "smallest break-even share (%)")
        A("pvBeMinSet", _lab(lo), "setting with the smallest")
        A("pvBeMedian", _fr(statistics.median(be.values()), 0), "median break-even share (%)")
        A("pvBeMax", _fr(be[hi], 0), "largest break-even share (%)")
        A("pvBeMaxSet", _lab(hi), "setting with the largest")
        A("pvBeCapped", sum(bool(S[k].get("breakeven_ci_upper_capped")) for k in be),
          "break-even CIs whose upper limit is 100% (>= 2.5% of replicates help at no share)")
    for g, G, d in CLEAN_GROUP:
        bg = [x for k, x in be.items() if S[k]["group"] == g]
        A(f"pvBe{G}N", len(bg), f"{d}: settings with a break-even share")
        if bg:
            A(f"pvBe{G}Min", _fr(min(bg), 0), f"{d}: smallest break-even share (%)")
            A(f"pvBe{G}Max", _fr(max(bg), 0), f"{d}: largest break-even share (%)")


def _r19(A, r19):
    if not r19 or "rho" not in r19:
        return
    A.section = "R19 H4b second form (label-robust, holdout_contains)"
    A("hFourBRho", _num(r19["rho"], 2, True), "Spearman rho over cells")
    A("hFourBP", f"{r19['p_perm']:.2f}", "two-sided permutation p")
    A("hFourBPOne", f"{r19['p_perm_one_sided_negative']:.2f}", "one-sided permutation p (rho < 0)")
    A("hFourBPerm", _int(r19["n_perm"]), "permutations")
    A("hFourBBcaCI", _numci(r19["bca_ci"], 2, True), "BCa 95% interval (5,000 question resamples)")
    # the percentile interval the paper already prints in tab:contains (2,000 draws, seed 0; R19 reproduces it), and
    # the one from the 5,000 draws behind the BCa interval. rho lies outside both: the bootstrap distribution is skewed
    # (BCa bias correction z0 = hFourBZ)
    A("hFourBPctCI", _numci(r19["percentile_ci_stored"], 2, True),
      "percentile 95% interval as in tab:contains (2,000 question resamples)")
    A("hFourBPctSameCI", _numci(r19["percentile_ci_5000"], 2, True),
      "percentile 95% interval from the 5,000 resamples behind the BCa interval")
    A("hFourBZ", f"{r19['bca_z0']:.2f}", "BCa bias correction z0 (share of replicates below rho = Phi(z0))")
    A("hFourBBelow", _fr(_ncdf(r19["bca_z0"]), 0), "share of bootstrap replicates below rho (%), Phi(z0)")
    A("hFourBN", r19["n_cells"], "cells"); A("hFourBQ", r19["n_questions"], "questions")


def _uncited(A, rev):
    ag = rev.get("judge19", {}).get("agreement", {})
    if ag:
        A.section = "Already computed, now cited: PR§19.2 matcher-judge agreement per condition (conflict responses)"
        for c, T, d in JUDGE_COND:
            v = ag.get(c)
            if not v:
                continue
            A(f"scJudgeOnly{T}", _fr(v["judge_only"]), f"{d}: judge says correct, matcher says not (% of responses)")
            A(f"scMatcherOnly{T}", _fr(v["matcher_only"]), f"{d}: matcher says correct, judge says not (%)")
            A(f"scAgree{T}", _fr(v["agree"]), f"{d}: agreement (%)")
            A(f"scN{T}", _int(v["n"]), f"{d}: responses")
    r2 = rev.get("r2", {})
    cb = {k.split(":", 1)[1]: v["closedbook"]["conflict"] for k, v in r2.items()
          if k.startswith("CBs:") and "closedbook" in v and _ok(v["closedbook"].get("conflict"))}
    if cb:
        A.section = "Already computed, now cited: ConflictBank closed-book accuracy (conflict questions, one-way strict)"
        lo, hi = min(cb.values()), max(cb.values())
        names = lambda x: " and ".join(NICE.get(g, g) for g in sorted(cb) if abs(cb[g] - x) < 1e-12)
        A("cbClosedMin", _fr(lo), f"lowest closed-book accuracy over the {len(cb)} generators with a closed-book run (%)")
        A("cbClosedMinGen", names(lo), "generator(s) at the minimum")
        A("cbClosedMax", _fr(hi), "highest closed-book accuracy (%)")
        A("cbClosedMaxGen", names(hi), "generator(s) at the maximum (ties joined)")
        A("cbClosedN", len(cb), "generators with a ConflictBank closed-book run")


def _h31_rows(rev):
    """Per H31 generator: closed-book and no-flag accuracy (points), follow rates (fractions), effects (points), p*."""
    hx, ha, hb, r2 = rev.get("H31x", {}), rev.get("H31a", {}), rev.get("H31b", {}), rev.get("r2", {})
    rows = []
    for g in H31_ORDER:
        v = hx.get(g)
        if not v:
            continue
        na = next((t["n"] for k, t in ha.items() if k.startswith(g + ":")), None)
        nb = next((t["n"] for k, t in hb.items() if k.startswith(g + ":")), None)
        plain = ((v["plain_right_arm"] * na + v["plain_wrong_arm"] * nb) / (na + nb)) if na and nb else None
        cbv = r2.get(f"CBs:{g}", {}).get("closedbook", {}).get("conflict")
        rows.append({"gen": g, "closed": None if cbv is None else 100 * cbv, "plain": plain, **v})
    return rows


def _own_draw(rows, rev):
    """H31 generators whose no-flag accuracy in the H31 run differs (as printed) from the one-answer run's (tab_cb,
    \\zpSg<G>Conflict): their H31 run drew its own no-flag answers (PR§20.1 double plain draw)."""
    r2 = rev.get("r2", {})
    out = []
    for r in rows:
        cb = r2.get(f"CBs:{r['gen']}", {}).get("plain", {}).get(PRIMARY, {}).get("conflict")
        if _ok(r["plain"]) and _ok(cb) and _p(r["plain"]) != _fr(cb):
            out.append(NICE[r["gen"]])
    return out


def _deference(A, rev):
    rows = _h31_rows(rev)
    if not rows:
        return
    A.section = ("Deference table (tab_deference): the no-flag column. The other columns already have macros: closed-book "
                 "\\zcSg<G>Conflict, f_R \\rfolRight<G> (\\rfolRight<G>CI), f_W \\rfolWrong<G> (\\rfolWrong<G>CI), "
                 "right/wrong effects \\hThirtyOneA<G> / \\hThirtyOneB<G>, p* \\be<G> (\\be<G>CI); G = Mini, Luna, Haiku, Terra. "
                 "deferPlain is the H31 run's own no-flag draw; where it differs from \\zpSg<G>Conflict (the one-answer "
                 f"run's, PR§20.1): {', '.join(_own_draw(rows, rev)) or 'none'}")
    for r in rows:
        A(f"deferPlain{GEN_A[r['gen']]}", _p(r["plain"]), f"{NICE[r['gen']]}: no-flag accuracy on the H31 run's "
          "conflict questions (%; question-weighted over the right and wrong arms)")


# --------------------------------------------------------------------------- catalogue
def catalogue11(R) -> str:
    sec, out = [None], []

    def describe(section, name, value, meaning):
        if section != sec[0]:
            out.append(f"\n## {section}\n")
            sec[0] = section
        out.append(f"- `\\{name}` = `{value}` — {meaning}")

    numbers11(R, lambda n, v: None, describe)
    head = ("# PR§20 macros (numbers11, experiments/conflict/assets11.py)\n\n"
            "Generated from R[\"rev\"][\"r11\"] (rev11_* modules) and stored R[\"rev\"] results; every R-number is "
            "exploratory (PR§20). Values are as generated into numbers.tex. Signed values carry a real minus/plus "
            "(`$-$`, `$+$`); paper_assets.numbers also emits an unsigned twin `\\a<name>` for every signed macro whose "
            "name does not end in CI. Percentages are written without the % sign. \"Significant\" in the paper means "
            "Holm-adjusted; counts below that use a 95% CI are labelled as such.\n")
    return head + "\n".join(out) + "\n"


# --------------------------------------------------------------------------- figure
def fig_riskcov11(R, out: Path):
    """R16: conflict-question operating points, one arrow per setting from no flag to the oracle neutral flag.
    x = coverage (1 - abstention), y = selective accuracy (one-way matcher, ratio estimand)."""
    S = R.get("rev", {}).get("r11", {}).get("R16", {}).get("settings")
    if not S:
        return
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch
    from .paper_assets import INK2, _style
    _style()
    fig, ax = plt.subplots(figsize=(3.1, 2.75))
    order = {"multi": 0, "minimal": 1, "single": 2, "forced": 3}
    for k, v in sorted(S.items(), key=lambda kv: order.get(kv[1]["policy"], 9)):
        p, f = v["points"]["plain"]["conflict"], v["points"]["oracle+flag"]["conflict"]
        x0, y0 = p["coverage"]["est"], p["selective"]["est"]
        x1, y1 = f["coverage"]["est"], f["selective"]["est"]
        col = POL_COL.get(v["policy"], INK2)
        ax.scatter([x0], [y0], s=7, color=col, edgecolor="none", zorder=3)
        if math.hypot(x1 - x0, y1 - y0) > 1e-9:
            ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=6, lw=0.8, color=col,
                                         linestyle="--" if v["policy"] == "minimal" else "-", shrinkA=0, shrinkB=0,
                                         zorder=2, joinstyle="miter"))
    ax.set_xlim(30, 102); ax.set_ylim(0, 85)
    ax.set_xlabel("coverage (%, 1 $-$ abstention)", fontsize=8)
    ax.set_ylabel("selective accuracy (%)", fontsize=8)
    ax.tick_params(labelsize=8)
    # the RAGuard settings sit apart (strong models abstain on most claims): one direct label under their points
    rg = [v["points"][a]["conflict"] for v in S.values() if v["bench"] == "RAGuard" for a in ("plain", "oracle+flag")]
    if rg:
        ax.text(sum(p["coverage"]["est"] for p in rg) / len(rg), min(p["selective"]["est"] for p in rg) - 3.5, "RAGuard",
                fontsize=7, color=INK2, ha="center", va="top")
    # legend names as tab_riskcov's policy headers
    pol = [(("multi",), "multi-answer", "-"), (("minimal",), "minimal skeleton", "--"),
           (("single",), "one answer or “unknown”", "-"), (("forced",), "one answer, forced", "-")]
    hs = [Line2D([], [], color=POL_COL[ps[0]], lw=1.0, ls=ls, marker=">", ms=3.5, label=lab)
          for ps, lab, ls in pol if any(v["policy"] in ps for v in S.values())]
    # how to read an arrow: dot = no flag, head = oracle neutral flag
    ax.annotate("", xy=(37, 7), xytext=(32.5, 7), arrowprops=dict(arrowstyle="-|>", mutation_scale=6, lw=0.8,
                color=INK2, shrinkA=0, shrinkB=0))
    ax.scatter([32.5], [7], s=7, color=INK2, edgecolor="none", zorder=3)
    ax.text(38, 7, "no flag $\\rightarrow$ oracle neutral flag", fontsize=7, color=INK2, va="center", ha="left")
    fig.tight_layout(pad=0.3, rect=(0, 0.12, 1, 1))
    fig.legend(handles=hs, fontsize=7, loc="lower center", ncol=2, bbox_to_anchor=(0.53, 0.0), handlelength=2.2,
               columnspacing=1.0, labelspacing=0.3, borderaxespad=0.1)
    fig.savefig(out / "fig_riskcov.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_riskcov.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------- tables
def tab_deference11(R, out: Path):
    rows = _h31_rows(R.get("rev", {}))
    if not rows:
        return
    ha, hb = R["rev"].get("H31a", {}), R["rev"].get("H31b", {})
    nq = (next(iter(ha.values()))["n"] + next(iter(hb.values()))["n"]) if ha and hb else "--"
    # generators whose H31 run drew its own no-flag answers (PR§20.1 double plain draw): the Plain column then differs
    # from the one-answer run's no-flag accuracy in tab_cb
    other = _own_draw(rows, R["rev"])
    draw = (f" For {' and '.join(other)} the H31 run drew its own no-flag answers, so Plain differs from their "
            r"one-answer no-flag accuracy elsewhere in the paper (\mbox{PR\S20.1})." if other else "")
    # R9 (PR§20.3): main-text tables mark any cell whose run-condition has more than 1% missing final answers
    from .rev11_prior import CLOSED, H31
    miss = {(a[0], a[1]) for a in R["rev"].get("r11", {}).get("R9", {}).get("above_1pct", [])}
    dag = lambda run, cond: r"$^\ddagger$" if (run, cond) in miss else ""
    lines = []
    for r in rows:
        fr, fw = r["follow_right"], r["follow_wrong"]
        g, h = r["gen"], H31.get(r["gen"])
        lines.append(f"{NICE[g]} & {_p(r['closed'])}{dag(CLOSED.get(g), 'closedbook')} & {_p(r['plain'])}{dag(h, 'plain')}"
                     f" & {_fr(fr, 0)} & {_fr(fw, 0)} & "
                     f"{_p(r['right'], sign=True)} & {_p(r['wrong'], sign=True)} & {r['breakeven']:.2f} \\\\")
        lines.append(rf"& & & {{\scriptsize {_frci(r['follow_right_ci'], 0)}}} & {{\scriptsize {_frci(r['follow_wrong_ci'], 0)}}}"
                     rf" & & & {{\scriptsize [{r['be_low']:.2f}, {r['be_high']:.2f}]}} \\[1pt]")
    # full model names at \small do not fit one column with seven numeric columns: shrink-only resize
    (out / "tab_deference.tex").write_text(r"""\begin{table}[t]
\centering\small
\setlength{\tabcolsep}{2.5pt}
\resizebox{\ifdim\width>\columnwidth\columnwidth\else\width\fi}{!}{%
\begin{tabular}{@{}lrrrrrrr@{}}
\toprule
& \multicolumn{2}{c}{Accuracy} & \multicolumn{2}{c}{Follow rate} & \multicolumn{2}{c}{Flag effect} & \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Generator & Closed & Plain & $f_R$ & $f_W$ & Right & Wrong & $p^*$ \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}}
\caption{Deference to a randomised directional flag on ConflictBank (H31; """ + str(nq) + r""" conflict questions, one-answer
prompt, one-way matcher). \emph{Closed}: closed-book accuracy (\%). \emph{Plain}: accuracy with the passages and no
flag in the H31 run (\%).""" + draw + r""" \emph{Right}, \emph{Wrong}: the registered H31 effects $\Delta_R$ and $\Delta_W$
(points) of a flag that names the false passage (right) or the true one (wrong). $f_R$, $f_W$: follow rates (\%,
registered descriptives without a test, PR\S16.3; question-bootstrap 95\% CIs below): among questions answered wrongly
without a flag, the share of right-flag responses that are correct; among questions answered correctly, the share of
wrong-flag responses that contain the misinformation answer (bidirectional match); abstentions count as neither.
$p^*$: break-even direction precision $-\Delta_W/(\Delta_R-\Delta_W)$ (exploratory; bootstrap 95\% CI below).""" + (
    r" $^\ddagger$: more than 1\% of responses lack a final answer (R9)." if any("ddagger" in l for l in lines) else "") + r"""}
\label{tab:deference}
\end{table}
""")


def tab_prior11(R, out: Path):
    r11 = R.get("rev", {}).get("r11", {})
    r2, r3, r1 = r11.get("R2", {}), r11.get("R3", {}), r11.get("R1", {})
    dets = [d for d in R3_ORDER if (isinstance(r2.get(d), dict) and "single_side" in r2[d])
            or (isinstance(r3.get(d), dict) and "single_side" in r3[d])]
    if not dets:
        return
    nq = next((r2[d]["n_conflict"] for d in dets if isinstance(r2.get(d), dict) and "n_conflict" in r2[d]), None)
    lines = []
    for d in dets:
        s = r2.get(d, {}).get("single_side") if isinstance(r2.get(d), dict) else None
        p = r3.get(d) if isinstance(r3.get(d), dict) and "single_side" in r3[d] else None
        sc = lambda x: rf"{{\scriptsize {x}}}"
        c2 = (f"{r2[d]['n_known']} & {s['n_known_flags']}/{s['n_unknown_flags']} & {_fr(s['known'])} & {_fr(s['unknown'])} & "
              f"{_fr(s['diff'], sign=True)}" if s else "-- & -- & -- & -- & --")
        c2ci = (f"& & & {sc(_frci(s['known_ci']))} & {sc(_frci(s['unknown_ci']))} & {sc(_frci(s['diff_ci'], sign=True))}"
                if s else "& & & & &")
        c3 = f"{_fr(p['single_side'])} & {_fr(p['current'])}" if p else "-- & --"
        c3ci = f"{sc(_frci(p['single_ci']))} & {sc(_frci(p['current_ci']))}" if p else "&"
        lines.append(f"{NICE[d]} & {c2} & {c3} \\\\")
        lines.append(f"{c2ci} & {c3ci} \\\\[1pt]")
    # R1 is not in this table (its macros clf* carry it into the text)
    no_cb = [NICE[d] for d in dets if not (isinstance(r2.get(d), dict) and "single_side" in r2[d])]
    no_cb_txt = (f" {' and '.join(no_cb)} {'has' if len(no_cb) == 1 else 'have'} no closed-book run." if no_cb else "")
    (out / "tab_prior.tex").write_text(r"""\begin{table*}[t]
\centering\small
\setlength{\tabcolsep}{5pt}
\begin{tabular}{@{}lrrrrrrr@{}}
\toprule
& \multicolumn{5}{c}{R2: by the detector's closed-book knowledge (single-side)} & \multicolumn{2}{c}{R3: all gold-pair flags} \\
\cmidrule(lr){2-6}\cmidrule(lr){7-8}
Detector & Known q. & Flags & Known & Unknown & Difference & Single-side & All verdicts \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\caption{Direction precision and prior knowledge on ConflictBank (PR\S20.2 R2, R3, exploratory; \%, 95\% bootstrap CIs
below). Direction precision: among a detector's flags on the gold pair, the share naming the misinformation passage.
\emph{R2}, in the detector's run with gpt-4o-mini as generator: a question is \emph{known} to the detector model when
its closed-book answer is correct on at least two of three seeds (or its one seed); \emph{Known q.}: such conflict
questions (of """ + str(nq) + r"""); \emph{Flags}: flags naming exactly one side, on known/unknown questions;
\emph{Known}, \emph{Unknown}: direction precision over those flags; \emph{Difference}: known minus unknown (points);
question-bootstrap CIs.""" + no_cb_txt + r""" \emph{R3}, pooled over the detector's crossed generator cells: direction precision counting
only flags that name exactly one side (\emph{Single-side}) or every flag, with ``both'' and ``unclear'' as misses
(\emph{All verdicts}, the earlier definition); question-cluster bootstrap CIs.}
\label{tab:prior}
\end{table*}
""")


def tab_missing11(R, out: Path, top: int = 25):
    r9 = R.get("rev", {}).get("r11", {}).get("R9")
    if not r9 or "above_1pct" not in r9:
        return
    ab = r9["above_1pct"]
    lines = []
    for run, c, n, ms, cs in ab[:top]:
        lines.append(rf"\texttt{{{_esc(run)}}} & {_cond(c)} & {_fr(ms)} & {_fr(cs) if cs is not None else '--'} \\")
    more = len(ab) - min(len(ab), top)
    rest = ""
    if more:
        tail = ab[top:]
        rest = (f" The {more} further cells above 1\\% (not shown) miss {_fr(min(t[3] for t in tail))}--"
                f"{_fr(max(t[3] for t in tail))}\\%.")
    t = r9["totals"]
    caps = {}
    for x in r9["runs"].values():
        if x.get("cap"):
            caps.setdefault(int(x["cap"]), set()).add(_g(x["model"]))
    cap_txt = " and ".join(f"{_int(c)} tokens for {', '.join(sorted(ms))}" for c, ms in sorted(caps.items()))
    from collections import Counter
    by_gen = Counter(_g(r9["runs"][r]["model"]) for r, *_ in ab)
    gen_txt = ", ".join(f"{k} {g}" for g, k in by_gen.most_common())
    ns = [x[2] for x in ab[:top]]
    (out / "tab_missing.tex").write_text(r"""\begin{table}[t]
\centering\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}llrr@{}}
\toprule
Run & Condition & Missing & At cap \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\caption{Responses with no final-answer line (PR\S20.3 R9, exploratory): the """ + str(min(len(ab), top)) + r""" worst of the
""" + str(r9["n_cells_above_1pct"]) + r""" run $\times$ condition cells above 1\% (of """ + str(r9["n_cells"]) + r""" cells in
""" + str(r9["n_runs"]) + r""" runs; """ + f"{100 * t['missing_share']:.2f}" + r"""\% of all """ + _int(t["n"]) + r""" responses); by generator, the
cells above 1\% are """ + gen_txt + r""". A cell holds one response per question and seed (""" + f"{_int(min(ns))}--{_int(max(ns))}" + r""" here).
\emph{Missing}: \% of the cell's responses. \emph{At cap}: \% that stopped at the completion-token cap, which is
""" + cap_txt + r""" (not defined for multi-call methods).""" + rest + r"""}
\label{tab:missing}
\end{table}
""")


def tab_riskcov11(R, out: Path):
    S = R.get("rev", {}).get("r11", {}).get("R16", {}).get("settings")
    if not S:
        return
    bench_order = {"ConflictBank": 0, "CB majority": 1, "RAMDocs": 2, "RAGuard": 3}
    gen_order = {g: i for i, g in enumerate(("gpt-4o-mini", "gpt-4.1", "Qwen2.5-7B", "gpt-6-luna", "gpt-6.1-terra",
                                             "claude-haiku-5-5", "claude-sonnet-5-5"))}
    lines = []
    for p, P, d in POLICY:
        ks = sorted((k for k, v in S.items() if v["policy"] == p),
                    key=lambda k: (bench_order.get(S[k]["bench"], 9), gen_order.get(S[k]["gen"], 9)))
        if not ks:
            continue
        lines.append(r"\midrule" + "\n" + rf"\multicolumn{{9}}{{@{{}}l}}{{\emph{{Answer policy: {d}}}}} \\")
        for k in ks:
            v = S[k]
            pl, fl = v["points"]["plain"]["conflict"], v["points"]["oracle+flag"]["conflict"]
            dd = v["vs_plain"]["oracle+flag"]["conflict"]
            cells = []
            for m in ("coverage", "selective", "misinfo", "strict"):
                mark = r"$^\dagger$" if _sig(dd.get(m, {}).get("ci")) else r"\phantom{$^\dagger$}"
                cells += [_p(pl[m]["est"]), _p(fl[m]["est"]) + mark]
            lines.append(f"{v['bench']}, {_g(v['gen'])} & " + " & ".join(cells) + r" \\")
    (out / "tab_riskcov.tex").write_text(r"""\begin{table*}[t]
\centering\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lrrrrrrrr@{}}
\toprule
& \multicolumn{2}{c}{Coverage} & \multicolumn{2}{c}{Selective accuracy} & \multicolumn{2}{c}{Misinformation} & \multicolumn{2}{c}{Strict accuracy} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}
Setting & No flag & Flag & No flag & Flag & No flag & Flag & No flag & Flag \\
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\caption{Risk--coverage operating points on conflict questions (PR\S20.4 R16, exploratory; """ + str(len(S)) + r""" settings):
no flag against the oracle neutral flag, in \%. \emph{Coverage}: 1 $-$ abstention; \emph{Selective accuracy}: correct
responses over answered (non-abstaining) responses, summed over questions and seeds; \emph{Misinformation}: share of responses containing
a wrong answer as a whole-word span; \emph{Strict accuracy}: one-way strict. All four use the one-way matcher (PR\S20.3
R12). $^\dagger$: the paired question-bootstrap 95\% CI of the change excludes zero (unadjusted). The minimal skeleton
states no answer policy. Figure~\ref{fig:riskcov} plots the coverage and selective-accuracy columns.}
\label{tab:riskcov}
\end{table*}
""")


def tab_prev11(R, out: Path):
    r18 = R.get("rev", {}).get("r11", {}).get("R18")
    if not r18 or "settings" not in r18:
        return
    S = r18["settings"]
    pre_order = {"CB": 0, "CBs": 1, "CBmaj": 2, "CBf": 3, "CBn": 4, "RD": 5, "RG": 6}
    gen_order = {g: i for i, g in enumerate(("gpt-4o-mini", "gpt-4.1", "Qwen2.5-7B", "gpt-6-luna", "gpt-6.1-terra",
                                             "claude-haiku-5-5", "claude-sonnet-5-5"))}

    def order(k):
        pre, rest = k.split(":", 1)
        g, _, sk = rest.partition("|")
        return pre_order.get(pre, 9), gen_order.get(g, 9), sk == "minimal"
    lines = []
    for g, G, d in CLEAN_GROUP:
        ks = sorted((k for k, v in S.items() if v["group"] == g), key=order)
        if not ks:
            continue
        lines.append(r"\midrule" + "\n" + rf"\multicolumn{{4}}{{@{{}}l}}{{\emph{{{d}}}}} \\")
        for k in ks:
            v = S[k]
            if v.get("breakeven") is not None:
                lo, hi = v["breakeven_ci"]
                be = (f"{_fr(v['breakeven'], 0)} [{_fr(lo, 0)}, {_fr(hi, 0)}]"
                      + (r"$^\dagger$" if v.get("breakeven_ci_upper_capped") else ""))
            else:
                be = {"helps_all": r"$\uparrow$", "hurts_all": r"$\downarrow$"}.get(v["regime"], "--")
            # every row names its prompt (the group header names the benchmark); the majority variant is one-answer
            lab = (f"{'majority' if v['bench'] == 'CB majority' else PREV_POL.get(v['policy'], v['policy'])}, "
                   f"{_g(v['gen'])}")
            lines.append(f"{lab} & {_p(v['K']['est'], sign=True)} & {_p(v['C']['est'], sign=True)} & {be} \\\\")
    (out / "tab_prev11.tex").write_text(r"""\begin{table}[t]
\centering\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lrrl@{}}
\toprule
Setting & $K$ & $C$ & Break-even [95\% CI] \\
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\caption{Prevalence (PR\S20.4 R18, exploratory): the gpt-4o-mini LLM detector's directional flag. $K$, $C$: its effect
(points) on conflict and on clean questions; at a conflict share $\pi$ the effect is $\pi K + (1-\pi) C$.
\emph{Break-even}: $-C/(K-C)$, the share (\%) above which the flag helps, where $K > 0 > C$; CI from a joint question
bootstrap. $^\dagger$: at least 2.5\% of replicates help at no share, so the upper limit is 100. $\uparrow$: helps at
every share ($K, C \geq 0$); $\downarrow$: hurts at every share ($K, C \leq 0$). Rows name the prompt (\emph{majority}:
the one-answer prompt on the majority variant); settings are grouped by how their clean questions were built.}
\label{tab:prev11}
\end{table}
""")


def write11(R, out: Path):
    out = Path(out)
    fig_riskcov11(R, out)
    tab_deference11(R, out); tab_prior11(R, out); tab_missing11(R, out); tab_riskcov11(R, out); tab_prev11(R, out)
