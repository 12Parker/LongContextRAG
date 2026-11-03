"""
Paper macros and tables for the PR§21 runs (rev21.py: R["rev"]["r21"] holds runs/rev21/rev21.json's parts c1, c3, c4,
c5, c6, c7, c9, c11). numbers21 runs inside paper_assets.numbers after numbers11; write21 after write11.
Scales as rev21 stores them: rates, shares, accuracies and precisions are fractions; effects, sums and MAEs are points
(x100); break-even p* is a fraction. A part that did not run emits no macros, and its table rows read "not run".
"""
from __future__ import annotations

from pathlib import Path

from .assets11 import _Adder, _fr, _frci, _num, _numci, _ok, _p, _pci

GEN = (("mini", "Mini", "gpt-4o-mini"), ("haiku", "Haiku", "Claude Haiku"), ("luna", "Luna", "gpt-6-luna"),
       ("terra", "Terra", "gpt-6.1-terra"))
FRESH = (("luna", "Luna", "gpt-6-luna"), ("haiku", "Haiku", "Claude Haiku"), ("terra", "Terra", "gpt-6.1-terra"),
         ("mini", "Mini", "gpt-4o-mini"))


def _d(x):
    return isinstance(x, dict)


def _rng(A, name, vals, fmt, meaning):
    vals = [v for v in vals if _ok(v)]
    if vals:
        A(f"{name}Min", fmt(min(vals)), f"smallest {meaning}")
        A(f"{name}Max", fmt(max(vals)), f"largest {meaning}")


# --------------------------------------------------------------------------- macros
def numbers21(R, add, describe=None):
    A = _Adder(add, describe)
    r = R.get("rev", {}).get("r21", {})
    _c1(A, r.get("c1", {})); _c4(A, r.get("c4", {})); _c5(A, r.get("c5", {})); _c6(A, r.get("c6", {}))
    _c9(A, r.get("c9", {})); _c3(A, r.get("c3", {})); _c7(A, r.get("c7", {})); _c11(A, r.get("c11", {}))


def _c1(A, c1):
    A.section = "C1 artefact probes (PR§21.1, ConflictBank eval)"
    ao, sp, sw, swb, dr = [], [], [], [], []
    outcomes = {"survives": 0, "inverts": 0, "chance": 0}
    for s, S, nice in GEN:
        o = c1.get(s)
        if not _d(o):
            continue
        if "answer_only" in o:
            a = o["answer_only"]
            A(f"cxAo{S}", _fr(a["accuracy"]["est"], 0), f"{nice}: answer-only two-choice accuracy (%; mean of both orders)")
            A(f"cxAo{S}CI", _frci(a["accuracy"]["ci"], 0), f"95% CI of cxAo{S}")
            A(f"cxAoSame{S}", _fr(a["same_both_orders"], 0), f"{nice}: share answering the same in both orders (%)")
            ao.append(a["accuracy"]["est"])
        if "single_passage" in o:
            p = o["single_passage"]
            A(f"cxSp{S}", _fr(p["pairwise"]["est"], 0), f"{nice}: single-passage pairwise accuracy (%)")
            A(f"cxSp{S}CI", _frci(p["pairwise"]["ci"], 0), f"95% CI of cxSp{S}")
            A(f"cxSpMis{S}", _fr(p["share_false_misinfo"]["est"], 0), f"{nice}: misinformation passages scored >= 50 (%)")
            A(f"cxSpCor{S}", _fr(p["share_false_correct_conflict"]["est"], 0), f"{nice}: correct passages scored >= 50 (%)")
            sp.append(p["pairwise"]["est"])
        if "swap" in o:
            w = o["swap"]
            A(f"cxSwap{S}", _fr(w["content_precision"]["est"], 0), f"{nice}: content precision after the object swap (%)")
            A(f"cxSwap{S}CI", _frci(w["content_precision"]["ci"], 0), f"95% CI of cxSwap{S}")
            A(f"cxSwapBefore{S}", _fr(w["before_single_side"]["est"], 0), f"{nice}: single-side precision on the same items before the swap (%)")
            A(f"cxSwapOut{S}", w["outcome"] or "--", f"{nice}: registered swap outcome")
            sw.append(w["content_precision"]["est"]); swb.append(w["before_single_side"]["est"])
            if w["outcome"] in outcomes:
                outcomes[w["outcome"]] += 1
        if _d(o.get("eval_direction_11_3")):
            A(f"cxDirEval{S}", _fr(o["eval_direction_11_3"]["est"], 0), f"{nice}: eval direction accuracy (§11.3, %)")
            dr.append(o["eval_direction_11_3"]["est"])
    any_swap = next((c1[s]["swap"] for s, *_ in GEN if _d(c1.get(s)) and "swap" in c1[s]), None)
    if any_swap:
        A("cxSwapN", any_swap["n_items"], "usable object-swap items")
    _rng(A, "cxAo", ao, lambda v: _fr(v, 0), "answer-only accuracy (%)")
    _rng(A, "cxSp", sp, lambda v: _fr(v, 0), "single-passage pairwise accuracy (%)")
    _rng(A, "cxSwap", sw, lambda v: _fr(v, 0), "content precision after the swap (%)")
    _rng(A, "cxSwapBefore", swb, lambda v: _fr(v, 0), "single-side precision before the swap (%)")
    _rng(A, "cxDirEval", dr, lambda v: _fr(v, 0), "eval direction accuracy (%)")
    if sw:
        for k, nm in (("survives", "Survive"), ("inverts", "Invert"), ("chance", "Chance")):
            A(f"cxSwap{nm}N", outcomes[k], f"detectors whose swap outcome is '{k}'")
        A("cxProbeN", len(sw), "detectors with the swap probe")
    if ao:
        A("cxAoAboveDirN", sum(bool(c1[s].get("rule_a_no_passage_needed")) for s, *_ in GEN if _d(c1.get(s))),
          "detectors whose answer-only accuracy is at least their direction accuracy minus 5 points (rule a)")


def _h31_macros(A, pre, s, nice):
    A(f"{pre}FolR", _fr(s["f_R"]["est"], 0), f"{nice}: f_R (%)")
    A(f"{pre}FolRCI", _frci(s["f_R"]["ci"], 0), f"95% CI of {pre}FolR")
    A(f"{pre}FolW", _fr(s["f_W"]["est"], 0), f"{nice}: f_W (%)")
    A(f"{pre}FolWCI", _frci(s["f_W"]["ci"], 0), f"95% CI of {pre}FolW")
    A(f"{pre}DR", _p(s["dR"], 1, True), f"{nice}: right-flag effect (points)")
    A(f"{pre}DW", _p(s["dW"], 1, True), f"{nice}: wrong-flag effect (points)")
    A(f"{pre}Sum", _p(s["sum"], 1, True), f"{nice}: Δ_R + Δ_W (points)")
    A(f"{pre}SumCI", _pci(s["sum_ci"]), f"95% CI of {pre}Sum")
    A(f"{pre}Be", _num(s["breakeven"], 2), f"{nice}: break-even p*")
    A(f"{pre}BeCI", _numci(s["breakeven_ci"], 2), f"95% CI of {pre}Be")


def _c4(A, c4):
    A.section = "C4 randomised direction on ConflictBank-fresh (PR§21.2, H40)"
    holds = 0; n = 0
    for s, S, nice in FRESH:
        o = c4.get(s)
        if not _d(o):
            continue
        _h31_macros(A, f"cxFr{S}", o, f"{nice} (fresh)")
        if "H40a" in o:
            n += 1; holds += o["H40a"]["holds"]
            A(f"cxFr{S}Pred", _p(o["H40c"]["pred"], 1, True), f"{nice}: frozen prediction (points)")
            A(f"cxFr{S}Obs", _p(o["H40c"]["obs"], 1, True), f"{nice}: observed effect over 300 conflict questions (points)")
    if n:
        A("cxFrHoldN", holds, "generators whose fresh f_R and f_W both lie in the frozen eval CIs (H40a)")
        A("cxFrN", n, "generators tested in H40")
    b = c4.get("H40b")
    if _d(b):
        for k, nm in (("a", "R"), ("b", "W"), ("c", "Sum")):
            A(f"cxFr{nm}SigN", sum(v["supported"] for v in b[k].values()), f"H40b ({k}) supported after Holm")
    c = c4.get("H40c")
    if _d(c):
        A("cxFrMae", f"{c['mae']:.1f}", "H40c MAE of the frozen predictions (points)")
        A("cxFrMaeZero", f"{c['mae_zero']:.1f}", "MAE of predicting zero (points)")
        A("cxFrMaeOnes", f"{c['mae_ones']:.1f}", "MAE of the formula with fR = fW = 1 (points)")
        A("cxFrHolds", "holds" if c["holds"] else "fails", "H40c verdict (MAE <= 3)")


def _c5(A, c5):
    A.section = "C5 stated confidence (PR§21.3, H41)"
    wd = []
    sa = sb = n = 0
    for s, S, nice in GEN:
        o = c5.get(s)
        if not _d(o):
            continue
        n += 1
        for arm, an in (("c55", "Low"), ("c95", "High")):
            A(f"cxCf{S}FolW{an}", _fr(o[arm]["f_W"]["est"], 0), f"{nice}: f_W at stated {arm[1:]}% (%)")
            A(f"cxCf{S}FolR{an}", _fr(o[arm]["f_R"]["est"], 0), f"{nice}: f_R at stated {arm[1:]}% (%)")
        A(f"cxCf{S}WDiff", _p(o["H41a"]["diff"], 1, True), f"{nice}: Δ_W(55) − Δ_W(95) (points)")
        A(f"cxCf{S}WDiffCI", _pci(o["H41a"]["ci"]), f"95% CI of cxCf{S}WDiff")
        A(f"cxCf{S}RDiff", _p(o["H41b"]["diff"], 1, True), f"{nice}: Δ_R(95) − Δ_R(55) (points)")
        A(f"cxCf{S}RDiffCI", _pci(o["H41b"]["ci"]), f"95% CI of cxCf{S}RDiff")
        wd.append(o["H41a"]["diff"]); sa += o["H41a"]["supported"]; sb += o["H41b"]["supported"]
    if n:
        A("cxCfN", n, "generators in H41")
        A("cxCfWSigN", sa, "H41a supported after Holm")
        A("cxCfRSigN", sb, "H41b supported after Holm")
        _rng(A, "cxCfWDiff", wd, lambda v: _p(v, 1, True), "Δ_W(55) − Δ_W(95) (points)")


def _c6(A, c6):
    A.section = "C6, C8 Sonnet and Qwen on the H31 design (PR§21.4)"
    for s, S, nice in (("sonnet", "Sonnet", "Claude Sonnet"), ("qwen", "Qwen", "Qwen2.5-7B")):
        if _d(c6.get(s)):
            _h31_macros(A, f"cxRd{S}", c6[s], f"{nice} (eval, one seed)")


def _c9(A, c9):
    A.section = "C9 closed-book on fresh and for Sonnet (PR§21.5)"
    cb = c9.get("closedbook", {})
    vals = []
    for s, S in (("mini", "Mini"), ("luna", "Luna"), ("haiku", "Haiku"), ("terra", "Terra"), ("qwen", "Qwen"),
                 ("sonnet_eval", "SonnetEval")):
        o = cb.get(s)
        if _d(o):
            A(f"cxClosed{S}", _fr(o["accuracy"]["est"], 1), f"{s}: closed-book accuracy (%)")
            A(f"cxClosedKnown{S}", o["n_known_conflict"], f"{s}: known conflict questions")
            if s != "sonnet_eval":
                vals.append(o["accuracy"]["est"])
    _rng(A, "cxClosedFresh", vals, lambda v: _fr(v, 1), "fresh closed-book accuracy (%)")
    diffs, pos = [], 0
    for s, S in (("mini", "Mini"), ("luna", "Luna"), ("haiku", "Haiku"), ("terra", "Terra"), ("qwen", "Qwen")):
        o = c9.get("R2_fresh", {}).get(s)
        if _d(o):
            ss = o["single_side"]
            A(f"cxPriorFr{S}Known", _fr(ss["known"], 0), f"{s}: fresh single-side precision on known facts (%)")
            A(f"cxPriorFr{S}Unknown", _fr(ss["unknown"], 0), f"{s}: on unknown facts (%)")
            A(f"cxPriorFr{S}KnownN", o["n_known"], f"{s}: known fresh conflict questions")
            if _ok(ss["diff"]):
                diffs.append(100 * ss["diff"]); pos += bool(_ok(ss["diff_ci"][0]) and ss["diff_ci"][0] > 0)
    if diffs:
        _rng(A, "cxPriorFrDiff", diffs, lambda v: _p(v, 0, True), "fresh known-minus-unknown precision (points)")
        A("cxPriorFrDiffPos", pos, "fresh detectors whose known-minus-unknown CI lies above zero")
        A("cxPriorFrN", len(diffs), "fresh detectors in the split")
    o = c9.get("R2_sonnet_eval")
    if _d(o):
        ss = o["single_side"]
        A("cxPriorSonnetKnown", _fr(ss["known"], 0), "Sonnet detector: eval precision on known facts (%)")
        A("cxPriorSonnetUnknown", _fr(ss["unknown"], 0), "on unknown facts (%)")
        A("cxPriorSonnetKnownN", o["n_known"], "known eval conflict questions for Sonnet")
    kn = [v["n_known"] for v in c9.get("R4_fresh", {}).values() if _d(v)]
    if kn:
        A("cxFolFreshKnownMax", max(kn), "largest number of known questions among the fresh H31 arms")


def _c3(A, c3):
    A.section = "C3 HealthContradict answer key (PR§21.6)"
    for s, S, nice in GEN[:3]:
        o = c3.get(s)
        if not _d(o):
            continue
        A(f"cxHcKey{S}", _p(o["answer_key"], 1), f"{nice}: answer-key accuracy (%)")
        kd = o["key_minus_oracle_dir"]
        A(f"cxHcKeyDir{S}", _p(kd["diff"], 1, True), f"{nice}: answer key minus oracle directional flag (points)")
        A(f"cxHcKeyDir{S}CI", _pci([kd["ci_low"], kd["ci_high"]]), f"95% topic-cluster CI of cxHcKeyDir{S}")
        kp = o["key_minus_plain"]
        A(f"cxHcKeyPlain{S}", _p(kp["diff"], 1, True), f"{nice}: answer key minus no flag (points)")


def _c7(A, c7):
    if not _d(c7.get("haiku_detector")):
        return
    A.section = "C7 Haiku detector on the RAMDocs hold-out (PR§21.7)"
    for who, W in (("haiku_detector", "Haiku"), ("mini_detector", "Mini")):
        for c, C in (("llm_dir+flag_directional", "Dir"), ("llm_dir+remove_suspect", "Rem")):
            for part, P in (("conflict", "K"), ("clean", "C")):
                e = c7[who].get(f"{c}|{part}")
                if _d(e):
                    A(f"cxHd{W}{C}{P}", _p(e["effect"], 1, True), f"{who} {c} effect on {part} questions (points)")
                    A(f"cxHd{W}{C}{P}CI", _pci(e["ci"]), "95% CI")
    for who, W in (("haiku", "Haiku"), ("mini", "Mini")):
        f = c7["flags"][who]
        A(f"cxHd{W}FlagClean", _fr(f["flag_rate_clean"], 0), f"{who} detector: clean questions flagged (%)")
        A(f"cxHd{W}FlagConf", _fr(f["flag_rate_conflict"], 0), f"{who} detector: conflict questions flagged (%)")


def _c11(A, c11):
    if not _d(c11.get("nli_sent")):
        return
    A.section = "C11 sentence-pair NLI on ConflictBank (PR§21.8)"
    f = c11["flags"]["nli_sent"]
    A("cxNsFlagConf", _fr(f["flag_rate_conflict"], 0), "nli_sent: conflict questions with a flag (%)")
    A("cxNsFlagClean", _fr(f["flag_rate_clean"], 0), "nli_sent: clean questions with a flag (%)")
    pl = c11.get("quality", {}).get("pair_level") or {}
    if pl:
        A("cxNsPairP", _fr(pl.get("precision"), 0), "nli_sent pair-level precision (%)")
        A("cxNsPairR", _fr(pl.get("recall"), 0), "nli_sent pair-level recall (%)")
    for c, C in (("nli_sent+flag", "Flag"), ("nli_sent+remove", "Rem")):
        for part, P in (("conflict", "K"), ("clean", "C")):
            e = c11["nli_sent"].get(f"{c}|{part}")
            if _d(e):
                A(f"cxNs{C}{P}", _p(e["effect"], 1, True), f"{c} effect on {part} questions (points)")
                A(f"cxNs{C}{P}CI", _pci(e["ci"]), "95% CI")


# --------------------------------------------------------------------------- tables
def tab_probe21(R, out: Path):
    c1 = R.get("rev", {}).get("r21", {}).get("c1", {})
    rows = []
    for s, S, nice in GEN:
        o = c1.get(s)
        if not _d(o):
            rows.append(f"{nice} & \\multicolumn{{6}}{{l}}{{not run}} \\\\"); continue
        a, p, w = o.get("answer_only"), o.get("single_passage"), o.get("swap")
        cell = lambda d, k: _fr(d[k]["est"], 0) if _d(d) else "--"
        ci = lambda d, k: f"{{\\scriptsize {_frci(d[k]['ci'], 0)}}}" if _d(d) else ""
        rows.append(f"{nice} & {_fr(o['eval_direction_11_3']['est'], 0)} & {cell(a, 'accuracy')} & {cell(p, 'pairwise')} & "
                    f"{_fr(w['before_single_side']['est'], 0) if _d(w) else '--'} & {cell(w, 'content_precision')} & "
                    f"{w['outcome'] if _d(w) else '--'} \\\\")
        rows.append(f" & & {ci(a, 'accuracy')} & {ci(p, 'pairwise')} & & {ci(w, 'content_precision')} & \\\\[1pt]")
    n = next((c1[s]["swap"]["n_items"] for s, *_ in GEN if _d(c1.get(s)) and "swap" in c1[s]), "--")
    tex = r"""\begin{table}[t]
\centering\small
\setlength{\tabcolsep}{3pt}
\resizebox{\ifdim\width>\columnwidth\columnwidth\else\width\fi}{!}{%
\begin{tabular}{@{}lrrrrrl@{}}
\toprule
& Direction & Answer & Single & \multicolumn{2}{c}{Object swap} & \\
\cmidrule(lr){5-6}
Detector & accuracy & only & passage & before & after & Outcome \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\caption{Artefact probes on ConflictBank's 300 conflict questions (C1, \mbox{PR\S21.1}; exploratory, one seed;
percentages, 95\% item-bootstrap CIs below). \emph{Direction accuracy}: the detector's direction accuracy on the eval
set (\mbox{PR\S11.3}). \emph{Answer only}: no passage; the share of the two candidate answers picked correctly,
averaged over both orders. \emph{Single passage}: one passage alone scored for the probability that it is false; the
share of questions whose misinformation passage scores higher (ties count half). \emph{Object swap}: on """ + str(n) + r"""
usable items the two answers are exchanged between the passages, so each passage keeps its style and argues the other
side; the share of single-side verdicts on the true pair that name the passage stating the wrong answer, before and
after the swap. \emph{Outcome} (registered): survives if the CI lies above 50, inverts if below, chance if it
includes 50.}
\label{tab:probe21}
\end{table}
"""
    (Path(out) / "tab_probe21.tex").write_text(tex)


def tab_defer21(R, out: Path):
    r = R.get("rev", {}).get("r21", {})
    c4, c5, c6 = r.get("c4", {}), r.get("c5", {}), r.get("c6", {})
    rows = []

    def line(label, s, extra=""):
        if not _d(s):
            return [f"{label} & \\multicolumn{{5}}{{l}}{{not run}} \\\\"]
        return [f"{label} & {_fr(s['f_R']['est'], 0)} & {_fr(s['f_W']['est'], 0)} & {_p(s['dR'], 1, True)} & "
                f"{_p(s['dW'], 1, True)} & {_num(s['breakeven'], 2)}{extra} \\\\",
                f" & {{\\scriptsize {_frci(s['f_R']['ci'], 0)}}} & {{\\scriptsize {_frci(s['f_W']['ci'], 0)}}} & & & "
                f"{{\\scriptsize {_numci(s['breakeven_ci'], 2)}}} \\\\[1pt]"]
    rows.append(r"\multicolumn{6}{@{}l}{\emph{ConflictBank-fresh, randomised direction (H40)}} \\")
    for s, S, nice in FRESH:
        o = c4.get(s)
        mark = "" if s != "mini" else r"$^\dagger$"
        rows += line(nice + mark, o)
    rows.append(r"\midrule")
    rows.append(r"\multicolumn{6}{@{}l}{\emph{Eval set, one seed (C6, C8)}} \\")
    rows += line("Claude Sonnet", c6.get("sonnet")) + line("Qwen2.5-7B", c6.get("qwen"))
    rows.append(r"\midrule")
    rows.append(r"\multicolumn{6}{@{}l}{\emph{Eval set, stated confidence 55\% / 95\% (H41)}} \\")
    for s, S, nice in GEN:
        o = c5.get(s)
        rows += line(f"{nice}, 55\\%", o.get("c55") if _d(o) else None) + line(f"{nice}, 95\\%", o.get("c95") if _d(o) else None)
    tex = r"""\begin{table}[t]
\centering\small
\setlength{\tabcolsep}{3pt}
\resizebox{\ifdim\width>\columnwidth\columnwidth\else\width\fi}{!}{%
\begin{tabular}{@{}lrrrrr@{}}
\toprule
Generator & $f_R$ & $f_W$ & Right & Wrong & $p^*$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\caption{Deference to randomised directional flags in the \mbox{PR\S21} runs (one-answer prompt, one-way matcher;
columns as Table~\ref{tab:deference}). Fresh: 300 new conflict questions, predictions frozen before the flagged runs.
$^\dagger$: gpt-4o-mini's fresh run predates \mbox{PR\S21} (H36). Stated confidence: the same flags, each suspect
labelled with a detector confidence of 55\% or 95\%.}
\label{tab:defer21}
\end{table}
"""
    (Path(out) / "tab_defer21.tex").write_text(tex)


def write21(R, out: Path):
    tab_probe21(R, out); tab_defer21(R, out)
