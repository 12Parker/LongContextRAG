"""
Paper macros and appendix tables for §19 (revision10.py). numbers10 runs inside paper_assets.numbers after numbers9;
write10 after write9.
"""
from __future__ import annotations

from pathlib import Path

from .assets7 import _ci_pt, _pt

GTAG = {"gpt-4o-mini": "Mini", "gpt-6-luna": "Luna", "claude-haiku-5-5": "Haiku", "gpt-6.1-terra": "Terra"}
HCTAG = {"mini": "Mini", "luna": "Luna", "haiku": "Haiku"}
NICE10 = {"gpt-4o-mini": "gpt-4o-mini", "gpt-6-luna": "gpt-6-luna", "claude-haiku-5-5": "Claude Haiku",
          "gpt-6.1-terra": "gpt-6.1-terra", "Qwen2.5-7B": "Qwen2.5-7B"}


def _yn(b):
    return "yes" if b else "no"


def numbers10(R, add):
    rev = R["rev"]
    f = rev.get("fresh19")
    if f and "H32" in f:
        h = f["H32"]
        add("frN", f["n_cells"]); add("frSlope", f"{h['slope']:.2f}")
        add("frSlopeCI", f"[{h['slope_ci'][0]:.2f}, {h['slope_ci'][1]:.2f}]")
        add("frIntercept", _pt(h["intercept"], True))
        add("frMae", _pt(h["mae"])); add("frMaeCI", f"[{_pt(h['mae_ci'][0])}, {_pt(h['mae_ci'][1])}]")
        add("frR", f"{h['r']:.2f}"); add("frRWithin", f"{h['r_within']:.2f}")
        for k, tag in (("gen_mean", "Gen"), ("ones", "Ones"), ("marginal", "Marg"), ("zero", "Zero")):
            add(f"frMae{tag}", _pt(h["baseline_mae"][k]))
        add("frPGen", f"{h['p_holm_vs_gen_mean']:.3f}"); add("frPOnes", f"{h['p_holm_vs_ones']:.3f}")
        add("frSignAgree", h["sign_agree"])
        add("frCOne", _yn(h["c1"])); add("frCTwo", _yn(h["c2"])); add("frCThree", _yn(h["c3"]))
        add("frSupported", _yn(h["supported"]))
        if "max_frozen_drift" in f:
            add("frDrift", f"{f['max_frozen_drift']:.2f}")
        s = f["H33"]
        add("frBigN", s["n_big"]); add("frBigAgree", s["agree"])
        add("frHarmN", len(s["harm_tests"])); add("frHarmConf", s["harm_confirmed"]); add("frSignSupported", _yn(s["supported"]))
        add("frWithinP", f"{f['H34']['p']:.3f}"); add("frWithinSupported", _yn(f["H34"]["supported"]))
        c = f.get("H35")
        if c:
            add("frCleanMae", _pt(c["mae"])); add("frCleanSign", c["sign_agree"]); add("frCleanN", c["n"])
            add("frCleanObs", _pt(c["obs_mean"], True)); add("frCleanPred", _pt(c["pred_mean"], True))
        st = f.get("H36")
        if st:
            add("frFollowR", f"{100 * st['follow_right']:.0f}"); add("frFollowW", f"{100 * st['follow_wrong']:.0f}")
            add("frFollowRCI", f"[{100 * st['follow_right_ci'][0]:.0f}, {100 * st['follow_right_ci'][1]:.0f}]")
            add("frFollowWCI", f"[{100 * st['follow_wrong_ci'][0]:.0f}, {100 * st['follow_wrong_ci'][1]:.0f}]")
            add("frStable", _yn(st["supported"]))
        for g, v in f.get("selection", {}).items():
            t = GTAG[g]
            add(f"frSel{t}", NICE10.get(v["chosen"], v["chosen"])); add(f"frSel{t}Obs", _pt(v["chosen_obs"], True))
            add(f"frBest{t}", NICE10.get(v["best"], v["best"])); add(f"frBest{t}Obs", _pt(v["best_obs"], True))
            if v["mini_obs"] is not None:
                add(f"frMiniDet{t}Obs", _pt(v["mini_obs"], True))
    hc = rev.get("hc19", {})
    for g, v in hc.items():
        t = HCTAG[g]
        add(f"hcPlain{t}", _pt(v["plain"])); add(f"hcN{t}", v["n"])
        if "closedbook" in v:
            add(f"hcCb{t}", _pt(v["closedbook"]))
        for key, tag in (("H37a", "Dir"), ("H37b", "Neu"), ("H37c", "DirNeu"), ("llm", "Llm")):
            if key in v:
                add(f"hc{tag}{t}", _pt(v[key]["diff"], True)); add(f"hc{tag}{t}CI", _ci_pt(v[key]["ci_low"], v[key]["ci_high"]))
                if "p_holm" in v[key]:
                    add(f"hc{tag}{t}P", f"{v[key]['p_holm']:.3f}")
        if "llm" in v and "pred" in v["llm"]:
            add(f"hcLlmPred{t}", _pt(v["llm"]["pred"], True)); add(f"hcLlmRight{t}", f"{100 * v['llm']['right']:.0f}")
            add(f"hcLlmWrong{t}", f"{100 * v['llm']['wrong']:.0f}")
    if hc:
        add("hcTopics", next(iter(hc.values()))["H37a"]["n_topics"] if "H37a" in next(iter(hc.values())) else "--")
    lab = rev.get("hc19_labels")
    if lab:
        add("hcLabN", lab["n"]); add("hcLabCorrect", f"{100 * lab['correct_supports']:.0f}")
        add("hcLabMis", f"{100 * lab['misinfo_supports_wrong']:.0f}"); add("hcLabBoth", f"{100 * lab['both']:.0f}")
    j = rev.get("judge19", {})
    for fam, tag in (("H30a", "Thirty"), ("H23", "TwentyThree")):
        t = j.get(fam)
        if not t or not t.get("judge"):
            continue
        for sc, stag in (("judge", "J"), ("matcher", "M")):
            p = t[sc]
            add(f"jd{tag}{stag}", _pt(p["diff"], True)); add(f"jd{tag}{stag}CI", _ci_pt(p["ci_low"], p["ci_high"]))
            add(f"jd{tag}{stag}NinetyCI", _ci_pt(p["ci90_low"], p["ci90_high"]))
            add(f"jd{tag}{stag}Eq", _yn(p["equivalent"])); add(f"jd{tag}{stag}Margin", _pt(p["margin"]))
            add(f"jd{tag}{stag}K", p["n_cells"])
        d = t["matcher_minus_judge"]
        add(f"jd{tag}Shift", _pt(d["diff"], True)); add(f"jd{tag}ShiftCI", _ci_pt(d["ci_low"], d["ci_high"]))
    ag = j.get("agreement", {})
    if ag:
        n = sum(v["n"] for v in ag.values())
        add("jdN", f"{n:,}"); add("jdAgree", f"{100 * sum(v['agree'] * v['n'] for v in ag.values()) / n:.1f}")


def tab_fresh10(R, out: Path):
    f = R["rev"].get("fresh19")
    if not f or "cells" not in f:
        return
    rows = []
    for name, c in sorted(f["cells"].items(), key=lambda kv: (kv[1]["gen"], kv[1]["det"])):
        cl = c.get("clean", {})
        rows.append(f"{NICE10[c['gen']]} & {NICE10[c['det']]} & {100 * c['right']:.0f} & {100 * c['wrong']:.0f} & "
                    f"{_pt(c['plain'])} & {_pt(c['pred'], True)} & {_pt(c['obs'], True)} & "
                    f"{_pt(cl.get('pred'), True)} & {_pt(cl.get('obs'), True)} \\\\")
    (out / "tab_fresh.tex").write_text(r"""\begin{table*}[t]
\centering\small
\setlength{\tabcolsep}{5pt}
\begin{tabular}{@{}llrrrrrrr@{}}
\toprule
& & \multicolumn{5}{c}{Conflict questions} & \multicolumn{2}{c}{Clean} \\
\cmidrule(lr){3-7}\cmidrule(lr){8-9}
Generator & Detector & Right & Wrong & Acc. & Pred. & Obs. & Pred. & Obs. \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\caption{Prospective test on ConflictBank-fresh (\S19.1; 300 conflict and 100 clean questions no earlier run used).
\emph{Right}/\emph{Wrong}: \% of questions on which the detector names exactly the false side of a gold pair / marks a
correct passage. \emph{Acc.}: no-flag accuracy. \emph{Pred.}: the frozen prediction, committed before any flagged
answer existed; \emph{Obs.}: observed directional-flag effect, in points.}
\label{tab:fresh}
\end{table*}
""")


def tab_hc10(R, out: Path):
    hc = R["rev"].get("hc19", {})
    if not hc:
        return
    gname = {"mini": "gpt-4o-mini", "luna": "gpt-6-luna", "haiku": "Claude Haiku"}
    rows = []
    for g, v in hc.items():
        cell = lambda k: f"{_pt(v[k]['diff'], True)} {_ci_pt(v[k]['ci_low'], v[k]['ci_high'])}" if k in v else "--"
        pred = _pt(v["llm"]["pred"], True) if "llm" in v and "pred" in v["llm"] else "--"
        rows.append(f"{gname[g]} & {_pt(v['plain'])} & {_pt(v.get('closedbook'))} & {cell('H37b')} & {cell('H37a')} & "
                    f"{cell('llm')} & {pred} \\\\")
    (out / "tab_hc.tex").write_text(r"""\begin{table*}[t]
\centering\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lrrllll@{}}
\toprule
& & & \multicolumn{3}{c}{Effect against no flag (points)} & \\
\cmidrule(lr){4-6}
Generator & No flag & Closed-book & Oracle neutral & Oracle directional & LLM directional & Pred. \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\caption{HealthContradict (\S19.3): yes/no health questions with one real web page per side, accuracy in \% and
effects in points with topic-cluster bootstrap 95\% CIs. \emph{Pred.}: the frozen follow-rate prediction for the LLM
directional flag (gpt-4o-mini detector), committed before any flagged answer existed.}
\label{tab:hc}
\end{table*}
""")


def write10(R, out: Path):
    tab_fresh10(R, out)
    tab_hc10(R, out)
