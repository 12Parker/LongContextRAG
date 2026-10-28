"""
Paper macros and tables for the fifth revision (PREREGISTRATION.md §15; computed in revision5.py).
Appends to numbers.tex, so it runs after paper_assets.numbers.
"""
from __future__ import annotations

from pathlib import Path

from .paper_assets import PRIMARY, ROBUST, _nan, _p, cell, pct

GEN_A = {"gpt-4o-mini": "Mini", "gpt-4.1": "Four", "Qwen2.5-7B": "Qwen", "gpt-6-luna": "Luna", "gpt-6.1-terra": "Terra",
         "claude-haiku-5-5": "Haiku", "claude-sonnet-5-5": "Sonnet"}
PRE_A = {"CBs": "Sg", "RD": "Rd", "CBmaj": "Mj", "CBf": "Fc", "RG": "Rg", "CB": "Cb", "CBn": "Mn"}
NICE = {"gpt-4o-mini": "gpt-4o-mini", "gpt-4.1": "gpt-4.1", "Qwen2.5-7B": "Qwen2.5-7B", "gpt-6-luna": "gpt-6-luna",
        "gpt-6.1-terra": "gpt-6.1-terra", "claude-haiku-5-5": "Claude Haiku", "claude-sonnet-5-5": "Claude Sonnet"}


def _ci(t, lo="ci_low", hi="ci_high"):
    return f"[{pct(t[lo], True)}, {pct(t[hi], True)}]"


def _f2(x):
    return "--" if _nan(x) else (f"$-${abs(x):.2f}" if x < 0 else f"{x:.2f}")


def _abbr(key):
    pre, gen = key.split(":", 1)
    return PRE_A.get(pre, pre) + GEN_A.get(gen.split("|")[0], gen)


def numbers5(R, add):
    rev = R["rev"]
    # registered real-detector tests (gpt-4o-mini LLM detector, directional flag), counted by outcome
    groups = {"Sg": (("H15", "H19", "H25"), "ConflictBank", "LLM dir. flag > plain (all q.)"),
              "Mj": (("H17", "H19", "H25"), "CB majority", "LLM dir. flag vs plain (conflict)"),
              "Rg": (("H26",), "RAGuard", "LLM dir. flag vs plain (conflict)")}
    for nm, (fams, pre, what) in groups.items():
        ts = [v for f in fams for k, v in rev.get(f, {}).get(PRIMARY, {}).items() if k.startswith(pre + ",") and k.endswith(what)]
        add(f"nReal{nm}Tests", len(ts))
        add(f"nReal{nm}Pos", sum(v["p_holm"] < 0.05 and v["diff"] > 0 for v in ts))
        add(f"nReal{nm}Neg", sum(v["p_holm"] < 0.05 and v["diff"] < 0 for v in ts))
    red = rev.get("H24_redo")
    if red:
        rg = red["regression"]
        add("hRedoAddD", pct(rg["add_D"]["est"], True)); add("hRedoAddDCI", _ci(rg["add_D"], "ci_low", "ci_high"))
        add("hRedoAddG", pct(rg["add_G"]["est"], True)); add("hRedoAddGCI", _ci(rg["add_G"]))
        add("hRedoFeD", _f2(rg["fe_D"]["est"])); add("hRedoFeDCI", f"[{_f2(rg['fe_D']['ci_low'])}, {_f2(rg['fe_D']['ci_high'])}]")
        add("hRedoAddDCoef", _f2(rg["add_D"]["est"])); add("hRedoAddGCoef", _f2(rg["add_G"]["est"]))
        add("hRedoPermP", f"{red['perm_fe_D']['p_one_sided']:.2f}"); add("hRedoNPerm", red["perm_fe_D"]["n_perm"])
        add("hRedoCalSlope", _f2(rg["cal_slope"]["est"]))
        add("hRedoCalSlopeCI", f"[{_f2(rg['cal_slope']['ci_low'])}, {_f2(rg['cal_slope']['ci_high'])}]")
        mx = (red.get("mixed") or {}).get("fe_det_gen_re_q", {})
        for k, nm in (("d", "D"), ("w", "W"), ("g", "G")):
            if k in mx:
                add(f"hRedoMix{nm}", pct(mx[k]["est"], True)); add(f"hRedoMix{nm}CI", _ci(mx[k]))
        cr = (red.get("mixed") or {}).get("crossed_re", {})
        for k, nm in (("d", "D"), ("w", "W")):
            if k in cr:
                add(f"hRedoCross{nm}", pct(cr[k]["est"], True))
        cv = red["lodo"]
        for k, nm in (("G only", "G"), ("D + G", "Add"), ("gap D - G (registered)", "Gap"), ("mechanism score", "Mech")):
            add(f"hRedoCv{nm}", _f2(cv[k]["r2_cv"]))
        if "post_registration" in red:
            pr = red["post_registration"]
            add("hRedoPostRho", f"{pr['rho']:+.2f}"); add("hRedoPostP", _p(pr["p_one_sided"])); add("hRedoPostCells", pr["n_cells"])
        comp = red["compliance"]
        fr = [v["follow_right"] for v in comp.values() if v["follow_right"] is not None]
        fw = [v["follow_wrong"] for v in comp.values() if v["follow_wrong"] is not None]
        add("cmpRightMin", pct(min(fr), nd=0)); add("cmpRightMax", pct(max(fr), nd=0))
        add("cmpWrongMin", pct(min(fw), nd=0)); add("cmpWrongMax", pct(max(fw), nd=0))
        for g, v in comp.items():
            add(f"cmpRight{GEN_A[g]}", pct(v["follow_right"], nd=0)); add(f"cmpWrong{GEN_A[g]}", pct(v["follow_wrong"], nd=0))
        dd = red["detection_differences"]
        add("detDiffMax", max(v["n_differ"] for v in dd.values()))
        for d, v in dd.items():
            add(f"detDiff{GEN_A[d]}", v["n_differ"])
    np5 = rev.get("neutral_pooled5")
    if np5:
        add("neuBootCI", f"[{pct(np5['ci_cluster_boot'][0], True)}, {pct(np5['ci_cluster_boot'][1], True)}]")
        add("neuPI", f"[{pct(np5['pred_interval'][0], True)}, {pct(np5['pred_interval'][1], True)}]")
        add("neuPIlow", pct(np5["pred_interval"][0], True)); add("neuPIhigh", pct(np5["pred_interval"][1], True))
        for m, nm in (("intercept (OpenAI, unknown allowed, synthetic)", "Base"), ("Anthropic", "Anthropic"), ("Qwen", "Qwen"),
                      ("forced (no unknown)", "Forced"), ("natural data", "Natural")):
            t = np5["moderators"][m]
            add(f"neuMod{nm}", pct(t["est"], True)); add(f"neuMod{nm}CI", _ci(t))
    t5 = rev.get("tost5", {})
    for pre, nm in (("H23", "TwentyThree"), ("H20 neutral", "TwentyNeu"), ("H20 one-wrong", "TwentyOw")):
        sel = [v for k, v in t5.items() if k.split(": ", 1)[1].startswith(pre)]
        add(f"nTost{nm}Eq", sum(v["equivalent"] for v in sel)); add(f"nTost{nm}Tests", len(sel))
    k23 = [v["diff"] for k, v in t5.items() if k.split(": ", 1)[1].startswith("H23")]
    if k23:
        add("keyDiffMin", pct(min(k23), True)); add("keyDiffMax", pct(max(k23), True))
    for k, v in t5.items():
        st, what = k.split(": ", 1)
        if what.startswith("H23"):
            add(f"tostKey{_label_abbr(st)}", f"{_ci(v, 'ci90_low', 'ci90_high')}")
    sr = rev.get("H21c_ratio", {})
    neu = [v for k, v in sr.items() if "neutral" in k]
    if neu:
        add("selNeuMin", pct(min(v["diff"] for v in neu), True)); add("selNeuMax", pct(max(v["diff"] for v in neu), True))
        add("nSelNeuSig", sum(v["supported"] for v in neu)); add("nSelNeuTests", len(neu))
        add("nSelNeuEq", sum(v["tost"]["equivalent"] for v in neu))
        dr = [v for k, v in sr.items() if "dir." in k]
        add("selDirMin", pct(min(v["diff"] for v in dr), True)); add("selDirMax", pct(max(v["diff"] for v in dr), True))
    for k, v in sr.items():
        st, what = k.split(": ", 1)
        nm = f"selR{_label_abbr(st)}{'Flag' if what.startswith('neutral') else 'Dir'}"
        add(nm, pct(v["diff"], True)); add(nm + "CI", _ci(v)); add(nm + "P", _p(v["p_holm"]))
        add(nm + "A", pct(v["sel_a"])); add(nm + "B", pct(v["sel_b"]))
    ut = rev.get("utility", {})
    for c in (0, 1, 2, 4):
        neu_ = [d["neutral"][f"c{c}"] for d in ut.values() if "neutral" in d]
        add(f"nUtilNeuPos{_w(c)}", sum(t["sig"] and t["diff"] > 0 for t in neu_))
        add(f"nUtilNeuNeg{_w(c)}", sum(t["sig"] and t["diff"] < 0 for t in neu_))
        add(f"nUtilNeuAbove{_w(c)}", sum(t["diff"] > 0 for t in neu_))
    add("nUtilSettings", sum("neutral" in d for d in ut.values()))
    for key, d in ut.items():
        for nm, cs in d.items():
            for c, t in cs.items():
                add(f"ut{_abbr(key)}{nm.replace('-', '').capitalize()}{_w(int(c[1:]))}", pct(t["diff"], True))
    tr = rev.get("transitions", {})
    for key, d in tr.items():
        for nm, T in d.items():
            for a, row in T.items():
                n = sum(row.values())
                for b, v in row.items():
                    add(f"tr{_abbr(key)}{nm.replace('-', '').capitalize()}{a.capitalize()}{b.capitalize()}",
                        pct(v / n, nd=0) if n else "--")
                add(f"tr{_abbr(key)}{nm.replace('-', '').capitalize()}{a.capitalize()}N", n)
    for key, v in rev.get("rd_plain_mix", {}).items():
        a = _abbr(key)
        add(f"mix{a}Misinfo", pct(v["misinfo"], nd=0)); add(f"mix{a}Recall", pct(v["gold_recall"], nd=0))
        add(f"mix{a}NAnswers", f"{v['n_answers']:.1f}")
    pf = rev.get("parse_failures", {})
    add("parseFailTotal", sum(v["failed"] for v in pf.values())); add("parseFailN", sum(v["n"] for v in pf.values()))
    # H27
    for f_, nm in (("H27a", "A"), ("H27b", "B"), ("H27c", "C")):
        fam = rev.get(f_, {}).get(PRIMARY, {})
        if not fam:
            continue
        add(f"nHTwentySeven{nm}Tests", len(fam))
        add(f"nHTwentySeven{nm}Pos", sum(v["supported"] and v["diff"] > 0 for v in fam.values()))
        add(f"nHTwentySeven{nm}Neg", sum(v["supported"] and v["diff"] < 0 for v in fam.values()))
        add(f"nHTwentySeven{nm}Above", sum(v["diff"] > 0 for v in fam.values()))
        ds = [v["diff"] for v in fam.values()]
        add(f"hTwentySeven{nm}Min", pct(min(ds), True)); add(f"hTwentySeven{nm}Max", pct(max(ds), True))
        for k, v in fam.items():
            g, rest = k.split(", detector ")
            d = rest.split(":")[0]
            add(f"hTwentySeven{nm}{GEN_A[g]}{GEN_A[d]}", pct(v["diff"], True))
            add(f"hTwentySeven{nm}{GEN_A[g]}{GEN_A[d]}P", _p(v["p_holm"]))
    for key, fam in rev.get("H28", {}).items():
        pre, g = key.split(":")
        T = ("Wc" if pre == "WC" else "Amb") + GEN_A[g]
        for m_ in ("both_answers", PRIMARY, "gold_recall"):
            for k, v in fam.get(m_, {}).items():
                nm = T + ("Dir" if "dir." in k else "Rem") + ("Rec" if m_ == "gold_recall" else "")
                add(f"hTwentyEight{nm}", pct(v["diff"], True)); add(f"hTwentyEight{nm}P", _p(v["p_holm"]))
                add(f"hTwentyEight{nm}CI", _ci(v))
        if "plain" in fam:
            add(f"hTwentyEight{T}Plain", pct(fam["plain"] if pre == "WC" else fam["plain"][PRIMARY]))
        if "n" in fam:
            add(f"hTwentyEight{T}N", fam["n"])
        if "oracle_flag" in fam:
            add(f"hTwentyEight{T}OracleFlag", pct(fam["oracle_flag"]["diff"], True))
    h29 = rev.get("H29", {})
    for k, v in h29.get("a", {}).items():
        nm = ("Mini" if k.startswith("gpt-4o-mini") else "Luna") + ("Rev" if "reversed" in k else "Para")
        add(f"hTwentyNine{nm}", pct(v["diff"], True)); add(f"hTwentyNine{nm}CI", _ci(v)); add(f"hTwentyNine{nm}P", _p(v["p_holm"]))
        add(f"hTwentyNine{nm}Orig", pct(v["mean_a"])); add(f"hTwentyNine{nm}Probe", pct(v["mean_b"]))
        tb = h29["b_tost"][k]
        add(f"hTwentyNine{nm}Eq", "yes" if tb["equivalent"] else "no")
        add(f"hTwentyNine{nm}TostCI", _ci(tb, "ci90_low", "ci90_high"))
        ds = h29["desc"][k]
        add(f"hTwentyNine{nm}WrongOrig", pct(ds["wrong_side_orig"])); add(f"hTwentyNine{nm}WrongProbe", pct(ds["wrong_side_probe"]))
    for k, v in h29.get("c", {}).items():
        nm = ("Mini" if k.startswith("gpt-4o-mini") else "Luna") + ("Rev" if "reversed" in k else "Para")
        add(f"hTwentyNineC{nm}", pct(v["diff"], True)); add(f"hTwentyNineC{nm}P", _p(v["p_holm"]))
        add(f"hTwentyNineC{nm}Gain", pct(v["gain_probe"], True)); add(f"hTwentyNineC{nm}GainOrig", pct(v["gain_orig"], True))
    for k, v in h29.get("para_subset", {}).items():
        nm = ("Mini" if k.startswith("gpt-4o-mini") else "Luna")
        add(f"hTwentyNineSub{nm}", pct(v["diff"], True)); add(f"hTwentyNineSub{nm}N", v["n"])
    if h29.get("para_check"):
        pc = h29["para_check"]
        add("paraFail", pc["passages"]); add("paraItemsPass", pc["items_all_pass"])


def _w(c):
    return {0: "Zero", 1: "One", 2: "Two", 4: "Four"}[c]


def _label_abbr(label):
    LAB_A = {"ConflictBank": "Sg", "RAMDocs": "Rd", "CB majority": "Mj", "ConflictBank forced": "Fc", "RAGuard": "Rg"}
    st, gen = label.split(", ", 1)
    return LAB_A[st] + GEN_A[gen]


# --------------------------------------------------------------------------- tables
def tab_cells5(R, out: Path):
    """tab_cells with the H27 hint and detector-alone columns (overrides paper_assets' version)."""
    rev = R["rev"]
    cells = rev.get("H24", {}).get("cells", {})
    if not cells:
        return
    fa = rev.get("H27a", {}).get(PRIMARY, {}); fc = rev.get("H27c", {}).get(PRIMARY, {})
    def find(fam, g, d):
        return next((v for k, v in fam.items() if k.startswith(f"{g}, detector {d}:")), None)
    rows = []
    for k, v in sorted(cells.items(), key=lambda kv: kv[1]["mech"]):
        g, d = k.split("|")
        a, c = find(fa, g, d), find(fc, g, d)
        flag = v["effect"]["mean_b"]
        rows.append(f"{NICE[d]} & {NICE[g]} & {pct(v['det_acc'])} & {pct(v['wrong_dir'])} & {pct(v['gen_acc'])} & "
                    f"{pct(v['mech'], True)} & {cell(v['effect'])} & {pct(flag)} & "
                    f"{pct(a['mean_a']) if a else '--'} & {pct(c['mean_a']) if c else '--'} & "
                    f"{cell(a) if a else '--'} & {cell(c) if c else '--'} \\\\")
    (out / "tab_cells.tex").write_text(r"""\begin{table*}[t]
\centering\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{llrrrrrrrrrr}
\toprule
& & \multicolumn{2}{c}{Detector} & & & & \multicolumn{3}{c}{Accuracy with} & \multicolumn{2}{c}{Flag minus} \\
\cmidrule(lr){3-4}\cmidrule(lr){8-10}\cmidrule(lr){11-12}
Detector & Generator & Right & Wrong & Gen.\ acc. & Score & Gain & flag & hint & det.\ alone & hint & det.\ alone \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\caption{Every detector--generator cell of H24 and H27 (ConflictBank, one-answer prompt, 300 conflict questions; \%),
sorted by the exploratory score. \emph{Right}: share of questions on which the detector flags the true contradiction
and names exactly its misinformation passage (the one definition of direction accuracy used throughout, computed on the
detections that cell's generator saw); \emph{Wrong}: share on which it marks a correct passage as suspect;
\emph{Gen.\ acc.}: the generator's accuracy with no flags; \emph{Score}: right direction where the generator errs minus a
correct passage marked where it is right; \emph{Gain}: the LLM directional flag's change in points. \emph{Hint}: the
generator given the detector model's own plain answer, with no passage marked (H27); \emph{det.\ alone}: the detector
model's own plain accuracy. A detector's statistics can differ slightly between generators because concurrent runs
made independent detector calls (up to \detDiffMax\ of 400 questions differ for one detector). Bold: significant under
the paper's scheme (Gain: exploratory family; hint and det.\ alone columns: H27, Holm across cells).}
\label{tab:cells}
\end{table*}
""")


def tab_h24redo(R, out: Path):
    red = R["rev"].get("H24_redo")
    if not red:
        return
    rg, cv, mx = red["regression"], red["lodo"], (red.get("mixed") or {}).get("fe_det_gen_re_q", {})
    f2ci = lambda t: f"{_f2(t['est'])} [{_f2(t['ci_low'])}, {_f2(t['ci_high'])}]"
    comp = red["compliance"]
    crow = "\n".join(f"{NICE[g]} & {pct(v['follow_right'], nd=0)} ({v['n_right']}) & {pct(v['follow_wrong'], nd=0)} ({v['n_wrong']}) \\\\"
                     for g, v in comp.items())
    (out / "tab_h24redo.tex").write_text(r"""\begin{table}[t]
\centering\footnotesize
\setlength{\tabcolsep}{3pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lr}
\toprule
\multicolumn{2}{l}{\emph{Cell level (""" + str(len(red["cells"])) + r""" cells): accuracy with the flag, $A$}} \\
$A \sim D + G$: coefficient of $D$ & """ + f2ci(rg["add_D"]) + r""" \\
\quad coefficient of $G$ & """ + f2ci(rg["add_G"]) + r""" \\
$A \sim D$ + generator FE: coefficient of $D$ & """ + f2ci(rg["fe_D"]) + r""" \\
\quad exact permutation of $D$ across detectors & $p=""" + f"{red['perm_fe_D']['p_one_sided']:.2f}" + r"""$ \\
\midrule
\multicolumn{2}{l}{\emph{Leave one detector out: $R^2$ for the gain}} \\
""" + "\n".join(f"{k.replace('gap D - G (registered)', 'gap $D-G$ (registered)').replace('D + G', '$D+G$').replace('G only', '$G$ only')} & {_f2(v['r2_cv'])} \\\\"
                for k, v in cv.items() if k != "mechanism score, no fit") + r"""
\midrule
\multicolumn{2}{l}{\emph{Question level (""" + str(red["mixed"]["n_rows"]) + r""" cell-questions), points}} \\
detector right on $q$ & """ + (f"{pct(mx['d']['est'], True)} {_ci(mx['d'])}" if mx else "--") + r""" \\
detector marks a correct passage & """ + (f"{pct(mx['w']['est'], True)} {_ci(mx['w'])}" if mx else "--") + r""" \\
generator right on its own & """ + (f"{pct(mx['g']['est'], True)} {_ci(mx['g'])}" if mx else "--") + r""" \\
\midrule
\multicolumn{2}{l}{\emph{Calibration: gain $\sim$ score}} \\
slope & """ + f2ci(rg["cal_slope"]) + r""" \\
\bottomrule
\end{tabular}}

\vspace{4pt}
\begin{tabular}{lrr}
\toprule
Generator & Follows a needed & Follows a wrong- \\
& right flag & side flag \\
\midrule
""" + crow + r"""
\bottomrule
\end{tabular}
\caption{H24 without the shared $-G$ term (\S15.4, re-analysis of existing cells). $D$: the detector's direction
accuracy; $G$: the generator's no-flag accuracy. CIs: 2,000 question resamples, shared across cells. The question-level
model is a linear probability model with detector and generator fixed effects and a random intercept per question.
Lower table: share of questions the generator gets right when the detector is right and the generator alone is wrong
in every seed, and share it gets wrong when the detector marks a correct passage and the generator alone is right in
every seed (count of cell-questions in brackets).}
\label{tab:h24redo}
\end{table}
""")


def tab_pooled(R, out: Path):
    np5 = R["rev"].get("neutral_pooled5")
    if not np5:
        return
    lab = lambda n: (n.replace("CBs:", "CB single, ").replace("CBf:", "CB forced, ").replace("CBmaj:", "CB majority, ")
                     .replace("CBn:", "CB minimal, ").replace("CB:", "CB multi, ").replace("RD:", "RAMDocs, ").replace("RG:", "RAGuard, ")
                     .replace("|default", " (default)").replace("|minimal", " (minimal)").replace("claude-haiku-5-5", "Claude Haiku")
                     .replace("claude-sonnet-5-5", "Claude Sonnet"))
    ss = sorted(np5["settings"], key=lambda s: s["diff"])
    half = (len(ss) + 1) // 2
    def row(s):
        return f"{lab(s['name'])} & {pct(s['diff'], True)}"
    body = "\n".join(f"{row(a)} & {row(b) if b else ' & '} \\\\" for a, b in
                     zip(ss[:half], ss[half:] + [None] * (2 * half - len(ss))))
    mods = "\n".join(f"\\multicolumn{{2}}{{l}}{{{k}}} & \\multicolumn{{2}}{{l}}{{{pct(v['est'], True)} {_ci(v)}}} \\\\"
                     for k, v in np5["moderators"].items())
    (out / "tab_pooled.tex").write_text(r"""\begin{table*}[t]
\centering\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrlr}
\toprule
Setting & Neutral flag & Setting & Neutral flag \\
\midrule
""" + body + r"""
\midrule
\multicolumn{4}{l}{\emph{Meta-regression (points; CIs from the question-cluster bootstrap)}} \\
""" + mods + r"""
\bottomrule
\end{tabular}
\caption{The """ + str(np5["k"]) + r""" settings pooled for the oracle neutral flag (conflict questions, change in points).
The pooled estimate was registered as exploratory in \S13.4. Pooled: \neuPooled\ (DerSimonian--Laird); 95\% CI from a
bootstrap that resamples questions within each benchmark jointly across the settings that share them \neuBootCI;
prediction interval for a new setting \neuPI. Moderators (\S15.4): model family (reference OpenAI), no ``unknown'' allowed
(forced prompt) and natural (RAMDocs, RAGuard) rather than generated data (ConflictBank).}
\label{tab:pooled}
\end{table*}
""")


def tab_utility(R, out: Path):
    rev = R["rev"]
    ut, tr = rev.get("utility", {}), rev.get("transitions", {})
    keys = [k for k in ("CBs:gpt-4o-mini", "CBs:gpt-6-luna", "CBs:gpt-6.1-terra", "CBs:claude-haiku-5-5", "CBs:claude-sonnet-5-5",
                        "CBf:gpt-4o-mini", "CBmaj:gpt-4o-mini", "RG:gpt-4o-mini", "RG:gpt-6-luna", "RG:claude-haiku-5-5",
                        "RD:gpt-6-luna", "RD:claude-haiku-5-5") if k in ut and "neutral" in ut[k]]
    lab = lambda k: (k.replace("CBs:", "CB single, ").replace("CBf:", "CB forced, ").replace("CBmaj:", "CB majority, ")
                     .replace("RD:", "RAMDocs, ").replace("RG:", "RAGuard, ").replace("claude-haiku-5-5", "Claude Haiku")
                     .replace("claude-sonnet-5-5", "Claude Sonnet"))
    rows = []
    for k in keys:
        n = ut[k]["neutral"]; d = ut[k].get("directional", {})
        T = tr.get(k, {}).get("neutral", {})
        mis = T.get("misinfo", {}); nm = sum(mis.values()) or 1
        cor = T.get("correct", {}); nc = sum(cor.values()) or 1
        rows.append(f"{lab(k)} & " + " & ".join(cell(n[f'c{c}']) for c in (0, 1, 2, 4)) + " & " +
                    (cell(d['c2']) if d else "--") + f" & {pct(mis.get('abstain', 0) / nm, nd=0)} & {pct(mis.get('correct', 0) / nm, nd=0)}"
                    f" & {pct(cor.get('abstain', 0) / nc, nd=0)} \\\\")
    (out / "tab_utility.tex").write_text(r"""\begin{table*}[t]
\centering\footnotesize
\setlength{\tabcolsep}{4pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrrrrrr}
\toprule
& \multicolumn{4}{c}{Neutral flag, utility change, cost $c$ of misinformation} & Dir.\ flag & \multicolumn{3}{c}{Neutral flag: plain answer $\to$ flagged answer} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-6}\cmidrule(lr){7-9}
Setting & $c=0$ & $c=1$ & $c=2$ & $c=4$ & $c=2$ & misinfo$\to$abstain & misinfo$\to$correct & correct$\to$abstain \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\caption{Abstention-aware view of the oracle flags (conflict questions; \S15.4, re-analysis). Utility per answer: $+1$
correct, $-c$ if it contains the misinformation, 0 for abstaining or any other wrong answer; $c=0$ is strict accuracy.
Bold: significant in one exploratory Holm family per $c$ over all """ + r"""\nUtilSettings\ settings. Right: of the
question--seed pairs whose plain answer contained the misinformation (or was correct), the share whose flagged answer
abstained or was correct.}
\label{tab:utility}
\end{table*}
""")


def fig_reliability5(R, out: Path):
    """Left: cell gain against the exploratory score (H24 re-analysis). Right: accuracy with the detector's directional
    flag against accuracy with the detector model's answer as a hint (H27). Replaces paper_assets.fig_reliability."""
    import matplotlib.pyplot as plt
    from .paper_assets import INK2, NEUTRAL, _style
    rev = R["rev"]
    cells = rev.get("H24", {}).get("cells", {})
    fa = rev.get("H27a", {}).get(PRIMARY, {})
    if len(cells) < 3:
        return
    det_col = {"gpt-4o-mini": "#2a78d6", "gpt-4.1": "#eb6834", "gpt-6-luna": "#1baf7a", "gpt-6.1-terra": "#eda100",
               "claude-haiku-5-5": "#e87ba4", "claude-sonnet-5-5": "#008300"}
    gen_mark = {"gpt-4o-mini": "o", "gpt-6-luna": "s", "gpt-4.1": "^", "gpt-6.1-terra": "D", "claude-haiku-5-5": "P",
                "Qwen2.5-7B": "v"}
    short = {"gpt-4o-mini": "4o-mini", "gpt-4.1": "4.1", "gpt-6-luna": "6-luna", "gpt-6.1-terra": "6.1-terra",
             "claude-haiku-5-5": "Haiku 5.5", "claude-sonnet-5-5": "Sonnet 5.5", "Qwen2.5-7B": "Qwen-7B"}
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(3.3, 2.2))
    ax = axes[0]
    for key, v in cells.items():
        g, d = key.split("|"); e = v["effect"]
        ax.errorbar(100 * v["mech"], 100 * e["diff"], yerr=[[100 * (e["diff"] - e["ci_low"])], [100 * (e["ci_high"] - e["diff"])]],
                    fmt="none", ecolor=det_col.get(d, NEUTRAL), elinewidth=0.5, alpha=0.45, zorder=2)
        ax.scatter(100 * v["mech"], 100 * e["diff"], s=16, marker=gen_mark.get(g, "o"), color=det_col.get(d, NEUTRAL),
                   edgecolor="white", linewidth=0.7, zorder=3)
    lim = [min(-20, ax.get_xlim()[0]), max(30, ax.get_xlim()[1])]
    ax.plot(lim, lim, ls="--", color=INK2, lw=0.6, zorder=1); ax.axhline(0, color=INK2, lw=0.4)
    ax.set_xlabel("score (points)", fontsize=6); ax.set_ylabel("gain from LLM dir. flag (points)", fontsize=6)
    ax.set_title("gain vs score (exploratory)", fontsize=6.5, color=INK2, pad=2)
    ax = axes[1]
    for key, v in cells.items():
        g, d = key.split("|")
        t = next((x for k, x in fa.items() if k.startswith(f"{g}, detector {d}:")), None)
        if not t:
            continue
        ax.scatter(100 * t["mean_a"], 100 * t["mean_b"], s=16, marker=gen_mark.get(g, "o"), color=det_col.get(d, NEUTRAL),
                   edgecolor="white", linewidth=0.7, zorder=3)
    if fa:
        lo_, hi_ = ax.get_xlim(); lo2, hi2 = ax.get_ylim(); lim = [min(lo_, lo2), max(hi_, hi2)]
        ax.plot(lim, lim, ls="--", color=INK2, lw=0.6, zorder=1)
    ax.set_xlabel("accuracy with hint (%)", fontsize=6); ax.set_ylabel("accuracy with dir. flag (%)", fontsize=6)
    ax.set_title("flag vs hint (H27)", fontsize=6.5, color=INK2, pad=2)
    for a in axes:
        a.tick_params(labelsize=6)
    hd = [plt.Line2D([], [], marker="o", ls="", ms=4, color=c, label=short[k])
          for k, c in det_col.items() if any(key.endswith("|" + k) for key in cells)]
    hg = [plt.Line2D([], [], marker=m, ls="", ms=4, color=INK2, label=short[k])
          for k, m in gen_mark.items() if any(key.startswith(k + "|") for key in cells)]
    fig.legend(handles=hd, title="detector (colour)", fontsize=5.5, title_fontsize=6, ncol=3, frameon=False,
               loc="upper left", bbox_to_anchor=(0.0, 0.0), handletextpad=0.1, columnspacing=0.6)
    fig.legend(handles=hg, title="generator (shape)", fontsize=5.5, title_fontsize=6, ncol=3, frameon=False,
               loc="upper right", bbox_to_anchor=(1.0, 0.0), handletextpad=0.1, columnspacing=0.6)
    fig.tight_layout(w_pad=0.8)
    fig.savefig(out / "fig_reliability.pdf", bbox_inches="tight")
    plt.close(fig)


def write5(R, out: Path):
    fig_reliability5(R, out)
    tab_cells5(R, out); tab_h24redo(R, out); tab_pooled(R, out); tab_utility(R, out)
