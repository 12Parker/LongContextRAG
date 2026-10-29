"""
Paper macros and tables for the sixth revision (PREREGISTRATION.md §16; computed in revision6.py).
numbers6 runs inside paper_assets.numbers (before the unsigned \\a variants are made); write6 runs after write5 and
replaces tab_pooled, tab_utility and tab_h24redo, and adds the verdict-key column to tab_cells.
"""
from __future__ import annotations

import math
from pathlib import Path

from .assets5 import GEN_A, NICE, _ci, _f2
from .paper_assets import PRIMARY, _nan, _p, cell, pct

W = {0: "Zero", 1: "One", 2: "Two", 4: "Four"}
POL = {"multi": "Multi", "minimal": "Minimal", "single": "Single", "forced": "Forced"}


def _lab(n):
    return (n.replace("CBs:", "CB one-answer, ").replace("CBf:", "CB forced, ").replace("CBmaj:", "CB majority, ")
            .replace("CBn:", "CB minimal, ").replace("CB:", "CB multi-answer, ").replace("RD:", "RAMDocs, ")
            .replace("RG:", "RAGuard, ").replace("|default", " (multi-answer)").replace("|minimal", " (minimal)")
            .replace("claude-haiku-5-5", "Claude Haiku").replace("claude-sonnet-5-5", "Claude Sonnet"))


def _abbr(key):
    PRE = {"CBs": "Sg", "RD": "Rd", "CBmaj": "Mj", "CBf": "Fc", "RG": "Rg", "CB": "Cb", "CBn": "Mn"}
    pre, gen = key.split(":", 1)
    g, _, sk = gen.partition("|")
    return PRE[pre] + GEN_A[g] + ("Min" if sk == "minimal" else "")


def _mci(m):
    c = m.get("ci_model")
    return f"[{pct(c[0], True)}, {pct(c[1], True)}]" if c else "--"


def numbers6(R, add):
    rev = R["rev"]
    m = rev.get("neutral_pooled6")
    if m:
        add("neuMlMu", pct(m["mu"], True)); add("neuMlCI", f"[{pct(m['ci'][0], True)}, {pct(m['ci'][1], True)}]")
        add("neuMlPI", f"[{pct(m['pred_interval'][0], True)}, {pct(m['pred_interval'][1], True)}]")
        add("neuMlK", m["k"])
        for pol, t in m["by_policy"].items():
            add(f"neuPol{POL[pol]}", pct(t["mu"], True)); add(f"neuPol{POL[pol]}K", t["k"])
            add(f"neuPol{POL[pol]}CI", "--" if _nan(t["ci_low"]) else _ci(t))
        for k, nm in (("several answers allowed", "Several"), ("forced (no unknown)", "Forced"), ("Anthropic", "Anthropic"),
                      ("Qwen", "Qwen"), ("natural misinformation (RAGuard)", "Natural"),
                      ("intercept (OpenAI, one answer, generated)", "Base")):
            t = m["moderators"][k]
            add(f"neuMl{nm}", pct(t["est"], True)); add(f"neuMl{nm}CI", _ci(t))
    rc = rev.get("H24_recast")
    if rc:
        for nm in ("P", "R"):
            t = rc["P_R_fe"][nm]
            add(f"rec{nm}Coef", _f2(t["est"])); add(f"rec{nm}CoefCI", f"[{_f2(t['ci_low'])}, {_f2(t['ci_high'])}]")
            add(f"rec{nm}Perm", f"{rc[f'perm_{nm}']['p_one_sided']:.3f}")
        add("recCvPRG", _f2(rc["lodo_r2"]["P + R + G"])); add("recCvG", _f2(rc["lodo_r2"]["G only"]))
        pb = rc["perm_P"]["by_detector"]; rb = rc["perm_R"]["by_detector"]
        others = [v for k, v in pb.items() if k != "gpt-4o-mini"]
        add("precMini", pct(pb["gpt-4o-mini"])); add("precOtherMin", pct(min(others))); add("precOtherMax", pct(max(others)))
        for d, v in rb.items():
            add(f"recall{GEN_A[d]}", pct(v))
        for d, v in pb.items():
            add(f"prec{GEN_A[d]}Det", pct(v))
        for nm, k in (("D", "d"), ("W", "w")):
            t = rc["qfe"][k]
            add(f"qfe{nm}", pct(t["est"], True)); add(f"qfe{nm}CI", _ci(t))
            add(f"qfe{nm}Lodo", f"{pct(t['lodo_min'], True)} to {pct(t['lodo_max'], True)}")
        f = rc["follow"]
        fr = [v["follow_right"] for v in f.values() if v["follow_right"] is not None]
        fw = [v["follow_wrong"] for v in f.values() if v["follow_wrong"] is not None]
        nl = [x for v in f.values() for x in (v["null_right"], v["null_wrong"]) if x is not None]
        add("folRightMin", pct(min(fr), nd=0)); add("folRightMax", pct(max(fr), nd=0))
        add("folWrongMin", pct(min(fw), nd=0)); add("folWrongMax", pct(max(fw), nd=0))
        add("folNullMax", pct(max(nl), nd=0) if nl else "--")
    for f_, nm in (("H31a", "A"), ("H31b", "B"), ("H31c", "C")):
        fam = rev.get(f_, {})
        if not fam:
            continue
        ds = [v["diff"] for v in fam.values()]
        add(f"hThirtyOne{nm}Min", pct(min(ds), True)); add(f"hThirtyOne{nm}Max", pct(max(ds), True))
        add(f"nHThirtyOne{nm}Sig", sum(v["supported"] for v in fam.values())); add(f"nHThirtyOne{nm}Tests", len(fam))
        for k, v in fam.items():
            g = k.split(":")[0]
            add(f"hThirtyOne{nm}{GEN_A[g]}", pct(v["diff"], True)); add(f"hThirtyOne{nm}{GEN_A[g]}P", _p(v["p_holm"]))
            add(f"hThirtyOne{nm}{GEN_A[g]}CI", _ci(v))
    rf = rev.get("H31_follow", {})
    if rf:
        for g, v in rf.items():
            add(f"rfolRight{GEN_A[g]}", pct(v["follow_right"], nd=0)); add(f"rfolWrong{GEN_A[g]}", pct(v["follow_wrong"], nd=0))
        add("rfolRightMin", pct(min(v["follow_right"] for v in rf.values()), nd=0))
        add("rfolRightMax", pct(max(v["follow_right"] for v in rf.values()), nd=0))
        add("rfolWrongMin", pct(min(v["follow_wrong"] for v in rf.values()), nd=0))
        add("rfolWrongMax", pct(max(v["follow_wrong"] for v in rf.values()), nd=0))
        add("nRandRight", next(iter(rf.values()))["n_right_q"]); add("nRandWrong", next(iter(rf.values()))["n_wrong_q"])
    for f_, nm in (("H30a", "A"), ("H30b", "B")):
        fam = rev.get(f_, {}).get(PRIMARY, {})
        if fam:
            ds = [v["diff"] for v in fam.values()]
            add(f"nHThirty{nm}Tests", len(fam)); add(f"nHThirty{nm}Sig", sum(v["supported"] for v in fam.values()))
            add(f"nHThirty{nm}Pos", sum(v["supported"] and v["diff"] > 0 for v in fam.values()))
            add(f"hThirty{nm}Min", pct(min(ds), True)); add(f"hThirty{nm}Max", pct(max(ds), True))
    tt = rev.get("H30a_tost", {})
    if tt:
        add("nHThirtyTostEq", sum(v["equivalent"] for v in tt.values()))
    for k, v in rev.get("H14_matched", {}).items():
        nm = "Ast" if k.endswith("astute (all q.)") else "Tr"
        add(f"hFourteenMatched{nm}", pct(v["diff"], True)); add(f"hFourteenMatched{nm}P", _p(v["p"]))
    ut = rev.get("utility6", {})
    if ut:
        add("nUtilSixSettings", sum("neutral" in d for d in ut.values()))
        for c in (0, 1, 2, 4):
            neu = [d["neutral"][f"c{c}"] for d in ut.values() if "neutral" in d]
            add(f"nUtilSixPos{W[c]}", sum(t["sig"] and t["diff"] > 0 for t in neu))
            add(f"nUtilSixNeg{W[c]}", sum(t["sig"] and t["diff"] < 0 for t in neu))
        for key, d in ut.items():
            for nm, cs in d.items():
                for c, t in cs.items():
                    add(f"us{_abbr(key)}{nm.capitalize()}{W[int(c[1:])]}", pct(t["diff"], True))
    for key, T in rev.get("transitions_null", {}).items():
        for a, row in T.items():
            n = sum(row.values())
            for b, v in row.items():
                add(f"trNull{_abbr(key)}{a.capitalize()}{b.capitalize()}", pct(v / n, nd=0) if n else "--")


def tab_pooled6(R, out: Path):
    m = R["rev"].get("neutral_pooled6")
    if not m:
        return
    ss = sorted(m["settings"], key=lambda s: (s["policy"], s["diff"]))
    half = (len(ss) + 1) // 2
    PS = {"multi": "multi", "minimal": "minimal", "single": "one", "forced": "forced"}
    row = lambda s: f"{PS[s['policy']]} & {_lab(s['name'])} & {pct(s['diff'], True)}" if s else " & & "
    body = "\n".join(f"{row(a)} & {row(b)} \\\\" for a, b in zip(ss[:half], ss[half:] + [None] * (2 * half - len(ss))))
    pol = {"multi": "multi-answer", "minimal": "minimal", "single": "one answer", "forced": "forced"}
    bp = "\n".join(f"\\multicolumn{{3}}{{l}}{{pooled, {pol[k]} prompt ({v['k']} settings)}} & \\multicolumn{{3}}{{l}}{{{pct(v['mu'], True)} "
                   f"{'' if _nan(v['ci_low']) else _ci(v)}}} \\\\" for k, v in m["by_policy"].items())
    mods = "\n".join(f"\\multicolumn{{3}}{{l}}{{{k}}} & \\multicolumn{{3}}{{l}}{{{pct(v['est'], True)} {_ci(v)}}} \\\\"
                     for k, v in m["moderators"].items())
    (out / "tab_pooled.tex").write_text(r"""\begin{table*}[t]
\centering\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llrllr}
\toprule
Policy & Setting & Neutral flag & Policy & Setting & Neutral flag \\
\midrule
""" + body + r"""
\midrule
\multicolumn{6}{l}{\emph{Multilevel model (random effects for generator, benchmark and setting; REML)}} \\
\multicolumn{3}{l}{pooled, all settings} & \multicolumn{3}{l}{\neuMlMu\ \neuMlCI; prediction interval \neuMlPI} \\
""" + bp + r"""
\midrule
\multicolumn{6}{l}{\emph{Meta-regression (same model)}} \\
""" + mods + r"""
\bottomrule
\end{tabular}
\caption{The """ + str(m["k"]) + r""" settings on misinformation benchmarks (RAMDocs, ConflictBank and its one-answer, minimal,
forced-choice and majority variants, RAGuard) in which the oracle neutral flag was run; WikiContradict (genuine
conflicts) and CONFLICTS are not pooled (conflict questions, change in
points), grouped by answer policy. The pooled estimate was registered as exploratory (\S13.4) and the multilevel model,
the policy split and the moderators in \S16.4; they use data already seen. The prediction interval describes this
design space (our generators, benchmarks and prompts), not deployments in general. \emph{Natural misinformation} is
RAGuard only; RAMDocs' misinformation is synthetic. The moderator \emph{several answers allowed} codes the
multi-answer and minimal prompts together, against one-answer prompts; the policy rows above separate them. Settings
share questions, so the model is fitted with their sampling covariance, estimated by a joint question bootstrap
(\S17.1: each replicate resamples each benchmark's questions once for every setting that uses them). Treating the
settings as independent gives """ + _mci(m) + r"""; conditional on these settings (question sampling only) the pooled
effect is \neuMlCIDesign.}
\label{tab:pooled}
\end{table*}
""")


def tab_utility6(R, out: Path):
    rev = R["rev"]
    ut, tr, nl = rev.get("utility6", {}), rev.get("transitions6", {}), rev.get("transitions_null", {})
    keys = [k for k in ("RD:gpt-4o-mini|default", "RD:gpt-6-luna", "RD:claude-haiku-5-5", "CB:gpt-4o-mini",
                        "CBs:gpt-4o-mini", "CBs:gpt-6-luna", "CBs:gpt-6.1-terra", "CBs:claude-haiku-5-5", "CBs:claude-sonnet-5-5",
                        "CBf:gpt-4o-mini", "CBmaj:gpt-4o-mini", "RG:gpt-4o-mini", "RG:gpt-6-luna", "RG:claude-haiku-5-5")
            if k in ut and "neutral" in ut[k]]
    share = lambda T, a, b: (pct(T.get(a, {}).get(b, 0) / max(sum(T.get(a, {}).values()), 1), nd=0)) if T else "--"
    rows = []
    for k in keys:
        n = ut[k]["neutral"]; d = ut[k].get("directional", {})
        T = tr.get(k, {}).get("neutral", {}); N = nl.get(k, {})
        rows.append(f"{_lab(k)} & " + " & ".join(cell(n[f'c{c}']) for c in (0, 1, 2, 4)) + " & " +
                    (cell(d['c2']) if d else "--") +
                    f" & {share(T, 'misinfo', 'abstain')} ({share(N, 'misinfo', 'abstain')}) & {share(T, 'misinfo', 'correct')} ({share(N, 'misinfo', 'correct')})"
                    f" & {share(T, 'correct', 'abstain')} ({share(N, 'correct', 'abstain')}) & {share(T, 'correct', 'misinfo')} ({share(N, 'correct', 'misinfo')}) \\\\")
    (out / "tab_utility.tex").write_text(r"""\begin{table*}[t]
\centering\footnotesize
\setlength{\tabcolsep}{3.5pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrrrrrrr}
\toprule
& \multicolumn{4}{c}{Neutral flag, utility change at misinformation cost $c$} & Dir.\ flag & \multicolumn{4}{c}{Neutral flag: plain answer $\to$ flagged answer (null)} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-6}\cmidrule(lr){7-10}
Setting & $c=0$ & $c=1$ & $c=2$ & $c=4$ & $c=2$ & mis.$\to$abst. & mis.$\to$corr. & corr.$\to$abst. & corr.$\to$mis. \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\caption{Abstention-aware view of the oracle flags (conflict questions; \S15.4 and \S16.4, re-analyses of data already
seen). Utility per answer: $+1$ correct, $-c$ if it contains the misinformation, 0 for abstaining or any other wrong
answer; $c=0$ is strict accuracy. Bold: significant in one Holm family per $c$ over all \nUtilSixSettings\ settings of
Table~\ref{tab:pooled} (a selection is shown). Right: of the question--seed pairs whose plain answer contained the
misinformation (or was correct), the share whose flagged answer abstained, was correct or contained the misinformation;
in brackets the same share between two plain runs with different seeds, the churn with no flag.}
\label{tab:utility}
\end{table*}
""")


def tab_h24redo6(R, out: Path):
    rev = R["rev"]
    red, rc = rev.get("H24_redo"), rev.get("H24_recast")
    if not (red and rc):
        return
    rg = red["regression"]
    f2ci = lambda t: f"{_f2(t['est'])} [{_f2(t['ci_low'])}, {_f2(t['ci_high'])}]"
    cv = red["lodo"]
    fol, rf = rc["follow"], rev.get("H31_follow", {})
    gens = [g for g in fol]
    crow = "\n".join(
        f"{NICE[g]} & {pct(fol[g]['follow_right'], nd=0)} & {pct(fol[g]['follow_wrong'], nd=0)} & {pct(fol[g]['abstain_wrong'], nd=0)} & "
        f"{pct(rf[g]['follow_right'], nd=0) if g in rf else '--'} & {pct(rf[g]['follow_wrong'], nd=0) if g in rf else '--'} \\\\" for g in gens)
    (out / "tab_h24redo.tex").write_text(r"""\begin{table}[t]
\centering\footnotesize
\setlength{\tabcolsep}{3pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lr}
\toprule
\multicolumn{2}{l}{\emph{Cell level (""" + str(len(red["cells"])) + r""" cells): accuracy with the flag, $A$}} \\
$A \sim D + G$: coefficient of $D$ (hit rate) & """ + f2ci(rg["add_D"]) + r""" \\
\quad coefficient of $G$ & """ + f2ci(rg["add_G"]) + r""" \\
$A \sim D$ + generator FE, permutation of $D$ & $p=""" + f"{red['perm_fe_D']['p_one_sided']:.2f}" + r"""$ \\
$A \sim P + R$ + gen.\ FE: precision $P$ & \recPCoef\ \recPCoefCI; $p=\recPPerm$ \\
\quad recall $R$ & \recRCoef\ \recRCoefCI; $p=\recRPerm$ \\
\midrule
\multicolumn{2}{l}{\emph{Leave one detector out: $R^2$ for the gain}} \\
$G$ only & """ + _f2(cv["G only"]["r2_cv"]) + r""" \\
gap $D-G$ (registered) & """ + _f2(cv["gap D - G (registered)"]["r2_cv"]) + r""" \\
$P+R+G$ & \recCvPRG \\
score (exploratory) & """ + _f2(cv["mechanism score"]["r2_cv"]) + r""" \\
\midrule
\multicolumn{2}{l}{\emph{Question level, question/detector/generator FE (points)}} \\
detector right on $q$ & \qfeD\ \qfeDCI \\
detector marks a correct passage & \qfeW\ \qfeWCI \\
\bottomrule
\end{tabular}}

\vspace{4pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lrrrrr}
\toprule
& \multicolumn{3}{c}{H24 cells (observational)} & \multicolumn{2}{c}{H31 (randomised)} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-6}
Generator & right & wrong & wrong, abst. & right & wrong \\
\midrule
""" + crow + r"""
\bottomrule
\end{tabular}}
\caption{H24 re-analysed (\S15.4, \S16.4; data already seen). $D$: hit rate (the detector flags the gold pair and names
exactly its misinformation passage); $P$: direction precision (of those gold-pair flags, the share naming it); $R$: recall (the
share of questions whose gold pair is flagged); $G$: the generator's no-flag accuracy. CIs: 2,000 question resamples
shared across cells (300 with question fixed effects; the leave-one-detector-out range of the question-level
coefficients is \qfeDLodo\ and \qfeWLodo). $p$: exact permutation of the detectors' values across the six detectors; the design is unbalanced (the Claude
detectors serve three generators), so detectors are only approximately exchangeable.
Lower table, one definition of following: a right flag is followed when the flagged answer is correct (questions the
generator gets wrong in every plain seed); a wrong flag is followed when the flagged answer contains the
misinformation (questions it gets right in every plain seed); \emph{abst.}: the flagged answer abstains instead. With
no flag, a held-out plain seed changes outcome on at most \folNullMax\% of such questions.}
\label{tab:h24redo}
\end{table}
""")


def tab_cells6(R, out: Path):
    """Adds the verdict-key column (H30) to tab_cells as written by assets5."""
    p = out / "tab_cells.tex"
    fa = R["rev"].get("H30a", {}).get(PRIMARY, {})
    if not p.exists():
        return
    s = p.read_text()
    s = s.replace(r"\begin{tabular}{llrrrrrrrrrr}", r"\begin{tabular}{llrrrrrrrrrrr}")
    s = s.replace(r"& \multicolumn{3}{c}{Accuracy with} & \multicolumn{2}{c}{Flag minus} \\",
                  r"& \multicolumn{4}{c}{Accuracy with} & \multicolumn{2}{c}{Flag minus} \\")
    s = s.replace(r"\cmidrule(lr){8-10}\cmidrule(lr){11-12}", r"\cmidrule(lr){8-11}\cmidrule(lr){12-13}")
    s = s.replace(r"Gain & flag & hint & det.\ alone & hint & det.\ alone \\", r"Gain & flag & verdict & hint & det.\ alone & hint & det.\ alone \\")
    out_lines = []
    for line in s.splitlines():
        if line.endswith(r"\\") and " & " in line and not line.startswith(("Detector", "&")):
            parts = line[:-2].rstrip().split(" & ")
            if len(parts) == 12:
                det, gen = parts[0], parts[1]
                g = next((k for k, v in NICE.items() if v == gen), gen); d = next((k for k, v in NICE.items() if v == det), det)
                t = next((v for k, v in fa.items() if k.startswith(f"{g}, detector {d}:")), None)
                parts.insert(8, pct(t["verdict"]) if t else "--")
                line = " & ".join(parts) + r" \\"
        out_lines.append(line)
    s = "\n".join(out_lines)
    s = s.replace(r"\emph{Hint}: the", r"\emph{Verdict}: the detector's judgement passed as an answer key (``the answer $X$ is false'' for the answer its suspect passage supports; H30). \emph{Hint}: the")
    p.write_text(s + "\n")


def fig_reliability6(R, out: Path):
    """Left: H31 right- and wrong-flag effects per generator. Right: flag accuracy against verdict key (open) and
    hint (filled) accuracy per H24 cell. Replaces the figure written by assets5."""
    import matplotlib.pyplot as plt
    from .paper_assets import INK2, NEUTRAL, _style
    rev = R["rev"]
    a, b = rev.get("H31a", {}), rev.get("H31b", {})
    cells = rev.get("H24", {}).get("cells", {})
    fa = rev.get("H27a", {}).get(PRIMARY, {}); fv = rev.get("H30a", {}).get(PRIMARY, {})
    if not a or not cells:
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
    gens = [k.split(":")[0] for k in a]
    for n, g in enumerate(gens):
        for off, fam, col in ((-0.17, a, "#2a78d6"), (0.17, b, "#eb6834")):
            t = next(v for k, v in fam.items() if k.startswith(g + ":"))
            ax.bar(n + off, 100 * t["diff"], width=0.32, color=col, edgecolor="white", linewidth=0.6)
            ax.errorbar(n + off, 100 * t["diff"], yerr=[[100 * (t["diff"] - t["ci_low"])], [100 * (t["ci_high"] - t["diff"])]],
                        fmt="none", ecolor=INK2, elinewidth=0.6)
    ax.axhline(0, color=INK2, lw=0.5)
    ax.set_xticks(range(len(gens))); ax.set_xticklabels([short[g] for g in gens], fontsize=5.5, rotation=25)
    ax.set_ylabel("change in accuracy (points)", fontsize=6)
    ax.set_title("randomised direction (H31)", fontsize=6.5, color=INK2, pad=2)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color="#2a78d6", label="right flag"),
                       plt.Rectangle((0, 0), 1, 1, color="#eb6834", label="wrong flag")], fontsize=5.5, loc="lower left")
    ax = axes[1]
    for key in cells:
        g, d = key.split("|")
        h = next((x for k, x in fa.items() if k.startswith(f"{g}, detector {d}:")), None)
        v = next((x for k, x in fv.items() if k.startswith(f"{g}, detector {d}:")), None)
        if h:
            ax.scatter(100 * h["mean_a"], 100 * h["mean_b"], s=14, marker=gen_mark.get(g, "o"), color=det_col.get(d, NEUTRAL),
                       edgecolor="white", linewidth=0.6, zorder=3)
        if v:
            ax.scatter(100 * v["verdict"], 100 * v["mean_b"], s=14, marker=gen_mark.get(g, "o"), facecolor="white",
                       edgecolor=det_col.get(d, NEUTRAL), linewidth=0.8, zorder=3)
    lo_, hi_ = ax.get_xlim(); lo2, hi2 = ax.get_ylim(); lim = [min(lo_, lo2), max(hi_, hi2)]
    ax.plot(lim, lim, ls="--", color=INK2, lw=0.6, zorder=1)
    ax.set_xlabel("verdict (open) or hint (filled), %", fontsize=6)
    ax.set_ylabel("accuracy with dir. flag (%)", fontsize=6)
    ax.set_title("flag vs verdict and hint", fontsize=6.5, color=INK2, pad=2)
    for x in axes:
        x.tick_params(labelsize=6)
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


def write6(R, out: Path):
    fig_reliability6(R, out)
    tab_pooled6(R, out); tab_utility6(R, out); tab_h24redo6(R, out); tab_cells6(R, out)
