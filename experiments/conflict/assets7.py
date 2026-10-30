"""
Paper macros, tables and the reliability figure for the seventh revision (re-analyses in revision7.py). numbers7 runs
inside paper_assets.numbers after numbers6; write7 runs after write6 and replaces fig_reliability.
revision7 stores points (x100) where revision6 stored fractions; `_pt` converts.
"""
from __future__ import annotations

from pathlib import Path

from .assets5 import GEN_A, NICE, _f2
from .paper_assets import _nan, pct

H31_ORDER = ("gpt-4o-mini", "gpt-6-luna", "claude-haiku-5-5", "gpt-6.1-terra")


def _pt(x, sign=False, nd=1):
    return "--" if x is None or _nan(x) else pct(x / 100, sign, nd)


def _ci_pt(lo, hi, nd=1):
    return f"[{_pt(lo, True, nd)}, {_pt(hi, True, nd)}]"


def numbers7(R, add):
    rev = R["rev"]
    for key, nm in (("H30a: LLM dir. flag minus verdict key", "PoolThirtyA"), ("H30b: verdict key minus hint", "PoolThirtyB"),
                    ("H23: oracle dir. flag minus answer key", "PoolTwentyThree")):
        t = rev.get("pooled7", {}).get(key)
        if t:
            add(f"{nm}", _pt(t["diff"], True)); add(f"{nm}CI", _ci_pt(t["ci_low"], t["ci_high"]))
            add(f"{nm}CINinety", _ci_pt(t["ci90_low"], t["ci90_high"]))
            add(f"{nm}P", "$<$0.001" if t["p"] < 0.001 else f"{t['p']:.3f}")
            add(f"{nm}Eq", "yes" if t["equivalent"] else "no"); add(f"{nm}K", t["n_cells"])
            add(f"{nm}Ahead", t["n_ahead"]); add(f"{nm}Behind", t["n_behind"])
    for d, p in rev.get("detector_precision7", {}).items():
        lo, hi = rev["detector_precision7_ci"][d]
        add(f"dirPrec{GEN_A[d]}", pct(p)); add(f"dirPrec{GEN_A[d]}CI", f"[{pct(lo)}, {pct(hi)}]")
    dp = rev.get("detector_precision7", {})
    if dp:
        others = [v for k, v in dp.items() if k != "gpt-4o-mini"]
        add("dirPrecOtherMin", pct(min(others))); add("dirPrecOtherMax", pct(max(others)))
    hx = rev.get("H31x", {})
    for g, v in hx.items():
        G = GEN_A[g]
        add(f"be{G}", f"{v['breakeven']:.2f}"); add(f"be{G}CI", f"[{v['be_low']:.2f}, {v['be_high']:.2f}]")
        add(f"rfolRight{G}CI", f"[{pct(v['follow_right_ci'][0], nd=0)}, {pct(v['follow_right_ci'][1], nd=0)}]")
        add(f"rfolWrong{G}CI", f"[{pct(v['follow_wrong_ci'][0], nd=0)}, {pct(v['follow_wrong_ci'][1], nd=0)}]")
        add(f"netHalf{G}", _pt(v["net"]["0.50"], True)); add(f"netObs{G}", _pt(v["net"][f"{v['d_obs']:.2f}"], True))
        for k, x in v["net"].items():
            if k.startswith("at "):
                det = k[3:].split(" (")[0]
                add(f"netAt{G}{GEN_A[det]}", _pt(x, True))
        add(f"head{G}Right", pct(v["headroom_right"], nd=0)); add(f"head{G}Wrong", pct(v["headroom_wrong"], nd=0))
    if hx:
        bes = [hx[g]["breakeven"] for g in hx]
        add("beMin", f"{min(bes):.2f}"); add("beMax", f"{max(bes):.2f}")
        add("dObs", f"{next(iter(hx.values()))['d_obs']:.2f}")
    p = rev.get("predict7")
    if p:
        add("predN", p["n"]); add("predR", f"{p['r']:.2f}"); add("predRCI", f"[{p['r_ci'][0]:.2f}, {p['r_ci'][1]:.2f}]")
        add("predRWithin", f"{p['r_within']:.2f}"); add("predSlope", f"{p['slope']:.2f}")
        add("predIntercept", _f2(p["intercept"])); add("predMae", f"{p['mae']:.1f}"); add("predSign", p["sign_agree"])
        add("predHarmObs", p["harm_obs"]); add("predHarmCalled", p["harm_called"])
        t = p.get("transfer")
        if t:
            add("predTrN", t["n"]); add("predTrR", f"{t['r']:.2f}"); add("predTrSlope", f"{t['slope']:.2f}")
            add("predTrMae", f"{t['mae']:.1f}"); add("predTrSign", t["sign_agree"])
        a = p["all"]
        add("predAllN", a["n"]); add("predAllR", f"{a['r']:.2f}"); add("predAllMae", f"{a['mae']:.1f}")
        add("predFollowRight", pct(p["mean_follow"][0], nd=0)); add("predFollowWrong", pct(p["mean_follow"][1], nd=0))
    fz = rev.get("H24_frozen")
    if fz:
        add("frozenShareMin", pct(min(fz["share_identical"].values()), nd=0))
        add("frozenMaxShift", f"{fz['max_gain_shift']:.1f}"); add("frozenR", f"{fz['r_gain_full_same']:.3f}")
    f = rev.get("fdr7")
    if f:
        add("fdrN", f["n_tests"]); add("fdrHolm", f["holm_sig"]); add("fdrBH", f["bh_sig"])
        add("fdrBhNotHolm", f["bh_not_holm"]); add("fdrHolmNotBh", f["holm_not_bh"])
    pv = rev.get("prevalence7", {})
    for key, nm in (("CBs:gpt-4o-mini|llm_dir+flag_directional", "SgMiniLlm"), ("CBs:gpt-4o-mini|oracle+flag_directional", "SgMiniDir"),
                    ("CBs:gpt-4o-mini|oracle+flag", "SgMiniNeu"), ("RD:gpt-4o-mini|default|llm_dir+flag_directional", "RdMiniLlm")):
        t = pv.get(key)
        if t:
            add(f"prev{nm}Five", _pt(t["pi0.05"], True)); add(f"prev{nm}Twenty", _pt(t["pi0.20"], True))
            add(f"prev{nm}Clean", _pt(t["clean"], True)); add(f"prev{nm}Conf", _pt(t["conflict"], True))
            add(f"prev{nm}Be", "--" if not t["breakeven_pi"] or t["breakeven_pi"] > 1 else pct(t["breakeven_pi"], nd=0))
    llm = [t for k, t in pv.items() if k.endswith("|llm_dir+flag_directional")]
    if llm:
        add("prevN", len(llm)); add("prevNegFive", sum(t["pi0.05"] < 0 for t in llm)); add("prevPosFive", sum(t["pi0.05"] > 0 for t in llm))
        add("prevNegConf", sum(t["conflict"] < 0 for t in llm))
    rg = rev.get("raguard7", {})
    pl = {k.split("|")[0]: v for k, v in rg.items() if k.endswith("|plain")}
    cb = {k.split("|")[0]: v for k, v in rg.items() if k.endswith("|closedbook")}
    if pl:
        add("rgAbstMin", _pt(min(v["abstain"] for v in pl.values()))); add("rgAbstMax", _pt(max(v["abstain"] for v in pl.values())))
        add("rgAnsMin", _pt(min(v["answered_acc"] for v in pl.values()))); add("rgAnsMax", _pt(max(v["answered_acc"] for v in pl.values())))
    if cb:
        add("rgCbMin", _pt(min(v["correct"] for v in cb.values()))); add("rgCbMax", _pt(max(v["correct"] for v in cb.values())))


def tab_breakeven7(R, out: Path):
    rev = R["rev"]; hx = rev.get("H31x", {}); dp = rev.get("detector_precision7", {})
    if not hx:
        return
    best = max(dp, key=dp.get) if dp else None
    lines = []
    for g in H31_ORDER:
        if g not in hx:
            continue
        v = hx[g]
        fr = f"{pct(v['follow_right'], nd=0)} [{pct(v['follow_right_ci'][0], nd=0)}, {pct(v['follow_right_ci'][1], nd=0)}]"
        fw = f"{pct(v['follow_wrong'], nd=0)} [{pct(v['follow_wrong_ci'][0], nd=0)}, {pct(v['follow_wrong_ci'][1], nd=0)}]"
        mini = next((x for k, x in v["net"].items() if k.startswith("at gpt-4o-mini")), None)
        bb = next((x for k, x in v["net"].items() if best and k.startswith(f"at {best}")), None)
        lines.append(f"{NICE[g]} & {_pt(v['right'], True)} & {_pt(v['wrong'], True)} & {v['breakeven']:.2f} "
                     f"[{v['be_low']:.2f}, {v['be_high']:.2f}] & {fr} & {fw} & {pct(v['headroom_right'], nd=0)} & "
                     f"{pct(v['headroom_wrong'], nd=0)} & {_pt(v['net']['0.50'], True)} & {_pt(mini, True)} & {_pt(bb, True)} \\\\")
    pr = rev.get("predict7", {})
    (out / "tab_breakeven.tex").write_text(r"""\begin{table*}[t]
\centering\small
\setlength{\tabcolsep}{3.5pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrlllrrrrr}
\toprule
& \multicolumn{2}{c}{Effect (points)} & & \multicolumn{2}{c}{Follow rate (\%)} & \multicolumn{2}{c}{Share of headroom (\%)} & \multicolumn{3}{c}{Net at direction precision} \\
\cmidrule(lr){2-3}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-11}
Generator & Right & Wrong & Break-even [95\% CI] & Right & Wrong & Right & Wrong & 0.50 & """ + pct(dp.get("gpt-4o-mini", float("nan"))) + r"""\% & """ + pct(dp.get(best, float("nan"))) + r"""\% \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}}
\caption{Randomised direction (H31; ConflictBank, one-answer prompt, conflict questions), re-analysed
(exploratory). \emph{Break-even}: the direction precision at which a flag on the true pair neither helps nor hurts,
$-W/(R-W)$ for right-flag effect $R$ and wrong-flag effect $W$; bootstrap CI within arms. \emph{Follow rate}: a right
flag is followed when an answer that was wrong without the flag becomes correct, a wrong flag when a correct answer comes to
contain the misinformation; question-bootstrap CIs. \emph{Share of headroom}: $R$ over the no-flag error rate of the
right arm, and $-W$ over the no-flag accuracy of the wrong arm. \emph{Net}: $pR+(1-p)W$ at direction precision $p$:
0.50, gpt-4o-mini's on ConflictBank, and the most precise detector's (""" + (NICE.get(best, "--") if best else "--") + r"""). These
nets assume the detector flags the true pair; real detectors also flag other pairs and clean questions.
Figure~\ref{fig:reliability} (right) uses the follow rates to predict the """ + str(pr.get("all", {}).get("n", "--")) + r""" crossed cells.}
\label{tab:breakeven}
\end{table*}
""")


def tab_prevalence7(R, out: Path):
    from .assets6 import _lab
    pv = R["rev"].get("prevalence7", {})
    if not pv:
        return
    COND = {"oracle+flag": "oracle, neutral", "oracle+flag_directional": "oracle, directional",
            "llm_dir+flag_directional": "LLM detector, directional"}
    lines = []
    for key in sorted(k for k in pv if k.endswith("|llm_dir+flag_directional")):
        set_, cond = key.rsplit("|", 1)
        t = pv[key]
        be = "--" if not t["breakeven_pi"] or t["breakeven_pi"] > 1 else pct(t["breakeven_pi"], nd=0)
        lines.append(f"{_lab(set_)} & {_pt(t['conflict'], True)} & {_pt(t['clean'], True)} & "
                     f"{_pt(t['pi0.05'], True)} & {_pt(t['pi0.20'], True)} & {be} \\\\")
    (out / "tab_prevalence.tex").write_text(r"""\begin{table}[t]
\centering\scriptsize
\setlength{\tabcolsep}{2.5pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lrrrrr}
\toprule
Setting & Conflict & Clean & 5\% & 20\% & Break-even \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}}
\caption{The gpt-4o-mini LLM detector's directional flag: effect (points, against no flags) on conflict and clean
questions, and reweighted to a 5\% and a 20\% share of conflict questions (exploratory). \emph{Break-even}: the share
of conflict questions (\%) above which the flag helps, where it hurts on clean questions. Oracle flags never fire on
clean questions, so their reweighted effect is their conflict effect times the share.}
\label{tab:prevalence}
\end{table}
""")


def fig_reliability7(R, out: Path):
    """Left: H31 right- and wrong-flag effects per generator. Right: each crossed cell's flag effect predicted from
    H31's follow rates against the observed effect (filled: the generator's own follow rates; open: the mean of the
    four, for generators with no H31 run)."""
    import matplotlib.pyplot as plt
    from .paper_assets import INK2, NEUTRAL, _style
    rev = R["rev"]
    a, b = rev.get("H31a", {}), rev.get("H31b", {})
    p = rev.get("predict7")
    if not a or not p:
        return
    det_col = {"gpt-4o-mini": "#2a78d6", "gpt-4.1": "#eb6834", "gpt-6-luna": "#1baf7a", "gpt-6.1-terra": "#eda100",
               "claude-haiku-5-5": "#e87ba4", "claude-sonnet-5-5": "#008300"}
    gen_mark = {"gpt-4o-mini": "o", "gpt-6-luna": "s", "gpt-4.1": "^", "gpt-6.1-terra": "D", "claude-haiku-5-5": "P",
                "Qwen2.5-7B": "v"}
    short = {"gpt-4o-mini": "4o-mini", "gpt-4.1": "4.1", "gpt-6-luna": "6-luna", "gpt-6.1-terra": "6.1-terra",
             "claude-haiku-5-5": "Haiku 5.5", "claude-sonnet-5-5": "Sonnet 5.5", "Qwen2.5-7B": "Qwen-7B"}
    # right/wrong bars in neutral greys, so they are not read as the detector colours of the right panel
    RIGHT_C, WRONG_C = "#3d3d3d", "#b3b3b3"
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(3.3, 2.3))
    ax = axes[0]
    gens = [k.split(":")[0] for k in a]
    for n, g in enumerate(gens):
        for off, fam, col in ((-0.17, a, RIGHT_C), (0.17, b, WRONG_C)):
            t = next(v for k, v in fam.items() if k.startswith(g + ":"))
            ax.bar(n + off, 100 * t["diff"], width=0.32, color=col, edgecolor="white", linewidth=0.6)
            ax.errorbar(n + off, 100 * t["diff"], yerr=[[100 * (t["diff"] - t["ci_low"])], [100 * (t["ci_high"] - t["diff"])]],
                        fmt="none", ecolor=INK2, elinewidth=0.6)
    ax.axhline(0, color=INK2, lw=0.5)
    ax.set_xticks(range(len(gens))); ax.set_xticklabels([short[g] for g in gens], fontsize=6, rotation=35, ha="right",
                                                        rotation_mode="anchor")
    ax.set_ylabel("change in accuracy (points)", fontsize=7)
    ax.set_title("(a) H31: random direction", fontsize=6.5, color=INK2, pad=3)
    ax.set_ylim(top=max(ax.get_ylim()[1], 75))
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=RIGHT_C, label="right flag"),
                       plt.Rectangle((0, 0), 1, 1, color=WRONG_C, label="wrong flag")], fontsize=6, loc="upper center",
              ncol=2, frameon=False, handlelength=1, columnspacing=0.8, borderaxespad=0.1)
    ax = axes[1]
    for c in p["cells"]:
        col = det_col.get(c["det"], NEUTRAL)
        kw = dict(color=col, edgecolor="white") if c["kind"] == "own" else dict(facecolor="white", edgecolor=col)
        ax.scatter(c["pred"], c["obs"], s=14, marker=gen_mark.get(c["gen"], "o"), linewidth=0.8, zorder=3, **kw)
    lo_, hi_ = ax.get_xlim(); lo2, hi2 = ax.get_ylim(); lim = [min(lo_, lo2), max(hi_, hi2)]
    ax.plot(lim, lim, ls="--", color=INK2, lw=0.6, zorder=1)
    ax.axhline(0, color=INK2, lw=0.4, zorder=1); ax.axvline(0, color=INK2, lw=0.4, zorder=1)
    ax.set_xlabel("predicted from H31 (points)", fontsize=7)
    ax.set_ylabel("observed flag effect (points)", fontsize=7)
    ax.set_title(f"(b) {len(p['cells'])} crossed cells, r = {p['all']['r']:.2f}", fontsize=6.5, color=INK2, pad=3)
    for x in axes:
        x.tick_params(labelsize=7)
    cells = {(c["gen"], c["det"]) for c in p["cells"]}
    hd = [plt.Line2D([], [], marker="o", ls="", ms=4, color=c, label=short[k])
          for k, c in det_col.items() if any(d == k for _, d in cells)]
    hg = [plt.Line2D([], [], marker=m, ls="", ms=4, color=INK2, label=short[k])
          for k, m in gen_mark.items() if any(g == k for g, _ in cells)]
    fig.tight_layout(w_pad=2.0, rect=(0, 0.1, 1, 1))
    # two stacked legend rows under both panels, one per encoding, so they cannot collide
    fig.legend(handles=hd, title="(b) detector (colour)", fontsize=6, title_fontsize=6.5, ncol=6, frameon=False,
               loc="upper center", bbox_to_anchor=(0.5, 0.11), handletextpad=0.05, columnspacing=0.5)
    fig.legend(handles=hg, title="(b) generator (shape; open = no H31 run)", fontsize=6, title_fontsize=6.5, ncol=6,
               frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.0), handletextpad=0.05, columnspacing=0.5)
    fig.savefig(out / "fig_reliability.pdf", bbox_inches="tight")
    plt.close(fig)


def write7(R, out: Path):
    fig_reliability7(R, out)
    tab_breakeven7(R, out); tab_prevalence7(R, out)
