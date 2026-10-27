"""
Tables (LaTeX, booktabs), figures (PDF) and number macros for the paper, from the results dict
built by paper_stats.py. Colors follow the dataviz reference palette (validated: categorical slots
1-4 pass CVD and normal-vision checks on the light surface; two slots are below 3:1 contrast, so
every mark is also direct-labelled or shape-coded).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

PRIMARY, ROBUST = "ramdocs_strict_oneway", "ramdocs_strict"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SURFACE = "#fcfcfb"
CAT = {"oracle": "#2a78d6", "llm_dir": "#eb6834", "nli": "#1baf7a", "caag": "#eda100"}
NEUTRAL = "#8a8984"
MARK = {"flag": "o", "flag_directional": "^", "remove": "s", "remove_suspect": "s"}
FORMAT_NAME = {"flag": "flag", "flag_directional": "directional flag", "remove": "remove",
               "remove_suspect": "remove suspect"}
DET_NAME = {"oracle": "Oracle", "llm_dir": "LLM", "nli": "NLI", "caag": "CAAG", "random": "Random",
            "none": ""}
COND_NAME = {
    "instruct": "Instruction (v1)", "instruct_v2": "Instruction (v2)",
    "oracle+flag": "Oracle, flag", "oracle+flag_directional": "Oracle, directional flag",
    "oracle+remove_suspect": "Oracle, remove suspect", "random+flag": "Random, flag",
    "caag+flag": "CAAG, flag", "caag+remove": "CAAG, remove", "nli+flag": "NLI, flag",
    "nli+remove": "NLI, remove", "nli+remove_suspect": "NLI, remove",
    "llm_dir+flag": "LLM, flag", "llm_dir+flag_directional": "LLM, directional flag",
    "llm_dir+remove_suspect": "LLM, remove suspect", "madam": "MADAM-RAG",
}


def _nan(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


def f3(x, sign=True):
    return "--" if _nan(x) else (f"{x:+.3f}" if sign else f"{x:.3f}")


def pct(x, sign=False, nd=1):
    """Percentage points; signed values use a real minus/plus sign."""
    if _nan(x):
        return "--"
    if not sign:
        return f"{100*x:.{nd}f}"
    v = f"{100*x:+.{nd}f}"
    if v in (f"-{0:.{nd}f}", f"+{0:.{nd}f}"):
        v = f"{0:.{nd}f}"
        return v
    return ("$-$" if v[0] == "-" else "$+$") + v[1:]


def cell(t, holm_key="p_holm"):
    """Difference in points, bold if significant after Holm (or raw p when no family)."""
    if not t or _nan(t.get("diff")):
        return "--"
    s = pct(t["diff"], sign=True)
    p = t.get(holm_key, t.get("p"))
    return f"\\textbf{{{s}}}" if (p is not None and not _nan(p) and p < 0.05) else s


def _holm_table(table: dict) -> dict:
    """Holm across all (condition, subset) comparisons of a table, so bold = family-wise significant."""
    from .stats import holm
    keys = {f"{c}|{s}": v["p"] for c, d in table.items() for s, v in d.items() if not _nan(v.get("p"))}
    adj = holm(keys)
    for c, d in table.items():
        for s, v in d.items():
            v["p_table_holm"] = adj.get(f"{c}|{s}", float("nan"))
    return table


# --------------------------------------------------------------------------- tables
def tab_detectors(R, out: Path):
    q = R["H1"]["quality"]
    rows = []
    for n, name in (("caag", "CAAG (heuristic)"), ("nli", "NLI cross-encoder"), ("llm_dir", "LLM (gpt-4o-mini)")):
        v = q[n]; d = v["doc_remove"]
        rows.append(f"{name} & {pct(v['pair_precision'])} & {pct(v['pair_recall'])} & {pct(v['pair_f1'])} & "
                    f"{pct(v['clean_fpr'])} & {pct(v['direction_accuracy']) if v['direction_accuracy'] is not None else '--'} & "
                    f"{pct(d['conflict_q_all_misinfo_marked'])} & {pct(d['clean_q_any_correct_marked'])} & "
                    f"{v['latency_s']:.2f} & {v['cost_per_1k_q']:.3f} \\\\")
    body = "\n".join(rows)
    (out / "tab_detectors.tex").write_text(rf"""\begin{{table*}}[t]
\centering\small
\begin{{tabular}}{{lrrrrrrrrr}}
\toprule
& \multicolumn{{3}}{{c}}{{Pair level}} & Clean-q. & Direc- & \multicolumn{{2}}{{c}}{{Removal, per question}} & Latency & Cost \\
\cmidrule(lr){{2-4}}\cmidrule(lr){{7-8}}
Detector & P & R & F1 & flagged & tion & misinfo gone & correct lost & (s) & (\$/1k q.) \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\caption{{Detectors on the RAMDocs hold-out split (300 questions; percentages). \emph{{Clean-q.\ flagged}}: share of
conflict-free questions with at least one flag. \emph{{Direction}}: share of correctly flagged contradictions whose suspect
side is the misinformation. \emph{{Misinfo gone}}: conflict questions whose misinformation documents are all removed
when removing the suspect (or, without direction, the second passage). \emph{{Correct lost}}: clean questions that lose
at least one correct document. Latency is per question; CAAG and NLI run locally.}}
\label{{tab:detectors}}
\end{{table*}}
""")


def tab_main(R, out: Path):
    for m, tag in ((PRIMARY, ""), (ROBUST, "_robust")):
        t = _holm_table(R["main_table"][m])
        base = R["plain_means"][m]
        groups = [("Prompt only", ["instruct", "instruct_v2"]),
                  ("Oracle detector", ["oracle+flag", "oracle+flag_directional", "oracle+remove_suspect"]),
                  ("Real detectors", ["random+flag", "caag+flag", "caag+remove", "nli+flag", "nli+remove",
                                      "llm_dir+flag", "llm_dir+flag_directional", "llm_dir+remove_suspect"]),
                  ("Reference", ["madam"])]
        lines = [f"No flags (plain) & {pct(base['all'])} & {pct(base['conflict'])} & {pct(base['clean'])} & "
                 f"{R['main_costs']['plain']['prompt_tokens']:.0f} \\\\", "\\midrule"]
        for gname, cs in groups:
            lines.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{{gname}}}}} \\\\")
            for c in cs:
                if c not in t:
                    continue
                lines.append(f"\\quad {COND_NAME[c]} & {cell(t[c]['all'], 'p_table_holm')} & "
                             f"{cell(t[c]['conflict'], 'p_table_holm')} & {cell(t[c]['clean'], 'p_table_holm')} & "
                             f"{R['main_costs'][c]['prompt_tokens']:.0f} \\\\")
        metric_name = "one-way strict accuracy (primary)" if m == PRIMARY else "bidirectional strict accuracy (robustness)"
        (out / f"tab_main{tag}.tex").write_text(rf"""\begin{{table}}[t]
\centering\small
\setlength{{\tabcolsep}}{{2.5pt}}
\begin{{tabular}}{{lrrrr}}
\toprule
Condition & All & Conflict & Clean & Tokens \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}
\caption{{RAMDocs hold-out (300 questions: 139 conflict, 126 clean), gpt-4o-mini, 3 seeds, {metric_name}.
First row: accuracy (\%) with no flags. Other rows: paired difference from it in points; \textbf{{bold}} = significant
after Holm correction across the table (sign-flip test, $p<0.05$). Tokens: mean prompt tokens per call
(MADAM-RAG: per question, all calls).}}
\label{{tab:main{tag}}}
\end{{table}}
""")


def tab_gpt41(R, out: Path):
    if "gpt41_table" not in R:
        return
    t = _holm_table(R["gpt41_table"][PRIMARY])
    b = R["gpt41_plain"][PRIMARY]
    lines = [f"No flags (plain) & {pct(b['all'])} & {pct(b['conflict'])} & {pct(b['clean'])} \\\\", "\\midrule"]
    for c in t:
        lines.append(f"{COND_NAME[c]} & {cell(t[c]['all'], 'p_table_holm')} & {cell(t[c]['conflict'], 'p_table_holm')} & "
                     f"{cell(t[c]['clean'], 'p_table_holm')} \\\\")
    (out / "tab_gpt41.tex").write_text(rf"""\begin{{table}}[t]
\centering\small
\begin{{tabular}}{{lrrr}}
\toprule
Condition & All & Conflict & Clean \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}
\caption{{Replication with gpt-4.1 as generator (RAMDocs hold-out, 3 seeds, one-way strict accuracy; detector fixed
to gpt-4o-mini). Same conventions as Table~\ref{{tab:main}}.}}
\label{{tab:gpt41}}
\end{{table}}
""")


def tab_wiki(R, out: Path):
    if "H8" not in R:
        return
    t = R["H8"]["both_answers"]
    lines = [f"No flags (plain) & {pct(R['wiki_plain'])} \\\\", "\\midrule"]
    for k, v in t.items():
        c = k.replace(" vs plain", "")
        lines.append(f"{COND_NAME[c]} & {cell(v)} \\\\")
    (out / "tab_wiki.tex").write_text(rf"""\begin{{table}}[t]
\centering\small
\begin{{tabular}}{{lr}}
\toprule
Condition & Both answers \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}
\caption{{WikiContradict (253 genuine conflicts between two Wikipedia passages; the reference answer is both sides),
gpt-4o-mini, 3 seeds. First row: share of answers giving both sides (\%); other rows: difference in points,
\textbf{{bold}} = significant after Holm within H8.}}
\label{{tab:wiki}}
\end{{table}}
""")


def tab_hypotheses(R, out: Path):
    two_sided = {"H5", "H7"}

    def status(fam_by_metric, key, metric=PRIMARY, h=""):
        v = fam_by_metric[metric][key]
        ok = v["supported"]
        rob = all(fam_by_metric[m][key]["supported"] for m in fam_by_metric)
        if h in two_sided:  # no predicted direction: report significance only
            return ("significant" + ("" if rob else "$^\\dagger$")) if ok else "n.s."
        return ("supported" + ("" if rob else "$^\\dagger$")) if ok else "not supported"
    rows = []
    for k, v in R["H1"]["tests"].items():
        rows.append(("H1", k, f"{pct(v['diff'], True)} [{pct(v['ci_low'], True)}, {pct(v['ci_high'], True)}]",
                     _p(v["p_holm"]), "supported" if v["supported"] else "not supported"))
    for h in ("H2", "H3", "H5", "H7", "H9"):
        if h not in R:
            continue
        for k, v in R[h][PRIMARY].items():
            rows.append((h, k, f"{pct(v['diff'], True)} [{pct(v['ci_low'], True)}, {pct(v['ci_high'], True)}]",
                         _p(v["p_holm"]), status(R[h], k, h=h)))
    h4 = R["H4"][PRIMARY]
    rows.append(("H4a", f"false-flag cells below plain on clean q.",
                 f"{h4['n_harmful']} of {h4['n_cells']} cells", "Holm", "supported" if h4["n_harmful"] else "not supported"))
    dr = h4["dose_response"]
    rows.append(("H4b", "harm grows with clean-q. exposure (Spearman)", f"$\\rho={dr['spearman_rho']:+.2f}$",
                 _p(dr["p_perm"]), "supported" if dr["p_perm"] < 0.05 and dr["spearman_rho"] < 0 else "not supported"))
    rows.append(("H4c", "removal harms clean q. more than directional flag", f"{h4['n_remove_worse']} of "
                 f"{len(h4['remove_vs_flag'])} matched cells", "Holm", "supported" if h4["n_remove_worse"] else "not supported"))
    s6 = R["H6"][PRIMARY]["summary"]
    rows.append(("H6", "document-level placement inside actual CI",
                 f"{s6['doc']['inside']}/{s6['doc']['n']} (pair: {s6['pair']['inside']}/{s6['pair']['n']})",
                 f"MAE {100*s6['doc']['mae']:.1f} vs {100*s6['pair']['mae']:.1f}",
                 "supported" if s6["doc"]["mae"] < s6["pair"]["mae"] else "not supported"))
    if "H8" in R:
        for k, v in R["H8"]["both_answers"].items():
            directional = k.startswith("oracle+flag")
            out_s = (("supported" if v["supported"] else "not supported") if directional
                     else ("significant" if v["supported"] else "n.s."))
            rows.append(("H8", "WikiContradict: " + k, f"{pct(v['diff'], True)} [{pct(v['ci_low'], True)}, {pct(v['ci_high'], True)}]",
                         _p(v["p_holm"]), out_s))
    body = "\n".join(f"{a} & {b.replace('_', chr(92)+'_').replace('>', '$>$').replace('<', '$<$')} & {c} & {d} & {e} \\\\"
                     for a, b, c, d, e in rows)
    (out / "tab_hypotheses.tex").write_text(rf"""\begin{{table*}}[t]
\centering\small
\begin{{tabular}}{{lp{{6.0cm}}lll}}
\toprule
& Pre-registered test & Estimate [95\% CI] & $p_{{\text{{Holm}}}}$ & Outcome \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\caption{{All pre-registered tests on the confirmatory split (differences in points of one-way strict accuracy unless
stated). Holm correction within each family. Directional predictions are \emph{{supported}} or \emph{{not supported}};
tests with no predicted direction (H5, H7, and H8 apart from the oracle flag) are \emph{{significant}} or \emph{{n.s.}}
$^\dagger$: holds under the primary metric but not under the bidirectional robustness metric.}}
\label{{tab:hypotheses}}
\end{{table*}}
""")


def _p(p):
    if _nan(p):
        return "--"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


# --------------------------------------------------------------------------- figures
def _style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
        "axes.facecolor": "white", "legend.frameon": False, "pdf.fonttype": 42})


def fig_tradeoff(R, out: Path, table_key: str = "main_table", name: str = "fig_tradeoff.pdf"):
    """Benefit on conflict questions vs harm on clean ones, per detector and format.
    Colour = detector (legend), shape = format (legend); direct labels only away from the origin cluster."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    _style()
    t = R[table_key][PRIMARY]
    fig, ax = plt.subplots(figsize=(3.3, 3.0))
    ax.axhline(0, color=INK2, lw=0.8); ax.axvline(0, color=INK2, lw=0.8)
    short = {"flag": "flag", "flag_directional": "dir. flag", "remove": "remove", "remove_suspect": "remove"}
    # hand-placed labels (offset in points, alignment) for the points the text discusses
    place = {"oracle+remove_suspect": (6, -3, "left"), "oracle+flag_directional": (6, 2, "left"),
             "llm_dir+flag_directional": (-6, -10, "right"), "llm_dir+remove_suspect": (6, -3, "left"),
             "caag+remove": (-6, 5, "right"), "nli+remove": (-6, -10, "right"),
             "oracle+flag": (6, -4, "left")}
    for c, v in t.items():
        det, _, mode = c.partition("+")
        if not mode or (mode == "remove" and f"{det}+remove_suspect" in t):
            continue  # without direction, remove == remove_suspect: plot once
        x, y = 100 * v["clean"]["diff"], 100 * v["conflict"]["diff"]
        color = CAT.get(det, NEUTRAL)
        ax.errorbar(x, y, xerr=[[x - 100 * v["clean"]["ci_low"]], [100 * v["clean"]["ci_high"] - x]],
                    yerr=[[y - 100 * v["conflict"]["ci_low"]], [100 * v["conflict"]["ci_high"] - y]],
                    fmt="none", ecolor=color, elinewidth=0.6, alpha=0.55, zorder=2)
        ax.scatter(x, y, s=36, marker=MARK.get(mode, "o"), color=color, edgecolor="white", linewidth=1.2, zorder=3)
        if c in place:
            dx, dy, ha = place[c]
            ax.annotate(f"{DET_NAME[det]} {short[mode]}", (x, y), xytext=(dx, dy), textcoords="offset points",
                        fontsize=6, color=INK, ha=ha)
    for c in ("instruct", "instruct_v2"):
        if c in t:
            v = t[c]
            ax.scatter(100 * v["clean"]["diff"], 100 * v["conflict"]["diff"], s=24, marker="D", color="white",
                       edgecolor=INK2, linewidth=1.0, zorder=3)
    ax.set_xlabel("Change on clean questions (points)")
    ax.set_ylabel("Change on conflict questions (points)")
    dets = [Line2D([], [], marker="o", ls="", color=CAT[d], label=DET_NAME[d]) for d in ("oracle", "llm_dir", "nli", "caag")]
    dets.append(Line2D([], [], marker="o", ls="", color=NEUTRAL, label="Random"))
    fmts = [Line2D([], [], marker=MARK[m], ls="", color=INK2, label=FORMAT_NAME[m])
            for m in ("flag", "flag_directional", "remove_suspect")]
    fmts.append(Line2D([], [], marker="D", ls="", markerfacecolor="white", markeredgecolor=INK2, label="instruction only"))
    l1 = ax.legend(handles=dets, fontsize=5.8, loc="upper left", handletextpad=0.2, borderaxespad=0.2)
    ax.add_artist(l1)
    ax.legend(handles=fmts, fontsize=5.8, loc="lower left", handletextpad=0.2, borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(out / name); plt.close(fig)


def _diverging():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("bluered", ["#e34948", "#f0efec", "#2a78d6"])


def fig_grid(R, out: Path, grid: dict):
    """Precision x direction heatmaps at recall 1, per mode and subset."""
    import matplotlib.pyplot as plt
    _style()
    modes = [("flag_directional", "Directional flag"), ("remove_suspect", "Remove suspect")]
    subsets = [("conflict", "Conflict questions"), ("clean", "Clean questions"), ("all", "All questions")]
    ps, ds = [0.25, 0.5, 0.75, 1.0], [0.5, 0.75, 1.0]
    vmax = 30
    fig, axes = plt.subplots(2, 3, figsize=(6.6, 3.6), sharex=True, sharey=True)
    for i, (mode, mname) in enumerate(modes):
        for j, (s, sname) in enumerate(subsets):
            ax = axes[i, j]
            Z = np.full((len(ps), len(ds)), np.nan); S = np.zeros_like(Z, dtype=bool)
            for a, p in enumerate(ps):
                for b, d in enumerate(ds):
                    v = grid.get((mode, p, 1.0, d, s))
                    if v:
                        Z[a, b] = 100 * v["diff"]; S[a, b] = v["p_holm"] < 0.05
            ax.imshow(Z, cmap=_diverging(), vmin=-vmax, vmax=vmax, origin="lower", aspect="auto")
            for a in range(len(ps)):
                for b in range(len(ds)):
                    if not np.isnan(Z[a, b]):
                        txt = f"{int(round(Z[a, b])):+d}" + ("*" if S[a, b] else "")
                        ax.text(b, a, txt, ha="center", va="center", fontsize=6.5,
                                color="white" if abs(Z[a, b]) > 0.6 * vmax else INK)
            ax.set_xticks(range(len(ds)), [f"{d:g}" for d in ds]); ax.set_yticks(range(len(ps)), [f"{p:g}" for p in ps])
            ax.grid(False)
            if i == 0:
                ax.set_title(sname, fontsize=8, color=INK)
            if j == 0:
                ax.set_ylabel(f"{mname}\nprecision")
            if i == 1:
                ax.set_xlabel("direction accuracy")
    fig.tight_layout()
    fig.savefig(out / "fig_grid.pdf"); plt.close(fig)


def fig_placement(R, out: Path):
    import matplotlib.pyplot as plt
    _style()
    cases = R["H6"][PRIMARY]["cases"]
    fig, axes = plt.subplots(1, 2, figsize=(4.4, 2.2), sharex=True, sharey=True)
    for ax, space, title in ((axes[0], "doc", "Document-level placement"), (axes[1], "pair", "Pair-level placement")):
        lo = -32; hi = 28
        ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.8, ls="--")
        for k, v in cases.items():
            cond, sub, sp = k.split("|")
            if sp != space:
                continue
            det = cond.split("+")[0]
            ax.errorbar(100 * v["predicted"], 100 * v["actual"],
                        yerr=[[100 * (v["actual"] - v["ci_low"])], [100 * (v["ci_high"] - v["actual"])]],
                        fmt="none", ecolor=CAT[det], elinewidth=0.6, alpha=0.7)
            ax.scatter(100 * v["predicted"], 100 * v["actual"], color=CAT[det],
                       marker="o" if sub == "conflict" else "s", s=22, edgecolor="white", linewidth=1.0, zorder=3)
        ax.set_title(title, fontsize=8, color=INK); ax.set_xlabel("Predicted change (points)")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    axes[0].set_ylabel("Actual change (points)")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=CAT[d], label=DET_NAME[d]) for d in ("caag", "nli", "llm_dir")] + \
        [Line2D([], [], marker="o", ls="", color=INK2, label="conflict q."),
         Line2D([], [], marker="s", ls="", color=INK2, label="clean q.")]
    axes[1].legend(handles=h, fontsize=5.5, loc="upper left", handletextpad=0.2)
    fig.tight_layout()
    fig.savefig(out / "fig_placement.pdf"); plt.close(fig)


# --------------------------------------------------------------------------- macros
def numbers(R, out: Path):
    from .paper_stats import macro
    m = []
    t = R["main_table"][PRIMARY]
    add = lambda n, v: m.append(macro(n, v))
    b = R["plain_means"][PRIMARY]
    add("plainAll", pct(b["all"])); add("plainConf", pct(b["conflict"])); add("plainClean", pct(b["clean"]))
    keymap = {"OracleFlag": "oracle+flag", "OracleDir": "oracle+flag_directional", "OracleRem": "oracle+remove_suspect",
              "LlmFlag": "llm_dir+flag", "LlmDir": "llm_dir+flag_directional", "LlmRem": "llm_dir+remove_suspect",
              "NliFlag": "nli+flag", "NliRem": "nli+remove", "CaagFlag": "caag+flag", "CaagRem": "caag+remove",
              "RandFlag": "random+flag", "InstrOne": "instruct", "InstrTwo": "instruct_v2", "Madam": "madam"}
    for k, c in keymap.items():
        for s, sn in (("all", "All"), ("conflict", "Conf"), ("clean", "Clean")):
            v = t[c][s]
            add(f"d{k}{sn}", pct(v["diff"], True)); add(f"d{k}{sn}CI", f"[{pct(v['ci_low'], True)}, {pct(v['ci_high'], True)}]")
    q = R["H1"]["quality"]
    for n, k in (("caag", "Caag"), ("nli", "Nli"), ("llm_dir", "Llm")):
        add(f"fOne{k}", pct(q[n]["pair_f1"])); add(f"cleanFpr{k}", pct(q[n]["clean_fpr"]))
        add(f"misGone{k}", pct(q[n]["doc_remove"]["conflict_q_all_misinfo_marked"]))
        add(f"corLost{k}", pct(q[n]["doc_remove"]["clean_q_any_correct_marked"]))
    add("dirAccLlm", pct(q["llm_dir"]["direction_accuracy"]))
    h4 = R["H4"][PRIMARY]
    add("hFourHarm", h4["n_harmful"]); add("hFourCells", h4["n_cells"])
    add("hFourRho", f"{h4['dose_response']['spearman_rho']:+.2f}")
    add("hFourRemWorse", h4["n_remove_worse"]); add("hFourMatched", len(h4["remove_vs_flag"]))
    s6 = R["H6"][PRIMARY]["summary"]
    add("hSixDocIn", s6["doc"]["inside"]); add("hSixPairIn", s6["pair"]["inside"]); add("hSixN", s6["doc"]["n"])
    add("hSixDocMae", pct(s6["doc"]["mae"])); add("hSixPairMae", pct(s6["pair"]["mae"]))
    if "H8" in R:
        add("wikiPlain", pct(R["wiki_plain"]))
        for c, k in (("oracle+flag", "OracleFlag"), ("llm_dir+flag", "LlmFlag"), ("instruct_v2", "InstrTwo"),
                     ("nli+flag", "NliFlag"), ("caag+flag", "CaagFlag"), ("instruct", "InstrOne")):
            v = R["H8"]["both_answers"][f"{c} vs plain"]
            add(f"wiki{k}", pct(v["diff"], True)); add(f"wiki{k}CI", f"[{pct(v['ci_low'], True)}, {pct(v['ci_high'], True)}]")
    if "wiki_gpt41" in R:
        add("wikiGptPlain", pct(R["wiki_gpt41"]["plain"]))
        v = R["wiki_gpt41"]["oracle+flag"]
        add("wikiGptOracleFlag", pct(v["diff"], True)); add("wikiGptOracleFlagCI", f"[{pct(v['ci_low'], True)}, {pct(v['ci_high'], True)}]")
    if "gpt41_table" in R:
        g = R["gpt41_table"][PRIMARY]; gb = R["gpt41_plain"][PRIMARY]
        add("gptPlainConf", pct(gb["conflict"])); add("gptPlainClean", pct(gb["clean"]))
        for c, k in (("oracle+flag", "OracleFlag"), ("oracle+flag_directional", "OracleDir"),
                     ("oracle+remove_suspect", "OracleRem"), ("llm_dir+remove_suspect", "LlmRem"),
                     ("llm_dir+flag_directional", "LlmDir")):
            for s, sn in (("conflict", "Conf"), ("clean", "Clean"), ("all", "All")):
                add(f"gpt{k}{sn}", pct(g[c][s]["diff"], True))
    if "madam_all500" in R:
        add("madamOneWay", pct(R["madam_all500"][PRIMARY])); add("madamBidir", pct(R["madam_all500"][ROBUST]))
    if "orig_format" in R:
        o = R["orig_format"][PRIMARY]
        for s, sn in (("all", "All"), ("conflict", "Conf"), ("clean", "Clean")):
            add(f"origCaag{sn}", pct(o[s]["diff"], True))
    for c, k in (("plain", "Plain"), ("oracle+flag", "OracleFlag"), ("oracle+flag_directional", "OracleDir"),
                 ("oracle+remove_suspect", "OracleRem")):
        if c in R.get("mechanism", {}):
            add(f"wrong{k}", pct(R["mechanism"][c]["wrong_rate"])); add(f"gold{k}", pct(R["mechanism"][c]["gold_recall"]))
    if "caag_sweep" in R:
        b = R["caag_sweep"]["best_pair_f1"]
        add("caagBestFOne", pct(b["pair_f1"])); add("caagBestRate", pct(b["pair_flag_rate"], nd=0))
        add("caagBestCleanFpr", pct(b["q_fpr"], nd=0)); add("caagBestThS", f"{b['theta_s']:.2f}")
        add("caagBestThC", f"{b['theta_c']:.2f}")
    gcells = R.get("grid_cells", {})
    for mode, mk in (("flag_directional", "Dir"), ("remove_suspect", "Rem")):
        for pp, pk in (("1", "One"), ("0.75", "Tqu"), ("0.5", "Half"), ("0.25", "Qu")):
            for dd, dk in (("1", "DOne"), ("0.75", "DTqu"), ("0.5", "DHalf")):
                for sub, sk in (("all", "All"), ("conflict", "Conf"), ("clean", "Clean")):
                    v = gcells.get(f"{mode}|{pp}|1|{dd}|{sub}")
                    if v:
                        add(f"g{mk}{pk}{dk}{sk}", pct(v["diff"], True, 0))
    from .costs import total_spend
    add("totalSpend", f"\\${total_spend():.2f}")
    # unsigned variants (\\a<name>) for use after verbs: "lowers accuracy by 6.7 points"
    import re as _re
    extra = []
    for line in m:
        mm = _re.match(r"\\newcommand\{\\([A-Za-z]+)\}\{\$([-+])\$(.*)\}\n", line)
        if mm and not mm.group(1).endswith("CI"):
            extra.append(macro("a" + mm.group(1), mm.group(3)))
    (out / "numbers.tex").write_text("".join(m + extra))


def write_all(R, out: Path):
    tab_detectors(R, out); tab_main(R, out); tab_gpt41(R, out); tab_wiki(R, out); tab_hypotheses(R, out)
    numbers(R, out)
    app_prompts(out)
    tab_appendix(R, out)
    fig_tradeoff(R, out); fig_placement(R, out)
    grid = {}
    if "grid_cells" in R:
        for key, v in R["grid_cells"].items():
            mode, p, r, d, s = key.split("|")
            grid[(mode, float(p), float(r), float(d), s)] = v
        fig_grid(R, out, grid)


def app_prompts(out: Path):
    """Verbatim prompts for the appendix, taken from the code that ran."""
    from .detectors import LLM_DETECT_DIRECTIONAL_PROMPT
    from .prompts import BASE, INSTRUCTION, INSTRUCTION_V2, build_prompt
    from .detectors import Detection
    ex = build_prompt("Who founded X?", ["<passage 0>", "<passage 1>", "<passage 2>"], "flag_directional",
                      [Detection(0, 2, 1.0, "contradiction", [2])])[0]
    def vb(title, text):
        import textwrap
        wrapped = "\n".join(textwrap.fill(l, 52) if l.strip() else "" for l in text.splitlines())
        return f"\\paragraph{{{title}}}\n\\begin{{footnotesize}}\\begin{{verbatim}}\n{wrapped}\n\\end{{verbatim}}\\end{{footnotesize}}\n"
    (out / "app_prompts.tex").write_text(
        vb("Generation skeleton.", BASE) + vb("Instruction v1.", INSTRUCTION) + vb("Instruction v2.", INSTRUCTION_V2)
        + vb("Example: directional flag (passage 2 suspect).", ex)
        + vb("Directional LLM detector.", LLM_DETECT_DIRECTIONAL_PROMPT))


def tab_appendix(R, out: Path):
    """Appendix tables: label robustness, CAAG sweep and ablation, CONFLICTS."""
    parts = []
    if "contains" in R:
        c = R["contains"]
        rows = []
        for h in ("H2", "H3"):
            for k, v in c[h][PRIMARY].items():
                vb = c[h][ROBUST][k]
                rows.append(f"{h} & {k.replace('_', chr(92)+'_').replace('<', '$<$').replace('>', '$>$')} & "
                            f"{pct(v['diff'], True)} [{pct(v['ci_low'], True)}, {pct(v['ci_high'], True)}] & {_p(v['p_holm'])} & "
                            f"{pct(vb['diff'], True)} & {_p(vb['p_holm'])} \\\\")
        if "H4" in c:
            h4 = c["H4"][PRIMARY]
            rows.append(f"H4 & false-flag cells harmful on clean q.; Spearman $\\rho$ & {h4['n_harmful']}/{h4['n_cells']}; "
                        f"$\\rho={h4['dose_response']['spearman_rho']:+.2f}$ & {_p(h4['dose_response']['p_perm'])} & & \\\\")
        parts.append(rf"""\begin{{table*}}[t]
\centering\small
\begin{{tabular}}{{llrlrl}}
\toprule
& Test (labels: \emph{{contains}}) & One-way [95\% CI] & $p_{{\text{{Holm}}}}$ & Bidir. & $p_{{\text{{Holm}}}}$ \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\caption{{Label robustness: pre-registered tests with every suspect document treated as misinformation
({c['n']['conflict']} conflict and {c['n']['clean']} clean hold-out questions).}}
\label{{tab:contains}}
\end{{table*}}
""")
    if "caag_sweep" in R:
        sw = R["caag_sweep"]; ab = R["caag_ablation"]
        def row(name, r):
            return (f"{name} & {r['theta_s']:.2f} & {r['theta_c']:.2f} & {pct(r['pair_precision'])} & {pct(r['pair_recall'])} & "
                    f"{pct(r['pair_f1'])} & {pct(r['q_fpr'])} \\\\")
        abl = "\n".join(f"{k.replace('caag_', '').replace('_only', ' only').replace('caag', 'all signals')} & "
                        f"{pct(v['pair_f1'])} & {pct(v['pair_flag_rate'])} & {pct(v['q_fpr'])} \\\\"
                        for k, v in ab.items())
        parts.append(rf"""\begin{{table}}[t]
\centering\small
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{lrrrrrr}}
\toprule
CAAG setting & $\theta_s$ & $\theta_c$ & P & R & F1 & Clean FPR \\
\midrule
{row('Default', sw['default'])}
{row('Best pair F1', sw['best_pair_f1'])}
{row('Best question F1', sw['best_q_f1'])}
\bottomrule
\end{{tabular}}

\vspace{{4pt}}
\begin{{tabular}}{{lrrr}}
\toprule
Signals (defaults) & Pair F1 & Flag rate & Clean FPR \\
\midrule
{abl}
\bottomrule
\end{{tabular}}
\caption{{CAAG on the RAMDocs hold-out (\%). Top: thresholds from a sweep over {sw['n_settings']} settings
($\theta_s\in[0.40,0.90]$, $\theta_c\in[0.10,0.60]$). Bottom: each signal alone.}}
\label{{tab:caag}}
\end{{table}}
""")
    if "conflicts" in R:
        cp = R["conflicts_plain"]
        rows = [f"No flags (plain) & {pct(cp['all'])} & {pct(cp['conflict'])} & {pct(cp['clean'])} \\\\", "\\midrule"]
        for c, d in R["conflicts"].items():
            rows.append(f"{COND_NAME[c]} & {cell(d['all'], 'p')} & {cell(d['conflict'], 'p')} & {cell(d['clean'], 'p')} \\\\")
        det = "; ".join(f"{DET_NAME[n]}: flags {pct(v['q_recall'])}\\% of conflict and {pct(v['clean_fpr'])}\\% of clean questions"
                        for n, v in R.get("conflicts_detect", {}).items())
        parts.append(rf"""\begin{{table}}[t]
\centering\small
\begin{{tabular}}{{lrrr}}
\toprule
Condition & All & Conflict & No conflict \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\caption{{CONFLICTS, 237 automatically scorable questions (69 conflict), gpt-4o-mini, one seed; exploratory.
Contains-match accuracy (\%) and differences in points; no difference is significant (bold would mark raw $p<0.05$).
{det}.}}
\label{{tab:conflicts}}
\end{{table}}
""")
    (out / "app_tables.tex").write_text("\n".join(parts))
