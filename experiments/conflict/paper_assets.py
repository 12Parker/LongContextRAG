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
    if "sig" in t:   # the paper-wide scheme (revision_stats.assign_significance)
        return f"\\textbf{{{s}}}" if t["sig"] else s
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


def _with_labelled(R, table: dict, gen: str, m: str) -> dict:
    lab = R.get("rev", {}).get("labelled", {}).get(gen, {}).get(m, {})
    return {c: {**d, "lab": lab[c]["labelled"]} if c in lab else dict(d) for c, d in table.items()}


def tab_main(R, out: Path):
    for m, tag in ((PRIMARY, ""), (ROBUST, "_robust")):
        t = _holm_table(_with_labelled(R, R["main_table"][m], "gpt-4o-mini", m))
        base = R["plain_means"][m]
        groups = [("Prompt only", ["instruct", "instruct_v2"]),
                  ("Oracle detector", ["oracle+flag", "oracle+flag_directional", "oracle+remove_suspect"]),
                  ("Real detectors (exploratory)", ["random+flag", "caag+flag", "caag+remove", "nli+flag", "nli+remove",
                                      "llm_dir+flag", "llm_dir+flag_directional", "llm_dir+remove_suspect"]),
                  ("Reference", ["madam"])]
        lab_base = R.get("rev", {}).get("plain_labelled", {}).get(m)
        lines = [f"No flags (plain) & {pct(base['all'])} & {pct(lab_base) if lab_base is not None else '--'} & "
                 f"{pct(base['conflict'])} & {pct(base['clean'])} & "
                 f"{R['main_costs']['plain']['prompt_tokens']:.0f} \\\\", "\\midrule"]
        for gname, cs in groups:
            lines.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{gname}}}}} \\\\")
            for c in cs:
                if c not in t:
                    continue
                lines.append(f"\\quad {COND_NAME[c]} & {cell(t[c]['all'], 'p_table_holm')} & "
                             f"{cell(t[c].get('lab'), 'p_table_holm')} & "
                             f"{cell(t[c]['conflict'], 'p_table_holm')} & {cell(t[c]['clean'], 'p_table_holm')} & "
                             f"{R['main_costs'][c]['prompt_tokens']:.0f} \\\\")
        metric_name = "one-way strict accuracy (primary)" if m == PRIMARY else "bidirectional strict accuracy (robustness)"
        (out / f"tab_main{tag}.tex").write_text(rf"""\begin{{table}}[t]
\centering\small
\setlength{{\tabcolsep}}{{2.2pt}}
\resizebox{{\columnwidth}}{{!}}{{%
\begin{{tabular}}{{lrrrrr}}
\toprule
Condition & All & Lab. & Conflict & Clean & Tokens \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}}}
\caption{{RAMDocs hold-out, gpt-4o-mini, 3 seeds, {metric_name}. \emph{{All}}: 300 questions, including 35 whose
conflict status the label rule leaves open; \emph{{Lab.}}: the 265 labelled ones (139 conflict, 126 clean).
First row: accuracy (\%) with no flags. Other rows: paired difference from it in points; \textbf{{bold}} = significant
after Holm correction across the table (sign-flip test, $p<0.05$). Real-detector rows were not pre-registered as
tests. Tokens: mean prompt tokens per call (MADAM-RAG: per question, all calls).}}
\label{{tab:main{tag}}}
\end{{table}}
""")


def tab_gpt41(R, out: Path):
    if "gpt41_table" not in R:
        return
    t = _holm_table(R["gpt41_table"][PRIMARY])
    b = R["gpt41_plain"][PRIMARY]
    lb = R.get("rev", {}).get("plain_labelled_gpt41")
    lines = [f"No flags (plain) & {pct(b['all'])} & {pct(lb) if lb is not None else '--'} & {pct(b['conflict'])} & "
             f"{pct(b['clean'])} \\\\", "\\midrule"]
    lab = R.get("rev", {}).get("labelled", {}).get("gpt-4.1", {}).get(PRIMARY, {})
    for c in t:
        lines.append(f"{COND_NAME[c]} & {cell(t[c]['all'])} & {cell(lab[c]['labelled']) if c in lab else '--'} & "
                     f"{cell(t[c]['conflict'])} & {cell(t[c]['clean'])} \\\\")
    (out / "tab_gpt41.tex").write_text(rf"""\begin{{table}}[t]
\centering\footnotesize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{lrrrr}}
\toprule
Condition & All & Lab. & Conflict & Clean \\
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


def _esc(k):
    return k.replace("_", "\\_").replace(">", "$>$").replace("<", "$<$")


def _est(v):
    return f"{pct(v['diff'], True)} [{pct(v['ci_low'], True)}, {pct(v['ci_high'], True)}]"


def tab_hypotheses(R, out: Path):
    """Every pre-registered test (§4, §8, §9). Robust: the same outcome under the bidirectional metric.
    Rows marked (r) are re-analyses of existing data registered in §9.3."""
    two_sided = {"H5", "H7"}
    rev = R.get("rev", {})

    def outcome(v, h, k=""):
        if h in two_sided or (h == "H10" and "oracle+flag vs" in k) or (h == "H8" and not k.startswith("oracle+flag")):
            return "significant" if v["supported"] else "n.s."
        return "supported" if v["supported"] else "not supported"

    def rob(fam, k, h):
        if ROBUST not in fam:
            return "--"
        return "yes" if outcome(fam[ROBUST][k], h, k) == outcome(fam[PRIMARY][k], h, k) else "no"

    rows = []
    h1 = R["H1"]["tests"]
    for k, v in h1.items():
        if k.startswith("direction") and "direction_ci" in rev:
            d = rev["direction_ci"]
            est = f"{pct(d['estimate'])}\\% [{pct(d['ci_low'])}, {pct(d['ci_high'])}]"
            rows.append(("H1", "direction accuracy llm\\_dir $>$ 50\\% (cluster CI\\textsuperscript{r})", est,
                         _p(v["p_holm"]), "supported" if d["ci_low"] > 0.5 else "not supported", "--"))
        else:
            rows.append(("H1", _esc(k), _est(v), _p(v["p_holm"]), outcome(v, "H1"), "--"))
    for h in ("H2", "H3"):
        for k, v in R[h][PRIMARY].items():
            rows.append((h, _esc(k), _est(v), _p(v["p_holm"]), outcome(v, h), rob(R[h], k, h)))
    if "H4" in rev:
        h4p, h4r = rev["H4"][PRIMARY], rev["H4"][ROBUST]
        for mode, name in (("flag_directional", "directional flag"), ("remove_suspect", "removal")):
            a, b = h4p["by_mode"][mode], h4r["by_mode"][mode]
            rows.append(("H4a", f"false-flag cells below plain on clean q., {name}\\textsuperscript{{r}}",
                         f"{a['n_harmful']} of {a['n']} cells", "Holm", "supported" if a["n_harmful"] else "not supported",
                         "yes" if bool(a["n_harmful"]) == bool(b["n_harmful"]) else "no"))
        dr, drr = h4p["dose_response"], h4r["dose_response"]
        rows.append(("H4b", f"harm grows with clean-q.\\ exposure, Spearman over {dr['n_cells']} cells\\textsuperscript{{r}}",
                     f"$\\rho={dr['spearman_rho']:+.2f}$ [${dr['ci_low']:+.2f}$, ${dr['ci_high']:+.2f}$]", "--",
                     "supported" if dr["ci_high"] < 0 else "not supported", "yes" if (drr["ci_high"] < 0) == (dr["ci_high"] < 0) else "no"))
        rv = h4p["remove_vs_flag"]
        rows.append(("H4c", "removal harms clean q.\\ more than directional flag\\textsuperscript{r}",
                     f"{rv['n_remove_worse']} of {rv['n']} matched cells", "Holm", "count only", "--"))
    for h in ("H5",):
        for k, v in R[h][PRIMARY].items():
            rows.append((h, _esc(k), _est(v), _p(v["p_holm"]), outcome(v, h), rob(R[h], k, h)))
    if "H5_tost" in rev:
        t = rev["H5_tost"][PRIMARY]
        rows.append(("H5", f"equivalence within $\\pm${100*t['margin']:.0f} points (TOST)\\textsuperscript{{r}}",
                     f"90\\% CI [{pct(t['ci90_low'], True)}, {pct(t['ci90_high'], True)}]", "--",
                     "equivalent" if t["equivalent"] else "not equivalent",
                     "yes" if rev["H5_tost"][ROBUST]["equivalent"] == t["equivalent"] else "no"))
    if "H6" in rev:
        s6, s6r = rev["H6"][PRIMARY]["summary"], rev["H6"][ROBUST]["summary"]
        rows.append(("H6", "actual effect inside the 95\\% interval of the document-level prediction\\textsuperscript{r}",
                     f"{s6['doc']['inside']} of {s6['doc']['n']} (pair level: {s6['pair']['inside']} of {s6['pair']['n']})",
                     "--", "descriptive", "--"))
        d, dr_ = s6["doc_minus_pair_mae"], s6r["doc_minus_pair_mae"]
        rows.append(("H6", "document-level error $<$ pair-level error (paired, exact sign-flip)\\textsuperscript{r}",
                     f"{pct(d['diff'], True)} [{pct(d['ci_low'], True)}, {pct(d['ci_high'], True)}]",
                     _p(d["p_exact_signflip"]), "supported" if d["p_exact_signflip"] < 0.05 and d["diff"] < 0 else "not supported",
                     "yes" if (dr_["p_exact_signflip"] < 0.05) == (d["p_exact_signflip"] < 0.05) else "no"))
    for h in ("H7", "H9"):
        if h in R:
            for k, v in R[h][PRIMARY].items():
                rows.append((h, _esc(k), _est(v), _p(v["p_holm"]), outcome(v, h), rob(R[h], k, h)))
    if "H8" in R:
        for k, v in R["H8"]["both_answers"].items():
            rows.append(("H8", "WikiContradict: " + _esc(k), _est(v), _p(v["p_holm"]), outcome(v, "H8", k), "--"))
    for h in ("H10", "H11", "H12", "H13", "H14"):
        if h in rev:
            for k, v in rev[h][PRIMARY].items():
                two = h == "H14" or " vs " in k
                o = ("significant" if v["supported"] else "n.s.") if two else ("supported" if v["supported"] else "not supported")
                ob = rev[h][ROBUST][k]["supported"] == v["supported"]
                rows.append((h, _esc(k), _est(v), _p(v["p_holm"]), o, "yes" if ob else "no"))
    body = "\n".join(" & ".join(r) + " \\\\" for r in rows)
    (out / "tab_hypotheses.tex").write_text(rf"""\begin{{table*}}[t]
\centering\footnotesize
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{lp{{6.3cm}}llll}}
\toprule
& Pre-registered test & Estimate [95\% CI] & $p_{{\text{{Holm}}}}$ & Outcome & Robust \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\caption{{Every pre-registered test on the confirmatory split (one-way strict accuracy; differences in points). Holm
correction within each family. H1--H9 were registered before the confirmatory run; H10, H11 and the rows marked
\textsuperscript{{r}} after a first review of the draft, and H12--H14 after a second one (each before its runs;
\textsuperscript{{r}} rows are re-analyses of data that already existed). Directional predictions are \emph{{supported}} or \emph{{not
supported}}; tests with no predicted sign are \emph{{significant}} or \emph{{n.s.}} \emph{{Robust}}: the same outcome under
the bidirectional metric.}}
\label{{tab:hypotheses}}
\end{{table*}}
""")


BASELINE_NAME = {"astute": "Astute RAG", "astute_multi": "Astute RAG (multi)", "trustrag": "TrustRAG",
                 "trustrag_multi": "TrustRAG (multi)", "madam": "MADAM-RAG",
                 "llm_dir+flag_directional": "LLM detector, dir.\\ flag",
                 "llm_dir+flag_directional_instruct_v2": "LLM detector, dir.\\ flag + instr.",
                 "llm_dir+flag_instruct_v2": "LLM detector, flag + instr.",
                 "llm_dir+remove_suspect": "LLM detector, remove suspect",
                 "instruct_v2": "Instruction (v2) only",
                 "oracle+flag_instruct_v2": "Oracle, flag + instr.",
                 "oracle+flag_directional_instruct_v2": "Oracle, dir.\\ flag + instr."}


def tab_baselines(R, out: Path):
    """Our directional interface against published conflict-robust RAG methods, plus the
    resolution-instruction conditions (§10)."""
    r2 = R.get("rev", {}).get("r2", {})
    if "gpt-4o-mini" not in r2:
        return
    mt = R["main_table"][PRIMARY]; lab = R["rev"]["labelled"]["gpt-4o-mini"][PRIMARY]
    a, b, cb = r2.get("gpt-4o-mini"), r2.get("gpt-4.1|seed0"), r2.get("CB:gpt-4o-mini")
    rows_ = ["instruct_v2", "oracle+flag_instruct_v2", "oracle+flag_directional_instruct_v2", "madam", "astute",
             "astute_multi", "trustrag", "trustrag_multi", "llm_dir+flag_instruct_v2",
             "llm_dir+flag_directional_instruct_v2", "llm_dir+flag_directional", "llm_dir+remove_suspect"]

    def eff(src, c, s):
        return src["effects"][PRIMARY].get(c, {}).get(s) if src else None

    def ramdocs_mini(c, s):
        if c in mt:   # registered / main-table conditions
            return lab[c]["labelled"] if s == "labelled" else mt[c][s]
        return eff(a, c, s)

    def calls(src, c):
        v = (src or {}).get("cost", {}).get(c)
        return f"{v['calls']:.0f}" if v else "--"

    lines = [f"No flags (plain), accuracy & {pct(R['rev']['plain_labelled'][PRIMARY])} & {pct(R['plain_means'][PRIMARY]['conflict'])} & "
             f"{pct(R['plain_means'][PRIMARY]['clean'])} & "
             f"{pct(b['plain'][PRIMARY]['labelled']) if b else '--'} & "
             f"{pct(cb['plain'][PRIMARY]['all']) if cb else '--'} & {pct(cb['plain'][PRIMARY]['conflict']) if cb else '--'} & "
             f"{pct(cb['plain'][PRIMARY]['clean']) if cb else '--'} & 1 \\\\", "\\midrule"]
    for c in rows_:
        cbc = {"astute_multi": None, "trustrag_multi": None}.get(c, c)
        cells = [cell(ramdocs_mini(c, s)) for s in ("labelled", "conflict", "clean")]
        cells.append(cell(eff(b, c, "labelled")))
        cells += [cell(eff(cb, cbc, s)) if cbc else "--" for s in ("labelled", "conflict", "clean")]
        n_calls = {"madam": "--", "astute": "2", "astute_multi": "2", "trustrag": "3", "trustrag_multi": "3"}.get(c, "1")
        if c == "madam":
            n_calls = f"{R['main_costs']['madam']['prompt_tokens'] / R['main_costs']['plain']['prompt_tokens']:.0f}$\\times$tok."
        lines.append(f"{BASELINE_NAME.get(c, COND_NAME.get(c, c))} & " + " & ".join(cells) + f" & {n_calls} \\\\")
        if c in ("oracle+flag_directional_instruct_v2", "trustrag_multi"):
            lines.append("\\midrule")
    (out / "tab_baselines.tex").write_text(rf"""\begin{{table*}}[t]
\centering\small
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{lrrrrrrrr}}
\toprule
& \multicolumn{{3}}{{c}}{{RAMDocs, gpt-4o-mini}} & RAMDocs, gpt-4.1 & \multicolumn{{3}}{{c}}{{ConflictBank, gpt-4o-mini}} & \\
\cmidrule(lr){{2-4}}\cmidrule(lr){{5-5}}\cmidrule(lr){{6-8}}
Method & Lab. & Conflict & Clean & Lab. & All & Conflict & Clean & Calls \\
\midrule
{{chr(10).join(lines)}}
\bottomrule
\end{{tabular}}
\caption{{Resolution instructions and published baselines. First row: accuracy (\%, one-way strict) with no flags; other rows:
paired difference from it in points. \emph{{Lab.}}: the labelled RAMDocs questions; ConflictBank questions are all
labelled. RAMDocs gpt-4o-mini and ConflictBank: 3 seeds; RAMDocs gpt-4.1: seed 0. \emph{{instr.}}: the v2 reliability
instruction. \emph{{(multi)}}: the baseline's final step asks for every valid answer, which RAMDocs needs (every
ConflictBank question has one answer). Calls: generator calls per question (LLM-detector rows add one detector call).
\textbf{{Bold}}: significant under the paper's single scheme (Section~\ref{{sec:setup}}).}}
\label{{tab:baselines}}
\end{{table*}}
""".replace("{chr(10).join(lines)}", chr(10).join(lines)))


def tab_cb(R, out: Path):
    """ConflictBank: the main effects for three generators (H12)."""
    r2 = R.get("rev", {}).get("r2", {})
    keys = [k for k in ("CB:gpt-4o-mini", "CB:gpt-4.1", "CB:Qwen2.5-7B") if k in r2]
    if not keys:
        return
    cols = [("oracle+flag", "conflict"), ("oracle+flag_directional", "conflict"), ("oracle+remove_suspect", "conflict"),
            ("llm_dir+flag", "conflict"), ("llm_dir+flag_directional", "conflict"), ("llm_dir+flag_directional", "clean"),
            ("llm_dir+remove_suspect", "conflict"), ("llm_dir+remove_suspect", "clean")]
    lines = []
    for k in keys:
        v = r2[k]
        seeds = "3" if k == "CB:gpt-4o-mini" else "1"
        lines.append(f"{k[3:]} & {seeds} & {pct(v['plain'][PRIMARY]['conflict'])} & " +
                     " & ".join(cell(v["effects"][PRIMARY].get(c, {}).get(s)) for c, s in cols) + " \\\\")
    (out / "tab_cb.tex").write_text(rf"""\begin{{table*}}[t]
\centering\small
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{lrrrrrrrrrr}}
\toprule
& & Plain & \multicolumn{{3}}{{c}}{{Oracle, conflict q.}} & \multicolumn{{3}}{{c}}{{LLM detector, flag / dir.\ flag}} & \multicolumn{{2}}{{c}}{{LLM, remove}} \\
\cmidrule(lr){{4-6}}\cmidrule(lr){{7-9}}\cmidrule(lr){{10-11}}
Generator & Seeds & conflict & flag & dir.\ flag & remove & flag & dir.\ conf. & dir.\ clean & conflict & clean \\
\midrule
LINES
\bottomrule
\end{{tabular}}
\caption{{ConflictBank (300 conflict, 100 clean questions; one correct answer each; LLM-written misinformation built from
Wikidata facts), default skeleton. \emph{{Plain}}: accuracy (\%) on conflict questions with no flags; other columns:
paired difference in points. The LLM detector is gpt-4o-mini for every generator. \textbf{{Bold}}: significant under the
paper's single scheme.}}
\label{{tab:cb}}
\end{{table*}}
""".replace("LINES", chr(10).join(lines)))


def tab_generality(R, out: Path):
    """Generator x skeleton: oracle effects on conflict questions and the real detector's effects."""
    g = R.get("rev", {}).get("generality")
    if not g:
        return
    from .stats import holm
    cols = [("oracle+flag", "conflict"), ("oracle+flag_directional", "conflict"), ("oracle+remove_suspect", "conflict"),
            ("llm_dir+flag_directional", "conflict"), ("llm_dir+flag_directional", "clean"),
            ("llm_dir+remove_suspect", "conflict"), ("llm_dir+remove_suspect", "clean")]
    keys = {f"{gk}|{c}|{s}": v["effects"][PRIMARY][c][s]["p"] for gk, v in g.items() for c, s in cols
            if c in v["effects"][PRIMARY]}
    adj = holm(keys)
    lines = []
    order = [f"{gen}|{sk}" for gen in ("gpt-4o-mini", "gpt-4.1", "Qwen2.5-7B") for sk in ("default", "minimal")]
    for gk in order:
        if gk not in g:
            continue
        v = g[gk]; gen, sk = gk.split("|")
        cells = []
        for c, s in cols:
            t = v["effects"][PRIMARY].get(c, {}).get(s)
            cells.append(cell({**t, "p_holm": adj[f"{gk}|{c}|{s}"]}) if t else "--")
        lines.append(f"{gen} & {sk} & {pct(v['plain'][PRIMARY]['conflict'])} & " + " & ".join(cells) + " \\\\")
    (out / "tab_generality.tex").write_text(rf"""\begin{{table*}}[t]
\centering\small
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{llrrrrrrrr}}
\toprule
& & Plain & \multicolumn{{3}}{{c}}{{Oracle, conflict q.}} & \multicolumn{{2}}{{c}}{{LLM, directional flag}} & \multicolumn{{2}}{{c}}{{LLM, remove suspect}} \\
\cmidrule(lr){{4-6}}\cmidrule(lr){{7-8}}\cmidrule(lr){{9-10}}
Generator & Skeleton & conflict & flag & dir.\ flag & remove & conflict & clean & conflict & clean \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}
\caption{{The main effects across three generators and two prompt skeletons (RAMDocs hold-out, 3 seeds, one-way strict
accuracy). \emph{{Plain}}: accuracy (\%) on conflict questions with no flags. Other columns: paired difference from plain
in points. \emph{{Default}} skeleton tells the model to give every valid answer; \emph{{minimal}} drops that policy
(Appendix~\ref{{app:prompts}}). \textbf{{Bold}}: significant after Holm across the table. The LLM detector is gpt-4o-mini
for every generator. Qwen2.5-7B-Instruct runs locally (4-bit).}}
\label{{tab:generality}}
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
    fig, ax = plt.subplots(figsize=(3.3, 3.5))
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
    # legends below the plot, so they never cover points or labels
    fig.legend(handles=dets + fmts, fontsize=6.2, loc="lower center", ncol=3, handletextpad=0.2,
               columnspacing=0.8, borderaxespad=0.1)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
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
                        ax.text(b, a, txt, ha="center", va="center", fontsize=7.5,
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
    """Predicted (with the pre-registered bootstrap prediction interval, x) vs actual (95% CI, y)."""
    import matplotlib.pyplot as plt
    _style()
    act = R["H6"][PRIMARY]["cases"]
    pred = R.get("rev", {}).get("H6", {}).get(PRIMARY, {}).get("cases", {})
    fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.5), sharex=True, sharey=True)
    for ax, space, title in ((axes[0], "doc", "Document-level placement"), (axes[1], "pair", "Pair-level placement")):
        lo = -32; hi = 28
        ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.8, ls="--")
        for k, v in act.items():
            cond, sub, sp = k.split("|")
            if sp != space:
                continue
            det = cond.split("+")[0]
            pv = pred.get(k, {})
            x = 100 * pv.get("predicted", v["predicted"])
            xerr = ([[x - 100 * pv["pred_low"]], [100 * pv["pred_high"] - x]] if pv else None)
            ax.errorbar(x, 100 * v["actual"], xerr=xerr,
                        yerr=[[100 * (v["actual"] - v["ci_low"])], [100 * (v["ci_high"] - v["actual"])]],
                        fmt="none", ecolor=CAT[det], elinewidth=0.6, alpha=0.7)
            ax.scatter(x, 100 * v["actual"], color=CAT[det],
                       marker="o" if sub == "conflict" else "s", s=24, edgecolor="white", linewidth=1.0, zorder=3)
        ax.set_title(title, fontsize=8.5, color=INK); ax.set_xlabel("Predicted change (points)", fontsize=8.5)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.tick_params(labelsize=7.5)
    axes[0].set_ylabel("Actual change (points)", fontsize=8.5)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=CAT[d], label=DET_NAME[d]) for d in ("caag", "nli", "llm_dir")] + \
        [Line2D([], [], marker="o", ls="", color=INK2, label="conflict q."),
         Line2D([], [], marker="s", ls="", color=INK2, label="clean q.")]
    axes[1].legend(handles=h, fontsize=6.5, loc="upper left", handletextpad=0.2)
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
    tr = R["main_table"][ROBUST]
    for k, c in (("LlmDir", "llm_dir+flag_directional"), ("LlmRem", "llm_dir+remove_suspect")):
        for sb, sn in (("all", "All"), ("clean", "Clean"), ("conflict", "Conf")):
            add(f"r{k}{sn}", pct(tr[c][sb]["diff"], True))
    tt = _holm_table({c: dict(d) for c, d in R["main_table"][PRIMARY].items()})
    add("pLlmDirConfRaw", f"{tt['llm_dir+flag_directional']['conflict']['p']:.3f}")
    add("pLlmDirConfHolm", f"{tt['llm_dir+flag_directional']['conflict']['p_table_holm']:.2f}")
    q = R["H1"]["quality"]
    for n, k in (("caag", "Caag"), ("nli", "Nli"), ("llm_dir", "Llm")):
        add(f"fOne{k}", pct(q[n]["pair_f1"])); add(f"cleanFpr{k}", pct(q[n]["clean_fpr"]))
        add(f"misGone{k}", pct(q[n]["doc_remove"]["conflict_q_all_misinfo_marked"]))
        add(f"corLost{k}", pct(q[n]["doc_remove"]["clean_q_any_correct_marked"]))
    add("dirAccLlm", pct(q["llm_dir"]["direction_accuracy"]))
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
                     ("llm_dir+flag_directional", "LlmDir"), ("instruct_v2", "InstrTwo")):
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
    rev = R.get("rev", {})
    if "H4" in rev:
        h = rev["H4"][PRIMARY]
        add("hFourDirHarm", h["by_mode"]["flag_directional"]["n_harmful"]); add("hFourDirN", h["by_mode"]["flag_directional"]["n"])
        add("hFourRemHarm", h["by_mode"]["remove_suspect"]["n_harmful"]); add("hFourRemN", h["by_mode"]["remove_suspect"]["n"])
        dr = h["dose_response"]
        add("hFourRho", f"{dr['spearman_rho']:+.2f}".replace("-", "$-$"))
        add("hFourRhoCI", f"[{dr['ci_low']:+.2f}, {dr['ci_high']:+.2f}]".replace("-", "$-$").replace("+", "$+$"))
        add("hFourRhoCells", dr["n_cells"])
        add("hFourRemWorse", h["remove_vs_flag"]["n_remove_worse"]); add("hFourMatched", h["remove_vs_flag"]["n"])
    if "H6" in rev:
        s6 = rev["H6"][PRIMARY]["summary"]
        add("hSixDocIn", s6["doc"]["inside"]); add("hSixPairIn", s6["pair"]["inside"]); add("hSixN", s6["doc"]["n"])
        add("hSixDocMae", pct(s6["doc"]["mae"])); add("hSixPairMae", pct(s6["pair"]["mae"]))
        d = s6["doc_minus_pair_mae"]
        add("hSixMaeDiff", pct(d["diff"], True)); add("hSixMaeDiffCI", f"[{pct(d['ci_low'], True)}, {pct(d['ci_high'], True)}]")
        add("hSixMaeP", f"{d['p_exact_signflip']:.2f}"); add("hSixDocBetter", d["n_doc_better"])
    if "H5_tost" in rev:
        t5 = rev["H5_tost"][PRIMARY]
        add("tostLo", pct(t5["ci90_low"], True)); add("tostHi", pct(t5["ci90_high"], True))
        add("tostMargin", f"{100*t5['margin']:.0f}")
    if "direction_ci" in rev:
        dc = rev["direction_ci"]
        add("dirAccLlmCI", f"[{pct(dc['ci_low'])}, {pct(dc['ci_high'])}]")
    if "mde_conflict" in rev:
        add("mdeLo", pct(rev["mde_conflict"]["min"], nd=0)); add("mdeHi", pct(rev["mde_conflict"]["max"], nd=0))
        add("mdeMed", pct(rev["mde_conflict"]["median"]))
    if "plain_labelled" in rev:
        add("plainLab", pct(rev["plain_labelled"][PRIMARY]))
    for gen, gk in (("gpt-4o-mini", ""), ("gpt-4.1", "gpt")):
        lab = rev.get("labelled", {}).get(gen, {}).get(PRIMARY, {})
        for c, k in (("llm_dir+flag_directional", "LlmDir"), ("llm_dir+remove_suspect", "LlmRem"),
                     ("oracle+flag_directional", "OracleDir"), ("oracle+remove_suspect", "OracleRem")):
            if c in lab:
                add(f"{gk}lab{k}", pct(lab[c]["labelled"]["diff"], True))
                add(f"{gk}lab{k}CI", f"[{pct(lab[c]['labelled']['ci_low'], True)}, {pct(lab[c]['labelled']['ci_high'], True)}]")
                add(f"{gk}excl{k}", pct(lab[c]["excluded"]["diff"], True))
    # generator x skeleton
    tag = {"gpt-4o-mini|default": "Mini", "gpt-4o-mini|minimal": "MiniMin", "gpt-4.1|default": "Four",
           "gpt-4.1|minimal": "FourMin", "Qwen2.5-7B|default": "Qwen", "Qwen2.5-7B|minimal": "QwenMin"}
    for gk, v in rev.get("generality", {}).items():
        T = tag[gk]
        add(f"nAborted{T}", v["n_aborted"]); add(f"nRows{T}", v["n_rows"])
        add(f"plain{T}Conf", pct(v["plain"][PRIMARY]["conflict"])); add(f"plain{T}Clean", pct(v["plain"][PRIMARY]["clean"]))
        for c, k in (("oracle+flag", "OracleFlag"), ("oracle+flag_directional", "OracleDir"),
                     ("oracle+remove_suspect", "OracleRem"), ("llm_dir+flag", "LlmFlag"),
                     ("llm_dir+flag_directional", "LlmDir"), ("llm_dir+remove_suspect", "LlmRem")):
            e = v["effects"][PRIMARY].get(c)
            if not e:
                continue
            for s_, sn in (("conflict", "Conf"), ("clean", "Clean"), ("labelled", "Lab")):
                add(f"x{T}{k}{sn}", pct(e[s_]["diff"], True))
                add(f"x{T}{k}{sn}CI", f"[{pct(e[s_]['ci_low'], True)}, {pct(e[s_]['ci_high'], True)}]")
        for c, k in (("plain", "Plain"), ("oracle+flag", "OracleFlag"), ("oracle+flag_directional", "OracleDir")):
            if c in v["mechanism"]:
                add(f"wrong{T}{k}", pct(v["mechanism"][c]["wrong_rate"])); add(f"gold{T}{k}", pct(v["mechanism"][c]["gold_recall"]))
    if "by_answers" in rev:
        add("nConfMulti", rev["n_conflict_multi"]); add("nConfSingle", rev["n_conflict_single"])
        for key, v in rev["by_answers"].items():
            gk, name = key.rsplit("|", 1)
            T = tag[gk] + ("Single" if name == "single" else "Multi")
            add(f"plain{T}", pct(v["plain"]))
            for c, k in (("oracle+flag", "OracleFlag"), ("oracle+flag_directional", "OracleDir"),
                         ("oracle+remove_suspect", "OracleRem")):
                if c in v:
                    add(f"y{T}{k}", pct(v[c]["diff"], True))
                    add(f"y{T}{k}CI", f"[{pct(v[c]['ci_low'], True)}, {pct(v[c]['ci_high'], True)}]")
    for gk, t in rev.get("real_direction", {}).get(PRIMARY, {}).items():
        T = tag[gk]
        add(f"rd{T}", pct(t["diff"], True)); add(f"rd{T}CI", f"[{pct(t['ci_low'], True)}, {pct(t['ci_high'], True)}]")
        add(f"rd{T}P", f"{t['p']:.3f}" if t["p"] >= 0.001 else "<0.001")
    for c, k in (("oracle_inv+flag_directional", "Dir"), ("oracle_inv+remove_suspect", "Rem")):
        if c in rev.get("adversarial", {}):
            for sb, sn in (("conflict", "Conf"), ("clean", "Clean")):
                v = rev["adversarial"][c][sb]
                add(f"adv{k}{sn}", pct(v["diff"], True)); add(f"adv{k}{sn}CI", f"[{pct(v['ci_low'], True)}, {pct(v['ci_high'], True)}]")
            add(f"wrongAdv{k}", pct(R["rev"]["adversarial_wrong"][c]) if "adversarial_wrong" in rev else "--")
    for h in ("H10", "H11"):
        if h in rev:
            fam = rev[h][PRIMARY]
            hn = {"H10": "HTen", "H11": "HEleven"}[h]
            add(f"n{hn}Supported", sum(v["supported"] for v in fam.values())); add(f"n{hn}Tests", len(fam))
    # second revision (§10)
    import re as _re2
    T2 = {"gpt-4o-mini": "Mini", "gpt-4.1|seed0": "Four", "gpt-4o-mini|qwen-detector": "QDet",
          "CB:gpt-4o-mini": "CbMini", "CB:gpt-4.1": "CbFour", "CB:Qwen2.5-7B": "CbQwen"}
    cname = lambda c: "".join(w.capitalize() for w in _re2.split(r"[^a-z0-9]+", c.replace("_v2", "_two")) if w and not w.isdigit())
    for gk, v in rev.get("r2", {}).items():
        T = T2[gk]
        for sb in ("all", "conflict", "clean", "labelled"):
            add(f"zp{T}{sb.capitalize()}", pct(v["plain"][PRIMARY][sb]))
        for c, d in v["effects"][PRIMARY].items():
            for sb, t in d.items():
                nm = f"z{T}{cname(c)}{sb.capitalize()}"
                add(nm, pct(t["diff"], True)); add(nm + "CI", f"[{pct(t['ci_low'], True)}, {pct(t['ci_high'], True)}]")
        for c, w in v.get("wrong_rate", {}).items():
            add(f"zw{T}{cname(c)}", pct(w))
    for h in ("H12", "H13", "H14"):
        if h in rev:
            hn = {"H12": "HTwelve", "H13": "HThirteen", "H14": "HFourteen"}[h]
            fam = rev[h][PRIMARY]
            add(f"n{hn}Supported", sum(v["supported"] for v in fam.values())); add(f"n{hn}Tests", len(fam))
            for n_, (k, v) in enumerate(fam.items()):
                add(f"{hn.lower()}P{chr(65 + n_)}", _p(v["p_holm"]))
    # registered round-2 tests by short name, and detector quality on the new runs
    short = {"H13": {"RAMDocs, gpt-4o-mini: dir.": "DirMini", "RAMDocs, gpt-4.1: dir.": "DirFour",
                     "ConflictBank, gpt-4o-mini: dir.": "DirCb"},
             "H14": {"RAMDocs, gpt-4o-mini: LLM dir. flag vs astute": "AstMini",
                     "RAMDocs, gpt-4o-mini: LLM dir. flag vs trustrag": "TrMini",
                     "RAMDocs, gpt-4.1: LLM dir. flag vs astute": "AstFour",
                     "RAMDocs, gpt-4.1: LLM dir. flag vs trustrag": "TrFour",
                     "ConflictBank, gpt-4o-mini: LLM dir. flag vs astute": "AstCb",
                     "ConflictBank, gpt-4o-mini: LLM dir. flag vs trustrag": "TrCb"}}
    for h, hn in (("H13", "hThirteen"), ("H14", "hFourteen")):
        for tk, t in rev.get(h, {}).get(PRIMARY, {}).items():
            for pre, nm in short[h].items():
                if tk.startswith(pre):
                    add(f"{hn}{nm}", pct(t["diff"], True))
                    add(f"{hn}{nm}CI", f"[{pct(t['ci_low'], True)}, {pct(t['ci_high'], True)}]")
                    add(f"{hn}{nm}P", _p(t["p_holm"]))
    for grp, v in rev.get("baseline_recall", {}).items():
        for c, k in (("plain", "Plain"), ("astute_multi", "Ast"), ("trustrag_multi", "Tr")):
            add(f"rec{grp.capitalize()}{k}", pct(v[c]))
    import json as _json
    from .paper_stats import RUNS as _RUNS
    for run, pre in (("cb_main", "cb"), ("qwendet", "qdet")):
        f = Path(_RUNS[run]) / "detector_quality.json"
        if f.exists():
            dq = _json.loads(f.read_text()).get("llm_dir", {})
            ql = dq.get("question_level", {})
            add(f"{pre}DirAccLlm", pct(dq["direction_accuracy"]) if dq.get("direction_accuracy") is not None else "--")
            add(f"{pre}Fpr", pct(ql.get("false_positive_rate", float("nan"))))
            add(f"{pre}Dir", pct(dq["direction_accuracy"]) if dq.get("direction_accuracy") is not None else "--")
    be = rev.get("break_even", {})
    for mode, mk in (("flag_directional", "Dir"), ("remove_suspect", "Rem")):
        v = be.get(f"{mode}|r1|d1")
        add(f"breakEven{mk}", f"{v:g}" if v else "none")
    for k, v in rev.get("prevalence", {}).items():
        gk, sk, c = k.split("|")
        T = {"gpt-4o-mini": "Mini", "gpt-4.1": "Four", "Qwen2.5-7B": "Qwen"}[gk] + ("Min" if sk == "minimal" else "")
        nm = f"prev{T}{'Dir' if 'directional' in c else 'Rem'}"
        add(nm, pct(v["break_even_pi"], nd=0) if v["break_even_pi"] is not None else "--")
    if "n_exploratory" in rev:
        add("nExploratory", rev["n_exploratory"].get(PRIMARY, 0))
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
    tab_generality(R, out); tab_baselines(R, out); tab_cb(R, out)
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
    from .prompts import BASE, BASE_MINIMAL, INSTRUCTION, INSTRUCTION_V2, build_prompt
    from .detectors import Detection
    ex = build_prompt("Who founded X?", ["<passage 0>", "<passage 1>", "<passage 2>"], "flag_directional",
                      [Detection(0, 2, 1.0, "contradiction", [2])])[0]
    def vb(title, text):
        import textwrap
        wrapped = "\n".join(textwrap.fill(l, 52) if l.strip() else "" for l in text.splitlines())
        return f"\\paragraph{{{title}}}\n\\begin{{footnotesize}}\\begin{{verbatim}}\n{wrapped}\n\\end{{verbatim}}\\end{{footnotesize}}\n"
    (out / "app_prompts.tex").write_text(
        vb("Generation skeleton (default).", BASE) + vb("Minimal skeleton (no answer policy).", BASE_MINIMAL) + vb("Instruction v1.", INSTRUCTION) + vb("Instruction v2.", INSTRUCTION_V2)
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
            h4 = c["H4"][PRIMARY]; bm = h4["by_mode"]; dr = h4["dose_response"]
            rows.append(f"H4 & harmful cells, clean q.\\ (dir.\\ flag; removal); $\\rho$ & "
                        f"{bm['flag_directional']['n_harmful']}/{bm['flag_directional']['n']}; "
                        f"{bm['remove_suspect']['n_harmful']}/{bm['remove_suspect']['n']}; "
                        f"$\\rho={dr['spearman_rho']:+.2f}$ [{dr['ci_low']:+.2f}, {dr['ci_high']:+.2f}] & & & \\\\")
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
