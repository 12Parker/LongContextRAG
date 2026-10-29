"""
Tables (LaTeX, booktabs), figures (PDF) and number macros for the paper, from the results dict
built by paper_stats.py. Colors follow the dataviz reference palette (validated: categorical slots
1-4 pass CVD and normal-vision checks on the light surface; two slots are below 3:1 contrast, so
every mark is also direct-labelled or shape-coded).
"""
from __future__ import annotations

import json
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
First row: accuracy (\%) with no flags. Other rows: paired difference from it in points; \textbf{{bold}} = significant under the paper's single scheme (Section~\ref{{sec:setup}}: a registered test's Holm-adjusted $p$ within its family, otherwise Holm over all exploratory comparisons). Real-detector rows were not pre-registered as
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
    g = R.get("wiki_gpt41", {})
    gp = pct(g["plain"]) if "plain" in g else "--"
    lines = [f"No flags (plain) & {pct(R['wiki_plain'])} & {gp} \\\\", "\\midrule"]
    for k, v in t.items():
        c = k.replace(" vs plain", "")
        lines.append(f"{COND_NAME[c]} & {cell(v)} & {cell(g[c]) if c in g else '--'} \\\\")
    (out / "tab_wiki.tex").write_text(rf"""\begin{{table}}[t]
\centering\small
\begin{{tabular}}{{lrr}}
\toprule
Condition & gpt-4o-mini & gpt-4.1 \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}
\caption{{WikiContradict (253 genuine conflicts between two Wikipedia passages; the reference answer is both sides),
3 seeds (gpt-4.1: the oracle flag only). First row: share of answers giving both sides (\%); other rows: difference in points,
\textbf{{bold}} = significant under the paper's single scheme (H8 is a registered family; the gpt-4.1
column is exploratory).}}
\label{{tab:wiki}}
\end{{table}}
""")


def _num_word(n: int) -> str:
    ones = "Zero One Two Three Four Five Six Seven Eight Nine Ten Eleven Twelve Thirteen Fourteen Fifteen Sixteen " \
           "Seventeen Eighteen Nineteen".split()
    tens = "_ _ Twenty Thirty Forty Fifty Sixty Seventy Eighty Ninety".split()
    return ones[n] if n < 20 else tens[n // 10] + (ones[n % 10] if n % 10 else "")


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
                         _pq(v), "supported" if d["ci_low"] > 0.5 else "not supported", "--"))
        else:
            rows.append(("H1", _esc(k), _est(v), _pq(v), outcome(v, "H1"), "--"))
    for h in ("H2", "H3"):
        for k, v in R[h][PRIMARY].items():
            rows.append((h, _esc(k), _est(v), _pq(v), outcome(v, h), rob(R[h], k, h)))
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
            rows.append((h, _esc(k), _est(v), _pq(v), outcome(v, h), rob(R[h], k, h)))
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
                rows.append((h, _esc(k), _est(v), _pq(v), outcome(v, h), rob(R[h], k, h)))
    if "H8" in R:
        for k, v in R["H8"]["both_answers"].items():
            rows.append(("H8", "WikiContradict: " + _esc(k), _est(v), _pq(v), outcome(v, "H8", k), "--"))
    n_first = len(rows)
    for h in ("H10", "H11", "H12", "H13", "H14", "H15", "H16", "H17", "H18", "H19", "H20", "H22", "H23", "H25", "H26"):
        if h in rev:
            for k, v in rev[h][PRIMARY].items():
                two = h == "H14" or " vs " in k
                o = ("significant" if v["supported"] else "n.s.") if two else ("supported" if v["supported"] else "not supported")
                ob = rev[h][ROBUST][k]["supported"] == v["supported"]
                kk = k.replace("(labelled)", "(all questions)") if k.startswith(("ConflictBank", "CB majority", "RAGuard")) else k
                rows.append((h, _esc(kk), _est(v), _pq(v), o, "yes" if ob else "no"))
    head = r"""\begin{table*}[%s]
\centering\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lp{6.6cm}llll}
\toprule
& Pre-registered test & Estimate [95\%% CI] & $p_{\text{Holm}}$ ($q_{\text{BH}}$) & Outcome & Robust \\
\midrule
"""
    legend = (r"Directional predictions are \emph{supported} or \emph{not supported}; tests with no predicted sign are "
              r"\emph{significant} or \emph{n.s.} \emph{Robust}: the same outcome under the bidirectional metric (robustness "
              r"to the RAMDocs label rule is in Table~\ref{tab:contains}, where H4b's correlation weakens). "
              r"$p$-values come from 10,000 sign-flip permutations, extended to 200,000 when none reaches the observed "
              r"statistic, Holm-adjusted within the family; this is the significance used everywhere in the paper. "
              r"In brackets: the Benjamini--Hochberg $q$ over all registered tests of the programme (primary metric).")
    FAMILY = {"H1": "detection quality", "H2": "a correct neutral flag", "H3": "what direction adds",
              "H4a": "false flags", "H4b": "false flags", "H4c": "false flags", "H5": "the heuristic detector vs random",
              "H6": "does the grid predict real detectors", "H7": "prompt-only instructions",
              "H8": "genuine conflicts (WikiContradict)", "H9": "gpt-4.1", "H10": "without the answer policy (minimal skeleton)",
              "H11": "an open-weights generator", "H12": "ConflictBank", "H13": "neutral flag + resolution instruction",
              "H14": "published baselines", "H15": "ConflictBank, single-answer prompt",
              "H16": "information vs direction (one-wrong flag)", "H17": "majority misinformation",
              "H18": "a single-call baseline", "H19": "current reasoning models", "H20": "forced choice (no ``unknown'')",
              "H21": "risk and coverage", "H22": "pair topology", "H23": "answer key",
              "H24": "detector versus generator reliability (crossed)", "H25": "Claude", "H26": "RAGuard (real misleading posts)",
              "H27": "the detector model's answer as a hint", "H28": "LLM flags on genuine conflicts",
              "H29": "artefact probe of detector direction", "H30": "the detector's verdict as an answer key",
              "H31": "randomised direction"}

    def body(rs):
        out, last = [], None
        for r in rs:
            fam = FAMILY.get(r[0], "")
            if fam != last:
                out.append(rf"\multicolumn{{6}}{{l}}{{\emph{{{r[0].rstrip('abc')}: {fam}}}}} \\")
                last = fam
            out.append(" & ".join(r) + r" \\")
        return "\n".join(out)
    for m_, famd in rev.get("H21", {}).items():
        lab = {"misinfo": "misinformation rate", "abstain": "abstention rate", "selective": "selective accuracy"}[m_]
        if m_ == "selective" and rev.get("H21c_ratio"):   # §15.4.5 / §16.4.7: the ratio estimand of Table 1
            famd, lab = rev["H21c_ratio"], "selective accuracy (ratio)"
        for k, v in famd.items():
            rows.append(("H21", _esc(k.replace("vs plain", f"vs plain, {lab}")), _est(v), _pq(v),
                         "significant" if v["supported"] else "n.s.", "--"))
    h24 = rev.get("H24", {})
    if "a" in h24:
        a = h24["a"]
        rows.append(("H24", f"gain rises with detector-minus-generator reliability, Spearman over {a['n_cells']} cells",
                     f"$\\rho={a['rho']:+.2f}$", _p(a["p_one_sided"]), "supported" if a["p_one_sided"] < 0.05 else "not supported", "--"))
    if "b" in h24:
        b = h24["b"]
        rows.append(("H24", "per-question slope of gain on reliability gap, within cells",
                     f"{pct(b['slope'], True)} [{pct(b['ci_low'], True)}, {pct(b['ci_high'], True)}]", _p(b["p_one_sided"]),
                     "supported" if b["p_one_sided"] < 0.05 else "not supported", "--"))
    # §15: H27 (three families), H28 (per generator and benchmark), H29 (direction and gain)
    for f_ in ("H27a", "H27b", "H27c"):
        for k, v in rev.get(f_, {}).get(PRIMARY, {}).items():
            ob = rev[f_][ROBUST][k]["supported"] == v["supported"]
            rows.append(("H27", _esc(k), _est(v), _pq(v), "significant" if v["supported"] else "n.s.", "yes" if ob else "no"))
    for f_ in ("H30a", "H30b"):
        for k, v in rev.get(f_, {}).get(PRIMARY, {}).items():
            ob = rev[f_][ROBUST][k]["supported"] == v["supported"]
            rows.append(("H30", _esc(k), _est(v), _pq(v), "significant" if v["supported"] else "n.s.", "yes" if ob else "no"))
    for f_, pred in (("H31a", True), ("H31b", True), ("H31c", False)):
        for k, v in rev.get(f_, {}).items():
            o = ("supported" if v["supported"] else "not supported") if pred else ("significant" if v["supported"] else "n.s.")
            rows.append(("H31", _esc(k), _est(v), _pq(v), o, "--"))
    for key, fam_ in rev.get("H28", {}).items():
        for m_, lab in (("both_answers", ""), (PRIMARY, ""), ("gold_recall", ", gold recall")):
            for k, v in fam_.get(m_, {}).items():
                rows.append(("H28", _esc(k + lab), _est(v), _pq(v), "significant" if v["supported"] else "n.s.", "--"))
    h29 = rev.get("H29", {})
    for k, v in h29.get("a", {}).items():
        rows.append(("H29", _esc(f"{k}: direction accuracy vs original"), _est(v), _pq(v),
                     "significant" if v["supported"] else "n.s.", "--"))
    for k, v in h29.get("c", {}).items():
        rows.append(("H29", _esc(f"{k}: gpt-4o-mini's gain vs original"), _est(v), _pq(v),
                     "significant" if v["supported"] else "n.s.", "--"))
    # one table per registration stage, so no table outgrows a page
    stages = [("H10", "H11"), ("H12", "H13", "H14"), ("H15", "H16", "H17", "H18", "H19"),
              ("H20", "H21", "H22", "H23", "H24", "H25", "H26"), ("H27", "H28", "H29"), ("H30", "H31")]
    stage_rows = [[r for r in rows[n_first:] if r[0].rstrip("abc") in st] for st in stages]
    body1 = body(rows[:n_first])
    stage_txt = ["the first addendum (\\S9; H10--H11)", "the second (\\S10; H12--H14)",
                 "the third and fourth (\\S11--\\S12; H15--H19)", "the fifth and sixth (\\S13--\\S14; H20--H26)",
                 "the seventh (\\S15; H27--H29)", "the eighth (\\S16; H30--H31)"]
    parts = [head % "p" + body1 + "\n" + rf"""\bottomrule
\end{{tabular}}
\caption{{Pre-registered tests registered before the confirmatory run (RAMDocs hold-out unless stated; one-way
strict accuracy; differences in points). Rows marked \textsuperscript{{r}} are re-analyses registered later, before
they were computed. {legend}}}
\label{{tab:hypotheses}}
\end{{table*}}
"""]
    MAX_ROWS = 26   # long test names wrap, so a page holds about 26 rows plus the caption
    chunks = []
    for st, rs in zip(stage_txt, stage_rows):
        n_parts = -(-len(rs) // MAX_ROWS)
        for k in range(0, len(rs), MAX_ROWS):
            chunks.append((st + (f" (part {k // MAX_ROWS + 1} of {n_parts})" if n_parts > 1 else ""), rs[k:k + MAX_ROWS]))
    for n_, (st, rs) in enumerate(chunks):
        lab = f"tab:hypotheses{n_ + 2}"
        parts.append(head % "p" + body(rs) + "\n" + rf"""\bottomrule
\end{{tabular}}
\caption{{Pre-registered tests added in {st}, each committed before the runs it covers. Same conventions as
Table~\ref{{tab:hypotheses}}. Settings name the benchmark and generator. ConflictBank rows from H15 on use the
single-answer prompt unless marked \emph{{forced}}; \emph{{CB majority}} has three wrong-side passages per conflict
question; RAGuard's misleading posts are real Reddit text.}}
\label{{{lab}}}
\end{{table*}}
""")
    (out / "tab_hypotheses.tex").write_text("\n".join(parts))


BASELINE_NAME = {"self_identify": "Single call (name suspects, then answer)", "astute": "Astute RAG", "astute_multi": "Astute RAG (multi)", "trustrag": "TrustRAG",
                 "trustrag_multi": "TrustRAG (multi)", "madam": "MADAM-RAG",
                 "llm_dir+flag_directional": "LLM detector, dir.\\ flag",
                 "llm_dir+flag_directional_instruct_v2": "LLM detector, dir.\\ flag + instr.",
                 "llm_dir+flag_instruct_v2": "LLM detector, flag + instr.",
                 "llm_dir+remove_suspect": "LLM detector, remove suspect",
                 "instruct_v2": "Instruction (v2) only",
                 "oracle+flag_instruct_v2": "Oracle, flag + instr.",
                 "oracle+flag_directional_instruct_v2": "Oracle, dir.\\ flag + instr."}


def tab_baselines(R, out: Path):
    """Resolution instructions, published baselines and a single-call baseline against the LLM detector's
    directional flag (§10, §11)."""
    r2 = R.get("rev", {}).get("r2", {})
    if "gpt-4o-mini" not in r2:
        return
    mt = R["main_table"][PRIMARY]; lab = R["rev"]["labelled"]["gpt-4o-mini"][PRIMARY]
    a = r2.get("gpt-4o-mini"); rd = r2.get("RD:gpt-4o-mini")
    b = r2.get("gpt-4.1|all") or r2.get("gpt-4.1|seed0"); b0 = r2.get("gpt-4.1|seed0")
    cbs, maj = r2.get("CBs:gpt-4o-mini"), r2.get("CBmaj:gpt-4o-mini")
    rows_ = ["instruct_v2", "oracle+flag_instruct_v2", "oracle+flag_directional_instruct_v2", "madam", "astute",
             "astute_multi", "trustrag", "trustrag_multi", "self_identify", "llm_dir+flag_instruct_v2",
             "llm_dir+flag_directional_instruct_v2", "llm_dir+flag_directional", "llm_dir+remove_suspect"]
    calls = {"madam": r"9$\times$tok.", "astute": "2", "astute_multi": "2", "trustrag": "3", "trustrag_multi": "3",
             "instruct_v2": "1", "self_identify": "1", "oracle+flag_instruct_v2": "1",
             "oracle+flag_directional_instruct_v2": "1"}

    def eff(src, c, sb):
        return src["effects"][PRIMARY].get(c, {}).get(sb) if src else None

    def ramdocs_mini(c, sb):
        if c in mt:
            return lab[c]["labelled"] if sb == "labelled" else mt[c][sb]
        return eff(a, c, sb) or eff(rd, c, sb)

    def gpt41(c, sb):
        src = b if (b and c in b["effects"][PRIMARY]) else b0
        t = eff(src, c, sb)
        if t is None and c in R.get("gpt41_table", {}).get(PRIMARY, {}) and sb != "labelled":
            t = R["gpt41_table"][PRIMARY][c][sb]
        return t

    seed0 = b is b0

    def mark(c):   # gpt-4.1 cells computed on seed 0 only (round-2 runs), relative to the seed-0 plain
        from_r2 = bool(b and c in b["effects"][PRIMARY])
        return r"$^\dagger$" if (from_r2 and seed0) else ""

    pl = lambda src, sb: pct(src["plain"][PRIMARY][sb]) if src else "--"
    lines = [f"No flags (plain), accuracy & {pct(R['rev']['plain_labelled'][PRIMARY])} & {pct(R['plain_means'][PRIMARY]['conflict'])} & "
             f"{pct(R['plain_means'][PRIMARY]['clean'])} & {pl(b, 'labelled')}{chr(36)+'^'+chr(92)+'dagger'+chr(36) if seed0 else ''} & {pl(b, 'conflict')}{chr(36)+'^'+chr(92)+'dagger'+chr(36) if seed0 else ''} & "
             f"{pl(cbs, 'conflict')} & {pl(cbs, 'clean')} & {pl(maj, 'conflict')} & {pl(maj, 'clean')} & 1 \\\\", r"\midrule"]
    for c in rows_:
        cells = [cell(ramdocs_mini(c, sb)) for sb in ("labelled", "conflict", "clean")]
        g = [gpt41(c, sb) for sb in ("labelled", "conflict")]
        cells += [cell(t) + (mark(c) if t else "") for t in g]
        cells += [cell(eff(cbs, c, sb)) for sb in ("conflict", "clean")]
        cells += [cell(eff(maj, c, sb)) for sb in ("conflict", "clean")]
        n_calls = calls.get(c, "1+1")
        lines.append(f"{BASELINE_NAME.get(c, COND_NAME.get(c, c))} & " + " & ".join(cells) + f" & {n_calls} \\\\")
        if c in ("oracle+flag_directional_instruct_v2", "self_identify"):
            lines.append(r"\midrule")
    (out / "tab_baselines.tex").write_text(r"""\begin{table*}[t]
\centering\small
\setlength{\tabcolsep}{3.6pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrrrrrrrr}
\toprule
& \multicolumn{3}{c}{RAMDocs, gpt-4o-mini} & \multicolumn{2}{c}{RAMDocs, gpt-4.1} & \multicolumn{2}{c}{ConflictBank} & \multicolumn{2}{c}{CB majority} & \\
\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}
Method & Lab. & Conflict & Clean & Lab. & Conflict & Conflict & Clean & Conflict & Clean & Calls \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}}
\caption{Resolution instructions, published baselines and a single-call baseline. First row: accuracy (\%, one-way
strict) with no flags; other rows: paired difference from it in points. RAMDocs uses the multi-answer prompt (it has
same-name questions with several answers); ConflictBank and its majority variant (three wrong-side passages, one
correct) use the single-answer prompt and gpt-4o-mini. \emph{Lab.}: labelled RAMDocs questions. 3 seeds, except
$^\dagger$: gpt-4.1 seed 0 only, relative to the seed-0 no-flag accuracy in the first row (other gpt-4.1 cells are
relative to its 3-seed no-flag accuracy, Table~\ref{tab:gpt41}). \emph{instr.}: the v2 reliability instruction. \emph{(multi)}: the baseline's final
step asks for every valid answer. \emph{Single call}: one call that names suspect passages, then answers. Calls per
question; \emph{1+1}: one detector call and one generator call. \textbf{Bold}: significant under the paper's single
scheme.}
\label{tab:baselines}
\end{table*}
""")


def tab_risk(R, out: Path):
    """Risk and coverage on conflict questions (§13, H21): correct / misinformation / abstention, and accuracy on
    answered questions, for no flags, the oracle neutral flag and the oracle directional flag."""
    r2 = R.get("rev", {}).get("r2", {})
    spec = [("RD", "RAMDocs, multi-answer prompt"), ("CB", "ConflictBank, multi-answer prompt"),
            ("CBs", "ConflictBank, one-answer prompt"), ("CBf", "ConflictBank, forced choice")]   # RAGuard: Table tab_cb
    gens = [("gpt-4o-mini", "gpt-4o-mini"), ("gpt-6-luna", "gpt-6-luna"), ("claude-haiku-5-5", "Claude Haiku 5.5"),
            ("gpt-6.1-terra", "gpt-6.1-terra"), ("claude-sonnet-5-5", "Claude Sonnet 5.5"), ("gpt-4.1", "gpt-4.1"),
            ("Qwen2.5-7B", "Qwen2.5-7B")]
    conds = ("plain", "oracle+flag", "oracle+flag_directional")
    h21 = R.get("rev", {}).get("H21", {})

    def sig(key, c, m_):
        if c == "plain" or m_ not in h21:
            return False
        label = {"CBs": "ConflictBank", "CBf": "ConflictBank forced"}.get(key.split(":")[0])
        if not label:
            v = r2[key].get("risk", {}).get(m_, {}).get(c, {})
            return bool(v.get("sig"))
        tk = f"{label}, {key.split(':')[1]}: {'neutral flag' if c == 'oracle+flag' else 'dir. flag'} vs plain (conflict)"
        if m_ == "selective":   # §15.4.5: the ratio estimand, which is what the Sel. column shows
            return bool(R["rev"].get("H21c_ratio", {}).get(tk, {}).get("supported"))
        return bool(h21[m_].get(tk, {}).get("supported"))
    lines = []
    for pre, title in spec:
        rows_ = [(f"{pre}:{g}", name) for g, name in gens if f"{pre}:{g}" in r2 and "answer_mix" in r2[f"{pre}:{g}"]]
        if not rows_:
            continue
        lines.append(rf"\midrule" + "\n" + rf"\multicolumn{{13}}{{l}}{{\emph{{{title}}}}} \\")
        for key, name in rows_:
            mix = r2[key]["answer_mix"]; cells = []
            for c in conds:
                if c not in mix:
                    cells += ["--"] * 4; continue
                m = mix[c]
                sel = m["out_correct"] / (1 - m["unknown"]) if m["unknown"] < 1 else float("nan")
                vals = [(pct(m["out_correct"]), None), (pct(m["wrong"]), "misinfo"), (pct(m["unknown"]), "abstain"), (pct(sel), "selective")]
                cells += [rf"\textbf{{{v}}}" if (mk and sig(key, c, mk)) else v for v, mk in vals]
            lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    if not lines:
        return
    (out / "tab_risk.tex").write_text(r"""\begin{table*}[t]
\centering\small
\setlength{\tabcolsep}{3.5pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrrrrrrrrrr}
\toprule
& \multicolumn{4}{c}{No flags} & \multicolumn{4}{c}{Oracle, neutral flag} & \multicolumn{4}{c}{Oracle, directional flag} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}
Generator & Correct & Misinfo. & Abstain & Sel. & Correct & Misinfo. & Abstain & Sel. & Correct & Misinfo. & Abstain & Sel. \\
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}}
\caption{Risk and coverage on conflict questions (\%): strictly correct answers, answers that contain the
misinformation, abstentions (``unknown''), and selective accuracy (correct answers over non-abstained answers, pooled
over questions and seeds). \textbf{Bold}: the change from no flags is significant (H21 for
ConflictBank one-answer and forced; the paper's exploratory family elsewhere). The answer policy decides what a neutral
flag does: under a prompt that asks for every valid answer it adds the misinformation to the answer; under a one-answer
prompt that allows ``unknown'' it moves answers, misinformed and correct, into abstention; under forced choice it has
no abstention to trade.}
\label{tab:risk}
\end{table*}
""")


def fig_reliability(R, out: Path):
    """H24: each (detector, generator) cell's gain from the LLM directional flag against the detector's direction
    accuracy minus the generator's unaided accuracy (ConflictBank, one-answer prompt, conflict questions)."""
    import matplotlib.pyplot as plt
    cells = R.get("rev", {}).get("H24", {}).get("cells", {})
    if len(cells) < 3:
        return
    det_col = {"gpt-4o-mini": "#2a78d6", "gpt-4.1": "#eb6834", "gpt-6-luna": "#1baf7a", "gpt-6.1-terra": "#eda100",
               "claude-haiku-5-5": "#e87ba4", "claude-sonnet-5-5": "#008300"}
    gen_mark = {"gpt-4o-mini": "o", "gpt-6-luna": "s", "gpt-4.1": "^", "gpt-6.1-terra": "D", "claude-haiku-5-5": "P",
                "Qwen2.5-7B": "v"}
    short = {"gpt-4o-mini": "4o-mini", "gpt-4.1": "4.1", "gpt-6-luna": "6-luna", "gpt-6.1-terra": "6.1-terra",
             "claude-haiku-5-5": "Haiku 5.5", "claude-sonnet-5-5": "Sonnet 5.5", "Qwen2.5-7B": "Qwen-7B"}
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(3.3, 2.35), sharey=True)
    panels = (("gap", "detector right minus\ngenerator right (points)", "registered (H24)"),
              ("mech", "right where generator errs minus\nwrong where it is right (points)", "exploratory"))
    for ax, (fld, xl, tl) in zip(axes, panels):
        ax.axhline(0, color=INK2, lw=0.6); ax.axvline(0, color=INK2, lw=0.6)
        for key, v in cells.items():
            g, d = key.split("|")
            e = v["effect"]
            ax.errorbar(100 * v[fld], 100 * e["diff"],
                        yerr=[[100 * (e["diff"] - e["ci_low"])], [100 * (e["ci_high"] - e["diff"])]],
                        fmt="none", ecolor=det_col.get(d, NEUTRAL), elinewidth=0.5, alpha=0.45, zorder=2)
            ax.scatter(100 * v[fld], 100 * e["diff"], s=18, marker=gen_mark.get(g, "o"), color=det_col.get(d, NEUTRAL),
                       edgecolor="white", linewidth=0.8, zorder=3)
        ax.set_xlabel(xl, fontsize=6); ax.set_title(tl, fontsize=6.5, color=INK2, pad=2)
        ax.tick_params(labelsize=6)
    axes[0].set_ylabel("gain from LLM directional flag (points)", fontsize=6)
    hd = [plt.Line2D([], [], marker="o", ls="", ms=4, color=c, label=short[k])
          for k, c in det_col.items() if any(key.endswith("|" + k) for key in cells)]
    hg = [plt.Line2D([], [], marker=m, ls="", ms=4, color=INK2, label=short[k])
          for k, m in gen_mark.items() if any(key.startswith(k + "|") for key in cells)]
    fig.legend(handles=hd, title="detector (colour)", fontsize=5.5, title_fontsize=6, ncol=3, frameon=False,
               loc="upper left", bbox_to_anchor=(0.0, 0.0), handletextpad=0.1, columnspacing=0.6)
    fig.legend(handles=hg, title="generator (shape)", fontsize=5.5, title_fontsize=6, ncol=3, frameon=False,
               loc="upper right", bbox_to_anchor=(1.0, 0.0), handletextpad=0.1, columnspacing=0.6)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(out / "fig_reliability.pdf", bbox_inches="tight")
    plt.close(fig)
    nm = lambda k: k.replace("claude-", "Claude ").replace("-5-5", " 5.5").replace("haiku", "Haiku").replace("sonnet", "Sonnet")
    rows_ = sorted(cells.items(), key=lambda kv: kv[1]["gap"])
    body = "\n".join(f"{nm(k.split('|')[1])} & {nm(k.split('|')[0])} & {pct(v['det_acc'])} & {pct(v['gen_acc'])} & "
                     f"{pct(v['wrong_dir'])} & {pct(v['gap'], True)} & {pct(v['mech'], True)} & {cell(v['effect'])} \\\\"
                     for k, v in rows_)
    (out / "tab_cells.tex").write_text(r"""\begin{table}[t]
\centering\footnotesize
\setlength{\tabcolsep}{2.5pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{llrrrrrr}
\toprule
Detector & Generator & Dir.\ acc. & Gen.\ acc. & Wrong & Gap & Score & Gain \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}}
\caption{Every detector--generator cell of H24 (ConflictBank, one-answer prompt, 300 conflict questions; \%).
\emph{Dir.\ acc.}: share of questions on which the detector flags the true contradiction and names its misinformation
passage; \emph{Gen.\ acc.}: the generator's accuracy with no flags; \emph{Wrong}: share of questions on which it marks
a correct passage as suspect; \emph{Score} (exploratory): right direction where the generator errs minus a correct
passage marked where it is right; \emph{Gain}: the LLM directional flag's change in points. Figure~\ref{fig:reliability} plots the same cells.}
\label{tab:cells}
\end{table}
""")


def tab_cb(R, out: Path):
    """ConflictBank under every prompt and generator, and the majority variant (§10, §11). Conflict-question
    differences from no flags; LLM-detector clean-question cost in the last column."""
    r2 = R.get("rev", {}).get("r2", {})
    spec = [("CBs:gpt-4o-mini", "gpt-4o-mini", "single", 3), ("CB:gpt-4o-mini", "gpt-4o-mini", "multi", 3),
            ("CBn:gpt-4o-mini", "gpt-4o-mini", "minimal", 3), ("CBs:gpt-4.1", "gpt-4.1", "single", 1),
            ("CB:gpt-4.1", "gpt-4.1", "multi", 1), ("CBs:Qwen2.5-7B", "Qwen2.5-7B", "single", 1),
            ("CB:Qwen2.5-7B", "Qwen2.5-7B", "multi", 1),
            ("CBs:gpt-6-luna", "gpt-6-luna", "single", 3), ("CBs:gpt-6.1-terra", "gpt-6.1-terra", "single", 1),
            ("CBs:claude-haiku-5-5", "Claude Haiku 5.5", "single", 3), ("CBs:claude-sonnet-5-5", "Claude Sonnet 5.5", "single", 1),
            ("CBf:gpt-4o-mini", "gpt-4o-mini", "forced", 3), ("CBf:gpt-6-luna", "gpt-6-luna", "forced", 3),
            ("CBf:claude-haiku-5-5", "Claude Haiku 5.5", "forced", 3), ("CBf:gpt-6.1-terra", "gpt-6.1-terra", "forced", 1),
            None,
            ("CBmaj:gpt-4o-mini", "gpt-4o-mini", "single", 3), ("CBmaj:gpt-6-luna", "gpt-6-luna", "single", 3),
            ("CBmaj:gpt-6.1-terra", "gpt-6.1-terra", "single", 1), ("CBmaj:claude-haiku-5-5", "Claude Haiku 5.5", "single", 3),
            "RG",
            ("RG:gpt-4o-mini", "gpt-4o-mini", "single", 3), ("RG:gpt-6-luna", "gpt-6-luna", "single", 3),
            ("RG:claude-haiku-5-5", "Claude Haiku 5.5", "single", 3), ("RG:gpt-6.1-terra", "gpt-6.1-terra", "single", 1),
            ("RG:claude-sonnet-5-5", "Claude Sonnet 5.5", "single", 1)]
    if not any(k and k[0] in r2 for k in spec):
        return
    cols = ["oracle+flag", "oracle+flag_onewrong", "oracle+flag_directional", "oracle+remove_suspect",
            "llm_dir+flag", "llm_dir+flag_directional", "self_identify"]
    lines = []
    for sp in spec:
        if sp is None:
            lines.append(r"\midrule" + "\n" + r"\multicolumn{13}{l}{\emph{Majority misinformation: three wrong-side passages, one correct}} \\")
            continue
        if sp == "RG":
            lines.append(r"\midrule" + "\n" + r"\multicolumn{13}{l}{\emph{RAGuard: real Reddit posts labelled supporting or misleading (294 conflict claims)}} \\")
            continue
        key, gen, prompt, seeds = sp
        if key not in r2:
            continue
        v = r2[key]; e = v["effects"][PRIMARY]
        cells = [cell(e[c]["conflict"]) if c in e else "--" for c in cols]
        clean = cell(e["llm_dir+flag_directional"]["clean"]) if "llm_dir+flag_directional" in e else "--"
        cb = pct(v["closedbook"]["conflict"]) if "closedbook" in v else "--"
        lines.append(f"{gen} & {prompt} & {seeds} & {cb} & {pct(v['plain'][PRIMARY]['conflict'])} & " + " & ".join(cells)
                     + f" & {clean} \\\\")
    (out / "tab_cb.tex").write_text(rf"""\begin{{table*}}[t]
\centering\small
\setlength{{\tabcolsep}}{{3.6pt}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{llrrrrrrrrrrr}}
\toprule
& & & \multicolumn{{2}}{{c}}{{Accuracy}} & \multicolumn{{4}}{{c}}{{Oracle pairs}} & \multicolumn{{2}}{{c}}{{LLM detector}} & Single & LLM dir. \\
\cmidrule(lr){{4-5}}\cmidrule(lr){{6-9}}\cmidrule(lr){{10-11}}
Generator & Prompt & Seeds & Closed & Plain & Flag & One wrong & Dir.\ flag & Remove & Flag & Dir.\ flag & call & clean q. \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}}}
\caption{{ConflictBank conflict questions (300; one correct answer each) and RAGuard conflict claims (294; answer true
or false): accuracy (\%, one-way strict) closed-book and with the passages but no flags, then the paired difference
in points made by each format. \emph{{forced}}: the one-answer prompt without the option to answer ``unknown''. \emph{{Prompt}}: the
answer policy: \emph{{single}} says there is one answer, \emph{{multi}} asks for every valid answer (our default
skeleton), \emph{{minimal}} states no policy. \emph{{One wrong}}: the neutral flag plus ``exactly one of the two
contains misinformation''. \emph{{Single call}}: one call that names the suspect passages, then answers. Last column:
the LLM directional flag's change on the 100 clean questions. The LLM detector is gpt-4o-mini for every generator.
\textbf{{Bold}}: significant under the paper's single scheme.}}
\label{{tab:cb}}
\end{{table*}}
""")


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
    r2 = R.get("rev", {}).get("r2", {})
    for key, gen, seeds in (("RD:gpt-6-luna", "gpt-6-luna", 3), ("RD:gpt-6.1-terra", "gpt-6.1-terra", 1)):
        if key in r2:
            e = r2[key]["effects"][PRIMARY]
            cells = [cell(e[c][s]) if c in e else "--" for c, s in cols]
            lines.append(f"{gen}$^\\ddagger$ & default & {pct(r2[key]['plain'][PRIMARY]['conflict'])} & " + " & ".join(cells) + " \\\\")
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
\caption{{The main effects across generators and prompt skeletons (RAMDocs hold-out, 3 seeds, one-way strict
accuracy). \emph{{Plain}}: accuracy (\%) on conflict questions with no flags. Other columns: paired difference from plain
in points. \emph{{Default}} skeleton tells the model to give every valid answer; \emph{{minimal}} drops that policy
(Appendix~\ref{{app:prompts}}). \textbf{{Bold}}: significant under the paper's single scheme (Section~\ref{{sec:setup}}: a registered test's Holm-adjusted $p$ within its family, otherwise Holm over all exploratory comparisons). The LLM detector is gpt-4o-mini
for every generator. Qwen2.5-7B-Instruct runs locally (4-bit), seed 0. $^\ddagger$: current reasoning models at their
default temperature (gpt-6-luna 3 seeds, gpt-6.1-terra seed 0).}}
\label{{tab:generality}}
\end{{table*}}
""")


def _p(p):
    if _nan(p):
        return "--"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def _pq(v):
    """Holm-adjusted p and, in brackets, the Benjamini-Hochberg q over every registered test (revision7.fdr)."""
    q = v.get("q_bh")
    return _p(v["p_holm"]) + ("" if q is None or _nan(q) else f" ({_p(q)})")


# --------------------------------------------------------------------------- figures
def _style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
        "axes.facecolor": "white", "legend.frameon": False, "pdf.fonttype": 42})


OUTCOME = (("correct", "correct", "#2a78d6"), ("misinfo", "includes the misinformation", "#eb6834"),
           ("abstain", "abstains (\u201cunknown\u201d)", "#1baf7a"), ("other", "other error", "#eda100"))


def fig_answers(R, out: Path):
    """What each flag does to answers on conflict questions (gpt-4o-mini, oracle pairs), under the
    multi-answer and the single-answer prompt (§11). Stacked shares; every segment >= 6% is labelled."""
    import matplotlib.pyplot as plt
    r2 = R.get("rev", {}).get("r2", {})
    panels = [(k, t) for k, t in (("CB:gpt-4o-mini", "ConflictBank, multi-answer prompt"),
                                  ("CBs:gpt-4o-mini", "ConflictBank, single-answer prompt"),
                                  ("RD:gpt-4o-mini", "RAMDocs, multi-answer prompt")) if k in r2]
    if not panels:
        return
    conds = (("plain", "no flags"), ("oracle+flag", "neutral flag"), ("oracle+flag_onewrong", "\u201cone side is wrong\u201d"),
             ("oracle+flag_directional", "directional flag"))
    _style()
    n_rows = [sum(c in r2[k]["answer_mix"] for c, _ in conds) for k, _ in panels]
    fig, axes = plt.subplots(len(panels), 1, figsize=(3.3, 0.75 + 0.2 * sum(n_rows) + 0.25 * len(panels)), sharex=True,
                             gridspec_kw={"height_ratios": [n + 0.9 for n in n_rows]})
    axes = np.atleast_1d(axes)
    for ax, (key, title) in zip(axes, panels):
        mix = r2[key]["answer_mix"]
        rows_ = [(lab, mix[c]) for c, lab in conds if c in mix]
        for y, (lab, m) in enumerate(rows_[::-1]):
            left = 0.0
            for k, _, col in OUTCOME:
                w = m[f"out_{k}"]
                ax.barh(y, w, left=left, height=0.72, color=col, edgecolor="white", linewidth=1.0)
                if w >= 0.06:
                    ax.text(left + w / 2, y, f"{100 * w:.0f}", ha="center", va="center", fontsize=6,
                            color="white" if k in ("correct", "misinfo") else INK)
                left += w
        ax.set_yticks(range(len(rows_)), [lab for lab, _ in rows_[::-1]], fontsize=6.5)
        ax.set_title(title, fontsize=7, loc="left", color=INK, pad=2)
        ax.grid(False); ax.set_xlim(0, 1); ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    axes[-1].set_xticks([0, .25, .5, .75, 1], ["0", "25", "50", "75", "100%"], fontsize=6.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in OUTCOME]
    fig.legend(handles, [lab for _, lab, _ in OUTCOME], loc="lower center", ncol=2, fontsize=6.3,
               bbox_to_anchor=(0.55, -0.01), handlelength=1, columnspacing=1)
    fig.tight_layout(rect=(0, 0.09, 1, 1), h_pad=0.6)
    fig.savefig(out / "fig_answers.pdf", bbox_inches="tight")
    plt.close(fig)


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
          "CB:gpt-4o-mini": "CbMini", "CB:gpt-4.1": "CbFour", "CB:Qwen2.5-7B": "CbQwen",
          "gpt-4.1|all": "FourAll", "CBs:gpt-4o-mini": "SgMini", "CBs:gpt-4.1": "SgFour", "CBs:Qwen2.5-7B": "SgQwen",
          "CBn:gpt-4o-mini": "MnMini", "CBmaj:gpt-4o-mini": "MjMini", "CBmaj:gpt-4.1": "MjFour",
          "RD:gpt-4o-mini": "RdMini", "CBs:gpt-6-luna": "SgLuna", "CBs:gpt-6.1-terra": "SgTerra",
          "RD:gpt-6-luna": "RdLuna", "RD:gpt-6.1-terra": "RdTerra", "CBmaj:gpt-6-luna": "MjLuna",
          "CBmaj:gpt-6.1-terra": "MjTerra"}
    cname = lambda c: "".join(w.capitalize() for w in _re2.split(r"[^a-z0-9]+", c.replace("_v2", "_two")) if w and not w.isdigit())
    GEN_A = {"gpt-4o-mini": "Mini", "gpt-4.1": "Four", "Qwen2.5-7B": "Qwen", "gpt-6-luna": "Luna", "gpt-6.1-terra": "Terra",
             "claude-haiku-5-5": "Haiku", "claude-sonnet-5-5": "Sonnet"}
    PRE_A = {"CBs": "Sg", "RD": "Rd", "CBmaj": "Mj", "CBf": "Fc", "RG": "Rg", "CB": "Cb", "CBn": "Mn"}
    LAB_A = {"ConflictBank": "Sg", "RAMDocs": "Rd", "CB majority": "Mj", "ConflictBank forced": "Fc", "RAGuard": "Rg"}

    def abbrev(key):
        if key in T2:
            return T2[key]
        pre, gen = key.split(":", 1)
        return PRE_A[pre] + GEN_A[gen]
    for gk, v in rev.get("r2", {}).items():
        T = abbrev(gk)
        for sb in ("all", "conflict", "clean", "labelled"):
            add(f"zp{T}{sb.capitalize()}", pct(v["plain"][PRIMARY][sb]))
        for c, d in v["effects"][PRIMARY].items():
            for sb, t in d.items():
                nm = f"z{T}{cname(c)}{sb.capitalize()}"
                add(nm, pct(t["diff"], True)); add(nm + "CI", f"[{pct(t['ci_low'], True)}, {pct(t['ci_high'], True)}]")
        for c, w in v.get("wrong_rate", {}).items():
            add(f"zw{T}{cname(c)}", pct(w))
        for c, mix in v.get("answer_mix", {}).items():
            for k_, x in mix.items():
                add(f"zm{T}{cname(c)}{cname(k_)}", pct(x))
        for sb, x in v.get("closedbook", {}).items():
            add(f"zc{T}{sb.capitalize()}", pct(x))
    HN = {"H12": "HTwelve", "H13": "HThirteen", "H14": "HFourteen", "H15": "HFifteen", "H16": "HSixteen",
          "H17": "HSeventeen", "H18": "HEighteen"}
    SET_T = {"ConflictBank, gpt-4o-mini": "SgMini", "ConflictBank, gpt-4.1": "SgFour", "ConflictBank, Qwen2.5-7B": "SgQwen",
             "RAMDocs, gpt-4o-mini": "RdMini", "CB majority, gpt-4o-mini": "MjMini", "CB majority, gpt-4.1": "MjFour",
             "ConflictBank, gpt-6-luna": "SgLuna", "ConflictBank, gpt-6.1-terra": "SgTerra",
             "RAMDocs, gpt-6-luna": "RdLuna", "RAMDocs, gpt-6.1-terra": "RdTerra",
             "CB majority, gpt-6-luna": "MjLuna", "CB majority, gpt-6.1-terra": "MjTerra"}
    TEST_T = {"neutral flag vs plain (conflict)": "Flag", "dir. flag > neutral flag (conflict)": "DirFlag",
              "dir. flag > plain (conflict)": "Dir", "LLM dir. flag > plain (all q.)": "Llm",
              "dir. flag > one-wrong flag (conflict)": "DirOw", "one-wrong flag vs plain (conflict)": "Ow",
              "LLM dir. flag vs plain (conflict)": "Llm", "LLM dir. flag vs self-identify (labelled)": "Self"}
    HN["H19"] = "HNineteen"
    HN.update({"H20": "HTwenty", "H21": "HTwentyOne", "H22": "HTwentyTwo", "H23": "HTwentyThree",
               "H24": "HTwentyFour", "H25": "HTwentyFive", "H26": "HTwentySix"})
    TEST_T.update({"one-wrong without pairs vs with pairs (conflict)": "OwNp", "neutral without pairs vs with pairs (conflict)": "FlagNp",
                   "LLM dir. flag without pairs vs with pairs (conflict)": "LlmNp", "dir. flag vs answer key (conflict)": "Key"})

    def set_abbrev(label):
        if label in SET_T:
            return SET_T[label]
        pre, gen = label.rsplit(", ", 1)
        return LAB_A[pre] + GEN_A[gen]
    for h in ("H15", "H16", "H17", "H18", "H19", "H20", "H22", "H23", "H25", "H26"):   # \h<Family><Setting><Test>
        for tk, t in rev.get(h, {}).get(PRIMARY, {}).items():
            st, what = tk.split(": ", 1)
            nm = f"{HN[h][0].lower()}{HN[h][1:]}{set_abbrev(st)}{TEST_T[what]}"
            add(nm, pct(t["diff"], True)); add(nm + "CI", f"[{pct(t['ci_low'], True)}, {pct(t['ci_high'], True)}]")
            add(nm + "P", _p(t["p_holm"]))
    for name, d in rev.get("closedbook", {}).items():
        k = {"ConflictBank": "Cb", "RAMDocs": "Rd"}[name]
        add(f"cbk{k}NKnown", d["n_known"]); add(f"cbk{k}NConf", d["n_conflict"])
        for grp in ("known", "unknown"):
            g = d[grp]; G = grp.capitalize()
            add(f"cbk{k}{G}N", g["n"])
            add(f"cbk{k}{G}Dir", pct(g["direction"]) if g["direction"] is not None else "--")
            for c, cn in (("oracle+flag", "Flag"), ("oracle+flag_directional", "OracleDir"), ("llm_dir+flag_directional", "LlmDir")):
                if c in g:
                    add(f"cbk{k}{G}{cn}", pct(g[c]["diff"], True))
                    add(f"cbk{k}{G}{cn}CI", f"[{pct(g[c]['ci_low'], True)}, {pct(g[c]['ci_high'], True)}]")
    add("Nempty", rev.get("n_empty_reasoning", 0))
    jm = Path(__file__).resolve().parents[2] / "runs/judge_matcher.json"
    if jm.exists():   # LLM-judge check of the string matcher (§13.4, exploratory)
        js = json.loads(jm.read_text())["summary"]
        add("judgeN", js["n"]); add("judgeAgree", pct(js["agreement"])); add("judgeKappa", f"{js['kappa']:.2f}")
        add("judgeOnly", js["judge_only"]); add("matcherOnly", js["matcher_only"])
    np_ = rev.get("neutral_pooled")
    if np_:   # every oracle neutral-flag setting on conflict questions; random-effects pooled estimate (§13.4)
        add("neuPooled", pct(np_["mu"], True)); add("neuPooledCI", f"[{pct(np_['ci_low'], True)}, {pct(np_['ci_high'], True)}]")
        add("neuPooledTau", pct(np_["tau"])); add("nNeutralSettings", np_["k"])
        add("nNeutralHarm", np_["n_harm"]); add("nNeutralHelp", np_["n_help"])
    for tk, t in rev.get("tost_claims", {}).items():
        nm = "tost" + "".join(w.capitalize() for w in tk.split("_"))
        nm = _re2.sub(r"\d+", lambda m_: _num_word(int(m_.group())), nm)   # TeX macro names cannot hold digits
        add(nm, f"[{pct(t['ci90_low'], True)}, {pct(t['ci90_high'], True)}]")
        add(nm + "Eq", "equivalent" if t["equivalent"] else "not equivalent")
    for label, t in rev.get("H13_2x2", {}).items():
        k = "Mini" if label.startswith("RAMDocs, gpt-4o") else "Four" if "4.1" in label else "Cb"
        add(f"hThirteenX{k}", pct(t["diff"], True)); add(f"hThirteenX{k}CI", f"[{pct(t['ci_low'], True)}, {pct(t['ci_high'], True)}]")
    for m_, famd in rev.get("H21", {}).items():      # \hTwentyOne<Setting><Cond><Metric>
        for tk, t in famd.items():
            st, what = tk.split(": ", 1)
            nm = f"hTwentyOne{set_abbrev(st)}{'Flag' if what.startswith('neutral') else 'Dir'}{m_.capitalize()}"
            add(nm, pct(t["diff"], True)); add(nm + "CI", f"[{pct(t['ci_low'], True)}, {pct(t['ci_high'], True)}]")
            add(nm + "P", _p(t["p_holm"]))
        add(f"nHTwentyOne{m_.capitalize()}Supported", sum(v["supported"] for v in famd.values()))
        add(f"nHTwentyOne{m_.capitalize()}Tests", len(famd))
    for gk, v in rev.get("r2", {}).items():   # risk/coverage per condition: \zr<Set><Cond><Metric>
        for m_, d in v.get("risk", {}).items():
            for c, t in d.items():
                if not _nan(t.get("diff")):
                    add(f"zr{abbrev(gk)}{cname(c)}{m_.capitalize()}", pct(t["diff"], True))
        for c, mix in v.get("answer_mix", {}).items():
            sel = mix["out_correct"] / (1 - mix["unknown"]) if mix["unknown"] < 1 else float("nan")
            add(f"zs{abbrev(gk)}{cname(c)}", pct(sel))
    for key, suf in (("H24", ""), ("H24_s13", "Sthirteen"), ("H24_maj", "Maj")):
        h24 = rev.get(key, {})
        if "a" in h24:
            add(f"hTwentyFour{suf}Rho", f"{h24['a']['rho']:+.2f}"); add(f"hTwentyFour{suf}P", _p(h24["a"]["p_one_sided"]))
            add(f"hTwentyFour{suf}Cells", h24["a"]["n_cells"])
            add(f"hTwentyFour{suf}Pos", h24["n_positive"])
            if h24.get("break_even_gap") is not None:
                add(f"hTwentyFour{suf}BreakEven", pct(h24["break_even_gap"], True))
        if "mech" in h24:
            add(f"hTwentyFour{suf}MechRho", f"{h24['mech']['rho']:.2f}"); add(f"hTwentyFour{suf}MechR", f"{h24['mech']['r']:.2f}")
            add(f"hTwentyFour{suf}GapR", f"{h24['mech']['r_gap']:.2f}")
        if key == "H24":   # per-detector flag rates, read off the cells
            seen = set()   # detections are rerun per generator and differ slightly; report the first cell's
            for ck, cv in h24.get("cells", {}).items():
                dn = GEN_A[ck.split("|")[1]]
                if dn in seen:
                    continue
                seen.add(dn)
                add(f"detWrongDir{dn}", pct(cv["wrong_dir"])); add(f"detAnyFlag{dn}", pct(cv["any_flag"]))
                add(f"detRightDir{dn}", pct(cv["det_acc"]))
        if "b" in h24:   # points of gain per point of gap
            add(f"hTwentyFour{suf}Slope", f"{h24['b']['slope']:.2f}"); add(f"hTwentyFour{suf}SlopeP", _p(h24["b"]["p_one_sided"]))
            add(f"hTwentyFour{suf}SlopeCI", f"[{h24['b']['ci_low']:.2f}, {h24['b']['ci_high']:.2f}]")
    for g, v in rev.get("majority_breakdown", {}).items():
        k = "".join(w.capitalize() for w in g.replace("3", "three").replace("2", "two").replace("1", "one").split())
        add(f"mjb{k}N", v["n"]); add(f"mjb{k}Gain", pct(v["gain"], True))
    for ct, v in rev.get("conflicts_by_type", {}).items():
        k = "".join(w.capitalize() for w in _re2.split(r"[^a-z]+", ct.lower()) if w)
        add(f"cft{k}N", v["n"]); add(f"cft{k}Plain", pct(v["plain"]))
        for c in ("llm_dir+flag", "llm_dir+flag_directional", "instruct_v2"):
            add(f"cft{k}{cname(c)}", pct(v[c]["diff"], True) if not _nan(v[c].get("diff")) else "--")
    for h in ("H12", "H13", "H14", "H15", "H16", "H17", "H18", "H19", "H20", "H22", "H23", "H25", "H26"):
        if h in rev:
            hn = HN[h]
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
    for run, pre in (("cb_main", "cb"), ("qwendet", "qdet"), ("cbmaj_single", "cbmaj")):
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
    from .assets5 import numbers5
    numbers5(R, add)   # §15 (revision5)
    from .assets6 import numbers6
    numbers6(R, add)   # §16 (revision6)
    from .assets7 import numbers7
    numbers7(R, add)   # seventh-review re-analyses (revision7)
    from .assets8 import numbers8
    numbers8(R, add)   # eighth-review re-analyses, corrected baselines, semantic re-score
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
    fig_tradeoff(R, out); fig_placement(R, out); fig_answers(R, out); fig_reliability(R, out); tab_risk(R, out)
    from .assets5 import write5
    write5(R, out)   # after fig_reliability: replaces its tab_cells
    from .assets6 import write6
    write6(R, out)   # §16: replaces tab_pooled, tab_utility, tab_h24redo; adds the verdict column to tab_cells
    from .assets7 import write7
    write7(R, out)   # seventh review: replaces fig_reliability; adds tab_breakeven, tab_prevalence
    from .assets8 import write8
    write8(R, out)   # eighth review: tab_selection, tab_semantic, tab_baselines8
    grid = {}
    if "grid_cells" in R:
        for key, v in R["grid_cells"].items():
            mode, p, r, d, s = key.split("|")
            grid[(mode, float(p), float(r), float(d), s)] = v
        fig_grid(R, out, grid)


def app_prompts(out: Path):
    """Verbatim prompts for the appendix, taken from the code that ran."""
    from .detectors import LLM_DETECT_DIRECTIONAL_PROMPT, LLM_DETECT_PROMPT
    from .judge_matcher import PROMPT as JUDGE_PROMPT
    from .prepare_cb_probe import PARA_PROMPT
    from .prompts import (BASE, BASE_FORCED, BASE_MINIMAL, BASE_SINGLE, CLOSED_BOOK, INSTRUCTION, INSTRUCTION_PICK,
                          INSTRUCTION_SELF, INSTRUCTION_V2, build_prompt)
    from .detectors import Detection
    ex = build_prompt("Who founded X?", ["<passage 0>", "<passage 1>", "<passage 2>"], "flag_directional",
                      [Detection(0, 2, 1.0, "contradiction", [2])])[0]
    ow = build_prompt("Who founded X?", ["<passage 0>", "<passage 1>", "<passage 2>"], "flag_onewrong",
                      [Detection(0, 2, 1.0, "contradiction", [2])], skeleton="single")[0]
    def vb(title, text):
        import textwrap
        wrapped = "\n".join(textwrap.fill(l, 52) if l.strip() else "" for l in text.splitlines())
        return f"\\paragraph{{{title}}}\n\\begin{{footnotesize}}\\begin{{verbatim}}\n{wrapped}\n\\end{{verbatim}}\\end{{footnotesize}}\n"
    (out / "app_prompts.tex").write_text(
        vb("Generation skeleton (default).", BASE) + vb("Minimal skeleton (no answer policy).", BASE_MINIMAL)
        + vb("Single-answer skeleton.", BASE_SINGLE) + vb("Closed book (single-answer version).", CLOSED_BOOK["single"])
        + vb("Instruction v1.", INSTRUCTION) + vb("Instruction v2.", INSTRUCTION_V2)
        + vb("Single call (inserted as the instruction).", INSTRUCTION_SELF)
        + vb("Example: directional flag (passage 2 suspect).", ex)
        + vb("Example: one-wrong flag, single-answer skeleton.", ow)
        + vb("Directional LLM detector.", LLM_DETECT_DIRECTIONAL_PROMPT)
        + vb("LLM detector without direction.", LLM_DETECT_PROMPT)
        + vb("Forced-choice skeleton (no ``unknown'').", BASE_FORCED)
        + vb("Pick instruction (with the neutral flag).", INSTRUCTION_PICK)
        + vb("Example: neutral flag, single-answer skeleton.", _P3("flag"))
        + vb("Example: neutral flag without pairs.", _P3("flag_nopairs"))
        + vb("Example: one-wrong flag without pairs.", _P3("flag_onewrong_nopairs"))
        + vb("Example: directional flag without pairs.", _P3("flag_directional_nopairs"))
        + vb("Example: answer key (also the verdict key, with the detector's answers).",
             build_prompt("Who founded X?", ["<passage 0>", "<passage 1>", "<passage 2>"], "answer_key", skeleton="single",
                          wrong_answers=["<wrong answer>"])[0])
        + vb("Example: hint (the detector model's own answer).",
             build_prompt("Who founded X?", ["<passage 0>", "<passage 1>"], "hint", skeleton="single",
                          hint=["<other model's answer>"])[0])
        + vb("Example: WikiContradict (default skeleton, neutral flag).",
             build_prompt("<question>", ["<passage 0>", "<passage 1>"], "flag", [Detection(0, 1, 1.0, "contradiction", [])])[0])
        + vb("Paraphrase prompt (artefact probe, H29).", PARA_PROMPT)
        + vb("LLM judge of the string matcher.", JUDGE_PROMPT))


def _P3(mode):
    from .detectors import Detection
    from .prompts import build_prompt
    return build_prompt("Who founded X?", ["<passage 0>", "<passage 1>", "<passage 2>"], mode,
                        [Detection(0, 2, 1.0, "contradiction", [2])], skeleton="single")[0]


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
({c['n']['conflict']} conflict and {c['n']['clean']} clean hold-out questions; this rule assigns the 35 questions the
main analysis excludes). The bootstrap distribution of the H4 correlation is skewed, so its point estimate lies near the
end of its percentile interval.}}
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
        from .stats import holm
        adj = holm({f"{c}|{s}": v["p"] for c, d in R["conflicts"].items() for s, v in d.items() if not _nan(v.get("p"))})
        hc = lambda c, s: cell({**R["conflicts"][c][s], "p_tab": adj.get(f"{c}|{s}", float("nan"))}, "p_tab")
        for c, d in R["conflicts"].items():
            rows.append(f"{COND_NAME[c]} & {hc(c, 'all')} & {hc(c, 'conflict')} & {hc(c, 'clean')} \\\\")
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
\caption{{CONFLICTS, 237 automatically scorable questions (69 conflict), gpt-4o-mini, 3 seeds; exploratory.
Contains-match accuracy (\%) and differences in points. These comparisons use their own Holm family (this table),
more lenient than the paper's single exploratory family; bold would mark Holm-adjusted $p<0.05$ in it, and no difference
reaches it, so none is significant under either.
{det}.}}
\label{{tab:conflicts}}
\end{{table}}
""")
    (out / "app_tables.tex").write_text("\n".join(parts))
