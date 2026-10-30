"""
Paper macros and tables for the eighth revision (re-analyses in revision8.py; corrected baselines and the semantic
re-score from their own runs). numbers8 runs inside paper_assets.numbers after numbers7; write8 after write7.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from .assets5 import GEN_A, NICE, _f2
from .assets7 import H31_ORDER, _ci_pt, _pt
from .paper_assets import _nan, pct

ROOT = Path(__file__).resolve().parents[2]
CONTRAST_NAME = {"neutral": ("plain", "neutral flag"), "H23": ("answer key", "oracle directional flag"),
                 "H30": ("verdict key", "LLM directional flag"), "H31": ("plain", "randomised directional flag")}
BENCH = {"RD": "RAMDocs", "CBs": "ConflictBank"}


def _r(x):
    return "--" if x is None or _nan(x) else f"{x:.2f}"


def numbers8(R, add):
    rev = R["rev"]
    m = rev.get("neutral_pooled6")
    if m and m.get("ci_model"):
        add("neuMlCIModel", f"[{pct(m['ci_model'][0], True)}, {pct(m['ci_model'][1], True)}]")
        m8 = rev["neutral_pooled8"]
        add("neuMlCIDesign", f"[{pct(m8['ci_design'][0], True)}, {pct(m8['ci_design'][1], True)}]")
        add("neuMlMuModel", pct(m["mu_model"], True)); add("neuMlCorr", f"{m8['mean_corr_same_pool']:.2f}")
        add("neuMlEqual", pct(m8["equal_weight"], True))
        add("neuMlEqualCI", f"[{pct(m8['equal_weight_ci'][0], True)}, {pct(m8['equal_weight_ci'][1], True)}]")
        add("neuMlNBoot", m8["n_boot"])
    r31 = rev.get("realised31", {})
    for g, t in r31.items():
        G = GEN_A[g]
        add(f"realThirtyOne{G}", pct(t["diff"], True)); add(f"realThirtyOne{G}CI", f"[{pct(t['ci_low'], True)}, {pct(t['ci_high'], True)}]")
    if r31:
        add("nRealThirtyOneSig", sum(t["p_holm"] < 0.05 and t["diff"] < 0 for t in r31.values()))
        add("nRealThirtyOne", len(r31))
        add("shareRight", pct(next(iter(r31.values()))["share_right"], nd=0))
    sel = rev.get("selection8", {})
    t = sel.get("claude-haiku-5-5|gpt-4o-mini")
    if t and "marginal_net" in t:
        add("selHaikuMiniMarg", _pt(100 * t["marginal_net"], True)); add("selHaikuMiniObs", _pt(100 * t["obs"], True))
        add("selHaikuMiniRecall", pct(t["recall"], nd=0)); add("selHaikuMiniPrec", pct(t["precision"], nd=0))
        for nm, st in t["strata"].items():
            N = nm.capitalize()
            add(f"selHaikuMini{N}Share", pct(st["share"], nd=0))
            add(f"selHaikuMini{N}Base", "--" if st["base_acc"] is None else pct(st["base_acc"], nd=0))
            add(f"selHaikuMini{N}Gain", "--" if st["gain"] is None else pct(st["gain"], True))
            add(f"selHaikuMini{N}Contrib", pct(st["contrib"], True))
        add("selHaikuMiniBase", pct(t["base_acc"], nd=0))
    if sel:
        own = [v for v in sel.values() if "marginal_net" in v]
        add("selMargSign", sum(np.sign(v["marginal_net"]) == np.sign(v["obs"]) for v in own)); add("selN", len(own))
        add("selMargMae", f"{100 * np.mean([abs(v['marginal_net'] - v['obs']) for v in own]):.1f}")
        add("selMargCovMae", f"{100 * np.mean([abs(v['marginal_net_cov'] - v['obs']) for v in own]):.1f}")
        add("selMargR", _r(float(np.corrcoef([v['marginal_net'] for v in own], [v['obs'] for v in own])[0, 1])))
    for c in rev.get("predict7", {}).get("cells", []):
        if (c["gen"], c["det"]) == ("claude-haiku-5-5", "gpt-4o-mini"):
            add("predHaikuMini", _pt(c["pred"], True))
    p = rev.get("predict8")
    if p:
        fr = p["follow_rate"]
        add("predEightR", _r(fr["r"])); add("predEightRCI", f"[{_r(fr['r_ci'][0])}, {_r(fr['r_ci'][1])}]")
        add("predEightMae", f"{100 * fr['mae']:.1f}"); add("predEightMaeCI", f"[{100 * fr['mae_ci'][0]:.1f}, {100 * fr['mae_ci'][1]:.1f}]")
        add("predEightSign", fr["sign_agree"]); add("predEightN", fr["n"])
        add("predZeroMae", f"{100 * p['zero']['mae']:.1f}")
        add("predMargR", _r(p["marginal"]["r"])); add("predMargMae", f"{100 * p['marginal']['mae']:.1f}")
        add("predMargSign", p["marginal"]["sign_agree"])
        add("predGenR", _r(p["generator_mean"]["r"])); add("predGenMae", f"{100 * p['generator_mean']['mae']:.1f}")
        add("predGenSign", p["generator_mean"]["sign_agree"])
        x = p.get("cross_fit")
        if x:
            add("predXfR", _r(x["r"])); add("predXfMae", f"{100 * x['mae']:.1f}"); add("predXfN", x["n"])
            add("predXfSign", x["sign_agree"])
    h = rev.get("h23_het")
    if h:
        add("hTwentyThreeEqN", h["n_equivalent"]); add("hTwentyThreeK", h["k"])
        add("hTwentyThreeTau", f"{h['tau']:.1f}"); add("hTwentyThreeIsq", pct(h["I2"], nd=0))
    pv = rev.get("prevalence8")
    if pv:
        add("prevNegFiveCI", f"[{pv['n_neg_ci'][0]}, {pv['n_neg_ci'][1]}]")
        add("prevSigNeg", pv["n_sig_neg"]); add("prevSigPos", pv["n_sig_pos"])
    tr = rev.get("truncation8")
    if tr:
        add("truncMaxEmpty", pct(tr["max_empty"])); add("truncMaxNoFinal", pct(tr["max_no_final"]))
        add("truncMedNoFinal", pct(tr["median_no_final"]))
        k, c = tr["max_no_final_at"].rsplit("|", 1)
        g = k.split(":", 1)[-1].split("|")[0]
        add("truncMaxAt", f"{NICE.get(g, g)}, \\texttt{{{c.replace('_', '-')}}}")
    _numbers_runs(R, add)


# --------------------------------------------------------------------------- §17.2 / §17.3 outputs (own runs)
COND_SHORT = {"plain": "no flag", "oracle+flag": "neutral flag (oracle)", "oracle+flag_directional": "directional flag (oracle)",
              "answer_key": "answer key", "llm_dir+flag_directional": "directional flag (LLM)",
              "llm_dir+verdict_key": "verdict key (LLM)", "synth-p1-r1-all-d0.5+flag_directional": "randomised direction (H31)"}


def _bm(t, sign=True):
    return f"{pct(t[0], sign)} [{pct(t[1], True)}, {pct(t[2], True)}]"


def _numbers_runs(R, add):
    rev = R.get("rev", {})
    j = rev.get("judge8")
    if j:
        bc = j["by_condition"]
        n = sum(v["n"] for v in bc.values())
        add("semN", n); add("semAgree", pct(sum(v["agree"] * v["n"] for v in bc.values()) / n))
        for c, nm in (("plain", "Plain"), ("answer_key", "Key"), ("oracle+flag_directional", "Dir"), ("oracle+flag", "Neu"),
                      ("llm_dir+flag_directional", "Llm"), ("llm_dir+verdict_key", "Vkey")):
            if c in bc:
                add(f"semJudgeOnly{nm}", pct(bc[c]["judge_only"])); add(f"semMatcherOnly{nm}", pct(bc[c]["matcher_only"]))
        add("semRejected", pct(sum(v["rejected_but_counted_wrong"] * v["n"] for v in bc.values()) / n))
        dd = [v["judge_minus_matcher"][0] for v in j["contrasts"].values()]
        add("semDidMin", pct(min(dd), True)); add("semDidMax", pct(max(dd), True)); add("semK", len(dd))
        add("semNq", next(iter(j["contrasts"].values()))["n"])
        for k, v in j["contrasts"].items():
            h, b, g = k.split("|")
            nm = f"sem{h.replace('H', 'H').replace('neutral', 'Neu')}{b}{GEN_A[g]}"
            nm = nm.replace("H23", "TwentyThree").replace("H30", "Thirty").replace("H31", "ThirtyOne")
            add(nm + "M", _bm(v["matcher"])); add(nm + "J", _bm(v["judge"])); add(nm + "D", _bm(v["judge_minus_matcher"]))
        for g, G in (("gpt-4o-mini", "Mini"), ("gpt-6-luna", "Luna"), ("claude-haiku-5-5", "Haiku")):
            w = j.get("wiki", {})
            if f"{g}|effect" in w:
                add(f"wikiOrig{G}Plain", pct(w[f"{g}|plain"]["original"])); add(f"wikiOrig{G}Dir", pct(w[f"{g}|llm_dir+flag_directional"]["original"]))
                add(f"wikiOrig{G}Eff", _bm(w[f"{g}|effect"]["original"])); add(f"awikiOrig{G}EffPt", pct(abs(w[f"{g}|effect"]["original"][0]))); add(f"wikiBoth{G}Eff", _bm(w[f"{g}|effect"]["both"]))
        rg = j.get("raguard")
        if rg:
            add("rgAudN", rg["n"]); add("rgAudContradict", pct(rg["contradict"], nd=0)); add("rgAudTemporal", pct(rg["temporal"], nd=0))
            add("rgAudBFalse", pct(rg["b_false_yes"], nd=0)); add("rgAudBUnclear", pct(rg["b_false_unclear"], nd=0))
            add("rgAudASupports", pct(rg["a_supports_yes"], nd=0)); add("rgAudVerified", pct(rg["verified"], nd=0))
    b = rev.get("baselines8")
    if b:
        names = {"CB multi: TrustRAG with minus without system message (labelled)": "blTrustSysCb",
                 "CB majority: TrustRAG with minus without system message (labelled)": "blTrustSysMj",
                 "RAMDocs: TrustRAG with minus without system message (labelled)": "blTrustSysRd",
                 "RAMDocs: TrustRAG multi with minus without system message (labelled)": "blTrustSysRdMulti",
                 "CB multi: windowed NLI flag vs plain (conflict)": "blNliWinFlag",
                 "CB multi: windowed NLI remove vs plain (conflict)": "blNliWinRem",
                 "CB multi: sentence-window NLI flag vs plain (conflict)": "blNliSentFlag",
                 "CB multi: sentence-window NLI remove vs plain (conflict)": "blNliSentRem",
                 "CB multi: truncated NLI flag vs plain (conflict)": "blNliFlag",
                 "CB multi: truncated NLI remove vs plain (conflict)": "blNliRem",
                 "CB multi (registered H14 prompt): LLM dir. flag vs TrustRAG+sys (labelled)": "blHFourteenCb",
                 "CB one-answer (matched): LLM dir. flag vs TrustRAG+sys (labelled)": "blHFourteenCbs",
                 "CB majority (one-answer): LLM dir. flag vs TrustRAG+sys (labelled)": "blHFourteenMj",
                 "RAMDocs (multi-answer): LLM dir. flag vs TrustRAG multi+sys (labelled)": "blHFourteenRd"}
        for k, t in {**b["tests"], **b["H14_corrected"]}.items():
            nm = names.get(k)
            if nm is None or _nan(t["diff"]):
                continue
            add(nm, f"{pct(t['diff'], True)} [{pct(t['ci_low'], True)}, {pct(t['ci_high'], True)}]")
            if "p_holm" in t:
                add(nm + "Sig", "significant" if t["p_holm"] < 0.05 else "not significant")
        hf = b["H14_corrected"]
        add("blHFourteenNSig", sum(t["p_holm"] < 0.05 for t in hf.values())); add("blHFourteenN", len(hf))
        add("blHFourteenNPos", sum(t["p_holm"] < 0.05 and t["diff"] > 0 for t in hf.values()))
        for k, v in b.get("detect", {}).items():
            nm = {"nli": "Nli", "nli_win": "NliWin", "nli_win300": "NliSent"}.get(k)
            if nm:
                add(f"det{nm}Recall", pct(v["pair"]["recall"])); add(f"det{nm}Prec", pct(v["pair"]["precision"]))
                add(f"det{nm}CleanFpr", pct(v["q"]["false_positive_rate"]))
                add(f"det{nm}FlagRate", pct(v["flag_rate"]))


def tab_semantic8(R, out: Path):
    j = R.get("rev", {}).get("judge8")
    if not j:
        return
    lines = []
    for k, v in j["contrasts"].items():
        h, b, g = k.split("|")
        a_, b_ = CONTRAST_NAME[h]
        lines.append(f"{BENCH[b]}, {NICE[g]} & {b_} vs {a_} & {_bm(v['matcher'])} & {_bm(v['judge'])} & {_bm(v['judge_minus_matcher'])} \\\\")
    cond = [f"{COND_SHORT.get(c, c)} & {v['n']} & {pct(v['matcher'])} & {pct(v['judge'])} & {pct(v['judge_only'])} & {pct(v['matcher_only'])} \\\\"
            for c, v in sorted(j["by_condition"].items(), key=lambda kv: COND_SHORT.get(kv[0], kv[0]))]
    (out / "tab_semantic.tex").write_text(r"""\begin{table*}[t]
\centering\scriptsize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llrrr}
\toprule
Setting & Contrast & Matcher (points) & Judge (points) & Judge $-$ matcher \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}

\vspace{4pt}
\begin{tabular}{lrrrrr}
\toprule
Condition & Outputs & Matcher (\%) & Judge (\%) & Judge only (\%) & Matcher only (\%) \\
\midrule
""" + "\n".join(cond) + r"""
\bottomrule
\end{tabular}
\caption{Semantic re-score (\S17.3; exploratory). An outside-family judge (gemini-3.5-flash-lite, temperature 0) reads
the question, the reference and known wrong answers and the response, and classifies the final answer; judge-strict
means it gives the reference answer, asserts no wrong one and does not abstain. Top: each contrast on the same
""" + str(next(iter(j["contrasts"].values()))["n"]) + r""" conflict questions (seed 0) under the matcher and the judge, and their difference, with question-bootstrap
95\% CIs. Bottom: per condition, the share scored correct by each and the shares where only one does. The judge is
an LLM given the answer keys: it checks the matcher's semantics, not the benchmark labels.}
\label{tab:semantic}
\end{table*}
""")


def tab_selection8(R, out: Path):
    sel = R["rev"].get("selection8", {}); p7 = R["rev"].get("predict7", {})
    if not sel:
        return
    pred = {f"{c['gen']}|{c['det']}": c["pred"] for c in p7.get("cells", [])}
    lines = []
    order = {g: k for k, g in enumerate(H31_ORDER)}
    for key in sorted(sel, key=lambda k: (order.get(k.split("|")[0], 9), k)):
        v = sel[key]; g, d = key.split("|")
        st = v["strata"]
        cell = lambda nm: (f"{pct(st[nm]['share'], nd=0)} & {'--' if st[nm]['base_acc'] is None else pct(st[nm]['base_acc'], nd=0)} & "
                           f"{'--' if st[nm]['gain'] is None else pct(st[nm]['gain'], True)}")
        lines.append(f"{NICE[g]} & {NICE[d]} & {pct(v['precision'], nd=0) if v['precision'] is not None else '--'} & "
                     f"{pct(v['recall'], nd=0)} & {cell('right')} & {cell('wrong')} & {pct(st['rest']['contrib'], True)} & "
                     f"{_pt(100 * v['marginal_net'], True) if 'marginal_net' in v else '--'} & {_pt(pred.get(key), True)} & "
                     f"{pct(v['obs'], True)} \\\\")
    (out / "tab_selection.tex").write_text(r"""\begin{table*}[t]
\centering\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{llrrrrrrrrrrrr}
\toprule
& & & & \multicolumn{3}{c}{Flag names the false side} & \multicolumn{3}{c}{Flag marks a correct passage} & Rest & \multicolumn{3}{c}{Gain (points)} \\
\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){12-14}
Generator & Detector & Prec. & Rec. & Share & Acc. & Gain & Share & Acc. & Gain & contrib. & Marginal & Question & Observed \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\caption{The crossed cells (ConflictBank, one-answer prompt, conflict questions) split by what the flag does on each
question (exploratory, \S17.1). \emph{Prec.}, \emph{Rec.}: direction precision and recall of the gold pair (\%).
\emph{Share}: share of the cell's questions; \emph{Acc.}: the generator's no-flag accuracy on them (\%); \emph{Gain}:
the flag's mean effect on them (points). \emph{Rest}: the remaining questions' contribution to the gain.
\emph{Marginal}: the break-even calculation, $pR+(1-p)W$ with H31's $R$ and $W$ at the cell's precision $p$, which assumes
every question is flagged on its true pair and that flagged questions are typical; \emph{Question}: the follow-rate
account from each question's own flag and no-flag answer; both are retrospective. Generators without an H31 run have no
marginal prediction; their question-level prediction uses mean follow rates.}
\label{tab:selection}
\end{table*}
""")


def write8(R, out: Path):
    tab_selection8(R, out)
    tab_semantic8(R, out)
    b = R.get("rev", {}).get("baselines8_table")
    if b:
        (out / "tab_baselines8.tex").write_text(b)
