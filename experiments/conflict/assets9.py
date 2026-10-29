"""
Paper macros and the appendix table for the ninth revision (re-analyses in revision9.py, §18). numbers9 runs inside
paper_assets.numbers after numbers8; write9 after write8.
"""
from __future__ import annotations

from pathlib import Path

from .assets7 import _ci_pt, _pt

FAM = {"H30a": "Thirty", "H23": "TwentyThree"}


def numbers9(R, add):
    rev = R["rev"]
    a = rev.get("pair_audit9")
    if a:
        add("pairN", a["n_q"]); add("pairConfN", a["n_conflict"]); add("pairMultiN", a["n_multi"])
        add("pairSingleN", a["n_single"]); add("pairContraN", a["n_pairs"]); add("pairCrossLb", a["cross_lb"])
        add("pairCrossLbShare", f"{100 * a['cross_lb_share']:.0f}")
    e = rev.get("entity9", {}).get("pooled", {})
    for arm, tag in (("neutral", "Neu"), ("directional", "Dir")):
        t = e.get(arm)
        if not t:
            continue
        for part, ptag in (("single", "Single"), ("multi", "Multi")):
            add(f"ent{tag}{ptag}", _pt(t[part]["diff"], True)); add(f"ent{tag}{ptag}CI", _ci_pt(*t[part]["ci"]))
            add(f"ent{tag}{ptag}N", t[part]["n"])
        add(f"ent{tag}K", t["single"]["k"])
    for fam, tag in FAM.items():
        t = rev.get("agree9", {}).get(fam)
        if not t:
            continue
        add(f"agr{tag}K", t["discord"]["k"])
        for key, ktag in (("discord", "Dis"), ("flag_only", "FlagOnly"), ("key_only", "KeyOnly"), ("null", "Null"),
                          ("discord_matched", "DisMatched")):
            if t.get(key):   # shares, so unsigned
                add(f"agr{tag}{ktag}", _pt(t[key]["share"]))
                add(f"agr{tag}{ktag}CI", f"[{_pt(t[key]['ci'][0])}, {_pt(t[key]['ci'][1])}]")
        add(f"agr{tag}NullK", t["k_null"])


def tab_entity9(R, out: Path):
    s = R["rev"].get("entity9", {}).get("settings", {})
    if not s:
        return
    rows = []
    for name, v in sorted(s.items()):
        for arm in ("neutral", "directional"):
            t = v.get(arm)
            if not t or not t["single"] or not t["multi"]:
                continue
            gen = name.split(":", 1)[1].replace("|default", "").replace("|minimal", ", minimal")
            rows.append(f"{gen} & {arm} & {_pt(t['single']['diff'], True)} {_ci_pt(*t['single']['ci'])} & "
                        f"{_pt(t['multi']['diff'], True)} {_ci_pt(*t['multi']['ci'])} & {_pt(t['gap'], True)} \\\\")
    a = R["rev"].get("pair_audit9", {})
    (out / "tab_entity.tex").write_text(r"""\begin{table*}[t]
\centering\small
\setlength{\tabcolsep}{6pt}
\begin{tabular}{@{}llrrr@{}}
\toprule
Generator & Oracle flag & Single-entity & Multi-entity & Gap \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\caption{RAMDocs hold-out, oracle flag minus no flag on conflict questions, in points, split by whether the correct
passages give one answer (""" + str(a.get("n_single", "")) + r""" \emph{single-entity} questions, where no oracle pair
can join passages about different entities) or several (""" + str(a.get("n_multi", "")) + r"""), with question-bootstrap
95\% CIs (\S18.2; exploratory).}
\label{tab:entity}
\end{table*}
""")


def write9(R, out: Path):
    tab_entity9(R, out)
