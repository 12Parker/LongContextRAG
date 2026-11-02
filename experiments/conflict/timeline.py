"""
Registration timeline (PREREGISTRATION §20.1; paper Appendix F): when each addendum was committed, when the remote
first received it, and when its first run started.

    python -m experiments.conflict.timeline --record   # writes runs/registration_timeline.json (needs gh and the
                                                       # original run folders; file birth times are local metadata)

Commit times are local git metadata; push times come from GitHub's repository activity API (third party); run starts
are the earliest file birth time in each section's run folders on the machine that ran them, recorded once here
because a clone cannot reproduce them.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/registration_timeline.json"
PR = "experiments/conflict/PREREGISTRATION.md"
# registration commit -> (section label, run folders it registered; empty = no runs)
SECTIONS = {
    "a6f7124": ("base (H1--H9)", ["holdout_main", "holdout_detect", "holdout_gpt41", "holdout_contains",
                                  "holdout_caag_original", "wikicontradict_main", "wikicontradict_gpt41", "conflicts_auto"]),
    "ad5535d": ("9 (H10--H11)", ["holdout_minimal", "holdout_gpt41_minimal", "holdout_qwen", "holdout_qwen_minimal"]),
    "b2ca800": ("9.4 (Qwen seed)", ["holdout_qwen"]),
    "222d52f": ("9.5 (inverted oracle)", ["holdout_adversarial"]),
    "f248eea": ("9.6 (Qwen aborts)", ["holdout_qwen"]),
    "a76ff84": ("10 (H12--H14)", ["holdout_round2", "holdout_gpt41_round2", "holdout_qwendet", "cb_main", "cb_gpt41",
                                  "cb_qwen"]),
    "e58249f": ("11 (H15--H18)", ["cb_single", "cb_minimal", "cb_single_gpt41", "cb_single_qwen", "cbmaj_single",
                                  "holdout_extra"]),
    "886e5c1": ("12 (H19)", ["cb_single_luna", "holdout_luna", "cbmaj_single_luna", "cb_single_terra", "holdout_terra",
                             "cbmaj_single_terra"]),
    "108e41d": ("13 (H20--H24)", ["cb_forced_*", "ctl_*", "cross_cb_*", "cross_mj_*"]),
    "eca911b": ("14 (H25--H26)", ["cb_single_haiku", "cb_single_sonnet", "holdout_haiku", "holdout_sonnet",
                                  "cbmaj_single_haiku", "rg_*"]),
    "89f6728": ("15 (H27--H29)", ["hint_cb_*", "probe_*", "wiki_llm_*", "holdout_rem_*"]),
    "13602c7": ("16 (H30--H31)", ["randdir_cb_*", "vkey_cb_*"]),
    "88b5f7d": ("17 (re-analyses, baselines, judge)", ["rev8_*", "judge8"]),
    "21d916c": ("17.6 (deviations)", ["judge8"]),
    "85c3c85": ("18 (re-analyses)", []),
    "d6185df": ("19 (H32--H38)", ["fresh_cb_*", "hc_*", "randdir_fresh_mini", "judge10"]),
    "d7e8d76": ("20 (re-analyses)", []),
}


def _git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def record():
    acts = json.loads(subprocess.run(["gh", "api", "--paginate", "repos/{owner}/{repo}/activity?per_page=100"],
                                     cwd=ROOT, capture_output=True, text=True, check=True).stdout.replace("][", ","))
    pushes = sorted((a["timestamp"], a["after"]) for a in acts if a["activity_type"] in ("push", "force_push"))
    out = []
    for short, (label, pats) in SECTIONS.items():
        full = _git("rev-parse", short).strip()
        committed = _git("show", "-s", "--format=%cI", full).strip()
        pushed = next((t for t, after in pushes
                       if subprocess.run(["git", "merge-base", "--is-ancestor", full, after], cwd=ROOT,
                                         capture_output=True).returncode == 0), None)
        dirs = sorted({d for p in pats for d in glob.glob(str(ROOT / "runs" / p)) if os.path.isdir(d)})
        files = [f for d in dirs for f in glob.glob(d + "/**", recursive=True) if os.path.isfile(f)]
        first = min(((os.stat(f).st_birthtime, f) for f in files), default=None)
        out.append({"commit": full, "section": label, "committed": datetime.fromisoformat(committed)
                    .astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "pushed": pushed,
                    "first_run_file": _iso(first[0]) if first else None,
                    "first_run_path": os.path.relpath(first[1], ROOT) if first else None})
    OUT.write_text(json.dumps(out, indent=1))
    for r in out:
        print(f"{r['section']:<36} commit {r['committed']}  push {r['pushed']}  first run {r['first_run_file']}")


def tab_timeline(out: Path):
    if not OUT.exists():
        return
    rows = []
    for r in json.loads(OUT.read_text()):
        c, p, f = (datetime.strptime(x, "%Y-%m-%dT%H:%M:%SZ") if x else None
                   for x in (r["committed"], r["pushed"], r["first_run_file"]))
        fmt = lambda t: t.strftime("%m-%d %H:%M") if t else "--"
        mark = lambda t: "--" if not (t and f) else ("yes" if t <= f else "no")
        name = f"PR\\S{r['section']}" if r["section"][0].isdigit() else r["section"].capitalize()
        rows.append(f"{name} & {fmt(c)} & {fmt(p)} & {fmt(f)} & {mark(c)} & {mark(p)} \\\\")
    (out / "tab_timeline.tex").write_text(r"""\begin{table}[t]
\centering\small
\setlength{\tabcolsep}{3pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{@{}lccccc@{}}
\toprule
& & & First run & \multicolumn{2}{c}{Before first run?} \\
\cmidrule(l){5-6}
Registration & Commit & Push & file & Commit & Push \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\caption{Registration timeline (UTC, 2026). \emph{Commit}: local git time; \emph{Push}: when the remote first held
the commit (GitHub's activity log, a third-party record); \emph{First run file}: the earliest file in the runs that
section registered (local file metadata). Sections without runs (PR\S18, PR\S20) register analyses only. PR\S9.4 and
PR\S9.6 changed the Qwen runs already under way, and PR\S17.6 followed the judge outputs (PR\S20.1).}
\label{tab:timeline}
\end{table}
""")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    args = ap.parse_args(argv)
    if args.record:
        record()


if __name__ == "__main__":
    main()
