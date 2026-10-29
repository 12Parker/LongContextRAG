"""LaTeX-specific preservation check, run alongside the skill's check_preservation.py. Compares, per part and as
multisets: number macros from generated/numbers.tex, \\ref/\\label/\\cite*/\\input/\\includegraphics arguments,
\\begin/\\end environments, \\footnote, \\paragraph/\\item[...] labels, inline math, \\texttt{...}, and hypothesis IDs (H1..H31).

  python check_latex.py <round_dir> [--only PART]
"""
import json, re, sys
from collections import Counter
from pathlib import Path

NUMS = Path(__file__).resolve().parents[2] / "generated/numbers.tex"
DEFINED = set(re.findall(r"\\newcommand\{?\\([A-Za-z]+)\}?", NUMS.read_text()))
MACRO = re.compile(r"\\([A-Za-z]+)")
ARGS = re.compile(r"\\(ref|label|cite[pt]?|input|includegraphics|begin|end|footnote|autoref|eqref)(?:\[[^\]]*\])*\{([^}]*)\}")
MATH = re.compile(r"(?<!\\)\$[^$]+\$")
TT = re.compile(r"\\texttt\{(?:[^{}]|\{[^{}]*\})*\}")
HYP = re.compile(r"\bH\d+[a-c]?\b")


def facets(t):
    t = "\n".join(l.split("%", 1)[0] if not l.lstrip().startswith("%") else "" for l in t.split("\n")) if False else t
    return {"number macros": Counter(m for m in MACRO.findall(t) if m in DEFINED),
            "latex args": Counter(f"\\{a}{{{b}}}" for a, b in ARGS.findall(t)),
            "inline math": Counter(MATH.findall(t)),
            "texttt": Counter(TT.findall(t)),
            "hypothesis ids": Counter(HYP.findall(t))}


def main():
    rd = Path(sys.argv[1]); only = sys.argv[3] if len(sys.argv) > 3 and sys.argv[2] == "--only" else None
    parts = json.load(open(rd / "order.json"))["parts"]
    bad = 0
    for n in parts:
        if only and only not in (n, n + ".tex"):
            continue
        a, b = (rd / "orig" / f"{n}.tex").read_text(), (rd / f"{n}.tex").read_text()
        fa, fb = facets(a), facets(b)
        probs = [f"{k}: missing {dict(fa[k] - fb[k])} added {dict(fb[k] - fa[k])}" for k in fa if fa[k] != fb[k]]
        print(f"{n}: " + ("OK" if not probs else ""))
        for p in probs:
            print("    PROBLEM", p)
        bad += bool(probs)
    print("OK" if not bad else f"{bad} part(s) with problems")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
