"""Second copy-edit pass (after §18-§19): like rounds.py, but related.tex and app_grid.tex are unchanged since the
first pass and left out, and appendix.tex is cut into four line ranges.

  python rounds.py split <round_dir>
  python rounds.py join  <round_dir>
"""
import json, sys
from pathlib import Path

SEC = Path(__file__).resolve().parents[2] / "sections"
# part name -> (source file, first line, last line) 1-based inclusive; None = whole file
PARTS = [("00_abstract", "abstract.tex", None), ("01_intro", "intro.tex", None),
         ("03_method", "method.tex", None), ("04_setup", "setup.tex", None), ("05_results", "results.tex", None),
         ("06_discussion", "discussion.tex", None), ("07_limitations", "limitations.tex", None),
         ("08_app_prompts_additional", "appendix.tex", (1, 135)),
         ("09_app_further_semantic", "appendix.tex", (136, 255)),
         ("10_app_fresh", "appendix.tex", (256, 293)), ("11_app_repro", "appendix.tex", (294, None))]


def chunks():
    out = []
    for name, f, rng in PARTS:
        lines = (SEC / f).read_text(encoding="utf-8").split("\n")
        a, b = rng or (1, None)
        out.append((name, f, "\n".join(lines[a - 1:b])))
    return out


def split(rd):
    (rd / "orig").mkdir(parents=True, exist_ok=True)
    for name, f, text in chunks():
        (rd / f"{name}.tex").write_text(text, encoding="utf-8")
        (rd / "orig" / f"{name}.tex").write_text(text, encoding="utf-8")
        print(f"{name}.tex ({f}): {len(text.split())} words")
    json.dump({"paper": str(SEC), "ext": ".tex", "parts": [p[0] for p in PARTS]}, open(rd / "order.json", "w"), indent=1)
    for f in {p[1] for p in PARTS}:
        assert "\n".join(t for n, ff, t in chunks() if ff == f) == (SEC / f).read_text(encoding="utf-8"), f
    print("round trip OK")


def join(rd):
    by_file = {}
    for name, f, _ in PARTS:
        by_file.setdefault(f, []).append(name)
    for f, names in by_file.items():
        orig = "\n".join((rd / "orig" / f"{n}.tex").read_text(encoding="utf-8") for n in names)
        if (SEC / f).read_text(encoding="utf-8") != orig:
            sys.exit(f"{f} changed since the split; refusing to overwrite it.")
    for f, names in by_file.items():
        new = "\n".join((rd / f"{n}.tex").read_text(encoding="utf-8") for n in names)
        old = (SEC / f).read_text(encoding="utf-8")
        (SEC / f).write_text(new, encoding="utf-8")
        print(f"{f}: {len(old.split())} -> {len(new.split())} words")


if __name__ == "__main__":
    {"split": split, "join": join}[sys.argv[1]](Path(sys.argv[2]).resolve())
