"""
Check the environment before spending money.

    python -m experiments.conflict.check_env          # offline + model downloads
    python -m experiments.conflict.check_env --live   # also one tiny real API call (~$0.00001)
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys

OK, BAD = "  [ok]  ", "  [FAIL]"
failures = 0


def require(cond, msg):
    if not cond:
        raise RuntimeError(msg)
    return None


def check(label, fn):
    global failures
    try:
        msg = fn()
        print(f"{OK} {label}{(': ' + str(msg)) if msg else ''}")
    except Exception as e:  # noqa: BLE001
        failures += 1
        print(f"{BAD} {label}: {type(e).__name__}: {e}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--skip-models", action="store_true")
    args = ap.parse_args(argv)

    print("Python / packages")
    check("python >= 3.10", lambda: require(sys.version_info >= (3, 10), sys.version) or sys.version.split()[0])
    for p in ["openai", "sentence_transformers", "torch", "spacy", "rank_bm25", "datasets",
              "numpy", "scipy", "sklearn", "tqdm"]:
        check(p, lambda p=p: getattr(importlib.import_module(p), "__version__", "installed"))
    check("spaCy en_core_web_sm", lambda: __import__("spacy").load("en_core_web_sm").meta["version"])

    print("Data")
    from .data import load
    check("RAMDocs", lambda: f"{len(load('ramdocs'))} questions")
    check("CONFLICTS", lambda: f"{len(load('conflicts'))} questions")

    print("Keys")
    check("OPENAI_API_KEY set", lambda: require(os.getenv("OPENAI_API_KEY"), "missing; put it in .env") or "yes")

    if not args.skip_models:
        print("Models (downloads from Hugging Face on first run)")
        def emb():
            from .detectors import STEmbedder
            v = STEmbedder().embed_documents(["hello"])[0]
            return f"dim={len(v)}"
        check("all-MiniLM-L6-v2", emb)

        def nli():
            from .detectors import NLIDetector
            from .data import Example
            d = NLIDetector()
            ex = Example("t", "t", "q", ["A man is sleeping in his bed.", "A man is running a marathon right now."])
            s = d.detect(ex).scores[(0, 1)]
            if s < 0.5:
                raise RuntimeError(f"contradiction prob {s:.2f} on an obvious contradiction; check label mapping")
            return f"contradiction label idx={d.c_idx}, p={s:.2f} on sanity pair"
        check("NLI cross-encoder + label sanity", nli)
        check("NarrativeQA (HF)", lambda: f"{len(__import__('datasets').load_dataset('narrativeqa', split='test'))} rows")

    if args.live:
        print("Live API")
        def live():
            from .llm import OpenAILLM
            r = OpenAILLM(max_tokens=5)("Reply with the word ok.", seed=12345)
            return f"'{r.text.strip()}' tokens={r.prompt_tokens}+{r.completion_tokens} cost=${r.cost_usd:.7f}"
        check("gpt-4o-mini", live)

    print("\nAll checks passed." if not failures else f"\n{failures} check(s) failed.")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
