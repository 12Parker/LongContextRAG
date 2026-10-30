"""
Build HoH-QA items with real Wikipedia paragraphs (smoke test for a planned registration).

    HoH-QA (Ouyang et al., ACL 2025): HF russwest404/HoH-QAs @ d4236a2 (Apache-2.0), downloaded to
    data/external/hoh_src/hoh_qas_240601_241201.parquet
    python -m experiments.conflict.prepare_hoh --n-conflict 20 --n-clean 20

Each HoH item is a fact that changed between two monthly Wikipedia dumps (June-December 2024): a question, the
current answer and evidence sentence, and the outdated answer and evidence sentence with their dump dates. Only the
sentences ship, so for each item we fetch the article as of each dump date through the MediaWiki API
(prop=revisions, rvstart=<date>, rvdir=older), render that revision to text (action=parse) and take the paragraph
that best contains the evidence sentence. A conflict item gets the current paragraph, the outdated paragraph and two
other paragraphs of the current article; a clean item gets the current paragraph and three other paragraphs. Types:
the current paragraph is correct, the outdated one misinfo (stale), other paragraphs noise unless they contain an
answer. Passage order is shuffled per item. A dated variant prefixes every passage with its revision date.

Sampling: items with one outdated version, current and outdated answers that differ and are not substrings of each
other, answers of at most 6 words, changes observed in 2024-09..2024-12; seed 0. Items whose evidence cannot be
located in the fetched revision (token overlap < 0.6) are skipped and counted.

Writes data/external/hoh/hoh_eval.jsonl and hoh_dated.jsonl (not committed) and splits/hoh_smoke.txt; with --name full,
hoh_full.jsonl, hoh_full_dated.jsonl and splits/hoh_full.txt (pages of --exclude-split left out).
"""
from __future__ import annotations

import argparse
import html
import json
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/external/hoh_src/hoh_qas_240601_241201.parquet"
OUT = ROOT / "data/external/hoh"
SPLITS = Path(__file__).resolve().parent / "splits"
API = "https://en.wikipedia.org/w/api.php"
UA = "LongContextRAG-research/0.1 (https://github.com/12Parker/LongContextRAG; academic study) python-urllib"
CACHE = ROOT / "data/external/hoh_src/api_cache.jsonl"


def _get(params, cache):
    key = json.dumps(params, sort_keys=True)
    if key in cache:
        return cache[key]
    url = API + "?" + urllib.parse.urlencode({**params, "format": "json", "formatversion": 2})
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                out = json.loads(r.read().decode())
            break
        except urllib.error.HTTPError as e:   # 429: honour Retry-After
            time.sleep(float(e.headers.get("Retry-After") or 10 * (attempt + 1)))
        except Exception:
            time.sleep(5 * (attempt + 1))
    else:
        raise RuntimeError(f"MediaWiki API failed: {params}")
    cache[key] = out
    with open(CACHE, "a") as f:
        f.write(json.dumps({"k": key, "v": out}) + "\n")
    time.sleep(1.0)   # one request per second, serial (Wikimedia API etiquette)
    return out


def revision_paragraphs(page_id, date, cache):
    q = _get({"action": "query", "prop": "revisions", "pageids": page_id, "rvlimit": 1, "rvdir": "older",
              "rvstart": f"{date}T00:00:00Z", "rvprop": "ids|timestamp"}, cache)
    revs = q["query"]["pages"][0].get("revisions")
    if not revs:
        return None, []
    rev = revs[0]
    p = _get({"action": "parse", "oldid": rev["revid"], "prop": "text", "disablelimitreport": 1, "disableeditsection": 1}, cache)
    h = p.get("parse", {}).get("text", "")
    paras = []
    for m in re.finditer(r"<p>(.*?)</p>", h, re.S):
        t = re.sub(r"<sup[^>]*>.*?</sup>", "", m.group(1), flags=re.S)
        t = html.unescape(re.sub(r"<[^>]+>", "", t))
        t = " ".join(t.split())
        if len(t.split()) >= 12:
            paras.append(t)
    return rev["timestamp"][:10], paras


def _toks(s):
    return set(re.findall(r"\w+", s.lower()))


def best_para(paras, sentence):
    st = _toks(sentence)
    scored = [(len(st & _toks(p)) / max(len(st), 1), k) for k, p in enumerate(paras)]
    return max(scored) if scored else (0.0, None)


def norm(s):
    return " ".join(re.findall(r"\w+", s.lower()))


def main(argv=None):
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-conflict", type=int, default=20)
    ap.add_argument("--n-clean", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="smoke", choices=("smoke", "full"),
                    help="smoke: hoh_eval/hoh_dated + splits/hoh_smoke.txt; full: hoh_full/hoh_full_dated + splits/hoh_full.txt")
    ap.add_argument("--exclude-split", default=None, help="ids file whose pages are excluded (e.g. the smoke split)")
    args = ap.parse_args(argv)
    cache = {}
    if CACHE.exists():
        for l in open(CACHE):
            r = json.loads(l); cache[r["k"]] = r["v"]
    d = pd.read_parquet(SRC)
    rows = []
    for r in d.itertuples(index=False):
        if len(r.outdated_infos) != 1 or str(r.last_modified_time)[:7] < "2024-09":
            continue
        old = r.outdated_infos[0]
        a, b = norm(r.answer), norm(old["answer"])
        if not a or not b or a in b or b in a or len(r.answer.split()) > 6 or len(old["answer"].split()) > 6:
            continue
        rows.append(r)
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    items, skipped, used_pages = [], {"no_revision": 0, "no_match": 0}, set()
    if args.exclude_split:   # ids are hoh-<page id>-<n>
        used_pages |= {l.strip().split("-")[1] for l in open(args.exclude_split) if l.strip()}
    excluded = set(used_pages)
    want = {"conflict": args.n_conflict, "clean": args.n_clean}
    for r in rows:
        if not any(want.values()):
            break
        kind = "conflict" if want["conflict"] else "clean"
        pid = r.document["id"]
        if pid in used_pages:
            continue
        old = r.outdated_infos[0]
        cur_date, cur_paras = revision_paragraphs(pid, str(r.last_modified_time)[:10], cache)
        if not cur_paras:
            skipped["no_revision"] += 1; continue
        sc, k = best_para(cur_paras, r.evidence)
        if sc < 0.6 or norm(r.answer) not in norm(cur_paras[k]):
            skipped["no_match"] += 1; continue
        cur = cur_paras[k]
        passages = [(cur, "correct", cur_date)]
        if kind == "conflict":
            old_date, old_paras = revision_paragraphs(pid, old["last_modified_time"][:10], cache)
            if not old_paras:
                skipped["no_revision"] += 1; continue
            so, ko = best_para(old_paras, old["evidence"])
            if so < 0.6 or norm(old["answer"]) not in norm(old_paras[ko]) or old_paras[ko] == cur:
                skipped["no_match"] += 1; continue
            passages.append((old_paras[ko], "misinfo", old_date))
        others = [p for j, p in enumerate(cur_paras) if j != k and p not in [x for x, _, _ in passages]]
        rng.shuffle(others)
        for p in others[: 4 - len(passages)]:
            t = "correct" if norm(r.answer) in norm(p) else ("misinfo" if norm(old["answer"]) in norm(p) else "noise")
            passages.append((p, t, cur_date))
        random.Random(f"{pid}-{r.question}").shuffle(passages)
        used_pages.add(pid)
        want[kind] -= 1
        items.append({"id": f"hoh-{pid}-{len(items)}", "question": r.question, "gold": r.answer,
                      "wrong": old["answer"] if kind == "conflict" else None,
                      "passages": [p for p, _, _ in passages], "types": [t for _, t, _ in passages],
                      "dates": [dt for _, _, dt in passages], "has_conflict": kind == "conflict",
                      "title": r.document["title"], "changed": str(r.last_modified_time)[:10]})
    OUT.mkdir(parents=True, exist_ok=True)
    stem, dstem, split = (("hoh_eval", "hoh_dated", "hoh_smoke") if args.name == "smoke" else
                          ("hoh_full", "hoh_full_dated", "hoh_full"))
    assert not {o["id"].split("-")[1] for o in items} & excluded
    (OUT / f"{stem}.jsonl").write_text("".join(json.dumps(o) + "\n" for o in items))
    dated = [{**o, "passages": [f"(Last modified: {dt}) {p}" for p, dt in zip(o["passages"], o["dates"])]} for o in items]
    (OUT / f"{dstem}.jsonl").write_text("".join(json.dumps(o) + "\n" for o in dated))
    (SPLITS / f"{split}.txt").write_text("".join(o["id"] + "\n" for o in sorted(items, key=lambda o: o["id"])))
    print(f"{len(items)} items ({sum(o['has_conflict'] for o in items)} conflict); skipped {skipped}; "
          f"API calls cached: {len(cache)}")


if __name__ == "__main__":
    main()
