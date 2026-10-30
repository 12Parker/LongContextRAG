# Round C triage: the reader's view and concision

**Principles:** Peyton Jones (one idea; cut restatement; finding before method); Mensh & Kording (each section does its own job); omit needless words; collapse doubled hedges; use the defined term. The cut targets were 5% for the main body and 4% for limitations and the appendix, and no whole paragraph could be deleted.

## Per-part results

Both checks pass for every part. Total prose words went from 10,539 to 10,061 (−4.5%).

| Part | Change |
|---|---|
| abstract | −6.0% |
| intro | −5.2% |
| related | −5.2% |
| method | −5.1% |
| setup | −5.1% |
| results | −5.1% |
| discussion | 0 |
| limitations | −4.0% |
| app. part 1 | −4.2% |
| app. part 2 | −4.1% |
| reproducibility | −4.1% |
| grid | −4.6% |

## Reviewed

I read the word diffs of every body part. The deleted sentences were these, and each only restated a neighbouring sentence:
- **Intro:** "A natural fix detects the contradiction and tells the generator." The next sentence says the same thing, with its citation.
- **Method:** the one-wrong flag's "The flag conveys that the conflict is misinformation, not its direction." The sentence before already gives that definition.
- **Setup:** WikiContradict's "We score whether an answer gives both." Metrics states the same scoring rule.
- **App. grid:** a rhetorical question, and "If the grid only described synthetic detectors, it would be of limited use."
- **App. part 1:** a repeated NarrativeQA sentence.
- **Reproducibility:** "; no row was re-sampled" in one item. The Run-interruptions item still says it for every resumed run (checked).

I found no change to a hedge, a registration label, a sign word, a number or a macro.

## Fixed after the round (coordinator)

1. **Results:** "gpt-4o-mini's share of misinformed answers" → "misinformation share" (the glossary term; "misinformed answers" means something else in H21).
2. **App. semantic re-score:** "Given the gold answers, the judge…" → "Given the gold and wrong answers". Source: `judge8.ANSWER_PROMPT` gives the judge both lists.
3. **App. pilot paragraph:** two bare "They" sentences were restored to "The pilots", since after the flag numbers "They" read as "the flags".

## Whole paper vs HEAD (dc00333)

- `check_preservation.py --paper`: words 11,037 → 10,592; no number gone; no occurrence removed.
- `check_latex.py`, run over the whole paper: nothing missing. The four additions are deliberate coordinator fixes:
  - `\hTwentyFourCells` in the intro;
  - `\ref{app:grid}`;
  - `$\pm$` in the abstract;
  - "H24" in Limitations.
- The tectonic build has no errors or undefined references. The PDF is 54 pages, and the overfull boxes are the same ones as before, all in generated tables.
- Tests: 86 passed.
- **Page fit:** the body now ends about halfway down page 8's right column, and Limitations starts on page 8 instead of at the top of page 9.

## Left for the authors (new this round)

- **Freed space:** about half a column on page 8. Either use it (CONTEXT lists a compact Table 24 for the main text as blocked by space) or force Limitations back to page 9.
- **Setup:** the reason for the 4,096-token budget ("since they reason before answering") was cut; restore it if wanted.
- **Intro:** "Naming the false side helps on every benchmark that labels one" is benchmark-level. Results lists generator-level exceptions (gpt-6.1-terra on RAGuard; Qwen2.5-7B and minimal-prompt gpt-4o-mini on RAMDocs).
- **Abstract:** "depends on answer policy and model family" has no "coded after seeing data" tag; the intro has one.
- **Discussion:** says "break-even precision" where Results says "break-even direction precision".
- **Results:** the remaining "changes accuracy insignificantly" (H25/H26 sentence); "the hypothesis tables" has no `\ref`.
- **App. grid:** "twice" and "at most two points" have no macros.
