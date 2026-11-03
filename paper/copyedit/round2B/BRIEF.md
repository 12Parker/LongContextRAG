# Copy-editing brief, Round 2B (sentence and paragraph craft): *Which Side Is Wrong? What Contradiction Flags Do to Retrieval-Augmented Generation Under Misinformation*

You are an expert technical editor for research papers in NLP (ACL Rolling Review, long paper). You are editing one section of the paper. This is a second copy-edit pass. A first pass of three rounds (A coherence, B sentence craft, C concision) was done before the paper's ninth and tenth revisions (§18, §19), which added and trimmed text since; read `paper/copyedit/roundA/TRIAGE.md`, `roundB/TRIAGE.md` and `roundC/TRIAGE.md` first, do not undo their fixes, and do not re-report their "Left for the authors" items. New or rewritten text (HealthContradict, the prospective test H32–H36, the judge re-score H38) has not been edited yet and needs the most attention. Round 2A (coherence and clarity) of this pass is done; also read `paper/copyedit/round2A/TRIAGE.md` and keep its fixes. Change how your section reads, **never what it says**.

**The paper's one idea:** a contradiction flag in RAG helps against misinformation only if it says which side is false, and then only when its detector's direction is right often enough (above a break-even precision that rises with the generator's accuracy); a neutral "these passages conflict" flag does not reliably help, and what a detector contributes is its verdict, which under the paper's string metric works about as well passed as an answer key (an outside-family judge favours the flag by about 2.5 points). A follow-rate account frozen in advance predicts the flag's effect in new detector-generator cells, but overstates harm on real health web pages. Audience: ACL reviewers who know RAG and knowledge conflicts but not this paper.

## Principles to apply

**Gopen and Swan, "The Science of Scientific Writing" (American Scientist, 1990).** Readers expect information in particular places.
- *Topic position*: begin a sentence with old, familiar information that links back, and with whose story it tells.
- *Stress position*: end a sentence with the new or most important information.
- Put the subject first and its verb soon after.
- Put the action in the verb.
- One point per sentence and per paragraph.

**Joseph Williams, *Style: Lessons in Clarity and Grace*.**
- Make the main characters (the methods, the system, the models, "we") the subjects.
- Express their actions as verbs, not nominalisations: "we measured", not "a measurement was performed".
- Prefer the active voice, unless the passive keeps a familiar subject in the topic position.
- Cut empty openers ("It is worth noting that", "In order to", "There are … that").

**Mensh and Kording, "Ten simple rules for structuring papers" (PLOS Computational Biology, 2017).**
- Give each paragraph context → content → conclusion, with one message.
- Use the same term for the same concept every time.
- The first sentence of each (sub)section says what it will show.

**Common practice.**
- Put compared numbers next to each other.
- Write parallel lists in parallel form.
- Do not start a sentence with a bare "this", "it" or "these": add the noun.

## What you may change

- Reorder clauses and sentences, and rewrite sentences to follow the principles.
- Split or join sentences.
- Turn runs of items into short lists, or short lists back into prose. (In the main body, sections 00–07, do not add new list environments or paragraph breaks: the body must end on page 8 and currently overflows by about four lines.)
- Edit lead-ins and captions written in your file.
- Remove words and redundant sentences.

**Your section must not get longer.** In the main body (parts 00–07), prefer edits that shorten.

## What you must not change

- **Numbers.** Every number, interval, percentage, count, p-value, date, price, version and identifier stays exactly as written, with the same unit and the same metric attached. Do not round, and do not drop any number. A script checks that the set of numbers in your file is unchanged; if it isn't, your edit is rejected. (If a pointer you would like to add contains a digit, list it as an issue instead.)
- **Number macros.** Almost every result is a LaTeX macro such as `\neuMlMu`, `\predEightR` or `\hThirtyOneAMin`, defined in `paper/generated/numbers.tex`. Treat each macro exactly like a number: keep every one, with the same metric, sign wording and sentence role. Watch macros whose value is negative: some sentences use an `a`-prefixed macro (absolute value, e.g. `\ahThirtyOneBMax`) after a verb like "lowers"; do not swap a verb ("lowers" ↔ "changes") without checking that the macro's sign still reads correctly. A second script checks macros.
- **Claims and hedging.** Do not strengthen or weaken any claim. Keep each of these qualifiers where it is:
  - "pre-registered"/"registered", "exploratory", "descriptive", "retrospective", "post hoc", "coded after seeing data";
  - "association", "hypothesis", "equivalent within ±3", "not significant", "significantly";
  - "about", "at most", "may", "suggests", "on average", "only".

  Never move a result from one of these categories to another. In this paper "significant" means significant under Holm; "equivalent" means a TOST passed. Do not use either word loosely.
- **Contributions and limitations.** Never cut a contribution claim or soften a limitation, even if it seems repetitive.
- **Table bodies.** Do not edit table rows, except to fix an obvious typo in a label.
- **Code, identifiers, paths, quoted prompts.** Anything in `\texttt{}`, verbatim, or quoted prompt text stays verbatim. Model names (`gpt-4o-mini`, `gpt-6.1-terra`, Claude Haiku 5.5, Qwen2.5-7B) stay exactly as written.
- **LaTeX structure.** Keep every `\label`, `\ref`, `\cite`/`\citep`/`\citet`, `\input`, `\includegraphics`, `\begin`/`\end`, `\footnote`, `\paragraph{}` head (you may reword its text to state the finding), inline math `$...$`, and hypothesis ID (H1–H38). Do not add citations. Keep `\ `, `~` and `--` usage; the file must still compile.
- **Headings.** Keep `\section`/`\subsection` text, so cross-references still work.
- Do not add new content, examples or results.
- Do not reuse a word that is a defined label in the paper (below: *directional*, *one-wrong*, *answer key*, *conflict question*, *clean*, *contradiction*, *ambiguity*, *oracle*, *robust*, *suspect*, *follow rate*, *break-even*) in a looser sense.

## How to work

1. Read the whole paper once, quickly, for context (read-only: do not edit it), and read the TRIAGE files named above so you don't undo their fixes. Its sections are `paper/sections/{abstract,intro,related,method,setup,results,discussion,limitations,appendix,app_grid}.tex` under `/Users/cameronparker/Code/School/DeepLearning/LongContextRAG/`. Look up a macro's value with `grep '\\newcommand{\\MACRO}' paper/generated/numbers.tex` when you need to know its sign or size.
2. Edit only your assigned file(s) in `paper/copyedit/round2B/`, in place. The untouched copy is in `orig/`. Before you finish, run both checks from `paper/copyedit/tools/` and get no problems:
   - `python3 ~/.claude/skills/paper-copyedit/scripts/check_preservation.py ../round2B --only <part>`
   - `python3 check_latex.py ../round2B --only <part>`
   Compare word counts (prose words after ≤ before).
3. Write `<part>.notes.md` next to your file, with:
   - the main kinds of change you made;
   - the word count before and after;
   - **Issues for the authors**: things you saw but must not fix, quoting the exact sentence;
   - **Coherence issues across sections**: give both locations.
4. Reply with one short paragraph: what you changed, the word counts, and how many issues you listed.

Do not edit `paper/sections/`, `paper/generated/`, or any file other than your part(s) and their notes. Do not run git commands that change anything. Do not compile the paper.

## Glossary of the paper's terms (use them consistently)

**Setup**
- *generator*: the LLM that answers the question from the passages (gpt-4o-mini is the main one; seven in all, from OpenAI, Anthropic and Qwen).
- *detector*: returns flagged passage pairs; a *directional* detector also names a *suspect* side per pair. Detectors: CAAG (heuristic), NLI (cross-encoder), LLM (one gpt-4o-mini call unless stated), Random (placebo at CAAG's flag rate), Oracle (gold pairs and gold misinformation side), inverted oracle (gold pairs, names the correct side).
- *synthetic detectors / the grid*: detectors built from gold labels with chosen precision p, recall r and direction accuracy d (36 cells).
- *format*: how detector output reaches the generator. *plain* (no flags); *instruction* (generic check-for-contradictions, v1/v2); *flag* or *neutral flag* ("possible contradiction", no side); *one-wrong flag* (says one side is misinformation, not which); *directional flag* (names the likely misinformation); *remove* (deletes the suspect passage); *single call*; *closed book*. Controls: *pick*, *no pairs*, *answer key* ("The answer X is false", no passage marked).
- *skeleton / prompt / answer policy*: *default* (asks for every valid answer), *minimal* (states no policy), *single* (exactly one correct answer), *forced* (single without "unknown"). *multi* variants of baselines ask for every valid answer.
- *verdict key* (H30): the detector's judgement passed as an answer key. *own-answer hint* (H27): the detector model's own plain answer as a hint.

**Data**
- RAMDocs (pilot 200 / confirmatory hold-out 300: 139 conflict, 126 clean, 35 excluded); ConflictBank (400 sampled: 300 conflict, 100 clean; one gold answer); *majority* variant (three wrong-object passages vs one correct); RAGuard (PolitiFact claims, Reddit posts with behavioural labels; its "oracle" is a label transfer, not a factual oracle); WikiContradict (genuine conflicts; reference answer is both sides); CONFLICTS (real search results, 237 scorable).
- *conflict question*: has a contradiction pair; *clean question*: has none. *contradiction* pair: one side misinformation, answers differ; *ambiguity*: both correct, different answers.

**Metrics and statistics**
- *accuracy*: answer includes every gold answer and no wrong answer (one-way whole-word matching is primary; *bidirectional* is the robustness metric). *robust*: also holds under the bidirectional metric.
- misinformation share, abstention ("unknown") share, selective accuracy (correct over answered).
- *significant*: Holm-adjusted p within the hypothesis's family. *equivalent*: TOST at ±3 points passed. CIs are bootstrap 95%, unadjusted.
- *pooled*: multilevel (random-effects) estimate across settings; *setting*: a generator × benchmark × prompt combination.
- *H31* (randomised direction): every conflict question flagged on its true pair, direction random; *right flag* names the false side, *wrong flag* names the correct side. *follow rates* f_R, f_W. *break-even direction precision* p* = −W/(R−W): a calibration for randomised flags on true pairs, not a decision rule.
- *crossed design / cells* (H24): six detectors × six generators on ConflictBank. *follow-rate account*: retrospective decomposition Δ̂ of each cell's effect; exploratory.
- *prevalence reweighting*: estimates at a 5% share of conflict questions.
- *registered*: H1–H9 before any confirmatory data; H10–H38 in nine later addenda, each before its runs (§19 holds H32–H38). Re-analyses are exploratory.
- *prospective test* (H32–H36, §19): ConflictBank-fresh, 400 unseen ConflictBank questions; *frozen prediction*: computed from detections and no-flag answers and committed before any flagged answer existed; 20 *fresh cells* (five detectors × four generators).
- *HealthContradict* (H37): yes/no health questions with one real web page per side; its page labels are used "as released" (a label transfer, not verified truth).
- *judge* / *judge-strict*: the outside-family LLM judge (gemini-3.5-flash-lite) of §17.3; *matcher* / *string metric*: the paper's one-way string match. H38: full judge re-score of the pooled H30a/H23 equivalence tests (equivalence holds under the matcher, not under the judge).
