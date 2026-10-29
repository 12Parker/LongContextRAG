"""
Contradiction detectors behind one interface:

    detector.detect(example) -> DetectorOutput(pairs=[Detection(i, j, score, type)], ...)

`pairs` are the flagged pairs (i < j). `scores` holds a continuous score for every
pair the detector evaluated, so precision/recall curves, AUROC and threshold sweeps
can be computed offline without re-running the detector.
"""
from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, field
from typing import Protocol

from .data import Example


@dataclass
class Detection:
    i: int
    j: int
    score: float
    type: str = "contradiction"
    # passages in (i, j) the detector believes are the misinformation side; empty if it
    # cannot tell (CAAG, NLI). Used by the flag_directional and remove_suspect modes.
    suspects: list[int] = field(default_factory=list)


@dataclass
class DetectorOutput:
    pairs: list[Detection]
    scores: dict[tuple[int, int], float] = field(default_factory=dict)
    latency_s: float = 0.0
    cost_usd: float = 0.0
    extra: dict = field(default_factory=dict)


class Detector(Protocol):
    name: str
    def detect(self, ex: Example) -> DetectorOutput: ...


# --------------------------------------------------------------------------- embeddings
class STEmbedder:
    """sentence-transformers model exposing the LangChain `embed_documents` interface
    that the original CAAG detector expects. Default matches the repo (all-MiniLM-L6-v2)."""
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.m.encode(list(texts), normalize_embeddings=False).tolist()


# --------------------------------------------------------------------------- CAAG
class CAAGDetector:
    """The paper's heuristic detector, unchanged (see caag_core.py).

    Runs once with both gates fully open and records, for every pair, the semantic
    similarity and the raw signal total. Flags are then applied with (theta_s, theta_c)
    exactly as the original code does: flag iff sim >= theta_s and total*sim >= theta_c.
    This makes the theta sweep (reviewer comment 6) free.
    """
    def __init__(self, embedder, theta_s: float = 0.65, theta_c: float = 0.35,
                 use_negation=True, use_entity=True, use_temporal=True, name: str = "caag"):
        from .caag_core import ContradictionDetector
        self.theta_s, self.theta_c, self.name = theta_s, theta_c, name
        self.core = ContradictionDetector(
            embeddings=embedder, similarity_threshold=-2.0, contradiction_threshold=float("-inf"),
            use_semantic=True, use_negation=use_negation, use_entity=use_entity, use_temporal=use_temporal)

    def detect(self, ex: Example) -> DetectorOutput:
        t0 = time.perf_counter()
        raw = self.core.detect_contradictions([{"text": p} for p in ex.passages])
        dt = time.perf_counter() - t0
        rows, scores, pairs = {}, {}, []
        for cp in raw:
            i, j = sorted((cp.doc_idx_1, cp.doc_idx_2))
            sim = cp.evidence.get("semantic_similarity", 0.0)
            signals = [s for s in ("negation", "entity", "temporal")
                       if (s == "negation" and "negation" in cp.evidence)
                       or (s == "entity" and "entity_conflicts" in cp.evidence)
                       or (s == "temporal" and "temporal_conflicts" in cp.evidence)]
            rows[(i, j)] = {"sim": float(sim), "final": float(cp.contradiction_score),
                            "type": cp.contradiction_type, "signals": signals}
            gated = cp.contradiction_score if sim >= self.theta_s else 0.0
            scores[(i, j)] = float(gated)
            if sim >= self.theta_s and cp.contradiction_score >= self.theta_c:
                pairs.append(Detection(i, j, float(cp.contradiction_score), cp.contradiction_type))
        return DetectorOutput(pairs, scores, dt, 0.0, {"caag_rows": {f"{i},{j}": v for (i, j), v in rows.items()}})


# --------------------------------------------------------------------------- NLI
class NLIDetector:
    """Off-the-shelf NLI cross-encoder (the 'NLI-Filter' baseline reviewers will expect).

    NLI is asymmetric, so both directions are scored and the max contradiction
    probability is used. Passages are truncated to `max_chars`; NLI models are trained
    on sentence pairs, so long-passage scores are a known weakness (report it).
    """
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base",
                 threshold: float = 0.5, max_chars: int = 1500, batch_size: int = 32,
                 name: str = "nli"):
        from sentence_transformers import CrossEncoder
        self.ce = CrossEncoder(model_name)
        self.threshold, self.max_chars, self.bs, self.name = threshold, max_chars, batch_size, name
        self.c_idx = self._contradiction_index()

    def _contradiction_index(self) -> int:
        id2label = getattr(getattr(self.ce, "config", None), "id2label", None) or {}
        for k, v in id2label.items():
            if str(v).lower().startswith("contra"):
                return int(k)
        raise RuntimeError(f"Could not find a contradiction label in {id2label}")

    def detect(self, ex: Example) -> DetectorOutput:
        ps = [p[: self.max_chars] for p in ex.passages]
        idx = [(i, j) for i in range(len(ps)) for j in range(i + 1, len(ps))]
        t0 = time.perf_counter()
        if not idx:
            return DetectorOutput([], {}, 0.0)
        inputs = [(ps[i], ps[j]) for i, j in idx] + [(ps[j], ps[i]) for i, j in idx]
        probs = self.ce.predict(inputs, apply_softmax=True, batch_size=self.bs, show_progress_bar=False)
        dt = time.perf_counter() - t0
        n = len(idx)
        scores, pairs = {}, []
        for k, (i, j) in enumerate(idx):
            s = float(max(probs[k][self.c_idx], probs[n + k][self.c_idx]))
            scores[(i, j)] = s
            if s >= self.threshold:
                pairs.append(Detection(i, j, s, "nli"))
        return DetectorOutput(pairs, scores, dt)


# --------------------------------------------------------------------------- LLM
LLM_DETECT_PROMPT = """You will see a question and {n} numbered passages retrieved for it.
Identify every pair of passages that CONTRADICT each other about the answer to the question,
i.e. they cannot both be true about the same entity at the same time. Passages that are about
different entities with the same name, give complementary details, or describe different times
are NOT contradictions; note them as "ambiguity", "complementary" or "temporal" instead.

Question: {question}

{passages}

Return JSON only: {{"pairs": [{{"i": <int>, "j": <int>, "relation": "contradiction|ambiguity|complementary|temporal"}}]}}
Use the passage numbers shown. Return {{"pairs": []}} if there are none."""


LLM_DETECT_DIRECTIONAL_PROMPT = """You will see a question and {n} numbered passages retrieved for it.
Some passages may contain misinformation: false claims about the answer to the question.
Identify every pair of passages that CONTRADICT each other about the answer to the question,
i.e. they cannot both be true about the same entity at the same time. Passages that are about
different entities with the same name, give complementary details, or describe different times
are NOT contradictions; note them as "ambiguity", "complementary" or "temporal" instead.

For each contradiction, say which passage of the pair is more likely the misinformation, judging
by agreement with the other passages and by plausibility: its number, "both" if both are likely
false, or "unclear" if you cannot tell.

Question: {question}

{passages}

Return JSON only: {{"pairs": [{{"i": <int>, "j": <int>, "relation": "contradiction|ambiguity|complementary|temporal", "misinformation": <int>|"both"|"unclear"}}]}}
Use the passage numbers shown. "misinformation" is needed only for contradictions. Return {{"pairs": []}} if there are none."""


class LLMDetector:
    """Single-call LLM detector (comparable to ConflictRAG's 'CoT Detection' baseline).

    With `directional=True` it also says which side of each contradiction is likely the
    misinformation, which fills `Detection.suspects` for flag_directional / remove_suspect.
    """
    def __init__(self, llm, max_chars: int = 1500, seed: int = 0, directional: bool = False,
                 name: str | None = None):
        self.llm, self.max_chars, self.seed, self.directional = llm, max_chars, seed, directional
        self.name = name or ("llm_dir" if directional else "llm")

    def detect(self, ex: Example) -> DetectorOutput:
        body = "\n\n".join(f"[{k}] {p[: self.max_chars]}" for k, p in enumerate(ex.passages))
        template = LLM_DETECT_DIRECTIONAL_PROMPT if self.directional else LLM_DETECT_PROMPT
        prompt = template.format(n=len(ex.passages), question=ex.question, passages=body)
        r = self.llm(prompt, seed=self.seed, json_mode=True, max_tokens=1024 if self.directional else 512)
        try:
            items = json.loads(r.text).get("pairs", [])
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", r.text, re.S)
            try:
                items = json.loads(m.group(0)).get("pairs", []) if m else []
            except json.JSONDecodeError:
                items = []
        pairs, scores, n = [], {}, len(ex.passages)
        for it in items if isinstance(items, list) else []:
            try:
                i, j = sorted((int(it["i"]), int(it["j"])))
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= i < j < n and it.get("relation", "contradiction") == "contradiction" and (i, j) not in scores:
                suspects = []
                if self.directional:
                    mis = it.get("misinformation")
                    if str(mis).lower() == "both":
                        suspects = [i, j]
                    else:
                        try:
                            suspects = [int(mis)] if int(mis) in (i, j) else []
                        except (TypeError, ValueError):
                            suspects = []
                # neutral type: it is rendered into the prompt as "<type> conflict"
                pairs.append(Detection(i, j, 1.0, "contradiction", suspects))
                scores[(i, j)] = 1.0
        return DetectorOutput(pairs, scores, r.latency_s, r.cost_usd, {"raw": r.text, "relations": items})


# --------------------------------------------------------------------------- controls
class RandomDetector:
    """Placebo: flags each pair independently with probability `rate`
    (set it to CAAG's observed per-pair flag rate). Seeded per example."""
    def __init__(self, rate: float, seed: int = 0, name: str = "random"):
        self.rate, self.seed, self.name = rate, seed, name

    def detect(self, ex: Example) -> DetectorOutput:
        rng = random.Random(f"{self.seed}-{ex.id}")
        n = len(ex.passages)
        # type is rendered into the prompt ("<type> conflict"), so controls use a neutral
        # label; "random conflict" / "oracle conflict" would tell the model what it is seeing
        pairs = [Detection(i, j, 1.0, "contradiction") for i in range(n) for j in range(i + 1, n)
                 if rng.random() < self.rate]
        return DetectorOutput(pairs, {(d.i, d.j): 1.0 for d in pairs})


class OracleDetector:
    """Upper bound from gold pair labels (RAMDocs). Unavailable for question-level-only data."""
    name = "oracle"

    def detect(self, ex: Example) -> DetectorOutput:
        if ex.gold_pairs is None:
            raise ValueError(f"{ex.dataset} has no pair-level gold labels; oracle not defined.")
        types = ex.meta.get("doc_types", [])
        pairs = [Detection(i, j, 1.0, "contradiction",
                           [k for k in (i, j) if k < len(types) and types[k] == "misinfo"])
                 for (i, j), lab in ex.gold_pairs.items() if lab == "contradiction"]
        return DetectorOutput(pairs, {(d.i, d.j): 1.0 for d in pairs})


class InvertedOracleDetector(OracleDetector):
    """Adversarial control: the gold contradiction pairs, but the suspect is always the side that
    is NOT misinformation (PREREGISTRATION.md §9.5)."""
    name = "oracle_inv"

    def detect(self, ex: Example) -> DetectorOutput:
        out = super().detect(ex)
        for d in out.pairs:
            d.suspects = [k for k in (d.i, d.j) if k not in d.suspects]
        return out


def synth_name(precision: float, recall: float, pool: str = "all", direction: float | None = None) -> str:
    return f"synth-p{precision:g}-r{recall:g}-{pool}" + ("" if direction is None else f"-d{direction:g}")


def parse_synth_name(name: str) -> tuple[float, float, str, float | None]:
    _, p, r, pool, *d = name.split("-")
    return float(p[1:]), float(r[1:]), pool, (float(d[0][1:]) if d else None)


class SyntheticDetector:
    """Flags with a chosen pair-level precision and recall, built from gold pair labels.

    Each true contradiction pair is flagged with probability `recall`. False flags are drawn
    from non-contradiction pairs whose gold relation is in `pool` ("all" = any of them) at a
    single per-pair rate, set over the whole example list so that expected dataset-level
    precision equals `precision`. Because the rate is dataset-wide, conflict-free questions get
    false flags too, which is what H3 needs.

    Every pair gets one uniform draw keyed by (seed, example, pair) and shared across grid
    cells, so cells are nested: a higher-recall cell flags a superset of a lower one's true
    pairs, and a lower-precision cell a superset of the false ones. Grid comparisons stay paired.

    If the pool is too small to reach the target precision, the rate is capped at 1 and the
    achieved precision is higher than requested; `capped` records this.

    `direction` (None = no direction) sets `suspects` for the flag_directional and
    remove_suspect modes. A true flag points at the gold misinformation side(s) with
    probability `direction`; otherwise it points at the other passage (for a pair where both
    sides are misinformation, at one of them only). A false flag has no misinformation side, so
    it points at one passage chosen at random, independent of `direction`. Direction draws are
    shared across cells too, so a higher-accuracy cell gets right every pair a lower one does.
    """
    RELATIONS = ("ambiguity", "unrelated", "consistent")

    def __init__(self, examples: list[Example], precision: float, recall: float,
                 pool: str = "all", seed: int = 0, direction: float | None = None, name: str | None = None):
        if not 0 < precision <= 1 or not 0 <= recall <= 1:
            raise ValueError(f"need 0 < precision <= 1 and 0 <= recall <= 1, got {precision}, {recall}")
        if direction is not None and not 0 <= direction <= 1:
            raise ValueError(f"need 0 <= direction <= 1, got {direction}")
        self.pool = set(self.RELATIONS) if pool == "all" else {pool}
        if not self.pool <= set(self.RELATIONS):
            raise ValueError(f"pool must be 'all' or one of {self.RELATIONS}, got {pool!r}")
        n_gold = n_pool = 0
        for ex in examples:
            if ex.gold_pairs is None:
                raise ValueError(f"{ex.dataset} has no pair-level gold labels; synthetic detector not defined.")
            for lab in ex.gold_pairs.values():
                n_gold += lab == "contradiction"
                n_pool += lab in self.pool
        want_fp = recall * n_gold * (1 - precision) / precision
        self.capped = want_fp > n_pool
        self.fp_rate = min(1.0, want_fp / n_pool) if n_pool else 0.0
        self.precision, self.recall, self.seed, self.direction = precision, recall, seed, direction
        self.name = name or synth_name(precision, recall, pool, direction)

    def _u(self, ex: Example, i: int, j: int, tag: str = "") -> float:
        return random.Random(f"{self.seed}-{ex.id}-{i}-{j}{tag}").random()

    def _suspects(self, ex: Example, i: int, j: int, lab: str) -> list[int]:
        if self.direction is None:
            return []
        pick = [i, j][self._u(ex, i, j, "-side") < 0.5]
        if lab != "contradiction":
            return [pick]
        types = ex.meta.get("doc_types", [])
        true = [k for k in (i, j) if k < len(types) and types[k] == "misinfo"]
        if self._u(ex, i, j, "-dir") < self.direction:
            return true
        wrong = [k for k in (i, j) if k not in true]
        return wrong or [pick]

    def detect(self, ex: Example) -> DetectorOutput:
        pairs = []
        for (i, j), lab in sorted(ex.gold_pairs.items()):
            u = self._u(ex, i, j)
            if (lab == "contradiction" and u < self.recall) or (lab in self.pool and u < self.fp_rate):
                pairs.append(Detection(i, j, 1.0, "contradiction", self._suspects(ex, i, j, lab)))
        return DetectorOutput(pairs, {(d.i, d.j): 1.0 for d in pairs},
                              extra={"fp_rate": self.fp_rate, "capped": self.capped})


class NoDetector:
    name = "none"

    def detect(self, ex: Example) -> DetectorOutput:
        return DetectorOutput([])


def suspect_set(out: DetectorOutput) -> frozenset[int]:
    """The passages a directional detector marks as misinformation, over its contradiction pairs."""
    return frozenset(k for d in out.pairs if d.type == "contradiction" for k in d.suspects)


class AgreeDetector:
    """§17 agreement gate: pass on the primary detector's output only when a second, different detector marks
    the same set of passages as misinformation; otherwise flag nothing (the generator gets the plain prompt).
    Both detectors are always called, so cost and latency are the sum of the two."""
    def __init__(self, primary, secondary, name: str = "agree"):
        self.primary, self.secondary, self.name = primary, secondary, name

    def detect(self, ex: Example) -> DetectorOutput:
        a, b = self.primary.detect(ex), self.secondary.detect(ex)
        ok = bool(suspect_set(a)) and suspect_set(a) == suspect_set(b)
        return DetectorOutput(a.pairs if ok else [], a.scores if ok else {}, a.latency_s + b.latency_s,
                              a.cost_usd + b.cost_usd,
                              {"agree": ok, "primary": a.extra, "secondary": b.extra,
                               "secondary_pairs": [vars(d) for d in b.pairs]})


LLM_ANSWER_JUDGE_PROMPT = """You will see a question and {n} numbered passages retrieved for it.
The passages may disagree about the answer, and some may be false or out of date.
List each distinct answer to the question that the passages support. Then say which of those answers, if any, is most
likely false or no longer true, judging by the passages and your own knowledge. Give none if you cannot tell or if the
passages agree.

Question: {question}

{passages}

Return JSON only: {{"answers": [<answer strings>], "false": [<answer strings from "answers">]}}"""


class LLMAnswerJudge:
    """Answer-level verdict: names the false answer(s) directly, without pointing at passages. Used with the
    verdict_key prompt mode, which renders 'The answer "X" is false.' for each one; no passage is marked."""
    name = "llm_ans"

    def __init__(self, llm, max_chars: int = 1500, seed: int = 0):
        self.llm, self.max_chars, self.seed = llm, max_chars, seed

    def detect(self, ex: Example) -> DetectorOutput:
        body = "\n\n".join(f"[{k}] {p[: self.max_chars]}" for k, p in enumerate(ex.passages))
        r = self.llm(LLM_ANSWER_JUDGE_PROMPT.format(n=len(ex.passages), question=ex.question, passages=body),
                     seed=self.seed, json_mode=True, max_tokens=512)
        try:
            obj = json.loads(r.text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", r.text, re.S)
            try:
                obj = json.loads(m.group(0)) if m else {}
            except json.JSONDecodeError:
                obj = {}
        false = [str(a) for a in obj.get("false", []) if isinstance(a, (str, int, float)) and str(a).strip()]
        return DetectorOutput([], {}, r.latency_s, r.cost_usd,
                              {"raw": r.text, "false_answers": false, "answers": obj.get("answers", [])})
