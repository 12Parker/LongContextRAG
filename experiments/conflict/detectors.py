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


class LLMDetector:
    """Single-call LLM detector (comparable to ConflictRAG's 'CoT Detection' baseline)."""
    def __init__(self, llm, max_chars: int = 1500, seed: int = 0, name: str = "llm"):
        self.llm, self.max_chars, self.seed, self.name = llm, max_chars, seed, name

    def detect(self, ex: Example) -> DetectorOutput:
        body = "\n\n".join(f"[{k}] {p[: self.max_chars]}" for k, p in enumerate(ex.passages))
        prompt = LLM_DETECT_PROMPT.format(n=len(ex.passages), question=ex.question, passages=body)
        r = self.llm(prompt, seed=self.seed, json_mode=True, max_tokens=512)
        try:
            items = json.loads(r.text).get("pairs", [])
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", r.text, re.S)
            items = json.loads(m.group(0)).get("pairs", []) if m else []
        pairs, scores, n = [], {}, len(ex.passages)
        for it in items:
            try:
                i, j = sorted((int(it["i"]), int(it["j"])))
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= i < j < n and it.get("relation", "contradiction") == "contradiction":
                pairs.append(Detection(i, j, 1.0, "llm"))
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
        pairs = [Detection(i, j, 1.0, "random") for i in range(n) for j in range(i + 1, n)
                 if rng.random() < self.rate]
        return DetectorOutput(pairs, {(d.i, d.j): 1.0 for d in pairs})


class OracleDetector:
    """Upper bound from gold pair labels (RAMDocs). Unavailable for question-level-only data."""
    name = "oracle"

    def detect(self, ex: Example) -> DetectorOutput:
        if ex.gold_pairs is None:
            raise ValueError(f"{ex.dataset} has no pair-level gold labels; oracle not defined.")
        pairs = [Detection(i, j, 1.0, "oracle") for (i, j), lab in ex.gold_pairs.items() if lab == "contradiction"]
        return DetectorOutput(pairs, {(d.i, d.j): 1.0 for d in pairs})


class NoDetector:
    name = "none"

    def detect(self, ex: Example) -> DetectorOutput:
        return DetectorOutput([])
