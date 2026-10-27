"""
Thin OpenAI wrapper.

- Uses the API's reported usage (prompt/completion tokens), never word counts.
- Cost uses PRICES below; state the price and date in the paper (reviewer comment 8).
- Every call is cached on disk by (model, messages, temperature, seed, max_tokens),
  so reruns are free and an interrupted run resumes where it stopped.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

# USD per 1M tokens. GPT-4o-mini list price; CONFIRM on openai.com/api/pricing and
# record the date you checked in the paper.
PRICES = {"gpt-4o-mini": {"input": 0.15, "output": 0.60}}
PRICES_CHECKED = "UNVERIFIED - fill in date checked"

CACHE_PATH = Path(os.getenv("CONFLICT_LLM_CACHE", Path(__file__).resolve().parents[2] / "runs" / "llm_cache.sqlite"))


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_s: float
    cached: bool


def price(model: str, pt: int, ct: int) -> float:
    p = PRICES.get(model)
    if p is None:
        raise KeyError(f"No price for {model}; add it to PRICES with the date checked.")
    return (pt * p["input"] + ct * p["output"]) / 1e6


class _Cache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT)")
        self.lock = threading.Lock()

    def get(self, k):
        with self.lock:
            r = self.db.execute("SELECT v FROM c WHERE k=?", (k,)).fetchone()
        return json.loads(r[0]) if r else None

    def put(self, k, v):
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO c VALUES (?,?)", (k, json.dumps(v)))
            self.db.commit()


class OpenAILLM:
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.1,
                 max_tokens: int = 256, cache_path: Path = CACHE_PATH):
        from openai import OpenAI
        self.client = OpenAI()  # reads OPENAI_API_KEY
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens
        self.cache = _Cache(cache_path)

    def __call__(self, prompt: str, seed: int = 0, json_mode: bool = False,
                 max_tokens: int | None = None) -> LLMResult:
        msgs = [{"role": "user", "content": prompt}]
        mt = max_tokens or self.max_tokens
        key = hashlib.sha256(json.dumps(
            [self.model, msgs, self.temperature, seed, mt, json_mode]).encode()).hexdigest()
        hit = self.cache.get(key)
        if hit:
            return LLMResult(**{**hit, "cached": True})

        kw = dict(model=self.model, messages=msgs, temperature=self.temperature,
                  seed=seed, max_tokens=mt)
        if json_mode:
            kw["response_format"] = {"type": "json_object"}
        t0 = time.perf_counter()
        resp = self._with_retry(kw)
        dt = time.perf_counter() - t0
        u = resp.usage
        res = dict(text=resp.choices[0].message.content or "",
                   prompt_tokens=u.prompt_tokens, completion_tokens=u.completion_tokens,
                   cost_usd=price(self.model, u.prompt_tokens, u.completion_tokens),
                   latency_s=dt)
        self.cache.put(key, res)
        return LLMResult(**res, cached=False)

    def _with_retry(self, kw, attempts: int = 6):
        from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
        for a in range(attempts):
            try:
                return self.client.chat.completions.create(**kw)
            except (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError):
                if a == attempts - 1:
                    raise
                time.sleep(min(60, 2 ** a))


class FakeLLM:
    """Deterministic stand-in for tests and dry runs. `responder(prompt) -> str`."""
    model = "fake"

    def __init__(self, responder=None, latency_s: float = 0.0):
        self.responder = responder or (lambda p: "Answer: unknown")
        self.latency_s = latency_s
        self.calls = 0

    def __call__(self, prompt: str, seed: int = 0, json_mode: bool = False, max_tokens=None) -> LLMResult:
        self.calls += 1
        txt = self.responder(prompt)
        pt, ct = len(prompt) // 4, len(txt) // 4
        return LLMResult(txt, pt, ct, 0.0, self.latency_s, False)
