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

# USD per 1M tokens, standard tier. "cached_input" applies to prompt tokens the API reports as
# served from its prompt cache (usage.prompt_tokens_details.cached_tokens).
PRICES = {"gpt-4o-mini": {"input": 0.15, "cached_input": 0.075, "output": 0.60},
          "gpt-4.1": {"input": 2.00, "cached_input": 0.50, "output": 8.00},
          # current reasoning models (§12): reasoning tokens are billed as output
          "gpt-6-luna": {"input": 0.10, "cached_input": 0.01, "output": 0.50},
          "gpt-6.1-terra": {"input": 1.00, "cached_input": 0.05, "output": 5.00}}
# Reasoning models take max_completion_tokens (reasoning tokens count against it) and only their default
# temperature; they get a larger output budget so reasoning cannot crowd out the answer (§12).
REASONING_MODELS = {"gpt-6-luna", "gpt-6.1-terra"}
# Claude through Anthropic's OpenAI-compatible endpoint (§13): no temperature parameter (deprecated for these
# models), max_tokens as usual, and no json_object response format (the detector falls back to its regex parse)
PRICES.update({"claude-haiku-5-5": {"input": 0.10, "cached_input": 0.01, "output": 0.50},
               "claude-sonnet-5-5": {"input": 2.00, "cached_input": 0.10, "output": 10.00}})
PROVIDERS = {"claude-": ("https://api.anthropic.com/v1/", "ANTHROPIC_API_KEY"),
             "gemini-": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GOOGLE_API_KEY")}
# §17.3 judge, outside every generator's family; thinking off. Checked 2026-10-29, https://ai.google.dev/gemini-api/docs/pricing
PRICES.update({"gemini-3.5-flash-lite": {"input": 0.30, "output": 2.50}})
NO_THINKING_MODELS = {"gemini-3.5-flash-lite"}
NO_TEMPERATURE_MODELS = {"claude-haiku-5-5", "claude-sonnet-5-5"}
REASONING_MAX_TOKENS = 4096
PRICES_CHECKED = ("2026-10-27, https://developers.openai.com/api/docs/pricing "
                  "(gpt-4o-mini: $0.15 / $0.075 cached / $0.60; gpt-4.1: $2.00 / $0.50 cached / $8.00)")

CACHE_PATH = Path(os.getenv("CONFLICT_LLM_CACHE", Path(__file__).resolve().parents[2] / "runs" / "llm_cache.sqlite"))


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_s: float
    cached: bool
    cached_prompt_tokens: int = 0     # of prompt_tokens, served from OpenAI's prompt cache


def price(model: str, pt: int, ct: int, cached_pt: int = 0) -> float:
    p = PRICES.get(model)
    if p is None:
        raise KeyError(f"No price for {model}; add it to PRICES with the date checked.")
    cached_pt = min(cached_pt, pt)
    return ((pt - cached_pt) * p["input"] + cached_pt * p.get("cached_input", p["input"])
            + ct * p["output"]) / 1e6


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


def register_price(model: str, input_per_m: float, output_per_m: float,
                   cached_input_per_m: float | None = None) -> None:
    """Add a price for a model served elsewhere (e.g. an open-weights model). Record the source
    and date in the run's config (run.py stores --price-* there)."""
    PRICES[model] = {"input": input_per_m, "output": output_per_m,
                     "cached_input": input_per_m if cached_input_per_m is None else cached_input_per_m}


class OpenAILLM:
    """Chat-completions client. With `base_url` it talks to any OpenAI-compatible server
    (vLLM, Together, OpenRouter, ...), reading the key from `api_key_env`."""
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.1,
                 max_tokens: int = 256, cache_path: Path = CACHE_PATH,
                 base_url: str | None = None, api_key_env: str = "OPENAI_API_KEY"):
        from openai import OpenAI
        for prefix, (url, env) in PROVIDERS.items():
            if model.startswith(prefix) and base_url is None:
                base_url, api_key_env = url, env
        key = os.getenv(api_key_env)
        if base_url and not key:
            raise RuntimeError(f"{api_key_env} is not set (needed for {base_url})")
        self.client = OpenAI(base_url=base_url, api_key=key) if base_url else OpenAI()
        self.model, self.temperature, self.max_tokens, self.base_url = model, temperature, max_tokens, base_url
        self.cache = _Cache(cache_path)
        price(model, 0, 0)  # fail now, not after the first call, if there is no price

    def __call__(self, prompt: str, seed: int = 0, json_mode: bool = False,
                 max_tokens: int | None = None, system: str | None = None) -> LLMResult:
        msgs = [{"role": "user", "content": prompt}]
        if system is not None:   # §17.2; without one, messages (and cache keys) are unchanged
            msgs = [{"role": "system", "content": system}] + msgs
        mt = max_tokens or self.max_tokens
        parts = [self.model, msgs, self.temperature, seed, mt, json_mode]
        if self.base_url:  # same model name on another server is a different model
            parts.append(self.base_url)
        if self.model in NO_TEMPERATURE_MODELS:   # the effective output budget differs from mt (see below)
            parts.append(f"max_tokens={max(mt, REASONING_MAX_TOKENS)}")
        key = hashlib.sha256(json.dumps(parts).encode()).hexdigest()
        hit = self.cache.get(key)
        if hit:
            return LLMResult(**{**hit, "cached": True})

        if self.model in REASONING_MODELS:
            kw = dict(model=self.model, messages=msgs, seed=seed,
                      max_completion_tokens=max(mt, REASONING_MAX_TOKENS))
        elif self.model in NO_TEMPERATURE_MODELS:   # they also think before answering: same budget as reasoning models
            kw = dict(model=self.model, messages=msgs, seed=seed, max_tokens=max(mt, REASONING_MAX_TOKENS))
        else:
            kw = dict(model=self.model, messages=msgs, temperature=self.temperature,
                      seed=seed, max_tokens=mt)
        if json_mode and self.model not in NO_TEMPERATURE_MODELS:
            kw["response_format"] = {"type": "json_object"}
        if self.model in NO_THINKING_MODELS:   # Gemini's endpoint rejects `seed`; temperature 0 instead
            kw["reasoning_effort"] = "minimal"; kw.pop("seed", None)
        t0 = time.perf_counter()
        try:
            resp = self._with_retry(kw)
        except _Degenerate as e:
            # Ollama aborts a generation that loops on one token. It is a degenerate output, like a
            # truncated one, and is scored as it is (no final-answer line). Cached so reruns agree.
            res = dict(text=f"{ABORTED_PREFIX} {e}]", prompt_tokens=0, completion_tokens=mt, cost_usd=0.0,
                       latency_s=time.perf_counter() - t0, cached_prompt_tokens=0)
            self.cache.put(key, res)
            return LLMResult(**res, cached=False)
        dt = time.perf_counter() - t0
        u = resp.usage
        cpt = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
        res = dict(text=resp.choices[0].message.content or "",
                   prompt_tokens=u.prompt_tokens, completion_tokens=u.completion_tokens,
                   cost_usd=price(self.model, u.prompt_tokens, u.completion_tokens, cpt),
                   latency_s=dt, cached_prompt_tokens=cpt)
        self.cache.put(key, res)
        return LLMResult(**res, cached=False)

    def _with_retry(self, kw, attempts: int = 6):
        from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
        for a in range(attempts):
            try:
                return self.client.chat.completions.create(**kw)
            except (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError) as e:
                if self.base_url and "repeat limit" in str(e):
                    raise _Degenerate("token repeat limit reached") from e
                if a == attempts - 1:
                    raise
                time.sleep(min(60, 2 ** a))


ABORTED_PREFIX = "[generation aborted by the server:"


class _Degenerate(Exception):
    pass


class FakeLLM:
    """Deterministic stand-in for tests and dry runs. `responder(prompt) -> str`."""
    model = "fake"

    def __init__(self, responder=None, latency_s: float = 0.0):
        self.responder = responder or (lambda p: "Answer: unknown")
        self.latency_s = latency_s
        self.calls = 0

    def __call__(self, prompt: str, seed: int = 0, json_mode: bool = False, max_tokens=None, system=None) -> LLMResult:
        self.calls += 1
        txt = self.responder(prompt)
        pt, ct = len(prompt) // 4, len(txt) // 4
        return LLMResult(txt, pt, ct, 0.0, self.latency_s, False)
