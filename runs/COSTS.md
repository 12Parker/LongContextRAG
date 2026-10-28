# API spend ledger (2026-10-28)

- Real API calls: 300,262
- Tokens: 427,015,396 prompt, 55,941,347 completion
- **Total spend: $185.14** of a $200 budget
- Prices: 2026-10-27, https://developers.openai.com/api/docs/pricing (gpt-4o-mini: $0.15 / $0.075 cached / $0.60; gpt-4.1: $2.00 / $0.50 cached / $8.00)

Computed from runs/llm_cache.sqlite (one entry per real call; cache hits cost nothing).
Calls priced before cached-token pricing was added are counted at the full input rate.
