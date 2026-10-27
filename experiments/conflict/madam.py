"""
MADAM-RAG (Wang et al. 2025) ported from RAMDocs/run_madam_rag.py
(github.com/HanNight/RAMDocs @ d44454c) so it runs on GPT-4o-mini through the same
cached client as every other condition. Prompts and the early-stopping rule are
copied verbatim; only `call_llm` changes. The public script uses greedy decoding and
128 new tokens per call, so pass an LLM built with temperature=0.0 and max_tokens=128
to match it.
"""
from __future__ import annotations

from .metrics import normalize_answer

AGENT_WITH_HISTORY = """You are an agent reading a document to answer a question.

Question: {query}
Document: {document}

The following reponses are from other agents as additional information.
{history}
Answer the question based on the document and other agents' response. Provide your answer and a step-by-step reasoning explanation.  
Please follow the format: 'Answer: {{}}. Explanation: {{}}.''"""

AGENT = """You are an agent reading a document to answer a question.

Question: {query}
Document: {document}

Answer the question based only on this document. Provide your answer and a step-by-step reasoning explanation.
Please follow the format: 'Answer: {{}}. Explanation: {{}}.''"""

AGGREGATOR = """You are an aggregator reading answers from multiple agents.

If there are multiple answers, please provide all possible correct answers and also provide a step-by-step reasoning explanation. If there is no correct answer, please reply 'unknown'.
Please follow the format: 'All Correct Answers: []. Explanation: {{}}.'

The following are examples:
Question: In which year was Michael Jordan born?
Agent responses:
Agent 1: Answer: 1963. Explanation: The document clearly states that Michael Jeffrey Jordan was born on February 17, 1963. 
Agent 2: Answer: 1956. Explanation: The document states that Michael Irwin Jordan was born on February 25, 1956. However, it's important to note that this document seems to be about a different Michael Jordan, who is an American scientist, not the basketball player. The other agents' responses do not align with the information provided in the document.
Agent 3: Answer: 1998. Explanation: The According to the document provided, Michael Jeffrey Jordan was born on February 17, 1998.
Agent 4: Answer: Unknown. Explanation: The provided document focuses on Jordan's college and early professional career, mentioning his college championship in 1982 and his entry into the NBA in 1984, but it does not include information about his birth year.
All Correct Answers: ["1963", "1956"]. Explanation: Agent 1 is talking about the basketball player Michael Jeffrey Jordan, who was born on Februray 17, 1963, so 1963 is correct. Agent 2 is talking about another person named Michael Jordan, who is an American scientist, and he was born in 1956. Therefore, the answer 1956 from Agent 2 is also correct. Agent 3 provides an error stating Michael Jordan's birth year as 1998, which is incorrect. Based on the correct information from Agent 1, Michael Jeffrey Jordan was born on February 17, 1963. Agent 4 does not provide any useful information.

Question: {query}
Agent responses:
{joined}
"""


def _split(resp: str) -> tuple[str, str]:
    a = resp[resp.find("Answer: ") + len("Answer: "): resp.find("Explanation")].strip()
    e = resp[resp.find("Explanation: ") + len("Explanation: "):]
    return a, e


def madam_rag(query: str, documents: list[str], llm, seed: int = 0, num_rounds: int = 3) -> dict:
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0, "calls": 0, "latency_s": 0.0}

    def call(prompt):
        r = llm(prompt, seed=seed)
        usage["prompt_tokens"] += r.prompt_tokens; usage["completion_tokens"] += r.completion_tokens
        usage["cost_usd"] += r.cost_usd; usage["calls"] += 1; usage["latency_s"] += r.latency_s
        return r.text.strip()

    def aggregate(outputs):
        joined = "\n".join(f"Agent {i+1}: {r}" for i, r in enumerate(outputs))
        return call(AGGREGATOR.format(query=query, joined=joined))

    records = {"round1": {"answers": [], "explanations": []}}
    outputs = []
    for doc in documents:
        resp = call(AGENT.format(query=query, document=doc))
        a, e = _split(resp)
        records["round1"]["answers"].append(a); records["round1"]["explanations"].append(e)
        outputs.append(resp)
    records["round1"]["aggregation"] = aggregate(outputs)
    final = records["round1"]["aggregation"]  # used if num_rounds == 1

    for t in range(1, num_rounds):
        key = f"round{t+1}"
        records[key] = {"answers": [], "explanations": []}
        new = []
        for i, doc in enumerate(documents):
            hist = "\n".join(f"Agent {j+1}: {outputs[j]}" for j in range(len(documents)) if j != i)
            resp = call(AGENT_WITH_HISTORY.format(query=query, document=doc, history=hist))
            a, e = _split(resp)
            records[key]["answers"].append(a); records[key]["explanations"].append(e)
            new.append(resp)
        outputs = new
        cur = [normalize_answer(a) for a in records[key]["answers"]]
        prev = [normalize_answer(a) for a in records[f"round{t}"]["answers"]]
        if all(c in p or p in c for c, p in zip(cur, prev)):
            final = records[f"round{t}"]["aggregation"]
            break
        records[key]["aggregation"] = aggregate(outputs)
        final = records[key]["aggregation"]

    return {"final": final or "", "records": records, "usage": usage}
