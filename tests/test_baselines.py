"""Astute RAG / TrustRAG reimplementations (PREREGISTRATION.md §10) and the ConflictBank scorer."""
import numpy as np

from experiments.conflict.baselines import (ASTUTE_STEP3_MULTI, TRUST_FINAL_MULTI_SUFFIX, astute_rag,
                                            trustrag, trustrag_filter)
from experiments.conflict.data import Example
from experiments.conflict.llm import FakeLLM
from experiments.conflict.metrics import score


def test_astute_two_calls_and_answer_tag():
    seen = []

    def resp(p):
        seen.append(p)
        return "I don't know." if p.startswith("Generate a document") else "groups...\n<ANSWER> Paris </ANSWER>"
    llm = FakeLLM(resp)
    out = astute_rag("Capital of France?", ["Paris is the capital.", "Lyon is the capital."], llm, multi=True)
    assert llm.calls == 2 and out["usage"]["calls"] == 2
    assert out["n_internal"] == 0 and "Memorized Document" not in seen[1]     # "I don't know" -> m = 0
    assert ASTUTE_STEP3_MULTI in seen[1]
    assert out["final"].endswith("Final answer(s): Paris")


def test_trustrag_filter_keeps_all_without_ngram_overlap():
    calls = []
    embed = lambda texts: calls.append(1) or np.eye(len(texts))
    kept = trustrag_filter(["alpha beta gamma", "delta epsilon zeta", "eta theta iota"], embed)
    assert kept == [0, 1, 2] and not calls          # ROUGE gate fails -> no embedding, no filtering


def test_trustrag_three_calls_and_multi_suffix():
    seen = []
    llm = FakeLLM(lambda p: seen.append(p) or "Final answer(s): Paris")
    out = trustrag("Capital of France?", ["Paris is the capital."], llm, embed=None, multi=True)
    assert llm.calls == 3 and out["kept"] == [0]
    assert seen[2].endswith(TRUST_FINAL_MULTI_SUFFIX)


def test_conflictbank_scored_like_ramdocs():
    ex = Example(id="cb-1", dataset="conflictbank", question="q", passages=["a", "b"], gold_answers=["Paris"],
                 wrong_answers=["Lyon"], has_conflict=True)
    assert score(ex, "Final answer(s): Paris")["ramdocs_strict_oneway"] == 1.0
    assert score(ex, "Final answer(s): Paris | Lyon")["ramdocs_strict_oneway"] == 0.0
