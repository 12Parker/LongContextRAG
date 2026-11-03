"""The extracted detector must stay identical to the original, and the offline
thresholding in CAAGDetector must reproduce the original's flags exactly."""
import itertools
import random
from pathlib import Path

import pytest

from experiments.conflict.caag_core import ContradictionDetector
from experiments.conflict.data import Example
from experiments.conflict.detectors import CAAGDetector

ROOT = Path(__file__).resolve().parents[1]


def test_source_is_verbatim():
    orig = (ROOT / "hybrid/narrativeqa_hybrid_rag_contradiction_aware.py").read_text().splitlines()[55:423]
    core = (ROOT / "experiments/conflict/caag_core.py").read_text().splitlines()
    start = core.index(orig[0])
    assert core[start:start + len(orig)] == orig


SENTS = [
    "The treaty was signed in 1848 after the war ended, but the border remained disputed.",
    "He never returned to London; however, his letters arrived every week in 1850.",
    "Initially the ship sailed north, then later it turned toward the southern islands.",
    "The population was 3,559 people according to the census bureau in 2010.",
    "The population was 4,120 people according to the census bureau in 2020.",
    "She did not believe the story, although the captain insisted it was true.",
    "Before the storm, the village had 200 residents and two churches.",
    "After the storm, nothing remained of the village except the old church.",
]


@pytest.mark.parametrize("theta_s,theta_c", [(0.65, 0.35), (0.3, 0.2), (0.0, 0.1), (0.5, 0.5)])
def test_offline_threshold_matches_original(emb, theta_s, theta_c):
    rng = random.Random(0)
    for trial in range(15):
        ps = rng.sample(SENTS, rng.randint(3, 6))
        orig = ContradictionDetector(emb, similarity_threshold=theta_s, contradiction_threshold=theta_c)
        want = {tuple(sorted((c.doc_idx_1, c.doc_idx_2))) for c in orig.detect_contradictions([{"text": p} for p in ps])}
        got = {(d.i, d.j) for d in CAAGDetector(emb, theta_s, theta_c).detect(Example("x", "t", "q", ps)).pairs}
        assert got == want, (trial, ps)
