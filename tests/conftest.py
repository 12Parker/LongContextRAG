import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
DATA = Path(__file__).resolve().parents[1] / "data" / "external"
needs_data = pytest.mark.skipif(not (DATA / "RAMDocs").exists(), reason="run setup_env.sh to fetch data")


class HashEmbedder:
    """Offline stand-in for MiniLM: hashed bag-of-words vectors (cosine ~ lexical overlap)."""
    def embed_documents(self, texts):
        out = []
        for t in texts:
            v = np.zeros(256)
            for w in t.lower().split():
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 256] += 1
            out.append(v.tolist())
        return out


@pytest.fixture
def emb():
    return HashEmbedder()
