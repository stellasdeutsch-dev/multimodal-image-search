"""FastAPI serving smoke test. The searcher is injected (NumPy index + fake
encoder), so it runs with no torch/faiss. Skips if fastapi isn't installed."""

import numpy as np
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from src import api  # noqa: E402
from src.index import VectorIndex  # noqa: E402
from src.llm import QueryAssistant  # noqa: E402
from src.search import Searcher  # noqa: E402


class FakeEncoder:
    def encode_text(self, texts):
        return np.array([[1.0, 0.0, 0.0]] * len(texts), dtype="float32")

    def encode_images(self, images):
        return np.stack(images).astype("float32")


def _inject():
    idx = VectorIndex(dim=3, use_faiss=False).build(np.eye(3, dtype="float32"), np.array([10, 20, 30]))
    manifest = {10: {"path": "a", "width": 1, "height": 1},
                20: {"path": "b", "width": 1, "height": 1},
                30: {"path": "c", "width": 1, "height": 1}}
    api.STATE["searcher"] = Searcher(FakeEncoder(), idx, manifest, top_k=3)
    api.STATE["llm"] = QueryAssistant({"enabled": False})


def test_health_and_text_search():
    _inject()
    with TestClient(api.app) as client:
        assert client.get("/health").json()["n_images"] == 3
        r = client.post("/search/text", json={"query": "cat", "k": 2})
        assert r.status_code == 200
        results = r.json()["results"]
        assert results[0]["id"] == 10
        assert results[0]["path"] == "a"
