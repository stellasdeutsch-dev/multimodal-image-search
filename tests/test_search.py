"""Searcher tests — inject a fake encoder so the retrieval logic is tested
without the heavy CLIP stack."""

import numpy as np

from src.index import VectorIndex
from src.search import Searcher


class FakeEncoder:
    def __init__(self, table: dict[str, np.ndarray]):
        self.table = table

    def encode_text(self, texts):
        return np.stack([self.table[t] for t in texts]).astype("float32")

    def encode_images(self, images):
        return np.stack(images).astype("float32")


def _build_searcher():
    vecs = np.eye(3, dtype="float32")
    ids = np.array([100, 200, 300])
    index = VectorIndex(dim=3, use_faiss=False).build(vecs, ids)
    enc = FakeEncoder({"cat": np.array([1, 0, 0], dtype="float32")})
    manifest = {100: {"path": "a", "width": 1, "height": 1},
                200: {"path": "b", "width": 1, "height": 1},
                300: {"path": "c", "width": 1, "height": 1}}
    return Searcher(enc, index, manifest, top_k=3)


def test_search_text_ranking_and_payload():
    s = _build_searcher()
    res = s.search_text("cat", k=3)
    assert res[0]["id"] == 100
    assert res[0]["path"] == "a"
    assert res[0]["score"] > res[1]["score"]


def test_search_image_uses_vector_directly():
    s = _build_searcher()
    # search_image takes ONE image; the encoder stacks it into a (1, d) batch.
    res = s.search_image(np.array([0, 1, 0], dtype="float32"), k=1)
    assert res[0]["id"] == 200
