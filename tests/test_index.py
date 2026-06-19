"""VectorIndex tests — exercise the NumPy backend (no FAISS/torch needed)."""

import numpy as np

from src.index import VectorIndex


def test_numpy_index_topk_self_match():
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((50, 16)).astype("float32")
    ids = np.arange(50) * 10
    idx = VectorIndex(dim=16, index_type="flat", metric="ip", use_faiss=False).build(vecs, ids)

    scores, got = idx.search(vecs[3:4], k=5)
    assert got.shape == (1, 5)
    assert got[0, 0] == 30                     # ids[3] == 30
    assert scores[0, 0] >= scores[0, 1]        # sorted descending
    assert scores[0, 0] > 0.99                 # cosine self-similarity ~ 1


def test_k_larger_than_corpus():
    vecs = np.eye(3, dtype="float32")
    idx = VectorIndex(dim=3, use_faiss=False).build(vecs, np.array([1, 2, 3]))
    scores, ids = idx.search(vecs[:1], k=10)
    assert scores.shape[1] == 3                 # clamped to corpus size


def test_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    vecs = rng.standard_normal((20, 8)).astype("float32")
    ids = np.arange(20)
    idx = VectorIndex(dim=8, use_faiss=False).build(vecs, ids)
    path = tmp_path / "index.faiss"
    idx.save(path)

    loaded = VectorIndex.load(path)
    s1, i1 = idx.search(vecs[:2], 3)
    s2, i2 = loaded.search(vecs[:2], 3)
    assert np.array_equal(i1, i2)
    assert np.allclose(s1, s2)
