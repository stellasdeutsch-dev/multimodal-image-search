"""Phase 3 — vector index over L2-normalized embeddings.

Uses FAISS when installed; otherwise a NumPy brute-force fallback so the whole
retrieval path runs anywhere (local demo, CI, tests). Inner product on
normalized vectors == cosine similarity.

Usage:  python -m src.index --config configs/index.yaml --build
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from .common import l2_normalize, load_config, setup_logging

log = logging.getLogger("mmsearch")

try:
    import faiss

    _HAS_FAISS = True
except Exception:  # pragma: no cover - environment dependent
    faiss = None
    _HAS_FAISS = False


def _sibling(path: str | Path, ext: str) -> Path:
    p = Path(path)
    return p.with_name(p.stem + ext)


class VectorIndex:
    def __init__(
        self,
        dim: int,
        index_type: str = "flat",
        metric: str = "ip",
        ivf_nlist: int = 1024,
        ivf_nprobe: int = 16,
        hnsw_m: int = 32,
        hnsw_ef_search: int = 64,
        use_faiss: bool | None = None,
    ) -> None:
        self.dim = int(dim)
        self.index_type = index_type
        self.metric = metric
        self.ivf_nlist = ivf_nlist
        self.ivf_nprobe = ivf_nprobe
        self.hnsw_m = hnsw_m
        self.hnsw_ef_search = hnsw_ef_search
        self.use_faiss = _HAS_FAISS if use_faiss is None else (use_faiss and _HAS_FAISS)
        self.ids: np.ndarray | None = None
        self._index = None      # FAISS index
        self._matrix = None     # NumPy fallback matrix

    # ---- build ----
    def build(self, embeddings: np.ndarray, ids: np.ndarray) -> "VectorIndex":
        mat = l2_normalize(embeddings)
        self.ids = np.asarray(ids).astype("int64")
        if mat.shape[0] != self.ids.shape[0]:
            raise ValueError("embeddings and ids length mismatch")
        if self.use_faiss:
            self._build_faiss(mat)
        else:
            self._matrix = mat
            log.info("Built NumPy index: %d x %d", *mat.shape)
        return self

    def _build_faiss(self, mat: np.ndarray) -> None:
        m = faiss.METRIC_INNER_PRODUCT if self.metric == "ip" else faiss.METRIC_L2
        if self.index_type == "flat":
            index = faiss.IndexFlatIP(self.dim) if self.metric == "ip" else faiss.IndexFlatL2(self.dim)
        elif self.index_type == "ivf":
            quant = faiss.IndexFlatIP(self.dim) if self.metric == "ip" else faiss.IndexFlatL2(self.dim)
            nlist = max(1, min(self.ivf_nlist, mat.shape[0] // 39 or 1))
            index = faiss.IndexIVFFlat(quant, self.dim, nlist, m)
            log.info("Training IVF (nlist=%d) ...", nlist)
            index.train(mat)
            index.nprobe = self.ivf_nprobe
        elif self.index_type == "hnsw":
            index = faiss.IndexHNSWFlat(self.dim, self.hnsw_m, m)
            index.hnsw.efSearch = self.hnsw_ef_search
        else:
            raise ValueError(f"unknown index_type {self.index_type!r}")
        index.add(mat)
        self._index = index
        log.info("Built FAISS %s index: ntotal=%d", self.index_type, index.ntotal)

    # ---- search ----
    def search(self, queries: np.ndarray, k: int = 10):
        q = l2_normalize(queries)
        if q.ndim == 1:
            q = q[None, :]
        if self.use_faiss and self._index is not None:
            scores, internal = self._index.search(q, k)
            if self.metric != "ip":
                # FAISS L2 returns squared distances; convert to cosine similarity
                # (vectors are L2-normalized: d^2 = 2 - 2*cos) so the returned
                # `score` means "higher == more similar" for every metric/backend.
                scores = 1.0 - scores / 2.0
            valid = internal >= 0
            clipped = np.clip(internal, 0, len(self.ids) - 1)
            ext = np.where(valid, self.ids[clipped], -1)
            return scores.astype("float32"), ext.astype("int64")
        return self._search_numpy(q, k)

    def _search_numpy(self, q: np.ndarray, k: int):
        # Inputs are L2-normalized, so inner product == cosine similarity for
        # both metrics; rank descending (higher == more similar).
        sims = q @ self._matrix.T            # (n, N)
        n, N = sims.shape
        k = min(k, N)
        part = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        rows = np.arange(n)[:, None]
        order = np.argsort(-sims[rows, part], axis=1)
        idx = part[rows, order]
        scores = sims[rows, idx]
        ext = self.ids[idx]
        return scores.astype("float32"), ext.astype("int64")

    # ---- persistence ----
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        backend = "faiss" if (self.use_faiss and self._index is not None) else "numpy"
        np.save(_sibling(path, ".ids.npy"), self.ids)
        _sibling(path, ".meta.json").write_text(
            json.dumps({"dim": self.dim, "index_type": self.index_type,
                        "metric": self.metric, "backend": backend})
        )
        if backend == "faiss":
            faiss.write_index(self._index, str(path))
        else:
            np.save(_sibling(path, ".vectors.npy"), self._matrix)
        log.info("Saved %s index -> %s", backend, path)

    @classmethod
    def load(cls, path: str | Path, ivf_nprobe: int | None = None,
             hnsw_ef_search: int | None = None) -> "VectorIndex":
        path = Path(path)
        meta = json.loads(_sibling(path, ".meta.json").read_text())
        obj = cls(dim=meta["dim"], index_type=meta["index_type"],
                  metric=meta["metric"], use_faiss=(meta["backend"] == "faiss"))
        obj.ids = np.load(_sibling(path, ".ids.npy"))
        if meta["backend"] == "faiss":
            if not _HAS_FAISS:
                raise RuntimeError("Index built with FAISS but faiss is not installed here.")
            obj._index = faiss.read_index(str(path))
            if ivf_nprobe is not None and hasattr(obj._index, "nprobe"):
                obj._index.nprobe = ivf_nprobe
            if hnsw_ef_search is not None and hasattr(obj._index, "hnsw"):
                obj._index.hnsw.efSearch = hnsw_ef_search
        else:
            obj._matrix = np.load(_sibling(path, ".vectors.npy"))
        return obj


def build_from_config(cfg: dict) -> VectorIndex:
    emb = np.load(cfg["embeddings_path"])
    ids = np.load(cfg["ids_path"])
    idx = VectorIndex(
        dim=emb.shape[1],
        index_type=cfg.get("index_type", "flat"),
        metric=cfg.get("metric", "ip"),
        ivf_nlist=cfg.get("ivf_nlist", 1024),
        ivf_nprobe=cfg.get("ivf_nprobe", 16),
        hnsw_m=cfg.get("hnsw_m", 32),
        hnsw_ef_search=cfg.get("hnsw_ef_search", 64),
    )
    idx.build(emb, ids)
    idx.save(cfg["index_path"])
    return idx


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description="Build / inspect the vector index.")
    ap.add_argument("--config", default="configs/index.yaml")
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.build:
        build_from_config(cfg)
    else:
        log.info("Nothing to do. Pass --build to build the index.")


if __name__ == "__main__":
    main()
