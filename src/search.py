"""Core retrieval: turn a text or image query into ranked image results.

The Searcher takes an encoder (anything with encode_text / encode_images), a
VectorIndex, and a manifest dict. This dependency injection keeps the core
testable without the heavy CLIP stack (see tests/test_search.py)."""

from __future__ import annotations

import logging

from .common import load_config, manifest_to_dict, read_manifest
from .index import VectorIndex

log = logging.getLogger("mmsearch")


class Searcher:
    def __init__(self, encoder, index: VectorIndex, manifest: dict[int, dict], top_k: int = 24) -> None:
        self.encoder = encoder
        self.index = index
        self.manifest = manifest
        self.top_k = top_k

    @classmethod
    def from_config(cls, cfg: dict) -> "Searcher":
        from .model import ClipEncoder

        encoder = ClipEncoder(cfg["model"], cfg["pretrained"], cfg.get("device", "auto"))
        # Optional serve-time ANN tuning (override the values baked into the index).
        index = VectorIndex.load(
            cfg["index_path"],
            ivf_nprobe=cfg.get("ivf_nprobe"),
            hnsw_ef_search=cfg.get("hnsw_ef_search"),
        )
        manifest = manifest_to_dict(read_manifest(cfg["manifest_path"]))
        return cls(encoder, index, manifest, top_k=cfg.get("top_k", 24))

    def _format(self, scores, ids) -> list[dict]:
        results = []
        for s, i in zip(scores, ids):
            i = int(i)
            if i < 0:
                continue
            rec = self.manifest.get(i, {})
            results.append({"id": i, "score": float(s), **rec})
        return results

    def search_text(self, query: str, k: int | None = None) -> list[dict]:
        q = self.encoder.encode_text([query])
        scores, ids = self.index.search(q, k or self.top_k)
        return self._format(scores[0], ids[0])

    def search_image(self, image, k: int | None = None) -> list[dict]:
        q = self.encoder.encode_images([image])
        scores, ids = self.index.search(q, k or self.top_k)
        return self._format(scores[0], ids[0])


def load_searcher(config_path: str = "configs/serve.yaml") -> Searcher:
    return Searcher.from_config(load_config(config_path))
