"""CLIP encoder wrapper (OpenCLIP). Encodes images and text into a shared,
L2-normalized embedding space. torch / open_clip are imported lazily so the
rest of the package (index, tests) runs without the heavy ML stack."""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from .common import select_device

log = logging.getLogger("mmsearch")


class ClipEncoder:
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str = "auto",
    ) -> None:
        import open_clip  # noqa: F401
        import torch

        self.torch = torch
        self.device = select_device(device)
        log.info("Loading CLIP %s (%s) on %s", model_name, pretrained, self.device)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        self.dim = int(self.model.visual.output_dim)

    def _to_numpy(self, feats) -> np.ndarray:
        feats = feats.float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.detach().cpu().numpy().astype("float32")

    def encode_image_tensors(self, batch, fp16: bool = False) -> np.ndarray:
        torch = self.torch
        batch = batch.to(self.device)
        with torch.no_grad():
            if fp16 and self.device == "cuda":
                with torch.autocast("cuda"):
                    feats = self.model.encode_image(batch)
            else:
                feats = self.model.encode_image(batch)
        return self._to_numpy(feats)

    def encode_images(self, images: Sequence) -> np.ndarray:
        torch = self.torch
        tensors = torch.stack([self.preprocess(im.convert("RGB")) for im in images])
        return self.encode_image_tensors(tensors)

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        torch = self.torch
        tokens = self.tokenizer(list(texts)).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_text(tokens)
        return self._to_numpy(feats)
