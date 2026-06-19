"""Phase 2 — batch-encode every image in the manifest with CLIP.

Heavy GPU step: run on the cluster via slurm/embed.slurm. Locally it works on
MPS/CPU for a tiny sample (smoke test only). Writes aligned embeddings + ids.

Usage:  python -m src.embed --config configs/embed.yaml --fp16
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np

from .common import load_config, read_manifest, setup_logging
from .model import ClipEncoder

log = logging.getLogger("mmsearch")


def _make_loader(paths, preprocess, batch_size: int, num_workers: int):
    import torch
    from PIL import Image

    class _Dataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return len(paths)

        def __getitem__(self, i: int):
            with Image.open(paths[i]) as im:
                return preprocess(im.convert("RGB"))

    return torch.utils.data.DataLoader(
        _Dataset(),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )


def run(cfg: dict) -> None:
    df = read_manifest(cfg["manifest_path"])
    paths = df["path"].tolist()
    ids = df["id"].to_numpy().astype("int64")
    if not paths:
        raise SystemExit("Manifest is empty — run ingest first.")

    encoder = ClipEncoder(cfg["model"], cfg["pretrained"], cfg.get("device", "auto"))
    loader = _make_loader(paths, encoder.preprocess, cfg["batch_size"], cfg["num_workers"])

    run_w = None
    if cfg.get("wandb", False):
        try:
            import wandb

            run_w = wandb.init(project=cfg.get("wandb_project", "multimodal-image-search"), config=cfg)
        except Exception as e:  # pragma: no cover
            log.warning("wandb disabled (%s)", e)

    chunks, n, t0 = [], 0, time.time()
    for batch in loader:
        feats = encoder.encode_image_tensors(batch, fp16=cfg.get("fp16", False))
        chunks.append(feats)
        n += len(batch)
        dt = time.time() - t0
        ips = n / dt if dt else 0.0
        log.info("embedded %d/%d  (%.1f img/s)", n, len(paths), ips)
        if run_w is not None:
            run_w.log({"images": n, "img_per_s": ips})

    embeddings = np.concatenate(chunks, axis=0).astype("float32")
    np.save(cfg["embeddings_path"], embeddings)
    np.save(cfg["ids_path"], ids)
    dt = time.time() - t0
    log.info(
        "Done: %d embeddings (dim=%d) in %.1fs (%.1f img/s) -> %s",
        embeddings.shape[0], embeddings.shape[1], dt, n / dt if dt else 0.0, cfg["embeddings_path"],
    )
    if run_w is not None:
        run_w.finish()


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description="Batch CLIP embedding.")
    ap.add_argument("--config", default="configs/embed.yaml")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.fp16:
        cfg["fp16"] = True
    if args.device:
        cfg["device"] = args.device
    run(cfg)


if __name__ == "__main__":
    main()
