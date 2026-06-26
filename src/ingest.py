"""Phase 1 — scan an image directory into a validated, deduped manifest.parquet.

Usage:  python -m src.ingest --config configs/data.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path

from PIL import Image

from .common import load_config, setup_logging, write_manifest
from .schema import validate_manifest

log = logging.getLogger("mmsearch")


def _hash_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def build_manifest(cfg: dict):
    import pandas as pd

    images_dir = Path(cfg["images_dir"])
    exts = {e.lower() for e in cfg["extensions"]}
    pattern = "**/*" if cfg.get("recursive", True) else "*"
    candidates = [p for p in images_dir.glob(pattern) if p.suffix.lower() in exts]
    log.info("Found %d candidate files under %s", len(candidates), images_dir)

    rows, seen = [], set()
    for p in sorted(candidates):
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                w, h = im.size
        except Exception as e:  # corrupt / unreadable
            log.warning("Skip unreadable %s (%s)", p, e)
            continue
        if min(w, h) < cfg.get("min_size", 0):
            continue
        if cfg.get("dedup", True):
            digest = _hash_file(p)
            if digest in seen:
                continue
            seen.add(digest)
        rows.append({"path": str(p), "width": w, "height": h})

    df = pd.DataFrame(rows)
    df.insert(0, "id", range(len(df)))
    validate_manifest(df)
    write_manifest(df, cfg["manifest_path"])
    log.info("Wrote manifest: %d images -> %s", len(df), cfg["manifest_path"])
    return df


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description="Build the image manifest.")
    ap.add_argument("--config", default="configs/data.yaml")
    args = ap.parse_args()
    build_manifest(load_config(args.config))


if __name__ == "__main__":
    main()
