"""Shared utilities: config loading, device selection, normalization, manifest I/O."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

LOGGER_NAME = "mmsearch"
log = logging.getLogger(LOGGER_NAME)


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(LOGGER_NAME)


def load_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as e:  # pragma: no cover
        raise ImportError("PyYAML is required to load configs (pip install pyyaml)") from e
    with open(path) as f:
        return yaml.safe_load(f)


def select_device(pref: str = "auto") -> str:
    """Resolve a device string. 'auto' -> cuda > mps > cpu."""
    pref = (pref or "auto").lower()
    if pref != "auto":
        return pref
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)


# ---- manifest I/O (pandas/pyarrow) ----

def read_manifest(path: str | Path):
    import pandas as pd

    return pd.read_parquet(path)


def write_manifest(df, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def manifest_to_dict(df) -> dict[int, dict]:
    """Map image id -> record (path, width, height) for fast lookup at query time."""
    out: dict[int, dict] = {}
    for row in df.itertuples(index=False):
        out[int(row.id)] = {
            "path": str(row.path),
            "width": int(row.width),
            "height": int(row.height),
        }
    return out
