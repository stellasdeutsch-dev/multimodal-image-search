"""Explicit inter-stage data contracts.

Lightweight, dependency-free column/shape checks so a malformed artifact fails
fast at the producing stage with a clear message, instead of surfacing as a
KeyError three stages downstream. Validators are called at every stage boundary
(ingest write, embed/index/search read). The column-set constants are the single
source of truth for the manifest schema."""

from __future__ import annotations

import numpy as np

MANIFEST_COLUMNS = ["id", "path", "width", "height"]


def missing_columns(present, required) -> list:
    """Pure helper: which required columns are absent from `present`."""
    have = set(present)
    return [c for c in required if c not in have]


def validate_columns(df, required, name: str):
    miss = missing_columns(list(df.columns), required)
    if miss:
        raise ValueError(f"{name} is missing columns {miss}; has {list(df.columns)}")
    return df


def validate_manifest(df):
    """Check a manifest DataFrame carries the id/path/width/height contract."""
    return validate_columns(df, MANIFEST_COLUMNS, "manifest")


def validate_embeddings(embeddings: np.ndarray, ids: np.ndarray) -> None:
    """Check embeddings/ids form an aligned (n, d) float matrix and (n,) ids."""
    embeddings = np.asarray(embeddings)
    ids = np.asarray(ids)
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2-D (n, d), got shape {embeddings.shape}")
    if embeddings.shape[0] != ids.shape[0]:
        raise ValueError(
            f"embeddings ({embeddings.shape[0]}) and ids ({ids.shape[0]}) are misaligned"
        )
