"""Explicit inter-stage data contracts.

Lightweight checks so a malformed artifact fails fast at the producing stage with
a clear message, instead of surfacing as a KeyError (or silently wrong numbers)
three stages downstream. Validators check column presence AND values
(uniqueness, ranges, finiteness). Called at every stage boundary."""

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
    """Manifest must carry the id/path/width/height contract with unique ids and
    positive dimensions."""
    validate_columns(df, MANIFEST_COLUMNS, "manifest")
    if len(df):
        if df["id"].duplicated().any():
            raise ValueError("manifest has duplicate ids")
        if (df[["width", "height"]].to_numpy() <= 0).any():
            raise ValueError("manifest has non-positive width/height")
    return df


def validate_embeddings(embeddings: np.ndarray, ids: np.ndarray) -> None:
    """Embeddings/ids must be an aligned (n, d) finite matrix with unique ids."""
    embeddings = np.asarray(embeddings)
    ids = np.asarray(ids)
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2-D (n, d), got shape {embeddings.shape}")
    if embeddings.shape[0] != ids.shape[0]:
        raise ValueError(
            f"embeddings ({embeddings.shape[0]}) and ids ({ids.shape[0]}) are misaligned"
        )
    if not np.isfinite(embeddings).all():
        raise ValueError("embeddings contain NaN/inf")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("ids are not unique")
