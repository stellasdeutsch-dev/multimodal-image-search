"""Data-contract validators — pure (no pandas needed: validators are duck-typed
on .columns)."""

import numpy as np
import pytest

from src.schema import MANIFEST_COLUMNS, missing_columns, validate_embeddings, validate_manifest


class _DF:
    def __init__(self, cols):
        self.columns = cols


def test_missing_columns():
    assert missing_columns(["id", "path"], MANIFEST_COLUMNS) == ["width", "height"]
    assert missing_columns(MANIFEST_COLUMNS, MANIFEST_COLUMNS) == []


def test_validate_manifest_ok_and_bad():
    validate_manifest(_DF(MANIFEST_COLUMNS))                 # does not raise
    with pytest.raises(ValueError):
        validate_manifest(_DF(["id", "path"]))


def test_validate_embeddings():
    validate_embeddings(np.zeros((3, 8), "float32"), np.arange(3))   # ok
    with pytest.raises(ValueError):
        validate_embeddings(np.zeros((3, 8)), np.arange(2))          # misaligned
    with pytest.raises(ValueError):
        validate_embeddings(np.zeros(8), np.arange(8))               # not 2-D
