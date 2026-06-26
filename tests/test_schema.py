"""Data-contract validators — column presence AND value-level checks."""

import numpy as np
import pandas as pd
import pytest

from src.schema import MANIFEST_COLUMNS, missing_columns, validate_embeddings, validate_manifest


def _manifest(n=3):
    return pd.DataFrame({"id": range(n), "path": [f"{i}.jpg" for i in range(n)],
                         "width": [10] * n, "height": [10] * n})


def test_missing_columns_helper():
    assert missing_columns(["id", "path"], MANIFEST_COLUMNS) == ["width", "height"]
    assert missing_columns(MANIFEST_COLUMNS, MANIFEST_COLUMNS) == []


def test_manifest_ok():
    validate_manifest(_manifest())                       # does not raise


def test_manifest_missing_column():
    with pytest.raises(ValueError):
        validate_manifest(_manifest().drop(columns=["width"]))


def test_manifest_duplicate_id_rejected():
    m = _manifest()
    m.loc[2, "id"] = 0                                    # duplicate of row 0
    with pytest.raises(ValueError):
        validate_manifest(m)


def test_manifest_nonpositive_dimension_rejected():
    m = _manifest()
    m.loc[1, "width"] = 0
    with pytest.raises(ValueError):
        validate_manifest(m)


def test_embeddings_ok():
    validate_embeddings(np.zeros((3, 8), "float32"), np.arange(3))


def test_embeddings_value_checks():
    with pytest.raises(ValueError):
        validate_embeddings(np.zeros((3, 8)), np.arange(2))      # misaligned
    with pytest.raises(ValueError):
        validate_embeddings(np.zeros(8), np.arange(8))           # not 2-D
    bad = np.zeros((2, 4))
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        validate_embeddings(bad, np.arange(2))                   # NaN
    with pytest.raises(ValueError):
        validate_embeddings(np.zeros((2, 4)), np.array([5, 5]))  # duplicate ids
