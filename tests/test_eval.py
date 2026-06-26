"""Retrieval metrics — pure functions, no heavy deps."""

from eval.evaluate import average_precision, recall_at_k


def test_recall_at_k():
    assert recall_at_k([5, 1, 2], [1], 1) == 0.0      # relevant not in top-1
    assert recall_at_k([5, 1, 2], [1], 2) == 1.0      # relevant within top-2
    assert recall_at_k([5, 1, 2], [9], 3) == 0.0      # relevant absent


def test_average_precision():
    # relevant {1,2}; ranked [1,0,2] -> hits at ranks 1 and 3 -> (1/1 + 2/3)/2
    ap = average_precision([1, 0, 2], [1, 2])
    assert abs(ap - ((1.0 + 2 / 3) / 2)) < 1e-9
    assert average_precision([0, 0], [1]) == 0.0
