import math

import pytest

from recall.evaluation.metrics import (
    ndcg_at_k,
    percentile,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], {"a", "d"}) == 0.5
    assert recall_at_k(["x", "a"], {"a"}, k=1) == 0.0
    with pytest.raises(ValueError):
        recall_at_k(["a"], set())


def test_recall_at_k_with_empty_string():
    # Empty strings should be treated as never relevant
    assert recall_at_k(["", "a"], {"a"}) == 1.0
    assert recall_at_k(["", "b"], {"a"}) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(["x"], {"a"}) == 0.0


def test_reciprocal_rank_with_empty_string():
    # Empty strings should be skipped, so first relevant is at position 1 (0-based)
    assert reciprocal_rank(["", "a"], {"a"}) == pytest.approx(0.5)


def test_ndcg():
    assert ndcg_at_k(["a", "x", "b"], {"a", "b"}) == pytest.approx(1.5 / (1 + 1 / math.log2(3)))
    assert ndcg_at_k(["a", "a"], {"a"}) == pytest.approx(1.0)
    assert ndcg_at_k(["x"], {"a"}) == 0.0


def test_percentile():
    assert percentile([float(i) for i in range(1, 101)], 95) == pytest.approx(95.05)
    assert percentile([7.0], 95) == 7.0
