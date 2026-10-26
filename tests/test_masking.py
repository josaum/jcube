"""Tests for relation-aware graph masking."""

from collections import Counter

import pytest

from event_jepa_cube.masking import relation_aware_mask


def test_mask_is_deterministic_and_has_requested_size():
    edges = [(0, 1), (1, 2), (2, 3)]
    first = relation_aware_mask(4, edges, 0.5, seed=42)
    second = relation_aware_mask(4, edges, 0.5, seed=42)

    assert first == second
    assert sum(first) == 2


def test_high_degree_nodes_are_masked_more_often():
    hub = 0
    edges = [(hub, index) for index in range(1, 12)]
    counts = Counter()
    for seed in range(200):
        mask = relation_aware_mask(12, edges, 1 / 3, seed=seed)
        for index, masked in enumerate(mask):
            counts[index] += masked

    assert counts[hub] > counts[1]


def test_mask_validates_edges_and_ratio():
    with pytest.raises(ValueError, match="mask_ratio"):
        relation_aware_mask(2, [], 1.5)
    with pytest.raises(ValueError, match="outside"):
        relation_aware_mask(2, [(0, 3)], 0.5)
