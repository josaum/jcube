"""Relation-aware masking for dynamic graph experiments."""

from __future__ import annotations

import random
from collections.abc import Iterable


def relation_aware_mask(
    num_nodes: int,
    edges: Iterable[tuple[int, int]],
    mask_ratio: float,
    *,
    centrality_power: float = 1.0,
    mask_high_centrality: bool = True,
    seed: int | None = None,
) -> list[bool]:
    """Sample nodes using degree-derived relational centrality.

    Higher-degree nodes are preferentially masked by default, exposing the
    model to the loss of structurally important context. Set
    ``mask_high_centrality=False`` to prefer peripheral nodes instead.
    """
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")
    if not 0.0 <= mask_ratio <= 1.0:
        raise ValueError("mask_ratio must be in [0, 1]")
    if centrality_power < 0.0:
        raise ValueError("centrality_power must be non-negative")

    degrees = [0] * num_nodes
    for source, target in edges:
        if not 0 <= source < num_nodes or not 0 <= target < num_nodes:
            raise ValueError("edge endpoint is outside the node range")
        degrees[source] += 1
        degrees[target] += 1

    count = round(num_nodes * mask_ratio)
    if count == 0:
        return [False] * num_nodes
    if count == num_nodes:
        return [True] * num_nodes

    weights = [(degree + 1.0) ** centrality_power for degree in degrees]
    if not mask_high_centrality:
        weights = [1.0 / weight for weight in weights]

    rng = random.Random(seed)
    selected: set[int] = set()
    while len(selected) < count:
        selected.add(rng.choices(range(num_nodes), weights=weights, k=1)[0])
    return [index in selected for index in range(num_nodes)]
