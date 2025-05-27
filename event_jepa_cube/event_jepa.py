"""Simplified Event-JEPA processor."""
from __future__ import annotations

from typing import Iterable, List

from .sequence import EventSequence


class EventJEPA:
    """Lightweight processor for irregular event sequences."""

    def __init__(self, embedding_dim: int = 768, num_levels: int = 1, temporal_resolution: str = "adaptive") -> None:
        self.embedding_dim = embedding_dim
        self.num_levels = num_levels
        self.temporal_resolution = temporal_resolution

    def process(self, sequence: EventSequence) -> List[float]:
        """Aggregate embeddings by simple averaging."""
        if not sequence.embeddings:
            return []
        dim = len(sequence.embeddings[0])
        agg = [0.0] * dim
        for emb in sequence.embeddings:
            for i, v in enumerate(emb):
                agg[i] += v
        count = float(len(sequence.embeddings))
        return [v / count for v in agg]

    def detect_patterns(self, representation: Iterable[float]) -> List[int]:
        """Dummy pattern detector returning index of top values."""
        sorted_idx = sorted(range(len(representation)), key=lambda i: representation[i], reverse=True)
        return sorted_idx[: min(5, len(sorted_idx))]

    def predict_next(self, sequence: EventSequence, num_steps: int = 1) -> List[List[float]]:
        """Generate simple predictions by repeating the last embedding."""
        if not sequence.embeddings:
            return []
        last = sequence.embeddings[-1]
        return [list(last) for _ in range(num_steps)]
