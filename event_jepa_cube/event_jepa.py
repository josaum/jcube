"""Simplified Event-JEPA processor."""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable
from typing import Callable, List, Optional

from .sequence import EventSequence


class EventJEPA:
    """Lightweight processor for irregular event sequences."""

    def __init__(
        self,
        embedding_dim: int = 768,
        num_levels: int = 1,
        temporal_resolution: str = "adaptive",
        regularizer: Optional[Callable[[List[List[float]]], float]] = None,
        reg_weight: float = 0.05,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.num_levels = num_levels
        self.temporal_resolution = temporal_resolution
        self.regularizer = regularizer
        self.reg_weight = reg_weight

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _weighted_aggregate(
        embeddings: List[List[float]],
        timestamps: List[float],
        alpha: float = 1.0,
    ) -> List[float]:
        """Aggregate *embeddings* using exponential-decay weighting.

        w_i = exp(-alpha * (t_max - t_i))
        """
        if not embeddings:
            return []
        dim = len(embeddings[0])
        t_max = max(timestamps)
        weights: List[float] = [math.exp(-alpha * (t_max - t)) for t in timestamps]
        total_w = sum(weights)
        if total_w == 0.0:
            total_w = 1.0
        agg = [0.0] * dim
        for emb, w in zip(embeddings, weights):
            for i, v in enumerate(emb):
                agg[i] += v * w
        return [v / total_w for v in agg]

    def _partition_adaptive(
        self,
        embeddings: List[List[float]],
        timestamps: List[float],
    ) -> List[List[int]]:
        """Split indices into windows using gaps > median inter-event interval."""
        if len(timestamps) <= 1:
            return [list(range(len(timestamps)))]
        gaps = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
        median_gap = statistics.median(gaps)
        windows: List[List[int]] = [[0]]
        for i in range(1, len(timestamps)):
            if (timestamps[i] - timestamps[i - 1]) > median_gap:
                windows.append([i])
            else:
                windows[-1].append(i)
        return windows

    def _partition_fixed(
        self,
        embeddings: List[List[float]],
        timestamps: List[float],
    ) -> List[List[int]]:
        """Split indices into equal-width time bins."""
        if not timestamps:
            return []
        t_min, t_max = timestamps[0], timestamps[-1]
        if t_min == t_max:
            return [list(range(len(timestamps)))]
        num_bins = max(2, len(timestamps) // 2)
        bin_width = (t_max - t_min) / num_bins
        windows: List[List[int]] = [[] for _ in range(num_bins)]
        for i, t in enumerate(timestamps):
            b = int((t - t_min) / bin_width)
            if b >= num_bins:
                b = num_bins - 1
            windows[b].append(i)
        return [w for w in windows if w]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, sequence: EventSequence) -> List[float]:
        """Hierarchical temporal aggregation.

        1. Sort events by timestamp.
        2. Partition into temporal windows (adaptive or fixed).
        3. For each hierarchical level: aggregate within windows using
           exponential-decay weighting, then merge windows for the next level.
        4. Return the final aggregated representation.
        """
        if not sequence.embeddings:
            return []

        # Sort by timestamp
        order = sorted(range(len(sequence.timestamps)), key=lambda i: sequence.timestamps[i])
        embeddings = [sequence.embeddings[i] for i in order]
        timestamps = [sequence.timestamps[i] for i in order]

        for _level in range(self.num_levels):
            # Partition
            if self.temporal_resolution == "adaptive":
                windows = self._partition_adaptive(embeddings, timestamps)
            else:
                windows = self._partition_fixed(embeddings, timestamps)

            # Aggregate each window
            new_embeddings: List[List[float]] = []
            new_timestamps: List[float] = []
            for win in windows:
                win_embs = [embeddings[i] for i in win]
                win_ts = [timestamps[i] for i in win]
                agg = self._weighted_aggregate(win_embs, win_ts)
                new_embeddings.append(agg)
                new_timestamps.append(max(win_ts))

            embeddings = new_embeddings
            timestamps = new_timestamps

        # Final aggregation across remaining windows
        return self._weighted_aggregate(embeddings, timestamps)

    def detect_patterns(self, representation: Iterable[float]) -> List[int]:
        """Detect salient dimensions via z-score thresholding.

        Dimensions with |z| > 1.5 are considered salient.  If fewer than 2
        dimensions pass the threshold, fall back to the top-5 by magnitude.
        """
        vals = list(representation)
        if len(vals) < 2:
            return list(range(len(vals)))

        mean = statistics.mean(vals)
        stdev = statistics.pstdev(vals)

        if stdev == 0.0:
            return list(range(min(5, len(vals))))

        z_scores = [(abs((v - mean) / stdev), i) for i, v in enumerate(vals)]
        salient = [i for z, i in z_scores if z > 1.5]

        if len(salient) < 2:
            # Fall back to top-5 by absolute value
            sorted_idx = sorted(range(len(vals)), key=lambda i: abs(vals[i]), reverse=True)
            return sorted_idx[: min(5, len(sorted_idx))]

        return sorted(salient)

    def predict_next(self, sequence: EventSequence, num_steps: int = 1) -> List[List[float]]:
        """Exponentially-weighted moving-trend prediction.

        1. Compute deltas between consecutive embeddings.
        2. Weight deltas by recency using timestamps.
        3. Extrapolate from the last embedding.
        """
        if not sequence.embeddings:
            return []
        if len(sequence.embeddings) < 2:
            return [list(sequence.embeddings[-1]) for _ in range(num_steps)]

        # Sort by timestamp
        order = sorted(range(len(sequence.timestamps)), key=lambda i: sequence.timestamps[i])
        embeddings = [sequence.embeddings[i] for i in order]
        timestamps = [sequence.timestamps[i] for i in order]

        dim = len(embeddings[0])

        # Compute deltas and their associated timestamps (use midpoint)
        deltas: List[List[float]] = []
        delta_times: List[float] = []
        for j in range(1, len(embeddings)):
            delta = [embeddings[j][d] - embeddings[j - 1][d] for d in range(dim)]
            deltas.append(delta)
            delta_times.append(timestamps[j])

        # Exponential weighting by recency
        t_max = max(delta_times)
        alpha = 1.0
        weights = [math.exp(-alpha * (t_max - t)) for t in delta_times]
        total_w = sum(weights)
        if total_w == 0.0:
            total_w = 1.0

        trend = [0.0] * dim
        for delta, w in zip(deltas, weights):
            for d in range(dim):
                trend[d] += delta[d] * w
        trend = [v / total_w for v in trend]

        # Extrapolate
        last = embeddings[-1]
        predictions: List[List[float]] = []
        for step in range(1, num_steps + 1):
            pred = [last[d] + trend[d] * step for d in range(dim)]
            predictions.append(pred)
        return predictions

    def compute_regularized_loss(
        self,
        embeddings: List[List[float]],
        prediction_loss: float,
    ) -> float:
        """Combine prediction loss with an optional regularizer.

        L = L_pred + reg_weight * L_reg
        """
        if self.regularizer is None:
            return prediction_loss
        reg_loss = self.regularizer(embeddings)
        return prediction_loss + self.reg_weight * reg_loss
