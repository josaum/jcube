"""Simplified Event-JEPA processor with V-JEPA 2.1 improvements.

Uses numpy-accelerated operations when available, with pure-Python fallback.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable
from typing import Any, Callable, Dict, List, Optional

from . import numpy_ops as npo
from .sequence import EventSequence


class EventJEPA:
    """Lightweight processor for irregular event sequences.

    Supports V-JEPA 2.1 improvements: multi-level processing, dense context
    loss, modality-specific configs, and position-aware prediction.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        num_levels: int = 1,
        temporal_resolution: str = "adaptive",
        regularizer: Optional[Callable[[List[List[float]]], float]] = None,
        reg_weight: float = 0.05,
        modality_aware: bool = False,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.num_levels = num_levels
        self.temporal_resolution = temporal_resolution
        self.regularizer = regularizer
        self.reg_weight = reg_weight
        self.modality_aware = modality_aware
        self._modality_configs: Dict[str, Dict[str, Any]] = {}
        self._modality_offsets: Dict[str, List[float]] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _weighted_aggregate(
        embeddings: List[List[float]],
        timestamps: List[float],
        alpha: float = 1.0,
    ) -> List[float]:
        """Aggregate *embeddings* using exponential-decay weighting."""
        return npo.weighted_aggregate(embeddings, timestamps, alpha)

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

    def _get_modality_resolution(self, modality: str) -> str:
        """Get temporal resolution for a modality, falling back to default."""
        if self.modality_aware:
            cfg = self._modality_configs.get(modality, {})
            val = cfg.get("temporal_resolution")
            if isinstance(val, str):
                return val
        return self.temporal_resolution

    def _get_modality_alpha(self, modality: str) -> float:
        """Get aggregation alpha for a modality, falling back to 1.0."""
        if self.modality_aware:
            cfg = self._modality_configs.get(modality, {})
            val = cfg.get("alpha")
            if isinstance(val, (int, float)):
                return float(val)
        return 1.0

    def _apply_modality_offset(
        self,
        embeddings: List[List[float]],
        modality: str,
    ) -> List[List[float]]:
        """Add modality-specific embedding offset if registered."""
        if not self.modality_aware:
            return embeddings
        offset = self._modality_offsets.get(modality)
        if offset is None:
            return embeddings
        return npo.apply_offset(embeddings, offset)

    @staticmethod
    def _temporal_position_encoding(timestamp: float, dim: int) -> List[float]:
        """Sinusoidal position encoding for a temporal position."""
        return npo.temporal_position_encoding(timestamp, dim)

    @staticmethod
    def _compute_temporal_distances(
        context_timestamps: List[float],
        mask_timestamps: List[float],
    ) -> List[float]:
        """Min temporal distance from each context timestamp to nearest mask timestamp."""
        return npo.compute_temporal_distances(context_timestamps, mask_timestamps)

    # ------------------------------------------------------------------
    # Modality configuration
    # ------------------------------------------------------------------

    def register_modality_config(
        self,
        modality: str,
        temporal_resolution: Optional[str] = None,
        alpha: Optional[float] = None,
    ) -> None:
        """Register modality-specific processing parameters."""
        cfg: Dict[str, Any] = {}
        if temporal_resolution is not None:
            cfg["temporal_resolution"] = temporal_resolution
        if alpha is not None:
            cfg["alpha"] = alpha
        self._modality_configs[modality] = cfg

    def set_modality_offset(self, modality: str, offset: List[float]) -> None:
        """Set an additive modality embedding offset (V-JEPA 2.1 modality tokens)."""
        self._modality_offsets[modality] = offset

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, sequence: EventSequence) -> List[float]:
        """Hierarchical temporal aggregation."""
        levels = self.process_multilevel(sequence)
        return levels[-1] if levels else []

    def process_multilevel(self, sequence: EventSequence) -> List[List[float]]:
        """Hierarchical temporal aggregation returning all intermediate levels."""
        if not sequence.embeddings:
            return [[]]

        # Sort by timestamp
        order = sorted(range(len(sequence.timestamps)), key=lambda i: sequence.timestamps[i])
        embeddings = [sequence.embeddings[i] for i in order]
        timestamps = [sequence.timestamps[i] for i in order]

        # Apply modality offset
        embeddings = self._apply_modality_offset(embeddings, sequence.modality)

        resolution = self._get_modality_resolution(sequence.modality)
        alpha = self._get_modality_alpha(sequence.modality)

        level_outputs: List[List[float]] = []

        for _level in range(self.num_levels):
            # Partition
            if resolution == "adaptive":
                windows = self._partition_adaptive(embeddings, timestamps)
            else:
                windows = self._partition_fixed(embeddings, timestamps)

            # Aggregate each window
            new_embeddings: List[List[float]] = []
            new_timestamps: List[float] = []
            for win in windows:
                win_embs = [embeddings[i] for i in win]
                win_ts = [timestamps[i] for i in win]
                agg = npo.weighted_aggregate(win_embs, win_ts, alpha=alpha)
                new_embeddings.append(agg)
                new_timestamps.append(max(win_ts))

            # Store intermediate level representation
            level_outputs.append(npo.weighted_aggregate(new_embeddings, new_timestamps, alpha=alpha))

            embeddings = new_embeddings
            timestamps = new_timestamps

        # Final aggregation across remaining windows
        final = npo.weighted_aggregate(embeddings, timestamps, alpha=alpha)
        level_outputs.append(final)

        return level_outputs

    @staticmethod
    def fuse_multilevel(level_representations: List[List[float]]) -> List[float]:
        """Fuse multi-level representations via element-wise mean-pooling."""
        return npo.fuse_multilevel(level_representations)

    def detect_patterns(self, representation: Iterable[float]) -> List[int]:
        """Detect salient dimensions via z-score thresholding."""
        return npo.detect_patterns_zscore(list(representation))

    def predict_next(self, sequence: EventSequence, num_steps: int = 1) -> List[List[float]]:
        """Exponentially-weighted moving-trend prediction."""
        if not sequence.embeddings:
            return []
        if len(sequence.embeddings) < 2:
            return [list(sequence.embeddings[-1]) for _ in range(num_steps)]

        # Sort by timestamp
        order = sorted(range(len(sequence.timestamps)), key=lambda i: sequence.timestamps[i])
        embeddings = [sequence.embeddings[i] for i in order]
        timestamps = [sequence.timestamps[i] for i in order]

        trend = npo.compute_trend(embeddings, timestamps, alpha=1.0)
        return npo.extrapolate(embeddings[-1], trend, num_steps)

    def predict_next_positional(
        self,
        sequence: EventSequence,
        target_timestamps: List[float],
    ) -> List[List[float]]:
        """Position-aware prediction conditioned on explicit target timestamps."""
        if not sequence.embeddings:
            return []
        if len(sequence.embeddings) < 2:
            return [list(sequence.embeddings[-1]) for _ in target_timestamps]

        # Sort by timestamp
        order = sorted(range(len(sequence.timestamps)), key=lambda i: sequence.timestamps[i])
        embeddings = [sequence.embeddings[i] for i in order]
        timestamps = [sequence.timestamps[i] for i in order]

        dim = len(embeddings[0])
        last = embeddings[-1]
        last_t = timestamps[-1]

        trend = npo.compute_trend(embeddings, timestamps, alpha=1.0)

        # Normalize dt to "steps" — trend is per-step, not per-second
        n = len(timestamps)
        avg_interval = (timestamps[-1] - timestamps[0]) / max(n - 1, 1)
        if avg_interval == 0.0:
            avg_interval = 1.0

        # Predict at each target timestamp with position-modulated trend
        predictions: List[List[float]] = []
        for target_t in target_timestamps:
            dt_steps = (target_t - last_t) / avg_interval
            pos_enc = npo.temporal_position_encoding(target_t - last_t, dim)
            pred = [last[d] + trend[d] * dt_steps * (1.0 + 0.1 * pos_enc[d]) for d in range(dim)]
            predictions.append(pred)
        return predictions

    def compute_regularized_loss(
        self,
        embeddings: List[List[float]],
        prediction_loss: float,
    ) -> float:
        """Combine prediction loss with an optional regularizer."""
        if self.regularizer is None:
            return prediction_loss
        reg_loss = self.regularizer(embeddings)
        return prediction_loss + self.reg_weight * reg_loss

    def compute_dense_loss(
        self,
        context_embeddings: List[List[float]],
        context_timestamps: List[float],
        target_embeddings: List[List[float]],
        mask_timestamps: List[float],
        prediction_loss: float,
        lambda_coeff: float = 0.5,
        distance_floor: float = 1.0,
    ) -> float:
        """Dense prediction loss with distance-weighted context supervision."""
        if not context_embeddings or not mask_timestamps:
            return prediction_loss

        distances = npo.compute_temporal_distances(context_timestamps, mask_timestamps)
        ctx_loss = npo.dense_context_loss(
            context_embeddings, target_embeddings, distances, lambda_coeff, distance_floor
        )
        return prediction_loss + ctx_loss

    def compute_multilevel_loss(
        self,
        level_embeddings: List[List[List[float]]],
        prediction_loss: float,
        level_weights: Optional[List[float]] = None,
    ) -> float:
        """Apply regularization at each hierarchical level (deep self-supervision)."""
        if self.regularizer is None:
            return prediction_loss

        n_levels = len(level_embeddings)
        if level_weights is None:
            level_weights = [1.0 / n_levels] * n_levels

        total_reg = 0.0
        for level_embs, w in zip(level_embeddings, level_weights):
            if level_embs:
                total_reg += w * self.regularizer(level_embs)

        return prediction_loss + self.reg_weight * total_reg

    @staticmethod
    def context_lambda_schedule(
        current_step: int,
        warmup_start: int = 50,
        warmup_end: int = 100,
        max_lambda: float = 0.5,
    ) -> float:
        """Compute context loss lambda with linear warmup schedule."""
        if current_step < warmup_start:
            return 0.0
        if current_step >= warmup_end:
            return max_lambda
        progress = (current_step - warmup_start) / max(warmup_end - warmup_start, 1)
        return max_lambda * progress
