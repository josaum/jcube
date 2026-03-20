"""Simplified Event-JEPA processor with V-JEPA 2.1 improvements."""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable
from typing import Any, Callable, Dict, List, Optional

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
        return [[v + o for v, o in zip(emb, offset)] for emb in embeddings]

    @staticmethod
    def _temporal_position_encoding(timestamp: float, dim: int) -> List[float]:
        """Sinusoidal position encoding for a temporal position."""
        encoding = [0.0] * dim
        for i in range(dim):
            if i % 2 == 0:
                encoding[i] = math.sin(timestamp / (10000.0 ** (i / dim)))
            else:
                encoding[i] = math.cos(timestamp / (10000.0 ** ((i - 1) / dim)))
        return encoding

    @staticmethod
    def _compute_temporal_distances(
        context_timestamps: List[float],
        mask_timestamps: List[float],
    ) -> List[float]:
        """Min temporal distance from each context timestamp to nearest mask timestamp."""
        distances: List[float] = []
        for ct in context_timestamps:
            min_dist = min((abs(ct - mt) for mt in mask_timestamps), default=0.0)
            distances.append(min_dist)
        return distances

    # ------------------------------------------------------------------
    # Modality configuration
    # ------------------------------------------------------------------

    def register_modality_config(
        self,
        modality: str,
        temporal_resolution: Optional[str] = None,
        alpha: Optional[float] = None,
    ) -> None:
        """Register modality-specific processing parameters.

        When ``modality_aware=True``, sequences with this modality will use
        the registered temporal_resolution and alpha instead of the defaults.
        """
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
        """Hierarchical temporal aggregation.

        1. Sort events by timestamp.
        2. Partition into temporal windows (adaptive or fixed).
        3. For each hierarchical level: aggregate within windows using
           exponential-decay weighting, then merge windows for the next level.
        4. Return the final aggregated representation.
        """
        levels = self.process_multilevel(sequence)
        return levels[-1] if levels else []

    def process_multilevel(self, sequence: EventSequence) -> List[List[float]]:
        """Hierarchical temporal aggregation returning all intermediate levels.

        Returns a list of representations, one per hierarchical level plus the
        final aggregation.  The last element matches what ``process()`` returns.

        Inspired by V-JEPA 2.1 deep self-supervision: intermediate encoder
        representations are preserved for multi-level loss computation.
        """
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
                agg = self._weighted_aggregate(win_embs, win_ts, alpha=alpha)
                new_embeddings.append(agg)
                new_timestamps.append(max(win_ts))

            # Store intermediate level representation
            level_outputs.append(self._weighted_aggregate(new_embeddings, new_timestamps, alpha=alpha))

            embeddings = new_embeddings
            timestamps = new_timestamps

        # Final aggregation across remaining windows
        final = self._weighted_aggregate(embeddings, timestamps, alpha=alpha)
        level_outputs.append(final)

        return level_outputs

    @staticmethod
    def fuse_multilevel(level_representations: List[List[float]]) -> List[float]:
        """Fuse multi-level representations via element-wise mean-pooling.

        Inspired by V-JEPA 2.1's multi-level concatenation + MLP fusion,
        adapted to a zero-dependency setting using simple averaging.

        Args:
            level_representations: List of embedding vectors, one per level.

        Returns:
            Single fused representation vector.
        """
        if not level_representations:
            return []
        non_empty = [r for r in level_representations if r]
        if not non_empty:
            return []
        dim = len(non_empty[0])
        n = len(non_empty)
        fused = [0.0] * dim
        for rep in non_empty:
            for i, v in enumerate(rep):
                fused[i] += v
        return [v / n for v in fused]

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

    def predict_next_positional(
        self,
        sequence: EventSequence,
        target_timestamps: List[float],
    ) -> List[List[float]]:
        """Position-aware prediction conditioned on explicit target timestamps.

        Instead of linear trend extrapolation by integer step count, this
        method predicts at specific future timestamps using sinusoidal temporal
        position encoding to modulate the trend by temporal distance.

        Inspired by V-JEPA 2.1's predictor which conditions on explicit
        spatio-temporal positions of masked tokens.

        Args:
            sequence: Input event sequence.
            target_timestamps: Future timestamps to predict at.

        Returns:
            List of predicted embeddings, one per target timestamp.
        """
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

        # Compute trend (same as predict_next)
        deltas: List[List[float]] = []
        delta_times: List[float] = []
        for j in range(1, len(embeddings)):
            delta = [embeddings[j][d] - embeddings[j - 1][d] for d in range(dim)]
            deltas.append(delta)
            delta_times.append(timestamps[j])

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

        # Predict at each target timestamp with position-modulated trend
        predictions: List[List[float]] = []
        for target_t in target_timestamps:
            dt = target_t - last_t
            pos_enc = self._temporal_position_encoding(dt, dim)
            # Modulate trend by position encoding: scale by (1 + pos_enc)
            pred = [last[d] + trend[d] * dt * (1.0 + 0.1 * pos_enc[d]) for d in range(dim)]
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
        """Dense prediction loss with distance-weighted context supervision.

        Implements V-JEPA 2.1's key insight: supervise ALL tokens, not just
        masked ones.  Context (visible) tokens are supervised with a weight
        inversely proportional to their temporal distance to the nearest
        masked token:

            lambda_i = lambda_coeff / sqrt(max(d_min(i, M), distance_floor))

        Combined loss: L_dense = prediction_loss + L_ctx

        Args:
            context_embeddings: Predicted embeddings for context (visible) tokens.
            context_timestamps: Timestamps of context tokens.
            target_embeddings: Target embeddings for context tokens (from EMA encoder).
            mask_timestamps: Timestamps of masked tokens.
            prediction_loss: Loss on masked token predictions (L_predict).
            lambda_coeff: Maximum context loss weight.
            distance_floor: Minimum distance to avoid division by zero.

        Returns:
            Combined dense loss: L_predict + L_ctx.
        """
        if not context_embeddings or not mask_timestamps:
            return prediction_loss

        distances = self._compute_temporal_distances(context_timestamps, mask_timestamps)
        dim = len(context_embeddings[0])

        total_ctx_loss = 0.0
        total_weight = 0.0
        for i, (pred, target) in enumerate(zip(context_embeddings, target_embeddings)):
            d = max(distances[i], distance_floor)
            weight = lambda_coeff / math.sqrt(d)
            token_loss = sum(abs(pred[d_] - target[d_]) for d_ in range(dim)) / dim
            total_ctx_loss += weight * token_loss
            total_weight += weight

        if total_weight > 0.0:
            ctx_loss = total_ctx_loss / total_weight
        else:
            ctx_loss = 0.0

        return prediction_loss + ctx_loss

    def compute_multilevel_loss(
        self,
        level_embeddings: List[List[List[float]]],
        prediction_loss: float,
        level_weights: Optional[List[float]] = None,
    ) -> float:
        """Apply regularization at each hierarchical level (deep self-supervision).

        Inspired by V-JEPA 2.1's approach of applying the self-supervised
        objective at multiple intermediate encoder layers.

        Args:
            level_embeddings: Embeddings at each level; each element is a list
                of embedding vectors for that level.
            prediction_loss: Base prediction loss.
            level_weights: Per-level regularization weights. Defaults to uniform.

        Returns:
            Combined loss: L_pred + sum_l(w_l * reg_weight * regularizer(level_l)).
        """
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
        """Compute context loss lambda with linear warmup schedule.

        V-JEPA 2.1 uses a progressive warmup of the context loss coefficient
        to stabilize training (epochs 50-100 in the paper).

        Args:
            current_step: Current training step or epoch.
            warmup_start: Step at which warmup begins.
            warmup_end: Step at which warmup completes.
            max_lambda: Maximum lambda value after warmup.

        Returns:
            Lambda value for the current step.
        """
        if current_step < warmup_start:
            return 0.0
        if current_step >= warmup_end:
            return max_lambda
        progress = (current_step - warmup_start) / max(warmup_end - warmup_start, 1)
        return max_lambda * progress
