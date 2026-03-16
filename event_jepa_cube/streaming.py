"""Online/incremental EventJEPA processing with streaming support."""

from __future__ import annotations

import math
from collections import deque

from .sequence import EventSequence


class StreamingJEPA:
    """Incremental EventJEPA that processes events one-at-a-time.

    Maintains a running representation that updates with O(1) per event
    instead of O(n) full reprocessing. Uses exponential moving averages
    and online statistics for pattern detection.

    The representation converges to the same result as batch EventJEPA
    for large sequences, with bounded approximation error for small ones.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        alpha: float = 1.0,
        window_size: int | None = None,
    ) -> None:
        """
        Args:
            embedding_dim: Dimension of embedding vectors.
            alpha: Exponential decay rate for temporal weighting.
            window_size: Optional sliding window size. If None, uses all history.
        """
        self.embedding_dim = embedding_dim
        self.alpha = alpha
        self.window_size = window_size

        self._representation: list[float] = [0.0] * embedding_dim
        self._count: int = 0
        self._last_timestamp: float | None = None
        # Welford's online algorithm state (per dimension)
        self._running_mean: list[float] = [0.0] * embedding_dim
        self._running_m2: list[float] = [0.0] * embedding_dim
        # Sliding window for predict_next
        maxlen = window_size if window_size is not None else None
        self._recent_embeddings: deque[list[float]] = deque(maxlen=maxlen)
        self._recent_timestamps: deque[float] = deque(maxlen=maxlen)

    def update(self, embedding: list[float], timestamp: float) -> list[float]:
        """Process a single new event and return updated representation.

        Uses exponential moving average weighted by time delta:
        weight = exp(-alpha * (t_new - t_last))
        new_repr = weight * old_repr + (1 - weight) * new_embedding

        Also updates online statistics (Welford's) for pattern detection.

        Args:
            embedding: New event embedding vector.
            timestamp: Event timestamp.

        Returns:
            Updated representation vector.
        """
        if self._count == 0:
            # First event: representation is the embedding itself
            self._representation = list(embedding)
        else:
            dt = timestamp - self._last_timestamp if self._last_timestamp is not None else 0.0
            decay = math.exp(-self.alpha * dt)
            self._representation = [decay * r + (1.0 - decay) * e for r, e in zip(self._representation, embedding)]

        self._count += 1
        self._last_timestamp = timestamp

        # Welford's online update for each dimension
        for d in range(len(embedding)):
            delta = embedding[d] - self._running_mean[d]
            self._running_mean[d] += delta / self._count
            delta2 = embedding[d] - self._running_mean[d]
            self._running_m2[d] += delta * delta2

        # Store in sliding window
        self._recent_embeddings.append(list(embedding))
        self._recent_timestamps.append(timestamp)

        return list(self._representation)

    def update_batch(self, embeddings: list[list[float]], timestamps: list[float]) -> list[float]:
        """Process multiple events sequentially, return final representation."""
        result: list[float] = []
        for emb, ts in zip(embeddings, timestamps):
            result = self.update(emb, ts)
        return result

    def get_representation(self) -> list[float]:
        """Return current representation without processing new events."""
        return list(self._representation)

    def detect_patterns(self) -> list[int]:
        """Detect salient dimensions using online statistics.

        Uses Welford's running mean/variance for z-score computation
        without needing to store all historical embeddings.

        Dimensions with |z| > 1.5 are considered salient. If fewer than 2
        dimensions pass the threshold, falls back to top-5 by magnitude.
        """
        if self._count < 2:
            return list(range(min(5, self.embedding_dim)))

        rep = self._representation
        # Compute per-dimension variance from Welford state
        variances = [self._running_m2[d] / (self._count - 1) for d in range(self.embedding_dim)]

        # Compute z-scores of representation relative to running stats
        z_scores: list[float] = []
        for d in range(self.embedding_dim):
            stdev = math.sqrt(variances[d]) if variances[d] > 0.0 else 0.0
            if stdev > 0.0:
                z_scores.append(abs((rep[d] - self._running_mean[d]) / stdev))
            else:
                z_scores.append(0.0)

        salient = [d for d in range(self.embedding_dim) if z_scores[d] > 1.5]

        if len(salient) < 2:
            sorted_idx = sorted(range(self.embedding_dim), key=lambda d: abs(rep[d]), reverse=True)
            return sorted_idx[: min(5, len(sorted_idx))]

        return sorted(salient)

    def predict_next(self, num_steps: int = 1) -> list[list[float]]:
        """Predict next embeddings using recent trend.

        Uses the sliding window of recent embeddings for trend estimation
        via exponentially-weighted moving deltas.
        """
        recent = list(self._recent_embeddings)
        if not recent:
            return []
        if len(recent) < 2:
            return [list(recent[-1]) for _ in range(num_steps)]

        dim = len(recent[0])
        recent_ts = list(self._recent_timestamps)

        # Compute deltas between consecutive recent embeddings
        deltas: list[list[float]] = []
        delta_times: list[float] = []
        for j in range(1, len(recent)):
            delta = [recent[j][d] - recent[j - 1][d] for d in range(dim)]
            deltas.append(delta)
            delta_times.append(recent_ts[j])

        # Exponential weighting by recency
        t_max = max(delta_times)
        weights = [math.exp(-self.alpha * (t_max - t)) for t in delta_times]
        total_w = sum(weights)
        if total_w == 0.0:
            total_w = 1.0

        trend = [0.0] * dim
        for delta, w in zip(deltas, weights):
            for d in range(dim):
                trend[d] += delta[d] * w
        trend = [v / total_w for v in trend]

        # Extrapolate from last embedding
        last = recent[-1]
        predictions: list[list[float]] = []
        for step in range(1, num_steps + 1):
            pred = [last[d] + trend[d] * step for d in range(dim)]
            predictions.append(pred)
        return predictions

    def reset(self) -> None:
        """Reset all internal state."""
        self._representation = [0.0] * self.embedding_dim
        self._count = 0
        self._last_timestamp = None
        self._running_mean = [0.0] * self.embedding_dim
        self._running_m2 = [0.0] * self.embedding_dim
        maxlen = self.window_size if self.window_size is not None else None
        self._recent_embeddings = deque(maxlen=maxlen)
        self._recent_timestamps = deque(maxlen=maxlen)

    @property
    def count(self) -> int:
        """Number of events processed."""
        return self._count

    @property
    def last_timestamp(self) -> float | None:
        """Most recent event timestamp."""
        return self._last_timestamp

    def snapshot(self) -> dict[str, object]:
        """Capture current state as a serializable dict.

        Returns dict with representation, count, statistics --
        can be stored and restored.
        """
        return {
            "embedding_dim": self.embedding_dim,
            "alpha": self.alpha,
            "window_size": self.window_size,
            "representation": list(self._representation),
            "count": self._count,
            "last_timestamp": self._last_timestamp,
            "running_mean": list(self._running_mean),
            "running_m2": list(self._running_m2),
            "recent_embeddings": [list(e) for e in self._recent_embeddings],
            "recent_timestamps": list(self._recent_timestamps),
        }

    @classmethod
    def from_snapshot(cls, snapshot: dict[str, object]) -> StreamingJEPA:
        """Restore from a previously captured snapshot."""
        embedding_dim = int(snapshot["embedding_dim"])  # type: ignore[arg-type]
        alpha = float(snapshot["alpha"])  # type: ignore[arg-type]
        window_size_raw = snapshot["window_size"]
        window_size = int(window_size_raw) if window_size_raw is not None else None  # type: ignore[arg-type]

        instance = cls(embedding_dim=embedding_dim, alpha=alpha, window_size=window_size)
        instance._representation = list(snapshot["representation"])  # type: ignore[arg-type]
        instance._count = int(snapshot["count"])  # type: ignore[arg-type]
        ts_raw = snapshot["last_timestamp"]
        instance._last_timestamp = float(ts_raw) if ts_raw is not None else None  # type: ignore[arg-type]
        instance._running_mean = list(snapshot["running_mean"])  # type: ignore[arg-type]
        instance._running_m2 = list(snapshot["running_m2"])  # type: ignore[arg-type]

        maxlen = window_size if window_size is not None else None
        recent_embs: list[list[float]] = snapshot["recent_embeddings"]  # type: ignore[assignment]
        recent_ts: list[float] = snapshot["recent_timestamps"]  # type: ignore[assignment]
        instance._recent_embeddings = deque([list(e) for e in recent_embs], maxlen=maxlen)
        instance._recent_timestamps = deque(list(recent_ts), maxlen=maxlen)

        return instance

    @classmethod
    def from_sequence(
        cls,
        sequence: EventSequence,
        embedding_dim: int = 768,
        alpha: float = 1.0,
        window_size: int | None = None,
    ) -> StreamingJEPA:
        """Initialize from an existing EventSequence (warm start).

        Processes the full sequence to establish state, then ready
        for incremental updates.

        Args:
            sequence: EventSequence to initialize from.
            embedding_dim: Dimension of embedding vectors.
            alpha: Exponential decay rate for temporal weighting.
            window_size: Optional sliding window size.
        """
        if sequence.embeddings:
            embedding_dim = len(sequence.embeddings[0])

        instance = cls(embedding_dim=embedding_dim, alpha=alpha, window_size=window_size)

        # Sort by timestamp and process in order
        order = sorted(range(len(sequence.timestamps)), key=lambda i: sequence.timestamps[i])
        for i in order:
            instance.update(sequence.embeddings[i], sequence.timestamps[i])

        return instance


class StreamBuffer:
    """Accumulates streaming events and periodically flushes to batch processing.

    Bridges streaming and batch worlds: collects events in a buffer,
    and when a flush condition is met (count or time threshold),
    yields an EventSequence for batch pipeline processing.
    """

    def __init__(
        self,
        flush_count: int = 100,
        flush_interval: float | None = None,
    ) -> None:
        """
        Args:
            flush_count: Flush when this many events accumulate.
            flush_interval: Flush when this many seconds elapse since last flush.
        """
        self.flush_count = flush_count
        self.flush_interval = flush_interval

        self._embeddings: list[list[float]] = []
        self._timestamps: list[float] = []
        self._modality: str = "text"
        self._last_flush_time: float | None = None

    def add(
        self,
        embedding: list[float],
        timestamp: float,
        modality: str = "text",
    ) -> EventSequence | None:
        """Add event. Returns EventSequence if flush triggered, else None."""
        self._embeddings.append(list(embedding))
        self._timestamps.append(timestamp)
        self._modality = modality

        # Check flush_count trigger
        if len(self._embeddings) >= self.flush_count:
            return self._do_flush()

        # Check flush_interval trigger
        if self.flush_interval is not None:
            if self._last_flush_time is None:
                self._last_flush_time = timestamp
            if (timestamp - self._last_flush_time) >= self.flush_interval:
                return self._do_flush()

        return None

    def flush(self) -> EventSequence | None:
        """Force flush. Returns EventSequence if buffer non-empty, else None."""
        if not self._embeddings:
            return None
        return self._do_flush()

    def _do_flush(self) -> EventSequence:
        """Internal: create EventSequence from buffer and clear."""
        seq = EventSequence(
            embeddings=self._embeddings,
            timestamps=self._timestamps,
            modality=self._modality,
        )
        self._embeddings = []
        self._timestamps = []
        self._last_flush_time = None
        return seq

    @property
    def pending(self) -> int:
        """Number of events in buffer."""
        return len(self._embeddings)
