"""NumPy-accelerated operations with pure-Python fallback.

When numpy is available, all heavy operations use vectorized BLAS-backed
routines.  When numpy is absent, the module falls back to equivalent
pure-Python implementations so the core package stays zero-dependency.

Also provides mmap-backed array storage for large sequences.
"""

from __future__ import annotations

import mmap
import os
import struct
import tempfile
from typing import List, Optional, Tuple

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]
    HAS_NUMPY = False


# -----------------------------------------------------------------------
# Type aliases
# -----------------------------------------------------------------------
# When numpy is available we work with ndarrays internally.
# The public API still accepts/returns List[float] for compatibility.


def to_array(data: list | object) -> object:
    """Convert list(s) to ndarray if numpy is available, else passthrough."""
    if HAS_NUMPY:
        return np.asarray(data, dtype=np.float64)
    return data


def to_list(data: object) -> list:
    """Convert ndarray back to nested Python list."""
    if HAS_NUMPY and isinstance(data, np.ndarray):
        return data.tolist()
    return list(data)  # type: ignore[arg-type]


# -----------------------------------------------------------------------
# Weighted aggregation
# -----------------------------------------------------------------------


def weighted_aggregate(
    embeddings: List[List[float]],
    timestamps: List[float],
    alpha: float = 1.0,
) -> List[float]:
    """Exponential-decay weighted average across embeddings."""
    if not embeddings:
        return []
    if HAS_NUMPY:
        emb = np.asarray(embeddings, dtype=np.float64)  # (N, D)
        ts = np.asarray(timestamps, dtype=np.float64)  # (N,)
        t_max = ts.max()
        weights = np.exp(-alpha * (t_max - ts))  # (N,)
        total_w = weights.sum()
        if total_w == 0.0:
            total_w = 1.0
        agg = (weights[:, None] * emb).sum(axis=0) / total_w  # (D,)
        return agg.tolist()
    # Fallback: pure Python
    import math

    dim = len(embeddings[0])
    t_max = max(timestamps)
    weights = [math.exp(-alpha * (t_max - t)) for t in timestamps]
    total_w = sum(weights)
    if total_w == 0.0:
        total_w = 1.0
    agg = [0.0] * dim
    for emb, w in zip(embeddings, weights):
        for i, v in enumerate(emb):
            agg[i] += v * w
    return [v / total_w for v in agg]


# -----------------------------------------------------------------------
# Fuse multilevel
# -----------------------------------------------------------------------


def fuse_multilevel(level_representations: List[List[float]]) -> List[float]:
    """Element-wise mean of non-empty level representations."""
    if not level_representations:
        return []
    non_empty = [r for r in level_representations if r]
    if not non_empty:
        return []
    if HAS_NUMPY:
        arr = np.asarray(non_empty, dtype=np.float64)
        return arr.mean(axis=0).tolist()
    dim = len(non_empty[0])
    n = len(non_empty)
    fused = [0.0] * dim
    for rep in non_empty:
        for i, v in enumerate(rep):
            fused[i] += v
    return [v / n for v in fused]


# -----------------------------------------------------------------------
# Dense loss
# -----------------------------------------------------------------------


def compute_temporal_distances(
    context_timestamps: List[float],
    mask_timestamps: List[float],
) -> List[float]:
    """Min temporal distance from each context timestamp to nearest mask timestamp."""
    if HAS_NUMPY:
        ct = np.asarray(context_timestamps, dtype=np.float64)[:, None]  # (C, 1)
        mt = np.asarray(mask_timestamps, dtype=np.float64)[None, :]  # (1, M)
        dists = np.abs(ct - mt).min(axis=1)  # (C,)
        return dists.tolist()
    distances: List[float] = []
    for c in context_timestamps:
        min_dist = min((abs(c - m) for m in mask_timestamps), default=0.0)
        distances.append(min_dist)
    return distances


def dense_context_loss(
    context_embeddings: List[List[float]],
    target_embeddings: List[List[float]],
    distances: List[float],
    lambda_coeff: float,
    distance_floor: float,
) -> float:
    """Compute weighted context L1 loss."""
    if HAS_NUMPY:
        pred = np.asarray(context_embeddings, dtype=np.float64)  # (C, D)
        tgt = np.asarray(target_embeddings, dtype=np.float64)  # (C, D)
        d = np.asarray(distances, dtype=np.float64)  # (C,)
        d = np.maximum(d, distance_floor)
        weights = lambda_coeff / np.sqrt(d)  # (C,)
        token_loss = np.abs(pred - tgt).mean(axis=1)  # (C,)
        total_weight = weights.sum()
        if total_weight == 0.0:
            return 0.0
        return float((weights * token_loss).sum() / total_weight)
    # Fallback
    import math

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
        return total_ctx_loss / total_weight
    return 0.0


# -----------------------------------------------------------------------
# Predict next (trend extrapolation)
# -----------------------------------------------------------------------


def compute_trend(
    embeddings: List[List[float]],
    timestamps: List[float],
    alpha: float = 1.0,
) -> List[float]:
    """Compute exponentially-weighted trend from consecutive deltas."""
    if HAS_NUMPY:
        emb = np.asarray(embeddings, dtype=np.float64)  # (N, D)
        ts = np.asarray(timestamps, dtype=np.float64)  # (N,)
        deltas = np.diff(emb, axis=0)  # (N-1, D)
        delta_times = ts[1:]  # (N-1,)
        t_max = delta_times.max()
        weights = np.exp(-alpha * (t_max - delta_times))  # (N-1,)
        total_w = weights.sum()
        if total_w == 0.0:
            total_w = 1.0
        trend = (weights[:, None] * deltas).sum(axis=0) / total_w  # (D,)
        return trend.tolist()
    # Fallback
    import math

    dim = len(embeddings[0])
    deltas = []
    delta_times = []
    for j in range(1, len(embeddings)):
        delta = [embeddings[j][d] - embeddings[j - 1][d] for d in range(dim)]
        deltas.append(delta)
        delta_times.append(timestamps[j])
    t_max = max(delta_times)
    weights = [math.exp(-alpha * (t_max - t)) for t in delta_times]
    total_w = sum(weights)
    if total_w == 0.0:
        total_w = 1.0
    trend = [0.0] * dim
    for delta, w in zip(deltas, weights):
        for d in range(dim):
            trend[d] += delta[d] * w
    return [v / total_w for v in trend]


def extrapolate(last: List[float], trend: List[float], num_steps: int) -> List[List[float]]:
    """Extrapolate from last embedding along trend."""
    if HAS_NUMPY:
        last_arr = np.asarray(last, dtype=np.float64)  # (D,)
        trend_arr = np.asarray(trend, dtype=np.float64)  # (D,)
        steps = np.arange(1, num_steps + 1, dtype=np.float64)[:, None]  # (S, 1)
        preds = last_arr[None, :] + trend_arr[None, :] * steps  # (S, D)
        return preds.tolist()
    dim = len(last)
    predictions = []
    for step in range(1, num_steps + 1):
        pred = [last[d] + trend[d] * step for d in range(dim)]
        predictions.append(pred)
    return predictions


# -----------------------------------------------------------------------
# Positional encoding
# -----------------------------------------------------------------------


def temporal_position_encoding(timestamp: float, dim: int) -> List[float]:
    """Sinusoidal position encoding."""
    if HAS_NUMPY:
        indices = np.arange(dim, dtype=np.float64)
        # For even: sin(t / 10000^(i/dim)), for odd: cos(t / 10000^((i-1)/dim))
        freqs = np.where(
            indices % 2 == 0,
            indices / dim,
            (indices - 1) / dim,
        )
        angles = timestamp / (10000.0**freqs)
        encoding = np.where(indices % 2 == 0, np.sin(angles), np.cos(angles))
        return encoding.tolist()
    import math

    encoding = [0.0] * dim
    for i in range(dim):
        if i % 2 == 0:
            encoding[i] = math.sin(timestamp / (10000.0 ** (i / dim)))
        else:
            encoding[i] = math.cos(timestamp / (10000.0 ** ((i - 1) / dim)))
    return encoding


# -----------------------------------------------------------------------
# Cosine similarity (single pair and batched)
# -----------------------------------------------------------------------


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors."""
    if HAS_NUMPY:
        va = np.asarray(a, dtype=np.float64)
        vb = np.asarray(b, dtype=np.float64)
        na = np.linalg.norm(va)
        nb = np.linalg.norm(vb)
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(np.dot(va, vb) / (na * nb))
    import math

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def batch_cosine_similarity(
    matrix_a: List[List[float]],
    matrix_b: List[List[float]],
) -> List[List[float]]:
    """Pairwise cosine similarity between rows of two matrices.

    Returns an (N_a, N_b) similarity matrix.
    """
    if HAS_NUMPY:
        a = np.asarray(matrix_a, dtype=np.float64)  # (Na, D)
        b = np.asarray(matrix_b, dtype=np.float64)  # (Nb, D)
        norms_a = np.linalg.norm(a, axis=1, keepdims=True)  # (Na, 1)
        norms_b = np.linalg.norm(b, axis=1, keepdims=True)  # (Nb, 1)
        # Avoid division by zero
        norms_a = np.maximum(norms_a, 1e-30)
        norms_b = np.maximum(norms_b, 1e-30)
        a_normed = a / norms_a
        b_normed = b / norms_b
        sim = a_normed @ b_normed.T  # (Na, Nb)
        return sim.tolist()
    # Fallback: O(Na * Nb * D)
    result = []
    for row_a in matrix_a:
        sims = [cosine_similarity(row_a, row_b) for row_b in matrix_b]
        result.append(sims)
    return result


# -----------------------------------------------------------------------
# Streaming EMA (vectorized batch)
# -----------------------------------------------------------------------


def streaming_ema_batch(
    representation: List[float],
    embeddings: List[List[float]],
    timestamps: List[float],
    alpha: float,
    last_timestamp: Optional[float],
    count: int,
    running_mean: List[float],
    running_m2: List[float],
) -> Tuple[List[float], float, int, List[float], List[float]]:
    """Process a batch of events through streaming EMA + Welford stats.

    Returns (new_repr, last_ts, new_count, new_mean, new_m2).
    """
    if not embeddings:
        return representation, last_timestamp or 0.0, count, running_mean, running_m2

    if HAS_NUMPY:
        emb = np.asarray(embeddings, dtype=np.float64)  # (N, D)
        ts = np.asarray(timestamps, dtype=np.float64)  # (N,)
        n = len(timestamps)

        repr_arr = np.asarray(representation, dtype=np.float64)  # (D,)
        mean_arr = np.asarray(running_mean, dtype=np.float64)  # (D,)
        m2_arr = np.asarray(running_m2, dtype=np.float64)  # (D,)

        current_count = count
        prev_ts = last_timestamp

        for i in range(n):
            e = emb[i]
            t = ts[i]
            if current_count == 0:
                repr_arr = e.copy()
            else:
                dt = t - prev_ts if prev_ts is not None else 0.0
                decay = np.exp(-alpha * dt)
                repr_arr = decay * repr_arr + (1.0 - decay) * e

            current_count += 1
            prev_ts = float(t)

            # Welford update (vectorized per-event)
            delta = e - mean_arr
            mean_arr = mean_arr + delta / current_count
            delta2 = e - mean_arr
            m2_arr = m2_arr + delta * delta2

        return (
            repr_arr.tolist(),
            float(prev_ts),
            current_count,
            mean_arr.tolist(),
            m2_arr.tolist(),
        )

    # Fallback: pure Python
    import math

    dim = len(representation)
    repr_list = list(representation)
    mean_list = list(running_mean)
    m2_list = list(running_m2)
    current_count = count
    prev_ts = last_timestamp

    for emb_item, t in zip(embeddings, timestamps):
        if current_count == 0:
            repr_list = list(emb_item)
        else:
            dt = t - prev_ts if prev_ts is not None else 0.0
            decay = math.exp(-alpha * dt)
            repr_list = [decay * r + (1.0 - decay) * e for r, e in zip(repr_list, emb_item)]

        current_count += 1
        prev_ts = t

        for d in range(dim):
            delta = emb_item[d] - mean_list[d]
            mean_list[d] += delta / current_count
            delta2 = emb_item[d] - mean_list[d]
            m2_list[d] += delta * delta2

    return repr_list, prev_ts, current_count, mean_list, m2_list


# -----------------------------------------------------------------------
# Detect patterns (z-score thresholding)
# -----------------------------------------------------------------------


def detect_patterns_zscore(
    values: List[float],
    z_threshold: float = 1.5,
    fallback_k: int = 5,
) -> List[int]:
    """Return indices of salient dimensions via z-score thresholding."""
    if len(values) < 2:
        return list(range(len(values)))

    if HAS_NUMPY:
        vals = np.asarray(values, dtype=np.float64)
        mean = vals.mean()
        std = vals.std()  # population std
        if std == 0.0:
            return list(range(min(fallback_k, len(values))))
        z = np.abs((vals - mean) / std)
        salient = np.where(z > z_threshold)[0].tolist()
        if len(salient) < 2:
            sorted_idx = np.argsort(-np.abs(vals))[:min(fallback_k, len(values))]
            return sorted_idx.tolist()
        return sorted(salient)

    import statistics

    mean = statistics.mean(values)
    stdev = statistics.pstdev(values)
    if stdev == 0.0:
        return list(range(min(fallback_k, len(values))))
    z_scores = [(abs((v - mean) / stdev), i) for i, v in enumerate(values)]
    salient = [i for z, i in z_scores if z > z_threshold]
    if len(salient) < 2:
        sorted_idx = sorted(range(len(values)), key=lambda i: abs(values[i]), reverse=True)
        return sorted_idx[: min(fallback_k, len(sorted_idx))]
    return sorted(salient)


# -----------------------------------------------------------------------
# Modality offset
# -----------------------------------------------------------------------


def apply_offset(
    embeddings: List[List[float]],
    offset: List[float],
) -> List[List[float]]:
    """Add offset vector to each embedding."""
    if HAS_NUMPY:
        emb = np.asarray(embeddings, dtype=np.float64)
        off = np.asarray(offset, dtype=np.float64)
        return (emb + off).tolist()
    return [[v + o for v, o in zip(emb, offset)] for emb in embeddings]


# -----------------------------------------------------------------------
# Mmap-backed embedding storage
# -----------------------------------------------------------------------


class MmapEmbeddingStore:
    """Memory-mapped storage for large embedding sequences.

    Stores embeddings as a flat array of float64 in a temp file,
    accessible via numpy mmap or fallback file I/O.
    """

    FLOAT64_SIZE = 8

    def __init__(self, embedding_dim: int, capacity: int = 0) -> None:
        self.embedding_dim = embedding_dim
        self._count = 0
        self._capacity = max(capacity, 256)
        self._fd: Optional[int] = None
        self._path: Optional[str] = None
        self._mmap: Optional[mmap.mmap] = None
        self._np_mmap: Optional[object] = None
        self._timestamps: List[float] = []

        if embedding_dim > 0:
            self._allocate(self._capacity)

    def _allocate(self, capacity: int) -> None:
        """Allocate (or grow) the backing file."""
        byte_size = capacity * self.embedding_dim * self.FLOAT64_SIZE
        if self._fd is not None:
            # Grow existing file
            if self._mmap is not None:
                self._mmap.close()
            os.ftruncate(self._fd, byte_size)
        else:
            fd, path = tempfile.mkstemp(suffix=".mmap", prefix="jcube_emb_")
            self._fd = fd
            self._path = path
            os.ftruncate(fd, byte_size)

        self._mmap = mmap.mmap(self._fd, byte_size)
        self._capacity = capacity

        if HAS_NUMPY:
            self._np_mmap = np.ndarray(
                (capacity, self.embedding_dim),
                dtype=np.float64,
                buffer=self._mmap,
            )

    def _ensure_capacity(self, needed: int) -> None:
        if needed > self._capacity:
            new_cap = max(needed, self._capacity * 2)
            self._allocate(new_cap)

    def append(self, embedding: List[float], timestamp: float) -> None:
        """Append a single embedding."""
        self._ensure_capacity(self._count + 1)
        if HAS_NUMPY and self._np_mmap is not None:
            self._np_mmap[self._count] = embedding
        else:
            offset = self._count * self.embedding_dim * self.FLOAT64_SIZE
            data = struct.pack(f"{self.embedding_dim}d", *embedding)
            self._mmap[offset : offset + len(data)] = data  # type: ignore[index]
        self._timestamps.append(timestamp)
        self._count += 1

    def append_batch(self, embeddings: List[List[float]], timestamps: List[float]) -> None:
        """Append multiple embeddings at once."""
        n = len(embeddings)
        if n == 0:
            return
        self._ensure_capacity(self._count + n)
        if HAS_NUMPY and self._np_mmap is not None:
            self._np_mmap[self._count : self._count + n] = embeddings
        else:
            for emb in embeddings:
                offset = self._count * self.embedding_dim * self.FLOAT64_SIZE
                data = struct.pack(f"{self.embedding_dim}d", *emb)
                self._mmap[offset : offset + len(data)] = data  # type: ignore[index]
                self._count += 1
            self._timestamps.extend(timestamps)
            return
        self._timestamps.extend(timestamps)
        self._count += n

    def get_embeddings(self, start: int = 0, end: Optional[int] = None) -> List[List[float]]:
        """Read embeddings as nested lists."""
        if end is None:
            end = self._count
        end = min(end, self._count)
        if start >= end:
            return []
        if HAS_NUMPY and self._np_mmap is not None:
            return self._np_mmap[start:end].tolist()
        result = []
        for i in range(start, end):
            offset = i * self.embedding_dim * self.FLOAT64_SIZE
            data = self._mmap[offset : offset + self.embedding_dim * self.FLOAT64_SIZE]  # type: ignore[index]
            row = list(struct.unpack(f"{self.embedding_dim}d", data))
            result.append(row)
        return result

    def get_numpy(self, start: int = 0, end: Optional[int] = None) -> object:
        """Return numpy view of stored embeddings (zero-copy if numpy available)."""
        if end is None:
            end = self._count
        end = min(end, self._count)
        if HAS_NUMPY and self._np_mmap is not None:
            return self._np_mmap[start:end]
        return self.get_embeddings(start, end)

    def get_timestamps(self, start: int = 0, end: Optional[int] = None) -> List[float]:
        if end is None:
            end = self._count
        return self._timestamps[start:end]

    @property
    def count(self) -> int:
        return self._count

    def clear(self) -> None:
        self._count = 0
        self._timestamps.clear()

    def close(self) -> None:
        """Release resources."""
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        self._np_mmap = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        if self._path is not None:
            try:
                os.unlink(self._path)
            except OSError:
                pass
            self._path = None

    def __del__(self) -> None:
        self.close()

    def __len__(self) -> int:
        return self._count
