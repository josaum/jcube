"""Leakage-aware evaluation and shared graph storage utilities."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any

import pyarrow as pa

from .arrow_ipc import ArrowIpcSharedTensor, ArrowIpcTensorView
from .sequence import EventSequence


def truncate_sequence_at_time(sequence: EventSequence, cutoff: float, *, inclusive: bool = True) -> EventSequence:
    """Return only events strictly before (or at) a label/outcome timestamp.

    Use this before fitting a downstream probe so outcome information cannot
    leak into the representation used as a feature.
    """
    embeddings: list[list[float]] = []
    timestamps: list[float] = []
    for embedding, timestamp in zip(sequence.embeddings, sequence.timestamps):
        if (inclusive and timestamp <= cutoff) or (not inclusive and timestamp < cutoff):
            embeddings.append(list(embedding))
            timestamps.append(timestamp)
    return EventSequence(embeddings=embeddings, timestamps=timestamps, modality=sequence.modality)


def temporal_probe_split(
    records: list[tuple[EventSequence, Any, float]],
    train_fraction: float = 0.8,
) -> tuple[list[tuple[EventSequence, Any]], list[tuple[EventSequence, Any]]]:
    """Build chronological, time-truncated train/test probe datasets.

    Each record is ``(sequence, label, label_time)``. Records are sorted by
    label time before splitting. Every sequence is truncated at its own label
    time, ensuring no post-outcome event becomes a probe feature.
    """
    if len(records) < 2:
        raise ValueError("At least two probe records are required")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")

    ordered = sorted(records, key=lambda record: record[2])
    split = int(len(ordered) * train_fraction)
    split = min(max(split, 1), len(ordered) - 1)

    def build(subset: list[tuple[EventSequence, Any, float]]) -> list[tuple[EventSequence, Any]]:
        return [(truncate_sequence_at_time(sequence, label_time), label) for sequence, label, label_time in subset]

    return build(ordered[:split]), build(ordered[split:])


class SharedGraphStore:
    """Zero-copy Arrow IPC file-backed graph tensor shared across DDP worker processes.

    Rank 0 creates and populates the uncompressed `.arrow` IPC file; other ranks
    attach by name and map the same physical Arrow data buffer pages in-place
    without deserializing separate graph copies or requiring `numpy`.
    """

    def __init__(self, name: str, shape: tuple[int, ...], dtype: str = "int64", *, create: bool = False) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ValueError("name must contain only letters, numbers, dots, underscores, and hyphens")
        if not shape or any(size <= 0 for size in shape):
            raise ValueError("shape must contain positive dimensions")

        self.name = name
        self.shape = tuple(int(d) for d in shape)
        self.dtype = dtype
        self.path = Path(tempfile.gettempdir()) / f"jcube_graph_{name}.arrow"
        if not create and not self.path.exists():
            raise FileNotFoundError(f"Shared graph does not exist: {self.path}")
        self._tensor: ArrowIpcSharedTensor | None = ArrowIpcSharedTensor(
            self.path,
            self.shape,
            dtype=dtype,
            create=create,
        )

    @property
    def edges(self) -> ArrowIpcTensorView:
        """Return the zero-copy shared Arrow IPC tensor view; mutations are visible to every rank."""
        if self._tensor is None:
            raise RuntimeError("SharedGraphStore is closed")
        return self._tensor.view

    def flush(self) -> None:
        """Flush dirty memory-mapped pages to the underlying `.arrow` IPC file."""
        if self._tensor is not None:
            self._tensor.flush()

    def read_arrow(self) -> pa.Table:
        """Read the backing `.arrow` file zero-copy as a `pyarrow.Table`."""
        if self._tensor is None:
            raise RuntimeError("SharedGraphStore is closed")
        return self._tensor.read_arrow()

    def to_ipc_bytes(self) -> bytes:
        """Return the raw `.arrow` IPC file bytes."""
        if self._tensor is None:
            raise RuntimeError("SharedGraphStore is closed")
        return self._tensor.to_ipc_bytes()

    def close(self) -> None:
        """Flush and release this rank's view without deleting shared data."""
        if self._tensor is not None:
            self._tensor.flush()
            self._tensor.close()
            self._tensor = None

    def unlink(self) -> None:
        """Close views and remove the backing `.arrow` file (called by the owner only)."""
        self.close()
        if self.path.exists():
            self.path.unlink()

    def __enter__(self) -> SharedGraphStore:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
