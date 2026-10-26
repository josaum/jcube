"""Tests for leakage-aware probes and shared graph storage."""

import uuid

import pytest

from event_jepa_cube.evaluation import SharedGraphStore, temporal_probe_split, truncate_sequence_at_time
from event_jepa_cube.sequence import EventSequence


def _sequence(offset: int = 0) -> EventSequence:
    return EventSequence(
        embeddings=[[float(offset + 1), 0.0], [float(offset + 2), 0.0], [999.0, 999.0]],
        timestamps=[1.0 + offset, 2.0 + offset, 3.0 + offset],
    )


def test_truncation_removes_outcome_and_future_events():
    truncated = truncate_sequence_at_time(_sequence(), cutoff=2.5)

    assert truncated.timestamps == [1.0, 2.0]
    assert truncated.embeddings == [[1.0, 0.0], [2.0, 0.0]]
    assert 999.0 not in truncated.embeddings


def test_temporal_probe_split_is_chronological_and_leakage_free():
    records = [(_sequence(2), "later", 4.5), (_sequence(0), "earlier", 2.5), (_sequence(1), "middle", 3.5)]

    train, test = temporal_probe_split(records, train_fraction=0.5)

    assert [label for _, label in train] == ["earlier"]
    assert [label for _, label in test] == ["middle", "later"]
    for sequence, _ in train + test:
        assert 999.0 not in sequence.embeddings
    assert max(sequence.timestamps[-1] for sequence, _ in train) < min(
        sequence.timestamps[-1] for sequence, _ in test
    )


def test_temporal_probe_split_validates_inputs():
    with pytest.raises(ValueError, match="At least two"):
        temporal_probe_split([])
    with pytest.raises(ValueError, match="train_fraction"):
        temporal_probe_split([(_sequence(), 1, 2.0), (_sequence(), 2, 3.0)], train_fraction=1.0)


def test_shared_graph_store_attach_and_mutate():
    name = f"test-{uuid.uuid4().hex}"
    owner = SharedGraphStore(name, (2, 3), create=True)
    owner.edges[:] = [[1, 2, 3], [4, 5, 6]]
    owner.edges.flush()

    worker = SharedGraphStore(name, (2, 3))
    assert worker.edges.tolist() == [[1, 2, 3], [4, 5, 6]]
    worker.edges[0, 0] = 42
    worker.edges.flush()
    assert owner.edges[0, 0] == 42

    worker.close()
    owner.unlink()
    with pytest.raises(FileNotFoundError):
        SharedGraphStore(name, (2, 3))
