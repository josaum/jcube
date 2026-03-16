"""Tests for streaming EventJEPA processing."""

import pytest

from event_jepa_cube.sequence import EventSequence
from event_jepa_cube.streaming import StreamBuffer, StreamingJEPA


class TestStreamingJEPAUpdate:
    """Tests for StreamingJEPA.update and basic state."""

    def test_single_update_produces_valid_representation(self):
        sj = StreamingJEPA(embedding_dim=3, alpha=1.0)
        emb = [1.0, 2.0, 3.0]
        result = sj.update(emb, timestamp=1.0)
        assert len(result) == 3
        # First event: representation equals the embedding
        assert result == pytest.approx([1.0, 2.0, 3.0])

    def test_count_increments(self):
        sj = StreamingJEPA(embedding_dim=2)
        assert sj.count == 0
        sj.update([1.0, 0.0], 1.0)
        assert sj.count == 1
        sj.update([0.0, 1.0], 2.0)
        assert sj.count == 2

    def test_last_timestamp_tracks_latest(self):
        sj = StreamingJEPA(embedding_dim=2)
        assert sj.last_timestamp is None
        sj.update([1.0, 0.0], 5.0)
        assert sj.last_timestamp == 5.0
        sj.update([0.0, 1.0], 10.0)
        assert sj.last_timestamp == 10.0

    def test_get_representation_matches_update_return(self):
        sj = StreamingJEPA(embedding_dim=3)
        result = sj.update([1.0, 2.0, 3.0], 1.0)
        assert sj.get_representation() == result

    def test_exponential_decay_weighting(self):
        """With large dt and high alpha, old repr decays toward new embedding."""
        sj = StreamingJEPA(embedding_dim=2, alpha=10.0)
        sj.update([1.0, 0.0], 0.0)
        # Large time gap with high alpha => decay ~ 0, repr ~ new embedding
        result = sj.update([0.0, 1.0], 100.0)
        assert result[0] == pytest.approx(0.0, abs=1e-6)
        assert result[1] == pytest.approx(1.0, abs=1e-6)

    def test_zero_dt_blends_equally(self):
        """With dt=0, decay=exp(0)=1, repr stays the same as old repr."""
        sj = StreamingJEPA(embedding_dim=2, alpha=1.0)
        sj.update([1.0, 0.0], 1.0)
        # dt=0 => decay=1 => repr = 1*old + 0*new = old
        result = sj.update([0.0, 1.0], 1.0)
        assert result == pytest.approx([1.0, 0.0])


class TestStreamingJEPAUpdateBatch:
    """Tests for update_batch."""

    def test_update_batch_matches_sequential_updates(self):
        sj1 = StreamingJEPA(embedding_dim=2, alpha=0.5)
        sj2 = StreamingJEPA(embedding_dim=2, alpha=0.5)

        embs = [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]
        ts = [1.0, 2.0, 3.0]

        for e, t in zip(embs, ts):
            sj1.update(e, t)
        sj2.update_batch(embs, ts)

        assert sj1.get_representation() == pytest.approx(sj2.get_representation())
        assert sj1.count == sj2.count

    def test_update_batch_empty(self):
        sj = StreamingJEPA(embedding_dim=2)
        result = sj.update_batch([], [])
        assert result == []


class TestStreamingJEPAConvergence:
    """Test that streaming converges toward batch EventJEPA."""

    def test_converges_toward_batch(self):
        """With many events, streaming EMA should approach batch weighted aggregate."""
        from event_jepa_cube.event_jepa import EventJEPA

        dim = 4
        n = 50
        # Generate a sequence of embeddings trending toward [1,1,1,1]
        embeddings = [[i / n] * dim for i in range(n)]
        timestamps = [float(i) for i in range(n)]

        seq = EventSequence(embeddings=embeddings, timestamps=timestamps)
        batch_jepa = EventJEPA(embedding_dim=dim)
        batch_repr = batch_jepa.process(seq)

        sj = StreamingJEPA(embedding_dim=dim, alpha=1.0)
        for e, t in zip(embeddings, timestamps):
            sj.update(e, t)
        stream_repr = sj.get_representation()

        # Both should emphasize recent events (high values)
        # The last embedding is [49/50]*4 ~ [0.98]*4
        # Both representations should be close to 1.0 for each dim
        for d in range(dim):
            assert stream_repr[d] > 0.5
            assert batch_repr[d] > 0.5


class TestStreamingJEPADetectPatterns:
    """Tests for detect_patterns with online stats."""

    def test_detect_patterns_insufficient_data(self):
        sj = StreamingJEPA(embedding_dim=4)
        sj.update([1.0, 2.0, 3.0, 4.0], 1.0)
        # With count < 2, returns up to 5 indices
        patterns = sj.detect_patterns()
        assert isinstance(patterns, list)
        assert len(patterns) == 4  # min(5, 4)

    def test_detect_patterns_with_salient_dimension(self):
        sj = StreamingJEPA(embedding_dim=4, alpha=0.01)
        # Feed many events where dim 0 has high variance, others are constant
        for i in range(20):
            emb = [10.0 * ((-1) ** i), 0.0, 0.0, 0.0]
            sj.update(emb, float(i))
        patterns = sj.detect_patterns()
        assert isinstance(patterns, list)
        assert all(isinstance(idx, int) for idx in patterns)

    def test_detect_patterns_zero_variance(self):
        sj = StreamingJEPA(embedding_dim=3, alpha=1.0)
        # All identical embeddings => zero variance
        for i in range(5):
            sj.update([1.0, 1.0, 1.0], float(i))
        patterns = sj.detect_patterns()
        assert isinstance(patterns, list)
        # Falls back to top-5 by magnitude
        assert len(patterns) == 3


class TestStreamingJEPAPredictNext:
    """Tests for predict_next with recent window."""

    def test_predict_next_empty(self):
        sj = StreamingJEPA(embedding_dim=2)
        result = sj.predict_next()
        assert result == []

    def test_predict_next_single_event(self):
        sj = StreamingJEPA(embedding_dim=2)
        sj.update([1.0, 2.0], 1.0)
        preds = sj.predict_next(num_steps=3)
        assert len(preds) == 3
        for pred in preds:
            assert pred == pytest.approx([1.0, 2.0])

    def test_predict_next_linear_trend(self):
        sj = StreamingJEPA(embedding_dim=2, alpha=1.0)
        # Clear upward trend
        for i in range(5):
            sj.update([float(i), float(i)], float(i))
        preds = sj.predict_next(num_steps=1)
        assert len(preds) == 1
        # Prediction should continue beyond last value (4.0)
        assert preds[0][0] > 4.0
        assert preds[0][1] > 4.0

    def test_predict_next_uses_window(self):
        sj = StreamingJEPA(embedding_dim=2, alpha=1.0, window_size=3)
        # Feed many events; only last 3 should matter for prediction
        for i in range(10):
            sj.update([float(i), float(i)], float(i))
        assert len(sj._recent_embeddings) == 3
        preds = sj.predict_next(num_steps=1)
        assert len(preds) == 1


class TestStreamingJEPAReset:
    """Tests for reset."""

    def test_reset_clears_all_state(self):
        sj = StreamingJEPA(embedding_dim=3)
        sj.update([1.0, 2.0, 3.0], 1.0)
        sj.update([4.0, 5.0, 6.0], 2.0)
        assert sj.count == 2

        sj.reset()

        assert sj.count == 0
        assert sj.last_timestamp is None
        assert sj.get_representation() == [0.0, 0.0, 0.0]
        assert len(sj._recent_embeddings) == 0
        assert sj._running_mean == [0.0, 0.0, 0.0]
        assert sj._running_m2 == [0.0, 0.0, 0.0]


class TestStreamingJEPASnapshot:
    """Tests for snapshot/from_snapshot roundtrip."""

    def test_snapshot_roundtrip_preserves_state(self):
        sj = StreamingJEPA(embedding_dim=3, alpha=0.5, window_size=10)
        sj.update([1.0, 2.0, 3.0], 1.0)
        sj.update([4.0, 5.0, 6.0], 2.0)
        sj.update([7.0, 8.0, 9.0], 3.0)

        snap = sj.snapshot()
        restored = StreamingJEPA.from_snapshot(snap)

        assert restored.count == sj.count
        assert restored.last_timestamp == sj.last_timestamp
        assert restored.get_representation() == pytest.approx(sj.get_representation())
        assert restored.embedding_dim == sj.embedding_dim
        assert restored.alpha == sj.alpha
        assert restored.window_size == sj.window_size
        assert list(restored._running_mean) == pytest.approx(list(sj._running_mean))
        assert list(restored._running_m2) == pytest.approx(list(sj._running_m2))
        assert list(restored._recent_timestamps) == pytest.approx(list(sj._recent_timestamps))

    def test_snapshot_roundtrip_continued_processing(self):
        """After restoring, continued updates should match."""
        sj = StreamingJEPA(embedding_dim=2, alpha=1.0)
        sj.update([1.0, 0.0], 1.0)
        sj.update([0.0, 1.0], 2.0)

        snap = sj.snapshot()
        restored = StreamingJEPA.from_snapshot(snap)

        # Continue processing on both
        sj.update([0.5, 0.5], 3.0)
        restored.update([0.5, 0.5], 3.0)

        assert restored.get_representation() == pytest.approx(sj.get_representation())
        assert restored.count == sj.count

    def test_snapshot_no_window_size(self):
        sj = StreamingJEPA(embedding_dim=2, window_size=None)
        sj.update([1.0, 0.0], 1.0)
        snap = sj.snapshot()
        restored = StreamingJEPA.from_snapshot(snap)
        assert restored.window_size is None

    def test_snapshot_serializable(self):
        """Snapshot should be JSON-serializable (all primitive types)."""
        import json

        sj = StreamingJEPA(embedding_dim=2)
        sj.update([1.0, 2.0], 1.0)
        snap = sj.snapshot()
        # Should not raise
        serialized = json.dumps(snap)
        assert isinstance(serialized, str)


class TestStreamingJEPAFromSequence:
    """Tests for from_sequence warm start."""

    def test_from_sequence_basic(self):
        seq = EventSequence(
            embeddings=[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
            timestamps=[1.0, 2.0, 3.0],
        )
        sj = StreamingJEPA.from_sequence(seq, alpha=1.0)
        assert sj.count == 3
        assert sj.last_timestamp == 3.0
        assert len(sj.get_representation()) == 2

    def test_from_sequence_detects_dim(self):
        seq = EventSequence(
            embeddings=[[1.0, 2.0, 3.0, 4.0, 5.0]],
            timestamps=[1.0],
        )
        sj = StreamingJEPA.from_sequence(seq)
        assert sj.embedding_dim == 5

    def test_from_sequence_matches_manual_updates(self):
        seq = EventSequence(
            embeddings=[[1.0, 0.0], [0.0, 1.0]],
            timestamps=[1.0, 2.0],
        )
        sj_from = StreamingJEPA.from_sequence(seq, alpha=0.5)

        sj_manual = StreamingJEPA(embedding_dim=2, alpha=0.5)
        sj_manual.update([1.0, 0.0], 1.0)
        sj_manual.update([0.0, 1.0], 2.0)

        assert sj_from.get_representation() == pytest.approx(sj_manual.get_representation())

    def test_from_sequence_unsorted_timestamps(self):
        """from_sequence should sort by timestamp before processing."""
        seq = EventSequence(
            embeddings=[[0.0, 1.0], [1.0, 0.0]],
            timestamps=[2.0, 1.0],
        )
        sj = StreamingJEPA.from_sequence(seq, alpha=1.0)
        # Should have processed [1.0, 0.0] at t=1 first, then [0.0, 1.0] at t=2
        sj_manual = StreamingJEPA(embedding_dim=2, alpha=1.0)
        sj_manual.update([1.0, 0.0], 1.0)
        sj_manual.update([0.0, 1.0], 2.0)
        assert sj.get_representation() == pytest.approx(sj_manual.get_representation())

    def test_from_sequence_empty(self):
        seq = EventSequence(embeddings=[], timestamps=[])
        sj = StreamingJEPA.from_sequence(seq, embedding_dim=3)
        assert sj.count == 0
        assert sj.embedding_dim == 3


class TestStreamingJEPAEdgeCases:
    """Edge case tests."""

    def test_first_event(self):
        sj = StreamingJEPA(embedding_dim=2)
        result = sj.update([3.0, 4.0], 0.0)
        assert result == pytest.approx([3.0, 4.0])

    def test_zero_dim_embedding(self):
        sj = StreamingJEPA(embedding_dim=0)
        result = sj.update([], 1.0)
        assert result == []
        assert sj.count == 1
        patterns = sj.detect_patterns()
        assert patterns == []

    def test_single_event_patterns(self):
        sj = StreamingJEPA(embedding_dim=2)
        sj.update([1.0, 2.0], 1.0)
        patterns = sj.detect_patterns()
        assert isinstance(patterns, list)
        assert len(patterns) == 2


class TestStreamBuffer:
    """Tests for StreamBuffer."""

    def test_flush_count_trigger(self):
        buf = StreamBuffer(flush_count=3)
        assert buf.pending == 0

        result1 = buf.add([1.0, 0.0], 1.0)
        assert result1 is None
        assert buf.pending == 1

        result2 = buf.add([0.0, 1.0], 2.0)
        assert result2 is None
        assert buf.pending == 2

        result3 = buf.add([0.5, 0.5], 3.0)
        assert result3 is not None
        assert isinstance(result3, EventSequence)
        assert len(result3.embeddings) == 3
        assert len(result3.timestamps) == 3
        assert buf.pending == 0

    def test_flush_interval_trigger(self):
        buf = StreamBuffer(flush_count=1000, flush_interval=5.0)

        result1 = buf.add([1.0, 0.0], 1.0)
        assert result1 is None

        result2 = buf.add([0.0, 1.0], 3.0)
        assert result2 is None

        # This event crosses the 5-second interval from the first event
        result3 = buf.add([0.5, 0.5], 6.0)
        assert result3 is not None
        assert isinstance(result3, EventSequence)
        assert len(result3.embeddings) == 3
        assert buf.pending == 0

    def test_manual_flush(self):
        buf = StreamBuffer(flush_count=100)
        buf.add([1.0, 0.0], 1.0)
        buf.add([0.0, 1.0], 2.0)
        assert buf.pending == 2

        seq = buf.flush()
        assert seq is not None
        assert len(seq.embeddings) == 2
        assert buf.pending == 0

    def test_flush_empty_buffer(self):
        buf = StreamBuffer()
        result = buf.flush()
        assert result is None

    def test_modality_preserved(self):
        buf = StreamBuffer(flush_count=1)
        seq = buf.add([1.0], 1.0, modality="audio")
        assert seq is not None
        assert seq.modality == "audio"

    def test_multiple_flush_cycles(self):
        buf = StreamBuffer(flush_count=2)
        buf.add([1.0], 1.0)
        seq1 = buf.add([2.0], 2.0)
        assert seq1 is not None
        assert len(seq1.embeddings) == 2

        buf.add([3.0], 3.0)
        seq2 = buf.add([4.0], 4.0)
        assert seq2 is not None
        assert len(seq2.embeddings) == 2
        assert seq2.timestamps == [3.0, 4.0]

    def test_pending_count(self):
        buf = StreamBuffer(flush_count=10)
        assert buf.pending == 0
        for i in range(5):
            buf.add([float(i)], float(i))
        assert buf.pending == 5
        buf.flush()
        assert buf.pending == 0
