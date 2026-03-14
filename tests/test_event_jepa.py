"""Tests for EventJEPA processor."""
import pytest

from event_jepa_cube.event_jepa import EventJEPA
from event_jepa_cube.sequence import EventSequence


class TestProcess:
    """Tests for EventJEPA.process."""

    def test_process_empty(self):
        jepa = EventJEPA(embedding_dim=3)
        seq = EventSequence(embeddings=[], timestamps=[])
        result = jepa.process(seq)
        assert result == []

    def test_process_single(self):
        jepa = EventJEPA(embedding_dim=3)
        seq = EventSequence(embeddings=[[1.0, 2.0, 3.0]], timestamps=[1.0])
        result = jepa.process(seq)
        assert len(result) == 3
        # Single embedding with exponential weight exp(0)=1 should return itself
        assert result == pytest.approx([1.0, 2.0, 3.0])

    def test_process_basic(self):
        jepa = EventJEPA(embedding_dim=3)
        seq = EventSequence(
            embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            timestamps=[1.0, 2.0, 3.0],
        )
        result = jepa.process(seq)
        assert len(result) == 3
        # Result should be non-empty floats
        assert all(isinstance(v, float) for v in result)

    def test_process_respects_num_levels(self):
        # Use non-uniform timestamps so adaptive partitioning creates
        # multiple windows that get further aggregated at higher levels.
        seq = EventSequence(
            embeddings=[
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.5, 0.5, 0.5],
                [0.0, 0.0, 0.0],
            ],
            timestamps=[1.0, 1.1, 5.0, 5.1, 10.0, 10.1, 15.0, 15.1],
        )
        result_1 = EventJEPA(embedding_dim=3, num_levels=1).process(seq)
        result_3 = EventJEPA(embedding_dim=3, num_levels=3).process(seq)
        # Both should produce valid 3-dim representations
        assert len(result_1) == 3
        assert len(result_3) == 3
        # num_levels is used (doesn't crash, produces finite values)
        assert all(isinstance(v, float) for v in result_3)

    def test_process_adaptive_vs_fixed(self):
        # Many events with irregular spacing so adaptive and fixed partition differently
        seq = EventSequence(
            embeddings=[
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.8, 0.2, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [0.1, 0.9, 0.0],
            ],
            timestamps=[1.0, 1.5, 2.0, 10.0, 10.5, 11.0],
        )
        adaptive = EventJEPA(embedding_dim=3, temporal_resolution="adaptive").process(seq)
        fixed = EventJEPA(embedding_dim=3, temporal_resolution="fixed").process(seq)
        # Both should produce valid 3-dim representations
        assert len(adaptive) == 3
        assert len(fixed) == 3
        assert all(isinstance(v, float) for v in adaptive)
        assert all(isinstance(v, float) for v in fixed)


class TestDetectPatterns:
    """Tests for EventJEPA.detect_patterns."""

    def test_detect_patterns_returns_indices(self):
        jepa = EventJEPA(embedding_dim=5)
        rep = [0.1, 0.2, 5.0, 0.1, 0.15]  # dimension 2 is salient
        indices = jepa.detect_patterns(rep)
        assert isinstance(indices, list)
        assert all(isinstance(i, int) for i in indices)
        assert all(0 <= i < len(rep) for i in indices)

    def test_detect_patterns_zero_variance(self):
        jepa = EventJEPA(embedding_dim=4)
        rep = [1.0, 1.0, 1.0, 1.0]
        indices = jepa.detect_patterns(rep)
        assert isinstance(indices, list)
        assert all(isinstance(i, int) for i in indices)
        # With zero variance, falls back to top-5 (or all 4 since len < 5)
        assert len(indices) == 4


class TestPredictNext:
    """Tests for EventJEPA.predict_next."""

    def test_predict_next_empty(self):
        jepa = EventJEPA(embedding_dim=3)
        seq = EventSequence(embeddings=[], timestamps=[])
        result = jepa.predict_next(seq)
        assert result == []

    def test_predict_next_single(self):
        jepa = EventJEPA(embedding_dim=3)
        seq = EventSequence(embeddings=[[1.0, 2.0, 3.0]], timestamps=[1.0])
        result = jepa.predict_next(seq, num_steps=3)
        assert len(result) == 3
        # Single embedding: all predictions should be copies of it
        for pred in result:
            assert pred == [1.0, 2.0, 3.0]

    def test_predict_next_trend(self):
        jepa = EventJEPA(embedding_dim=2)
        # Clear linear trend: each step adds [1.0, 1.0]
        seq = EventSequence(
            embeddings=[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
            timestamps=[1.0, 2.0, 3.0, 4.0],
        )
        preds = jepa.predict_next(seq, num_steps=1)
        assert len(preds) == 1
        # Prediction should continue upward: each dim > 3.0
        assert preds[0][0] > 3.0
        assert preds[0][1] > 3.0

    def test_predict_next_num_steps(self):
        jepa = EventJEPA(embedding_dim=2)
        seq = EventSequence(
            embeddings=[[0.0, 0.0], [1.0, 1.0]],
            timestamps=[1.0, 2.0],
        )
        preds = jepa.predict_next(seq, num_steps=5)
        assert len(preds) == 5


class TestComputeRegularizedLoss:
    """Tests for EventJEPA.compute_regularized_loss."""

    def test_compute_regularized_loss_no_reg(self):
        jepa = EventJEPA(embedding_dim=3, regularizer=None)
        loss = jepa.compute_regularized_loss([[0.1, 0.2]], prediction_loss=1.5)
        assert loss == 1.5

    def test_compute_regularized_loss_with_reg(self):
        # Simple regularizer: sum of all values
        def simple_reg(embeddings):
            return sum(sum(row) for row in embeddings)

        jepa = EventJEPA(
            embedding_dim=3,
            regularizer=simple_reg,
            reg_weight=0.1,
        )
        embeddings = [[1.0, 2.0, 3.0]]
        prediction_loss = 1.0
        loss = jepa.compute_regularized_loss(embeddings, prediction_loss)
        expected = 1.0 + 0.1 * 6.0  # 1.0 + 0.1 * (1+2+3)
        assert loss == pytest.approx(expected)
