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
        # Four clusters with unequal inter-cluster gaps so that adaptive
        # partitioning produces different windows at each hierarchical level.
        seq = EventSequence(
            embeddings=[
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 1.0, 0.0],
                [0.1, 0.9, 0.0],
                [0.0, 0.0, 1.0],
                [0.1, 0.0, 0.9],
                [0.5, 0.5, 0.0],
                [0.4, 0.6, 0.0],
            ],
            timestamps=[1.0, 1.1, 2.0, 2.1, 10.0, 10.1, 11.0, 11.1],
        )
        result_1 = EventJEPA(embedding_dim=3, num_levels=1).process(seq)
        result_3 = EventJEPA(embedding_dim=3, num_levels=3).process(seq)
        # With clustered timestamps and multiple levels, representations differ
        assert result_1 != result_3

    def test_process_adaptive_vs_fixed(self):
        # Irregular spacing where adaptive (gap > median) and fixed
        # (equal-width time bins) produce different window groupings.
        seq = EventSequence(
            embeddings=[
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.8, 0.2, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [0.1, 0.9, 0.0],
                [0.2, 0.8, 0.0],
                [0.5, 0.5, 0.0],
            ],
            timestamps=[0.0, 0.1, 0.2, 5.0, 5.1, 9.0, 9.1, 10.0],
        )
        adaptive = EventJEPA(embedding_dim=3, temporal_resolution="adaptive").process(seq)
        fixed = EventJEPA(embedding_dim=3, temporal_resolution="fixed").process(seq)
        assert adaptive != fixed


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
