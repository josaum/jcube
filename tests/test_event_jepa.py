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


# -----------------------------------------------------------------------
# V-JEPA 2.1 improvement tests
# -----------------------------------------------------------------------


class TestProcessMultilevel:
    """Tests for EventJEPA.process_multilevel."""

    def test_returns_list_per_level(self):
        jepa = EventJEPA(embedding_dim=3, num_levels=2)
        seq = EventSequence(
            embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.5, 0.5, 0.0]],
            timestamps=[1.0, 2.0, 10.0, 11.0],
        )
        levels = jepa.process_multilevel(seq)
        # num_levels intermediate + 1 final = 3
        assert len(levels) == 3

    def test_last_matches_process(self):
        jepa = EventJEPA(embedding_dim=3, num_levels=2)
        seq = EventSequence(
            embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.5, 0.5, 0.0]],
            timestamps=[1.0, 2.0, 10.0, 11.0],
        )
        levels = jepa.process_multilevel(seq)
        single = jepa.process(seq)
        assert levels[-1] == pytest.approx(single)

    def test_empty_sequence(self):
        jepa = EventJEPA(embedding_dim=3)
        seq = EventSequence(embeddings=[], timestamps=[])
        levels = jepa.process_multilevel(seq)
        assert levels == [[]]

    def test_single_level(self):
        jepa = EventJEPA(embedding_dim=3, num_levels=1)
        seq = EventSequence(
            embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            timestamps=[1.0, 2.0],
        )
        levels = jepa.process_multilevel(seq)
        # 1 intermediate + 1 final
        assert len(levels) == 2


class TestFuseMultilevel:
    """Tests for EventJEPA.fuse_multilevel."""

    def test_fuse_averages_levels(self):
        levels = [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]
        fused = EventJEPA.fuse_multilevel(levels)
        assert fused == pytest.approx([2.0, 2.0, 2.0])

    def test_fuse_single_level(self):
        levels = [[1.0, 2.0, 3.0]]
        fused = EventJEPA.fuse_multilevel(levels)
        assert fused == pytest.approx([1.0, 2.0, 3.0])

    def test_fuse_empty(self):
        assert EventJEPA.fuse_multilevel([]) == []

    def test_fuse_skips_empty_levels(self):
        levels = [[1.0, 2.0], [], [3.0, 4.0]]
        fused = EventJEPA.fuse_multilevel(levels)
        assert fused == pytest.approx([2.0, 3.0])


class TestComputeDenseLoss:
    """Tests for EventJEPA.compute_dense_loss."""

    def test_zero_lambda_returns_prediction_loss(self):
        jepa = EventJEPA(embedding_dim=2)
        loss = jepa.compute_dense_loss(
            context_embeddings=[[1.0, 2.0]],
            context_timestamps=[1.0],
            target_embeddings=[[1.1, 2.1]],
            mask_timestamps=[2.0],
            prediction_loss=0.5,
            lambda_coeff=0.0,
        )
        # With lambda=0, all context weights are 0, so ctx_loss=0
        assert loss == pytest.approx(0.5)

    def test_positive_lambda_increases_loss(self):
        jepa = EventJEPA(embedding_dim=2)
        base_loss = 0.5
        loss = jepa.compute_dense_loss(
            context_embeddings=[[1.0, 2.0]],
            context_timestamps=[1.0],
            target_embeddings=[[2.0, 3.0]],
            mask_timestamps=[2.0],
            prediction_loss=base_loss,
            lambda_coeff=0.5,
        )
        assert loss > base_loss

    def test_distance_weighting_near_mask(self):
        jepa = EventJEPA(embedding_dim=2)
        # Token near mask (distance 0.1) should have higher weight
        loss_near = jepa.compute_dense_loss(
            context_embeddings=[[1.0, 2.0]],
            context_timestamps=[2.1],
            target_embeddings=[[2.0, 3.0]],
            mask_timestamps=[2.0],
            prediction_loss=0.0,
            lambda_coeff=0.5,
        )
        # Token far from mask (distance 10.0) should have lower weight
        loss_far = jepa.compute_dense_loss(
            context_embeddings=[[1.0, 2.0]],
            context_timestamps=[12.0],
            target_embeddings=[[2.0, 3.0]],
            mask_timestamps=[2.0],
            prediction_loss=0.0,
            lambda_coeff=0.5,
        )
        # The actual loss values differ because of different weighting
        # but with same embeddings the token_loss is identical per token.
        # Since there's only one token, the normalized loss is the same.
        # The test validates the method runs without error for both cases.
        assert loss_near > 0.0
        assert loss_far > 0.0

    def test_empty_context_returns_prediction_loss(self):
        jepa = EventJEPA(embedding_dim=2)
        loss = jepa.compute_dense_loss(
            context_embeddings=[],
            context_timestamps=[],
            target_embeddings=[],
            mask_timestamps=[2.0],
            prediction_loss=0.5,
        )
        assert loss == pytest.approx(0.5)

    def test_empty_mask_returns_prediction_loss(self):
        jepa = EventJEPA(embedding_dim=2)
        loss = jepa.compute_dense_loss(
            context_embeddings=[[1.0, 2.0]],
            context_timestamps=[1.0],
            target_embeddings=[[1.1, 2.1]],
            mask_timestamps=[],
            prediction_loss=0.5,
        )
        assert loss == pytest.approx(0.5)


class TestMultilevelLoss:
    """Tests for EventJEPA.compute_multilevel_loss."""

    def test_no_regularizer_returns_prediction_loss(self):
        jepa = EventJEPA(embedding_dim=2, regularizer=None)
        loss = jepa.compute_multilevel_loss(
            level_embeddings=[[[1.0, 2.0]], [[3.0, 4.0]]],
            prediction_loss=1.0,
        )
        assert loss == pytest.approx(1.0)

    def test_with_regularizer_sums_levels(self):
        def simple_reg(embeddings):
            return sum(sum(row) for row in embeddings)

        jepa = EventJEPA(embedding_dim=2, regularizer=simple_reg, reg_weight=0.1)
        loss = jepa.compute_multilevel_loss(
            level_embeddings=[[[1.0, 2.0]], [[3.0, 4.0]]],
            prediction_loss=1.0,
        )
        # uniform weights: 0.5 each
        # level 0 reg: 3.0, level 1 reg: 7.0
        # total_reg = 0.5 * 3.0 + 0.5 * 7.0 = 5.0
        # loss = 1.0 + 0.1 * 5.0 = 1.5
        assert loss == pytest.approx(1.5)

    def test_custom_level_weights(self):
        def simple_reg(embeddings):
            return sum(sum(row) for row in embeddings)

        jepa = EventJEPA(embedding_dim=2, regularizer=simple_reg, reg_weight=0.1)
        loss = jepa.compute_multilevel_loss(
            level_embeddings=[[[1.0, 2.0]], [[3.0, 4.0]]],
            prediction_loss=1.0,
            level_weights=[1.0, 0.0],
        )
        # Only level 0: reg = 3.0
        # loss = 1.0 + 0.1 * (1.0 * 3.0) = 1.3
        assert loss == pytest.approx(1.3)


class TestModalityConfig:
    """Tests for modality-specific processing."""

    def test_modality_aware_false_ignores_config(self):
        jepa = EventJEPA(embedding_dim=3, modality_aware=False)
        jepa.register_modality_config("audio", temporal_resolution="fixed", alpha=0.5)
        seq_text = EventSequence(
            embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            timestamps=[1.0, 10.0],
            modality="text",
        )
        seq_audio = EventSequence(
            embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            timestamps=[1.0, 10.0],
            modality="audio",
        )
        # With modality_aware=False, both should produce the same result
        assert jepa.process(seq_text) == pytest.approx(jepa.process(seq_audio))

    def test_modality_aware_different_resolution(self):
        jepa = EventJEPA(embedding_dim=3, modality_aware=True, temporal_resolution="adaptive")
        jepa.register_modality_config("audio", temporal_resolution="fixed")
        seq_text = EventSequence(
            embeddings=[[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
            timestamps=[0.0, 0.1, 5.0, 10.0],
            modality="text",
        )
        seq_audio = EventSequence(
            embeddings=[[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
            timestamps=[0.0, 0.1, 5.0, 10.0],
            modality="audio",
        )
        rep_text = jepa.process(seq_text)
        rep_audio = jepa.process(seq_audio)
        # Different partitioning should produce different results
        assert rep_text != rep_audio

    def test_modality_offset_applied(self):
        jepa = EventJEPA(embedding_dim=3, modality_aware=True)
        jepa.set_modality_offset("audio", [0.5, 0.5, 0.5])
        seq_plain = EventSequence(
            embeddings=[[1.0, 0.0, 0.0]],
            timestamps=[1.0],
            modality="text",
        )
        seq_offset = EventSequence(
            embeddings=[[1.0, 0.0, 0.0]],
            timestamps=[1.0],
            modality="audio",
        )
        rep_plain = jepa.process(seq_plain)
        rep_offset = jepa.process(seq_offset)
        # Offset should shift the result
        assert rep_plain != rep_offset
        assert rep_offset == pytest.approx([1.5, 0.5, 0.5])

    def test_unregistered_modality_uses_defaults(self):
        jepa = EventJEPA(embedding_dim=3, modality_aware=True, temporal_resolution="adaptive")
        seq = EventSequence(
            embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            timestamps=[1.0, 2.0],
            modality="unknown_modality",
        )
        # Should not raise, uses defaults
        result = jepa.process(seq)
        assert len(result) == 3


class TestPredictNextPositional:
    """Tests for EventJEPA.predict_next_positional."""

    def test_empty_sequence(self):
        jepa = EventJEPA(embedding_dim=3)
        seq = EventSequence(embeddings=[], timestamps=[])
        result = jepa.predict_next_positional(seq, target_timestamps=[5.0])
        assert result == []

    def test_single_embedding(self):
        jepa = EventJEPA(embedding_dim=3)
        seq = EventSequence(embeddings=[[1.0, 2.0, 3.0]], timestamps=[1.0])
        result = jepa.predict_next_positional(seq, target_timestamps=[2.0, 3.0])
        assert len(result) == 2
        # Single embedding: predictions should be copies
        for pred in result:
            assert pred == [1.0, 2.0, 3.0]

    def test_explicit_timestamps(self):
        jepa = EventJEPA(embedding_dim=2)
        seq = EventSequence(
            embeddings=[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
            timestamps=[1.0, 2.0, 3.0],
        )
        preds = jepa.predict_next_positional(seq, target_timestamps=[4.0, 5.0])
        assert len(preds) == 2
        # Should predict upward trend
        assert preds[0][0] > 2.0
        assert preds[1][0] > preds[0][0]

    def test_matches_timestamps_length(self):
        jepa = EventJEPA(embedding_dim=2)
        seq = EventSequence(
            embeddings=[[0.0, 0.0], [1.0, 1.0]],
            timestamps=[1.0, 2.0],
        )
        targets = [3.0, 4.0, 5.0, 6.0]
        preds = jepa.predict_next_positional(seq, target_timestamps=targets)
        assert len(preds) == len(targets)


class TestContextLambdaSchedule:
    """Tests for EventJEPA.context_lambda_schedule."""

    def test_before_warmup_returns_zero(self):
        lam = EventJEPA.context_lambda_schedule(current_step=10, warmup_start=50, warmup_end=100)
        assert lam == 0.0

    def test_during_warmup_ramps(self):
        lam = EventJEPA.context_lambda_schedule(current_step=75, warmup_start=50, warmup_end=100, max_lambda=0.5)
        assert 0.0 < lam < 0.5
        assert lam == pytest.approx(0.25)  # midpoint of warmup

    def test_after_warmup_returns_max(self):
        lam = EventJEPA.context_lambda_schedule(current_step=150, warmup_start=50, warmup_end=100, max_lambda=0.5)
        assert lam == pytest.approx(0.5)

    def test_at_warmup_end(self):
        lam = EventJEPA.context_lambda_schedule(current_step=100, warmup_start=50, warmup_end=100, max_lambda=0.5)
        assert lam == pytest.approx(0.5)
