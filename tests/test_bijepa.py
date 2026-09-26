"""Tests for bidirectional cycle-consistent JEPA training (BiJEPATrainer).

Skipped entirely if PyTorch is not available.
"""

import pytest

torch = pytest.importorskip("torch")

from event_jepa_cube.bijepa import BiJEPATrainer  # noqa: E402
from event_jepa_cube.predictors import MLPPredictor  # noqa: E402
from event_jepa_cube.regularizers import WeakSIGReg  # noqa: E402
from event_jepa_cube.sequence import EventSequence  # noqa: E402

EMB_DIM = 8
CONTEXT_LEN = 5
HIDDEN_DIM = 16
NUM_STEPS = 2


def _make_predictor(num_steps: int = NUM_STEPS) -> MLPPredictor:
    return MLPPredictor(
        embedding_dim=EMB_DIM,
        context_length=CONTEXT_LEN,
        hidden_dim=HIDDEN_DIM,
        num_steps=num_steps,
    )


def _make_sequence(length: int = 30, dim: int = EMB_DIM) -> EventSequence:
    """Linearly evolving sequence: learnable in both directions."""
    embeddings = [[0.1 * (i + d) for d in range(dim)] for i in range(length)]
    timestamps = [float(i) for i in range(length)]
    return EventSequence(embeddings=embeddings, timestamps=timestamps)


class TestBiJEPATrainerConstruction:
    def test_auto_mirrored_backward_predictor(self):
        torch.manual_seed(42)
        trainer = BiJEPATrainer(_make_predictor())
        bwd = trainer.backward_predictor
        # Backward predictor must invert the forward mapping's shape:
        # it consumes the target window and reconstructs the context window.
        assert bwd.embedding_dim == EMB_DIM
        assert bwd.context_length == NUM_STEPS
        assert bwd.num_steps == CONTEXT_LEN

    def test_mismatched_backward_predictor_rejected(self):
        torch.manual_seed(42)
        bad_bwd = MLPPredictor(
            embedding_dim=EMB_DIM, context_length=CONTEXT_LEN, hidden_dim=HIDDEN_DIM, num_steps=NUM_STEPS
        )
        # A backward predictor with forward-shaped windows cannot close the
        # cycle; constructing with it must fail loudly, not at train time.
        with pytest.raises(ValueError, match="mirror"):
            BiJEPATrainer(_make_predictor(), backward_predictor=bad_bwd)

    def test_mismatched_embedding_dim_rejected(self):
        torch.manual_seed(42)
        bad_bwd = MLPPredictor(embedding_dim=EMB_DIM * 2, context_length=NUM_STEPS, num_steps=CONTEXT_LEN)
        with pytest.raises(ValueError, match="embedding_dim"):
            BiJEPATrainer(_make_predictor(), backward_predictor=bad_bwd)


class TestBiJEPALosses:
    def test_loss_components_present_and_scalar(self):
        torch.manual_seed(42)
        trainer = BiJEPATrainer(_make_predictor())
        contexts = torch.randn(4, CONTEXT_LEN, EMB_DIM)
        targets = torch.randn(4, NUM_STEPS, EMB_DIM)
        losses = trainer.compute_losses(contexts, targets)
        for key in ("total", "forward", "backward", "cycle", "norm", "reg"):
            assert losses[key].dim() == 0, key

    def test_total_includes_cycle_and_norm(self):
        torch.manual_seed(42)
        contexts = torch.randn(4, CONTEXT_LEN, EMB_DIM)
        targets = torch.randn(4, NUM_STEPS, EMB_DIM)

        torch.manual_seed(7)
        with_cycle = BiJEPATrainer(_make_predictor(), cycle_weight=1.0, norm_weight=0.0)
        torch.manual_seed(7)
        without_cycle = BiJEPATrainer(_make_predictor(), cycle_weight=0.0, norm_weight=0.0)

        l_with = with_cycle.compute_losses(contexts, targets)
        l_without = without_cycle.compute_losses(contexts, targets)

        # Same seeds -> same predictors; the only difference is the cycle
        # term's contribution to the total. If this fails, cycle_weight is
        # decorative and the bidirectional objective is silently gone.
        expected = l_without["total"] + l_with["cycle"]
        assert torch.allclose(l_with["total"], expected, atol=1e-6)

    def test_regularizer_composes(self):
        torch.manual_seed(42)
        trainer = BiJEPATrainer(
            _make_predictor(),
            regularizer=WeakSIGReg(sketch_dim=8),
            reg_weight=0.05,
        )
        contexts = torch.randn(8, CONTEXT_LEN, EMB_DIM)
        targets = torch.randn(8, NUM_STEPS, EMB_DIM)
        losses = trainer.compute_losses(contexts, targets)
        assert losses["reg"].item() > 0.0


class TestBiJEPATraining:
    def test_training_reduces_loss(self):
        torch.manual_seed(42)
        trainer = BiJEPATrainer(_make_predictor(), lr=1e-2)
        sequences = [_make_sequence(30)]
        history = trainer.train(sequences, epochs=60, patience=60)
        # The bidirectional objective must be learnable on a linear
        # sequence; otherwise the joint optimization is broken.
        assert history["train_losses"][-1] < history["train_losses"][0]

    def test_training_improves_cycle_consistency(self):
        torch.manual_seed(42)
        trainer = BiJEPATrainer(_make_predictor(), lr=1e-2)
        sequences = [_make_sequence(30)]

        before = trainer.evaluate(sequences)["cycle_mse"]
        trainer.train(sequences, epochs=60, patience=60)
        after = trainer.evaluate(sequences)["cycle_mse"]

        # Cycle consistency is the point of BiJEPA: the round trip
        # g(f(ctx)) ~ ctx must actually tighten with training.
        assert after < before

    def test_backward_predictor_learns_inverse(self):
        torch.manual_seed(42)
        trainer = BiJEPATrainer(_make_predictor(), lr=1e-2)
        sequences = [_make_sequence(30)]

        before = trainer.evaluate(sequences)["backward_mse"]
        trainer.train(sequences, epochs=60, patience=60)
        after = trainer.evaluate(sequences)["backward_mse"]

        # The inverse (future -> past) direction must improve too — that is
        # the "informative signal in the inverse relationship" BiJEPA adds.
        assert after < before

    def test_history_tracks_components(self):
        torch.manual_seed(42)
        trainer = BiJEPATrainer(_make_predictor())
        history = trainer.train([_make_sequence(20)], epochs=3, patience=3)
        n = len(history["train_losses"])
        assert n > 0
        assert len(history["forward_losses"]) == n
        assert len(history["backward_losses"]) == n
        assert len(history["cycle_losses"]) == n
        assert len(history["norm_losses"]) == n
        assert len(history["val_losses"]) == n

    def test_empty_sequences(self):
        torch.manual_seed(42)
        trainer = BiJEPATrainer(_make_predictor())
        history = trainer.train([_make_sequence(3)], epochs=5)  # too short for windows
        assert history["train_losses"] == []
        metrics = trainer.evaluate([_make_sequence(3)])
        assert metrics["forward_mse"] == 0.0


class TestBiJEPAEvaluate:
    def test_evaluate_metric_keys(self):
        torch.manual_seed(42)
        trainer = BiJEPATrainer(_make_predictor())
        metrics = trainer.evaluate([_make_sequence(20)])
        assert set(metrics) == {"forward_mse", "backward_mse", "cycle_mse", "cosine_similarity"}
        assert metrics["forward_mse"] > 0.0
