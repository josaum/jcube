"""Tests for learnable neural predictors (MLPPredictor, TransformerPredictor, PredictorTrainer).

Skipped entirely if PyTorch is not available.
"""

import pytest

torch = pytest.importorskip("torch")

from event_jepa_cube.predictors import MLPPredictor, PredictorTrainer, TransformerPredictor  # noqa: E402
from event_jepa_cube.sequence import EventSequence  # noqa: E402

# Use small dims for fast tests
EMB_DIM = 8
CONTEXT_LEN = 5
HIDDEN_DIM = 16


def _make_sequence(length: int = 20, dim: int = EMB_DIM) -> EventSequence:
    """Create a simple EventSequence with linearly increasing values."""
    embeddings = [[float(i + d) for d in range(dim)] for i in range(length)]
    timestamps = [float(i) for i in range(length)]
    return EventSequence(embeddings=embeddings, timestamps=timestamps)


def _make_raw_embeddings(length: int = 20, dim: int = EMB_DIM) -> list:
    """Create raw embedding lists (not wrapped in EventSequence)."""
    return [[float(i + d) for d in range(dim)] for i in range(length)]


class TestMLPPredictor:
    """Tests for MLPPredictor."""

    def test_forward_shape(self):
        """Test that forward produces correct output shape."""
        torch.manual_seed(42)
        predictor = MLPPredictor(embedding_dim=EMB_DIM, context_length=CONTEXT_LEN, hidden_dim=HIDDEN_DIM, num_steps=1)
        context = torch.randn(4, CONTEXT_LEN, EMB_DIM)
        out = predictor(context)
        assert out.shape == (4, 1, EMB_DIM)

    def test_forward_multi_step(self):
        """Test multi-step prediction output shape."""
        torch.manual_seed(42)
        predictor = MLPPredictor(embedding_dim=EMB_DIM, context_length=CONTEXT_LEN, hidden_dim=HIDDEN_DIM, num_steps=3)
        context = torch.randn(2, CONTEXT_LEN, EMB_DIM)
        out = predictor(context)
        assert out.shape == (2, 3, EMB_DIM)

    def test_predict_from_sequence(self):
        """Test predict_from_sequence returns correct number of predictions."""
        torch.manual_seed(42)
        predictor = MLPPredictor(embedding_dim=EMB_DIM, context_length=CONTEXT_LEN, hidden_dim=HIDDEN_DIM, num_steps=1)
        embeddings = _make_raw_embeddings(10)
        result = predictor.predict_from_sequence(embeddings)
        assert len(result) == 1
        assert len(result[0]) == EMB_DIM

    def test_predict_from_sequence_override_steps(self):
        """Test overriding num_steps in predict_from_sequence."""
        torch.manual_seed(42)
        predictor = MLPPredictor(embedding_dim=EMB_DIM, context_length=CONTEXT_LEN, hidden_dim=HIDDEN_DIM, num_steps=1)
        embeddings = _make_raw_embeddings(10)
        result = predictor.predict_from_sequence(embeddings, num_steps=3)
        assert len(result) == 3
        for pred in result:
            assert len(pred) == EMB_DIM

    def test_predict_from_sequence_short_padding(self):
        """Test that short sequences are padded correctly."""
        torch.manual_seed(42)
        predictor = MLPPredictor(embedding_dim=EMB_DIM, context_length=CONTEXT_LEN, hidden_dim=HIDDEN_DIM, num_steps=1)
        # Only 2 embeddings, context_length is 5
        embeddings = _make_raw_embeddings(2)
        result = predictor.predict_from_sequence(embeddings)
        assert len(result) == 1
        assert len(result[0]) == EMB_DIM


class TestTransformerPredictor:
    """Tests for TransformerPredictor."""

    def test_forward_shape(self):
        """Test that forward produces correct output shape."""
        torch.manual_seed(42)
        predictor = TransformerPredictor(
            embedding_dim=EMB_DIM, context_length=CONTEXT_LEN, num_heads=2, num_layers=1, num_steps=1
        )
        context = torch.randn(4, CONTEXT_LEN, EMB_DIM)
        out = predictor(context)
        assert out.shape == (4, 1, EMB_DIM)

    def test_forward_multi_step(self):
        """Test multi-step prediction output shape."""
        torch.manual_seed(42)
        predictor = TransformerPredictor(
            embedding_dim=EMB_DIM, context_length=CONTEXT_LEN, num_heads=2, num_layers=1, num_steps=3
        )
        context = torch.randn(2, CONTEXT_LEN, EMB_DIM)
        out = predictor(context)
        assert out.shape == (2, 3, EMB_DIM)

    def test_forward_shorter_context(self):
        """Test that sequences shorter than context_length work."""
        torch.manual_seed(42)
        predictor = TransformerPredictor(
            embedding_dim=EMB_DIM, context_length=CONTEXT_LEN, num_heads=2, num_layers=1, num_steps=1
        )
        # seq_len=3, which is less than context_length=5
        context = torch.randn(2, 3, EMB_DIM)
        out = predictor(context)
        assert out.shape == (2, 1, EMB_DIM)

    def test_predict_from_sequence(self):
        """Test predict_from_sequence returns correct number of predictions."""
        torch.manual_seed(42)
        predictor = TransformerPredictor(
            embedding_dim=EMB_DIM, context_length=CONTEXT_LEN, num_heads=2, num_layers=1, num_steps=1
        )
        embeddings = _make_raw_embeddings(10)
        result = predictor.predict_from_sequence(embeddings)
        assert len(result) == 1
        assert len(result[0]) == EMB_DIM

    def test_predict_from_sequence_short_padding(self):
        """Test that short sequences are padded correctly."""
        torch.manual_seed(42)
        predictor = TransformerPredictor(
            embedding_dim=EMB_DIM, context_length=CONTEXT_LEN, num_heads=2, num_layers=1, num_steps=1
        )
        embeddings = _make_raw_embeddings(2)
        result = predictor.predict_from_sequence(embeddings)
        assert len(result) == 1
        assert len(result[0]) == EMB_DIM


class TestPredictorTrainer:
    """Tests for PredictorTrainer."""

    def test_prepare_dataset_shapes(self):
        """Test that prepare_dataset creates correct sliding windows."""
        torch.manual_seed(42)
        predictor = MLPPredictor(embedding_dim=EMB_DIM, context_length=CONTEXT_LEN, hidden_dim=HIDDEN_DIM, num_steps=1)
        trainer = PredictorTrainer(predictor)
        seq = _make_sequence(20)
        contexts, targets = trainer.prepare_dataset([seq])

        # 20 embeddings, context=5, steps=1 -> 15 windows
        expected_windows = 20 - CONTEXT_LEN - 1 + 1
        assert contexts.shape == (expected_windows, CONTEXT_LEN, EMB_DIM)
        assert targets.shape == (expected_windows, 1, EMB_DIM)

    def test_prepare_dataset_raw_embeddings(self):
        """Test prepare_dataset with raw embedding lists."""
        torch.manual_seed(42)
        predictor = MLPPredictor(embedding_dim=EMB_DIM, context_length=CONTEXT_LEN, hidden_dim=HIDDEN_DIM, num_steps=1)
        trainer = PredictorTrainer(predictor)
        embs = _make_raw_embeddings(20)
        contexts, targets = trainer.prepare_dataset([embs])

        expected_windows = 20 - CONTEXT_LEN - 1 + 1
        assert contexts.shape == (expected_windows, CONTEXT_LEN, EMB_DIM)
        assert targets.shape == (expected_windows, 1, EMB_DIM)

    def test_prepare_dataset_multi_step(self):
        """Test prepare_dataset with multi-step targets."""
        torch.manual_seed(42)
        predictor = MLPPredictor(embedding_dim=EMB_DIM, context_length=CONTEXT_LEN, hidden_dim=HIDDEN_DIM, num_steps=3)
        trainer = PredictorTrainer(predictor)
        seq = _make_sequence(20)
        contexts, targets = trainer.prepare_dataset([seq])

        expected_windows = 20 - CONTEXT_LEN - 3 + 1
        assert contexts.shape == (expected_windows, CONTEXT_LEN, EMB_DIM)
        assert targets.shape == (expected_windows, 3, EMB_DIM)

    def test_prepare_dataset_multiple_sequences(self):
        """Test prepare_dataset with multiple sequences."""
        torch.manual_seed(42)
        predictor = MLPPredictor(embedding_dim=EMB_DIM, context_length=CONTEXT_LEN, hidden_dim=HIDDEN_DIM, num_steps=1)
        trainer = PredictorTrainer(predictor)
        seqs = [_make_sequence(10), _make_sequence(15)]
        contexts, targets = trainer.prepare_dataset(seqs)

        expected = (10 - CONTEXT_LEN - 1 + 1) + (15 - CONTEXT_LEN - 1 + 1)
        assert contexts.shape[0] == expected

    def test_train_returns_history(self):
        """Test that train runs and returns history with expected keys."""
        torch.manual_seed(42)
        predictor = MLPPredictor(embedding_dim=EMB_DIM, context_length=CONTEXT_LEN, hidden_dim=HIDDEN_DIM, num_steps=1)
        trainer = PredictorTrainer(predictor, lr=1e-2)
        seq = _make_sequence(30)
        history = trainer.train([seq], epochs=20, patience=50)

        assert "train_losses" in history
        assert "val_losses" in history
        assert "best_epoch" in history
        assert len(history["train_losses"]) > 0
        assert len(history["val_losses"]) > 0

    def test_train_loss_decreases(self):
        """Test that training loss generally decreases."""
        torch.manual_seed(42)
        predictor = MLPPredictor(embedding_dim=EMB_DIM, context_length=3, hidden_dim=HIDDEN_DIM, num_steps=1)
        trainer = PredictorTrainer(predictor, lr=1e-2)
        # Use a simple pattern that's easy to learn
        embs = [[float(i)] * EMB_DIM for i in range(50)]
        history = trainer.train([embs], epochs=50, patience=100)

        losses = history["train_losses"]
        # First loss should be larger than last loss (training made progress)
        assert losses[0] > losses[-1]

    def test_train_early_stopping(self):
        """Test that early stopping works."""
        torch.manual_seed(42)
        predictor = MLPPredictor(embedding_dim=EMB_DIM, context_length=CONTEXT_LEN, hidden_dim=HIDDEN_DIM, num_steps=1)
        trainer = PredictorTrainer(predictor, lr=1e-5)  # Very low lr to trigger early stopping
        seq = _make_sequence(30)
        history = trainer.train([seq], epochs=200, patience=5)

        # Should stop before 200 epochs
        assert len(history["train_losses"]) < 200

    def test_evaluate_returns_metrics(self):
        """Test that evaluate returns expected metrics."""
        torch.manual_seed(42)
        predictor = MLPPredictor(embedding_dim=EMB_DIM, context_length=CONTEXT_LEN, hidden_dim=HIDDEN_DIM, num_steps=1)
        trainer = PredictorTrainer(predictor)
        seq = _make_sequence(20)
        metrics = trainer.evaluate([seq])

        assert "mse" in metrics
        assert "mae" in metrics
        assert "cosine_similarity" in metrics
        assert metrics["mse"] >= 0.0
        assert metrics["mae"] >= 0.0

    def test_train_with_transformer(self):
        """Test training with TransformerPredictor."""
        torch.manual_seed(42)
        predictor = TransformerPredictor(
            embedding_dim=EMB_DIM, context_length=CONTEXT_LEN, num_heads=2, num_layers=1, num_steps=1
        )
        trainer = PredictorTrainer(predictor, lr=1e-3)
        seq = _make_sequence(30)
        history = trainer.train([seq], epochs=10, patience=20)

        assert len(history["train_losses"]) > 0

    def test_train_with_regularizer(self):
        """Test training with SIGReg regularizer."""
        from event_jepa_cube.regularizers import SIGReg

        torch.manual_seed(42)
        predictor = MLPPredictor(embedding_dim=EMB_DIM, context_length=CONTEXT_LEN, hidden_dim=HIDDEN_DIM, num_steps=1)
        reg = SIGReg(num_directions=4)
        trainer = PredictorTrainer(predictor, lr=1e-3, regularizer=reg, reg_weight=0.01)
        seq = _make_sequence(30)
        history = trainer.train([seq], epochs=10, patience=20)

        assert len(history["train_losses"]) > 0

    def test_evaluate_empty_sequence(self):
        """Test evaluate with sequence too short to create windows."""
        torch.manual_seed(42)
        predictor = MLPPredictor(embedding_dim=EMB_DIM, context_length=CONTEXT_LEN, hidden_dim=HIDDEN_DIM, num_steps=1)
        trainer = PredictorTrainer(predictor)
        # Only 3 embeddings with context_length=5, not enough for any window
        short_seq = _make_sequence(3)
        metrics = trainer.evaluate([short_seq])

        assert metrics["mse"] == 0.0
        assert metrics["mae"] == 0.0
        assert metrics["cosine_similarity"] == 0.0
