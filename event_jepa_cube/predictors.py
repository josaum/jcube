"""Learnable neural predictors for EventJEPA sequences.

Provides MLP and Transformer-based predictors that replace trend extrapolation
with trained models. Supports optional SIGReg regularization to prevent
representation collapse.

Requires PyTorch. Install with: pip install event-jepa-cube[torch]
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


class MLPPredictor(nn.Module):
    """MLP-based sequence predictor for EventJEPA.

    Takes a fixed-size context window of recent embeddings and predicts
    the next embedding(s). Can be trained with SIGReg regularization
    to prevent representation collapse.

    Architecture:
        [context_dim] -> Linear -> GELU -> Linear -> GELU -> Linear -> [embedding_dim * num_steps]
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        context_length: int = 5,
        hidden_dim: int = 256,
        num_steps: int = 1,
    ) -> None:
        """Initialize MLPPredictor.

        Args:
            embedding_dim: Dimension of each embedding.
            context_length: Number of recent embeddings to use as input.
            hidden_dim: Hidden layer dimension.
            num_steps: Number of future steps to predict.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.context_length = context_length
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps

        input_dim = context_length * embedding_dim
        output_dim = num_steps * embedding_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, context: Tensor) -> Tensor:
        """Forward pass.

        Args:
            context: Tensor of shape (batch, context_length, embedding_dim)

        Returns:
            Predictions of shape (batch, num_steps, embedding_dim)
        """
        batch_size = context.shape[0]
        flat = context.reshape(batch_size, -1)
        out = self.net(flat)
        return out.reshape(batch_size, self.num_steps, self.embedding_dim)

    @torch.no_grad()
    def predict_from_sequence(
        self,
        sequence_embeddings: list[list[float]],
        num_steps: int | None = None,
    ) -> list[list[float]]:
        """Convenience method: predict from raw Python lists.

        Takes the last context_length embeddings from the sequence,
        runs inference, returns Python lists.

        Args:
            sequence_embeddings: List of embedding vectors.
            num_steps: Override num_steps (uses self.num_steps if None).

        Returns:
            List of predicted embedding vectors.
        """
        self.eval()
        steps = num_steps if num_steps is not None else self.num_steps

        # Pad if sequence is shorter than context_length
        embs = list(sequence_embeddings)
        while len(embs) < self.context_length:
            embs.insert(0, [0.0] * self.embedding_dim)

        # Take last context_length embeddings
        context_list = embs[-self.context_length :]
        context_tensor = torch.tensor([context_list], dtype=torch.float32)

        # If num_steps differs from self.num_steps, we need to handle it
        if steps != self.num_steps:
            # Run multiple passes if requesting more steps, or slice if fewer
            predictions: list[list[float]] = []
            current_context = context_list[:]
            remaining = steps
            while remaining > 0:
                ctx = torch.tensor([current_context[-self.context_length :]], dtype=torch.float32)
                out = self.forward(ctx)
                batch_preds = out[0].tolist()
                take = min(remaining, self.num_steps)
                predictions.extend(batch_preds[:take])
                remaining -= take
                # Shift context forward
                for pred in batch_preds[:take]:
                    current_context.append(pred)
            return predictions

        out = self.forward(context_tensor)
        return out[0].tolist()


class TransformerPredictor(nn.Module):
    """Lightweight Transformer predictor for EventJEPA sequences.

    Uses self-attention over a context window with learned positional
    encodings and a causal prediction head.

    Architecture:
        Embedding projection -> Positional encoding -> N transformer layers -> Linear head
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        context_length: int = 16,
        num_heads: int = 4,
        num_layers: int = 2,
        num_steps: int = 1,
    ) -> None:
        """Initialize TransformerPredictor.

        Args:
            embedding_dim: Dimension of each embedding.
            context_length: Maximum context window.
            num_heads: Attention heads.
            num_layers: Transformer encoder layers.
            num_steps: Future steps to predict.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.context_length = context_length
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_steps = num_steps

        # Learned positional embeddings
        self.pos_embedding = nn.Embedding(context_length, embedding_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Linear prediction head
        self.head = nn.Linear(embedding_dim, num_steps * embedding_dim)

    def forward(self, context: Tensor) -> Tensor:
        """Forward pass.

        Args:
            context: (batch, seq_len, embedding_dim) where seq_len <= context_length

        Returns:
            (batch, num_steps, embedding_dim)
        """
        batch_size, seq_len, _ = context.shape

        # Add positional embeddings
        positions = torch.arange(seq_len, device=context.device)
        pos_emb = self.pos_embedding(positions)
        x = context + pos_emb.unsqueeze(0)

        # Run through transformer
        x = self.transformer(x)

        # Use last token's representation for prediction
        last_token = x[:, -1, :]  # (batch, embedding_dim)
        out = self.head(last_token)  # (batch, num_steps * embedding_dim)

        return out.reshape(batch_size, self.num_steps, self.embedding_dim)

    @torch.no_grad()
    def predict_from_sequence(
        self,
        sequence_embeddings: list[list[float]],
        num_steps: int | None = None,
    ) -> list[list[float]]:
        """Convenience method: predict from raw Python lists.

        Takes the last context_length embeddings from the sequence,
        runs inference, returns Python lists.

        Args:
            sequence_embeddings: List of embedding vectors.
            num_steps: Override num_steps (uses self.num_steps if None).

        Returns:
            List of predicted embedding vectors.
        """
        self.eval()
        steps = num_steps if num_steps is not None else self.num_steps

        # Pad if sequence is shorter than context_length
        embs = list(sequence_embeddings)
        while len(embs) < self.context_length:
            embs.insert(0, [0.0] * self.embedding_dim)

        # Take last context_length embeddings
        context_list = embs[-self.context_length :]
        context_tensor = torch.tensor([context_list], dtype=torch.float32)

        if steps != self.num_steps:
            predictions: list[list[float]] = []
            current_context = context_list[:]
            remaining = steps
            while remaining > 0:
                ctx = current_context[-self.context_length :]
                ctx_tensor = torch.tensor([ctx], dtype=torch.float32)
                out = self.forward(ctx_tensor)
                batch_preds = out[0].tolist()
                take = min(remaining, self.num_steps)
                predictions.extend(batch_preds[:take])
                remaining -= take
                for pred in batch_preds[:take]:
                    current_context.append(pred)
            return predictions

        out = self.forward(context_tensor)
        return out[0].tolist()


class PredictorTrainer:
    """Training loop for sequence predictors with optional SIGReg.

    Handles:
    - Sliding window dataset creation from EventSequences
    - MSE prediction loss + optional regularization loss
    - Training loop with early stopping
    """

    def __init__(
        self,
        predictor: nn.Module,
        lr: float = 1e-3,
        regularizer: Any = None,
        reg_weight: float = 0.05,
    ) -> None:
        """Initialize PredictorTrainer.

        Args:
            predictor: MLPPredictor or TransformerPredictor.
            lr: Learning rate.
            regularizer: Optional SIGReg/WeakSIGReg/RDMReg instance.
            reg_weight: Regularization weight.
        """
        self.predictor = predictor
        self.lr = lr
        self.regularizer = regularizer
        self.reg_weight = reg_weight
        self.optimizer = torch.optim.Adam(predictor.parameters(), lr=lr)

    def _get_context_length(self) -> int:
        """Get context_length from predictor."""
        return getattr(self.predictor, "context_length", 5)

    def _get_num_steps(self) -> int:
        """Get num_steps from predictor."""
        return getattr(self.predictor, "num_steps", 1)

    def prepare_dataset(
        self,
        sequences: list[Any],
        context_length: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Create sliding window training data from EventSequences.

        Args:
            sequences: List of EventSequence objects or list of list[list[float]] embeddings.
            context_length: Override predictor's context_length.

        Returns:
            Tuple of (contexts_tensor, targets_tensor)
        """
        ctx_len = context_length if context_length is not None else self._get_context_length()
        num_steps = self._get_num_steps()

        all_contexts: list[list[list[float]]] = []
        all_targets: list[list[list[float]]] = []

        for seq in sequences:
            # Extract embeddings from EventSequence or raw list
            if hasattr(seq, "embeddings"):
                embeddings = seq.embeddings
            else:
                embeddings = seq

            # Create sliding windows
            total_needed = ctx_len + num_steps
            for i in range(len(embeddings) - total_needed + 1):
                context_window = embeddings[i : i + ctx_len]
                target_window = embeddings[i + ctx_len : i + ctx_len + num_steps]
                all_contexts.append(context_window)
                all_targets.append(target_window)

        if not all_contexts:
            return torch.zeros(0, ctx_len, 1), torch.zeros(0, num_steps, 1)

        contexts_tensor = torch.tensor(all_contexts, dtype=torch.float32)
        targets_tensor = torch.tensor(all_targets, dtype=torch.float32)

        return contexts_tensor, targets_tensor

    def train(
        self,
        sequences: list[Any],
        epochs: int = 100,
        patience: int = 10,
        val_split: float = 0.1,
    ) -> dict[str, Any]:
        """Train the predictor on sequences.

        Args:
            sequences: Training data (EventSequence objects or raw embeddings).
            epochs: Maximum training epochs.
            patience: Early stopping patience.
            val_split: Validation fraction.

        Returns:
            Training history dict with train_losses, val_losses, best_epoch.
        """
        contexts, targets = self.prepare_dataset(sequences)

        if contexts.shape[0] == 0:
            return {"train_losses": [], "val_losses": [], "best_epoch": 0}

        # Split into train/val
        n = contexts.shape[0]
        n_val = max(1, int(n * val_split))
        n_train = n - n_val

        # Shuffle
        perm = torch.randperm(n)
        contexts = contexts[perm]
        targets = targets[perm]

        train_ctx, val_ctx = contexts[:n_train], contexts[n_train:]
        train_tgt, val_tgt = targets[:n_train], targets[n_train:]

        mse_loss = nn.MSELoss()

        train_losses: list[float] = []
        val_losses: list[float] = []
        best_val_loss = float("inf")
        best_epoch = 0
        epochs_without_improvement = 0

        self.predictor.train()

        for epoch in range(epochs):
            # Training step
            self.optimizer.zero_grad()
            pred = self.predictor(train_ctx)
            loss = mse_loss(pred, train_tgt)

            # Optional regularization
            if self.regularizer is not None:
                # Flatten predictions for regularizer: (N * num_steps, embedding_dim)
                flat_pred = pred.reshape(-1, pred.shape[-1])
                reg_loss = self.regularizer.compute_loss(flat_pred)
                loss = loss + self.reg_weight * reg_loss

            loss.backward()
            self.optimizer.step()
            train_losses.append(loss.item())

            # Validation step
            self.predictor.eval()
            with torch.no_grad():
                val_pred = self.predictor(val_ctx)
                val_loss = mse_loss(val_pred, val_tgt)
            val_losses.append(val_loss.item())
            self.predictor.train()

            # Early stopping
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                break

        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_epoch": best_epoch,
        }

    def evaluate(self, sequences: list[Any]) -> dict[str, float]:
        """Evaluate predictor on sequences.

        Returns dict with mse, mae, cosine_similarity metrics.
        """
        contexts, targets = self.prepare_dataset(sequences)

        if contexts.shape[0] == 0:
            return {"mse": 0.0, "mae": 0.0, "cosine_similarity": 0.0}

        self.predictor.eval()
        with torch.no_grad():
            predictions = self.predictor(contexts)

            # MSE
            mse = nn.functional.mse_loss(predictions, targets).item()

            # MAE
            mae = nn.functional.l1_loss(predictions, targets).item()

            # Cosine similarity (average over all prediction vectors)
            pred_flat = predictions.reshape(-1, predictions.shape[-1])
            tgt_flat = targets.reshape(-1, targets.shape[-1])
            cos_sim = nn.functional.cosine_similarity(pred_flat, tgt_flat, dim=1).mean().item()

        return {
            "mse": mse,
            "mae": mae,
            "cosine_similarity": cos_sim,
        }
