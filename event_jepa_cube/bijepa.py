"""Bidirectional JEPA training with cycle consistency (BiJEPA).

Standard JEPA prediction is uni-directional: a predictor maps context
representations to target representations. BiJEPA (arXiv:2603.00049) adds the
inverse mapping (target -> context) and enforces cycle-consistent
predictability between the two segments, capturing the informative signal in
the inverse relationship that uni-directional prediction discards. Symmetric
prediction removes the need for a stop-gradient/EMA teacher but admits
representation explosion, which is closed by norm regularization (NormReg).

Loss:
    L = L_fwd + L_bwd
        + cycle_weight * (L_cycle_ctx + L_cycle_tgt)
        + norm_weight * NormReg(predictions)
        + reg_weight * regularizer(predictions)        # optional SIGReg family

where
    L_fwd       = MSE(f(ctx), tgt)          f: forward predictor
    L_bwd       = MSE(g(tgt), ctx)          g: backward predictor
    L_cycle_ctx = MSE(g(f(ctx)), ctx)       round trip through the future
    L_cycle_tgt = MSE(f(g(tgt)), tgt)       round trip through the past

Requires PyTorch. Install with: pip install event-jepa-cube[torch]
"""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn as nn
from torch import Tensor

from .predictors import MLPPredictor, make_sliding_windows
from .regularizers import NormReg


def _predictor_dims(predictor: nn.Module) -> tuple[int, int, int]:
    """Return (embedding_dim, context_length, num_steps) from a predictor."""
    return (
        cast(int, predictor.embedding_dim),
        cast(int, predictor.context_length),
        cast(int, predictor.num_steps),
    )


class BiJEPATrainer:
    """Training loop for bidirectional cycle-consistent JEPA prediction.

    Trains a forward predictor (context -> target) and a backward predictor
    (target -> context) jointly with cycle-consistency losses and norm
    regularization. An optional distributional regularizer (SIGReg,
    WeakSIGReg, RDMReg) can be applied to the forward predictions, mirroring
    PredictorTrainer.

    Args:
        forward_predictor: MLPPredictor or TransformerPredictor mapping
            (batch, context_length, dim) -> (batch, num_steps, dim).
        backward_predictor: Predictor mapping (batch, num_steps, dim) ->
            (batch, context_length, dim). If None, an MLPPredictor mirroring
            the forward predictor's dimensions is created.
        lr: Learning rate for both predictors.
        cycle_weight: Weight of the two cycle-consistency terms.
        norm_weight: Weight of the NormReg term. Set to 0.0 to disable.
        regularizer: Optional SIGReg/WeakSIGReg/RDMReg instance.
        reg_weight: Weight of the optional distributional regularizer.
    """

    def __init__(
        self,
        forward_predictor: nn.Module,
        backward_predictor: nn.Module | None = None,
        lr: float = 1e-3,
        cycle_weight: float = 1.0,
        norm_weight: float = 0.01,
        regularizer: Any = None,
        reg_weight: float = 0.05,
    ) -> None:
        self.forward_predictor = forward_predictor
        if backward_predictor is None:
            fwd_dim, fwd_ctx, fwd_steps = _predictor_dims(forward_predictor)
            backward_predictor = MLPPredictor(
                embedding_dim=fwd_dim,
                context_length=fwd_steps,
                hidden_dim=getattr(forward_predictor, "hidden_dim", 256),
                num_steps=fwd_ctx,
            )
        self.backward_predictor = backward_predictor
        self._validate_shapes()

        self.lr = lr
        self.cycle_weight = cycle_weight
        self.norm_weight = norm_weight
        self.norm_reg = NormReg() if norm_weight > 0.0 else None
        self.regularizer = regularizer
        self.reg_weight = reg_weight
        self.optimizer = torch.optim.Adam(
            list(forward_predictor.parameters()) + list(backward_predictor.parameters()),
            lr=lr,
        )

    def _validate_shapes(self) -> None:
        fwd, bwd = self.forward_predictor, self.backward_predictor
        if fwd.embedding_dim != bwd.embedding_dim:
            raise ValueError(f"embedding_dim mismatch: forward={fwd.embedding_dim}, backward={bwd.embedding_dim}")
        if fwd.num_steps != bwd.context_length or fwd.context_length != bwd.num_steps:
            raise ValueError(
                "backward predictor must mirror the forward predictor: expected "
                f"context_length={fwd.num_steps}, num_steps={fwd.context_length}, got "
                f"context_length={bwd.context_length}, num_steps={bwd.num_steps}"
            )

    def compute_losses(self, contexts: Tensor, targets: Tensor) -> dict[str, Tensor]:
        """Compute all BiJEPA loss components for a batch.

        Args:
            contexts: (batch, context_length, dim)
            targets: (batch, num_steps, dim)

        Returns:
            Dict with total, forward, backward, cycle, norm, and reg loss tensors.
        """
        mse = nn.functional.mse_loss

        pred_tgt = self.forward_predictor(contexts)
        pred_ctx = self.backward_predictor(targets)
        loss_fwd = mse(pred_tgt, targets)
        loss_bwd = mse(pred_ctx, contexts)

        cycle_ctx = self.backward_predictor(pred_tgt)
        cycle_tgt = self.forward_predictor(pred_ctx)
        loss_cycle = mse(cycle_ctx, contexts) + mse(cycle_tgt, targets)

        total = loss_fwd + loss_bwd + self.cycle_weight * loss_cycle

        flat_pred = pred_tgt.reshape(-1, pred_tgt.shape[-1])
        if self.norm_reg is not None:
            loss_norm = self.norm_reg.compute_loss(flat_pred)
            total = total + self.norm_weight * loss_norm
        else:
            loss_norm = torch.tensor(0.0, device=contexts.device)

        if self.regularizer is not None:
            loss_reg = self.regularizer.compute_loss(flat_pred)
            total = total + self.reg_weight * loss_reg
        else:
            loss_reg = torch.tensor(0.0, device=contexts.device)

        return {
            "total": total,
            "forward": loss_fwd,
            "backward": loss_bwd,
            "cycle": loss_cycle,
            "norm": loss_norm,
            "reg": loss_reg,
        }

    def prepare_dataset(self, sequences: list[Any]) -> tuple[Tensor, Tensor]:
        """Create sliding (context, target) windows sized for the forward predictor."""
        _, ctx_len, num_steps = _predictor_dims(self.forward_predictor)
        return make_sliding_windows(sequences, ctx_len, num_steps)

    def train(
        self,
        sequences: list[Any],
        epochs: int = 100,
        patience: int = 10,
        val_split: float = 0.1,
    ) -> dict[str, Any]:
        """Train both predictors on sequences.

        Args:
            sequences: Training data (EventSequence objects or raw embeddings).
            epochs: Maximum training epochs.
            patience: Early stopping patience on validation total loss.
            val_split: Validation fraction.

        Returns:
            Training history dict with train_losses, val_losses, best_epoch,
            and per-component histories (forward_losses, backward_losses,
            cycle_losses, norm_losses).
        """
        contexts, targets = self.prepare_dataset(sequences)

        empty_history: dict[str, Any] = {
            "train_losses": [],
            "val_losses": [],
            "forward_losses": [],
            "backward_losses": [],
            "cycle_losses": [],
            "norm_losses": [],
            "best_epoch": 0,
        }
        if contexts.shape[0] == 0:
            return empty_history

        n = contexts.shape[0]
        n_val = max(1, int(n * val_split))
        n_train = n - n_val

        perm = torch.randperm(n)
        contexts = contexts[perm]
        targets = targets[perm]

        train_ctx, val_ctx = contexts[:n_train], contexts[n_train:]
        train_tgt, val_tgt = targets[:n_train], targets[n_train:]

        history = empty_history
        best_val_loss = float("inf")
        epochs_without_improvement = 0

        self.forward_predictor.train()
        self.backward_predictor.train()

        for epoch in range(epochs):
            self.optimizer.zero_grad()
            losses = self.compute_losses(train_ctx, train_tgt)
            losses["total"].backward()
            self.optimizer.step()

            history["train_losses"].append(losses["total"].item())
            history["forward_losses"].append(losses["forward"].item())
            history["backward_losses"].append(losses["backward"].item())
            history["cycle_losses"].append(losses["cycle"].item())
            history["norm_losses"].append(losses["norm"].item())

            self.forward_predictor.eval()
            self.backward_predictor.eval()
            with torch.no_grad():
                val_losses = self.compute_losses(val_ctx, val_tgt)
            history["val_losses"].append(val_losses["total"].item())
            self.forward_predictor.train()
            self.backward_predictor.train()

            if val_losses["total"].item() < best_val_loss:
                best_val_loss = val_losses["total"].item()
                history["best_epoch"] = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                break

        return history

    def evaluate(self, sequences: list[Any]) -> dict[str, float]:
        """Evaluate both directions and cycle consistency on sequences.

        Returns dict with forward_mse, backward_mse, cycle_mse, and
        forward cosine_similarity metrics.
        """
        contexts, targets = self.prepare_dataset(sequences)

        if contexts.shape[0] == 0:
            return {"forward_mse": 0.0, "backward_mse": 0.0, "cycle_mse": 0.0, "cosine_similarity": 0.0}

        self.forward_predictor.eval()
        self.backward_predictor.eval()
        with torch.no_grad():
            losses = self.compute_losses(contexts, targets)
            pred_tgt = self.forward_predictor(contexts)
            pred_flat = pred_tgt.reshape(-1, pred_tgt.shape[-1])
            tgt_flat = targets.reshape(-1, targets.shape[-1])
            cos_sim = nn.functional.cosine_similarity(pred_flat, tgt_flat, dim=1).mean().item()

        return {
            "forward_mse": losses["forward"].item(),
            "backward_mse": losses["backward"].item(),
            "cycle_mse": (losses["cycle"].item() / 2.0),
            "cosine_similarity": cos_sim,
        }
