"""Runnable sample experiments for the five open research gaps.

Run with: python -m examples.representation_gap_experiments
These are small synthetic checks, not production-performance claims.
"""

from __future__ import annotations

import json
import math
import uuid

import torch

from event_jepa_cube.evaluation import SharedGraphStore, temporal_probe_split
from event_jepa_cube.masking import relation_aware_mask
from event_jepa_cube.predictors import MLPPredictor, PredictorTrainer, make_sliding_windows
from event_jepa_cube.regularizers import (
    CorrelatedSIGReg,
    SACReg,
    SubSIGReg,
    TemporalDifferenceReg,
    WeakSIGReg,
    effective_rank,
)
from event_jepa_cube.sequence import EventSequence

DIM = 4
CTX = 4
EPOCHS = 35


def _sequences(count: int = 12, length: int = 24, dim: int = DIM) -> list[list[list[float]]]:
    generator = torch.Generator().manual_seed(42)
    sequences = []
    for _ in range(count):
        matrix = torch.randn(dim, dim, generator=generator) * (0.7 / math.sqrt(dim))
        state = torch.randn(dim, generator=generator)
        sequence = []
        for _ in range(length):
            state = matrix @ state + 0.03 * torch.randn(dim, generator=generator)
            sequence.append(state.tolist())
        sequences.append(sequence)
    return sequences


def run_leakage_and_shared_graph() -> dict[str, object]:
    early = EventSequence([[0.0, 0.0], [1.0, 1.0], [999.0, 999.0]], [1.0, 2.0, 3.0])
    late = EventSequence([[0.0, 0.0], [2.0, 2.0], [999.0, 999.0]], [1.0, 3.0, 4.0])
    train, test = temporal_probe_split([(early, 0, 2.5), (late, 1, 3.5)], train_fraction=0.5)
    leaked_features = sum(embedding[0] == 999.0 for sequence, _ in train + test for embedding in sequence.embeddings)

    name = f"sample-{uuid.uuid4().hex}"
    owner = SharedGraphStore(name, (2, 2), create=True)
    owner.edges[:] = [[0, 1], [1, 0]]
    owner.edges.flush()
    worker = SharedGraphStore(name, (2, 2))
    worker.edges[0, 0] = 7
    worker.edges.flush()
    shared_value = int(owner.edges[0, 0])
    worker.close()
    owner.unlink()
    return {
        "probe_train": len(train),
        "probe_test": len(test),
        "leaked_features": leaked_features,
        "shared_value": shared_value,
    }


def run_regularizer_ab() -> list[dict[str, float | str]]:
    train = _sequences()
    test = _sequences(count=4)
    contexts, _ = make_sliding_windows(test, CTX, 1)
    rows = []
    for name, regularizer in (
        ("WeakSIGReg", WeakSIGReg(sketch_dim=DIM)),
        ("SACReg", SACReg(eps=1e-4)),
        ("SubSIGReg", SubSIGReg(num_subspaces=2, subspace_dim=2, num_directions=8)),
    ):
        torch.manual_seed(7)
        predictor = MLPPredictor(embedding_dim=DIM, context_length=CTX, hidden_dim=16, num_steps=1)
        PredictorTrainer(predictor, lr=0.01, regularizer=regularizer, reg_weight=0.01).train(
            train, epochs=EPOCHS, patience=EPOCHS
        )
        metrics = PredictorTrainer(predictor).evaluate(test)
        with torch.no_grad():
            predictions = predictor(contexts)
            rank = effective_rank(predictions.reshape(-1, DIM)).item()
        rows.append({"regularizer": name, "heldout_mse": metrics["mse"], "effective_rank": rank})
    return rows


def run_temporal_ab() -> list[dict[str, float | int]]:
    sequences = _sequences()
    contexts, targets = make_sliding_windows(sequences, CTX, 2)
    current = contexts[:, -1:]
    rows = []
    for delta_weight in (0.0, 1.0):
        torch.manual_seed(11)
        predictor = MLPPredictor(embedding_dim=DIM, context_length=CTX, hidden_dim=16, num_steps=2)
        optimizer = torch.optim.Adam(predictor.parameters(), lr=0.01)
        objective = TemporalDifferenceReg(delta_weight=delta_weight)
        for _ in range(EPOCHS):
            optimizer.zero_grad()
            prediction = predictor(contexts)
            loss = objective.compute_loss(current, prediction, targets)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            prediction = predictor(contexts)
            delta_prediction = prediction - current
            delta_target = targets - current
            delta_mse = torch.nn.functional.mse_loss(delta_prediction, delta_target).item()
        rows.append(
            {
                "delta_weight": delta_weight,
                "delta_mse": delta_mse,
                "effective_rank": effective_rank(prediction.reshape(-1, DIM)).item(),
            }
        )
    return rows


def run_multimodal_and_masking() -> dict[str, object]:
    correlated = CorrelatedSIGReg(rho=0.5)
    target = torch.full((4, 4), 0.5).fill_diagonal_(1.0)
    factor = torch.linalg.cholesky(target).t()
    centered = factor - factor.mean(dim=0, keepdim=True)
    independent_loss = correlated.compute_loss(torch.eye(4)).item()
    correlated_loss = correlated.compute_loss(centered).item()
    edges = [(0, index) for index in range(1, 12)]
    mask = relation_aware_mask(12, edges, 1 / 3, seed=42)
    return {
        "independent_loss": independent_loss,
        "correlated_loss": correlated_loss,
        "hub_masked": bool(mask[0]),
        "masked_nodes": sum(mask),
    }


def main() -> None:
    summary = {
        "leakage_and_shared_graph": run_leakage_and_shared_graph(),
        "regularizer_ab": run_regularizer_ab(),
        "temporal_difference_ab": run_temporal_ab(),
        "multimodal_and_masking": run_multimodal_and_masking(),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
