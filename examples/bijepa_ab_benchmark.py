"""A/B benchmark: unidirectional PredictorTrainer vs BiJEPATrainer.

Same forward architecture, same data, same optimizer settings; the only
difference is the training objective (plain MSE vs BiJEPA's bidirectional +
cycle-consistency + NormReg objective). Two evaluations:

1. cross-system: held-out sequences from NEW random linear dynamical systems
   (out-of-distribution generalization)
2. same-system: future windows of the TRAINING systems
   (in-distribution forecasting)

Results are recorded in docs/SOTA_2026.md. Requires torch:
    pip install -e ".[torch]"
    python examples/bijepa_ab_benchmark.py
"""

from __future__ import annotations

import math

import torch

from event_jepa_cube.bijepa import BiJEPATrainer
from event_jepa_cube.predictors import MLPPredictor, PredictorTrainer

DIM = 16
CTX = 5
STEPS = 2
HID = 64
EPOCHS = 150
LR = 1e-2
N_SEQ = 24
NOISE = 0.05
SEEDS = 5
CYCLE_WEIGHT = 0.5
NORM_WEIGHT = 0.01


def _random_system(g: torch.Generator) -> torch.Tensor:
    """Random linear dynamics with spectral radius ~0.95 (stable, non-dying)."""
    a = torch.randn(DIM, DIM, generator=g) * (0.9 / math.sqrt(DIM))
    return a / max(1e-6, torch.linalg.eigvals(a).abs().max().item()) * 0.95


def _rollout(a: torch.Tensor, length: int, g: torch.Generator) -> list[list[float]]:
    x = torch.randn(DIM, generator=g)
    embs = []
    for _ in range(length):
        x = a @ x + NOISE * torch.randn(DIM, generator=g)
        embs.append([float(v) for v in x])
    return embs


def make_cross_system(n_seq: int, length: int, seed: int) -> list[list[list[float]]]:
    g = torch.Generator().manual_seed(seed)
    return [_rollout(_random_system(g), length, g) for _ in range(n_seq)]


def make_same_system_split(
    n_seq: int, train_len: int, test_len: int, seed: int
) -> tuple[list[list[list[float]]], list[list[list[float]]]]:
    g = torch.Generator().manual_seed(seed)
    train, test = [], []
    for _ in range(n_seq):
        embs = _rollout(_random_system(g), train_len + test_len, g)
        train.append(embs[:train_len])
        test.append(embs[train_len:])
    return train, test


def train_pair(train_seqs: list[list[list[float]]], seed: int) -> tuple[MLPPredictor, MLPPredictor]:
    torch.manual_seed(seed)
    uni_pred = MLPPredictor(embedding_dim=DIM, context_length=CTX, hidden_dim=HID, num_steps=STEPS)
    PredictorTrainer(uni_pred, lr=LR).train(train_seqs, epochs=EPOCHS, patience=EPOCHS)

    torch.manual_seed(seed)
    bi_fwd = MLPPredictor(embedding_dim=DIM, context_length=CTX, hidden_dim=HID, num_steps=STEPS)
    BiJEPATrainer(bi_fwd, lr=LR, cycle_weight=CYCLE_WEIGHT, norm_weight=NORM_WEIGHT).train(
        train_seqs, epochs=EPOCHS, patience=EPOCHS
    )
    return uni_pred, bi_fwd


def summarize(rows: list[dict[str, float]], key: str) -> tuple[float, float]:
    vals = [r[key] for r in rows]
    mean = sum(vals) / len(vals)
    std = (sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)) ** 0.5
    return mean, std


def run(name: str, datasets: list[tuple[list, list]]) -> None:
    print(f"\n== {name} ==")
    uni_all, bi_all, wins = [], [], 0
    for seed, (train_seqs, test_seqs) in enumerate(datasets):
        uni_pred, bi_fwd = train_pair(train_seqs, seed)
        m_uni = PredictorTrainer(uni_pred).evaluate(test_seqs)
        m_bi = PredictorTrainer(bi_fwd).evaluate(test_seqs)
        uni_all.append(m_uni)
        bi_all.append(m_bi)
        wins += m_bi["mse"] < m_uni["mse"]
        print(
            f"seed {seed}: uni mse={m_uni['mse']:.4f} cos={m_uni['cosine_similarity']:.3f} | "
            f"bi mse={m_bi['mse']:.4f} cos={m_bi['cosine_similarity']:.3f}"
        )
    for label, rows in (("uni", uni_all), ("bi", bi_all)):
        mse_m, mse_s = summarize(rows, "mse")
        cos_m, cos_s = summarize(rows, "cosine_similarity")
        print(f"{label}: held-out MSE {mse_m:.4f} ± {mse_s:.4f} | cosine {cos_m:.3f} ± {cos_s:.3f}")
    print(f"BiJEPA lower held-out MSE in {wins}/{len(datasets)} seeds")


if __name__ == "__main__":
    run(
        "cross-system (out-of-distribution)",
        [(make_cross_system(N_SEQ, 40, 1000 + s), make_cross_system(8, 40, 9000 + s)) for s in range(SEEDS)],
    )
    run(
        "same-system future windows (in-distribution)",
        [make_same_system_split(N_SEQ, 40, 15, 2000 + s) for s in range(SEEDS)],
    )
