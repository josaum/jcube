"""JEPA training pipeline with curriculum learning.

Two-phase approach for efficient training on consumer hardware:

  Phase A: Extract hidden states from frozen Qwen (slow, run once, cached)
  Phase B: Train projection + predictor on cached features (instant per step)

The JEPA objective:
  - Context encoder: projection(cached_hidden[context_events]) → context_rep
  - Target encoder: EMA(projection)(cached_hidden[target_events]) → target_rep
  - Predictor: MLP(context_rep) → predicted_target_rep
  - Loss: MSE(predicted, target) + covariance_reg + variance_reg

Uses CooldownSchedule for LR, curriculum learning for difficulty.

Requires: torch, transformers, duckdb, pyarrow
"""

from __future__ import annotations

import copy
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .lora_encoder import (
    HiddenExtractor,
    ProjectionHead,
    RowSerializer,
    detect_device,
)
from .training import CooldownSchedule


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------


class Predictor(nn.Module):
    """MLP: context_embedding → predicted_target_embedding.

    Discarded after training — only the projection head matters.
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# EMA copy of projection head
# ---------------------------------------------------------------------------


class EMAProjection:
    """EMA-updated target projection (no gradient)."""

    def __init__(self, projection: ProjectionHead, tau: float = 0.996) -> None:
        self.projection = copy.deepcopy(projection)
        self.projection.eval()
        self.tau = tau
        for p in self.projection.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, online: ProjectionHead) -> None:
        for p_ema, p_online in zip(
            self.projection.parameters(), online.parameters()
        ):
            p_ema.data.mul_(self.tau).add_(p_online.data, alpha=1 - self.tau)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


# ---------------------------------------------------------------------------
# Regularization (differentiable SIGReg-lite)
# ---------------------------------------------------------------------------


def cov_reg(emb: torch.Tensor) -> torch.Tensor:
    """Off-diagonal covariance penalty → decorrelate dimensions."""
    if emb.shape[0] < 2:
        return torch.tensor(0.0, device=emb.device)
    centered = emb - emb.mean(dim=0, keepdim=True)
    cov = (centered.T @ centered) / (emb.shape[0] - 1)
    off = cov - torch.diag(cov.diag())
    return off.pow(2).sum() / (cov.shape[0] ** 2)


def var_reg(emb: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """Variance hinge loss → prevent collapse."""
    if emb.shape[0] < 2:
        return torch.tensor(0.0, device=emb.device)
    return torch.relu(gamma - emb.std(dim=0)).mean()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_raw_events(
    db_path: str,
    entity_type: str,
    limit_entities: int = 100,
    limit_events: int = 500,
) -> Dict[str, List[Tuple[float, str]]]:
    """Load raw events as (timestamp, serialized_text) per entity."""
    from .materializer import Materializer, _to_epoch

    mat = Materializer(db_path, embedding_dim=64)
    mat.connect()
    tables = mat.scan()

    typed = [tt for tt in tables if tt.entity_type == entity_type]
    if not typed:
        mat.close()
        return {}

    con = mat._con()
    serializer = RowSerializer()

    # Phase 1: richest entities
    count_parts = []
    for tt in typed:
        count_parts.append(
            f'SELECT CAST("{tt.entity_column}" AS VARCHAR) AS eid, COUNT(*) AS cnt '
            f'FROM "{tt.name}" '
            f'WHERE "{tt.entity_column}" IS NOT NULL AND "{tt.timestamp_column}" IS NOT NULL '
            f'GROUP BY "{tt.entity_column}"'
        )
    union = " UNION ALL ".join(count_parts)
    top_sql = (
        f"SELECT eid, SUM(cnt) AS total FROM ({union}) "
        f"GROUP BY eid ORDER BY total DESC LIMIT {limit_entities}"
    )
    rows = con.execute(top_sql).fetchall()
    target_ids = {str(r[0]) for r in rows}

    # Phase 2: load + serialize
    entity_events: Dict[str, List[Tuple[float, str]]] = {}
    id_list = ", ".join(f"'{eid}'" for eid in target_ids)

    for tt in typed:
        where = (
            f'WHERE CAST("{tt.entity_column}" AS VARCHAR) IN ({id_list}) '
            f'AND "{tt.timestamp_column}" IS NOT NULL'
        )
        try:
            arrow = con.execute(
                f'SELECT * FROM "{tt.name}" {where} '
                f'ORDER BY "{tt.timestamp_column}"'
            ).fetch_arrow_table()
        except Exception:
            continue
        if len(arrow) == 0:
            continue

        rows_py = arrow.to_pydict()
        eid_col = rows_py[tt.entity_column]
        ts_col = rows_py[tt.timestamp_column]
        skip = {tt.entity_column, tt.timestamp_column, "source_db"}
        ctx_cols = [c for c in arrow.schema.names if c not in skip]

        for i in range(len(eid_col)):
            eid = str(eid_col[i])
            ts_val = ts_col[i]
            if ts_val is None:
                continue
            epoch = _to_epoch(ts_val)
            if epoch is None:
                continue

            col_vals = {c: rows_py[c][i] for c in ctx_cols}
            text = serializer.serialize_event(tt.name, ts_val, col_vals)
            entity_events.setdefault(eid, []).append((epoch, text))

    mat.close()

    for eid in entity_events:
        entity_events[eid].sort(key=lambda x: x[0])
        if len(entity_events[eid]) > limit_events:
            entity_events[eid] = entity_events[eid][-limit_events:]

    return entity_events


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------


@dataclass
class CurriculumPhase:
    name: str
    ctx_ratio_min: float
    ctx_ratio_max: float
    predict_steps: int
    description: str = ""


CURRICULUM: List[CurriculumPhase] = [
    CurriculumPhase("easy", 0.7, 0.9, 1, "Lots of context, predict 1 step"),
    CurriculumPhase("medium", 0.5, 0.7, 3, "Moderate context, 3 steps"),
    CurriculumPhase("hard", 0.3, 0.5, 5, "Sparse context, 5 steps"),
]


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


@dataclass
class TrainSample:
    entity_id: str
    ctx_indices: List[int]  # indices into the flat event list
    tgt_indices: List[int]


class JEPATrainer:
    """Two-phase JEPA trainer.

    Phase A: Extract hidden states from frozen Qwen (cached to disk).
    Phase B: Train projection + predictor on cached features.

    Example::

        trainer = JEPATrainer("data/my.db")
        metrics = trainer.train("INTERNACAO", limit_entities=50)
        trainer.save("adapters/internacao/")
    """

    def __init__(
        self,
        db_path: str,
        model_name: str = "Qwen/Qwen3.5-0.8B",
        embedding_dim: int = 64,
        predictor_hidden: int = 256,
        ema_tau: float = 0.996,
        device: Optional[str] = None,
        max_length: int = 512,
        cache_dir: str = ".cache/jepa",
    ) -> None:
        self.db_path = db_path
        self.device = device or detect_device()
        self.embedding_dim = embedding_dim
        self.model_name = model_name
        self.max_length = max_length
        self.cache_dir = cache_dir
        self.ema_tau = ema_tau
        self.predictor_hidden = predictor_hidden

        os.makedirs(cache_dir, exist_ok=True)

        self.schedule = CooldownSchedule(
            primary_steps=1000,
            cooldown_steps=200,
            warmup_steps=100,
        )

    def train(
        self,
        entity_type: str,
        *,
        limit_entities: int = 50,
        limit_events: int = 300,
        batch_size: int = 16,
        base_lr: float = 3e-4,
        cov_weight: float = 0.04,
        var_weight: float = 0.04,
        extract_batch_size: int = 4,
        log_every: int = 20,
        seed: int = 42,
    ) -> Dict[str, Any]:
        random.seed(seed)
        torch.manual_seed(seed)

        # ---- Load raw events from DB ----
        print(f"[1/4] Loading events for {entity_type}...")
        t0 = time.time()
        raw = load_raw_events(self.db_path, entity_type, limit_entities, limit_events)
        n_events = sum(len(v) for v in raw.values())
        print(f"  {len(raw)} entities, {n_events:,} events ({time.time()-t0:.1f}s)")

        if n_events < 10:
            return {"error": "not enough events"}

        # ---- Flatten events for batch extraction ----
        flat_texts: List[str] = []
        flat_ts: List[float] = []
        entity_ranges: Dict[str, Tuple[int, int]] = {}  # eid → (start, end)

        serializer = RowSerializer()
        for eid, events in raw.items():
            start = len(flat_texts)
            for epoch, text in events:
                flat_texts.append(text)
                flat_ts.append(epoch)
            entity_ranges[eid] = (start, len(flat_texts))

        print(f"  Flat event count: {len(flat_texts)}")

        # ---- Phase A: Extract hidden states (cached) ----
        cache_path = os.path.join(
            self.cache_dir, f"{entity_type}_{len(flat_texts)}.pt"
        )
        print(f"[2/4] Extracting hidden states (Qwen prefill-only)...")

        extractor = HiddenExtractor(
            model_name=self.model_name,
            device=self.device,
            max_length=self.max_length,
        )
        hidden_dim = extractor.hidden_dim
        all_hidden = extractor.extract_and_cache(
            flat_texts, cache_path, batch_size=extract_batch_size
        )
        # Free the base model from memory
        del extractor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  Hidden states: {all_hidden.shape} (H={hidden_dim})")

        # ---- Phase B: Train projection + predictor ----
        print(f"[3/4] Training projection + predictor...")

        projection = ProjectionHead(hidden_dim, self.embedding_dim).to(self.device)
        ema_proj = EMAProjection(projection, tau=self.ema_tau)
        predictor = Predictor(self.embedding_dim, self.predictor_hidden).to(self.device)

        trainable = list(projection.parameters()) + list(predictor.parameters())
        n_params = sum(p.numel() for p in trainable)
        optimizer = torch.optim.AdamW(trainable, lr=base_lr, weight_decay=0.01)

        total_steps = self.schedule.total_steps
        print(f"  Trainable params: {n_params:,}")
        print(f"  Total steps: {total_steps} ({len(CURRICULUM)} phases)")
        print(f"  Device: {self.device}")
        print()

        losses_log: List[float] = []
        global_step = 0

        for phase_idx, phase in enumerate(CURRICULUM):
            if global_step >= total_steps:
                break

            steps_this_phase = total_steps // len(CURRICULUM)
            phase_end = min(global_step + steps_this_phase, total_steps)

            print(f"  === Phase {phase_idx}: {phase.name} — {phase.description} ===")

            # Build training samples for this phase
            samples = self._build_samples(entity_ranges, len(flat_texts), phase)
            if not samples:
                print("    No samples, skip")
                continue
            random.shuffle(samples)

            phase_loss = 0.0
            phase_steps = 0
            t_phase = time.time()
            sidx = 0

            while global_step < phase_end:
                # Cycle samples
                if sidx + batch_size > len(samples):
                    samples = self._build_samples(entity_ranges, len(flat_texts), phase)
                    random.shuffle(samples)
                    sidx = 0

                batch = samples[sidx : sidx + batch_size]
                sidx += batch_size

                # LR
                lr_mult = self.schedule.get_lr_multiplier(global_step)
                for pg in optimizer.param_groups:
                    pg["lr"] = base_lr * lr_mult

                # Gather context hidden states → aggregate → project
                ctx_embs = self._aggregate_and_project(
                    batch, "ctx_indices", all_hidden, projection
                )

                # Gather target hidden states → aggregate → EMA project
                with torch.no_grad():
                    tgt_embs = self._aggregate_and_project(
                        batch, "tgt_indices", all_hidden, ema_proj.projection
                    )

                # Predict
                pred_embs = predictor(ctx_embs)

                # Loss: JEPA MSE + regularization
                loss = nn.functional.mse_loss(pred_embs, tgt_embs)

                if cov_weight > 0 and ctx_embs.shape[0] > 1:
                    loss = loss + cov_weight * cov_reg(ctx_embs)
                if var_weight > 0 and ctx_embs.shape[0] > 1:
                    loss = loss + var_weight * var_reg(ctx_embs)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()

                # EMA
                ema_proj.update(projection)

                loss_val = loss.item()
                phase_loss += loss_val
                phase_steps += 1
                global_step += 1
                losses_log.append(loss_val)

                if phase_steps % log_every == 0:
                    avg = phase_loss / phase_steps
                    lr = optimizer.param_groups[0]["lr"]
                    elapsed = time.time() - t_phase
                    steps_s = phase_steps / max(elapsed, 0.01)
                    print(
                        f"    step {global_step:5d} | "
                        f"loss={avg:.6f} | lr={lr:.2e} | "
                        f"{steps_s:.0f} steps/s"
                    )

            if phase_steps > 0:
                print(f"    Phase avg loss: {phase_loss / phase_steps:.6f}")
            print()

        # ---- Results ----
        print(f"[4/4] Done. {global_step} steps, final loss={losses_log[-1]:.6f}")

        # Store for save/inference
        self._projection = projection
        self._predictor = predictor
        self._ema_proj = ema_proj
        self._hidden_dim = hidden_dim
        self._all_hidden = all_hidden
        self._entity_ranges = entity_ranges
        self._flat_ts = flat_ts

        return {
            "entity_type": entity_type,
            "total_steps": global_step,
            "final_loss": losses_log[-1] if losses_log else None,
            "losses": losses_log,
            "trainable_params": n_params,
            "hidden_dim": hidden_dim,
            "entities": len(raw),
            "events": n_events,
        }

    def _build_samples(
        self,
        entity_ranges: Dict[str, Tuple[int, int]],
        total_events: int,
        phase: CurriculumPhase,
    ) -> List[TrainSample]:
        """Build context/target pairs using temporal masking."""
        samples = []
        for eid, (start, end) in entity_ranges.items():
            n = end - start
            if n < 4:
                continue

            ratio = random.uniform(phase.ctx_ratio_min, phase.ctx_ratio_max)
            split = max(2, int(n * ratio))
            tgt_end = min(split + phase.predict_steps, n)

            ctx_idx = list(range(start, start + split))
            tgt_idx = list(range(start + split, start + tgt_end))
            if not tgt_idx:
                continue

            samples.append(TrainSample(
                entity_id=eid,
                ctx_indices=ctx_idx,
                tgt_indices=tgt_idx,
            ))
        return samples

    def _aggregate_and_project(
        self,
        batch: List[TrainSample],
        idx_field: str,
        all_hidden: torch.Tensor,
        projection: nn.Module,
    ) -> torch.Tensor:
        """Gather hidden states by indices, mean-pool per sample, project."""
        pooled = []
        for sample in batch:
            indices = getattr(sample, idx_field)
            # Gather and mean-pool hidden states for this sample's events
            h = all_hidden[indices]  # (K, H)
            pooled.append(h.mean(dim=0))  # (H,)

        stacked = torch.stack(pooled).to(self.device)  # (B, H)
        return projection(stacked)  # (B, D)

    def encode_entity(self, entity_id: str) -> Optional[torch.Tensor]:
        """Encode an entity using the trained projection on cached hiddens."""
        if not hasattr(self, "_projection"):
            return None
        rng = self._entity_ranges.get(entity_id)
        if rng is None:
            return None

        start, end = rng
        h = self._all_hidden[start:end].mean(dim=0, keepdim=True).to(self.device)
        with torch.no_grad():
            return self._projection(h).cpu()

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self._projection.state_dict(), os.path.join(path, "projection.pt"))
        torch.save(self._predictor.state_dict(), os.path.join(path, "predictor.pt"))
        torch.save(
            {"hidden_dim": self._hidden_dim, "embedding_dim": self.embedding_dim},
            os.path.join(path, "config.pt"),
        )
        print(f"Saved to {path}")
