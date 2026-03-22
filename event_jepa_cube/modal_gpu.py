"""Modal GPU backend for JEPA training.

Offloads Qwen prefill + JEPA projection training to cloud A100-80GB.
Local machine loads DB data, serializes events, sends text to GPU.

Usage:
    modal run event_jepa_cube/modal_gpu.py --entity-type INTERNACAO
    modal run event_jepa_cube/modal_gpu.py --entity-type INTERNACAO --total-steps 5000
"""

from __future__ import annotations

import modal

# ---------------------------------------------------------------------------
# Modal app + volumes + image
# ---------------------------------------------------------------------------

app = modal.App("jepa-trainer")

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
jepa_cache = modal.Volume.from_name("jepa-cache", create_if_missing=True)

VOLUMES = {
    "/root/.cache/huggingface": hf_cache,
    "/root/jepa-artifacts": jepa_cache,
}

gpu_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu24.04", add_python="3.12"
    )
    .entrypoint([])
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .uv_pip_install(
        "torch>=2.6",
        "transformers>=4.50",
        "peft>=0.14",
        "accelerate>=0.35",
        "duckdb>=1.2.0",
        "pyarrow>=18.0",
        "huggingface-hub>=0.27",
        "numpy>=2.0",
        "bitsandbytes>=0.44",
    )
)


# ---------------------------------------------------------------------------
# Helpers (run inside GPU container)
# ---------------------------------------------------------------------------


def _get_hidden_dim(config) -> int:
    """Get hidden dimension from model config, handling nested/multimodal configs."""
    # Direct attribute (most models)
    for attr in ("hidden_size", "d_model", "dim", "n_embd"):
        val = getattr(config, attr, None)
        if isinstance(val, int):
            return val
    # Nested text_config (VLMs like Qwen3.5)
    text_cfg = getattr(config, "text_config", None)
    if text_cfg is not None:
        for attr in ("hidden_size", "d_model", "dim"):
            val = getattr(text_cfg, attr, None)
            if isinstance(val, int):
                return val
    raise ValueError(f"Cannot determine hidden_dim from config: {config}")


# ---------------------------------------------------------------------------
# GPU: extract hidden states
# ---------------------------------------------------------------------------


@app.function(
    image=gpu_image,
    gpu="A100-80GB",
    timeout=1800,
    volumes=VOLUMES,
)
def extract_hidden_states(
    texts: list[str],
    model_name: str = "Qwen/Qwen3.5-0.8B",
    max_length: int = 512,
    batch_size: int = 64,
) -> list[list[float]]:
    """Prefill-only hidden state extraction on A100.

    Uses AutoModel (no lm_head) + bf16 for max throughput.
    Returns mean-pooled last hidden states as float lists.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = "cuda"
    print(f"[extract] device={device}, texts={len(texts)}, model={model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    ).to(device).eval()

    hidden_dim = _get_hidden_dim(model.config)
    print(f"  hidden_dim={hidden_dim}, dtype=bf16, attn=sdpa")

    all_hidden: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        tokens = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
            )

        hidden = outputs.last_hidden_state  # (B, T, H)
        mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)

        for row in pooled.cpu().float().tolist():
            all_hidden.append(row)

        done = min(i + batch_size, len(texts))
        if (i // batch_size) % 5 == 0:
            print(f"  {done}/{len(texts)} ({done * 100 // len(texts)}%)")

    print(f"  Done: {len(all_hidden)} × {hidden_dim}")
    return all_hidden


# ---------------------------------------------------------------------------
# GPU: full JEPA training
# ---------------------------------------------------------------------------


@app.function(
    image=gpu_image,
    gpu="A100-80GB",
    timeout=3600,
    volumes=VOLUMES,
)
def train_on_gpu(
    flat_texts: list[str],
    flat_ts: list[float],
    entity_ranges: dict[str, list[int]],
    model_name: str = "Qwen/Qwen3.5-0.8B",
    embedding_dim: int = 64,
    max_length: int = 512,
    total_steps: int = 2000,
    batch_size: int = 32,
    base_lr: float = 3e-4,
) -> dict:
    """Full JEPA training on A100-80GB: extract + train.

    Phase A: Frozen Qwen bf16 → prefill → mean-pooled hidden states (cached on GPU)
    Phase B: Train projection + predictor with JEPA objective + regularizers
    """
    import copy
    import math
    import random
    import time

    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoTokenizer

    device = "cuda"
    random.seed(42)
    torch.manual_seed(42)

    n_events = len(flat_texts)
    n_entities = len(entity_ranges)
    print(f"=== JEPA Training on A100-80GB ===")
    print(f"Model: {model_name}")
    print(f"Events: {n_events:,}, Entities: {n_entities}")
    print(f"Steps: {total_steps}, Batch: {batch_size}, LR: {base_lr}")

    # ================================================================
    # Phase A: Extract hidden states (bf16, SDPA attention)
    # ================================================================
    print(f"\n[Phase A] Extracting hidden states...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    ).to(device).eval()

    hidden_dim = _get_hidden_dim(base_model.config)
    print(f"  hidden_dim={hidden_dim}")

    all_hidden_list: list[torch.Tensor] = []
    extract_bs = 64  # A100 can handle big batches

    for i in range(0, n_events, extract_bs):
        batch = flat_texts[i : i + extract_bs]
        tokens = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = base_model(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
            )

        hidden = outputs.last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)
        all_hidden_list.append(pooled.float())  # keep on GPU, float32

        done = min(i + extract_bs, n_events)
        if (i // extract_bs) % 5 == 0:
            print(f"  {done}/{n_events} ({done * 100 // n_events}%)")

    all_hidden = torch.cat(all_hidden_list, dim=0)  # (N, H) on GPU
    extract_time = time.time() - t0
    print(f"  Done: {all_hidden.shape} in {extract_time:.1f}s")
    print(f"  Throughput: {n_events / extract_time:.0f} events/s")

    # Free base model
    del base_model, all_hidden_list
    torch.cuda.empty_cache()

    # ================================================================
    # Phase B: Train projection + predictor (JEPA objective)
    # ================================================================
    print(f"\n[Phase B] Training projection + predictor...")

    # Projection: H → D
    mid = max(embedding_dim * 4, hidden_dim // 4)
    projection = nn.Sequential(
        nn.Linear(hidden_dim, mid),
        nn.GELU(),
        nn.LayerNorm(mid),
        nn.Linear(mid, embedding_dim),
    ).to(device)

    # EMA target projection
    ema_proj = copy.deepcopy(projection).eval()
    for p in ema_proj.parameters():
        p.requires_grad = False
    ema_tau = 0.996

    # Predictor MLP (discarded after training)
    predictor = nn.Sequential(
        nn.Linear(embedding_dim, 256),
        nn.GELU(),
        nn.LayerNorm(256),
        nn.Linear(256, embedding_dim),
    ).to(device)

    trainable = [p for p in list(projection.parameters()) + list(predictor.parameters()) if p.requires_grad]
    n_params = sum(p.numel() for p in trainable)
    optimizer = torch.optim.AdamW(trainable, lr=base_lr, weight_decay=0.01)
    print(f"  Trainable params: {n_params:,}")

    # Ranges
    ranges = {k: tuple(v) for k, v in entity_ranges.items()}

    # Smooth curriculum: linearly ramp difficulty over all steps
    # At step 0:          ctx_ratio=0.9, predict_steps=1
    # At total_steps:     ctx_ratio=0.3, predict_steps=5
    # No abrupt phase transitions

    # LR schedule: warmup → constant → cosine cooldown
    warmup = min(200, total_steps // 10)
    cooldown = min(400, total_steps // 5)
    primary = total_steps - cooldown

    def lr_mult(step: int) -> float:
        if step < warmup:
            return 0.1 + 0.9 * (step / warmup)
        if step < primary:
            return 1.0
        cs = step - primary
        if cs >= cooldown:
            return 0.002
        return 0.002 + 0.998 * 0.5 * (1.0 + math.cos(math.pi * cs / cooldown))

    def curriculum_at(step: int) -> tuple[float, float, int]:
        """Smooth curriculum: returns (ctx_ratio_min, ctx_ratio_max, pred_steps)."""
        progress = min(step / max(total_steps - 1, 1), 1.0)
        # Context ratio shrinks: 0.85→0.35
        ctx_max = 0.9 - 0.55 * progress
        ctx_min = max(ctx_max - 0.15, 0.2)
        # Prediction steps grow: 1→5
        pred_steps = max(1, int(1 + 4 * progress))
        return ctx_min, ctx_max, pred_steps

    losses_log: list[float] = []
    running_loss = 0.0
    t_train = time.time()

    # Build initial sample pool
    def build_samples(ctx_min: float, ctx_max: float, pred_steps: int):
        s = []
        for eid, (start, end) in ranges.items():
            n = end - start
            if n < 4:
                continue
            ratio = random.uniform(ctx_min, ctx_max)
            split = max(2, int(n * ratio))
            te = min(split + pred_steps, n)
            ctx_idx = list(range(start, start + split))
            tgt_idx = list(range(start + split, start + te))
            if tgt_idx:
                s.append((ctx_idx, tgt_idx))
        random.shuffle(s)
        return s

    ctx_min, ctx_max, pred_steps = curriculum_at(0)
    samples = build_samples(ctx_min, ctx_max, pred_steps)
    si = 0
    rebuild_every = max(50, n_entities * 2)  # rebuild samples periodically

    for step in range(total_steps):
        # Smooth curriculum: rebuild samples with current difficulty
        if step % rebuild_every == 0:
            ctx_min, ctx_max, pred_steps = curriculum_at(step)
            samples = build_samples(ctx_min, ctx_max, pred_steps)
            si = 0

        if si + batch_size > len(samples):
            random.shuffle(samples)
            si = 0

        batch_s = samples[si : si + batch_size]
        si += batch_size

        # LR schedule
        m = lr_mult(step)
        for pg in optimizer.param_groups:
            pg["lr"] = base_lr * m

        # Gather + pool hidden states
        ctx_pooled = []
        tgt_pooled = []
        for ctx_idx, tgt_idx in batch_s:
            ctx_pooled.append(all_hidden[ctx_idx].mean(dim=0))
            tgt_pooled.append(all_hidden[tgt_idx].mean(dim=0))

        ctx_h = torch.stack(ctx_pooled)  # already on GPU
        tgt_h = torch.stack(tgt_pooled)

        # Forward
        ctx_emb = projection(ctx_h)
        with torch.no_grad():
            tgt_emb = ema_proj(tgt_h)

        pred_emb = predictor(ctx_emb)

        # JEPA loss: smooth L1 (more robust than MSE for larger errors)
        loss = nn.functional.smooth_l1_loss(pred_emb, tgt_emb)

        B = ctx_emb.shape[0]
        if B > 1:
            # Covariance regularization (decorrelate dimensions — SIGReg-lite)
            centered = ctx_emb - ctx_emb.mean(dim=0, keepdim=True)
            cov = (centered.T @ centered) / (B - 1)
            off_diag = cov - torch.diag(cov.diag())
            cov_loss = off_diag.pow(2).sum() / (cov.shape[0] ** 2)

            # Variance regularization (prevent collapse)
            var_loss = torch.relu(1.0 - ctx_emb.std(dim=0)).mean()

            # Ramp regularization strength with curriculum progress
            progress = step / max(total_steps - 1, 1)
            reg_w = 0.01 + 0.09 * progress  # 0.01 → 0.10
            loss = loss + reg_w * (cov_loss + var_loss)

        # Backward + step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        # EMA update (ramp tau: 0.99 → 0.999 over training)
        progress = step / max(total_steps - 1, 1)
        tau = 0.99 + 0.009 * progress
        with torch.no_grad():
            for pe, po in zip(ema_proj.parameters(), projection.parameters()):
                pe.data.mul_(tau).add_(po.data, alpha=1 - tau)

        lv = loss.item()
        running_loss = 0.95 * running_loss + 0.05 * lv if step > 0 else lv
        losses_log.append(lv)

        if (step + 1) % 100 == 0:
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t_train
            sps = (step + 1) / elapsed
            print(
                f"  step {step + 1:5d}/{total_steps} | "
                f"loss={running_loss:.6f} | lr={lr:.2e} | "
                f"ctx=[{ctx_min:.2f},{ctx_max:.2f}] pred={pred_steps} | "
                f"{sps:.0f} steps/s"
            )

    train_time = time.time() - t_train
    total_time = extract_time + train_time
    print(f"\nTraining done: {total_steps} steps in {train_time:.1f}s")
    print(f"Total (extract + train): {total_time:.1f}s")

    # ================================================================
    # Encode all entities with trained projection
    # ================================================================
    print(f"\nEncoding {n_entities} entities...")
    entity_embeddings = {}
    projection.eval()
    with torch.no_grad():
        for eid, (start, end) in ranges.items():
            h = all_hidden[start:end].mean(dim=0, keepdim=True)
            emb = projection(h).cpu().squeeze(0).tolist()
            entity_embeddings[eid] = emb

    # Save projection to volume
    import json
    import os

    artifact_dir = "/root/jepa-artifacts/latest"
    os.makedirs(artifact_dir, exist_ok=True)

    torch.save(projection.cpu().state_dict(), f"{artifact_dir}/projection.pt")
    with open(f"{artifact_dir}/config.json", "w") as f:
        json.dump({"hidden_dim": hidden_dim, "embedding_dim": embedding_dim, "mid": mid}, f)
    with open(f"{artifact_dir}/embeddings.json", "w") as f:
        json.dump(entity_embeddings, f)

    jepa_cache.commit()
    print(f"  Saved to {artifact_dir}")

    return {
        "total_steps": total_steps,
        "final_loss": losses_log[-1] if losses_log else None,
        "losses": losses_log,
        "trainable_params": n_params,
        "hidden_dim": hidden_dim,
        "embedding_dim": embedding_dim,
        "n_entities": n_entities,
        "n_events": n_events,
        "entity_embeddings": entity_embeddings,
        "extract_time_s": extract_time,
        "train_time_s": train_time,
        "total_time_s": total_time,
    }


# ---------------------------------------------------------------------------
# GPU: LoRA fine-tune + JEPA with lookahead
# ---------------------------------------------------------------------------


@app.function(
    image=gpu_image,
    gpu="A100-80GB",
    timeout=7200,
    volumes=VOLUMES,
)
def train_lora_jepa(
    flat_texts: list[str],
    flat_ts: list[float],
    entity_ranges: dict[str, list[int]],
    model_name: str = "Qwen/Qwen3.5-0.8B",
    embedding_dim: int = 64,
    max_length: int = 512,
    # LoRA config
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_epochs: int = 2,
    lora_lr: float = 2e-5,
    lora_batch_size: int = 8,
    # JEPA config
    jepa_steps: int = 5000,
    jepa_batch_size: int = 32,
    jepa_lr: float = 3e-4,
    # Lookahead
    lookahead_steps: int = 5,
    lookahead_decay: float = 0.7,
) -> dict:
    """Full pipeline: LoRA fine-tune → extract → JEPA + lookahead.

    Phase A: LoRA fine-tune on domain text (causal LM loss)
    Phase B: Extract hidden states from LoRA-adapted model
    Phase C: Train projection + predictor with JEPA + multi-step lookahead

    The lookahead loss predicts representations at t+1, t+2, ..., t+K,
    weighted by exponential decay. This captures longer-range temporal
    dependencies in the latent space.
    """
    import copy
    import math
    import random
    import time

    import torch
    import torch.nn as nn

    device = "cuda"
    random.seed(42)
    torch.manual_seed(42)

    n_events = len(flat_texts)
    n_entities = len(entity_ranges)
    print(f"=== LoRA-JEPA Training on A100-80GB ===")
    print(f"Model: {model_name}")
    print(f"Events: {n_events:,}, Entities: {n_entities}")
    print(f"LoRA: rank={lora_rank}, epochs={lora_epochs}, lr={lora_lr}")
    print(f"JEPA: steps={jepa_steps}, batch={jepa_batch_size}, lr={jepa_lr}")
    print(f"Lookahead: K={lookahead_steps}, decay={lookahead_decay}")

    # ================================================================
    # Phase A: LoRA fine-tune (causal LM on domain text)
    # ================================================================
    print(f"\n[Phase A] LoRA fine-tuning on domain text...")
    t0 = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load full causal LM for LoRA fine-tuning
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    ).to(device)

    # Apply LoRA
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    lora_model = get_peft_model(base_model, lora_config)
    lora_trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    lora_total = sum(p.numel() for p in lora_model.parameters())
    print(f"  LoRA params: {lora_trainable:,} / {lora_total:,} "
          f"({lora_trainable * 100 / lora_total:.2f}%)")

    # LoRA training: causal LM loss on domain text
    lora_optimizer = torch.optim.AdamW(
        [p for p in lora_model.parameters() if p.requires_grad],
        lr=lora_lr,
        weight_decay=0.01,
    )

    lora_model.train()
    lora_losses = []

    # Shuffle and batch texts
    indices = list(range(n_events))
    for epoch in range(lora_epochs):
        random.shuffle(indices)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_events, lora_batch_size):
            batch_idx = indices[i : i + lora_batch_size]
            batch_texts = [flat_texts[j] for j in batch_idx]

            tokens = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)

            # Causal LM loss: predict next token
            labels = tokens["input_ids"].clone()
            labels[tokens["attention_mask"] == 0] = -100

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = lora_model(
                    input_ids=tokens["input_ids"],
                    attention_mask=tokens["attention_mask"],
                    labels=labels,
                )
                loss = outputs.loss

            lora_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(lora_model.parameters(), 1.0)
            lora_optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            lora_losses.append(loss.item())

            if n_batches % 100 == 0:
                print(f"  Epoch {epoch + 1}/{lora_epochs}, "
                      f"batch {n_batches}: loss={epoch_loss / n_batches:.4f}")

        if n_batches > 0:
            print(f"  Epoch {epoch + 1} done: avg_loss={epoch_loss / n_batches:.4f}")

    lora_time = time.time() - t0
    print(f"  LoRA fine-tuning done in {lora_time:.1f}s")

    # ================================================================
    # Phase B: Extract hidden states from LoRA-adapted model
    # ================================================================
    print(f"\n[Phase B] Extracting hidden states from LoRA-adapted model...")
    t1 = time.time()

    # Switch to base model mode for extraction (keep LoRA weights)
    lora_model.eval()

    # Get hidden dim from text config
    hidden_dim = _get_hidden_dim(base_model.config)
    print(f"  hidden_dim={hidden_dim}")

    all_hidden_list: list[torch.Tensor] = []
    extract_bs = 32

    for i in range(0, n_events, extract_bs):
        batch = flat_texts[i : i + extract_bs]
        tokens = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = lora_model(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
                output_hidden_states=True,
            )

        # Get last hidden state (before lm_head)
        hidden = outputs.hidden_states[-1]
        mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)
        all_hidden_list.append(pooled.float())

        done = min(i + extract_bs, n_events)
        if (i // extract_bs) % 10 == 0:
            print(f"  {done}/{n_events} ({done * 100 // n_events}%)")

    all_hidden = torch.cat(all_hidden_list, dim=0)  # (N, H) on GPU
    extract_time = time.time() - t1
    print(f"  Done: {all_hidden.shape} in {extract_time:.1f}s")

    # Free LoRA model
    del lora_model, base_model, all_hidden_list
    torch.cuda.empty_cache()

    # ================================================================
    # Phase C: JEPA + multi-step lookahead
    # ================================================================
    print(f"\n[Phase C] Training projection + predictor with lookahead...")

    # Projection: H → D
    mid = max(embedding_dim * 4, hidden_dim // 4)
    projection = nn.Sequential(
        nn.Linear(hidden_dim, mid),
        nn.GELU(),
        nn.LayerNorm(mid),
        nn.Linear(mid, embedding_dim),
    ).to(device)

    # EMA target projection
    ema_proj = copy.deepcopy(projection).eval()
    for p in ema_proj.parameters():
        p.requires_grad = False

    # Predictor: context_emb → predicted_target_emb
    predictor = nn.Sequential(
        nn.Linear(embedding_dim, 256),
        nn.GELU(),
        nn.LayerNorm(256),
        nn.Linear(256, embedding_dim),
    ).to(device)

    # Lookahead predictor: (embedding, step_encoding) → predicted_future_emb
    # Takes embedding + step number as input
    lookahead_predictor = nn.Sequential(
        nn.Linear(embedding_dim + 1, 256),  # +1 for step encoding
        nn.GELU(),
        nn.LayerNorm(256),
        nn.Linear(256, embedding_dim),
    ).to(device)

    trainable = [
        p for p in (
            list(projection.parameters()) +
            list(predictor.parameters()) +
            list(lookahead_predictor.parameters())
        ) if p.requires_grad
    ]
    n_params = sum(p.numel() for p in trainable)
    optimizer = torch.optim.AdamW(trainable, lr=jepa_lr, weight_decay=0.01)
    print(f"  Trainable params: {n_params:,}")

    # Ranges
    ranges = {k: tuple(v) for k, v in entity_ranges.items()}

    # Smooth curriculum
    warmup = min(200, jepa_steps // 10)
    cooldown = min(400, jepa_steps // 5)
    primary = jepa_steps - cooldown

    def lr_mult(step: int) -> float:
        if step < warmup:
            return 0.1 + 0.9 * (step / warmup)
        if step < primary:
            return 1.0
        cs = step - primary
        if cs >= cooldown:
            return 0.002
        return 0.002 + 0.998 * 0.5 * (1.0 + math.cos(math.pi * cs / cooldown))

    def curriculum_at(step: int) -> tuple[float, float, int]:
        progress = min(step / max(jepa_steps - 1, 1), 1.0)
        ctx_max = 0.9 - 0.55 * progress
        ctx_min = max(ctx_max - 0.15, 0.2)
        pred_steps = max(1, int(1 + (lookahead_steps - 1) * progress))
        return ctx_min, ctx_max, pred_steps

    def build_samples(ctx_min, ctx_max, pred_steps):
        s = []
        for eid, (start, end) in ranges.items():
            n = end - start
            if n < 4:
                continue
            ratio = random.uniform(ctx_min, ctx_max)
            split = max(2, int(n * ratio))
            te = min(split + pred_steps, n)
            ctx_idx = list(range(start, start + split))
            tgt_idx = list(range(start + split, start + te))
            if tgt_idx:
                s.append((ctx_idx, tgt_idx))
        random.shuffle(s)
        return s

    losses_log: list[float] = []
    jepa_losses_log: list[float] = []
    lookahead_losses_log: list[float] = []
    running_loss = 0.0
    t_train = time.time()
    rebuild_every = max(50, n_entities * 2)

    ctx_min, ctx_max, pred_steps = curriculum_at(0)
    samples = build_samples(ctx_min, ctx_max, pred_steps)
    si = 0

    for step in range(jepa_steps):
        if step % rebuild_every == 0:
            ctx_min, ctx_max, pred_steps = curriculum_at(step)
            samples = build_samples(ctx_min, ctx_max, pred_steps)
            si = 0

        if si + jepa_batch_size > len(samples):
            random.shuffle(samples)
            si = 0

        batch_s = samples[si : si + jepa_batch_size]
        si += jepa_batch_size

        # LR
        m = lr_mult(step)
        for pg in optimizer.param_groups:
            pg["lr"] = jepa_lr * m

        # --- JEPA loss: context → predict target ---
        ctx_pooled = []
        tgt_pooled = []
        for ctx_idx, tgt_idx in batch_s:
            ctx_pooled.append(all_hidden[ctx_idx].mean(dim=0))
            tgt_pooled.append(all_hidden[tgt_idx].mean(dim=0))

        ctx_h = torch.stack(ctx_pooled)
        tgt_h = torch.stack(tgt_pooled)

        ctx_emb = projection(ctx_h)
        with torch.no_grad():
            tgt_emb = ema_proj(tgt_h)

        pred_emb = predictor(ctx_emb)
        jepa_loss = nn.functional.smooth_l1_loss(pred_emb, tgt_emb)

        # --- Lookahead loss: predict t+1, t+2, ..., t+K ---
        la_loss = torch.tensor(0.0, device=device)
        la_count = 0

        for ctx_idx, tgt_idx in batch_s:
            n_future = len(tgt_idx)
            if n_future < 2:
                continue

            # Context embedding for this sample
            ctx_e = projection(all_hidden[ctx_idx].mean(dim=0).unsqueeze(0))  # (1, D)

            for k in range(min(n_future, lookahead_steps)):
                # Target at step k
                with torch.no_grad():
                    tgt_k = ema_proj(all_hidden[tgt_idx[k]].unsqueeze(0))  # (1, D)

                # Predict step k: concat context with normalized step number
                step_enc = torch.tensor([[k / lookahead_steps]], device=device)
                la_input = torch.cat([ctx_e, step_enc], dim=-1)  # (1, D+1)
                pred_k = lookahead_predictor(la_input)  # (1, D)

                weight = lookahead_decay ** k
                la_loss = la_loss + weight * nn.functional.smooth_l1_loss(pred_k, tgt_k)
                la_count += 1

        if la_count > 0:
            la_loss = la_loss / la_count

        # --- Regularization ---
        B = ctx_emb.shape[0]
        reg_loss = torch.tensor(0.0, device=device)
        if B > 1:
            centered = ctx_emb - ctx_emb.mean(dim=0, keepdim=True)
            cov = (centered.T @ centered) / (B - 1)
            off_diag = cov - torch.diag(cov.diag())
            cov_loss = off_diag.pow(2).sum() / (cov.shape[0] ** 2)
            var_loss = torch.relu(1.0 - ctx_emb.std(dim=0)).mean()

            progress = step / max(jepa_steps - 1, 1)
            reg_w = 0.01 + 0.09 * progress
            reg_loss = reg_w * (cov_loss + var_loss)

        # --- Total loss ---
        loss = jepa_loss + 0.5 * la_loss + reg_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        # EMA update
        progress = step / max(jepa_steps - 1, 1)
        tau = 0.99 + 0.009 * progress
        with torch.no_grad():
            for pe, po in zip(ema_proj.parameters(), projection.parameters()):
                pe.data.mul_(tau).add_(po.data, alpha=1 - tau)

        lv = loss.item()
        running_loss = 0.95 * running_loss + 0.05 * lv if step > 0 else lv
        losses_log.append(lv)
        jepa_losses_log.append(jepa_loss.item())
        lookahead_losses_log.append(la_loss.item() if la_count > 0 else 0.0)

        if (step + 1) % 100 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  step {step + 1:5d}/{jepa_steps} | "
                f"total={running_loss:.4f} jepa={jepa_losses_log[-1]:.4f} "
                f"la={lookahead_losses_log[-1]:.4f} | lr={lr:.2e} | "
                f"ctx=[{ctx_min:.2f},{ctx_max:.2f}] K={pred_steps}"
            )

    train_time = time.time() - t_train
    total_time = lora_time + extract_time + train_time
    print(f"\nJEPA training done: {jepa_steps} steps in {train_time:.1f}s")
    print(f"Total pipeline: {total_time:.1f}s")

    # ================================================================
    # Encode all entities
    # ================================================================
    print(f"\nEncoding {n_entities} entities...")
    entity_embeddings = {}
    projection.eval()
    with torch.no_grad():
        for eid, (start, end) in ranges.items():
            h = all_hidden[start:end].mean(dim=0, keepdim=True)
            emb = projection(h).cpu().squeeze(0).tolist()
            entity_embeddings[eid] = emb

    # Save artifacts
    import json
    import os

    artifact_dir = "/root/jepa-artifacts/latest-lora"
    os.makedirs(artifact_dir, exist_ok=True)
    torch.save(projection.cpu().state_dict(), f"{artifact_dir}/projection.pt")
    torch.save(lookahead_predictor.cpu().state_dict(), f"{artifact_dir}/lookahead.pt")
    with open(f"{artifact_dir}/config.json", "w") as f:
        json.dump({
            "hidden_dim": hidden_dim, "embedding_dim": embedding_dim,
            "lora_rank": lora_rank, "lookahead_steps": lookahead_steps,
        }, f)
    with open(f"{artifact_dir}/embeddings.json", "w") as f:
        json.dump(entity_embeddings, f)
    jepa_cache.commit()
    print(f"  Saved to {artifact_dir}")

    return {
        "total_steps": jepa_steps,
        "final_loss": losses_log[-1] if losses_log else None,
        "final_jepa_loss": jepa_losses_log[-1] if jepa_losses_log else None,
        "final_lookahead_loss": lookahead_losses_log[-1] if lookahead_losses_log else None,
        "losses": losses_log,
        "lora_losses": lora_losses,
        "trainable_params": n_params,
        "lora_params": lora_trainable,
        "hidden_dim": hidden_dim,
        "embedding_dim": embedding_dim,
        "n_entities": n_entities,
        "n_events": n_events,
        "entity_embeddings": entity_embeddings,
        "lora_time_s": lora_time,
        "extract_time_s": extract_time,
        "train_time_s": train_time,
        "total_time_s": total_time,
    }


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(
    action: str = "lora",
    entity_type: str = "INTERNACAO",
    db_path: str = "data/aggregated_fixed_union.db",
    limit_entities: int = 50,
    limit_events: int = 300,
    model_name: str = "Qwen/Qwen3.5-0.8B",
    total_steps: int = 5000,
    batch_size: int = 32,
    lora_epochs: int = 2,
    lookahead_steps: int = 5,
):
    """Local: load graph context → serialize → send to GPU for training.

    Actions:
        lora  — Full pipeline: graph context + LoRA + JEPA + lookahead
        train — Legacy: single-entity-type, no LoRA, no lookahead
    """
    import math
    import sys
    import time

    sys.path.insert(0, ".")

    # Load data with graph context (cross-table relationships)
    from event_jepa_cube.graph_loader import EntityGraph

    print(f"Loading graph context for {entity_type} from {db_path}...")
    t0 = time.time()
    graph = EntityGraph(db_path)
    graph.discover()
    contexts = graph.load_all_contexts(
        f"ID_CD_{entity_type}",
        limit_entities=limit_entities,
        max_events=limit_events,
    )
    graph.close()

    # Flatten for Modal
    flat_texts: list[str] = []
    flat_ts: list[float] = []
    entity_ranges: dict[str, list[int]] = {}

    for ctx in contexts:
        start = len(flat_texts)
        for e in ctx.events:
            flat_texts.append(e.text)
            flat_ts.append(e.epoch)
        entity_ranges[ctx.entity_id] = [start, len(flat_texts)]

    n_events = len(flat_texts)
    elapsed = time.time() - t0
    print(f"  {len(contexts)} entities, {n_events:,} events ({elapsed:.1f}s)")
    print(f"  Avg {n_events / len(contexts):.0f} events/entity "
          f"(graph context from {sum(len(c.related_ids) for c in contexts) / len(contexts):.1f} "
          f"related types)")

    if action == "lora":
        print(f"\nSending to GPU: LoRA + JEPA + lookahead "
              f"({total_steps} steps, K={lookahead_steps})...")
        result = train_lora_jepa.remote(
            flat_texts=flat_texts,
            flat_ts=flat_ts,
            entity_ranges=entity_ranges,
            model_name=model_name,
            embedding_dim=64,
            lora_epochs=lora_epochs,
            jepa_steps=total_steps,
            jepa_batch_size=batch_size,
            lookahead_steps=lookahead_steps,
        )

        print(f"\n{'=' * 60}")
        print(f"RESULTS — LoRA-JEPA + Lookahead")
        print(f"{'=' * 60}")
        print(f"JEPA steps:     {result['total_steps']}")
        print(f"Final loss:     {result['final_loss']:.6f}")
        print(f"  JEPA loss:    {result['final_jepa_loss']:.6f}")
        print(f"  Lookahead:    {result['final_lookahead_loss']:.6f}")
        print(f"LoRA time:      {result['lora_time_s']:.1f}s")
        print(f"Extract time:   {result['extract_time_s']:.1f}s")
        print(f"Train time:     {result['train_time_s']:.1f}s")
        print(f"Total time:     {result['total_time_s']:.1f}s")
        print(f"JEPA params:    {result['trainable_params']:,}")
        print(f"LoRA params:    {result['lora_params']:,}")
        print(f"Hidden dim:     {result['hidden_dim']}")
        print(f"Embed dim:      {result['embedding_dim']}")
        print(f"Entities:       {result['n_entities']}")
        print(f"Events:         {result['n_events']:,}")

        # LoRA loss curve
        if result.get("lora_losses"):
            ll = result["lora_losses"]
            print(f"\nLoRA loss ({len(ll)} batches): "
                  f"{ll[0]:.4f} → {ll[-1]:.4f}")

        # JEPA loss curve
        losses = result["losses"]
        n = len(losses)
        mx = max(losses) if losses else 1
        print(f"\nJEPA+Lookahead loss ({n} steps):")
        for i in range(0, n, max(1, n // 20)):
            bar = "█" * max(1, int(losses[i] * 50 / mx))
            print(f"  {i:5d} │ {losses[i]:.6f} {bar}")

    elif action == "train":
        print(f"\nSending to GPU for training ({total_steps} steps, batch={batch_size})...")
        result = train_on_gpu.remote(
            flat_texts=flat_texts,
            flat_ts=flat_ts,
            entity_ranges=entity_ranges,
            model_name=model_name,
            embedding_dim=64,
            total_steps=total_steps,
            batch_size=batch_size,
        )

        print(f"\n{'=' * 60}")
        print(f"RESULTS")
        print(f"{'=' * 60}")
        print(f"Steps:        {result['total_steps']}")
        print(f"Final loss:   {result['final_loss']:.6f}")
        print(f"Total time:   {result['total_time_s']:.1f}s")

    # Entity embeddings (common)
    embs = result.get("entity_embeddings", {})
    if embs:
        print(f"\nLearned embeddings ({len(embs)} entities):")
        for eid, emb in list(embs.items())[:10]:
            norm = math.sqrt(sum(v * v for v in emb))
            top = " ".join(f"{v:+.3f}" for v in emb[:8])
            print(f"  #{eid}: ‖e‖={norm:.3f}  [{top} ...]")
        if len(embs) > 10:
            print(f"  ... and {len(embs) - 10} more")
