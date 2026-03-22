"""LoRA encoder using Qwen as backbone for temporal JEPA embeddings.

Two-phase architecture for efficient training on consumer hardware:

  Phase A (extract, run once):
    Frozen Qwen base → prefill → mean-pooled hidden states → cache to disk
    This is SLOW (~6s/batch on MPS) but only runs once per dataset.

  Phase B (train, fast iteration):
    Cached hidden states → trainable projection → embedding_dim
    Projection + predictor train with JEPA loss. Instant per step.

  Phase C (LoRA fine-tune, GPU only):
    Once projection is validated, optionally fine-tune LoRA adapters
    on GPU via vLLM or dedicated training server.

Requires: torch, transformers, peft
"""

from __future__ import annotations

import datetime
import math
import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Row serialization — DB rows → compact text for the LM
# ---------------------------------------------------------------------------


class RowSerializer:
    """Converts database rows to compact text tokens.

    Format per event:
        [table_short|2024-01-15 10:30] col1=val1 col2=val2 ...

    Events are joined by newline. Most recent events kept when truncating.
    """

    MAX_VAL_LEN = 48

    @staticmethod
    def shorten_table(name: str) -> str:
        return name.replace("agg_tb_", "")

    @staticmethod
    def format_ts(value: Any) -> str:
        if isinstance(value, datetime.datetime):
            return value.strftime("%Y-%m-%d %H:%M")
        if isinstance(value, datetime.date):
            return value.strftime("%Y-%m-%d")
        if isinstance(value, (int, float)):
            try:
                dt = datetime.datetime.fromtimestamp(value)
                return dt.strftime("%Y-%m-%d %H:%M")
            except (OSError, ValueError):
                return str(value)[:16]
        return str(value)[:16]

    @staticmethod
    def format_val(value: Any) -> str:
        if value is None:
            return "NULL"
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return "NULL"
            return f"{value:.2f}"
        if isinstance(value, bool):
            return "T" if value else "F"
        s = str(value)
        if len(s) > RowSerializer.MAX_VAL_LEN:
            s = s[: RowSerializer.MAX_VAL_LEN]
        return s

    @classmethod
    def serialize_event(
        cls,
        table: str,
        timestamp: Any,
        columns: Dict[str, Any],
    ) -> str:
        """Serialize one event (row) to a compact text line."""
        parts = [f"[{cls.shorten_table(table)}|{cls.format_ts(timestamp)}]"]
        for k, v in columns.items():
            if v is not None:
                parts.append(f"{k}={cls.format_val(v)}")
        return " ".join(parts)

    @classmethod
    def serialize_sequence(cls, events: List[str], max_chars: int = 3000) -> str:
        """Join event lines. Keep most recent if over budget."""
        text = "\n".join(events)
        if len(text) > max_chars:
            text = text[-max_chars:]
            nl = text.find("\n")
            if nl > 0:
                text = text[nl + 1 :]
        return text


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Hidden state extractor (Phase A — run once, no gradient)
# ---------------------------------------------------------------------------


class HiddenExtractor:
    """Extracts hidden states from frozen Qwen base model.

    Prefill-only: one forward pass per batch, no decoding, no LoRA.
    Output: mean-pooled last hidden layer per input text.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-0.8B",
        device: Optional[str] = None,
        max_length: int = 512,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device_name = device or detect_device()
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # fp32 on MPS/CPU, fp16 on CUDA
        dtype = torch.float16 if self.device_name == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype, trust_remote_code=True
        )
        self.model.to(self.device_name)
        self.model.eval()

        # Freeze everything
        for p in self.model.parameters():
            p.requires_grad = False

        self.hidden_dim = self.model.config.hidden_size

    @torch.no_grad()
    def extract(self, texts: List[str]) -> torch.Tensor:
        """Prefill-only: text → mean-pooled hidden states. (B, H)"""
        tokens = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device_name)

        outputs = self.model(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
            output_hidden_states=True,
            use_cache=False,
        )

        hidden = outputs.hidden_states[-1]  # (B, T, H)
        mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)
        return pooled.cpu().float()  # always return cpu float32

    def extract_and_cache(
        self,
        all_texts: List[str],
        cache_path: str,
        batch_size: int = 4,
    ) -> torch.Tensor:
        """Extract hidden states for all texts and save to disk.

        Returns (N, H) tensor. Cached for reuse.
        """
        if os.path.exists(cache_path):
            cached = torch.load(cache_path, weights_only=True)
            if cached.shape[0] == len(all_texts):
                print(f"  Loaded cached hidden states from {cache_path}")
                return cached

        all_hidden: List[torch.Tensor] = []
        n = len(all_texts)

        for i in range(0, n, batch_size):
            batch = all_texts[i : i + batch_size]
            h = self.extract(batch)
            all_hidden.append(h)
            if (i // batch_size) % 10 == 0:
                print(f"  Extracting: {i}/{n} ({i*100//n}%)", flush=True)

        result = torch.cat(all_hidden, dim=0)  # (N, H)
        torch.save(result, cache_path)
        print(f"  Cached {result.shape} to {cache_path}")
        return result


# ---------------------------------------------------------------------------
# Trainable projection head (Phase B — fast training)
# ---------------------------------------------------------------------------


class ProjectionHead(nn.Module):
    """Trainable projection: hidden_dim → embedding_dim.

    This is what we train with JEPA loss on cached hidden states.
    Small enough for instant forward/backward.
    """

    def __init__(self, hidden_dim: int, embedding_dim: int = 64) -> None:
        super().__init__()
        mid = max(embedding_dim * 2, hidden_dim // 4)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, mid),
            nn.GELU(),
            nn.LayerNorm(mid),
            nn.Linear(mid, embedding_dim),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """(B, H) → (B, D)"""
        return self.net(hidden_states)
