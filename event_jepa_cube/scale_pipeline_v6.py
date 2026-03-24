"""V6 Dense Temporal JEPA pipeline — World Model Architecture.

Builds on V5's 35.2M-node / 165M-edge healthcare TKG. Key changes from V5:
    - REMOVES all auxiliary topological losses (edge_type_pred, link_pred,
      temporal_ord, contrastive, dgi, dense_ctx, VICReg variance hinge)
    - ADDS Dense Temporal Lookahead: predict EMA representation of EACH future
      event individually, conditioned on context + time query
    - ADDS Weak-SIGReg: off-diagonal covariance penalty only (no variance hinge)
    - ADDS NumericEncoder: xVal-style Fourier encoding for billing amounts,
      quantities (VL_TOTAL etc.) — lost as text in V5
    - 2-phase curriculum (foundation without TGN, temporal with TGN)
    - latent_dim=128, edge_feat_dim=112 (64 pred + 16 time + 32 numeric)
    - V5 epoch 1 warm-start: pad 64-dim with zeros to 128-dim

Loss function:
    L = L_dense_lookahead + lambda * L_weak_sigreg

That's it. Two terms. Everything else is architectural.

Output contract (unchanged):
    (num_nodes, 128) tensor saved as node_embeddings.pt

Usage:
    modal run --detach event_jepa_cube/scale_pipeline_v6.py --action full
    modal run --detach event_jepa_cube/scale_pipeline_v6.py --action train
    modal run event_jepa_cube/scale_pipeline_v6.py --action materialize
    modal run event_jepa_cube/scale_pipeline_v6.py --action train --config '{"epochs":3}'
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, TypedDict, cast

import modal

if TYPE_CHECKING:
    import torch


class MaterializeStats(TypedDict):
    n_edges: int
    edge_file_mb: float
    edge_time_s: float
    n_ontology_nodes: int
    ontology_file_mb: float
    ontology_time_s: float


class TrainResult(TypedDict):
    total_steps: int
    final_loss: float | None
    final_dense_loss: float | None
    final_sigreg_loss: float | None
    train_time_s: float
    num_nodes: int
    num_edges: int
    num_predicates: int
    latent_dim: int
    epochs: int
    gps_layers: int

try:
    import duckdb
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
except ImportError:
    duckdb = None  # type: ignore
    pa = None  # type: ignore
    pc = None  # type: ignore
    pq = None  # type: ignore


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class V6Config:
    """All hyperparameters centralized for V6 Dense Temporal JEPA."""

    # Dimensions — 128 for single A100-80GB
    latent_dim: int = 128
    time_dim: int = 16
    numeric_dim: int = 32
    edge_feat_dim: int = 112  # pred_emb(64) + time(16) + numeric(32)
    pred_emb_dim: int = 64  # predicate embedding dimension for edge features
    rwse_dim: int = 16
    lappe_dim: int = 0  # disabled in V6 (simplification)

    # NumericEncoder
    numeric_n_frequencies: int = 16

    # GraphGPS
    gps_layers: int = 3
    gps_heads: int = 8
    gps_ffn_mult: int = 4
    gps_dropout: float = 0.1

    # GatedGCN
    gcn_bn: bool = True

    # TGN Memory
    tgn_dim: int = 64
    tgn_msg_dim: int = 64

    # Training
    batch_size: int = 512
    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_frac: float = 0.1
    grad_clip: float = 1.0
    emb_lr_mult: float = 10.0
    emb_momentum: float = 0.0
    emb_weight_decay: float = 0.0

    # EMA
    ema_tau_start: float = 0.996
    ema_tau_end: float = 0.9999

    # Dense Temporal Lookahead
    lookahead_steps: int = 5
    lookahead_decay: float = 0.8

    # Dense Temporal Predictor
    predictor_hidden_mult: int = 2

    # Weak-SIGReg
    sigreg_lambda: float = 0.04  # lambda for L_weak_sigreg

    # BGE-M3
    bge_model: str = "BAAI/bge-m3"
    bge_hidden_dim: int = 1024
    bge_max_length: int = 128
    bge_batch_size: int = 128

    # Encoder name for compatibility
    encoder_model: str = "BAAI/bge-m3"

    # Data paths
    parquet_path: str = "/data/jcube_graph_v6.parquet"
    ontology_path: str = "/data/ontology_nodes.parquet"
    artifact_dir: str = "/root/jepa-artifacts/tkg-v6"
    v5_artifact_dir: str = "/root/jepa-artifacts/tkg-v5"

    # AMP
    use_amp: bool = True
    amp_dtype: str = "bfloat16"

    # Workers
    num_workers: int = 4
    prefetch_factor: int = 2

    # Single-hospital mode (filter subgraph at load time)
    hospital_filter: str = ""  # e.g. "GHO-BRADESCO" — empty = all hospitals
    tgn_from_epoch: int = 3  # override: set to 0 for single-hospital fast test

    # Logging
    log_every: int = 50

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, s: str) -> V6Config:
        return cls(**json.loads(s))


@dataclass
class CurriculumPhase:
    """Progressive training curriculum phase (V6: 2 phases only)."""

    name: str
    epoch_start: int
    epoch_end: int  # exclusive
    num_neighbors: list[int]
    tgn_enabled: bool = False
    node_sample_frac: float = 1.0


# V6 curriculum: 2 phases (foundation without TGN, temporal with TGN)
# Same two losses the entire time. Only change is TGN activation.
CURRICULUM_PHASES_V6: list[CurriculumPhase] = [
    CurriculumPhase(
        name="foundation",
        epoch_start=0, epoch_end=3,
        num_neighbors=[15, 10],
        tgn_enabled=False,
        node_sample_frac=0.02,  # 10% of nodes → ~6.2K batches/epoch (was 62K)
    ),
    CurriculumPhase(
        name="temporal",
        epoch_start=3, epoch_end=10,
        num_neighbors=[15, 10, 5],
        tgn_enabled=True,
        node_sample_frac=0.05,  # 25% of nodes → ~15.7K batches/epoch
    ),
]


# ---------------------------------------------------------------------------
# Timestamp priority for picking the best temporal column per table
# ---------------------------------------------------------------------------

TIMESTAMP_PRIORITY = [
    "DH_CADASTRO", "DH_REALIZACAO", "DH_INICIO", "DH_FIM",
    "DH_ATUALIZACAO", "DT_INTER", "DT_ALTA", "DT_NASC",
]

# Instance entity types get source_db prefix to avoid cross-hospital ID collisions.
# Ontology types (codes, categories, procedures) stay global.
INSTANCE_TYPES = {
    'PACIENTE', 'INTERNACAO', 'BENEFICIARIO', 'FATURA', 'PESSOA',
    'ORCAMENTO', 'HOSPITAL', 'COOPERADO', 'EVOLUCAO', 'COTACAO',
    'TERCEIRO', 'RELATORIO', 'AUDITORIA', 'LEITO', 'MEDICO',
    'RH', 'LOGIN', 'ACESSO', 'PERMISSAO', 'TICKETS',
    'LOG', 'STATUS', 'CONTROLE', 'CAPTA', 'PS', 'ITEM',
    'AVEXAME', 'AVESP', 'ENVIO', 'NEGOCIACAO', 'OXITERAPIA',
    'VISITA', 'FEEDBACK', 'PROTOCOLO', 'INTERACAO', 'PRODUTO',
    'RESPOSTA', 'QUESTIONARIO', 'EQPMULTI', 'CAMPO', 'HISTORICO',
    'LOCALIZACAO', 'DADO', 'CONTATO', 'ENDERECO', 'TELEFONE',
    'EMAIL', 'MENSAGEM', 'ARQUIVO', 'ARQ', 'ANEXO', 'EQUIPE',
    'SOLICITACAO', 'DATA', 'HEMOTERAPIA', 'PREV', 'PRE',
    'LANCAMENTO', 'PARCELA', 'CONTEXTO', 'AUXILIAR', 'SAIDAPRODS',
}

_TS_TYPES = {"TIMESTAMP", "TIMESTAMP WITH TIME ZONE", "DATE", "DATETIME", "TIMESTAMPTZ"}

# Numeric columns to extract per table (column_name -> predicate_suffix)
NUMERIC_COLUMNS = {
    "VL_TOTAL": True,
    "VL_GLOSA_FECHAMENTO": True,
    "VL_GLOSA": True,
    "NR_QTD_GLOSADO": True,
    "NR_QTD": True,
    "VL_UNITARIO": True,
    "VL_COBRADO": True,
}


# ---------------------------------------------------------------------------
# Modal infrastructure
# ---------------------------------------------------------------------------

scale_app = modal.App("jcube-tkg-jepa-v6")

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
jepa_cache = modal.Volume.from_name("jepa-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("jcube-data", create_if_missing=True)

VOLUMES = {
    "/root/.cache/huggingface": hf_cache,
    "/root/jepa-artifacts": jepa_cache,
    "/data": data_vol,
}

# CPU image for DuckDB materialization (no GPU needed)
cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("duckdb>=1.2.0", "pyarrow>=18.0")
)

# V6 GPU image — BGE-M3 via sentence-transformers, PyG
gpu_image_v6 = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.0-devel-ubuntu24.04", add_python="3.12"
    )
    .entrypoint([])
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .pip_install(
        "torch>=2.6",
        "numpy>=2.0",
        "pyarrow>=18.0",
        "transformers>=4.50",
        "sentence-transformers>=3.0",
        "accelerate>=0.35",
        "huggingface-hub>=0.27",
    )
    .pip_install("torch_geometric>=2.6")
    .pip_install(
        "torch_scatter", "torch_sparse", "torch_cluster",  # no pyg-lib (no wheel for torch 2.10)
        find_links="https://data.pyg.org/whl/torch-2.10.0+cu128.html",
    )
)


# ---------------------------------------------------------------------------
# Remote DuckDB materialization — V6: adds numeric_value column
# ---------------------------------------------------------------------------


@scale_app.function(
    image=cpu_image,
    timeout=7200,
    volumes=VOLUMES,
    cpu=8,
    memory=32768,
)
def materialize_remote(
    catalog_json: str,
    db_path: str = "/data/aggregated_fixed_union.db",
    output_path: str = "/data/jcube_graph_v6.parquet",
) -> MaterializeStats:
    """Materialize edges inside Modal container.

    V6 change: adds numeric_value column to edges for xVal encoding.
    Edges from tables with VL_TOTAL etc. carry the numeric value;
    edges without numeric values get 0.0.
    """
    import json
    import time

    import duckdb
    import pyarrow.parquet as pq_local

    catalog = json.loads(catalog_json)
    tables = catalog.get("tables", [])

    ts_priority = [
        "DH_CADASTRO", "DH_REALIZACAO", "DH_INICIO", "DH_FIM",
        "DH_ATUALIZACAO", "DT_INTER", "DT_ALTA", "DT_NASC",
    ]
    ts_types = {"TIMESTAMP", "TIMESTAMP WITH TIME ZONE", "DATE", "DATETIME", "TIMESTAMPTZ"}

    numeric_cols_set = {
        "VL_TOTAL", "VL_GLOSA_FECHAMENTO", "VL_GLOSA",
        "NR_QTD_GLOSADO", "NR_QTD", "VL_UNITARIO", "VL_COBRADO",
    }

    selects = []
    for table in tables:
        t_name = table["name"]
        cols = table.get("columns", [])
        id_cols = [c["name"] for c in cols if c["name"].startswith("ID_CD_")]
        ts_cols = [c["name"] for c in cols
                   if c.get("data_type", "") in ts_types
                   or c["name"].startswith("DH_") or c["name"].startswith("DT_")]
        if len(id_cols) < 2 or not ts_cols:
            continue
        t_col = ts_cols[0]
        col_names = [c["name"] for c in cols]
        for tp in ts_priority:
            if tp in col_names:
                t_col = tp
                break
        subject_col = id_cols[0]
        subj_type = subject_col.replace("ID_CD_", "")

        # Find numeric column for this table (first match wins)
        num_col = None
        for nc_name in numeric_cols_set:
            if nc_name in col_names:
                num_col = nc_name
                break

        numeric_expr = f'COALESCE(CAST("{num_col}" AS DOUBLE), 0.0)' if num_col else "0.0"

        for obj_col in id_cols[1:]:
            obj_type = obj_col.replace("ID_CD_", "")
            pred_name = f"HAS_{obj_type}_VIA_{t_name}"

            if subj_type in INSTANCE_TYPES:
                subj_expr = f"source_db || '/' || '{subject_col}_' || CAST(\"{subject_col}\" AS VARCHAR)"
            else:
                subj_expr = f"'{subject_col}_' || CAST(\"{subject_col}\" AS VARCHAR)"

            if obj_type in INSTANCE_TYPES:
                obj_expr = f"source_db || '/' || '{obj_col}_' || CAST(\"{obj_col}\" AS VARCHAR)"
            else:
                obj_expr = f"'{obj_col}_' || CAST(\"{obj_col}\" AS VARCHAR)"

            selects.append(
                f"SELECT "
                f"{subj_expr} AS subject_id, "
                f"'{pred_name}' AS predicate, "
                f"{obj_expr} AS object_id, "
                f"EPOCH(\"{t_col}\") AS t_epoch, "
                f"{numeric_expr} AS numeric_value "
                f"FROM \"{t_name}\" "
                f"WHERE \"{subject_col}\" IS NOT NULL "
                f"AND \"{obj_col}\" IS NOT NULL "
                f"AND \"{t_col}\" IS NOT NULL"
            )

    query = " UNION ALL ".join(selects)
    print(f"Query: {len(selects)} SELECT statements (V6 with numeric_value)")

    # Connect with out-of-core settings
    con = duckdb.connect(db_path, read_only=True)
    con.execute("PRAGMA temp_directory='/data/duckdb_temp'")
    con.execute("PRAGMA threads=8")
    con.execute("PRAGMA memory_limit='24GB'")

    # Materialize edges to Parquet
    import os
    edge_time = 0.0
    if os.path.exists(output_path) and os.path.getsize(output_path) > 100_000_000:
        meta = pq_local.read_metadata(output_path)
        # Check if V6 schema (has numeric_value column)
        schema = pq_local.read_schema(output_path)
        has_numeric = "numeric_value" in schema.names
        file_mb = os.path.getsize(output_path) / 1e6
        if has_numeric:
            print(f"  SKIP edges -- already materialized (V6): {meta.num_rows:,} edges ({file_mb:.1f} MB)")
            edge_time = 0.0
        else:
            print("  Existing file lacks numeric_value column, re-materializing...")
            os.remove(output_path)
            has_numeric = False
    else:
        has_numeric = False
        meta = None

    if not has_numeric or meta is None:
        print("Materializing edges to Parquet (ZSTD, out-of-core, V6 with numeric_value)...")
        t0 = time.time()
        con.execute(
            f"COPY ({query}) TO '{output_path}' "
            f"(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 1000000)"
        )
        edge_time = time.time() - t0
        meta = pq_local.read_metadata(output_path)
        file_mb = os.path.getsize(output_path) / 1e6
        print(f"  {meta.num_rows:,} edges in {edge_time:.1f}s ({file_mb:.1f} MB)")

    # ---------------------------------------------------------------
    # Extract ONTOLOGY node texts (same as V5)
    # ---------------------------------------------------------------
    print("\nExtracting ontology node texts from dictionary tables...")
    t1 = time.time()

    fk_usage: dict[str, int] = {}
    for t in tables:
        for c in t.get("columns", []):
            if c["name"].startswith("ID_CD_"):
                col_name = c["name"]
                fk_usage[col_name] = fk_usage.get(col_name, 0) + 1

    ontology_queries = []
    for t in tables:
        cols = t.get("columns", [])
        row_count = t.get("row_count", 0)
        id_cols_t = [c["name"] for c in cols if c["name"].startswith("ID_CD_")]
        name_cols = [c["name"] for c in cols
                     if c["name"].startswith("NM_") or c["name"].startswith("DS_")]

        if len(id_cols_t) != 1 or not name_cols or row_count > 50000:
            continue
        entity_type = id_cols_t[0].replace("ID_CD_", "")
        if entity_type in INSTANCE_TYPES:
            continue
        refs = fk_usage.get(id_cols_t[0], 0)
        if refs < 3:
            continue

        text_parts = [f"'<{entity_type}>'"]
        for nc in name_cols[:4]:
            text_parts.append(f"' {nc}=' || COALESCE(CAST(\"{nc}\" AS VARCHAR), '')")
        text_expr = " || ".join(text_parts)

        ontology_queries.append(
            f"SELECT '{id_cols_t[0]}_' || CAST(\"{id_cols_t[0]}\" AS VARCHAR) AS node_id, "
            f"'{entity_type}' AS entity_type, "
            f"{text_expr} AS text_payload "
            f"FROM \"{t['name']}\" "
            f"WHERE \"{id_cols_t[0]}\" IS NOT NULL"
        )

    ontology_path = "/data/ontology_nodes.parquet"
    if ontology_queries:
        union_q = " UNION ALL ".join(ontology_queries)
        con.execute(f"""
            COPY ({union_q})
            TO '{ontology_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        onto_meta = pq_local.read_metadata(ontology_path)
        onto_mb = os.path.getsize(ontology_path) / 1e6
        text_time = time.time() - t1
        print(f"  {onto_meta.num_rows:,} ontology nodes in {text_time:.1f}s ({onto_mb:.1f} MB)")
        print(f"  From {len(ontology_queries)} dictionary tables")
    else:
        onto_meta = type("M", (), {"num_rows": 0})()
        onto_mb = 0.0
        text_time = 0.0
        print("  No ontology tables found")

    con.close()
    data_vol.commit()

    return {
        "n_edges": meta.num_rows,
        "edge_file_mb": file_mb,
        "edge_time_s": edge_time,
        "n_ontology_nodes": onto_meta.num_rows,
        "ontology_file_mb": onto_mb,
        "ontology_time_s": text_time,
    }


# ---------------------------------------------------------------------------
# BGE-M3 Ontology Encoder (Modal cls) — from V5, unchanged
# ---------------------------------------------------------------------------


@scale_app.cls(
    image=gpu_image_v6,
    gpu="A100-80GB",
    timeout=3600,
    volumes=VOLUMES,
)
class BGEM3Encoder:
    """BGE-M3 encoder for ontology node texts.

    Loads model once per container (via @modal.enter), then processes
    batches of unique texts. CLS pooling. Projects 1024 -> latent_dim.

    Caches embeddings to volume — skip on restart if file exists + count matches.
    """

    @modal.enter()
    def load_model(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.device = "cuda"
        model_name = "BAAI/bge-m3"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name, torch_dtype=torch.bfloat16,
        ).to(self.device).eval()

    @modal.method()
    def embed_batch(
        self, node_ids: list[str], texts: list[str], max_length: int = 128
    ) -> list[tuple[str, list[float]]]:
        """Encode a batch of texts using BGE-M3 CLS pooling.

        Returns list of (node_id, embedding_1024d) pairs.
        """
        import torch

        results = []
        bs = 64
        for i in range(0, len(texts), bs):
            batch_ids = node_ids[i:i + bs]
            batch_texts = texts[i:i + bs]

            tokens = self.tokenizer(
                batch_texts, return_tensors="pt",
                padding=True, truncation=True, max_length=max_length,
            ).to(self.device)

            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = self.model(
                    input_ids=tokens["input_ids"],
                    attention_mask=tokens["attention_mask"],
                )

            # CLS pooling (BGE-M3 is bidirectional — CLS token is index 0)
            cls_emb = outputs.last_hidden_state[:, 0, :]  # (B, 1024)

            for nid, emb in zip(batch_ids, cls_emb.cpu().float().tolist()):
                results.append((nid, emb))

        return results


# ---------------------------------------------------------------------------
# Neural network modules (defined inside factory for Modal serialization)
# ---------------------------------------------------------------------------


def _build_nn_modules() -> dict[str, Any]:
    """Returns all nn.Module classes for V6. Called inside GPU context."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # ------------------------------------------------------------------
    # ContinuousTemporalEncoder — from V4/V5, unchanged
    # ------------------------------------------------------------------

    class ContinuousTemporalEncoder(nn.Module):
        """Continuous Temporal Positional Encoding phi(delta_t).

        Learnable harmonic encoder maps continuous time deltas to dense vectors.
        Adapts to the specific temporal scales of hospital data (hours/days/months).
        """

        def __init__(self, time_dim: int):
            super().__init__()
            assert time_dim % 2 == 0, "time_dim must be even"
            self.time_dim = time_dim
            self.omega = nn.Parameter(torch.empty(1, time_dim // 2))
            self.phi = nn.Parameter(torch.empty(1, time_dim // 2))
            nn.init.xavier_uniform_(self.omega)
            nn.init.zeros_(self.phi)

        def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
            delta_t = delta_t.view(-1, 1).float()
            delta_t_days = delta_t / 86400.0
            phase = delta_t_days * self.omega + self.phi
            return torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)

    # ------------------------------------------------------------------
    # NumericEncoder — NEW: xVal-style Fourier for continuous values
    # ------------------------------------------------------------------

    class NumericEncoder(nn.Module):
        """xVal-style learnable Fourier encoding for continuous numeric values.

        Maps scalar values (billing amounts, quantities, lab results) to dense
        vectors using learnable frequency bases. Log-scale normalization handles
        the massive range (R$1 to R$48M) without explosion.

        Architecture:
            x -> log1p(|x|)*sign(x) -> phases = x_norm * freqs -> sin/cos -> proj -> out

        For edges without numeric values (value=0.0), the encoder produces a
        near-zero vector since log1p(0)=0.
        """

        def __init__(self, n_frequencies: int = 16, out_dim: int = 32):
            super().__init__()
            self.n_frequencies = n_frequencies
            self.out_dim = out_dim
            self.freqs = nn.Parameter(torch.randn(n_frequencies))
            self.proj = nn.Linear(n_frequencies * 2, out_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Encode scalar values to dense vectors.

            Args:
                x: (E,) or (E, 1) scalar values

            Returns:
                (E, out_dim) encoded numeric features
            """
            x = x.view(-1).float()
            # Log-scale normalization: handles R$1 to R$48M range
            x_norm = torch.log1p(x.abs()) * x.sign()
            # Fourier phases with learnable frequencies
            phases = x_norm.unsqueeze(-1) * self.freqs  # (E, n_freq)
            fourier = torch.cat([phases.sin(), phases.cos()], dim=-1)  # (E, n_freq*2)
            return self.proj(fourier)  # (E, out_dim)

    # ------------------------------------------------------------------
    # GraphNorm (lightweight, per-graph normalization) — from V5
    # ------------------------------------------------------------------

    class GraphNorm(nn.Module):
        """Graph Normalization (Cai et al. 2021).

        Normalizes node features per subgraph with a learnable mean shift.
        """

        def __init__(self, dim: int):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.bias = nn.Parameter(torch.zeros(dim))
            self.mean_scale = nn.Parameter(torch.ones(1))

        def forward(self, x: torch.Tensor, batch: torch.Tensor | None = None) -> torch.Tensor:
            if batch is None:
                # Single graph — treat all nodes as one graph
                mean = x.mean(dim=0, keepdim=True)
                x = x - self.mean_scale * mean
                var = x.pow(2).mean(dim=0, keepdim=True)
                x = x / (var + 1e-6).sqrt()
                return x * self.weight + self.bias

            # Per-graph normalization
            from torch_geometric.utils import scatter
            graph_mean = scatter(x, batch, dim=0, reduce="mean")
            node_mean = graph_mean[batch]
            x = x - self.mean_scale * node_mean

            graph_var = scatter(x.pow(2), batch, dim=0, reduce="mean")
            node_var = graph_var[batch]
            x = x / (node_var + 1e-6).sqrt()

            return x * self.weight + self.bias

    # ------------------------------------------------------------------
    # GatedGCNLayer — edge-gated local MPNN, from V5
    # ------------------------------------------------------------------

    class GatedGCNLayer(nn.Module):
        """Edge-Gated Graph Convolutional Layer (Bresson & Laurent 2017).

        h_i' = h_i + ReLU(BN(A*h_i + sum_j (sigma_ij * B*h_j)))
        sigma_ij = sigmoid(C*h_i + D*h_j + E*e_ij)
        e_ij' = e_ij + ReLU(BN(C*h_i + D*h_j + E*e_ij))
        """

        def __init__(self, dim: int, edge_dim: int, use_bn: bool = True, dropout: float = 0.1):
            super().__init__()
            self.A = nn.Linear(dim, dim, bias=False)
            self.B = nn.Linear(dim, dim, bias=False)
            self.C = nn.Linear(dim, dim, bias=False)
            self.D = nn.Linear(dim, dim, bias=False)
            self.E = nn.Linear(edge_dim, dim, bias=False)

            self.bn_node = nn.BatchNorm1d(dim) if use_bn else nn.Identity()
            self.edge_proj = nn.Linear(dim, edge_dim, bias=False) if dim != edge_dim else nn.Identity()
            self.bn_edge = nn.BatchNorm1d(edge_dim) if use_bn else nn.Identity()

            self.dropout = nn.Dropout(dropout)

        def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_attr: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            src, dst = edge_index[0], edge_index[1]

            Ah = self.A(x)
            Bh = self.B(x)
            Ch = self.C(x)
            Dh = self.D(x)

            Ee = self.E(edge_attr)
            gate_input = Ch[src] + Dh[dst] + Ee
            sigma = torch.sigmoid(gate_input)

            messages = sigma * Bh[src]

            from torch_geometric.utils import scatter
            gate_sum = scatter(sigma, dst, dim=0, dim_size=x.size(0), reduce="sum").clamp(min=1e-6)
            msg_sum = scatter(messages, dst, dim=0, dim_size=x.size(0), reduce="sum")

            agg = msg_sum / gate_sum

            h_new = x + F.relu(self.bn_node(Ah + agg))
            h_new = self.dropout(h_new)

            e_new = edge_attr + F.relu(self.bn_edge(self.edge_proj(gate_input)))
            e_new = self.dropout(e_new)

            return h_new, e_new

    # ------------------------------------------------------------------
    # GPSLayer — GatedGCN + global attention + FFN, from V5
    # ------------------------------------------------------------------

    class GPSLayer(nn.Module):
        """Graph-GPS Layer (Rampasek et al. 2022).

        Combines:
          1. GatedGCN for local message passing
          2. nn.MultiheadAttention for global attention within subgraph
          3. FFN (2-layer MLP with GELU)
          4. GraphNorm + residual connections
        """

        def __init__(
            self,
            dim: int,
            edge_dim: int,
            heads: int = 8,
            ffn_mult: int = 4,
            dropout: float = 0.1,
            use_bn: bool = True,
        ):
            super().__init__()
            self.local_norm = GraphNorm(dim)
            self.local_mpnn = GatedGCNLayer(dim, edge_dim, use_bn=use_bn, dropout=dropout)

            self.global_norm = GraphNorm(dim)
            self.global_attn = nn.MultiheadAttention(
                embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True,
            )

            self.ffn_norm = GraphNorm(dim)
            self.ffn = nn.Sequential(
                nn.Linear(dim, dim * ffn_mult),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim * ffn_mult, dim),
                nn.Dropout(dropout),
            )

            self.dropout = nn.Dropout(dropout)

        def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_attr: torch.Tensor,
            batch: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # 1. Local MPNN (GatedGCN) with pre-norm + residual
            h_local = self.local_norm(x, batch)
            h_local, edge_attr_new = self.local_mpnn(h_local, edge_index, edge_attr)
            x = x + self.dropout(h_local)

            # 2. Global attention within each subgraph
            h_global = self.global_norm(x, batch)

            if batch is not None:
                from torch_geometric.utils import unbatch
                graphs = unbatch(h_global, batch)
                attn_out_parts = []
                for g in graphs:
                    g_unsq = g.unsqueeze(0)
                    ao, _ = self.global_attn(g_unsq, g_unsq, g_unsq)
                    attn_out_parts.append(ao.squeeze(0))
                attn_out = torch.cat(attn_out_parts, dim=0)
            else:
                h_unsq = h_global.unsqueeze(0)
                attn_out, _ = self.global_attn(h_unsq, h_unsq, h_unsq)
                attn_out = attn_out.squeeze(0)

            x = x + self.dropout(attn_out)

            # 3. FFN with pre-norm + residual
            h_ffn = self.ffn_norm(x, batch)
            h_ffn = self.ffn(h_ffn)
            x = x + h_ffn

            return x, edge_attr_new

    # ------------------------------------------------------------------
    # TGNMemory — per-node GRU memory on CPU, from V5
    # ------------------------------------------------------------------

    class TGNMemory(nn.Module):
        """Temporal Graph Network Memory (Rossi et al. 2020).

        Per-node GRU memory stored in CPU RAM (~9 GB for 35M nodes x 64-dim).
        Gather/scatter per batch: move only active nodes' memory to GPU.
        """

        def __init__(self, num_nodes: int, mem_dim: int, msg_dim: int, edge_feat_dim: int):
            super().__init__()
            self.num_nodes = num_nodes
            self.mem_dim = mem_dim
            self.msg_dim = msg_dim

            self.memory: torch.Tensor
            self.last_update: torch.Tensor
            self.register_buffer(
                "memory", torch.zeros(num_nodes, mem_dim), persistent=False
            )
            self.register_buffer(
                "last_update", torch.zeros(num_nodes, dtype=torch.int64), persistent=False
            )

            # Message function: cat[mem_src, mem_dst, edge_feat] -> msg_dim
            self.msg_fn = nn.Linear(2 * mem_dim + edge_feat_dim, msg_dim)
            self.gru = nn.GRUCell(msg_dim, mem_dim)

        def get_memory(self, node_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
            """Gather memory for a batch of nodes. Returns (B, mem_dim) on device."""
            memory = cast(torch.Tensor, self.memory)
            return memory[node_ids.cpu()].to(device)

        def compute_messages(
            self,
            src_ids: torch.Tensor,
            dst_ids: torch.Tensor,
            edge_feat: torch.Tensor,
            device: torch.device,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Compute messages for edges in batch."""
            memory = cast(torch.Tensor, self.memory)
            mem_src = memory[src_ids.cpu()].to(device)
            mem_dst = memory[dst_ids.cpu()].to(device)
            msg_input = torch.cat([mem_src, mem_dst, edge_feat], dim=-1)
            messages = self.msg_fn(msg_input)

            all_nodes = torch.cat([src_ids, dst_ids]).cpu()
            all_msgs = torch.cat([messages, messages], dim=0)

            unique_nodes, inverse = torch.unique(all_nodes, return_inverse=True)

            agg = torch.zeros(unique_nodes.size(0), self.msg_dim, device=device)
            # FIX: scatter_reduce_ to safely aggregate concurrent messages (avoid nondeterministic overwrite)
            agg.scatter_reduce_(
                0,
                inverse.to(device).unsqueeze(-1).expand(-1, self.msg_dim),
                all_msgs,
                reduce="mean",
                include_self=False,
            )

            return unique_nodes, agg, messages

        def update_memory(
            self,
            unique_nodes: torch.Tensor,
            agg_messages: torch.Tensor,
            timestamps: torch.Tensor | None = None,
        ) -> None:
            """Update memory for nodes in this batch using GRU."""
            device = agg_messages.device
            memory = cast(torch.Tensor, self.memory)
            last_update = cast(torch.Tensor, self.last_update)
            old_mem = memory[unique_nodes.cpu()].to(device)
            new_mem = self.gru(agg_messages, old_mem)

            memory[unique_nodes.cpu()] = new_mem.detach().cpu()
            if timestamps is not None:
                last_update[unique_nodes.cpu()] = timestamps.cpu()

        def detach_memory(self) -> None:
            """Detach memory from computation graph (call between epochs)."""
            memory = cast(torch.Tensor, self.memory)
            memory.detach_()

        def reset_memory(self) -> None:
            """Zero out all memory (for fresh start)."""
            memory = cast(torch.Tensor, self.memory)
            last_update = cast(torch.Tensor, self.last_update)
            memory.zero_()
            last_update.zero_()

        def state_for_checkpoint(self) -> dict[str, torch.Tensor]:
            memory = cast(torch.Tensor, self.memory)
            last_update = cast(torch.Tensor, self.last_update)
            return {
                "memory": memory.clone(),
                "last_update": last_update.clone(),
            }

        def load_checkpoint_state(self, state: dict[str, torch.Tensor]) -> None:
            memory = cast(torch.Tensor, self.memory)
            last_update = cast(torch.Tensor, self.last_update)
            memory.copy_(state["memory"])
            last_update.copy_(state["last_update"])

    # ------------------------------------------------------------------
    # TemporalGPSEncoder — V6: edge_feat_dim=112 (pred 64 + time 16 + numeric 32)
    # ------------------------------------------------------------------

    class TemporalGPSEncoder(nn.Module):
        """GraphGPS encoder with temporal + numeric edge features.

        V6 changes from V5:
          - Edge features include numeric encoding: cat(pred_emb(64), phi(dt)(16), numeric(32)) = 112
          - latent_dim = 128 (up from 64)
          - No LapPE (removed for simplification)
        """

        def __init__(self, cfg: V6Config, num_predicates: int):
            super().__init__()
            self.cfg = cfg
            dim = cfg.latent_dim

            # Predicate embedding — 64-dim for edge features (not full latent_dim)
            self.pred_emb = nn.Embedding(num_predicates, cfg.pred_emb_dim)

            # Temporal encoder
            self.time_encoder = ContinuousTemporalEncoder(cfg.time_dim)

            # Numeric encoder (xVal Fourier)
            self.numeric_encoder = NumericEncoder(
                n_frequencies=cfg.numeric_n_frequencies,
                out_dim=cfg.numeric_dim,
            )

            # Edge feature projection:
            # cat(pred_emb(pred_emb_dim), phi(dt)(time_dim), numeric(numeric_dim)) -> edge_feat_dim
            raw_edge_dim = cfg.pred_emb_dim + cfg.time_dim + cfg.numeric_dim
            self.edge_proj = nn.Linear(raw_edge_dim, cfg.edge_feat_dim)

            # RWSE projection (optional positional encoding)
            self.rwse_proj = nn.Linear(cfg.rwse_dim, dim) if cfg.rwse_dim > 0 else None

            # GPS layers
            self.gps_layers = nn.ModuleList([
                GPSLayer(
                    dim=dim,
                    edge_dim=cfg.edge_feat_dim,
                    heads=cfg.gps_heads,
                    ffn_mult=cfg.gps_ffn_mult,
                    dropout=cfg.gps_dropout,
                    use_bn=cfg.gcn_bn,
                )
                for _ in range(cfg.gps_layers)
            ])

            # Final norm
            self.final_norm = nn.LayerNorm(dim)

        def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_attr_idx: torch.Tensor,
            node_time: torch.Tensor | None = None,
            edge_time: torch.Tensor | None = None,
            edge_numeric: torch.Tensor | None = None,
            batch: torch.Tensor | None = None,
            rwse: torch.Tensor | None = None,
        ) -> torch.Tensor:
            """Forward pass through the V6 GPS encoder.

            Args:
                x: Node features (N, dim)
                edge_index: (2, E)
                edge_attr_idx: (E,) predicate type indices
                node_time: (N,) node timestamps
                edge_time: (E,) edge timestamps
                edge_numeric: (E,) numeric values for edges
                batch: (N,) graph membership
                rwse: (N, rwse_dim) random walk structural encoding

            Returns:
                (N, dim) encoded node representations
            """
            # Predicate embedding (64-dim)
            e = self.pred_emb(edge_attr_idx)

            # Temporal encoding
            if node_time is not None and edge_time is not None:
                target_nodes = edge_index[1]
                t_target = node_time[target_nodes]
                delta_t = t_target - edge_time

                # Strict causality: mask out future edges (delta_t < 0)
                causal_mask = delta_t >= 0
                if not causal_mask.all():
                    edge_index = edge_index[:, causal_mask]
                    delta_t = delta_t[causal_mask]
                    e = e[causal_mask]
                    if edge_time is not None:
                        edge_time = edge_time[causal_mask]
                    if edge_numeric is not None:
                        edge_numeric = edge_numeric[causal_mask]

                phi_t = self.time_encoder(delta_t)  # (E, time_dim=16)
            else:
                phi_t = torch.zeros(
                    e.shape[0], self.cfg.time_dim, device=e.device, dtype=e.dtype
                )

            # Numeric encoding
            if edge_numeric is not None:
                num_e = self.numeric_encoder(edge_numeric)  # (E, numeric_dim=32)
            else:
                num_e = torch.zeros(
                    e.shape[0], self.cfg.numeric_dim, device=e.device, dtype=e.dtype
                )

            # Concatenate: pred(64) + time(16) + numeric(32) = 112
            e = torch.cat([e, phi_t, num_e], dim=-1)

            # Project edge features to edge_feat_dim
            e = self.edge_proj(e)

            # Add positional encodings to node features
            if rwse is not None and self.rwse_proj is not None:
                x = x + self.rwse_proj(rwse)

            # GPS layers
            for gps in self.gps_layers:
                x, e = gps(x, edge_index, e, batch=batch)

            return self.final_norm(x)

    # ------------------------------------------------------------------
    # DenseTemporalPredictor — NEW: replaces GraphPredictor + MultiTaskHead
    # ------------------------------------------------------------------

    class DenseTemporalPredictor(nn.Module):
        """Predicts future state from context + time query.

        Given the context representation z_context (from the online encoder)
        and a temporal query q(dt) encoding the time offset, predicts what
        the EMA target representation should look like at that future time.

        Architecture:
            input = cat(z_context, q(dt)) -> MLP -> z_pred

        The time query uses the same ContinuousTemporalEncoder as the backbone,
        allowing the predictor to generate time-specific predictions.
        """

        def __init__(self, dim: int, time_dim: int, hidden_mult: int = 2):
            super().__init__()
            self.dim = dim
            self.time_dim = time_dim

            # Time query encoder (separate from backbone temporal encoder)
            self.query_encoder = ContinuousTemporalEncoder(time_dim)

            # Predictor MLP: cat(context, time_query) -> hidden -> output
            hidden_dim = dim * hidden_mult
            self.net = nn.Sequential(
                nn.Linear(dim + time_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, dim),
            )

        def forward(
            self,
            z_context: torch.Tensor,
            delta_t: torch.Tensor,
        ) -> torch.Tensor:
            """Predict future state from context + time query.

            Args:
                z_context: (B, dim) context representation from online encoder
                delta_t: (B,) time offsets in seconds

            Returns:
                z_pred: (B, dim) predicted future state
            """
            q = self.query_encoder(delta_t)  # (B, time_dim)
            h = torch.cat([z_context, q], dim=-1)  # (B, dim + time_dim)
            return self.net(h)  # (B, dim)

    # ------------------------------------------------------------------
    # WeakSIGRegLoss — off-diagonal covariance penalty only
    # ------------------------------------------------------------------

    class WeakSIGRegLoss(nn.Module):
        """Weak SIGReg: off-diagonal covariance penalty.

        Only penalizes correlation between embedding dimensions (decorrelation).
        No variance hinge (that caused oscillation in V5).
        No invariance term (that's covered by the dense lookahead loss).

        L = sum(off_diag(cov(z))^2) / D
        """

        def __init__(self) -> None:
            super().__init__()

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            """Compute weak SIGReg loss.

            Args:
                z: (B, D) representations

            Returns:
                scalar loss
            """
            B, D = z.shape
            if B < 2:
                return torch.tensor(0.0, device=z.device, dtype=z.dtype)

            z_c = z - z.mean(dim=0)
            cov = (z_c.T @ z_c) / (B - 1)
            off_diag = cov - torch.diag(cov.diag())
            return off_diag.pow(2).sum() / D

    return {
        "ContinuousTemporalEncoder": ContinuousTemporalEncoder,
        "NumericEncoder": NumericEncoder,
        "GraphNorm": GraphNorm,
        "GatedGCNLayer": GatedGCNLayer,
        "GPSLayer": GPSLayer,
        "TGNMemory": TGNMemory,
        "TemporalGPSEncoder": TemporalGPSEncoder,
        "DenseTemporalPredictor": DenseTemporalPredictor,
        "WeakSIGRegLoss": WeakSIGRegLoss,
    }


# ---------------------------------------------------------------------------
# V6 Trainer
# ---------------------------------------------------------------------------


class V6Trainer:
    """Orchestrates V6 training: graph loading, encoding, model init, training loop.

    Key differences from V5Trainer:
        - 2-phase curriculum (foundation + temporal), not 4
        - Dense Temporal Lookahead loss (predict EMA target per future event)
        - Weak-SIGReg loss (off-diagonal covariance only)
        - No VICReg, no MultiTaskHead, no auxiliary losses
        - NumericEncoder for edge numeric values
        - 128-dim latent (up from 64), V5 epoch 1 warm-start
    """

    def __init__(
        self,
        cfg: V6Config,
        nn_modules: dict,
        parquet_path: str,
        ontology_path: str,
    ):
        import gc
        import random

        import numpy as np
        import torch

        self.cfg = cfg
        self.nn = nn_modules
        self.device = torch.device("cuda")

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        # Load graph (with numeric values)
        self._load_graph(parquet_path)

        # Build vocabulary (with caching)
        self._build_vocab_cached()

        # Encode ontology with BGE-M3 (with caching)
        self._encode_ontology_cached(ontology_path)

        # Init embeddings (with V5 warm-start)
        self._init_embeddings()

        # Build graph data object
        self._build_graph_data()

        # Build CSR cache
        self._build_csr_cache()

        # Create models
        self._create_models()

        # Create optimizer and scaler
        self._create_optimizer()

        # Training state
        self.start_epoch = 0
        self.total_steps = 0
        self.losses_log: list[dict[str, float]] = []

        # Try resume
        self._resume_checkpoint()

        gc.collect()
        torch.cuda.empty_cache()

    def _load_graph(self, parquet_path: str) -> None:
        """Load the materialized TKG from Parquet with numeric values."""
        import numpy as np
        import pyarrow as pa_mod
        import pyarrow.compute as pc_mod
        import pyarrow.parquet as pq_mod
        import torch

        hospital = self.cfg.hospital_filter
        label = f" [FILTER: {hospital}]" if hospital else ""
        print(f"Loading TKG from Parquet (V6 with numeric_value){label}...")
        t0 = time.time()

        table = pq_mod.read_table(parquet_path)

        # Single-hospital filter: keep only edges where subject OR object contains the hospital prefix
        if hospital:
            mask_s = pc_mod.match_substring(table.column("subject_id"), hospital)
            mask_o = pc_mod.match_substring(table.column("object_id"), hospital)
            table = table.filter(pc_mod.or_(mask_s, mask_o))
            print(f"  Filtered to {hospital}: {table.num_rows:,} edges (from full graph)")

        self.n_edges = table.num_rows
        print(f"  {self.n_edges:,} edges loaded in {time.time() - t0:.1f}s")

        # C++ fast path: PyArrow dictionary encoding
        print("  Dictionary encoding (C++)...")

        subj = table.column("subject_id")
        obj = table.column("object_id")
        all_nodes_chunked = pa_mod.chunked_array(subj.chunks + obj.chunks)
        unique_nodes = pc_mod.unique(all_nodes_chunked)
        self.num_nodes = len(unique_nodes)
        print(f"  {self.num_nodes:,} unique nodes (C++ dedup)")

        src_idx = pc_mod.index_in(subj, value_set=unique_nodes).to_numpy(zero_copy_only=False).astype(np.int64)
        dst_idx = pc_mod.index_in(obj, value_set=unique_nodes).to_numpy(zero_copy_only=False).astype(np.int64)

        self.node_vocab = {s: i for i, s in enumerate(unique_nodes.to_pylist())}
        self.node_list = list(self.node_vocab.keys())

        # Predicate encoding
        pred_chunked = table.column("predicate")
        pred_unique = pc_mod.unique(pa_mod.chunked_array(pred_chunked.chunks))
        self.pred_list = pred_unique.to_pylist()
        attr_idx = pc_mod.index_in(pred_chunked, value_set=pred_unique).to_numpy(zero_copy_only=False).astype(np.int64)

        # Timestamps
        time_parts = []
        for chunk in table.column("t_epoch").iterchunks():
            time_parts.append(chunk.to_numpy(zero_copy_only=False))
        time_np = np.concatenate(time_parts)
        del time_parts

        # Numeric values (V6 new column)
        has_numeric_col = "numeric_value" in table.schema.names
        if has_numeric_col:
            print("  Loading numeric_value column...")
            num_parts = []
            for chunk in table.column("numeric_value").iterchunks():
                num_parts.append(chunk.to_numpy(zero_copy_only=False))
            numeric_np = np.concatenate(num_parts).astype(np.float32)
            del num_parts
            n_nonzero = np.count_nonzero(numeric_np)
            print(f"  {n_nonzero:,} / {len(numeric_np):,} edges have nonzero numeric values")
        else:
            print("  WARNING: No numeric_value column in Parquet, using zeros")
            numeric_np = np.zeros(self.n_edges, dtype=np.float32)

        self.num_predicates = len(self.pred_list)

        self.edge_index = torch.stack([
            torch.from_numpy(src_idx).to(torch.int64),
            torch.from_numpy(dst_idx).to(torch.int64),
        ], dim=0)
        self.edge_attr = torch.from_numpy(attr_idx).to(torch.int64)
        self.edge_time = torch.from_numpy(time_np).to(torch.int64)
        self.edge_numeric = torch.from_numpy(numeric_np).to(torch.float32)

        print(f"  {self.num_nodes:,} nodes, {self.num_predicates} predicates")
        print(f"  Loaded in {time.time() - t0:.1f}s")

        del table, subj, obj, src_idx, dst_idx, attr_idx, time_np, numeric_np
        import gc
        gc.collect()

    def _build_vocab_cached(self) -> None:
        """Build node vocabulary, caching to volume."""
        import os

        cache_path = os.path.join(self.cfg.artifact_dir, "node_vocab.json")
        os.makedirs(self.cfg.artifact_dir, exist_ok=True)

        if os.path.exists(cache_path):
            with open(cache_path) as f:
                cached = json.load(f)
            if cached.get("num_nodes") == self.num_nodes:
                print(f"  Node vocab cache hit ({self.num_nodes:,} nodes)")
                return

        print(f"  Caching node vocab ({self.num_nodes:,} nodes)...")
        with open(cache_path, "w") as f:
            json.dump({"num_nodes": self.num_nodes, "num_predicates": self.num_predicates}, f)
        jepa_cache.commit()

    def _encode_ontology_cached(self, ontology_path: str) -> None:
        """Encode ontology texts with BGE-M3, with caching to volume."""
        import os

        import pyarrow.parquet as pq_mod
        import torch

        cache_path = os.path.join(self.cfg.artifact_dir, "ontology_bge_embeddings.pt")
        cache_meta_path = os.path.join(self.cfg.artifact_dir, "ontology_bge_meta.json")

        # Load ontology texts
        self.onto_indices = []
        self.onto_texts_matched = []
        ontology_node_ids = []
        ontology_texts = []

        if os.path.exists(ontology_path):
            onto_table = pq_mod.read_table(ontology_path)
            ontology_node_ids = onto_table.column("node_id").to_pylist()
            ontology_texts = onto_table.column("text_payload").to_pylist()
            del onto_table
        n_onto = len(ontology_node_ids)

        # Match ontology nodes to graph
        for nid, txt in zip(ontology_node_ids, ontology_texts):
            if nid in self.node_vocab:
                self.onto_indices.append(self.node_vocab[nid])
                self.onto_texts_matched.append(txt)

        n_matched = len(self.onto_indices)
        print(f"  {n_onto:,} ontology nodes in parquet, {n_matched:,} matched in graph")

        # Check cache
        if os.path.exists(cache_path) and os.path.exists(cache_meta_path):
            with open(cache_meta_path) as f:
                meta = json.load(f)
            if meta.get("n_matched") == n_matched:
                print(f"  Ontology BGE cache hit ({n_matched:,} embeddings)")
                self.onto_raw_embeddings = torch.load(cache_path, weights_only=True)
                return

        if n_matched == 0:
            self.onto_raw_embeddings = None
            return

        # Encode with BGE-M3 locally
        print(f"  Encoding {n_matched:,} ontology texts with BGE-M3...")
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.cfg.bge_model)
        model = AutoModel.from_pretrained(
            self.cfg.bge_model, torch_dtype=torch.bfloat16,
        ).to(self.device).eval()

        all_onto_embs = []
        bs = self.cfg.bge_batch_size
        t0 = time.time()
        for i in range(0, n_matched, bs):
            batch_texts = self.onto_texts_matched[i:i + bs]
            tokens = tokenizer(
                batch_texts, return_tensors="pt",
                padding=True, truncation=True, max_length=self.cfg.bge_max_length,
            ).to(self.device)

            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(**tokens)
                cls_emb = out.last_hidden_state[:, 0, :]
                all_onto_embs.append(cls_emb.cpu().float())

            if (i // bs) % 50 == 0 and i > 0:
                print(f"    {i:,}/{n_matched:,} encoded")

        self.onto_raw_embeddings = torch.cat(all_onto_embs, dim=0)
        print(f"  Ontology encoded: {self.onto_raw_embeddings.shape} in {time.time() - t0:.1f}s")

        # Cache to volume
        os.makedirs(self.cfg.artifact_dir, exist_ok=True)
        torch.save(self.onto_raw_embeddings, cache_path)
        with open(cache_meta_path, "w") as f:
            json.dump({"n_matched": n_matched, "bge_model": self.cfg.bge_model}, f)
        jepa_cache.commit()
        print(f"  Ontology embeddings cached to {cache_path}")

        # Free BGE-M3
        del model, tokenizer
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        print("  BGE-M3 freed from VRAM")

    def _init_embeddings(self) -> None:
        """Initialize node embedding table with V5 epoch 1 warm-start.

        V6 warm-start strategy:
            v5_emb (35.2M, 64) -> pad with zeros -> (35.2M, 128)
            Second 64 dims are "temporal slots" that TGN memory will fill.
        """
        import os

        import torch
        import torch.nn as nn

        dim = self.cfg.latent_dim  # 128

        # Create embedding table
        self.node_emb = nn.Embedding(self.num_nodes, dim).to(self.device)
        nn.init.xavier_uniform_(self.node_emb.weight)

        # V5 epoch 1 warm-start: pad 64-dim -> 128-dim with zeros
        v5_emb_path = os.path.join(self.cfg.v5_artifact_dir, "node_emb_epoch_1.pt")
        v5_emb_alt = os.path.join(self.cfg.v5_artifact_dir, "node_embeddings.pt")

        warm_path = v5_emb_path if os.path.exists(v5_emb_path) else v5_emb_alt

        if self.cfg.hospital_filter:
            print(f"  Single-hospital mode — skipping V5 warm-start (different node vocab)")
        elif os.path.exists(warm_path):
            print(f"  Loading V5 embeddings for warm-start from {warm_path}...")
            v5_state = torch.load(warm_path, weights_only=True)

            if isinstance(v5_state, dict) and "weight" in v5_state:
                v5_emb = v5_state["weight"]
            elif isinstance(v5_state, torch.Tensor):
                v5_emb = v5_state
            else:
                v5_emb = None
                print("  Warning: V5 checkpoint format not recognized, skipping warm-start")

            if v5_emb is not None:
                v5_dim = v5_emb.shape[1]
                if v5_emb.shape[0] == self.num_nodes:
                    if v5_dim < dim:
                        # Pad with zeros (not random — preserve V5 spatial structure)
                        # Second half are "temporal slots" TGN will fill
                        print(f"  V5 warm-start: padding {v5_dim} -> {dim} with zeros")
                        frozen_set = set(self.onto_indices)
                        instance_mask = torch.ones(self.num_nodes, dtype=torch.bool)
                        if frozen_set:
                            for idx in self.onto_indices:
                                instance_mask[idx] = False

                        with torch.no_grad():
                            chunk_size = 500_000
                            for start in range(0, self.num_nodes, chunk_size):
                                end = min(start + chunk_size, self.num_nodes)
                                chunk_mask = instance_mask[start:end]
                                if chunk_mask.any():
                                    v5_chunk = v5_emb[start:end][chunk_mask].to(self.device)
                                    # Zero-pad: cat[v5_emb, zeros]
                                    padded = torch.cat([
                                        v5_chunk,
                                        torch.zeros(v5_chunk.shape[0], dim - v5_dim, device=self.device),
                                    ], dim=1)
                                    indices = torch.arange(start, end, device=self.device)[chunk_mask]
                                    self.node_emb.weight[indices] = padded.float()

                        n_warm = instance_mask.sum().item()
                        print(f"  V5 warm-start: {n_warm:,} instance nodes padded {v5_dim} -> {dim}")
                    elif v5_dim == dim:
                        # Same dim — direct copy for instance nodes
                        print(f"  V5 warm-start: same dim ({v5_dim}), direct copy")
                        frozen_set = set(self.onto_indices)
                        instance_mask = torch.ones(self.num_nodes, dtype=torch.bool)
                        for idx in self.onto_indices:
                            instance_mask[idx] = False
                        with torch.no_grad():
                            chunk_size = 500_000
                            for start in range(0, self.num_nodes, chunk_size):
                                end = min(start + chunk_size, self.num_nodes)
                                chunk_mask = instance_mask[start:end]
                                if chunk_mask.any():
                                    v5_chunk = v5_emb[start:end][chunk_mask].to(self.device)
                                    indices = torch.arange(start, end, device=self.device)[chunk_mask]
                                    self.node_emb.weight[indices] = v5_chunk.float()
                        n_warm = instance_mask.sum().item()
                        print(f"  V5 warm-start: {n_warm:,} instance nodes copied")
                    else:
                        print(f"  V5 dim ({v5_dim}) > V6 dim ({dim}), skipping warm-start")
                elif v5_emb.shape[0] != self.num_nodes:
                    print(f"  V5 node count mismatch ({v5_emb.shape[0]} vs {self.num_nodes}), skipping warm-start")
                del v5_emb
        else:
            print("  No V5 embeddings found, using random init")

        # Ontology projection: BGE 1024 -> latent_dim with Xavier init
        self.onto_projection = nn.Linear(self.cfg.bge_hidden_dim, dim, bias=False).to(self.device)
        nn.init.xavier_uniform_(self.onto_projection.weight)

        # Overwrite ontology node embeddings with projected BGE vectors
        self.n_frozen = 0
        if self.onto_raw_embeddings is not None and len(self.onto_indices) > 0:
            onto_idx_tensor = torch.tensor(self.onto_indices, dtype=torch.long)
            with torch.no_grad():
                projected = self.onto_projection(self.onto_raw_embeddings.to(self.device))
                self.node_emb.weight[onto_idx_tensor] = projected.float()
            self.n_frozen = len(self.onto_indices)
            print(f"  {self.n_frozen:,} ontology nodes -> frozen BGE-M3 embeddings (1024 -> {dim})")
            print(f"  {self.num_nodes - self.n_frozen:,} instance nodes -> learnable")
            del self.onto_raw_embeddings

        # Create frozen mask
        self.frozen_mask = torch.zeros(self.num_nodes, dtype=torch.bool, device=self.device)
        if self.onto_indices:
            self.frozen_mask[torch.tensor(self.onto_indices, dtype=torch.long, device=self.device)] = True

        # Cache frozen embeddings for hard-reset after each optimizer step
        # (AdamW weight decay + momentum would otherwise drift frozen nodes)
        self.frozen_embs_cache = self.node_emb.weight[self.frozen_mask].detach().clone()

    def _build_graph_data(self) -> None:
        """Build PyG Data object with numeric edge attribute."""
        import torch
        from torch_geometric.data import Data

        # Compute node_time: max edge timestamp per node
        print("  Computing node timestamps (max edge time per node)...")
        self.node_time = torch.zeros(self.num_nodes, dtype=torch.int64)
        src_nodes = self.edge_index[0]
        dst_nodes = self.edge_index[1]
        self.node_time.scatter_reduce_(0, src_nodes, self.edge_time, reduce="amax", include_self=False)
        self.node_time.scatter_reduce_(0, dst_nodes, self.edge_time, reduce="amax", include_self=False)
        print(f"  Node time range: {self.node_time.min().item()} -- {self.node_time.max().item()}")

        self.graph_data = Data(
            x=torch.arange(self.num_nodes),
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            edge_time=self.edge_time,
            edge_numeric=self.edge_numeric,  # V6 new
            node_time=self.node_time,
            num_nodes=self.num_nodes,
        )

        # Nodes with edges (for training)
        self.unique_src = torch.unique(self.edge_index[0])
        print(f"  Nodes with outgoing edges: {self.unique_src.shape[0]:,}")

    def _build_csr_cache(self) -> None:
        """Pre-build CSR adjacency and cache to volume."""
        import os

        import torch

        csr_cache_path = os.path.join(self.cfg.artifact_dir, "csr_cache_v6.pt")
        if os.path.exists(csr_cache_path):
            print("  CSR cache exists, PyG will use optimized path")
            return

        print("  Pre-building CSR adjacency for NeighborLoader...")
        t0 = time.time()
        from torch_geometric.loader import NeighborLoader
        dummy_loader = NeighborLoader(
            self.graph_data,
            num_neighbors=[5],
            batch_size=32,
            input_nodes=self.unique_src[:32],
            num_workers=0,
        )
        _ = next(iter(dummy_loader))
        del dummy_loader

        torch.save({"built": True, "num_nodes": self.num_nodes}, csr_cache_path)
        jepa_cache.commit()
        print(f"  CSR pre-built in {time.time() - t0:.1f}s")

    def _create_models(self) -> None:
        """Instantiate all V6 neural network models."""
        import copy

        cfg = self.cfg
        M = self.nn

        # Online encoder (V6: edge_feat_dim=112 with numeric)
        self.online_encoder = M["TemporalGPSEncoder"](cfg, self.num_predicates).to(self.device)

        # Target encoder (EMA, no grad)
        self.target_encoder = copy.deepcopy(self.online_encoder).to(self.device)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Dense Temporal Predictor (replaces GraphPredictor + MultiTaskHead)
        self.predictor = M["DenseTemporalPredictor"](
            dim=cfg.latent_dim,
            time_dim=cfg.time_dim,
            hidden_mult=cfg.predictor_hidden_mult,
        ).to(self.device)

        # Weak-SIGReg loss (replaces VICReg)
        self.sigreg = M["WeakSIGRegLoss"]().to(self.device)

        # TGN Memory (on CPU)
        self.tgn = M["TGNMemory"](
            num_nodes=self.num_nodes,
            mem_dim=cfg.tgn_dim,
            msg_dim=cfg.tgn_msg_dim,
            edge_feat_dim=cfg.edge_feat_dim,
        )
        # Keep TGN parameters on GPU for gradient computation
        self.tgn.msg_fn = self.tgn.msg_fn.to(self.device)
        self.tgn.gru = self.tgn.gru.to(self.device)

        # Parameter counts
        n_encoder = sum(p.numel() for p in self.online_encoder.parameters())
        n_predictor = sum(p.numel() for p in self.predictor.parameters())
        n_emb = self.node_emb.weight.numel()
        n_tgn = sum(p.numel() for p in self.tgn.parameters())
        n_onto_proj = sum(p.numel() for p in self.onto_projection.parameters())

        print(f"\n{'=' * 50}")
        print("V6 Dense Temporal JEPA — Model Summary")
        print(f"{'=' * 50}")
        print(f"  Online encoder:     {n_encoder:,} params")
        print(f"  Predictor:          {n_predictor:,} params")
        print(f"  Node embeddings:    {n_emb:,} params ({n_emb * 4 / 1e9:.2f} GB)")
        print(f"  Onto projection:    {n_onto_proj:,} params")
        print(f"  TGN:                {n_tgn:,} params")
        print(f"  TGN memory:         {self.num_nodes * cfg.tgn_dim * 4 / 1e9:.2f} GB (CPU)")
        total_trainable = n_encoder + n_predictor + n_emb + n_tgn + n_onto_proj
        print(f"  Total trainable:    {total_trainable:,} params")
        print(f"  Loss:               L_dense_lookahead + {cfg.sigreg_lambda} * L_weak_sigreg")
        print("  Curriculum:         2 phases (foundation, temporal)")
        print(
            "  Edge features:      "
            f"pred({cfg.pred_emb_dim}) + time({cfg.time_dim}) + "
            f"numeric({cfg.numeric_dim}) = {cfg.edge_feat_dim}"
        )

    def _create_optimizer(self) -> None:
        """Create split optimizers and AMP scaler.

        The embedding table is too large for AdamW state at 128 dims:
        35.2M x 128 parameters implies ~36 GB of optimizer buffers for
        exp_avg and exp_avg_sq alone. Keep `node_emb` on plain SGD and
        use AdamW only for the smaller neural modules.
        """
        import torch

        cfg = self.cfg
        self.emb_trainable = list(self.node_emb.parameters())
        self.net_trainable = (
            list(self.online_encoder.parameters())
            + list(self.predictor.parameters())
            + list(self.onto_projection.parameters())
            + list(self.tgn.msg_fn.parameters())
            + list(self.tgn.gru.parameters())
        )
        self.trainable = self.emb_trainable + self.net_trainable

        self.optimizer_emb = torch.optim.SGD(
            self.emb_trainable,
            lr=cfg.lr * cfg.emb_lr_mult,
            momentum=cfg.emb_momentum,
            weight_decay=cfg.emb_weight_decay,
        )
        self.optimizer_net = torch.optim.AdamW(
            self.net_trainable,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp)

    def _get_phase(self, epoch: int) -> CurriculumPhase:
        """Return the current CurriculumPhase for the given epoch.

        In single-hospital mode (hospital_filter set), overrides:
        - TGN enabled from tgn_from_epoch (default 0 for fast test)
        - node_sample_frac = 1.0 (small graph, no sampling needed)
        """
        for phase in CURRICULUM_PHASES_V6:
            if phase.epoch_start <= epoch < phase.epoch_end:
                p = phase
                break
        else:
            p = CURRICULUM_PHASES_V6[-1]

        # Single-hospital override: TGN from epoch 0, 100% sampling
        if self.cfg.hospital_filter:
            p = CurriculumPhase(
                name=p.name,
                epoch_start=p.epoch_start,
                epoch_end=p.epoch_end,
                num_neighbors=p.num_neighbors,
                tgn_enabled=(epoch >= self.cfg.tgn_from_epoch),
                node_sample_frac=1.0,
            )
        return p

    def _create_loader(self, phase: CurriculumPhase) -> tuple[Any, Any]:
        """Create NeighborLoader with curriculum-appropriate hops."""
        import torch
        from torch_geometric.loader import NeighborLoader

        input_nodes = self.unique_src
        if phase.node_sample_frac < 1.0:
            n_sample = int(len(input_nodes) * phase.node_sample_frac)
            perm = torch.randperm(len(input_nodes))[:n_sample]
            input_nodes = input_nodes[perm]
            print(f"  Sampling {n_sample:,} / {len(self.unique_src):,} nodes for phase '{phase.name}'")

        loader = NeighborLoader(
            self.graph_data,
            num_neighbors=phase.num_neighbors,
            batch_size=self.cfg.batch_size,
            input_nodes=input_nodes,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            persistent_workers=True,
            prefetch_factor=self.cfg.prefetch_factor,
        )
        return loader, input_nodes

    def _lr_schedule(self, step: int, total_steps: int) -> float:
        """Warmup + cosine annealing LR schedule."""
        warmup_steps = int(total_steps * self.cfg.warmup_frac)
        warmup_steps = max(warmup_steps, 100)

        if step < warmup_steps:
            return 0.1 + 0.9 * (step / warmup_steps)
        else:
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return max(0.01, 0.5 * (1 + math.cos(math.pi * progress)))

    def _compute_dense_lookahead_loss(
        self,
        ctx_repr: torch.Tensor,
        ema_target: torch.Tensor,
        batch_node_time: torch.Tensor,
        B: int,
    ) -> torch.Tensor:
        """Compute Dense Temporal Lookahead loss.

        For each seed node i, predicts the EMA representation of future
        events at specific time offsets. The predictor sees the context
        (online encoder output) and must match the EMA target for each
        future time step.

        Key difference from V5 lookahead: explicitly encodes delta-t as a
        positional query, generating time-specific predictions.

        Args:
            ctx_repr: (N_batch, dim) online encoder output
            ema_target: (N_batch, dim) EMA target output (detached)
            batch_node_time: (N_batch,) node timestamps
            B: number of seed nodes

        Returns:
            dense_loss: scalar
        """
        import torch
        import torch.nn.functional as F

        cfg = self.cfg
        device = self.device

        ctx_B = ctx_repr[:B]  # (B, dim) — seed node context
        ema_B = ema_target[:B]  # (B, dim) — seed node EMA targets

        if B <= cfg.lookahead_steps:
            # Not enough nodes for multi-step prediction — single step
            dummy_dt = torch.ones(B, device=device) * 86400.0  # 1 day
            z_pred = self.predictor(ctx_B, dummy_dt)
            return F.mse_loss(z_pred, ema_B)

        total_loss = torch.tensor(0.0, device=device)
        n_terms = 0

        # Use node timestamps to compute realistic delta-t offsets
        # FIX: NeighborLoader uses shuffle=True — seed nodes are in random order.
        # Must sort chronologically before shifting, otherwise we predict random
        # patients' futures from unrelated patients' contexts.
        times_B = batch_node_time[:B].float()
        sorted_idx = torch.argsort(times_B)
        ctx_B = ctx_B[sorted_idx]
        ema_B = ema_B[sorted_idx]
        times_B = times_B[sorted_idx]

        for k in range(cfg.lookahead_steps):
            # Shift target by k positions (now correctly time-ordered)
            shift = k + 1
            if shift >= B:
                break

            # Target: EMA embeddings shifted by k positions
            shifted_target = ema_B[shift:]  # (B - shift, dim)
            context = ctx_B[:B - shift]  # (B - shift, dim)

            # Delta-t: time difference between context and shifted target
            dt = (times_B[shift:] - times_B[:B - shift]).clamp(min=1.0)  # (B - shift,)

            # Predict
            z_pred = self.predictor(context, dt)  # (B - shift, dim)

            # Continuous time decay: 7-day half-life
            # Each node gets its own weight based on actual chronological gap,
            # not array index. ICU events 10 min apart get ~1.0 weight;
            # outpatient visits 6 months apart get ~0.0 weight.
            half_life_sec = 7.0 * 86400.0  # 7 days
            decay_rate = math.log(2) / half_life_sec
            weights = torch.exp(-decay_rate * dt)  # (B - shift,)

            # Per-node weighted MSE (unreduced then weighted)
            step_loss = F.mse_loss(z_pred, shifted_target.detach(), reduction='none').mean(dim=-1)
            weighted_loss = (weights * step_loss).mean()
            total_loss = total_loss + weighted_loss
            n_terms += 1

        if n_terms > 0:
            total_loss = total_loss / n_terms

        return total_loss

    def _train_step(
        self,
        batch: Any,
        phase: CurriculumPhase,
        total_steps: int,
        total_expected_steps: int,
    ) -> dict[str, float]:
        """Single training step: forward + loss + backward + EMA + TGN.

        V6 loss: L = L_dense_lookahead + lambda * L_weak_sigreg
        That's it. Two terms.
        """
        import torch
        import torch.nn as nn

        cfg = self.cfg
        device = self.device

        batch = batch.to(device)
        B = batch.batch_size

        # LR schedule
        lr_mult = self._lr_schedule(total_steps, total_expected_steps)
        for pg in self.optimizer_net.param_groups:
            pg["lr"] = cfg.lr * lr_mult
        for pg in self.optimizer_emb.param_groups:
            pg["lr"] = cfg.lr * cfg.emb_lr_mult * lr_mult

        # AMP context
        amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bfloat16" else torch.float16

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=cfg.use_amp):
            # Get node features from embedding table
            x = self.node_emb(batch.x.to(device))  # (N_batch, 128)

            # Add TGN memory if enabled in this phase
            if phase.tgn_enabled:
                batch_node_ids = batch.x.to(device)
                tgn_mem = self.tgn.get_memory(batch_node_ids, device)
                # Additive fusion: project TGN 64-dim to 128-dim latent space
                # TGN memory fills the "temporal slots" of the embedding
                if tgn_mem.shape[1] < cfg.latent_dim:
                    tgn_mem_full = torch.cat([
                        torch.zeros(
                            tgn_mem.shape[0],
                            cfg.latent_dim - cfg.tgn_dim,
                            device=device,
                            dtype=tgn_mem.dtype,
                        ),
                        tgn_mem,
                    ], dim=-1)
                else:
                    tgn_mem_full = tgn_mem
                x = x + tgn_mem_full

            # Temporal and numeric info
            batch_node_time = batch.node_time.to(device) if hasattr(batch, 'node_time') else None
            batch_edge_time = batch.edge_time.to(device) if hasattr(batch, 'edge_time') else None
            batch_edge_numeric = batch.edge_numeric.to(device) if hasattr(batch, 'edge_numeric') else None
            batch_vec = (
                batch.batch.to(device)
                if getattr(batch, 'batch', None) is not None
                else torch.zeros(x.shape[0], dtype=torch.long, device=device)
            )

            # Online encoder
            ctx_repr = self.online_encoder(
                x, batch.edge_index.to(device), batch.edge_attr.to(device),
                node_time=batch_node_time, edge_time=batch_edge_time,
                edge_numeric=batch_edge_numeric,
                batch=batch_vec,
            )

            # Target encoder (EMA, no grad)
            with torch.no_grad():
                x_target = self.node_emb(batch.x.to(device))
                if phase.tgn_enabled:
                    x_target = x_target + tgn_mem_full.detach()
                tgt_repr = self.target_encoder(
                    x_target, batch.edge_index.to(device), batch.edge_attr.to(device),
                    node_time=batch_node_time, edge_time=batch_edge_time,
                    edge_numeric=batch_edge_numeric,
                    batch=batch_vec,
                )

            # ============================================================
            # LOSS 1: Dense Temporal Lookahead
            # ============================================================
            dense_loss = self._compute_dense_lookahead_loss(
                ctx_repr, tgt_repr.detach(), batch_node_time, B
            )

            # ============================================================
            # LOSS 2: Weak-SIGReg (off-diagonal covariance only)
            # ============================================================
            sigreg_loss = self.sigreg(ctx_repr[:B])

            # ============================================================
            # TOTAL LOSS: L = L_dense + lambda * L_sigreg
            # ============================================================
            loss = dense_loss + cfg.sigreg_lambda * sigreg_loss

        # Backward with scaler
        self.optimizer_emb.zero_grad(set_to_none=True)
        self.optimizer_net.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer_net)
        nn.utils.clip_grad_norm_(self.net_trainable, cfg.grad_clip)
        self.scaler.step(self.optimizer_emb)
        self.scaler.step(self.optimizer_net)
        self.scaler.update()

        # FIX: Hard-reset frozen ontology nodes (zeroing grad is insufficient —
        # AdamW weight decay + momentum buffers still mutate frozen embeddings)
        if self.n_frozen > 0:
            with torch.no_grad():
                self.node_emb.weight[self.frozen_mask] = self.frozen_embs_cache

        # EMA update
        progress = total_steps / max(total_expected_steps, 1)
        tau = cfg.ema_tau_start + (cfg.ema_tau_end - cfg.ema_tau_start) * progress
        with torch.no_grad():
            for pe, po in zip(self.target_encoder.parameters(), self.online_encoder.parameters()):
                pe.data.mul_(tau).add_(po.data, alpha=1 - tau)

        # TGN memory update
        if phase.tgn_enabled and batch.edge_index.size(1) > 0:
            with torch.no_grad():
                ei = batch.edge_index.to(device)
                # Build edge features for TGN message computation
                pred_e = self.online_encoder.pred_emb(batch.edge_attr.to(device))
                if batch_edge_time is not None and batch_node_time is not None:
                    target_n = ei[1]
                    dt = batch_node_time[target_n] - batch_edge_time
                    phi_t = self.online_encoder.time_encoder(dt)
                else:
                    phi_t = torch.zeros(pred_e.size(0), cfg.time_dim, device=device)

                if batch_edge_numeric is not None:
                    num_e = self.online_encoder.numeric_encoder(batch_edge_numeric)
                else:
                    num_e = torch.zeros(pred_e.size(0), cfg.numeric_dim, device=device)

                raw_edge_feat = torch.cat([pred_e, phi_t, num_e], dim=-1)
                edge_feat = self.online_encoder.edge_proj(raw_edge_feat)

                # Compute and update TGN memory
                src_ids = batch.x[ei[0].cpu()]
                dst_ids = batch.x[ei[1].cpu()]
                unique_nodes, agg_msgs, _ = self.tgn.compute_messages(
                    src_ids, dst_ids, edge_feat, device
                )
                self.tgn.update_memory(unique_nodes, agg_msgs)

        metrics = {
            "total_loss": loss.item(),
            "dense_loss": dense_loss.item(),
            "sigreg_loss": sigreg_loss.item(),
            "lr": cfg.lr * lr_mult,
            "tau": tau,
        }
        return metrics

    def _checkpoint(self, epoch: int) -> None:
        """Full state checkpoint to Modal volume."""
        import torch

        ckpt_dir = self.cfg.artifact_dir
        os.makedirs(ckpt_dir, exist_ok=True)

        ckpt = {
            "epoch": epoch,
            "total_steps": self.total_steps,
            "optimizer_emb": self.optimizer_emb.state_dict(),
            "optimizer_net": self.optimizer_net.state_dict(),
            "scaler": self.scaler.state_dict(),
            "online_encoder": self.online_encoder.state_dict(),
            "target_encoder": self.target_encoder.state_dict(),
            "predictor": self.predictor.state_dict(),
            "onto_projection": self.onto_projection.state_dict(),
            "tgn_msg_fn": self.tgn.msg_fn.state_dict(),
            "tgn_gru": self.tgn.gru.state_dict(),
            "tgn_memory": self.tgn.state_for_checkpoint(),
            "rng_cpu": torch.random.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state(),
            "losses_log": self.losses_log,
            "config": self.cfg.to_json(),
        }
        ckpt_path = os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch + 1}.pt")
        torch.save(ckpt, ckpt_path)

        # Save latest checkpoint pointer
        torch.save(ckpt, os.path.join(ckpt_dir, "checkpoint_latest.pt"))

        # Save node embeddings as bare tensor for probe compatibility
        emb_path = os.path.join(ckpt_dir, f"node_emb_epoch_{epoch + 1}.pt")
        torch.save(self.node_emb.weight.detach().cpu(), emb_path)
        torch.save(
            self.node_emb.weight.detach().cpu(),
            os.path.join(ckpt_dir, "node_embeddings.pt"),
        )

        # Save config
        with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
            f.write(self.cfg.to_json())

        jepa_cache.commit()
        print(f"  Checkpoint saved: epoch {epoch + 1}")

    def _resume_checkpoint(self) -> None:
        """Load and restore all state from checkpoint if available."""
        import torch

        ckpt_path = os.path.join(self.cfg.artifact_dir, "checkpoint_latest.pt")
        if self.cfg.hospital_filter:
            print(f"  Single-hospital mode ({self.cfg.hospital_filter}) — skipping checkpoint resume")
            return
        if not os.path.exists(ckpt_path):
            print("  No checkpoint found, starting fresh")
            return

        print("  Resuming from checkpoint...")
        ckpt = torch.load(ckpt_path, weights_only=False)

        self.start_epoch = ckpt["epoch"] + 1
        self.total_steps = ckpt["total_steps"]
        self.losses_log = ckpt.get("losses_log", [])

        self.online_encoder.load_state_dict(ckpt["online_encoder"])
        self.target_encoder.load_state_dict(ckpt["target_encoder"])
        self.predictor.load_state_dict(ckpt["predictor"])

        if "onto_projection" in ckpt:
            self.onto_projection.load_state_dict(ckpt["onto_projection"])

        self.tgn.msg_fn.load_state_dict(ckpt["tgn_msg_fn"])
        self.tgn.gru.load_state_dict(ckpt["tgn_gru"])

        if "tgn_memory" in ckpt:
            self.tgn.load_checkpoint_state(ckpt["tgn_memory"])

        if "optimizer_emb" in ckpt and "optimizer_net" in ckpt:
            self.optimizer_emb.load_state_dict(ckpt["optimizer_emb"])
            self.optimizer_net.load_state_dict(ckpt["optimizer_net"])
        else:
            print("  Legacy checkpoint optimizer state detected; reinitializing optimizers for SGD/AdamW split")
        self.scaler.load_state_dict(ckpt["scaler"])

        if "rng_cpu" in ckpt:
            torch.random.set_rng_state(ckpt["rng_cpu"])
        if "rng_cuda" in ckpt:
            torch.cuda.set_rng_state(ckpt["rng_cuda"])

        print(f"  Resumed from epoch {self.start_epoch}, step {self.total_steps}")

    def train(self) -> TrainResult:
        """Main training loop over epochs/batches with 2-phase curriculum."""
        cfg = self.cfg

        # Estimate total steps
        steps_per_epoch = len(self.unique_src) // cfg.batch_size
        total_expected_steps = cfg.epochs * steps_per_epoch

        print(f"\n{'=' * 60}")
        print("V6 Dense Temporal JEPA Training")
        print(f"{'=' * 60}")
        print(f"  Nodes:          {self.num_nodes:,}")
        print(f"  Edges:          {self.n_edges:,}")
        print(f"  Predicates:     {self.num_predicates}")
        print(f"  Latent dim:     {cfg.latent_dim}")
        print(
            f"  Edge feat dim:  {cfg.edge_feat_dim} "
            f"(pred={cfg.pred_emb_dim} + time={cfg.time_dim} + num={cfg.numeric_dim})"
        )
        print(f"  GPS layers:     {cfg.gps_layers}")
        print(f"  GPS heads:      {cfg.gps_heads}")
        print(f"  Batch size:     {cfg.batch_size}")
        print(f"  Epochs:         {cfg.epochs} (starting from {self.start_epoch})")
        print(f"  Est. steps/ep:  {steps_per_epoch:,}")
        print(f"  Est. total:     {total_expected_steps:,}")
        print(f"  AMP:            {cfg.use_amp} ({cfg.amp_dtype})")
        print(f"  Frozen onto:    {self.n_frozen:,}")
        print(f"  SIGReg lambda:  {cfg.sigreg_lambda}")
        print("  Curriculum:     foundation (ep 0-2, no TGN) -> temporal (ep 3-9, TGN on)")

        t_train = time.time()
        current_phase_name = None

        for epoch in range(self.start_epoch, cfg.epochs):
            phase = self._get_phase(epoch)

            # Phase transition logging
            if phase.name != current_phase_name:
                current_phase_name = phase.name
                print(f"\n>>> Phase: {phase.name} (epochs {phase.epoch_start}-{phase.epoch_end - 1})")
                print(f"    Hops: {phase.num_neighbors}")
                print(f"    TGN:  {phase.tgn_enabled}")
                print(f"    Loss: L_dense_lookahead + {cfg.sigreg_lambda} * L_weak_sigreg")

                # Detach TGN memory at phase transition if newly enabled
                if phase.tgn_enabled and epoch == phase.epoch_start:
                    self.tgn.detach_memory()

            # Create loader for this phase
            loader, input_nodes = self._create_loader(phase)
            epoch_batches = 0
            epoch_t0 = time.time()

            for batch in loader:
                metrics = self._train_step(
                    batch, phase, self.total_steps, total_expected_steps
                )

                self.total_steps += 1
                epoch_batches += 1
                self.losses_log.append(metrics)

                # Logging every N batches
                if epoch_batches % cfg.log_every == 0:
                    elapsed = time.time() - epoch_t0
                    batch_per_s = epoch_batches / elapsed if elapsed > 0 else 0

                    parts = [
                        f"epoch {epoch + 1} batch {epoch_batches}",
                        f"total={metrics['total_loss']:.4f}",
                        f"dense={metrics['dense_loss']:.4f}",
                        f"sigreg={metrics['sigreg_loss']:.4f}",
                        f"lr={metrics['lr']:.2e}",
                        f"{batch_per_s:.1f} b/s",
                        f"phase={phase.name}",
                    ]
                    print("  " + " | ".join(parts))

            # End of epoch
            epoch_elapsed = time.time() - epoch_t0
            avg_total = 0.0
            avg_dense = 0.0
            avg_sigreg = 0.0
            if epoch_batches > 0 and self.losses_log:
                recent = self.losses_log[-epoch_batches:]
                avg_total = sum(m["total_loss"] for m in recent) / epoch_batches
                avg_dense = sum(m["dense_loss"] for m in recent) / epoch_batches
                avg_sigreg = sum(m["sigreg_loss"] for m in recent) / epoch_batches

            print(
                f"  Epoch {epoch + 1}/{cfg.epochs}: "
                f"total={avg_total:.4f} dense={avg_dense:.4f} sigreg={avg_sigreg:.4f} "
                f"({epoch_batches} batches, {epoch_elapsed:.0f}s)"
            )

            # Detach TGN memory between epochs
            if phase.tgn_enabled:
                self.tgn.detach_memory()

            # Checkpoint
            self._checkpoint(epoch)

        train_time = time.time() - t_train
        print(f"\nTraining complete: {self.total_steps} steps in {train_time:.1f}s")

        # Save final artifacts
        self._save_final_artifacts(train_time)

        return {
            "total_steps": self.total_steps,
            "final_loss": self.losses_log[-1]["total_loss"] if self.losses_log else None,
            "final_dense_loss": self.losses_log[-1]["dense_loss"] if self.losses_log else None,
            "final_sigreg_loss": self.losses_log[-1]["sigreg_loss"] if self.losses_log else None,
            "train_time_s": train_time,
            "num_nodes": self.num_nodes,
            "num_edges": self.n_edges,
            "num_predicates": self.num_predicates,
            "latent_dim": self.cfg.latent_dim,
            "epochs": self.cfg.epochs,
            "gps_layers": self.cfg.gps_layers,
        }

    def _save_final_artifacts(self, train_time: float) -> None:
        """Save final model artifacts."""
        import torch

        artifact_dir = self.cfg.artifact_dir
        os.makedirs(artifact_dir, exist_ok=True)

        # Bare tensor for probe compatibility (num_nodes, 128)
        torch.save(
            self.node_emb.weight.detach().cpu(),
            os.path.join(artifact_dir, "node_embeddings.pt"),
        )

        # Model state dicts
        torch.save(
            self.online_encoder.cpu().state_dict(),
            os.path.join(artifact_dir, "encoder.pt"),
        )
        torch.save(
            self.predictor.cpu().state_dict(),
            os.path.join(artifact_dir, "predictor.pt"),
        )
        torch.save(
            self.onto_projection.cpu().state_dict(),
            os.path.join(artifact_dir, "onto_projection.pt"),
        )

        # Config
        with open(os.path.join(artifact_dir, "config.json"), "w") as f:
            f.write(self.cfg.to_json())

        # Loss log
        with open(os.path.join(artifact_dir, "losses.json"), "w") as f:
            json.dump(self.losses_log, f)

        # Training summary
        with open(os.path.join(artifact_dir, "summary.json"), "w") as f:
            json.dump({
                "version": "v6",
                "architecture": "Dense Temporal JEPA",
                "num_nodes": self.num_nodes,
                "num_predicates": self.num_predicates,
                "latent_dim": self.cfg.latent_dim,
                "edge_feat_dim": self.cfg.edge_feat_dim,
                "n_edges": self.n_edges,
                "n_ontology_frozen": self.n_frozen,
                "n_instance_learnable": self.num_nodes - self.n_frozen,
                "epochs": self.cfg.epochs,
                "total_steps": self.total_steps,
                "train_time_s": train_time,
                "gps_layers": self.cfg.gps_layers,
                "losses": ["dense_lookahead", "weak_sigreg"],
                "sigreg_lambda": self.cfg.sigreg_lambda,
                "curriculum_phases": [p.name for p in CURRICULUM_PHASES_V6],
                "numeric_encoding": True,
                "warm_start": "v5_epoch1_zeropad",
            }, f, indent=2)

        # Node vocabulary for downstream
        vocab_path = os.path.join(artifact_dir, "node_vocab_sample.json")
        sample_vocab = dict(list(self.node_vocab.items())[:10000])
        with open(vocab_path, "w") as f:
            json.dump(sample_vocab, f)

        jepa_cache.commit()
        print(f"  Final artifacts saved to {artifact_dir}")


# ---------------------------------------------------------------------------
# Modal GPU training function
# ---------------------------------------------------------------------------


@scale_app.function(
    image=gpu_image_v6,
    gpu="A100-80GB",
    timeout=86400,  # 24h (Modal max)
    volumes=VOLUMES,
    memory=204800,  # 200GB RAM for TGN memory + graph
)
def train_tkg_jepa_v6(
    parquet_path: str = "/data/jcube_graph_v6.parquet",
    ontology_path: str = "/data/ontology_nodes.parquet",
    config_json: str | None = None,
) -> TrainResult:
    """Train V6 Dense Temporal JEPA on the full TKG.

    Architecture: 3-layer GraphGPS + TGN + Dense Lookahead + Weak-SIGReg.
    Single A100-80GB. 128-dim latent.

    Loss: L = L_dense_lookahead + lambda * L_weak_sigreg
    """
    import torch

    print("=" * 60)
    print("V6 Dense Temporal JEPA (World Model Architecture)")
    print("=" * 60)

    # Parse config
    if config_json:
        cfg = V6Config.from_json(config_json)
    else:
        cfg = V6Config()

    cfg.parquet_path = parquet_path
    cfg.ontology_path = ontology_path

    print(f"Config: {cfg.to_json()}")
    print(f"CUDA: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Build nn modules (deferred import pattern for Modal serialization)
    nn_modules = _build_nn_modules()

    # Create trainer and run
    trainer = V6Trainer(cfg, nn_modules, parquet_path, ontology_path)
    result = trainer.train()

    return result


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------


@scale_app.local_entrypoint()
def main(
    action: str = "full",
    catalog_path: str = "data/ai_friendly_catalog.json",
    config: str = "",
) -> None:
    """V6 Dense Temporal JEPA pipeline (all heavy work on Modal).

    Actions:
        materialize -- DuckDB -> Parquet edges + ontology texts + numeric values (Modal CPU)
        train       -- Train V6 Dense Temporal JEPA (Modal GPU, A100-80GB)
        full        -- All steps end-to-end

    Usage:
        modal run --detach event_jepa_cube/scale_pipeline_v6.py --action full
        modal run --detach event_jepa_cube/scale_pipeline_v6.py --action train
        modal run event_jepa_cube/scale_pipeline_v6.py --action train --config '{"epochs":1,"gps_layers":2}'
        modal run event_jepa_cube/scale_pipeline_v6.py --action materialize

    Prerequisites:
        - DB uploaded to Modal volume:
          modal volume put jcube-data data/aggregated_fixed_union.db /aggregated_fixed_union.db
        - Catalog uploaded: modal volume put jcube-data data/ai_friendly_catalog.json /ai_friendly_catalog.json
    """
    import json
    import sys

    sys.path.insert(0, ".")

    # Parse config overrides
    config_json = None
    if config:
        base = V6Config()
        overrides = json.loads(config)
        for k, v in overrides.items():
            if hasattr(base, k):
                setattr(base, k, v)
        config_json = base.to_json()
        print(f"Config overrides: {overrides}")

    if action in ("materialize", "full"):
        print("=" * 60)
        print("STEP 1: Materialize TKG on Modal (V6 with numeric_value column)")
        print("=" * 60)

        with open(catalog_path) as f:
            catalog_json_str = f.read()

        stats = materialize_remote.remote(catalog_json=catalog_json_str)
        print(f"\n  {stats['n_edges']:,} edges ({stats['edge_file_mb']:.1f} MB)")
        print(f"  {stats['n_ontology_nodes']:,} ontology nodes ({stats['ontology_file_mb']:.1f} MB)")

        if action == "materialize":
            return

    if action in ("train", "full"):
        print("\n" + "=" * 60)
        print("STEP 2: Train V6 Dense Temporal JEPA on A100")
        print("  Loss: L_dense_lookahead + lambda * L_weak_sigreg")
        print("  Curriculum: foundation (3 ep) -> temporal (7 ep)")
        print("=" * 60)

        result = train_tkg_jepa_v6.remote(
            parquet_path="/data/jcube_graph_v6.parquet",
            ontology_path="/data/ontology_nodes.parquet",
            config_json=config_json,
        )

        print(f"\n{'=' * 60}")
        print("RESULTS -- V6 Dense Temporal JEPA")
        print(f"{'=' * 60}")
        print(f"  Steps:        {result['total_steps']:,}")
        print(f"  Final loss:   {result['final_loss']}")
        print(f"  Dense loss:   {result['final_dense_loss']}")
        print(f"  SIGReg loss:  {result['final_sigreg_loss']}")
        print(f"  Train time:   {result['train_time_s']:.1f}s")
        print(f"  Nodes:        {result['num_nodes']:,}")
        print(f"  Edges:        {result['num_edges']:,}")
        print(f"  Predicates:   {result['num_predicates']}")
        print(f"  Latent dim:   {result['latent_dim']}")
        print(f"  GPS layers:   {result['gps_layers']}")


# ---------------------------------------------------------------------------
# CLI for local-only usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="V6 Dense Temporal JEPA Pipeline")
    parser.add_argument("--action", choices=["materialize", "train", "full"], default="full")
    parser.add_argument("--catalog", default="data/ai_friendly_catalog.json")
    parser.add_argument("--config", default="", help="JSON config overrides")
    args = parser.parse_args()

    if args.action == "materialize":
        from event_jepa_cube.scale_pipeline import materialize
        materialize(catalog_path=args.catalog)
    else:
        print("Training requires Modal. Use: modal run --detach event_jepa_cube/scale_pipeline_v6.py --action train")
