"""V5 SOTA Graph-JEPA pipeline — GraphGPS + TGN + VICReg + 8-loss curriculum.

Builds on V4's 35.2M-node / 165M-edge healthcare TKG. Key upgrades:
    - BGE-M3 ontology encoder (replaces Qwen 0.8B) — CLS pooling, 1024→128
    - 5-layer GraphGPS backbone: GatedGCN (local) + MultiheadAttention (global)
    - TGN per-node GRU memory (128-dim, CPU-resident)
    - VICReg loss (invariance + variance + covariance)
    - 8 losses with learnable uncertainty weighting (Kendall 2018)
    - 4-phase progressive curriculum (warmup → local → expand → full)
    - Per-epoch checkpointing with full resume (optimizer, scaler, TGN, RNG)
    - V4 warm-start: Linear(64→128) projection of instance embeddings

Output contract (unchanged from V4):
    (num_nodes, 128) tensor saved as node_embeddings.pt

Usage:
    modal run event_jepa_cube/scale_pipeline_v5.py --action full
    modal run event_jepa_cube/scale_pipeline_v5.py --action train
    modal run event_jepa_cube/scale_pipeline_v5.py --action materialize
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

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
class V5Config:
    """All hyperparameters centralized."""

    # Dimensions — 64 for single A100, 128 for 4x A100 FSDP
    latent_dim: int = 64
    time_dim: int = 16
    edge_feat_dim: int = 80  # latent_dim + time_dim
    rwse_dim: int = 16
    lappe_dim: int = 16

    # GraphGPS
    gps_layers: int = 3  # 3 for single A100, 5 for 4x
    gps_heads: int = 4
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

    # EMA
    ema_tau_start: float = 0.996
    ema_tau_end: float = 0.9999

    # VICReg weights
    vicreg_inv_weight: float = 25.0
    vicreg_var_weight: float = 25.0
    vicreg_cov_weight: float = 1.0

    # Lookahead
    lookahead_steps: int = 5
    lookahead_decay: float = 0.7

    # BGE-M3
    bge_model: str = "BAAI/bge-m3"
    bge_hidden_dim: int = 1024
    bge_max_length: int = 128
    bge_batch_size: int = 128

    # Encoder name for Qwen compatibility
    encoder_model: str = "BAAI/bge-m3"

    # Contrastive
    ntxent_temperature: float = 0.07

    # Data paths
    parquet_path: str = "/data/jcube_graph.parquet"
    ontology_path: str = "/data/ontology_nodes.parquet"
    artifact_dir: str = "/root/jepa-artifacts/tkg-v5"
    v4_artifact_dir: str = "/root/jepa-artifacts/tkg-fullscale"

    # AMP
    use_amp: bool = True
    amp_dtype: str = "bfloat16"

    # Workers
    num_workers: int = 4
    prefetch_factor: int = 2

    # Logging
    log_every: int = 50

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "V5Config":
        return cls(**json.loads(s))


@dataclass
class CurriculumPhase:
    """Progressive training curriculum phase."""

    name: str
    epoch_start: int
    epoch_end: int  # exclusive
    num_neighbors: list[int]
    all_hospitals: bool
    active_losses: list[str]
    tgn_enabled: bool = False
    lappe_enabled: bool = False
    node_sample_frac: float = 1.0


# The 4 curriculum phases
CURRICULUM_PHASES: list[CurriculumPhase] = [
    CurriculumPhase(
        name="warmup",
        epoch_start=0, epoch_end=2,
        num_neighbors=[15],
        all_hospitals=False,
        active_losses=["vicreg", "lookahead"],
        tgn_enabled=False,
        lappe_enabled=False,
    ),
    CurriculumPhase(
        name="local",
        epoch_start=2, epoch_end=4,
        num_neighbors=[15, 10],
        all_hospitals=False,
        active_losses=["vicreg", "lookahead", "edge_type_pred", "link_pred"],
        tgn_enabled=False,
        lappe_enabled=False,
    ),
    CurriculumPhase(
        name="expand",
        epoch_start=4, epoch_end=6,
        num_neighbors=[15, 10, 5],
        all_hospitals=True,
        active_losses=[
            "vicreg", "lookahead", "edge_type_pred", "link_pred",
            "temporal_ord", "contrastive",
        ],
        tgn_enabled=True,
        lappe_enabled=False,
    ),
    CurriculumPhase(
        name="full",
        epoch_start=6, epoch_end=10,
        num_neighbors=[25, 20, 15, 10, 5],
        all_hospitals=True,
        active_losses=[
            "vicreg", "lookahead", "edge_type_pred", "link_pred",
            "temporal_ord", "contrastive", "dgi", "dense_ctx",
        ],
        tgn_enabled=True,
        lappe_enabled=True,
        node_sample_frac=0.5,  # 50% sampling for 5-hop explosion
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


# ---------------------------------------------------------------------------
# Modal infrastructure
# ---------------------------------------------------------------------------

import modal

scale_app_v5 = modal.App("jcube-tkg-jepa-v5")

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

# V5 GPU image — BGE-M3 via sentence-transformers, no peft/cugraph
gpu_image_v5 = (
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
        "torch_scatter", "torch_sparse", "torch_cluster",
        find_links="https://data.pyg.org/whl/torch-2.10.0+cu128.html",
    )
)


# ---------------------------------------------------------------------------
# Remote DuckDB materialization — VERBATIM from V4 (lines 343-523)
# ---------------------------------------------------------------------------


@scale_app_v5.function(
    image=cpu_image,
    timeout=7200,
    volumes=VOLUMES,
    cpu=8,
    memory=32768,
)
def materialize_remote(
    catalog_json: str,
    db_path: str = "/data/aggregated_fixed_union.db",
    output_path: str = "/data/jcube_graph.parquet",
) -> dict:
    """Materialize 671M edges inside Modal container (plenty of disk/RAM).

    Also extracts unique entity texts for semantic encoding.
    """
    import json
    import time
    import duckdb
    import pyarrow.parquet as pq_local

    catalog = json.loads(catalog_json)
    tables = catalog.get("tables", [])

    # Build the UNION ALL query
    ts_priority = [
        "DH_CADASTRO", "DH_REALIZACAO", "DH_INICIO", "DH_FIM",
        "DH_ATUALIZACAO", "DT_INTER", "DT_ALTA", "DT_NASC",
    ]
    ts_types = {"TIMESTAMP", "TIMESTAMP WITH TIME ZONE", "DATE", "DATETIME", "TIMESTAMPTZ"}

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
                f"EPOCH(\"{t_col}\") AS t_epoch "
                f"FROM \"{t_name}\" "
                f"WHERE \"{subject_col}\" IS NOT NULL "
                f"AND \"{obj_col}\" IS NOT NULL "
                f"AND \"{t_col}\" IS NOT NULL"
            )

    query = " UNION ALL ".join(selects)
    print(f"Query: {len(selects)} SELECT statements")

    # Connect with out-of-core settings to handle 671M rows without OOM
    con = duckdb.connect(db_path, read_only=True)
    con.execute("PRAGMA temp_directory='/data/duckdb_temp'")
    con.execute("PRAGMA threads=8")
    con.execute("PRAGMA memory_limit='24GB'")

    # Materialize edges to Parquet
    import os
    if os.path.exists(output_path) and os.path.getsize(output_path) > 100_000_000:
        meta = pq_local.read_metadata(output_path)
        file_mb = os.path.getsize(output_path) / 1e6
        print(f"  SKIP edges -- already materialized: {meta.num_rows:,} edges ({file_mb:.1f} MB)")
        edge_time = 0.0
    else:
        print("Materializing edges to Parquet (ZSTD, out-of-core)...")
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
    # Extract ONTOLOGY node texts (the ~1% that need semantic encoding)
    # ---------------------------------------------------------------
    print("\nExtracting ontology node texts from dictionary tables...")
    t1 = time.time()

    from collections import Counter as Ctr
    fk_usage = Ctr()
    for t in tables:
        for c in t.get("columns", []):
            if c["name"].startswith("ID_CD_"):
                fk_usage[c["name"]] += 1

    ontology_queries = []
    for t in tables:
        cols = t.get("columns", [])
        row_count = t.get("row_count", 0)
        id_cols = [c["name"] for c in cols if c["name"].startswith("ID_CD_")]
        name_cols = [c["name"] for c in cols
                     if c["name"].startswith("NM_") or c["name"].startswith("DS_")]

        if len(id_cols) != 1 or not name_cols or row_count > 50000:
            continue
        entity_type = id_cols[0].replace("ID_CD_", "")
        if entity_type in INSTANCE_TYPES:
            continue
        refs = fk_usage.get(id_cols[0], 0)
        if refs < 3:
            continue

        text_parts = [f"'<{entity_type}>'"]
        for nc in name_cols[:4]:
            text_parts.append(f"' {nc}=' || COALESCE(CAST(\"{nc}\" AS VARCHAR), '')")
        text_expr = " || ".join(text_parts)

        ontology_queries.append(
            f"SELECT '{id_cols[0]}_' || CAST(\"{id_cols[0]}\" AS VARCHAR) AS node_id, "
            f"'{entity_type}' AS entity_type, "
            f"{text_expr} AS text_payload "
            f"FROM \"{t['name']}\" "
            f"WHERE \"{id_cols[0]}\" IS NOT NULL"
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
# BGE-M3 Ontology Encoder (Modal cls)
# ---------------------------------------------------------------------------


@scale_app_v5.cls(
    image=gpu_image_v5,
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
    def load_model(self):
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

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
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
# Neural network modules (defined at module level for pickling)
# ---------------------------------------------------------------------------
# These will be imported inside the GPU function, but we define the class
# bodies here and instantiate them inside the GPU context.


def _build_nn_modules():
    """Returns all nn.Module classes for V5. Called inside GPU context."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # ------------------------------------------------------------------
    # ContinuousTemporalEncoder — from V4
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
    # GraphNorm (lightweight, per-graph normalization)
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

        def forward(self, x: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
            if batch is None:
                # Single graph — treat all nodes as one graph
                mean = x.mean(dim=0, keepdim=True)
                x = x - self.mean_scale * mean
                var = x.pow(2).mean(dim=0, keepdim=True)
                x = x / (var + 1e-6).sqrt()
                return x * self.weight + self.bias

            # Per-graph normalization
            from torch_geometric.utils import scatter
            # Compute per-graph mean
            graph_mean = scatter(x, batch, dim=0, reduce="mean")  # (num_graphs, D)
            node_mean = graph_mean[batch]  # (N, D)
            x = x - self.mean_scale * node_mean

            # Compute per-graph variance
            graph_var = scatter(x.pow(2), batch, dim=0, reduce="mean")
            node_var = graph_var[batch]
            x = x / (node_var + 1e-6).sqrt()

            return x * self.weight + self.bias

    # ------------------------------------------------------------------
    # GatedGCNLayer — edge-gated local MPNN
    # ------------------------------------------------------------------

    class GatedGCNLayer(nn.Module):
        """Edge-Gated Graph Convolutional Layer (Bresson & Laurent 2017).

        Implements the edge-gated message passing with transforms A,B,C,D,E + BN.

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

            # Node transforms
            Ah = self.A(x)
            Bh = self.B(x)
            Ch = self.C(x)
            Dh = self.D(x)

            # Edge gating
            Ee = self.E(edge_attr)
            gate_input = Ch[src] + Dh[dst] + Ee
            sigma = torch.sigmoid(gate_input)

            # Gated messages
            messages = sigma * Bh[src]  # (E, dim)

            # Aggregate: sum of gated messages per destination node
            from torch_geometric.utils import scatter
            gate_sum = scatter(sigma, dst, dim=0, dim_size=x.size(0), reduce="sum").clamp(min=1e-6)
            msg_sum = scatter(messages, dst, dim=0, dim_size=x.size(0), reduce="sum")

            # Normalized aggregation
            agg = msg_sum / gate_sum

            # Node update with residual
            h_new = x + F.relu(self.bn_node(Ah + agg))
            h_new = self.dropout(h_new)

            # Edge update with residual
            e_new = edge_attr + F.relu(self.bn_edge(self.edge_proj(gate_input)))
            e_new = self.dropout(e_new)

            return h_new, e_new

    # ------------------------------------------------------------------
    # GPSLayer — GatedGCN + global attention + FFN
    # ------------------------------------------------------------------

    class GPSLayer(nn.Module):
        """Graph-GPS Layer (Rampasek et al. 2022).

        Combines:
          1. GatedGCN for local (neighborhood) message passing
          2. nn.MultiheadAttention for global (all-to-all within subgraph) attention
          3. FFN (2-layer MLP with GELU)
          4. GraphNorm + residual connections

        Each component has its own pre-norm and residual.
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
            # Local: GatedGCN
            self.local_norm = GraphNorm(dim)
            self.local_mpnn = GatedGCNLayer(dim, edge_dim, use_bn=use_bn, dropout=dropout)

            # Global: MultiheadAttention (within each subgraph)
            self.global_norm = GraphNorm(dim)
            self.global_attn = nn.MultiheadAttention(
                embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True,
            )

            # FFN
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
            batch: Optional[torch.Tensor] = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # 1. Local MPNN (GatedGCN) with pre-norm + residual
            h_local = self.local_norm(x, batch)
            h_local, edge_attr_new = self.local_mpnn(h_local, edge_index, edge_attr)
            x = x + self.dropout(h_local)

            # 2. Global attention within each subgraph
            h_global = self.global_norm(x, batch)

            if batch is not None:
                # Process per-graph for proper attention masking
                from torch_geometric.utils import unbatch
                graphs = unbatch(h_global, batch)
                attn_out_parts = []
                for g in graphs:
                    g_unsq = g.unsqueeze(0)  # (1, N_g, D)
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
    # TGNMemory — per-node GRU memory on CPU
    # ------------------------------------------------------------------

    class TGNMemory(nn.Module):
        """Temporal Graph Network Memory (Rossi et al. 2020).

        Per-node GRU memory stored in CPU RAM (~16.8 GB for 35M nodes x 128-dim).
        Gather/scatter per batch: move only active nodes' memory to GPU.

        Message: Linear(cat[mem_src, mem_dst, pred_emb, phi(dt)]) -> msg_dim
        Aggregation: last message per node per batch
        Update: GRUCell(msg, memory)
        """

        def __init__(self, num_nodes: int, mem_dim: int, msg_dim: int, edge_feat_dim: int):
            super().__init__()
            self.num_nodes = num_nodes
            self.mem_dim = mem_dim
            self.msg_dim = msg_dim

            # Per-node memory on CPU (never moves to GPU as a whole)
            self.register_buffer(
                "memory", torch.zeros(num_nodes, mem_dim), persistent=False
            )
            # Last-update timestamps for staleness detection
            self.register_buffer(
                "last_update", torch.zeros(num_nodes, dtype=torch.int64), persistent=False
            )

            # Message function: cat[mem_src, mem_dst, edge_feat] -> msg_dim
            # edge_feat = cat(pred_emb(latent_dim), phi(dt)(time_dim))
            self.msg_fn = nn.Linear(2 * mem_dim + edge_feat_dim, msg_dim)
            self.gru = nn.GRUCell(msg_dim, mem_dim)

        def get_memory(self, node_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
            """Gather memory for a batch of nodes. Returns (B, mem_dim) on device."""
            return self.memory[node_ids.cpu()].to(device)

        def compute_messages(
            self,
            src_ids: torch.Tensor,
            dst_ids: torch.Tensor,
            edge_feat: torch.Tensor,
            device: torch.device,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Compute messages for edges in batch.

            Returns:
                unique_nodes: unique node IDs involved (CPU)
                agg_messages: aggregated messages per unique node (device)
                raw: intermediate for debugging
            """
            mem_src = self.memory[src_ids.cpu()].to(device)
            mem_dst = self.memory[dst_ids.cpu()].to(device)
            msg_input = torch.cat([mem_src, mem_dst, edge_feat], dim=-1)
            messages = self.msg_fn(msg_input)  # (E, msg_dim)

            # Last-message aggregation per destination node
            all_nodes = torch.cat([src_ids, dst_ids]).cpu()
            all_msgs = torch.cat([messages, messages], dim=0)

            unique_nodes, inverse = torch.unique(all_nodes, return_inverse=True)

            # Scatter: last message per node (approximate — uses max index trick)
            agg = torch.zeros(unique_nodes.size(0), self.msg_dim, device=device)
            # Use scatter with overwrite (last wins, which is fine for batched training)
            agg.scatter_(0, inverse.to(device).unsqueeze(-1).expand(-1, self.msg_dim), all_msgs)

            return unique_nodes, agg, messages

        def update_memory(
            self,
            unique_nodes: torch.Tensor,
            agg_messages: torch.Tensor,
            timestamps: Optional[torch.Tensor] = None,
        ):
            """Update memory for nodes in this batch using GRU."""
            # Move relevant memory to device of agg_messages
            device = agg_messages.device
            old_mem = self.memory[unique_nodes.cpu()].to(device)
            new_mem = self.gru(agg_messages, old_mem)

            # Write back to CPU buffer
            self.memory[unique_nodes.cpu()] = new_mem.detach().cpu()
            if timestamps is not None:
                self.last_update[unique_nodes.cpu()] = timestamps.cpu()

        def detach_memory(self):
            """Detach memory from computation graph (call between epochs)."""
            self.memory.detach_()

        def reset_memory(self):
            """Zero out all memory (for fresh start)."""
            self.memory.zero_()
            self.last_update.zero_()

        def state_for_checkpoint(self) -> dict:
            """Return memory state for serialization."""
            return {
                "memory": self.memory.clone(),
                "last_update": self.last_update.clone(),
            }

        def load_checkpoint_state(self, state: dict):
            """Restore memory from checkpoint."""
            self.memory.copy_(state["memory"])
            self.last_update.copy_(state["last_update"])

    # ------------------------------------------------------------------
    # TemporalGPSEncoder — 5 GPS layers + PE + temporal
    # ------------------------------------------------------------------

    class TemporalGPSEncoder(nn.Module):
        """5-layer GraphGPS encoder with temporal encodings.

        Combines:
          - Predicate embedding (latent_dim)
          - Continuous temporal encoding phi(dt) (time_dim)
          - RWSE positional encoding (optional)
          - Laplacian PE (optional, phase 3 only)
          - 5 stacked GPSLayer blocks
          - Causal masking (future edges masked out)
        """

        def __init__(self, cfg: V5Config, num_predicates: int):
            super().__init__()
            self.cfg = cfg
            dim = cfg.latent_dim

            # Predicate embedding
            self.pred_emb = nn.Embedding(num_predicates, dim)

            # Temporal encoder
            self.time_encoder = ContinuousTemporalEncoder(cfg.time_dim)

            # Edge feature projection: cat(pred_emb(dim), phi(dt)(time_dim)) -> edge_feat_dim
            raw_edge_dim = dim + cfg.time_dim
            self.edge_proj = nn.Linear(raw_edge_dim, cfg.edge_feat_dim)

            # RWSE projection (optional positional encoding)
            self.rwse_proj = nn.Linear(cfg.rwse_dim, dim) if cfg.rwse_dim > 0 else None

            # LapPE projection (optional, enabled in phase 3)
            self.lappe_proj = nn.Linear(cfg.lappe_dim, dim) if cfg.lappe_dim > 0 else None

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
            node_time: Optional[torch.Tensor] = None,
            edge_time: Optional[torch.Tensor] = None,
            batch: Optional[torch.Tensor] = None,
            rwse: Optional[torch.Tensor] = None,
            lappe: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """Forward pass through the GPS encoder.

            Args:
                x: Node features (N, dim)
                edge_index: (2, E) edge indices
                edge_attr_idx: (E,) predicate type indices
                node_time: (N,) node timestamps
                edge_time: (E,) edge timestamps
                batch: (N,) graph membership for batched processing
                rwse: (N, rwse_dim) random walk structural encoding
                lappe: (N, lappe_dim) Laplacian positional encoding

            Returns:
                (N, dim) encoded node representations
            """
            # Build edge features: cat(pred_emb, phi(dt))
            e = self.pred_emb(edge_attr_idx)

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

                phi_t = self.time_encoder(delta_t)
                e = torch.cat([e, phi_t], dim=-1)
            else:
                zeros = torch.zeros(
                    e.shape[0], self.cfg.time_dim, device=e.device, dtype=e.dtype
                )
                e = torch.cat([e, zeros], dim=-1)

            # Project edge features to fixed dim
            e = self.edge_proj(e)

            # Add positional encodings to node features
            if rwse is not None and self.rwse_proj is not None:
                x = x + self.rwse_proj(rwse)

            if lappe is not None and self.lappe_proj is not None:
                x = x + self.lappe_proj(lappe)

            # GPS layers
            for gps in self.gps_layers:
                x, e = gps(x, edge_index, e, batch=batch)

            return self.final_norm(x)

    # ------------------------------------------------------------------
    # VICRegLoss — invariance + variance + covariance
    # ------------------------------------------------------------------

    class VICRegLoss(nn.Module):
        """VICReg Loss (Bardes et al. 2022), ported from Mycelia OLL.

        L = lambda_inv * L_inv + lambda_var * L_var + lambda_cov * L_cov

        - Invariance: MSE(online_proj, ema_target)
        - Variance: hinge loss on std(z) >= 1 per dimension
        - Covariance: off-diagonal covariance -> 0 (decorrelation)
        """

        def __init__(
            self,
            inv_weight: float = 25.0,
            var_weight: float = 25.0,
            cov_weight: float = 1.0,
        ):
            super().__init__()
            self.inv_weight = inv_weight
            self.var_weight = var_weight
            self.cov_weight = cov_weight

        def forward(
            self,
            online: torch.Tensor,
            target: torch.Tensor,
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            """Compute VICReg loss.

            Args:
                online: (B, D) online encoder output
                target: (B, D) EMA target encoder output (detached)

            Returns:
                (total_loss, metrics_dict)
            """
            # Invariance: MSE between online and target
            inv_loss = F.mse_loss(online, target)

            # Variance: std(z) should be >= 1 per dimension (prevent collapse)
            # Computed on online representations
            std_online = torch.sqrt(online.var(dim=0) + 1e-4)
            var_loss = torch.mean(F.relu(1.0 - std_online))

            std_target = torch.sqrt(target.var(dim=0) + 1e-4)
            var_loss = var_loss + torch.mean(F.relu(1.0 - std_target))

            # Covariance: off-diagonal of cov matrix should be 0
            N, D = online.shape
            online_centered = online - online.mean(dim=0)
            cov_online = (online_centered.T @ online_centered) / max(N - 1, 1)
            # Zero diagonal, sum squared off-diagonal
            off_diag_online = cov_online - torch.diag(torch.diag(cov_online))
            cov_loss = off_diag_online.pow(2).sum() / D

            target_centered = target - target.mean(dim=0)
            cov_target = (target_centered.T @ target_centered) / max(N - 1, 1)
            off_diag_target = cov_target - torch.diag(torch.diag(cov_target))
            cov_loss = cov_loss + off_diag_target.pow(2).sum() / D

            total = (
                self.inv_weight * inv_loss
                + self.var_weight * var_loss
                + self.cov_weight * cov_loss
            )

            metrics = {
                "vicreg_total": total,
                "vicreg_inv": inv_loss,
                "vicreg_var": var_loss,
                "vicreg_cov": cov_loss,
            }
            return total, metrics

    # ------------------------------------------------------------------
    # MultiTaskHead — 8 losses with learnable uncertainty weighting
    # ------------------------------------------------------------------

    class MultiTaskHead(nn.Module):
        """Multi-task loss head with learnable log(sigma^2) per task (Kendall 2018).

        L_total = sum_i [exp(-log_sigma2_i) * L_i + 0.5 * log_sigma2_i]

        Each loss is computed only when active in the current curriculum phase.

        Losses:
            1. vicreg:         VICReg (invariance + variance + covariance)
            2. lookahead:      Multi-step prediction (smooth_l1)
            3. edge_type_pred: Edge type classification (cross-entropy)
            4. temporal_ord:   Temporal ordering (BCE)
            5. link_pred:      Link prediction (BCE)
            6. dgi:            Deep Graph Infomax (BCE)
            7. contrastive:    NT-Xent contrastive (same entity at different times)
            8. dense_ctx:      Dense context prediction (smooth_l1)
        """

        LOSS_NAMES = [
            "vicreg", "lookahead", "edge_type_pred", "temporal_ord",
            "link_pred", "dgi", "contrastive", "dense_ctx",
        ]

        def __init__(self, cfg: V5Config, num_predicates: int):
            super().__init__()
            self.cfg = cfg
            dim = cfg.latent_dim

            # Learnable log(sigma^2) per task — initialized so exp(-s) ~ 1
            self.log_sigma2 = nn.ParameterDict({
                name: nn.Parameter(torch.zeros(1))
                for name in self.LOSS_NAMES
            })

            # VICReg module
            self.vicreg = VICRegLoss(
                inv_weight=cfg.vicreg_inv_weight,
                var_weight=cfg.vicreg_var_weight,
                cov_weight=cfg.vicreg_cov_weight,
            )

            # Edge type prediction head: cat(z_src, z_dst) -> num_predicates
            self.edge_type_head = nn.Sequential(
                nn.Linear(2 * dim, dim),
                nn.GELU(),
                nn.LayerNorm(dim),
                nn.Linear(dim, num_predicates),
            )

            # Temporal ordering head: cat(z_a, z_b) -> 1 (a before b)
            self.temporal_ord_head = nn.Sequential(
                nn.Linear(2 * dim, dim),
                nn.GELU(),
                nn.LayerNorm(dim),
                nn.Linear(dim, 1),
            )

            # Link prediction: bilinear(z_src, z_dst) -> scalar
            self.link_pred_bilinear = nn.Bilinear(dim, dim, 1)

            # DGI: bilinear(z_node, z_subgraph) -> scalar
            self.dgi_bilinear = nn.Bilinear(dim, dim, 1)

            # Contrastive projection head for NT-Xent
            self.contrastive_proj = nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
            )

            # Dense context head
            self.dense_ctx_head = nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
            )

        def forward(
            self,
            online_repr: torch.Tensor,
            ema_target: torch.Tensor,
            predictor_out: torch.Tensor,
            edge_index: torch.Tensor,
            edge_attr_idx: torch.Tensor,
            edge_time: Optional[torch.Tensor],
            batch: Optional[torch.Tensor],
            active_losses: list[str],
            B: int,
        ) -> tuple[torch.Tensor, dict[str, float]]:
            """Compute active losses and combine with uncertainty weighting.

            Args:
                online_repr: (N_batch, D) online encoder output
                ema_target: (N_batch, D) EMA target encoder output
                predictor_out: (B, K, D) or (B, D) predictor output
                edge_index: (2, E_batch) batch edge indices
                edge_attr_idx: (E_batch,) predicate indices
                edge_time: (E_batch,) edge timestamps
                batch: (N_batch,) graph membership
                active_losses: list of active loss names for current phase
                B: number of target (seed) nodes

            Returns:
                (total_loss, metrics_dict)
            """
            device = online_repr.device
            metrics: dict[str, float] = {}
            total_loss = torch.tensor(0.0, device=device)

            ctx_target = online_repr[:B]
            ema_target_B = ema_target[:B]

            # 1. VICReg
            if "vicreg" in active_losses:
                vic_loss, vic_metrics = self.vicreg(ctx_target, ema_target_B)
                s = self.log_sigma2["vicreg"]
                weighted = torch.exp(-s) * vic_loss + 0.5 * s
                total_loss = total_loss + weighted.squeeze()
                for k, v in vic_metrics.items():
                    metrics[k] = v.item()

            # 2. Lookahead
            if "lookahead" in active_losses:
                pred = predictor_out
                if pred.dim() == 2:
                    pred = pred.unsqueeze(1)
                la_loss = F.smooth_l1_loss(pred[:, 0], ema_target_B)

                # Multi-step shifted predictions
                if B > self.cfg.lookahead_steps:
                    for k in range(1, self.cfg.lookahead_steps):
                        shifted_target = ema_target_B.roll(-k, dims=0)[:B - k]
                        shifted_pred = pred[:B - k, min(k, pred.shape[1] - 1)]
                        la_loss = la_loss + (self.cfg.lookahead_decay ** k) * F.smooth_l1_loss(
                            shifted_pred, shifted_target
                        )
                    la_loss = la_loss / self.cfg.lookahead_steps

                s = self.log_sigma2["lookahead"]
                weighted = torch.exp(-s) * la_loss + 0.5 * s
                total_loss = total_loss + weighted.squeeze()
                metrics["lookahead"] = la_loss.item()

            # 3. Edge type prediction
            if "edge_type_pred" in active_losses and edge_index.size(1) > 0:
                # Sample edges for classification (max 4096 to limit compute)
                n_edges = edge_index.size(1)
                max_edges = min(4096, n_edges)
                if n_edges > max_edges:
                    perm = torch.randperm(n_edges, device=device)[:max_edges]
                    ei_sample = edge_index[:, perm]
                    ea_sample = edge_attr_idx[perm]
                else:
                    ei_sample = edge_index
                    ea_sample = edge_attr_idx

                src_repr = online_repr[ei_sample[0]]
                dst_repr = online_repr[ei_sample[1]]
                edge_logits = self.edge_type_head(torch.cat([src_repr, dst_repr], dim=-1))
                edge_loss = F.cross_entropy(edge_logits, ea_sample)

                s = self.log_sigma2["edge_type_pred"]
                weighted = torch.exp(-s) * edge_loss + 0.5 * s
                total_loss = total_loss + weighted.squeeze()
                metrics["edge_pred"] = edge_loss.item()

            # 4. Temporal ordering
            if "temporal_ord" in active_losses and edge_time is not None and edge_index.size(1) > 1:
                # Sample pairs of edges and predict which came first
                n_edges = edge_index.size(1)
                n_pairs = min(2048, n_edges // 2)
                if n_pairs > 0:
                    perm = torch.randperm(n_edges, device=device)
                    idx_a = perm[:n_pairs]
                    idx_b = perm[n_pairs:2 * n_pairs]

                    nodes_a = edge_index[0, idx_a]
                    nodes_b = edge_index[0, idx_b]
                    z_a = online_repr[nodes_a]
                    z_b = online_repr[nodes_b]

                    logits = self.temporal_ord_head(torch.cat([z_a, z_b], dim=-1)).squeeze(-1)
                    labels = (edge_time[idx_a] < edge_time[idx_b]).float()
                    temp_loss = F.binary_cross_entropy_with_logits(logits, labels)

                    s = self.log_sigma2["temporal_ord"]
                    weighted = torch.exp(-s) * temp_loss + 0.5 * s
                    total_loss = total_loss + weighted.squeeze()
                    metrics["temporal_ord"] = temp_loss.item()

            # 5. Link prediction
            if "link_pred" in active_losses and edge_index.size(1) > 0:
                n_edges = edge_index.size(1)
                n_pos = min(2048, n_edges)
                perm = torch.randperm(n_edges, device=device)[:n_pos]

                src_pos = online_repr[edge_index[0, perm]]
                dst_pos = online_repr[edge_index[1, perm]]
                pos_scores = self.link_pred_bilinear(src_pos, dst_pos).squeeze(-1)

                # Negative samples: random destination nodes
                neg_dst_idx = torch.randint(0, online_repr.size(0), (n_pos,), device=device)
                dst_neg = online_repr[neg_dst_idx]
                neg_scores = self.link_pred_bilinear(src_pos, dst_neg).squeeze(-1)

                link_logits = torch.cat([pos_scores, neg_scores])
                link_labels = torch.cat([
                    torch.ones(n_pos, device=device),
                    torch.zeros(n_pos, device=device),
                ])
                link_loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)

                s = self.log_sigma2["link_pred"]
                weighted = torch.exp(-s) * link_loss + 0.5 * s
                total_loss = total_loss + weighted.squeeze()
                metrics["link_pred"] = link_loss.item()

            # 6. Deep Graph Infomax (DGI)
            if "dgi" in active_losses and batch is not None:
                from torch_geometric.utils import scatter
                # Subgraph summary: mean pool per graph
                graph_summary = scatter(
                    online_repr, batch, dim=0, reduce="mean"
                )  # (num_graphs, D)
                node_summary = graph_summary[batch]  # (N_batch, D)

                # Positive: real node-graph pairs
                pos_scores = self.dgi_bilinear(
                    online_repr[:B], node_summary[:B]
                ).squeeze(-1)

                # Negative: shuffled node-graph pairs
                shuffle_idx = torch.randperm(B, device=device)
                neg_scores = self.dgi_bilinear(
                    online_repr[:B][shuffle_idx], node_summary[:B]
                ).squeeze(-1)

                dgi_logits = torch.cat([pos_scores, neg_scores])
                dgi_labels = torch.cat([
                    torch.ones(B, device=device),
                    torch.zeros(B, device=device),
                ])
                dgi_loss = F.binary_cross_entropy_with_logits(dgi_logits, dgi_labels)

                s = self.log_sigma2["dgi"]
                weighted = torch.exp(-s) * dgi_loss + 0.5 * s
                total_loss = total_loss + weighted.squeeze()
                metrics["dgi"] = dgi_loss.item()

            # 7. NT-Xent contrastive (same entity at different time offsets)
            if "contrastive" in active_losses and B > 1:
                # Use online and EMA as two "views" of the same entity
                z_online = self.contrastive_proj(ctx_target)
                z_target = self.contrastive_proj(ema_target_B)

                # L2 normalize
                z_online = F.normalize(z_online, dim=-1)
                z_target = F.normalize(z_target, dim=-1)

                # NT-Xent: similarity matrix
                sim = z_online @ z_target.T / self.cfg.ntxent_temperature  # (B, B)
                labels = torch.arange(B, device=device)
                contrastive_loss = (
                    F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)
                ) / 2.0

                s = self.log_sigma2["contrastive"]
                weighted = torch.exp(-s) * contrastive_loss + 0.5 * s
                total_loss = total_loss + weighted.squeeze()
                metrics["contrastive"] = contrastive_loss.item()

            # 8. Dense context prediction
            if "dense_ctx" in active_losses and B > 1:
                # Predict distance-weighted mean of neighbor representations
                if batch is not None:
                    from torch_geometric.utils import scatter
                    # For each seed node, compute mean of its neighbors' representations
                    dst_nodes = edge_index[1]
                    src_nodes = edge_index[0]

                    # Focus on edges where dst is a seed node (idx < B)
                    seed_mask = dst_nodes < B
                    if seed_mask.any():
                        seed_dst = dst_nodes[seed_mask]
                        neighbor_src = src_nodes[seed_mask]
                        neighbor_repr = online_repr[neighbor_src]

                        # Distance weighting via edge time difference
                        if edge_time is not None:
                            et_masked = edge_time[seed_mask].float()
                            # Inverse time distance as weight (more recent = higher weight)
                            max_t = et_masked.max().clamp(min=1.0)
                            weights = (et_masked / max_t).clamp(min=0.01).unsqueeze(-1)
                            weighted_repr = neighbor_repr * weights
                        else:
                            weighted_repr = neighbor_repr

                        # Mean pool per seed node
                        ctx_mean = scatter(
                            weighted_repr, seed_dst, dim=0,
                            dim_size=B, reduce="mean"
                        )  # (B, D)

                        # Predict from node repr
                        ctx_pred = self.dense_ctx_head(ctx_target)
                        dense_loss = F.smooth_l1_loss(ctx_pred, ctx_mean.detach())

                        s = self.log_sigma2["dense_ctx"]
                        weighted_l = torch.exp(-s) * dense_loss + 0.5 * s
                        total_loss = total_loss + weighted_l.squeeze()
                        metrics["dense_ctx"] = dense_loss.item()

            # Add regularization terms from log_sigma2
            # (already included in each loss's 0.5 * s term)
            sigma_summary = {
                f"sigma2_{name}": torch.exp(self.log_sigma2[name]).item()
                for name in active_losses
                if name in self.log_sigma2
            }
            metrics.update(sigma_summary)
            metrics["total_loss"] = total_loss.item()

            return total_loss, metrics

    # ------------------------------------------------------------------
    # GraphPredictor — from V4 (MLP, multi-step)
    # ------------------------------------------------------------------

    class GraphPredictor(nn.Module):
        """Predicts target node representation from context."""

        def __init__(self, dim: int, hidden: int = 256, n_steps: int = 1):
            super().__init__()
            self.n_steps = n_steps
            self.net = nn.Sequential(
                nn.Linear(dim, hidden),
                nn.GELU(),
                nn.LayerNorm(hidden),
                nn.Linear(hidden, dim * n_steps),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.net(x)
            if self.n_steps > 1:
                return out.reshape(x.shape[0], self.n_steps, -1)
            return out

    return {
        "ContinuousTemporalEncoder": ContinuousTemporalEncoder,
        "GraphNorm": GraphNorm,
        "GatedGCNLayer": GatedGCNLayer,
        "GPSLayer": GPSLayer,
        "TGNMemory": TGNMemory,
        "TemporalGPSEncoder": TemporalGPSEncoder,
        "VICRegLoss": VICRegLoss,
        "MultiTaskHead": MultiTaskHead,
        "GraphPredictor": GraphPredictor,
    }


# ---------------------------------------------------------------------------
# V5 Trainer
# ---------------------------------------------------------------------------


class V5Trainer:
    """Orchestrates V5 training: graph loading, encoding, model init, training loop.

    Constructed inside the GPU function with all imports available.
    """

    def __init__(
        self,
        cfg: V5Config,
        nn_modules: dict,
        parquet_path: str,
        ontology_path: str,
    ):
        import copy
        import gc
        import random

        import numpy as np
        import torch
        import torch.nn as nn

        self.cfg = cfg
        self.nn = nn_modules
        self.device = torch.device("cuda")

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        # Load graph
        self._load_graph(parquet_path)

        # Build vocabulary (with caching)
        self._build_vocab_cached()

        # Encode ontology with BGE-M3 (with caching)
        self._encode_ontology_cached(ontology_path)

        # Init embeddings (with V4 warm-start)
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

    def _load_graph(self, parquet_path: str):
        """Load the materialized TKG from Parquet using C++ fast path."""
        import numpy as np
        import pyarrow as pa_mod
        import pyarrow.compute as pc_mod
        import pyarrow.parquet as pq_mod
        import torch

        print("Loading TKG from Parquet...")
        t0 = time.time()

        table = pq_mod.read_table(parquet_path)
        self.n_edges = table.num_rows
        print(f"  {self.n_edges:,} edges loaded in {time.time() - t0:.1f}s")

        # C++ fast path: PyArrow dictionary encoding
        print("  Dictionary encoding (C++)...")
        t1 = time.time()

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

        self.num_predicates = len(self.pred_list)

        self.edge_index = torch.stack([
            torch.from_numpy(src_idx).to(torch.int64),
            torch.from_numpy(dst_idx).to(torch.int64),
        ], dim=0)
        self.edge_attr = torch.from_numpy(attr_idx).to(torch.int64)
        self.edge_time = torch.from_numpy(time_np).to(torch.int64)

        print(f"  {self.num_nodes:,} nodes, {self.num_predicates} predicates")
        print(f"  Loaded in {time.time() - t0:.1f}s")

        del table, subj, obj, src_idx, dst_idx, attr_idx, time_np
        import gc
        gc.collect()

    def _build_vocab_cached(self):
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

    def _encode_ontology_cached(self, ontology_path: str):
        """Encode ontology texts with BGE-M3, with caching to volume."""
        import os
        import torch
        import pyarrow.parquet as pq_mod

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

        # Encode with BGE-M3 locally (model is small enough for single GPU)
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

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(**tokens)
                # CLS pooling (BGE-M3 is bidirectional)
                cls_emb = out.last_hidden_state[:, 0, :]  # (B, 1024)
                all_onto_embs.append(cls_emb.cpu().float())

            if (i // bs) % 50 == 0 and i > 0:
                print(f"    {i:,}/{n_matched:,} encoded")

        self.onto_raw_embeddings = torch.cat(all_onto_embs, dim=0)  # (n_matched, 1024)
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

    def _init_embeddings(self):
        """Initialize node embedding table with hetero init + V4 warm-start."""
        import os
        import torch
        import torch.nn as nn

        dim = self.cfg.latent_dim

        # Create embedding table
        self.node_emb = nn.Embedding(self.num_nodes, dim).to(self.device)
        nn.init.xavier_uniform_(self.node_emb.weight)

        # V4 warm-start: project 64-dim V4 embeddings to 128-dim
        v4_emb_path = os.path.join(self.cfg.v4_artifact_dir, "node_embeddings.pt")
        if os.path.exists(v4_emb_path):
            print("  Loading V4 embeddings for warm-start...")
            v4_state = torch.load(v4_emb_path, weights_only=True)
            # V4 saved as state_dict with 'weight' key or bare tensor
            if isinstance(v4_state, dict) and "weight" in v4_state:
                v4_emb = v4_state["weight"]
            elif isinstance(v4_state, torch.Tensor):
                v4_emb = v4_state
            else:
                v4_emb = None
                print("  Warning: V4 checkpoint format not recognized, skipping warm-start")

            if v4_emb is not None:
                v4_dim = v4_emb.shape[1]
                if v4_emb.shape[0] == self.num_nodes and v4_dim != dim:
                    # Project V4 embeddings to V5 dim
                    proj = nn.Linear(v4_dim, dim, bias=False)
                    nn.init.kaiming_uniform_(proj.weight)
                    proj = proj.to(self.device)

                    # Only warm-start instance nodes (not ontology)
                    frozen_set = set(self.onto_indices)
                    instance_mask = torch.ones(self.num_nodes, dtype=torch.bool)
                    if frozen_set:
                        for idx in self.onto_indices:
                            instance_mask[idx] = False

                    with torch.no_grad():
                        # Process in chunks to avoid OOM
                        chunk_size = 500_000
                        for start in range(0, self.num_nodes, chunk_size):
                            end = min(start + chunk_size, self.num_nodes)
                            chunk_mask = instance_mask[start:end]
                            if chunk_mask.any():
                                v4_chunk = v4_emb[start:end][chunk_mask].to(self.device)
                                projected = proj(v4_chunk)
                                indices = torch.arange(start, end, device=self.device)[chunk_mask]
                                self.node_emb.weight[indices] = projected.float()

                    n_warm = instance_mask.sum().item()
                    print(f"  V4 warm-start: {n_warm:,} instance nodes projected {v4_dim} -> {dim}")
                    del proj
                elif v4_emb.shape[0] != self.num_nodes:
                    print(f"  V4 node count mismatch ({v4_emb.shape[0]} vs {self.num_nodes}), skipping warm-start")
                del v4_emb
        else:
            print("  No V4 embeddings found, using random init")

        # Ontology projection: BGE 1024 -> latent_dim with Xavier init
        self.onto_projection = nn.Linear(self.cfg.bge_hidden_dim, dim, bias=False).to(self.device)
        nn.init.xavier_uniform_(self.onto_projection.weight)

        # Overwrite ontology node embeddings with projected BGE vectors
        self.n_frozen = 0
        if self.onto_raw_embeddings is not None and len(self.onto_indices) > 0:
            onto_idx_tensor = torch.tensor(self.onto_indices, dtype=torch.long)
            with torch.no_grad():
                # Project 1024 -> latent_dim
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

    def _build_graph_data(self):
        """Build PyG Data object for NeighborLoader."""
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
            node_time=self.node_time,
            num_nodes=self.num_nodes,
        )

        # Nodes with edges (for training)
        self.unique_src = torch.unique(self.edge_index[0])
        print(f"  Nodes with outgoing edges: {self.unique_src.shape[0]:,}")

    def _build_csr_cache(self):
        """Pre-build CSR adjacency and cache to volume (fix epoch boundary thrash)."""
        import os
        import torch

        csr_cache_path = os.path.join(self.cfg.artifact_dir, "csr_cache.pt")
        if os.path.exists(csr_cache_path):
            print("  CSR cache exists, PyG will use optimized path")
            return

        print("  Pre-building CSR adjacency for NeighborLoader...")
        t0 = time.time()
        # Force PyG to build its internal CSR representation
        # by accessing the graph via a small dummy loader
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

        # Save a flag file so we know CSR was built
        torch.save({"built": True, "num_nodes": self.num_nodes}, csr_cache_path)
        jepa_cache.commit()
        print(f"  CSR pre-built in {time.time() - t0:.1f}s")

    def _create_models(self):
        """Instantiate all neural network models."""
        import copy
        import torch
        import torch.nn as nn

        cfg = self.cfg
        M = self.nn

        # Online encoder
        self.online_encoder = M["TemporalGPSEncoder"](cfg, self.num_predicates).to(self.device)

        # Target encoder (EMA, no grad)
        self.target_encoder = copy.deepcopy(self.online_encoder).to(self.device)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Predictor (discarded after training)
        self.predictor = M["GraphPredictor"](
            cfg.latent_dim, hidden=cfg.latent_dim * 2, n_steps=cfg.lookahead_steps
        ).to(self.device)

        # Multi-task loss head
        self.multi_task = M["MultiTaskHead"](cfg, self.num_predicates).to(self.device)

        # TGN Memory (on CPU)
        self.tgn = M["TGNMemory"](
            num_nodes=self.num_nodes,
            mem_dim=cfg.tgn_dim,
            msg_dim=cfg.tgn_msg_dim,
            edge_feat_dim=cfg.edge_feat_dim,
        )
        # Keep TGN parameters on GPU for gradient computation, but memory buffer on CPU
        self.tgn.msg_fn = self.tgn.msg_fn.to(self.device)
        self.tgn.gru = self.tgn.gru.to(self.device)

        # Parameter counts
        n_encoder = sum(p.numel() for p in self.online_encoder.parameters())
        n_predictor = sum(p.numel() for p in self.predictor.parameters())
        n_multitask = sum(p.numel() for p in self.multi_task.parameters())
        n_emb = self.node_emb.weight.numel()
        n_tgn = sum(p.numel() for p in self.tgn.parameters())

        print(f"\n=== V5 Model Summary ===")
        print(f"  Online encoder:  {n_encoder:,} params")
        print(f"  Predictor:       {n_predictor:,} params")
        print(f"  Multi-task head: {n_multitask:,} params")
        print(f"  Node embeddings: {n_emb:,} params ({n_emb * 4 / 1e9:.2f} GB)")
        print(f"  TGN:             {n_tgn:,} params")
        print(f"  TGN memory:      {self.num_nodes * cfg.tgn_dim * 4 / 1e9:.2f} GB (CPU)")
        total_trainable = n_encoder + n_predictor + n_multitask + n_emb + n_tgn
        print(f"  Total trainable: {total_trainable:,} params")

    def _create_optimizer(self):
        """Create optimizer and AMP scaler."""
        import torch

        cfg = self.cfg
        trainable = (
            list(self.node_emb.parameters())
            + list(self.online_encoder.parameters())
            + list(self.predictor.parameters())
            + list(self.multi_task.parameters())
            + list(self.tgn.msg_fn.parameters())
            + list(self.tgn.gru.parameters())
        )
        self.trainable = trainable
        self.optimizer = torch.optim.AdamW(
            trainable, lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp)

    def _get_phase(self, epoch: int) -> CurriculumPhase:
        """Return the current CurriculumPhase for the given epoch."""
        for phase in CURRICULUM_PHASES:
            if phase.epoch_start <= epoch < phase.epoch_end:
                return phase
        # Default: last phase
        return CURRICULUM_PHASES[-1]

    def _create_loader(self, phase: CurriculumPhase):
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

    def _train_step(
        self,
        batch,
        phase: CurriculumPhase,
        total_steps: int,
        total_expected_steps: int,
    ) -> dict[str, float]:
        """Single training step: forward + loss + backward + EMA + frozen grad zero + TGN."""
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        cfg = self.cfg
        device = self.device

        batch = batch.to(device)
        B = batch.batch_size

        # LR schedule
        lr_mult = self._lr_schedule(total_steps, total_expected_steps)
        for pg in self.optimizer.param_groups:
            pg["lr"] = cfg.lr * lr_mult

        # AMP context
        amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bfloat16" else torch.float16

        with torch.cuda.amp.autocast(dtype=amp_dtype, enabled=cfg.use_amp):
            # Get node features from embedding table
            x = self.node_emb(batch.x.to(device))  # (N_batch, dim)

            # Add TGN memory if enabled
            if phase.tgn_enabled:
                batch_node_ids = batch.x.to(device)
                tgn_mem = self.tgn.get_memory(batch_node_ids, device)
                x = x + tgn_mem  # Additive fusion

            # Temporal info
            batch_node_time = batch.node_time.to(device) if hasattr(batch, 'node_time') else None
            batch_edge_time = batch.edge_time.to(device) if hasattr(batch, 'edge_time') else None
            batch_vec = batch.batch.to(device) if getattr(batch, 'batch', None) is not None else torch.zeros(x.shape[0], dtype=torch.long, device=device)

            # Online encoder
            ctx_repr = self.online_encoder(
                x, batch.edge_index.to(device), batch.edge_attr.to(device),
                node_time=batch_node_time, edge_time=batch_edge_time,
                batch=batch_vec,
            )

            # Target encoder (EMA, no grad)
            with torch.no_grad():
                x_target = self.node_emb(batch.x.to(device))
                if phase.tgn_enabled:
                    x_target = x_target + tgn_mem.detach()
                tgt_repr = self.target_encoder(
                    x_target, batch.edge_index.to(device), batch.edge_attr.to(device),
                    node_time=batch_node_time, edge_time=batch_edge_time,
                    batch=batch_vec,
                )

            # Predictor
            pred_out = self.predictor(ctx_repr[:B])

            # Multi-task loss
            loss, metrics = self.multi_task(
                online_repr=ctx_repr,
                ema_target=tgt_repr.detach(),
                predictor_out=pred_out,
                edge_index=batch.edge_index.to(device),
                edge_attr_idx=batch.edge_attr.to(device),
                edge_time=batch_edge_time,
                batch=batch_vec,
                active_losses=phase.active_losses,
                B=B,
            )

        # Backward with scaler
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.trainable, cfg.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Zero out gradients for frozen ontology nodes
        if self.n_frozen > 0:
            with torch.no_grad():
                if self.node_emb.weight.grad is not None:
                    self.node_emb.weight.grad[self.frozen_mask] = 0.0

        # EMA update
        progress = total_steps / max(total_expected_steps, 1)
        tau = cfg.ema_tau_start + (cfg.ema_tau_end - cfg.ema_tau_start) * progress
        with torch.no_grad():
            for pe, po in zip(self.target_encoder.parameters(), self.online_encoder.parameters()):
                pe.data.mul_(tau).add_(po.data, alpha=1 - tau)

        # TGN memory update
        if phase.tgn_enabled and batch.edge_index.size(1) > 0:
            with torch.no_grad():
                # Build edge features for TGN message computation
                ei = batch.edge_index.to(device)
                pred_e = self.online_encoder.pred_emb(batch.edge_attr.to(device))
                if batch_edge_time is not None and batch_node_time is not None:
                    target_n = ei[1]
                    dt = batch_node_time[target_n] - batch_edge_time
                    phi_t = self.online_encoder.time_encoder(dt)
                    edge_feat = torch.cat([pred_e, phi_t], dim=-1)
                else:
                    zeros = torch.zeros(pred_e.size(0), cfg.time_dim, device=device)
                    edge_feat = torch.cat([pred_e, zeros], dim=-1)

                # Project to edge_feat_dim
                edge_feat = self.online_encoder.edge_proj(edge_feat)

                # Compute and update TGN memory
                src_ids = batch.x[ei[0].cpu()]
                dst_ids = batch.x[ei[1].cpu()]
                unique_nodes, agg_msgs, _ = self.tgn.compute_messages(
                    src_ids, dst_ids, edge_feat, device
                )
                self.tgn.update_memory(unique_nodes, agg_msgs)

        metrics["lr"] = cfg.lr * lr_mult
        metrics["tau"] = tau
        return metrics

    def _checkpoint(self, epoch: int):
        """Full state checkpoint to Modal volume."""
        import torch

        ckpt_dir = self.cfg.artifact_dir
        os.makedirs(ckpt_dir, exist_ok=True)

        ckpt = {
            "epoch": epoch,
            "total_steps": self.total_steps,
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "online_encoder": self.online_encoder.state_dict(),
            "target_encoder": self.target_encoder.state_dict(),
            "predictor": self.predictor.state_dict(),
            "multi_task": self.multi_task.state_dict(),
            "tgn_msg_fn": self.tgn.msg_fn.state_dict(),
            "tgn_gru": self.tgn.gru.state_dict(),
            "tgn_memory": self.tgn.state_for_checkpoint(),
            "rng_cpu": torch.random.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state(),
            "losses_log": self.losses_log,
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

    def _resume_checkpoint(self):
        """Load and restore all state from checkpoint if available."""
        import torch

        ckpt_path = os.path.join(self.cfg.artifact_dir, "checkpoint_latest.pt")
        if not os.path.exists(ckpt_path):
            print("  No checkpoint found, starting fresh")
            return

        print(f"  Resuming from checkpoint...")
        ckpt = torch.load(ckpt_path, weights_only=False)

        self.start_epoch = ckpt["epoch"] + 1
        self.total_steps = ckpt["total_steps"]
        self.losses_log = ckpt.get("losses_log", [])

        self.online_encoder.load_state_dict(ckpt["online_encoder"])
        self.target_encoder.load_state_dict(ckpt["target_encoder"])
        self.predictor.load_state_dict(ckpt["predictor"])
        self.multi_task.load_state_dict(ckpt["multi_task"])
        self.tgn.msg_fn.load_state_dict(ckpt["tgn_msg_fn"])
        self.tgn.gru.load_state_dict(ckpt["tgn_gru"])

        if "tgn_memory" in ckpt:
            self.tgn.load_checkpoint_state(ckpt["tgn_memory"])

        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scaler.load_state_dict(ckpt["scaler"])

        if "rng_cpu" in ckpt:
            torch.random.set_rng_state(ckpt["rng_cpu"])
        if "rng_cuda" in ckpt:
            torch.cuda.set_rng_state(ckpt["rng_cuda"])

        print(f"  Resumed from epoch {self.start_epoch}, step {self.total_steps}")

    def train(self) -> dict:
        """Main training loop over epochs/batches with curriculum switching."""
        import torch

        cfg = self.cfg

        # Estimate total steps
        steps_per_epoch = len(self.unique_src) // cfg.batch_size
        total_expected_steps = cfg.epochs * steps_per_epoch

        print(f"\n{'=' * 60}")
        print(f"V5 Graph-JEPA Training")
        print(f"{'=' * 60}")
        print(f"  Nodes:          {self.num_nodes:,}")
        print(f"  Edges:          {self.n_edges:,}")
        print(f"  Predicates:     {self.num_predicates}")
        print(f"  Latent dim:     {cfg.latent_dim}")
        print(f"  GPS layers:     {cfg.gps_layers}")
        print(f"  Batch size:     {cfg.batch_size}")
        print(f"  Epochs:         {cfg.epochs} (starting from {self.start_epoch})")
        print(f"  Est. steps/ep:  {steps_per_epoch:,}")
        print(f"  Est. total:     {total_expected_steps:,}")
        print(f"  AMP:            {cfg.use_amp} ({cfg.amp_dtype})")
        print(f"  Frozen onto:    {self.n_frozen:,}")

        t_train = time.time()
        current_phase_name = None

        for epoch in range(self.start_epoch, cfg.epochs):
            phase = self._get_phase(epoch)

            # Phase transition logging
            if phase.name != current_phase_name:
                current_phase_name = phase.name
                print(f"\n>>> Curriculum Phase: {phase.name} (epochs {phase.epoch_start}-{phase.epoch_end - 1})")
                print(f"    Hops: {phase.num_neighbors}")
                print(f"    Losses: {phase.active_losses}")
                print(f"    TGN: {phase.tgn_enabled}")
                print(f"    LapPE: {phase.lappe_enabled}")

                # Reset TGN memory at phase transition if newly enabled
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
                        f"total={metrics.get('total_loss', 0):.4f}",
                    ]
                    # Add individual loss components
                    for key in [
                        "vicreg_inv", "vicreg_var", "vicreg_cov",
                        "lookahead", "edge_pred", "temporal_ord",
                        "link_pred", "dgi", "contrastive", "dense_ctx",
                    ]:
                        if key in metrics:
                            parts.append(f"{key}={metrics[key]:.4f}")

                    parts.append(f"lr={metrics.get('lr', 0):.2e}")
                    parts.append(f"{batch_per_s:.1f} batch/s")
                    parts.append(f"phase={phase.name}")

                    print("  " + " | ".join(parts))

            # End of epoch
            epoch_elapsed = time.time() - epoch_t0
            avg_loss = 0.0
            if epoch_batches > 0 and self.losses_log:
                recent = self.losses_log[-epoch_batches:]
                avg_loss = sum(m.get("total_loss", 0) for m in recent) / epoch_batches

            print(
                f"  Epoch {epoch + 1}/{cfg.epochs}: avg_loss={avg_loss:.4f} "
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
            "final_loss": self.losses_log[-1].get("total_loss", 0) if self.losses_log else None,
            "train_time_s": train_time,
            "num_nodes": self.num_nodes,
            "num_edges": self.n_edges,
            "num_predicates": self.num_predicates,
            "latent_dim": self.cfg.latent_dim,
            "epochs": self.cfg.epochs,
            "gps_layers": self.cfg.gps_layers,
        }

    def _save_final_artifacts(self, train_time: float):
        """Save final model artifacts."""
        import torch

        artifact_dir = self.cfg.artifact_dir
        os.makedirs(artifact_dir, exist_ok=True)

        # Bare tensor for probe compatibility
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
            self.multi_task.cpu().state_dict(),
            os.path.join(artifact_dir, "multi_task.pt"),
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
                "num_nodes": self.num_nodes,
                "num_predicates": self.num_predicates,
                "latent_dim": self.cfg.latent_dim,
                "n_edges": self.n_edges,
                "n_ontology_frozen": self.n_frozen,
                "n_instance_learnable": self.num_nodes - self.n_frozen,
                "epochs": self.cfg.epochs,
                "total_steps": self.total_steps,
                "train_time_s": train_time,
                "gps_layers": self.cfg.gps_layers,
                "curriculum_phases": [p.name for p in CURRICULUM_PHASES],
            }, f, indent=2)

        # Node vocabulary for downstream
        vocab_path = os.path.join(artifact_dir, "node_vocab_sample.json")
        # Save first 10k entries as sample (full vocab too large for JSON)
        sample_vocab = dict(list(self.node_vocab.items())[:10000])
        with open(vocab_path, "w") as f:
            json.dump(sample_vocab, f)

        jepa_cache.commit()
        print(f"  Final artifacts saved to {artifact_dir}")


# ---------------------------------------------------------------------------
# Modal GPU training function
# ---------------------------------------------------------------------------


@scale_app_v5.function(
    image=gpu_image_v5,
    gpu="A100-80GB",
    timeout=86400,  # 24h
    volumes=VOLUMES,
    memory=204800,  # 200GB RAM for TGN memory + graph
)
def train_tkg_jepa_v5(
    parquet_path: str = "/data/jcube_graph.parquet",
    ontology_path: str = "/data/ontology_nodes.parquet",
    config_json: Optional[str] = None,
) -> dict:
    """Train V5 Graph-JEPA on the full 165M-edge TKG.

    Architecture: 5-layer GraphGPS + TGN + VICReg + 8 curriculum losses.
    Single A100-80GB (upgrade to 4x A100 FSDP in Phase 2).
    """
    import torch

    print("=" * 60)
    print("V5 Graph-JEPA Training (GraphGPS + TGN + VICReg)")
    print("=" * 60)

    # Parse config
    if config_json:
        cfg = V5Config.from_json(config_json)
    else:
        cfg = V5Config()

    cfg.parquet_path = parquet_path
    cfg.ontology_path = ontology_path

    print(f"Config: {cfg.to_json()}")
    print(f"CUDA: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Build nn modules
    nn_modules = _build_nn_modules()

    # Create trainer and run
    trainer = V5Trainer(cfg, nn_modules, parquet_path, ontology_path)
    result = trainer.train()

    return result


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------


@scale_app_v5.local_entrypoint()
def main(
    action: str = "full",
    catalog_path: str = "data/ai_friendly_catalog.json",
    config: str = "",
):
    """V5 SOTA Graph-JEPA pipeline (all heavy work on Modal).

    Actions:
        materialize -- DuckDB -> Parquet edges + ontology texts (Modal CPU)
        train       -- Train V5 Graph-JEPA (Modal GPU, A100-80GB)
        full        -- All steps end-to-end

    Usage:
        modal run event_jepa_cube/scale_pipeline_v5.py --action full
        modal run event_jepa_cube/scale_pipeline_v5.py --action train
        modal run event_jepa_cube/scale_pipeline_v5.py --action train --config '{"epochs":1,"gps_layers":2,"batch_size":256}'
        modal run event_jepa_cube/scale_pipeline_v5.py --action materialize

    Prerequisites:
        - DB uploaded to Modal volume: modal volume put jcube-data data/aggregated_fixed_union.db /aggregated_fixed_union.db
        - Catalog uploaded: modal volume put jcube-data data/ai_friendly_catalog.json /ai_friendly_catalog.json
    """
    import json
    import sys
    import time

    sys.path.insert(0, ".")

    # Parse config overrides
    config_json = None
    if config:
        # Merge overrides into default config
        base = V5Config()
        overrides = json.loads(config)
        for k, v in overrides.items():
            if hasattr(base, k):
                setattr(base, k, v)
        config_json = base.to_json()
        print(f"Config overrides: {overrides}")

    if action in ("materialize", "full"):
        print("=" * 60)
        print("STEP 1: Materialize TKG on Modal (remote DuckDB)")
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
        print("STEP 2: Train V5 Graph-JEPA on A100 (BGE-M3 + GPS + TGN + VICReg)")
        print("=" * 60)

        result = train_tkg_jepa_v5.remote(
            parquet_path="/data/jcube_graph.parquet",
            ontology_path="/data/ontology_nodes.parquet",
            config_json=config_json,
        )

        print(f"\n{'=' * 60}")
        print(f"RESULTS -- V5 TKG Graph-JEPA")
        print(f"{'=' * 60}")
        print(f"  Steps:      {result['total_steps']:,}")
        print(f"  Final loss: {result['final_loss']}")
        print(f"  Train time: {result['train_time_s']:.1f}s")
        print(f"  Nodes:      {result['num_nodes']:,}")
        print(f"  Edges:      {result['num_edges']:,}")
        print(f"  Predicates: {result['num_predicates']}")
        print(f"  Latent dim: {result['latent_dim']}")
        print(f"  GPS layers: {result['gps_layers']}")


# ---------------------------------------------------------------------------
# CLI for local-only usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="V5 TKG-JEPA Pipeline")
    parser.add_argument("--action", choices=["materialize", "train", "full"], default="full")
    parser.add_argument("--catalog", default="data/ai_friendly_catalog.json")
    parser.add_argument("--config", default="", help="JSON config overrides")
    args = parser.parse_args()

    # For local-only materialization (no Modal)
    if args.action == "materialize":
        from event_jepa_cube.scale_pipeline import materialize
        materialize(catalog_path=args.catalog)
    else:
        print("Training requires Modal. Use: modal run event_jepa_cube/scale_pipeline_v5.py --action train")
# v5-fix-1774275723
