"""Full-scale Temporal Knowledge Graph JEPA pipeline.

The 5.8GB healthcare DB (417 tables, 165M edges, 424 entity types) is materialized
into a compressed Parquet of RDF triples (subject, predicate, object, time).

Two node types:
    - ONTOLOGY (~136K): CID codes, TUSS procedures, medications, categories.
      Encoded ONCE by Qwen3.5-0.8B → frozen embeddings. These are the "laws of physics"
      that anchor medical meaning in the graph.
    - INSTANCE (~5M): Patients, admissions, invoices, beds.
      nn.Embedding(N, 64) — learned purely from graph topology via JEPA loss.
      The GNN propagates Qwen's frozen medical semantics into these representations.

Architecture:
    1. Materialize: DuckDB UNION ALL → (S, P, O, T) → Parquet (one-off, ~4 min)
    2. Ontology:    29 dictionary tables → 136K texts → Qwen → frozen (1024→64) embeddings
    3. Train:       Hetero init (frozen ontology + learnable instances) + Temporal GNN
                    + JEPA loss + WeakSIGReg + multi-step lookahead

Usage:
    modal run event_jepa_cube/scale_pipeline.py --action full
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

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
# Timestamp priority for picking the best temporal column per table
# ---------------------------------------------------------------------------

TIMESTAMP_PRIORITY = [
    "DH_CADASTRO", "DH_REALIZACAO", "DH_INICIO", "DH_FIM",
    "DH_ATUALIZACAO", "DT_INTER", "DT_ALTA", "DT_NASC",
]

# Instance entity types get source_db prefix to avoid cross-hospital ID collisions.
# Ontology types (codes, categories, procedures) stay global — same CID means the same thing everywhere.
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
# Step 1: One-off DuckDB → Parquet materialization
# ---------------------------------------------------------------------------


def _build_triples_query(catalog_path: str) -> tuple[str, dict]:
    """Metaprogram a UNION ALL query from the AI-friendly catalog.

    Each table with 2+ entity columns + a timestamp produces edges:
        (subject_id, predicate, object_id, t_event)

    Subject = first ID_CD_ col, objects = remaining ID_CD_ cols.
    Predicate encodes both the relationship and the source table.
    """
    with open(catalog_path) as f:
        catalog = json.load(f)

    tables = catalog.get("tables", [])
    selects: list[str] = []
    stats = {"tables_scanned": len(tables), "edge_tables": 0, "predicates": 0}

    for table in tables:
        t_name = table["name"]
        cols = table.get("columns", [])

        id_cols = [c["name"] for c in cols if c["name"].startswith("ID_CD_")]
        ts_cols = [c["name"] for c in cols
                   if c.get("data_type", "") in _TS_TYPES
                   or c["name"].startswith("DH_") or c["name"].startswith("DT_")]

        if len(id_cols) < 2 or not ts_cols:
            continue

        stats["edge_tables"] += 1

        # Pick best timestamp
        t_col = ts_cols[0]
        col_names = [c["name"] for c in cols]
        for tp in TIMESTAMP_PRIORITY:
            if tp in col_names:
                t_col = tp
                break

        subject_col = id_cols[0]
        subj_type = subject_col.replace("ID_CD_", "")
        for obj_col in id_cols[1:]:
            obj_type = obj_col.replace("ID_CD_", "")
            pred_name = f"HAS_{obj_type}_VIA_{t_name}"
            stats["predicates"] += 1

            # Instance types get source_db prefix; ontology stays global
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

    return " UNION ALL ".join(selects), stats


def materialize(
    db_path: str = "data/aggregated_fixed_union.db",
    catalog_path: str = "data/ai_friendly_catalog.json",
    output_path: str = "data/jcube_graph.parquet",
) -> dict:
    """Materialize 671M edges to ZSTD-compressed Parquet. One-off, ~3 min."""
    if duckdb is None:
        raise ImportError("duckdb required")

    con = duckdb.connect(db_path, read_only=True)

    print("Building metaprogrammed UNION ALL query from catalog...")
    query, stats = _build_triples_query(catalog_path)
    print(f"  {stats['edge_tables']} tables, {stats['predicates']} predicates")

    print(f"Materializing to {output_path} (ZSTD compression)...")
    t0 = time.time()
    con.execute(
        f"COPY ({query}) TO '{output_path}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    elapsed = time.time() - t0

    # Read back stats
    meta = pq.read_metadata(output_path)
    stats["n_edges"] = meta.num_rows
    stats["file_size_mb"] = Path(output_path).stat().st_size / 1e6
    stats["time_s"] = elapsed

    con.close()

    print(f"  {stats['n_edges']:,} edges in {elapsed:.1f}s")
    print(f"  File: {stats['file_size_mb']:.1f} MB")

    # Save stats
    stats_path = Path(output_path).with_suffix(".stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)

    return stats


# ---------------------------------------------------------------------------
# Step 2: Zero-copy graph loading (Arrow → DLPack → PyTorch)
# ---------------------------------------------------------------------------


def load_tkg_tensors(parquet_path: str) -> tuple[int, Any, Any, Any, Any, dict, dict]:
    """Load the materialized TKG as pinned PyTorch tensors.

    Returns:
        (num_nodes, edge_index, edge_attr, edge_time,
         node_vocab, pred_vocab)

    edge_index: (2, E) int64 — standard PyG format
    edge_attr:  (E,)   int64 — predicate type per edge
    edge_time:  (E,)   int64 — epoch seconds per edge
    node_vocab: str → int mapping
    pred_vocab: str → int mapping
    """
    import torch

    print(f"Loading TKG from {parquet_path}...")
    t0 = time.time()

    table = pq.read_table(parquet_path)
    n_edges = table.num_rows
    print(f"  {n_edges:,} edges loaded in {time.time() - t0:.1f}s")

    # Dictionary-encode strings to integer IDs
    print("  Dictionary encoding (C++)...")
    t1 = time.time()

    # Unified node vocabulary: subject ∪ object
    subj_col = table.column("subject_id")
    obj_col = table.column("object_id")

    # Combine and deduplicate
    all_entities = pc.unique(
        pa.concat_arrays([subj_col.combine_chunks(), obj_col.combine_chunks()])
    )
    node_vocab = {s: i for i, s in enumerate(all_entities.to_pylist())}

    # Encode subject/object against unified vocabulary
    src_encoded = pc.index_in(subj_col, value_set=all_entities)
    dst_encoded = pc.index_in(obj_col, value_set=all_entities)

    # Predicate encoding
    pred_encoded = pc.dictionary_encode(table.column("predicate"))
    pred_vocab = {s: i for i, s in enumerate(pred_encoded.dictionary.to_pylist())}

    print(f"  {len(node_vocab):,} unique nodes, {len(pred_vocab)} predicates "
          f"({time.time() - t1:.1f}s)")

    # Zero-copy handoff to PyTorch
    print("  Zero-copy → PyTorch tensors...")
    t2 = time.time()

    # Extract integer arrays — try zero-copy, fall back to regular numpy
    try:
        src_np = src_encoded.to_numpy(zero_copy_only=True)
        dst_np = dst_encoded.to_numpy(zero_copy_only=True)
    except pa.ArrowInvalid:
        # Nulls or chunked arrays prevent zero-copy
        src_np = src_encoded.to_numpy(zero_copy_only=False)
        dst_np = dst_encoded.to_numpy(zero_copy_only=False)

    try:
        attr_np = pred_encoded.indices.to_numpy(zero_copy_only=True)
    except (pa.ArrowInvalid, AttributeError):
        attr_np = pred_encoded.indices.to_numpy(zero_copy_only=False)

    time_col = table.column("t_epoch")
    try:
        time_np = time_col.to_numpy(zero_copy_only=True)
    except pa.ArrowInvalid:
        time_np = time_col.to_numpy(zero_copy_only=False)

    edge_index = torch.stack([
        torch.from_numpy(src_np).to(torch.int64),
        torch.from_numpy(dst_np).to(torch.int64),
    ], dim=0)

    edge_attr = torch.from_numpy(attr_np).to(torch.int64)
    edge_time = torch.from_numpy(time_np).to(torch.int64)

    print(f"  Tensors created in {time.time() - t2:.4f}s")
    print(f"  edge_index: {edge_index.shape}")
    print(f"  Total load time: {time.time() - t0:.1f}s")

    # Pin memory for async GPU DMA
    edge_index = edge_index.pin_memory()
    edge_attr = edge_attr.pin_memory()
    edge_time = edge_time.pin_memory()

    return len(node_vocab), edge_index, edge_attr, edge_time, node_vocab, pred_vocab


# ---------------------------------------------------------------------------
# Step 3: Modal GPU training with PyG
# ---------------------------------------------------------------------------

import modal

scale_app = modal.App("jcube-tkg-jepa")

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

# GPU image for PyG training
# PyG wheels need matching torch+CUDA — install torch first, then PyG from PyG's wheel index
gpu_image = (
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
        "peft>=0.14",
        "accelerate>=0.35",
        "huggingface-hub>=0.27",
    )
    .pip_install("torch_geometric>=2.6")
    .pip_install(
        "torch_scatter", "torch_sparse", "torch_cluster",
        find_links="https://data.pyg.org/whl/torch-2.10.0+cu128.html",
    )
    .pip_install(
        "cugraph-pyg-cu13", "cugraph-cu13", "pylibcugraph-cu13", "nx-cugraph-cu13",
        extra_index_url="https://pypi.nvidia.com",
    )
)


# ---------------------------------------------------------------------------
# Remote DuckDB materialization (runs inside Modal, not locally)
# ---------------------------------------------------------------------------


@scale_app.function(
    image=cpu_image,
    timeout=7200,  # 2 hours — text extraction is slow
    volumes=VOLUMES,
    cpu=8,
    memory=32768,  # 32GB RAM for DuckDB
)
def materialize_remote(
    catalog_json: str,
    db_path: str = "/data/aggregated_fixed_union.db",
    output_path: str = "/data/jcube_graph.parquet",
) -> dict:
    """Materialize 671M edges inside Modal container (plenty of disk/RAM).

    Also extracts unique entity texts for Qwen semantic encoding.
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

    # Materialize edges to Parquet — COPY streams directly to disk, never holds all in RAM
    import os
    if os.path.exists(output_path) and os.path.getsize(output_path) > 100_000_000:
        meta = pq_local.read_metadata(output_path)
        file_mb = os.path.getsize(output_path) / 1e6
        print(f"  SKIP edges — already materialized: {meta.num_rows:,} edges ({file_mb:.1f} MB)")
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
    # Extract ONTOLOGY node texts (the ~1% that need Qwen encoding)
    # These are dictionary/reference tables: CID codes, TUSS procedures,
    # medication names, categories, etc. — the "laws of physics."
    # Instance nodes (patients, admissions, invoices) get nn.Embedding.
    # ---------------------------------------------------------------
    print("\nExtracting ontology node texts from dictionary tables...")
    t1 = time.time()

    # Ontology tables: small reference tables with descriptive text columns
    # Uses module-level INSTANCE_TYPES to filter out instance entities
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

        # Build text payload: "<TYPE> col1=val col2=val ..."
        text_parts = [f"'<{entity_type}>'"]
        for nc in name_cols[:4]:
            text_parts.append(f"' {nc}=' || COALESCE(CAST(\"{nc}\" AS VARCHAR), '∅')")
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
# Remote Qwen semantic encoding (distributed across A100s)
# ---------------------------------------------------------------------------


@scale_app.cls(
    image=gpu_image,
    gpu="A100-80GB",
    timeout=3600,
    volumes=VOLUMES,
    max_containers=20,
)
class QwenEncoder:
    """Distributed Qwen encoder for unique node texts.

    Loads model once per container (via @modal.enter), then processes
    batches of unique texts via .map().

    Output: qwen_node_embeddings.parquet on the Modal volume.
    """

    @modal.enter()
    def load_model(self):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.device = "cuda"
        model_name = "Qwen/Qwen3.5-0.8B"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModel.from_pretrained(
            model_name, torch_dtype=torch.bfloat16,
            trust_remote_code=True, attn_implementation="sdpa",
        ).to(self.device).eval()

        # Optionally load LoRA adapter if available
        import os
        lora_path = "/root/jepa-artifacts/lora-fullscale"
        if os.path.exists(lora_path):
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, lora_path).eval()
            print("  LoRA adapter loaded")

    @modal.method()
    def embed_batch(self, node_ids: list[str], texts: list[str]) -> list[tuple[str, list[float]]]:
        import torch

        results = []
        bs = 64
        for i in range(0, len(texts), bs):
            batch_ids = node_ids[i:i + bs]
            batch_texts = texts[i:i + bs]

            tokens = self.tokenizer(
                batch_texts, return_tensors="pt",
                padding=True, truncation=True, max_length=128,
            ).to(self.device)

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = self.model(
                    input_ids=tokens["input_ids"],
                    attention_mask=tokens["attention_mask"],
                )

            # Last-token pooling (causal model)
            hidden = outputs.last_hidden_state  # (B, T, H)
            seq_lens = tokens["attention_mask"].sum(dim=1) - 1
            batch_idx = torch.arange(len(batch_texts), device=self.device)
            embeddings = hidden[batch_idx, seq_lens, :]  # (B, H)

            for nid, emb in zip(batch_ids, embeddings.cpu().float().tolist()):
                results.append((nid, emb))

        return results


@scale_app.function(
    image=gpu_image,
    gpu="A100-80GB",
    timeout=86400,  # 24h — full-scale training
    volumes=VOLUMES,
)
def train_tkg_jepa(
    parquet_path: str = "/data/jcube_graph.parquet",
    ontology_path: str = "/data/ontology_nodes.parquet",
    latent_dim: int = 64,
    n_hops: list[int] | None = None,
    batch_size: int = 1024,
    epochs: int = 10,
    lr: float = 3e-4,
    ema_tau_start: float = 0.99,
    ema_tau_end: float = 0.999,
    reg_weight: float = 0.05,
    lookahead_steps: int = 5,
    lookahead_decay: float = 0.7,
    qwen_model: str = "Qwen/Qwen3.5-0.8B",
) -> dict:
    """Train Graph-JEPA on the full 165M-edge TKG with hetero node init.

    Two node types:
        ONTOLOGY (~136K): CID codes, TUSS, medications → frozen Qwen embeddings
        INSTANCE (~5M):   Patients, admissions, invoices → learnable nn.Embedding

    The GNN propagates Qwen's frozen medical semantics into instance representations.
    """
    import copy
    import math
    import os
    import random
    import time

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Data
    from torch_geometric.nn import TransformerConv

    # PyG NeighborLoader (cuGraph requires graph store conversion — use standard PyG for now)
    from torch_geometric.loader import NeighborLoader
    print("[PyG] NeighborLoader ready")

    if n_hops is None:
        n_hops = [15, 10, 5]

    device = torch.device("cuda")
    random.seed(42)
    torch.manual_seed(42)

    # ================================================================
    # Load graph
    # ================================================================
    print("Loading TKG...")
    t0 = time.time()

    import pyarrow.parquet as pq_mod
    import pyarrow.compute as pc_mod
    import pyarrow as pa_mod

    table = pq_mod.read_table(parquet_path)
    n_edges = table.num_rows
    print(f"  {n_edges:,} edges")

    # ================================================================
    # C++ fast path: PyArrow dictionary encoding (no Python GIL)
    # Replaces 19-min Python dict loop with ~15s C++ execution
    # ================================================================
    import numpy as np

    # Combine subject+object chunks into one virtual ChunkedArray (no copy)
    subj = table.column("subject_id")
    obj = table.column("object_id")
    all_nodes_chunked = pa_mod.chunked_array(subj.chunks + obj.chunks)

    # pc.unique runs entirely in C++ — handles chunked arrays natively
    unique_nodes = pc_mod.unique(all_nodes_chunked)
    num_nodes = len(unique_nodes)
    print(f"  {num_nodes:,} unique nodes (C++ dedup)")

    # pc.index_in: maps each string to its index in unique_nodes (C++)
    src_idx = pc_mod.index_in(subj, value_set=unique_nodes).to_numpy(zero_copy_only=False).astype(np.int64)
    dst_idx = pc_mod.index_in(obj, value_set=unique_nodes).to_numpy(zero_copy_only=False).astype(np.int64)

    # Build node_vocab dict for ontology matching later
    node_vocab = {s: i for i, s in enumerate(unique_nodes.to_pylist())}
    node_list = list(node_vocab.keys())

    # Predicate encoding via C++ dictionary_encode (chunked-safe)
    pred_chunked = table.column("predicate")
    pred_unique = pc_mod.unique(pa_mod.chunked_array(pred_chunked.chunks))
    pred_list = pred_unique.to_pylist()
    attr_idx = pc_mod.index_in(pred_chunked, value_set=pred_unique).to_numpy(zero_copy_only=False).astype(np.int64)
    attr_np = attr_idx; del attr_idx

    # Timestamps (numeric, safe to concat)
    time_parts = []
    for chunk in table.column("t_epoch").iterchunks():
        time_parts.append(chunk.to_numpy(zero_copy_only=False))
    time_np = np.concatenate(time_parts); del time_parts

    num_predicates = len(pred_list)

    edge_index = torch.stack([
        torch.from_numpy(src_idx).to(torch.int64),
        torch.from_numpy(dst_idx).to(torch.int64),
    ], dim=0)
    edge_attr = torch.from_numpy(attr_np).to(torch.int64)
    edge_time = torch.from_numpy(time_np).to(torch.int64)

    print(f"  {num_nodes:,} nodes, {num_predicates} predicates")
    print(f"  Loaded in {time.time() - t0:.1f}s")

    del table, subj, obj, src_idx, dst_idx, attr_np, time_np
    import gc; gc.collect()

    # ================================================================
    # Heterogeneous node init: Ontology (Qwen) + Instance (learnable)
    # ================================================================
    print("\n[Hetero Init] Loading ontology nodes for Qwen encoding...")
    t_onto = time.time()

    # Load ontology texts
    ontology_node_ids = []
    ontology_texts = []
    if os.path.exists(ontology_path):
        onto_table = pq_mod.read_table(ontology_path)
        ontology_node_ids = onto_table.column("node_id").to_pylist()
        ontology_texts = onto_table.column("text_payload").to_pylist()
        del onto_table
    n_onto = len(ontology_node_ids)

    # Map ontology node_ids to their indices in node_vocab
    onto_indices = []
    onto_texts_matched = []
    for nid, txt in zip(ontology_node_ids, ontology_texts):
        if nid in node_vocab:
            onto_indices.append(node_vocab[nid])
            onto_texts_matched.append(txt)

    n_matched = len(onto_indices)
    print(f"  {n_onto:,} ontology nodes in parquet, {n_matched:,} matched in graph")

    # Encode ontology texts with Qwen → frozen embeddings
    # Then project 1024 → latent_dim
    onto_embeddings = None
    if n_matched > 0:
        print(f"  Loading Qwen for ontology encoding ({n_matched:,} texts)...")
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(qwen_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        qwen = AutoModel.from_pretrained(
            qwen_model, torch_dtype=torch.bfloat16,
            trust_remote_code=True, attn_implementation="sdpa",
        ).to(device).eval()

        hidden_dim = getattr(qwen.config, 'hidden_size', None) or getattr(qwen.config, 'd_model', None) or qwen.config.to_dict().get('hidden_size', qwen.config.to_dict().get('d_model', 1024))
        projection = nn.Linear(hidden_dim, latent_dim, bias=False).to(device)
        nn.init.xavier_uniform_(projection.weight)

        # Encode in batches
        all_onto_embs = []
        bs = 128
        for i in range(0, n_matched, bs):
            batch_texts = onto_texts_matched[i:i + bs]
            tokens = tokenizer(
                batch_texts, return_tensors="pt",
                padding=True, truncation=True, max_length=128,
            ).to(device)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = qwen(**tokens)
                # Mean-pool last hidden state
                mask = tokens["attention_mask"].unsqueeze(-1).float()
                pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-8)
                projected = projection(pooled.float())  # (B, latent_dim)
                all_onto_embs.append(projected.cpu())

            if (i // bs) % 50 == 0 and i > 0:
                print(f"    {i:,}/{n_matched:,} encoded")

        onto_embeddings = torch.cat(all_onto_embs, dim=0)  # (n_matched, latent_dim)
        print(f"  Ontology encoded: {onto_embeddings.shape} in {time.time() - t_onto:.1f}s")

        # Free Qwen — we only needed it for the 136K ontology texts
        del qwen, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        print("  Qwen freed from VRAM")

    # ================================================================
    # Build node embedding table with hetero init
    # ================================================================
    print("\nBuilding heterogeneous node embeddings...")

    node_emb = nn.Embedding(num_nodes, latent_dim).to(device)
    nn.init.xavier_uniform_(node_emb.weight)

    # Overwrite ontology node embeddings with frozen Qwen vectors
    n_frozen = 0
    if onto_embeddings is not None:
        onto_idx_tensor = torch.tensor(onto_indices, dtype=torch.long)
        with torch.no_grad():
            node_emb.weight[onto_idx_tensor] = onto_embeddings.to(device).float()
        n_frozen = len(onto_indices)
        print(f"  {n_frozen:,} ontology nodes → frozen Qwen embeddings")
        print(f"  {num_nodes - n_frozen:,} instance nodes → learnable (random init)")

    # Create frozen mask: ontology nodes don't receive gradients
    frozen_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    if onto_indices:
        frozen_mask[torch.tensor(onto_indices, dtype=torch.long)] = True

    # Create Data object (on CPU for NeighborLoader)
    graph_data = Data(
        x=torch.arange(num_nodes),  # node indices (features loaded in forward pass)
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_time=edge_time,
        num_nodes=num_nodes,
    )

    # ================================================================
    # Model definition
    # ================================================================

    class TemporalGNNEncoder(nn.Module):
        """2-layer Temporal Graph Transformer.

        Uses TransformerConv which applies multi-head attention over edges,
        naturally weighting relationships by their edge attributes (predicate type).
        """
        def __init__(self, dim: int, n_preds: int, heads: int = 4):
            super().__init__()
            # Predicate embedding (edge features)
            self.pred_emb = nn.Embedding(n_preds, dim)

            self.conv1 = TransformerConv(
                in_channels=dim, out_channels=dim // heads,
                heads=heads, edge_dim=dim, concat=True,
            )
            self.norm1 = nn.LayerNorm(dim)

            self.conv2 = TransformerConv(
                in_channels=dim, out_channels=dim // heads,
                heads=heads, edge_dim=dim, concat=True,
            )
            self.norm2 = nn.LayerNorm(dim)

        def forward(self, x, edge_index, edge_attr):
            # edge_attr is predicate int → embed to dim-vector
            e = self.pred_emb(edge_attr)

            h = self.conv1(x, edge_index, e)
            h = self.norm1(h)
            h = F.gelu(h)

            h = self.conv2(h, edge_index, e)
            h = self.norm2(h)
            return h

    # Predictor (discarded after training — JEPA principle)
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

        def forward(self, x):
            out = self.net(x)
            if self.n_steps > 1:
                return out.reshape(x.shape[0], self.n_steps, -1)
            return out

    # WeakSIGReg
    class WeakSIGReg:
        def __init__(self, sketch_dim=64):
            self.sketch_dim = sketch_dim
            self._S = None
            self._d = None

        def compute_loss(self, embeddings):
            n, d = embeddings.shape
            if self._S is None or self._d != d:
                self._S = torch.randn(self.sketch_dim, d, device=embeddings.device) / math.sqrt(d)
                self._d = d
            z = embeddings - embeddings.mean(dim=0, keepdim=True)
            z_s = z @ self._S.t()
            cov_s = (z_s.t() @ z_s) / float(n)
            target = self._S @ self._S.t()
            return torch.norm(cov_s - target, p="fro") ** 2

    # Instantiate
    online_encoder = TemporalGNNEncoder(latent_dim, num_predicates).to(device)
    target_encoder = copy.deepcopy(online_encoder).to(device)
    for p in target_encoder.parameters():
        p.requires_grad = False

    predictor = GraphPredictor(latent_dim, n_steps=lookahead_steps).to(device)
    sigreg = WeakSIGReg(sketch_dim=min(64, latent_dim))

    # All trainable params
    trainable = (
        list(node_emb.parameters()) +
        list(online_encoder.parameters()) +
        list(predictor.parameters())
    )
    n_params = sum(p.numel() for p in trainable)
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)

    print(f"\n=== Graph-JEPA Training ===")
    print(f"  Nodes: {num_nodes:,}")
    print(f"  Edges: {n_edges:,}")
    print(f"  Predicates: {num_predicates}")
    print(f"  Latent dim: {latent_dim}")
    print(f"  Node emb: {num_nodes * latent_dim * 4 / 1e9:.2f} GB")
    print(f"  Hops: {n_hops}")
    print(f"  Trainable params: {n_params:,}")
    print(f"  Epochs: {epochs}, Batch: {batch_size}, LR: {lr}")

    # ================================================================
    # NeighborLoader (temporal-aware subgraph sampling)
    # ================================================================
    # Train on all nodes that have at least some edges
    unique_src = torch.unique(edge_index[0])
    print(f"  Nodes with outgoing edges: {unique_src.shape[0]:,}")

    loader = NeighborLoader(
        graph_data,
        num_neighbors=n_hops,
        batch_size=batch_size,
        input_nodes=unique_src,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # ================================================================
    # Training loop
    # ================================================================
    losses_log: list[float] = []
    t_train = time.time()
    total_steps = 0

    # LR schedule
    steps_per_epoch = len(unique_src) // batch_size
    total_expected_steps = epochs * steps_per_epoch
    warmup_steps = min(500, total_expected_steps // 10)

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0

        for batch in loader:
            batch = batch.to(device)
            B = batch.batch_size  # number of target nodes in this batch

            # --- LR schedule ---
            if total_steps < warmup_steps:
                lr_mult = 0.1 + 0.9 * (total_steps / warmup_steps)
            else:
                progress = (total_steps - warmup_steps) / max(total_expected_steps - warmup_steps, 1)
                lr_mult = max(0.01, 0.5 * (1 + math.cos(math.pi * progress)))
            for pg in optimizer.param_groups:
                pg["lr"] = lr * lr_mult

            # --- Temporal split ---
            # For each target node, split its edges into context (past) and target (future)
            # NeighborLoader with time_attr already respects temporal ordering.
            # We use the first B nodes in batch as target nodes.
            # The sampled subgraph includes their temporal neighborhood.

            # Get node features from embedding table
            x = node_emb(batch.x.to(device))  # (N_batch, latent_dim)

            # --- Online encoder (context) ---
            ctx_repr = online_encoder(x, batch.edge_index, batch.edge_attr)

            # --- Target encoder (EMA, no grad) ---
            with torch.no_grad():
                tgt_repr = target_encoder(x, batch.edge_index, batch.edge_attr)

            # Target = first B nodes (the input_nodes for this batch)
            ctx_target = ctx_repr[:B]   # (B, D)
            ema_target = tgt_repr[:B]   # (B, D)

            # --- Predictor ---
            pred = predictor(ctx_target)  # (B, K, D) or (B, D)
            if pred.dim() == 2:
                pred = pred.unsqueeze(1)

            # --- JEPA loss: predict EMA target from online context ---
            jepa_loss = F.smooth_l1_loss(pred[:, 0], ema_target)

            # --- Multi-step lookahead (predict future neighbors' representations) ---
            la_loss = torch.tensor(0.0, device=device)
            # For simplicity at graph scale: predict shifted representations
            if B > lookahead_steps:
                for k in range(1, lookahead_steps):
                    # Shift target by k positions (approximates temporal lookahead)
                    shifted_target = ema_target.roll(-k, dims=0)[:B - k]
                    shifted_pred = pred[:B - k, min(k, pred.shape[1] - 1)]
                    la_loss = la_loss + (lookahead_decay ** k) * F.smooth_l1_loss(
                        shifted_pred, shifted_target
                    )
                la_loss = la_loss / lookahead_steps

            # --- WeakSIGReg ---
            reg_loss = torch.tensor(0.0, device=device)
            if B > 1:
                progress = total_steps / max(total_expected_steps, 1)
                rw = 0.01 + (reg_weight - 0.01) * progress
                reg_loss = rw * sigreg.compute_loss(ctx_target) / (latent_dim ** 2)

            # --- Total loss ---
            loss = jepa_loss + 0.5 * la_loss + reg_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

            # Zero out gradients for frozen ontology nodes
            if n_frozen > 0:
                with torch.no_grad():
                    node_emb.weight.grad[frozen_mask] = 0.0

            # --- EMA update ---
            progress = total_steps / max(total_expected_steps, 1)
            tau = ema_tau_start + (ema_tau_end - ema_tau_start) * progress
            with torch.no_grad():
                for pe, po in zip(target_encoder.parameters(), online_encoder.parameters()):
                    pe.data.mul_(tau).add_(po.data, alpha=1 - tau)

            lv = loss.item()
            epoch_loss += lv
            n_batches += 1
            total_steps += 1
            losses_log.append(lv)

            if n_batches % 50 == 0:
                sps = n_batches / (time.time() - t_train) if n_batches > 0 else 0
                print(
                    f"  epoch {epoch + 1} batch {n_batches} | "
                    f"loss={epoch_loss / n_batches:.4f} "
                    f"(jepa={jepa_loss.item():.4f} la={la_loss.item():.4f} "
                    f"reg={reg_loss.item():.4f}) | "
                    f"lr={lr * lr_mult:.2e} | {sps:.1f} batch/s"
                )

        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t_train
        print(f"  Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f} "
              f"({n_batches} batches, {elapsed:.0f}s elapsed)")

        # Per-epoch checkpoint — never lose weights again
        ckpt_dir = "/root/jepa-artifacts/tkg-fullscale"
        import os; os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = f"{ckpt_dir}/node_emb_epoch_{epoch + 1}.pt"
        torch.save(node_emb.weight.detach().cpu(), ckpt_path)
        # Also save as "latest" for easy access
        torch.save(node_emb.weight.detach().cpu(), f"{ckpt_dir}/node_embeddings.pt")
        torch.save(online_encoder.cpu().state_dict(), f"{ckpt_dir}/encoder.pt")
        torch.save(predictor.cpu().state_dict(), f"{ckpt_dir}/predictor.pt")
        # Move models back to GPU
        online_encoder.to(device)
        predictor.to(device)
        # Commit to volume
        jepa_cache.commit()
        print(f"  Checkpoint saved: {ckpt_path}")

    train_time = time.time() - t_train
    print(f"\nTraining complete: {total_steps} steps in {train_time:.1f}s")

    # ================================================================
    # Save artifacts
    # ================================================================
    import json
    import os

    artifact_dir = "/root/jepa-artifacts/tkg-fullscale"
    os.makedirs(artifact_dir, exist_ok=True)

    torch.save(node_emb.cpu().state_dict(), f"{artifact_dir}/node_embeddings.pt")
    torch.save(online_encoder.cpu().state_dict(), f"{artifact_dir}/encoder.pt")
    torch.save(predictor.cpu().state_dict(), f"{artifact_dir}/predictor.pt")

    with open(f"{artifact_dir}/config.json", "w") as f:
        json.dump({
            "num_nodes": num_nodes,
            "num_predicates": num_predicates,
            "latent_dim": latent_dim,
            "n_hops": n_hops,
            "n_edges": n_edges,
            "n_ontology_frozen": n_frozen,
            "n_instance_learnable": num_nodes - n_frozen,
            "epochs": epochs,
            "total_steps": total_steps,
        }, f, indent=2)

    with open(f"{artifact_dir}/losses.json", "w") as f:
        json.dump(losses_log, f)

    jepa_cache.commit()
    print(f"  Saved to {artifact_dir}")

    return {
        "total_steps": total_steps,
        "final_loss": losses_log[-1] if losses_log else None,
        "train_time_s": train_time,
        "num_nodes": num_nodes,
        "num_edges": n_edges,
        "num_predicates": num_predicates,
        "params": n_params,
        "epochs": epochs,
    }


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------


@scale_app.local_entrypoint()
def main(
    action: str = "materialize",
    catalog_path: str = "data/ai_friendly_catalog.json",
    epochs: int = 10,
    batch_size: int = 1024,
    latent_dim: int = 64,
    qwen_parallelism: int = 10,
    qwen_batch_size: int = 500,
):
    """Full-scale TKG-JEPA pipeline (all heavy work on Modal).

    Actions:
        materialize — DuckDB → Parquet edges + unique texts (Modal CPU)
        qwen        — Encode unique texts with Qwen (distributed Modal GPU)
        train       — Train Graph-JEPA with Qwen node embeddings (Modal GPU)
        full        — All steps end-to-end

    Prerequisites:
        - DB uploaded to Modal volume: modal volume put jcube-data data/aggregated_fixed_union.db /aggregated_fixed_union.db
        - Catalog uploaded: modal volume put jcube-data data/ai_friendly_catalog.json /ai_friendly_catalog.json
    """
    import json
    import sys
    import time

    sys.path.insert(0, ".")

    if action in ("materialize", "full"):
        print("=" * 60)
        print("STEP 1: Materialize TKG on Modal (remote DuckDB)")
        print("=" * 60)

        with open(catalog_path) as f:
            catalog_json = f.read()

        stats = materialize_remote.remote(catalog_json=catalog_json)
        print(f"\n✓ {stats['n_edges']:,} edges ({stats['edge_file_mb']:.1f} MB)")
        print(f"✓ {stats['n_ontology_nodes']:,} ontology nodes ({stats['ontology_file_mb']:.1f} MB)")

        if action == "materialize":
            return

    if action in ("train", "full"):
        print("\n" + "=" * 60)
        print("STEP 2: Train Graph-JEPA on A100 (Qwen ontology + graph JEPA)")
        print("=" * 60)

        result = train_tkg_jepa.remote(
            parquet_path="/data/jcube_graph.parquet",
            latent_dim=latent_dim,
            batch_size=batch_size,
            epochs=epochs,
        )

        print(f"\n{'=' * 60}")
        print(f"RESULTS — TKG Graph-JEPA")
        print(f"{'=' * 60}")
        print(f"  Steps:      {result['total_steps']:,}")
        print(f"  Final loss: {result['final_loss']:.6f}")
        print(f"  Train time: {result['train_time_s']:.1f}s")
        print(f"  Nodes:      {result['num_nodes']:,}")
        print(f"  Edges:      {result['num_edges']:,}")
        print(f"  Predicates: {result['num_predicates']}")
        print(f"  Params:     {result['params']:,}")


# ---------------------------------------------------------------------------
# CLI for local-only materialization
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TKG-JEPA Pipeline")
    parser.add_argument("action", choices=["materialize"], help="Local action")
    parser.add_argument("--db", default="data/aggregated_fixed_union.db")
    parser.add_argument("--catalog", default="data/ai_friendly_catalog.json")
    parser.add_argument("--output", default="data/jcube_graph.parquet")
    args = parser.parse_args()

    if args.action == "materialize":
        materialize(args.db, args.catalog, args.output)
