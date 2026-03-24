# JCUBE Digital Twin: Temporal Knowledge Graph JEPA

## Overview

JCUBE transforms a 5.8 GB DuckDB healthcare database (417 tables, 58M rows, 52 hospital sources) into a **continuous 128-dimensional physics engine** where every entity — patients, admissions, invoices, procedures, diagnoses, hospitals — has a learned position in latent space.

**Input:** Raw relational database (DuckDB)
**Output:** `(35.2M nodes, 128)` embedding tensor — one vector per entity

Vector distance = operational similarity. Cosine similarity replaces SQL JOINs.

---

## Architecture (V6 — Current)

```
┌─────────────────────────────────────────────────────────────────┐
│  DuckDB (5.8 GB)  →  165.6M RDF Triples (Parquet, 882 MB)     │
│  417 tables           (subject, predicate, object, time, value) │
│  52 hospital sources  Source_db-scoped node IDs                 │
└───────────────────────────────┬─────────────────────────────────┘
                                │ PyArrow C++ (zero-copy)
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  Graph Construction                                             │
│                                                                 │
│  35.2M nodes  ─  207 entity types  ─  1,041 predicate types    │
│                                                                 │
│  Node types:    INTERNACAO (1.2M), PACIENTE (1.2M),            │
│                 FATURA (1.9M), CID (457K), TUSS (193K), ...    │
│                                                                 │
│  Edge features: pred_emb(64) + Φ(Δt)(16) + numeric(32) = 112  │
│                 ↑              ↑             ↑                  │
│                 predicate      temporal      xVal Fourier       │
│                 embedding      encoding      (VL_TOTAL, etc.)   │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  Heterogeneous Initialization                                   │
│                                                                 │
│  54K ontology nodes  →  BGE-M3 (568M, bidirectional, pt-BR)    │
│  (CID, TUSS, etc.)     Frozen anchors — semantic gravity wells  │
│                                                                 │
│  35.1M instance nodes → Learnable embeddings (128-dim)          │
│  (patients, admissions)  Warm-start from V5 epoch 1 (64→128)   │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  V6 Training: Dense Temporal World Model                        │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  GraphGPS Encoder (3 layers)                            │    │
│  │  ├─ GatedGCN (local message passing)                    │    │
│  │  ├─ MultiHeadAttention (global, 8 heads)                │    │
│  │  ├─ GraphNorm + residual + FFN                          │    │
│  │  └─ Causal masking (future edges masked)                │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  TGN Memory (per-node GRU, CPU-resident, 9 GB)         │    │
│  │  ├─ Message: Linear(mem_src || mem_dst || edge_feat)    │    │
│  │  ├─ Update: GRUCell(message, memory)                    │    │
│  │  └─ Activated in Phase 1 (temporal, epoch 3+)           │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  Loss = L_dense_lookahead + 0.04 * L_weak_sigreg               │
│                                                                 │
│  L_dense: predict future event representations in latent space  │
│           with continuous time decay (7-day half-life)          │
│           W(Δt) = exp(-ln(2)/τ * Δt)                           │
│                                                                 │
│  L_sigreg: off-diagonal covariance penalty only                 │
│            prevents collapse without forcing variance targets    │
│                                                                 │
│  NO graph reconstruction. NO edge classification.               │
│  100% gradient budget on temporal prediction.                   │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  Output: (35.2M, 128) embedding tensor                         │
│                                                                 │
│  Applications:                                                  │
│  ├─ Anomaly detection (z-score from centroid)                  │
│  ├─ Semantic search (cosine similarity, 300ms)                 │
│  ├─ Billing denial prediction (LightGBM, AUC 0.69)            │
│  ├─ Length of stay estimation (MAE 2.3d for medium stays)      │
│  ├─ Discharge type prediction (per-hospital LightGBM)          │
│  ├─ Embedding algebra (hospital DNA, trajectory vectors)        │
│  ├─ Entity resolution (cross-hospital patient matching)         │
│  └─ Milvus vector search (api.getjai.com integration)           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Pipeline

### Step 1: Triple Materialization (DuckDB → Parquet)

Metaprogrammed SQL generates 1,129 SELECT statements from the AI-friendly catalog. Each table with 2+ entity columns + timestamp produces RDF-style triples:

```sql
SELECT source_db || '/' || 'ID_CD_INTERNACAO_' || CAST(ID_CD_INTERNACAO AS VARCHAR) AS subject_id,
       'HAS_PACIENTE_VIA_agg_tb_capta_internacao_cain' AS predicate,
       source_db || '/' || 'ID_CD_PACIENTE_' || CAST(ID_CD_PACIENTE AS VARCHAR) AS object_id,
       EPOCH(DH_CADASTRO) AS t_epoch,
       VL_TOTAL AS numeric_value
FROM agg_tb_capta_internacao_cain
WHERE ID_CD_INTERNACAO IS NOT NULL AND ID_CD_PACIENTE IS NOT NULL
```

**Key design decisions:**
- Node IDs scoped by `source_db/` to prevent cross-hospital ID collision
- `COPY ... TO ... (PARQUET, ZSTD, ROW_GROUP_SIZE 1000000)` for out-of-core execution
- Numeric values (VL_TOTAL, VL_GLOSA, NR_QTD) preserved as edge attributes
- Result: 165.6M edges, 882 MB ZSTD-compressed Parquet

### Step 2: Vocabulary Construction (PyArrow C++)

```python
all_nodes = pa.chunked_array(subj.chunks + obj.chunks)
unique_nodes = pc.unique(all_nodes)  # C++ dedup, 80s (was 19 min in Python)
```

35.2M unique nodes, 1,041 predicate types, 207 entity types.

### Step 3: Ontology Encoding (BGE-M3)

54K dictionary/ontology nodes (CID codes, TUSS procedures, categories) encoded with BGE-M3 (568M params, bidirectional, pt-BR native). CLS pooling → 1024-dim → project to 128-dim. Cached on Modal volume — subsequent runs skip encoding entirely.

### Step 4: Graph-JEPA Training (A100-80GB)

**Two-phase curriculum:**

| Phase | Epochs | Hops | TGN | Sampling |
|-------|--------|------|-----|----------|
| Foundation | 0-2 | [15, 10] | Off | 2% |
| Temporal | 3-9 | [15, 10, 5] | On | 5% |

Same two losses the entire time. Only change is TGN activation.

**Optimizer split:** SGD for embeddings (no momentum buffers — saves 36 GB), AdamW for encoder/predictor.

---

## Key Components

### NumericEncoder (xVal-style)

Continuous values (billing amounts, quantities) encoded via learnable Fourier frequencies instead of text tokenization:

```python
x_norm = log1p(|x|) * sign(x)
phases = x_norm * learnable_frequencies
output = Linear(cat[sin(phases), cos(phases)])
```

An admission billed R$500 and one billed R$500,000 now have DIFFERENT edge features. In V5, they were identical (same predicate type `HAS_FATURA`).

### DenseTemporalPredictor

Given context state + time query Δt, predicts the future latent state:

```python
z_pred = predictor(z_context, Δt)
loss = MSE(z_pred, ema_target) * exp(-decay * Δt)
```

Continuous time decay (7-day half-life): ICU events 10 min apart get weight ~1.0; outpatient visits 6 months apart get weight ~0.0. Respects the bursty, non-uniform nature of healthcare data.

### Weak-SIGReg

Off-diagonal covariance penalty only. No variance hinge (which caused instability in V5):

```python
cov = (z_centered.T @ z_centered) / (N - 1)
loss = ||cov - diag(cov)||²_F / dim
```

Forces dimensions to be independent (no redundant features) without dictating their scale.

### TGN Memory

Per-node GRU memory (64-dim, CPU-resident, 9 GB). Tracks each entity's evolving state:

```
When admission A interacts with patient B at time t:
  msg = Linear(cat[mem_A, mem_B, edge_feat])
  mem_A = GRU(msg, mem_A)
  mem_B = GRU(msg, mem_B)
```

Activated in Phase 1 (epoch 3+) after the spatial foundation is stable.

---

## Version History

| Version | Architecture | Loss | Dim | Key Result |
|---------|-------------|------|-----|------------|
| V1-V3 | Hash projection + EventJEPA | L2 | 64 | Proof of concept |
| V4 | TransformerConv + LoRA Qwen + graph context | JEPA + lookahead + WeakSIGReg | 64 | Glosa AUC 0.690, anomaly z=28.3 |
| V5 | GraphGPS + BGE-M3 + VICReg + 8 losses | VICReg + 6 auxiliary | 64 | Epoch 1 excellent, epoch 2 degraded (loss conflict) |
| **V6** | **GraphGPS + BGE-M3 + Dense Lookahead + Weak-SIGReg + NumericEncoder + TGN** | **Dense + SIGReg (2 losses only)** | **128** | **Clean monotonic convergence, no epoch degradation** |

### V5 → V6 Changes

**Removed** (caused V5 epoch 2 degradation):
- VICReg variance hinge (caused oscillation with frozen anchors)
- Edge type prediction (1041-class CE — fought temporal prediction)
- Link prediction (bilinear BCE — graph reconstruction ≠ temporal prediction)
- Temporal ordering, contrastive, DGI, dense context losses
- 4-phase curriculum (replaced by 2-phase)
- Kendall uncertainty weighting (replaced by fixed λ)

**Added:**
- NumericEncoder (xVal Fourier for billing amounts)
- DenseTemporalPredictor (context + time query → future state)
- Continuous time decay (7-day half-life, replaces step-based decay)
- SGD for embeddings (saves 36 GB VRAM)
- Node sampling (2%/5% — sample efficient, not brute force)

---

## Downstream Applications

### Anomaly Detection
```bash
python -m event_jepa_cube.exploit_twin --graph data/jcube_graph.parquet \
    --weights data/weights/node_embeddings.pt anomalies INTERNACAO -n 20
```
Z-score from entity-type centroid. Top anomalies at z>20 in production.

### Semantic Search
```bash
python -m event_jepa_cube.exploit_twin similar ID_CD_INTERNACAO_53873 -k 10
```
Cosine similarity on 128-dim vectors. Sub-second on 1.2M admissions.

### Predictive Probes (LightGBM)
- Billing denial (Glosa): ROC-AUC 0.69
- Length of stay: MAE 2.3 days (medium stays)
- Discharge type: per-hospital macro-AUC 0.58-0.76

### Reports (Modal CPU → LaTeX → PDF)
- Anomaly report: per-hospital, per-source, with AI interpretation
- Admission-discharge analysis: LightGBM per hospital, business narrative
- Embedding algebra: hospital DNA, trajectory vectors, entity resolution

### Mycelia Integration
- Push to Milvus (api.getjai.com) for agent-accessible vector search
- Oxigraph: RDF triples queryable via SPARQL
- OLL alignment: bridge ontology embeddings ↔ operational embeddings

---

## Infrastructure

| Component | Technology |
|-----------|-----------|
| Database | DuckDB 1.5 |
| Graph framework | PyTorch Geometric 2.7 |
| GNN backbone | Custom GraphGPS (GatedGCN + MultiHeadAttention) |
| Temporal memory | Custom TGN (GRU, CPU-resident) |
| Ontology encoder | BGE-M3 (BAAI, 568M params) |
| Training compute | Modal A100-80GB (single GPU) |
| Report generation | Modal CPU + pdflatex |
| Vector search | Milvus via Mycelia API |
| Data format | ZSTD Parquet (event stream) |

---

## Discharge Type Reference

The discharge type is determined by:
1. `IN_SITUACAO = 2` in `agg_tb_capta_internacao_cain` (confirms discharge)
2. Last record per internação in `agg_tb_capta_evo_status_caes` (ORDER BY `DH_CADASTRO DESC`, rn=1)
3. `FL_DESOSPITALIZACAO` maps to `DS_FINAL_MONITORAMENTO` via `agg_tb_capta_tipo_final_monit_fmon`

Categories: ALTA_NORMAL (72%), EM_CURSO (13%), ALTA_COMPLEXA (3.5%), OBITO (2.5%), ADMINISTRATIVO (0.5%), TRANSFERENCIA (0.3%)

## CID Reference

CID codes are in `agg_tb_capta_cid_caci`:
- `ID_CD_INTERNACAO` — links to admission
- `ID_CD_EVOLUCAO` — links to clinical evolution
- `ID_CD_CID` — the CID code ID
- `DS_DESCRICAO` — the CID description text (NOT `DS_CID`)
