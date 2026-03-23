# V5 Postmortem: GraphGPS + VICReg + 8-Loss Architecture

**Date:** 2026-03-23
**Duration:** ~18 hours (design → training → evaluation → kill)
**Cost:** ~$80 in A100 compute (Modal)
**Verdict:** Epoch 1 is production. Epoch 2+ degraded. Architecture partially successful.

---

## Executive Summary

V5 introduced SOTA graph learning components (GraphGPS, TGN memory, VICReg, 8 auxiliary losses, BGE-M3 anchors, progressive curriculum) to the JCUBE temporal knowledge graph. The spatial structure improvements were significant — anomaly detection improved 38%, semantic search improved 19%. However, the multi-loss architecture **backfired at epoch 2**: every downstream metric degraded when auxiliary topological losses (edge_type_pred, link_pred) activated.

**Key lesson:** Graph reconstruction losses conflict with temporal prediction. A model can't simultaneously optimize "memorize the graph structure" and "predict the future."

---

## What Worked

### 1. BGE-M3 Ontology Anchors (replacing Qwen 0.8B)

| Metric | Qwen (V4) | BGE-M3 (V5) |
|--------|-----------|-------------|
| Encoding time (54K texts) | 15 min | 30s (cached: 0s) |
| Model size | 752M params | 568M params |
| Attention | Causal (decoder) | Bidirectional (encoder) |
| Portuguese | Chinese-first tokenizer | Native multilingual |

BGE-M3 produces denser, more discriminative ontology embeddings because bidirectional attention sees the full context. The frozen anchors exert cleaner gravitational pull on instance nodes.

### 2. GraphGPS Backbone (replacing TransformerConv)

The 3-layer GPS (GatedGCN local + MultiheadAttention global + GraphNorm) was strictly superior to V4's 2-layer TransformerConv:

| Metric | V4 (TransformerConv) | V5 Epoch 1 (GPS) |
|--------|---------------------|-------------------|
| Type coherence (inter/intra) | 30x | 30x (maintained) |
| Anomaly top z-score | 20.54 | **28.34** (+38%) |
| Semantic search top-1 sim | 0.776 | **0.920** (+19%) |
| Cross-type linking | 0.694 | **0.900** (+30%) |
| Isotropy (active dims) | 64/64 | **64/64** (no collapse) |

The GPS global attention allows distant nodes to communicate directly — an admission at GHO-BRADESCO can attend to a similar admission at PASA without needing 5 hops to reach it. This explains the massive jump in semantic search similarity.

### 3. Source_db Scoping (cross-hospital ID collision fix)

V4 had phantom edges: `INTERNACAO_117926` from BRADESCO and `INTERNACAO_117926` from CNU were the same node. V5 prefixed all IDs with `source_db/`, creating proper isolation. This:
- Eliminated 971 spurious exam edges on admission #117926
- Doubled node count (17.4M → 35.2M) — each hospital's entities are now independent
- Made anomaly detection hospital-specific (anomalies are scoped to their source system)

### 4. C++ Fast Path (vocab build optimization)

| Operation | V4 (Python dict) | V5 (PyArrow C++) |
|-----------|------------------|-------------------|
| Vocab dedup (165M edges) | 19 min | **80s** (14x faster) |
| Index mapping (330M lookups) | included above | **15s** |

Using `pc.unique()` and `pc.index_in()` instead of Python dict comprehension eliminated the 19-minute bottleneck that blocked every run.

### 5. Infrastructure Caching

| Cache | First Run | Subsequent Runs |
|-------|-----------|-----------------|
| Ontology BGE-M3 embeddings | 30s | **0s** |
| Node vocabulary | 80s | **0s** |
| CSR adjacency hint | built | **reused** |
| Edge Parquet | 230s | **skipped** (file exists) |

Total cold-start saving: ~6 minutes per run after the first materialization.

### 6. VICReg Loss (epoch 1 only)

VICReg's variance-invariance-covariance decomposition produced a well-structured embedding space in epoch 1:
- **Invariance** dropped from 0.725 → 0.331 (online/EMA converging)
- **Variance** expanded from 0.038 → 0.098 (using more dimensions)
- **Covariance** increased from 1.0 → 2.4 (active decorrelation)

The embedding space after epoch 1 was well-spread, non-collapsed, and utilized all 64 dimensions.

---

## What Failed

### 1. Auxiliary Loss Conflict (THE critical failure)

V5 epoch 2 activated `edge_type_pred` (CE on 1041 predicates) and `link_pred` (bilinear BCE). Every downstream metric degraded:

| Metric | Epoch 1 | Epoch 2 | Delta |
|--------|---------|---------|-------|
| Glosa AUC (LightGBM) | **0.690** | 0.674 | -2.4% |
| LOS MAE | **6.86d** | 6.93d | +1.0% |
| Anomaly top z | **28.34** | 25.12 | -11.3% |
| Semantic search | **0.920** | 0.891 | -3.2% |

**Root cause analysis:**

Edge type prediction forces the model to learn: "given two node embeddings, what type of edge connects them?" This pushes embeddings to encode the **identity** of relationships (HAS_FATURA vs HAS_EXAME vs HAS_CID). But the primary JEPA objective (VICReg + lookahead) wants embeddings to encode the **trajectory** of clinical outcomes.

These objectives are fundamentally incompatible:
- Edge prediction reward: "make nodes that share edge types look similar"
- JEPA reward: "make nodes with similar futures look similar"

A patient who has the same procedures as another but a completely different outcome (one dies, one recovers) should have DIFFERENT embeddings for JEPA, but SIMILAR embeddings for edge prediction.

The gradient signals cancel. Epoch 2 spent its gradient budget trying to satisfy both, satisfying neither.

### 2. VICReg Variance Spikes

Periodic spikes in training (batches 950, 1150, 1650, 2400, 2500, etc.) where total loss jumped from ~10 to ~50:

```
batch 900  | total=18.69 | vicreg_var=0.058
batch 950  | total=57.27 | vicreg_var=0.368  ← 6x spike
batch 1000 | total=18.72 | vicreg_var=0.071  ← recovers
```

**Root cause:** The frozen ontology anchors (54K nodes with immutable BGE-M3 embeddings) have a different variance distribution than the trainable instance nodes. When a batch happens to sample a subgraph dominated by ontology nodes, the VICReg variance hinge (`std(z) >= 1`) fires aggressively because the frozen embeddings don't comply with the variance target.

The model recovers within ~50 batches because the gradient only affects trainable parameters, but the spikes waste compute and inject noise.

### 3. NeighborLoader Epoch Boundary Thrash

Despite `persistent_workers=True`, the PyG NeighborSampler without `pyg-lib` rebuilds its internal sampling structures at every epoch boundary:

| Epoch | Throughput at start | Throughput at steady state |
|-------|--------------------|-----------------------------|
| 1 | 3.3 → 5.8 b/s | 5.8 b/s |
| 2 | **0.0** → 0.9 b/s | ~2.0 b/s (never recovered fully) |
| 3 | **0.0** → 0.3 b/s | killed |

**Root cause:** `pyg-lib` (C++ sampler) doesn't have wheels for torch 2.10. The Python fallback sampler constructs a CSR-like index on first access, which takes ~30 minutes for 165M edges. `persistent_workers=True` keeps the worker processes alive but doesn't cache the sampling index across epoch boundaries.

**Impact:** Epoch 2 took ~10x longer than epoch 1. Epoch 3 was killed because 0.3 b/s meant ~30 hours per epoch.

### 4. VICReg Covariance Dominance

By batch 5000, the VICReg covariance term dominated the loss:
- Total: ~11.0
- VICReg cov: ~2.1 (19% of total)
- VICReg inv: ~0.39 (3.5%)
- Lookahead: ~0.175 (1.6%)

The covariance decorrelation consumed most of the loss signal, while the actual predictive objective (lookahead) was a tiny fraction. This means the model spent most of its gradient budget making dimensions independent rather than learning useful clinical patterns.

### 5. 128-dim Attempt OOM

The initial V5 deployment with `latent_dim=128` OOMed on a single A100-80GB:
```
RuntimeError: CUDA out of memory. Tried to allocate 16.78 GiB.
GPU has 79.25 GiB of which 11.47 GiB is free.
```

Embedding table (35.2M × 128 × 4 bytes) = 18 GB. With optimizer states (another 18 GB) and activations, it exceeded 80 GB. Fell back to `latent_dim=64`.

---

## What We Learned

### 1. Loss Function Minimalism

More losses ≠ better embeddings. The 8-loss MultiTaskHead with Kendall uncertainty weighting was theoretically elegant but practically harmful. The auxiliary losses (edge_pred, link_pred) actively degraded the primary objective.

**V6 decision:** Two losses only. Dense temporal lookahead + Weak-SIGReg. Zero graph reconstruction.

### 2. Epoch 1 is Often Sufficient

For spatial/structural tasks (anomaly detection, semantic search, type coherence), epoch 1 produced optimal results. Additional epochs only help if they add new signal (e.g., TGN memory). If they add conflicting signal (auxiliary losses), they degrade.

**V6 decision:** Use V5 epoch 1 as warm-start. Don't retrain the spatial structure — extend it with temporal memory.

### 3. VICReg vs Weak-SIGReg

VICReg's variance hinge (`std >= 1` per dimension) is too aggressive for heterogeneous graphs with frozen anchors. It causes periodic spikes and fights the natural distribution of ontology embeddings.

Weak-SIGReg (off-diagonal covariance penalty only) achieves decorrelation without forcing variance targets. It's permissive where VICReg is prescriptive.

**V6 decision:** Replace VICReg with Weak-SIGReg. Let the dense loss drive the representation; let SIGReg just prevent collapse.

### 4. Numeric Values Matter

The probes showed that CID at admission is the #1 predictor of discharge outcome (`cid_code_enc` importance = 1999, vs ~960 for embedding dims). But billing amounts (VL_TOTAL), quantities (NR_QTD_GLOSADO), and LOS are completely invisible — they're tokenized as text by BGE-M3 and lose their magnitude.

An admission billed R$500 and one billed R$500,000 have the same edge type `HAS_FATURA`. The graph can't distinguish them.

**V6 decision:** xVal-style Fourier encoding for numeric columns. `edge_feat = cat[pred_emb, Φ(Δt), numeric_enc(VL_TOTAL)]`.

### 5. Data Pipeline is King

The biggest wins came from data pipeline improvements, not architecture:
- Source_db scoping (+38% anomaly detection)
- C++ vocab build (14x faster)
- Caching (6 min saved per run)
- Correct CID column (`DS_DESCRICAO` not `DS_CID`)
- Real discharge types (FL_DESOSPITALIZACAO, not FL_TIPO_ALTA)
- IN_SITUACAO=2 filter (exclude active admissions from training)

**V6 decision:** Add numeric value column to Parquet. Use correct table relationships (TB_CAPTA_CID_CACI via ID_CD_INTERNACAO, TB_CAPTA_EVO_STATUS_CAES for discharge type).

---

## Production Configuration

**V5 Epoch 1 is the production model.** Weights at `/cache/tkg-v5/node_emb_epoch_1.pt` (8.4 GB, 35.2M × 64).

Used by:
- Anomaly detection reports (anomaly_report_v5_*.pdf)
- Admission-discharge analysis (admission_discharge_report_v5_*.pdf)
- Cluster reports (cluster_report_v5_*.pdf)
- Embedding algebra analysis (embedding_algebra_v5_*.pdf)
- Milvus push (api.getjai.com, collection: jcube_hospital_twin)
- Semantic search + similarity (exploit_twin.py)

### Downstream Results (V5 Epoch 1, LightGBM probes)

| Task | Metric | Value |
|------|--------|-------|
| Billing denial (Glosa) | ROC-AUC | **0.690** |
| Length of stay | MAE | **6.86 days** |
| Length of stay (medium 4-14d) | MAE | **2.3 days** |
| Anomaly detection | Top z-score | **28.34** |
| Semantic search | Top-1 cosine sim | **0.920** |
| Type coherence | Inter/intra ratio | **30x** |
| Isotropy | Active dimensions | **64/64** |
| Discharge prediction | Macro-AUC (per hospital) | **0.58-0.76** |

---

## Timeline

| Time | Event |
|------|-------|
| 11:20 | V5 deployed on A100 |
| 11:42 | Vocab + ontology loaded (caches hit) |
| 11:44 | Training started (batch 1, phase=warmup) |
| 12:05 | Batch 6000, loss 10.6, throughput 5.0 b/s |
| 14:15 | Epoch 1 complete (62,752 batches, 5391s) |
| 14:15 | Checkpoint saved: node_emb_epoch_1.pt (8.4 GB) |
| 14:16 | Phase 1 (local) activated: +edge_type_pred, +link_pred |
| 14:16 | Epoch 2 started |
| 17:30 | Epoch 2 complete |
| 17:30 | Checkpoint saved: node_emb_epoch_2.pt (8.4 GB) |
| 17:31 | Probes launched on both checkpoints |
| 18:00 | **Epoch 2 degradation confirmed** (all metrics worse) |
| 18:30 | Training killed. Epoch 1 designated as production. |

---

## Cost Breakdown

| Item | Cost |
|------|------|
| A100-80GB training (~7h) | ~$28 |
| A100-80GB failed runs (OOM, crashes, restarts) | ~$35 |
| CPU probe containers (~10 runs) | ~$5 |
| CPU report generation (~15 runs) | ~$8 |
| Volume storage (50 GB) | ~$2 |
| **Total V5 session** | **~$78** |

---

## V6 Implications

V5 proved that:
1. GraphGPS + BGE-M3 anchors produce excellent spatial embeddings ✅
2. TGN memory modules can be integrated without OOM ✅
3. Multi-objective training with conflicting losses degrades performance ❌
4. VICReg variance hinge causes instability with frozen nodes ❌
5. Numeric values in edges are invisible to the graph ❌

V6 keeps (1) and (2), fixes (3) with a single dense loss, fixes (4) with Weak-SIGReg, and fixes (5) with xVal numeric encoding. The architecture simplifies from 9 nn.Module classes + 8 losses to 10 nn.Module classes + 2 losses.

The gradient budget is unified: 100% temporal prediction, 0% graph reconstruction.
