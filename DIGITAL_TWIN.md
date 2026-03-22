# Digital Twin: Timestamp-Driven Database Intelligence

## The Method

The Digital Twin turns any DuckDB database into a living temporal model. Every row is an **event** that happened to an **entity** at a **point in time**. The system discovers these axes automatically, encodes rows into fixed-dimension vectors, and runs EventJEPA's hierarchical temporal aggregation to produce representations, detect patterns, and predict what comes next.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  DuckDB Database (local file or remote)                     │
│  417 tables, 58M rows, 52 hospital sources                  │
└─────────────┬───────────────────────────────────────────────┘
              │ Apache Arrow (zero-copy)
              ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 1: Schema Twin (digital_twin.py)                     │
│  - Table catalog, column profiles, FK discovery             │
│  - Domain grouping, source DB tracking                      │
│  - Static metadata layer                                    │
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 2: Temporal Twin (materializer.py)                   │
│                                                             │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ Entity      │  │ Column       │  │ Hash Projection   │  │
│  │ Discovery   │→ │ Encoder      │→ │ (count-sketch)    │  │
│  │ (2-phase)   │  │ (per-type)   │  │ → fixed dim=64    │  │
│  └─────────────┘  └──────────────┘  └─────────┬─────────┘  │
│                                                │            │
│                                                ▼            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ EventJEPA                                            │   │
│  │                                                      │   │
│  │  events sorted by timestamp                          │   │
│  │      │                                               │   │
│  │      ▼                                               │   │
│  │  adaptive temporal windowing (gap > median)          │   │
│  │      │                                               │   │
│  │      ▼                                               │   │
│  │  exponential-decay aggregation per window             │   │
│  │  w_i = exp(-α * (t_max - t_i))                       │   │
│  │      │                                               │   │
│  │      ▼                                               │   │
│  │  hierarchical levels (merge windows → re-partition)   │   │
│  │      │                                               │   │
│  │      ▼                                               │   │
│  │  representation + patterns + predictions             │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### The 2-Phase Materialization

The naive approach (load all rows, filter in Python) doesn't scale to 58M rows. The materializer uses a 2-phase strategy:

**Phase 1 — Entity Discovery** (lightweight, no data transfer):
```sql
-- For each of 118 INTERNACAO tables, count events per entity
SELECT CAST(ID_CD_INTERNACAO AS VARCHAR) AS eid, COUNT(*) AS cnt
FROM "agg_tb_capta_evolucao_caev" WHERE ID_CD_INTERNACAO IS NOT NULL
GROUP BY ID_CD_INTERNACAO
UNION ALL
SELECT CAST(ID_CD_INTERNACAO AS VARCHAR) AS eid, COUNT(*) AS cnt
FROM "agg_tb_fatura_itens_fait" WHERE ID_CD_INTERNACAO IS NOT NULL
GROUP BY ID_CD_INTERNACAO
-- ... 116 more tables
-- Then: GROUP BY eid ORDER BY SUM(cnt) DESC LIMIT 10
```

This finds the 10 richest admissions across all 118 tables without transferring any data.

**Phase 2 — Targeted Extraction** (Arrow, only matching rows):
```sql
-- For each table, only fetch rows for the 10 target entities
SELECT * FROM "agg_tb_capta_evolucao_caev"
WHERE CAST(ID_CD_INTERNACAO AS VARCHAR) IN ('1106','93','14319',...)
AND DH_CADASTRO IS NOT NULL
ORDER BY DH_CADASTRO
```

Result: 40,975 events for 10 entities from 118 tables in ~42 seconds.

---

## The Entity Hierarchy

Entities are ranked by semantic richness. The materializer picks the highest-ranked entity column available in each table:

| Rank | Entity Column | Type | Tables | What It Represents |
|------|--------------|------|--------|-------------------|
| 1 | `ID_CD_INTERNACAO` | INTERNACAO | 118 | Hospital admission — the richest lifecycle. Spans billing, clinical evolution, medications, lab exams, audit, predictions. |
| 2 | `ID_CD_PACIENTE` | PACIENTE | 36 | Patient — cross-admission timeline. Tracks a person across multiple hospital visits. |
| 3 | `ID_CD_FATURA` | FATURA | 13 | Invoice — billing lifecycle from creation through items, glosas (denials), and settlement. |
| 4 | `ID_CD_PESSOA` | PESSOA | 15 | Person (CRM) — contacts, communications, relationships. |
| 5 | `ID_CD_ORCAMENTO` | ORCAMENTO | 14 | Budget estimate — from request through itemization to approval. |
| 6 | `ID_CD_HOSPITAL` | HOSPITAL | 17 | Hospital — configuration, staffing, equipment across all its admissions. |
| 7 | `ID_CD_RELATORIO` | RELATORIO | 11 | Audit report — lifecycle of a clinical audit document. |
| 8 | `ID_CD_AUDITORIA` | AUDITORIA | 3 | Audit record — specific audit actions and findings. |
| 9 | `ID_CD_EVOLUCAO` | EVOLUCAO | 2 | Clinical evolution — day-by-day patient state changes. |

**Why this order matters:** A single `INTERNACAO` (#1106) generated 5,000 events across 54 tables over 9.7 years. That's the full story: who the patient is, every clinical note, every medication, every billing line item, every audit finding, every prediction of discharge date — all ordered by when it happened.

### Timestamp Priority

The materializer picks the best available timestamp per table:

| Priority | Column | Meaning |
|----------|--------|---------|
| 1 | `DH_CADASTRO` | Record creation time (394 tables) |
| 2 | `DH_ATUALIZACAO` | Last update time (164 tables) |
| 3 | `DH_VISITA` | Clinical visit timestamp |
| 4 | `DH_PROCEDIMENTO` | Procedure timestamp |
| 5 | `DH_ADMISSAO_HOSP` | Hospital admission time |
| 6+ | Any TIMESTAMP column | Fallback |

`DH_CADASTRO` is the backbone — it captures *when the system learned about this event*, which is the ground truth for temporal ordering.

---

## Column Encoding

Every non-ID, non-timestamp column becomes part of the embedding. The encoder handles heterogeneous types:

### Numeric (BIGINT, FLOAT, DECIMAL) → 1 dimension
```
encode(x) = sign(x) * log(1 + |x|) / 20
```
Log-scale normalization maps values to roughly [-1, 1]. This handles the extreme range in healthcare data (IDs in the millions, monetary values from 0.01 to 1M+).

### Timestamp (context timestamps, not the event timestamp) → 6 dimensions
```
encode(dt) = [sin(2π·hour/24), cos(2π·hour/24),    # time-of-day cycle
              sin(2π·dow/7),    cos(2π·dow/7),       # day-of-week cycle
              sin(2π·month/12), cos(2π·month/12)]    # seasonal cycle
```
Cyclical encoding preserves the fact that 23:00 is close to 01:00, December is close to January. A procedure at 2am vs 2pm carries clinical signal (emergency vs scheduled).

### Text/VARCHAR → 16 dimensions
```
encode(text) = L2_normalize(trigram_hash_accumulate(text, dim=16))
```
Character trigram hashing: for each 3-character window, hash to a bucket index and sign, accumulate, then L2-normalize. This creates a lightweight "fingerprint" of the text that preserves similarity (similar strings hash similarly).

### Boolean → 1 dimension
```
encode(True) = 1.0, encode(False) = 0.0
```

### NULL → zeros
Any null value produces a zero vector for its slot.

### Hash Projection (count-sketch)

When the raw encoding exceeds the target dimension (e.g., a table with 40 columns produces a 200-dim raw vector but we target dim=64), we project down using deterministic hash projection:

```python
for i, v in enumerate(raw_vec):
    h = ((i + 1) * 2654435761) & 0xFFFFFFFF   # Knuth multiplicative hash
    bucket = h % target_dim
    sign = +1 if (h >> 16) & 1 else -1
    output[bucket] += sign * v
```

This is a count-sketch — it preserves approximate inner products (and therefore cosine similarity) while reducing dimensionality. Different tables with different numbers of columns all produce the same 64-dim output.

---

## EventJEPA: The Temporal Engine (V-JEPA 2.1)

PR #4 merged V-JEPA 2.1 improvements: multi-level processing with intermediate representations at every hierarchical level, dense context loss, numpy-accelerated operations, modality-aware configuration, and a two-phase training schedule. All operations fall back to pure Python when numpy is absent.

### Multi-Level Processing (`process_multilevel`)

The core change: instead of only returning the final aggregated representation, every hierarchical level now emits its own intermediate representation. This enables deep self-supervision — regularization at every temporal scale.

```
Input:  [e1, e2, e3, e4, e5, e6, e7, e8, e9, e10]
                    ↓ sort by timestamp
                    ↓ apply modality offset (if modality_aware=True)
                    ↓ partition by gaps > median
         Window A: [e1, e2, e3]  Window B: [e4, e5]  Window C: [e6, e7, e8, e9, e10]
                    ↓ exponential-decay aggregate per window
                    ↓ w_i = exp(-α * (t_max - t_i))
Level 1 output: weighted_aggregate([agg_A, agg_B, agg_C])  ← INTRA-DAY patterns
                    ↓ re-partition [agg_A, agg_B, agg_C]
                    ↓ aggregate again
Level 2 output: weighted_aggregate([agg_AB, agg_C])         ← CROSS-DAY patterns
                    ↓ final aggregation
Level 3 output: final_representation                         ← CROSS-MONTH trajectory
```

**Each level captures a different temporal scale:**

| Level | Temporal Scale | What It Captures (Healthcare) | Example |
|-------|---------------|------------------------------|---------|
| 1 | Intra-day | Burst of activity within a single visit or shift | Morning labs + afternoon procedure + evening vitals |
| 2 | Cross-day | Patterns across days/weeks | Daily improvement trend, escalating medication changes |
| 3 | Cross-month | Long-arc trajectory | Readmission cycle, chronic condition progression |

```python
from event_jepa_cube import EventJEPA

jepa = EventJEPA(embedding_dim=64, num_levels=3)

# Get all intermediate levels
levels = jepa.process_multilevel(sequence)
# levels[0] = intra-day representation
# levels[1] = cross-day representation
# levels[2] = cross-month representation
# levels[3] = final aggregation

# Fuse into a single representation (element-wise mean-pooling)
fused = EventJEPA.fuse_multilevel(levels)
```

**Adaptive windowing**: gaps between events that exceed the median inter-event interval create window boundaries. This naturally segments hospital admissions into "bursts of activity" — admission day, daily rounds, procedure days, discharge planning.

**Exponential decay**: within each window, recent events count more. The alpha parameter can be configured per modality:
```
w_i = exp(-α * (t_max - t_i))
```

### Multi-Level Loss Fusion (`compute_multilevel_loss`)

Each hierarchical level gets its own regularization loss. The losses combine with configurable per-level weights:

```
L_total = L_pred + reg_weight * Σ_level (w_level * Regularizer(level_embeddings))
```

```python
from event_jepa_cube import EventJEPA, SIGReg

jepa = EventJEPA(
    embedding_dim=64,
    num_levels=3,
    regularizer=SIGReg(num_directions=32).compute_loss,
    reg_weight=0.05,
)

# Process a batch of sequences, collecting embeddings at each level
level_embeddings = [[], [], []]  # one list per level
for seq in batch:
    levels = jepa.process_multilevel(seq)
    for i, level_rep in enumerate(levels[:-1]):  # exclude final agg
        level_embeddings[i].append(level_rep)

# Multi-level loss: regularizer applied at EVERY level
# Weights increase with temporal scale — coarse patterns get more weight
loss = jepa.compute_multilevel_loss(
    level_embeddings,
    prediction_loss=pred_loss,
    level_weights=[0.2, 0.3, 0.5],  # coarse levels weighted more
)
```

**Why per-level regularization matters:** Without it, the final representation might look well-distributed, but intermediate levels can collapse. An admission that looks "normal" at the monthly level might have a collapsed intra-day representation — hiding the fact that all its daily patterns are identical. Per-level regularization prevents this.

### Dense Context Loss (`compute_dense_loss`)

Standard JEPA masks a region and predicts it. Dense context loss adds a second signal: every *context* token (non-masked) also gets a prediction target, weighted by its temporal distance to the masked region. Tokens close to the mask boundary get strong gradients; distant tokens get weak ones.

```
Distance weighting: w_i = λ / max(d_i, floor)
Where d_i = min temporal distance from context token i to any masked token
```

```python
# Split sequence into context (visible) and target (masked)
context_embs = [seq.embeddings[i] for i in context_indices]
context_ts = [seq.timestamps[i] for i in context_indices]
target_embs = [seq.embeddings[i] for i in target_indices]
mask_ts = [seq.timestamps[i] for i in target_indices]

# Dense loss: prediction_loss + distance-weighted context supervision
total_loss = jepa.compute_dense_loss(
    context_embeddings=context_embs,
    context_timestamps=context_ts,
    target_embeddings=target_embs,
    mask_timestamps=mask_ts,
    prediction_loss=pred_loss,
    lambda_coeff=0.5,       # context loss weight
    distance_floor=1.0,     # minimum distance (prevents division by zero)
)
```

**Healthcare intuition:** When predicting a masked "procedure day", the dense loss also trains on the admission day and the day after — with the admission day getting a weaker signal (it's temporally distant) and the day after getting a stronger one (it's right next to the mask boundary). This creates a smoother gradient landscape.

The context lambda ramps up during training via a warmup schedule:

```python
# Don't apply dense loss early in training — let the model stabilize first
lam = EventJEPA.context_lambda_schedule(
    current_step=step,
    warmup_start=50,   # start ramping at step 50
    warmup_end=100,    # full strength at step 100
    max_lambda=0.5,
)
```

### Position-Aware Prediction (`predict_next_positional`)

Standard `predict_next` extrapolates by integer steps. Position-aware prediction takes explicit target timestamps and modulates the trend with sinusoidal position encoding:

```python
# "What will this admission look like on these specific future dates?"
import datetime

now = datetime.datetime(2026, 3, 20).timestamp()
targets = [
    now + 7 * 86400,   # 1 week from now
    now + 30 * 86400,  # 1 month from now
    now + 90 * 86400,  # 3 months from now
]

predictions = jepa.predict_next_positional(sequence, target_timestamps=targets)
# predictions[0] = predicted state in 1 week
# predictions[1] = predicted state in 1 month
# predictions[2] = predicted state in 3 months
```

The position encoding uses sinusoidal functions at multiple frequencies:
```
pos_enc(dt, d) = sin(dt / 10000^(2i/d))  for even dimensions
                 cos(dt / 10000^(2i/d))  for odd dimensions
```

This allows the model to distinguish "predict 1 week ahead" from "predict 3 months ahead" even when the trend vector is the same.

### Modality-Aware Configuration

Different data sources (modalities) can have different temporal dynamics. A billing event every month is normal; a vital sign event every month is alarming. V-JEPA 2.1 adds per-modality configuration:

```python
jepa = EventJEPA(embedding_dim=64, num_levels=2, modality_aware=True)

# Clinical events: adaptive windowing, fast decay (recent matters most)
jepa.register_modality_config("clinical", temporal_resolution="adaptive", alpha=2.0)

# Billing events: fixed windowing, slow decay (all history matters)
jepa.register_modality_config("billing", temporal_resolution="fixed", alpha=0.5)

# Modality tokens: additive offset that identifies the source type
# (V-JEPA 2.1 style — no architectural change needed)
jepa.set_modality_offset("clinical", [0.1] * 64)
jepa.set_modality_offset("billing", [-0.1] * 64)
```

In the digital twin, the materializer encodes all rows with `modality="db_row"`. But you can split by domain group and process with different modality configs:

```python
# Process billing tables with billing-tuned parameters
billing_seq = EventSequence(embeddings=..., timestamps=..., modality="billing")
clinical_seq = EventSequence(embeddings=..., timestamps=..., modality="clinical")

rep_billing = jepa.process(billing_seq)    # uses alpha=0.5, fixed windows
rep_clinical = jepa.process(clinical_seq)  # uses alpha=2.0, adaptive windows
```

### NumPy Acceleration

All heavy operations dispatch to numpy when available (`numpy_ops.py`, 600+ lines):

| Operation | Pure Python | NumPy | Speedup |
|-----------|------------|-------|---------|
| Weighted aggregation | O(N*D) loop | BLAS matrix multiply | ~10-50x |
| Cosine similarity | O(D) loop | `np.dot` + norms | ~5-20x |
| Trend computation | O(N*D) loop | Vectorized diff + weighted mean | ~10-30x |
| Pattern detection | O(D) loop + sort | `np.abs`, `np.argsort` | ~5-15x |
| Streaming EMA batch | Sequential per-event | Vectorized `np.exp` + cumsum | ~20-50x |

The fallback is always available — `HAS_NUMPY` flag controls dispatch. Zero-dependency core guarantee is preserved.

```python
from event_jepa_cube import HAS_NUMPY
print(f"NumPy acceleration: {'ON' if HAS_NUMPY else 'OFF (pure Python fallback)'}")
```

### Detect Patterns (Z-Score Thresholding)

After aggregation, salient dimensions are detected via z-score (now numpy-accelerated):
```
salient = {i : |z_i| > 1.5}
```

In the digital twin context, salient dimensions correspond to encoding slots that are unusually active for this entity — flagging which aspects of its lifecycle are distinctive. An admission with many salient dimensions in the "billing" encoding range had an unusual billing pattern.

### CooldownSchedule (Two-Phase Training)

V-JEPA 2.1 training recipe: a primary phase (warmup → constant LR) followed by a cooldown phase with cosine-decaying LR and optionally increased input resolution.

```
LR
 │
 │     ┌──────────────────────────┐
 │    /                            \
 │   /  primary phase               \  cooldown
 │  / (warmup → constant)            \ (cosine decay)
 │ /                                   \___
 └─────────────────────────────────────────── steps
   0    12K         135K          147K
```

```python
from event_jepa_cube import CooldownSchedule

schedule = CooldownSchedule(
    primary_steps=135000,
    cooldown_steps=12000,
    warmup_steps=12000,
    warmup_lr_ratio=0.19,
    cooldown_start_lr_ratio=1.14,
    cooldown_end_lr_ratio=0.002,
    cooldown_resolution_scale=1.5,  # 1.5x sequence length during cooldown
)

for step in range(schedule.total_steps):
    lr = base_lr * schedule.get_lr_multiplier(step)
    seq_len = schedule.get_sequence_length(base_length=1000, step=step)
    # During cooldown: LR decays, sequence length increases
    # This forces the model to handle longer contexts at lower learning rates
```

**Healthcare application:** During primary training, process admission sequences up to 1000 events. During cooldown, increase to 1500 events with decaying LR — this teaches the model to handle the longest, most complex admissions (multi-year, 5000+ events) without destabilizing the learned representations.

---

## Regularizers

When training learnable predictors (MLPPredictor, TransformerPredictor) on top of EventJEPA embeddings, regularizers prevent representation collapse — where all entities get mapped to similar vectors, destroying the information.

### SIGReg (Sketched Isotropic Gaussian)

**What it does:** Forces the distribution of embeddings across a batch to look like a standard Gaussian N(0, I).

**How:** Projects embeddings onto M random directions, then uses the Epps-Pulley characteristic function test to measure how far each 1D projection is from N(0,1).

**When to use:** Default choice. Best for unsupervised or self-supervised training where you want maximally informative, non-collapsed representations.

**Healthcare example:**
```python
from event_jepa_cube import SIGReg, EventJEPA

# Train a predictor that forecasts admission trajectories
# SIGReg prevents all admissions from collapsing to the same representation
jepa = EventJEPA(embedding_dim=64, regularizer=SIGReg(num_directions=32).compute_loss, reg_weight=0.05)

# With SIGReg, the embedding space spreads out:
# - Short-stay admissions cluster in one region
# - ICU admissions in another
# - Chronic/long-stay in another
# Without SIGReg, they all collapse to a single point
```

### WeakSIGReg (Covariance Targeting)

**What it does:** Targets only the covariance matrix (not the full distribution), making it faster and more suitable when you already have some supervisory signal.

**How:** Sketches the empirical covariance and compares to the identity matrix via random projections.

**When to use:** When you have labeled data (e.g., "this admission resulted in readmission within 30 days") and the primary loss already provides gradient signal. WeakSIGReg acts as a stabilizer.

**Healthcare example:**
```python
from event_jepa_cube import WeakSIGReg

# Training a readmission predictor with labeled outcomes
# The classification loss provides direction, WeakSIGReg prevents collapse
regularizer = WeakSIGReg(sketch_dim=32)

# The loss function becomes:
# L = L_classification + 0.05 * WeakSIGReg(embeddings)
# This ensures the embedding space doesn't degenerate even when
# the classification loss has many local minima
```

### RDMReg (Rectified Distribution Matching)

**What it does:** Matches embeddings to a Rectified Generalized Gaussian — producing sparse, non-negative representations where many dimensions are exactly zero.

**How:** Applies ReLU to embeddings, then uses sliced 2-Wasserstein distance to match the distribution to a target with controlled sparsity.

**When to use:** When you want interpretable, sparse representations. Each non-zero dimension has a clear meaning ("this admission has high activity in the billing dimension but zero in pharmacy").

**Healthcare example:**
```python
from event_jepa_cube import RDMReg

# Build sparse admission profiles for audit prioritization
# 70% of dimensions will be zero — only the "active" aspects light up
regularizer = RDMReg(p=2.0, target_sparsity=0.7, num_projections=32)

# Result: admission 1106 has non-zero values in dims [4, 9, 13, 17, 22]
# → these map to clinical evolution, billing disputes, specialist reviews
# Admission 93 has non-zero values in dims [11, 35, 43, 49]
# → these map to pharmacy, ICU monitoring, discharge planning
# Sparse = interpretable = auditable
```

### Choosing a Regularizer

| Scenario | Regularizer | Why |
|----------|-------------|-----|
| Unsupervised temporal clustering | **SIGReg** | Maximum information, no collapse |
| Supervised prediction (readmission, cost) | **WeakSIGReg** | Stabilizer alongside primary loss |
| Audit prioritization, explainability | **RDMReg** | Sparse = each dimension is meaningful |
| Real-time streaming (StreamingJEPA) | None | O(1) updates don't batch well for regularization |

---

## Use Cases

### 1. Admission Lifecycle Monitoring

**Goal:** Track every hospital admission as a temporal entity, detect anomalies in real-time.

```python
from event_jepa_cube import Materializer

mat = Materializer("data/hospital.db", embedding_dim=64)
mat.connect()
mat.scan()

# Materialize the 100 longest-running admissions
result = mat.materialize("INTERNACAO", limit_entities=100)
result = mat.process(result, num_levels=2, num_prediction_steps=5)

# Each admission now has:
# - A 64-dim representation capturing its full history
# - Salient dimensions flagging what's unusual
# - 5-step predictions of where it's heading

# Compare to historical baselines:
for eid, timeline in result.timelines.items():
    rep = result.representations[eid]
    patterns = result.patterns[eid]
    predictions = result.predictions[eid]

    # An admission with 54 source tables and 5000 events is complex
    if timeline.event_count > 1000 and len(timeline.source_tables) > 30:
        print(f"Complex admission {eid}: {timeline.event_count} events "
              f"over {timeline.time_span_days:.0f} days")
        print(f"  Salient dims: {patterns[:10]}")
```

**Real result:** Admission #1106 — 5,000 events, 54 tables, 3,556 days. The salient dimensions reveal which subsystems are most active (billing disputes, clinical evolution, specialist reviews).

### 2. Patient Journey Across Admissions

**Goal:** Build a cross-admission timeline for each patient to detect patterns that span hospitalizations.

```python
# Patients appear in 36 tables
result = mat.materialize("PACIENTE", limit_entities=50)
result = mat.process(result, num_levels=3)  # 3 levels for longer timelines

for eid, tl in sorted(result.timelines.items(), key=lambda x: -x[1].time_span_days)[:5]:
    print(f"Patient {eid}: {tl.event_count} events over {tl.time_span_days/365:.1f} years")
    print(f"  Tables: {tl.source_tables}")
```

**Real result:** Patient #1639 — 940 events over 11.5 years across 13 tables. The temporal representation captures their entire healthcare journey: initial consultations, multiple admissions, billing cycles, follow-up audits.

### 3. Invoice Audit Prioritization

**Goal:** Score invoices by anomaly level to prioritize human review.

```python
import math

result = mat.materialize("FATURA", limit_entities=200)
result = mat.process(result)

# Rank by representation norm (unusual invoices have higher norms)
scored = []
for eid in result.representations:
    rep = result.representations[eid]
    norm = math.sqrt(sum(v*v for v in rep))
    n_patterns = len(result.patterns[eid])
    scored.append((eid, norm, n_patterns, result.timelines[eid].event_count))

# High norm + many salient dimensions = unusual billing pattern
scored.sort(key=lambda x: x[1] * x[2], reverse=True)
for eid, norm, pats, events in scored[:10]:
    print(f"Invoice {eid}: score={norm*pats:.1f}, events={events}")
```

### 4. Cross-Hospital Comparison

**Goal:** Compare how different source hospitals behave for the same entity type.

```python
# Materialize admissions from GHO-BRADESCO only
result_bradesco = mat.materialize(
    "INTERNACAO",
    limit_entities=50,
    source_db_filter="GHO-BRADESCO",
)
result_bradesco = mat.process(result_bradesco)

# Same for GHO-PETROBRAS
result_petrobras = mat.materialize(
    "INTERNACAO",
    limit_entities=50,
    source_db_filter="GHO-PETROBRAS",
)
result_petrobras = mat.process(result_petrobras)

# Compare average representations
import statistics
for dim in range(64):
    vals_b = [result_bradesco.representations[e][dim] for e in result_bradesco.representations]
    vals_p = [result_petrobras.representations[e][dim] for e in result_petrobras.representations]
    if vals_b and vals_p:
        diff = abs(statistics.mean(vals_b) - statistics.mean(vals_p))
        if diff > 0.5:  # significant divergence
            print(f"Dim {dim}: Bradesco={statistics.mean(vals_b):.3f} vs "
                  f"Petrobras={statistics.mean(vals_p):.3f} (delta={diff:.3f})")
```

### 5. Real-Time Admission Tracking with StreamingJEPA

**Goal:** Update admission representations as new events arrive, without reprocessing the full history.

```python
from event_jepa_cube import StreamingJEPA

# Initialize streaming processors for active admissions
streams = {}

def on_new_event(admission_id: str, embedding: list[float], timestamp: float):
    if admission_id not in streams:
        streams[admission_id] = StreamingJEPA(embedding_dim=64, alpha=1.0)

    stream = streams[admission_id]
    stream.update(embedding, timestamp)  # O(1) per event

    # Get current representation (no reprocessing)
    rep = stream.get_representation()
    patterns = stream.detect_patterns()

    if len(patterns) > 10:  # many salient dimensions = something unusual
        alert(f"Admission {admission_id}: {len(patterns)} salient dimensions detected")
```

### 6. Prediction Monitoring (Discharge Forecasting)

**Goal:** Use EventJEPA predictions to forecast when an admission will reach a "discharge-like" state.

```python
result = mat.materialize("INTERNACAO", limit_entities=20)
result = mat.process(result, num_prediction_steps=10)

for eid in result.predictions:
    preds = result.predictions[eid]
    rep = result.representations[eid]

    # Track how the prediction norm changes over forecast steps
    # Decreasing norm often indicates convergence toward a "settled" state (discharge)
    norms = [math.sqrt(sum(v*v for v in p)) for p in preds]
    trend = norms[-1] - norms[0]

    if trend < -0.5:
        print(f"Admission {eid}: prediction trend declining ({norms[0]:.2f} → {norms[-1]:.2f})")
        print(f"  Likely approaching discharge/resolution")
    elif trend > 0.5:
        print(f"Admission {eid}: prediction trend increasing ({norms[0]:.2f} → {norms[-1]:.2f})")
        print(f"  Likely escalating (transfer, complications)")
```

---

## API Reference

Start the server:
```bash
python -m event_jepa_cube.twin_api --db data/aggregated_fixed_union.db --port 8000
```

### Schema Twin Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/twin/connect` | POST | Connect to a DB, build schema twin |
| `/twin/snapshot` | GET | Full schema twin (all tables, columns, FKs) |
| `/twin/tables` | GET | List tables (filter by domain, min_rows, sort) |
| `/twin/table/{name}` | GET | Full column profiles for a table |
| `/twin/table/{name}/sample` | GET | Sample rows from a table |
| `/twin/domains` | GET | Domain groups with stats |
| `/twin/foreign-keys` | GET | Discovered FK relationships |
| `/twin/sources` | GET | All source databases |
| `/twin/query` | POST | Execute arbitrary SQL |
| `/twin/graph` | GET | Nodes + edges for graph visualization |
| `/twin/search` | GET | Search tables/columns by name |

### Temporal Twin Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/twin/entity-types` | GET | Discover entity types with table counts |
| `/twin/materialize` | POST | Materialize temporal lifecycles + run EventJEPA |
| `/twin/entity/{type}/{id}` | GET | Single entity timeline with full results |

### Materialize Request

```json
POST /twin/materialize
{
  "entity_type": "INTERNACAO",
  "limit_entities": 10,
  "limit_events_per_entity": 5000,
  "source_db_filter": "GHO-BRADESCO",
  "run_jepa": true,
  "num_levels": 2,
  "temporal_resolution": "adaptive",
  "num_prediction_steps": 3
}
```

### Materialize Response

```json
{
  "entity_type": "INTERNACAO",
  "tables_scanned": 118,
  "entities_found": 10,
  "total_events": 40975,
  "embedding_dim": 64,
  "duration_s": 44.2,
  "timelines": [
    {
      "entity_id": "1106",
      "entity_type": "INTERNACAO",
      "event_count": 5000,
      "time_span_days": 3556.2,
      "source_tables": ["agg_tb_capta_evolucao_caev", "agg_tb_fatura_itens_fait", ...],
      "representation_dim": 64,
      "representation_norm": 1.9579,
      "salient_dimensions": [4, 9, 13, 14, 15, 17, 22, 26, ...],
      "prediction_steps": 3
    }
  ]
}
```

---

## Performance

Tested on a 6GB DuckDB with 417 tables, 58M rows, 52 hospital sources.

| Operation | Time |
|-----------|------|
| Schema twin (no profiling) | 5s |
| Schema twin (full profiling + FK) | 57s |
| Entity type scan | 0.3s |
| Materialize 5 patients (4,917 events) | 1.6s |
| Materialize 10 admissions (40,975 events) | 42s |
| EventJEPA on 10 admissions (dim=64) | 1.6s |
| EventJEPA on 10 admissions (dim=2448, no projection) | 87s |

The hash projection (2448 → 64 dims) gives a **55x speedup** on EventJEPA with minimal information loss.
