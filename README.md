# Event-JEPA-Cube

**A Python framework for processing long, irregular event sequences and multi-semantic entity relationships using Joint Embedding Predictive Architecture (JEPA) principles.**

Event-JEPA-Cube addresses fundamental limitations of standard Transformer architectures — fixed context windows, quadratic memory scaling, and single-modality bias — by providing hierarchical temporal processing, multi-modal embedding support, and mathematically grounded regularization within a unified entity representation system.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-34%20passed-brightgreen.svg)]()

---

## Project Goals

1. **Process irregular event sequences at scale** — Handle 5K-10K+ events with O(m log m) memory complexity, compared to O(n^2) in standard Transformers, using hierarchical temporal aggregation that respects irregular timestamps.

2. **Unify multi-modal entity representations** — Combine text, image, audio, sensor, and behavioral embeddings into a single entity representation (the "Embedding Cube") with cross-modal relationship discovery via cosine similarity.

3. **Ground embeddings in theory** — Implement SIGReg, WeakSIGReg, and RDMReg regularizers from the latest JEPA research (LeJEPA, 2024-2025) that enforce mathematically optimal embedding distributions, replacing brittle heuristics like stop-gradients and EMA teachers.

4. **Stay lightweight and extensible** — Zero dependencies for the core library. PyTorch is optional (only needed for regularizers). Decorator-based plugin system for custom embeddings and models.

---

## Installation

```bash
# Core library (zero dependencies)
pip install -e .

# With PyTorch for JEPA regularizers
pip install -e ".[torch]"

# Development tools (pytest, ruff, mypy)
pip install -e ".[dev]"

# Everything
pip install -e ".[all]"
```

---

## Quick Start

```python
from event_jepa_cube import EventJEPA, EventSequence

# 1. Create an event sequence with embeddings and timestamps
sequence = EventSequence(
    embeddings=[[0.1, 0.2, 0.3], [0.0, 0.1, 0.0], [0.2, 0.2, 0.2]],
    timestamps=[1.0, 2.0, 3.0],
    modality="text"
)

# 2. Process with hierarchical temporal aggregation
processor = EventJEPA(embedding_dim=3, num_levels=2, temporal_resolution="adaptive")
representation = processor.process(sequence)

# 3. Detect salient patterns (z-score thresholding)
patterns = processor.detect_patterns(representation)

# 4. Predict next events (trend-based extrapolation)
predictions = processor.predict_next(sequence, num_steps=3)

print("Representation:", representation)  # Temporally-weighted aggregate
print("Patterns:", patterns)              # Indices of salient dimensions
print("Predictions:", predictions)        # Next predicted embeddings
```

Run the included example:

```bash
python example.py
```

---

## Core Components

### EventJEPA — Sequence Processor

Processes irregular event sequences through multi-level temporal windowing with exponential time-decay weighting.

```python
from event_jepa_cube import EventJEPA, EventSequence

# Simulated user journey: browsing, idle, then purchase
sequence = EventSequence(
    embeddings=[
        [0.8, 0.1, 0.0],   # browse product A
        [0.7, 0.2, 0.1],   # browse product B
        [0.1, 0.0, 0.0],   # idle
        [0.2, 0.9, 0.8],   # add to cart
        [0.1, 0.8, 0.9],   # purchase
    ],
    timestamps=[1.0, 2.0, 50.0, 51.0, 52.0],  # note the irregular gap
    modality="behavior"
)

processor = EventJEPA(
    embedding_dim=3,
    num_levels=2,              # 2 hierarchical levels
    temporal_resolution="adaptive"  # split at natural timestamp gaps
)

# Process: groups [browse, browse] and [idle, cart, purchase] into
# separate windows based on the gap, aggregates with recency weighting,
# then merges at the next level.
representation = processor.process(sequence)

# Detect which embedding dimensions are most salient
salient_dims = processor.detect_patterns(representation)

# Predict next 3 events by exponentially-weighted trend extrapolation
next_events = processor.predict_next(sequence, num_steps=3)
```

**How it works:**

| Parameter | Effect |
|-----------|--------|
| `num_levels` | Number of hierarchical aggregation passes. Level 1 groups events into windows; level 2 merges windows; etc. |
| `temporal_resolution` | `"adaptive"`: splits at gaps > median inter-event interval. `"fixed"`: equal-width time bins. |
| Time-decay | Within each window, recent events get exponentially higher weight: `w_i = exp(-alpha * (t_max - t_i))` |

### EmbeddingCube — Entity Relationship Manager

Manages multi-semantic entities and discovers relationships through cosine similarity across shared embedding modalities.

```python
from event_jepa_cube import EmbeddingCube, Entity

# Create entities with multi-modal embeddings
laptop = Entity(
    embeddings={
        "text": [0.9, 0.1, 0.0, 0.2],     # from product description
        "image": [0.3, 0.8, 0.1, 0.0],     # from product photo
        "behavior": [0.5, 0.5, 0.7, 0.1],  # from purchase patterns
    },
    hierarchy_info={"category": "electronics", "subcategory": "laptops"}
)

phone = Entity(
    embeddings={
        "text": [0.85, 0.15, 0.05, 0.18],  # similar text embedding
        "image": [0.2, 0.7, 0.2, 0.1],     # somewhat similar image
        "behavior": [0.6, 0.4, 0.8, 0.05], # similar purchase pattern
    },
    hierarchy_info={"category": "electronics", "subcategory": "phones"}
)

headphones = Entity(
    embeddings={
        "text": [0.1, 0.0, 0.9, 0.8],     # very different text
        "audio": [0.5, 0.5, 0.5, 0.5],     # different modality entirely
    },
    hierarchy_info={"category": "electronics", "subcategory": "audio"}
)

cube = EmbeddingCube()
cube.add_entity(laptop)
cube.add_entity(phone)
cube.add_entity(headphones)

# Discover relationships (cosine similarity >= threshold)
relationships = cube.discover_relationships(
    entity_ids=[laptop.id],
    threshold=0.7
)
# laptop <-> phone: high cosine similarity on shared modalities + hierarchy bonus
# laptop <-> headphones: only "text" is shared, low similarity — excluded

print(relationships)  # {laptop.id: [phone.id]}
```

**Similarity computation:**
1. Find shared embedding modalities between two entities
2. Compute cosine similarity per shared modality
3. Average across modalities
4. Add +0.1 bonus (capped at 1.0) if `hierarchy_info["category"]` matches
5. Include only if final similarity >= `threshold`

---

## JEPA Regularizers

Event-JEPA-Cube implements three regularizers from the latest JEPA research that enforce mathematically optimal embedding distributions. These require PyTorch.

### SIGReg — Sketched Isotropic Gaussian Regularization

From [LeJEPA (arXiv:2511.08544)](https://arxiv.org/abs/2511.08544). Enforces an isotropic Gaussian distribution on embeddings — proven to be the optimal distribution that minimizes worst-case downstream prediction risk.

**How it works:** Projects embeddings onto M random unit-norm directions, then compares each 1D projected distribution to N(0,1) using the Epps-Pulley characteristic function test via trapezoidal quadrature.

```python
from event_jepa_cube.regularizers import SIGReg

sigreg = SIGReg(
    num_directions=64,          # M random projection directions
    sigma=1.0,                  # Epps-Pulley weighting window
    num_quadrature_points=17,   # integration precision
)

# In your training loop:
embedding_batch = encoder(input_batch)  # (N, d) tensor
reg_loss = sigreg.compute_loss(embedding_batch)
total_loss = prediction_loss + 0.05 * reg_loss
total_loss.backward()
```

### WeakSIGReg — Covariance Regularization

From [Weak-SIGReg (arXiv:2603.05924)](https://arxiv.org/abs/2603.05924). A computationally cheaper variant that targets only the covariance matrix (not the full characteristic function), suitable as a general training stabilizer for supervised settings.

```python
from event_jepa_cube.regularizers import WeakSIGReg

weak = WeakSIGReg(sketch_dim=64)  # random sketch dimension K

# Loss = ||sketch(Cov(Z)) - sketch(I)||_F^2
reg_loss = weak.compute_loss(embedding_batch)
```

### RDMReg — Rectified Distribution Matching

From [Rectified LpJEPA (arXiv:2602.01456)](https://arxiv.org/abs/2602.01456). Generalizes SIGReg by matching embeddings to a Rectified Generalized Gaussian (RGG) distribution, enabling explicit control over sparsity (up to 95% zero entries).

```python
from event_jepa_cube.regularizers import RDMReg

rdmreg = RDMReg(
    p=2.0,                # distribution shape (2.0=Gaussian, 1.0=Laplacian)
    target_sparsity=0.5,  # 50% of entries will be zero
    num_projections=64,   # sliced Wasserstein projections
)

# Uses sliced 2-Wasserstein distance with ReLU rectification
reg_loss = rdmreg.compute_loss(embedding_batch)
```

### Choosing a Regularizer

| Regularizer | Use When | Complexity | Sparsity |
|-------------|----------|------------|----------|
| **SIGReg** | Self-supervised pretraining, replacing stop-gradients/EMA | O(N * M) | Dense |
| **WeakSIGReg** | Supervised training stabilization, preventing collapse | O(N * K) | Dense |
| **RDMReg** | Need sparse/interpretable features (anomaly detection, efficient inference) | O(N * M * log N) | Controllable |

### Integrated Usage

```python
from event_jepa_cube import EventJEPA
from event_jepa_cube.regularizers import SIGReg

processor = EventJEPA(
    embedding_dim=768,
    num_levels=3,
    regularizer=SIGReg(num_directions=64),
    reg_weight=0.05,  # lambda balancing prediction vs regularization
)

# Combined loss: L = L_pred + 0.05 * L_SIGReg
total_loss = processor.compute_regularized_loss(embeddings, prediction_loss)
```

---

## Extensibility

### Custom Embedding Types

Register custom embedding processors via decorators:

```python
from event_jepa_cube import register_embedding_type

@register_embedding_type("sensor")
class SensorEmbedding:
    def __init__(self, dim: int, sampling_rate: float):
        self.dim = dim
        self.sampling_rate = sampling_rate

    def encode(self, raw_signal):
        # Your encoding logic
        return embedding_vector
```

### Custom Relationship Models

Register custom models for the EmbeddingCube:

```python
from event_jepa_cube import register_model

@register_model("graph_attention")
class GraphAttentionModel:
    def __init__(self, input_dim, heads=4):
        self.input_dim = input_dim
        self.heads = heads

    def forward(self, entity_embeddings):
        # Your relationship modeling logic
        return attention_scores

# Load into a cube
cube.load_registered_model("graph_attention", input_dim=768, heads=4)
```

---

## Architecture Overview

```
event_jepa_cube/
├── __init__.py          # Public API (EventJEPA, EmbeddingCube, Entity, etc.)
├── sequence.py          # EventSequence & Entity dataclasses
├── event_jepa.py        # Hierarchical temporal processor
├── embedding_cube.py    # Multi-semantic entity manager
├── registry.py          # Plugin system (@register_embedding_type, @register_model)
└── regularizers.py      # SIGReg, WeakSIGReg, RDMReg (requires PyTorch)
```

### Design Principles

- **Zero-dependency core** — The core library uses only Python stdlib. No numpy, no torch. This keeps the install lightweight and the API portable.
- **Optional PyTorch** — Regularizers live in a separate module with a graceful `ImportError` fallback. Install with `pip install -e ".[torch]"` when needed.
- **Dataclass-first** — `EventSequence` and `Entity` are plain dataclasses with validation. No framework lock-in.
- **Plugin registries** — `@register_embedding_type` and `@register_model` decorators allow extending the framework without modifying source code.

### Theoretical Performance Characteristics

> The figures below are architectural targets based on JEPA's structural advantages
> over token-based Transformers. They are not measured benchmarks of this library.

| Metric | Token-Based Transformers | Event-JEPA-Cube | Why |
|--------|--------------------------|-----------------|-----|
| Max Sequence Length | ~2K-4K tokens | 5K-10K+ events | Hierarchical windowing avoids full-sequence attention |
| Memory | O(n^2) | O(m log m) | Window-based aggregation, not pairwise attention |
| Modality Support | Single / forced fusion | Native multi-modal | Embedding Cube stores per-modality embeddings natively |
| Regularization | Stop-gradients, EMA, schedulers | Single SIGReg objective | Mathematically grounded (LeJEPA Theorem 4) |

---

## Industry Applications

| Domain | Use Cases | Key Components |
|--------|-----------|----------------|
| **E-Commerce** | User journey modeling, product recommendations, dynamic pricing | EventJEPA for session sequences, EmbeddingCube for product entities |
| **Healthcare** | Patient journey analysis, treatment optimization, readmission prediction | Multi-level temporal aggregation over irregular clinical events |
| **Manufacturing / IoT** | Predictive maintenance, anomaly detection, supply chain events | Sensor event sequences with adaptive temporal resolution |
| **Security** | Intrusion detection, access pattern analysis, fraud detection | RDMReg for sparse anomaly representations, pattern detection |

---

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Lint and format
ruff check .
ruff format .

# Type check
mypy event_jepa_cube/
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full development guide.

---

## References

The regularizer implementations are based on the following papers:

- **LeJEPA** — Provable and Scalable Self-Supervised Learning Without the Heuristics. Introduces SIGReg and proves the isotropic Gaussian is the optimal JEPA embedding distribution. [arXiv:2511.08544](https://arxiv.org/abs/2511.08544)

- **Weak-SIGReg** — Covariance Regularization for Stable Deep Learning. Adapts SIGReg as a general optimization stabilizer for supervised training. [arXiv:2603.05924](https://arxiv.org/abs/2603.05924)

- **Rectified LpJEPA** — Joint-Embedding Predictive Architectures with Sparse and Maximum-Entropy Representations. Generalizes Gaussian JEPAs via RDMReg for controllable sparsity. [arXiv:2602.01456](https://arxiv.org/abs/2602.01456)

## Citation

```bibtex
@software{event_jepa_cube2024,
  author = {Agourakis, Dionisio Chiuratto},
  title = {Event-JEPA-Cube: Event Sequence Processing and Entity Relationships},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/josaum/jcube}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.
