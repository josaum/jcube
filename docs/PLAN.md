# Implementation Plan — Event-JEPA-Cube

## Overview

Transform the current stub/reference implementation into a functional JEPA
framework grounded in the latest theoretical advancements: **SIGReg** (LeJEPA),
**Weak-SIGReg**, and **RDMReg** (Rectified LpJEPA). Work is organized into
6 phases, ordered by dependency and priority.

---

## Phase 1: Project Foundation (P0 gaps)

Establish the build, test, and packaging infrastructure that everything else
depends on.

### 1.1 Add `pyproject.toml`

Create a proper Python package with build metadata:

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "event-jepa-cube"
version = "0.1.0"
requires-python = ">=3.9"
dependencies = []  # core has zero deps

[project.optional-dependencies]
torch = ["torch>=2.0"]          # for SIGReg/RDMReg implementations
dev   = ["pytest>=7.0", "pytest-cov", "ruff", "mypy"]
all   = ["event-jepa-cube[torch,dev]"]
```

- Core library stays zero-dependency (numpy-only math via stdlib or optional numpy)
- Torch is an optional dependency for the regularization modules
- Dev tools grouped for contributors

### 1.2 Add `.gitignore`

Standard Python `.gitignore` covering `__pycache__/`, `*.egg-info/`, `.env`,
`dist/`, `build/`, `.mypy_cache/`, `.pytest_cache/`, `.ruff_cache/`.

### 1.3 Add test infrastructure

```
tests/
├── __init__.py
├── test_sequence.py        # EventSequence & Entity validation
├── test_event_jepa.py      # EventJEPA process/detect/predict
├── test_embedding_cube.py  # EmbeddingCube add/discover/load
├── test_registry.py        # register/get embedding types & models
└── test_regularizers.py    # SIGReg, Weak-SIGReg, RDMReg (Phase 3)
```

Add `pytest.ini` or `[tool.pytest.ini_options]` in `pyproject.toml`.

### 1.4 Fix README / API alignment

- Update Quick Start to use `EventSequence` objects (not raw kwargs)
- Remove `embedding_types` parameter that doesn't exist
- Add note that performance benchmarks are theoretical targets
- Fix citation with actual author and repo URL

**Files changed:** `pyproject.toml` (new), `.gitignore` (new), `tests/` (new),
`README.md` (edit)

---

## Phase 2: Core Algorithm Implementation (P1 gaps — replace stubs)

Replace stub methods with real algorithms. These are prerequisite to the
JEPA-specific regularization work in Phase 3.

### 2.1 Temporal-aware processing in `EventJEPA.process()`

Replace simple averaging with hierarchical temporal aggregation:

```python
def process(self, sequence: EventSequence) -> List[float]:
    """Hierarchical temporal aggregation respecting irregular timestamps."""
    # 1. Sort by timestamp (handle out-of-order events)
    # 2. Partition into temporal windows based on temporal_resolution:
    #    - "adaptive": use timestamp gaps to find natural breakpoints
    #    - "fixed": equal-width time bins
    # 3. For each level in num_levels:
    #    a. Aggregate embeddings within each window (weighted by recency)
    #    b. Apply exponential decay weighting based on time deltas
    #    c. Merge windows for next level (coarser granularity)
    # 4. Return final aggregated representation
```

Key design decisions:
- Exponential time-decay weighting: `w_i = exp(-alpha * (t_max - t_i))`
- Adaptive windowing: split at gaps > median inter-event interval
- Multi-level: each level doubles the window size

### 2.2 Real pattern detection in `EventJEPA.detect_patterns()`

Replace index sorting with actual pattern detection:

```python
def detect_patterns(self, representation: List[float]) -> List[int]:
    """Detect salient dimensions via z-score thresholding."""
    # 1. Compute mean and std of representation values
    # 2. Compute z-scores for each dimension
    # 3. Return indices where |z-score| > threshold (default 1.5)
    # Falls back to top-k if fewer than min_patterns found
```

### 2.3 Meaningful prediction in `EventJEPA.predict_next()`

Replace last-embedding repetition with trend-based extrapolation:

```python
def predict_next(self, sequence: EventSequence, num_steps: int = 1) -> List[List[float]]:
    """Predict next embeddings using exponentially-weighted moving trend."""
    # 1. Compute pairwise deltas between consecutive embeddings
    # 2. Weight deltas by recency (exponential decay on timestamps)
    # 3. Extrapolate: last_embedding + step * weighted_trend
    # 4. Return list of num_steps predicted embeddings
```

### 2.4 Cosine similarity in `EmbeddingCube.discover_relationships()`

Replace category string matching with actual embedding similarity:

```python
def discover_relationships(self, entity_ids, threshold=0.5):
    """Discover relationships via cosine similarity across shared modalities."""
    # 1. For each entity_id, get entity
    # 2. For each other entity, compute similarity:
    #    a. Find shared embedding modalities
    #    b. Cosine similarity per shared modality
    #    c. Average across modalities
    # 3. Include if similarity >= threshold
    # 4. Optionally boost if hierarchy_info overlaps (additive bonus)
```

### 2.5 Error handling for `load_registered_model()`

Raise `KeyError` instead of silently returning `None`.

**Files changed:** `event_jepa_cube/event_jepa.py`, `event_jepa_cube/embedding_cube.py`

---

## Phase 3: JEPA Regularization Module (new — core contribution)

Implement the three regularizers from the latest research as a new module.
This is the most architecturally significant phase.

### 3.1 New module: `event_jepa_cube/regularizers.py`

```python
# Module structure:

class SIGReg:
    """Sketched Isotropic Gaussian Regularization (LeJEPA).

    Enforces isotropic Gaussian distribution on embeddings via
    random projections and Epps-Pulley characteristic function matching.

    Loss = (1/M) * sum_{a in A} EP({a^T z_n})

    where EP compares empirical CF to N(0,1) CF: phi(t) = exp(-t^2/2)
    """

    def __init__(self, num_directions: int = 64, sigma: float = 1.0):
        # num_directions (M): number of random unit-norm projections
        # sigma: Epps-Pulley weighting window width

    def compute_loss(self, embeddings: Tensor) -> Tensor:
        # 1. Sample M random directions from unit hypersphere S^{d-1}
        # 2. Project embeddings: z_proj = A^T @ embeddings  (M x N)
        # 3. For each projection, compute Epps-Pulley statistic:
        #    a. Compute empirical CF: phi_hat(t) = (1/N) sum exp(i*t*z_j)
        #    b. Compare to Gaussian CF: phi(t) = exp(-t^2/2)
        #    c. EP = N * integral |phi_hat(t) - phi(t)|^2 * w(t) dt
        #    d. Approximate integral via trapezoidal rule (17 points, [-5,5])
        # 4. Return mean EP across all M directions


class WeakSIGReg:
    """Covariance-targeting variant of SIGReg for supervised settings.

    Targets covariance matrix instead of full characteristic function.
    More computationally efficient, suitable as training stabilizer.

    Loss = ||sketch(Cov(Z)) - sketch(I)||_F^2
    """

    def __init__(self, sketch_dim: int = 32):
        # sketch_dim: dimension of random sketch matrix

    def compute_loss(self, embeddings: Tensor) -> Tensor:
        # 1. Center embeddings: Z_c = Z - mean(Z)
        # 2. Compute sketched covariance: S @ (Z_c^T Z_c / N) @ S^T
        # 3. Compare to sketched identity: S @ I @ S^T
        # 4. Return Frobenius norm of difference


class RDMReg:
    """Rectified Distribution Matching Regularization (Rectified LpJEPA).

    Matches embeddings to Rectified Generalized Gaussian distribution
    for sparse, non-negative representations.

    Generalizes SIGReg: when rectification removed and p=2, recovers SIGReg.
    """

    def __init__(self, p: float = 2.0, target_sparsity: float = 0.0,
                 num_projections: int = 64):
        # p: Lp norm parameter (2.0 = Gaussian, 1.0 = Laplacian)
        # target_sparsity: fraction of expected zero entries [0, 1)
        # num_projections: number of random slicing directions

    def compute_loss(self, embeddings: Tensor) -> Tensor:
        # 1. Apply ReLU rectification: z_rect = max(0, z)
        # 2. Sample random projections c ~ Unif(S^{d-1})
        # 3. For each projection:
        #    a. Project: z_proj = c^T @ z_rect, y_proj = c^T @ y_target
        #    b. Sort both in ascending order
        #    c. Sliced Wasserstein: L = (1/B) ||sort(z_proj) - sort(y_proj)||^2
        # 4. Return mean loss across projections

    def _sample_rgg(self, n: int, d: int) -> Tensor:
        # Sample from Rectified Generalized Gaussian with params (mu, sigma, p)
        # mu derived from target_sparsity via inverse CDF
```

### 3.2 Integrate regularizers into `EventJEPA`

Add optional regularization to the processing pipeline:

```python
class EventJEPA:
    def __init__(self, ..., regularizer=None, reg_weight=0.05):
        self.regularizer = regularizer  # SIGReg, WeakSIGReg, or RDMReg
        self.reg_weight = reg_weight

    def compute_regularized_loss(self, embeddings, prediction_loss):
        """Combined JEPA loss: L = L_pred + lambda * L_reg"""
        if self.regularizer is None:
            return prediction_loss
        reg_loss = self.regularizer.compute_loss(embeddings)
        return prediction_loss + self.reg_weight * reg_loss
```

### 3.3 Register regularizers in the registry

```python
@register_model('sigreg')
class SIGReg: ...

@register_model('weak_sigreg')
class WeakSIGReg: ...

@register_model('rdmreg')
class RDMReg: ...
```

### 3.4 Pure-Python fallback implementations

For users without PyTorch, provide numpy/stdlib math fallbacks for the core
computations (characteristic function, covariance sketching, sorted Wasserstein).
These will be slower but functional.

**Files changed:** `event_jepa_cube/regularizers.py` (new),
`event_jepa_cube/event_jepa.py` (edit), `event_jepa_cube/__init__.py` (edit)

---

## Phase 4: Testing & Quality (P1-P2 gaps)

### 4.1 Comprehensive test suite

```
tests/
├── test_sequence.py         # Dataclass validation, edge cases
├── test_event_jepa.py       # Temporal processing, pattern detection, prediction
├── test_embedding_cube.py   # Similarity computation, threshold behavior, errors
├── test_registry.py         # Register/retrieve, name collisions, missing keys
├── test_regularizers.py     # SIGReg, WeakSIGReg, RDMReg correctness:
│   ├── test_sigreg_gaussian_embeddings_low_loss
│   ├── test_sigreg_collapsed_embeddings_high_loss
│   ├── test_weak_sigreg_identity_covariance
│   ├── test_rdmreg_sparsity_control
│   └── test_rdmreg_recovers_sigreg_when_p2
└── test_example.py          # Smoke test for example.py
```

### 4.2 CI/CD via GitHub Actions

`.github/workflows/ci.yml`:
- Matrix: Python 3.9, 3.10, 3.11, 3.12
- Steps: install, ruff lint, mypy type-check, pytest with coverage
- Optional: torch tests only on 3.11 (to limit CI cost)

### 4.3 Linter and formatter config

Add to `pyproject.toml`:
```toml
[tool.ruff]
line-length = 120
target-version = "py39"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]

[tool.mypy]
python_version = "3.9"
warn_return_any = true
```

**Files changed:** `tests/` (new files), `.github/workflows/ci.yml` (new),
`pyproject.toml` (edit)

---

## Phase 5: Documentation & Developer Experience (P2-P3 gaps)

### 5.1 Update README.md

- Fix Quick Start to match real API
- Add regularizer usage examples:
  ```python
  from event_jepa_cube import EventJEPA, SIGReg

  processor = EventJEPA(
      embedding_dim=768,
      regularizer=SIGReg(num_directions=64),
      reg_weight=0.05
  )
  ```
- Add section on choosing regularizer (SIGReg vs WeakSIGReg vs RDMReg)
- Update performance claims with citations to LeJEPA paper
- Fix citation block with actual author/URL

### 5.2 Add `CONTRIBUTING.md`

Cover: dev setup, running tests, code style, PR process.

### 5.3 Add `CLAUDE.md`

```markdown
# CLAUDE.md
## Build & Test
- Install: `pip install -e ".[dev]"`
- Test: `pytest tests/ -v`
- Lint: `ruff check .`
- Type check: `mypy event_jepa_cube/`

## Architecture
- Core has zero external dependencies
- Regularizers require PyTorch (optional)
- All public API in __init__.py
```

### 5.4 Add `CHANGELOG.md`

Start with v0.1.0 covering the initial implementation.

**Files changed:** `README.md` (edit), `CONTRIBUTING.md` (new), `CLAUDE.md` (new),
`CHANGELOG.md` (new)

---

## Phase 6: Advanced Features (future work)

Not in scope for initial implementation but documented for roadmap:

- **Learnable predictor networks** — replace trend extrapolation with small MLPs
- **Multi-view augmentation** — event sequence augmentation strategies (temporal
  jitter, subsequence sampling, modality dropout)
- **Streaming/online mode** — process events incrementally without re-computing
  full sequence
- **Benchmarks directory** — reproducible performance measurements with
  synthetic and real datasets
- **Dockerfile** — containerized dev environment

---

## Dependency Graph

```
Phase 1 (Foundation)
  └─► Phase 2 (Core Algorithms)
       └─► Phase 3 (Regularizers)  ◄── Main contribution
            └─► Phase 4 (Testing & CI)
                 └─► Phase 5 (Docs)
                      └─► Phase 6 (Future)
```

Phases 1-2 can be partially parallelized (tests can be written alongside
algorithm implementation). Phase 3 is the most technically demanding and
represents the novel contribution.

---

## Estimated Scope

| Phase | New/Changed Files | Complexity |
|-------|-------------------|------------|
| 1     | 4 new, 1 edit     | Low        |
| 2     | 2 edits           | Medium     |
| 3     | 1 new, 2 edits    | High       |
| 4     | 6 new, 1 edit     | Medium     |
| 5     | 4 new/edits       | Low        |
| 6     | Future            | —          |
