# Changelog

## [0.3.0] - 2026-07-26

### Added
- `event_jepa_cube/train_local.py` — Modal-free entrypoint running the same
  `V6Trainer` on local GPU hardware (K8s/Kueue, Slurm, bare box). Installs a
  stub for `import modal` when the SDK is absent, so the training image needs
  no cloud vendor SDK.
- Preflight guards that refuse to start rather than failing mid-epoch: no CUDA
  device, insufficient free disk at `--artifact-dir`, estimated VRAM above
  device capacity; plus automatic bf16 -> fp16 downgrade on pre-Ampere GPUs
  (Turing has no bfloat16).
- `--check-image`: CUDA + pyg_lib verification, non-zero exit, usable as a
  gating Job before committing GPU hours.
- `deploy/k8s-gpu-job.yaml` — Kueue-queued training + image-check Jobs.
- `docs/CLUSTER_TRAINING.md` — local-cluster runbook and sizing math.


## [0.2.1] - 2026-07-25

### Fixed (V6 training pipeline — see docs/V63_PREFLIGHT.md)
- DDP now actually syncs encoder gradients (DDP wrapper; predictor/TGN
  grads all-reduced manually) — previously 4-GPU runs trained divergent
  encoders and kept only rank 0's
- TGN msg_fn/GRU now train (batch integration inside the gradient path);
  previously they ran only under no_grad and stayed at random init
- Node embedding gradients are sparse (removes an 18GB dense grad per step)
- Checkpoint latest is a JSON pointer; dropped duplicate per-epoch writes
  (~90GB/epoch → ~45GB/epoch of volume churn)
- Deterministic numeric-column priority in materialization (was set
  iteration — hash-randomized across processes)
- Rank-0-only ontology encoding/caching with DDP barriers
- Materialize step respects --parquet-path/--ontology-path
- Removed the no-op CSR "cache"; honest causality/curriculum comments

### Added
- `lookahead_mode` config: "cross_entity" (legacy V6.2 objective) vs
  "same_entity" (time-cutoff, entity-grounded, causal) for the V6.3 A/B
- `--action smoke`: cheap image pre-flight (pyg_lib presence, sampler run)
  before paying for H100 hours
- `docs/V63_PREFLIGHT.md`: audit findings, fixes, and the gated launch order
- Tests for the pure pipeline logic (numeric priority, config round-trip,
  checkpoint key cleaning, curriculum bounds)

## [0.2.0] - 2026-07-24

### Added
- `NormReg` regularizer — representation-norm regularization from BiJEPA
  (arXiv:2603.00049); closes the norm-explosion and collapse-to-zero escapes
  of symmetric prediction, composable with SIGReg/WeakSIGReg/RDMReg
- `BiJEPATrainer` — bidirectional cycle-consistent JEPA training
  (forward + mirrored backward predictor, cycle-consistency losses,
  NormReg, optional SIGReg-family regularization)
- `make_sliding_windows` helper in `predictors` (extracted from
  `PredictorTrainer.prepare_dataset`, shared with `BiJEPATrainer`)
- `docs/SOTA_2026.md` — July 2026 JEPA SOTA gap analysis and V6.3
  experiment plan

## [0.1.0] - 2026-03-14

### Added
- Core `EventJEPA` processor with hierarchical temporal aggregation
- `EmbeddingCube` for multi-semantic entity relationship discovery
- `EventSequence` and `Entity` dataclasses
- Decorator-based extension registries (`@register_embedding_type`, `@register_model`)
- JEPA regularizers: `SIGReg`, `WeakSIGReg`, `RDMReg`
- Runnable example in `example.py`
- Project packaging via `pyproject.toml`
- CI/CD via GitHub Actions
- Comprehensive test suite
