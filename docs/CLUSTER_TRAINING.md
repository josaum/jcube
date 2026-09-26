# Running jcube V6 training on a local GPU cluster

The V6 pipeline (`scale_pipeline_v6.py`) is Modal-decorated end to end. This
document covers the Modal-free path: `event_jepa_cube/train_local.py`, which
runs the same `V6Trainer` on local hardware (Kubernetes/Kueue, Slurm, or a
bare box) and is submittable as a normal GPU Job.

## Why a separate entrypoint

`scale_pipeline_v6.py` decorates its functions with `@scale_app.function(gpu=...)`,
mounts `modal.Volume`s at `/data` and `/cache`, and its `__main__` refuses with
"requires Modal". A pod running `modal run` would try to provision *Modal*
containers, not use the GPU it was scheduled on. `train_local.py` reuses the
trainer directly against local paths.

It also installs a stub for `import modal` when the SDK is absent, so the
training image does **not** need a cloud vendor's SDK. (If the real SDK is
installed, the stub steps aside.)

## Preflight — what it refuses to do

Training is expensive and shared clusters are shared. The runner fails fast on:

| Check | Why |
|-------|-----|
| No CUDA device | The job landed on a CPU-only node. It would never train; better to exit than to look busy. |
| Free space `< --min-free-gb` at `--artifact-dir` | A full-graph epoch writes a ~27GB checkpoint **plus** an ~18GB embedding tensor. Filling a shared volume mid-epoch is worse than not starting. |
| Estimated VRAM > device VRAM | Warns before the run rather than OOM-ing an hour in. |
| `bf16` on pre-Ampere GPUs | Turing (sm_75, e.g. RTX 8000) has no bfloat16. The V6 default `amp_dtype="bfloat16"` is auto-downgraded to `float16`. |

## Order of operations

**1. Image check first — cheap, one GPU, no data.**

```bash
python -m event_jepa_cube.train_local --check-image
```

Must report `cuda_available: True` and `with_pyg_lib: True`. A missing
`pyg_lib` silently demotes `NeighborLoader` to the Python sampler — the exact
failure that cost V5 a 10x epoch slowdown. Exits non-zero if either fails, so
it works directly as a gating Job.

**2. Single-hospital smoke.**

```bash
python -m event_jepa_cube.train_local \
    --parquet-path /data/jcube_graph_v6.parquet \
    --ontology-path /data/ontology_nodes.parquet \
    --artifact-dir /artifacts/tkg-v6.2-smoke \
    --config '{"hospital_filter":"GHO-BRADESCO","epochs":2}'
```

Watch the `[DIAG]` lines in the first steps: `emb_grad_norm` must be nonzero
(embeddings are learning) and `sparse_rows` should be a few thousand.

**3. The lookahead A/B** — repeat step 2 with
`"lookahead_mode":"same_entity"` and compare probes. See
[V63_PREFLIGHT.md](V63_PREFLIGHT.md) for the full gate.

**4. Multi-GPU**, only once single-GPU is proven:

```bash
torchrun --nproc_per_node=2 -m event_jepa_cube.train_local ...
```

`torchrun` supplies `RANK`/`WORLD_SIZE`/`LOCAL_RANK`; the runner initializes
NCCL with `env://`. Set `NCCL_SOCKET_IFNAME` to the GPU-fabric interface —
without it NCCL may bind the management NIC and collapse throughput.

## Kubernetes / Kueue

`deploy/k8s-gpu-job.yaml` renders two Jobs: the image check and the training
run. Substitute the `${...}` placeholders.

The `nvidia.com/gpu` request is what keeps the job off CPU-only nodes — a node
advertising zero GPU units can never satisfy it. Don't pin training to a
storage/CPU node; it will sit `Pending` indefinitely.

`suspend: true` lets Kueue admit the job rather than racing the scheduler, and
`backoffLimit: 0` avoids silently restarting a half-written epoch.

## Sizing

Static VRAM (embedding table + TGN memory, before activations) is roughly:

```
num_nodes x (latent_dim + tgn_dim) x 4 bytes x 1.35 headroom
```

At the V6.2 production dims (`latent_dim=128`, `tgn_dim=64`), the full
35.2M-node graph needs **~27GB before activations** — it does not fit a single
24GB-class card. On heterogeneous or modest GPUs, use `hospital_filter` to
train a subgraph, or lower `latent_dim`. Aggregate VRAM across different cards
is not a single pool; DDP replicates the model per device rather than sharding
it.

## Artifacts and storage health

Checkpoints are large and written every epoch. Point `--artifact-dir` at a
volume with real headroom and a healthy backing pool — not a root filesystem
near capacity, and not a storage cluster in a degraded/backfill-full state.
`--min-free-gb` (default 64) is the guard; raise it for full-graph runs.
