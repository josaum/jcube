"""Modal-free entrypoint for V6 Dense Temporal JEPA training.

`scale_pipeline_v6.py` is decorated end to end for Modal (`@scale_app.function`,
`modal.Volume` mounts, `modal run` entrypoint). This module runs the SAME
trainer on local hardware — a Kubernetes/Kueue GPU Job, a Slurm allocation, or
a bare box — against local paths.

It reuses `V6Trainer` unchanged, so the DDP / TGN-gradient / sparse-embedding
fixes apply identically here.

Preflight refuses to start (rather than dying mid-epoch) on:
    - no CUDA device                  (e.g. job landed on a CPU-only node)
    - insufficient free disk          (checkpoints are tens of GB per epoch)
    - insufficient VRAM for the graph (estimated after the graph loads)
and it downgrades bf16 -> fp16 automatically on pre-Ampere GPUs (Turing has
no bfloat16).

Usage — single GPU:
    python -m event_jepa_cube.train_local \\
        --parquet-path /data/jcube_graph_v6.parquet \\
        --ontology-path /data/ontology_nodes.parquet \\
        --artifact-dir /cephfs/jcube/tkg-v6.2 \\
        --config '{"hospital_filter":"GHO-BRADESCO","epochs":2}'

Usage — multi-GPU on one node (torchrun sets RANK/WORLD_SIZE/LOCAL_RANK):
    torchrun --nproc_per_node=2 -m event_jepa_cube.train_local ...

Usage — image check only (no data, no GPU work; verifies pyg_lib):
    python -m event_jepa_cube.train_local --check-image
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from typing import Any

# Bytes of node-embedding + TGN state per node, at the V6.2 defaults, as a
# fraction of latent/tgn dims: fp32 weights only (SGD carries no momentum
# state, and embedding grads are sparse).
_BYTES_PER_FLOAT = 4
# Headroom over the estimated static footprint for activations, the encoder,
# AMP buffers, and allocator fragmentation.
_VRAM_HEADROOM = 1.35


class _LocalVolume:
    """No-op stand-in for a modal.Volume on local storage.

    V6Trainer calls .commit() after writing artifacts; on a local filesystem
    the write IS the commit.
    """

    def commit(self) -> None:
        return None

    def reload(self) -> None:
        return None


def _install_modal_stub() -> None:
    """Make `import modal` succeed without the Modal SDK installed.

    Lets the training image skip a cloud vendor's SDK entirely. Only the
    module-level surface `scale_pipeline_v6` touches at import time is
    stubbed — decorators become identity functions.
    """
    import types

    if "modal" in sys.modules:
        return
    try:  # real SDK present — use it, nothing to stub
        import modal  # noqa: F401

        return
    except ImportError:
        pass

    class _Chainable:
        """Accepts any attribute/call and returns itself (Image builder, App)."""

        def __getattr__(self, _name: str) -> Any:
            return self

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            # Used both as builder step (returns self) and as decorator
            # (@app.function(...) -> decorator -> original fn).
            if len(args) == 1 and not kwargs and callable(args[0]):
                return args[0]
            return self

    stub = types.ModuleType("modal")
    stub.App = lambda *a, **k: _Chainable()  # type: ignore[attr-defined]
    stub.Image = _Chainable()  # type: ignore[attr-defined]
    stub.Volume = _Chainable()  # type: ignore[attr-defined]
    stub.Secret = _Chainable()  # type: ignore[attr-defined]
    stub.enter = lambda *a, **k: lambda fn: fn  # type: ignore[attr-defined]
    stub.method = lambda *a, **k: lambda fn: fn  # type: ignore[attr-defined]
    stub.exit = lambda *a, **k: lambda fn: fn  # type: ignore[attr-defined]
    sys.modules["modal"] = stub


def check_image() -> dict[str, Any]:
    """Verify the runtime can actually sample graphs (the pyg_lib trap).

    Mirrors scale_pipeline_v6's `--action smoke`: a missing pyg_lib silently
    demotes NeighborLoader to the Python sampler, which cost V5 a 10x epoch
    slowdown.
    """
    import torch

    result: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        devices = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            devices.append(
                {
                    "name": props.name,
                    "vram_gb": round(props.total_memory / 1e9, 1),
                    "capability": f"{props.major}.{props.minor}",
                    "bf16": props.major >= 8,
                }
            )
        result["devices"] = devices

    try:
        import torch_geometric

        result["torch_geometric"] = torch_geometric.__version__
        result["with_pyg_lib"] = bool(torch_geometric.typing.WITH_PYG_LIB)
    except Exception as exc:  # noqa: BLE001 — report, don't crash the check
        result["torch_geometric_error"] = repr(exc)
    return result


def preflight_disk(artifact_dir: str, min_free_gb: float) -> None:
    """Refuse to start when the artifact volume cannot hold the checkpoints.

    A full-graph epoch writes a ~27GB checkpoint plus an ~18GB embedding
    tensor. Filling a shared volume mid-run is worse than not starting.
    """
    os.makedirs(artifact_dir, exist_ok=True)
    free_gb = shutil.disk_usage(artifact_dir).free / 1e9
    if free_gb < min_free_gb:
        raise SystemExit(
            f"PREFLIGHT FAIL: {artifact_dir} has {free_gb:.1f} GB free, "
            f"need >= {min_free_gb:.1f} GB.\n"
            "  Point --artifact-dir at a volume with room (NOT a root disk near "
            "capacity), or lower --min-free-gb if you know the run is small."
        )
    print(f"  [preflight] {artifact_dir}: {free_gb:.1f} GB free (>= {min_free_gb:.1f} required)")


def preflight_device(cfg: Any) -> None:
    """Require a GPU, and downgrade bf16 on pre-Ampere hardware.

    Turing (sm_75, e.g. RTX 8000) has no bfloat16; the V6 default amp_dtype
    would break or silently degrade there.
    """
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "PREFLIGHT FAIL: no CUDA device visible.\n"
            "  This job needs a GPU node. Check the node selector / GPU resource "
            "request — CPU-only nodes will never schedule this successfully."
        )

    caps = [torch.cuda.get_device_properties(i).major for i in range(torch.cuda.device_count())]
    if cfg.amp_dtype == "bfloat16" and min(caps) < 8:
        names = [torch.cuda.get_device_properties(i).name for i in range(torch.cuda.device_count())]
        print(
            f"  [preflight] bf16 unsupported on {names} (compute capability < 8.0); falling back to amp_dtype=float16"
        )
        cfg.amp_dtype = "float16"


def estimate_vram_gb(num_nodes: int, cfg: Any) -> float:
    """Estimated static VRAM for the embedding table + TGN memory, with headroom."""
    static_bytes = float(num_nodes * (cfg.latent_dim + cfg.tgn_dim) * _BYTES_PER_FLOAT)
    return static_bytes / 1e9 * _VRAM_HEADROOM


def preflight_vram(num_nodes: int, cfg: Any) -> None:
    """Warn (loudly) when the loaded graph likely exceeds device memory."""
    import torch

    needed = estimate_vram_gb(num_nodes, cfg)
    available = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  [preflight] est. static VRAM {needed:.1f} GB vs {available:.1f} GB on device 0")
    if needed > available:
        print(
            f"  [preflight] *** WARNING: {num_nodes:,} nodes at latent_dim={cfg.latent_dim} "
            f"likely will not fit. Use a hospital_filter subgraph, lower latent_dim, "
            "or a larger GPU. Continuing — expect CUDA OOM."
        )


def build_config(config_json: str, artifact_dir: str, parquet: str, ontology: str) -> Any:
    """Build a V6Config from CLI overrides and local paths."""
    if "modal" not in sys.modules:
        _install_modal_stub()
    from event_jepa_cube.scale_pipeline_v6 import V6Config

    cfg = V6Config()
    if config_json:
        overrides = json.loads(config_json)
        unknown = [k for k in overrides if not hasattr(cfg, k)]
        if unknown:
            raise SystemExit(f"Unknown config keys: {unknown}")
        for key, value in overrides.items():
            setattr(cfg, key, value)

    cfg.parquet_path = parquet
    cfg.ontology_path = ontology
    cfg.artifact_dir = artifact_dir
    # Local runs have no V5/V6 Modal volumes to warm-start from unless the
    # operator staged them; point both at the artifact dir so the existing
    # lookup finds staged checkpoints or falls back to cold start.
    cfg.v5_artifact_dir = artifact_dir
    cfg.v6_artifact_dir = artifact_dir
    return cfg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local (Modal-free) V6 JEPA training")
    parser.add_argument("--check-image", action="store_true", help="verify runtime + pyg_lib, then exit")
    parser.add_argument("--parquet-path", default="/data/jcube_graph_v6.parquet")
    parser.add_argument("--ontology-path", default="/data/ontology_nodes.parquet")
    parser.add_argument("--artifact-dir", default="/artifacts/tkg-v6.2")
    parser.add_argument("--config", default="", help="JSON overrides for V6Config")
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=64.0,
        help="refuse to start with less free space at --artifact-dir (default 64)",
    )
    args = parser.parse_args(argv)

    _install_modal_stub()

    if args.check_image:
        report = check_image()
        for key, value in report.items():
            print(f"  {key}: {value}")
        ok = report.get("with_pyg_lib") and report.get("cuda_available")
        if not report.get("cuda_available"):
            print("\n  *** No CUDA device — this node cannot run training.")
        if not report.get("with_pyg_lib"):
            print("\n  *** pyg_lib NOT active — NeighborLoader will fall back to the")
            print("  *** Python sampler (the V5 10x slowdown). Fix the image first.")
        if ok:
            print("\n  Image OK.")
        return 0 if ok else 1

    import torch
    import torch.distributed as dist

    from event_jepa_cube import scale_pipeline_v6 as spv
    from event_jepa_cube.scale_pipeline_v6 import V6Trainer, _build_nn_modules

    # Local filesystem: writes are durable, no volume commits needed. The
    # trainer only ever calls .commit()/.reload() on these, so a no-op stands
    # in for the Modal Volume handle.
    for _volume_attr in ("jepa_cache", "data_vol", "hf_cache"):
        setattr(spv, _volume_attr, _LocalVolume())

    # torchrun/Slurm set these; absent means single process.
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    is_main = rank == 0

    cfg = build_config(args.config, args.artifact_dir, args.parquet_path, args.ontology_path)
    cfg.use_ddp = world_size > 1
    cfg.num_gpus = world_size

    if is_main:
        print("=" * 60)
        print("V6 Dense Temporal JEPA — LOCAL runner (no Modal)")
        print("=" * 60)
        preflight_disk(args.artifact_dir, args.min_free_gb)
    preflight_device(cfg)

    if world_size > 1:
        # env:// picks up MASTER_ADDR/MASTER_PORT from torchrun; NCCL interface
        # selection (e.g. NCCL_SOCKET_IFNAME) stays an operator concern.
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)

    try:
        nn_modules = _build_nn_modules()
        trainer = V6Trainer(
            cfg, nn_modules, cfg.parquet_path, cfg.ontology_path, rank=local_rank, world_size=world_size
        )
        if is_main:
            preflight_vram(trainer.num_nodes, cfg)
        result = trainer.train()
    finally:
        if world_size > 1:
            dist.destroy_process_group()

    if is_main:
        print(f"\n{'=' * 60}")
        print("RESULTS")
        print(f"{'=' * 60}")
        for key in ("total_steps", "final_loss", "final_dense_loss", "train_time_s", "num_nodes"):
            print(f"  {key}: {result.get(key)}")
        print(f"  artifacts: {cfg.artifact_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
