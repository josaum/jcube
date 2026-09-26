"""Tests for the Modal-free local training entrypoint.

The GPU path can't run here; these cover the preflight logic and the
Modal-independence that makes the cluster image possible.
"""

import subprocess
import sys
import textwrap

import pytest

from event_jepa_cube.train_local import (
    _install_modal_stub,
    build_config,
    estimate_vram_gb,
    preflight_disk,
)

# Run in a fresh interpreter: installing the stub mutates sys.modules, which
# would leak into other tests in this session. A subprocess also reproduces
# the real condition — a cluster image with no Modal SDK installed at all.
_MODAL_FREE_IMPORT = textwrap.dedent(
    """
    import builtins, sys
    _real = builtins.__import__
    def blocked(name, *a, **k):
        if name == "modal" or name.startswith("modal."):
            raise ImportError("simulated: modal not installed")
        return _real(name, *a, **k)
    builtins.__import__ = blocked
    sys.modules.pop("modal", None)

    from event_jepa_cube.train_local import _install_modal_stub
    _install_modal_stub()
    builtins.__import__ = _real

    from event_jepa_cube import scale_pipeline_v6 as spv
    # Decorated entrypoints must survive as plain callables, not Modal handles.
    assert callable(spv.materialize_remote), "materialize_remote not callable"
    assert callable(spv.train_tkg_jepa_v6), "train_tkg_jepa_v6 not callable"
    assert spv.V6Config().version == "v6.2"
    print("MODAL_FREE_IMPORT_OK")
    """
)


class TestModalIndependence:
    """The cluster image must not need a cloud vendor's SDK to train locally."""

    def test_pipeline_imports_without_modal_sdk(self):
        proc = subprocess.run(
            [sys.executable, "-c", _MODAL_FREE_IMPORT],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert "MODAL_FREE_IMPORT_OK" in proc.stdout, f"stdout={proc.stdout}\nstderr={proc.stderr}"

    def test_stub_is_noop_when_sdk_present(self):
        # Modal is installed in this environment; the stub must not shadow it.
        pytest.importorskip("modal")
        _install_modal_stub()
        import modal

        assert not getattr(modal, "__file__", "").endswith("train_local.py")


class TestDiskPreflight:
    """Checkpoints are tens of GB per epoch; filling a shared volume mid-run
    is worse than refusing to start."""

    def test_refuses_when_space_insufficient(self, tmp_path):
        with pytest.raises(SystemExit, match="PREFLIGHT FAIL"):
            preflight_disk(str(tmp_path), min_free_gb=10_000_000.0)

    def test_passes_with_modest_requirement(self, tmp_path):
        preflight_disk(str(tmp_path), min_free_gb=0.001)

    def test_creates_missing_artifact_dir(self, tmp_path):
        target = tmp_path / "nested" / "artifacts"
        preflight_disk(str(target), min_free_gb=0.001)
        assert target.is_dir()


if "modal" not in sys.modules:
    _install_modal_stub()


class TestVramEstimate:
    """Sizing must flag graphs that cannot fit before burning GPU hours."""

    def test_full_graph_exceeds_a_24gb_card(self):
        from event_jepa_cube.scale_pipeline_v6 import V6Config

        # 35.2M nodes at the production V6.2 dims: embeddings + TGN memory
        # alone are ~27GB, which is why the full graph needs a bigger device.
        needed = estimate_vram_gb(35_200_000, V6Config())
        assert needed > 24.0

    def test_hospital_subgraph_fits_a_small_card(self):
        from event_jepa_cube.scale_pipeline_v6 import V6Config

        needed = estimate_vram_gb(500_000, V6Config())
        assert needed < 20.0

    def test_scales_with_latent_dim(self):
        from event_jepa_cube.scale_pipeline_v6 import V6Config

        small = V6Config(latent_dim=64)
        large = V6Config(latent_dim=128)
        assert estimate_vram_gb(1_000_000, large) > estimate_vram_gb(1_000_000, small)


class TestBuildConfig:
    def test_paths_override_modal_defaults(self, tmp_path):
        cfg = build_config("", str(tmp_path), "/local/graph.parquet", "/local/onto.parquet")
        assert cfg.parquet_path == "/local/graph.parquet"
        assert cfg.artifact_dir == str(tmp_path)
        # Warm-start lookup must point at local storage, not Modal volume paths.
        assert cfg.v5_artifact_dir == str(tmp_path)
        assert cfg.v6_artifact_dir == str(tmp_path)

    def test_json_overrides_applied(self, tmp_path):
        cfg = build_config('{"epochs": 2, "hospital_filter": "GHO-BRADESCO"}', str(tmp_path), "g", "o")
        assert cfg.epochs == 2
        assert cfg.hospital_filter == "GHO-BRADESCO"

    def test_unknown_config_key_rejected(self, tmp_path):
        # A typo'd key must fail fast, not silently train with defaults.
        with pytest.raises(SystemExit, match="Unknown config keys"):
            build_config('{"epocs": 2}', str(tmp_path), "g", "o")
