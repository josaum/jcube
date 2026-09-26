"""Tests for the pure (non-Modal, non-GPU) logic of scale_pipeline_v6.

Skipped entirely if the modal package is not installed (the module imports
modal at top level). GPU/Modal behavior is validated by `--action smoke` and
the single-hospital run — see docs/V63_PREFLIGHT.md.
"""

import pytest

pytest.importorskip("modal")

from event_jepa_cube.scale_pipeline_v6 import (  # noqa: E402
    CURRICULUM_PHASES_V6,
    NUMERIC_PRIORITY,
    V6Config,
    _clean_state_dict,
    _pick_numeric_column,
)


class TestNumericColumnPriority:
    """The numeric column a table contributes must be deterministic.

    The original implementation iterated a Python set ("first match wins"),
    so hash randomization could silently switch a table between VL_TOTAL and
    VL_GLOSA across runs — different graph, same filename, no signal.
    """

    def test_totals_win_over_components(self):
        cols = ["ID_CD_FATURA", "VL_GLOSA", "VL_TOTAL", "NR_QTD"]
        assert _pick_numeric_column(cols) == "VL_TOTAL"

    def test_values_win_over_quantities(self):
        cols = ["NR_QTD", "VL_GLOSA"]
        assert _pick_numeric_column(cols) == "VL_GLOSA"

    def test_quantity_only_table(self):
        assert _pick_numeric_column(["NR_QTD", "DS_X"]) == "NR_QTD"

    def test_no_numeric_column(self):
        assert _pick_numeric_column(["ID_CD_X", "DS_Y"]) is None

    def test_priority_is_ordered_not_a_set(self):
        assert isinstance(NUMERIC_PRIORITY, tuple)
        assert NUMERIC_PRIORITY[0] == "VL_TOTAL"


class TestV6Config:
    def test_json_round_trip_preserves_lookahead_mode(self):
        cfg = V6Config(lookahead_mode="same_entity")
        restored = V6Config.from_json(cfg.to_json())
        assert restored.lookahead_mode == "same_entity"

    def test_default_mode_is_legacy_cross_entity(self):
        # Default must stay cross_entity so existing V6.2 checkpoints/configs
        # keep their behavior; same_entity is opted into for the A/B.
        assert V6Config().lookahead_mode == "cross_entity"

    def test_sparse_embedding_prereqs(self):
        # node_emb uses sparse grads: SGD supports them only with zero
        # momentum and zero weight decay. Guard the defaults.
        cfg = V6Config()
        assert cfg.emb_momentum == 0.0
        assert cfg.emb_weight_decay == 0.0


class TestCleanStateDict:
    """Legacy V6.2 checkpoints carry torch.compile ("_orig_mod.") prefixes;
    current checkpoints save bare-module keys. Resume must accept both."""

    def test_strips_compile_and_ddp_prefixes(self):
        state = {
            "_orig_mod.pred_emb.weight": 1,
            "module._orig_mod.edge_proj.weight": 2,
            "final_norm.bias": 3,
        }
        cleaned = _clean_state_dict(state)
        assert set(cleaned) == {"pred_emb.weight", "edge_proj.weight", "final_norm.bias"}
        assert cleaned["pred_emb.weight"] == 1
        assert cleaned["edge_proj.weight"] == 2

    def test_bare_keys_pass_through(self):
        state = {"a.b": 1}
        assert _clean_state_dict(state) == {"a.b": 1}


class TestCurriculum:
    def test_phases_cover_epochs_contiguously(self):
        # The entrypoint's cost/time messaging derives from these bounds;
        # they must tile [0, epochs) without gaps or overlap.
        cfg = V6Config()
        bounds = [(p.epoch_start, p.epoch_end) for p in CURRICULUM_PHASES_V6]
        assert bounds[0][0] == 0
        for (_, prev_end), (start, _) in zip(bounds, bounds[1:]):
            assert start == prev_end
        assert bounds[-1][1] == cfg.epochs
