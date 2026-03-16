"""Tests for the Pipeline orchestrator.

Requires DuckDB for integration tests.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

duckdb = pytest.importorskip("duckdb")

from event_jepa_cube.orchestrator import Pipeline, PipelineError  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def duckdb_config():
    return {"database": ":memory:", "embedding_dim": 3}


@pytest.fixture
def pipeline(duckdb_config):
    p = Pipeline(duckdb_config=duckdb_config)
    yield p
    p.shutdown()


@pytest.fixture
def populated_pipeline(pipeline):
    """Pipeline with sample data."""
    conn = pipeline._connector._ensure_open()
    conn.execute("""
        CREATE TABLE event_sequences (
            sequence_id VARCHAR,
            embedding FLOAT[],
            timestamp DOUBLE,
            modality VARCHAR
        )
    """)
    conn.execute("""
        INSERT INTO event_sequences VALUES
        ('seq1', [1.0, 0.0, 0.0], 1.0, 'text'),
        ('seq1', [0.0, 1.0, 0.0], 2.0, 'text'),
        ('seq1', [0.0, 0.0, 1.0], 3.0, 'text'),
        ('seq2', [0.5, 0.5, 0.0], 1.0, 'text'),
        ('seq2', [0.0, 0.5, 0.5], 2.0, 'text')
    """)
    return pipeline


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInit:
    def test_minimal_pipeline(self, duckdb_config):
        p = Pipeline(duckdb_config=duckdb_config)
        assert p._connector is not None
        assert p._mycelia is None
        assert p._bandit is None
        assert p._cascade is None
        p.shutdown()

    def test_empty_pipeline(self):
        p = Pipeline()
        assert p._connector is None
        p.shutdown()

    def test_with_cascade_levels(self, duckdb_config):
        p = Pipeline(
            duckdb_config=duckdb_config,
            cascade_levels=[
                {"name": "level1", "num_prediction_steps": 3},
                {"name": "level2", "num_prediction_steps": 2},
            ],
        )
        assert p._cascade is not None
        p.shutdown()


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------


class TestRun:
    def test_run_produces_results(self, populated_pipeline):
        result = populated_pipeline.run(sync_to_mycelia=False)
        assert "representations" in result
        assert "predictions" in result
        assert len(result["representations"]) == 2  # seq1, seq2

    def test_run_without_connector_raises(self):
        p = Pipeline()
        with pytest.raises(PipelineError, match="DuckDB"):
            p.run()

    def test_run_with_mycelia_sync(self, populated_pipeline):
        mock_mycelia = MagicMock()
        mock_mycelia.sync_pipeline_results.return_value = {"synced": 2}
        populated_pipeline._mycelia = mock_mycelia

        result = populated_pipeline.run(sync_to_mycelia=True)
        assert result["mycelia_sync"] == {"synced": 2}
        mock_mycelia.sync_pipeline_results.assert_called_once()


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


class TestStreaming:
    def test_process_event(self, pipeline):
        rep = pipeline.process_event("stream1", [1.0, 2.0, 3.0], 1.0)
        assert len(rep) == 3

    def test_get_streaming_processor_reuses(self, pipeline):
        p1 = pipeline.get_streaming_processor("s1", embedding_dim=3)
        p2 = pipeline.get_streaming_processor("s1")
        assert p1 is p2

    def test_separate_streams(self, pipeline):
        pipeline.process_event("s1", [1.0, 0.0, 0.0], 1.0)
        pipeline.process_event("s2", [0.0, 1.0, 0.0], 1.0)
        r1 = pipeline.get_streaming_processor("s1").get_representation()
        r2 = pipeline.get_streaming_processor("s2").get_representation()
        assert r1 != r2


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class TestSearch:
    def test_search_without_mycelia_raises(self, pipeline):
        with pytest.raises(PipelineError, match="MyceliaStore"):
            pipeline.search_similar("col", [1.0, 2.0])

    @patch("urllib.request.urlopen")
    def test_search_similar(self, mock_urlopen, pipeline):
        import json

        resp = MagicMock()
        resp.status = 200
        resp.read.return_value = json.dumps({"results": [{"id": "v1", "score": 0.9}]}).encode()
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp

        from event_jepa_cube.mycelia_store import MyceliaStore

        pipeline._mycelia = MyceliaStore("https://test.local", api_key="key")
        results = pipeline.search_similar("col", [1.0, 2.0])
        assert len(results) == 1

    def test_search_gepa_local(self, pipeline):
        vectors = {
            "v1": [1.0, 0.0, 0.0],
            "v2": [0.0, 1.0, 0.0],
            "v3": [0.9, 0.1, 0.0],
        }
        result = pipeline.search_gepa_local(vectors, seed_vector=[1.0, 0.0, 0.0], limit=2)
        assert len(result.results) <= 2
        assert result.best_score > 0


# ---------------------------------------------------------------------------
# Cascade
# ---------------------------------------------------------------------------


class TestCascade:
    def test_cascade_not_configured_raises(self, pipeline):
        with pytest.raises(PipelineError, match="Cascade"):
            pipeline.poll_cascade()

    def test_start_cascade_not_configured_raises(self, pipeline):
        with pytest.raises(PipelineError, match="Cascade"):
            pipeline.start_cascade()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_context_manager(self, duckdb_config):
        with Pipeline(duckdb_config=duckdb_config) as p:
            assert p._connector is not None
        assert p._connector is None

    def test_shutdown_clears_streaming(self, pipeline):
        pipeline.process_event("s1", [1.0, 2.0, 3.0], 1.0)
        assert len(pipeline._streaming) == 1
        pipeline.shutdown()
        assert len(pipeline._streaming) == 0

    def test_ingest_without_connector_raises(self):
        p = Pipeline()
        with pytest.raises(PipelineError, match="DuckDB"):
            p.ingest_sources({"pg": "postgresql://localhost/test"})
