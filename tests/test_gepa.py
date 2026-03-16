"""Tests for GEPASearch — Guided Embedding-space Pattern Aggregation.

All HTTP calls are mocked via ``unittest.mock.patch`` so no live
Mycelia instance is required.
"""

from __future__ import annotations

import json
import math
from unittest.mock import MagicMock, patch

import pytest

from event_jepa_cube.gepa import (
    GEPAError,
    GEPAResult,
    GEPASearch,
    _cosine_similarity,
    _normalize,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def gepa():
    return GEPASearch(
        base_url="https://test.mycelia.local",
        api_key="test-key",
        namespace="test-ns",
    )


def _mock_response(body: dict | list | None = None, status: int = 200):
    """Create a mock urllib response context manager."""
    resp = MagicMock()
    resp.status = status
    resp.read.return_value = json.dumps(body).encode() if body is not None else b""
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    return resp


# ---------------------------------------------------------------------------
# GEPAResult dataclass
# ---------------------------------------------------------------------------


class TestGEPAResult:
    def test_default_values(self):
        r = GEPAResult(results=[], final_query=[0.1, 0.2])
        assert r.results == []
        assert r.final_query == [0.1, 0.2]
        assert r.trajectory == []
        assert r.iterations_run == 0
        assert r.best_score == 0.0

    def test_full_construction(self):
        r = GEPAResult(
            results=[{"id": "v1", "score": 0.9, "distance": 0.1}],
            final_query=[1.0, 0.0],
            trajectory=[[0.5, 0.5], [0.8, 0.2]],
            iterations_run=3,
            best_score=0.95,
        )
        assert len(r.results) == 1
        assert r.iterations_run == 3
        assert r.best_score == 0.95
        assert len(r.trajectory) == 2


# ---------------------------------------------------------------------------
# Headers
# ---------------------------------------------------------------------------


class TestHeaders:
    def test_headers_with_auth_and_namespace(self, gepa):
        h = gepa._headers()
        assert h["Authorization"] == "Bearer test-key"
        assert h["X-Namespace"] == "test-ns"
        assert h["Content-Type"] == "application/json"

    def test_headers_without_auth(self):
        g = GEPASearch("https://x.local")
        h = g._headers()
        assert "Authorization" not in h
        assert "X-Namespace" not in h


# ---------------------------------------------------------------------------
# Remote search (mocked)
# ---------------------------------------------------------------------------


class TestRemoteSearch:
    @patch("urllib.request.urlopen")
    def test_search_calls_gepa_endpoint(self, mock_urlopen, gepa):
        api_response = {
            "results": [{"id": "v1", "distance": 0.1, "score": 0.9}],
            "final_query": [0.8, 0.2],
            "trajectory": [[0.5, 0.5], [0.8, 0.2]],
            "iterations_run": 5,
            "best_score": 0.9,
        }
        mock_urlopen.return_value = _mock_response(api_response)

        result = gepa.search("my_col", seed_vector=[0.5, 0.5], iterations=5, limit=10)

        assert isinstance(result, GEPAResult)
        assert len(result.results) == 1
        assert result.results[0]["id"] == "v1"
        assert result.best_score == 0.9
        assert result.iterations_run == 5
        assert result.final_query == [0.8, 0.2]

        # Verify the request was made to the correct endpoint
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert "/v2/search/my_col/gepa" in req.full_url
        body = json.loads(req.data)
        assert body["seed_vector"] == [0.5, 0.5]
        assert body["iterations"] == 5
        assert body["limit"] == 10

    @patch("urllib.request.urlopen")
    def test_search_sends_all_params(self, mock_urlopen, gepa):
        mock_urlopen.return_value = _mock_response(
            {"results": [], "final_query": [0.0], "trajectory": [], "iterations_run": 3, "best_score": 0.0}
        )
        gepa.search(
            "col",
            seed_vector=[0.1],
            iterations=3,
            population_size=20,
            mutation_rate=0.05,
            diversity_weight=0.5,
            limit=5,
        )

        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data)
        assert body["population_size"] == 20
        assert body["mutation_rate"] == 0.05
        assert body["diversity_weight"] == 0.5

    @patch("urllib.request.urlopen")
    def test_search_http_error(self, mock_urlopen, gepa):
        import io
        import urllib.error

        err_body = io.BytesIO(b'{"detail": "collection not found"}')
        mock_urlopen.side_effect = urllib.error.HTTPError("https://x", 404, "Not Found", {}, err_body)

        with pytest.raises(GEPAError, match="HTTP 404"):
            gepa.search("missing_col", seed_vector=[0.1, 0.2])

    @patch("urllib.request.urlopen")
    def test_search_connection_error(self, mock_urlopen, gepa):
        import urllib.error

        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        with pytest.raises(GEPAError, match="Connection error"):
            gepa.search("col", seed_vector=[0.1, 0.2])


# ---------------------------------------------------------------------------
# Local search
# ---------------------------------------------------------------------------


class TestLocalSearch:
    def test_finds_nearest_neighbor(self, gepa):
        """search_local should find the nearest vector to the seed."""
        vectors = {
            "close": [0.9, 0.1],
            "far": [0.0, 1.0],
            "medium": [0.5, 0.5],
        }
        result = gepa.search_local(
            vectors,
            seed_vector=[1.0, 0.0],
            iterations=3,
            population_size=5,
            mutation_rate=0.01,
            diversity_weight=0.0,
            limit=3,
            seed_random=42,
        )

        assert isinstance(result, GEPAResult)
        assert len(result.results) == 3
        # The closest vector should be "close"
        assert result.results[0]["id"] == "close"
        assert result.iterations_run == 3

    def test_empty_vectors(self, gepa):
        """search_local with empty dict should return empty result."""
        result = gepa.search_local({}, seed_vector=[1.0, 0.0])
        assert result.results == []
        assert result.iterations_run == 0
        assert result.best_score == 0.0

    def test_single_vector(self, gepa):
        """search_local with a single vector should still work."""
        vectors = {"only": [1.0, 0.0, 0.0]}
        result = gepa.search_local(
            vectors,
            seed_vector=[1.0, 0.0, 0.0],
            iterations=2,
            population_size=3,
            mutation_rate=0.01,
            limit=5,
            seed_random=42,
        )
        assert len(result.results) == 1
        assert result.results[0]["id"] == "only"
        assert result.best_score > 0.9

    def test_identical_vectors(self, gepa):
        """search_local with identical vectors should handle gracefully."""
        vectors = {
            "a": [1.0, 0.0],
            "b": [1.0, 0.0],
            "c": [1.0, 0.0],
        }
        result = gepa.search_local(
            vectors,
            seed_vector=[1.0, 0.0],
            iterations=3,
            population_size=5,
            mutation_rate=0.01,
            limit=3,
            seed_random=42,
        )
        assert len(result.results) == 3
        # All should have high scores since they're identical
        for r in result.results:
            assert r["score"] > 0.9

    def test_trajectory_recorded(self, gepa):
        """Trajectory should have entries for initial + each iteration."""
        vectors = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
        result = gepa.search_local(
            vectors,
            seed_vector=[0.5, 0.5],
            iterations=4,
            population_size=5,
            mutation_rate=0.05,
            seed_random=42,
        )
        # trajectory: 1 initial + 4 iterations = 5 entries
        assert len(result.trajectory) == 5

    def test_convergence_score_improves(self, gepa):
        """After iterations, best_score should be reasonable."""
        # Create a clear target
        vectors = {
            "target": [1.0, 0.0, 0.0, 0.0],
            "distractor1": [0.0, 1.0, 0.0, 0.0],
            "distractor2": [0.0, 0.0, 1.0, 0.0],
        }
        # Start from a nearby seed
        result = gepa.search_local(
            vectors,
            seed_vector=[0.8, 0.1, 0.1, 0.0],
            iterations=10,
            population_size=15,
            mutation_rate=0.05,
            diversity_weight=0.0,
            limit=3,
            seed_random=42,
        )
        # With enough iterations and low diversity weight, should converge well
        assert result.best_score > 0.8

    def test_mutation_rate_zero_no_perturbation(self, gepa):
        """With mutation_rate=0, population should not change from seed."""
        vectors = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
        result = gepa.search_local(
            vectors,
            seed_vector=[0.7, 0.7],
            iterations=3,
            population_size=5,
            mutation_rate=0.0,
            diversity_weight=0.0,
            limit=2,
            seed_random=42,
        )
        # With zero mutation, all candidates are the same as seed
        # The final query should be essentially the seed
        cos_sim = _cosine_similarity(result.final_query, [0.7, 0.7])
        assert cos_sim > 0.999

    def test_diversity_weight_one_maximizes_spread(self, gepa):
        """With diversity_weight=1.0, spread-out candidates are preferred."""
        vectors = {
            "a": [1.0, 0.0, 0.0],
            "b": [0.0, 1.0, 0.0],
            "c": [0.0, 0.0, 1.0],
        }
        result = gepa.search_local(
            vectors,
            seed_vector=[0.5, 0.5, 0.0],
            iterations=5,
            population_size=10,
            mutation_rate=0.1,
            diversity_weight=1.0,
            limit=3,
            seed_random=42,
        )
        # Should still return results
        assert len(result.results) == 3
        assert result.iterations_run == 5

    def test_limit_respected(self, gepa):
        """Limit should cap the number of results."""
        vectors = {f"v{i}": [float(i), float(10 - i)] for i in range(10)}
        result = gepa.search_local(
            vectors,
            seed_vector=[5.0, 5.0],
            iterations=2,
            population_size=5,
            mutation_rate=0.05,
            limit=3,
            seed_random=42,
        )
        assert len(result.results) <= 3

    def test_results_sorted_by_score(self, gepa):
        """Results should be sorted by score descending."""
        vectors = {
            "a": [1.0, 0.0],
            "b": [0.7, 0.7],
            "c": [0.0, 1.0],
        }
        result = gepa.search_local(
            vectors,
            seed_vector=[0.9, 0.1],
            iterations=3,
            population_size=5,
            mutation_rate=0.02,
            diversity_weight=0.0,
            limit=3,
            seed_random=42,
        )
        scores = [r["score"] for r in result.results]
        assert scores == sorted(scores, reverse=True)

    def test_results_contain_expected_keys(self, gepa):
        """Each result dict should have id, distance, score."""
        vectors = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
        result = gepa.search_local(
            vectors,
            seed_vector=[0.5, 0.5],
            iterations=1,
            population_size=3,
            limit=2,
            seed_random=42,
        )
        for r in result.results:
            assert "id" in r
            assert "distance" in r
            assert "score" in r


# ---------------------------------------------------------------------------
# evolve_query
# ---------------------------------------------------------------------------


class TestEvolveQuery:
    def test_moves_toward_high_scoring_vectors(self, gepa):
        """evolved query should be closer to high-scoring vectors."""
        seed = [0.5, 0.5]
        feedback_vectors = [
            [1.0, 0.0],  # high score
            [0.0, 1.0],  # low score
        ]
        feedback_scores = [0.9, 0.1]

        evolved = gepa.evolve_query(
            seed_vector=seed,
            feedback_vectors=feedback_vectors,
            feedback_scores=feedback_scores,
            mutation_rate=0.0,  # no noise so result is deterministic
            seed_random=42,
        )

        # Evolved should be closer to [1, 0] direction than [0, 1]
        sim_to_high = _cosine_similarity(evolved, [1.0, 0.0])
        sim_to_low = _cosine_similarity(evolved, [0.0, 1.0])
        assert sim_to_high > sim_to_low

    def test_with_uniform_scores(self, gepa):
        """With uniform scores, evolved query should be near the centroid."""
        seed = [0.0, 0.0, 1.0]
        feedback_vectors = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
        feedback_scores = [1.0, 1.0]

        evolved = gepa.evolve_query(
            seed_vector=seed,
            feedback_vectors=feedback_vectors,
            feedback_scores=feedback_scores,
            mutation_rate=0.0,
            seed_random=42,
        )

        assert len(evolved) == 3
        # Should be a valid vector (not all zeros)
        assert any(abs(x) > 0.01 for x in evolved)

    def test_mutation_adds_perturbation(self, gepa):
        """With mutation_rate > 0, result should differ from zero-mutation result."""
        seed = [0.5, 0.5]
        feedback_vectors = [[1.0, 0.0]]
        feedback_scores = [1.0]

        no_mutation = gepa.evolve_query(seed, feedback_vectors, feedback_scores, mutation_rate=0.0, seed_random=42)
        with_mutation = gepa.evolve_query(seed, feedback_vectors, feedback_scores, mutation_rate=0.5, seed_random=42)

        # They should differ
        assert no_mutation != with_mutation

    def test_mismatched_lengths_raises_error(self, gepa):
        with pytest.raises(GEPAError, match="same length"):
            gepa.evolve_query(
                seed_vector=[0.5, 0.5],
                feedback_vectors=[[1.0, 0.0], [0.0, 1.0]],
                feedback_scores=[0.9],
            )

    def test_empty_feedback_raises_error(self, gepa):
        with pytest.raises(GEPAError, match="At least one feedback"):
            gepa.evolve_query(
                seed_vector=[0.5, 0.5],
                feedback_vectors=[],
                feedback_scores=[],
            )


# ---------------------------------------------------------------------------
# Vector math helpers
# ---------------------------------------------------------------------------


class TestVectorMath:
    def test_cosine_similarity_identical(self):
        assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal(self):
        assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_cosine_similarity_opposite(self):
        assert _cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_cosine_similarity_zero_vector(self):
        assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_normalize(self):
        n = _normalize([3.0, 4.0])
        assert n[0] == pytest.approx(0.6)
        assert n[1] == pytest.approx(0.8)
        assert math.sqrt(sum(x * x for x in n)) == pytest.approx(1.0)

    def test_normalize_zero_vector(self):
        n = _normalize([0.0, 0.0])
        assert n == [0.0, 0.0]


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrors:
    @patch("urllib.request.urlopen")
    def test_http_error_raises_gepa_error(self, mock_urlopen, gepa):
        import io
        import urllib.error

        err_body = io.BytesIO(b'{"detail": "bad request"}')
        mock_urlopen.side_effect = urllib.error.HTTPError("https://x", 400, "Bad Request", {}, err_body)
        with pytest.raises(GEPAError, match="HTTP 400"):
            gepa.search("col", seed_vector=[0.1, 0.2])

    @patch("urllib.request.urlopen")
    def test_connection_error_raises_gepa_error(self, mock_urlopen, gepa):
        import urllib.error

        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
        with pytest.raises(GEPAError, match="Connection error"):
            gepa.search("col", seed_vector=[0.1, 0.2])

    @patch("urllib.request.urlopen")
    def test_unexpected_response_type(self, mock_urlopen, gepa):
        mock_urlopen.return_value = _mock_response([1, 2, 3])
        with pytest.raises(GEPAError, match="Unexpected response type"):
            gepa.search("col", seed_vector=[0.1, 0.2])
