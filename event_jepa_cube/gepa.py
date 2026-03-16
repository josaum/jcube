"""GEPA — Guided Embedding-space Pattern Aggregation.

Iterative embedding evolution search using reward-based mutation
(diversity + similarity score) to converge on the most relevant region
of the embedding space.  Pure vector math, no LLM dependency.

The :class:`GEPASearch` client supports two modes:

1. **Remote** — calls the Mycelia API endpoint
   ``POST /v2/search/{collection}/gepa``.
2. **Local** — pure-Python evolutionary search against an in-memory
   vector dict (useful for offline use or testing).

Zero external dependencies — uses only the Python standard library.

Example::

    gepa = GEPASearch("https://api.getjai.com", api_key="...")
    result = gepa.search("embeddings", seed_vector=[0.1, 0.2, ...])
    print(result.best_score, result.iterations_run)

    # Or locally:
    vecs = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    result = gepa.search_local(vecs, seed_vector=[0.7, 0.7])
"""

from __future__ import annotations

import json
import logging
import math
import random
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class GEPAResult:
    """Result of a GEPA evolutionary search."""

    results: list[dict[str, Any]]  # [{id, distance, score}, ...]
    final_query: list[float]  # Evolved query vector
    trajectory: list[list[float]] = field(default_factory=list)  # Query vector at each iteration
    iterations_run: int = 0  # Actual iterations completed
    best_score: float = 0.0  # Best similarity score found


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class GEPAError(Exception):
    """Error from GEPA search operations."""


# ---------------------------------------------------------------------------
# Vector math helpers (stdlib only)
# ---------------------------------------------------------------------------


def _dot(a: list[float], b: list[float]) -> float:
    """Dot product of two vectors."""
    return sum(x * y for x, y in zip(a, b))


def _norm(v: list[float]) -> float:
    """Euclidean norm of a vector."""
    return math.sqrt(sum(x * x for x in v))


def _normalize(v: list[float]) -> list[float]:
    """Return unit vector, or zero vector if norm is zero."""
    n = _norm(v)
    if n == 0.0:
        return [0.0] * len(v)
    return [x / n for x in v]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    na = _norm(a)
    nb = _norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return _dot(a, b) / (na * nb)


def _euclidean_distance(a: list[float], b: list[float]) -> float:
    """Euclidean distance between two vectors."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _mutate(v: list[float], mutation_rate: float, rng: random.Random) -> list[float]:
    """Add Gaussian noise scaled by *mutation_rate*."""
    return [x + rng.gauss(0.0, mutation_rate) for x in v]


def _weighted_centroid(
    vectors: list[list[float]],
    weights: list[float],
) -> list[float]:
    """Weighted centroid of a set of vectors."""
    dim = len(vectors[0])
    result = [0.0] * dim
    total_weight = sum(weights)
    if total_weight == 0.0:
        return list(vectors[0])
    for vec, w in zip(vectors, weights):
        for i in range(dim):
            result[i] += w * vec[i]
    return [x / total_weight for x in result]


# ---------------------------------------------------------------------------
# GEPASearch
# ---------------------------------------------------------------------------


class GEPASearch:
    """GEPA iterative embedding evolution search.

    Performs pure vector-space query refinement: starting from a seed
    embedding, iteratively mutates the query vector using reward signals
    (similarity scores + diversity) to converge on the most relevant
    region of the embedding space.

    This is useful for:

    - Finding similar event sequences without text queries
    - Exploring embedding neighborhoods iteratively
    - RAG-style retrieval using only vector math (no LLM)

    Args:
        base_url: Mycelia API base URL.
        api_key: API key or Bearer token for authentication.
        namespace: Optional namespace scope for multi-tenant deployments.
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        namespace: str | None = None,
        timeout: int = 30,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._namespace = namespace
        self._timeout = timeout

    # ------------------------------------------------------------------
    # HTTP helpers (same pattern as MyceliaStore)
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        if self._namespace:
            headers["X-Namespace"] = self._namespace
        return headers

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> Any:
        """Make an HTTP request to the Mycelia API."""
        url = f"{self._base_url}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(url, data=data, headers=self._headers(), method=method)

        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                if resp.status == 204:
                    return None
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as e:
            body_text = e.read().decode("utf-8", errors="replace") if e.fp else ""
            raise GEPAError(f"HTTP {e.code} on {method} {path}: {body_text}") from e
        except urllib.error.URLError as e:
            raise GEPAError(f"Connection error on {method} {path}: {e.reason}") from e

    def _post(self, path: str, body: dict[str, Any] | None = None) -> Any:
        return self._request("POST", path, body)

    # ------------------------------------------------------------------
    # Remote search (Mycelia API)
    # ------------------------------------------------------------------

    def search(
        self,
        collection: str,
        seed_vector: list[float],
        iterations: int = 5,
        population_size: int = 10,
        mutation_rate: float = 0.1,
        diversity_weight: float = 0.3,
        limit: int = 10,
    ) -> GEPAResult:
        """Run GEPA search via the Mycelia API.

        Args:
            collection: Mycelia collection to search.
            seed_vector: Initial query embedding.
            iterations: Number of evolution iterations.
            population_size: Candidate query variants per iteration.
            mutation_rate: Magnitude of random perturbations.
            diversity_weight: Weight for diversity vs. similarity score.
            limit: Final top-k results.

        Returns:
            GEPAResult with results, evolution trajectory, and final query.
        """
        body: dict[str, Any] = {
            "seed_vector": seed_vector,
            "iterations": iterations,
            "population_size": population_size,
            "mutation_rate": mutation_rate,
            "diversity_weight": diversity_weight,
            "limit": limit,
        }
        result = self._post(f"/v2/search/{collection}/gepa", body)

        if not isinstance(result, dict):
            raise GEPAError(f"Unexpected response type from GEPA endpoint: {type(result)}")

        return GEPAResult(
            results=result.get("results", []),
            final_query=result.get("final_query", seed_vector),
            trajectory=result.get("trajectory", []),
            iterations_run=result.get("iterations_run", iterations),
            best_score=result.get("best_score", 0.0),
        )

    # ------------------------------------------------------------------
    # Local search (pure Python)
    # ------------------------------------------------------------------

    def search_local(
        self,
        vectors: dict[str, list[float]],
        seed_vector: list[float],
        iterations: int = 5,
        population_size: int = 10,
        mutation_rate: float = 0.1,
        diversity_weight: float = 0.3,
        limit: int = 10,
        seed_random: int | None = None,
    ) -> GEPAResult:
        """Run GEPA search locally without Mycelia API.

        Pure Python implementation for offline use or testing.
        Performs the same evolutionary search against a local vector dict.

        Args:
            vectors: Dict of ``{id: embedding}`` to search against.
            seed_vector: Initial query embedding.
            iterations: Number of evolution iterations.
            population_size: Candidate query variants per iteration.
            mutation_rate: Magnitude of random perturbations.
            diversity_weight: Weight for diversity vs. similarity score.
            limit: Final top-k results.
            seed_random: Optional random seed for reproducibility.

        Returns:
            GEPAResult with results, evolution trajectory, and final query.
        """
        if not vectors:
            return GEPAResult(
                results=[],
                final_query=list(seed_vector),
                trajectory=[],
                iterations_run=0,
                best_score=0.0,
            )

        rng = random.Random(seed_random)
        vec_ids = list(vectors.keys())
        vec_matrix = [vectors[vid] for vid in vec_ids]

        # 1. Initialize population: seed + random mutations
        population = [list(seed_vector)]
        for _ in range(population_size - 1):
            population.append(_mutate(seed_vector, mutation_rate, rng))

        trajectory: list[list[float]] = [list(seed_vector)]

        # 2. Evolutionary loop
        for _iteration in range(iterations):
            # a. Score each candidate: avg cosine similarity to all vectors
            candidate_scores: list[float] = []
            for candidate in population:
                sims = [_cosine_similarity(candidate, v) for v in vec_matrix]
                # Best similarity against any stored vector
                candidate_scores.append(max(sims) if sims else 0.0)

            # b. Diversity bonus: avg pairwise distance between candidates
            diversity_scores: list[float] = []
            for i, candidate in enumerate(population):
                if len(population) <= 1:
                    diversity_scores.append(0.0)
                    continue
                distances = []
                for j, other in enumerate(population):
                    if i != j:
                        distances.append(_euclidean_distance(candidate, other))
                diversity_scores.append(sum(distances) / len(distances))

            # Normalize diversity scores to [0, 1]
            max_div = max(diversity_scores) if diversity_scores else 1.0
            if max_div > 0:
                diversity_scores = [d / max_div for d in diversity_scores]

            # c. Combined fitness
            fitness = [
                (1.0 - diversity_weight) * sim + diversity_weight * div
                for sim, div in zip(candidate_scores, diversity_scores)
            ]

            # d. Select top candidates (keep top half, at least 1)
            keep = max(1, population_size // 2)
            ranked = sorted(range(len(population)), key=lambda i: fitness[i], reverse=True)
            survivors = [population[i] for i in ranked[:keep]]

            # e. Mutate to refill population
            new_population: list[list[float]] = list(survivors)
            while len(new_population) < population_size:
                parent = survivors[rng.randint(0, len(survivors) - 1)]
                new_population.append(_mutate(parent, mutation_rate, rng))

            population = new_population

            # f. Record trajectory: best candidate
            ranked[0]
            trajectory.append(list(population[0]))

        # 3. Final: take the best candidate
        final_scores = []
        for candidate in population:
            sims = [_cosine_similarity(candidate, v) for v in vec_matrix]
            final_scores.append(max(sims) if sims else 0.0)

        best_candidate_idx = max(range(len(population)), key=lambda i: final_scores[i])
        final_query = population[best_candidate_idx]
        best_score = final_scores[best_candidate_idx]

        # Find top-k nearest vectors to the final query
        scored_vectors = []
        for vid, vec in zip(vec_ids, vec_matrix):
            sim = _cosine_similarity(final_query, vec)
            dist = _euclidean_distance(final_query, vec)
            scored_vectors.append({"id": vid, "distance": dist, "score": sim})

        scored_vectors.sort(key=lambda x: x["score"], reverse=True)
        top_results = scored_vectors[:limit]

        return GEPAResult(
            results=top_results,
            final_query=final_query,
            trajectory=trajectory,
            iterations_run=iterations,
            best_score=best_score,
        )

    # ------------------------------------------------------------------
    # Single-step evolution
    # ------------------------------------------------------------------

    def evolve_query(
        self,
        seed_vector: list[float],
        feedback_vectors: list[list[float]],
        feedback_scores: list[float],
        mutation_rate: float = 0.1,
        seed_random: int | None = None,
    ) -> list[float]:
        """Single-step query evolution using explicit feedback.

        Given a current query and scored results, produce an improved query
        by moving toward high-scoring results and away from low-scoring ones.

        Uses weighted centroid: ``new_query = normalize(sum(score_i * vector_i))``
        Then adds controlled perturbation.

        Args:
            seed_vector: Current query vector.
            feedback_vectors: Result vectors from previous search.
            feedback_scores: Relevance scores for each result (higher = better).
            mutation_rate: Perturbation magnitude.
            seed_random: Optional random seed for reproducibility.

        Returns:
            Evolved query vector.

        Raises:
            GEPAError: If feedback_vectors and feedback_scores have different lengths,
                or if no feedback is provided.
        """
        if len(feedback_vectors) != len(feedback_scores):
            raise GEPAError(
                f"feedback_vectors ({len(feedback_vectors)}) and "
                f"feedback_scores ({len(feedback_scores)}) must have the same length"
            )
        if not feedback_vectors:
            raise GEPAError("At least one feedback vector is required")

        rng = random.Random(seed_random)

        # Weighted centroid of feedback vectors
        centroid = _weighted_centroid(feedback_vectors, feedback_scores)
        evolved = _normalize(centroid)

        # Blend with seed: 50% evolved direction, 50% original
        dim = len(seed_vector)
        blended = [(seed_vector[i] + evolved[i]) / 2.0 for i in range(dim)]
        blended = _normalize(blended)

        # Add perturbation
        if mutation_rate > 0:
            blended = _mutate(blended, mutation_rate, rng)

        return blended
