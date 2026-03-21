"""Embedding Cube with cached norms and numpy-accelerated similarity.

Uses batch cosine similarity via numpy when available, with norm caching
to avoid recomputing per-entity norms on every relationship query.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any, Dict, List, Optional

from . import numpy_ops as npo
from .registry import get_model
from .sequence import Entity

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False


class EmbeddingCube:
    """Container for multi-semantic entities and relationship models.

    Caches per-entity embedding norms and uses batch cosine similarity
    for fast relationship discovery.
    """

    def __init__(self) -> None:
        self.entities: Dict[str, Entity] = {}
        self.models: Dict[str, Any] = {}
        # Norm cache: {(entity_id, modality): norm_value}
        self._norm_cache: Dict[tuple[str, str], float] = {}
        # Numpy embedding cache: {(entity_id, modality): ndarray}
        self._array_cache: Dict[tuple[str, str], Any] = {}

    def _invalidate_cache(self, entity_id: str) -> None:
        """Remove cached data for an entity."""
        keys_to_remove = [k for k in self._norm_cache if k[0] == entity_id]
        for k in keys_to_remove:
            del self._norm_cache[k]
        keys_to_remove = [k for k in self._array_cache if k[0] == entity_id]
        for k in keys_to_remove:
            del self._array_cache[k]

    def _cache_entity(self, entity: Entity) -> None:
        """Pre-compute and cache norms for an entity."""
        for mod, emb in entity.embeddings.items():
            key = (entity.id, mod)
            if _HAS_NUMPY:
                arr = np.asarray(emb, dtype=np.float64)
                self._array_cache[key] = arr
                self._norm_cache[key] = float(np.linalg.norm(arr))
            else:
                self._norm_cache[key] = math.sqrt(sum(x * x for x in emb))

    def add_entity(self, entity: Entity) -> None:
        self._invalidate_cache(entity.id)
        self.entities[entity.id] = entity
        self._cache_entity(entity)

    def add_model(self, name: str, model: Any) -> None:
        self.models[name] = model

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        return npo.cosine_similarity(a, b)

    def _cosine_similarity_cached(
        self,
        id_a: str,
        id_b: str,
        modality: str,
        emb_a: List[float],
        emb_b: List[float],
    ) -> float:
        """Cosine similarity using cached norms."""
        norm_a = self._norm_cache.get((id_a, modality))
        norm_b = self._norm_cache.get((id_b, modality))

        if norm_a is None:
            norm_a = math.sqrt(sum(x * x for x in emb_a))
            self._norm_cache[(id_a, modality)] = norm_a
        if norm_b is None:
            norm_b = math.sqrt(sum(x * x for x in emb_b))
            self._norm_cache[(id_b, modality)] = norm_b

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        if _HAS_NUMPY:
            arr_a = self._array_cache.get((id_a, modality))
            arr_b = self._array_cache.get((id_b, modality))
            if arr_a is None:
                arr_a = np.asarray(emb_a, dtype=np.float64)
                self._array_cache[(id_a, modality)] = arr_a
            if arr_b is None:
                arr_b = np.asarray(emb_b, dtype=np.float64)
                self._array_cache[(id_b, modality)] = arr_b
            dot = float(np.dot(arr_a, arr_b))
        else:
            dot = sum(x * y for x, y in zip(emb_a, emb_b))

        return dot / (norm_a * norm_b)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def discover_relationships(
        self,
        entity_ids: Iterable[str],
        threshold: float = 0.5,
    ) -> Dict[str, List[str]]:
        """Discover relationships using cosine similarity.

        Uses cached norms and batch numpy operations when available.
        """
        id_list = list(entity_ids)
        relationships: Dict[str, List[str]] = {}

        # Collect all entity ids for batch processing
        all_entity_ids = list(self.entities.keys())

        for id_a in id_list:
            entity_a = self.entities.get(id_a)
            if entity_a is None:
                continue

            related: List[str] = []
            cat_a = entity_a.hierarchy_info.get("category")

            for id_b in all_entity_ids:
                if id_b == id_a:
                    continue
                entity_b = self.entities[id_b]

                # Shared modalities
                shared = set(entity_a.embeddings.keys()) & set(entity_b.embeddings.keys())
                if not shared:
                    continue

                sims = [
                    self._cosine_similarity_cached(
                        id_a, id_b, mod, entity_a.embeddings[mod], entity_b.embeddings[mod]
                    )
                    for mod in shared
                ]
                avg_sim = sum(sims) / len(sims)

                # Hierarchy bonus
                cat_b = entity_b.hierarchy_info.get("category")
                if cat_a is not None and cat_a == cat_b:
                    avg_sim = min(avg_sim + 0.1, 1.0)

                if avg_sim >= threshold:
                    related.append(id_b)

            if related:
                relationships[id_a] = related

        return relationships

    def discover_relationships_batch(
        self,
        entity_ids: Iterable[str],
        threshold: float = 0.5,
    ) -> Dict[str, List[str]]:
        """Batch relationship discovery using matrix operations.

        Groups entities by shared modalities and computes similarity
        matrices in one shot. Significantly faster for large entity sets.
        """
        id_list = list(entity_ids)
        all_ids = list(self.entities.keys())

        if not _HAS_NUMPY or not id_list or not all_ids:
            return self.discover_relationships(id_list, threshold)

        # Collect all modalities
        all_modalities: set[str] = set()
        for eid in set(id_list) | set(all_ids):
            entity = self.entities.get(eid)
            if entity:
                all_modalities.update(entity.embeddings.keys())

        relationships: Dict[str, List[str]] = {}

        for id_a in id_list:
            entity_a = self.entities.get(id_a)
            if entity_a is None:
                continue

            cat_a = entity_a.hierarchy_info.get("category")
            # For each modality entity_a has, batch compute similarity
            # against all other entities that share that modality
            per_entity_sims: Dict[str, list[float]] = {}
            per_entity_counts: Dict[str, int] = {}

            for mod in entity_a.embeddings:
                vec_a = entity_a.embeddings[mod]
                # Gather all entities with this modality (excluding self)
                batch_ids = []
                batch_vecs = []
                for id_b in all_ids:
                    if id_b == id_a:
                        continue
                    eb = self.entities[id_b]
                    if mod in eb.embeddings:
                        batch_ids.append(id_b)
                        batch_vecs.append(eb.embeddings[mod])

                if not batch_vecs:
                    continue

                # Batch cosine: (1, N) result
                sim_row = npo.batch_cosine_similarity([vec_a], batch_vecs)[0]

                for id_b, sim in zip(batch_ids, sim_row):
                    if id_b not in per_entity_sims:
                        per_entity_sims[id_b] = []
                        per_entity_counts[id_b] = 0
                    per_entity_sims[id_b].append(sim)
                    per_entity_counts[id_b] += 1

            related: List[str] = []
            for id_b, sims in per_entity_sims.items():
                avg_sim = sum(sims) / len(sims)
                cat_b = self.entities[id_b].hierarchy_info.get("category")
                if cat_a is not None and cat_a == cat_b:
                    avg_sim = min(avg_sim + 0.1, 1.0)
                if avg_sim >= threshold:
                    related.append(id_b)

            if related:
                relationships[id_a] = related

        return relationships

    def load_registered_model(self, name: str, *args: Any, **kwargs: Any) -> None:
        """Load a model from the registry and add it."""
        model_cls = get_model(name)
        if model_cls is None:
            raise KeyError(f"Model '{name}' is not registered")
        self.add_model(name, model_cls(*args, **kwargs))
