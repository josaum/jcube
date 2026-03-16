"""Simple implementation of the Embedding Cube."""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List

from .sequence import Entity
from .registry import get_model


class EmbeddingCube:
    """Container for multi-semantic entities and relationship models."""

    def __init__(self) -> None:
        self.entities: Dict[str, Entity] = {}
        self.models: Dict[str, Any] = {}

    def add_entity(self, entity: Entity) -> None:
        self.entities[entity.id] = entity

    def add_model(self, name: str, model: Any) -> None:
        self.models[name] = model

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
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

        For each pair of entities:
        1. Find shared embedding modalities.
        2. Compute cosine similarity per shared modality.
        3. Average across modalities.
        4. Add a +0.1 bonus (capped at 1.0) if hierarchy_info categories match.
        5. Include the relationship only if the final similarity >= *threshold*.
        """
        id_list = list(entity_ids)
        relationships: Dict[str, List[str]] = {}

        for idx_a, id_a in enumerate(id_list):
            entity_a = self.entities.get(id_a)
            if entity_a is None:
                continue

            related: List[str] = []
            for id_b, entity_b in self.entities.items():
                if id_b == id_a:
                    continue

                # Shared modalities
                shared = set(entity_a.embeddings.keys()) & set(entity_b.embeddings.keys())
                if not shared:
                    continue

                sims = [
                    self._cosine_similarity(entity_a.embeddings[mod], entity_b.embeddings[mod])
                    for mod in shared
                ]
                avg_sim = sum(sims) / len(sims)

                # Hierarchy bonus
                cat_a = entity_a.hierarchy_info.get("category")
                cat_b = entity_b.hierarchy_info.get("category")
                if cat_a is not None and cat_a == cat_b:
                    avg_sim = min(avg_sim + 0.1, 1.0)

                if avg_sim >= threshold:
                    related.append(id_b)

            if related:
                relationships[id_a] = related

        return relationships

    def load_registered_model(self, name: str, *args: Any, **kwargs: Any) -> None:
        """Load a model from the registry and add it.

        Raises ``KeyError`` if the model is not registered.
        """
        model_cls = get_model(name)
        if model_cls is None:
            raise KeyError(f"Model '{name}' is not registered")
        self.add_model(name, model_cls(*args, **kwargs))
