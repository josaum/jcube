"""Simple implementation of the Embedding Cube."""
from __future__ import annotations

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

    def discover_relationships(self, entity_ids: Iterable[str], threshold: float = 0.5) -> Dict[str, List[str]]:
        """Dummy relationship discovery based on shared hierarchy info."""
        relationships: Dict[str, List[str]] = {}
        for entity_id in entity_ids:
            target = self.entities.get(entity_id)
            if not target:
                continue
            related = [
                e_id
                for e_id, entity in self.entities.items()
                if e_id != entity_id and entity.hierarchy_info.get("category") == target.hierarchy_info.get("category")
            ]
            if related:
                relationships[entity_id] = related
        return relationships

    def load_registered_model(self, name: str, *args, **kwargs) -> None:
        model_cls = get_model(name)
        if model_cls:
            self.add_model(name, model_cls(*args, **kwargs))
