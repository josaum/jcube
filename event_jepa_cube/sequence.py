"""Data structures for sequences and entities."""
from dataclasses import dataclass, field
from typing import Any, Dict, List
import uuid

@dataclass
class EventSequence:
    """A sequence of event embeddings with timestamps."""

    embeddings: List[List[float]]
    timestamps: List[float]
    modality: str = "text"

    def __post_init__(self) -> None:
        if len(self.embeddings) != len(self.timestamps):
            raise ValueError("Embeddings and timestamps must have the same length")

@dataclass
class Entity:
    """Represents a multi-semantic entity for the Embedding Cube."""

    embeddings: Dict[str, List[float]]
    hierarchy_info: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
