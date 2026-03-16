"""Core classes for Event-JEPA-Cube framework."""

from .sequence import EventSequence, Entity
from .event_jepa import EventJEPA
from .embedding_cube import EmbeddingCube
from .registry import register_embedding_type, register_model

__all__ = [
    "EventSequence",
    "Entity",
    "EventJEPA",
    "EmbeddingCube",
    "register_embedding_type",
    "register_model",
]

# Regularizers are optional (require PyTorch)
try:
    from .regularizers import SIGReg, WeakSIGReg, RDMReg

    __all__ += ["SIGReg", "WeakSIGReg", "RDMReg"]
except ImportError:
    pass
