"""Core classes for Event-JEPA-Cube framework."""

from .embedding_cube import EmbeddingCube
from .event_jepa import EventJEPA
from .registry import register_embedding_type, register_model
from .sequence import Entity, EventSequence

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
    from .regularizers import RDMReg, SIGReg, WeakSIGReg

    __all__ += ["SIGReg", "WeakSIGReg", "RDMReg"]
except ImportError:
    pass

# DuckDB connector is optional (requires duckdb)
try:
    from .duckdb_connector import DuckDBConnector

    __all__ += ["DuckDBConnector"]
except ImportError:
    pass

# Trigger engine is optional (requires duckdb)
try:
    from .triggers import AlertRule, StopHandle, TriggerEngine, register_action

    __all__ += ["TriggerEngine", "AlertRule", "StopHandle", "register_action"]
except ImportError:
    pass

# Cascade pipeline is optional (requires duckdb)
try:
    from .cascade import CascadeLevel, ForecastCascade

    __all__ += ["CascadeLevel", "ForecastCascade"]
except ImportError:
    pass
