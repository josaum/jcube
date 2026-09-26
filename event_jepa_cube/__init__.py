"""Core classes for Event-JEPA-Cube framework."""

from .embedding_cube import EmbeddingCube
from .event_jepa import EventJEPA
from .registry import register_embedding_type, register_model
from .sequence import Entity, EventSequence
from .training import CooldownSchedule

__all__ = [
    "EventSequence",
    "Entity",
    "EventJEPA",
    "EmbeddingCube",
    "register_embedding_type",
    "register_model",
    "CooldownSchedule",
]

# Regularizers are optional (require PyTorch)
try:
    from .regularizers import NormReg, RDMReg, SIGReg, WeakSIGReg

    __all__ += ["SIGReg", "WeakSIGReg", "RDMReg", "NormReg"]
except ImportError:
    pass

# Predictors are optional (require PyTorch)
try:
    from .predictors import MLPPredictor, PredictorTrainer, TransformerPredictor

    __all__ += ["MLPPredictor", "TransformerPredictor", "PredictorTrainer"]
except ImportError:
    pass

# Bidirectional cycle-consistent JEPA training is optional (requires PyTorch)
try:
    from .bijepa import BiJEPATrainer

    __all__ += ["BiJEPATrainer"]
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

# Mycelia store connector (no external deps — uses stdlib urllib)
from .mycelia_store import MyceliaError, MyceliaStore

__all__ += ["MyceliaStore", "MyceliaError"]

# LEIO/Code ingestion adapters (no external deps)
from .code_ingestion import (
    build_code_symbol_payloads,
    build_embeddings_by_symbol_from_vocab,
    infer_code_language,
    load_embedding_matrix,
    load_leio_query_cache,
    load_node_vocab,
    sync_leio_query_cache_from_artifacts,
    sync_leio_query_cache_to_mycelia,
)

__all__ += [
    "load_leio_query_cache",
    "load_node_vocab",
    "load_embedding_matrix",
    "infer_code_language",
    "build_code_symbol_payloads",
    "build_embeddings_by_symbol_from_vocab",
    "sync_leio_query_cache_from_artifacts",
    "sync_leio_query_cache_to_mycelia",
]

# Streaming JEPA (no external deps)
from .streaming import StreamBuffer, StreamingJEPA

__all__ += ["StreamingJEPA", "StreamBuffer"]

# Numpy-accelerated ops and mmap store (numpy optional)
from .numpy_ops import HAS_NUMPY, MmapEmbeddingStore

__all__ += ["HAS_NUMPY", "MmapEmbeddingStore"]

# Bandit client (no external deps — uses stdlib urllib)
from .bandit import BanditClient, BanditError, CascadeBandit

__all__ += ["BanditClient", "BanditError", "CascadeBandit"]

# GEPA search (no external deps — uses stdlib urllib)
from .gepa import GEPAError, GEPAResult, GEPASearch

__all__ += ["GEPASearch", "GEPAResult", "GEPAError"]

# Arrow Flight transfer is optional (requires pyarrow)
try:
    from .flight_transfer import FlightTransfer

    __all__ += ["FlightTransfer"]
except ImportError:
    pass

# Pipeline orchestrator is optional (requires duckdb)
try:
    from .orchestrator import Pipeline, PipelineError

    __all__ += ["Pipeline", "PipelineError"]
except ImportError:
    pass

# Digital Twin is optional (requires duckdb + pyarrow)
try:
    from .digital_twin import DigitalTwin, TwinSnapshot

    __all__ += ["DigitalTwin", "TwinSnapshot"]
except ImportError:
    pass

# Materializer is optional (requires duckdb + pyarrow)
try:
    from .materializer import MaterializationResult, Materializer, result_to_mycelia_payloads

    __all__ += ["Materializer", "MaterializationResult", "result_to_mycelia_payloads"]
except ImportError:
    pass
