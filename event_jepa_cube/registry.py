"""Simple registries for embeddings and models."""
from typing import Callable, Dict

_embedding_registry: Dict[str, Callable] = {}
_model_registry: Dict[str, Callable] = {}


def register_embedding_type(name: str) -> Callable[[Callable], Callable]:
    """Decorator to register custom embedding types."""

    def decorator(cls_or_fn: Callable) -> Callable:
        _embedding_registry[name] = cls_or_fn
        return cls_or_fn

    return decorator


def get_embedding_type(name: str) -> Callable | None:
    return _embedding_registry.get(name)


def register_model(name: str) -> Callable[[Callable], Callable]:
    """Decorator to register custom relationship models."""

    def decorator(cls_or_fn: Callable) -> Callable:
        _model_registry[name] = cls_or_fn
        return cls_or_fn

    return decorator


def get_model(name: str) -> Callable | None:
    return _model_registry.get(name)
