"""Tests for the embedding and model registries."""
from event_jepa_cube.registry import (
    _embedding_registry,
    _model_registry,
    get_embedding_type,
    get_model,
    register_embedding_type,
    register_model,
)


class TestEmbeddingRegistry:
    """Tests for embedding type registration."""

    def test_register_and_get_embedding_type(self):
        @register_embedding_type("test_emb_type")
        class MyEmbedding:
            pass

        result = get_embedding_type("test_emb_type")
        assert result is MyEmbedding

    def test_get_missing_embedding_type(self):
        result = get_embedding_type("does_not_exist_emb")
        assert result is None


class TestModelRegistry:
    """Tests for model registration."""

    def test_register_and_get_model(self):
        @register_model("test_model_type")
        class MyModel:
            pass

        result = get_model("test_model_type")
        assert result is MyModel

    def test_get_missing_model(self):
        result = get_model("does_not_exist_model")
        assert result is None


class TestDecoratorPreservesClass:
    """Test that decorators return the original class unchanged."""

    def test_decorator_preserves_class(self):
        @register_embedding_type("preserved_class_test")
        class OriginalClass:
            value = 42

        assert OriginalClass.value == 42
        assert OriginalClass is get_embedding_type("preserved_class_test")
