"""Tests for EmbeddingCube."""

import math

import pytest

from event_jepa_cube.embedding_cube import EmbeddingCube
from event_jepa_cube.sequence import Entity


class TestAddEntity:
    """Tests for adding and retrieving entities."""

    def test_add_entity(self):
        cube = EmbeddingCube()
        entity = Entity(embeddings={"text": [0.1, 0.2, 0.3]})
        cube.add_entity(entity)
        assert entity.id in cube.entities
        assert cube.entities[entity.id] is entity


class TestDiscoverRelationships:
    """Tests for discover_relationships."""

    def test_discover_relationships_similar(self):
        cube = EmbeddingCube()
        e1 = Entity(
            embeddings={"text": [1.0, 0.0, 0.0]},
            hierarchy_info={"category": "A"},
            id="e1",
        )
        e2 = Entity(
            embeddings={"text": [1.0, 0.1, 0.0]},
            hierarchy_info={"category": "A"},
            id="e2",
        )
        cube.add_entity(e1)
        cube.add_entity(e2)
        rels = cube.discover_relationships(["e1", "e2"], threshold=0.5)
        assert "e1" in rels
        assert "e2" in rels["e1"]

    def test_discover_relationships_threshold(self):
        cube = EmbeddingCube()
        e1 = Entity(embeddings={"text": [1.0, 0.0, 0.0]}, id="e1")
        e2 = Entity(embeddings={"text": [0.5, 0.5, 0.5]}, id="e2")
        cube.add_entity(e1)
        cube.add_entity(e2)
        # Very high threshold should exclude weakly similar entities
        rels = cube.discover_relationships(["e1"], threshold=0.99)
        assert "e1" not in rels

    def test_discover_relationships_no_shared_modality(self):
        cube = EmbeddingCube()
        e1 = Entity(embeddings={"text": [1.0, 0.0]}, id="e1")
        e2 = Entity(embeddings={"image": [1.0, 0.0]}, id="e2")
        cube.add_entity(e1)
        cube.add_entity(e2)
        rels = cube.discover_relationships(["e1"], threshold=0.0)
        assert "e1" not in rels

    def test_discover_relationships_hierarchy_bonus(self):
        cube = EmbeddingCube()
        # Cosine similarity of these two is about 0.577 (1/sqrt3)
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [1.0 / math.sqrt(3), 1.0 / math.sqrt(3), 1.0 / math.sqrt(3)]
        # Without category match, sim ~ 0.577; threshold 0.65 would exclude
        e1 = Entity(embeddings={"text": vec_a}, hierarchy_info={"category": "X"}, id="e1")
        e2 = Entity(embeddings={"text": vec_b}, hierarchy_info={"category": "X"}, id="e2")
        cube.add_entity(e1)
        cube.add_entity(e2)
        # With +0.1 bonus: 0.577 + 0.1 = 0.677 >= 0.65
        rels = cube.discover_relationships(["e1"], threshold=0.65)
        assert "e1" in rels
        assert "e2" in rels["e1"]

        # Without category match, same threshold should exclude
        cube2 = EmbeddingCube()
        e3 = Entity(embeddings={"text": vec_a}, hierarchy_info={"category": "X"}, id="e3")
        e4 = Entity(embeddings={"text": vec_b}, hierarchy_info={"category": "Y"}, id="e4")
        cube2.add_entity(e3)
        cube2.add_entity(e4)
        rels2 = cube2.discover_relationships(["e3"], threshold=0.65)
        assert "e3" not in rels2


class TestCosineSimilarity:
    """Tests for _cosine_similarity static method."""

    def test_cosine_similarity_identical(self):
        sim = EmbeddingCube._cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert sim == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal(self):
        sim = EmbeddingCube._cosine_similarity([1.0, 0.0], [0.0, 1.0])
        assert sim == pytest.approx(0.0)


class TestModels:
    """Tests for model management."""

    def test_load_registered_model_missing(self):
        cube = EmbeddingCube()
        with pytest.raises(KeyError, match="not registered"):
            cube.load_registered_model("nonexistent_model")

    def test_add_model(self):
        cube = EmbeddingCube()

        class DummyModel:
            pass

        model = DummyModel()
        cube.add_model("dummy", model)
        assert "dummy" in cube.models
        assert cube.models["dummy"] is model
