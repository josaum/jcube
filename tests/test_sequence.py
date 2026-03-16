"""Tests for EventSequence and Entity data structures."""

import uuid

import pytest

from event_jepa_cube.sequence import Entity, EventSequence


class TestEventSequence:
    """Tests for EventSequence."""

    def test_creation_with_valid_data(self):
        seq = EventSequence(
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
            timestamps=[1.0, 2.0],
        )
        assert len(seq.embeddings) == 2
        assert len(seq.timestamps) == 2
        assert seq.modality == "text"

    def test_validation_error_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            EventSequence(
                embeddings=[[0.1, 0.2], [0.3, 0.4]],
                timestamps=[1.0],
            )

    def test_empty_sequence(self):
        seq = EventSequence(embeddings=[], timestamps=[])
        assert seq.embeddings == []
        assert seq.timestamps == []


class TestEntity:
    """Tests for Entity."""

    def test_auto_generates_uuid(self):
        entity = Entity(embeddings={"text": [0.1, 0.2]})
        # Should be a valid UUID string
        parsed = uuid.UUID(entity.id)
        assert str(parsed) == entity.id

    def test_custom_hierarchy_info(self):
        entity = Entity(
            embeddings={"text": [0.1]},
            hierarchy_info={"category": "electronics", "level": 2},
        )
        assert entity.hierarchy_info["category"] == "electronics"
        assert entity.hierarchy_info["level"] == 2

    def test_default_hierarchy_info_is_empty(self):
        entity = Entity(embeddings={"text": [0.1]})
        assert entity.hierarchy_info == {}
