"""Minimal usage example for Event-JEPA-Cube."""
from event_jepa_cube import (
    Entity,
    EmbeddingCube,
    EventJEPA,
    EventSequence,
)


def main() -> None:
    # Create dummy event sequence
    embeddings = [[0.1, 0.2, 0.3], [0.0, 0.1, 0.0], [0.2, 0.2, 0.2]]
    timestamps = [1.0, 2.0, 3.0]
    sequence = EventSequence(embeddings=embeddings, timestamps=timestamps)

    processor = EventJEPA(embedding_dim=3)
    representation = processor.process(sequence)
    patterns = processor.detect_patterns(representation)
    predictions = processor.predict_next(sequence, num_steps=2)

    print("Representation:", representation)
    print("Patterns:", patterns)
    print("Predictions:", predictions)

    cube = EmbeddingCube()
    product = Entity(
        embeddings={"text": [0.1, 0.2, 0.3]},
        hierarchy_info={"category": "electronics"},
    )
    cube.add_entity(product)
    relationships = cube.discover_relationships([product.id])
    print("Relationships:", relationships)


if __name__ == "__main__":
    main()
