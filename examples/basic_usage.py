"""Minimal usage example for Event-JEPA-Cube."""

from event_jepa_cube import (
    CooldownSchedule,
    EmbeddingCube,
    Entity,
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

    # --- V-JEPA 2.1 improvements ---

    # Multi-level processing (deep self-supervision)
    processor_ml = EventJEPA(embedding_dim=3, num_levels=2)
    levels = processor_ml.process_multilevel(sequence)
    fused = EventJEPA.fuse_multilevel(levels)
    print("Multi-level representations:", len(levels), "levels")
    print("Fused representation:", fused)

    # Dense context loss
    context_embs = [[0.1, 0.2, 0.3], [0.2, 0.2, 0.2]]
    context_ts = [1.0, 3.0]
    target_embs = [[0.11, 0.19, 0.31], [0.19, 0.21, 0.19]]
    mask_ts = [2.0]
    dense_loss = processor.compute_dense_loss(
        context_embeddings=context_embs,
        context_timestamps=context_ts,
        target_embeddings=target_embs,
        mask_timestamps=mask_ts,
        prediction_loss=0.5,
        lambda_coeff=0.5,
    )
    print("Dense loss:", dense_loss)

    # Context lambda warmup schedule
    for step in [0, 25, 50, 75, 100, 150]:
        lam = EventJEPA.context_lambda_schedule(step, warmup_start=50, warmup_end=100)
        print(f"  Step {step}: lambda={lam:.3f}")

    # Modality-aware processing
    processor_modal = EventJEPA(embedding_dim=3, modality_aware=True)
    processor_modal.register_modality_config("audio", temporal_resolution="fixed", alpha=0.5)
    processor_modal.set_modality_offset("audio", [0.01, 0.01, 0.01])
    audio_seq = EventSequence(embeddings=embeddings, timestamps=timestamps, modality="audio")
    audio_rep = processor_modal.process(audio_seq)
    print("Audio representation:", audio_rep)

    # Position-aware prediction
    future_predictions = processor.predict_next_positional(sequence, target_timestamps=[4.0, 5.0, 10.0])
    print("Positional predictions:", future_predictions)

    # Cooldown training schedule
    schedule = CooldownSchedule(primary_steps=1000, cooldown_steps=200, warmup_steps=100)
    for step in [0, 50, 100, 500, 1000, 1100, 1200]:
        lr = schedule.get_lr_multiplier(step)
        res = schedule.get_resolution_scale(step)
        phase = "cooldown" if schedule.is_cooldown(step) else "primary"
        print(f"  Step {step}: lr_mult={lr:.4f}, resolution_scale={res:.1f} ({phase})")

    # Embedding cube relationships
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
