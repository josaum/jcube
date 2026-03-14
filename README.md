# Event-JEPA-Cube

A Python framework for processing long, irregular event sequences and multi-semantic entity relationships. Event-JEPA-Cube addresses limitations of standard Transformer architectures by offering hierarchical temporal processing and supporting multi-modal embeddings within a unified entity representation system.

This repository now includes a lightweight reference implementation of the core
classes along with a runnable `example.py` demonstrating basic usage.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

## Features

- **Event-JEPA**: Efficiently process long sequences of irregular events
- **Multi-Modal Support**: Works with text, images, audio, sensor data, or any embedding type
- **Embedding Cube**: Multi-dimensional embedding space for complex entity relationships
- **Hierarchical Processing**: Multi-level temporal aggregation for both micro and macro patterns
- **Multi-Semantic Entities**: Combine textual, visual, behavioral, and other contextual embeddings
- **JEPA Regularizers**: SIGReg, WeakSIGReg, and RDMReg for structured embedding spaces
- **Extensible Architecture**: Custom embeddings, relationship models, and hierarchical processors

## Installation

```bash
pip install event-jepa-cube              # Core (zero dependencies)
pip install event-jepa-cube[torch]       # With PyTorch for regularizers
pip install event-jepa-cube[dev]         # Development tools
```

## Quick Start

Here's a minimal example to get you started:

```python
from event_jepa_cube import EventJEPA, EventSequence

# Create event sequence from your data
sequence = EventSequence(
    embeddings=embeddings,      # List of embedding vectors
    timestamps=timestamps,       # List of timestamps
    modality='text'
)

# Process with hierarchical temporal aggregation
processor = EventJEPA(embedding_dim=768, num_levels=3)
representation = processor.process(sequence)
patterns = processor.detect_patterns(representation)
```

The file [`example.py`](example.py) contains a small runnable demonstration of
these steps.

## Core Components

### Event-JEPA

Event-JEPA specializes in handling long, irregularly timed sequences:

```python
from event_jepa_cube import EventJEPA, EventSequence

# Create event sequence
sequence = EventSequence(
    embeddings=embeddings,        # shape: [seq_len, embedding_dim]
    timestamps=timestamps,        # shape: [seq_len]
    modality='text'
)

# Initialize processor
processor = EventJEPA(
    embedding_dim=768,
    num_levels=3,
    temporal_resolution='adaptive'
)

# Process sequence
output = processor.process(sequence)

# Make predictions
predictions = processor.predict_next(sequence, num_steps=5)
```

### Embedding Cube

The Embedding Cube manages multi-semantic entity representations:

```python
from event_jepa_cube import EmbeddingCube, Entity

# Create multi-semantic entity
product = Entity(
    embeddings={
        'text': text_embedding,
        'image': image_embedding,
        'purchase_pattern': behavior_embedding
    },
    hierarchy_info={
        'type': 'product',
        'category': 'electronics',
        'subcategory': 'phones'
    }
)

cube = EmbeddingCube()
cube.add_entity(product)

# Find relationships
relationships = cube.discover_relationships(
    entity_ids=[product.id],
    threshold=0.8
)
```

## JEPA Regularizers

Event-JEPA-Cube includes regularizers from recent JEPA research to enforce
structured embedding distributions. These require PyTorch (`pip install event-jepa-cube[torch]`).

```python
from event_jepa_cube.regularizers import SIGReg, WeakSIGReg, RDMReg

# SIGReg: Enforce isotropic Gaussian distribution (LeJEPA)
sigreg = SIGReg(num_directions=64, sigma=1.0)
loss = sigreg.compute_loss(embedding_batch)  # PyTorch tensor

# WeakSIGReg: Covariance regularization for supervised training
weak = WeakSIGReg(sketch_dim=64)
loss = weak.compute_loss(embedding_batch)

# RDMReg: Sparse representations via Rectified Generalized Gaussian
rdmreg = RDMReg(p=2.0, target_sparsity=0.5, num_projections=64)
loss = rdmreg.compute_loss(embedding_batch)

# Integrate with EventJEPA
processor = EventJEPA(
    embedding_dim=768,
    regularizer=sigreg,
    reg_weight=0.05
)
```

## Advanced Usage

### Custom Embedding Types

Register your own embedding types:

```python
from event_jepa_cube import register_embedding_type

@register_embedding_type('custom')
class CustomEmbedding:
    def __init__(self, dim):
        self.dim = dim

    def process(self, data):
        # Your custom embedding logic
        return embeddings
```

### Custom Relationship Models

Add custom models to the Embedding Cube:

```python
from event_jepa_cube import register_model

@register_model('custom_relationship')
class CustomRelationshipModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.transform = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.transform(x)

cube.add_model('custom_relationship', CustomRelationshipModel(768, 128))
```

## Technical Architecture

### Performance Benchmarks

> **Note:** The figures below are theoretical targets based on the JEPA
> architecture's structural advantages over token-based Transformers. They are
> not measured benchmarks of this specific library.

| Metric | Traditional (Token-Based) | Event-JEPA-Cube | Improvement |
|--------|-------------------------|-----------------|-------------|
| Max Sequence Length | ~2K-4K tokens | 5K-10K+ events | ~5x-10x |
| Memory (GPU) | ~16GB | ~8GB | ~50% reduction |
| Processing Speed | 100 tokens/sec | 1000 events/sec | ~10x |
| Modality Support | Single/forced fusion | Native multi-modal | Significant |

### Scaling Advantages

- Handles 5K+ events efficiently (equivalent to tens of thousands of tokens)
- Native multi-modal support without forced alignment
- Memory efficient: O(m log m) vs O(n^2) in traditional Transformers
- Intelligent temporal processing for irregular timestamps

## Industry Applications

- **E-Commerce**: User journey modeling, product recommendations, dynamic pricing
- **Healthcare**: Patient journey analysis, treatment optimization, resource allocation
- **Manufacturing/IoT**: Predictive maintenance, anomaly detection, supply chain events
- **Security**: Intrusion detection, access pattern analysis, fraud detection

## References

The regularizer implementations are based on the following papers:

- **LeJEPA** (SIGReg): [arXiv:2511.08544](https://arxiv.org/abs/2511.08544)
- **Weak-SIGReg**: [arXiv:2603.05924](https://arxiv.org/abs/2603.05924)
- **Rectified LpJEPA** (RDMReg): [arXiv:2602.01456](https://arxiv.org/abs/2602.01456)

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on:

- Opening issues
- Submitting pull requests
- Code review process
- Development setup

## Citation

If you use Event-JEPA-Cube in your research, please cite:

```bibtex
@software{event_jepa_cube2024,
  author = {Agourakis, Dionisio Chiuratto},
  title = {Event-JEPA-Cube: Event Sequence Processing and Entity Relationships},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/josaum/jcube}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
