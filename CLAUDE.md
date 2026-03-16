# CLAUDE.md

## Build & Test
- Install dev: `pip install -e ".[dev]"`
- Install with torch: `pip install -e ".[torch]"`
- Install all: `pip install -e ".[all]"`
- Test: `pytest tests/ -v`
- Lint: `ruff check .`
- Format: `ruff format .`
- Type check: `mypy event_jepa_cube/`

## Architecture
- Core package (`event_jepa_cube/`) has zero external dependencies
- Regularizers (`regularizers.py`) require PyTorch (optional dependency)
- All public API exported from `__init__.py`
- Extension system via decorator registries in `registry.py`

## Key Modules
- `sequence.py` — EventSequence and Entity dataclasses
- `event_jepa.py` — EventJEPA processor (temporal aggregation, pattern detection, prediction)
- `embedding_cube.py` — EmbeddingCube for multi-semantic entity relationships
- `regularizers.py` — SIGReg, WeakSIGReg, RDMReg (requires PyTorch)
- `registry.py` — Decorator-based plugin registries

## Conventions
- Type annotations on all public functions
- No external dependencies in core (torch is optional)
- Tests in `tests/` using pytest
