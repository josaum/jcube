# Contributing to Event-JEPA-Cube

Event-JEPA-Cube supports Python 3.10–3.12. Use the `.[test]` extra for a complete
local test environment, or `.[all]` when working on PyTorch-dependent features.

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/josaum/jcube.git
   cd jcube
   ```

2. Install in development mode:
   ```bash
   pip install -e ".[test]"
   ```

3. For PyTorch-dependent features:
   ```bash
   pip install -e ".[all]"
   ```

## Running Tests

```bash
python -m pytest -q
```

## Code Quality

Before submitting a PR, ensure:

```bash
ruff check .
ruff format --check .
mypy event_jepa_cube/ --ignore-missing-imports
```

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with clear commit messages
3. Add tests for new functionality
4. Ensure all local checks pass and wait for lint, Python 3.10–3.12, and PyTorch CI jobs to go green
5. Submit a PR with a description of the changes and the validation performed

## Code Style

- Follow PEP 8 conventions
- Use type annotations for all public APIs
- Keep required runtime dependencies minimal: PyArrow and DuckDB are required, while other heavy integrations stay optional
- PyTorch-dependent code goes in `regularizers.py` with appropriate guards
