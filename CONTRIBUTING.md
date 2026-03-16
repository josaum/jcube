# Contributing to Event-JEPA-Cube

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/josaum/jcube.git
   cd jcube
   ```

2. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

3. For PyTorch-dependent features:
   ```bash
   pip install -e ".[all]"
   ```

## Running Tests

```bash
pytest tests/ -v
```

## Code Quality

Before submitting a PR, ensure:

```bash
ruff check .
ruff format --check .
mypy event_jepa_cube/
```

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with clear commit messages
3. Add tests for new functionality
4. Ensure all checks pass
5. Submit a PR with a description of the changes

## Code Style

- Follow PEP 8 conventions
- Use type annotations for all public APIs
- Keep the core package free of external dependencies
- PyTorch-dependent code goes in `regularizers.py` with appropriate guards
