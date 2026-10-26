FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY event_jepa_cube ./event_jepa_cube
COPY tests ./tests
COPY examples ./examples

RUN python -m pip install ".[test]"

CMD ["python", "-m", "pytest", "-q"]
