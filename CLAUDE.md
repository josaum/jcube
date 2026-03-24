# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test
- Install dev: `pip install -e ".[dev]"`
- Install with torch: `pip install -e ".[torch]"`
- Install for training: `pip install -e ".[train]"` (torch + transformers + peft + accelerate + duckdb + pyarrow)
- Install all: `pip install -e ".[all]"`
- Test: `pytest tests/ -v`
- Single test: `pytest tests/test_streaming.py -v`
- Lint: `ruff check .`
- Format: `ruff format .`
- Type check: `mypy event_jepa_cube/`
- Run digital twin API: `digital-twin --db /path/to/db.duckdb --port 8000`
- GPU training (Modal): `modal run event_jepa_cube/scale_pipeline_v6.py --action full`

## Architecture

The package has a layered dependency structure — core modules use only stdlib, with heavier deps imported lazily and guarded by try/except in `__init__.py`.

### Core Layer (zero dependencies)
- `sequence.py` — `EventSequence` and `Entity` dataclasses
- `event_jepa.py` — Hierarchical temporal processor (multi-level windowing, time-decay weighting, pattern detection, trend prediction)
- `embedding_cube.py` — Multi-modal entity manager (cosine similarity across shared modalities)
- `registry.py` — Plugin system (`@register_embedding_type`, `@register_model`)
- `training.py` — `CooldownSchedule` (V-JEPA 2.1 two-phase LR schedule)
- `streaming.py` — `StreamingJEPA` for O(1) per-event incremental processing with numpy acceleration
- `mycelia_store.py` — Mycelia vector DB client (stdlib urllib, stores/searches embeddings)
- `bandit.py` — Contextual bandit client for adaptive decisions (LinUCB via Mycelia API)
- `gepa.py` — GEPA evolutionary embedding search (remote via Mycelia or local in-memory)

### DuckDB Layer (requires `duckdb`, optionally `pyarrow`)
- `duckdb_connector.py` — Multi-source data warehouse connector (Postgres, MySQL, SQLite, Arrow Flight → DuckDB UNION ALL)
- `materializer.py` — Turns flat DB tables into `EventSequence` temporal lifecycles (entity-ID + timestamp → embeddings)
- `digital_twin.py` — DB introspection/profiling → rich metadata graph (`DigitalTwin`, `TwinSnapshot`)
- `triggers.py` — `TriggerEngine` watches tables for new records, runs JEPA incrementally, fires alert rules
- `cascade.py` — `ForecastCascade` chains trigger levels (patient → department → financial predictions)
- `orchestrator.py` — `Pipeline` wires all components: DuckDB → Cascade → Mycelia → StreamingJEPA → GEPA

### PyTorch Layer (requires `torch`)
- `regularizers.py` — SIGReg, WeakSIGReg, RDMReg (JEPA embedding regularizers from LeJEPA papers)
- `predictors.py` — `MLPPredictor`, `TransformerPredictor` (replace trend extrapolation with trained models)

### Training Layer (requires `torch` + `transformers` + `duckdb` + `pyarrow`)
- `lora_encoder.py` — Qwen backbone with LoRA; Phase A: extract/cache hidden states, Phase B: train projection
- `jepa_trainer.py` — JEPA training with curriculum learning (context encoder → EMA target → predictor)
- `graph_loader.py` — Graph-relational context loader (resolves entity relations across bridge tables)
- `scale_pipeline.py` — V5 full-scale TKG pipeline (417 tables → 165M edges → Parquet → GNN + JEPA)
- `scale_pipeline_v6.py` — V6 Dense Temporal JEPA (simplified: dense lookahead + Weak-SIGReg only, latent_dim=128)

### Deployment / Bridge
- `twin_api.py` — FastAPI server exposing digital twin + materializer (entry point: `digital-twin` CLI)
- `exploit_twin.py` — Semantic search / anomaly detection on trained embeddings (numpy, cosine similarity)
- `jcube_bridge.py` — Push 17M×64 embeddings to Mycelia/Milvus for production search
- `modal_gpu.py` — Modal cloud GPU backend for training (A100-80GB)
- `run_probe.py` — Embedding quality probes (LightGBM) on Modal CPU

### Data Flow (production pipeline)
```
DB tables → Materializer → (S,P,O,T) Parquet → scale_pipeline_v6 (Modal GPU)
    → node_embeddings.pt → jcube_bridge → Mycelia → agent queries
```

## Conventions
- Type annotations on all public functions
- No external dependencies in core — torch, duckdb, pyarrow are optional with graceful ImportError fallback
- All public API exported from `__init__.py`
- Ruff config: line-length=120, target Python 3.9, rules E/F/I/UP/B
- Tests in `tests/` mirror module names (`test_<module>.py`)
