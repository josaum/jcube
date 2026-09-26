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
- Pre-flight before ANY paid training run: `--action smoke`, then the gated order in `docs/V63_PREFLIGHT.md`

## Architecture

The package has a layered dependency structure — core modules use only stdlib, with heavier deps imported lazily and guarded by try/except in `__init__.py`.

### Core Layer (zero dependencies)
- `sequence.py` — `EventSequence` and `Entity` dataclasses
- `event_jepa.py` — Hierarchical temporal processor (multi-level windowing, time-decay weighting, pattern detection, trend prediction)
- `embedding_cube.py` — Multi-modal entity manager (cosine similarity across shared modalities)
- `registry.py` — Plugin system (`@register_embedding_type`, `@register_model`)
- `training.py` — `CooldownSchedule` (V-JEPA 2.1 two-phase LR schedule)
- `streaming.py` — `StreamingJEPA` for O(1) per-event incremental processing with numpy acceleration
- `mycelia_store.py` — Mycelia vector DB client (Flight-first for bulk vector ingestion, HTTP for search/admin APIs, preserves scope + metadata + relations + text snippets)
- `code_ingestion.py` — LEIO Code / code-symbol ingestion adapters (query-cache → rich Mycelia payloads)
- `bandit.py` — Contextual bandit client for adaptive decisions (LinUCB via Mycelia API)
- `gepa.py` — GEPA evolutionary embedding search (remote via Mycelia or local in-memory)

### DuckDB Layer (requires `duckdb`, optionally `pyarrow`)
- `duckdb_connector.py` — Multi-source data warehouse connector (Postgres, MySQL, SQLite, Arrow Flight → DuckDB UNION ALL)
- `materializer.py` — Turns flat DB tables into `EventSequence` temporal lifecycles (entity-ID + timestamp → embeddings)
- `digital_twin.py` — DB introspection/profiling → rich metadata graph (`DigitalTwin`, `TwinSnapshot`)
- `triggers.py` — `TriggerEngine` watches tables for new records, runs JEPA incrementally, fires alert rules
- `cascade.py` — `ForecastCascade` chains trigger levels (patient → department → financial predictions)
- `orchestrator.py` — `Pipeline` wires all components: DuckDB → Cascade → Mycelia → StreamingJEPA → GEPA, and defaults Mycelia vector sync to `vector_ingest_transport="flight"`

### PyTorch Layer (requires `torch`)
- `regularizers.py` — SIGReg, WeakSIGReg, RDMReg, NormReg (JEPA embedding regularizers from LeJEPA/BiJEPA papers)
- `predictors.py` — `MLPPredictor`, `TransformerPredictor` (replace trend extrapolation with trained models)
- `bijepa.py` — `BiJEPATrainer` bidirectional cycle-consistent JEPA training (BiJEPA, arXiv:2603.00049; see `docs/SOTA_2026.md`)

### Training Layer (requires `torch` + `transformers` + `duckdb` + `pyarrow`)
- `lora_encoder.py` — Qwen backbone with LoRA; Phase A: extract/cache hidden states, Phase B: train projection
- `jepa_trainer.py` — JEPA training with curriculum learning (context encoder → EMA target → predictor)
- `graph_loader.py` — Graph-relational context loader (resolves entity relations across bridge tables)
- `scale_pipeline.py` — V5 full-scale TKG pipeline (417 tables → 165M edges → Parquet → GNN + JEPA)
- `scale_pipeline_v6.py` — V6 Dense Temporal JEPA (simplified: dense lookahead + Weak-SIGReg only, latent_dim=128)

### Deployment / Bridge
- `twin_api.py` — FastAPI server exposing digital twin + materializer (entry point: `digital-twin` CLI)
- `exploit_twin.py` — Semantic search / anomaly detection on trained embeddings (numpy, cosine similarity)
- `jcube_bridge.py` — Push embeddings to Mycelia/Milvus for production search through `MyceliaStore.store_vectors()` over Arrow Flight `do_exchange("vectors:{collection}")`
- `code_ingestion.py` keeps code-symbol sync backend-agnostic: training may happen on Modal, but ingestion into Mycelia should consume artifacts through a stable adapter instead of ad hoc scripts
- `modal_gpu.py` — Modal cloud GPU backend for training (A100-80GB)
- `run_probe.py` — Embedding quality probes (LightGBM) on Modal CPU

### Data Flow (production pipeline)
```
DB tables → Materializer → (S,P,O,T) Parquet → scale_pipeline_v6 (Modal GPU)
    → node_embeddings.pt → jcube_bridge (Flight-first bulk ingest) → Mycelia → agent queries

LEIO Code query-cache → code_ingestion.py → MyceliaStore.store_code_symbols()
    → code-symbol collections in Mycelia/Milvus
```

### Mycelia transport rules

- Vector ingestion is `Flight-first`. `MyceliaStore.store_vectors()` streams Arrow batches to `vectors:{collection}` and keeps the rich row contract intact.
- Search, collection lifecycle, and lighter control operations still use HTTP/JSON.
- `jcube_bridge.py` and `Pipeline` default to Flight for vector writes; only override to HTTP intentionally.
- The generic Mycelia Flight path preserves scope (`tenant_id`, `repo`, `rev`), `filter_tag`, and `metadata`, and folds unsupported extra columns into `metadata` instead of dropping them.
- The specialized `LEIO Code` Milvus reinjection script is a different path: it still talks to Milvus directly because it fills a custom multi-vector schema (`code_vec`, `semantic_vec`, `ontology_vec`, `execution_vec_bin`) rather than the generic Mycelia collection API.

## Conventions
- Type annotations on all public functions
- No external dependencies in core — torch, duckdb, pyarrow are optional with graceful ImportError fallback
- All public API exported from `__init__.py`
- Ruff config: line-length=120, target Python 3.9, rules E/F/I/UP/B
- Tests in `tests/` mirror module names (`test_<module>.py`)
- Use `LEIO Code` first when the task is cross-cutting across `jcube`, `mycelia`, deploy targets, or operational topology; prefer semantic lookup before raw grep.
- Retroaliment `LEIO Code` when a real drift is discovered around code ingestion, Mycelia sync, Flight transport, or Modal artifacts by adding a focused doctor, test, or canonical evidence path.
