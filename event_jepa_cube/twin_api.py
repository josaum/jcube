"""FastAPI server that builds and serves a digital twin of any DuckDB database.

Usage::

    # Start server pointing at a local DB
    uvicorn event_jepa_cube.twin_api:app --reload

    # Then POST to connect:
    curl -X POST http://localhost:8000/twin/connect \
      -H 'Content-Type: application/json' \
      -d '{"db_path": "/path/to/my.db"}'

    # Or run directly:
    python -m event_jepa_cube.twin_api --db /path/to/my.db --port 8000
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .digital_twin import DigitalTwin, snapshot_to_dict
from .materializer import Materializer, result_to_dict

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

_twins: dict[str, DigitalTwin] = {}        # db_path → twin
_snapshots: dict[str, dict[str, Any]] = {} # db_path → serialized snapshot
_materializers: dict[str, Materializer] = {}  # db_path → materializer
_active_db: str | None = None

# Catalog state — loaded once, keyed by catalog path
_catalogs: dict[str, dict[str, Any]] = {}  # catalog_path → {"meta": ..., "tables": {name: entry}}

# Embedding state — lazy loaded on first request
_emb_data: dict[str, Any] = {}  # "embeddings", "node_names", "node_to_idx", "dim", "loaded"

# CLI-configured paths (set in main() before server starts)
_default_catalog_path: str = "data/ai_friendly_catalog.json"
_default_graph_path: str = "data/jcube_graph.parquet"
_default_weights_path: str = "data/weights/node_emb_epoch_2.pt"


# ---------------------------------------------------------------------------
# Catalog helpers
# ---------------------------------------------------------------------------


def _load_catalog(catalog_path: str) -> dict[str, Any]:
    """Load and index catalog from JSON file. Returns {"meta": ..., "tables": {name: entry}}."""
    if catalog_path in _catalogs:
        return _catalogs[catalog_path]

    if not os.path.exists(catalog_path):
        raise FileNotFoundError(f"Catalog not found: {catalog_path}")

    with open(catalog_path, encoding="utf-8") as f:
        raw = json.load(f)

    tables_list: list[dict[str, Any]] = raw.get("tables", [])
    tables_by_name: dict[str, dict[str, Any]] = {t["name"]: t for t in tables_list}

    result: dict[str, Any] = {
        "meta": {k: v for k, v in raw.items() if k != "tables"},
        "tables": tables_by_name,
    }
    _catalogs[catalog_path] = result
    return result


def _catalog_to_domain_groups(catalog: dict[str, Any]) -> dict[str, Any]:
    """Build domain_groups dict from catalog categories."""
    groups: dict[str, dict[str, Any]] = {}
    for tname, entry in catalog["tables"].items():
        cat = entry.get("category", "General")
        if cat not in groups:
            groups[cat] = {"tables": [], "total_rows": 0, "description": cat}
        groups[cat]["tables"].append(tname)
        groups[cat]["total_rows"] += entry.get("row_count", 0)
    return groups


def _catalog_fks(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract FK relationships from catalog into a flat list."""
    fks = []
    for tname, entry in catalog["tables"].items():
        for rel in entry.get("relationships", []):
            if rel.get("relationship_type") == "foreign_key":
                fks.append({
                    "from_table": tname,
                    "from_column": rel.get("column", ""),
                    "to_table": rel.get("likely_references", ""),
                    "confidence": 1.0,
                    "join_example": rel.get("join_example", ""),
                })
    return fks


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------


def _load_embeddings(
    graph_path: str | None = None,
    weights_path: str | None = None,
) -> None:
    """Load embeddings lazily on first request."""
    if _emb_data.get("loaded"):
        return

    gp = graph_path or _default_graph_path
    wp = weights_path or _default_weights_path

    import torch
    import pyarrow.parquet as pq
    import pyarrow as pa
    import pyarrow.compute as pc

    print(f"Loading embedding graph vocabulary from {gp}...")
    t0 = time.time()
    table = pq.read_table(gp, columns=["subject_id", "object_id"])
    all_nodes = pa.chunked_array(
        table.column("subject_id").chunks + table.column("object_id").chunks
    )
    unique_nodes = pc.unique(all_nodes)
    node_names = unique_nodes.to_numpy(zero_copy_only=False).astype(object)
    del table, all_nodes, unique_nodes

    print(f"Loading weights from {wp}...")
    state = torch.load(wp, map_location="cpu", weights_only=True)
    if isinstance(state, torch.Tensor):
        embeddings = state.float().numpy()
    elif isinstance(state, dict) and "weight" in state:
        embeddings = state["weight"].float().numpy()
    else:
        embeddings = list(state.values())[0].float().numpy()

    if len(node_names) != embeddings.shape[0]:
        print(f"WARNING: {len(node_names)} names vs {embeddings.shape[0]} vectors — truncating")
        n = min(len(node_names), embeddings.shape[0])
        node_names = node_names[:n]
        embeddings = embeddings[:n]

    _emb_data["embeddings"] = embeddings
    _emb_data["node_names"] = node_names
    _emb_data["node_to_idx"] = {str(name): i for i, name in enumerate(node_names)}
    _emb_data["dim"] = embeddings.shape[1]
    _emb_data["loaded"] = True
    print(f"Embeddings loaded: {embeddings.shape} in {time.time() - t0:.1f}s")


def _ensure_embeddings() -> None:
    if not _emb_data.get("loaded"):
        _load_embeddings()


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI):  # type: ignore[type-arg]
    yield
    for twin in _twins.values():
        twin.close()
    for mat in _materializers.values():
        mat.close()
    _twins.clear()
    _snapshots.clear()
    _materializers.clear()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Digital Twin API",
    description="Connect to any DuckDB database and get a full digital twin: "
    "schema, column profiles, FK relationships, domain grouping, embedding search.",
    version="0.2.0",
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class ConnectRequest(BaseModel):
    db_path: str = Field(..., description="Path to a DuckDB database file")
    read_only: bool = Field(True, description="Open in read-only mode")
    profile_columns: bool = Field(True, description="Profile every column")
    discover_fks: bool = Field(True, description="Discover FK relationships (skipped if catalog_path given)")
    max_fk_checks: int = Field(2000, description="Max FK overlap checks")
    catalog_path: str | None = Field(
        None,
        description="Path to ai_friendly_catalog.json. If provided, FKs and domain groups "
        "are loaded from catalog instead of being inferred at runtime. "
        "Defaults to data/ai_friendly_catalog.json if that file exists.",
    )


class QueryRequest(BaseModel):
    sql: str = Field(..., description="SQL query to execute")
    db_path: str | None = Field(None, description="Target DB (uses active if omitted)")
    limit: int = Field(1000, description="Max rows to return")


class TableSampleRequest(BaseModel):
    table: str
    db_path: str | None = None
    limit: int = Field(20, ge=1, le=1000)


class MaterializeRequest(BaseModel):
    entity_type: str = Field(..., description="Entity type: INTERNACAO, PACIENTE, FATURA, etc.")
    db_path: str | None = Field(None, description="Target DB (uses active if omitted)")
    limit_entities: int | None = Field(100, description="Max entities to materialize (None = all)")
    limit_events_per_entity: int | None = Field(5000, description="Max events per entity")
    source_db_filter: str | None = Field(None, description="Only use rows from this source_db")
    table_filter: list[str] | None = Field(None, description="Only use these tables")
    run_jepa: bool = Field(True, description="Run EventJEPA on materialized sequences")
    num_levels: int = Field(2, description="EventJEPA hierarchical levels")
    temporal_resolution: str = Field("adaptive", description="adaptive or fixed")
    num_prediction_steps: int = Field(3, description="Prediction horizon")


class VectorSearchRequest(BaseModel):
    vector: list[float] = Field(..., description="Query vector (must match embedding dim)")
    k: int = Field(5, ge=1, le=500, description="Number of nearest neighbours")
    entity_type: str = Field("", description="Filter results to this entity type prefix")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_materializer(db_path: str | None = None) -> Materializer:
    global _active_db
    path = db_path or _active_db
    if path is None:
        raise HTTPException(404, "No database connected. POST /twin/connect first.")
    if path not in _materializers:
        mat = Materializer(path, read_only=True)
        mat.connect()
        mat.scan()
        _materializers[path] = mat
    return _materializers[path]


def _get_twin(db_path: str | None = None) -> DigitalTwin:
    global _active_db
    path = db_path or _active_db
    if path is None or path not in _twins:
        raise HTTPException(404, "No twin connected. POST /twin/connect first.")
    return _twins[path]


def _get_snapshot(db_path: str | None = None) -> dict[str, Any]:
    global _active_db
    path = db_path or _active_db
    if path is None or path not in _snapshots:
        raise HTTPException(404, "No snapshot built. POST /twin/connect first.")
    return _snapshots[path]


def _resolve_catalog_path(catalog_path: str | None) -> str | None:
    """Return an effective catalog path: explicit arg > default file if it exists > None."""
    if catalog_path:
        return catalog_path
    if os.path.exists(_default_catalog_path):
        return _default_catalog_path
    return None


# ---------------------------------------------------------------------------
# /twin/connect
# ---------------------------------------------------------------------------


@app.post("/twin/connect")
async def connect(req: ConnectRequest) -> dict[str, Any]:
    """Connect to a DuckDB database and build its digital twin.

    If a catalog_path is provided (or data/ai_friendly_catalog.json exists),
    FK relationships and domain groups are loaded from it instead of being
    discovered at runtime — much faster for large warehouses.
    """
    global _active_db

    if req.db_path in _twins:
        _twins[req.db_path].close()

    effective_catalog = _resolve_catalog_path(req.catalog_path)
    use_catalog = effective_catalog is not None

    twin = DigitalTwin(req.db_path, read_only=req.read_only)
    loop = asyncio.get_event_loop()

    # Build twin; skip FK discovery when catalog provides them
    snapshot = await loop.run_in_executor(
        None,
        lambda: twin.build(
            profile_columns=req.profile_columns,
            discover_fks=(req.discover_fks and not use_catalog),
            max_fk_checks=req.max_fk_checks,
        ),
    )

    serialized = snapshot_to_dict(snapshot)

    # Overlay catalog FK + domain data when available
    if use_catalog:
        try:
            catalog = _load_catalog(effective_catalog)  # type: ignore[arg-type]

            # Inject domain_group into each table entry from catalog category
            for tname, tdata in serialized.get("tables", {}).items():
                cat_entry = catalog["tables"].get(tname)
                if cat_entry:
                    tdata["domain_group"] = cat_entry.get("category", "General")

            # Replace domain_groups with catalog-derived ones
            serialized["domain_groups"] = _catalog_to_domain_groups(catalog)

            # Replace FK list with catalog-derived one
            serialized["foreign_keys"] = _catalog_fks(catalog)

            # Attach catalog path for reference
            serialized["catalog_path"] = effective_catalog

        except Exception as exc:
            # Non-fatal: log and continue with whatever was built
            print(f"WARNING: Failed to overlay catalog from {effective_catalog}: {exc}")

    _twins[req.db_path] = twin
    _snapshots[req.db_path] = serialized
    _active_db = req.db_path

    domain_summary = {
        k: {"table_count": len(v["tables"]) if isinstance(v.get("tables"), list) else 0,
            "total_rows": v.get("total_rows", 0),
            "description": v.get("description", k)}
        for k, v in serialized.get("domain_groups", {}).items()
    }

    return {
        "status": "connected",
        "db_path": req.db_path,
        "db_size_mb": snapshot.db_size_mb,
        "total_tables": snapshot.total_tables,
        "total_rows": snapshot.total_rows,
        "total_columns": snapshot.total_columns,
        "source_databases": snapshot.source_databases,
        "domain_groups": domain_summary,
        "foreign_keys_found": len(serialized.get("foreign_keys", [])),
        "catalog_loaded": use_catalog,
        "catalog_path": effective_catalog,
        "fingerprint": snapshot.fingerprint,
        "build_duration_s": snapshot.build_duration_s,
    }


# ---------------------------------------------------------------------------
# Core schema / table endpoints
# ---------------------------------------------------------------------------


@app.get("/twin/snapshot")
async def get_snapshot(db_path: str | None = None) -> dict[str, Any]:
    """Return the full digital twin snapshot (all tables, columns, FKs)."""
    return _get_snapshot(db_path)


@app.get("/twin/tables")
async def list_tables(
    db_path: str | None = None,
    domain: str | None = Query(None, description="Filter by domain/category group"),
    min_rows: int = Query(0, description="Minimum row count"),
    sort_by: str = Query("name", description="Sort by: name, rows, columns"),
) -> list[dict[str, Any]]:
    """List all tables with summary stats, optionally filtered by domain/category."""
    snap = _get_snapshot(db_path)
    tables = list(snap["tables"].values())

    if domain:
        tables = [t for t in tables if t.get("domain_group") == domain]
    if min_rows > 0:
        tables = [t for t in tables if t.get("row_count", 0) >= min_rows]

    key_map = {"name": "name", "rows": "row_count", "columns": "column_count"}
    sort_key = key_map.get(sort_by, "name")
    tables.sort(key=lambda t: t.get(sort_key, 0), reverse=(sort_by in ("rows", "columns")))

    return [
        {
            "name": t["name"],
            "row_count": t.get("row_count", 0),
            "column_count": t.get("column_count", 0),
            "domain_group": t.get("domain_group", "General"),
            "size_category": t.get("size_category", ""),
            "has_timestamps": t.get("has_timestamps", False),
            "source_databases": t.get("source_databases", []),
        }
        for t in tables
    ]


@app.get("/twin/table/{table_name}")
async def get_table(table_name: str, db_path: str | None = None) -> dict[str, Any]:
    """Get full profile for a specific table including all column details."""
    snap = _get_snapshot(db_path)
    if table_name not in snap["tables"]:
        raise HTTPException(404, f"Table '{table_name}' not found")
    return snap["tables"][table_name]


@app.get("/twin/table/{table_name}/sample")
async def sample_table(
    table_name: str,
    limit: int = Query(20, ge=1, le=1000),
    db_path: str | None = None,
) -> dict[str, Any]:
    """Get sample rows from a table."""
    twin = _get_twin(db_path)
    try:
        rows = twin.table_sample(table_name, limit=limit)
    except Exception as e:
        raise HTTPException(400, str(e))
    return {"table": table_name, "count": len(rows), "rows": rows}


@app.get("/twin/domains")
async def list_domains(db_path: str | None = None) -> dict[str, Any]:
    """List all domain groups / categories with their tables and stats."""
    snap = _get_snapshot(db_path)
    return snap["domain_groups"]


@app.get("/twin/foreign-keys")
async def list_foreign_keys(
    db_path: str | None = None,
    min_confidence: float = Query(0.0, ge=0, le=1),
    table: str | None = Query(None, description="Filter by table name (from or to)"),
) -> list[dict[str, Any]]:
    """List FK relationships (from catalog if available, otherwise discovered at runtime)."""
    snap = _get_snapshot(db_path)
    fks = snap.get("foreign_keys", [])

    if min_confidence > 0:
        fks = [fk for fk in fks if fk.get("confidence", 1.0) >= min_confidence]
    if table:
        fks = [fk for fk in fks if table in (fk.get("from_table", ""), fk.get("to_table", ""))]

    return fks


@app.get("/twin/sources")
async def list_sources(db_path: str | None = None) -> dict[str, Any]:
    """List all source databases found in the data."""
    snap = _get_snapshot(db_path)
    return {
        "source_databases": snap.get("source_databases", []),
        "total": len(snap.get("source_databases", [])),
    }


@app.post("/twin/query")
async def run_query(req: QueryRequest) -> dict[str, Any]:
    """Execute arbitrary read-only SQL and return results as JSON."""
    twin = _get_twin(req.db_path)
    try:
        loop = asyncio.get_event_loop()
        arrow = await loop.run_in_executor(None, lambda: twin.query_arrow(req.sql))
        if len(arrow) > req.limit:
            arrow = arrow.slice(0, req.limit)
        rows = arrow.to_pylist()
        return {
            "sql": req.sql,
            "row_count": len(rows),
            "columns": arrow.schema.names,
            "rows": rows,
        }
    except Exception as e:
        raise HTTPException(400, f"Query error: {e}")


@app.get("/twin/graph")
async def get_graph(
    db_path: str | None = None,
    min_confidence: float = Query(0.3, description="Min FK confidence to include edges"),
) -> dict[str, Any]:
    """Return the twin as a graph (nodes=tables, edges=FK relationships).

    Useful for visualization with tools like D3 or Cytoscape.
    """
    snap = _get_snapshot(db_path)

    nodes = [
        {
            "id": name,
            "row_count": t.get("row_count", 0),
            "domain": t.get("domain_group", "General"),
            "size": t.get("size_category", ""),
            "columns": t.get("column_count", 0),
        }
        for name, t in snap["tables"].items()
    ]

    edges = [
        {
            "source": fk.get("from_table", ""),
            "target": fk.get("to_table", ""),
            "column": fk.get("from_column", ""),
            "confidence": fk.get("confidence", 1.0),
        }
        for fk in snap.get("foreign_keys", [])
        if fk.get("confidence", 1.0) >= min_confidence
    ]

    return {"nodes": nodes, "edges": edges}


@app.get("/twin/search")
async def search(
    q: str = Query(..., description="Search term (matches table/column names)"),
    db_path: str | None = None,
) -> dict[str, Any]:
    """Search tables and columns by name."""
    snap = _get_snapshot(db_path)
    q_lower = q.lower()
    table_hits = []
    column_hits = []

    for tname, tdata in snap["tables"].items():
        if q_lower in tname.lower():
            table_hits.append({
                "table": tname,
                "row_count": tdata.get("row_count", 0),
                "domain": tdata.get("domain_group", "General"),
            })
        for col in tdata.get("columns", []):
            col_name = col.get("name", col) if isinstance(col, dict) else str(col)
            if q_lower in col_name.lower():
                column_hits.append({
                    "table": tname,
                    "column": col_name,
                    "dtype": col.get("dtype", "") if isinstance(col, dict) else "",
                    "distinct_count": col.get("distinct_count", None) if isinstance(col, dict) else None,
                })

    return {
        "query": q,
        "table_matches": len(table_hits),
        "column_matches": len(column_hits),
        "tables": table_hits[:50],
        "columns": column_hits[:50],
    }


# ---------------------------------------------------------------------------
# Catalog endpoints
# ---------------------------------------------------------------------------


@app.get("/twin/catalog/table/{name}")
async def get_catalog_entry(
    name: str,
    catalog_path: str | None = Query(None, description="Path to catalog JSON (uses default if omitted)"),
) -> dict[str, Any]:
    """Return the full catalog entry for a table.

    Includes business_context, description, relationships, key_concepts,
    common_queries, columns with healthcare annotations, and more.
    """
    effective = _resolve_catalog_path(catalog_path)
    if effective is None:
        raise HTTPException(404, "No catalog available. Provide --catalog on startup or pass catalog_path.")
    try:
        catalog = _load_catalog(effective)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))

    entry = catalog["tables"].get(name)
    if entry is None:
        raise HTTPException(404, f"Table '{name}' not found in catalog")
    return entry


@app.get("/twin/catalog/meta")
async def get_catalog_meta(
    catalog_path: str | None = Query(None, description="Path to catalog JSON"),
) -> dict[str, Any]:
    """Return catalog-level metadata: terminology, query guidelines, statistics."""
    effective = _resolve_catalog_path(catalog_path)
    if effective is None:
        raise HTTPException(404, "No catalog available.")
    try:
        catalog = _load_catalog(effective)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    return catalog["meta"]


@app.get("/twin/catalog/categories")
async def get_catalog_categories(
    catalog_path: str | None = Query(None, description="Path to catalog JSON"),
) -> dict[str, Any]:
    """List all category groups from the catalog with table names."""
    effective = _resolve_catalog_path(catalog_path)
    if effective is None:
        raise HTTPException(404, "No catalog available.")
    try:
        catalog = _load_catalog(effective)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    return _catalog_to_domain_groups(catalog)


# ---------------------------------------------------------------------------
# Temporal / EventJEPA endpoints
# ---------------------------------------------------------------------------


@app.get("/twin/entity-types")
async def list_entity_types(db_path: str | None = None) -> dict[str, Any]:
    """Discover entity types in the database and how many tables/rows each spans."""
    loop = asyncio.get_event_loop()
    mat = await loop.run_in_executor(None, lambda: _get_materializer(db_path))
    summary = mat.scan_entity_types()
    return {
        "entity_types": {
            etype: {
                "table_count": len(info["tables"]),
                "total_rows": info["total_rows"],
                "entity_column": info["entity_column"],
                "tables": info["tables"][:20],
            }
            for etype, info in summary.items()
        },
        "total_types": len(summary),
    }


@app.post("/twin/materialize")
async def materialize(req: MaterializeRequest) -> dict[str, Any]:
    """Materialize temporal event sequences for an entity type.

    Groups rows by entity ID across all matching tables, orders by timestamp,
    encodes context columns into embeddings, and optionally runs EventJEPA.
    """
    loop = asyncio.get_event_loop()
    mat = await loop.run_in_executor(None, lambda: _get_materializer(req.db_path))

    result = await loop.run_in_executor(
        None,
        lambda: mat.materialize(
            req.entity_type,
            limit_entities=req.limit_entities,
            limit_events_per_entity=req.limit_events_per_entity,
            table_filter=req.table_filter,
            source_db_filter=req.source_db_filter,
        ),
    )

    if req.run_jepa and result.entities_found > 0:
        result = await loop.run_in_executor(
            None,
            lambda: mat.process(
                result,
                num_levels=req.num_levels,
                temporal_resolution=req.temporal_resolution,
                num_prediction_steps=req.num_prediction_steps,
            ),
        )

    return result_to_dict(result)


@app.get("/twin/entity/{entity_type}/{entity_id}")
async def get_entity_timeline(
    entity_type: str,
    entity_id: str,
    db_path: str | None = None,
    run_jepa: bool = Query(True, description="Run EventJEPA on this entity"),
    num_prediction_steps: int = Query(3, description="Prediction horizon"),
) -> dict[str, Any]:
    """Get the full temporal lifecycle for a single entity."""
    loop = asyncio.get_event_loop()
    mat = await loop.run_in_executor(None, lambda: _get_materializer(db_path))

    result = await loop.run_in_executor(
        None,
        lambda: mat.materialize(
            entity_type,
            limit_entities=None,
            limit_events_per_entity=10000,
        ),
    )

    if entity_id not in result.timelines:
        raise HTTPException(404, f"Entity {entity_type}/{entity_id} not found")

    timeline = result.timelines[entity_id]
    result.timelines = {entity_id: timeline}
    result.entities_found = 1
    result.total_events = timeline.event_count

    if run_jepa and timeline.event_count >= 2:
        result = await loop.run_in_executor(
            None,
            lambda: mat.process(result, num_prediction_steps=num_prediction_steps),
        )

    serialized = result_to_dict(result)
    seq = timeline.sequence
    serialized["event_timestamps"] = [
        {"index": i, "epoch": seq.timestamps[i]}
        for i in range(len(seq.timestamps))
    ]
    return serialized


# ---------------------------------------------------------------------------
# Embedding endpoints (V5 prod embeddings — lazy loaded)
# ---------------------------------------------------------------------------


class SimilarResult(BaseModel):
    node_id: str
    similarity: float


class AnomalyResult(BaseModel):
    node_id: str
    z_score: float
    distance: float


@app.get("/twin/embedding/info")
async def embedding_info() -> dict[str, Any]:
    """Get embedding metadata (triggers lazy load)."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _ensure_embeddings)
    return {
        "num_nodes": len(_emb_data["node_names"]),
        "dim": _emb_data["dim"],
        "loaded": True,
    }


@app.get("/twin/similar/{node_id}")
async def find_similar(
    node_id: str,
    k: int = Query(default=5, ge=1, le=500),
    entity_type: str = Query(default="", description="Filter results to this entity type"),
) -> dict[str, Any]:
    """Find the k nearest neighbours to node_id by cosine similarity.

    node_id formats accepted:
      - V4: GHO-BRADESCO/ID_CD_INTERNACAO_123
      - V3: ID_CD_INTERNACAO_123
    """
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _ensure_embeddings)

    embeddings: np.ndarray = _emb_data["embeddings"]
    node_names: np.ndarray = _emb_data["node_names"]
    node_to_idx: dict[str, int] = _emb_data["node_to_idx"]

    if node_id not in node_to_idx:
        raise HTTPException(404, f"Node '{node_id}' not found in embedding space")

    idx = node_to_idx[node_id]
    query_vec = embeddings[idx].reshape(1, -1)
    nq = float(np.linalg.norm(query_vec).clip(min=1e-8))

    if entity_type:
        candidates = [
            i for i, name in enumerate(node_names)
            if entity_type in str(name) and i != idx
        ]
        if not candidates:
            return {"query": node_id, "matches": [], "note": f"No nodes matching type '{entity_type}'"}
        candidate_idx = np.array(candidates)
        candidate_vecs = embeddings[candidate_idx]
    else:
        candidate_idx = np.arange(len(embeddings))
        candidate_vecs = embeddings

    nc = np.linalg.norm(candidate_vecs, axis=1).clip(min=1e-8)
    sims = (candidate_vecs @ query_vec.T).squeeze() / (nc * nq)

    top_k = min(k + 1, len(sims))
    top_indices = np.argpartition(-sims, top_k)[:top_k]
    top_indices = top_indices[np.argsort(-sims[top_indices])]

    matches = []
    for i in top_indices:
        real_idx = int(candidate_idx[i]) if entity_type else int(i)
        if real_idx == idx:
            continue
        matches.append({"node_id": str(node_names[real_idx]), "similarity": round(float(sims[i]), 4)})
        if len(matches) >= k:
            break

    return {"query": node_id, "matches": matches}


@app.get("/twin/anomalies/{entity_type}")
async def find_anomalies(
    entity_type: str,
    top: int = Query(default=10, ge=1, le=500),
    source_db: str = Query(default="", description="Filter by hospital/source prefix"),
) -> dict[str, Any]:
    """Find top anomalous nodes of an entity type by z-score distance from centroid."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _ensure_embeddings)

    embeddings: np.ndarray = _emb_data["embeddings"]
    node_names: np.ndarray = _emb_data["node_names"]

    matches = []
    for i, name in enumerate(node_names):
        s = str(name)
        if f"_CD_{entity_type}_" not in s and f"_{entity_type}_" not in s:
            continue
        if source_db and source_db not in s:
            continue
        matches.append(i)

    if not matches:
        raise HTTPException(404, f"No nodes found for entity type '{entity_type}'")

    idx = np.array(matches)
    vecs = embeddings[idx]
    centroid = vecs.mean(axis=0, keepdims=True)
    dists = np.linalg.norm(vecs - centroid, axis=1)
    mean_d = float(dists.mean())
    std_d = float(max(dists.std(), 1e-8))

    top_n = min(top, len(dists))
    top_indices = np.argpartition(-dists, top_n)[:top_n]
    top_indices = top_indices[np.argsort(-dists[top_indices])]

    results = [
        {
            "node_id": str(node_names[idx[i]]),
            "z_score": round(float((dists[i] - mean_d) / std_d), 2),
            "distance": round(float(dists[i]), 4),
        }
        for i in top_indices
    ]

    return {
        "entity_type": entity_type,
        "source_db": source_db or "all",
        "total_nodes": len(matches),
        "mean_dist": round(mean_d, 4),
        "std_dist": round(std_d, 4),
        "anomalies": results,
    }


@app.post("/twin/search-vector")
async def search_by_vector(req: VectorSearchRequest) -> dict[str, Any]:
    """Find nearest nodes given a raw query vector.

    Body: {"vector": [...64 floats...], "k": 5, "entity_type": "INTERNACAO"}
    """
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _ensure_embeddings)

    query_vec = np.array(req.vector, dtype=np.float32).reshape(1, -1)
    if query_vec.shape[1] != _emb_data["dim"]:
        raise HTTPException(
            400,
            f"Vector dim {query_vec.shape[1]} does not match embedding dim {_emb_data['dim']}",
        )

    embeddings: np.ndarray = _emb_data["embeddings"]
    node_names: np.ndarray = _emb_data["node_names"]
    nq = float(np.linalg.norm(query_vec).clip(min=1e-8))

    if req.entity_type:
        candidates = [i for i, name in enumerate(node_names) if req.entity_type in str(name)]
        if not candidates:
            return {"matches": [], "note": f"No nodes matching entity_type '{req.entity_type}'"}
        candidate_idx = np.array(candidates)
        candidate_vecs = embeddings[candidate_idx]
    else:
        candidate_idx = np.arange(len(embeddings))
        candidate_vecs = embeddings

    nc = np.linalg.norm(candidate_vecs, axis=1).clip(min=1e-8)
    sims = (candidate_vecs @ query_vec.T).squeeze() / (nc * nq)

    top_k = min(req.k, len(sims))
    top_indices = np.argpartition(-sims, top_k)[:top_k]
    top_indices = top_indices[np.argsort(-sims[top_indices])]

    matches = [
        {
            "node_id": str(node_names[int(candidate_idx[i]) if req.entity_type else int(i)]),
            "similarity": round(float(sims[i]), 4),
        }
        for i in top_indices
    ]

    return {"matches": matches}


# Legacy alias: POST /twin/embedding/search (old path)
@app.post("/twin/embedding/search")
async def _search_vector_alias(request: dict[str, Any]) -> dict[str, Any]:
    """Legacy alias for POST /twin/search-vector."""
    vector = request.get("vector")
    k = int(request.get("k", 10))
    entity_type = str(request.get("entity_type", ""))
    if not vector or not isinstance(vector, list):
        raise HTTPException(400, "Request must contain 'vector' (list of floats)")
    req = VectorSearchRequest(vector=vector, k=k, entity_type=entity_type)
    return await search_by_vector(req)


# Legacy aliases: /twin/embedding/similar, /twin/embedding/anomalies, /twin/embedding/vector
# These MUST be declared before the catch-all /twin/embedding/{node_id:path} below.


@app.get("/twin/embedding/similar/{node_id}")
async def _find_similar_alias(
    node_id: str,
    k: int = Query(default=10, ge=1, le=100),
    entity_type: str = Query(default=""),
) -> dict[str, Any]:
    """Legacy alias for GET /twin/similar/{node_id}."""
    return await find_similar(node_id, k=k, entity_type=entity_type)


@app.get("/twin/embedding/anomalies/{entity_type}")
async def _find_anomalies_alias(
    entity_type: str,
    top: int = Query(default=20, ge=1, le=500),
    source_db: str = Query(default=""),
) -> dict[str, Any]:
    """Legacy alias for GET /twin/anomalies/{entity_type}."""
    return await find_anomalies(entity_type, top=top, source_db=source_db)


@app.get("/twin/embedding/vector/{node_id}")
async def _get_vector_alias(node_id: str) -> dict[str, Any]:
    """Legacy alias for GET /twin/embedding/{node_id}."""
    return await get_embedding(node_id)


# Catch-all MUST come last — it will match any /twin/embedding/<anything> not already handled.
@app.get("/twin/embedding/{node_id:path}")
async def get_embedding(node_id: str) -> dict[str, Any]:
    """Get the raw 64-dim embedding vector for a node.

    Accepts both V4 (GHO-BRADESCO/ID_CD_INTERNACAO_123) and V3 (ID_CD_INTERNACAO_123) IDs.
    Use URL-encoding for slashes if your HTTP client does not support path parameters with slashes.
    """
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _ensure_embeddings)

    node_to_idx: dict[str, int] = _emb_data["node_to_idx"]
    if node_id not in node_to_idx:
        raise HTTPException(404, f"Node '{node_id}' not found in embedding space")

    idx = node_to_idx[node_id]
    vec = _emb_data["embeddings"][idx]
    return {
        "node_id": node_id,
        "dim": int(len(vec)),
        "vector": [round(float(v), 6) for v in vec],
    }


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "connected_databases": list(_twins.keys()),
        "active_db": _active_db,
        "embeddings_loaded": bool(_emb_data.get("loaded")),
        "catalog_loaded": bool(_catalogs),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Digital Twin API server")
    parser.add_argument("--db", type=str, help="Auto-connect to this DuckDB path on startup")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-profile", action="store_true", help="Skip column profiling")
    parser.add_argument("--no-fks", action="store_true", help="Skip FK discovery (use catalog instead)")
    parser.add_argument(
        "--catalog",
        type=str,
        default="data/ai_friendly_catalog.json",
        help="Path to ai_friendly_catalog.json (default: data/ai_friendly_catalog.json)",
    )
    parser.add_argument(
        "--graph",
        type=str,
        default="data/jcube_graph.parquet",
        help="Graph parquet for building node vocabulary",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="data/weights/node_emb_epoch_2.pt",
        help="Embedding weights file (V4 64-dim .pt)",
    )
    args = parser.parse_args()

    # Update module-level defaults so lazy loaders pick them up
    global _default_catalog_path, _default_graph_path, _default_weights_path
    _default_catalog_path = args.catalog
    _default_graph_path = args.graph
    _default_weights_path = args.weights

    if args.db:
        use_catalog = os.path.exists(args.catalog)
        print(f"Building digital twin for {args.db} (catalog={'yes' if use_catalog else 'no'})...")
        twin = DigitalTwin(args.db)
        snap = twin.build(
            profile_columns=not args.no_profile,
            discover_fks=(not args.no_fks and not use_catalog),
        )
        serialized = snapshot_to_dict(snap)

        if use_catalog:
            try:
                catalog = _load_catalog(args.catalog)
                for tname, tdata in serialized.get("tables", {}).items():
                    cat_entry = catalog["tables"].get(tname)
                    if cat_entry:
                        tdata["domain_group"] = cat_entry.get("category", "General")
                serialized["domain_groups"] = _catalog_to_domain_groups(catalog)
                serialized["foreign_keys"] = _catalog_fks(catalog)
                serialized["catalog_path"] = args.catalog
                print(f"Catalog overlaid: {len(catalog['tables'])} tables, "
                      f"{len(serialized['foreign_keys'])} FKs")
            except Exception as exc:
                print(f"WARNING: Catalog overlay failed: {exc}")

        _twins[args.db] = twin
        _snapshots[args.db] = serialized
        global _active_db
        _active_db = args.db
        print(
            f"Twin ready: {snap.total_tables} tables, {snap.total_rows:,} rows "
            f"(built in {snap.build_duration_s}s)"
        )

    # Embeddings are lazy — only pre-load if weights exist and startup is acceptable
    if os.path.exists(args.weights) and os.path.exists(args.graph):
        print(f"Pre-loading embeddings from {args.weights} ...")
        _load_embeddings(graph_path=args.graph, weights_path=args.weights)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
