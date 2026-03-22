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
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .digital_twin import DigitalTwin, snapshot_to_dict
from .materializer import Materializer, result_to_dict

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

_twins: dict[str, DigitalTwin] = {}  # db_path → twin
_snapshots: dict[str, dict[str, Any]] = {}  # db_path → serialized snapshot
_materializers: dict[str, Materializer] = {}  # db_path → materializer
_active_db: str | None = None


@asynccontextmanager
async def _lifespan(app: FastAPI):  # type: ignore[type-arg]
    yield
    # Cleanup on shutdown
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
    "schema, column profiles, FK relationships, domain grouping.",
    version="0.1.0",
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
    db_path: str = Field(..., description="Path to a DuckDB database file (local or remote)")
    read_only: bool = Field(True, description="Open in read-only mode")
    profile_columns: bool = Field(True, description="Profile every column (slower but richer)")
    discover_fks: bool = Field(True, description="Discover foreign key relationships")
    max_fk_checks: int = Field(2000, description="Max FK overlap checks")


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


# ---------------------------------------------------------------------------
# Helpers
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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/twin/connect")
async def connect(req: ConnectRequest) -> dict[str, Any]:
    """Connect to a DuckDB database and build its digital twin.

    This is the main entry point. It opens the database, introspects every
    table, profiles columns, discovers FK relationships, and returns a
    summary. The full twin is cached and available via other endpoints.
    """
    global _active_db

    # Close existing twin for this path if any
    if req.db_path in _twins:
        _twins[req.db_path].close()

    twin = DigitalTwin(req.db_path, read_only=req.read_only)

    # Build in a thread to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    snapshot = await loop.run_in_executor(
        None,
        lambda: twin.build(
            profile_columns=req.profile_columns,
            discover_fks=req.discover_fks,
            max_fk_checks=req.max_fk_checks,
        ),
    )

    serialized = snapshot_to_dict(snapshot)
    _twins[req.db_path] = twin
    _snapshots[req.db_path] = serialized
    _active_db = req.db_path

    # Return summary (not full snapshot — use /twin/snapshot for that)
    return {
        "status": "connected",
        "db_path": req.db_path,
        "db_size_mb": snapshot.db_size_mb,
        "total_tables": snapshot.total_tables,
        "total_rows": snapshot.total_rows,
        "total_columns": snapshot.total_columns,
        "source_databases": snapshot.source_databases,
        "domain_groups": {
            k: {"table_count": len(v.tables), "total_rows": v.total_rows, "description": v.description}
            for k, v in snapshot.domain_groups.items()
        },
        "foreign_keys_found": len(snapshot.foreign_keys),
        "fingerprint": snapshot.fingerprint,
        "build_duration_s": snapshot.build_duration_s,
    }


@app.get("/twin/snapshot")
async def get_snapshot(db_path: str | None = None) -> dict[str, Any]:
    """Return the full digital twin snapshot (all tables, columns, FKs)."""
    return _get_snapshot(db_path)


@app.get("/twin/tables")
async def list_tables(
    db_path: str | None = None,
    domain: str | None = Query(None, description="Filter by domain group"),
    min_rows: int = Query(0, description="Minimum row count"),
    sort_by: str = Query("name", description="Sort by: name, rows, columns"),
) -> list[dict[str, Any]]:
    """List all tables with summary stats, optionally filtered by domain."""
    snap = _get_snapshot(db_path)
    tables = list(snap["tables"].values())

    if domain:
        tables = [t for t in tables if t["domain_group"] == domain]
    if min_rows > 0:
        tables = [t for t in tables if t["row_count"] >= min_rows]

    key_map = {"name": "name", "rows": "row_count", "columns": "column_count"}
    sort_key = key_map.get(sort_by, "name")
    tables.sort(key=lambda t: t[sort_key], reverse=(sort_by in ("rows", "columns")))

    # Return lightweight list (no column details)
    return [
        {
            "name": t["name"],
            "row_count": t["row_count"],
            "column_count": t["column_count"],
            "domain_group": t["domain_group"],
            "size_category": t["size_category"],
            "has_timestamps": t["has_timestamps"],
            "source_databases": t["source_databases"],
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
    """Get sample rows from a table (returned as Arrow-backed dicts)."""
    twin = _get_twin(db_path)
    try:
        rows = twin.table_sample(table_name, limit=limit)
    except Exception as e:
        raise HTTPException(400, str(e))
    return {"table": table_name, "count": len(rows), "rows": rows}


@app.get("/twin/domains")
async def list_domains(db_path: str | None = None) -> dict[str, Any]:
    """List all domain groups with their tables and stats."""
    snap = _get_snapshot(db_path)
    return snap["domain_groups"]


@app.get("/twin/foreign-keys")
async def list_foreign_keys(
    db_path: str | None = None,
    min_confidence: float = Query(0.0, ge=0, le=1),
    table: str | None = Query(None, description="Filter by table name (from or to)"),
) -> list[dict[str, Any]]:
    """List discovered foreign key relationships."""
    snap = _get_snapshot(db_path)
    fks = snap["foreign_keys"]

    if min_confidence > 0:
        fks = [fk for fk in fks if fk["confidence"] >= min_confidence]
    if table:
        fks = [fk for fk in fks if table in (fk["from_table"], fk["to_table"])]

    return fks


@app.get("/twin/sources")
async def list_sources(db_path: str | None = None) -> dict[str, Any]:
    """List all source databases found in the data."""
    snap = _get_snapshot(db_path)
    return {
        "source_databases": snap["source_databases"],
        "total": len(snap["source_databases"]),
    }


@app.post("/twin/query")
async def run_query(req: QueryRequest) -> dict[str, Any]:
    """Execute arbitrary read-only SQL and return results as JSON."""
    twin = _get_twin(req.db_path)
    try:
        loop = asyncio.get_event_loop()
        arrow = await loop.run_in_executor(None, lambda: twin.query_arrow(req.sql))
        # Apply limit
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

    Useful for visualization with tools like D3, Cytoscape, etc.
    """
    snap = _get_snapshot(db_path)

    nodes = [
        {
            "id": name,
            "row_count": t["row_count"],
            "domain": t["domain_group"],
            "size": t["size_category"],
            "columns": t["column_count"],
        }
        for name, t in snap["tables"].items()
    ]

    edges = [
        {
            "source": fk["from_table"],
            "target": fk["to_table"],
            "column": fk["from_column"],
            "confidence": fk["confidence"],
        }
        for fk in snap["foreign_keys"]
        if fk["confidence"] >= min_confidence
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
                "row_count": tdata["row_count"],
                "domain": tdata["domain_group"],
            })
        for col in tdata["columns"]:
            if q_lower in col["name"].lower():
                column_hits.append({
                    "table": tname,
                    "column": col["name"],
                    "dtype": col["dtype"],
                    "distinct_count": col["distinct_count"],
                })

    return {
        "query": q,
        "table_matches": len(table_hits),
        "column_matches": len(column_hits),
        "tables": table_hits[:50],
        "columns": column_hits[:50],
    }


# ---------------------------------------------------------------------------
# Temporal / EventJEPA endpoints
# ---------------------------------------------------------------------------


@app.get("/twin/entity-types")
async def list_entity_types(db_path: str | None = None) -> dict[str, Any]:
    """Discover entity types in the database and how many tables/rows each spans.

    This scans for tables that have both an entity ID column (ID_CD_INTERNACAO,
    ID_CD_PACIENTE, etc.) and a timestamp column, then groups them by entity type.
    """
    loop = asyncio.get_event_loop()
    mat = await loop.run_in_executor(None, lambda: _get_materializer(db_path))
    summary = mat.scan_entity_types()
    return {
        "entity_types": {
            etype: {
                "table_count": len(info["tables"]),
                "total_rows": info["total_rows"],
                "entity_column": info["entity_column"],
                "tables": info["tables"][:20],  # cap for response size
            }
            for etype, info in summary.items()
        },
        "total_types": len(summary),
    }


@app.post("/twin/materialize")
async def materialize(req: MaterializeRequest) -> dict[str, Any]:
    """Materialize temporal event sequences for an entity type.

    Groups rows by entity ID across all matching tables, orders by timestamp,
    encodes context columns into embeddings, and optionally runs EventJEPA
    to produce temporal representations, pattern detection, and predictions.

    This is the core temporal endpoint — it turns flat tables into entity lifecycles.
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
    """Get the full temporal lifecycle for a single entity.

    Returns its event sequence across all tables, optionally with
    EventJEPA representation, patterns, and predictions.
    """
    loop = asyncio.get_event_loop()
    mat = await loop.run_in_executor(None, lambda: _get_materializer(db_path))

    # Materialize just this one entity (we fetch all then filter — efficient
    # because we limit to 1 entity)
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

    # Filter to just this entity
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

    # For single entity, also include the raw event timestamps
    seq = timeline.sequence
    serialized["event_timestamps"] = [
        {"index": i, "epoch": seq.timestamps[i]}
        for i in range(len(seq.timestamps))
    ]

    return serialized


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "connected_databases": list(_twins.keys()),
        "active_db": _active_db,
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
    parser.add_argument("--no-fks", action="store_true", help="Skip FK discovery")
    args = parser.parse_args()

    if args.db:
        # Pre-build twin before starting server
        print(f"Building digital twin for {args.db} ...")
        twin = DigitalTwin(args.db)
        snap = twin.build(
            profile_columns=not args.no_profile,
            discover_fks=not args.no_fks,
        )
        _twins[args.db] = twin
        _snapshots[args.db] = snapshot_to_dict(snap)
        global _active_db
        _active_db = args.db
        print(
            f"Twin ready: {snap.total_tables} tables, {snap.total_rows:,} rows, "
            f"{len(snap.foreign_keys)} FK candidates "
            f"(built in {snap.build_duration_s}s)"
        )

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
