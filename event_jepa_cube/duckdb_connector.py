"""DuckDB multi-source data warehouse connector.

Connects to one or more databases (DuckDB, PostgreSQL, MySQL, SQLite,
Arrow Flight), builds an instant data warehouse via UNION ALL BY NAME,
and runs the EventJEPA + EmbeddingCube pipeline automatically.

Requires DuckDB. Install with: pip install event-jepa-cube[duckdb]
For Arrow performance: pip install event-jepa-cube[duckdb-arrow]
"""

from __future__ import annotations

import re
from typing import Any

try:
    import duckdb

    _DUCKDB_AVAILABLE = True
except ImportError:
    _DUCKDB_AVAILABLE = False

try:
    import pyarrow  # noqa: F401

    _ARROW_AVAILABLE = True
except ImportError:
    _ARROW_AVAILABLE = False

from .embedding_cube import EmbeddingCube
from .event_jepa import EventJEPA
from .sequence import Entity, EventSequence

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Default column name mappings
_DEFAULT_SEQ_COLUMNS: dict[str, str] = {
    "sequence_id": "sequence_id",
    "embedding": "embedding",
    "timestamp": "timestamp",
    "modality": "modality",
}

_DEFAULT_ENTITY_COLUMNS: dict[str, str] = {
    "entity_id": "entity_id",
    "modality": "modality",
    "embedding": "embedding",
    "category": "category",
}


def _require_duckdb() -> None:
    if not _DUCKDB_AVAILABLE:
        raise ImportError(
            "DuckDB is required for the DuckDB connector. Install with: pip install event-jepa-cube[duckdb]"
        )


def _validate_identifier(name: str) -> str:
    """Validate a SQL identifier and return it quoted.

    Raises ``ValueError`` if the name is not a safe identifier.
    """
    if not _IDENTIFIER_RE.match(name):
        raise ValueError(f"Invalid SQL identifier: {name!r}")
    return f'"{name}"'


class DuckDBConnector:
    """Connect to multiple databases, build a warehouse, and run pipelines.

    Example::

        with DuckDBConnector(embedding_dim=3) as conn:
            conn.attach("pg", "dbname=prod host=localhost", db_type="postgres")
            conn.attach("my", "host=localhost user=root database=logs", db_type="mysql")
            results = conn.run_from_sources(
                sources=[],  # already attached above
                tables=["event_sequences"],
            )

    Or the one-liner::

        with DuckDBConnector(embedding_dim=3) as conn:
            results = conn.run_from_sources(
                sources=[
                    {"name": "pg", "connection_string": "dbname=prod host=localhost", "db_type": "postgres"},
                ],
                tables=["event_sequences"],
            )
    """

    def __init__(
        self,
        database: str = ":memory:",
        embedding_dim: int = 768,
        num_levels: int = 1,
        temporal_resolution: str = "adaptive",
    ) -> None:
        _require_duckdb()
        self._conn: duckdb.DuckDBPyConnection | None = duckdb.connect(database)
        self._jepa = EventJEPA(
            embedding_dim=embedding_dim,
            num_levels=num_levels,
            temporal_resolution=temporal_resolution,
        )
        self._cube = EmbeddingCube()
        self._attached: list[str] = []

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the DuckDB connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> DuckDBConnector:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def _ensure_open(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            raise RuntimeError("DuckDBConnector is closed")
        return self._conn

    # ------------------------------------------------------------------
    # Multi-source attach
    # ------------------------------------------------------------------

    def attach(
        self,
        name: str,
        connection_string: str,
        db_type: str = "duckdb",
        read_only: bool = True,
    ) -> None:
        """Attach an external database to the workspace.

        Args:
            name: Alias for the attached database.
            connection_string: Connection string (file path for duckdb/sqlite,
                DSN for postgres/mysql).
            db_type: One of ``"duckdb"``, ``"postgres"``, ``"mysql"``, ``"sqlite"``.
            read_only: Open in read-only mode (default ``True``).

        DuckDB scanner extensions (postgres, mysql, sqlite) are installed and
        loaded automatically.
        """
        conn = self._ensure_open()
        safe_name = _validate_identifier(name)

        if db_type != "duckdb":
            conn.execute(f"INSTALL {db_type}")
            conn.execute(f"LOAD {db_type}")

        ro = ", READ_ONLY" if read_only else ""
        type_clause = f" (TYPE {db_type}{ro})" if db_type != "duckdb" else " (READ_ONLY)" if read_only else ""
        conn.execute(f"ATTACH '{connection_string}' AS {safe_name}{type_clause}")
        self._attached.append(name)

    def attach_flight(
        self,
        name: str,
        endpoint: str,
    ) -> None:
        """Attach an Arrow Flight SQL endpoint.

        Requires the ``airport`` community extension.
        """
        conn = self._ensure_open()
        safe_name = _validate_identifier(name)
        conn.execute("INSTALL airport FROM community")
        conn.execute("LOAD airport")
        conn.execute(f"ATTACH '{endpoint}' AS {safe_name} (TYPE airport)")
        self._attached.append(name)

    def detach(self, name: str) -> None:
        """Detach a previously attached database."""
        conn = self._ensure_open()
        safe_name = _validate_identifier(name)
        conn.execute(f"DETACH {safe_name}")
        if name in self._attached:
            self._attached.remove(name)

    @property
    def attached_databases(self) -> list[str]:
        """List currently attached database aliases (excluding the default)."""
        return list(self._attached)

    # ------------------------------------------------------------------
    # Instant data warehouse builder
    # ------------------------------------------------------------------

    def _find_tables_in_db(self, db_name: str) -> list[str]:
        """Return table names available in an attached database."""
        conn = self._ensure_open()
        _validate_identifier(db_name)
        rows = conn.execute(
            f"SELECT table_name FROM information_schema.tables WHERE table_catalog = '{db_name}'"
        ).fetchall()
        return [r[0] for r in rows]

    def build_warehouse(
        self,
        tables: list[str],
        *,
        warehouse_db: str = "warehouse",
        source_filter: dict[str, list[str]] | None = None,
    ) -> dict[str, int]:
        """Build a unified warehouse from attached databases using UNION ALL BY NAME.

        For each table name in *tables*, scans all attached databases for that
        table, unions them with ``UNION ALL BY NAME``, and adds a ``sourcedb``
        column identifying the origin database.  Results are materialized into
        local DuckDB tables under a warehouse schema.

        Args:
            tables: Table names to unify (e.g. ``["event_sequences", "entities"]``).
            warehouse_db: Schema name for the warehouse (default ``"warehouse"``).
            source_filter: Optional mapping of ``{table: [db_names]}`` to limit
                which sources are included per table.

        Returns:
            Dict mapping table name to total row count.
        """
        conn = self._ensure_open()
        safe_wh = _validate_identifier(warehouse_db)

        conn.execute(f"CREATE SCHEMA IF NOT EXISTS {safe_wh}")

        row_counts: dict[str, int] = {}

        for table in tables:
            safe_table = _validate_identifier(table)
            allowed_dbs = source_filter.get(table, self._attached) if source_filter else self._attached

            # Find which attached DBs actually have this table
            parts: list[str] = []
            for db_name in allowed_dbs:
                if db_name not in self._attached:
                    continue
                available = self._find_tables_in_db(db_name)
                if table in available:
                    safe_db = _validate_identifier(db_name)
                    parts.append(f"SELECT *, '{db_name}' AS sourcedb FROM {safe_db}.{safe_table}")

            if not parts:
                row_counts[table] = 0
                continue

            union_sql = " UNION ALL BY NAME ".join(parts)
            conn.execute(f"CREATE OR REPLACE TABLE {safe_wh}.{safe_table} AS ({union_sql})")

            count = conn.execute(f"SELECT COUNT(*) FROM {safe_wh}.{safe_table}").fetchone()
            row_counts[table] = count[0] if count else 0

        return row_counts

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_sequences(
        self,
        table: str = "event_sequences",
        *,
        query: str | None = None,
        column_map: dict[str, str] | None = None,
    ) -> dict[str, EventSequence]:
        """Load event sequences from a DuckDB table or custom query.

        Each row is one event.  Rows sharing the same ``sequence_id`` are
        grouped into a single :class:`EventSequence`.

        Args:
            table: Table name to query (ignored if *query* is provided).
            query: Raw SQL query to execute instead of the default
                ``SELECT * FROM <table>``.
            column_map: Mapping from logical names (``sequence_id``,
                ``embedding``, ``timestamp``, ``modality``) to actual column
                names in the table.

        Returns:
            Dict mapping sequence_id to :class:`EventSequence`.
        """
        conn = self._ensure_open()
        cmap = {**_DEFAULT_SEQ_COLUMNS, **(column_map or {})}

        if query is None:
            query = f"SELECT * FROM {_validate_identifier(table)}"

        result = conn.execute(query)
        col_names = [desc[0] for desc in result.description]

        # Build column index lookup
        col_idx: dict[str, int] = {}
        for logical, actual in cmap.items():
            if actual in col_names:
                col_idx[logical] = col_names.index(actual)

        if "sequence_id" not in col_idx or "embedding" not in col_idx or "timestamp" not in col_idx:
            raise ValueError(
                f"Required columns not found. Expected {cmap['sequence_id']}, "
                f"{cmap['embedding']}, {cmap['timestamp']} in: {col_names}"
            )

        rows = result.fetchall()

        # Group by sequence_id
        groups: dict[str, dict[str, Any]] = {}
        for row in rows:
            sid = str(row[col_idx["sequence_id"]])
            emb = [float(v) for v in row[col_idx["embedding"]]]
            ts = float(row[col_idx["timestamp"]])
            mod = str(row[col_idx["modality"]]) if "modality" in col_idx else "text"

            if sid not in groups:
                groups[sid] = {"embeddings": [], "timestamps": [], "modality": mod}
            groups[sid]["embeddings"].append(emb)
            groups[sid]["timestamps"].append(ts)

        return {
            sid: EventSequence(
                embeddings=data["embeddings"],
                timestamps=data["timestamps"],
                modality=data["modality"],
            )
            for sid, data in groups.items()
        }

    def load_entities(
        self,
        table: str = "entities",
        *,
        query: str | None = None,
        column_map: dict[str, str] | None = None,
    ) -> dict[str, Entity]:
        """Load entities from a DuckDB table or custom query.

        Each row is one modality embedding for an entity.  Rows sharing the
        same ``entity_id`` are merged into a single :class:`Entity`.

        Args:
            table: Table name to query (ignored if *query* is provided).
            query: Raw SQL query to execute.
            column_map: Mapping from logical names (``entity_id``,
                ``modality``, ``embedding``, ``category``) to actual column
                names.

        Returns:
            Dict mapping entity_id to :class:`Entity`.
        """
        conn = self._ensure_open()
        cmap = {**_DEFAULT_ENTITY_COLUMNS, **(column_map or {})}

        if query is None:
            query = f"SELECT * FROM {_validate_identifier(table)}"

        result = conn.execute(query)
        col_names = [desc[0] for desc in result.description]

        col_idx: dict[str, int] = {}
        for logical, actual in cmap.items():
            if actual in col_names:
                col_idx[logical] = col_names.index(actual)

        if "entity_id" not in col_idx or "modality" not in col_idx or "embedding" not in col_idx:
            raise ValueError(
                f"Required columns not found. Expected {cmap['entity_id']}, "
                f"{cmap['modality']}, {cmap['embedding']} in: {col_names}"
            )

        rows = result.fetchall()

        entities: dict[str, dict[str, Any]] = {}
        for row in rows:
            eid = str(row[col_idx["entity_id"]])
            mod = str(row[col_idx["modality"]])
            emb = [float(v) for v in row[col_idx["embedding"]]]
            has_cat = "category" in col_idx and row[col_idx["category"]] is not None
            cat = str(row[col_idx["category"]]) if has_cat else None

            if eid not in entities:
                entities[eid] = {"embeddings": {}, "hierarchy_info": {}}
            entities[eid]["embeddings"][mod] = emb
            if cat is not None:
                entities[eid]["hierarchy_info"]["category"] = cat

        return {
            eid: Entity(
                embeddings=data["embeddings"],
                hierarchy_info=data["hierarchy_info"],
                id=eid,
            )
            for eid, data in entities.items()
        }

    # ------------------------------------------------------------------
    # Write-back
    # ------------------------------------------------------------------

    def write_representations(
        self,
        results: dict[str, list[float]],
        table: str = "representations",
    ) -> None:
        """Write sequence representations to DuckDB."""
        conn = self._ensure_open()
        safe = _validate_identifier(table)
        conn.execute(f"CREATE TABLE IF NOT EXISTS {safe} (sequence_id VARCHAR, representation FLOAT[])")
        for sid, rep in results.items():
            conn.execute(f"INSERT INTO {safe} VALUES ($1, $2)", [sid, rep])

    def write_patterns(
        self,
        results: dict[str, list[int]],
        table: str = "patterns",
    ) -> None:
        """Write detected patterns (salient dimensions) to DuckDB."""
        conn = self._ensure_open()
        safe = _validate_identifier(table)
        conn.execute(f"CREATE TABLE IF NOT EXISTS {safe} (sequence_id VARCHAR, salient_dimensions INTEGER[])")
        for sid, dims in results.items():
            conn.execute(f"INSERT INTO {safe} VALUES ($1, $2)", [sid, dims])

    def write_predictions(
        self,
        results: dict[str, list[list[float]]],
        table: str = "predictions",
    ) -> None:
        """Write predictions to DuckDB."""
        conn = self._ensure_open()
        safe = _validate_identifier(table)
        conn.execute(f"CREATE TABLE IF NOT EXISTS {safe} (sequence_id VARCHAR, step INTEGER, prediction FLOAT[])")
        for sid, preds in results.items():
            for step, pred in enumerate(preds, start=1):
                conn.execute(f"INSERT INTO {safe} VALUES ($1, $2, $3)", [sid, step, pred])

    def write_relationships(
        self,
        relationships: dict[str, list[str]],
        table: str = "relationships",
    ) -> None:
        """Write entity relationships to DuckDB."""
        conn = self._ensure_open()
        safe = _validate_identifier(table)
        conn.execute(f"CREATE TABLE IF NOT EXISTS {safe} (entity_id VARCHAR, related_entity_id VARCHAR)")
        for eid, related in relationships.items():
            for rid in related:
                conn.execute(f"INSERT INTO {safe} VALUES ($1, $2)", [eid, rid])

    # ------------------------------------------------------------------
    # Single-sequence processing (used by triggers)
    # ------------------------------------------------------------------

    def process_sequence(
        self, sequence_id: str, sequence: EventSequence, num_prediction_steps: int = 3
    ) -> dict[str, Any]:
        """Run the pipeline for a single sequence and return results.

        Returns:
            Dict with keys ``sequence_id``, ``representation``, ``patterns``,
            ``predictions``.
        """
        rep = self._jepa.process(sequence)
        return {
            "sequence_id": sequence_id,
            "representation": rep,
            "patterns": self._jepa.detect_patterns(rep),
            "predictions": self._jepa.predict_next(sequence, num_prediction_steps),
        }

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def run_pipeline(
        self,
        *,
        sequences_table: str = "event_sequences",
        entities_table: str | None = None,
        sequence_query: str | None = None,
        entity_query: str | None = None,
        column_map_sequences: dict[str, str] | None = None,
        column_map_entities: dict[str, str] | None = None,
        num_prediction_steps: int = 3,
        relationship_threshold: float = 0.5,
        write_results: bool = True,
    ) -> dict[str, Any]:
        """Run the full pipeline: load, process, detect, predict, and optionally discover relationships.

        Args:
            sequences_table: Table to load event sequences from.
            entities_table: Optional table to load entities from for
                relationship discovery.
            sequence_query: Raw SQL override for loading sequences.
            entity_query: Raw SQL override for loading entities.
            column_map_sequences: Column name mapping for sequences.
            column_map_entities: Column name mapping for entities.
            num_prediction_steps: Number of prediction steps.
            relationship_threshold: Cosine similarity threshold for
                entity relationships.
            write_results: Write results back to DuckDB tables.

        Returns:
            Dict with keys ``representations``, ``patterns``,
            ``predictions``, and optionally ``relationships``.
        """
        # Load sequences
        sequences = self.load_sequences(sequences_table, query=sequence_query, column_map=column_map_sequences)

        representations: dict[str, list[float]] = {}
        patterns: dict[str, list[int]] = {}
        predictions: dict[str, list[list[float]]] = {}

        for sid, seq in sequences.items():
            rep = self._jepa.process(seq)
            representations[sid] = rep
            patterns[sid] = self._jepa.detect_patterns(rep)
            predictions[sid] = self._jepa.predict_next(seq, num_prediction_steps)

        result: dict[str, Any] = {
            "representations": representations,
            "patterns": patterns,
            "predictions": predictions,
        }

        # Entity relationships (optional)
        if entities_table is not None or entity_query is not None:
            entities = self.load_entities(
                entities_table or "entities",
                query=entity_query,
                column_map=column_map_entities,
            )
            self._cube = EmbeddingCube()
            for entity in entities.values():
                self._cube.add_entity(entity)
            relationships = self._cube.discover_relationships(list(entities.keys()), threshold=relationship_threshold)
            result["relationships"] = relationships
        else:
            result["relationships"] = None

        # Write back
        if write_results:
            self.write_representations(representations)
            self.write_patterns(patterns)
            self.write_predictions(predictions)
            if result["relationships"] is not None:
                self.write_relationships(result["relationships"])

        return result

    # ------------------------------------------------------------------
    # End-to-end convenience
    # ------------------------------------------------------------------

    def run_from_sources(
        self,
        sources: list[dict[str, str]],
        tables: list[str],
        *,
        num_prediction_steps: int = 3,
        relationship_threshold: float = 0.5,
        sequences_table: str = "event_sequences",
        entities_table: str | None = None,
        write_results: bool = True,
    ) -> dict[str, Any]:
        """Full end-to-end: attach sources, build warehouse, run pipeline.

        Args:
            sources: List of dicts with keys ``name``, ``connection_string``,
                and optionally ``db_type`` (default ``"duckdb"``).
            tables: Table names to union across sources via
                ``UNION ALL BY NAME``.
            num_prediction_steps: Number of prediction steps.
            relationship_threshold: Relationship discovery threshold.
            sequences_table: Table name for event sequences in the warehouse.
            entities_table: Optional table name for entities in the warehouse.
            write_results: Write results back to DuckDB tables.

        Returns:
            Dict with ``warehouse_row_counts`` and pipeline result keys
            (``representations``, ``patterns``, ``predictions``,
            ``relationships``).
        """
        # Attach sources
        for src in sources:
            self.attach(
                name=src["name"],
                connection_string=src["connection_string"],
                db_type=src.get("db_type", "duckdb"),
                read_only=src.get("read_only", "true").lower() != "false"
                if isinstance(src.get("read_only"), str)
                else True,
            )

        # Build warehouse
        row_counts = self.build_warehouse(tables, warehouse_db="warehouse")

        # Run pipeline on warehouse tables
        wh_seq = f"warehouse.{sequences_table}"
        wh_ent = f"warehouse.{entities_table}" if entities_table else None

        pipeline_result = self.run_pipeline(
            sequence_query=f"SELECT * FROM {wh_seq}" if sequences_table in tables else None,
            sequences_table=sequences_table,
            entity_query=f"SELECT * FROM {wh_ent}" if entities_table and entities_table in tables else None,
            entities_table=entities_table,
            num_prediction_steps=num_prediction_steps,
            relationship_threshold=relationship_threshold,
            write_results=write_results,
        )

        pipeline_result["warehouse_row_counts"] = row_counts
        return pipeline_result
