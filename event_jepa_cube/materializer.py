"""EventSequence materializer for DuckDB databases.

Scans a database for tables with entity-ID + timestamp columns, encodes
context columns into fixed-dimension embeddings, and produces
EventSequences that can be fed into EventJEPA.

The core idea: every row in the database is an *event* that happened to
an *entity* at a *point in time*.  The materializer turns flat tables into
temporal lifecycles.

Requires: duckdb, pyarrow
"""

from __future__ import annotations

import hashlib
import math
import struct
import time
from dataclasses import dataclass, field
from typing import Any

try:
    import duckdb

    _DUCKDB = True
except ImportError:
    _DUCKDB = False

try:
    import pyarrow as pa  # noqa: F401

    _ARROW = True
except ImportError:
    _ARROW = False

from .event_jepa import EventJEPA
from .sequence import EventSequence

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Entity ID columns ranked by semantic importance.
# The materializer groups events by the first matching column it finds.
ENTITY_HIERARCHY: list[str] = [
    "ID_CD_INTERNACAO",  # hospital admission — richest lifecycle
    "ID_CD_PACIENTE",  # patient
    "ID_CD_FATURA",  # invoice
    "ID_CD_PESSOA",  # person (CRM)
    "ID_CD_ORCAMENTO",  # budget
    "ID_CD_HOSPITAL",  # hospital
    "ID_CD_RELATORIO",  # report
    "ID_CD_AUDITORIA",  # audit
    "ID_CD_EVOLUCAO",  # clinical evolution
    "ID_CD_TICKET",  # support ticket
]

# Preferred timestamp column (DH_CADASTRO = creation timestamp)
TIMESTAMP_PRIORITY: list[str] = [
    "DH_CADASTRO",
    "DH_ATUALIZACAO",
    "created_at",
    "updated_at",
    "DH_VISITA",
    "DH_PROCEDIMENTO",
    "DH_ADMISSAO_HOSP",
    "DH_INICIO",
    "DH_ALTA",
    "DT_CRIACAO",
]

_TS_TYPES = {"TIMESTAMP", "TIMESTAMPTZ", "TIMESTAMP WITH TIME ZONE", "DATE", "DATETIME"}
_NUMERIC_TYPES = {"BIGINT", "INTEGER", "SMALLINT", "TINYINT", "FLOAT", "DOUBLE", "DECIMAL", "HUGEINT"}


# ---------------------------------------------------------------------------
# Column encoder
# ---------------------------------------------------------------------------


class ColumnEncoder:
    """Encodes heterogeneous database columns into a fixed-dimension float vector.

    Strategy per type:
    - **Numeric** (BIGINT, FLOAT, etc.): log-scale normalization → 1 dimension
    - **Categorical** (VARCHAR with low cardinality): hash-based one-hot into
      ``cat_dims`` dimensions (locality-sensitive hashing)
    - **Text** (VARCHAR with high cardinality): character trigram hash →
      ``text_dims`` dimensions
    - **Timestamp** (used as context, not the event timestamp): cyclical
      encoding (hour-of-day, day-of-week, month) → 6 dimensions
    - **Boolean**: 0.0 / 1.0 → 1 dimension
    - **NULL**: 0.0 for the slot

    The output dimension is deterministic given the column schema.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        cat_dims: int = 8,
        text_dims: int = 16,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.cat_dims = cat_dims
        self.text_dims = text_dims

    def dims_for_column(self, dtype: str, is_timestamp: bool = False) -> int:
        """Return how many dimensions a column of this type contributes."""
        base = dtype.upper().split("(")[0].strip()
        if is_timestamp:
            return 6  # sin/cos for hour, dow, month
        if base == "BOOLEAN":
            return 1
        if base in _NUMERIC_TYPES:
            return 1
        if base == "VARCHAR":
            return self.text_dims  # hash embedding
        return 1  # fallback

    def encode_value(self, value: Any, dtype: str, is_timestamp: bool = False) -> list[float]:
        """Encode a single value into a float vector."""
        base = dtype.upper().split("(")[0].strip()

        if value is None:
            dim = self.dims_for_column(dtype, is_timestamp)
            return [0.0] * dim

        if is_timestamp:
            return self._encode_timestamp(value)

        if base == "BOOLEAN":
            return [1.0 if value else 0.0]

        if base in _NUMERIC_TYPES:
            return [self._encode_numeric(value)]

        if base == "VARCHAR":
            return self._encode_text(str(value))

        # Fallback: hash to single float
        return [self._hash_float(str(value))]

    def _encode_numeric(self, value: Any) -> float:
        """Log-scale normalization for numeric values."""
        try:
            v = float(value)
            if v == 0:
                return 0.0
            sign = 1.0 if v > 0 else -1.0
            return sign * math.log1p(abs(v)) / 20.0  # normalize roughly to [-1, 1]
        except (ValueError, TypeError):
            return 0.0

    def _encode_timestamp(self, value: Any) -> list[float]:
        """Cyclical time encoding: hour, day-of-week, month → 6 dims (sin/cos pairs)."""
        try:
            import datetime

            if isinstance(value, (int, float)):
                dt = datetime.datetime.fromtimestamp(value)
            elif isinstance(value, str):
                dt = datetime.datetime.fromisoformat(value)
            elif isinstance(value, datetime.date) and not isinstance(value, datetime.datetime):
                dt = datetime.datetime(value.year, value.month, value.day)
            else:
                dt = value

            hour_frac = dt.hour / 24.0
            dow_frac = dt.weekday() / 7.0
            month_frac = (dt.month - 1) / 12.0

            return [
                math.sin(2 * math.pi * hour_frac),
                math.cos(2 * math.pi * hour_frac),
                math.sin(2 * math.pi * dow_frac),
                math.cos(2 * math.pi * dow_frac),
                math.sin(2 * math.pi * month_frac),
                math.cos(2 * math.pi * month_frac),
            ]
        except Exception:
            return [0.0] * 6

    def _encode_text(self, text: str) -> list[float]:
        """Character trigram hashing into fixed-dim vector."""
        vec = [0.0] * self.text_dims
        if not text:
            return vec
        text = text.lower().strip()
        # Character trigrams
        for i in range(max(1, len(text) - 2)):
            trigram = text[i : i + 3]
            h = int(hashlib.md5(trigram.encode(), usedforsecurity=False).hexdigest()[:8], 16)
            idx = h % self.text_dims
            sign = 1.0 if (h >> 31) & 1 else -1.0
            vec[idx] += sign
        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def _hash_float(self, text: str) -> float:
        """Hash any string to a float in [-1, 1]."""
        h = hashlib.md5(text.encode(), usedforsecurity=False).digest()[:4]
        return (struct.unpack("<I", h)[0] / (2**32)) * 2 - 1

    def encode_row(
        self,
        row: dict[str, Any],
        columns: list[tuple[str, str, bool]],
    ) -> list[float]:
        """Encode an entire row into a fixed-dimension vector.

        Args:
            row: Column name → value mapping.
            columns: List of (col_name, dtype, is_context_timestamp) tuples
                describing which columns to encode and their types.

        Returns:
            Fixed-dimension float vector.
        """
        parts: list[float] = []
        for col_name, dtype, is_ts in columns:
            parts.extend(self.encode_value(row.get(col_name), dtype, is_ts))
        return parts

    def project(self, vec: list[float], target_dim: int) -> list[float]:
        """Project a vector to target_dim using deterministic hash projection.

        Uses a sparse random projection (count-sketch style) that preserves
        approximate distances.  Much faster than a full matrix multiply.
        """
        if len(vec) <= target_dim:
            # Pad with zeros
            return vec + [0.0] * (target_dim - len(vec))

        out = [0.0] * target_dim
        for i, v in enumerate(vec):
            if v == 0.0:
                continue
            # Deterministic hash: which output bucket and sign
            h = ((i + 1) * 2654435761) & 0xFFFFFFFF  # Knuth multiplicative hash
            idx = h % target_dim
            sign = 1.0 if (h >> 16) & 1 else -1.0
            out[idx] += sign * v
        return out


# ---------------------------------------------------------------------------
# Table scanner
# ---------------------------------------------------------------------------


@dataclass
class TemporalTable:
    """A table that can produce event sequences."""

    name: str
    entity_column: str  # the entity ID column to group by
    entity_type: str  # e.g. "INTERNACAO", "PACIENTE"
    timestamp_column: str  # the primary timestamp column
    context_columns: list[tuple[str, str, bool]] = field(default_factory=list)
    # (col_name, dtype, is_context_timestamp)
    row_count: int = 0
    embedding_dim: int = 0  # computed from context columns


@dataclass
class EntityTimeline:
    """A materialized timeline for one entity across one or more tables."""

    entity_id: str
    entity_type: str
    sequence: EventSequence
    source_tables: list[str] = field(default_factory=list)
    event_count: int = 0
    time_span_days: float = 0.0


@dataclass
class MaterializationResult:
    """Result of materializing event sequences from a database."""

    entity_type: str
    tables_scanned: int
    entities_found: int
    total_events: int
    embedding_dim: int
    timelines: dict[str, EntityTimeline] = field(default_factory=dict)
    duration_s: float = 0.0
    # EventJEPA results (if processed)
    representations: dict[str, list[float]] = field(default_factory=dict)
    patterns: dict[str, list[int]] = field(default_factory=dict)
    predictions: dict[str, list[list[float]]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Materializer
# ---------------------------------------------------------------------------


class Materializer:
    """Materializes EventSequences from a DuckDB database.

    Workflow:
    1. ``scan()`` — discover which tables have entity ID + timestamp columns
    2. ``materialize()`` — pull events for a given entity type, encode rows,
       build EventSequences grouped by entity ID
    3. ``process()`` — run EventJEPA on the materialized sequences

    Example::

        mat = Materializer("data/my.db")
        mat.connect()

        # What temporal tables exist?
        tables = mat.scan()

        # Materialize all internacao (admission) timelines
        result = mat.materialize("INTERNACAO", limit_entities=100)

        # Run EventJEPA
        result = mat.process(result, num_prediction_steps=3)

        mat.close()
    """

    def __init__(
        self,
        db_path: str,
        *,
        read_only: bool = True,
        embedding_dim: int = 64,
        text_dims: int = 16,
    ) -> None:
        if not _DUCKDB:
            raise ImportError("duckdb is required")
        self._db_path = db_path
        self._read_only = read_only
        self._conn: duckdb.DuckDBPyConnection | None = None
        self._encoder = ColumnEncoder(embedding_dim=embedding_dim, text_dims=text_dims)
        self._temporal_tables: list[TemporalTable] | None = None

    def connect(self) -> None:
        if self._conn is None:
            self._conn = duckdb.connect(self._db_path, read_only=self._read_only)

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> Materializer:
        self.connect()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _con(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            self.connect()
        return self._conn  # type: ignore[return-value]

    # -- scan ---------------------------------------------------------------

    def scan(self) -> list[TemporalTable]:
        """Scan the database and return all tables that can produce event sequences.

        For each table, picks the best entity ID column (by ENTITY_HIERARCHY rank)
        and the best timestamp column (by TIMESTAMP_PRIORITY rank).
        """
        con = self._con()

        # Get all columns
        arrow = con.execute(
            "SELECT table_name, column_name, data_type "
            "FROM information_schema.columns "
            "WHERE table_schema = 'main' "
            "ORDER BY table_name, ordinal_position"
        ).fetch_arrow_table()

        table_cols: dict[str, list[tuple[str, str]]] = {}
        for i in range(len(arrow)):
            tname = arrow.column("table_name")[i].as_py()
            cname = arrow.column("column_name")[i].as_py()
            dtype = arrow.column("data_type")[i].as_py()
            table_cols.setdefault(tname, []).append((cname, dtype))

        # Build entity_col_upper → rank lookup
        entity_rank = {col.upper(): rank for rank, col in enumerate(ENTITY_HIERARCHY)}
        ts_rank = {col.upper(): rank for rank, col in enumerate(TIMESTAMP_PRIORITY)}

        results: list[TemporalTable] = []

        for tname, cols in table_cols.items():
            # Find best entity ID column
            best_entity: tuple[str, int] | None = None  # (col_name, rank)
            best_ts: tuple[str, int] | None = None

            col_info: list[tuple[str, str]] = []  # (name, dtype)
            for cname, dtype in cols:
                col_info.append((cname, dtype))
                upper = cname.upper()

                # Check entity
                if upper in entity_rank:
                    rank = entity_rank[upper]
                    if best_entity is None or rank < best_entity[1]:
                        best_entity = (cname, rank)

                # Check timestamp
                base_type = dtype.upper().split("(")[0].strip()
                if upper in ts_rank and base_type in _TS_TYPES:
                    rank = ts_rank[upper]
                    if best_ts is None or rank < best_ts[1]:
                        best_ts = (cname, rank)
                elif base_type in _TS_TYPES and best_ts is None:
                    # Fallback: any timestamp column
                    best_ts = (cname, 999)

            if best_entity is None or best_ts is None:
                continue

            entity_col = best_entity[0]
            ts_col = best_ts[0]

            # Determine entity type from column name
            entity_type = entity_col.upper().replace("ID_CD_", "").replace("ID_", "")

            # Context columns = everything except entity_id, timestamp, source_db
            skip = {entity_col.upper(), ts_col.upper(), "SOURCE_DB"}
            context: list[tuple[str, str, bool]] = []
            for cname, dtype in col_info:
                if cname.upper() in skip:
                    continue
                base = dtype.upper().split("(")[0].strip()
                is_ctx_ts = base in _TS_TYPES
                context.append((cname, dtype, is_ctx_ts))

            # Compute embedding dim from context columns
            emb_dim = sum(self._encoder.dims_for_column(dt, is_ts) for _, dt, is_ts in context)

            # Row count
            try:
                rc = con.execute(f'SELECT COUNT(*) FROM "{tname}"').fetchone()[0]  # type: ignore[index]
            except Exception:
                rc = 0

            results.append(
                TemporalTable(
                    name=tname,
                    entity_column=entity_col,
                    entity_type=entity_type,
                    timestamp_column=ts_col,
                    context_columns=context,
                    row_count=rc,
                    embedding_dim=emb_dim,
                )
            )

        self._temporal_tables = results
        return results

    def scan_entity_types(self) -> dict[str, dict[str, Any]]:
        """Return a summary of entity types found, with table counts and total rows."""
        if self._temporal_tables is None:
            self.scan()
        assert self._temporal_tables is not None

        summary: dict[str, dict[str, Any]] = {}
        for tt in self._temporal_tables:
            et = tt.entity_type
            if et not in summary:
                summary[et] = {"tables": [], "total_rows": 0, "entity_column": tt.entity_column}
            summary[et]["tables"].append(tt.name)
            summary[et]["total_rows"] += tt.row_count

        return dict(sorted(summary.items(), key=lambda x: -x[1]["total_rows"]))

    # -- materialize --------------------------------------------------------

    def materialize(
        self,
        entity_type: str,
        *,
        limit_entities: int | None = None,
        limit_events_per_entity: int | None = 5000,
        table_filter: list[str] | None = None,
        source_db_filter: str | None = None,
        entity_ids: list[str] | None = None,
    ) -> MaterializationResult:
        """Materialize EventSequences for a given entity type.

        Uses a 2-phase approach for efficiency:
          Phase 1: Discover the richest entity IDs (most events across tables)
          Phase 2: Targeted queries with WHERE entity_id IN (...) per table

        Args:
            entity_type: e.g. "INTERNACAO", "PACIENTE", "FATURA"
            limit_entities: Max number of entities to materialize (None = all)
            limit_events_per_entity: Max events per entity (oldest dropped)
            table_filter: Only use these tables (None = all matching)
            source_db_filter: Only use rows from this source_db
            entity_ids: Explicit list of entity IDs to materialize (skips Phase 1)
        """
        t0 = time.time()
        con = self._con()

        if self._temporal_tables is None:
            self.scan()
        assert self._temporal_tables is not None

        # Find tables for this entity type
        tables = [
            tt for tt in self._temporal_tables
            if tt.entity_type == entity_type
        ]
        if table_filter:
            tables = [tt for tt in tables if tt.name in table_filter]

        if not tables:
            return MaterializationResult(
                entity_type=entity_type,
                tables_scanned=0,
                entities_found=0,
                total_events=0,
                embedding_dim=0,
                duration_s=time.time() - t0,
            )

        # Common embedding dim — cap at encoder's embedding_dim to keep
        # EventJEPA fast. Tables with more columns hash-project down.
        raw_max = max(tt.embedding_dim for tt in tables)
        max_emb_dim = min(raw_max, self._encoder.embedding_dim)
        entity_col_name = tables[0].entity_column

        # ---- Phase 1: Discover target entity IDs ---------------------------
        if entity_ids is not None:
            target_ids = set(entity_ids)
        elif limit_entities is not None:
            # Find entities with the most events by counting across tables.
            # Use a UNION ALL of lightweight count queries (no data transfer).
            count_parts: list[str] = []
            for tt in tables:
                safe_tbl = f'"{tt.name}"'
                safe_eid = f'"{tt.entity_column}"'
                where = f"WHERE {safe_eid} IS NOT NULL"
                if source_db_filter:
                    where += f" AND source_db = '{source_db_filter}'"
                count_parts.append(
                    f"SELECT CAST({safe_eid} AS VARCHAR) AS eid, COUNT(*) AS cnt "
                    f"FROM {safe_tbl} {where} GROUP BY {safe_eid}"
                )

            if count_parts:
                union_sql = " UNION ALL ".join(count_parts)
                top_sql = (
                    f"SELECT eid, SUM(cnt) AS total FROM ({union_sql}) "
                    f"GROUP BY eid ORDER BY total DESC LIMIT {limit_entities}"
                )
                try:
                    rows = con.execute(top_sql).fetchall()
                    target_ids = {str(r[0]) for r in rows}
                except Exception:
                    # Fallback: just pick from the first table
                    tt0 = tables[0]
                    rows = con.execute(
                        f'SELECT DISTINCT CAST("{tt0.entity_column}" AS VARCHAR) '
                        f'FROM "{tt0.name}" LIMIT {limit_entities}'
                    ).fetchall()
                    target_ids = {str(r[0]) for r in rows}
            else:
                target_ids = set()
        else:
            target_ids = None  # no filter — load all

        # ---- Phase 2: Targeted queries per table ----------------------------
        entity_events: dict[str, list[tuple[float, list[float], str]]] = {}

        for tt in tables:
            safe_tbl = f'"{tt.name}"'
            safe_eid = f'"{tt.entity_column}"'
            safe_ts = f'"{tt.timestamp_column}"'

            where_parts = [f"{safe_eid} IS NOT NULL", f"{safe_ts} IS NOT NULL"]
            if source_db_filter:
                where_parts.append(f"source_db = '{source_db_filter}'")

            # Push entity filter into SQL for massive speedup
            if target_ids is not None and len(target_ids) <= 500:
                id_list = ", ".join(f"'{eid}'" for eid in target_ids)
                where_parts.append(f"CAST({safe_eid} AS VARCHAR) IN ({id_list})")

            where = "WHERE " + " AND ".join(where_parts)

            try:
                arrow_table = con.execute(
                    f"SELECT * FROM {safe_tbl} {where} ORDER BY {safe_ts}"
                ).fetch_arrow_table()
            except Exception:
                continue

            if len(arrow_table) == 0:
                continue

            col_names = arrow_table.schema.names

            # Build column encoding plan
            encode_plan: list[tuple[str, str, bool]] = []
            for cname, dtype, is_ts in tt.context_columns:
                if cname in col_names:
                    encode_plan.append((cname, dtype, is_ts))

            # Process rows via Arrow columnar access
            rows_py = arrow_table.to_pydict()
            eid_col = rows_py[tt.entity_column]
            ts_col = rows_py[tt.timestamp_column]
            n_rows = len(eid_col)

            for i in range(n_rows):
                eid_val = eid_col[i]
                ts_val = ts_col[i]
                if eid_val is None or ts_val is None:
                    continue

                eid_str = str(eid_val)

                # Skip if not in target set (for large target sets not pushed to SQL)
                if target_ids is not None and eid_str not in target_ids:
                    continue

                epoch = _to_epoch(ts_val)
                if epoch is None:
                    continue

                # Encode context columns
                emb: list[float] = []
                for cname, dtype, is_ts in encode_plan:
                    val = rows_py[cname][i]
                    emb.extend(self._encoder.encode_value(val, dtype, is_ts))

                # Project to fixed dimension
                emb = self._encoder.project(emb, max_emb_dim)

                if eid_str not in entity_events:
                    entity_events[eid_str] = []
                entity_events[eid_str].append((epoch, emb, tt.name))

        # Build EventSequences
        timelines: dict[str, EntityTimeline] = {}
        total_events = 0

        for eid, events in entity_events.items():
            # Sort by timestamp
            events.sort(key=lambda e: e[0])

            # Apply per-entity event limit (keep most recent)
            if limit_events_per_entity and len(events) > limit_events_per_entity:
                events = events[-limit_events_per_entity:]

            timestamps = [e[0] for e in events]
            embeddings = [e[1] for e in events]
            source_tables = list({e[2] for e in events})

            # Time span
            if len(timestamps) >= 2:
                span_days = (timestamps[-1] - timestamps[0]) / 86400.0
            else:
                span_days = 0.0

            seq = EventSequence(
                embeddings=embeddings,
                timestamps=timestamps,
                modality="db_row",
            )

            timelines[eid] = EntityTimeline(
                entity_id=eid,
                entity_type=entity_type,
                sequence=seq,
                source_tables=source_tables,
                event_count=len(events),
                time_span_days=span_days,
            )
            total_events += len(events)

        return MaterializationResult(
            entity_type=entity_type,
            tables_scanned=len(tables),
            entities_found=len(timelines),
            total_events=total_events,
            embedding_dim=max_emb_dim if tables else 0,
            timelines=timelines,
            duration_s=round(time.time() - t0, 2),
        )

    # -- process with EventJEPA ---------------------------------------------

    def process(
        self,
        result: MaterializationResult,
        *,
        num_levels: int = 2,
        temporal_resolution: str = "adaptive",
        num_prediction_steps: int = 3,
    ) -> MaterializationResult:
        """Run EventJEPA on materialized timelines.

        Populates ``result.representations``, ``result.patterns``, and
        ``result.predictions``.
        """
        jepa = EventJEPA(
            embedding_dim=result.embedding_dim,
            num_levels=num_levels,
            temporal_resolution=temporal_resolution,
        )

        for eid, timeline in result.timelines.items():
            seq = timeline.sequence
            if len(seq.embeddings) < 2:
                continue

            rep = jepa.process(seq)
            result.representations[eid] = rep
            result.patterns[eid] = jepa.detect_patterns(rep)
            result.predictions[eid] = jepa.predict_next(seq, num_prediction_steps)

        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_epoch(value: Any) -> float | None:
    """Convert a timestamp-like value to Unix epoch seconds."""
    import datetime

    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, datetime.datetime):
        return value.timestamp()
    if isinstance(value, datetime.date):
        return datetime.datetime(value.year, value.month, value.day).timestamp()
    if isinstance(value, str):
        try:
            return datetime.datetime.fromisoformat(value).timestamp()
        except ValueError:
            return None
    return None


def result_to_dict(result: MaterializationResult) -> dict[str, Any]:
    """Serialize a MaterializationResult to a JSON-safe dict."""
    timelines_summary: list[dict[str, Any]] = []
    for eid, tl in sorted(result.timelines.items(), key=lambda x: -x[1].event_count):
        entry: dict[str, Any] = {
            "entity_id": eid,
            "entity_type": tl.entity_type,
            "event_count": tl.event_count,
            "time_span_days": round(tl.time_span_days, 1),
            "source_tables": tl.source_tables,
        }
        if eid in result.representations:
            rep = result.representations[eid]
            entry["representation_dim"] = len(rep)
            entry["representation_norm"] = round(math.sqrt(sum(v * v for v in rep)), 4)
        if eid in result.patterns:
            entry["salient_dimensions"] = result.patterns[eid]
        if eid in result.predictions:
            entry["prediction_steps"] = len(result.predictions[eid])
        timelines_summary.append(entry)

    return {
        "entity_type": result.entity_type,
        "tables_scanned": result.tables_scanned,
        "entities_found": result.entities_found,
        "total_events": result.total_events,
        "embedding_dim": result.embedding_dim,
        "duration_s": result.duration_s,
        "timelines": timelines_summary,
    }
