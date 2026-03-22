"""Digital Twin builder for DuckDB databases.

Connects to a DuckDB database via Apache Arrow, introspects every table,
profiles columns, discovers foreign-key relationships, groups tables by
domain, and exposes the result as a rich metadata graph — effectively a
"digital twin" of the database.

Requires: duckdb, pyarrow
"""

from __future__ import annotations

import hashlib
import math
import re
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


def _require() -> None:
    if not _DUCKDB:
        raise ImportError("duckdb is required: pip install duckdb")
    if not _ARROW:
        raise ImportError("pyarrow is required: pip install pyarrow")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    nullable: bool = True
    total_count: int = 0
    null_count: int = 0
    distinct_count: int = 0
    min_value: Any = None
    max_value: Any = None
    mean_value: float | None = None
    sample_values: list[Any] = field(default_factory=list)
    is_id_column: bool = False
    is_timestamp: bool = False
    is_foreign_key_candidate: bool = False


@dataclass
class TableProfile:
    name: str
    row_count: int = 0
    column_count: int = 0
    columns: list[ColumnProfile] = field(default_factory=list)
    domain_group: str = ""
    source_databases: list[str] = field(default_factory=list)
    size_category: str = ""  # small/medium/large/xlarge
    has_timestamps: bool = False
    primary_key_candidates: list[str] = field(default_factory=list)


@dataclass
class ForeignKeyCandidate:
    from_table: str
    from_column: str
    to_table: str
    to_column: str
    confidence: float = 0.0  # 0-1 based on value overlap
    sample_overlap: int = 0


@dataclass
class DomainGroup:
    name: str
    prefix: str
    tables: list[str] = field(default_factory=list)
    total_rows: int = 0
    description: str = ""


@dataclass
class TwinSnapshot:
    """Complete digital twin of a database."""

    db_path: str
    db_size_mb: float
    total_tables: int
    total_rows: int
    total_columns: int
    tables: dict[str, TableProfile] = field(default_factory=dict)
    domain_groups: dict[str, DomainGroup] = field(default_factory=dict)
    foreign_keys: list[ForeignKeyCandidate] = field(default_factory=list)
    source_databases: list[str] = field(default_factory=list)
    built_at: float = 0.0
    build_duration_s: float = 0.0
    fingerprint: str = ""


# ---------------------------------------------------------------------------
# Domain grouping rules
# ---------------------------------------------------------------------------

_DOMAIN_DESCRIPTIONS: dict[str, str] = {
    "capta": "Hospital capture/audit management",
    "cfg": "System configuration tables",
    "fatura": "Billing and invoicing",
    "crm": "Patient/contact relationship management",
    "auditoria": "Clinical audit records",
    "fin": "Financial management",
    "formulario": "Form definitions and submissions",
    "relatorio": "Reports and report archives",
    "farm": "Pharmacy management",
    "login": "User authentication and sessions",
    "produtos": "Product catalog",
    "tea": "Therapy/treatment management",
    "tickets": "Support ticket system",
    "log": "System logs",
    "rh": "Human resources",
    "pendencias": "Pending items and follow-ups",
    "pericia": "Medical expertise/review",
    "sistemas": "System configuration",
    "orcamento": "Budget and cost estimation",
    "beneficiarios": "Beneficiary/member records",
    "campanhas": "Campaign management",
}


def _classify_domain(table_name: str) -> tuple[str, str]:
    """Return (group_name, prefix) for a table name."""
    parts = table_name.lower().replace("agg_", "").split("_")
    # Skip the 'tb' prefix
    start = 1 if len(parts) > 1 and parts[0] == "tb" else 0
    if start < len(parts):
        key = parts[start]
        desc = _DOMAIN_DESCRIPTIONS.get(key, f"{key} domain tables")
        prefix = f"agg_tb_{key}"
        return key, prefix
    return "other", table_name


def _size_category(row_count: int) -> str:
    if row_count < 100:
        return "tiny"
    if row_count < 10_000:
        return "small"
    if row_count < 100_000:
        return "medium"
    if row_count < 1_000_000:
        return "large"
    return "xlarge"


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

_ID_PATTERN = re.compile(r"^(ID_CD_|id_cd_|ID_|id_)", re.IGNORECASE)
_TS_TYPES = {"TIMESTAMP", "TIMESTAMPTZ", "TIMESTAMP WITH TIME ZONE", "DATE", "DATETIME"}
_NUMERIC_TYPES = {"BIGINT", "INTEGER", "SMALLINT", "TINYINT", "FLOAT", "DOUBLE", "DECIMAL", "HUGEINT"}


class DigitalTwin:
    """Builds and holds a digital twin of a DuckDB database."""

    def __init__(self, db_path: str, *, read_only: bool = True) -> None:
        _require()
        self._db_path = db_path
        self._read_only = read_only
        self._conn: duckdb.DuckDBPyConnection | None = None
        self._snapshot: TwinSnapshot | None = None

    # -- connection ---------------------------------------------------------

    def connect(self) -> None:
        if self._conn is not None:
            return
        self._conn = duckdb.connect(self._db_path, read_only=self._read_only)

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> DigitalTwin:
        self.connect()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _con(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            self.connect()
        return self._conn  # type: ignore[return-value]

    # -- introspection via Arrow -------------------------------------------

    def _list_tables(self) -> list[str]:
        arrow = self._con().execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' ORDER BY table_name"
        ).fetch_arrow_table()
        return arrow.column("table_name").to_pylist()

    def _describe_table(self, table: str) -> list[dict[str, Any]]:
        arrow = self._con().execute(
            "SELECT column_name, data_type, is_nullable "
            f'FROM information_schema.columns WHERE table_name = \'{table}\' '
            "ORDER BY ordinal_position"
        ).fetch_arrow_table()
        return [
            {
                "name": arrow.column("column_name")[i].as_py(),
                "dtype": arrow.column("data_type")[i].as_py(),
                "nullable": arrow.column("is_nullable")[i].as_py() == "YES",
            }
            for i in range(len(arrow))
        ]

    def _row_count(self, table: str) -> int:
        return self._con().execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]  # type: ignore[index]

    def _profile_column(self, table: str, col_name: str, dtype: str, total_rows: int) -> dict[str, Any]:
        """Profile a single column using Arrow for data transfer."""
        con = self._con()
        safe_col = f'"{col_name}"'
        safe_tbl = f'"{table}"'

        stats: dict[str, Any] = {}

        # Null count
        r = con.execute(f"SELECT COUNT(*) FILTER (WHERE {safe_col} IS NULL) FROM {safe_tbl}").fetchone()
        stats["null_count"] = r[0] if r else 0  # type: ignore[index]

        # Distinct count (approximate for large tables)
        if total_rows > 500_000:
            r = con.execute(f"SELECT approx_count_distinct({safe_col}) FROM {safe_tbl}").fetchone()
        else:
            r = con.execute(f"SELECT COUNT(DISTINCT {safe_col}) FROM {safe_tbl}").fetchone()
        stats["distinct_count"] = r[0] if r else 0  # type: ignore[index]

        # Min/max for orderable types
        base_type = dtype.upper().split("(")[0].strip()
        if base_type in _NUMERIC_TYPES or base_type in _TS_TYPES or base_type == "VARCHAR":
            try:
                r = con.execute(
                    f"SELECT MIN({safe_col}), MAX({safe_col}) FROM {safe_tbl}"
                ).fetchone()
                if r:
                    stats["min_value"] = _serialize(r[0])
                    stats["max_value"] = _serialize(r[1])
            except Exception:
                pass

        # Mean for numeric
        if base_type in _NUMERIC_TYPES:
            try:
                r = con.execute(f"SELECT AVG({safe_col}::DOUBLE) FROM {safe_tbl}").fetchone()
                if r and r[0] is not None:
                    stats["mean_value"] = float(r[0])
            except Exception:
                pass

        # Sample values (via Arrow)
        try:
            arrow = con.execute(
                f"SELECT DISTINCT {safe_col} FROM {safe_tbl} "
                f"WHERE {safe_col} IS NOT NULL LIMIT 5"
            ).fetch_arrow_table()
            stats["sample_values"] = [_serialize(v) for v in arrow.column(0).to_pylist()]
        except Exception:
            stats["sample_values"] = []

        return stats

    # -- source databases ---------------------------------------------------

    def _discover_source_dbs(self, tables: list[str]) -> list[str]:
        """Find all distinct source_db values across tables that have the column."""
        con = self._con()
        sources: set[str] = set()
        for t in tables:
            try:
                arrow = con.execute(
                    f'SELECT DISTINCT source_db FROM "{t}" WHERE source_db IS NOT NULL LIMIT 100'
                ).fetch_arrow_table()
                sources.update(arrow.column(0).to_pylist())
            except Exception:
                continue
        return sorted(sources)

    # -- FK discovery -------------------------------------------------------

    def _discover_foreign_keys(
        self,
        tables: dict[str, TableProfile],
        *,
        max_checks: int = 2000,
        sample_size: int = 200,
    ) -> list[ForeignKeyCandidate]:
        """Discover FK relationships by matching column names and checking value overlap."""
        con = self._con()

        # Build index: column_name → [(table, dtype, distinct_count)]
        col_index: dict[str, list[tuple[str, str, int]]] = {}
        for tname, tprof in tables.items():
            for cp in tprof.columns:
                if cp.is_id_column and cp.distinct_count > 1:
                    key = cp.name.upper()
                    col_index.setdefault(key, []).append((tname, cp.dtype, cp.distinct_count))

        candidates: list[ForeignKeyCandidate] = []
        checks = 0

        for col_key, locations in col_index.items():
            if len(locations) < 2:
                continue
            # Check all pairs
            for i in range(len(locations)):
                for j in range(i + 1, len(locations)):
                    if checks >= max_checks:
                        break
                    t1, _, d1 = locations[i]
                    t2, _, d2 = locations[j]
                    if t1 == t2:
                        continue

                    # The table with fewer distinct values is likely the "parent"
                    if d1 <= d2:
                        parent_t, child_t = t1, t2
                    else:
                        parent_t, child_t = t2, t1

                    col_name = next(
                        cp.name for cp in tables[t1].columns if cp.name.upper() == col_key
                    )
                    col_name2 = next(
                        cp.name for cp in tables[t2].columns if cp.name.upper() == col_key
                    )

                    # Check value overlap via sample
                    try:
                        r = con.execute(
                            f'SELECT COUNT(*) FROM ('
                            f'SELECT DISTINCT "{col_name}" AS v FROM "{parent_t}" '
                            f'WHERE "{col_name}" IS NOT NULL LIMIT {sample_size}'
                            f') a INNER JOIN ('
                            f'SELECT DISTINCT "{col_name2}" AS v FROM "{child_t}" '
                            f'WHERE "{col_name2}" IS NOT NULL LIMIT {sample_size}'
                            f') b ON a.v = b.v'
                        ).fetchone()
                        overlap = r[0] if r else 0  # type: ignore[index]
                    except Exception:
                        overlap = 0
                    checks += 1

                    if overlap > 0:
                        confidence = min(overlap / sample_size, 1.0)
                        candidates.append(
                            ForeignKeyCandidate(
                                from_table=child_t,
                                from_column=col_name2 if child_t == t2 else col_name,
                                to_table=parent_t,
                                to_column=col_name if parent_t == t1 else col_name2,
                                confidence=confidence,
                                sample_overlap=overlap,
                            )
                        )

        # Sort by confidence descending
        candidates.sort(key=lambda c: c.confidence, reverse=True)
        return candidates

    # -- full build ---------------------------------------------------------

    def build(
        self,
        *,
        profile_columns: bool = True,
        discover_fks: bool = True,
        max_fk_checks: int = 2000,
    ) -> TwinSnapshot:
        """Build the complete digital twin snapshot.

        This is the main entry point. It introspects every table, profiles
        columns (optionally), discovers foreign keys, groups by domain, and
        returns a ``TwinSnapshot``.
        """
        import os

        t0 = time.time()
        self.connect()

        db_size = os.path.getsize(self._db_path) / (1024 * 1024) if os.path.isfile(self._db_path) else 0.0
        table_names = self._list_tables()

        # Introspect tables
        tables: dict[str, TableProfile] = {}
        total_rows = 0
        total_cols = 0

        for tname in table_names:
            cols_info = self._describe_table(tname)
            row_count = self._row_count(tname)
            total_rows += row_count
            total_cols += len(cols_info)

            columns: list[ColumnProfile] = []
            pk_candidates: list[str] = []

            for ci in cols_info:
                cp = ColumnProfile(
                    name=ci["name"],
                    dtype=ci["dtype"],
                    nullable=ci["nullable"],
                    total_count=row_count,
                )

                base_type = ci["dtype"].upper().split("(")[0].strip()
                cp.is_id_column = bool(_ID_PATTERN.match(ci["name"]))
                cp.is_timestamp = base_type in _TS_TYPES

                if profile_columns and row_count > 0:
                    stats = self._profile_column(tname, ci["name"], ci["dtype"], row_count)
                    cp.null_count = stats.get("null_count", 0)
                    cp.distinct_count = stats.get("distinct_count", 0)
                    cp.min_value = stats.get("min_value")
                    cp.max_value = stats.get("max_value")
                    cp.mean_value = stats.get("mean_value")
                    cp.sample_values = stats.get("sample_values", [])

                    # FK candidate: ID column with moderate cardinality
                    if cp.is_id_column and 1 < cp.distinct_count < row_count * 0.9:
                        cp.is_foreign_key_candidate = True

                    # PK candidate: unique non-null
                    if cp.is_id_column and cp.distinct_count == row_count and cp.null_count == 0:
                        pk_candidates.append(ci["name"])

                columns.append(cp)

            group_name, prefix = _classify_domain(tname)

            tables[tname] = TableProfile(
                name=tname,
                row_count=row_count,
                column_count=len(cols_info),
                columns=columns,
                domain_group=group_name,
                size_category=_size_category(row_count),
                has_timestamps=any(c.is_timestamp for c in columns),
                primary_key_candidates=pk_candidates,
            )

        # Source databases
        source_dbs = self._discover_source_dbs(table_names)
        for tname, tp in tables.items():
            try:
                arrow = self._con().execute(
                    f'SELECT DISTINCT source_db FROM "{tname}" WHERE source_db IS NOT NULL LIMIT 100'
                ).fetch_arrow_table()
                tp.source_databases = arrow.column(0).to_pylist()
            except Exception:
                tp.source_databases = []

        # Domain groups
        domain_groups: dict[str, DomainGroup] = {}
        for tname, tp in tables.items():
            gname = tp.domain_group
            if gname not in domain_groups:
                _, prefix = _classify_domain(tname)
                domain_groups[gname] = DomainGroup(
                    name=gname,
                    prefix=prefix,
                    description=_DOMAIN_DESCRIPTIONS.get(gname, f"{gname} domain"),
                )
            domain_groups[gname].tables.append(tname)
            domain_groups[gname].total_rows += tp.row_count

        # FK discovery
        fks: list[ForeignKeyCandidate] = []
        if discover_fks:
            fks = self._discover_foreign_keys(tables, max_checks=max_fk_checks)

        duration = time.time() - t0

        # Fingerprint = hash of table names + row counts for change detection
        fp_data = "|".join(f"{t}:{tables[t].row_count}" for t in sorted(tables))
        fingerprint = hashlib.sha256(fp_data.encode()).hexdigest()[:16]

        self._snapshot = TwinSnapshot(
            db_path=self._db_path,
            db_size_mb=round(db_size, 2),
            total_tables=len(tables),
            total_rows=total_rows,
            total_columns=total_cols,
            tables=tables,
            domain_groups=domain_groups,
            foreign_keys=fks,
            source_databases=source_dbs,
            built_at=t0,
            build_duration_s=round(duration, 2),
            fingerprint=fingerprint,
        )
        return self._snapshot

    @property
    def snapshot(self) -> TwinSnapshot | None:
        return self._snapshot

    # -- query helpers (post-build) -----------------------------------------

    def query_arrow(self, sql: str) -> Any:
        """Execute arbitrary SQL and return an Arrow table."""
        return self._con().execute(sql).fetch_arrow_table()

    def table_sample(self, table: str, limit: int = 10) -> list[dict[str, Any]]:
        """Return sample rows from a table as list of dicts."""
        arrow = self._con().execute(f'SELECT * FROM "{table}" LIMIT {limit}').fetch_arrow_table()
        return arrow.to_pylist()

    def table_arrow(self, table: str, *, limit: int | None = None) -> Any:
        """Return a table as an Arrow Table (zero-copy from DuckDB)."""
        sql = f'SELECT * FROM "{table}"'
        if limit:
            sql += f" LIMIT {limit}"
        return self._con().execute(sql).fetch_arrow_table()


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _serialize(val: Any) -> Any:
    """Make a value JSON-safe."""
    if val is None:
        return None
    if isinstance(val, (int, float, str, bool)):
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return str(val)
        return val
    return str(val)


def snapshot_to_dict(snap: TwinSnapshot) -> dict[str, Any]:
    """Convert a TwinSnapshot to a JSON-serializable dict."""
    return {
        "db_path": snap.db_path,
        "db_size_mb": snap.db_size_mb,
        "total_tables": snap.total_tables,
        "total_rows": snap.total_rows,
        "total_columns": snap.total_columns,
        "source_databases": snap.source_databases,
        "fingerprint": snap.fingerprint,
        "built_at": snap.built_at,
        "build_duration_s": snap.build_duration_s,
        "domain_groups": {
            name: {
                "name": dg.name,
                "prefix": dg.prefix,
                "description": dg.description,
                "table_count": len(dg.tables),
                "total_rows": dg.total_rows,
                "tables": dg.tables,
            }
            for name, dg in snap.domain_groups.items()
        },
        "tables": {
            name: _table_to_dict(tp) for name, tp in snap.tables.items()
        },
        "foreign_keys": [
            {
                "from_table": fk.from_table,
                "from_column": fk.from_column,
                "to_table": fk.to_table,
                "to_column": fk.to_column,
                "confidence": round(fk.confidence, 3),
                "sample_overlap": fk.sample_overlap,
            }
            for fk in snap.foreign_keys
        ],
    }


def _table_to_dict(tp: TableProfile) -> dict[str, Any]:
    return {
        "name": tp.name,
        "row_count": tp.row_count,
        "column_count": tp.column_count,
        "domain_group": tp.domain_group,
        "size_category": tp.size_category,
        "has_timestamps": tp.has_timestamps,
        "source_databases": tp.source_databases,
        "primary_key_candidates": tp.primary_key_candidates,
        "columns": [
            {
                "name": c.name,
                "dtype": c.dtype,
                "nullable": c.nullable,
                "null_count": c.null_count,
                "null_pct": round(c.null_count / c.total_count * 100, 1) if c.total_count > 0 else 0,
                "distinct_count": c.distinct_count,
                "min_value": _serialize(c.min_value),
                "max_value": _serialize(c.max_value),
                "mean_value": round(c.mean_value, 4) if c.mean_value is not None else None,
                "sample_values": c.sample_values,
                "is_id_column": c.is_id_column,
                "is_timestamp": c.is_timestamp,
                "is_foreign_key_candidate": c.is_foreign_key_candidate,
            }
            for c in tp.columns
        ],
    }
