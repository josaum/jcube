"""Graph-relational context loader for JEPA training.

Given an entity (e.g., INTERNACAO #1106), resolves ALL related entities
across bridge tables and loads their events as unified temporal context.

An admission's embedding should reflect not just its own events, but
the patient's history, related procedures, billing, audit findings, etc.

This is the "all connected tables at [0,t)" architecture.

Requires: duckdb, pyarrow
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

try:
    import duckdb
except ImportError:
    duckdb = None  # type: ignore

from .materializer import ENTITY_HIERARCHY, TIMESTAMP_PRIORITY, _to_epoch
from .lora_encoder import RowSerializer

# Core entity types for graph resolution (top of hierarchy)
CORE_ENTITIES = [
    "ID_CD_INTERNACAO",
    "ID_CD_PACIENTE",
    "ID_CD_FATURA",
    "ID_CD_HOSPITAL",
    "ID_CD_ORCAMENTO",
    "ID_CD_RELATORIO",
    "ID_CD_AUDITORIA",
    "ID_CD_EVOLUCAO",
]

_TS_TYPES = {"TIMESTAMP", "TIMESTAMPTZ", "TIMESTAMP WITH TIME ZONE", "DATE", "DATETIME"}


@dataclass
class GraphEvent:
    """A single event with its entity graph context."""
    epoch: float
    text: str
    entity_type: str  # which entity column this came from
    table: str


@dataclass
class EntityContext:
    """Full graph context for a single entity."""
    entity_type: str
    entity_id: str
    events: list[GraphEvent] = field(default_factory=list)
    related_ids: dict[str, set[str]] = field(default_factory=dict)
    n_tables: int = 0


class EntityGraph:
    """Discovers entity relationships and loads cross-table context."""

    def __init__(self, db_path: str) -> None:
        if duckdb is None:
            raise ImportError("duckdb required")
        self.db_path = db_path
        self.con = duckdb.connect(db_path, read_only=True)
        self.serializer = RowSerializer()

        # Discovered structures
        self._bridges: dict[tuple[str, str], list[str]] = {}  # (typeA, typeB) → [table,...]
        self._table_meta: dict[str, dict] = {}  # table → {entity_cols, ts_col, all_cols}
        self._discovered = False

    def discover(self) -> None:
        """Scan DB for entity relationships and temporal tables."""
        t0 = time.time()
        tables = self.con.execute("SHOW TABLES").fetchall()

        for (tname,) in tables:
            cols = self.con.execute(f'DESCRIBE "{tname}"').fetchall()
            col_names = [c[0] for c in cols]
            col_types = {c[0]: c[1] for c in cols}

            # Find entity columns
            entity_cols = [c for c in col_names if c in CORE_ENTITIES]

            # Find best timestamp column
            ts_col = None
            for tp in TIMESTAMP_PRIORITY:
                if tp in col_names and col_types.get(tp, "").upper().split("(")[0].strip() in _TS_TYPES:
                    ts_col = tp
                    break
            if ts_col is None:
                for c in col_names:
                    if col_types.get(c, "").upper().split("(")[0].strip() in _TS_TYPES:
                        ts_col = c
                        break

            if not entity_cols or not ts_col:
                continue

            # Store table metadata
            skip = set(entity_cols) | {ts_col, "source_db"}
            ctx_cols = [c for c in col_names if c not in skip]
            self._table_meta[tname] = {
                "entity_cols": entity_cols,
                "ts_col": ts_col,
                "ctx_cols": ctx_cols,
            }

            # Record FK bridges (tables with 2+ entity types)
            if len(entity_cols) >= 2:
                for i in range(len(entity_cols)):
                    for j in range(i + 1, len(entity_cols)):
                        key = tuple(sorted([entity_cols[i], entity_cols[j]]))
                        self._bridges.setdefault(key, []).append(tname)

        self._discovered = True
        n_bridges = sum(len(v) for v in self._bridges.values())
        print(f"  Graph discovered: {len(self._table_meta)} temporal tables, "
              f"{len(self._bridges)} relationship types, {n_bridges} bridge tables "
              f"({time.time() - t0:.1f}s)")

    def resolve_related(
        self,
        entity_type: str,
        entity_id: str,
        max_bridges: int = 3,
        max_related: int = 50,
    ) -> dict[str, set[str]]:
        """Resolve all related entity IDs across types via bridge tables.

        Returns: {entity_type: {entity_id, ...}}
        """
        related: dict[str, set[str]] = {entity_type: {entity_id}}

        for other_type in CORE_ENTITIES:
            if other_type == entity_type:
                continue

            key = tuple(sorted([entity_type, other_type]))
            bridge_tables = self._bridges.get(key, [])
            if not bridge_tables:
                continue

            ids: set[str] = set()
            for bt in bridge_tables[:max_bridges]:
                try:
                    rows = self.con.execute(
                        f'SELECT DISTINCT CAST("{other_type}" AS VARCHAR) '
                        f'FROM "{bt}" '
                        f'WHERE CAST("{entity_type}" AS VARCHAR) = ? '
                        f'AND "{other_type}" IS NOT NULL '
                        f'LIMIT {max_related}',
                        [entity_id],
                    ).fetchall()
                    ids.update(str(r[0]) for r in rows)
                except Exception:
                    continue

            if ids:
                related[other_type] = ids

        return related

    def load_context(
        self,
        entity_type: str,
        entity_id: str,
        max_events: int = 500,
        max_events_per_type: int = 100,
    ) -> EntityContext:
        """Load full graph context for an entity.

        Resolves relationships, then loads events from all related entities
        across all tables, sorted by timestamp.
        """
        if not self._discovered:
            self.discover()

        related = self.resolve_related(entity_type, entity_id)
        ctx = EntityContext(
            entity_type=entity_type,
            entity_id=entity_id,
            related_ids=related,
        )

        tables_used = set()

        # For each related entity type, load events from matching tables
        for etype, eids in related.items():
            id_list = ", ".join(f"'{eid}'" for eid in eids)
            events_for_type: list[GraphEvent] = []

            for tname, meta in self._table_meta.items():
                if etype not in meta["entity_cols"]:
                    continue

                ts_col = meta["ts_col"]
                ctx_cols = meta["ctx_cols"]

                try:
                    arrow = self.con.execute(
                        f'SELECT * FROM "{tname}" '
                        f'WHERE CAST("{etype}" AS VARCHAR) IN ({id_list}) '
                        f'AND "{ts_col}" IS NOT NULL '
                        f'ORDER BY "{ts_col}" '
                        f'LIMIT {max_events_per_type}'
                    ).fetch_arrow_table()
                except Exception:
                    continue

                if len(arrow) == 0:
                    continue

                tables_used.add(tname)
                rows_py = arrow.to_pydict()
                ts_values = rows_py[ts_col]

                for i in range(len(ts_values)):
                    ts_val = ts_values[i]
                    if ts_val is None:
                        continue
                    epoch = _to_epoch(ts_val)
                    if epoch is None:
                        continue

                    col_vals = {c: rows_py[c][i] for c in ctx_cols if c in rows_py}
                    # Tag text with entity type for the model to learn
                    text = f"<{etype.replace('ID_CD_', '')}> " + self.serializer.serialize_event(
                        tname, ts_val, col_vals
                    )
                    events_for_type.append(GraphEvent(
                        epoch=epoch,
                        text=text,
                        entity_type=etype,
                        table=tname,
                    ))

            ctx.events.extend(events_for_type)

        # Sort by timestamp, trim to budget
        ctx.events.sort(key=lambda e: e.epoch)
        if len(ctx.events) > max_events:
            ctx.events = ctx.events[-max_events:]  # keep most recent
        ctx.n_tables = len(tables_used)

        return ctx

    def load_all_contexts(
        self,
        entity_type: str,
        limit_entities: int = 50,
        max_events: int = 500,
    ) -> list[EntityContext]:
        """Load graph contexts for the richest entities of a given type.

        Returns list of EntityContext sorted by event count (richest first).
        """
        if not self._discovered:
            self.discover()

        # Find richest entities
        tables_for_type = [
            tname for tname, meta in self._table_meta.items()
            if entity_type in meta["entity_cols"]
        ]
        if not tables_for_type:
            return []

        count_parts = []
        for tname in tables_for_type:
            meta = self._table_meta[tname]
            count_parts.append(
                f'SELECT CAST("{entity_type}" AS VARCHAR) AS eid, COUNT(*) AS cnt '
                f'FROM "{tname}" '
                f'WHERE "{entity_type}" IS NOT NULL AND "{meta["ts_col"]}" IS NOT NULL '
                f'GROUP BY "{entity_type}"'
            )

        union = " UNION ALL ".join(count_parts)
        top_sql = (
            f"SELECT eid, SUM(cnt) AS total FROM ({union}) "
            f"GROUP BY eid ORDER BY total DESC LIMIT {limit_entities}"
        )
        rows = self.con.execute(top_sql).fetchall()
        target_ids = [str(r[0]) for r in rows]

        print(f"  Loading graph contexts for {len(target_ids)} {entity_type} entities...")
        t0 = time.time()

        contexts = []
        for i, eid in enumerate(target_ids):
            ctx = self.load_context(entity_type, eid, max_events=max_events)
            contexts.append(ctx)
            if (i + 1) % 10 == 0:
                n_evts = sum(len(c.events) for c in contexts)
                print(f"    {i + 1}/{len(target_ids)}: {n_evts:,} events total")

        elapsed = time.time() - t0
        total_events = sum(len(c.events) for c in contexts)
        total_relations = sum(len(c.related_ids) for c in contexts)
        print(f"  Loaded {total_events:,} events across {total_relations} relationships "
              f"({elapsed:.1f}s)")

        return contexts

    def close(self) -> None:
        self.con.close()
