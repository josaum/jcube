"""Cascading forecast pipeline.

Chains multiple TriggerEngine levels so that predictions at one level
automatically become input events for the next.  Patient-level predictions
cascade into department-level forecasts, which cascade into financial
projections — all reactive, all updating in real-time.

Example::

    with DuckDBConnector(embedding_dim=3) as conn:
        cascade = ForecastCascade(conn, source_table="event_sequences")
        cascade.add_level(CascadeLevel(name="patient", num_prediction_steps=3))
        cascade.add_level(CascadeLevel(name="department", num_prediction_steps=3))
        cascade.add_level(CascadeLevel(name="financial", num_prediction_steps=2))

        # Single call runs the whole chain reactively
        handle = cascade.watch_async(interval_seconds=5)
        # ... new events land in event_sequences ...
        handle.stop()

Requires DuckDB.  Install with: pip install event-jepa-cube[duckdb]
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .duckdb_connector import DuckDBConnector, _require_duckdb, _validate_identifier
from .triggers import AlertRule, StopHandle, TriggerEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CascadeLevel
# ---------------------------------------------------------------------------


@dataclass
class CascadeLevel:
    """Configuration for one level in a forecast cascade.

    Args:
        name: Unique identifier for this level (e.g. ``"patient"``,
            ``"department"``, ``"financial"``).  Must be a valid SQL
            identifier (alphanumeric + underscores).
        num_prediction_steps: How many steps ahead to predict at this level.
        rules: Alert rules evaluated after each poll at this level.
        view_query: Optional SQL that transforms the upstream predictions
            table into this level's input events.  The query receives the
            upstream predictions table name as ``{upstream_table}`` and must
            produce columns ``sequence_id``, ``embedding``, ``timestamp``,
            ``modality``.  If ``None``, upstream predictions pass through
            with a default column mapping.
    """

    name: str
    num_prediction_steps: int = 3
    rules: list[AlertRule] = field(default_factory=list)
    view_query: str | None = None


# ---------------------------------------------------------------------------
# ForecastCascade
# ---------------------------------------------------------------------------


class ForecastCascade:
    """Multi-level reactive forecast pipeline.

    Predictions at each level automatically flow into the next level as
    input events.  The entire cascade is driven by a single ``watch()``
    or ``watch_async()`` call.

    Internally, each level gets its own namespaced tables:

    - ``_cascade_{name}_predictions_live`` — live predictions for this level
    - ``_cascade_{name}_alerts`` — alerts fired at this level

    Between levels, a DuckDB VIEW reshapes the upstream predictions into
    the event-sequence format expected by the downstream TriggerEngine.

    Example::

        cascade = ForecastCascade(conn, source_table="event_sequences")
        cascade.add_level(CascadeLevel("patient", num_prediction_steps=3))
        cascade.add_level(CascadeLevel(
            "department",
            num_prediction_steps=3,
            view_query=(
                "SELECT department_id AS sequence_id, prediction AS embedding, "
                "updated_at AS timestamp, 'prediction' AS modality "
                "FROM {upstream_table} "
                "JOIN patient_dept ON {upstream_table}.sequence_id = patient_dept.patient_id"
            ),
        ))
        cascade.add_level(CascadeLevel("financial", num_prediction_steps=2))
        cascade.watch(interval_seconds=5)
    """

    def __init__(
        self,
        connector: DuckDBConnector,
        *,
        source_table: str = "event_sequences",
        source_column_map: dict[str, str] | None = None,
    ) -> None:
        _require_duckdb()
        self._connector = connector
        self._source_table = source_table
        self._source_column_map = source_column_map
        self._levels: list[CascadeLevel] = []
        self._engines: list[TriggerEngine] = []
        self._built = False

    # ------------------------------------------------------------------
    # Level management
    # ------------------------------------------------------------------

    def add_level(self, level: CascadeLevel) -> None:
        """Append a cascade level.

        Levels are processed in the order they are added.  The first level
        watches the raw source table; subsequent levels watch the preceding
        level's predictions.
        """
        _validate_identifier(level.name)
        if any(lv.name == level.name for lv in self._levels):
            raise ValueError(f"Duplicate cascade level name: {level.name!r}")
        self._levels.append(level)
        self._built = False

    @property
    def levels(self) -> list[str]:
        """Return ordered level names."""
        return [lv.name for lv in self._levels]

    # ------------------------------------------------------------------
    # Internal wiring
    # ------------------------------------------------------------------

    def _table_prefix(self, level_name: str) -> str:
        return f"_cascade_{level_name}"

    def _predictions_table(self, level_name: str) -> str:
        return f"_cascade_{level_name}_predictions_live"

    def _alerts_table(self, level_name: str) -> str:
        return f"_cascade_{level_name}_alerts"

    def _input_view(self, level_name: str) -> str:
        return f"_cascade_{level_name}_input"

    def _build(self) -> None:
        """Wire up TriggerEngines and inter-level views."""
        if self._built:
            return
        if not self._levels:
            raise ValueError("No cascade levels defined")

        conn = self._connector._ensure_open()
        self._engines.clear()

        for i, level in enumerate(self._levels):
            prefix = self._table_prefix(level.name)

            if i == 0:
                # First level watches the raw source table
                engine = TriggerEngine(
                    self._connector,
                    sequences_table=self._source_table,
                    column_map=self._source_column_map,
                    num_prediction_steps=level.num_prediction_steps,
                    table_prefix=prefix,
                )
            else:
                # Ensure the upstream engine's tables exist before creating the view
                self._engines[-1]._ensure_tables()

                # Create a view that reshapes upstream predictions → events
                prev_level = self._levels[i - 1]
                upstream_table = self._predictions_table(prev_level.name)
                view_name = self._input_view(level.name)
                safe_view = _validate_identifier(view_name)

                if level.view_query is not None:
                    # User-provided SQL with {upstream_table} placeholder
                    view_sql = level.view_query.format(upstream_table=_validate_identifier(upstream_table))
                else:
                    # Default: map prediction columns to event-sequence format
                    safe_upstream = _validate_identifier(upstream_table)
                    view_sql = (
                        f"SELECT sequence_id, prediction AS embedding, "
                        f"updated_at AS timestamp, 'prediction' AS modality "
                        f"FROM {safe_upstream}"
                    )

                conn.execute(f"CREATE OR REPLACE VIEW {safe_view} AS ({view_sql})")

                engine = TriggerEngine(
                    self._connector,
                    sequences_table=view_name,
                    column_map={
                        "sequence_id": "sequence_id",
                        "embedding": "embedding",
                        "timestamp": "timestamp",
                        "modality": "modality",
                    },
                    num_prediction_steps=level.num_prediction_steps,
                    table_prefix=prefix,
                )

            # Register rules
            for rule in level.rules:
                engine.add_rule(rule)

            self._engines.append(engine)

        self._built = True

    # ------------------------------------------------------------------
    # Core polling
    # ------------------------------------------------------------------

    def poll_once(self) -> dict[str, list[dict[str, Any]]]:
        """Poll all levels in cascade order.

        Returns:
            Dict mapping level name to list of fired alerts.
        """
        self._build()
        results: dict[str, list[dict[str, Any]]] = {}
        for level, engine in zip(self._levels, self._engines):
            alerts = engine.poll_once()
            results[level.name] = alerts
        return results

    # ------------------------------------------------------------------
    # Watch loops
    # ------------------------------------------------------------------

    def watch(
        self,
        interval_seconds: float = 10.0,
        max_iterations: int | None = None,
    ) -> None:
        """Blocking watch loop that polls all cascade levels each iteration.

        Args:
            interval_seconds: Seconds between polls.
            max_iterations: Stop after this many iterations (``None`` for infinite).
        """
        self._build()
        logger.info(
            "ForecastCascade watching %d levels: %s (interval=%.1fs)",
            len(self._levels),
            [lv.name for lv in self._levels],
            interval_seconds,
        )

        iterations = 0
        while max_iterations is None or iterations < max_iterations:
            try:
                result = self.poll_once()
                total_alerts = sum(len(a) for a in result.values())
                if total_alerts:
                    logger.info("Cascade poll fired %d total alert(s)", total_alerts)
            except Exception:
                logger.exception("Error during cascade poll")
            iterations += 1
            if max_iterations is None or iterations < max_iterations:
                time.sleep(interval_seconds)

    def watch_async(self, interval_seconds: float = 10.0) -> StopHandle:
        """Start watching in a background thread.

        Returns:
            A :class:`StopHandle` to stop the watcher.
        """
        stop_event = threading.Event()

        def _loop() -> None:
            self._build()
            logger.info(
                "ForecastCascade async watching %d levels: %s (interval=%.1fs)",
                len(self._levels),
                [lv.name for lv in self._levels],
                interval_seconds,
            )
            while not stop_event.is_set():
                try:
                    result = self.poll_once()
                    total_alerts = sum(len(a) for a in result.values())
                    if total_alerts:
                        logger.info("Cascade poll fired %d total alert(s)", total_alerts)
                except Exception:
                    logger.exception("Error during cascade poll")
                stop_event.wait(timeout=interval_seconds)

        thread = threading.Thread(target=_loop, daemon=True, name="forecast-cascade")
        thread.start()
        return StopHandle(thread, stop_event)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_predictions(
        self,
        level: str,
        sequence_id: str | None = None,
    ) -> dict[str, list[list[float]]] | list[list[float]]:
        """Get live predictions for a cascade level.

        Args:
            level: Level name (e.g. ``"patient"``).
            sequence_id: If provided, return predictions for this sequence
                only (as a list of vectors).  Otherwise return a dict
                mapping sequence_id to predictions.

        Returns:
            Predictions for the requested level.
        """
        self._build()
        engine = self._get_engine(level)

        if sequence_id is not None:
            return engine.get_live_predictions(sequence_id)

        # Return all predictions for this level
        conn = self._connector._ensure_open()
        preds_t = _validate_identifier(self._predictions_table(level))
        rows = conn.execute(
            f"SELECT sequence_id, step, prediction FROM {preds_t} ORDER BY sequence_id, step"
        ).fetchall()

        result: dict[str, list[list[float]]] = {}
        for sid, _step, pred in rows:
            if sid not in result:
                result[sid] = []
            result[sid].append([float(v) for v in pred])
        return result

    def get_alerts(
        self,
        level: str | None = None,
        sequence_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query alerts across one or all cascade levels.

        Args:
            level: Filter to a specific level.  If ``None``, returns alerts
                from all levels (with an added ``level`` key).
            sequence_id: Filter by sequence.
            limit: Max alerts per level.

        Returns:
            List of alert dicts, each with an added ``level`` key.
        """
        self._build()

        if level is not None:
            engine = self._get_engine(level)
            alerts = engine.get_alerts(sequence_id=sequence_id, limit=limit)
            for a in alerts:
                a["level"] = level
            return alerts

        # All levels
        all_alerts: list[dict[str, Any]] = []
        for lvl, engine in zip(self._levels, self._engines):
            alerts = engine.get_alerts(sequence_id=sequence_id, limit=limit)
            for a in alerts:
                a["level"] = lvl.name
            all_alerts.extend(alerts)

        # Sort by fired_at descending, limit
        all_alerts.sort(key=lambda a: a.get("fired_at", 0), reverse=True)
        return all_alerts[:limit]

    def get_engine(self, level: str) -> TriggerEngine:
        """Get the TriggerEngine for a specific level.

        Useful for adding rules or actions after construction.
        """
        self._build()
        return self._get_engine(level)

    def _get_engine(self, level: str) -> TriggerEngine:
        for lvl, engine in zip(self._levels, self._engines):
            if lvl.name == level:
                return engine
        raise ValueError(f"Unknown cascade level: {level!r}")

    def add_action(self, level: str, name: str, callback: Callable[[dict[str, Any]], None]) -> None:
        """Register a custom action on a specific level's engine."""
        self._build()
        self._get_engine(level).add_action(name, callback)

    def add_rule(self, level: str, rule: AlertRule) -> None:
        """Add an alert rule to a specific level after construction."""
        self._build()
        self._get_engine(level).add_rule(rule)
