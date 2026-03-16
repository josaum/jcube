"""Auto-trigger reactive pipeline.

Watches DuckDB tables for new records, runs the EventJEPA pipeline
incrementally on affected sequences, pre-fills prediction tables,
evaluates alert rules, and fires registered action callbacks.

Requires DuckDB. Install with: pip install event-jepa-cube[duckdb]
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from .duckdb_connector import DuckDBConnector, _require_duckdb, _validate_identifier

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Action registry
# ---------------------------------------------------------------------------

_action_registry: dict[str, Callable[[dict[str, Any]], None]] = {}


def register_action(name: str) -> Callable[[Callable[[dict[str, Any]], None]], Callable[[dict[str, Any]], None]]:
    """Decorator to register a named action callback.

    Example::

        @register_action("log")
        def log_alert(alert: dict) -> None:
            print(f"[{alert['severity']}] {alert['message']}")
    """

    def decorator(fn: Callable[[dict[str, Any]], None]) -> Callable[[dict[str, Any]], None]:
        _action_registry[name] = fn
        return fn

    return decorator


def get_action(name: str) -> Callable[[dict[str, Any]], None] | None:
    """Retrieve a registered action by name."""
    return _action_registry.get(name)


# ---------------------------------------------------------------------------
# Built-in actions
# ---------------------------------------------------------------------------


@register_action("log")
def _log_action(alert: dict[str, Any]) -> None:
    """Built-in action: log the alert."""
    level = {"info": logging.INFO, "warning": logging.WARNING, "critical": logging.CRITICAL}.get(
        alert.get("severity", "info"), logging.INFO
    )
    logger.log(
        level, "[%s] seq=%s rule=%s: %s", alert["severity"], alert["sequence_id"], alert["rule_name"], alert["message"]
    )


# ---------------------------------------------------------------------------
# AlertRule
# ---------------------------------------------------------------------------


@dataclass
class AlertRule:
    """A rule that evaluates pipeline results and may trigger actions.

    The *condition* callable receives a single-sequence pipeline result dict
    with keys: ``sequence_id``, ``representation``, ``patterns``,
    ``predictions``.  It should return ``True`` to fire the alert.

    The *message* string may contain ``{sequence_id}``, ``{patterns}``,
    ``{num_predictions}`` placeholders that are auto-formatted.
    """

    name: str
    condition: Callable[[dict[str, Any]], bool]
    severity: str = "info"
    message: str = "Alert triggered for {sequence_id}"
    actions: list[str] = field(default_factory=lambda: ["log"])


# ---------------------------------------------------------------------------
# Stop handle for background watch
# ---------------------------------------------------------------------------


class StopHandle:
    """Handle returned by :meth:`TriggerEngine.watch_async` to stop the background watcher."""

    def __init__(self, thread: threading.Thread, stop_event: threading.Event) -> None:
        self._thread = thread
        self._stop_event = stop_event

    def stop(self) -> None:
        """Signal the watcher to stop and wait for it to finish."""
        self._stop_event.set()
        self._thread.join(timeout=30)

    @property
    def is_running(self) -> bool:
        return self._thread.is_alive()


# ---------------------------------------------------------------------------
# TriggerEngine
# ---------------------------------------------------------------------------


class TriggerEngine:
    """Reactive engine that watches for new events and runs the pipeline automatically.

    Example::

        with DuckDBConnector(embedding_dim=3) as conn:
            engine = TriggerEngine(conn)
            engine.add_rule(AlertRule(
                name="high_activity",
                condition=lambda r: len(r["patterns"]) > 3,
                severity="warning",
                message="High pattern activity on {sequence_id}",
            ))
            engine.watch(interval_seconds=5)  # blocking

    Or in the background::

        handle = engine.watch_async(interval_seconds=5)
        # ... do other work ...
        handle.stop()
    """

    def __init__(
        self,
        connector: DuckDBConnector,
        *,
        sequences_table: str = "event_sequences",
        column_map: dict[str, str] | None = None,
        num_prediction_steps: int = 3,
    ) -> None:
        _require_duckdb()
        self._connector = connector
        self._sequences_table = sequences_table
        self._column_map = column_map
        self._num_prediction_steps = num_prediction_steps
        self._rules: dict[str, AlertRule] = {}
        self._actions: dict[str, Callable[[dict[str, Any]], None]] = {}
        self._high_water_mark: float = 0.0
        self._initialized = False

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _ensure_tables(self) -> None:
        """Create internal tracking tables if they don't exist."""
        if self._initialized:
            return
        conn = self._connector._ensure_open()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS _alerts (
                alert_id VARCHAR,
                sequence_id VARCHAR,
                rule_name VARCHAR,
                severity VARCHAR,
                message VARCHAR,
                fired_at DOUBLE,
                pipeline_result VARCHAR
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS _predictions_live (
                sequence_id VARCHAR,
                step INTEGER,
                prediction FLOAT[],
                updated_at DOUBLE
            )
        """)
        self._initialized = True

    def _detect_high_water_mark(self) -> None:
        """Set the initial high-water mark to the current max timestamp."""
        conn = self._connector._ensure_open()
        cmap = {**{"timestamp": "timestamp"}, **(self._column_map or {})}
        ts_col = _validate_identifier(cmap["timestamp"])
        safe_table = _validate_identifier(self._sequences_table)
        try:
            row = conn.execute(f"SELECT MAX({ts_col}) FROM {safe_table}").fetchone()
            self._high_water_mark = float(row[0]) if row and row[0] is not None else 0.0
        except Exception:
            self._high_water_mark = 0.0

    # ------------------------------------------------------------------
    # Rules and actions
    # ------------------------------------------------------------------

    def add_rule(self, rule: AlertRule) -> None:
        """Register an alert rule."""
        self._rules[rule.name] = rule

    def remove_rule(self, name: str) -> None:
        """Remove an alert rule by name."""
        self._rules.pop(name, None)

    def add_action(self, name: str, callback: Callable[[dict[str, Any]], None]) -> None:
        """Register a custom action callback (in addition to the global registry)."""
        self._actions[name] = callback

    @property
    def rules(self) -> list[AlertRule]:
        return list(self._rules.values())

    @property
    def high_water_mark(self) -> float:
        return self._high_water_mark

    # ------------------------------------------------------------------
    # Core polling
    # ------------------------------------------------------------------

    def _get_new_records_query(self) -> str:
        """Build SQL to fetch records newer than the high-water mark."""
        cmap = {**{"sequence_id": "sequence_id", "timestamp": "timestamp"}, **(self._column_map or {})}
        ts_col = _validate_identifier(cmap["timestamp"])
        safe_table = _validate_identifier(self._sequences_table)
        return f"SELECT * FROM {safe_table} WHERE {ts_col} > {self._high_water_mark}"

    def _resolve_action(self, name: str) -> Callable[[dict[str, Any]], None] | None:
        """Look up an action by name: local registry first, then global."""
        return self._actions.get(name) or get_action(name)

    def _format_message(self, rule: AlertRule, result: dict[str, Any]) -> str:
        """Format an alert message with pipeline result placeholders."""
        return rule.message.format(
            sequence_id=result.get("sequence_id", ""),
            patterns=result.get("patterns", []),
            num_predictions=len(result.get("predictions", [])),
        )

    def _fire_alert(self, rule: AlertRule, result: dict[str, Any]) -> dict[str, Any]:
        """Write alert to DB and fire action callbacks."""
        conn = self._connector._ensure_open()
        message = self._format_message(rule, result)
        alert_id = str(uuid.uuid4())
        fired_at = time.time()

        alert = {
            "alert_id": alert_id,
            "sequence_id": result["sequence_id"],
            "rule_name": rule.name,
            "severity": rule.severity,
            "message": message,
            "fired_at": fired_at,
        }

        # Write to alerts table
        result_json = json.dumps(
            {k: v for k, v in result.items() if k != "representation"},
            default=str,
        )
        conn.execute(
            'INSERT INTO "_alerts" VALUES ($1, $2, $3, $4, $5, $6, $7)',
            [alert_id, result["sequence_id"], rule.name, rule.severity, message, fired_at, result_json],
        )

        # Fire action callbacks
        for action_name in rule.actions:
            action_fn = self._resolve_action(action_name)
            if action_fn is not None:
                try:
                    action_fn(alert)
                except Exception:
                    logger.exception("Action %s failed for alert %s", action_name, alert_id)

        return alert

    def _upsert_predictions(self, sequence_id: str, predictions: list[list[float]]) -> None:
        """Upsert live predictions for a sequence (delete old + insert new)."""
        conn = self._connector._ensure_open()
        now = time.time()
        conn.execute('DELETE FROM "_predictions_live" WHERE sequence_id = $1', [sequence_id])
        for step, pred in enumerate(predictions, start=1):
            conn.execute(
                'INSERT INTO "_predictions_live" VALUES ($1, $2, $3, $4)',
                [sequence_id, step, pred, now],
            )

    def poll_once(self) -> list[dict[str, Any]]:
        """Check for new records, process affected sequences, evaluate rules.

        Returns:
            List of fired alerts (may be empty if no rules triggered).
        """
        self._ensure_tables()

        if self._high_water_mark == 0.0:
            self._detect_high_water_mark()
            return []  # First poll: just set the baseline

        # Find new records
        query = self._get_new_records_query()
        new_sequences = self._connector.load_sequences(query=query, column_map=self._column_map)

        if not new_sequences:
            return []

        # Update high-water mark from new records
        cmap = {**{"timestamp": "timestamp"}, **(self._column_map or {})}
        ts_col = _validate_identifier(cmap["timestamp"])
        safe_table = _validate_identifier(self._sequences_table)
        conn = self._connector._ensure_open()
        row = conn.execute(f"SELECT MAX({ts_col}) FROM {safe_table}").fetchone()
        if row and row[0] is not None:
            self._high_water_mark = float(row[0])

        # For each affected sequence, load FULL history and process
        affected_sids = list(new_sequences.keys())
        all_alerts: list[dict[str, Any]] = []

        for sid in affected_sids:
            # Load full sequence history for this sequence_id
            cmap_full = {**{"sequence_id": "sequence_id"}, **(self._column_map or {})}
            sid_col = _validate_identifier(cmap_full["sequence_id"])
            full_query = f"SELECT * FROM {safe_table} WHERE {sid_col} = '{sid}'"
            full_sequences = self._connector.load_sequences(query=full_query, column_map=self._column_map)

            if sid not in full_sequences:
                continue

            full_seq = full_sequences[sid]
            result = self._connector.process_sequence(sid, full_seq, self._num_prediction_steps)

            # Pre-fill live predictions
            self._upsert_predictions(sid, result["predictions"])

            # Write updated representation
            self._connector.write_representations({sid: result["representation"]})

            # Evaluate rules
            for rule in self._rules.values():
                try:
                    if rule.condition(result):
                        alert = self._fire_alert(rule, result)
                        all_alerts.append(alert)
                except Exception:
                    logger.exception("Rule %s failed on sequence %s", rule.name, sid)

        return all_alerts

    # ------------------------------------------------------------------
    # Watch loops
    # ------------------------------------------------------------------

    def watch(self, interval_seconds: float = 10.0, max_iterations: int | None = None) -> None:
        """Blocking watch loop that polls for new records.

        Args:
            interval_seconds: Seconds between polls.
            max_iterations: Stop after this many iterations (``None`` for infinite).
        """
        self._ensure_tables()
        self._detect_high_water_mark()
        logger.info("TriggerEngine watching %s (interval=%.1fs)", self._sequences_table, interval_seconds)

        iterations = 0
        while max_iterations is None or iterations < max_iterations:
            try:
                alerts = self.poll_once()
                if alerts:
                    logger.info("Poll fired %d alert(s)", len(alerts))
            except Exception:
                logger.exception("Error during poll")
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
            self._ensure_tables()
            self._detect_high_water_mark()
            logger.info("TriggerEngine async watching %s (interval=%.1fs)", self._sequences_table, interval_seconds)
            while not stop_event.is_set():
                try:
                    alerts = self.poll_once()
                    if alerts:
                        logger.info("Poll fired %d alert(s)", len(alerts))
                except Exception:
                    logger.exception("Error during poll")
                stop_event.wait(timeout=interval_seconds)

        thread = threading.Thread(target=_loop, daemon=True, name="trigger-engine")
        thread.start()
        return StopHandle(thread, stop_event)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_alerts(self, sequence_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        """Query fired alerts from the ``_alerts`` table.

        Args:
            sequence_id: Filter by sequence (optional).
            limit: Max number of alerts to return.

        Returns:
            List of alert dicts.
        """
        self._ensure_tables()
        conn = self._connector._ensure_open()
        if sequence_id:
            rows = conn.execute(
                'SELECT * FROM "_alerts" WHERE sequence_id = $1 ORDER BY fired_at DESC LIMIT $2',
                [sequence_id, limit],
            ).fetchall()
        else:
            rows = conn.execute(f'SELECT * FROM "_alerts" ORDER BY fired_at DESC LIMIT {limit}').fetchall()
        cols = ["alert_id", "sequence_id", "rule_name", "severity", "message", "fired_at", "pipeline_result"]
        return [dict(zip(cols, row)) for row in rows]

    def get_live_predictions(self, sequence_id: str) -> list[list[float]]:
        """Get the latest pre-filled predictions for a sequence.

        Returns:
            List of prediction vectors ordered by step.
        """
        self._ensure_tables()
        conn = self._connector._ensure_open()
        rows = conn.execute(
            'SELECT prediction FROM "_predictions_live" WHERE sequence_id = $1 ORDER BY step',
            [sequence_id],
        ).fetchall()
        return [[float(v) for v in row[0]] for row in rows]
