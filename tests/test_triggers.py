"""Tests for the auto-trigger reactive pipeline.

Skipped entirely if DuckDB is not available.
"""

import time

import pytest

duckdb = pytest.importorskip("duckdb")

from event_jepa_cube.duckdb_connector import DuckDBConnector  # noqa: E402, I001
from event_jepa_cube.triggers import (  # noqa: E402
    AlertRule,
    TriggerEngine,
    get_action,
    register_action,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def connector():
    conn = DuckDBConnector(database=":memory:", embedding_dim=3)
    db = conn._ensure_open()
    db.execute("""
        CREATE TABLE event_sequences (
            sequence_id VARCHAR,
            embedding FLOAT[],
            timestamp DOUBLE,
            modality VARCHAR
        )
    """)
    yield conn
    conn.close()


@pytest.fixture
def engine(connector):
    return TriggerEngine(connector, sequences_table="event_sequences", num_prediction_steps=2)


def _insert_event(connector: DuckDBConnector, sid: str, emb: list[float], ts: float) -> None:
    """Helper to insert a single event row."""
    conn = connector._ensure_open()
    conn.execute(
        "INSERT INTO event_sequences VALUES ($1, $2, $3, $4)",
        [sid, emb, ts, "text"],
    )


# ---------------------------------------------------------------------------
# Action registry
# ---------------------------------------------------------------------------


class TestActionRegistry:
    def test_builtin_log_action_exists(self):
        assert get_action("log") is not None

    def test_register_custom_action(self):
        fired = []

        @register_action("test_custom")
        def custom_action(alert: dict) -> None:
            fired.append(alert)

        fn = get_action("test_custom")
        assert fn is not None
        fn({"test": True})
        assert len(fired) == 1

    def test_get_missing_action(self):
        assert get_action("nonexistent_action_xyz") is None


# ---------------------------------------------------------------------------
# AlertRule
# ---------------------------------------------------------------------------


class TestAlertRule:
    def test_create_rule(self):
        rule = AlertRule(
            name="test",
            condition=lambda r: True,
            severity="warning",
            message="Alert for {sequence_id}",
        )
        assert rule.name == "test"
        assert rule.severity == "warning"
        assert rule.actions == ["log"]

    def test_custom_actions(self):
        rule = AlertRule(
            name="test",
            condition=lambda r: True,
            actions=["log", "webhook"],
        )
        assert rule.actions == ["log", "webhook"]


# ---------------------------------------------------------------------------
# TriggerEngine setup
# ---------------------------------------------------------------------------


class TestTriggerEngineSetup:
    def test_add_remove_rule(self, engine):
        rule = AlertRule(name="r1", condition=lambda r: True)
        engine.add_rule(rule)
        assert len(engine.rules) == 1
        engine.remove_rule("r1")
        assert len(engine.rules) == 0

    def test_add_action(self, engine):
        calls = []
        engine.add_action("custom", lambda a: calls.append(a))
        assert engine._resolve_action("custom") is not None

    def test_ensure_tables_creates_tables(self, engine, connector):
        engine._ensure_tables()
        conn = connector._ensure_open()
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            ).fetchall()
        ]
        assert "_alerts" in tables
        assert "_predictions_live" in tables

    def test_high_water_mark_initially_zero(self, engine):
        assert engine.high_water_mark == 0.0

    def test_detect_high_water_mark(self, engine, connector):
        _insert_event(connector, "s1", [1.0, 0.0, 0.0], 10.0)
        _insert_event(connector, "s1", [0.0, 1.0, 0.0], 20.0)
        engine._detect_high_water_mark()
        assert engine.high_water_mark == 20.0


# ---------------------------------------------------------------------------
# poll_once
# ---------------------------------------------------------------------------


class TestPollOnce:
    def test_first_poll_sets_baseline(self, engine, connector):
        _insert_event(connector, "s1", [1.0, 0.0, 0.0], 1.0)
        _insert_event(connector, "s1", [0.0, 1.0, 0.0], 2.0)

        # First poll: sets baseline, returns empty
        alerts = engine.poll_once()
        assert alerts == []
        assert engine.high_water_mark == 2.0

    def test_second_poll_detects_new_records(self, engine, connector):
        # Seed initial data
        _insert_event(connector, "s1", [1.0, 0.0, 0.0], 1.0)
        _insert_event(connector, "s1", [0.0, 1.0, 0.0], 2.0)
        engine.poll_once()  # baseline

        # Insert new record
        _insert_event(connector, "s1", [0.0, 0.0, 1.0], 3.0)

        # Second poll should process
        alerts = engine.poll_once()
        # No rules → no alerts, but processing happened
        assert alerts == []
        assert engine.high_water_mark == 3.0

    def test_poll_fires_alert_on_rule_match(self, engine, connector):
        # Always-true rule
        engine.add_rule(
            AlertRule(
                name="always_fire",
                condition=lambda r: True,
                severity="warning",
                message="Triggered for {sequence_id}",
                actions=[],  # no actions, just track
            )
        )

        # Seed + baseline
        _insert_event(connector, "s1", [1.0, 0.0, 0.0], 1.0)
        _insert_event(connector, "s1", [0.0, 1.0, 0.0], 2.0)
        engine.poll_once()

        # New event triggers processing
        _insert_event(connector, "s1", [0.0, 0.0, 1.0], 3.0)
        alerts = engine.poll_once()

        assert len(alerts) == 1
        assert alerts[0]["rule_name"] == "always_fire"
        assert alerts[0]["severity"] == "warning"
        assert "s1" in alerts[0]["message"]

    def test_poll_does_not_fire_on_false_condition(self, engine, connector):
        engine.add_rule(
            AlertRule(
                name="never_fire",
                condition=lambda r: False,
            )
        )

        _insert_event(connector, "s1", [1.0, 0.0, 0.0], 1.0)
        _insert_event(connector, "s1", [0.0, 1.0, 0.0], 2.0)
        engine.poll_once()

        _insert_event(connector, "s1", [0.0, 0.0, 1.0], 3.0)
        alerts = engine.poll_once()
        assert alerts == []

    def test_poll_prefills_live_predictions(self, engine, connector):
        _insert_event(connector, "s1", [1.0, 0.0, 0.0], 1.0)
        _insert_event(connector, "s1", [0.0, 1.0, 0.0], 2.0)
        engine.poll_once()  # baseline

        _insert_event(connector, "s1", [0.0, 0.0, 1.0], 3.0)
        engine.poll_once()

        preds = engine.get_live_predictions("s1")
        assert len(preds) == 2  # num_prediction_steps=2
        assert all(len(p) == 3 for p in preds)  # embedding_dim=3

    def test_poll_handles_multiple_sequences(self, engine, connector):
        engine.add_rule(AlertRule(name="all", condition=lambda r: True, actions=[]))

        # Seed two sequences
        _insert_event(connector, "s1", [1.0, 0.0, 0.0], 1.0)
        _insert_event(connector, "s2", [0.5, 0.5, 0.0], 1.0)
        engine.poll_once()

        # New events for both
        _insert_event(connector, "s1", [0.0, 1.0, 0.0], 2.0)
        _insert_event(connector, "s2", [0.0, 0.5, 0.5], 2.0)
        alerts = engine.poll_once()

        assert len(alerts) == 2
        sids = {a["sequence_id"] for a in alerts}
        assert sids == {"s1", "s2"}

    def test_poll_no_new_records_returns_empty(self, engine, connector):
        _insert_event(connector, "s1", [1.0, 0.0, 0.0], 1.0)
        engine.poll_once()  # baseline

        alerts = engine.poll_once()  # no new data
        assert alerts == []


# ---------------------------------------------------------------------------
# Alert persistence
# ---------------------------------------------------------------------------


class TestAlertPersistence:
    def test_alerts_written_to_db(self, engine, connector):
        engine.add_rule(AlertRule(name="persist_test", condition=lambda r: True, actions=[]))

        _insert_event(connector, "s1", [1.0, 0.0, 0.0], 1.0)
        _insert_event(connector, "s1", [0.0, 1.0, 0.0], 2.0)
        engine.poll_once()

        _insert_event(connector, "s1", [0.0, 0.0, 1.0], 3.0)
        engine.poll_once()

        alerts = engine.get_alerts()
        assert len(alerts) == 1
        assert alerts[0]["rule_name"] == "persist_test"

    def test_get_alerts_filters_by_sequence(self, engine, connector):
        engine.add_rule(AlertRule(name="multi", condition=lambda r: True, actions=[]))

        _insert_event(connector, "s1", [1.0, 0.0, 0.0], 1.0)
        _insert_event(connector, "s2", [0.5, 0.5, 0.0], 1.5)
        engine.poll_once()

        _insert_event(connector, "s1", [0.0, 1.0, 0.0], 2.0)
        _insert_event(connector, "s2", [0.0, 0.5, 0.5], 2.5)
        engine.poll_once()

        s1_alerts = engine.get_alerts(sequence_id="s1")
        assert all(a["sequence_id"] == "s1" for a in s1_alerts)


# ---------------------------------------------------------------------------
# Action firing
# ---------------------------------------------------------------------------


class TestActionFiring:
    def test_custom_action_is_called(self, engine, connector):
        fired_alerts = []
        engine.add_action("capture", lambda a: fired_alerts.append(a))
        engine.add_rule(
            AlertRule(
                name="fire_action",
                condition=lambda r: True,
                actions=["capture"],
            )
        )

        _insert_event(connector, "s1", [1.0, 0.0, 0.0], 1.0)
        engine.poll_once()

        _insert_event(connector, "s1", [0.0, 1.0, 0.0], 2.0)
        engine.poll_once()

        assert len(fired_alerts) == 1
        assert fired_alerts[0]["sequence_id"] == "s1"

    def test_multiple_actions_fired(self, engine, connector):
        calls_a = []
        calls_b = []
        engine.add_action("a", lambda alert: calls_a.append(alert))
        engine.add_action("b", lambda alert: calls_b.append(alert))
        engine.add_rule(
            AlertRule(
                name="multi_action",
                condition=lambda r: True,
                actions=["a", "b"],
            )
        )

        _insert_event(connector, "s1", [1.0, 0.0, 0.0], 1.0)
        engine.poll_once()

        _insert_event(connector, "s1", [0.0, 1.0, 0.0], 2.0)
        engine.poll_once()

        assert len(calls_a) == 1
        assert len(calls_b) == 1


# ---------------------------------------------------------------------------
# Watch loop
# ---------------------------------------------------------------------------


class TestWatch:
    def test_watch_with_max_iterations(self, engine, connector):
        _insert_event(connector, "s1", [1.0, 0.0, 0.0], 1.0)
        # Should not hang — max_iterations limits the loop
        engine.watch(interval_seconds=0.01, max_iterations=3)

    def test_watch_async_starts_and_stops(self, engine, connector):
        _insert_event(connector, "s1", [1.0, 0.0, 0.0], 1.0)
        handle = engine.watch_async(interval_seconds=0.05)
        assert handle.is_running
        time.sleep(0.15)  # let it poll a few times
        handle.stop()
        assert not handle.is_running

    def test_watch_async_detects_new_records(self, engine, connector):
        fired = []
        engine.add_action("track", lambda a: fired.append(a))
        engine.add_rule(AlertRule(name="async_test", condition=lambda r: True, actions=["track"]))

        _insert_event(connector, "s1", [1.0, 0.0, 0.0], 1.0)
        handle = engine.watch_async(interval_seconds=0.05)
        time.sleep(0.15)  # let baseline establish

        # Insert new event
        _insert_event(connector, "s1", [0.0, 1.0, 0.0], 2.0)
        time.sleep(0.2)  # let it detect

        handle.stop()
        assert len(fired) >= 1
        assert fired[0]["sequence_id"] == "s1"


# ---------------------------------------------------------------------------
# process_sequence (on DuckDBConnector)
# ---------------------------------------------------------------------------


class TestProcessSequence:
    def test_process_sequence_returns_result(self, connector):
        from event_jepa_cube.sequence import EventSequence

        seq = EventSequence(
            embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            timestamps=[1.0, 2.0, 3.0],
        )
        result = connector.process_sequence("test_seq", seq, num_prediction_steps=2)

        assert result["sequence_id"] == "test_seq"
        assert len(result["representation"]) == 3
        assert isinstance(result["patterns"], list)
        assert len(result["predictions"]) == 2
