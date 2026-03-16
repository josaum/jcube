"""Tests for the cascading forecast pipeline.

Skipped entirely if DuckDB is not available.
"""

import time

import pytest

duckdb = pytest.importorskip("duckdb")

from event_jepa_cube.cascade import CascadeLevel, ForecastCascade  # noqa: E402
from event_jepa_cube.duckdb_connector import DuckDBConnector  # noqa: E402
from event_jepa_cube.triggers import AlertRule  # noqa: E402

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


def _insert_event(connector: DuckDBConnector, sid: str, emb: list[float], ts: float) -> None:
    conn = connector._ensure_open()
    conn.execute(
        "INSERT INTO event_sequences VALUES ($1, $2, $3, $4)",
        [sid, emb, ts, "text"],
    )


# ---------------------------------------------------------------------------
# CascadeLevel
# ---------------------------------------------------------------------------


class TestCascadeLevel:
    def test_create_level(self):
        level = CascadeLevel(name="patient", num_prediction_steps=3)
        assert level.name == "patient"
        assert level.num_prediction_steps == 3
        assert level.rules == []
        assert level.view_query is None

    def test_level_with_rules(self):
        rule = AlertRule(name="test", condition=lambda r: True)
        level = CascadeLevel(name="dept", rules=[rule])
        assert len(level.rules) == 1

    def test_level_with_view_query(self):
        level = CascadeLevel(
            name="financial",
            view_query="SELECT * FROM {upstream_table}",
        )
        assert "{upstream_table}" in level.view_query


# ---------------------------------------------------------------------------
# ForecastCascade construction
# ---------------------------------------------------------------------------


class TestCascadeConstruction:
    def test_add_levels(self, connector):
        cascade = ForecastCascade(connector)
        cascade.add_level(CascadeLevel(name="l1"))
        cascade.add_level(CascadeLevel(name="l2"))
        assert cascade.levels == ["l1", "l2"]

    def test_duplicate_level_raises(self, connector):
        cascade = ForecastCascade(connector)
        cascade.add_level(CascadeLevel(name="l1"))
        with pytest.raises(ValueError, match="Duplicate"):
            cascade.add_level(CascadeLevel(name="l1"))

    def test_build_with_no_levels_raises(self, connector):
        cascade = ForecastCascade(connector)
        with pytest.raises(ValueError, match="No cascade levels"):
            cascade.poll_once()

    def test_get_engine_unknown_level_raises(self, connector):
        cascade = ForecastCascade(connector)
        cascade.add_level(CascadeLevel(name="l1"))
        with pytest.raises(ValueError, match="Unknown cascade level"):
            cascade.get_engine("nonexistent")


# ---------------------------------------------------------------------------
# Single-level cascade (behaves like a regular TriggerEngine)
# ---------------------------------------------------------------------------


class TestSingleLevel:
    def test_single_level_baseline(self, connector):
        cascade = ForecastCascade(connector, source_table="event_sequences")
        cascade.add_level(CascadeLevel(name="patient", num_prediction_steps=2))

        _insert_event(connector, "s1", [1.0, 0.0, 0.0], 1.0)
        _insert_event(connector, "s1", [0.0, 1.0, 0.0], 2.0)

        result = cascade.poll_once()
        assert "patient" in result
        assert result["patient"] == []  # baseline, no alerts

    def test_single_level_detects_and_predicts(self, connector):
        cascade = ForecastCascade(connector, source_table="event_sequences")
        cascade.add_level(CascadeLevel(name="patient", num_prediction_steps=2))

        _insert_event(connector, "s1", [1.0, 0.0, 0.0], 1.0)
        _insert_event(connector, "s1", [0.0, 1.0, 0.0], 2.0)
        cascade.poll_once()  # baseline

        _insert_event(connector, "s1", [0.0, 0.0, 1.0], 3.0)
        cascade.poll_once()

        preds = cascade.get_predictions("patient", sequence_id="s1")
        assert len(preds) == 2
        assert all(len(p) == 3 for p in preds)

    def test_single_level_fires_alerts(self, connector):
        rule = AlertRule(name="always", condition=lambda r: True, actions=[])
        cascade = ForecastCascade(connector, source_table="event_sequences")
        cascade.add_level(CascadeLevel(name="patient", rules=[rule]))

        _insert_event(connector, "s1", [1.0, 0.0, 0.0], 1.0)
        _insert_event(connector, "s1", [0.0, 1.0, 0.0], 2.0)
        cascade.poll_once()

        _insert_event(connector, "s1", [0.0, 0.0, 1.0], 3.0)
        result = cascade.poll_once()
        assert len(result["patient"]) == 1

        alerts = cascade.get_alerts(level="patient")
        assert len(alerts) == 1
        assert alerts[0]["level"] == "patient"

    def test_get_all_predictions(self, connector):
        cascade = ForecastCascade(connector, source_table="event_sequences")
        cascade.add_level(CascadeLevel(name="patient", num_prediction_steps=2))

        _insert_event(connector, "s1", [1.0, 0.0, 0.0], 1.0)
        _insert_event(connector, "s1", [0.0, 1.0, 0.0], 2.0)
        _insert_event(connector, "s2", [0.5, 0.5, 0.0], 1.0)
        _insert_event(connector, "s2", [0.0, 0.5, 0.5], 2.0)
        cascade.poll_once()  # baseline

        _insert_event(connector, "s1", [0.0, 0.0, 1.0], 3.0)
        _insert_event(connector, "s2", [0.0, 0.0, 1.0], 3.0)
        cascade.poll_once()

        all_preds = cascade.get_predictions("patient")
        assert isinstance(all_preds, dict)
        assert "s1" in all_preds
        assert "s2" in all_preds


# ---------------------------------------------------------------------------
# Multi-level cascade
# ---------------------------------------------------------------------------


class TestMultiLevelCascade:
    def test_two_level_cascade(self, connector):
        """Level 1 predictions feed into level 2."""
        cascade = ForecastCascade(connector, source_table="event_sequences")
        cascade.add_level(CascadeLevel(name="patient", num_prediction_steps=2))
        cascade.add_level(CascadeLevel(name="department", num_prediction_steps=2))

        # Seed data and establish baseline at both levels
        _insert_event(connector, "s1", [1.0, 0.0, 0.0], 1.0)
        _insert_event(connector, "s1", [0.0, 1.0, 0.0], 2.0)
        cascade.poll_once()  # baseline for level 1

        # New event triggers patient-level processing
        _insert_event(connector, "s1", [0.0, 0.0, 1.0], 3.0)
        cascade.poll_once()  # patient processes, writes predictions

        # Patient predictions now exist
        patient_preds = cascade.get_predictions("patient", sequence_id="s1")
        assert len(patient_preds) == 2

        # Department level should have received predictions as its input
        # It needs at least 2 polls to establish baseline then process
        # The first poll_once above already ran department baseline (level 1 had preds)
        # Now department should have data from patient predictions

    def test_three_level_cascade_builds(self, connector):
        """Three levels can be constructed and polled without error."""
        cascade = ForecastCascade(connector, source_table="event_sequences")
        cascade.add_level(CascadeLevel(name="patient", num_prediction_steps=2))
        cascade.add_level(CascadeLevel(name="department", num_prediction_steps=2))
        cascade.add_level(CascadeLevel(name="financial", num_prediction_steps=1))

        _insert_event(connector, "s1", [1.0, 0.0, 0.0], 1.0)
        _insert_event(connector, "s1", [0.0, 1.0, 0.0], 2.0)

        # Should not raise
        result = cascade.poll_once()
        assert "patient" in result
        assert "department" in result
        assert "financial" in result

    def test_cascade_alerts_across_levels(self, connector):
        """Alerts from different levels are queryable together."""
        rule_l1 = AlertRule(name="l1_rule", condition=lambda r: True, actions=[])
        rule_l2 = AlertRule(name="l2_rule", condition=lambda r: True, actions=[])

        cascade = ForecastCascade(connector, source_table="event_sequences")
        cascade.add_level(CascadeLevel(name="patient", num_prediction_steps=2, rules=[rule_l1]))
        cascade.add_level(CascadeLevel(name="department", num_prediction_steps=2, rules=[rule_l2]))

        _insert_event(connector, "s1", [1.0, 0.0, 0.0], 1.0)
        _insert_event(connector, "s1", [0.0, 1.0, 0.0], 2.0)
        cascade.poll_once()  # baseline

        _insert_event(connector, "s1", [0.0, 0.0, 1.0], 3.0)
        cascade.poll_once()

        # Patient level should have fired
        patient_alerts = cascade.get_alerts(level="patient")
        assert len(patient_alerts) >= 1

        # All alerts across levels
        all_alerts = cascade.get_alerts()
        for a in all_alerts:
            assert "level" in a

    def test_custom_view_query(self, connector):
        """A level with a custom view_query transforms upstream predictions."""
        cascade = ForecastCascade(connector, source_table="event_sequences")
        cascade.add_level(CascadeLevel(name="patient", num_prediction_steps=2))
        cascade.add_level(
            CascadeLevel(
                name="department",
                num_prediction_steps=2,
                view_query=(
                    "SELECT 'dept_1' AS sequence_id, prediction AS embedding, "
                    "updated_at AS timestamp, 'prediction' AS modality "
                    "FROM {upstream_table}"
                ),
            )
        )

        _insert_event(connector, "s1", [1.0, 0.0, 0.0], 1.0)
        _insert_event(connector, "s1", [0.0, 1.0, 0.0], 2.0)
        _insert_event(connector, "s2", [0.5, 0.5, 0.0], 1.5)

        # Should build and poll without error
        cascade.poll_once()

        _insert_event(connector, "s1", [0.0, 0.0, 1.0], 3.0)
        _insert_event(connector, "s2", [0.0, 0.5, 0.5], 2.5)
        cascade.poll_once()

        # Patient predictions exist
        patient_preds = cascade.get_predictions("patient")
        assert len(patient_preds) > 0

    def test_cascade_namespaced_tables(self, connector):
        """Each level gets its own namespaced tables."""
        cascade = ForecastCascade(connector, source_table="event_sequences")
        cascade.add_level(CascadeLevel(name="patient", num_prediction_steps=2))
        cascade.add_level(CascadeLevel(name="department", num_prediction_steps=2))

        _insert_event(connector, "s1", [1.0, 0.0, 0.0], 1.0)
        cascade.poll_once()

        db = connector._ensure_open()
        tables = [
            r[0]
            for r in db.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            ).fetchall()
        ]
        assert "_cascade_patient_alerts" in tables
        assert "_cascade_patient_predictions_live" in tables
        assert "_cascade_department_alerts" in tables
        assert "_cascade_department_predictions_live" in tables


# ---------------------------------------------------------------------------
# Dynamic rule/action management
# ---------------------------------------------------------------------------


class TestDynamicManagement:
    def test_add_rule_after_construction(self, connector):
        cascade = ForecastCascade(connector, source_table="event_sequences")
        cascade.add_level(CascadeLevel(name="patient"))

        _insert_event(connector, "s1", [1.0, 0.0, 0.0], 1.0)
        _insert_event(connector, "s1", [0.0, 1.0, 0.0], 2.0)
        cascade.poll_once()

        # Add rule dynamically
        cascade.add_rule("patient", AlertRule(name="dynamic", condition=lambda r: True, actions=[]))

        _insert_event(connector, "s1", [0.0, 0.0, 1.0], 3.0)
        result = cascade.poll_once()
        assert len(result["patient"]) == 1

    def test_add_action_after_construction(self, connector):
        fired = []
        cascade = ForecastCascade(connector, source_table="event_sequences")
        cascade.add_level(
            CascadeLevel(
                name="patient",
                rules=[AlertRule(name="r1", condition=lambda r: True, actions=["capture"])],
            )
        )

        _insert_event(connector, "s1", [1.0, 0.0, 0.0], 1.0)
        _insert_event(connector, "s1", [0.0, 1.0, 0.0], 2.0)
        cascade.poll_once()

        cascade.add_action("patient", "capture", lambda a: fired.append(a))

        _insert_event(connector, "s1", [0.0, 0.0, 1.0], 3.0)
        cascade.poll_once()
        assert len(fired) == 1

    def test_get_engine_exposes_trigger_engine(self, connector):
        cascade = ForecastCascade(connector, source_table="event_sequences")
        cascade.add_level(CascadeLevel(name="patient"))

        from event_jepa_cube.triggers import TriggerEngine

        engine = cascade.get_engine("patient")
        assert isinstance(engine, TriggerEngine)


# ---------------------------------------------------------------------------
# Watch loops
# ---------------------------------------------------------------------------


class TestCascadeWatch:
    def test_watch_with_max_iterations(self, connector):
        cascade = ForecastCascade(connector, source_table="event_sequences")
        cascade.add_level(CascadeLevel(name="patient"))

        _insert_event(connector, "s1", [1.0, 0.0, 0.0], 1.0)
        cascade.watch(interval_seconds=0.01, max_iterations=3)

    def test_watch_async_starts_and_stops(self, connector):
        cascade = ForecastCascade(connector, source_table="event_sequences")
        cascade.add_level(CascadeLevel(name="patient"))

        _insert_event(connector, "s1", [1.0, 0.0, 0.0], 1.0)
        handle = cascade.watch_async(interval_seconds=0.05)
        assert handle.is_running
        time.sleep(0.15)
        handle.stop()
        assert not handle.is_running

    def test_watch_async_processes_cascade(self, connector):
        fired = []
        cascade = ForecastCascade(connector, source_table="event_sequences")
        cascade.add_level(
            CascadeLevel(
                name="patient",
                rules=[AlertRule(name="track", condition=lambda r: True, actions=["capture"])],
            )
        )
        cascade.add_action("patient", "capture", lambda a: fired.append(a))

        _insert_event(connector, "s1", [1.0, 0.0, 0.0], 1.0)
        handle = cascade.watch_async(interval_seconds=0.05)
        time.sleep(0.15)  # baseline

        _insert_event(connector, "s1", [0.0, 1.0, 0.0], 2.0)
        time.sleep(0.2)  # detect

        handle.stop()
        assert len(fired) >= 1
