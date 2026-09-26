"""Tests for the Digital Twin API runtime hardening."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

duckdb = pytest.importorskip("duckdb")

from event_jepa_cube.twin_api import app, configure_runtime, reset_runtime_state  # noqa: E402


def _build_test_db(tmp_path: Path) -> str:
    db_path = tmp_path / "sample.duckdb"
    conn = duckdb.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE events (
            id INTEGER,
            label VARCHAR,
            timestamp DOUBLE
        )
        """
    )
    conn.execute(
        """
        INSERT INTO events VALUES
        (1, 'alpha', 1.0),
        (2, 'beta', 2.0)
        """
    )
    conn.close()
    return str(db_path)


@pytest.fixture(autouse=True)
def _reset_runtime() -> None:
    configure_runtime(
        mode="local",
        allow_connect=True,
        require_explicit_db=False,
        allowed_db_roots=[],
        reset_state=True,
    )
    yield
    reset_runtime_state()


def test_query_endpoint_allows_read_only_select(tmp_path: Path) -> None:
    db_path = _build_test_db(tmp_path)

    with TestClient(app) as client:
        connect = client.post("/twin/connect", json={"db_path": db_path, "profile_columns": False})
        assert connect.status_code == 200

        response = client.post(
            "/twin/query",
            json={"sql": "SELECT id, label FROM events ORDER BY id", "limit": 10},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["row_count"] == 2
        assert payload["rows"][0] == {"id": 1, "label": "alpha"}


def test_connect_rejects_writable_mode(tmp_path: Path) -> None:
    db_path = _build_test_db(tmp_path)

    with TestClient(app) as client:
        response = client.post("/twin/connect", json={"db_path": db_path, "read_only": False})
        assert response.status_code == 400
        assert "Writable database connections are disabled" in response.text


def test_query_endpoint_rejects_write_sql(tmp_path: Path) -> None:
    db_path = _build_test_db(tmp_path)

    with TestClient(app) as client:
        connect = client.post("/twin/connect", json={"db_path": db_path, "profile_columns": False})
        assert connect.status_code == 200

        response = client.post("/twin/query", json={"sql": "DROP TABLE events"})
        assert response.status_code == 400
        assert "read-only" in response.text or "write or schema-changing" in response.text


def test_server_mode_disables_connect_and_requires_explicit_db(tmp_path: Path) -> None:
    db_path = _build_test_db(tmp_path)

    with TestClient(app) as client:
        connect = client.post("/twin/connect", json={"db_path": db_path, "profile_columns": False})
        assert connect.status_code == 200

        configure_runtime(
            mode="server",
            allow_connect=False,
            require_explicit_db=True,
            reset_state=False,
        )

        missing_db = client.get("/twin/snapshot")
        assert missing_db.status_code == 400
        assert "db_path is required in server mode" in missing_db.text

        explicit_db = client.get("/twin/snapshot", params={"db_path": db_path})
        assert explicit_db.status_code == 200

        blocked_connect = client.post("/twin/connect", json={"db_path": db_path})
        assert blocked_connect.status_code == 403
