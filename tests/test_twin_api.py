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


def test_entity_timeline_endpoint_single_entity(tmp_path: Path) -> None:
    db_path = tmp_path / "clinical.duckdb"
    conn = duckdb.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE internacoes (
            ID_CD_INTERNACAO VARCHAR,
            DH_EVENTO TIMESTAMP,
            VL_TOTAL DOUBLE
        )
        """
    )
    conn.execute(
        """
        INSERT INTO internacoes VALUES
        ('101', TIMESTAMP '2026-01-01 08:00:00', 100.0),
        ('101', TIMESTAMP '2026-01-02 09:00:00', 250.0),
        ('202', TIMESTAMP '2026-01-01 10:00:00', 50.0)
        """
    )
    conn.close()

    with TestClient(app) as client:
        resp = client.get(
            "/twin/entity/INTERNACAO/101",
            params={"db_path": str(db_path), "run_jepa": True, "num_prediction_steps": 2},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["entities_found"] == 1
        assert [t["entity_id"] for t in data["timelines"]] == ["101"]
        assert len(data["event_timestamps"]) == 2

        missing = client.get("/twin/entity/INTERNACAO/999", params={"db_path": str(db_path)})
        assert missing.status_code == 404


def test_load_embeddings_and_vector_endpoints_edge_cases(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    torch = pytest.importorskip("torch")
    from event_jepa_cube.twin_api import _load_embeddings

    graph_path = tmp_path / "graph.parquet"
    weights_path = tmp_path / "weights.pt"

    table = pa.table(
        {
            "subject_id": ["GHO/ID_CD_INTERNACAO_1", "GHO/ID_CD_INTERNACAO_2"],
            "object_id": ["GHO/ID_CD_PACIENTE_10", "GHO/ID_CD_FATURA_99"],
        }
    )
    pq.write_table(table, graph_path)

    weights = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    torch.save(weights, weights_path)

    _load_embeddings(graph_path=str(graph_path), weights_path=str(weights_path))

    with TestClient(app) as client:
        info = client.get("/twin/embedding/info")
        assert info.status_code == 200
        assert info.json() == {"num_nodes": 4, "dim": 4, "loaded": True}

        # k >= num_nodes (previously triggered np.argpartition out-of-bounds)
        sim_all = client.get("/twin/similar/GHO%2FID_CD_INTERNACAO_1", params={"k": 10})
        assert sim_all.status_code == 200
        assert len(sim_all.json()["matches"]) == 3

        # Single candidate after entity_type filter (previously triggered 0-D .squeeze() crash)
        sim_one = client.get(
            "/twin/similar/GHO%2FID_CD_INTERNACAO_1",
            params={"k": 5, "entity_type": "FATURA"},
        )
        assert sim_one.status_code == 200
        assert len(sim_one.json()["matches"]) == 1
        assert sim_one.json()["matches"][0]["node_id"] == "GHO/ID_CD_FATURA_99"

        # Anomalies when top >= matching nodes
        anom = client.get("/twin/anomalies/INTERNACAO", params={"top": 10})
        assert anom.status_code == 200
        assert anom.json()["total_nodes"] == 2
        assert len(anom.json()["anomalies"]) == 2

        # Vector search with single candidate and with k >= total nodes
        vec_one = client.post(
            "/twin/search-vector",
            json={"vector": [0.0, 0.0, 1.0, 0.0], "k": 5, "entity_type": "FATURA"},
        )
        assert vec_one.status_code == 200
        assert len(vec_one.json()["matches"]) == 1
        assert vec_one.json()["matches"][0]["node_id"] == "GHO/ID_CD_FATURA_99"
