"""Tests for FlightTransfer — Arrow Flight streaming for Mycelia <-> DuckDB.

All Flight RPC calls are mocked. DuckDB tests use in-memory connections.
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

pa = pytest.importorskip("pyarrow")
flight_mod = pytest.importorskip("pyarrow.flight")
ipc_mod = pytest.importorskip("pyarrow.ipc")

duckdb = pytest.importorskip("duckdb")

from event_jepa_cube.flight_transfer import FlightTransfer, _derive_flight_url, _vector_schema  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_arrow_table(n: int = 5, dim: int = 4) -> pa.Table:
    """Create a sample Arrow table with id + embedding columns."""
    ids = [f"vec_{i}" for i in range(n)]
    embeddings = [[float(j + i * 0.1) for j in range(dim)] for i in range(n)]
    return pa.table(
        {
            "id": pa.array(ids, type=pa.string()),
            "embedding": pa.array(embeddings, type=pa.list_(pa.float32())),
        }
    )


def _mock_flight_info(table: pa.Table) -> MagicMock:
    """Create a mock FlightInfo with one endpoint."""
    ticket = MagicMock()
    endpoint = MagicMock()
    endpoint.ticket = ticket
    info = MagicMock()
    info.endpoints = [endpoint]
    return info


def _mock_reader(table: pa.Table) -> MagicMock:
    """Create a mock FlightStreamReader that returns a table."""
    reader = MagicMock()
    reader.read_all.return_value = table
    return reader


def _mock_writer() -> MagicMock:
    """Create a mock FlightStreamWriter."""
    writer = MagicMock()
    metadata_reader = MagicMock()
    return writer, metadata_reader


class _FakeConnector:
    """Minimal DuckDBConnector stand-in backed by a real in-memory DuckDB."""

    def __init__(self) -> None:
        self._conn = duckdb.connect(":memory:")

    def _ensure_open(self) -> duckdb.DuckDBPyConnection:
        return self._conn

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_table():
    return _make_arrow_table()


@pytest.fixture
def connector():
    c = _FakeConnector()
    yield c
    c.close()


# ---------------------------------------------------------------------------
# Unit tests: helpers
# ---------------------------------------------------------------------------


class TestDeriveFlightUrl:
    def test_https_url(self):
        assert _derive_flight_url("https://api.getjai.com") == "grpc://api.getjai.com:8815"

    def test_http_with_port(self):
        assert _derive_flight_url("http://localhost:8080") == "grpc://localhost:8815"

    def test_trailing_slash(self):
        assert _derive_flight_url("https://host.example.com/") == "grpc://host.example.com:8815"


class TestVectorSchema:
    def test_schema_fields(self):
        schema = _vector_schema()
        assert schema.field("id").type == pa.string()
        assert schema.field("embedding").type == pa.list_(pa.float32())


# ---------------------------------------------------------------------------
# FlightTransfer construction
# ---------------------------------------------------------------------------


class TestFlightTransferInit:
    @patch("event_jepa_cube.flight_transfer.flight.FlightClient")
    def test_init_creates_client(self, mock_client_cls):
        mock_client_cls.return_value = MagicMock()
        ft = FlightTransfer("https://api.getjai.com", api_key="secret")
        mock_client_cls.assert_called_once()
        assert ft._api_key == "secret"
        assert ft._flight_url == "grpc://api.getjai.com:8815"

    @patch("event_jepa_cube.flight_transfer.flight.FlightClient")
    def test_init_no_api_key(self, mock_client_cls):
        mock_client_cls.return_value = MagicMock()
        ft = FlightTransfer("https://api.getjai.com")
        assert ft._api_key is None

    @patch("event_jepa_cube.flight_transfer.flight.FlightClient")
    def test_init_connection_error(self, mock_client_cls):
        mock_client_cls.side_effect = Exception("Connection refused")
        with pytest.raises(ConnectionError, match="Failed to connect"):
            FlightTransfer("https://api.getjai.com")


# ---------------------------------------------------------------------------
# stream_to_duckdb
# ---------------------------------------------------------------------------


class TestStreamToDuckDB:
    @patch("event_jepa_cube.flight_transfer.flight.FlightClient")
    def test_stream_to_duckdb(self, mock_client_cls, sample_table, connector):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        info = _mock_flight_info(sample_table)
        mock_client.get_flight_info.return_value = info
        mock_client.do_get.return_value = _mock_reader(sample_table)

        ft = FlightTransfer("https://api.getjai.com")
        result = ft.stream_to_duckdb(connector, "my_vectors")

        assert result == "my_vectors"

        # Verify data landed in DuckDB
        conn = connector._ensure_open()
        rows = conn.execute('SELECT COUNT(*) FROM "my_vectors"').fetchone()
        assert rows[0] == 5

    @patch("event_jepa_cube.flight_transfer.flight.FlightClient")
    def test_stream_to_duckdb_custom_table_name(self, mock_client_cls, sample_table, connector):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        info = _mock_flight_info(sample_table)
        mock_client.get_flight_info.return_value = info
        mock_client.do_get.return_value = _mock_reader(sample_table)

        ft = FlightTransfer("https://api.getjai.com")
        result = ft.stream_to_duckdb(connector, "source_col", table_name="target_tbl")

        assert result == "target_tbl"
        conn = connector._ensure_open()
        rows = conn.execute('SELECT COUNT(*) FROM "target_tbl"').fetchone()
        assert rows[0] == 5

    @patch("event_jepa_cube.flight_transfer.flight.FlightClient")
    def test_stream_to_duckdb_no_endpoints(self, mock_client_cls, connector):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        info = MagicMock()
        info.endpoints = []
        mock_client.get_flight_info.return_value = info

        ft = FlightTransfer("https://api.getjai.com")
        with pytest.raises(ValueError, match="No endpoints"):
            ft.stream_to_duckdb(connector, "empty_col")

    @patch("event_jepa_cube.flight_transfer.flight.FlightClient")
    def test_stream_to_duckdb_connection_error(self, mock_client_cls, connector):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.get_flight_info.side_effect = Exception("Connection refused")

        ft = FlightTransfer("https://api.getjai.com")
        with pytest.raises(ConnectionError, match="Failed to get flight info"):
            ft.stream_to_duckdb(connector, "unreachable")


# ---------------------------------------------------------------------------
# stream_from_duckdb
# ---------------------------------------------------------------------------


class TestStreamFromDuckDB:
    @patch("event_jepa_cube.flight_transfer.flight.FlightClient")
    def test_stream_from_duckdb(self, mock_client_cls, sample_table, connector):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        writer, metadata = _mock_writer()
        mock_client.do_put.return_value = (writer, metadata)

        # Seed DuckDB with data
        conn = connector._ensure_open()
        conn.register("_seed", sample_table)
        conn.execute('CREATE TABLE "vectors" AS SELECT * FROM "_seed"')
        conn.unregister("_seed")

        ft = FlightTransfer("https://api.getjai.com")
        count = ft.stream_from_duckdb(connector, 'SELECT * FROM "vectors"', "target_col")

        assert count == 5
        writer.write_table.assert_called_once()
        writer.close.assert_called_once()

    @patch("event_jepa_cube.flight_transfer.flight.FlightClient")
    def test_stream_from_duckdb_column_rename(self, mock_client_cls, connector):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        writer, metadata = _mock_writer()
        mock_client.do_put.return_value = (writer, metadata)

        # Create table with non-standard column names
        conn = connector._ensure_open()
        conn.execute("CREATE TABLE src (vec_id VARCHAR, vec_emb FLOAT[4])")
        conn.execute("INSERT INTO src VALUES ('a', [1.0, 2.0, 3.0, 4.0])")

        ft = FlightTransfer("https://api.getjai.com")
        count = ft.stream_from_duckdb(
            connector,
            "SELECT vec_id, vec_emb FROM src",
            "target",
            id_column="vec_id",
            embedding_column="vec_emb",
        )
        assert count == 1

        # Verify the written table had canonical column names
        written_table = writer.write_table.call_args[0][0]
        assert "id" in written_table.column_names
        assert "embedding" in written_table.column_names


# ---------------------------------------------------------------------------
# stream_between_collections
# ---------------------------------------------------------------------------


class TestStreamBetweenCollections:
    @patch("event_jepa_cube.flight_transfer.flight.FlightClient")
    def test_stream_between(self, mock_client_cls, sample_table):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        info = _mock_flight_info(sample_table)
        mock_client.get_flight_info.return_value = info
        mock_client.do_get.return_value = _mock_reader(sample_table)

        writer, metadata = _mock_writer()
        mock_client.do_put.return_value = (writer, metadata)

        ft = FlightTransfer("https://api.getjai.com")
        count = ft.stream_between_collections("src_col", "dst_col")

        assert count == 5
        writer.write_table.assert_called_once()
        writer.close.assert_called_once()

    @patch("event_jepa_cube.flight_transfer.flight.FlightClient")
    def test_stream_between_no_source_endpoints(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        info = MagicMock()
        info.endpoints = []
        mock_client.get_flight_info.return_value = info

        ft = FlightTransfer("https://api.getjai.com")
        with pytest.raises(ValueError, match="No endpoints"):
            ft.stream_between_collections("missing", "dst")


# ---------------------------------------------------------------------------
# IPC export / import
# ---------------------------------------------------------------------------


class TestIPCExport:
    @patch("event_jepa_cube.flight_transfer.flight.FlightClient")
    def test_export_to_ipc(self, mock_client_cls, sample_table):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        info = _mock_flight_info(sample_table)
        mock_client.get_flight_info.return_value = info
        mock_client.do_get.return_value = _mock_reader(sample_table)

        ft = FlightTransfer("https://api.getjai.com")

        with tempfile.NamedTemporaryFile(suffix=".arrow", delete=False) as f:
            path = f.name

        try:
            count = ft.export_to_ipc("my_col", path)
            assert count == 5

            # Verify the file is readable
            with pa.OSFile(path, "rb") as source:
                reader = ipc_mod.open_file(source)
                result = reader.read_all()
            assert len(result) == 5
            assert "id" in result.column_names
            assert "embedding" in result.column_names
        finally:
            os.unlink(path)

    @patch("event_jepa_cube.flight_transfer.flight.FlightClient")
    def test_export_no_endpoints(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        info = MagicMock()
        info.endpoints = []
        mock_client.get_flight_info.return_value = info

        ft = FlightTransfer("https://api.getjai.com")
        with pytest.raises(ValueError, match="No endpoints"):
            ft.export_to_ipc("empty", "/tmp/out.arrow")


class TestIPCImport:
    @patch("event_jepa_cube.flight_transfer.flight.FlightClient")
    def test_import_from_ipc(self, mock_client_cls, sample_table):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        writer, metadata = _mock_writer()
        mock_client.do_put.return_value = (writer, metadata)

        # Write a temp IPC file
        with tempfile.NamedTemporaryFile(suffix=".arrow", delete=False) as f:
            path = f.name

        with pa.OSFile(path, "wb") as sink:
            w = ipc_mod.new_file(sink, sample_table.schema)
            w.write_table(sample_table)
            w.close()

        try:
            ft = FlightTransfer("https://api.getjai.com")
            count = ft.import_from_ipc(path, "target_col")

            assert count == 5
            writer.write_table.assert_called_once()
            writer.close.assert_called_once()
        finally:
            os.unlink(path)

    @patch("event_jepa_cube.flight_transfer.flight.FlightClient")
    def test_import_nonexistent_file(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        ft = FlightTransfer("https://api.getjai.com")
        with pytest.raises(FileNotFoundError):
            ft.import_from_ipc("/nonexistent/path/data.arrow", "col")


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    @patch("event_jepa_cube.flight_transfer.flight.FlightClient")
    def test_connection_refused_on_get_flight_info(self, mock_client_cls, connector):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.get_flight_info.side_effect = Exception("Connection refused")

        ft = FlightTransfer("https://api.getjai.com")
        with pytest.raises(ConnectionError, match="Failed to get flight info"):
            ft.stream_to_duckdb(connector, "col")

    @patch("event_jepa_cube.flight_transfer.flight.FlightClient")
    def test_missing_collection_no_endpoints(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        info = MagicMock()
        info.endpoints = []
        mock_client.get_flight_info.return_value = info

        ft = FlightTransfer("https://api.getjai.com")
        with pytest.raises(ValueError, match="No endpoints"):
            ft.export_to_ipc("nonexistent", "/tmp/out.arrow")
