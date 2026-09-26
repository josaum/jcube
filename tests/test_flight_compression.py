"""JCUBE standalone Flight compression and bounded-reader regressions."""

from __future__ import annotations

import os
import tempfile
from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pyarrow.ipc as ipc
import pytest

from event_jepa_cube.flight_compression import (
    PYTHON_CLIENT32_POSTDECODE,
    ROUTE_KEYS,
    FlightIpcDirection,
    FlightIpcObservation,
    FlightPolicyConfigError,
    FlightPostDecodeBudgetError,
    FlightRpcAction,
)
from event_jepa_cube.flight_transfer import FlightTransfer
from event_jepa_cube.mycelia_store import MyceliaStore


def _table() -> pa.Table:
    return pa.table(
        {
            "id": pa.array(["v1"], type=pa.string()),
            "embedding": pa.array([[1.0, 2.0]], type=pa.list_(pa.float32())),
        }
    )


def _reader(table: pa.Table) -> MagicMock:
    reader = MagicMock()
    reader.schema = table.schema
    reader.__iter__.return_value = iter([SimpleNamespace(data=batch) for batch in table.to_batches()])
    return reader


def _flight_info() -> MagicMock:
    endpoint = MagicMock()
    endpoint.ticket = MagicMock()
    info = MagicMock()
    info.endpoints = [endpoint]
    return info


def test_transfer_rejects_non_none_before_connect() -> None:
    with patch.dict(os.environ, {"MYCELIA_FLIGHT_IPC_COMPRESSION": "lz4"}, clear=True):
        with patch("event_jepa_cube.flight_transfer.flight.FlightClient") as client:
            with pytest.raises(FlightPolicyConfigError, match="compression forbidden"):
                FlightTransfer("https://api.getjai.com")
    client.assert_not_called()


def test_transfer_three_put_paths_use_explicit_none_and_emit_once() -> None:
    observed = []
    client = MagicMock()
    writer = MagicMock()
    client.do_put.return_value = (writer, MagicMock())
    with patch("event_jepa_cube.flight_transfer.flight.FlightClient", return_value=client):
        transfer = FlightTransfer("https://api.getjai.com", observation_hook=observed.append)

    table = _table()
    result = MagicMock()
    result.to_arrow_table.return_value = table
    connection = MagicMock()
    connection.execute.return_value = result
    connector = MagicMock()
    connector._ensure_open.return_value = connection
    assert transfer.stream_from_duckdb(connector, "select * from source", "target") == 1

    client.get_flight_info.return_value = _flight_info()
    source_reader = _reader(table)
    client.do_get.return_value = source_reader
    assert transfer.stream_between_collections("source", "target") == 1
    source_reader.read_all.assert_not_called()

    with tempfile.NamedTemporaryFile(suffix=".arrow") as handle:
        with pa.OSFile(handle.name, "wb") as sink:
            with ipc.new_file(sink, table.schema) as ipc_writer:
                ipc_writer.write_table(table)
        assert transfer.import_from_ipc(handle.name, "target") == 1

    assert client.do_put.call_count == 3
    for call in client.do_put.call_args_list:
        options = call.args[2]
        assert options.write_options.compression is None
    assert [(item.direction, item.action) for item in observed] == [
        (FlightIpcDirection.WRITE, FlightRpcAction.DO_PUT),
        (FlightIpcDirection.READ, FlightRpcAction.DO_GET),
        (FlightIpcDirection.WRITE, FlightRpcAction.DO_PUT),
        (FlightIpcDirection.WRITE, FlightRpcAction.DO_PUT),
    ]


def test_transfer_reader_rejects_one_byte_over_without_read_all() -> None:
    client = MagicMock()
    client.get_flight_info.return_value = _flight_info()
    oversized = SimpleNamespace(nbytes=PYTHON_CLIENT32_POSTDECODE.max_batch_bytes + 1)
    reader = MagicMock()
    reader.schema = pa.schema([])
    reader.__iter__.return_value = iter([SimpleNamespace(data=oversized)])
    client.do_get.return_value = reader
    with patch("event_jepa_cube.flight_transfer.flight.FlightClient", return_value=client):
        transfer = FlightTransfer("https://api.getjai.com")

    with pytest.raises(FlightPostDecodeBudgetError, match="^flight post-decode budget exceeded$"):
        transfer.stream_to_duckdb(MagicMock(), "source")
    reader.read_all.assert_not_called()


def test_transfer_has_no_raw_read_all_bypass() -> None:
    import event_jepa_cube.flight_transfer as module

    assert "reader.read_all()" not in open(module.__file__, encoding="utf-8").read()


def test_store_preserves_bearer_namespace_and_explicit_none_without_api_import() -> None:
    import event_jepa_cube.mycelia_store as module

    source = open(module.__file__, encoding="utf-8").read()
    assert "from mycelia.flight" not in source
    with patch.dict(os.environ, {}, clear=True):
        store = MyceliaStore(
            "https://api.getjai.com",
            api_key="secret",
            namespace="tenant-ns",
            vector_ingest_transport="flight",
        )
        options = store._flight_options()
    assert options.headers == [
        (b"authorization", b"Bearer secret"),
        (b"x-mycelia-namespace", b"tenant-ns"),
    ]
    assert options.write_options.compression is None


def test_store_exchange_writer_and_ack_reader_are_policy_wired_and_bounded() -> None:
    observed = []
    store = MyceliaStore(
        "https://api.getjai.com",
        api_key="secret",
        namespace="tenant-ns",
        vector_ingest_transport="flight",
        observation_hook=observed.append,
    )
    client = MagicMock()
    writer = MagicMock()
    ack_batch = pa.record_batch(
        {
            "collection": ["vectors"],
            "inserted": pa.array([1], type=pa.int64()),
            "dimension": pa.array([2], type=pa.int64()),
        }
    )
    ack_reader = _reader(pa.Table.from_batches([ack_batch]))
    client.do_exchange.return_value = (writer, ack_reader)
    store._flight_client = client

    result = store._store_vectors_flight("vectors", [{"id": "v1", "embedding": [1.0, 2.0], "tenant_id": "workspace"}])

    assert result == {"collection": "vectors", "inserted": 1, "dimension": 2}
    assert writer.begin.call_args.kwargs["options"].compression is None
    ack_reader.read_all.assert_not_called()
    options = client.do_exchange.call_args.kwargs["options"]
    assert (b"authorization", b"Bearer secret") in options.headers
    assert (b"x-mycelia-namespace", b"tenant-ns") in options.headers
    assert options.write_options.compression is None
    assert [(item.direction, item.action) for item in observed] == [
        (FlightIpcDirection.WRITE, FlightRpcAction.DO_EXCHANGE),
        (FlightIpcDirection.READ, FlightRpcAction.DO_EXCHANGE),
    ]
    assert all(set(item.as_dict()) == {field.name for field in fields(FlightIpcObservation)} for item in observed)


def test_store_rejects_non_none_before_any_network_io() -> None:
    with patch.dict(os.environ, {"MYCELIA_FLIGHT_IPC_COMPRESSION": "zstd"}, clear=True):
        with patch("event_jepa_cube.mycelia_store.flight.connect") as connect:
            with pytest.raises(FlightPolicyConfigError, match="compression forbidden"):
                MyceliaStore("https://api.getjai.com", vector_ingest_transport="flight")
    connect.assert_not_called()


def test_jcube_adapter_is_leaf_and_exact_route() -> None:
    import event_jepa_cube.flight_compression as compression

    assert ROUTE_KEYS == ("jcube.client.dynamic",)
    assert "mycelia.flight" not in open(compression.__file__, encoding="utf-8").read()
