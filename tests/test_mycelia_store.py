"""Tests for MyceliaStore — Mycelia API vector store connector."""

from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from event_jepa_cube.code_ingestion import (
    build_code_symbol_payloads,
    build_embeddings_by_symbol_from_vocab,
    infer_code_language,
    load_node_vocab,
    sync_leio_query_cache_from_artifacts,
    sync_leio_query_cache_to_mycelia,
)
from event_jepa_cube.jcube_bridge import (
    BridgeConfig,
    build_node_payload,
    build_node_scalar_fields,
    describe_node,
    push_embeddings_to_mycelia,
)
from event_jepa_cube.mycelia_store import MyceliaError, MyceliaStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store():
    return MyceliaStore(
        base_url="https://test.mycelia.local",
        api_key="test-key",
        namespace="test-ns",
        vector_ingest_transport="http",
    )


def _mock_response(body: dict | list | None = None, status: int = 200):
    """Create a mock urllib response context manager."""
    resp = MagicMock()
    resp.status = status
    resp.read.return_value = json.dumps(body).encode() if body is not None else b""
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class _CaptureHandler(BaseHTTPRequestHandler):
    requests: list[tuple[str, str, dict | list | None]] = []

    def do_HEAD(self) -> None:  # noqa: N802
        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b""
        body = json.loads(raw.decode()) if raw else None
        self.__class__.requests.append((self.command, self.path, body))

        if self.path == "/v2/collections":
            payload = {"name": body["name"], "dimension": body["model"]["dimension"]}
        elif self.path.endswith("/vectors"):
            payload = {"inserted": len(body["vectors"])}
        elif self.path.startswith("/v2/search/"):
            payload = {"results": [{"id": "GHO-BRADESCO/ID_CD_INTERNACAO_123", "score": 0.99}]}
        else:
            payload = {}

        encoded = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return


class _VectorFlightHarness:
    def __init__(self) -> None:
        pa = pytest.importorskip("pyarrow")
        flight = pytest.importorskip("pyarrow.flight")
        self._pa = pa
        self._flight = flight
        self.tables: list[object] = []
        self.batch_counts: list[int] = []

        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        host, port = sock.getsockname()
        sock.close()
        self._location = f"grpc://{host}:{port}"

        harness = self

        class _Server(flight.FlightServerBase):
            def do_exchange(self, context, descriptor, reader, writer):  # type: ignore[no-untyped-def]
                assert descriptor.command == b"vectors:jcube_twin_v5" or descriptor.command.startswith(b"vectors:")
                batches = [chunk.data for chunk in reader if chunk.data is not None]
                table = pa.Table.from_batches(batches) if batches else pa.table({})
                harness.tables.append(table)
                harness.batch_counts.append(len(batches))
                ack_schema = pa.schema([("collection", pa.utf8()), ("inserted", pa.int64()), ("dimension", pa.int32())])
                ack_batch = pa.record_batch(
                    [
                        [descriptor.command.decode("utf-8").split(":", 1)[1]],
                        [table.num_rows],
                        [len(table.column("embedding")[0].as_py())],
                    ],
                    schema=ack_schema,
                )
                writer.begin(ack_schema)
                writer.write_batch(ack_batch)

        self._server = _Server(self._location)
        self._thread = threading.Thread(target=self._server.serve, daemon=True)

    @property
    def location(self) -> str:
        return self._location

    def start(self) -> _VectorFlightHarness:
        self._thread.start()
        return self

    def close(self) -> None:
        self._server.shutdown()
        self._thread.join(timeout=2)


# ---------------------------------------------------------------------------
# Headers
# ---------------------------------------------------------------------------


class TestHeaders:
    def test_headers_with_auth_and_namespace(self, store):
        h = store._headers()
        assert h["Authorization"] == "Bearer test-key"
        assert h["X-Namespace"] == "test-ns"
        assert h["Content-Type"] == "application/json"

    def test_headers_without_auth(self):
        s = MyceliaStore("https://x.local")
        h = s._headers()
        assert "Authorization" not in h
        assert "X-Namespace" not in h

    def test_vector_ingest_defaults_to_flight_first(self):
        s = MyceliaStore("https://x.local")
        assert s._vector_ingest_transport == "flight"


# ---------------------------------------------------------------------------
# Collection management
# ---------------------------------------------------------------------------


class TestCollections:
    @patch("urllib.request.urlopen")
    def test_collection_exists_true(self, mock_urlopen, store):
        mock_urlopen.return_value = _mock_response(status=200)
        assert store.collection_exists("my_col") is True

    @patch("urllib.request.urlopen")
    def test_collection_exists_false(self, mock_urlopen, store):
        import urllib.error

        mock_urlopen.side_effect = urllib.error.HTTPError("https://x", 404, "Not Found", {}, None)
        assert store.collection_exists("missing") is False

    @patch("urllib.request.urlopen")
    def test_ensure_collection_creates(self, mock_urlopen, store):
        # First call: HEAD → 404 (doesn't exist)
        import urllib.error

        head_err = urllib.error.HTTPError("https://x", 404, "Not Found", {}, None)
        create_resp = _mock_response({"name": "test", "dimension": 8, "status": "created"})

        mock_urlopen.side_effect = [head_err, create_resp]
        result = store.ensure_collection("test", dimension=8)
        assert result["name"] == "test"

    @patch("urllib.request.urlopen")
    def test_ensure_collection_already_exists(self, mock_urlopen, store):
        # HEAD → 200, GET → collection details
        head_resp = _mock_response(status=200)
        get_resp = _mock_response({"name": "test", "dimension": 8, "vector_count": 42})

        mock_urlopen.side_effect = [head_resp, get_resp]
        result = store.ensure_collection("test", dimension=8)
        assert result["vector_count"] == 42

    @patch("urllib.request.urlopen")
    def test_list_collections(self, mock_urlopen, store):
        mock_urlopen.return_value = _mock_response([{"name": "a"}, {"name": "b"}])
        result = store.list_collections()
        assert len(result) == 2

    @patch("urllib.request.urlopen")
    def test_delete_collection(self, mock_urlopen, store):
        mock_urlopen.return_value = _mock_response(status=204)
        store.delete_collection("my_col")
        mock_urlopen.assert_called_once()


# ---------------------------------------------------------------------------
# Vector storage
# ---------------------------------------------------------------------------


class TestVectors:
    @patch("urllib.request.urlopen")
    def test_store_vectors(self, mock_urlopen, store):
        mock_urlopen.return_value = _mock_response({"inserted": 2})
        result = store.store_vectors("col", {"v1": [1.0, 2.0], "v2": [3.0, 4.0]})
        assert result["inserted"] == 2

        # Verify the request body
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        body = json.loads(req.data)
        assert len(body["vectors"]) == 2
        assert all("id" in v and "embedding" in v for v in body["vectors"])

    @patch("urllib.request.urlopen")
    def test_store_representations(self, mock_urlopen, store):
        mock_urlopen.return_value = _mock_response({"inserted": 1})
        store.store_representations("col", {"r1": [0.1, 0.2, 0.3]})

        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data)
        assert body["vectors"][0]["filter_tag"] == "representation"

    @patch("urllib.request.urlopen")
    def test_store_vectors_with_scope_and_metadata(self, mock_urlopen, store):
        mock_urlopen.return_value = _mock_response({"inserted": 1})
        store.store_vectors(
            "col",
            {"v1": [1.0, 2.0]},
            filter_tag="representation",
            tenant_id="workspace",
            repo="mycelia-workspace",
            rev="43f5e23",
            metadata_by_id={"v1": {"path": "jcube/event_jepa_cube/materializer.py", "kind": "representation"}},
            relations_by_id={"v1": ["materializer", "jcube", "materializer"]},
            text_by_id={"v1": "materializer representation vector"},
            scalar_fields_by_id={"v1": {"source_db": "GHO-BRADESCO", "entity_type": "INTERNACAO"}},
        )

        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data)
        row = body["vectors"][0]
        assert row["tenant_id"] == "workspace"
        assert row["repo"] == "mycelia-workspace"
        assert row["rev"] == "43f5e23"
        assert row["metadata"]["path"] == "jcube/event_jepa_cube/materializer.py"
        assert row["relations"] == ["materializer", "jcube"]
        assert row["text_snippet"] == "materializer representation vector"
        assert row["source_db"] == "GHO-BRADESCO"
        assert row["entity_type"] == "INTERNACAO"

    def test_store_vectors_over_flight_preserves_rich_columns(self):
        harness = _VectorFlightHarness().start()
        try:
            store = MyceliaStore(
                base_url="https://test.mycelia.local",
                api_key="test-key",
                namespace="test-ns",
                flight_url=harness.location,
                vector_ingest_transport="flight",
            )
            call_options = store._flight_options()
            assert call_options.write_options.compression is None
            assert (b"authorization", b"Bearer test-key") in call_options.headers
            assert (b"x-mycelia-namespace", b"test-ns") in call_options.headers
            result = store.store_vectors(
                "jcube_twin_v5",
                {"v1": [1.0, 2.0], "v2": [3.0, 4.0]},
                filter_tag="graph_node",
                tenant_id="workspace",
                repo="mycelia-workspace",
                rev="43f5e23",
                metadata_by_id={"v1": {"path": "src/auth.rs", "kind": "function"}},
                relations_by_id={"v1": ["kind:code_symbol", "language:rust", "language:rust"]},
                text_by_id={"v1": "code symbol name=validate_bearer"},
                scalar_fields_by_id={"v1": {"source_db": "GHO-BRADESCO", "entity_type": "INTERNACAO"}},
            )
        finally:
            harness.close()

        assert result["collection"] == "jcube_twin_v5"
        assert result["inserted"] == 2
        table = harness.tables[0]
        assert table.num_rows == 2
        assert "tenant_id" in table.column_names
        assert "repo" in table.column_names
        assert "rev" in table.column_names
        assert "metadata" in table.column_names
        assert "relations" in table.column_names
        assert "text_snippet" in table.column_names
        assert "source_db" in table.column_names
        assert table.column("metadata")[0].as_py()["path"] == "src/auth.rs"
        assert table.column("relations")[0].as_py() == ["kind:code_symbol", "language:rust"]
        assert table.column("text_snippet")[0].as_py() == "code symbol name=validate_bearer"
        assert table.column("source_db")[0].as_py() == "GHO-BRADESCO"
        assert table.column("entity_type")[0].as_py() == "INTERNACAO"

    def test_store_vectors_over_flight_streams_in_batches(self):
        harness = _VectorFlightHarness().start()
        try:
            store = MyceliaStore(
                base_url="https://test.mycelia.local",
                api_key="test-key",
                namespace="test-ns",
                flight_url=harness.location,
                vector_ingest_transport="flight",
                vector_ingest_batch_size=1,
            )
            store.store_vectors(
                "jcube_twin_v5",
                {"v1": [1.0, 2.0], "v2": [3.0, 4.0]},
                tenant_id="workspace",
            )
        finally:
            harness.close()

        assert harness.batch_counts == [2]

    @patch("urllib.request.urlopen")
    def test_store_predictions(self, mock_urlopen, store):
        mock_urlopen.return_value = _mock_response({"inserted": 3})
        preds = {"seq1": [[1.0, 2.0], [3.0, 4.0]], "seq2": [[5.0, 6.0]]}
        store.store_predictions("col", preds)

        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data)
        ids = {v["id"] for v in body["vectors"]}
        assert "seq1_step_1" in ids
        assert "seq1_step_2" in ids
        assert "seq2_step_1" in ids
        assert all(v["filter_tag"] == "prediction" for v in body["vectors"])

    @patch("urllib.request.urlopen")
    def test_get_vectors(self, mock_urlopen, store):
        mock_urlopen.return_value = _mock_response([{"id": "v1", "embedding": [1.0, 2.0]}])
        result = store.get_vectors("col", ids=["v1"])
        assert result[0]["id"] == "v1"

    @patch("urllib.request.urlopen")
    def test_delete_vectors(self, mock_urlopen, store):
        mock_urlopen.return_value = _mock_response(status=204)
        store.delete_vectors("col", ["v1", "v2"])


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class TestSearch:
    def test_build_filter_expr(self):
        expr = MyceliaStore.build_filter_expr(
            'kind == "node"',
            tenant_id="workspace",
            repo="mycelia-workspace",
            rev="43f5e23",
            filter_tag="representation",
            source_db="GHO-BRADESCO",
            entity_type="INTERNACAO",
        )
        assert expr == (
            '(kind == "node") and tenant_id == "workspace" and repo == "mycelia-workspace" '
            'and rev == "43f5e23" and filter_tag == "representation" and source_db == "GHO-BRADESCO" '
            'and entity_type == "INTERNACAO"'
        )

    @patch("urllib.request.urlopen")
    def test_search_by_vector(self, mock_urlopen, store):
        mock_urlopen.return_value = _mock_response({"results": [{"id": "v1", "distance": 0.1, "score": 0.9}]})
        results = store.search_similar("col", vector=[1.0, 2.0])
        assert len(results) == 1
        assert results[0]["id"] == "v1"

    @patch("urllib.request.urlopen")
    def test_search_by_ids(self, mock_urlopen, store):
        mock_urlopen.return_value = _mock_response({"results": [{"id": "v2", "distance": 0.2}]})
        results = store.search_similar("col", ids=["v1"])
        assert results[0]["id"] == "v2"

    @patch("urllib.request.urlopen")
    def test_search_hybrid(self, mock_urlopen, store):
        mock_urlopen.return_value = _mock_response({"results": [{"id": "v1", "score": 0.85}]})
        results = store.search_hybrid(
            "col",
            query_text="test query",
            alpha=0.8,
            tenant_id="workspace",
            repo="mycelia-workspace",
            rev="43f5e23",
            filter_tag="representation",
            source_db="GHO-BRADESCO",
            entity_type="INTERNACAO",
        )
        assert len(results) == 1
        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data)
        assert body["filter"] == (
            'tenant_id == "workspace" and repo == "mycelia-workspace" and rev == "43f5e23" '
            'and filter_tag == "representation" and source_db == "GHO-BRADESCO" and entity_type == "INTERNACAO"'
        )

    @patch("urllib.request.urlopen")
    def test_search_rag(self, mock_urlopen, store):
        mock_urlopen.return_value = _mock_response({"chunks": [{"text": "answer", "score": 0.9}]})
        results = store.search_rag("col", query_text="what is X?", tenant_id="workspace")
        assert results[0]["text"] == "answer"
        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data)
        assert body["filter"] == 'tenant_id == "workspace"'


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


class TestEmbed:
    @patch("urllib.request.urlopen")
    def test_embed_with_model(self, mock_urlopen, store):
        mock_urlopen.return_value = _mock_response({"embeddings": [[0.1, 0.2, 0.3]]})
        result = store.embed([{"text": "hello"}], model="e5-base")
        assert len(result) == 1
        assert len(result[0]) == 3

    @patch("urllib.request.urlopen")
    def test_embed_with_collection(self, mock_urlopen, store):
        mock_urlopen.return_value = _mock_response({"embeddings": [[0.4, 0.5]]})
        result = store.embed([{"text": "hello"}], collection="my_col")
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Arrow IPC
# ---------------------------------------------------------------------------


class TestArrow:
    @pytest.fixture(autouse=True)
    def _skip_no_arrow(self):
        pytest.importorskip("pyarrow")

    @patch("urllib.request.urlopen")
    def test_to_arrow(self, mock_urlopen, store):
        import pyarrow as pa

        mock_urlopen.return_value = _mock_response(
            [{"id": "v1", "embedding": [1.0, 2.0]}, {"id": "v2", "embedding": [3.0, 4.0]}]
        )
        table = store.to_arrow("col")
        assert isinstance(table, pa.Table)
        assert table.num_rows == 2
        assert "id" in table.column_names
        assert "embedding" in table.column_names

    @patch("urllib.request.urlopen")
    def test_from_arrow(self, mock_urlopen, store):
        import pyarrow as pa

        mock_urlopen.return_value = _mock_response({"inserted": 2})
        table = pa.table(
            {
                "id": pa.array(["v1", "v2"], type=pa.string()),
                "embedding": pa.array([[1.0, 2.0], [3.0, 4.0]], type=pa.list_(pa.float32())),
            }
        )
        result = store.from_arrow("col", table)
        assert result["inserted"] == 2


# ---------------------------------------------------------------------------
# Pipeline sync
# ---------------------------------------------------------------------------


class TestPipelineSync:
    @patch("urllib.request.urlopen")
    def test_sync_pipeline_results(self, mock_urlopen, store):
        import urllib.error

        # Mock calls: HEAD 404 (create col), POST create, POST vectors
        # repeated for predictions collection
        head_404 = urllib.error.HTTPError("x", 404, "Not Found", {}, None)
        create_resp = _mock_response({"name": "reps", "dimension": 3})
        store_resp = _mock_response({"inserted": 2})
        create_resp2 = _mock_response({"name": "preds", "dimension": 3})
        store_resp2 = _mock_response({"inserted": 3})

        mock_urlopen.side_effect = [
            head_404,
            create_resp,
            store_resp,
            head_404,
            create_resp2,
            store_resp2,
        ]

        pipeline_result = {
            "representations": {"s1": [0.1, 0.2, 0.3], "s2": [0.4, 0.5, 0.6]},
            "predictions": {"s1": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], "s2": [[7.0, 8.0, 9.0]]},
        }

        result = store.sync_pipeline_results(
            pipeline_result,
            representations_collection="reps",
            predictions_collection="preds",
        )
        assert result["representations_stored"] == 2
        assert result["predictions_stored"] == 3

    @patch("urllib.request.urlopen")
    def test_sync_pipeline_results_propagates_scope_and_rich_rows(self, mock_urlopen, store):
        import urllib.error

        head_404 = urllib.error.HTTPError("x", 404, "Not Found", {}, None)
        create_resp = _mock_response({"name": "reps", "dimension": 3})
        store_resp = _mock_response({"inserted": 1})
        create_resp2 = _mock_response({"name": "preds", "dimension": 3})
        store_resp2 = _mock_response({"inserted": 2})

        mock_urlopen.side_effect = [
            head_404,
            create_resp,
            store_resp,
            head_404,
            create_resp2,
            store_resp2,
        ]

        pipeline_result = {
            "scope": {"tenant_id": "workspace", "repo": "mycelia-workspace", "rev": "43f5e23"},
            "representations": {"s1": [0.1, 0.2, 0.3]},
            "representation_metadata": {
                "s1": {
                    "path": "jcube/event_jepa_cube/materializer.py",
                    "kind": "representation",
                }
            },
            "representation_relations": {"s1": ["jcube", "materializer", "jcube"]},
            "representation_text": {"s1": "materialized event representation"},
            "predictions": {"s1": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]},
            "prediction_metadata": {
                "s1_step_1": {"sequence_id": "s1", "step": 1},
                "s1_step_2": {"sequence_id": "s1", "step": 2},
            },
            "prediction_relations": {"s1_step_1": ["prediction", "step:1"], "s1_step_2": ["prediction", "step:2"]},
            "prediction_text": {"s1_step_1": "prediction step 1", "s1_step_2": "prediction step 2"},
        }

        result = store.sync_pipeline_results(
            pipeline_result,
            representations_collection="reps",
            predictions_collection="preds",
        )

        assert result["representations_stored"] == 1
        assert result["predictions_stored"] == 2

        rep_req = mock_urlopen.call_args_list[2][0][0]
        rep_body = json.loads(rep_req.data)
        rep_row = rep_body["vectors"][0]
        assert rep_row["tenant_id"] == "workspace"
        assert rep_row["repo"] == "mycelia-workspace"
        assert rep_row["rev"] == "43f5e23"
        assert rep_row["metadata"]["path"] == "jcube/event_jepa_cube/materializer.py"
        assert rep_row["relations"] == ["jcube", "materializer"]
        assert rep_row["text_snippet"] == "materialized event representation"

        pred_req = mock_urlopen.call_args_list[5][0][0]
        pred_body = json.loads(pred_req.data)
        pred_rows = {row["id"]: row for row in pred_body["vectors"]}
        assert pred_rows["s1_step_1"]["tenant_id"] == "workspace"
        assert pred_rows["s1_step_1"]["metadata"]["step"] == 1
        assert pred_rows["s1_step_1"]["relations"] == ["prediction", "step:1"]
        assert pred_rows["s1_step_1"]["text_snippet"] == "prediction step 1"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrors:
    @patch("urllib.request.urlopen")
    def test_http_error_raises_mycelia_error(self, mock_urlopen, store):
        import io
        import urllib.error

        err_body = io.BytesIO(b'{"detail": "bad request"}')
        mock_urlopen.side_effect = urllib.error.HTTPError("https://x", 400, "Bad Request", {}, err_body)
        with pytest.raises(MyceliaError, match="HTTP 400"):
            store.get_collection("bad")

    @patch("urllib.request.urlopen")
    def test_connection_error_raises_mycelia_error(self, mock_urlopen, store):
        import urllib.error

        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
        with pytest.raises(MyceliaError, match="Connection error"):
            store.list_collections()

    def test_arrow_not_available(self, store):
        import event_jepa_cube.mycelia_store as mod

        original = mod._ARROW_AVAILABLE
        try:
            mod._ARROW_AVAILABLE = False
            with pytest.raises(ImportError, match="pyarrow"):
                store.to_arrow("col")
        finally:
            mod._ARROW_AVAILABLE = original


class TestJcubeBridge:
    def test_describe_node_handles_scoped_and_unscoped_ids(self):
        scoped = describe_node("GHO-BRADESCO/ID_CD_INTERNACAO_123")
        assert scoped.source_db == "GHO-BRADESCO"
        assert scoped.entity_type == "INTERNACAO"
        assert scoped.entity_id == "123"
        assert scoped.scoped is True
        assert scoped.node_kind == "instance"

        unscoped = describe_node("ID_CD_CID_A419")
        assert unscoped.source_db is None
        assert unscoped.entity_type == "CID"
        assert unscoped.entity_id == "A419"
        assert unscoped.scoped is False

    def test_build_node_payload_includes_metadata_relations_and_text(self):
        metadata, relations, text = build_node_payload(
            "GHO-BRADESCO/ID_CD_INTERNACAO_123",
            repo="jcube",
            rev="43f5e23",
        )

        assert metadata["source_db"] == "GHO-BRADESCO"
        assert metadata["entity_type"] == "INTERNACAO"
        assert metadata["entity_id"] == "123"
        assert metadata["repo"] == "jcube"
        assert metadata["rev"] == "43f5e23"
        assert "kind:graph_node" in relations
        assert "entity_type:internacao" in relations
        assert "source_db:gho-bradesco" in relations
        assert "graph node=GHO-BRADESCO/ID_CD_INTERNACAO_123" in text

    def test_build_node_scalar_fields_promotes_filterable_fields(self):
        fields = build_node_scalar_fields("GHO-BRADESCO/ID_CD_INTERNACAO_123")
        assert fields == {
            "source_db": "GHO-BRADESCO",
            "entity_type": "INTERNACAO",
            "node_kind": "instance",
        }

    def test_push_embeddings_to_mycelia_uses_flight_for_bulk_vectors(self):
        _CaptureHandler.requests = []
        server = ThreadingHTTPServer(("127.0.0.1", 0), _CaptureHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        harness = _VectorFlightHarness().start()

        try:
            store = MyceliaStore(
                f"http://127.0.0.1:{server.server_port}",
                api_key="secret",
                flight_url=harness.location,
                vector_ingest_transport="flight",
            )
            config = BridgeConfig(
                base_url=f"http://127.0.0.1:{server.server_port}",
                api_key="secret",
                namespace=None,
                collection="jcube_twin_v5",
                filter_tag="graph_node",
                batch_size=2,
                graph_path="/tmp/jcube_graph.parquet",
                weights_path="/tmp/node_emb.pt",
                tenant_id="workspace",
                repo="jcube",
                rev="43f5e23",
                key_types=frozenset({"INTERNACAO", "CID"}),
            )
            node_names = [
                "GHO-BRADESCO/ID_CD_INTERNACAO_123",
                "ID_CD_CID_A419",
                "ONTOLOGY_CLASS_X",
            ]
            embeddings = np.array(
                [
                    [0.1, 0.2, 0.3],
                    [0.3, 0.2, 0.1],
                    [0.9, 0.8, 0.7],
                ],
                dtype=np.float32,
            )

            result = push_embeddings_to_mycelia(store, config, node_names, embeddings)
        finally:
            server.shutdown()
            thread.join(timeout=2)
            server.server_close()
            harness.close()

        assert result["selected"] == 2
        assert result["inserted"] == 2
        assert result["batch_count"] == 1

        collection_post = _CaptureHandler.requests[0]
        search_post = _CaptureHandler.requests[1]

        assert collection_post[1] == "/v2/collections"
        assert collection_post[2]["name"] == "jcube_twin_v5"

        table = harness.tables[0]
        assert table.num_rows == 2
        assert table.column("tenant_id")[0].as_py() == "workspace"
        assert table.column("repo")[0].as_py() == "jcube"
        assert table.column("rev")[0].as_py() == "43f5e23"
        assert table.column("filter_tag")[0].as_py() == "graph_node"
        assert table.column("source_db")[0].as_py() == "GHO-BRADESCO"
        assert table.column("entity_type")[0].as_py() == "INTERNACAO"
        assert table.column("node_kind")[0].as_py() == "instance"
        assert table.column("metadata")[0].as_py()["entity_type"] == "INTERNACAO"
        assert "source_db:gho-bradesco" in table.column("relations")[0].as_py()
        assert table.column("text_snippet")[0].as_py()
        assert table.column("metadata")[1].as_py()["entity_type"] == "CID"

        assert search_post[1] == "/v2/search/jcube_twin_v5"
        assert search_post[2]["filter"] == (
            'tenant_id == "workspace" and repo == "jcube" and rev == "43f5e23" and filter_tag == "graph_node"'
        )


class TestCodeIngestion:
    def test_infer_code_language(self):
        assert infer_code_language("src/lib.rs") == "rust"
        assert infer_code_language("app/main.py") == "python"
        assert infer_code_language("web/page.tsx") == "tsx"
        assert infer_code_language("pkg/client.ts") == "typescript"
        assert infer_code_language("pkg/client.js") == "javascript"
        assert infer_code_language("README") == "unknown"

    def test_build_code_symbol_payloads(self):
        query_cache = {
            "revision": "3406e103330d0a47f50ff75a2c56e5424fad6d9d",
            "symbols": {
                "urn:one": {
                    "iri": "urn:one",
                    "name": "validate_bearer",
                    "qual_name": "auth::validate_bearer",
                    "kind": "function",
                    "path": "mycelia-gateway/src/auth.rs",
                    "line": 42,
                },
                "urn:two": {
                    "iri": "urn:two",
                    "name": "issue_token",
                    "qual_name": "auth::issue_token",
                    "kind": "function",
                    "path": "mycelia-api/mycelia/auth.py",
                    "line": 12,
                },
            },
            "callers_by_symbol": {"urn:one": [{"symbol": "urn:caller"}]},
            "callees_by_symbol": {"urn:one": [{"symbol": "urn:callee-a"}, {"symbol": "urn:callee-b"}]},
            "callsites_by_symbol": {"urn:one": [{"path": "main.rs", "line": 9}]},
        }
        embeddings = {
            "urn:one": [0.1, 0.2, 0.3],
            "missing": [0.9, 0.8, 0.7],
        }

        payload = build_code_symbol_payloads(
            query_cache,
            embeddings,
            tenant_id="workspace",
            repo="mycelia-workspace",
        )

        assert payload["scope"] == {
            "tenant_id": "workspace",
            "repo": "mycelia-workspace",
            "rev": "3406e103330d0a47f50ff75a2c56e5424fad6d9d",
        }
        assert payload["vectors"] == {"urn:one": [0.1, 0.2, 0.3]}
        assert payload["metadata_by_id"]["urn:one"]["language"] == "rust"
        assert payload["metadata_by_id"]["urn:one"]["caller_count"] == 1
        assert payload["metadata_by_id"]["urn:one"]["callee_count"] == 2
        assert payload["metadata_by_id"]["urn:one"]["callsite_count"] == 1
        assert payload["relations_by_id"]["urn:one"] == [
            "kind:code_symbol",
            "symbol_kind:function",
            "language:rust",
            "path:mycelia-gateway/src/auth.rs",
            "repo:mycelia-workspace",
        ]
        assert "name=validate_bearer" in payload["text_by_id"]["urn:one"]
        assert payload["scalar_fields_by_id"]["urn:one"] == {
            "path": "mycelia-gateway/src/auth.rs",
            "language": "rust",
            "symbol_kind": "function",
        }

    def test_load_node_vocab_and_build_embeddings_by_symbol_from_vocab(self, tmp_path):
        vocab_path = tmp_path / "node_vocab.json"
        vocab_path.write_text(json.dumps({"urn:one": 0, "urn:two": 1, "num_nodes": 2}))
        vocab = load_node_vocab(vocab_path)
        assert vocab == {"urn:one": 0, "urn:two": 1}

        embeddings = build_embeddings_by_symbol_from_vocab(
            vocab,
            np.array([[0.1, 0.2], [0.3, 0.4], [9.0, 9.0]], dtype=np.float32),
            symbol_ids=["urn:two", "missing"],
        )
        assert embeddings == {"urn:two": [0.30000001192092896, 0.4000000059604645]}

    def test_load_node_vocab_prefers_full_vocab_artifact(self, tmp_path):
        (tmp_path / "node_vocab.json").write_text(json.dumps({"num_nodes": 2, "num_predicates": 1}))
        (tmp_path / "node_vocab_sample.json").write_text(json.dumps({"urn:sample": 0}))
        (tmp_path / "node_vocab_full.json").write_text(json.dumps({"urn:one": 0, "urn:two": 1}))

        vocab = load_node_vocab(tmp_path)

        assert vocab == {"urn:one": 0, "urn:two": 1}

    @patch("urllib.request.urlopen")
    def test_store_code_symbols(self, mock_urlopen, store):
        mock_urlopen.return_value = _mock_response({"inserted": 1})
        store.store_code_symbols(
            "code_symbols",
            {"urn:one": [0.1, 0.2, 0.3]},
            tenant_id="workspace",
            repo="mycelia-workspace",
            rev="3406e103",
            metadata_by_id={"urn:one": {"path": "src/auth.rs"}},
            relations_by_id={"urn:one": ["kind:code_symbol", "language:rust"]},
            text_by_id={"urn:one": "code symbol name=validate_bearer"},
            scalar_fields_by_id={"urn:one": {"path": "src/auth.rs", "language": "rust", "symbol_kind": "function"}},
        )

        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data)
        row = body["vectors"][0]
        assert row["filter_tag"] == "code_symbol"
        assert row["path"] == "src/auth.rs"
        assert row["language"] == "rust"
        assert row["symbol_kind"] == "function"

    @patch("urllib.request.urlopen")
    def test_sync_leio_query_cache_to_mycelia(self, mock_urlopen, store):
        import urllib.error

        head_404 = urllib.error.HTTPError("x", 404, "Not Found", {}, None)
        create_resp = _mock_response({"name": "code_symbols", "dimension": 3})
        store_resp = _mock_response({"inserted": 1})
        mock_urlopen.side_effect = [head_404, create_resp, store_resp]

        result = sync_leio_query_cache_to_mycelia(
            store,
            "code_symbols",
            {
                "revision": "3406e103",
                "symbols": {
                    "urn:one": {
                        "iri": "urn:one",
                        "name": "validate_bearer",
                        "qual_name": "auth::validate_bearer",
                        "kind": "function",
                        "path": "mycelia-gateway/src/auth.rs",
                        "line": 42,
                    }
                },
            },
            {"urn:one": [0.1, 0.2, 0.3]},
            tenant_id="workspace",
            repo="mycelia-workspace",
        )

        assert result["stored"] == 1
        assert result["symbols"] == 1
        assert result["scope"] == {
            "tenant_id": "workspace",
            "repo": "mycelia-workspace",
            "rev": "3406e103",
        }

        req = mock_urlopen.call_args_list[2][0][0]
        body = json.loads(req.data)
        row = body["vectors"][0]
        assert row["filter_tag"] == "code_symbol"
        assert row["path"] == "mycelia-gateway/src/auth.rs"
        assert row["language"] == "rust"

    @patch("urllib.request.urlopen")
    def test_sync_leio_query_cache_from_artifacts(self, mock_urlopen, store, tmp_path):
        import urllib.error

        head_404 = urllib.error.HTTPError("x", 404, "Not Found", {}, None)
        create_resp = _mock_response({"name": "code_symbols", "dimension": 3})
        store_resp = _mock_response({"inserted": 1})
        mock_urlopen.side_effect = [head_404, create_resp, store_resp]

        node_vocab_path = tmp_path / "node_vocab.json"
        node_vocab_path.write_text(json.dumps({"urn:one": 0, "urn:other": 1}))

        result = sync_leio_query_cache_from_artifacts(
            store,
            "code_symbols",
            {
                "revision": "3406e103",
                "symbols": {
                    "urn:one": {
                        "iri": "urn:one",
                        "name": "validate_bearer",
                        "qual_name": "auth::validate_bearer",
                        "kind": "function",
                        "path": "mycelia-gateway/src/auth.rs",
                        "line": 42,
                    }
                },
            },
            node_vocab=node_vocab_path,
            embedding_matrix=np.array([[0.1, 0.2, 0.3], [9.0, 9.0, 9.0]], dtype=np.float32),
            tenant_id="workspace",
            repo="mycelia-workspace",
        )

        assert result["stored"] == 1
        assert result["symbols"] == 1
        req = mock_urlopen.call_args_list[2][0][0]
        body = json.loads(req.data)
        row = body["vectors"][0]
        assert row["id"] == "urn:one"
        assert row["filter_tag"] == "code_symbol"
