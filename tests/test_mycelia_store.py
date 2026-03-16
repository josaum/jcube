"""Tests for MyceliaStore — Mycelia API vector store connector.

All HTTP calls are mocked via ``unittest.mock.patch`` so no live
Mycelia instance is required.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

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
    )


def _mock_response(body: dict | list | None = None, status: int = 200):
    """Create a mock urllib response context manager."""
    resp = MagicMock()
    resp.status = status
    resp.read.return_value = json.dumps(body).encode() if body is not None else b""
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    return resp


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
        results = store.search_hybrid("col", query_text="test query", alpha=0.8)
        assert len(results) == 1

    @patch("urllib.request.urlopen")
    def test_search_rag(self, mock_urlopen, store):
        mock_urlopen.return_value = _mock_response({"chunks": [{"text": "answer", "score": 0.9}]})
        results = store.search_rag("col", query_text="what is X?")
        assert results[0]["text"] == "answer"


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
