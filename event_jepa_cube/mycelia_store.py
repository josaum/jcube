"""Mycelia API connector for embedding storage and similarity search.

Bridges jcube's pipeline with the Mycelia vector database (Milvus backend),
enabling persistent embedding storage, similarity search across sequences,
and efficient Arrow IPC data transfer to/from DuckDB.

The Mycelia API provides:
- Collection management with auto-embedding and schema inference
- Dense, hybrid, and RAG-optimized similarity search
- Array-of-Structs storage for multi-vector entities
- Pre-trained and fine-tuned encoders (SIGReg)

Zero required dependencies for REST-only management/search paths (uses ``urllib``
from stdlib). ``pyarrow`` is required for Flight-first vector data operations.

Example::

    store = MyceliaStore("https://api.getjai.com", api_key="...")
    store.ensure_collection("patient_embeddings", dimension=768)
    store.store_representations("patient_embeddings", {"p1": [0.1, ...], "p2": [0.2, ...]})
    results = store.search_similar("patient_embeddings", query_vector=[0.1, ...], limit=5)
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from types import TracebackType
from typing import Any, Callable, cast
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

try:
    import pyarrow as pa
    import pyarrow.flight as flight

    _ARROW_AVAILABLE = True
except ImportError:
    _ARROW_AVAILABLE = False
    flight = None  # type: ignore[assignment]


def _require_arrow() -> None:
    if not _ARROW_AVAILABLE:
        raise ImportError("pyarrow is required for Arrow data transfer. Install with: pip install pyarrow>=23.0.1")


def _derive_flight_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    host = parsed.hostname or "localhost"
    port = os.getenv("MYCELIA_FLIGHT_PORT", "8815")
    return f"grpc://{host}:{port}"


def _resolve_flight_auth_header(api_key: str | None) -> str | None:
    explicit = os.getenv("MYCELIA_FLIGHT_AUTHORIZATION", "").strip()
    if explicit:
        return explicit

    bearer = os.getenv("MYCELIA_FLIGHT_BEARER_TOKEN", "").strip()
    if bearer:
        return bearer if bearer.lower().startswith("bearer ") else f"Bearer {bearer}"

    if api_key:
        return api_key if api_key.lower().startswith("bearer ") else f"Bearer {api_key}"

    legacy_secret = os.getenv("MYCELIA_FLIGHT_SECRET", "").strip()
    if legacy_secret:
        return f"Bearer {legacy_secret}"
    return None


def _local_flight_call_options(
    timeout: float | None,
    api_key: str | None,
    namespace: str | None,
    write_options: Any,
) -> Any:
    headers: list[tuple[bytes, bytes]] = []
    authorization = _resolve_flight_auth_header(api_key)
    if authorization:
        headers.append((b"authorization", authorization.encode()))
    if namespace:
        headers.append((b"x-mycelia-namespace", namespace.encode()))
    return flight.FlightCallOptions(timeout=timeout, headers=headers or None, write_options=write_options)


def _coerce_arrow_scalar(column: Any, index: int) -> Any:
    value = column[index].as_py()
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value
    return value


def _infer_arrow_array(values: list[Any]) -> Any:
    if all(value is None or isinstance(value, bool) for value in values):
        return pa.array(values, type=pa.bool_())
    if all(value is None or isinstance(value, int) for value in values):
        return pa.array(values, type=pa.int64())
    if all(value is None or isinstance(value, (int, float)) for value in values):
        return pa.array(values, type=pa.float64())
    return pa.array(values, type=pa.string())


def _dedupe_strings(values: list[str] | None) -> list[str]:
    if not values:
        return []
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def _escape_filter_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _expect_dict(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MyceliaError(f"{context} returned {type(value).__name__}, expected dict")
    return cast(dict[str, Any], value)


def _expect_list(value: Any, context: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise MyceliaError(f"{context} returned {type(value).__name__}, expected list[dict]")
    return cast(list[dict[str, Any]], value)


def _expect_vector_list(value: Any, context: str) -> list[list[float]]:
    if not isinstance(value, list):
        raise MyceliaError(f"{context} returned {type(value).__name__}, expected list")
    return cast(list[list[float]], value)


class MyceliaStore:
    """Client for the Mycelia API (Milvus-backed vector store).

    Stores and retrieves embedding vectors with Flight-first bulk transport and
    Arrow IPC integration for DuckDB.

    Args:
        base_url: Mycelia API base URL (e.g. ``"https://api.getjai.com"``).
        api_key: API key or Bearer token for authentication.
        namespace: Optional namespace scope for multi-tenant deployments.
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        namespace: str | None = None,
        timeout: int = 30,
        *,
        flight_url: str | None = None,
        vector_ingest_transport: str | None = None,
        vector_ingest_batch_size: int | None = None,
        compression_config: Any | None = None,
        observation_hook: Callable[[Any], None] | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._namespace = namespace
        self._timeout = timeout
        self._flight_url = flight_url or os.getenv("MYCELIA_FLIGHT_URL") or _derive_flight_url(self._base_url)
        transport = (vector_ingest_transport or os.getenv("MYCELIA_VECTOR_INGEST_TRANSPORT") or "").strip().lower()
        if not transport:
            transport = "flight"
        if transport not in {"flight", "http"}:
            raise ValueError(f"unsupported vector_ingest_transport: {transport}")
        self._vector_ingest_transport = transport
        self._observation_hook = observation_hook
        self._flight_compression = None
        if transport == "flight":
            _require_arrow()
            from event_jepa_cube.flight_compression import FlightCompressionConfig

            self._flight_compression = compression_config or FlightCompressionConfig.from_env()
        self._vector_ingest_batch_size = max(
            1,
            int(vector_ingest_batch_size or os.getenv("MYCELIA_VECTOR_INGEST_BATCH_SIZE") or 4096),
        )
        self._flight_client: Any | None = None

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        if self._namespace:
            headers["X-Namespace"] = self._namespace
        return headers

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any] | list[Any] | None:
        """Make an HTTP request to the Mycelia API."""
        url = f"{self._base_url}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(url, data=data, headers=self._headers(), method=method)

        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                if resp.status == 204:
                    return None
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as e:
            body_text = e.read().decode("utf-8", errors="replace") if e.fp else ""
            raise MyceliaError(f"HTTP {e.code} on {method} {path}: {body_text}") from e
        except urllib.error.URLError as e:
            raise MyceliaError(f"Connection error on {method} {path}: {e.reason}") from e

    def _get(self, path: str) -> Any:
        return self._request("GET", path)

    def _post(self, path: str, body: dict[str, Any] | None = None) -> Any:
        return self._request("POST", path, body)

    def _delete(self, path: str, body: dict[str, Any] | None = None) -> Any:
        return self._request("DELETE", path, body)

    def _flight_options(self) -> Any:
        _require_arrow()
        if self._flight_compression is None:
            raise MyceliaError("Flight transport is not configured")
        return _local_flight_call_options(
            float(self._timeout),
            self._api_key,
            self._namespace,
            self._flight_compression.options_for("jcube.client.dynamic"),
        )

    def _get_flight_client(self) -> Any:
        _require_arrow()
        if self._flight_client is None:
            try:
                self._flight_client = flight.connect(self._flight_url)
            except Exception as exc:  # pragma: no cover - depends on runtime connectivity
                raise MyceliaError(f"Failed to connect to Flight endpoint {self._flight_url}: {exc}") from exc
        return self._flight_client

    def close(self) -> None:
        if self._flight_client is not None:
            self._flight_client.close()
            self._flight_client = None

    def __enter__(self) -> MyceliaStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def _collection_schema(self, name: str) -> dict[str, Any]:
        return _expect_dict(self._get(f"/v2/collections/{name}/schema"), "_collection_schema")

    def _build_vector_rows(
        self,
        vectors: dict[str, list[float]],
        *,
        filter_tag: str | None = None,
        tenant_id: str | None = None,
        repo: str | None = None,
        rev: str | None = None,
        metadata_by_id: dict[str, dict[str, Any]] | None = None,
        relations_by_id: dict[str, list[str]] | None = None,
        text_by_id: dict[str, str] | None = None,
        scalar_fields_by_id: dict[str, dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        metadata_by_id = metadata_by_id or {}
        relations_by_id = relations_by_id or {}
        text_by_id = text_by_id or {}
        scalar_fields_by_id = scalar_fields_by_id or {}

        payload: list[dict[str, Any]] = []
        for vid, emb in vectors.items():
            row: dict[str, Any] = {"id": vid, "embedding": emb}
            if filter_tag:
                row["filter_tag"] = filter_tag
            if tenant_id:
                row["tenant_id"] = tenant_id
            if repo:
                row["repo"] = repo
            if rev:
                row["rev"] = rev

            metadata = metadata_by_id.get(vid)
            if metadata:
                row["metadata"] = metadata

            relations = _dedupe_strings(relations_by_id.get(vid))
            if relations:
                row["relations"] = relations

            text_snippet = text_by_id.get(vid)
            if text_snippet:
                row["text_snippet"] = text_snippet

            scalar_fields = scalar_fields_by_id.get(vid)
            if scalar_fields:
                for key, value in scalar_fields.items():
                    if (
                        value is None
                        or key in row
                        or key
                        in {
                            "id",
                            "embedding",
                            "metadata",
                            "relations",
                            "text_snippet",
                        }
                    ):
                        continue
                    row[key] = value

            payload.append(row)
        return payload

    def _store_vectors_http(self, collection: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        return _expect_dict(
            self._post(f"/v2/collections/{collection}/vectors", {"vectors": rows}),
            "_store_vectors_http",
        )

    def _vector_schema_from_rows(self, rows: list[dict[str, Any]]) -> tuple[Any, list[str]]:
        _require_arrow()
        optional_names = sorted({key for row in rows for key in row if key not in {"id", "embedding"}})
        fields = [
            pa.field("id", pa.string()),
            pa.field("embedding", pa.list_(pa.float32())),
        ]
        for name in optional_names:
            values = [row.get(name) for row in rows]
            if name == "relations":
                dtype = pa.list_(pa.string())
            elif name in {"filter_tag", "tenant_id", "repo", "rev", "text_snippet"}:
                dtype = pa.string()
            elif name == "metadata":
                dtype = pa.array(values).type
            else:
                dtype = _infer_arrow_array(values).type
            fields.append(pa.field(name, dtype))
        return pa.schema(fields), optional_names

    def _table_from_vector_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        schema: Any | None = None,
        optional_names: list[str] | None = None,
    ) -> Any:
        _require_arrow()
        if schema is None or optional_names is None:
            schema, optional_names = self._vector_schema_from_rows(rows)

        ids = [str(row["id"]) for row in rows]
        embeddings = [list(row["embedding"]) for row in rows]

        columns: dict[str, Any] = {
            "id": pa.array(ids, type=schema.field("id").type),
            "embedding": pa.array(embeddings, type=schema.field("embedding").type),
        }

        for name in optional_names:
            values = [row.get(name) for row in rows]
            columns[name] = pa.array(values, type=schema.field(name).type)

        return pa.table(columns, schema=schema)

    def _store_vectors_flight(self, collection: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        _require_arrow()
        from event_jepa_cube.flight_compression import (
            FlightIpcDirection,
            FlightRpcAction,
            collect_flight_table,
            logical_nbytes,
            observe_flight_operation,
        )

        if self._flight_compression is None:
            raise MyceliaError("Flight transport is not configured")
        route = "jcube.client.dynamic"
        client = self._get_flight_client()
        descriptor = flight.FlightDescriptor.for_command(f"vectors:{collection}".encode())
        schema, optional_names = self._vector_schema_from_rows(rows)
        with observe_flight_operation(
            route=route,
            codec=self._flight_compression.codec_for(route),
            direction=FlightIpcDirection.WRITE,
            action=FlightRpcAction.DO_EXCHANGE,
            hook=self._observation_hook,
        ) as write_observation:
            writer, reader = client.do_exchange(descriptor, options=self._flight_options())
            writer.begin(schema, options=self._flight_compression.options_for(route))
            for start in range(0, len(rows), self._vector_ingest_batch_size):
                batch_rows = rows[start : start + self._vector_ingest_batch_size]
                table = self._table_from_vector_rows(
                    batch_rows,
                    schema=schema,
                    optional_names=optional_names,
                )
                write_observation.add_logical_bytes(logical_nbytes(table))
                writer.write_table(table)
            writer.done_writing()
        try:
            with observe_flight_operation(
                route=route,
                codec=self._flight_compression.codec_for(route),
                direction=FlightIpcDirection.READ,
                action=FlightRpcAction.DO_EXCHANGE,
                hook=self._observation_hook,
            ) as read_observation:
                ack = collect_flight_table(reader)
                read_observation.add_logical_bytes(logical_nbytes(ack))
        finally:
            writer.close()
        if ack.num_rows == 0:
            return {"collection": collection, "inserted": 0, "dimension": 0}
        return {
            "collection": _coerce_arrow_scalar(ack.column("collection"), 0) or collection,
            "inserted": int(_coerce_arrow_scalar(ack.column("inserted"), 0) or 0),
            "dimension": int(_coerce_arrow_scalar(ack.column("dimension"), 0) or 0),
        }

    def _head(self, path: str) -> bool:
        """HEAD request, returns True if 2xx."""
        url = f"{self._base_url}{path}"
        req = urllib.request.Request(url, headers=self._headers(), method="HEAD")
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return bool(200 <= resp.status < 300)
        except urllib.error.HTTPError:
            return False
        except urllib.error.URLError:
            return False

    @staticmethod
    def build_filter_expr(
        filter_expr: str | None = None,
        *,
        tenant_id: str | None = None,
        repo: str | None = None,
        rev: str | None = None,
        filter_tag: str | None = None,
        source_db: str | None = None,
        entity_type: str | None = None,
    ) -> str | None:
        """Build a canonical Mycelia/LEIO filter expression.

        The returned expression targets top-level scalar fields that match the
        LEIO/Milvus operational scope: ``tenant_id``, ``repo``, ``rev``, and
        ``filter_tag``.
        """

        clauses: list[str] = []
        if filter_expr:
            clauses.append(f"({filter_expr})")

        for key, value in (
            ("tenant_id", tenant_id),
            ("repo", repo),
            ("rev", rev),
            ("filter_tag", filter_tag),
            ("source_db", source_db),
            ("entity_type", entity_type),
        ):
            if value:
                clauses.append(f'{key} == "{_escape_filter_value(value)}"')

        if not clauses:
            return None
        return " and ".join(clauses)

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def ensure_collection(
        self,
        name: str,
        dimension: int,
        modality: str = "tabular",
        model: str | None = None,
    ) -> dict[str, Any]:
        """Create a collection if it doesn't already exist.

        Args:
            name: Collection name (2-224 chars).
            dimension: Embedding dimension.
            modality: One of ``"text"``, ``"image"``, ``"tabular"``,
                ``"multimodal"``, ``"hybrid"``.
            model: Optional encoder model name.

        Returns:
            Collection details dict.
        """
        if self.collection_exists(name):
            return self.get_collection(name)

        body: dict[str, Any] = {
            "name": name,
            "modality": modality,
            "model": {"dimension": dimension},
        }
        if model:
            body["model"] = model if isinstance(model, dict) else {"name": model, "dimension": dimension}
        result = self._post("/v2/collections", body)
        logger.info("Created Mycelia collection %r (dim=%d)", name, dimension)
        return _expect_dict(result, "ensure_collection")

    def collection_exists(self, name: str) -> bool:
        """Check if a collection exists."""
        return self._head(f"/v2/collections/{name}")

    def get_collection(self, name: str) -> dict[str, Any]:
        """Get collection details."""
        return _expect_dict(self._get(f"/v2/collections/{name}"), "get_collection")

    def list_collections(self) -> list[dict[str, Any]]:
        """List all collections."""
        return _expect_list(self._get("/v2/collections"), "list_collections")

    def delete_collection(self, name: str) -> None:
        """Delete a collection and all its data."""
        self._delete(f"/v2/collections/{name}")
        logger.info("Deleted Mycelia collection %r", name)

    # ------------------------------------------------------------------
    # Vector storage
    # ------------------------------------------------------------------

    def store_vectors(
        self,
        collection: str,
        vectors: dict[str, list[float]],
        filter_tag: str | None = None,
        *,
        tenant_id: str | None = None,
        repo: str | None = None,
        rev: str | None = None,
        metadata_by_id: dict[str, dict[str, Any]] | None = None,
        relations_by_id: dict[str, list[str]] | None = None,
        text_by_id: dict[str, str] | None = None,
        scalar_fields_by_id: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Insert pre-computed vectors into a collection.

        Args:
            collection: Collection name.
            vectors: Mapping of ID to embedding vector.
            filter_tag: Optional tag for filtering.
            tenant_id: Optional top-level tenant scope field.
            repo: Optional top-level repository scope field.
            rev: Optional top-level revision scope field.
            metadata_by_id: Optional per-vector metadata payloads.
            relations_by_id: Optional per-vector relation/tag arrays.
            text_by_id: Optional per-vector text snippet payload.

        Returns:
            Ingestion result.
        """
        rows = self._build_vector_rows(
            vectors,
            filter_tag=filter_tag,
            tenant_id=tenant_id,
            repo=repo,
            rev=rev,
            metadata_by_id=metadata_by_id,
            relations_by_id=relations_by_id,
            text_by_id=text_by_id,
            scalar_fields_by_id=scalar_fields_by_id,
        )
        if self._vector_ingest_transport == "flight":
            return self._store_vectors_flight(collection, rows)
        return self._store_vectors_http(collection, rows)

    def store_representations(
        self,
        collection: str,
        representations: dict[str, list[float]],
        *,
        tenant_id: str | None = None,
        repo: str | None = None,
        rev: str | None = None,
        metadata_by_id: dict[str, dict[str, Any]] | None = None,
        relations_by_id: dict[str, list[str]] | None = None,
        text_by_id: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Store pipeline representations as vectors.

        Convenience wrapper that tags vectors with ``filter_tag="representation"``.
        """
        return self.store_vectors(
            collection,
            representations,
            filter_tag="representation",
            tenant_id=tenant_id,
            repo=repo,
            rev=rev,
            metadata_by_id=metadata_by_id,
            relations_by_id=relations_by_id,
            text_by_id=text_by_id,
        )

    def store_predictions(
        self,
        collection: str,
        predictions: dict[str, list[list[float]]],
        *,
        tenant_id: str | None = None,
        repo: str | None = None,
        rev: str | None = None,
        metadata_by_id: dict[str, dict[str, Any]] | None = None,
        relations_by_id: dict[str, list[str]] | None = None,
        text_by_id: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Store pipeline predictions as vectors.

        Each prediction step is stored as ``{sequence_id}_step_{n}``.
        """
        flat: dict[str, list[float]] = {}
        for sid, steps in predictions.items():
            for step_num, pred in enumerate(steps, start=1):
                flat[f"{sid}_step_{step_num}"] = pred
        return self.store_vectors(
            collection,
            flat,
            filter_tag="prediction",
            tenant_id=tenant_id,
            repo=repo,
            rev=rev,
            metadata_by_id=metadata_by_id,
            relations_by_id=relations_by_id,
            text_by_id=text_by_id,
        )

    def store_code_symbols(
        self,
        collection: str,
        vectors: dict[str, list[float]],
        *,
        tenant_id: str | None = None,
        repo: str | None = None,
        rev: str | None = None,
        metadata_by_id: dict[str, dict[str, Any]] | None = None,
        relations_by_id: dict[str, list[str]] | None = None,
        text_by_id: dict[str, str] | None = None,
        scalar_fields_by_id: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Store LEIO-derived code symbol vectors."""
        return self.store_vectors(
            collection,
            vectors,
            filter_tag="code_symbol",
            tenant_id=tenant_id,
            repo=repo,
            rev=rev,
            metadata_by_id=metadata_by_id,
            relations_by_id=relations_by_id,
            text_by_id=text_by_id,
            scalar_fields_by_id=scalar_fields_by_id,
        )

    def get_vectors(
        self,
        collection: str,
        ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve vectors by ID.

        Args:
            collection: Collection name.
            ids: Vector IDs to retrieve. If ``None``, returns paginated results.

        Returns:
            List of vector dicts with ``id`` and ``embedding`` keys.
        """
        path = f"/v2/collections/{collection}/vectors"
        if ids:
            id_param = ",".join(ids)
            path += f"?ids={id_param}"
        return _expect_list(self._get(path), "get_vectors")

    def delete_vectors(self, collection: str, ids: list[str]) -> None:
        """Delete vectors by ID."""
        self._delete(f"/v2/collections/{collection}/vectors", {"ids": ids})

    # ------------------------------------------------------------------
    # Similarity search
    # ------------------------------------------------------------------

    def search_similar(
        self,
        collection: str,
        vector: list[float] | None = None,
        vectors: list[list[float]] | None = None,
        ids: list[str] | None = None,
        limit: int = 10,
        filter_expr: str | None = None,
        *,
        tenant_id: str | None = None,
        repo: str | None = None,
        rev: str | None = None,
        filter_tag: str | None = None,
        source_db: str | None = None,
        entity_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Nearest-neighbor search in a collection.

        Provide exactly one of ``vector``, ``vectors``, or ``ids``.

        Args:
            collection: Collection name.
            vector: Single query vector.
            vectors: Multiple query vectors.
            ids: Search by existing vector IDs.
            limit: Top-k results.
            filter_expr: Optional raw filter expression.
            tenant_id: Optional tenant scope filter.
            repo: Optional repository scope filter.
            rev: Optional revision scope filter.
            filter_tag: Optional filter tag scope.

        Returns:
            List of result dicts with ``id``, ``distance``, ``score``.
        """
        body: dict[str, Any] = {"limit": limit}
        if vector is not None:
            body["vectors"] = [vector]
        elif vectors is not None:
            body["vectors"] = vectors
        elif ids is not None:
            body["ids"] = ids
        merged_filter = self.build_filter_expr(
            filter_expr,
            tenant_id=tenant_id,
            repo=repo,
            rev=rev,
            filter_tag=filter_tag,
            source_db=source_db,
            entity_type=entity_type,
        )
        if merged_filter:
            body["filter"] = merged_filter

        result = self._post(f"/v2/search/{collection}", body)
        return _expect_list(result.get("results", result) if isinstance(result, dict) else result, "search_similar")

    def search_hybrid(
        self,
        collection: str,
        query_text: str | None = None,
        vectors: list[list[float]] | None = None,
        alpha: float = 0.7,
        limit: int = 10,
        filter_expr: str | None = None,
        *,
        tenant_id: str | None = None,
        repo: str | None = None,
        rev: str | None = None,
        filter_tag: str | None = None,
        source_db: str | None = None,
        entity_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Dense + sparse hybrid search with WeightedRanker fusion.

        Args:
            collection: Collection name.
            query_text: Text to auto-encode for both dense and sparse.
            vectors: Pre-computed dense vectors.
            alpha: Dense/sparse blend (0.0=sparse, 1.0=dense).
            limit: Top-k results.
            filter_expr: Optional raw filter expression.
            tenant_id: Optional tenant scope filter.
            repo: Optional repository scope filter.
            rev: Optional revision scope filter.
            filter_tag: Optional filter tag scope.
        """
        body: dict[str, Any] = {"alpha": alpha, "limit": limit}
        if query_text:
            body["query_text"] = query_text
        if vectors:
            body["vectors"] = vectors
        merged_filter = self.build_filter_expr(
            filter_expr,
            tenant_id=tenant_id,
            repo=repo,
            rev=rev,
            filter_tag=filter_tag,
            source_db=source_db,
            entity_type=entity_type,
        )
        if merged_filter:
            body["filter"] = merged_filter
        result = self._post(f"/v2/search/{collection}/hybrid", body)
        return _expect_list(result.get("results", result) if isinstance(result, dict) else result, "search_hybrid")

    def search_rag(
        self,
        collection: str,
        query_text: str,
        limit: int = 10,
        filter_expr: str | None = None,
        *,
        tenant_id: str | None = None,
        repo: str | None = None,
        rev: str | None = None,
        filter_tag: str | None = None,
        source_db: str | None = None,
        entity_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """RAG-optimized retrieval with optional cross-encoder reranking.

        Args:
            collection: Collection name.
            query_text: Natural language query.
            limit: Top-k results.

        Returns:
            List of chunk dicts with ``text``, ``metadata``, ``score``.
        """
        body: dict[str, Any] = {"query_text": query_text, "limit": limit}
        merged_filter = self.build_filter_expr(
            filter_expr,
            tenant_id=tenant_id,
            repo=repo,
            rev=rev,
            filter_tag=filter_tag,
            source_db=source_db,
            entity_type=entity_type,
        )
        if merged_filter:
            body["filter"] = merged_filter
        result = self._post(f"/v2/search/{collection}/rag", body)
        return _expect_list(result.get("chunks", result) if isinstance(result, dict) else result, "search_rag")

    # ------------------------------------------------------------------
    # Embedding generation
    # ------------------------------------------------------------------

    def embed(
        self,
        data: list[dict[str, Any]],
        model: str | None = None,
        collection: str | None = None,
        modality: str | None = None,
    ) -> list[list[float]]:
        """Generate embeddings via Mycelia's encoder models.

        Args:
            data: List of data objects to embed.
            model: Explicit model name from the model registry.
            collection: Use a collection's trained encoder.
            modality: Explicit modality override.

        Returns:
            List of embedding vectors.
        """
        body: dict[str, Any] = {"data": data}
        if model:
            body["model"] = model
        if collection:
            body["collection"] = collection
        if modality:
            body["modality"] = modality

        result = self._post("/v2/embed", body)
        return _expect_vector_list(result.get("embeddings", result) if isinstance(result, dict) else result, "embed")

    # ------------------------------------------------------------------
    # Arrow IPC bulk transfer
    # ------------------------------------------------------------------

    def to_arrow(
        self,
        collection: str,
        ids: list[str] | None = None,
    ) -> Any:
        """Fetch vectors as a PyArrow Table for efficient bulk transfer.

        Args:
            collection: Collection name.
            ids: Optional ID filter. If ``None``, fetches all vectors.

        Returns:
            ``pyarrow.Table`` with columns ``id`` (string) and
            ``embedding`` (list<float32>).
        """
        _require_arrow()
        vectors = self.get_vectors(collection, ids=ids)

        vec_ids = [v["id"] for v in vectors]
        embeddings = [v["embedding"] for v in vectors]

        table = pa.table(
            {
                "id": pa.array(vec_ids, type=pa.string()),
                "embedding": pa.array(embeddings, type=pa.list_(pa.float32())),
            }
        )
        return table

    def from_arrow(
        self,
        collection: str,
        table: Any,
        id_column: str = "id",
        embedding_column: str = "embedding",
        filter_tag: str | None = None,
        *,
        tenant_id: str | None = None,
        repo: str | None = None,
        rev: str | None = None,
    ) -> dict[str, Any]:
        """Insert vectors from a PyArrow Table into Mycelia.

        Args:
            collection: Collection name.
            table: ``pyarrow.Table`` with id and embedding columns.
            id_column: Column name for vector IDs.
            embedding_column: Column name for embedding vectors.
            filter_tag: Optional tag for filtering.

        Returns:
            Ingestion result.
        """
        _require_arrow()
        ids = table.column(id_column).to_pylist()
        embeddings = table.column(embedding_column).to_pylist()
        vectors = dict(zip(ids, embeddings))
        return self.store_vectors(
            collection,
            vectors,
            filter_tag=filter_tag,
            tenant_id=tenant_id,
            repo=repo,
            rev=rev,
        )

    def register_in_duckdb(
        self,
        connector: Any,
        collection: str,
        table_name: str | None = None,
        ids: list[str] | None = None,
    ) -> str:
        """Register a Mycelia collection as a DuckDB table via Arrow.

        Fetches vectors from Mycelia, converts to Arrow, and registers
        as a queryable DuckDB table.  This enables SQL queries over
        Mycelia-stored embeddings and seamless pipeline integration.

        Args:
            connector: A :class:`DuckDBConnector` instance.
            collection: Mycelia collection name.
            table_name: DuckDB table name (defaults to collection name).
            ids: Optional ID filter.

        Returns:
            The DuckDB table name.
        """
        _require_arrow()
        arrow_table = self.to_arrow(collection, ids=ids)
        tbl_name = table_name or collection

        conn = connector._ensure_open()
        conn.execute(f'DROP TABLE IF EXISTS "{tbl_name}"')
        conn.register(f"_arrow_{tbl_name}", arrow_table)
        conn.execute(f'CREATE TABLE "{tbl_name}" AS SELECT * FROM "_arrow_{tbl_name}"')
        conn.unregister(f"_arrow_{tbl_name}")

        logger.info(
            "Registered Mycelia collection %r as DuckDB table %r (%d rows)",
            collection,
            tbl_name,
            len(arrow_table),
        )
        return tbl_name

    # ------------------------------------------------------------------
    # Pipeline sync
    # ------------------------------------------------------------------

    def sync_pipeline_results(
        self,
        pipeline_result: dict[str, Any],
        representations_collection: str | None = None,
        predictions_collection: str | None = None,
        dimension: int | None = None,
        *,
        tenant_id: str | None = None,
        repo: str | None = None,
        rev: str | None = None,
    ) -> dict[str, Any]:
        """Sync a jcube pipeline result to Mycelia collections.

        Stores representations and predictions from a
        :meth:`DuckDBConnector.run_pipeline` result.

        Args:
            pipeline_result: Dict from ``run_pipeline()`` with keys
                ``representations`` and ``predictions``.
            representations_collection: Collection name for representations.
            predictions_collection: Collection name for predictions.
            dimension: Embedding dimension (auto-detected if not provided).

        Returns:
            Dict with ``representations_stored`` and ``predictions_stored`` counts.
        """
        reps = pipeline_result.get("representations", {})
        preds = pipeline_result.get("predictions", {})
        stored: dict[str, Any] = {}

        scope = pipeline_result.get("scope", {}) if isinstance(pipeline_result.get("scope"), dict) else {}
        effective_tenant_id = tenant_id or scope.get("tenant_id")
        effective_repo = repo or scope.get("repo")
        effective_rev = rev or scope.get("rev")

        rep_metadata = pipeline_result.get("representation_metadata")
        rep_relations = pipeline_result.get("representation_relations")
        rep_text = pipeline_result.get("representation_text")
        pred_metadata = pipeline_result.get("prediction_metadata")
        pred_relations = pipeline_result.get("prediction_relations")
        pred_text = pipeline_result.get("prediction_text")

        if reps and representations_collection:
            dim = dimension or len(next(iter(reps.values())))
            self.ensure_collection(representations_collection, dimension=dim)
            self.store_representations(
                representations_collection,
                reps,
                tenant_id=effective_tenant_id,
                repo=effective_repo,
                rev=effective_rev,
                metadata_by_id=rep_metadata if isinstance(rep_metadata, dict) else None,
                relations_by_id=rep_relations if isinstance(rep_relations, dict) else None,
                text_by_id=rep_text if isinstance(rep_text, dict) else None,
            )
            stored["representations_stored"] = len(reps)

        if preds and predictions_collection:
            first_pred = next(iter(preds.values()))[0]
            dim = dimension or len(first_pred)
            self.ensure_collection(predictions_collection, dimension=dim)
            self.store_predictions(
                predictions_collection,
                preds,
                tenant_id=effective_tenant_id,
                repo=effective_repo,
                rev=effective_rev,
                metadata_by_id=pred_metadata if isinstance(pred_metadata, dict) else None,
                relations_by_id=pred_relations if isinstance(pred_relations, dict) else None,
                text_by_id=pred_text if isinstance(pred_text, dict) else None,
            )
            stored["predictions_stored"] = sum(len(s) for s in preds.values())

        return stored

    def sync_cascade_level(
        self,
        cascade: Any,
        level: str,
        collection: str | None = None,
    ) -> dict[str, Any]:
        """Sync a cascade level's live predictions to Mycelia.

        Args:
            cascade: A :class:`ForecastCascade` instance.
            level: Level name to sync.
            collection: Collection name (defaults to ``"{level}_predictions"``).

        Returns:
            Dict with sync metadata.
        """
        preds = cascade.get_predictions(level)
        if not isinstance(preds, dict) or not preds:
            return {"synced": 0}

        coll = collection or f"{level}_predictions"

        # Auto-detect dimension from first prediction
        first_seq = next(iter(preds.values()))
        if first_seq:
            dim = len(first_seq[0])
            self.ensure_collection(coll, dimension=dim)
            self.store_predictions(coll, preds)
            return {"collection": coll, "synced": sum(len(s) for s in preds.values())}

        return {"synced": 0}


class MyceliaError(Exception):
    """Error from the Mycelia API."""
