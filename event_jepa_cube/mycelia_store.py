"""Mycelia API connector for embedding storage and similarity search.

Bridges jcube's pipeline with the Mycelia vector database (Milvus backend),
enabling persistent embedding storage, similarity search across sequences,
and efficient Arrow IPC data transfer to/from DuckDB.

The Mycelia API provides:
- Collection management with auto-embedding and schema inference
- Dense, hybrid, and RAG-optimized similarity search
- Array-of-Structs storage for multi-vector entities
- Pre-trained and fine-tuned encoders (SIGReg)

Zero required dependencies (uses ``urllib`` from stdlib).  Optional
``pyarrow`` enables Arrow IPC bulk transfer to DuckDB.

Example::

    store = MyceliaStore("https://api.getjai.com", api_key="...")
    store.ensure_collection("patient_embeddings", dimension=768)
    store.store_representations("patient_embeddings", {"p1": [0.1, ...], "p2": [0.2, ...]})
    results = store.search_similar("patient_embeddings", query_vector=[0.1, ...], limit=5)
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

try:
    import pyarrow as pa

    _ARROW_AVAILABLE = True
except ImportError:
    _ARROW_AVAILABLE = False


def _require_arrow() -> None:
    if not _ARROW_AVAILABLE:
        raise ImportError("pyarrow is required for Arrow data transfer. Install with: pip install pyarrow>=14.0")


class MyceliaStore:
    """Client for the Mycelia API (Milvus-backed vector store).

    Stores and retrieves embedding vectors via REST, with optional Arrow IPC
    bulk transfer for integration with DuckDB.

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
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._namespace = namespace
        self._timeout = timeout

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

    def _head(self, path: str) -> bool:
        """HEAD request, returns True if 2xx."""
        url = f"{self._base_url}{path}"
        req = urllib.request.Request(url, headers=self._headers(), method="HEAD")
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return 200 <= resp.status < 300
        except urllib.error.HTTPError:
            return False
        except urllib.error.URLError:
            return False

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
        return result

    def collection_exists(self, name: str) -> bool:
        """Check if a collection exists."""
        return self._head(f"/v2/collections/{name}")

    def get_collection(self, name: str) -> dict[str, Any]:
        """Get collection details."""
        return self._get(f"/v2/collections/{name}")

    def list_collections(self) -> list[dict[str, Any]]:
        """List all collections."""
        return self._get("/v2/collections")

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
    ) -> dict[str, Any]:
        """Insert pre-computed vectors into a collection.

        Args:
            collection: Collection name.
            vectors: Mapping of ID to embedding vector.
            filter_tag: Optional tag for filtering.

        Returns:
            Ingestion result.
        """
        payload = [
            {"id": vid, "embedding": emb, **({"filter_tag": filter_tag} if filter_tag else {})}
            for vid, emb in vectors.items()
        ]
        return self._post(f"/v2/collections/{collection}/vectors", {"vectors": payload})

    def store_representations(
        self,
        collection: str,
        representations: dict[str, list[float]],
    ) -> dict[str, Any]:
        """Store pipeline representations as vectors.

        Convenience wrapper that tags vectors with ``filter_tag="representation"``.
        """
        return self.store_vectors(collection, representations, filter_tag="representation")

    def store_predictions(
        self,
        collection: str,
        predictions: dict[str, list[list[float]]],
    ) -> dict[str, Any]:
        """Store pipeline predictions as vectors.

        Each prediction step is stored as ``{sequence_id}_step_{n}``.
        """
        flat: dict[str, list[float]] = {}
        for sid, steps in predictions.items():
            for step_num, pred in enumerate(steps, start=1):
                flat[f"{sid}_step_{step_num}"] = pred
        return self.store_vectors(collection, flat, filter_tag="prediction")

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
        return self._get(path)

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
    ) -> list[dict[str, Any]]:
        """Nearest-neighbor search in a collection.

        Provide exactly one of ``vector``, ``vectors``, or ``ids``.

        Args:
            collection: Collection name.
            vector: Single query vector.
            vectors: Multiple query vectors.
            ids: Search by existing vector IDs.
            limit: Top-k results.
            filter_expr: Optional metadata filter.

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
        if filter_expr:
            body["filter"] = filter_expr

        result = self._post(f"/v2/search/{collection}", body)
        return result.get("results", result) if isinstance(result, dict) else result

    def search_hybrid(
        self,
        collection: str,
        query_text: str | None = None,
        vectors: list[list[float]] | None = None,
        alpha: float = 0.7,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Dense + sparse hybrid search with WeightedRanker fusion.

        Args:
            collection: Collection name.
            query_text: Text to auto-encode for both dense and sparse.
            vectors: Pre-computed dense vectors.
            alpha: Dense/sparse blend (0.0=sparse, 1.0=dense).
            limit: Top-k results.
        """
        body: dict[str, Any] = {"alpha": alpha, "limit": limit}
        if query_text:
            body["query_text"] = query_text
        if vectors:
            body["vectors"] = vectors
        result = self._post(f"/v2/search/{collection}/hybrid", body)
        return result.get("results", result) if isinstance(result, dict) else result

    def search_rag(
        self,
        collection: str,
        query_text: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """RAG-optimized retrieval with optional cross-encoder reranking.

        Args:
            collection: Collection name.
            query_text: Natural language query.
            limit: Top-k results.

        Returns:
            List of chunk dicts with ``text``, ``metadata``, ``score``.
        """
        result = self._post(f"/v2/search/{collection}/rag", {"query_text": query_text, "limit": limit})
        return result.get("chunks", result) if isinstance(result, dict) else result

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
        return result.get("embeddings", result) if isinstance(result, dict) else result

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
        return self.store_vectors(collection, vectors, filter_tag=filter_tag)

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

        if reps and representations_collection:
            dim = dimension or len(next(iter(reps.values())))
            self.ensure_collection(representations_collection, dimension=dim)
            self.store_representations(representations_collection, reps)
            stored["representations_stored"] = len(reps)

        if preds and predictions_collection:
            first_pred = next(iter(preds.values()))[0]
            dim = dimension or len(first_pred)
            self.ensure_collection(predictions_collection, dimension=dim)
            self.store_predictions(predictions_collection, preds)
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
