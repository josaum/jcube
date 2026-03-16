"""Arrow Flight streaming for high-throughput bulk transfer between Mycelia and DuckDB.

Uses PyArrow Flight RPC for zero-copy streaming of embedding vectors,
bypassing JSON serialization overhead. Supports both Flight SQL (DuckDB)
and raw Flight (Mycelia) protocols.

Requires ``pyarrow``. Install with: ``pip install pyarrow>=14.0``

Example::

    from event_jepa_cube.flight_transfer import FlightTransfer

    ft = FlightTransfer("https://api.getjai.com", api_key="...")
    ft.stream_to_duckdb(connector, "patient_embeddings")
    ft.export_to_ipc("patient_embeddings", "/tmp/backup.arrow")
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

try:
    import pyarrow as pa
    import pyarrow.flight as flight
    import pyarrow.ipc as ipc

    _ARROW_AVAILABLE = True
except ImportError:
    _ARROW_AVAILABLE = False

_FLIGHT_PORT = 8815

# Schema matching mycelia_store.py: id (string) + embedding (list<float32>)
_VECTOR_SCHEMA_FIELDS = [
    ("id", "string"),
    ("embedding", "list<float32>"),
]


def _require_arrow() -> None:
    if not _ARROW_AVAILABLE:
        raise ImportError("pyarrow is required for Arrow Flight transfer. Install with: pip install pyarrow>=14.0")


def _vector_schema() -> Any:
    """Return the canonical PyArrow schema for vector data."""
    _require_arrow()
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("embedding", pa.list_(pa.float32())),
        ]
    )


def _derive_flight_url(base_url: str) -> str:
    """Derive the Flight RPC endpoint from a Mycelia REST base URL.

    Replaces the port with the Flight convention port (8815) and uses
    the ``grpc`` scheme.

    Args:
        base_url: Mycelia REST API URL (e.g. ``"https://api.getjai.com"``).

    Returns:
        Flight endpoint URL (e.g. ``"grpc://api.getjai.com:8815"``).
    """
    parsed = urlparse(base_url)
    host = parsed.hostname or "localhost"
    return f"grpc://{host}:{_FLIGHT_PORT}"


class FlightTransfer:
    """Arrow Flight streaming for bulk Mycelia <-> DuckDB vector transfer.

    Uses PyArrow Flight RPC for zero-copy streaming of embedding vectors,
    bypassing JSON serialization. Supports both Flight SQL (DuckDB) and
    raw Flight (Mycelia) protocols.

    Args:
        mycelia_url: Mycelia API base URL (e.g. ``"https://api.getjai.com"``).
            The Flight endpoint is derived automatically (same host, port 8815).
        api_key: Optional API key for authentication.
        timeout: Connection timeout in seconds.
    """

    def __init__(self, mycelia_url: str, api_key: str | None = None, timeout: int = 30) -> None:
        _require_arrow()
        self._mycelia_url = mycelia_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._flight_url = _derive_flight_url(mycelia_url)
        self._client = self._connect()

    def _connect(self) -> Any:
        """Create a Flight client to the Mycelia Flight endpoint."""
        kwargs: dict[str, Any] = {}
        if self._api_key:
            kwargs["generic_headers"] = [("authorization", f"Bearer {self._api_key}")]
        try:
            client = flight.FlightClient(self._flight_url, **kwargs)
            return client
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Flight endpoint {self._flight_url}: {e}") from e

    # ------------------------------------------------------------------
    # Mycelia -> DuckDB
    # ------------------------------------------------------------------

    def stream_to_duckdb(
        self,
        connector: Any,
        collection: str,
        table_name: str | None = None,
        batch_size: int = 10000,
    ) -> str:
        """Stream vectors from Mycelia collection directly into DuckDB via Flight.

        Instead of fetching all vectors as JSON then converting to Arrow,
        streams RecordBatches directly.

        Args:
            connector: DuckDBConnector instance.
            collection: Mycelia collection name.
            table_name: DuckDB table name (defaults to collection).
            batch_size: Records per batch (hint to server).

        Returns:
            DuckDB table name.
        """
        tbl_name = table_name or collection
        descriptor = flight.FlightDescriptor.for_path(collection)
        options = flight.FlightCallOptions(timeout=self._timeout)

        # Get flight info for the collection
        try:
            info = self._client.get_flight_info(descriptor, options)
        except Exception as e:
            raise ConnectionError(f"Failed to get flight info for collection {collection!r}: {e}") from e

        if not info.endpoints:
            raise ValueError(f"No endpoints returned for collection {collection!r}")

        # Stream from the first endpoint
        endpoint = info.endpoints[0]
        reader = self._client.do_get(endpoint.ticket, options)

        # Read all batches into an Arrow table
        table = reader.read_all()

        # Register in DuckDB
        conn = connector._ensure_open()
        temp_name = f"_arrow_flight_{tbl_name}"
        conn.execute(f'DROP TABLE IF EXISTS "{tbl_name}"')
        conn.register(temp_name, table)
        conn.execute(f'CREATE TABLE "{tbl_name}" AS SELECT * FROM "{temp_name}"')
        conn.unregister(temp_name)

        logger.info(
            "Streamed %d vectors from Mycelia collection %r into DuckDB table %r",
            len(table),
            collection,
            tbl_name,
        )
        return tbl_name

    # ------------------------------------------------------------------
    # DuckDB -> Mycelia
    # ------------------------------------------------------------------

    def stream_from_duckdb(
        self,
        connector: Any,
        query: str,
        collection: str,
        id_column: str = "id",
        embedding_column: str = "embedding",
    ) -> int:
        """Stream query results from DuckDB into a Mycelia collection.

        Args:
            connector: DuckDBConnector instance.
            query: SQL query producing id + embedding columns.
            collection: Target Mycelia collection.
            id_column: Column name for IDs.
            embedding_column: Column name for embeddings.

        Returns:
            Number of vectors streamed.
        """
        conn = connector._ensure_open()
        result = conn.execute(query)

        # Fetch as Arrow (use to_arrow_table if available, else fall back)
        if hasattr(result, "to_arrow_table"):
            arrow_table = result.to_arrow_table()
        else:
            arrow_table = result.fetch_arrow_table()

        # Rename columns to canonical names if needed
        if id_column != "id" or embedding_column != "embedding":
            rename_map = {id_column: "id", embedding_column: "embedding"}
            new_names = [rename_map.get(name, name) for name in arrow_table.column_names]
            arrow_table = arrow_table.rename_columns(new_names)

        # Select only id and embedding columns
        arrow_table = arrow_table.select(["id", "embedding"])

        # Upload via Flight do_put
        descriptor = flight.FlightDescriptor.for_path(collection)
        options = flight.FlightCallOptions(timeout=self._timeout)

        writer, _ = self._client.do_put(descriptor, arrow_table.schema, options)
        writer.write_table(arrow_table)
        writer.close()

        total = len(arrow_table)
        logger.info(
            "Streamed %d vectors from DuckDB into Mycelia collection %r",
            total,
            collection,
        )
        return total

    # ------------------------------------------------------------------
    # Collection -> Collection
    # ------------------------------------------------------------------

    def stream_between_collections(
        self,
        source_collection: str,
        target_collection: str,
        filter_tag: str | None = None,
    ) -> int:
        """Stream vectors between two Mycelia collections.

        Useful for replication, backup, or migration.

        Args:
            source_collection: Source collection name.
            target_collection: Target collection name.
            filter_tag: Optional tag to filter source vectors.

        Returns:
            Number of vectors streamed.
        """
        # Read from source
        descriptor = flight.FlightDescriptor.for_path(source_collection)
        options = flight.FlightCallOptions(timeout=self._timeout)

        info = self._client.get_flight_info(descriptor, options)
        if not info.endpoints:
            raise ValueError(f"No endpoints returned for source collection {source_collection!r}")

        endpoint = info.endpoints[0]
        reader = self._client.do_get(endpoint.ticket, options)
        table = reader.read_all()

        # Apply filter_tag if specified
        if filter_tag is not None and "filter_tag" in table.column_names:
            mask = pa.compute.equal(table.column("filter_tag"), filter_tag)
            table = table.filter(mask)

        # Select only id and embedding columns for the target
        columns_to_select = ["id", "embedding"]
        available = [c for c in columns_to_select if c in table.column_names]
        table = table.select(available)

        # Write to target
        target_descriptor = flight.FlightDescriptor.for_path(target_collection)
        writer, _ = self._client.do_put(target_descriptor, table.schema, options)
        writer.write_table(table)
        writer.close()

        total = len(table)
        logger.info(
            "Streamed %d vectors from collection %r to %r",
            total,
            source_collection,
            target_collection,
        )
        return total

    # ------------------------------------------------------------------
    # IPC file export / import
    # ------------------------------------------------------------------

    def export_to_ipc(self, collection: str, path: str) -> int:
        """Export collection to Arrow IPC file on disk.

        Args:
            collection: Mycelia collection name.
            path: Output ``.arrow`` or ``.feather`` file path.

        Returns:
            Number of records exported.
        """
        descriptor = flight.FlightDescriptor.for_path(collection)
        options = flight.FlightCallOptions(timeout=self._timeout)

        info = self._client.get_flight_info(descriptor, options)
        if not info.endpoints:
            raise ValueError(f"No endpoints returned for collection {collection!r}")

        endpoint = info.endpoints[0]
        reader = self._client.do_get(endpoint.ticket, options)
        table = reader.read_all()

        with pa.OSFile(path, "wb") as sink:
            writer = ipc.new_file(sink, table.schema)
            writer.write_table(table)
            writer.close()

        total = len(table)
        logger.info("Exported %d records from collection %r to %s", total, collection, path)
        return total

    def import_from_ipc(self, path: str, collection: str) -> int:
        """Import vectors from Arrow IPC file into Mycelia collection.

        Args:
            path: Input ``.arrow`` or ``.feather`` file path.
            collection: Target collection name.

        Returns:
            Number of records imported.
        """
        with pa.OSFile(path, "rb") as source:
            reader = ipc.open_file(source)
            table = reader.read_all()

        descriptor = flight.FlightDescriptor.for_path(collection)
        options = flight.FlightCallOptions(timeout=self._timeout)

        writer, _ = self._client.do_put(descriptor, table.schema, options)
        writer.write_table(table)
        writer.close()

        total = len(table)
        logger.info("Imported %d records from %s into collection %r", total, path, collection)
        return total
