"""End-to-end pipeline orchestrator that wires all jcube components together.

Connects the full data flow::

    Data sources → DuckDB warehouse → ForecastCascade → MyceliaStore
                                        ↕                    ↕
                                  TriggerEngine          Similarity search
                                        ↕                    ↕
                                  StreamingJEPA ←── BanditClient (adaptive)
                                        ↕
                                  GEPASearch (embedding evolution)

Zero required dependencies — all component imports are lazy and optional.
The orchestrator gracefully degrades when components are unavailable.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .bandit import BanditClient
    from .cascade import ForecastCascade
    from .duckdb_connector import DuckDBConnector
    from .mycelia_store import MyceliaStore
    from .streaming import StreamingJEPA


class Pipeline:
    """End-to-end orchestrator for the jcube processing stack.

    Wires together DuckDB, ForecastCascade, MyceliaStore, BanditClient,
    StreamingJEPA, and GEPASearch into a single cohesive pipeline.

    Components are optional — the pipeline works with whatever subset
    is configured.

    Example::

        pipeline = Pipeline(
            duckdb_config={"database": "warehouse.duckdb", "embedding_dim": 768},
            mycelia_config={"base_url": "https://api.getjai.com", "api_key": "..."},
            cascade_levels=[
                {"name": "patient", "num_prediction_steps": 3},
                {"name": "department", "num_prediction_steps": 3},
            ],
        )
        pipeline.ingest_sources({"postgres": "postgresql://..."})
        result = pipeline.run()
        pipeline.shutdown()
    """

    def __init__(
        self,
        duckdb_config: dict[str, Any] | None = None,
        mycelia_config: dict[str, Any] | None = None,
        bandit_config: dict[str, Any] | None = None,
        cascade_levels: list[dict[str, Any]] | None = None,
        source_table: str = "event_sequences",
        entities_table: str | None = None,
    ) -> None:
        self._source_table = source_table
        self._entities_table = entities_table
        self._connector: DuckDBConnector | None = None
        self._mycelia: MyceliaStore | None = None
        self._mycelia_scope: dict[str, Any] = {}
        self._bandit: BanditClient | None = None
        self._cascade: ForecastCascade | None = None
        self._streaming: dict[str, StreamingJEPA] = {}

        if duckdb_config:
            self._init_duckdb(duckdb_config)
        if mycelia_config:
            self._init_mycelia(mycelia_config)
        if bandit_config:
            self._init_bandit(bandit_config)
        if cascade_levels and self._connector:
            self._init_cascade(cascade_levels)

    # ------------------------------------------------------------------
    # Component initialization
    # ------------------------------------------------------------------

    def _init_duckdb(self, config: dict[str, Any]) -> None:
        from .duckdb_connector import DuckDBConnector

        self._connector = DuckDBConnector(**config)
        logger.info("DuckDB connector initialized: %s", config.get("database", ":memory:"))

    def _init_mycelia(self, config: dict[str, Any]) -> None:
        from .mycelia_store import MyceliaStore

        runtime_config = dict(config)
        self._mycelia_scope = {
            key: runtime_config.pop(key)
            for key in ("tenant_id", "repo", "rev")
            if key in runtime_config and runtime_config[key] is not None
        }
        runtime_config.setdefault("vector_ingest_transport", "flight")
        self._mycelia = MyceliaStore(**runtime_config)
        logger.info("MyceliaStore initialized: %s", runtime_config.get("base_url"))

    def _init_bandit(self, config: dict[str, Any]) -> None:
        from .bandit import BanditClient

        self._bandit = BanditClient(**config)
        logger.info("BanditClient initialized")

    def _init_cascade(self, levels: list[dict[str, Any]]) -> None:
        from .cascade import CascadeLevel, ForecastCascade

        if self._connector is None:
            raise PipelineError("DuckDB connector must be configured before cascade setup")
        self._cascade = ForecastCascade(self._connector, source_table=self._source_table)
        for level_cfg in levels:
            self._cascade.add_level(CascadeLevel(**level_cfg))
        logger.info("ForecastCascade initialized with %d levels", len(levels))

    def _normalize_pipeline_result(self, result: Any) -> dict[str, Any]:
        """Normalize connector/materializer results to a dict for callers."""
        try:
            from .materializer import MaterializationResult, result_to_dict

            if isinstance(result, MaterializationResult):
                return result_to_dict(result)
        except ImportError:
            pass

        if isinstance(result, dict):
            return result

        raise PipelineError(f"Unsupported pipeline result type: {type(result)!r}")

    def _build_mycelia_sync_payload(self, result: Any) -> dict[str, Any]:
        """Build the richest Mycelia sync payload available for the result."""
        try:
            from .materializer import MaterializationResult, result_to_mycelia_payloads

            if isinstance(result, MaterializationResult):
                return result_to_mycelia_payloads(
                    result,
                    tenant_id=self._mycelia_scope.get("tenant_id"),
                    repo=self._mycelia_scope.get("repo"),
                    rev=self._mycelia_scope.get("rev"),
                )
        except ImportError:
            pass

        if isinstance(result, dict):
            return result

        raise PipelineError(f"Unsupported pipeline sync payload type: {type(result)!r}")

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def ingest_sources(
        self,
        sources: dict[str, str],
        tables: list[str] | None = None,
    ) -> dict[str, int]:
        """Attach external databases and build warehouse tables.

        Args:
            sources: Mapping of name to connection string.
                Supported: ``postgresql://``, ``mysql://``, ``sqlite://``,
                DuckDB file paths.
            tables: Tables to replicate. If ``None``, uses source_table.

        Returns:
            Row counts per table.
        """
        if not self._connector:
            raise PipelineError("DuckDB connector not configured")

        tbl_list = tables or [self._source_table]
        source_configs = [
            {"name": name, "connection_string": connection_string}
            for name, connection_string in sources.items()
        ]
        return self._connector.run_from_sources(source_configs, tbl_list)

    def ingest_from_mycelia(
        self,
        collection: str,
        table_name: str | None = None,
    ) -> str:
        """Load a Mycelia collection into DuckDB as a table.

        Args:
            collection: Mycelia collection name.
            table_name: DuckDB table name (defaults to collection).

        Returns:
            DuckDB table name.
        """
        if not self._mycelia:
            raise PipelineError("MyceliaStore not configured")
        if not self._connector:
            raise PipelineError("DuckDB connector not configured")
        return self._mycelia.register_in_duckdb(self._connector, collection, table_name)

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------

    def run(
        self,
        sequences_table: str | None = None,
        entities_table: str | None = None,
        sync_to_mycelia: bool = True,
        representations_collection: str | None = None,
        predictions_collection: str | None = None,
    ) -> dict[str, Any]:
        """Run the full pipeline: process → cascade → persist → search.

        Args:
            sequences_table: Override source table for sequences.
            entities_table: Override entity table.
            sync_to_mycelia: Whether to persist results to Mycelia.
            representations_collection: Mycelia collection for representations.
            predictions_collection: Mycelia collection for predictions.

        Returns:
            Dict with pipeline results: representations, predictions,
            patterns, relationships, and sync metadata.
        """
        if not self._connector:
            raise PipelineError("DuckDB connector not configured")

        seq_tbl = sequences_table or self._source_table
        ent_tbl = entities_table or self._entities_table

        # 1. Run batch pipeline
        raw_result = self._connector.run_pipeline(
            sequences_table=seq_tbl,
            entities_table=ent_tbl,
        )
        result = self._normalize_pipeline_result(raw_result)
        logger.info(
            "Pipeline complete: %d representations, %d predictions",
            len(result.get("representations", {})),
            len(result.get("predictions", {})),
        )

        # 2. Sync to Mycelia if configured
        if sync_to_mycelia and self._mycelia:
            sync_payload = self._build_mycelia_sync_payload(raw_result)
            sync_result = self._mycelia.sync_pipeline_results(
                sync_payload,
                representations_collection=representations_collection or f"{seq_tbl}_representations",
                predictions_collection=predictions_collection or f"{seq_tbl}_predictions",
                tenant_id=self._mycelia_scope.get("tenant_id"),
                repo=self._mycelia_scope.get("repo"),
                rev=self._mycelia_scope.get("rev"),
            )
            result["mycelia_sync"] = sync_result
            logger.info("Synced to Mycelia: %s", sync_result)

        return result

    # ------------------------------------------------------------------
    # Cascade operations
    # ------------------------------------------------------------------

    def start_cascade(self, interval_seconds: float = 5.0) -> Any:
        """Start the cascade pipeline in the background.

        Returns:
            StopHandle for stopping the cascade.
        """
        if not self._cascade:
            raise PipelineError("Cascade not configured")
        return self._cascade.watch_async(interval_seconds=interval_seconds)

    def poll_cascade(self) -> None:
        """Poll all cascade levels once (synchronous)."""
        if not self._cascade:
            raise PipelineError("Cascade not configured")
        self._cascade.poll_once()

    def sync_cascade_to_mycelia(self) -> dict[str, Any]:
        """Sync all cascade level predictions to Mycelia.

        Returns:
            Dict of level_name → sync metadata.
        """
        if not self._cascade or not self._mycelia:
            raise PipelineError("Cascade and MyceliaStore must both be configured")

        sync_results = {}
        for level in self._cascade._levels:
            result = self._mycelia.sync_cascade_level(self._cascade, level.name)
            sync_results[level.name] = result
        return sync_results

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def get_streaming_processor(
        self,
        stream_id: str,
        embedding_dim: int | None = None,
        alpha: float = 1.0,
        window_size: int | None = None,
    ) -> StreamingJEPA:
        """Get or create a StreamingJEPA for a given stream.

        Args:
            stream_id: Unique identifier for the stream.
            embedding_dim: Embedding dimension (auto-detected from connector if not provided).
            alpha: Exponential decay rate.
            window_size: Optional sliding window size.

        Returns:
            StreamingJEPA instance.
        """
        from .streaming import StreamingJEPA

        if stream_id not in self._streaming:
            dim = embedding_dim or (self._connector._jepa.embedding_dim if self._connector else 768)
            self._streaming[stream_id] = StreamingJEPA(embedding_dim=dim, alpha=alpha, window_size=window_size)
        return self._streaming[stream_id]

    def process_event(
        self,
        stream_id: str,
        embedding: list[float],
        timestamp: float,
    ) -> list[float]:
        """Process a single streaming event.

        Args:
            stream_id: Stream identifier.
            embedding: Event embedding vector.
            timestamp: Event timestamp.

        Returns:
            Updated representation.
        """
        processor = self.get_streaming_processor(stream_id, embedding_dim=len(embedding))
        return processor.update(embedding, timestamp)

    # ------------------------------------------------------------------
    # Search operations
    # ------------------------------------------------------------------

    def search_similar(
        self,
        collection: str,
        vector: list[float],
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search for similar vectors in Mycelia.

        Args:
            collection: Collection to search.
            vector: Query vector.
            limit: Top-k results.
        """
        if not self._mycelia:
            raise PipelineError("MyceliaStore not configured")
        return self._mycelia.search_similar(collection, vector=vector, limit=limit)

    def search_gepa(
        self,
        collection: str,
        seed_vector: list[float],
        iterations: int = 5,
        limit: int = 10,
    ) -> Any:
        """Run GEPA evolutionary search on a Mycelia collection.

        Args:
            collection: Collection to search.
            seed_vector: Initial query embedding.
            iterations: Evolution iterations.
            limit: Final top-k results.
        """
        from .gepa import GEPASearch

        if self._mycelia is None:
            raise PipelineError("MyceliaStore not configured")
        gepa = GEPASearch(
            base_url=self._mycelia._base_url,
            api_key=self._mycelia._api_key,
            namespace=self._mycelia._namespace,
        )
        return gepa.search(collection, seed_vector, iterations=iterations, limit=limit)

    def search_gepa_local(
        self,
        vectors: dict[str, list[float]],
        seed_vector: list[float],
        iterations: int = 5,
        limit: int = 10,
    ) -> Any:
        """Run GEPA evolutionary search locally (no API needed).

        Args:
            vectors: Dict of {id: embedding} to search.
            seed_vector: Initial query embedding.
            iterations: Evolution iterations.
            limit: Final top-k results.
        """
        from .gepa import GEPASearch

        gepa = GEPASearch(base_url="unused")
        return gepa.search_local(vectors, seed_vector, iterations=iterations, limit=limit)

    # ------------------------------------------------------------------
    # Bandit-powered adaptive selection
    # ------------------------------------------------------------------

    def select_cascade_levels(
        self,
        context: list[float],
        k: int = 2,
    ) -> list[str]:
        """Use bandits to select top-k cascade levels for a context.

        Args:
            context: Sequence representation vector.
            k: Number of levels to select.

        Returns:
            List of level names ordered by bandit score.
        """
        if not self._bandit or not self._cascade:
            raise PipelineError("BanditClient and Cascade must both be configured")

        from .bandit import CascadeBandit

        cb = CascadeBandit(self._bandit)
        cb.setup_from_cascade(self._cascade)
        return cb.select_levels(context, k=k)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Close all connections and release resources."""
        if self._connector:
            self._connector.close()
            self._connector = None
        self._streaming.clear()
        logger.info("Pipeline shut down")

    def __enter__(self) -> Pipeline:
        return self

    def __exit__(self, *args: Any) -> None:
        self.shutdown()


class PipelineError(Exception):
    """Error from the Pipeline orchestrator."""
