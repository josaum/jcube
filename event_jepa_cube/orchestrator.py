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
from typing import Any

logger = logging.getLogger(__name__)


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
        self._connector = None
        self._mycelia = None
        self._bandit = None
        self._cascade = None
        self._streaming: dict[str, Any] = {}

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

        self._mycelia = MyceliaStore(**config)
        logger.info("MyceliaStore initialized: %s", config.get("base_url"))

    def _init_bandit(self, config: dict[str, Any]) -> None:
        from .bandit import BanditClient

        self._bandit = BanditClient(**config)
        logger.info("BanditClient initialized")

    def _init_cascade(self, levels: list[dict[str, Any]]) -> None:
        from .cascade import CascadeLevel, ForecastCascade

        self._cascade = ForecastCascade(self._connector, source_table=self._source_table)
        for level_cfg in levels:
            self._cascade.add_level(CascadeLevel(**level_cfg))
        logger.info("ForecastCascade initialized with %d levels", len(levels))

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
        return self._connector.run_from_sources(sources, tbl_list)

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
        result = self._connector.run_pipeline(
            sequences_table=seq_tbl,
            entities_table=ent_tbl,
        )
        logger.info(
            "Pipeline complete: %d representations, %d predictions",
            len(result.get("representations", {})),
            len(result.get("predictions", {})),
        )

        # 2. Sync to Mycelia if configured
        if sync_to_mycelia and self._mycelia:
            sync_result = self._mycelia.sync_pipeline_results(
                result,
                representations_collection=representations_collection or f"{seq_tbl}_representations",
                predictions_collection=predictions_collection or f"{seq_tbl}_predictions",
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
    ) -> Any:
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
            dim = embedding_dim or (self._connector.embedding_dim if self._connector else 768)
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
