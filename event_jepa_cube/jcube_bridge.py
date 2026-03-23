"""Bridge JCUBE digital twin embeddings to Mycelia (api.getjai.com).

Pushes 17.4M×64 Graph-JEPA embeddings to Milvus via the MyceliaStore REST client,
enabling semantic search, anomaly detection, and audit decisions through the
Mycelia agent infrastructure.

Usage:
    python -m event_jepa_cube.jcube_bridge push \
        --graph data/jcube_graph.parquet \
        --weights data/weights/node_emb_epoch_2.pt \
        --api https://api.getjai.com

    python -m event_jepa_cube.jcube_bridge search \
        --graph data/jcube_graph.parquet \
        --weights data/weights/node_emb_epoch_2.pt \
        --api https://api.getjai.com \
        --query "GHO-BRADESCO/ID_CD_INTERNACAO_53873" -k 10
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Any, Optional

import numpy as np

from .mycelia_store import MyceliaStore, MyceliaError

logger = logging.getLogger(__name__)

# Default collection names
TWIN_COLLECTION = "jcube_hospital_twin"
ONTOLOGY_COLLECTION = "jcube_ontology"
EMBEDDING_DIM = 64


def _load_node_vocab(graph_parquet: str) -> np.ndarray:
    """Load node names from graph parquet using C++ fast path."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyarrow.compute as pc

    table = pq.read_table(graph_parquet, columns=["subject_id", "object_id"])
    all_nodes = pa.chunked_array(
        table.column("subject_id").chunks + table.column("object_id").chunks
    )
    unique = pc.unique(all_nodes)
    names = unique.to_numpy(zero_copy_only=False).astype(object)
    del table, all_nodes, unique
    return names


def _load_embeddings(weights_path: str) -> np.ndarray:
    """Load embedding tensor from checkpoint."""
    import torch

    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if isinstance(state, torch.Tensor):
        return state.float().numpy()
    elif isinstance(state, dict) and "weight" in state:
        return state["weight"].float().numpy()
    else:
        return list(state.values())[0].float().numpy()


def _parse_node_id(name: str) -> dict[str, str]:
    """Extract source_db and entity_type from node ID.

    Examples:
        "GHO-BRADESCO/ID_CD_INTERNACAO_117926" → {source_db: "GHO-BRADESCO", entity_type: "INTERNACAO", raw_id: "117926"}
        "ID_CD_CID_A419" → {source_db: "", entity_type: "CID", raw_id: "A419"}
    """
    name = str(name)
    source_db = ""
    rest = name

    if "/" in name:
        source_db, rest = name.split("/", 1)

    parts = rest.split("_")
    entity_type = ""
    raw_id = rest

    if len(parts) >= 3 and parts[0] == "ID" and parts[1] == "CD":
        entity_type = parts[2]
        raw_id = "_".join(parts[3:]) if len(parts) > 3 else ""

    return {"source_db": source_db, "entity_type": entity_type, "raw_id": raw_id}


class JCubeBridge:
    """Bridge between JCUBE digital twin and Mycelia API.

    Loads trained embeddings + node vocabulary and pushes them to Milvus
    via the Mycelia REST API for semantic search and anomaly detection.
    """

    def __init__(
        self,
        api_url: str = "https://api.getjai.com",
        api_key: Optional[str] = None,
        namespace: Optional[str] = None,
    ):
        self.store = MyceliaStore(api_url, api_key=api_key, namespace=namespace)
        self.node_names: Optional[np.ndarray] = None
        self.embeddings: Optional[np.ndarray] = None
        self.node_to_idx: dict[str, int] = {}
        self._loaded = False

    def load_twin(self, graph_parquet: str, weights_path: str) -> None:
        """Load node vocabulary and embeddings from files."""
        print(f"Loading node vocabulary from {graph_parquet}...")
        t0 = time.time()
        self.node_names = _load_node_vocab(graph_parquet)
        print(f"  {len(self.node_names):,} nodes in {time.time() - t0:.1f}s")

        print(f"Loading embeddings from {weights_path}...")
        t1 = time.time()
        self.embeddings = _load_embeddings(weights_path)
        print(f"  {self.embeddings.shape} in {time.time() - t1:.1f}s")

        assert len(self.node_names) == self.embeddings.shape[0], (
            f"Mismatch: {len(self.node_names)} names vs {self.embeddings.shape[0]} vectors"
        )

        self.node_to_idx = {str(name): i for i, name in enumerate(self.node_names)}
        self._loaded = True

    def push_to_milvus(
        self,
        collection: str = TWIN_COLLECTION,
        batch_size: int = 1000,
        entity_types: Optional[set[str]] = None,
        source_dbs: Optional[set[str]] = None,
        limit: Optional[int] = None,
    ) -> dict[str, Any]:
        """Push embeddings to Mycelia Milvus.

        Args:
            collection: Milvus collection name.
            batch_size: Vectors per HTTP request.
            entity_types: Filter to specific entity types (e.g., {"INTERNACAO", "PACIENTE"}).
            source_dbs: Filter to specific hospitals (e.g., {"GHO-BRADESCO"}).
            limit: Max vectors to push (for testing).

        Returns:
            Stats dict with counts.
        """
        assert self._loaded, "Call load_twin() first"

        dim = self.embeddings.shape[1]
        print(f"\nPushing to {self.store._base_url}")
        print(f"  Collection: {collection}")
        print(f"  Dimension: {dim}")

        # Create collection
        try:
            self.store.ensure_collection(collection, dimension=dim, modality="tabular")
            print(f"  Collection ready")
        except MyceliaError as e:
            print(f"  Collection error (continuing): {e}")

        # Prepare vectors
        n_total = len(self.node_names)
        n_pushed = 0
        n_skipped = 0
        t0 = time.time()

        batch_vectors: dict[str, list[float]] = {}

        for i in range(n_total):
            if limit and n_pushed >= limit:
                break

            name = str(self.node_names[i])
            parsed = _parse_node_id(name)

            # Filter by entity type
            if entity_types and parsed["entity_type"] not in entity_types:
                n_skipped += 1
                continue

            # Filter by source_db
            if source_dbs and parsed["source_db"] not in source_dbs:
                n_skipped += 1
                continue

            batch_vectors[name] = self.embeddings[i].tolist()

            if len(batch_vectors) >= batch_size:
                try:
                    self.store.store_vectors(
                        collection,
                        batch_vectors,
                        filter_tag=parsed.get("source_db", ""),
                    )
                    n_pushed += len(batch_vectors)
                except MyceliaError as e:
                    logger.warning("Batch insert failed: %s", e)

                if n_pushed % (batch_size * 10) == 0:
                    elapsed = time.time() - t0
                    rate = n_pushed / max(elapsed, 0.1)
                    remaining = (n_total - i) / max(rate, 1)
                    print(f"  {n_pushed:>10,} / {n_total:,} pushed ({rate:.0f}/s, ~{remaining:.0f}s remaining)")

                batch_vectors = {}

        # Final batch
        if batch_vectors:
            try:
                self.store.store_vectors(collection, batch_vectors)
                n_pushed += len(batch_vectors)
            except MyceliaError as e:
                logger.warning("Final batch insert failed: %s", e)

        elapsed = time.time() - t0
        stats = {
            "collection": collection,
            "n_pushed": n_pushed,
            "n_skipped": n_skipped,
            "elapsed_s": elapsed,
            "rate": n_pushed / max(elapsed, 0.1),
        }

        print(f"\n  Done: {n_pushed:,} pushed, {n_skipped:,} skipped in {elapsed:.1f}s")
        return stats

    def search(
        self,
        query: str,
        collection: str = TWIN_COLLECTION,
        k: int = 10,
        entity_type: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Search for similar entities via Mycelia.

        Args:
            query: Entity ID (e.g., "GHO-BRADESCO/ID_CD_INTERNACAO_53873").
            collection: Milvus collection name.
            k: Number of results.
            entity_type: Filter to specific entity type.

        Returns:
            List of {id, score} dicts.
        """
        assert self._loaded, "Call load_twin() first"

        if query not in self.node_to_idx:
            raise KeyError(f"Entity '{query}' not in graph")

        idx = self.node_to_idx[query]
        vec = self.embeddings[idx].tolist()

        filter_expr = None
        if entity_type:
            filter_expr = f'entity_type == "{entity_type}"'

        # Use raw API: POST /v2/search/{collection}
        body: dict[str, Any] = {"vectors": [vec], "top_k": k + 1}
        if filter_expr:
            body["filter"] = filter_expr

        try:
            resp = self.store._post(f"/v2/search/{collection}", body)
            matches = resp.get("results", [{}])[0].get("matches", []) if resp else []
        except MyceliaError:
            # Fallback to store.search_similar
            matches = self.store.search_similar(collection, vector=vec, limit=k + 1) or []

        # Remove self-match
        return [m for m in matches if m.get("id") != query][:k]

    def anomalies(
        self,
        entity_type: str = "INTERNACAO",
        source_db: Optional[str] = None,
        top_n: int = 20,
    ) -> list[dict[str, Any]]:
        """Compute anomaly scores (centroid distance).

        Runs locally on loaded embeddings — doesn't require Milvus.
        """
        assert self._loaded, "Call load_twin() first"

        # Build mask for entity type
        mask = np.zeros(len(self.node_names), dtype=bool)
        for i, name in enumerate(self.node_names):
            parsed = _parse_node_id(str(name))
            if parsed["entity_type"] != entity_type:
                continue
            if source_db and parsed["source_db"] != source_db:
                continue
            mask[i] = True

        vecs = self.embeddings[mask]
        names = self.node_names[mask]

        if len(vecs) == 0:
            return []

        centroid = vecs.mean(axis=0, keepdims=True)
        dists = np.linalg.norm(vecs - centroid, axis=1)
        z = (dists - dists.mean()) / max(dists.std(), 1e-8)

        top_idx = np.argsort(-z)[:top_n]
        return [
            {
                "id": str(names[i]),
                "z_score": float(z[i]),
                "distance": float(dists[i]),
                **_parse_node_id(str(names[i])),
            }
            for i in top_idx
        ]


def main():
    parser = argparse.ArgumentParser(description="JCUBE ↔ Mycelia Bridge")
    parser.add_argument("--graph", required=True, help="Path to jcube_graph.parquet")
    parser.add_argument("--weights", required=True, help="Path to node_emb_epoch_N.pt")
    parser.add_argument("--api", default="https://api.getjai.com", help="Mycelia API URL")
    parser.add_argument("--api-key", default=None, help="API key")
    parser.add_argument("--namespace", default=None, help="Mycelia namespace")

    sub = parser.add_subparsers(dest="command")

    # Push
    push_p = sub.add_parser("push", help="Push embeddings to Milvus")
    push_p.add_argument("--collection", default=TWIN_COLLECTION)
    push_p.add_argument("--batch-size", type=int, default=1000)
    push_p.add_argument("--entity-types", nargs="*", help="Filter entity types")
    push_p.add_argument("--source-dbs", nargs="*", help="Filter source DBs")
    push_p.add_argument("--limit", type=int, default=None, help="Max vectors")

    # Search
    search_p = sub.add_parser("search", help="Search similar entities")
    search_p.add_argument("--query", required=True, help="Entity ID to search for")
    search_p.add_argument("-k", type=int, default=10)
    search_p.add_argument("--entity-type", default=None)
    search_p.add_argument("--collection", default=TWIN_COLLECTION)

    # Anomalies
    anom_p = sub.add_parser("anomalies", help="Find anomalies")
    anom_p.add_argument("--entity-type", default="INTERNACAO")
    anom_p.add_argument("--source-db", default=None)
    anom_p.add_argument("-n", type=int, default=20)

    args = parser.parse_args()

    bridge = JCubeBridge(api_url=args.api, api_key=args.api_key, namespace=args.namespace)
    bridge.load_twin(args.graph, args.weights)

    if args.command == "push":
        etypes = set(args.entity_types) if args.entity_types else None
        sdbs = set(args.source_dbs) if args.source_dbs else None
        stats = bridge.push_to_milvus(
            collection=args.collection,
            batch_size=args.batch_size,
            entity_types=etypes,
            source_dbs=sdbs,
            limit=args.limit,
        )
        print(f"\nStats: {stats}")

    elif args.command == "search":
        results = bridge.search(
            args.query, collection=args.collection, k=args.k, entity_type=args.entity_type,
        )
        print(f"\nTop {args.k} similar to {args.query}:")
        for r in results:
            print(f"  {r.get('id', '?'):50s}  score={r.get('distance', 0):.4f}")

    elif args.command == "anomalies":
        results = bridge.anomalies(
            entity_type=args.entity_type, source_db=args.source_db, top_n=args.n,
        )
        print(f"\nTop {args.n} anomalous {args.entity_type}:")
        for r in results:
            print(f"  {r['id']:50s}  z={r['z_score']:.2f}  dist={r['distance']:.4f}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
