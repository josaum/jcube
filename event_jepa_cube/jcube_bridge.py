"""Push graph node embeddings to Mycelia/Milvus through the canonical Flight path.

This bridge keeps the hot path simple:

- load node vocabulary from graph parquet
- load node embedding tensor
- enrich each node with scope, metadata, relations, and text
- push batches through ``MyceliaStore.store_vectors()`` over Flight ``do_exchange``

The bridge keeps the richer row contract from ``MyceliaStore`` while using
Arrow Flight for the bulk insert hot path.

Usage::

    MYCELIA_API_KEY=... modal run --detach event_jepa_cube/jcube_bridge.py
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

try:
    import modal
except ImportError:  # pragma: no cover - optional runtime dependency
    modal = None

KEY_TYPES = frozenset(
    {
        "INTERNACAO",
        "PACIENTE",
        "FATURA",
        "CID",
        "TUSS",
        "HOSPITAL",
        "EVOLUCAO",
        "AUDITORIA",
        "MEDICO",
    }
)
DEFAULT_BASE_URL = "https://api.getjai.com"
DEFAULT_COLLECTION = "jcube_twin_v5"
DEFAULT_FILTER_TAG = "graph_node"
DEFAULT_BATCH_SIZE = 10_000
DEFAULT_GRAPH_PATH = "/data/jcube_graph.parquet"
DEFAULT_WEIGHTS_PATH = "/cache/tkg-v5/node_emb_epoch_1.pt"
_NODE_ID_RE = re.compile(
    r"^(?:(?P<source_db>[^/]+)/)?ID_CD_(?P<entity_type>[A-Z0-9]+)_(?P<entity_id>.+)$"
)


@dataclass(frozen=True)
class BridgeConfig:
    base_url: str
    api_key: str
    namespace: str | None
    collection: str
    filter_tag: str
    batch_size: int
    graph_path: str
    weights_path: str
    tenant_id: str | None
    repo: str | None
    rev: str | None
    key_types: frozenset[str]


@dataclass(frozen=True)
class NodeDescriptor:
    node_id: str
    source_db: str | None
    entity_type: str | None
    entity_id: str | None
    scoped: bool
    node_kind: str


def load_bridge_config() -> BridgeConfig:
    api_key = (
        os.getenv("MYCELIA_API_KEY")
        or os.getenv("MYCELIA_BEARER_TOKEN")
        or os.getenv("FLIGHT_SECRET")
    )
    if not api_key:
        raise RuntimeError("Set MYCELIA_API_KEY (or MYCELIA_BEARER_TOKEN) before running jcube_bridge.")

    key_types_env = os.getenv("JCUBE_KEY_TYPES")
    key_types = KEY_TYPES
    if key_types_env:
        parsed = {item.strip().upper() for item in key_types_env.split(",") if item.strip()}
        if parsed:
            key_types = frozenset(parsed)

    return BridgeConfig(
        base_url=os.getenv("MYCELIA_BASE_URL", DEFAULT_BASE_URL),
        api_key=api_key,
        namespace=os.getenv("MYCELIA_NAMESPACE"),
        collection=os.getenv("MYCELIA_COLLECTION", DEFAULT_COLLECTION),
        filter_tag=os.getenv("MYCELIA_FILTER_TAG", DEFAULT_FILTER_TAG),
        batch_size=int(os.getenv("JCUBE_BATCH_SIZE", str(DEFAULT_BATCH_SIZE))),
        graph_path=os.getenv("JCUBE_GRAPH_PATH", DEFAULT_GRAPH_PATH),
        weights_path=os.getenv("JCUBE_WEIGHTS_PATH", DEFAULT_WEIGHTS_PATH),
        tenant_id=os.getenv("MYCELIA_TENANT_ID"),
        repo=os.getenv("MYCELIA_REPO", "jcube"),
        rev=os.getenv("MYCELIA_REV") or os.getenv("JCUBE_REV"),
        key_types=key_types,
    )


def describe_node(node_id: str) -> NodeDescriptor:
    text = str(node_id)
    match = _NODE_ID_RE.match(text)
    if not match:
        return NodeDescriptor(
            node_id=text,
            source_db=None,
            entity_type=None,
            entity_id=None,
            scoped=False,
            node_kind="other",
        )

    source_db = match.group("source_db")
    entity_type = match.group("entity_type")
    entity_id = match.group("entity_id")
    return NodeDescriptor(
        node_id=text,
        source_db=source_db,
        entity_type=entity_type,
        entity_id=entity_id,
        scoped=source_db is not None,
        node_kind="instance",
    )


def should_push_node(node_id: str, key_types: frozenset[str] = KEY_TYPES) -> bool:
    descriptor = describe_node(node_id)
    return descriptor.entity_type in key_types


def build_node_payload(
    node_id: str,
    *,
    repo: str | None = None,
    rev: str | None = None,
) -> tuple[dict[str, Any], list[str], str]:
    descriptor = describe_node(node_id)
    metadata: dict[str, Any] = {
        "node_id": descriptor.node_id,
        "node_kind": descriptor.node_kind,
        "scoped": descriptor.scoped,
    }
    relations = ["kind:graph_node", f"node_kind:{descriptor.node_kind}"]
    text_parts = [f"graph node={descriptor.node_id}"]

    if descriptor.entity_type:
        metadata["entity_type"] = descriptor.entity_type
        relations.append(f"entity_type:{descriptor.entity_type.lower()}")
        text_parts.append(f"type={descriptor.entity_type}")
    if descriptor.entity_id:
        metadata["entity_id"] = descriptor.entity_id
        relations.append(f"entity:{descriptor.entity_id}")
        text_parts.append(f"entity_id={descriptor.entity_id}")
    if descriptor.source_db:
        metadata["source_db"] = descriptor.source_db
        relations.append(f"source_db:{descriptor.source_db.lower()}")
        text_parts.append(f"source_db={descriptor.source_db}")
    if repo:
        metadata["repo"] = repo
        relations.append(f"repo:{repo.lower()}")
    if rev:
        metadata["rev"] = rev

    return metadata, relations, " ".join(text_parts)


def build_node_scalar_fields(node_id: str) -> dict[str, Any]:
    descriptor = describe_node(node_id)
    scalar_fields: dict[str, Any] = {}
    if descriptor.source_db:
        scalar_fields["source_db"] = descriptor.source_db
    if descriptor.entity_type:
        scalar_fields["entity_type"] = descriptor.entity_type
    if descriptor.node_kind:
        scalar_fields["node_kind"] = descriptor.node_kind
    return scalar_fields


def push_embeddings_to_mycelia(
    store: Any,
    config: BridgeConfig,
    node_names: list[str],
    embeddings: Any,
) -> dict[str, Any]:
    if len(node_names) != len(embeddings):
        raise ValueError(f"Mismatch: {len(node_names)} node names vs {len(embeddings)} vectors")

    dimension = len(embeddings[0]) if len(embeddings) else 0
    if dimension <= 0:
        raise ValueError("No embeddings to push")

    store.ensure_collection(config.collection, dimension=dimension, modality="tabular")

    selected_ids = [str(node_id) for node_id in node_names if should_push_node(str(node_id), config.key_types)]
    if not selected_ids:
        return {
            "collection": config.collection,
            "selected": 0,
            "inserted": 0,
            "batch_count": 0,
            "dimension": dimension,
        }

    selected_indices = [
        idx
        for idx, node_id in enumerate(node_names)
        if should_push_node(str(node_id), config.key_types)
    ]
    total_inserted = 0
    batch_count = 0

    for batch_start in range(0, len(selected_indices), config.batch_size):
        batch_indices = selected_indices[batch_start:batch_start + config.batch_size]
        batch_vectors: dict[str, list[float]] = {}
        metadata_by_id: dict[str, dict[str, Any]] = {}
        relations_by_id: dict[str, list[str]] = {}
        text_by_id: dict[str, str] = {}
        scalar_fields_by_id: dict[str, dict[str, Any]] = {}

        for idx in batch_indices:
            node_id = str(node_names[idx])
            vector = embeddings[idx]
            batch_vectors[node_id] = vector.tolist() if hasattr(vector, "tolist") else list(vector)
            metadata, relations, text_snippet = build_node_payload(
                node_id,
                repo=config.repo,
                rev=config.rev,
            )
            metadata_by_id[node_id] = metadata
            relations_by_id[node_id] = relations
            text_by_id[node_id] = text_snippet
            scalar_fields_by_id[node_id] = build_node_scalar_fields(node_id)

        response = store.store_vectors(
            config.collection,
            batch_vectors,
            filter_tag=config.filter_tag,
            tenant_id=config.tenant_id,
            repo=config.repo,
            rev=config.rev,
            metadata_by_id=metadata_by_id,
            relations_by_id=relations_by_id,
            text_by_id=text_by_id,
            scalar_fields_by_id=scalar_fields_by_id,
        )
        inserted = response.get("inserted") if isinstance(response, dict) else None
        total_inserted += int(inserted) if isinstance(inserted, int) else len(batch_vectors)
        batch_count += 1

    verification = store.search_similar(
        config.collection,
        vector=embeddings[selected_indices[0]].tolist(),
        limit=5,
        tenant_id=config.tenant_id,
        repo=config.repo,
        rev=config.rev,
        filter_tag=config.filter_tag,
    )

    return {
        "collection": config.collection,
        "selected": len(selected_indices),
        "inserted": total_inserted,
        "batch_count": batch_count,
        "dimension": dimension,
        "verification": verification,
    }


def _push_embeddings_impl() -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    import torch

    from .mycelia_store import MyceliaStore

    config = load_bridge_config()

    print("[1/4] Loading node vocabulary...")
    table = pq.read_table(config.graph_path, columns=["subject_id", "object_id"])
    all_nodes = pa.chunked_array(table.column("subject_id").chunks + table.column("object_id").chunks)
    node_names = pc.unique(all_nodes).to_numpy(zero_copy_only=False)
    print(f"  {len(node_names):,} nodes")

    print("[2/4] Loading embeddings...")
    state = torch.load(config.weights_path, map_location="cpu", weights_only=True)
    embeddings = (state if isinstance(state, torch.Tensor) else list(state.values())[0]).float().numpy()
    print(f"  shape={embeddings.shape}")

    print("[3/4] Connecting to Mycelia...")
    store = MyceliaStore(
        config.base_url,
        api_key=config.api_key,
        namespace=config.namespace,
        vector_ingest_transport="flight",
    )

    print("[4/4] Pushing scoped graph vectors...")
    result = push_embeddings_to_mycelia(store, config, list(node_names), embeddings)
    print(result)
    return result


if modal is not None:  # pragma: no branch - import-time configuration
    app = modal.App("jcube-milvus-bridge")
    cache_vol = modal.Volume.from_name("jepa-cache", create_if_missing=True)
    data_vol = modal.Volume.from_name("jcube-data", create_if_missing=True)

    image = modal.Image.debian_slim(python_version="3.12").pip_install(
        "torch>=2.6",
        "numpy>=2.0",
        "pyarrow>=23.0.1",
    )

    @app.function(
        volumes={"/cache": cache_vol, "/data": data_vol},
        image=image,
        memory=65536,
        cpu=8,
        timeout=7200,
    )
    def push_embeddings() -> dict[str, Any]:
        return _push_embeddings_impl()

    @app.local_entrypoint()
    def main() -> None:
        push_embeddings.remote()

else:
    def push_embeddings() -> dict[str, Any]:  # type: ignore[no-redef]
        raise RuntimeError("modal is required to run jcube_bridge.push_embeddings")


__all__ = [
    "BridgeConfig",
    "NodeDescriptor",
    "build_node_payload",
    "build_node_scalar_fields",
    "describe_node",
    "load_bridge_config",
    "push_embeddings",
    "push_embeddings_to_mycelia",
    "should_push_node",
]
