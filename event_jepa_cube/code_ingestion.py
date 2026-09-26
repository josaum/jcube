"""Adapters for treating LEIO Code exports as first-class code ingestion input.

This module bridges the LEIO structural cache into the same rich payload dialect
used by the rest of the jcube/Mycelia stack:

- vectors keyed by stable symbol URNs
- canonical scope (tenant_id/repo/rev)
- metadata / relations / text_snippet
- filter-friendly scalar fields
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

NODE_VOCAB_CANDIDATES = (
    "node_vocab_full.json",
    "node_vocab_sample.json",
    "node_vocab.json",
)


def load_leio_query_cache(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError("LEIO query cache must be a JSON object")
    return payload


def load_node_vocab(path: str | Path) -> dict[str, int] | list[str]:
    resolved = Path(path)
    if resolved.is_dir():
        match = next((resolved / name for name in NODE_VOCAB_CANDIDATES if (resolved / name).exists()), None)
        if match is None:
            raise FileNotFoundError(f"no node vocab artifact found under {resolved}")
        resolved = match

    with resolved.open() as fh:
        payload = json.load(fh)
    if isinstance(payload, dict):
        reserved = {"num_nodes", "num_predicates", "version", "artifact_dir"}
        normalized = {
            str(key): int(value)
            for key, value in payload.items()
            if key not in reserved and isinstance(value, int)
        }
        if normalized:
            return normalized
    if isinstance(payload, list):
        return [str(item) for item in payload]
    raise ValueError(f"node vocab must be a list of IDs or a mapping of ID -> index: {resolved}")


def load_embedding_matrix(path: str | Path) -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise ImportError("torch is required to load embedding tensor artifacts") from exc

    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict):
        return next(iter(state.values()))
    return state


def build_embeddings_by_symbol_from_vocab(
    node_vocab: dict[str, int] | list[str],
    embedding_matrix: Any,
    *,
    symbol_ids: list[str] | None = None,
) -> dict[str, list[float]]:
    if isinstance(node_vocab, list):
        index_by_id = {node_id: idx for idx, node_id in enumerate(node_vocab)}
    else:
        index_by_id = dict(node_vocab)

    target_ids = symbol_ids or list(index_by_id.keys())
    embeddings: dict[str, list[float]] = {}
    for symbol_id in target_ids:
        index = index_by_id.get(symbol_id)
        if index is None:
            continue
        vector = embedding_matrix[index]
        embeddings[symbol_id] = vector.tolist() if hasattr(vector, "tolist") else list(vector)
    return embeddings


def infer_code_language(path: str) -> str:
    lower = path.lower()
    if lower.endswith(".rs"):
        return "rust"
    if lower.endswith(".py"):
        return "python"
    if lower.endswith(".tsx"):
        return "tsx"
    if lower.endswith(".ts"):
        return "typescript"
    if lower.endswith(".jsx"):
        return "jsx"
    if lower.endswith(".js"):
        return "javascript"
    return "unknown"


def build_code_symbol_payloads(
    query_cache: dict[str, Any],
    embeddings_by_symbol: dict[str, Any],
    *,
    tenant_id: str | None = None,
    repo: str | None = None,
    rev: str | None = None,
) -> dict[str, Any]:
    symbols = query_cache.get("symbols")
    if not isinstance(symbols, dict):
        raise ValueError("query cache is missing a symbols map")

    callers_by_symbol = query_cache.get("callers_by_symbol")
    callees_by_symbol = query_cache.get("callees_by_symbol")
    callsites_by_symbol = query_cache.get("callsites_by_symbol")
    effective_rev = rev or query_cache.get("revision")
    scope = {key: value for key, value in {"tenant_id": tenant_id, "repo": repo, "rev": effective_rev}.items() if value}

    vectors: dict[str, list[float]] = {}
    metadata_by_id: dict[str, dict[str, Any]] = {}
    relations_by_id: dict[str, list[str]] = {}
    text_by_id: dict[str, str] = {}
    scalar_fields_by_id: dict[str, dict[str, Any]] = {}

    for symbol_id, vector in embeddings_by_symbol.items():
        entry = symbols.get(symbol_id)
        if not isinstance(entry, dict):
            continue

        path = str(entry.get("path", ""))
        kind = str(entry.get("kind", "unknown"))
        language = infer_code_language(path)
        name = str(entry.get("name", ""))
        qual_name = str(entry.get("qual_name", name))
        line = entry.get("line")

        values = vector.tolist() if hasattr(vector, "tolist") else list(vector)
        vectors[symbol_id] = values

        caller_count = len(callers_by_symbol.get(symbol_id, [])) if isinstance(callers_by_symbol, dict) else 0
        callee_count = len(callees_by_symbol.get(symbol_id, [])) if isinstance(callees_by_symbol, dict) else 0
        callsite_count = len(callsites_by_symbol.get(symbol_id, [])) if isinstance(callsites_by_symbol, dict) else 0

        metadata_by_id[symbol_id] = {
            "iri": entry.get("iri", symbol_id),
            "name": name,
            "qual_name": qual_name,
            "kind": kind,
            "path": path,
            "line": line,
            "language": language,
            "caller_count": caller_count,
            "callee_count": callee_count,
            "callsite_count": callsite_count,
        }

        relations = [
            "kind:code_symbol",
            f"symbol_kind:{kind}",
            f"language:{language}",
        ]
        if path:
            relations.append(f"path:{path}")
        if repo:
            relations.append(f"repo:{repo.lower()}")
        relations_by_id[symbol_id] = relations

        text_by_id[symbol_id] = (
            f"code symbol name={name} qual_name={qual_name} kind={kind} language={language} "
            f"path={path} line={line} callers={caller_count} callees={callee_count} callsites={callsite_count}"
        )

        scalar_fields_by_id[symbol_id] = {
            "path": path or None,
            "language": language,
            "symbol_kind": kind,
        }

    return {
        "scope": scope,
        "vectors": vectors,
        "metadata_by_id": metadata_by_id,
        "relations_by_id": relations_by_id,
        "text_by_id": text_by_id,
        "scalar_fields_by_id": scalar_fields_by_id,
    }


def sync_leio_query_cache_to_mycelia(
    store: Any,
    collection: str,
    query_cache: dict[str, Any] | str | Path,
    embeddings_by_symbol: dict[str, Any],
    *,
    tenant_id: str | None = None,
    repo: str | None = None,
    rev: str | None = None,
) -> dict[str, Any]:
    cache = load_leio_query_cache(query_cache) if isinstance(query_cache, (str, Path)) else query_cache
    payload = build_code_symbol_payloads(
        cache,
        embeddings_by_symbol,
        tenant_id=tenant_id,
        repo=repo,
        rev=rev,
    )

    vectors = payload["vectors"]
    if not vectors:
        return {
            "stored": 0,
            "symbols": 0,
            "scope": payload["scope"],
        }

    first_vector = next(iter(vectors.values()))
    store.ensure_collection(collection, dimension=len(first_vector), modality="tabular")
    result = store.store_code_symbols(
        collection,
        vectors,
        tenant_id=tenant_id or payload["scope"].get("tenant_id"),
        repo=repo or payload["scope"].get("repo"),
        rev=rev or payload["scope"].get("rev"),
        metadata_by_id=payload["metadata_by_id"],
        relations_by_id=payload["relations_by_id"],
        text_by_id=payload["text_by_id"],
        scalar_fields_by_id=payload["scalar_fields_by_id"],
    )
    return {
        "stored": result.get("inserted", len(vectors)) if isinstance(result, dict) else len(vectors),
        "symbols": len(vectors),
        "scope": payload["scope"],
        "result": result,
    }


def sync_leio_query_cache_from_artifacts(
    store: Any,
    collection: str,
    query_cache: dict[str, Any] | str | Path,
    *,
    node_vocab: dict[str, int] | list[str] | str | Path,
    embedding_matrix: Any,
    tenant_id: str | None = None,
    repo: str | None = None,
    rev: str | None = None,
) -> dict[str, Any]:
    cache = load_leio_query_cache(query_cache) if isinstance(query_cache, (str, Path)) else query_cache
    vocab = load_node_vocab(node_vocab) if isinstance(node_vocab, (str, Path)) else node_vocab
    symbol_ids = list(cache.get("symbols", {}).keys()) if isinstance(cache.get("symbols"), dict) else None
    embeddings_by_symbol = build_embeddings_by_symbol_from_vocab(vocab, embedding_matrix, symbol_ids=symbol_ids)
    return sync_leio_query_cache_to_mycelia(
        store,
        collection,
        cache,
        embeddings_by_symbol,
        tenant_id=tenant_id,
        repo=repo,
        rev=rev,
    )


__all__ = [
    "build_code_symbol_payloads",
    "build_embeddings_by_symbol_from_vocab",
    "infer_code_language",
    "load_embedding_matrix",
    "load_leio_query_cache",
    "load_node_vocab",
    "sync_leio_query_cache_from_artifacts",
    "sync_leio_query_cache_to_mycelia",
]
