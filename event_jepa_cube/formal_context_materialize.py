"""Materialize LEIO formal-context exports into JCUBE graph artifacts.

This restores the vendored JCUBE handoff expected by:

- `make jcube-materialize-formal-context`
- `scripts/jcube-modal.sh --pipeline formal-context-train`
- `scripts/jcube-modal.sh --pipeline formal-context-sync`

Input contract:
    .leio-code/exports/formal-context-v1/
        manifest.json
        objects.jsonl
        attributes.jsonl
        incidences.jsonl

Output contract:
    graph parquet with columns:
        subject_id, predicate, object_id, t_epoch, numeric_value,
        subject_kind, object_kind
    ontology parquet with columns:
        node_id, text_payload, category, label
    summary json with counts and paths
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - exercised via CLI/runtime
    raise ImportError("pyarrow is required for formal-context materialization") from exc


GRAPH_ARTIFACT_VERSION = 1


@dataclass
class FormalContextBundle:
    export_dir: Path
    manifest: dict[str, Any]
    objects: dict[str, dict[str, Any]]
    attributes: dict[str, dict[str, Any]]
    incidences_path: Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"expected JSON object rows in {path}")
            yield row


def _load_jsonl_map(path: Path, *, key: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in _iter_jsonl(path):
        row_key = str(row[key])
        rows[row_key] = row
    return rows


def load_formal_context_bundle(export_dir: str | Path) -> FormalContextBundle:
    root = Path(export_dir)
    manifest_path = root / "manifest.json"
    manifest = _load_json(manifest_path)

    objects_path = root / str(manifest.get("objects_path", "objects.jsonl"))
    attributes_path = root / str(manifest.get("attributes_path", "attributes.jsonl"))
    incidences_path = root / str(manifest.get("incidences_path", "incidences.jsonl"))

    for path in (objects_path, attributes_path, incidences_path):
        if not path.exists():
            raise FileNotFoundError(f"formal-context payload missing: {path}")

    return FormalContextBundle(
        export_dir=root,
        manifest=manifest,
        objects=_load_jsonl_map(objects_path, key="id"),
        attributes=_load_jsonl_map(attributes_path, key="id"),
        incidences_path=incidences_path,
    )


def describe_formal_context(export_dir: str | Path) -> dict[str, Any]:
    bundle = load_formal_context_bundle(export_dir)
    object_kinds = Counter()
    for row in bundle.objects.values():
        object_kinds[str(row.get("kind", "unknown"))] += 1

    attribute_categories = Counter()
    for row in bundle.attributes.values():
        attribute_categories[str(row.get("category", "attribute"))] += 1

    return {
        "version": GRAPH_ARTIFACT_VERSION,
        "formal_context_dir": str(bundle.export_dir),
        "manifest_version": bundle.manifest.get("version"),
        "object_count": len(bundle.objects),
        "attribute_count": len(bundle.attributes),
        "incidence_count": int(bundle.manifest.get("incidence_count", 0)),
        "object_kinds": dict(sorted(object_kinds.items())),
        "attribute_categories": dict(sorted(attribute_categories.items())),
        "described_at": _utc_now(),
    }


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _ontology_text(attr_id: str, category: str, label: str) -> str:
    normalized = label.replace(":", " ")
    return (
        f"formal context attribute id={attr_id} category={category} "
        f"label={label} normalized={normalized}"
    )


def _write_ontology_parquet(attributes: dict[str, dict[str, Any]], output_path: Path) -> int:
    _ensure_parent(output_path)
    ordered = list(attributes.items())
    table = pa.table(
        {
            "node_id": [attr_id for attr_id, _ in ordered],
            "text_payload": [
                _ontology_text(
                    attr_id,
                    str(row.get("category", "attribute")),
                    str(row.get("label", attr_id)),
                )
                for attr_id, row in ordered
            ],
            "category": [str(row.get("category", "attribute")) for _, row in ordered],
            "label": [str(row.get("label", attr_id)) for attr_id, row in ordered],
        }
    )
    pq.write_table(table, output_path, compression="zstd")
    return table.num_rows


def _flush_graph_batch(
    writer: pq.ParquetWriter | None,
    schema: pa.Schema,
    output_path: Path,
    columns: dict[str, list[Any]],
) -> pq.ParquetWriter | None:
    if not columns["subject_id"]:
        return writer
    _ensure_parent(output_path)
    table = pa.table(columns, schema=schema)
    if writer is None:
        writer = pq.ParquetWriter(output_path, schema=schema, compression="zstd")
    writer.write_table(table)
    for values in columns.values():
        values.clear()
    return writer


def materialize_formal_context(
    export_dir: str | Path,
    *,
    graph_out: str | Path,
    ontology_out: str | Path,
    summary_out: str | Path,
    batch_size: int = 100_000,
) -> dict[str, Any]:
    bundle = load_formal_context_bundle(export_dir)
    graph_path = Path(graph_out)
    ontology_path = Path(ontology_out)
    summary_path = Path(summary_out)

    graph_schema = pa.schema(
        [
            pa.field("subject_id", pa.string(), nullable=False),
            pa.field("predicate", pa.string(), nullable=False),
            pa.field("object_id", pa.string(), nullable=False),
            pa.field("t_epoch", pa.int64(), nullable=False),
            pa.field("numeric_value", pa.float32(), nullable=False),
            pa.field("subject_kind", pa.string(), nullable=False),
            pa.field("object_kind", pa.string(), nullable=False),
        ]
    )
    graph_columns: dict[str, list[Any]] = {
        "subject_id": [],
        "predicate": [],
        "object_id": [],
        "t_epoch": [],
        "numeric_value": [],
        "subject_kind": [],
        "object_kind": [],
    }

    predicate_counts: Counter[str] = Counter()
    node_ids: set[str] = set()
    missing_objects = 0
    missing_attributes = 0
    edge_count = 0
    writer: pq.ParquetWriter | None = None

    for incidence in _iter_jsonl(bundle.incidences_path):
        subject_id = str(incidence["object_id"])
        attribute_id = str(incidence["attribute_id"])

        subject = bundle.objects.get(subject_id)
        attribute = bundle.attributes.get(attribute_id)
        if subject is None:
            missing_objects += 1
            continue
        if attribute is None:
            missing_attributes += 1
            continue

        predicate = str(attribute.get("category", "attribute"))
        graph_columns["subject_id"].append(subject_id)
        graph_columns["predicate"].append(predicate)
        graph_columns["object_id"].append(attribute_id)
        graph_columns["t_epoch"].append(0)
        graph_columns["numeric_value"].append(1.0)
        graph_columns["subject_kind"].append(str(subject.get("kind", "unknown")))
        graph_columns["object_kind"].append("attribute")

        predicate_counts[predicate] += 1
        node_ids.add(subject_id)
        node_ids.add(attribute_id)
        edge_count += 1

        if len(graph_columns["subject_id"]) >= batch_size:
            writer = _flush_graph_batch(writer, graph_schema, graph_path, graph_columns)

    writer = _flush_graph_batch(writer, graph_schema, graph_path, graph_columns)
    if writer is None:
        _ensure_parent(graph_path)
        pq.write_table(pa.table({name: [] for name in graph_schema.names}, schema=graph_schema), graph_path, compression="zstd")
    else:
        writer.close()

    ontology_count = _write_ontology_parquet(bundle.attributes, ontology_path)

    summary = describe_formal_context(bundle.export_dir)
    summary.update(
        {
            "graph_path": str(graph_path),
            "ontology_path": str(ontology_path),
            "summary_path": str(summary_path),
            "graph_edge_count": edge_count,
            "graph_node_count": len(node_ids),
            "ontology_node_count": ontology_count,
            "predicate_count": len(predicate_counts),
            "predicate_histogram": dict(sorted(predicate_counts.items())),
            "missing_object_rows": missing_objects,
            "missing_attribute_rows": missing_attributes,
            "materialized_at": _utc_now(),
        }
    )

    _ensure_parent(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_dir", help="LEIO formal-context export directory")
    parser.add_argument("--graph-out", help="Output Parquet path for the JCUBE graph")
    parser.add_argument("--ontology-out", help="Output Parquet path for ontology nodes")
    parser.add_argument("--summary-out", help="Output JSON path for materialization summary")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100_000,
        help="How many graph edges to buffer before writing a Parquet batch",
    )
    parser.add_argument(
        "--describe-only",
        action="store_true",
        help="Print formal-context summary without writing Parquet artifacts",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    export_dir = Path(args.export_dir)

    if args.describe_only:
        _print_json(describe_formal_context(export_dir))
        return

    if not args.graph_out or not args.ontology_out or not args.summary_out:
        raise SystemExit(
            "--graph-out, --ontology-out, and --summary-out are required unless --describe-only is set"
        )

    summary = materialize_formal_context(
        export_dir,
        graph_out=args.graph_out,
        ontology_out=args.ontology_out,
        summary_out=args.summary_out,
        batch_size=args.batch_size,
    )
    _print_json(summary)


if __name__ == "__main__":
    main()
