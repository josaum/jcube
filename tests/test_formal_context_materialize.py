from __future__ import annotations

import json

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from event_jepa_cube.formal_context_materialize import describe_formal_context, materialize_formal_context


def _write_jsonl(path, rows) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_materialize_formal_context_round_trip(tmp_path) -> None:
    export_dir = tmp_path / "formal-context-v1"
    export_dir.mkdir()

    objects = [
        {
            "id": "file:src/main.py",
            "kind": "file",
            "label": "src/main.py",
            "path": "src/main.py",
            "line": 1,
            "language": "python",
            "metadata": {},
        },
        {
            "id": "symbol:python:src/main.py:10:function:main",
            "kind": "symbol",
            "label": "main",
            "path": "src/main.py",
            "line": 10,
            "language": "python",
            "metadata": {"symbol_kind": "function"},
        },
    ]
    attributes = [
        {"id": "attr:kind:file", "label": "kind:file", "category": "kind"},
        {"id": "attr:topic:entrypoint", "label": "topic:entrypoint", "category": "topic"},
    ]
    incidences = [
        {"object_id": "file:src/main.py", "attribute_id": "attr:kind:file"},
        {"object_id": "symbol:python:src/main.py:10:function:main", "attribute_id": "attr:kind:file"},
        {"object_id": "symbol:python:src/main.py:10:function:main", "attribute_id": "attr:topic:entrypoint"},
    ]
    manifest = {
        "version": 1,
        "repo_root": str(tmp_path),
        "indexed_at": "2026-05-27T00:00:00Z",
        "exported_at": "2026-05-27T00:00:01Z",
        "objects_path": "objects.jsonl",
        "attributes_path": "attributes.jsonl",
        "incidences_path": "incidences.jsonl",
        "object_count": len(objects),
        "attribute_count": len(attributes),
        "incidence_count": len(incidences),
        "object_kinds": {"file": 1, "symbol": 1},
    }

    _write_jsonl(export_dir / "objects.jsonl", objects)
    _write_jsonl(export_dir / "attributes.jsonl", attributes)
    _write_jsonl(export_dir / "incidences.jsonl", incidences)
    (export_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    graph_out = export_dir / "jcube" / "leio_code_graph_v1.parquet"
    ontology_out = export_dir / "jcube" / "leio_code_ontology_nodes_v1.parquet"
    summary_out = export_dir / "jcube" / "summary.json"

    described = describe_formal_context(export_dir)
    assert described["object_count"] == 2
    assert described["attribute_count"] == 2
    assert described["incidence_count"] == 3

    summary = materialize_formal_context(
        export_dir,
        graph_out=graph_out,
        ontology_out=ontology_out,
        summary_out=summary_out,
        batch_size=2,
    )

    graph = pq.read_table(graph_out).to_pydict()
    assert graph["subject_id"] == [
        "file:src/main.py",
        "symbol:python:src/main.py:10:function:main",
        "symbol:python:src/main.py:10:function:main",
    ]
    assert graph["predicate"] == ["kind", "kind", "topic"]
    assert graph["object_id"] == [
        "attr:kind:file",
        "attr:kind:file",
        "attr:topic:entrypoint",
    ]
    assert graph["numeric_value"] == [1.0, 1.0, 1.0]

    ontology = pq.read_table(ontology_out).to_pydict()
    assert ontology["node_id"] == ["attr:kind:file", "attr:topic:entrypoint"]
    assert ontology["category"] == ["kind", "topic"]
    assert "normalized=topic entrypoint" in ontology["text_payload"][1]

    written_summary = json.loads(summary_out.read_text(encoding="utf-8"))
    assert written_summary == summary
    assert summary["graph_edge_count"] == 3
    assert summary["graph_node_count"] == 4
    assert summary["ontology_node_count"] == 2
    assert summary["predicate_histogram"] == {"kind": 2, "topic": 1}
