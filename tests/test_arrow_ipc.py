"""Tests for the low-level Apache Arrow IPC engine (Rust + PyArrow fallback)."""

from __future__ import annotations

import math
import uuid
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as pa_ipc
import pytest

from event_jepa_cube import arrow_ipc
from event_jepa_cube.evaluation import SharedGraphStore
from event_jepa_cube.numpy_ops import (
    MmapEmbeddingStore,
    batch_cosine_similarity,
    compute_trend,
    streaming_ema_batch,
    weighted_aggregate,
)


def test_rust_arrow_ipc_backend_loaded():
    info = arrow_ipc.backend_info()
    assert info["pyarrow_version"]
    if arrow_ipc.has_rust_arrow_ipc():
        assert info["engine"] == "rust-arrow-ipc"
        assert info["rust_loaded"] is True
        assert info["rust_version"] == "0.3.0"
    else:
        assert info["engine"] == "pyarrow-ipc"
        assert info["rust_loaded"] is False


def test_sequence_matrix_vector_ipc_roundtrip():
    embs = [[1.0, 2.5, -3.0], [4.0, 5.5, 6.0]]
    ts = [10.0, 20.0]
    seq_ipc = arrow_ipc.sequence_to_ipc_stream(embs, ts)
    dec_embs, dec_ts = arrow_ipc.sequence_from_ipc_stream(seq_ipc)
    assert dec_embs == embs
    assert dec_ts == ts

    mat_ipc = arrow_ipc.matrix_to_ipc_stream(embs)
    assert arrow_ipc.matrix_from_ipc_stream(mat_ipc) == embs

    vec = [0.25, -1.5, 3.75]
    vec_ipc = arrow_ipc.vector_to_ipc_stream(vec)
    assert arrow_ipc.vector_from_ipc_stream(vec_ipc) == vec


def test_rust_vs_pyarrow_ipc_kernel_parity(monkeypatch: pytest.MonkeyPatch):
    embs = [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]
    ts = [1.0, 2.5, 4.0, 7.5]
    seq_ipc = arrow_ipc.sequence_to_ipc_stream(embs, ts)

    mat_a = [[1.0, 0.0, 2.0], [0.5, -1.0, 3.0]]
    mat_b = [[1.0, 1.0, 1.0], [-2.0, 0.5, 1.0], [0.0, 2.0, 4.0]]
    ipc_a = arrow_ipc.matrix_to_ipc_stream(mat_a)
    ipc_b = arrow_ipc.matrix_to_ipc_stream(mat_b)
    init_state_ipc = arrow_ipc.ema_state_to_ipc_stream([0.5, 0.5, 0.5], [0.1, 0.2, 0.3], [0.01, 0.02, 0.03])

    # Run with Rust Arrow IPC backend
    monkeypatch.delenv("JCUBE_NO_RUST", raising=False)
    agg_rust = arrow_ipc.vector_from_ipc_stream(arrow_ipc.weighted_aggregate_arrow_ipc(seq_ipc, alpha=0.25))
    trend_rust = arrow_ipc.vector_from_ipc_stream(arrow_ipc.compute_trend_arrow_ipc(seq_ipc, alpha=0.25))
    ema_ipc_r, last_ts_r, count_r = arrow_ipc.streaming_ema_batch_arrow_ipc(
        seq_ipc,
        state_ipc_bytes=init_state_ipc,
        alpha=0.2,
        last_ts=0.5,
        count=2,
    )
    ema_r = arrow_ipc.ema_state_from_ipc_stream(ema_ipc_r)
    sim_rust = arrow_ipc.matrix_from_ipc_stream(arrow_ipc.batch_cosine_similarity_arrow_ipc(ipc_a, ipc_b))
    topk_rust = arrow_ipc.top_k_cosine_arrow_ipc(ipc_b, [1.0, 1.0, 1.0], k=2)
    anom_rust = arrow_ipc.centroid_anomalies_arrow_ipc(ipc_b, [0, 1, 2], k=2)

    # Run with pure PyArrow IPC fallback
    monkeypatch.setenv("JCUBE_NO_RUST", "1")
    agg_py = arrow_ipc.vector_from_ipc_stream(arrow_ipc.weighted_aggregate_arrow_ipc(seq_ipc, alpha=0.25))
    trend_py = arrow_ipc.vector_from_ipc_stream(arrow_ipc.compute_trend_arrow_ipc(seq_ipc, alpha=0.25))
    ema_ipc_p, last_ts_p, count_p = arrow_ipc.streaming_ema_batch_arrow_ipc(
        seq_ipc,
        state_ipc_bytes=init_state_ipc,
        alpha=0.2,
        last_ts=0.5,
        count=2,
    )
    ema_p = arrow_ipc.ema_state_from_ipc_stream(ema_ipc_p)
    sim_py = arrow_ipc.matrix_from_ipc_stream(arrow_ipc.batch_cosine_similarity_arrow_ipc(ipc_a, ipc_b))
    topk_py = arrow_ipc.top_k_cosine_arrow_ipc(ipc_b, [1.0, 1.0, 1.0], k=2)
    anom_py = arrow_ipc.centroid_anomalies_arrow_ipc(ipc_b, [0, 1, 2], k=2)

    assert agg_rust == pytest.approx(agg_py, rel=1e-12, abs=1e-12)
    assert trend_rust == pytest.approx(trend_py, rel=1e-12, abs=1e-12)
    assert count_r == count_p == 6
    assert last_ts_r == pytest.approx(last_ts_p)
    for vec_r, vec_p in zip(ema_r, ema_p):
        assert vec_r == pytest.approx(vec_p, rel=1e-12, abs=1e-12)
    for row_r, row_p in zip(sim_rust, sim_py):
        assert row_r == pytest.approx(row_p, rel=1e-12, abs=1e-12)
    assert [idx for idx, _ in topk_rust] == [idx for idx, _ in topk_py]
    assert [s for _, s in topk_rust] == pytest.approx([s for _, s in topk_py], rel=1e-12, abs=1e-12)
    assert [(p, r) for p, r, _ in anom_rust] == [(p, r) for p, r, _ in anom_py]
    assert [s for _, _, s in anom_rust] == pytest.approx([s for _, _, s in anom_py], rel=1e-12, abs=1e-12)


def test_shared_graph_store_is_valid_arrow_ipc_file():
    name = f"ipc-graph-{uuid.uuid4().hex}"
    owner = SharedGraphStore(name, (3, 2), dtype="int64", create=True)
    try:
        assert owner.path.suffix == ".arrow"
        owner.edges[:] = [[10, 20], [30, 40], [50, 60]]
        owner.edges.flush()

        # Verify the backing file is a standard Apache Arrow IPC file
        table = owner.read_arrow()
        assert isinstance(table, pa.Table)
        assert table.column("values").to_pylist() == [10, 20, 30, 40, 50, 60]
        assert table.schema.metadata[b"jcube.shape"] == b"3,2"

        # Attach a second worker handle and mutate in-place
        worker = SharedGraphStore(name, (3, 2), dtype="int64", create=False)
        try:
            assert worker.edges.tolist() == [[10, 20], [30, 40], [50, 60]]
            worker.edges[1, 1] = 999
            worker.edges.flush()
            assert owner.edges[1, 1] == 999
            assert owner.read_arrow().column("values").to_pylist() == [10, 20, 30, 999, 50, 60]
        finally:
            worker.close()
    finally:
        owner.unlink()


def test_shared_graph_store_rust_and_pyarrow_interoperability(monkeypatch: pytest.MonkeyPatch):
    name = f"ipc-interop-{uuid.uuid4().hex}"
    monkeypatch.delenv("JCUBE_NO_RUST", raising=False)
    owner = SharedGraphStore(name, (2, 2), dtype="int64", create=True)
    try:
        owner.edges[:] = [[1, 2], [3, 4]]
        owner.edges.flush()

        # Attach via pure PyArrow fallback and mutate in-place
        monkeypatch.setenv("JCUBE_NO_RUST", "1")
        worker = SharedGraphStore(name, (2, 2), dtype="int64", create=False)
        try:
            assert worker.edges.tolist() == [[1, 2], [3, 4]]
            worker.edges[0, 1] = 777
            worker.edges.flush()
        finally:
            worker.close()

        monkeypatch.delenv("JCUBE_NO_RUST", raising=False)
        assert owner.edges[0, 1] == 777
        assert owner.read_arrow().column("values").to_pylist() == [1, 777, 3, 4]
    finally:
        owner.unlink()


def test_mmap_embedding_store_arrow_ipc_backing_and_roundtrip():
    store = MmapEmbeddingStore(embedding_dim=4, capacity=4)
    try:
        assert store.path is not None
        assert store.path.suffix == ".arrow"
        store.append([1.0, 2.0, 3.0, 4.0], 100.0)
        store.append_batch([[5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]], [200.0, 300.0])

        assert len(store) == 3
        table = store.to_arrow()
        assert table.num_rows == 3
        assert table.column("timestamp").to_pylist() == [100.0, 200.0, 300.0]

        ipc_bytes = store.to_ipc_bytes()
        restored = MmapEmbeddingStore.from_ipc_bytes(ipc_bytes)
        try:
            assert len(restored) == 3
            assert restored.get_timestamps() == [100.0, 200.0, 300.0]
            assert restored.get_embeddings() == [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ]
        finally:
            restored.close()
    finally:
        store.close()


def test_twin_api_load_embeddings_from_arrow_ipc_files(tmp_path: Path):
    from event_jepa_cube import twin_api

    graph_arrow = tmp_path / "graph.arrow"
    weights_arrow = tmp_path / "weights.arrow"

    graph_schema = pa.schema(
        [
            pa.field("subject_id", pa.string(), nullable=False),
            pa.field("object_id", pa.string(), nullable=False),
        ]
    )
    graph_batch = pa.RecordBatch.from_arrays(
        [
            pa.array(["GHO/ID_CD_INTERNACAO_1", "GHO/ID_CD_INTERNACAO_2"]),
            pa.array(["GHO/ID_CD_PACIENTE_10", "GHO/ID_CD_PACIENTE_20"]),
        ],
        schema=graph_schema,
    )
    with pa.OSFile(str(graph_arrow), "wb") as sink:
        with pa_ipc.new_file(sink, graph_schema) as writer:
            writer.write_batch(graph_batch)

    tensor = arrow_ipc.ArrowIpcSharedTensor(weights_arrow, (4, 3), dtype="float32", create=True)
    try:
        tensor.view[:] = [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        tensor.flush()
    finally:
        tensor.close()

    twin_api.reset_runtime_state()
    try:
        twin_api._load_embeddings(str(graph_arrow), str(weights_arrow))
        state = twin_api._state()
        assert state.emb_data["loaded"] is True
        assert state.emb_data["dim"] == 3
        assert len(state.emb_data["node_names"]) == 4
        assert math.isclose(float(state.emb_data["embeddings"][1][0]), 0.9, rel_tol=1e-5)
    finally:
        twin_api.reset_runtime_state()


def test_numpy_ops_high_level_functions_with_rust_arrow_ipc():
    embs = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    ts = [1.0, 2.0, 3.0]
    agg = weighted_aggregate(embs, ts, alpha=0.0)
    assert agg == pytest.approx([3.0, 4.0])

    trend = compute_trend(embs, ts, alpha=0.0)
    assert trend == pytest.approx([2.0, 2.0])

    sims = batch_cosine_similarity([[1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]])
    assert sims[0] == pytest.approx([1.0, 0.0])

    rep, last_ts, count, mean, m2 = streaming_ema_batch(
        [0.0, 0.0],
        embs,
        ts,
        alpha=0.5,
        last_timestamp=None,
        count=0,
        running_mean=[0.0, 0.0],
        running_m2=[0.0, 0.0],
    )
    assert count == 3
    assert last_ts == 3.0
    assert mean == pytest.approx([3.0, 4.0])
    assert m2 == pytest.approx([8.0, 8.0])
    assert len(rep) == 2
