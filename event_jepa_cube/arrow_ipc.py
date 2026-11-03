"""Low-level Apache Arrow IPC compute & zero-copy shared memory bridge (Rust + PyArrow).

Provides:
1. Native Rust Arrow IPC engine bindings (`libjcube_arrow_ipc`) via `ctypes` with
   automatic fallback to pure PyArrow IPC when the compiled library is absent or
   disabled via `JCUBE_NO_RUST=1`.
2. Zero-copy cross-process memory-mapped tensors stored inside standard uncompressed
   Apache Arrow IPC files (`.arrow`), allowing in-place mutation (`view[i, j] = v`,
   `view[:] = rows`) without requiring `numpy`.
3. Arrow IPC stream/file kernels for sequence aggregation, OLS trend, streaming EMA,
   batch cosine similarity, top-k cosine search, and centroid anomaly detection.
"""

from __future__ import annotations

import ctypes
import math
import mmap
import os
import platform
import shutil
import subprocess
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.ipc as pa_ipc

_DTYPE_SPEC: dict[str, tuple[pa.DataType, str, int, str]] = {
    "int64": (pa.int64(), "q", 8, "int64"),
    "i64": (pa.int64(), "q", 8, "int64"),
    "<i8": (pa.int64(), "q", 8, "int64"),
    "int32": (pa.int32(), "i", 4, "int32"),
    "i32": (pa.int32(), "i", 4, "int32"),
    "<i4": (pa.int32(), "i", 4, "int32"),
    "float64": (pa.float64(), "d", 8, "float64"),
    "f64": (pa.float64(), "d", 8, "float64"),
    "<f8": (pa.float64(), "d", 8, "float64"),
    "float32": (pa.float32(), "f", 4, "float32"),
    "f32": (pa.float32(), "f", 4, "float32"),
    "<f4": (pa.float32(), "f", 4, "float32"),
    "uint8": (pa.uint8(), "B", 1, "uint8"),
    "u8": (pa.uint8(), "B", 1, "uint8"),
    "|u1": (pa.uint8(), "B", 1, "uint8"),
}


def normalize_dtype(dtype: str) -> tuple[pa.DataType, str, int, str]:
    """Return `(arrow_type, struct_format, elem_bytes, canonical_name)` for `dtype`."""
    key = str(dtype).strip().lower()
    if key not in _DTYPE_SPEC:
        raise ValueError(f"Unsupported Arrow IPC tensor dtype: {dtype!r}")
    return _DTYPE_SPEC[key]


class _IpcBuffer(ctypes.Structure):
    _fields_ = [
        ("ptr", ctypes.POINTER(ctypes.c_uint8)),
        ("len", ctypes.c_size_t),
        ("cap", ctypes.c_size_t),
        ("status", ctypes.c_int32),
    ]


class _TensorFileLayout(ctypes.Structure):
    _fields_ = [
        ("data_offset", ctypes.c_size_t),
        ("data_byte_len", ctypes.c_size_t),
        ("file_byte_len", ctypes.c_size_t),
        ("elem_size", ctypes.c_size_t),
        ("num_elements", ctypes.c_size_t),
        ("status", ctypes.c_int32),
    ]


class _StreamingEmaResult(ctypes.Structure):
    _fields_ = [
        ("state_ipc", _IpcBuffer),
        ("last_timestamp", ctypes.c_double),
        ("count", ctypes.c_uint64),
    ]


_RUST_LIB: ctypes.CDLL | None = None
_RUST_LIB_PATH: Path | None = None
_RUST_LOAD_ATTEMPTED: bool = False


def _shared_lib_filename() -> str:
    sys_name = platform.system().lower()
    if sys_name == "darwin":
        return "libjcube_arrow_ipc.dylib"
    if sys_name == "windows":
        return "jcube_arrow_ipc.dll"
    return "libjcube_arrow_ipc.so"


def _configure_cdll(lib: ctypes.CDLL) -> ctypes.CDLL:
    lib.jcube_ipc_version.argtypes = []
    lib.jcube_ipc_version.restype = ctypes.c_char_p

    lib.jcube_ipc_free_buffer.argtypes = [_IpcBuffer]
    lib.jcube_ipc_free_buffer.restype = None

    lib.jcube_ipc_create_tensor_file.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        ctypes.c_char_p,
    ]
    lib.jcube_ipc_create_tensor_file.restype = _TensorFileLayout

    lib.jcube_ipc_inspect_tensor_file.argtypes = [ctypes.c_char_p]
    lib.jcube_ipc_inspect_tensor_file.restype = _TensorFileLayout

    lib.jcube_ipc_weighted_aggregate.argtypes = [
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_double,
    ]
    lib.jcube_ipc_weighted_aggregate.restype = _IpcBuffer

    lib.jcube_ipc_compute_trend.argtypes = [
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_double,
    ]
    lib.jcube_ipc_compute_trend.restype = _IpcBuffer

    lib.jcube_ipc_streaming_ema_batch.argtypes = [
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int32,
        ctypes.c_uint64,
    ]
    lib.jcube_ipc_streaming_ema_batch.restype = _StreamingEmaResult

    lib.jcube_ipc_batch_cosine_similarity.argtypes = [
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.jcube_ipc_batch_cosine_similarity.restype = _IpcBuffer

    lib.jcube_ipc_top_k_cosine.argtypes = [
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_ssize_t,
    ]
    lib.jcube_ipc_top_k_cosine.restype = _IpcBuffer

    lib.jcube_ipc_centroid_anomalies.argtypes = [
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    lib.jcube_ipc_centroid_anomalies.restype = _IpcBuffer
    return lib


def _get_rust_lib() -> ctypes.CDLL | None:
    """Return the loaded Rust Arrow IPC shared library if available and enabled."""
    global _RUST_LIB, _RUST_LIB_PATH, _RUST_LOAD_ATTEMPTED
    if os.environ.get("JCUBE_NO_RUST", "").strip().lower() in {"1", "true", "yes"}:
        return None
    if _RUST_LIB is not None:
        return _RUST_LIB
    if _RUST_LOAD_ATTEMPTED:
        return None
    _RUST_LOAD_ATTEMPTED = True

    repo_root = Path(__file__).resolve().parent.parent
    lib_name = _shared_lib_filename()
    candidates: list[Path] = []
    env_path = os.environ.get("JCUBE_RUST_LIB_PATH", "").strip()
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(
        [
            repo_root / "rust" / "target" / "release" / lib_name,
            repo_root / "rust" / "target" / "debug" / lib_name,
            Path(__file__).resolve().parent / lib_name,
        ]
    )

    for candidate in candidates:
        if candidate.is_file():
            try:
                _RUST_LIB = _configure_cdll(ctypes.CDLL(str(candidate)))
                _RUST_LIB_PATH = candidate
                return _RUST_LIB
            except OSError:
                continue

    auto_build = os.environ.get("JCUBE_AUTO_BUILD_RUST", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    manifest = repo_root / "rust" / "Cargo.toml"
    cargo_bin = shutil.which("cargo")
    if auto_build and manifest.is_file() and cargo_bin is not None:
        try:
            subprocess.run(
                [cargo_bin, "build", "--release", "--manifest-path", str(manifest)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=120,
            )
            built = repo_root / "rust" / "target" / "release" / lib_name
            if built.is_file():
                _RUST_LIB = _configure_cdll(ctypes.CDLL(str(built)))
                _RUST_LIB_PATH = built
                return _RUST_LIB
        except Exception:
            return None

    return None


def _consume_ipc_buffer(lib: ctypes.CDLL, buf: _IpcBuffer) -> bytes:
    try:
        raw = ctypes.string_at(buf.ptr, buf.len) if buf.ptr and buf.len > 0 else b""
        if buf.status != 0:
            err_msg = raw.decode("utf-8", errors="replace") or "Rust Arrow IPC kernel error"
            raise RuntimeError(err_msg)
        return raw
    finally:
        lib.jcube_ipc_free_buffer(buf)


def has_rust_arrow_ipc() -> bool:
    """Return `True` when the native Rust Arrow IPC library is active."""
    return _get_rust_lib() is not None


def backend_info() -> dict[str, Any]:
    """Return diagnostic metadata about the active Arrow IPC backend."""
    lib = _get_rust_lib()
    version = "pyarrow-fallback"
    if lib is not None:
        raw_ver = lib.jcube_ipc_version()
        if raw_ver:
            version = raw_ver.decode("utf-8", errors="replace")
    return {
        "engine": "rust-arrow-ipc" if lib is not None else "pyarrow-ipc",
        "rust_loaded": lib is not None,
        "rust_library_path": str(_RUST_LIB_PATH) if _RUST_LIB_PATH is not None else None,
        "rust_version": version,
        "pyarrow_version": pa.__version__,
    }


# ============================================================================
# Arrow IPC Stream / File Serialization Helpers
# ============================================================================


def sequence_to_ipc_stream(
    embeddings: Sequence[Sequence[float]],
    timestamps: Sequence[float],
) -> bytes:
    """Serialize `(timestamps, embeddings)` into an Apache Arrow IPC stream."""
    n = len(embeddings)
    if n != len(timestamps):
        raise ValueError(f"Length mismatch: {n} embeddings vs {len(timestamps)} timestamps")
    dim = len(embeddings[0]) if n > 0 else 0
    ts_arr = pa.array([float(t) for t in timestamps], type=pa.float64())
    flat_vals = [float(x) for row in embeddings for x in row]
    val_arr = pa.array(flat_vals, type=pa.float64())
    fsl_type = pa.list_(pa.field("item", pa.float64(), nullable=False), dim)
    emb_arr = pa.FixedSizeListArray.from_arrays(val_arr, type=fsl_type)
    schema = pa.schema(
        [
            pa.field("timestamp", pa.float64(), nullable=False),
            pa.field("embedding", fsl_type, nullable=False),
        ]
    )
    batch = pa.RecordBatch.from_arrays([ts_arr, emb_arr], schema=schema)
    sink = pa.BufferOutputStream()
    with pa_ipc.new_stream(sink, schema) as writer:
        writer.write_batch(batch)
    return cast(bytes, sink.getvalue().to_pybytes())


def sequence_from_ipc_stream(ipc_bytes: bytes) -> tuple[list[list[float]], list[float]]:
    """Deserialize `(embeddings, timestamps)` from an Apache Arrow IPC stream or file."""
    if not ipc_bytes:
        return [], []
    reader: Any
    if ipc_bytes[:6] == b"ARROW1":
        reader = pa_ipc.open_file(pa.py_buffer(ipc_bytes))
        batches = [reader.get_batch(i) for i in range(reader.num_record_batches)]
    else:
        reader = pa_ipc.open_stream(pa.py_buffer(ipc_bytes))
        batches = list(reader)

    timestamps: list[float] = []
    embeddings: list[list[float]] = []
    for batch in batches:
        if batch.num_rows == 0:
            continue
        ts_col = batch.column(0).to_pylist()
        timestamps.extend(float(t) for t in ts_col)
        if batch.num_columns == 2 and isinstance(batch.column(1).type, pa.FixedSizeListType):
            emb_col = batch.column(1).to_pylist()
            embeddings.extend([[float(x) for x in row] for row in emb_col])
        else:
            cols = [batch.column(c).to_pylist() for c in range(1, batch.num_columns)]
            for r in range(batch.num_rows):
                embeddings.append([float(col[r]) for col in cols])
    return embeddings, timestamps


def matrix_to_ipc_stream(
    matrix: Sequence[Sequence[float]],
    field_name: str = "matrix",
) -> bytes:
    """Serialize a 2D float matrix into an Apache Arrow IPC stream."""
    n = len(matrix)
    dim = len(matrix[0]) if n > 0 else 0
    if n == 0 or dim == 0:
        schema = pa.schema([pa.field(field_name, pa.float64(), nullable=False)])
        batch = pa.RecordBatch.from_arrays([pa.array([], type=pa.float64())], schema=schema)
        sink = pa.BufferOutputStream()
        with pa_ipc.new_stream(sink, schema) as writer:
            writer.write_batch(batch)
        return cast(bytes, sink.getvalue().to_pybytes())

    flat_vals = [float(x) for row in matrix for x in row]
    val_arr = pa.array(flat_vals, type=pa.float64())
    fsl_type = pa.list_(pa.field("item", pa.float64(), nullable=False), dim)
    fsl_arr = pa.FixedSizeListArray.from_arrays(val_arr, type=fsl_type)
    schema = pa.schema([pa.field(field_name, fsl_type, nullable=False)])
    batch = pa.RecordBatch.from_arrays([fsl_arr], schema=schema)
    sink = pa.BufferOutputStream()
    with pa_ipc.new_stream(sink, schema) as writer:
        writer.write_batch(batch)
    return cast(bytes, sink.getvalue().to_pybytes())


def matrix_from_ipc_stream(ipc_bytes: bytes) -> list[list[float]]:
    """Deserialize a 2D float matrix from an Apache Arrow IPC stream or file."""
    if not ipc_bytes:
        return []
    reader: Any
    if ipc_bytes[:6] == b"ARROW1":
        reader = pa_ipc.open_file(pa.py_buffer(ipc_bytes))
        schema = reader.schema
        batches = [reader.get_batch(i) for i in range(reader.num_record_batches)]
    else:
        reader = pa_ipc.open_stream(pa.py_buffer(ipc_bytes))
        schema = reader.schema
        batches = list(reader)

    meta = schema.metadata or {}
    shape_bytes = meta.get(b"jcube.shape")
    rows: list[list[float]] = []
    for batch in batches:
        if batch.num_rows == 0 or batch.num_columns == 0:
            continue
        col0 = batch.column(0)
        if batch.num_columns == 1 and isinstance(col0.type, pa.FixedSizeListType):
            for row in col0.to_pylist():
                rows.append([float(x) for x in row])
        elif batch.num_columns == 1 and shape_bytes is not None:
            dims = [int(x) for x in shape_bytes.decode("utf-8").split(",") if x.strip()]
            flat = [float(x) for x in col0.to_pylist()]
            if len(dims) == 2 and dims[0] * dims[1] == len(flat):
                d = dims[1]
                for r in range(dims[0]):
                    rows.append(flat[r * d : (r + 1) * d])
        else:
            cols = [batch.column(c).to_pylist() for c in range(batch.num_columns)]
            for r in range(batch.num_rows):
                rows.append([float(c[r]) for c in cols])
    return rows


def vector_to_ipc_stream(
    values: Sequence[float],
    field_name: str = "values",
) -> bytes:
    """Serialize a 1D float vector into an Apache Arrow IPC stream."""
    arr = pa.array([float(v) for v in values], type=pa.float64())
    schema = pa.schema([pa.field(field_name, pa.float64(), nullable=False)])
    batch = pa.RecordBatch.from_arrays([arr], schema=schema)
    sink = pa.BufferOutputStream()
    with pa_ipc.new_stream(sink, schema) as writer:
        writer.write_batch(batch)
    return cast(bytes, sink.getvalue().to_pybytes())


def vector_from_ipc_stream(ipc_bytes: bytes) -> list[float]:
    """Deserialize a 1D float vector from an Apache Arrow IPC stream or file."""
    if not ipc_bytes:
        return []
    reader: Any
    if ipc_bytes[:6] == b"ARROW1":
        reader = pa_ipc.open_file(pa.py_buffer(ipc_bytes))
        batches = [reader.get_batch(i) for i in range(reader.num_record_batches)]
    else:
        reader = pa_ipc.open_stream(pa.py_buffer(ipc_bytes))
        batches = list(reader)
    out: list[float] = []
    for batch in batches:
        if batch.num_rows == 0 or batch.num_columns == 0:
            continue
        out.extend(float(v) for v in batch.column(0).to_pylist())
    return out


# ============================================================================
# Low-Level Arrow IPC Compute Kernels (Rust C-ABI + PyArrow IPC Fallback)
# ============================================================================


def weighted_aggregate_arrow_ipc(ipc_bytes: bytes, alpha: float = 0.1) -> bytes:
    """Compute time-decay weighted aggregate over an Arrow IPC sequence stream."""
    lib = _get_rust_lib()
    if lib is not None:
        buf = lib.jcube_ipc_weighted_aggregate(ipc_bytes, len(ipc_bytes), float(alpha))
        return _consume_ipc_buffer(lib, buf)

    embeddings, timestamps = sequence_from_ipc_stream(ipc_bytes)
    if not embeddings:
        return vector_to_ipc_stream([], field_name="representation")
    dim = len(embeddings[0])
    max_t = max(timestamps)
    weights = [math.exp(-alpha * (max_t - t)) for t in timestamps]
    total_w = sum(weights) or 1.0
    result = [0.0] * dim
    for emb, w in zip(embeddings, weights):
        for j in range(dim):
            result[j] += emb[j] * w
    return vector_to_ipc_stream([v / total_w for v in result], field_name="representation")


def compute_trend_arrow_ipc(ipc_bytes: bytes, alpha: float = 1.0) -> bytes:
    """Compute exponentially-weighted delta trend over an Arrow IPC sequence stream."""
    lib = _get_rust_lib()
    if lib is not None:
        buf = lib.jcube_ipc_compute_trend(ipc_bytes, len(ipc_bytes), float(alpha))
        return _consume_ipc_buffer(lib, buf)

    embeddings, timestamps = sequence_from_ipc_stream(ipc_bytes)
    if not embeddings:
        return vector_to_ipc_stream([], field_name="trend")
    dim = len(embeddings[0])
    if len(embeddings) < 2:
        return vector_to_ipc_stream([0.0] * dim, field_name="trend")

    deltas = [[embeddings[j][d] - embeddings[j - 1][d] for d in range(dim)] for j in range(1, len(embeddings))]
    delta_times = timestamps[1:]
    t_max = max(delta_times)
    weights = [math.exp(-alpha * (t_max - t)) for t in delta_times]
    total_w = sum(weights) or 1.0
    trend = [0.0] * dim
    for delta, w in zip(deltas, weights):
        for d in range(dim):
            trend[d] += delta[d] * w
    return vector_to_ipc_stream([v / total_w for v in trend], field_name="trend")


def ema_state_to_ipc_stream(
    representation: Sequence[float],
    running_mean: Sequence[float] | None = None,
    running_m2: Sequence[float] | None = None,
) -> bytes:
    """Serialize `(representation, running_mean, running_m2)` into an Arrow IPC stream."""
    dim = len(representation)
    mean_vals = [float(x) for x in running_mean] if running_mean is not None else [0.0] * dim
    m2_vals = [float(x) for x in running_m2] if running_m2 is not None else [0.0] * dim
    schema = pa.schema(
        [
            pa.field("representation", pa.float64(), nullable=False),
            pa.field("running_mean", pa.float64(), nullable=False),
            pa.field("running_m2", pa.float64(), nullable=False),
        ]
    )
    batch = pa.RecordBatch.from_arrays(
        [
            pa.array([float(x) for x in representation], type=pa.float64()),
            pa.array(mean_vals, type=pa.float64()),
            pa.array(m2_vals, type=pa.float64()),
        ],
        schema=schema,
    )
    sink = pa.BufferOutputStream()
    with pa_ipc.new_stream(sink, schema) as writer:
        writer.write_batch(batch)
    return cast(bytes, sink.getvalue().to_pybytes())


def ema_state_from_ipc_stream(ipc_bytes: bytes) -> tuple[list[float], list[float], list[float]]:
    """Deserialize `(representation, running_mean, running_m2)` from an Arrow IPC stream."""
    if not ipc_bytes:
        return [], [], []
    reader = pa_ipc.open_stream(pa.py_buffer(ipc_bytes))
    repr_out: list[float] = []
    mean_out: list[float] = []
    m2_out: list[float] = []
    for batch in reader:
        if batch.num_rows == 0 or batch.num_columns == 0:
            continue
        repr_out.extend(float(v) for v in batch.column(0).to_pylist())
        if batch.num_columns >= 3:
            mean_out.extend(float(v) for v in batch.column(1).to_pylist())
            m2_out.extend(float(v) for v in batch.column(2).to_pylist())
    if not mean_out and repr_out:
        mean_out = [0.0] * len(repr_out)
    if not m2_out and repr_out:
        m2_out = [0.0] * len(repr_out)
    return repr_out, mean_out, m2_out


def streaming_ema_batch_arrow_ipc(
    ipc_bytes: bytes,
    state_ipc_bytes: bytes = b"",
    alpha: float = 0.1,
    last_ts: float | None = None,
    count: int = 0,
) -> tuple[bytes, float | None, int]:
    """Update a streaming time-decayed EMA + Welford state over an Arrow IPC sequence stream."""
    lib = _get_rust_lib()
    if lib is not None:
        res = lib.jcube_ipc_streaming_ema_batch(
            ipc_bytes,
            len(ipc_bytes),
            state_ipc_bytes,
            len(state_ipc_bytes),
            float(alpha),
            float(last_ts if last_ts is not None else 0.0),
            1 if last_ts is not None else 0,
            int(count),
        )
        out_ipc = _consume_ipc_buffer(lib, res.state_ipc)
        new_count = int(res.count)
        new_last_ts = float(res.last_timestamp) if new_count > 0 else last_ts
        return out_ipc, new_last_ts, new_count

    embeddings, timestamps = sequence_from_ipc_stream(ipc_bytes)
    repr_list, mean_list, m2_list = ema_state_from_ipc_stream(state_ipc_bytes) if state_ipc_bytes else ([], [], [])
    if not embeddings:
        return (
            ema_state_to_ipc_stream(repr_list, mean_list, m2_list),
            last_ts,
            count,
        )

    dim = len(embeddings[0])
    if len(repr_list) != dim:
        repr_list = [0.0] * dim
    if len(mean_list) != dim:
        mean_list = [0.0] * dim
    if len(m2_list) != dim:
        m2_list = [0.0] * dim

    cur_count = count
    prev_ts = last_ts
    for emb_item, t_val in zip(embeddings, timestamps):
        if cur_count == 0:
            repr_list = list(emb_item)
        else:
            dt = t_val - prev_ts if prev_ts is not None else 0.0
            decay = math.exp(-alpha * dt)
            repr_list = [decay * r + (1.0 - decay) * e for r, e in zip(repr_list, emb_item)]
        cur_count += 1
        prev_ts = t_val
        for d in range(dim):
            delta = emb_item[d] - mean_list[d]
            mean_list[d] += delta / cur_count
            delta2 = emb_item[d] - mean_list[d]
            m2_list[d] += delta * delta2

    return (
        ema_state_to_ipc_stream(repr_list, mean_list, m2_list),
        prev_ts,
        cur_count,
    )


def batch_cosine_similarity_arrow_ipc(ipc_a: bytes, ipc_b: bytes) -> bytes:
    """Compute pairwise cosine similarity matrix (`N_a x N_b`) over Arrow IPC streams."""
    lib = _get_rust_lib()
    if lib is not None:
        buf = lib.jcube_ipc_batch_cosine_similarity(ipc_a, len(ipc_a), ipc_b, len(ipc_b))
        return _consume_ipc_buffer(lib, buf)

    mat_a = matrix_from_ipc_stream(ipc_a)
    mat_b = matrix_from_ipc_stream(ipc_b)
    if not mat_a or not mat_b:
        return matrix_to_ipc_stream([], field_name="similarity")

    def _norm(rows: list[list[float]]) -> list[list[float]]:
        out: list[list[float]] = []
        for r in rows:
            nrm = max(math.sqrt(sum(x * x for x in r)), 1e-12)
            out.append([x / nrm for x in r])
        return out

    a_norm = _norm(mat_a)
    b_norm = _norm(mat_b)
    sims = [[sum(x * y for x, y in zip(ra, rb)) for rb in b_norm] for ra in a_norm]
    return matrix_to_ipc_stream(sims, field_name="similarity")


def top_k_cosine_arrow_ipc(
    matrix_ipc: bytes,
    query: Sequence[float],
    k: int = 10,
    exclude_row: int = -1,
) -> list[tuple[int, float]]:
    """Compute top-`k` `(row_index, cosine_similarity)` over an Arrow IPC matrix."""
    if k <= 0 or not matrix_ipc or not query:
        return []
    query_ipc = vector_to_ipc_stream(query, field_name="query")
    lib = _get_rust_lib()
    if lib is not None:
        buf = lib.jcube_ipc_top_k_cosine(
            matrix_ipc,
            len(matrix_ipc),
            query_ipc,
            len(query_ipc),
            int(k),
            int(exclude_row),
        )
        res_bytes = _consume_ipc_buffer(lib, buf)
        reader = pa_ipc.open_stream(pa.py_buffer(res_bytes))
        out: list[tuple[int, float]] = []
        for batch in reader:
            if batch.num_rows == 0:
                continue
            idxs = batch.column(0).to_pylist()
            scores = batch.column(1).to_pylist()
            out.extend((int(i), float(s)) for i, s in zip(idxs, scores))
        return out

    rows = matrix_from_ipc_stream(matrix_ipc)
    q_norm = math.sqrt(sum(float(x) * float(x) for x in query))
    scored: list[tuple[int, float]] = []
    for idx, row in enumerate(rows):
        if idx == exclude_row:
            continue
        if q_norm <= 0.0:
            scored.append((idx, 0.0))
            continue
        dot = sum(a * b for a, b in zip(row, query))
        r_norm = math.sqrt(sum(a * a for a in row))
        denom = max(r_norm * q_norm, 1e-8)
        scored.append((idx, dot / denom))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:k]


def centroid_anomalies_arrow_ipc(
    matrix_ipc: bytes,
    indices: Sequence[int],
    k: int = 20,
) -> list[tuple[int, int, float]]:
    """Compute top-`k` `(pos, row_index, anomaly_score)` over an Arrow IPC matrix."""
    if k <= 0 or not matrix_ipc or not indices:
        return []
    idx_ipc = vector_to_ipc_stream([float(i) for i in indices], field_name="indices")
    lib = _get_rust_lib()
    if lib is not None:
        buf = lib.jcube_ipc_centroid_anomalies(
            matrix_ipc,
            len(matrix_ipc),
            idx_ipc,
            len(idx_ipc),
            int(k),
        )
        res_bytes = _consume_ipc_buffer(lib, buf)
        reader = pa_ipc.open_stream(pa.py_buffer(res_bytes))
        out: list[tuple[int, int, float]] = []
        for batch in reader:
            if batch.num_rows == 0:
                continue
            pos_list = batch.column(0).to_pylist()
            row_list = batch.column(1).to_pylist()
            score_list = batch.column(2).to_pylist()
            out.extend((int(p), int(r), float(s)) for p, r, s in zip(pos_list, row_list, score_list))
        return out

    rows = matrix_from_ipc_stream(matrix_ipc)
    valid = [(pos, int(r)) for pos, r in enumerate(indices) if 0 <= int(r) < len(rows)]
    if not valid:
        return []
    dim = len(rows[valid[0][1]])
    centroid = [0.0] * dim
    for _, r in valid:
        for j, val in enumerate(rows[r]):
            centroid[j] += val
    inv_n = 1.0 / len(valid)
    centroid = [c * inv_n for c in centroid]
    c_norm = max(math.sqrt(sum(c * c for c in centroid)), 1e-8)
    centroid = [c / c_norm for c in centroid]

    scored: list[tuple[int, int, float]] = []
    for pos, r in valid:
        row = rows[r]
        r_norm = max(math.sqrt(sum(x * x for x in row)), 1e-8)
        sim = sum(x * c for x, c in zip(row, centroid)) / r_norm
        scored.append((pos, r, 1.0 - sim))
    scored.sort(key=lambda item: item[2], reverse=True)
    return scored[:k]


# ============================================================================
# Zero-Copy Shared Memory Tensor over Uncompressed Arrow IPC Files (`.arrow`)
# ============================================================================


def _create_or_inspect_ipc_tensor_file(
    path: Path,
    shape: tuple[int, ...],
    dtype: str,
    create: bool,
) -> tuple[int, int, int, int, str]:
    """Return `(data_offset, data_byte_len, elem_size, num_elements, canonical_dtype)`."""
    arrow_dt, _fmt, elem_size, canon_dtype = normalize_dtype(dtype)
    num_elements = math.prod(shape)

    lib = _get_rust_lib()
    if lib is not None:
        path_bytes = str(path).encode("utf-8")
        if create:
            shape_arr = (ctypes.c_size_t * len(shape))(*shape)
            layout = lib.jcube_ipc_create_tensor_file(
                path_bytes,
                shape_arr,
                len(shape),
                canon_dtype.encode("utf-8"),
            )
        else:
            layout = lib.jcube_ipc_inspect_tensor_file(path_bytes)
        if layout.status == 0:
            return (
                int(layout.data_offset),
                int(layout.data_byte_len),
                int(layout.elem_size),
                int(layout.num_elements),
                canon_dtype,
            )

    # Pure PyArrow IPC fallback when Rust library is disabled/unavailable
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            b"jcube.shape": ",".join(str(d) for d in shape).encode("utf-8"),
            b"jcube.dtype": canon_dtype.encode("utf-8"),
            b"jcube.format": b"arrow-ipc-shared-tensor-v1",
        }
        schema = pa.schema([pa.field("values", arrow_dt, nullable=False)], metadata=meta)
        zero_buf = pa.allocate_buffer(num_elements * elem_size, resizable=False)
        ctypes.memset(zero_buf.address, 0, num_elements * elem_size)
        arr = pa.Array.from_buffers(arrow_dt, num_elements, [None, zero_buf])
        batch = pa.RecordBatch.from_arrays([arr], schema=schema)
        with pa.OSFile(str(path), "wb") as sink:
            with pa_ipc.new_file(sink, schema) as writer:
                writer.write_batch(batch)

    if not path.is_file():
        raise FileNotFoundError(f"Shared Arrow IPC file not found: {path}")

    with pa.memory_map(str(path), "r") as mmap_source:
        full_buf = mmap_source.read_buffer()
        reader = pa_ipc.open_file(full_buf)
        batch = reader.get_batch(0)
        col = batch.column(0)
        val_buf = col.buffers()[1]
        if val_buf is None:
            raise RuntimeError(f"Arrow IPC file {path} has no values buffer")
        data_offset = int(val_buf.address - full_buf.address)
        data_byte_len = int(len(col) * elem_size)
        return data_offset, data_byte_len, elem_size, int(len(col)), canon_dtype


class ArrowIpcTensorView:
    """Zero-copy N-dimensional mutable view over an uncompressed Arrow IPC buffer.

    Works without requiring `numpy` while remaining 100% compatible with indexing,
    slice assignment (`view[:] = ...`, `view[i] = ...`, `view[i, j] = ...`),
    iteration, `.flush()`, and `.tolist()`.
    """

    def __init__(
        self,
        flat_view: Any,
        shape: tuple[int, ...],
        dtype: str,
        row_offset: int = 0,
        flush_fn: Any = None,
    ) -> None:
        self._flat: Any = flat_view
        self.shape = shape
        self.dtype = dtype
        self._row_offset = row_offset
        self._is_int = dtype in {"int64", "int32", "uint8"}
        self._flush_fn = flush_fn

    def flush(self) -> None:
        """Flush the underlying memory-mapped Arrow IPC file."""
        if self._flush_fn is not None:
            self._flush_fn()

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def _coerce_scalar(self, val: Any) -> int | float:
        return int(val) if self._is_int else float(val)

    def _stride0(self) -> int:
        return math.prod(self.shape[1:]) if len(self.shape) > 1 else 1

    def __len__(self) -> int:
        return self.shape[0] if self.shape else 0

    def __iter__(self) -> Iterator[Any]:
        for i in range(len(self)):
            yield self[i]

    def tolist(self) -> list[Any]:
        if len(self.shape) == 1:
            return [self._flat[self._row_offset + i] for i in range(self.shape[0])]
        stride = self._stride0()
        out: list[Any] = []
        for i in range(self.shape[0]):
            sub = ArrowIpcTensorView(
                self._flat,
                self.shape[1:],
                self.dtype,
                row_offset=self._row_offset + i * stride,
            )
            out.append(sub.tolist())
        return out

    def __getitem__(self, idx: Any) -> Any:
        if isinstance(idx, tuple):
            if len(idx) == 0:
                return self
            if len(idx) == 1:
                return self[idx[0]]
            first, rest = idx[0], idx[1:]
            sub = self[first]
            if isinstance(sub, ArrowIpcTensorView):
                return sub[rest]
            return sub

        if isinstance(idx, slice):
            start, stop, step = idx.indices(self.shape[0])
            if step == 1 and len(self.shape) >= 1:
                stride = self._stride0()
                count = max(0, stop - start)
                return ArrowIpcTensorView(
                    self._flat,
                    (count, *self.shape[1:]),
                    self.dtype,
                    row_offset=self._row_offset + start * stride,
                    flush_fn=self._flush_fn,
                )
            return [self[i] for i in range(start, stop, step)]

        i = int(idx)
        if i < 0:
            i += self.shape[0]
        if i < 0 or i >= self.shape[0]:
            raise IndexError(f"Index {idx} out of bounds for axis 0 with size {self.shape[0]}")

        if len(self.shape) == 1:
            return self._flat[self._row_offset + i]

        stride = self._stride0()
        return ArrowIpcTensorView(
            self._flat,
            self.shape[1:],
            self.dtype,
            row_offset=self._row_offset + i * stride,
            flush_fn=self._flush_fn,
        )

    def __setitem__(self, idx: Any, value: Any) -> None:
        if isinstance(idx, tuple):
            if len(idx) == 1:
                self[idx[0]] = value
                return
            first, rest = idx[0], idx[1:]
            sub = self[first]
            if not isinstance(sub, ArrowIpcTensorView):
                raise TypeError("Cannot tuple-index into scalar element")
            sub[rest] = value
            return

        if isinstance(idx, slice):
            start, stop, step = idx.indices(self.shape[0])
            indices = range(start, stop, step)
            if isinstance(value, (int, float)):
                scalar = self._coerce_scalar(value)
                stride = self._stride0()
                for i in indices:
                    base = self._row_offset + i * stride
                    for k in range(stride):
                        self._flat[base + k] = scalar
                return
            val_list = list(value)
            if len(val_list) != len(indices):
                raise ValueError(f"Slice assignment length mismatch: expected {len(indices)}, got {len(val_list)}")
            for i, v in zip(indices, val_list):
                self[i] = v
            return

        i = int(idx)
        if i < 0:
            i += self.shape[0]
        if i < 0 or i >= self.shape[0]:
            raise IndexError(f"Index {idx} out of bounds for axis 0 with size {self.shape[0]}")

        if len(self.shape) == 1:
            self._flat[self._row_offset + i] = self._coerce_scalar(value)
            return

        sub = self[i]
        sub[:] = value

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ArrowIpcTensorView):
            return self.tolist() == other.tolist()
        if hasattr(other, "tolist"):
            return bool(self.tolist() == other.tolist())
        if isinstance(other, Iterable):
            return self.tolist() == list(other)
        return False

    def __repr__(self) -> str:
        return f"ArrowIpcTensorView(shape={self.shape}, dtype={self.dtype!r}, data={self.tolist()})"


class ArrowIpcSharedTensor:
    """Zero-copy cross-process tensor backed by an uncompressed Apache Arrow IPC `.arrow` file."""

    def __init__(
        self,
        path: Path,
        shape: tuple[int, ...],
        dtype: str = "int64",
        create: bool = True,
    ) -> None:
        self.path = Path(path)
        self.shape = tuple(int(d) for d in shape)
        arrow_dt, fmt_char, elem_size, canon_dtype = normalize_dtype(dtype)
        self.dtype = canon_dtype
        self._arrow_type = arrow_dt
        self._fmt_char = fmt_char
        self._elem_size = elem_size

        data_offset, data_byte_len, _, num_elements, _ = _create_or_inspect_ipc_tensor_file(
            self.path,
            self.shape,
            canon_dtype,
            create=create,
        )
        self._data_offset = data_offset
        self._data_byte_len = data_byte_len
        self._num_elements = num_elements

        self._file = open(self.path, "r+b")  # noqa: SIM115
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_WRITE)
        raw_slice = memoryview(self._mmap)[self._data_offset : self._data_offset + self._data_byte_len]
        self._typed_view: Any = cast(Any, raw_slice).cast(self._fmt_char)
        self.view = ArrowIpcTensorView(
            self._typed_view,
            self.shape,
            self.dtype,
            flush_fn=self.flush,
        )

    def flush(self) -> None:
        """Flush dirty memory-mapped pages to the underlying `.arrow` IPC file."""
        if hasattr(self, "_mmap") and not self._mmap.closed:
            self._mmap.flush()

    def read_arrow(self) -> pa.Table:
        """Read the `.arrow` IPC file zero-copy as a `pyarrow.Table`."""
        self.flush()
        with pa.memory_map(str(self.path), "r") as source:
            return cast(pa.Table, pa_ipc.open_file(source).read_all())

    def to_ipc_bytes(self) -> bytes:
        """Return the raw `.arrow` IPC file bytes."""
        self.flush()
        return self.path.read_bytes()

    def close(self) -> None:
        """Release memoryviews, close the `mmap`, and close the file descriptor."""
        if hasattr(self, "view"):
            del self.view
        if hasattr(self, "_typed_view"):
            self._typed_view.release()
            del self._typed_view
        if hasattr(self, "_mmap") and not self._mmap.closed:
            self._mmap.close()
        if hasattr(self, "_file") and not self._file.closed:
            self._file.close()

    def unlink(self) -> None:
        """Close handles and delete the backing `.arrow` file from disk."""
        self.close()
        if self.path.exists():
            self.path.unlink()
