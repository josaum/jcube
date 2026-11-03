//! Native Arrow IPC low-level compute and zero-copy shared memory engine for Event-JEPA-Cube (`jcube`).
//!
//! Provides:
//! 1. Uncompressed Arrow IPC File (`.arrow`) zero-copy tensor creation and physical buffer offset
//!    inspection for cross-process memory-mapped mutation (`SharedGraphStore`, `MmapEmbeddingStore`).
//! 2. Arrow IPC Stream/File sequence kernels (`weighted_aggregate_ipc`, `compute_trend_ipc`,
//!    `streaming_ema_batch_ipc`).
//! 3. Arrow IPC Stream/File matrix & similarity kernels (`batch_cosine_similarity_ipc`,
//!    `top_k_cosine_ipc`, `centroid_anomalies_ipc`) accelerated with `rayon`.
//! 4. C-ABI exports (`jcube_ipc_*`) for zero-dependency `ctypes` binding from Python.

use std::collections::HashMap;
use std::ffi::{c_char, CStr};
use std::fs::OpenOptions;
use std::io::Cursor;
use std::path::Path;
use std::slice;
use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, Float64Array, Int32Array, Int64Array,
    RecordBatch, UInt8Array,
};
use arrow_ipc::reader::{FileReader, StreamReader};
use arrow_ipc::writer::{FileWriter, IpcWriteOptions, StreamWriter};
use arrow_ipc::{root_as_footer, root_as_message};
use arrow_schema::{DataType, Field, Schema};
use memmap2::Mmap;
use rayon::prelude::*;

const ARROW_MAGIC: &[u8; 6] = b"ARROW1";
const VERSION_CSTR: &[u8] = b"0.3.0\0";

/// Buffer returned across the C-ABI boundary.
///
/// When `status == 0`, `ptr..ptr+len` holds valid Arrow IPC stream bytes.
/// When `status != 0`, `ptr..ptr+len` holds a UTF-8 error message.
/// Must be freed via [`jcube_ipc_free_buffer`].
#[repr(C)]
#[derive(Debug)]
pub struct IpcBuffer {
    pub ptr: *mut u8,
    pub len: usize,
    pub cap: usize,
    pub status: i32,
}

impl IpcBuffer {
    fn from_vec(mut bytes: Vec<u8>, status: i32) -> Self {
        let ptr = bytes.as_mut_ptr();
        let len = bytes.len();
        let cap = bytes.capacity();
        std::mem::forget(bytes);
        Self {
            ptr,
            len,
            cap,
            status,
        }
    }

    fn ok(bytes: Vec<u8>) -> Self {
        Self::from_vec(bytes, 0)
    }

    fn err(msg: impl Into<String>) -> Self {
        Self::from_vec(msg.into().into_bytes(), 1)
    }
}

/// Physical layout of the primary value buffer inside an uncompressed Arrow IPC `.arrow` file.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorFileLayout {
    pub data_offset: usize,
    pub data_byte_len: usize,
    pub file_byte_len: usize,
    pub elem_size: usize,
    pub num_elements: usize,
    pub status: i32,
}

impl TensorFileLayout {
    fn err() -> Self {
        Self {
            data_offset: 0,
            data_byte_len: 0,
            file_byte_len: 0,
            elem_size: 0,
            num_elements: 0,
            status: 1,
        }
    }
}

/// Return packet for streaming EMA updates over Arrow IPC.
#[repr(C)]
#[derive(Debug)]
pub struct StreamingEmaResult {
    pub state_ipc: IpcBuffer,
    pub last_timestamp: f64,
    pub count: u64,
}

fn dtype_info(dtype: &str) -> Result<(DataType, &'static str, usize), String> {
    match dtype.trim().to_ascii_lowercase().as_str() {
        "int64" | "i64" | "<i8" => Ok((DataType::Int64, "int64", 8)),
        "int32" | "i32" | "<i4" => Ok((DataType::Int32, "int32", 4)),
        "float64" | "f64" | "<f8" => Ok((DataType::Float64, "float64", 8)),
        "float32" | "f32" | "<f4" => Ok((DataType::Float32, "float32", 4)),
        "uint8" | "u8" | "|u1" => Ok((DataType::UInt8, "uint8", 1)),
        other => Err(format!("Unsupported Arrow IPC tensor dtype: {other}")),
    }
}

/// Creates an uncompressed Arrow IPC `.arrow` file containing a single record batch with a
/// contiguous, 64-byte-aligned primitive buffer of length `shape.iter().product()`, and returns
/// its exact physical byte offset for zero-copy `mmap` sharing across processes.
pub fn create_ipc_tensor_file(
    path: &Path,
    shape: &[usize],
    dtype: &str,
) -> Result<TensorFileLayout, String> {
    if shape.is_empty() {
        return Err("Tensor shape must have at least 1 dimension".to_string());
    }
    let (arrow_dt, norm_dtype, elem_size) = dtype_info(dtype)?;
    let num_elements = shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| "Tensor shape product overflowed usize".to_string())?;

    let shape_str = shape
        .iter()
        .map( usize::to_string)
        .collect::<Vec<_>>()
        .join(",");

    let mut metadata = HashMap::new();
    metadata.insert("jcube.shape".to_string(), shape_str);
    metadata.insert("jcube.dtype".to_string(), norm_dtype.to_string());
    metadata.insert(
        "jcube.format".to_string(),
        "arrow-ipc-shared-tensor-v1".to_string(),
    );

    let schema = Arc::new(Schema::new_with_metadata(
        vec![Field::new("values", arrow_dt.clone(), false)],
        metadata,
    ));

    let array: ArrayRef = match arrow_dt {
        DataType::Int64 => Arc::new(Int64Array::from(vec![0i64; num_elements])),
        DataType::Int32 => Arc::new(Int32Array::from(vec![0i32; num_elements])),
        DataType::Float64 => Arc::new(Float64Array::from(vec![0.0f64; num_elements])),
        DataType::Float32 => Arc::new(Float32Array::from(vec![0.0f32; num_elements])),
        DataType::UInt8 => Arc::new(UInt8Array::from(vec![0u8; num_elements])),
        _ => unreachable!(),
    };

    let batch = RecordBatch::try_new(schema.clone(), vec![array])
        .map_err(|e| format!("Failed to build Arrow RecordBatch: {e}"))?;

    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create parent directory: {e}"))?;
        }
    }

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)
        .map_err(|e| format!("Failed to create Arrow IPC file {}: {e}", path.display()))?;

    let options = IpcWriteOptions::default();
    let mut writer = FileWriter::try_new_with_options(&file, &schema, options)
        .map_err(|e| format!("Failed to initialize Arrow IPC FileWriter: {e}"))?;
    writer
        .write(&batch)
        .map_err(|e| format!("Failed to write Arrow IPC RecordBatch: {e}"))?;
    writer
        .finish()
        .map_err(|e| format!("Failed to finish Arrow IPC FileWriter: {e}"))?;
    file.sync_all()
        .map_err(|e| format!("Failed to sync Arrow IPC file: {e}"))?;

    let layout = inspect_ipc_tensor_file(path)?;
    if layout.elem_size != elem_size || layout.num_elements != num_elements {
        return Err(format!(
            "Arrow IPC tensor layout mismatch: expected {}x{}B, got {}x{}B",
            num_elements, elem_size, layout.num_elements, layout.elem_size
        ));
    }
    Ok(layout)
}

/// Inspects an existing uncompressed Arrow IPC `.arrow` tensor file and returns the exact physical
/// byte offset and byte length of the primary `"values"` data buffer inside the file.
pub fn inspect_ipc_tensor_file(path: &Path) -> Result<TensorFileLayout, String> {
    let file = OpenOptions::new()
        .read(true)
        .open(path)
        .map_err(|e| format!("Failed to open Arrow IPC file {}: {e}", path.display()))?;
    let mmap = unsafe { Mmap::map(&file) }
        .map_err(|e| format!("Failed to mmap Arrow IPC file {}: {e}", path.display()))?;

    let file_len = mmap.len();
    if file_len < 18 {
        return Err("File is too small to be a valid Arrow IPC file".to_string());
    }
    if &mmap[..6] != ARROW_MAGIC || &mmap[file_len - 6..] != ARROW_MAGIC {
        return Err("Invalid Arrow IPC magic header/trailer".to_string());
    }

    let footer_len_bytes: [u8; 4] = mmap[file_len - 10..file_len - 6]
        .try_into()
        .map_err(|_| "Failed to read footer length".to_string())?;
    let footer_len = i32::from_le_bytes(footer_len_bytes) as usize;
    if footer_len + 10 > file_len {
        return Err("Corrupt Arrow IPC footer length".to_string());
    }
    let footer_offset = file_len - 10 - footer_len;
    let footer = root_as_footer(&mmap[footer_offset..file_len - 10])
        .map_err(|e| format!("Failed to parse Arrow IPC FlatBuffer footer: {e}"))?;

    let batches = footer
        .recordBatches()
        .ok_or_else(|| "Arrow IPC file has no recordBatches in footer".to_string())?;
    if batches.is_empty() {
        return Err("Arrow IPC file contains zero record batches".to_string());
    }
    let block = batches.get(0);
    let block_offset = block.offset() as usize;
    let meta_len = block.metaDataLength() as usize;
    if block_offset + meta_len > footer_offset {
        return Err("Arrow IPC RecordBatch block exceeds file bounds".to_string());
    }

    // Parse message prefix (continuation marker 0xFFFFFFFF + 4-byte metadata size)
    let msg_slice = &mmap[block_offset..block_offset + meta_len];
    let msg_fb_slice = if msg_slice.len() >= 8 && msg_slice[..4] == [0xFF, 0xFF, 0xFF, 0xFF] {
        let size = i32::from_le_bytes(msg_slice[4..8].try_into().unwrap()) as usize;
        &msg_slice[8..8 + size]
    } else if msg_slice.len() >= 4 {
        let size = i32::from_le_bytes(msg_slice[..4].try_into().unwrap()) as usize;
        &msg_slice[4..4 + size]
    } else {
        return Err("Arrow IPC RecordBatch metadata header is truncated".to_string());
    };

    let message = root_as_message(msg_fb_slice)
        .map_err(|e| format!("Failed to parse Arrow IPC RecordBatch Message: {e}"))?;
    let rb = message
        .header_as_record_batch()
        .ok_or_else(|| "First block in Arrow IPC file is not a RecordBatch".to_string())?;
    if rb.compression().is_some() {
        return Err(
            "Compressed Arrow IPC files cannot be directly mutated via zero-copy mmap".to_string(),
        );
    }

    let buffers = rb
        .buffers()
        .ok_or_else(|| "RecordBatch has no buffer descriptors".to_string())?;
    if buffers.len() < 2 {
        return Err("RecordBatch has fewer than 2 buffers".to_string());
    }
    // Buffer 0 is the validity bitmap (empty for non-nullable primitives); Buffer 1 is the values buffer.
    let val_buf = buffers.get(1);
    let body_start = block_offset + meta_len;
    let data_offset = body_start + (val_buf.offset() as usize);
    let data_byte_len = val_buf.length() as usize;

    // Also inspect schema to determine element size and row count.
    let reader = FileReader::try_new(Cursor::new(&mmap[..]), None)
        .map_err(|e| format!("Failed to open Arrow IPC FileReader: {e}"))?;
    let schema = reader.schema();
    if schema.fields().is_empty() {
        return Err("Arrow IPC schema has no fields".to_string());
    }
    let elem_size = match schema.field(0).data_type() {
        DataType::Int64 | DataType::Float64 => 8,
        DataType::Int32 | DataType::Float32 => 4,
        DataType::UInt8 => 1,
        other => {
            return Err(format!(
                "Unsupported Arrow IPC tensor field DataType: {other:?}"
            ))
        }
    };
    let num_elements = rb.length() as usize;
    let expected_bytes = num_elements * elem_size;
    if data_byte_len < expected_bytes || data_offset + expected_bytes > file_len {
        return Err(format!(
            "Invalid Arrow IPC buffer range: offset={data_offset}, len={data_byte_len}, expected={expected_bytes}"
        ));
    }

    Ok(TensorFileLayout {
        data_offset,
        data_byte_len: expected_bytes,
        file_byte_len: file_len,
        elem_size,
        num_elements,
        status: 0,
    })
}

fn read_ipc_batches(ipc_bytes: &[u8]) -> Result<(Arc<Schema>, Vec<RecordBatch>), String> {
    if ipc_bytes.len() >= 8 && &ipc_bytes[..6] == ARROW_MAGIC {
        let reader = FileReader::try_new(Cursor::new(ipc_bytes), None)
            .map_err(|e| format!("Failed to open Arrow IPC File: {e}"))?;
        let schema = reader.schema();
        let mut batches = Vec::new();
        for batch_res in reader {
            batches.push(batch_res.map_err(|e| format!("Failed reading Arrow IPC File batch: {e}"))?);
        }
        Ok((schema, batches))
    } else {
        let reader = StreamReader::try_new(Cursor::new(ipc_bytes), None)
            .map_err(|e| format!("Failed to open Arrow IPC Stream: {e}"))?;
        let schema = reader.schema();
        let mut batches = Vec::new();
        for batch_res in reader {
            batches.push(
                batch_res.map_err(|e| format!("Failed reading Arrow IPC Stream batch: {e}"))?,
            );
        }
        Ok((schema, batches))
    }
}

fn write_f64_vector_ipc_stream(field_name: &str, values: Vec<f64>) -> Result<Vec<u8>, String> {
    let schema = Arc::new(Schema::new(vec![Field::new(
        field_name,
        DataType::Float64,
        false,
    )]));
    let array: ArrayRef = Arc::new(Float64Array::from(values));
    let batch = RecordBatch::try_new(schema.clone(), vec![array])
        .map_err(|e| format!("Failed to create vector RecordBatch: {e}"))?;
    let mut out = Vec::with_capacity(512);
    {
        let mut writer = StreamWriter::try_new(&mut out, &schema)
            .map_err(|e| format!("Failed to create Arrow IPC StreamWriter: {e}"))?;
        writer
            .write(&batch)
            .map_err(|e| format!("Failed to write Arrow IPC stream batch: {e}"))?;
        writer
            .finish()
            .map_err(|e| format!("Failed to finish Arrow IPC stream: {e}"))?;
    }
    Ok(out)
}

fn extract_f64_column(col: &dyn Array) -> Result<Vec<f64>, String> {
    if let Some(f64_arr) = col.as_any().downcast_ref::<Float64Array>() {
        Ok(f64_arr.values().to_vec())
    } else if let Some(f32_arr) = col.as_any().downcast_ref::<Float32Array>() {
        Ok(f32_arr.values().iter().map(|&v| v as f64).collect())
    } else if let Some(i64_arr) = col.as_any().downcast_ref::<Int64Array>() {
        Ok(i64_arr.values().iter().map(|&v| v as f64).collect())
    } else {
        Err(format!(
            "Expected numeric array (Float64/Float32/Int64), got {:?}",
            col.data_type()
        ))
    }
}

/// Extracts `(timestamps, flat_embeddings, n_rows, dim)` from an Arrow IPC sequence stream/file.
///
/// Accepts either:
/// - 2 columns: `(timestamp, embedding: FixedSizeList<Float64|Float32>(D))`
/// - `1 + D` columns: `(timestamp, d_0, d_1, ..., d_{D-1})`
fn decode_sequence_ipc(ipc_bytes: &[u8]) -> Result<(Vec<f64>, Vec<f64>, usize, usize), String> {
    let (_schema, batches) = read_ipc_batches(ipc_bytes)?;
    let mut timestamps = Vec::new();
    let mut flat_embeddings = Vec::new();
    let mut dim: Option<usize> = None;
    let mut total_rows = 0usize;

    for batch in &batches {
        let n = batch.num_rows();
        if n == 0 {
            continue;
        }
        if batch.num_columns() < 2 {
            return Err("Sequence Arrow IPC batch requires at least 2 columns".to_string());
        }
        let ts_col = extract_f64_column(batch.column(0).as_ref())?;
        timestamps.extend_from_slice(&ts_col);

        if batch.num_columns() == 2 {
            if let Some(fsl) = batch
                .column(1)
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
            {
                let d = fsl.value_length() as usize;
                if let Some(prev_d) = dim {
                    if prev_d != d {
                        return Err(format!("Inconsistent embedding dimension: {prev_d} vs {d}"));
                    }
                } else {
                    dim = Some(d);
                }
                let values = extract_f64_column(fsl.values().as_ref())?;
                if values.len() != n * d {
                    return Err("FixedSizeList values length mismatch".to_string());
                }
                flat_embeddings.extend_from_slice(&values);
                total_rows += n;
                continue;
            }
        }

        // Multi-column scalar layout: column 0 is timestamp, columns 1..=D are dimensions
        let d = batch.num_columns() - 1;
        if let Some(prev_d) = dim {
            if prev_d != d {
                return Err(format!("Inconsistent embedding dimension: {prev_d} vs {d}"));
            }
        } else {
            dim = Some(d);
        }
        let cols: Vec<Vec<f64>> = (1..=d)
            .map(|c| extract_f64_column(batch.column(c).as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        flat_embeddings.reserve(n * d);
        for r in 0..n {
            for col in &cols {
                flat_embeddings.push(col[r]);
            }
        }
        total_rows += n;
    }

    Ok((timestamps, flat_embeddings, total_rows, dim.unwrap_or(0)))
}

/// Extracts a 2D matrix `(flat_values, n_rows, dim)` from an Arrow IPC stream/file.
fn decode_matrix_ipc(ipc_bytes: &[u8]) -> Result<(Vec<f64>, usize, usize), String> {
    let (schema, batches) = read_ipc_batches(ipc_bytes)?;
    let mut flat = Vec::new();
    let mut dim: Option<usize> = None;
    let mut total_rows = 0usize;

    // Check if this is a flattened tensor file with `jcube.shape` metadata
    let meta_shape: Option<Vec<usize>> = schema.metadata().get("jcube.shape").and_then(|s| {
        s.split(',')
            .map(|p| p.trim().parse::<usize>().ok())
            .collect()
    });

    for batch in &batches {
        let n = batch.num_rows();
        if n == 0 {
            continue;
        }
        if batch.num_columns() == 1 {
            if let Some(fsl) = batch
                .column(0)
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
            {
                let d = fsl.value_length() as usize;
                dim = Some(d);
                let values = extract_f64_column(fsl.values().as_ref())?;
                flat.extend_from_slice(&values);
                total_rows += n;
                continue;
            }
            if let Some(ref shape) = meta_shape {
                if shape.len() == 2 && shape[0] * shape[1] == n {
                    let values = extract_f64_column(batch.column(0).as_ref())?;
                    return Ok((values, shape[0], shape[1]));
                }
            }
        }
        let d = batch.num_columns();
        dim = Some(d);
        let cols: Vec<Vec<f64>> = (0..d)
            .map(|c| extract_f64_column(batch.column(c).as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        flat.reserve(n * d);
        for r in 0..n {
            for col in &cols {
                flat.push(col[r]);
            }
        }
        total_rows += n;
    }

    Ok((flat, total_rows, dim.unwrap_or(0)))
}

fn decode_vector_ipc(ipc_bytes: &[u8]) -> Result<Vec<f64>, String> {
    if ipc_bytes.is_empty() {
        return Ok(Vec::new());
    }
    let (_schema, batches) = read_ipc_batches(ipc_bytes)?;
    let mut out = Vec::new();
    for batch in &batches {
        if batch.num_rows() == 0 || batch.num_columns() == 0 {
            continue;
        }
        let vals = extract_f64_column(batch.column(0).as_ref())?;
        out.extend_from_slice(&vals);
    }
    Ok(out)
}

/// Computes an exponentially time-decayed weighted average of embeddings over an Arrow IPC stream.
pub fn weighted_aggregate_ipc(ipc_bytes: &[u8], alpha: f64) -> Result<Vec<u8>, String> {
    let (timestamps, flat_emb, n, dim) = decode_sequence_ipc(ipc_bytes)?;
    if n == 0 || dim == 0 {
        return write_f64_vector_ipc_stream("representation", Vec::new());
    }
    let max_t = timestamps.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut weights = Vec::with_capacity(n);
    let mut total_w = 0.0f64;
    for &t in &timestamps {
        let w = (-alpha * (max_t - t)).exp();
        weights.push(w);
        total_w += w;
    }
    if total_w == 0.0 {
        total_w = 1.0;
    }

    let mut result = vec![0.0f64; dim];
    for (row_idx, &w) in weights.iter().enumerate() {
        let row = &flat_emb[row_idx * dim..(row_idx + 1) * dim];
        for (j, &val) in row.iter().enumerate() {
            result[j] += val * w;
        }
    }
    for val in &mut result {
        *val /= total_w;
    }

    write_f64_vector_ipc_stream("representation", result)
}

/// Computes exponentially-weighted trend from consecutive deltas in an Arrow IPC sequence stream.
pub fn compute_trend_ipc(ipc_bytes: &[u8], alpha: f64) -> Result<Vec<u8>, String> {
    let (timestamps, flat_emb, n, dim) = decode_sequence_ipc(ipc_bytes)?;
    if n == 0 || dim == 0 {
        return write_f64_vector_ipc_stream("trend", Vec::new());
    }
    if n < 2 {
        return write_f64_vector_ipc_stream("trend", vec![0.0f64; dim]);
    }

    let delta_times = &timestamps[1..];
    let t_max = delta_times
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let mut weights = Vec::with_capacity(n - 1);
    let mut total_w = 0.0f64;
    for &t in delta_times {
        let w = (-alpha * (t_max - t)).exp();
        weights.push(w);
        total_w += w;
    }
    if total_w == 0.0 {
        total_w = 1.0;
    }

    let mut trend = vec![0.0f64; dim];
    for (step_idx, &w) in weights.iter().enumerate() {
        let prev_row = &flat_emb[step_idx * dim..(step_idx + 1) * dim];
        let curr_row = &flat_emb[(step_idx + 1) * dim..(step_idx + 2) * dim];
        for j in 0..dim {
            trend[j] += (curr_row[j] - prev_row[j]) * w;
        }
    }
    for val in &mut trend {
        *val /= total_w;
    }

    write_f64_vector_ipc_stream("trend", trend)
}

type EmaStateVectors = (Vec<f64>, Vec<f64>, Vec<f64>);

fn decode_ema_state_ipc(ipc_bytes: &[u8]) -> Result<EmaStateVectors, String> {
    if ipc_bytes.is_empty() {
        return Ok((Vec::new(), Vec::new(), Vec::new()));
    }
    let (_schema, batches) = read_ipc_batches(ipc_bytes)?;
    let mut repr = Vec::new();
    let mut mean = Vec::new();
    let mut m2 = Vec::new();
    for batch in &batches {
        if batch.num_rows() == 0 || batch.num_columns() == 0 {
            continue;
        }
        repr.extend_from_slice(&extract_f64_column(batch.column(0).as_ref())?);
        if batch.num_columns() >= 3 {
            mean.extend_from_slice(&extract_f64_column(batch.column(1).as_ref())?);
            m2.extend_from_slice(&extract_f64_column(batch.column(2).as_ref())?);
        }
    }
    if mean.is_empty() && !repr.is_empty() {
        mean = vec![0.0f64; repr.len()];
    }
    if m2.is_empty() && !repr.is_empty() {
        m2 = vec![0.0f64; repr.len()];
    }
    Ok((repr, mean, m2))
}

fn write_ema_state_ipc(repr: Vec<f64>, mean: Vec<f64>, m2: Vec<f64>) -> Result<Vec<u8>, String> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("representation", DataType::Float64, false),
        Field::new("running_mean", DataType::Float64, false),
        Field::new("running_m2", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Float64Array::from(repr)),
            Arc::new(Float64Array::from(mean)),
            Arc::new(Float64Array::from(m2)),
        ],
    )
    .map_err(|e| format!("Failed to create EMA state RecordBatch: {e}"))?;
    let mut out = Vec::with_capacity(512);
    {
        let mut writer = StreamWriter::try_new(&mut out, &schema)
            .map_err(|e| format!("Failed to create EMA StreamWriter: {e}"))?;
        writer.write(&batch).map_err(|e| e.to_string())?;
        writer.finish().map_err(|e| e.to_string())?;
    }
    Ok(out)
}

/// Processes a batch of timestamped embeddings through streaming EMA + Welford statistics over Arrow IPC.
pub fn streaming_ema_batch_ipc(
    ipc_bytes: &[u8],
    state_ipc_bytes: &[u8],
    alpha: f64,
    last_ts: f64,
    has_last_ts: bool,
    count: u64,
) -> Result<(Vec<u8>, f64, u64), String> {
    let (timestamps, flat_emb, n, dim) = decode_sequence_ipc(ipc_bytes)?;
    let (mut repr, mut mean, mut m2) = decode_ema_state_ipc(state_ipc_bytes)?;
    if n == 0 || dim == 0 {
        let out_ipc = write_ema_state_ipc(repr, mean, m2)?;
        return Ok((out_ipc, last_ts, count));
    }
    if repr.len() != dim {
        repr = vec![0.0f64; dim];
    }
    if mean.len() != dim {
        mean = vec![0.0f64; dim];
    }
    if m2.len() != dim {
        m2 = vec![0.0f64; dim];
    }

    let mut cur_count = count;
    let mut prev_ts = last_ts;
    let mut has_prev = has_last_ts;

    for (i, &t_val) in timestamps.iter().enumerate() {
        let e_row = &flat_emb[i * dim..(i + 1) * dim];
        if cur_count == 0 {
            repr.copy_from_slice(e_row);
        } else {
            let dt = if has_prev { t_val - prev_ts } else { 0.0 };
            let decay = (-alpha * dt).exp();
            let one_minus = 1.0 - decay;
            for (r, &e) in repr.iter_mut().zip(e_row.iter()) {
                *r = decay * (*r) + one_minus * e;
            }
        }
        cur_count += 1;
        prev_ts = t_val;
        has_prev = true;

        let inv_c = 1.0 / (cur_count as f64);
        for d in 0..dim {
            let delta = e_row[d] - mean[d];
            mean[d] += delta * inv_c;
            let delta2 = e_row[d] - mean[d];
            m2[d] += delta * delta2;
        }
    }

    let out_ipc = write_ema_state_ipc(repr, mean, m2)?;
    Ok((out_ipc, prev_ts, cur_count))
}

/// Computes pairwise cosine similarity matrix (`N_a x N_b`) between two Arrow IPC matrix streams.
pub fn batch_cosine_similarity_ipc(ipc_a: &[u8], ipc_b: &[u8]) -> Result<Vec<u8>, String> {
    let (flat_a, n_a, dim_a) = decode_matrix_ipc(ipc_a)?;
    let (flat_b, n_b, dim_b) = decode_matrix_ipc(ipc_b)?;

    if n_a == 0 || n_b == 0 || dim_a == 0 || dim_b == 0 {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "similarity",
            DataType::Float64,
            false,
        )]));
        let batch = RecordBatch::new_empty(schema.clone());
        let mut out = Vec::new();
        let mut writer = StreamWriter::try_new(&mut out, &schema)
            .map_err(|e| format!("Failed to create empty StreamWriter: {e}"))?;
        writer
            .write(&batch)
            .map_err(|e| format!("Failed to write empty batch: {e}"))?;
        writer
            .finish()
            .map_err(|e| format!("Failed to finish empty stream: {e}"))?;
        return Ok(out);
    }
    if dim_a != dim_b {
        return Err(format!(
            "Dimension mismatch in batch_cosine_similarity_ipc: {dim_a} vs {dim_b}"
        ));
    }
    let dim = dim_a;

    let norm_rows = |flat: &[f64], rows: usize| -> Vec<f64> {
        let mut normed = vec![0.0f64; rows * dim];
        for r in 0..rows {
            let src = &flat[r * dim..(r + 1) * dim];
            let dst = &mut normed[r * dim..(r + 1) * dim];
            let norm = src.iter().map(|&v| v * v).sum::<f64>().sqrt().max(1e-12);
            for (d, &s) in dst.iter_mut().zip(src.iter()) {
                *d = s / norm;
            }
        }
        normed
    };

    let a_norm = norm_rows(&flat_a, n_a);
    let b_norm = norm_rows(&flat_b, n_b);

    let mut sims = vec![0.0f64; n_a * n_b];
    if n_a * n_b >= 4096 {
        sims.par_chunks_mut(n_b)
            .enumerate()
            .for_each(|(i, row_out)| {
                let a_row = &a_norm[i * dim..(i + 1) * dim];
                for (j, out_val) in row_out.iter_mut().enumerate() {
                    let b_row = &b_norm[j * dim..(j + 1) * dim];
                    *out_val = a_row.iter().zip(b_row.iter()).map(|(&x, &y)| x * y).sum();
                }
            });
    } else {
        for i in 0..n_a {
            let a_row = &a_norm[i * dim..(i + 1) * dim];
            let row_out = &mut sims[i * n_b..(i + 1) * n_b];
            for (j, out_val) in row_out.iter_mut().enumerate() {
                let b_row = &b_norm[j * dim..(j + 1) * dim];
                *out_val = a_row.iter().zip(b_row.iter()).map(|(&x, &y)| x * y).sum();
            }
        }
    }

    let item_field = Arc::new(Field::new("item", DataType::Float64, false));
    let fsl_type = DataType::FixedSizeList(item_field.clone(), n_b as i32);
    let schema = Arc::new(Schema::new(vec![Field::new("similarity", fsl_type, false)]));
    let values_arr: ArrayRef = Arc::new(Float64Array::from(sims));
    let fsl_arr: ArrayRef = Arc::new(
        FixedSizeListArray::try_new(item_field, n_b as i32, values_arr, None)
            .map_err(|e| format!("Failed to construct FixedSizeListArray: {e}"))?,
    );
    let batch = RecordBatch::try_new(schema.clone(), vec![fsl_arr])
        .map_err(|e| format!("Failed to create similarity RecordBatch: {e}"))?;

    let mut out = Vec::with_capacity(n_a * n_b * 8 + 512);
    {
        let mut writer = StreamWriter::try_new(&mut out, &schema)
            .map_err(|e| format!("Failed to create Arrow IPC StreamWriter: {e}"))?;
        writer
            .write(&batch)
            .map_err(|e| format!("Failed to write similarity batch: {e}"))?;
        writer
            .finish()
            .map_err(|e| format!("Failed to finish similarity stream: {e}"))?;
    }
    Ok(out)
}

/// Computes exact top-`k` cosine similarity rows against a 2D Arrow IPC matrix stream/file.
///
/// Returns an Arrow IPC stream with schema `(index: Int64, score: Float64)`.
pub fn top_k_cosine_ipc(
    matrix_ipc: &[u8],
    query_ipc: &[u8],
    k: usize,
    exclude_row: isize,
) -> Result<Vec<u8>, String> {
    let (flat_mat, n_rows, dim) = decode_matrix_ipc(matrix_ipc)?;
    let query = decode_vector_ipc(query_ipc)?;
    if n_rows == 0 || dim == 0 || k == 0 || query.is_empty() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("index", DataType::Int64, false),
            Field::new("score", DataType::Float64, false),
        ]));
        let batch = RecordBatch::new_empty(schema.clone());
        let mut out = Vec::new();
        let mut writer = StreamWriter::try_new(&mut out, &schema)
            .map_err(|e| format!("Failed to create StreamWriter: {e}"))?;
        writer.write(&batch).map_err(|e| e.to_string())?;
        writer.finish().map_err(|e| e.to_string())?;
        return Ok(out);
    }
    if query.len() != dim {
        return Err(format!(
            "Query dimension ({}) does not match matrix dimension ({dim})",
            query.len()
        ));
    }

    let q_norm = query.iter().map(|&v| v * v).sum::<f64>().sqrt();
    let score_row = |r: usize| -> (usize, f64) {
        if exclude_row >= 0 && r == (exclude_row as usize) {
            return (r, f64::NEG_INFINITY);
        }
        if q_norm <= 0.0 {
            return (r, 0.0);
        }
        let row = &flat_mat[r * dim..(r + 1) * dim];
        let mut dot = 0.0f64;
        let mut r_sq = 0.0f64;
        for (&x, &q) in row.iter().zip(query.iter()) {
            dot += x * q;
            r_sq += x * x;
        }
        let denom = (r_sq.sqrt() * q_norm).max(1e-8);
        (r, dot / denom)
    };

    let mut scored: Vec<(usize, f64)> = if n_rows >= 2048 {
        (0..n_rows).into_par_iter().map(score_row).collect()
    } else {
        (0..n_rows).map(score_row).collect()
    };

    scored.retain(|&(_, s)| s.is_finite());
    let take_k = k.min(scored.len());
    if take_k > 0 {
        scored.select_nth_unstable_by(take_k - 1, |a, b| b.1.total_cmp(&a.1));
        scored.truncate(take_k);
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    }

    let indices: Vec<i64> = scored.iter().map(|&(i, _)| i as i64).collect();
    let scores: Vec<f64> = scored.iter().map(|&(_, s)| s).collect();

    let schema = Arc::new(Schema::new(vec![
        Field::new("index", DataType::Int64, false),
        Field::new("score", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(indices)),
            Arc::new(Float64Array::from(scores)),
        ],
    )
    .map_err(|e| format!("Failed to build top_k RecordBatch: {e}"))?;

    let mut out = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut out, &schema)
            .map_err(|e| format!("Failed to create top_k StreamWriter: {e}"))?;
        writer.write(&batch).map_err(|e| e.to_string())?;
        writer.finish().map_err(|e| e.to_string())?;
    }
    Ok(out)
}

/// Computes centroid cosine distance anomaly scores for a subset of row indices in an Arrow IPC matrix.
///
/// Returns an Arrow IPC stream with schema `(pos: Int64, row_index: Int64, anomaly_score: Float64)`.
pub fn centroid_anomalies_ipc(
    matrix_ipc: &[u8],
    indices_ipc: &[u8],
    k: usize,
) -> Result<Vec<u8>, String> {
    let (flat_mat, n_rows, dim) = decode_matrix_ipc(matrix_ipc)?;
    let raw_indices = decode_vector_ipc(indices_ipc)?;
    let valid_indices: Vec<(usize, usize)> = raw_indices
        .iter()
        .enumerate()
        .filter_map(|(pos, &idx)| {
            let r = idx as usize;
            if r < n_rows {
                Some((pos, r))
            } else {
                None
            }
        })
        .collect();

    let schema = Arc::new(Schema::new(vec![
        Field::new("pos", DataType::Int64, false),
        Field::new("row_index", DataType::Int64, false),
        Field::new("anomaly_score", DataType::Float64, false),
    ]));

    if valid_indices.is_empty() || dim == 0 || k == 0 {
        let batch = RecordBatch::new_empty(schema.clone());
        let mut out = Vec::new();
        let mut writer = StreamWriter::try_new(&mut out, &schema).map_err(|e| e.to_string())?;
        writer.write(&batch).map_err(|e| e.to_string())?;
        writer.finish().map_err(|e| e.to_string())?;
        return Ok(out);
    }

    let mut centroid = vec![0.0f64; dim];
    for &(_, r) in &valid_indices {
        let row = &flat_mat[r * dim..(r + 1) * dim];
        for (c, &v) in centroid.iter_mut().zip(row.iter()) {
            *c += v;
        }
    }
    let inv_n = 1.0 / (valid_indices.len() as f64);
    for c in &mut centroid {
        *c *= inv_n;
    }
    let c_norm = centroid.iter().map(|&v| v * v).sum::<f64>().sqrt().max(1e-8);
    for c in &mut centroid {
        *c /= c_norm;
    }

    let mut scored: Vec<(usize, usize, f64)> = valid_indices
        .iter()
        .map(|&(pos, r)| {
            let row = &flat_mat[r * dim..(r + 1) * dim];
            let mut dot = 0.0f64;
            let mut r_sq = 0.0f64;
            for (&x, &c) in row.iter().zip(centroid.iter()) {
                dot += x * c;
                r_sq += x * x;
            }
            let r_norm = r_sq.sqrt().max(1e-8);
            let sim = dot / r_norm;
            (pos, r, 1.0 - sim)
        })
        .collect();

    let take_k = k.min(scored.len());
    if take_k > 0 {
        scored.select_nth_unstable_by(take_k - 1, |a, b| b.2.total_cmp(&a.2));
        scored.truncate(take_k);
        scored.sort_by(|a, b| b.2.total_cmp(&a.2));
    }

    let pos_vec: Vec<i64> = scored.iter().map(|&(p, _, _)| p as i64).collect();
    let row_vec: Vec<i64> = scored.iter().map(|&(_, r, _)| r as i64).collect();
    let score_vec: Vec<f64> = scored.iter().map(|&(_, _, s)| s).collect();

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(pos_vec)),
            Arc::new(Int64Array::from(row_vec)),
            Arc::new(Float64Array::from(score_vec)),
        ],
    )
    .map_err(|e| e.to_string())?;

    let mut out = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut out, &schema).map_err(|e| e.to_string())?;
        writer.write(&batch).map_err(|e| e.to_string())?;
        writer.finish().map_err(|e| e.to_string())?;
    }
    Ok(out)
}

// ============================================================================
// C-ABI Exports for Python (`ctypes`)
// ============================================================================

/// Returns the static version string of the Rust Arrow IPC engine.
#[no_mangle]
pub extern "C" fn jcube_ipc_version() -> *const c_char {
    VERSION_CSTR.as_ptr().cast::<c_char>()
}

/// Frees an [`IpcBuffer`] previously returned by a `jcube_ipc_*` function.
#[no_mangle]
pub extern "C" fn jcube_ipc_free_buffer(buf: IpcBuffer) {
    if !buf.ptr.is_null() && buf.cap > 0 {
        unsafe {
            drop(Vec::from_raw_parts(buf.ptr, buf.len, buf.cap));
        }
    }
}

/// C-ABI wrapper around [`create_ipc_tensor_file`].
///
/// # Safety
/// `path_ptr` and `dtype_ptr` must be valid null-terminated UTF-8 C strings, and `shape_ptr`
/// must point to `ndim` valid `usize` elements.
#[no_mangle]
pub unsafe extern "C" fn jcube_ipc_create_tensor_file(
    path_ptr: *const c_char,
    shape_ptr: *const usize,
    ndim: usize,
    dtype_ptr: *const c_char,
) -> TensorFileLayout {
    if path_ptr.is_null() || shape_ptr.is_null() || dtype_ptr.is_null() || ndim == 0 {
        return TensorFileLayout::err();
    }
    let Ok(path_str) = unsafe { CStr::from_ptr(path_ptr) }.to_str() else {
        return TensorFileLayout::err();
    };
    let Ok(dtype_str) = unsafe { CStr::from_ptr(dtype_ptr) }.to_str() else {
        return TensorFileLayout::err();
    };
    let shape = unsafe { slice::from_raw_parts(shape_ptr, ndim) };
    create_ipc_tensor_file(Path::new(path_str), shape, dtype_str)
        .unwrap_or_else(|_| TensorFileLayout::err())
}

/// C-ABI wrapper around [`inspect_ipc_tensor_file`].
///
/// # Safety
/// `path_ptr` must be a valid null-terminated UTF-8 C string.
#[no_mangle]
pub unsafe extern "C" fn jcube_ipc_inspect_tensor_file(
    path_ptr: *const c_char,
) -> TensorFileLayout {
    if path_ptr.is_null() {
        return TensorFileLayout::err();
    }
    let Ok(path_str) = unsafe { CStr::from_ptr(path_ptr) }.to_str() else {
        return TensorFileLayout::err();
    };
    inspect_ipc_tensor_file(Path::new(path_str)).unwrap_or_else(|_| TensorFileLayout::err())
}

/// C-ABI wrapper around [`weighted_aggregate_ipc`].
///
/// # Safety
/// `ipc_ptr` must point to `ipc_len` valid bytes.
#[no_mangle]
pub unsafe extern "C" fn jcube_ipc_weighted_aggregate(
    ipc_ptr: *const u8,
    ipc_len: usize,
    alpha: f64,
) -> IpcBuffer {
    if ipc_ptr.is_null() && ipc_len > 0 {
        return IpcBuffer::err("Null ipc_ptr");
    }
    let bytes = if ipc_len == 0 {
        &[]
    } else {
        unsafe { slice::from_raw_parts(ipc_ptr, ipc_len) }
    };
    match weighted_aggregate_ipc(bytes, alpha) {
        Ok(out) => IpcBuffer::ok(out),
        Err(e) => IpcBuffer::err(e),
    }
}

/// C-ABI wrapper around [`compute_trend_ipc`].
///
/// # Safety
/// `ipc_ptr` must point to `ipc_len` valid bytes.
#[no_mangle]
pub unsafe extern "C" fn jcube_ipc_compute_trend(
    ipc_ptr: *const u8,
    ipc_len: usize,
    alpha: f64,
) -> IpcBuffer {
    if ipc_ptr.is_null() && ipc_len > 0 {
        return IpcBuffer::err("Null ipc_ptr");
    }
    let bytes = if ipc_len == 0 {
        &[]
    } else {
        unsafe { slice::from_raw_parts(ipc_ptr, ipc_len) }
    };
    match compute_trend_ipc(bytes, alpha) {
        Ok(out) => IpcBuffer::ok(out),
        Err(e) => IpcBuffer::err(e),
    }
}

/// C-ABI wrapper around [`streaming_ema_batch_ipc`].
///
/// # Safety
/// `ipc_ptr` and `state_ptr` must point to valid byte slices of their respective lengths.
#[no_mangle]
pub unsafe extern "C" fn jcube_ipc_streaming_ema_batch(
    ipc_ptr: *const u8,
    ipc_len: usize,
    state_ptr: *const u8,
    state_len: usize,
    alpha: f64,
    last_ts: f64,
    has_last_ts: i32,
    count: u64,
) -> StreamingEmaResult {
    let bytes = if ipc_ptr.is_null() || ipc_len == 0 {
        &[]
    } else {
        unsafe { slice::from_raw_parts(ipc_ptr, ipc_len) }
    };
    let state_bytes = if state_ptr.is_null() || state_len == 0 {
        &[]
    } else {
        unsafe { slice::from_raw_parts(state_ptr, state_len) }
    };
    match streaming_ema_batch_ipc(bytes, state_bytes, alpha, last_ts, has_last_ts != 0, count) {
        Ok((state_ipc, new_ts, new_count)) => StreamingEmaResult {
            state_ipc: IpcBuffer::ok(state_ipc),
            last_timestamp: new_ts,
            count: new_count,
        },
        Err(e) => StreamingEmaResult {
            state_ipc: IpcBuffer::err(e),
            last_timestamp: last_ts,
            count,
        },
    }
}

/// C-ABI wrapper around [`batch_cosine_similarity_ipc`].
///
/// # Safety
/// `ipc_a_ptr` and `ipc_b_ptr` must point to valid byte slices.
#[no_mangle]
pub unsafe extern "C" fn jcube_ipc_batch_cosine_similarity(
    ipc_a_ptr: *const u8,
    ipc_a_len: usize,
    ipc_b_ptr: *const u8,
    ipc_b_len: usize,
) -> IpcBuffer {
    let bytes_a = if ipc_a_ptr.is_null() || ipc_a_len == 0 {
        &[]
    } else {
        unsafe { slice::from_raw_parts(ipc_a_ptr, ipc_a_len) }
    };
    let bytes_b = if ipc_b_ptr.is_null() || ipc_b_len == 0 {
        &[]
    } else {
        unsafe { slice::from_raw_parts(ipc_b_ptr, ipc_b_len) }
    };
    match batch_cosine_similarity_ipc(bytes_a, bytes_b) {
        Ok(out) => IpcBuffer::ok(out),
        Err(e) => IpcBuffer::err(e),
    }
}

/// C-ABI wrapper around [`top_k_cosine_ipc`].
///
/// # Safety
/// `matrix_ptr` and `query_ptr` must point to valid byte slices.
#[no_mangle]
pub unsafe extern "C" fn jcube_ipc_top_k_cosine(
    matrix_ptr: *const u8,
    matrix_len: usize,
    query_ptr: *const u8,
    query_len: usize,
    k: usize,
    exclude_row: isize,
) -> IpcBuffer {
    let mat_bytes = if matrix_ptr.is_null() || matrix_len == 0 {
        &[]
    } else {
        unsafe { slice::from_raw_parts(matrix_ptr, matrix_len) }
    };
    let q_bytes = if query_ptr.is_null() || query_len == 0 {
        &[]
    } else {
        unsafe { slice::from_raw_parts(query_ptr, query_len) }
    };
    match top_k_cosine_ipc(mat_bytes, q_bytes, k, exclude_row) {
        Ok(out) => IpcBuffer::ok(out),
        Err(e) => IpcBuffer::err(e),
    }
}

/// C-ABI wrapper around [`centroid_anomalies_ipc`].
///
/// # Safety
/// `matrix_ptr` and `indices_ptr` must point to valid byte slices.
#[no_mangle]
pub unsafe extern "C" fn jcube_ipc_centroid_anomalies(
    matrix_ptr: *const u8,
    matrix_len: usize,
    indices_ptr: *const u8,
    indices_len: usize,
    k: usize,
) -> IpcBuffer {
    let mat_bytes = if matrix_ptr.is_null() || matrix_len == 0 {
        &[]
    } else {
        unsafe { slice::from_raw_parts(matrix_ptr, matrix_len) }
    };
    let idx_bytes = if indices_ptr.is_null() || indices_len == 0 {
        &[]
    } else {
        unsafe { slice::from_raw_parts(indices_ptr, indices_len) }
    };
    match centroid_anomalies_ipc(mat_bytes, idx_bytes, k) {
        Ok(out) => IpcBuffer::ok(out),
        Err(e) => IpcBuffer::err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use memmap2::MmapMut;

    fn build_test_sequence_ipc(timestamps: &[f64], rows: &[Vec<f64>]) -> Vec<u8> {
        let dim = rows.first().map_or(0, Vec::len);
        let item_field = Arc::new(Field::new("item", DataType::Float64, false));
        let schema = Arc::new(Schema::new(vec![
            Field::new("timestamp", DataType::Float64, false),
            Field::new(
                "embedding",
                DataType::FixedSizeList(item_field.clone(), dim as i32),
                false,
            ),
        ]));
        let flat: Vec<f64> = rows.iter().flatten().copied().collect();
        let ts_arr: ArrayRef = Arc::new(Float64Array::from(timestamps.to_vec()));
        let val_arr: ArrayRef = Arc::new(Float64Array::from(flat));
        let fsl_arr: ArrayRef =
            Arc::new(FixedSizeListArray::try_new(item_field, dim as i32, val_arr, None).unwrap());
        let batch = RecordBatch::try_new(schema.clone(), vec![ts_arr, fsl_arr]).unwrap();
        let mut buf = Vec::new();
        let mut writer = StreamWriter::try_new(&mut buf, &schema).unwrap();
        writer.write(&batch).unwrap();
        writer.finish().unwrap();
        buf
    }

    fn build_test_matrix_ipc(rows: &[Vec<f64>]) -> Vec<u8> {
        let dim = rows.first().map_or(0, Vec::len);
        let item_field = Arc::new(Field::new("item", DataType::Float64, false));
        let schema = Arc::new(Schema::new(vec![Field::new(
            "matrix",
            DataType::FixedSizeList(item_field.clone(), dim as i32),
            false,
        )]));
        let flat: Vec<f64> = rows.iter().flatten().copied().collect();
        let val_arr: ArrayRef = Arc::new(Float64Array::from(flat));
        let fsl_arr: ArrayRef =
            Arc::new(FixedSizeListArray::try_new(item_field, dim as i32, val_arr, None).unwrap());
        let batch = RecordBatch::try_new(schema.clone(), vec![fsl_arr]).unwrap();
        let mut buf = Vec::new();
        let mut writer = StreamWriter::try_new(&mut buf, &schema).unwrap();
        writer.write(&batch).unwrap();
        writer.finish().unwrap();
        buf
    }

    #[test]
    fn test_zero_copy_arrow_ipc_tensor_file_mutation() {
        let tmp_path = std::env::temp_dir().join(format!(
            "jcube_rust_test_{}.arrow",
            std::process::id()
        ));
        let layout = create_ipc_tensor_file(&tmp_path, &[3, 2], "int64").unwrap();
        assert_eq!(layout.status, 0);
        assert_eq!(layout.num_elements, 6);
        assert_eq!(layout.elem_size, 8);
        assert_eq!(layout.data_byte_len, 48);
        assert_eq!(layout.data_offset % 8, 0);

        // Mutate the physical Arrow IPC buffer via MmapMut
        {
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&tmp_path)
                .unwrap();
            let mut mmap = unsafe { MmapMut::map_mut(&file).unwrap() };
            let slice_bytes =
                &mut mmap[layout.data_offset..layout.data_offset + layout.data_byte_len];
            let values: &mut [i64] = unsafe {
                slice::from_raw_parts_mut(slice_bytes.as_mut_ptr().cast::<i64>(), 6)
            };
            values.copy_from_slice(&[10, 20, 30, 40, 50, 60]);
            mmap.flush().unwrap();
        }

        // Read back with standard Arrow IPC FileReader and verify our in-place mutation!
        let file = OpenOptions::new().read(true).open(&tmp_path).unwrap();
        let mut reader = FileReader::try_new(file, None).unwrap();
        assert_eq!(
            reader.schema().metadata().get("jcube.shape").map(String::as_str),
            Some("3,2")
        );
        let batch = reader.next().unwrap().unwrap();
        let arr = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(arr.values(), &[10, 20, 30, 40, 50, 60]);

        let _ = std::fs::remove_file(&tmp_path);
    }

    #[test]
    fn test_weighted_aggregate_and_trend_ipc() {
        let ipc = build_test_sequence_ipc(
            &[1.0, 2.0, 3.0],
            &[vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]],
        );
        let agg_ipc = weighted_aggregate_ipc(&ipc, 0.0).unwrap();
        let agg = decode_vector_ipc(&agg_ipc).unwrap();
        assert!((agg[0] - 3.0).abs() < 1e-12);
        assert!((agg[1] - 4.0).abs() < 1e-12);

        let trend_ipc = compute_trend_ipc(&ipc, 0.0).unwrap();
        let trend = decode_vector_ipc(&trend_ipc).unwrap();
        assert!((trend[0] - 2.0).abs() < 1e-12);
        assert!((trend[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_streaming_ema_and_cosine_ipc() {
        let ipc = build_test_sequence_ipc(&[1.0, 2.0], &[vec![1.0, 0.0], vec![0.0, 1.0]]);
        let (state_ipc, last_ts, count) =
            streaming_ema_batch_ipc(&ipc, &[], 0.1, 0.0, false, 0).unwrap();
        assert_eq!(count, 2);
        assert!((last_ts - 2.0).abs() < 1e-12);
        let state = decode_vector_ipc(&state_ipc).unwrap();
        assert_eq!(state.len(), 2);

        let mat_a = build_test_matrix_ipc(&[vec![1.0, 0.0], vec![0.0, 1.0]]);
        let mat_b = build_test_matrix_ipc(&[vec![1.0, 0.0], vec![1.0, 1.0]]);
        let sim_ipc = batch_cosine_similarity_ipc(&mat_a, &mat_b).unwrap();
        let (sims, rows, cols) = decode_matrix_ipc(&sim_ipc).unwrap();
        assert_eq!((rows, cols), (2, 2));
        assert!((sims[0] - 1.0).abs() < 1e-12);
        assert!((sims[1] - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12);
    }
}
