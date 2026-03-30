"""Zero-copy embedding push to Milvus via Arrow Flight do_exchange.

Splits into multiple do_exchange sessions to avoid timeout.
Each session pushes ~2M vectors (40 batches × 50K).

Usage:
    modal run --detach event_jepa_cube/jcube_bridge.py
"""
import modal

app = modal.App("jcube-milvus-bridge")
cache_vol = modal.Volume.from_name("jepa-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("jcube-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch>=2.6", "numpy>=2.0", "pyarrow>=18.0")
)

FLIGHT_URL = "grpc+tls://flight.getjai.com:443"
FLIGHT_SECRET = "mycelia_dev_flight_secret"
COLLECTION = "jcube_twin_v5"
BATCH_SIZE = 50_000
SESSION_SIZE = 200_000  # vectors per session (smaller to allow flush between sessions)
KEY_TYPES = {"INTERNACAO", "PACIENTE", "FATURA", "CID", "TUSS", "HOSPITAL", "EVOLUCAO", "AUDITORIA", "MEDICO"}


@app.function(
    volumes={"/cache": cache_vol, "/data": data_vol},
    image=image,
    memory=65536,
    cpu=8,
    timeout=7200,
)
def push_embeddings():
    import time
    import numpy as np
    import torch
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    import pyarrow.flight as flight
    import json

    cache_vol.reload()
    data_vol.reload()

    t0 = time.time()

    # 1. Load
    print("[1/4] Loading node vocabulary...")
    table = pq.read_table("/data/jcube_graph.parquet", columns=["subject_id", "object_id"])
    all_nodes = pa.chunked_array(table.column("subject_id").chunks + table.column("object_id").chunks)
    names_np = pc.unique(all_nodes).to_numpy(zero_copy_only=False)
    del table, all_nodes
    print(f"  {len(names_np):,} nodes in {time.time()-t0:.1f}s")

    print("[2/4] Loading V5 prod embeddings...")
    state = torch.load("/cache/tkg-v5/node_emb_epoch_1.pt", map_location="cpu", weights_only=True)
    emb = (state if isinstance(state, torch.Tensor) else list(state.values())[0]).float().numpy()
    print(f"  Shape: {emb.shape}")
    assert len(names_np) == emb.shape[0]

    # 2. Filter
    print("[3/4] Filtering key entities...")
    selected = [i for i, n in enumerate(names_np) if any(f"_CD_{et}_" in str(n) for et in KEY_TYPES)]
    sel_idx = np.array(selected)
    n_total = len(sel_idx)
    print(f"  {n_total:,} key entities")

    # 3. Stream in sessions
    print(f"[4/4] Streaming via do_exchange in sessions of {SESSION_SIZE:,}...")
    schema = pa.schema([("id", pa.utf8()), ("vector", pa.list_(pa.float32()))])

    client = flight.FlightClient(FLIGHT_URL)
    options = flight.FlightCallOptions(
        headers=[(b"authorization", f"Bearer {FLIGHT_SECRET}".encode())],
        timeout=540,  # 9 min per session
    )

    total_pushed = 0
    t1 = time.time()
    session_num = 0

    for session_start in range(0, n_total, SESSION_SIZE):
        session_end = min(session_start + SESSION_SIZE, n_total)
        session_idx = sel_idx[session_start:session_end]
        session_num += 1

        # Use do_action (collection_insert_vectors) instead of do_exchange
        # do_exchange is only implemented for embed/ingest commands
        ids = [str(names_np[i]) for i in session_idx]
        vecs = [emb[i].tolist() for i in session_idx]

        # Split into sub-batches of 10K (smaller to avoid Milvus buffer OOM)
        SUB_BATCH = 10000
        session_inserted = 0
        for sb_start in range(0, len(ids), SUB_BATCH):
            sb_ids = ids[sb_start:sb_start + SUB_BATCH]
            sb_vecs = vecs[sb_start:sb_start + SUB_BATCH]
            result = list(client.do_action(
                flight.Action("collection_insert_vectors", json.dumps({
                    "collection_name": COLLECTION,
                    "ids": sb_ids,
                    "vectors": sb_vecs,
                }).encode()),
                options
            ))
            resp = json.loads(result[0].body.to_pybytes())
            session_inserted += resp.get("inserted", 0)

        # Flush after each session to free Milvus insert buffer
        try:
            list(client.do_action(
                flight.Action("collection_flush", json.dumps({
                    "collection_name": COLLECTION,
                }).encode()),
                options
            ))
        except Exception:
            # Flush not implemented — wait for auto-flush
            import time as _t; _t.sleep(5)

        total_pushed += session_inserted
        elapsed = time.time() - t1
        rate = total_pushed / max(elapsed, 1)
        print(f"  Session {session_num}: +{session_inserted:,} | total={total_pushed:,} / {n_total:,} | {rate:.0f}/s | {elapsed:.0f}s")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"DONE: {total_pushed:,} vectors pushed in {elapsed:.1f}s")
    print(f"{'='*60}")

    # Verify
    import time as _t; _t.sleep(3)
    print("\nVerification search...")
    test_vec = emb[sel_idx[0]].tolist()
    test_id = str(names_np[sel_idx[0]])
    results = list(client.do_action(
        flight.Action("search", json.dumps({
            "collection": COLLECTION,
            "query_vectors": [test_vec],
            "top_k": 5,
            "metric": "COSINE",
        }).encode()),
        options
    ))
    result_table = pa.ipc.open_stream(results[0].body.to_pybytes()).read_all()
    print(f"  Query: {test_id}")
    for i in range(result_table.num_rows):
        print(f"    rank={result_table.column('rank')[i].as_py()} id={result_table.column('result_id')[i].as_py()} sim={result_table.column('distance')[i].as_py():.4f}")


@app.local_entrypoint()
def main():
    push_embeddings.remote()
