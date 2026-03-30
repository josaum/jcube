"""V6.2 Epoch 1 Verification — runs on Modal CPU.

Executes the 3-step verification protocol:
1. Delta check: did embeddings move from warm-start?
2. Structural health: isotropy, inter-type distance, anomaly z-max
3. Linear probes: Glosa AUC, LOS MAE

Usage:
    modal run event_jepa_cube/verify_v62_epoch1.py
"""
import modal

app = modal.App("jcube-verify-v62")
data_volume = modal.Volume.from_name("jcube-data")
cache_volume = modal.Volume.from_name("jepa-cache")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch>=2.6", "numpy>=2.0", "pyarrow>=18.0", "scikit-learn>=1.4", "duckdb>=1.0.0")
)


@app.function(image=image, volumes={"/data": data_volume, "/cache": cache_volume}, memory=65536, timeout=3600)
def verify():
    import os, time
    import torch
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    import duckdb
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import roc_auc_score, average_precision_score

    cache_volume.reload()
    data_volume.reload()

    WARM_START = "/cache/tkg-v6/node_emb_epoch_3.pt"
    V62_EP1 = "/cache/tkg-v6.2/node_emb_epoch_1.pt"
    V62_EP2 = "/cache/tkg-v6.2/node_emb_epoch_2.pt"
    V62_EP3 = "/cache/tkg-v6.2/node_emb_epoch_3.pt"
    V62_EP4 = "/cache/tkg-v6.2/node_emb_epoch_4.pt"
    V62_EP5 = "/cache/tkg-v6.2/node_emb_epoch_5.pt"
    GRAPH = "/data/jcube_graph.parquet"
    DB = "/data/aggregated_fixed_union.db"

    print("=" * 70)
    print("V6.2 EPOCH 1 VERIFICATION PROTOCOL")
    print("=" * 70)

    # Check what's available
    for label, path in [("Warm-start (V6 ep3)", WARM_START), ("V6.2 epoch 1", V62_EP1), ("V6.2 epoch 2", V62_EP2), ("V6.2 epoch 3 (TGN)", V62_EP3), ("V6.2 epoch 4 (TGN)", V62_EP4), ("V6.2 epoch 5 (TGN)", V62_EP5)]:
        if os.path.exists(path):
            sz = os.path.getsize(path) / 1e9
            print(f"  {label:25s}  {path}  ({sz:.1f} GB)")
        else:
            print(f"  {label:25s}  NOT FOUND")

    # Find the latest V6.2 checkpoint
    target = None
    for p in [V62_EP5, V62_EP4, V62_EP3, V62_EP2, V62_EP1]:
        if os.path.exists(p):
            target = p
            break

    if target is None:
        print("\n  NO V6.2 CHECKPOINTS FOUND — training may not have completed epoch 1 yet.")
        # List what's in tkg-v6.2/
        v62_dir = "/cache/tkg-v6.2"
        if os.path.exists(v62_dir):
            print(f"\n  Contents of {v62_dir}:")
            for f in sorted(os.listdir(v62_dir)):
                sz = os.path.getsize(os.path.join(v62_dir, f)) / 1e9
                print(f"    {f:40s}  {sz:.2f} GB")
        else:
            print(f"  {v62_dir} does not exist")
        return

    print(f"\n  Target checkpoint: {target}")

    # ================================================================
    # STEP 1: DELTA VERIFICATION
    # ================================================================
    print("\n" + "=" * 70)
    print("[STEP 1] DELTA VERIFICATION — Did embeddings move?")
    print("=" * 70)

    ws = torch.load(WARM_START, map_location="cpu", weights_only=True)
    if isinstance(ws, dict) and "weight" in ws:
        ws = ws["weight"]
    ws = ws.float()

    e1 = torch.load(target, map_location="cpu", weights_only=True)
    if isinstance(e1, dict) and "weight" in e1:
        e1 = e1["weight"]
    e1 = e1.float()

    diff = (ws - e1).abs()
    print(f"  Warm-start shape: {ws.shape}")
    print(f"  V6.2 shape:       {e1.shape}")
    print(f"  Max movement:     {diff.max().item():.6f}")
    print(f"  Mean movement:    {diff.mean().item():.6f}")
    print(f"  Median movement:  {diff.median().item():.6f}")

    # Per-row (node) movement
    row_movement = diff.mean(dim=1)
    moved_nodes = (row_movement > 1e-6).sum().item()
    total_nodes = row_movement.shape[0]
    print(f"  Nodes that moved:  {moved_nodes:,} / {total_nodes:,} ({100*moved_nodes/total_nodes:.1f}%)")

    if diff.max().item() < 1e-8:
        print("\n  FAIL: Embeddings did NOT move. Delta sync or gradient flow broken.")
        return
    else:
        print(f"\n  PASS: Embeddings moved! Max delta = {diff.max().item():.6f}")

    del ws, diff  # free memory

    # ================================================================
    # STEP 2: STRUCTURAL HEALTH CHECK
    # ================================================================
    print("\n" + "=" * 70)
    print("[STEP 2] STRUCTURAL HEALTH — Isotropy, Inter-dist, Anomaly z")
    print("=" * 70)

    emb = e1.numpy()
    del e1

    # Load vocab
    t0 = time.time()
    table = pq.read_table(GRAPH, columns=["subject_id", "object_id"])
    all_nodes = pa.chunked_array(table.column("subject_id").chunks + table.column("object_id").chunks)
    unique_nodes = pc.unique(all_nodes)
    node_names = unique_nodes.to_numpy(zero_copy_only=False).astype(object)
    del table, all_nodes, unique_nodes
    print(f"  Vocab: {len(node_names):,} nodes in {time.time()-t0:.1f}s")

    assert len(node_names) == emb.shape[0]

    # Build type masks
    node_to_idx = {}
    entity_type_mask = {}
    for i, name in enumerate(node_names):
        sname = str(name)
        node_to_idx[sname] = i
        rest = sname.split("/", 1)[-1] if "/" in sname else sname
        parts = rest.split("_")
        if len(parts) >= 3 and parts[0] == "ID" and parts[1] == "CD":
            etype = parts[2]
            if etype not in entity_type_mask:
                entity_type_mask[etype] = []
            entity_type_mask[etype].append(i)
    for etype in entity_type_mask:
        idx_arr = np.array(entity_type_mask[etype])
        mask = np.zeros(len(node_names), dtype=bool)
        mask[idx_arr] = True
        entity_type_mask[etype] = mask

    # Isotropy
    per_dim_var = np.var(emb, axis=0)
    isotropy = per_dim_var.min() / max(per_dim_var.max(), 1e-8)
    effective_dims = np.sum(per_dim_var > per_dim_var.max() * 0.01)
    print(f"  Isotropy:         {isotropy:.4f}  {'PASS' if isotropy > 0.050 else 'WARN (<0.050)'}")
    print(f"  Effective dims:   {effective_dims}/{emb.shape[1]}")

    # Inter-type distance
    type_centroids = {}
    for etype, mask in entity_type_mask.items():
        vecs = emb[mask]
        if len(vecs) >= 10:
            type_centroids[etype] = vecs.mean(axis=0)
    if len(type_centroids) >= 2:
        from itertools import combinations
        centroid_vecs = np.array(list(type_centroids.values()))
        inter_dists = [np.linalg.norm(centroid_vecs[i] - centroid_vecs[j])
                       for i, j in combinations(range(len(centroid_vecs)), 2)]
        inter_dist = np.mean(inter_dists)
    else:
        inter_dist = 0.0
    print(f"  Inter-type dist:  {inter_dist:.4f}")

    # Anomaly z-max
    if "INTERNACAO" in entity_type_mask:
        mask = entity_type_mask["INTERNACAO"]
        vecs = emb[mask]
        centroid = vecs.mean(axis=0, keepdims=True)
        dists = np.linalg.norm(vecs - centroid, axis=1)
        z = (dists - dists.mean()) / max(dists.std(), 1e-8)
        z_max = z.max()
        print(f"  Anomaly z-max:    {z_max:.2f}  {'WARN (>30)' if z_max > 30 else 'OK'}")

    # ================================================================
    # STEP 3: LINEAR PROBES
    # ================================================================
    print("\n" + "=" * 70)
    print("[STEP 3] LINEAR PROBES — Glosa AUC + LOS MAE")
    print("=" * 70)

    # L2 normalize
    norms = np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-8)
    emb_normed = emb / norms

    con = duckdb.connect(DB, read_only=True)

    # Glosa
    glosa_rows = con.execute("""
        SELECT source_db, CAST(ID_CD_INTERNACAO AS VARCHAR) AS eid,
               CASE WHEN SUM(CASE WHEN FL_GLOSA = 'S' THEN 1 ELSE 0 END) > 0 THEN 1 ELSE 0 END AS has_glosa
        FROM agg_tb_fatura_fatu WHERE ID_CD_INTERNACAO IS NOT NULL AND source_db IS NOT NULL
        GROUP BY source_db, ID_CD_INTERNACAO
    """).fetchall()

    los_rows = con.execute("""
        SELECT source_db, CAST(ID_CD_INTERNACAO AS VARCHAR) AS eid,
               DATEDIFF('day', DH_ADMISSAO_HOSP, DH_FINALIZACAO) AS los
        FROM agg_tb_capta_internacao_cain
        WHERE DH_ADMISSAO_HOSP IS NOT NULL AND DH_FINALIZACAO IS NOT NULL
          AND source_db IS NOT NULL AND DATEDIFF('day', DH_ADMISSAO_HOSP, DH_FINALIZACAO) BETWEEN 1 AND 365
    """).fetchall()
    con.close()

    # Glosa probe
    X_list, y_list = [], []
    for source_db, eid, label in glosa_rows:
        for key in [f"{source_db}/ID_CD_INTERNACAO_{eid}", f"ID_CD_INTERNACAO_{eid}"]:
            if key in node_to_idx:
                X_list.append(emb_normed[node_to_idx[key]])
                y_list.append(float(label))
                break
    X, y = np.array(X_list), np.array(y_list)
    perm = np.random.RandomState(42).permutation(len(y))
    split = int(len(y) * 0.8)

    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=42)
    clf.fit(X[perm[:split]], y[perm[:split]])
    probs = clf.predict_proba(X[perm[split:]])[:, 1]
    g_auc = roc_auc_score(y[perm[split:]], probs)
    g_ap = average_precision_score(y[perm[split:]], probs)
    print(f"  Glosa AUC:        {g_auc:.4f}  {'IMPROVED' if g_auc > 0.510 else 'baseline (~0.50)'}")
    print(f"  Glosa AP:         {g_ap:.4f}")

    # LOS probe
    X_list, y_list = [], []
    for source_db, eid, los in los_rows:
        for key in [f"{source_db}/ID_CD_INTERNACAO_{eid}", f"ID_CD_INTERNACAO_{eid}"]:
            if key in node_to_idx:
                X_list.append(emb_normed[node_to_idx[key]])
                y_list.append(float(los))
                break
    X, y = np.array(X_list), np.array(y_list)
    perm = np.random.RandomState(42).permutation(len(y))
    split = int(len(y) * 0.8)

    reg = Ridge(alpha=1.0)
    reg.fit(X[perm[:split]], np.log1p(y[perm[:split]]))
    preds = np.expm1(reg.predict(X[perm[split:]]))
    actuals = y[perm[split:]]
    ss_res = np.sum((actuals - preds) ** 2)
    ss_tot = np.sum((actuals - actuals.mean()) ** 2)
    l_r2 = 1.0 - ss_res / max(ss_tot, 1e-8)
    l_mae = np.mean(np.abs(actuals - preds))
    l_medae = np.median(np.abs(actuals - preds))

    print(f"  LOS R²:           {l_r2:.4f}")
    print(f"  LOS MAE:          {l_mae:.2f} days")
    print(f"  LOS MedAE:        {l_medae:.2f} days")

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 70)
    print("V6.2 EPOCH 1 VERIFICATION SUMMARY")
    print("=" * 70)
    print(f"  Delta moved:      {'YES' if diff.max().item() > 1e-8 else 'NO'}" if 'diff' in dir() else "  Delta:            checked above")
    print(f"  Isotropy:         {isotropy:.4f}  (target > 0.050)")
    print(f"  Glosa AUC:        {g_auc:.4f}  (target > 0.510)")
    print(f"  LOS MAE:          {l_mae:.2f}d  (target < 7.17)")
    print(f"  Anomaly z-max:    {z_max:.2f}  (target < 30)")
    print("=" * 70)


@app.local_entrypoint()
def main():
    verify.remote()
