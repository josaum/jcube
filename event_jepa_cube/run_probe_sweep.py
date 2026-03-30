"""Sweep linear probes across V6 epoch checkpoints to find temporal sweet spot.

Loads vocab + labels ONCE, then iterates over each checkpoint.
Fast: LogisticRegression + Ridge only (no LightGBM).

Usage:
    modal run --detach event_jepa_cube/run_probe_sweep.py
"""
import modal

scale_app = modal.App("jcube-probe-sweep")
data_volume = modal.Volume.from_name("jcube-data")
cache_volume = modal.Volume.from_name("jepa-cache")

probe_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.6",
        "numpy>=2.0",
        "pyarrow>=18.0",
        "scikit-learn>=1.4",
        "duckdb>=1.0.0",
    )
)


@scale_app.function(
    image=probe_image,
    volumes={"/data": data_volume, "/cache": cache_volume},
    memory=65536,  # 64GB — loading multiple 18GB checkpoints
    timeout=7200,
)
def run_sweep():
    import os
    import time
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

    GRAPH = "/data/jcube_graph.parquet"
    DB = "/data/aggregated_fixed_union.db"

    # Checkpoints to sweep: V6 epochs 3-10 + V6.1 final
    CHECKPOINTS = []
    for ep in range(3, 11):
        p = f"/cache/tkg-v6/node_emb_epoch_{ep}.pt"
        if os.path.exists(p):
            CHECKPOINTS.append((f"V6-ep{ep}", p))
    # V6.1 epochs
    for ep in range(1, 6):
        p = f"/cache/tkg-v6.1/node_emb_epoch_{ep}.pt"
        if os.path.exists(p):
            CHECKPOINTS.append((f"V6.1-ep{ep}", p))
    p_final = "/cache/tkg-v6.1/node_embeddings.pt"
    if os.path.exists(p_final):
        CHECKPOINTS.append(("V6.1-final", p_final))

    print("=" * 80)
    print("JCUBE PROBE SWEEP — Finding Temporal Sweet Spot")
    print("=" * 80)
    print(f"\nCheckpoints found: {len(CHECKPOINTS)}")
    for name, path in CHECKPOINTS:
        sz = os.path.getsize(path) / 1e9
        print(f"  {name:15s}  {path}  ({sz:.1f} GB)")

    # ================================================================
    # 1. Load vocab (once)
    # ================================================================
    print("\n[1] Loading node vocabulary...")
    t0 = time.time()
    table = pq.read_table(GRAPH, columns=["subject_id", "object_id"])
    all_nodes = pa.chunked_array(table.column("subject_id").chunks + table.column("object_id").chunks)
    unique_nodes = pc.unique(all_nodes)
    node_names = unique_nodes.to_numpy(zero_copy_only=False).astype(object)
    del table, all_nodes, unique_nodes
    print(f"  {len(node_names):,} nodes in {time.time()-t0:.1f}s")

    # Build node index
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

    # ================================================================
    # 2. Load labels (once)
    # ================================================================
    print("\n[2] Loading labels from DuckDB...")
    con = duckdb.connect(DB, read_only=True)

    # Glosa labels
    glosa_rows = con.execute("""
        SELECT source_db, CAST(ID_CD_INTERNACAO AS VARCHAR) AS eid,
               CASE WHEN SUM(CASE WHEN FL_GLOSA = 'S' THEN 1 ELSE 0 END) > 0 THEN 1 ELSE 0 END AS has_glosa
        FROM agg_tb_fatura_fatu
        WHERE ID_CD_INTERNACAO IS NOT NULL AND source_db IS NOT NULL
        GROUP BY source_db, ID_CD_INTERNACAO
    """).fetchall()

    # LOS labels
    los_rows = con.execute("""
        SELECT source_db, CAST(ID_CD_INTERNACAO AS VARCHAR) AS eid,
               DATEDIFF('day', DH_ADMISSAO_HOSP, DH_FINALIZACAO) AS los
        FROM agg_tb_capta_internacao_cain
        WHERE DH_ADMISSAO_HOSP IS NOT NULL AND DH_FINALIZACAO IS NOT NULL
          AND source_db IS NOT NULL
          AND DATEDIFF('day', DH_ADMISSAO_HOSP, DH_FINALIZACAO) BETWEEN 1 AND 365
    """).fetchall()
    con.close()

    # Map to indices (once)
    glosa_idx, glosa_labels = [], []
    for source_db, eid, label in glosa_rows:
        for key in [f"{source_db}/ID_CD_INTERNACAO_{eid}", f"ID_CD_INTERNACAO_{eid}"]:
            if key in node_to_idx:
                glosa_idx.append(node_to_idx[key])
                glosa_labels.append(float(label))
                break
    glosa_idx = np.array(glosa_idx)
    glosa_labels = np.array(glosa_labels)

    los_idx, los_labels = [], []
    for source_db, eid, los in los_rows:
        for key in [f"{source_db}/ID_CD_INTERNACAO_{eid}", f"ID_CD_INTERNACAO_{eid}"]:
            if key in node_to_idx:
                los_idx.append(node_to_idx[key])
                los_labels.append(float(los))
                break
    los_idx = np.array(los_idx)
    los_labels = np.array(los_labels)

    print(f"  Glosa: {len(glosa_idx):,} samples ({glosa_labels.sum():.0f} positive, {100*glosa_labels.mean():.1f}%)")
    print(f"  LOS:   {len(los_idx):,} samples (median={np.median(los_labels):.0f}d, mean={los_labels.mean():.1f}d)")

    # Fixed train/test splits
    g_perm = np.random.RandomState(42).permutation(len(glosa_idx))
    g_split = int(len(glosa_idx) * 0.8)
    l_perm = np.random.RandomState(42).permutation(len(los_idx))
    l_split = int(len(los_idx) * 0.8)

    # ================================================================
    # 3. Sweep checkpoints
    # ================================================================
    print("\n" + "=" * 80)
    print(f"{'Checkpoint':15s} | {'Glosa AUC':>9s} | {'Glosa AP':>8s} | {'LOS R²':>7s} | {'LOS MAE':>7s} | {'MedAE':>6s} | {'Isotropy':>8s} | {'InterDist':>9s} | {'Anom z-max':>10s}")
    print("-" * 80)

    results = []

    for ckpt_name, ckpt_path in CHECKPOINTS:
        t1 = time.time()
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "weight" in state:
            emb = state["weight"].float().numpy()
        elif isinstance(state, torch.Tensor):
            emb = state.float().numpy()
        else:
            emb = list(state.values())[0].float().numpy()
        load_time = time.time() - t1

        assert len(node_names) == emb.shape[0], f"{ckpt_name}: shape mismatch {emb.shape[0]} vs {len(node_names)}"

        # L2-normalize
        norms = np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-8)
        emb_normed = emb / norms

        # --- Glosa (LogisticRegression) ---
        X_g = emb_normed[glosa_idx]
        y_g = glosa_labels
        X_g_train, X_g_test = X_g[g_perm[:g_split]], X_g[g_perm[g_split:]]
        y_g_train, y_g_test = y_g[g_perm[:g_split]], y_g[g_perm[g_split:]]

        try:
            clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=42, n_jobs=-1)
            clf.fit(X_g_train, y_g_train)
            g_probs = clf.predict_proba(X_g_test)[:, 1]
            g_auc = roc_auc_score(y_g_test, g_probs)
            g_ap = average_precision_score(y_g_test, g_probs)
        except Exception:
            g_auc, g_ap = 0.5, 0.0

        # --- LOS (Ridge) ---
        X_l = emb_normed[los_idx]
        y_l = los_labels
        X_l_train, X_l_test = X_l[l_perm[:l_split]], X_l[l_perm[l_split:]]
        y_l_train, y_l_test = y_l[l_perm[:l_split]], y_l[l_perm[l_split:]]

        reg = Ridge(alpha=1.0)
        reg.fit(X_l_train, np.log1p(y_l_train))
        l_preds = np.expm1(reg.predict(X_l_test))
        l_actuals = y_l_test

        ss_res = np.sum((l_actuals - l_preds) ** 2)
        ss_tot = np.sum((l_actuals - l_actuals.mean()) ** 2)
        l_r2 = 1.0 - ss_res / max(ss_tot, 1e-8)
        l_mae = np.mean(np.abs(l_actuals - l_preds))
        l_medae = np.median(np.abs(l_actuals - l_preds))

        # --- Isotropy ---
        per_dim_var = np.var(emb, axis=0)
        isotropy = per_dim_var.min() / max(per_dim_var.max(), 1e-8)

        # --- Inter-type distance ---
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

        # --- Anomaly z-max ---
        if "INTERNACAO" in entity_type_mask:
            mask = entity_type_mask["INTERNACAO"]
            vecs = emb[mask]
            centroid = vecs.mean(axis=0, keepdims=True)
            dists = np.linalg.norm(vecs - centroid, axis=1)
            z = (dists - dists.mean()) / max(dists.std(), 1e-8)
            z_max = z.max()
        else:
            z_max = 0.0

        row = {
            "name": ckpt_name, "glosa_auc": g_auc, "glosa_ap": g_ap,
            "los_r2": l_r2, "los_mae": l_mae, "los_medae": l_medae,
            "isotropy": isotropy, "inter_dist": inter_dist, "z_max": z_max,
        }
        results.append(row)

        print(f"{ckpt_name:15s} | {g_auc:9.4f} | {g_ap:8.4f} | {l_r2:7.4f} | {l_mae:7.2f} | {l_medae:6.2f} | {isotropy:8.4f} | {inter_dist:9.4f} | {z_max:10.2f}  ({load_time:.0f}s)")

        # Free memory
        del emb, emb_normed, state
        import gc; gc.collect()

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 80)
    print("SUMMARY — Best checkpoint per metric:")
    print("=" * 80)

    best_glosa = max(results, key=lambda r: r["glosa_auc"])
    best_los = min(results, key=lambda r: r["los_mae"])
    best_isotropy = max(results, key=lambda r: r["isotropy"])
    best_zmax = max(results, key=lambda r: r["z_max"])

    print(f"  Best Glosa AUC:  {best_glosa['name']:15s}  AUC={best_glosa['glosa_auc']:.4f}")
    print(f"  Best LOS MAE:    {best_los['name']:15s}  MAE={best_los['los_mae']:.2f}d")
    print(f"  Best Isotropy:   {best_isotropy['name']:15s}  ratio={best_isotropy['isotropy']:.4f}")
    print(f"  Best Anomaly z:  {best_zmax['name']:15s}  z_max={best_zmax['z_max']:.2f}")

    print("\n" + "=" * 80)
    print("SWEEP COMPLETE")
    print("=" * 80)


@scale_app.local_entrypoint()
def main():
    run_sweep.remote()
