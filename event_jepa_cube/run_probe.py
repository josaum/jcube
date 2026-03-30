"""Run probes against trained embeddings on Modal CPU.

Usage:
    modal run event_jepa_cube/run_probe.py
"""
import modal

scale_app = modal.App("jcube-probe")
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
    memory=32768,
    timeout=3600,
)
def run_probes():
    import os
    import sys
    import time
    import torch
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyarrow.compute as pc

    # Force volume refresh to get latest checkpoints
    cache_volume.reload()
    data_volume.reload()

    GRAPH = "/data/jcube_graph.parquet"
    # Auto-detect latest version: V6.1 > V6 > V5
    if os.path.exists("/cache/tkg-v6.1/node_embeddings.pt"):
        WEIGHTS = "/cache/tkg-v6.1/node_embeddings.pt"
    elif os.path.exists("/cache/tkg-v6.1/node_emb_epoch_5.pt"):
        WEIGHTS = "/cache/tkg-v6.1/node_emb_epoch_5.pt"
    elif os.path.exists("/cache/tkg-v6/node_emb_epoch_3.pt"):
        WEIGHTS = "/cache/tkg-v6/node_emb_epoch_3.pt"
    else:
        WEIGHTS = "/cache/tkg-fullscale/node_embeddings.pt"
    DB = "/data/aggregated_fixed_union.db"

    # Verify file size
    wsize = os.path.getsize(WEIGHTS)
    print(f"Weights file: {WEIGHTS} ({wsize / 1e9:.1f} GB)")

    print("=" * 60)
    print("JCUBE DIGITAL TWIN — PROBE EVALUATION")
    print("=" * 60)

    # 1. Load node vocabulary (C++ fast path)
    print("\n[1] Loading node vocabulary...")
    t0 = time.time()
    table = pq.read_table(GRAPH, columns=["subject_id", "object_id"])
    all_nodes = pa.chunked_array(table.column("subject_id").chunks + table.column("object_id").chunks)
    unique_nodes = pc.unique(all_nodes)
    node_names = unique_nodes.to_numpy(zero_copy_only=False).astype(object)
    del table, all_nodes, unique_nodes
    print(f"  {len(node_names):,} nodes in {time.time()-t0:.1f}s")

    # 2. Load embeddings
    print("\n[2] Loading epoch 2 embeddings...")
    t1 = time.time()
    state = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "weight" in state:
        embeddings = state["weight"].float().numpy()
    elif isinstance(state, torch.Tensor):
        embeddings = state.float().numpy()
    else:
        embeddings = list(state.values())[0].float().numpy()
    print(f"  {embeddings.shape} in {time.time()-t1:.1f}s")

    assert len(node_names) == embeddings.shape[0], \
        f"Mismatch: {len(node_names)} names vs {embeddings.shape[0]} vectors"

    # L2-normalize for linear probes (embeddings have very small norms ~0.05)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-8)
    embeddings_normed = embeddings / norms
    print(f"  Norm range: [{norms.min():.4f}, {norms.max():.4f}], mean={norms.mean():.4f}")

    # 3. Build type masks
    print("\n[3] Building entity type index...")
    node_to_idx = {}
    entity_type_mask = {}
    for i, name in enumerate(node_names):
        sname = str(name)
        node_to_idx[sname] = i
        # Handle both V3 format "ID_CD_INTERNACAO_123" and V4 "GHO-BRADESCO/ID_CD_INTERNACAO_123"
        rest = sname.split("/", 1)[-1] if "/" in sname else sname
        parts = rest.split("_")
        if len(parts) >= 3 and parts[0] == "ID" and parts[1] == "CD":
            etype = parts[2]
            if etype not in entity_type_mask:
                entity_type_mask[etype] = []
            entity_type_mask[etype].append(i)

    # Convert to numpy masks
    for etype in entity_type_mask:
        idx_arr = np.array(entity_type_mask[etype])
        mask = np.zeros(len(node_names), dtype=bool)
        mask[idx_arr] = True
        entity_type_mask[etype] = mask

    print(f"  {len(entity_type_mask)} entity types")
    top_types = sorted(entity_type_mask.items(), key=lambda x: -x[1].sum())[:10]
    for etype, mask in top_types:
        print(f"    {etype}: {mask.sum():,}")

    # ================================================================
    # PROBE A: Type Coherence (unsupervised)
    # ================================================================
    print("\n" + "=" * 60)
    print("[PROBE A] TYPE COHERENCE")
    print("=" * 60)
    type_centroids = {}
    for etype, mask in entity_type_mask.items():
        vecs = embeddings[mask]
        if len(vecs) < 10:
            continue
        centroid = vecs.mean(axis=0)
        type_centroids[etype] = centroid
        intra_dist = np.linalg.norm(vecs - centroid, axis=1).mean()
        if mask.sum() > 1000:
            print(f"  {etype:30s}  n={mask.sum():>10,}  intra_dist={intra_dist:.4f}")

    if len(type_centroids) >= 2:
        centroid_vecs = np.array(list(type_centroids.values()))
        centroid_names = list(type_centroids.keys())
        from itertools import combinations
        inter_dists = []
        for i, j in combinations(range(len(centroid_vecs)), 2):
            inter_dists.append(np.linalg.norm(centroid_vecs[i] - centroid_vecs[j]))
        print(f"\n  Avg inter-type centroid distance: {np.mean(inter_dists):.4f}")

    # ================================================================
    # PROBE B: Anomaly Detection (unsupervised)
    # ================================================================
    print("\n" + "=" * 60)
    print("[PROBE B] ANOMALY DETECTION — INTERNACAO")
    print("=" * 60)
    if "INTERNACAO" in entity_type_mask:
        mask = entity_type_mask["INTERNACAO"]
        vecs = embeddings[mask]
        names = node_names[mask]
        centroid = vecs.mean(axis=0, keepdims=True)
        dists = np.linalg.norm(vecs - centroid, axis=1)
        z = (dists - dists.mean()) / max(dists.std(), 1e-8)
        top = np.argsort(-z)[:10]
        print(f"  {mask.sum():,} admissions scored")
        print(f"  Mean dist: {dists.mean():.4f}, Std: {dists.std():.4f}")
        for i in top:
            print(f"    {names[i]:45s}  z={z[i]:.2f}")

    # ================================================================
    # PROBE C: Glosa Risk (Linear Probe — LogisticRegression)
    # ================================================================
    print("\n" + "=" * 60)
    print("[PROBE C] GLOSA (BILLING DENIAL) — Linear Probe")
    print("=" * 60)
    import duckdb
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, classification_report, average_precision_score

    try:
        con = duckdb.connect(DB, read_only=True)
        glosa_q = """
            SELECT source_db,
                   CAST(ID_CD_INTERNACAO AS VARCHAR) AS eid,
                   CASE WHEN SUM(CASE WHEN FL_GLOSA = 'S' THEN 1 ELSE 0 END) > 0 THEN 1 ELSE 0 END AS has_glosa
            FROM agg_tb_fatura_fatu
            WHERE ID_CD_INTERNACAO IS NOT NULL AND source_db IS NOT NULL
            GROUP BY source_db, ID_CD_INTERNACAO
        """
        rows = con.execute(glosa_q).fetchall()
        con.close()

        X_list, y_list = [], []
        for source_db, eid, label in rows:
            for key in [f"{source_db}/ID_CD_INTERNACAO_{eid}", f"ID_CD_INTERNACAO_{eid}"]:
                if key in node_to_idx:
                    X_list.append(embeddings_normed[node_to_idx[key]])
                    y_list.append(float(label))
                    break

        if len(X_list) > 100:
            X = np.array(X_list)
            y = np.array(y_list)
            n_pos = int(y.sum())
            print(f"  Dataset: {len(y):,} admissions, {n_pos:,} with glosa ({100*n_pos/len(y):.1f}%)")

            perm = np.random.RandomState(42).permutation(len(y))
            split = int(len(y) * 0.8)
            X_train, X_test = X[perm[:split]], X[perm[split:]]
            y_train, y_test = y[perm[:split]], y[perm[split:]]

            clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=42, n_jobs=-1)
            clf.fit(X_train, y_train)
            probs = clf.predict_proba(X_test)[:, 1]
            preds = clf.predict(X_test)
            auc = roc_auc_score(y_test, probs)
            ap = average_precision_score(y_test, probs)

            print(f"\n  === TEST SET RESULTS ===")
            print(f"  ROC-AUC: {auc:.4f}")
            print(f"  Average Precision: {ap:.4f}")
            print(classification_report(y_test, preds, target_names=["No Glosa", "Glosa"]))

            # Top weight dimensions
            w = np.abs(clf.coef_[0])
            top_dims = np.argsort(-w)[:10]
            print(f"  Top 10 embedding dimensions (by |weight|):")
            for d in top_dims:
                print(f"    dim[{d:>3}]: |w|={w[d]:.4f}")
        else:
            print(f"  Only {len(X_list)} matched — not enough")
    except Exception as e:
        import traceback
        print(f"  SKIP: {e}")
        traceback.print_exc()

    # ================================================================
    # PROBE D: Length of Stay (Linear Probe — Ridge Regression)
    # ================================================================
    print("\n" + "=" * 60)
    print("[PROBE D] LENGTH OF STAY — Linear Probe (Ridge)")
    print("=" * 60)

    try:
        con = duckdb.connect(DB, read_only=True)
        los_q = """
            SELECT source_db,
                   CAST(ID_CD_INTERNACAO AS VARCHAR) AS eid,
                   DATEDIFF('day', DH_ADMISSAO_HOSP, DH_FINALIZACAO) AS los
            FROM agg_tb_capta_internacao_cain
            WHERE DH_ADMISSAO_HOSP IS NOT NULL AND DH_FINALIZACAO IS NOT NULL
              AND source_db IS NOT NULL
              AND DATEDIFF('day', DH_ADMISSAO_HOSP, DH_FINALIZACAO) BETWEEN 1 AND 365
        """
        rows = con.execute(los_q).fetchall()
        con.close()

        X_list, y_list = [], []
        for source_db, eid, los in rows:
            for key in [f"{source_db}/ID_CD_INTERNACAO_{eid}", f"ID_CD_INTERNACAO_{eid}"]:
                if key in node_to_idx:
                    X_list.append(embeddings_normed[node_to_idx[key]])
                    y_list.append(float(los))
                    break

        if len(X_list) > 100:
            X = np.array(X_list)
            y = np.array(y_list)
            print(f"  Dataset: {len(y):,} admissions")
            print(f"  LOS range: {y.min():.0f} - {y.max():.0f} days, median={np.median(y):.0f}, mean={y.mean():.1f}")

            perm = np.random.RandomState(42).permutation(len(y))
            split = int(len(y) * 0.8)
            X_train, X_test = X[perm[:split]], X[perm[split:]]
            y_train, y_test = y[perm[:split]], y[perm[split:]]

            # Log-transform target (LOS is right-skewed)
            from sklearn.linear_model import Ridge
            reg = Ridge(alpha=1.0)
            reg.fit(X_train, np.log1p(y_train))
            preds = np.expm1(reg.predict(X_test))
            actuals = y_test

            ss_res = np.sum((actuals - preds) ** 2)
            ss_tot = np.sum((actuals - actuals.mean()) ** 2)
            r2 = 1.0 - ss_res / max(ss_tot, 1e-8)
            mae = np.mean(np.abs(actuals - preds))
            mape = np.mean(np.abs(actuals - preds) / actuals.clip(min=1)) * 100
            median_ae = np.median(np.abs(actuals - preds))

            print(f"\n  === TEST SET RESULTS ===")
            print(f"  R²   = {r2:.4f}")
            print(f"  MAE  = {mae:.2f} days")
            print(f"  MedAE = {median_ae:.2f} days")
            print(f"  MAPE = {mape:.1f}%")

            for bucket_name, lo, hi in [("Short (1-3d)", 1, 3), ("Medium (4-14d)", 4, 14), ("Long (15-60d)", 15, 60), ("Extended (60+d)", 60, 365)]:
                mask_b = (actuals >= lo) & (actuals <= hi)
                if mask_b.sum() > 0:
                    bucket_mae = np.mean(np.abs(actuals[mask_b] - preds[mask_b]))
                    print(f"    {bucket_name:20s}: n={mask_b.sum():>6,}  MAE={bucket_mae:.1f} days")

            w = np.abs(reg.coef_)
            top_dims = np.argsort(-w)[:10]
            print(f"\n  Top 10 embedding dimensions (by |weight|):")
            for d in top_dims:
                print(f"    dim[{d:>3}]: |w|={w[d]:.4f}")
        else:
            print(f"  Only {len(X_list)} matched — not enough")
    except Exception as e:
        import traceback
        print(f"  SKIP: {e}")
        traceback.print_exc()

    # ================================================================
    # PROBE E: Semantic Search Demo
    # ================================================================
    print("\n" + "=" * 60)
    print("[PROBE E] SEMANTIC SEARCH DEMO")
    print("=" * 60)
    if "INTERNACAO" in entity_type_mask:
        mask = entity_type_mask["INTERNACAO"]
        adm_indices = np.where(mask)[0]
        if len(adm_indices) > 0:
            sample_idx = adm_indices[0]
            sample_name = str(node_names[sample_idx])
            query_vec = embeddings[sample_idx].reshape(1, -1)

            # Same-type search
            adm_vecs = embeddings[mask]
            adm_names = node_names[mask]
            nq = np.linalg.norm(query_vec).clip(min=1e-8)
            nc = np.linalg.norm(adm_vecs, axis=1).clip(min=1e-8)
            sims = (adm_vecs @ query_vec.T).squeeze() / (nc * nq)
            top = np.argsort(-sims)[1:6]

            print(f"  Query: {sample_name}")
            print(f"  Top 5 similar admissions:")
            for i in top:
                print(f"    {adm_names[i]:45s}  sim={sims[i]:.4f}")

            # Cross-type: find related PACIENTE
            if "PACIENTE" in entity_type_mask:
                pmask = entity_type_mask["PACIENTE"]
                pvecs = embeddings[pmask]
                pnames = node_names[pmask]
                np2 = np.linalg.norm(pvecs, axis=1).clip(min=1e-8)
                psims = (pvecs @ query_vec.T).squeeze() / (np2 * nq)
                ptop = np.argsort(-psims)[:3]
                print(f"\n  Closest patients to {sample_name}:")
                for i in ptop:
                    print(f"    {pnames[i]:45s}  sim={psims[i]:.4f}")

    # ================================================================
    # PROBE F: Isotropy Check
    # ================================================================
    print("\n" + "=" * 60)
    print("[PROBE F] EMBEDDING QUALITY — ISOTROPY")
    print("=" * 60)
    per_dim_var = np.var(embeddings, axis=0)
    isotropy = per_dim_var.min() / max(per_dim_var.max(), 1e-8)
    effective_dims = np.sum(per_dim_var > per_dim_var.max() * 0.01)
    print(f"  Isotropy (min/max variance ratio): {isotropy:.4f}")
    print(f"  Effective dimensions (>1% of max var): {effective_dims}/{embeddings.shape[1]}")
    print(f"  Per-dim variance range: [{per_dim_var.min():.6f}, {per_dim_var.max():.6f}]")

    print("\n" + "=" * 60)
    print("PROBES COMPLETE")
    print("=" * 60)


@scale_app.local_entrypoint()
def main():
    run_probes.remote()
# V4 probe - 1774247977
# epoch2-1774305118
# v6ep5 1774373537
