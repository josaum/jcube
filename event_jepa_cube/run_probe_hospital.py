"""Single-hospital probe — runs on filtered subgraph embeddings."""
import os
import modal

scale_app = modal.App("jcube-probe-hospital")
cache_volume = modal.Volume.from_name("jepa-cache", create_if_missing=True)
data_volume = modal.Volume.from_name("jcube-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch>=2.6", "numpy>=2.0", "pyarrow>=18.0", "duckdb>=1.0.0",
                 "scikit-learn>=1.4", "lightgbm>=4.0")
)

@scale_app.function(
    volumes={"/cache": cache_volume, "/data": data_volume},
    image=image, memory=16384, cpu=4, timeout=3600,
)
def run_hospital_probes(hospital: str = "GHO-BRADESCO"):
    import time, torch, numpy as np
    import pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
    import duckdb
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, classification_report, mean_absolute_error, r2_score

    cache_volume.reload()
    data_volume.reload()

    GRAPH = "/data/jcube_graph_v6.parquet"
    DB = "/data/aggregated_fixed_union.db"

    # Find the latest hospital checkpoint
    hospital_safe = hospital.replace("/", "_").replace(" ", "_")
    hospital_dir = f"/cache/tkg-v6-{hospital_safe}"
    weights_candidates = [
        f"{hospital_dir}/node_embeddings.pt",
        f"{hospital_dir}/node_emb_epoch_4.pt",
        f"{hospital_dir}/node_emb_epoch_3.pt",
    ]
    # Also check if BRADESCO wrote to tkg-v6 directly (the bug)
    for i in range(10, 0, -1):
        weights_candidates.append(f"/cache/tkg-v6/node_emb_epoch_{i}.pt")
    weights_candidates.append("/cache/tkg-v6/node_embeddings.pt")

    WEIGHTS = None
    for w in weights_candidates:
        if os.path.exists(w):
            size_gb = os.path.getsize(w) / 1e9
            # BRADESCO weights should be ~1.4 GB, not 16.8 GB
            if size_gb < 5.0:  # single-hospital
                WEIGHTS = w
                break

    if not WEIGHTS:
        print(f"ERROR: No single-hospital weights found for {hospital}")
        return

    print("=" * 60)
    print(f"JCUBE SINGLE-HOSPITAL PROBE: {hospital}")
    print(f"Weights: {WEIGHTS} ({os.path.getsize(WEIGHTS)/1e9:.1f} GB)")
    print("=" * 60)

    # 1. Load filtered graph vocab
    print("\n[1] Loading filtered graph vocabulary...")
    t0 = time.time()
    table = pq.read_table(GRAPH)
    mask_s = pc.match_substring(table.column("subject_id"), hospital)
    mask_o = pc.match_substring(table.column("object_id"), hospital)
    table = table.filter(pc.or_(mask_s, mask_o))
    print(f"  {table.num_rows:,} edges for {hospital}")

    all_nodes = pa.chunked_array(table.column("subject_id").chunks + table.column("object_id").chunks)
    unique_nodes = pc.unique(all_nodes)
    node_names = unique_nodes.to_numpy(zero_copy_only=False).astype(object)
    print(f"  {len(node_names):,} nodes in {time.time()-t0:.1f}s")

    # 2. Load embeddings
    print("\n[2] Loading embeddings...")
    state = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    if isinstance(state, torch.Tensor):
        embeddings = state.float().numpy()
    elif isinstance(state, dict) and "weight" in state:
        embeddings = state["weight"].float().numpy()
    else:
        embeddings = list(state.values())[0].float().numpy()
    print(f"  Shape: {embeddings.shape}")

    assert len(node_names) == embeddings.shape[0], \
        f"Mismatch: {len(node_names)} names vs {embeddings.shape[0]} vectors"

    dim = embeddings.shape[1]

    # 3. Entity type index
    print("\n[3] Building entity type index...")
    type_map = {}
    for i, name in enumerate(node_names):
        s = str(name)
        # Parse: GHO-BRADESCO/ID_CD_INTERNACAO_123 -> INTERNACAO
        if "/ID_CD_" in s:
            etype = s.split("/ID_CD_")[1].split("_")[0]
        elif "ID_CD_" in s:
            etype = s.split("ID_CD_")[1].split("_")[0]
        else:
            etype = "OTHER"
        if etype not in type_map:
            type_map[etype] = []
        type_map[etype].append(i)

    print(f"  {len(type_map)} entity types")
    for etype in sorted(type_map, key=lambda x: -len(type_map[x]))[:10]:
        print(f"    {etype}: {len(type_map[etype]):,}")

    # 4. Type coherence
    print(f"\n[PROBE A] TYPE COHERENCE — {hospital}")
    inter_centroids = []
    for etype, indices in sorted(type_map.items(), key=lambda x: -len(x[1]))[:20]:
        if len(indices) < 10:
            continue
        idx = np.array(indices[:10000])
        vecs = embeddings[idx]
        centroid = vecs.mean(axis=0, keepdims=True)
        dists = np.linalg.norm(vecs - centroid, axis=1)
        intra = dists.mean()
        inter_centroids.append(centroid[0])
        print(f"  {etype:25s} n={len(indices):>8,}  intra_dist={intra:.4f}")

    if len(inter_centroids) > 1:
        centroids = np.array(inter_centroids)
        from scipy.spatial.distance import pdist
        inter = pdist(centroids).mean()
        print(f"\n  Avg inter-type centroid distance: {inter:.4f}")

    # 5. Anomaly detection
    print(f"\n[PROBE B] ANOMALY DETECTION — INTERNACAO ({hospital})")
    if "INTERNACAO" in type_map:
        idx = np.array(type_map["INTERNACAO"])
        vecs = embeddings[idx]
        centroid = vecs.mean(axis=0, keepdims=True)
        dists = np.linalg.norm(vecs - centroid, axis=1)
        mean_d, std_d = dists.mean(), max(dists.std(), 1e-8)
        z_scores = (dists - mean_d) / std_d
        top10 = np.argsort(-z_scores)[:10]
        print(f"  {len(idx):,} admissions, mean_dist={mean_d:.4f}, std={std_d:.4f}")
        for rank, i in enumerate(top10):
            print(f"    {node_names[idx[i]]:50s} z={z_scores[i]:.2f}")

    # 6. Glosa prediction
    print(f"\n[PROBE C] GLOSA — LightGBM ({hospital})")
    con = duckdb.connect(DB, read_only=True)
    node_to_idx = {str(n): i for i, n in enumerate(node_names)}

    try:
        rows = con.execute(f"""
            SELECT CAST(ID_CD_INTERNACAO AS VARCHAR) AS iid,
                   MAX(CASE WHEN FL_GLOSA = 'S' THEN 1 ELSE 0 END) AS has_glosa
            FROM agg_tb_fatura_fatu
            WHERE source_db = '{hospital}' AND ID_CD_INTERNACAO IS NOT NULL
            GROUP BY ID_CD_INTERNACAO
        """).fetchall()

        X, y = [], []
        for iid, glosa in rows:
            key = f"{hospital}/ID_CD_INTERNACAO_{iid}"
            if key in node_to_idx:
                X.append(embeddings[node_to_idx[key]])
                y.append(glosa)

        X, y = np.array(X), np.array(y)
        print(f"  Dataset: {len(X):,} admissions, {y.sum():,} with glosa ({100*y.mean():.1f}%)")

        if len(X) > 100 and y.sum() > 10 and (len(y) - y.sum()) > 10:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
            clf = lgb.LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                      class_weight="balanced", verbose=-1, n_jobs=-1)
            clf.fit(X_tr, y_tr)
            probs = clf.predict_proba(X_te)[:, 1]
            auc = roc_auc_score(y_te, probs)
            print(f"  ROC-AUC: {auc:.4f}")
        else:
            print("  Not enough data for LightGBM")
    except Exception as e:
        print(f"  Error: {e}")

    # 7. LOS prediction
    print(f"\n[PROBE D] LOS — LightGBM ({hospital})")
    try:
        rows = con.execute(f"""
            SELECT CAST(ID_CD_INTERNACAO AS VARCHAR) AS iid,
                   DATEDIFF('day', DH_ADMISSAO_HOSP, COALESCE(DH_FINALIZACAO, CURRENT_TIMESTAMP)) AS los
            FROM agg_tb_capta_internacao_cain
            WHERE source_db = '{hospital}'
              AND DH_ADMISSAO_HOSP IS NOT NULL
              AND DATEDIFF('day', DH_ADMISSAO_HOSP, COALESCE(DH_FINALIZACAO, CURRENT_TIMESTAMP)) BETWEEN 1 AND 365
        """).fetchall()

        X, y = [], []
        for iid, los in rows:
            key = f"{hospital}/ID_CD_INTERNACAO_{iid}"
            if key in node_to_idx:
                X.append(embeddings[node_to_idx[key]])
                y.append(los)

        X, y = np.array(X), np.array(y)
        print(f"  Dataset: {len(X):,} admissions, median LOS={np.median(y):.0f}d")

        if len(X) > 100:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
            reg = lgb.LGBMRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                                     verbose=-1, n_jobs=-1)
            reg.fit(X_tr, np.log1p(y_tr))
            preds = np.expm1(reg.predict(X_te))
            mae = mean_absolute_error(y_te, preds)
            r2 = r2_score(y_te, preds)
            print(f"  R² = {r2:.4f}")
            print(f"  MAE = {mae:.1f} days")

            # Bucketed
            for label, lo, hi in [("Short 1-3d", 1, 3), ("Medium 4-14d", 4, 14),
                                   ("Long 15-60d", 15, 60), ("Extended 60+d", 60, 365)]:
                mask = (y_te >= lo) & (y_te <= hi)
                if mask.sum() > 0:
                    print(f"    {label:15s}: n={mask.sum():>5,}  MAE={mean_absolute_error(y_te[mask], preds[mask]):.1f}d")
        else:
            print("  Not enough data")
    except Exception as e:
        print(f"  Error: {e}")

    # 8. Isotropy
    print(f"\n[PROBE E] ISOTROPY ({hospital})")
    sample_idx = np.random.choice(len(embeddings), min(50000, len(embeddings)), replace=False)
    sample = embeddings[sample_idx]
    var_per_dim = sample.var(axis=0)
    iso = var_per_dim.min() / max(var_per_dim.max(), 1e-12)
    active = (var_per_dim > 0.01 * var_per_dim.max()).sum()
    print(f"  Isotropy: {iso:.4f}")
    print(f"  Active dims: {active}/{dim}")

    # 9. Semantic search
    print(f"\n[PROBE F] SEMANTIC SEARCH ({hospital})")
    if "INTERNACAO" in type_map and len(type_map["INTERNACAO"]) > 5:
        query_idx = type_map["INTERNACAO"][0]
        query_vec = embeddings[query_idx].reshape(1, -1)
        intern_idx = np.array(type_map["INTERNACAO"])
        intern_vecs = embeddings[intern_idx]
        nq = np.linalg.norm(query_vec).clip(min=1e-8)
        ni = np.linalg.norm(intern_vecs, axis=1).clip(min=1e-8)
        sims = (intern_vecs @ query_vec.T).squeeze() / (ni * nq)
        top5 = np.argsort(-sims)[1:6]
        print(f"  Query: {node_names[query_idx]}")
        for i in top5:
            print(f"    {node_names[intern_idx[i]]:50s} sim={sims[i]:.4f}")

    con.close()
    print(f"\n{'=' * 60}")
    print("PROBES COMPLETE")
    print(f"{'=' * 60}")


@scale_app.local_entrypoint()
def main():
    run_hospital_probes.remote(hospital="GHO-BRADESCO")
