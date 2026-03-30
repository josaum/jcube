#!/usr/bin/env python3
"""Analyze specific admissions in depth — runs on Modal with access to embeddings."""
from __future__ import annotations
import modal

app = modal.App("jcube-analyze-admission")
jepa_cache = modal.Volume.from_name("jepa-cache", create_if_missing=False)
data_vol = modal.Volume.from_name("jcube-data", create_if_missing=False)
VOLUMES = {"/cache": jepa_cache, "/data": data_vol}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch>=2.2", "numpy>=1.26", "duckdb>=1.2.0", "pyarrow>=18.0")
)

@app.function(image=image, volumes=VOLUMES, memory=65536, timeout=600)
def analyze(iid: int = 121120, pid_target: int = 79706):
    import torch, numpy as np, duckdb
    import pyarrow.parquet as pq, pyarrow.compute as pc, pyarrow as pa

    jepa_cache.reload()
    data_vol.reload()

    DB = "/data/aggregated_fixed_union.db"
    EMB = "/cache/tkg-v6.2/node_emb_epoch_3.pt"
    GRAPH = "/data/jcube_graph.parquet"
    SRC = "GHO-BRADESCO"

    # Load vocab + embeddings
    print("Loading vocab...")
    table = pq.read_table(GRAPH, columns=["subject_id", "object_id"])
    subj, obj = table.column("subject_id"), table.column("object_id")
    unique_nodes = pc.unique(pa.chunked_array(subj.chunks + obj.chunks)).to_numpy(zero_copy_only=False).astype(object)
    node_to_idx = {str(n): i for i, n in enumerate(unique_nodes)}
    del table, subj, obj

    print("Loading embeddings...")
    state = torch.load(EMB, map_location="cpu", weights_only=True)
    if isinstance(state, torch.Tensor):
        emb = state.numpy().astype(np.float32)
    elif isinstance(state, dict) and "weight" in state:
        emb = state["weight"].numpy().astype(np.float32)
    else:
        emb = list(state.values())[0].numpy().astype(np.float32)
    print(f"Embeddings: {emb.shape}")

    con = duckdb.connect(DB, read_only=True)

    # Build death centroid
    obito_rows = con.execute(f"""
        WITH last_st AS (
            SELECT es.ID_CD_INTERNACAO,
                   ROW_NUMBER() OVER (PARTITION BY es.ID_CD_INTERNACAO ORDER BY es.DH_CADASTRO DESC) AS rn,
                   es.FL_DESOSPITALIZACAO
            FROM agg_tb_capta_evo_status_caes es WHERE es.source_db = '{SRC}'
        )
        SELECT DISTINCT ls.ID_CD_INTERNACAO
        FROM last_st ls
        JOIN agg_tb_capta_tipo_final_monit_fmon f
            ON ls.FL_DESOSPITALIZACAO = f.ID_CD_FINAL_MONITORAMENTO AND f.source_db = '{SRC}'
        WHERE ls.rn = 1 AND UPPER(f.DS_FINAL_MONITORAMENTO) LIKE '%BITO%'
    """).fetchall()
    obito_iids = [r[0] for r in obito_rows]
    obito_indices = [node_to_idx[f"{SRC}/ID_CD_INTERNACAO_{oid}"] for oid in obito_iids if f"{SRC}/ID_CD_INTERNACAO_{oid}" in node_to_idx]
    obito_vecs = emb[np.array(obito_indices)]
    death_centroid = obito_vecs.mean(axis=0)
    death_centroid_norm = death_centroid / (np.linalg.norm(death_centroid) + 1e-9)
    print(f"Death centroid from {len(obito_indices)} OBITO embeddings")

    # Build BRADESCO mask for context
    brad_mask = np.array([f"{SRC}/ID_CD_INTERNACAO_" in str(n) for n in unique_nodes], dtype=bool)
    brad_indices = np.where(brad_mask)[0]
    brad_vecs = emb[brad_indices]
    brad_norms = np.linalg.norm(brad_vecs, axis=1, keepdims=True).clip(min=1e-9)
    brad_vecs_n = brad_vecs / brad_norms
    all_death_cos = brad_vecs_n @ death_centroid_norm

    # ═══════════════════════════════════════════════
    # ANALYSIS 1: Admission 121120
    # ═══════════════════════════════════════════════
    print("\n" + "="*70)
    print(f"ANALYSIS: ADMISSION {iid}")
    print("="*70)

    node_key = f"{SRC}/ID_CD_INTERNACAO_{iid}"
    if node_key not in node_to_idx:
        print(f"  NOT FOUND in embedding space!")
    else:
        idx = node_to_idx[node_key]
        vec = emb[idx]
        vec_norm = vec / (np.linalg.norm(vec) + 1e-9)
        cos_death = float(np.dot(vec_norm, death_centroid_norm))
        pct = 100.0 * (all_death_cos < cos_death).sum() / len(all_death_cos)

        print(f"  Death proximity (cosine): {cos_death:.4f}")
        print(f"  Percentile among all BRADESCO: P{pct:.1f}")
        print(f"  Context: mean={all_death_cos.mean():.4f}, std={all_death_cos.std():.4f}, "
              f"P50={np.median(all_death_cos):.4f}, P90={np.percentile(all_death_cos, 90):.4f}, "
              f"P99={np.percentile(all_death_cos, 99):.4f}, max={all_death_cos.max():.4f}")

        # 10 nearest neighbors
        sims = brad_vecs_n @ vec_norm
        top_k = np.argsort(-sims)[:11]
        print(f"\n  10 NEAREST NEIGHBORS:")
        for j in top_k:
            name = str(unique_nodes[brad_indices[j]])
            sim = sims[j]
            try:
                niid = int(name.split("ID_CD_INTERNACAO_")[1])
            except:
                continue
            if niid == iid:
                continue
            # Get details
            det = con.execute(f"""
                SELECT i.DH_ADMISSAO_HOSP::DATE, i.DH_FINALIZACAO::DATE, i.IN_SITUACAO,
                       h.NM_HOSPITAL,
                       DATEDIFF('day', i.DH_ADMISSAO_HOSP::DATE, COALESCE(i.DH_FINALIZACAO, CURRENT_DATE)::DATE) as los,
                       (SELECT STRING_AGG(DISTINCT DS_DESCRICAO, ' | ')
                        FROM agg_tb_capta_cid_caci WHERE source_db = '{SRC}' AND ID_CD_INTERNACAO = {niid}
                        AND DS_DESCRICAO IS NOT NULL) as cids
                FROM agg_tb_capta_internacao_cain i
                LEFT JOIN agg_tb_capta_add_hospitais_caho h
                    ON i.ID_CD_HOSPITAL = h.ID_CD_HOSPITAL AND i.source_db = h.source_db
                WHERE i.source_db = '{SRC}' AND i.ID_CD_INTERNACAO = {niid}
            """).fetchone()
            if det:
                status = "ABERTO" if det[2] != 2 else "ALTA"
                print(f"    IID={niid} cos={sim:.4f} LOS={det[4]}d {status} Hosp={det[3]} CID={str(det[5])[:70]}")

        # 5 most similar OBITO cases
        obito_vecs_n = obito_vecs / np.linalg.norm(obito_vecs, axis=1, keepdims=True).clip(min=1e-9)
        obito_sims = obito_vecs_n @ vec_norm
        top_obito = np.argsort(-obito_sims)[:5]
        print(f"\n  5 MOST SIMILAR OBITO ADMISSIONS:")
        for j in top_obito:
            oid = obito_iids[j]
            sim = obito_sims[j]
            det = con.execute(f"""
                SELECT h.NM_HOSPITAL,
                       DATEDIFF('day', i.DH_ADMISSAO_HOSP::DATE, i.DH_FINALIZACAO::DATE) as los,
                       (SELECT STRING_AGG(DISTINCT DS_DESCRICAO, ' | ')
                        FROM agg_tb_capta_cid_caci WHERE source_db = '{SRC}' AND ID_CD_INTERNACAO = {oid}
                        AND DS_DESCRICAO IS NOT NULL) as cids
                FROM agg_tb_capta_internacao_cain i
                LEFT JOIN agg_tb_capta_add_hospitais_caho h
                    ON i.ID_CD_HOSPITAL = h.ID_CD_HOSPITAL AND i.source_db = h.source_db
                WHERE i.source_db = '{SRC}' AND i.ID_CD_INTERNACAO = {oid}
            """).fetchone()
            if det:
                print(f"    IID={oid} cos={sim:.4f} LOS={det[1]}d Hosp={det[0]} CID={str(det[2])[:70]}")

    # ═══════════════════════════════════════════════
    # ANALYSIS 2: Patient 79706 trajectory
    # ═══════════════════════════════════════════════
    print("\n" + "="*70)
    print(f"ANALYSIS: PATIENT {pid_target} (trajectory)")
    print("="*70)

    # Get all admissions for this patient
    adm_rows = con.execute(f"""
        SELECT i.ID_CD_INTERNACAO, i.DH_ADMISSAO_HOSP::DATE, i.DH_FINALIZACAO::DATE,
               i.IN_SITUACAO, h.NM_HOSPITAL,
               DATEDIFF('day', i.DH_ADMISSAO_HOSP::DATE, COALESCE(i.DH_FINALIZACAO, CURRENT_DATE)::DATE) as los,
               (SELECT STRING_AGG(DISTINCT DS_DESCRICAO, ' | ')
                FROM agg_tb_capta_cid_caci WHERE source_db = '{SRC}' AND ID_CD_INTERNACAO = i.ID_CD_INTERNACAO
                AND DS_DESCRICAO IS NOT NULL) as cids
        FROM agg_tb_capta_internacao_cain i
        LEFT JOIN agg_tb_capta_add_hospitais_caho h
            ON i.ID_CD_HOSPITAL = h.ID_CD_HOSPITAL AND i.source_db = h.source_db
        WHERE i.source_db = '{SRC}' AND i.ID_CD_PACIENTE = {pid_target}
        ORDER BY i.DH_ADMISSAO_HOSP
    """).fetchall()

    print(f"  {len(adm_rows)} admissions for patient {pid_target}:")
    emb_list = []
    for row in adm_rows:
        aiid, adm, alta, sit, hosp, los, cids = row
        status = "ABERTO" if sit != 2 else "ALTA"
        key = f"{SRC}/ID_CD_INTERNACAO_{aiid}"
        has_emb = key in node_to_idx
        cos_d = "---"
        if has_emb:
            v = emb[node_to_idx[key]]
            vn = v / (np.linalg.norm(v) + 1e-9)
            cos_d = f"{float(np.dot(vn, death_centroid_norm)):.4f}"
            emb_list.append((aiid, v))
        print(f"    IID={aiid} Adm={adm} Alta={alta} {status} LOS={los}d Hosp={hosp}")
        print(f"      CID={str(cids)[:80]}")
        print(f"      Death proximity: {cos_d} | Embedding: {'YES' if has_emb else 'NO'}")

    # Trajectory analysis
    if len(emb_list) >= 2:
        print(f"\n  TRAJECTORY ANALYSIS ({len(emb_list)} embeddings):")
        for i in range(1, len(emb_list)):
            prev_iid, prev_vec = emb_list[i-1]
            curr_iid, curr_vec = emb_list[i]
            delta = curr_vec - prev_vec
            velocity = float(np.linalg.norm(delta))
            delta_norm = delta / (velocity + 1e-9)
            direction = float(np.dot(delta_norm, death_centroid_norm))
            risk = velocity * max(direction, 0)
            print(f"    {prev_iid} → {curr_iid}: velocity={velocity:.4f}, direction_to_death={direction:+.4f}, risk_score={risk:.4f}")

    con.close()
    print("\nDone.")


@app.local_entrypoint()
def main():
    analyze.remote(iid=121120, pid_target=79706)
