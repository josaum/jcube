#!/usr/bin/env python3
"""Explore all entity types in the V6.2 temporal embedding space."""
from __future__ import annotations
import modal

app = modal.App("jcube-explore-entities")
jepa_cache = modal.Volume.from_name("jepa-cache", create_if_missing=False)
data_vol = modal.Volume.from_name("jcube-data", create_if_missing=False)
VOLUMES = {"/cache": jepa_cache, "/data": data_vol}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch>=2.2", "numpy>=1.26", "duckdb>=1.2.0", "pyarrow>=18.0", "scikit-learn>=1.4")
)

@app.function(image=image, volumes=VOLUMES, memory=65536, timeout=900)
def explore():
    import torch, numpy as np, duckdb, re
    from collections import Counter
    import pyarrow.parquet as pq, pyarrow.compute as pc, pyarrow as pa

    jepa_cache.reload()
    data_vol.reload()

    DB = "/data/aggregated_fixed_union.db"
    EMB = "/cache/tkg-v6.2/node_emb_epoch_3.pt"
    GRAPH = "/data/jcube_graph.parquet"
    SRC = "GHO-BRADESCO"

    # Load vocab
    print("Loading vocab...")
    table = pq.read_table(GRAPH, columns=["subject_id", "object_id"])
    subj, obj = table.column("subject_id"), table.column("object_id")
    unique_nodes = pc.unique(pa.chunked_array(subj.chunks + obj.chunks)).to_numpy(zero_copy_only=False).astype(object)
    node_to_idx = {str(n): i for i, n in enumerate(unique_nodes)}
    del table, subj, obj
    print(f"  {len(unique_nodes):,} nodes")

    # Load embeddings
    print("Loading embeddings...")
    state = torch.load(EMB, map_location="cpu", weights_only=True)
    if isinstance(state, torch.Tensor):
        emb = state.numpy().astype(np.float32)
    elif isinstance(state, dict) and "weight" in state:
        emb = state["weight"].numpy().astype(np.float32)
    else:
        emb = list(state.values())[0].numpy().astype(np.float32)
    print(f"  Embeddings: {emb.shape}")

    # ═══════════════════════════════════════════════
    # 1. ENTITY TYPE CENSUS
    # ═══════════════════════════════════════════════
    print("\n" + "="*70)
    print("ENTITY TYPE CENSUS")
    print("="*70)

    # Parse node names to extract entity types
    type_counter = Counter()
    brad_type_counter = Counter()
    for n in unique_nodes:
        s = str(n)
        # Extract the ID_CD_XXX pattern
        match = re.search(r'/(ID_CD_\w+?)_\d+', s)
        if match:
            etype = match.group(1)
            type_counter[etype] += 1
            if SRC in s:
                brad_type_counter[etype] += 1
        else:
            # Try other patterns
            if '/TUSS_' in s:
                type_counter['TUSS'] += 1
                if SRC in s: brad_type_counter['TUSS'] += 1
            elif '/CID_' in s:
                type_counter['CID_CODE'] += 1
                if SRC in s: brad_type_counter['CID_CODE'] += 1
            else:
                type_counter['OTHER'] += 1

    print(f"\n  ALL source_dbs ({len(type_counter)} entity types):")
    for etype, count in type_counter.most_common(30):
        brad = brad_type_counter.get(etype, 0)
        print(f"    {etype:40s} {count:>10,}  (BRADESCO: {brad:,})")

    # ═══════════════════════════════════════════════
    # 2. BRADESCO ENTITY EMBEDDINGS ANALYSIS
    # ═══════════════════════════════════════════════
    print("\n" + "="*70)
    print("BRADESCO ENTITY EMBEDDINGS")
    print("="*70)

    # Build masks for BRADESCO entity types
    entity_types = {
        "INTERNACAO": f"{SRC}/ID_CD_INTERNACAO_",
        "PACIENTE": f"{SRC}/ID_CD_PACIENTE_",
        "MEDICO_HOSPITAL": f"{SRC}/ID_CD_MEDICO_HOSPITAL_",
        "HOSPITAL": f"{SRC}/ID_CD_HOSPITAL_",
        "EVOLUCAO": f"{SRC}/ID_CD_EVOLUCAO_",
        "CONVENIO": f"{SRC}/ID_CD_CONVENIO_",
        "ORIGEM": f"{SRC}/ID_CD_ORIGEM_",
        "CID": f"{SRC}/ID_CD_CID_",
        "SUBCATEGORIA": f"{SRC}/ID_CD_SUBCATEGORIA_",
        "PRODUTO": f"{SRC}/ID_CD_PRODUTO_",
        "ESPECIALIDADE": f"{SRC}/ID_CD_ESPECIALIDADE_",
        "FINAL_MONITORAMENTO": f"{SRC}/ID_CD_FINAL_MONITORAMENTO_",
        "STATUS": f"{SRC}/ID_CD_STATUS_",
        "TIPO_EVENTO": f"{SRC}/ID_CD_TIPO_EVENTO_",
        "EVENTO_ADVERSO": f"{SRC}/ID_CD_EVENTO_ADVERSO_",
        "LOCALIZACAO": f"{SRC}/ID_CD_LOCALIZACAO_",
    }

    # Build death centroid first
    con = duckdb.connect(DB, read_only=True)
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
    obito_iids = {r[0] for r in obito_rows}

    # Also build "alta normal" centroid for algebra
    alta_rows = con.execute(f"""
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
        WHERE ls.rn = 1 AND UPPER(f.DS_FINAL_MONITORAMENTO) LIKE '%MELHORAD%'
    """).fetchall()
    alta_iids = {r[0] for r in alta_rows}
    print(f"\n  OBITO admissions: {len(obito_iids):,}")
    print(f"  ALTA NORMAL admissions: {len(alta_iids):,}")

    # Build centroids
    def _build_centroid(iids, prefix="ID_CD_INTERNACAO"):
        indices = [node_to_idx[f"{SRC}/{prefix}_{iid}"]
                   for iid in iids if f"{SRC}/{prefix}_{iid}" in node_to_idx]
        if len(indices) < 5:
            return None, 0
        vecs = emb[np.array(indices)]
        c = vecs.mean(axis=0)
        return c / (np.linalg.norm(c) + 1e-9), len(indices)

    death_centroid_n, n_death = _build_centroid(obito_iids)
    alta_centroid_n, n_alta = _build_centroid(alta_iids)

    # The "outcome vector": death - alta (direction of deterioration)
    outcome_vector = death_centroid_n - alta_centroid_n
    outcome_vector_n = outcome_vector / (np.linalg.norm(outcome_vector) + 1e-9)
    print(f"  Death centroid from {n_death} embeddings")
    print(f"  Alta centroid from {n_alta} embeddings")
    print(f"  Outcome vector norm: {np.linalg.norm(outcome_vector):.4f}")

    # ═══════════════════════════════════════════════
    # 3. ENTITY TYPE ANALYSIS
    # ═══════════════════════════════════════════════
    print("\n" + "="*70)
    print("ENTITY ANALYSIS vs DEATH/ALTA CENTROIDS")
    print("="*70)

    for etype, prefix in entity_types.items():
        mask = np.array([prefix in str(n) for n in unique_nodes], dtype=bool)
        n_ents = mask.sum()
        if n_ents < 3:
            continue

        vecs = emb[mask]
        norms = np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-9)
        vecs_n = vecs / norms

        # Cosine to death and alta centroids
        cos_death = vecs_n @ death_centroid_n
        cos_alta = vecs_n @ alta_centroid_n
        # Projection on outcome vector
        proj_outcome = vecs_n @ outcome_vector_n

        print(f"\n  {etype} ({n_ents:,} entities):")
        print(f"    cos_death:   mean={cos_death.mean():.4f} std={cos_death.std():.4f} "
              f"min={cos_death.min():.4f} max={cos_death.max():.4f}")
        print(f"    cos_alta:    mean={cos_alta.mean():.4f} std={cos_alta.std():.4f} "
              f"min={cos_alta.min():.4f} max={cos_alta.max():.4f}")
        print(f"    proj_outcome: mean={proj_outcome.mean():.4f} std={proj_outcome.std():.4f}")

    # ═══════════════════════════════════════════════
    # 4. EMBEDDING ALGEBRA: DOCTOR RISK PROFILES
    # ═══════════════════════════════════════════════
    print("\n" + "="*70)
    print("DOCTOR EMBEDDINGS: WHO IS CLOSEST TO DEATH?")
    print("="*70)

    # Get doctor info
    doc_rows = con.execute(f"""
        SELECT m.ID_CD_MEDICO_HOSPITAL, m.NM_MEDICO_HOSPITAL, m.DS_CONSELHO_CLASSE
        FROM agg_tb_capta_cfg_medico_hospital_ccmh m
        WHERE m.source_db = '{SRC}'
          AND m.DS_CONSELHO_CLASSE IS NOT NULL
          AND m.DS_CONSELHO_CLASSE NOT LIKE '0000%%'
    """).fetchall()
    doc_info = {r[0]: {"nome": r[1], "crm": r[2]} for r in doc_rows}

    doc_death_scores = []
    for did, info in doc_info.items():
        key = f"{SRC}/ID_CD_MEDICO_HOSPITAL_{did}"
        if key not in node_to_idx:
            continue
        vec = emb[node_to_idx[key]]
        vec_n = vec / (np.linalg.norm(vec) + 1e-9)
        cos_d = float(np.dot(vec_n, death_centroid_n))
        cos_a = float(np.dot(vec_n, alta_centroid_n))
        proj = float(np.dot(vec_n, outcome_vector_n))
        doc_death_scores.append({
            "id": did, "nome": info["nome"], "crm": info["crm"],
            "cos_death": cos_d, "cos_alta": cos_a, "proj_outcome": proj,
        })

    doc_death_scores.sort(key=lambda x: -x["cos_death"])
    print(f"\n  TOP 10 DOCTORS closest to DEATH centroid:")
    for d in doc_death_scores[:10]:
        print(f"    CRM={d['crm']:>12s} {d['nome']:30s} cos_death={d['cos_death']:.4f} cos_alta={d['cos_alta']:.4f} proj={d['proj_outcome']:+.4f}")

    print(f"\n  TOP 10 DOCTORS closest to ALTA centroid:")
    doc_death_scores.sort(key=lambda x: -x["cos_alta"])
    for d in doc_death_scores[:10]:
        print(f"    CRM={d['crm']:>12s} {d['nome']:30s} cos_death={d['cos_death']:.4f} cos_alta={d['cos_alta']:.4f} proj={d['proj_outcome']:+.4f}")

    # ═══════════════════════════════════════════════
    # 5. HOSPITAL EMBEDDINGS: RISK LANDSCAPE
    # ═══════════════════════════════════════════════
    print("\n" + "="*70)
    print("HOSPITAL EMBEDDINGS: RISK LANDSCAPE")
    print("="*70)

    hosp_rows = con.execute(f"""
        SELECT ID_CD_HOSPITAL, NM_HOSPITAL
        FROM agg_tb_capta_add_hospitais_caho
        WHERE source_db = '{SRC}'
    """).fetchall()
    hosp_info = {r[0]: r[1] for r in hosp_rows}

    hosp_scores = []
    for hid, nome in hosp_info.items():
        key = f"{SRC}/ID_CD_HOSPITAL_{hid}"
        if key not in node_to_idx:
            continue
        vec = emb[node_to_idx[key]]
        vec_n = vec / (np.linalg.norm(vec) + 1e-9)
        cos_d = float(np.dot(vec_n, death_centroid_n))
        cos_a = float(np.dot(vec_n, alta_centroid_n))
        proj = float(np.dot(vec_n, outcome_vector_n))
        hosp_scores.append({
            "id": hid, "nome": nome, "cos_death": cos_d, "cos_alta": cos_a, "proj_outcome": proj,
        })

    hosp_scores.sort(key=lambda x: -x["cos_death"])
    print(f"\n  HOSPITALS ranked by proximity to DEATH centroid:")
    for h in hosp_scores[:15]:
        print(f"    {h['nome']:45s} cos_death={h['cos_death']:.4f} cos_alta={h['cos_alta']:.4f} proj={h['proj_outcome']:+.4f}")

    # ═══════════════════════════════════════════════
    # 6. EMBEDDING ALGEBRA: CREATIVE OPERATIONS
    # ═══════════════════════════════════════════════
    print("\n" + "="*70)
    print("EMBEDDING ALGEBRA: CREATIVE OPERATIONS")
    print("="*70)

    # 6a: "Escalation vector" — Meropenem - Ceftriaxona
    # (what does moving from simple to heavy antibiotic look like in embedding space?)
    print("\n  6a. ANTIBIOTIC ESCALATION VECTOR (Meropenem - Ceftriaxona)")
    # Find product nodes for these medications
    med_nodes = {}
    for n in unique_nodes:
        s = str(n)
        if SRC in s and "/ID_CD_PRODUTO_" in s:
            med_nodes[s] = node_to_idx[s]

    # We need to find which product IDs correspond to Meropenem and Ceftriaxona
    med_rows = con.execute(f"""
        SELECT DISTINCT ID_CD_CFG_PRODUTO, NM_TITULO_COMERCIAL
        FROM agg_tb_capta_produtos_capr
        WHERE source_db = '{SRC}'
          AND (LOWER(NM_TITULO_COMERCIAL) LIKE '%meropenem%'
               OR LOWER(NM_TITULO_COMERCIAL) LIKE '%ceftriaxona%'
               OR LOWER(NM_TITULO_COMERCIAL) LIKE '%vancomicina%'
               OR LOWER(NM_TITULO_COMERCIAL) LIKE '%linezolida%')
    """).fetchall()
    print(f"    Medication config IDs: {med_rows}")

    # 6b: "Hospital difference vector" — Copa Dor vs Oswaldo Cruz
    print("\n  6b. HOSPITAL DIFFERENCE: Copa Dor vs Oswaldo Cruz")
    copa_key = f"{SRC}/ID_CD_HOSPITAL_7"  # Copa Dor
    osw_key = f"{SRC}/ID_CD_HOSPITAL_32917"  # Oswaldo Cruz
    if copa_key in node_to_idx and osw_key in node_to_idx:
        copa_vec = emb[node_to_idx[copa_key]]
        osw_vec = emb[node_to_idx[osw_key]]
        diff = copa_vec - osw_vec
        diff_n = diff / (np.linalg.norm(diff) + 1e-9)
        # Project diff onto outcome vector
        proj_diff = float(np.dot(diff_n, outcome_vector_n))
        cos_diff_death = float(np.dot(diff_n, death_centroid_n))
        print(f"    Copa Dor - Oswaldo Cruz:")
        print(f"      Projection on outcome (death-alta) axis: {proj_diff:+.4f}")
        print(f"      Copa Dor closer to death? {float(np.dot(copa_vec / (np.linalg.norm(copa_vec)+1e-9), death_centroid_n)):.4f} vs Oswaldo Cruz {float(np.dot(osw_vec / (np.linalg.norm(osw_vec)+1e-9), death_centroid_n)):.4f}")

    # 6c: "CID severity in embedding space"
    print("\n  6c. CID SEVERITY RANKING (by projection on outcome vector)")
    cid_mask = np.array([f"{SRC}/ID_CD_CID_" in str(n) for n in unique_nodes], dtype=bool)
    if cid_mask.sum() > 0:
        cid_indices = np.where(cid_mask)[0]
        cid_vecs = emb[cid_indices]
        cid_norms = np.linalg.norm(cid_vecs, axis=1, keepdims=True).clip(min=1e-9)
        cid_projs = (cid_vecs / cid_norms) @ outcome_vector_n

        # Get CID descriptions
        cid_desc_map = {}
        cid_rows = con.execute(f"""
            SELECT DISTINCT ID_CD_CID, DS_DESCRICAO
            FROM agg_tb_capta_cid_caci
            WHERE source_db = '{SRC}' AND DS_DESCRICAO IS NOT NULL
        """).fetchall()
        for r in cid_rows:
            cid_desc_map[r[0]] = r[1]

        cid_scores = []
        for i, idx in enumerate(cid_indices):
            node_name = str(unique_nodes[idx])
            try:
                cid_id = int(node_name.split("ID_CD_CID_")[1])
                desc = cid_desc_map.get(cid_id, f"CID_{cid_id}")
            except:
                desc = node_name[-30:]
            cid_scores.append((desc, float(cid_projs[i])))

        cid_scores.sort(key=lambda x: -x[1])
        print(f"    TOP 15 CIDs most aligned with DEATH direction:")
        for desc, proj in cid_scores[:15]:
            print(f"      {proj:+.4f}  {desc[:60]}")

        print(f"\n    TOP 15 CIDs most aligned with RECOVERY direction:")
        cid_scores.sort(key=lambda x: x[1])
        for desc, proj in cid_scores[:15]:
            print(f"      {proj:+.4f}  {desc[:60]}")

    # 6d: "Patient evolution topology"
    print("\n  6d. PATIENT EMBEDDING EVOLUTION (top 5 most-changed patients)")
    pat_mask = np.array([f"{SRC}/ID_CD_PACIENTE_" in str(n) for n in unique_nodes], dtype=bool)
    if pat_mask.sum() > 0:
        pat_indices = np.where(pat_mask)[0]
        pat_vecs = emb[pat_indices]
        # For each patient, find their admissions and check if the PATIENT embedding
        # is close to their latest admission embedding
        print(f"    {pat_mask.sum():,} patient entities in embedding space")
        # Sample: get 5 patients with multiple admissions
        multi_adm = con.execute(f"""
            SELECT ID_CD_PACIENTE, COUNT(DISTINCT ID_CD_INTERNACAO) as n_adm,
                   ARRAY_AGG(ID_CD_INTERNACAO ORDER BY DH_ADMISSAO_HOSP) as iids
            FROM agg_tb_capta_internacao_cain
            WHERE source_db = '{SRC}' AND DH_ADMISSAO_HOSP IS NOT NULL
            GROUP BY ID_CD_PACIENTE
            HAVING COUNT(DISTINCT ID_CD_INTERNACAO) >= 3
            ORDER BY COUNT(DISTINCT ID_CD_INTERNACAO) DESC
            LIMIT 10
        """).fetchall()
        for pid, n_adm, iids in multi_adm[:5]:
            pat_key = f"{SRC}/ID_CD_PACIENTE_{pid}"
            if pat_key not in node_to_idx:
                continue
            pat_vec = emb[node_to_idx[pat_key]]
            pat_vec_n = pat_vec / (np.linalg.norm(pat_vec) + 1e-9)
            # Get embeddings for each admission
            adm_vecs = []
            for aiid in iids:
                akey = f"{SRC}/ID_CD_INTERNACAO_{aiid}"
                if akey in node_to_idx:
                    adm_vecs.append((aiid, emb[node_to_idx[akey]]))
            if len(adm_vecs) < 2:
                continue
            # Patient embedding distance to each admission
            print(f"\n    Patient {pid} ({n_adm} admissions):")
            print(f"      Patient embedding cos_death={float(np.dot(pat_vec_n, death_centroid_n)):.4f} "
                  f"cos_alta={float(np.dot(pat_vec_n, alta_centroid_n)):.4f}")
            for aiid, avec in adm_vecs:
                avec_n = avec / (np.linalg.norm(avec) + 1e-9)
                cos_pat = float(np.dot(pat_vec_n, avec_n))
                cos_d = float(np.dot(avec_n, death_centroid_n))
                print(f"      Adm {aiid}: cos_to_patient={cos_pat:.4f} cos_death={cos_d:.4f}")

    # 6e: "Specialty embeddings"
    print("\n\n  6e. SPECIALTY EMBEDDINGS (ID_CD_ESPECIALIDADE)")
    spec_rows = con.execute(f"""
        SELECT DISTINCT m.ID_CD_ESPECIALIDADE, mh.DS_ESPECIALIDADE_DRG
        FROM agg_tb_capta_internacao_medico_hospital_cimh mh
        JOIN agg_tb_capta_cfg_medico_hospital_ccmh m
            ON mh.ID_CD_MEDICO_HOSPITAL = m.ID_CD_MEDICO_HOSPITAL AND mh.source_db = m.source_db
        WHERE mh.source_db = '{SRC}' AND mh.DS_ESPECIALIDADE_DRG IS NOT NULL
    """).fetchall()
    spec_map = {r[0]: r[1] for r in spec_rows}

    spec_scores = []
    for sid, sname in spec_map.items():
        key = f"{SRC}/ID_CD_ESPECIALIDADE_{sid}"
        if key not in node_to_idx:
            continue
        vec = emb[node_to_idx[key]]
        vec_n = vec / (np.linalg.norm(vec) + 1e-9)
        cos_d = float(np.dot(vec_n, death_centroid_n))
        cos_a = float(np.dot(vec_n, alta_centroid_n))
        proj = float(np.dot(vec_n, outcome_vector_n))
        spec_scores.append((sname, cos_d, cos_a, proj))

    spec_scores.sort(key=lambda x: -x[3])
    print(f"    Specialties ranked by projection on outcome vector (death→alta):")
    for name, cos_d, cos_a, proj in spec_scores:
        print(f"      {proj:+.4f}  cos_death={cos_d:.4f} cos_alta={cos_a:.4f}  {name}")

    con.close()
    print("\n\nDone.")


@app.local_entrypoint()
def main():
    explore.remote()
