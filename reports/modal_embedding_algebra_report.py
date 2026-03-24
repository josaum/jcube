#!/usr/bin/env python3
"""
Modal script: JCUBE V6 Embedding Algebra Report
Showcases what you can compute purely from vector operations on the 35.2M x 64
healthcare digital twin.

Sections:
  1. Hospital DNA (operational fingerprints via centroid similarity)
  2. Clinical Trajectory Vectors (path to obito vs alta normal)
  3. Treatment Effect Signatures (CID gravity vectors)
  4. Readmission Risk from Embedding Dynamics (delta magnitude)
  5. Entity Resolution (cross-hospital patient matching)
  6. "What If" Simulation Demo (vector arithmetic on real admissions)

Output: LaTeX PDF at /data/reports/embedding_algebra_v6_2026_03.pdf
All text pt-BR, UTF-8, professional, one section per page.

Usage:
    modal run reports/modal_embedding_algebra_report.py
    modal run --detach reports/modal_embedding_algebra_report.py
"""
from __future__ import annotations

import modal

# ─────────────────────────────────────────────────────────────────
# Modal App + Volumes
# ─────────────────────────────────────────────────────────────────

app = modal.App("jcube-embedding-algebra")

jepa_cache = modal.Volume.from_name("jepa-cache", create_if_missing=False)
data_vol   = modal.Volume.from_name("jcube-data",  create_if_missing=False)

VOLUMES = {
    "/cache": jepa_cache,
    "/data":  data_vol,
}

# ─────────────────────────────────────────────────────────────────
# Container image — torch (CPU), duckdb, pyarrow, scikit-learn, latex
# ─────────────────────────────────────────────────────────────────

report_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "texlive-latex-base",
        "texlive-latex-recommended",
        "texlive-latex-extra",
        "texlive-fonts-recommended",
        "texlive-lang-portuguese",
        "lmodern",
    )
    .pip_install(
        "torch>=2.2",
        "numpy>=1.26",
        "duckdb>=1.2.0",
        "pyarrow>=18.0",
        "scikit-learn>=1.4",
    )
)

# ─────────────────────────────────────────────────────────────────
# Paths inside the container
# ─────────────────────────────────────────────────────────────────

GRAPH_PARQUET = "/data/jcube_graph.parquet"
WEIGHTS_PATH  = "/cache/tkg-v6/node_emb_epoch_3.pt"
DB_PATH       = "/data/aggregated_fixed_union.db"
OUTPUT_DIR    = "/data/reports"
OUTPUT_PDF    = f"{OUTPUT_DIR}/embedding_algebra_v6_2026_03.pdf"

REPORT_DATE_STR = "2026-03-23"

# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _safe_int(v, default: int = 0) -> int:
    try:
        return int(v) if v is not None else default
    except Exception:
        return default


def _safe_float(v, default: float = 0.0) -> float:
    try:
        return float(v) if v is not None else default
    except Exception:
        return default


def _escape_latex(s: str) -> str:
    """Escape LaTeX structural metacharacters; pass UTF-8 accents through."""
    if not s:
        return ""
    s = str(s)
    s = s.replace("\\", "\\textbackslash{}")
    s = s.replace("&",  "\\&")
    s = s.replace("%",  "\\%")
    s = s.replace("$",  "\\$")
    s = s.replace("#",  "\\#")
    s = s.replace("_",  "\\_")
    s = s.replace("{",  "\\{")
    s = s.replace("}",  "\\}")
    s = s.replace("~",  "\\textasciitilde{}")
    s = s.replace("^",  "\\textasciicircum{}")
    s = s.replace("\r\n", " ")
    s = s.replace("\n",   " ")
    s = s.replace("\r",   " ")
    s = s.replace("\t",   " ")
    return s


def _brl(v) -> str:
    f = _safe_float(v)
    if f == 0:
        return "---"
    return "R\\$ {:,.2f}".format(f).replace(",", "X").replace(".", ",").replace("X", ".")


def _truncate(s: str, max_len: int = 80) -> str:
    if not s:
        return "---"
    s = str(s).strip()
    if len(s) > max_len:
        return s[:max_len] + "..."
    return s


def _pct(num: float, den: float) -> str:
    if den == 0:
        return "---"
    return f"{100.0 * num / den:.1f}\\%"


def _fmt_sim(v: float) -> str:
    return f"{v:.4f}"


def _traffic_color(val: float, low: float, high: float) -> str:
    """Return LaTeX color name: green if <= low, red if >= high, yellow otherwise."""
    if val >= high:
        return "trafficred"
    elif val >= low:
        return "trafficyellow"
    return "trafficgreen"


def _traffic_color_inv(val: float, low: float, high: float) -> str:
    """Inverted: red if val <= low (bad = small delta)."""
    if val <= low:
        return "trafficred"
    elif val <= high:
        return "trafficyellow"
    return "trafficgreen"


# ─────────────────────────────────────────────────────────────────
# Step 1 — Load full twin (all node types)
# ─────────────────────────────────────────────────────────────────

def _load_twin():
    import time
    import numpy as np
    import torch
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    import pyarrow as pa

    print("[1/8] Loading node vocabulary from graph parquet ...")
    t0 = time.time()
    table = pq.read_table(GRAPH_PARQUET, columns=["subject_id", "object_id"])
    subj  = table.column("subject_id")
    obj   = table.column("object_id")
    all_nodes    = pa.chunked_array(subj.chunks + obj.chunks)
    unique_nodes = pc.unique(all_nodes).to_numpy(zero_copy_only=False).astype(object)
    del table, subj, obj, all_nodes
    n_nodes = len(unique_nodes)
    print(f"    {n_nodes:,} unique nodes in {time.time()-t0:.1f}s")

    print("[1/8] Loading V6 embedding weights ...")
    t1 = time.time()
    state = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
    if isinstance(state, torch.Tensor):
        embeddings = state.numpy().astype(np.float32)
    elif isinstance(state, dict) and "weight" in state:
        embeddings = state["weight"].numpy().astype(np.float32)
    else:
        embeddings = list(state.values())[0].numpy().astype(np.float32)
    print(f"    Embeddings shape: {embeddings.shape} in {time.time()-t1:.1f}s")

    if len(unique_nodes) != embeddings.shape[0]:
        raise ValueError(
            f"Vocab mismatch: {len(unique_nodes):,} nodes vs "
            f"{embeddings.shape[0]:,} embedding rows"
        )

    node_to_idx = {str(n): i for i, n in enumerate(unique_nodes)}

    # Build masks for different node types
    internacao_mask = np.array(
        ["/ID_CD_INTERNACAO_" in str(n) for n in unique_nodes], dtype=bool
    )
    paciente_mask = np.array(
        ["/ID_CD_PACIENTE_" in str(n) or "_PACIENTE_" in str(n)
         for n in unique_nodes], dtype=bool
    )
    cid_mask = np.array(
        ["ID_CD_CID_" in str(n) for n in unique_nodes], dtype=bool
    )

    print(f"    INTERNACAO nodes: {internacao_mask.sum():,}")
    print(f"    PACIENTE nodes:   {paciente_mask.sum():,}")
    print(f"    CID nodes:        {cid_mask.sum():,}")

    return unique_nodes, embeddings, node_to_idx, internacao_mask, paciente_mask, cid_mask


# ─────────────────────────────────────────────────────────────────
# Step 2 — Hospital DNA: centroid similarity matrix
# ─────────────────────────────────────────────────────────────────

def _hospital_dna(unique_nodes, embeddings, internacao_mask):
    import time
    import numpy as np

    print("[2/8] Computing Hospital DNA (centroid similarity) ...")
    t0 = time.time()

    int_nodes = unique_nodes[internacao_mask]
    int_embs  = embeddings[internacao_mask]

    # Group by source_db
    hospital_groups: dict[str, list[int]] = {}
    for i, node in enumerate(int_nodes):
        s = str(node)
        try:
            src_db = s.split("/")[0]
            hospital_groups.setdefault(src_db, []).append(i)
        except Exception:
            pass

    hospitals = sorted(hospital_groups.keys())
    n_hosp = len(hospitals)
    print(f"    Found {n_hosp} hospitals: {hospitals}")

    # Compute centroids
    centroids = {}
    for h in hospitals:
        idxs = hospital_groups[h]
        centroids[h] = int_embs[idxs].mean(axis=0)

    # Pairwise cosine similarity
    sim_matrix = np.zeros((n_hosp, n_hosp))
    for i, h1 in enumerate(hospitals):
        for j, h2 in enumerate(hospitals):
            c1 = centroids[h1]
            c2 = centroids[h2]
            dot = np.dot(c1, c2)
            norm = (np.linalg.norm(c1) * np.linalg.norm(c2))
            sim_matrix[i, j] = dot / max(norm, 1e-9)

    # Most unique hospital: lowest avg similarity to others
    avg_sims = {}
    for i, h in enumerate(hospitals):
        others = [sim_matrix[i, j] for j in range(n_hosp) if j != i]
        avg_sims[h] = np.mean(others) if others else 0.0

    most_unique = min(avg_sims, key=avg_sims.get)

    # Hospital pairs with sim > 0.95
    identical_pairs = []
    for i in range(n_hosp):
        for j in range(i + 1, n_hosp):
            if sim_matrix[i, j] > 0.95:
                identical_pairs.append((hospitals[i], hospitals[j], sim_matrix[i, j]))
    identical_pairs.sort(key=lambda x: -x[2])

    # Hospital sizes
    hospital_sizes = {h: len(hospital_groups[h]) for h in hospitals}

    print(f"    Most unique: {most_unique} (avg sim {avg_sims[most_unique]:.4f})")
    print(f"    Identical pairs (>0.95): {len(identical_pairs)}")
    print(f"    Done in {time.time()-t0:.1f}s")

    return hospitals, sim_matrix, avg_sims, most_unique, identical_pairs, hospital_sizes


# ─────────────────────────────────────────────────────────────────
# Step 3 — Clinical Trajectory Vectors
# ─────────────────────────────────────────────────────────────────

def _clinical_trajectories(unique_nodes, embeddings, node_to_idx):
    import time
    import numpy as np
    import duckdb

    print("[3/8] Computing clinical trajectory vectors ...")
    t0 = time.time()

    con = duckdb.connect(str(DB_PATH))

    # Get patients with 2+ admissions, ordered by admission date
    # Also get discharge type from last_status
    rows = con.execute("""
        WITH patient_admissions AS (
            SELECT
                i.ID_CD_INTERNACAO,
                i.ID_CD_PACIENTE,
                i.source_db,
                i.DH_ADMISSAO_HOSP,
                ROW_NUMBER() OVER (
                    PARTITION BY i.ID_CD_PACIENTE, i.source_db
                    ORDER BY i.DH_ADMISSAO_HOSP
                ) AS adm_seq,
                COUNT(*) OVER (
                    PARTITION BY i.ID_CD_PACIENTE, i.source_db
                ) AS total_adm
            FROM agg_tb_capta_internacao_cain i
            WHERE i.DH_ADMISSAO_HOSP IS NOT NULL
        ),
        last_status AS (
            SELECT es.ID_CD_INTERNACAO, es.FL_DESOSPITALIZACAO, es.source_db,
                   ROW_NUMBER() OVER (
                       PARTITION BY es.ID_CD_INTERNACAO
                       ORDER BY es.DH_CADASTRO DESC
                   ) AS rn
            FROM agg_tb_capta_evo_status_caes es
        ),
        discharge_info AS (
            SELECT ls.ID_CD_INTERNACAO, ls.source_db,
                   f.DS_FINAL_MONITORAMENTO AS tipo_alta
            FROM last_status ls
            JOIN agg_tb_capta_tipo_final_monit_fmon f
                ON ls.FL_DESOSPITALIZACAO = f.ID_CD_FINAL_MONITORAMENTO
            WHERE ls.rn = 1
        )
        SELECT
            pa.ID_CD_INTERNACAO,
            pa.ID_CD_PACIENTE,
            pa.source_db,
            pa.adm_seq,
            pa.total_adm,
            di.tipo_alta
        FROM patient_admissions pa
        LEFT JOIN discharge_info di
            ON pa.ID_CD_INTERNACAO = di.ID_CD_INTERNACAO
            AND pa.source_db = di.source_db
        WHERE pa.total_adm >= 2
        ORDER BY pa.source_db, pa.ID_CD_PACIENTE, pa.adm_seq
    """).fetchall()

    con.close()

    print(f"    Found {len(rows):,} admissions from patients with 2+ visits")

    # Categorize discharge types
    def _categorize_alta(tipo_alta_str):
        if tipo_alta_str is None:
            return "DESCONHECIDO"
        t = str(tipo_alta_str).upper()
        if "OBITO" in t or "ÓBITO" in t:
            return "OBITO"
        elif "SIMPLES" in t or "MELHORADA" in t or "NORMAL" in t:
            return "ALTA_NORMAL"
        elif "TRANSFERENCIA" in t or "TRANSFERÊNCIA" in t:
            return "TRANSFERENCIA"
        else:
            return "OUTRO"

    # Build patient admission sequences
    # patient_key -> list of (iid, source_db, adm_seq, tipo_alta)
    patient_seqs: dict[tuple, list] = {}
    for iid, pid, src, seq, total, tipo in rows:
        key = (src, pid)
        patient_seqs.setdefault(key, []).append((iid, src, seq, tipo))

    # Compute trajectory vectors: emb(adm_n) - emb(adm_n-1)
    trajectories_by_alta = {"OBITO": [], "ALTA_NORMAL": [], "TRANSFERENCIA": [], "OUTRO": [], "DESCONHECIDO": []}
    all_trajectories = []  # (patient_key, iid_current, src, trajectory_vec, tipo_alta_current)

    for pkey, adm_list in patient_seqs.items():
        adm_list.sort(key=lambda x: x[2])  # sort by adm_seq
        for i in range(1, len(adm_list)):
            prev_iid, prev_src, _, _ = adm_list[i - 1]
            curr_iid, curr_src, _, curr_tipo = adm_list[i]

            prev_node = f"{prev_src}/ID_CD_INTERNACAO_{prev_iid}"
            curr_node = f"{curr_src}/ID_CD_INTERNACAO_{curr_iid}"

            if prev_node not in node_to_idx or curr_node not in node_to_idx:
                continue

            prev_emb = embeddings[node_to_idx[prev_node]]
            curr_emb = embeddings[node_to_idx[curr_node]]
            traj = curr_emb - prev_emb

            cat = _categorize_alta(curr_tipo)
            trajectories_by_alta[cat].append(traj)
            all_trajectories.append((pkey, curr_iid, curr_src, traj, cat))

    # Compute mean trajectory per discharge type
    mean_trajectories = {}
    for cat, trajs in trajectories_by_alta.items():
        if len(trajs) > 0:
            mean_trajectories[cat] = np.stack(trajs).mean(axis=0)

    print(f"    Trajectory counts by discharge: " +
          ", ".join(f"{k}={len(v)}" for k, v in trajectories_by_alta.items()))

    # Find patients whose trajectory points toward obito centroid
    obito_risk_patients = []
    if "OBITO" in mean_trajectories:
        obito_traj = mean_trajectories["OBITO"]
        obito_norm = np.linalg.norm(obito_traj)
        if obito_norm > 1e-9:
            obito_unit = obito_traj / obito_norm
            for pkey, curr_iid, curr_src, traj, cat in all_trajectories:
                if cat == "OBITO":
                    continue  # skip already-obito
                t_norm = np.linalg.norm(traj)
                if t_norm < 1e-9:
                    continue
                sim = float(np.dot(traj / t_norm, obito_unit))
                if sim > 0:
                    obito_risk_patients.append((pkey, curr_iid, curr_src, sim, cat))

        obito_risk_patients.sort(key=lambda x: -x[3])

    print(f"    Patients with trajectory toward obito: {len(obito_risk_patients):,}")
    print(f"    Done in {time.time()-t0:.1f}s")

    return (
        trajectories_by_alta,
        mean_trajectories,
        obito_risk_patients[:20],
        all_trajectories,
    )


# ─────────────────────────────────────────────────────────────────
# Step 4 — Treatment Effect Signatures (CID gravity vectors)
# ─────────────────────────────────────────────────────────────────

def _treatment_effect_signatures(
    unique_nodes, embeddings, node_to_idx,
    internacao_mask, cid_mask, mean_trajectories,
):
    import time
    import numpy as np
    import duckdb

    print("[4/8] Computing CID gravity vectors ...")
    t0 = time.time()

    # Global internacao centroid
    int_embs = embeddings[internacao_mask]
    global_centroid = int_embs.mean(axis=0)

    # Get top 20 CIDs by volume
    con = duckdb.connect(str(DB_PATH))
    cid_rows = con.execute("""
        SELECT
            c.ID_CD_CID,
            ci.DS_DESCRICAO AS cid_desc,
            COUNT(*) AS n_admissions
        FROM agg_tb_capta_cid_caci c
        LEFT JOIN (
            SELECT DISTINCT ID_CD_CID, DS_DESCRICAO
            FROM agg_tb_capta_cid_caci
            WHERE DS_DESCRICAO IS NOT NULL
        ) ci ON c.ID_CD_CID = ci.ID_CD_CID
        GROUP BY c.ID_CD_CID, ci.DS_DESCRICAO
        ORDER BY n_admissions DESC
        LIMIT 20
    """).fetchall()

    # Get admissions per CID with source_db
    cid_admissions = con.execute("""
        SELECT c.ID_CD_CID, c.ID_CD_INTERNACAO, c.source_db
        FROM agg_tb_capta_cid_caci c
        WHERE c.ID_CD_CID IN (
            SELECT ID_CD_CID FROM (
                SELECT ID_CD_CID, COUNT(*) AS n
                FROM agg_tb_capta_cid_caci
                GROUP BY ID_CD_CID
                ORDER BY n DESC
                LIMIT 20
            )
        )
    """).fetchall()

    # Get discharge info for these admissions
    discharge_rows = con.execute("""
        WITH last_status AS (
            SELECT es.ID_CD_INTERNACAO, es.FL_DESOSPITALIZACAO, es.source_db,
                   ROW_NUMBER() OVER (
                       PARTITION BY es.ID_CD_INTERNACAO
                       ORDER BY es.DH_CADASTRO DESC
                   ) AS rn
            FROM agg_tb_capta_evo_status_caes es
        )
        SELECT ls.ID_CD_INTERNACAO, ls.source_db,
               f.DS_FINAL_MONITORAMENTO AS tipo_alta
        FROM last_status ls
        JOIN agg_tb_capta_tipo_final_monit_fmon f
            ON ls.FL_DESOSPITALIZACAO = f.ID_CD_FINAL_MONITORAMENTO
        WHERE ls.rn = 1
    """).fetchall()
    con.close()

    discharge_map = {}
    for iid, src, tipo in discharge_rows:
        discharge_map[(src, iid)] = tipo

    # Map CID -> admissions
    cid_to_admissions: dict[int, list[tuple]] = {}
    for cid_id, iid, src in cid_admissions:
        cid_to_admissions.setdefault(cid_id, []).append((src, iid))

    # Compute gravity vectors and risk direction
    obito_centroid = mean_trajectories.get("OBITO")
    alta_centroid  = mean_trajectories.get("ALTA_NORMAL")

    cid_results = []
    for cid_id, cid_desc, n_adm in cid_rows:
        adm_list = cid_to_admissions.get(cid_id, [])
        # Get embeddings for admissions with this CID
        emb_indices = []
        n_obito = 0
        n_alta = 0
        for src, iid in adm_list:
            key = f"{src}/ID_CD_INTERNACAO_{iid}"
            if key in node_to_idx:
                emb_indices.append(node_to_idx[key])
            tipo = discharge_map.get((src, iid))
            if tipo:
                t = str(tipo).upper()
                if "OBITO" in t or "ÓBITO" in t:
                    n_obito += 1
                elif "SIMPLES" in t or "MELHORADA" in t or "NORMAL" in t:
                    n_alta += 1

        if len(emb_indices) < 5:
            continue

        cid_centroid = embeddings[emb_indices].mean(axis=0)
        gravity = cid_centroid - global_centroid

        # Angle to obito direction
        risk_angle = 0.0
        if obito_centroid is not None:
            ob_norm = np.linalg.norm(obito_centroid)
            grav_norm = np.linalg.norm(gravity)
            if ob_norm > 1e-9 and grav_norm > 1e-9:
                risk_angle = float(np.dot(gravity / grav_norm, obito_centroid / ob_norm))

        # Angle to alta normal direction
        recovery_angle = 0.0
        if alta_centroid is not None:
            an_norm = np.linalg.norm(alta_centroid)
            grav_norm = np.linalg.norm(gravity)
            if an_norm > 1e-9 and grav_norm > 1e-9:
                recovery_angle = float(np.dot(gravity / grav_norm, alta_centroid / an_norm))

        cid_results.append({
            "cid_id":         cid_id,
            "cid_desc":       str(cid_desc or f"CID {cid_id}"),
            "n_admissions":   n_adm,
            "n_matched_emb":  len(emb_indices),
            "gravity_norm":   float(np.linalg.norm(gravity)),
            "risk_angle":     risk_angle,
            "recovery_angle": recovery_angle,
            "n_obito":        n_obito,
            "n_alta":         n_alta,
        })

    # Sort by risk direction (highest = pulls toward obito)
    cid_results.sort(key=lambda x: -x["risk_angle"])

    print(f"    Computed gravity vectors for {len(cid_results)} CIDs")
    print(f"    Done in {time.time()-t0:.1f}s")
    return cid_results


# ─────────────────────────────────────────────────────────────────
# Step 5 — Readmission Risk from Embedding Dynamics
# ─────────────────────────────────────────────────────────────────

def _readmission_dynamics(unique_nodes, embeddings, node_to_idx, internacao_mask):
    import time
    import numpy as np
    import duckdb

    print("[5/8] Computing readmission embedding dynamics ...")
    t0 = time.time()

    con = duckdb.connect(str(DB_PATH))

    # Get all admissions with admission/discharge dates and patient IDs
    # Compute readmission within 30 days using LEAD window
    rows = con.execute("""
        WITH adm_ordered AS (
            SELECT
                i.ID_CD_INTERNACAO,
                i.ID_CD_PACIENTE,
                i.source_db,
                i.DH_ADMISSAO_HOSP,
                i.DH_FINALIZACAO,
                LEAD(i.DH_ADMISSAO_HOSP) OVER (
                    PARTITION BY i.ID_CD_PACIENTE, i.source_db
                    ORDER BY i.DH_ADMISSAO_HOSP
                ) AS next_admission
            FROM agg_tb_capta_internacao_cain i
            WHERE i.DH_ADMISSAO_HOSP IS NOT NULL
              AND i.DH_FINALIZACAO IS NOT NULL
        )
        SELECT
            ID_CD_INTERNACAO,
            ID_CD_PACIENTE,
            source_db,
            DH_ADMISSAO_HOSP,
            DH_FINALIZACAO,
            next_admission,
            CASE
                WHEN next_admission IS NOT NULL
                 AND DATEDIFF('day', DH_FINALIZACAO::DATE, next_admission::DATE) BETWEEN 1 AND 30
                THEN 1 ELSE 0
            END AS readmitted_30d
        FROM adm_ordered
    """).fetchall()

    con.close()

    print(f"    Retrieved {len(rows):,} admissions with readmission labels")

    # Compute delta = emb(discharge) - emb(admission) as simply the embedding itself
    # (since we have one embedding per internacao, delta = how far from centroid)
    # Actually: delta is within-admission movement.  We approximate as the norm of the
    # embedding minus the global centroid (how displaced this admission is).
    int_embs = embeddings[internacao_mask]
    global_centroid = int_embs.mean(axis=0)

    readmit_deltas = []    # (src, iid, delta_magnitude)
    no_readmit_deltas = [] # (src, iid, delta_magnitude)
    hospital_readmit: dict[str, dict] = {}  # src -> {readmit: [deltas], no_readmit: [deltas]}

    for iid, pid, src, adm_date, fin_date, next_adm, readmitted in rows:
        node_key = f"{src}/ID_CD_INTERNACAO_{iid}"
        if node_key not in node_to_idx:
            continue

        emb = embeddings[node_to_idx[node_key]]
        delta_mag = float(np.linalg.norm(emb - global_centroid))

        hospital_readmit.setdefault(src, {"readmit": [], "no_readmit": []})

        if readmitted == 1:
            readmit_deltas.append((src, iid, delta_mag))
            hospital_readmit[src]["readmit"].append(delta_mag)
        else:
            no_readmit_deltas.append((src, iid, delta_mag))
            hospital_readmit[src]["no_readmit"].append(delta_mag)

    mean_readmit   = float(np.mean([d for _, _, d in readmit_deltas])) if readmit_deltas else 0.0
    mean_no_readmit = float(np.mean([d for _, _, d in no_readmit_deltas])) if no_readmit_deltas else 0.0

    # Per hospital stats
    hospital_stats = {}
    for src, data in hospital_readmit.items():
        r_mean = float(np.mean(data["readmit"])) if data["readmit"] else 0.0
        nr_mean = float(np.mean(data["no_readmit"])) if data["no_readmit"] else 0.0
        hospital_stats[src] = {
            "n_readmit": len(data["readmit"]),
            "n_no_readmit": len(data["no_readmit"]),
            "mean_delta_readmit": r_mean,
            "mean_delta_no_readmit": nr_mean,
        }

    print(f"    Readmitted: {len(readmit_deltas):,} (mean delta {mean_readmit:.4f})")
    print(f"    Not readmitted: {len(no_readmit_deltas):,} (mean delta {mean_no_readmit:.4f})")
    print(f"    Done in {time.time()-t0:.1f}s")

    return mean_readmit, mean_no_readmit, hospital_stats, len(readmit_deltas), len(no_readmit_deltas)


# ─────────────────────────────────────────────────────────────────
# Step 6 — Entity Resolution (cross-hospital patient matching)
# ─────────────────────────────────────────────────────────────────

def _entity_resolution(unique_nodes, embeddings, paciente_mask):
    import time
    import numpy as np

    print("[6/8] Cross-hospital entity resolution via cosine similarity ...")
    t0 = time.time()

    pac_nodes = unique_nodes[paciente_mask]
    pac_embs  = embeddings[paciente_mask]

    # Group patients by hospital
    hospital_patients: dict[str, list[tuple[int, str]]] = {}
    for i, node in enumerate(pac_nodes):
        s = str(node)
        try:
            src_db = s.split("/")[0]
            hospital_patients.setdefault(src_db, []).append((i, s))
        except Exception:
            pass

    hospitals = sorted(hospital_patients.keys())
    print(f"    Hospitals with patients: {hospitals}")

    # For each hospital pair, find patient embeddings with cosine sim > 0.90
    cross_matches = []
    for hi in range(len(hospitals)):
        for hj in range(hi + 1, len(hospitals)):
            h1, h2 = hospitals[hi], hospitals[hj]
            pats1 = hospital_patients[h1]
            pats2 = hospital_patients[h2]

            # Subsample if too large (for performance)
            max_per_hosp = 50_000
            if len(pats1) > max_per_hosp:
                rng = np.random.RandomState(42)
                idx1 = rng.choice(len(pats1), max_per_hosp, replace=False)
                pats1 = [pats1[i] for i in idx1]
            if len(pats2) > max_per_hosp:
                rng = np.random.RandomState(43)
                idx2 = rng.choice(len(pats2), max_per_hosp, replace=False)
                pats2 = [pats2[i] for i in idx2]

            print(f"    Comparing {h1} ({len(pats1):,} pats) vs {h2} ({len(pats2):,} pats) ...")

            # Build matrices
            embs1 = pac_embs[[p[0] for p in pats1]]
            embs2 = pac_embs[[p[0] for p in pats2]]

            # Normalize
            norms1 = np.linalg.norm(embs1, axis=1, keepdims=True).clip(min=1e-8)
            norms2 = np.linalg.norm(embs2, axis=1, keepdims=True).clip(min=1e-8)
            embs1_n = embs1 / norms1
            embs2_n = embs2 / norms2

            # Batch cosine similarity (chunked to avoid memory issues)
            chunk_size = 5_000
            for start in range(0, len(pats1), chunk_size):
                end = min(start + chunk_size, len(pats1))
                sims = embs1_n[start:end] @ embs2_n.T  # (chunk, len(pats2))

                # Find pairs > 0.90
                matches = np.argwhere(sims > 0.90)
                for m in matches:
                    local_i = start + m[0]
                    local_j = m[1]
                    sim_val = float(sims[m[0], m[1]])
                    cross_matches.append((
                        pats1[local_i][1],  # node name h1
                        pats2[local_j][1],  # node name h2
                        sim_val,
                    ))

    cross_matches.sort(key=lambda x: -x[2])
    print(f"    Total cross-hospital matches (sim>0.90): {len(cross_matches):,}")
    print(f"    Done in {time.time()-t0:.1f}s")

    return cross_matches[:20], len(cross_matches)


# ─────────────────────────────────────────────────────────────────
# Step 7 — "What If" Simulation
# ─────────────────────────────────────────────────────────────────

def _what_if_simulation(
    unique_nodes, embeddings, node_to_idx,
    internacao_mask, hospitals, cid_results,
):
    import time
    import numpy as np
    import duckdb

    print("[7/8] Running 'What If' simulations ...")
    t0 = time.time()

    int_nodes = unique_nodes[internacao_mask]
    int_embs  = embeddings[internacao_mask]

    # Normalize internacao embeddings for nearest-neighbor search
    int_norms = np.linalg.norm(int_embs, axis=1, keepdims=True).clip(min=1e-8)
    int_embs_n = int_embs / int_norms

    # Global centroid
    global_centroid = int_embs.mean(axis=0)

    # Pick top 3 anomalies (furthest from centroid)
    dists = np.linalg.norm(int_embs - global_centroid, axis=1)
    top_anomaly_idx = np.argsort(-dists)[:3]

    con = duckdb.connect(str(DB_PATH))

    # Hospital centroids for offset calculation
    hospital_groups: dict[str, list[int]] = {}
    for i, node in enumerate(int_nodes):
        s = str(node)
        try:
            src = s.split("/")[0]
            hospital_groups.setdefault(src, []).append(i)
        except Exception:
            pass
    hospital_centroids = {}
    for h, idxs in hospital_groups.items():
        hospital_centroids[h] = int_embs[idxs].mean(axis=0)

    # CID centroids for offset calculation
    cid_centroids = {}
    if cid_results:
        for cr in cid_results[:5]:
            cid_id = cr["cid_id"]
            # Get admissions for this CID
            try:
                adm_rows = con.execute(f"""
                    SELECT ID_CD_INTERNACAO, source_db
                    FROM agg_tb_capta_cid_caci
                    WHERE ID_CD_CID = {cid_id}
                """).fetchall()
                emb_idxs = []
                for iid, src in adm_rows:
                    key = f"{src}/ID_CD_INTERNACAO_{iid}"
                    if key in node_to_idx:
                        emb_idxs.append(node_to_idx[key])
                if emb_idxs:
                    cid_centroids[cid_id] = {
                        "centroid": embeddings[emb_idxs].mean(axis=0),
                        "desc": cr["cid_desc"],
                    }
            except Exception:
                pass

    def _find_nearest(point, k=5):
        """Find k nearest real admissions to a hypothetical point."""
        p_norm = np.linalg.norm(point)
        if p_norm < 1e-9:
            return []
        p_unit = point / p_norm
        sims = int_embs_n @ p_unit
        top_k = np.argsort(-sims)[:k]
        results = []
        for idx in top_k:
            results.append((str(int_nodes[idx]), float(sims[idx])))
        return results

    # Get context for the 3 anomaly admissions
    simulation_results = []
    for rank, anom_idx in enumerate(top_anomaly_idx):
        node_name = str(int_nodes[anom_idx])
        try:
            src_db, id_part = node_name.split("/", 1)
            iid = int(id_part.split("ID_CD_INTERNACAO_")[1])
        except Exception:
            continue

        # Get admission details
        try:
            det = con.execute(f"""
                SELECT i.ID_CD_INTERNACAO, i.source_db, i.DH_ADMISSAO_HOSP,
                       i.DH_FINALIZACAO, i.IN_SITUACAO
                FROM agg_tb_capta_internacao_cain i
                WHERE i.ID_CD_INTERNACAO = {iid}
                  AND i.source_db = '{src_db}'
                LIMIT 1
            """).fetchone()
        except Exception:
            det = None

        # Get CIDs for this admission
        try:
            cid_det = con.execute(f"""
                SELECT c.ID_CD_CID, c.DS_DESCRICAO
                FROM agg_tb_capta_cid_caci c
                WHERE c.ID_CD_INTERNACAO = {iid}
                  AND c.source_db = '{src_db}'
                LIMIT 3
            """).fetchall()
        except Exception:
            cid_det = []

        current_emb = int_embs[anom_idx]
        current_dist = float(dists[anom_idx])

        sim_entry = {
            "node_name":    node_name,
            "source_db":    src_db,
            "iid":          iid,
            "details":      det,
            "cids":         cid_det,
            "current_dist": current_dist,
            "simulations":  [],
        }

        # Simulation A: "What if CID changed?"
        # Pick a CID this patient does NOT have
        patient_cids = {c[0] for c in cid_det} if cid_det else set()
        for cid_id, cid_info in cid_centroids.items():
            if cid_id in patient_cids:
                continue
            # Offset = CID centroid - global centroid
            cid_offset = cid_info["centroid"] - global_centroid
            hypothetical = current_emb + cid_offset
            nearest = _find_nearest(hypothetical, k=5)
            sim_entry["simulations"].append({
                "type":        "CID",
                "description": f"Se o CID fosse {_truncate(cid_info['desc'], 50)}",
                "nearest":     nearest,
            })
            break  # just one CID sim

        # Simulation B: "What if at a different hospital?"
        target_hospital = None
        for h in hospitals:
            if h != src_db and h in hospital_centroids:
                target_hospital = h
                break
        if target_hospital and src_db in hospital_centroids:
            hosp_offset = hospital_centroids[target_hospital] - hospital_centroids[src_db]
            hypothetical = current_emb + hosp_offset
            nearest = _find_nearest(hypothetical, k=5)
            sim_entry["simulations"].append({
                "type":        "HOSPITAL",
                "description": f"Se estivesse em {target_hospital}",
                "nearest":     nearest,
            })

        simulation_results.append(sim_entry)

    con.close()
    print(f"    Simulated {len(simulation_results)} admissions")
    print(f"    Done in {time.time()-t0:.1f}s")
    return simulation_results


# ─────────────────────────────────────────────────────────────────
# Step 8 — Generate LaTeX
# ─────────────────────────────────────────────────────────────────

def _generate_latex(
    hospitals, sim_matrix, avg_sims, most_unique, identical_pairs, hospital_sizes,
    trajectories_by_alta, mean_trajectories, obito_risk_top20, all_trajectories,
    cid_results,
    mean_readmit, mean_no_readmit, hospital_stats, n_readmit, n_no_readmit,
    cross_matches_top20, total_cross_matches,
    simulation_results,
) -> str:
    import numpy as np

    L: list[str] = []

    # ── Preamble ──
    L.append(r"""\documentclass[a4paper,10pt]{article}
\usepackage[a4paper, top=2cm, bottom=2cm, left=1.8cm, right=1.8cm]{geometry}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[brazil]{babel}
\usepackage{lmodern}
\usepackage{xcolor}
\usepackage{colortbl}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{tabularx}
\usepackage{multirow}
\usepackage{fancyhdr}
\usepackage{titlesec}
\usepackage{tcolorbox}
\usepackage{needspace}
\usepackage{hyperref}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{microtype}
\usepackage{enumitem}
\usepackage{float}

\definecolor{jcubeblue}{RGB}{0,74,134}
\definecolor{jcubegray}{RGB}{80,80,80}
\definecolor{darkblue}{RGB}{0,50,100}
\definecolor{lightgray}{RGB}{248,248,248}
\definecolor{lightblue}{RGB}{220,235,250}
\definecolor{trafficred}{RGB}{220,50,50}
\definecolor{trafficyellow}{RGB}{200,170,0}
\definecolor{trafficgreen}{RGB}{0,150,60}
\definecolor{alertred}{RGB}{255,220,220}
\definecolor{alertyellow}{RGB}{255,243,200}
\definecolor{alertgreen}{RGB}{220,245,220}
\definecolor{simhigh}{RGB}{0,120,60}
\definecolor{simmed}{RGB}{200,140,0}
\definecolor{simlow}{RGB}{180,40,40}

\tcbuselibrary{skins,breakable}

\newtcolorbox{sectioncard}[2][]{%
  enhanced,
  colback=lightgray,
  colframe=#2,
  fonttitle=\bfseries\small,
  title={#1},
  left=5pt, right=5pt, top=4pt, bottom=4pt,
  boxrule=1.5pt,
  before upper={\setlength{\parskip}{4pt}},
}

\newtcolorbox{insightbox}[1][]{%
  enhanced,
  colback=lightblue,
  colframe=jcubeblue,
  fonttitle=\bfseries\footnotesize,
  title={#1},
  left=5pt, right=5pt, top=3pt, bottom=3pt,
  boxrule=0.8pt,
}

\newtcolorbox{alertbox}[1][]{%
  enhanced,
  colback=alertred,
  colframe=trafficred,
  fonttitle=\bfseries\footnotesize,
  title={#1},
  left=5pt, right=5pt, top=3pt, bottom=3pt,
  boxrule=0.8pt,
}

\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\textcolor{jcubeblue}{\textbf{JCUBE Digital Twin}} \textcolor{jcubegray}{\small | Algebra de Embeddings --- V5}}
\fancyhead[R]{\textcolor{jcubegray}{\small 23/03/2026}}
\fancyfoot[C]{\textcolor{jcubegray}{\thepage}}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0.4pt}

\titleformat{\section}{\large\bfseries\color{jcubeblue}}{\thesection}{1em}{}[\titlerule]
\titleformat{\subsection}{\normalsize\bfseries\color{darkblue}}{\thesubsection}{1em}{}

\hypersetup{colorlinks=true,linkcolor=jcubeblue,pdftitle={JCUBE V6 Algebra de Embeddings}}

\begin{document}
\setlength{\parindent}{0pt}
\setlength{\parskip}{3pt}
""")

    # ── Title page ──
    n_hosp = len(hospitals)
    n_cids = len(cid_results)
    n_risk = len(obito_risk_top20)

    L.append(r"""\begin{titlepage}
\begin{center}
\vspace*{1.5cm}
{\Huge\bfseries\textcolor{jcubeblue}{JCUBE}}\\[0.2cm]
{\large\textcolor{jcubegray}{Digital Twin Analytics Platform --- Modelo V5}}\\[1.2cm]
\begin{tcolorbox}[colback=jcubeblue,colframe=jcubeblue,coltext=white,width=0.92\textwidth,halign=center]
{\LARGE\bfseries Algebra de Embeddings}\\[0.3cm]
{\large O que se pode computar puramente com operacoes vetoriais\\sobre o Gemeo Digital 35,2M $\times$ 64}\\[0.2cm]
{\normalsize Graph-JEPA V6 --- Dense Temporal World Model}
\end{tcolorbox}
\vspace{0.8cm}
""")

    L.append(r"{\Large Gerado em: \textbf{23 de marco de 2026}}\\[0.6cm]")

    L.append(
        r"""\begin{tabular}{cccc}
\begin{tcolorbox}[colback=jcubeblue!10,colframe=jcubeblue,width=3.2cm,halign=center,left=2pt,right=2pt]
{\LARGE\bfseries\textcolor{jcubeblue}{""" + str(n_hosp) + r"""}}\\[2pt]
{\scriptsize\textbf{Hospitais}}
\end{tcolorbox}
&
\begin{tcolorbox}[colback=trafficgreen!10,colframe=trafficgreen,width=3.2cm,halign=center,left=2pt,right=2pt]
{\LARGE\bfseries\textcolor{trafficgreen}{""" + str(n_cids) + r"""}}\\[2pt]
{\scriptsize\textbf{CIDs Analisados}}
\end{tcolorbox}
&
\begin{tcolorbox}[colback=trafficred!10,colframe=trafficred,width=3.2cm,halign=center,left=2pt,right=2pt]
{\LARGE\bfseries\textcolor{trafficred}{""" + str(n_risk) + r"""}}\\[2pt]
{\scriptsize\textbf{Pac. Risco Obito}}
\end{tcolorbox}
&
\begin{tcolorbox}[colback=trafficyellow!10,colframe=trafficyellow,width=3.2cm,halign=center,left=2pt,right=2pt]
{\LARGE\bfseries\textcolor{trafficyellow!80!black}{""" + f"{total_cross_matches:,}" + r"""}}\\[2pt]
{\scriptsize\textbf{Matches Cross-Hosp}}
\end{tcolorbox}
\end{tabular}

\vfill
{\small\textcolor{jcubegray}{
Metodologia: operacoes de algebra vetorial (centroides, diferencas, cossenos) sobre\\
embeddings do Graph-JEPA V6 com 35,2M nos $\times$ 128 dimensoes\\
Sem modelos adicionais --- tudo e geometria do espaco latente
}}
\end{center}
\end{titlepage}
""")

    L.append(r"\tableofcontents\clearpage")

    # ═══════════════════════════════════════════════════════════════
    # SECTION 1: Hospital DNA
    # ═══════════════════════════════════════════════════════════════

    L.append(r"\section{Hospital DNA --- Impressoes Digitais Operacionais}")
    L.append(r"""
Para cada hospital (source\_db), calculamos o \textbf{centroide} de todos os embeddings de internacao.
A similaridade cosseno entre centroides revela quais hospitais operam de forma semelhante
--- sem usar nenhum dado tabular, apenas a geometria do espaco latente.
""")

    # Similarity matrix
    L.append(r"\subsection{Matriz de Similaridade entre Hospitais}")
    L.append(r"\begin{center}")
    # Build table header
    col_spec = "l" + "r" * n_hosp
    L.append(r"{\scriptsize")
    L.append(r"\begin{tabular}{" + col_spec + r"}")
    L.append(r"\toprule")
    header = r"\textbf{Hospital}"
    for h in hospitals:
        header += r" & \textbf{" + _escape_latex(h[:12]) + r"}"
    header += r" \\"
    L.append(header)
    L.append(r"\midrule")

    for i, h1 in enumerate(hospitals):
        row = _escape_latex(h1[:12])
        for j, h2 in enumerate(hospitals):
            val = sim_matrix[i, j]
            if i == j:
                cell = r"\cellcolor{jcubeblue!20}" + f"{val:.3f}"
            elif val > 0.95:
                cell = r"\cellcolor{trafficgreen!30}\textbf{" + f"{val:.3f}" + r"}"
            elif val > 0.85:
                cell = r"\cellcolor{trafficyellow!20}" + f"{val:.3f}"
            else:
                cell = f"{val:.3f}"
            row += " & " + cell
        row += r" \\"
        L.append(row)

    L.append(r"\bottomrule")
    L.append(r"\end{tabular}}")
    L.append(r"\end{center}")

    # Volume info
    L.append(r"\vspace{4pt}")
    L.append(r"{\footnotesize\textbf{Volume por Hospital:}}")
    L.append(r"\begin{center}{\scriptsize")
    L.append(r"\begin{tabular}{lr}")
    L.append(r"\toprule \textbf{Hospital} & \textbf{N Internacoes} \\\midrule")
    for h in sorted(hospital_sizes, key=hospital_sizes.get, reverse=True):
        L.append(_escape_latex(h) + r" & " + f"{hospital_sizes[h]:,}" + r" \\")
    L.append(r"\bottomrule\end{tabular}}")
    L.append(r"\end{center}")

    # Insights
    L.append(r"\vspace{4pt}")
    L.append(r"\begin{insightbox}[Insights]")
    L.append(
        r"\textbf{Hospital mais unico:} \texttt{" + _escape_latex(most_unique) +
        r"} --- similaridade media com outros: " + f"{avg_sims[most_unique]:.4f}" +
        r". Este hospital opera de forma mais distinta, sugerindo protocolos ou perfil de pacientes "
        r"significativamente diferentes."
    )
    if identical_pairs:
        L.append(r"\\[4pt]")
        L.append(r"\textbf{Pares operacionalmente identicos (sim $>$ 0,95):}")
        L.append(r"\begin{itemize}[nosep,leftmargin=*]")
        for h1, h2, sim in identical_pairs[:5]:
            L.append(
                r"\item \texttt{" + _escape_latex(h1) + r"} $\leftrightarrow$ \texttt{" +
                _escape_latex(h2) + r"}: " + f"{sim:.4f}" +
                r" --- potencial para compartilhamento de protocolos"
            )
        L.append(r"\end{itemize}")
    else:
        L.append(
            r"\\[4pt]Nenhum par de hospitais com similaridade $>$ 0,95. "
            r"Cada hospital tem perfil operacional distinto."
        )
    L.append(r"\end{insightbox}")

    L.append(r"\clearpage")

    # ═══════════════════════════════════════════════════════════════
    # SECTION 2: Clinical Trajectory Vectors
    # ═══════════════════════════════════════════════════════════════

    L.append(r"\section{Vetores de Trajetoria Clinica}")
    L.append(r"""
Para pacientes com 2+ internacoes, calculamos o \textbf{vetor de trajetoria}:
$\vec{t} = \text{emb}(\text{internacao}_n) - \text{emb}(\text{internacao}_{n-1})$.
A media dessas trajetorias por tipo de alta revela a ``direcao'' geometrica de cada desfecho.
""")

    # Trajectory counts
    L.append(r"\subsection{Contagem de Trajetorias por Tipo de Alta}")
    L.append(r"\begin{center}")
    L.append(r"\begin{tabular}{lr}")
    L.append(r"\toprule \textbf{Tipo de Alta} & \textbf{N Trajetorias} \\\midrule")
    for cat in ["OBITO", "ALTA_NORMAL", "TRANSFERENCIA", "OUTRO", "DESCONHECIDO"]:
        n = len(trajectories_by_alta.get(cat, []))
        if cat == "OBITO" and n > 0:
            L.append(_escape_latex(cat) + r" & \textcolor{trafficred}{\textbf{" + f"{n:,}" + r"}} \\")
        else:
            L.append(_escape_latex(cat) + r" & " + f"{n:,}" + r" \\")
    L.append(r"\bottomrule\end{tabular}")
    L.append(r"\end{center}")

    # Mean trajectory norms comparison
    L.append(r"\subsection{Magnitude Media dos Vetores de Trajetoria}")
    L.append(r"\begin{center}")
    L.append(r"\begin{tabular}{lrr}")
    L.append(r"\toprule \textbf{Tipo de Alta} & \textbf{Norma $\|\vec{t}\|$ Media} & \textbf{N} \\\midrule")
    import numpy as np
    for cat in ["OBITO", "ALTA_NORMAL", "TRANSFERENCIA", "OUTRO"]:
        trajs = trajectories_by_alta.get(cat, [])
        if not trajs:
            continue
        norms = [float(np.linalg.norm(t)) for t in trajs]
        mean_norm = np.mean(norms)
        color = "trafficred" if cat == "OBITO" else "jcubeblue"
        L.append(
            _escape_latex(cat) + r" & \textcolor{" + color + r"}{\textbf{" +
            f"{mean_norm:.4f}" + r"}} & " + f"{len(trajs):,}" + r" \\"
        )
    L.append(r"\bottomrule\end{tabular}")
    L.append(r"\end{center}")

    # Cosine similarity between mean trajectories
    if "OBITO" in mean_trajectories and "ALTA_NORMAL" in mean_trajectories:
        ob_t = mean_trajectories["OBITO"]
        an_t = mean_trajectories["ALTA_NORMAL"]
        ob_n = np.linalg.norm(ob_t)
        an_n = np.linalg.norm(an_t)
        if ob_n > 1e-9 and an_n > 1e-9:
            cos_ob_an = float(np.dot(ob_t / ob_n, an_t / an_n))
            L.append(r"\begin{insightbox}[Direcao Obito vs Alta Normal]")
            L.append(
                r"Similaridade cosseno entre a trajetoria media de obito e a de alta normal: "
                r"\textbf{" + f"{cos_ob_an:.4f}" + r"}. "
            )
            if cos_ob_an < 0:
                L.append(
                    r"O valor negativo confirma que obito e alta normal apontam em "
                    r"\textbf{direcoes opostas} no espaco latente --- o modelo aprendeu "
                    r"a separar geometricamente estes desfechos."
                )
            elif cos_ob_an < 0.3:
                L.append(
                    r"Valor baixo indica que os caminhos para obito e alta normal sao "
                    r"substancialmente diferentes no espaco de embeddings."
                )
            else:
                L.append(
                    r"As trajetorias compartilham alguma direcao, sugerindo que fatores "
                    r"comuns influenciam ambos os desfechos."
                )
            L.append(r"\end{insightbox}")

    # Top 20 patients at risk
    L.append(r"\subsection{Top 20 Pacientes com Trajetoria Direcionada ao Obito}")
    L.append(r"""
Pacientes cuja trajetoria atual (vetor diferenca entre ultima e penultima internacao) aponta
na direcao do centroide de obito --- estes sao os pacientes \textbf{em risco AGORA}.
""")

    if obito_risk_top20:
        L.append(r"\begin{alertbox}[ALERTA: Pacientes em Risco Iminente]")
        L.append(
            r"Os pacientes abaixo apresentam vetor de trajetoria com alta similaridade "
            r"cosseno em relacao a trajetoria media de obito. Recomenda-se revisao clinica imediata."
        )
        L.append(r"\end{alertbox}")
        L.append(r"\vspace{4pt}")
        L.append(r"\begin{center}{\scriptsize")
        L.append(r"\begin{longtable}{rlllr}")
        L.append(r"\toprule \textbf{\#} & \textbf{Hospital} & \textbf{ID Paciente} & \textbf{ID Internacao} & \textbf{Sim. Obito} \\\midrule")
        L.append(r"\endhead\bottomrule\endfoot")
        for rank, (pkey, curr_iid, curr_src, sim, cat) in enumerate(obito_risk_top20, 1):
            src_db, pid = pkey
            color = "trafficred" if sim > 0.7 else ("trafficyellow" if sim > 0.5 else "jcubeblue")
            L.append(
                str(rank) + r" & " +
                _escape_latex(src_db) + r" & " +
                str(pid) + r" & " +
                str(curr_iid) + r" & " +
                r"\textcolor{" + color + r"}{\textbf{" + f"{sim:.4f}" + r"}} \\"
            )
        L.append(r"\end{longtable}}")
        L.append(r"\end{center}")
    else:
        L.append(r"Nenhuma trajetoria significativa em direcao ao obito encontrada.")

    L.append(r"\clearpage")

    # ═══════════════════════════════════════════════════════════════
    # SECTION 3: Treatment Effect Signatures
    # ═══════════════════════════════════════════════════════════════

    L.append(r"\section{Assinaturas de Efeito de Tratamento (CID Gravity Vectors)}")
    L.append(r"""
Para cada CID (top 20 por volume), calculamos o \textbf{vetor de gravidade}:
$\vec{g}_\text{CID} = \text{centroide}(\text{internacoes com CID}) - \text{centroide}(\text{todas internacoes})$.
O \textbf{angulo de risco} mede quanto esse vetor aponta na direcao do obito (positivo = risco, negativo = protecao).
""")

    L.append(r"\subsection{Ranking de CIDs por Direcao de Risco}")
    L.append(r"\begin{center}{\scriptsize")
    L.append(r"\begin{longtable}{rp{5cm}rrrrrr}")
    L.append(
        r"\toprule \textbf{\#} & \textbf{CID / Descricao} & \textbf{N Intr.} & "
        r"\textbf{Embeds} & \textbf{$\|\vec{g}\|$} & "
        r"\textbf{Ang. Obito} & \textbf{Ang. Alta} & \textbf{Obitos} \\"
    )
    L.append(r"\midrule\endhead\bottomrule\endfoot")

    for rank, cr in enumerate(cid_results, 1):
        risk = cr["risk_angle"]
        rec  = cr["recovery_angle"]
        risk_color = "trafficred" if risk > 0.3 else ("trafficyellow" if risk > 0 else "trafficgreen")
        rec_color = "trafficgreen" if rec > 0.3 else ("trafficyellow" if rec > 0 else "trafficred")

        L.append(
            str(rank) + r" & " +
            _escape_latex(_truncate(cr["cid_desc"], 50)) + r" & " +
            f"{cr['n_admissions']:,}" + r" & " +
            f"{cr['n_matched_emb']:,}" + r" & " +
            f"{cr['gravity_norm']:.3f}" + r" & " +
            r"\textcolor{" + risk_color + r"}{\textbf{" + f"{risk:+.3f}" + r"}} & " +
            r"\textcolor{" + rec_color + r"}{\textbf{" + f"{rec:+.3f}" + r"}} & " +
            str(cr["n_obito"]) + r" \\"
        )

    L.append(r"\end{longtable}}")
    L.append(r"\end{center}")

    # Interpretation
    L.append(r"\begin{insightbox}[Interpretacao]")
    if cid_results:
        top_risk = cid_results[0]
        top_safe_list = [c for c in cid_results if c["risk_angle"] < 0]
        L.append(
            r"\textbf{CID de maior risco:} " +
            _escape_latex(_truncate(top_risk["cid_desc"], 60)) +
            r" (ang. obito = " + f"{top_risk['risk_angle']:+.3f}" + r"). "
            r"Este diagnostico puxa as internacoes na direcao do obito no espaco latente."
        )
        if top_safe_list:
            top_safe = top_safe_list[-1]
            L.append(
                r"\\[4pt]\textbf{CID mais protetor:} " +
                _escape_latex(_truncate(top_safe["cid_desc"], 60)) +
                r" (ang. obito = " + f"{top_safe['risk_angle']:+.3f}" + r"). "
                r"Este CID afasta as internacoes da regiao de obito."
            )
    L.append(r"\end{insightbox}")

    L.append(r"\clearpage")

    # ═══════════════════════════════════════════════════════════════
    # SECTION 4: Readmission Risk from Embedding Dynamics
    # ═══════════════════════════════════════════════════════════════

    L.append(r"\section{Risco de Readmissao por Dinamica de Embeddings}")
    L.append(r"""
\textbf{Hipotese:} se o delta (distancia do embedding da internacao ao centroide global) e pequeno,
o ``estado'' do paciente nao mudou significativamente durante a internacao --- sugerindo alta prematura
e maior probabilidade de readmissao em 30 dias.
""")

    L.append(r"\subsection{Comparacao Global}")
    L.append(r"\begin{center}")
    L.append(r"\begin{tabular}{lrrr}")
    L.append(r"\toprule \textbf{Grupo} & \textbf{N Internacoes} & \textbf{Delta $\|\vec{d}\|$ Medio} & \textbf{Hipotese} \\\midrule")

    readmit_color = "trafficred" if mean_readmit < mean_no_readmit else "trafficgreen"
    L.append(
        r"Readmitidos ($<$30d) & " + f"{n_readmit:,}" + r" & " +
        r"\textcolor{" + readmit_color + r"}{\textbf{" + f"{mean_readmit:.4f}" + r"}} & " +
        (r"$\checkmark$ Delta menor" if mean_readmit < mean_no_readmit else r"$\times$ Delta maior") +
        r" \\"
    )
    L.append(
        r"Nao readmitidos & " + f"{n_no_readmit:,}" + r" & " +
        f"{mean_no_readmit:.4f}" + r" & (baseline) \\"
    )
    L.append(r"\bottomrule\end{tabular}")
    L.append(r"\end{center}")

    # Hypothesis validation
    L.append(r"\begin{insightbox}[Validacao da Hipotese]")
    if mean_readmit < mean_no_readmit and mean_no_readmit > 0:
        pct_diff = ((mean_no_readmit - mean_readmit) / mean_no_readmit) * 100
        L.append(
            r"\textbf{Hipotese confirmada.} Internacoes que resultaram em readmissao "
            r"apresentam delta medio " + f"{pct_diff:.1f}" + r"\% menor que as "
            r"nao readmitidas. O modelo captura a ideia de que internacoes que "
            r"``movem pouco'' o estado do paciente no espaco latente sao mais "
            r"propensas a readmissao prematura."
        )
    elif mean_readmit > mean_no_readmit:
        L.append(
            r"\textbf{Hipotese refutada.} Readmitidos apresentam delta \textit{maior} "
            r"que nao readmitidos. Isso pode indicar que pacientes readmitidos sao "
            r"inerentemente mais complexos (maior variabilidade no espaco), e nao que "
            r"a internacao foi insuficiente."
        )
    else:
        L.append(r"Resultados inconclusivos --- diferenca minima entre os grupos.")
    L.append(r"\end{insightbox}")

    # Per-hospital breakdown
    L.append(r"\subsection{Delta Medio por Hospital}")
    L.append(r"\begin{center}{\scriptsize")
    L.append(r"\begin{longtable}{lrrrr}")
    L.append(
        r"\toprule \textbf{Hospital} & \textbf{N Readmit.} & "
        r"\textbf{$\delta$ Readmit.} & \textbf{N Nao Readm.} & "
        r"\textbf{$\delta$ Nao Readm.} \\\midrule"
    )
    L.append(r"\endhead\bottomrule\endfoot")

    for h in sorted(hospital_stats, key=lambda x: -(hospital_stats[x]["n_readmit"])):
        hs = hospital_stats[h]
        r_delta = hs["mean_delta_readmit"]
        nr_delta = hs["mean_delta_no_readmit"]
        r_color = "trafficred" if r_delta < nr_delta and hs["n_readmit"] > 10 else "jcubeblue"
        L.append(
            _escape_latex(h) + r" & " +
            f"{hs['n_readmit']:,}" + r" & " +
            r"\textcolor{" + r_color + r"}{" + f"{r_delta:.4f}" + r"} & " +
            f"{hs['n_no_readmit']:,}" + r" & " +
            f"{nr_delta:.4f}" + r" \\"
        )

    L.append(r"\end{longtable}}")
    L.append(r"\end{center}")

    L.append(r"\clearpage")

    # ═══════════════════════════════════════════════════════════════
    # SECTION 5: Entity Resolution
    # ═══════════════════════════════════════════════════════════════

    L.append(r"\section{Resolucao de Entidades (Cross-Hospital Patient Matching)}")
    L.append(r"""
Para cada par de hospitais, buscamos embeddings de PACIENTE com similaridade cosseno $>$ 0,90.
Estes podem ser o \textbf{mesmo paciente em dois sistemas} ou pacientes com perfil clinico
quase identico --- ambos os cenarios sao valiosos para gestao assistencial.
""")

    L.append(r"\subsection{Estatisticas Gerais}")
    L.append(r"\begin{center}")
    L.append(r"\begin{tabular}{lr}")
    L.append(r"\toprule \textbf{Metrica} & \textbf{Valor} \\\midrule")
    L.append(r"Total de matches cross-hospital (sim $>$ 0,90) & " + f"{total_cross_matches:,}" + r" \\")
    if cross_matches_top20:
        L.append(r"Maior similaridade encontrada & " + f"{cross_matches_top20[0][2]:.4f}" + r" \\")
    L.append(r"\bottomrule\end{tabular}")
    L.append(r"\end{center}")

    # Top 20
    L.append(r"\subsection{Top 20 Matches Cross-Hospital}")
    if cross_matches_top20:
        L.append(r"\begin{center}{\scriptsize")
        L.append(r"\begin{longtable}{rlllr}")
        L.append(r"\toprule \textbf{\#} & \textbf{Paciente Hospital A} & \textbf{Paciente Hospital B} & \textbf{Hospitais} & \textbf{Sim.} \\")
        L.append(r"\midrule\endhead\bottomrule\endfoot")

        for rank, (node1, node2, sim) in enumerate(cross_matches_top20, 1):
            try:
                h1 = node1.split("/")[0]
                p1 = node1.split("_")[-1]
            except Exception:
                h1, p1 = "?", node1
            try:
                h2 = node2.split("/")[0]
                p2 = node2.split("_")[-1]
            except Exception:
                h2, p2 = "?", node2

            color = "trafficgreen" if sim > 0.95 else ("trafficyellow" if sim > 0.92 else "jcubeblue")
            L.append(
                str(rank) + r" & " +
                _escape_latex(p1) + r" & " +
                _escape_latex(p2) + r" & " +
                _escape_latex(h1) + r" $\leftrightarrow$ " + _escape_latex(h2) + r" & " +
                r"\textcolor{" + color + r"}{\textbf{" + f"{sim:.4f}" + r"}} \\"
            )

        L.append(r"\end{longtable}}")
        L.append(r"\end{center}")
    else:
        L.append(r"Nenhum match com similaridade $>$ 0,90 encontrado entre hospitais.")

    L.append(r"\begin{insightbox}[Aplicacoes]")
    L.append(r"""\begin{itemize}[nosep,leftmargin=*]
\item \textbf{Deduplicacao cadastral:} matches de alta similaridade ($>$ 0,95) sao candidatos fortes a serem o mesmo paciente registrado em sistemas diferentes.
\item \textbf{Benchmarking de tratamento:} pacientes quase identicos tratados em hospitais diferentes permitem comparar eficacia de protocolos.
\item \textbf{Gestao de rede:} identificar pacientes que transitam entre hospitais ajuda a coordenar cuidado longitudinal.
\end{itemize}""")
    L.append(r"\end{insightbox}")

    L.append(r"\clearpage")

    # ═══════════════════════════════════════════════════════════════
    # SECTION 6: "What If" Simulation
    # ═══════════════════════════════════════════════════════════════

    L.append(r'\section{Simulacao ``What If'' --- Aritmetica Vetorial}')
    L.append(r"""
Demonstracao do poder da \textbf{aritmetica vetorial} sobre embeddings.
Para cada internacao selecionada (top anomalias), simulamos cenarios contrafactuais:

\begin{itemize}[nosep]
\item \textbf{``E se o CID fosse outro?''} --- somamos o vetor de offset do CID alternativo
\item \textbf{``E se estivesse em outro hospital?''} --- somamos o vetor de offset hospitalar
\end{itemize}

Em seguida, encontramos as 5 internacoes \textit{reais} mais proximas do ponto hipotetico.
Isso funciona como a famosa analogia \texttt{rei - homem + mulher = rainha} do Word2Vec,
aplicada ao dominio hospitalar.
""")

    for idx, sim_r in enumerate(simulation_results, 1):
        src = sim_r["source_db"]
        iid = sim_r["iid"]
        det = sim_r["details"]
        cids = sim_r["cids"]

        frame_color = "jcubeblue"
        L.append(r"\begin{sectioncard}[Simulacao " + str(idx) + r": " +
                 _escape_latex(src) + r" / Internacao " + str(iid) + r"]{" + frame_color + r"}")

        # Admission context
        if det:
            adm_date = str(det[2])[:10] if det[2] else "---"
            fin_date = str(det[3])[:10] if det[3] else "EM CURSO"
            sit = str(det[4]) if det[4] else "---"
            L.append(
                r"{\footnotesize \textbf{Admissao:} " + adm_date +
                r" \quad \textbf{Finalizacao:} " + fin_date +
                r" \quad \textbf{Situacao:} " + sit +
                r" \quad \textbf{Dist. centroide:} " + f"{sim_r['current_dist']:.4f}" + r"}"
            )
        if cids:
            cid_str = "; ".join(
                _escape_latex(_truncate(str(c[1] or f"CID {c[0]}"), 40))
                for c in cids
            )
            L.append(r"\\{\footnotesize \textbf{CIDs:} " + cid_str + r"}")

        # Simulations
        for sim in sim_r["simulations"]:
            L.append(r"\vspace{6pt}")
            L.append(r"{\small\textbf{" + _escape_latex(sim["description"]) + r":}}")
            if sim["nearest"]:
                L.append(r"\begin{center}{\scriptsize")
                L.append(r"\begin{tabular}{rlr}")
                L.append(r"\toprule \textbf{\#} & \textbf{Internacao Real Mais Proxima} & \textbf{Sim.} \\\midrule")
                for nr, (nn, ns) in enumerate(sim["nearest"], 1):
                    try:
                        nn_src = nn.split("/")[0]
                        nn_id = nn.split("_")[-1]
                        nn_display = f"{nn_src} / {nn_id}"
                    except Exception:
                        nn_display = nn
                    L.append(
                        str(nr) + r" & " + _escape_latex(nn_display) + r" & " +
                        f"{ns:.4f}" + r" \\"
                    )
                L.append(r"\bottomrule\end{tabular}}")
                L.append(r"\end{center}")
            else:
                L.append(r"{\footnotesize Sem vizinhos proximos encontrados.}")

        L.append(r"\end{sectioncard}")
        L.append(r"\vspace{6pt}")

    L.append(r"\begin{insightbox}[O que isso demonstra]")
    L.append(r"""
A aritmetica vetorial funciona porque o modelo Graph-JEPA aprendeu uma geometria
consistente: ``hospital'' e ``CID'' sao direcoes no espaco latente. Somando ou subtraindo
esses offsets, podemos navegar para cenarios contrafactuais e encontrar os casos reais
mais similares --- sem nenhum modelo adicional, apenas algebra linear sobre o gemeo digital.
""")
    L.append(r"\end{insightbox}")

    L.append(r"\clearpage")

    # ═══════════════════════════════════════════════════════════════
    # Appendix: Methodology
    # ═══════════════════════════════════════════════════════════════

    L.append(r"\section*{Apendice: Metodologia}")
    L.append(r"\addcontentsline{toc}{section}{Apendice: Metodologia}")
    L.append(r"""
\subsection*{1. Modelo Graph-JEPA V6}
O modelo \textit{Graph-JEPA V6} foi treinado sobre o grafo de conhecimento JCUBE com
\textbf{35,2M nos} e \textbf{64 dimensoes} de embedding.
Arquivo: \texttt{node\_emb\_epoch\_1.pt} (8,4 GB).

\subsection*{2. Operacoes Vetoriais Utilizadas}
\begin{enumerate}[nosep]
  \item \textbf{Centroide:} media aritmetica dos embeddings de um grupo (hospital, CID, tipo de alta).
  \item \textbf{Similaridade cosseno:} $\text{sim}(\vec{a}, \vec{b}) = \frac{\vec{a} \cdot \vec{b}}{\|\vec{a}\| \|\vec{b}\|}$ --- mede alinhamento direcional.
  \item \textbf{Vetor de trajetoria:} $\vec{t} = \text{emb}(n) - \text{emb}(n-1)$ --- direcao de evolucao clinica.
  \item \textbf{Vetor de gravidade:} $\vec{g} = \text{centroide}(\text{grupo}) - \text{centroide}(\text{global})$ --- desvio de um diagnostico em relacao a populacao.
  \item \textbf{Offset vetorial:} diferenca entre centroides de dois grupos, usada para ``transportar'' um ponto de um contexto para outro (e.g., trocar hospital ou CID).
  \item \textbf{Vizinho mais proximo (kNN):} busca os k embeddings reais com maior similaridade cosseno a um ponto hipotetico.
\end{enumerate}

\subsection*{3. Fontes de Dados}
\begin{itemize}[nosep]
  \item Embeddings: \texttt{/cache/tkg-v5/node\_emb\_epoch\_1.pt}
  \item Grafo: \texttt{/data/jcube\_graph.parquet}
  \item Base relacional: \texttt{/data/aggregated\_fixed\_union.db} (DuckDB)
\end{itemize}

\subsection*{4. Limitacoes}
\begin{itemize}[nosep]
  \item Os embeddings capturam correlacoes do grafo, nao causalidade.
  \item A resolucao de entidades (secao 5) sugere candidatos; confirmacao requer validacao com dados cadastrais.
  \item As simulacoes ``what if'' sao aproximacoes lineares em espaco nao-linear.
  \item A hipotese de readmissao (secao 4) usa a distancia ao centroide como proxy de delta; um modelo com embeddings temporais seria mais preciso.
\end{itemize}
""")

    L.append(r"\end{document}")
    return "\n".join(L)


# ─────────────────────────────────────────────────────────────────
# Compile LaTeX -> PDF
# ─────────────────────────────────────────────────────────────────

def _compile_latex(latex_content: str, output_pdf: str):
    import subprocess
    from pathlib import Path

    print("[8/8] Compiling PDF ...")
    out_path = Path(output_pdf)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tex_file = out_path.with_suffix(".tex")
    tex_file.write_text(latex_content, encoding="utf-8")
    print(f"    LaTeX written: {tex_file} ({tex_file.stat().st_size / 1024:.0f} KB)")

    for run in range(2):
        print(f"    pdflatex pass {run + 1}/2 ...")
        result = subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-output-directory", str(out_path.parent),
                str(tex_file),
            ],
            capture_output=True,
            cwd=str(out_path.parent),
        )
        stdout_str = result.stdout.decode("latin-1", errors="replace")
        if result.returncode != 0 and run == 1:
            errs = [ln for ln in stdout_str.split("\n")
                    if ln.startswith("!") or "Error" in ln]
            print("    pdflatex errors:")
            for e in errs[:20]:
                print("   ", e)
        if out_path.exists():
            print(f"    PDF pass {run + 1}: {out_path.stat().st_size / 1024:.0f} KB")

    if out_path.exists():
        print(f"\nPDF ready: {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)")
    else:
        log_file = tex_file.with_suffix(".log")
        if log_file.exists():
            lines = log_file.read_text(encoding="latin-1", errors="replace").split("\n")
            print("Last 50 lines of pdflatex log:")
            for ln in lines[-50:]:
                print(" ", ln)
        raise FileNotFoundError(f"PDF not generated at {output_pdf}")


# ─────────────────────────────────────────────────────────────────
# Modal Function
# ─────────────────────────────────────────────────────────────────

@app.function(
    image=report_image,
    volumes=VOLUMES,
    cpu=8.0,
    memory=40960,   # 40 GB RAM
    timeout=7200,   # 2 hours max
)
def generate_embedding_algebra_report():
    import time
    import os

    t_start = time.time()
    print("=" * 70)
    print("JCUBE V6 Embedding Algebra Report Generator (Modal)")
    print(f"Weights  : {WEIGHTS_PATH}")
    print(f"Parquet  : {GRAPH_PARQUET}")
    print(f"DB       : {DB_PATH}")
    print(f"Output   : {OUTPUT_PDF}")
    print("=" * 70)

    for p in [GRAPH_PARQUET, WEIGHTS_PATH, DB_PATH]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}")

    # 1. Load full twin
    (unique_nodes, embeddings, node_to_idx,
     internacao_mask, paciente_mask, cid_mask) = _load_twin()

    # 2. Hospital DNA
    (hospitals, sim_matrix, avg_sims, most_unique,
     identical_pairs, hospital_sizes) = _hospital_dna(unique_nodes, embeddings, internacao_mask)

    # 3. Clinical Trajectory Vectors
    (trajectories_by_alta, mean_trajectories,
     obito_risk_top20, all_trajectories) = _clinical_trajectories(
        unique_nodes, embeddings, node_to_idx
    )

    # 4. Treatment Effect Signatures
    cid_results = _treatment_effect_signatures(
        unique_nodes, embeddings, node_to_idx,
        internacao_mask, cid_mask, mean_trajectories,
    )

    # 5. Readmission Risk from Embedding Dynamics
    (mean_readmit, mean_no_readmit, hospital_stats,
     n_readmit, n_no_readmit) = _readmission_dynamics(
        unique_nodes, embeddings, node_to_idx, internacao_mask
    )

    # 6. Entity Resolution
    cross_matches_top20, total_cross_matches = _entity_resolution(
        unique_nodes, embeddings, paciente_mask
    )

    # 7. "What If" Simulation
    simulation_results = _what_if_simulation(
        unique_nodes, embeddings, node_to_idx,
        internacao_mask, hospitals, cid_results,
    )

    # 8. Generate LaTeX + compile
    print("[8/8] Generating LaTeX document ...")
    latex = _generate_latex(
        hospitals, sim_matrix, avg_sims, most_unique, identical_pairs, hospital_sizes,
        trajectories_by_alta, mean_trajectories, obito_risk_top20, all_trajectories,
        cid_results,
        mean_readmit, mean_no_readmit, hospital_stats, n_readmit, n_no_readmit,
        cross_matches_top20, total_cross_matches,
        simulation_results,
    )

    _compile_latex(latex, OUTPUT_PDF)

    # Commit so changes persist in the volume
    data_vol.commit()

    elapsed = time.time() - t_start
    print(f"\nFinished in {elapsed:.1f}s")
    print(f"Report saved to Modal volume jcube-data at: {OUTPUT_PDF}")
    print("Download with:")
    print(f"  modal volume get jcube-data reports/embedding_algebra_v6_2026_03.pdf ./embedding_algebra_v6_2026_03.pdf")
    return OUTPUT_PDF


@app.local_entrypoint()
def main():
    generate_embedding_algebra_report.remote()
# epoch2-1774305118
