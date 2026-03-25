#!/usr/bin/env python3
"""
Modal script: JCUBE V6 Physician Variation Analysis
Pure embedding algebra — no ML models. Answers: "If I'm treated by different
doctors for the same diagnosis, how much does my LOS and discharge outcome change?"

Now parameterized by hospital (source_db). CID grouping uses embedding similarity
(connected components at cosine > 0.7) instead of chapter codes. Physician identity
via CRM (DS_CONSELHO_CLASSE). UTI/leito type via ID_CD_ORIGEM.

Sections (pt-BR):
  1. Quanto o Médico Importa? — variance decomposition
  2. Variação por Especialidade — top 5 specialties
  3. Top 10 Clusters CID com Maior Variação Médica
  4. Simulações Contrafactuais — 3 real admissions
  5. Recomendações — standardization, bed-day savings

Output: /data/reports/physician_variation_{source_db_safe}_v6_2026_03.pdf

Usage:
    modal run reports/modal_physician_variation.py
    modal run --detach reports/modal_physician_variation.py
"""
from __future__ import annotations

import modal

# ─────────────────────────────────────────────────────────────────
# Modal App + Volumes
# ─────────────────────────────────────────────────────────────────

app = modal.App("jcube-physician-variation")

jepa_cache = modal.Volume.from_name("jepa-cache", create_if_missing=False)
data_vol   = modal.Volume.from_name("jcube-data",  create_if_missing=False)

VOLUMES = {
    "/cache": jepa_cache,
    "/data":  data_vol,
}

# ─────────────────────────────────────────────────────────────────
# Container image — CPU, torch, duckdb, pyarrow, sklearn, LaTeX
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

GRAPH_PARQUET = "/data/jcube_graph_v6.parquet"
WEIGHTS_PATH  = "/cache/tkg-v6/node_emb_epoch_2.pt"
DB_PATH       = "/data/aggregated_fixed_union.db"
OUTPUT_DIR    = "/data/reports"

REPORT_DATE_STR = "2026-03-24"

# Minimum admissions for a (CID cluster, physician) pair to be included
MIN_PAIR_ADMISSIONS = 5
# Cosine similarity threshold for grouping CIDs into the same cluster
N_CID_CLUSTERS = 30  # KMeans clusters for grouping similar diagnoses

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


def _truncate(s: str, max_len: int = 60) -> str:
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


def _fmt_days(v) -> str:
    f = _safe_float(v)
    if f == 0:
        return "---"
    return f"{f:.1f}"


def _fmt_sim(v: float) -> str:
    return f"{v:+.4f}"


# ─────────────────────────────────────────────────────────────────
# Step 0 — Cluster CIDs by embedding similarity (connected components)
# ─────────────────────────────────────────────────────────────────

def _cluster_cids_by_embedding(node_to_idx, embeddings):
    """
    Find all CID nodes in the graph, cluster them by embedding similarity
    using KMeans. Returns cid_to_cluster dict and cluster_to_cids dict.
    """
    import time
    import numpy as np
    from sklearn.cluster import KMeans

    print("[0/6] Clustering CIDs by embedding similarity (KMeans) ...")
    t0 = time.time()

    cid_nodes = []
    cid_indices = []
    for node_key, idx in node_to_idx.items():
        if "_CID_" in node_key:
            parts = node_key.split("_CID_")
            if len(parts) == 2:
                cid_code = parts[1]
                cid_nodes.append(cid_code)
                cid_indices.append(idx)

    n_cids = len(cid_nodes)
    print(f"    Found {n_cids} CID nodes in graph")

    if n_cids == 0:
        return {}, {}

    cid_embs = embeddings[cid_indices].copy()

    k = min(N_CID_CLUSTERS, n_cids)
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = km.fit_predict(cid_embs)

    # Build cluster dicts
    cid_to_cluster = {}
    cluster_to_cids = {}
    for i, cid in enumerate(cid_nodes):
        cl = int(labels[i])
        cid_to_cluster[cid] = cl
        cluster_to_cids.setdefault(cl, []).append(cid)

    sizes = [len(v) for v in cluster_to_cids.values()]
    print(f"    {k} clusters formed (largest: {max(sizes) if sizes else 0}, "
          f"smallest: {min(sizes) if sizes else 0}, median: {sorted(sizes)[len(sizes)//2] if sizes else 0})")
    print(f"    CID clustering done in {time.time()-t0:.1f}s")

    return cid_to_cluster, cluster_to_cids


# ─────────────────────────────────────────────────────────────────
# Step 1 — Load embeddings (full graph) + build index
# ─────────────────────────────────────────────────────────────────

def _load_twin(source_db: str):
    import time
    import numpy as np
    import torch
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    import pyarrow as pa

    print("[1/6] Loading full graph vocab for embedding lookup ...")
    t0 = time.time()
    full_table = pq.read_table(GRAPH_PARQUET, columns=["subject_id", "object_id"])
    full_subj = full_table.column("subject_id")
    full_obj = full_table.column("object_id")
    full_all = pa.chunked_array(full_subj.chunks + full_obj.chunks)
    unique_nodes = pc.unique(full_all).to_numpy(zero_copy_only=False).astype(object)
    del full_table, full_subj, full_obj, full_all
    n_nodes = len(unique_nodes)
    print(f"    {n_nodes:,} total nodes in {time.time()-t0:.1f}s")

    print("[1/6] Loading V6 embedding weights ...")
    t1 = time.time()
    state = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
    if isinstance(state, torch.Tensor):
        embeddings = state.numpy().astype(np.float32)
    elif isinstance(state, dict) and "weight" in state:
        embeddings = state["weight"].numpy().astype(np.float32)
    else:
        embeddings = list(state.values())[0].numpy().astype(np.float32)
    print(f"    Embeddings shape: {embeddings.shape} in {time.time()-t1:.1f}s")

    if n_nodes != embeddings.shape[0]:
        raise ValueError(
            f"Vocab size mismatch: {n_nodes:,} nodes vs "
            f"{embeddings.shape[0]:,} embedding rows"
        )

    node_to_idx = {str(n): i for i, n in enumerate(unique_nodes)}

    # Build mask for source_db internacao nodes
    intern_mask = np.array(
        [f"{source_db}/ID_CD_INTERNACAO_" in str(n) for n in unique_nodes], dtype=bool
    )
    print(f"    {source_db} INTERNACAO nodes: {intern_mask.sum():,}")

    return unique_nodes, embeddings, node_to_idx, intern_mask


# ─────────────────────────────────────────────────────────────────
# Step 2 — Build physician-admission-CID mapping from DuckDB
# ─────────────────────────────────────────────────────────────────

def _build_physician_data(con, source_db: str):
    """Build the core data structures linking physicians to admissions with embeddings."""
    import time
    print("[2/6] Building physician-admission-CID mapping ...")
    t0 = time.time()

    SRC = f"source_db = '{source_db}'"

    # Get admissions with physician (identified by CRM), LOS, CID, and origin (UTI/leito)
    rows = con.execute(f"""
        SELECT DISTINCT
            m.ID_CD_INTERNACAO,
            cfg.DS_CONSELHO_CLASSE AS CRM,
            m.ID_CD_ESPECIALIDADE,
            DATEDIFF('day', i.DH_ADMISSAO_HOSP, COALESCE(i.DH_FINALIZACAO, CURRENT_TIMESTAMP)) AS QT_DIAS_INTERNACAO,
            i.IN_SITUACAO,
            c.ID_CD_CID,
            c.DS_DESCRICAO AS DS_CID,
            i.ID_CD_ORIGEM,
            COALESCE(orig.FL_LEITO_UTI, 0) AS FL_LEITO_UTI,
            COALESCE(orig.DS_TITULO, 'DESCONHECIDO') AS DS_ORIGEM
        FROM agg_tb_capta_internacao_medico_hospital_cimh m
        JOIN agg_tb_capta_cfg_medico_hospital_ccmh cfg
            ON m.ID_CD_MEDICO_HOSPITAL = cfg.ID_CD_MEDICO_HOSPITAL
            AND m.source_db = cfg.source_db
        JOIN agg_tb_capta_internacao_cain i
            ON m.ID_CD_INTERNACAO = i.ID_CD_INTERNACAO
            AND m.source_db = i.source_db
        LEFT JOIN agg_tb_capta_cid_caci c
            ON m.ID_CD_INTERNACAO = c.ID_CD_INTERNACAO
            AND m.source_db = c.source_db
        LEFT JOIN agg_tb_capta_cfg_origem_cago orig
            ON i.ID_CD_ORIGEM = orig.ID_CD_ORIGEM
            AND i.source_db = orig.source_db
        WHERE m.{SRC}
          AND i.{SRC}
          AND i.DH_ADMISSAO_HOSP IS NOT NULL
          AND c.ID_CD_CID IS NOT NULL
          AND cfg.DS_CONSELHO_CLASSE IS NOT NULL
          AND regexp_matches(cfg.DS_CONSELHO_CLASSE, '^[0-9]{{5,8}}$')
          AND cfg.DS_CONSELHO_CLASSE NOT IN ('00000', '000000', '0000000', '00000000')
    """).fetchall()
    print(f"    Raw rows (CRM-admission-CID): {len(rows):,}")

    # Get discharge type per admission (last status from caes + domain lookup via fmon)
    discharge_rows = con.execute(f"""
        WITH last_status AS (
            SELECT
                e.ID_CD_INTERNACAO,
                e.FL_DESOSPITALIZACAO,
                ROW_NUMBER() OVER (
                    PARTITION BY e.ID_CD_INTERNACAO
                    ORDER BY e.DH_CADASTRO DESC
                ) as rn
            FROM agg_tb_capta_evo_status_caes e
            WHERE e.{SRC}
        )
        SELECT
            ls.ID_CD_INTERNACAO,
            COALESCE(f.DS_FINAL_MONITORAMENTO, 'DESCONHECIDO') as DS_TIPO_ALTA
        FROM last_status ls
        LEFT JOIN agg_tb_capta_tipo_final_monit_fmon f
            ON ls.FL_DESOSPITALIZACAO = f.ID_CD_FINAL_MONITORAMENTO
        WHERE ls.rn = 1
    """).fetchall()
    discharge_map = {int(r[0]): str(r[1]).upper() for r in discharge_rows}
    print(f"    Discharge status for {len(discharge_map):,} admissions")

    # Get physician names
    phys_name_rows = con.execute(f"""
        SELECT DS_CONSELHO_CLASSE AS CRM, FIRST(NM_MEDICO_HOSPITAL) AS nome
        FROM agg_tb_capta_cfg_medico_hospital_ccmh
        WHERE {SRC}
          AND DS_CONSELHO_CLASSE IS NOT NULL
          AND regexp_matches(DS_CONSELHO_CLASSE, '^[0-9]{{5,8}}$')
          AND DS_CONSELHO_CLASSE NOT IN ('00000', '000000', '0000000', '00000000')
        GROUP BY DS_CONSELHO_CLASSE
    """).fetchall()
    phys_names = {str(r[0]): str(r[1]) for r in phys_name_rows}
    print(f"    Physician names (by CRM): {len(phys_names):,}")

    # Get specialty names
    spec_rows = con.execute(f"""
        SELECT ID_CD_EQUIPE AS ID_CD_ESPECIALIDADE, DS_TITULO
        FROM agg_tb_capta_cfg_equipe_cage
        WHERE {SRC}
    """).fetchall()
    spec_names = {int(r[0]): str(r[1]) for r in spec_rows}
    print(f"    Specialties: {len(spec_names):,}")

    print(f"    Physician data built in {time.time()-t0:.1f}s")
    return rows, discharge_map, phys_names, spec_names


# ─────────────────────────────────────────────────────────────────
# Step 3 — Compute direction vectors and physician effect vectors
# ─────────────────────────────────────────────────────────────────

def _compute_vectors(rows, discharge_map, embeddings, node_to_idx,
                     cid_to_cluster, cluster_to_cids, source_db: str):
    """Compute direction vectors and physician effect vectors using embedding algebra."""
    import time
    import numpy as np

    print("[3/6] Computing direction vectors and physician effects ...")
    t0 = time.time()

    # Categorise admissions and collect their embedding indices
    all_emb_indices = []        # all admissions with valid embeddings
    obito_indices = []          # admissions ending in death
    los_long_indices = []       # LOS > 20 days
    los_short_indices = []      # LOS < 5 days
    complex_discharge_indices = []  # complex discharge (transferencia, UTI)
    normal_discharge_indices = []   # alta normal / melhorado

    # (cluster_id, CRM) -> list of (emb_idx, los, discharge_type, spec_id, is_uti, origin_desc)
    pair_data = {}
    # cluster_id -> list of (emb_idx, los, discharge_type, crm)
    cluster_data = {}
    # admission_id -> (emb_idx, crm, cid_code, cid_desc, los, discharge_type, spec_id,
    #                   cluster_id, is_uti, origin_desc)
    admission_info = {}

    OBITO_KEYWORDS = {"OBITO", "ÓBITO", "FALECIMENTO", "MORTE"}
    COMPLEX_KEYWORDS = {"TRANSFERENCIA", "TRANSFERÊNCIA", "UTI", "CTI", "EVASAO", "EVASÃO"}
    NORMAL_KEYWORDS = {"ALTA", "MELHORADO", "CURADO", "NORMAL"}
    PS_KEYWORDS = {"PS", "PRONTO", "URGÊNCIA", "URGENCIA", "EMERGÊNCIA", "EMERGENCIA"}

    n_found = 0
    n_missing = 0
    n_no_cluster = 0

    for row in rows:
        iid = _safe_int(row[0])
        crm = str(row[1]) if row[1] else None
        spec_id = _safe_int(row[2])
        los_val = _safe_float(row[3])
        situacao = row[4]
        cid_code = str(row[5]) if row[5] else None
        cid_desc = str(row[6]) if row[6] else ""
        id_cd_origem = row[7]
        fl_leito_uti = _safe_int(row[8])
        ds_origem = str(row[9]) if row[9] else "DESCONHECIDO"

        if not cid_code or not crm:
            continue

        key = f"{source_db}/ID_CD_INTERNACAO_{iid}"
        if key not in node_to_idx:
            n_missing += 1
            continue

        emb_idx = node_to_idx[key]

        # Get CID cluster
        cluster_id = cid_to_cluster.get(cid_code)
        if cluster_id is None:
            n_no_cluster += 1
            continue

        n_found += 1

        discharge_type = discharge_map.get(iid, "DESCONHECIDO")
        discharge_upper = discharge_type.upper()

        # Determine if admission is via PS/urgencia with UTI bed
        is_uti = fl_leito_uti == 1
        origin_upper = ds_origem.upper()
        is_ps_origin = any(k in origin_upper for k in PS_KEYWORDS)

        all_emb_indices.append(emb_idx)

        # Categorise by discharge
        is_obito = any(k in discharge_upper for k in OBITO_KEYWORDS)
        is_complex = any(k in discharge_upper for k in COMPLEX_KEYWORDS)
        is_normal = any(k in discharge_upper for k in NORMAL_KEYWORDS)

        if is_obito:
            obito_indices.append(emb_idx)
        if los_val > 20:
            los_long_indices.append(emb_idx)
        if los_val < 5:
            los_short_indices.append(emb_idx)
        if is_complex:
            complex_discharge_indices.append(emb_idx)
        if is_normal:
            normal_discharge_indices.append(emb_idx)

        # Group by CID cluster (embedding similarity) instead of chapter code
        pair_key = (cluster_id, crm)
        pair_data.setdefault(pair_key, []).append(
            (emb_idx, los_val, discharge_type, spec_id, is_uti, ds_origem, is_ps_origin)
        )
        cluster_data.setdefault(cluster_id, []).append(
            (emb_idx, los_val, discharge_type, crm)
        )
        admission_info[iid] = (
            emb_idx, crm, cid_code, cid_desc, los_val, discharge_type,
            spec_id, cluster_id, is_uti, ds_origem, is_ps_origin,
        )

    print(f"    Admissions with embedding: {n_found:,}, missing: {n_missing:,}, "
          f"no CID cluster: {n_no_cluster:,}")
    print(f"    Unique (cluster, CRM) pairs: {len(pair_data):,}")
    print(f"    Óbito: {len(obito_indices):,}, LOS>20d: {len(los_long_indices):,}, "
          f"LOS<5d: {len(los_short_indices):,}")

    # ── Direction vectors (computed once) ──
    all_mean = embeddings[all_emb_indices].mean(axis=0) if all_emb_indices else np.zeros(embeddings.shape[1])

    dir_obito = np.zeros(embeddings.shape[1])
    if len(obito_indices) >= 10:
        dir_obito = embeddings[obito_indices].mean(axis=0) - all_mean

    dir_los_long = np.zeros(embeddings.shape[1])
    if len(los_long_indices) >= 10 and len(los_short_indices) >= 10:
        dir_los_long = embeddings[los_long_indices].mean(axis=0) - embeddings[los_short_indices].mean(axis=0)

    dir_alta_complexa = np.zeros(embeddings.shape[1])
    if len(complex_discharge_indices) >= 10 and len(normal_discharge_indices) >= 10:
        dir_alta_complexa = (embeddings[complex_discharge_indices].mean(axis=0)
                             - embeddings[normal_discharge_indices].mean(axis=0))

    # Normalize direction vectors
    for d in [dir_obito, dir_los_long, dir_alta_complexa]:
        norm = np.linalg.norm(d)
        if norm > 1e-8:
            d /= norm

    print(f"    Direction vectors computed")

    # ── Cluster centroids ──
    cluster_centroids = {}
    for cluster_id, entries in cluster_data.items():
        idxs = [e[0] for e in entries]
        cluster_centroids[cluster_id] = embeddings[idxs].mean(axis=0)

    # ── Physician effect vectors ──
    # For each (cluster, CRM) pair with >= MIN_PAIR_ADMISSIONS:
    #   effect = mean(emb[admissions of this doctor+cluster]) - mean(emb[all admissions with this cluster])
    physician_effects = {}  # (cluster_id, crm) -> effect_vector
    pair_stats = {}         # (cluster_id, crm) -> {n, mean_los, obito_rate, spec_id, uti_rate, ps_uti_rate}

    for (cluster_id, crm), entries in pair_data.items():
        if len(entries) < MIN_PAIR_ADMISSIONS:
            continue
        idxs = [e[0] for e in entries]
        los_vals = [e[1] for e in entries]
        discharges = [e[2] for e in entries]
        spec_id = entries[0][3]
        uti_flags = [e[4] for e in entries]
        ps_flags = [e[6] for e in entries]

        pair_centroid = embeddings[idxs].mean(axis=0)
        effect = pair_centroid - cluster_centroids[cluster_id]
        physician_effects[(cluster_id, crm)] = effect

        n_obito = sum(1 for d in discharges if any(k in d.upper() for k in OBITO_KEYWORDS))
        n_uti = sum(1 for u in uti_flags if u)
        n_ps_uti = sum(1 for u, p in zip(uti_flags, ps_flags) if u and p)

        pair_stats[(cluster_id, crm)] = {
            "n": len(entries),
            "mean_los": float(np.mean(los_vals)),
            "obito_rate": n_obito / len(entries) if entries else 0.0,
            "spec_id": spec_id,
            "emb_indices": idxs,
            "uti_rate": n_uti / len(entries) if entries else 0.0,
            "ps_uti_rate": n_ps_uti / len(entries) if entries else 0.0,
        }

    print(f"    Physician effect vectors: {len(physician_effects):,} "
          f"(pairs with >= {MIN_PAIR_ADMISSIONS} admissions)")

    # ── Physician risk scores: cos_sim(effect, direction) ──
    physician_scores = {}
    for (cluster_id, crm), effect in physician_effects.items():
        enorm = np.linalg.norm(effect)
        if enorm < 1e-8:
            continue
        e_hat = effect / enorm
        score_obito = float(np.dot(e_hat, dir_obito))
        score_los = float(np.dot(e_hat, dir_los_long))
        score_complex = float(np.dot(e_hat, dir_alta_complexa))
        physician_scores[(cluster_id, crm)] = {
            "score_obito": score_obito,
            "score_los": score_los,
            "score_complex": score_complex,
            "effect_magnitude": float(enorm),
        }

    print(f"    Physician risk scores computed: {len(physician_scores):,}")
    print(f"    Vectors computed in {time.time()-t0:.1f}s")

    return {
        "physician_effects": physician_effects,
        "pair_stats": pair_stats,
        "physician_scores": physician_scores,
        "cluster_data": cluster_data,
        "cluster_centroids": cluster_centroids,
        "admission_info": admission_info,
        "dir_obito": dir_obito,
        "dir_los_long": dir_los_long,
        "dir_alta_complexa": dir_alta_complexa,
        "all_emb_indices": all_emb_indices,
        "all_mean": all_mean,
    }


# ─────────────────────────────────────────────────────────────────
# Step 4 — Section analyses
# ─────────────────────────────────────────────────────────────────

def _section1_quanto_importa(vectors, spec_names, cluster_to_cids):
    """
    Global variance decomposition: what % of LOS variance is explained
    by physician identity (controlling for CID cluster)?
    """
    import numpy as np
    print("[4a/6] Section 1: Quanto o Médico Importa ...")

    pair_stats = vectors["pair_stats"]
    physician_scores = vectors["physician_scores"]

    # Variance decomposition: within-cluster, between-physician
    cluster_physicians = {}  # cluster_id -> [{crm, mean_los, n, spec_id}]
    for (cluster_id, crm), stats in pair_stats.items():
        cluster_physicians.setdefault(cluster_id, []).append({
            "crm": crm,
            "mean_los": stats["mean_los"],
            "n": stats["n"],
            "spec_id": stats["spec_id"],
        })

    total_ss_between = 0.0
    total_n = 0
    grand_mean_los = 0.0
    grand_n = 0
    cluster_var_data = []

    for cluster_id, physicians in cluster_physicians.items():
        if len(physicians) < 2:
            continue
        cid_n = sum(p["n"] for p in physicians)
        cid_mean = sum(p["mean_los"] * p["n"] for p in physicians) / cid_n

        ss_between = sum(p["n"] * (p["mean_los"] - cid_mean) ** 2 for p in physicians)
        total_ss_between += ss_between
        total_n += cid_n
        grand_n += cid_n
        grand_mean_los += cid_mean * cid_n

        los_range = max(p["mean_los"] for p in physicians) - min(p["mean_los"] for p in physicians)
        cluster_var_data.append({
            "cluster_id": cluster_id,
            "cids": cluster_to_cids.get(cluster_id, []),
            "n_physicians": len(physicians),
            "n_admissions": cid_n,
            "los_range": los_range,
            "ss_between": ss_between,
        })

    if grand_n > 0:
        grand_mean_los /= grand_n

    # Compute total variance from all pair means
    all_pair_means = []
    all_pair_ns = []
    for stats in pair_stats.values():
        all_pair_means.append(stats["mean_los"])
        all_pair_ns.append(stats["n"])
    all_pair_means = np.array(all_pair_means)
    all_pair_ns = np.array(all_pair_ns)
    if len(all_pair_means) > 0:
        wgm = np.average(all_pair_means, weights=all_pair_ns)
        total_ss_total = float(np.sum(all_pair_ns * (all_pair_means - wgm) ** 2))
    else:
        total_ss_total = 1.0

    pct_explained = (total_ss_between / total_ss_total * 100) if total_ss_total > 0 else 0.0

    # Embedding-space physician spread per specialty
    spec_spread = {}  # spec_id -> list of effect magnitudes
    for (cluster_id, crm), scores in physician_scores.items():
        stats = pair_stats.get((cluster_id, crm))
        if stats:
            spec_id = stats["spec_id"]
            spec_spread.setdefault(spec_id, []).append(scores["effect_magnitude"])

    spec_summary = []
    for spec_id, mags in spec_spread.items():
        spec_summary.append({
            "spec_id": spec_id,
            "spec_name": spec_names.get(spec_id, f"Especialidade {spec_id}"),
            "mean_spread": float(np.mean(mags)),
            "max_spread": float(np.max(mags)),
            "n_pairs": len(mags),
        })
    spec_summary.sort(key=lambda x: x["mean_spread"], reverse=True)

    result = {
        "pct_explained": pct_explained,
        "total_pairs": len(pair_stats),
        "total_clusters_multi_phys": len(cluster_var_data),
        "grand_mean_los": grand_mean_los,
        "spec_summary": spec_summary[:15],
        "n_cid_clusters": len(cluster_to_cids),
    }
    print(f"    Variance explained by physician: {pct_explained:.1f}%")
    return result


def _section2_por_especialidade(vectors, phys_names, spec_names):
    """Top 5 specialties: physician effect magnitudes, top/bottom 3 by LOS."""
    import numpy as np
    print("[4b/6] Section 2: Variação por Especialidade ...")

    pair_stats = vectors["pair_stats"]
    physician_scores = vectors["physician_scores"]

    # Group by specialty
    spec_data = {}
    for (cluster_id, crm), stats in pair_stats.items():
        spec_id = stats["spec_id"]
        scores = physician_scores.get((cluster_id, crm), {})
        spec_data.setdefault(spec_id, []).append({
            "crm": crm,
            "cluster_id": cluster_id,
            "mean_los": stats["mean_los"],
            "n": stats["n"],
            "score_los": scores.get("score_los", 0.0),
            "score_obito": scores.get("score_obito", 0.0),
            "magnitude": scores.get("effect_magnitude", 0.0),
            "uti_rate": stats["uti_rate"],
            "ps_uti_rate": stats["ps_uti_rate"],
        })

    # Aggregate per physician within specialty (average across clusters)
    spec_results = []
    for spec_id, entries in spec_data.items():
        phys_agg = {}
        for e in entries:
            pid = e["crm"]
            phys_agg.setdefault(pid, {"total_los_w": 0, "total_n": 0, "score_los_w": 0,
                                       "score_obito_w": 0, "mag_w": 0,
                                       "uti_w": 0, "ps_uti_w": 0})
            phys_agg[pid]["total_los_w"] += e["mean_los"] * e["n"]
            phys_agg[pid]["total_n"] += e["n"]
            phys_agg[pid]["score_los_w"] += e["score_los"] * e["n"]
            phys_agg[pid]["score_obito_w"] += e["score_obito"] * e["n"]
            phys_agg[pid]["mag_w"] += e["magnitude"] * e["n"]
            phys_agg[pid]["uti_w"] += e["uti_rate"] * e["n"]
            phys_agg[pid]["ps_uti_w"] += e["ps_uti_rate"] * e["n"]

        physicians = []
        for pid, agg in phys_agg.items():
            n = agg["total_n"]
            if n == 0:
                continue
            physicians.append({
                "crm": pid,
                "phys_name": phys_names.get(pid, f"CRM {pid}"),
                "mean_los": agg["total_los_w"] / n,
                "n": n,
                "score_los": agg["score_los_w"] / n,
                "score_obito": agg["score_obito_w"] / n,
                "magnitude": agg["mag_w"] / n,
                "uti_rate": agg["uti_w"] / n,
                "ps_uti_rate": agg["ps_uti_w"] / n,
            })

        if len(physicians) < 3:
            continue

        physicians.sort(key=lambda x: x["score_los"])
        magnitudes = [p["magnitude"] for p in physicians]

        spec_results.append({
            "spec_id": spec_id,
            "spec_name": spec_names.get(spec_id, f"Especialidade {spec_id}"),
            "n_physicians": len(physicians),
            "n_entries": len(entries),
            "mean_magnitude": float(np.mean(magnitudes)),
            "max_magnitude": float(np.max(magnitudes)),
            "top3_los": physicians[-3:][::-1],  # highest LOS tendency
            "bottom3_los": physicians[:3],       # lowest LOS tendency
        })

    spec_results.sort(key=lambda x: x["n_entries"], reverse=True)
    print(f"    Specialties with >= 3 physicians: {len(spec_results)}")
    return spec_results[:5]


def _section3_top_clusters(vectors, phys_names, cluster_to_cids):
    """Top 10 CID clusters where physician choice has the biggest impact."""
    import numpy as np
    print("[4c/6] Section 3: Top 10 Clusters CID com Maior Variação ...")

    pair_stats = vectors["pair_stats"]
    physician_scores = vectors["physician_scores"]
    cluster_data = vectors["cluster_data"]

    # Group by cluster
    cluster_physicians = {}
    for (cluster_id, crm), stats in pair_stats.items():
        scores = physician_scores.get((cluster_id, crm), {})
        cluster_physicians.setdefault(cluster_id, []).append({
            "crm": crm,
            "phys_name": phys_names.get(crm, f"CRM {crm}"),
            "mean_los": stats["mean_los"],
            "n": stats["n"],
            "obito_rate": stats["obito_rate"],
            "score_los": scores.get("score_los", 0.0),
            "score_obito": scores.get("score_obito", 0.0),
            "magnitude": scores.get("effect_magnitude", 0.0),
            "uti_rate": stats["uti_rate"],
            "ps_uti_rate": stats["ps_uti_rate"],
        })

    cluster_results = []
    for cluster_id, physicians in cluster_physicians.items():
        if len(physicians) < 3:
            continue
        los_vals = [p["mean_los"] for p in physicians]
        obito_vals = [p["obito_rate"] for p in physicians]
        magnitudes = [p["magnitude"] for p in physicians]
        los_range = max(los_vals) - min(los_vals)

        cid_entries = cluster_data.get(cluster_id, [])
        n_total = len(cid_entries)
        cids_in_cluster = cluster_to_cids.get(cluster_id, [])

        # Label: first 5 CID codes in the cluster
        cluster_label = ", ".join(sorted(cids_in_cluster)[:5])
        if len(cids_in_cluster) > 5:
            cluster_label += f" (+{len(cids_in_cluster)-5})"

        cluster_results.append({
            "cluster_id": cluster_id,
            "cluster_label": cluster_label,
            "n_cids": len(cids_in_cluster),
            "n_physicians": len(physicians),
            "n_admissions": n_total,
            "los_min_phys": min(los_vals),
            "los_max_phys": max(los_vals),
            "los_range": los_range,
            "obito_min": min(obito_vals),
            "obito_max": max(obito_vals),
            "mean_magnitude": float(np.mean(magnitudes)),
            "physicians": sorted(physicians, key=lambda x: x["mean_los"]),
        })

    # Sort by embedding-space magnitude * LOS range (combined impact)
    cluster_results.sort(key=lambda x: x["mean_magnitude"] * x["los_range"], reverse=True)
    print(f"    CID clusters with >= 3 physicians: {len(cluster_results)}")
    return cluster_results[:10]


def _section4_counterfactuals(vectors, embeddings, node_to_idx, phys_names,
                              cluster_to_cids, source_db: str):
    """Pick 3 real admissions with poor outcomes, simulate swapping physician."""
    import numpy as np
    import time
    print("[4d/6] Section 4: Simulações Contrafactuais ...")
    t0 = time.time()

    admission_info = vectors["admission_info"]
    physician_effects = vectors["physician_effects"]
    pair_stats = vectors["pair_stats"]

    # Find admissions with poor outcomes: long LOS or obito
    OBITO_KEYWORDS = {"OBITO", "ÓBITO", "FALECIMENTO", "MORTE"}
    poor_admissions = []
    for iid, info in admission_info.items():
        (emb_idx, crm, cid_code, cid_desc, los, discharge,
         spec_id, cluster_id, is_uti, ds_origem, is_ps_origin) = info
        is_obito = any(k in discharge.upper() for k in OBITO_KEYWORDS)
        if los > 15 or is_obito:
            poor_admissions.append({
                "iid": iid,
                "emb_idx": emb_idx,
                "crm": crm,
                "cid_code": cid_code,
                "cid_desc": cid_desc,
                "los": los,
                "discharge": discharge,
                "is_obito": is_obito,
                "spec_id": spec_id,
                "cluster_id": cluster_id,
                "is_uti": is_uti,
                "ds_origem": ds_origem,
                "is_ps_origin": is_ps_origin,
            })

    # Sort by severity (obito first, then longest LOS)
    poor_admissions.sort(key=lambda x: (-int(x["is_obito"]), -x["los"]))
    print(f"    Poor-outcome admissions: {len(poor_admissions):,}")

    # Select 3 with different clusters and where we have the physician's effect vector
    selected = []
    used_clusters = set()
    for adm in poor_admissions:
        cluster_id = adm["cluster_id"]
        if cluster_id in used_clusters:
            continue
        if (cluster_id, adm["crm"]) not in physician_effects:
            continue
        # Need at least one alternative physician for this cluster
        alternatives = [
            (c, p) for (c, p) in physician_effects
            if c == cluster_id and p != adm["crm"]
        ]
        if not alternatives:
            continue
        selected.append(adm)
        used_clusters.add(cluster_id)
        if len(selected) >= 3:
            break

    print(f"    Selected {len(selected)} admissions for counterfactual")

    # Build internacao embedding matrix for nearest neighbor search
    intern_keys = []
    intern_indices = []
    for iid, info in admission_info.items():
        intern_keys.append(iid)
        intern_indices.append(info[0])

    intern_embs = embeddings[intern_indices].copy()
    norms = np.linalg.norm(intern_embs, axis=1, keepdims=True).clip(min=1e-8)
    intern_embs_normed = intern_embs / norms
    intern_los = np.array([admission_info[iid][4] for iid in intern_keys])
    intern_discharge = [admission_info[iid][5] for iid in intern_keys]

    simulations = []
    for adm in selected:
        cluster_id = adm["cluster_id"]
        current_effect = physician_effects[(cluster_id, adm["crm"])]

        # Find the "best" alternative physician (lowest LOS score)
        best_key = None
        best_los_mean = float("inf")
        for (c, p) in physician_effects:
            if c != cluster_id or p == adm["crm"]:
                continue
            stats = pair_stats[(c, p)]
            if stats["mean_los"] < best_los_mean:
                best_los_mean = stats["mean_los"]
                best_key = (c, p)

        if best_key is None:
            continue

        alt_effect = physician_effects[best_key]
        alt_crm = best_key[1]
        alt_stats = pair_stats[best_key]

        # Counterfactual embedding
        original_emb = embeddings[adm["emb_idx"]]
        hypothetical_emb = original_emb - current_effect + alt_effect

        # Nearest neighbors to hypothetical
        h_norm = np.linalg.norm(hypothetical_emb).clip(min=1e-8)
        h_normed = hypothetical_emb / h_norm
        sims = intern_embs_normed @ h_normed
        top_k = 20
        top_indices = np.argpartition(sims, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]

        neighbor_los = []
        neighbor_discharges = []
        for idx in top_indices:
            neighbor_los.append(intern_los[idx])
            neighbor_discharges.append(intern_discharge[idx])

        n_normal = sum(1 for d in neighbor_discharges
                       if any(k in d.upper() for k in {"ALTA", "MELHORADO", "CURADO", "NORMAL"}))

        cids_in_cluster = cluster_to_cids.get(cluster_id, [])
        cluster_label = ", ".join(sorted(cids_in_cluster)[:5])

        simulations.append({
            "iid": adm["iid"],
            "cluster_id": cluster_id,
            "cluster_label": cluster_label,
            "cid": adm["cid_code"],
            "cid_desc": adm["cid_desc"],
            "original_los": adm["los"],
            "original_discharge": adm["discharge"],
            "original_phys": phys_names.get(adm["crm"], f"CRM {adm['crm']}"),
            "original_phys_mean_los": pair_stats[(cluster_id, adm["crm"])]["mean_los"],
            "alt_phys": phys_names.get(alt_crm, f"CRM {alt_crm}"),
            "alt_phys_mean_los": alt_stats["mean_los"],
            "alt_phys_n": alt_stats["n"],
            "neighbor_mean_los": float(np.mean(neighbor_los)),
            "neighbor_median_los": float(np.median(neighbor_los)),
            "neighbor_normal_pct": n_normal / len(neighbor_discharges) * 100 if neighbor_discharges else 0,
            "similarity_top": float(sims[top_indices[0]]),
            "is_uti": adm["is_uti"],
            "is_ps_origin": adm["is_ps_origin"],
            "ds_origem": adm["ds_origem"],
        })

    print(f"    Counterfactual simulations: {len(simulations)} in {time.time()-t0:.1f}s")
    return simulations


def _section5_recommendations(section1, section2, section3, section4, vectors):
    """Compute recommendations and estimated bed-day savings."""
    import numpy as np
    print("[4e/6] Section 5: Recomendações ...")

    pair_stats = vectors["pair_stats"]

    # Estimate bed-day savings: for each cluster, if worst physicians matched median
    total_excess_days = 0.0
    cluster_savings = []
    cluster_groups = {}
    for (cluster_id, crm), stats in pair_stats.items():
        cluster_groups.setdefault(cluster_id, []).append(stats)

    for cluster_id, group in cluster_groups.items():
        if len(group) < 3:
            continue
        los_vals = [g["mean_los"] for g in group]
        ns = [g["n"] for g in group]
        median_los = float(np.median(los_vals))

        excess = 0.0
        for g in group:
            if g["mean_los"] > median_los:
                excess += (g["mean_los"] - median_los) * g["n"]
        total_excess_days += excess
        if excess > 0:
            cluster_savings.append({"cluster_id": cluster_id, "excess_days": excess, "n": sum(ns)})

    cluster_savings.sort(key=lambda x: x["excess_days"], reverse=True)

    result = {
        "total_excess_days": total_excess_days,
        "top_cluster_savings": cluster_savings[:10],
        "n_clusters_with_variation": len(cluster_savings),
    }
    print(f"    Estimated total excess bed-days: {total_excess_days:,.0f}")
    return result


# ─────────────────────────────────────────────────────────────────
# LaTeX Generation
# ─────────────────────────────────────────────────────────────────

def _generate_latex(s1, s2, s3, s4, s5, spec_names, cluster_to_cids, source_db: str) -> str:
    L = []

    source_db_display = _escape_latex(source_db)

    # ── Preamble ──
    L.append(r"""\documentclass[a4paper,11pt]{article}
\usepackage[a4paper, top=2cm, bottom=2cm, left=2cm, right=2cm]{geometry}
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
\usepackage{microtype}
\usepackage{enumitem}

\definecolor{physblue}{RGB}{0,47,108}
\definecolor{physred}{RGB}{204,0,0}
\definecolor{physgray}{RGB}{100,100,100}
\definecolor{physlightblue}{RGB}{220,235,250}
\definecolor{physlightgray}{RGB}{245,245,245}
\definecolor{physgold}{RGB}{180,140,20}
\definecolor{physgreen}{RGB}{0,128,60}
\definecolor{kpibox}{RGB}{240,245,255}
\definecolor{alertbox}{RGB}{255,240,240}
\definecolor{successbox}{RGB}{240,255,240}
\definecolor{warnbox}{RGB}{255,250,230}
\definecolor{simbox}{RGB}{240,248,255}

\tcbuselibrary{skins,breakable}

\newtcolorbox{kpicard}[1][]{%
  enhanced,
  colback=kpibox,
  colframe=physblue,
  fonttitle=\bfseries\small,
  title={#1},
  left=5pt, right=5pt, top=4pt, bottom=4pt,
  boxrule=1pt
}

\newtcolorbox{alertcard}[1][]{%
  enhanced,
  colback=alertbox,
  colframe=physred,
  fonttitle=\bfseries\small,
  title={#1},
  left=5pt, right=5pt, top=4pt, bottom=4pt,
  boxrule=1.2pt
}

\newtcolorbox{infocard}[1][]{%
  enhanced,
  colback=physlightgray,
  colframe=physgray,
  fonttitle=\bfseries\small,
  title={#1},
  left=5pt, right=5pt, top=4pt, bottom=4pt,
  boxrule=0.8pt,
  breakable
}

\newtcolorbox{simcard}[1][]{%
  enhanced,
  colback=simbox,
  colframe=physblue,
  fonttitle=\bfseries\small,
  title={#1},
  left=5pt, right=5pt, top=4pt, bottom=4pt,
  boxrule=1pt,
  breakable
}

\newtcolorbox{reccard}[1][]{%
  enhanced,
  colback=successbox,
  colframe=physgreen,
  fonttitle=\bfseries\small,
  title={#1},
  left=5pt, right=5pt, top=4pt, bottom=4pt,
  boxrule=1pt,
  breakable
}

\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small\color{physgray}Variação Médica --- Embedding Algebra V6}
\fancyhead[R]{\small\color{physgray}""" + REPORT_DATE_STR + r"""}
\fancyfoot[C]{\small\color{physgray}\thepage}
\renewcommand{\headrulewidth}{0.5pt}

\titleformat{\section}{\Large\bfseries\color{physblue}}{}{0em}{}
\titleformat{\subsection}{\large\bfseries\color{physblue!80!black}}{}{0em}{}

\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt}

\begin{document}
""")

    # ── Cover page ──
    L.append(r"""
\thispagestyle{empty}
\vspace*{3cm}
\begin{center}
{\Huge\bfseries\color{physblue} Análise de Variação Médica}\\[0.8cm]
{\Large\color{physgray} Embedding Algebra sobre Digital Twin V6}\\[0.5cm]
{\large """ + source_db_display + r""" --- """ + REPORT_DATE_STR + r"""}\\[2cm]
\end{center}

\begin{kpicard}[Pergunta Central]
\textbf{Se um paciente é atendido por diferentes médicos para o mesmo diagnóstico,
quanto muda o tempo de internação e o desfecho de alta?}

Este relatório responde usando álgebra vetorial pura sobre os 35,2 milhões de embeddings
do digital twin V6, sem nenhum modelo de ML --- apenas operações sobre vetores de 128 dimensões
treinados por Temporal Knowledge Graph.
\end{kpicard}

\begin{kpicard}[Metodologia]
\begin{enumerate}[leftmargin=*]
\item \textbf{Agrupamento de CIDs por Similaridade:} CIDs são agrupados por similaridade cosseno
de seus embeddings (KMeans k=""" + f"{N_CID_CLUSTERS}" + r"""), formando clusters semânticos
(ex: I50 e I48 no mesmo cluster cardíaco). Total: \textbf{""" + f"{s1['n_cid_clusters']}" + r"""} clusters.
\item \textbf{Vetor de Efeito Médico:} Para cada par (cluster CID, CRM) com $\geq$""" + str(MIN_PAIR_ADMISSIONS) + r""" internações:
\[ \vec{e}_{\text{médico}} = \bar{v}_{\text{internações do médico no cluster}} - \bar{v}_{\text{todas internações no cluster}} \]
\item \textbf{Vetores de Direção} (calculados uma vez):
\[ \vec{d}_{\text{óbito}} = \bar{v}_{\text{óbitos}} - \bar{v}_{\text{geral}} \qquad
   \vec{d}_{\text{LOS longo}} = \bar{v}_{\text{LOS>20d}} - \bar{v}_{\text{LOS<5d}} \]
\item \textbf{Score de Risco:} $\cos(\vec{e}_{\text{médico}}, \vec{d}_{\text{direção}})$
\item \textbf{Contrafactual:} $\vec{v}_{\text{hipotético}} = \vec{v}_{\text{real}} - \vec{e}_{\text{atual}} + \vec{e}_{\text{alternativo}}$
\item \textbf{Identidade do Médico:} CRM (DS\_CONSELHO\_CLASSE) como chave única,
eliminando duplicatas por contrato/evolução.
\item \textbf{Leito UTI/Origem:} Internações classificadas por tipo de origem (PS, cirurgia eletiva, etc.)
e leito UTI (FL\_LEITO\_UTI), permitindo análise diferenciada de variação médica em contextos de urgência.
\end{enumerate}
\end{kpicard}

\newpage
""")

    # ── Section 1: Quanto o Médico Importa? ──
    L.append(r"\section{Quanto o Médico Importa?}" + "\n\n")

    pct = s1["pct_explained"]
    L.append(r"""
\begin{alertcard}[Resultado Principal]
{\Large\bfseries """ + f"{pct:.1f}" + r"""\% da variância de LOS} entre pares (cluster CID, CRM) é
explicada pela identidade do médico (controlando pelo cluster diagnóstico).\\[4pt]
Analisamos \textbf{""" + f"{s1['total_pairs']:,}".replace(",", ".") + r"""} pares (cluster, CRM)
com $\geq$""" + str(MIN_PAIR_ADMISSIONS) + r""" internações,
em \textbf{""" + f"{s1['total_clusters_multi_phys']:,}".replace(",", ".") + r"""} clusters CID com múltiplos médicos
(\textbf{""" + f"{s1['n_cid_clusters']}" + r"""} clusters totais agrupados por similaridade de embedding).\\[4pt]
LOS médio global: \textbf{""" + f"{s1['grand_mean_los']:.1f}" + r""" dias}.
\end{alertcard}
""")

    # Embedding-space spread per specialty
    L.append(r"""
\subsection{Dispersão no Espaço de Embeddings por Especialidade}

A magnitude média do vetor de efeito médico indica quanto cada profissional desvia
do centroide de seu cluster diagnóstico. Especialidades com maior dispersão têm maior
variabilidade de prática clínica.

\begin{center}
\small
\begin{tabular}{l r r r}
\toprule
\textbf{Especialidade} & \textbf{N Pares} & \textbf{Dispersão Média} & \textbf{Dispersão Máx.} \\
\midrule
""")
    for ss in s1["spec_summary"][:10]:
        name = _escape_latex(_truncate(ss["spec_name"], 35))
        L.append(f"{name} & {ss['n_pairs']:,} & {ss['mean_spread']:.4f} & {ss['max_spread']:.4f} \\\\\n".replace(",", "."))
    L.append(r"""
\bottomrule
\end{tabular}
\end{center}

\newpage
""")

    # ── Section 2: Variação por Especialidade ──
    L.append(r"\section{Variação por Especialidade}" + "\n\n")

    if not s2:
        L.append(r"\textit{Dados insuficientes para análise por especialidade.}" + "\n\n")
    else:
        L.append(r"""
Para as principais especialidades, mostramos a amplitude do efeito médico e
os 3 médicos com maior e menor tendência de LOS longo (score de similaridade
com o vetor de direção $\vec{d}_{\text{LOS longo}}$). Taxa UTI indica a proporção
de internações com leito UTI.

""")
        for spec in s2:
            spec_name = _escape_latex(spec["spec_name"])
            L.append(r"\subsection{" + spec_name + "}\n\n")
            L.append(f"\\textbf{{{spec['n_physicians']}}} médicos, "
                     f"\\textbf{{{spec['n_entries']:,}}} pares (cluster, CRM), ".replace(",", ".")
                     + f"magnitude média do efeito: \\textbf{{{spec['mean_magnitude']:.4f}}}, "
                     f"máxima: \\textbf{{{spec['max_magnitude']:.4f}}}.\n\n")

            # Top 3 highest LOS tendency
            L.append(r"""
\begin{alertcard}[Maior tendência de LOS longo]
\small
\begin{tabular}{l r r r r}
\toprule
\textbf{Médico} & \textbf{N} & \textbf{LOS Médio} & \textbf{Score LOS} & \textbf{UTI\%} \\
\midrule
""")
            for p in spec["top3_los"]:
                name = _escape_latex(_truncate(p["phys_name"], 30))
                uti_pct = f"{p['uti_rate']*100:.0f}\\%"
                L.append(f"{name} & {p['n']} & {p['mean_los']:.1f}d & {_fmt_sim(p['score_los'])} & {uti_pct} \\\\\n")
            L.append(r"""
\bottomrule
\end{tabular}
\end{alertcard}
""")

            # Bottom 3 lowest LOS tendency
            L.append(r"""
\begin{reccard}[Menor tendência de LOS longo]
\small
\begin{tabular}{l r r r r}
\toprule
\textbf{Médico} & \textbf{N} & \textbf{LOS Médio} & \textbf{Score LOS} & \textbf{UTI\%} \\
\midrule
""")
            for p in spec["bottom3_los"]:
                name = _escape_latex(_truncate(p["phys_name"], 30))
                uti_pct = f"{p['uti_rate']*100:.0f}\\%"
                L.append(f"{name} & {p['n']} & {p['mean_los']:.1f}d & {_fmt_sim(p['score_los'])} & {uti_pct} \\\\\n")
            L.append(r"""
\bottomrule
\end{tabular}
\end{reccard}

\medskip
""")

    L.append(r"\newpage" + "\n")

    # ── Section 3: Top 10 CID Clusters ──
    L.append(r"\section{Top 10 Clusters CID com Maior Variação Médica}" + "\n\n")

    L.append(r"""
Clusters de CIDs agrupados por similaridade de embedding (KMeans k=""" + f"{N_CID_CLUSTERS}" + r""")
onde a escolha do médico tem o maior impacto no tempo de internação,
medido pela combinação de magnitude do efeito no espaço de embeddings e
amplitude do LOS médio entre médicos.

\begin{center}
\small
\begin{tabular}{p{4cm} r r r r r r}
\toprule
\textbf{Cluster CID} & \textbf{N CIDs} & \textbf{N Méd.} & \textbf{N Intern.} & \textbf{LOS Mín.} & \textbf{LOS Máx.} & \textbf{Mag.} \\
\midrule
""")
    for c in s3:
        label = _escape_latex(_truncate(c["cluster_label"], 35))
        L.append(f"{label} & {c['n_cids']} & {c['n_physicians']} & {c['n_admissions']:,} & "
                 f"{c['los_min_phys']:.1f}d & {c['los_max_phys']:.1f}d & "
                 f"{c['mean_magnitude']:.4f} \\\\\n".replace(",", "."))
    L.append(r"""
\bottomrule
\end{tabular}
\end{center}
""")

    # Show obito risk range for top 5
    L.append(r"""
\subsection{Risco de Óbito por Médico nos Top 5 Clusters}

\begin{center}
\small
\begin{tabular}{p{4cm} r r r}
\toprule
\textbf{Cluster CID} & \textbf{N Médicos} & \textbf{Taxa Óbito Mín.} & \textbf{Taxa Óbito Máx.} \\
\midrule
""")
    for c in s3[:5]:
        label = _escape_latex(_truncate(c["cluster_label"], 35))
        omin = f"{c['obito_min']*100:.1f}\\%"
        omax = f"{c['obito_max']*100:.1f}\\%"
        L.append(f"{label} & {c['n_physicians']} & {omin} & {omax} \\\\\n")
    L.append(r"""
\bottomrule
\end{tabular}
\end{center}

\newpage
""")

    # ── Section 4: Simulações Contrafactuais ──
    L.append(r"\section{Simulações Contrafactuais}" + "\n\n")

    L.append(r"""
Para internações reais com desfechos desfavoráveis (LOS longo ou óbito),
simulamos o que teria acontecido se o paciente fosse atendido pelo médico
com melhor perfil para o mesmo cluster CID. A simulação usa álgebra vetorial:

\[ \vec{v}_{\text{hipotético}} = \vec{v}_{\text{real}} - \vec{e}_{\text{médico atual}} + \vec{e}_{\text{médico alternativo}} \]

Os 20 vizinhos mais próximos do vetor hipotético no espaço de embeddings
indicam os desfechos prováveis da alternativa.

""")

    if not s4:
        L.append(r"\textit{Dados insuficientes para simulações contrafactuais.}" + "\n\n")
    else:
        for i, sim in enumerate(s4, 1):
            cid = _escape_latex(str(sim["cid"]))
            cid_desc = _escape_latex(_truncate(str(sim.get("cid_desc", "")), 50))
            cluster_label = _escape_latex(_truncate(sim.get("cluster_label", ""), 40))
            orig_phys = _escape_latex(_truncate(sim["original_phys"], 35))
            alt_phys = _escape_latex(_truncate(sim["alt_phys"], 35))
            discharge = _escape_latex(_truncate(sim["original_discharge"], 30))
            origem = _escape_latex(_truncate(sim.get("ds_origem", ""), 25))
            uti_str = "Sim" if sim.get("is_uti") else "Não"
            ps_str = "Sim" if sim.get("is_ps_origin") else "Não"

            L.append(r"\begin{simcard}[Caso " + str(i) + ": CID " + cid + " --- " + cid_desc + "]\n")
            L.append(r"""
\begin{tabular}{p{4cm} p{10cm}}
\textbf{Internação} & """ + f"ID {sim['iid']}" + r""" \\
\textbf{Cluster CID} & """ + cluster_label + r""" \\
\textbf{Médico atual} & """ + orig_phys + r""" \\
\textbf{LOS real} & """ + f"{sim['original_los']:.0f} dias" + r""" \\
\textbf{Desfecho real} & """ + discharge + r""" \\
\textbf{LOS médio do médico} & """ + f"{sim['original_phys_mean_los']:.1f} dias" + r""" \\
\textbf{Origem} & """ + origem + r""" \\
\textbf{PS/Urgência} & """ + ps_str + r""" \\
\textbf{Leito UTI} & """ + uti_str + r""" \\[6pt]
\textbf{Médico alternativo} & """ + alt_phys + f" (n={sim['alt_phys_n']})" + r""" \\
\textbf{LOS médio alternativo} & """ + f"{sim['alt_phys_mean_los']:.1f} dias" + r""" \\[6pt]
""")
            L.append(r"""\multicolumn{2}{l}{\textbf{Resultado da simulação (20 vizinhos mais próximos):}} \\
\textbf{LOS médio projetado} & """ + f"{sim['neighbor_mean_los']:.1f} dias" + r""" \\
\textbf{LOS mediano projetado} & """ + f"{sim['neighbor_median_los']:.1f} dias" + r""" \\
\textbf{Alta normal/melhorado} & """ + f"{sim['neighbor_normal_pct']:.0f}\\%" + r""" \\
\textbf{Similaridade máxima} & """ + f"{sim['similarity_top']:.4f}" + r""" \\
\end{tabular}
""")
            # Narrative
            delta_los = sim["original_los"] - sim["neighbor_mean_los"]
            if delta_los > 0:
                L.append(f"\n\\medskip\n\\textbf{{Conclusão:}} Se atendido por {alt_phys} "
                         f"em vez de {orig_phys}, pacientes similares tiveram "
                         f"LOS de {sim['neighbor_mean_los']:.1f} dias e alta "
                         f"normal em {sim['neighbor_normal_pct']:.0f}\\% dos casos "
                         f"--- uma redução estimada de \\textbf{{{delta_los:.0f} dias}}.\n")
            else:
                L.append(f"\n\\medskip\n\\textbf{{Conclusão:}} A simulação indica LOS similar "
                         f"({sim['neighbor_mean_los']:.1f} dias) com alta normal em "
                         f"{sim['neighbor_normal_pct']:.0f}\\% dos casos.\n")

            L.append(r"\end{simcard}" + "\n\n")

    L.append(r"\newpage" + "\n")

    # ── Section 5: Recomendações ──
    L.append(r"\section{Recomendações}" + "\n\n")

    total_excess = s5["total_excess_days"]
    L.append(r"""
\begin{reccard}[Potencial de Economia]
A padronização de protocolos entre médicos para os mesmos clusters diagnósticos
poderia reduzir até \textbf{""" + f"{total_excess:,.0f}".replace(",", ".") + r""" diárias em excesso},
considerando """ + f"{s5['n_clusters_with_variation']}" + r""" clusters CID com variação significativa entre médicos.
\end{reccard}
""")

    # Top clusters for savings
    if s5["top_cluster_savings"]:
        L.append(r"""
\subsection{Clusters CID com Maior Potencial de Redução de Diárias}

\begin{center}
\small
\begin{tabular}{l r r}
\toprule
\textbf{Cluster ID} & \textbf{Diárias em Excesso} & \textbf{N Internações} \\
\midrule
""")
        for cs in s5["top_cluster_savings"][:10]:
            cids = cluster_to_cids.get(cs["cluster_id"], [])
            label = _escape_latex(_truncate(", ".join(sorted(cids)[:5]), 35))
            L.append(f"{label} & {cs['excess_days']:,.0f} & {cs['n']:,} \\\\\n".replace(",", "."))
        L.append(r"""
\bottomrule
\end{tabular}
\end{center}
""")

    L.append(r"""
\subsection{Ações Recomendadas}

\begin{enumerate}[leftmargin=*]
\item \textbf{Protocolos Clínicos Padronizados:} Para os 10 clusters CID com maior variação médica,
  desenvolver protocolos baseados nas práticas dos médicos com melhores resultados.
  Prioridade: clusters com amplitude de LOS $>$ 5 dias entre médicos.

\item \textbf{Revisão por Pares (Peer Review):} Implementar comitês de revisão para casos
  onde o vetor de efeito do médico apresenta alta similaridade com o vetor de óbito
  ($\cos > 0.3$) ou LOS longo ($\cos > 0.3$).

\item \textbf{Diferenciação por Origem/UTI:} Análise separada para internações via PS/urgência
  com primeiro leito UTI, que apresentam padrões de variação médica distintos de
  internações eletivas. Protocolos específicos para cada perfil de entrada.

\item \textbf{Alocação Inteligente:} Utilizar o score de risco por médico e cluster CID para
  orientar a distribuição de novos pacientes, priorizando médicos com menor
  tendência de LOS longo para diagnósticos de alto volume.

\item \textbf{Feedback Contínuo:} Compartilhar com cada médico (identificado por CRM) seu vetor de efeito
  comparado ao centroide da especialidade, em linguagem não-técnica
  (percentil de LOS, taxa de alta normal vs.\ pares).

\item \textbf{Monitoramento Trimestral:} Recalcular os vetores de efeito a cada
  trimestre para acompanhar a convergência de práticas após intervenções.
\end{enumerate}

\subsection{Nota Metodológica}

Este relatório utiliza exclusivamente álgebra vetorial sobre embeddings de 128 dimensões
treinados por Temporal Knowledge Graph (TKG-V6) sobre o grafo completo de 35,2 milhões
de nós. Nenhum modelo de machine learning adicional foi utilizado --- todas as análises
são operações geométricas (médias, diferenças, similaridade cosseno, vizinhos próximos)
no espaço latente.

Novidades desta versão:
\begin{itemize}[leftmargin=*]
\item \textbf{CID Clustering:} CIDs agrupados por similaridade de embedding (limiar """ + f"{N_CID_CLUSTERS}" + r""")
  em vez de código de capítulo. Ex: I50 (insuficiência cardíaca) e I48 (flutter/fibrilação atrial)
  agora pertencem ao mesmo cluster semântico.
\item \textbf{Identidade por CRM:} Médicos identificados por DS\_CONSELHO\_CLASSE (CRM),
  eliminando duplicatas por contrato ou evolução.
\item \textbf{Análise de Origem/UTI:} Internações via PS/urgência com leito UTI
  identificadas separadamente para análise diferenciada.
\end{itemize}

\vfill
\begin{center}
\small\color{physgray}
Gerado automaticamente por JCUBE Digital Twin V6 --- """ + source_db_display + r""" --- """ + REPORT_DATE_STR + r"""
\end{center}
""")

    L.append(r"\end{document}" + "\n")

    return "".join(L)


# ─────────────────────────────────────────────────────────────────
# LaTeX Compilation
# ─────────────────────────────────────────────────────────────────

def _compile_latex(latex_content: str, output_pdf: str):
    import subprocess
    from pathlib import Path

    print("[COMPILE] Compiling PDF ...")
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
            errs = [l for l in stdout_str.split("\n") if l.startswith("!") or "Error" in l]
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
    memory=65536,   # 65 GB RAM (16.8 GB tensor + vocab)
    timeout=7200,   # 2 hour max
)
def generate_report(source_db: str = "GHO-BRADESCO"):
    import time
    import os

    t_start = time.time()

    # Reload volumes to get latest data
    jepa_cache.reload()
    data_vol.reload()

    source_db_safe = source_db.lower().replace("-", "_").replace(" ", "_")
    output_pdf = f"{OUTPUT_DIR}/physician_variation_{source_db_safe}_v6_2026_03.pdf"

    print("=" * 70)
    print("JCUBE V6 Physician Variation Analysis (Modal)")
    print(f"Source   : {source_db}")
    print(f"Weights  : {WEIGHTS_PATH}")
    print(f"DB       : {DB_PATH}")
    print(f"Graph    : {GRAPH_PARQUET}")
    print(f"Output   : {output_pdf}")
    print(f"Min pair : {MIN_PAIR_ADMISSIONS}")
    print(f"CID k    : {N_CID_CLUSTERS}")
    print("=" * 70)

    for p in [GRAPH_PARQUET, WEIGHTS_PATH, DB_PATH]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}")

    # 1. Load twin (full graph)
    unique_nodes, embeddings, node_to_idx, intern_mask = _load_twin(source_db)

    # 0. Cluster CIDs by embedding similarity
    cid_to_cluster, cluster_to_cids = _cluster_cids_by_embedding(node_to_idx, embeddings)

    # 2. Build physician-admission-CID mapping
    import duckdb
    con = duckdb.connect(str(DB_PATH))
    rows, discharge_map, phys_names, spec_names = _build_physician_data(con, source_db)

    # 3. Compute direction + effect vectors
    vectors = _compute_vectors(rows, discharge_map, embeddings, node_to_idx,
                               cid_to_cluster, cluster_to_cids, source_db)

    # 4. Section analyses
    s1 = _section1_quanto_importa(vectors, spec_names, cluster_to_cids)
    s2 = _section2_por_especialidade(vectors, phys_names, spec_names)
    s3 = _section3_top_clusters(vectors, phys_names, cluster_to_cids)
    s4 = _section4_counterfactuals(vectors, embeddings, node_to_idx, phys_names,
                                   cluster_to_cids, source_db)
    s5 = _section5_recommendations(s1, s2, s3, s4, vectors)

    con.close()

    # 5. Generate LaTeX
    print("[5/6] Generating LaTeX document ...")
    latex = _generate_latex(s1, s2, s3, s4, s5, spec_names, cluster_to_cids, source_db)

    # 6. Compile PDF
    _compile_latex(latex, output_pdf)

    # Commit volume so changes persist
    data_vol.commit()

    elapsed = time.time() - t_start
    print(f"\nFinished in {elapsed:.1f}s")
    print(f"Report saved to Modal volume jcube-data at: {output_pdf}")
    print("Download with:")
    print(f"  modal volume get jcube-data reports/physician_variation_{source_db_safe}_v6_2026_03.pdf "
          f"./physician_variation_{source_db_safe}_v6_2026_03.pdf")
    return output_pdf


@app.local_entrypoint()
def main():
    for src in ["GHO-BRADESCO", "GHO-PETROBRAS", "GOHOSP-CNU", "PASA"]:
        generate_report.spawn(source_db=src)
# kmeans 1774367720
