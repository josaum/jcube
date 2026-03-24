#!/usr/bin/env python3
"""
Modal script: JCUBE V6 Bradesco Executive Report
Comprehensive healthcare analytics report for GHO-BRADESCO executives.

Architecture:
  1. Load V6 BRADESCO-specific embeddings (2.86M nodes x 128 dim, 4 epochs TGN)
  2. Query DuckDB filtered by source_db = 'GHO-BRADESCO'
  3. Compute anomalies, readmissions, LOS analysis, financial metrics
  4. HDBSCAN clustering on admission embeddings
  5. Embedding algebra: similar patients, what-if, risk clusters
  6. Generate LaTeX -> PDF (pt-BR, professional, executive tone)

Sections:
  1. Capa e Resumo Executivo — KPIs, headline finding
  2. Perfil Operacional — trends, top CIDs, TUSS, LOS distribution
  3. Deteccao de Anomalias Prioritarias — top 20 with narrative
  4. Analise de Permanencia (LOS) — by CID, tipo alta, excess days
  5. Analise Financeira — billing trends, glosas, recovery estimate
  6. Readmissoes — 30-day rate, frequent flyers, avoidable cost
  7. Simulacao por Algebra de Embeddings — similar patients, clusters
  8. Recomendacoes Estrategicas — prioritized actions with ROI

Output: /data/reports/bradesco_executive_v6_2026_03.pdf

Usage:
    modal run reports/modal_bradesco_executive.py
    modal run --detach reports/modal_bradesco_executive.py
"""
from __future__ import annotations

import modal

# ─────────────────────────────────────────────────────────────────
# Modal App + Volumes
# ─────────────────────────────────────────────────────────────────

app = modal.App("jcube-bradesco-executive")

jepa_cache = modal.Volume.from_name("jepa-cache", create_if_missing=False)
data_vol   = modal.Volume.from_name("jcube-data",  create_if_missing=False)

VOLUMES = {
    "/cache": jepa_cache,
    "/data":  data_vol,
}

# ─────────────────────────────────────────────────────────────────
# Container image — CPU + torch, duckdb, sklearn, hdbscan, LaTeX
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
        "hdbscan>=0.8.33",
    )
)

# ─────────────────────────────────────────────────────────────────
# Paths inside the container
# ─────────────────────────────────────────────────────────────────

GRAPH_PARQUET = "/data/jcube_graph_v6.parquet"
WEIGHTS_PATH  = "/cache/tkg-v6/node_emb_epoch_2.pt"
DB_PATH       = "/data/aggregated_fixed_union.db"
OUTPUT_DIR    = "/data/reports"
OUTPUT_PDF    = f"{OUTPUT_DIR}/bradesco_executive_v6_2026_03.pdf"

REPORT_DATE_STR = "2026-03-24"
SOURCE_DB       = "GHO-BRADESCO"

Z_THRESHOLD        = 2.0
HDBSCAN_MIN_CLUSTER = 80
HDBSCAN_MIN_SAMPLES = 10
TOP_ANOMALIES       = 20

# ─────────────────────────────────────────────────────────────────
# Helpers (all run inside container)
# ─────────────────────────────────────────────────────────────────

def _fmt_date(d) -> str:
    if d is None:
        return "---"
    try:
        if isinstance(d, str):
            return d[:10]
        return d.strftime("%d/%m/%Y")
    except Exception:
        return str(d)[:10]


def _safe_int(v, default=0) -> int:
    try:
        return int(v) if v is not None else default
    except Exception:
        return default


def _safe_float(v, default=0.0) -> float:
    try:
        return float(v) if v is not None else default
    except Exception:
        return default


def _escape_latex(s: str) -> str:
    """Escape LaTeX special characters while preserving UTF-8 accents."""
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
    """Format as R$ for LaTeX (escaped $)."""
    f = _safe_float(v)
    if f == 0:
        return "---"
    return "R\\$ {:,.2f}".format(f).replace(",", "X").replace(".", ",").replace("X", ".")


def _brl_millions(v) -> str:
    """Format as R$ X,X mi for LaTeX."""
    f = _safe_float(v)
    if f == 0:
        return "---"
    mi = f / 1_000_000
    if mi >= 1:
        return "R\\$ {:,.1f} mi".format(mi).replace(",", "X").replace(".", ",").replace("X", ".")
    return _brl(v)


def _brl_plain(v) -> str:
    f = _safe_float(v)
    if f == 0:
        return "---"
    return "R$ {:,.2f}".format(f).replace(",", "X").replace(".", ",").replace("X", ".")


def _pct(num: float, den: float, decimals: int = 1) -> str:
    if den == 0:
        return "---"
    return f"{100.0 * num / den:.{decimals}f}\\%"


def _pct_plain(num: float, den: float, decimals: int = 1) -> str:
    if den == 0:
        return "---"
    return f"{100.0 * num / den:.{decimals}f}%"


def _truncate(s: str, max_len: int = 100) -> str:
    if not s:
        return "---"
    s = str(s).strip()
    if len(s) > max_len:
        return s[:max_len] + "..."
    return s


def _exec(con, q: str):
    cur  = con.execute(q)
    cols = [d[0] for d in cur.description]
    return cols, cur.fetchall()


def _rows_to_dicts(cols, rows):
    return [dict(zip(cols, r)) for r in rows]


# ─────────────────────────────────────────────────────────────────
# Step 1 — Load V6 embeddings (BRADESCO-only subgraph)
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
    # Build BRADESCO-filtered node list from parquet
    brad_nodes_set = set(
        pc.unique(pa.chunked_array(subj.chunks + obj.chunks)).to_pylist()
    )
    del table, subj, obj
    print(f"    {len(brad_nodes_set):,} BRADESCO nodes in {time.time()-t0:.1f}s")

    # Load FULL graph vocab (needed to index into full-graph embeddings)
    print("[1/8] Loading full graph vocab for embedding lookup ...")
    t1 = time.time()
    full_table = pq.read_table(GRAPH_PARQUET, columns=["subject_id", "object_id"])
    full_subj = full_table.column("subject_id")
    full_obj = full_table.column("object_id")
    full_all = pa.chunked_array(full_subj.chunks + full_obj.chunks)
    unique_nodes = pc.unique(full_all).to_numpy(zero_copy_only=False).astype(object)
    del full_table, full_subj, full_obj, full_all
    n_nodes = len(unique_nodes)
    print(f"    {n_nodes:,} total nodes in {time.time()-t1:.1f}s")

    print("[1/8] Loading V6 embedding weights ...")
    t2 = time.time()
    state = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
    if isinstance(state, torch.Tensor):
        embeddings = state.numpy().astype(np.float32)
    elif isinstance(state, dict) and "weight" in state:
        embeddings = state["weight"].numpy().astype(np.float32)
    else:
        embeddings = list(state.values())[0].numpy().astype(np.float32)
    print(f"    Embeddings shape: {embeddings.shape} in {time.time()-t2:.1f}s")

    if n_nodes != embeddings.shape[0]:
        raise ValueError(
            f"Vocab size mismatch: {n_nodes:,} nodes vs "
            f"{embeddings.shape[0]:,} embedding rows"
        )

    node_to_idx = {str(n): i for i, n in enumerate(unique_nodes)}

    # Build masks for BRADESCO nodes
    bradesco_internacao_mask = np.array(
        [f"{SOURCE_DB}/ID_CD_INTERNACAO_" in str(n) for n in unique_nodes], dtype=bool
    )
    bradesco_paciente_mask = np.array(
        [f"{SOURCE_DB}/ID_CD_PACIENTE_" in str(n) or
         (SOURCE_DB in str(n) and "_PACIENTE_" in str(n))
         for n in unique_nodes], dtype=bool
    )

    print(f"    BRADESCO INTERNACAO nodes: {bradesco_internacao_mask.sum():,}")
    print(f"    BRADESCO PACIENTE nodes:   {bradesco_paciente_mask.sum():,}")

    return unique_nodes, embeddings, node_to_idx, bradesco_internacao_mask, bradesco_paciente_mask


# ─────────────────────────────────────────────────────────────────
# Step 2 — KPIs and Operational Profile from DuckDB
# ─────────────────────────────────────────────────────────────────

def _fetch_kpis(con):
    """Fetch headline KPIs for the executive summary."""
    import time
    print("[2/8] Fetching executive KPIs ...")
    t0 = time.time()

    SRC_FILTER = f"source_db = '{SOURCE_DB}'"

    # Total internacoes, pacientes, periodo
    cols, rows = _exec(con, f"""
        SELECT
            COUNT(DISTINCT ID_CD_INTERNACAO)   AS total_internacoes,
            COUNT(DISTINCT ID_CD_PACIENTE)     AS total_pacientes,
            MIN(DH_ADMISSAO_HOSP)::DATE        AS dt_inicio,
            MAX(DH_ADMISSAO_HOSP)::DATE        AS dt_fim,
            AVG(DATEDIFF('day', DH_ADMISSAO_HOSP::DATE,
                COALESCE(DH_FINALIZACAO, CURRENT_TIMESTAMP)::DATE)) AS avg_los,
            MEDIAN(DATEDIFF('day', DH_ADMISSAO_HOSP::DATE,
                COALESCE(DH_FINALIZACAO, CURRENT_TIMESTAMP)::DATE)) AS median_los
        FROM agg_tb_capta_internacao_cain
        WHERE {SRC_FILTER}
    """)
    kpi_base = dict(zip(cols, rows[0])) if rows else {}

    # Total billing, glosa
    cols, rows = _exec(con, f"""
        SELECT
            SUM(VL_TOTAL)                   AS total_faturado,
            SUM(VL_GLOSA_FECHAMENTO)        AS total_glosado,
            SUM(VL_LIQUIDO_FAT)             AS total_liquido,
            COUNT(DISTINCT ID_CD_INTERNACAO) AS n_faturas_internacoes
        FROM agg_tb_fatura_fatu
        WHERE {SRC_FILTER}
    """)
    kpi_billing = dict(zip(cols, rows[0])) if rows else {}

    # Readmission 30d rate
    cols, rows = _exec(con, f"""
        WITH ordered_adm AS (
            SELECT ID_CD_PACIENTE, ID_CD_INTERNACAO,
                   DH_ADMISSAO_HOSP::DATE AS dt_adm,
                   LAG(DH_FINALIZACAO::DATE) OVER (
                       PARTITION BY ID_CD_PACIENTE
                       ORDER BY DH_ADMISSAO_HOSP
                   ) AS prev_discharge
            FROM agg_tb_capta_internacao_cain
            WHERE {SRC_FILTER}
              AND DH_ADMISSAO_HOSP IS NOT NULL
        )
        SELECT
            COUNT(*) FILTER (WHERE DATEDIFF('day', prev_discharge, dt_adm) BETWEEN 1 AND 30) AS readmissions_30d,
            COUNT(*) FILTER (WHERE prev_discharge IS NOT NULL) AS total_with_prior
        FROM ordered_adm
    """)
    kpi_readmit = dict(zip(cols, rows[0])) if rows else {}

    # Mortality rate (from discharge type)
    cols, rows = _exec(con, f"""
        WITH last_status AS (
            SELECT es.ID_CD_INTERNACAO,
                   ROW_NUMBER() OVER (
                       PARTITION BY es.ID_CD_INTERNACAO
                       ORDER BY es.DH_CADASTRO DESC
                   ) AS rn,
                   es.FL_DESOSPITALIZACAO
            FROM agg_tb_capta_evo_status_caes es
            WHERE es.{SRC_FILTER}
        ),
        discharge AS (
            SELECT ls.ID_CD_INTERNACAO,
                   f.DS_FINAL_MONITORAMENTO AS tipo_alta
            FROM last_status ls
            JOIN agg_tb_capta_tipo_final_monit_fmon f
                ON ls.FL_DESOSPITALIZACAO = f.ID_CD_FINAL_MONITORAMENTO
            WHERE ls.rn = 1
        )
        SELECT
            COUNT(*) AS total_discharged,
            COUNT(*) FILTER (WHERE UPPER(tipo_alta) LIKE '%%OBITO%%'
                             OR UPPER(tipo_alta) LIKE '%%ÓBITO%%') AS n_obito
        FROM discharge d
        JOIN agg_tb_capta_internacao_cain i
            ON d.ID_CD_INTERNACAO = i.ID_CD_INTERNACAO
            AND i.{SRC_FILTER}
    """)
    kpi_mortality = dict(zip(cols, rows[0])) if rows else {}

    kpis = {**kpi_base, **kpi_billing, **kpi_readmit, **kpi_mortality}
    print(f"    KPIs fetched in {time.time()-t0:.1f}s")
    return kpis


def _fetch_operational_profile(con):
    """Fetch trend data, top CIDs, TUSS, LOS distribution, tipo alta."""
    import time
    print("[3/8] Fetching operational profile ...")
    t0 = time.time()

    SRC = f"source_db = '{SOURCE_DB}'"
    profile = {}

    # Monthly admissions trend
    cols, rows = _exec(con, f"""
        SELECT DATE_TRUNC('month', DH_ADMISSAO_HOSP)::DATE AS mes,
               COUNT(DISTINCT ID_CD_INTERNACAO) AS n_internacoes,
               COUNT(DISTINCT ID_CD_PACIENTE)   AS n_pacientes
        FROM agg_tb_capta_internacao_cain
        WHERE {SRC} AND DH_ADMISSAO_HOSP IS NOT NULL
        GROUP BY mes
        ORDER BY mes
    """)
    profile["monthly_admissions"] = _rows_to_dicts(cols, rows)

    # Top 10 CIDs by volume
    cols, rows = _exec(con, f"""
        SELECT c.DS_DESCRICAO AS cid_desc,
               COUNT(DISTINCT c.ID_CD_INTERNACAO) AS n_internacoes
        FROM agg_tb_capta_cid_caci c
        WHERE c.{SRC} AND c.DS_DESCRICAO IS NOT NULL
        GROUP BY c.DS_DESCRICAO
        ORDER BY n_internacoes DESC
        LIMIT 10
    """)
    profile["top_cids_volume"] = _rows_to_dicts(cols, rows)

    # Top 10 CIDs by cost
    cols, rows = _exec(con, f"""
        SELECT c.DS_DESCRICAO AS cid_desc,
               SUM(f.VL_TOTAL) AS total_faturado,
               COUNT(DISTINCT c.ID_CD_INTERNACAO) AS n_internacoes
        FROM agg_tb_capta_cid_caci c
        JOIN agg_tb_fatura_fatu f
            ON c.ID_CD_INTERNACAO = f.ID_CD_INTERNACAO
            AND c.source_db = f.source_db
        WHERE c.{SRC} AND c.DS_DESCRICAO IS NOT NULL
        GROUP BY c.DS_DESCRICAO
        ORDER BY total_faturado DESC
        LIMIT 10
    """)
    profile["top_cids_cost"] = _rows_to_dicts(cols, rows)

    # Top 10 procedures (TUSS) by volume
    cols, rows = _exec(con, f"""
        SELECT COALESCE(p.DS_DESCRICAO, CAST(p.CD_PROCEDIMENTO AS VARCHAR)) AS proc_desc,
               COUNT(*) AS n_procedimentos
        FROM agg_tb_fatura_procedimentos_fapr p
        WHERE p.{SRC}
        GROUP BY proc_desc
        ORDER BY n_procedimentos DESC
        LIMIT 10
    """)
    profile["top_procedures"] = _rows_to_dicts(cols, rows)

    # LOS distribution buckets
    cols, rows = _exec(con, f"""
        SELECT
            CASE
                WHEN los <= 1  THEN '0-1 dia'
                WHEN los <= 3  THEN '2-3 dias'
                WHEN los <= 7  THEN '4-7 dias'
                WHEN los <= 14 THEN '8-14 dias'
                WHEN los <= 30 THEN '15-30 dias'
                ELSE '>30 dias'
            END AS faixa_los,
            COUNT(*) AS n_internacoes,
            CASE
                WHEN los <= 1  THEN 1
                WHEN los <= 3  THEN 2
                WHEN los <= 7  THEN 3
                WHEN los <= 14 THEN 4
                WHEN los <= 30 THEN 5
                ELSE 6
            END AS sort_key
        FROM (
            SELECT DATEDIFF('day', DH_ADMISSAO_HOSP::DATE,
                   COALESCE(DH_FINALIZACAO, CURRENT_TIMESTAMP)::DATE) AS los
            FROM agg_tb_capta_internacao_cain
            WHERE {SRC} AND DH_ADMISSAO_HOSP IS NOT NULL
        ) sub
        GROUP BY faixa_los, sort_key
        ORDER BY sort_key
    """)
    profile["los_distribution"] = _rows_to_dicts(cols, rows)

    # Discharge type distribution
    cols, rows = _exec(con, f"""
        WITH last_status AS (
            SELECT es.ID_CD_INTERNACAO,
                   ROW_NUMBER() OVER (
                       PARTITION BY es.ID_CD_INTERNACAO
                       ORDER BY es.DH_CADASTRO DESC
                   ) AS rn,
                   es.FL_DESOSPITALIZACAO
            FROM agg_tb_capta_evo_status_caes es
            WHERE es.{SRC}
        )
        SELECT f.DS_FINAL_MONITORAMENTO AS tipo_alta,
               COUNT(DISTINCT ls.ID_CD_INTERNACAO) AS n_internacoes
        FROM last_status ls
        JOIN agg_tb_capta_tipo_final_monit_fmon f
            ON ls.FL_DESOSPITALIZACAO = f.ID_CD_FINAL_MONITORAMENTO
        WHERE ls.rn = 1
        GROUP BY tipo_alta
        ORDER BY n_internacoes DESC
    """)
    profile["discharge_types"] = _rows_to_dicts(cols, rows)

    print(f"    Operational profile fetched in {time.time()-t0:.1f}s")
    return profile


# ─────────────────────────────────────────────────────────────────
# Step 3 — Anomaly Detection
# ─────────────────────────────────────────────────────────────────

def _detect_anomalies(embeddings, internacao_mask, unique_nodes):
    import time
    import numpy as np

    print("[4/8] Computing anomaly z-scores (BRADESCO) ...")
    t0 = time.time()

    vecs  = embeddings[internacao_mask]
    names = unique_nodes[internacao_mask]

    centroid = vecs.mean(axis=0)
    dists    = np.linalg.norm(vecs - centroid, axis=1)
    mean_d   = dists.mean()
    std_d    = dists.std()
    z_scores = (dists - mean_d) / (std_d + 1e-9)

    # Global z-map for all BRADESCO internacoes
    global_z_map = {}
    for i, name in enumerate(names):
        s = str(name)
        try:
            iid = int(s.split("ID_CD_INTERNACAO_")[1])
            global_z_map[iid] = float(z_scores[i])
        except Exception:
            pass

    anomaly_mask  = z_scores > Z_THRESHOLD
    anomaly_names = names[anomaly_mask]
    anomaly_z     = z_scores[anomaly_mask]
    order         = np.argsort(-anomaly_z)
    anomaly_names = anomaly_names[order]
    anomaly_z     = anomaly_z[order]

    total_anomalies = int(anomaly_mask.sum())
    print(f"    {total_anomalies:,} anomalies (z>{Z_THRESHOLD})")
    print(f"    Top z-score: {anomaly_z[0]:.2f}" if len(anomaly_z) > 0 else "    No anomalies")

    # Extract IDs for top N
    top_ids = []
    for n in anomaly_names[:TOP_ANOMALIES]:
        s = str(n)
        try:
            iid = int(s.split("ID_CD_INTERNACAO_")[1])
            top_ids.append(iid)
        except Exception:
            pass

    print(f"    Done in {time.time()-t0:.1f}s")
    return top_ids, anomaly_z[:TOP_ANOMALIES], global_z_map, total_anomalies, vecs, names


def _fetch_anomaly_details(con, top_ids, anomaly_z_arr):
    """Fetch full details for top anomalies from DuckDB."""
    import time
    print(f"    Fetching details for {len(top_ids)} top anomalies ...")
    t0 = time.time()

    SRC = f"source_db = '{SOURCE_DB}'"

    if not top_ids:
        return []

    # Insert into temp table
    con.execute("""
        CREATE OR REPLACE TEMP TABLE tmp_anom_ids (iid INTEGER)
    """)
    vals = ", ".join(f"({iid})" for iid in top_ids)
    con.execute(f"INSERT INTO tmp_anom_ids VALUES {vals}")

    z_map = {iid: float(z) for iid, z in zip(top_ids, anomaly_z_arr)}

    # Core admission data
    cols, rows = _exec(con, f"""
        SELECT i.ID_CD_INTERNACAO, i.ID_CD_PACIENTE,
            i.DH_ADMISSAO_HOSP, i.DH_FINALIZACAO,
            DATEDIFF('day', i.DH_ADMISSAO_HOSP::DATE,
                COALESCE(i.DH_FINALIZACAO, CURRENT_TIMESTAMP)::DATE) AS los_dias,
            i.ID_CD_HOSPITAL, i.IN_SITUACAO
        FROM agg_tb_capta_internacao_cain i
        JOIN tmp_anom_ids t ON i.ID_CD_INTERNACAO = t.iid
        WHERE i.{SRC}
    """)
    admissions = _rows_to_dicts(cols, rows)
    for a in admissions:
        a["z_score"] = z_map.get(a["ID_CD_INTERNACAO"], 2.01)

    # Billing
    cols, rows = _exec(con, f"""
        SELECT f.ID_CD_INTERNACAO,
            SUM(f.VL_TOTAL) AS vl_total,
            SUM(f.VL_GLOSA_FECHAMENTO) AS vl_glosa,
            SUM(f.VL_LIQUIDO_FAT) AS vl_liquido
        FROM agg_tb_fatura_fatu f
        JOIN tmp_anom_ids t ON f.ID_CD_INTERNACAO = t.iid
        WHERE f.{SRC}
        GROUP BY f.ID_CD_INTERNACAO
    """)
    bill_map = {d["ID_CD_INTERNACAO"]: d for d in _rows_to_dicts(cols, rows)}

    # CIDs
    cols, rows = _exec(con, f"""
        SELECT c.ID_CD_INTERNACAO,
            STRING_AGG(DISTINCT COALESCE(c.DS_DESCRICAO, '?'), ' | ') AS cids
        FROM agg_tb_capta_cid_caci c
        JOIN tmp_anom_ids t ON c.ID_CD_INTERNACAO = t.iid
        WHERE c.{SRC}
        GROUP BY c.ID_CD_INTERNACAO
    """)
    cid_map = {d["ID_CD_INTERNACAO"]: d for d in _rows_to_dicts(cols, rows)}

    # Procedures
    try:
        cols, rows = _exec(con, f"""
            SELECT p.ID_CD_INTERNACAO,
                COUNT(*) AS n_procedimentos
            FROM agg_tb_fatura_procedimentos_fapr p
            JOIN tmp_anom_ids t ON p.ID_CD_INTERNACAO = t.iid
            WHERE p.{SRC}
            GROUP BY p.ID_CD_INTERNACAO
        """)
        proc_map = {d["ID_CD_INTERNACAO"]: d for d in _rows_to_dicts(cols, rows)}
    except Exception:
        proc_map = {}

    # Glosas
    try:
        cols, rows = _exec(con, f"""
            SELECT g.ID_CD_INTERNACAO,
                COUNT(*) AS n_glosas,
                SUM(g.VL_GLOSADO) AS vl_glosado
            FROM agg_tb_fatura_glosa_fatg g
            JOIN tmp_anom_ids t ON g.ID_CD_INTERNACAO = t.iid
            WHERE g.{SRC}
            GROUP BY g.ID_CD_INTERNACAO
        """)
        glosa_map = {d["ID_CD_INTERNACAO"]: d for d in _rows_to_dicts(cols, rows)}
    except Exception:
        glosa_map = {}

    # Hospital baseline
    cols, rows = _exec(con, f"""
        SELECT
            AVG(DATEDIFF('day', DH_ADMISSAO_HOSP::DATE,
                COALESCE(DH_FINALIZACAO, CURRENT_TIMESTAMP)::DATE)) AS baseline_los,
            AVG(f.vl_total) AS baseline_billing
        FROM agg_tb_capta_internacao_cain i
        LEFT JOIN (
            SELECT ID_CD_INTERNACAO, source_db, SUM(VL_TOTAL) AS vl_total
            FROM agg_tb_fatura_fatu WHERE {SRC}
            GROUP BY ID_CD_INTERNACAO, source_db
        ) f ON i.ID_CD_INTERNACAO = f.ID_CD_INTERNACAO AND i.source_db = f.source_db
        WHERE i.{SRC}
    """)
    baseline = dict(zip(cols, rows[0])) if rows else {}

    # Merge
    for a in admissions:
        iid = a["ID_CD_INTERNACAO"]
        a["billing"]   = bill_map.get(iid, {})
        a["cids"]      = cid_map.get(iid, {}).get("cids", "---")
        a["n_procs"]   = _safe_int(proc_map.get(iid, {}).get("n_procedimentos"))
        a["glosa"]     = glosa_map.get(iid, {})
        a["baseline"]  = baseline

    # Sort by z-score descending
    admissions.sort(key=lambda x: -x.get("z_score", 0))

    con.execute("DROP TABLE IF EXISTS tmp_anom_ids")
    print(f"    Done in {time.time()-t0:.1f}s")
    return admissions


# ─────────────────────────────────────────────────────────────────
# Step 4 — LOS Analysis
# ─────────────────────────────────────────────────────────────────

def _fetch_los_analysis(con):
    """LOS by CID, by discharge type, excess days."""
    import time
    print("[5/8] Fetching LOS analysis ...")
    t0 = time.time()
    SRC = f"source_db = '{SOURCE_DB}'"
    los_data = {}

    # LOS by CID group (top 10)
    cols, rows = _exec(con, f"""
        SELECT c.DS_DESCRICAO AS cid_desc,
            COUNT(DISTINCT i.ID_CD_INTERNACAO) AS n_internacoes,
            AVG(DATEDIFF('day', i.DH_ADMISSAO_HOSP::DATE,
                COALESCE(i.DH_FINALIZACAO, CURRENT_TIMESTAMP)::DATE)) AS avg_los,
            MEDIAN(DATEDIFF('day', i.DH_ADMISSAO_HOSP::DATE,
                COALESCE(i.DH_FINALIZACAO, CURRENT_TIMESTAMP)::DATE)) AS median_los,
            MAX(DATEDIFF('day', i.DH_ADMISSAO_HOSP::DATE,
                COALESCE(i.DH_FINALIZACAO, CURRENT_TIMESTAMP)::DATE)) AS max_los
        FROM agg_tb_capta_internacao_cain i
        JOIN agg_tb_capta_cid_caci c
            ON i.ID_CD_INTERNACAO = c.ID_CD_INTERNACAO
            AND i.source_db = c.source_db
        WHERE i.{SRC} AND c.DS_DESCRICAO IS NOT NULL
          AND i.DH_ADMISSAO_HOSP IS NOT NULL
        GROUP BY c.DS_DESCRICAO
        HAVING COUNT(DISTINCT i.ID_CD_INTERNACAO) >= 10
        ORDER BY avg_los DESC
        LIMIT 10
    """)
    los_data["by_cid"] = _rows_to_dicts(cols, rows)

    # LOS by discharge type
    cols, rows = _exec(con, f"""
        WITH last_status AS (
            SELECT es.ID_CD_INTERNACAO,
                   ROW_NUMBER() OVER (
                       PARTITION BY es.ID_CD_INTERNACAO
                       ORDER BY es.DH_CADASTRO DESC
                   ) AS rn,
                   es.FL_DESOSPITALIZACAO
            FROM agg_tb_capta_evo_status_caes es
            WHERE es.{SRC}
        )
        SELECT f.DS_FINAL_MONITORAMENTO AS tipo_alta,
            COUNT(DISTINCT i.ID_CD_INTERNACAO) AS n_internacoes,
            AVG(DATEDIFF('day', i.DH_ADMISSAO_HOSP::DATE,
                COALESCE(i.DH_FINALIZACAO, CURRENT_TIMESTAMP)::DATE)) AS avg_los,
            MEDIAN(DATEDIFF('day', i.DH_ADMISSAO_HOSP::DATE,
                COALESCE(i.DH_FINALIZACAO, CURRENT_TIMESTAMP)::DATE)) AS median_los
        FROM last_status ls
        JOIN agg_tb_capta_tipo_final_monit_fmon f
            ON ls.FL_DESOSPITALIZACAO = f.ID_CD_FINAL_MONITORAMENTO
        JOIN agg_tb_capta_internacao_cain i
            ON ls.ID_CD_INTERNACAO = i.ID_CD_INTERNACAO AND i.{SRC}
        WHERE ls.rn = 1 AND i.DH_ADMISSAO_HOSP IS NOT NULL
        GROUP BY tipo_alta
        ORDER BY avg_los DESC
    """)
    los_data["by_discharge"] = _rows_to_dicts(cols, rows)

    # Admissions exceeding >2x median LOS for their CID
    cols, rows = _exec(con, f"""
        WITH cid_median AS (
            SELECT c.DS_DESCRICAO AS cid_desc,
                MEDIAN(DATEDIFF('day', i.DH_ADMISSAO_HOSP::DATE,
                    COALESCE(i.DH_FINALIZACAO, CURRENT_TIMESTAMP)::DATE)) AS med_los,
                COUNT(DISTINCT i.ID_CD_INTERNACAO) AS cid_vol
            FROM agg_tb_capta_internacao_cain i
            JOIN agg_tb_capta_cid_caci c
                ON i.ID_CD_INTERNACAO = c.ID_CD_INTERNACAO
                AND i.source_db = c.source_db
            WHERE i.{SRC} AND c.DS_DESCRICAO IS NOT NULL
              AND i.DH_ADMISSAO_HOSP IS NOT NULL
            GROUP BY c.DS_DESCRICAO
            HAVING COUNT(DISTINCT i.ID_CD_INTERNACAO) >= 10
        )
        SELECT cm.cid_desc,
            COUNT(DISTINCT i.ID_CD_INTERNACAO) AS n_excess,
            cm.med_los,
            AVG(DATEDIFF('day', i.DH_ADMISSAO_HOSP::DATE,
                COALESCE(i.DH_FINALIZACAO, CURRENT_TIMESTAMP)::DATE)) AS avg_actual_los,
            SUM(DATEDIFF('day', i.DH_ADMISSAO_HOSP::DATE,
                COALESCE(i.DH_FINALIZACAO, CURRENT_TIMESTAMP)::DATE) - cm.med_los) AS excess_days
        FROM agg_tb_capta_internacao_cain i
        JOIN agg_tb_capta_cid_caci c
            ON i.ID_CD_INTERNACAO = c.ID_CD_INTERNACAO
            AND i.source_db = c.source_db
        JOIN cid_median cm ON c.DS_DESCRICAO = cm.cid_desc
        WHERE i.{SRC}
          AND i.DH_ADMISSAO_HOSP IS NOT NULL
          AND DATEDIFF('day', i.DH_ADMISSAO_HOSP::DATE,
              COALESCE(i.DH_FINALIZACAO, CURRENT_TIMESTAMP)::DATE) > 2 * cm.med_los
        GROUP BY cm.cid_desc, cm.med_los
        ORDER BY excess_days DESC
        LIMIT 10
    """)
    los_data["excess_los"] = _rows_to_dicts(cols, rows)

    # Total excess bed-days
    try:
        cols2, rows2 = _exec(con, f"""
            WITH cid_med AS (
                SELECT c.DS_DESCRICAO AS cid_desc,
                    MEDIAN(DATEDIFF('day', i.DH_ADMISSAO_HOSP::DATE,
                        COALESCE(i.DH_FINALIZACAO, CURRENT_TIMESTAMP)::DATE)) AS med_los
                FROM agg_tb_capta_internacao_cain i
                JOIN agg_tb_capta_cid_caci c
                    ON i.ID_CD_INTERNACAO = c.ID_CD_INTERNACAO
                    AND i.source_db = c.source_db
                WHERE i.{SRC} AND c.DS_DESCRICAO IS NOT NULL
                  AND i.DH_ADMISSAO_HOSP IS NOT NULL
                GROUP BY c.DS_DESCRICAO
                HAVING COUNT(DISTINCT i.ID_CD_INTERNACAO) >= 10
            )
            SELECT
                SUM(DATEDIFF('day', i.DH_ADMISSAO_HOSP::DATE,
                    COALESCE(i.DH_FINALIZACAO, CURRENT_TIMESTAMP)::DATE) - cm.med_los) AS total_excess_days,
                COUNT(DISTINCT i.ID_CD_INTERNACAO) AS n_excess_admissions
            FROM agg_tb_capta_internacao_cain i
            JOIN agg_tb_capta_cid_caci c
                ON i.ID_CD_INTERNACAO = c.ID_CD_INTERNACAO AND i.source_db = c.source_db
            JOIN cid_med cm ON c.DS_DESCRICAO = cm.cid_desc
            WHERE i.{SRC} AND i.DH_ADMISSAO_HOSP IS NOT NULL
              AND DATEDIFF('day', i.DH_ADMISSAO_HOSP::DATE,
                  COALESCE(i.DH_FINALIZACAO, CURRENT_TIMESTAMP)::DATE) > 2 * cm.med_los
        """)
        los_data["excess_totals"] = dict(zip(cols2, rows2[0])) if rows2 else {}
    except Exception:
        los_data["excess_totals"] = {}

    print(f"    LOS analysis done in {time.time()-t0:.1f}s")
    return los_data


# ─────────────────────────────────────────────────────────────────
# Step 5 — Financial Analysis
# ─────────────────────────────────────────────────────────────────

def _fetch_financial_analysis(con):
    """Monthly billing trends, glosa analysis, recovery estimates."""
    import time
    print("[6/8] Fetching financial analysis ...")
    t0 = time.time()
    SRC = f"source_db = '{SOURCE_DB}'"
    fin = {}

    # Monthly billing trend
    cols, rows = _exec(con, f"""
        SELECT DATE_TRUNC('month', i.DH_ADMISSAO_HOSP)::DATE AS mes,
            SUM(f.VL_TOTAL) AS total_faturado,
            SUM(f.VL_GLOSA_FECHAMENTO) AS total_glosado,
            SUM(f.VL_LIQUIDO_FAT) AS total_liquido,
            COUNT(DISTINCT f.ID_CD_INTERNACAO) AS n_internacoes
        FROM agg_tb_fatura_fatu f
        JOIN agg_tb_capta_internacao_cain i
            ON f.ID_CD_INTERNACAO = i.ID_CD_INTERNACAO AND f.source_db = i.source_db
        WHERE f.{SRC} AND i.DH_ADMISSAO_HOSP IS NOT NULL
        GROUP BY mes
        ORDER BY mes
    """)
    fin["monthly_billing"] = _rows_to_dicts(cols, rows)

    # Top 10 CIDs by glosa amount
    cols, rows = _exec(con, f"""
        SELECT c.DS_DESCRICAO AS cid_desc,
            SUM(g.VL_GLOSADO) AS total_glosado,
            COUNT(DISTINCT g.ID_CD_INTERNACAO) AS n_internacoes,
            SUM(f.VL_TOTAL) AS total_faturado
        FROM agg_tb_fatura_glosa_fatg g
        JOIN agg_tb_capta_cid_caci c
            ON g.ID_CD_INTERNACAO = c.ID_CD_INTERNACAO AND g.source_db = c.source_db
        JOIN agg_tb_fatura_fatu f
            ON g.ID_CD_INTERNACAO = f.ID_CD_INTERNACAO AND g.source_db = f.source_db
        WHERE g.{SRC} AND c.DS_DESCRICAO IS NOT NULL
        GROUP BY c.DS_DESCRICAO
        ORDER BY total_glosado DESC
        LIMIT 10
    """)
    fin["top_cids_glosa"] = _rows_to_dicts(cols, rows)

    # Glosa reasons (top motivos)
    try:
        cols, rows = _exec(con, f"""
            SELECT COALESCE(g.DS_MOTIVO_GLOSA, 'Sem motivo informado') AS motivo,
                COUNT(*) AS n_glosas,
                SUM(g.VL_GLOSADO) AS total_glosado
            FROM agg_tb_fatura_glosa_fatg g
            WHERE g.{SRC}
            GROUP BY motivo
            ORDER BY total_glosado DESC
            LIMIT 10
        """)
        fin["glosa_reasons"] = _rows_to_dicts(cols, rows)
    except Exception:
        fin["glosa_reasons"] = []

    # Denied items patterns (recovery estimate)
    try:
        cols, rows = _exec(con, f"""
            SELECT
                SUM(CASE WHEN g.FL_ACEITE_GLOSA = 'N' THEN g.VL_GLOSADO ELSE 0 END) AS vl_contestado,
                SUM(CASE WHEN g.FL_ACEITE_GLOSA = 'S' THEN g.VL_GLOSADO ELSE 0 END) AS vl_aceito,
                SUM(g.VL_GLOSADO) AS vl_total_glosado,
                COUNT(*) AS n_total_glosas
            FROM agg_tb_fatura_glosa_fatg g
            WHERE g.{SRC}
        """)
        fin["glosa_summary"] = dict(zip(cols, rows[0])) if rows else {}
    except Exception:
        fin["glosa_summary"] = {}

    print(f"    Financial analysis done in {time.time()-t0:.1f}s")
    return fin


# ─────────────────────────────────────────────────────────────────
# Step 6 — Readmissions
# ─────────────────────────────────────────────────────────────────

def _fetch_readmissions(con):
    """30-day readmission analysis, frequent flyers."""
    import time
    print("[7/8] Fetching readmission analysis ...")
    t0 = time.time()
    SRC = f"source_db = '{SOURCE_DB}'"
    readmit = {}

    # 30-day readmission by CID
    cols, rows = _exec(con, f"""
        WITH adm_seq AS (
            SELECT i.ID_CD_INTERNACAO, i.ID_CD_PACIENTE,
                   i.DH_ADMISSAO_HOSP::DATE AS dt_adm,
                   LAG(i.DH_FINALIZACAO::DATE) OVER (
                       PARTITION BY i.ID_CD_PACIENTE
                       ORDER BY i.DH_ADMISSAO_HOSP
                   ) AS prev_discharge
            FROM agg_tb_capta_internacao_cain i
            WHERE i.{SRC} AND i.DH_ADMISSAO_HOSP IS NOT NULL
        ),
        readmissions AS (
            SELECT a.ID_CD_INTERNACAO,
                   CASE WHEN DATEDIFF('day', a.prev_discharge, a.dt_adm) BETWEEN 1 AND 30
                        THEN 1 ELSE 0 END AS is_readmission
            FROM adm_seq a
            WHERE a.prev_discharge IS NOT NULL
        )
        SELECT c.DS_DESCRICAO AS cid_desc,
            COUNT(DISTINCT r.ID_CD_INTERNACAO) AS total_adm,
            SUM(r.is_readmission) AS n_readmissions,
            100.0 * SUM(r.is_readmission) / COUNT(DISTINCT r.ID_CD_INTERNACAO) AS readmit_rate
        FROM readmissions r
        JOIN agg_tb_capta_cid_caci c
            ON r.ID_CD_INTERNACAO = c.ID_CD_INTERNACAO AND c.{SRC}
        WHERE c.DS_DESCRICAO IS NOT NULL
        GROUP BY c.DS_DESCRICAO
        HAVING COUNT(DISTINCT r.ID_CD_INTERNACAO) >= 20
        ORDER BY readmit_rate DESC
        LIMIT 10
    """)
    readmit["by_cid"] = _rows_to_dicts(cols, rows)

    # Frequent flyers: patients with 3+ admissions in 12 months
    cols, rows = _exec(con, f"""
        WITH recent_adm AS (
            SELECT ID_CD_PACIENTE,
                   COUNT(DISTINCT ID_CD_INTERNACAO) AS n_admissions,
                   MIN(DH_ADMISSAO_HOSP)::DATE AS first_adm,
                   MAX(DH_ADMISSAO_HOSP)::DATE AS last_adm
            FROM agg_tb_capta_internacao_cain
            WHERE {SRC}
              AND DH_ADMISSAO_HOSP >= CURRENT_DATE - INTERVAL '12 months'
            GROUP BY ID_CD_PACIENTE
            HAVING COUNT(DISTINCT ID_CD_INTERNACAO) >= 3
        )
        SELECT
            COUNT(*) AS n_patients,
            SUM(n_admissions) AS total_admissions,
            AVG(n_admissions) AS avg_admissions,
            MAX(n_admissions) AS max_admissions
        FROM recent_adm
    """)
    readmit["frequent_flyers_summary"] = dict(zip(cols, rows[0])) if rows else {}

    # Top 10 frequent flyers detail
    cols, rows = _exec(con, f"""
        WITH recent_adm AS (
            SELECT i.ID_CD_PACIENTE,
                   COUNT(DISTINCT i.ID_CD_INTERNACAO) AS n_admissions,
                   SUM(f.vl_total) AS total_billing
            FROM agg_tb_capta_internacao_cain i
            LEFT JOIN (
                SELECT ID_CD_INTERNACAO, source_db, SUM(VL_TOTAL) AS vl_total
                FROM agg_tb_fatura_fatu WHERE {SRC}
                GROUP BY ID_CD_INTERNACAO, source_db
            ) f ON i.ID_CD_INTERNACAO = f.ID_CD_INTERNACAO AND i.source_db = f.source_db
            WHERE i.{SRC}
              AND i.DH_ADMISSAO_HOSP >= CURRENT_DATE - INTERVAL '12 months'
            GROUP BY i.ID_CD_PACIENTE
            HAVING COUNT(DISTINCT i.ID_CD_INTERNACAO) >= 3
        )
        SELECT ID_CD_PACIENTE, n_admissions, total_billing
        FROM recent_adm
        ORDER BY n_admissions DESC, total_billing DESC
        LIMIT 10
    """)
    readmit["frequent_flyers_top10"] = _rows_to_dicts(cols, rows)

    # Estimated cost of avoidable readmissions (assume 50% of readmissions avoidable)
    cols, rows = _exec(con, f"""
        WITH adm_seq AS (
            SELECT i.ID_CD_INTERNACAO, i.ID_CD_PACIENTE,
                   i.DH_ADMISSAO_HOSP::DATE AS dt_adm,
                   LAG(i.DH_FINALIZACAO::DATE) OVER (
                       PARTITION BY i.ID_CD_PACIENTE
                       ORDER BY i.DH_ADMISSAO_HOSP
                   ) AS prev_discharge
            FROM agg_tb_capta_internacao_cain i
            WHERE i.{SRC} AND i.DH_ADMISSAO_HOSP IS NOT NULL
        ),
        readmissions AS (
            SELECT a.ID_CD_INTERNACAO
            FROM adm_seq a
            WHERE a.prev_discharge IS NOT NULL
              AND DATEDIFF('day', a.prev_discharge, a.dt_adm) BETWEEN 1 AND 30
        )
        SELECT
            COUNT(*) AS n_readmissions,
            SUM(f.vl_total) AS total_readmission_billing
        FROM readmissions r
        JOIN (
            SELECT ID_CD_INTERNACAO, source_db, SUM(VL_TOTAL) AS vl_total
            FROM agg_tb_fatura_fatu WHERE {SRC}
            GROUP BY ID_CD_INTERNACAO, source_db
        ) f ON r.ID_CD_INTERNACAO = f.ID_CD_INTERNACAO
    """)
    readmit["readmission_cost"] = dict(zip(cols, rows[0])) if rows else {}

    print(f"    Readmission analysis done in {time.time()-t0:.1f}s")
    return readmit


# ─────────────────────────────────────────────────────────────────
# Step 7 — Embedding Algebra (Clusters + Similar Patients)
# ─────────────────────────────────────────────────────────────────

def _embedding_analysis(embeddings, internacao_mask, unique_nodes, node_to_idx,
                        top_anomaly_ids, con):
    """HDBSCAN clustering + similar patient lookup + what-if analysis."""
    import time
    import numpy as np

    print("[8/8] Embedding algebra analysis ...")
    t0 = time.time()
    SRC = f"source_db = '{SOURCE_DB}'"

    int_vecs  = embeddings[internacao_mask]
    int_names = unique_nodes[internacao_mask]

    results = {}

    # ── HDBSCAN on admission embeddings ──
    print("    Running HDBSCAN clustering ...")
    try:
        import hdbscan
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=HDBSCAN_MIN_CLUSTER,
            min_samples=HDBSCAN_MIN_SAMPLES,
            metric="euclidean",
            core_dist_n_jobs=-1,
        )
        labels = clusterer.fit_predict(int_vecs)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise    = (labels == -1).sum()
        print(f"    HDBSCAN: {n_clusters} clusters, {n_noise:,} noise points")
    except Exception as e:
        print(f"    HDBSCAN failed: {e}, falling back to KMeans")
        from sklearn.cluster import MiniBatchKMeans
        km = MiniBatchKMeans(n_clusters=12, random_state=42, batch_size=10000)
        labels = km.fit_predict(int_vecs)
        n_clusters = 12
        n_noise = 0

    # Build cluster -> internacao IDs mapping
    cluster_iids: dict[int, list[int]] = {}
    for i, name in enumerate(int_names):
        lbl = int(labels[i])
        if lbl == -1:
            continue
        s = str(name)
        try:
            iid = int(s.split("ID_CD_INTERNACAO_")[1])
            cluster_iids.setdefault(lbl, []).append(iid)
        except Exception:
            pass

    # Characterize top clusters
    print("    Characterizing clusters ...")
    cluster_profiles = []
    for lbl in sorted(cluster_iids.keys()):
        iids = cluster_iids[lbl]
        if len(iids) < 20:
            continue
        # Sample up to 5000 for efficiency
        sample = iids[:5000]
        id_list = ", ".join(str(x) for x in sample)
        try:
            cols, rows = _exec(con, f"""
                SELECT
                    COUNT(DISTINCT i.ID_CD_INTERNACAO) AS n_adm,
                    AVG(DATEDIFF('day', i.DH_ADMISSAO_HOSP::DATE,
                        COALESCE(i.DH_FINALIZACAO, CURRENT_TIMESTAMP)::DATE)) AS avg_los,
                    AVG(f.vl_total) AS avg_billing,
                    AVG(f.vl_glosa) AS avg_glosa
                FROM agg_tb_capta_internacao_cain i
                LEFT JOIN (
                    SELECT ID_CD_INTERNACAO, source_db,
                           SUM(VL_TOTAL) AS vl_total,
                           SUM(VL_GLOSA_FECHAMENTO) AS vl_glosa
                    FROM agg_tb_fatura_fatu WHERE {SRC}
                    GROUP BY ID_CD_INTERNACAO, source_db
                ) f ON i.ID_CD_INTERNACAO = f.ID_CD_INTERNACAO AND i.source_db = f.source_db
                WHERE i.{SRC} AND i.ID_CD_INTERNACAO IN ({id_list})
            """)
            stats = dict(zip(cols, rows[0])) if rows else {}

            # Top CID
            cols2, rows2 = _exec(con, f"""
                SELECT c.DS_DESCRICAO AS cid_desc, COUNT(*) AS cnt
                FROM agg_tb_capta_cid_caci c
                WHERE c.{SRC} AND c.ID_CD_INTERNACAO IN ({id_list})
                  AND c.DS_DESCRICAO IS NOT NULL
                GROUP BY c.DS_DESCRICAO
                ORDER BY cnt DESC
                LIMIT 3
            """)
            top_cids = _rows_to_dicts(cols2, rows2)

            # Mortality in cluster
            cols3, rows3 = _exec(con, f"""
                WITH ls AS (
                    SELECT es.ID_CD_INTERNACAO,
                           ROW_NUMBER() OVER (
                               PARTITION BY es.ID_CD_INTERNACAO
                               ORDER BY es.DH_CADASTRO DESC
                           ) AS rn,
                           es.FL_DESOSPITALIZACAO
                    FROM agg_tb_capta_evo_status_caes es
                    WHERE es.{SRC} AND es.ID_CD_INTERNACAO IN ({id_list})
                )
                SELECT
                    COUNT(*) AS n_discharged,
                    COUNT(*) FILTER (
                        WHERE UPPER(f.DS_FINAL_MONITORAMENTO) LIKE '%%OBITO%%'
                           OR UPPER(f.DS_FINAL_MONITORAMENTO) LIKE '%%ÓBITO%%'
                    ) AS n_obito
                FROM ls
                JOIN agg_tb_capta_tipo_final_monit_fmon f
                    ON ls.FL_DESOSPITALIZACAO = f.ID_CD_FINAL_MONITORAMENTO
                WHERE ls.rn = 1
            """)
            mort = dict(zip(cols3, rows3[0])) if rows3 else {}

            cluster_profiles.append({
                "label": lbl,
                "n_admissions": len(iids),
                "avg_los": _safe_float(stats.get("avg_los")),
                "avg_billing": _safe_float(stats.get("avg_billing")),
                "avg_glosa": _safe_float(stats.get("avg_glosa")),
                "top_cids": top_cids,
                "n_obito": _safe_int(mort.get("n_obito")),
                "n_discharged": _safe_int(mort.get("n_discharged")),
            })
        except Exception as e:
            print(f"    Cluster {lbl} characterization failed: {e}")
            continue

    # Sort by risk (mortality rate, then avg billing)
    for cp in cluster_profiles:
        denom = max(cp["n_discharged"], 1)
        cp["obito_rate"] = 100.0 * cp["n_obito"] / denom
    cluster_profiles.sort(key=lambda x: (-x["obito_rate"], -x["avg_billing"]))
    results["clusters"] = cluster_profiles[:10]  # top 10 clusters

    # ── Similar patients for top anomalies ──
    print("    Finding similar patients for anomalies ...")
    norms = np.linalg.norm(int_vecs, axis=1, keepdims=True).clip(min=1e-8)
    int_vecs_norm = int_vecs / norms

    similar_patients = {}
    for iid in top_anomaly_ids[:10]:
        key = f"{SOURCE_DB}/ID_CD_INTERNACAO_{iid}"
        if key not in node_to_idx:
            continue
        idx = node_to_idx[key]
        qvec = embeddings[idx].astype(np.float32)
        qnorm = np.linalg.norm(qvec).clip(min=1e-8)
        sims = int_vecs_norm @ (qvec / qnorm)
        top_k = np.argsort(-sims)[:6]
        sim_list = []
        for j in top_k:
            name = str(int_names[j])
            if name == key:
                continue
            try:
                sim_iid = int(name.split("ID_CD_INTERNACAO_")[1])
                sim_list.append((sim_iid, float(sims[j])))
            except Exception:
                pass
            if len(sim_list) >= 5:
                break
        similar_patients[iid] = sim_list

    # Fetch details for similar patients
    all_sim_iids = set()
    for sims in similar_patients.values():
        for sim_iid, _ in sims:
            all_sim_iids.add(sim_iid)

    sim_details = {}
    if all_sim_iids:
        id_list = ", ".join(str(x) for x in all_sim_iids)
        try:
            cols, rows = _exec(con, f"""
                SELECT i.ID_CD_INTERNACAO,
                    DATEDIFF('day', i.DH_ADMISSAO_HOSP::DATE,
                        COALESCE(i.DH_FINALIZACAO, CURRENT_TIMESTAMP)::DATE) AS los_dias,
                    f.vl_total
                FROM agg_tb_capta_internacao_cain i
                LEFT JOIN (
                    SELECT ID_CD_INTERNACAO, source_db, SUM(VL_TOTAL) AS vl_total
                    FROM agg_tb_fatura_fatu WHERE {SRC}
                    GROUP BY ID_CD_INTERNACAO, source_db
                ) f ON i.ID_CD_INTERNACAO = f.ID_CD_INTERNACAO AND i.source_db = f.source_db
                WHERE i.{SRC} AND i.ID_CD_INTERNACAO IN ({id_list})
            """)
            for r in _rows_to_dicts(cols, rows):
                sim_details[r["ID_CD_INTERNACAO"]] = r
        except Exception:
            pass

    results["similar_patients"] = similar_patients
    results["similar_details"]  = sim_details

    print(f"    Embedding analysis done in {time.time()-t0:.1f}s")
    return results


# ─────────────────────────────────────────────────────────────────
# LaTeX Generation
# ─────────────────────────────────────────────────────────────────

def _generate_latex(kpis, profile, anomaly_admissions, los_data,
                    fin_data, readmit_data, emb_results,
                    total_anomalies) -> str:

    L = []

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

\definecolor{bradblue}{RGB}{0,47,108}
\definecolor{bradred}{RGB}{204,0,0}
\definecolor{bradgray}{RGB}{100,100,100}
\definecolor{bradlightblue}{RGB}{220,235,250}
\definecolor{bradlightgray}{RGB}{245,245,245}
\definecolor{bradgold}{RGB}{180,140,20}
\definecolor{bradgreen}{RGB}{0,128,60}
\definecolor{anomred}{RGB}{180,20,20}
\definecolor{anomorange}{RGB}{220,100,0}
\definecolor{anomyellow}{RGB}{160,120,0}
\definecolor{kpibox}{RGB}{240,245,255}
\definecolor{alertbox}{RGB}{255,240,240}
\definecolor{successbox}{RGB}{240,255,240}
\definecolor{warnbox}{RGB}{255,250,230}

\tcbuselibrary{skins,breakable}

\newtcolorbox{kpicard}[1][]{%
  enhanced,
  colback=kpibox,
  colframe=bradblue,
  fonttitle=\bfseries\small,
  title={#1},
  left=5pt, right=5pt, top=4pt, bottom=4pt,
  boxrule=1pt
}

\newtcolorbox{alertcard}[1][]{%
  enhanced,
  colback=alertbox,
  colframe=bradred,
  fonttitle=\bfseries\small,
  title={#1},
  left=5pt, right=5pt, top=4pt, bottom=4pt,
  boxrule=1.2pt
}

\newtcolorbox{infocard}[1][]{%
  enhanced,
  colback=bradlightgray,
  colframe=bradgray,
  fonttitle=\bfseries\small,
  title={#1},
  left=5pt, right=5pt, top=4pt, bottom=4pt,
  boxrule=0.8pt,
  breakable
}

\newtcolorbox{highlightcard}[1][]{%
  enhanced,
  colback=warnbox,
  colframe=bradgold,
  fonttitle=\bfseries\small,
  title={#1},
  left=5pt, right=5pt, top=4pt, bottom=4pt,
  boxrule=1pt
}

\newtcolorbox{anomalycard}[2][]{%
  enhanced,
  colback=bradlightgray,
  colframe=#2,
  fonttitle=\bfseries\footnotesize,
  title={#1},
  left=5pt, right=5pt, top=4pt, bottom=4pt,
  boxrule=1.5pt,
  before upper={\setlength{\parskip}{2pt}},
  before={\needspace{0.9\textheight}}
}

\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\textcolor{bradblue}{\textbf{JCUBE}} \textcolor{bradgray}{\small | GHO-BRADESCO --- Relatorio Executivo V6}}
\fancyhead[R]{\textcolor{bradgray}{\small """ + REPORT_DATE_STR.replace("-", "/") + r"""}}
\fancyfoot[C]{\textcolor{bradgray}{\thepage}}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0.4pt}

\titleformat{\section}{\Large\bfseries\color{bradblue}}{\thesection}{1em}{}[\titlerule]
\titleformat{\subsection}{\large\bfseries\color{bradblue!80!black}}{\thesubsection}{1em}{}
\titleformat{\subsubsection}{\normalsize\bfseries\color{bradgray}}{\thesubsubsection}{1em}{}

\hypersetup{colorlinks=true,linkcolor=bradblue,pdftitle={GHO-BRADESCO Relatorio Executivo V6}}

\begin{document}
\setlength{\parindent}{0pt}
\setlength{\parskip}{4pt}
""")

    # ════════════════════════════════════════════════════════════
    # SECTION 1 — CAPA E RESUMO EXECUTIVO
    # ════════════════════════════════════════════════════════════
    total_internacoes = _safe_int(kpis.get("total_internacoes"))
    total_pacientes   = _safe_int(kpis.get("total_pacientes"))
    total_faturado    = _safe_float(kpis.get("total_faturado"))
    total_glosado     = _safe_float(kpis.get("total_glosado"))
    total_liquido     = _safe_float(kpis.get("total_liquido"))
    avg_los           = _safe_float(kpis.get("avg_los"))
    median_los        = _safe_float(kpis.get("median_los"))
    dt_inicio         = _fmt_date(kpis.get("dt_inicio"))
    dt_fim            = _fmt_date(kpis.get("dt_fim"))
    readmit_30d       = _safe_int(kpis.get("readmissions_30d"))
    total_with_prior  = _safe_int(kpis.get("total_with_prior"))
    n_obito           = _safe_int(kpis.get("n_obito"))
    total_discharged  = _safe_int(kpis.get("total_discharged"))
    taxa_glosa        = 100.0 * total_glosado / total_faturado if total_faturado > 0 else 0.0
    taxa_readmit      = 100.0 * readmit_30d / total_with_prior if total_with_prior > 0 else 0.0
    taxa_obito        = 100.0 * n_obito / total_discharged if total_discharged > 0 else 0.0

    # Headline finding: estimated recoverable
    vl_contestado = _safe_float(fin_data.get("glosa_summary", {}).get("vl_contestado"))
    headline_value = vl_contestado if vl_contestado > 0 else total_glosado * 0.3

    L.append(r"""\begin{titlepage}
\begin{center}
\vspace*{1cm}
{\Huge\bfseries\textcolor{bradblue}{JCUBE}}\\[0.15cm]
{\large\textcolor{bradgray}{Digital Twin Analytics --- Graph-JEPA V6}}\\[0.8cm]
\begin{tcolorbox}[colback=bradblue,colframe=bradblue,coltext=white,width=0.95\textwidth,halign=center]
{\LARGE\bfseries Relatorio Executivo de Saude}\\[0.3cm]
{\large GHO-BRADESCO}\\[0.2cm]
{\normalsize 2,86 milhoes de nos $\times$ 128 dimensoes --- 4 epocas TGN com memoria temporal}
\end{tcolorbox}
\vspace{0.6cm}
""")
    L.append(
        r"{\Large Periodo: \textbf{" + _escape_latex(dt_inicio) +
        r"} a \textbf{" + _escape_latex(dt_fim) + r"}}\\[0.3cm]"
    )
    L.append(r"{\large Gerado em: \textbf{24 de marco de 2026}}\\[1cm]")

    # Headline finding box
    L.append(r"""
\begin{tcolorbox}[colback=bradred!8,colframe=bradred,width=0.88\textwidth,halign=center]
{\large\bfseries\textcolor{bradred}{Achado Principal}}\\[4pt]
{\Large """ + _brl_millions(headline_value) + r""" em potencial de recuperacao identificado}\\[2pt]
{\normalsize via analise de glosas contestaveis + padroes anomalos de faturamento}
\end{tcolorbox}
\vspace{0.5cm}
""")

    # KPI summary boxes
    L.append(r"""
\begin{tabular}{cccc}
\begin{tcolorbox}[colback=kpibox,colframe=bradblue,width=3.5cm,halign=center,left=2pt,right=2pt]
{\LARGE\bfseries\textcolor{bradblue}{""" + f"{total_internacoes:,}".replace(",", ".") + r"""}}\\[2pt]
{\small Internacoes}
\end{tcolorbox}
&
\begin{tcolorbox}[colback=kpibox,colframe=bradblue,width=3.5cm,halign=center,left=2pt,right=2pt]
{\LARGE\bfseries\textcolor{bradblue}{""" + _brl_millions(total_faturado) + r"""}}\\[2pt]
{\small Total Faturado}
\end{tcolorbox}
&
\begin{tcolorbox}[colback=alertbox,colframe=bradred,width=3.5cm,halign=center,left=2pt,right=2pt]
{\LARGE\bfseries\textcolor{bradred}{""" + f"{taxa_glosa:.1f}\\%" + r"""}}\\[2pt]
{\small Taxa de Glosa}
\end{tcolorbox}
&
\begin{tcolorbox}[colback=kpibox,colframe=bradblue,width=3.5cm,halign=center,left=2pt,right=2pt]
{\LARGE\bfseries\textcolor{bradblue}{""" + f"{avg_los:.1f}" + r"""d}}\\[2pt]
{\small LOS Medio}
\end{tcolorbox}
\end{tabular}

\vspace{0.3cm}

\begin{tabular}{ccc}
\begin{tcolorbox}[colback=warnbox,colframe=bradgold,width=4.2cm,halign=center,left=2pt,right=2pt]
{\LARGE\bfseries\textcolor{bradgold}{""" + f"{taxa_readmit:.1f}\\%" + r"""}}\\[2pt]
{\small Readmissao 30 dias}
\end{tcolorbox}
&
\begin{tcolorbox}[colback=alertbox,colframe=bradred,width=4.2cm,halign=center,left=2pt,right=2pt]
{\LARGE\bfseries\textcolor{bradred}{""" + f"{taxa_obito:.1f}\\%" + r"""}}\\[2pt]
{\small Taxa de Obito}
\end{tcolorbox}
&
\begin{tcolorbox}[colback=kpibox,colframe=bradblue,width=4.2cm,halign=center,left=2pt,right=2pt]
{\LARGE\bfseries\textcolor{bradblue}{""" + f"{total_pacientes:,}".replace(",", ".") + r"""}}\\[2pt]
{\small Pacientes Unicos}
\end{tcolorbox}
\end{tabular}
""")

    L.append(r"""
\vfill
{\small\textcolor{bradgray}{Confidencial --- Preparado exclusivamente para Bradesco Saude.\\
Dados processados via JCUBE Graph-JEPA V6 (TGN com memoria temporal).\\
Este documento contem informacoes estrategicas e nao deve ser distribuido externamente.}}
\end{center}
\end{titlepage}
""")

    # ── Table of Contents ──
    L.append(r"""
\tableofcontents
\newpage
""")

    # ════════════════════════════════════════════════════════════
    # SECTION 2 — PERFIL OPERACIONAL
    # ════════════════════════════════════════════════════════════
    L.append(r"\section{Perfil Operacional}" + "\n")

    # 2.1 Monthly admissions trend
    L.append(r"\subsection{Tendencia Mensal de Internacoes}" + "\n")
    monthly = profile.get("monthly_admissions", [])
    if monthly:
        L.append(r"\begin{infocard}[Volume mensal de internacoes]" + "\n")
        L.append(r"\begin{tabularx}{\textwidth}{l r r}" + "\n")
        L.append(r"\toprule" + "\n")
        L.append(r"\textbf{Mes} & \textbf{Internacoes} & \textbf{Pacientes} \\" + "\n")
        L.append(r"\midrule" + "\n")
        for m in monthly[-12:]:  # last 12 months
            mes = _escape_latex(str(m.get("mes", "---"))[:7])
            n_int = _safe_int(m.get("n_internacoes"))
            n_pac = _safe_int(m.get("n_pacientes"))
            L.append(f"{mes} & {n_int:,} & {n_pac:,} \\\\\n".replace(",", "."))
        L.append(r"\bottomrule" + "\n")
        L.append(r"\end{tabularx}" + "\n")
        L.append(r"\end{infocard}" + "\n\n")

    # 2.2 Top CIDs by volume
    L.append(r"\subsection{Top 10 CIDs por Volume}" + "\n")
    top_cids_vol = profile.get("top_cids_volume", [])
    if top_cids_vol:
        L.append(r"\begin{infocard}[Diagnosticos mais frequentes]" + "\n")
        L.append(r"\begin{tabularx}{\textwidth}{r X r}" + "\n")
        L.append(r"\toprule" + "\n")
        L.append(r"\textbf{\#} & \textbf{CID / Diagnostico} & \textbf{Internacoes} \\" + "\n")
        L.append(r"\midrule" + "\n")
        for i, c in enumerate(top_cids_vol, 1):
            desc = _escape_latex(_truncate(str(c.get("cid_desc", "---")), 60))
            n = _safe_int(c.get("n_internacoes"))
            L.append(f"{i} & {desc} & {n:,} \\\\\n".replace(",", "."))
        L.append(r"\bottomrule" + "\n")
        L.append(r"\end{tabularx}" + "\n")
        L.append(r"\end{infocard}" + "\n\n")

    # 2.3 Top CIDs by cost
    L.append(r"\subsection{Top 10 CIDs por Custo}" + "\n")
    top_cids_cost = profile.get("top_cids_cost", [])
    if top_cids_cost:
        L.append(r"\begin{infocard}[Diagnosticos de maior custo]" + "\n")
        L.append(r"\begin{tabularx}{\textwidth}{r X r r}" + "\n")
        L.append(r"\toprule" + "\n")
        L.append(r"\textbf{\#} & \textbf{CID / Diagnostico} & \textbf{Total Faturado} & \textbf{Internacoes} \\" + "\n")
        L.append(r"\midrule" + "\n")
        for i, c in enumerate(top_cids_cost, 1):
            desc = _escape_latex(_truncate(str(c.get("cid_desc", "---")), 50))
            vl = _brl(_safe_float(c.get("total_faturado")))
            n = _safe_int(c.get("n_internacoes"))
            L.append(f"{i} & {desc} & {vl} & {n:,} \\\\\n".replace(",", "."))
        L.append(r"\bottomrule" + "\n")
        L.append(r"\end{tabularx}" + "\n")
        L.append(r"\end{infocard}" + "\n\n")

    # 2.4 Top procedures
    L.append(r"\subsection{Top 10 Procedimentos (TUSS)}" + "\n")
    top_procs = profile.get("top_procedures", [])
    if top_procs:
        L.append(r"\begin{infocard}[Procedimentos mais realizados]" + "\n")
        L.append(r"\begin{tabularx}{\textwidth}{r X r}" + "\n")
        L.append(r"\toprule" + "\n")
        L.append(r"\textbf{\#} & \textbf{Procedimento} & \textbf{Quantidade} \\" + "\n")
        L.append(r"\midrule" + "\n")
        for i, p in enumerate(top_procs, 1):
            desc = _escape_latex(_truncate(str(p.get("proc_desc", "---")), 60))
            n = _safe_int(p.get("n_procedimentos"))
            L.append(f"{i} & {desc} & {n:,} \\\\\n".replace(",", "."))
        L.append(r"\bottomrule" + "\n")
        L.append(r"\end{tabularx}" + "\n")
        L.append(r"\end{infocard}" + "\n\n")

    # 2.5 LOS distribution
    L.append(r"\subsection{Distribuicao de Permanencia (LOS)}" + "\n")
    los_dist = profile.get("los_distribution", [])
    if los_dist:
        total_los = sum(_safe_int(d.get("n_internacoes")) for d in los_dist)
        L.append(r"\begin{infocard}[Histograma de permanencia hospitalar]" + "\n")
        L.append(r"\begin{tabularx}{\textwidth}{l r r}" + "\n")
        L.append(r"\toprule" + "\n")
        L.append(r"\textbf{Faixa de LOS} & \textbf{Internacoes} & \textbf{\% do Total} \\" + "\n")
        L.append(r"\midrule" + "\n")
        for d in los_dist:
            faixa = _escape_latex(str(d.get("faixa_los", "---")))
            n = _safe_int(d.get("n_internacoes"))
            pct = 100.0 * n / total_los if total_los > 0 else 0.0
            L.append(f"{faixa} & {n:,} & {pct:.1f}\\% \\\\\n".replace(",", "."))
        L.append(r"\bottomrule" + "\n")
        L.append(r"\end{tabularx}" + "\n")
        L.append(r"\end{infocard}" + "\n\n")

    # 2.6 Discharge types
    L.append(r"\subsection{Distribuicao por Tipo de Alta}" + "\n")
    dtypes = profile.get("discharge_types", [])
    if dtypes:
        total_dis = sum(_safe_int(d.get("n_internacoes")) for d in dtypes)
        L.append(r"\begin{infocard}[Desfechos das internacoes]" + "\n")
        L.append(r"\begin{tabularx}{\textwidth}{X r r}" + "\n")
        L.append(r"\toprule" + "\n")
        L.append(r"\textbf{Tipo de Alta} & \textbf{Internacoes} & \textbf{\% do Total} \\" + "\n")
        L.append(r"\midrule" + "\n")
        for d in dtypes:
            tipo = _escape_latex(_truncate(str(d.get("tipo_alta", "---")), 50))
            n = _safe_int(d.get("n_internacoes"))
            pct = 100.0 * n / total_dis if total_dis > 0 else 0.0
            L.append(f"{tipo} & {n:,} & {pct:.1f}\\% \\\\\n".replace(",", "."))
        L.append(r"\bottomrule" + "\n")
        L.append(r"\end{tabularx}" + "\n")
        L.append(r"\end{infocard}" + "\n")

    L.append(r"\newpage" + "\n")

    # ════════════════════════════════════════════════════════════
    # SECTION 3 — DETECCAO DE ANOMALIAS PRIORITARIAS
    # ════════════════════════════════════════════════════════════
    L.append(r"\section{Deteccao de Anomalias Prioritarias}" + "\n")

    L.append(r"""
\begin{alertcard}[Resumo de Anomalias]
O gemeo digital identificou \textbf{""" + f"{total_anomalies:,}".replace(",", ".") + r"""} internacoes com comportamento
anomalo (z-score $>$ 2,0) no universo GHO-BRADESCO.
As """ + str(len(anomaly_admissions)) + r""" mais criticas sao detalhadas abaixo, com contexto clinico e financeiro.
\end{alertcard}
\vspace{0.3cm}
""")

    # Financial impact summary of anomalies
    anom_total_billing = sum(_safe_float(a.get("billing", {}).get("vl_total")) for a in anomaly_admissions)
    anom_total_glosa   = sum(_safe_float(a.get("glosa", {}).get("vl_glosado")) for a in anomaly_admissions)
    if anom_total_billing > 0:
        L.append(r"""
\begin{highlightcard}[Impacto Financeiro das Anomalias]
\begin{itemize}[leftmargin=1.5em]
\item Faturamento total das """ + str(len(anomaly_admissions)) + r""" internacoes anomalas: \textbf{""" + _brl(anom_total_billing) + r"""}
\item Glosas associadas: \textbf{""" + _brl(anom_total_glosa) + r"""}
\end{itemize}
\end{highlightcard}
\vspace{0.3cm}
""")

    # Anomaly cards
    for idx, a in enumerate(anomaly_admissions):
        z = _safe_float(a.get("z_score"))
        vl_total = _safe_float(a.get("billing", {}).get("vl_total"))
        los = _safe_int(a.get("los_dias"))
        iid = _safe_int(a.get("ID_CD_INTERNACAO"))
        pid = _safe_int(a.get("ID_CD_PACIENTE"))
        adm_date = _fmt_date(a.get("DH_ADMISSAO_HOSP"))
        fin_date = _fmt_date(a.get("DH_FINALIZACAO"))
        cids = _escape_latex(_truncate(str(a.get("cids", "---")), 80))
        n_procs = _safe_int(a.get("n_procs"))
        vl_glosa = _safe_float(a.get("glosa", {}).get("vl_glosado"))
        baseline_los = _safe_float(a.get("baseline", {}).get("baseline_los"))
        baseline_bill = _safe_float(a.get("baseline", {}).get("baseline_billing"))

        # Severity classification
        if z >= 5 or vl_total >= 500_000:
            sev_color = "anomred"
            sev_label = "CRITICO"
        elif z >= 3:
            sev_color = "anomorange"
            sev_label = "ALTO"
        else:
            sev_color = "anomyellow"
            sev_label = "MODERADO"

        title = f"Anomalia {idx+1}/{len(anomaly_admissions)} --- Internacao {iid} [{sev_label}]"

        L.append(f"\\begin{{anomalycard}}[{_escape_latex(title)}]{{{sev_color}}}\n")

        # Data table
        L.append(r"\begin{tabularx}{\textwidth}{l X l X}" + "\n")
        L.append(f"\\textbf{{Paciente:}} & {pid} & \\textbf{{Z-Score:}} & {z:.2f} \\\\\n")
        L.append(f"\\textbf{{Admissao:}} & {_escape_latex(adm_date)} & \\textbf{{Alta:}} & {_escape_latex(fin_date)} \\\\\n")
        L.append(f"\\textbf{{LOS:}} & {los} dias & \\textbf{{Procedimentos:}} & {n_procs} \\\\\n")
        L.append(f"\\textbf{{Faturamento:}} & {_brl(vl_total)} & \\textbf{{Glosa:}} & {_brl(vl_glosa)} \\\\\n")
        L.append(r"\end{tabularx}" + "\n")
        L.append(f"\\textbf{{CIDs:}} {cids}\n\n")

        # Narrative: why is this anomalous
        reasons = []
        if baseline_los > 0 and los > 2 * baseline_los:
            mult = los / baseline_los
            reasons.append(f"permanencia de {los} dias ({mult:.1f}x a media de {baseline_los:.0f} dias)")
        if baseline_bill > 0 and vl_total > 3 * baseline_bill:
            mult = vl_total / baseline_bill
            reasons.append(f"faturamento {mult:.1f}x acima da media ({_brl_plain(baseline_bill)})")
        if vl_glosa > 0:
            reasons.append(f"glosa de {_brl_plain(vl_glosa)}")
        if z >= 5:
            reasons.append(f"z-score extremo ({z:.1f})")
        if not reasons:
            reasons.append(f"desvio significativo no espaco de embeddings (z={z:.1f})")

        reason_text = "; ".join(reasons)
        L.append(f"\\textbf{{Por que e anomala:}} {_escape_latex(reason_text)}.\n\n")

        L.append(r"\end{anomalycard}" + "\n\n")

    L.append(r"\newpage" + "\n")

    # ════════════════════════════════════════════════════════════
    # SECTION 4 — ANALISE DE PERMANENCIA
    # ════════════════════════════════════════════════════════════
    L.append(r"\section{Analise de Permanencia (LOS)}" + "\n")

    # 4.1 LOS by CID
    L.append(r"\subsection{Permanencia por Diagnostico (Top 10)}" + "\n")
    los_by_cid = los_data.get("by_cid", [])
    if los_by_cid:
        L.append(r"\begin{infocard}[Diagnosticos com maior permanencia media]" + "\n")
        L.append(r"\footnotesize" + "\n")
        L.append(r"\begin{tabularx}{\textwidth}{r X r r r r}" + "\n")
        L.append(r"\toprule" + "\n")
        L.append(r"\textbf{\#} & \textbf{CID} & \textbf{N} & \textbf{Media} & \textbf{Mediana} & \textbf{Max} \\" + "\n")
        L.append(r"\midrule" + "\n")
        for i, c in enumerate(los_by_cid, 1):
            desc = _escape_latex(_truncate(str(c.get("cid_desc", "---")), 40))
            n = _safe_int(c.get("n_internacoes"))
            avg = _safe_float(c.get("avg_los"))
            med = _safe_float(c.get("median_los"))
            mx = _safe_int(c.get("max_los"))
            L.append(f"{i} & {desc} & {n:,} & {avg:.1f}d & {med:.0f}d & {mx}d \\\\\n".replace(",", "."))
        L.append(r"\bottomrule" + "\n")
        L.append(r"\end{tabularx}" + "\n")
        L.append(r"\end{infocard}" + "\n\n")

    # 4.2 LOS by discharge type
    L.append(r"\subsection{Permanencia por Tipo de Alta}" + "\n")
    los_by_dis = los_data.get("by_discharge", [])
    if los_by_dis:
        L.append(r"\begin{infocard}[LOS medio por desfecho]" + "\n")
        L.append(r"\begin{tabularx}{\textwidth}{X r r r}" + "\n")
        L.append(r"\toprule" + "\n")
        L.append(r"\textbf{Tipo de Alta} & \textbf{N} & \textbf{Media (dias)} & \textbf{Mediana (dias)} \\" + "\n")
        L.append(r"\midrule" + "\n")
        for d in los_by_dis:
            tipo = _escape_latex(_truncate(str(d.get("tipo_alta", "---")), 40))
            n = _safe_int(d.get("n_internacoes"))
            avg = _safe_float(d.get("avg_los"))
            med = _safe_float(d.get("median_los"))
            L.append(f"{tipo} & {n:,} & {avg:.1f} & {med:.0f} \\\\\n".replace(",", "."))
        L.append(r"\bottomrule" + "\n")
        L.append(r"\end{tabularx}" + "\n")
        L.append(r"\end{infocard}" + "\n\n")

    # 4.3 Excess LOS
    L.append(r"\subsection{Internacoes com Permanencia Excessiva ($>$2x mediana do CID)}" + "\n")
    excess_los = los_data.get("excess_los", [])
    excess_totals = los_data.get("excess_totals", {})
    total_excess_days = _safe_int(excess_totals.get("total_excess_days"))
    n_excess_adm = _safe_int(excess_totals.get("n_excess_admissions"))

    if total_excess_days > 0:
        # Estimate cost per bed-day (use avg daily billing as proxy)
        avg_daily_cost = total_faturado / (total_internacoes * avg_los) if (total_internacoes * avg_los) > 0 else 2000
        excess_cost = total_excess_days * avg_daily_cost
        L.append(r"""
\begin{alertcard}[Custo Estimado de Dias Excedentes]
\textbf{""" + f"{n_excess_adm:,}".replace(",", ".") + r"""} internacoes excederam 2x a mediana de permanencia para seu CID,
gerando \textbf{""" + f"{total_excess_days:,}".replace(",", ".") + r"""} diarias excedentes.\\
Custo estimado: \textbf{""" + _brl(excess_cost) + r"""} (com base no custo medio diario de """ + _brl(avg_daily_cost) + r""")
\end{alertcard}
\vspace{0.3cm}
""")

    if excess_los:
        L.append(r"\begin{infocard}[Top 10 CIDs com maior excesso de permanencia]" + "\n")
        L.append(r"\footnotesize" + "\n")
        L.append(r"\begin{tabularx}{\textwidth}{r X r r r r}" + "\n")
        L.append(r"\toprule" + "\n")
        L.append(r"\textbf{\#} & \textbf{CID} & \textbf{N Excess.} & \textbf{Mediana CID} & \textbf{Media Real} & \textbf{Dias Excess.} \\" + "\n")
        L.append(r"\midrule" + "\n")
        for i, c in enumerate(excess_los, 1):
            desc = _escape_latex(_truncate(str(c.get("cid_desc", "---")), 35))
            n = _safe_int(c.get("n_excess"))
            med = _safe_float(c.get("med_los"))
            avg_real = _safe_float(c.get("avg_actual_los"))
            excess = _safe_int(c.get("excess_days"))
            L.append(f"{i} & {desc} & {n:,} & {med:.0f}d & {avg_real:.0f}d & {excess:,} \\\\\n".replace(",", "."))
        L.append(r"\bottomrule" + "\n")
        L.append(r"\end{tabularx}" + "\n")
        L.append(r"\end{infocard}" + "\n")

    L.append(r"\newpage" + "\n")

    # ════════════════════════════════════════════════════════════
    # SECTION 5 — ANALISE FINANCEIRA
    # ════════════════════════════════════════════════════════════
    L.append(r"\section{Analise Financeira}" + "\n")

    # 5.1 Monthly billing trend
    L.append(r"\subsection{Tendencia Mensal de Faturamento}" + "\n")
    monthly_bill = fin_data.get("monthly_billing", [])
    if monthly_bill:
        L.append(r"\begin{infocard}[Evolucao mensal de faturamento e glosas]" + "\n")
        L.append(r"\footnotesize" + "\n")
        L.append(r"\begin{tabularx}{\textwidth}{l r r r r}" + "\n")
        L.append(r"\toprule" + "\n")
        L.append(r"\textbf{Mes} & \textbf{Faturado} & \textbf{Glosado} & \textbf{Liquido} & \textbf{Taxa Glosa} \\" + "\n")
        L.append(r"\midrule" + "\n")
        for m in monthly_bill[-12:]:
            mes = _escape_latex(str(m.get("mes", "---"))[:7])
            fat = _brl(_safe_float(m.get("total_faturado")))
            glo = _brl(_safe_float(m.get("total_glosado")))
            liq = _brl(_safe_float(m.get("total_liquido")))
            total_f = _safe_float(m.get("total_faturado"))
            total_g = _safe_float(m.get("total_glosado"))
            tg = f"{100*total_g/total_f:.1f}\\%" if total_f > 0 else "---"
            L.append(f"{mes} & {fat} & {glo} & {liq} & {tg} \\\\\n")
        L.append(r"\bottomrule" + "\n")
        L.append(r"\end{tabularx}" + "\n")
        L.append(r"\end{infocard}" + "\n\n")

    # 5.2 Top CIDs by glosa
    L.append(r"\subsection{Top 10 Diagnosticos por Valor de Glosa}" + "\n")
    top_glosa_cids = fin_data.get("top_cids_glosa", [])
    if top_glosa_cids:
        L.append(r"\begin{infocard}[CIDs com maior impacto de glosa]" + "\n")
        L.append(r"\footnotesize" + "\n")
        L.append(r"\begin{tabularx}{\textwidth}{r X r r r}" + "\n")
        L.append(r"\toprule" + "\n")
        L.append(r"\textbf{\#} & \textbf{CID} & \textbf{Glosado} & \textbf{Faturado} & \textbf{Taxa} \\" + "\n")
        L.append(r"\midrule" + "\n")
        for i, c in enumerate(top_glosa_cids, 1):
            desc = _escape_latex(_truncate(str(c.get("cid_desc", "---")), 35))
            glo = _brl(_safe_float(c.get("total_glosado")))
            fat = _brl(_safe_float(c.get("total_faturado")))
            tf = _safe_float(c.get("total_faturado"))
            tg = _safe_float(c.get("total_glosado"))
            taxa = f"{100*tg/tf:.1f}\\%" if tf > 0 else "---"
            L.append(f"{i} & {desc} & {glo} & {fat} & {taxa} \\\\\n")
        L.append(r"\bottomrule" + "\n")
        L.append(r"\end{tabularx}" + "\n")
        L.append(r"\end{infocard}" + "\n\n")

    # 5.3 Top glosa reasons
    L.append(r"\subsection{Principais Motivos de Glosa}" + "\n")
    glosa_reasons = fin_data.get("glosa_reasons", [])
    if glosa_reasons:
        L.append(r"\begin{infocard}[Motivos de glosa mais frequentes]" + "\n")
        L.append(r"\begin{tabularx}{\textwidth}{r X r r}" + "\n")
        L.append(r"\toprule" + "\n")
        L.append(r"\textbf{\#} & \textbf{Motivo} & \textbf{Quantidade} & \textbf{Valor Total} \\" + "\n")
        L.append(r"\midrule" + "\n")
        for i, r in enumerate(glosa_reasons, 1):
            motivo = _escape_latex(_truncate(str(r.get("motivo", "---")), 50))
            n = _safe_int(r.get("n_glosas"))
            vl = _brl(_safe_float(r.get("total_glosado")))
            L.append(f"{i} & {motivo} & {n:,} & {vl} \\\\\n".replace(",", "."))
        L.append(r"\bottomrule" + "\n")
        L.append(r"\end{tabularx}" + "\n")
        L.append(r"\end{infocard}" + "\n\n")

    # 5.4 Recovery estimate
    glosa_summary = fin_data.get("glosa_summary", {})
    vl_contestado = _safe_float(glosa_summary.get("vl_contestado"))
    vl_aceito     = _safe_float(glosa_summary.get("vl_aceito"))
    vl_total_glos = _safe_float(glosa_summary.get("vl_total_glosado"))
    if vl_total_glos > 0:
        L.append(r"\subsection{Estimativa de Recuperacao}" + "\n")
        contest_rate = 100.0 * vl_contestado / vl_total_glos if vl_total_glos > 0 else 0
        L.append(r"""
\begin{highlightcard}[Potencial de Recuperacao Financeira]
\begin{itemize}[leftmargin=1.5em]
\item Total glosado na base: \textbf{""" + _brl(vl_total_glos) + r"""}
\item Valor contestado (nao aceito pelo prestador): \textbf{""" + _brl(vl_contestado) + r"""} (""" + f"{contest_rate:.1f}\\%" + r""")
\item Valor aceito pela operadora: \textbf{""" + _brl(vl_aceito) + r"""}
\item \textbf{Estimativa conservadora de recuperacao:} """ + _brl(vl_contestado * 0.5) + r""" (50\% do contestado)
\end{itemize}
\end{highlightcard}
""")

    L.append(r"\newpage" + "\n")

    # ════════════════════════════════════════════════════════════
    # SECTION 6 — READMISSOES
    # ════════════════════════════════════════════════════════════
    L.append(r"\section{Readmissoes}" + "\n")

    # Overall readmission stats
    L.append(r"""
\begin{kpicard}[Indicadores de Readmissao]
\begin{itemize}[leftmargin=1.5em]
\item Taxa de readmissao em 30 dias: \textbf{""" + f"{taxa_readmit:.1f}\\%" + r"""}
\item Readmissoes identificadas: \textbf{""" + f"{readmit_30d:,}".replace(",", ".") + r"""}
\end{itemize}
\end{kpicard}
\vspace{0.3cm}
""")

    # 6.1 Readmission by CID
    L.append(r"\subsection{Taxa de Readmissao por Diagnostico}" + "\n")
    readmit_by_cid = readmit_data.get("by_cid", [])
    if readmit_by_cid:
        L.append(r"\begin{infocard}[CIDs com maior taxa de readmissao em 30 dias]" + "\n")
        L.append(r"\footnotesize" + "\n")
        L.append(r"\begin{tabularx}{\textwidth}{r X r r r}" + "\n")
        L.append(r"\toprule" + "\n")
        L.append(r"\textbf{\#} & \textbf{CID} & \textbf{Total} & \textbf{Readm.} & \textbf{Taxa} \\" + "\n")
        L.append(r"\midrule" + "\n")
        for i, c in enumerate(readmit_by_cid, 1):
            desc = _escape_latex(_truncate(str(c.get("cid_desc", "---")), 40))
            total = _safe_int(c.get("total_adm"))
            n_r = _safe_int(c.get("n_readmissions"))
            rate = _safe_float(c.get("readmit_rate"))
            L.append(f"{i} & {desc} & {total:,} & {n_r:,} & {rate:.1f}\\% \\\\\n".replace(",", "."))
        L.append(r"\bottomrule" + "\n")
        L.append(r"\end{tabularx}" + "\n")
        L.append(r"\end{infocard}" + "\n\n")

    # 6.2 Frequent flyers
    L.append(r"\subsection{Pacientes com 3+ Internacoes em 12 Meses}" + "\n")
    ff_summary = readmit_data.get("frequent_flyers_summary", {})
    ff_n = _safe_int(ff_summary.get("n_patients"))
    ff_total = _safe_int(ff_summary.get("total_admissions"))
    ff_avg = _safe_float(ff_summary.get("avg_admissions"))
    ff_max = _safe_int(ff_summary.get("max_admissions"))

    if ff_n > 0:
        L.append(r"""
\begin{alertcard}[Pacientes Frequentes (Frequent Flyers)]
\begin{itemize}[leftmargin=1.5em]
\item \textbf{""" + f"{ff_n:,}".replace(",", ".") + r"""} pacientes com 3 ou mais internacoes nos ultimos 12 meses
\item Total de internacoes desse grupo: \textbf{""" + f"{ff_total:,}".replace(",", ".") + r"""}
\item Media de internacoes por paciente: \textbf{""" + f"{ff_avg:.1f}" + r"""}
\item Maximo individual: \textbf{""" + f"{ff_max}" + r""" internacoes}
\end{itemize}
\end{alertcard}
\vspace{0.3cm}
""")

    ff_top10 = readmit_data.get("frequent_flyers_top10", [])
    if ff_top10:
        L.append(r"\begin{infocard}[Top 10 pacientes por numero de internacoes]" + "\n")
        L.append(r"\begin{tabularx}{\textwidth}{r r r r}" + "\n")
        L.append(r"\toprule" + "\n")
        L.append(r"\textbf{\#} & \textbf{Paciente ID} & \textbf{Internacoes} & \textbf{Faturamento Total} \\" + "\n")
        L.append(r"\midrule" + "\n")
        for i, p in enumerate(ff_top10, 1):
            pid = _safe_int(p.get("ID_CD_PACIENTE"))
            n_adm = _safe_int(p.get("n_admissions"))
            bill = _brl(_safe_float(p.get("total_billing")))
            L.append(f"{i} & {pid} & {n_adm} & {bill} \\\\\n")
        L.append(r"\bottomrule" + "\n")
        L.append(r"\end{tabularx}" + "\n")
        L.append(r"\end{infocard}" + "\n\n")

    # 6.3 Cost of avoidable readmissions
    readmit_cost = readmit_data.get("readmission_cost", {})
    n_readmit_total = _safe_int(readmit_cost.get("n_readmissions"))
    vl_readmit_total = _safe_float(readmit_cost.get("total_readmission_billing"))
    if vl_readmit_total > 0:
        avoidable_estimate = vl_readmit_total * 0.5  # 50% assumed avoidable
        L.append(r"\subsection{Custo Estimado de Readmissoes Evitaveis}" + "\n")
        L.append(r"""
\begin{highlightcard}[Impacto Financeiro de Readmissoes]
\begin{itemize}[leftmargin=1.5em]
\item Readmissoes em 30 dias identificadas: \textbf{""" + f"{n_readmit_total:,}".replace(",", ".") + r"""}
\item Faturamento total das readmissoes: \textbf{""" + _brl(vl_readmit_total) + r"""}
\item Estimativa de readmissoes evitaveis (50\%): \textbf{""" + _brl(avoidable_estimate) + r"""}
\item Custo medio por readmissao: \textbf{""" + _brl(vl_readmit_total / max(n_readmit_total, 1)) + r"""}
\end{itemize}
\footnotesize{Nota: a taxa de evitabilidade de 50\% e uma estimativa conservadora baseada em literatura internacional.
Programas de gestao de alta e acompanhamento pos-internacao podem reduzir readmissoes em ate 30\%.}
\end{highlightcard}
""")

    L.append(r"\newpage" + "\n")

    # ════════════════════════════════════════════════════════════
    # SECTION 7 — SIMULACAO POR ALGEBRA DE EMBEDDINGS
    # ════════════════════════════════════════════════════════════
    L.append(r"\section{Simulacao por Algebra de Embeddings}" + "\n")

    L.append(r"""
\begin{kpicard}[Sobre esta Secao]
Os embeddings de 128 dimensoes capturam a ``assinatura'' completa de cada internacao:
trajetoria clinica, faturamento, procedimentos, glosas e desfecho.
Operacoes de algebra vetorial permitem comparar internacoes, agrupar riscos e simular cenarios.
\end{kpicard}
\vspace{0.3cm}
""")

    # 7.1 Clusters de risco
    L.append(r"\subsection{Clusters de Risco (HDBSCAN)}" + "\n")
    clusters = emb_results.get("clusters", [])
    if clusters:
        n_cluster_total = sum(c["n_admissions"] for c in clusters)
        L.append(r"""
\begin{infocard}[Agrupamento por densidade --- HDBSCAN]
O algoritmo identificou clusters naturais de internacoes com perfis similares no espaco de embeddings.
Total de internacoes clusterizadas: \textbf{""" + f"{n_cluster_total:,}".replace(",", ".") + r"""}
\end{infocard}
\vspace{0.3cm}
""")
        for i, cl in enumerate(clusters):
            n_adm = cl["n_admissions"]
            avg_los_cl = cl["avg_los"]
            avg_bill = cl["avg_billing"]
            avg_glo = cl["avg_glosa"]
            obit_rate = cl["obito_rate"]
            top_cids = cl.get("top_cids", [])

            # Auto-name cluster
            if obit_rate > 5:
                cl_name = "Alto Risco (Obito)"
                cl_color = "anomred"
            elif avg_los_cl > 20:
                cl_name = "Permanencia Prolongada"
                cl_color = "anomorange"
            elif avg_bill > 100000:
                cl_name = "Alto Custo"
                cl_color = "anomyellow"
            elif avg_glo > avg_bill * 0.1 and avg_bill > 0:
                cl_name = "Alta Glosa"
                cl_color = "anomyellow"
            else:
                cl_name = "Perfil Padrao"
                cl_color = "bradblue"

            cid_text = ", ".join(_escape_latex(_truncate(str(c.get("cid_desc", "?")), 30))
                                for c in top_cids[:3]) if top_cids else "---"

            L.append(f"\\begin{{tcolorbox}}[colback=bradlightgray,colframe={cl_color},"
                     f"fonttitle=\\bfseries\\small,"
                     f"title={{Cluster {i+1}: {_escape_latex(cl_name)} ({n_adm:,} internacoes)}}]\n".replace(",", "."))
            L.append(r"\begin{tabularx}{\textwidth}{X X X X}" + "\n")
            L.append(f"\\textbf{{LOS medio:}} {avg_los_cl:.1f}d & "
                     f"\\textbf{{Fat. medio:}} {_brl(avg_bill)} & "
                     f"\\textbf{{Glosa media:}} {_brl(avg_glo)} & "
                     f"\\textbf{{Taxa obito:}} {obit_rate:.1f}\\% \\\\\n")
            L.append(r"\end{tabularx}" + "\n")
            L.append(f"\\textbf{{CIDs dominantes:}} {cid_text}\n")
            L.append(r"\end{tcolorbox}" + "\n")
            L.append(r"\vspace{0.15cm}" + "\n")

    # 7.2 Pacientes similares (for top anomalies)
    L.append(r"\subsection{Pacientes Similares (Top Anomalias)}" + "\n")
    similar_patients = emb_results.get("similar_patients", {})
    similar_details  = emb_results.get("similar_details", {})
    if similar_patients:
        L.append(r"""
\begin{infocard}[Analise ``E se?'']
Para cada anomalia, buscamos internacoes com trajetoria vetorial mais proxima.
A comparacao permite entender se o desfecho da anomalia e atipico em relacao a pacientes similares.
\end{infocard}
\vspace{0.3cm}
""")
        shown = 0
        for iid, sims in similar_patients.items():
            if shown >= 5:
                break
            if not sims:
                continue
            # Get anomaly info
            anom_info = None
            for a in anomaly_admissions:
                if a.get("ID_CD_INTERNACAO") == iid:
                    anom_info = a
                    break
            if not anom_info:
                continue

            anom_los = _safe_int(anom_info.get("los_dias"))
            anom_bill = _safe_float(anom_info.get("billing", {}).get("vl_total"))

            L.append(f"\\textbf{{Anomalia {iid} (LOS={anom_los}d, Fat={_brl(anom_bill)}):}}\n\n")
            L.append(r"\begin{tabularx}{\textwidth}{r r r r}" + "\n")
            L.append(r"\toprule" + "\n")
            L.append(r"\textbf{ID Similar} & \textbf{Similaridade} & \textbf{LOS} & \textbf{Faturamento} \\" + "\n")
            L.append(r"\midrule" + "\n")
            for sim_iid, sim_score in sims[:3]:
                det = similar_details.get(sim_iid, {})
                s_los = _safe_int(det.get("los_dias"))
                s_bill = _brl(_safe_float(det.get("vl_total")))
                L.append(f"{sim_iid} & {sim_score:.4f} & {s_los}d & {s_bill} \\\\\n")
            L.append(r"\bottomrule" + "\n")
            L.append(r"\end{tabularx}" + "\n")
            L.append(r"\vspace{0.2cm}" + "\n")
            shown += 1

    L.append(r"\newpage" + "\n")

    # ════════════════════════════════════════════════════════════
    # SECTION 8 — RECOMENDACOES ESTRATEGICAS
    # ════════════════════════════════════════════════════════════
    L.append(r"\section{Recomendacoes Estrategicas}" + "\n")

    # Calculate ROI estimates for recommendations
    recovery_val = vl_contestado * 0.5 if vl_contestado > 0 else total_glosado * 0.15
    readmit_savings = vl_readmit_total * 0.3 if vl_readmit_total > 0 else 0
    excess_savings = 0
    if total_excess_days > 0 and total_faturado > 0 and total_internacoes > 0 and avg_los > 0:
        daily_cost = total_faturado / (total_internacoes * avg_los)
        excess_savings = total_excess_days * daily_cost * 0.2  # 20% reducible
    total_opportunity = recovery_val + readmit_savings + excess_savings

    L.append(r"""
\begin{tcolorbox}[colback=bradblue!5,colframe=bradblue,width=\textwidth]
{\large\bfseries Oportunidade Total Estimada: """ + _brl_millions(total_opportunity) + r"""}\\[4pt]
{\small Soma de: recuperacao de glosas + readmissoes evitaveis + reducao de diarias excedentes}
\end{tcolorbox}
\vspace{0.5cm}
""")

    # Quick Wins
    L.append(r"\subsection{Acoes Imediatas (Quick Wins)}" + "\n")
    L.append(r"""
\begin{tcolorbox}[colback=successbox,colframe=bradgreen,fonttitle=\bfseries,title={Prazo: 0--3 meses}]
\begin{enumerate}[leftmargin=2em]
\item \textbf{Auditoria das """ + str(len(anomaly_admissions)) + r""" anomalias prioritarias:}
      Revisar manualmente as internacoes com z-score $>$ 3,0. Potencial de recuperacao imediata
      em faturamento incorreto ou glosas indevidas.
      \textit{ROI estimado: """ + _brl(recovery_val * 0.3) + r"""}

\item \textbf{Padronizacao de codificacao CID:}
      Identificamos inconsistencias na codificacao que impactam a analise de custos e desfechos.
      Treinamento da equipe de codificacao pode reduzir glosas por erro de documentacao.

\item \textbf{Revisao de glosas recorrentes:}
      Os 3 principais motivos de glosa representam a maioria do valor glosado.
      Criar checklists pre-faturamento para esses itens especificos.""")

    if glosa_reasons:
        top_reason = _escape_latex(_truncate(str(glosa_reasons[0].get("motivo", "")), 50))
        if top_reason and top_reason != "---":
            L.append(f"\n      Principal motivo: \\textit{{{top_reason}}}")

    L.append(r"""
\end{enumerate}
\end{tcolorbox}
\vspace{0.3cm}
""")

    # Medium-term
    L.append(r"\subsection{Acoes de Medio Prazo}" + "\n")
    L.append(r"""
\begin{tcolorbox}[colback=warnbox,colframe=bradgold,fonttitle=\bfseries,title={Prazo: 3--12 meses}]
\begin{enumerate}[leftmargin=2em]
\item \textbf{Programa de reducao de readmissoes:}
      Implementar protocolo de alta estruturada e acompanhamento pos-internacao
      para os CIDs com maior taxa de readmissao. Alvo: reduzir readmissao 30d em 20\%.
      \textit{Economia estimada: """ + _brl(readmit_savings) + r"""}

\item \textbf{Gestao de permanencia por CID:}
      Estabelecer metas de LOS por diagnostico baseadas nas medianas identificadas.
      Priorizar os 10 CIDs com maior excesso de permanencia.
      \textit{Economia estimada: """ + _brl(excess_savings) + r"""}

\item \textbf{Comite de analise de anomalias:}
      Reuniao mensal de revisao das anomalias detectadas pelo gemeo digital,
      com participacao de auditoria medica, faturamento e gestao assistencial.

\item \textbf{Protocolo para pacientes frequentes:}
      Programa de gestao de caso para os """ + f"{ff_n}" + r""" pacientes com 3+ internacoes,
      incluindo plano terapeutico individualizado e monitoramento ambulatorial.
\end{enumerate}
\end{tcolorbox}
\vspace{0.3cm}
""")

    # Long-term
    L.append(r"\subsection{Acoes de Longo Prazo}" + "\n")
    L.append(r"""
\begin{tcolorbox}[colback=bradlightblue,colframe=bradblue,fonttitle=\bfseries,title={Prazo: 12--24 meses}]
\begin{enumerate}[leftmargin=2em]
\item \textbf{Alerta preditivo em tempo real:}
      Integrar o modelo de embeddings ao sistema hospitalar para detectar anomalias
      durante a internacao (nao apenas pos-alta), permitindo intervencao precoce.

\item \textbf{Score de risco por paciente:}
      Utilizar os embeddings de pacientes para calcular um score continuo de risco,
      priorizando recursos de auditoria e gestao de caso.

\item \textbf{Benchmarking operacional continuo:}
      Expandir a analise para comparacao entre unidades da rede Bradesco,
      identificando best practices e oportunidades de padronizacao.

\item \textbf{Modelo preditivo de LOS e custo:}
      Treinar modelo supervisionado sobre os embeddings para prever permanencia
      e custo no momento da admissao, alimentando a autorizacao e o planejamento
      de leitos.
\end{enumerate}
\end{tcolorbox}
""")

    # Final summary
    L.append(r"""
\vspace{0.5cm}
\begin{tcolorbox}[colback=bradblue!8,colframe=bradblue,width=\textwidth]
\begin{center}
{\large\bfseries Resumo de Impacto Projetado}\\[6pt]
\begin{tabular}{l r r}
\toprule
\textbf{Iniciativa} & \textbf{Economia Estimada} & \textbf{Prazo} \\
\midrule
Recuperacao de glosas contestaveis & """ + _brl(recovery_val) + r""" & 0--6 meses \\
Reducao de readmissoes evitaveis & """ + _brl(readmit_savings) + r""" & 6--12 meses \\
Otimizacao de permanencia & """ + _brl(excess_savings) + r""" & 6--18 meses \\
\midrule
\textbf{Total} & \textbf{""" + _brl(total_opportunity) + r"""} & \\
\bottomrule
\end{tabular}
\end{center}
\end{tcolorbox}
""")

    # ── Closing ──
    L.append(r"""
\vfill
\begin{center}
{\small\textcolor{bradgray}{
Este relatorio foi gerado automaticamente pela plataforma JCUBE.\\
Os valores apresentados sao estimativas baseadas em modelos de embeddings\\
e devem ser validados pela equipe clinica e de auditoria antes de acao.\\
\textbf{Confidencial} --- Bradesco Saude --- Marco de 2026
}}
\end{center}

\end{document}
""")

    return "".join(L)


# ─────────────────────────────────────────────────────────────────
# PDF Compilation
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
    memory=40960,   # 40 GB RAM
    timeout=7200,   # 2 hour max
)
def generate_report():
    import time
    import os
    import duckdb

    t_start = time.time()
    print("=" * 70)
    print("JCUBE V6 Bradesco Executive Report Generator (Modal)")
    print(f"Source   : {SOURCE_DB}")
    print(f"Weights  : {WEIGHTS_PATH}")
    print(f"DB       : {DB_PATH}")
    print(f"Graph    : {GRAPH_PARQUET}")
    print(f"Output   : {OUTPUT_PDF}")
    print(f"Z thresh : {Z_THRESHOLD}")
    print("=" * 70)

    for p in [GRAPH_PARQUET, WEIGHTS_PATH, DB_PATH]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}")

    # 1. Load twin
    unique_nodes, embeddings, node_to_idx, intern_mask, patient_mask = _load_twin()

    # 2. KPIs
    con = duckdb.connect(str(DB_PATH))
    kpis = _fetch_kpis(con)

    # 3. Operational profile
    profile = _fetch_operational_profile(con)

    # 4. Anomaly detection
    top_ids, anomaly_z, global_z_map, total_anomalies, _vecs, _names = _detect_anomalies(
        embeddings, intern_mask, unique_nodes
    )
    anomaly_admissions = _fetch_anomaly_details(con, top_ids, anomaly_z)
    print(f"    Anomaly admissions to report: {len(anomaly_admissions)}")

    # 5. LOS analysis
    los_data = _fetch_los_analysis(con)

    # 6. Financial analysis
    fin_data = _fetch_financial_analysis(con)

    # 7. Readmissions
    readmit_data = _fetch_readmissions(con)

    # 8. Embedding analysis (clusters + similar patients)
    emb_results = _embedding_analysis(
        embeddings, intern_mask, unique_nodes, node_to_idx,
        top_ids, con
    )

    con.close()

    # 9. Generate LaTeX
    print("[LATEX] Generating LaTeX document ...")
    latex = _generate_latex(
        kpis, profile, anomaly_admissions, los_data,
        fin_data, readmit_data, emb_results, total_anomalies
    )

    # 10. Compile PDF
    _compile_latex(latex, OUTPUT_PDF)

    # Commit volume so changes persist
    data_vol.commit()

    elapsed = time.time() - t_start
    print(f"\nFinished in {elapsed:.1f}s")
    print(f"Report saved to Modal volume jcube-data at: {OUTPUT_PDF}")
    print("Download with:")
    print(f"  modal volume get jcube-data reports/bradesco_executive_v6_2026_03.pdf ./bradesco_executive_v6_2026_03.pdf")
    return OUTPUT_PDF


@app.local_entrypoint()
def main():
    generate_report.remote()
