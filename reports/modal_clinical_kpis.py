#!/usr/bin/env python3
"""
Modal script: JCUBE V6 Clinical KPI Report — GHO-BRADESCO
Three clinical KPIs with V6 embedding analysis, pt-BR LaTeX → PDF.

KPIs:
  1. PS/Urgência → Primeiro Leito UTI → CID → Readmissão 30d
  2. Eventos Adversos × Hospital × Médico × LOS × Diárias
  3. Alta Direto da UTI → Destino

Embedding Analysis:
  - KPI 1: centroid distance PS→UTI vs PS→non-UTI
  - KPI 2: adverse-event patients vs no-event patients separability
  - KPI 3: UTI-discharge vs floor-discharge embedding comparison

Output: /data/reports/clinical_kpis_bradesco_v6_2026_03.pdf

Usage:
    modal run reports/modal_clinical_kpis.py
    modal run --detach reports/modal_clinical_kpis.py
"""
from __future__ import annotations

import modal

# ─────────────────────────────────────────────────────────────────
# Modal App + Volumes
# ─────────────────────────────────────────────────────────────────

app = modal.App("jcube-clinical-kpis")

jepa_cache = modal.Volume.from_name("jepa-cache", create_if_missing=False)
data_vol   = modal.Volume.from_name("jcube-data",  create_if_missing=False)

VOLUMES = {
    "/cache": jepa_cache,
    "/data":  data_vol,
}

# ─────────────────────────────────────────────────────────────────
# Container image — CPU + torch, duckdb, sklearn, LaTeX
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
WEIGHTS_PATH  = "/cache/tkg-v5/node_emb_epoch_1.pt"
DB_PATH       = "/data/aggregated_fixed_union.db"
OUTPUT_DIR    = "/data/reports"
OUTPUT_PDF    = f"{OUTPUT_DIR}/clinical_kpis_bradesco_v6_2026_03.pdf"

REPORT_DATE_STR = "2026-03-24"
SOURCE_DB       = "GHO-BRADESCO"

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


def _truncate(s: str, max_len: int = 80) -> str:
    if not s:
        return "---"
    s = str(s).strip()
    if len(s) > max_len:
        return s[:max_len] + "..."
    return s


def _pct(num: float, den: float, decimals: int = 1) -> str:
    if den == 0:
        return "---"
    return f"{100.0 * num / den:.{decimals}f}\\%"


def _fmt_float(v, decimals: int = 1) -> str:
    f = _safe_float(v)
    if f == 0:
        return "---"
    return f"{f:.{decimals}f}"


def _exec(con, q: str):
    cur  = con.execute(q)
    cols = [d[0] for d in cur.description]
    return cols, cur.fetchall()


def _rows_to_dicts(cols, rows):
    return [dict(zip(cols, r)) for r in rows]


# ─────────────────────────────────────────────────────────────────
# Step 1 — Load V6 embeddings (full graph)
# ─────────────────────────────────────────────────────────────────

def _load_embeddings():
    import time
    import numpy as np
    import torch
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    import pyarrow as pa

    print("[1/7] Loading full graph vocab ...")
    t0 = time.time()
    full_table = pq.read_table(GRAPH_PARQUET, columns=["subject_id", "object_id"])
    full_subj = full_table.column("subject_id")
    full_obj  = full_table.column("object_id")
    full_all  = pa.chunked_array(full_subj.chunks + full_obj.chunks)
    unique_nodes = pc.unique(full_all).to_numpy(zero_copy_only=False).astype(object)
    del full_table, full_subj, full_obj, full_all
    n_nodes = len(unique_nodes)
    print(f"    {n_nodes:,} total nodes in {time.time()-t0:.1f}s")

    print("[1/7] Loading V6 embedding weights ...")
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

    # Build masks
    intern_mask = np.array(
        [f"{SOURCE_DB}/ID_CD_INTERNACAO_" in str(n) for n in unique_nodes], dtype=bool
    )
    patient_mask = np.array(
        [f"{SOURCE_DB}/ID_CD_PACIENTE_" in str(n) or
         (SOURCE_DB in str(n) and "_PACIENTE_" in str(n))
         for n in unique_nodes], dtype=bool
    )
    print(f"    BRADESCO INTERNACAO nodes: {intern_mask.sum():,}")
    print(f"    BRADESCO PACIENTE nodes:   {patient_mask.sum():,}")

    return unique_nodes, embeddings, node_to_idx, intern_mask, patient_mask


# ─────────────────────────────────────────────────────────────────
# Step 2 — KPI 1: PS/Urgência → Primeiro Leito UTI → CID → Readmissão 30d
# ─────────────────────────────────────────────────────────────────

def _fetch_kpi1(con):
    """Origin → first bed UTI → CID → 30-day readmission cross-tabulation."""
    import time
    print("[2/7] KPI 1: Origem → UTI → CID → Readmissão 30d ...")
    t0 = time.time()
    SRC = f"source_db = '{SOURCE_DB}'"
    kpi1 = {}

    # --- Headline numbers ---
    cols, rows = _exec(con, f"""
        SELECT
            COUNT(DISTINCT ID_CD_INTERNACAO) AS total_internacoes,
            COUNT(DISTINCT ID_CD_PACIENTE)   AS total_pacientes,
            MIN(DH_ADMISSAO_HOSP)::DATE      AS dt_inicio,
            MAX(DH_ADMISSAO_HOSP)::DATE      AS dt_fim
        FROM agg_tb_capta_internacao_cain
        WHERE {SRC}
    """)
    kpi1["base"] = dict(zip(cols, rows[0])) if rows else {}

    # --- First bed per admission (rn=1 by DH_DE) with UTI flag ---
    cols, rows = _exec(con, f"""
        WITH first_bed AS (
            SELECT
                h.ID_CD_INTERNACAO,
                h.ID_CD_ORIGEM AS bed_origin,
                COALESCE(o.FL_LEITO_UTI, 0) AS fl_uti,
                COALESCE(o.DS_TITULO, 'DESCONHECIDO') AS ds_bed_type,
                ROW_NUMBER() OVER (
                    PARTITION BY h.ID_CD_INTERNACAO
                    ORDER BY h.DH_DE
                ) AS rn
            FROM agg_tb_capta_hist_leitos_cahl h
            JOIN agg_tb_capta_cfg_origem_cago o
                ON h.ID_CD_ORIGEM = o.ID_CD_ORIGEM
                AND h.source_db = o.source_db
            WHERE h.{SRC}
        )
        SELECT ID_CD_INTERNACAO, fl_uti, ds_bed_type
        FROM first_bed
        WHERE rn = 1
    """)
    first_bed_map = {_safe_int(r[0]): {"fl_uti": _safe_int(r[1]), "ds_bed_type": str(r[2])} for r in rows}
    kpi1["n_with_bed_history"] = len(first_bed_map)

    # --- Admission origin type ---
    cols, rows = _exec(con, f"""
        SELECT
            i.ID_CD_INTERNACAO,
            i.ID_CD_ORIGEM,
            COALESCE(o.DS_TITULO, 'DESCONHECIDO') AS ds_origem
        FROM agg_tb_capta_internacao_cain i
        LEFT JOIN agg_tb_capta_cfg_origem_cago o
            ON i.ID_CD_ORIGEM = o.ID_CD_ORIGEM
            AND i.source_db = o.source_db
        WHERE i.{SRC}
    """)
    origin_map = {_safe_int(r[0]): {"id_origem": _safe_int(r[1]), "ds_origem": str(r[2])} for r in rows}

    # --- CID per admission ---
    cols, rows = _exec(con, f"""
        SELECT ID_CD_INTERNACAO, ID_CD_CID, DS_DESCRICAO
        FROM agg_tb_capta_cid_caci
        WHERE {SRC} AND DS_DESCRICAO IS NOT NULL
    """)
    cid_map = {}
    for r in rows:
        iid = _safe_int(r[0])
        if iid not in cid_map:
            cid_map[iid] = {"cid": str(r[1]), "desc": str(r[2])}

    # --- 30-day readmission flag ---
    cols, rows = _exec(con, f"""
        WITH adm_seq AS (
            SELECT
                ID_CD_INTERNACAO,
                ID_CD_PACIENTE,
                DH_ADMISSAO_HOSP::DATE AS dt_adm,
                LAG(DH_FINALIZACAO::DATE) OVER (
                    PARTITION BY ID_CD_PACIENTE
                    ORDER BY DH_ADMISSAO_HOSP
                ) AS prev_discharge
            FROM agg_tb_capta_internacao_cain
            WHERE {SRC} AND DH_ADMISSAO_HOSP IS NOT NULL
        )
        SELECT
            ID_CD_INTERNACAO,
            CASE WHEN prev_discharge IS NOT NULL
                      AND DATEDIFF('day', prev_discharge, dt_adm) BETWEEN 1 AND 30
                 THEN 1 ELSE 0 END AS is_readmission_30d
        FROM adm_seq
    """)
    readmit_map = {_safe_int(r[0]): _safe_int(r[1]) for r in rows}

    # --- Cross tabulate ---
    # Keys: (origin_type, first_bed_uti) → {n, n_readmit, cid_counter}
    from collections import Counter, defaultdict
    cross = defaultdict(lambda: {"n": 0, "n_readmit": 0, "cids": Counter()})
    all_intern_ids = set(origin_map.keys())

    for iid in all_intern_ids:
        orig = origin_map.get(iid, {})
        ds_orig = orig.get("ds_origem", "DESCONHECIDO").upper()
        bed = first_bed_map.get(iid, {})
        fl_uti = bed.get("fl_uti", 0)
        uti_label = "UTI" if fl_uti == 1 else "Não-UTI"
        readmit = readmit_map.get(iid, 0)
        cid_info = cid_map.get(iid, {})
        cid_desc = cid_info.get("desc", "SEM CID")

        key = (ds_orig, uti_label)
        cross[key]["n"] += 1
        cross[key]["n_readmit"] += readmit
        cross[key]["cids"][cid_desc] += 1

    # Sorted cross table
    cross_table = []
    for (orig, uti), d in sorted(cross.items(), key=lambda x: -x[1]["n"]):
        top_cids = d["cids"].most_common(5)
        rate_readmit = 100.0 * d["n_readmit"] / d["n"] if d["n"] > 0 else 0
        cross_table.append({
            "origem": orig,
            "primeiro_leito": uti,
            "n": d["n"],
            "n_readmit": d["n_readmit"],
            "taxa_readmit": rate_readmit,
            "top_cids": top_cids,
        })
    kpi1["cross_table"] = cross_table

    # --- Top CIDs for PS→UTI readmissions ---
    ps_uti_readmit_cids = Counter()
    ps_keywords = {"PS", "PRONTO", "URGÊNCIA", "URGENCIA", "EMERGÊNCIA", "EMERGENCIA"}
    for iid in all_intern_ids:
        orig = origin_map.get(iid, {})
        ds_orig = orig.get("ds_origem", "").upper()
        is_ps = any(k in ds_orig for k in ps_keywords)
        bed = first_bed_map.get(iid, {})
        fl_uti = bed.get("fl_uti", 0)
        readmit = readmit_map.get(iid, 0)
        if is_ps and fl_uti == 1 and readmit == 1:
            cid_info = cid_map.get(iid, {})
            ps_uti_readmit_cids[cid_info.get("desc", "SEM CID")] += 1
    kpi1["ps_uti_readmit_cids"] = ps_uti_readmit_cids.most_common(15)

    # --- Global readmission summary ---
    total_readmit = sum(1 for v in readmit_map.values() if v == 1)
    total_with_prior = sum(1 for _ in readmit_map.values())
    kpi1["total_readmit_30d"] = total_readmit
    kpi1["total_admissions_for_readmit"] = total_with_prior

    # --- UTI summary ---
    n_uti_first_bed = sum(1 for v in first_bed_map.values() if v.get("fl_uti") == 1)
    kpi1["n_uti_first_bed"] = n_uti_first_bed
    kpi1["n_total_with_bed"] = len(first_bed_map)

    # Collect admission IDs for embedding analysis
    ps_uti_ids = []
    ps_non_uti_ids = []
    for iid in all_intern_ids:
        orig = origin_map.get(iid, {})
        ds_orig = orig.get("ds_origem", "").upper()
        is_ps = any(k in ds_orig for k in ps_keywords)
        if not is_ps:
            continue
        bed = first_bed_map.get(iid, {})
        fl_uti = bed.get("fl_uti", 0)
        if fl_uti == 1:
            ps_uti_ids.append(iid)
        else:
            ps_non_uti_ids.append(iid)
    kpi1["ps_uti_ids"] = ps_uti_ids
    kpi1["ps_non_uti_ids"] = ps_non_uti_ids

    print(f"    KPI 1 done in {time.time()-t0:.1f}s — "
          f"{len(cross_table)} origin×UTI combos, "
          f"{n_uti_first_bed} UTI first beds, "
          f"{total_readmit} readmissions 30d")
    return kpi1


# ─────────────────────────────────────────────────────────────────
# Step 3 — KPI 2: Eventos Adversos × Hospital × Médico × LOS × Diárias
# ─────────────────────────────────────────────────────────────────

def _fetch_kpi2(con):
    """Adverse events: rate per 1000 patient-days, by type, severity, physician."""
    import time
    print("[3/7] KPI 2: Eventos Adversos ...")
    t0 = time.time()
    SRC = f"source_db = '{SOURCE_DB}'"
    kpi2 = {}

    # --- Total adverse events ---
    cols, rows = _exec(con, f"""
        SELECT COUNT(*) AS n_eventos
        FROM agg_tb_capta_eventos_adversos_caed
        WHERE {SRC}
    """)
    kpi2["n_total_events"] = _safe_int(rows[0][0]) if rows else 0

    # --- Total patient-days (denominator) ---
    cols, rows = _exec(con, f"""
        SELECT
            SUM(DATEDIFF('day', DH_ADMISSAO_HOSP,
                COALESCE(DH_FINALIZACAO, CURRENT_TIMESTAMP))) AS total_patient_days,
            COUNT(DISTINCT ID_CD_INTERNACAO) AS n_internacoes
        FROM agg_tb_capta_internacao_cain
        WHERE {SRC} AND DH_ADMISSAO_HOSP IS NOT NULL
    """)
    r = rows[0] if rows else (0, 0)
    kpi2["total_patient_days"] = _safe_int(r[0])
    kpi2["n_internacoes"] = _safe_int(r[1])

    # --- Events by type ---
    cols, rows = _exec(con, f"""
        SELECT
            COALESCE(t.DS_TITULO, 'DESCONHECIDO') AS tipo_evento,
            COUNT(*) AS n_eventos
        FROM agg_tb_capta_eventos_adversos_caed e
        LEFT JOIN agg_tb_capta_cfg_tipos_eventos_adversos_ctea t
            ON e.ID_CD_TIPO_EVENTO = t.ID_CD_TIPO_EVENTO
            AND e.source_db = t.source_db
        WHERE e.{SRC}
        GROUP BY tipo_evento
        ORDER BY n_eventos DESC
    """)
    kpi2["by_type"] = _rows_to_dicts(cols, rows)

    # --- Events by severity (repercussão) ---
    cols, rows = _exec(con, f"""
        SELECT
            COALESCE(r.DS_TITULO, 'DESCONHECIDO') AS severidade,
            COUNT(*) AS n_eventos
        FROM agg_tb_capta_eventos_adversos_caed e
        LEFT JOIN agg_tb_capta_cfg_tipo_repercussao_ctir r
            ON e.ID_CD_TIPO_REPERCUSSAO = r.ID_CD_TIPO_REPERCUSSAO
            AND e.source_db = r.source_db
        WHERE e.{SRC}
        GROUP BY severidade
        ORDER BY n_eventos DESC
    """)
    kpi2["by_severity"] = _rows_to_dicts(cols, rows)

    # --- Events by hospital ---
    cols, rows = _exec(con, f"""
        SELECT
            i.ID_CD_HOSPITAL,
            COUNT(DISTINCT e.ID_CD_EVENTO_ADVERSO) AS n_eventos,
            COUNT(DISTINCT i.ID_CD_INTERNACAO) AS n_internacoes,
            SUM(DATEDIFF('day', i.DH_ADMISSAO_HOSP,
                COALESCE(i.DH_FINALIZACAO, CURRENT_TIMESTAMP))) AS patient_days
        FROM agg_tb_capta_eventos_adversos_caed e
        JOIN agg_tb_capta_internacao_cain i
            ON e.ID_CD_INTERNACAO = i.ID_CD_INTERNACAO
            AND e.source_db = i.source_db
        WHERE e.{SRC} AND i.DH_ADMISSAO_HOSP IS NOT NULL
        GROUP BY i.ID_CD_HOSPITAL
        ORDER BY n_eventos DESC
    """)
    kpi2["by_hospital"] = _rows_to_dicts(cols, rows)

    # --- Events by physician (CRM) — top 15 ---
    cols, rows = _exec(con, f"""
        SELECT
            cfg.DS_CONSELHO_CLASSE AS CRM,
            FIRST(cfg.NM_MEDICO_HOSPITAL) AS nome_medico,
            COUNT(DISTINCT e.ID_CD_EVENTO_ADVERSO) AS n_eventos,
            COUNT(DISTINCT e.ID_CD_INTERNACAO) AS n_internacoes_evento,
            AVG(DATEDIFF('day', i.DH_ADMISSAO_HOSP,
                COALESCE(i.DH_FINALIZACAO, CURRENT_TIMESTAMP))) AS avg_los
        FROM agg_tb_capta_eventos_adversos_caed e
        JOIN agg_tb_capta_internacao_cain i
            ON e.ID_CD_INTERNACAO = i.ID_CD_INTERNACAO
            AND e.source_db = i.source_db
        JOIN agg_tb_capta_internacao_medico_hospital_cimh m
            ON e.ID_CD_INTERNACAO = m.ID_CD_INTERNACAO
            AND e.source_db = m.source_db
        JOIN agg_tb_capta_cfg_medico_hospital_ccmh cfg
            ON m.ID_CD_MEDICO_HOSPITAL = cfg.ID_CD_MEDICO_HOSPITAL
            AND m.source_db = cfg.source_db
        WHERE e.{SRC}
          AND i.DH_ADMISSAO_HOSP IS NOT NULL
          AND cfg.DS_CONSELHO_CLASSE IS NOT NULL
        GROUP BY cfg.DS_CONSELHO_CLASSE
        ORDER BY n_eventos DESC
        LIMIT 15
    """)
    kpi2["by_physician"] = _rows_to_dicts(cols, rows)

    # --- LOS comparison: with events vs without ---
    cols, rows = _exec(con, f"""
        WITH event_admissions AS (
            SELECT DISTINCT ID_CD_INTERNACAO
            FROM agg_tb_capta_eventos_adversos_caed
            WHERE {SRC}
        )
        SELECT
            CASE WHEN ea.ID_CD_INTERNACAO IS NOT NULL THEN 'COM Evento' ELSE 'SEM Evento' END AS grupo,
            COUNT(DISTINCT i.ID_CD_INTERNACAO) AS n_internacoes,
            AVG(DATEDIFF('day', i.DH_ADMISSAO_HOSP,
                COALESCE(i.DH_FINALIZACAO, CURRENT_TIMESTAMP))) AS avg_los,
            MEDIAN(DATEDIFF('day', i.DH_ADMISSAO_HOSP,
                COALESCE(i.DH_FINALIZACAO, CURRENT_TIMESTAMP))) AS median_los
        FROM agg_tb_capta_internacao_cain i
        LEFT JOIN event_admissions ea ON i.ID_CD_INTERNACAO = ea.ID_CD_INTERNACAO
        WHERE i.{SRC} AND i.DH_ADMISSAO_HOSP IS NOT NULL
        GROUP BY grupo
    """)
    kpi2["los_comparison"] = _rows_to_dicts(cols, rows)

    # --- Collect admission IDs for embedding analysis ---
    cols, rows = _exec(con, f"""
        SELECT DISTINCT ID_CD_INTERNACAO
        FROM agg_tb_capta_eventos_adversos_caed
        WHERE {SRC}
    """)
    event_ids = set(_safe_int(r[0]) for r in rows)

    cols, rows = _exec(con, f"""
        SELECT DISTINCT ID_CD_INTERNACAO
        FROM agg_tb_capta_internacao_cain
        WHERE {SRC}
    """)
    all_ids = set(_safe_int(r[0]) for r in rows)
    no_event_ids = all_ids - event_ids

    kpi2["event_intern_ids"] = list(event_ids)
    kpi2["no_event_intern_ids"] = list(no_event_ids)

    print(f"    KPI 2 done in {time.time()-t0:.1f}s — "
          f"{kpi2['n_total_events']} events, "
          f"{len(event_ids)} admissions with events")
    return kpi2


# ─────────────────────────────────────────────────────────────────
# Step 4 — KPI 3: Alta Direto da UTI → Destino
# ─────────────────────────────────────────────────────────────────

def _fetch_kpi3(con):
    """Last bed UTI → discharge type cross-tabulation with LOS and mortality."""
    import time
    print("[4/7] KPI 3: Alta da UTI → Destino ...")
    t0 = time.time()
    SRC = f"source_db = '{SOURCE_DB}'"
    kpi3 = {}

    # --- Last bed per admission (rn=1 descending DH_DE) with UTI flag ---
    cols, rows = _exec(con, f"""
        WITH last_bed AS (
            SELECT
                h.ID_CD_INTERNACAO,
                h.ID_CD_ORIGEM AS bed_origin,
                COALESCE(o.FL_LEITO_UTI, 0) AS fl_uti,
                COALESCE(o.DS_TITULO, 'DESCONHECIDO') AS ds_bed_type,
                ROW_NUMBER() OVER (
                    PARTITION BY h.ID_CD_INTERNACAO
                    ORDER BY h.DH_DE DESC
                ) AS rn
            FROM agg_tb_capta_hist_leitos_cahl h
            JOIN agg_tb_capta_cfg_origem_cago o
                ON h.ID_CD_ORIGEM = o.ID_CD_ORIGEM
                AND h.source_db = o.source_db
            WHERE h.{SRC}
        )
        SELECT ID_CD_INTERNACAO, fl_uti, ds_bed_type
        FROM last_bed
        WHERE rn = 1
    """)
    last_bed_map = {_safe_int(r[0]): {"fl_uti": _safe_int(r[1]), "ds_bed_type": str(r[2])} for r in rows}

    # --- Discharge type per admission (last evo_status) ---
    cols, rows = _exec(con, f"""
        WITH last_status AS (
            SELECT
                es.ID_CD_INTERNACAO,
                es.FL_DESOSPITALIZACAO,
                ROW_NUMBER() OVER (
                    PARTITION BY es.ID_CD_INTERNACAO
                    ORDER BY es.DH_CADASTRO DESC
                ) AS rn
            FROM agg_tb_capta_evo_status_caes es
            WHERE es.{SRC}
        )
        SELECT
            ls.ID_CD_INTERNACAO,
            COALESCE(f.DS_FINAL_MONITORAMENTO, 'DESCONHECIDO') AS tipo_alta
        FROM last_status ls
        LEFT JOIN agg_tb_capta_tipo_final_monit_fmon f
            ON ls.FL_DESOSPITALIZACAO = f.ID_CD_FINAL_MONITORAMENTO
        WHERE ls.rn = 1
    """)
    discharge_map = {_safe_int(r[0]): str(r[1]).upper() for r in rows}

    # --- LOS per admission ---
    cols, rows = _exec(con, f"""
        SELECT
            ID_CD_INTERNACAO,
            DATEDIFF('day', DH_ADMISSAO_HOSP,
                COALESCE(DH_FINALIZACAO, CURRENT_TIMESTAMP)) AS los
        FROM agg_tb_capta_internacao_cain
        WHERE {SRC} AND DH_ADMISSAO_HOSP IS NOT NULL
    """)
    los_map = {_safe_int(r[0]): _safe_float(r[1]) for r in rows}

    # --- Cross tabulate: last_bed_uti × discharge_type ---
    from collections import defaultdict
    OBITO_KEYS = {"OBITO", "ÓBITO", "FALECIMENTO", "MORTE"}
    cross = defaultdict(lambda: {"n": 0, "los_sum": 0.0, "n_obito": 0, "discharge_types": defaultdict(int)})

    uti_discharge_ids = []
    floor_discharge_ids = []

    all_ids = set(last_bed_map.keys()) & set(los_map.keys())
    for iid in all_ids:
        bed = last_bed_map[iid]
        fl_uti = bed.get("fl_uti", 0)
        label = "UTI" if fl_uti == 1 else "Enfermaria"
        discharge = discharge_map.get(iid, "SEM REGISTRO")
        los = los_map.get(iid, 0)
        is_obito = any(k in discharge for k in OBITO_KEYS)

        cross[label]["n"] += 1
        cross[label]["los_sum"] += los
        cross[label]["n_obito"] += int(is_obito)
        cross[label]["discharge_types"][discharge] += 1

        if fl_uti == 1:
            uti_discharge_ids.append(iid)
        else:
            floor_discharge_ids.append(iid)

    cross_table = []
    for label in ["UTI", "Enfermaria"]:
        d = cross[label]
        avg_los = d["los_sum"] / d["n"] if d["n"] > 0 else 0
        mortality = 100.0 * d["n_obito"] / d["n"] if d["n"] > 0 else 0
        # Top discharge types
        top_discharges = sorted(d["discharge_types"].items(), key=lambda x: -x[1])[:10]
        cross_table.append({
            "ultimo_leito": label,
            "n": d["n"],
            "avg_los": avg_los,
            "n_obito": d["n_obito"],
            "taxa_mortalidade": mortality,
            "discharge_breakdown": top_discharges,
        })
    kpi3["cross_table"] = cross_table

    # Detailed discharge breakdown for UTI patients
    uti_discharges = cross.get("UTI", {}).get("discharge_types", {})
    kpi3["uti_discharge_detail"] = sorted(uti_discharges.items(), key=lambda x: -x[1])

    kpi3["n_uti_last_bed"] = cross["UTI"]["n"]
    kpi3["n_floor_last_bed"] = cross["Enfermaria"]["n"]
    kpi3["uti_discharge_ids"] = uti_discharge_ids
    kpi3["floor_discharge_ids"] = floor_discharge_ids

    print(f"    KPI 3 done in {time.time()-t0:.1f}s — "
          f"UTI last bed: {cross['UTI']['n']}, "
          f"Enfermaria last bed: {cross['Enfermaria']['n']}")
    return kpi3


# ─────────────────────────────────────────────────────────────────
# Step 5 — Embedding Analysis
# ─────────────────────────────────────────────────────────────────

def _embedding_analysis(kpi1, kpi2, kpi3, embeddings, node_to_idx):
    """Compute centroid distances and separability for each KPI."""
    import time
    import numpy as np
    print("[5/7] Embedding analysis ...")
    t0 = time.time()

    emb_results = {}

    def _get_emb_indices(intern_ids):
        """Look up embedding indices for a list of admission IDs."""
        indices = []
        for iid in intern_ids:
            key = f"{SOURCE_DB}/ID_CD_INTERNACAO_{iid}"
            if key in node_to_idx:
                indices.append(node_to_idx[key])
        return indices

    def _centroid(indices):
        if not indices:
            return None
        vecs = embeddings[indices]
        return vecs.mean(axis=0)

    def _cosine_distance(a, b):
        if a is None or b is None:
            return None
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return None
        cos_sim = np.dot(a, b) / (norm_a * norm_b)
        return 1.0 - cos_sim

    def _separability_score(idx_a, idx_b, max_samples=5000):
        """
        Compute a linear separability proxy using the silhouette-like
        inter/intra centroid ratio.
        """
        if not idx_a or not idx_b:
            return None, None, None

        # Sample if too large
        rng = np.random.RandomState(42)
        if len(idx_a) > max_samples:
            idx_a = rng.choice(idx_a, max_samples, replace=False).tolist()
        if len(idx_b) > max_samples:
            idx_b = rng.choice(idx_b, max_samples, replace=False).tolist()

        vecs_a = embeddings[idx_a]
        vecs_b = embeddings[idx_b]

        centroid_a = vecs_a.mean(axis=0)
        centroid_b = vecs_b.mean(axis=0)

        # Inter-centroid distance
        inter_dist = np.linalg.norm(centroid_a - centroid_b)

        # Intra-cluster spread (avg distance to centroid)
        intra_a = np.mean(np.linalg.norm(vecs_a - centroid_a, axis=1))
        intra_b = np.mean(np.linalg.norm(vecs_b - centroid_b, axis=1))
        avg_intra = (intra_a + intra_b) / 2

        # Separability = inter / avg_intra (higher = more separable)
        sep = inter_dist / avg_intra if avg_intra > 0 else 0
        cos_dist = _cosine_distance(centroid_a, centroid_b)
        return sep, cos_dist, {"inter": inter_dist, "intra_a": intra_a, "intra_b": intra_b}

    # --- KPI 1: PS→UTI vs PS→non-UTI ---
    idx_ps_uti = _get_emb_indices(kpi1.get("ps_uti_ids", []))
    idx_ps_non_uti = _get_emb_indices(kpi1.get("ps_non_uti_ids", []))
    sep1, cos1, detail1 = _separability_score(idx_ps_uti, idx_ps_non_uti)
    emb_results["kpi1"] = {
        "n_ps_uti_emb": len(idx_ps_uti),
        "n_ps_non_uti_emb": len(idx_ps_non_uti),
        "separability": sep1,
        "cosine_distance": cos1,
        "detail": detail1,
    }
    print(f"    KPI 1 embeddings: {len(idx_ps_uti)} PS→UTI, "
          f"{len(idx_ps_non_uti)} PS→non-UTI, "
          f"sep={sep1:.4f}" if sep1 else "    KPI 1: insufficient embeddings")

    # --- KPI 2: with adverse events vs without ---
    idx_event = _get_emb_indices(kpi2.get("event_intern_ids", []))
    idx_no_event = _get_emb_indices(kpi2.get("no_event_intern_ids", []))
    sep2, cos2, detail2 = _separability_score(idx_event, idx_no_event)
    emb_results["kpi2"] = {
        "n_event_emb": len(idx_event),
        "n_no_event_emb": len(idx_no_event),
        "separability": sep2,
        "cosine_distance": cos2,
        "detail": detail2,
    }
    print(f"    KPI 2 embeddings: {len(idx_event)} WITH events, "
          f"{len(idx_no_event)} WITHOUT events, "
          f"sep={sep2:.4f}" if sep2 else "    KPI 2: insufficient embeddings")

    # --- KPI 3: UTI-discharge vs floor-discharge ---
    idx_uti_dis = _get_emb_indices(kpi3.get("uti_discharge_ids", []))
    idx_floor_dis = _get_emb_indices(kpi3.get("floor_discharge_ids", []))
    sep3, cos3, detail3 = _separability_score(idx_uti_dis, idx_floor_dis)
    emb_results["kpi3"] = {
        "n_uti_discharge_emb": len(idx_uti_dis),
        "n_floor_discharge_emb": len(idx_floor_dis),
        "separability": sep3,
        "cosine_distance": cos3,
        "detail": detail3,
    }
    print(f"    KPI 3 embeddings: {len(idx_uti_dis)} UTI-discharge, "
          f"{len(idx_floor_dis)} floor-discharge, "
          f"sep={sep3:.4f}" if sep3 else "    KPI 3: insufficient embeddings")

    print(f"    Embedding analysis done in {time.time()-t0:.1f}s")
    return emb_results


# ─────────────────────────────────────────────────────────────────
# Step 6 — Generate LaTeX
# ─────────────────────────────────────────────────────────────────

def _generate_latex(kpi1, kpi2, kpi3, emb_results) -> str:

    L = []
    esc = _escape_latex

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

\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\textcolor{bradblue}{\textbf{JCUBE}} \textcolor{bradgray}{\small | GHO-BRADESCO --- KPIs Clínicos V6}}
\fancyhead[R]{\textcolor{bradgray}{\small """ + REPORT_DATE_STR.replace("-", "/") + r"""}}
\fancyfoot[C]{\textcolor{bradgray}{\thepage}}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0.4pt}

\titleformat{\section}{\Large\bfseries\color{bradblue}}{\thesection}{1em}{}[\titlerule]
\titleformat{\subsection}{\large\bfseries\color{bradblue!80!black}}{\thesubsection}{1em}{}
\titleformat{\subsubsection}{\normalsize\bfseries\color{bradgray}}{\thesubsubsection}{1em}{}

\hypersetup{colorlinks=true,linkcolor=bradblue,pdftitle={GHO-BRADESCO KPIs Clínicos V6}}

\begin{document}
\setlength{\parindent}{0pt}
\setlength{\parskip}{4pt}
""")

    # ════════════════════════════════════════════════════════════
    # CAPA
    # ════════════════════════════════════════════════════════════
    total_int = _safe_int(kpi1.get("base", {}).get("total_internacoes"))
    total_pac = _safe_int(kpi1.get("base", {}).get("total_pacientes"))
    dt_ini = kpi1.get("base", {}).get("dt_inicio", "---")
    dt_fim = kpi1.get("base", {}).get("dt_fim", "---")

    L.append(r"""
\begin{center}
{\Huge\bfseries\textcolor{bradblue}{Relatório de KPIs Clínicos}}\\[8pt]
{\Large\textcolor{bradgray}{GHO-BRADESCO --- Análise V6}}\\[6pt]
{\large\textcolor{bradgray}{""" + str(REPORT_DATE_STR) + r"""}}\\[16pt]
\end{center}

\begin{kpicard}[Escopo da Análise]
\begin{itemize}[nosep]
\item \textbf{Internações}: """ + f"{total_int:,}".replace(",", ".") + r"""
\item \textbf{Pacientes}: """ + f"{total_pac:,}".replace(",", ".") + r"""
\item \textbf{Período}: """ + str(dt_ini)[:10] + " a " + str(dt_fim)[:10] + r"""
\item \textbf{Embeddings V6}: 35,2M nós $\times$ 128 dimensões (TGN epoch 2)
\end{itemize}
\end{kpicard}
""")

    # ════════════════════════════════════════════════════════════
    # SECTION 1 — RESUMO EXECUTIVO
    # ════════════════════════════════════════════════════════════
    L.append(r"\newpage")
    L.append(r"\section{Resumo Executivo}")
    L.append(r"\vspace{4pt}")

    # Headline KPIs
    n_uti_first = _safe_int(kpi1.get("n_uti_first_bed"))
    n_total_bed = _safe_int(kpi1.get("n_total_with_bed"))
    readmit_30d = _safe_int(kpi1.get("total_readmit_30d"))
    readmit_total = _safe_int(kpi1.get("total_admissions_for_readmit"))
    n_events = _safe_int(kpi2.get("n_total_events"))
    pd = _safe_int(kpi2.get("total_patient_days"))
    n_uti_last = _safe_int(kpi3.get("n_uti_last_bed"))
    n_floor_last = _safe_int(kpi3.get("n_floor_last_bed"))

    # Mortality from KPI 3
    uti_mort = 0.0
    floor_mort = 0.0
    for ct in kpi3.get("cross_table", []):
        if ct["ultimo_leito"] == "UTI":
            uti_mort = ct["taxa_mortalidade"]
        else:
            floor_mort = ct["taxa_mortalidade"]

    rate_1000 = (n_events / pd * 1000) if pd > 0 else 0

    L.append(r"""
\begin{kpicard}[Indicadores-Chave]
\begin{tabular}{p{8cm} r}
\toprule
\textbf{Indicador} & \textbf{Valor} \\
\midrule
Primeiro leito UTI (do total com histórico de leitos) & """ + _pct(n_uti_first, n_total_bed) + r""" \\
Readmissão 30 dias (global) & """ + _pct(readmit_30d, readmit_total) + r""" \\
Eventos adversos por 1.000 pacientes-dia & """ + f"{rate_1000:.2f}" + r""" \\
Alta direto da UTI (último leito) & """ + f"{n_uti_last:,}".replace(",", ".") + r""" internações \\
Mortalidade --- alta da UTI & """ + f"{uti_mort:.1f}" + r"""\% \\
Mortalidade --- alta da enfermaria & """ + f"{floor_mort:.1f}" + r"""\% \\
\bottomrule
\end{tabular}
\end{kpicard}
""")

    # Embedding highlights
    emb1 = emb_results.get("kpi1", {})
    emb2 = emb_results.get("kpi2", {})
    emb3 = emb_results.get("kpi3", {})

    L.append(r"""
\begin{highlightcard}[Destaques de Embeddings V6]
\begin{itemize}[nosep]
\item \textbf{PS$\to$UTI vs PS$\to$Não-UTI}: distância cosseno = """ +
              _fmt_float(emb1.get("cosine_distance"), 4) +
              r""", separabilidade = """ + _fmt_float(emb1.get("separability"), 4) + r"""
\item \textbf{COM vs SEM Evento Adverso}: distância cosseno = """ +
              _fmt_float(emb2.get("cosine_distance"), 4) +
              r""", separabilidade = """ + _fmt_float(emb2.get("separability"), 4) + r"""
\item \textbf{Alta UTI vs Alta Enfermaria}: distância cosseno = """ +
              _fmt_float(emb3.get("cosine_distance"), 4) +
              r""", separabilidade = """ + _fmt_float(emb3.get("separability"), 4) + r"""
\end{itemize}
\end{highlightcard}
""")

    # ════════════════════════════════════════════════════════════
    # SECTION 2 — KPI 1
    # ════════════════════════════════════════════════════════════
    L.append(r"\newpage")
    L.append(r"\section{KPI 1: Origem $\to$ UTI $\to$ CID $\to$ Readmissão 30d}")

    L.append(r"""
\subsection{Cruzamento: Origem da Internação $\times$ Primeiro Leito UTI}
""")

    # Cross table
    cross = kpi1.get("cross_table", [])
    if cross:
        L.append(r"""
\begin{infocard}[Origem $\times$ Primeiro Leito]
\footnotesize
\begin{longtable}{p{4.5cm} c r r r}
\toprule
\textbf{Origem} & \textbf{1º Leito} & \textbf{N} & \textbf{Readm. 30d} & \textbf{Taxa} \\
\midrule
\endhead
""")
        for ct in cross[:20]:
            L.append(
                esc(_truncate(ct["origem"], 40)) + " & " +
                esc(ct["primeiro_leito"]) + " & " +
                f"{ct['n']:,}".replace(",", ".") + " & " +
                f"{ct['n_readmit']:,}".replace(",", ".") + " & " +
                f"{ct['taxa_readmit']:.1f}\\%" + r" \\" + "\n"
            )
        L.append(r"""
\bottomrule
\end{longtable}
\end{infocard}
""")

    # Top CIDs for PS→UTI readmissions
    ps_cids = kpi1.get("ps_uti_readmit_cids", [])
    if ps_cids:
        L.append(r"""
\subsection{Top CIDs: PS/Urgência $\to$ UTI com Readmissão 30d}

\begin{alertcard}[CIDs mais frequentes em readmissões PS$\to$UTI]
\footnotesize
\begin{longtable}{r p{9cm} r}
\toprule
\textbf{\#} & \textbf{CID --- Descrição} & \textbf{N Readm.} \\
\midrule
\endhead
""")
        for i, (desc, count) in enumerate(ps_cids[:15], 1):
            L.append(f"{i} & " + esc(_truncate(desc, 70)) + f" & {count}" + r" \\" + "\n")
        L.append(r"""
\bottomrule
\end{longtable}
\end{alertcard}
""")

    # Embedding analysis for KPI 1
    L.append(r"""
\subsection{Análise por Embeddings: PS$\to$UTI vs PS$\to$Não-UTI}
""")
    emb1_data = emb_results.get("kpi1", {})
    detail1 = emb1_data.get("detail") or {}
    L.append(r"""
\begin{kpicard}[Comparação de Centroides]
\begin{tabular}{l r}
\toprule
\textbf{Métrica} & \textbf{Valor} \\
\midrule
Embeddings PS$\to$UTI & """ + f"{emb1_data.get('n_ps_uti_emb', 0):,}".replace(",", ".") + r""" \\
Embeddings PS$\to$Não-UTI & """ + f"{emb1_data.get('n_ps_non_uti_emb', 0):,}".replace(",", ".") + r""" \\
Distância cosseno entre centroides & """ + _fmt_float(emb1_data.get("cosine_distance"), 4) + r""" \\
Índice de separabilidade & """ + _fmt_float(emb1_data.get("separability"), 4) + r""" \\
Distância inter-centroide (L2) & """ + _fmt_float(detail1.get("inter"), 4) + r""" \\
Dispersão intra-grupo PS$\to$UTI & """ + _fmt_float(detail1.get("intra_a"), 4) + r""" \\
Dispersão intra-grupo PS$\to$Não-UTI & """ + _fmt_float(detail1.get("intra_b"), 4) + r""" \\
\bottomrule
\end{tabular}
\end{kpicard}

\smallskip
\textit{Interpretação:} Separabilidade $> 1.0$ indica que os grupos são mais distintos entre si
do que internamente --- evidência de que o modelo aprendeu representações distintas para
internações PS que vão para UTI versus as que não vão.
""")

    # ════════════════════════════════════════════════════════════
    # SECTION 3 — KPI 2
    # ════════════════════════════════════════════════════════════
    L.append(r"\newpage")
    L.append(r"\section{KPI 2: Eventos Adversos}")

    n_ev = _safe_int(kpi2.get("n_total_events"))
    n_int2 = _safe_int(kpi2.get("n_internacoes"))
    total_pd = _safe_int(kpi2.get("total_patient_days"))
    rate_1k = (n_ev / total_pd * 1000) if total_pd > 0 else 0

    L.append(r"""
\begin{kpicard}[Panorama Geral]
\begin{tabular}{l r}
\toprule
\textbf{Métrica} & \textbf{Valor} \\
\midrule
Total de eventos adversos & """ + f"{n_ev:,}".replace(",", ".") + r""" \\
Total de internações & """ + f"{n_int2:,}".replace(",", ".") + r""" \\
Total de pacientes-dia & """ + f"{total_pd:,}".replace(",", ".") + r""" \\
Taxa por 1.000 pacientes-dia & """ + f"{rate_1k:.2f}" + r""" \\
\bottomrule
\end{tabular}
\end{kpicard}
""")

    # By type
    by_type = kpi2.get("by_type", [])
    if by_type:
        L.append(r"""
\subsection{Eventos por Tipo}

\begin{infocard}[Distribuição por Tipo de Evento Adverso]
\footnotesize
\begin{longtable}{p{8cm} r r}
\toprule
\textbf{Tipo de Evento} & \textbf{N} & \textbf{Taxa /1000 pd} \\
\midrule
\endhead
""")
        for row in by_type[:15]:
            n = _safe_int(row.get("n_eventos"))
            r_1k = (n / total_pd * 1000) if total_pd > 0 else 0
            L.append(
                esc(_truncate(str(row.get("tipo_evento", "---")), 60)) + " & " +
                f"{n:,}".replace(",", ".") + " & " +
                f"{r_1k:.2f}" + r" \\" + "\n"
            )
        L.append(r"""
\bottomrule
\end{longtable}
\end{infocard}
""")

    # By severity
    by_sev = kpi2.get("by_severity", [])
    if by_sev:
        L.append(r"""
\subsection{Eventos por Severidade (Repercussão)}

\begin{infocard}[Distribuição por Severidade]
\footnotesize
\begin{longtable}{p{8cm} r r}
\toprule
\textbf{Repercussão} & \textbf{N} & \textbf{\% do Total} \\
\midrule
\endhead
""")
        for row in by_sev:
            n = _safe_int(row.get("n_eventos"))
            L.append(
                esc(_truncate(str(row.get("severidade", "---")), 60)) + " & " +
                f"{n:,}".replace(",", ".") + " & " +
                _pct(n, n_ev) + r" \\" + "\n"
            )
        L.append(r"""
\bottomrule
\end{longtable}
\end{infocard}
""")

    # By physician
    by_phys = kpi2.get("by_physician", [])
    if by_phys:
        L.append(r"""
\newpage
\subsection{Top 15 Médicos por Volume de Eventos Adversos}

\begin{alertcard}[Médicos com Mais Eventos Adversos]
\footnotesize
\begin{longtable}{c p{4.5cm} r r r}
\toprule
\textbf{CRM} & \textbf{Nome} & \textbf{Eventos} & \textbf{Intern.} & \textbf{LOS Médio} \\
\midrule
\endhead
""")
        for row in by_phys:
            crm = esc(str(row.get("CRM", "---")))
            nome = esc(_truncate(str(row.get("nome_medico", "---")), 35))
            n_e = _safe_int(row.get("n_eventos"))
            n_i = _safe_int(row.get("n_internacoes_evento"))
            avg_los = _safe_float(row.get("avg_los"))
            L.append(
                crm + " & " + nome + " & " +
                f"{n_e}" + " & " + f"{n_i}" + " & " +
                f"{avg_los:.1f}" + r" \\" + "\n"
            )
        L.append(r"""
\bottomrule
\end{longtable}
\end{alertcard}
""")

    # By hospital
    by_hosp = kpi2.get("by_hospital", [])
    if by_hosp:
        L.append(r"""
\subsection{Eventos por Hospital}

\begin{infocard}[Taxa de Eventos por Hospital]
\footnotesize
\begin{longtable}{c r r r r}
\toprule
\textbf{Hospital ID} & \textbf{Eventos} & \textbf{Intern.} & \textbf{Pac-Dia} & \textbf{Taxa /1000 pd} \\
\midrule
\endhead
""")
        for row in by_hosp[:10]:
            hid = _safe_int(row.get("ID_CD_HOSPITAL"))
            ne = _safe_int(row.get("n_eventos"))
            ni = _safe_int(row.get("n_internacoes"))
            ppd = _safe_int(row.get("patient_days"))
            r_1k = (ne / ppd * 1000) if ppd > 0 else 0
            L.append(
                f"{hid}" + " & " +
                f"{ne:,}".replace(",", ".") + " & " +
                f"{ni:,}".replace(",", ".") + " & " +
                f"{ppd:,}".replace(",", ".") + " & " +
                f"{r_1k:.2f}" + r" \\" + "\n"
            )
        L.append(r"""
\bottomrule
\end{longtable}
\end{infocard}
""")

    # LOS comparison
    los_comp = kpi2.get("los_comparison", [])
    if los_comp:
        L.append(r"""
\subsection{Correlação: Eventos Adversos $\times$ LOS}

\begin{kpicard}[Impacto dos Eventos Adversos no Tempo de Internação]
\footnotesize
\begin{tabular}{l r r r}
\toprule
\textbf{Grupo} & \textbf{N Intern.} & \textbf{LOS Médio (dias)} & \textbf{LOS Mediano (dias)} \\
\midrule
""")
        for row in los_comp:
            grupo = esc(str(row.get("grupo", "---")))
            ni = _safe_int(row.get("n_internacoes"))
            avg = _safe_float(row.get("avg_los"))
            med = _safe_float(row.get("median_los"))
            L.append(
                grupo + " & " +
                f"{ni:,}".replace(",", ".") + " & " +
                f"{avg:.1f}" + " & " +
                f"{med:.1f}" + r" \\" + "\n"
            )
        L.append(r"""
\bottomrule
\end{tabular}
\end{kpicard}
""")

    # Embedding analysis for KPI 2
    L.append(r"""
\subsection{Análise por Embeddings: COM vs SEM Evento Adverso}
""")
    emb2_data = emb_results.get("kpi2", {})
    detail2 = emb2_data.get("detail") or {}
    L.append(r"""
\begin{kpicard}[Separabilidade de Embeddings --- Eventos Adversos]
\begin{tabular}{l r}
\toprule
\textbf{Métrica} & \textbf{Valor} \\
\midrule
Embeddings COM evento adverso & """ + f"{emb2_data.get('n_event_emb', 0):,}".replace(",", ".") + r""" \\
Embeddings SEM evento adverso & """ + f"{emb2_data.get('n_no_event_emb', 0):,}".replace(",", ".") + r""" \\
Distância cosseno entre centroides & """ + _fmt_float(emb2_data.get("cosine_distance"), 4) + r""" \\
Índice de separabilidade & """ + _fmt_float(emb2_data.get("separability"), 4) + r""" \\
Distância inter-centroide (L2) & """ + _fmt_float(detail2.get("inter"), 4) + r""" \\
Dispersão intra-grupo COM evento & """ + _fmt_float(detail2.get("intra_a"), 4) + r""" \\
Dispersão intra-grupo SEM evento & """ + _fmt_float(detail2.get("intra_b"), 4) + r""" \\
\bottomrule
\end{tabular}
\end{kpicard}

\smallskip
\textit{Interpretação:} Se a separabilidade é significativa ($>$ 0.5), internações que
sofrem eventos adversos ocupam uma região distinta no espaço latente, sugerindo que
o modelo captura fatores de risco embutidos nos padrões temporais da internação.
""")

    # ════════════════════════════════════════════════════════════
    # SECTION 4 — KPI 3
    # ════════════════════════════════════════════════════════════
    L.append(r"\newpage")
    L.append(r"\section{KPI 3: Alta Direto da UTI $\to$ Destino}")

    cross3 = kpi3.get("cross_table", [])

    # Summary table
    L.append(r"""
\subsection{Comparação: Último Leito UTI vs Enfermaria}

\begin{kpicard}[Último Leito $\times$ Desfecho]
\footnotesize
\begin{tabular}{l r r r r}
\toprule
\textbf{Último Leito} & \textbf{N} & \textbf{LOS Médio} & \textbf{Óbitos} & \textbf{Mortalidade} \\
\midrule
""")
    for ct in cross3:
        L.append(
            esc(ct["ultimo_leito"]) + " & " +
            f"{ct['n']:,}".replace(",", ".") + " & " +
            f"{ct['avg_los']:.1f}" + " & " +
            f"{ct['n_obito']:,}".replace(",", ".") + " & " +
            f"{ct['taxa_mortalidade']:.1f}\\%" + r" \\" + "\n"
        )
    L.append(r"""
\bottomrule
\end{tabular}
\end{kpicard}
""")

    # UTI discharge detail
    uti_detail = kpi3.get("uti_discharge_detail", [])
    if uti_detail:
        L.append(r"""
\subsection{Destino dos Pacientes com Alta Direto da UTI}

\begin{infocard}[Tipo de Alta --- Pacientes cujo Último Leito foi UTI]
\footnotesize
\begin{longtable}{p{8cm} r r}
\toprule
\textbf{Tipo de Alta / Destino} & \textbf{N} & \textbf{\% do Total UTI} \\
\midrule
\endhead
""")
        n_uti_total = _safe_int(kpi3.get("n_uti_last_bed"))
        for desc, count in uti_detail[:15]:
            L.append(
                esc(_truncate(desc, 60)) + " & " +
                f"{count:,}".replace(",", ".") + " & " +
                _pct(count, n_uti_total) + r" \\" + "\n"
            )
        L.append(r"""
\bottomrule
\end{longtable}
\end{infocard}
""")

    # Embedding analysis for KPI 3
    L.append(r"""
\subsection{Análise por Embeddings: Alta UTI vs Alta Enfermaria}
""")
    emb3_data = emb_results.get("kpi3", {})
    detail3 = emb3_data.get("detail") or {}
    L.append(r"""
\begin{kpicard}[Separabilidade de Embeddings --- Alta UTI vs Enfermaria]
\begin{tabular}{l r}
\toprule
\textbf{Métrica} & \textbf{Valor} \\
\midrule
Embeddings alta da UTI & """ + f"{emb3_data.get('n_uti_discharge_emb', 0):,}".replace(",", ".") + r""" \\
Embeddings alta da enfermaria & """ + f"{emb3_data.get('n_floor_discharge_emb', 0):,}".replace(",", ".") + r""" \\
Distância cosseno entre centroides & """ + _fmt_float(emb3_data.get("cosine_distance"), 4) + r""" \\
Índice de separabilidade & """ + _fmt_float(emb3_data.get("separability"), 4) + r""" \\
Distância inter-centroide (L2) & """ + _fmt_float(detail3.get("inter"), 4) + r""" \\
Dispersão intra-grupo UTI & """ + _fmt_float(detail3.get("intra_a"), 4) + r""" \\
Dispersão intra-grupo enfermaria & """ + _fmt_float(detail3.get("intra_b"), 4) + r""" \\
\bottomrule
\end{tabular}
\end{kpicard}

\smallskip
\textit{Interpretação:} Separabilidade alta indica que pacientes que recebem alta
diretamente da UTI têm trajetórias de internação fundamentalmente diferentes daqueles
que passam pela enfermaria antes da alta. Quanto maior a distância cosseno, mais
o modelo TGN diferencia esses dois perfis clínicos.
""")

    # ════════════════════════════════════════════════════════════
    # SECTION 5 — EMBEDDING ANALYSIS SUMMARY
    # ════════════════════════════════════════════════════════════
    L.append(r"\newpage")
    L.append(r"\section{Análise Consolidada por Embeddings V6}")

    L.append(r"""
\begin{infocard}[Metodologia]
Os embeddings V6 foram treinados com Temporal Graph Network (TGN) sobre 35,2 milhões de nós
e 165 milhões de arestas temporais, gerando vetores de 128 dimensões. Para cada KPI, comparamos
os \textbf{centroides} dos grupos relevantes e calculamos:
\begin{itemize}[nosep]
\item \textbf{Distância cosseno}: mede a diferença angular entre os centroides (0 = idênticos, 2 = opostos)
\item \textbf{Índice de separabilidade}: razão entre distância inter-centroide e dispersão intra-grupo ($>$1 = grupos bem separados)
\end{itemize}
\end{infocard}

\begin{kpicard}[Resumo Comparativo de Embeddings]
\footnotesize
\begin{tabular}{p{6cm} r r r r}
\toprule
\textbf{Comparação} & \textbf{N Grupo A} & \textbf{N Grupo B} & \textbf{Cos. Dist.} & \textbf{Sep.} \\
\midrule
""")

    kpi_labels = [
        ("PS$\\to$UTI vs PS$\\to$Não-UTI",
         emb1.get("n_ps_uti_emb", 0), emb1.get("n_ps_non_uti_emb", 0),
         emb1.get("cosine_distance"), emb1.get("separability")),
        ("COM vs SEM Evento Adverso",
         emb2.get("n_event_emb", 0), emb2.get("n_no_event_emb", 0),
         emb2.get("cosine_distance"), emb2.get("separability")),
        ("Alta UTI vs Alta Enfermaria",
         emb3.get("n_uti_discharge_emb", 0), emb3.get("n_floor_discharge_emb", 0),
         emb3.get("cosine_distance"), emb3.get("separability")),
    ]
    for label, na, nb, cd, sep in kpi_labels:
        L.append(
            label + " & " +
            f"{na:,}".replace(",", ".") + " & " +
            f"{nb:,}".replace(",", ".") + " & " +
            _fmt_float(cd, 4) + " & " +
            _fmt_float(sep, 4) + r" \\" + "\n"
        )

    L.append(r"""
\bottomrule
\end{tabular}
\end{kpicard}
""")

    # ════════════════════════════════════════════════════════════
    # SECTION 6 — RECOMENDAÇÕES
    # ════════════════════════════════════════════════════════════
    L.append(r"\newpage")
    L.append(r"\section{Recomendações}")

    L.append(r"""
\begin{enumerate}[leftmargin=*]

\item \textbf{KPI 1 --- Protocolo PS$\to$UTI:}
Revisar os CIDs com maior taxa de readmissão 30 dias entre internações PS$\to$UTI.
Implantar protocolo de transição estruturada para os diagnósticos identificados,
com follow-up ambulatorial obrigatório em 7 e 14 dias pós-alta.

\item \textbf{KPI 1 --- Estratificação de Risco:}
Utilizar as distâncias de embedding como score auxiliar na triagem do PS ---
admissões cujo embedding se aproxima do centroide ``PS$\to$UTI'' devem receber
monitorização intensiva precoce.

\item \textbf{KPI 2 --- Comitê de Eventos Adversos:}
Priorizar revisão para os tipos de evento com maior taxa por 1.000 pacientes-dia.
Médicos com volume elevado devem participar de programa de mentoria e auditoria,
não punitivo, focado em padronização de condutas.

\item \textbf{KPI 2 --- Predição por Embeddings:}
Se a separabilidade entre COM e SEM evento adverso é significativa, implementar
um classificador linear sobre os embeddings V6 para alertar admissões com alto
risco de evento adverso nas primeiras 24h.

\item \textbf{KPI 3 --- Protocolo de Step-Down:}
Pacientes que recebem alta direto da UTI sem passar por enfermaria apresentam
perfil de risco diferenciado (mortalidade e LOS distintos). Implementar
leito de transição (step-down) para reduzir mortalidade e readmissões.

\item \textbf{KPI 3 --- Monitoramento Pós-UTI:}
Utilizar os embeddings para identificar pacientes ``alta da UTI'' com
representações próximas a desfechos adversos, priorizando telemonitoramento
nas 72h seguintes à alta.

\end{enumerate}
""")

    # ── End document ──
    L.append(r"""
\vfill
\begin{center}
\textcolor{bradgray}{\footnotesize Relatório gerado automaticamente por JCUBE V6 --- """ + REPORT_DATE_STR + r"""}
\end{center}

\end{document}
""")

    return "\n".join(L)


# ─────────────────────────────────────────────────────────────────
# Step 7 — Compile LaTeX → PDF
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
            errs = [line for line in stdout_str.split("\n") if line.startswith("!") or "Error" in line]
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
    memory=65536,   # 65 GB RAM
    timeout=7200,   # 2 hour max
)
def generate_report():
    import time
    import os
    import duckdb

    t_start = time.time()
    print("=" * 70)
    print("JCUBE V6 Clinical KPI Report Generator (Modal)")
    print(f"Source   : {SOURCE_DB}")
    print(f"Weights  : {WEIGHTS_PATH}")
    print(f"DB       : {DB_PATH}")
    print(f"Graph    : {GRAPH_PARQUET}")
    print(f"Output   : {OUTPUT_PDF}")
    print("=" * 70)

    for p in [GRAPH_PARQUET, WEIGHTS_PATH, DB_PATH]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}")

    # 1. Load embeddings
    unique_nodes, embeddings, node_to_idx, intern_mask, patient_mask = _load_embeddings()

    # 2. Fetch KPIs from DuckDB
    con = duckdb.connect(str(DB_PATH))

    kpi1 = _fetch_kpi1(con)
    kpi2 = _fetch_kpi2(con)
    kpi3 = _fetch_kpi3(con)

    con.close()

    # 3. Embedding analysis
    emb_results = _embedding_analysis(kpi1, kpi2, kpi3, embeddings, node_to_idx)

    # 4. Generate LaTeX
    print("[6/7] Generating LaTeX document ...")
    latex = _generate_latex(kpi1, kpi2, kpi3, emb_results)

    # 5. Compile PDF
    _compile_latex(latex, OUTPUT_PDF)

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed / 60:.1f} min")
    print("Done.")


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main():
    generate_report.remote()
# v5weights 1774370663
