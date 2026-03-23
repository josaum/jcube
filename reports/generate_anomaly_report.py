#!/usr/bin/env python3
"""
JCUBE Anomaly Report Generator
Generates a comprehensive PDF anomaly report from Digital Twin embeddings
and DuckDB data, covering ALL hospitals and ALL sources.
"""

from __future__ import annotations

import os
import sys
import time
import subprocess
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import duckdb
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow as pa

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
BASE_DIR = Path("/Users/josaum/projects/jcube")
GRAPH_PARQUET = BASE_DIR / "data" / "jcube_graph.parquet"
WEIGHTS_PATH = BASE_DIR / "data" / "weights" / "node_emb_epoch_2.pt"
DB_PATH = BASE_DIR / "data" / "aggregated_fixed_union.db"
REPORT_DATE = datetime(2026, 3, 23)
WINDOW_DAYS = 30
START_DATE = REPORT_DATE - timedelta(days=WINDOW_DAYS)
OUTPUT_PDF = BASE_DIR / "reports" / "anomaly_report_2026_03.pdf"
Z_THRESHOLD = 2.0


# ─────────────────────────────────────────────────────────────────
# STEP 1 – Load Digital Twin Embeddings
# ─────────────────────────────────────────────────────────────────

def load_twin():
    import torch
    print("[1/5] Loading node vocabulary from graph parquet …")
    t0 = time.time()
    table = pq.read_table(GRAPH_PARQUET, columns=["subject_id", "object_id"])
    subj = table.column("subject_id")
    obj = table.column("object_id")
    all_nodes = pa.chunked_array(subj.chunks + obj.chunks)
    unique_nodes = pc.unique(all_nodes).to_numpy(zero_copy_only=False).astype(object)
    del table, subj, obj, all_nodes
    print(f"    {len(unique_nodes):,} unique nodes in {time.time()-t0:.1f}s")

    print("[1/5] Loading embedding weights …")
    t1 = time.time()
    state = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
    if isinstance(state, torch.Tensor):
        embeddings = state.numpy().astype(np.float32)
    elif isinstance(state, dict) and "weight" in state:
        embeddings = state["weight"].numpy().astype(np.float32)
    else:
        embeddings = list(state.values())[0].numpy().astype(np.float32)
    print(f"    Embeddings shape: {embeddings.shape} in {time.time()-t1:.1f}s")

    assert len(unique_nodes) == embeddings.shape[0]

    node_to_idx = {str(n): i for i, n in enumerate(unique_nodes)}

    internacao_mask = np.array(
        ["ID_CD_INTERNACAO_" in str(n) for n in unique_nodes], dtype=bool
    )
    print(f"    INTERNACAO nodes: {internacao_mask.sum():,}")
    return unique_nodes, embeddings, node_to_idx, internacao_mask


# ─────────────────────────────────────────────────────────────────
# STEP 2 – Anomaly Detection via Z-Score
# ─────────────────────────────────────────────────────────────────

def detect_anomalies_zscore(embeddings, internacao_mask, unique_nodes):
    print("[2/5] Computing anomaly z-scores …")
    t0 = time.time()
    vecs = embeddings[internacao_mask]
    names = unique_nodes[internacao_mask]

    centroid = vecs.mean(axis=0)
    dists = np.linalg.norm(vecs - centroid, axis=1)

    mean_d = dists.mean()
    std_d = dists.std()
    z_scores = (dists - mean_d) / (std_d + 1e-9)

    anomaly_mask = z_scores > Z_THRESHOLD
    anomaly_names = names[anomaly_mask]
    anomaly_z = z_scores[anomaly_mask]

    order = np.argsort(-anomaly_z)
    anomaly_names = anomaly_names[order]
    anomaly_z = anomaly_z[order]

    print(f"    {anomaly_mask.sum():,} anomalies (z>{Z_THRESHOLD}) found in {time.time()-t0:.1f}s")

    internacao_ids = []
    for n in anomaly_names:
        s = str(n)
        try:
            iid = int(s.split("ID_CD_INTERNACAO_")[1])
            internacao_ids.append(iid)
        except Exception:
            internacao_ids.append(None)

    return internacao_ids, anomaly_z


# ─────────────────────────────────────────────────────────────────
# STEP 3 – Batch similar admissions via cosine similarity
# ─────────────────────────────────────────────────────────────────

def batch_find_similar(embeddings, node_to_idx, internacao_mask, unique_nodes,
                       target_iids: list[int], k: int = 5) -> dict[int, list[tuple[str, float]]]:
    """Pre-compute k most similar internacoes for a batch of target IDs."""
    print(f"    Pre-computing similarities for {len(target_iids)} anomalies …")
    t0 = time.time()
    int_vecs = embeddings[internacao_mask].astype(np.float32)
    int_names = unique_nodes[internacao_mask]

    # Normalize all internacao vectors once
    norms = np.linalg.norm(int_vecs, axis=1, keepdims=True).clip(min=1e-8)
    int_vecs_norm = int_vecs / norms

    results = {}
    for iid in target_iids:
        key = f"ID_CD_INTERNACAO_{iid}"
        if key not in node_to_idx:
            results[iid] = []
            continue
        idx_global = node_to_idx[key]
        # find local index in internacao_mask
        qvec = embeddings[idx_global].astype(np.float32)
        qnorm = np.linalg.norm(qvec).clip(min=1e-8)
        qvec_norm = qvec / qnorm
        sims = int_vecs_norm @ qvec_norm  # (N_intern,)
        top = np.argsort(-sims)[:k+1]
        sim_list = []
        for i in top:
            name = str(int_names[i])
            if name == key:
                continue
            sim_list.append((name, float(sims[i])))
            if len(sim_list) >= k:
                break
        results[iid] = sim_list

    print(f"    Done in {time.time()-t0:.1f}s")
    return results


# ─────────────────────────────────────────────────────────────────
# STEP 4 – DuckDB: Fetch full details
# ─────────────────────────────────────────────────────────────────

def fetch_admission_details(internacao_ids: list[int], anomaly_z: np.ndarray) -> list[dict]:
    print(f"[3/5] Fetching DuckDB details for {len(internacao_ids)} anomalies …")
    t0 = time.time()

    valid_ids = [i for i in internacao_ids if i is not None]
    z_map = {iid: float(z) for iid, z in zip(internacao_ids, anomaly_z) if iid is not None}

    # Use a fresh read-write connection with temp tables
    con_rw = duckdb.connect(str(DB_PATH))
    con_rw.execute("CREATE OR REPLACE TEMP TABLE tmp_anomaly_ids (iid INTEGER)")
    # batch insert
    batch_size = 1000
    for i in range(0, len(valid_ids), batch_size):
        batch = valid_ids[i:i+batch_size]
        vals = ",".join(f"({v})" for v in batch)
        con_rw.execute(f"INSERT INTO tmp_anomaly_ids VALUES {vals}")

    # ── Core internacao data (last 30 days) ──
    q_inter = f"""
    SELECT i.ID_CD_INTERNACAO, i.ID_CD_PACIENTE, i.source_db,
        i.DH_ADMISSAO_HOSP, i.DH_FINALIZACAO,
        i.DS_DESCRICAO, i.DS_HISTORICO, i.DS_MOTIVO,
        i.DS_CONDUTA_INTERNACAO, i.DS_DESCRICAO_EVOLUCAO,
        CASE WHEN i.DH_FINALIZACAO IS NOT NULL
            THEN DATEDIFF('day', i.DH_ADMISSAO_HOSP::DATE, i.DH_FINALIZACAO::DATE)
            ELSE DATEDIFF('day', i.DH_ADMISSAO_HOSP::DATE, DATE '2026-03-23')
        END AS LOS_DIAS,
        i.NR_SENHA, i.NR_GUIA_AUTORIZACAO, i.ID_CD_HOSPITAL, i.IN_SITUACAO
    FROM agg_tb_capta_internacao_cain i
    JOIN tmp_anomaly_ids t ON i.ID_CD_INTERNACAO = t.iid
    WHERE i.DH_ADMISSAO_HOSP >= '{START_DATE.strftime('%Y-%m-%d')}'
      AND i.DH_ADMISSAO_HOSP <= '{REPORT_DATE.strftime('%Y-%m-%d')}'
    ORDER BY i.source_db, i.ID_CD_INTERNACAO
    """
    cur = con_rw.execute(q_inter)
    inter_cols = [d[0] for d in cur.description]
    inter_rows = cur.fetchall()

    if not inter_rows:
        print("    No admissions in 30-day window — using most recent 2000 anomalies …")
        q_inter2 = f"""
        SELECT i.ID_CD_INTERNACAO, i.ID_CD_PACIENTE, i.source_db,
            i.DH_ADMISSAO_HOSP, i.DH_FINALIZACAO,
            i.DS_DESCRICAO, i.DS_HISTORICO, i.DS_MOTIVO,
            i.DS_CONDUTA_INTERNACAO, i.DS_DESCRICAO_EVOLUCAO,
            CASE WHEN i.DH_FINALIZACAO IS NOT NULL
                THEN DATEDIFF('day', i.DH_ADMISSAO_HOSP::DATE, i.DH_FINALIZACAO::DATE)
                ELSE 0 END AS LOS_DIAS,
            i.NR_SENHA, i.NR_GUIA_AUTORIZACAO, i.ID_CD_HOSPITAL, i.IN_SITUACAO
        FROM agg_tb_capta_internacao_cain i
        JOIN tmp_anomaly_ids t ON i.ID_CD_INTERNACAO = t.iid
        WHERE i.DH_ADMISSAO_HOSP > '2000-01-01'
        ORDER BY i.DH_ADMISSAO_HOSP DESC
        LIMIT 2000
        """
        cur = con_rw.execute(q_inter2)
        inter_cols = [d[0] for d in cur.description]
        inter_rows = cur.fetchall()

    admissions = [dict(zip(inter_cols, r)) for r in inter_rows]
    for a in admissions:
        a["Z_SCORE"] = z_map.get(a["ID_CD_INTERNACAO"], 2.01)

    # Create a temp table of VALID internacao ids (those found in internacao table)
    valid_found = [a["ID_CD_INTERNACAO"] for a in admissions]
    con_rw.execute("CREATE OR REPLACE TEMP TABLE tmp_valid_ids (iid INTEGER)")
    for i in range(0, len(valid_found), batch_size):
        batch = valid_found[i:i+batch_size]
        if batch:
            vals = ",".join(f"({v})" for v in batch)
            con_rw.execute(f"INSERT INTO tmp_valid_ids VALUES {vals}")

    if not valid_found:
        return admissions

    # ── Fatura (billing) ──
    print("    Fetching billing data …")
    q_fat = """
    SELECT f.ID_CD_INTERNACAO, f.source_db,
        SUM(f.VL_TOTAL) AS vl_total,
        SUM(f.VL_RH) AS vl_rh, SUM(f.VL_MAT) AS vl_mat,
        SUM(f.VL_MED) AS vl_med, SUM(f.VL_EQP) AS vl_eqp,
        SUM(f.VL_OPME) AS vl_opme, SUM(f.VL_SADT) AS vl_sadt,
        SUM(f.VL_TAXA) AS vl_taxa,
        SUM(f.VL_GLOSA_FECHAMENTO) AS vl_glosa_total,
        SUM(f.VL_LIQUIDO_FAT) AS vl_liquido,
        SUM(f.VL_DIVERGENCIA) AS vl_divergencia,
        COUNT(*) AS n_faturas
    FROM agg_tb_fatura_fatu f
    JOIN tmp_valid_ids t ON f.ID_CD_INTERNACAO = t.iid
    GROUP BY f.ID_CD_INTERNACAO, f.source_db
    """
    fat_rows = con_rw.execute(q_fat).fetchall()
    fat_cols = [d[0] for d in con_rw.execute(q_fat).description]
    fatura_map = {dict(zip(fat_cols, r))["ID_CD_INTERNACAO"]: dict(zip(fat_cols, r)) for r in fat_rows}

    # ── Fatura itens ──
    print("    Fetching fatura items …")
    q_fit = """
    SELECT fi.ID_CD_INTERNACAO,
        COUNT(*) AS n_itens,
        SUM(fi.VL_TOTAL_FATURADO) AS vl_total_itens,
        SUM(CASE WHEN fi.FL_GLOSA = 'S' THEN 1 ELSE 0 END) AS n_itens_glosados,
        COUNT(DISTINCT fi.FL_TIPO_PRODUTO) AS n_tipos_produto
    FROM agg_tb_fatura_itens_fait fi
    JOIN tmp_valid_ids t ON fi.ID_CD_INTERNACAO = t.iid
    GROUP BY fi.ID_CD_INTERNACAO
    """
    fit_rows = con_rw.execute(q_fit).fetchall()
    fit_cols = [d[0] for d in con_rw.execute(q_fit).description]
    fit_map = {dict(zip(fit_cols, r))["ID_CD_INTERNACAO"]: dict(zip(fit_cols, r)) for r in fit_rows}

    # ── Glosas ──
    print("    Fetching glosa data …")
    q_glo = """
    SELECT g.ID_CD_INTERNACAO, g.source_db,
        COUNT(*) AS n_glosas,
        SUM(g.VL_GLOSADO) AS vl_glosado_total,
        SUM(CASE WHEN g.FL_ACEITE_GLOSA = 'S' THEN g.VL_GLOSADO ELSE 0 END) AS vl_aceito,
        SUM(CASE WHEN g.FL_ACEITE_GLOSA = 'N' THEN g.VL_GLOSADO ELSE 0 END) AS vl_recusado
    FROM agg_tb_fatura_glosa_fatg g
    JOIN tmp_valid_ids t ON g.ID_CD_INTERNACAO = t.iid
    GROUP BY g.ID_CD_INTERNACAO, g.source_db
    """
    glo_rows = con_rw.execute(q_glo).fetchall()
    glo_cols = [d[0] for d in con_rw.execute(q_glo).description]
    glosa_map = {dict(zip(glo_cols, r))["ID_CD_INTERNACAO"]: dict(zip(glo_cols, r)) for r in glo_rows}

    # ── Negociações ──
    print("    Fetching negotiation data …")
    q_neg = """
    SELECT n.ID_CD_INTERNACAO,
        COUNT(*) AS n_negociacoes,
        SUM(n.VL_TOTAL) AS vl_negociado_total,
        STRING_AGG(DISTINCT CAST(n.FL_TIPO_NEGOCIACAO AS VARCHAR), ', ') AS tipos_negociacao
    FROM agg_tb_capta_negociacoes_auditoria_cnau n
    JOIN tmp_valid_ids t ON n.ID_CD_INTERNACAO = t.iid
    GROUP BY n.ID_CD_INTERNACAO
    """
    neg_rows = con_rw.execute(q_neg).fetchall()
    neg_cols = [d[0] for d in con_rw.execute(q_neg).description]
    neg_map = {dict(zip(neg_cols, r))["ID_CD_INTERNACAO"]: dict(zip(neg_cols, r)) for r in neg_rows}

    # ── CIDs ──
    print("    Fetching CID data …")
    q_cid = """
    SELECT c.ID_CD_INTERNACAO,
        STRING_AGG(DISTINCT COALESCE(c.DS_DESCRICAO,'?'), ' | ') AS cids,
        COUNT(*) AS n_cids,
        STRING_AGG(CASE WHEN c.IN_PRINCIPAL = 'S' THEN c.DS_DESCRICAO END, ', ') AS cid_principal
    FROM agg_tb_capta_cid_caci c
    JOIN tmp_valid_ids t ON c.ID_CD_INTERNACAO = t.iid
    GROUP BY c.ID_CD_INTERNACAO
    """
    cid_rows = con_rw.execute(q_cid).fetchall()
    cid_cols = [d[0] for d in con_rw.execute(q_cid).description]
    cid_map = {dict(zip(cid_cols, r))["ID_CD_INTERNACAO"]: dict(zip(cid_cols, r)) for r in cid_rows}

    # ── Procedimentos ──
    print("    Fetching procedures …")
    q_proc = """
    SELECT p.ID_CD_INTERNACAO, COUNT(*) AS n_procedimentos,
        0 AS vl_proc_total
    FROM agg_tb_fatura_procedimentos_fapr p
    JOIN tmp_valid_ids t ON p.ID_CD_INTERNACAO = t.iid
    GROUP BY p.ID_CD_INTERNACAO
    """
    proc_rows = con_rw.execute(q_proc).fetchall()
    proc_cols = [d[0] for d in con_rw.execute(q_proc).description]
    proc_map = {dict(zip(proc_cols, r))["ID_CD_INTERNACAO"]: dict(zip(proc_cols, r)) for r in proc_rows}

    # ── Auditoria (RAH) ──
    print("    Fetching audit data …")
    q_rah = """
    SELECT r.ID_CD_INTERNACAO,
        COUNT(*) AS n_auditorias,
        MAX(r.DH_ADMISSAO) AS ultima_auditoria
    FROM agg_tb_formulario_rah_completo_frco r
    JOIN tmp_valid_ids t ON r.ID_CD_INTERNACAO = t.iid
    GROUP BY r.ID_CD_INTERNACAO
    """
    rah_rows = con_rw.execute(q_rah).fetchall()
    rah_cols = [d[0] for d in con_rw.execute(q_rah).description]
    rah_map = {dict(zip(rah_cols, r))["ID_CD_INTERNACAO"]: dict(zip(rah_cols, r)) for r in rah_rows}

    # ── Evolução clínica ──
    print("    Fetching clinical notes …")
    q_evo = """
    SELECT e.ID_CD_INTERNACAO, COUNT(*) AS n_evolucoes,
        MAX(e.DH_EVOLUCAO) AS ultima_evolucao
    FROM agg_tb_capta_evolucao_caev e
    JOIN tmp_valid_ids t ON e.ID_CD_INTERNACAO = t.iid
    GROUP BY e.ID_CD_INTERNACAO
    """
    evo_rows = con_rw.execute(q_evo).fetchall()
    evo_cols = [d[0] for d in con_rw.execute(q_evo).description]
    evo_map = {dict(zip(evo_cols, r))["ID_CD_INTERNACAO"]: dict(zip(evo_cols, r)) for r in evo_rows}

    # ── Eventos adversos ──
    q_ev = """
    SELECT ev.ID_CD_INTERNACAO, COUNT(*) AS n_eventos
    FROM agg_tb_capta_eventos_adversos_caed ev
    JOIN tmp_valid_ids t ON ev.ID_CD_INTERNACAO = t.iid
    GROUP BY ev.ID_CD_INTERNACAO
    """
    ev_rows = con_rw.execute(q_ev).fetchall()
    ev_cols = [d[0] for d in con_rw.execute(q_ev).description]
    ev_map = {dict(zip(ev_cols, r))["ID_CD_INTERNACAO"]: dict(zip(ev_cols, r)) for r in ev_rows}

    # ── OPME ──
    opme_map = {}
    try:
        q_opme = """
        SELECT op.ID_CD_INTERNACAO, COUNT(*) AS n_opme
        FROM agg_tb_capta_anexos_opme_caop op
        JOIN tmp_valid_ids t ON op.ID_CD_INTERNACAO = t.iid
        GROUP BY op.ID_CD_INTERNACAO
        """
        opme_rows = con_rw.execute(q_opme).fetchall()
        opme_cols = [d[0] for d in con_rw.execute(q_opme).description]
        opme_map = {dict(zip(opme_cols, r))["ID_CD_INTERNACAO"]: dict(zip(opme_cols, r)) for r in opme_rows}
    except Exception as e:
        print(f"    OPME query skipped: {e}")

    # ── Hospital names ──
    hosp_map = {}
    try:
        q_hosp = "SELECT ID_CD_HOSPITAL, DS_DESCRICAO AS NM_HOSPITAL, source_db FROM agg_tb_crm_hospitais_crho"
        hosp_rows = con_rw.execute(q_hosp).fetchall()
        hosp_cols = [d[0] for d in con_rw.execute(q_hosp).description]
        for r in hosp_rows:
            d = dict(zip(hosp_cols, r))
            k = (d.get("ID_CD_HOSPITAL"), d.get("source_db"))
            hosp_map[k] = d.get("NM_HOSPITAL", "")
    except Exception:
        pass

    # ── Merge ──
    for a in admissions:
        iid = a["ID_CD_INTERNACAO"]
        src = a.get("source_db", "")
        a["fatura"] = fatura_map.get(iid, {})
        a["fatura_itens"] = fit_map.get(iid, {})
        a["glosa"] = glosa_map.get(iid, {})
        a["negociacoes"] = neg_map.get(iid, {})
        a["cids"] = cid_map.get(iid, {})
        a["procedimentos"] = proc_map.get(iid, {})
        a["auditoria"] = rah_map.get(iid, {})
        a["evolucao"] = evo_map.get(iid, {})
        a["eventos_adversos"] = ev_map.get(iid, {})
        a["opme"] = opme_map.get(iid, {})
        a["nm_hospital"] = hosp_map.get((a.get("ID_CD_HOSPITAL"), src), f"Hospital #{a.get('ID_CD_HOSPITAL','?')}")

    con_rw.close()
    print(f"    Done in {time.time()-t0:.1f}s — {len(admissions)} admissions enriched")
    return admissions


# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────

def fmt_date(d) -> str:
    if d is None:
        return "—"
    try:
        if isinstance(d, str):
            return d[:10]
        return d.strftime("%d/%m/%Y")
    except Exception:
        return str(d)[:10]

def safe_int(v, default=0) -> int:
    try:
        return int(v) if v is not None else default
    except Exception:
        return default

def safe_float(v, default=0.0) -> float:
    try:
        return float(v) if v is not None else default
    except Exception:
        return default

def escape_latex(s: str) -> str:
    if not s:
        return ""
    s = str(s)
    replacements = [
        ("\\", "\\textbackslash{}"),
        ("&", "\\&"),
        ("%", "\\%"),
        ("$", "\\$"),
        ("#", "\\#"),
        ("_", "\\_"),
        ("{", "\\{"),
        ("}", "\\}"),
        ("~", "\\textasciitilde{}"),
        ("^", "\\textasciicircum{}"),
        ("<", "\\textless{}"),
        (">", "\\textgreater{}"),
        ("|", "\\textbar{}"),
        ("\r\n", " "),
        ("\n", " "),
        ("\r", " "),
        ("\t", " "),
    ]
    for old, new in replacements:
        s = s.replace(old, new)
    return s

def brl(v) -> str:
    f = safe_float(v)
    if f == 0:
        return "---"
    return "R\\$ {:,.2f}".format(f).replace(",","X").replace(".",",").replace("X",".")

def z_color(z: float) -> str:
    if z >= 5:
        return "anomred"
    elif z >= 3:
        return "anomorange"
    else:
        return "anomyellow"

def z_label(z: float) -> str:
    if z >= 5:
        return "CR\\'{I}TICO"
    elif z >= 3:
        return "ALTO"
    else:
        return "MODERADO"

def truncate(s: str, max_len: int = 200) -> str:
    if not s:
        return "---"
    s = str(s).strip()
    if len(s) > max_len:
        return s[:max_len] + "..."
    return s


# ─────────────────────────────────────────────────────────────────
# STEP 5 – Generate LaTeX
# ─────────────────────────────────────────────────────────────────

def generate_latex(admissions: list[dict], similar_map: dict) -> str:
    now_str = "23 de mar\\c{c}o de 2026"

    sources: dict[str, list[dict]] = {}
    for a in admissions:
        src = a.get("source_db", "DESCONHECIDO")
        sources.setdefault(src, []).append(a)

    total_anomalies = len(admissions)
    total_critical = sum(1 for a in admissions if a["Z_SCORE"] >= 5)
    total_high = sum(1 for a in admissions if 3 <= a["Z_SCORE"] < 5)
    total_moderate = sum(1 for a in admissions if 2 <= a["Z_SCORE"] < 3)
    total_vl_glosa = sum(safe_float(a["glosa"].get("vl_glosado_total")) for a in admissions)
    total_vl_fatura = sum(safe_float(a["fatura"].get("vl_total")) for a in admissions)
    avg_los = float(np.mean([safe_float(a.get("LOS_DIAS", 0)) for a in admissions])) if admissions else 0.0

    L = []  # lines list

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
\usepackage{hyperref}
\usepackage{amsmath}
\usepackage{microtype}
\usepackage{enumitem}

\definecolor{jcubeblue}{RGB}{0,74,134}
\definecolor{jcubegray}{RGB}{80,80,80}
\definecolor{anomred}{RGB}{180,20,20}
\definecolor{anomorange}{RGB}{220,100,0}
\definecolor{anomyellow}{RGB}{160,120,0}
\definecolor{lightgray}{RGB}{248,248,248}
\definecolor{darkblue}{RGB}{0,50,100}
\definecolor{lightblue}{RGB}{220,235,250}

\tcbuselibrary{skins,breakable}
\newtcolorbox{anomalycard}[2][]{%
  breakable, enhanced,
  colback=lightgray,
  colframe=#2,
  fonttitle=\bfseries\footnotesize,
  title={#1},
  left=5pt, right=5pt, top=4pt, bottom=4pt,
  boxrule=1.5pt,
  before upper={\setlength{\parskip}{2pt}}
}

\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\textcolor{jcubeblue}{\textbf{JCUBE Digital Twin}} \textcolor{jcubegray}{\small | Relat\'{o}rio de Anomalias}}
\fancyhead[R]{\textcolor{jcubegray}{\small 23/03/2026}}
\fancyfoot[C]{\textcolor{jcubegray}{\thepage}}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0.4pt}

\titleformat{\section}{\large\bfseries\color{jcubeblue}}{\thesection}{1em}{}[\titlerule]
\titleformat{\subsection}{\normalsize\bfseries\color{darkblue}}{\thesubsection}{1em}{}

\hypersetup{colorlinks=true,linkcolor=jcubeblue,pdftitle={JCUBE Anomalias}}

\begin{document}
\setlength{\parindent}{0pt}
\setlength{\parskip}{3pt}
""")

    # ── Title Page ──
    L.append(r"""\begin{titlepage}
\begin{center}
\vspace*{1.5cm}
{\Huge\bfseries\textcolor{jcubeblue}{JCUBE}}\\[0.2cm]
{\large\textcolor{jcubegray}{Digital Twin Analytics Platform}}\\[1.2cm]
\begin{tcolorbox}[colback=jcubeblue,colframe=jcubeblue,coltext=white,width=0.92\textwidth,halign=center]
{\LARGE\bfseries Relat\'{o}rio de Anomalias em Interna\c{c}\~{o}es}\\[0.3cm]
{\large An\'{a}lise via Embeddings do G\^{e}meo Digital --- Graph-JEPA}
\end{tcolorbox}
\vspace{0.8cm}
""")
    L.append(r"{\Large Per\'{i}odo: \textbf{" +
             escape_latex(START_DATE.strftime("%d/%m/%Y")) + r"} a \textbf{" +
             escape_latex(REPORT_DATE.strftime("%d/%m/%Y")) + r"}}\\[0.4cm]")
    L.append(r"{\large Gerado em: \textbf{23 de mar\c{c}o de 2026}}\\[1.5cm]")

    # Severity boxes
    L.append(r"""\begin{tabular}{ccc}
\begin{tcolorbox}[colback=anomred!10,colframe=anomred,width=3.8cm,halign=center,left=3pt,right=3pt]
{\LARGE\bfseries\textcolor{anomred}{""" + str(total_critical) + r"""}}\\[2pt]
{\small\textbf{CR\'ITICOS} (z$\geq$5)}
\end{tcolorbox}
&
\begin{tcolorbox}[colback=anomorange!10,colframe=anomorange,width=3.8cm,halign=center,left=3pt,right=3pt]
{\LARGE\bfseries\textcolor{anomorange}{""" + str(total_high) + r"""}}\\[2pt]
{\small\textbf{ALTOS} (3$\leq$z$<$5)}
\end{tcolorbox}
&
\begin{tcolorbox}[colback=anomyellow!10,colframe=anomyellow,width=3.8cm,halign=center,left=3pt,right=3pt]
{\LARGE\bfseries\textcolor{anomyellow}{""" + str(total_moderate) + r"""}}\\[2pt]
{\small\textbf{MODERADOS} (2$\leq$z$<$3)}
\end{tcolorbox}
\end{tabular}

\vspace{0.6cm}
\begin{tcolorbox}[colback=lightblue,colframe=jcubeblue,width=0.85\textwidth,halign=center]
{\large\bfseries Total de Anomalias: """ + str(total_anomalies) + r""" \quad | \quad Sistemas Hospitalares: """ + str(len(sources)) + r"""}\\[4pt]
""")
    if total_vl_fatura > 0:
        L.append(r"LOS M\'{e}dio: \textbf{" + f"{avg_los:.1f}" + r"} dias \quad | \quad Total Faturado: \textbf{" +
                 brl(total_vl_fatura) + r"}\\")
    else:
        L.append(r"LOS M\'{e}dio: \textbf{" + f"{avg_los:.1f}" + r"} dias\\")
    L.append(r"""\end{tcolorbox}

\vfill
{\small\textcolor{jcubegray}{
Metodologia: Z-score sobre dist\^{a}ncia euclidiana ao centr\'{o}ide dos embeddings JEPA\\
Limiar: z $>$ """ + str(Z_THRESHOLD) + r""" --- Modelo: node\_emb\_epoch\_2 (17.394.136 n\'{o}s $\times$ 64 dim)\\
Banco de dados: aggregated\_fixed\_union.db --- Vers\~{a}o: 2026-03
}}
\end{center}
\end{titlepage}
""")

    L.append(r"\tableofcontents\clearpage")

    # ── Executive Summary ──
    L.append(r"\section{Sum\'{a}rio Executivo}")
    L.append(r"""
Este relat\'{o}rio apresenta \textbf{todas as interna\c{c}\~{o}es an\^{o}malas} detectadas pelo
\textit{Digital Twin} JCUBE no per\'{i}odo de \textbf{""" +
        escape_latex(START_DATE.strftime("%d/%m/%Y")) + r"""} a \textbf{""" +
        escape_latex(REPORT_DATE.strftime("%d/%m/%Y")) + r"""}.

A detec\c{c}\~{a}o utiliza \textbf{z-score} sobre a dist\^{a}ncia euclidiana de cada interna\c{c}\~{a}o ao
centr\'{o}ide de todas as interna\c{c}\~{o}es no espa\c{c}o de embeddings do modelo \textit{Graph-JEPA}
treinado (17,4M n\'{o}s $\times$ 64 dimens\~{o}es). Interna\c{c}\~{o}es com $z > """ +
        str(Z_THRESHOLD) + r"""$ s\~{a}o classificadas como an\^{o}malas.

\subsection{M\'{e}tricas Globais}
\begin{center}
\begin{tabular}{lr}
\toprule
\textbf{M\'{e}trica} & \textbf{Valor} \\
\midrule
Total de anomalias detectadas & """ + str(total_anomalies) + r""" \\
Sistemas hospitalares (fontes) & """ + str(len(sources)) + r""" \\
Cr\'{i}ticos (z $\geq$ 5) & \textcolor{anomred}{\textbf{""" + str(total_critical) + r"""}} \\
Altos (3 $\leq$ z $<$ 5) & \textcolor{anomorange}{\textbf{""" + str(total_high) + r"""}} \\
Moderados (2 $\leq$ z $<$ 3) & \textcolor{anomyellow}{\textbf{""" + str(total_moderate) + r"""}} \\
LOS m\'{e}dio das anomalias & """ + f"{avg_los:.1f}" + r""" dias \\
""")
    if total_vl_fatura > 0:
        L.append(r"Total faturado (anomalias) & " + brl(total_vl_fatura) + r" \\" + "\n")
    if total_vl_glosa > 0:
        L.append(r"Total glosado (anomalias) & \textcolor{anomred}{" + brl(total_vl_glosa) + r"} \\" + "\n")
    L.append(r"""\bottomrule
\end{tabular}
\end{center}
""")

    # Summary by source
    L.append(r"\subsection{Anomalias por Sistema Hospitalar}")
    L.append(r"""\begin{center}
\begin{longtable}{lrrrr}
\toprule
\textbf{Sistema / Fonte} & \textbf{Total} & \textbf{Cr\'{i}t.} & \textbf{Alto} & \textbf{LOS M\'{e}d.} \\
\midrule
\endhead
\bottomrule
\endfoot
""")
    for src in sorted(sources.keys()):
        alist = sources[src]
        n = len(alist)
        nc = sum(1 for a in alist if a["Z_SCORE"] >= 5)
        nh = sum(1 for a in alist if 3 <= a["Z_SCORE"] < 5)
        avg_l = float(np.mean([safe_float(a.get("LOS_DIAS", 0)) for a in alist]))
        L.append(
            escape_latex(src) + r" & " + str(n) +
            r" & \textcolor{anomred}{" + str(nc) + r"}" +
            r" & \textcolor{anomorange}{" + str(nh) + r"}" +
            r" & " + f"{avg_l:.1f}" + r" \\" + "\n"
        )
    L.append(r"\end{longtable}\end{center}" + "\n\n")
    L.append(r"\clearpage")

    # ── Per-source sections ──
    for src in sorted(sources.keys()):
        alist = sorted(sources[src], key=lambda a: -a["Z_SCORE"])
        section_label = escape_latex(src)
        L.append(r"\section{" + section_label + r"}")

        n = len(alist)
        nc = sum(1 for a in alist if a["Z_SCORE"] >= 5)
        nh = sum(1 for a in alist if 3 <= a["Z_SCORE"] < 5)
        nm = sum(1 for a in alist if 2 <= a["Z_SCORE"] < 3)
        avg_l = float(np.mean([safe_float(a.get("LOS_DIAS", 0)) for a in alist]))
        max_z = max((a["Z_SCORE"] for a in alist), default=0.0)
        vl_glo_src = sum(safe_float(a["glosa"].get("vl_glosado_total")) for a in alist)
        vl_fat_src = sum(safe_float(a["fatura"].get("vl_total")) for a in alist)

        L.append(r"""\begin{tcolorbox}[colback=lightblue,colframe=jcubeblue,title={\textbf{Resumo --- """ + section_label + r"""}}]
\begin{tabular}{ll@{\quad}ll@{\quad}ll}
Total: \textbf{""" + str(n) + r"""} &
Cr\'{i}t.: \textcolor{anomred}{\textbf{""" + str(nc) + r"""}} &
Altos: \textcolor{anomorange}{\textbf{""" + str(nh) + r"""}} &
Mod.: \textcolor{anomyellow}{\textbf{""" + str(nm) + r"""}} &
LOS m\'{e}d.: \textbf{""" + f"{avg_l:.1f}" + r"""d} &
Maior z: \textbf{""" + f"{max_z:.2f}" + r"""} \\
""")
        if vl_fat_src > 0:
            L.append(r"\multicolumn{2}{l}{Total faturado: \textbf{" + brl(vl_fat_src) + r"}} & ")
        if vl_glo_src > 0:
            L.append(r"\multicolumn{2}{l}{\textcolor{anomred}{Total glosado: \textbf{" + brl(vl_glo_src) + r"}}} & & \\")
        L.append(r"""
\end{tabular}
\end{tcolorbox}
""")

        # Summary table
        L.append(r"\subsection{Tabela Resumo}")
        L.append(r"""\begin{center}
\begin{longtable}{>{\scriptsize}r>{\scriptsize}r>{\scriptsize}c>{\scriptsize}c>{\scriptsize}r>{\scriptsize}r>{\scriptsize}r>{\scriptsize}r}
\toprule
\textbf{Intern.} & \textbf{Pac.} & \textbf{Admiss\~{a}o} & \textbf{Alta} & \textbf{LOS} & \textbf{Z} & \textbf{Faturado} & \textbf{Glosado} \\
\midrule
\endhead
\bottomrule
\endfoot
""")
        for a in alist:
            iid = a["ID_CD_INTERNACAO"]
            pid = a.get("ID_CD_PACIENTE", "?")
            adm = fmt_date(a.get("DH_ADMISSAO_HOSP"))
            alta = fmt_date(a.get("DH_FINALIZACAO"))
            los = safe_int(a.get("LOS_DIAS", 0))
            z = a["Z_SCORE"]
            vl_f = safe_float(a["fatura"].get("vl_total"))
            vl_g = safe_float(a["glosa"].get("vl_glosado_total"))
            color = z_color(z)
            vl_f_str = brl(vl_f) if vl_f > 0 else "---"
            vl_g_str = r"\textcolor{anomred}{" + brl(vl_g) + r"}" if vl_g > 0 else "---"
            L.append(
                f"\\textcolor{{{color}}}{{\\textbf{{{iid}}}}} & {pid} & {adm} & {alta} & {los}d & "
                f"\\textcolor{{{color}}}{{{z:.2f}}} & {vl_f_str} & {vl_g_str}" + r" \\" + "\n"
            )
        L.append(r"\end{longtable}\end{center}" + "\n")

        # Detailed cards
        L.append(r"\subsection{Fichas Detalhadas}")

        for a in alist:
            iid = a["ID_CD_INTERNACAO"]
            pid = a.get("ID_CD_PACIENTE", "?")
            z = a["Z_SCORE"]
            los = safe_int(a.get("LOS_DIAS", 0))
            adm = fmt_date(a.get("DH_ADMISSAO_HOSP"))
            alta = fmt_date(a.get("DH_FINALIZACAO"))
            color = z_color(z)
            severity = z_label(z)
            nm_hosp = escape_latex(str(a.get("nm_hospital") or "---"))
            senha = escape_latex(str(a.get("NR_SENHA") or "---"))
            guia = escape_latex(str(a.get("NR_GUIA_AUTORIZACAO") or "---"))

            card_title = (
                f"Interna\\c{{c}}\\~{{a}}o \\#{iid} | Pac. \\#{pid} | {severity} (z={z:.2f}) | "
                f"LOS: {los}d | {adm}\\,$\\to$\\,{alta}"
            )

            L.append(r"\begin{anomalycard}[" + card_title + r"]{" + color + r"}")

            # Identification row
            L.append(
                r"\textbf{Hospital:} " + nm_hosp + r"\quad " +
                r"\textbf{Fonte:} " + escape_latex(src) + r"\quad " +
                r"\textbf{Senha:} " + senha + r"\quad " +
                r"\textbf{Guia:} " + guia + r"\\" + "\n"
            )

            # CIDs
            cids_data = a.get("cids", {})
            n_cids = safe_int(cids_data.get("n_cids"))
            cid_princ = escape_latex(truncate(str(cids_data.get("cid_principal") or ""), 120))
            all_cids = escape_latex(truncate(str(cids_data.get("cids") or ""), 280))
            if n_cids > 0:
                L.append(r"\textbf{CID Principal:} {\small " + cid_princ + r"}\quad\textbf{Total CIDs:} " + str(n_cids) + r"\\" + "\n")
                if all_cids and all_cids != "---":
                    L.append(r"\textbf{CIDs:} {\scriptsize " + all_cids + r"}\\" + "\n")

            # Clinical texts
            for label, field, maxl in [
                ("Diagn.\\'{o}stico", "DS_DESCRICAO", 180),
                ("Hist\\'{o}rico", "DS_HISTORICO", 180),
                ("Motivo", "DS_MOTIVO", 130),
            ]:
                val = truncate(str(a.get(field) or ""), maxl)
                if val and val != "---":
                    L.append(r"\textbf{" + label + r":} {\scriptsize " + escape_latex(val) + r"}\\" + "\n")

            # Financial table
            fat = a.get("fatura", {})
            glo = a.get("glosa", {})
            neg = a.get("negociacoes", {})
            if fat or glo or neg:
                L.append(r"\vspace{2pt}{\footnotesize\begin{tabular}{@{}ll@{\hspace{12pt}}ll@{\hspace{12pt}}ll@{}}" + "\n")
                L.append(r"\toprule\multicolumn{6}{c}{\textbf{Dados Financeiros}}\\\midrule" + "\n")
                if fat:
                    L.append(
                        r"VL Total & " + brl(fat.get("vl_total")) +
                        r" & VL RH & " + brl(fat.get("vl_rh")) +
                        r" & VL Mat. & " + brl(fat.get("vl_mat")) + r" \\" + "\n"
                    )
                    L.append(
                        r"VL Med. & " + brl(fat.get("vl_med")) +
                        r" & VL OPME & " + brl(fat.get("vl_opme")) +
                        r" & VL SADT & " + brl(fat.get("vl_sadt")) + r" \\" + "\n"
                    )
                    L.append(
                        r"VL L\'{i}q. & " + brl(fat.get("vl_liquido")) +
                        r" & Glosa Fat. & \textcolor{anomred}{" + brl(fat.get("vl_glosa_total")) + r"}" +
                        r" & Diverg. & \textcolor{anomorange}{" + brl(fat.get("vl_divergencia")) + r"}" +
                        r" \\" + "\n"
                    )
                if glo:
                    L.append(r"\midrule\multicolumn{6}{c}{\textbf{Glosas}}\\\midrule" + "\n")
                    L.append(
                        r"N Glosas & \textbf{" + str(safe_int(glo.get("n_glosas"))) + r"}" +
                        r" & VL Glosado & \textcolor{anomred}{\textbf{" + brl(glo.get("vl_glosado_total")) + r"}}" +
                        r" & Aceito & " + brl(glo.get("vl_aceito")) + r" \\" + "\n"
                    )
                if neg:
                    L.append(r"\midrule\multicolumn{6}{c}{\textbf{Negocia\c{c}\~{o}es de Auditoria}}\\\midrule" + "\n")
                    tipos = escape_latex(truncate(str(neg.get("tipos_negociacao") or ""), 60))
                    L.append(
                        r"N Negoc. & \textbf{" + str(safe_int(neg.get("n_negociacoes"))) + r"}" +
                        r" & VL Negoc. & " + brl(neg.get("vl_negociado_total")) +
                        r" & Tipos & {\scriptsize " + tipos + r"} \\" + "\n"
                    )
                L.append(r"\bottomrule\end{tabular}}" + "\n")

            # Operational counts
            proc = a.get("procedimentos", {})
            fit = a.get("fatura_itens", {})
            evo = a.get("evolucao", {})
            aud = a.get("auditoria", {})
            ev = a.get("eventos_adversos", {})
            opme = a.get("opme", {})

            L.append(r"\vspace{2pt}{\footnotesize " +
                r"\textbf{Proced.:} " + str(safe_int(proc.get("n_procedimentos"))) +
                r"\quad\textbf{Evolu\c{c}\~{o}es:} " + str(safe_int(evo.get("n_evolucoes"))) +
                r"\quad\textbf{Audit. RAH:} " + str(safe_int(aud.get("n_auditorias"))) +
                r"\quad\textbf{Eventos Adv.:} " + str(safe_int(ev.get("n_eventos"))) +
                r"\quad\textbf{Itens Fat.:} " + str(safe_int(fit.get("n_itens"))) +
                r"\quad\textbf{OPME:} " + str(safe_int(opme.get("n_opme"))) +
                r"}" + "\n"
            )

            # Similar admissions
            similar = similar_map.get(iid, [])
            if similar:
                L.append(r"\vspace{2pt}{\scriptsize\textbf{Interna\c{c}\~{o}es similares (cosine):} ")
                parts = []
                for sname, ssim in similar:
                    # Extract the ID part and escape it properly
                    if "ID_CD_INTERNACAO_" in sname:
                        raw_id = sname.split("ID_CD_INTERNACAO_")[1]
                        sid = "\\#" + escape_latex(raw_id)
                    else:
                        # For non-INTERNACAO similar nodes, just show a cleaned version
                        clean = sname.replace("ID_CD_", "").replace("_", " ")[:30]
                        sid = escape_latex(clean)
                    parts.append(f"{sid} ({ssim:.3f})")
                L.append(", ".join(parts) + "}\n")

            L.append(r"\end{anomalycard}" + "\n\n")

        L.append(r"\clearpage")

    # ── Appendix ──
    L.append(r"\section*{Ap\^{e}ndice: Metodologia de Detec\c{c}\~{a}o}")
    L.append(r"\addcontentsline{toc}{section}{Ap\^{e}ndice: Metodologia}")
    L.append(r"""
\subsection*{1. Modelo Graph-JEPA}
O modelo \textit{Graph-JEPA} foi treinado sobre o grafo de conhecimento JCUBE com \textbf{17.394.136 n\'{o}s}
e \textbf{64 dimens\~{o}es} de embedding. Cada n\'{o} representa uma entidade (interna\c{c}\~{a}o, paciente,
fatura, m\'{e}dico, etc.) e as arestas representam rela\c{c}\~{o}es entre elas (165 milh\~{o}es de triplas).

\subsection*{2. Detec\c{c}\~{a}o de Anomalias via Z-score}
Para cada interna\c{c}\~{a}o com n\'{o} no grafo:
\begin{enumerate}[nosep]
  \item Calcula-se a \textbf{dist\^{a}ncia euclidiana} do embedding ao \textbf{centr\'{o}ide} de todas as interna\c{c}\~{o}es.
  \item Calcula-se o \textbf{z-score}: $z = \frac{d - \mu}{\sigma}$, onde $\mu$ e $\sigma$ s\~{a}o a m\'{e}dia e desvio padr\~{a}o das dist\^{a}ncias.
  \item Interna\c{c}\~{o}es com $z > """ + str(Z_THRESHOLD) + r"""$ s\~{a}o classificadas como an\^{o}malas.
\end{enumerate}

\subsection*{3. Classifica\c{c}\~{a}o de Severidade}
\begin{itemize}[nosep]
  \item \textcolor{anomred}{\textbf{CR\'ITICO}}: z $\geq$ 5
  \item \textcolor{anomorange}{\textbf{ALTO}}: 3 $\leq$ z $<$ 5
  \item \textcolor{anomyellow}{\textbf{MODERADO}}: 2 $\leq$ z $<$ 3
\end{itemize}

\subsection*{4. Similaridade Sem\^{a}ntica}
A busca por interna\c{c}\~{o}es similares utiliza \textbf{similaridade por cosseno} no espa\c{c}o de embeddings.
Normaliza-se todos os vetores de interna\c{c}\~{a}o uma vez e calcula-se o produto interno para efici\^{e}ncia.

\subsection*{5. Enriquecimento de Dados}
Cada anomalia \'{e} enriquecida com dados do banco \texttt{aggregated\_fixed\_union.db} via DuckDB:
faturamento, glosas, negocia\c{c}\~{o}es de auditoria, CIDs, procedimentos, evolu\c{c}\~{o}es cl\'{i}nicas,
formul\'{a}rios RAH, eventos adversos e OPME.
""")

    L.append(r"\end{document}")
    return "\n".join(L)


# ─────────────────────────────────────────────────────────────────
# STEP 6 – Compile PDF
# ─────────────────────────────────────────────────────────────────

def compile_latex(latex_content: str, output_pdf: Path):
    print("[5/5] Compiling PDF …")
    tex_file = output_pdf.with_suffix(".tex")
    tex_file.write_text(latex_content, encoding="utf-8")
    print(f"    LaTeX written: {tex_file} ({tex_file.stat().st_size/1024:.0f} KB)")

    for run in range(2):
        print(f"    pdflatex pass {run+1}/2 …")
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode",
             "-output-directory", str(output_pdf.parent),
             str(tex_file)],
            capture_output=True, cwd=str(output_pdf.parent),
        )
        stdout_str = result.stdout.decode("latin-1", errors="replace")
        if result.returncode != 0 and run == 1:
            log_lines = stdout_str.split("\n")
            errs = [l for l in log_lines if l.startswith("!") or "Error" in l]
            print("    pdflatex errors:")
            for e in errs[:20]:
                print("   ", e)
        if output_pdf.exists():
            size_kb = output_pdf.stat().st_size / 1024
            print(f"    PDF pass {run+1}: {size_kb:.0f} KB")

    if output_pdf.exists():
        print(f"\nPDF ready: {output_pdf}  ({output_pdf.stat().st_size/1024:.0f} KB)")
    else:
        # Print last 40 lines of log
        log_file = tex_file.with_suffix(".log")
        if log_file.exists():
            lines = log_file.read_text(encoding="latin-1", errors="replace").split("\n")
            print("Last 40 lines of pdflatex log:")
            for l in lines[-40:]:
                print(" ", l)
        raise FileNotFoundError(f"PDF not found at {output_pdf}")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    print("=" * 60)
    print("JCUBE Anomaly Report Generator")
    print(f"Period: {START_DATE.strftime('%Y-%m-%d')} to {REPORT_DATE.strftime('%Y-%m-%d')}")
    print(f"Z-threshold: {Z_THRESHOLD}")
    print("=" * 60)

    # 1. Load twin
    unique_nodes, embeddings, node_to_idx, internacao_mask = load_twin()

    # 2. Detect anomalies
    internacao_ids, anomaly_z = detect_anomalies_zscore(embeddings, internacao_mask, unique_nodes)

    # 3. Fetch DuckDB details
    admissions = fetch_admission_details(internacao_ids, anomaly_z)
    print(f"    Total admissions to report: {len(admissions)}")

    # 4. Batch similar admissions lookup
    valid_iids = [a["ID_CD_INTERNACAO"] for a in admissions]
    similar_map = batch_find_similar(
        embeddings, node_to_idx, internacao_mask, unique_nodes,
        valid_iids, k=5
    )

    # 5. Generate LaTeX
    print("[4/5] Generating LaTeX document …")
    latex = generate_latex(admissions, similar_map)

    # 6. Compile PDF
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    compile_latex(latex, OUTPUT_PDF)

    elapsed = time.time() - t_start
    print(f"\nFinished in {elapsed:.1f}s")
    print(f"Report: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
