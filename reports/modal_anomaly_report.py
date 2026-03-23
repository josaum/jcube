#!/usr/bin/env python3
"""
Modal script: JCUBE V4 Anomaly Report Generator (v2 — Explained)
Runs on Modal, loads 8.4GB V4 embeddings from jepa-cache volume,
queries DuckDB from jcube-data volume, generates LaTeX → PDF.

New in v2:
  - "Justificativa da Anomalia" per anomaly with:
    * Hospital baseline comparison (LOS, billing, procedures, exams, glosa rate)
    * 5 most similar admissions comparison
    * Top deviating embedding dimensions mapped to feature importance
    * Concrete DuckDB facts
    * Natural language audit justification

Usage:
    modal run reports/modal_anomaly_report.py
    modal run --detach reports/modal_anomaly_report.py
"""
from __future__ import annotations

import modal

# ─────────────────────────────────────────────────────────────────
# Modal App + Volumes
# ─────────────────────────────────────────────────────────────────

app = modal.App("jcube-anomaly-report")

jepa_cache = modal.Volume.from_name("jepa-cache", create_if_missing=False)
data_vol   = modal.Volume.from_name("jcube-data",  create_if_missing=False)

VOLUMES = {
    "/cache": jepa_cache,
    "/data":  data_vol,
}

# ─────────────────────────────────────────────────────────────────
# Container image — needs torch (CPU-only), duckdb, pyarrow, latex
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
    )
)

# ─────────────────────────────────────────────────────────────────
# Paths inside the container
# ─────────────────────────────────────────────────────────────────

GRAPH_PARQUET  = "/data/jcube_graph.parquet"
WEIGHTS_PATH   = "/cache/tkg-fullscale/node_embeddings.pt"
DB_PATH        = "/data/aggregated_fixed_union.db"
OUTPUT_DIR     = "/data/reports"
OUTPUT_PDF     = f"{OUTPUT_DIR}/anomaly_report_v4_explained_2026_03.pdf"

REPORT_DATE_STR = "2026-03-23"
START_DATE_STR  = "2026-02-21"
Z_THRESHOLD     = 2.0

# ─────────────────────────────────────────────────────────────────
# Feature importance mapping for embedding dimensions
# ─────────────────────────────────────────────────────────────────

DIM_MEANING = {
    16: "padr\u00e3o de faturamento",
    28: "complexidade cl\u00ednica",
    30: "volume de procedimentos",
    46: "trajet\u00f3ria temporal",
    53: "risco de glosa",
    61: "padr\u00e3o operacional",
}

# ─────────────────────────────────────────────────────────────────
# Helpers (all run inside container)
# ─────────────────────────────────────────────────────────────────

def _fmt_date(d) -> str:
    if d is None:
        return "\u2014"
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
    """Escape LaTeX special characters while preserving UTF-8 accents.

    The document uses \\usepackage[utf8]{inputenc} so ç, ã, é, etc.
    pass through directly. We only escape structural LaTeX metacharacters.
    """
    if not s:
        return ""
    s = str(s)
    # Order matters: backslash first, then the rest
    s = s.replace("\\", "\\textbackslash{}")
    s = s.replace("&",  "\\&")
    s = s.replace("%",  "\\%")
    s = s.replace("$",  "\\$")
    s = s.replace("#",  "\\#")
    s = s.replace("_",  "\\_")
    # Do NOT escape { } ~ ^ — these corrupt UTF-8 accented text
    # like "internação" → "interna\\{}c{c}\\{}~{a}o"
    # Instead, only escape literal braces that aren't part of LaTeX commands
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

def _brl_plain(v) -> str:
    """BRL without LaTeX escapes — for use in f-strings that will be escaped later."""
    f = _safe_float(v)
    if f == 0:
        return "---"
    return "R$ {:,.2f}".format(f).replace(",", "X").replace(".", ",").replace("X", ".")

def _z_color(z: float) -> str:
    if z >= 5:
        return "anomred"
    elif z >= 3:
        return "anomorange"
    return "anomyellow"

def _z_label(z: float) -> str:
    if z >= 5:
        return "CR\\'{I}TICO"
    elif z >= 3:
        return "ALTO"
    return "MODERADO"

def _truncate(s: str, max_len: int = 200) -> str:
    if not s:
        return "---"
    s = str(s).strip()
    if len(s) > max_len:
        return s[:max_len] + "..."
    return s

def _pct_diff(val, baseline) -> str:
    """Return formatted '+X%' or '-X%' deviation from baseline."""
    if baseline == 0:
        return "N/A"
    diff = ((val - baseline) / baseline) * 100
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:.0f}\\%"

def _mult(val, baseline) -> str:
    """Return 'Xx' multiplier string."""
    if baseline == 0:
        return "N/A"
    m = val / baseline
    return f"{m:.1f}x"


# ─────────────────────────────────────────────────────────────────
# Step 1 – Load twin (V4: source_db-prefixed node IDs)
# ─────────────────────────────────────────────────────────────────

def _load_twin():
    import time
    import numpy as np
    import torch
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    import pyarrow as pa

    print("[1/6] Loading node vocabulary from graph parquet ...")
    t0 = time.time()
    table = pq.read_table(GRAPH_PARQUET, columns=["subject_id", "object_id"])
    subj  = table.column("subject_id")
    obj   = table.column("object_id")
    all_nodes = pa.chunked_array(subj.chunks + obj.chunks)
    unique_nodes = pc.unique(all_nodes).to_numpy(zero_copy_only=False).astype(object)
    del table, subj, obj, all_nodes
    n_nodes = len(unique_nodes)
    print(f"    {n_nodes:,} unique nodes in {time.time()-t0:.1f}s")

    print("[1/6] Loading V4 embedding weights ...")
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
            f"Vocab size mismatch: {len(unique_nodes):,} nodes vs "
            f"{embeddings.shape[0]:,} embedding rows"
        )

    # V4 node format: "GHO-BRADESCO/ID_CD_INTERNACAO_117926"
    node_to_idx = {str(n): i for i, n in enumerate(unique_nodes)}

    internacao_mask = np.array(
        ["/ID_CD_INTERNACAO_" in str(n) for n in unique_nodes], dtype=bool
    )
    print(f"    INTERNACAO nodes (V4 prefixed): {internacao_mask.sum():,}")
    return unique_nodes, embeddings, node_to_idx, internacao_mask


# ─────────────────────────────────────────────────────────────────
# Step 2 – Anomaly detection via Z-score
# ─────────────────────────────────────────────────────────────────

def _detect_anomalies(embeddings, internacao_mask, unique_nodes):
    import time
    import numpy as np

    print("[2/6] Computing anomaly z-scores ...")
    t0 = time.time()
    vecs  = embeddings[internacao_mask]
    names = unique_nodes[internacao_mask]

    centroid = vecs.mean(axis=0)
    dists    = np.linalg.norm(vecs - centroid, axis=1)

    mean_d = dists.mean()
    std_d  = dists.std()
    z_scores = (dists - mean_d) / (std_d + 1e-9)

    anomaly_mask  = z_scores > Z_THRESHOLD
    anomaly_names = names[anomaly_mask]
    anomaly_z     = z_scores[anomaly_mask]

    order         = np.argsort(-anomaly_z)
    anomaly_names = anomaly_names[order]
    anomaly_z     = anomaly_z[order]

    print(f"    {anomaly_mask.sum():,} anomalies (z>{Z_THRESHOLD}) in {time.time()-t0:.1f}s")

    # V4 node format: "<source_db>/ID_CD_INTERNACAO_<int>"
    internacao_records = []
    for n in anomaly_names:
        s = str(n)
        try:
            src_db, id_part = s.split("/", 1)
            iid = int(id_part.split("ID_CD_INTERNACAO_")[1])
            internacao_records.append((src_db, iid))
        except Exception:
            internacao_records.append((None, None))

    return internacao_records, anomaly_z, centroid, vecs, names


# ─────────────────────────────────────────────────────────────────
# Step 3 – Per-anomaly embedding dimension analysis
# ─────────────────────────────────────────────────────────────────

def _analyze_embedding_dimensions(embeddings, node_to_idx, internacao_mask,
                                   unique_nodes, records: list[tuple]) -> dict:
    """
    For each anomaly, find which embedding dimensions deviate most from the
    internacao centroid. Returns dict keyed by (src_db, iid).
    """
    import numpy as np

    print("[3/6] Analyzing embedding dimension deviations ...")

    int_vecs = embeddings[internacao_mask].astype(np.float32)
    centroid = int_vecs.mean(axis=0)   # shape: (64,)
    std_per_dim = int_vecs.std(axis=0).clip(min=1e-8)

    dim_analysis = {}
    for src_db, iid in records:
        if src_db is None or iid is None:
            dim_analysis[(src_db, iid)] = []
            continue
        key = f"{src_db}/ID_CD_INTERNACAO_{iid}"
        if key not in node_to_idx:
            dim_analysis[(src_db, iid)] = []
            continue
        idx_global = node_to_idx[key]
        vec = embeddings[idx_global].astype(np.float32)
        # z-score per dimension
        dim_z = (vec - centroid) / std_per_dim
        # top 3 most deviating dimensions (by |z|)
        top_dims = np.argsort(-np.abs(dim_z))[:6]
        result = []
        for d in top_dims:
            meaning = DIM_MEANING.get(int(d), f"dim{d}")
            result.append((int(d), float(dim_z[d]), meaning))
        dim_analysis[(src_db, iid)] = result

    return dim_analysis


# ─────────────────────────────────────────────────────────────────
# Step 4 – Batch similar admissions via cosine similarity
# ─────────────────────────────────────────────────────────────────

def _batch_find_similar(embeddings, node_to_idx, internacao_mask, unique_nodes,
                        records: list[tuple[str, int]], k: int = 5):
    import time
    import numpy as np

    print(f"    Pre-computing similarities for {len(records)} anomalies ...")
    t0 = time.time()

    int_vecs  = embeddings[internacao_mask].astype(np.float32)
    int_names = unique_nodes[internacao_mask]

    norms         = np.linalg.norm(int_vecs, axis=1, keepdims=True).clip(min=1e-8)
    int_vecs_norm = int_vecs / norms

    results = {}
    for src_db, iid in records:
        if src_db is None or iid is None:
            continue
        key = f"{src_db}/ID_CD_INTERNACAO_{iid}"
        if key not in node_to_idx:
            results[(src_db, iid)] = []
            continue
        idx_global = node_to_idx[key]
        qvec  = embeddings[idx_global].astype(np.float32)
        qnorm = np.linalg.norm(qvec).clip(min=1e-8)
        qvec_norm = qvec / qnorm
        sims = int_vecs_norm @ qvec_norm
        top  = np.argsort(-sims)[:k + 1]
        sim_list = []
        for i in top:
            name = str(int_names[i])
            if name == key:
                continue
            sim_list.append((name, float(sims[i])))
            if len(sim_list) >= k:
                break
        results[(src_db, iid)] = sim_list

    print(f"    Done in {time.time()-t0:.1f}s")
    return results


# ─────────────────────────────────────────────────────────────────
# Step 5 – DuckDB: Fetch full admission details + baselines
# ─────────────────────────────────────────────────────────────────

def _fetch_admission_details(records: list[tuple[str, int]], anomaly_z):
    import time
    import duckdb

    print(f"[4/6] Fetching DuckDB details for {len(records)} anomalies ...")
    t0 = time.time()

    z_map     = {r: float(z) for r, z in zip(records, anomaly_z) if r[0] is not None}
    valid_ids = [(src, iid) for src, iid in records if iid is not None]

    con = duckdb.connect(str(DB_PATH))

    con.execute("""
        CREATE OR REPLACE TEMP TABLE tmp_anomaly_ids (
            source_db VARCHAR,
            iid       INTEGER
        )
    """)
    batch_size = 1000
    for i in range(0, len(valid_ids), batch_size):
        batch = valid_ids[i:i + batch_size]
        vals  = ", ".join(f"('{src}', {iid})" for src, iid in batch)
        con.execute(f"INSERT INTO tmp_anomaly_ids VALUES {vals}")

    # ── Core internacao ──
    q_inter = f"""
    SELECT i.ID_CD_INTERNACAO, i.ID_CD_PACIENTE, i.source_db,
        i.DH_ADMISSAO_HOSP, i.DH_FINALIZACAO,
        i.DS_DESCRICAO, i.DS_HISTORICO, i.DS_MOTIVO,
        i.DS_CONDUTA_INTERNACAO, i.DS_DESCRICAO_EVOLUCAO,
        CASE WHEN i.DH_FINALIZACAO IS NOT NULL
            THEN DATEDIFF('day', i.DH_ADMISSAO_HOSP::DATE, i.DH_FINALIZACAO::DATE)
            ELSE DATEDIFF('day', i.DH_ADMISSAO_HOSP::DATE, DATE '{REPORT_DATE_STR}')
        END AS LOS_DIAS,
        i.NR_SENHA, i.NR_GUIA_AUTORIZACAO, i.ID_CD_HOSPITAL, i.IN_SITUACAO
    FROM agg_tb_capta_internacao_cain i
    JOIN tmp_anomaly_ids t
      ON i.ID_CD_INTERNACAO = t.iid
     AND i.source_db = t.source_db
    WHERE i.DH_ADMISSAO_HOSP >= '{START_DATE_STR}'
      AND i.DH_ADMISSAO_HOSP <= '{REPORT_DATE_STR}'
    ORDER BY i.source_db, i.ID_CD_INTERNACAO
    """
    cur        = con.execute(q_inter)
    inter_cols = [d[0] for d in cur.description]
    inter_rows = cur.fetchall()

    if not inter_rows:
        print("    No admissions in 30-day window -- querying most recent 2000 anomalies ...")
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
        JOIN tmp_anomaly_ids t
          ON i.ID_CD_INTERNACAO = t.iid
         AND i.source_db = t.source_db
        WHERE i.DH_ADMISSAO_HOSP > '2000-01-01'
        ORDER BY i.DH_ADMISSAO_HOSP DESC
        LIMIT 2000
        """
        cur        = con.execute(q_inter2)
        inter_cols = [d[0] for d in cur.description]
        inter_rows = cur.fetchall()

    admissions = [dict(zip(inter_cols, r)) for r in inter_rows]
    for a in admissions:
        key = (a["source_db"], a["ID_CD_INTERNACAO"])
        a["Z_SCORE"] = z_map.get(key, 2.01)

    # Build tmp_valid_ids with confirmed (source_db, iid) pairs
    valid_found = [(a["source_db"], a["ID_CD_INTERNACAO"]) for a in admissions]
    con.execute("""
        CREATE OR REPLACE TEMP TABLE tmp_valid_ids (
            source_db VARCHAR,
            iid       INTEGER
        )
    """)
    for i in range(0, len(valid_found), batch_size):
        batch = valid_found[i:i + batch_size]
        if batch:
            vals = ", ".join(f"('{src}', {iid})" for src, iid in batch)
            con.execute(f"INSERT INTO tmp_valid_ids VALUES {vals}")

    if not valid_found:
        return admissions, {}

    def _exec(q):
        cur = con.execute(q)
        cols = [d[0] for d in cur.description]
        return cols, cur.fetchall()

    # ── Fatura ──
    print("    Fetching billing ...")
    fat_cols, fat_rows = _exec("""
        SELECT f.ID_CD_INTERNACAO, f.source_db,
            SUM(f.VL_TOTAL)               AS vl_total,
            SUM(f.VL_RH)                  AS vl_rh,
            SUM(f.VL_MAT)                 AS vl_mat,
            SUM(f.VL_MED)                 AS vl_med,
            SUM(f.VL_EQP)                 AS vl_eqp,
            SUM(f.VL_OPME)                AS vl_opme,
            SUM(f.VL_SADT)                AS vl_sadt,
            SUM(f.VL_TAXA)                AS vl_taxa,
            SUM(f.VL_GLOSA_FECHAMENTO)    AS vl_glosa_total,
            SUM(f.VL_LIQUIDO_FAT)         AS vl_liquido,
            SUM(f.VL_DIVERGENCIA)         AS vl_divergencia,
            COUNT(*)                      AS n_faturas
        FROM agg_tb_fatura_fatu f
        JOIN tmp_valid_ids t
          ON f.ID_CD_INTERNACAO = t.iid
         AND f.source_db = t.source_db
        GROUP BY f.ID_CD_INTERNACAO, f.source_db
    """)
    fatura_map = {
        (dict(zip(fat_cols, r))["source_db"], dict(zip(fat_cols, r))["ID_CD_INTERNACAO"]):
        dict(zip(fat_cols, r)) for r in fat_rows
    }

    # ── Fatura itens ──
    print("    Fetching fatura items ...")
    try:
        fit_cols, fit_rows = _exec("""
            SELECT fi.ID_CD_INTERNACAO, fi.source_db,
                COUNT(*) AS n_itens,
                SUM(fi.VL_TOTAL_FATURADO) AS vl_total_itens,
                SUM(CASE WHEN fi.FL_GLOSA = 'S' THEN 1 ELSE 0 END) AS n_itens_glosados,
                COUNT(DISTINCT fi.FL_TIPO_PRODUTO) AS n_tipos_produto
            FROM agg_tb_fatura_itens_fait fi
            JOIN tmp_valid_ids t
              ON fi.ID_CD_INTERNACAO = t.iid
             AND fi.source_db = t.source_db
            GROUP BY fi.ID_CD_INTERNACAO, fi.source_db
        """)
        fit_map = {
            (dict(zip(fit_cols, r))["source_db"], dict(zip(fit_cols, r))["ID_CD_INTERNACAO"]):
            dict(zip(fit_cols, r)) for r in fit_rows
        }
    except Exception as e:
        print(f"    fatura_itens skipped: {e}")
        fit_map = {}

    # ── Glosas ──
    print("    Fetching glosas ...")
    glo_cols, glo_rows = _exec("""
        SELECT g.ID_CD_INTERNACAO, g.source_db,
            COUNT(*) AS n_glosas,
            SUM(g.VL_GLOSADO) AS vl_glosado_total,
            SUM(CASE WHEN g.FL_ACEITE_GLOSA = 'S' THEN g.VL_GLOSADO ELSE 0 END) AS vl_aceito,
            SUM(CASE WHEN g.FL_ACEITE_GLOSA = 'N' THEN g.VL_GLOSADO ELSE 0 END) AS vl_recusado
        FROM agg_tb_fatura_glosa_fatg g
        JOIN tmp_valid_ids t
          ON g.ID_CD_INTERNACAO = t.iid
         AND g.source_db = t.source_db
        GROUP BY g.ID_CD_INTERNACAO, g.source_db
    """)
    glosa_map = {
        (dict(zip(glo_cols, r))["source_db"], dict(zip(glo_cols, r))["ID_CD_INTERNACAO"]):
        dict(zip(glo_cols, r)) for r in glo_rows
    }

    # ── Negociacoes ──
    print("    Fetching negotiations ...")
    try:
        neg_cols, neg_rows = _exec("""
            SELECT n.ID_CD_INTERNACAO, n.source_db,
                COUNT(*) AS n_negociacoes,
                SUM(n.VL_TOTAL) AS vl_negociado_total,
                STRING_AGG(DISTINCT CAST(n.FL_TIPO_NEGOCIACAO AS VARCHAR), ', ') AS tipos_negociacao
            FROM agg_tb_capta_negociacoes_auditoria_cnau n
            JOIN tmp_valid_ids t
              ON n.ID_CD_INTERNACAO = t.iid
             AND n.source_db = t.source_db
            GROUP BY n.ID_CD_INTERNACAO, n.source_db
        """)
        neg_map = {
            (dict(zip(neg_cols, r))["source_db"], dict(zip(neg_cols, r))["ID_CD_INTERNACAO"]):
            dict(zip(neg_cols, r)) for r in neg_rows
        }
    except Exception as e:
        print(f"    negociacoes skipped: {e}")
        neg_map = {}

    # ── CIDs ──
    print("    Fetching CIDs ...")
    cid_cols, cid_rows = _exec("""
        SELECT c.ID_CD_INTERNACAO, c.source_db,
            STRING_AGG(DISTINCT COALESCE(c.DS_DESCRICAO,'?'), ' | ') AS cids,
            COUNT(*) AS n_cids,
            STRING_AGG(CASE WHEN c.IN_PRINCIPAL = 'S' THEN c.DS_DESCRICAO END, ', ') AS cid_principal
        FROM agg_tb_capta_cid_caci c
        JOIN tmp_valid_ids t
          ON c.ID_CD_INTERNACAO = t.iid
         AND c.source_db = t.source_db
        GROUP BY c.ID_CD_INTERNACAO, c.source_db
    """)
    cid_map = {
        (dict(zip(cid_cols, r))["source_db"], dict(zip(cid_cols, r))["ID_CD_INTERNACAO"]):
        dict(zip(cid_cols, r)) for r in cid_rows
    }

    # ── Procedimentos ──
    print("    Fetching procedures ...")
    try:
        proc_cols, proc_rows = _exec("""
            SELECT p.ID_CD_INTERNACAO, p.source_db,
                COUNT(*) AS n_procedimentos,
                0 AS vl_proc_total
            FROM agg_tb_fatura_procedimentos_fapr p
            JOIN tmp_valid_ids t
              ON p.ID_CD_INTERNACAO = t.iid
             AND p.source_db = t.source_db
            GROUP BY p.ID_CD_INTERNACAO, p.source_db
        """)
        proc_map = {
            (dict(zip(proc_cols, r))["source_db"], dict(zip(proc_cols, r))["ID_CD_INTERNACAO"]):
            dict(zip(proc_cols, r)) for r in proc_rows
        }
    except Exception as e:
        print(f"    procedimentos skipped: {e}")
        proc_map = {}

    # ── Exames ──
    print("    Fetching exams ...")
    try:
        exam_cols, exam_rows = _exec("""
            SELECT e.ID_CD_INTERNACAO, e.source_db,
                COUNT(*) AS n_exames
            FROM agg_tb_capta_av_exame_caex e
            JOIN tmp_valid_ids t
              ON e.ID_CD_INTERNACAO = t.iid
             AND e.source_db = t.source_db
            GROUP BY e.ID_CD_INTERNACAO, e.source_db
        """)
        exam_map = {
            (dict(zip(exam_cols, r))["source_db"], dict(zip(exam_cols, r))["ID_CD_INTERNACAO"]):
            dict(zip(exam_cols, r)) for r in exam_rows
        }
    except Exception as e:
        print(f"    exames skipped: {e}")
        exam_map = {}

    # ── RAH Auditoria ──
    print("    Fetching audits ...")
    try:
        rah_cols, rah_rows = _exec("""
            SELECT r.ID_CD_INTERNACAO, r.source_db,
                COUNT(*) AS n_auditorias,
                MAX(r.DH_ADMISSAO) AS ultima_auditoria
            FROM agg_tb_formulario_rah_completo_frco r
            JOIN tmp_valid_ids t
              ON r.ID_CD_INTERNACAO = t.iid
             AND r.source_db = t.source_db
            GROUP BY r.ID_CD_INTERNACAO, r.source_db
        """)
        rah_map = {
            (dict(zip(rah_cols, r))["source_db"], dict(zip(rah_cols, r))["ID_CD_INTERNACAO"]):
            dict(zip(rah_cols, r)) for r in rah_rows
        }
    except Exception as e:
        print(f"    RAH skipped: {e}")
        rah_map = {}

    # ── Evolucao clinica ──
    print("    Fetching clinical evolution ...")
    try:
        evo_cols, evo_rows = _exec("""
            SELECT e.ID_CD_INTERNACAO, e.source_db,
                COUNT(*) AS n_evolucoes,
                MAX(e.DH_EVOLUCAO) AS ultima_evolucao
            FROM agg_tb_capta_evolucao_caev e
            JOIN tmp_valid_ids t
              ON e.ID_CD_INTERNACAO = t.iid
             AND e.source_db = t.source_db
            GROUP BY e.ID_CD_INTERNACAO, e.source_db
        """)
        evo_map = {
            (dict(zip(evo_cols, r))["source_db"], dict(zip(evo_cols, r))["ID_CD_INTERNACAO"]):
            dict(zip(evo_cols, r)) for r in evo_rows
        }
    except Exception as e:
        print(f"    evolucao skipped: {e}")
        evo_map = {}

    # ── Eventos adversos ──
    ev_map: dict = {}
    try:
        ev_cols, ev_rows = _exec("""
            SELECT ev.ID_CD_INTERNACAO, ev.source_db,
                COUNT(*) AS n_eventos
            FROM agg_tb_capta_eventos_adversos_caed ev
            JOIN tmp_valid_ids t
              ON ev.ID_CD_INTERNACAO = t.iid
             AND ev.source_db = t.source_db
            GROUP BY ev.ID_CD_INTERNACAO, ev.source_db
        """)
        ev_map = {
            (dict(zip(ev_cols, r))["source_db"], dict(zip(ev_cols, r))["ID_CD_INTERNACAO"]):
            dict(zip(ev_cols, r)) for r in ev_rows
        }
    except Exception as e:
        print(f"    eventos_adversos skipped: {e}")

    # ── OPME ──
    opme_map: dict = {}
    try:
        opme_cols, opme_rows = _exec("""
            SELECT op.ID_CD_INTERNACAO, op.source_db,
                COUNT(*) AS n_opme
            FROM agg_tb_capta_anexos_opme_caop op
            JOIN tmp_valid_ids t
              ON op.ID_CD_INTERNACAO = t.iid
             AND op.source_db = t.source_db
            GROUP BY op.ID_CD_INTERNACAO, op.source_db
        """)
        opme_map = {
            (dict(zip(opme_cols, r))["source_db"], dict(zip(opme_cols, r))["ID_CD_INTERNACAO"]):
            dict(zip(opme_cols, r)) for r in opme_rows
        }
    except Exception as e:
        print(f"    OPME skipped: {e}")

    # ── Readmissao (< 30 dias) ──
    # Find patients who had a PRIOR discharge within 30 days before this admission
    print("    Fetching readmission data ...")
    readmission_set: set = set()
    try:
        r30_cols, r30_rows = _exec(f"""
            WITH anomaly_adm AS (
                SELECT i.ID_CD_INTERNACAO, i.ID_CD_PACIENTE, i.source_db,
                       i.DH_ADMISSAO_HOSP
                FROM agg_tb_capta_internacao_cain i
                JOIN tmp_valid_ids t
                  ON i.ID_CD_INTERNACAO = t.iid
                 AND i.source_db = t.source_db
            ),
            prior_discharge AS (
                SELECT p.ID_CD_PACIENTE, p.source_db,
                       MAX(p.DH_FINALIZACAO) AS last_discharge
                FROM agg_tb_capta_internacao_cain p
                WHERE p.DH_FINALIZACAO IS NOT NULL
                  AND p.DH_FINALIZACAO > '2000-01-01'
                GROUP BY p.ID_CD_PACIENTE, p.source_db
            )
            SELECT a.ID_CD_INTERNACAO, a.source_db
            FROM anomaly_adm a
            JOIN prior_discharge pd
              ON a.ID_CD_PACIENTE = pd.ID_CD_PACIENTE
             AND a.source_db = pd.source_db
            WHERE DATEDIFF('day', pd.last_discharge::DATE, a.DH_ADMISSAO_HOSP::DATE) <= 30
              AND DATEDIFF('day', pd.last_discharge::DATE, a.DH_ADMISSAO_HOSP::DATE) > 0
        """)
        for row in r30_rows:
            d = dict(zip(r30_cols, row))
            readmission_set.add((d["source_db"], d["ID_CD_INTERNACAO"]))
    except Exception as e:
        print(f"    readmission check skipped: {e}")

    # ── Hospital names ──
    hosp_map: dict = {}
    try:
        hosp_cols, hosp_rows = _exec(
            "SELECT ID_CD_HOSPITAL, DS_DESCRICAO AS NM_HOSPITAL, source_db "
            "FROM agg_tb_crm_hospitais_crho"
        )
        for r in hosp_rows:
            d = dict(zip(hosp_cols, r))
            k = (d.get("source_db"), d.get("ID_CD_HOSPITAL"))
            hosp_map[k] = d.get("NM_HOSPITAL", "")
    except Exception:
        pass

    # ── Hospital-level baseline (per hospital_id + source_db) ──
    print("    Computing hospital baselines ...")
    hospital_baseline: dict = {}
    try:
        hb_cols, hb_rows = _exec(f"""
            WITH fat_agg AS (
                SELECT f.ID_CD_INTERNACAO, f.source_db,
                       SUM(f.VL_TOTAL) AS vl_total,
                       SUM(f.VL_GLOSA_FECHAMENTO) AS vl_glosa
                FROM agg_tb_fatura_fatu f
                GROUP BY f.ID_CD_INTERNACAO, f.source_db
            ),
            proc_agg AS (
                SELECT p.ID_CD_INTERNACAO, p.source_db,
                       COUNT(*) AS n_proc
                FROM agg_tb_fatura_procedimentos_fapr p
                GROUP BY p.ID_CD_INTERNACAO, p.source_db
            ),
            exam_agg AS (
                SELECT e.ID_CD_INTERNACAO, e.source_db,
                       COUNT(*) AS n_exam
                FROM agg_tb_capta_av_exame_caex e
                GROUP BY e.ID_CD_INTERNACAO, e.source_db
            )
            SELECT
                i.ID_CD_HOSPITAL,
                i.source_db,
                COUNT(DISTINCT i.ID_CD_INTERNACAO)  AS n_internacoes,
                AVG(CASE WHEN i.DH_FINALIZACAO IS NOT NULL
                    THEN DATEDIFF('day', i.DH_ADMISSAO_HOSP::DATE, i.DH_FINALIZACAO::DATE)
                    ELSE NULL END)                   AS avg_los,
                AVG(COALESCE(fa.vl_total, 0))        AS avg_billing,
                AVG(COALESCE(fa.vl_glosa, 0))        AS avg_glosa,
                AVG(COALESCE(pr.n_proc, 0))          AS avg_proc,
                AVG(COALESCE(ex.n_exam, 0))          AS avg_exam
            FROM agg_tb_capta_internacao_cain i
            LEFT JOIN fat_agg  fa ON i.ID_CD_INTERNACAO = fa.ID_CD_INTERNACAO AND i.source_db = fa.source_db
            LEFT JOIN proc_agg pr ON i.ID_CD_INTERNACAO = pr.ID_CD_INTERNACAO AND i.source_db = pr.source_db
            LEFT JOIN exam_agg ex ON i.ID_CD_INTERNACAO = ex.ID_CD_INTERNACAO AND i.source_db = ex.source_db
            WHERE i.DH_ADMISSAO_HOSP >= '{START_DATE_STR}'
              AND i.DH_ADMISSAO_HOSP <= '{REPORT_DATE_STR}'
            GROUP BY i.ID_CD_HOSPITAL, i.source_db
        """)
        for row in hb_rows:
            d = dict(zip(hb_cols, row))
            k = (d["source_db"], d["ID_CD_HOSPITAL"])
            hospital_baseline[k] = d
    except Exception as e:
        print(f"    hospital baseline skipped: {e}")

    # ── Merge enrichment ──
    for a in admissions:
        iid = a["ID_CD_INTERNACAO"]
        src = a.get("source_db", "")
        k   = (src, iid)
        a["fatura"]           = fatura_map.get(k, {})
        a["fatura_itens"]     = fit_map.get(k, {})
        a["glosa"]            = glosa_map.get(k, {})
        a["negociacoes"]      = neg_map.get(k, {})
        a["cids"]             = cid_map.get(k, {})
        a["procedimentos"]    = proc_map.get(k, {})
        a["exames"]           = exam_map.get(k, {})
        a["auditoria"]        = rah_map.get(k, {})
        a["evolucao"]         = evo_map.get(k, {})
        a["eventos_adversos"] = ev_map.get(k, {})
        a["opme"]             = opme_map.get(k, {})
        a["readmissao_30d"]   = k in readmission_set
        a["nm_hospital"]      = hosp_map.get(
            (src, a.get("ID_CD_HOSPITAL")),
            f"Hospital \\#{a.get('ID_CD_HOSPITAL', '?')}"
        )
        hosp_key = (src, a.get("ID_CD_HOSPITAL"))
        a["hospital_baseline"] = hospital_baseline.get(hosp_key, {})

    con.close()
    print(f"    Done in {time.time()-t0:.1f}s -- {len(admissions)} admissions enriched")
    return admissions, hospital_baseline


# ─────────────────────────────────────────────────────────────────
# Step 5b – Fetch DuckDB details for similar admissions
# ─────────────────────────────────────────────────────────────────

def _fetch_similar_details(similar_map: dict) -> dict:
    """
    For each set of similar admissions, fetch their LOS + billing from DuckDB
    to enable direct comparison in the justification section.
    Returns dict: (name_str) -> {"los": N, "vl_total": X, "source_db": ..., "iid": ...}
    """
    import duckdb

    print("    Fetching details for similar admissions ...")
    all_similar_ids = {}
    for sim_list in similar_map.values():
        for name, _sim in sim_list:
            if "/ID_CD_INTERNACAO_" in name:
                try:
                    src_part, id_part = name.split("/", 1)
                    iid = int(id_part.split("ID_CD_INTERNACAO_")[1])
                    all_similar_ids[name] = (src_part, iid)
                except Exception:
                    pass

    if not all_similar_ids:
        return {}

    con = duckdb.connect(str(DB_PATH))
    try:
        con.execute("""
            CREATE OR REPLACE TEMP TABLE tmp_sim_ids (
                source_db VARCHAR,
                iid       INTEGER
            )
        """)
        unique_pairs = list(set(all_similar_ids.values()))
        batch_size = 500
        for i in range(0, len(unique_pairs), batch_size):
            batch = unique_pairs[i:i + batch_size]
            vals  = ", ".join(f"('{src}', {iid})" for src, iid in batch)
            con.execute(f"INSERT INTO tmp_sim_ids VALUES {vals}")

        cur = con.execute(f"""
            SELECT i.ID_CD_INTERNACAO, i.source_db,
                CASE WHEN i.DH_FINALIZACAO IS NOT NULL
                    THEN DATEDIFF('day', i.DH_ADMISSAO_HOSP::DATE, i.DH_FINALIZACAO::DATE)
                    ELSE DATEDIFF('day', i.DH_ADMISSAO_HOSP::DATE, DATE '{REPORT_DATE_STR}')
                END AS los_dias,
                COALESCE(f.vl_total, 0) AS vl_total,
                COALESCE(p.n_proc, 0)   AS n_proc
            FROM agg_tb_capta_internacao_cain i
            JOIN tmp_sim_ids t
              ON i.ID_CD_INTERNACAO = t.iid AND i.source_db = t.source_db
            LEFT JOIN (
                SELECT ID_CD_INTERNACAO, source_db, SUM(VL_TOTAL) AS vl_total
                FROM agg_tb_fatura_fatu
                GROUP BY ID_CD_INTERNACAO, source_db
            ) f ON i.ID_CD_INTERNACAO = f.ID_CD_INTERNACAO AND i.source_db = f.source_db
            LEFT JOIN (
                SELECT ID_CD_INTERNACAO, source_db, COUNT(*) AS n_proc
                FROM agg_tb_fatura_procedimentos_fapr
                GROUP BY ID_CD_INTERNACAO, source_db
            ) p ON i.ID_CD_INTERNACAO = p.ID_CD_INTERNACAO AND i.source_db = p.source_db
        """)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        result = {}
        for row in rows:
            d = dict(zip(cols, row))
            key = f"{d['source_db']}/ID_CD_INTERNACAO_{d['ID_CD_INTERNACAO']}"
            result[key] = d
    except Exception as e:
        print(f"    similar_details skipped: {e}")
        result = {}
    finally:
        con.close()
    return result


# ─────────────────────────────────────────────────────────────────
# Step 6 – Generate LaTeX
# ─────────────────────────────────────────────────────────────────

def _build_justification(a: dict, similar_map: dict, similar_details: dict,
                          dim_analysis: dict) -> list[str]:
    """
    Build the LaTeX content for the 'Justificativa da Anomalia' section.
    Returns a list of LaTeX lines.
    """
    iid = a["ID_CD_INTERNACAO"]
    src = a.get("source_db", "")
    z   = a["Z_SCORE"]

    fat      = a.get("fatura", {})
    glo      = a.get("glosa", {})
    neg      = a.get("negociacoes", {})
    proc     = a.get("procedimentos", {})
    exam     = a.get("exames", {})
    evo      = a.get("evolucao", {})
    baseline = a.get("hospital_baseline", {})

    vl_total   = _safe_float(fat.get("vl_total"))
    vl_glosa   = _safe_float(glo.get("vl_glosado_total")) or _safe_float(fat.get("vl_glosa_total"))
    n_proc     = _safe_int(proc.get("n_procedimentos"))
    n_exam     = _safe_int(exam.get("n_exames"))
    n_neg      = _safe_int(neg.get("n_negociacoes"))
    n_evo      = _safe_int(evo.get("n_evolucoes"))
    los        = _safe_int(a.get("LOS_DIAS", 0))
    readmit    = a.get("readmissao_30d", False)

    avg_los    = _safe_float(baseline.get("avg_los"))
    avg_bill   = _safe_float(baseline.get("avg_billing"))
    avg_glosa  = _safe_float(baseline.get("avg_glosa"))
    avg_proc   = _safe_float(baseline.get("avg_proc"))
    avg_exam   = _safe_float(baseline.get("avg_exam"))

    L: list[str] = []

    # ── Natural language summary (Justificativa) ──
    summary_parts = []
    if avg_bill > 0 and vl_total > 0:
        mult = vl_total / avg_bill
        summary_parts.append(
            f"faturamento de {_brl_plain(vl_total)} ({mult:.1f}x a média de "
            f"{_brl_plain(avg_bill)} do hospital)"
        )
    elif vl_total > 0:
        summary_parts.append(f"faturamento de {_brl_plain(vl_total)}")

    if avg_proc > 0 and n_proc > 0:
        summary_parts.append(f"{n_proc} procedimentos (média do hospital: {avg_proc:.0f})")
    elif n_proc > 0:
        summary_parts.append(f"{n_proc} procedimentos")

    if avg_los > 0 and los > 0:
        summary_parts.append(f"LOS de {los} dias (média: {avg_los:.1f}d)")
    elif los > 0:
        summary_parts.append(f"LOS de {los} dias")

    if avg_glosa > 0 and vl_glosa > 0:
        glosa_rate     = (vl_glosa / vl_total * 100) if vl_total > 0 else 0
        avg_glosa_rate = (avg_glosa / avg_bill * 100) if avg_bill > 0 else 0
        summary_parts.append(
            f"taxa de glosa de {glosa_rate:.0f}\\% "
            f"(média: {avg_glosa_rate:.0f}\\%)"
        )
    elif vl_glosa > 0:
        glosa_rate = (vl_glosa / vl_total * 100) if vl_total > 0 else 0
        summary_parts.append(f"taxa de glosa de {glosa_rate:.0f}\\%")

    if n_evo > 0:
        summary_parts.append(f"{n_evo} evolu\\c{{c}}\\~{{o}}es clínicas")

    if readmit:
        summary_parts.append("readmiss\\~{{a}}o em menos de 30 dias")

    # Embedding dimension deviation
    dim_info = dim_analysis.get((src, iid), [])
    top2_dims = dim_info[:2] if dim_info else []

    # Build specific recommendation based on actual data deviations
    recs = []
    if avg_bill > 0 and vl_total > 0 and vl_total / avg_bill > 2.0:
        recs.append(f"Verificar justificativa para faturamento {vl_total/avg_bill:.1f}x acima da média do hospital")
    if avg_proc > 0 and n_proc > 0 and n_proc / avg_proc > 2.0:
        recs.append(f"Auditar os {n_proc} procedimentos realizados (média do hospital: {avg_proc:.0f})")
    if vl_total > 0 and vl_glosa > 0 and (vl_glosa / vl_total) > 0.10:
        recs.append(f"Investigar taxa de glosa de {vl_glosa/vl_total*100:.0f}\\% — acima do aceitável")
    if avg_los > 0 and los > 0 and los / avg_los > 2.0:
        recs.append(f"Avaliar permanência de {los} dias (média: {avg_los:.0f}d) — possível ineficiência ou complicação")
    if readmit:
        recs.append("Investigar causa da readmissão em menos de 30 dias — possível alta prematura")
    if n_evo == 0 and los > 3:
        recs.append(f"Internação de {los} dias sem registros de evolução clínica — possível falha de documentação")
    if n_proc == 0 and vl_total > 0:
        recs.append("Faturamento sem procedimentos registrados — possível inconsistência no registro")
    if not recs:
        recs.append(f"Internação apresenta z-score={z:.2f}, indicando perfil atípico para o hospital")

    if summary_parts:
        joined = "; ".join(summary_parts) + "."
        L.append(
            r"\vspace{2pt}\begin{tcolorbox}[colback=yellow!5,colframe=anomorange,"
            r"title={\textbf{Justificativa da Anomalia}},fonttitle=\bfseries\small,"
            r"left=4pt,right=4pt,top=3pt,bottom=3pt]"
        )
        L.append(r"{\small Esta internação apresenta: " + _escape_latex(joined) + r"}")
        if readmit:
            L.append(
                r"\\\textcolor{anomred}{\textbf{Alerta:} Paciente readmitido em menos de 30 dias.}"
            )
        # Specific recommendations (not generic boilerplate)
        L.append(r"\\\textbf{Recomendações específicas:}")
        L.append(r"\begin{itemize}[nosep,leftmargin=12pt]")
        for rec in recs:
            L.append(r"\item {\small " + _escape_latex(rec) + r"}")
        L.append(r"\end{itemize}")
        L.append(r"\end{tcolorbox}")
    else:
        L.append(
            r"\vspace{2pt}\begin{tcolorbox}[colback=yellow!5,colframe=anomorange,"
            r"title={\textbf{Justificativa da Anomalia}}]"
        )
        L.append(r"\begin{itemize}[nosep,leftmargin=12pt]")
        for rec in recs:
            L.append(r"\item {\small " + _escape_latex(rec) + r"}")
        L.append(r"\end{itemize}")
        L.append(r"\end{tcolorbox}")

    # ── Baseline comparison table ──
    if avg_bill > 0 or avg_los > 0:
        L.append(r"\vspace{4pt}{\footnotesize\textbf{Comparação com Baseline do Hospital:}}")
        L.append(
            r"\begin{center}\begin{tabular}{l r r r}"
            r"\toprule"
            r"\textbf{Métrica} & \textbf{Esta Intern.} & "
            r"\textbf{Média Hospital} & \textbf{Desvio} \\"
            r"\midrule"
        )
        rows_bl = []
        if avg_los > 0:
            rows_bl.append((
                "LOS",
                f"{los}d",
                f"{avg_los:.1f}d",
                _pct_diff(los, avg_los),
            ))
        if avg_bill > 0:
            rows_bl.append((
                "Faturamento",
                _brl(vl_total) if vl_total else "---",
                _brl(avg_bill),
                _pct_diff(vl_total, avg_bill) if vl_total else "---",
            ))
        if avg_proc > 0:
            rows_bl.append((
                "Procedimentos",
                str(n_proc),
                f"{avg_proc:.0f}",
                _pct_diff(n_proc, avg_proc) if n_proc else "---",
            ))
        if avg_exam > 0:
            rows_bl.append((
                "Exames",
                str(n_exam),
                f"{avg_exam:.0f}",
                _pct_diff(n_exam, avg_exam) if n_exam else "---",
            ))
        if avg_glosa > 0:
            rows_bl.append((
                "Glosas",
                _brl(vl_glosa) if vl_glosa else "---",
                _brl(avg_glosa),
                _pct_diff(vl_glosa, avg_glosa) if vl_glosa else "---",
            ))
        for metric, this_val, avg_val, diff in rows_bl:
            # Colour deviations red if above +50%
            diff_colored = diff
            if diff not in ("---", "N/A") and diff.startswith("+"):
                try:
                    pct_val = float(diff.replace("+", "").replace("\\%", "").replace("%", ""))
                    if pct_val >= 100:
                        diff_colored = r"\textcolor{anomred}{\textbf{" + diff + r"}}"
                    elif pct_val >= 50:
                        diff_colored = r"\textcolor{anomorange}{" + diff + r"}"
                except Exception:
                    pass
            L.append(
                _escape_latex(metric) + " & " + this_val + " & " +
                avg_val + " & " + diff_colored + r" \\"
            )
        L.append(r"\bottomrule\end{tabular}\end{center}" + "\n")

    # ── Similar admissions comparison ──
    similar = similar_map.get((src, iid), [])
    if similar:
        L.append(r"\vspace{2pt}{\footnotesize\textbf{Internações Similares (cosine):}\\[2pt]}")
        L.append(
            r"\begin{tabular}{l r r r l}"
            r"\toprule"
            r"\textbf{Intern.} & \textbf{Simil.} & \textbf{LOS} & \textbf{Faturado} & \textbf{Status} \\"
            r"\midrule"
        )
        for sname, ssim in similar:
            sd = similar_details.get(sname, {})
            s_los  = _safe_int(sd.get("los_dias"))
            s_bill = _safe_float(sd.get("vl_total"))
            # Simple label for node id
            if "/ID_CD_INTERNACAO_" in sname:
                try:
                    s_src, s_id_part = sname.split("/", 1)
                    s_raw_id = s_id_part.split("ID_CD_INTERNACAO_")[1]
                    sid_label = f"\\#{s_raw_id} ({_escape_latex(s_src)})"
                except Exception:
                    sid_label = _escape_latex(sname[:30])
            else:
                sid_label = _escape_latex(sname[:30])
            bill_str = _brl(s_bill) if s_bill else "---"
            los_str  = f"{s_los}d" if s_los else "---"
            L.append(
                sid_label + " & " +
                f"{ssim:.3f}" + " & " +
                los_str + " & " +
                bill_str + " & " +
                r"normal \\"
            )
        # Comparison summary
        valid_bills = [_safe_float(similar_details.get(n, {}).get("vl_total"))
                       for n, _ in similar
                       if _safe_float(similar_details.get(n, {}).get("vl_total")) > 0]
        if valid_bills and vl_total > 0:
            avg_sim_bill = sum(valid_bills) / len(valid_bills)
            if vl_total > avg_sim_bill * 1.2:
                comparison_note = (
                    r"\multicolumn{5}{l}{\textcolor{anomred}{\small "
                    r"$\rightarrow$ Esta internação é significativamente "
                    r"mais cara que suas similares.}} \\"
                )
                L.append(comparison_note)
        L.append(r"\bottomrule\end{tabular}" + "\n")

    # ── Top deviating embedding dimensions ──
    if dim_info:
        L.append(r"\vspace{2pt}{\footnotesize\textbf{Dimensões de embedding mais desviantes:} ")
        dim_parts = []
        for d, dz, meaning in dim_info[:4]:
            sign = "+" if dz >= 0 else ""
            dim_parts.append(f"dim[{d}] ({_escape_latex(meaning)}, z={sign}{dz:.1f})")
        L.append(", ".join(dim_parts) + "}")

    return L


def _generate_latex(admissions: list[dict], similar_map: dict,
                    similar_details: dict, dim_analysis: dict) -> str:
    import numpy as np

    sources: dict[str, list[dict]] = {}
    for a in admissions:
        src = a.get("source_db", "DESCONHECIDO")
        sources.setdefault(src, []).append(a)

    total_anomalies = len(admissions)
    total_critical  = sum(1 for a in admissions if a["Z_SCORE"] >= 5)
    total_high      = sum(1 for a in admissions if 3 <= a["Z_SCORE"] < 5)
    total_moderate  = sum(1 for a in admissions if 2 <= a["Z_SCORE"] < 3)
    total_vl_glosa  = sum(_safe_float(a["glosa"].get("vl_glosado_total")) for a in admissions)
    total_vl_fatura = sum(_safe_float(a["fatura"].get("vl_total")) for a in admissions)
    avg_los = float(np.mean([_safe_float(a.get("LOS_DIAS", 0)) for a in admissions])) if admissions else 0.0

    L = []

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
\fancyhead[L]{\textcolor{jcubeblue}{\textbf{JCUBE Digital Twin}} \textcolor{jcubegray}{\small | Relatório de Anomalias --- V4 (Explicado)}}
\fancyhead[R]{\textcolor{jcubegray}{\small 23/03/2026}}
\fancyfoot[C]{\textcolor{jcubegray}{\thepage}}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0.4pt}

\titleformat{\section}{\large\bfseries\color{jcubeblue}}{\thesection}{1em}{}[\titlerule]
\titleformat{\subsection}{\normalsize\bfseries\color{darkblue}}{\thesubsection}{1em}{}

\hypersetup{colorlinks=true,linkcolor=jcubeblue,pdftitle={JCUBE V4 Anomalias Explicadas}}

\begin{document}
\setlength{\parindent}{0pt}
\setlength{\parskip}{3pt}
""")

    # ── Title page ──
    L.append(r"""\begin{titlepage}
\begin{center}
\vspace*{1.5cm}
{\Huge\bfseries\textcolor{jcubeblue}{JCUBE}}\\[0.2cm]
{\large\textcolor{jcubegray}{Digital Twin Analytics Platform --- Modelo V4 (Relatório Explicado)}}\\[1.2cm]
\begin{tcolorbox}[colback=jcubeblue,colframe=jcubeblue,coltext=white,width=0.92\textwidth,halign=center]
{\LARGE\bfseries Relatório de Anomalias em Internações}\\[0.3cm]
{\large Análise via Embeddings do Gêmeo Digital --- Graph-JEPA V4 (35,2M nós $\times$ 64 dim)}\\[0.2cm]
{\normalsize Com Justificativa de Auditoria por Internação}
\end{tcolorbox}
\vspace{0.8cm}
""")
    L.append(
        r"{\Large Período: \textbf{" + _escape_latex(START_DATE_STR) +
        r"} a \textbf{" + _escape_latex(REPORT_DATE_STR) + r"}}\\[0.4cm]"
    )
    L.append(r"{\large Gerado em: \textbf{23 de março de 2026}}\\[1.5cm]")

    L.append(
        r"""\begin{tabular}{ccc}
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
"""
    )
    if total_vl_fatura > 0:
        L.append(
            r"LOS Médio: \textbf{" + f"{avg_los:.1f}" +
            r"} dias \quad | \quad Total Faturado: \textbf{" + _brl(total_vl_fatura) + r"}\\"
        )
    else:
        L.append(r"LOS Médio: \textbf{" + f"{avg_los:.1f}" + r"} dias\\")
    L.append(r"""\end{tcolorbox}

\vfill
{\small\textcolor{jcubegray}{
Metodologia: Z-score sobre distância euclidiana ao centróide dos embeddings JEPA V4\\
Limiar: z $>$ """ + str(Z_THRESHOLD) + r""" --- Modelo V4: 35.2M nós $\times$ 64 dim\\
Cada anomalia inclui: Justificativa, Baseline do Hospital, Similares, Dimensões de Embedding
}}
\end{center}
\end{titlepage}
""")

    L.append(r"\tableofcontents\clearpage")

    # ── Executive Summary ──
    L.append(r"\section{Sumário Executivo}")
    L.append(
        r"""
Este relatório apresenta \textbf{todas as internações anômalas} detectadas pelo
\textit{Digital Twin} JCUBE no período de \textbf{""" +
        _escape_latex(START_DATE_STR) + r"""} a \textbf{""" +
        _escape_latex(REPORT_DATE_STR) + r"""}.

Para cada anomalia, o relatório inclui uma \textbf{Justificativa de Auditoria} baseada em:
comparação com o baseline do hospital, internações semanticamente similares,
análise das dimensões de embedding mais desviantes e dados concretos do DuckDB.

\subsection{Métricas Globais}
\begin{center}
\begin{tabular}{lr}
\toprule
\textbf{Métrica} & \textbf{Valor} \\
\midrule
Total de anomalias detectadas & """ + str(total_anomalies) + r""" \\
Sistemas hospitalares (fontes) & """ + str(len(sources)) + r""" \\
Críticos (z $\geq$ 5) & \textcolor{anomred}{\textbf{""" + str(total_critical) + r"""}} \\
Altos (3 $\leq$ z $<$ 5) & \textcolor{anomorange}{\textbf{""" + str(total_high) + r"""}} \\
Moderados (2 $\leq$ z $<$ 3) & \textcolor{anomyellow}{\textbf{""" + str(total_moderate) + r"""}} \\
LOS médio das anomalias & """ + f"{avg_los:.1f}" + r""" dias \\
"""
    )
    if total_vl_fatura > 0:
        L.append(r"Total faturado (anomalias) & " + _brl(total_vl_fatura) + r" \\" + "\n")
    if total_vl_glosa > 0:
        L.append(r"Total glosado (anomalias) & \textcolor{anomred}{" + _brl(total_vl_glosa) + r"} \\" + "\n")
    L.append(r"""\bottomrule
\end{tabular}
\end{center}
""")

    # Summary by source
    L.append(r"\subsection{Anomalias por Sistema Hospitalar}")
    L.append(r"""\begin{center}
\begin{longtable}{lrrrr}
\toprule
\textbf{Sistema / Fonte} & \textbf{Total} & \textbf{Crít.} & \textbf{Alto} & \textbf{LOS Méd.} \\
\midrule
\endhead
\bottomrule
\endfoot
""")
    for src in sorted(sources.keys()):
        alist  = sources[src]
        n      = len(alist)
        nc     = sum(1 for a in alist if a["Z_SCORE"] >= 5)
        nh     = sum(1 for a in alist if 3 <= a["Z_SCORE"] < 5)
        avg_l  = float(np.mean([_safe_float(a.get("LOS_DIAS", 0)) for a in alist]))
        L.append(
            _escape_latex(src) + r" & " + str(n) +
            r" & \textcolor{anomred}{" + str(nc) + r"}" +
            r" & \textcolor{anomorange}{" + str(nh) + r"}" +
            r" & " + f"{avg_l:.1f}" + r" \\" + "\n"
        )
    L.append(r"\end{longtable}\end{center}" + "\n\n")
    L.append(r"\clearpage")

    # ── Per-source sections ──
    for src in sorted(sources.keys()):
        alist         = sorted(sources[src], key=lambda a: -a["Z_SCORE"])
        section_label = _escape_latex(src)
        L.append(r"\section{" + section_label + r"}")

        n         = len(alist)
        nc        = sum(1 for a in alist if a["Z_SCORE"] >= 5)
        nh        = sum(1 for a in alist if 3 <= a["Z_SCORE"] < 5)
        nm        = sum(1 for a in alist if 2 <= a["Z_SCORE"] < 3)
        avg_l     = float(np.mean([_safe_float(a.get("LOS_DIAS", 0)) for a in alist]))
        max_z     = max((a["Z_SCORE"] for a in alist), default=0.0)
        vl_glo    = sum(_safe_float(a["glosa"].get("vl_glosado_total")) for a in alist)
        vl_fat    = sum(_safe_float(a["fatura"].get("vl_total")) for a in alist)

        L.append(
            r"""\begin{tcolorbox}[colback=lightblue,colframe=jcubeblue,title={\textbf{Resumo --- """ +
            section_label + r"""}}]
\begin{tabular}{ll@{\quad}ll@{\quad}ll}
Total: \textbf{""" + str(n) + r"""} &
Crít.: \textcolor{anomred}{\textbf{""" + str(nc) + r"""}} &
Altos: \textcolor{anomorange}{\textbf{""" + str(nh) + r"""}} &
Mod.: \textcolor{anomyellow}{\textbf{""" + str(nm) + r"""}} &
LOS méd.: \textbf{""" + f"{avg_l:.1f}" + r"""d} &
Maior z: \textbf{""" + f"{max_z:.2f}" + r"""} \\
"""
        )
        if vl_fat > 0:
            L.append(r"\multicolumn{2}{l}{Total faturado: \textbf{" + _brl(vl_fat) + r"}} & ")
        if vl_glo > 0:
            L.append(r"\multicolumn{2}{l}{\textcolor{anomred}{Total glosado: \textbf{" + _brl(vl_glo) + r"}}} & & \\")
        L.append(r"""
\end{tabular}
\end{tcolorbox}
""")

        # Summary table
        L.append(r"\subsection{Tabela Resumo}")
        L.append(r"""\begin{center}
\begin{longtable}{>{\scriptsize}r>{\scriptsize}r>{\scriptsize}c>{\scriptsize}c>{\scriptsize}r>{\scriptsize}r>{\scriptsize}r>{\scriptsize}r}
\toprule
\textbf{Intern.} & \textbf{Pac.} & \textbf{Admissão} & \textbf{Alta} & \textbf{LOS} & \textbf{Z} & \textbf{Faturado} & \textbf{Glosado} \\
\midrule
\endhead
\bottomrule
\endfoot
""")
        for a in alist:
            iid   = a["ID_CD_INTERNACAO"]
            pid   = a.get("ID_CD_PACIENTE", "?")
            adm   = _fmt_date(a.get("DH_ADMISSAO_HOSP"))
            alta  = _fmt_date(a.get("DH_FINALIZACAO"))
            los   = _safe_int(a.get("LOS_DIAS", 0))
            z     = a["Z_SCORE"]
            vl_f  = _safe_float(a["fatura"].get("vl_total"))
            vl_g  = _safe_float(a["glosa"].get("vl_glosado_total"))
            color = _z_color(z)
            vl_f_str = _brl(vl_f) if vl_f > 0 else "---"
            vl_g_str = r"\textcolor{anomred}{" + _brl(vl_g) + r"}" if vl_g > 0 else "---"
            L.append(
                f"\\textcolor{{{color}}}{{\\textbf{{{iid}}}}} & {pid} & {adm} & {alta} & {los}d & "
                f"\\textcolor{{{color}}}{{{z:.2f}}} & {vl_f_str} & {vl_g_str}" + r" \\" + "\n"
            )
        L.append(r"\end{longtable}\end{center}" + "\n")

        # Detailed cards
        L.append(r"\subsection{Fichas Detalhadas}")
        for a in alist:
            iid      = a["ID_CD_INTERNACAO"]
            pid      = a.get("ID_CD_PACIENTE", "?")
            z        = a["Z_SCORE"]
            los      = _safe_int(a.get("LOS_DIAS", 0))
            adm      = _fmt_date(a.get("DH_ADMISSAO_HOSP"))
            alta     = _fmt_date(a.get("DH_FINALIZACAO"))
            color    = _z_color(z)
            severity = _z_label(z)
            nm_hosp  = _escape_latex(str(a.get("nm_hospital") or "---"))
            senha    = _escape_latex(str(a.get("NR_SENHA") or "---"))
            guia     = _escape_latex(str(a.get("NR_GUIA_AUTORIZACAO") or "---"))
            readmit  = a.get("readmissao_30d", False)
            readmit_flag = r"\textcolor{anomred}{\textbf{SIM}}" if readmit else "N\u00e3o"

            card_title = (
                f"Interna\\c{{c}}\\~{{a}}o \\#{iid} | {section_label} | "
                f"{severity} (z={z:.2f})"
            )
            L.append(r"\begin{anomalycard}[" + card_title + r"]{" + color + r"}")

            # Header: Patient, dates, hospital
            L.append(
                r"\textbf{Dados da Internação:} " +
                r"Paciente \#" + str(pid) +
                r" $\mid$ Admissão: " + adm +
                r" $\mid$ Alta: " + alta +
                r" $\mid$ LOS: \textbf{" + str(los) + r"d}" +
                r" $\mid$ Readmissão {<}30d: " + readmit_flag + r"\\"
            )
            L.append(
                r"\textbf{Hospital:} " + nm_hosp +
                r"\quad\textbf{Fonte:} " + _escape_latex(src) +
                r"\quad\textbf{Senha:} " + senha +
                r"\quad\textbf{Guia:} " + guia + r"\\"
            )

            # CIDs
            cids_data  = a.get("cids", {})
            n_cids     = _safe_int(cids_data.get("n_cids"))
            cid_princ  = _escape_latex(_truncate(str(cids_data.get("cid_principal") or ""), 120))
            all_cids   = _escape_latex(_truncate(str(cids_data.get("cids") or ""), 280))
            if n_cids > 0:
                L.append(r"\textbf{CID Principal:} {\small " + cid_princ + r"}\quad\textbf{Total CIDs:} " + str(n_cids) + r"\\" + "\n")
                if all_cids and all_cids != "---":
                    L.append(r"\textbf{CIDs:} {\scriptsize " + all_cids + r"}\\" + "\n")

            for label, field, maxl in [
                ("Diagn.\óstico", "DS_DESCRICAO", 180),
                ("Hist\órico",    "DS_HISTORICO", 180),
                ("Motivo",            "DS_MOTIVO",    130),
            ]:
                val = _truncate(str(a.get(field) or ""), maxl)
                if val and val != "---":
                    L.append(r"\textbf{" + label + r":} {\scriptsize " + _escape_latex(val) + r"}\\" + "\n")

            # ── Justificativa da Anomalia (new section) ──
            just_lines = _build_justification(a, similar_map, similar_details, dim_analysis)
            L.extend(just_lines)

            # Financial details
            fat = a.get("fatura", {})
            glo = a.get("glosa", {})
            neg = a.get("negociacoes", {})
            if fat or glo or neg:
                L.append(r"\vspace{2pt}{\footnotesize\begin{tabular}{@{}ll@{\hspace{12pt}}ll@{\hspace{12pt}}ll@{}}" + "\n")
                L.append(r"\toprule\multicolumn{6}{c}{\textbf{Dados Financeiros Detalhados}}\\\midrule" + "\n")
                if fat:
                    L.append(
                        r"VL Total & " + _brl(fat.get("vl_total")) +
                        r" & VL RH & " + _brl(fat.get("vl_rh")) +
                        r" & VL Mat. & " + _brl(fat.get("vl_mat")) + r" \\" + "\n"
                    )
                    L.append(
                        r"VL Med. & " + _brl(fat.get("vl_med")) +
                        r" & VL OPME & " + _brl(fat.get("vl_opme")) +
                        r" & VL SADT & " + _brl(fat.get("vl_sadt")) + r" \\" + "\n"
                    )
                    L.append(
                        r"VL Líq. & " + _brl(fat.get("vl_liquido")) +
                        r" & Glosa Fat. & \textcolor{anomred}{" + _brl(fat.get("vl_glosa_total")) + r"}" +
                        r" & Diverg. & \textcolor{anomorange}{" + _brl(fat.get("vl_divergencia")) + r"}" +
                        r" \\" + "\n"
                    )
                if glo:
                    L.append(r"\midrule\multicolumn{6}{c}{\textbf{Glosas}}\\\midrule" + "\n")
                    L.append(
                        r"N Glosas & \textbf{" + str(_safe_int(glo.get("n_glosas"))) + r"}" +
                        r" & VL Glosado & \textcolor{anomred}{\textbf{" + _brl(glo.get("vl_glosado_total")) + r"}}" +
                        r" & Aceito & " + _brl(glo.get("vl_aceito")) + r" \\" + "\n"
                    )
                if neg:
                    L.append(r"\midrule\multicolumn{6}{c}{\textbf{Negociações de Auditoria}}\\\midrule" + "\n")
                    tipos = _escape_latex(_truncate(str(neg.get("tipos_negociacao") or ""), 60))
                    L.append(
                        r"N Negoc. & \textbf{" + str(_safe_int(neg.get("n_negociacoes"))) + r"}" +
                        r" & VL Negoc. & " + _brl(neg.get("vl_negociado_total")) +
                        r" & Tipos & {\scriptsize " + tipos + r"} \\" + "\n"
                    )
                L.append(r"\bottomrule\end{tabular}}" + "\n")

            # Activity summary
            proc = a.get("procedimentos", {})
            fit  = a.get("fatura_itens", {})
            evo  = a.get("evolucao", {})
            aud  = a.get("auditoria", {})
            ev   = a.get("eventos_adversos", {})
            opme = a.get("opme", {})
            exm  = a.get("exames", {})

            L.append(
                r"\vspace{2pt}{\footnotesize " +
                r"\textbf{Proced.:} " + str(_safe_int(proc.get("n_procedimentos"))) +
                r"\quad\textbf{Exames:} " + str(_safe_int(exm.get("n_exames"))) +
                r"\quad\textbf{Evoluções:} " + str(_safe_int(evo.get("n_evolucoes"))) +
                r"\quad\textbf{Audit. RAH:} " + str(_safe_int(aud.get("n_auditorias"))) +
                r"\quad\textbf{Eventos Adv.:} " + str(_safe_int(ev.get("n_eventos"))) +
                r"\quad\textbf{Itens Fat.:} " + str(_safe_int(fit.get("n_itens"))) +
                r"\quad\textbf{OPME:} " + str(_safe_int(opme.get("n_opme"))) +
                r"}" + "\n"
            )

            L.append(r"\end{anomalycard}" + "\n\n")

        L.append(r"\clearpage")

    # ── Appendix ──
    L.append(r"\section*{Apêndice: Metodologia de Detecção e Justificação}")
    L.append(r"\addcontentsline{toc}{section}{Apêndice: Metodologia}")
    L.append(r"""
\subsection*{1. Modelo Graph-JEPA V4}
O modelo \textit{Graph-JEPA V4} foi treinado sobre o grafo de conhecimento JCUBE com \textbf{35,2M nós}
e \textbf{64 dimensões} de embedding. Cada nó representa uma entidade (internação, paciente,
fatura, médico, etc.) e as arestas representam relações entre elas.

\subsection*{2. Detecção de Anomalias via Z-score}
Para cada internação com nó no grafo:
\begin{enumerate}[nosep]
  \item Calcula-se a \textbf{distância euclidiana} do embedding ao \textbf{centróide} de todas as internações.
  \item Calcula-se o \textbf{z-score}: $z = \frac{d - \mu}{\sigma}$, onde $\mu$ e $\sigma$ são a média e desvio padrão das distâncias.
  \item Internações com $z > """ + str(Z_THRESHOLD) + r"""$ são classificadas como anômalas.
\end{enumerate}

\subsection*{3. Classificação de Severidade}
\begin{itemize}[nosep]
  \item \textcolor{anomred}{\textbf{CR\'ITICO}}: z $\geq$ 5
  \item \textcolor{anomorange}{\textbf{ALTO}}: 3 $\leq$ z $<$ 5
  \item \textcolor{anomyellow}{\textbf{MODERADO}}: 2 $\leq$ z $<$ 3
\end{itemize}

\subsection*{4. Justificativa de Auditoria (Novo em V2)}
Para cada anomalia, o relatório calcula:
\begin{itemize}[nosep]
  \item \textbf{Baseline do hospital}: média de LOS, faturamento, procedimentos, exames e glosas
    para todas as internações do mesmo hospital no período.
  \item \textbf{Internações similares}: as 5 internações com maior similaridade por cosseno
    no espaço de embeddings, com seus LOS e faturamentos reais.
  \item \textbf{Dimensões de embedding}: as dimensões que mais desviam do centróide,
    mapeadas para seu significado funcional (faturamento, procedimentos, glosa, etc.).
  \item \textbf{Readmissão}: se o paciente foi readmitido em menos de 30 dias.
\end{itemize}

\subsection*{5. Mapeamento de Dimensões de Embedding}
\begin{itemize}[nosep]
  \item dim[16]: padrão de faturamento
  \item dim[28]: complexidade clínica
  \item dim[30]: volume de procedimentos
  \item dim[46]: trajetória temporal
  \item dim[53]: risco de glosa
  \item dim[61]: padrão operacional
\end{itemize}
""")
    L.append(r"\end{document}")
    return "\n".join(L)


# ─────────────────────────────────────────────────────────────────
# Step 7 – Compile LaTeX → PDF
# ─────────────────────────────────────────────────────────────────

def _compile_latex(latex_content: str, output_pdf: str):
    import subprocess
    import os
    from pathlib import Path

    print("[6/6] Compiling PDF ...")
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
    # 8.4 GB embeddings + DuckDB queries -- use a memory-rich CPU instance
    cpu=8.0,
    memory=32768,   # 32 GB RAM
    timeout=3600,   # 1 hour max
)
def generate_report():
    import time
    import os

    t_start = time.time()
    print("=" * 70)
    print("JCUBE V4 Anomaly Report Generator v2 (Modal) -- Explained")
    print(f"Period : {START_DATE_STR} -> {REPORT_DATE_STR}")
    print(f"Weights: {WEIGHTS_PATH}")
    print(f"DB     : {DB_PATH}")
    print(f"Output : {OUTPUT_PDF}")
    print(f"Z thr  : {Z_THRESHOLD}")
    print("=" * 70)

    # Sanity check volumes are mounted
    for p in [GRAPH_PARQUET, WEIGHTS_PATH, DB_PATH]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}")

    # 1. Load twin
    unique_nodes, embeddings, node_to_idx, internacao_mask = _load_twin()

    # 2. Detect anomalies (returns centroid + vecs for later use)
    records, anomaly_z, _centroid, _vecs, _names = _detect_anomalies(
        embeddings, internacao_mask, unique_nodes
    )

    # 3. Analyze embedding dimensions per anomaly
    dim_analysis = _analyze_embedding_dimensions(
        embeddings, node_to_idx, internacao_mask, unique_nodes, records
    )

    # 4. Fetch DuckDB details + hospital baselines
    admissions, _hospital_baseline = _fetch_admission_details(records, anomaly_z)
    print(f"    Total admissions to report: {len(admissions)}")

    # 5. Batch similar admissions lookup
    valid_records = [(a["source_db"], a["ID_CD_INTERNACAO"]) for a in admissions]
    print("[5/6] Computing similar admissions ...")
    similar_map = _batch_find_similar(
        embeddings, node_to_idx, internacao_mask, unique_nodes,
        valid_records, k=5,
    )

    # 5b. Fetch DuckDB details for similar admissions
    similar_details = _fetch_similar_details(similar_map)

    # 6. Generate LaTeX
    print("[5/6] Generating LaTeX document ...")
    latex = _generate_latex(admissions, similar_map, similar_details, dim_analysis)

    # 7. Compile PDF
    _compile_latex(latex, OUTPUT_PDF)

    # Commit volume so changes persist
    data_vol.commit()

    elapsed = time.time() - t_start
    print(f"\nFinished in {elapsed:.1f}s")
    print(f"Report saved to Modal volume jcube-data at: {OUTPUT_PDF}")
    print("Download with:")
    print(f"  modal volume get jcube-data reports/anomaly_report_v4_explained_2026_03.pdf ./anomaly_report_v4_explained_2026_03.pdf")
    return OUTPUT_PDF
