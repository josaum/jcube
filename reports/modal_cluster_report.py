#!/usr/bin/env python3
"""
Modal script: JCUBE V5 Patient Clustering Report
Runs on Modal, loads 8.4 GB V5 embeddings from jepa-cache volume,
clusters PACIENTE nodes with k-means, queries DuckDB from jcube-data volume,
and generates a professional LaTeX -> PDF report in pt-BR.

Key features (V5):
  - Auto-named clusters based on dominant characteristics
  - Per-cluster narrative explaining who these patients are and what to do
  - Per-hospital breakdown within each cluster
  - Clusters sorted by criticality (death rate + readmission first)
  - Executive summary with traffic-light color coding
  - Comparison against global baseline for each metric
  - One cluster card per page
  - All text in pt-BR, UTF-8, no LaTeX accent commands

Steps:
  1. Load node vocabulary from graph parquet (pa.chunked_array dedup)
  2. Load V5 embeddings (35.2M x 64)
  3. Extract PACIENTE node rows
  4. Run scikit-learn KMeans (k=18) on patient embeddings
  5. For each cluster query DuckDB: discharge types, DRG, LOS, CIDs,
     readmissions, hospital, billing, IN_SITUACAO
  6. Generate LaTeX report with auto-names, narratives, baselines
  7. Compile to PDF with pdflatex (2 passes)

Usage:
    modal run --detach reports/modal_cluster_report.py
"""
from __future__ import annotations

import modal

# ─────────────────────────────────────────────────────────────────
# Modal App + Volumes
# ─────────────────────────────────────────────────────────────────

app = modal.App("jcube-cluster-report-v5")

jepa_cache = modal.Volume.from_name("jepa-cache", create_if_missing=False)
data_vol   = modal.Volume.from_name("jcube-data",  create_if_missing=False)

VOLUMES = {
    "/cache": jepa_cache,
    "/data":  data_vol,
}

# ─────────────────────────────────────────────────────────────────
# Container image -- torch (CPU), duckdb, pyarrow, scikit-learn, latex
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
WEIGHTS_PATH  = "/cache/tkg-v5/node_emb_epoch_1.pt"
DB_PATH       = "/data/aggregated_fixed_union.db"
OUTPUT_DIR    = "/data/reports"
OUTPUT_PDF    = f"{OUTPUT_DIR}/cluster_report_v5_2026_03.pdf"

REPORT_DATE_STR = "2026-03-23"

K_CLUSTERS = 18

# ─────────────────────────────────────────────────────────────────
# Helpers
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
    f = _safe_float(v)
    if f == 0:
        return "---"
    return "R$ {:,.2f}".format(f).replace(",", "X").replace(".", ",").replace("X", ".")


def _truncate(s: str, max_len: int = 120) -> str:
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


def _pct_plain(num: float, den: float) -> str:
    if den == 0:
        return "---"
    return f"{100.0 * num / den:.1f}%"


def _compare_str(val: float, baseline: float, fmt: str = ".1f") -> str:
    """Return 'val (Xx baseline)' string."""
    if baseline == 0:
        return f"{val:{fmt}}"
    ratio = val / baseline
    return f"{val:{fmt}} ({ratio:.1f}x media)"


# ─────────────────────────────────────────────────────────────────
# Step 1 -- Load node vocabulary + embeddings
# ─────────────────────────────────────────────────────────────────

def _load_paciente_embeddings():
    import time
    import numpy as np
    import torch
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    import pyarrow as pa

    print("[1/5] Loading node vocabulary from graph parquet ...")
    t0 = time.time()
    table = pq.read_table(GRAPH_PARQUET, columns=["subject_id", "object_id"])
    subj  = table.column("subject_id")
    obj   = table.column("object_id")
    all_nodes    = pa.chunked_array(subj.chunks + obj.chunks)
    unique_nodes = pc.unique(all_nodes).to_numpy(zero_copy_only=False).astype(object)
    del table, subj, obj, all_nodes
    print(f"    {len(unique_nodes):,} unique nodes  ({time.time()-t0:.1f}s)")

    print("[1/5] Loading V5 embeddings ...")
    t1 = time.time()
    state = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
    if isinstance(state, torch.Tensor):
        embeddings = state.numpy().astype(np.float32)
    elif isinstance(state, dict) and "weight" in state:
        embeddings = state["weight"].numpy().astype(np.float32)
    else:
        embeddings = list(state.values())[0].numpy().astype(np.float32)
    print(f"    Embeddings shape: {embeddings.shape}  ({time.time()-t1:.1f}s)")

    if len(unique_nodes) != embeddings.shape[0]:
        raise ValueError(
            f"Vocab mismatch: {len(unique_nodes):,} nodes vs "
            f"{embeddings.shape[0]:,} embedding rows"
        )

    # V4 node format: "GHO-BRADESCO/ID_CD_PACIENTE_12345"
    paciente_mask = np.array(
        ["_PACIENTE_" in str(n) for n in unique_nodes], dtype=bool
    )
    print(f"    PACIENTE nodes: {paciente_mask.sum():,}")

    pac_nodes = unique_nodes[paciente_mask]
    pac_embs  = embeddings[paciente_mask]

    return pac_nodes, pac_embs


# ─────────────────────────────────────────────────────────────────
# Step 2 -- K-means clustering
# ─────────────────────────────────────────────────────────────────

def _cluster_patients(pac_nodes, pac_embs):
    import time
    import numpy as np
    from sklearn.cluster import KMeans

    n_patients = len(pac_nodes)
    k = K_CLUSTERS
    if n_patients < k * 5:
        k = max(2, n_patients // 5)
        print(f"    Adjusting k to {k} (only {n_patients} patients)")

    print(f"[2/5] Running KMeans (k={k}) on {n_patients:,} patient embeddings ...")
    t0 = time.time()

    km = KMeans(n_clusters=k, random_state=42, n_init="auto", max_iter=300)
    labels = km.fit_predict(pac_embs)

    print(f"    Clustering done in {time.time()-t0:.1f}s  inertia={km.inertia_:.2f}")

    # Parse node names: "GHO-BRADESCO/ID_CD_PACIENTE_12345"
    patient_records = []
    for idx, node_name in enumerate(pac_nodes):
        s = str(node_name)
        try:
            src_db, id_part = s.split("/", 1)
            pid = int(id_part.split("ID_CD_PACIENTE_")[1])
        except Exception:
            src_db = "UNKNOWN"
            pid    = None
        patient_records.append({
            "node":     s,
            "source_db": src_db,
            "pid":       pid,
            "cluster":   int(labels[idx]),
        })

    cluster_sizes: dict[int, int] = {}
    for rec in patient_records:
        cluster_sizes[rec["cluster"]] = cluster_sizes.get(rec["cluster"], 0) + 1

    return patient_records, labels, k, cluster_sizes


# ─────────────────────────────────────────────────────────────────
# Step 3 -- DuckDB characterization per cluster
# ─────────────────────────────────────────────────────────────────

def _characterize_clusters(patient_records: list[dict], labels, k: int):
    import duckdb
    import time

    print(f"[3/5] Characterizing {k} clusters via DuckDB ...")
    t0 = time.time()

    con = duckdb.connect(str(DB_PATH))

    con.execute("""
        CREATE OR REPLACE TEMP TABLE tmp_cluster_patients (
            cluster_id  INTEGER,
            source_db   VARCHAR,
            pid         INTEGER
        )
    """)

    batch_size = 2000
    valid = [(r["cluster"], r["source_db"], r["pid"])
             for r in patient_records if r["pid"] is not None]

    for i in range(0, len(valid), batch_size):
        batch = valid[i:i + batch_size]
        vals  = ", ".join(f"({c}, '{src}', {pid})" for c, src, pid in batch)
        con.execute(f"INSERT INTO tmp_cluster_patients VALUES {vals}")

    print(f"    Inserted {len(valid):,} patient rows into temp table")

    def _exec(q):
        cur  = con.execute(q)
        cols = [d[0] for d in cur.description]
        return cols, cur.fetchall()

    cluster_data: dict[int, dict] = {i: {} for i in range(k)}

    # -- Internacao: LOS, IN_SITUACAO (with COALESCE for open admissions) --
    print("    Querying LOS and situacao ...")
    try:
        los_cols, los_rows = _exec("""
            SELECT
                cp.cluster_id,
                COUNT(DISTINCT i.ID_CD_INTERNACAO)  AS n_internacoes,
                AVG(DATEDIFF('day', i.DH_ADMISSAO_HOSP::DATE,
                    COALESCE(i.DH_FINALIZACAO, CURRENT_TIMESTAMP)::DATE)) AS avg_los,
                SUM(CASE WHEN i.IN_SITUACAO = 1 THEN 1 ELSE 0 END) AS n_situacao_1,
                SUM(CASE WHEN i.IN_SITUACAO = 2 THEN 1 ELSE 0 END) AS n_situacao_2,
                SUM(CASE WHEN i.IN_SITUACAO = 3 THEN 1 ELSE 0 END) AS n_situacao_3
            FROM agg_tb_capta_internacao_cain i
            JOIN tmp_cluster_patients cp
              ON i.ID_CD_PACIENTE = cp.pid
             AND i.source_db = cp.source_db
            GROUP BY cp.cluster_id
        """)
        for row in los_rows:
            d = dict(zip(los_cols, row))
            cid = d["cluster_id"]
            cluster_data[cid].update({
                "n_internacoes": _safe_int(d.get("n_internacoes")),
                "avg_los":       _safe_float(d.get("avg_los")),
                "n_situacao_1":  _safe_int(d.get("n_situacao_1")),
                "n_situacao_2":  _safe_int(d.get("n_situacao_2")),
                "n_situacao_3":  _safe_int(d.get("n_situacao_3")),
            })
    except Exception as e:
        print(f"    LOS/situacao skipped: {e}")

    # -- Discharge types (FL_TIPO_ALTA) --
    print("    Querying discharge types ...")
    try:
        alta_cols, alta_rows = _exec("""
            SELECT
                cp.cluster_id,
                COALESCE(a.FL_TIPO_ALTA, 'ND') AS fl_tipo_alta,
                COUNT(*) AS n_altas
            FROM agg_tb_orcamento_evo_alta_oral a
            JOIN tmp_cluster_patients cp
              ON a.ID_CD_PACIENTE = cp.pid
             AND a.source_db = cp.source_db
            GROUP BY cp.cluster_id, a.FL_TIPO_ALTA
            ORDER BY cp.cluster_id, n_altas DESC
        """)
        alta_map: dict[int, dict[str, int]] = {}
        for row in alta_rows:
            d  = dict(zip(alta_cols, row))
            c  = d["cluster_id"]
            ft = str(d["fl_tipo_alta"] or "ND")
            alta_map.setdefault(c, {})[ft] = _safe_int(d["n_altas"])
        for c, counts in alta_map.items():
            cluster_data[c]["alta_counts"] = counts
    except Exception as e:
        print(f"    discharge types skipped: {e}")

    # -- DRG groups (DS_DRG_ALTA) --
    print("    Querying DRG groups ...")
    try:
        drg_cols, drg_rows = _exec("""
            SELECT
                cp.cluster_id,
                COALESCE(d.DS_DRG_ALTA, 'ND') AS ds_drg_alta,
                COUNT(*) AS n_drg
            FROM agg_tb_capta_drg_3m_cgs_temp_cdct d
            JOIN tmp_cluster_patients cp
              ON d.ID_CD_PACIENTE = cp.pid
             AND d.source_db = cp.source_db
            GROUP BY cp.cluster_id, d.DS_DRG_ALTA
            ORDER BY cp.cluster_id, n_drg DESC
        """)
        drg_map: dict[int, list[tuple[str, int]]] = {}
        for row in drg_rows:
            d = dict(zip(drg_cols, row))
            c = d["cluster_id"]
            drg_map.setdefault(c, []).append(
                (str(d["ds_drg_alta"] or "ND"), _safe_int(d["n_drg"]))
            )
        for c, lst in drg_map.items():
            cluster_data[c]["drg_top"] = lst[:5]
    except Exception as e:
        print(f"    DRG skipped: {e}")

    # -- CIDs --
    print("    Querying CIDs ...")
    try:
        cid_cols, cid_rows = _exec("""
            SELECT
                cp.cluster_id,
                COALESCE(c.DS_DESCRICAO, 'ND') AS ds_descricao,
                COUNT(*) AS n_cid
            FROM agg_tb_capta_cid_caci c
            JOIN tmp_cluster_patients cp
              ON c.ID_CD_PACIENTE = cp.pid
             AND c.source_db = cp.source_db
            GROUP BY cp.cluster_id, c.DS_DESCRICAO
            ORDER BY cp.cluster_id, n_cid DESC
        """)
        cid_map: dict[int, list[tuple[str, int]]] = {}
        for row in cid_rows:
            d = dict(zip(cid_cols, row))
            c = d["cluster_id"]
            cid_map.setdefault(c, []).append(
                (str(d["ds_descricao"] or "ND"), _safe_int(d["n_cid"]))
            )
        for c, lst in cid_map.items():
            cluster_data[c]["cid_top"] = lst[:5]
    except Exception as e:
        print(f"    CIDs skipped: {e}")

    # -- Billing --
    print("    Querying billing ...")
    try:
        fat_cols, fat_rows = _exec("""
            SELECT
                cp.cluster_id,
                COUNT(DISTINCT f_agg.ID_CD_INTERNACAO) AS n_faturas,
                AVG(f_agg.vl_total)                AS avg_billing,
                SUM(f_agg.vl_total)                AS total_billing
            FROM (
                SELECT ID_CD_INTERNACAO, source_db, SUM(VL_TOTAL) AS vl_total
                FROM agg_tb_fatura_fatu
                GROUP BY ID_CD_INTERNACAO, source_db
            ) f_agg
            JOIN agg_tb_capta_internacao_cain f
              ON f_agg.ID_CD_INTERNACAO = f.ID_CD_INTERNACAO
             AND f_agg.source_db = f.source_db
            JOIN tmp_cluster_patients cp
              ON f.ID_CD_PACIENTE = cp.pid
             AND f.source_db = cp.source_db
            GROUP BY cp.cluster_id
        """)
        for row in fat_rows:
            d = dict(zip(fat_cols, row))
            c = d["cluster_id"]
            cluster_data[c].update({
                "avg_billing":   _safe_float(d.get("avg_billing")),
                "total_billing": _safe_float(d.get("total_billing")),
                "n_faturas":     _safe_int(d.get("n_faturas")),
            })
    except Exception as e:
        print(f"    billing skipped: {e}")

    # -- Readmission rate (30d) --
    print("    Querying readmission rates ...")
    try:
        r30_cols, r30_rows = _exec("""
            WITH ordered AS (
                SELECT
                    i.ID_CD_PACIENTE,
                    i.source_db,
                    i.DH_ADMISSAO_HOSP,
                    i.DH_FINALIZACAO,
                    LAG(i.DH_FINALIZACAO) OVER (
                        PARTITION BY i.ID_CD_PACIENTE, i.source_db
                        ORDER BY i.DH_ADMISSAO_HOSP
                    ) AS prev_discharge
                FROM agg_tb_capta_internacao_cain i
                JOIN tmp_cluster_patients cp
                  ON i.ID_CD_PACIENTE = cp.pid
                 AND i.source_db = cp.source_db
            ),
            readmit AS (
                SELECT DISTINCT ID_CD_PACIENTE, source_db
                FROM ordered
                WHERE prev_discharge IS NOT NULL
                  AND prev_discharge > '2000-01-01'
                  AND DATEDIFF('day', prev_discharge::DATE, DH_ADMISSAO_HOSP::DATE) > 0
                  AND DATEDIFF('day', prev_discharge::DATE, DH_ADMISSAO_HOSP::DATE) <= 30
            )
            SELECT
                cp.cluster_id,
                COUNT(DISTINCT r.ID_CD_PACIENTE) AS n_readmit
            FROM readmit r
            JOIN tmp_cluster_patients cp
              ON r.ID_CD_PACIENTE = cp.pid
             AND r.source_db = cp.source_db
            GROUP BY cp.cluster_id
        """)
        for row in r30_rows:
            d = dict(zip(r30_cols, row))
            cluster_data[d["cluster_id"]]["n_readmit"] = _safe_int(d.get("n_readmit"))
    except Exception as e:
        print(f"    readmission skipped: {e}")

    # -- Hospital (source_db) distribution per cluster --
    print("    Querying hospital distribution ...")
    try:
        src_cols, src_rows = _exec("""
            SELECT
                cluster_id,
                source_db,
                COUNT(*) AS n_pats
            FROM tmp_cluster_patients
            GROUP BY cluster_id, source_db
            ORDER BY cluster_id, n_pats DESC
        """)
        src_map: dict[int, list[tuple[str, int]]] = {}
        for row in src_rows:
            d = dict(zip(src_cols, row))
            c = d["cluster_id"]
            src_map.setdefault(c, []).append(
                (str(d["source_db"] or "ND"), _safe_int(d["n_pats"]))
            )
        for c, lst in src_map.items():
            cluster_data[c]["hospital_top"] = lst
    except Exception as e:
        print(f"    hospital distribution skipped: {e}")

    # -- Per-hospital breakdown within each cluster --
    print("    Querying per-hospital breakdown per cluster ...")
    try:
        hb_cols, hb_rows = _exec("""
            WITH fat_agg AS (
                SELECT ID_CD_INTERNACAO, source_db, SUM(VL_TOTAL) AS vl_total
                FROM agg_tb_fatura_fatu
                GROUP BY ID_CD_INTERNACAO, source_db
            ),
            ordered AS (
                SELECT
                    i.ID_CD_PACIENTE, i.source_db, i.DH_ADMISSAO_HOSP,
                    LAG(i.DH_FINALIZACAO) OVER (
                        PARTITION BY i.ID_CD_PACIENTE, i.source_db
                        ORDER BY i.DH_ADMISSAO_HOSP
                    ) AS prev_discharge
                FROM agg_tb_capta_internacao_cain i
                JOIN tmp_cluster_patients cp
                  ON i.ID_CD_PACIENTE = cp.pid AND i.source_db = cp.source_db
            ),
            readmit_pats AS (
                SELECT DISTINCT ID_CD_PACIENTE, source_db
                FROM ordered
                WHERE prev_discharge IS NOT NULL
                  AND DATEDIFF('day', prev_discharge::DATE, DH_ADMISSAO_HOSP::DATE) BETWEEN 1 AND 30
            ),
            alta_info AS (
                SELECT a.ID_CD_PACIENTE, a.source_db, a.FL_TIPO_ALTA
                FROM agg_tb_orcamento_evo_alta_oral a
            )
            SELECT
                cp.cluster_id,
                cp.source_db AS hospital,
                COUNT(DISTINCT cp.pid) AS n_patients,
                AVG(DATEDIFF('day', i.DH_ADMISSAO_HOSP::DATE,
                    COALESCE(i.DH_FINALIZACAO, CURRENT_TIMESTAMP)::DATE)) AS avg_los,
                AVG(COALESCE(fa.vl_total, 0)) AS avg_billing,
                COUNT(DISTINCT rp.ID_CD_PACIENTE) AS n_readmit,
                SUM(CASE WHEN ai.FL_TIPO_ALTA = 'OD' THEN 1 ELSE 0 END) AS n_od
            FROM tmp_cluster_patients cp
            JOIN agg_tb_capta_internacao_cain i
              ON cp.pid = i.ID_CD_PACIENTE AND cp.source_db = i.source_db
            LEFT JOIN fat_agg fa
              ON i.ID_CD_INTERNACAO = fa.ID_CD_INTERNACAO AND i.source_db = fa.source_db
            LEFT JOIN readmit_pats rp
              ON cp.pid = rp.ID_CD_PACIENTE AND cp.source_db = rp.source_db
            LEFT JOIN alta_info ai
              ON cp.pid = ai.ID_CD_PACIENTE AND cp.source_db = ai.source_db
            GROUP BY cp.cluster_id, cp.source_db
            ORDER BY cp.cluster_id, n_patients DESC
        """)
        hosp_breakdown: dict[int, list[dict]] = {}
        for row in hb_rows:
            d = dict(zip(hb_cols, row))
            c = d["cluster_id"]
            hosp_breakdown.setdefault(c, []).append({
                "hospital":    str(d["hospital"] or "ND"),
                "n_patients":  _safe_int(d["n_patients"]),
                "avg_los":     _safe_float(d["avg_los"]),
                "avg_billing": _safe_float(d["avg_billing"]),
                "n_readmit":   _safe_int(d["n_readmit"]),
                "n_od":        _safe_int(d["n_od"]),
            })
        for c, lst in hosp_breakdown.items():
            cluster_data[c]["hospital_breakdown"] = lst
    except Exception as e:
        print(f"    per-hospital breakdown skipped: {e}")

    con.close()
    print(f"    DuckDB characterization done in {time.time()-t0:.1f}s")
    return cluster_data


# ─────────────────────────────────────────────────────────────────
# Step 3b -- Auto-name clusters
# ─────────────────────────────────────────────────────────────────

def _auto_name_cluster(cd: dict, n_patients: int) -> str:
    """
    Generate a descriptive name for the cluster based on its dominant characteristics.
    """
    avg_los    = _safe_float(cd.get("avg_los", 0))
    avg_bill   = _safe_float(cd.get("avg_billing", 0))
    n_readmit  = _safe_int(cd.get("n_readmit", 0))
    alta_counts = cd.get("alta_counts", {})
    total_altas = sum(alta_counts.values()) if alta_counts else 0
    n_od       = _safe_int(alta_counts.get("OD", 0))
    od_rate    = (n_od / total_altas * 100) if total_altas > 0 else 0
    readm_rate = (n_readmit / n_patients * 100) if n_patients > 0 else 0
    hospital_top = cd.get("hospital_top", [])
    drg_top    = cd.get("drg_top", [])

    # Check dominant hospital (>60%)
    dominant_hospital = None
    if hospital_top:
        total_hosp = sum(cnt for _, cnt in hospital_top)
        if total_hosp > 0 and hospital_top[0][1] / total_hosp > 0.60:
            dominant_hospital = hospital_top[0][0]

    # Priority-based naming
    if od_rate > 5:
        name = "Risco Elevado de Obito"
    elif avg_los > 20 and avg_bill > 100_000:
        name = "Internacoes Complexas de Alta Gravidade"
    elif readm_rate > 15:
        name = "Alta Readmissao --- Qualidade da Alta em Questao"
    elif avg_bill < 5_000 and avg_los < 3:
        name = "Internacoes Curtas de Baixa Complexidade"
    elif dominant_hospital:
        # Use the most common DRG if available
        if drg_top and drg_top[0][0] != "ND":
            drg_label = _truncate(drg_top[0][0], 40)
            name = f"Perfil {dominant_hospital} --- {drg_label}"
        else:
            name = f"Perfil {dominant_hospital}"
    elif drg_top and drg_top[0][0] != "ND":
        name = _truncate(drg_top[0][0], 60)
    else:
        if avg_los > 15:
            name = "Permanencia Prolongada"
        elif avg_bill > 50_000:
            name = "Faturamento Elevado"
        else:
            name = "Perfil Misto"

    return name


def _generate_narrative(cd: dict, n_patients: int, cluster_name: str,
                         global_avg_los: float, global_avg_bill: float,
                         global_readm_rate: float, global_od_rate: float) -> str:
    """
    Generate a 3-4 sentence narrative explaining the cluster in business terms.
    """
    avg_los     = _safe_float(cd.get("avg_los", 0))
    avg_bill    = _safe_float(cd.get("avg_billing", 0))
    n_readmit   = _safe_int(cd.get("n_readmit", 0))
    n_intr      = _safe_int(cd.get("n_internacoes", 0))
    alta_counts = cd.get("alta_counts", {})
    total_altas = sum(alta_counts.values()) if alta_counts else 0
    n_od        = _safe_int(alta_counts.get("OD", 0))
    od_rate     = (n_od / total_altas * 100) if total_altas > 0 else 0
    readm_rate  = (n_readmit / n_patients * 100) if n_patients > 0 else 0
    cid_top     = cd.get("cid_top", [])
    hospital_top = cd.get("hospital_top", [])

    parts = []

    # Who are these patients
    if hospital_top:
        hosp_str = ", ".join(h for h, _ in hospital_top[:3])
        parts.append(
            f"Este cluster agrupa {n_patients} pacientes, predominantemente dos sistemas {hosp_str}, "
            f"totalizando {n_intr} internacoes."
        )
    else:
        parts.append(f"Este cluster agrupa {n_patients} pacientes com {n_intr} internacoes.")

    # What happens to them
    metrics = []
    if avg_los > 0:
        los_cmp = f"{avg_los:.1f} dias"
        if global_avg_los > 0:
            los_cmp += f" ({avg_los/global_avg_los:.1f}x a media global de {global_avg_los:.1f}d)"
        metrics.append(f"LOS medio de {los_cmp}")
    if avg_bill > 0:
        bill_cmp = _brl_plain(avg_bill)
        if global_avg_bill > 0:
            bill_cmp += f" ({avg_bill/global_avg_bill:.1f}x a media global)"
        metrics.append(f"faturamento medio de {bill_cmp}")
    if metrics:
        parts.append("Caracterizam-se por " + " e ".join(metrics) + ".")

    # Why they group together / risks
    risks = []
    if od_rate > 5:
        risks.append(f"taxa de obito de {od_rate:.1f}% ({n_od} casos)")
    if readm_rate > 15:
        risks.append(f"readmissao de {readm_rate:.1f}%")
    if cid_top:
        top_cid = cid_top[0][0]
        risks.append(f"CID predominante: {top_cid}")

    if risks:
        parts.append("Pontos de atencao: " + "; ".join(risks) + ".")

    # What action to take
    if od_rate > 5:
        parts.append("Recomenda-se revisao dos protocolos clinicos e analise dos casos de obito para identificar oportunidades de intervencao precoce.")
    elif readm_rate > 15:
        parts.append("Recomenda-se revisao dos criterios de alta e implementacao de acompanhamento pos-alta para reduzir readmissoes.")
    elif avg_bill > 0 and global_avg_bill > 0 and avg_bill / global_avg_bill > 2:
        parts.append("Recomenda-se auditoria dos itens faturados para validar a proporcionalidade dos custos.")
    elif avg_los > 0 and global_avg_los > 0 and avg_los / global_avg_los > 2:
        parts.append("Recomenda-se avaliacao da eficiencia operacional para reduzir o tempo de permanencia.")
    else:
        parts.append("Cluster dentro dos parametros esperados; manter monitoramento de rotina.")

    return " ".join(parts)


def _criticality_score(cd: dict, n_patients: int) -> float:
    """Higher score = more critical. Used for sorting clusters."""
    alta_counts = cd.get("alta_counts", {})
    total_altas = sum(alta_counts.values()) if alta_counts else 0
    n_od        = _safe_int(alta_counts.get("OD", 0))
    od_rate     = (n_od / total_altas) if total_altas > 0 else 0
    n_readmit   = _safe_int(cd.get("n_readmit", 0))
    readm_rate  = (n_readmit / n_patients) if n_patients > 0 else 0
    avg_los     = _safe_float(cd.get("avg_los", 0))
    avg_bill    = _safe_float(cd.get("avg_billing", 0))

    score = od_rate * 100 + readm_rate * 50 + (avg_los / 100) + (avg_bill / 100_000)
    return score


def _traffic_light(cd: dict, n_patients: int) -> str:
    """Return 'red', 'yellow', or 'green' for traffic-light coloring."""
    alta_counts = cd.get("alta_counts", {})
    total_altas = sum(alta_counts.values()) if alta_counts else 0
    n_od        = _safe_int(alta_counts.get("OD", 0))
    od_rate     = (n_od / total_altas * 100) if total_altas > 0 else 0
    n_readmit   = _safe_int(cd.get("n_readmit", 0))
    readm_rate  = (n_readmit / n_patients * 100) if n_patients > 0 else 0
    avg_los     = _safe_float(cd.get("avg_los", 0))
    avg_bill    = _safe_float(cd.get("avg_billing", 0))

    if od_rate > 5 or readm_rate > 15 or avg_los > 20 or avg_bill > 100_000:
        return "red"
    elif od_rate > 2 or readm_rate > 10 or avg_los > 10 or avg_bill > 50_000:
        return "yellow"
    return "green"


# ─────────────────────────────────────────────────────────────────
# Step 4 -- Generate LaTeX
# ─────────────────────────────────────────────────────────────────

def _generate_latex(
    patient_records: list[dict],
    k: int,
    cluster_sizes: dict[int, int],
    cluster_data: dict[int, dict],
) -> str:

    total_patients = len(patient_records)

    # Compute global baselines
    total_internacoes = sum(
        _safe_int(cluster_data.get(c, {}).get("n_internacoes", 0)) for c in range(k)
    )
    total_billing = sum(
        _safe_float(cluster_data.get(c, {}).get("total_billing", 0)) for c in range(k)
    )
    los_vals = [_safe_float(cluster_data.get(c, {}).get("avg_los", 0)) for c in range(k) if _safe_float(cluster_data.get(c, {}).get("avg_los", 0)) > 0]
    global_avg_los = sum(los_vals) / len(los_vals) if los_vals else 0.0

    bill_vals = [_safe_float(cluster_data.get(c, {}).get("avg_billing", 0)) for c in range(k) if _safe_float(cluster_data.get(c, {}).get("avg_billing", 0)) > 0]
    global_avg_bill = sum(bill_vals) / len(bill_vals) if bill_vals else 0.0

    total_readmit = sum(_safe_int(cluster_data.get(c, {}).get("n_readmit", 0)) for c in range(k))
    global_readm_rate = (total_readmit / total_patients * 100) if total_patients > 0 else 0

    total_od = sum(_safe_int(cluster_data.get(c, {}).get("alta_counts", {}).get("OD", 0)) for c in range(k))
    total_altas_all = sum(sum(cluster_data.get(c, {}).get("alta_counts", {}).values()) for c in range(k))
    global_od_rate = (total_od / total_altas_all * 100) if total_altas_all > 0 else 0

    # Sort clusters by criticality (highest death/readmission first)
    sorted_clusters = sorted(
        range(k),
        key=lambda c: -_criticality_score(cluster_data.get(c, {}), cluster_sizes.get(c, 0))
    )

    # Auto-name clusters
    cluster_names: dict[int, str] = {}
    for c in range(k):
        cluster_names[c] = _auto_name_cluster(cluster_data.get(c, {}), cluster_sizes.get(c, 0))

    L: list[str] = []

    # -- Preamble --
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

\definecolor{jcubeblue}{RGB}{0,74,134}
\definecolor{jcubegray}{RGB}{80,80,80}
\definecolor{clustergreen}{RGB}{0,120,60}
\definecolor{clusterorange}{RGB}{200,90,0}
\definecolor{clusterred}{RGB}{180,20,20}
\definecolor{lightgray}{RGB}{248,248,248}
\definecolor{darkblue}{RGB}{0,50,100}
\definecolor{lightblue}{RGB}{220,235,250}
\definecolor{alertyellow}{RGB}{255,243,200}
\definecolor{alertred}{RGB}{255,220,220}
\definecolor{trafficred}{RGB}{220,50,50}
\definecolor{trafficyellow}{RGB}{200,170,0}
\definecolor{trafficgreen}{RGB}{0,150,60}

\tcbuselibrary{skins,breakable}

\newtcolorbox{clustercard}[2][]{%
  enhanced,
  colback=lightgray,
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
\fancyhead[L]{\textcolor{jcubeblue}{\textbf{JCUBE Digital Twin}} \textcolor{jcubegray}{\small | Relatorio de Clusterizacao de Pacientes --- V5}}
\fancyhead[R]{\textcolor{jcubegray}{\small 23/03/2026}}
\fancyfoot[C]{\textcolor{jcubegray}{\thepage}}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0.4pt}

\titleformat{\section}{\large\bfseries\color{jcubeblue}}{\thesection}{1em}{}[\titlerule]
\titleformat{\subsection}{\normalsize\bfseries\color{darkblue}}{\thesubsection}{1em}{}

\hypersetup{colorlinks=true,linkcolor=jcubeblue,pdftitle={JCUBE V5 Clusterizacao de Pacientes}}

\begin{document}
\setlength{\parindent}{0pt}
\setlength{\parskip}{3pt}
""")

    # -- Title page --
    n_red = sum(1 for c in range(k) if _traffic_light(cluster_data.get(c, {}), cluster_sizes.get(c, 0)) == "red")
    n_yellow = sum(1 for c in range(k) if _traffic_light(cluster_data.get(c, {}), cluster_sizes.get(c, 0)) == "yellow")
    n_green = k - n_red - n_yellow

    L.append(r"""\begin{titlepage}
\begin{center}
\vspace*{1.5cm}
{\Huge\bfseries\textcolor{jcubeblue}{JCUBE}}\\[0.2cm]
{\large\textcolor{jcubegray}{Digital Twin Analytics Platform --- Modelo V5}}\\[1.2cm]
\begin{tcolorbox}[colback=jcubeblue,colframe=jcubeblue,coltext=white,width=0.92\textwidth,halign=center]
{\LARGE\bfseries Relatorio de Clusterizacao de Pacientes}\\[0.3cm]
{\large Analise via Embeddings do Gemeo Digital --- Graph-JEPA V5 (35,2M nos $\times$ 64 dim)}\\[0.2cm]
{\normalsize K-Means sobre nos PACIENTE --- Epoch 1}
\end{tcolorbox}
\vspace{0.8cm}
""")

    L.append(r"{\Large Gerado em: \textbf{23 de marco de 2026}}\\[0.6cm]")

    L.append(
        r"""\begin{tabular}{ccc}
\begin{tcolorbox}[colback=jcubeblue!10,colframe=jcubeblue,width=3.8cm,halign=center,left=3pt,right=3pt]
{\LARGE\bfseries\textcolor{jcubeblue}{""" + str(total_patients) + r"""}}\\[2pt]
{\small\textbf{Pacientes Unicos}}
\end{tcolorbox}
&
\begin{tcolorbox}[colback=clustergreen!10,colframe=clustergreen,width=3.8cm,halign=center,left=3pt,right=3pt]
{\LARGE\bfseries\textcolor{clustergreen}{""" + str(k) + r"""}}\\[2pt]
{\small\textbf{Clusters}}
\end{tcolorbox}
&
\begin{tcolorbox}[colback=clusterred!10,colframe=clusterred,width=3.8cm,halign=center,left=3pt,right=3pt]
{\LARGE\bfseries\textcolor{clusterred}{""" + str(n_red) + r"""}}\\[2pt]
{\small\textbf{Clusters Criticos}}
\end{tcolorbox}
\end{tabular}

\vfill
{\small\textcolor{jcubegray}{
Metodologia: K-Means sobre embeddings JEPA V5 de nos PACIENTE do grafo de conhecimento JCUBE\\
Clusters auto-nomeados por caracteristicas dominantes, ordenados por criticidade\\
Semaforo: vermelho (obito $>$5\% ou readmissao $>$15\% ou LOS $>$20d ou fat. $>$R\$100k),
amarelo (metricas intermediarias), verde (dentro dos parametros)
}}
\end{center}
\end{titlepage}
""")

    L.append(r"\tableofcontents\clearpage")

    # -- Executive Summary --
    L.append(r"\section{Sumario Executivo}")
    L.append(r"""
Este relatorio apresenta a \textbf{clusterizacao de pacientes} do \textit{Digital Twin} JCUBE,
utilizando embeddings do modelo Graph-JEPA V5 com \textbf{35,2M nos} $\times$ \textbf{64 dimensoes}.

Os clusters sao \textbf{auto-nomeados} com base em suas caracteristicas dominantes,
\textbf{ordenados por criticidade} (maior taxa de obito e readmissao primeiro),
e acompanhados de \textbf{narrativas interpretativas} e \textbf{detalhamento por hospital}.
""")

    # Global stats
    L.append(r"""\subsection{Metricas Globais}
\begin{center}
\begin{tabular}{lr}
\toprule
\textbf{Metrica} & \textbf{Valor} \\
\midrule
""")
    L.append(r"Pacientes unicos no modelo & " + f"{total_patients:,}" + r" \\" + "\n")
    L.append(r"Clusters gerados (k-means) & " + str(k) + r" \\" + "\n")
    L.append(r"Total internacoes associadas & " + f"{total_internacoes:,}" + r" \\" + "\n")
    L.append(r"LOS medio global & " + f"{global_avg_los:.1f}" + r" dias \\" + "\n")
    if total_billing > 0:
        L.append(r"Faturamento total & " + _brl(total_billing) + r" \\" + "\n")
    if global_avg_bill > 0:
        L.append(r"Faturamento medio por internacao & " + _brl(global_avg_bill) + r" \\" + "\n")
    L.append(r"Taxa de readmissao global (30d) & " + f"{global_readm_rate:.1f}\\%" + r" \\" + "\n")
    L.append(r"Taxa de obito global & " + f"{global_od_rate:.1f}\\%" + r" \\" + "\n")
    L.append(r"Clusters criticos (vermelho) & \textcolor{clusterred}{\textbf{" + str(n_red) + r"}} \\" + "\n")
    L.append(r"Clusters atencao (amarelo) & \textcolor{clusterorange}{\textbf{" + str(n_yellow) + r"}} \\" + "\n")
    L.append(r"Clusters normais (verde) & \textcolor{clustergreen}{\textbf{" + str(n_green) + r"}} \\" + "\n")
    L.append(r"""\bottomrule
\end{tabular}
\end{center}
""")

    # -- Executive summary table with traffic-light colors (sorted by criticality) --
    L.append(r"\subsection{Ranking de Clusters por Criticidade}")
    L.append(r"""\begin{center}
{\scriptsize
\begin{longtable}{>{\scriptsize}c>{\scriptsize}p{4.5cm}>{\scriptsize}r>{\scriptsize}r>{\scriptsize}r>{\scriptsize}r>{\scriptsize}r>{\scriptsize}r>{\scriptsize}c}
\toprule
\textbf{\#} & \textbf{Nome do Cluster} & \textbf{Pac.} & \textbf{Intr.} & \textbf{LOS med.} & \textbf{Fat. med.} &
\textbf{Readm.} & \textbf{Obitos} & \textbf{Status} \\
\midrule
\endhead
\bottomrule
\endfoot
""")
    for rank, c in enumerate(sorted_clusters, 1):
        cd        = cluster_data.get(c, {})
        n_pac     = cluster_sizes.get(c, 0)
        n_intr    = _safe_int(cd.get("n_internacoes", 0))
        avg_los   = _safe_float(cd.get("avg_los", 0))
        avg_bill  = _safe_float(cd.get("avg_billing", 0))
        n_readmit = _safe_int(cd.get("n_readmit", 0))
        alta_counts = cd.get("alta_counts", {})
        n_od      = _safe_int(alta_counts.get("OD", 0))
        total_altas = sum(alta_counts.values()) if alta_counts else 0
        name      = cluster_names.get(c, "Cluster " + str(c + 1))
        light     = _traffic_light(cd, n_pac)

        los_str   = f"{avg_los:.1f}d" if avg_los > 0 else "---"
        bill_str  = _brl(avg_bill) if avg_bill > 0 else "---"
        readm_str = str(n_readmit)
        od_str    = str(n_od)

        # Compare with global
        los_cmp = ""
        if avg_los > 0 and global_avg_los > 0:
            los_cmp = f" ({avg_los/global_avg_los:.1f}x)"
        bill_cmp = ""
        if avg_bill > 0 and global_avg_bill > 0:
            bill_cmp = f" ({avg_bill/global_avg_bill:.1f}x)"

        if light == "red":
            light_str = r"\textcolor{trafficred}{$\blacksquare$}"
        elif light == "yellow":
            light_str = r"\textcolor{trafficyellow}{$\blacksquare$}"
        else:
            light_str = r"\textcolor{trafficgreen}{$\blacksquare$}"

        L.append(
            str(rank) + " & " + _escape_latex(name) + " & " +
            str(n_pac) + " & " + str(n_intr) + " & " +
            los_str + los_cmp + " & " + bill_str + bill_cmp + " & " +
            readm_str + " & " + od_str + " & " + light_str + r" \\" + "\n"
        )
    L.append(r"\end{longtable}}" + "\n")
    L.append(r"""
{\footnotesize
\textcolor{trafficred}{$\blacksquare$} Critico: obito $>$5\%, readmissao $>$15\%, LOS $>$20d ou fat. $>$R\$100k \quad
\textcolor{trafficyellow}{$\blacksquare$} Atencao: metricas intermediarias \quad
\textcolor{trafficgreen}{$\blacksquare$} Normal: dentro dos parametros\\
Os valores entre parenteses indicam o multiplo em relacao a media global.
}
""")
    L.append(r"\end{center}")
    L.append(r"\clearpage")

    # -- Per-cluster detailed pages --
    L.append(r"\section{Fichas Detalhadas por Cluster}")

    for rank, c in enumerate(sorted_clusters, 1):
        cd        = cluster_data.get(c, {})
        n_pac     = cluster_sizes.get(c, 0)
        n_intr    = _safe_int(cd.get("n_internacoes", 0))
        avg_los   = _safe_float(cd.get("avg_los", 0))
        avg_bill  = _safe_float(cd.get("avg_billing", 0))
        total_bill = _safe_float(cd.get("total_billing", 0))
        n_readmit = _safe_int(cd.get("n_readmit", 0))
        n_situacao_1 = _safe_int(cd.get("n_situacao_1", 0))
        n_situacao_2 = _safe_int(cd.get("n_situacao_2", 0))
        n_situacao_3 = _safe_int(cd.get("n_situacao_3", 0))
        alta_counts  = cd.get("alta_counts", {})
        drg_top      = cd.get("drg_top", [])
        cid_top      = cd.get("cid_top", [])
        hospital_top = cd.get("hospital_top", [])
        hospital_bkdn = cd.get("hospital_breakdown", [])

        n_od  = _safe_int(alta_counts.get("OD", 0))
        total_altas = sum(alta_counts.values()) if alta_counts else 0
        od_rate = (n_od / total_altas * 100) if total_altas > 0 else 0
        readm_rate = (n_readmit / n_pac * 100) if n_pac > 0 else 0

        name  = cluster_names.get(c, "Cluster " + str(c + 1))
        light = _traffic_light(cd, n_pac)

        if light == "red":
            frame_color = "clusterred"
        elif light == "yellow":
            frame_color = "clusterorange"
        else:
            frame_color = "jcubeblue"

        light_icon = (
            r"\textcolor{trafficred}{$\blacksquare$ CRITICO}" if light == "red"
            else r"\textcolor{trafficyellow}{$\blacksquare$ ATENCAO}" if light == "yellow"
            else r"\textcolor{trafficgreen}{$\blacksquare$ NORMAL}"
        )

        card_title = f"\\#{rank} --- {_escape_latex(name)} --- {n_pac} pacientes {light_icon}"

        L.append(r"\begin{clustercard}[{" + card_title + r"}]{" + frame_color + r"}")

        # -- Header row --
        los_str  = f"{avg_los:.1f} dias" if avg_los > 0 else "---"
        bill_str = _brl(avg_bill) if avg_bill > 0 else "---"
        L.append(
            r"\textbf{Pacientes:} " + str(n_pac) +
            r" $\mid$ \textbf{Internacoes:} " + str(n_intr) +
            r" $\mid$ \textbf{LOS medio:} " + los_str +
            r" $\mid$ \textbf{Fat. medio/intr.:} " + bill_str +
            r"\\"
        )

        readm_pct = f"{readm_rate:.1f}\\%" if n_pac > 0 else "---"
        od_pct    = f"{od_rate:.1f}\\%" if total_altas > 0 else "---"
        L.append(
            r"\textbf{Readmissao 30d:} " + str(n_readmit) + " pacientes (" + readm_pct + r")" +
            r" $\mid$ \textbf{Obitos (OD):} " + str(n_od) + " (" + od_pct + r")" +
            r"\\"
        )

        # -- Narrative (NEW in V5) --
        narrative = _generate_narrative(
            cd, n_pac, name,
            global_avg_los, global_avg_bill,
            global_readm_rate, global_od_rate
        )
        L.append(
            r"\begin{tcolorbox}[colback=lightblue,colframe=jcubeblue,"
            r"title={\textbf{Interpretacao do Cluster}},fonttitle=\bfseries\small,"
            r"left=4pt,right=4pt,top=3pt,bottom=3pt]"
        )
        L.append(r"{\small " + _escape_latex(narrative) + r"}")
        L.append(r"\end{tcolorbox}")

        # -- Comparison against global baseline (NEW in V5) --
        L.append(r"\vspace{4pt}{\footnotesize\textbf{Comparacao com Baseline Global:}}")
        L.append(
            r"\begin{center}\begin{tabular}{l r r r}"
            r"\toprule"
            r"\textbf{Metrica} & \textbf{Cluster} & \textbf{Media Global} & \textbf{Razao} \\"
            r"\midrule"
        )
        if avg_los > 0:
            ratio = f"{avg_los/global_avg_los:.1f}x" if global_avg_los > 0 else "---"
            color = "clusterred" if global_avg_los > 0 and avg_los / global_avg_los > 2 else ("clusterorange" if global_avg_los > 0 and avg_los / global_avg_los > 1.5 else "jcubeblue")
            L.append(f"LOS medio & {avg_los:.1f}d & {global_avg_los:.1f}d & \\textcolor{{{color}}}{{\\textbf{{{ratio}}}}} \\\\")
        if avg_bill > 0:
            ratio = f"{avg_bill/global_avg_bill:.1f}x" if global_avg_bill > 0 else "---"
            color = "clusterred" if global_avg_bill > 0 and avg_bill / global_avg_bill > 2 else ("clusterorange" if global_avg_bill > 0 and avg_bill / global_avg_bill > 1.5 else "jcubeblue")
            L.append(f"Faturamento medio & {_brl(avg_bill)} & {_brl(global_avg_bill)} & \\textcolor{{{color}}}{{\\textbf{{{ratio}}}}} \\\\")
        if n_pac > 0:
            ratio = f"{readm_rate/global_readm_rate:.1f}x" if global_readm_rate > 0 else "---"
            color = "clusterred" if global_readm_rate > 0 and readm_rate / global_readm_rate > 2 else "jcubeblue"
            L.append(f"Readmissao 30d & {readm_rate:.1f}\\% & {global_readm_rate:.1f}\\% & \\textcolor{{{color}}}{{\\textbf{{{ratio}}}}} \\\\")
        if total_altas > 0:
            ratio = f"{od_rate/global_od_rate:.1f}x" if global_od_rate > 0 else "---"
            color = "clusterred" if global_od_rate > 0 and od_rate / global_od_rate > 2 else "jcubeblue"
            L.append(f"Taxa de obito & {od_rate:.1f}\\% & {global_od_rate:.1f}\\% & \\textcolor{{{color}}}{{\\textbf{{{ratio}}}}} \\\\")
        L.append(r"\bottomrule\end{tabular}\end{center}")

        # -- Per-hospital breakdown (NEW in V5) --
        if hospital_bkdn:
            L.append(r"\vspace{4pt}{\footnotesize\textbf{Detalhamento por Hospital:}\\[2pt]}")
            L.append(r"\begin{center}")
            L.append(r"{\scriptsize\begin{tabular}{lrrrrr}")
            L.append(r"\toprule\textbf{Hospital} & \textbf{Pac.} & \textbf{LOS med.} & \textbf{Fat. med.} & \textbf{Readm.} & \textbf{Obitos} \\\midrule")
            for hb in hospital_bkdn[:8]:
                h_readm_pct = f" ({100*hb['n_readmit']/hb['n_patients']:.0f}\\%)" if hb["n_patients"] > 0 else ""
                L.append(
                    _escape_latex(hb["hospital"]) + " & " +
                    str(hb["n_patients"]) + " & " +
                    (f"{hb['avg_los']:.1f}d" if hb["avg_los"] > 0 else "---") + " & " +
                    (_brl(hb["avg_billing"]) if hb["avg_billing"] > 0 else "---") + " & " +
                    str(hb["n_readmit"]) + h_readm_pct + " & " +
                    str(hb["n_od"]) + r" \\"
                )
            L.append(r"\bottomrule\end{tabular}}")
            L.append(r"\end{center}")

        # -- Two-column: discharge types + DRG --
        L.append(r"\vspace{4pt}")
        L.append(r"\begin{minipage}[t]{0.48\textwidth}")
        L.append(r"{\footnotesize\textbf{Tipos de Alta:}\\[2pt]}")
        if alta_counts:
            L.append(r"\begin{tabular}{llr}")
            L.append(r"\toprule\textbf{Cod.} & \textbf{Descricao} & \textbf{N} \\\midrule")
            alta_labels = {"AA": "Administrativa", "HO": "Hospitalar/Clinica", "OD": "Obito", "AC": "Alta Complexa"}
            for code, cnt in sorted(alta_counts.items(), key=lambda x: -x[1]):
                label = alta_labels.get(code, _escape_latex(str(code)))
                if code == "OD":
                    L.append(
                        _escape_latex(code) + " & " +
                        _escape_latex(label) + " & " +
                        r"\textcolor{clusterred}{\textbf{" + str(cnt) + r"}} \\"
                    )
                else:
                    L.append(_escape_latex(code) + " & " + _escape_latex(label) + " & " + str(cnt) + r" \\")
            L.append(r"\bottomrule\end{tabular}")
        else:
            L.append(r"Sem dados de alta disponivel.")
        L.append(r"\end{minipage}\hfill")

        L.append(r"\begin{minipage}[t]{0.48\textwidth}")
        if drg_top:
            L.append(r"{\footnotesize\textbf{DRG mais frequentes:}\\[2pt]}")
            L.append(r"\begin{tabular}{lr}")
            L.append(r"\toprule\textbf{DRG} & \textbf{N} \\\midrule")
            for drg_name, drg_cnt in drg_top[:5]:
                L.append(_escape_latex(_truncate(drg_name, 50)) + " & " + str(drg_cnt) + r" \\")
            L.append(r"\bottomrule\end{tabular}")
        L.append(r"\end{minipage}")

        # -- CIDs top 5 --
        if cid_top:
            L.append(r"\vspace{4pt}{\footnotesize\textbf{CIDs mais frequentes:}\\[2pt]}")
            L.append(r"\begin{tabular}{lr}")
            L.append(r"\toprule\textbf{CID / Descricao} & \textbf{N} \\\midrule")
            for cid_name, cid_cnt in cid_top[:5]:
                L.append(_escape_latex(_truncate(cid_name, 80)) + " & " + str(cid_cnt) + r" \\")
            L.append(r"\bottomrule\end{tabular}")

        L.append(r"\end{clustercard}")
        L.append(r"\newpage")

    # -- Appendix: methodology --
    L.append(r"\section*{Apendice: Metodologia de Clusterizacao}")
    L.append(r"\addcontentsline{toc}{section}{Apendice: Metodologia}")
    L.append(r"""
\subsection*{1. Modelo Graph-JEPA V5}
O modelo \textit{Graph-JEPA V5} foi treinado sobre o grafo de conhecimento JCUBE com \textbf{35,2M nos}
e \textbf{64 dimensoes} de embedding. O arquivo utilizado e \texttt{node\_emb\_epoch\_1.pt} (8,4 GB).

\subsection*{2. Extracao de Nos PACIENTE}
Os nos cujo ID contem \texttt{\_PACIENTE\_} sao filtrados do vocabulario total do grafo.
O formato dos IDs e: \texttt{SOURCE\_DB/ID\_CD\_PACIENTE\_<int>}.

\subsection*{3. Clusterizacao K-Means}
\begin{enumerate}[nosep]
  \item Os embeddings dos nos PACIENTE sao extraidos (dimensao 64).
  \item O algoritmo \textbf{K-Means} (scikit-learn, \texttt{n\_init=auto}) e executado com k = """ + str(k) + r""".
  \item Cada paciente recebe um label de cluster (0 a k-1).
\end{enumerate}

\subsection*{4. Auto-nomeacao de Clusters (V5)}
Cada cluster recebe um nome automatico baseado em:
\begin{itemize}[nosep]
  \item Taxa de obito $>$ 5\% $\rightarrow$ ``Risco Elevado de Obito''
  \item LOS $>$ 20d e fat. $>$ R\$100k $\rightarrow$ ``Internacoes Complexas de Alta Gravidade''
  \item Readmissao $>$ 15\% $\rightarrow$ ``Alta Readmissao --- Qualidade da Alta em Questao''
  \item Fat. $<$ R\$5k e LOS $<$ 3d $\rightarrow$ ``Internacoes Curtas de Baixa Complexidade''
  \item Hospital dominante ($>$60\%) $\rightarrow$ ``Perfil [HOSPITAL]''
  \item Caso contrario: DRG mais frequente ou classificacao generica
\end{itemize}

\subsection*{5. Ordenacao por Criticidade}
Os clusters sao ordenados do mais critico para o menos critico, usando a formula:
$$score = taxa\_obito \times 100 + taxa\_readmissao \times 50 + \frac{LOS}{100} + \frac{faturamento}{100.000}$$

\subsection*{6. Semaforo de Criticidade}
\begin{itemize}[nosep]
  \item \textcolor{trafficred}{$\blacksquare$} \textbf{Vermelho}: obito $>$5\%, readmissao $>$15\%, LOS $>$20d, ou faturamento $>$R\$100k
  \item \textcolor{trafficyellow}{$\blacksquare$} \textbf{Amarelo}: obito $>$2\%, readmissao $>$10\%, LOS $>$10d, ou faturamento $>$R\$50k
  \item \textcolor{trafficgreen}{$\blacksquare$} \textbf{Verde}: dentro dos parametros esperados
\end{itemize}
""")

    L.append(r"\end{document}")
    return "\n".join(L)


# ─────────────────────────────────────────────────────────────────
# Step 5 -- Compile LaTeX -> PDF
# ─────────────────────────────────────────────────────────────────

def _compile_latex(latex_content: str, output_pdf: str):
    import subprocess
    from pathlib import Path

    print("[5/5] Compiling PDF ...")
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
    timeout=7200,   # 2 hours max
)
def generate_cluster_report():
    import time
    import os

    t_start = time.time()
    print("=" * 70)
    print("JCUBE V5 Patient Clustering Report Generator (Modal)")
    print(f"Weights  : {WEIGHTS_PATH}")
    print(f"Parquet  : {GRAPH_PARQUET}")
    print(f"DB       : {DB_PATH}")
    print(f"Output   : {OUTPUT_PDF}")
    print(f"K        : {K_CLUSTERS}")
    print("=" * 70)

    for p in [GRAPH_PARQUET, WEIGHTS_PATH, DB_PATH]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}")

    # 1. Load PACIENTE embeddings
    pac_nodes, pac_embs = _load_paciente_embeddings()

    # 2. Cluster
    patient_records, labels, k, cluster_sizes = _cluster_patients(pac_nodes, pac_embs)
    print(f"    Cluster sizes: { {c: cluster_sizes[c] for c in sorted(cluster_sizes)} }")

    # 3. Characterize via DuckDB (includes per-hospital breakdown)
    cluster_data = _characterize_clusters(patient_records, labels, k)

    # 4. Generate LaTeX
    print("[4/5] Generating LaTeX document ...")
    latex = _generate_latex(
        patient_records, k, cluster_sizes, cluster_data,
    )

    # 5. Compile to PDF
    _compile_latex(latex, OUTPUT_PDF)

    # Commit so changes persist in the volume
    data_vol.commit()

    elapsed = time.time() - t_start
    print(f"\nFinished in {elapsed:.1f}s")
    print(f"Report saved to Modal volume jcube-data at: {OUTPUT_PDF}")
    print("Download with:")
    print(f"  modal volume get jcube-data reports/cluster_report_v5_2026_03.pdf ./cluster_report_v5_2026_03.pdf")
    return OUTPUT_PDF


@app.local_entrypoint()
def main():
    generate_cluster_report.remote()
# epoch2-1774305118
