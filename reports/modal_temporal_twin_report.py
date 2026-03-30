#!/usr/bin/env python3
"""
Modal script: JCUBE V6.2 Temporal Digital Twin Report
Showcases what a TEMPORAL digital twin can do that a static one cannot.
Uses V6.2 epoch 5 embeddings (128-dim, 35.2M nodes, trained with TGN temporal
graph network on 4x H100).

Sections:
  1. Velocidade de Evolucao Clinica (trajectory speed by discharge type)
  2. Previsao de Destino por Analogia Geometrica (KNN prognosis)
  3. Deteccao de Trajetoria Anomala (deviation from CID mean trajectory)
  4. Simulacao Contrafactual Temporal (hospital swap counterfactuals)
  5. Clustering Temporal de Perfis de Evolucao (K-means on trajectory vectors)

Output: LaTeX PDF at /data/reports/temporal_twin_v6.2_2026_03.pdf
All text pt-BR, UTF-8, professional, one section per page.

Usage:
    modal run reports/modal_temporal_twin_report.py
    modal run --detach reports/modal_temporal_twin_report.py
"""
from __future__ import annotations

import modal

# ─────────────────────────────────────────────────────────────────
# Modal App + Volumes
# ─────────────────────────────────────────────────────────────────

app = modal.App("jcube-temporal-twin-report")

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
WEIGHTS_PATH  = "/cache/tkg-v6.2/node_embeddings.pt"
DB_PATH       = "/data/aggregated_fixed_union.db"
OUTPUT_DIR    = "/data/reports"
OUTPUT_PDF    = f"{OUTPUT_DIR}/temporal_twin_v6.2_2026_03.pdf"

REPORT_DATE_STR = "2026-03-28"

MAJOR_HOSPITALS = [
    "GHO-BRADESCO", "GHO-PETROBRAS", "PASA", "GOHOSP-CNU", "GHO-CASSI",
]

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


def _fmt_date(d) -> str:
    if d is None:
        return "---"
    try:
        if isinstance(d, str):
            return d[:10]
        return d.strftime("%d/%m/%Y")
    except Exception:
        return str(d)[:10]


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


def _categorize_alta(tipo_alta_str) -> str:
    """Categorize discharge type string into normalized categories."""
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

    print("[1/7] Loading node vocabulary from graph parquet ...")
    t0 = time.time()
    table = pq.read_table(GRAPH_PARQUET, columns=["subject_id", "object_id"])
    subj  = table.column("subject_id")
    obj   = table.column("object_id")
    all_nodes    = pa.chunked_array(subj.chunks + obj.chunks)
    unique_nodes = pc.unique(all_nodes).to_numpy(zero_copy_only=False).astype(object)
    del table, subj, obj, all_nodes
    n_nodes = len(unique_nodes)
    print(f"    {n_nodes:,} unique nodes in {time.time()-t0:.1f}s")

    print("[1/7] Loading V6.2 embedding weights ...")
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
# Step 2 — Velocidade de Evolucao Clinica
# ─────────────────────────────────────────────────────────────────

def _trajectory_speed(unique_nodes, embeddings, node_to_idx):
    """Compute trajectory speed (||delta||) per patient with 2+ admissions.

    Returns:
        speed_by_alta: dict mapping discharge category to list of (speed, patient_key, iid, src)
        top_fastest: top 20 patients with highest trajectory speed
        top_slowest: top 20 patients with lowest (non-zero) trajectory speed
        stats_by_alta: dict mapping category to (mean, std, n)
    """
    import time
    import numpy as np
    import duckdb

    print("[2/7] Computing trajectory speed (Velocidade de Evolucao Clinica) ...")
    t0 = time.time()

    con = duckdb.connect(str(DB_PATH))

    rows = con.execute("""
        WITH patient_admissions AS (
            SELECT
                i.ID_CD_INTERNACAO,
                i.ID_CD_PACIENTE,
                i.source_db,
                i.DH_ADMISSAO_HOSP,
                i.IN_SITUACAO,
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
            di.tipo_alta,
            pa.IN_SITUACAO
        FROM patient_admissions pa
        LEFT JOIN discharge_info di
            ON pa.ID_CD_INTERNACAO = di.ID_CD_INTERNACAO
            AND pa.source_db = di.source_db
        WHERE pa.total_adm >= 2
        ORDER BY pa.source_db, pa.ID_CD_PACIENTE, pa.adm_seq
    """).fetchall()

    con.close()
    print(f"    Found {len(rows):,} admissions from patients with 2+ visits")

    # Build patient admission sequences
    patient_seqs: dict[tuple, list] = {}
    for iid, pid, src, seq, total, tipo, situacao in rows:
        key = (src, pid)
        patient_seqs.setdefault(key, []).append((iid, src, seq, tipo, situacao))

    # Compute trajectory vectors and speeds
    speed_by_alta: dict[str, list] = {
        "OBITO": [], "ALTA_NORMAL": [], "TRANSFERENCIA": [], "OUTRO": [], "DESCONHECIDO": [],
    }
    all_speeds: list[tuple] = []  # (speed, patient_key, curr_iid, src, cat)

    for pkey, adm_list in patient_seqs.items():
        adm_list.sort(key=lambda x: x[2])
        for i in range(1, len(adm_list)):
            prev_iid, prev_src, _, _, _ = adm_list[i - 1]
            curr_iid, curr_src, _, curr_tipo, _ = adm_list[i]

            prev_node = f"{prev_src}/ID_CD_INTERNACAO_{prev_iid}"
            curr_node = f"{curr_src}/ID_CD_INTERNACAO_{curr_iid}"

            if prev_node not in node_to_idx or curr_node not in node_to_idx:
                continue

            prev_emb = embeddings[node_to_idx[prev_node]]
            curr_emb = embeddings[node_to_idx[curr_node]]
            delta = curr_emb - prev_emb
            speed = float(np.linalg.norm(delta))

            cat = _categorize_alta(curr_tipo)
            speed_by_alta[cat].append((speed, pkey, curr_iid, curr_src))
            all_speeds.append((speed, pkey, curr_iid, curr_src, cat))

    # Stats per category
    stats_by_alta = {}
    for cat, items in speed_by_alta.items():
        if items:
            speeds = [s for s, _, _, _ in items]
            stats_by_alta[cat] = (float(np.mean(speeds)), float(np.std(speeds)), len(items))
        else:
            stats_by_alta[cat] = (0.0, 0.0, 0)

    # Top 20 fastest and slowest
    all_speeds.sort(key=lambda x: -x[0])
    top_fastest = all_speeds[:20]

    nonzero = [s for s in all_speeds if s[0] > 1e-6]
    nonzero.sort(key=lambda x: x[0])
    top_slowest = nonzero[:20]

    print(f"    Total trajectories: {len(all_speeds):,}")
    for cat, (m, s, n) in stats_by_alta.items():
        if n > 0:
            print(f"    {cat}: mean={m:.4f}, std={s:.4f}, n={n:,}")
    print(f"    Done in {time.time()-t0:.1f}s")

    return speed_by_alta, top_fastest, top_slowest, stats_by_alta


# ─────────────────────────────────────────────────────────────────
# Step 3 — Previsao de Destino por Analogia Geometrica
# ─────────────────────────────────────────────────────────────────

def _geometric_prognosis(unique_nodes, embeddings, node_to_idx, internacao_mask):
    """Pick 10 recent admissions, find 5 nearest completed neighbors, show outcomes."""
    import time
    import numpy as np
    import duckdb

    print("[3/7] Computing geometric prognosis (Analogia Geometrica) ...")
    t0 = time.time()

    con = duckdb.connect(str(DB_PATH))

    # Get recent admissions (last 500 by admission date) — mix of open and closed
    recent_rows = con.execute("""
        SELECT source_db, CAST(ID_CD_INTERNACAO AS VARCHAR) AS eid,
               CAST(ID_CD_PACIENTE AS VARCHAR) AS pid,
               DH_ADMISSAO_HOSP, DH_FINALIZACAO,
               DATEDIFF('day', DH_ADMISSAO_HOSP,
                        COALESCE(DH_FINALIZACAO, CURRENT_DATE)) AS los,
               IN_SITUACAO
        FROM agg_tb_capta_internacao_cain
        WHERE DH_ADMISSAO_HOSP IS NOT NULL AND source_db IS NOT NULL
        ORDER BY DH_ADMISSAO_HOSP DESC
        LIMIT 500
    """).fetchall()

    # Get all completed admissions with outcomes for neighbor lookup
    completed_rows = con.execute("""
        WITH last_status AS (
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
        SELECT i.source_db, CAST(i.ID_CD_INTERNACAO AS VARCHAR) AS eid,
               CAST(i.ID_CD_PACIENTE AS VARCHAR) AS pid,
               i.DH_ADMISSAO_HOSP, i.DH_FINALIZACAO,
               DATEDIFF('day', i.DH_ADMISSAO_HOSP, i.DH_FINALIZACAO) AS los,
               i.IN_SITUACAO,
               di.tipo_alta
        FROM agg_tb_capta_internacao_cain i
        LEFT JOIN discharge_info di
            ON i.ID_CD_INTERNACAO = di.ID_CD_INTERNACAO
            AND i.source_db = di.source_db
        WHERE i.DH_ADMISSAO_HOSP IS NOT NULL
          AND i.DH_FINALIZACAO IS NOT NULL
          AND i.source_db IS NOT NULL
    """).fetchall()

    # CID per admission
    cid_rows = con.execute("""
        SELECT source_db, CAST(ID_CD_INTERNACAO AS VARCHAR) AS eid,
               DS_DESCRICAO AS cid_desc
        FROM agg_tb_capta_cid_caci
        WHERE source_db IS NOT NULL AND DS_DESCRICAO IS NOT NULL
    """).fetchall()

    # Glosa (disallowance) per admission
    glosa_rows = con.execute("""
        SELECT source_db, CAST(ID_CD_INTERNACAO AS VARCHAR) AS eid,
               SUM(CASE WHEN FL_GLOSA = 'S' THEN COALESCE(VL_TOTAL, 0) ELSE 0 END) AS total_glosa
        FROM agg_tb_fatura_fatu
        WHERE source_db IS NOT NULL AND ID_CD_INTERNACAO IS NOT NULL
        GROUP BY source_db, eid
    """).fetchall()

    con.close()

    # Build lookup maps
    cid_map: dict[tuple, str] = {}
    for src, eid, desc in cid_rows:
        cid_map[(src, eid)] = str(desc)

    glosa_map: dict[tuple, float] = {}
    for src, eid, gl in glosa_rows:
        glosa_map[(src, eid)] = _safe_float(gl)

    completed_map: dict[tuple, dict] = {}
    for src, eid, pid, adm, fin, los, sit, tipo in completed_rows:
        node_key = f"{src}/ID_CD_INTERNACAO_{eid}"
        if node_key in node_to_idx:
            completed_map[(src, eid)] = {
                "pid": pid, "adm": adm, "fin": fin, "los": _safe_int(los),
                "tipo_alta": _categorize_alta(tipo), "tipo_raw": tipo,
                "glosa": glosa_map.get((src, eid), 0.0),
                "cid": cid_map.get((src, eid), "---"),
                "node_key": node_key,
            }

    # Build completed embedding matrix for KNN
    completed_keys = list(completed_map.keys())
    completed_indices = [node_to_idx[completed_map[k]["node_key"]] for k in completed_keys]
    completed_embs = embeddings[completed_indices]

    # Normalize for cosine similarity
    norms = np.linalg.norm(completed_embs, axis=1, keepdims=True).clip(min=1e-8)
    completed_embs_n = completed_embs / norms

    # Pick 10 random recent admissions that have embeddings
    rng = np.random.RandomState(42)
    query_candidates = []
    for src, eid, pid, adm, fin, los, sit in recent_rows:
        node_key = f"{src}/ID_CD_INTERNACAO_{eid}"
        if node_key in node_to_idx:
            query_candidates.append({
                "src": src, "eid": eid, "pid": pid,
                "adm": adm, "fin": fin, "los": _safe_int(los),
                "sit": _safe_int(sit),
                "cid": cid_map.get((src, eid), "---"),
                "node_key": node_key,
            })
    if len(query_candidates) > 10:
        idx = rng.choice(len(query_candidates), 10, replace=False)
        queries = [query_candidates[i] for i in idx]
    else:
        queries = query_candidates[:10]

    # For each query, find 5 nearest completed neighbors
    prognosis_results = []
    for q in queries:
        q_emb = embeddings[node_to_idx[q["node_key"]]]
        q_norm = np.linalg.norm(q_emb)
        if q_norm < 1e-8:
            continue
        q_unit = q_emb / q_norm

        sims = completed_embs_n @ q_unit
        # Exclude self
        top_indices = np.argsort(-sims)
        neighbors = []
        for idx in top_indices:
            if len(neighbors) >= 5:
                break
            ckey = completed_keys[idx]
            if ckey == (q["src"], q["eid"]):
                continue
            info = completed_map[ckey]
            neighbors.append({
                "src": ckey[0], "eid": ckey[1],
                "sim": float(sims[idx]),
                **info,
            })

        prognosis_results.append({"query": q, "neighbors": neighbors})

    print(f"    Computed prognosis for {len(prognosis_results)} admissions")
    print(f"    Completed admissions in KNN pool: {len(completed_keys):,}")
    print(f"    Done in {time.time()-t0:.1f}s")

    return prognosis_results


# ─────────────────────────────────────────────────────────────────
# Step 4 — Deteccao de Trajetoria Anomala
# ─────────────────────────────────────────────────────────────────

def _anomalous_trajectories(unique_nodes, embeddings, node_to_idx):
    """For patients with 3+ admissions, detect trajectories that deviate from CID mean."""
    import time
    import numpy as np
    import duckdb

    print("[4/7] Detecting anomalous trajectories ...")
    t0 = time.time()

    con = duckdb.connect(str(DB_PATH))

    # Get patients with 3+ admissions
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
        )
        SELECT ID_CD_INTERNACAO, ID_CD_PACIENTE, source_db, adm_seq, total_adm
        FROM patient_admissions
        WHERE total_adm >= 3
        ORDER BY source_db, ID_CD_PACIENTE, adm_seq
    """).fetchall()

    # CID per admission
    cid_rows = con.execute("""
        SELECT source_db, CAST(ID_CD_INTERNACAO AS VARCHAR) AS eid,
               DS_DESCRICAO AS cid_desc
        FROM agg_tb_capta_cid_caci
        WHERE source_db IS NOT NULL AND DS_DESCRICAO IS NOT NULL
    """).fetchall()

    # Admission context for display
    adm_context = con.execute("""
        SELECT source_db, CAST(ID_CD_INTERNACAO AS VARCHAR) AS eid,
               CAST(ID_CD_PACIENTE AS VARCHAR) AS pid,
               DH_ADMISSAO_HOSP,
               DATEDIFF('day', DH_ADMISSAO_HOSP,
                        COALESCE(DH_FINALIZACAO, CURRENT_DATE)) AS los
        FROM agg_tb_capta_internacao_cain
        WHERE DH_ADMISSAO_HOSP IS NOT NULL AND source_db IS NOT NULL
    """).fetchall()

    con.close()

    cid_map: dict[tuple, str] = {}
    for src, eid, desc in cid_rows:
        cid_map[(src, eid)] = str(desc)

    context_map: dict[tuple, dict] = {}
    for src, eid, pid, adm, los in adm_context:
        context_map[(src, eid)] = {"pid": pid, "adm": adm, "los": _safe_int(los)}

    print(f"    Found {len(rows):,} admissions from patients with 3+ visits")

    # Build patient sequences
    patient_seqs: dict[tuple, list] = {}
    for iid, pid, src, seq, total in rows:
        key = (src, pid)
        patient_seqs.setdefault(key, []).append((iid, src, seq))

    # Compute all trajectory deltas and group by CID
    cid_deltas: dict[str, list] = {}  # cid_desc -> list of delta vectors
    patient_deltas: list[dict] = []    # all individual deltas with context

    for pkey, adm_list in patient_seqs.items():
        adm_list.sort(key=lambda x: x[2])
        for i in range(1, len(adm_list)):
            prev_iid, prev_src, _ = adm_list[i - 1]
            curr_iid, curr_src, _ = adm_list[i]

            prev_node = f"{prev_src}/ID_CD_INTERNACAO_{prev_iid}"
            curr_node = f"{curr_src}/ID_CD_INTERNACAO_{curr_iid}"

            if prev_node not in node_to_idx or curr_node not in node_to_idx:
                continue

            prev_emb = embeddings[node_to_idx[prev_node]]
            curr_emb = embeddings[node_to_idx[curr_node]]
            delta = curr_emb - prev_emb

            cid = cid_map.get((curr_src, str(curr_iid)), "DESCONHECIDO")
            cid_deltas.setdefault(cid, []).append(delta)

            ctx = context_map.get((curr_src, str(curr_iid)), {})
            patient_deltas.append({
                "pkey": pkey, "curr_iid": curr_iid, "src": curr_src,
                "delta": delta, "cid": cid,
                "pid": ctx.get("pid", "?"), "adm": ctx.get("adm"),
                "los": ctx.get("los", 0),
            })

    # Compute mean trajectory per CID
    cid_mean_deltas: dict[str, np.ndarray] = {}
    for cid, deltas in cid_deltas.items():
        if len(deltas) >= 5:  # need sufficient samples
            cid_mean_deltas[cid] = np.stack(deltas).mean(axis=0)

    print(f"    CIDs with sufficient trajectories for mean: {len(cid_mean_deltas)}")

    # Score each patient delta by cosine similarity to CID mean
    anomaly_scores: list[dict] = []
    for pd in patient_deltas:
        cid = pd["cid"]
        if cid not in cid_mean_deltas:
            continue

        mean_d = cid_mean_deltas[cid]
        delta = pd["delta"]

        d_norm = np.linalg.norm(delta)
        m_norm = np.linalg.norm(mean_d)
        if d_norm < 1e-8 or m_norm < 1e-8:
            continue

        cos_sim = float(np.dot(delta / d_norm, mean_d / m_norm))
        anomaly_scores.append({
            "cos_sim": cos_sim,
            "speed": float(d_norm),
            **pd,
        })

    # Sort by lowest cosine similarity (most anomalous)
    anomaly_scores.sort(key=lambda x: x["cos_sim"])
    top_anomalous = anomaly_scores[:20]

    print(f"    Total scored trajectories: {len(anomaly_scores):,}")
    if top_anomalous:
        print(f"    Most anomalous cos_sim: {top_anomalous[0]['cos_sim']:.4f}")
    print(f"    Done in {time.time()-t0:.1f}s")

    return top_anomalous, len(anomaly_scores)


# ─────────────────────────────────────────────────────────────────
# Step 5 — Simulacao Contrafactual Temporal
# ─────────────────────────────────────────────────────────────────

def _counterfactual_simulation(unique_nodes, embeddings, node_to_idx, internacao_mask):
    """Hospital swap counterfactual: patient_emb - hospital_Y_centroid + hospital_X_centroid."""
    import time
    import numpy as np
    import duckdb

    print("[5/7] Running counterfactual hospital simulations ...")
    t0 = time.time()

    int_nodes = unique_nodes[internacao_mask]
    int_embs  = embeddings[internacao_mask]

    # Build hospital centroids
    hospital_groups: dict[str, list[int]] = {}
    for i, node in enumerate(int_nodes):
        s = str(node)
        try:
            src_db = s.split("/")[0]
            hospital_groups.setdefault(src_db, []).append(i)
        except Exception:
            pass

    hospital_centroids = {}
    for h, idxs in hospital_groups.items():
        hospital_centroids[h] = int_embs[idxs].mean(axis=0)

    available_hospitals = [h for h in MAJOR_HOSPITALS if h in hospital_centroids]
    print(f"    Available major hospitals: {available_hospitals}")

    if len(available_hospitals) < 2:
        # Fallback: use whatever hospitals exist
        available_hospitals = sorted(hospital_groups.keys())[:5]
        print(f"    Fallback to: {available_hospitals}")

    # Normalize all internacao embeddings for neighbor search
    int_norms = np.linalg.norm(int_embs, axis=1, keepdims=True).clip(min=1e-8)
    int_embs_n = int_embs / int_norms

    con = duckdb.connect(str(DB_PATH))

    # Get admission details for context lookup
    adm_rows = con.execute("""
        WITH last_status AS (
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
        SELECT i.source_db, CAST(i.ID_CD_INTERNACAO AS VARCHAR) AS eid,
               CAST(i.ID_CD_PACIENTE AS VARCHAR) AS pid,
               DATEDIFF('day', i.DH_ADMISSAO_HOSP,
                        COALESCE(i.DH_FINALIZACAO, CURRENT_DATE)) AS los,
               di.tipo_alta
        FROM agg_tb_capta_internacao_cain i
        LEFT JOIN discharge_info di
            ON i.ID_CD_INTERNACAO = di.ID_CD_INTERNACAO
            AND i.source_db = di.source_db
        WHERE i.DH_ADMISSAO_HOSP IS NOT NULL AND i.source_db IS NOT NULL
    """).fetchall()

    cid_rows = con.execute("""
        SELECT source_db, CAST(ID_CD_INTERNACAO AS VARCHAR) AS eid,
               DS_DESCRICAO AS cid_desc
        FROM agg_tb_capta_cid_caci
        WHERE source_db IS NOT NULL AND DS_DESCRICAO IS NOT NULL
    """).fetchall()

    con.close()

    adm_map: dict[tuple, dict] = {}
    for src, eid, pid, los, tipo in adm_rows:
        adm_map[(src, eid)] = {
            "pid": pid, "los": _safe_int(los),
            "tipo_alta": _categorize_alta(tipo), "tipo_raw": tipo,
        }

    cid_map: dict[tuple, str] = {}
    for src, eid, desc in cid_rows:
        cid_map[(src, eid)] = str(desc)

    # Node name to (src, eid) lookup
    def _parse_node(node_name: str):
        try:
            src, id_part = str(node_name).split("/", 1)
            eid = id_part.split("ID_CD_INTERNACAO_")[1]
            return src, eid
        except Exception:
            return None, None

    def _find_nearest(point, k=5):
        """Find k nearest real internacao nodes to a hypothetical point."""
        p_norm = np.linalg.norm(point)
        if p_norm < 1e-8:
            return []
        p_unit = point / p_norm
        sims = int_embs_n @ p_unit
        top_k = np.argsort(-sims)[:k]
        results = []
        for idx in top_k:
            nn_name = str(int_nodes[idx])
            nn_src, nn_eid = _parse_node(nn_name)
            nn_info = adm_map.get((nn_src, nn_eid), {})
            results.append({
                "node": nn_name, "sim": float(sims[idx]),
                "src": nn_src, "eid": nn_eid,
                "los": nn_info.get("los", 0),
                "tipo_alta": nn_info.get("tipo_alta", "?"),
                "cid": cid_map.get((nn_src, nn_eid), "---"),
            })
        return results

    # For each pair (hospital_origin, hospital_target), pick random patients
    # and simulate the transfer
    rng = np.random.RandomState(42)
    simulation_results = []

    for h_origin in available_hospitals:
        for h_target in available_hospitals:
            if h_origin == h_target:
                continue

            origin_indices = hospital_groups.get(h_origin, [])
            if len(origin_indices) < 10:
                continue

            # Pick 3 random patients from origin
            sample_idx = rng.choice(len(origin_indices), min(3, len(origin_indices)), replace=False)
            for si in sample_idx:
                idx = origin_indices[si]
                original_emb = int_embs[idx]
                node_name = str(int_nodes[idx])

                # Counterfactual: patient_emb - origin_centroid + target_centroid
                counterfactual_emb = (
                    original_emb
                    - hospital_centroids[h_origin]
                    + hospital_centroids[h_target]
                )

                # Find nearest real admissions to counterfactual
                nn_original = _find_nearest(original_emb, k=3)
                nn_counter  = _find_nearest(counterfactual_emb, k=3)

                src, eid = _parse_node(node_name)
                info = adm_map.get((src, eid), {})

                simulation_results.append({
                    "node": node_name, "src": src, "eid": eid,
                    "h_origin": h_origin, "h_target": h_target,
                    "original_los": info.get("los", 0),
                    "original_tipo": info.get("tipo_alta", "?"),
                    "original_cid": cid_map.get((src, eid), "---"),
                    "nn_original": nn_original,
                    "nn_counter": nn_counter,
                })

    # Score by "dramatic difference" = change in dominant discharge type or LOS
    for sr in simulation_results:
        orig_tipos = [n["tipo_alta"] for n in sr["nn_original"]]
        cf_tipos   = [n["tipo_alta"] for n in sr["nn_counter"]]
        orig_los   = [n["los"] for n in sr["nn_original"] if n["los"] > 0]
        cf_los     = [n["los"] for n in sr["nn_counter"] if n["los"] > 0]

        # Score: tipo change (obito appeared/disappeared) + LOS difference
        drama = 0.0
        if "OBITO" in cf_tipos and "OBITO" not in orig_tipos:
            drama += 10.0
        elif "OBITO" in orig_tipos and "OBITO" not in cf_tipos:
            drama += 8.0
        if orig_los and cf_los:
            los_diff = abs(np.mean(cf_los) - np.mean(orig_los))
            drama += los_diff / 10.0
        sr["drama_score"] = drama

    simulation_results.sort(key=lambda x: -x["drama_score"])
    top_simulations = simulation_results[:5]

    print(f"    Total simulations: {len(simulation_results):,}")
    print(f"    Top drama score: {top_simulations[0]['drama_score']:.2f}" if top_simulations else "    No simulations")
    print(f"    Done in {time.time()-t0:.1f}s")

    return top_simulations, len(simulation_results), available_hospitals


# ─────────────────────────────────────────────────────────────────
# Step 6 — Clustering Temporal de Perfis de Evolucao
# ─────────────────────────────────────────────────────────────────

def _trajectory_clustering(unique_nodes, embeddings, node_to_idx):
    """K-means (k=10) on trajectory vectors (deltas, not raw embeddings)."""
    import time
    import numpy as np
    import duckdb
    from sklearn.cluster import KMeans

    print("[6/7] Clustering trajectory profiles (K-means on deltas) ...")
    t0 = time.time()

    con = duckdb.connect(str(DB_PATH))

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
            di.tipo_alta
        FROM patient_admissions pa
        LEFT JOIN discharge_info di
            ON pa.ID_CD_INTERNACAO = di.ID_CD_INTERNACAO
            AND pa.source_db = di.source_db
        WHERE pa.total_adm >= 2
        ORDER BY pa.source_db, pa.ID_CD_PACIENTE, pa.adm_seq
    """).fetchall()

    cid_rows = con.execute("""
        SELECT source_db, CAST(ID_CD_INTERNACAO AS VARCHAR) AS eid,
               DS_DESCRICAO AS cid_desc
        FROM agg_tb_capta_cid_caci
        WHERE source_db IS NOT NULL AND DS_DESCRICAO IS NOT NULL
    """).fetchall()

    con.close()

    cid_map: dict[tuple, str] = {}
    for src, eid, desc in cid_rows:
        cid_map[(src, eid)] = str(desc)

    # Build patient sequences
    patient_seqs: dict[tuple, list] = {}
    for iid, pid, src, seq, tipo in rows:
        key = (src, pid)
        patient_seqs.setdefault(key, []).append((iid, src, seq, tipo))

    # Compute trajectory deltas
    trajectory_data: list[dict] = []  # each entry has delta, metadata

    for pkey, adm_list in patient_seqs.items():
        adm_list.sort(key=lambda x: x[2])
        for i in range(1, len(adm_list)):
            prev_iid, prev_src, _, _ = adm_list[i - 1]
            curr_iid, curr_src, _, curr_tipo = adm_list[i]

            prev_node = f"{prev_src}/ID_CD_INTERNACAO_{prev_iid}"
            curr_node = f"{curr_src}/ID_CD_INTERNACAO_{curr_iid}"

            if prev_node not in node_to_idx or curr_node not in node_to_idx:
                continue

            prev_emb = embeddings[node_to_idx[prev_node]]
            curr_emb = embeddings[node_to_idx[curr_node]]
            delta = curr_emb - prev_emb

            cat = _categorize_alta(curr_tipo)
            cid = cid_map.get((curr_src, str(curr_iid)), "DESCONHECIDO")

            trajectory_data.append({
                "delta": delta,
                "speed": float(np.linalg.norm(delta)),
                "pkey": pkey, "src": curr_src,
                "iid": curr_iid, "cat": cat, "cid": cid,
            })

    if len(trajectory_data) < 100:
        print("    Not enough trajectories for clustering")
        return [], {}

    # Stack all deltas for K-means
    delta_matrix = np.stack([td["delta"] for td in trajectory_data])
    print(f"    Trajectory matrix: {delta_matrix.shape}")

    # K-means k=10
    n_clusters = min(10, len(trajectory_data) // 10)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(delta_matrix)

    # Assign cluster labels
    for i, td in enumerate(trajectory_data):
        td["cluster"] = int(labels[i])

    # Compute cluster profiles
    cluster_profiles: dict[int, dict] = {}
    for cl in range(n_clusters):
        members = [td for td in trajectory_data if td["cluster"] == cl]
        speeds = [td["speed"] for td in members]
        cats = [td["cat"] for td in members]
        cids = [td["cid"] for td in members]

        # Dominant discharge type
        cat_counts: dict[str, int] = {}
        for c in cats:
            cat_counts[c] = cat_counts.get(c, 0) + 1
        dominant_cat = max(cat_counts, key=cat_counts.get) if cat_counts else "?"
        cat_pct = cat_counts.get(dominant_cat, 0) / max(len(members), 1) * 100

        # Dominant CID
        cid_counts: dict[str, int] = {}
        for c in cids:
            if c != "DESCONHECIDO":
                cid_counts[c] = cid_counts.get(c, 0) + 1
        top_cids = sorted(cid_counts.items(), key=lambda x: -x[1])[:3]

        cluster_profiles[cl] = {
            "n_patients": len(members),
            "mean_speed": float(np.mean(speeds)),
            "std_speed": float(np.std(speeds)),
            "dominant_cat": dominant_cat,
            "cat_pct": cat_pct,
            "cat_counts": cat_counts,
            "top_cids": top_cids,
        }

    # Sort clusters by size (largest first)
    sorted_clusters = sorted(cluster_profiles.items(), key=lambda x: -x[1]["n_patients"])

    # Narrative archetype interpretation
    for cl, profile in cluster_profiles.items():
        speed = profile["mean_speed"]
        dom = profile["dominant_cat"]
        if dom == "OBITO" and speed > np.median([p["mean_speed"] for p in cluster_profiles.values()]):
            profile["archetype"] = "Declinio rapido com desfecho adverso"
            profile["archetype_desc"] = (
                "Pacientes com alta velocidade de mudanca no espaco latente "
                "e predominancia de obito. Sugere deterioracao clinica acelerada."
            )
        elif dom == "OBITO":
            profile["archetype"] = "Declinio lento com desfecho adverso"
            profile["archetype_desc"] = (
                "Pacientes com baixa velocidade de mudanca mas desfecho de obito. "
                "Sugere condicoes cronicas com longa trajetoria ate o desfecho."
            )
        elif dom == "ALTA_NORMAL" and speed > np.median([p["mean_speed"] for p in cluster_profiles.values()]):
            profile["archetype"] = "Recuperacao rapida"
            profile["archetype_desc"] = (
                "Pacientes com grande deslocamento no espaco latente entre internacoes "
                "e alta normal. Perfil de intervencao eficaz e recuperacao significativa."
            )
        elif dom == "ALTA_NORMAL":
            profile["archetype"] = "Estabilidade cronica com alta"
            profile["archetype_desc"] = (
                "Pacientes com pouca mudanca entre internacoes e desfecho positivo. "
                "Perfil de condicoes estáveis com reinternacoes de rotina."
            )
        elif dom == "TRANSFERENCIA":
            profile["archetype"] = "Perfil de transferencia"
            profile["archetype_desc"] = (
                "Cluster dominado por transferencias entre unidades. "
                "Pacientes que requerem mudanca de nivel de complexidade."
            )
        else:
            profile["archetype"] = "Perfil misto"
            profile["archetype_desc"] = (
                "Cluster com distribuicao heterogenea de desfechos. "
                "Pode representar pacientes em fases intermediarias de tratamento."
            )

    print(f"    {n_clusters} clusters computed")
    for cl, prof in sorted_clusters[:5]:
        print(f"    Cluster {cl}: n={prof['n_patients']:,}, speed={prof['mean_speed']:.4f}, "
              f"dom={prof['dominant_cat']} ({prof['cat_pct']:.0f}%), archetype={prof['archetype']}")
    print(f"    Done in {time.time()-t0:.1f}s")

    return sorted_clusters, cluster_profiles


# ─────────────────────────────────────────────────────────────────
# Step 7 — Generate LaTeX
# ─────────────────────────────────────────────────────────────────

def _generate_latex(
    stats_by_alta, top_fastest, top_slowest,
    prognosis_results,
    top_anomalous, total_anomaly_scored,
    top_simulations, total_simulations, available_hospitals,
    sorted_clusters, cluster_profiles,
    n_embeddings, emb_dim,
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
\definecolor{clusterA}{RGB}{52,101,164}
\definecolor{clusterB}{RGB}{78,154,6}
\definecolor{clusterC}{RGB}{196,160,0}
\definecolor{clusterD}{RGB}{204,0,0}
\definecolor{clusterE}{RGB}{117,80,123}

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

\newtcolorbox{counterbox}[1][]{%
  enhanced,
  colback=alertyellow,
  colframe=trafficyellow,
  fonttitle=\bfseries\footnotesize,
  title={#1},
  left=5pt, right=5pt, top=3pt, bottom=3pt,
  boxrule=0.8pt,
}

\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\textcolor{jcubeblue}{\textbf{JCUBE Digital Twin}} \textcolor{jcubegray}{\small | Temporal Twin --- V6.2}}
\fancyhead[R]{\textcolor{jcubegray}{\small 28/03/2026}}
\fancyfoot[C]{\textcolor{jcubegray}{\thepage}}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0.4pt}

\titleformat{\section}{\large\bfseries\color{jcubeblue}}{\thesection}{1em}{}[\titlerule]
\titleformat{\subsection}{\normalsize\bfseries\color{darkblue}}{\thesubsection}{1em}{}

\hypersetup{colorlinks=true,linkcolor=jcubeblue,pdftitle={JCUBE V6.2 Gemeo Digital Temporal}}

\begin{document}
\setlength{\parindent}{0pt}
\setlength{\parskip}{3pt}
""")

    # ── Title page ──
    n_total_traj = sum(s[2] for s in stats_by_alta.values())
    n_clusters = len(cluster_profiles)
    n_anomalies = len(top_anomalous)
    n_sims = total_simulations

    L.append(r"""\begin{titlepage}
\begin{center}
\vspace*{1.5cm}
{\Huge\bfseries\textcolor{jcubeblue}{JCUBE}}\\[0.2cm]
{\large\textcolor{jcubegray}{Digital Twin Analytics Platform --- Modelo V6.2}}\\[1.2cm]
\begin{tcolorbox}[colback=jcubeblue,colframe=jcubeblue,coltext=white,width=0.92\textwidth,halign=center]
{\LARGE\bfseries Gemeo Digital Temporal}\\[0.3cm]
{\large O que um twin temporal pode fazer\\que um twin estatico nao pode}\\[0.2cm]
{\normalsize TGN Temporal Graph Network --- 4$\times$ H100 --- Epoch 5}
\end{tcolorbox}
\vspace{0.8cm}
""")

    L.append(r"{\Large Gerado em: \textbf{28 de marco de 2026}}\\[0.6cm]")

    L.append(
        r"""\begin{tabular}{cccc}
\begin{tcolorbox}[colback=jcubeblue!10,colframe=jcubeblue,width=3.2cm,halign=center,left=2pt,right=2pt]
{\LARGE\bfseries\textcolor{jcubeblue}{""" + f"{n_total_traj:,}" + r"""}}\\[2pt]
{\scriptsize\textbf{Trajetorias}}
\end{tcolorbox}
&
\begin{tcolorbox}[colback=trafficgreen!10,colframe=trafficgreen,width=3.2cm,halign=center,left=2pt,right=2pt]
{\LARGE\bfseries\textcolor{trafficgreen}{""" + str(n_clusters) + r"""}}\\[2pt]
{\scriptsize\textbf{Arquetipos}}
\end{tcolorbox}
&
\begin{tcolorbox}[colback=trafficred!10,colframe=trafficred,width=3.2cm,halign=center,left=2pt,right=2pt]
{\LARGE\bfseries\textcolor{trafficred}{""" + str(n_anomalies) + r"""}}\\[2pt]
{\scriptsize\textbf{Anomalias Top}}
\end{tcolorbox}
&
\begin{tcolorbox}[colback=trafficyellow!10,colframe=trafficyellow,width=3.2cm,halign=center,left=2pt,right=2pt]
{\LARGE\bfseries\textcolor{trafficyellow!80!black}{""" + f"{n_sims:,}" + r"""}}\\[2pt]
{\scriptsize\textbf{Contrafactuais}}
\end{tcolorbox}
\end{tabular}

\vfill
{\small\textcolor{jcubegray}{
Metodologia: operacoes de algebra vetorial temporal sobre embeddings do\\
TGN V6.2 com 35,2M nos $\times$ 128 dimensoes --- treinado em 4$\times$ H100\\
Vetores de trajetoria, KNN temporal, clustering de deltas, simulacao contrafactual
}}
\end{center}
\end{titlepage}
""")

    L.append(r"\tableofcontents\clearpage")

    # ═══════════════════════════════════════════════════════════════
    # SECTION 1: Velocidade de Evolucao Clinica
    # ═══════════════════════════════════════════════════════════════

    L.append(r"\section{Velocidade de Evolucao Clinica}")
    L.append(r"""
Para pacientes com 2+ internacoes, computamos o \textbf{vetor de trajetoria}:
$\vec{\delta} = \text{emb}(\text{internacao}_n) - \text{emb}(\text{internacao}_{n-1})$.
A \textbf{velocidade de evolucao} e a norma $\|\vec{\delta}\|$ --- quanto maior, mais
o paciente ``mudou'' no espaco latente entre internacoes consecutivas.

\textbf{Por que isso importa:} um twin estatico compara pacientes em um unico ponto
no tempo. Um twin temporal mede a \emph{taxa de mudanca}, revelando quem esta evoluindo
rapido (potencial deterioracao ou recuperacao) e quem esta estagnado (possivel alta prematura
ou tratamento ineficaz).
""")

    # Stats table
    L.append(r"\subsection{Velocidade Media por Tipo de Alta}")
    L.append(r"\begin{center}")
    L.append(r"\begin{tabular}{lrrr}")
    L.append(r"\toprule \textbf{Tipo de Alta} & \textbf{$\|\vec{\delta}\|$ Media} & \textbf{Desvio Padrao} & \textbf{N Trajetorias} \\\midrule")
    for cat in ["OBITO", "ALTA_NORMAL", "TRANSFERENCIA", "OUTRO", "DESCONHECIDO"]:
        mean, std, n = stats_by_alta.get(cat, (0, 0, 0))
        if n == 0:
            continue
        color = "trafficred" if cat == "OBITO" else "jcubeblue"
        L.append(
            _escape_latex(cat) + r" & \textcolor{" + color + r"}{\textbf{" +
            f"{mean:.4f}" + r"}} & " + f"{std:.4f}" + r" & " + f"{n:,}" + r" \\"
        )
    L.append(r"\bottomrule\end{tabular}")
    L.append(r"\end{center}")

    # Insight
    obito_speed = stats_by_alta.get("OBITO", (0, 0, 0))[0]
    alta_speed = stats_by_alta.get("ALTA_NORMAL", (0, 0, 0))[0]
    if obito_speed > 0 and alta_speed > 0:
        ratio = obito_speed / alta_speed if alta_speed > 1e-9 else 0
        L.append(r"\begin{insightbox}[Velocidade Obito vs Alta Normal]")
        L.append(
            r"Pacientes com desfecho obito apresentam velocidade media de " +
            f"{obito_speed:.4f}" + r", enquanto alta normal apresenta " +
            f"{alta_speed:.4f}" + r" (razao: " + f"{ratio:.2f}x" + r"). "
        )
        if ratio > 1.2:
            L.append(
                r"O grupo de obito evolui \textbf{mais rapido} no espaco latente, "
                r"sugerindo que a deterioracao clinica produz mudancas maiores "
                r"nos embeddings temporais do que a recuperacao gradual."
            )
        elif ratio < 0.8:
            L.append(
                r"O grupo de alta normal evolui mais rapido, sugerindo que a recuperacao "
                r"produz reorganizacao significativa no espaco latente."
            )
        else:
            L.append(
                r"As velocidades sao comparaveis, indicando que ambos os desfechos "
                r"envolvem mudancas significativas na representacao temporal."
            )
        L.append(r"\end{insightbox}")

    # Top 20 fastest
    L.append(r"\subsection{Top 20 Pacientes com Maior Velocidade (Evolucao Rapida)}")
    L.append(r"""
Pacientes cuja trajetoria apresenta a maior norma $\|\vec{\delta}\|$ --- indicando
\textbf{mudanca acelerada} no espaco latente. Podem representar deterioracao rapida
ou resposta intensa ao tratamento.
""")
    if top_fastest:
        L.append(r"\begin{center}{\scriptsize")
        L.append(r"\begin{longtable}{rlllr}")
        L.append(r"\toprule \textbf{\#} & \textbf{Hospital} & \textbf{ID Paciente} & \textbf{Tipo Alta} & \textbf{$\|\vec{\delta}\|$} \\\midrule")
        L.append(r"\endhead\bottomrule\endfoot")
        for rank, (speed, pkey, curr_iid, curr_src, cat) in enumerate(top_fastest, 1):
            src_db, pid = pkey
            color = "trafficred" if cat == "OBITO" else ("trafficyellow" if cat == "OUTRO" else "jcubeblue")
            L.append(
                str(rank) + r" & " +
                _escape_latex(str(src_db)[:20]) + r" & " +
                str(pid) + r" & " +
                _escape_latex(cat) + r" & " +
                r"\textcolor{" + color + r"}{\textbf{" + f"{speed:.4f}" + r"}} \\"
            )
        L.append(r"\end{longtable}}")
        L.append(r"\end{center}")

    # Top 20 slowest
    L.append(r"\subsection{Top 20 Pacientes com Menor Velocidade (Estagnacao)}")
    L.append(r"""
Pacientes cuja trajetoria e quase estacionaria --- $\|\vec{\delta}\| \approx 0$.
Possivel indicacao de \textbf{alta prematura} (paciente nao mudou o suficiente)
ou \textbf{tratamento sem efeito} (reinternacao sem evolucao).
""")
    if top_slowest:
        L.append(r"\begin{center}{\scriptsize")
        L.append(r"\begin{longtable}{rlllr}")
        L.append(r"\toprule \textbf{\#} & \textbf{Hospital} & \textbf{ID Paciente} & \textbf{Tipo Alta} & \textbf{$\|\vec{\delta}\|$} \\\midrule")
        L.append(r"\endhead\bottomrule\endfoot")
        for rank, (speed, pkey, curr_iid, curr_src, cat) in enumerate(top_slowest, 1):
            src_db, pid = pkey
            color = _traffic_color_inv(speed, 0.001, 0.01)
            L.append(
                str(rank) + r" & " +
                _escape_latex(str(src_db)[:20]) + r" & " +
                str(pid) + r" & " +
                _escape_latex(cat) + r" & " +
                r"\textcolor{" + color + r"}{\textbf{" + f"{speed:.6f}" + r"}} \\"
            )
        L.append(r"\end{longtable}}")
        L.append(r"\end{center}")

    L.append(r"\clearpage")

    # ═══════════════════════════════════════════════════════════════
    # SECTION 2: Previsao de Destino por Analogia Geometrica
    # ═══════════════════════════════════════════════════════════════

    L.append(r"\section{Previsao de Destino por Analogia Geometrica}")
    L.append(r"""
Para cada internacao atual (aberta ou recente), buscamos os 5 vizinhos mais proximos
no espaco de embeddings entre internacoes \textbf{ja concluidas}. Os desfechos reais
dos vizinhos servem como \textbf{prognostico por analogia geometrica} --- sem nenhum
modelo de classificacao, apenas KNN no espaco latente.

\textbf{Por que isso importa:} um twin estatico nao consegue diferenciar entre dois
pacientes no mesmo estado \emph{agora}. O twin temporal codifica a \emph{historia}
do paciente no embedding, permitindo prognosticos baseados em trajetorias similares.
""")

    for idx, pr in enumerate(prognosis_results):
        q = pr["query"]
        nn = pr["neighbors"]

        L.append(r"\needspace{8\baselineskip}")
        L.append(r"\subsection{Caso " + str(idx + 1) + r": " +
                 _escape_latex(str(q['src'])[:20]) + r" / Internacao " +
                 _escape_latex(str(q['eid'])) + r"}")

        # Query admission info
        status_str = "EM CURSO" if q.get("sit") != 2 else "FINALIZADA"
        L.append(r"\begin{sectioncard}[Internacao Consulta]{jcubeblue}")
        L.append(
            r"\textbf{Hospital:} \texttt{" + _escape_latex(str(q['src'])) + r"} \quad "
            r"\textbf{Paciente:} " + _escape_latex(str(q['pid'])) + r" \quad "
            r"\textbf{LOS:} " + str(q['los']) + r" dias \quad "
            r"\textbf{Status:} " + _escape_latex(status_str) + r" \quad "
            r"\textbf{CID:} " + _escape_latex(_truncate(str(q.get('cid', '---')), 50))
        )
        L.append(r"\end{sectioncard}")
        L.append(r"\vspace{2pt}")

        # Neighbors table
        if nn:
            L.append(r"\begin{center}{\scriptsize")
            L.append(r"\begin{tabular}{rllrrll}")
            L.append(r"\toprule \textbf{\#} & \textbf{Hospital} & \textbf{Paciente} & \textbf{LOS} & \textbf{Sim.} & \textbf{Desfecho} & \textbf{CID} \\\midrule")
            for ni, n in enumerate(nn, 1):
                tipo_color = "trafficred" if n.get("tipo_alta") == "OBITO" else "jcubeblue"
                L.append(
                    str(ni) + r" & " +
                    _escape_latex(str(n.get('src', '?'))[:15]) + r" & " +
                    _escape_latex(str(n.get('pid', '?'))) + r" & " +
                    str(n.get('los', 0)) + r" & " +
                    f"{n.get('sim', 0):.4f}" + r" & " +
                    r"\textcolor{" + tipo_color + r"}{" + _escape_latex(str(n.get('tipo_alta', '?'))) + r"} & " +
                    _escape_latex(_truncate(str(n.get('cid', '---')), 30)) + r" \\"
                )
            L.append(r"\bottomrule\end{tabular}}")
            L.append(r"\end{center}")

            # Summary insight
            tipos = [n.get("tipo_alta", "?") for n in nn]
            los_vals = [n.get("los", 0) for n in nn if n.get("los", 0) > 0]
            dominant_tipo = max(set(tipos), key=tipos.count)
            avg_los = np.mean(los_vals) if los_vals else 0

            L.append(r"\begin{insightbox}[Prognostico Geometrico]")
            L.append(
                r"Desfecho predominante entre vizinhos: \textbf{" +
                _escape_latex(dominant_tipo) + r"} (" +
                str(tipos.count(dominant_tipo)) + r"/5). "
                r"LOS medio dos vizinhos: \textbf{" + f"{avg_los:.0f}" + r" dias}."
            )
            if dominant_tipo == "OBITO":
                L.append(
                    r" \textcolor{trafficred}{\textbf{ATENCAO:}} a maioria dos vizinhos "
                    r"geometricos teve desfecho de obito --- sugere-se revisao clinica."
                )
            L.append(r"\end{insightbox}")

        L.append(r"\vspace{6pt}")

    L.append(r"\clearpage")

    # ═══════════════════════════════════════════════════════════════
    # SECTION 3: Deteccao de Trajetoria Anomala
    # ═══════════════════════════════════════════════════════════════

    L.append(r"\section{Deteccao de Trajetoria Anomala}")
    L.append(r"""
Para pacientes com 3+ internacoes, calculamos a \textbf{direcao esperada} como a
trajetoria media de todos os pacientes com o mesmo CID. Uma trajetoria e
\textbf{anomala} quando seu vetor $\vec{\delta}$ desvia significativamente da direcao
esperada (baixa similaridade cosseno com a media do CID).

\textbf{Por que isso importa:} deteccao de anomalias tradicionais compara valores
estaticos (LOS, custos). A deteccao temporal identifica pacientes cuja
\emph{evolucao no tempo} e atipica --- mesmo que seus valores pontuais estejam normais.
""")

    L.append(r"\subsection{Top 20 Trajetorias Mais Anomalas}")
    L.append(
        r"Total de trajetorias avaliadas: \textbf{" + f"{total_anomaly_scored:,}" + r"}. "
        r"Abaixo, as 20 com menor similaridade cosseno em relacao a media do CID."
    )

    if top_anomalous:
        L.append(r"\begin{alertbox}[ALERTA: Trajetorias Atipicas]")
        L.append(
            r"Os pacientes abaixo apresentam evolucao temporal significativamente diferente "
            r"de outros pacientes com o mesmo CID. Recomenda-se investigacao clinica."
        )
        L.append(r"\end{alertbox}")
        L.append(r"\vspace{4pt}")
        L.append(r"\begin{center}{\scriptsize")
        L.append(r"\begin{longtable}{rllllrr}")
        L.append(r"\toprule \textbf{\#} & \textbf{Hospital} & \textbf{Paciente} & \textbf{Internacao} & \textbf{CID} & \textbf{Cos. CID} & \textbf{Velocidade} \\\midrule")
        L.append(r"\endhead\bottomrule\endfoot")
        for rank, anom in enumerate(top_anomalous, 1):
            cos_val = anom["cos_sim"]
            color = "trafficred" if cos_val < 0 else ("trafficyellow" if cos_val < 0.3 else "jcubeblue")
            L.append(
                str(rank) + r" & " +
                _escape_latex(str(anom['src'])[:15]) + r" & " +
                _escape_latex(str(anom['pid'])) + r" & " +
                str(anom['curr_iid']) + r" & " +
                _escape_latex(_truncate(str(anom['cid']), 25)) + r" & " +
                r"\textcolor{" + color + r"}{\textbf{" + f"{cos_val:.4f}" + r"}} & " +
                f"{anom['speed']:.4f}" + r" \\"
            )
        L.append(r"\end{longtable}}")
        L.append(r"\end{center}")

        # Interpretation
        neg_count = sum(1 for a in top_anomalous if a["cos_sim"] < 0)
        L.append(r"\begin{insightbox}[Interpretacao]")
        L.append(
            r"Das 20 trajetorias mais anomalas, \textbf{" + str(neg_count) +
            r"} apresentam similaridade cosseno \textbf{negativa}, significando que "
            r"a evolucao do paciente vai na \textbf{direcao oposta} a esperada para "
            r"pacientes com o mesmo CID. Isto pode indicar: complicacoes inesperadas, "
            r"erro diagnostico, ou resposta atipica ao tratamento."
        )
        L.append(r"\end{insightbox}")
    else:
        L.append(r"Nenhuma trajetoria anomala significativa encontrada.")

    L.append(r"\clearpage")

    # ═══════════════════════════════════════════════════════════════
    # SECTION 4: Simulacao Contrafactual Temporal
    # ═══════════════════════════════════════════════════════════════

    L.append(r"\section{Simulacao Contrafactual Temporal}")
    L.append(r"""
\textbf{Pergunta:} ``E se este paciente tivesse sido internado no hospital X em vez do hospital Y?''

\textbf{Metodo:} tomamos o embedding do paciente, subtraimos o centroide do hospital de
origem e somamos o centroide do hospital de destino:
$$\vec{e}_{\text{contrafactual}} = \vec{e}_{\text{paciente}} - \vec{c}_{\text{origem}} + \vec{c}_{\text{destino}}$$
Buscamos as internacoes reais mais proximas do embedding contrafactual para inferir
o desfecho provavel.

\textbf{Por que isso importa:} simulacoes contrafactuais sao \emph{impossiveis} com um
twin estatico. O twin temporal, por codificar o perfil operacional de cada hospital
no espaco de embeddings, permite ``transportar'' um paciente geometricamente entre
hospitais e observar o que muda.
""")

    if top_simulations:
        for idx, sim in enumerate(top_simulations):
            L.append(r"\needspace{10\baselineskip}")
            L.append(r"\subsection{Contrafactual " + str(idx + 1) + r": " +
                     _escape_latex(str(sim['h_origin'])[:20]) + r" $\rightarrow$ " +
                     _escape_latex(str(sim['h_target'])[:20]) + r"}")

            L.append(r"\begin{counterbox}[Cenario Contrafactual]")
            L.append(
                r"\textbf{Paciente real:} Hospital \texttt{" + _escape_latex(str(sim['h_origin'])) + r"}, "
                r"Internacao " + _escape_latex(str(sim['eid'])) + r", "
                r"LOS real: " + str(sim['original_los']) + r" dias, "
                r"Desfecho real: " + _escape_latex(str(sim['original_tipo'])) + r", "
                r"CID: " + _escape_latex(_truncate(str(sim.get('original_cid', '---')), 40)) + r"\\[4pt]"
                r"\textbf{Simulacao:} E se estivesse em \texttt{" +
                _escape_latex(str(sim['h_target'])) + r"}?"
            )
            L.append(r"\end{counterbox}")
            L.append(r"\vspace{2pt}")

            # Real neighbors
            L.append(r"{\footnotesize\textbf{Vizinhos do embedding real:}}")
            if sim['nn_original']:
                L.append(r"\begin{center}{\scriptsize")
                L.append(r"\begin{tabular}{rlrrll}")
                L.append(r"\toprule \textbf{\#} & \textbf{Hospital} & \textbf{LOS} & \textbf{Sim.} & \textbf{Desfecho} & \textbf{CID} \\\midrule")
                for ni, n in enumerate(sim['nn_original'], 1):
                    tc = "trafficred" if n.get("tipo_alta") == "OBITO" else "jcubeblue"
                    L.append(
                        str(ni) + r" & " +
                        _escape_latex(str(n.get('src', '?'))[:12]) + r" & " +
                        str(n.get('los', 0)) + r" & " +
                        f"{n.get('sim', 0):.4f}" + r" & " +
                        r"\textcolor{" + tc + r"}{" + _escape_latex(str(n.get('tipo_alta', '?'))) + r"} & " +
                        _escape_latex(_truncate(str(n.get('cid', '---')), 25)) + r" \\"
                    )
                L.append(r"\bottomrule\end{tabular}}")
                L.append(r"\end{center}")

            # Counterfactual neighbors
            L.append(r"{\footnotesize\textbf{Vizinhos do embedding contrafactual:}}")
            if sim['nn_counter']:
                L.append(r"\begin{center}{\scriptsize")
                L.append(r"\begin{tabular}{rlrrll}")
                L.append(r"\toprule \textbf{\#} & \textbf{Hospital} & \textbf{LOS} & \textbf{Sim.} & \textbf{Desfecho} & \textbf{CID} \\\midrule")
                for ni, n in enumerate(sim['nn_counter'], 1):
                    tc = "trafficred" if n.get("tipo_alta") == "OBITO" else "jcubeblue"
                    L.append(
                        str(ni) + r" & " +
                        _escape_latex(str(n.get('src', '?'))[:12]) + r" & " +
                        str(n.get('los', 0)) + r" & " +
                        f"{n.get('sim', 0):.4f}" + r" & " +
                        r"\textcolor{" + tc + r"}{" + _escape_latex(str(n.get('tipo_alta', '?'))) + r"} & " +
                        _escape_latex(_truncate(str(n.get('cid', '---')), 25)) + r" \\"
                    )
                L.append(r"\bottomrule\end{tabular}}")
                L.append(r"\end{center}")

            # Interpretation
            orig_tipos = [n.get("tipo_alta", "?") for n in sim['nn_original']]
            cf_tipos   = [n.get("tipo_alta", "?") for n in sim['nn_counter']]
            orig_los   = [n.get("los", 0) for n in sim['nn_original'] if n.get("los", 0) > 0]
            cf_los     = [n.get("los", 0) for n in sim['nn_counter'] if n.get("los", 0) > 0]

            L.append(r"\begin{insightbox}[Analise Contrafactual]")
            interp_parts = []
            if orig_los and cf_los:
                avg_orig = np.mean(orig_los)
                avg_cf = np.mean(cf_los)
                diff = avg_cf - avg_orig
                if abs(diff) > 1:
                    direction = "maior" if diff > 0 else "menor"
                    interp_parts.append(
                        f"LOS medio dos vizinhos contrafactuais: {avg_cf:.0f} dias "
                        f"(vs {avg_orig:.0f} dias no cenario real --- {direction})"
                    )

            obito_orig = orig_tipos.count("OBITO")
            obito_cf   = cf_tipos.count("OBITO")
            if obito_orig != obito_cf:
                if obito_cf > obito_orig:
                    interp_parts.append(
                        r"\textcolor{trafficred}{\textbf{RISCO:}} o cenario contrafactual "
                        f"apresenta {obito_cf} vizinhos com obito (vs {obito_orig} no real)"
                    )
                else:
                    interp_parts.append(
                        r"\textcolor{trafficgreen}{\textbf{BENEFICIO:}} o cenario contrafactual "
                        f"apresenta {obito_cf} vizinhos com obito (vs {obito_orig} no real)"
                    )

            if interp_parts:
                L.append(". ".join(interp_parts) + ".")
            else:
                L.append("Os cenarios real e contrafactual apresentam resultados similares.")
            L.append(r"\end{insightbox}")
            L.append(r"\vspace{6pt}")
    else:
        L.append(r"Nao foi possivel gerar simulacoes contrafactuais com os hospitais disponiveis.")

    L.append(r"\clearpage")

    # ═══════════════════════════════════════════════════════════════
    # SECTION 5: Clustering Temporal de Perfis de Evolucao
    # ═══════════════════════════════════════════════════════════════

    L.append(r"\section{Clustering Temporal de Perfis de Evolucao}")
    L.append(r"""
Aplicamos K-means ($k=""" + str(len(cluster_profiles)) + r"""$) sobre os \textbf{vetores de trajetoria}
$\vec{\delta}$ (nao sobre os embeddings brutos). Cada cluster representa um
\textbf{arquetipo de evolucao temporal}: pacientes cujas trajetorias no espaco latente
seguem padroes similares.

\textbf{Por que isso importa:} clustering em embeddings estaticos agrupa pacientes que
``se parecem agora''. Clustering em deltas agrupa pacientes que ``mudam da mesma forma'' ---
revelando arquetipos de evolucao que transcendem diagnosticos e hospitais individuais.
""")

    # Summary table of all clusters
    L.append(r"\subsection{Resumo dos Arquetipos}")
    L.append(r"\begin{center}{\footnotesize")
    L.append(r"\begin{tabular}{rrlrrr}")
    L.append(r"\toprule \textbf{Cluster} & \textbf{N Pac.} & \textbf{Arquetipo} & \textbf{$\|\vec{\delta}\|$ Med.} & \textbf{Desfecho Dom.} & \textbf{\% Dom.} \\\midrule")

    cluster_colors = ["clusterA", "clusterB", "clusterC", "clusterD", "clusterE",
                      "jcubeblue", "trafficgreen", "trafficyellow", "trafficred", "jcubegray"]

    for rank, (cl, prof) in enumerate(sorted_clusters):
        color = cluster_colors[rank % len(cluster_colors)]
        L.append(
            r"\textcolor{" + color + r"}{\textbf{" + str(cl) + r"}} & " +
            f"{prof['n_patients']:,}" + r" & " +
            _escape_latex(prof.get('archetype', '?')[:30]) + r" & " +
            f"{prof['mean_speed']:.4f}" + r" & " +
            _escape_latex(prof['dominant_cat']) + r" & " +
            f"{prof['cat_pct']:.0f}\\%" + r" \\"
        )
    L.append(r"\bottomrule\end{tabular}}")
    L.append(r"\end{center}")

    # Detailed profiles for top 5 largest clusters
    L.append(r"\subsection{Detalhamento dos 5 Maiores Arquetipos}")

    for rank, (cl, prof) in enumerate(sorted_clusters[:5]):
        color = cluster_colors[rank % len(cluster_colors)]
        L.append(r"\needspace{10\baselineskip}")
        L.append(
            r"\subsubsection{Cluster " + str(cl) + r": " +
            _escape_latex(prof.get('archetype', 'Perfil misto')) + r"}"
        )

        L.append(r"\begin{sectioncard}[Arquetipo: " +
                 _escape_latex(prof.get('archetype', '?')) + r"]{" + color + r"}")

        L.append(
            r"\textbf{Pacientes:} " + f"{prof['n_patients']:,}" + r" \quad "
            r"\textbf{Velocidade media:} " + f"{prof['mean_speed']:.4f}" + r" $\pm$ " +
            f"{prof['std_speed']:.4f}"
        )

        L.append(r"\\[4pt]\textbf{Distribuicao de desfechos:}")
        L.append(r"\begin{itemize}[nosep,leftmargin=*]")
        for cat, count in sorted(prof['cat_counts'].items(), key=lambda x: -x[1]):
            pct = count / max(prof['n_patients'], 1) * 100
            L.append(
                r"\item " + _escape_latex(cat) + r": " +
                f"{count:,}" + r" (" + f"{pct:.1f}" + r"\%)"
            )
        L.append(r"\end{itemize}")

        if prof.get('top_cids'):
            L.append(r"\\[2pt]\textbf{CIDs mais frequentes:}")
            L.append(r"\begin{itemize}[nosep,leftmargin=*]")
            for cid_desc, cid_count in prof['top_cids']:
                L.append(
                    r"\item " + _escape_latex(_truncate(cid_desc, 60)) +
                    r" (" + f"{cid_count:,}" + r")"
                )
            L.append(r"\end{itemize}")

        L.append(r"\\[4pt]\textbf{Interpretacao narrativa:} " +
                 _escape_latex(prof.get('archetype_desc', '')))

        L.append(r"\end{sectioncard}")
        L.append(r"\vspace{6pt}")

    L.append(r"\clearpage")

    # ═══════════════════════════════════════════════════════════════
    # Methodology Appendix
    # ═══════════════════════════════════════════════════════════════

    L.append(r"\section*{Apendice: Metodologia}")
    L.append(r"\addcontentsline{toc}{section}{Apendice: Metodologia}")

    L.append(r"\subsection*{1. Modelo de Embeddings}")
    L.append(r"""
\begin{itemize}[nosep]
  \item \textbf{Arquitetura:} TGN (Temporal Graph Network) V6.2 com Dense Temporal JEPA
  \item \textbf{Dimensao:} """ + str(emb_dim) + r""" dimensoes por no
  \item \textbf{Nos:} """ + f"{n_embeddings:,}" + r""" entidades (internacoes, pacientes, CIDs, hospitais)
  \item \textbf{Treinamento:} 4$\times$ H100 GPU, epoch 5, Weak-SIGReg + dense lookahead
  \item \textbf{Regularizacao:} Weak-SIGReg para evitar colapso dimensional
\end{itemize}
""")

    L.append(r"\subsection*{2. Conceito: Twin Temporal vs Twin Estatico}")
    L.append(r"""
Um \textbf{gemeo digital estatico} captura o estado atual de cada entidade como um ponto
fixo no espaco de embeddings. Um \textbf{gemeo digital temporal} codifica a
\emph{historia de mudancas} de cada entidade ao longo do tempo. A diferenca fundamental:
\begin{itemize}[nosep]
  \item \textbf{Estatico:} ``Como este paciente e agora?'' $\rightarrow$ embedding unico
  \item \textbf{Temporal:} ``Como este paciente \emph{evoluiu}?'' $\rightarrow$ sequencia de embeddings
  \item O \textbf{vetor de trajetoria} $\vec{\delta}$ captura a direcao e magnitude da mudanca
\end{itemize}
""")

    L.append(r"\subsection*{3. Fontes de Dados}")
    L.append(r"""
\begin{itemize}[nosep]
  \item Embeddings: \texttt{/cache/tkg-v6.2/node\_embeddings.pt}
  \item Grafo: \texttt{/data/jcube\_graph.parquet}
  \item Base relacional: \texttt{/data/aggregated\_fixed\_union.db} (DuckDB)
\end{itemize}
""")

    L.append(r"\subsection*{4. Analises Temporais}")
    L.append(r"""
\begin{itemize}[nosep]
  \item \textbf{Secao 1 --- Velocidade:} $\|\vec{\delta}\| = \|\text{emb}(t_n) - \text{emb}(t_{n-1})\|$ para pacientes com 2+ internacoes
  \item \textbf{Secao 2 --- Analogia:} KNN ($k=5$) em embeddings de internacoes concluidas para prognostico
  \item \textbf{Secao 3 --- Anomalia:} $\cos(\vec{\delta}_{\text{paciente}}, \overline{\vec{\delta}}_{\text{CID}})$ para detectar desvios da trajetoria esperada
  \item \textbf{Secao 4 --- Contrafactual:} $\vec{e}_{\text{cf}} = \vec{e} - \vec{c}_{\text{origem}} + \vec{c}_{\text{destino}}$ para simulacao de transferencia hospitalar
  \item \textbf{Secao 5 --- Clustering:} K-means ($k=""" + str(len(cluster_profiles)) + r"""$) em $\vec{\delta}$ para identificar arquetipos de evolucao
\end{itemize}
""")

    L.append(r"\subsection*{5. Limitacoes}")
    L.append(r"""
\begin{itemize}[nosep]
  \item Os embeddings capturam correlacoes do grafo temporal, nao causalidade.
  \item Simulacoes contrafactuais sao aproximacoes lineares em espaco nao-linear.
  \item A deteccao de anomalias depende de volume suficiente por CID para estimar a media.
  \item O clustering de trajetorias assume que padroes lineares capturam a estrutura principal.
  \item Resultados devem ser validados clinicamente antes de uso em decisoes medicas.
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

    print("[7/7] Compiling PDF ...")
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
def generate_temporal_twin_report():
    import time
    import os
    import numpy as np

    t_start = time.time()
    print("=" * 70)
    print("JCUBE V6.2 Temporal Digital Twin Report Generator (Modal)")
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

    n_embeddings = embeddings.shape[0]
    emb_dim = embeddings.shape[1]

    # 2. Velocidade de Evolucao Clinica
    (speed_by_alta, top_fastest, top_slowest,
     stats_by_alta) = _trajectory_speed(unique_nodes, embeddings, node_to_idx)

    # 3. Previsao de Destino por Analogia Geometrica
    prognosis_results = _geometric_prognosis(
        unique_nodes, embeddings, node_to_idx, internacao_mask
    )

    # 4. Deteccao de Trajetoria Anomala
    top_anomalous, total_anomaly_scored = _anomalous_trajectories(
        unique_nodes, embeddings, node_to_idx
    )

    # 5. Simulacao Contrafactual Temporal
    (top_simulations, total_simulations,
     available_hospitals) = _counterfactual_simulation(
        unique_nodes, embeddings, node_to_idx, internacao_mask
    )

    # 6. Clustering Temporal de Perfis de Evolucao
    sorted_clusters, cluster_profiles = _trajectory_clustering(
        unique_nodes, embeddings, node_to_idx
    )

    # 7. Generate LaTeX + compile
    print("[7/7] Generating LaTeX document ...")
    latex = _generate_latex(
        stats_by_alta, top_fastest, top_slowest,
        prognosis_results,
        top_anomalous, total_anomaly_scored,
        top_simulations, total_simulations, available_hospitals,
        sorted_clusters, cluster_profiles,
        n_embeddings, emb_dim,
    )

    _compile_latex(latex, OUTPUT_PDF)

    # Commit so changes persist in the volume
    data_vol.commit()

    elapsed = time.time() - t_start
    print(f"\nFinished in {elapsed:.1f}s")
    print(f"Report saved to Modal volume jcube-data at: {OUTPUT_PDF}")
    print("Download with:")
    print(f"  modal volume get jcube-data reports/temporal_twin_v6.2_2026_03.pdf ./temporal_twin_v6.2_2026_03.pdf")
    return OUTPUT_PDF


@app.local_entrypoint()
def main():
    generate_temporal_twin_report.remote()
