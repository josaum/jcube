#!/usr/bin/env python3
"""
Modal script: JCUBE Admission-Discharge Analysis — v5
Runs on Modal, loads V5 embeddings from jepa-cache volume,
queries DuckDB from jcube-data volume.

Architecture: Supervised -> Extract -> Characterize

  Part 1 — Per-hospital LightGBM multiclass discharge prediction
      Target: ALTA_NORMAL / OBITO / ALTA_COMPLEXA / TRANSFERENCIA / EVASAO
      Features: 64-dim V5 embedding + tabular (FL_INTER_URGENCIA, day_of_week,
                month, prior_admission_count, convenio_encoded, cid_code_encoded)
      5-fold stratified CV, macro-AUC + per-class metrics
      Extract leaf indices: model.predict(X, pred_leaf=True)

  Part 2 — Leaf-based clustering (key innovation)
      Group admissions by most common leaf path across trees
      HDBSCAN on leaf index matrix (n_samples x n_trees)
      Minimum cluster size = 50, noise points labeled -1

  Part 3 — HDBSCAN on raw embeddings (comparison/fallback)
      HDBSCAN(min_cluster_size=100, min_samples=10) on 64-dim embeddings
      Natural density clusters, no forced k
      Noise points reported as "Nao agrupados (outliers)"

  Part 4 — Per-cluster characterization
      Discharge distribution, top CIDs, avg LOS, billing
      LightGBM feature importance per cluster
      Traffic-light: RED / YELLOW / GREEN
      Auto-named clusters from dominant characteristics

  Part 5 — Cross-hospital summary
      Embedding dims globally predictive of obito
      CIDs with highest obito rate
      Per-hospital model performance comparison

Discharge type from:
  - agg_tb_capta_evo_status_caes (FL_DESOSPITALIZACAO — last record)
  - agg_tb_capta_tipo_final_monit_fmon (description lookup)
  - Filter: IN_SITUACAO = 2 in agg_tb_capta_internacao_cain

Output:
  - LaTeX PDF at /data/reports/admission_discharge_report_v5_2026_03.pdf
    pt-BR, UTF-8, professional, one section per hospital

Usage:
    modal run --detach reports/admission_discharge_analysis.py
"""
from __future__ import annotations

import modal

# ─────────────────────────────────────────────────────────────────
# Modal App + Volumes
# ─────────────────────────────────────────────────────────────────

app = modal.App("jcube-admission-discharge")

jepa_cache = modal.Volume.from_name("jepa-cache", create_if_missing=False)
data_vol   = modal.Volume.from_name("jcube-data",  create_if_missing=False)

VOLUMES = {
    "/cache": jepa_cache,
    "/data":  data_vol,
}

# ─────────────────────────────────────────────────────────────────
# Container image — CPU + torch, duckdb, sklearn, lgbm, hdbscan, LaTeX
# ─────────────────────────────────────────────────────────────────

analysis_image = (
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
        "lightgbm>=4.3",
        "hdbscan>=0.8.33",
    )
)

# ─────────────────────────────────────────────────────────────────
# Paths inside the container
# ─────────────────────────────────────────────────────────────────

GRAPH_PARQUET = "/data/jcube_graph.parquet"
WEIGHTS_PATH  = "/cache/tkg-v5/node_emb_epoch_1.pt"
DB_PATH       = "/data/aggregated_fixed_union.db"
OUTPUT_DIR    = "/data/reports"
OUTPUT_PDF    = f"{OUTPUT_DIR}/admission_discharge_report_v5_2026_03.pdf"

REPORT_DATE_STR = "2026-03-23"

# Minimum discharged internacoes for a hospital to be included in Part 1
MIN_DISCHARGED = 100

# HDBSCAN parameters
HDBSCAN_MIN_CLUSTER_SIZE_LEAF  = 50
HDBSCAN_MIN_SAMPLES_LEAF       = 5
HDBSCAN_MIN_CLUSTER_SIZE_EMB   = 100
HDBSCAN_MIN_SAMPLES_EMB        = 10

# ─────────────────────────────────────────────────────────────────
# Discharge category mapping
# ─────────────────────────────────────────────────────────────────

DISCHARGE_CATEGORY_MAP: dict[str, str] = {
    "Alta hospitalar simples":                      "ALTA_NORMAL",
    "Alta hospitalar melhorada":                    "ALTA_NORMAL",
    "Monitoramento em andamento":                   "EM_CURSO",
    "Alta hospitalar complexa":                     "ALTA_COMPLEXA",
    "Obito":                                        "OBITO",
    "Transferencia para outra unidade":             "TRANSFERENCIA",
    "Transferencia para outra Unidade Hospitalar":  "TRANSFERENCIA",
    "Alta Administrativa":                          "ADMINISTRATIVO",
    "Finalizacao de pre-monitoramento":             "ADMINISTRATIVO",
    "Alta por Evasao":                              "EVASAO",
    "Alta RN para cadastro proprio":                "NEONATAL",
}

TARGET_CATEGORIES = ["ALTA_NORMAL", "OBITO", "ALTA_COMPLEXA", "TRANSFERENCIA", "EVASAO"]

CAT_TO_INT = {c: i for i, c in enumerate(TARGET_CATEGORIES)}
INT_TO_CAT = {i: c for c, i in CAT_TO_INT.items()}

TRAFFIC_LIGHT = {
    "OBITO":          "red",
    "EVASAO":         "red",
    "ALTA_COMPLEXA":  "yellow",
    "TRANSFERENCIA":  "yellow",
    "ALTA_NORMAL":    "green",
    "EM_CURSO":       "gray",
    "ADMINISTRATIVO": "gray",
    "NEONATAL":       "gray",
}

DIM_LABELS: dict[int, str] = {
    16: "billing pattern",
    28: "clinical complexity",
    30: "procedure volume",
    32: "stay duration signal",
    46: "temporal trajectory",
    53: "glosa risk",
    61: "operational pattern",
}


# ─────────────────────────────────────────────────────────────────
# Shared helpers
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


def _pct_plain(num: float, den: float, decimals: int = 1) -> str:
    if den == 0:
        return "---"
    return f"{100.0 * num / den:.{decimals}f}%"


def _brl_plain(v) -> str:
    f = _safe_float(v)
    if f == 0:
        return "---"
    return "R$ {:,.2f}".format(f).replace(",", "X").replace(".", ",").replace("X", ".")


def _exec(con, q: str):
    cur  = con.execute(q)
    cols = [d[0] for d in cur.description]
    return cols, cur.fetchall()


def _rows_to_dicts(cols, rows):
    return [dict(zip(cols, r)) for r in rows]


def _truncate(s: str, max_len: int = 80) -> str:
    if not s:
        return "---"
    s = str(s).strip()
    return (s[:max_len] + "...") if len(s) > max_len else s


def _dim_label(dim_n: int) -> str:
    return DIM_LABELS.get(dim_n, f"dim{dim_n}")


def _normalize_ds(ds: str) -> str:
    import unicodedata
    s = str(ds or "").strip()
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    return s


def _map_category(ds: str) -> str:
    norm = _normalize_ds(ds)
    for key, cat in DISCHARGE_CATEGORY_MAP.items():
        if _normalize_ds(key) == norm:
            return cat
    nl = norm.lower()
    if "obito" in nl or "obit" in nl:
        return "OBITO"
    if "evasao" in nl or "evasion" in nl:
        return "EVASAO"
    if "complexa" in nl:
        return "ALTA_COMPLEXA"
    if "transferencia" in nl or "transfer" in nl:
        return "TRANSFERENCIA"
    if "andamento" in nl or "monitoramento" in nl:
        return "EM_CURSO"
    if "administrativa" in nl or "pre-monitoramento" in nl or "premonitoramento" in nl or "finalizacao" in nl or "finalizac" in nl:
        return "ADMINISTRATIVO"
    if "neonatal" in nl or " rn " in nl or "cadastro proprio" in nl or "recem" in nl:
        return "NEONATAL"
    if "simples" in nl or "melhorada" in nl or "alta hosp" in nl:
        return "ALTA_NORMAL"
    return "OUTRO"


def _traffic_light(obito_rate: float, evasao_rate: float, complexa_rate: float) -> str:
    if obito_rate > 5 or evasao_rate > 3:
        return "RED"
    if complexa_rate > 10:
        return "YELLOW"
    return "GREEN"


def _cluster_stats(records_subset: list[dict]) -> dict:
    """Compute discharge rates, avg LOS, top CIDs for a group of records."""
    import numpy as np

    n = len(records_subset)
    if n == 0:
        return {"n_total": 0, "obito_rate": 0.0, "evasao_rate": 0.0,
                "complexa_rate": 0.0, "avg_los": 0.0, "cat_counts": {},
                "top_cids": [], "traffic": "GREEN"}

    cat_counts: dict[str, int] = {}
    cid_counts: dict[str, int] = {}
    los_vals = []

    for r in records_subset:
        cat = r.get("discharge_category", "OUTRO")
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
        cid = str(r.get("cid_code", "ND") or "ND")
        cid_counts[cid] = cid_counts.get(cid, 0) + 1
        los = r.get("los_days")
        if los is not None:
            los_vals.append(_safe_float(los))

    n_em_curso    = cat_counts.get("EM_CURSO", 0)
    n_discharged  = n - n_em_curso
    n_obito       = cat_counts.get("OBITO", 0)
    n_evasao      = cat_counts.get("EVASAO", 0)
    n_complexa    = cat_counts.get("ALTA_COMPLEXA", 0)

    obito_rate    = (n_obito   / n_discharged * 100) if n_discharged > 0 else 0.0
    evasao_rate   = (n_evasao  / n_discharged * 100) if n_discharged > 0 else 0.0
    complexa_rate = (n_complexa / n_discharged * 100) if n_discharged > 0 else 0.0
    avg_los       = float(np.mean(los_vals)) if los_vals else 0.0

    top_cids = sorted(cid_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "n_total":       n,
        "n_discharged":  n_discharged,
        "cat_counts":    cat_counts,
        "obito_rate":    obito_rate,
        "evasao_rate":   evasao_rate,
        "complexa_rate": complexa_rate,
        "avg_los":       avg_los,
        "top_cids":      top_cids,
        "traffic":       _traffic_light(obito_rate, evasao_rate, complexa_rate),
    }


# ─────────────────────────────────────────────────────────────────
# Step 1 — Load embeddings, index INTERNACAO nodes
# ─────────────────────────────────────────────────────────────────

def _load_internacao_embeddings():
    """
    Returns:
        unique_nodes : np.ndarray of node name strings
        embeddings   : np.ndarray shape (N, 64) float32
        node_to_idx  : dict  node_name -> row_index
        intern_mask  : bool array — True where node is an INTERNACAO node
    """
    import time
    import numpy as np
    import torch
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    import pyarrow as pa

    print("[1/7] Loading node vocabulary from graph parquet ...")
    t0 = time.time()
    table        = pq.read_table(GRAPH_PARQUET, columns=["subject_id", "object_id"])
    subj         = table.column("subject_id")
    obj          = table.column("object_id")
    all_nodes    = pa.chunked_array(subj.chunks + obj.chunks)
    unique_nodes = pc.unique(all_nodes).to_numpy(zero_copy_only=False).astype(object)
    del table, subj, obj, all_nodes
    print(f"    {len(unique_nodes):,} unique nodes  ({time.time()-t0:.1f}s)")

    print("[1/7] Loading V5 embeddings ...")
    t1    = time.time()
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

    node_to_idx = {str(n): i for i, n in enumerate(unique_nodes)}
    intern_mask = np.array(
        ["/ID_CD_INTERNACAO_" in str(n) for n in unique_nodes], dtype=bool
    )
    print(f"    INTERNACAO nodes: {intern_mask.sum():,}")
    return unique_nodes, embeddings, node_to_idx, intern_mask


# ─────────────────────────────────────────────────────────────────
# Step 2 — Build discharge type table from DuckDB
# ─────────────────────────────────────────────────────────────────

def _build_discharge_table(con) -> list[dict]:
    """
    Returns one dict per internacao with:
      source_db, ID_CD_INTERNACAO, ID_CD_PACIENTE,
      tipo_alta, discharge_category,
      fl_urgencia, id_convenio,
      admission_dow, admission_month, admission_year,
      los_days, prior_admissions, cid_code, cid_desc

    Filters: IN_SITUACAO = 2 (discharged).
    Uses LAST status record per internacao (ORDER BY DH_CADASTRO DESC).
    """
    import time
    print("[2/7] Building discharge table from DuckDB ...")
    t0 = time.time()

    # Last discharge type per internacao
    discharge_q = """
        WITH last_status AS (
            SELECT
                es.ID_CD_INTERNACAO,
                es.source_db,
                es.FL_DESOSPITALIZACAO,
                ROW_NUMBER() OVER (
                    PARTITION BY es.ID_CD_INTERNACAO, es.source_db
                    ORDER BY es.DH_CADASTRO DESC
                ) AS rn
            FROM agg_tb_capta_evo_status_caes es
        )
        SELECT
            ls.ID_CD_INTERNACAO,
            ls.source_db,
            COALESCE(f.DS_FINAL_MONITORAMENTO, 'ND') AS tipo_alta
        FROM last_status ls
        LEFT JOIN agg_tb_capta_tipo_final_monit_fmon f
            ON ls.FL_DESOSPITALIZACAO = f.ID_CD_FINAL_MONITORAMENTO
        WHERE ls.rn = 1
    """
    try:
        dc_cols, dc_rows = _exec(con, discharge_q)
        discharge_map: dict[tuple, str] = {}
        for r in _rows_to_dicts(dc_cols, dc_rows):
            key = (r["ID_CD_INTERNACAO"], r["source_db"])
            discharge_map[key] = str(r.get("tipo_alta") or "ND")
        print(f"    Discharge map size: {len(discharge_map):,}  ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"    ERROR building discharge map: {e}")
        discharge_map = {}

    # Base internacoes — filter IN_SITUACAO = 2 (discharged)
    base_q = """
        SELECT
            i.ID_CD_INTERNACAO,
            i.ID_CD_PACIENTE,
            i.source_db,
            COALESCE(i.FL_INTER_URGENCIA, 0)               AS fl_urgencia,
            COALESCE(i.ID_CD_CONVENIO, -1)                 AS id_convenio,
            DAYOFWEEK(TRY_CAST(i.DH_ADMISSAO_HOSP AS DATE)) AS admission_dow,
            MONTH(TRY_CAST(i.DH_ADMISSAO_HOSP AS DATE))     AS admission_month,
            YEAR(TRY_CAST(i.DH_ADMISSAO_HOSP AS DATE))      AS admission_year,
            CASE
                WHEN i.DH_FINALIZACAO IS NOT NULL
                 AND i.DH_FINALIZACAO > '2000-01-01'
                THEN DATEDIFF('day',
                    TRY_CAST(i.DH_ADMISSAO_HOSP AS DATE),
                    TRY_CAST(i.DH_FINALIZACAO   AS DATE))
                ELSE NULL
            END AS los_days
        FROM agg_tb_capta_internacao_cain i
        WHERE i.IN_SITUACAO = 2
          AND i.DH_ADMISSAO_HOSP IS NOT NULL
    """
    try:
        cols, rows = _exec(con, base_q)
        records    = _rows_to_dicts(cols, rows)
        print(f"    Base internacoes (IN_SITUACAO=2): {len(records):,}  ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"    ERROR querying base internacoes: {e}")
        raise

    # Attach discharge category
    for rec in records:
        key               = (rec["ID_CD_INTERNACAO"], rec["source_db"])
        tipo              = discharge_map.get(key, "SEM_STATUS")
        rec["tipo_alta"]          = tipo
        rec["discharge_category"] = _map_category(tipo)

    # CIDs
    print("    Fetching admission CIDs ...")
    cid_map: dict[tuple, tuple] = {}
    try:
        cid_cols, cid_rows = _exec(con, """
            SELECT
                c.ID_CD_INTERNACAO,
                c.source_db,
                FIRST(c.ID_CD_CID   ORDER BY c.ID_CD_CID) AS cid_code,
                FIRST(c.DS_DESCRICAO ORDER BY c.ID_CD_CID) AS cid_desc
            FROM agg_tb_capta_cid_caci c
            WHERE c.ID_CD_CID IS NOT NULL
            GROUP BY c.ID_CD_INTERNACAO, c.source_db
        """)
        for r in _rows_to_dicts(cid_cols, cid_rows):
            key          = (r["ID_CD_INTERNACAO"], r["source_db"])
            cid_map[key] = (r.get("cid_code") or "ND", r.get("cid_desc") or "ND")
        print(f"    CID map size: {len(cid_map):,}")
    except Exception as e:
        print(f"    CID fetch skipped: {e}")

    # Prior admissions count
    print("    Computing prior admission counts ...")
    prior_map: dict[tuple, int] = {}
    try:
        prior_cols, prior_rows = _exec(con, """
            SELECT
                ID_CD_INTERNACAO,
                source_db,
                (ROW_NUMBER() OVER (
                    PARTITION BY ID_CD_PACIENTE, source_db
                    ORDER BY DH_ADMISSAO_HOSP
                ) - 1) AS prior_admissions
            FROM agg_tb_capta_internacao_cain
            WHERE DH_ADMISSAO_HOSP IS NOT NULL
        """)
        for r in _rows_to_dicts(prior_cols, prior_rows):
            key            = (r["ID_CD_INTERNACAO"], r["source_db"])
            prior_map[key] = _safe_int(r.get("prior_admissions"))
        print(f"    Prior admissions map: {len(prior_map):,}")
    except Exception as e:
        print(f"    Prior admissions skipped: {e}")

    for rec in records:
        key                     = (rec["ID_CD_INTERNACAO"], rec["source_db"])
        cid_code, cid_desc      = cid_map.get(key, ("ND", "ND"))
        rec["cid_code"]         = cid_code
        rec["cid_desc"]         = cid_desc
        rec["prior_admissions"] = prior_map.get(key, 0)

    elapsed = time.time() - t0
    print(f"    Feature table built: {len(records):,} rows  ({elapsed:.1f}s)")
    return records


# ─────────────────────────────────────────────────────────────────
# Step 3 — Attach embeddings to feature records
# ─────────────────────────────────────────────────────────────────

def _attach_embeddings(
    records: list[dict],
    embeddings,
    node_to_idx: dict,
) -> tuple[list[dict], int]:
    import numpy as np

    print("[3/7] Attaching embeddings to feature records ...")
    matched = []
    missing = 0
    for rec in records:
        key = f"{rec['source_db']}/ID_CD_INTERNACAO_{rec['ID_CD_INTERNACAO']}"
        if key in node_to_idx:
            idx              = node_to_idx[key]
            rec["embedding"] = embeddings[idx].tolist()
            matched.append(rec)
        else:
            missing += 1

    print(f"    Matched: {len(matched):,}  Missing: {missing:,}")
    return matched, missing


# ─────────────────────────────────────────────────────────────────
# Step 4 — Part 1: Per-hospital LightGBM + leaf extraction
# ─────────────────────────────────────────────────────────────────

def _base_feature_names() -> list[str]:
    return (
        [f"emb_{i}" for i in range(64)]
        + [
            "fl_urgencia",
            "admission_dow",
            "admission_month",
            "admission_year",
            "id_convenio_enc",
            "cid_code_enc",
            "prior_admissions",
        ]
    )


def _build_encoders(records: list[dict]) -> tuple[dict, dict]:
    all_convenios = sorted({str(r.get("id_convenio", -1)) for r in records})
    all_cids      = sorted({str(r.get("cid_code", "ND"))   for r in records})
    return (
        {v: i for i, v in enumerate(all_convenios)},
        {v: i for i, v in enumerate(all_cids)},
    )


def _encode_row(rec: dict, conv_enc: dict, cid_enc: dict) -> list[float]:
    emb = rec["embedding"]
    return emb + [
        float(_safe_int(rec.get("fl_urgencia", 0))),
        float(_safe_int(rec.get("admission_dow", 0))),
        float(_safe_int(rec.get("admission_month", 0))),
        float(_safe_int(rec.get("admission_year", 0))),
        float(conv_enc.get(str(rec.get("id_convenio", -1)), 0)),
        float(cid_enc.get(str(rec.get("cid_code", "ND")), 0)),
        float(_safe_int(rec.get("prior_admissions", 0))),
    ]


def _run_lgbm_discharge(source_db: str, records: list[dict], n_folds: int = 5) -> dict:
    """
    LightGBM multiclass: predict discharge category.
    Extracts leaf indices from the final full-data model for Part 2 clustering.
    Returns result dict including 'leaf_matrix' (n_samples x n_trees) and
    'eligible_records' for downstream clustering.
    """
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, precision_score, recall_score
    import lightgbm as lgb

    eligible = [
        r for r in records
        if r.get("discharge_category") in TARGET_CATEGORIES
        and r.get("embedding") is not None
    ]

    if len(eligible) < MIN_DISCHARGED:
        return {
            "source_db": source_db,
            "error":     f"Only {len(eligible)} eligible rows (< {MIN_DISCHARGED})",
        }

    conv_enc, cid_enc = _build_encoders(eligible)
    feat_names        = _base_feature_names()

    X_rows, y_rows = [], []
    for rec in eligible:
        X_rows.append(_encode_row(rec, conv_enc, cid_enc))
        y_rows.append(CAT_TO_INT[rec["discharge_category"]])

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.int32)

    class_counts = {INT_TO_CAT[c]: int((y == c).sum()) for c in sorted(set(y))}
    print(f"    [{source_db}] class dist: {class_counts}")

    n_classes = len(set(y))
    if n_classes < 2:
        return {
            "source_db":    source_db,
            "error":        f"Only one class present: {class_counts}",
            "class_counts": class_counts,
        }

    min_cls      = min(class_counts.values())
    actual_folds = min(n_folds, min_cls)
    if actual_folds < 2:
        return {
            "source_db":    source_db,
            "error":        f"Smallest class has {min_cls} samples — need >= 2 for CV",
            "class_counts": class_counts,
        }

    skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)

    params = {
        "objective":         "multiclass",
        "num_class":         len(CAT_TO_INT),
        "n_estimators":      300,
        "learning_rate":     0.05,
        "num_leaves":        63,
        "min_child_samples": max(5, len(X) // 300),
        "subsample":         0.8,
        "colsample_bytree":  0.8,
        "reg_alpha":         0.1,
        "reg_lambda":        1.0,
        "class_weight":      "balanced",
        "n_jobs":            -1,
        "verbose":           -1,
        "random_state":      42,
    }

    all_y_true, all_y_pred, all_y_prob = [], [], []
    fold_importances = []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(30, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )
        preds = model.predict(X_val)
        probs = model.predict_proba(X_val)
        all_y_true.extend(y_val.tolist())
        all_y_pred.extend(preds.tolist())
        all_y_prob.extend(probs.tolist())
        fold_importances.append(model.feature_importances_.copy())

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_prob = np.array(all_y_prob)

    classes_present = sorted(set(all_y_true))
    try:
        auc = float(roc_auc_score(
            all_y_true,
            all_y_prob[:, :len(CAT_TO_INT)],
            multi_class="ovr",
            average="macro",
            labels=classes_present,
        ))
    except Exception:
        auc = float("nan")

    pr_labels = classes_present
    try:
        per_class_pr = {}
        prec = precision_score(all_y_true, all_y_pred, labels=pr_labels, average=None, zero_division=0)
        rec  = recall_score(   all_y_true, all_y_pred, labels=pr_labels, average=None, zero_division=0)
        for i, lbl in enumerate(pr_labels):
            per_class_pr[INT_TO_CAT[lbl]] = {
                "precision": float(prec[i]),
                "recall":    float(rec[i]),
            }
    except Exception:
        per_class_pr = {}

    mean_imp  = np.mean(fold_importances, axis=0)
    imp_pairs = sorted(
        zip(feat_names, mean_imp.tolist()),
        key=lambda x: x[1], reverse=True,
    )
    top10_all = imp_pairs[:10]
    top_emb   = [(n, float(v)) for n, v in imp_pairs if n.startswith("emb_")][:10]
    top_tab   = [(n, float(v)) for n, v in imp_pairs if not n.startswith("emb_")][:10]

    # Train final model on ALL data to extract leaf indices for clustering
    print(f"    [{source_db}] Training final model for leaf extraction ...")
    final_model = lgb.LGBMClassifier(**params)
    final_model.fit(X, y, callbacks=[lgb.log_evaluation(period=-1)])

    # Leaf indices: shape (n_samples, n_trees * n_classes) -> we want raw leaves
    try:
        leaf_matrix = final_model.predict(X, pred_leaf=True)  # (n_samples, n_trees)
        print(f"    [{source_db}] Leaf matrix shape: {leaf_matrix.shape}")
    except Exception as e:
        print(f"    [{source_db}] Leaf extraction failed: {e}")
        leaf_matrix = None

    narrative = _part1_narrative(source_db, top10_all, per_class_pr, class_counts)
    print(f"    [{source_db}] AUC macro={auc:.3f}")

    return {
        "source_db":        source_db,
        "n_samples":        int(len(X)),
        "class_counts":     class_counts,
        "auc_macro":        auc,
        "per_class_pr":     per_class_pr,
        "top10_all":        [(n, float(v)) for n, v in top10_all],
        "top_emb":          top_emb,
        "top_tab":          top_tab,
        "n_folds":          actual_folds,
        "narrative":        narrative,
        # For Part 2
        "leaf_matrix":      leaf_matrix,
        "eligible_records": eligible,
        "feature_names":    feat_names,
        "mean_importances": mean_imp.tolist(),
    }


def _part1_narrative(
    source_db: str,
    top10: list,
    per_class_pr: dict,
    class_counts: dict,
) -> str:
    top_names  = [n for n, _ in top10[:3]]
    top_labels = []
    for n in top_names:
        if n.startswith("emb_"):
            dim = int(n.split("_")[1])
            top_labels.append(f"embedding dim[{dim}] ({_dim_label(dim)})")
        elif n == "fl_urgencia":
            top_labels.append("urgencia flag")
        elif n == "id_convenio_enc":
            top_labels.append("convenio")
        elif n == "prior_admissions":
            top_labels.append("admissoes anteriores")
        elif n == "cid_code_enc":
            top_labels.append("CID de admissao")
        else:
            top_labels.append(n)

    feat_str  = ", ".join(top_labels[:3]) if top_labels else "embedding features"
    obito_pr  = per_class_pr.get("OBITO", {})
    obito_p   = obito_pr.get("precision", None)
    obito_r   = obito_pr.get("recall",    None)
    n_obito   = class_counts.get("OBITO", 0)
    total     = sum(class_counts.values())

    parts = [f"Em {source_db}, os principais preditores de desfecho sao: {feat_str}."]
    if n_obito > 0 and total > 0:
        obito_pct = 100.0 * n_obito / total
        parts.append(
            f"Obito representa {obito_pct:.1f}% das saidas "
            f"({n_obito} casos no conjunto de treino/validacao)."
        )
    if obito_p is not None and obito_r is not None:
        parts.append(
            f"O modelo identifica obito com precisao={obito_p:.2f} e "
            f"recall={obito_r:.2f} (macro-AUC reportado acima)."
        )
    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────
# Step 5 — Part 2: Leaf-based clustering (HDBSCAN on leaf indices)
# ─────────────────────────────────────────────────────────────────

def _run_leaf_clustering(p1_result: dict) -> dict:
    """
    Uses leaf index matrix from LightGBM to cluster admissions.
    Each sample's position in the leaf space reflects the exact decision
    path — samples in the same leaf group share the same logic.

    Strategy:
      1. HDBSCAN on leaf index matrix (treated as categorical distances)
      2. Fall back to most-common-leaf grouping if HDBSCAN produces too few clusters
    """
    import numpy as np
    import hdbscan

    source_db = p1_result["source_db"]

    if "error" in p1_result:
        return {"source_db": source_db, "error": p1_result["error"], "method": "leaf"}

    leaf_matrix = p1_result.get("leaf_matrix")
    eligible    = p1_result.get("eligible_records", [])

    if leaf_matrix is None or len(eligible) == 0:
        return {
            "source_db": source_db,
            "error":     "No leaf matrix available",
            "method":    "leaf",
        }

    n_samples = leaf_matrix.shape[0]
    if n_samples < HDBSCAN_MIN_CLUSTER_SIZE_LEAF * 2:
        return {
            "source_db": source_db,
            "error":     f"Too few samples for leaf clustering: {n_samples}",
            "method":    "leaf",
        }

    print(f"    [{source_db}] HDBSCAN on leaf matrix {leaf_matrix.shape} ...")

    # Normalize leaf indices per tree to [0,1] to balance tree contribution
    leaf_float = leaf_matrix.astype(np.float32)
    for col in range(leaf_float.shape[1]):
        col_max = leaf_float[:, col].max()
        if col_max > 0:
            leaf_float[:, col] /= col_max

    try:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE_LEAF,
            min_samples=HDBSCAN_MIN_SAMPLES_LEAF,
            metric="euclidean",
            core_dist_n_jobs=-1,
        )
        labels = clusterer.fit_predict(leaf_float)
    except Exception as e:
        print(f"    [{source_db}] HDBSCAN leaf failed: {e}")
        return {
            "source_db": source_db,
            "error":     f"HDBSCAN failed: {e}",
            "method":    "leaf",
        }

    unique_labels = set(labels)
    n_clusters    = len(unique_labels - {-1})
    n_noise       = int((labels == -1).sum())
    print(f"    [{source_db}] Leaf clusters: {n_clusters}, noise: {n_noise:,}")

    if n_clusters == 0:
        return {
            "source_db": source_db,
            "error":     "HDBSCAN found 0 clusters in leaf space",
            "method":    "leaf",
        }

    # Build cluster -> records mapping
    clusters: dict[int, list[dict]] = {lbl: [] for lbl in unique_labels if lbl != -1}
    noise_records: list[dict] = []

    for i, rec in enumerate(eligible):
        lbl = int(labels[i])
        if lbl == -1:
            noise_records.append(rec)
        else:
            clusters[lbl].append(rec)

    # Compute per-cluster stats
    cluster_stats_map: dict[int, dict] = {}
    for lbl, recs in clusters.items():
        stats = _cluster_stats(recs)
        stats["cluster_id"]  = lbl
        stats["cluster_type"] = "leaf"
        cluster_stats_map[lbl] = stats

    # Sort by criticality
    traffic_order = {"RED": 0, "YELLOW": 1, "GREEN": 2}
    sorted_clusters = sorted(
        cluster_stats_map.items(),
        key=lambda x: (
            traffic_order.get(x[1].get("traffic", "GREEN"), 2),
            -x[1].get("obito_rate", 0),
        ),
    )

    # Noise summary
    noise_stats = _cluster_stats(noise_records) if noise_records else {"n_total": 0}

    return {
        "source_db":       source_db,
        "method":          "leaf",
        "n_records":       n_samples,
        "n_clusters":      n_clusters,
        "n_noise":         n_noise,
        "sorted_clusters": sorted_clusters,
        "noise_stats":     noise_stats,
    }


# ─────────────────────────────────────────────────────────────────
# Step 6 — Part 3: HDBSCAN on raw embeddings (comparison/fallback)
# ─────────────────────────────────────────────────────────────────

def _run_embedding_clustering(source_db: str, records: list[dict]) -> dict:
    """
    HDBSCAN on raw 64-dim INTERNACAO embeddings.
    Finds natural density clusters without forcing a k.
    """
    import numpy as np
    import hdbscan

    emb_records = [r for r in records if r.get("embedding") is not None]

    if len(emb_records) < HDBSCAN_MIN_CLUSTER_SIZE_EMB * 2:
        return {
            "source_db": source_db,
            "method":    "embedding",
            "error":     f"Only {len(emb_records)} records with embeddings",
        }

    X = np.array([r["embedding"] for r in emb_records], dtype=np.float32)
    print(f"    [{source_db}] HDBSCAN on embedding matrix {X.shape} ...")

    try:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE_EMB,
            min_samples=HDBSCAN_MIN_SAMPLES_EMB,
            metric="euclidean",
            core_dist_n_jobs=-1,
        )
        labels = clusterer.fit_predict(X)
    except Exception as e:
        print(f"    [{source_db}] HDBSCAN embedding failed: {e}")
        return {
            "source_db": source_db,
            "method":    "embedding",
            "error":     f"HDBSCAN failed: {e}",
        }

    unique_labels = set(labels)
    n_clusters    = len(unique_labels - {-1})
    n_noise       = int((labels == -1).sum())
    print(f"    [{source_db}] Embedding clusters: {n_clusters}, noise: {n_noise:,}")

    if n_clusters == 0:
        return {
            "source_db": source_db,
            "method":    "embedding",
            "error":     "HDBSCAN found 0 clusters in embedding space",
        }

    clusters: dict[int, list[dict]] = {lbl: [] for lbl in unique_labels if lbl != -1}
    noise_records: list[dict] = []

    for i, rec in enumerate(emb_records):
        lbl = int(labels[i])
        if lbl == -1:
            noise_records.append(rec)
        else:
            clusters[lbl].append(rec)

    cluster_stats_map: dict[int, dict] = {}
    for lbl, recs in clusters.items():
        stats = _cluster_stats(recs)
        stats["cluster_id"]   = lbl
        stats["cluster_type"] = "embedding"
        cluster_stats_map[lbl] = stats

    traffic_order = {"RED": 0, "YELLOW": 1, "GREEN": 2}
    sorted_clusters = sorted(
        cluster_stats_map.items(),
        key=lambda x: (
            traffic_order.get(x[1].get("traffic", "GREEN"), 2),
            -x[1].get("obito_rate", 0),
        ),
    )

    noise_stats = _cluster_stats(noise_records) if noise_records else {"n_total": 0}

    return {
        "source_db":       source_db,
        "method":          "embedding",
        "n_records":       len(emb_records),
        "n_clusters":      n_clusters,
        "n_noise":         n_noise,
        "sorted_clusters": sorted_clusters,
        "noise_stats":     noise_stats,
    }


# ─────────────────────────────────────────────────────────────────
# Step 7 — Part 5: Cross-hospital summary
# ─────────────────────────────────────────────────────────────────

def _cross_hospital_summary(all_records: list[dict], p1_results: list[dict]) -> dict:
    import numpy as np
    print("[7/7] Computing cross-hospital summary ...")

    # Embedding dims vs obito
    obito_embs    = []
    nonobito_embs = []
    for r in all_records:
        if r.get("embedding") is None:
            continue
        if r.get("discharge_category") == "OBITO":
            obito_embs.append(r["embedding"])
        elif r.get("discharge_category") in TARGET_CATEGORIES:
            nonobito_embs.append(r["embedding"])

    dim_delta: list[tuple[int, float]] = []
    if obito_embs and nonobito_embs:
        O      = np.array(obito_embs,    dtype=np.float32)
        N      = np.array(nonobito_embs, dtype=np.float32)
        mean_o = O.mean(axis=0)
        mean_n = N.mean(axis=0)
        delta  = np.abs(mean_o - mean_n)
        dim_delta = sorted(enumerate(delta.tolist()), key=lambda x: x[1], reverse=True)[:10]

    # CIDs by obito rate
    cid_stats: dict[str, dict] = {}
    for r in all_records:
        cat = r.get("discharge_category")
        if cat not in TARGET_CATEGORIES:
            continue
        cid = r.get("cid_code", "ND") or "ND"
        if cid not in cid_stats:
            cid_stats[cid] = {"total": 0, "obito": 0, "desc": r.get("cid_desc", "ND")}
        cid_stats[cid]["total"] += 1
        if cat == "OBITO":
            cid_stats[cid]["obito"] += 1

    top_cids_obito = sorted(
        [
            (cid, v["total"], v["obito"], 100.0 * v["obito"] / v["total"], v["desc"])
            for cid, v in cid_stats.items()
            if v["total"] >= 30
        ],
        key=lambda x: x[3], reverse=True,
    )[:15]

    # Convenios by evasao rate
    conv_stats: dict[str, dict] = {}
    for r in all_records:
        cat = r.get("discharge_category")
        if cat not in TARGET_CATEGORIES:
            continue
        conv = str(r.get("id_convenio", -1))
        if conv not in conv_stats:
            conv_stats[conv] = {"total": 0, "evasao": 0}
        conv_stats[conv]["total"] += 1
        if cat == "EVASAO":
            conv_stats[conv]["evasao"] += 1

    top_convs_evasao = sorted(
        [
            (conv, v["total"], v["evasao"], 100.0 * v["evasao"] / v["total"])
            for conv, v in conv_stats.items()
            if v["total"] >= 50 and v["evasao"] > 0
        ],
        key=lambda x: x[3], reverse=True,
    )[:10]

    # Per-hospital model performance ranking
    hospital_ranking = sorted(
        [
            {
                "source_db": r["source_db"],
                "auc":       r.get("auc_macro", float("nan")),
                "n":         r.get("n_samples", 0),
                "error":     r.get("error", None),
            }
            for r in p1_results
        ],
        key=lambda x: (0 if x["error"] is None else 1, -x["auc"] if not x.get("error") else 0),
    )

    return {
        "dim_delta_top10":    dim_delta,
        "top_cids_obito":     top_cids_obito,
        "top_convs_evasao":   top_convs_evasao,
        "n_obito_total":      len(obito_embs),
        "n_nonobito_total":   len(nonobito_embs),
        "hospital_ranking":   hospital_ranking,
    }


# ─────────────────────────────────────────────────────────────────
# Print helpers (stdout)
# ─────────────────────────────────────────────────────────────────

def _print_part1(results: list[dict]) -> None:
    print("\n" + "=" * 70)
    print("PARTE 1 — PREVISAO DE DESFECHO POR HOSPITAL (LightGBM)")
    print("=" * 70)
    for r in results:
        src = r["source_db"]
        if "error" in r:
            print(f"\n  [{src}] ERRO: {r['error']}")
            continue
        print(f"\n  [{src}]  n={r['n_samples']:,}  macro-AUC={r['auc_macro']:.3f}  folds={r['n_folds']}")
        print(f"  Distribuicao de classes:")
        for cat, cnt in sorted(r["class_counts"].items()):
            print(f"    {cat:20s}: {cnt:,}")
        print(f"  Por classe (precision / recall):")
        for cat, pr in r.get("per_class_pr", {}).items():
            print(f"    {cat:20s}: P={pr['precision']:.2f}  R={pr['recall']:.2f}")
        print(f"  Top-10 features:")
        for feat, imp in r.get("top10_all", [])[:10]:
            label = _dim_label(int(feat.split("_")[1])) if feat.startswith("emb_") else feat
            print(f"    {feat:30s} ({label:30s}): {imp:.1f}")
        print(f"  Narrativa: {r.get('narrative', '')}")


def _print_cluster_result(cr: dict, part_label: str) -> None:
    src = cr["source_db"]
    if "error" in cr:
        print(f"\n  [{src}] ERRO ({part_label}): {cr['error']}")
        return
    method = cr.get("method", "?")
    print(f"\n  [{src}] {part_label} ({method}): {cr['n_records']:,} internacoes, "
          f"{cr['n_clusters']} clusters, {cr['n_noise']:,} outliers")
    for c_id, cd in cr["sorted_clusters"][:5]:
        n        = cd.get("n_total", 0)
        traffic  = cd.get("traffic", "GREEN")
        obito    = cd.get("obito_rate", 0)
        evasao   = cd.get("evasao_rate", 0)
        complexa = cd.get("complexa_rate", 0)
        avg_los  = cd.get("avg_los", 0)
        top_cids = cd.get("top_cids", [])[:3]
        print(
            f"    Cluster {c_id:3d} [{traffic:6s}] n={n:5,}  "
            f"obito={obito:.1f}%  evasao={evasao:.1f}%  "
            f"complexa={complexa:.1f}%  avg_LOS={avg_los:.1f}d"
        )
        if top_cids:
            cids_str = ", ".join(f"{cid}({cnt})" for cid, cnt in top_cids)
            print(f"             Top CIDs: {cids_str}")


def _print_part5(summary: dict) -> None:
    print("\n" + "=" * 70)
    print("PARTE 5 — RESUMO CRUZADO ENTRE HOSPITAIS")
    print("=" * 70)

    print("\n  Ranking de hospitais por AUC do modelo:")
    for h in summary.get("hospital_ranking", []):
        if h.get("error"):
            print(f"    {h['source_db']:30s} ERRO: {h['error']}")
        else:
            print(f"    {h['source_db']:30s} AUC={h['auc']:.3f}  n={h['n']:,}")

    print("\n  Dimensoes de embedding mais preditivas de obito (delta media global):")
    for dim, delta in summary.get("dim_delta_top10", []):
        print(f"    dim[{dim:2d}] ({_dim_label(dim):30s}): delta={delta:.4f}")

    print("\n  CIDs com maior taxa de obito (min 30 casos):")
    for cid, total, n_obito, rate, desc in summary.get("top_cids_obito", [])[:10]:
        print(f"    {cid:10s} {rate:5.1f}%  ({n_obito}/{total})  {_truncate(str(desc), 50)}")

    print("\n  Convenios com maior taxa de evasao (min 50 casos):")
    for conv, total, n_ev, rate in summary.get("top_convs_evasao", [])[:10]:
        print(f"    conv={conv:10s} {rate:5.1f}%  ({n_ev}/{total})")


# ─────────────────────────────────────────────────────────────────
# Cluster narrative helpers
# ─────────────────────────────────────────────────────────────────

def _cluster_narrative(c_id: int, cd: dict, source_db: str) -> str:
    n        = cd.get("n_total", 0)
    obito    = cd.get("obito_rate", 0)
    evasao   = cd.get("evasao_rate", 0)
    complexa = cd.get("complexa_rate", 0)
    avg_los  = cd.get("avg_los", 0)
    traffic  = cd.get("traffic", "GREEN")

    if traffic == "RED" and obito > evasao:
        return (
            f"Cluster {c_id} agrupa {n:,} internacoes com alta mortalidade ({obito:.1f}%). "
            f"LOS medio de {avg_los:.1f} dias. "
            f"Recomenda-se revisao do protocolo de atencao para os CIDs dominantes."
        )
    elif traffic == "RED" and evasao >= obito:
        return (
            f"Cluster {c_id} apresenta taxa de evasao elevada ({evasao:.1f}%) entre {n:,} internacoes. "
            f"Possiveis causas: insatisfacao do paciente, falta de cobertura do convenio, "
            f"ou problemas de comunicacao. LOS medio: {avg_los:.1f} dias."
        )
    elif traffic == "YELLOW":
        return (
            f"Cluster {c_id} tem {complexa:.1f}% de altas complexas em {n:,} internacoes "
            f"(LOS medio {avg_los:.1f} dias). "
            f"Monitorar necessidade de suporte pos-alta (home care, reabilitacao)."
        )
    else:
        return (
            f"Cluster {c_id}: {n:,} internacoes com perfil de baixo risco. "
            f"LOS medio de {avg_los:.1f} dias."
        )


def _auto_cluster_name(cd: dict) -> str:
    """Generate a short descriptive name for a cluster based on dominant characteristics."""
    obito    = cd.get("obito_rate", 0)
    evasao   = cd.get("evasao_rate", 0)
    complexa = cd.get("complexa_rate", 0)
    avg_los  = cd.get("avg_los", 0)
    top_cids = cd.get("top_cids", [])

    dominant_cid = top_cids[0][0] if top_cids else "ND"

    if obito > 15:
        return f"Alta Mortalidade ({dominant_cid})"
    if obito > 5:
        return f"Risco Elevado de Obito ({dominant_cid})"
    if evasao > 10:
        return f"Alta Taxa de Evasao"
    if evasao > 3:
        return f"Evasao Moderada ({dominant_cid})"
    if complexa > 20:
        return f"Alta Complexa Frequente ({dominant_cid})"
    if complexa > 10:
        return f"Complexidade Moderada ({dominant_cid})"
    if avg_los > 15:
        return f"Internacao Prolongada (LOS>{avg_los:.0f}d)"
    if avg_los > 7:
        return f"Internacao Media ({dominant_cid})"
    return f"Perfil Normal ({dominant_cid})"


# ─────────────────────────────────────────────────────────────────
# LaTeX / PDF generation
# ─────────────────────────────────────────────────────────────────

def _escape_latex(s: str) -> str:
    if not s:
        return ""
    s = str(s)
    s = s.replace("\\", "\\textbackslash{}")
    s = s.replace("&",  "\\&")
    s = s.replace("%",  "\\%")
    s = s.replace("$",  "\\$")
    s = s.replace("#",  "\\#")
    s = s.replace("_",  "\\_")
    s = s.replace("^",  "\\^{}")
    s = s.replace("{",  "\\{")
    s = s.replace("}",  "\\}")
    s = s.replace("\r\n", " ")
    s = s.replace("\n",   " ")
    s = s.replace("\r",   " ")
    s = s.replace("\t",   " ")
    return s


def _latex_color(traffic: str) -> str:
    return {
        "RED":    "red!20",
        "YELLOW": "yellow!30",
        "GREEN":  "green!15",
    }.get(traffic, "gray!10")


def _latex_cat_color(cat: str) -> str:
    return {
        "OBITO":          "red!30",
        "EVASAO":         "red!20",
        "ALTA_COMPLEXA":  "orange!30",
        "TRANSFERENCIA":  "yellow!30",
        "ALTA_NORMAL":    "green!20",
        "EM_CURSO":       "blue!10",
        "ADMINISTRATIVO": "gray!15",
        "NEONATAL":       "cyan!15",
        "OUTRO":          "gray!10",
        "SEM_STATUS":     "gray!5",
    }.get(cat, "gray!10")


def _latex_cluster_table(sorted_clusters: list, show_name: bool = True) -> list[str]:
    """Render a longtable of cluster stats. Returns list of lines."""
    lines = []
    lines.append(r"\begin{longtable}{clllrrrrr}")
    lines.append(r"\toprule")
    hdr = r"ID & Semaforo & Nome & N & Obito\% & Evasao\% & Complexa\% & LOS med & Top CID \\"
    lines.append(hdr)
    lines.append(r"\midrule")
    lines.append(r"\endhead")

    for c_id, cd in sorted_clusters:
        traffic  = cd.get("traffic", "GREEN")
        n        = cd.get("n_total", 0)
        obito    = cd.get("obito_rate", 0)
        evasao   = cd.get("evasao_rate", 0)
        complexa = cd.get("complexa_rate", 0)
        avg_los  = cd.get("avg_los", 0)
        top_cids = cd.get("top_cids", [])
        cid_str  = top_cids[0][0] if top_cids else "ND"
        name     = _auto_cluster_name(cd) if show_name else str(c_id)

        row_color = _latex_color(traffic)
        signal    = {
            "RED":    r"\textcolor{red}{\textbf{ALTO}}",
            "YELLOW": r"\textcolor{orange}{\textbf{MED}}",
            "GREEN":  r"\textcolor{green!60!black}{OK}",
        }.get(traffic, "---")

        lines.append(
            r"\rowcolor{" + row_color + r"}" +
            f"{c_id} & {signal} & {_escape_latex(name)} & {n:,} & {obito:.1f}\\% & "
            f"{evasao:.1f}\\% & {complexa:.1f}\\% & {avg_los:.1f}d & {_escape_latex(cid_str)} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{longtable}")
    return lines


def _latex_cluster_detail(sorted_clusters: list, source_db: str, max_risky: int = 5) -> list[str]:
    """Detail cards for the riskiest clusters."""
    lines = []
    risky = [
        (c_id, cd) for c_id, cd in sorted_clusters
        if cd.get("traffic") in ("RED", "YELLOW")
    ][:max_risky]

    if not risky:
        return lines

    lines.append(r"\vspace{4pt}\noindent\textbf{Clusters mais criticos (detalhamento):}")
    lines.append("")

    for c_id, cd in risky:
        traffic   = cd.get("traffic", "GREEN")
        n         = cd.get("n_total", 0)
        obito     = cd.get("obito_rate", 0)
        evasao    = cd.get("evasao_rate", 0)
        complexa  = cd.get("complexa_rate", 0)
        avg_los   = cd.get("avg_los", 0)
        top_cids  = cd.get("top_cids", [])[:5]
        cat_counts = cd.get("cat_counts", {})
        n_total_c  = cd.get("n_total", 1)
        row_color  = _latex_color(traffic)
        name       = _auto_cluster_name(cd)

        lines.append(r"\vspace{6pt}")
        lines.append(r"\noindent\colorbox{" + row_color + r"}{\parbox{\textwidth}{")
        lines.append(
            r"\textbf{Cluster " + str(c_id) + r" --- " +
            _escape_latex(name) + r"} \quad " +
            f"n={n:,} $\\cdot$ obito={obito:.1f}\\% $\\cdot$ evasao={evasao:.1f}\\% $\\cdot$ "
            f"complexa={complexa:.1f}\\% $\\cdot$ LOS={avg_los:.1f}d"
        )

        if top_cids:
            cids_tex = ", ".join(
                f"{_escape_latex(str(cid))} ({cnt})" for cid, cnt in top_cids
            )
            lines.append(r"\\ \textit{Top CIDs: " + cids_tex + r"}")

        # Discharge distribution
        cat_parts = []
        for cat in TARGET_CATEGORIES:
            cnt = cat_counts.get(cat, 0)
            if cnt > 0:
                cat_parts.append(
                    f"{_escape_latex(cat)}: {cnt} ({100.0*cnt/n_total_c:.1f}\\%)"
                )
        if cat_parts:
            lines.append(r"\\ \textit{Desfechos: " + "; ".join(cat_parts) + r"}")

        narr = _cluster_narrative(c_id, cd, source_db)
        lines.append(r"\\ " + _escape_latex(narr))
        lines.append(r"}}")
        lines.append("")

    return lines


def _gen_latex(
    p1_results:      list[dict],
    p2_leaf_results: list[dict],
    p3_emb_results:  list[dict],
    p5_summary:      dict,
    all_records:     list[dict],
) -> str:
    lines = []

    # Preamble
    lines.append(r"""\documentclass[11pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[brazil]{babel}
\usepackage{lmodern}
\usepackage{geometry}
\geometry{margin=2cm}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{colortbl}
\usepackage{xcolor}
\usepackage{parskip}
\usepackage{fancyhdr}
\usepackage{titlesec}
\usepackage{microtype}
\usepackage{hyperref}
\hypersetup{colorlinks=true,linkcolor=blue!60!black,urlcolor=blue}

\definecolor{redlight}{HTML}{FFCCCC}
\definecolor{yellowlight}{HTML}{FFF3CC}
\definecolor{greenlight}{HTML}{CCFFCC}
\definecolor{graylight}{HTML}{F0F0F0}

\pagestyle{fancy}
\fancyhf{}
\rhead{JCUBE -- Analise de Admissao e Alta Hospitalar}
\lhead{Confidencial}
\rfoot{\thepage}
\lfoot{""" + REPORT_DATE_STR + r"""}

\titleformat{\section}{\large\bfseries}{}{0em}{}[\titlerule]
\titleformat{\subsection}{\normalsize\bfseries}{}{0em}{}

\begin{document}
""")

    # Title
    lines.append(r"""
\begin{center}
{\LARGE \textbf{Analise de Admissao e Alta Hospitalar}}\\[4pt]
{\large JCUBE -- Embeddings V5 Graph-JEPA + LightGBM + HDBSCAN}\\[2pt]
{\normalsize """ + REPORT_DATE_STR + r"""}
\end{center}
\vspace{6pt}
\noindent\rule{\textwidth}{0.4pt}
\vspace{6pt}
""")

    # Executive summary
    n_total     = len(all_records)
    cat_global: dict[str, int] = {}
    for r in all_records:
        c = r.get("discharge_category", "OUTRO")
        cat_global[c] = cat_global.get(c, 0) + 1

    hospitals = sorted({r["source_db"] for r in all_records})

    lines.append(r"\section{Resumo Executivo}")
    lines.append(
        f"Este relatorio analisa {n_total:,} internacoes com alta confirmada "
        f"(\\texttt{{IN\\_SITUACAO=2}}) em {len(hospitals)} hospitais, "
        f"utilizando embeddings de 64 dimensoes treinados com Graph-JEPA V5 "
        f"(35,2M nos, epoch 1). "
        f"O tipo de alta real e obtido da ultima entrada em "
        r"\texttt{agg\_tb\_capta\_evo\_status\_caes}, "
        f"descrito via "
        r"\texttt{agg\_tb\_capta\_tipo\_final\_monit\_fmon}."
        f" A metodologia substitui K-Means por HDBSCAN sobre indices de folhas do "
        r"LightGBM (Part 2) e sobre embeddings brutos (Part 3), "
        r"eliminando clusters artificiais e ruido forcado."
    )
    lines.append("")

    # Hospital ranking table
    lines.append(r"\vspace{4pt}\noindent\textbf{Ranking de hospitais por AUC do modelo:}")
    lines.append("")
    lines.append(r"\begin{center}")
    lines.append(r"\begin{tabular}{lrrr}")
    lines.append(r"\toprule")
    lines.append(r"Hospital & N & Macro-AUC & Status \\")
    lines.append(r"\midrule")
    for h in p5_summary.get("hospital_ranking", []):
        if h.get("error"):
            lines.append(
                f"{_escape_latex(h['source_db'])} & --- & --- & "
                r"\textcolor{red}{Sem dados} \\"
            )
        else:
            auc_color = "green!20" if h["auc"] >= 0.75 else ("yellow!20" if h["auc"] >= 0.60 else "red!20")
            lines.append(
                r"\rowcolor{" + auc_color + r"}" +
                f"{_escape_latex(h['source_db'])} & {h['n']:,} & {h['auc']:.3f} & OK \\\\"
            )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{center}")
    lines.append("")

    # Global discharge distribution
    lines.append(r"\vspace{4pt}\noindent\textbf{Distribuicao global de desfechos (internacoes com alta):}")
    lines.append("")
    lines.append(r"\begin{center}")
    lines.append(r"\begin{tabular}{lrrl}")
    lines.append(r"\toprule")
    lines.append(r"Categoria & N & \% & Semaforo \\")
    lines.append(r"\midrule")

    for cat in ["ALTA_NORMAL", "EM_CURSO", "ALTA_COMPLEXA", "OBITO",
                "TRANSFERENCIA", "ADMINISTRATIVO", "EVASAO", "NEONATAL", "OUTRO", "SEM_STATUS"]:
        cnt = cat_global.get(cat, 0)
        if cnt == 0:
            continue
        pct   = 100.0 * cnt / n_total if n_total > 0 else 0
        color = _latex_cat_color(cat)
        signal = {
            "OBITO":         r"\textcolor{red}{\textbf{$\bullet$}}",
            "EVASAO":        r"\textcolor{red!70!black}{\textbf{$\bullet$}}",
            "ALTA_COMPLEXA": r"\textcolor{orange}{\textbf{$\bullet$}}",
            "TRANSFERENCIA": r"\textcolor{orange!80!black}{$\bullet$}",
            "ALTA_NORMAL":   r"\textcolor{green!60!black}{$\bullet$}",
        }.get(cat, r"\textcolor{gray}{$\circ$}")
        lines.append(
            r"\rowcolor{" + color + r"}" +
            f"{_escape_latex(cat)} & {cnt:,} & {pct:.1f}\\% & {signal} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{center}")
    lines.append(r"\newpage")

    # ── Part 1: LightGBM per hospital ─────────────────────────────────
    lines.append(r"\section{Parte 1 --- Previsao de Desfecho por Hospital (LightGBM)}")
    lines.append(
        r"Para cada hospital com mais de " + str(MIN_DISCHARGED) +
        r" internacoes com desfecho conhecido (IN\_SITUACAO=2), treinamos um modelo "
        r"LightGBM multiclasse com 5-fold CV estratificado. "
        r"Features: embedding de 64 dimensoes + urgencia, dia da semana, mes, ano, "
        r"convenio (encoded), CID (encoded), admissoes anteriores. "
        r"Classes alvo: ALTA\_NORMAL, OBITO, ALTA\_COMPLEXA, TRANSFERENCIA, EVASAO "
        r"(exclui EM\_CURSO, NEONATAL, ADMINISTRATIVO). "
        r"Apos o treinamento, extraimos os indices de folha (leaf indices) do modelo "
        r"final para uso no clustering da Parte 2."
    )
    lines.append("")

    for r in p1_results:
        src = _escape_latex(r["source_db"])
        lines.append(r"\subsection{" + src + r"}")

        if "error" in r:
            lines.append(r"\textcolor{red}{Erro: " + _escape_latex(r["error"]) + r"}")
            lines.append("")
            continue

        auc = r.get("auc_macro", float("nan"))
        n   = r.get("n_samples", 0)
        lines.append(
            f"\\textbf{{n={n:,}}} internacoes com desfecho $\\cdot$ "
            f"\\textbf{{macro-AUC: {auc:.3f}}} $\\cdot$ "
            f"{r.get('n_folds', 5)}-fold CV estratificado"
        )
        lines.append("")

        # Class distribution + per-class PR
        lines.append(r"\begin{tabular}{lrr|lrr}")
        lines.append(r"\toprule")
        lines.append(r"Classe & N & \% & Classe & Precisao & Recall \\")
        lines.append(r"\midrule")

        cats  = sorted(r.get("class_counts", {}).items())
        pr    = r.get("per_class_pr", {})
        n_tot = sum(v for _, v in cats)

        for i in range(0, len(cats), 2):
            cat1, cnt1 = cats[i]   if i     < len(cats) else ("", 0)
            cat2, cnt2 = cats[i+1] if (i+1) < len(cats) else ("", 0)
            pct1 = 100.0 * cnt1 / n_tot if n_tot > 0 and cnt1 else 0
            pr2  = pr.get(cat2, {})
            p2s  = f"{pr2.get('precision', 0):.2f}" if isinstance(pr2.get("precision"), float) else "---"
            r2s  = f"{pr2.get('recall',    0):.2f}" if isinstance(pr2.get("recall"),    float) else "---"
            if cat1:
                lines.append(
                    f"{_escape_latex(cat1)} & {cnt1:,} & {pct1:.1f}\\% & "
                    f"{_escape_latex(cat2)} & {p2s} & {r2s} \\\\"
                )

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append("")

        # Top-10 feature importances
        lines.append(r"\vspace{4pt}\noindent\textbf{Top-10 features por importancia:}")
        lines.append("")
        lines.append(r"\begin{tabular}{llr}")
        lines.append(r"\toprule")
        lines.append(r"Feature & Semantica & Importancia \\")
        lines.append(r"\midrule")
        for feat, imp in r.get("top10_all", [])[:10]:
            if feat.startswith("emb_"):
                dim       = int(feat.split("_")[1])
                label     = _dim_label(dim)
                feat_disp = f"emb[{dim}]"
            else:
                label     = feat
                feat_disp = feat
            lines.append(
                f"{_escape_latex(feat_disp)} & {_escape_latex(label)} & {imp:.1f} \\\\"
            )
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append("")

        narr = r.get("narrative", "")
        if narr:
            lines.append(r"\vspace{4pt}\noindent\textit{" + _escape_latex(narr) + r"}")
        lines.append("")

    lines.append(r"\newpage")

    # ── Part 2: Leaf-based clustering ─────────────────────────────────
    lines.append(r"\section{Parte 2 --- Clustering por Folhas do LightGBM (HDBSCAN)}")
    lines.append(
        r"Inovacao central: cada internacao e representada pelos seus indices de folha "
        r"em cada arvore do LightGBM. Amostras na mesma folha compartilham exatamente o "
        r"mesmo caminho de decisao. Aplicamos HDBSCAN sobre essa matriz de folhas para "
        r"encontrar grupos semanticamente coerentes sem forcamos um $k$ pre-definido. "
        r"Tamanho minimo de cluster: " + str(HDBSCAN_MIN_CLUSTER_SIZE_LEAF) + r" internacoes. "
        r"Pontos nao agrupados (outliers) sao reportados separadamente, "
        r"nao distorcendo a analise."
        r"\par"
        r"Semaforo: \textcolor{red}{\textbf{VERMELHO}} = obito $>5\%$ ou evasao $>3\%$; "
        r"\textcolor{orange}{\textbf{AMARELO}} = alta complexa $>10\%$; "
        r"\textcolor{green!60!black}{\textbf{VERDE}} = padrao normal."
    )
    lines.append("")

    for cr in p2_leaf_results:
        src = _escape_latex(cr["source_db"])
        lines.append(r"\subsection{" + src + r"}")

        if "error" in cr:
            lines.append(r"\textcolor{red}{Erro: " + _escape_latex(cr["error"]) + r"}")
            lines.append("")
            continue

        n_noise = cr.get("n_noise", 0)
        lines.append(
            f"{cr['n_records']:,} internacoes $\\cdot$ "
            f"{cr['n_clusters']} clusters $\\cdot$ "
            f"{n_noise:,} outliers (nao agrupados)"
        )
        lines.append("")

        lines.extend(_latex_cluster_table(cr["sorted_clusters"]))

        # Noise stats
        ns = cr.get("noise_stats", {})
        if ns.get("n_total", 0) > 0:
            lines.append(
                f"\\vspace{{4pt}}\\noindent\\textit{{Nao agrupados (outliers): "
                f"{ns['n_total']:,} internacoes, "
                f"obito={ns.get('obito_rate', 0):.1f}\\%, "
                f"evasao={ns.get('evasao_rate', 0):.1f}\\%.}}"
            )
            lines.append("")

        lines.extend(_latex_cluster_detail(cr["sorted_clusters"], cr["source_db"]))
        lines.append(r"\newpage")

    # ── Part 3: HDBSCAN on raw embeddings ─────────────────────────────
    lines.append(r"\section{Parte 3 --- Clustering por Embeddings Brutos (HDBSCAN)}")
    lines.append(
        r"Como comparacao e alternativa, aplicamos HDBSCAN diretamente sobre os "
        r"embeddings de 64 dimensoes de cada internacao. Este metodo nao usa o "
        r"LightGBM e pode capturar padroes geometricos no espaco de embeddings "
        r"que o modelo supervisionado nao aprendeu. "
        r"Tamanho minimo de cluster: " + str(HDBSCAN_MIN_CLUSTER_SIZE_EMB) + r" internacoes."
    )
    lines.append("")

    for cr in p3_emb_results:
        src = _escape_latex(cr["source_db"])
        lines.append(r"\subsection{" + src + r"}")

        if "error" in cr:
            lines.append(r"\textcolor{red}{Erro: " + _escape_latex(cr["error"]) + r"}")
            lines.append("")
            continue

        n_noise = cr.get("n_noise", 0)
        lines.append(
            f"{cr['n_records']:,} internacoes $\\cdot$ "
            f"{cr['n_clusters']} clusters $\\cdot$ "
            f"{n_noise:,} outliers"
        )
        lines.append("")

        lines.extend(_latex_cluster_table(cr["sorted_clusters"]))

        ns = cr.get("noise_stats", {})
        if ns.get("n_total", 0) > 0:
            lines.append(
                f"\\vspace{{4pt}}\\noindent\\textit{{Nao agrupados (outliers): "
                f"{ns['n_total']:,} internacoes, "
                f"obito={ns.get('obito_rate', 0):.1f}\\%, "
                f"evasao={ns.get('evasao_rate', 0):.1f}\\%.}}"
            )
            lines.append("")

        lines.extend(_latex_cluster_detail(cr["sorted_clusters"], cr["source_db"]))
        lines.append(r"\newpage")

    # ── Part 5: Cross-hospital summary ────────────────────────────────
    lines.append(r"\section{Parte 5 --- Resumo Cruzado entre Hospitais}")

    # Dim delta
    lines.append(r"\subsection{Dimensoes de Embedding mais Preditivas de Obito}")
    lines.append(
        r"Diferenca absoluta entre a media dos embeddings de internacoes com "
        r"desfecho OBITO versus todos os outros desfechos (apenas internacoes "
        r"com alta confirmada, IN\_SITUACAO=2). "
        r"Dimensoes com maior delta separam melhor os casos fatais na admissao."
    )
    lines.append("")

    n_obito = p5_summary.get("n_obito_total", 0)
    n_non   = p5_summary.get("n_nonobito_total", 0)
    lines.append(f"Total analisado: {n_obito:,} obitos vs {n_non:,} outros desfechos.")
    lines.append("")

    lines.append(r"\begin{tabular}{clr}")
    lines.append(r"\toprule")
    lines.append(r"Dim & Semantica & Delta medio \\")
    lines.append(r"\midrule")
    for dim, delta in p5_summary.get("dim_delta_top10", []):
        lines.append(f"emb[{dim}] & {_escape_latex(_dim_label(dim))} & {delta:.4f} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append("")

    # CIDs by obito rate
    lines.append(r"\subsection{CIDs com Maior Taxa de Obito (min 30 casos)}")
    lines.append("")
    lines.append(r"\begin{longtable}{llrrr}")
    lines.append(r"\toprule")
    lines.append(r"CID & Descricao & Total & Obitos & Taxa\% \\")
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    for cid, total, n_obito_c, rate, desc in p5_summary.get("top_cids_obito", []):
        color = "red!20" if rate > 20 else ("orange!20" if rate > 10 else "yellow!10")
        lines.append(
            r"\rowcolor{" + color + r"}" +
            f"{_escape_latex(str(cid))} & {_escape_latex(_truncate(str(desc), 50))} & "
            f"{total:,} & {n_obito_c:,} & {rate:.1f}\\% \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{longtable}")

    # Convenios by evasao rate
    lines.append(r"\subsection{Convenios com Maior Taxa de Evasao (min 50 casos)}")
    lines.append("")
    lines.append(r"\begin{tabular}{lrrr}")
    lines.append(r"\toprule")
    lines.append(r"Convenio ID & Total & Evasoes & Taxa\% \\")
    lines.append(r"\midrule")
    for conv, total, n_ev, rate in p5_summary.get("top_convs_evasao", []):
        color = "red!15" if rate > 5 else "yellow!15"
        lines.append(
            r"\rowcolor{" + color + r"}" +
            f"{_escape_latex(str(conv))} & {total:,} & {n_ev:,} & {rate:.1f}\\% \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append("")

    lines.append(r"\end{document}")
    return "\n".join(lines)


def _compile_pdf(latex_str: str, output_pdf: str) -> bool:
    import os
    import subprocess

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tex_path = output_pdf.replace(".pdf", ".tex")

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_str)
    print(f"    LaTeX written: {tex_path}")

    for pass_n in (1, 2):
        cmd = [
            "pdflatex",
            "-interaction=nonstopmode",
            "-output-directory", OUTPUT_DIR,
            tex_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"    pdflatex pass {pass_n} FAILED (returncode={result.returncode})")
            print(result.stdout[-3000:])
            print(result.stderr[-1000:])
            if pass_n == 2:
                return False
        else:
            print(f"    pdflatex pass {pass_n} OK")

    if os.path.exists(output_pdf):
        size = os.path.getsize(output_pdf)
        print(f"    PDF generated: {output_pdf}  ({size/1024:.0f} KB)")
        return True
    print("    PDF not found after compilation")
    return False


# ─────────────────────────────────────────────────────────────────
# Modal entrypoint
# ─────────────────────────────────────────────────────────────────

@app.function(
    image=analysis_image,
    volumes=VOLUMES,
    cpu=8,
    memory=40960,
    timeout=7200,
)
def run_analysis():
    import time
    import duckdb

    t_global = time.time()
    print("=" * 70)
    print("JCUBE Admission-Discharge Analysis V5")
    print(f"Date: {REPORT_DATE_STR}")
    print("Architecture: LightGBM -> Leaf HDBSCAN -> Embedding HDBSCAN -> Report")
    print("=" * 70)

    # 1. Load embeddings
    unique_nodes, embeddings, node_to_idx, intern_mask = _load_internacao_embeddings()

    # 2. Build feature + discharge table (IN_SITUACAO = 2 only)
    con     = duckdb.connect(str(DB_PATH))
    records = _build_discharge_table(con)
    con.close()

    # 3. Attach embeddings
    records_with_emb, n_missing = _attach_embeddings(records, embeddings, node_to_idx)
    print(f"    {len(records_with_emb):,} records with embeddings "
          f"({n_missing:,} missing, {len(records):,} total)")

    # Show global discharge category breakdown
    print("\n[3b/7] Global discharge category distribution:")
    cat_global: dict[str, int] = {}
    for r in records:
        c = r.get("discharge_category", "OUTRO")
        cat_global[c] = cat_global.get(c, 0) + 1
    for cat, cnt in sorted(cat_global.items(), key=lambda x: x[1], reverse=True):
        print(f"    {cat:25s}: {cnt:>8,}  ({100.0*cnt/len(records):.1f}%)")

    # 4. Part 1: Per-hospital LightGBM + leaf extraction
    print("\n[4/7] Part 1: Per-hospital LightGBM discharge prediction + leaf extraction ...")
    source_dbs = sorted({r["source_db"] for r in records_with_emb})
    p1_results: list[dict] = []

    for src in source_dbs:
        src_records = [r for r in records_with_emb if r["source_db"] == src]
        n_eligible  = sum(
            1 for r in src_records
            if r.get("discharge_category") in TARGET_CATEGORIES
        )
        print(f"\n  {src}: {len(src_records):,} total, {n_eligible:,} eligible for training")
        if n_eligible < MIN_DISCHARGED:
            p1_results.append({
                "source_db": src,
                "error":     f"Only {n_eligible} eligible rows (< {MIN_DISCHARGED})",
            })
            continue
        result = _run_lgbm_discharge(src, src_records)
        p1_results.append(result)

    _print_part1(p1_results)

    # 5. Part 2: Leaf-based HDBSCAN clustering
    print("\n[5/7] Part 2: Leaf-based HDBSCAN clustering ...")
    p2_leaf_results: list[dict] = []
    for p1_r in p1_results:
        src = p1_r["source_db"]
        print(f"\n  {src}: running leaf clustering ...")
        result = _run_leaf_clustering(p1_r)
        p2_leaf_results.append(result)
        _print_cluster_result(result, "Part 2 (leaf)")

    # 6. Part 3: HDBSCAN on raw embeddings
    print("\n[6/7] Part 3: HDBSCAN on raw embeddings (comparison) ...")
    p3_emb_results: list[dict] = []
    for src in source_dbs:
        src_records = [r for r in records_with_emb if r["source_db"] == src]
        print(f"\n  {src}: running embedding clustering on {len(src_records):,} records ...")
        result = _run_embedding_clustering(src, src_records)
        p3_emb_results.append(result)
        _print_cluster_result(result, "Part 3 (embedding)")

    # 7. Part 5: Cross-hospital summary
    p5_summary = _cross_hospital_summary(records_with_emb, p1_results)
    _print_part5(p5_summary)

    # Generate LaTeX PDF
    print("\n[7/7] Generating LaTeX PDF report ...")
    latex_str = _gen_latex(
        p1_results,
        p2_leaf_results,
        p3_emb_results,
        p5_summary,
        records,
    )
    ok = _compile_pdf(latex_str, OUTPUT_PDF)

    # Free leaf matrices (large) before exit
    for r in p1_results:
        r.pop("leaf_matrix", None)
        r.pop("eligible_records", None)

    elapsed = time.time() - t_global
    print(f"\n{'='*70}")
    if ok:
        print(f"DONE. PDF: {OUTPUT_PDF}  Total time: {elapsed:.0f}s")
    else:
        print(f"DONE (PDF failed). Total time: {elapsed:.0f}s")
    print("=" * 70)


@app.local_entrypoint()
def main():
    run_analysis.remote()
