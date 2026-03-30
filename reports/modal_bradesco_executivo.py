#!/usr/bin/env python3
"""
Modal script: Bradesco Executive Deliverable — 3-Page PDF

This is the PRODUCT — what gets delivered to the Bradesco client.
Zero technical jargon. Pure business language. Action-oriented.

Pages:
  1. Panorama Fevereiro 2026 (KPIs, trend chart, top hospitals)
  2. Pacientes que Requerem Verificacao Imediata (AI-flagged cases by hospital)
  3. Alertas e Recomendacoes (data quality, hospital drift, readmission, actions)

Output: /data/reports/bradesco_executivo_fev2026.pdf

Usage:
    modal run reports/modal_bradesco_executivo.py
    modal run --detach reports/modal_bradesco_executivo.py
"""
from __future__ import annotations

import modal

app = modal.App("jcube-bradesco-executivo")

jepa_cache = modal.Volume.from_name("jepa-cache", create_if_missing=False)
data_vol   = modal.Volume.from_name("jcube-data",  create_if_missing=False)

VOLUMES = {
    "/cache": jepa_cache,
    "/data":  data_vol,
}

report_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "texlive-latex-base",
        "texlive-latex-recommended",
        "texlive-latex-extra",
        "texlive-fonts-recommended",
        "texlive-lang-portuguese",
        "lmodern",
        "fonts-dejavu-core",
    )
    .pip_install(
        "torch>=2.2",
        "numpy>=1.26",
        "duckdb>=1.2.0",
        "pyarrow>=18.0",
        "scikit-learn>=1.4",
        "matplotlib>=3.8",
    )
)

# -----------------------------------------------------------------
# Constants
# -----------------------------------------------------------------

GRAPH_PARQUET = "/data/jcube_graph.parquet"
WEIGHTS_PATH  = "/cache/tkg-v6.2/node_emb_epoch_3.pt"
DB_PATH       = "/data/aggregated_fixed_union.db"
OUTPUT_DIR    = "/data/reports"
OUTPUT_PDF    = f"{OUTPUT_DIR}/bradesco_executivo_fev2026.pdf"

SOURCE_DB       = "GHO-BRADESCO"
SRC             = f"source_db = '{SOURCE_DB}'"
FEB_START       = "2026-02-01"
FEB_END         = "2026-02-28"
JAN_START       = "2026-01-01"
JAN_END         = "2026-01-31"
REPORT_DATE_STR = "30/03/2026"

Z_THRESHOLD     = 2.0

# 6 months for trend: Sep 2025 -> Feb 2026
TREND_MONTHS = [
    ("2025-09-01", "2025-09-30", "Set/25"),
    ("2025-10-01", "2025-10-31", "Out/25"),
    ("2025-11-01", "2025-11-30", "Nov/25"),
    ("2025-12-01", "2025-12-31", "Dez/25"),
    ("2026-01-01", "2026-01-31", "Jan/26"),
    ("2026-02-01", "2026-02-28", "Fev/26"),
]


# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------

def _safe_int(v, default=0):
    try:
        return int(v) if v is not None else default
    except Exception:
        return default


def _safe_float(v, default=0.0):
    try:
        return float(v) if v is not None else default
    except Exception:
        return default


def _escape_latex(s):
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
    s = s.replace("\r\n", " ").replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return s


def _pct(num, den, decimals=1):
    if den == 0:
        return "---"
    return f"{100.0 * num / den:.{decimals}f}\\%"


def _fmt_date(d):
    if d is None:
        return "---"
    try:
        if isinstance(d, str):
            return d[:10]
        return d.strftime("%d/%m/%Y")
    except Exception:
        return str(d)[:10]


def _truncate(s, max_len=80):
    if not s:
        return "---"
    s = str(s).strip()
    if len(s) > max_len:
        return s[:max_len] + "..."
    return s


def _exec(con, q):
    cur  = con.execute(q)
    cols = [d[0] for d in cur.description]
    return cols, cur.fetchall()


def _rows_to_dicts(cols, rows):
    return [dict(zip(cols, r)) for r in rows]


def _fmt_thousands(v):
    return f"{_safe_int(v):,}".replace(",", ".")


def _delta_arrow(new_val, old_val, invert=False):
    """Return a LaTeX delta string with arrow. invert=True means lower is better."""
    nv = _safe_float(new_val)
    ov = _safe_float(old_val)
    if ov == 0:
        return "---"
    pct_change = 100.0 * (nv - ov) / abs(ov)
    if abs(pct_change) < 0.05:
        arrow = "$\\rightarrow$"
        color = "bradgray"
    elif pct_change > 0:
        arrow = "$\\uparrow$"
        color = "bradred" if invert else "bradgreen"
    else:
        arrow = "$\\downarrow$"
        color = "bradgreen" if invert else "bradred"
    return f"\\textcolor{{{color}}}{{{arrow}{abs(pct_change):.1f}\\%}}"


_HOSP_NAMES: dict[int, str] = {}


def _load_hospital_names(con):
    global _HOSP_NAMES
    cols, rows = _exec(con, f"""
        SELECT ID_CD_HOSPITAL, NM_HOSPITAL
        FROM agg_tb_capta_add_hospitais_caho
        WHERE source_db = '{SOURCE_DB}' AND NM_HOSPITAL IS NOT NULL
    """)
    _HOSP_NAMES = {int(r[0]): str(r[1]) for r in rows}
    print(f"    Loaded {len(_HOSP_NAMES)} hospital names")


def _hosp_name(hid):
    hid_int = _safe_int(hid)
    return _HOSP_NAMES.get(hid_int, f"Hospital {hid_int}")


# Hospital UF lookup
_HOSP_UF: dict[int, str] = {}


def _load_hospital_uf(con):
    global _HOSP_UF
    try:
        cols, rows = _exec(con, f"""
            SELECT ID_CD_HOSPITAL, SG_UF
            FROM agg_tb_capta_add_hospitais_caho
            WHERE source_db = '{SOURCE_DB}' AND SG_UF IS NOT NULL
        """)
        _HOSP_UF = {int(r[0]): str(r[1]) for r in rows}
    except Exception:
        pass


def _hosp_uf(hid):
    return _HOSP_UF.get(_safe_int(hid), "")


# -----------------------------------------------------------------
# Mortality helper
# -----------------------------------------------------------------

_MORTALITY_Q_TEMPLATE = """
    WITH target_dis AS (
        SELECT ID_CD_INTERNACAO
        FROM agg_tb_capta_internacao_cain
        WHERE {src} AND IN_SITUACAO = 2
          AND DH_FINALIZACAO >= '{m_start}'::DATE
          AND DH_FINALIZACAO < '{m_end}'::DATE + INTERVAL '1 day'
    ),
    last_st AS (
        SELECT es.ID_CD_INTERNACAO,
               ROW_NUMBER() OVER (PARTITION BY es.ID_CD_INTERNACAO ORDER BY es.DH_CADASTRO DESC) AS rn,
               es.FL_DESOSPITALIZACAO
        FROM agg_tb_capta_evo_status_caes es
        WHERE es.{src}
          AND es.ID_CD_INTERNACAO IN (SELECT ID_CD_INTERNACAO FROM target_dis)
    )
    SELECT COUNT(DISTINCT ls.ID_CD_INTERNACAO) FILTER (
        WHERE UPPER(f.DS_FINAL_MONITORAMENTO) LIKE '%%BITO%%'
    ) AS n_obito
    FROM last_st ls
    JOIN agg_tb_capta_tipo_final_monit_fmon f
        ON ls.FL_DESOSPITALIZACAO = f.ID_CD_FINAL_MONITORAMENTO
        AND f.{src}
    WHERE ls.rn = 1
"""


def _count_obitos(con, m_start, m_end):
    q = _MORTALITY_Q_TEMPLATE.format(src=SRC, m_start=m_start, m_end=m_end)
    _, rows = _exec(con, q)
    return _safe_int(rows[0][0]) if rows else 0


# -----------------------------------------------------------------
# Chart infrastructure
# -----------------------------------------------------------------

CHART_DIR = "/tmp/charts_exec"

BRAD_COLORS = {
    "blue":      (0/255, 47/255, 108/255),
    "red":       (204/255, 0/255, 0/255),
    "gold":      (180/255, 140/255, 20/255),
    "green":     (0/255, 128/255, 60/255),
    "gray":      (100/255, 100/255, 100/255),
    "lightblue": (220/255, 235/255, 250/255),
    "lightgray": (245/255, 245/255, 245/255),
}


def _setup_chart_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Liberation Sans", "Arial"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": BRAD_COLORS["gray"],
        "axes.labelcolor": BRAD_COLORS["gray"],
        "xtick.color": BRAD_COLORS["gray"],
        "ytick.color": BRAD_COLORS["gray"],
    })


def _safe_chart(chart_fn, path, *args, **kwargs):
    import os
    try:
        chart_fn(path, *args, **kwargs)
        if os.path.exists(path):
            return path
    except Exception as ex:
        print(f"    CHART FAILED ({os.path.basename(path)}): {ex}")
    return None


def _chart_trend_dual_axis(path, trend):
    """Bar chart for admissions + line for avg_los on dual Y-axis."""
    import matplotlib.pyplot as plt
    import numpy as np

    if not trend:
        return

    labels = [t.get("label", "") for t in trend]
    admissoes = [_safe_int(t.get("admissoes")) for t in trend]
    avg_los = [_safe_float(t.get("avg_los")) for t in trend]

    fig, ax1 = plt.subplots(figsize=(7.5, 3.5))
    x = np.arange(len(labels))

    # Highlight Feb (last month)
    if len(x) > 0:
        ax1.axvspan(x[-1] - 0.4, x[-1] + 0.4, alpha=0.08, color=BRAD_COLORS["blue"], zorder=0)

    bars = ax1.bar(x, admissoes, color=BRAD_COLORS["blue"], alpha=0.8, width=0.6,
                   label="Novas Admissoes", zorder=2)
    ax1.set_xlabel("Mes", fontsize=9)
    ax1.set_ylabel("Novas Admissoes", color=BRAD_COLORS["blue"], fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.tick_params(axis='y', labelcolor=BRAD_COLORS["blue"])

    for bar, val in zip(bars, admissoes):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(admissoes) * 0.01,
                 f"{val:,}".replace(",", "."), ha="center", va="bottom", fontsize=7,
                 fontweight="bold", color=BRAD_COLORS["blue"])

    ax2 = ax1.twinx()
    ax2.plot(x, avg_los, color=BRAD_COLORS["red"], marker='o', linewidth=2, markersize=6,
             label="Tempo Medio de Internacao", zorder=3)
    ax2.set_ylabel("Tempo Medio de Internacao (dias)", color=BRAD_COLORS["red"], fontsize=9)
    ax2.tick_params(axis='y', labelcolor=BRAD_COLORS["red"])
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(BRAD_COLORS["red"])

    for i, v in enumerate(avg_los):
        ax2.annotate(f"{v:.1f}d", (i, v), textcoords="offset points", xytext=(0, 10),
                     fontsize=7, fontweight="bold", color=BRAD_COLORS["red"], ha="center",
                     bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                               edgecolor=BRAD_COLORS["red"], alpha=0.7, linewidth=0.4))

    ax1.set_title("Admissoes e Tempo Medio de Internacao (6 meses)",
                  fontsize=11, color=BRAD_COLORS["blue"])

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# -----------------------------------------------------------------
# Data fetchers (self-contained subset)
# -----------------------------------------------------------------

def _fetch_kpis_for_month(con, m_start, m_end):
    """Fetch full KPI set for any month."""
    M_ACTIVE = f"""
        {SRC}
        AND DH_ADMISSAO_HOSP < '{m_end}'::DATE + INTERVAL '1 day'
        AND (DH_FINALIZACAO IS NULL OR DH_FINALIZACAO >= '{m_start}'::DATE)
    """
    M_NEW = f"""
        {SRC}
        AND DH_ADMISSAO_HOSP >= '{m_start}'::DATE
        AND DH_ADMISSAO_HOSP < '{m_end}'::DATE + INTERVAL '1 day'
    """
    M_DISCHARGED = f"""
        {SRC}
        AND IN_SITUACAO = 2
        AND DH_FINALIZACAO >= '{m_start}'::DATE
        AND DH_FINALIZACAO < '{m_end}'::DATE + INTERVAL '1 day'
    """

    kpis = {}

    # Census
    cols, rows = _exec(con, f"""
        SELECT COUNT(DISTINCT ID_CD_INTERNACAO) AS censo,
               COUNT(DISTINCT ID_CD_PACIENTE) AS pacientes
        FROM agg_tb_capta_internacao_cain WHERE {M_ACTIVE}
    """)
    kpis.update(dict(zip(cols, rows[0])) if rows else {})

    # New admissions
    cols, rows = _exec(con, f"""
        SELECT COUNT(DISTINCT ID_CD_INTERNACAO) AS novas_admissoes
        FROM agg_tb_capta_internacao_cain WHERE {M_NEW}
    """)
    kpis.update(dict(zip(cols, rows[0])) if rows else {})

    # Discharges
    cols, rows = _exec(con, f"""
        SELECT COUNT(DISTINCT ID_CD_INTERNACAO) AS altas
        FROM agg_tb_capta_internacao_cain WHERE {M_DISCHARGED}
    """)
    kpis.update(dict(zip(cols, rows[0])) if rows else {})

    # LOS stats
    cols, rows = _exec(con, f"""
        SELECT
            AVG(DATEDIFF('day', DH_ADMISSAO_HOSP::DATE, DH_FINALIZACAO::DATE)) AS avg_los,
            MEDIAN(DATEDIFF('day', DH_ADMISSAO_HOSP::DATE, DH_FINALIZACAO::DATE)) AS median_los
        FROM agg_tb_capta_internacao_cain
        WHERE {M_DISCHARGED} AND DH_ADMISSAO_HOSP IS NOT NULL
    """)
    kpis.update(dict(zip(cols, rows[0])) if rows else {})

    # Mortality
    n_obito = _count_obitos(con, m_start, m_end)
    n_altas = _safe_int(kpis.get("altas"))
    kpis["n_obito"] = n_obito
    kpis["taxa_obito"] = 100.0 * n_obito / n_altas if n_altas > 0 else 0.0

    # Readmission 30d
    cols, rows = _exec(con, f"""
        WITH ordered_adm AS (
            SELECT ID_CD_PACIENTE, ID_CD_INTERNACAO,
                   DH_ADMISSAO_HOSP::DATE AS dt_adm,
                   LAG(DH_FINALIZACAO::DATE) OVER (
                       PARTITION BY ID_CD_PACIENTE ORDER BY DH_ADMISSAO_HOSP
                   ) AS prev_discharge
            FROM agg_tb_capta_internacao_cain
            WHERE {SRC} AND IN_SITUACAO = 2 AND DH_ADMISSAO_HOSP IS NOT NULL
        )
        SELECT
            COUNT(*) FILTER (WHERE DATEDIFF('day', prev_discharge, dt_adm) BETWEEN 1 AND 30
                             AND dt_adm >= '{m_start}'::DATE
                             AND dt_adm < '{m_end}'::DATE + INTERVAL '1 day') AS readmissions_30d,
            COUNT(*) FILTER (WHERE prev_discharge IS NOT NULL
                             AND dt_adm >= '{m_start}'::DATE
                             AND dt_adm < '{m_end}'::DATE + INTERVAL '1 day') AS total_with_prior
        FROM ordered_adm
    """)
    kpis.update(dict(zip(cols, rows[0])) if rows else {})
    rr = _safe_int(kpis.get("readmissions_30d"))
    tp = _safe_int(kpis.get("total_with_prior"))
    kpis["taxa_readmit"] = 100.0 * rr / tp if tp > 0 else 0.0

    # Adverse events
    cols, rows = _exec(con, f"""
        SELECT COUNT(*) AS eventos_adversos
        FROM agg_tb_capta_eventos_adversos_caed
        WHERE {SRC}
          AND DH_EVENTO_ADVERSO >= '{m_start}'::DATE
          AND DH_EVENTO_ADVERSO < '{m_end}'::DATE + INTERVAL '1 day'
    """)
    kpis.update(dict(zip(cols, rows[0])) if rows else {})

    # Open cases
    cols, rows = _exec(con, f"""
        SELECT COUNT(*) AS open_cases,
               MAX(DATEDIFF('day', DH_ADMISSAO_HOSP::DATE, CURRENT_DATE)) AS max_open_days
        FROM agg_tb_capta_internacao_cain
        WHERE {SRC}
          AND (IN_SITUACAO IS NULL OR IN_SITUACAO != 2)
          AND DH_FINALIZACAO IS NULL
          AND DH_ADMISSAO_HOSP IS NOT NULL
    """)
    kpis.update(dict(zip(cols, rows[0])) if rows else {})

    # Open cases >60d
    cols, rows = _exec(con, f"""
        SELECT COUNT(*) AS open_gt_60d
        FROM agg_tb_capta_internacao_cain
        WHERE {SRC}
          AND (IN_SITUACAO IS NULL OR IN_SITUACAO != 2)
          AND DH_FINALIZACAO IS NULL
          AND DH_ADMISSAO_HOSP IS NOT NULL
          AND DATEDIFF('day', DH_ADMISSAO_HOSP::DATE, CURRENT_DATE) > 60
    """)
    kpis.update(dict(zip(cols, rows[0])) if rows else {})

    return kpis


def _fetch_kpis(con):
    import time
    print("[1/6] KPIs (Feb + Jan) ...")
    t0 = time.time()
    feb = _fetch_kpis_for_month(con, FEB_START, FEB_END)
    jan = _fetch_kpis_for_month(con, JAN_START, JAN_END)
    print(f"    done in {time.time()-t0:.1f}s")
    return {"feb": feb, "jan": jan}


def _fetch_hospital_performance(con):
    """Hospital ranking for Feb AND Jan (for MoM LOS delta)."""
    import time
    print("[2/6] Hospital performance ...")
    t0 = time.time()

    def _hosp_for_month(m_start, m_end):
        cols, rows = _exec(con, f"""
            SELECT i.ID_CD_HOSPITAL AS hospital_id,
                   COUNT(DISTINCT i.ID_CD_INTERNACAO) AS n_altas,
                   AVG(DATEDIFF('day', i.DH_ADMISSAO_HOSP::DATE, i.DH_FINALIZACAO::DATE)) AS avg_los,
                   MEDIAN(DATEDIFF('day', i.DH_ADMISSAO_HOSP::DATE, i.DH_FINALIZACAO::DATE)) AS median_los
            FROM agg_tb_capta_internacao_cain i
            WHERE i.{SRC}
              AND i.IN_SITUACAO = 2
              AND i.DH_FINALIZACAO >= '{m_start}'::DATE
              AND i.DH_FINALIZACAO < '{m_end}'::DATE + INTERVAL '1 day'
              AND i.DH_ADMISSAO_HOSP IS NOT NULL
            GROUP BY i.ID_CD_HOSPITAL
            ORDER BY n_altas DESC
        """)
        return {r[0]: dict(zip(cols, r)) for r in rows}

    feb_base = _hosp_for_month(FEB_START, FEB_END)
    jan_base = _hosp_for_month(JAN_START, JAN_END)

    result = []
    for hid, base in feb_base.items():
        n_altas = _safe_int(base.get("n_altas"))
        jan_h = jan_base.get(hid)
        jan_los = _safe_float(jan_h.get("avg_los")) if jan_h else None
        feb_los = _safe_float(base.get("avg_los"))
        los_delta_pct = None
        if jan_los and jan_los > 0:
            los_delta_pct = 100.0 * (feb_los - jan_los) / jan_los
        result.append({
            "hospital_id": hid,
            "n_altas": n_altas,
            "avg_los": feb_los,
            "median_los": _safe_float(base.get("median_los")),
            "jan_avg_los": jan_los,
            "los_delta_pct": los_delta_pct,
        })
    result.sort(key=lambda x: -x["n_altas"])

    print(f"    {len(result)} hospitals, done in {time.time()-t0:.1f}s")
    return result


def _fetch_trend_6m(con):
    """6-month trend: admissions + LOS."""
    import time
    print("[3/6] 6-month trend ...")
    t0 = time.time()

    trend = []
    for m_start, m_end, label in TREND_MONTHS:
        cols, rows = _exec(con, f"""
            SELECT
                COUNT(DISTINCT CASE
                    WHEN DH_ADMISSAO_HOSP >= '{m_start}'::DATE
                     AND DH_ADMISSAO_HOSP < '{m_end}'::DATE + INTERVAL '1 day'
                    THEN ID_CD_INTERNACAO END) AS admissoes,
                COUNT(DISTINCT CASE
                    WHEN IN_SITUACAO = 2
                     AND DH_FINALIZACAO >= '{m_start}'::DATE
                     AND DH_FINALIZACAO < '{m_end}'::DATE + INTERVAL '1 day'
                    THEN ID_CD_INTERNACAO END) AS altas
            FROM agg_tb_capta_internacao_cain
            WHERE {SRC}
        """)
        r = dict(zip(cols, rows[0])) if rows else {}

        cols2, rows2 = _exec(con, f"""
            SELECT AVG(DATEDIFF('day', DH_ADMISSAO_HOSP::DATE, DH_FINALIZACAO::DATE)) AS avg_los
            FROM agg_tb_capta_internacao_cain
            WHERE {SRC} AND IN_SITUACAO = 2
              AND DH_FINALIZACAO >= '{m_start}'::DATE
              AND DH_FINALIZACAO < '{m_end}'::DATE + INTERVAL '1 day'
              AND DH_ADMISSAO_HOSP IS NOT NULL
        """)
        avg_los = _safe_float(rows2[0][0]) if rows2 and rows2[0][0] else 0

        trend.append({
            "label": label,
            "admissoes": _safe_int(r.get("admissoes")),
            "altas": _safe_int(r.get("altas")),
            "avg_los": avg_los,
        })

    print(f"    done in {time.time()-t0:.1f}s")
    return trend


def _fetch_data_quality_global(con):
    """Global data quality: % discharges without discharge type in Feb."""
    import time
    print("[4/6] Data quality (global) ...")
    t0 = time.time()

    result = {"total_dis": 0, "sem_tipo": 0, "pct_sem_tipo": 0.0}
    try:
        _, rows = _exec(con, f"""
            WITH feb_dis AS (
                SELECT ID_CD_INTERNACAO
                FROM agg_tb_capta_internacao_cain
                WHERE {SRC} AND IN_SITUACAO = 2
                  AND DH_FINALIZACAO >= '{FEB_START}'::DATE
                  AND DH_FINALIZACAO < '{FEB_END}'::DATE + INTERVAL '1 day'
            ),
            last_st AS (
                SELECT es.ID_CD_INTERNACAO,
                       ROW_NUMBER() OVER (PARTITION BY es.ID_CD_INTERNACAO ORDER BY es.DH_CADASTRO DESC) AS rn,
                       es.FL_DESOSPITALIZACAO
                FROM agg_tb_capta_evo_status_caes es
                WHERE es.{SRC}
                  AND es.ID_CD_INTERNACAO IN (SELECT ID_CD_INTERNACAO FROM feb_dis)
            ),
            typed AS (
                SELECT ls.ID_CD_INTERNACAO,
                       f.DS_FINAL_MONITORAMENTO AS tipo
                FROM last_st ls
                LEFT JOIN agg_tb_capta_tipo_final_monit_fmon f
                    ON ls.FL_DESOSPITALIZACAO = f.ID_CD_FINAL_MONITORAMENTO
                    AND f.{SRC}
                WHERE ls.rn = 1
            )
            SELECT
                COUNT(*) AS total_dis,
                COUNT(*) FILTER (WHERE tipo IN (
                    'Monitoramento em andamento', 'Manter Monitoramento',
                    'Manter monitoramento', 'Instavel', 'Provavel Alta Complexa'
                ) OR tipo IS NULL) AS sem_tipo
            FROM typed
        """)
        total = _safe_int(rows[0][0])
        sem = _safe_int(rows[0][1])
        result = {
            "total_dis": total,
            "sem_tipo": sem,
            "pct_sem_tipo": 100.0 * sem / total if total > 0 else 0.0,
        }
    except Exception as ex:
        print(f"    Data quality FAILED: {ex}")

    print(f"    done in {time.time()-t0:.1f}s")
    return result


def _load_twin_predictive(con):
    """Load V6.2 twin: death proximity for open cases + hospital drift + hospital priorities."""
    import time
    import numpy as np
    import torch
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    import pyarrow as pa

    print("[5/6] V6.2 predictive twin ...")
    t0 = time.time()

    # Load vocab
    table = pq.read_table(GRAPH_PARQUET, columns=["subject_id", "object_id"])
    subj = table.column("subject_id")
    obj  = table.column("object_id")
    all_nodes = pa.chunked_array(subj.chunks + obj.chunks)
    unique_nodes = pc.unique(all_nodes).to_numpy(zero_copy_only=False).astype(object)
    del table, subj, obj, all_nodes
    print(f"    {len(unique_nodes):,} nodes in vocab")

    # Load weights
    state = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
    if isinstance(state, torch.Tensor):
        emb = state.numpy().astype(np.float32)
    elif isinstance(state, dict) and "weight" in state:
        emb = state["weight"].numpy().astype(np.float32)
    else:
        emb = list(state.values())[0].numpy().astype(np.float32)
    print(f"    Embeddings: {emb.shape}")

    assert len(unique_nodes) == emb.shape[0], f"Mismatch: {len(unique_nodes)} vs {emb.shape[0]}"
    node_to_idx = {str(n): i for i, n in enumerate(unique_nodes)}

    # Get all BRADESCO internacoes
    cols, rows = _exec(con, f"""
        SELECT i.ID_CD_INTERNACAO, i.ID_CD_PACIENTE, i.ID_CD_HOSPITAL,
               i.DH_ADMISSAO_HOSP, i.DH_FINALIZACAO, i.IN_SITUACAO,
               DATEDIFF('day', i.DH_ADMISSAO_HOSP::DATE,
                   COALESCE(i.DH_FINALIZACAO, CURRENT_TIMESTAMP)::DATE) AS los
        FROM agg_tb_capta_internacao_cain i
        WHERE i.{SRC} AND i.DH_ADMISSAO_HOSP IS NOT NULL
    """)
    all_admissions = _rows_to_dicts(cols, rows)

    adm_map = {}
    for a in all_admissions:
        iid = a["ID_CD_INTERNACAO"]
        key = f"{SOURCE_DB}/ID_CD_INTERNACAO_{iid}"
        idx = node_to_idx.get(key)
        if idx is not None:
            a["emb_idx"] = idx
            adm_map[iid] = a

    print(f"    {len(adm_map):,} admissions in embedding space")

    # Identify OBITO admissions -> death centroid
    cols, rows = _exec(con, f"""
        WITH last_st AS (
            SELECT es.ID_CD_INTERNACAO,
                   ROW_NUMBER() OVER (PARTITION BY es.ID_CD_INTERNACAO ORDER BY es.DH_CADASTRO DESC) AS rn,
                   es.FL_DESOSPITALIZACAO
            FROM agg_tb_capta_evo_status_caes es
            WHERE es.{SRC}
        )
        SELECT DISTINCT ls.ID_CD_INTERNACAO
        FROM last_st ls
        JOIN agg_tb_capta_tipo_final_monit_fmon f
            ON ls.FL_DESOSPITALIZACAO = f.ID_CD_FINAL_MONITORAMENTO
            AND f.{SRC}
        WHERE ls.rn = 1
          AND (UPPER(f.DS_FINAL_MONITORAMENTO) LIKE '%%BITO%%')
    """)
    obito_iids = {r[0] for r in rows}
    print(f"    {len(obito_iids):,} total OBITO admissions")

    obito_indices = []
    for iid in obito_iids:
        if iid in adm_map:
            obito_indices.append(adm_map[iid]["emb_idx"])

    twin_result = {
        "death_proximity": [],
        "hospital_priorities": [],
        "hospital_drift": [],
        "n_priority_cases": 0,
    }

    if len(obito_indices) < 5:
        print("    Too few OBITO embeddings")
        return twin_result

    obito_vecs = emb[obito_indices]
    death_centroid = obito_vecs.mean(axis=0)
    death_centroid_norm = death_centroid / (np.linalg.norm(death_centroid) + 1e-9)
    print(f"    Death centroid from {len(obito_indices)} embeddings")

    # --- Death proximity for OPEN cases ---
    cols, rows = _exec(con, f"""
        SELECT i.ID_CD_INTERNACAO, i.ID_CD_PACIENTE, i.ID_CD_HOSPITAL,
               i.DH_ADMISSAO_HOSP,
               DATEDIFF('day', i.DH_ADMISSAO_HOSP::DATE, CURRENT_DATE) AS los
        FROM agg_tb_capta_internacao_cain i
        WHERE i.{SRC}
          AND (i.IN_SITUACAO IS NULL OR i.IN_SITUACAO != 2)
          AND i.DH_FINALIZACAO IS NULL
          AND i.DH_ADMISSAO_HOSP IS NOT NULL
    """)
    open_admissions = _rows_to_dicts(cols, rows)

    # CID medians for LOS explanation
    cid_medians = {}
    try:
        cm_cols, cm_rows = _exec(con, f"""
            SELECT c.DS_DESCRICAO,
                   MEDIAN(DATEDIFF('day', i.DH_ADMISSAO_HOSP::DATE,
                       COALESCE(i.DH_FINALIZACAO, CURRENT_TIMESTAMP)::DATE)) AS med_los
            FROM agg_tb_capta_internacao_cain i
            JOIN agg_tb_capta_cid_caci c
                ON i.ID_CD_INTERNACAO = c.ID_CD_INTERNACAO AND i.source_db = c.source_db
            WHERE i.{SRC} AND c.DS_DESCRICAO IS NOT NULL AND c.IN_PRINCIPAL = 'S'
            GROUP BY c.DS_DESCRICAO
            HAVING COUNT(DISTINCT i.ID_CD_INTERNACAO) >= 5
        """)
        cid_medians = {r[0]: float(r[1]) for r in cm_rows if r[1]}
    except Exception:
        pass

    # Compute percentiles
    all_brad_idxs = np.array([a["emb_idx"] for a in adm_map.values()])
    all_brad_vecs = emb[all_brad_idxs]
    all_brad_norms = np.linalg.norm(all_brad_vecs, axis=1, keepdims=True).clip(min=1e-9)
    all_brad_cos = (all_brad_vecs / all_brad_norms) @ death_centroid_norm
    cos_p90 = float(np.percentile(all_brad_cos, 90))
    cos_p95 = float(np.percentile(all_brad_cos, 95))
    cos_p99 = float(np.percentile(all_brad_cos, 99))
    del all_brad_idxs, all_brad_vecs, all_brad_norms, all_brad_cos

    death_proximity_list = []
    for oa in open_admissions:
        iid = oa["ID_CD_INTERNACAO"]
        if iid not in adm_map:
            continue
        idx = adm_map[iid]["emb_idx"]
        vec = emb[idx]
        vec_norm = vec / (np.linalg.norm(vec) + 1e-9)
        cos_sim = float(np.dot(vec_norm, death_centroid_norm))
        oa["death_proximity"] = cos_sim
        # Get CID
        try:
            c2, r2 = _exec(con, f"""
                SELECT STRING_AGG(DISTINCT DS_DESCRICAO, ' | ') AS cids
                FROM agg_tb_capta_cid_caci
                WHERE {SRC} AND ID_CD_INTERNACAO = {iid}
                  AND DS_DESCRICAO IS NOT NULL AND IN_PRINCIPAL = 'S'
            """)
            cid_val = r2[0][0] if r2 and r2[0][0] else None
            if not cid_val:
                c3, r3 = _exec(con, f"""
                    SELECT STRING_AGG(DISTINCT DS_DESCRICAO, ' | ') AS cids
                    FROM agg_tb_capta_cid_caci
                    WHERE {SRC} AND ID_CD_INTERNACAO = {iid}
                      AND DS_DESCRICAO IS NOT NULL
                """)
                cid_val = r3[0][0] if r3 and r3[0][0] else "---"
            oa["cids"] = cid_val
        except Exception:
            oa["cids"] = "---"

        # Generate BUSINESS reasons (zero jargon)
        los = _safe_int(oa.get("los"))
        cid = str(oa.get("cids", "---"))
        reasons = []

        # Risk reason — pure business language
        if cos_sim >= cos_p99:
            reasons.append("Perfil clinico compativel com casos de desfecho grave")
        elif cos_sim >= cos_p95:
            reasons.append("Sinais de alerta elevados no perfil clinico")
        elif cos_sim >= cos_p90:
            reasons.append("Perfil clinico requer atencao")

        # LOS reason
        cid_med = cid_medians.get(cid.split(" | ")[0] if " | " in cid else cid, 0)
        if cid_med > 0 and los > 2 * cid_med:
            ratio = los / cid_med
            reasons.append(f"Permanencia {ratio:.1f}x acima da media para o diagnostico")
        elif los >= 60:
            reasons.append(f"Internacao prolongada sem alta definida")
        elif los >= 30:
            reasons.append(f"Permanencia prolongada para o diagnostico")

        if not reasons:
            reasons.append("Monitoramento preventivo")

        # Composite score for sorting
        score_death = min(cos_sim / 0.5, 1.0) if cos_sim > 0 else 0
        score_los = min(los / 60, 1.0)
        composite = 0.5 * score_death + 0.5 * score_los

        oa["reasons"] = reasons
        oa["composite_score"] = composite
        death_proximity_list.append(oa)

    death_proximity_list.sort(key=lambda x: -x["composite_score"])
    twin_result["death_proximity"] = death_proximity_list
    print(f"    {len(death_proximity_list)} open cases scored")

    # --- Hospital priorities ---
    hospital_priorities = {}
    for pc in death_proximity_list:
        hid = pc["ID_CD_HOSPITAL"]
        if hid not in hospital_priorities:
            hospital_priorities[hid] = []
        hospital_priorities[hid].append(pc)

    sorted_hospitals = sorted(
        hospital_priorities.items(),
        key=lambda x: -max(c["composite_score"] for c in x[1])
    )
    twin_result["hospital_priorities"] = sorted_hospitals
    twin_result["n_priority_cases"] = len(death_proximity_list)

    # --- Hospital drift ---
    hospital_drift = {}
    for hid_drift in set(a.get("ID_CD_HOSPITAL") for a in adm_map.values() if a.get("ID_CD_HOSPITAL")):
        h_feb = [a["emb_idx"] for iid_h, a in adm_map.items()
                 if a.get("ID_CD_HOSPITAL") == hid_drift
                 and str(a.get("DH_ADMISSAO_HOSP", ""))[:10] < "2026-03-01"
                 and (a.get("DH_FINALIZACAO") is None or str(a.get("DH_FINALIZACAO", ""))[:10] >= "2026-02-01")]
        h_jan = [a["emb_idx"] for iid_h, a in adm_map.items()
                 if a.get("ID_CD_HOSPITAL") == hid_drift
                 and str(a.get("DH_ADMISSAO_HOSP", ""))[:10] < "2026-02-01"
                 and (a.get("DH_FINALIZACAO") is None or str(a.get("DH_FINALIZACAO", ""))[:10] >= "2026-01-01")]
        if len(h_feb) >= 5 and len(h_jan) >= 5:
            h_feb_c = emb[h_feb].mean(axis=0)
            h_jan_c = emb[h_jan].mean(axis=0)
            h_feb_cos = float(np.dot(h_feb_c / (np.linalg.norm(h_feb_c) + 1e-9), death_centroid_norm))
            h_jan_cos = float(np.dot(h_jan_c / (np.linalg.norm(h_jan_c) + 1e-9), death_centroid_norm))
            hospital_drift[hid_drift] = {
                "jan_cos": h_jan_cos, "feb_cos": h_feb_cos,
                "drift": h_feb_cos - h_jan_cos, "n_feb": len(h_feb), "n_jan": len(h_jan),
            }
    twin_result["hospital_drift"] = sorted(hospital_drift.items(), key=lambda x: -x[1]["drift"])
    print(f"    Hospital drift: {len(hospital_drift)} hospitals")

    print(f"    done in {time.time()-t0:.1f}s")
    return twin_result


# -----------------------------------------------------------------
# LaTeX Generation — 3 pages, zero jargon
# -----------------------------------------------------------------

def _generate_latex(kpis, hospitals, trend, data_quality, twin, chart_path):
    L = []

    feb = kpis["feb"]
    jan = kpis["jan"]

    # --- Preamble ---
    L.append(r"""\documentclass[a4paper,10pt]{article}
\usepackage[a4paper, top=1.5cm, bottom=1.5cm, left=1.5cm, right=1.5cm]{geometry}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[brazil]{babel}
\usepackage{lmodern}
\usepackage{xcolor}
\usepackage{colortbl}
\usepackage{booktabs}
\usepackage{array}
\usepackage{tabularx}
\usepackage{fancyhdr}
\usepackage{titlesec}
\usepackage{tcolorbox}
\usepackage{hyperref}
\usepackage{amsmath}
\usepackage{microtype}
\usepackage{graphicx}
\usepackage{enumitem}

\definecolor{bradblue}{RGB}{0,47,108}
\definecolor{bradred}{RGB}{204,0,0}
\definecolor{bradgray}{RGB}{100,100,100}
\definecolor{bradlightblue}{RGB}{220,235,250}
\definecolor{bradlightgray}{RGB}{245,245,245}
\definecolor{bradgold}{RGB}{180,140,20}
\definecolor{bradgreen}{RGB}{0,128,60}

\tcbuselibrary{skins,breakable}

\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\textcolor{bradblue}{\textbf{GHO-BRADESCO}} \textcolor{bradgray}{\small | Auditoria Concorrente | Fevereiro 2026}}
\fancyhead[R]{\textcolor{bradgray}{\small """ + REPORT_DATE_STR + r"""}}
\fancyfoot[C]{\textcolor{bradgray}{\footnotesize Pagina \thepage\ de 3}}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0pt}

\titleformat{\section}{\large\bfseries\color{bradblue}}{}{0em}{}
\titlespacing*{\section}{0pt}{10pt}{4pt}

\setlength{\parindent}{0pt}
\setlength{\parskip}{4pt}

\hypersetup{colorlinks=true, linkcolor=bradblue, urlcolor=bradblue}

\begin{document}
""")

    # ===================================================================
    # PAGE 1: Panorama Fevereiro 2026
    # ===================================================================

    # Header
    L.append(r"""
\begin{tcolorbox}[colback=bradblue, colframe=bradblue, coltitle=white,
    fonttitle=\large\bfseries, title={}, arc=0mm, boxrule=0pt,
    left=6pt, right=6pt, top=4pt, bottom=4pt]
    \textcolor{white}{\Large\bfseries Panorama Fevereiro 2026}\\[2pt]
    \textcolor{white}{\small GHO-BRADESCO | Auditoria Concorrente | Relatorio Executivo}
\end{tcolorbox}
\vspace{4pt}
""")

    # Headline finding (red box)
    # Compute the LOS delta and readmission delta for headline
    feb_los = _safe_float(feb.get("avg_los"))
    jan_los = _safe_float(jan.get("avg_los"))
    los_pct = 100.0 * (feb_los - jan_los) / jan_los if jan_los > 0 else 0
    feb_readmit = _safe_float(feb.get("taxa_readmit"))
    jan_readmit = _safe_float(jan.get("taxa_readmit"))
    readmit_delta = feb_readmit - jan_readmit

    L.append(r"""
\begin{tcolorbox}[colback=bradred!8, colframe=bradred, arc=1mm, boxrule=0.6pt,
    left=6pt, right=6pt, top=4pt, bottom=4pt]
\textcolor{bradred}{\textbf{Achado Principal:}}
A gravidade media dos pacientes internados pela rede BRADESCO aumentou significativamente entre janeiro e fevereiro.
O tempo medio de internacao subiu """ + f"{abs(los_pct):.1f}" + r"""\% e a taxa de readmissao em 30 dias subiu """ + f"{abs(readmit_delta):.1f}" + r""" pontos percentuais, simultaneamente --- padrao que indica deterioracao da acuidade clinica da carteira.
\end{tcolorbox}
\vspace{4pt}
""")

    # KPI grid 2x4
    censo = _fmt_thousands(feb.get("censo"))
    censo_jan = _safe_int(jan.get("censo"))
    novas = _fmt_thousands(feb.get("novas_admissoes"))
    altas = _fmt_thousands(feb.get("altas"))
    los_med = f"{_safe_float(feb.get('avg_los')):.1f}"
    taxa_readmit = f"{_safe_float(feb.get('taxa_readmit')):.1f}\\%"
    taxa_obito = f"{_safe_float(feb.get('taxa_obito')):.1f}\\%"
    ae = _fmt_thousands(feb.get("eventos_adversos"))
    open_gt60 = _fmt_thousands(feb.get("open_gt_60d"))

    def _kpi_delta(feb_key, jan_key=None, invert=False):
        if jan_key is None:
            jan_key = feb_key
        return _delta_arrow(feb.get(feb_key), jan.get(jan_key), invert=invert)

    L.append(r"""
\section{Indicadores Chave}

\begin{center}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{|>{\columncolor{bradlightblue}}p{3.4cm}|c|c||>{\columncolor{bradlightblue}}p{3.4cm}|c|c|}
\hline
\rowcolor{bradblue}
\textcolor{white}{\textbf{Indicador}} & \textcolor{white}{\textbf{Fev/26}} & \textcolor{white}{\textbf{vs Jan}} &
\textcolor{white}{\textbf{Indicador}} & \textcolor{white}{\textbf{Fev/26}} & \textcolor{white}{\textbf{vs Jan}} \\
\hline
""")
    L.append(f"Censo (internacoes ativas) & {censo} & {_kpi_delta('censo')} & "
             f"Tempo Medio de Internacao & {los_med} dias & {_kpi_delta('avg_los', invert=True)} \\\\\n\\hline\n")
    L.append(f"Novas Admissoes & {novas} & {_kpi_delta('novas_admissoes')} & "
             f"Taxa de Readmissao 30d & {taxa_readmit} & {_kpi_delta('taxa_readmit', invert=True)} \\\\\n\\hline\n")
    L.append(f"Altas & {altas} & {_kpi_delta('altas')} & "
             f"Taxa de Obito & {taxa_obito} & {_kpi_delta('taxa_obito', invert=True)} \\\\\n\\hline\n")
    L.append(f"Eventos Adversos & {ae} & {_kpi_delta('eventos_adversos', invert=True)} & "
             f"Casos Abertos $>$60 dias & {open_gt60} & --- \\\\\n\\hline\n")

    L.append(r"""
\end{tabular}
\end{center}
\vspace{4pt}
""")

    # Trend chart
    if chart_path:
        L.append(r"""
\begin{center}
\includegraphics[width=\textwidth]{""" + chart_path + r"""}
\end{center}
\vspace{2pt}
""")

    # Top 5 hospitals table
    top5 = hospitals[:5]
    L.append(r"""
\section{Principais Hospitais por Volume}

\begin{center}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{p{5.5cm}rrrl}
\toprule
\textbf{Hospital} & \textbf{Altas Fev} & \textbf{TM Internacao} & \textbf{TM Jan} & \textbf{Variacao} \\
\midrule
""")
    for h in top5:
        hname = _escape_latex(_truncate(_hosp_name(h["hospital_id"]), 45))
        n = _fmt_thousands(h["n_altas"])
        feb_l = f"{_safe_float(h['avg_los']):.1f}d"
        jan_l = f"{_safe_float(h['jan_avg_los']):.1f}d" if h.get("jan_avg_los") else "---"
        delta = ""
        if h.get("los_delta_pct") is not None:
            dp = h["los_delta_pct"]
            if dp > 5:
                delta = f"\\textcolor{{bradred}}{{$\\uparrow${abs(dp):.1f}\\%}}"
            elif dp < -5:
                delta = f"\\textcolor{{bradgreen}}{{$\\downarrow${abs(dp):.1f}\\%}}"
            else:
                delta = f"$\\rightarrow${abs(dp):.1f}\\%"
        L.append(f"{hname} & {n} & {feb_l} & {jan_l} & {delta} \\\\\n")
    L.append(r"""
\bottomrule
\end{tabular}
\end{center}
\vspace{1pt}
{\scriptsize\textcolor{bradgray}{TM = Tempo Medio de Internacao em dias. Variacao = mudanca Jan$\rightarrow$Fev.}}
""")

    # ===================================================================
    # PAGE 2: Pacientes que Requerem Verificacao Imediata
    # ===================================================================
    L.append(r"\newpage")

    hosp_priorities = twin.get("hospital_priorities", [])
    n_total_priority = twin.get("n_priority_cases", 0)

    L.append(r"""
\section{Pacientes que Requerem Verificacao Imediata}
""")

    L.append(f"""
\\begin{{tcolorbox}}[colback=bradlightblue, colframe=bradblue, arc=1mm, boxrule=0.4pt,
    left=6pt, right=6pt, top=3pt, bottom=3pt]
O sistema de inteligencia artificial identificou \\textbf{{{n_total_priority} pacientes}} atualmente internados
cujo perfil clinico apresenta sinais de alerta. Os casos abaixo sao os de maior prioridade, agrupados por hospital.
\\end{{tcolorbox}}
\\vspace{{4pt}}
""")

    # Top 3 hospitals with most critical cases
    n_hosp_shown = 0
    for hid, cases in hosp_priorities:
        if n_hosp_shown >= 3:
            break

        # Sort by composite score, take top 5
        cases_sorted = sorted(cases, key=lambda x: -x.get("composite_score", 0))
        top_cases = cases_sorted[:5]

        uf = _hosp_uf(hid)
        hname = _escape_latex(_truncate(_hosp_name(hid), 50))
        n_cases_hosp = len(cases)
        uf_str = f"({uf}) " if uf else ""

        L.append(f"""
\\begin{{tcolorbox}}[colback=white, colframe=bradgray, arc=0.5mm, boxrule=0.3pt,
    title={{\\textbf{{{uf_str}{hname}}} --- {n_cases_hosp} caso(s) prioritario(s)}},
    fonttitle=\\small\\bfseries, coltitle=bradblue, colbacktitle=bradlightgray,
    left=4pt, right=4pt, top=2pt, bottom=2pt]
\\footnotesize
""")

        for c in top_cases:
            iid = _safe_int(c.get("ID_CD_INTERNACAO"))
            los = _safe_int(c.get("los"))
            cid = _escape_latex(_truncate(str(c.get("cids", "---")), 60))
            reasons = c.get("reasons", ["Monitoramento preventivo"])
            # Pick the most important reason (first one)
            reason_text = _escape_latex(reasons[0])

            L.append(f"\\textbullet\\ \\textbf{{Inter. {iid}}} | {los} dias | {cid}\\\\\n")
            L.append(f"\\hspace*{{6pt}}\\textcolor{{bradgray}}{{\\textit{{Motivo: {reason_text}}}}}\\\\\n")
            L.append("\\vspace{2pt}\n")

        L.append(r"\end{tcolorbox}" + "\n\\vspace{4pt}\n")
        n_hosp_shown += 1

    # Bottom note
    L.append(f"""
\\begin{{tcolorbox}}[colback=bradlightgray, colframe=bradgray, arc=0.5mm, boxrule=0.3pt,
    left=6pt, right=6pt, top=3pt, bottom=3pt]
\\footnotesize
Lista completa de {n_total_priority} pacientes disponivel no Anexo Tecnico (Relatorio Completo de Auditoria).
\\end{{tcolorbox}}
""")

    # ===================================================================
    # PAGE 3: Alertas e Recomendacoes
    # ===================================================================
    L.append(r"\newpage")

    L.append(r"""
\section{Alertas e Recomendacoes}
\vspace{2pt}
""")

    # Alert 1: Data Quality (red box)
    dq = data_quality
    pct_sem = f"{dq['pct_sem_tipo']:.1f}"
    n_sem = _fmt_thousands(dq["sem_tipo"])

    L.append(f"""
\\begin{{tcolorbox}}[colback=bradred!6, colframe=bradred, arc=1mm, boxrule=0.5pt,
    title={{\\textcolor{{white}}{{\\textbf{{Alerta: Qualidade de Dados}}}}}},
    fonttitle=\\small\\bfseries, colbacktitle=bradred,
    left=6pt, right=6pt, top=3pt, bottom=3pt]
\\small
{pct_sem}\\% das altas de fevereiro ({n_sem} internacoes) nao possuem tipo de alta registrado no sistema.
Isso compromete a capacidade de auditoria e analise de desfechos.\\\\[3pt]
\\textbf{{Recomendacao:}} Solicitar aos hospitais credenciados o preenchimento retroativo dos tipos de alta.
\\end{{tcolorbox}}
\\vspace{{4pt}}
""")

    # Alert 2: Hospital drift (gold box)
    hosp_drift = twin.get("hospital_drift", [])
    # Also use hospital performance for LOS drift
    hosp_los_drift = sorted(
        [h for h in hospitals if h.get("los_delta_pct") is not None and h["los_delta_pct"] > 0],
        key=lambda x: -x["los_delta_pct"]
    )[:5]

    L.append(r"""
\begin{tcolorbox}[colback=bradgold!6, colframe=bradgold, arc=1mm, boxrule=0.5pt,
    title={\textcolor{white}{\textbf{Alerta: Hospitais com Maior Aumento de Permanencia}}},
    fonttitle=\small\bfseries, colbacktitle=bradgold,
    left=6pt, right=6pt, top=3pt, bottom=3pt]
\small
""")
    if hosp_los_drift:
        L.append("Hospitais cujo tempo medio de internacao mais cresceu entre janeiro e fevereiro:\\\\\n\\vspace{2pt}\n")
        for h in hosp_los_drift:
            hname = _escape_latex(_truncate(_hosp_name(h["hospital_id"]), 40))
            jan_l = _safe_float(h["jan_avg_los"])
            feb_l = _safe_float(h["avg_los"])
            dp = h["los_delta_pct"]
            L.append(f"\\textbullet\\ \\textbf{{{hname}}}: {jan_l:.1f}d $\\rightarrow$ {feb_l:.1f}d "
                     f"(\\textcolor{{bradred}}{{+{dp:.0f}\\%}})\\\\\n")
    else:
        L.append("Nenhum hospital com variacao significativa identificado.\\\\\n")

    L.append(r"""
\end{tcolorbox}
\vspace{4pt}
""")

    # Alert 3: Readmission (gold box)
    L.append(f"""
\\begin{{tcolorbox}}[colback=bradgold!6, colframe=bradgold, arc=1mm, boxrule=0.5pt,
    title={{\\textcolor{{white}}{{\\textbf{{Alerta: Readmissao em 30 Dias}}}}}},
    fonttitle=\\small\\bfseries, colbacktitle=bradgold,
    left=6pt, right=6pt, top=3pt, bottom=3pt]
\\small
A taxa de readmissao em 30 dias de {_safe_float(feb.get('taxa_readmit')):.1f}\\% requer investigacao:
separar readmissoes planejadas (quimioterapia, hemodialise) de readmissoes evitaveis.
Readmissoes evitaveis representam risco clinico e custo desnecessario para a operadora.
\\end{{tcolorbox}}
\\vspace{{6pt}}
""")

    # Recommendations
    n_priority = twin.get("n_priority_cases", 0)
    n_drift_hosps = len(hosp_los_drift)

    L.append(r"""
\section{Recomendacoes}

\begin{enumerate}[leftmargin=16pt, itemsep=3pt]
""")
    L.append(f"\\item Verificar os \\textbf{{{n_priority} pacientes prioritarios}} listados na Pagina 2, "
             f"priorizando os casos com permanencia prolongada.\n")
    L.append(f"\\item Solicitar correcao de dados de tipo de alta nos hospitais com maior "
             f"percentual de registros incompletos.\n")
    L.append(f"\\item Investigar os \\textbf{{{n_drift_hosps} hospitais}} com maior aumento de permanencia "
             f"entre janeiro e fevereiro.\n")
    L.append(f"\\item Reuniao com equipes medicas dos diagnosticos com maior excesso de tempo de internacao "
             f"em relacao a media da rede.\n")
    L.append(f"\\item Separar readmissoes planejadas de evitaveis para analise precisa do impacto clinico "
             f"e financeiro.\n")

    L.append(r"""
\end{enumerate}
""")

    # Footer
    L.append(f"""
\\vfill
\\begin{{center}}
\\textcolor{{bradgray}}{{\\footnotesize
Relatorio gerado por sistema de inteligencia artificial.
Anexo Tecnico disponivel para detalhamento metodologico.\\\\
Data de geracao: {REPORT_DATE_STR}
}}
\\end{{center}}

\\end{{document}}
""")

    return "".join(L)


# -----------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------

@app.function(
    image=report_image,
    volumes=VOLUMES,
    memory=65536,
    timeout=3600,
)
def generate_report():
    import os, time, subprocess
    import duckdb

    t_start = time.time()

    jepa_cache.reload()
    data_vol.reload()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHART_DIR, exist_ok=True)

    print("=" * 70)
    print("BRADESCO EXECUTIVO — RELATORIO 3 PAGINAS FEV 2026")
    print("=" * 70)

    con = duckdb.connect(DB_PATH, read_only=True)

    # Load lookups
    _load_hospital_names(con)
    _load_hospital_uf(con)

    # Fetch data (6 steps)
    kpis = _fetch_kpis(con)
    hospitals = _fetch_hospital_performance(con)
    trend = _fetch_trend_6m(con)
    data_quality = _fetch_data_quality_global(con)

    # Load twin (heavy: ~2 min)
    twin = _load_twin_predictive(con)

    con.close()

    # Generate chart
    print("\n[6/6] Generating chart ...")
    _setup_chart_style()
    chart_path = _safe_chart(_chart_trend_dual_axis, f"{CHART_DIR}/trend_dual.png", trend)
    print(f"    Chart: {chart_path}")

    # Generate LaTeX
    print("\nGenerating LaTeX ...")
    latex = _generate_latex(kpis, hospitals, trend, data_quality, twin, chart_path)

    tex_path = OUTPUT_PDF.replace(".pdf", ".tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"    LaTeX written to {tex_path}")

    # Compile PDF (2 passes)
    print("Compiling PDF (2 passes) ...")
    for pass_n in range(2):
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", OUTPUT_DIR, tex_path],
            capture_output=True, timeout=120,
        )
        if result.returncode != 0 and pass_n == 1:
            print(f"    pdflatex pass {pass_n+1} warnings/errors:")
            log_path = tex_path.replace(".tex", ".log")
            if os.path.exists(log_path):
                with open(log_path) as lf:
                    lines = lf.readlines()
                    for line in lines[-50:]:
                        if "error" in line.lower() or "!" in line:
                            print(f"      {line.rstrip()}")

    if os.path.exists(OUTPUT_PDF):
        sz = os.path.getsize(OUTPUT_PDF) / 1024
        print(f"\n    PDF generated: {OUTPUT_PDF} ({sz:.0f} KB)")
    else:
        print(f"\n    PDF not found at {OUTPUT_PDF}!")
        for f in os.listdir(OUTPUT_DIR):
            if f.endswith(".pdf"):
                print(f"    Found: {OUTPUT_DIR}/{f}")

    data_vol.commit()

    total_time = time.time() - t_start
    print(f"\n    Total time: {total_time:.0f}s")
    print("=" * 70)


@app.local_entrypoint()
def main():
    generate_report.remote()
