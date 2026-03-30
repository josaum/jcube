#!/usr/bin/env python3
"""
Modal script: JCUBE V6.2 — Retrospectiva Completa Fevereiro 2026 GHO-BRADESCO

Executive-first, decision-oriented report for the GHO-BRADESCO auditor/operator.
BRADESCO has NO billing data (fatura_fatu, itens_fait, glosa_fatg are empty).
Financial proxy: capta_produtos_capr (medications).
Hospital names from agg_tb_capta_add_hospitais_caho (88 hospitals with UF+name).

NEW SECTION ORDER (v2 — addressing 3 analytical design flaws):
  1.  Capa + Resumo Executivo (KPIs with MoM deltas)
  2.  Alerta Preditivo: Internacoes Abertas em Risco (TWIN-DRIVEN)
  3.  Desempenho por Hospital (ranking with MoM deltas)
  4.  Matriz de Risco CID x Hospital (cross-dimensional)
  5.  Tendencia 6 Meses (ALL KPIs monthly)
  6.  Perfil Clinico + LOS Analysis
  7.  Variacao por Medico (with CID-adjusted comparison)
  8.  Eventos Adversos
  9.  Medicamentos
  10. Nota Metodologica

Fixes from v1 retained:
  - tipo_final_monit_fmon filtered by source_db = 'GHO-BRADESCO' (avoids cartesian)
  - Physician mortality is per-internacao for THAT doctor
  - UTI filtered to Feb-active internacoes, days clamped to Feb range
  - CRM exclude '0000%' patterns
  - Discharge types grouped into categories
  - CID uses IN_PRINCIPAL = 'S'
  - No references to billing/faturamento/glosa

New in v2:
  - FLAW 1: Every metric has temporal context (MoM deltas, 6-month full KPI trend)
  - FLAW 2: Cross-dimensional analysis (CID x Hospital risk matrix, Doctor x CID LOS,
             Medication intensity x CID x Hospital)
  - FLAW 3: Twin as predictive spine — death proximity score, trajectory velocity,
             March projection. Section 2, not appendix.

Usage:
    modal run reports/modal_bradesco_fev2026.py
    modal run --detach reports/modal_bradesco_fev2026.py
"""
from __future__ import annotations

import modal

app = modal.App("jcube-bradesco-fev2026")

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
        "seaborn>=0.13",
    )
)

# -----------------------------------------------------------------
# Constants
# -----------------------------------------------------------------

GRAPH_PARQUET = "/data/jcube_graph.parquet"
WEIGHTS_PATH  = "/cache/tkg-v6.2/node_emb_epoch_3.pt"
DB_PATH       = "/data/aggregated_fixed_union.db"
OUTPUT_DIR    = "/data/reports"
OUTPUT_PDF    = f"{OUTPUT_DIR}/bradesco_fev2026_retrospectiva.pdf"

SOURCE_DB       = "GHO-BRADESCO"
SRC             = f"source_db = '{SOURCE_DB}'"
FEB_START       = "2026-02-01"
FEB_END         = "2026-02-28"
JAN_START       = "2026-01-01"
JAN_END         = "2026-01-31"
REPORT_DATE_STR = "29/03/2026"

Z_THRESHOLD     = 2.0
TOP_ANOMALIES   = 20

# 6 months for trend: Sep 2025 -> Feb 2026
TREND_MONTHS = [
    ("2025-09-01", "2025-09-30", "Set/25"),
    ("2025-10-01", "2025-10-31", "Out/25"),
    ("2025-11-01", "2025-11-30", "Nov/25"),
    ("2025-12-01", "2025-12-31", "Dez/25"),
    ("2026-01-01", "2026-01-31", "Jan/26"),
    ("2026-02-01", "2026-02-28", "Fev/26"),
]

# Discharge type grouping — maps DS_FINAL_MONITORAMENTO to 6 executive categories.
# Based on the 24 types in BRADESCO's tipo_final_monit_fmon table.
DISCHARGE_GROUP_MAP = {
    # ALTA NORMAL — desfecho esperado
    "Alta hospitalar melhorada":     "ALTA NORMAL",
    "Alta hospitalar":               "ALTA NORMAL",
    "Alta hospitalar complexa":      "ALTA COMPLEXA",
    "Alta hospitalar complexa: Recurso Projeto PAD Pont": "ALTA COMPLEXA",
    # OBITO
    "Óbito":                         "OBITO",
    "Alta da mãe/puérpera com óbito fetal": "OBITO",
    # TRANSFERENCIA
    "Transferência para outra Unidade Hospitalar": "TRANSFERENCIA",
    "Transferência para outro Hospital": "TRANSFERENCIA",
    "Transferência para Hospital de Retaguarda": "TRANSFERENCIA",
    "Transferência para Clínica de Transição": "TRANSFERENCIA",
    "Hospital de Retaguarda":        "TRANSFERENCIA",
    # NEONATAL / MATERNO
    "Alta da mãe/puérpera e do RN":  "NEONATAL",
    "Alta da mãe/puérpera e permanência do RN": "NEONATAL",
    "Alta RN para cadastro próprio": "NEONATAL",
    "Perinatal - transferência leito UTI": "NEONATAL",
    # ADMINISTRATIVO
    "Alta Administrativa":           "ADMINISTRATIVO",
    "Alta a pedido":                 "ADMINISTRATIVO",
    "Alta por Evasão":               "ADMINISTRATIVO",
    "Finalização de pré monitoramento": "ADMINISTRATIVO",
    "Finalização de pré-monitoramento": "ADMINISTRATIVO",
    # PROGRAMAS ESPECIAIS
    "Indicação de Home Care":        "PROGRAMA ESPECIAL",
    "GoCare":                        "PROGRAMA ESPECIAL",
    "Cuidar":                        "PROGRAMA ESPECIAL",
    "Programa Superação":            "PROGRAMA ESPECIAL",
    "Programa Doce Vida":            "PROGRAMA ESPECIAL",
    "Programa Inspirar":             "PROGRAMA ESPECIAL",
    "VIVApreparação":                "PROGRAMA ESPECIAL",
    "Clínica Especializada":         "PROGRAMA ESPECIAL",
    "Clínica Especializada (Oncologia)": "PROGRAMA ESPECIAL",
    "Clínica Especializada (Hemodiálise)": "PROGRAMA ESPECIAL",
    "Consultório Médico Cooperado":  "PROGRAMA ESPECIAL",
    "Centro de Infusão":             "PROGRAMA ESPECIAL",
    "Medicina Preventiva - NAS":     "PROGRAMA ESPECIAL",
    # STATUS INTERMEDIARIO (não deveria ser alta, mas aparece com IN_SITUACAO=2)
    "Monitoramento em andamento":    "SEM TIPO DE ALTA DEFINIDO",
    "Manter Monitoramento":          "SEM TIPO DE ALTA DEFINIDO",
    "Manter monitoramento":          "SEM TIPO DE ALTA DEFINIDO",
    "Provável Alta Complexa":        "SEM TIPO DE ALTA DEFINIDO",
    "Instável":                      "SEM TIPO DE ALTA DEFINIDO",
}


def _group_discharge_type(tipo_raw):
    """Map raw discharge type text to executive category."""
    if not tipo_raw:
        return "SEM TIPO DE ALTA DEFINIDO"
    s = str(tipo_raw).strip()
    # Direct lookup first
    if s in DISCHARGE_GROUP_MAP:
        return DISCHARGE_GROUP_MAP[s]
    # Keyword fallback
    up = s.upper()
    if "BITO" in up:
        return "OBITO"
    if "TRANSFER" in up:
        return "TRANSFERENCIA"
    if "ALTA" in up and ("MELHORAD" in up or "SIMPLES" in up or "HOSPITALAR" in up):
        return "ALTA NORMAL"
    if "EVAS" in up or "ADMINIST" in up or "PEDIDO" in up:
        return "ADMINISTRATIVO"
    return "OUTRO"


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


def _delta_arrow(new_val, old_val, fmt="float", invert=False):
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


_HOSP_NAMES: dict[int, str] = {}  # populated by _load_hospital_names()

def _load_hospital_names(con):
    """Load hospital name mapping from agg_tb_capta_add_hospitais_caho."""
    global _HOSP_NAMES
    cols, rows = _exec(con, f"""
        SELECT ID_CD_HOSPITAL, NM_HOSPITAL
        FROM agg_tb_capta_add_hospitais_caho
        WHERE source_db = '{SOURCE_DB}' AND NM_HOSPITAL IS NOT NULL
    """)
    _HOSP_NAMES = {int(r[0]): str(r[1]) for r in rows}
    print(f"    Loaded {len(_HOSP_NAMES)} hospital names")

def _hosp_name(hid):
    """Resolve hospital ID to name via caho lookup."""
    hid_int = _safe_int(hid)
    return _HOSP_NAMES.get(hid_int, f"Hospital {hid_int}")


# -----------------------------------------------------------------
# MORTALITY HELPER (reused across sections)
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
    """Count deaths in a given discharge period."""
    q = _MORTALITY_Q_TEMPLATE.format(src=SRC, m_start=m_start, m_end=m_end)
    _, rows = _exec(con, q)
    return _safe_int(rows[0][0]) if rows else 0


# -----------------------------------------------------------------
# Chart infrastructure
# -----------------------------------------------------------------

CHART_DIR = "/tmp/charts"

# Color palette matching LaTeX (RGB normalized to 0-1)
BRAD_COLORS = {
    "blue":      (0/255, 47/255, 108/255),
    "red":       (204/255, 0/255, 0/255),
    "gold":      (180/255, 140/255, 20/255),
    "green":     (0/255, 128/255, 60/255),
    "gray":      (100/255, 100/255, 100/255),
    "deathred":  (140/255, 0/255, 30/255),
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

def _latex_chart(charts, key, width=r"\textwidth"):
    path = charts.get(key) if charts else None
    if path is None:
        return ""
    return (
        f"\\begin{{center}}\n"
        f"\\includegraphics[width={width}]{{{path}}}\n"
        f"\\end{{center}}\n"
    )


# -----------------------------------------------------------------
# Chart functions (14 charts)
# -----------------------------------------------------------------

def _chart_kpi_gauges(path, kpis):
    """4 subplot semicircle donut gauges for key KPIs."""
    import matplotlib.pyplot as plt
    import numpy as np

    feb = kpis.get("feb", {})
    jan = kpis.get("jan", {})

    metrics = [
        ("Taxa Obito (%)", _safe_float(feb.get("taxa_obito")), 20.0, _safe_float(jan.get("taxa_obito")), True),
        ("Taxa Readmissao (%)", _safe_float(feb.get("taxa_readmit")), 30.0, _safe_float(jan.get("taxa_readmit")), True),
        ("LOS Medio (dias)", _safe_float(feb.get("avg_los")), 30.0, _safe_float(jan.get("avg_los")), True),
        ("Casos Abertos (n)", _safe_float(feb.get("open_cases")), max(_safe_float(feb.get("open_cases")) * 1.5, 100), _safe_float(jan.get("open_cases")), True),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(7.5, 2.8), subplot_kw={"aspect": "equal"})
    for ax, (title, val, max_val, jan_val, invert) in zip(axes, metrics):
        ratio = min(val / max_val, 1.0) if max_val > 0 else 0
        filled = ratio * 180
        empty = 180 - filled
        # Determine color
        if invert:
            color = BRAD_COLORS["red"] if ratio > 0.6 else (BRAD_COLORS["gold"] if ratio > 0.3 else BRAD_COLORS["green"])
        else:
            color = BRAD_COLORS["green"] if ratio > 0.6 else (BRAD_COLORS["gold"] if ratio > 0.3 else BRAD_COLORS["red"])

        theta1 = 0
        theta2 = filled
        theta3 = 180
        # Draw semicircle
        wedges = ax.pie(
            [filled, empty, 180],
            startangle=180,
            colors=[color, BRAD_COLORS["lightgray"], "white"],
            wedgeprops={"width": 0.35, "edgecolor": "white", "linewidth": 1},
            counterclock=False,
        )
        ax.set_title(title, fontsize=9, fontweight="bold", color=BRAD_COLORS["gray"], pad=4)
        # Center value — bigger font
        ax.text(0, 0.12, f"{val:.1f}", ha="center", va="center", fontsize=18, fontweight="bold", color=BRAD_COLORS["blue"])
        # MoM delta — bigger, bolder
        if jan_val > 0:
            delta = val - jan_val
            pct = 100.0 * delta / jan_val
            arrow = "\u2191" if delta > 0 else "\u2193"
            dcolor = BRAD_COLORS["red"] if (delta > 0 and invert) else (BRAD_COLORS["green"] if (delta <= 0 and invert) else BRAD_COLORS["gray"])
            ax.text(0, -0.25, f"{arrow} {pct:+.1f}% vs Jan", ha="center", va="center",
                    fontsize=8, fontweight="bold", color=dcolor,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor=dcolor, alpha=0.6, linewidth=0.5))
        else:
            ax.text(0, -0.25, "sem ref. Jan", ha="center", va="center", fontsize=7, color=BRAD_COLORS["gray"])

    fig.suptitle("KPIs Fevereiro 2026", fontsize=12, fontweight="bold", color=BRAD_COLORS["blue"], y=1.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def _chart_scatter_death_proximity(path, twin):
    """Scatter X=LOS, Y=death_proximity for open admissions.
    Rich version: risk zones, CID labels, interpretation guide."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    data = twin.get("death_proximity", [])
    if not data:
        return

    los_vals = np.array([_safe_float(d.get("los")) for d in data])
    cos_vals = np.array([_safe_float(d.get("death_proximity", d.get("cos_sim", 0))) for d in data])

    # Compute thresholds for risk zones
    cos_median = float(np.median(cos_vals))
    cos_p75 = float(np.percentile(cos_vals, 75))
    cos_p90 = float(np.percentile(cos_vals, 90))
    los_median = float(np.median(los_vals))

    # Top 5 hospitals by frequency
    from collections import Counter
    hids = [_safe_int(d.get("ID_CD_HOSPITAL")) for d in data]
    hosp_counts = Counter(hids)
    top5_hosps = [h for h, _ in hosp_counts.most_common(5)]

    fig, ax = plt.subplots(figsize=(8.5, 6))

    # ── Risk zone backgrounds ──
    x_max = max(los_vals) * 1.1
    y_min, y_max = min(cos_vals) * 0.98, max(cos_vals) * 1.02

    # Green zone: low LOS + low proximity
    ax.axhspan(y_min, cos_p75, xmin=0, xmax=1, alpha=0.06, color="green", zorder=0)
    # Yellow zone: medium proximity
    ax.axhspan(cos_p75, cos_p90, xmin=0, xmax=1, alpha=0.06, color="gold", zorder=0)
    # Red zone: high proximity
    ax.axhspan(cos_p90, y_max, xmin=0, xmax=1, alpha=0.08, color="red", zorder=0)

    # Reference lines
    ax.axhline(y=cos_median, color=BRAD_COLORS["gray"], linestyle="--", linewidth=0.8, alpha=0.5, zorder=1)
    ax.axhline(y=cos_p90, color=BRAD_COLORS["red"], linestyle="--", linewidth=0.8, alpha=0.6, zorder=1)
    ax.text(x_max * 0.98, cos_median, "Mediana", fontsize=6, color=BRAD_COLORS["gray"],
            ha="right", va="bottom", alpha=0.7)
    ax.text(x_max * 0.98, cos_p90, "Percentil 90", fontsize=6, color=BRAD_COLORS["red"],
            ha="right", va="bottom", alpha=0.7)

    # ── Plot points ──
    cmap = plt.cm.tab10
    for dp in data:
        x = _safe_float(dp.get("los"))
        y = _safe_float(dp.get("death_proximity", dp.get("cos_sim", 0)))
        hid = _safe_int(dp.get("ID_CD_HOSPITAL"))
        if hid in top5_hosps:
            cidx = top5_hosps.index(hid)
            color = cmap(cidx)
            ax.scatter(x, y, c=[color], s=50, alpha=0.85, edgecolors="white", linewidths=0.5, zorder=4)
        else:
            ax.scatter(x, y, c=[BRAD_COLORS["gray"]], s=25, alpha=0.35, zorder=3)

    # ── Label top 8 highest proximity with CID + hospital ──
    sorted_data = sorted(data, key=lambda d: -_safe_float(d.get("death_proximity", d.get("cos_sim", 0))))
    labeled_positions = []
    for dp in sorted_data[:8]:
        x = _safe_float(dp.get("los"))
        y = _safe_float(dp.get("death_proximity", dp.get("cos_sim", 0)))
        iid = _safe_int(dp.get("ID_CD_INTERNACAO"))
        cid = _truncate(str(dp.get("cid", dp.get("cids", "---"))), 40)
        hosp = _truncate(_hosp_name(dp.get("ID_CD_HOSPITAL")), 18)

        # Smart label positioning to avoid overlaps
        offset_x, offset_y = 8, 8
        for lx, ly in labeled_positions:
            if abs(x - lx) < 5 and abs(y - ly) < 0.005:
                offset_y += 12
        labeled_positions.append((x, y))

        label = f"{cid}\n{hosp} | {x:.0f}d"
        ax.annotate(label, (x, y), fontsize=5.5, fontweight="bold",
                    color=BRAD_COLORS["deathred"],
                    xytext=(offset_x, offset_y), textcoords="offset points",
                    arrowprops=dict(arrowstyle="-", color=BRAD_COLORS["gray"], lw=0.5),
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=BRAD_COLORS["gray"],
                              alpha=0.85, linewidth=0.5))

    # ── Legend for hospitals ──
    from matplotlib.lines import Line2D
    handles = []
    for i, hid in enumerate(top5_hosps):
        handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(i),
                              markersize=7, label=_truncate(_hosp_name(hid), 28)))
    handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=BRAD_COLORS["gray"],
                          markersize=5, label="Outros hospitais", alpha=0.5))
    ax.legend(handles=handles, fontsize=6, loc="lower right", framealpha=0.9,
              edgecolor=BRAD_COLORS["gray"])

    # ── Interpretation box ──
    interp_text = (
        "COMO LER: Cada ponto = 1 internacao aberta\n"
        "Eixo X = dias internado | Eixo Y = similaridade ao perfil de obito\n"
        "Zona vermelha (topo) = maior risco | Labels = casos prioritarios"
    )
    ax.text(0.02, 0.02, interp_text, transform=ax.transAxes, fontsize=6,
            verticalalignment="bottom", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=BRAD_COLORS["blue"], alpha=0.9, linewidth=0.8))

    # ── Zone labels on right margin ──
    ax.text(x_max * 1.01, (y_min + cos_p75) / 2, "BAIXO\nRISCO", fontsize=6,
            color="green", alpha=0.6, ha="left", va="center", fontweight="bold")
    ax.text(x_max * 1.01, (cos_p75 + cos_p90) / 2, "ATENCAO", fontsize=6,
            color=BRAD_COLORS["gold"], alpha=0.7, ha="left", va="center", fontweight="bold")
    ax.text(x_max * 1.01, (cos_p90 + y_max) / 2, "ALERTA", fontsize=6,
            color=BRAD_COLORS["red"], alpha=0.8, ha="left", va="center", fontweight="bold")

    ax.set_xlabel("Permanencia (dias)", fontsize=9)
    ax.set_ylabel("Proximidade ao Centroide de Obito (cosseno)", fontsize=9)
    ax.set_title("Mapa de Risco: Casos Abertos por Permanencia e Proximidade ao Obito",
                 fontsize=11, fontweight="bold", color=BRAD_COLORS["blue"])
    ax.set_xlim(left=0, right=x_max)
    ax.set_ylim(bottom=y_min, top=y_max)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def _chart_hospital_los_ranking(path, hospitals):
    """Horizontal bar chart of top 15 hospitals by volume, showing avg_los."""
    import matplotlib.pyplot as plt
    import numpy as np

    if not hospitals:
        return

    top = sorted(hospitals, key=lambda h: -_safe_int(h.get("n_altas")))[:15]
    top.reverse()  # bottom to top for horizontal bars

    names = [_truncate(_hosp_name(h.get("hospital_id")), 25) for h in top]
    los_vals = [_safe_float(h.get("avg_los")) for h in top]

    colors = []
    for v in los_vals:
        if v < 5:
            colors.append(BRAD_COLORS["green"])
        elif v <= 10:
            colors.append(BRAD_COLORS["gold"])
        else:
            colors.append(BRAD_COLORS["red"])

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    y_pos = np.arange(len(names))

    # Background color zones
    x_max = max(los_vals) * 1.3 if los_vals else 20
    ax.axvspan(0, 5, alpha=0.07, color="green", zorder=0)
    ax.axvspan(5, 10, alpha=0.07, color="gold", zorder=0)
    ax.axvspan(10, x_max, alpha=0.07, color="red", zorder=0)
    # Zone labels at top
    ax.text(2.5, len(names) + 0.2, "Adequado", fontsize=6, color="green", ha="center", fontweight="bold", alpha=0.7)
    ax.text(7.5, len(names) + 0.2, "Atencao", fontsize=6, color=BRAD_COLORS["gold"], ha="center", fontweight="bold", alpha=0.7)
    ax.text(min(12.5, x_max * 0.85), len(names) + 0.2, "Excesso", fontsize=6, color=BRAD_COLORS["red"], ha="center", fontweight="bold", alpha=0.7)

    bars = ax.barh(y_pos, los_vals, color=colors, edgecolor="white", linewidth=0.5, zorder=2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("LOS Medio (dias)")
    ax.set_title("Ranking de Hospitais por LOS Medio (Top 15 por volume)", fontsize=11, color=BRAD_COLORS["blue"])

    # Global median line (all hospitals)
    all_los = [_safe_float(h.get("avg_los")) for h in hospitals if _safe_float(h.get("avg_los")) > 0]
    if all_los:
        med = np.median(all_los)
        ax.axvline(med, color=BRAD_COLORS["blue"], linestyle="--", linewidth=1.2, alpha=0.7, zorder=3)
        ax.text(med + 0.2, len(names) - 0.5, f"Mediana Bradesco: {med:.1f}d",
                fontsize=7, fontweight="bold", color=BRAD_COLORS["blue"])

    # Annotate bars with value labels
    for bar, val in zip(bars, los_vals):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}d", va="center", fontsize=7, fontweight="bold", color=BRAD_COLORS["gray"])

    ax.set_xlim(0, x_max)

    # Interpretation box
    interp_text = (
        "COMO LER: Barras = LOS medio por hospital\n"
        "Verde (<5d) = adequado | Amarelo (5-10d) = atencao | Vermelho (>10d) = excesso\n"
        "Linha tracejada azul = mediana de TODOS os hospitais Bradesco"
    )
    ax.text(0.98, 0.02, interp_text, transform=ax.transAxes, fontsize=6,
            verticalalignment="bottom", horizontalalignment="right", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=BRAD_COLORS["blue"], alpha=0.9, linewidth=0.8))

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def _chart_trend_dual_axis(path, trend):
    """Bar chart for admissions + line for avg_los on dual Y-axis."""
    import matplotlib.pyplot as plt
    import numpy as np

    if not trend:
        return

    labels = [t.get("label", "") for t in trend]
    admissoes = [_safe_int(t.get("admissoes")) for t in trend]
    avg_los = [_safe_float(t.get("avg_los")) for t in trend]

    fig, ax1 = plt.subplots(figsize=(7.5, 4))
    x = np.arange(len(labels))

    # Highlight Feb (last month) with shaded background
    if len(x) > 0:
        ax1.axvspan(x[-1] - 0.4, x[-1] + 0.4, alpha=0.08, color=BRAD_COLORS["blue"], zorder=0)
        ax1.text(x[-1], max(admissoes) * 1.05 if admissoes else 1, "Fev/26",
                 fontsize=7, fontweight="bold", color=BRAD_COLORS["blue"], ha="center", alpha=0.7)

    bars = ax1.bar(x, admissoes, color=BRAD_COLORS["blue"], alpha=0.8, width=0.6, label="Admissoes", zorder=2)
    ax1.set_xlabel("Mes")
    ax1.set_ylabel("Admissoes", color=BRAD_COLORS["blue"])
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.tick_params(axis='y', labelcolor=BRAD_COLORS["blue"])

    # Data labels on bars (admission count)
    for bar, val in zip(bars, admissoes):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(admissoes) * 0.01,
                 f"{val:,}".replace(",", "."), ha="center", va="bottom", fontsize=7,
                 fontweight="bold", color=BRAD_COLORS["blue"])

    ax2 = ax1.twinx()
    ax2.plot(x, avg_los, color=BRAD_COLORS["red"], marker='o', linewidth=2, markersize=6, label="LOS Medio", zorder=3)
    ax2.set_ylabel("LOS Medio (dias)", color=BRAD_COLORS["red"])
    ax2.tick_params(axis='y', labelcolor=BRAD_COLORS["red"])
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(BRAD_COLORS["red"])

    # Data labels on LOS line
    for i, v in enumerate(avg_los):
        ax2.annotate(f"{v:.1f}d", (i, v), textcoords="offset points", xytext=(0, 10),
                     fontsize=7, fontweight="bold", color=BRAD_COLORS["red"], ha="center",
                     bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor=BRAD_COLORS["red"],
                               alpha=0.7, linewidth=0.4))

    ax1.set_title("Tendencia: Admissoes e LOS Medio (6 meses)", fontsize=11, color=BRAD_COLORS["blue"])

    # Interpretation box
    interp_text = (
        "COMO LER: Barras azuis = volume de admissoes\n"
        "Linha vermelha = LOS medio | Destaque = mes de referencia (Fev)"
    )
    ax1.text(0.02, 0.98, interp_text, transform=ax1.transAxes, fontsize=6,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                       edgecolor=BRAD_COLORS["blue"], alpha=0.9, linewidth=0.8))

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def _chart_risk_heatmap(path, risk_matrix):
    """Seaborn heatmap: rows=CIDs, cols=hospitals, cell=avg_los colored by ratio to median."""
    import matplotlib.pyplot as plt
    import numpy as np

    cells = risk_matrix.get("cells", [])
    cid_medians = risk_matrix.get("cid_medians", {})
    top_cids = risk_matrix.get("top_cids", [])
    top_hosps = risk_matrix.get("top_hospitals", [])

    if not cells or not top_cids or not top_hosps:
        return

    import seaborn as sns

    # Build matrix
    cid_labels = [_truncate(c, 45) for c in top_cids]
    hosp_labels = [_truncate(_hosp_name(h), 20) for h in top_hosps]

    n_cids = len(top_cids)
    n_hosps = len(top_hosps)
    matrix = np.full((n_cids, n_hosps), np.nan)
    ratio_matrix = np.full((n_cids, n_hosps), np.nan)
    annot_matrix = [['' for _ in range(n_hosps)] for _ in range(n_cids)]

    # Index cells
    cell_lookup = {}
    for c in cells:
        key = (c["cid_desc"], c["hospital_id"])
        cell_lookup[key] = _safe_float(c["avg_los"])

    for i, cid in enumerate(top_cids):
        med = _safe_float(cid_medians.get(cid, 0))
        for j, hid in enumerate(top_hosps):
            val = cell_lookup.get((cid, hid))
            if val is not None:
                matrix[i, j] = val
                ratio = val / med if med > 0 else 1.0
                ratio_matrix[i, j] = ratio
                # Annotation: LOS value + ratio to median
                annot_matrix[i][j] = f"{val:.0f}d\n({ratio:.1f}x)"

    # Add right-side labels for CID medians
    cid_median_labels = []
    for cid in top_cids:
        med = _safe_float(cid_medians.get(cid, 0))
        cid_median_labels.append(f"  med={med:.0f}d" if med > 0 else "")

    fig, ax = plt.subplots(figsize=(12, 6))
    # Diverging colormap centered at 1.0: RdYlGn_r
    sns.heatmap(ratio_matrix, ax=ax, cmap="RdYlGn_r", center=1.0, vmin=0.5, vmax=2.0,
                annot=np.array(annot_matrix, dtype=object), fmt='s',
                xticklabels=hosp_labels, yticklabels=cid_labels,
                linewidths=0.5, linecolor='white',
                annot_kws={"fontsize": 6},
                cbar_kws={"label": "Ratio vs Mediana CID (1.0 = na media)", "shrink": 0.8})

    # Add CID median labels on the right
    for i, label in enumerate(cid_median_labels):
        ax.text(n_hosps + 0.1, i + 0.5, label, fontsize=6, va="center", color=BRAD_COLORS["gray"])

    ax.set_title("Matriz de Risco: LOS por CID x Hospital", fontsize=11, color=BRAD_COLORS["blue"])
    ax.set_xlabel("Hospital")
    ax.set_ylabel("CID Principal")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=7)

    # Interpretation box
    interp_text = (
        "COMO LER: Cada celula = LOS medio (dias) + ratio vs mediana do CID\n"
        "Vermelho = LOS > 1.5x mediana do CID | Branco = na media | Verde = abaixo da media\n"
        "Celula vazia = sem internacoes nesse cruzamento CID x Hospital"
    )
    ax.text(0.5, -0.18, interp_text, transform=ax.transAxes, fontsize=6,
            verticalalignment="top", horizontalalignment="center", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=BRAD_COLORS["blue"], alpha=0.9, linewidth=0.8))

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def _chart_hospital_bubble(path, hospitals):
    """Scatter/bubble: X=n_altas, Y=avg_los, size=taxa_obito, color=n_ae."""
    import matplotlib.pyplot as plt
    import numpy as np

    if not hospitals:
        return

    n_altas = [_safe_int(h.get("n_altas")) for h in hospitals]
    avg_los = [_safe_float(h.get("avg_los")) for h in hospitals]
    taxa_ob = [_safe_float(h.get("taxa_obito")) for h in hospitals]
    n_ae = [_safe_int(h.get("n_ae")) for h in hospitals]

    # Better size scaling
    sizes = [max(30, min(600, t * 50 + 30)) for t in taxa_ob]

    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Quadrant lines at median volume (X) and median LOS (Y)
    valid_altas = [v for v in n_altas if v > 0]
    valid_los = [v for v in avg_los if v > 0]
    med_volume = float(np.median(valid_altas)) if valid_altas else 50
    med_los = float(np.median(valid_los)) if valid_los else 8

    ax.axvline(med_volume, color=BRAD_COLORS["gray"], linestyle="--", linewidth=0.8, alpha=0.5, zorder=1)
    ax.axhline(med_los, color=BRAD_COLORS["gray"], linestyle="--", linewidth=0.8, alpha=0.5, zorder=1)

    # Quadrant labels
    x_max = max(n_altas) * 1.15 if n_altas else 100
    y_max = max(avg_los) * 1.15 if avg_los else 20
    # Top-left: Baixo Volume + Alto LOS = Investigar
    ax.text(med_volume * 0.15, y_max * 0.95, "Baixo Vol + Alto LOS\n= INVESTIGAR",
            fontsize=6, color=BRAD_COLORS["red"], alpha=0.6, fontweight="bold", va="top")
    # Top-right: Alto Volume + Alto LOS = Atencao
    ax.text(x_max * 0.75, y_max * 0.95, "Alto Vol + Alto LOS\n= ATENCAO",
            fontsize=6, color=BRAD_COLORS["gold"], alpha=0.7, fontweight="bold", va="top")
    # Bottom-right: Alto Volume + Baixo LOS = Eficiente
    ax.text(x_max * 0.75, med_los * 0.15, "Alto Vol + Baixo LOS\n= EFICIENTE",
            fontsize=6, color=BRAD_COLORS["green"], alpha=0.7, fontweight="bold", va="bottom")

    scatter = ax.scatter(n_altas, avg_los, s=sizes, c=n_ae,
                         cmap="YlOrRd", alpha=0.7, edgecolors=BRAD_COLORS["gray"], linewidths=0.5, zorder=3)

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label("Eventos Adversos (cor)", fontsize=8)

    # Label ALL hospitals with >100 altas
    labeled = set()
    for h in hospitals:
        na = _safe_int(h.get("n_altas"))
        if na > 100:
            x = na
            y = _safe_float(h.get("avg_los"))
            label = _truncate(_hosp_name(h.get("hospital_id")), 20)
            ax.annotate(label, (x, y), fontsize=6, color=BRAD_COLORS["blue"],
                        xytext=(5, 5), textcoords="offset points",
                        arrowprops=dict(arrowstyle="-", color=BRAD_COLORS["gray"], lw=0.4),
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.7, linewidth=0.3))
            labeled.add(_safe_int(h.get("hospital_id")))

    # Also label top 5 by taxa_obito if not already labeled
    sorted_h = sorted(hospitals, key=lambda h: -_safe_float(h.get("taxa_obito")))
    for h in sorted_h[:5]:
        hid = _safe_int(h.get("hospital_id"))
        if hid not in labeled:
            x = _safe_int(h.get("n_altas"))
            y = _safe_float(h.get("avg_los"))
            label = _truncate(_hosp_name(h.get("hospital_id")), 20)
            ax.annotate(label, (x, y), fontsize=6, color=BRAD_COLORS["red"],
                        xytext=(5, -10), textcoords="offset points",
                        arrowprops=dict(arrowstyle="-", color=BRAD_COLORS["red"], lw=0.4))
            labeled.add(hid)

    # Size legend
    from matplotlib.lines import Line2D
    legend_sizes = [2, 5, 10]
    legend_handles = []
    for s in legend_sizes:
        sz = max(30, min(600, s * 50 + 30))
        legend_handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=BRAD_COLORS["gray"],
                                     markersize=sz**0.5, label=f"Obito {s}%", alpha=0.5))
    ax.legend(handles=legend_handles, title="Tamanho = Taxa Obito", title_fontsize=7,
              fontsize=6, loc="lower left", framealpha=0.9)

    ax.set_xlabel("Volume de Altas")
    ax.set_ylabel("LOS Medio (dias)")
    ax.set_title("Hospitais: Volume x LOS x Mortalidade x Eventos Adversos",
                 fontsize=10, color=BRAD_COLORS["blue"])

    # Interpretation box
    interp_text = (
        "COMO LER: Eixo X = volume de altas | Eixo Y = LOS medio (dias)\n"
        "Tamanho = taxa de obito | Cor = qtd eventos adversos\n"
        "Tracejado = medianas | Quadrantes indicam perfil do hospital"
    )
    ax.text(0.02, 0.02, interp_text, transform=ax.transAxes, fontsize=6,
            verticalalignment="bottom", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=BRAD_COLORS["blue"], alpha=0.9, linewidth=0.8))

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def _chart_cid_volume_bars(path, profile):
    """Horizontal bar chart of top 15 CIDs by volume."""
    import matplotlib.pyplot as plt
    import numpy as np

    data = profile.get("top_cids_volume", [])
    if not data:
        return

    top = data[:15]
    top = list(reversed(top))

    names = [_truncate(str(d.get("cid_desc", "---")), 45) for d in top]
    vals = [_safe_int(d.get("n_internacoes")) for d in top]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, vals, color=BRAD_COLORS["blue"], edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Internacoes")
    ax.set_title("Top 15 Diagnosticos Principais por Volume", fontsize=11, color=BRAD_COLORS["blue"])

    # Value labels at end of each bar
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + max(vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:,}".replace(",", "."), va="center", fontsize=7, fontweight="bold",
                color=BRAD_COLORS["gray"])

    # Median reference line
    if vals:
        med = float(np.median(vals))
        ax.axvline(med, color=BRAD_COLORS["gray"], linestyle="--", linewidth=0.8, alpha=0.5)
        ax.text(med, len(vals) - 0.3, f"Mediana: {med:.0f}", fontsize=6, color=BRAD_COLORS["gray"])

    # Interpretation box
    interp_text = (
        "COMO LER: Barras = total de internacoes por CID principal (Fev/26)\n"
        "Os 15 CIDs mais frequentes representam o perfil clinico dominante"
    )
    ax.text(0.98, 0.02, interp_text, transform=ax.transAxes, fontsize=6,
            verticalalignment="bottom", horizontalalignment="right", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=BRAD_COLORS["blue"], alpha=0.9, linewidth=0.8))

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def _chart_los_distribution(path, profile):
    """Vertical bar chart of LOS distribution buckets."""
    import matplotlib.pyplot as plt
    import numpy as np

    data = profile.get("los_distribution", [])
    if not data:
        return

    labels = [str(d.get("faixa_los", "---")) for d in data]
    vals = [_safe_int(d.get("n_internacoes")) for d in data]
    total = sum(vals) if vals else 1

    # Find the median bucket (the one that crosses 50% cumulative)
    cumsum = 0
    median_idx = 0
    for i, v in enumerate(vals):
        cumsum += v
        if cumsum >= total / 2:
            median_idx = i
            break

    # Colors: highlight median bucket
    colors = []
    for i in range(len(vals)):
        if i == median_idx:
            colors.append(BRAD_COLORS["gold"])
        else:
            colors.append(BRAD_COLORS["blue"])

    fig, ax = plt.subplots(figsize=(4, 3.5))
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Internacoes")
    ax.set_title("Distribuicao de Permanencia", fontsize=10, color=BRAD_COLORS["blue"])

    # Percentage labels on top of each bar
    for bar, val in zip(bars, vals):
        pct = 100.0 * val / total if total > 0 else 0
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.01,
                f"{pct:.0f}%", ha="center", va="bottom", fontsize=7, fontweight="bold",
                color=BRAD_COLORS["gray"])

    # Mark median bucket with a star
    ax.text(median_idx, vals[median_idx] + max(vals) * 0.08, "\u2605",
            ha="center", va="bottom", fontsize=12, color=BRAD_COLORS["gold"])

    # Interpretation box
    interp_text = (
        "COMO LER: Cada barra = faixa de dias\n"
        "% = proporcao das internacoes\n"
        "Estrela = faixa mediana"
    )
    ax.text(0.98, 0.98, interp_text, transform=ax.transAxes, fontsize=5.5,
            verticalalignment="top", horizontalalignment="right", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=BRAD_COLORS["blue"], alpha=0.9, linewidth=0.8))

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def _chart_discharge_donut(path, profile):
    """Donut chart of discharge categories."""
    import matplotlib.pyplot as plt

    data = profile.get("discharge_types_grouped", [])
    if not data:
        return

    color_map = {
        "ALTA NORMAL": BRAD_COLORS["green"],
        "ALTA COMPLEXA": (0.2, 0.5, 0.3),
        "OBITO": BRAD_COLORS["deathred"],
        "TRANSFERENCIA": BRAD_COLORS["gold"],
        "NEONATAL": BRAD_COLORS["lightblue"],
        "ADMINISTRATIVO": BRAD_COLORS["gray"],
        "PROGRAMA ESPECIAL": BRAD_COLORS["blue"],
        "SEM TIPO DE ALTA DEFINIDO": (0.85, 0.55, 0.15),
        "OUTRO": BRAD_COLORS["lightgray"],
    }

    labels = [str(d.get("grupo", "---")) for d in data]
    vals = [_safe_int(d.get("n")) for d in data]
    total = sum(vals) if vals else 1
    colors = [color_map.get(l, BRAD_COLORS["lightgray"]) for l in labels]

    # Find largest category
    max_idx = vals.index(max(vals)) if vals else 0

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    wedges, texts, autotexts = ax.pie(
        vals, labels=None, colors=colors, autopct='%1.1f%%',
        startangle=90, pctdistance=0.78,
        wedgeprops={"width": 0.4, "edgecolor": "white", "linewidth": 1.5},
    )
    for i, t in enumerate(autotexts):
        pct_val = 100.0 * vals[i] / total if total > 0 else 0
        if i == max_idx:
            t.set_fontsize(8)
            t.set_fontweight("bold")
        else:
            t.set_fontsize(6.5)
        # Hide very small slices percentage to avoid clutter
        if pct_val < 2:
            t.set_text("")

    # Bold the legend label for largest category
    legend_labels = []
    for i, l in enumerate(labels):
        pct_val = 100.0 * vals[i] / total if total > 0 else 0
        display = f"{l} ({vals[i]})"
        legend_labels.append(display)

    leg = ax.legend(legend_labels, fontsize=6, loc="center left", bbox_to_anchor=(0.88, 0.5))
    # Bold the largest category in legend
    if max_idx < len(leg.get_texts()):
        leg.get_texts()[max_idx].set_fontweight("bold")

    # Center text: interpretation
    ax.text(0, 0, f"N={total:,}".replace(",", "."), ha="center", va="center",
            fontsize=9, fontweight="bold", color=BRAD_COLORS["blue"])
    ax.text(0, -0.15, "altas Fev/26", ha="center", va="center",
            fontsize=6, color=BRAD_COLORS["gray"])

    ax.set_title("Tipos de Alta", fontsize=10, color=BRAD_COLORS["blue"])
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def _chart_ae_by_type(path, ae):
    """Horizontal bar chart of adverse event types."""
    import matplotlib.pyplot as plt
    import numpy as np

    data = ae.get("by_type", [])
    if not data:
        return

    top = list(reversed(data[:10]))
    names = [_truncate(str(d.get("tipo_evento", "---")), 30) for d in top]
    vals = [_safe_int(d.get("n_eventos")) for d in top]

    # Color top 3 (highest values, which are last after reverse) in red, rest in lighter color
    n = len(vals)
    colors = []
    for i in range(n):
        if i >= n - 3:  # top 3 are at the end (bottom of reversed list = top values)
            colors.append(BRAD_COLORS["red"])
        else:
            colors.append(BRAD_COLORS["gray"])

    fig, ax = plt.subplots(figsize=(7.5, 4))
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Eventos")
    ax.set_title("Eventos Adversos por Tipo (Top 10)", fontsize=11, color=BRAD_COLORS["blue"])

    # Value labels at end of each bar
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + max(vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val}", va="center", fontsize=7, fontweight="bold", color=BRAD_COLORS["gray"])

    # Median reference line
    if vals:
        med = float(np.median(vals))
        ax.axvline(med, color=BRAD_COLORS["gray"], linestyle="--", linewidth=0.8, alpha=0.5)
        ax.text(med, len(vals) - 0.3, f"Mediana: {med:.0f}", fontsize=6, color=BRAD_COLORS["gray"])

    # Interpretation box
    interp_text = (
        "COMO LER: Barras = contagem de eventos adversos por tipo\n"
        "Vermelho = 3 tipos mais frequentes | Cinza = demais tipos"
    )
    ax.text(0.98, 0.02, interp_text, transform=ax.transAxes, fontsize=6,
            verticalalignment="bottom", horizontalalignment="right", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=BRAD_COLORS["blue"], alpha=0.9, linewidth=0.8))

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def _chart_ae_by_hospital(path, ae, hospitals):
    """Bar chart of AE rate per 100 admissions per hospital, sorted by rate."""
    import matplotlib.pyplot as plt
    import numpy as np

    ae_hosp = ae.get("by_hospital", [])
    if not ae_hosp:
        return

    # Build denominator map from hospitals
    hosp_altas = {_safe_int(h.get("hospital_id")): _safe_int(h.get("n_altas")) for h in hospitals}

    # Compute rate for each hospital and sort by rate (descending)
    enriched = []
    for d in ae_hosp[:15]:
        hid = _safe_int(d.get("hospital_id"))
        n_altas = hosp_altas.get(hid, 0)
        n_ev = _safe_int(d.get("n_eventos"))
        rate = 100.0 * n_ev / n_altas if n_altas > 0 else 0
        enriched.append({"hospital_id": hid, "n_eventos": n_ev, "n_altas": n_altas, "rate": rate})

    # Sort by rate descending, then reverse for horizontal bar (bottom-to-top)
    enriched = sorted(enriched, key=lambda d: d["rate"])[-10:]

    names = [_truncate(_hosp_name(d.get("hospital_id")), 25) for d in enriched]
    rates = [d["rate"] for d in enriched]
    vals = [d["n_eventos"] for d in enriched]
    n_altas_list = [d["n_altas"] for d in enriched]

    fig, ax = plt.subplots(figsize=(7.5, 4))
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, rates, color=BRAD_COLORS["red"], edgecolor="white", linewidth=0.5, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Taxa de EA por 100 Admissoes")
    ax.set_title("Eventos Adversos por Hospital (ordenado por taxa)", fontsize=11, color=BRAD_COLORS["blue"])

    # Annotate with rate + absolute count
    for bar, val, rate, na in zip(bars, vals, rates, n_altas_list):
        ax.text(bar.get_width() + max(rates) * 0.02, bar.get_y() + bar.get_height() / 2,
                f"{rate:.1f}/100adm ({val} EA / {na} altas)", va="center", fontsize=6,
                fontweight="bold", color=BRAD_COLORS["gray"])

    # Median rate line
    if rates:
        med_rate = float(np.median(rates))
        ax.axvline(med_rate, color=BRAD_COLORS["gray"], linestyle="--", linewidth=0.8, alpha=0.5)
        ax.text(med_rate, len(rates) - 0.3, f"Mediana: {med_rate:.1f}", fontsize=6, color=BRAD_COLORS["gray"])

    # Interpretation box
    interp_text = (
        "COMO LER: Barras = taxa de eventos adversos por 100 admissoes\n"
        "Ordenado por taxa (nao por contagem absoluta)\n"
        "Anotacao = taxa + (absoluto EA / total altas)"
    )
    ax.text(0.98, 0.02, interp_text, transform=ax.transAxes, fontsize=6,
            verticalalignment="bottom", horizontalalignment="right", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=BRAD_COLORS["blue"], alpha=0.9, linewidth=0.8))

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def _chart_med_top10(path, meds):
    """Horizontal bar chart of top 10 medications."""
    import matplotlib.pyplot as plt
    import numpy as np

    data = meds.get("top_medications", [])[:10]
    if not data:
        return

    top = list(reversed(data))
    names = [_truncate(str(d.get("medicamento", "---")), 30) for d in top]
    vals = [_safe_int(d.get("n_prescricoes")) for d in top]

    fig, ax = plt.subplots(figsize=(7.5, 4))
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, vals, color=BRAD_COLORS["blue"], edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Prescricoes")
    ax.set_title("Top 10 Medicamentos Mais Prescritos", fontsize=11, color=BRAD_COLORS["blue"])

    # Value labels at end of each bar
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + max(vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:,}".replace(",", "."), va="center", fontsize=7, fontweight="bold",
                color=BRAD_COLORS["gray"])

    # Interpretation box
    interp_text = (
        "COMO LER: Barras = total de prescricoes por medicamento (Fev/26)\n"
        "Fonte: capta_produtos_capr (proxy financeiro, sem dados de faturamento)"
    )
    ax.text(0.98, 0.02, interp_text, transform=ax.transAxes, fontsize=6,
            verticalalignment="bottom", horizontalalignment="right", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=BRAD_COLORS["blue"], alpha=0.9, linewidth=0.8))

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def _chart_heatmap_trajectory(path, twin):
    """2D hexbin: X=velocity, Y=death proximity."""
    import matplotlib.pyplot as plt
    import numpy as np

    data = twin.get("trajectory_velocity", [])
    if not data or len(data) < 3:
        return

    velocities = [_safe_float(d.get("velocity")) for d in data]
    death_cos = [_safe_float(d.get("curr_death_proximity", d.get("death_cos", 0))) for d in data]

    fig, ax = plt.subplots(figsize=(7.5, 4))
    hb = ax.hexbin(velocities, death_cos, gridsize=15, cmap="YlOrRd", mincnt=1)
    cb = fig.colorbar(hb, ax=ax, shrink=0.8)
    cb.set_label("Contagem", fontsize=8)

    ax.set_xlabel("Velocidade de Trajetoria")
    ax.set_ylabel("Proximidade ao Obito (cosseno)")
    ax.set_title("Mapa de Calor: Velocidade x Proximidade ao Obito",
                 fontsize=11, color=BRAD_COLORS["blue"])
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def _chart_trend_sparklines(path, trend):
    """2x3 grid of sparklines for 6 KPIs over 6 months."""
    import matplotlib.pyplot as plt
    import numpy as np

    if not trend or len(trend) < 2:
        return

    kpi_defs = [
        ("Admissoes", "admissoes", False),
        ("LOS Medio", "avg_los", True),
        ("Taxa Obito (%)", "taxa_obito", True),
        ("Eventos Adversos", "n_ae", True),
        ("Readmissao (%)", "taxa_readmit", True),
        ("Prescricoes", "n_prescricoes", False),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(7.5, 3))
    axes = axes.flatten()

    for ax, (title, key, invert) in zip(axes, kpi_defs):
        vals = [_safe_float(t.get(key)) for t in trend]
        x = np.arange(len(vals))
        color = BRAD_COLORS["blue"]

        # Color line red if last value is worse than first
        if len(vals) >= 2:
            if invert and vals[-1] > vals[0]:
                color = BRAD_COLORS["red"]
            elif not invert and vals[-1] < vals[0]:
                color = BRAD_COLORS["red"]

        ax.plot(x, vals, color=color, linewidth=1.5, marker='', zorder=2)
        ax.fill_between(x, vals, alpha=0.1, color=color)
        ax.set_title(title, fontsize=7, color=BRAD_COLORS["gray"], pad=2)

        # Bold dot on last point (Feb)
        if vals:
            ax.plot(len(vals) - 1, vals[-1], 'o', color=color, markersize=5, zorder=3)

        # Show first and last values as text
        if vals:
            first = vals[0]
            last = vals[-1]
            fmt_first = f"{first:.1f}" if first < 1000 else f"{int(first):,}".replace(",", ".")
            fmt_last = f"{last:.1f}" if last < 1000 else f"{int(last):,}".replace(",", ".")
            ax.annotate(fmt_first, (0, first), fontsize=5.5, color=BRAD_COLORS["gray"],
                        xytext=(-2, -8), textcoords="offset points", ha="left")
            ax.annotate(fmt_last, (len(vals) - 1, last), fontsize=7, fontweight="bold",
                        color=color, xytext=(2, 4), textcoords="offset points")

        # X-axis: month labels
        month_labels = [str(t.get("label", ""))[:7] for t in trend]  # "2025-09" etc
        short_labels = []
        for ml in month_labels:
            try:
                parts = ml.split("-")
                short_labels.append(f"{parts[1]}/{parts[0][2:]}")  # "09/25"
            except Exception:
                short_labels.append(ml)
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, fontsize=5, color=BRAD_COLORS["gray"], rotation=0)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(True)
        ax.spines["bottom"].set_linewidth(0.5)
        ax.grid(False)

    fig.suptitle("Sparklines: Tendencia 6 Meses", fontsize=10, fontweight="bold",
                 color=BRAD_COLORS["blue"], y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# -----------------------------------------------------------------
# NEW: Hospital Risk Radar chart
# -----------------------------------------------------------------

def _chart_hospital_risk_radar(path, twin, hospitals=None):
    """Scatter: cos_alta (X) vs cos_death (Y) per hospital, colored by proj_outcome."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np

    data = twin.get("hospital_risk_landscape", [])
    if not data or len(data) < 3:
        return

    cos_a = np.array([d["cos_alta"] for d in data])
    cos_d = np.array([d["cos_death"] for d in data])
    proj = np.array([d["proj_outcome"] for d in data])
    sizes = np.array([max(d.get("n_feb_admissions", 1), 1) for d in data])
    # Normalize sizes for display (min 30, max 400)
    s_min, s_max = sizes.min(), sizes.max()
    if s_max > s_min:
        s_norm = 30 + 370 * (sizes - s_min) / (s_max - s_min)
    else:
        s_norm = np.full_like(sizes, 100, dtype=float)

    fig, ax = plt.subplots(figsize=(9, 7))

    # Diagonal: cos_death = cos_alta
    lo = min(cos_a.min(), cos_d.min()) - 0.02
    hi = max(cos_a.max(), cos_d.max()) + 0.02
    ax.plot([lo, hi], [lo, hi], "--", color=BRAD_COLORS["gray"], alpha=0.5, linewidth=1, label="Neutro (cos_death = cos_alta)")

    # Color gradient red-green based on proj_outcome
    cmap = mcolors.LinearSegmentedColormap.from_list("rg", [BRAD_COLORS["green"], (0.9, 0.9, 0.2), BRAD_COLORS["red"]])
    # Normalize proj to 0-1
    p_min, p_max = proj.min(), proj.max()
    if p_max > p_min:
        p_norm = (proj - p_min) / (p_max - p_min)
    else:
        p_norm = np.full_like(proj, 0.5)

    scatter = ax.scatter(cos_a, cos_d, s=s_norm, c=p_norm, cmap=cmap, alpha=0.75,
                         edgecolors="white", linewidths=0.5, zorder=5)

    # Label all hospitals
    for d_item in data:
        hname = _hosp_name(d_item["hospital_id"])
        if len(hname) > 25:
            hname = hname[:23] + ".."
        ax.annotate(hname, (d_item["cos_alta"], d_item["cos_death"]),
                    fontsize=5.5, alpha=0.8, ha="center", va="bottom",
                    textcoords="offset points", xytext=(0, 4))

    # Quadrant labels
    mid_x = (cos_a.min() + cos_a.max()) / 2
    mid_y = (cos_d.min() + cos_d.max()) / 2
    ax.text(cos_a.min() + 0.003, cos_d.max() - 0.003, "ALTO RISCO",
            fontsize=9, fontweight="bold", color=BRAD_COLORS["red"], alpha=0.4, va="top", ha="left")
    ax.text(cos_a.max() - 0.003, cos_d.min() + 0.003, "BOA PERFORMANCE",
            fontsize=9, fontweight="bold", color=BRAD_COLORS["green"], alpha=0.4, va="bottom", ha="right")
    ax.text(cos_a.max() - 0.003, cos_d.max() - 0.003, "CASE MIX GRAVE",
            fontsize=9, fontweight="bold", color=BRAD_COLORS["gold"], alpha=0.4, va="top", ha="right")
    ax.text(cos_a.min() + 0.003, cos_d.min() + 0.003, "BAIXO VOLUME?",
            fontsize=9, fontweight="bold", color=BRAD_COLORS["gray"], alpha=0.4, va="bottom", ha="left")

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Projecao no eixo obito-alta", fontsize=8)

    ax.set_xlabel("Similaridade Cosseno com Alta Melhorada", fontsize=10)
    ax.set_ylabel("Similaridade Cosseno com Obito", fontsize=10)
    ax.set_title("Radar de Risco Hospitalar: Alinhamento com Desfecho", fontsize=12,
                 color=BRAD_COLORS["blue"], fontweight="bold")

    # Interpretation box
    ax.text(0.02, 0.02,
            "COMO LER: Hospitais acima da diagonal tem perfil de embedding\n"
            "mais alinhado com obitos. Abaixo = altas bem-sucedidas.\n"
            "Tamanho = volume de internacoes Fev.",
            transform=ax.transAxes, fontsize=7,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=BRAD_COLORS["gray"], alpha=0.9),
            va="bottom", ha="left")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# -----------------------------------------------------------------
# NEW: Chronic Patient Trajectories chart
# -----------------------------------------------------------------

def _chart_chronic_trajectories(path, twin):
    """Line chart: death_proximity over admission sequence for top 5 deteriorating patients."""
    import matplotlib.pyplot as plt
    import numpy as np

    data = twin.get("chronic_trajectories", [])
    if not data:
        return

    # Pick top 5 deteriorating patients
    top5 = [d for d in data if d.get("is_deteriorating")][:5]
    if len(top5) < 2:
        top5 = data[:5]  # fallback: top 5 by slope

    chron_p75 = twin.get("chronic_p75", 0)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors_list = [BRAD_COLORS["red"], BRAD_COLORS["blue"], BRAD_COLORS["gold"],
                   BRAD_COLORS["green"], BRAD_COLORS["gray"]]

    for i, patient in enumerate(top5):
        dp_series = patient.get("dp_series", [])
        if not dp_series:
            continue
        x_vals = list(range(1, len(dp_series) + 1))
        color = colors_list[i % len(colors_list)]
        pid_label = f"Pac. {patient['ID_CD_PACIENTE']} (n={patient['n_admissions']}, slope={patient['slope']:.4f})"
        ax.plot(x_vals, dp_series, marker="o", markersize=4, linewidth=1.5,
                color=color, label=pid_label, alpha=0.85)

    # Reference line at P75
    if chron_p75 > 0:
        ax.axhline(y=chron_p75, color=BRAD_COLORS["red"], linestyle="--", alpha=0.5, linewidth=1)
        ax.text(ax.get_xlim()[1] * 0.98, chron_p75, f"P75 = {chron_p75:.3f}",
                ha="right", va="bottom", fontsize=7, color=BRAD_COLORS["red"], alpha=0.7)

    ax.set_xlabel("Numero da Internacao (sequencial)", fontsize=10)
    ax.set_ylabel("Proximidade ao Obito (cosseno)", fontsize=10)
    ax.set_title("Pacientes Cronicos: Trajetoria de Proximidade ao Obito",
                 fontsize=12, color=BRAD_COLORS["blue"], fontweight="bold")
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9)

    # Interpretation box
    ax.text(0.02, 0.02,
            "Linhas ascendentes = paciente se aproximando do\n"
            "perfil de obito a cada internacao. Slope positivo\n"
            "acima do P75 = deterioracao critica.",
            transform=ax.transAxes, fontsize=7,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=BRAD_COLORS["gray"], alpha=0.9),
            va="bottom", ha="left")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# -----------------------------------------------------------------
# NEW: Medication Landscape chart
# -----------------------------------------------------------------

def _chart_medication_landscape(path, twin):
    """Horizontal bar chart: top 20 medications by proj_outcome (death-alta axis)."""
    import matplotlib.pyplot as plt
    import numpy as np

    data = twin.get("medication_embeddings", [])
    if not data or len(data) < 3:
        return

    # Sort by proj_outcome descending, take top 10 + bottom 10
    sorted_meds = sorted(data, key=lambda x: -x["proj_outcome"])
    top_death = sorted_meds[:10]
    top_alta = sorted_meds[-10:]
    # Combine: death-aligned first, then recovery-aligned
    display = top_death + list(reversed(top_alta))
    # Remove duplicates while keeping order
    seen = set()
    unique_display = []
    for m in display:
        if m["medicamento"] not in seen:
            seen.add(m["medicamento"])
            unique_display.append(m)
    display = unique_display[:20]

    # Sort for display: most positive (death) at top
    display.sort(key=lambda x: x["proj_outcome"])

    names = [d["medicamento"][:35] + (".." if len(d["medicamento"]) > 35 else "") for d in display]
    proj_vals = [d["proj_outcome"] for d in display]

    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = np.arange(len(names))
    colors = [BRAD_COLORS["red"] if v > 0 else BRAD_COLORS["green"] for v in proj_vals]

    ax.barh(y_pos, proj_vals, color=colors, alpha=0.75, edgecolor="white", height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=7)
    ax.axvline(x=0, color=BRAD_COLORS["gray"], linewidth=1, alpha=0.5)
    ax.set_xlabel("Projecao no eixo Obito-Alta (positivo = morte, negativo = recuperacao)", fontsize=9)
    ax.set_title("Paisagem de Medicamentos: Alinhamento com Desfecho",
                 fontsize=12, color=BRAD_COLORS["blue"], fontweight="bold")

    # Interpretation box
    ax.text(0.98, 0.02,
            "Barras vermelhas = medicamentos prescritos em\n"
            "contextos similares ao perfil de obito.\n"
            "NAO indica causalidade, apenas associacao\n"
            "com gravidade clinica.",
            transform=ax.transAxes, fontsize=7,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=BRAD_COLORS["gray"], alpha=0.9),
            va="bottom", ha="right")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# -----------------------------------------------------------------
# Chart orchestrator
# -----------------------------------------------------------------

def _generate_all_charts(kpis, hospitals, trend, profile, los, risk_matrix,
                         doc_cid, med_intensity, docs, ae, meds, twin, twin_stats):
    import os
    os.makedirs(CHART_DIR, exist_ok=True)
    _setup_chart_style()

    charts = {}
    charts["kpi_gauges"] = _safe_chart(_chart_kpi_gauges, f"{CHART_DIR}/kpi_gauges.png", kpis)
    charts["scatter_death"] = _safe_chart(_chart_scatter_death_proximity, f"{CHART_DIR}/scatter_death.png", twin)
    charts["hosp_los_bars"] = _safe_chart(_chart_hospital_los_ranking, f"{CHART_DIR}/hosp_los.png", hospitals)
    charts["trend_dual"] = _safe_chart(_chart_trend_dual_axis, f"{CHART_DIR}/trend_dual.png", trend)
    charts["risk_heatmap"] = _safe_chart(_chart_risk_heatmap, f"{CHART_DIR}/risk_heatmap.png", risk_matrix)
    charts["hosp_bubble"] = _safe_chart(_chart_hospital_bubble, f"{CHART_DIR}/hosp_bubble.png", hospitals)
    charts["cid_bars"] = _safe_chart(_chart_cid_volume_bars, f"{CHART_DIR}/cid_bars.png", profile)
    charts["los_dist"] = _safe_chart(_chart_los_distribution, f"{CHART_DIR}/los_dist.png", profile)
    charts["discharge_donut"] = _safe_chart(_chart_discharge_donut, f"{CHART_DIR}/discharge_donut.png", profile)
    charts["ae_type"] = _safe_chart(_chart_ae_by_type, f"{CHART_DIR}/ae_type.png", ae)
    charts["ae_hospital"] = _safe_chart(_chart_ae_by_hospital, f"{CHART_DIR}/ae_hosp.png", ae, hospitals)
    charts["med_bars"] = _safe_chart(_chart_med_top10, f"{CHART_DIR}/med_bars.png", meds)
    charts["heatmap_traj"] = _safe_chart(_chart_heatmap_trajectory, f"{CHART_DIR}/heatmap_traj.png", twin)
    charts["sparklines"] = _safe_chart(_chart_trend_sparklines, f"{CHART_DIR}/sparklines.png", trend)
    charts["hosp_risk_radar"] = _safe_chart(_chart_hospital_risk_radar, f"{CHART_DIR}/hosp_risk_radar.png", twin, hospitals)
    charts["chronic_traj"] = _safe_chart(_chart_chronic_trajectories, f"{CHART_DIR}/chronic_traj.png", twin)
    charts["med_landscape"] = _safe_chart(_chart_medication_landscape, f"{CHART_DIR}/med_landscape.png", twin)

    generated = sum(1 for v in charts.values() if v is not None)
    print(f"    {generated}/{len(charts)} charts generated")
    return charts


# -----------------------------------------------------------------
# SECTION DATA FETCHERS
# -----------------------------------------------------------------

def _fetch_kpis_for_month(con, m_start, m_end):
    """Fetch full KPI set for any month. Returns dict."""
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

    # Medications
    cols, rows = _exec(con, f"""
        SELECT COUNT(*) AS total_prescricoes
        FROM agg_tb_capta_produtos_capr p
        JOIN agg_tb_capta_internacao_cain i
            ON p.ID_CD_INTERNACAO = i.ID_CD_INTERNACAO AND p.source_db = i.source_db
        WHERE p.{SRC}
          AND i.DH_ADMISSAO_HOSP < '{m_end}'::DATE + INTERVAL '1 day'
          AND (i.DH_FINALIZACAO IS NULL OR i.DH_FINALIZACAO >= '{m_start}'::DATE)
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

    return kpis


def _fetch_kpis(con):
    """KPIs for Feb + Jan for MoM deltas."""
    import time
    print("[01/14] KPIs (Feb + Jan) ...")
    t0 = time.time()
    feb = _fetch_kpis_for_month(con, FEB_START, FEB_END)
    jan = _fetch_kpis_for_month(con, JAN_START, JAN_END)
    print(f"    done in {time.time()-t0:.1f}s")
    return {"feb": feb, "jan": jan}


def _fetch_hospital_performance(con):
    """Hospital ranking for Feb AND Jan (for MoM LOS delta)."""
    import time
    print("[02/14] Hospital performance ...")
    t0 = time.time()

    def _hosp_for_month(m_start, m_end):
        # Base: volume + LOS per hospital
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
        hosp_base = {r[0]: dict(zip(cols, r)) for r in rows}

        # Mortality per hospital
        cols, rows = _exec(con, f"""
            WITH m_dis AS (
                SELECT ID_CD_INTERNACAO, ID_CD_HOSPITAL
                FROM agg_tb_capta_internacao_cain
                WHERE {SRC} AND IN_SITUACAO = 2
                  AND DH_FINALIZACAO >= '{m_start}'::DATE
                  AND DH_FINALIZACAO < '{m_end}'::DATE + INTERVAL '1 day'
            ),
            last_status AS (
                SELECT es.ID_CD_INTERNACAO,
                       ROW_NUMBER() OVER (PARTITION BY es.ID_CD_INTERNACAO ORDER BY es.DH_CADASTRO DESC) AS rn,
                       es.FL_DESOSPITALIZACAO
                FROM agg_tb_capta_evo_status_caes es
                WHERE es.{SRC}
                  AND es.ID_CD_INTERNACAO IN (SELECT ID_CD_INTERNACAO FROM m_dis)
            ),
            discharge_with_type AS (
                SELECT fd.ID_CD_HOSPITAL, fd.ID_CD_INTERNACAO,
                       CASE WHEN UPPER(f.DS_FINAL_MONITORAMENTO) LIKE '%%BITO%%'
                            THEN 1 ELSE 0 END AS is_obito
                FROM m_dis fd
                JOIN last_status ls ON fd.ID_CD_INTERNACAO = ls.ID_CD_INTERNACAO AND ls.rn = 1
                JOIN agg_tb_capta_tipo_final_monit_fmon f
                    ON ls.FL_DESOSPITALIZACAO = f.ID_CD_FINAL_MONITORAMENTO
                    AND f.{SRC}
            )
            SELECT ID_CD_HOSPITAL AS hospital_id, SUM(is_obito) AS n_obito
            FROM discharge_with_type
            GROUP BY ID_CD_HOSPITAL
        """)
        hosp_mort = {r[0]: _safe_int(r[1]) for r in rows}

        # Adverse events per hospital
        cols, rows = _exec(con, f"""
            SELECT i.ID_CD_HOSPITAL AS hospital_id, COUNT(*) AS n_ae
            FROM agg_tb_capta_eventos_adversos_caed e
            JOIN agg_tb_capta_internacao_cain i
                ON e.ID_CD_INTERNACAO = i.ID_CD_INTERNACAO AND e.source_db = i.source_db
            WHERE e.{SRC}
              AND e.DH_EVENTO_ADVERSO >= '{m_start}'::DATE
              AND e.DH_EVENTO_ADVERSO < '{m_end}'::DATE + INTERVAL '1 day'
            GROUP BY i.ID_CD_HOSPITAL
        """)
        hosp_ae = {r[0]: _safe_int(r[1]) for r in rows}

        # Readmissions per hospital
        cols, rows = _exec(con, f"""
            WITH ordered_adm AS (
                SELECT ID_CD_PACIENTE, ID_CD_INTERNACAO, ID_CD_HOSPITAL,
                       DH_ADMISSAO_HOSP::DATE AS dt_adm,
                       LAG(DH_FINALIZACAO::DATE) OVER (
                           PARTITION BY ID_CD_PACIENTE ORDER BY DH_ADMISSAO_HOSP
                       ) AS prev_discharge
                FROM agg_tb_capta_internacao_cain
                WHERE {SRC} AND IN_SITUACAO = 2 AND DH_ADMISSAO_HOSP IS NOT NULL
            )
            SELECT ID_CD_HOSPITAL AS hospital_id,
                   COUNT(*) FILTER (WHERE DATEDIFF('day', prev_discharge, dt_adm) BETWEEN 1 AND 30
                                    AND dt_adm >= '{m_start}'::DATE
                                    AND dt_adm < '{m_end}'::DATE + INTERVAL '1 day') AS readmit_30d
            FROM ordered_adm
            WHERE prev_discharge IS NOT NULL
              AND dt_adm >= '{m_start}'::DATE
              AND dt_adm < '{m_end}'::DATE + INTERVAL '1 day'
            GROUP BY ID_CD_HOSPITAL
        """)
        hosp_readmit = {r[0]: _safe_int(r[1]) for r in rows}

        result = []
        for hid, base in hosp_base.items():
            n_altas = _safe_int(base.get("n_altas"))
            result.append({
                "hospital_id": hid,
                "n_altas": n_altas,
                "avg_los": _safe_float(base.get("avg_los")),
                "median_los": _safe_float(base.get("median_los")),
                "n_obito": hosp_mort.get(hid, 0),
                "taxa_obito": 100.0 * hosp_mort.get(hid, 0) / n_altas if n_altas > 0 else 0,
                "n_ae": hosp_ae.get(hid, 0),
                "readmit_30d": hosp_readmit.get(hid, 0),
            })
        result.sort(key=lambda x: -x["n_altas"])
        return result

    feb_hosps = _hosp_for_month(FEB_START, FEB_END)
    jan_hosps = _hosp_for_month(JAN_START, JAN_END)
    jan_map = {h["hospital_id"]: h for h in jan_hosps}

    # Attach Jan LOS delta to Feb hospitals
    for h in feb_hosps:
        hid = h["hospital_id"]
        jan_h = jan_map.get(hid)
        if jan_h and jan_h["avg_los"] > 0:
            h["jan_avg_los"] = jan_h["avg_los"]
            h["los_delta_pct"] = 100.0 * (h["avg_los"] - jan_h["avg_los"]) / jan_h["avg_los"]
        else:
            h["jan_avg_los"] = None
            h["los_delta_pct"] = None

    print(f"    {len(feb_hosps)} hospitals, done in {time.time()-t0:.1f}s")
    return feb_hosps


def _fetch_trend_6m(con):
    """6-month trend with ALL KPIs: admissions, discharges, LOS mean+median,
    mortality rate, adverse events, readmission rate, medication prescriptions."""
    import time
    print("[03/14] 6-month trend ...")
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

        # LOS for discharges
        cols2, rows2 = _exec(con, f"""
            SELECT AVG(DATEDIFF('day', DH_ADMISSAO_HOSP::DATE, DH_FINALIZACAO::DATE)) AS avg_los,
                   MEDIAN(DATEDIFF('day', DH_ADMISSAO_HOSP::DATE, DH_FINALIZACAO::DATE)) AS median_los
            FROM agg_tb_capta_internacao_cain
            WHERE {SRC} AND IN_SITUACAO = 2
              AND DH_FINALIZACAO >= '{m_start}'::DATE
              AND DH_FINALIZACAO < '{m_end}'::DATE + INTERVAL '1 day'
              AND DH_ADMISSAO_HOSP IS NOT NULL
        """)
        avg_los = _safe_float(rows2[0][0]) if rows2 and rows2[0][0] else 0
        median_los = _safe_float(rows2[0][1]) if rows2 and rows2[0][1] else 0

        n_obito = _count_obitos(con, m_start, m_end)
        n_altas = _safe_int(r.get("altas"))

        # Adverse events for this month
        cols3, rows3 = _exec(con, f"""
            SELECT COUNT(*) AS n_ae
            FROM agg_tb_capta_eventos_adversos_caed
            WHERE {SRC}
              AND DH_EVENTO_ADVERSO >= '{m_start}'::DATE
              AND DH_EVENTO_ADVERSO < '{m_end}'::DATE + INTERVAL '1 day'
        """)
        n_ae = _safe_int(rows3[0][0]) if rows3 else 0

        # Readmissions
        cols4, rows4 = _exec(con, f"""
            WITH ordered_adm AS (
                SELECT ID_CD_PACIENTE, DH_ADMISSAO_HOSP::DATE AS dt_adm,
                       LAG(DH_FINALIZACAO::DATE) OVER (
                           PARTITION BY ID_CD_PACIENTE ORDER BY DH_ADMISSAO_HOSP
                       ) AS prev_discharge
                FROM agg_tb_capta_internacao_cain
                WHERE {SRC} AND IN_SITUACAO = 2 AND DH_ADMISSAO_HOSP IS NOT NULL
            )
            SELECT
                COUNT(*) FILTER (WHERE DATEDIFF('day', prev_discharge, dt_adm) BETWEEN 1 AND 30
                                 AND dt_adm >= '{m_start}'::DATE
                                 AND dt_adm < '{m_end}'::DATE + INTERVAL '1 day') AS readmit,
                COUNT(*) FILTER (WHERE prev_discharge IS NOT NULL
                                 AND dt_adm >= '{m_start}'::DATE
                                 AND dt_adm < '{m_end}'::DATE + INTERVAL '1 day') AS total_prior
            FROM ordered_adm
        """)
        n_readmit = _safe_int(rows4[0][0]) if rows4 else 0
        n_prior = _safe_int(rows4[0][1]) if rows4 else 0

        # Medications
        cols5, rows5 = _exec(con, f"""
            SELECT COUNT(*) AS n_prescricoes
            FROM agg_tb_capta_produtos_capr p
            JOIN agg_tb_capta_internacao_cain i
                ON p.ID_CD_INTERNACAO = i.ID_CD_INTERNACAO AND p.source_db = i.source_db
            WHERE p.{SRC}
              AND i.DH_ADMISSAO_HOSP < '{m_end}'::DATE + INTERVAL '1 day'
              AND (i.DH_FINALIZACAO IS NULL OR i.DH_FINALIZACAO >= '{m_start}'::DATE)
        """)
        n_prescricoes = _safe_int(rows5[0][0]) if rows5 else 0

        trend.append({
            "label": label,
            "admissoes": _safe_int(r.get("admissoes")),
            "altas": n_altas,
            "avg_los": avg_los,
            "median_los": median_los,
            "n_obito": n_obito,
            "taxa_obito": 100.0 * n_obito / n_altas if n_altas > 0 else 0,
            "n_ae": n_ae,
            "readmit": n_readmit,
            "taxa_readmit": 100.0 * n_readmit / n_prior if n_prior > 0 else 0,
            "n_prescricoes": n_prescricoes,
        })

    print(f"    done in {time.time()-t0:.1f}s")
    return trend


def _fetch_clinical_profile(con):
    """Top CIDs (IN_PRINCIPAL='S'), LOS distribution, discharge types (grouped)."""
    import time
    print("[04/14] Clinical profile ...")
    t0 = time.time()

    FEB_ACTIVE = f"""
        i.{SRC}
        AND i.DH_ADMISSAO_HOSP < '{FEB_END}'::DATE + INTERVAL '1 day'
        AND (i.DH_FINALIZACAO IS NULL OR i.DH_FINALIZACAO >= '{FEB_START}'::DATE)
    """

    profile = {}

    # Top 15 CIDs by volume
    cols, rows = _exec(con, f"""
        SELECT c.DS_DESCRICAO AS cid_desc,
               COUNT(DISTINCT c.ID_CD_INTERNACAO) AS n_internacoes,
               AVG(DATEDIFF('day', i.DH_ADMISSAO_HOSP::DATE,
                   COALESCE(i.DH_FINALIZACAO, CURRENT_TIMESTAMP)::DATE)) AS avg_los
        FROM agg_tb_capta_cid_caci c
        JOIN agg_tb_capta_internacao_cain i
            ON c.ID_CD_INTERNACAO = i.ID_CD_INTERNACAO AND c.source_db = i.source_db
        WHERE {FEB_ACTIVE}
          AND c.DS_DESCRICAO IS NOT NULL
          AND c.IN_PRINCIPAL = 'S'
        GROUP BY c.DS_DESCRICAO
        ORDER BY n_internacoes DESC
        LIMIT 15
    """)
    profile["top_cids_volume"] = _rows_to_dicts(cols, rows)

    # LOS distribution for Feb discharges
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
            SELECT DATEDIFF('day', DH_ADMISSAO_HOSP::DATE, DH_FINALIZACAO::DATE) AS los
            FROM agg_tb_capta_internacao_cain
            WHERE {SRC} AND IN_SITUACAO = 2
              AND DH_FINALIZACAO >= '{FEB_START}'::DATE
              AND DH_FINALIZACAO < '{FEB_END}'::DATE + INTERVAL '1 day'
              AND DH_ADMISSAO_HOSP IS NOT NULL
        ) sub
        GROUP BY faixa_los, sort_key
        ORDER BY sort_key
    """)
    profile["los_distribution"] = _rows_to_dicts(cols, rows)

    # Discharge types grouped
    cols, rows = _exec(con, f"""
        WITH feb_discharges AS (
            SELECT ID_CD_INTERNACAO
            FROM agg_tb_capta_internacao_cain
            WHERE {SRC} AND IN_SITUACAO = 2
              AND DH_FINALIZACAO >= '{FEB_START}'::DATE
              AND DH_FINALIZACAO < '{FEB_END}'::DATE + INTERVAL '1 day'
        ),
        last_status AS (
            SELECT es.ID_CD_INTERNACAO,
                   ROW_NUMBER() OVER (PARTITION BY es.ID_CD_INTERNACAO ORDER BY es.DH_CADASTRO DESC) AS rn,
                   es.FL_DESOSPITALIZACAO
            FROM agg_tb_capta_evo_status_caes es
            WHERE es.{SRC}
              AND es.ID_CD_INTERNACAO IN (SELECT ID_CD_INTERNACAO FROM feb_discharges)
        )
        SELECT f.DS_FINAL_MONITORAMENTO AS tipo_alta,
               COUNT(DISTINCT ls.ID_CD_INTERNACAO) AS n_internacoes
        FROM last_status ls
        JOIN agg_tb_capta_tipo_final_monit_fmon f
            ON ls.FL_DESOSPITALIZACAO = f.ID_CD_FINAL_MONITORAMENTO
            AND f.{SRC}
        WHERE ls.rn = 1
        GROUP BY tipo_alta
        ORDER BY n_internacoes DESC
    """)
    raw_dtypes = _rows_to_dicts(cols, rows)
    grouped = {}
    for d in raw_dtypes:
        grp = _group_discharge_type(d.get("tipo_alta", ""))
        grouped[grp] = grouped.get(grp, 0) + _safe_int(d.get("n_internacoes"))
    profile["discharge_types_grouped"] = [
        {"grupo": g, "n": n}
        for g, n in sorted(grouped.items(), key=lambda x: -x[1])
    ]

    print(f"    done in {time.time()-t0:.1f}s")
    return profile


def _fetch_los_analysis(con):
    """LOS deep-dive: by CID, excess, same CID across hospitals."""
    import time
    print("[05/14] LOS analysis ...")
    t0 = time.time()

    FEB_DIS = f"""
        i.{SRC}
        AND i.IN_SITUACAO = 2
        AND i.DH_FINALIZACAO >= '{FEB_START}'::DATE
        AND i.DH_FINALIZACAO < '{FEB_END}'::DATE + INTERVAL '1 day'
        AND i.DH_ADMISSAO_HOSP IS NOT NULL
    """

    los = {}

    # LOS by CID top 10 by avg LOS
    cols, rows = _exec(con, f"""
        SELECT c.DS_DESCRICAO AS cid_desc,
            COUNT(DISTINCT i.ID_CD_INTERNACAO) AS n,
            AVG(DATEDIFF('day', i.DH_ADMISSAO_HOSP::DATE, i.DH_FINALIZACAO::DATE)) AS avg_los,
            MEDIAN(DATEDIFF('day', i.DH_ADMISSAO_HOSP::DATE, i.DH_FINALIZACAO::DATE)) AS median_los,
            MAX(DATEDIFF('day', i.DH_ADMISSAO_HOSP::DATE, i.DH_FINALIZACAO::DATE)) AS max_los
        FROM agg_tb_capta_internacao_cain i
        JOIN agg_tb_capta_cid_caci c
            ON i.ID_CD_INTERNACAO = c.ID_CD_INTERNACAO AND i.source_db = c.source_db
        WHERE {FEB_DIS}
          AND c.DS_DESCRICAO IS NOT NULL
          AND c.IN_PRINCIPAL = 'S'
        GROUP BY c.DS_DESCRICAO
        HAVING COUNT(DISTINCT i.ID_CD_INTERNACAO) >= 5
        ORDER BY avg_los DESC
        LIMIT 10
    """)
    los["by_cid"] = _rows_to_dicts(cols, rows)

    # Excess LOS: >2x median for that CID
    cols, rows = _exec(con, f"""
        WITH cid_median AS (
            SELECT c.DS_DESCRICAO AS cid_desc,
                MEDIAN(DATEDIFF('day', i.DH_ADMISSAO_HOSP::DATE,
                    COALESCE(i.DH_FINALIZACAO, CURRENT_TIMESTAMP)::DATE)) AS med_los
            FROM agg_tb_capta_internacao_cain i
            JOIN agg_tb_capta_cid_caci c
                ON i.ID_CD_INTERNACAO = c.ID_CD_INTERNACAO AND i.source_db = c.source_db
            WHERE i.{SRC} AND c.DS_DESCRICAO IS NOT NULL AND c.IN_PRINCIPAL = 'S'
              AND i.DH_ADMISSAO_HOSP IS NOT NULL
            GROUP BY c.DS_DESCRICAO
            HAVING COUNT(DISTINCT i.ID_CD_INTERNACAO) >= 10
        )
        SELECT cm.cid_desc,
            COUNT(DISTINCT i.ID_CD_INTERNACAO) AS n_excess,
            cm.med_los,
            AVG(DATEDIFF('day', i.DH_ADMISSAO_HOSP::DATE, i.DH_FINALIZACAO::DATE)) AS avg_actual,
            SUM(DATEDIFF('day', i.DH_ADMISSAO_HOSP::DATE, i.DH_FINALIZACAO::DATE) - cm.med_los) AS excess_days
        FROM agg_tb_capta_internacao_cain i
        JOIN agg_tb_capta_cid_caci c
            ON i.ID_CD_INTERNACAO = c.ID_CD_INTERNACAO AND i.source_db = c.source_db
        JOIN cid_median cm ON c.DS_DESCRICAO = cm.cid_desc
        WHERE i.{SRC}
          AND c.IN_PRINCIPAL = 'S'
          AND i.IN_SITUACAO = 2
          AND i.DH_FINALIZACAO >= '{FEB_START}'::DATE
          AND i.DH_FINALIZACAO < '{FEB_END}'::DATE + INTERVAL '1 day'
          AND i.DH_ADMISSAO_HOSP IS NOT NULL
          AND DATEDIFF('day', i.DH_ADMISSAO_HOSP::DATE, i.DH_FINALIZACAO::DATE) > 2 * cm.med_los
        GROUP BY cm.cid_desc, cm.med_los
        ORDER BY excess_days DESC
        LIMIT 10
    """)
    los["excess"] = _rows_to_dicts(cols, rows)

    print(f"    done in {time.time()-t0:.1f}s")
    return los


def _fetch_risk_matrix(con):
    """FLAW 2a: CID x Hospital risk matrix. Top 10 CIDs x Top 10 hospitals.
    Flag cells where LOS > 1.5x the CID median across all hospitals."""
    import time
    print("[06/14] Risk matrix CID x Hospital ...")
    t0 = time.time()

    FEB_DIS = f"""
        i.{SRC}
        AND i.IN_SITUACAO = 2
        AND i.DH_FINALIZACAO >= '{FEB_START}'::DATE
        AND i.DH_FINALIZACAO < '{FEB_END}'::DATE + INTERVAL '1 day'
        AND i.DH_ADMISSAO_HOSP IS NOT NULL
    """

    # Top 10 CIDs by volume
    cols, rows = _exec(con, f"""
        SELECT c.DS_DESCRICAO AS cid_desc,
               COUNT(DISTINCT i.ID_CD_INTERNACAO) AS n
        FROM agg_tb_capta_internacao_cain i
        JOIN agg_tb_capta_cid_caci c
            ON i.ID_CD_INTERNACAO = c.ID_CD_INTERNACAO AND i.source_db = c.source_db
        WHERE {FEB_DIS} AND c.IN_PRINCIPAL = 'S' AND c.DS_DESCRICAO IS NOT NULL
        GROUP BY c.DS_DESCRICAO
        HAVING COUNT(DISTINCT i.ID_CD_INTERNACAO) >= 5
        ORDER BY n DESC
        LIMIT 10
    """)
    top_cids = [r[0] for r in rows]

    if not top_cids:
        print(f"    no CIDs found, done in {time.time()-t0:.1f}s")
        return {"cells": [], "cid_medians": {}, "top_cids": [], "top_hospitals": []}

    cid_list_sql = ", ".join(f"'{c}'" for c in top_cids)

    # CID-global medians
    cols, rows = _exec(con, f"""
        SELECT c.DS_DESCRICAO AS cid_desc,
               MEDIAN(DATEDIFF('day', i.DH_ADMISSAO_HOSP::DATE, i.DH_FINALIZACAO::DATE)) AS median_los
        FROM agg_tb_capta_internacao_cain i
        JOIN agg_tb_capta_cid_caci c
            ON i.ID_CD_INTERNACAO = c.ID_CD_INTERNACAO AND i.source_db = c.source_db
        WHERE {FEB_DIS} AND c.IN_PRINCIPAL = 'S'
          AND c.DS_DESCRICAO IN ({cid_list_sql})
        GROUP BY c.DS_DESCRICAO
    """)
    cid_medians = {r[0]: _safe_float(r[1]) for r in rows}

    # Cross: CID x Hospital
    cols, rows = _exec(con, f"""
        SELECT c.DS_DESCRICAO AS cid_desc,
               i.ID_CD_HOSPITAL AS hospital_id,
               COUNT(DISTINCT i.ID_CD_INTERNACAO) AS n,
               AVG(DATEDIFF('day', i.DH_ADMISSAO_HOSP::DATE, i.DH_FINALIZACAO::DATE)) AS avg_los
        FROM agg_tb_capta_internacao_cain i
        JOIN agg_tb_capta_cid_caci c
            ON i.ID_CD_INTERNACAO = c.ID_CD_INTERNACAO AND i.source_db = c.source_db
        WHERE {FEB_DIS} AND c.IN_PRINCIPAL = 'S'
          AND c.DS_DESCRICAO IN ({cid_list_sql})
        GROUP BY c.DS_DESCRICAO, i.ID_CD_HOSPITAL
        HAVING COUNT(DISTINCT i.ID_CD_INTERNACAO) >= 3
        ORDER BY c.DS_DESCRICAO, avg_los DESC
    """)
    cells = _rows_to_dicts(cols, rows)

    # Identify top 10 hospitals by total volume in this matrix
    hosp_vol = {}
    for c in cells:
        hid = c["hospital_id"]
        hosp_vol[hid] = hosp_vol.get(hid, 0) + _safe_int(c["n"])
    top_hosps = sorted(hosp_vol, key=lambda h: -hosp_vol[h])[:10]

    print(f"    {len(cells)} cells, done in {time.time()-t0:.1f}s")
    return {
        "cells": cells,
        "cid_medians": cid_medians,
        "top_cids": top_cids,
        "top_hospitals": top_hosps,
    }


def _fetch_doctor_cid_comparison(con):
    """FLAW 2b: Doctor x CID LOS comparison. For doctors with >=5 cases of the SAME CID,
    compare their LOS to the CID median."""
    import time
    print("[07/14] Doctor x CID comparison ...")
    t0 = time.time()

    CRM_FILTER = """
        m.DS_CONSELHO_CLASSE IS NOT NULL
        AND m.DS_CONSELHO_CLASSE NOT LIKE '0000%%'
        AND LENGTH(m.DS_CONSELHO_CLASSE) >= 4
    """

    cols, rows = _exec(con, f"""
        WITH feb_active AS (
            SELECT i.ID_CD_INTERNACAO,
                   DATEDIFF('day', i.DH_ADMISSAO_HOSP::DATE,
                       COALESCE(i.DH_FINALIZACAO, CURRENT_TIMESTAMP)::DATE) AS los
            FROM agg_tb_capta_internacao_cain i
            WHERE i.{SRC}
              AND i.DH_ADMISSAO_HOSP < '{FEB_END}'::DATE + INTERVAL '1 day'
              AND (i.DH_FINALIZACAO IS NULL OR i.DH_FINALIZACAO >= '{FEB_START}'::DATE)
        ),
        doc_cid AS (
            SELECT
                m.DS_CONSELHO_CLASSE AS crm,
                m.NM_MEDICO_HOSPITAL AS nome,
                c.DS_DESCRICAO AS cid_desc,
                fa.ID_CD_INTERNACAO,
                fa.los
            FROM agg_tb_capta_internacao_medico_hospital_cimh mh
            JOIN agg_tb_capta_cfg_medico_hospital_ccmh m
                ON mh.ID_CD_MEDICO_HOSPITAL = m.ID_CD_MEDICO_HOSPITAL
                AND mh.source_db = m.source_db
            JOIN feb_active fa ON mh.ID_CD_INTERNACAO = fa.ID_CD_INTERNACAO
            JOIN agg_tb_capta_cid_caci c
                ON mh.ID_CD_INTERNACAO = c.ID_CD_INTERNACAO
                AND mh.source_db = c.source_db
            WHERE mh.{SRC}
              AND {CRM_FILTER}
              AND c.IN_PRINCIPAL = 'S'
              AND c.DS_DESCRICAO IS NOT NULL
        ),
        cid_medians AS (
            SELECT cid_desc, MEDIAN(los) AS med_los
            FROM doc_cid
            GROUP BY cid_desc
            HAVING COUNT(DISTINCT ID_CD_INTERNACAO) >= 10
        )
        SELECT
            dc.crm, dc.nome, dc.cid_desc,
            COUNT(DISTINCT dc.ID_CD_INTERNACAO) AS n_cases,
            AVG(dc.los) AS doc_avg_los,
            cm.med_los AS cid_median_los,
            AVG(dc.los) - cm.med_los AS los_excess
        FROM doc_cid dc
        JOIN cid_medians cm ON dc.cid_desc = cm.cid_desc
        GROUP BY dc.crm, dc.nome, dc.cid_desc, cm.med_los
        HAVING COUNT(DISTINCT dc.ID_CD_INTERNACAO) >= 5
        ORDER BY los_excess DESC
        LIMIT 20
    """)
    result = _rows_to_dicts(cols, rows)
    print(f"    {len(result)} doctor-CID pairs, done in {time.time()-t0:.1f}s")
    return result


def _fetch_med_intensity_matrix(con):
    """FLAW 2c: Medication intensity CID x Hospital — which hospitals prescribe
    significantly more medications per admission for the same CID."""
    import time
    print("[08/14] Medication intensity CID x Hospital ...")
    t0 = time.time()

    cols, rows = _exec(con, f"""
        WITH feb_active AS (
            SELECT ID_CD_INTERNACAO
            FROM agg_tb_capta_internacao_cain
            WHERE {SRC}
              AND DH_ADMISSAO_HOSP < '{FEB_END}'::DATE + INTERVAL '1 day'
              AND (DH_FINALIZACAO IS NULL OR DH_FINALIZACAO >= '{FEB_START}'::DATE)
        ),
        inter_meds AS (
            SELECT p.ID_CD_INTERNACAO, COUNT(*) AS n_meds
            FROM agg_tb_capta_produtos_capr p
            WHERE p.{SRC}
              AND p.ID_CD_INTERNACAO IN (SELECT ID_CD_INTERNACAO FROM feb_active)
            GROUP BY p.ID_CD_INTERNACAO
        ),
        cid_hosp_meds AS (
            SELECT c.DS_DESCRICAO AS cid_desc,
                   i.ID_CD_HOSPITAL AS hospital_id,
                   i.ID_CD_INTERNACAO,
                   COALESCE(im.n_meds, 0) AS n_meds
            FROM agg_tb_capta_internacao_cain i
            JOIN agg_tb_capta_cid_caci c
                ON i.ID_CD_INTERNACAO = c.ID_CD_INTERNACAO AND i.source_db = c.source_db
            LEFT JOIN inter_meds im ON i.ID_CD_INTERNACAO = im.ID_CD_INTERNACAO
            WHERE i.{SRC}
              AND i.DH_ADMISSAO_HOSP < '{FEB_END}'::DATE + INTERVAL '1 day'
              AND (i.DH_FINALIZACAO IS NULL OR i.DH_FINALIZACAO >= '{FEB_START}'::DATE)
              AND c.IN_PRINCIPAL = 'S'
              AND c.DS_DESCRICAO IS NOT NULL
        ),
        cid_global AS (
            SELECT cid_desc,
                   AVG(n_meds) AS global_avg_meds,
                   COUNT(DISTINCT ID_CD_INTERNACAO) AS global_n
            FROM cid_hosp_meds
            GROUP BY cid_desc
            HAVING COUNT(DISTINCT ID_CD_INTERNACAO) >= 10
        )
        SELECT chm.cid_desc, chm.hospital_id,
               COUNT(DISTINCT chm.ID_CD_INTERNACAO) AS n,
               AVG(chm.n_meds) AS hosp_avg_meds,
               cg.global_avg_meds,
               AVG(chm.n_meds) / NULLIF(cg.global_avg_meds, 0) AS ratio
        FROM cid_hosp_meds chm
        JOIN cid_global cg ON chm.cid_desc = cg.cid_desc
        GROUP BY chm.cid_desc, chm.hospital_id, cg.global_avg_meds
        HAVING COUNT(DISTINCT chm.ID_CD_INTERNACAO) >= 3
           AND AVG(chm.n_meds) / NULLIF(cg.global_avg_meds, 0) > 1.5
        ORDER BY ratio DESC
        LIMIT 15
    """)
    result = _rows_to_dicts(cols, rows)
    print(f"    {len(result)} med-intensity anomalies, done in {time.time()-t0:.1f}s")
    return result


def _fetch_physicians(con):
    """Physician variation: CRM-validated, min 5 cases."""
    import time
    print("[09/14] Physician analysis ...")
    t0 = time.time()

    CRM_FILTER = """
        m.DS_CONSELHO_CLASSE IS NOT NULL
        AND m.DS_CONSELHO_CLASSE NOT LIKE '0000%%'
        AND LENGTH(m.DS_CONSELHO_CLASSE) >= 4
    """

    docs = {}

    cols, rows = _exec(con, f"""
        WITH feb_admissions AS (
            SELECT i.ID_CD_INTERNACAO,
                   DATEDIFF('day', i.DH_ADMISSAO_HOSP::DATE,
                       COALESCE(i.DH_FINALIZACAO, CURRENT_TIMESTAMP)::DATE) AS los
            FROM agg_tb_capta_internacao_cain i
            WHERE i.{SRC}
              AND i.DH_ADMISSAO_HOSP < '{FEB_END}'::DATE + INTERVAL '1 day'
              AND (i.DH_FINALIZACAO IS NULL OR i.DH_FINALIZACAO >= '{FEB_START}'::DATE)
        ),
        doc_admissions AS (
            SELECT
                m.DS_CONSELHO_CLASSE AS crm,
                m.NM_MEDICO_HOSPITAL AS nome,
                fa.ID_CD_INTERNACAO,
                fa.los
            FROM agg_tb_capta_internacao_medico_hospital_cimh mh
            JOIN agg_tb_capta_cfg_medico_hospital_ccmh m
                ON mh.ID_CD_MEDICO_HOSPITAL = m.ID_CD_MEDICO_HOSPITAL
                AND mh.source_db = m.source_db
            JOIN feb_admissions fa ON mh.ID_CD_INTERNACAO = fa.ID_CD_INTERNACAO
            WHERE mh.{SRC}
              AND {CRM_FILTER}
        )
        SELECT
            da.crm,
            da.nome,
            COUNT(DISTINCT da.ID_CD_INTERNACAO) AS n_cases,
            AVG(da.los) AS avg_los,
            MEDIAN(da.los) AS median_los,
            COALESCE(AVG(pr.n_meds), 0) AS avg_meds_per_case,
            COALESCE(SUM(pr.total_qty), 0) AS total_med_units
        FROM doc_admissions da
        LEFT JOIN (
            SELECT ID_CD_INTERNACAO,
                   COUNT(*) AS n_meds,
                   SUM(NR_QUANTIDADE) AS total_qty
            FROM agg_tb_capta_produtos_capr WHERE {SRC}
            GROUP BY ID_CD_INTERNACAO
        ) pr ON da.ID_CD_INTERNACAO = pr.ID_CD_INTERNACAO
        GROUP BY da.crm, da.nome
        HAVING COUNT(DISTINCT da.ID_CD_INTERNACAO) >= 5
        ORDER BY avg_los DESC
        LIMIT 20
    """)
    docs["variation_los"] = _rows_to_dicts(cols, rows)

    # Physician mortality
    cols, rows = _exec(con, f"""
        WITH feb_discharged AS (
            SELECT ID_CD_INTERNACAO
            FROM agg_tb_capta_internacao_cain
            WHERE {SRC} AND IN_SITUACAO = 2
              AND DH_FINALIZACAO >= '{FEB_START}'::DATE
              AND DH_FINALIZACAO < '{FEB_END}'::DATE + INTERVAL '1 day'
        ),
        last_status AS (
            SELECT es.ID_CD_INTERNACAO,
                   ROW_NUMBER() OVER (PARTITION BY es.ID_CD_INTERNACAO ORDER BY es.DH_CADASTRO DESC) AS rn,
                   es.FL_DESOSPITALIZACAO
            FROM agg_tb_capta_evo_status_caes es
            WHERE es.{SRC}
              AND es.ID_CD_INTERNACAO IN (SELECT ID_CD_INTERNACAO FROM feb_discharged)
        ),
        discharge_info AS (
            SELECT ls.ID_CD_INTERNACAO,
                   CASE WHEN UPPER(f.DS_FINAL_MONITORAMENTO) LIKE '%%BITO%%'
                        THEN 1 ELSE 0 END AS is_obito
            FROM last_status ls
            JOIN agg_tb_capta_tipo_final_monit_fmon f
                ON ls.FL_DESOSPITALIZACAO = f.ID_CD_FINAL_MONITORAMENTO
                AND f.{SRC}
            WHERE ls.rn = 1
        ),
        doc_discharge AS (
            SELECT
                m.DS_CONSELHO_CLASSE AS crm,
                m.NM_MEDICO_HOSPITAL AS nome,
                mh.ID_CD_INTERNACAO,
                di.is_obito
            FROM agg_tb_capta_internacao_medico_hospital_cimh mh
            JOIN agg_tb_capta_cfg_medico_hospital_ccmh m
                ON mh.ID_CD_MEDICO_HOSPITAL = m.ID_CD_MEDICO_HOSPITAL
                AND mh.source_db = m.source_db
            JOIN discharge_info di ON mh.ID_CD_INTERNACAO = di.ID_CD_INTERNACAO
            WHERE mh.{SRC}
              AND {CRM_FILTER}
        )
        SELECT
            crm, nome,
            COUNT(DISTINCT ID_CD_INTERNACAO) AS n_discharged,
            SUM(is_obito) AS n_obito,
            100.0 * SUM(is_obito) / NULLIF(COUNT(DISTINCT ID_CD_INTERNACAO), 0) AS taxa_obito
        FROM doc_discharge
        GROUP BY crm, nome
        HAVING COUNT(DISTINCT ID_CD_INTERNACAO) >= 5
        ORDER BY taxa_obito DESC
        LIMIT 15
    """)
    docs["mortality"] = _rows_to_dicts(cols, rows)

    print(f"    done in {time.time()-t0:.1f}s")
    return docs


def _fetch_adverse_events(con):
    """Adverse events by TYPE, by hospital, multi-event."""
    import time
    print("[10/14] Adverse events ...")
    t0 = time.time()

    ae = {}

    try:
        # Exclude 'Outros' from main ranking — it's a catch-all that hides real types
        cols, rows = _exec(con, f"""
            SELECT
                COALESCE(t.DS_TITULO, 'Tipo nao informado') AS tipo_evento,
                COUNT(*) AS n_eventos
            FROM agg_tb_capta_eventos_adversos_caed e
            LEFT JOIN agg_tb_capta_cfg_tipos_eventos_adversos_ctea t
                ON e.ID_CD_TIPO_EVENTO = t.ID_CD_TIPO_EVENTO
                AND e.source_db = t.source_db
            WHERE e.{SRC}
              AND e.DH_EVENTO_ADVERSO >= '{FEB_START}'::DATE
              AND e.DH_EVENTO_ADVERSO < '{FEB_END}'::DATE + INTERVAL '1 day'
              AND COALESCE(t.DS_TITULO, '') != 'Outros'
            GROUP BY tipo_evento
            ORDER BY n_eventos DESC
            LIMIT 15
        """)
        ae["by_type"] = _rows_to_dicts(cols, rows)

        # Count 'Outros' separately for the note
        cols2, rows2 = _exec(con, f"""
            SELECT COUNT(*) AS n_outros
            FROM agg_tb_capta_eventos_adversos_caed e
            LEFT JOIN agg_tb_capta_cfg_tipos_eventos_adversos_ctea t
                ON e.ID_CD_TIPO_EVENTO = t.ID_CD_TIPO_EVENTO
                AND e.source_db = t.source_db
            WHERE e.{SRC}
              AND e.DH_EVENTO_ADVERSO >= '{FEB_START}'::DATE
              AND e.DH_EVENTO_ADVERSO < '{FEB_END}'::DATE + INTERVAL '1 day'
              AND COALESCE(t.DS_TITULO, '') = 'Outros'
        """)
        ae["n_outros"] = _safe_int(rows2[0][0]) if rows2 else 0
    except Exception as ex:
        print(f"    AE by type (ctea) failed: {ex}, fallback to DS_DESCRICAO")
        cols, rows = _exec(con, f"""
            SELECT COALESCE(e.DS_DESCRICAO, 'Tipo nao informado') AS tipo_evento,
                   COUNT(*) AS n_eventos
            FROM agg_tb_capta_eventos_adversos_caed e
            WHERE e.{SRC}
              AND e.DH_EVENTO_ADVERSO >= '{FEB_START}'::DATE
              AND e.DH_EVENTO_ADVERSO < '{FEB_END}'::DATE + INTERVAL '1 day'
            GROUP BY tipo_evento
            ORDER BY n_eventos DESC
            LIMIT 15
        """)
        ae["by_type"] = _rows_to_dicts(cols, rows)
        ae["n_outros"] = 0

    try:
        cols, rows = _exec(con, f"""
            SELECT i.ID_CD_HOSPITAL AS hospital_id, COUNT(*) AS n_eventos
            FROM agg_tb_capta_eventos_adversos_caed e
            JOIN agg_tb_capta_internacao_cain i
                ON e.ID_CD_INTERNACAO = i.ID_CD_INTERNACAO AND e.source_db = i.source_db
            WHERE e.{SRC}
              AND e.DH_EVENTO_ADVERSO >= '{FEB_START}'::DATE
              AND e.DH_EVENTO_ADVERSO < '{FEB_END}'::DATE + INTERVAL '1 day'
            GROUP BY i.ID_CD_HOSPITAL
            ORDER BY n_eventos DESC
            LIMIT 10
        """)
        ae["by_hospital"] = _rows_to_dicts(cols, rows)
    except Exception:
        ae["by_hospital"] = []

    try:
        cols, rows = _exec(con, f"""
            SELECT
                e.ID_CD_INTERNACAO,
                i.ID_CD_HOSPITAL,
                COUNT(*) AS n_eventos,
                STRING_AGG(DISTINCT COALESCE(t.DS_TITULO, e.DS_DESCRICAO, '?'), ' | ') AS tipos
            FROM agg_tb_capta_eventos_adversos_caed e
            JOIN agg_tb_capta_internacao_cain i
                ON e.ID_CD_INTERNACAO = i.ID_CD_INTERNACAO AND e.source_db = i.source_db
            LEFT JOIN agg_tb_capta_cfg_tipos_eventos_adversos_ctea t
                ON e.ID_CD_TIPO_EVENTO = t.ID_CD_TIPO_EVENTO
                AND e.source_db = t.source_db
            WHERE e.{SRC}
              AND e.DH_EVENTO_ADVERSO >= '{FEB_START}'::DATE
              AND e.DH_EVENTO_ADVERSO < '{FEB_END}'::DATE + INTERVAL '1 day'
            GROUP BY e.ID_CD_INTERNACAO, i.ID_CD_HOSPITAL
            HAVING COUNT(*) >= 2
            ORDER BY n_eventos DESC
            LIMIT 10
        """)
        ae["multi_event"] = _rows_to_dicts(cols, rows)
    except Exception:
        ae["multi_event"] = []

    print(f"    done in {time.time()-t0:.1f}s")
    return ae


def _fetch_medications(con):
    """Medications: top 20, intensity by CID, top admissions by volume."""
    import time
    print("[11/14] Medications ...")
    t0 = time.time()

    FEB_JOIN = f"""
        p.{SRC}
        AND i.DH_ADMISSAO_HOSP < '{FEB_END}'::DATE + INTERVAL '1 day'
        AND (i.DH_FINALIZACAO IS NULL OR i.DH_FINALIZACAO >= '{FEB_START}'::DATE)
    """

    meds = {}

    cols, rows = _exec(con, f"""
        SELECT
            COUNT(*) AS total_produtos,
            COUNT(DISTINCT p.ID_CD_INTERNACAO) AS n_internacoes_com_produto,
            COUNT(DISTINCT p.NM_TITULO_COMERCIAL) AS n_medicamentos_distintos,
            SUM(p.NR_QUANTIDADE) AS total_unidades
        FROM agg_tb_capta_produtos_capr p
        JOIN agg_tb_capta_internacao_cain i
            ON p.ID_CD_INTERNACAO = i.ID_CD_INTERNACAO AND p.source_db = i.source_db
        WHERE {FEB_JOIN}
    """)
    meds["summary"] = dict(zip(cols, rows[0])) if rows else {}

    cols, rows = _exec(con, f"""
        SELECT
            p.NM_TITULO_COMERCIAL AS medicamento,
            COUNT(*) AS n_prescricoes,
            SUM(p.NR_QUANTIDADE) AS total_unidades,
            COUNT(DISTINCT p.ID_CD_INTERNACAO) AS n_internacoes
        FROM agg_tb_capta_produtos_capr p
        JOIN agg_tb_capta_internacao_cain i
            ON p.ID_CD_INTERNACAO = i.ID_CD_INTERNACAO AND p.source_db = i.source_db
        WHERE {FEB_JOIN} AND p.NM_TITULO_COMERCIAL IS NOT NULL
        GROUP BY p.NM_TITULO_COMERCIAL
        ORDER BY n_prescricoes DESC
        LIMIT 20
    """)
    meds["top_medications"] = _rows_to_dicts(cols, rows)

    cols, rows = _exec(con, f"""
        SELECT
            c.DS_DESCRICAO AS cid_desc,
            COUNT(*) AS n_prescricoes,
            COUNT(DISTINCT p.ID_CD_INTERNACAO) AS n_internacoes,
            ROUND(COUNT(*) * 1.0 / COUNT(DISTINCT p.ID_CD_INTERNACAO), 1) AS prescricoes_por_inter
        FROM agg_tb_capta_produtos_capr p
        JOIN agg_tb_capta_internacao_cain i
            ON p.ID_CD_INTERNACAO = i.ID_CD_INTERNACAO AND p.source_db = i.source_db
        JOIN agg_tb_capta_cid_caci c
            ON p.ID_CD_INTERNACAO = c.ID_CD_INTERNACAO AND p.source_db = c.source_db
        WHERE {FEB_JOIN}
          AND c.DS_DESCRICAO IS NOT NULL
          AND c.IN_PRINCIPAL = 'S'
        GROUP BY c.DS_DESCRICAO
        HAVING COUNT(DISTINCT p.ID_CD_INTERNACAO) >= 5
        ORDER BY prescricoes_por_inter DESC
        LIMIT 10
    """)
    meds["intensity_by_cid"] = _rows_to_dicts(cols, rows)

    print(f"    done in {time.time()-t0:.1f}s")
    return meds


def _load_twin_predictive(con):
    """FLAW 3: Twin as predictive spine.
    - Death proximity score for open admissions
    - Trajectory velocity for patients with 2+ admissions
    - March projection (centroid movement Jan->Feb->forecast)
    Also returns anomalies for backward compatibility."""
    import time
    import numpy as np
    import torch
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    import pyarrow as pa

    print("[12/14] V6.2 predictive twin ...")
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

    # ---- Get all BRADESCO internacoes ----
    cols, rows = _exec(con, f"""
        SELECT i.ID_CD_INTERNACAO, i.ID_CD_PACIENTE, i.ID_CD_HOSPITAL,
               i.DH_ADMISSAO_HOSP, i.DH_FINALIZACAO, i.IN_SITUACAO,
               DATEDIFF('day', i.DH_ADMISSAO_HOSP::DATE,
                   COALESCE(i.DH_FINALIZACAO, CURRENT_TIMESTAMP)::DATE) AS los
        FROM agg_tb_capta_internacao_cain i
        WHERE i.{SRC} AND i.DH_ADMISSAO_HOSP IS NOT NULL
    """)
    all_admissions = _rows_to_dicts(cols, rows)

    # Map iid -> admission info
    adm_map = {}
    for a in all_admissions:
        iid = a["ID_CD_INTERNACAO"]
        key = f"{SOURCE_DB}/ID_CD_INTERNACAO_{iid}"
        idx = node_to_idx.get(key)
        if idx is not None:
            a["emb_idx"] = idx
            adm_map[iid] = a

    print(f"    {len(adm_map):,} admissions found in embedding space")

    # ---- Identify OBITO admissions (death centroid) ----
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
    print(f"    {len(obito_iids):,} total OBITO admissions in DB")

    # Build death centroid
    obito_indices = []
    for iid in obito_iids:
        if iid in adm_map:
            obito_indices.append(adm_map[iid]["emb_idx"])

    twin_result = {
        "death_proximity": [],
        "trajectory_velocity": [],
        "march_projection": {},
        "anomalies": [],
        "stats": {},
    }

    if len(obito_indices) < 5:
        print("    Too few OBITO embeddings for death centroid")
        twin_result["stats"] = {
            "n_obito_embedded": len(obito_indices),
            "embedding_dim": emb.shape[1],
            "total_nodes": len(unique_nodes),
        }
        return twin_result

    obito_vecs = emb[obito_indices]
    death_centroid = obito_vecs.mean(axis=0)
    death_centroid_norm = death_centroid / (np.linalg.norm(death_centroid) + 1e-9)
    print(f"    Death centroid built from {len(obito_indices)} OBITO embeddings")

    # ---- Build ALTA MELHORADA centroid ----
    cols_alta, rows_alta = _exec(con, f"""
        WITH last_st AS (
            SELECT es.ID_CD_INTERNACAO,
                   ROW_NUMBER() OVER (PARTITION BY es.ID_CD_INTERNACAO ORDER BY es.DH_CADASTRO DESC) AS rn,
                   es.FL_DESOSPITALIZACAO
            FROM agg_tb_capta_evo_status_caes es WHERE es.{SRC}
        )
        SELECT DISTINCT ls.ID_CD_INTERNACAO
        FROM last_st ls
        JOIN agg_tb_capta_tipo_final_monit_fmon f
            ON ls.FL_DESOSPITALIZACAO = f.ID_CD_FINAL_MONITORAMENTO AND f.{SRC}
        WHERE ls.rn = 1 AND UPPER(f.DS_FINAL_MONITORAMENTO) LIKE '%%MELHORAD%%'
    """)
    alta_iids = {r[0] for r in rows_alta}
    alta_indices = []
    for iid_a in alta_iids:
        if iid_a in adm_map:
            alta_indices.append(adm_map[iid_a]["emb_idx"])

    alta_centroid_norm = None
    outcome_vector_norm = None
    if len(alta_indices) >= 5:
        alta_vecs = emb[alta_indices]
        alta_centroid = alta_vecs.mean(axis=0)
        alta_centroid_norm = alta_centroid / (np.linalg.norm(alta_centroid) + 1e-9)
        outcome_vector = death_centroid - alta_centroid
        outcome_vector_norm = outcome_vector / (np.linalg.norm(outcome_vector) + 1e-9)
        print(f"    Alta centroid built from {len(alta_indices)} ALTA MELHORADA embeddings")
    else:
        print(f"    Too few ALTA MELHORADA embeddings ({len(alta_indices)}) for alta centroid")

    # ---- 3a: Death proximity for OPEN cases ----
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
        # Get CID — try principal first, fallback to any CID
        try:
            c2, r2 = _exec(con, f"""
                SELECT STRING_AGG(DISTINCT DS_DESCRICAO, ' | ') AS cids
                FROM agg_tb_capta_cid_caci
                WHERE {SRC} AND ID_CD_INTERNACAO = {iid}
                  AND DS_DESCRICAO IS NOT NULL AND IN_PRINCIPAL = 'S'
            """)
            cid_val = r2[0][0] if r2 and r2[0][0] else None
            if not cid_val:
                # Fallback: any CID for this admission
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
        death_proximity_list.append(oa)

    death_proximity_list.sort(key=lambda x: -x["death_proximity"])
    twin_result["death_proximity"] = death_proximity_list[:30]
    print(f"    {len(death_proximity_list)} open cases scored for death proximity")

    # ---- 3b: Trajectory velocity for patients with 2+ admissions ----
    # Group admissions by patient, sorted chronologically
    patient_admissions: dict[int, list] = {}
    for iid, a in adm_map.items():
        pid = a["ID_CD_PACIENTE"]
        if pid is not None:
            if pid not in patient_admissions:
                patient_admissions[pid] = []
            patient_admissions[pid].append(a)

    trajectory_list = []
    for pid, admissions in patient_admissions.items():
        if len(admissions) < 2:
            continue
        admissions.sort(key=lambda x: str(x.get("DH_ADMISSAO_HOSP", "")))
        prev = admissions[-2]
        curr = admissions[-1]
        prev_vec = emb[prev["emb_idx"]]
        curr_vec = emb[curr["emb_idx"]]
        delta = curr_vec - prev_vec
        velocity = float(np.linalg.norm(delta))
        # Direction: cosine of delta with death centroid
        delta_norm = delta / (np.linalg.norm(delta) + 1e-9)
        direction_to_death = float(np.dot(delta_norm, death_centroid_norm))
        # Risk score: velocity * direction (higher = moving fast toward death)
        risk_score = velocity * max(direction_to_death, 0)
        curr_cos = float(np.dot(curr_vec / (np.linalg.norm(curr_vec) + 1e-9), death_centroid_norm))

        # Only include patients whose current admission is still open OR recent
        if curr.get("IN_SITUACAO") != 2 or curr.get("DH_FINALIZACAO") is None:
            is_open = True
        else:
            is_open = False

        trajectory_list.append({
            "ID_CD_PACIENTE": pid,
            "prev_iid": prev["ID_CD_INTERNACAO"],
            "curr_iid": curr["ID_CD_INTERNACAO"],
            "hospital_id": curr.get("ID_CD_HOSPITAL"),
            "velocity": velocity,
            "direction_to_death": direction_to_death,
            "risk_score": risk_score,
            "curr_death_proximity": curr_cos,
            "is_open": is_open,
            "los": curr.get("los", 0),
        })

    trajectory_list.sort(key=lambda x: -x["risk_score"])
    # Save top 50 overall + all open cases (whichever is larger)
    open_trajs = [t for t in trajectory_list if t["is_open"]]
    top_trajs = trajectory_list[:50]
    # Merge: top 50 + all open (dedup by patient)
    seen_pids = set()
    merged = []
    for t in open_trajs + top_trajs:
        if t["ID_CD_PACIENTE"] not in seen_pids:
            seen_pids.add(t["ID_CD_PACIENTE"])
            merged.append(t)
    merged.sort(key=lambda x: -x["risk_score"])
    twin_result["trajectory_velocity"] = merged
    print(f"    {len(trajectory_list)} patients with 2+ admissions scored for trajectory ({len(open_trajs)} open)")

    # ---- 3c: March projection ----
    # Get Feb-active admissions centroid
    feb_indices = []
    for iid, a in adm_map.items():
        adm_date = str(a.get("DH_ADMISSAO_HOSP", ""))[:10]
        fin_date = str(a.get("DH_FINALIZACAO", ""))[:10] if a.get("DH_FINALIZACAO") else None
        # Feb-active: admitted before end of Feb AND (not discharged OR discharged in/after Feb)
        if adm_date and adm_date < "2026-03-01":
            if fin_date is None or fin_date >= "2026-02-01":
                feb_indices.append(a["emb_idx"])

    # Jan-active
    jan_indices = []
    for iid, a in adm_map.items():
        adm_date = str(a.get("DH_ADMISSAO_HOSP", ""))[:10]
        fin_date = str(a.get("DH_FINALIZACAO", ""))[:10] if a.get("DH_FINALIZACAO") else None
        if adm_date and adm_date < "2026-02-01":
            if fin_date is None or fin_date >= "2026-01-01":
                jan_indices.append(a["emb_idx"])

    if len(feb_indices) > 0 and len(jan_indices) > 0:
        feb_centroid = emb[feb_indices].mean(axis=0)
        jan_centroid = emb[jan_indices].mean(axis=0)
        feb_cos = float(np.dot(
            feb_centroid / (np.linalg.norm(feb_centroid) + 1e-9),
            death_centroid_norm
        ))
        jan_cos = float(np.dot(
            jan_centroid / (np.linalg.norm(jan_centroid) + 1e-9),
            death_centroid_norm
        ))
        drift = feb_cos - jan_cos
        twin_result["march_projection"] = {
            "jan_cos_death": jan_cos,
            "feb_cos_death": feb_cos,
            "drift": drift,
            "direction": "toward death cluster" if drift > 0 else "away from death cluster",
            "n_jan": len(jan_indices),
            "n_feb": len(feb_indices),
        }
        print(f"    March projection: Jan cos={jan_cos:.4f}, Feb cos={feb_cos:.4f}, drift={drift:+.4f}")

    # ---- Hospital drift contribution ----
    # For each hospital, compute how much its Feb admissions' centroid
    # moved toward death vs Jan
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
    print(f"    Hospital drift computed for {len(hospital_drift)} hospitals")

    # ---- Risk-adjusted mortality ----
    # For each doctor: compute average admission-time death proximity of their patients
    # Then compare observed mortality vs expected (based on case severity)
    risk_adj_list = []
    try:
        ra_cols, ra_rows = _exec(con, f"""
            WITH feb_discharges AS (
                SELECT i.ID_CD_INTERNACAO, i.ID_CD_HOSPITAL
                FROM agg_tb_capta_internacao_cain i
                WHERE i.{SRC} AND i.IN_SITUACAO = 2
                  AND i.DH_FINALIZACAO >= '{FEB_START}'::DATE
                  AND i.DH_FINALIZACAO < '{FEB_END}'::DATE + INTERVAL '1 day'
            ),
            doc_links AS (
                SELECT mh.ID_CD_INTERNACAO,
                       m.DS_CONSELHO_CLASSE AS crm,
                       m.NM_MEDICO_HOSPITAL AS nome
                FROM agg_tb_capta_internacao_medico_hospital_cimh mh
                JOIN agg_tb_capta_cfg_medico_hospital_ccmh m
                    ON mh.ID_CD_MEDICO_HOSPITAL = m.ID_CD_MEDICO_HOSPITAL
                    AND mh.source_db = m.source_db
                WHERE mh.{SRC}
                  AND m.DS_CONSELHO_CLASSE IS NOT NULL
                  AND m.DS_CONSELHO_CLASSE NOT LIKE '0000%%'
                  AND mh.ID_CD_INTERNACAO IN (SELECT ID_CD_INTERNACAO FROM feb_discharges)
            ),
            last_st AS (
                SELECT es.ID_CD_INTERNACAO,
                       ROW_NUMBER() OVER (PARTITION BY es.ID_CD_INTERNACAO ORDER BY es.DH_CADASTRO DESC) AS rn,
                       es.FL_DESOSPITALIZACAO
                FROM agg_tb_capta_evo_status_caes es
                WHERE es.{SRC}
                  AND es.ID_CD_INTERNACAO IN (SELECT ID_CD_INTERNACAO FROM feb_discharges)
            ),
            obito_flag AS (
                SELECT DISTINCT ls.ID_CD_INTERNACAO, 1 AS is_obito
                FROM last_st ls
                JOIN agg_tb_capta_tipo_final_monit_fmon f
                    ON ls.FL_DESOSPITALIZACAO = f.ID_CD_FINAL_MONITORAMENTO
                    AND f.{SRC}
                WHERE ls.rn = 1
                  AND UPPER(f.DS_FINAL_MONITORAMENTO) LIKE '%%BITO%%'
            )
            SELECT dl.crm, dl.nome, dl.ID_CD_INTERNACAO,
                   COALESCE(ob.is_obito, 0) AS is_obito
            FROM doc_links dl
            LEFT JOIN obito_flag ob ON dl.ID_CD_INTERNACAO = ob.ID_CD_INTERNACAO
        """)
        # Group by doctor
        doc_cases: dict[str, dict] = {}
        for r in ra_rows:
            crm_val = str(r[0])
            nome_val = str(r[1] or "---")
            iid_val = r[2]
            is_ob = int(r[3])
            if crm_val not in doc_cases:
                doc_cases[crm_val] = {"nome": nome_val, "iids": [], "obitos": 0}
            doc_cases[crm_val]["iids"].append(iid_val)
            doc_cases[crm_val]["obitos"] += is_ob

        for crm_val, info in doc_cases.items():
            if len(info["iids"]) < 5:
                continue
            # Compute average death_proximity for this doctor's patients
            prox_vals = []
            for iid_val in info["iids"]:
                if iid_val in adm_map:
                    idx_val = adm_map[iid_val]["emb_idx"]
                    vec_val = emb[idx_val]
                    vec_n = vec_val / (np.linalg.norm(vec_val) + 1e-9)
                    prox_vals.append(float(np.dot(vec_n, death_centroid_norm)))
            if not prox_vals:
                continue
            avg_risk = float(np.mean(prox_vals))
            n_cases_doc = len(info["iids"])
            n_obito_doc = info["obitos"]
            obs_rate = n_obito_doc / n_cases_doc if n_cases_doc > 0 else 0
            # Expected rate: normalize avg_risk to a mortality probability
            # Use global mortality rate scaled by relative risk
            global_mort = len(obito_iids) / max(len(adm_map), 1)
            # avg_risk is a cosine similarity; higher = more like death
            # Scale: expected = global_mort * (avg_risk / global_avg_risk)
            # Compute global avg death proximity across all BRADESCO admissions
            _all_idxs = np.array([a["emb_idx"] for a in adm_map.values()])
            _all_vecs = emb[_all_idxs]
            _all_norms = np.linalg.norm(_all_vecs, axis=1, keepdims=True).clip(min=1e-9)
            _all_cos = (_all_vecs / _all_norms) @ death_centroid_norm
            all_avg_risk = float(np.mean(_all_cos))
            del _all_idxs, _all_vecs, _all_norms, _all_cos
            expected_rate = global_mort * (avg_risk / (all_avg_risk + 1e-9)) if all_avg_risk > 0 else global_mort
            ratio = obs_rate / (expected_rate + 1e-9) if expected_rate > 0 else 0
            risk_adj_list.append({
                "crm": crm_val,
                "nome": info["nome"],
                "n_cases": n_cases_doc,
                "avg_admission_risk": avg_risk,
                "n_obito": n_obito_doc,
                "observed_rate": obs_rate,
                "expected_rate": expected_rate,
                "risk_adjusted_ratio": ratio,
            })
        risk_adj_list.sort(key=lambda x: -x["risk_adjusted_ratio"])
        print(f"    Risk-adjusted mortality computed for {len(risk_adj_list)} doctors")
    except Exception as ex:
        print(f"    Risk-adjusted mortality FAILED: {ex}")
    twin_result["risk_adjusted_mortality"] = risk_adj_list

    # ---- Anomalies (z-score from centroid, for backward compatibility) ----
    if len(feb_indices) >= 10:
        feb_vecs = emb[feb_indices]
        centroid = feb_vecs.mean(axis=0)
        dists = np.linalg.norm(feb_vecs - centroid, axis=1)
        mean_d = dists.mean()
        std_d = dists.std()

        # Map feb_indices back to iids for z-score
        feb_iid_list = []
        feb_idx_list = []
        for iid, a in adm_map.items():
            adm_date = str(a.get("DH_ADMISSAO_HOSP", ""))[:10]
            fin_date = str(a.get("DH_FINALIZACAO", ""))[:10] if a.get("DH_FINALIZACAO") else None
            if adm_date and adm_date < "2026-03-01":
                if fin_date is None or fin_date >= "2026-02-01":
                    feb_iid_list.append(iid)
                    feb_idx_list.append(a["emb_idx"])

        feb_vecs2 = emb[feb_idx_list]
        dists2 = np.linalg.norm(feb_vecs2 - centroid, axis=1)
        z_scores = (dists2 - mean_d) / (std_d + 1e-9)

        n_anomalies = int((z_scores > Z_THRESHOLD).sum())
        order = np.argsort(-z_scores)
        top_ids = [feb_iid_list[order[i]] for i in range(min(TOP_ANOMALIES, len(order)))
                   if z_scores[order[i]] > Z_THRESHOLD]

        anomalies = []
        for iid in top_ids:
            a = adm_map.get(iid, {})
            z_idx = feb_iid_list.index(iid)
            z = float(z_scores[z_idx])
            entry = {
                "ID_CD_INTERNACAO": iid,
                "ID_CD_PACIENTE": a.get("ID_CD_PACIENTE"),
                "ID_CD_HOSPITAL": a.get("ID_CD_HOSPITAL"),
                "DH_ADMISSAO_HOSP": a.get("DH_ADMISSAO_HOSP"),
                "DH_FINALIZACAO": a.get("DH_FINALIZACAO"),
                "los": a.get("los", 0),
                "z_score": z,
            }
            try:
                c2, r2 = _exec(con, f"""
                    SELECT STRING_AGG(DISTINCT DS_DESCRICAO, ' | ') AS cids
                    FROM agg_tb_capta_cid_caci
                    WHERE {SRC} AND ID_CD_INTERNACAO = {iid}
                      AND DS_DESCRICAO IS NOT NULL AND IN_PRINCIPAL = 'S'
                """)
                entry["cids"] = r2[0][0] if r2 and r2[0][0] else "---"
            except Exception:
                entry["cids"] = "---"
            entry["n_meds"] = 0
            try:
                c3, r3 = _exec(con, f"""
                    SELECT COUNT(*) AS n FROM agg_tb_capta_produtos_capr
                    WHERE {SRC} AND ID_CD_INTERNACAO = {iid}
                """)
                entry["n_meds"] = _safe_int(r3[0][0]) if r3 else 0
            except Exception:
                pass
            anomalies.append(entry)
        anomalies.sort(key=lambda x: -x.get("z_score", 0))
        twin_result["anomalies"] = anomalies

        twin_result["stats"] = {
            "n_feb_embedded": len(feb_indices),
            "n_anomalies": n_anomalies,
            "n_obito_embedded": len(obito_indices),
            "mean_z": float(z_scores.mean()),
            "max_z": float(z_scores.max()),
            "embedding_dim": emb.shape[1],
            "total_nodes": len(unique_nodes),
        }

    # ---- 3e: HOSPITAL PRIORITY LIST ----
    # Merge all open cases with their scores and generate reasons
    print("    Building hospital priority list ...")

    # Get CID medians for LOS explanation
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

    # Build trajectory lookup: patient_id -> trajectory info
    traj_lookup = {}
    for t in trajectory_list:
        if t.get("is_open"):
            traj_lookup[t["ID_CD_PACIENTE"]] = t

    # Get adverse events for open cases
    open_iids = [dp["ID_CD_INTERNACAO"] for dp in death_proximity_list]
    ae_map = {}
    if open_iids:
        try:
            id_list = ", ".join(str(x) for x in open_iids[:500])
            ae_cols, ae_rows = _exec(con, f"""
                SELECT e.ID_CD_INTERNACAO, COUNT(*) as n_ae,
                       STRING_AGG(DISTINCT t.DS_TITULO, ', ') as ae_types
                FROM agg_tb_capta_eventos_adversos_caed e
                LEFT JOIN agg_tb_capta_cfg_tipos_eventos_adversos_ctea t
                    ON e.ID_CD_TIPO_EVENTO = t.ID_CD_TIPO_EVENTO AND e.source_db = t.source_db
                WHERE e.{SRC} AND e.ID_CD_INTERNACAO IN ({id_list})
                GROUP BY e.ID_CD_INTERNACAO
            """)
            ae_map = {r[0]: {"n_ae": int(r[1]), "ae_types": str(r[2] or "")} for r in ae_rows}
        except Exception:
            pass

    # Get n_admissions per patient (for repeat admission flag)
    patient_adm_count = {}
    for pid, adms in patient_admissions.items():
        patient_adm_count[pid] = len(adms)

    # Compute percentiles for death proximity
    all_brad_vecs = emb[np.array([adm_map[iid]["emb_idx"] for iid in adm_map])]
    all_brad_norms = np.linalg.norm(all_brad_vecs, axis=1, keepdims=True).clip(min=1e-9)
    all_brad_cos = (all_brad_vecs / all_brad_norms) @ death_centroid_norm
    cos_p90 = float(np.percentile(all_brad_cos, 90))
    cos_p95 = float(np.percentile(all_brad_cos, 95))
    cos_p99 = float(np.percentile(all_brad_cos, 99))

    # Build enriched list for ALL open cases
    priority_cases = []
    for dp in death_proximity_list:
        iid = dp["ID_CD_INTERNACAO"]
        pid = dp.get("ID_CD_PACIENTE")
        hid = dp.get("ID_CD_HOSPITAL")
        los = _safe_int(dp.get("los"))
        cos_d = _safe_float(dp.get("death_proximity"))
        cid = str(dp.get("cids", "---"))

        # Compute composite risk score
        # Components: death proximity (0-1), LOS excess (0-1), trajectory (0-1)
        score_death = min(cos_d / 0.5, 1.0) if cos_d > 0 else 0  # normalize
        score_los = min(los / 60, 1.0)  # 60+ days = max

        # Trajectory component
        traj = traj_lookup.get(pid)
        score_traj = 0
        if traj:
            score_traj = min(traj.get("risk_score", 0) / 0.05, 1.0)

        # AE component
        ae_info = ae_map.get(iid, {})
        score_ae = min(ae_info.get("n_ae", 0) / 3, 1.0)

        # Composite (weighted)
        composite = 0.35 * score_death + 0.30 * score_los + 0.20 * score_traj + 0.15 * score_ae

        # Generate REASONS in Portuguese
        reasons = []

        # Death proximity reason
        if cos_d >= cos_p99:
            reasons.append(f"Proximidade ao perfil de obito no P99 ({cos_d:.3f})")
        elif cos_d >= cos_p95:
            reasons.append(f"Proximidade ao perfil de obito no P95 ({cos_d:.3f})")
        elif cos_d >= cos_p90:
            reasons.append(f"Proximidade ao perfil de obito acima do P90 ({cos_d:.3f})")

        # LOS reason
        cid_med = cid_medians.get(cid.split(" | ")[0] if " | " in cid else cid, 0)
        if cid_med > 0 and los > 2 * cid_med:
            ratio = los / cid_med
            reasons.append(f"Permanencia de {los}d e {ratio:.1f}x a mediana de {cid_med:.0f}d para este CID")
        elif los >= 60:
            reasons.append(f"Internacao prolongada: {los} dias sem alta")
        elif los >= 30:
            reasons.append(f"Permanencia elevada: {los} dias")

        # Trajectory reason
        if traj and traj.get("direction_to_death", 0) > 0.2:
            n_adm = patient_adm_count.get(pid, 1)
            reasons.append(f"Trajetoria em direcao ao cluster de obito (+{traj['direction_to_death']:.2f}), {n_adm}a internacao")

        # AE reason
        if ae_info.get("n_ae", 0) > 0:
            reasons.append(f"{ae_info['n_ae']} evento(s) adverso(s): {ae_info.get('ae_types', '?')}")

        # If no specific reasons but high composite
        if not reasons and composite > 0.3:
            reasons.append(f"Score composto elevado ({composite:.2f}) sem fator dominante isolado")

        if not reasons:
            reasons.append("Monitoramento preventivo")

        priority_cases.append({
            "ID_CD_INTERNACAO": iid,
            "ID_CD_PACIENTE": pid,
            "ID_CD_HOSPITAL": hid,
            "los": los,
            "cid": cid,
            "death_proximity": cos_d,
            "composite_score": composite,
            "reasons": reasons,
            "n_admissions": patient_adm_count.get(pid, 1),
        })

    # Sort by composite score
    priority_cases.sort(key=lambda x: -x["composite_score"])

    # Group by hospital
    hospital_priorities = {}
    for pc in priority_cases:
        hid = pc["ID_CD_HOSPITAL"]
        if hid not in hospital_priorities:
            hospital_priorities[hid] = []
        hospital_priorities[hid].append(pc)

    # Sort hospitals by max composite in their cases
    sorted_hospitals = sorted(
        hospital_priorities.items(),
        key=lambda x: -max(c["composite_score"] for c in x[1])
    )

    twin_result["hospital_priorities"] = sorted_hospitals
    twin_result["n_priority_cases"] = len(priority_cases)
    print(f"    {len(priority_cases)} cases in {len(hospital_priorities)} hospitals prioritized")

    # ---- 3f: HOSPITAL RISK LANDSCAPE (cos_death vs cos_alta per hospital) ----
    hospital_risk_landscape = []
    if alta_centroid_norm is not None and outcome_vector_norm is not None:
        print("    Building hospital risk landscape ...")
        # Get Feb admissions per hospital for sizing
        hosp_feb_counts = {}
        for iid_hrl, a_hrl in adm_map.items():
            adm_dt = str(a_hrl.get("DH_ADMISSAO_HOSP", ""))[:10]
            fin_dt = str(a_hrl.get("DH_FINALIZACAO", ""))[:10] if a_hrl.get("DH_FINALIZACAO") else None
            if adm_dt and adm_dt < "2026-03-01":
                if fin_dt is None or fin_dt >= "2026-02-01":
                    hid_hrl = a_hrl.get("ID_CD_HOSPITAL")
                    if hid_hrl is not None:
                        hosp_feb_counts[hid_hrl] = hosp_feb_counts.get(hid_hrl, 0) + 1

        for hid_rl in set(a.get("ID_CD_HOSPITAL") for a in adm_map.values() if a.get("ID_CD_HOSPITAL")):
            h_key = f"{SOURCE_DB}/ID_CD_HOSPITAL_{hid_rl}"
            h_idx = node_to_idx.get(h_key)
            if h_idx is None:
                continue
            h_vec = emb[h_idx]
            h_vec_n = h_vec / (np.linalg.norm(h_vec) + 1e-9)
            h_cos_d = float(np.dot(h_vec_n, death_centroid_norm))
            h_cos_a = float(np.dot(h_vec_n, alta_centroid_norm))
            h_proj = float(np.dot(h_vec_n, outcome_vector_norm))
            hospital_risk_landscape.append({
                "hospital_id": hid_rl,
                "cos_death": h_cos_d,
                "cos_alta": h_cos_a,
                "proj_outcome": h_proj,
                "n_feb_admissions": hosp_feb_counts.get(hid_rl, 0),
            })
        hospital_risk_landscape.sort(key=lambda x: -x["proj_outcome"])
        print(f"    Hospital risk landscape: {len(hospital_risk_landscape)} hospitals scored")
    twin_result["hospital_risk_landscape"] = hospital_risk_landscape

    # ---- 3g: CHRONIC PATIENT TRAJECTORIES (5+ admissions, slope of death proximity) ----
    chronic_trajectories = []
    if len(obito_indices) >= 5:
        print("    Building chronic patient trajectories ...")
        # Compute P75 of death proximity across all admissions
        _chron_cos_vals = all_brad_cos.flatten()
        chron_p75 = float(np.percentile(_chron_cos_vals, 75))

        n_chronic_candidates = 0
        for pid_ct, adms_ct in patient_admissions.items():
            if len(adms_ct) < 5 or len(adms_ct) > 30:
                continue
            n_chronic_candidates += 1
            if n_chronic_candidates > 1000:
                break
            # Sort chronologically
            adms_sorted = sorted(adms_ct, key=lambda x: str(x.get("DH_ADMISSAO_HOSP", "")))
            # Compute death proximity for each admission
            dp_series = []
            for a_ct in adms_sorted:
                idx_ct = a_ct["emb_idx"]
                v_ct = emb[idx_ct]
                v_ct_n = v_ct / (np.linalg.norm(v_ct) + 1e-9)
                dp_series.append(float(np.dot(v_ct_n, death_centroid_norm)))

            # Fit linear regression: slope of death_proximity over admission index
            x_seq = np.arange(len(dp_series), dtype=np.float64)
            coeffs = np.polyfit(x_seq, dp_series, 1)
            slope_ct = float(coeffs[0])

            last_adm = adms_sorted[-1]
            last_cos_d = dp_series[-1]
            first_cos_d = dp_series[0]

            # Get CID for last admission
            last_cid_ct = "---"
            try:
                _, r_cid = _exec(con, f"""
                    SELECT STRING_AGG(DISTINCT DS_DESCRICAO, ' | ')
                    FROM agg_tb_capta_cid_caci
                    WHERE {SRC} AND ID_CD_INTERNACAO = {last_adm['ID_CD_INTERNACAO']}
                      AND DS_DESCRICAO IS NOT NULL AND IN_PRINCIPAL = 'S'
                """)
                if r_cid and r_cid[0][0]:
                    last_cid_ct = str(r_cid[0][0])
            except Exception:
                pass

            is_open_ct = last_adm.get("IN_SITUACAO") != 2 or last_adm.get("DH_FINALIZACAO") is None

            chronic_trajectories.append({
                "ID_CD_PACIENTE": pid_ct,
                "n_admissions": len(adms_sorted),
                "first_cos_death": first_cos_d,
                "last_cos_death": last_cos_d,
                "slope": slope_ct,
                "last_iid": last_adm["ID_CD_INTERNACAO"],
                "last_hospital_id": last_adm.get("ID_CD_HOSPITAL"),
                "last_los": _safe_int(last_adm.get("los")),
                "last_cid": last_cid_ct,
                "is_open": is_open_ct,
                "dp_series": dp_series,  # full series for chart
            })

        # Sort by slope descending (fastest deterioration)
        chronic_trajectories.sort(key=lambda x: -x["slope"])
        # Flag deteriorating: slope > 0 AND last admission > P75
        for ct in chronic_trajectories:
            ct["is_deteriorating"] = ct["slope"] > 0 and ct["last_cos_death"] > chron_p75
        chronic_trajectories = chronic_trajectories[:30]
        twin_result["chronic_p75"] = chron_p75
        print(f"    Chronic trajectories: {len(chronic_trajectories)} patients (top 30 by slope)")
    twin_result["chronic_trajectories"] = chronic_trajectories

    # ---- 3h: MEDICATION EMBEDDING LANDSCAPE ----
    medication_embeddings = []
    if alta_centroid_norm is not None and outcome_vector_norm is not None:
        print("    Building medication embedding landscape ...")
        # Map product_id -> medication name
        try:
            _, rows_med_map = _exec(con, f"""
                SELECT ID_CD_PRODUTO, NM_TITULO_COMERCIAL
                FROM agg_tb_capta_produtos_capr
                WHERE {SRC} AND NM_TITULO_COMERCIAL IS NOT NULL
            """)
            prod_to_med = {}
            med_products = {}  # med_name -> list of product_ids
            for r_pm in rows_med_map:
                prod_id = r_pm[0]
                med_nm = str(r_pm[1]).strip()
                prod_to_med[prod_id] = med_nm
                if med_nm not in med_products:
                    med_products[med_nm] = set()
                med_products[med_nm].add(prod_id)

            for med_name_ml, prod_ids_ml in med_products.items():
                if len(prod_ids_ml) < 10:
                    continue
                # Find product embeddings
                prod_emb_indices = []
                for pid_ml in prod_ids_ml:
                    p_key = f"{SOURCE_DB}/ID_CD_PRODUTO_{pid_ml}"
                    p_idx = node_to_idx.get(p_key)
                    if p_idx is not None:
                        prod_emb_indices.append(p_idx)
                if len(prod_emb_indices) < 3:
                    continue
                # Compute medication centroid
                med_vecs = emb[prod_emb_indices]
                med_centroid = med_vecs.mean(axis=0)
                med_centroid_n = med_centroid / (np.linalg.norm(med_centroid) + 1e-9)
                m_cos_d = float(np.dot(med_centroid_n, death_centroid_norm))
                m_cos_a = float(np.dot(med_centroid_n, alta_centroid_norm))
                m_proj = float(np.dot(med_centroid_n, outcome_vector_norm))
                medication_embeddings.append({
                    "medicamento": med_name_ml,
                    "n_prescricoes": len(prod_ids_ml),
                    "n_embedded": len(prod_emb_indices),
                    "cos_death": m_cos_d,
                    "cos_alta": m_cos_a,
                    "proj_outcome": m_proj,
                })

            medication_embeddings.sort(key=lambda x: -x["cos_death"])
            print(f"    Medication embeddings: {len(medication_embeddings)} medications scored")
        except Exception as ex_med:
            print(f"    Medication embedding landscape FAILED: {ex_med}")
    twin_result["medication_embeddings"] = medication_embeddings

    print(f"    done in {time.time()-t0:.1f}s")
    return twin_result


# -----------------------------------------------------------------
# Data Quality Score per Hospital (Improvement 3)
# -----------------------------------------------------------------

def _fetch_data_quality(con):
    """Compute data quality score per hospital for Feb-active admissions."""
    import time
    print("[12.1/14] Data quality scores per hospital ...")
    t0 = time.time()

    results = []
    try:
        # Get hospitals with >=10 admissions in Feb
        h_cols, h_rows = _exec(con, f"""
            SELECT i.ID_CD_HOSPITAL, COUNT(DISTINCT i.ID_CD_INTERNACAO) AS n_adm
            FROM agg_tb_capta_internacao_cain i
            WHERE i.{SRC}
              AND i.DH_ADMISSAO_HOSP IS NOT NULL
              AND i.DH_ADMISSAO_HOSP::DATE < '{FEB_END}'::DATE + INTERVAL '1 day'
              AND (i.DH_FINALIZACAO IS NULL OR i.DH_FINALIZACAO::DATE >= '{FEB_START}'::DATE)
            GROUP BY i.ID_CD_HOSPITAL
            HAVING COUNT(DISTINCT i.ID_CD_INTERNACAO) >= 10
        """)
        hospital_list = [(r[0], int(r[1])) for r in h_rows]

        for hid_q, n_adm_q in hospital_list:
            # 1. pct_alta_sem_tipo: discharges in Feb where tipo is "SEM TIPO DE ALTA DEFINIDO"
            try:
                _, r1 = _exec(con, f"""
                    WITH feb_dis AS (
                        SELECT i.ID_CD_INTERNACAO
                        FROM agg_tb_capta_internacao_cain i
                        WHERE i.{SRC} AND i.ID_CD_HOSPITAL = {hid_q}
                          AND i.IN_SITUACAO = 2
                          AND i.DH_FINALIZACAO >= '{FEB_START}'::DATE
                          AND i.DH_FINALIZACAO < '{FEB_END}'::DATE + INTERVAL '1 day'
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
                            'Manter monitoramento', 'Instável', 'Provável Alta Complexa'
                        ) OR tipo IS NULL) AS sem_tipo
                    FROM typed
                """)
                total_dis = _safe_int(r1[0][0])
                sem_tipo = _safe_int(r1[0][1])
                pct_alta_sem_tipo = 100.0 * sem_tipo / total_dis if total_dis > 0 else 0.0
            except Exception:
                pct_alta_sem_tipo = 0.0

            # 2. pct_cid_sem_principal: % of Feb-active admissions without IN_PRINCIPAL='S' CID
            try:
                _, r2 = _exec(con, f"""
                    WITH feb_adm AS (
                        SELECT i.ID_CD_INTERNACAO
                        FROM agg_tb_capta_internacao_cain i
                        WHERE i.{SRC} AND i.ID_CD_HOSPITAL = {hid_q}
                          AND i.DH_ADMISSAO_HOSP::DATE < '{FEB_END}'::DATE + INTERVAL '1 day'
                          AND (i.DH_FINALIZACAO IS NULL OR i.DH_FINALIZACAO::DATE >= '{FEB_START}'::DATE)
                    ),
                    has_principal AS (
                        SELECT DISTINCT c.ID_CD_INTERNACAO
                        FROM agg_tb_capta_cid_caci c
                        WHERE c.{SRC} AND c.IN_PRINCIPAL = 'S'
                          AND c.ID_CD_INTERNACAO IN (SELECT ID_CD_INTERNACAO FROM feb_adm)
                    )
                    SELECT
                        (SELECT COUNT(*) FROM feb_adm) AS total,
                        (SELECT COUNT(*) FROM feb_adm WHERE ID_CD_INTERNACAO NOT IN (SELECT ID_CD_INTERNACAO FROM has_principal)) AS sem_principal
                """)
                total_adm_q = _safe_int(r2[0][0])
                sem_princ = _safe_int(r2[0][1])
                pct_cid_sem_principal = 100.0 * sem_princ / total_adm_q if total_adm_q > 0 else 0.0
            except Exception:
                pct_cid_sem_principal = 0.0

            # 3. pct_crm_invalido: % of doctor-admission links with DS_CONSELHO_CLASSE LIKE '0000%' or NULL
            try:
                _, r3 = _exec(con, f"""
                    WITH feb_adm AS (
                        SELECT i.ID_CD_INTERNACAO
                        FROM agg_tb_capta_internacao_cain i
                        WHERE i.{SRC} AND i.ID_CD_HOSPITAL = {hid_q}
                          AND i.DH_ADMISSAO_HOSP::DATE < '{FEB_END}'::DATE + INTERVAL '1 day'
                          AND (i.DH_FINALIZACAO IS NULL OR i.DH_FINALIZACAO::DATE >= '{FEB_START}'::DATE)
                    )
                    SELECT
                        COUNT(*) AS total_links,
                        COUNT(*) FILTER (WHERE m.DS_CONSELHO_CLASSE IS NULL
                            OR m.DS_CONSELHO_CLASSE LIKE '0000%%') AS invalid_crm
                    FROM agg_tb_capta_internacao_medico_hospital_cimh mh
                    JOIN agg_tb_capta_cfg_medico_hospital_ccmh m
                        ON mh.ID_CD_MEDICO_HOSPITAL = m.ID_CD_MEDICO_HOSPITAL
                        AND mh.source_db = m.source_db
                    WHERE mh.{SRC}
                      AND mh.ID_CD_INTERNACAO IN (SELECT ID_CD_INTERNACAO FROM feb_adm)
                """)
                total_links = _safe_int(r3[0][0])
                invalid_crm = _safe_int(r3[0][1])
                pct_crm_invalido = 100.0 * invalid_crm / total_links if total_links > 0 else 0.0
            except Exception:
                pct_crm_invalido = 0.0

            quality_score = 100.0 - (pct_alta_sem_tipo + pct_cid_sem_principal + pct_crm_invalido) / 3.0

            results.append({
                "hospital_id": hid_q,
                "n_admissions": n_adm_q,
                "pct_alta_sem_tipo": pct_alta_sem_tipo,
                "pct_cid_sem_principal": pct_cid_sem_principal,
                "pct_crm_invalido": pct_crm_invalido,
                "quality_score": quality_score,
            })

        results.sort(key=lambda x: x["quality_score"])
        print(f"    Data quality computed for {len(results)} hospitals")
    except Exception as ex:
        print(f"    Data quality FAILED: {ex}")

    print(f"    done in {time.time()-t0:.1f}s")
    return results


# -----------------------------------------------------------------
# LaTeX Generation
# -----------------------------------------------------------------

def _generate_latex(kpis, hospitals, trend, profile, los, risk_matrix,
                    doc_cid_comp, med_intensity, docs, ae, meds,
                    twin, twin_stats, charts=None, data_quality=None):
    L = []

    feb = kpis["feb"]
    jan = kpis["jan"]

    # -- Preamble --
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
\usepackage{graphicx}
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
\definecolor{deathred}{RGB}{140,0,30}

\tcbuselibrary{skins,breakable}

\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\textcolor{bradblue}{\textbf{JCUBE V6.2}} \textcolor{bradgray}{\small | GHO-BRADESCO --- Retrospectiva Fevereiro 2026}}
\fancyhead[R]{\textcolor{bradgray}{\small """ + REPORT_DATE_STR + r"""}}
\fancyfoot[C]{\textcolor{bradgray}{\small Auditoria Operacional --- \thepage}}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0.4pt}

\titleformat{\section}{\Large\bfseries\color{bradblue}}{\thesection}{1em}{}[\titlerule]
\titleformat{\subsection}{\large\bfseries\color{bradblue!80!black}}{\thesubsection}{1em}{}
\titleformat{\subsubsection}{\normalsize\bfseries\color{bradgray}}{\thesubsubsection}{1em}{}

\hypersetup{colorlinks=true,linkcolor=bradblue,
  pdftitle={GHO-BRADESCO Retrospectiva Fevereiro 2026 - JCUBE V6.2}}

\begin{document}
\setlength{\parindent}{0pt}
\setlength{\parskip}{4pt}
""")

    # ================================================================
    # SECTION 1 -- CAPA + RESUMO EXECUTIVO (with MoM deltas)
    # ================================================================
    censo = _safe_int(feb.get("censo"))
    novas = _safe_int(feb.get("novas_admissoes"))
    altas_f = _safe_int(feb.get("altas"))
    avg_los_f = _safe_float(feb.get("avg_los"))
    med_los_f = _safe_float(feb.get("median_los"))
    n_obito_f = _safe_int(feb.get("n_obito"))
    taxa_obito_f = _safe_float(feb.get("taxa_obito"))
    taxa_readmit_f = _safe_float(feb.get("taxa_readmit"))
    n_ae_f = _safe_int(feb.get("eventos_adversos"))
    open_cases = _safe_int(feb.get("open_cases"))
    max_open = _safe_int(feb.get("max_open_days"))
    n_presc_f = _safe_int(feb.get("total_prescricoes"))

    # Jan values for delta
    novas_j = _safe_int(jan.get("novas_admissoes"))
    altas_j = _safe_int(jan.get("altas"))
    avg_los_j = _safe_float(jan.get("avg_los"))
    taxa_obito_j = _safe_float(jan.get("taxa_obito"))
    taxa_readmit_j = _safe_float(jan.get("taxa_readmit"))
    n_ae_j = _safe_int(jan.get("eventos_adversos"))
    n_presc_j = _safe_int(jan.get("total_prescricoes"))

    d_admissoes = _delta_arrow(novas, novas_j, invert=False)
    d_altas = _delta_arrow(altas_f, altas_j, invert=False)
    d_los = _delta_arrow(avg_los_f, avg_los_j, invert=True)
    d_obito = _delta_arrow(taxa_obito_f, taxa_obito_j, invert=True)
    d_readmit = _delta_arrow(taxa_readmit_f, taxa_readmit_j, invert=True)
    d_ae = _delta_arrow(n_ae_f, n_ae_j, invert=True)

    L.append(r"""\begin{titlepage}
\begin{center}
\vspace*{0.8cm}
{\Huge\bfseries\textcolor{bradblue}{JCUBE}}\\[0.1cm]
{\large\textcolor{bradgray}{Digital Twin Analytics --- Dense Temporal JEPA V6.2}}\\[0.6cm]
\begin{tcolorbox}[colback=bradblue,colframe=bradblue,coltext=white,width=0.95\textwidth,halign=center]
{\LARGE\bfseries Retrospectiva Completa --- Fevereiro 2026}\\[0.3cm]
{\large GHO-BRADESCO (Auditoria Operacional)}\\[0.2cm]
{\normalsize 35,2M nos $\times$ 128 dim --- Temporal Graph Network (4$\times$ H100) --- Epoca 3}
\end{tcolorbox}
\vspace{0.5cm}
""")

    L.append(r"{\Large Periodo de analise: \textbf{01/02/2026 a 28/02/2026}}\\[0.2cm]")
    L.append(r"{\large Gerado em: \textbf{29 de marco de 2026}}\\[0.8cm]")

    # Headline findings with MoM deltas — lead with drift crisis
    march_proj_title = twin.get("march_projection", {})
    drift_val_title = _safe_float(march_proj_title.get("drift"))
    drift_str_title = f"{drift_val_title:+.3f}"

    L.append(r"""
\begin{tcolorbox}[colback=bradred!8,colframe=bradred,width=0.88\textwidth,halign=center]
{\large\bfseries\textcolor{bradred}{Achados Principais de Fevereiro (com delta Jan$\rightarrow$Fev)}}\\[4pt]
\begin{itemize}[leftmargin=1.5em,itemsep=2pt]
\item \textbf{ALERTA:} O centroide operacional BRADESCO moveu-se \textbf{""" + drift_str_title + r"""} em direcao ao cluster de obito entre janeiro e fevereiro. Simultaneamente, LOS subiu """ + d_los + r""" e readmissao subiu """ + d_readmit + r""". Quando ambos sobem juntos, a acuidade clinica (gravidade media) da carteira deteriorou.
\item \textbf{""" + _fmt_thousands(censo) + r"""} internacoes ativas | """ + _fmt_thousands(novas) + r""" novas """ + d_admissoes + r""" | """ + _fmt_thousands(altas_f) + r""" altas """ + d_altas + r"""
\item LOS medio: \textbf{""" + f"{avg_los_f:.1f}" + r"""d} """ + d_los + r""" (mediana """ + f"{med_los_f:.0f}" + r"""d)
\item Taxa de obito: \textbf{""" + f"{taxa_obito_f:.1f}" + r"""\%} """ + d_obito + r""" | Readmissao 30d: \textbf{""" + f"{taxa_readmit_f:.1f}" + r"""\%} """ + d_readmit + r"""
\item \textbf{""" + str(n_ae_f) + r"""} eventos adversos """ + d_ae + r""" | \textbf{""" + str(open_cases) + r"""} casos abertos (max """ + str(max_open) + r"""d)
\end{itemize}
{\footnotesize\textcolor{bradgray}{GHO-BRADESCO opera como auditora/operadora --- dados de faturamento hospitalar nao disponiveis.}}
\end{tcolorbox}
\vspace{0.4cm}
""")

    # KPI grid (2x4 boxes with delta arrows)
    L.append(r"""
\begin{tabular}{cccc}
\begin{tcolorbox}[colback=kpibox,colframe=bradblue,width=3.5cm,halign=center,left=2pt,right=2pt]
{\LARGE\bfseries\textcolor{bradblue}{""" + _fmt_thousands(novas) + r"""}}\\[2pt]
{\small Novas Admissoes}\\{\scriptsize """ + d_admissoes + r"""}
\end{tcolorbox} &
\begin{tcolorbox}[colback=kpibox,colframe=bradblue,width=3.5cm,halign=center,left=2pt,right=2pt]
{\LARGE\bfseries\textcolor{bradblue}{""" + _fmt_thousands(altas_f) + r"""}}\\[2pt]
{\small Altas}\\{\scriptsize """ + d_altas + r"""}
\end{tcolorbox} &
\begin{tcolorbox}[colback=kpibox,colframe=bradblue,width=3.5cm,halign=center,left=2pt,right=2pt]
{\LARGE\bfseries\textcolor{bradblue}{""" + f"{avg_los_f:.1f}" + r"""d}}\\[2pt]
{\small LOS Medio}\\{\scriptsize """ + d_los + r"""}
\end{tcolorbox} &
\begin{tcolorbox}[colback=alertbox,colframe=bradred,width=3.5cm,halign=center,left=2pt,right=2pt]
{\LARGE\bfseries\textcolor{bradred}{""" + f"{taxa_obito_f:.1f}\\%" + r"""}}\\[2pt]
{\small Obito}\\{\scriptsize """ + d_obito + r"""}
\end{tcolorbox}
\end{tabular}

\vspace{0.2cm}

\begin{tabular}{cccc}
\begin{tcolorbox}[colback=warnbox,colframe=bradgold,width=3.5cm,halign=center,left=2pt,right=2pt]
{\LARGE\bfseries\textcolor{bradgold}{""" + f"{taxa_readmit_f:.1f}\\%" + r"""}}\\[2pt]
{\small Readmissao 30d}\\{\scriptsize """ + d_readmit + r"""}
\end{tcolorbox} &
\begin{tcolorbox}[colback=alertbox,colframe=bradred,width=3.5cm,halign=center,left=2pt,right=2pt]
{\LARGE\bfseries\textcolor{bradred}{""" + str(n_ae_f) + r"""}}\\[2pt]
{\small Ev. Adversos}\\{\scriptsize """ + d_ae + r"""}
\end{tcolorbox} &
\begin{tcolorbox}[colback=alertbox,colframe=bradred,width=3.5cm,halign=center,left=2pt,right=2pt]
{\LARGE\bfseries\textcolor{bradred}{""" + str(open_cases) + r"""}}\\[2pt]
{\small Casos Abertos}
\end{tcolorbox} &
\begin{tcolorbox}[colback=kpibox,colframe=bradblue,width=3.5cm,halign=center,left=2pt,right=2pt]
{\LARGE\bfseries\textcolor{bradblue}{""" + _fmt_thousands(n_presc_f) + r"""}}\\[2pt]
{\small Prescricoes Med.}
\end{tcolorbox}
\end{tabular}
""")

    L.append(_latex_chart(charts, "kpi_gauges"))

    L.append(r"""
\vfill
{\small\textcolor{bradgray}{Confidencial --- Preparado exclusivamente para Bradesco Saude.\\
Dados: DuckDB (system of records) + JCUBE V6.2 Dense Temporal JEPA (TGN, 128-dim, 4$\times$H100).\\
Periodo: todo o mes de fevereiro 2026 (admissoes ativas, novas, altas, eventos).}}
\end{center}
\end{titlepage}

\tableofcontents
\newpage
""")

    # ================================================================
    # SECTION 2 -- ALERTA PREDITIVO (TWIN-DRIVEN -- FRONT AND CENTER)
    # ================================================================
    L.append(r"\section{Alerta Preditivo: Internacoes Abertas em Risco}" + "\n")

    death_prox = twin.get("death_proximity", [])
    trajectory = twin.get("trajectory_velocity", [])
    march_proj = twin.get("march_projection", {})

    L.append(r"""
\begin{tcolorbox}[enhanced,colback=deathred!5,colframe=deathred,fonttitle=\bfseries,
  title={Gemeo Digital Preditivo --- JCUBE V6.2},left=5pt,right=5pt,top=4pt,bottom=4pt,boxrule=1.5pt]
O Gemeo Digital V6.2 codifica cada internacao como um vetor de 128 dimensoes treinado sobre
35,2M nos em Temporal Graph Network. Para gerar \textbf{alertas preditivos}, computamos:
\begin{enumerate}[leftmargin=1.5em,itemsep=1pt]
\item \textbf{Centroide de obito:} media dos embeddings de TODAS as internacoes que terminaram em obito
\item \textbf{Proximidade ao obito:} similaridade cosseno entre cada caso ABERTO e o centroide de obito
\item \textbf{Velocidade de trajetoria:} para pacientes com 2+ internacoes, o delta vetorial entre internacoes consecutivas em direcao ao cluster de obito
\end{enumerate}
\end{tcolorbox}
\vspace{0.3cm}
""")

    L.append(_latex_chart(charts, "scatter_death"))

    # 2.1: Death proximity ranking
    L.append(r"\subsection{Ranking de Proximidade ao Obito --- Casos Abertos}" + "\n")

    if death_prox:
        L.append(r"""
\begin{tcolorbox}[enhanced,colback=alertbox,colframe=bradred,fonttitle=\bfseries\small,
  title={Top """ + str(min(20, len(death_prox))) + r""" casos abertos com maior similaridade ao centroide de obito},
  left=5pt,right=5pt,top=4pt,bottom=4pt,boxrule=1pt,breakable]
\footnotesize\begin{tabularx}{\textwidth}{r r X r r r}
\toprule \textbf{\#} & \textbf{ID} & \textbf{Hospital} & \textbf{Dias} & \textbf{Prox. Obito} & \textbf{CID Principal} \\ \midrule
""")
        for i, dp in enumerate(death_prox[:20], 1):
            iid = _safe_int(dp.get("ID_CD_INTERNACAO"))
            hosp = _escape_latex(_hosp_name(dp.get("ID_CD_HOSPITAL")))
            days = _safe_int(dp.get("los"))
            cos_s = _safe_float(dp.get("death_proximity"))
            cids = _escape_latex(_truncate(str(dp.get("cids", "---")), 55))
            # Color code by proximity
            if cos_s >= 0.8:
                color = "anomred"
            elif cos_s >= 0.6:
                color = "anomorange"
            else:
                color = "bradgray"
            L.append(f"{i} & {iid} & {hosp} & {days} & \\textcolor{{{color}}}{{\\textbf{{{cos_s:.3f}}}}} & {cids} \\\\\n")
        L.append(r"\bottomrule \end{tabularx}" + "\n")
        L.append(r"\end{tcolorbox}" + "\n\n")
    else:
        L.append("Nenhum caso aberto encontrado no espaco de embeddings.\n\n")

    # 2.2: Trajectory velocity
    L.append(r"\subsection{Velocidade de Trajetoria --- Pacientes com 2+ Internacoes}" + "\n")

    # Show open cases first, then high-risk closed
    traj_display = sorted(trajectory, key=lambda t: (-int(t.get("is_open", False)), -t.get("risk_score", 0)))
    if traj_display:
        L.append(r"""
\begin{tcolorbox}[enhanced,colback=warnbox,colframe=bradgold,fonttitle=\bfseries\small,
  title={Pacientes com maior velocidade em direcao ao cluster de obito},
  left=5pt,right=5pt,top=4pt,bottom=4pt,boxrule=1pt,breakable]
\footnotesize\begin{tabularx}{\textwidth}{r r X r r r l}
\toprule \textbf{\#} & \textbf{Pac.} & \textbf{Hospital} & \textbf{Vel.} & \textbf{Dir. Obito} & \textbf{Risk Score} & \textbf{Status} \\ \midrule
""")
        for i, t in enumerate(traj_display[:15], 1):
            pid = _safe_int(t.get("ID_CD_PACIENTE"))
            hosp = _escape_latex(_hosp_name(t.get("hospital_id")))
            vel = _safe_float(t.get("velocity"))
            dirn = _safe_float(t.get("direction_to_death"))
            risk = _safe_float(t.get("risk_score"))
            status = "ABERTO" if t.get("is_open") else "ALTA"
            if risk > 0.5:
                color = "anomred"
            elif risk > 0.2:
                color = "anomorange"
            else:
                color = "bradgray"
            L.append(f"{i} & {pid} & {hosp} & {vel:.3f} & {dirn:+.3f} & \\textcolor{{{color}}}{{\\textbf{{{risk:.3f}}}}} & {status} \\\\\n")
        L.append(r"\bottomrule \end{tabularx}" + "\n")
        L.append(r"\end{tcolorbox}" + "\n\n")
    else:
        L.append("Dados insuficientes para analise de trajetoria.\n\n")

    L.append(_latex_chart(charts, "heatmap_traj"))

    # ---- 2.x: Chronic Patient Trajectories ----
    chronic_data = twin.get("chronic_trajectories", [])
    if chronic_data:
        L.append(r"\subsection{Pacientes Cronicos: Trajetoria de Deterioracao}" + "\n")
        L.append(r"""
\begin{tcolorbox}[enhanced,colback=alertbox,colframe=bradred,fonttitle=\bfseries\small,
  title={Pacientes com 5+ Internacoes --- Analise de Tendencia},
  left=5pt,right=5pt,top=4pt,bottom=4pt,boxrule=1pt]
Para pacientes com 5 ou mais internacoes, calculamos a \textbf{proximidade ao obito} (cosseno)
em cada admissao e ajustamos uma regressao linear. O \textbf{slope} indica a taxa de
aproximacao ao cluster de obito por internacao. Pacientes com slope positivo e ultima
proximidade acima do P75 sao classificados como \textbf{em deterioracao}.
\end{tcolorbox}
\vspace{0.3cm}
""")
        L.append(_latex_chart(charts, "chronic_traj"))

        # Table: top 15 deteriorating patients
        L.append(r"""
\begin{tcolorbox}[enhanced,colback=bradlightgray,colframe=bradgray,fonttitle=\bfseries\small,
  title={Top 15 Pacientes por Taxa de Deterioracao},
  left=5pt,right=5pt,top=4pt,bottom=4pt,boxrule=0.8pt,breakable]
\footnotesize\begin{tabularx}{\textwidth}{r r r r r r X r l}
\toprule \textbf{\#} & \textbf{Pac.} & \textbf{N Int.} & \textbf{1o Cos} & \textbf{Ult Cos} & \textbf{Slope} & \textbf{Hospital Atual} & \textbf{LOS} & \textbf{Status} \\ \midrule
""")
        for idx_ct, ct in enumerate(chronic_data[:15], 1):
            ct_pid = _safe_int(ct.get("ID_CD_PACIENTE"))
            ct_n = ct.get("n_admissions", 0)
            ct_first = _safe_float(ct.get("first_cos_death"))
            ct_last = _safe_float(ct.get("last_cos_death"))
            ct_slope = _safe_float(ct.get("slope"))
            ct_hosp = _escape_latex(_hosp_name(ct.get("last_hospital_id")))
            ct_los = _safe_int(ct.get("last_los"))
            ct_status = "ABERTO" if ct.get("is_open") else "ALTA"
            ct_color = "anomred" if ct.get("is_deteriorating") else ("anomorange" if ct_slope > 0 else "bradgray")
            L.append(f"{idx_ct} & {ct_pid} & {ct_n} & {ct_first:.3f} & {ct_last:.3f} & \\textcolor{{{ct_color}}}{{\\textbf{{{ct_slope:+.4f}}}}} & {ct_hosp} & {ct_los}d & {ct_status} \\\\\n")
        L.append(r"\bottomrule \end{tabularx}" + "\n")
        L.append(r"\end{tcolorbox}" + "\n\n")

        # CID column note
        n_deteriorating = sum(1 for ct in chronic_data if ct.get("is_deteriorating"))
        L.append(f"\\textcolor{{bradgray}}{{\\footnotesize {n_deteriorating} pacientes classificados como em deterioracao (slope $>$ 0 e ultima proximidade $>$ P75).}}\n\n")

    # 2.3: March projection
    L.append(r"\subsection{Projecao de Marco --- Movimento do Centroide}" + "\n")

    if march_proj:
        jan_cos = _safe_float(march_proj.get("jan_cos_death"))
        feb_cos = _safe_float(march_proj.get("feb_cos_death"))
        drift = _safe_float(march_proj.get("drift"))
        direction = march_proj.get("direction", "---")
        n_jan = _safe_int(march_proj.get("n_jan"))
        n_feb = _safe_int(march_proj.get("n_feb"))

        drift_color = "bradred" if drift > 0 else "bradgreen"
        drift_arrow = "$\\uparrow$" if drift > 0 else "$\\downarrow$"

        L.append(r"""
\begin{tcolorbox}[enhanced,colback=bradlightgray,colframe=bradblue,fonttitle=\bfseries\small,
  title={Drift do centroide operacional vs. cluster de obito},
  left=5pt,right=5pt,top=4pt,bottom=4pt,boxrule=1pt]
\begin{tabularx}{\textwidth}{l X}
\textbf{Centroide Jan (N=""" + str(n_jan) + r"""):} & Cosseno ao obito = """ + f"{jan_cos:.4f}" + r""" \\
\textbf{Centroide Fev (N=""" + str(n_feb) + r"""):} & Cosseno ao obito = """ + f"{feb_cos:.4f}" + r""" \\
\textbf{Drift Jan$\rightarrow$Fev:} & \textcolor{""" + drift_color + r"""}{\textbf{""" + f"{drift:+.4f}" + r""" """ + drift_arrow + r"""}} (""" + _escape_latex(direction) + r""") \\
\end{tabularx}
\end{tcolorbox}
""")
        if drift > 0:
            L.append(r"""
\begin{tcolorbox}[enhanced,colback=alertbox,colframe=bradred,left=5pt,right=5pt,top=4pt,bottom=4pt,boxrule=1pt]
\textbf{Atencao:} O centroide operacional de fevereiro esta se movendo em direcao ao cluster de obito
em comparacao com janeiro. Isso indica que o perfil medio das internacoes esta ficando mais proximo
de casos que resultaram em morte. Se a tendencia se mantiver, marco pode apresentar piora nos indicadores.
\end{tcolorbox}
""")
        else:
            L.append(r"""
\begin{tcolorbox}[enhanced,colback=successbox,colframe=bradgreen,left=5pt,right=5pt,top=4pt,bottom=4pt,boxrule=1pt]
O centroide operacional de fevereiro esta se afastando do cluster de obito em comparacao com janeiro.
Tendencia positiva para marco, indicando perfil clinico medio melhor.
\end{tcolorbox}
""")
    else:
        L.append("Dados insuficientes para projecao de marco.\n")

    # 2.4: Hospital Drift Contribution
    hosp_drift_data = twin.get("hospital_drift", [])
    if hosp_drift_data:
        L.append(r"\subsection{Hospitais que Mais Contribuiram para o Drift}" + "\n")
        L.append(r"""
\begin{tcolorbox}[enhanced,colback=alertbox,colframe=bradred,fonttitle=\bfseries\small,
  title={Top 10 hospitais por drift positivo (movimento em direcao ao cluster de obito)},
  left=5pt,right=5pt,top=4pt,bottom=4pt,boxrule=1pt,breakable]
\footnotesize\begin{tabularx}{\textwidth}{r X r r r r r}
\toprule \textbf{\#} & \textbf{Hospital} & \textbf{N Fev} & \textbf{N Jan} & \textbf{Cos Jan} & \textbf{Cos Fev} & \textbf{Drift} \\ \midrule
""")
        for idx_hd, (hid_hd, hd_info) in enumerate(hosp_drift_data[:10], 1):
            hname_hd = _escape_latex(_hosp_name(hid_hd))
            n_feb_hd = _safe_int(hd_info.get("n_feb"))
            n_jan_hd = _safe_int(hd_info.get("n_jan"))
            jan_cos_hd = _safe_float(hd_info.get("jan_cos"))
            feb_cos_hd = _safe_float(hd_info.get("feb_cos"))
            drift_hd = _safe_float(hd_info.get("drift"))
            drift_color_hd = "anomred" if drift_hd > 0.05 else ("anomorange" if drift_hd > 0 else "bradgreen")
            L.append(f"{idx_hd} & {hname_hd} & {n_feb_hd} & {n_jan_hd} & {jan_cos_hd:.4f} & {feb_cos_hd:.4f} & \\textcolor{{{drift_color_hd}}}{{\\textbf{{{drift_hd:+.4f}}}}} \\\\\n")
        L.append(r"\bottomrule \end{tabularx}" + "\n")
        L.append(r"\end{tcolorbox}" + "\n\n")

    L.append(r"\newpage" + "\n")

    # ================================================================
    # SECTION 2.5 -- LISTA DE PRIORIDADES POR HOSPITAL
    # ================================================================
    hospital_prios = twin.get("hospital_priorities", [])
    if hospital_prios:
        L.append(r"\section{Lista de Prioridades por Hospital}" + "\n")
        L.append(r"""
\begin{tcolorbox}[enhanced,colback=alertbox,colframe=bradred,fonttitle=\bfseries\small,
  title={Acao Imediata: Pacientes para Verificacao por Hospital},
  left=5pt,right=5pt,top=4pt,bottom=4pt,boxrule=1.2pt]
\textbf{Objetivo:} Para cada hospital credenciado, listar os pacientes abertos que devem ser
verificados com prioridade, ordenados por risco composto (proximidade ao perfil de obito +
permanencia excessiva + trajetoria + eventos adversos). Cada paciente tem os \textbf{motivos}
da priorizacao em linguagem clara.

\textbf{Score composto:} 35\% proximidade obito + 30\% excesso de permanencia + 20\% trajetoria + 15\% eventos adversos.
\end{tcolorbox}
\vspace{0.3cm}
""")
        n_hospitals_shown = 0
        for hid, cases in hospital_prios:
            # Only show hospitals with at least 1 case with composite > 0.15
            actionable = [c for c in cases if c["composite_score"] > 0.15]
            if not actionable:
                continue
            n_hospitals_shown += 1
            if n_hospitals_shown > 15:  # limit to top 15 hospitals
                break

            hosp_name = _escape_latex(_hosp_name(hid))
            n_cases = len(cases)
            n_actionable = len(actionable)
            max_score = actionable[0]["composite_score"]

            L.append(f"\\subsection{{{hosp_name} ({n_actionable} pacientes prioritarios)}}\n")
            L.append(r"\begin{tcolorbox}[enhanced,colback=bradlightgray,colframe=bradgray,fonttitle=\bfseries\small,title={")
            L.append(f"{n_actionable} de {n_cases} casos abertos requerem atencao")
            L.append(r"},left=5pt,right=5pt,top=4pt,bottom=4pt,boxrule=0.8pt,breakable]" + "\n")
            L.append(r"\footnotesize" + "\n")

            for idx, c in enumerate(actionable[:8], 1):  # max 8 per hospital
                iid = _safe_int(c.get("ID_CD_INTERNACAO"))
                pid = _safe_int(c.get("ID_CD_PACIENTE"))
                c_los = _safe_int(c.get("los"))
                cid = _escape_latex(_truncate(str(c.get("cid", "---")), 55))
                score = c["composite_score"]
                n_adm = c.get("n_admissions", 1)
                reasons = c.get("reasons", ["---"])

                # Severity color
                if score >= 0.5:
                    sev, color = "CRITICO", "anomred"
                elif score >= 0.3:
                    sev, color = "ALTO", "anomorange"
                else:
                    sev, color = "MODERADO", "bradgold"

                L.append(f"\\textcolor{{{color}}}{{\\textbf{{{idx}. [{sev}] Internacao {iid}}}}} ")
                L.append(f"--- Paciente {pid}")
                if n_adm > 1:
                    L.append(f" ({n_adm}a internacao)")
                L.append(f" --- \\textbf{{{c_los} dias}} sem alta\\\\\n")
                L.append(f"\\textbf{{CID:}} {cid}\\\\\n")
                L.append(f"\\textbf{{Score:}} {score:.2f} | ")
                L.append(f"\\textbf{{Motivos:}}\n")
                L.append(r"\begin{itemize}[leftmargin=2em,itemsep=0pt,topsep=1pt]" + "\n")
                for reason in reasons:
                    L.append(f"\\item {_escape_latex(reason)}\n")
                L.append(r"\end{itemize}" + "\n")
                if idx < len(actionable[:8]):
                    L.append(r"\vspace{0.15cm}\hrule\vspace{0.15cm}" + "\n")

            L.append(r"\end{tcolorbox}" + "\n\n")

        L.append(r"\newpage" + "\n")

    # ================================================================
    # SECTION 3 -- DESEMPENHO POR HOSPITAL (with MoM LOS delta)
    # ================================================================
    L.append(r"\section{Desempenho por Hospital}" + "\n")

    L.append(r"""
\begin{tcolorbox}[enhanced,colback=bradlightgray,colframe=bradgray,fonttitle=\bfseries\small,
  title={Metodologia},left=5pt,right=5pt,top=4pt,bottom=4pt,boxrule=0.8pt]
Ranking baseado em altas de fevereiro (IN\_SITUACAO=2). Coluna $\Delta$LOS mostra a variacao
percentual do LOS medio de janeiro para fevereiro por hospital.
\end{tcolorbox}
\vspace{0.3cm}
""")

    L.append(_latex_chart(charts, "hosp_los_bars"))

    if hospitals:
        L.append(r"""
\begin{tcolorbox}[enhanced,colback=bradlightgray,colframe=bradgray,fonttitle=\bfseries\small,
  title={Ranking de Hospitais --- Altas de Fevereiro 2026},left=5pt,right=5pt,top=4pt,bottom=4pt,boxrule=0.8pt,breakable]
\footnotesize\begin{tabularx}{\textwidth}{r X r r r r r r}
\toprule \textbf{\#} & \textbf{Hospital} & \textbf{Altas} & \textbf{LOS Med} & \textbf{$\Delta$LOS} & \textbf{Obito\%} & \textbf{EA} & \textbf{Readm.} \\ \midrule
""")
        for i, h in enumerate(hospitals[:20], 1):
            hosp = _escape_latex(_hosp_name(h.get("hospital_id")))
            n = _safe_int(h.get("n_altas"))
            avg = _safe_float(h.get("avg_los"))
            tx = _safe_float(h.get("taxa_obito"))
            nae = _safe_int(h.get("n_ae"))
            readm = _safe_int(h.get("readmit_30d"))
            delta_los = h.get("los_delta_pct")
            if delta_los is not None:
                if delta_los > 5:
                    d_str = f"\\textcolor{{bradred}}{{$\\uparrow${abs(delta_los):.0f}\\%}}"
                elif delta_los < -5:
                    d_str = f"\\textcolor{{bradgreen}}{{$\\downarrow${abs(delta_los):.0f}\\%}}"
                else:
                    d_str = f"$\\rightarrow${abs(delta_los):.0f}\\%"
            else:
                d_str = "---"
            L.append(f"{i} & {hosp} & {n} & {avg:.1f}d & {d_str} & {tx:.1f}\\% & {nae} & {readm} \\\\\n")
        L.append(r"\bottomrule \end{tabularx}" + "\n")
        L.append(r"\end{tcolorbox}" + "\n")

    L.append(_latex_chart(charts, "hosp_bubble"))

    # ---- 3.x: Hospital Risk Radar (embedding landscape) ----
    hosp_risk_data = twin.get("hospital_risk_landscape", [])
    if hosp_risk_data:
        L.append(r"\subsection{Radar de Risco Hospitalar (Embedding Landscape)}" + "\n")
        L.append(r"""
\begin{tcolorbox}[enhanced,colback=warnbox,colframe=bradgold,fonttitle=\bfseries\small,
  title={Alinhamento Hospitalar com Desfecho via Embeddings},
  left=5pt,right=5pt,top=4pt,bottom=4pt,boxrule=1pt]
Cada hospital possui um embedding de 128 dimensoes na Temporal Graph Network. Comparamos
cada vetor hospitalar com dois polos: o \textbf{centroide de obito} (media dos embeddings de
internacoes que terminaram em morte) e o \textbf{centroide de alta melhorada} (media dos
embeddings de internacoes com desfecho positivo). A projecao no eixo obito$\rightarrow$alta indica o
alinhamento relativo.

\textbf{COMO LER:} Hospitais acima da diagonal tem perfil de embedding mais alinhado com obitos.
Hospitais abaixo tem perfil mais alinhado com altas bem-sucedidas. Tamanho = volume de internacoes.
\end{tcolorbox}
\vspace{0.3cm}
""")
        L.append(_latex_chart(charts, "hosp_risk_radar"))

        # Table: top hospitals sorted by proj_outcome (most death-aligned first)
        hosp_risk_sorted = sorted(hosp_risk_data, key=lambda x: -x["proj_outcome"])
        L.append(r"""
\begin{tcolorbox}[enhanced,colback=bradlightgray,colframe=bradgray,fonttitle=\bfseries\small,
  title={Hospitais por Alinhamento com Desfecho (Projecao no Eixo Obito-Alta)},
  left=5pt,right=5pt,top=4pt,bottom=4pt,boxrule=0.8pt,breakable]
\footnotesize\begin{tabularx}{\textwidth}{r X r r r r}
\toprule \textbf{\#} & \textbf{Hospital} & \textbf{N Fev} & \textbf{Cos Obito} & \textbf{Cos Alta} & \textbf{Projecao} \\ \midrule
""")
        for idx_hr, hr in enumerate(hosp_risk_sorted[:20], 1):
            hr_name = _escape_latex(_hosp_name(hr["hospital_id"]))
            hr_n = _safe_int(hr.get("n_feb_admissions"))
            hr_cd = _safe_float(hr.get("cos_death"))
            hr_ca = _safe_float(hr.get("cos_alta"))
            hr_proj = _safe_float(hr.get("proj_outcome"))
            hr_color = "anomred" if hr_proj > 0.05 else ("anomorange" if hr_proj > 0 else "bradgreen")
            L.append(f"{idx_hr} & {hr_name} & {hr_n} & {hr_cd:.4f} & {hr_ca:.4f} & \\textcolor{{{hr_color}}}{{\\textbf{{{hr_proj:+.4f}}}}} \\\\\n")
        L.append(r"\bottomrule \end{tabularx}" + "\n")
        L.append(r"\end{tcolorbox}" + "\n\n")

    L.append(r"\newpage" + "\n")

    # ================================================================
    # SECTION 4 -- MATRIZ DE RISCO CID x HOSPITAL
    # ================================================================
    L.append(r"\section{Matriz de Risco CID $\times$ Hospital}" + "\n")

    L.append(r"""
\begin{tcolorbox}[enhanced,colback=warnbox,colframe=bradgold,fonttitle=\bfseries\small,
  title={Analise Cross-Dimensional},left=5pt,right=5pt,top=4pt,bottom=4pt,boxrule=1pt]
Esta secao cruza diagnosticos, hospitais e medicos para identificar onde o desempenho
esta abaixo do esperado. Como BRADESCO nao tem dados de faturamento, o proxy de custo e:
\textbf{LOS excedente + intensidade de medicamentos + eventos adversos + mortalidade}.
\end{tcolorbox}
\vspace{0.3cm}
""")

    L.append(_latex_chart(charts, "risk_heatmap"))

    # 4.1: Risk matrix — vertical format (readable!)
    # Show only CID×Hospital pairs where LOS > 1.5× CID median (the actual alerts)
    rm_cells = risk_matrix.get("cells", [])
    rm_medians = risk_matrix.get("cid_medians", {})

    if rm_cells:
        # Filter to anomalous cells only and sort by excess days
        alerts = []
        for c in rm_cells:
            cd = c.get("cid_desc", "")
            med = _safe_float(rm_medians.get(cd))
            avg = _safe_float(c.get("avg_los"))
            n = _safe_int(c.get("n"))
            if med > 0 and avg > 1.5 * med and n >= 3:
                excess = avg - med
                alerts.append({**c, "med_cid": med, "excess": excess, "ratio": avg / med})
        alerts.sort(key=lambda x: -x["excess"] * _safe_int(x.get("n")))

        if alerts:
            L.append(r"\subsection{Alertas: Hospitais com LOS $>$1.5x a Mediana do CID}" + "\n")
            L.append(r"""
\begin{tcolorbox}[enhanced,colback=alertbox,colframe=bradred,fonttitle=\bfseries\small,
  title={Combinacoes CID $\times$ Hospital onde o LOS excede significativamente a referencia},
  left=5pt,right=5pt,top=4pt,bottom=4pt,boxrule=1pt,breakable]
Cada linha abaixo representa uma combinacao diagnostico + hospital onde a permanencia media
esta \textbf{acima de 1.5x} a mediana BRADESCO para esse diagnostico. Quanto maior o excesso $\times$ volume,
maior o impacto operacional.
\end{tcolorbox}
\vspace{0.2cm}

\begin{tcolorbox}[enhanced,colback=bradlightgray,colframe=bradgray,fonttitle=\bfseries\small,
  title={""" + str(len(alerts)) + r""" alertas de permanencia excessiva (ordenados por impacto)},
  left=5pt,right=5pt,top=4pt,bottom=4pt,boxrule=0.8pt,breakable]
\textbf{Objetivo:} Identificar combinacoes diagnostico + hospital onde a permanencia e significativamente
maior que a \textbf{mediana BRADESCO para o mesmo diagnostico (todos os hospitais)}. O ``Excesso'' mostra
quantos dias a mais o hospital mantem o paciente comparado a essa referencia.
\vspace{0.2cm}

\footnotesize\begin{tabularx}{\textwidth}{r X X r r r r}
\toprule \textbf{\#} & \textbf{Diagnostico (CID Principal)} & \textbf{Hospital} & \textbf{N} & \textbf{LOS Hosp.} & \textbf{Mediana BRAD} & \textbf{Excesso} \\ \midrule
""")
            for i, a in enumerate(alerts[:20], 1):
                cid = _escape_latex(_truncate(str(a.get("cid_desc", "---")), 55))
                hosp = _escape_latex(_truncate(_hosp_name(a.get("hospital_id")), 30))
                n = _safe_int(a.get("n"))
                avg = _safe_float(a.get("avg_los"))
                med = _safe_float(a.get("med_cid"))
                excess = _safe_float(a.get("excess"))
                ratio = _safe_float(a.get("ratio"))
                color = "anomred" if ratio >= 2.0 else "anomorange" if ratio >= 1.5 else "bradgray"
                L.append(f"{i} & {cid} & {hosp} & {n} & {avg:.0f}d & {med:.0f}d & \\textcolor{{{color}}}{{\\textbf{{+{excess:.0f}d ({ratio:.1f}x)}}}} \\\\\n")
            L.append(r"\bottomrule \end{tabularx}" + "\n")
            L.append(r"\end{tcolorbox}" + "\n\n")

    # 4.2: Doctor x CID comparison
    if doc_cid_comp:
        L.append(r"\subsection{Efeito Medico: Mesmo Diagnostico, Permanencias Diferentes}" + "\n")
        L.append(r"""
\begin{tcolorbox}[enhanced,colback=bradlightgray,colframe=bradgray,fonttitle=\bfseries\small,
  title={Medicos com $\geq$5 casos do MESMO CID --- LOS vs mediana BRADESCO},left=5pt,right=5pt,top=4pt,bottom=4pt,boxrule=0.8pt,breakable]
\textbf{Objetivo:} Para o MESMO diagnostico, comparar a permanencia media de cada medico com a
\textbf{mediana BRADESCO (todos hospitais/medicos)} para esse CID. ``Excesso'' positivo = medico
mantem pacientes mais tempo que o habitual; negativo = alta mais rapida.
\vspace{0.2cm}

\footnotesize\begin{tabularx}{\textwidth}{r l X r r r r}
\toprule \textbf{\#} & \textbf{CRM} & \textbf{Diagnostico (CID)} & \textbf{N} & \textbf{LOS Medico} & \textbf{Mediana BRAD} & \textbf{Excesso} \\ \midrule
""")
        for i, d in enumerate(doc_cid_comp[:15], 1):
            crm = _escape_latex(str(d.get("crm", "---"))[:10])
            cid = _escape_latex(_truncate(str(d.get("cid_desc", "---")), 50))
            n = _safe_int(d.get("n_cases"))
            doc_los = _safe_float(d.get("doc_avg_los"))
            cid_med = _safe_float(d.get("cid_median_los"))
            excess = _safe_float(d.get("los_excess"))
            if excess > 0:
                exc_str = f"\\textcolor{{bradred}}{{+{excess:.1f}d}}"
            else:
                exc_str = f"\\textcolor{{bradgreen}}{{{excess:.1f}d}}"
            L.append(f"{i} & {crm} & {cid} & {n} & {doc_los:.1f}d & {cid_med:.0f}d & {exc_str} \\\\\n")
        L.append(r"\bottomrule \end{tabularx}" + "\n")
        L.append(r"\end{tcolorbox}" + "\n\n")

    # 4.3: Medication intensity anomalies
    if med_intensity:
        L.append(r"\subsection{Intensidade de Medicamentos: Quais Hospitais Prescrevem Mais?}" + "\n")
        L.append(r"""
\begin{tcolorbox}[enhanced,colback=warnbox,colframe=bradgold,fonttitle=\bfseries\small,
  title={Hospitais que prescrevem $>$1.5x a media BRADESCO de medicamentos para o mesmo CID},left=5pt,right=5pt,top=4pt,bottom=4pt,boxrule=1pt,breakable]
\textbf{Objetivo:} Para o MESMO diagnostico, identificar hospitais que prescrevem significativamente
mais medicamentos por internacao do que a media BRADESCO. ``Ratio'' $>$ 1.5x indica excesso.
\vspace{0.2cm}

\footnotesize\begin{tabularx}{\textwidth}{r X l r r r}
\toprule \textbf{\#} & \textbf{Diagnostico (CID)} & \textbf{Hospital} & \textbf{N} & \textbf{Meds/Inter.} & \textbf{Ratio} \\ \midrule
""")
        for i, m in enumerate(med_intensity[:12], 1):
            cid = _escape_latex(_truncate(str(m.get("cid_desc", "---")), 50))
            hosp = _escape_latex(_truncate(_hosp_name(m.get("hospital_id")), 25))
            n = _safe_int(m.get("n"))
            h_avg = _safe_float(m.get("hosp_avg_meds"))
            ratio = _safe_float(m.get("ratio"))
            L.append(f"{i} & {cid} & {hosp} & {n} & {h_avg:.1f} & \\textcolor{{bradred}}{{\\textbf{{{ratio:.1f}x}}}} \\\\\n")
        L.append(r"\bottomrule \end{tabularx}" + "\n")
        L.append(r"\end{tcolorbox}" + "\n")

    L.append(r"\newpage" + "\n")

    # ================================================================
    # SECTION 5 -- TENDENCIA 6 MESES (ALL KPIs)
    # ================================================================
    L.append(r"\section{Tendencia 6 Meses (Set/2025 $\rightarrow$ Fev/2026)}" + "\n")

    L.append(_latex_chart(charts, "trend_dual"))
    L.append(_latex_chart(charts, "sparklines"))

    if trend:
        L.append(r"""
\begin{tcolorbox}[enhanced,colback=bradlightgray,colframe=bradgray,fonttitle=\bfseries\small,
  title={Evolucao mensal de TODOS os indicadores},left=3pt,right=3pt,top=3pt,bottom=3pt,boxrule=0.8pt,breakable]
\scriptsize\begin{tabularx}{\textwidth}{l r r r r r r r r}
\toprule \textbf{Mes} & \textbf{Adm.} & \textbf{Altas} & \textbf{LOS Med} & \textbf{LOS Mdn} & \textbf{Obito\%} & \textbf{EA} & \textbf{Readm.\%} & \textbf{Prescr.} \\ \midrule
""")
        for t in trend:
            lab = _escape_latex(t.get("label", ""))
            adm = _fmt_thousands(_safe_int(t.get("admissoes")))
            alt = _fmt_thousands(_safe_int(t.get("altas")))
            avl = _safe_float(t.get("avg_los"))
            mdl = _safe_float(t.get("median_los"))
            tob = _safe_float(t.get("taxa_obito"))
            nae = _safe_int(t.get("n_ae"))
            trm = _safe_float(t.get("taxa_readmit"))
            npr = _fmt_thousands(_safe_int(t.get("n_prescricoes")))
            L.append(f"{lab} & {adm} & {alt} & {avl:.1f}d & {mdl:.0f}d & {tob:.1f}\\% & {nae} & {trm:.1f}\\% & {npr} \\\\\n")
        L.append(r"\bottomrule \end{tabularx}" + "\n")
        L.append(r"\end{tcolorbox}" + "\n")

        # Narrative
        if len(trend) >= 2:
            t_feb = trend[-1]
            t_jan = trend[-2]
            var_adm = 100.0 * (_safe_int(t_feb["admissoes"]) - _safe_int(t_jan["admissoes"])) / max(_safe_int(t_jan["admissoes"]), 1)
            var_str = f"+{var_adm:.1f}" if var_adm >= 0 else f"{var_adm:.1f}"
            var_los = _safe_float(t_feb["avg_los"]) - _safe_float(t_jan["avg_los"])
            var_obito = _safe_float(t_feb["taxa_obito"]) - _safe_float(t_jan["taxa_obito"])
            L.append(r"""
\begin{tcolorbox}[enhanced,colback=kpibox,colframe=bradblue,fonttitle=\bfseries\small,
  title={Destaque Jan $\rightarrow$ Fev},left=5pt,right=5pt,top=4pt,bottom=4pt,boxrule=1pt]
""")
            L.append(f"Variacao de admissoes: \\textbf{{{var_str}\\%}}. ")
            L.append(f"LOS medio: {_safe_float(t_feb['avg_los']):.1f}d ({var_los:+.1f}d). ")
            L.append(f"Taxa de obito: {_safe_float(t_feb['taxa_obito']):.1f}\\% ({var_obito:+.1f}pp).\n")
            L.append(r"\end{tcolorbox}" + "\n")

    L.append(r"\newpage" + "\n")

    # ================================================================
    # SECTION 6 -- PERFIL CLINICO + LOS ANALYSIS
    # ================================================================
    L.append(r"\section{Perfil Clinico e Analise de Permanencia}" + "\n")

    L.append(_latex_chart(charts, "cid_bars"))

    # Top CIDs
    top_cids = profile.get("top_cids_volume", [])
    if top_cids:
        L.append(r"\subsection{Top 15 Diagnosticos Principais por Volume}" + "\n")
        L.append(r"""
\begin{tcolorbox}[enhanced,colback=bradlightgray,colframe=bradgray,fonttitle=\bfseries\small,
  title={CIDs principais (IN\_PRINCIPAL='S') em internacoes ativas em fevereiro},left=5pt,right=5pt,top=4pt,bottom=4pt,boxrule=0.8pt,breakable]
\footnotesize\begin{tabularx}{\textwidth}{r X r r}
\toprule \textbf{\#} & \textbf{Diagnostico Principal} & \textbf{N} & \textbf{LOS Med} \\ \midrule
""")
        for i, c in enumerate(top_cids, 1):
            desc = _escape_latex(_truncate(str(c.get("cid_desc", "---")), 50))
            n = _safe_int(c.get("n_internacoes"))
            al = _safe_float(c.get("avg_los"))
            L.append(f"{i} & {desc} & {n} & {al:.1f}d \\\\\n")
        L.append(r"\bottomrule \end{tabularx}" + "\n")
        L.append(r"\end{tcolorbox}" + "\n\n")

    L.append(r"\begin{minipage}{0.48\textwidth}" + "\n")
    L.append(_latex_chart(charts, "los_dist", width=r"\textwidth"))
    L.append(r"\end{minipage}\hfill\begin{minipage}{0.48\textwidth}" + "\n")
    L.append(_latex_chart(charts, "discharge_donut", width=r"\textwidth"))
    L.append(r"\end{minipage}" + "\n")

    # LOS distribution
    los_dist = profile.get("los_distribution", [])
    if los_dist:
        L.append(r"\subsection{Distribuicao de Permanencia}" + "\n")
        total_l = sum(_safe_int(d.get("n_internacoes")) for d in los_dist)
        L.append(r"""
\begin{tcolorbox}[enhanced,colback=bradlightgray,colframe=bradgray,fonttitle=\bfseries\small,
  title={Histograma de LOS --- altas de fevereiro},left=5pt,right=5pt,top=4pt,bottom=4pt,boxrule=0.8pt]
\begin{tabularx}{\textwidth}{l r r}
\toprule \textbf{Faixa} & \textbf{N} & \textbf{\%} \\ \midrule
""")
        for d in los_dist:
            f_l = _escape_latex(str(d.get("faixa_los", "---")))
            n = _safe_int(d.get("n_internacoes"))
            pct = 100.0 * n / total_l if total_l > 0 else 0
            L.append(f"{f_l} & {n} & {pct:.1f}\\% \\\\\n")
        L.append(r"\bottomrule \end{tabularx}" + "\n")
        L.append(r"\end{tcolorbox}" + "\n\n")

    # Discharge types
    dtypes = profile.get("discharge_types_grouped", [])
    if dtypes:
        L.append(r"\subsection{Tipos de Alta (Agrupados)}" + "\n")
        total_d = sum(_safe_int(d.get("n")) for d in dtypes)
        L.append(r"""
\begin{tcolorbox}[enhanced,colback=bradlightgray,colframe=bradgray,fonttitle=\bfseries\small,
  title={Desfechos agrupados --- altas de fevereiro},left=5pt,right=5pt,top=4pt,bottom=4pt,boxrule=0.8pt]
\begin{tabularx}{\textwidth}{X r r}
\toprule \textbf{Grupo de Desfecho} & \textbf{N} & \textbf{\%} \\ \midrule
""")
        for d in dtypes:
            grp = _escape_latex(str(d.get("grupo", "---")))
            n = _safe_int(d.get("n"))
            pct = 100.0 * n / total_d if total_d > 0 else 0
            L.append(f"{grp} & {n} & {pct:.1f}\\% \\\\\n")
        L.append(r"\bottomrule \end{tabularx}" + "\n")
        L.append(r"\end{tcolorbox}" + "\n\n")

    # LOS by CID
    los_by_cid = los.get("by_cid", [])
    if los_by_cid:
        L.append(r"\subsection{LOS por Diagnostico Principal (Top 10 por media)}" + "\n")
        L.append(r"""
\begin{tcolorbox}[enhanced,colback=bradlightgray,colframe=bradgray,fonttitle=\bfseries\small,
  title={Diagnosticos com maior permanencia media (altas fev)},left=5pt,right=5pt,top=4pt,bottom=4pt,boxrule=0.8pt,breakable]
\footnotesize\begin{tabularx}{\textwidth}{r X r r r r}
\toprule \textbf{\#} & \textbf{CID} & \textbf{N} & \textbf{Media} & \textbf{Mediana} & \textbf{Max} \\ \midrule
""")
        for i, c in enumerate(los_by_cid, 1):
            desc = _escape_latex(_truncate(str(c.get("cid_desc", "---")), 55))
            n = _safe_int(c.get("n"))
            avg = _safe_float(c.get("avg_los"))
            med = _safe_float(c.get("median_los"))
            mx = _safe_int(c.get("max_los"))
            L.append(f"{i} & {desc} & {n} & {avg:.1f}d & {med:.0f}d & {mx}d \\\\\n")
        L.append(r"\bottomrule \end{tabularx}" + "\n")
        L.append(r"\end{tcolorbox}" + "\n\n")

    # Excess LOS
    excess = los.get("excess", [])
    if excess:
        L.append(r"\subsection{Permanencia Excessiva ($>$2x mediana do CID)}" + "\n")
        total_ex_days = sum(_safe_int(e.get("excess_days")) for e in excess)
        L.append(r"""
\begin{tcolorbox}[enhanced,colback=alertbox,colframe=bradred,fonttitle=\bfseries\small,
  title={Dias excedentes em fevereiro},left=5pt,right=5pt,top=4pt,bottom=4pt,boxrule=1pt]
Total de \textbf{""" + _fmt_thousands(total_ex_days) + r"""} diarias excedentes identificadas.
\end{tcolorbox}
""")
        L.append(r"""
\begin{tcolorbox}[enhanced,colback=bradlightgray,colframe=bradgray,fonttitle=\bfseries\small,
  title={Top 10 CIDs com maior excesso},left=5pt,right=5pt,top=4pt,bottom=4pt,boxrule=0.8pt,breakable]
\footnotesize\begin{tabularx}{\textwidth}{r X r r r r}
\toprule \textbf{\#} & \textbf{CID} & \textbf{N} & \textbf{Med. CID} & \textbf{Med. Real} & \textbf{Dias Exc.} \\ \midrule
""")
        for i, e in enumerate(excess, 1):
            desc = _escape_latex(_truncate(str(e.get("cid_desc", "---")), 55))
            n = _safe_int(e.get("n_excess"))
            med = _safe_float(e.get("med_los"))
            avg_a = _safe_float(e.get("avg_actual"))
            ex_d = _safe_int(e.get("excess_days"))
            L.append(f"{i} & {desc} & {n} & {med:.0f}d & {avg_a:.0f}d & {_fmt_thousands(ex_d)} \\\\\n")
        L.append(r"\bottomrule \end{tabularx}" + "\n")
        L.append(r"\end{tcolorbox}" + "\n")

    L.append(r"\newpage" + "\n")

    # ================================================================
    # SECTION 7 -- VARIACAO POR MEDICO (with CID-adjusted)
    # ================================================================
    L.append(r"\section{Variacao por Medico}" + "\n")

    L.append(r"""
\begin{tcolorbox}[enhanced,colback=warnbox,colframe=bradgold,fonttitle=\bfseries\small,
  title={Criterios de Inclusao},left=5pt,right=5pt,top=4pt,bottom=4pt,boxrule=1pt]
Apenas medicos com CRM valido (excluindo padroes '0000*') e minimo de 5 internacoes
ativas em fevereiro. A Secao 4.2 apresenta a analise ajustada por CID (mesmo diagnostico,
diferentes medicos).
\end{tcolorbox}
\vspace{0.3cm}
""")

    var = docs.get("variation_los", [])
    if var:
        L.append(r"\subsection{LOS e Consumo de Medicamentos por Medico}" + "\n")
        L.append(r"""
\begin{tcolorbox}[enhanced,colback=bradlightgray,colframe=bradgray,fonttitle=\bfseries\small,
  title={Top 20 medicos por LOS medio (decrescente)},left=5pt,right=5pt,top=4pt,bottom=4pt,boxrule=0.8pt,breakable]
\footnotesize\begin{tabularx}{\textwidth}{r l X r r r r}
\toprule \textbf{\#} & \textbf{CRM} & \textbf{Nome} & \textbf{N} & \textbf{LOS Med} & \textbf{LOS Mdn} & \textbf{Meds/Caso} \\ \midrule
""")
        for i, d in enumerate(var, 1):
            crm = _escape_latex(str(d.get("crm", "---"))[:10])
            nome = _escape_latex(_truncate(str(d.get("nome", "---")), 25))
            n = _safe_int(d.get("n_cases"))
            avg = _safe_float(d.get("avg_los"))
            med = _safe_float(d.get("median_los"))
            meds_c = _safe_float(d.get("avg_meds_per_case"))
            L.append(f"{i} & {crm} & {nome} & {n} & {avg:.1f}d & {med:.0f}d & {meds_c:.1f} \\\\\n")
        L.append(r"\bottomrule \end{tabularx}" + "\n")
        L.append(r"\end{tcolorbox}" + "\n\n")

    risk_adj_mort = twin.get("risk_adjusted_mortality", [])
    if risk_adj_mort:
        L.append(r"\subsection{Mortalidade Ajustada ao Risco (JCUBE)}" + "\n")
        L.append(r"""
\begin{tcolorbox}[enhanced,colback=alertbox,colframe=bradred,fonttitle=\bfseries\small,
  title={Metodologia: Ajuste ao Risco via Embeddings},left=5pt,right=5pt,top=4pt,bottom=4pt,boxrule=1pt]
A mortalidade bruta por medico e enviesada pela gravidade dos casos. O JCUBE V6.2 computa para cada paciente
a \textbf{proximidade ao perfil de obito} na admissao (similaridade cosseno com o centroide de obito).
Medicos que atendem casos mais graves naturalmente tem pacientes com maior proximidade ao obito.
O \textbf{Ratio} compara a taxa observada vs. a esperada dado o perfil de risco:
Ratio $>$ 1.5 = mortalidade significativamente acima do esperado.
Ratio $<$ 0.5 = mortalidade abaixo do esperado (bom desempenho).
\end{tcolorbox}
\vspace{0.2cm}
""")
        L.append(r"""
\begin{tcolorbox}[enhanced,colback=bradlightgray,colframe=bradgray,fonttitle=\bfseries\small,
  title={Medicos com $\geq$5 altas em fevereiro --- ajustado ao risco},left=5pt,right=5pt,top=4pt,bottom=4pt,boxrule=0.8pt,breakable]
\footnotesize\begin{tabularx}{\textwidth}{r l X r r r r r r}
\toprule \textbf{\#} & \textbf{CRM} & \textbf{Nome} & \textbf{N Altas} & \textbf{Risco Med.} & \textbf{Obitos} & \textbf{Taxa Obs.} & \textbf{Taxa Esp.} & \textbf{Ratio} \\ \midrule
""")
        for i_ra, d_ra in enumerate(risk_adj_mort[:20], 1):
            crm_ra = _escape_latex(str(d_ra.get("crm", "---"))[:10])
            nome_ra = _escape_latex(_truncate(str(d_ra.get("nome", "---")), 22))
            n_ra = _safe_int(d_ra.get("n_cases"))
            avg_risk_ra = _safe_float(d_ra.get("avg_admission_risk"))
            n_ob_ra = _safe_int(d_ra.get("n_obito"))
            obs_rate_ra = _safe_float(d_ra.get("observed_rate")) * 100
            exp_rate_ra = _safe_float(d_ra.get("expected_rate")) * 100
            ratio_ra = _safe_float(d_ra.get("risk_adjusted_ratio"))
            if ratio_ra > 1.5:
                ratio_color = "anomred"
            elif ratio_ra > 1.0:
                ratio_color = "anomorange"
            elif ratio_ra < 0.5:
                ratio_color = "bradgreen"
            else:
                ratio_color = "bradgray"
            L.append(f"{i_ra} & {crm_ra} & {nome_ra} & {n_ra} & {avg_risk_ra:.3f} & {n_ob_ra} & {obs_rate_ra:.1f}\\% & {exp_rate_ra:.1f}\\% & \\textcolor{{{ratio_color}}}{{\\textbf{{{ratio_ra:.2f}}}}} \\\\\n")
        L.append(r"\bottomrule \end{tabularx}" + "\n")
        L.append(r"\end{tcolorbox}" + "\n")
    else:
        # Fallback to original mortality table
        mort = docs.get("mortality", [])
        if mort:
            L.append(r"\subsection{Taxa de Obito por Medico}" + "\n")
            L.append(r"""
\begin{tcolorbox}[enhanced,colback=alertbox,colframe=bradred,fonttitle=\bfseries\small,
  title={Nota Metodologica},left=5pt,right=5pt,top=4pt,bottom=4pt,boxrule=1pt]
Taxas de obito por medico devem ser interpretadas com cautela: medicos que atendem casos mais graves
naturalmente apresentam taxas maiores. Correlacao obito-medico calculada individualmente.
\end{tcolorbox}
\vspace{0.2cm}
""")
            L.append(r"""
\begin{tcolorbox}[enhanced,colback=bradlightgray,colframe=bradgray,fonttitle=\bfseries\small,
  title={Medicos com $\geq$5 altas em fevereiro},left=5pt,right=5pt,top=4pt,bottom=4pt,boxrule=0.8pt,breakable]
\footnotesize\begin{tabularx}{\textwidth}{r l X r r r}
\toprule \textbf{\#} & \textbf{CRM} & \textbf{Nome} & \textbf{Altas} & \textbf{Obitos} & \textbf{Taxa} \\ \midrule
""")
            for i, d in enumerate(mort, 1):
                crm = _escape_latex(str(d.get("crm", "---"))[:10])
                nome = _escape_latex(_truncate(str(d.get("nome", "---")), 25))
                nd = _safe_int(d.get("n_discharged"))
                no = _safe_int(d.get("n_obito"))
                tx = _safe_float(d.get("taxa_obito"))
                L.append(f"{i} & {crm} & {nome} & {nd} & {no} & {tx:.1f}\\% \\\\\n")
            L.append(r"\bottomrule \end{tabularx}" + "\n")
            L.append(r"\end{tcolorbox}" + "\n")

    L.append(r"\newpage" + "\n")

    # ================================================================
    # SECTION 8 -- EVENTOS ADVERSOS
    # ================================================================
    L.append(r"\section{Eventos Adversos}" + "\n")

    L.append(_latex_chart(charts, "ae_type"))
    L.append(_latex_chart(charts, "ae_hospital"))

    ae_types = ae.get("by_type", [])
    if ae_types:
        total_ae_t = sum(_safe_int(a.get("n_eventos")) for a in ae_types)
        L.append(r"\subsection{Por Tipo de Evento (DS\_TITULO)}" + "\n")
        L.append(r"\begin{tcolorbox}[enhanced,colback=bradlightgray,colframe=bradgray,fonttitle=\bfseries\small,title={" + str(total_ae_t) + r" eventos adversos em fevereiro},left=5pt,right=5pt,top=4pt,bottom=4pt,boxrule=0.8pt,breakable]" + "\n")
        L.append(r"\begin{tabularx}{\textwidth}{r X r r}" + "\n")
        L.append(r"\toprule \textbf{\#} & \textbf{Tipo} & \textbf{N} & \textbf{\%} \\ \midrule" + "\n")
        for i, a in enumerate(ae_types, 1):
            tipo = _escape_latex(_truncate(str(a.get("tipo_evento", "---")), 50))
            n = _safe_int(a.get("n_eventos"))
            pct = 100.0 * n / total_ae_t if total_ae_t > 0 else 0
            L.append(f"{i} & {tipo} & {n} & {pct:.1f}\\% \\\\\n")
        L.append(r"\bottomrule \end{tabularx}" + "\n")
        L.append(r"\end{tcolorbox}" + "\n\n")

        # Note about 'Outros' excluded from ranking
        n_outros = _safe_int(ae.get("n_outros"))
        if n_outros > 0:
            L.append(r"\begin{tcolorbox}[enhanced,colback=warnbox,colframe=bradgold,fonttitle=\bfseries\small,title={Eventos classificados como Outros},left=5pt,right=5pt,top=4pt,bottom=4pt,boxrule=0.8pt]" + "\n")
            L.append(f"\\textbf{{{n_outros}}} eventos adversos foram classificados como ``Outros'' pelos hospitais e excluidos do ranking acima. ")
            L.append("Analise dos relatos indica que a maioria sao \\textbf{{complicacoes pos-operatorias}} nao tipificadas corretamente no sistema. ")
            L.append("\\textbf{{Recomendacao:}} Solicitar aos hospitais credenciados a reclassificacao destes eventos para permitir analise epidemiologica adequada.\n")
            L.append(r"\end{tcolorbox}" + "\n\n")

    ae_hosp = ae.get("by_hospital", [])
    if ae_hosp:
        L.append(r"\subsection{Eventos Adversos por Hospital}" + "\n")
        L.append(r"\begin{tcolorbox}[enhanced,colback=bradlightgray,colframe=bradgray,fonttitle=\bfseries\small,title={Concentracao de eventos adversos},left=5pt,right=5pt,top=4pt,bottom=4pt,boxrule=0.8pt]" + "\n")
        L.append(r"\begin{tabularx}{\textwidth}{X r}" + "\n")
        L.append(r"\toprule \textbf{Hospital} & \textbf{N Eventos} \\ \midrule" + "\n")
        for a in ae_hosp:
            hosp = _escape_latex(_hosp_name(a.get("hospital_id")))
            n = _safe_int(a.get("n_eventos"))
            L.append(f"{hosp} & {n} \\\\\n")
        L.append(r"\bottomrule \end{tabularx}" + "\n")
        L.append(r"\end{tcolorbox}" + "\n\n")

    multi = ae.get("multi_event", [])
    if multi:
        L.append(r"\subsection{Internacoes com Multiplos Eventos Adversos}" + "\n")
        L.append(r"\begin{tcolorbox}[enhanced,colback=alertbox,colframe=bradred,fonttitle=\bfseries\small,title={Internacoes com 2+ eventos em fevereiro},left=5pt,right=5pt,top=4pt,bottom=4pt,boxrule=1pt,breakable]" + "\n")
        L.append(r"\footnotesize\begin{tabularx}{\textwidth}{r r r X}" + "\n")
        L.append(r"\toprule \textbf{ID Inter.} & \textbf{Hospital} & \textbf{N Ev.} & \textbf{Tipos} \\ \midrule" + "\n")
        for m in multi:
            iid = _safe_int(m.get("ID_CD_INTERNACAO"))
            hosp = _escape_latex(_hosp_name(m.get("ID_CD_HOSPITAL")))
            ne = _safe_int(m.get("n_eventos"))
            tipos = _escape_latex(_truncate(str(m.get("tipos", "---")), 45))
            L.append(f"{iid} & {hosp} & {ne} & {tipos} \\\\\n")
        L.append(r"\bottomrule \end{tabularx}" + "\n")
        L.append(r"\end{tcolorbox}" + "\n")

    L.append(r"\newpage" + "\n")

    # ================================================================
    # SECTION 8.5 -- QUALIDADE DE DADOS POR HOSPITAL
    # ================================================================
    dq = data_quality or []
    if dq:
        L.append(r"\section{Qualidade de Dados por Hospital}" + "\n")
        L.append(r"""
\begin{tcolorbox}[enhanced,colback=warnbox,colframe=bradgold,fonttitle=\bfseries\small,
  title={Impacto da Qualidade de Dados nas Analises},left=5pt,right=5pt,top=4pt,bottom=4pt,boxrule=1pt]
A qualidade das analises anteriores depende diretamente da qualidade dos dados registrados
por cada hospital. Hospitais com score baixo comprometem a capacidade de auditoria.
\textbf{Score} = 100 - media(Alta sem Tipo + CID sem Principal + CRM Invalido).
\textcolor{anomred}{Score $<$ 50 = critico} |
\textcolor{bradgold}{50--75 = atencao} |
\textcolor{bradgreen}{$>$ 75 = adequado}.
\end{tcolorbox}
\vspace{0.3cm}
""")
        L.append(r"""
\begin{tcolorbox}[enhanced,colback=bradlightgray,colframe=bradgray,fonttitle=\bfseries\small,
  title={Qualidade de dados por hospital (ordenado por score crescente)},
  left=5pt,right=5pt,top=4pt,bottom=4pt,boxrule=0.8pt,breakable]
\footnotesize\begin{tabularx}{\textwidth}{r X r r r r}
\toprule \textbf{\#} & \textbf{Hospital} & \textbf{Score} & \textbf{Alta s/ Tipo} & \textbf{CID s/ Princ.} & \textbf{CRM Inv.} \\ \midrule
""")
        for idx_dq, dq_row in enumerate(dq[:30], 1):
            hname_dq = _escape_latex(_hosp_name(dq_row.get("hospital_id")))
            score_dq = _safe_float(dq_row.get("quality_score"))
            pct_alta_dq = _safe_float(dq_row.get("pct_alta_sem_tipo"))
            pct_cid_dq = _safe_float(dq_row.get("pct_cid_sem_principal"))
            pct_crm_dq = _safe_float(dq_row.get("pct_crm_invalido"))
            if score_dq < 50:
                score_color_dq = "anomred"
            elif score_dq < 75:
                score_color_dq = "bradgold"
            else:
                score_color_dq = "bradgreen"
            L.append(f"{idx_dq} & {hname_dq} & \\textcolor{{{score_color_dq}}}{{\\textbf{{{score_dq:.1f}}}}} & {pct_alta_dq:.1f}\\% & {pct_cid_dq:.1f}\\% & {pct_crm_dq:.1f}\\% \\\\\n")
        L.append(r"\bottomrule \end{tabularx}" + "\n")
        L.append(r"\end{tcolorbox}" + "\n")

    L.append(r"\newpage" + "\n")

    # ================================================================
    # SECTION 9 -- MEDICAMENTOS
    # ================================================================
    L.append(r"\section{Medicamentos}" + "\n")

    L.append(r"""
\begin{tcolorbox}[enhanced,colback=warnbox,colframe=bradgold,fonttitle=\bfseries\small,
  title={Nota sobre Dados Financeiros},left=5pt,right=5pt,top=4pt,bottom=4pt,boxrule=1pt]
GHO-BRADESCO opera como \textbf{auditora/operadora} --- dados de faturamento hospitalar
nao estao disponiveis. A analise utiliza como proxy os dados de
\textbf{prescricoes de medicamentos} (tabela capta\_produtos\_capr).
\end{tcolorbox}
\vspace{0.3cm}
""")

    L.append(_latex_chart(charts, "med_bars"))

    ps = meds.get("summary", {})
    if ps:
        tp = _safe_int(ps.get("total_produtos"))
        ni = _safe_int(ps.get("n_internacoes_com_produto"))
        nd = _safe_int(ps.get("n_medicamentos_distintos"))
        tu = _safe_int(ps.get("total_unidades"))
        L.append(r"""
\begin{tcolorbox}[enhanced,colback=kpibox,colframe=bradblue,fonttitle=\bfseries\small,
  title={Resumo de Medicamentos --- Fevereiro},left=5pt,right=5pt,top=4pt,bottom=4pt,boxrule=1pt]
\begin{itemize}[leftmargin=1.5em]
""")
        L.append(f"\\item \\textbf{{{_fmt_thousands(tp)}}} prescricoes de medicamentos\n")
        L.append(f"\\item \\textbf{{{_fmt_thousands(ni)}}} internacoes com ao menos 1 prescricao\n")
        L.append(f"\\item \\textbf{{{_fmt_thousands(nd)}}} medicamentos distintos\n")
        L.append(f"\\item \\textbf{{{_fmt_thousands(tu)}}} unidades totais prescritas\n")
        if ni > 0:
            L.append(f"\\item Media: \\textbf{{{tp/ni:.1f}}} prescricoes por internacao\n")
        L.append(r"\end{itemize}" + "\n")
        L.append(r"\end{tcolorbox}" + "\n\n")

    top_meds = meds.get("top_medications", [])
    if top_meds:
        L.append(r"\subsection{Top 20 Medicamentos Mais Prescritos}" + "\n")
        L.append(r"\begin{tcolorbox}[enhanced,colback=bradlightgray,colframe=bradgray,fonttitle=\bfseries\small,title={Medicamentos por frequencia de prescricao},left=5pt,right=5pt,top=4pt,bottom=4pt,boxrule=0.8pt,breakable]" + "\n")
        L.append(r"\footnotesize\begin{tabularx}{\textwidth}{r X r r r}" + "\n")
        L.append(r"\toprule \textbf{\#} & \textbf{Medicamento} & \textbf{Prescr.} & \textbf{Unidades} & \textbf{Inter.} \\ \midrule" + "\n")
        for i, m in enumerate(top_meds, 1):
            med_name = _escape_latex(_truncate(str(m.get("medicamento", "---")), 35))
            np_ = _safe_int(m.get("n_prescricoes"))
            tu_ = _safe_int(m.get("total_unidades"))
            ni_ = _safe_int(m.get("n_internacoes"))
            L.append(f"{i} & {med_name} & {_fmt_thousands(np_)} & {_fmt_thousands(tu_)} & {ni_} \\\\\n")
        L.append(r"\bottomrule \end{tabularx}" + "\n")
        L.append(r"\end{tcolorbox}" + "\n\n")

    # ---- 9.x: Medication Embedding Landscape ----
    med_emb_data = twin.get("medication_embeddings", [])
    if med_emb_data:
        L.append(r"\subsection{Paisagem de Medicamentos: Alinhamento com Desfecho (Embeddings)}" + "\n")
        L.append(r"""
\begin{tcolorbox}[enhanced,colback=warnbox,colframe=bradgold,fonttitle=\bfseries\small,
  title={Medicamentos no Espaco de Embeddings},
  left=5pt,right=5pt,top=4pt,bottom=4pt,boxrule=1pt]
Para cada medicamento com 10+ prescricoes, computamos o \textbf{centroide dos embeddings} de
todos os produtos associados. A projecao no eixo obito$\rightarrow$alta indica se o medicamento
aparece em contextos mais alinhados com obito (positivo) ou com alta melhorada (negativo).

\textbf{Importante:} Barras vermelhas NAO significam que o medicamento causa obito --- indicam
associacao com \textbf{gravidade clinica}. Medicamentos de UTI e quimioterapicos tendem a
pontuar alto por serem prescritos em casos mais graves.
\end{tcolorbox}
\vspace{0.3cm}
""")
        L.append(_latex_chart(charts, "med_landscape"))

        # Table: top 10 death-aligned + top 10 recovery-aligned
        med_sorted_death = sorted(med_emb_data, key=lambda x: -x["proj_outcome"])
        med_sorted_alta = sorted(med_emb_data, key=lambda x: x["proj_outcome"])

        L.append(r"""
\begin{tcolorbox}[enhanced,colback=alertbox,colframe=bradred,fonttitle=\bfseries\small,
  title={Top 10 Medicamentos Alinhados com Perfil de Obito},
  left=5pt,right=5pt,top=4pt,bottom=4pt,boxrule=0.8pt,breakable]
\footnotesize\begin{tabularx}{\textwidth}{r X r r r r}
\toprule \textbf{\#} & \textbf{Medicamento} & \textbf{Prescr.} & \textbf{Cos Obito} & \textbf{Cos Alta} & \textbf{Projecao} \\ \midrule
""")
        for idx_md, md in enumerate(med_sorted_death[:10], 1):
            md_name = _escape_latex(_truncate(str(md.get("medicamento", "---")), 35))
            md_np = _safe_int(md.get("n_prescricoes"))
            md_cd = _safe_float(md.get("cos_death"))
            md_ca = _safe_float(md.get("cos_alta"))
            md_proj = _safe_float(md.get("proj_outcome"))
            L.append(f"{idx_md} & {md_name} & {_fmt_thousands(md_np)} & {md_cd:.4f} & {md_ca:.4f} & \\textcolor{{anomred}}{{\\textbf{{{md_proj:+.4f}}}}} \\\\\n")
        L.append(r"\bottomrule \end{tabularx}" + "\n")
        L.append(r"\end{tcolorbox}" + "\n\n")

        L.append(r"""
\begin{tcolorbox}[enhanced,colback=successbox,colframe=bradgreen,fonttitle=\bfseries\small,
  title={Top 10 Medicamentos Alinhados com Perfil de Recuperacao},
  left=5pt,right=5pt,top=4pt,bottom=4pt,boxrule=0.8pt,breakable]
\footnotesize\begin{tabularx}{\textwidth}{r X r r r r}
\toprule \textbf{\#} & \textbf{Medicamento} & \textbf{Prescr.} & \textbf{Cos Obito} & \textbf{Cos Alta} & \textbf{Projecao} \\ \midrule
""")
        for idx_ma, ma in enumerate(med_sorted_alta[:10], 1):
            ma_name = _escape_latex(_truncate(str(ma.get("medicamento", "---")), 35))
            ma_np = _safe_int(ma.get("n_prescricoes"))
            ma_cd = _safe_float(ma.get("cos_death"))
            ma_ca = _safe_float(ma.get("cos_alta"))
            ma_proj = _safe_float(ma.get("proj_outcome"))
            L.append(f"{idx_ma} & {ma_name} & {_fmt_thousands(ma_np)} & {ma_cd:.4f} & {ma_ca:.4f} & \\textcolor{{bradgreen}}{{\\textbf{{{ma_proj:+.4f}}}}} \\\\\n")
        L.append(r"\bottomrule \end{tabularx}" + "\n")
        L.append(r"\end{tcolorbox}" + "\n\n")

    meds_cid = meds.get("intensity_by_cid", [])
    if meds_cid:
        L.append(r"\subsection{Intensidade de Medicamentos por CID Principal}" + "\n")
        L.append(r"\begin{tcolorbox}[enhanced,colback=bradlightgray,colframe=bradgray,fonttitle=\bfseries\small,title={Diagnosticos que mais consomem medicamentos},left=5pt,right=5pt,top=4pt,bottom=4pt,boxrule=0.8pt,breakable]" + "\n")
        L.append(r"\footnotesize\begin{tabularx}{\textwidth}{r X r r r}" + "\n")
        L.append(r"\toprule \textbf{\#} & \textbf{CID} & \textbf{Prescr./Inter.} & \textbf{Prescricoes} & \textbf{Inter.} \\ \midrule" + "\n")
        for i, c in enumerate(meds_cid, 1):
            desc = _escape_latex(_truncate(str(c.get("cid_desc", "---")), 55))
            ppi = _safe_float(c.get("prescricoes_por_inter"))
            tp_ = _safe_int(c.get("n_prescricoes"))
            ni_ = _safe_int(c.get("n_internacoes"))
            L.append(f"{i} & {desc} & {ppi:.1f} & {_fmt_thousands(tp_)} & {ni_} \\\\\n")
        L.append(r"\bottomrule \end{tabularx}" + "\n")
        L.append(r"\end{tcolorbox}" + "\n")

    L.append(r"\newpage" + "\n")

    # ================================================================
    # SECTION 10 -- NOTA METODOLOGICA
    # ================================================================
    L.append(r"""
\section*{Nota Metodologica}
\addcontentsline{toc}{section}{Nota Metodologica}

\begin{tcolorbox}[enhanced,colback=bradlightgray,colframe=bradgray,fonttitle=\bfseries\small,
  title={Fontes de Dados e Tecnologia},left=5pt,right=5pt,top=4pt,bottom=4pt,boxrule=0.8pt,breakable]
\begin{itemize}[leftmargin=1.5em,itemsep=2pt]
\item \textbf{System of Records:} DuckDB (aggregated\_fixed\_union.db) com 411 tabelas, 46,7M registros
\item \textbf{Filtro:} source\_db = 'GHO-BRADESCO', periodo de atividade em fevereiro 2026
\item \textbf{Papel da Operadora:} GHO-BRADESCO atua como auditora/operadora --- dados de faturamento
  hospitalar (faturas, itens, glosas) nao sao registrados neste sistema. O proxy financeiro utilizado
  e a tabela de prescricoes de medicamentos (capta\_produtos\_capr)
\item \textbf{Digital Twin:} JCUBE V6.2 Dense Temporal JEPA
  \begin{itemize}
  \item 35,2M nos, 165,6M arestas, 128 dimensoes
  \item Arquitetura: GraphGPS (3 camadas) + TGN (memoria temporal) + BGE-M3 (ancoras ontologicas)
  \item Treinamento: 4$\times$ H100 (DDP com delta sync), 5 epocas (2 foundation + 3 temporal)
  \item Checkpoint utilizado: epoca 3
  \end{itemize}
\item \textbf{Alerta Preditivo (Secao 2):}
  \begin{itemize}
  \item \textbf{Centroide de obito:} media dos embeddings de todas as internacoes com desfecho OBITO
  \item \textbf{Proximidade ao obito:} similaridade cosseno entre embedding da internacao aberta e o centroide de obito
  \item \textbf{Velocidade de trajetoria:} norma do vetor diferenca entre embeddings de internacoes consecutivas do mesmo paciente
  \item \textbf{Direcao ao obito:} cosseno do vetor delta com o centroide de obito (positivo = movendo em direcao a obito)
  \item \textbf{Projecao de marco:} cosseno do centroide mensal (jan vs fev) com o centroide de obito, extrapolando tendencia
  \item \textbf{Centroide de alta melhorada:} media dos embeddings de internacoes com desfecho ALTA MELHORADA
  \item \textbf{Radar de risco hospitalar:} cosseno do embedding do hospital com centroide de obito vs centroide de alta
  \item \textbf{Trajetoria cronica:} regressao linear da proximidade ao obito ao longo de 5+ internacoes do mesmo paciente
  \item \textbf{Paisagem de medicamentos:} centroide dos embeddings de produtos por medicamento, projetado no eixo obito-alta
  \end{itemize}
\item \textbf{Matriz de Risco (Secao 4):}
  \begin{itemize}
  \item CID $\times$ Hospital: celulas com LOS $>$ 1.5$\times$ mediana do CID marcadas como anomalas
  \item Medico $\times$ CID: excesso de LOS ajustado pela mediana do diagnostico
  \item Intensidade de medicamentos: hospitais com $>$1.5$\times$ a media global de prescricoes por CID
  \end{itemize}
\item \textbf{Anomalias z-score:} distancia euclidiana ao centroide no espaco de embeddings, z $>$ 2.0
\item \textbf{Alta:} IN\_SITUACAO = 2 + ultimo FL\_DESOSPITALIZACAO da evo\_status + lookup tipo\_final\_monit filtrado por source\_db
\item \textbf{Tipo de Alta:} Agrupado em 5 categorias: Alta Normal, Obito, Transferencia, Administrativo, Outro
\item \textbf{CIDs:} Apenas diagnosticos principais (IN\_PRINCIPAL = 'S')
\item \textbf{Medicos:} CRM (DS\_CONSELHO\_CLASSE), excluindo padroes '0000*', minimo 5 casos
\item \textbf{Mortalidade por medico:} Correlacao obito-medico individual (nao taxa global)
\item \textbf{Tendencia:} 6 meses (Set/2025 a Fev/2026) com todos os KPIs: admissoes, altas, LOS media+mediana,
  taxa de obito, eventos adversos, taxa de readmissao, prescricoes de medicamentos
\item \textbf{Deltas MoM:} Todas as comparacoes Jan$\rightarrow$Fev incluem variacao percentual com seta direcional
\end{itemize}
\end{tcolorbox}
""")

    L.append(r"\end{document}")
    return "\n".join(L)


# -----------------------------------------------------------------
# Main report function
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

    print("=" * 70)
    print("JCUBE V6.2 — RETROSPECTIVA FEVEREIRO 2026 GHO-BRADESCO (v2)")
    print("=" * 70)

    con = duckdb.connect(DB_PATH, read_only=True)

    # Load hospital name mapping first
    _load_hospital_names(con)

    # Fetch all data (14 sections)
    kpis = _fetch_kpis(con)
    hospitals = _fetch_hospital_performance(con)
    trend = _fetch_trend_6m(con)
    profile = _fetch_clinical_profile(con)
    los = _fetch_los_analysis(con)
    risk_matrix = _fetch_risk_matrix(con)
    doc_cid_comp = _fetch_doctor_cid_comparison(con)
    med_intensity = _fetch_med_intensity_matrix(con)
    docs = _fetch_physicians(con)
    ae = _fetch_adverse_events(con)
    meds_data = _fetch_medications(con)

    # Load twin for predictive analysis (also computes risk-adjusted mortality + hospital drift)
    twin = _load_twin_predictive(con)

    # Data quality scores per hospital
    data_quality = _fetch_data_quality(con)

    con.close()

    twin_stats = twin.get("stats", {})

    # Generate charts
    print("\n[12.5/14] Generating matplotlib charts ...")
    import time as _t
    _tc = _t.time()
    charts = _generate_all_charts(
        kpis, hospitals, trend, profile, los, risk_matrix,
        doc_cid_comp, med_intensity, docs, ae, meds_data,
        twin, twin_stats,
    )
    print(f"    done in {_t.time()-_tc:.1f}s")

    # Generate LaTeX
    print("\n[13/14] Generating LaTeX ...")
    latex = _generate_latex(
        kpis, hospitals, trend, profile, los, risk_matrix,
        doc_cid_comp, med_intensity, docs, ae, meds_data,
        twin, twin_stats, charts=charts, data_quality=data_quality,
    )

    tex_path = OUTPUT_PDF.replace(".pdf", ".tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"    LaTeX written to {tex_path}")

    # Compile PDF
    print("[14/14] Compiling PDF (2 passes) ...")
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
