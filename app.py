# -*- coding: utf-8 -*-
# MP-Insight · logKd Prediction & Enrichment Explorer (publication-oriented)
# Key change: remove "Risk level" classification; keep physics-informed enrichment interpretation.

import io, os, json, math
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
from io import StringIO


import re

def _normalize_colname(c):
    c = str(c).strip().replace("\u00a0", " ")
    c = re.sub(r"\s+", " ", c)
    return c

NAME_MAP = [
    (r"^(plastic[\s_-]?density).*", "Plastic Density"),
    (r"^(ssa).*", "SSA"),
    (r"^(particle[\s_-]?size).*", "Particle Size"),
    (r"^(logk[_\s-]?ow|logkow)$", "LogKow"),
    (r"^e$", "E"), (r"^s$", "S"),
    (r"^a$", "A"), (r"^b$", "B"), (r"^v$", "V"),
    (r"^(ph)$", "PH"),
    (r"^(salinity).*", "Salinity"),
    (r"^(temp.*|temperature.*)", "Temperature"),
    (r"^(mps?[\s_/-]?type[s]?)$", "MPs types"),
    (r"^(age(ing)?[\s_-]?(type|status))$", "Ageing_Status"),
    (r"^(water[\s_-]?type)$", "Water Type"),
]

def standardize_input_df(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()
    mapped = {}
    for col in df.columns:
        low = _normalize_colname(col).lower()
        for patt, std in NAME_MAP:
            if re.match(patt, low):
                mapped[col] = std
                break
    df = df.rename(columns=mapped)
    return df


# ---------- Streamlit page config ----------
st.set_page_config(page_title="MP-Insight", layout="wide")

# ---------- Matplotlib global style ----------
mpl.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 300,
    "font.family": "Times New Roman",
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 14,
    "axes.linewidth": 1.2,
    "lines.linewidth": 2.0,
})
mpl.rcParams["mathtext.fontset"] = "custom"
mpl.rcParams["mathtext.rm"] = "Times New Roman"
mpl.rcParams["mathtext.it"] = "Times New Roman:italic"
mpl.rcParams["mathtext.bf"] = "Times New Roman:bold"

# ---------- Global CSS (keep mostly unchanged + FIX: enforce Times for markdown headings) ----------
st.markdown(r"""
<style>
:root{
  --font-body: 18px;
  --font-small: 16px;
  --font-label: 17px;
  --ctl-h: 48px;
  --ctl-radius: 10px;
  --text-color: #111;
  --ctl-bg: #f2f4f8;
  --ctl-border: #cfd6e4;
  --ctl-border-hover: #b9c6db;
  --ctl-border-focus: #8aa4c8;
}

html, body, [data-testid], [class^='st-']{
  font-family:"Times New Roman", Times, serif !important;
  font-size: var(--font-body) !important;
  line-height: 1.6 !important;
  color: var(--text-color) !important;
}

/* CRITICAL FIX: Streamlit markdown containers sometimes keep default sans font */
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] *{
  font-family:"Times New Roman", Times, serif !important;
}
[data-testid="stHeading"], [data-testid="stHeading"] *{
  font-family:"Times New Roman", Times, serif !important;
}

header[data-testid="stHeader"]{
  position:relative !important; top:0; background:#fff;
  border-bottom:1px solid #e8e8e8; height:auto !important; padding:0 !important;
}
.block-container{ max-width:1280px; padding-top:.8rem !important; }

h1{ font-size:34px !important; font-weight:800 !important; letter-spacing:.3px; }
h2{ font-size:22px !important; font-weight:700 !important; }
h3{ font-size:20px !important; font-weight:700 !important; }

/* form labels */
label,[data-baseweb="select"] label,[data-baseweb="input"] label,[data-baseweb="slider"] label{
  font-weight:650 !important; color:#000 !important; font-size: var(--font-label) !important;
}

/* Select */
[data-testid="stSelectbox"] > div[data-baseweb="select"]{
  box-shadow:none!important; background:transparent!important; border:none!important;
}
[data-testid="stSelectbox"] div[role="combobox"]{
  height: var(--ctl-h) !important; min-height: var(--ctl-h) !important;
  border-radius: var(--ctl-radius) !important;
  border: 1px solid var(--ctl-border) !important;
  background: var(--ctl-bg) !important;
  padding: 0 12px !important; display:flex !important; align-items:center !important;
  box-shadow:none !important; font-size: var(--font-body) !important;
}
[data-testid="stSelectbox"] div[role="combobox"]:hover{ border-color:var(--ctl-border-hover) !important; }
[data-testid="stSelectbox"] svg{ opacity:.85; transform:scale(1.05); }

/* NumberInput */
[data-testid="stNumberInput"] > div{
  display:flex !important; align-items:center !important;
  height:var(--ctl-h) !important; gap:0 !important;
  background:var(--ctl-bg) !important;
  border:1px solid var(--ctl-border) !important;
  border-radius:var(--ctl-radius) !important;
  overflow:hidden !important;
}
[data-testid="stNumberInput"] > div:hover{ border-color:var(--ctl-border-hover) !important; }
[data-testid="stNumberInput"] > div:focus-within{
  border-color:var(--ctl-border-focus) !important;
  box-shadow:0 0 0 2px rgba(154,185,225,.25) inset !important;
}
[data-testid="stNumberInput"] [data-baseweb="input"]{
  background:transparent !important; border:none !important; box-shadow:none !important;
}
[data-testid="stNumberInput"] input{
  height:100% !important; flex:1 1 auto !important;
  padding:0 12px !important; font-size:var(--font-body) !important;
  background:transparent !important; border:none !important; box-shadow:none !important;
  outline:none !important;
}
[data-testid="stNumberInput"] button{
  height:100% !important; width:46px !important;
  margin:0 !important; padding:0 !important;
  background:transparent !important; box-shadow:none !important;
  border:none !important; border-left:1px solid var(--ctl-border) !important;
  border-radius:0 !important; cursor:pointer !important;
}
[data-testid="stNumberInput"] button:last-of-type{
  border-radius:0 var(--ctl-radius) var(--ctl-radius) 0 !important;
}
[data-testid="stNumberInput"] button:hover{ background:#e9eef5 !important; }
[data-testid="stNumberInput"] svg{ transform:scale(1.18); opacity:.8; }

/* BaseWeb fallback */
[data-baseweb="input"], [data-baseweb="select"]{
  border:1px solid var(--ctl-border) !important;
  background:var(--ctl-bg) !important;
  border-radius:var(--ctl-radius) !important;
  box-shadow:none !important;
}

/* Slider label */
[data-testid="stSlider"] [data-baseweb="slider"]{ padding-top:6px !important; }
[data-testid="stSlider"] .stSliderLabel,
[data-testid="stSlider"] label{ font-size: var(--font-small) !important; }

/* KPI */
.kpi{background:#fff; border:1px solid #eee; border-radius:12px; padding:14px 16px; box-shadow:0 2px 8px rgba(0,0,0,.04);}

/* Note */
.note-box{
  border:1px solid #e8e8e8; background:#fafafa; border-radius:10px;
  padding:12px 14px; margin:10px 0 6px 0; font-size:18px !important; color:#333; line-height:1.55;
}
[data-testid="stTable"], [data-testid="stDataFrame"] *{
  font-family:'Times New Roman', Times, serif !important; font-size:16px !important;
}

/* Formula */
.formula-box{
  font-size:15px !important; line-height:1.32 !important;
  font-family:'Times New Roman', Times, serif !important; color:#222 !important;
}

/* Hide number_input extra help container */
[data-testid="stNumberInput"] > div:nth-child(3){ display: none !important; }

/* Sidebar section small heading */
.sb-h{
  font-family:"Times New Roman", Times, serif !important;
  font-weight:800 !important;
  font-size:18px !important;
  margin:8px 0 6px 0 !important;
}
</style>
""", unsafe_allow_html=True)

# ---------- Title ----------
st.markdown(
    '<h1 style="font-family:Times New Roman; font-weight:800;">MP-Insight · log K<sub>MP/W</sub> Prediction & Enrichment Explorer</h1>',
    unsafe_allow_html=True
)

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parent 

DATA_DIR    = ROOT / "data"
MODELS_DIR  = ROOT / "models"
CONFIG_DIR  = ROOT / "config"
REPORTS_DIR = ROOT / "reports"
LABEL_LOGK = r"$\log K_{MP/W}$"
LABEL_LOGK_HTML = r"log K<sub>MP/W</sub>"
LABEL_K_HTML = r"K<sub>MP/W</sub>"
# ---------- Utils ----------
def load_json(p: Path, default: Dict):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def get_training_uniques(mode: str):
    csv_path = DATA_DIR / ("view_known.csv" if mode == "Known polymer" else "view_unknown.csv")
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return {
            "MPs types": ["PE","PP","PS","PVC","PET","PA","PLA","CPE"],
            "Ageing_Status": ["Pristine","Aged"],
            "Water Type": ["Freshwater","Seawater"],
        }
    out = {}
    if "MPs types" in df.columns:
        out["MPs types"] = [x for x in df["MPs types"].dropna().astype(str).unique().tolist() if x != ""]
    if "Ageing_Status" in df.columns:
        out["Ageing_Status"] = [x for x in df["Ageing_Status"].dropna().astype(str).unique().tolist() if x != ""]
    if "Water Type" in df.columns:
        out["Water Type"] = [x for x in df["Water Type"].dropna().astype(str).unique().tolist() if x != ""]
    out.setdefault("MPs types", ["PE","PP","PS","PVC","PET","PA","PLA","CPE"])
    out.setdefault("Ageing_Status", ["Pristine","Aged"])
    out.setdefault("Water Type", ["Freshwater","Seawater"])
    return out

def get_required_cols(mode: str) -> List[str]:
    base = [
        "Ageing_Status", "Water Type", "Plastic Density",
        "SSA", "Particle Size", "LogKow", "E", "S", "A", "B", "V",
        "PH", "Salinity", "Temperature"
    ]
    if mode == "Known polymer":
        return ["MPs types"] + base
    return base

def validate_csv_columns(df: pd.DataFrame, required_cols: List[str]) -> Dict:
    df_cols = [c.strip() for c in df.columns.astype(str).tolist()]
    missing = [c for c in required_cols if c not in df_cols]
    extras  = [c for c in df_cols if c not in required_cols]
    status_rows = []
    for c in required_cols:
        status_rows.append({"Column": c, "Status": "✅ Present" if c in df_cols else "❌ Missing"})
    for c in extras:
        status_rows.append({"Column": c, "Status": "⚠️ Extra (not used)"})
    return {"missing": missing, "extras": extras, "status_df": pd.DataFrame(status_rows)}

def make_template_csv(required_cols: List[str], rows: int = 3) -> bytes:
    tpl = pd.DataFrame([{c: "" for c in required_cols} for _ in range(rows)])
    buf = StringIO()
    tpl.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8-sig")

# ---------- Core physics helpers ----------
def cmp_ugL_to_gmL(cmp_ugL: float) -> float:
    # 1 μg/L = 1e-9 g/mL
    return float(cmp_ugL) * 1e-9

def R_pred_from_logKd(logKd: float, C_MP_gmL: float) -> float:
    # R_pred = (Kd*C_MP)/(1+Kd*C_MP), Kd=10^(logKd)
    Kd = 10 ** float(logKd)
    x = Kd * float(C_MP_gmL)
    return float(x / (1.0 + x))

def phi_from_logKd(logKd: float, C_MP_gmL: float) -> float:
    # Φ = log10(Kd*C_MP) = logKd + log10(C_MP)
    return float(logKd) + math.log10(max(float(C_MP_gmL), 1e-300))

def logKd_star_from_Rref(R_ref: float, C_MP_gmL: float) -> float:
    # logKd* = log10( R_ref/((1-R_ref)*C_MP) )
    R_ref = float(R_ref)
    R_ref = min(max(R_ref, 1e-300), 1.0 - 1e-12)
    denom = (1.0 - R_ref) * max(float(C_MP_gmL), 1e-300)
    return math.log10(R_ref / denom)

# ---------- Plot helpers ----------
def plot_prediction_panel(y_pred, PI_low, PI_high, show_ref, logKd_star, paper=False):
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    ax.axvspan(PI_low, PI_high, alpha=0.12, lw=0)
    ax.axvline(y_pred, lw=2.2, color="k", label=r"Predicted $\log K_{MP/W}$")
    if show_ref:
        ax.axvline(logKd_star, lw=2.0, ls="--", color="k",
                   label=r"Threshold $\log K_{MP/W}^{*}$ for target $R_{ref}$")
    ax.set_xlabel(r"$\log K_{MP/W}$")
    ax.set_title(r"Panel A. Model-based prediction of $\log K_{MP/W}$", pad=6)
    ax.set_yticks([])
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="upper left")
    if paper:
        ax.grid(False)
        for s in ["top", "right", "left"]:
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig

def plot_scenario_Rpred_vs_CMP(y_pred, PI_low, PI_high, C_MP_current_ugL, paper=False):
    # Environmental-relevant default range: 1e-3 to 1e3 μg/L
    xs_ugL = np.logspace(-3, 3, 320)
    xs_gmL = xs_ugL * 1e-9

    R_mid = np.array([R_pred_from_logKd(y_pred, c) for c in xs_gmL])

    R_lo  = np.array([R_pred_from_logKd(PI_low,  c) for c in xs_gmL])
    R_hi  = np.array([R_pred_from_logKd(PI_high, c) for c in xs_gmL])
    R_low = np.minimum(R_lo, R_hi)
    R_high= np.maximum(R_lo, R_hi)

    c_cur = max(float(C_MP_current_ugL), 1e-12)
    r_cur = R_pred_from_logKd(y_pred, cmp_ugL_to_gmL(c_cur))

    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    ax.fill_between(
        xs_ugL,
        np.clip(R_low, 1e-12, 1.0),
        np.clip(R_high, 1e-12, 1.0),
        alpha=0.12, lw=0
    )
    ax.plot(xs_ugL, np.clip(R_mid, 1e-12, 1.0), lw=2.0, color="k")

    ax.axvline(c_cur, lw=1.2, ls="--", color="k")
    ax.scatter([c_cur], [max(r_cur, 1e-12)], s=28, color="k", zorder=5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-12, 1.0)
    ax.set_xlabel(r"$C_{MP}$ ($\mu$g/L)")
    ax.set_ylabel(r"Sorbed fraction ($R_{pred}$)")
    ax.set_title(r"Panel B. Scenario response: $R_{pred}$ vs $C_{MP}$", pad=6)

    ax.text(
        0.02, 0.02,
        "Shaded: 95% PI propagated from log KMP/W",
        transform=ax.transAxes,
        fontsize=11,
        va="bottom", ha="left"
    )

    if paper:
        ax.grid(False)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig

# ---------- Sidebar: Mode / Paper mode ----------
MODE = st.sidebar.selectbox("Mode", ["Known polymer", "Unknown polymer"], index=0)

if MODE == "Known polymer":
    MODEL_PATH = MODELS_DIR / "xgb_known_polymer.pkl"
    META_PATH  = CONFIG_DIR / "model_meta_known.json"
    CONF_PATH  = MODELS_DIR / "conformal_known.json"
else:
    MODEL_PATH = MODELS_DIR / "xgb_unknown_polymer.pkl"
    META_PATH  = CONFIG_DIR / "model_meta_unknown.json"
    CONF_PATH  = MODELS_DIR / "conformal_unknown.json"

pipe = None
try:
    if not MODEL_PATH.exists():
        st.error(f"Model file missing: {MODEL_PATH}")
    else:
        pipe = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Cannot load model: {e}")

meta = load_json(META_PATH, {"default_threshold": 4.0, "default_R": 0.8})
conf = load_json(CONF_PATH, {"q95": 0.5})
q95 = float(conf.get("q95", 0.5))

paper_mode = st.sidebar.toggle(
    "Paper snapshot mode",
    value=False,
    help="Use when you need publication-quality figures (PNG/SVG)."
)
if paper_mode:
    mpl.rcParams.update({"savefig.dpi": 600})

# ---------- Sidebar: Scenario inputs (minimal + clean) ----------
st.sidebar.markdown("<div class='sb-h'>Scenario inputs</div>", unsafe_allow_html=True)

# C_MP main
if "cmp_ugL" not in st.session_state:
    st.session_state["cmp_ugL"] = 0.501

st.sidebar.markdown("**C<sub>MP</sub> (μg/L)**", unsafe_allow_html=True)
C_MP_ugL = st.sidebar.number_input(
    " ",  # keep label visually clean (we render title above with HTML)
    value=float(st.session_state["cmp_ugL"]),
    min_value=0.0,
    step=0.01,
    format="%.3f",
    help="Scenario concentration used for interpretation only. Internally converted to g/mL (1 μg/L = 1e−9 g/mL).",
    key="cmp_number_input"
)
st.session_state["cmp_ugL"] = float(C_MP_ugL)
C_MP_gmL = cmp_ugL_to_gmL(C_MP_ugL)

# R_ref + show line
if "r_ref" not in st.session_state:
    st.session_state["r_ref"] = 1e-4

st.sidebar.markdown("**R<sub>ref</sub> (target sorbed fraction)**", unsafe_allow_html=True)
R_ref = st.sidebar.number_input(
    "  ",  # visual clean label; title above
    value=float(st.session_state["r_ref"]),
    min_value=1e-12,
    max_value=0.9,
    step=1e-4,
    format="%.1e",
    help="Defines a physical threshold line (log Kd*). Not used for prediction or classification.",
    key="rref_number_input"
)
st.session_state["r_ref"] = float(R_ref)

show_ref_line = st.sidebar.checkbox("Show threshold line (log K_MP/W*)", value=True)

# Advanced settings (NO nesting inside other expanders)
with st.sidebar.expander("Advanced settings", expanded=False):
    use_log_slider = st.toggle("Use log-scale slider for C_MP", value=False, key="use_log_slider")
    if use_log_slider:
        cur = max(float(st.session_state["cmp_ugL"]), 1e-12)
        cur_log = float(np.clip(np.log10(cur), -6.0, 6.0))
        log10_cmp = st.slider("log10(C_MP, μg/L)", -6.0, 6.0, value=cur_log, step=0.01)
        st.session_state["cmp_ugL"] = float(10 ** float(log10_cmp))
        st.caption(f"From slider: {st.session_state['cmp_ugL']:.3g} μg/L")

# Definitions expander (separate, not nested)
with st.sidebar.expander("Definitions (R_pred, Φ, log Kd*)", expanded=False):
    st.markdown(
        r"""
        <div class="formula-box">
          <b>Sorbed fraction:</b> R<sub>pred</sub> = (K<sub>MP/W</sub>C<sub>MP</sub>)/(1+K<sub>MP/W</sub>C<sub>MP</sub>)<br/>
          <b>Enrichment index:</b> Φ = log<sub>10</sub>(K<sub>MP/W</sub>C<sub>MP</sub>) = log K<sub>MP/Wd</sub> + log<sub>10</sub>(C<sub>MP</sub>)<br/>
          <b>Threshold line:</b> log K<sub>MP/W</sub><sup>*</sup> = log<sub>10</sub>[ R<sub>ref</sub> / ((1−R<sub>ref</sub>) C<sub>MP</sub>) ]
        </div>
        """,
        unsafe_allow_html=True
    )

# refresh derived values after advanced slider could update session_state
C_MP_ugL = float(st.session_state["cmp_ugL"])
C_MP_gmL = cmp_ugL_to_gmL(C_MP_ugL)
R_ref = float(st.session_state["r_ref"])

# ---------- Tabs ----------
tab_home, tab_single, tab_batch = st.tabs(["Home", "Single", "Batch CSV"])

# ---------- Home ----------
with tab_home:
    st.markdown("""
    <style>
      .hero{
        font-family:'Times New Roman',Times,serif;
        background:#f9fbff;
        border:1px solid #e5ecf5;
        border-radius:28px;
        padding:20px 18px;
        margin:4px 0 16px 0;
        text-align:center;
      }
      .hero h2{
        margin:.2rem 0 .25rem 0;
        font-weight:700;
        font-size: 30px !important;
      }
      .muted{ color:#555; opacity:.9; font-size:18px; }
      .authors{ display:flex; justify-content:center; gap:20px; margin:10px 0 16px 0; flex-wrap:wrap; }
      .author-card{ border:1px solid #e9ecef; border-radius:12px; padding:10px 18px; background:#fff; font-size:16px; }
      .intro-one{ font-family:'Times New Roman',Times,serif; font-size:20px; line-height:1.78; color:#111; text-align:justify; max-width:950px; margin:0 auto; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        "<div class='hero'>"
        "<h2>MP-Insight</h2>"
        f"<div class='muted'>{LABEL_LOGK_HTML} prediction with physics-informed enrichment interpretation.</div>"
        "</div>",
        unsafe_allow_html=True
    )

    # Put What/How at bottom (per your preference): show image + intro first, then What/How
    
    # Put What/How at bottom (per your preference): show image + intro first, then What/How
    hero_img = ROOT / "assets" / "mp_insight_overview.png"

    if hero_img.exists():
        try:
            b = hero_img.read_bytes()
            # 兼容不同 Streamlit 版本：新版本 use_container_width，旧版本 use_column_width
            try:
                st.image(b, use_container_width=True)
            except TypeError:
                st.image(b, use_column_width=True)
        except Exception as e:
            st.error(f"Failed to read/display hero image: {type(e).__name__}: {e}")
    else:
        st.markdown(
            "<div style='height:340px;border:1.5px dashed #cbd5e1;border-radius:12px;"
            "display:flex;align-items:center;justify-content:center;color:#667085;"
            "background:#f8fafc;font-size:14px;margin:14px 0'>"
            "Add overview image at <code>assets/mp_insight_overview.png</code></div>",
            unsafe_allow_html=True
        )

    st.markdown("""
    <div class="intro-one">
    Microplastics (MPs) are increasingly recognized as dynamic carriers of organic contaminants, yet reported sorption behavior varies substantially across polymer types and environmental conditions. MP-Insight is a machine-learning framework for predicting sorption log K<sub>MP/W</sub> by integrating pollutant descriptors (E, S, A, B, V, logKow). Here, E, S, A, B, and V are Abraham solute descriptors that quantify excess molar refraction (E), dipolarity/polarizability (S), hydrogen-bond acidity (A), hydrogen-bond basicity (B), and McGowan characteristic volume (V); these parameters describe intermolecular interaction potential and can be retrieved from the UFZ LSER database. Environmental factors (pH, salinity, temperature, water type) and microplastic attributes (polymer type, density, particle size, ageing status, specific surface area) are also incorporated. Trained on curated literature data and evaluated with rigorous cross-validation, MP-Insight provides consistent predictions across polymers and exposure scenarios. For environmental interpretation, the platform quantifies sorption-driven enrichment using the predicted sorbed fraction (R<sub>pred</sub>) and enrichment index (Φ) under a user-defined microplastic concentration (C<sub>MP</sub>), and optionally visualizes a physical threshold line (log K<sub>MP/W</sub><sup>*</sup>) defined by a target sorbed fraction (R<sub>ref</sub>).
    </div>

    <div class="note-box">
    <b>Scope:</b> Outputs quantify partitioning/enrichment potential (R<sub>pred</sub>, Φ) and are not biological effect or risk endpoints.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<h3 style='font-family:Times New Roman; font-weight:800;'>What it does</h3>", unsafe_allow_html=True)
        st.markdown("""
- Predicts sorption **logK_MP/W** for microplastic–pollutant systems.
- Translates predictions into scenario-based partitioning indicators (**R_pred**, **Φ**) under user-defined **C_MP**.
- Optionally visualizes a **physical threshold** (**logKMP/W*** ) defined by a target sorbed fraction (**R_ref**).
""")
    with col2:
        st.markdown("<h3 style='font-family:Times New Roman; font-weight:800;'>How to use</h3>", unsafe_allow_html=True)
        st.markdown("""
1. Go to **Single** → enter descriptors → click **Predict**.  
2. Set **C_MP** in the sidebar (and optionally **R_ref** for the threshold line).  
3. For batch predictions, open **Batch CSV** and upload the template file.
""")

# ---------- Single ----------
with tab_single:
    st.markdown("### Single prediction")

    uniq = get_training_uniques(MODE)
    defaults = {
        "MPs types": uniq["MPs types"][0] if uniq["MPs types"] else "PE",
        "Ageing_Status": uniq["Ageing_Status"][0] if uniq["Ageing_Status"] else "Pristine",
        "Water Type": uniq["Water Type"][0] if uniq["Water Type"] else "Freshwater",
        "Plastic Density": 0.95, "SSA": 0.0, "Particle Size": 300.0,
        "LogKow": 3.0, "E": 1.0, "S": 1.0, "A": 0.1, "B": 0.3, "V": 1.0,
        "PH": 7.0, "Salinity": 0.0, "Temperature": 25.0
    }
    cols_hint = ["MPs types","Ageing_Status","Water Type",
                 "Plastic Density","SSA","Particle Size","LogKow","E","S","A","B","V",
                 "PH","Salinity","Temperature"]

    inp = {}
    left, right = st.columns(2)
    idx = 0
    for c in cols_hint:
        if MODE == "Unknown polymer" and c == "MPs types":
            continue
        with (left if idx % 2 == 0 else right):
            if c in ["MPs types","Ageing_Status","Water Type"]:
                options = uniq.get(c, [])
                if not options:
                    inp[c] = st.text_input(c, value=str(defaults.get(c, "")))
                else:
                    inp[c] = st.selectbox(c, options=options, index=0, key=f"{c}_sb")
            else:
                inp[c] = st.number_input(c, value=float(defaults.get(c, 0.0)), key=f"{c}_num")
        idx += 1

    if st.button("Predict", type="primary", disabled=(pipe is None)):
        X = pd.DataFrame([inp])
        y_pred = float(pipe.predict(X)[0]) if pipe is not None else float("nan")
        PI_low, PI_high = y_pred - q95, y_pred + q95

        R_pred = R_pred_from_logKd(y_pred, C_MP_gmL)
        Phi = phi_from_logKd(y_pred, C_MP_gmL)
        logKd_star = logKd_star_from_Rref(R_ref, C_MP_gmL) if show_ref_line else None

        font_label = "font-size:17px; font-weight:700; color:#004C8C; font-family:'Times New Roman', serif;"
        font_value = "font-size:26px; font-weight:800; color:#000; font-family:'Times New Roman', serif;"
        box_style  = "background-color:#F3F7FB; border:1.5px solid #C9D7E3; border-radius:10px; padding:14px 18px; text-align:center;"

        # IMPORTANT: use HTML headings to ensure Times
        st.markdown("<h3 style='font-family:Times New Roman; font-weight:800;'>Model output</h3>", unsafe_allow_html=True)
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            st.markdown(f"""
            <div class="kpi" style="{box_style}">
              <div style="{font_label}">Predicted log K<sub>MP/W</sub></div>
              <div style="{font_value}">{y_pred:.2f}</div>
            </div>""", unsafe_allow_html=True)
        with r1c2:
            st.markdown(f"""
            <div class="kpi" style="{box_style}">
              <div style="{font_label}">95% Prediction Interval</div>
              <div style="{font_value}">[{PI_low:.2f}, {PI_high:.2f}]</div>
            </div>""", unsafe_allow_html=True)

        st.markdown(
            "<h3 style='font-family:Times New Roman; font-weight:800;'>"
            "Environmental implication (at specified C<sub>MP</sub>)"
            "</h3>",
            unsafe_allow_html=True
        )
        r2c1, r2c2 = st.columns(2)
        with r2c1:
            st.markdown(f"""
            <div class="kpi" style="{box_style}">
              <div style="{font_label}">Sorbed fraction (R<sub>pred</sub>)</div>
              <div style="{font_value}">{R_pred:.3e}</div>
            </div>""", unsafe_allow_html=True)
        with r2c2:
            st.markdown(f"""
            <div class="kpi" style="{box_style}">
              <div style="{font_label}">Enrichment index (Φ)</div>
              <div style="{font_value}">{Phi:.2f}</div>
            </div>""", unsafe_allow_html=True)

        colA, colB = st.columns(2)
        with colA:
            fig1 = plot_prediction_panel(
                y_pred=y_pred, PI_low=PI_low, PI_high=PI_high,
                show_ref=show_ref_line,
                logKd_star=(logKd_star if logKd_star is not None else 0.0),
                paper=paper_mode
            )
            st.pyplot(fig1, use_container_width=True)

        with colB:
            fig2 = plot_scenario_Rpred_vs_CMP(
                y_pred=y_pred, PI_low=PI_low, PI_high=PI_high,
                C_MP_current_ugL=C_MP_ugL,
                paper=paper_mode
            )
            st.pyplot(fig2, use_container_width=True)

        ref_line_text = ""
        if show_ref_line and logKd_star is not None:
            ref_line_text = (
                f"Threshold line: log K<sub>MP/W</sub><sup>*</sup> = <b>{logKd_star:.2f}</b>,"

                f"derived from mass balance to achieve the target sorbed fraction (R<sub>ref</sub>)."
            )

        st.markdown(
            f"""<div class='note-box'>
            <b>Interpretation (scenario-based).</b><br/>
            At C<sub>MP</sub> = <b>{C_MP_ugL:.3g} μg/L</b>, the predicted sorbed fraction is
            R<sub>pred</sub> = <b>{R_pred:.3e}</b>, corresponding to Φ = <b>{Phi:.2f}</b>.<br/>
            {ref_line_text}
            </div>""",
            unsafe_allow_html=True
        )

        if paper_mode:
            png1, svg1 = io.BytesIO(), io.BytesIO()
            fig1.savefig(png1, format="png", dpi=600, bbox_inches="tight"); png1.seek(0)
            fig1.savefig(svg1, format="svg", bbox_inches="tight"); svg1.seek(0)
            st.download_button("Download Panel A (PNG, 600 dpi)", data=png1, file_name="panel_A_prediction.png", mime="image/png")
            st.download_button("Download Panel A (SVG)", data=svg1.getvalue(), file_name="panel_A_prediction.svg", mime="image/svg+xml")

            png2, svg2 = io.BytesIO(), io.BytesIO()
            fig2.savefig(png2, format="png", dpi=600, bbox_inches="tight"); png2.seek(0)
            fig2.savefig(svg2, format="svg", bbox_inches="tight"); svg2.seek(0)
            st.download_button("Download Panel B (PNG, 600 dpi)", data=png2, file_name="panel_B_Rpred_vs_CMP.png", mime="image/png")
            st.download_button("Download Panel B (SVG)", data=svg2.getvalue(), file_name="panel_B_Rpred_vs_CMP.svg", mime="image/svg+xml")

        if st.toggle("Show explanation (SHAP)"):
            try:
                import shap
                X_pre = pipe.named_steps["pre"].transform(X)
                explainer = shap.TreeExplainer(pipe.named_steps["model"])
                sv = explainer.shap_values(X_pre)
                try:
                    feat_names = list(pipe.named_steps["pre"].get_feature_names_out())
                except Exception:
                    feat_names = [f"f{i}" for i in range(X_pre.shape[1])]
                vals = np.abs(np.array(sv)).ravel()
                order = np.argsort(vals)[::-1][:10]
                df_bar = pd.DataFrame({"Feature": [feat_names[i] for i in order], "SHAP": vals[order]})
                st.bar_chart(df_bar.set_index("Feature")["SHAP"])
            except Exception as e:
                st.warning(f"SHAP explanation unavailable: {e}")

# ---------- Batch CSV ----------
with tab_batch:
    st.markdown("### Batch CSV prediction")

    required_cols = get_required_cols(MODE)
    tpl_bytes = make_template_csv(required_cols)
    st.download_button(
        "Download CSV template",
        data=tpl_bytes,
        file_name=("template_known_polymer.csv" if MODE=="Known polymer" else "template_unknown_polymer.csv"),
        mime="text/csv",
    )

    f = st.file_uploader("Upload CSV", type=["csv"], key="csv_uploader_batch")

    if f is None:
        st.info("Upload a CSV to run batch prediction.")
    else:
        try:
            df_in = pd.read_csv(f)
        except Exception as e:
            st.error(f"Cannot read CSV: {e}")
            st.stop()

        # ✅ 列名标准化（用文件头已有函数）
        df_in = standardize_input_df(df_in)

        # 列检查
        check = validate_csv_columns(df_in, required_cols)
        st.markdown("**Column check**")
        st.dataframe(check["status_df"], use_container_width=True)

        if check["missing"]:
            st.error("Missing required columns:\n\n- " + "\n- ".join(check["missing"]))
            st.stop()

        if check["extras"]:
            st.warning("Extra columns will be ignored:\n\n- " + "\n- ".join(check["extras"]))

        if pipe is None:
            st.error("Model not loaded.")
            st.stop()

        # ✅ 对齐模型输入列（建议保留）
        try:
            expected_cols = list(pipe.feature_names_in_)
            missing_model = set(expected_cols) - set(df_in.columns)
            if missing_model:
                st.error(f"Model expects columns missing after normalization: {sorted(missing_model)}")
                st.stop()
            df_in = df_in[expected_cols]
        except Exception as e:
            st.error(f"Column alignment error: {e}")
            st.stop()

        # 预测
        preds = pipe.predict(df_in)

        out = df_in.copy()
        out["Predicted_logKMP_W"] = preds
        out["PI95_low"]  = out["Predicted_logKMP_W"] - q95
        out["PI95_high"] = out["Predicted_logKMP_W"] + q95
        out["Sorbed_fraction"] = out["Predicted_logKMP_W"].apply(lambda v: R_pred_from_logKd(v, C_MP_gmL))
        out["Enrichment_index"] = out["Predicted_logKMP_W"].apply(lambda v: phi_from_logKd(v, C_MP_gmL))

        st.markdown("**Preview (first 30 rows)**")
        st.dataframe(out.head(30), use_container_width=True)

        st.download_button(
            "Download results CSV",
            data=out.to_csv(index=False).encode("utf-8-sig"),
            file_name="mp_insight_batch_results.csv",
            mime="text/csv",
        )


    f = st.file_uploader("Upload CSV", type=["csv"], key="csv_uploader_batch")
    if f is None:
    st.info("Upload a CSV to run batch prediction.")
    st.stop()

    try:
        df_in = pd.read_csv(f)
    except Exception as e:
        st.error(f"Cannot read CSV: {e}")
        st.stop()

    # ==================================================
    # 🔥 关键新增：列名标准化（对齐训练阶段）
    # ==================================================
    import re

    def _normalize_colname(c):
        c = str(c).strip().replace("\u00a0", " ")
        c = re.sub(r"\s+", " ", c)
        return c

    NAME_MAP = [
        (r"^(plastic[\s_-]?density).*", "Plastic Density"),
        (r"^(ssa).*", "SSA"),
        (r"^(particle[\s_-]?size).*", "Particle Size"),
        (r"^(logk[_\s-]?ow|logkow)$", "LogKow"),
        (r"^e$", "E"), (r"^s$", "S"),
        (r"^a$", "A"), (r"^b$", "B"), (r"^v$", "V"),
        (r"^(ph)$", "PH"),
        (r"^(salinity).*", "Salinity"),
        (r"^(temp.*|temperature.*)", "Temperature"),
        (r"^(mps?[\s_/-]?type[s]?)$", "MPs types"),
        (r"^(age(ing)?[\s_-]?(type|status))$", "Ageing_Status"),
        (r"^(water[\s_-]?type)$", "Water Type"),
    ]

    def _map_columns(cols):
        mapped = {}
        for col in cols:
            low = _normalize_colname(col).lower()
            for patt, std in NAME_MAP:
                if re.match(patt, low):
                    mapped[col] = std
                    break
        return mapped

    df_in.columns = df_in.columns.str.strip()
    mapped = _map_columns(df_in.columns)
    df_in = standardize_input_df(df_in)

    # ==================================================
    # 列检查（标准化后再检查）
    # ==================================================
    check = validate_csv_columns(df_in, required_cols)
    st.markdown("**Column check**")
    st.dataframe(check["status_df"], use_container_width=True)

    if check["missing"]:
        st.error("Missing required columns:\n\n- " + "\n- ".join(check["missing"]))
        st.stop()

    if check["extras"]:
        st.warning("Extra columns will be ignored:\n\n- " + "\n- ".join(check["extras"]))

    if pipe is None:
        st.error("Model not loaded.")
        st.stop()

    # ==================================================
    # 🔥 强制列顺序与模型一致
    # ==================================================
    try:
        expected_cols = list(pipe.feature_names_in_)
        missing_model = set(expected_cols) - set(df_in.columns)
        if missing_model:
            st.error(f"Model expects columns missing after normalization: {missing_model}")
            st.stop()

        df_in = df_in[expected_cols]

    except Exception as e:
        st.error(f"Column alignment error: {e}")
        st.stop()

   # ==================================================
# 预测
# ==================================================
preds = pipe.predict(df_in)

out = df_in.copy()

# 建议顺便把空格去掉，变成更“可发表/可解析”的列名
out["Predicted_logKMP_W"] = preds
out["PI95_low"]  = out["Predicted_logKMP_W"] - q95
out["PI95_high"] = out["Predicted_logKMP_W"] + q95


# physics-informed derived metrics
out["Sorbed_fraction"] = out["Predicted_logKMP_W"].apply(lambda v: R_pred_from_logKd(v, C_MP_gmL))
out["Enrichment_index"] = out["Predicted_logKMP_W"].apply(lambda v: phi_from_logKd(v, C_MP_gmL))

# if show_ref_line:
#     out["Target_sorbed_fraction"] = float(R_ref)
#     out["logKd_threshold"] = float(logKd_star_from_Rref(R_ref, C_MP_gmL))

st.markdown("**Preview (first 30 rows)**")
st.dataframe(out.head(30), use_container_width=True)

st.download_button(
    "Download results CSV",
    data=out.to_csv(index=False).encode("utf-8-sig"),
    file_name="mp_insight_batch_results.csv",
    mime="text/csv"
)
