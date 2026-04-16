import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import joblib
import tensorflow as tf

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Accident Severity Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
FIG_DIR    = os.path.join(MODEL_DIR, "figures")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
LOGO_PATH  = os.path.join(ASSETS_DIR, "logo.png")

# ─────────────────────────────────────────────
# GLOBAL STYLE
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg:             #eef0f4;
    --surface:        #ffffff;
    --surface-2:      #f7f8fa;
    --border:         #e2e5ec;
    --text-h:         #0d1b2a;
    --text-b:         #2d3f55;
    --text-m:         #5a6e87;
    --text-s:         #8fa0b8;
    --accent:         #2563eb;
    --accent-dk:      #1d4ed8;
    --accent-lt:      #eff6ff;
    --fatal:          #c0392b;
    --fatal-bg:       #fff5f5;
    --fatal-border:   #f5c6c6;
    --serious:        #b45309;
    --serious-bg:     #fffbf0;
    --serious-border: #f0d080;
    --slight:         #15803d;
    --slight-bg:      #f0fdf4;
    --slight-border:  #a7f3c0;
    --radius-sm:      6px;
    --radius:         10px;
    --shadow:         0 1px 3px rgba(13,27,42,.07), 0 4px 14px rgba(13,27,42,.05);
    --shadow-md:      0 4px 18px rgba(13,27,42,.10), 0 1px 4px rgba(13,27,42,.06);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    color: var(--text-b);
    -webkit-font-smoothing: antialiased;
}
.stApp { background: var(--bg) !important; }
#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar — always visible ── */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e2e5ec !important;
    box-shadow: 2px 0 10px rgba(13,27,42,.04) !important;
    min-width: 250px !important;
    max-width: 250px !important;
    width: 250px !important;
    transform: none !important;
    visibility: visible !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 1rem 1rem !important;
    width: 250px !important;
}
/* hide the collapse/expand arrow button */
[data-testid="collapsedControl"] {
    display: none !important;
}
button[kind="header"] {
    display: none !important;
}

/* ── Radio nav ── */
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
    gap: 2px !important;
    display: flex !important;
    flex-direction: column !important;
}
[data-testid="stSidebar"] .stRadio label > div:first-child {
    display: none !important;
}
[data-testid="stSidebar"] .stRadio label {
    display: flex !important;
    align-items: center !important;
    width: 100% !important;
    padding: 9px 12px !important;
    border-radius: 6px !important;
    font-size: 0.875rem !important;
    font-weight: 400 !important;
    color: #5a6e87 !important;
    background: transparent !important;
    border: none !important;
    border-left: 3px solid transparent !important;
    margin: 1px 0 !important;
    cursor: pointer !important;
    transition: background .12s, color .12s, border-color .12s;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: #eff6ff !important;
    color: #2563eb !important;
    border-left-color: #2563eb !important;
}
[data-testid="stSidebar"] .stRadio label:has(input:checked) {
    background: #eff6ff !important;
    color: #2563eb !important;
    border-left-color: #2563eb !important;
    font-weight: 600 !important;
}
[data-testid="stSidebar"] .stRadio label p,
[data-testid="stSidebar"] .stRadio label span {
    color: inherit !important;
    font-size: 0.875rem !important;
    font-weight: inherit !important;
}

/* nav active indicator dot */
[data-testid="stSidebar"] .stRadio label:has(input:checked)::after {
    content: "";
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #2563eb;
    margin-left: auto;
    flex-shrink: 0;
}

/* ── Typography ── */
h1 { font-size:1.45rem !important; font-weight:700 !important;
     color:#0d1b2a !important; letter-spacing:-.4px; line-height:1.25 !important; }
h2 { font-size:1.1rem  !important; font-weight:600 !important; color:#0d1b2a !important; }
h3 { font-size:0.68rem !important; font-weight:600 !important; color:#8fa0b8 !important;
     text-transform:uppercase; letter-spacing:1px; margin-bottom:12px !important; }

/* ── Card ── */
.card {
    background: #ffffff;
    border: 1px solid #e2e5ec;
    border-radius: 10px;
    padding: 20px 22px;
    box-shadow: 0 1px 3px rgba(13,27,42,.07), 0 4px 14px rgba(13,27,42,.05);
    margin-bottom: 12px;
    transition: box-shadow .18s;
}
.card:hover {
    box-shadow: 0 4px 18px rgba(13,27,42,.10), 0 1px 4px rgba(13,27,42,.06);
}

/* ── KPI ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 10px;
    margin: 14px 0 18px;
}
.kpi {
    background: #ffffff;
    border: 1px solid #e2e5ec;
    border-radius: 10px;
    padding: 14px 16px 12px;
    box-shadow: 0 1px 3px rgba(13,27,42,.07);
    position: relative;
    overflow: hidden;
    transition: box-shadow .18s, transform .15s;
}
.kpi:hover {
    box-shadow: 0 4px 18px rgba(13,27,42,.10);
    transform: translateY(-1px);
}
.kpi::before {
    content: '';
    position: absolute; top:0; left:0; right:0; height:3px;
    background: var(--kpi-color, #2563eb);
}
.kpi .num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem; font-weight:500; line-height:1.1;
    color: #0d1b2a; margin-top:4px;
}
.kpi .lbl {
    font-size:0.67rem; font-weight:600; text-transform:uppercase;
    letter-spacing:.8px; color:#8fa0b8; margin-top:5px;
}

/* ── Module rows ── */
.mod-row {
    display:flex; align-items:center; gap:12px;
    padding:9px 0; border-bottom:1px solid #e2e5ec;
}
.mod-row:last-child { border-bottom:none; padding-bottom:0; }
.mod-icon {
    width:32px; height:32px; border-radius:8px;
    background: #eff6ff;
    display:flex; align-items:center; justify-content:center;
    flex-shrink:0;
}
.mod-title { font-weight:600; color:#0d1b2a; font-size:.83rem; }
.mod-desc  { font-size:.75rem; color:#5a6e87; margin-top:1px; }

/* ── Tags ── */
.tag {
    display:inline-block; padding:3px 10px; border-radius:99px;
    font-size:.71rem; font-weight:500; margin:3px 2px;
}
.tag-blue { background:#eff6ff; color:#2563eb; border:1px solid #bfdbfe; }
.tag-gray { background:#f1f5f9; color:#475569; border:1px solid #e2e8f0; }

/* ── Team grid ── */
.team-grid { display:grid; grid-template-columns:1fr 1fr; gap:7px; margin-top:6px; }
.team-pill {
    font-size:.78rem; color:#2d3f55; padding:7px 10px;
    background:#f7f8fa; border:1px solid #e2e5ec;
    border-radius:6px;
    display:flex; align-items:center; gap:8px;
}
.team-avatar {
    width:24px; height:24px; border-radius:50%;
    background: linear-gradient(135deg,#2563eb,#60a5fa);
    display:flex; align-items:center; justify-content:center;
    font-size:.6rem; font-weight:700; color:white; flex-shrink:0;
}

/* ── Page header ── */
.page-header {
    padding-bottom:14px; margin-bottom:18px;
    border-bottom:1px solid #e2e5ec;
}
.page-header h1 { margin:0 0 3px !important; }
.page-header p  { color:#5a6e87; font-size:.85rem; margin:0; }

/* ── Sec label ── */
.sec-label {
    display:block; font-size:.65rem; font-weight:600;
    text-transform:uppercase; letter-spacing:1.1px;
    color:#8fa0b8; margin-bottom:10px;
}

/* ── Input group label ── */
.ig-label {
    font-size:.72rem; font-weight:600; color:#2d3f55;
    text-transform:uppercase; letter-spacing:.5px;
    margin-bottom:8px; padding-bottom:6px;
    border-bottom:1px solid #e2e5ec;
}

/* ── Button ── */
.stButton > button {
    background: #2563eb !important; color:#fff !important;
    border:none !important; border-radius:6px !important;
    padding:10px 24px !important;
    font-family:'Inter',sans-serif !important; font-size:.875rem !important;
    font-weight:500 !important;
    box-shadow:0 1px 2px rgba(37,99,235,.3),0 2px 8px rgba(37,99,235,.18) !important;
    transition:background .15s, transform .1s !important;
}
.stButton > button:hover {
    background: #1d4ed8 !important;
    transform:translateY(-1px) !important;
}

/* ── Prediction box ── */
.pred-box { border-radius:10px; padding:28px 20px; text-align:center; margin-top:14px; }
.pred-fatal   { background:#fff5f5; border:1px solid #f5c6c6; }
.pred-serious { background:#fffbf0; border:1px solid #f0d080; }
.pred-slight  { background:#f0fdf4; border:1px solid #a7f3c0; }
.pred-eyebrow { font-size:.65rem; text-transform:uppercase; letter-spacing:1.2px;
                color:#8fa0b8; margin-bottom:10px; font-weight:600; }
.pred-label   { font-family:'JetBrains Mono',monospace; font-size:2rem; font-weight:500; }
.pred-model   { font-size:.73rem; color:#5a6e87; margin-top:7px; }

/* ── Notice ── */
.notice {
    background:#f7f8fa; border:1px solid #e2e5ec;
    border-left:3px solid #2563eb; border-radius:6px;
    padding:10px 14px; font-size:.8rem; color:#5a6e87;
}

/* ── Divider ── */
.divider { border:none; border-top:1px solid #e2e5ec; margin:14px 0; }

label { font-size:.82rem !important; color:#5a6e87 !important; font-weight:500 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MATPLOTLIB
# ─────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#e2e5ec",
    "axes.linewidth":    0.8,
    "axes.labelcolor":   "#5a6e87",
    "axes.labelsize":    9,
    "axes.labelpad":     5,
    "axes.titlesize":    10,
    "axes.titleweight":  "600",
    "axes.titlecolor":   "#0d1b2a",
    "axes.titlepad":     10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.color":       "#8fa0b8",
    "xtick.labelsize":   8,
    "xtick.major.pad":   4,
    "ytick.color":       "#8fa0b8",
    "ytick.labelsize":   8,
    "ytick.major.pad":   4,
    "grid.color":        "#edf0f5",
    "grid.linewidth":    0.8,
    "text.color":        "#2d3f55",
    "legend.fontsize":   8,
    "legend.framealpha": 1.0,
    "legend.edgecolor":  "#e2e5ec",
    "font.family":       "sans-serif",
    "figure.dpi":        120,
})

SEV_COLORS  = {1: "#c0392b", 2: "#b45309", 3: "#15803d"}
SEV_LABELS  = {1: "Fatal",   2: "Serious", 3: "Slight"}
SEV_HANDLES = [
    mpatches.Patch(color="#c0392b", label="Fatal"),
    mpatches.Patch(color="#b45309", label="Serious"),
    mpatches.Patch(color="#15803d", label="Slight"),
]

# ─────────────────────────────────────────────
# SVG ICON PATHS (Feather style)
# ─────────────────────────────────────────────
ICON = {
    "bar-chart":   '<line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>',
    "layers":      '<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>',
    "cpu":         '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/>',
    "eye":         '<path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/>',
    "activity":    '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
    "shield":      '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>',
    "trending-up": '<polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/>',
    "database":    '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>',
    "home":        '<path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/>',
    "file-text":   '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/>',
}

def svg_icon(key, size=15, color="#2563eb", stroke="2.2"):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" '
        f'stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round">'
        f'{ICON[key]}</svg>'
    )

def mod_icon_html(key):
    return f'<div class="mod-icon">{svg_icon(key, 15, "#2563eb")}</div>'

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
df_path = os.path.join(DATA_DIR, "merged_5year_dataset.csv")
if os.path.exists(df_path):
    df = pd.read_csv(df_path)
else:
    rng = np.random.default_rng(42)
    n   = 800
    df  = pd.DataFrame({
        "collision_severity":   rng.choice([1,2,3], n, p=[0.05,0.25,0.70]),
        "weather_conditions":   rng.choice(["Fine","Rain","Snow","Fog","High winds"], n),
        "road_type":            rng.choice(["Single carriageway","Dual carriageway","Roundabout","One way"], n),
        "light_conditions":     rng.choice(["Daylight","Dark – lit","Dark – unlit","Dusk/Dawn"], n),
        "junction_detail":      rng.choice(["Not at junction","T junction","Crossroads","Slip road"], n),
        "speed_limit":          rng.choice([20,30,40,50,60,70], n),
        "day_of_week":          rng.choice(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"], n),
        "number_of_vehicles":   rng.integers(1,5, n),
        "number_of_casualties": rng.integers(0,4, n),
    })

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
def _try(fn):
    try:    return fn()
    except: return None

lr           = _try(lambda: joblib.load(os.path.join(MODEL_DIR,"logistic_regression.pkl")))
rf           = _try(lambda: joblib.load(os.path.join(MODEL_DIR,"random_forest.pkl")))
xgb          = _try(lambda: joblib.load(os.path.join(MODEL_DIR,"xgboost.pkl")))
dl_model     = _try(lambda: tf.keras.models.load_model(os.path.join(MODEL_DIR,"dl_model.keras")))
preprocessor = _try(lambda: joblib.load(os.path.join(DATA_DIR,"preprocessor.pkl")))
svd          = _try(lambda: joblib.load(os.path.join(DATA_DIR,"svd_transformer.pkl")))
scaler_svd   = _try(lambda: joblib.load(os.path.join(DATA_DIR,"svd_scaler.pkl")))

# ─────────────────────────────────────────────
# SIDEBAR  — using st.sidebar natively
# ─────────────────────────────────────────────
with st.sidebar:
    # Brand header
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;padding:4px 0 16px;">
        <div style="width:36px;height:36px;border-radius:9px;background:#2563eb;
                    display:flex;align-items:center;justify-content:center;flex-shrink:0;">
            {svg_icon("shield", 18, "white", "2.2")}
        </div>
        <div>
            <div style="font-size:.88rem;font-weight:700;color:#0d1b2a;line-height:1.2;">
                Accident Severity
            </div>
            <div style="font-size:.62rem;color:#8fa0b8;text-transform:uppercase;
                        letter-spacing:.8px;margin-top:1px;">
                Analytics Platform
            </div>
        </div>
    </div>
    <div style="border-top:1px solid #e2e5ec;margin-bottom:14px;"></div>
    """, unsafe_allow_html=True)

    # Navigation label
    st.markdown(
        "<p style='font-size:.63rem;font-weight:600;text-transform:uppercase;"
        "letter-spacing:1px;color:#8fa0b8;margin:0 0 6px 2px;'>Navigation</p>",
        unsafe_allow_html=True
    )

    # Radio — Streamlit native (always visible, always works)
    page = st.radio(
        label="nav",
        options=[
            "Home",
            "Explore Data",
            "Model Comparison",
            "Predict Severity",
            "Explainability",
        ],
        label_visibility="collapsed",
    )
    # Footer
    st.markdown("""
    <div style="border-top:1px solid #e2e5ec;margin-top:28px;padding-top:14px;">
        <p style="font-size:.68rem;color:#b0bec5;line-height:1.7;margin:0;">
            MSc Data Analytics<br>
            National College of Ireland<br>
            Deep Learning &amp; Generative AI
        </p>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def make_fig(w=5.6, h=3.5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.12, right=0.97, top=0.91, bottom=0.20)
    return fig, ax

def show(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

def card_open(title=None):
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    if title:
        st.markdown(f"<h3>{title}</h3>", unsafe_allow_html=True)

def card_close():
    st.markdown("</div>", unsafe_allow_html=True)

def page_header(title, sub=""):
    sub_html = f'<p>{sub}</p>' if sub else ""
    st.markdown(f"""
    <div class="page-header">
        <h1>{title}</h1>{sub_html}
    </div>""", unsafe_allow_html=True)

def miss(fname):
    st.markdown(
        f"<div class='notice'><code>{fname}</code> not found — "
        f"add it to render this section.</div>",
        unsafe_allow_html=True,
    )

def leg(ax, loc="upper right"):
    ax.legend(handles=SEV_HANDLES, title="Severity", title_fontsize=7,
              loc=loc, frameon=True, fancybox=False,
              edgecolor="#e2e5ec", facecolor="white")


# ═══════════════════════════════════════════════════════════
# HOME
# ═══════════════════════════════════════════════════════════
if page == "Home":

    total   = len(df)
    fatal   = int((df["collision_severity"]==1).sum()) if "collision_severity" in df else 0
    serious = int((df["collision_severity"]==2).sum()) if "collision_severity" in df else 0
    slight  = int((df["collision_severity"]==3).sum()) if "collision_severity" in df else 0

    # ── Compact header ──
    st.markdown(f"""
    <div style="display:flex;align-items:center;justify-content:space-between;
                padding-bottom:14px;margin-bottom:14px;border-bottom:1px solid #e2e5ec;">
        <div>
            <div style="font-size:.63rem;font-weight:600;text-transform:uppercase;
                        letter-spacing:1.1px;color:#2563eb;margin-bottom:4px;">
                MSc Data Analytics &nbsp;·&nbsp; National College of Ireland
            </div>
            <h1 style="margin:0 0 2px;font-size:1.4rem;font-weight:700;color:#0d1b2a;">
                Accident Severity Intelligence Dashboard
            </h1>
            <p style="margin:0;font-size:.82rem;color:#5a6e87;">
                Deep Learning &amp; Generative AI &nbsp;·&nbsp; UK 5-Year Collision Dataset
            </p>
        </div>
        <div style="opacity:.5;">
            {svg_icon("activity", 28, "#2563eb", "1.8")}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI strip ──
    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi" style="--kpi-color:#2563eb;">
            <div class="lbl">Total Records</div>
            <div class="num">{total:,}</div>
        </div>
        <div class="kpi" style="--kpi-color:#c0392b;">
            <div class="lbl">Fatal</div>
            <div class="num" style="color:#c0392b;">{fatal:,}</div>
        </div>
        <div class="kpi" style="--kpi-color:#b45309;">
            <div class="lbl">Serious</div>
            <div class="num" style="color:#b45309;">{serious:,}</div>
        </div>
        <div class="kpi" style="--kpi-color:#15803d;">
            <div class="lbl">Slight</div>
            <div class="num" style="color:#15803d;">{slight:,}</div>
        </div>
        <div class="kpi" style="--kpi-color:#7c3aed;">
            <div class="lbl">Models Trained</div>
            <div class="num" style="color:#7c3aed;">4</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 3-column body ──
    c1, c2, c3 = st.columns([1.1, 1, 1], gap="medium")

    with c1:
        st.markdown(f"""
        <div class="card" style="margin-bottom:0;">
            <span class="sec-label">Project Overview</span>
            <p style="line-height:1.75;font-size:.83rem;color:#2d3f55;margin:0 0 14px;">
                Road accidents are a critical public safety challenge. This system predicts
                <strong style="color:#0d1b2a;">collision severity</strong> using deep learning,
                classical ML, and SHAP explainability — supporting data-driven decisions for
                road-safety authorities and emergency response teams.
            </p>
            <div style="border-top:1px solid #e2e5ec;padding-top:12px;">
                <span class="sec-label">Team Members</span>
                <div class="team-grid">
                    <div class="team-pill">
                        <div class="team-avatar">KS</div>
                        <span>Keerthika Santhanakrishnan</span>
                    </div>
                    <div class="team-pill">
                        <div class="team-avatar">KD</div>
                        <span>Kishorekumar Dhanabalan</span>
                    </div>
                    <div class="team-pill">
                        <div class="team-avatar">KP</div>
                        <span>Kabilan Ponnusamy</span>
                    </div>
                    <div class="team-pill">
                        <div class="team-avatar">AZ</div>
                        <span>Abrarudin Zahirudhin</span>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="card" style="margin-bottom:0;">
            <span class="sec-label">Dashboard Modules</span>
            <div class="mod-row">
                {mod_icon_html("bar-chart")}
                <div>
                    <div class="mod-title">Explore Data</div>
                    <div class="mod-desc">Patterns in UK 5-year collision data</div>
                </div>
            </div>
            <div class="mod-row">
                {mod_icon_html("layers")}
                <div>
                    <div class="mod-title">Model Comparison</div>
                    <div class="mod-desc">ML vs DL metrics &amp; confusion matrices</div>
                </div>
            </div>
            <div class="mod-row">
                {mod_icon_html("cpu")}
                <div>
                    <div class="mod-title">Predict Severity</div>
                    <div class="mod-desc">Real-time prediction, four models</div>
                </div>
            </div>
            <div class="mod-row">
                {mod_icon_html("eye")}
                <div>
                    <div class="mod-title">Explainability</div>
                    <div class="mod-desc">SHAP force &amp; waterfall plots</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="card" style="margin-bottom:0;">
            <span class="sec-label">Technologies</span>
            <div style="margin-bottom:14px;">
                <span class="tag tag-blue">TensorFlow / Keras</span>
                <span class="tag tag-blue">XGBoost</span>
                <span class="tag tag-blue">Scikit-Learn</span>
                <span class="tag tag-gray">Pandas &amp; NumPy</span>
                <span class="tag tag-gray">SHAP</span>
                <span class="tag tag-gray">Streamlit</span>
                <span class="tag tag-gray">Matplotlib</span>
            </div>
            <div style="border-top:1px solid #e2e5ec;padding-top:12px;">
                <span class="sec-label">Methodology</span>
                <div style="display:flex;flex-direction:column;gap:8px;margin-top:4px;">
                    <div style="display:flex;align-items:center;gap:8px;font-size:.78rem;color:#2d3f55;">
                        {svg_icon("trending-up", 13, "#2563eb")}
                        CRISP-DM aligned process
                    </div>
                    <div style="display:flex;align-items:center;gap:8px;font-size:.78rem;color:#2d3f55;">
                        {svg_icon("database", 13, "#2563eb")}
                        UK STATS19 collision records
                    </div>
                    <div style="display:flex;align-items:center;gap:8px;font-size:.78rem;color:#2d3f55;">
                        {svg_icon("shield", 13, "#2563eb")}
                        3-class severity classification
                    </div>
                    <div style="display:flex;align-items:center;gap:8px;font-size:.78rem;color:#2d3f55;">
                        {svg_icon("eye", 13, "#2563eb")}
                        SHAP-based explainability
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# EXPLORE DATA
# ═══════════════════════════════════════════════════════════
elif page == "Explore Data":
    page_header("Explore Dataset",
                "Key accident patterns across the UK 5-year collision dataset")

    def count_chart(col, title):
        fig, ax = make_fig()
        sns.countplot(data=df, x=col, hue="collision_severity",
                      palette=SEV_COLORS, ax=ax, zorder=3, width=0.56, linewidth=0)
        ax.yaxis.grid(True, zorder=0, linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)
        plt.xticks(rotation=28, ha="right", fontsize=8)
        ax.set_xlabel(""); ax.set_ylabel("Count")
        ax.set_title(title, pad=10)
        leg(ax)
        return fig

    r1, r2 = st.columns(2, gap="medium")
    with r1:
        card_open()
        fig, ax = make_fig()
        counts = df["collision_severity"].value_counts().sort_index()
        bars   = ax.bar([SEV_LABELS[k] for k in counts.index],
                        counts.values, color=[SEV_COLORS[k] for k in counts.index],
                        width=0.42, zorder=3, linewidth=0)
        ax.bar_label(bars, fmt="%d", padding=4, fontsize=8.5, color="#2d3f55", fontweight="500")
        ax.yaxis.grid(True, zorder=0, linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel("Collision Severity"); ax.set_ylabel("Count")
        ax.set_title("Severity Distribution", pad=10)
        show(fig); card_close()

    with r2:
        card_open()
        show(count_chart("weather_conditions", "Weather Conditions vs Severity"))
        card_close()

    r3, r4 = st.columns(2, gap="medium")
    with r3:
        card_open()
        show(count_chart("road_type", "Road Type vs Severity"))
        card_close()
    with r4:
        card_open()
        show(count_chart("light_conditions", "Light Conditions vs Severity"))
        card_close()

    r5, r6 = st.columns(2, gap="medium")
    with r5:
        card_open()
        show(count_chart("junction_detail", "Junction Detail vs Severity"))
        card_close()
    with r6:
        card_open()
        fig, ax = make_fig()
        ax.hist(df["speed_limit"].dropna(), bins=12, color="#2563eb",
                alpha=0.72, edgecolor="white", linewidth=0.8, zorder=3)
        ax.yaxis.grid(True, zorder=0, linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel("Speed Limit (mph)"); ax.set_ylabel("Count")
        ax.set_title("Speed Limit Distribution", pad=10)
        try:
            from scipy.stats import gaussian_kde
            kd  = gaussian_kde(df["speed_limit"].dropna())
            xs  = np.linspace(df["speed_limit"].min(), df["speed_limit"].max(), 300)
            ax2 = ax.twinx()
            ax2.plot(xs, kd(xs), color="#b45309", lw=1.8, linestyle="--", alpha=0.85)
            ax2.set_ylabel("Density", color="#b45309", fontsize=8)
            ax2.tick_params(colors="#b45309", labelsize=7)
            for sp in ax2.spines.values(): sp.set_visible(False)
        except Exception:
            pass
        show(fig); card_close()

    card_open("Correlation Heatmap — Numeric Features")
    num_df = df.select_dtypes(include=["int64","float64"])
    if num_df.shape[1] > 1:
        from matplotlib.colors import LinearSegmentedColormap
        fig, ax = plt.subplots(figsize=(10, 4.2))
        fig.patch.set_facecolor("white")
        fig.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.18)
        cmap = LinearSegmentedColormap.from_list("rb", ["#c0392b","#f8fafc","#2563eb"])
        sns.heatmap(num_df.corr(), cmap=cmap, center=0,
                    annot=True, fmt=".2f", linewidths=0.5, linecolor="#edf0f5",
                    annot_kws={"size":8,"color":"#2d3f55"}, ax=ax,
                    cbar_kws={"shrink":.65,"pad":.02})
        ax.tick_params(labelsize=8)
        show(fig)
    else:
        st.info("Not enough numeric columns.")
    card_close()


# ═══════════════════════════════════════════════════════════
# MODEL COMPARISON
# ═══════════════════════════════════════════════════════════
elif page == "Model Comparison":
    page_header("Model Comparison",
                "Logistic Regression · Random Forest · XGBoost · Deep Learning")

    card_open("Accuracy Comparison")
    comp_path = os.path.join(MODEL_DIR,"ml_vs_dl_comparison.csv")
    if os.path.exists(comp_path):
        cdf = pd.read_csv(comp_path)
        st.dataframe(
            cdf.style.highlight_max(axis=0, color="#eff6ff")
                     .set_properties(**{"font-size":"13px"}),
            use_container_width=True, hide_index=True,
        )
    else:
        stub = pd.DataFrame({
            "Model":      ["Logistic Regression","Random Forest","XGBoost","Deep Learning"],
            "Accuracy":   ["—","—","—","—"],
            "F1 (macro)": ["—","—","—","—"],
            "AUC-ROC":    ["—","—","—","—"],
        })
        st.dataframe(stub, use_container_width=True, hide_index=True)
        st.markdown(
            "<p style='font-size:.72rem;color:#8fa0b8;margin-top:4px;'>"
            "Place <code>ml_vs_dl_comparison.csv</code> in <code>models/</code>.</p>",
            unsafe_allow_html=True,
        )
    card_close()

    card_open("Training Curves — Deep Learning")
    tc1, tc2 = st.columns(2, gap="medium")
    with tc1:
        p = os.path.join(FIG_DIR,"training_accuracy_curve.png")
        if os.path.exists(p):
            st.image(p, caption="Training vs Validation Accuracy", use_container_width=True)
        else:
            miss("training_accuracy_curve.png")
    with tc2:
        p = os.path.join(FIG_DIR,"training_loss_curve.png")
        if os.path.exists(p):
            st.image(p, caption="Training vs Validation Loss", use_container_width=True)
        else:
            miss("training_loss_curve.png")
    card_close()

    card_open("Confusion Matrix — Deep Learning Model")
    cm_p = os.path.join(FIG_DIR,"confusion_matrix_dl.png")
    if os.path.exists(cm_p):
        _, cc, _ = st.columns([1,2,1])
        with cc:
            st.image(cm_p, use_container_width=True)
    else:
        miss("confusion_matrix_dl.png")
    card_close()

    card_open("SHAP Global Explainability")
    sc1, sc2 = st.columns(2, gap="medium")
    with sc1:
        p = os.path.join(FIG_DIR,"shap_summary.png")
        if os.path.exists(p):
            st.image(p, caption="SHAP Summary Plot", use_container_width=True)
        else:
            miss("shap_summary.png")
    with sc2:
        p = os.path.join(FIG_DIR,"shap_bar.png")
        if os.path.exists(p):
            st.image(p, caption="SHAP Feature Importance", use_container_width=True)
        else:
            miss("shap_bar.png")
    card_close()

    card_open("Interpretation")
    i1, i2 = st.columns(2, gap="large")
    with i1:
        st.markdown("""
        <p style="font-size:.875rem;line-height:1.85;color:#2d3f55;">
            <strong style="color:#0d1b2a;">Deep Learning</strong> captures complex nonlinear
            relationships between features, potentially improving performance over classical
            ML models on this multi-class task.<br><br>
            <strong style="color:#0d1b2a;">XGBoost</strong> performs strongly on tabular data
            and may rival or exceed DL depending on feature richness.
        </p>""", unsafe_allow_html=True)
    with i2:
        st.markdown("""
        <p style="font-size:.875rem;line-height:1.85;color:#2d3f55;">
            <strong style="color:#0d1b2a;">SHAP global plots</strong> identify the most influential
            features across all predictions, supporting transparency and stakeholder trust.<br><br>
            <strong style="color:#0d1b2a;">Confusion matrices</strong> expose class-specific
            weaknesses, especially for imbalanced classes (Fatal vs Serious vs Slight).
        </p>""", unsafe_allow_html=True)
    card_close()


# ═══════════════════════════════════════════════════════════
# PREDICT SEVERITY
# ═══════════════════════════════════════════════════════════
elif page == "Predict Severity":
    page_header("Predict Collision Severity",
                "Configure road conditions and choose a model for real-time severity prediction")

    card_open()
    col1, col2, col3 = st.columns(3, gap="large")
    with col1:
        st.markdown("<div class='ig-label'>Time &amp; Environment</div>", unsafe_allow_html=True)
        day     = st.selectbox("Day of Week",        sorted(df["day_of_week"].dropna().unique()))
        weather = st.selectbox("Weather Conditions", sorted(df["weather_conditions"].dropna().unique()))
    with col2:
        st.markdown("<div class='ig-label'>Road Characteristics</div>", unsafe_allow_html=True)
        road_type = st.selectbox("Road Type",       sorted(df["road_type"].dropna().unique()))
        junction  = st.selectbox("Junction Detail", sorted(df["junction_detail"].dropna().unique()))
    with col3:
        st.markdown("<div class='ig-label'>Other Factors</div>", unsafe_allow_html=True)
        speed_limit = st.slider("Speed Limit (mph)",
                                int(df["speed_limit"].min()),
                                int(df["speed_limit"].max()), 30)
        light = st.selectbox("Light Conditions", sorted(df["light_conditions"].dropna().unique()))
    card_close()

    m1, m2 = st.columns([2,1], gap="medium")
    with m1:
        model_choice = st.selectbox("Model",
            ["Deep Learning","XGBoost","Random Forest","Logistic Regression"])
    with m2:
        st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)
        run = st.button("Run Prediction", use_container_width=True)

    if run:
        input_df = pd.DataFrame([{
            "day_of_week": day, "weather_conditions": weather,
            "road_type": road_type, "speed_limit": speed_limit,
            "light_conditions": light, "junction_detail": junction,
        }])
        pred = None
        if preprocessor is None:
            pred = 3
        else:
            try:
                X = preprocessor.transform(input_df)
                if   model_choice == "Deep Learning"       and dl_model:
                    Xs = svd.transform(X); Xs = scaler_svd.transform(Xs)
                    pred = int(np.argmax(dl_model.predict(Xs)))
                elif model_choice == "XGBoost"             and xgb: pred = int(xgb.predict(X)[0])
                elif model_choice == "Random Forest"        and rf:  pred = int(rf.predict(X)[0])
                elif model_choice == "Logistic Regression" and lr:   pred = int(lr.predict(X)[0])
            except Exception as e:
                st.error(f"Prediction error: {e}")

        if pred is not None:
            cfg = {
                1: ("Fatal",   "pred-fatal",   "#c0392b"),
                2: ("Serious", "pred-serious", "#b45309"),
                3: ("Slight",  "pred-slight",  "#15803d"),
            }
            label, cls, col = cfg.get(pred, (f"Class {pred}","pred-slight","#15803d"))
            _, rc, _ = st.columns([1,1,1])
            with rc:
                st.markdown(f"""
                <div class="pred-box {cls}">
                    <div class="pred-eyebrow">Predicted Severity</div>
                    <div class="pred-label" style="color:{col};">{label}</div>
                    <div class="pred-model">{model_choice}</div>
                </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# EXPLAINABILITY
# ═══════════════════════════════════════════════════════════
elif page == "Explainability":
    page_header("Explainability",
                "SHAP-based local feature attribution for individual predictions — XGBoost")

    x_test_path = os.path.join(DATA_DIR,"X_test_processed.pkl")
    if not os.path.exists(x_test_path):
        st.markdown(f"""
        <div class="card" style="text-align:center;padding:52px 24px;">
            <div style="margin-bottom:12px;opacity:.4;">
                {svg_icon("file-text", 36, "#8fa0b8", "1.4")}
            </div>
            <p style="font-weight:600;color:#2d3f55;margin:0 0 5px;">
                X_test_processed.pkl not found
            </p>
            <p style="font-size:.8rem;color:#8fa0b8;margin:0;">
                Place the file in <code>data/</code> and refresh the page.
            </p>
        </div>""", unsafe_allow_html=True)
    else:
        X_test = joblib.load(x_test_path)
        n_rows = X_test.shape[0]

        card_open("Select Test Record")
        row = st.slider("Row index", 0, n_rows-1, 0)
        card_close()

        row_dense     = X_test[row].toarray()[0]
        feature_names = preprocessor.get_feature_names_out()

        with st.spinner("Computing SHAP values…"):
            explainer   = shap.TreeExplainer(xgb)
            shap_values = explainer.shap_values(X_test)

        ec1, ec2 = st.columns(2, gap="medium")
        with ec1:
            card_open("Force Plot")
            shap.force_plot(explainer.expected_value, shap_values[row],
                            row_dense, feature_names=feature_names,
                            matplotlib=True, show=False)
            plt.savefig("force_local.png", dpi=180, bbox_inches="tight", facecolor="white")
            plt.close()
            st.image("force_local.png", use_container_width=True)
            card_close()

        with ec2:
            card_open("Waterfall Plot")
            shap.plots._waterfall.waterfall_legacy(
                explainer.expected_value, shap_values[row],
                feature_names=feature_names, show=False)
            plt.savefig("waterfall_local.png", dpi=180, bbox_inches="tight", facecolor="white")
            plt.close()
            st.image("waterfall_local.png", use_container_width=True)
            card_close()

        pred      = int(xgb.predict(X_test[row])[0])
        sev_label = {1:"Fatal",2:"Serious",3:"Slight"}.get(pred, f"Class {pred}")
        sev_col   = {"Fatal":"#c0392b","Serious":"#b45309","Slight":"#15803d"}.get(sev_label,"#2563eb")

        card_open("Narrative Explanation")
        st.markdown(f"""
        <p style="font-size:.9rem;line-height:1.9;color:#2d3f55;margin:0;">
            The XGBoost model predicts a
            <strong style="color:{sev_col};">{sev_label}</strong> collision
            for record <strong style="color:#0d1b2a;">#{row}</strong>.
            The SHAP plots above show which features drove the prediction above or below the
            model's baseline expected value — features pushing severity up are red, those
            reducing it are blue. Focus on the highest-magnitude features to understand
            the primary drivers for this specific case.
        </p>""", unsafe_allow_html=True)
        card_close()