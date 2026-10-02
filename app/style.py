"""Visual identity for the ChurnRadar dashboard: CSS, palette and Plotly styling."""
import plotly.graph_objects as go

VIOLET = "#8B5CF6"
CYAN = "#22D3EE"
DANGER = "#FF4D6D"
WARN = "#FFB020"
SAFE = "#2EE59D"
MUTED = "#8A93B2"
RISK_COLORS = {"High": DANGER, "Medium": WARN, "Low": SAFE}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"], .stMarkdown, .stText { font-family: 'Inter', sans-serif; }
h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.02em; }

.stApp {
  background:
    radial-gradient(900px 500px at 8% -10%, rgba(139,92,246,0.22), transparent 60%),
    radial-gradient(800px 500px at 100% 0%, rgba(34,211,238,0.14), transparent 60%),
    radial-gradient(700px 600px at 50% 120%, rgba(255,77,109,0.10), transparent 60%),
    #070B16;
}
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2rem; max-width: 1320px; }

/* ---------- hero ---------- */
.hero { display:flex; align-items:center; justify-content:space-between; gap:1rem; flex-wrap:wrap; margin-bottom: 0.6rem; }
.hero-title { font-family:'Space Grotesk',sans-serif; font-size: 3rem; font-weight:700; line-height:1;
  background: linear-gradient(90deg, #fff 0%, #C4B5FD 40%, #22D3EE 100%);
  -webkit-background-clip:text; background-clip:text; color:transparent; margin:0; }
.hero-sub { color:#8A93B2; font-size:1.02rem; margin-top:.45rem; }
.radar { width:58px; height:58px; border-radius:50%; position:relative; flex:none;
  background: radial-gradient(circle, rgba(34,211,238,.25) 0 2px, transparent 3px),
              repeating-radial-gradient(circle, transparent 0 9px, rgba(34,211,238,.25) 10px 11px);
  border:1px solid rgba(34,211,238,.4); overflow:hidden; }
.radar::after { content:""; position:absolute; inset:0; border-radius:50%;
  background: conic-gradient(from 0deg, rgba(34,211,238,.55), transparent 25%);
  animation: sweep 2.8s linear infinite; }
@keyframes sweep { to { transform: rotate(360deg); } }
.status-pill { display:inline-flex; align-items:center; gap:.5rem; padding:.45rem .9rem; border-radius:999px;
  background: rgba(46,229,157,.08); border:1px solid rgba(46,229,157,.3); color:#BFF8E0; font-size:.85rem; }
.dot { width:8px; height:8px; border-radius:50%; background:#2EE59D; box-shadow:0 0 0 0 rgba(46,229,157,.7);
  animation: pulse 1.8s infinite; }
@keyframes pulse { 70% { box-shadow:0 0 0 10px rgba(46,229,157,0); } 100% { box-shadow:0 0 0 0 rgba(46,229,157,0); } }

/* ---------- glass containers ---------- */
[data-testid="stVerticalBlockBorderWrapper"] {
  background: linear-gradient(160deg, rgba(255,255,255,0.05), rgba(255,255,255,0.015));
  border: 1px solid rgba(255,255,255,0.08) !important; border-radius: 20px !important;
  backdrop-filter: blur(14px); box-shadow: 0 10px 40px -20px rgba(0,0,0,.6);
}
.section-label { font-family:'Space Grotesk',sans-serif; font-size:.78rem; letter-spacing:.14em;
  text-transform:uppercase; color:#A78BFA; margin-bottom:.2rem; }

/* ---------- tabs as pills ---------- */
.stTabs [data-baseweb="tab-list"] { gap:.4rem; background: rgba(255,255,255,.03); padding:.35rem;
  border-radius: 999px; border:1px solid rgba(255,255,255,.07); width: fit-content; }
.stTabs [data-baseweb="tab"] { border-radius:999px; padding:.45rem 1.1rem; height:auto; }
.stTabs [aria-selected="true"] { background: linear-gradient(90deg, rgba(139,92,246,.35), rgba(34,211,238,.25)); }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none; }

/* ---------- KPI cards ---------- */
.kpi { border-radius:18px; padding:1.05rem 1.2rem; position:relative; overflow:hidden;
  background: linear-gradient(160deg, rgba(255,255,255,0.06), rgba(255,255,255,0.015));
  border:1px solid rgba(255,255,255,0.08); }
.kpi::before { content:""; position:absolute; inset:0 auto 0 0; width:4px; background: var(--accent); }
.kpi-label { color:#8A93B2; font-size:.78rem; text-transform:uppercase; letter-spacing:.1em; }
.kpi-value { font-family:'Space Grotesk',sans-serif; font-size:2rem; font-weight:700; color:#fff; margin-top:.2rem; }
.kpi-hint { color:#8A93B2; font-size:.8rem; }

/* ---------- verdict ---------- */
.verdict { border-radius:20px; padding:1.1rem 1.3rem; text-align:center; border:1px solid var(--c);
  background: radial-gradient(120% 140% at 50% 0%, color-mix(in srgb, var(--c) 22%, transparent), transparent 70%);
  box-shadow: 0 0 40px -12px var(--c); }
.verdict-big { font-family:'Space Grotesk',sans-serif; font-size:1.6rem; font-weight:700; color:#fff; }
.verdict-small { color:#C9CEE3; font-size:.92rem; margin-top:.25rem; }

/* ---------- playbook ---------- */
.action { display:flex; gap:.8rem; align-items:flex-start; padding:.8rem .95rem; border-radius:14px; margin-bottom:.55rem;
  background: rgba(139,92,246,.08); border:1px solid rgba(139,92,246,.25); transition: transform .15s ease, border-color .15s; }
.action:hover { transform: translateX(4px); border-color: rgba(34,211,238,.5); }
.action-title { font-weight:600; color:#fff; }
.action-detail { color:#B9BFD6; font-size:.88rem; }
.action-driver { margin-left:auto; font-size:.72rem; color:#A78BFA; white-space:nowrap; padding:.15rem .5rem;
  border-radius:999px; border:1px solid rgba(167,139,250,.35); }

.stButton button, .stDownloadButton button { border-radius: 12px; border:1px solid rgba(255,255,255,.12);
  background: rgba(255,255,255,.04); transition: all .15s ease; }
.stButton button:hover, .stDownloadButton button:hover { border-color:#8B5CF6; color:#fff;
  box-shadow: 0 0 0 3px rgba(139,92,246,.2); transform: translateY(-1px); }
footer { visibility:hidden; }
</style>
"""


def hero(model_name: str, auc: float) -> str:
    return f"""
<div class="hero">
  <div style="display:flex; gap:1rem; align-items:center;">
    <div class="radar"></div>
    <div>
      <div class="hero-title">ChurnRadar</div>
      <div class="hero-sub">Spot customers about to leave, see why, and know what to do about it.</div>
    </div>
  </div>
  <div class="status-pill"><span class="dot"></span>Model live · {model_name} · ROC-AUC {auc:.3f}</div>
</div>"""


def kpi(label: str, value: str, hint: str = "", accent: str = VIOLET) -> str:
    return (f'<div class="kpi" style="--accent:{accent}"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div><div class="kpi-hint">{hint}</div></div>')


def section(label: str) -> str:
    return f'<div class="section-label">{label}</div>'


def style_fig(fig: go.Figure, height: int = 340) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#C9CEE3"),
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.08, x=0),
        hoverlabel=dict(bgcolor="#111830", bordercolor="#8B5CF6"),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", zeroline=False)
    return fig
