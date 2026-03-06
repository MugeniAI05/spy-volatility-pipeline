# =============================================================================
# streamlit_app.py — SPY Volatility Pipeline Dashboard
# Drop this file in your project root and run: streamlit run streamlit_app.py
# =============================================================================

import sqlite3
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SPY Vol Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0a0e17;
    color: #c9d1e0;
  }

  .main { background-color: #0a0e17; }
  .block-container { padding: 2rem 2.5rem 2rem 2.5rem; max-width: 1400px; }

  /* Header */
  .dash-header {
    border-bottom: 1px solid #1e2a3a;
    padding-bottom: 1.2rem;
    margin-bottom: 2rem;
  }
  .dash-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.5rem;
    font-weight: 600;
    color: #e8edf5;
    letter-spacing: 0.02em;
  }
  .dash-subtitle {
    font-size: 0.78rem;
    color: #4a6080;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-top: 0.2rem;
  }

  /* Metric cards */
  .metric-card {
    background: #0f1520;
    border: 1px solid #1a2535;
    border-radius: 4px;
    padding: 1.1rem 1.3rem;
    position: relative;
    overflow: hidden;
  }
  .metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: #1e6ef4;
  }
  .metric-card.warn::before { background: #f4a11e; }
  .metric-card.good::before { background: #1ef47a; }
  .metric-card.danger::before { background: #f41e3a; }

  .metric-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: #4a6080;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.4rem;
  }
  .metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.55rem;
    font-weight: 600;
    color: #e8edf5;
    line-height: 1;
  }
  .metric-delta {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #4a6080;
    margin-top: 0.35rem;
  }
  .metric-delta.up { color: #f4a11e; }
  .metric-delta.good { color: #1ef47a; }

  /* Section headers */
  .section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #4a6080;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    border-bottom: 1px solid #1a2535;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
    margin-top: 1.5rem;
  }

  /* Status badge */
  .badge {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    padding: 0.2rem 0.6rem;
    border-radius: 2px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  .badge-ok { background: #0d2e1a; color: #1ef47a; border: 1px solid #1ef47a33; }
  .badge-warn { background: #2e1f0d; color: #f4a11e; border: 1px solid #f4a11e33; }
  .badge-danger { background: #2e0d15; color: #f41e3a; border: 1px solid #f41e3a33; }

  /* Table */
  .stDataFrame { border: 1px solid #1a2535 !important; }
  thead tr th {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.65rem !important;
    text-transform: uppercase !important;
    color: #4a6080 !important;
    background: #0f1520 !important;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: #0a0e17;
    border-right: 1px solid #1a2535;
  }

  /* Tab styling */
  .stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: #0f1520;
    border-bottom: 1px solid #1a2535;
    border-radius: 4px 4px 0 0;
  }
  .stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #4a6080;
    padding: 0.7rem 1.2rem;
    border-radius: 0;
  }
  .stTabs [aria-selected="true"] {
    color: #e8edf5 !important;
    border-bottom: 2px solid #1e6ef4 !important;
    background: transparent !important;
  }

  /* Button */
  .stButton > button {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    background: #0f1520;
    color: #1e6ef4;
    border: 1px solid #1e6ef4;
    border-radius: 3px;
    padding: 0.5rem 1.2rem;
  }
  .stButton > button:hover {
    background: #1e6ef4;
    color: #fff;
  }

  /* Hide Streamlit branding */
  #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── DB path ───────────────────────────────────────────────────────────────────
DB_PATH = Path(__file__).parent / "quant.db"

# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_prices():
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql("SELECT date, open, high, low, close, adj_close, volume FROM price_history ORDER BY date ASC", conn)
    df["date"] = pd.to_datetime(df["date"])
    price_col = "adj_close" if df["adj_close"].notna().sum() > 10 else "close"
    df["price"] = df[price_col]
    df["log_return"] = np.log(df["price"] / df["price"].shift(1))
    for w in [10, 21, 63]:
        df[f"rvol_{w}d"] = df["log_return"].rolling(w).std() * np.sqrt(252) * 100
    return df.dropna(subset=["log_return"])

@st.cache_data(ttl=300)
def load_options():
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql("SELECT * FROM options_chain", conn)
    df["expiration"] = pd.to_datetime(df["expiration"])
    return df

@st.cache_data(ttl=300)
def load_analysis_results():
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql("SELECT * FROM analysis_results ORDER BY run_date DESC", conn)
    df["run_date"] = pd.to_datetime(df["run_date"])
    return df

@st.cache_data(ttl=300)
def load_etl_log():
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql("SELECT * FROM etl_log ORDER BY run_time DESC", conn)
    return df

# ── IV Surface helper ─────────────────────────────────────────────────────────
def build_iv_surface(options_df):
    calls = options_df[options_df["option_type"] == "call"].copy()
    calls = calls[(calls["implied_vol"] > 0.01) & (calls["implied_vol"] < 5.0)]
    if calls.empty:
        return pd.DataFrame()
    def atm_strike(group):
        return group.loc[group["implied_vol"].idxmin(), "strike"]
    atm = calls.groupby("expiration").apply(atm_strike).rename("atm_strike")
    calls = calls.merge(atm, on="expiration")
    calls["moneyness"] = calls["strike"] / calls["atm_strike"]
    bins = np.arange(0.80, 1.25, 0.05)
    labels = [f"{b:.2f}" for b in bins[:-1]]
    calls["moneyness_bin"] = pd.cut(calls["moneyness"], bins=bins, labels=labels, include_lowest=True)
    today = pd.Timestamp.today().normalize()
    calls["dte"] = (calls["expiration"] - today).dt.days
    surface = calls.pivot_table(index="moneyness_bin", columns="dte", values="implied_vol", aggfunc="mean")
    return surface.dropna(how="all", axis=0).dropna(how="all", axis=1)

# ── Load data ─────────────────────────────────────────────────────────────────
prices    = load_prices()
options   = load_options()
results   = load_analysis_results()
etl_log   = load_etl_log()

latest    = results.iloc[0] if not results.empty else None
last_price = prices.iloc[-1]["price"] if not prices.empty else None
last_date  = prices.iloc[-1]["date"] if not prices.empty else None

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="dash-header">
  <div class="dash-title">SPY // Volatility Intelligence</div>
  <div class="dash-subtitle">
    Last updated: {last_date.strftime('%Y-%m-%d') if last_date is not None else 'N/A'}
    &nbsp;·&nbsp; {len(prices)} trading days
    &nbsp;·&nbsp; {len(options)} option contracts
  </div>
</div>
""", unsafe_allow_html=True)

# ── Top metric cards ──────────────────────────────────────────────────────────
if latest is not None:
    vrp = latest["vrp"]
    vrp_class = "danger" if vrp > 0.20 else ("warn" if vrp > 0.10 else "good")
    rvol10 = latest["rvol_10d"]
    rvol21 = latest["rvol_21d"]
    rvol63 = latest["rvol_63d"]
    garch_vol = latest["garch_current_vol"]
    persistence = latest["garch_persistence"]
    atm_iv = latest["atm_iv"]

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">SPY Price</div>
          <div class="metric-value">${last_price:.2f}</div>
          <div class="metric-delta">{last_date.strftime('%b %d, %Y') if last_date else ''}</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">10d Realized Vol</div>
          <div class="metric-value">{rvol10:.1%}</div>
          <div class="metric-delta">21d: {rvol21:.1%} · 63d: {rvol63:.1%}</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">GARCH Cond. Vol</div>
          <div class="metric-value">{garch_vol:.1%}</div>
          <div class="metric-delta">Long-run: {latest['garch_longrun_vol']:.1%}</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Near ATM IV</div>
          <div class="metric-value">{atm_iv:.1%}</div>
          <div class="metric-delta">Nearest expiration</div>
        </div>""", unsafe_allow_html=True)

    with c5:
        vrp_delta_class = "up" if vrp > 0.10 else "good"
        st.markdown(f"""
        <div class="metric-card {vrp_class}">
          <div class="metric-label">VRP (IV − RVol)</div>
          <div class="metric-value">{vrp:+.1%}</div>
          <div class="metric-delta {vrp_delta_class}">{'Rich options ▲' if vrp > 0 else 'Cheap options ▼'}</div>
        </div>""", unsafe_allow_html=True)

    with c6:
        p_class = "warn" if persistence > 0.95 else ("good" if persistence < 0.85 else "")
        st.markdown(f"""
        <div class="metric-card {p_class}">
          <div class="metric-label">GARCH Persistence</div>
          <div class="metric-value">{persistence:.4f}</div>
          <div class="metric-delta">α+β · {'High vol memory' if persistence > 0.93 else 'Moderate'}</div>
        </div>""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Price & Volatility",
    "🌐  IV Surface",
    "🔍  Data Quality",
    "🗄️  Database"
])

# ═══════════════════════════════════════════════════════════════════
# TAB 1 — Price & Volatility
# ═══════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Price History & Realized Volatility</div>', unsafe_allow_html=True)

    # Date range filter
    col_l, col_r = st.columns([3, 1])
    with col_r:
        lookback = st.selectbox("Lookback", ["6M", "1Y", "2Y", "All"], index=2, label_visibility="collapsed")

    cutoff = {
        "6M": pd.Timestamp.today() - pd.DateOffset(months=6),
        "1Y": pd.Timestamp.today() - pd.DateOffset(years=1),
        "2Y": pd.Timestamp.today() - pd.DateOffset(years=2),
        "All": prices["date"].min(),
    }[lookback]

    px_view = prices[prices["date"] >= cutoff].copy()

    # Price chart with volume
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.5, 0.3, 0.2],
        vertical_spacing=0.03,
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=px_view["date"],
        open=px_view["open"], high=px_view["high"],
        low=px_view["low"], close=px_view["close"],
        name="SPY",
        increasing_line_color="#1ef47a",
        decreasing_line_color="#f41e3a",
        increasing_fillcolor="#0d2e1a",
        decreasing_fillcolor="#2e0d15",
        line_width=1,
    ), row=1, col=1)

    # Realized vol lines
    colors = {"rvol_10d": "#f4a11e", "rvol_21d": "#1e6ef4", "rvol_63d": "#9b5cf4"}
    labels = {"rvol_10d": "10d RVol", "rvol_21d": "21d RVol", "rvol_63d": "63d RVol"}
    for col, color in colors.items():
        fig.add_trace(go.Scatter(
            x=px_view["date"], y=px_view[col],
            name=labels[col], line=dict(color=color, width=1.5),
            mode="lines",
        ), row=2, col=1)

    # Add ATM IV reference line if available
    if latest is not None:
        fig.add_hline(
            y=latest["atm_iv"] * 100,
            line_dash="dot", line_color="#f41e3a", line_width=1,
            annotation_text=f"ATM IV {latest['atm_iv']:.0%}",
            annotation_font_color="#f41e3a",
            annotation_font_size=10,
            row=2, col=1
        )

    # Volume bars
    vol_colors = ["#1a2d1a" if c >= o else "#2d1a1a"
                  for c, o in zip(px_view["close"], px_view["open"])]
    fig.add_trace(go.Bar(
        x=px_view["date"], y=px_view["volume"],
        name="Volume", marker_color=vol_colors, showlegend=False,
    ), row=3, col=1)

    fig.update_layout(
        height=580,
        paper_bgcolor="#0a0e17",
        plot_bgcolor="#0f1520",
        font=dict(family="IBM Plex Mono", color="#4a6080", size=11),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
            font=dict(size=10), bgcolor="rgba(0,0,0,0)",
        ),
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    fig.update_xaxes(gridcolor="#1a2535", showgrid=True, zeroline=False)
    fig.update_yaxes(gridcolor="#1a2535", showgrid=True, zeroline=False)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1, title_font_size=10)
    fig.update_yaxes(title_text="RVol (%)", row=2, col=1, title_font_size=10)
    fig.update_yaxes(title_text="Volume", row=3, col=1, title_font_size=10)

    st.plotly_chart(fig, use_container_width=True)

    # GARCH section
    st.markdown('<div class="section-header">GARCH(1,1) Parameters</div>', unsafe_allow_html=True)

    if latest is not None:
        g1, g2, g3, g4 = st.columns(4)
        params = [
            ("ω (omega)", f"{latest['garch_omega']:.6f}", "Variance intercept"),
            ("α (alpha)", f"{latest['garch_alpha']:.4f}", "ARCH — shock impact"),
            ("β (beta)", f"{latest['garch_beta']:.4f}", "GARCH — vol persistence"),
            ("α+β", f"{latest['garch_persistence']:.4f}", "Total persistence"),
        ]
        for col, (label, val, desc) in zip([g1, g2, g3, g4], params):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="metric-label">{label}</div>
                  <div class="metric-value" style="font-size:1.2rem">{val}</div>
                  <div class="metric-delta">{desc}</div>
                </div>""", unsafe_allow_html=True)

    # Historical analysis results if more than one run
    if len(results) > 1:
        st.markdown('<div class="section-header">Historical Run Metrics</div>', unsafe_allow_html=True)
        hist_fig = go.Figure()
        hist_fig.add_trace(go.Scatter(
            x=results["run_date"], y=results["rvol_21d"] * 100,
            name="21d RVol", line=dict(color="#1e6ef4", width=2),
        ))
        hist_fig.add_trace(go.Scatter(
            x=results["run_date"], y=results["atm_iv"] * 100,
            name="ATM IV", line=dict(color="#f41e3a", width=2, dash="dot"),
        ))
        hist_fig.add_trace(go.Scatter(
            x=results["run_date"], y=results["garch_current_vol"] * 100,
            name="GARCH Vol", line=dict(color="#f4a11e", width=1.5),
        ))
        hist_fig.update_layout(
            height=250, paper_bgcolor="#0a0e17", plot_bgcolor="#0f1520",
            font=dict(family="IBM Plex Mono", color="#4a6080", size=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=0, r=0, t=10, b=0),
        )
        hist_fig.update_xaxes(gridcolor="#1a2535")
        hist_fig.update_yaxes(gridcolor="#1a2535", title_text="Vol (%)", title_font_size=10)
        st.plotly_chart(hist_fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 2 — IV Surface
# ═══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Implied Volatility Surface — Calls</div>', unsafe_allow_html=True)

    surface = build_iv_surface(options)

    if not surface.empty:
        col_a, col_b = st.columns([2, 1])

        with col_a:
            # Heatmap
            z_vals = surface.values.astype(float) * 100
            heat_fig = go.Figure(go.Heatmap(
                z=z_vals,
                x=[f"{int(c)}d" for c in surface.columns],
                y=[str(r) for r in surface.index],
                colorscale=[
                    [0.0, "#0d2e1a"], [0.3, "#1e6ef4"],
                    [0.6, "#f4a11e"], [1.0, "#f41e3a"]
                ],
                text=[[f"{v:.0f}%" for v in row] for row in z_vals],
                texttemplate="%{text}",
                textfont=dict(family="IBM Plex Mono", size=10, color="#e8edf5"),
                hoverongaps=False,
                colorbar=dict(
                    title=dict(text="IV %", font=dict(family="IBM Plex Mono", size=10, color="#4a6080")),
                    tickfont=dict(family="IBM Plex Mono", size=9, color="#4a6080"),
                ),
            ))
            heat_fig.update_layout(
                height=420,
                paper_bgcolor="#0a0e17",
                plot_bgcolor="#0f1520",
                font=dict(family="IBM Plex Mono", color="#4a6080", size=10),
                xaxis=dict(title="Days to Expiration", gridcolor="#1a2535"),
                yaxis=dict(title="Moneyness (K / ATM)", gridcolor="#1a2535"),
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(heat_fig, use_container_width=True)

        with col_b:
            # Vol skew for nearest expiration
            st.markdown('<div class="section-header">Vol Skew — Near Exp</div>', unsafe_allow_html=True)
            near_exp = options["expiration"].min()
            near_calls = options[
                (options["expiration"] == near_exp) &
                (options["option_type"] == "call") &
                (options["implied_vol"] > 0.01) &
                (options["implied_vol"] < 5.0)
            ].sort_values("strike")

            near_puts = options[
                (options["expiration"] == near_exp) &
                (options["option_type"] == "put") &
                (options["implied_vol"] > 0.01) &
                (options["implied_vol"] < 5.0)
            ].sort_values("strike")

            skew_fig = go.Figure()
            skew_fig.add_trace(go.Scatter(
                x=near_calls["strike"], y=near_calls["implied_vol"] * 100,
                name="Calls", line=dict(color="#1e6ef4", width=2), mode="lines+markers",
                marker=dict(size=4),
            ))
            skew_fig.add_trace(go.Scatter(
                x=near_puts["strike"], y=near_puts["implied_vol"] * 100,
                name="Puts", line=dict(color="#f41e3a", width=2), mode="lines+markers",
                marker=dict(size=4),
            ))
            if last_price:
                skew_fig.add_vline(
                    x=last_price, line_dash="dot", line_color="#4a6080",
                    annotation_text="Spot", annotation_font_size=9,
                    annotation_font_color="#4a6080",
                )
            skew_fig.update_layout(
                height=260,
                paper_bgcolor="#0a0e17", plot_bgcolor="#0f1520",
                font=dict(family="IBM Plex Mono", color="#4a6080", size=9),
                legend=dict(orientation="h", y=1.05, bgcolor="rgba(0,0,0,0)", font=dict(size=9)),
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis=dict(title="Strike", gridcolor="#1a2535"),
                yaxis=dict(title="IV %", gridcolor="#1a2535"),
            )
            st.plotly_chart(skew_fig, use_container_width=True)

            # VRP summary box
            if latest is not None:
                vrp = latest["vrp"]
                badge = "badge-danger" if vrp > 0.20 else ("badge-warn" if vrp > 0.05 else "badge-ok")
                badge_text = "RICH" if vrp > 0.05 else "CHEAP" if vrp < -0.05 else "FAIR"
                st.markdown(f"""
                <div class="metric-card" style="margin-top:1rem">
                  <div class="metric-label">Volatility Risk Premium</div>
                  <div style="display:flex; align-items:baseline; gap:0.8rem; margin-top:0.3rem">
                    <div class="metric-value" style="font-size:1.3rem">{vrp:+.1%}</div>
                    <span class="badge {badge}">{badge_text}</span>
                  </div>
                  <div class="metric-delta" style="margin-top:0.5rem">
                    ATM IV {latest['atm_iv']:.1%} vs 21d RVol {latest['rvol_21d']:.1%}
                  </div>
                </div>""", unsafe_allow_html=True)

        # 3D surface
        st.markdown('<div class="section-header">3D Vol Surface</div>', unsafe_allow_html=True)
        surf_3d = go.Figure(go.Surface(
            z=surface.values.astype(float) * 100,
            x=surface.columns.tolist(),
            y=list(range(len(surface.index))),
            colorscale=[
                [0.0, "#0d2e1a"], [0.3, "#1e6ef4"],
                [0.6, "#f4a11e"], [1.0, "#f41e3a"]
            ],
            contours_z=dict(show=True, usecolormap=True, highlightcolor="#e8edf5", project_z=False),
            colorbar=dict(
                tickfont=dict(family="IBM Plex Mono", size=9, color="#4a6080"),
            ),
        ))
        surf_3d.update_layout(
            height=480,
            paper_bgcolor="#0a0e17",
            scene=dict(
                bgcolor="#0f1520",
                xaxis=dict(title="DTE", gridcolor="#1a2535", backgroundcolor="#0f1520", color="#4a6080"),
                yaxis=dict(title="Moneyness Bin", gridcolor="#1a2535", backgroundcolor="#0f1520", color="#4a6080", tickvals=list(range(len(surface.index))), ticktext=list(surface.index)),
                zaxis=dict(title="IV %", gridcolor="#1a2535", backgroundcolor="#0f1520", color="#4a6080"),
            ),
            margin=dict(l=0, r=0, t=10, b=0),
            font=dict(family="IBM Plex Mono", color="#4a6080", size=10),
        )
        st.plotly_chart(surf_3d, use_container_width=True)

    else:
        st.info("No options data available for IV surface.")

# ═══════════════════════════════════════════════════════════════════
# TAB 3 — Data Quality
# ═══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Automated Quality Checks</div>', unsafe_allow_html=True)

    # Run checks inline without importing validate.py (avoids path issues)
    price_df = prices.copy().reset_index(drop=True)
    opts_df  = options.copy()

    checks = {}

    # Check 1: Price spikes
    price_df2 = price_df.copy().sort_values("date")
    price_df2["daily_return"] = price_df2["close"].pct_change()
    mean_r = price_df2["daily_return"].mean()
    std_r  = price_df2["daily_return"].std()
    price_df2["zscore"] = (price_df2["daily_return"] - mean_r) / std_r
    spikes = price_df2[price_df2["zscore"].abs() > 4.0][["date", "close", "daily_return", "zscore"]].copy()
    spikes["daily_return"] = spikes["daily_return"].map("{:.2%}".format)
    spikes["zscore"] = spikes["zscore"].map("{:.2f}".format)
    spikes["date"] = spikes["date"].dt.strftime("%Y-%m-%d")
    checks["Price Spikes (|z|>4)"] = spikes

    # Check 2: Zero/negative prices
    bad_price = price_df[(price_df["close"] <= 0) | (price_df["open"] <= 0)][["date", "open", "close"]].copy()
    checks["Zero/Negative Prices"] = bad_price

    # Check 3: OHLC consistency
    ohlc_bad = price_df[
        (price_df["high"] < price_df["low"]) |
        (price_df["high"] < price_df["close"]) |
        (price_df["low"] > price_df["close"])
    ][["date", "open", "high", "low", "close"]].copy()
    checks["OHLC Inconsistencies"] = ohlc_bad

    # Check 4: Date gaps
    price_sorted = price_df.sort_values("date").copy()
    price_sorted["gap"] = (price_sorted["date"] - price_sorted["date"].shift(1)).dt.days
    gaps = price_sorted[price_sorted["gap"] > 5][["date", "gap"]].copy()
    gaps["date"] = gaps["date"].dt.strftime("%Y-%m-%d")
    checks["Date Gaps >5 days"] = gaps

    # Check 5: Bid-ask inversions
    ba_inv = opts_df[opts_df["bid"] > opts_df["ask"]][["expiration", "option_type", "strike", "bid", "ask"]].copy()
    checks["Bid-Ask Inversions"] = ba_inv

    # Check 6: Wide spreads
    valid_opts = opts_df[(opts_df["bid"] > 0) & (opts_df["ask"] > 0) & (opts_df["ask"] >= opts_df["bid"])].copy()
    valid_opts["mid"] = (valid_opts["bid"] + valid_opts["ask"]) / 2
    valid_opts["spread_pct"] = (valid_opts["ask"] - valid_opts["bid"]) / valid_opts["mid"].replace(0, np.nan)
    wide = valid_opts[valid_opts["spread_pct"] > 0.50][["expiration", "option_type", "strike", "bid", "ask", "spread_pct"]].copy()
    wide["spread_pct"] = wide["spread_pct"].map("{:.1%}".format)
    checks["Wide Bid-Ask (>50%)"] = wide

    # Check 7: IV outliers
    iv_bad = opts_df[(opts_df["implied_vol"] <= 0) | (opts_df["implied_vol"] > 5.0)][
        ["expiration", "option_type", "strike", "implied_vol"]
    ].copy()
    checks["IV Outliers"] = iv_bad

    # Check 8: Freshness
    latest_price_date = price_df["date"].max()
    staleness = (pd.Timestamp.today() - latest_price_date).days
    fresh = staleness <= 5

    # Summary grid
    total_issues = sum(len(v) for v in checks.values())
    status_html = ""
    for name, df_check in checks.items():
        count = len(df_check)
        badge_cls = "badge-ok" if count == 0 else ("badge-warn" if count < 10 else "badge-danger")
        badge_txt = "PASS" if count == 0 else f"{count} issues"
        status_html += f'<div style="display:flex; justify-content:space-between; padding:0.45rem 0; border-bottom:1px solid #1a2535; font-family:IBM Plex Mono; font-size:0.75rem; color:#c9d1e0"><span>{name}</span><span class="badge {badge_cls}">{badge_txt}</span></div>'

    fresh_class = "badge-ok" if fresh else "badge-danger"
    fresh_txt = f"FRESH ({staleness}d ago)" if fresh else f"STALE ({staleness}d ago)"
    status_html += f'<div style="display:flex; justify-content:space-between; padding:0.45rem 0; border-bottom:1px solid #1a2535; font-family:IBM Plex Mono; font-size:0.75rem; color:#c9d1e0"><span>Data Freshness</span><span class="badge {fresh_class}">{fresh_txt}</span></div>'

    total_class = "badge-ok" if total_issues == 0 else ("badge-warn" if total_issues < 20 else "badge-danger")
    status_html += f'<div style="display:flex; justify-content:space-between; padding:0.6rem 0; font-family:IBM Plex Mono; font-size:0.8rem; font-weight:600; color:#e8edf5"><span>Total Anomalies</span><span class="badge {total_class}">{total_issues}</span></div>'

    qa_col1, qa_col2 = st.columns([1, 2])

    with qa_col1:
        st.markdown(f'<div style="background:#0f1520; border:1px solid #1a2535; border-radius:4px; padding:1rem">{status_html}</div>', unsafe_allow_html=True)

    with qa_col2:
        # Show detail for selected check
        selected = st.selectbox(
            "Inspect check",
            [k for k, v in checks.items() if len(v) > 0],
            label_visibility="collapsed",
        ) if any(len(v) > 0 for v in checks.values()) else None

        if selected and selected in checks:
            df_show = checks[selected]
            if not df_show.empty:
                st.dataframe(
                    df_show.head(20),
                    use_container_width=True,
                    height=300,
                )
            else:
                st.success("No issues found.")
        elif not any(len(v) > 0 for v in checks.values()):
            st.success("✓ All checks passed — data looks clean.")

    # Return distribution
    st.markdown('<div class="section-header">Return Distribution</div>', unsafe_allow_html=True)
    returns = price_df2["daily_return"].dropna() * 100
    ret_fig = go.Figure()
    ret_fig.add_trace(go.Histogram(
        x=returns, nbinsx=80, name="Daily Returns",
        marker_color="#1e6ef4", opacity=0.7,
        marker_line_color="#0a0e17", marker_line_width=0.5,
    ))
    # Normal overlay
    x_range = np.linspace(returns.min(), returns.max(), 200)
    norm_y = (1 / (returns.std() * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - returns.mean()) / returns.std()) ** 2)
    norm_y_scaled = norm_y * len(returns) * (returns.max() - returns.min()) / 80
    ret_fig.add_trace(go.Scatter(
        x=x_range, y=norm_y_scaled,
        name="Normal dist.", line=dict(color="#f4a11e", width=2, dash="dot"),
    ))
    # Mark spike threshold
    thresh = returns.mean() + 4 * returns.std()
    ret_fig.add_vline(x=thresh, line_dash="dot", line_color="#f41e3a", line_width=1,
                      annotation_text="+4σ", annotation_font_color="#f41e3a", annotation_font_size=9)
    ret_fig.add_vline(x=-thresh, line_dash="dot", line_color="#f41e3a", line_width=1,
                      annotation_text="−4σ", annotation_font_color="#f41e3a", annotation_font_size=9)
    ret_fig.update_layout(
        height=280,
        paper_bgcolor="#0a0e17", plot_bgcolor="#0f1520",
        font=dict(family="IBM Plex Mono", color="#4a6080", size=10),
        legend=dict(orientation="h", y=1.05, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(title="Daily Return (%)", gridcolor="#1a2535"),
        yaxis=dict(title="Count", gridcolor="#1a2535"),
    )
    st.plotly_chart(ret_fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 4 — Database
# ═══════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Database Explorer</div>', unsafe_allow_html=True)

    db_tab1, db_tab2, db_tab3, db_tab4 = st.tabs([
        "analysis_results", "price_history", "options_chain", "etl_log"
    ])

    with db_tab1:
        st.dataframe(results.style.format({
            c: "{:.4f}" for c in results.select_dtypes("float").columns
        }), use_container_width=True, height=300)

    with db_tab2:
        st.dataframe(prices[["date","open","high","low","close","adj_close","volume"]].tail(100), use_container_width=True, height=300)

    with db_tab3:
        st.dataframe(options.sort_values(["expiration","option_type","strike"]).head(200), use_container_width=True, height=300)

    with db_tab4:
        st.dataframe(etl_log, use_container_width=True, height=300)

    # Summary stats
    st.markdown('<div class="section-header">Table Row Counts</div>', unsafe_allow_html=True)
    with sqlite3.connect(DB_PATH) as conn:
        tables = ["price_history", "options_chain", "analysis_results", "etl_log"]
        counts = {t: pd.read_sql(f"SELECT COUNT(*) as n FROM {t}", conn).iloc[0]["n"] for t in tables}

    cols = st.columns(4)
    for col, (tname, count) in zip(cols, counts.items()):
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">{tname}</div>
              <div class="metric-value" style="font-size:1.3rem">{count:,}</div>
              <div class="metric-delta">rows</div>
            </div>""", unsafe_allow_html=True)
