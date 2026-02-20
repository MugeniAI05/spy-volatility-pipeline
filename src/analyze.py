# =============================================================================
# src/analyze.py — Time Series & Volatility Analysis
# Computes realized vol, GARCH(1,1), mean-reversion, IV surface, VRP.
# Saves results to DB and outputs charts.
# =============================================================================

import logging
import sqlite3
import warnings
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sqlalchemy import create_engine, text

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DB_PATH, TICKER, RVOL_WINDOWS, TRADING_DAYS_PER_YEAR,
    FIGURE_OUTPUT, SURFACE_OUTPUT, ALERT_VRP_THRESHOLD
)

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_prices() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql(
            "SELECT date, close, adj_close, volume FROM price_history ORDER BY date ASC", conn
        )
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    price_col   = "adj_close" if df["adj_close"].notna().sum() > 10 else "close"
    df["price"] = df[price_col]
    df["log_return"] = np.log(df["price"] / df["price"].shift(1))
    return df.dropna(subset=["log_return"])


def load_options() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql("SELECT * FROM options_chain", conn)
    df["expiration"] = pd.to_datetime(df["expiration"])
    return df


# ── Rolling Realized Volatility ───────────────────────────────────────────────

def rolling_realized_vol(df: pd.DataFrame) -> pd.DataFrame:
    result = df[["log_return"]].copy()
    for w in RVOL_WINDOWS:
        result[f"rvol_{w}d"] = df["log_return"].rolling(w).std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    log.info("[RVOL] Rolling realized volatility computed:")
    latest = result.iloc[-1]
    for w in RVOL_WINDOWS:
        log.info(f"  {w:>3}d RVol = {latest[f'rvol_{w}d']:.2%}")

    return result


# ── GARCH(1,1) ────────────────────────────────────────────────────────────────

def fit_garch(df: pd.DataFrame):
    try:
        from arch import arch_model
    except ImportError:
        log.error("arch package not installed. Run: pip install arch")
        return None, None

    returns_pct = df["log_return"] * 100
    model  = arch_model(returns_pct.dropna(), vol="Garch", p=1, q=1,
                        mean="Constant", dist="normal")
    result = model.fit(disp="off")

    cond_vol = result.conditional_volatility / 100 * np.sqrt(TRADING_DAYS_PER_YEAR)
    cond_vol.index = df.dropna(subset=["log_return"]).index[
        len(df.dropna()) - len(cond_vol):
    ]

    params      = result.params
    alpha       = params.get("alpha[1]", np.nan)
    beta        = params.get("beta[1]",  np.nan)
    omega       = params.get("omega",    np.nan)
    persistence = alpha + beta
    longrun_vol = np.sqrt(omega / (1 - persistence)) / 100 * np.sqrt(TRADING_DAYS_PER_YEAR)

    log.info("[GARCH] GARCH(1,1) model fitted:")
    log.info(f"  omega={omega:.6f}, alpha={alpha:.4f}, beta={beta:.4f}")
    log.info(f"  Persistence (alpha+beta) = {persistence:.4f}")
    log.info(f"  Long-run vol = {longrun_vol:.2%}")
    log.info(f"  Current conditional vol  = {cond_vol.iloc[-1]:.2%}")

    return result, cond_vol, {
        "omega": omega, "alpha": alpha, "beta": beta,
        "persistence": persistence, "longrun_vol": longrun_vol,
        "current_vol": cond_vol.iloc[-1],
    }


# ── Mean Reversion ────────────────────────────────────────────────────────────

def mean_reversion_analysis(df: pd.DataFrame) -> dict:
    from scipy import stats

    log_price = np.log(df["price"].dropna())
    ret       = df["log_return"].dropna()

    y = log_price.diff().dropna()
    x = log_price.shift(1).dropna()
    x, y = x.align(y, join="inner")
    slope, _, _, p_level, _ = stats.linregress(x, y)

    y2 = ret.diff().dropna()
    x2 = ret.shift(1).dropna()
    x2, y2 = x2.align(y2, join="inner")
    slope2, _, _, p_returns, _ = stats.linregress(x2, y2)

    lagged   = ret.shift(1).dropna()
    current  = ret.iloc[1:]
    ou_slope = np.polyfit(lagged, current, 1)[0]
    half_life = -np.log(2) / np.log(abs(ou_slope)) if ou_slope < 0 else np.inf

    log.info("[MEAN-REV] ADF-style regression results:")
    log.info(
        f"  Log price  — slope: {slope:.4f}, p-value: {p_level:.4f} "
        f"({'NON-stationary ✓' if p_level > 0.05 else 'Stationary'})"
    )
    log.info(
        f"  Log returns — slope: {slope2:.4f}, p-value: {p_returns:.4f} "
        f"({'Stationary ✓' if p_returns < 0.05 else 'NON-stationary'})"
    )
    log.info(f"  OU half-life (log returns): {half_life:.1f} trading days")

    return {
        "adf_level_slope": slope, "adf_level_pval": p_level,
        "adf_returns_slope": slope2, "adf_returns_pval": p_returns,
        "ou_half_life_days": half_life,
    }


# ── IV Surface ────────────────────────────────────────────────────────────────

def implied_vol_surface(options_df: pd.DataFrame) -> pd.DataFrame:
    if options_df.empty:
        log.warning("[IV SURFACE] No options data available.")
        return pd.DataFrame()

    calls = options_df[options_df["option_type"] == "call"].copy()
    calls = calls[(calls["implied_vol"] > 0.01) & (calls["implied_vol"] < 5.0)]
    if calls.empty:
        log.warning("[IV SURFACE] No valid call IV data.")
        return pd.DataFrame()

    def atm_strike(group):
        return group.loc[group["implied_vol"].idxmin(), "strike"]

    atm   = calls.groupby("expiration").apply(atm_strike).rename("atm_strike")
    calls = calls.merge(atm, on="expiration")
    calls["moneyness"] = calls["strike"] / calls["atm_strike"]

    bins   = np.arange(0.80, 1.25, 0.05)
    labels = [f"{b:.2f}" for b in bins[:-1]]
    calls["moneyness_bin"] = pd.cut(
        calls["moneyness"], bins=bins, labels=labels, include_lowest=True
    )

    today = pd.Timestamp.today().normalize()
    calls["dte"] = (calls["expiration"] - today).dt.days

    surface = calls.pivot_table(
        index="moneyness_bin", columns="dte",
        values="implied_vol", aggfunc="mean",
    )
    surface = surface.dropna(how="all", axis=0).dropna(how="all", axis=1)
    log.info(f"[IV SURFACE] Surface shape: {surface.shape} (moneyness bins × expirations)")
    return surface


# ── Volatility Risk Premium ───────────────────────────────────────────────────

def vol_risk_premium(price_df: pd.DataFrame, options_df: pd.DataFrame) -> dict | None:
    if options_df.empty:
        log.warning("[VRP] No options data for VRP calc.")
        return None

    rvol_21 = (
        price_df["log_return"].rolling(21).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    ).iloc[-1]

    near_exp = options_df["expiration"].min()
    atm_iv   = options_df[
        (options_df["expiration"] == near_exp) &
        (options_df["implied_vol"] > 0.01) &
        (options_df["implied_vol"] < 5.0)
    ]["implied_vol"].mean()

    vrp = atm_iv - rvol_21
    log.info("[VRP] Volatility Risk Premium:")
    log.info(f"  21d Realized Vol  = {rvol_21:.2%}")
    log.info(f"  Near-term ATM IV  = {atm_iv:.2%}")
    log.info(f"  VRP (IV - RVol)   = {vrp:+.2%}  ({'rich' if vrp > 0 else 'cheap'} options)")

    if abs(vrp) > ALERT_VRP_THRESHOLD:
        log.warning(f"  ⚠  VRP exceeds threshold ({ALERT_VRP_THRESHOLD:.0%}): {vrp:+.2%}")

    return {"rvol_21d": rvol_21, "atm_iv": atm_iv, "vrp": vrp}


# ── Persist results to DB ─────────────────────────────────────────────────────

def save_results_to_db(engine, rvol_df, garch_params, mr_stats, vrp_data):
    latest_rvol = rvol_df.iloc[-1]
    row = {
        "ticker":             TICKER,
        "rvol_10d":           latest_rvol.get("rvol_10d"),
        "rvol_21d":           latest_rvol.get("rvol_21d"),
        "rvol_63d":           latest_rvol.get("rvol_63d"),
        "garch_omega":        garch_params.get("omega")        if garch_params else None,
        "garch_alpha":        garch_params.get("alpha")        if garch_params else None,
        "garch_beta":         garch_params.get("beta")         if garch_params else None,
        "garch_persistence":  garch_params.get("persistence")  if garch_params else None,
        "garch_longrun_vol":  garch_params.get("longrun_vol")  if garch_params else None,
        "garch_current_vol":  garch_params.get("current_vol")  if garch_params else None,
        "atm_iv":             vrp_data.get("atm_iv")  if vrp_data else None,
        "vrp":                vrp_data.get("vrp")     if vrp_data else None,
        "ou_halflife":        mr_stats.get("ou_half_life_days"),
    }
    with engine.connect() as conn:
        conn.execute(
            text("""
                INSERT INTO analysis_results
                (ticker, rvol_10d, rvol_21d, rvol_63d,
                 garch_omega, garch_alpha, garch_beta, garch_persistence,
                 garch_longrun_vol, garch_current_vol, atm_iv, vrp, ou_halflife)
                VALUES
                (:ticker, :rvol_10d, :rvol_21d, :rvol_63d,
                 :garch_omega, :garch_alpha, :garch_beta, :garch_persistence,
                 :garch_longrun_vol, :garch_current_vol, :atm_iv, :vrp, :ou_halflife)
            """),
            row,
        )
        conn.commit()
    log.info("Results written to analysis_results table.")


# ── Charts ────────────────────────────────────────────────────────────────────

def plot_vol_analysis(price_df, rvol_df, garch_vol):
    sns.set_theme(style="darkgrid", palette="muted")
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
    fig.suptitle("SPY Volatility Analysis", fontsize=14, fontweight="bold")

    ax1 = axes[0]
    ax1.plot(price_df.index, price_df["price"], color="#2196F3", linewidth=1.2)
    ax1.set_title("Price History (Adjusted Close)")
    ax1.set_ylabel("Price ($)")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    ax2 = axes[1]
    colors = {"rvol_10d": "#FF5722", "rvol_21d": "#4CAF50", "rvol_63d": "#9C27B0"}
    labels = {"rvol_10d": "10d RVol", "rvol_21d": "21d RVol", "rvol_63d": "63d RVol"}
    for col, color in colors.items():
        if col in rvol_df.columns:
            ax2.plot(rvol_df.index, rvol_df[col] * 100,
                     label=labels[col], color=color, linewidth=1.0)
    ax2.set_title("Rolling Realized Volatility (Annualized %)")
    ax2.set_ylabel("Volatility (%)")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    ax3 = axes[2]
    if garch_vol is not None and len(garch_vol) > 0:
        ax3.plot(garch_vol.index, garch_vol * 100,
                 color="#FF9800", linewidth=1.0, label="GARCH(1,1)")
        ax3.set_title("GARCH(1,1) Conditional Volatility (Annualized %)")
        ax3.set_ylabel("Volatility (%)")
        ax3.legend(loc="upper right", fontsize=8)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    else:
        ax3.text(0.5, 0.5, "GARCH not computed (install arch)",
                 ha="center", va="center", transform=ax3.transAxes)

    plt.tight_layout()
    plt.savefig(FIGURE_OUTPUT, dpi=150, bbox_inches="tight")
    log.info(f"Saved: {FIGURE_OUTPUT}")
    plt.close()


def plot_iv_surface(surface: pd.DataFrame):
    if surface.empty:
        log.warning("Skipping IV surface plot — no data.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        surface.astype(float), ax=ax, cmap="RdYlGn_r",
        annot=True, fmt=".0%", linewidths=0.5,
        cbar_kws={"label": "Implied Volatility"},
    )
    ax.set_title("Implied Volatility Surface — Calls (Moneyness × Days to Expiry)", fontsize=12)
    ax.set_xlabel("Days to Expiration")
    ax.set_ylabel("Moneyness (Strike / ATM)")
    plt.tight_layout()
    plt.savefig(SURFACE_OUTPUT, dpi=150, bbox_inches="tight")
    log.info(f"Saved: {SURFACE_OUTPUT}")
    plt.close()


# ── Entry point ───────────────────────────────────────────────────────────────

def run_analysis(engine=None) -> dict:
    if engine is None:
        engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)

    log.info("=" * 60)
    log.info("ANALYSIS PIPELINE START")
    log.info("=" * 60)

    price_df   = load_prices()
    options_df = load_options()

    if price_df.empty:
        log.error("No price data in DB. Run etl.py first.")
        return {}

    rvol_df              = rolling_realized_vol(price_df)
    garch_result, garch_vol, garch_params = fit_garch(price_df)
    mr_stats             = mean_reversion_analysis(price_df)
    surface              = implied_vol_surface(options_df)
    vrp_data             = vol_risk_premium(price_df, options_df) if not options_df.empty else None

    plot_vol_analysis(price_df, rvol_df, garch_vol)
    plot_iv_surface(surface)
    save_results_to_db(engine, rvol_df, garch_params, mr_stats, vrp_data)

    log.info("=" * 60)
    log.info("ANALYSIS COMPLETE")
    log.info("=" * 60)

    return {
        "price_df": price_df, "rvol_df": rvol_df,
        "garch_result": garch_result, "garch_vol": garch_vol,
        "garch_params": garch_params, "mr_stats": mr_stats,
        "iv_surface": surface, "vrp_data": vrp_data,
    }


if __name__ == "__main__":
    import logging
    from config import LOG_PATH
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
    )
    run_analysis()
