# =============================================================================
# src/validate.py — Data Validation & Quality Checks
# Runs 8 automated checks on price and options data in the DB.
# Returns a dict of anomaly DataFrames and a total count.
# =============================================================================

import logging
import sqlite3

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DB_PATH, ZSCORE_SPIKE_THRESHOLD, MAX_BID_ASK_SPREAD_PCT,
    MAX_IV_THRESHOLD, MIN_OPTION_ROWS_PER_EXPIRY, MAX_STALE_DAYS
)

log = logging.getLogger(__name__)


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_price_data() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql("SELECT * FROM price_history ORDER BY date ASC", conn)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_options_data() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql("SELECT * FROM options_chain", conn)
    df["expiration"] = pd.to_datetime(df["expiration"])
    return df


# ── Individual checks ─────────────────────────────────────────────────────────

def check_price_spikes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("date")
    df["daily_return"] = df["close"].pct_change()
    mean_ret = df["daily_return"].mean()
    std_ret  = df["daily_return"].std()
    df["zscore"] = (df["daily_return"] - mean_ret) / std_ret

    spikes = df[df["zscore"].abs() > ZSCORE_SPIKE_THRESHOLD].copy()
    spikes["anomaly_type"] = "price_spike"
    spikes["detail"] = spikes.apply(
        lambda r: f"Return={r['daily_return']:.2%}, Z={r['zscore']:.2f}", axis=1
    )
    log.info(f"[CHECK 1] Price spikes (|z| > {ZSCORE_SPIKE_THRESHOLD}): {len(spikes)} found")
    for _, row in spikes.iterrows():
        log.warning(f"  Spike on {row['date'].date()}: {row['detail']}")
    return spikes[["date", "close", "daily_return", "zscore", "anomaly_type", "detail"]]


def check_zero_negative_prices(df: pd.DataFrame) -> pd.DataFrame:
    bad = df[(df["close"] <= 0) | (df["open"] <= 0) | (df["high"] <= 0) | (df["low"] <= 0)].copy()
    bad["anomaly_type"] = "zero_or_negative_price"
    bad["detail"] = "One or more OHLC fields <= 0"
    log.info(f"[CHECK 2] Zero/negative prices: {len(bad)} found")
    if not bad.empty:
        log.warning(bad[["date", "open", "high", "low", "close"]].to_string())
    return bad[["date", "close", "anomaly_type", "detail"]]


def check_ohlc_consistency(df: pd.DataFrame) -> pd.DataFrame:
    bad = df[
        (df["high"] < df["low"]) | (df["high"] < df["open"]) |
        (df["high"] < df["close"]) | (df["low"] > df["open"]) |
        (df["low"] > df["close"])
    ].copy()
    bad["anomaly_type"] = "ohlc_inconsistency"
    bad["detail"] = "OHLC relationship violated"
    log.info(f"[CHECK 3] OHLC inconsistencies: {len(bad)} found")
    for _, row in bad.iterrows():
        log.warning(f"  {row['date'].date()} O={row['open']} H={row['high']} L={row['low']} C={row['close']}")
    return bad[["date", "open", "high", "low", "close", "anomaly_type", "detail"]]


def check_missing_dates(df: pd.DataFrame, max_gap_days=5) -> pd.DataFrame:
    df = df.sort_values("date").copy()
    df["prev_date"] = df["date"].shift(1)
    df["gap_days"]  = (df["date"] - df["prev_date"]).dt.days
    gaps = df[df["gap_days"] > max_gap_days].copy()
    gaps["anomaly_type"] = "missing_date_gap"
    gaps["detail"] = gaps["gap_days"].apply(lambda g: f"Gap of {g} calendar days")
    log.info(f"[CHECK 4] Date gaps > {max_gap_days} days: {len(gaps)} found")
    for _, row in gaps.iterrows():
        log.warning(f"  Gap ending {row['date'].date()}: {row['detail']}")
    return gaps[["date", "gap_days", "anomaly_type", "detail"]]


def check_bid_ask_inversions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    inversions = df[df["bid"] > df["ask"]].copy()
    inversions["anomaly_type"] = "bid_ask_inversion"
    inversions["detail"] = inversions.apply(
        lambda r: f"bid={r['bid']:.2f} > ask={r['ask']:.2f}", axis=1
    )
    valid = df[(df["bid"] > 0) & (df["ask"] > 0) & (df["ask"] >= df["bid"])].copy()
    valid["mid"] = (valid["bid"] + valid["ask"]) / 2
    valid["spread_pct"] = (valid["ask"] - valid["bid"]) / valid["mid"].replace(0, np.nan)
    wide = valid[valid["spread_pct"] > MAX_BID_ASK_SPREAD_PCT].copy()
    wide["anomaly_type"] = "wide_bid_ask_spread"
    wide["detail"] = wide["spread_pct"].apply(lambda s: f"Spread={s:.1%}")
    result = pd.concat([inversions, wide], ignore_index=True)
    log.info(f"[CHECK 5] Bid-ask inversions: {len(inversions)} | Wide spreads: {len(wide)}")
    return result[["expiration", "option_type", "strike", "bid", "ask", "anomaly_type", "detail"]]


def check_iv_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    zero_iv = df[df["implied_vol"] <= 0].copy()
    zero_iv["anomaly_type"] = "zero_implied_vol"
    zero_iv["detail"] = "IV is zero or negative — likely missing data"
    high_iv = df[df["implied_vol"] > MAX_IV_THRESHOLD].copy()
    high_iv["anomaly_type"] = "extreme_implied_vol"
    high_iv["detail"] = high_iv["implied_vol"].apply(lambda v: f"IV={v:.1%}")
    result = pd.concat([zero_iv, high_iv], ignore_index=True)
    log.info(f"[CHECK 6] IV outliers — zero: {len(zero_iv)}, extreme: {len(high_iv)}")
    return result[["expiration", "option_type", "strike", "implied_vol", "anomaly_type", "detail"]]


def check_sparse_expirations(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.groupby(["expiration", "option_type"]).size().reset_index(name="strike_count")
    sparse = counts[counts["strike_count"] < MIN_OPTION_ROWS_PER_EXPIRY].copy()
    sparse["anomaly_type"] = "sparse_expiration"
    sparse["detail"] = sparse["strike_count"].apply(lambda c: f"Only {c} strikes loaded")
    log.info(f"[CHECK 7] Sparse expirations: {len(sparse)} found")
    for _, row in sparse.iterrows():
        log.warning(f"  {row['expiration'].date()} {row['option_type']}: {row['detail']}")
    return sparse


def check_data_freshness(price_df: pd.DataFrame) -> bool:
    """Returns True if data is fresh, False if stale."""
    if price_df.empty:
        log.warning("[CHECK 8] No price data to check freshness.")
        return False
    latest    = price_df["date"].max()
    staleness = (pd.Timestamp.today() - latest).days
    if staleness > MAX_STALE_DAYS:
        log.warning(
            f"[CHECK 8] Data is STALE — latest: {latest.date()}, {staleness} days ago"
        )
        return False
    log.info(f"[CHECK 8] Data freshness OK — latest: {latest.date()} ({staleness} days ago)")
    return True


# ── Summary printer ───────────────────────────────────────────────────────────

def print_summary(anomaly_dict: dict) -> int:
    print("\n" + "=" * 60)
    print("DATA QUALITY REPORT SUMMARY")
    print("=" * 60)
    total = 0
    for name, df in anomaly_dict.items():
        count  = len(df)
        total += count
        status = "  PASS" if count == 0 else f"  {count} ISSUE(S)"
        print(f"  {name:<35} {status}")
    print("-" * 60)
    print(f"  {'TOTAL ANOMALIES':<35} {total}")
    print("=" * 60 + "\n")
    return total


# ── Entry point ───────────────────────────────────────────────────────────────

def run_all_checks() -> tuple[dict, int]:
    log.info("=" * 60)
    log.info("VALIDATION PIPELINE START")
    log.info("=" * 60)

    price_df   = load_price_data()
    options_df = load_options_data()

    empty = pd.DataFrame()

    spikes    = check_price_spikes(price_df)        if not price_df.empty   else empty
    zeros     = check_zero_negative_prices(price_df) if not price_df.empty  else empty
    ohlc      = check_ohlc_consistency(price_df)    if not price_df.empty   else empty
    gaps      = check_missing_dates(price_df)        if not price_df.empty  else empty
    is_fresh  = check_data_freshness(price_df)

    ba_issues = check_bid_ask_inversions(options_df) if not options_df.empty else empty
    iv_issues = check_iv_outliers(options_df)         if not options_df.empty else empty
    sparse    = check_sparse_expirations(options_df)  if not options_df.empty else empty

    anomalies = {
        "Price Spikes":          spikes,
        "Zero/Negative Prices":  zeros,
        "OHLC Inconsistencies":  ohlc,
        "Missing Date Gaps":     gaps,
        "Bid-Ask Issues":        ba_issues,
        "IV Outliers":           iv_issues,
        "Sparse Expirations":    sparse,
    }

    total = print_summary(anomalies)
    log.info(f"VALIDATION COMPLETE — {total} total anomalies found.")
    return anomalies, total, is_fresh


if __name__ == "__main__":
    import logging
    from config import LOG_PATH
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
    )
    run_all_checks()
