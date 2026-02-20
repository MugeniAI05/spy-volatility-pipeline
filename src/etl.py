# =============================================================================
# src/etl.py — Data Ingestion & ETL
# Pulls SPY price history and options chain from yfinance → SQLite
# =============================================================================

import logging
import time

import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine, text

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DB_PATH, TICKER, PRICE_PERIOD, PRICE_INTERVAL,
    MAX_EXPIRATIONS, RETRY_ATTEMPTS, RETRY_DELAY
)

log = logging.getLogger(__name__)


# ── Database helpers ──────────────────────────────────────────────────────────

def get_engine():
    return create_engine(f"sqlite:///{DB_PATH}", echo=False)


def create_tables(engine):
    ddl = """
    CREATE TABLE IF NOT EXISTS price_history (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker      TEXT    NOT NULL,
        date        TEXT    NOT NULL,
        open        REAL,
        high        REAL,
        low         REAL,
        close       REAL,
        adj_close   REAL,
        volume      INTEGER,
        ingested_at TEXT    DEFAULT (datetime('now')),
        UNIQUE(ticker, date)
    );

    CREATE TABLE IF NOT EXISTS options_chain (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker          TEXT NOT NULL,
        expiration      TEXT NOT NULL,
        option_type     TEXT NOT NULL,
        strike          REAL NOT NULL,
        last_price      REAL,
        bid             REAL,
        ask             REAL,
        volume          INTEGER,
        open_interest   INTEGER,
        implied_vol     REAL,
        in_the_money    INTEGER,
        ingested_at     TEXT DEFAULT (datetime('now')),
        UNIQUE(ticker, expiration, option_type, strike)
    );

    CREATE TABLE IF NOT EXISTS etl_log (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        run_time    TEXT DEFAULT (datetime('now')),
        ticker      TEXT,
        step        TEXT,
        status      TEXT,
        rows_loaded INTEGER,
        message     TEXT
    );

    CREATE TABLE IF NOT EXISTS analysis_results (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        run_date        TEXT DEFAULT (datetime('now')),
        ticker          TEXT,
        rvol_10d        REAL,
        rvol_21d        REAL,
        rvol_63d        REAL,
        garch_omega     REAL,
        garch_alpha     REAL,
        garch_beta      REAL,
        garch_persistence REAL,
        garch_longrun_vol REAL,
        garch_current_vol REAL,
        atm_iv          REAL,
        vrp             REAL,
        ou_halflife     REAL
    );
    """
    with engine.connect() as conn:
        for stmt in ddl.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(text(stmt))
        conn.commit()
    log.info("Tables verified / created.")


def log_etl_event(engine, ticker, step, status, rows=0, message=""):
    with engine.connect() as conn:
        conn.execute(
            text(
                "INSERT INTO etl_log (ticker, step, status, rows_loaded, message) "
                "VALUES (:ticker, :step, :status, :rows, :message)"
            ),
            {"ticker": ticker, "step": step, "status": status, "rows": rows, "message": message},
        )
        conn.commit()


# ── Retry wrapper ─────────────────────────────────────────────────────────────

def retry_fetch(fn, *args, label="fetch", **kwargs):
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            log.warning(f"[{label}] Attempt {attempt}/{RETRY_ATTEMPTS} failed: {exc}")
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_DELAY)
    raise RuntimeError(f"[{label}] All {RETRY_ATTEMPTS} attempts failed.")


# ── Price history ─────────────────────────────────────────────────────────────

def ingest_price_history(engine, ticker=TICKER) -> pd.DataFrame:
    log.info(f"Ingesting price history for {ticker} ({PRICE_PERIOD})...")

    def _fetch():
        t = yf.Ticker(ticker)
        return t.history(period=PRICE_PERIOD, interval=PRICE_INTERVAL, auto_adjust=False)

    try:
        df = retry_fetch(_fetch, label="price_history")
    except RuntimeError as e:
        log_etl_event(engine, ticker, "price_history", "FAILED", message=str(e))
        log.error(str(e))
        return pd.DataFrame()

    if df.empty:
        msg = f"No price data returned for {ticker}."
        log.warning(msg)
        log_etl_event(engine, ticker, "price_history", "WARN", message=msg)
        return df

    df = df.reset_index()
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    df["ticker"] = ticker
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    missing_before = df.isnull().sum().sum()
    df[["open", "high", "low", "close", "adj_close"]] = (
        df[["open", "high", "low", "close", "adj_close"]].ffill().bfill()
    )
    df["volume"] = df["volume"].fillna(0).astype(int)
    if missing_before:
        log.info(f"  Filled {missing_before} missing price values.")

    cols = ["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]
    df = df[[c for c in cols if c in df.columns]]

    rows_loaded = 0
    with engine.connect() as conn:
        for _, row in df.iterrows():
            conn.execute(
                text(
                    "INSERT OR REPLACE INTO price_history "
                    "(ticker, date, open, high, low, close, adj_close, volume) "
                    "VALUES (:ticker, :date, :open, :high, :low, :close, :adj_close, :volume)"
                ),
                row.to_dict(),
            )
            rows_loaded += 1
        conn.commit()

    log.info(f"  Loaded {rows_loaded} price rows for {ticker}.")
    log_etl_event(engine, ticker, "price_history", "OK", rows=rows_loaded)
    return df


# ── Options chain ─────────────────────────────────────────────────────────────

def ingest_options_chain(engine, ticker=TICKER) -> pd.DataFrame:
    log.info(f"Ingesting options chain for {ticker}...")

    try:
        t = retry_fetch(lambda: yf.Ticker(ticker), label="options_ticker")
        expirations = t.options
    except Exception as e:
        log_etl_event(engine, ticker, "options_chain", "FAILED", message=str(e))
        log.error(f"Could not fetch expirations: {e}")
        return pd.DataFrame()

    if not expirations:
        msg = "No option expirations found."
        log.warning(msg)
        log_etl_event(engine, ticker, "options_chain", "WARN", message=msg)
        return pd.DataFrame()

    expirations = expirations[:MAX_EXPIRATIONS]
    log.info(f"  Pulling {len(expirations)} expiration dates: {list(expirations)}")

    all_frames = []
    for exp in expirations:
        try:
            chain = retry_fetch(t.option_chain, exp, label=f"chain_{exp}")
            for opt_type, frame in [("call", chain.calls), ("put", chain.puts)]:
                frame = frame.copy()
                frame["ticker"] = ticker
                frame["expiration"] = exp
                frame["option_type"] = opt_type
                all_frames.append(frame)
        except Exception as e:
            log.warning(f"  Skipping expiration {exp}: {e}")

    if not all_frames:
        log.error("No options data collected.")
        log_etl_event(engine, ticker, "options_chain", "FAILED", message="No data")
        return pd.DataFrame()

    df = pd.concat(all_frames, ignore_index=True)
    df = df.rename(columns={
        "contractSymbol": "contract_symbol",
        "lastPrice": "last_price",
        "openInterest": "open_interest",
        "impliedVolatility": "implied_vol",
        "inTheMoney": "in_the_money",
    })
    df["bid"] = df.get("bid", pd.Series(dtype=float)).fillna(0.0)
    df["ask"] = df.get("ask", pd.Series(dtype=float)).fillna(0.0)
    df["volume"] = df.get("volume", pd.Series(dtype=float)).fillna(0).astype(int)
    df["open_interest"] = df.get("open_interest", pd.Series(dtype=float)).fillna(0).astype(int)
    df["implied_vol"] = df.get("implied_vol", pd.Series(dtype=float)).fillna(0.0)
    df["in_the_money"] = df.get("in_the_money", pd.Series(dtype=bool)).fillna(False).astype(int)

    rows_loaded = 0
    with engine.connect() as conn:
        for _, row in df.iterrows():
            try:
                conn.execute(
                    text(
                        "INSERT OR REPLACE INTO options_chain "
                        "(ticker, expiration, option_type, strike, last_price, "
                        "bid, ask, volume, open_interest, implied_vol, in_the_money) "
                        "VALUES (:ticker, :expiration, :option_type, :strike, :last_price, "
                        ":bid, :ask, :volume, :open_interest, :implied_vol, :in_the_money)"
                    ),
                    {
                        "ticker": row["ticker"], "expiration": row["expiration"],
                        "option_type": row["option_type"], "strike": row.get("strike"),
                        "last_price": row.get("last_price"), "bid": row.get("bid"),
                        "ask": row.get("ask"), "volume": row.get("volume"),
                        "open_interest": row.get("open_interest"),
                        "implied_vol": row.get("implied_vol"),
                        "in_the_money": row.get("in_the_money"),
                    },
                )
                rows_loaded += 1
            except Exception as e:
                log.debug(f"  Row insert error (skipping): {e}")
        conn.commit()

    log.info(f"  Loaded {rows_loaded} option rows for {ticker}.")
    log_etl_event(engine, ticker, "options_chain", "OK", rows=rows_loaded)
    return df


# ── Entry point ───────────────────────────────────────────────────────────────

def run_etl(engine=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if engine is None:
        engine = get_engine()

    log.info("=" * 60)
    log.info("ETL PIPELINE START")
    log.info("=" * 60)

    create_tables(engine)
    price_df   = ingest_price_history(engine, TICKER)
    options_df = ingest_options_chain(engine, TICKER)

    log.info("=" * 60)
    log.info(f"ETL COMPLETE — Price rows: {len(price_df)}, Option rows: {len(options_df)}")
    log.info("=" * 60)
    return price_df, options_df


if __name__ == "__main__":
    import logging
    from config import LOG_PATH
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
    )
    run_etl()
