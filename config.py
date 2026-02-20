# =============================================================================
# config.py — Central configuration for the SPY Volatility Pipeline
# Edit this file to change tickers, thresholds, paths, and alert settings.
# =============================================================================

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DB_PATH     = BASE_DIR / "quant.db"
LOG_PATH    = BASE_DIR / "pipeline.log"
OUTPUT_DIR  = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

FIGURE_OUTPUT  = OUTPUT_DIR / "vol_analysis.png"
SURFACE_OUTPUT = OUTPUT_DIR / "iv_surface.png"

# ── Data ──────────────────────────────────────────────────────────────────────
TICKER          = "SPY"
PRICE_PERIOD    = "2y"
PRICE_INTERVAL  = "1d"
MAX_EXPIRATIONS = 5
RETRY_ATTEMPTS  = 3
RETRY_DELAY     = 5      # seconds between retries

# ── Validation thresholds ──────────────────────────────────────────────────────
ZSCORE_SPIKE_THRESHOLD    = 4.0
IQR_MULTIPLIER            = 3.0
MAX_BID_ASK_SPREAD_PCT    = 0.50
MAX_IV_THRESHOLD          = 5.0
MIN_OPTION_ROWS_PER_EXPIRY = 5
MAX_STALE_DAYS            = 5

# ── Analysis ──────────────────────────────────────────────────────────────────
RVOL_WINDOWS          = (10, 21, 63)
TRADING_DAYS_PER_YEAR = 252

# ── Alerts ────────────────────────────────────────────────────────────────────
# Set ALERT_EMAIL to your address and fill SMTP settings to get email alerts.
# Leave ALERT_EMAIL = "" to disable email and only log to console/file.

ALERT_EMAIL       = ""           # e.g. "you@gmail.com"
SMTP_HOST         = "smtp.gmail.com"
SMTP_PORT         = 587
SMTP_USER         = ""           # your Gmail address
SMTP_PASSWORD     = ""           # Gmail App Password (not your login password)
ALERT_FROM        = ""           # same as SMTP_USER usually

# Alert triggers — pipeline will send an alert if:
ALERT_ON_ANY_FAILURE   = True    # any stage crashes
ALERT_ON_ANOMALIES     = True    # validation finds issues
ALERT_ON_STALE_DATA    = True    # data is older than MAX_STALE_DAYS
ALERT_VRP_THRESHOLD    = 0.20    # alert if |VRP| > 20%  (unusual vol premium)

# ── Scheduler ──────────────────────────────────────────────────────────────────
# Market closes at 4:00 PM ET. Pipeline runs 15 min after close to allow
# yfinance data to settle. Uses your local system time — adjust if needed.
SCHEDULE_TIME = "16:15"          # 24h format, local time
