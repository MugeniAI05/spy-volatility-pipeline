# =============================================================================
# src/notifier.py — Email Alert System
# Sends alerts when the pipeline fails, finds anomalies, or spots unusual VRP.
# Configure SMTP settings in config.py to enable.
# =============================================================================

import logging
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    ALERT_EMAIL, SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD,
    ALERT_FROM, ALERT_ON_ANY_FAILURE, ALERT_ON_ANOMALIES,
    ALERT_ON_STALE_DATA, TICKER
)

log = logging.getLogger(__name__)


def _email_enabled() -> bool:
    return bool(ALERT_EMAIL and SMTP_USER and SMTP_PASSWORD)


def send_email(subject: str, body: str):
    """Send a plain-text email alert. No-ops if email is not configured."""
    if not _email_enabled():
        log.info(f"[ALERT] Email not configured. Would have sent: {subject}")
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = ALERT_FROM or SMTP_USER
    msg["To"]      = ALERT_EMAIL
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_USER, ALERT_EMAIL, msg.as_string())
        log.info(f"[ALERT] Email sent: {subject}")
    except Exception as e:
        log.error(f"[ALERT] Failed to send email: {e}")


def alert_pipeline_failure(stage: str, error: Exception):
    """Call this when a pipeline stage crashes."""
    if not ALERT_ON_ANY_FAILURE:
        return
    now     = datetime.now().strftime("%Y-%m-%d %H:%M")
    subject = f"[{TICKER} Pipeline] ❌ FAILURE — {stage} ({now})"
    body    = (
        f"The SPY volatility pipeline failed at stage: {stage}\n\n"
        f"Time:  {now}\n"
        f"Error: {type(error).__name__}: {error}\n\n"
        f"Check pipeline.log for full traceback."
    )
    log.error(f"[ALERT] Pipeline failure in {stage}: {error}")
    send_email(subject, body)


def alert_validation_issues(anomaly_dict: dict, total: int):
    """Call this when validation finds anomalies."""
    if not ALERT_ON_ANOMALIES or total == 0:
        return
    now     = datetime.now().strftime("%Y-%m-%d %H:%M")
    subject = f"[{TICKER} Pipeline] ⚠ {total} Data Quality Issues ({now})"

    lines = [f"Data quality check found {total} anomaly/anomalies:\n"]
    for name, df in anomaly_dict.items():
        count = len(df)
        if count > 0:
            lines.append(f"  • {name}: {count} issue(s)")
            # Show first 5 rows as a text table
            if not df.empty:
                lines.append(df.head(5).to_string(index=False))
                lines.append("")

    body = "\n".join(lines) + f"\nFull details in pipeline.log"
    log.warning(f"[ALERT] Sending validation anomaly alert ({total} issues).")
    send_email(subject, body)


def alert_stale_data(latest_date, staleness_days: int):
    """Call this when price data is too old."""
    if not ALERT_ON_STALE_DATA:
        return
    now     = datetime.now().strftime("%Y-%m-%d %H:%M")
    subject = f"[{TICKER} Pipeline] ⚠ STALE DATA — {staleness_days} days old ({now})"
    body    = (
        f"Price data appears stale.\n\n"
        f"Latest date in DB: {latest_date}\n"
        f"Days since update: {staleness_days}\n\n"
        f"Check your data source or yfinance connection."
    )
    log.warning(f"[ALERT] Stale data alert: {staleness_days} days since {latest_date}.")
    send_email(subject, body)


def alert_vrp_spike(vrp: float, threshold: float):
    """Call this when VRP is unusually large."""
    now     = datetime.now().strftime("%Y-%m-%d %H:%M")
    subject = f"[{TICKER} Pipeline] 📈 VRP SPIKE — {vrp:+.1%} ({now})"
    body    = (
        f"Volatility Risk Premium is unusually large.\n\n"
        f"VRP (IV - RVol):   {vrp:+.2%}\n"
        f"Alert threshold:   ±{threshold:.0%}\n"
        f"Interpretation:    {'Options appear RICH (sellers favored)' if vrp > 0 else 'Options appear CHEAP (buyers favored)'}\n\n"
        f"Review iv_surface.png and vol_analysis.png in the outputs/ folder."
    )
    log.warning(f"[ALERT] VRP spike alert: {vrp:+.2%}")
    send_email(subject, body)


def send_daily_summary(results: dict):
    """Send a daily results digest — runs even if no anomalies."""
    now     = datetime.now().strftime("%Y-%m-%d %H:%M")
    subject = f"[{TICKER} Pipeline] ✅ Daily Run Complete ({now})"

    vrp_data     = results.get("vrp_data")    or {}
    garch_params = results.get("garch_params") or {}
    rvol_df      = results.get("rvol_df")

    if rvol_df is not None and not rvol_df.empty:
        latest = rvol_df.iloc[-1]
        rvol_lines = (
            f"  10d RVol:  {latest.get('rvol_10d', float('nan')):.2%}\n"
            f"  21d RVol:  {latest.get('rvol_21d', float('nan')):.2%}\n"
            f"  63d RVol:  {latest.get('rvol_63d', float('nan')):.2%}"
        )
    else:
        rvol_lines = "  (not available)"

    garch_lines = (
        f"  Persistence: {garch_params.get('persistence', float('nan')):.4f}\n"
        f"  Long-run vol: {garch_params.get('longrun_vol', float('nan')):.2%}\n"
        f"  Current cond. vol: {garch_params.get('current_vol', float('nan')):.2%}"
        if garch_params else "  (not available)"
    )

    vrp_lines = (
        f"  ATM IV:   {vrp_data.get('atm_iv', float('nan')):.2%}\n"
        f"  21d RVol: {vrp_data.get('rvol_21d', float('nan')):.2%}\n"
        f"  VRP:      {vrp_data.get('vrp', float('nan')):+.2%}"
        if vrp_data else "  (not available)"
    )

    body = (
        f"SPY Volatility Pipeline — Daily Summary\n"
        f"Run time: {now}\n\n"
        f"─── Realized Volatility ───\n{rvol_lines}\n\n"
        f"─── GARCH(1,1) ───\n{garch_lines}\n\n"
        f"─── Volatility Risk Premium ───\n{vrp_lines}\n\n"
        f"Charts saved to outputs/ folder.\n"
    )

    send_email(subject, body)
