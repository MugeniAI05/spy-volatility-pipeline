#!/usr/bin/env python3
# =============================================================================
# run_pipeline.py — Master Orchestrator
#
# Runs all three stages in order:
#   1. ETL      (ingest_price_history + ingest_options_chain)
#   2. Validate (8 data quality checks)
#   3. Analyze  (realized vol, GARCH, IV surface, VRP, charts)
#
# Usage:
#   python run_pipeline.py              # run everything
#   python run_pipeline.py --etl-only  # just ingest data
#   python run_pipeline.py --no-email  # skip email alerts this run
# =============================================================================

import argparse
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path

# ── Bootstrap path so imports work from project root ──────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import LOG_PATH, DB_PATH, ALERT_VRP_THRESHOLD
from src.etl      import get_engine, run_etl
from src.validate import run_all_checks
from src.analyze  import run_analysis
from src.notifier import (
    alert_pipeline_failure, alert_validation_issues,
    alert_stale_data, alert_vrp_spike, send_daily_summary
)


# ── Logging ───────────────────────────────────────────────────────────────────
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH),
            logging.StreamHandler(sys.stdout),
        ],
    )

log = logging.getLogger(__name__)


# ── Main ──────────────────────────────────────────────────────────────────────
def main(args):
    setup_logging()
    start = datetime.now()

    log.info("=" * 70)
    log.info(f"  SPY VOLATILITY PIPELINE — {start.strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 70)

    engine = get_engine()

    # ── Stage 1: ETL ──────────────────────────────────────────────────────────
    log.info("\n▶  STAGE 1 / 3: ETL")
    try:
        price_df, options_df = run_etl(engine)
    except Exception as e:
        log.error(f"ETL stage crashed: {e}")
        log.error(traceback.format_exc())
        if not args.no_email:
            alert_pipeline_failure("ETL", e)
        sys.exit(1)

    if args.etl_only:
        log.info("--etl-only flag set. Stopping after ETL.")
        return

    # ── Stage 2: Validation ───────────────────────────────────────────────────
    log.info("\n▶  STAGE 2 / 3: VALIDATION")
    try:
        anomalies, total_issues, is_fresh = run_all_checks()
        if not args.no_email:
            if total_issues > 0:
                alert_validation_issues(anomalies, total_issues)
            if not is_fresh and price_df is not None and not price_df.empty:
                latest = price_df["date"].max() if "date" in price_df.columns else "unknown"
                staleness = (datetime.now().date() - 
                             datetime.strptime(str(latest), "%Y-%m-%d").date()).days if latest != "unknown" else -1
                alert_stale_data(latest, staleness)
    except Exception as e:
        log.error(f"Validation stage crashed: {e}")
        log.error(traceback.format_exc())
        if not args.no_email:
            alert_pipeline_failure("Validation", e)
        # Validation failure is non-fatal — continue to analysis
        log.warning("Continuing to analysis despite validation crash.")

    # ── Stage 3: Analysis ─────────────────────────────────────────────────────
    log.info("\n▶  STAGE 3 / 3: ANALYSIS")
    try:
        results = run_analysis(engine)
    except Exception as e:
        log.error(f"Analysis stage crashed: {e}")
        log.error(traceback.format_exc())
        if not args.no_email:
            alert_pipeline_failure("Analysis", e)
        sys.exit(1)

    # ── VRP alert ─────────────────────────────────────────────────────────────
    vrp_data = results.get("vrp_data")
    if vrp_data and not args.no_email:
        vrp = vrp_data.get("vrp", 0)
        if abs(vrp) > ALERT_VRP_THRESHOLD:
            alert_vrp_spike(vrp, ALERT_VRP_THRESHOLD)

    # ── Daily summary email ───────────────────────────────────────────────────
    if not args.no_email:
        send_daily_summary(results)

    # ── Done ──────────────────────────────────────────────────────────────────
    elapsed = (datetime.now() - start).total_seconds()
    log.info("=" * 70)
    log.info(f"  PIPELINE COMPLETE in {elapsed:.1f}s")
    log.info("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPY Volatility Pipeline")
    parser.add_argument(
        "--etl-only", action="store_true",
        help="Only run the ETL stage, skip validation and analysis."
    )
    parser.add_argument(
        "--no-email", action="store_true",
        help="Skip all email alerts for this run."
    )
    main(parser.parse_args())
