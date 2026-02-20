#!/usr/bin/env python3
# =============================================================================
# scheduler.py — Daily Market-Close Scheduler
#
# Keeps running in the background and triggers run_pipeline.py
# at SCHEDULE_TIME every weekday (Mon–Fri), skipping weekends.
#
# Start it once and leave it running:
#   python scheduler.py
#
# To run it in the background so VSCode can close:
#   Windows:  start /B python scheduler.py
#   Mac/Linux: nohup python scheduler.py &
#
# Stop it: Ctrl+C in the terminal, or kill the process.
# =============================================================================

import logging
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import SCHEDULE_TIME, LOG_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("scheduler")


def parse_schedule_time(t: str) -> tuple[int, int]:
    """Parse 'HH:MM' → (hour, minute)."""
    h, m = t.split(":")
    return int(h), int(m)


def next_run_datetime(hour: int, minute: int) -> datetime:
    """
    Return the next datetime when the pipeline should run.
    Skips weekends (Saturday=5, Sunday=6).
    """
    now  = datetime.now()
    candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

    # If today's scheduled time has already passed, move to tomorrow
    if candidate <= now:
        candidate += timedelta(days=1)

    # Skip weekends
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)

    return candidate


def run_pipeline():
    log.info("Triggering run_pipeline.py ...")
    try:
        result = subprocess.run(
            [sys.executable, str(ROOT / "run_pipeline.py")],
            capture_output=False,   # let output stream to the terminal
            cwd=str(ROOT),
        )
        if result.returncode == 0:
            log.info("run_pipeline.py completed successfully.")
        else:
            log.error(f"run_pipeline.py exited with code {result.returncode}.")
    except Exception as e:
        log.error(f"Failed to launch run_pipeline.py: {e}")


def main():
    hour, minute = parse_schedule_time(SCHEDULE_TIME)

    log.info("=" * 60)
    log.info(f"  SPY Volatility Pipeline Scheduler started")
    log.info(f"  Scheduled run time: {SCHEDULE_TIME} (local time, weekdays only)")
    log.info("  Press Ctrl+C to stop.")
    log.info("=" * 60)

    while True:
        next_run = next_run_datetime(hour, minute)
        wait_secs = (next_run - datetime.now()).total_seconds()

        log.info(
            f"Next run: {next_run.strftime('%Y-%m-%d %H:%M')} "
            f"(in {wait_secs / 3600:.1f} hours)"
        )

        # Sleep in small chunks so Ctrl+C is responsive
        while (datetime.now() < next_run):
            time.sleep(min(30, (next_run - datetime.now()).total_seconds()))

        # Double-check it's still a weekday (in case of clock drift)
        if datetime.now().weekday() < 5:
            run_pipeline()
        else:
            log.info("Weekend detected — skipping run.")

        # Small buffer before computing next run
        time.sleep(60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("Scheduler stopped by user.")
