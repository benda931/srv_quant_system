"""
scripts/agent_watchdog.py
--------------------------
סוכן 3: Watchdog — שומר מערכת אוטומטי

מסתכל על:
  - שינויים בקבצי Python → מאתחל את ה-Dash app אוטומטית
  - גיל נתוני ה-Parquet → מרענן אם ישנים מדי
  - בריאות המערכת → שולח התראה לקובץ log אם CRITICAL

הרצה:
  python scripts/agent_watchdog.py
  python scripts/agent_watchdog.py --auto-refresh-hours 13
  python scripts/agent_watchdog.py --no-file-watch   (רק data refresh, ללא restart)
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "logs" / "watchdog.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("agent_watchdog")

# Files to watch for changes (triggers app restart)
WATCHED_FILES = [
    "main.py",
    "analytics/stat_arb.py",
    "analytics/attribution.py",
    "config/settings.py",
    "ui/panels.py",
    "ui/analytics_tabs.py",
    "data/pipeline.py",
]

_DASH_PROC: subprocess.Popen | None = None


def _mtimes() -> dict[str, float]:
    """Return last-modified times for all watched files."""
    result = {}
    for rel in WATCHED_FILES:
        p = ROOT / rel
        try:
            result[rel] = p.stat().st_mtime
        except FileNotFoundError:
            result[rel] = 0.0
    return result


def _data_age_hours() -> float:
    """Return age of prices.parquet in hours."""
    p = ROOT / "data_lake" / "parquet" / "prices.parquet"
    try:
        age = time.time() - p.stat().st_mtime
        return age / 3600.0
    except FileNotFoundError:
        return float("inf")


def _start_dash() -> subprocess.Popen:
    log.info("Starting Dash app...")
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "main.py")],
        cwd=str(ROOT),
    )
    log.info("Dash PID=%d", proc.pid)
    return proc


def _stop_dash(proc: subprocess.Popen) -> None:
    if proc and proc.poll() is None:
        log.info("Stopping Dash app (PID=%d)...", proc.pid)
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def _refresh_data() -> None:
    """Force-refresh Parquet data via the pipeline agent."""
    log.info("Auto-refreshing data (prices.parquet is stale)...")
    try:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "agent_daily_pipeline.py"), "--force-refresh"],
            cwd=str(ROOT),
            timeout=300,
        )
        if result.returncode == 0:
            log.info("Data refresh complete.")
        else:
            log.warning("Data refresh returned non-zero exit code: %d", result.returncode)
    except Exception as e:
        log.error("Data refresh failed: %s", e)


def run(
    poll_interval: int = 5,
    auto_refresh_hours: float = 13.0,
    watch_files: bool = True,
    start_dash: bool = True,
) -> None:
    """Main watchdog loop."""
    global _DASH_PROC

    log.info("Watchdog started. poll=%ds, refresh_after=%.0fh, watch=%s", poll_interval, auto_refresh_hours, watch_files)

    prev_mtimes = _mtimes()
    last_refresh_check = time.time()
    refresh_check_interval = 3600  # check data age every hour

    if start_dash:
        _DASH_PROC = _start_dash()

    try:
        while True:
            time.sleep(poll_interval)

            # ── File change detection → restart Dash ─────────────────────
            if watch_files:
                curr = _mtimes()
                changed = [f for f in WATCHED_FILES if curr.get(f, 0) != prev_mtimes.get(f, 0)]
                if changed:
                    log.info("Files changed: %s — restarting Dash...", changed)
                    _stop_dash(_DASH_PROC)
                    time.sleep(2)
                    _DASH_PROC = _start_dash()
                    prev_mtimes = curr

            # ── Dash crash detection → restart ───────────────────────────
            if start_dash and _DASH_PROC and _DASH_PROC.poll() is not None:
                log.warning("Dash app crashed (exit=%d). Restarting in 5s...", _DASH_PROC.returncode)
                time.sleep(5)
                _DASH_PROC = _start_dash()
                prev_mtimes = _mtimes()

            # ── Stale data detection → refresh ───────────────────────────
            now = time.time()
            if now - last_refresh_check > refresh_check_interval:
                last_refresh_check = now
                age = _data_age_hours()
                if age > auto_refresh_hours:
                    log.info("prices.parquet is %.1fh old — triggering refresh", age)
                    _refresh_data()
                    if start_dash:
                        log.info("Restarting Dash after data refresh...")
                        _stop_dash(_DASH_PROC)
                        time.sleep(3)
                        _DASH_PROC = _start_dash()
                        prev_mtimes = _mtimes()

    except KeyboardInterrupt:
        log.info("Watchdog stopped by user.")
        if start_dash:
            _stop_dash(_DASH_PROC)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SRV Watchdog Agent")
    parser.add_argument("--poll-interval",        type=int,   default=5,    help="Seconds between file checks")
    parser.add_argument("--auto-refresh-hours",   type=float, default=13.0, help="Refresh data if older than N hours")
    parser.add_argument("--no-file-watch",        action="store_true",      help="Disable file change detection")
    parser.add_argument("--no-dash",              action="store_true",      help="Don't start the Dash app (data refresh only)")
    args = parser.parse_args()

    run(
        poll_interval=args.poll_interval,
        auto_refresh_hours=args.auto_refresh_hours,
        watch_files=not args.no_file_watch,
        start_dash=not args.no_dash,
    )
