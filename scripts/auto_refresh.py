"""
scripts/auto_refresh.py
========================
Auto Data Refresh + VIX Alert Monitor

רץ כ-daemon/scheduled task ומבצע:
  1. עדכון נתונים אוטומטי — בדיקת freshness + fetch מ-FMP
  2. ניטור VIX — התראה על spikes חריגים
  3. הפעלת pipeline אחרי refresh
  4. שליחת alerts ל-agent_bus

לוח זמנים:
  - End-of-day refresh: 17:00 ET (22:00 UTC) ימי חול
  - VIX check: כל 30 דקות בזמן מסחר
  - Stale data check: כל 6 שעות

הרצה:
  python scripts/auto_refresh.py              # daemon mode
  python scripts/auto_refresh.py --once       # single refresh
  python scripts/auto_refresh.py --check-vix  # VIX check only
"""
from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone, date
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "auto_refresh.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("auto_refresh")


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
VIX_ALERT_LEVELS = {
    "ELEVATED": 20.0,     # VIX above 20 — caution
    "HIGH": 25.0,         # VIX above 25 — reduce exposure
    "EXTREME": 30.0,      # VIX above 30 — consider flatten
    "CRISIS": 35.0,       # VIX above 35 — hard kill
}

STALE_DATA_HOURS = 20    # Data older than this triggers refresh
CHECK_INTERVAL_MINUTES = 30  # VIX check interval


# ─────────────────────────────────────────────────────────────────────────────
# Data freshness check
# ─────────────────────────────────────────────────────────────────────────────
def check_data_freshness() -> dict:
    """בדיקת עדכניות הנתונים."""
    prices_path = ROOT / "data_lake" / "parquet" / "prices.parquet"
    status = {"fresh": False, "last_date": None, "hours_stale": None, "needs_refresh": True}

    if not prices_path.exists():
        log.warning("prices.parquet not found — needs initial fetch")
        return status

    try:
        import pandas as pd
        prices = pd.read_parquet(prices_path)
        last_date = prices.index[-1]
        now = datetime.now()

        # Calculate staleness
        if hasattr(last_date, "date"):
            last_dt = datetime.combine(last_date.date(), datetime.min.time())
        else:
            last_dt = datetime.now() - timedelta(hours=100)

        hours_stale = (now - last_dt).total_seconds() / 3600

        # Account for weekends: if today is Saturday/Sunday, data from Friday is fresh
        today = date.today()
        if today.weekday() >= 5:  # Weekend
            friday = today - timedelta(days=today.weekday() - 4)
            if last_date.date() >= friday:
                hours_stale = 0

        status["fresh"] = hours_stale < STALE_DATA_HOURS
        status["last_date"] = str(last_date.date()) if hasattr(last_date, "date") else str(last_date)
        status["hours_stale"] = round(hours_stale, 1)
        status["needs_refresh"] = not status["fresh"]

    except Exception as e:
        log.warning("Freshness check failed: %s", e)

    return status


# ─────────────────────────────────────────────────────────────────────────────
# Data refresh
# ─────────────────────────────────────────────────────────────────────────────
def refresh_data(force: bool = False) -> dict:
    """רענון נתונים מ-FMP API + הפעלת pipeline."""
    freshness = check_data_freshness()

    if not force and freshness["fresh"]:
        log.info("Data is fresh (last: %s, %s hours ago) — skipping refresh",
                 freshness["last_date"], freshness["hours_stale"])
        return {"status": "skipped", "reason": "data is fresh", **freshness}

    log.info("Starting data refresh (last: %s, %s hours stale)...",
             freshness["last_date"], freshness["hours_stale"])

    # Run the pipeline with force refresh
    try:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "run_all.py"), "--force-refresh"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=300,
        )
        if result.returncode == 0:
            log.info("Pipeline completed successfully")
            return {"status": "success", "output": result.stdout[-500:]}
        else:
            log.warning("Pipeline failed: %s", result.stderr[-300:])
            return {"status": "failed", "error": result.stderr[-300:]}
    except Exception as e:
        log.error("Pipeline execution error: %s", e)
        return {"status": "error", "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# VIX monitoring
# ─────────────────────────────────────────────────────────────────────────────
def check_vix() -> dict:
    """בדיקת רמת VIX והתראה על spikes."""
    try:
        import pandas as pd
        prices_path = ROOT / "data_lake" / "parquet" / "prices.parquet"
        if not prices_path.exists():
            return {"status": "no_data", "vix": None}

        prices = pd.read_parquet(prices_path)
        vix_col = "^VIX" if "^VIX" in prices.columns else "VIX" if "VIX" in prices.columns else None
        if not vix_col:
            return {"status": "no_vix_column", "vix": None}

        vix_current = float(prices[vix_col].dropna().iloc[-1])
        vix_prev = float(prices[vix_col].dropna().iloc[-2]) if len(prices[vix_col].dropna()) > 1 else vix_current
        vix_change = vix_current - vix_prev
        vix_change_pct = vix_change / vix_prev * 100 if vix_prev > 0 else 0

        # 20-day stats
        vix_20d = prices[vix_col].dropna().tail(20)
        vix_mean_20d = float(vix_20d.mean())
        vix_std_20d = float(vix_20d.std())
        vix_z = (vix_current - vix_mean_20d) / vix_std_20d if vix_std_20d > 0 else 0

        # Alert level
        alert_level = "NORMAL"
        for level, threshold in sorted(VIX_ALERT_LEVELS.items(), key=lambda x: -x[1]):
            if vix_current >= threshold:
                alert_level = level
                break

        # Spike detection: daily change > 3 points or > 15%
        is_spike = abs(vix_change) > 3.0 or abs(vix_change_pct) > 15.0

        result = {
            "status": "ok",
            "vix_current": round(vix_current, 2),
            "vix_prev": round(vix_prev, 2),
            "vix_change": round(vix_change, 2),
            "vix_change_pct": round(vix_change_pct, 1),
            "vix_z_20d": round(vix_z, 2),
            "vix_mean_20d": round(vix_mean_20d, 2),
            "alert_level": alert_level,
            "is_spike": is_spike,
            "date": str(prices.index[-1].date()),
        }

        # Publish to agent bus if alert
        if alert_level != "NORMAL" or is_spike:
            try:
                from scripts.agent_bus import AgentBus
                bus = AgentBus()
                bus.publish("vix_monitor", {
                    **result,
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "message": f"VIX {alert_level}: {vix_current:.1f} ({'spike!' if is_spike else 'elevated'})",
                })
                log.warning("VIX ALERT: %s at %.1f (change: %+.1f / %+.1f%%)",
                            alert_level, vix_current, vix_change, vix_change_pct)
            except ImportError:
                pass

        return result

    except Exception as e:
        log.error("VIX check failed: %s", e)
        return {"status": "error", "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Market hours check
# ─────────────────────────────────────────────────────────────────────────────
def is_market_hours() -> bool:
    """בדיקה אם השוק פתוח (ET timezone approximation)."""
    # UTC offset: ET = UTC-5 (EST) or UTC-4 (EDT)
    now_utc = datetime.now(timezone.utc)
    et_offset = -5  # Approximate
    now_et = now_utc + timedelta(hours=et_offset)

    # Market hours: 9:30-16:00 ET, Monday-Friday
    if now_et.weekday() >= 5:  # Weekend
        return False
    if now_et.hour < 9 or (now_et.hour == 9 and now_et.minute < 30):
        return False
    if now_et.hour >= 16:
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Daemon loop
# ─────────────────────────────────────────────────────────────────────────────
def run_daemon():
    """לולאת daemon — refresh אוטומטי + VIX monitoring."""
    log.info("=" * 60)
    log.info("Auto Refresh Daemon starting")
    log.info("=" * 60)

    last_refresh = datetime.min
    last_vix_check = datetime.min

    while True:
        now = datetime.now()

        # End-of-day refresh: after market close (~17:00 ET ≈ 22:00 UTC)
        hours_since_refresh = (now - last_refresh).total_seconds() / 3600
        if hours_since_refresh >= 6:
            freshness = check_data_freshness()
            if freshness["needs_refresh"]:
                log.info("Triggering end-of-day data refresh")
                refresh_data(force=True)
                last_refresh = now
            else:
                log.debug("Data fresh — no refresh needed (last: %s)", freshness["last_date"])

        # VIX check: every 30 minutes
        minutes_since_vix = (now - last_vix_check).total_seconds() / 60
        if minutes_since_vix >= CHECK_INTERVAL_MINUTES:
            vix_status = check_vix()
            if vix_status.get("status") == "ok":
                log.info("VIX: %.1f (%s) | change: %+.1f | z: %+.2f",
                         vix_status["vix_current"], vix_status["alert_level"],
                         vix_status["vix_change"], vix_status["vix_z_20d"])
            last_vix_check = now

        # Sleep 5 minutes
        time.sleep(300)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    if "--once" in sys.argv:
        freshness = check_data_freshness()
        print(f"Data freshness: {json.dumps(freshness, indent=2)}")
        if freshness["needs_refresh"] or "--force" in sys.argv:
            result = refresh_data(force="--force" in sys.argv)
            print(f"Refresh result: {result['status']}")
        vix = check_vix()
        if vix.get("status") == "ok":
            print(f"VIX: {vix['vix_current']:.1f} ({vix['alert_level']}) | "
                  f"change: {vix['vix_change']:+.1f} ({vix['vix_change_pct']:+.1f}%) | "
                  f"z: {vix['vix_z_20d']:+.2f}")
    elif "--check-vix" in sys.argv:
        vix = check_vix()
        print(json.dumps(vix, indent=2))
    elif "--check-freshness" in sys.argv:
        print(json.dumps(check_data_freshness(), indent=2))
    else:
        run_daemon()


if __name__ == "__main__":
    main()
