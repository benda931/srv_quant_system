"""
agents/orchestrator.py
========================
Master Agent Orchestrator — 24/7 daemon with full autonomous operation

Manages the complete agent lifecycle with governance, crash recovery,
parallel execution, and self-improvement:

  ┌────────────────────────────────────────────────────────────────┐
  │              Master Orchestrator (24/7 Daemon)                 │
  │                                                                │
  │  DATA LAYER:                                                   │
  │  ├─ Data Refresh ──────── every 6h ────────────────────────┐  │
  │  ├─ VIX Monitor ─────────  every 30min ────────────────────┤  │
  │  ├─ Data Scout ──────────  06:05 weekdays ─────────────────┤  │
  │                                                                │
  │  SIGNAL LAYER:                                                 │
  │  ├─ Methodology ─────────  06:00 weekdays ─────────────────┤  │
  │  ├─ Pair Scan ───────────  06:15 weekdays ─────────────────┤  │
  │  ├─ Regime Forecaster ───  every 2h (market hours) ────────┤  │
  │                                                                │
  │  OPTIMIZATION LAYER:                                           │
  │  ├─ Optimizer ───────────  07:00 weekdays ─────────────────┤  │
  │  ├─ Auto-Improve ────────  07:30 weekdays ─────────────────┤  │
  │  ├─ Math ────────────────  Monday 08:00 ───────────────────┤  │
  │  ├─ Architect ───────────  09:00 weekdays ─────────────────┤  │
  │                                                                │
  │  EXECUTION LAYER:                                              │
  │  ├─ Portfolio Construct ── 16:30 weekdays ─────────────────┤  │
  │  ├─ Risk Guardian ──────── 16:45 weekdays ─────────────────┤  │
  │  ├─ Paper Trading ──────── 17:00 weekdays ─────────────────┤  │
  │                                                                │
  │  REPORTING:                                                    │
  │  ├─ Morning Brief ──────── 06:30 weekdays ─────────────────┤  │
  │  ├─ Evening Recap ──────── 17:30 weekdays ─────────────────┤  │
  │  ├─ Alpha Research ──────  Monday 10:00 ───────────────────┤  │
  │  ├─ Weekly Backtest ─────  Monday 09:00 ───────────────────┤  │
  │                                                                │
  │  INFRASTRUCTURE:                                               │
  │  ├─ Health Check ────────  every 15min ────────────────────┤  │
  │  └─ Self-Heal ───────────  every 10min ────────────────────┘  │
  └────────────────────────────────────────────────────────────────┘

Features:
  - Registry governance enforcement (FROZEN/BLOCKED agents skipped)
  - Crash recovery on startup (detects incomplete yesterday tasks)
  - Parallel execution of independent agents (ThreadPool)
  - Slack dispatch of critical alerts
  - Regime-reactive scheduling (VIX spike → trigger risk guardian)
  - Self-healing: stale agent reset, retry with backoff

Run:
  python agents/orchestrator.py                # daemon mode
  python agents/orchestrator.py --once         # single full cycle
  python agents/orchestrator.py --status       # show all agent status
  python agents/orchestrator.py --run <agent>  # run specific agent
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

from logging.handlers import RotatingFileHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        RotatingFileHandler(
            LOG_DIR / "orchestrator.log",
            maxBytes=50 * 1024 * 1024, backupCount=5, encoding="utf-8",
        ),
    ],
)
log = logging.getLogger("orchestrator")

# Load API keys
for env_file in [ROOT / ".env", ROOT / "agents" / "credentials" / "api_keys.env"]:
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                v = v.strip().strip("'\"")
                if v:
                    os.environ.setdefault(k.strip(), v)


# ─────────────────────────────────────────────────────────────────────────────
# Task definitions
# ─────────────────────────────────────────────────────────────────────────────

TASKS = {
    "data_refresh": {
        "schedule": "every_6h",
        "description": "רענון נתונים + בדיקת freshness",
        "function": "_task_data_refresh",
    },
    "vix_monitor": {
        "schedule": "every_30min",
        "description": "ניטור VIX והתראות",
        "function": "_task_vix_monitor",
    },
    "methodology": {
        "schedule": "06:00_weekdays",
        "depends_on": ["data_refresh"],
        "description": "סוכן מתודולוגיה — pipeline + backtest + methodology lab",
        "function": "_task_methodology",
    },
    "optimizer": {
        "schedule": "07:00_weekdays",
        "depends_on": ["methodology"],
        "description": "סוכן אופטימיזר — כיול פרמטרים + שיפור קוד",
        "function": "_task_optimizer",
    },
    "math": {
        "schedule": "monday_08:00",
        "description": "סוכן מתמטיקה — שיפור פורמולות דרך Claude + GPT",
        "function": "_task_math",
    },
    "paper_trading": {
        "schedule": "17:00_weekdays",
        "depends_on": ["data_refresh"],
        "description": "עדכון פורטפוליו paper trading",
        "function": "_task_paper_trading",
    },
    "morning_brief": {
        "schedule": "06:30_weekdays",
        "depends_on": ["methodology"],
        "description": "דוח בוקר יומי",
        "function": "_task_morning_brief",
    },
    "evening_recap": {
        "schedule": "17:30_weekdays",
        "depends_on": ["paper_trading"],
        "description": "סיכום ערב + P&L",
        "function": "_task_evening_recap",
    },
    "daily_report": {
        "schedule": "18:00_weekdays",
        "depends_on": ["evening_recap"],
        "description": "דוח PM יומי — Slack + file",
        "function": "_task_daily_report",
    },
    "health_check": {
        "schedule": "every_15min",
        "description": "בדיקת בריאות מערכת",
        "function": "_task_health_check",
    },
    "weekly_backtest": {
        "schedule": "monday_09:00",
        "depends_on": ["methodology"],
        "description": "backtest שבועי מלא עם methodology lab (13 אסטרטגיות)",
        "function": "_task_weekly_backtest",
    },
    "alpha_research": {
        "schedule": "monday_10:00",
        "depends_on": ["weekly_backtest"],
        "description": "מחקר alpha: OOS validation + regime-adaptive + GPT suggestions",
        "function": "_task_alpha_research",
    },
    "pair_scan": {
        "schedule": "06:15_weekdays",
        "depends_on": ["data_refresh"],
        "description": "סריקת זוגות סקטורים — 55 pairs cointegration",
        "function": "_task_pair_scan",
    },
    "architect": {
        "schedule": "09:00_weekdays",
        "depends_on": ["optimizer"],
        "description": "סוכן ארכיטקט — שיפור שיטתי של המערכת",
        "function": "_task_architect",
    },
    # ── Missing agents now wired in ──────────────────────────────
    "auto_improve": {
        "schedule": "07:30_weekdays",
        "depends_on": ["methodology", "optimizer"],
        "description": "שיפור עצמי אוטונומי — Bayesian tuning + backtest validation",
        "function": "_task_auto_improve",
    },
    "data_scout": {
        "schedule": "06:05_weekdays",
        "description": "סריקת מקורות חיצוניים — FRED, VIX anomalies, correlation breaks",
        "function": "_task_data_scout",
    },
    "regime_forecaster": {
        "schedule": "every_2h_market",
        "description": "חיזוי משטר שוק — ML + regime-switching (כל 2 שעות בשעות מסחר)",
        "function": "_task_regime_forecaster",
    },
    "portfolio_construction": {
        "schedule": "16:30_weekdays",
        "depends_on": ["methodology", "optimizer"],
        "description": "בניית פורטפוליו — sector weights + hedge + sizing",
        "function": "_task_portfolio_construction",
    },
    "risk_guardian": {
        "schedule": "16:45_weekdays",
        "depends_on": ["portfolio_construction"],
        "description": "שמירת סיכון — VIX kills, credit stress, exposure limits",
        "function": "_task_risk_guardian",
    },
    "alpha_decay": {
        "schedule": "08:00_weekdays",
        "depends_on": ["methodology"],
        "description": "ניטור דעיכת אלפא — signal staleness + position aging",
        "function": "_task_alpha_decay",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class Orchestrator:
    """
    Master orchestrator — manages all agent tasks.

    Features:
      - State persistence across restarts
      - Retry logic (max 2 retries per task per day)
      - Failed dependency enforcement (blocks downstream)
      - Cumulative performance tracking
      - Alert file dispatch (data/alerts/)
      - Log rotation for bus + orchestrator
      - Self-healing (cleans stale agents from registry)
    """

    MAX_RETRIES = 2  # Max retries per task per day

    def __init__(self):
        self.completed_today: set = set()
        self.failed_today: set = set()       # Track failed tasks to block downstream
        self.retry_count: Dict[str, int] = {}  # Retry counter per task
        self.last_check_date: str = ""
        self.last_run: Dict[str, datetime] = {}
        self.errors: Dict[str, str] = {}
        self.cumulative_stats: Dict[str, Any] = {}  # Cumulative metrics over days
        self.state_path = ROOT / "data" / "orchestrator_state.json"
        self.alerts_dir = ROOT / "data" / "alerts"
        self.alerts_dir.mkdir(parents=True, exist_ok=True)

        # Load state
        self._load_state()

        # Bus
        try:
            from scripts.agent_bus import AgentBus
            self.bus = AgentBus()
        except ImportError:
            self.bus = None

        # Crash recovery — detect incomplete tasks from yesterday
        self._crash_recovery()

        # Log rotation on startup
        self._rotate_logs()

    def _load_state(self):
        if self.state_path.exists():
            try:
                state = json.loads(self.state_path.read_text(encoding="utf-8"))
                saved_date = state.get("last_check_date", "")
                today = date.today().isoformat()

                # Only restore completed_today if same day
                if saved_date == today:
                    self.completed_today = set(state.get("completed_today", []))
                    self.failed_today = set(state.get("failed_today", []))
                    self.retry_count = state.get("retry_count", {})
                else:
                    # New day — reset
                    self.completed_today = set()
                    self.failed_today = set()
                    self.retry_count = {}

                self.last_check_date = saved_date
                self.last_run = {
                    k: datetime.fromisoformat(v)
                    for k, v in state.get("last_run", {}).items()
                }
                self.cumulative_stats = state.get("cumulative_stats", {})
            except Exception as e:
                log.warning("State load failed: %s — starting fresh", e)

    def _save_state(self):
        state = {
            "last_check_date": self.last_check_date,
            "last_run": {k: v.isoformat() for k, v in self.last_run.items()},
            "completed_today": list(self.completed_today),
            "failed_today": list(self.failed_today),
            "retry_count": self.retry_count,
            "cumulative_stats": self.cumulative_stats,
            "errors": self.errors,
            "last_save": datetime.now(timezone.utc).isoformat(),
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")

    def _reset_daily(self):
        today = date.today().isoformat()
        if self.last_check_date != today:
            # Archive yesterday's stats
            if self.last_check_date:
                self.cumulative_stats[self.last_check_date] = {
                    "completed": list(self.completed_today),
                    "failed": list(self.failed_today),
                    "errors": dict(self.errors),
                }
                # Keep only last 90 days of cumulative stats
                if len(self.cumulative_stats) > 90:
                    oldest = sorted(self.cumulative_stats.keys())[:-90]
                    for k in oldest:
                        del self.cumulative_stats[k]

            self.completed_today.clear()
            self.failed_today.clear()
            self.retry_count.clear()
            self.errors.clear()
            self.last_check_date = today
            log.info("═══ New day: %s — daily reset ═══", today)
            self._save_state()

    def _crash_recovery(self):
        """
        Detect incomplete tasks from a previous crash.
        If the orchestrator crashed mid-day, some tasks may be marked as
        'running' in the registry or have no completion record. Log warnings
        and mark them for retry.
        """
        try:
            today = date.today().isoformat()
            if self.last_check_date == today:
                # Same day — check for tasks that should have run but didn't
                expected_daily = [
                    t for t, cfg in TASKS.items()
                    if "_weekdays" in cfg["schedule"] and t not in self.completed_today and t not in self.failed_today
                ]
                if expected_daily and datetime.now().hour >= 18:
                    # Past market close and daily tasks not done → likely crash
                    log.warning(
                        "Crash recovery: %d tasks expected but incomplete: %s",
                        len(expected_daily), expected_daily[:5],
                    )
                    self._dispatch_alert(
                        "WARNING", "Possible crash recovery",
                        f"{len(expected_daily)} tasks incomplete: {', '.join(expected_daily[:5])}",
                    )
            elif self.last_check_date:
                # Different day — check yesterday's results
                yesterday_stats = self.cumulative_stats.get(self.last_check_date, {})
                yesterday_failed = yesterday_stats.get("failed", [])
                if yesterday_failed:
                    log.info(
                        "Startup: yesterday (%s) had %d failed tasks: %s",
                        self.last_check_date, len(yesterday_failed), yesterday_failed,
                    )
        except Exception as e:
            log.debug("Crash recovery check failed: %s", e)

    def _rotate_logs(self):
        """Rotate large log files and trim bus."""
        # Bus retention: keep last 500 entries per agent
        bus_path = ROOT / "logs" / "agent_bus.json"
        if bus_path.exists():
            try:
                bus_data = json.loads(bus_path.read_text(encoding="utf-8"))
                trimmed = False
                for agent, entries in bus_data.items():
                    if len(entries) > 500:
                        bus_data[agent] = entries[-500:]
                        trimmed = True
                if trimmed:
                    bus_path.write_text(json.dumps(bus_data, indent=2, default=str), encoding="utf-8")
                    log.info("Bus log trimmed (retention: 500 per agent)")
            except Exception:
                pass

        # Archive large log files (>50MB)
        for log_file in (ROOT / "logs").glob("*.log"):
            try:
                if log_file.stat().st_size > 50 * 1024 * 1024:
                    archive = log_file.with_suffix(f".{date.today().isoformat()}.log.bak")
                    log_file.rename(archive)
                    log.info("Archived large log: %s → %s", log_file.name, archive.name)
            except Exception:
                pass

    def _dispatch_alert(self, level: str, title: str, message: str):
        """Write alert to file (data/alerts/) + bus."""
        alert = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": level,  # INFO / WARNING / CRITICAL
            "title": title,
            "message": message,
        }
        # File alert
        alert_file = self.alerts_dir / f"{date.today().isoformat()}_{level.lower()}.jsonl"
        with open(alert_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(alert, default=str) + "\n")

        # Bus
        if self.bus:
            self.bus.publish("orchestrator_alert", alert)

        log.log(
            logging.CRITICAL if level == "CRITICAL" else logging.WARNING if level == "WARNING" else logging.INFO,
            "🔔 [%s] %s: %s", level, title, message[:100],
        )

    def _self_heal(self):
        """Clean stale agents from registry + restart failed tasks with retries."""
        try:
            from agents.shared.agent_registry import get_registry, AgentStatus
            reg = get_registry()
            agents = reg.all_agents()
            for name, info in agents.items():
                status = info.get("status", "")
                if status == "STALE":
                    reg.heartbeat(name, AgentStatus.IDLE)
                    log.info("Self-heal: reset stale agent '%s' to IDLE", name)
        except Exception:
            pass

        # Retry failed tasks (max MAX_RETRIES per day)
        for task_name in list(self.failed_today):
            retries = self.retry_count.get(task_name, 0)
            if retries < self.MAX_RETRIES:
                log.info("Retry #%d for failed task: %s", retries + 1, task_name)
                self.failed_today.discard(task_name)
                self.retry_count[task_name] = retries + 1

    def _should_run(self, task_name: str) -> bool:
        """
        Check if a task should run now.
        Enforces: dependency chain, registry governance, schedule.
        """
        task = TASKS[task_name]
        schedule = task["schedule"]
        now = datetime.now()

        # ── Registry governance: skip FROZEN/BLOCKED/DISABLED agents ─────
        if not self._check_registry_governance(task_name):
            return False

        # ── Dependency enforcement ───────────────────────────────────────
        for dep in task.get("depends_on", []):
            if dep in self.failed_today:
                return False
            if dep not in self.completed_today:
                return False

        # Already completed today (for daily tasks)
        if task_name in self.completed_today and "every_" not in schedule:
            return False

        # Already failed and exhausted retries
        if task_name in self.failed_today:
            return False

        # ── Schedule checks ──────────────────────────────────────────────
        if schedule == "every_6h":
            last = self.last_run.get(task_name)
            return last is None or (now - last).total_seconds() > 6 * 3600

        elif schedule == "every_30min":
            last = self.last_run.get(task_name)
            return last is None or (now - last).total_seconds() > 30 * 60

        elif schedule == "every_15min":
            last = self.last_run.get(task_name)
            return last is None or (now - last).total_seconds() > 15 * 60

        elif schedule == "every_2h_market":
            # Every 2 hours during US market hours (9:30-16:00 ET ≈ 14:30-21:00 UTC)
            if now.weekday() >= 5:
                return False
            if not (6 <= now.hour <= 18):  # Local time proxy for market hours
                return False
            last = self.last_run.get(task_name)
            return last is None or (now - last).total_seconds() > 2 * 3600

        elif schedule.endswith("_weekdays"):
            if now.weekday() >= 5:
                return False
            time_str = schedule.split("_")[0]
            h, m = map(int, time_str.split(":"))
            target = now.replace(hour=h, minute=m, second=0)
            diff = abs((now - target).total_seconds())
            return diff < 300

        elif schedule.startswith("monday_"):
            if now.weekday() != 0:
                return False
            time_str = schedule.split("_")[1]
            h, m = map(int, time_str.split(":"))
            target = now.replace(hour=h, minute=m, second=0)
            diff = abs((now - target).total_seconds())
            return diff < 300

        return False

    def _check_registry_governance(self, task_name: str) -> bool:
        """Check agent registry — skip FROZEN/BLOCKED/DISABLED agents."""
        try:
            from agents.shared.agent_registry import get_registry
            reg = get_registry()
            info = reg.get_status(task_name)
            if info:
                if info.get("frozen"):
                    log.debug("Skipping %s: FROZEN (%s)", task_name, info.get("frozen_reason", ""))
                    return False
                if info.get("disabled"):
                    log.debug("Skipping %s: DISABLED (%s)", task_name, info.get("disabled_reason", ""))
                    return False
                runtime = info.get("runtime_state", "")
                if runtime in ("BLOCKED", "MAINTENANCE"):
                    log.debug("Skipping %s: %s", task_name, runtime)
                    return False
        except Exception:
            pass  # Registry unavailable — allow execution
        return True

    # ─── Task Implementations ─────────────────────────────────

    def _task_data_refresh(self) -> dict:
        """רענון נתונים מ-FMP + בדיקת freshness."""
        from scripts.auto_refresh import check_data_freshness, refresh_data
        freshness = check_data_freshness()
        if freshness["needs_refresh"]:
            result = refresh_data(force=True)
            return {"status": result.get("status"), "freshness": freshness}
        return {"status": "fresh", "freshness": freshness}

    def _task_vix_monitor(self) -> dict:
        """ניטור VIX + credit spreads + correlation + regime alerts."""
        from scripts.auto_refresh import check_vix
        vix = check_vix()

        # VIX alert
        if vix.get("alert_level") not in ("NORMAL", None):
            self._dispatch_alert(
                "CRITICAL" if vix.get("alert_level") in ("EXTREME", "CRISIS") else "WARNING",
                f"VIX {vix['alert_level']}",
                f"VIX at {vix.get('vix_current', 0):.1f} (change: {vix.get('vix_change', 0):+.1f})",
            )

        # VIX spike detection
        if vix.get("is_spike"):
            self._dispatch_alert("CRITICAL", "VIX SPIKE",
                                 f"VIX jumped {vix.get('vix_change', 0):+.1f} ({vix.get('vix_change_pct', 0):+.1f}%)")

        # Extended regime check — credit spreads, correlation
        try:
            import pandas as pd, numpy as np
            prices = pd.read_parquet(ROOT / "data_lake" / "parquet" / "prices.parquet")

            # Credit spread check (HYG/IEF ratio)
            if "HYG" in prices.columns and "IEF" in prices.columns:
                spread = np.log(prices["HYG"] / prices["IEF"]).dropna()
                if len(spread) >= 60:
                    mu = spread.tail(60).mean()
                    sd = spread.tail(60).std()
                    z = (spread.iloc[-1] - mu) / sd if sd > 1e-10 else 0
                    if z < -2.0:
                        self._dispatch_alert("WARNING", "Credit stress",
                                             f"HYG/IEF spread z={z:.2f} — credit widening")
                        vix["credit_z"] = round(float(z), 3)

            # Sector correlation check
            from config.settings import get_settings
            sectors = get_settings().sector_list()
            avail = [s for s in sectors if s in prices.columns]
            if len(avail) >= 5:
                rets = np.log(prices[avail] / prices[avail].shift(1)).dropna()
                C = rets.tail(20).corr()
                iu = np.triu_indices(len(avail), k=1)
                avg_corr = float(np.mean(C.values[iu]))
                vix["avg_corr_20d"] = round(avg_corr, 4)
                if avg_corr > 0.70:
                    self._dispatch_alert("WARNING", "High correlation",
                                         f"20d avg correlation={avg_corr:.3f} — dispersion collapsing")
        except Exception:
            pass

        return vix

    def _task_methodology(self) -> dict:
        """הרצת סוכן מתודולוגיה."""
        try:
            from agents.methodology.agent_methodology import run
            report = run(once=True)
            return {"status": "ok", "conclusions": len(report.get("conclusions", [])),
                    "errors": len(report.get("errors", []))}
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    def _task_optimizer(self) -> dict:
        """הרצת סוכן אופטימיזר."""
        try:
            from agents.optimizer.agent_optimizer import run
            run(once=True)
            return {"status": "ok"}
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    def _task_math(self) -> dict:
        """הרצת סוכן מתמטיקה."""
        try:
            from agents.math.agent_math import run
            run(once=True)
            return {"status": "ok"}
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    def _task_paper_trading(self) -> dict:
        """עדכון paper trading."""
        try:
            result = subprocess.run(
                [sys.executable, str(ROOT / "analytics" / "paper_trader.py"), "--update"],
                cwd=str(ROOT), capture_output=True, text=True, timeout=120,
            )
            return {"status": "ok" if result.returncode == 0 else "failed",
                    "output": result.stdout[-300:]}
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    def _task_morning_brief(self) -> dict:
        """יצירת דוח בוקר."""
        try:
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "run_all.py")],
                cwd=str(ROOT), capture_output=True, text=True, timeout=300,
            )
            return {"status": "ok" if result.returncode == 0 else "failed"}
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    def _task_evening_recap(self) -> dict:
        """סיכום ערב — P&L + trade status."""
        try:
            from analytics.paper_trader import PaperTrader
            trader = PaperTrader()
            trader.load()
            recap = {
                "total_pnl": trader.portfolio.total_pnl,
                "total_pnl_pct": trader.portfolio.total_pnl_pct,
                "n_positions": len(trader.portfolio.positions),
                "n_trades": trader.portfolio.n_trades,
                "win_rate": trader.portfolio.win_rate,
            }
            if self.bus:
                self.bus.publish("evening_recap", {
                    **recap,
                    "date": date.today().isoformat(),
                    "ts": datetime.now(timezone.utc).isoformat(),
                })
            return {"status": "ok", **recap}
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    def _task_alpha_research(self) -> dict:
        """מחקר alpha: OOS validation + regime-adaptive + GPT suggestions."""
        try:
            import pandas as pd
            from config.settings import get_settings
            from analytics.alpha_research import run_alpha_research
            prices = pd.read_parquet(ROOT / "data_lake" / "parquet" / "prices.parquet")
            settings = get_settings()
            report = run_alpha_research(prices, settings, include_gpt=True)

            result = {
                "status": "ok",
                "oos_sharpe": report.best_single_strategy.out_of_sample_sharpe,
                "oos_valid": report.best_single_strategy.is_valid,
                "regime_sharpes": report.regime_adaptive.regime_sharpes,
                "ensemble_sharpe": report.ensemble_sharpe,
                "n_gpt_suggestions": len(report.gpt_suggestions),
                "recommendations": report.recommendations[:3],
            }

            # Alert if positive OOS found
            if report.best_single_strategy.is_valid:
                self._dispatch_alert("INFO", "Positive OOS Alpha Found",
                    f"OOS Sharpe={report.best_single_strategy.out_of_sample_sharpe:.3f}, "
                    f"params={report.best_single_strategy.params}")

            return result
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    def _task_weekly_backtest(self) -> dict:
        """הרצת backtest שבועי מלא עם methodology lab."""
        try:
            import pandas as pd
            from analytics.methodology_lab import MethodologyLab
            prices = pd.read_parquet(ROOT / "data_lake" / "parquet" / "prices.parquet")
            lab = MethodologyLab(prices, step=10)
            results = lab.run_all()
            lab.save_results()

            best = max(results.values(), key=lambda r: r.sharpe) if results else None
            return {
                "status": "ok",
                "n_strategies": len(results),
                "best_name": best.name if best else "N/A",
                "best_sharpe": best.sharpe if best else 0,
            }
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    def _task_pair_scan(self) -> dict:
        """סריקת זוגות סקטורים לאותות cointegration."""
        try:
            import pandas as pd
            from analytics.pair_scanner import scan_pairs
            from config.settings import get_settings
            prices = pd.read_parquet(ROOT / "data_lake" / "parquet" / "prices.parquet")
            sectors = get_settings().sector_list()
            pairs = scan_pairs(prices, sectors)

            top = pairs[:5] if pairs else []
            result = {
                "status": "ok",
                "n_pairs_scanned": len(pairs),
                "top_signals": [
                    {"pair": p.pair_name, "z": p.spread_z, "hl": p.half_life,
                     "dir": p.direction, "strength": p.signal_strength}
                    for p in top
                ],
            }
            if self.bus:
                self.bus.publish("pair_scan", result)
            return result
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    def _task_architect(self) -> dict:
        """סוכן ארכיטקט — שיפור שיטתי של המערכת."""
        try:
            from agents.architect.agent_architect import run as run_architect
            run_architect(once=True)
            return {"status": "ok"}
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    def _task_auto_improve(self) -> dict:
        """שיפור עצמי אוטונומי — Bayesian tuning + backtest validation."""
        try:
            from agents.auto_improve.engine import AutoImproveEngine
            engine = AutoImproveEngine()

            # Governance: verify prerequisites
            prereq = engine.check_governance_prerequisites()
            if not prereq.get("ready", False):
                return {"status": "skipped", "reason": prereq.get("reason", "prerequisites not met")}

            # Run evaluation → identify weaknesses → suggest → test → promote
            eval_result = engine.run_evaluation()
            weaknesses = engine.identify_weaknesses(eval_result)
            suggestions = engine.generate_smart_suggestions(weaknesses)

            tested = 0
            promoted = 0
            for suggestion in suggestions[:5]:
                test_result = engine.test_suggestion(suggestion)
                tested += 1
                if test_result.get("improved"):
                    engine.promote_if_better(suggestion, test_result)
                    promoted += 1

            result = {
                "status": "ok",
                "weaknesses": len(weaknesses),
                "suggestions": len(suggestions),
                "tested": tested,
                "promoted": promoted,
            }

            if promoted > 0:
                self._dispatch_alert("INFO", "Auto-Improve: Parameters Promoted",
                    f"{promoted} improvements promoted to settings.py")

            return result
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    def _task_data_scout(self) -> dict:
        """סריקת מקורות חיצוניים — FRED, VIX anomalies, correlation breaks."""
        try:
            from agents.data_scout.agent_data_scout import run
            report = run(once=True)
            n_anomalies = len(report.get("anomalies", [])) if isinstance(report, dict) else 0

            if n_anomalies > 0:
                self._dispatch_alert("WARNING", "Data Scout: Anomalies Detected",
                    f"{n_anomalies} data anomalies found — check scout report")

            return {"status": "ok", "anomalies": n_anomalies}
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    def _task_regime_forecaster(self) -> dict:
        """חיזוי משטר שוק — ML + regime-switching (intraday refresh)."""
        try:
            from agents.regime_forecaster.agent_regime_forecaster import run
            result = run(once=True)
            regime = result.get("current_regime", "UNKNOWN") if isinstance(result, dict) else "UNKNOWN"
            confidence = result.get("confidence", 0) if isinstance(result, dict) else 0

            # Alert on regime change
            last_regime = self.cumulative_stats.get("_last_regime")
            if last_regime and regime != last_regime:
                self._dispatch_alert("WARNING", f"Regime Change: {last_regime} → {regime}",
                    f"Confidence: {confidence:.0%}. Review position sizing.")
            self.cumulative_stats["_last_regime"] = regime

            return {"status": "ok", "regime": regime, "confidence": confidence}
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    def _task_portfolio_construction(self) -> dict:
        """בניית פורטפוליו — sector weights + hedge + sizing."""
        try:
            from agents.portfolio_construction.agent_portfolio_construction import run
            result = run(once=True)
            return {"status": "ok", "result": str(result)[:200] if result else "empty"}
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    def _task_risk_guardian(self) -> dict:
        """שמירת סיכון — VIX kills, credit stress, exposure limits."""
        try:
            from agents.risk_guardian.agent_risk_guardian import run
            result = run(once=True)
            vetoes = result.get("vetoes", []) if isinstance(result, dict) else []
            if vetoes:
                self._dispatch_alert("CRITICAL", "Risk Guardian: Trade Vetoes",
                    f"{len(vetoes)} trades vetoed: {vetoes[:3]}")
            return {"status": "ok", "vetoes": len(vetoes)}
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    def _task_alpha_decay(self) -> dict:
        """ניטור דעיכת אלפא — signal staleness + position aging."""
        try:
            from agents.alpha_decay.agent_alpha_decay import run
            result = run(once=True)
            stale = result.get("stale_signals", 0) if isinstance(result, dict) else 0
            if stale > 3:
                self._dispatch_alert("WARNING", "Alpha Decay: Stale Signals",
                    f"{stale} signals showing decay — consider position reduction")
            return {"status": "ok", "stale_signals": stale}
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    def _task_daily_report(self) -> dict:
        """דוח PM יומי — Slack + file."""
        try:
            from scripts.daily_report import generate_daily_report
            report = generate_daily_report(preview=False)
            n_lines = len(report.split("\n"))
            return {"status": "ok", "lines": n_lines}
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    def _task_health_check(self) -> dict:
        """בדיקת בריאות מערכת."""
        health = {"status": "ok", "checks": {}}

        # 1. Data freshness
        try:
            from scripts.auto_refresh import check_data_freshness
            fresh = check_data_freshness()
            health["checks"]["data_fresh"] = fresh["fresh"]
        except Exception:
            health["checks"]["data_fresh"] = False

        # 2. DuckDB accessible
        try:
            from db.connection import get_connection
            conn = get_connection()
            health["checks"]["duckdb"] = conn is not None
        except Exception:
            health["checks"]["duckdb"] = False

        # 3. Agent registry health
        try:
            from agents.shared.agent_registry import get_registry
            reg = get_registry()
            agents = reg.all_agents()
            health["checks"]["agents"] = {
                name: info.get("status", "UNKNOWN")
                for name, info in agents.items()
            }
        except Exception:
            health["checks"]["agents"] = {}

        # 4. Disk space
        try:
            import shutil
            usage = shutil.disk_usage(str(ROOT))
            health["checks"]["disk_free_gb"] = round(usage.free / 1e9, 1)
            health["checks"]["disk_ok"] = usage.free > 1e9  # >1GB free
        except Exception:
            health["checks"]["disk_ok"] = True

        # 5. Paper portfolio
        try:
            pp = ROOT / "data" / "paper_portfolio.json"
            if pp.exists():
                data = json.loads(pp.read_text(encoding="utf-8"))
                health["checks"]["paper_positions"] = len(data.get("positions", []))
                health["checks"]["paper_pnl"] = data.get("total_pnl_pct", 0)
        except Exception:
            pass

        return health

    # ─── Run Cycle ────────────────────────────────────────────

    def run_task(self, task_name: str) -> dict:
        """Run a single task with logging, retry tracking, and alert dispatch."""
        task = TASKS.get(task_name)
        if not task:
            log.error("Unknown task: %s", task_name)
            return {"status": "unknown_task"}

        func_name = task["function"]
        func = getattr(self, func_name, None)
        if not func:
            log.error("Function not found: %s", func_name)
            return {"status": "no_function"}

        retry = self.retry_count.get(task_name, 0)
        retry_str = f" (retry #{retry})" if retry > 0 else ""
        log.info("▶ Running: %s — %s%s", task_name, task["description"], retry_str)
        t0 = time.time()
        try:
            result = func()
            elapsed = time.time() - t0
            result["elapsed_s"] = round(elapsed, 1)

            self.last_run[task_name] = datetime.now()
            status = result.get("status", "unknown")

            if status in ("ok", "fresh"):
                self.completed_today.add(task_name)
                self.failed_today.discard(task_name)
                log.info("✓ %s completed in %.1fs", task_name, elapsed)
            else:
                self.errors[task_name] = result.get("error", "unknown")
                self.failed_today.add(task_name)
                log.warning("✗ %s failed in %.1fs: %s", task_name, elapsed,
                            result.get("error", "")[:100])
                # Alert on failure
                self._dispatch_alert("WARNING", f"Task failed: {task_name}",
                                     f"{task['description']} — {result.get('error', 'unknown')[:200]}")

            # Publish to bus
            if self.bus:
                self.bus.publish(f"orchestrator_{task_name}", {
                    "task": task_name, "ts": datetime.now(timezone.utc).isoformat(),
                    "retry": retry, **result,
                })

            self._save_state()
            return result

        except Exception as e:
            elapsed = time.time() - t0
            log.exception("✗ %s crashed after %.1fs", task_name, elapsed)
            self.errors[task_name] = str(e)
            self.failed_today.add(task_name)
            self._dispatch_alert("CRITICAL", f"Task crashed: {task_name}", str(e)[:300])
            self._save_state()
            return {"status": "crashed", "error": str(e), "elapsed_s": round(elapsed, 1)}

    def run_once(self) -> dict:
        """Run all tasks once in dependency order (full daily pipeline)."""
        log.info("════════════════════════════════════════════════")
        log.info("  Orchestrator — Single Full Cycle (%d tasks)", len(TASKS))
        log.info("════════════════════════════════════════════════")

        self._reset_daily()
        results = {}

        # Full ordered execution with all agents
        # Phase 1: Data + Monitoring (can run in parallel)
        phase1 = ["data_refresh", "vix_monitor", "health_check", "data_scout"]
        self._run_parallel(phase1, results)

        # Phase 2: Signal generation
        phase2 = ["methodology", "pair_scan", "regime_forecaster"]
        self._run_parallel(phase2, results)

        # Phase 3: Briefs + optimization
        for t in ["morning_brief", "alpha_decay", "optimizer", "auto_improve"]:
            if t in TASKS:
                results[t] = self.run_task(t)

        # Phase 4: Weekly tasks (Monday only)
        if datetime.now().weekday() == 0:
            for t in ["math", "weekly_backtest", "alpha_research"]:
                if t in TASKS:
                    results[t] = self.run_task(t)

        # Phase 5: Execution pipeline
        for t in ["architect", "portfolio_construction", "risk_guardian", "paper_trading", "evening_recap"]:
            if t in TASKS:
                results[t] = self.run_task(t)

        # Summary
        n_ok = sum(1 for r in results.values() if r.get("status") in ("ok", "fresh", "skipped"))
        n_fail = sum(1 for r in results.values() if r.get("status") in ("failed", "crashed"))
        log.info("════════════════════════════════════════════════")
        log.info("  Cycle complete: %d/%d OK, %d failed", n_ok, len(results), n_fail)
        log.info("════════════════════════════════════════════════")

        # Slack dispatch summary
        self._try_slack_summary(results)

        return results

    def _run_parallel(self, task_names: List[str], results: dict):
        """Run independent tasks in parallel using threads."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        valid = [t for t in task_names if t in TASKS]
        if not valid:
            return

        with ThreadPoolExecutor(max_workers=min(len(valid), 4)) as pool:
            futures = {pool.submit(self.run_task, t): t for t in valid}
            for fut in as_completed(futures):
                t = futures[fut]
                try:
                    results[t] = fut.result()
                except Exception as e:
                    results[t] = {"status": "crashed", "error": str(e)}

    def _try_slack_summary(self, results: dict):
        """Send daily summary to Slack if configured."""
        try:
            from config.settings import get_settings
            settings = get_settings()
            webhook = getattr(settings, "slack_webhook_url", "")
            if not webhook:
                return

            n_ok = sum(1 for r in results.values() if r.get("status") in ("ok", "fresh", "skipped"))
            n_fail = sum(1 for r in results.values() if r.get("status") in ("failed", "crashed"))
            failed_names = [t for t, r in results.items() if r.get("status") in ("failed", "crashed")]

            lines = [
                f":robot_face: *SRV Agent Orchestrator* — {date.today().isoformat()}",
                f"Tasks: {n_ok}/{len(results)} OK | {n_fail} failed",
            ]
            if failed_names:
                lines.append(f":x: Failed: `{', '.join(failed_names)}`")

            # Key metrics from results
            if "auto_improve" in results and results["auto_improve"].get("promoted", 0) > 0:
                lines.append(f":chart_with_upwards_trend: Auto-improve promoted {results['auto_improve']['promoted']} improvements")
            if "regime_forecaster" in results:
                lines.append(f":traffic_light: Regime: `{results['regime_forecaster'].get('regime', '?')}`")

            import urllib.request
            payload = json.dumps({"text": "\n".join(lines)}).encode("utf-8")
            req = urllib.request.Request(webhook, data=payload,
                                         headers={"Content-Type": "application/json"}, method="POST")
            urllib.request.urlopen(req, timeout=10)
        except Exception:
            pass

    def run_daemon(self):
        """Run as continuous daemon — checks tasks every 60 seconds with self-healing."""
        log.info("════════════════════════════════════════════════════════")
        log.info("  SRV Agent Orchestrator — DAEMON MODE (24/7)")
        log.info("  Tasks: %d | Check: 60s | Retries: %d/task", len(TASKS), self.MAX_RETRIES)
        log.info("  Registry governance: ENFORCED")
        log.info("  Parallel execution: ENABLED (ThreadPool)")
        log.info("  Crash recovery: ENABLED")
        log.info("════════════════════════════════════════════════════════")

        cycle_count = 0
        last_vix_spike_check = datetime.now()

        while True:
            try:
                self._reset_daily()
                cycle_count += 1

                # Collect due tasks
                due_tasks = [t for t in TASKS if self._should_run(t)]

                # Separate independent tasks (no unmet deps) for parallel execution
                independent = []
                sequential = []
                for t in due_tasks:
                    deps = TASKS[t].get("depends_on", [])
                    unmet = [d for d in deps if d not in self.completed_today]
                    if not unmet:
                        independent.append(t)
                    else:
                        sequential.append(t)

                # Run independent tasks in parallel
                if len(independent) > 1:
                    results_buf = {}
                    self._run_parallel(independent, results_buf)
                elif independent:
                    self.run_task(independent[0])

                # Run sequential tasks one by one
                for t in sequential:
                    if self._should_run(t):  # Re-check after parallel phase
                        self.run_task(t)

                # VIX-reactive: if VIX spiked, force risk guardian
                if (datetime.now() - last_vix_spike_check).total_seconds() > 300:
                    last_vix_spike_check = datetime.now()
                    self._vix_reactive_check()

                # Self-heal every 10 cycles (10 minutes)
                if cycle_count % 10 == 0:
                    self._self_heal()

                # Log rotation every 100 cycles (~1.5 hours)
                if cycle_count % 100 == 0:
                    self._rotate_logs()

                self._save_state()
                time.sleep(60)

            except KeyboardInterrupt:
                log.info("Daemon stopped by user (Ctrl+C)")
                self._dispatch_alert("INFO", "Orchestrator stopped", "Manual shutdown via Ctrl+C")
                break
            except Exception as e:
                log.exception("Daemon error: %s", e)
                self._dispatch_alert("CRITICAL", "Daemon error", str(e)[:300])
                time.sleep(60)

    def _vix_reactive_check(self):
        """
        If VIX spiked significantly, trigger risk_guardian out of schedule.
        This provides intraday risk protection for short-vol positions.
        """
        try:
            import pandas as pd
            prices = pd.read_parquet(ROOT / "data_lake" / "parquet" / "prices.parquet")
            vix_col = next((c for c in prices.columns if "VIX" in c.upper()), None)
            if not vix_col:
                return

            vix = prices[vix_col].dropna()
            if len(vix) < 5:
                return

            vix_now = float(vix.iloc[-1])
            vix_prev = float(vix.iloc[-2])
            vix_change = vix_now - vix_prev

            # Trigger risk guardian on VIX spike > +3 points
            if vix_change > 3.0 and "risk_guardian" not in self.completed_today:
                log.warning("VIX SPIKE: %.1f → %.1f (+%.1f) — triggering risk guardian",
                           vix_prev, vix_now, vix_change)
                self._dispatch_alert("CRITICAL", "VIX Spike → Risk Guardian Triggered",
                    f"VIX {vix_prev:.1f} → {vix_now:.1f} (+{vix_change:.1f})")
                self.run_task("risk_guardian")
        except Exception:
            pass

    def status(self) -> str:
        """Human-readable status report with cumulative stats."""
        lines = [
            "════════════════════════════════════════════════════",
            "  SRV Agent Orchestrator — Status Report",
            f"  Date: {date.today().isoformat()}",
            f"  Uptime days tracked: {len(self.cumulative_stats)}",
            "════════════════════════════════════════════════════",
            "",
        ]

        # Task status
        lines.append("  TASKS:")
        for name, task in TASKS.items():
            last = self.last_run.get(name)
            last_str = last.strftime("%H:%M") if last else "never"
            retries = self.retry_count.get(name, 0)

            if name in self.completed_today:
                icon = "✓"
            elif name in self.failed_today:
                icon = "✗"
            else:
                icon = "○"

            err = self.errors.get(name, "")
            parts = [f"    {icon} {name:<20} last={last_str:<6} sched={task['schedule']:<16}"]
            if retries > 0:
                parts.append(f" retries={retries}")
            if err:
                parts.append(f" err={err[:30]}")
            lines.append("".join(parts))

        # Failure summary
        if self.failed_today:
            lines.append(f"\n  ⚠ FAILED TODAY: {', '.join(self.failed_today)}")

        # Health
        health = self._task_health_check()
        lines.append("\n  HEALTH:")
        for k, v in health.get("checks", {}).items():
            if isinstance(v, dict):
                lines.append(f"    {k}: {json.dumps(v, default=str)[:60]}")
            else:
                icon = "✓" if v else "✗"
                lines.append(f"    {icon} {k}: {v}")

        # Alerts today
        alert_file = self.alerts_dir / f"{date.today().isoformat()}_warning.jsonl"
        alert_file_c = self.alerts_dir / f"{date.today().isoformat()}_critical.jsonl"
        n_alerts = 0
        for af in [alert_file, alert_file_c]:
            if af.exists():
                n_alerts += sum(1 for _ in af.read_text(encoding="utf-8").strip().split("\n") if _)
        if n_alerts:
            lines.append(f"\n  🔔 ALERTS TODAY: {n_alerts}")

        # Cumulative stats (last 7 days)
        recent_days = sorted(self.cumulative_stats.keys())[-7:]
        if recent_days:
            lines.append("\n  WEEKLY SUMMARY (last 7 days):")
            total_completed = 0
            total_failed = 0
            for d in recent_days:
                day_stats = self.cumulative_stats[d]
                n_c = len(day_stats.get("completed", []))
                n_f = len(day_stats.get("failed", []))
                total_completed += n_c
                total_failed += n_f
                icon = "✓" if n_f == 0 else "⚠"
                lines.append(f"    {icon} {d}: {n_c} completed, {n_f} failed")
            lines.append(f"    Total: {total_completed} completed, {total_failed} failed "
                         f"({total_completed/(total_completed+total_failed)*100:.0f}% success)" if total_completed + total_failed > 0 else "")

        lines.extend(["", "════════════════════════════════════════════════════"])
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    orch = Orchestrator()

    if "--status" in sys.argv:
        print(orch.status())
        return

    if "--once" in sys.argv:
        orch.run_once()
        return

    if "--run" in sys.argv:
        idx = sys.argv.index("--run")
        if idx + 1 < len(sys.argv):
            task = sys.argv[idx + 1]
            orch.run_task(task)
        else:
            print("Usage: --run <task_name>")
            print(f"Available: {list(TASKS.keys())}")
        return

    # Default: daemon mode
    orch.run_daemon()


if __name__ == "__main__":
    main()
