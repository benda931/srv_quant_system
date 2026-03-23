"""
agents/shared/agent_scheduler.py
---------------------------------
תזמון סוכנים מבוסס Cron — Agent Scheduler

תומך ב:
  - תזמון cron לכל סוכן
  - שרשרת תלויות (depends_on)
  - הרצה כ-subprocess
  - מצב --dry-run
  - לוגים בעברית

שימוש בספריית schedule (pip install schedule).
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import schedule

from agents.shared.agent_registry import get_registry, AgentStatus

# ── נתיבים ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent  # srv_quant_system/
SCHEDULE_CONFIG = ROOT / "config" / "agent_schedule.json"

log = logging.getLogger("agent_scheduler")

# ── לוח זמנים ברירת מחדל ────────────────────────────────────────────────────
# methodology: 06:00 ימי עבודה, optimizer: 07:00 ימי עבודה (תלוי ב-methodology)
# math: שני 08:00 שבועי
DEFAULT_SCHEDULE: dict[str, dict[str, Any]] = {
    "methodology": {
        "cron": "0 6 * * 1-5",
        "depends_on": [],
        "script": "agents/methodology/run.py",
        "description": "סוכן מתודולוגיה — סריקת שווקים יומית",
    },
    "optimizer": {
        "cron": "0 7 * * 1-5",
        "depends_on": ["methodology"],
        "script": "agents/optimizer/run.py",
        "description": "סוכן אופטימיזציה — מחכה לסיום מתודולוגיה",
    },
    "math": {
        "cron": "0 8 * * 1",
        "depends_on": [],
        "script": "agents/math/run.py",
        "description": "סוכן מתמטי — ניתוח שבועי (שני)",
    },
}


def load_schedule_config() -> dict[str, dict[str, Any]]:
    """
    טוען תצורת לוח זמנים מקובץ config/agent_schedule.json.
    אם הקובץ לא קיים — יוצר אותו עם ברירת מחדל.
    """
    if SCHEDULE_CONFIG.exists():
        try:
            return json.loads(SCHEDULE_CONFIG.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("שגיאה בקריאת תצורת לוח זמנים: %s — משתמש בברירת מחדל", exc)

    # יצירת קובץ ברירת מחדל
    SCHEDULE_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    SCHEDULE_CONFIG.write_text(
        json.dumps(DEFAULT_SCHEDULE, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("נוצר קובץ תצורת לוח זמנים: %s", SCHEDULE_CONFIG)
    return DEFAULT_SCHEDULE


# ── פענוח Cron פשוט ─────────────────────────────────────────────────────────
def _parse_cron_field(field: str, min_val: int, max_val: int) -> list[int]:
    """
    פענוח שדה cron בודד — תומך ב: *, */N, N, N-M, N,M,K
    """
    if field == "*":
        return list(range(min_val, max_val + 1))
    if field.startswith("*/"):
        step = int(field[2:])
        return list(range(min_val, max_val + 1, step))
    if "-" in field:
        start, end = field.split("-", 1)
        return list(range(int(start), int(end) + 1))
    if "," in field:
        return [int(x) for x in field.split(",")]
    return [int(field)]


def _parse_cron(expr: str) -> dict:
    """
    פענוח ביטוי cron מורחב: minute hour dom month dow
    תומך ב: *, */N, N, N-M, N,M,K לכל שדה.
    מחזיר dict עם minutes, hours, days_of_week (0=Mon..6=Sun).
    """
    parts = expr.strip().split()
    if len(parts) != 5:
        raise ValueError(f"ביטוי cron לא תקין: {expr!r}")

    minutes = _parse_cron_field(parts[0], 0, 59)
    hours = _parse_cron_field(parts[1], 0, 23)
    # dom, month — תמיכה בסיסית (רוב התזמונים לא משתמשים)
    dow_str = parts[4]

    # פענוח ימי שבוע — cron: 0=Sun..6=Sat  →  Python: 0=Mon..6=Sun
    if dow_str == "*":
        days_of_week = list(range(7))
    elif dow_str.startswith("*/"):
        step = int(dow_str[2:])
        days_of_week = list(range(0, 7, step))
    elif "-" in dow_str:
        start, end = dow_str.split("-")
        days_of_week = [
            (int(d) - 1) % 7 for d in range(int(start), int(end) + 1)
        ]
    elif "," in dow_str:
        days_of_week = [(int(d) - 1) % 7 for d in dow_str.split(",")]
    else:
        days_of_week = [(int(dow_str) - 1) % 7]

    # Backward compat: single minute/hour for simple expressions
    minute = minutes[0] if len(minutes) == 1 else minutes[0]
    hour = hours[0] if len(hours) == 1 else hours[0]

    return {
        "minute": minute, "hour": hour,
        "minutes": minutes, "hours": hours,
        "days_of_week": days_of_week,
    }


def _should_run_now(cron_expr: str) -> bool:
    """בודק אם הסוכן אמור לרוץ עכשיו (בדקה הנוכחית). תומך ב-*/N."""
    now = datetime.now()
    parsed = _parse_cron(cron_expr)
    return (
        now.hour in parsed.get("hours", [parsed["hour"]])
        and now.minute in parsed.get("minutes", [parsed["minute"]])
        and now.weekday() in parsed["days_of_week"]
    )


def _next_run_description(cron_expr: str) -> str:
    """מחזיר תיאור קריא של לוח הזמנים."""
    parsed = _parse_cron(cron_expr)
    day_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    days = ", ".join(day_names.get(d, "?") for d in sorted(parsed["days_of_week"]))
    return f"{parsed['hour']:02d}:{parsed['minute']:02d} on [{days}]"


# ── הרצת סוכן כ-subprocess ─────────────────────────────────────────────────
def run_agent(agent_name: str, script_path: str, dry_run: bool = False) -> bool:
    """
    מריץ סוכן כ-subprocess. מחזיר True אם הצליח.
    במצב dry_run — מדפיס בלבד בלי להריץ.
    """
    registry = get_registry()
    full_path = ROOT / script_path

    if dry_run:
        log.info("[DRY-RUN] היה מריץ: %s (%s)", agent_name, full_path)
        return True

    if not full_path.exists():
        log.warning("סקריפט לא נמצא: %s — מדלג על '%s'", full_path, agent_name)
        return False

    log.info("── מתחיל הרצת סוכן: %s ──", agent_name)
    registry.heartbeat(agent_name, AgentStatus.RUNNING)

    try:
        result = subprocess.run(
            [sys.executable, str(full_path)],
            capture_output=True,
            text=True,
            timeout=1800,  # 30 דקות מקסימום
            cwd=str(ROOT),
        )

        if result.returncode == 0:
            registry.heartbeat(agent_name, AgentStatus.COMPLETED)
            log.info("סוכן '%s' סיים בהצלחה (exit code 0)", agent_name)
            if result.stdout.strip():
                # מציג את 5 השורות האחרונות של הפלט
                last_lines = result.stdout.strip().split("\n")[-5:]
                for line in last_lines:
                    log.debug("  [%s] %s", agent_name, line)
            return True
        else:
            err_msg = result.stderr.strip()[-500:] if result.stderr else "no stderr"
            registry.heartbeat(agent_name, AgentStatus.FAILED, error=err_msg)
            log.error("סוכן '%s' נכשל (exit code %d): %s",
                      agent_name, result.returncode, err_msg[:200])
            return False

    except subprocess.TimeoutExpired:
        registry.heartbeat(agent_name, AgentStatus.FAILED, error="timeout after 30min")
        log.error("סוכן '%s' — timeout אחרי 30 דקות", agent_name)
        return False
    except Exception as exc:
        registry.heartbeat(agent_name, AgentStatus.FAILED, error=str(exc))
        log.error("שגיאה בהרצת סוכן '%s': %s", agent_name, exc)
        return False


# ── בדיקת תלויות ───────────────────────────────────────────────────────────
def _dependencies_met(agent_name: str, config: dict, completed_today: set[str]) -> bool:
    """
    בודק אם כל הסוכנים שה-agent תלוי בהם כבר סיימו היום.
    """
    agent_conf = config.get(agent_name, {})
    depends_on = agent_conf.get("depends_on", [])
    if not depends_on:
        return True

    for dep in depends_on:
        if dep not in completed_today:
            log.debug("תלות לא מולאה: %s ממתין ל-%s", agent_name, dep)
            return False
    return True


# ── לולאת תזמון ראשית ──────────────────────────────────────────────────────
class AgentScheduler:
    """
    מנוע תזמון — בודק כל דקה אילו סוכנים צריכים לרוץ.
    """

    def __init__(self, dry_run: bool = False) -> None:
        self.config = load_schedule_config()
        self.dry_run = dry_run
        # סוכנים שהושלמו היום — מתאפס בחצות
        self._completed_today: set[str] = set()
        self._last_check_date: Optional[str] = None

        # רישום כל הסוכנים ברג'יסטרי
        registry = get_registry()
        for name, conf in self.config.items():
            registry.register(name, conf.get("description", name))

    def _reset_daily(self) -> None:
        """איפוס רשימת ההשלמות בתחילת יום חדש."""
        today = datetime.now().strftime("%Y-%m-%d")
        if self._last_check_date != today:
            self._completed_today.clear()
            self._last_check_date = today
            log.info("── יום חדש: %s — איפוס רשימת השלמות ──", today)

    def check_and_run(self) -> None:
        """
        בודק אילו סוכנים צריכים לרוץ עכשיו ומריץ אותם.
        נקרא כל דקה מהלולאה הראשית.
        """
        self._reset_daily()

        for agent_name, agent_conf in self.config.items():
            cron_expr = agent_conf.get("cron", "")
            script = agent_conf.get("script", "")

            if not _should_run_now(cron_expr):
                continue

            # כבר רץ היום?
            if agent_name in self._completed_today:
                continue

            # בדיקת תלויות
            if not _dependencies_met(agent_name, self.config, self._completed_today):
                log.info("סוכן '%s' — תלויות לא מולאו, מדלג", agent_name)
                continue

            # הרצה
            success = run_agent(agent_name, script, dry_run=self.dry_run)
            if success:
                self._completed_today.add(agent_name)

    def run_all_once(self) -> dict[str, bool]:
        """
        מריץ את כל הסוכנים פעם אחת לפי סדר תלויות.
        מחזיר dict של {agent_name: success}.
        """
        results: dict[str, bool] = {}
        completed: set[str] = set()

        # מיון טופולוגי פשוט — סוכנים ללא תלויות קודם
        remaining = list(self.config.keys())
        max_iterations = len(remaining) * 2  # הגנה מפני לולאה אינסופית

        iteration = 0
        while remaining and iteration < max_iterations:
            iteration += 1
            for agent_name in list(remaining):
                if _dependencies_met(agent_name, self.config, completed):
                    script = self.config[agent_name].get("script", "")
                    success = run_agent(agent_name, script, dry_run=self.dry_run)
                    results[agent_name] = success
                    if success:
                        completed.add(agent_name)
                    remaining.remove(agent_name)

        # סוכנים שנשארו — תלויות לא נפתרו
        for agent_name in remaining:
            log.error("סוכן '%s' — תלויות לא ניתנות לפתרון", agent_name)
            results[agent_name] = False

        return results

    def run_single(self, agent_name: str) -> bool:
        """מריץ סוכן בודד לפי שם."""
        if agent_name not in self.config:
            log.error("סוכן '%s' לא נמצא בתצורה", agent_name)
            return False
        script = self.config[agent_name].get("script", "")
        return run_agent(agent_name, script, dry_run=self.dry_run)

    def start_loop(self) -> None:
        """
        לולאת תזמון ראשית — בודקת כל 30 שניות.
        """
        log.info("── מתחיל לולאת תזמון ──")
        log.info("סוכנים רשומים:")
        for name, conf in self.config.items():
            cron = conf.get("cron", "?")
            deps = conf.get("depends_on", [])
            desc = _next_run_description(cron)
            dep_str = f" (depends: {', '.join(deps)})" if deps else ""
            log.info("  %s: %s%s", name, desc, dep_str)

        if self.dry_run:
            log.info("── מצב DRY-RUN — לא יתבצעו הרצות בפועל ──")

        # תזמון בדיקה כל דקה באמצעות schedule
        schedule.every(30).seconds.do(self.check_and_run)

        # בדיקה ראשונית מיד
        self.check_and_run()

        try:
            while True:
                schedule.run_pending()
                time.sleep(10)
        except KeyboardInterrupt:
            log.info("── לולאת תזמון הופסקה (Ctrl+C) ──")


    def print_schedule(self) -> None:
        """הדפסת לוח הזמנים בצורה קריאה."""
        print(f"\nAgent Schedule (config: {SCHEDULE_CONFIG}):")
        print("=" * 65)
        for name, conf in self.config.items():
            cron = conf.get("cron", "?")
            deps = conf.get("depends_on", [])
            desc = _next_run_description(cron)
            dep_str = f"  depends: {', '.join(deps)}" if deps else ""
            script = conf.get("script", "?")
            print(f"  {name:<20} {desc:<30} {script}")
            if dep_str:
                print(f"  {'':<20} {dep_str}")
        print()


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sched = AgentScheduler(dry_run=True)
    sched.print_schedule()
