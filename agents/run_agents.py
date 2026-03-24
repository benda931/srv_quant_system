"""
agents/run_agents.py
--------------------
נקודת כניסה ראשית להרצת סוכנים — Main Entry Point

שימוש:
  python -m agents.run_agents                  # לולאת תזמון רציפה
  python -m agents.run_agents --once           # הרצה חד-פעמית של כל הסוכנים
  python -m agents.run_agents --agent math     # הרצת סוכן בודד
  python -m agents.run_agents --dry-run        # הדפסה בלבד — בלי הרצה בפועל
  python -m agents.run_agents --show-schedule  # הצגת לוח הזמנים
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ── וידוא שהפרויקט ב-sys.path ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent  # srv_quant_system/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.shared.agent_scheduler import AgentScheduler


def run_cycle():
    """Run one full agent cycle: Methodology -> Optimizer -> Math -> Architect with validation."""
    import json, shutil, subprocess, sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parent.parent  # srv_quant_system/
    reports_dir = ROOT / "agents" / "methodology" / "reports"
    settings_file = ROOT / "config" / "settings.py"

    log = logging.getLogger("agent_cycle")
    log.info("=== AGENT CYCLE START ===")

    # Step 1: Run Methodology Agent
    log.info("Step 1/4: Running Methodology Agent...")
    r1 = subprocess.run(
        [sys.executable, str(ROOT / "agents" / "methodology" / "agent_methodology.py"), "--once"],
        capture_output=True, text=True, timeout=600, cwd=str(ROOT)
    )
    if r1.returncode != 0:
        log.error("Methodology agent failed: %s", r1.stderr[-500:])
        return {"status": "FAILED", "step": "methodology", "error": r1.stderr[-200:]}
    log.info("Methodology OK")

    # Step 2: Run Optimizer Agent
    log.info("Step 2/4: Running Optimizer Agent...")
    # Backup settings first
    backup = settings_file.with_suffix(".py.cycle_backup")
    if settings_file.exists():
        shutil.copy2(settings_file, backup)

    r2 = subprocess.run(
        [sys.executable, str(ROOT / "agents" / "optimizer" / "agent_optimizer.py"), "--once"],
        capture_output=True, text=True, timeout=600, cwd=str(ROOT)
    )
    if r2.returncode != 0:
        log.warning("Optimizer failed (non-fatal): %s", r2.stderr[-200:])
    else:
        log.info("Optimizer OK")

    # Step 3: Run Math Agent
    log.info("Step 3/4: Running Math Agent...")
    r3 = subprocess.run(
        [sys.executable, str(ROOT / "agents" / "math" / "agent_math.py"), "--once"],
        capture_output=True, text=True, timeout=600, cwd=str(ROOT)
    )
    if r3.returncode != 0:
        log.warning("Math agent failed (non-fatal): %s", r3.stderr[-200:])
    else:
        log.info("Math OK")

    # Step 4: Architect
    log.info("Step 4/4: Running Architect Agent...")
    r4 = subprocess.run(
        [sys.executable, str(ROOT / "agents" / "architect" / "agent_architect.py"), "--once"],
        capture_output=True, text=True, timeout=600, cwd=str(ROOT)
    )
    if r4.returncode != 0:
        log.warning("Architect failed (non-fatal): %s", r4.stderr[-200:] if r4.stderr else "no output")
    else:
        log.info("Architect OK")

    log.info("=== AGENT CYCLE COMPLETE ===")
    return {
        "status": "OK",
        "methodology": r1.returncode == 0,
        "optimizer": r2.returncode == 0,
        "math": r3.returncode == 0,
        "architect": r4.returncode == 0,
    }


def setup_logging(verbose: bool = False) -> None:
    """הגדרת לוגים — קובץ + קונסולה."""
    log_file = ROOT / "logs" / "agent_scheduler.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    level = logging.DEBUG if verbose else logging.INFO

    # פורמט אחיד
    fmt = "%(asctime)s [%(name)s] %(levelname)s — %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    # Handler לקובץ
    file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    # Handler לקונסולה
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    # הגדרת root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


def parse_args() -> argparse.Namespace:
    """פענוח ארגומנטים מ-CLI."""
    parser = argparse.ArgumentParser(
        description="SRV Agent Scheduler — הרצה ותזמון סוכנים",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
דוגמאות:
  python -m agents.run_agents                  # לולאת תזמון
  python -m agents.run_agents --once           # הרצה חד-פעמית
  python -m agents.run_agents --agent math     # סוכן בודד
  python -m agents.run_agents --dry-run --once # סימולציה
        """,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="מצב סימולציה — הדפסה בלבד בלי הרצה בפועל",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="הרצה חד-פעמית של כל הסוכנים לפי סדר תלויות",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default=None,
        help="הרצת סוכן בודד לפי שם (methodology / optimizer / math)",
    )
    parser.add_argument(
        "--show-schedule",
        action="store_true",
        help="הצגת לוח הזמנים ויציאה",
    )
    parser.add_argument(
        "--cycle",
        action="store_true",
        help="הרצת מחזור סוכנים מלא: Methodology -> Optimizer -> Math",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="לוגים מפורטים (DEBUG)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(verbose=args.verbose)
    log = logging.getLogger("run_agents")

    log.info("══ SRV Agent Runner — starting ══")
    log.info("ROOT: %s", ROOT)

    scheduler = AgentScheduler(dry_run=args.dry_run)

    # ── מחזור סוכנים מלא ────────────────────────────────────────────────
    if args.cycle:
        log.info("מריץ מחזור סוכנים מלא (Methodology -> Optimizer -> Math -> Architect)")
        result = run_cycle()
        log.info("תוצאת מחזור: %s", result)
        return 0 if result.get("status") == "OK" else 1

    # ── הצגת לוח זמנים בלבד ────────────────────────────────────────────
    if args.show_schedule:
        scheduler.print_schedule()
        return 0

    # ── הרצת סוכן בודד ─────────────────────────────────────────────────
    if args.agent:
        log.info("מריץ סוכן בודד: %s", args.agent)
        success = scheduler.run_single(args.agent)
        return 0 if success else 1

    # ── הרצה חד-פעמית של כולם ──────────────────────────────────────────
    if args.once:
        log.info("מריץ את כל הסוכנים פעם אחת לפי סדר תלויות")
        results = scheduler.run_all_once()
        # סיכום
        log.info("── סיכום ──")
        for name, success in results.items():
            status = "OK" if success else "FAILED"
            log.info("  %s: %s", name, status)
        all_ok = all(results.values())
        return 0 if all_ok else 1

    # ── לולאת תזמון רציפה ──────────────────────────────────────────────
    log.info("מתחיל לולאת תזמון רציפה (Ctrl+C לעצירה)")
    scheduler.start_loop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
