"""
scripts/promote_params.py
===========================
Parameter Promotion Pipeline — calibrate → backtest → validate → promote

Reads calibration results from agents/methodology/reports/calibration_*.json
and applies the best parameters to config/settings.py if they pass validation.

Validation criteria:
  - Sharpe > 0 (must be positive)
  - Win rate > 50%
  - Max drawdown > -5%
  - At least 50 trades in backtest

Usage:
  python scripts/promote_params.py --check    # Show what would change
  python scripts/promote_params.py --apply    # Apply if validation passes
  python scripts/promote_params.py --revert   # Revert to backup
"""
from __future__ import annotations

import json
import logging
import shutil
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

log = logging.getLogger("promote_params")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")

CALIBRATION_DIR = ROOT / "agents" / "methodology" / "reports"
SETTINGS_PATH = ROOT / "config" / "settings.py"
BACKUP_PATH = SETTINGS_PATH.with_suffix(".py.pre_promote_bak")


def load_latest_calibration() -> Optional[dict]:
    """Load the most recent calibration results."""
    files = sorted(CALIBRATION_DIR.glob("calibration_*.json"), reverse=True)
    if not files:
        log.warning("No calibration files found in %s", CALIBRATION_DIR)
        return None
    path = files[0]
    log.info("Loading calibration: %s", path.name)
    return json.loads(path.read_text(encoding="utf-8"))


def validate_strategy(strategy: dict) -> tuple:
    """Validate a calibrated strategy meets promotion criteria."""
    sharpe = strategy.get("sharpe", -999)
    wr = strategy.get("win_rate", 0)
    dd = strategy.get("max_dd", -1)
    trades = strategy.get("trades", 0)

    issues = []
    if sharpe <= 0:
        issues.append(f"Sharpe {sharpe:.3f} <= 0")
    if wr <= 0.50:
        issues.append(f"Win rate {wr:.1%} <= 50%")
    if dd < -0.05:
        issues.append(f"Max DD {dd:.2%} < -5%")
    if trades < 50:
        issues.append(f"Only {trades} trades (need >= 50)")

    return len(issues) == 0, issues


def apply_param(content: str, param: str, value: Any) -> str:
    """Replace a parameter value in settings.py content."""
    import re
    # Match: param_name: type = Field(default=VALUE, ...)
    pattern = rf"({param}\s*:\s*\w+\s*=\s*Field\(default=)([^,\)]+)"
    match = re.search(pattern, content)
    if match:
        old_val = match.group(2).strip()
        new_content = content[:match.start(2)] + str(value) + content[match.end(2):]
        log.info("  %s: %s → %s", param, old_val, value)
        return new_content
    else:
        log.warning("  %s: NOT FOUND in settings.py", param)
        return content


def check_promotion(calibration: dict) -> dict:
    """Check what would be promoted without applying."""
    report = {"promotable": [], "rejected": []}

    for strategy_name, strategy_data in calibration.get("best_strategies", {}).items():
        valid, issues = validate_strategy(strategy_data)
        entry = {
            "name": strategy_name,
            "sharpe": strategy_data.get("sharpe"),
            "win_rate": strategy_data.get("win_rate"),
            "params": strategy_data.get("params", {}),
        }
        if valid:
            report["promotable"].append(entry)
        else:
            entry["issues"] = issues
            report["rejected"].append(entry)

    # Settings that would change
    if calibration.get("settings_updated"):
        report["settings_changes"] = calibration["settings_updated"]

    return report


def apply_promotion(calibration: dict) -> bool:
    """Apply validated parameters to settings.py."""
    settings_to_apply = calibration.get("settings_updated", {})
    if not settings_to_apply:
        log.warning("No settings_updated in calibration — nothing to apply")
        return False

    # Validate at least one strategy
    any_valid = False
    for _, strat in calibration.get("best_strategies", {}).items():
        valid, _ = validate_strategy(strat)
        if valid:
            any_valid = True
            break

    if not any_valid:
        log.warning("No strategy passes validation — skipping promotion")
        return False

    # Backup
    shutil.copy2(SETTINGS_PATH, BACKUP_PATH)
    log.info("Backup created: %s", BACKUP_PATH.name)

    # Apply
    content = SETTINGS_PATH.read_text(encoding="utf-8")
    for param, value in settings_to_apply.items():
        content = apply_param(content, param, value)

    SETTINGS_PATH.write_text(content, encoding="utf-8")

    # Validate import
    try:
        import importlib
        from config import settings as _s
        importlib.reload(_s)
        _s.Settings()
        log.info("✓ Settings validated after promotion")
        return True
    except Exception as e:
        log.error("✗ Settings validation FAILED — reverting: %s", e)
        shutil.copy2(BACKUP_PATH, SETTINGS_PATH)
        return False


def revert():
    """Revert to pre-promotion backup."""
    if BACKUP_PATH.exists():
        shutil.copy2(BACKUP_PATH, SETTINGS_PATH)
        log.info("Reverted to backup: %s", BACKUP_PATH.name)
    else:
        log.warning("No backup found at %s", BACKUP_PATH)


def main():
    calibration = load_latest_calibration()
    if not calibration:
        return

    if "--check" in sys.argv:
        report = check_promotion(calibration)
        print(json.dumps(report, indent=2, default=str))

    elif "--apply" in sys.argv:
        success = apply_promotion(calibration)
        print(f"Promotion: {'SUCCESS' if success else 'FAILED'}")

    elif "--revert" in sys.argv:
        revert()

    else:
        print("Usage: --check | --apply | --revert")


if __name__ == "__main__":
    main()
