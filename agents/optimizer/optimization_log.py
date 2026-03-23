"""
agents/optimizer/optimization_log.py
--------------------------------------
מעקב היסטוריית אופטימיזציה — Optimization Attempt Tracking

כל ניסיון אופטימיזציה (שינוי פרמטר, שינוי קוד) נרשם כאן עם:
  - מקור (optimizer / math)
  - סוג שינוי (param / code)
  - ערכים ישנים/חדשים
  - מטריקות לפני/אחרי
  - תוצאה (improved / reverted / failed)
  - דלתא Sharpe ו-IC

אחסון: agents/optimizer/optimization_history.json
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ── נתיבים ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
HISTORY_FILE = ROOT / "agents" / "optimizer" / "optimization_history.json"
MAX_HISTORY = 500  # מקסימום רשומות בקובץ — מונע תפיחה

log = logging.getLogger("optimization_log")


class OptimizationLog:
    """
    מנהל היסטוריית אופטימיזציה — שומר כל ניסיון שינוי בקובץ JSON.
    """

    def __init__(self, history_file: Path = HISTORY_FILE) -> None:
        self._path = history_file
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ── קריאה / כתיבה ─────────────────────────────────────────────────────
    def _load(self) -> list[dict]:
        """טוען היסטוריה מהדיסק."""
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    return data
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("שגיאה בקריאת היסטוריית אופטימיזציה: %s", exc)
        return []

    def _save(self, entries: list[dict]) -> None:
        """שומר היסטוריה לדיסק עם חיתוך אם נדרש."""
        # חיתוך — שומרים רק את N האחרונים
        if len(entries) > MAX_HISTORY:
            entries = entries[-MAX_HISTORY:]
        self._path.write_text(
            json.dumps(entries, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    # ── רישום ניסיון אופטימיזציה ──────────────────────────────────────────
    def log_attempt(self, entry: dict[str, Any]) -> None:
        """
        רושם ניסיון אופטימיזציה בודד.

        שדות מצופים (אבל לא נאכף — גמישות):
          timestamp       — נוצר אוטומטית אם חסר
          agent_source    — "optimizer" / "math"
          change_type     — "param" / "code"
          target_file     — קובץ יעד (e.g., "config/settings.py")
          param_name      — שם הפרמטר (לשינויי param)
          old_value       — ערך ישן
          new_value       — ערך חדש
          before_metrics  — מטריקות לפני {ic, sharpe, hit_rate, ...}
          after_metrics   — מטריקות אחרי
          outcome         — "improved" / "reverted" / "failed"
          delta_sharpe    — שינוי ב-Sharpe
          delta_ic        — שינוי ב-IC
        """
        entries = self._load()

        # הוספת timestamp אוטומטית אם חסר
        if "timestamp" not in entry:
            entry["timestamp"] = datetime.now(timezone.utc).isoformat()

        entries.append(entry)
        self._save(entries)
        log.info(
            "Optimization logged: %s → %s [%s] outcome=%s",
            entry.get("agent_source", "?"),
            entry.get("target_file", "?"),
            entry.get("change_type", "?"),
            entry.get("outcome", "?"),
        )

    # ── שאילתות ──────────────────────────────────────────────────────────
    def get_history(self, n: int = 50) -> list[dict]:
        """מחזיר את N הרשומות האחרונות (חדש ראשון)."""
        entries = self._load()
        return entries[-n:]

    def success_rate(self) -> float:
        """
        אחוז ניסיונות שהסתיימו בשיפור (improved) מתוך כלל הניסיונות.
        מחזיר 0.0 אם אין היסטוריה.
        """
        entries = self._load()
        if not entries:
            return 0.0
        improved = sum(1 for e in entries if e.get("outcome") == "improved")
        return improved / len(entries)

    def best_changes(self) -> list[dict]:
        """
        חמשת השינויים הטובים ביותר לפי delta_sharpe.
        ממוין מהגבוה לנמוך.
        """
        entries = self._load()
        # סינון רק שינויים עם delta_sharpe חוקי
        with_delta = [
            e for e in entries
            if isinstance(e.get("delta_sharpe"), (int, float))
        ]
        with_delta.sort(key=lambda e: e["delta_sharpe"], reverse=True)
        return with_delta[:5]

    def revert_reasons(self) -> dict[str, int]:
        """
        ספירת סיבות ביטול (revert) — מקבץ לפי שדה revert_reason.
        מחזיר dict {reason: count}.
        """
        entries = self._load()
        reasons: dict[str, int] = {}
        for e in entries:
            if e.get("outcome") == "reverted":
                reason = e.get("revert_reason", "unknown")
                reasons[reason] = reasons.get(reason, 0) + 1
        return reasons

    def recent_trend(self, n: int = 5) -> list[dict]:
        """
        מחזיר סיכום N הרצות האחרונות — לטרנד היסטורי.
        כולל רק שדות מפתח: timestamp, outcome, delta_sharpe, delta_ic.
        """
        entries = self._load()
        recent = entries[-n:]
        return [
            {
                "timestamp": e.get("timestamp"),
                "outcome": e.get("outcome"),
                "delta_sharpe": e.get("delta_sharpe"),
                "delta_ic": e.get("delta_ic"),
                "param_name": e.get("param_name"),
                "change_type": e.get("change_type"),
            }
            for e in recent
        ]


# ── Singleton ────────────────────────────────────────────────────────────────
_log_instance: Optional[OptimizationLog] = None


def get_optimization_log() -> OptimizationLog:
    """מחזיר singleton של לוג האופטימיזציה."""
    global _log_instance
    if _log_instance is None:
        _log_instance = OptimizationLog()
    return _log_instance


# ── CLI — הצגת סטטיסטיקות ──────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    opt_log = get_optimization_log()
    history = opt_log.get_history(n=100)

    if not history:
        print("No optimization history yet.")
        sys.exit(0)

    print(f"Optimization History ({HISTORY_FILE}):")
    print(f"  Total attempts:  {len(history)}")
    print(f"  Success rate:    {opt_log.success_rate():.1%}")
    print()

    best = opt_log.best_changes()
    if best:
        print("Top 5 changes by delta_sharpe:")
        for i, b in enumerate(best, 1):
            print(
                f"  {i}. {b.get('param_name', b.get('target_file', '?'))}"
                f"  Δsharpe={b.get('delta_sharpe', 0):+.4f}"
                f"  outcome={b.get('outcome', '?')}"
            )
    print()

    reasons = opt_log.revert_reasons()
    if reasons:
        print("Revert reasons:")
        for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"  {reason}: {count}")
