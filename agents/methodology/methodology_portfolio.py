"""
agents/methodology/methodology_portfolio.py
--------------------------------------------
מעקב היסטוריה של הרצות מתודולוגיה — שומר תוצאות, מנתח מגמות,
משווה ריצות ומזהה חולשות רג'ים.

Portfolio Tracker for methodology agent runs.
Stores up to 365 entries in a JSON file with thread-safe read/write.
"""
from __future__ import annotations

import json
import logging
import math
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# ── ברירות מחדל ──────────────────────────────────────────────────────────────
MAX_ENTRIES = 365
PORTFOLIO_FILE = Path(__file__).resolve().parent / "reports" / "portfolio.json"


class MethodologyPortfolio:
    """
    Thread-safe JSON-backed portfolio tracker for methodology run history.

    כל ריצה נשמרת עם: run_date, metrics, regime_breakdown,
    signal_stack_summary, parameters_snapshot.
    """

    def __init__(self, path: Path = PORTFOLIO_FILE, max_entries: int = MAX_ENTRIES) -> None:
        self._path = path
        self._max = max_entries
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ── קריאה / כתיבה בטוחה ──────────────────────────────────────────────────

    def _load(self) -> List[dict]:
        """טוען את ההיסטוריה מקובץ JSON."""
        if not self._path.exists():
            return []
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
            return []
        except Exception as exc:
            log.warning("Failed to load portfolio file: %s", exc)
            return []

    def _save(self, entries: List[dict]) -> None:
        """שומר את ההיסטוריה לקובץ JSON — מוגבל ב-max_entries."""
        trimmed = entries[-self._max:]
        self._path.write_text(
            json.dumps(trimmed, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    # ── ממשק ציבורי ──────────────────────────────────────────────────────────

    def add_run(self, run_result: dict) -> None:
        """הוספת ריצה חדשה להיסטוריה."""
        with self._lock:
            entries = self._load()
            entries.append(run_result)
            self._save(entries)
            log.debug(
                "MethodologyPortfolio: added run %s (total=%d)",
                run_result.get("run_date", "?"),
                len(entries),
            )

    def get_history(self, n: int = 30) -> List[dict]:
        """מחזיר את n הריצות האחרונות (ישן → חדש)."""
        with self._lock:
            entries = self._load()
            return entries[-n:]

    def get_trend(self, metric: str, n: int = 10) -> dict:
        """
        ניתוח מגמה של מדד מסוים על פני n ריצות אחרונות.

        Returns dict with: metric, values, direction (improving/declining/stable),
        slope, mean, last, first, delta_pct.
        """
        with self._lock:
            entries = self._load()

        recent = entries[-n:] if len(entries) >= n else entries
        values = []
        for entry in recent:
            metrics = entry.get("metrics", {})
            v = metrics.get(metric)
            if v is not None and _is_finite(v):
                values.append(float(v))

        if len(values) < 2:
            return {
                "metric": metric,
                "values": values,
                "direction": "insufficient_data",
                "slope": 0.0,
                "mean": values[0] if values else float("nan"),
                "last": values[-1] if values else float("nan"),
                "first": values[0] if values else float("nan"),
                "delta_pct": 0.0,
            }

        # חישוב שיפוע פשוט — ליניארי
        n_pts = len(values)
        x_mean = (n_pts - 1) / 2.0
        y_mean = sum(values) / n_pts
        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n_pts))
        slope = numerator / denominator if denominator > 1e-12 else 0.0

        # כיוון מגמה — מבוסס שיפוע מנורמל
        first_val = values[0]
        last_val = values[-1]
        delta_pct = ((last_val - first_val) / abs(first_val) * 100) if abs(first_val) > 1e-12 else 0.0

        # סף 3% לקביעת כיוון
        if delta_pct > 3.0:
            direction = "improving"
        elif delta_pct < -3.0:
            direction = "declining"
        else:
            direction = "stable"

        # עבור מדדים שליליים (כמו max_dd) — כיוון הפוך
        if metric in ("max_dd", "max_drawdown"):
            if direction == "improving":
                direction = "declining"  # max_dd going up is bad
            elif direction == "declining":
                direction = "improving"

        return {
            "metric": metric,
            "values": values,
            "direction": direction,
            "slope": round(slope, 6),
            "mean": round(y_mean, 6),
            "last": round(last_val, 6),
            "first": round(first_val, 6),
            "delta_pct": round(delta_pct, 2),
        }

    def compare_vs_previous(self, current: dict) -> dict:
        """
        השוואת ריצה נוכחית מול הריצה האחרונה בהיסטוריה.

        Returns dict with per-metric deltas.
        """
        with self._lock:
            entries = self._load()

        if not entries:
            return {"has_previous": False, "deltas": {}}

        previous = entries[-1]
        prev_metrics = previous.get("metrics", {})
        curr_metrics = current.get("metrics", {})

        deltas = {}
        for key in set(list(prev_metrics.keys()) + list(curr_metrics.keys())):
            old = prev_metrics.get(key)
            new = curr_metrics.get(key)
            if old is not None and new is not None and _is_finite(old) and _is_finite(new):
                old_f = float(old)
                new_f = float(new)
                abs_delta = new_f - old_f
                pct_delta = (abs_delta / abs(old_f) * 100) if abs(old_f) > 1e-12 else 0.0
                deltas[key] = {
                    "old": round(old_f, 6),
                    "new": round(new_f, 6),
                    "abs_delta": round(abs_delta, 6),
                    "pct_delta": round(pct_delta, 2),
                }

        return {
            "has_previous": True,
            "previous_date": previous.get("run_date", "?"),
            "deltas": deltas,
        }

    def best_run(self) -> dict:
        """מחזיר את הריצה עם ה-Sharpe הגבוה ביותר."""
        with self._lock:
            entries = self._load()

        if not entries:
            return {}

        best = None
        best_sharpe = -float("inf")
        for entry in entries:
            sharpe = entry.get("metrics", {}).get("sharpe")
            if sharpe is not None and _is_finite(sharpe) and float(sharpe) > best_sharpe:
                best_sharpe = float(sharpe)
                best = entry

        return best or {}

    def worst_regime(self) -> dict:
        """
        מזהה את הרג'ים שבעקביות הכי חלש — מבוסס על hit_rate ממוצע.

        Returns dict with: regime, avg_hit_rate, n_observations, consecutive_below_50.
        """
        with self._lock:
            entries = self._load()

        if not entries:
            return {}

        # אגרגציה לפי רג'ים על פני כל הריצות
        regime_stats: Dict[str, List[float]] = {}
        for entry in entries:
            breakdown = entry.get("regime_breakdown", {})
            for regime, regime_data in breakdown.items():
                hr = regime_data.get("hit_rate") or regime_data.get("ic_mean")
                if hr is not None and _is_finite(hr):
                    regime_stats.setdefault(regime, []).append(float(hr))

        if not regime_stats:
            return {}

        # מצא את הרג'ים עם ממוצע hit_rate הנמוך ביותר
        worst_regime = None
        worst_avg = float("inf")
        worst_data: Dict[str, Any] = {}

        for regime, hr_list in regime_stats.items():
            avg = sum(hr_list) / len(hr_list)

            # ספירת רצף ריצות מתחת ל-50%
            consecutive = 0
            max_consecutive = 0
            for hr in reversed(hr_list):
                if hr < 0.50:
                    consecutive += 1
                    max_consecutive = max(max_consecutive, consecutive)
                else:
                    break

            if avg < worst_avg:
                worst_avg = avg
                worst_regime = regime
                worst_data = {
                    "regime": regime,
                    "avg_hit_rate": round(avg, 4),
                    "n_observations": len(hr_list),
                    "consecutive_below_50": consecutive,
                    "max_consecutive_below_50": max_consecutive,
                    "last_3": [round(x, 4) for x in hr_list[-3:]],
                }

        return worst_data


# ── עזר ──────────────────────────────────────────────────────────────────────

def _is_finite(x: Any) -> bool:
    """בדיקה שערך הוא מספר סופי."""
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False
