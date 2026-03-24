"""
agents/architect/improvement_log.py
====================================
Tracks all Architect Agent scan cycles and improvements.
JSON-backed, thread-safe, singleton pattern (mirrors optimization_log.py).
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_LOG_PATH = Path(__file__).resolve().parent / "improvement_history.json"
_MAX_HISTORY = 500
_lock = threading.Lock()
_instance: Optional["ImprovementLog"] = None


class ImprovementLog:
    """Append-only log of Architect Agent improvement cycles."""

    def __init__(self, path: Path = _LOG_PATH):
        self._path = path
        self._data: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text(encoding="utf-8"))
            except Exception:
                self._data = []

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._data[-_MAX_HISTORY:], indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )

    # ── Public API ──────────────────────────────────────────────

    def log_cycle(self, entry: Dict[str, Any]) -> None:
        """Log a complete scan cycle."""
        with _lock:
            entry.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
            self._data.append(entry)
            self._save()

    def get_history(self, n: int = 50) -> List[Dict[str, Any]]:
        """Return last n cycles."""
        return self._data[-n:]

    def domain_success_rate(self, domain: str) -> float:
        """Fraction of cycles for this domain where validation passed."""
        relevant = [e for e in self._data if e.get("domain") == domain]
        if not relevant:
            return 0.0
        passed = sum(
            1 for e in relevant
            if e.get("validation_result", {}).get("tests_pass", False)
        )
        return passed / len(relevant)

    def last_scan_date(self, domain: str) -> Optional[str]:
        """ISO date of last scan for this domain, or None."""
        for entry in reversed(self._data):
            if entry.get("domain") == domain:
                return entry.get("timestamp", entry.get("cycle_id", ""))[:10]
        return None

    def get_pending_domains(self, all_domains: List[str], days_threshold: int = 7) -> List[str]:
        """Domains not scanned in last N days."""
        from datetime import timedelta
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days_threshold)).isoformat()[:10]
        pending = []
        for d in all_domains:
            last = self.last_scan_date(d)
            if last is None or last < cutoff:
                pending.append(d)
        return pending

    def total_improvements(self) -> int:
        """Total number of successful improvements made."""
        return sum(
            1 for e in self._data
            if e.get("validation_result", {}).get("tests_pass", False)
            and len(e.get("actions_taken", [])) > 0
        )

    def domain_stats(self) -> Dict[str, Dict[str, Any]]:
        """Per-domain statistics."""
        stats: Dict[str, Dict[str, Any]] = {}
        for entry in self._data:
            d = entry.get("domain", "unknown")
            if d not in stats:
                stats[d] = {"cycles": 0, "successes": 0, "last_scan": None}
            stats[d]["cycles"] += 1
            if entry.get("validation_result", {}).get("tests_pass", False):
                stats[d]["successes"] += 1
            stats[d]["last_scan"] = entry.get("timestamp", "")[:10]
        return stats


def get_improvement_log() -> ImprovementLog:
    """Singleton accessor."""
    global _instance
    if _instance is None:
        _instance = ImprovementLog()
    return _instance
