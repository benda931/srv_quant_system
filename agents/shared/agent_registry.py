"""
agents/shared/agent_registry.py
-------------------------------
רישום ומעקב בריאות סוכנים — Agent Registration & Health Tracking

כל סוכן נרשם פעם אחת ומדווח heartbeat בכל ריצה.
סוכן שלא דיווח 30 דקות נחשב STALE.
המצב נשמר ב-logs/agent_registry.json.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Optional

# ── נתיבים ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent  # srv_quant_system/
REGISTRY_FILE = ROOT / "logs" / "agent_registry.json"

# ── סטאטוסים אפשריים לסוכן ────────────────────────────────────────────────
class AgentStatus(str, Enum):
    IDLE      = "IDLE"
    RUNNING   = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED    = "FAILED"
    STALE     = "STALE"      # לא דיווח heartbeat מעל 30 דקות

# ── סף זמן לסטאטוס STALE ──────────────────────────────────────────────────
STALE_THRESHOLD = timedelta(minutes=30)

log = logging.getLogger("agent_registry")


class AgentRegistry:
    """
    מנהל רישום סוכנים — שומר מצב ב-JSON, תומך ב-heartbeat ובדיקות בריאות.
    """

    def __init__(self, registry_file: Path = REGISTRY_FILE) -> None:
        self._path = registry_file
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ── קריאה / כתיבה ─────────────────────────────────────────────────────
    def _load(self) -> dict[str, Any]:
        """טוען את קובץ הרישום מהדיסק."""
        if self._path.exists():
            try:
                return json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("שגיאה בקריאת רג'יסטרי: %s", exc)
                return {}
        return {}

    def _save(self, data: dict[str, Any]) -> None:
        """שומר את קובץ הרישום לדיסק."""
        self._path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    # ── רישום סוכן חדש ────────────────────────────────────────────────────
    def register(self, name: str, role: str) -> dict:
        """
        רושם סוכן חדש או מעדכן קיים.
        Returns the agent record dict.
        """
        data = self._load()
        now = datetime.now(timezone.utc).isoformat()

        if name in data:
            # עדכון תפקיד בלבד — לא מאפסים היסטוריה
            data[name]["role"] = role
            data[name]["registered_at"] = data[name].get("registered_at", now)
            log.info("סוכן '%s' עודכן (role=%s)", name, role)
        else:
            data[name] = {
                "role": role,
                "status": AgentStatus.IDLE.value,
                "registered_at": now,
                "last_heartbeat": None,
                "last_run": None,
                "last_error": None,
                "run_count": 0,
            }
            log.info("סוכן '%s' נרשם בהצלחה (role=%s)", name, role)

        self._save(data)
        return data[name]

    # ── דיווח heartbeat ──────────────────────────────────────────────────
    def heartbeat(self, name: str, status: AgentStatus = AgentStatus.RUNNING,
                  error: Optional[str] = None) -> None:
        """
        מעדכן heartbeat לסוכן — מצב ושעה אחרונה.
        """
        data = self._load()
        if name not in data:
            log.warning("heartbeat לסוכן לא רשום '%s' — רושם אוטומטית", name)
            data[name] = {
                "role": "unknown",
                "status": AgentStatus.IDLE.value,
                "registered_at": datetime.now(timezone.utc).isoformat(),
                "last_heartbeat": None,
                "last_run": None,
                "last_error": None,
                "run_count": 0,
            }

        now = datetime.now(timezone.utc).isoformat()
        data[name]["last_heartbeat"] = now
        data[name]["status"] = status.value

        if status == AgentStatus.COMPLETED:
            data[name]["last_run"] = now
            data[name]["run_count"] = data[name].get("run_count", 0) + 1
        elif status == AgentStatus.FAILED:
            data[name]["last_error"] = error or "unknown error"

        self._save(data)
        log.debug("heartbeat: %s → %s", name, status.value)

    # ── שאילתות ──────────────────────────────────────────────────────────
    def get_status(self, name: str) -> Optional[dict]:
        """מחזיר את רשומת הסוכן (עם בדיקת STALE) או None."""
        data = self._load()
        record = data.get(name)
        if record is None:
            return None
        # בדיקה אם הסוכן STALE
        self._check_stale(record)
        return record

    def all_agents(self) -> dict[str, dict]:
        """מחזיר את כל הסוכנים הרשומים (עם בדיקת STALE לכל אחד)."""
        data = self._load()
        for record in data.values():
            self._check_stale(record)
        return data

    def is_healthy(self, name: str) -> bool:
        """
        סוכן נחשב בריא אם הוא לא FAILED ולא STALE.
        """
        record = self.get_status(name)
        if record is None:
            return False
        return record["status"] not in (AgentStatus.FAILED.value, AgentStatus.STALE.value)

    # ── בדיקת STALE פנימית ──────────────────────────────────────────────
    @staticmethod
    def _check_stale(record: dict) -> None:
        """
        אם הסוכן RUNNING ולא דיווח heartbeat מעל 30 דקות — סימון STALE.
        """
        if record["status"] != AgentStatus.RUNNING.value:
            return
        hb = record.get("last_heartbeat")
        if hb is None:
            return
        try:
            last_hb = datetime.fromisoformat(hb)
            if datetime.now(timezone.utc) - last_hb > STALE_THRESHOLD:
                record["status"] = AgentStatus.STALE.value
        except (ValueError, TypeError):
            pass


# ── Singleton ────────────────────────────────────────────────────────────────
_registry: Optional[AgentRegistry] = None


def get_registry() -> AgentRegistry:
    """מחזיר singleton של הרג'יסטרי."""
    global _registry
    if _registry is None:
        _registry = AgentRegistry()
    return _registry


# ── CLI — הצגת מצב הרג'יסטרי ──────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    registry = get_registry()
    agents = registry.all_agents()

    if not agents:
        print("Registry is empty — no agents registered yet.")
        sys.exit(0)

    print(f"Agent Registry ({REGISTRY_FILE}):")
    print("=" * 72)
    for name, rec in agents.items():
        status = rec.get("status", "?")
        role = rec.get("role", "?")
        last_hb = (rec.get("last_heartbeat") or "never")[:19]
        runs = rec.get("run_count", 0)
        print(f"  {name:<30} status={status:<10} role={role:<20} "
              f"heartbeat={last_hb}  runs={runs}")
