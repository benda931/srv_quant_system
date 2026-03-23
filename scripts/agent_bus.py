"""
scripts/agent_bus.py
---------------------
AgentBus — ערוץ תקשורת משותף בין כל הסוכנים

כל סוכן יכול:
  1. לפרסם תוצאות → bus.publish(agent_name, payload)
  2. לקרוא את הסנאפשוט האחרון של סוכן אחר → bus.latest(agent_name)
  3. לקרוא את ההיסטוריה המלאה → bus.history(agent_name, n=10)

המידע נשמר בקובץ JSON: logs/agent_bus.json
פשוט, אמין, אין dependency חיצוני.

Format:
{
  "agent_daily_pipeline": [
    {"ts": "...", "status": "ok", "vol_breach": false, ...}
  ],
  "agent_portfolio_optimizer": [...],
  ...
}
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
BUS_FILE = ROOT / "logs" / "agent_bus.json"
MAX_HISTORY = 20   # רשומות מקסימום לכל סוכן

log = logging.getLogger("agent_bus")


# ─────────────────────────────────────────────────────────────────────────────
class AgentBus:
    """
    File-based inter-agent message bus.
    Thread-safe enough for our sequential single-process use-case.
    """

    def __init__(self, bus_file: Path = BUS_FILE) -> None:
        self._path = bus_file
        self._path.parent.mkdir(exist_ok=True)

    # ── Read ──────────────────────────────────────────────────────────────────
    def _load(self) -> dict:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def latest(self, agent_name: str) -> Optional[dict]:
        """Return the most recent published payload for this agent, or None."""
        data = self._load()
        entries = data.get(agent_name, [])
        return entries[-1] if entries else None

    def history(self, agent_name: str, n: int = 10) -> list[dict]:
        """Return the last n entries for this agent (oldest first)."""
        data = self._load()
        return data.get(agent_name, [])[-n:]

    def all_latest(self) -> dict[str, Optional[dict]]:
        """Return the latest entry for every agent on the bus."""
        data = self._load()
        return {name: entries[-1] if entries else None for name, entries in data.items()}

    # ── Write ─────────────────────────────────────────────────────────────────
    def publish(self, agent_name: str, payload: dict[str, Any]) -> None:
        """Publish a result snapshot from an agent to the bus."""
        data = self._load()
        if agent_name not in data:
            data[agent_name] = []

        entry = {"ts": datetime.now(timezone.utc).isoformat(), **payload}
        data[agent_name].append(entry)

        # Keep history bounded
        if len(data[agent_name]) > MAX_HISTORY:
            data[agent_name] = data[agent_name][-MAX_HISTORY:]

        self._path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        log.debug("AgentBus: %s published %d keys", agent_name, len(payload))

    def clear(self, agent_name: str) -> None:
        """Clear history for one agent (useful in tests)."""
        data = self._load()
        data.pop(agent_name, None)
        self._path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────────────────────
_bus: Optional[AgentBus] = None


def get_bus() -> AgentBus:
    global _bus
    if _bus is None:
        _bus = AgentBus()
    return _bus


# ─────────────────────────────────────────────────────────────────────────────
# CLI — show bus status
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    bus = get_bus()
    all_latest = bus.all_latest()
    if not all_latest:
        print("AgentBus is empty — no agents have published yet.")
        sys.exit(0)

    print(f"AgentBus snapshot ({BUS_FILE}):")
    print("=" * 60)
    for name, entry in all_latest.items():
        if entry:
            ts = entry.get("ts", "?")
            status = entry.get("status", entry.get("success", "?"))
            print(f"  {name:<35} ts={ts[:19]}  status={status}")
        else:
            print(f"  {name:<35} (empty)")
