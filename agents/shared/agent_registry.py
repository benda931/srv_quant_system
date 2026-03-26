"""
agents/shared/agent_registry.py
-------------------------------
Runtime Governance Engine — Agent Registration, Health Tracking & Lifecycle.

Original features (preserved):
  - AgentStatus enum (IDLE/RUNNING/COMPLETED/FAILED/STALE)
  - AgentRegistry with register/heartbeat/get_status/all_agents/is_healthy
  - JSON persistence to logs/agent_registry.json

Institutional upgrade:
  - RuntimeState enum with 12 lifecycle states
  - Backward-compatible mapping to legacy AgentStatus
  - Health scoring, failure escalation, freeze/disable governance
  - Dependency violation detection
  - Orchestration history and runtime metrics
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent  # srv_quant_system/
REGISTRY_FILE = ROOT / "logs" / "agent_registry.json"
ORCHESTRATION_HISTORY_FILE = ROOT / "logs" / "orchestration_history.json"

# ── Legacy statuses (preserved) ──────────────────────────────────────────────
class AgentStatus(str, Enum):
    IDLE      = "IDLE"
    RUNNING   = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED    = "FAILED"
    STALE     = "STALE"      # no heartbeat for 30+ minutes


# ── RuntimeState — institutional lifecycle ───────────────────────────────────
class RuntimeState(str, Enum):
    REGISTERED  = "REGISTERED"
    IDLE        = "IDLE"
    QUEUED      = "QUEUED"
    RUNNING     = "RUNNING"
    COMPLETED   = "COMPLETED"
    FAILED      = "FAILED"
    STALE       = "STALE"
    DEGRADED    = "DEGRADED"
    BLOCKED     = "BLOCKED"
    FROZEN      = "FROZEN"
    DISABLED    = "DISABLED"
    MAINTENANCE = "MAINTENANCE"


# ── Mapping RuntimeState -> legacy AgentStatus ───────────────────────────────
_RUNTIME_TO_LEGACY: Dict[str, str] = {
    RuntimeState.REGISTERED.value:  AgentStatus.IDLE.value,
    RuntimeState.IDLE.value:        AgentStatus.IDLE.value,
    RuntimeState.QUEUED.value:      AgentStatus.IDLE.value,
    RuntimeState.RUNNING.value:     AgentStatus.RUNNING.value,
    RuntimeState.COMPLETED.value:   AgentStatus.COMPLETED.value,
    RuntimeState.FAILED.value:      AgentStatus.FAILED.value,
    RuntimeState.STALE.value:       AgentStatus.STALE.value,
    RuntimeState.DEGRADED.value:    AgentStatus.FAILED.value,
    RuntimeState.BLOCKED.value:     AgentStatus.FAILED.value,
    RuntimeState.FROZEN.value:      AgentStatus.STALE.value,
    RuntimeState.DISABLED.value:    AgentStatus.STALE.value,
    RuntimeState.MAINTENANCE.value: AgentStatus.STALE.value,
}

def runtime_to_legacy(state: str) -> str:
    """Map a RuntimeState value to a legacy AgentStatus value."""
    return _RUNTIME_TO_LEGACY.get(state, AgentStatus.IDLE.value)


# ── Thresholds ───────────────────────────────────────────────────────────────
STALE_THRESHOLD = timedelta(minutes=30)
DEGRADED_FAILURE_THRESHOLD = 3
BLOCKED_FAILURE_THRESHOLD = 5
MAX_ORCHESTRATION_HISTORY = 500

log = logging.getLogger("agent_registry")


class AgentRegistry:
    """
    Runtime Governance Engine — manages agent lifecycle, health, and governance.

    Original API preserved:
      register(), heartbeat(), get_status(), all_agents(), is_healthy()

    Institutional additions:
      get_agent_health(), get_stale_agents(), get_blocked_agents(),
      get_failed_agents(), get_degraded_agents(), get_agents_ready_to_run(),
      get_dependency_violations(), freeze_agent(), unfreeze_agent(),
      disable_agent(), enable_agent(), escalate_failure(),
      get_runtime_metrics(), get_orchestration_history(), compute_machine_summary()
    """

    def __init__(self, registry_file: Path = REGISTRY_FILE) -> None:
        self._path = registry_file
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._history_path = ORCHESTRATION_HISTORY_FILE

    # ── Read / Write (original) ──────────────────────────────────────────────
    def _load(self) -> dict[str, Any]:
        """Load registry from disk."""
        if self._path.exists():
            try:
                return json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("Registry read error: %s", exc)
                return {}
        return {}

    def _save(self, data: dict[str, Any]) -> None:
        """Save registry to disk."""
        self._path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    # ── Register (original) ──────────────────────────────────────────────────
    def register(self, name: str, role: str) -> dict:
        """
        Register a new agent or update an existing one.
        Returns the agent record dict.
        """
        data = self._load()
        now = datetime.now(timezone.utc).isoformat()

        if name in data:
            data[name]["role"] = role
            data[name]["registered_at"] = data[name].get("registered_at", now)
            log.info("Agent '%s' updated (role=%s)", name, role)
        else:
            data[name] = {
                "role": role,
                "status": AgentStatus.IDLE.value,
                "runtime_state": RuntimeState.REGISTERED.value,
                "registered_at": now,
                "last_heartbeat": None,
                "last_run": None,
                "last_error": None,
                "run_count": 0,
                "failure_count": 0,
                "consecutive_failures": 0,
                "frozen": False,
                "frozen_reason": None,
                "disabled": False,
                "disabled_reason": None,
                "liveness_score": 1.0,
                "reliability_score": 1.0,
            }
            log.info("Agent '%s' registered (role=%s)", name, role)

        self._save(data)
        return data[name]

    # ── Heartbeat (original, extended) ───────────────────────────────────────
    def heartbeat(self, name: str, status: AgentStatus = AgentStatus.RUNNING,
                  error: Optional[str] = None) -> None:
        """
        Update heartbeat for an agent — status and timestamp.
        """
        data = self._load()
        if name not in data:
            log.warning("Heartbeat for unregistered agent '%s' — auto-registering", name)
            data[name] = {
                "role": "unknown",
                "status": AgentStatus.IDLE.value,
                "runtime_state": RuntimeState.IDLE.value,
                "registered_at": datetime.now(timezone.utc).isoformat(),
                "last_heartbeat": None,
                "last_run": None,
                "last_error": None,
                "run_count": 0,
                "failure_count": 0,
                "consecutive_failures": 0,
                "frozen": False,
                "frozen_reason": None,
                "disabled": False,
                "disabled_reason": None,
                "liveness_score": 1.0,
                "reliability_score": 1.0,
            }

        now = datetime.now(timezone.utc).isoformat()
        data[name]["last_heartbeat"] = now
        data[name]["status"] = status.value

        # Map to runtime state
        if status == AgentStatus.RUNNING:
            data[name]["runtime_state"] = RuntimeState.RUNNING.value
        elif status == AgentStatus.COMPLETED:
            data[name]["last_run"] = now
            data[name]["run_count"] = data[name].get("run_count", 0) + 1
            data[name]["consecutive_failures"] = 0
            data[name]["runtime_state"] = RuntimeState.COMPLETED.value
            # Improve reliability on success
            rel = data[name].get("reliability_score", 1.0)
            data[name]["reliability_score"] = min(1.0, rel + 0.05)
        elif status == AgentStatus.FAILED:
            data[name]["last_error"] = error or "unknown error"
            data[name]["failure_count"] = data[name].get("failure_count", 0) + 1
            data[name]["consecutive_failures"] = data[name].get("consecutive_failures", 0) + 1
            data[name]["runtime_state"] = RuntimeState.FAILED.value
            # Degrade reliability
            rel = data[name].get("reliability_score", 1.0)
            data[name]["reliability_score"] = max(0.0, rel - 0.15)
        elif status == AgentStatus.IDLE:
            data[name]["runtime_state"] = RuntimeState.IDLE.value

        # Record state change in orchestration history
        self._record_state_change(name, status.value)

        self._save(data)
        log.debug("heartbeat: %s -> %s", name, status.value)

    # ── Queries (original) ───────────────────────────────────────────────────
    def get_status(self, name: str) -> Optional[dict]:
        """Return agent record (with STALE check) or None."""
        data = self._load()
        record = data.get(name)
        if record is None:
            return None
        self._check_stale(record)
        return record

    def all_agents(self) -> dict[str, dict]:
        """Return all registered agents (with STALE check for each)."""
        data = self._load()
        for record in data.values():
            self._check_stale(record)
        return data

    def is_healthy(self, name: str) -> bool:
        """
        Agent is healthy if not FAILED and not STALE.
        """
        record = self.get_status(name)
        if record is None:
            return False
        return record["status"] not in (AgentStatus.FAILED.value, AgentStatus.STALE.value)

    # ── STALE check (original) ───────────────────────────────────────────────
    @staticmethod
    def _check_stale(record: dict) -> None:
        """
        If agent is RUNNING and no heartbeat for 30+ min -> mark STALE.
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
                record["runtime_state"] = RuntimeState.STALE.value
        except (ValueError, TypeError):
            pass

    # ══════════════════════════════════════════════════════════════════════════
    # INSTITUTIONAL ADDITIONS — Runtime Governance
    # ══════════════════════════════════════════════════════════════════════════

    def _ensure_extended_fields(self, record: dict) -> None:
        """Ensure institutional fields exist on legacy records."""
        defaults = {
            "runtime_state": _RUNTIME_TO_LEGACY.get(record.get("status", "IDLE"), "IDLE"),
            "failure_count": 0,
            "consecutive_failures": 0,
            "frozen": False,
            "frozen_reason": None,
            "disabled": False,
            "disabled_reason": None,
            "liveness_score": 1.0,
            "reliability_score": 1.0,
        }
        for key, default in defaults.items():
            if key not in record:
                record[key] = default

    # ── Agent health scoring ─────────────────────────────────────────────────
    def get_agent_health(self, name: str) -> Dict[str, Any]:
        """
        Compute health metrics for an agent.
        Returns liveness_score, reliability_score, failure_count, stale_duration.
        """
        record = self.get_status(name)
        if record is None:
            return {
                "name": name,
                "exists": False,
                "liveness_score": 0.0,
                "reliability_score": 0.0,
                "failure_count": 0,
                "stale_duration_seconds": 0,
            }

        self._ensure_extended_fields(record)

        # Compute liveness based on heartbeat freshness
        stale_seconds = 0
        liveness = 1.0
        hb = record.get("last_heartbeat")
        if hb:
            try:
                last_hb = datetime.fromisoformat(hb)
                age = (datetime.now(timezone.utc) - last_hb).total_seconds()
                stale_seconds = max(0, age)
                # Decay liveness: 1.0 at 0s, 0.5 at 15min, 0.0 at 60min
                liveness = max(0.0, 1.0 - (age / 3600.0))
            except (ValueError, TypeError):
                liveness = 0.5
        else:
            liveness = 0.5  # Never reported

        return {
            "name": name,
            "exists": True,
            "liveness_score": round(liveness, 3),
            "reliability_score": round(record.get("reliability_score", 1.0), 3),
            "failure_count": record.get("failure_count", 0),
            "consecutive_failures": record.get("consecutive_failures", 0),
            "stale_duration_seconds": round(stale_seconds, 1),
            "status": record.get("status"),
            "runtime_state": record.get("runtime_state"),
        }

    # ── Filtered agent queries ───────────────────────────────────────────────
    def get_stale_agents(self) -> List[str]:
        """Return names of agents in STALE state."""
        agents = self.all_agents()
        return [n for n, r in agents.items() if r.get("status") == AgentStatus.STALE.value]

    def get_blocked_agents(self) -> List[str]:
        """Return names of agents in BLOCKED runtime state."""
        agents = self.all_agents()
        return [
            n for n, r in agents.items()
            if r.get("runtime_state") == RuntimeState.BLOCKED.value
        ]

    def get_failed_agents(self) -> List[str]:
        """Return names of agents in FAILED state."""
        agents = self.all_agents()
        return [n for n, r in agents.items() if r.get("status") == AgentStatus.FAILED.value]

    def get_degraded_agents(self) -> List[str]:
        """Return names of agents in DEGRADED runtime state."""
        agents = self.all_agents()
        return [
            n for n, r in agents.items()
            if r.get("runtime_state") == RuntimeState.DEGRADED.value
        ]

    def get_agents_ready_to_run(self) -> List[str]:
        """
        Return agents that are not stale, not blocked, not frozen, not disabled.
        These agents are eligible for scheduling.
        """
        agents = self.all_agents()
        blocked_states = {
            RuntimeState.STALE.value,
            RuntimeState.BLOCKED.value,
            RuntimeState.FROZEN.value,
            RuntimeState.DISABLED.value,
            RuntimeState.MAINTENANCE.value,
        }
        ready = []
        for name, record in agents.items():
            self._ensure_extended_fields(record)
            rs = record.get("runtime_state", "IDLE")
            if rs not in blocked_states and not record.get("frozen") and not record.get("disabled"):
                ready.append(name)
        return ready

    # ── Dependency violation detection ───────────────────────────────────────
    def get_dependency_violations(self, dependency_map: Optional[Dict[str, List[str]]] = None) -> List[Dict[str, Any]]:
        """
        Return agents whose dependencies are stale or failed.
        dependency_map: {agent_name: [dep1, dep2, ...]}
        If not provided, returns empty list.
        """
        if not dependency_map:
            return []

        agents = self.all_agents()
        violations = []
        unhealthy_states = {
            AgentStatus.FAILED.value,
            AgentStatus.STALE.value,
        }

        for agent_name, deps in dependency_map.items():
            for dep in deps:
                dep_record = agents.get(dep)
                if dep_record is None:
                    violations.append({
                        "agent": agent_name,
                        "dependency": dep,
                        "issue": "dependency not registered",
                    })
                elif dep_record.get("status") in unhealthy_states:
                    violations.append({
                        "agent": agent_name,
                        "dependency": dep,
                        "issue": f"dependency is {dep_record.get('status')}",
                    })
        return violations

    # ── Freeze / Unfreeze ────────────────────────────────────────────────────
    def freeze_agent(self, name: str, reason: str = "manual freeze") -> bool:
        """Set agent to FROZEN state with reason. Returns True if agent exists."""
        data = self._load()
        if name not in data:
            return False
        self._ensure_extended_fields(data[name])
        data[name]["frozen"] = True
        data[name]["frozen_reason"] = reason
        data[name]["runtime_state"] = RuntimeState.FROZEN.value
        self._record_state_change(name, "FROZEN", reason)
        self._save(data)
        log.info("Agent '%s' FROZEN: %s", name, reason)
        return True

    def unfreeze_agent(self, name: str) -> bool:
        """Remove FROZEN state from agent. Returns True if agent exists."""
        data = self._load()
        if name not in data:
            return False
        self._ensure_extended_fields(data[name])
        data[name]["frozen"] = False
        data[name]["frozen_reason"] = None
        data[name]["runtime_state"] = RuntimeState.IDLE.value
        data[name]["status"] = AgentStatus.IDLE.value
        self._record_state_change(name, "UNFROZEN")
        self._save(data)
        log.info("Agent '%s' unfrozen", name)
        return True

    # ── Disable / Enable ─────────────────────────────────────────────────────
    def disable_agent(self, name: str, reason: str = "manual disable") -> bool:
        """Disable agent — excluded from all scheduling. Returns True if exists."""
        data = self._load()
        if name not in data:
            return False
        self._ensure_extended_fields(data[name])
        data[name]["disabled"] = True
        data[name]["disabled_reason"] = reason
        data[name]["runtime_state"] = RuntimeState.DISABLED.value
        self._record_state_change(name, "DISABLED", reason)
        self._save(data)
        log.info("Agent '%s' DISABLED: %s", name, reason)
        return True

    def enable_agent(self, name: str) -> bool:
        """Re-enable a disabled agent. Returns True if exists."""
        data = self._load()
        if name not in data:
            return False
        self._ensure_extended_fields(data[name])
        data[name]["disabled"] = False
        data[name]["disabled_reason"] = None
        data[name]["runtime_state"] = RuntimeState.IDLE.value
        data[name]["status"] = AgentStatus.IDLE.value
        self._record_state_change(name, "ENABLED")
        self._save(data)
        log.info("Agent '%s' enabled", name)
        return True

    # ── Failure escalation ───────────────────────────────────────────────────
    def escalate_failure(self, name: str) -> Dict[str, Any]:
        """
        Increment failure count. Auto-degrade after threshold.
        Returns current failure state.
        """
        data = self._load()
        if name not in data:
            return {"error": f"agent '{name}' not registered"}

        self._ensure_extended_fields(data[name])
        data[name]["failure_count"] = data[name].get("failure_count", 0) + 1
        data[name]["consecutive_failures"] = data[name].get("consecutive_failures", 0) + 1
        consec = data[name]["consecutive_failures"]

        result = {
            "agent": name,
            "failure_count": data[name]["failure_count"],
            "consecutive_failures": consec,
            "action": "none",
        }

        if consec >= BLOCKED_FAILURE_THRESHOLD:
            data[name]["runtime_state"] = RuntimeState.BLOCKED.value
            data[name]["status"] = AgentStatus.FAILED.value
            result["action"] = "BLOCKED — manual unfreeze required"
            self._record_state_change(name, "BLOCKED", f"consecutive_failures={consec}")
            log.error("Agent '%s' BLOCKED after %d consecutive failures", name, consec)
        elif consec >= DEGRADED_FAILURE_THRESHOLD:
            data[name]["runtime_state"] = RuntimeState.DEGRADED.value
            data[name]["status"] = AgentStatus.FAILED.value
            result["action"] = "DEGRADED — reduced priority"
            self._record_state_change(name, "DEGRADED", f"consecutive_failures={consec}")
            log.warning("Agent '%s' DEGRADED after %d consecutive failures", name, consec)

        # Decrease reliability
        rel = data[name].get("reliability_score", 1.0)
        data[name]["reliability_score"] = max(0.0, rel - 0.2)

        self._save(data)
        return result

    # ── Runtime metrics ──────────────────────────────────────────────────────
    def get_runtime_metrics(self) -> Dict[str, Any]:
        """
        Return aggregate metrics: counts by state.
        """
        agents = self.all_agents()
        total = len(agents)
        counts: Dict[str, int] = {}
        for record in agents.values():
            self._ensure_extended_fields(record)
            rs = record.get("runtime_state", "IDLE")
            counts[rs] = counts.get(rs, 0) + 1

        healthy_count = sum(
            1 for r in agents.values()
            if r.get("status") not in (AgentStatus.FAILED.value, AgentStatus.STALE.value)
        )

        return {
            "total_agents": total,
            "healthy": healthy_count,
            "stale": len(self.get_stale_agents()),
            "blocked": len(self.get_blocked_agents()),
            "failed": len(self.get_failed_agents()),
            "degraded": len(self.get_degraded_agents()),
            "frozen": sum(1 for r in agents.values() if r.get("frozen")),
            "disabled": sum(1 for r in agents.values() if r.get("disabled")),
            "state_distribution": counts,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ── Orchestration history ────────────────────────────────────────────────
    def _load_history(self) -> List[Dict[str, Any]]:
        if self._history_path.exists():
            try:
                return json.loads(self._history_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return []
        return []

    def _save_history(self, history: List[Dict[str, Any]]) -> None:
        if len(history) > MAX_ORCHESTRATION_HISTORY:
            history = history[-MAX_ORCHESTRATION_HISTORY:]
        self._history_path.parent.mkdir(parents=True, exist_ok=True)
        self._history_path.write_text(
            json.dumps(history, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    def _record_state_change(self, agent_name: str, new_state: str,
                              reason: str = "") -> None:
        """Append a state change to orchestration history."""
        history = self._load_history()
        history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": agent_name,
            "new_state": new_state,
            "reason": reason,
        })
        self._save_history(history)

    def get_orchestration_history(self, n: int = 50) -> List[Dict[str, Any]]:
        """Return the most recent n state changes."""
        history = self._load_history()
        return history[-n:]

    # ── Machine summary ──────────────────────────────────────────────────────
    def compute_machine_summary(self) -> Dict[str, Any]:
        """Machine-readable summary of the registry for downstream agents."""
        agents = self.all_agents()
        metrics = self.get_runtime_metrics()
        agent_summaries = {}
        for name, record in agents.items():
            self._ensure_extended_fields(record)
            agent_summaries[name] = {
                "status": record.get("status"),
                "runtime_state": record.get("runtime_state"),
                "reliability_score": record.get("reliability_score"),
                "consecutive_failures": record.get("consecutive_failures"),
                "last_heartbeat": record.get("last_heartbeat"),
                "run_count": record.get("run_count"),
                "frozen": record.get("frozen"),
                "disabled": record.get("disabled"),
            }

        return {
            "metrics": metrics,
            "agents": agent_summaries,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# ── Singleton ────────────────────────────────────────────────────────────────
_registry: Optional[AgentRegistry] = None


def get_registry() -> AgentRegistry:
    """Return singleton registry."""
    global _registry
    if _registry is None:
        _registry = AgentRegistry()
    return _registry


# ── CLI — display registry status ────────────────────────────────────────────
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
