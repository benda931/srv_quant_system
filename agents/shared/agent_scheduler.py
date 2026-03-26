"""
agents/shared/agent_scheduler.py
---------------------------------
Dependency-Aware Workflow Orchestrator — Agent Scheduling & Workflow Engine.

Original features (preserved):
  - Cron parsing and scheduling
  - Dependency chains (depends_on)
  - Subprocess execution
  - Config from agent_schedule.json
  - AgentScheduler with check_and_run, run_all_once, run_single, start_loop

Institutional upgrade:
  - DAILY_WORKFLOW with 10 stages and explicit dependency graph
  - Workflow-aware scheduling (check_dependencies_met, get_workflow_status)
  - Retry policy with exponential backoff
  - Global freeze capability
  - run_workflow_cycle() for full dependency-ordered execution
  - Machine summary for downstream agents
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import schedule

from agents.shared.agent_registry import (
    get_registry, AgentStatus, RuntimeState,
    BLOCKED_FAILURE_THRESHOLD,
)

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent  # srv_quant_system/
SCHEDULE_CONFIG = ROOT / "config" / "agent_schedule.json"
WORKFLOW_STATE_FILE = ROOT / "logs" / "workflow_state.json"
GLOBAL_FREEZE_FILE = ROOT / "logs" / "global_freeze.json"

log = logging.getLogger("agent_scheduler")

# ── Default schedule (original) ──────────────────────────────────────────────
DEFAULT_SCHEDULE: dict[str, dict[str, Any]] = {
    "methodology": {
        "cron": "0 6 * * 1-5",
        "depends_on": [],
        "script": "agents/methodology/run.py",
        "description": "Methodology agent — daily market scan",
    },
    "optimizer": {
        "cron": "0 7 * * 1-5",
        "depends_on": ["methodology"],
        "script": "agents/optimizer/run.py",
        "description": "Optimizer agent — waits for methodology",
    },
    "math": {
        "cron": "0 8 * * 1",
        "depends_on": [],
        "script": "agents/math/run.py",
        "description": "Math agent — weekly analysis (Monday)",
    },
}


# ── Daily workflow definition ────────────────────────────────────────────────
DAILY_WORKFLOW: List[Dict[str, Any]] = [
    {"stage": "data_scout", "agent": "data_scout", "depends_on": []},
    {"stage": "methodology", "agent": "methodology", "depends_on": ["data_scout"]},
    {"stage": "alpha_decay", "agent": "alpha_decay", "depends_on": ["methodology"]},
    {"stage": "optimizer", "agent": "optimizer", "depends_on": ["methodology"]},
    {"stage": "regime", "agent": "regime_forecaster", "depends_on": []},
    {"stage": "math", "agent": "math", "depends_on": ["methodology", "optimizer"]},
    {"stage": "portfolio", "agent": "portfolio_construction", "depends_on": ["methodology", "regime", "alpha_decay"]},
    {"stage": "risk", "agent": "risk_guardian", "depends_on": ["portfolio", "regime"]},
    {"stage": "execution", "agent": "execution", "depends_on": ["portfolio", "risk"]},
    {"stage": "auto_improve", "agent": "auto_improve", "depends_on": ["methodology", "optimizer", "alpha_decay"]},
]

# Map stage name -> workflow entry for fast lookup
_STAGE_MAP: Dict[str, Dict[str, Any]] = {s["stage"]: s for s in DAILY_WORKFLOW}
# Map agent name -> stage name
_AGENT_TO_STAGE: Dict[str, str] = {s["agent"]: s["stage"] for s in DAILY_WORKFLOW}

# ── Retry policy constants ───────────────────────────────────────────────────
MAX_RETRIES_PER_DAY = 2
BACKOFF_SCHEDULE = [60, 300, 900]  # seconds: 1min, 5min, 15min
DEGRADED_AFTER = 3   # consecutive failures
BLOCKED_AFTER = 5    # consecutive failures — manual unfreeze required


def load_schedule_config() -> dict[str, dict[str, Any]]:
    """
    Load schedule config from config/agent_schedule.json.
    Creates default if not exists.
    """
    if SCHEDULE_CONFIG.exists():
        try:
            return json.loads(SCHEDULE_CONFIG.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Schedule config read error: %s — using defaults", exc)

    SCHEDULE_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    SCHEDULE_CONFIG.write_text(
        json.dumps(DEFAULT_SCHEDULE, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("Created default schedule config: %s", SCHEDULE_CONFIG)
    return DEFAULT_SCHEDULE


# ── Cron parsing (original) ─────────────────────────────────────────────────
def _parse_cron_field(field: str, min_val: int, max_val: int) -> list[int]:
    """Parse a single cron field — supports: *, */N, N, N-M, N,M,K"""
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
    Parse cron expression: minute hour dom month dow
    Returns dict with minutes, hours, days_of_week (0=Mon..6=Sun).
    """
    parts = expr.strip().split()
    if len(parts) != 5:
        raise ValueError(f"Invalid cron expression: {expr!r}")

    minutes = _parse_cron_field(parts[0], 0, 59)
    hours = _parse_cron_field(parts[1], 0, 23)
    dow_str = parts[4]

    # Parse days of week — cron: 0=Sun..6=Sat -> Python: 0=Mon..6=Sun
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

    minute = minutes[0] if len(minutes) == 1 else minutes[0]
    hour = hours[0] if len(hours) == 1 else hours[0]

    return {
        "minute": minute, "hour": hour,
        "minutes": minutes, "hours": hours,
        "days_of_week": days_of_week,
    }


def _should_run_now(cron_expr: str) -> bool:
    """Check if agent should run now (current minute). Supports */N."""
    now = datetime.now()
    parsed = _parse_cron(cron_expr)
    return (
        now.hour in parsed.get("hours", [parsed["hour"]])
        and now.minute in parsed.get("minutes", [parsed["minute"]])
        and now.weekday() in parsed["days_of_week"]
    )


def _next_run_description(cron_expr: str) -> str:
    """Return human-readable schedule description."""
    parsed = _parse_cron(cron_expr)
    day_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    days = ", ".join(day_names.get(d, "?") for d in sorted(parsed["days_of_week"]))
    return f"{parsed['hour']:02d}:{parsed['minute']:02d} on [{days}]"


# ── Run agent as subprocess (original) ───────────────────────────────────────
def run_agent(agent_name: str, script_path: str, dry_run: bool = False) -> bool:
    """
    Run agent as subprocess. Returns True on success.
    In dry_run mode — logs only, no execution.
    """
    registry = get_registry()
    full_path = ROOT / script_path

    if dry_run:
        log.info("[DRY-RUN] Would run: %s (%s)", agent_name, full_path)
        return True

    if not full_path.exists():
        log.warning("Script not found: %s — skipping '%s'", full_path, agent_name)
        return False

    log.info("-- Starting agent: %s --", agent_name)
    registry.heartbeat(agent_name, AgentStatus.RUNNING)

    try:
        result = subprocess.run(
            [sys.executable, str(full_path)],
            capture_output=True,
            text=True,
            timeout=1800,  # 30 min max
            cwd=str(ROOT),
        )

        if result.returncode == 0:
            registry.heartbeat(agent_name, AgentStatus.COMPLETED)
            log.info("Agent '%s' completed successfully (exit code 0)", agent_name)
            if result.stdout.strip():
                last_lines = result.stdout.strip().split("\n")[-5:]
                for line in last_lines:
                    log.debug("  [%s] %s", agent_name, line)
            return True
        else:
            err_msg = result.stderr.strip()[-500:] if result.stderr else "no stderr"
            registry.heartbeat(agent_name, AgentStatus.FAILED, error=err_msg)
            log.error("Agent '%s' failed (exit code %d): %s",
                      agent_name, result.returncode, err_msg[:200])
            return False

    except subprocess.TimeoutExpired:
        registry.heartbeat(agent_name, AgentStatus.FAILED, error="timeout after 30min")
        log.error("Agent '%s' — timeout after 30 minutes", agent_name)
        return False
    except Exception as exc:
        registry.heartbeat(agent_name, AgentStatus.FAILED, error=str(exc))
        log.error("Error running agent '%s': %s", agent_name, exc)
        return False


# ── Dependency check (original) ──────────────────────────────────────────────
def _dependencies_met(agent_name: str, config: dict, completed_today: set[str]) -> bool:
    """Check if all dependencies for an agent have completed today."""
    agent_conf = config.get(agent_name, {})
    depends_on = agent_conf.get("depends_on", [])
    if not depends_on:
        return True

    for dep in depends_on:
        if dep not in completed_today:
            log.debug("Dependency not met: %s waiting for %s", agent_name, dep)
            return False
    return True


# ── Main scheduler class ────────────────────────────────────────────────────
class AgentScheduler:
    """
    Dependency-Aware Workflow Orchestrator.

    Original API preserved:
      check_and_run(), run_all_once(), run_single(), start_loop(), print_schedule()

    Institutional additions:
      check_dependencies_met(), get_workflow_status(), should_run_now(),
      apply_retry_policy(), get_global_freeze(), set_global_freeze(),
      clear_global_freeze(), run_workflow_cycle(), compute_machine_summary()
    """

    def __init__(self, dry_run: bool = False) -> None:
        self.config = load_schedule_config()
        self.dry_run = dry_run
        self._completed_today: set[str] = set()
        self._last_check_date: Optional[str] = None
        # Retry tracking: {agent_name: {"retries_today": int, "last_failure": iso, "backoff_until": iso}}
        self._retry_state: Dict[str, Dict[str, Any]] = {}
        # Workflow cycle tracking
        self._workflow_stage_status: Dict[str, str] = {}  # stage -> READY/RUNNING/COMPLETED/FAILED/BLOCKED

        # Register all agents
        registry = get_registry()
        for name, conf in self.config.items():
            registry.register(name, conf.get("description", name))

    # ── Original methods (preserved) ─────────────────────────────────────────
    def _reset_daily(self) -> None:
        """Reset completion list at start of new day."""
        today = datetime.now().strftime("%Y-%m-%d")
        if self._last_check_date != today:
            self._completed_today.clear()
            self._retry_state.clear()
            self._workflow_stage_status.clear()
            self._last_check_date = today
            log.info("-- New day: %s — resetting completions --", today)

    def check_and_run(self) -> None:
        """
        Check which agents should run now and execute them.
        Called every cycle from the main loop.
        """
        self._reset_daily()

        for agent_name, agent_conf in self.config.items():
            cron_expr = agent_conf.get("cron", "")
            script = agent_conf.get("script", "")

            if not _should_run_now(cron_expr):
                continue

            if agent_name in self._completed_today:
                continue

            if not _dependencies_met(agent_name, self.config, self._completed_today):
                log.info("Agent '%s' — dependencies not met, skipping", agent_name)
                continue

            success = run_agent(agent_name, script, dry_run=self.dry_run)
            if success:
                self._completed_today.add(agent_name)

    def run_all_once(self) -> dict[str, bool]:
        """
        Run all agents once in dependency order.
        Returns dict of {agent_name: success}.
        """
        results: dict[str, bool] = {}
        completed: set[str] = set()

        remaining = list(self.config.keys())
        max_iterations = len(remaining) * 2

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

        for agent_name in remaining:
            log.error("Agent '%s' — unresolvable dependencies", agent_name)
            results[agent_name] = False

        return results

    def run_single(self, agent_name: str) -> bool:
        """Run a single agent by name."""
        if agent_name not in self.config:
            log.error("Agent '%s' not found in config", agent_name)
            return False
        script = self.config[agent_name].get("script", "")
        return run_agent(agent_name, script, dry_run=self.dry_run)

    def start_loop(self) -> None:
        """
        Main scheduling loop — checks every 30 seconds.
        """
        log.info("-- Starting scheduling loop --")
        log.info("Registered agents:")
        for name, conf in self.config.items():
            cron = conf.get("cron", "?")
            deps = conf.get("depends_on", [])
            desc = _next_run_description(cron)
            dep_str = f" (depends: {', '.join(deps)})" if deps else ""
            log.info("  %s: %s%s", name, desc, dep_str)

        if self.dry_run:
            log.info("-- DRY-RUN mode — no actual execution --")

        schedule.every(30).seconds.do(self.check_and_run)
        self.check_and_run()

        try:
            while True:
                schedule.run_pending()
                time.sleep(10)
        except KeyboardInterrupt:
            log.info("-- Scheduling loop stopped (Ctrl+C) --")

    def print_schedule(self) -> None:
        """Print schedule in readable format."""
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

    # ══════════════════════════════════════════════════════════════════════════
    # INSTITUTIONAL ADDITIONS — Workflow Orchestrator
    # ══════════════════════════════════════════════════════════════════════════

    # ── Workflow dependency checking ─────────────────────────────────────────
    def check_dependencies_met(self, stage: str) -> bool:
        """
        Check if all dependencies for a workflow stage have completed
        and are not stale/failed.
        """
        stage_def = _STAGE_MAP.get(stage)
        if stage_def is None:
            return False

        deps = stage_def.get("depends_on", [])
        if not deps:
            return True

        registry = get_registry()
        for dep_stage in deps:
            # Check if dep stage completed in this cycle
            if self._workflow_stage_status.get(dep_stage) != "COMPLETED":
                return False
            # Also verify the agent is healthy in registry
            dep_def = _STAGE_MAP.get(dep_stage)
            if dep_def:
                dep_agent = dep_def["agent"]
                if not registry.is_healthy(dep_agent):
                    status = registry.get_status(dep_agent)
                    st = status.get("status", "?") if status else "unregistered"
                    log.debug("Dep %s agent %s is unhealthy (%s)", dep_stage, dep_agent, st)
                    return False

        return True

    # ── Workflow status ──────────────────────────────────────────────────────
    def get_workflow_status(self) -> Dict[str, Any]:
        """
        Return per-stage status: READY / BLOCKED / RUNNING / COMPLETED / FAILED.
        """
        self._reset_daily()
        registry = get_registry()
        result: Dict[str, Any] = {}

        for stage_def in DAILY_WORKFLOW:
            stage = stage_def["stage"]
            agent = stage_def["agent"]

            # Check if already tracked
            if stage in self._workflow_stage_status:
                result[stage] = {
                    "agent": agent,
                    "status": self._workflow_stage_status[stage],
                }
                continue

            # Check agent's registry state
            agent_record = registry.get_status(agent)
            if agent_record:
                rs = agent_record.get("runtime_state", agent_record.get("status", "IDLE"))
                if rs == RuntimeState.FROZEN.value or agent_record.get("frozen"):
                    result[stage] = {"agent": agent, "status": "BLOCKED"}
                    continue
                if rs == RuntimeState.DISABLED.value or agent_record.get("disabled"):
                    result[stage] = {"agent": agent, "status": "BLOCKED"}
                    continue
                if rs == RuntimeState.BLOCKED.value:
                    result[stage] = {"agent": agent, "status": "BLOCKED"}
                    continue

            # Check deps
            if self.check_dependencies_met(stage):
                result[stage] = {"agent": agent, "status": "READY"}
            else:
                result[stage] = {"agent": agent, "status": "BLOCKED"}

        return result

    # ── Should run now (workflow-aware) ──────────────────────────────────────
    def should_run_now(self, agent_name: str) -> Tuple[bool, str]:
        """
        Determine if an agent can run right now.
        Returns (can_run, reason).
        """
        # Check global freeze
        if self.get_global_freeze():
            return False, "global freeze active"

        registry = get_registry()

        # Check if agent exists in registry
        record = registry.get_status(agent_name)
        if record is None:
            return False, f"agent '{agent_name}' not registered"

        # Check frozen/disabled
        if record.get("frozen"):
            return False, f"agent is FROZEN: {record.get('frozen_reason', 'no reason')}"
        if record.get("disabled"):
            return False, f"agent is DISABLED: {record.get('disabled_reason', 'no reason')}"

        # Check runtime state
        rs = record.get("runtime_state", record.get("status", "IDLE"))
        if rs == RuntimeState.BLOCKED.value:
            return False, "agent is BLOCKED — manual unfreeze required"
        if rs == RuntimeState.RUNNING.value:
            return False, "agent is already RUNNING"

        # Check retry policy
        retry = self.apply_retry_policy(agent_name)
        if not retry["retry_allowed"]:
            return False, f"retry policy: cooldown {retry['cooldown_remaining']}s"

        # Check workflow dependencies
        stage = _AGENT_TO_STAGE.get(agent_name)
        if stage and not self.check_dependencies_met(stage):
            return False, "workflow dependencies not met"

        return True, "ready"

    # ── Retry policy ─────────────────────────────────────────────────────────
    def apply_retry_policy(self, agent_name: str) -> Dict[str, Any]:
        """
        Evaluate retry policy for an agent.
        Returns: retry_allowed, backoff_seconds, cooldown_remaining.
        """
        self._reset_daily()
        state = self._retry_state.get(agent_name, {
            "retries_today": 0,
            "last_failure": None,
            "backoff_until": None,
        })

        now = datetime.now(timezone.utc)
        retries_today = state.get("retries_today", 0)
        backoff_until_str = state.get("backoff_until")

        # Check max retries
        if retries_today >= MAX_RETRIES_PER_DAY:
            return {
                "retry_allowed": False,
                "backoff_seconds": 0,
                "cooldown_remaining": 0,
                "reason": f"max retries ({MAX_RETRIES_PER_DAY}) exhausted today",
                "retries_today": retries_today,
            }

        # Check backoff cooldown
        cooldown = 0
        if backoff_until_str:
            try:
                backoff_until = datetime.fromisoformat(backoff_until_str)
                if now < backoff_until:
                    cooldown = (backoff_until - now).total_seconds()
                    return {
                        "retry_allowed": False,
                        "backoff_seconds": BACKOFF_SCHEDULE[min(retries_today, len(BACKOFF_SCHEDULE) - 1)],
                        "cooldown_remaining": round(cooldown, 1),
                        "reason": "backoff cooldown active",
                        "retries_today": retries_today,
                    }
            except (ValueError, TypeError):
                pass

        # Compute next backoff
        idx = min(retries_today, len(BACKOFF_SCHEDULE) - 1)
        backoff = BACKOFF_SCHEDULE[idx]

        return {
            "retry_allowed": True,
            "backoff_seconds": backoff,
            "cooldown_remaining": 0,
            "reason": "ready",
            "retries_today": retries_today,
        }

    def _record_retry(self, agent_name: str) -> None:
        """Record a retry attempt and set backoff."""
        state = self._retry_state.get(agent_name, {
            "retries_today": 0,
            "last_failure": None,
            "backoff_until": None,
        })
        state["retries_today"] = state.get("retries_today", 0) + 1
        state["last_failure"] = datetime.now(timezone.utc).isoformat()

        idx = min(state["retries_today"], len(BACKOFF_SCHEDULE) - 1)
        backoff = BACKOFF_SCHEDULE[idx]
        state["backoff_until"] = (datetime.now(timezone.utc) + timedelta(seconds=backoff)).isoformat()

        self._retry_state[agent_name] = state

    # ── Global freeze ────────────────────────────────────────────────────────
    def get_global_freeze(self) -> bool:
        """Check if global freeze is active."""
        if GLOBAL_FREEZE_FILE.exists():
            try:
                data = json.loads(GLOBAL_FREEZE_FILE.read_text(encoding="utf-8"))
                return data.get("frozen", False)
            except (json.JSONDecodeError, OSError):
                return False
        return False

    def set_global_freeze(self, reason: str = "manual freeze") -> None:
        """Activate global freeze — no agents will be scheduled."""
        GLOBAL_FREEZE_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "frozen": True,
            "reason": reason,
            "frozen_at": datetime.now(timezone.utc).isoformat(),
        }
        GLOBAL_FREEZE_FILE.write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )
        log.warning("GLOBAL FREEZE activated: %s", reason)

    def clear_global_freeze(self) -> None:
        """Deactivate global freeze."""
        if GLOBAL_FREEZE_FILE.exists():
            data = {
                "frozen": False,
                "reason": None,
                "cleared_at": datetime.now(timezone.utc).isoformat(),
            }
            GLOBAL_FREEZE_FILE.write_text(
                json.dumps(data, indent=2), encoding="utf-8"
            )
        log.info("Global freeze cleared")

    # ── Workflow cycle execution ─────────────────────────────────────────────
    def run_workflow_cycle(self) -> Dict[str, Any]:
        """
        Run a full workflow cycle in dependency order.
        Skips blocked/frozen/disabled agents. Applies retry policy.
        Returns per-stage results.
        """
        self._reset_daily()

        if self.get_global_freeze():
            log.warning("Workflow cycle skipped — global freeze active")
            return {"status": "frozen", "stages": {}}

        registry = get_registry()
        results: Dict[str, Any] = {}
        cycle_start = datetime.now(timezone.utc).isoformat()

        # Topological execution — iterate until all stages resolved or stuck
        remaining = [s["stage"] for s in DAILY_WORKFLOW]
        max_iterations = len(remaining) * 2
        iteration = 0

        while remaining and iteration < max_iterations:
            iteration += 1
            progress = False

            for stage in list(remaining):
                stage_def = _STAGE_MAP[stage]
                agent = stage_def["agent"]

                # Check if we can run this stage
                can_run, reason = self.should_run_now(agent)
                if not can_run and reason != "workflow dependencies not met":
                    # Hard block — skip this stage
                    self._workflow_stage_status[stage] = "BLOCKED"
                    results[stage] = {
                        "agent": agent,
                        "status": "BLOCKED",
                        "reason": reason,
                    }
                    remaining.remove(stage)
                    progress = True
                    continue

                # Check workflow deps
                if not self.check_dependencies_met(stage):
                    continue  # Try later in this iteration

                # Execute
                self._workflow_stage_status[stage] = "RUNNING"
                script = self.config.get(agent, {}).get(
                    "script", f"agents/{agent}/run.py"
                )
                success = run_agent(agent, script, dry_run=self.dry_run)

                if success:
                    self._workflow_stage_status[stage] = "COMPLETED"
                    self._completed_today.add(agent)
                    results[stage] = {
                        "agent": agent,
                        "status": "COMPLETED",
                    }
                else:
                    self._workflow_stage_status[stage] = "FAILED"
                    self._record_retry(agent)
                    registry.escalate_failure(agent)
                    results[stage] = {
                        "agent": agent,
                        "status": "FAILED",
                    }

                remaining.remove(stage)
                progress = True

            if not progress:
                break  # No stage could be resolved — break to avoid infinite loop

        # Anything remaining is blocked by unresolved deps
        for stage in remaining:
            stage_def = _STAGE_MAP[stage]
            self._workflow_stage_status[stage] = "BLOCKED"
            results[stage] = {
                "agent": stage_def["agent"],
                "status": "BLOCKED",
                "reason": "unresolved dependencies",
            }

        # Save workflow state
        self._save_workflow_state(results, cycle_start)

        completed = sum(1 for r in results.values() if r["status"] == "COMPLETED")
        failed = sum(1 for r in results.values() if r["status"] == "FAILED")
        blocked = sum(1 for r in results.values() if r["status"] == "BLOCKED")

        return {
            "status": "completed",
            "cycle_start": cycle_start,
            "cycle_end": datetime.now(timezone.utc).isoformat(),
            "completed": completed,
            "failed": failed,
            "blocked": blocked,
            "total_stages": len(DAILY_WORKFLOW),
            "stages": results,
        }

    def _save_workflow_state(self, results: Dict, cycle_start: str) -> None:
        """Persist workflow cycle results."""
        WORKFLOW_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "cycle_start": cycle_start,
            "cycle_end": datetime.now(timezone.utc).isoformat(),
            "stages": results,
            "dry_run": self.dry_run,
        }
        WORKFLOW_STATE_FILE.write_text(
            json.dumps(state, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    # ── Machine summary ──────────────────────────────────────────────────────
    def compute_machine_summary(self) -> Dict[str, Any]:
        """Machine-readable summary of scheduler/workflow state."""
        registry = get_registry()
        workflow = self.get_workflow_status()
        metrics = registry.get_runtime_metrics()

        return {
            "global_freeze": self.get_global_freeze(),
            "workflow_stages": workflow,
            "registry_metrics": metrics,
            "completed_today": sorted(self._completed_today),
            "retry_state": {
                name: {
                    "retries_today": s.get("retries_today", 0),
                    "backoff_until": s.get("backoff_until"),
                }
                for name, s in self._retry_state.items()
            },
            "daily_workflow_stages": len(DAILY_WORKFLOW),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sched = AgentScheduler(dry_run=True)
    sched.print_schedule()
