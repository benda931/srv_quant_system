"""
agents/architect/agent_architect.py
=====================================
Architect Agent — Systematic Self-Improvement

Scans the system domain-by-domain, uses GPT for focused analysis,
delegates implementation to Claude Code, validates results.

Cycle: scan → GPT analyze → Claude implement → validate → log
One domain per cycle. Max 3 changes per cycle. No scattering.

Schedule: 09:00 weekdays (after optimizer finishes)
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# ─────────────────────────────────────────────────────────────────────────────
# Imports (all guarded for graceful degradation)
# ─────────────────────────────────────────────────────────────────────────────

_IMPORTS_OK = True
try:
    from agents.architect.scan_domains import (
        SCAN_DOMAINS, DOMAIN_MAP, DOMAIN_NAMES,
        ScanDomain, ScanResult, scan_domain,
    )
    from agents.architect.improvement_log import get_improvement_log
except ImportError as e:
    logger.error("Architect: scan_domains/improvement_log import failed: %s", e)
    _IMPORTS_OK = False

try:
    from agents.shared.agent_registry import get_registry, AgentStatus
except ImportError:
    get_registry = None

try:
    from scripts.agent_bus import get_bus
except ImportError:
    get_bus = None

try:
    from agents.shared.gpt_conversation import GPTConversation
except ImportError:
    GPTConversation = None

try:
    from scripts.claude_loop import send_to_claude, execute_actions
except ImportError:
    send_to_claude = None
    execute_actions = None


MAX_CHANGES_PER_CYCLE = 3
MAX_GPT_TURNS = 3
MAX_CLAUDE_TURNS = 6


# ─────────────────────────────────────────────────────────────────────────────
# Architect Agent
# ─────────────────────────────────────────────────────────────────────────────

class ArchitectAgent:
    """
    Meta-agent that systematically improves the SRV system.

    Works one domain at a time:
      1. Determine which domain needs attention
      2. Scan that domain (files + metrics + health checks)
      3. Send findings to GPT for focused analysis (3 short turns)
      4. Convert GPT recommendation to Claude task
      5. Claude implements the change
      6. Validate (tests + metrics)
      7. Log everything
    """

    def __init__(self):
        self.log = get_improvement_log()
        self.bus = get_bus() if get_bus else None
        self.registry = get_registry() if get_registry else None

        # GPT conversation for analysis
        self.gpt = None
        if GPTConversation is not None:
            self.gpt = GPTConversation(
                system_role="software architect and quantitative systems analyst"
            )

    # ── Step 1: Determine next domain ────────────────────────────

    def _determine_next_domain(self) -> ScanDomain:
        """
        Pick the highest-priority domain that needs attention.

        Priority logic:
          1. Urgent issues (failed agents, degraded metrics) → jump to front
          2. Domains not scanned in 7+ days → by priority order
          3. Default: cycle through priority 1→8
        """
        # Check for urgent issues from bus
        if self.bus:
            try:
                meth = self.bus.latest("agent_methodology")
                if isinstance(meth, dict):
                    ic = meth.get("ic_mean", meth.get("metrics", {}).get("ic_mean"))
                    if ic is not None and float(ic) < 0.01:
                        logger.info("URGENT: IC=%.4f < 0.01 → signal_quality", ic)
                        return DOMAIN_MAP["signal_quality"]
            except Exception:
                pass

            # Check for failed agents
            try:
                reg_data = self.bus.latest("agent_registry") or {}
                if isinstance(reg_data, dict):
                    for name, info in reg_data.items():
                        if isinstance(info, dict) and info.get("status") in ("FAILED", "STALE"):
                            logger.info("URGENT: Agent %s is %s → agent_health", name, info["status"])
                            return DOMAIN_MAP["agent_health"]
            except Exception:
                pass

        # Find domains not scanned in 7+ days
        pending = self.log.get_pending_domains(DOMAIN_NAMES, days_threshold=7)
        if pending:
            # Pick highest priority among pending
            for domain in SCAN_DOMAINS:
                if domain.name in pending:
                    logger.info("Pending domain: %s (not scanned in 7+ days)", domain.name)
                    return domain

        # Default: cycle through by priority
        domain_stats = self.log.domain_stats()
        min_cycles = min((domain_stats.get(d, {}).get("cycles", 0) for d in DOMAIN_NAMES), default=0)
        for domain in SCAN_DOMAINS:
            if domain_stats.get(domain.name, {}).get("cycles", 0) <= min_cycles:
                return domain

        return SCAN_DOMAINS[0]

    # ── Step 2: Scan domain ──────────────────────────────────────

    def _scan(self, domain: ScanDomain) -> ScanResult:
        """Scan the domain — read files, collect metrics, run health checks."""
        return scan_domain(domain)

    # ── Step 3: GPT analysis ─────────────────────────────────────

    def _analyze_with_gpt(self, domain: ScanDomain, scan: ScanResult) -> Dict[str, str]:
        """
        3-turn GPT conversation for focused analysis.

        Turn 1: Diagnosis — what's the main problem?
        Turn 2: Specifics — what exactly to change?
        Turn 3: Validation — how to verify the change worked?
        """
        if not self.gpt or not self.gpt.available:
            logger.warning("GPT not available — using heuristic analysis")
            return self._heuristic_analysis(domain, scan)

        # Reset conversation history for fresh cycle
        self.gpt.history = []

        # Build context summary (compact, not the whole codebase)
        code_summary = ""
        for fname, snippet in list(scan.file_snippets.items())[:3]:
            code_summary += f"\n### {fname}\n```python\n{snippet[:1500]}\n```\n"

        metrics_str = json.dumps(scan.metrics, indent=2, default=str)[:800]
        findings_str = "\n".join(f"- {f}" for f in scan.findings)

        # ── Turn 1: Diagnosis ────────────────────────────────────
        turn1 = (
            f"Domain: {domain.name} — {domain.description}\n\n"
            f"Current metrics:\n{metrics_str}\n\n"
            f"Health check findings:\n{findings_str}\n\n"
            f"Key code:\n{code_summary}\n\n"
            f"What is the SINGLE most impactful improvement for this domain? "
            f"Be specific — name the function, the metric, and the problem."
        )
        diagnosis = self.gpt._query(turn1)
        logger.info("GPT Turn 1 (diagnosis): %s", diagnosis[:200])

        # ── Turn 2: Specifics ────────────────────────────────────
        turn2 = (
            f"You identified: {diagnosis[:300]}\n\n"
            f"What EXACT code change or parameter adjustment fixes this? "
            f"Give me: file path, function name, and the specific change. "
            f"If it's a parameter, give the new value. "
            f"If it's code, describe the change in 2-3 sentences."
        )
        suggestion = self.gpt._query(turn2)
        logger.info("GPT Turn 2 (suggestion): %s", suggestion[:200])

        # ── Turn 3: Validation ───────────────────────────────────
        turn3 = (
            f"You suggest: {suggestion[:300]}\n\n"
            f"What test or metric would confirm this worked? "
            f"What could go wrong? "
            f"Give me a specific validation criterion."
        )
        validation = self.gpt._query(turn3)
        logger.info("GPT Turn 3 (validation): %s", validation[:200])

        return {
            "diagnosis": diagnosis,
            "suggestion": suggestion,
            "validation": validation,
        }

    def _heuristic_analysis(self, domain: ScanDomain, scan: ScanResult) -> Dict[str, str]:
        """Fallback analysis when GPT is unavailable."""
        problems = [f for f in scan.findings if f.startswith("FAILED") or f.startswith("MISSING")]
        if not problems:
            return {
                "diagnosis": "No issues detected in this domain.",
                "suggestion": "No changes needed.",
                "validation": "N/A",
            }
        return {
            "diagnosis": f"Issues found: {'; '.join(problems[:3])}",
            "suggestion": "Manual review recommended — GPT unavailable for detailed analysis.",
            "validation": "Run tests and check metrics after any manual changes.",
        }

    # ── Step 4: Create Claude task ───────────────────────────────

    def _create_task_for_claude(self, domain: ScanDomain, gpt_analysis: Dict[str, str]) -> Optional[Dict]:
        """
        Convert GPT recommendation to a structured task for Claude.

        Returns None if no changes needed.
        """
        suggestion = gpt_analysis.get("suggestion", "")
        diagnosis = gpt_analysis.get("diagnosis", "")

        if "no changes needed" in suggestion.lower() or "no issues" in diagnosis.lower():
            return None

        system_prompt = (
            f"You are a quantitative systems engineer improving the SRV Quant System.\n\n"
            f"DOMAIN: {domain.name} — {domain.description}\n\n"
            f"DIAGNOSIS: {diagnosis[:500]}\n\n"
            f"RECOMMENDATION: {suggestion[:500]}\n\n"
            f"RULES:\n"
            f"- Make MAXIMUM {MAX_CHANGES_PER_CYCLE} changes\n"
            f"- Each change must be testable\n"
            f"- Do NOT change unrelated code\n"
            f"- Focus on the specific recommendation above\n"
            f"- After changes, respond with a JSON action list\n\n"
            f"Respond with a JSON block containing your actions:\n"
            f'{{"actions": [{{"type": "edit_param", "file": "...", "param": "...", "value": ...}}, '
            f'{{"type": "done", "summary": "..."}}]}}'
        )

        return {
            "system_prompt": system_prompt,
            "initial_message": f"Please implement the recommended change for {domain.name}.",
            "max_turns": MAX_CLAUDE_TURNS,
        }

    # ── Step 5: Implement via Claude ─────────────────────────────

    def _implement(self, task: Dict) -> Dict[str, Any]:
        """Send task to Claude and execute returned actions."""
        if send_to_claude is None:
            logger.warning("Claude loop not available — skipping implementation")
            return {"status": "skipped", "reason": "claude_loop not available"}

        try:
            messages = []
            response_text, messages = send_to_claude(
                system_prompt=task["system_prompt"],
                messages=messages,
                user_content=task["initial_message"],
            )

            # Extract JSON actions from response
            actions = []
            try:
                # Look for JSON block in response
                import re
                json_match = re.search(r'\{[\s\S]*"actions"[\s\S]*\}', response_text)
                if json_match:
                    parsed = json.loads(json_match.group())
                    actions = parsed.get("actions", [])
            except (json.JSONDecodeError, AttributeError):
                logger.warning("Could not parse Claude response as JSON")

            # Cap at MAX_CHANGES_PER_CYCLE
            edit_actions = [a for a in actions if a.get("type") in ("edit_param", "edit_code")]
            if len(edit_actions) > MAX_CHANGES_PER_CYCLE:
                logger.warning("Capping actions from %d to %d", len(edit_actions), MAX_CHANGES_PER_CYCLE)
                # Keep first N edit actions + all non-edit actions
                kept = 0
                filtered = []
                for a in actions:
                    if a.get("type") in ("edit_param", "edit_code"):
                        if kept < MAX_CHANGES_PER_CYCLE:
                            filtered.append(a)
                            kept += 1
                    else:
                        filtered.append(a)
                actions = filtered

            # Execute
            if actions and execute_actions is not None:
                results = execute_actions(actions)
                return {"status": "executed", "actions": actions, "results": results}
            else:
                return {"status": "no_actions", "response": response_text[:500]}

        except Exception as e:
            logger.error("Claude implementation failed: %s", e, exc_info=True)
            return {"status": "failed", "error": str(e)}

    # ── Step 6: Validate ─────────────────────────────────────────

    def _validate(self, domain: ScanDomain, before_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that changes didn't break anything.

        1. All tests must pass
        2. For signal/risk domains, re-check metrics
        """
        # Run tests
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", str(ROOT / "tests"), "-q", "--tb=short"],
                capture_output=True, text=True, timeout=120, cwd=str(ROOT),
            )
            tests_pass = result.returncode == 0
            test_output = (result.stdout + result.stderr)[-500:]
        except Exception as e:
            tests_pass = False
            test_output = str(e)

        # Collect after-metrics if available
        after_metrics: Dict[str, Any] = {}
        if self.bus and domain.metrics_source.startswith("bus:"):
            agent_name = domain.metrics_source.split(":", 1)[1]
            try:
                after_metrics = self.bus.latest(agent_name) or {}
            except Exception:
                pass

        return {
            "tests_pass": tests_pass,
            "test_output": test_output,
            "metrics_before": before_metrics,
            "metrics_after": after_metrics,
        }

    # ── Main cycle ───────────────────────────────────────────────

    def run_cycle(self) -> Dict[str, Any]:
        """
        Execute one complete improvement cycle.

        Returns a full cycle report.
        """
        cycle_start = datetime.now(timezone.utc)
        logger.info("=" * 60)
        logger.info("ARCHITECT AGENT — Starting improvement cycle")
        logger.info("=" * 60)

        # 1. Determine domain
        domain = self._determine_next_domain()
        cycle_id = f"{cycle_start.strftime('%Y-%m-%d')}_{domain.name}"
        logger.info("Domain: %s (priority %d) — %s", domain.name, domain.priority, domain.description)

        # 2. Scan
        scan = self._scan(domain)
        logger.info("Scan: %d file snippets, %d findings", len(scan.file_snippets), len(scan.findings))

        # 3. GPT analysis
        before_metrics = scan.metrics.copy()
        gpt_analysis = self._analyze_with_gpt(domain, scan)

        # 4. Create Claude task
        task = self._create_task_for_claude(domain, gpt_analysis)

        # 5. Implement (if task exists)
        actions_taken = []
        implementation = {"status": "skipped"}
        if task:
            implementation = self._implement(task)
            actions_taken = implementation.get("actions", [])
            logger.info("Implementation: %s — %d actions", implementation["status"], len(actions_taken))
        else:
            logger.info("No changes needed for %s", domain.name)

        # 6. Validate
        validation = self._validate(domain, before_metrics)
        logger.info("Validation: tests_pass=%s", validation["tests_pass"])

        # 7. Build report
        report = {
            "cycle_id": cycle_id,
            "domain": domain.name,
            "timestamp": cycle_start.isoformat(),
            "duration_sec": (datetime.now(timezone.utc) - cycle_start).total_seconds(),
            "scan_findings": scan.findings,
            "gpt_diagnosis": gpt_analysis.get("diagnosis", ""),
            "gpt_suggestion": gpt_analysis.get("suggestion", ""),
            "gpt_validation": gpt_analysis.get("validation", ""),
            "actions_taken": [
                {"type": a.get("type"), "summary": str(a)[:200]}
                for a in actions_taken
            ],
            "validation_result": {
                "tests_pass": validation["tests_pass"],
                "metrics_before": {k: v for k, v in before_metrics.items()
                                   if not isinstance(v, (dict, list))},
                "metrics_after": {k: v for k, v in validation.get("metrics_after", {}).items()
                                  if not isinstance(v, (dict, list))},
            },
            "next_priority": self._next_domain_hint(domain),
        }

        # 8. Log + publish
        self.log.log_cycle(report)

        if self.bus:
            self.bus.publish("agent_architect", {
                "status": "completed",
                "domain": domain.name,
                "findings": len(scan.findings),
                "actions": len(actions_taken),
                "tests_pass": validation["tests_pass"],
                "diagnosis": gpt_analysis.get("diagnosis", "")[:200],
            })

        logger.info("Cycle complete: %s — %d actions, tests=%s",
                     domain.name, len(actions_taken), validation["tests_pass"])
        return report

    def _next_domain_hint(self, current: ScanDomain) -> str:
        """Suggest next domain after current."""
        idx = DOMAIN_NAMES.index(current.name) if current.name in DOMAIN_NAMES else 0
        next_idx = (idx + 1) % len(DOMAIN_NAMES)
        return DOMAIN_NAMES[next_idx]


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(once: bool = False) -> None:
    """Run the Architect Agent."""
    if not _IMPORTS_OK:
        logger.error("Architect: required imports failed — cannot run")
        return

    # Register
    if get_registry:
        registry = get_registry()
        registry.register("agent_architect", role="System self-improvement architect")
    else:
        registry = None

    bus = get_bus() if get_bus else None

    while True:
        try:
            if registry:
                registry.heartbeat("agent_architect", AgentStatus.RUNNING)

            agent = ArchitectAgent()
            report = agent.run_cycle()

            if registry:
                registry.heartbeat("agent_architect", AgentStatus.COMPLETED)

            logger.info(
                "Architect cycle done: domain=%s, actions=%d, tests=%s",
                report.get("domain"),
                len(report.get("actions_taken", [])),
                report.get("validation_result", {}).get("tests_pass"),
            )

        except Exception as exc:
            logger.error("Architect cycle failed: %s", exc, exc_info=True)
            if registry:
                registry.heartbeat("agent_architect", AgentStatus.FAILED, error=str(exc))
            if bus:
                bus.publish("agent_architect", {"status": "failed", "error": str(exc)})

        if once:
            break

        # Sleep until next day (agent runs once per day)
        logger.info("Sleeping until next scheduled run...")
        time.sleep(86400)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Architect Agent — System Self-Improvement")
    parser.add_argument("--once", action="store_true", default=True, help="Run one cycle and exit")
    parser.add_argument("--domain", type=str, help="Force specific domain (e.g., signal_quality)")
    args = parser.parse_args()

    if args.domain and args.domain in DOMAIN_MAP:
        # Override domain selection
        logger.info("Forcing domain: %s", args.domain)
        agent = ArchitectAgent()
        # Monkey-patch to force domain
        original = agent._determine_next_domain
        agent._determine_next_domain = lambda: DOMAIN_MAP[args.domain]
        report = agent.run_cycle()
        print(json.dumps(report, indent=2, default=str))
    else:
        run(once=args.once)
