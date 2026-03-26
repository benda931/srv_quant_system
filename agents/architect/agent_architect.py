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
import os
import re
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
        enrich_domain_with_dynamic_data,
    )
    from agents.architect.improvement_log import (
        get_improvement_log,
        get_architecture_debt_summary,
        get_recurring_domain_issues,
        get_failed_interventions,
        get_domain_health_trend,
        get_unresolved_flags,
    )
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

# ─── All 10 agents in the system ────────────────────────────────────────────
AGENT_NAMES: List[str] = [
    "agent_methodology", "agent_optimizer", "agent_risk_guardian",
    "agent_regime_forecaster", "agent_execution", "agent_data_scout",
    "agent_math", "agent_alpha_decay", "agent_portfolio_construction",
    "agent_architect",
]

# ─── Architecture bottleneck categories ─────────────────────────────────────
BOTTLENECK_CATEGORIES: List[str] = [
    "CONTRACT_DRIFT", "OVERCOUPLING", "DUPLICATED_LOGIC",
    "POOR_FAULT_ISOLATION", "OBSERVABILITY_GAP", "TEST_PROTECTION_GAP",
    "STATE_FRAGMENTATION", "CONFIGURATION_DRIFT",
    "AGENT_COORDINATION_FRICTION", "PERFORMANCE_BOTTLENECK",
    "DATA_LINEAGE_WEAKNESS", "RELIABILITY_RISK",
    "TECH_DEBT_ACCUMULATION", "LOW_PRIORITY_NO_ACTION",
]

# ─── Change policies ────────────────────────────────────────────────────────
CHANGE_POLICIES: List[str] = [
    "NO_ACTION", "SAFE_PATCH", "TARGETED_REFACTOR",
    "CONTRACT_HARDENING", "TEST_FIRST_REFACTOR",
    "OBSERVABILITY_ONLY", "FREEZE_PENDING_REVIEW",
    "ESCALATE_STRUCTURAL_REVIEW",
]


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
        """Scan the domain — read files, collect metrics, run health checks.
        Enriches the result with concrete quantitative metrics."""
        result = scan_domain(domain)

        # Enrich with concrete metrics from actual system state
        extra_metrics = self._collect_concrete_metrics(domain)
        if extra_metrics:
            result.metrics.update(extra_metrics)

        return result

    def _collect_concrete_metrics(self, domain: ScanDomain) -> Dict[str, Any]:
        """Collect actual quantitative metrics for a domain — not just file lists."""
        metrics: Dict[str, Any] = {}

        try:
            if domain.name == "signal_quality":
                # Get actual backtest numbers
                cache_path = ROOT / "logs" / "last_backtest.json"
                if cache_path.exists():
                    bt = json.loads(cache_path.read_text(encoding="utf-8"))
                    metrics["backtest_sharpe"] = bt.get("sharpe", 0)
                    metrics["backtest_ic"] = bt.get("ic_mean", 0)
                    metrics["backtest_hit_rate"] = bt.get("hit_rate", 0)
                    metrics["backtest_max_dd"] = bt.get("max_drawdown", bt.get("max_dd", 0))
                    metrics["regime_breakdown"] = bt.get("regime_breakdown", {})

                # Methodology agent smart conclusions
                if self.bus:
                    meth = self.bus.latest("agent_methodology")
                    if isinstance(meth, dict):
                        smart = meth.get("smart_conclusions", [])
                        if smart:
                            metrics["top_issue"] = smart[0].get("finding", "")
                            metrics["top_action"] = smart[0].get("action", "")
                            metrics["n_critical_issues"] = sum(
                                1 for s in smart if s.get("priority") in ("CRITICAL", "HIGH")
                            )

            elif domain.name == "risk_management":
                # Get actual risk metrics from tail_risk
                if self.bus:
                    meth = self.bus.latest("agent_methodology")
                    if isinstance(meth, dict):
                        tail = meth.get("tail_risk", {})
                        if tail:
                            metrics["es_97_5"] = tail.get("es_97_5_1d", 0)
                            metrics["var_97_5"] = tail.get("var_97_5_1d", 0)
                            metrics["es_var_ratio"] = tail.get("es_var_ratio", 0)
                            metrics["panic_coupling"] = tail.get("panic_coupling", False)
                        stress = meth.get("stress_summary", {})
                        if stress:
                            metrics["worst_stress_pnl"] = stress.get("worst_pnl_pct", 0)
                            metrics["worst_stress_scenario"] = stress.get("worst_scenario", "N/A")

            elif domain.name == "trade_execution":
                # Dispersion backtest results
                if self.bus:
                    meth = self.bus.latest("agent_methodology")
                    if isinstance(meth, dict):
                        disp = meth.get("dispersion_backtest", {})
                        if disp:
                            metrics["disp_sharpe"] = disp.get("sharpe", 0)
                            metrics["disp_win_rate"] = disp.get("win_rate", 0)
                            metrics["disp_total_trades"] = disp.get("total_trades", 0)
                            metrics["disp_max_dd"] = disp.get("max_drawdown", 0)

            elif domain.name == "agent_health":
                # Get optimizer and math agent results
                if self.bus:
                    opt = self.bus.latest("agent_optimizer")
                    if isinstance(opt, dict):
                        metrics["optimizer_outcome"] = opt.get("outcome", "unknown")
                        metrics["optimizer_delta_sharpe"] = opt.get("delta_sharpe", 0)
                    math_r = self.bus.latest("agent_math")
                    if isinstance(math_r, dict):
                        metrics["math_proposals_count"] = math_r.get("proposals_count", 0)

            elif domain.name == "data_quality":
                # Check parquet file freshness + completeness
                import pandas as pd
                prices_path = ROOT / "data_lake" / "parquet" / "prices.parquet"
                if prices_path.exists():
                    prices = pd.read_parquet(prices_path)
                    from datetime import date
                    last_date = prices.index[-1]
                    days_stale = (pd.Timestamp(date.today()) - last_date).days
                    metrics["data_staleness_days"] = int(days_stale)
                    metrics["n_sectors"] = len(prices.columns)
                    metrics["nan_pct"] = round(float(prices.isna().mean().mean()), 4)
                    metrics["date_range"] = f"{prices.index[0].date()} to {last_date.date()}"

        except Exception as e:
            logger.debug("Error collecting concrete metrics for %s: %s", domain.name, e)

        return metrics

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

        # ── Turn 1: Diagnosis with concrete numbers ─────────────
        # Build a focused metrics summary with actual numbers
        concrete_nums = ""
        m = scan.metrics
        if m.get("backtest_sharpe") is not None:
            concrete_nums += f"Sharpe={m['backtest_sharpe']}, IC={m.get('backtest_ic', 'N/A')}, "
            concrete_nums += f"HR={m.get('backtest_hit_rate', 'N/A')}, MaxDD={m.get('backtest_max_dd', 'N/A')}. "
        if m.get("top_issue"):
            concrete_nums += f"Top issue from PM analysis: {m['top_issue']}. "
        if m.get("disp_sharpe") is not None:
            concrete_nums += f"Dispersion Sharpe={m['disp_sharpe']}, WR={m.get('disp_win_rate', 'N/A')}. "
        if m.get("data_staleness_days") is not None:
            concrete_nums += f"Data staleness: {m['data_staleness_days']} days. "
        if m.get("worst_stress_pnl") is not None:
            concrete_nums += f"Worst stress: {m['worst_stress_pnl']}% ({m.get('worst_stress_scenario', '?')}). "

        turn1 = (
            f"Domain: {domain.name} — {domain.description}\n\n"
            f"CONCRETE METRICS:\n{concrete_nums if concrete_nums else metrics_str}\n\n"
            f"Health check findings:\n{findings_str}\n\n"
            f"Key code:\n{code_summary}\n\n"
            f"Given these SPECIFIC NUMBERS, what is the SINGLE most impactful improvement? "
            f"Name the function, the metric it affects, and the expected improvement in Sharpe/IC."
        )
        diagnosis = self.gpt._query(turn1)
        logger.info("GPT Turn 1 (diagnosis): %s", diagnosis[:200])

        # ── Turn 2: Specifics with parameter guidance ─────────
        turn2 = (
            f"You identified: {diagnosis[:300]}\n\n"
            f"What EXACT code change or parameter adjustment fixes this?\n"
            f"Give me:\n"
            f"1. File path and function name\n"
            f"2. The specific change (if parameter: name, current value, new value)\n"
            f"3. Mathematical justification (one sentence)\n"
            f"4. Expected quantitative impact (Sharpe delta or IC delta)\n"
            f"If code: describe the change in 2-3 sentences with the mathematical formula."
        )
        suggestion = self.gpt._query(turn2)
        logger.info("GPT Turn 2 (suggestion): %s", suggestion[:200])

        # ── Turn 3: Validation with specific criteria ─────────
        top_action = m.get("top_action", "")
        turn3 = (
            f"You suggest: {suggestion[:300]}\n\n"
            f"PM analysis suggests: {top_action[:200]}\n\n"
            f"1. What specific metric threshold confirms success? (e.g., 'Sharpe > 0.5')\n"
            f"2. What could go WRONG? Name the specific risk.\n"
            f"3. Should this change be applied unconditionally or only in certain regimes?\n"
            f"Give me a pass/fail criterion in one sentence."
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

        validation_criteria = gpt_analysis.get("validation", "tests must pass")

        system_prompt = (
            f"You are a quantitative systems engineer improving the SRV Quant System.\n\n"
            f"DOMAIN: {domain.name} — {domain.description}\n\n"
            f"DIAGNOSIS: {diagnosis[:500]}\n\n"
            f"RECOMMENDATION: {suggestion[:500]}\n\n"
            f"VALIDATION CRITERION: {validation_criteria[:300]}\n\n"
            f"RULES:\n"
            f"- Make MAXIMUM {MAX_CHANGES_PER_CYCLE} changes\n"
            f"- Each change must be testable\n"
            f"- Do NOT change unrelated code\n"
            f"- Focus on the specific recommendation above\n"
            f"- Parameter changes: use exact values, stay within settings.py Field bounds\n"
            f"- Prefer small incremental changes (10-20%) over large jumps\n"
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

        # ── Institutional Architecture Governance Pipeline ────────
        # Steps 6b-6k: Architecture-level analysis and governance

        # 6b. Assemble architecture inputs
        arch_inputs: Dict[str, Any] = {}
        try:
            arch_inputs = self.assemble_architecture_inputs()
        except Exception as e:
            logger.warning("Architecture input assembly failed: %s", e)

        # 6c. Diagnose architecture
        arch_diagnosis: Dict[str, Any] = {"primary_bottleneck": "LOW_PRIORITY_NO_ACTION", "severity": 0.0, "findings": [], "findings_count": 0}
        try:
            arch_diagnosis = self.diagnose_architecture(arch_inputs)
        except Exception as e:
            logger.warning("Architecture diagnosis failed: %s", e)

        # 6d. Check contracts
        contract_health: Dict[str, Any] = {"contract_integrity_score": 0.5, "violations": [], "drift_warnings": []}
        try:
            contract_health = self.check_contract_health()
        except Exception as e:
            logger.warning("Contract health check failed: %s", e)

        # 6e. Analyze dependencies
        dep_analysis: Dict[str, Any] = {"coupling_score": 0.5, "modularity_score": 0.5, "hotspot_files": []}
        try:
            dep_analysis = self.analyze_dependencies()
        except Exception as e:
            logger.warning("Dependency analysis failed: %s", e)

        # 6f. Analyze observability
        obs_analysis: Dict[str, Any] = {"observability_score": 0.5, "gaps": []}
        try:
            obs_analysis = self.analyze_observability()
        except Exception as e:
            logger.warning("Observability analysis failed: %s", e)

        # 6g. Compute architecture scores
        arch_scores: Dict[str, Any] = {}
        try:
            arch_scores = self.compute_architecture_scores(
                arch_inputs, arch_diagnosis, contract_health, dep_analysis, obs_analysis
            )
        except Exception as e:
            logger.warning("Architecture score computation failed: %s", e)

        # 6h. Determine change policy
        change_policy = "NO_ACTION"
        try:
            change_policy = self.determine_change_policy(arch_diagnosis)
        except Exception as e:
            logger.warning("Change policy determination failed: %s", e)

        # 6i. Assess blast radius before any change
        blast_radius: Dict[str, Any] = {"blast_radius_score": 0.0, "safe_to_proceed": True}
        try:
            changed_files = [a.get("file", "") for a in actions_taken if isinstance(a, dict) and a.get("file")]
            blast_radius = self.assess_blast_radius(changed_files if changed_files else None)
        except Exception as e:
            logger.warning("Blast radius assessment failed: %s", e)

        # 6j. Build downstream contract
        downstream: Dict[str, Any] = {}
        try:
            downstream = self.build_downstream_contract(arch_diagnosis, change_policy)
        except Exception as e:
            logger.warning("Downstream contract build failed: %s", e)

        # 6k. Build machine_summary
        machine_summary: Dict[str, Any] = {}
        try:
            machine_summary = self._build_machine_summary(
                diagnosis=arch_diagnosis,
                policy=change_policy,
                scores=arch_scores,
                actions_count=len(actions_taken),
                domain_hint=self._next_domain_hint(domain),
            )
        except Exception as e:
            logger.warning("Machine summary build failed: %s", e)

        # ── End Institutional Pipeline ────────────────────────────

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
            # Institutional governance additions
            "architecture_diagnosis": arch_diagnosis.get("primary_bottleneck", ""),
            "change_policy": change_policy,
            "architecture_scores": arch_scores,
            "contract_health": {
                "integrity_score": contract_health.get("contract_integrity_score", 0),
                "violations": len(contract_health.get("violations", [])),
                "drift_warnings": len(contract_health.get("drift_warnings", [])),
            },
            "blast_radius": blast_radius,
            "machine_summary": machine_summary,
        }

        # 8. Log + publish
        self.log.log_cycle(report)

        if self.bus:
            publish_payload = machine_summary if machine_summary else {
                "status": "completed",
                "domain": domain.name,
                "findings": len(scan.findings),
                "actions": len(actions_taken),
                "tests_pass": validation["tests_pass"],
                "diagnosis": gpt_analysis.get("diagnosis", "")[:200],
            }
            self.bus.publish("agent_architect", publish_payload)

        logger.info("Cycle complete: %s — %d actions, tests=%s",
                     domain.name, len(actions_taken), validation["tests_pass"])
        return report

    # ══════════════════════════════════════════════════════════════════════════
    # INSTITUTIONAL UPGRADE — Chief Systems Architect & Architecture Governance
    # ══════════════════════════════════════════════════════════════════════════

    # ── 1. Architecture Input Assembler ───────────────────────────

    def assemble_architecture_inputs(self) -> Dict[str, Any]:
        """
        Load machine_summary from ALL 10 agents + registry status + test results.
        Each with try/except for graceful degradation.
        """
        inputs: Dict[str, Any] = {
            "agent_summaries": {},
            "registry_status": {},
            "test_results": {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Collect machine_summary from each agent via bus
        for agent_name in AGENT_NAMES:
            try:
                if self.bus:
                    summary = self.bus.latest(agent_name)
                    if isinstance(summary, dict):
                        inputs["agent_summaries"][agent_name] = summary
                    else:
                        inputs["agent_summaries"][agent_name] = {"status": "no_data"}
                else:
                    inputs["agent_summaries"][agent_name] = {"status": "bus_unavailable"}
            except Exception as e:
                inputs["agent_summaries"][agent_name] = {"status": "error", "error": str(e)}

        # Registry status
        try:
            if self.registry:
                all_agents = self.registry.all_agents()
                inputs["registry_status"] = {
                    "total": len(all_agents),
                    "agents": {
                        name: info.get("status", "UNKNOWN") if isinstance(info, dict) else "UNKNOWN"
                        for name, info in all_agents.items()
                    },
                    "failed": [
                        n for n, info in all_agents.items()
                        if isinstance(info, dict) and info.get("status") in ("FAILED", "STALE")
                    ],
                }
            else:
                inputs["registry_status"] = {"status": "registry_unavailable"}
        except Exception as e:
            inputs["registry_status"] = {"status": "error", "error": str(e)}

        # Test results
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", str(ROOT / "tests"), "-q", "--tb=no",
                 "--ignore=" + str(ROOT / "tests" / "test_execution.py")],
                capture_output=True, text=True, timeout=120, cwd=str(ROOT),
            )
            output = result.stdout + result.stderr
            passed = failed = 0
            for line in output.splitlines():
                if "passed" in line:
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if p == "passed" and i > 0:
                            try: passed = int(parts[i - 1])
                            except ValueError: pass
                        if p == "failed" and i > 0:
                            try: failed = int(parts[i - 1])
                            except ValueError: pass
            inputs["test_results"] = {
                "passed": passed, "failed": failed,
                "all_pass": failed == 0, "output_tail": output[-300:]
            }
        except Exception as e:
            inputs["test_results"] = {"status": "error", "error": str(e), "all_pass": False}

        # Architecture debt from improvement log
        try:
            inputs["debt_summary"] = get_architecture_debt_summary()
        except Exception as e:
            inputs["debt_summary"] = {"status": "error", "error": str(e)}

        logger.info("Architecture inputs assembled: %d agent summaries, registry=%s, tests=%s",
                     len(inputs["agent_summaries"]),
                     inputs["registry_status"].get("total", "?"),
                     inputs["test_results"].get("all_pass", "?"))
        return inputs

    # ── 2. Architecture Diagnosis Engine ──────────────────────────

    def diagnose_architecture(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deterministic rule-based architecture diagnosis.
        Returns primary bottleneck, severity, and supporting evidence.
        """
        findings: List[Dict[str, Any]] = []
        summaries = inputs.get("agent_summaries", {})
        registry = inputs.get("registry_status", {})
        tests = inputs.get("test_results", {})
        debt = inputs.get("debt_summary", {})

        # Rule 1: CONTRACT_DRIFT — schema inconsistencies across agents
        schema_versions = set()
        for name, s in summaries.items():
            if isinstance(s, dict):
                v = s.get("schema_version") or s.get("version")
                if v:
                    schema_versions.add(str(v))
        if len(schema_versions) > 1:
            findings.append({
                "category": "CONTRACT_DRIFT",
                "severity": 0.8,
                "evidence": f"Multiple schema versions across agents: {schema_versions}",
            })

        # Rule 2: OVERCOUPLING — check dependency analysis (deferred to analyze_dependencies)
        # Placeholder: detected if any module has fan-in > 8
        # Actual check done when deps are available

        # Rule 3: POOR_FAULT_ISOLATION — failed agents affecting others
        failed_agents = registry.get("failed", [])
        if len(failed_agents) >= 2:
            findings.append({
                "category": "POOR_FAULT_ISOLATION",
                "severity": 0.9,
                "evidence": f"Multiple agents failed: {failed_agents}",
            })
        elif len(failed_agents) == 1:
            findings.append({
                "category": "RELIABILITY_RISK",
                "severity": 0.5,
                "evidence": f"Agent failed: {failed_agents[0]}",
            })

        # Rule 4: TEST_PROTECTION_GAP — tests failing
        if not tests.get("all_pass", True):
            n_failed = tests.get("failed", 0)
            findings.append({
                "category": "TEST_PROTECTION_GAP",
                "severity": min(0.6 + n_failed * 0.05, 1.0),
                "evidence": f"{n_failed} test(s) failing",
            })

        # Rule 5: STATE_FRAGMENTATION — agents without machine_summary
        missing_summaries = [
            n for n, s in summaries.items()
            if isinstance(s, dict) and s.get("status") in ("no_data", "error", "bus_unavailable")
        ]
        if len(missing_summaries) >= 3:
            findings.append({
                "category": "STATE_FRAGMENTATION",
                "severity": 0.7,
                "evidence": f"Agents missing machine_summary: {missing_summaries}",
            })

        # Rule 6: AGENT_COORDINATION_FRICTION — stale agents
        stale_agents = [
            n for n, st in registry.get("agents", {}).items()
            if st in ("STALE",)
        ]
        if stale_agents:
            findings.append({
                "category": "AGENT_COORDINATION_FRICTION",
                "severity": 0.5,
                "evidence": f"Stale agents: {stale_agents}",
            })

        # Rule 7: TECH_DEBT_ACCUMULATION — from debt summary
        total_debt = debt.get("total_unresolved", 0)
        if total_debt >= 5:
            findings.append({
                "category": "TECH_DEBT_ACCUMULATION",
                "severity": min(0.4 + total_debt * 0.05, 1.0),
                "evidence": f"{total_debt} unresolved debt items",
            })

        # Rule 8: OBSERVABILITY_GAP — checked in detail by analyze_observability
        # Placeholder: if many agents lack structured output
        agents_with_scores = sum(
            1 for s in summaries.values()
            if isinstance(s, dict) and any(
                k for k in s.keys() if "score" in k.lower() or "health" in k.lower()
            )
        )
        if agents_with_scores < len(summaries) * 0.5 and len(summaries) > 3:
            findings.append({
                "category": "OBSERVABILITY_GAP",
                "severity": 0.5,
                "evidence": f"Only {agents_with_scores}/{len(summaries)} agents report scores",
            })

        # Rule 9: DATA_LINEAGE_WEAKNESS — check if data_scout has lineage
        ds = summaries.get("agent_data_scout", {})
        if isinstance(ds, dict) and not ds.get("lineage_tracked") and ds.get("status") != "no_data":
            findings.append({
                "category": "DATA_LINEAGE_WEAKNESS",
                "severity": 0.4,
                "evidence": "Data scout not tracking lineage",
            })

        # Rule 10: CONFIGURATION_DRIFT — recurring domain issues
        try:
            recurring = get_recurring_domain_issues(n=5)
            if len(recurring) >= 3:
                findings.append({
                    "category": "CONFIGURATION_DRIFT",
                    "severity": 0.6,
                    "evidence": f"{len(recurring)} recurring domain issues detected",
                })
        except Exception:
            pass

        # Sort by severity descending
        findings.sort(key=lambda f: f["severity"], reverse=True)

        primary = findings[0]["category"] if findings else "LOW_PRIORITY_NO_ACTION"
        severity = findings[0]["severity"] if findings else 0.0

        diagnosis = {
            "primary_bottleneck": primary,
            "severity": severity,
            "findings": findings,
            "findings_count": len(findings),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        logger.info("Architecture diagnosis: %s (severity=%.2f, %d findings)",
                     primary, severity, len(findings))
        return diagnosis

    # ── 3. Contract Health Engine ─────────────────────────────────

    def check_contract_health(self) -> Dict[str, Any]:
        """
        Check machine_summary schemas, JSON state file consistency,
        and bus payload compatibility across agents.
        """
        violations: List[str] = []
        drift_warnings: List[str] = []
        checked = 0
        passed = 0

        # Check machine_summary schema presence per agent
        expected_fields = {"timestamp", "status"}
        for agent_name in AGENT_NAMES:
            try:
                if self.bus:
                    summary = self.bus.latest(agent_name)
                    if isinstance(summary, dict):
                        checked += 1
                        missing = expected_fields - set(summary.keys())
                        if not missing:
                            passed += 1
                        else:
                            drift_warnings.append(
                                f"{agent_name}: missing base fields {missing}"
                            )
                    else:
                        violations.append(f"{agent_name}: no machine_summary published")
            except Exception as e:
                violations.append(f"{agent_name}: contract check error: {e}")

        # Check JSON state file consistency
        state_dirs = [
            ROOT / "agents" / d for d in
            ["methodology", "optimizer", "risk_guardian", "regime_forecaster",
             "execution", "data_scout", "math", "alpha_decay",
             "portfolio_construction", "architect"]
        ]
        for sd in state_dirs:
            if sd.is_dir():
                for jf in sd.glob("*.json"):
                    try:
                        data = json.loads(jf.read_text(encoding="utf-8"))
                        if not isinstance(data, (dict, list)):
                            violations.append(f"{jf.name}: not a valid JSON structure")
                    except json.JSONDecodeError:
                        violations.append(f"{jf.name}: corrupt JSON")
                    except Exception:
                        pass

        # Check bus payload compatibility (all published payloads should be dicts)
        if self.bus:
            for agent_name in AGENT_NAMES:
                try:
                    payload = self.bus.latest(agent_name)
                    if payload is not None and not isinstance(payload, dict):
                        violations.append(
                            f"{agent_name}: bus payload is {type(payload).__name__}, expected dict"
                        )
                except Exception:
                    pass

        integrity_score = passed / max(checked, 1)
        result = {
            "contract_integrity_score": round(integrity_score, 3),
            "checked": checked,
            "passed": passed,
            "violations": violations,
            "drift_warnings": drift_warnings,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        logger.info("Contract health: integrity=%.2f, %d violations, %d warnings",
                     integrity_score, len(violations), len(drift_warnings))
        return result

    # ── 4. Dependency Analysis ────────────────────────────────────

    def analyze_dependencies(self) -> Dict[str, Any]:
        """
        Scan import statements in key modules, count fan-in/fan-out,
        identify oversized modules (>500 lines).
        """
        key_dirs = [
            ROOT / "analytics",
            ROOT / "agents",
            ROOT / "scripts",
            ROOT / "data_ops",
        ]

        module_imports: Dict[str, List[str]] = {}  # module -> list of imports
        module_lines: Dict[str, int] = {}
        hotspot_files: List[str] = []

        for d in key_dirs:
            if not d.is_dir():
                continue
            for py_file in d.rglob("*.py"):
                if "__pycache__" in str(py_file):
                    continue
                rel = str(py_file.relative_to(ROOT))
                try:
                    text = py_file.read_text(encoding="utf-8", errors="replace")
                    lines = text.splitlines()
                    module_lines[rel] = len(lines)
                    if len(lines) > 500:
                        hotspot_files.append(f"{rel} ({len(lines)} lines)")

                    imports = []
                    for line in lines:
                        stripped = line.strip()
                        if stripped.startswith("from ") or stripped.startswith("import "):
                            # Extract module name
                            match = re.match(r'(?:from|import)\s+([\w.]+)', stripped)
                            if match:
                                imports.append(match.group(1))
                    module_imports[rel] = imports
                except Exception:
                    pass

        # Fan-out: how many modules does each module import
        fan_out: Dict[str, int] = {m: len(imps) for m, imps in module_imports.items()}

        # Fan-in: how many modules import each module (by name matching)
        fan_in: Dict[str, int] = {}
        all_imported = []
        for imps in module_imports.values():
            all_imported.extend(imps)
        for mod in set(all_imported):
            fan_in[mod] = all_imported.count(mod)

        # Coupling score: average fan-out normalized
        avg_fan_out = sum(fan_out.values()) / max(len(fan_out), 1)
        coupling_score = min(avg_fan_out / 20.0, 1.0)  # 20 imports = max coupling

        # Modularity: inverse of coupling + penalty for oversized modules
        oversize_penalty = len(hotspot_files) * 0.05
        modularity_score = max(1.0 - coupling_score - oversize_penalty, 0.0)

        # Top coupled modules
        top_fan_out = sorted(fan_out.items(), key=lambda x: x[1], reverse=True)[:10]
        top_fan_in = sorted(fan_in.items(), key=lambda x: x[1], reverse=True)[:10]

        result = {
            "coupling_score": round(coupling_score, 3),
            "modularity_score": round(modularity_score, 3),
            "total_modules": len(module_imports),
            "avg_fan_out": round(avg_fan_out, 1),
            "hotspot_files": hotspot_files[:15],
            "top_fan_out": top_fan_out,
            "top_fan_in": top_fan_in,
            "oversized_count": len(hotspot_files),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        logger.info("Dependency analysis: coupling=%.2f, modularity=%.2f, %d hotspots",
                     coupling_score, modularity_score, len(hotspot_files))
        return result

    # ── 5. Observability Analysis ─────────────────────────────────

    def analyze_observability(self) -> Dict[str, Any]:
        """
        Check for RotatingFileHandler, structured logging, experiment IDs,
        and timestamps in state files.
        """
        gaps: List[str] = []
        checks_passed = 0
        checks_total = 0

        agent_dirs = [
            ROOT / "agents" / d for d in
            ["methodology", "optimizer", "risk_guardian", "regime_forecaster",
             "execution", "data_scout", "math", "alpha_decay",
             "portfolio_construction", "architect"]
        ]

        # Check 1: RotatingFileHandler usage in agent files
        checks_total += 1
        rfh_count = 0
        for ad in agent_dirs:
            if not ad.is_dir():
                continue
            for py in ad.glob("*.py"):
                try:
                    text = py.read_text(encoding="utf-8", errors="replace")
                    if "RotatingFileHandler" in text:
                        rfh_count += 1
                except Exception:
                    pass
        if rfh_count >= 3:
            checks_passed += 1
        else:
            gaps.append(f"RotatingFileHandler used in only {rfh_count} agent modules")

        # Check 2: Structured logging (JSON logs or dict-based logging)
        checks_total += 1
        structured_count = 0
        for ad in agent_dirs:
            if not ad.is_dir():
                continue
            for py in ad.glob("agent_*.py"):
                try:
                    text = py.read_text(encoding="utf-8", errors="replace")
                    if "machine_summary" in text or "json.dumps" in text:
                        structured_count += 1
                except Exception:
                    pass
        if structured_count >= 5:
            checks_passed += 1
        else:
            gaps.append(f"Structured logging in only {structured_count} agents")

        # Check 3: Experiment IDs in reports
        checks_total += 1
        experiment_id_count = 0
        for ad in agent_dirs:
            if not ad.is_dir():
                continue
            for py in ad.glob("*.py"):
                try:
                    text = py.read_text(encoding="utf-8", errors="replace")
                    if "experiment_id" in text or "cycle_id" in text or "run_id" in text:
                        experiment_id_count += 1
                except Exception:
                    pass
        if experiment_id_count >= 3:
            checks_passed += 1
        else:
            gaps.append(f"Experiment/run IDs found in only {experiment_id_count} modules")

        # Check 4: Timestamps in state files
        checks_total += 1
        ts_present = 0
        ts_checked = 0
        for ad in agent_dirs:
            if not ad.is_dir():
                continue
            for jf in ad.glob("*.json"):
                ts_checked += 1
                try:
                    data = json.loads(jf.read_text(encoding="utf-8"))
                    if isinstance(data, dict) and (
                        "timestamp" in data or "last_updated" in data or "ts" in data
                    ):
                        ts_present += 1
                    elif isinstance(data, list) and data:
                        first = data[0] if isinstance(data[0], dict) else {}
                        if "timestamp" in first:
                            ts_present += 1
                except Exception:
                    pass
        if ts_checked > 0 and ts_present / ts_checked >= 0.7:
            checks_passed += 1
        else:
            gaps.append(
                f"Timestamps in {ts_present}/{ts_checked} state files "
                f"({ts_present / max(ts_checked, 1):.0%})"
            )

        obs_score = checks_passed / max(checks_total, 1)
        result = {
            "observability_score": round(obs_score, 3),
            "checks_passed": checks_passed,
            "checks_total": checks_total,
            "gaps": gaps,
            "rotating_handler_count": rfh_count,
            "structured_logging_count": structured_count,
            "experiment_id_count": experiment_id_count,
            "timestamp_coverage": round(ts_present / max(ts_checked, 1), 3),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        logger.info("Observability: score=%.2f, %d/%d checks, %d gaps",
                     obs_score, checks_passed, checks_total, len(gaps))
        return result

    # ── 6. Change Policy Engine ───────────────────────────────────

    def determine_change_policy(self, diagnosis: Dict[str, Any]) -> str:
        """
        Map architecture diagnosis to a change policy.
        Deterministic rules based on bottleneck category and severity.
        """
        primary = diagnosis.get("primary_bottleneck", "LOW_PRIORITY_NO_ACTION")
        severity = diagnosis.get("severity", 0.0)
        n_findings = diagnosis.get("findings_count", 0)

        # Severity thresholds
        if severity >= 0.9 and n_findings >= 3:
            return "ESCALATE_STRUCTURAL_REVIEW"

        if severity >= 0.85:
            return "FREEZE_PENDING_REVIEW"

        # Category-specific policies
        policy_map = {
            "CONTRACT_DRIFT":               "CONTRACT_HARDENING",
            "OVERCOUPLING":                 "TARGETED_REFACTOR",
            "DUPLICATED_LOGIC":             "TARGETED_REFACTOR",
            "POOR_FAULT_ISOLATION":         "ESCALATE_STRUCTURAL_REVIEW",
            "OBSERVABILITY_GAP":            "OBSERVABILITY_ONLY",
            "TEST_PROTECTION_GAP":          "TEST_FIRST_REFACTOR",
            "STATE_FRAGMENTATION":          "CONTRACT_HARDENING",
            "CONFIGURATION_DRIFT":          "SAFE_PATCH",
            "AGENT_COORDINATION_FRICTION":  "SAFE_PATCH",
            "PERFORMANCE_BOTTLENECK":       "TARGETED_REFACTOR",
            "DATA_LINEAGE_WEAKNESS":        "OBSERVABILITY_ONLY",
            "RELIABILITY_RISK":             "SAFE_PATCH",
            "TECH_DEBT_ACCUMULATION":       "TEST_FIRST_REFACTOR",
            "LOW_PRIORITY_NO_ACTION":       "NO_ACTION",
        }

        policy = policy_map.get(primary, "SAFE_PATCH")

        # Downgrade if severity is low
        if severity < 0.3 and policy not in ("NO_ACTION",):
            policy = "NO_ACTION"

        logger.info("Change policy: %s (bottleneck=%s, severity=%.2f)",
                     policy, primary, severity)
        return policy

    # ── 7. Blast Radius Engine ────────────────────────────────────

    def assess_blast_radius(self, files_to_change: List[str] = None) -> Dict[str, Any]:
        """
        For a set of files about to be changed, assess downstream impact.
        Returns blast_radius_score (0-1) and safe_to_proceed.
        """
        if not files_to_change:
            return {
                "blast_radius_score": 0.0,
                "safe_to_proceed": True,
                "downstream_count": 0,
                "contracts_affected": False,
                "test_coverage_present": True,
            }

        downstream_count = 0
        contracts_affected = False
        test_files_exist = 0

        for fpath in files_to_change:
            # Count how many other files import from this module
            module_name = fpath.replace("/", ".").replace("\\", ".").rstrip(".py")
            parts = module_name.split(".")
            short_name = parts[-1] if parts else module_name

            try:
                for d in [ROOT / "analytics", ROOT / "agents", ROOT / "scripts"]:
                    if not d.is_dir():
                        continue
                    for py in d.rglob("*.py"):
                        if "__pycache__" in str(py):
                            continue
                        try:
                            text = py.read_text(encoding="utf-8", errors="replace")
                            if short_name in text and str(py) != str(ROOT / fpath):
                                downstream_count += 1
                        except Exception:
                            pass
            except Exception:
                pass

            # Check if contracts (schemas, JSON state) are in the changed files
            if any(kw in fpath for kw in ["schema", "contract", "registry", "bus"]):
                contracts_affected = True

            # Check for test coverage
            base = Path(fpath).stem
            test_path = ROOT / "tests" / f"test_{base}.py"
            if test_path.exists():
                test_files_exist += 1

        # Score: 0 = no blast, 1 = high blast
        blast_score = min(
            (downstream_count * 0.05)
            + (0.3 if contracts_affected else 0.0)
            + (0.2 if test_files_exist == 0 else 0.0),
            1.0,
        )
        safe = blast_score < 0.6 and not contracts_affected

        result = {
            "blast_radius_score": round(blast_score, 3),
            "safe_to_proceed": safe,
            "downstream_count": downstream_count,
            "contracts_affected": contracts_affected,
            "test_coverage_present": test_files_exist > 0,
            "files_assessed": len(files_to_change),
        }
        logger.info("Blast radius: score=%.2f, safe=%s, downstream=%d",
                     blast_score, safe, downstream_count)
        return result

    # ── 8. Architecture Scores ────────────────────────────────────

    def compute_architecture_scores(
        self,
        inputs: Dict[str, Any],
        diagnosis: Dict[str, Any],
        contracts: Dict[str, Any],
        deps: Dict[str, Any],
        obs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compute composite architecture scores from all analysis outputs.
        """
        # Contract integrity (direct from contract engine)
        contract_score = contracts.get("contract_integrity_score", 0.5)

        # Modularity (from dependency analysis)
        modularity_score = deps.get("modularity_score", 0.5)

        # Observability (from observability analysis)
        observability_score = obs.get("observability_score", 0.5)

        # Reliability: fraction of agents not failed
        registry = inputs.get("registry_status", {})
        total_agents = registry.get("total", len(AGENT_NAMES))
        n_failed = len(registry.get("failed", []))
        reliability_score = (total_agents - n_failed) / max(total_agents, 1)

        # Technical debt: inverse of unresolved issues
        debt = inputs.get("debt_summary", {})
        unresolved = debt.get("total_unresolved", 0)
        tech_debt_score = max(1.0 - (unresolved * 0.08), 0.0)

        # Test health
        tests = inputs.get("test_results", {})
        test_pass_rate = 1.0 if tests.get("all_pass", False) else (
            tests.get("passed", 0) / max(tests.get("passed", 0) + tests.get("failed", 1), 1)
        )

        # Architecture health: weighted composite
        arch_health = (
            contract_score * 0.20
            + modularity_score * 0.15
            + observability_score * 0.15
            + reliability_score * 0.20
            + tech_debt_score * 0.15
            + test_pass_rate * 0.15
        )

        # Change safety: inverse of diagnosis severity
        diag_severity = diagnosis.get("severity", 0.0)
        change_safety = max(1.0 - diag_severity, 0.0)

        scores = {
            "architecture_health_score": round(arch_health, 3),
            "modularity_score": round(modularity_score, 3),
            "contract_integrity_score": round(contract_score, 3),
            "observability_score": round(observability_score, 3),
            "reliability_score": round(reliability_score, 3),
            "technical_debt_score": round(tech_debt_score, 3),
            "test_pass_rate": round(test_pass_rate, 3),
            "architecture_change_safety_score": round(change_safety, 3),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        logger.info("Architecture scores: health=%.2f, modularity=%.2f, contracts=%.2f",
                     arch_health, modularity_score, contract_score)
        return scores

    # ── 9. Domain Prioritizer v2 ──────────────────────────────────

    def prioritize_domain_v2(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score-based domain prioritization replacing rotation.
        Score = business_criticality x failure_severity x recurrence x debt x blast_radius_inv
        """
        domain_scores: Dict[str, Dict[str, Any]] = {}
        debt = inputs.get("debt_summary", {})
        recurring = debt.get("recurring_domains", {})

        # Business criticality weights per domain
        criticality = {
            "signal_quality": 1.0, "risk_management": 0.95,
            "trade_execution": 0.9, "data_quality": 0.85,
            "agent_health": 0.8, "dashboard_accuracy": 0.5,
            "code_quality": 0.6, "performance": 0.65,
        }

        for domain in SCAN_DOMAINS:
            name = domain.name
            crit = criticality.get(name, 0.5)

            # Failure severity: check if bus has failure signals
            severity = 0.1
            try:
                if self.bus:
                    summary = self.bus.latest(f"agent_{name}") or {}
                    if isinstance(summary, dict):
                        if summary.get("status") == "failed":
                            severity = 1.0
                        elif summary.get("status") == "degraded":
                            severity = 0.7
                        elif any(
                            k for k in summary if "error" in str(k).lower()
                            and summary[k]
                        ):
                            severity = 0.5
            except Exception:
                pass

            # Recurrence from history
            recurrence = min(recurring.get(name, 0) * 0.15, 1.0)

            # Debt accumulation
            debt_count = debt.get("domain_debt", {}).get(name, 0)
            debt_factor = min(debt_count * 0.1, 1.0)

            # Days since last scan (urgency)
            last = self.log.last_scan_date(name)
            if last:
                try:
                    from datetime import timedelta
                    days_ago = (datetime.now(timezone.utc).date() -
                                datetime.fromisoformat(last).date()).days
                    staleness = min(days_ago / 14.0, 1.0)
                except Exception:
                    staleness = 0.5
            else:
                staleness = 1.0

            # Composite score
            composite = (
                crit * 0.30
                + severity * 0.25
                + recurrence * 0.15
                + debt_factor * 0.15
                + staleness * 0.15
            )

            domain_scores[name] = {
                "composite": round(composite, 3),
                "criticality": crit,
                "failure_severity": severity,
                "recurrence": recurrence,
                "debt_factor": debt_factor,
                "staleness": staleness,
            }

        # Sort by composite descending
        ranked = sorted(domain_scores.items(), key=lambda x: x[1]["composite"], reverse=True)
        top_domain = ranked[0][0] if ranked else DOMAIN_NAMES[0]

        result = {
            "recommended_domain": top_domain,
            "domain_scores": domain_scores,
            "ranking": [r[0] for r in ranked],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        logger.info("Domain prioritization v2: top=%s (score=%.3f)",
                     top_domain, domain_scores.get(top_domain, {}).get("composite", 0))
        return result

    # ── 10. Downstream Contract Builder ───────────────────────────

    def build_downstream_contract(self, diagnosis: Dict[str, Any], policy: str) -> Dict[str, Any]:
        """
        For each agent, determine what architecture actions are needed
        based on the diagnosis and selected policy.
        """
        contracts: Dict[str, Dict[str, Any]] = {}
        primary = diagnosis.get("primary_bottleneck", "LOW_PRIORITY_NO_ACTION")
        findings = diagnosis.get("findings", [])

        # Build set of affected categories
        affected_categories = {f["category"] for f in findings if isinstance(f, dict)}

        for agent_name in AGENT_NAMES:
            actions: List[str] = []
            priority = "LOW"

            if policy == "NO_ACTION":
                actions.append("no_action_required")
            elif policy == "FREEZE_PENDING_REVIEW":
                actions.append("freeze_changes")
                priority = "CRITICAL"
            elif policy == "ESCALATE_STRUCTURAL_REVIEW":
                actions.append("await_structural_review")
                priority = "CRITICAL"
            else:
                # Policy-specific actions per agent
                if "CONTRACT_DRIFT" in affected_categories:
                    actions.append("validate_machine_summary_schema")
                    priority = "HIGH"
                if "TEST_PROTECTION_GAP" in affected_categories:
                    actions.append("ensure_test_coverage")
                    priority = max(priority, "MEDIUM")
                if "OBSERVABILITY_GAP" in affected_categories:
                    actions.append("add_structured_logging")
                if "STATE_FRAGMENTATION" in affected_categories:
                    actions.append("consolidate_state_files")
                if "RELIABILITY_RISK" in affected_categories:
                    actions.append("add_error_handling")

                if not actions:
                    actions.append("monitor_only")

            contracts[agent_name] = {
                "actions": actions,
                "priority": priority,
                "policy": policy,
                "source_bottleneck": primary,
            }

        result = {
            "downstream_contracts": contracts,
            "policy": policy,
            "primary_bottleneck": primary,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        logger.info("Downstream contracts built for %d agents under policy=%s",
                     len(contracts), policy)
        return result

    # ── 11. Machine Summary Builder ───────────────────────────────

    def _build_machine_summary(
        self,
        diagnosis: Dict[str, Any],
        policy: str,
        scores: Dict[str, Any],
        actions_count: int,
        domain_hint: str,
    ) -> Dict[str, Any]:
        """
        Build the canonical machine_summary for downstream consumption.
        """
        # Collect unresolved flags
        try:
            unresolved = get_unresolved_flags()
        except Exception:
            unresolved = []

        severity = diagnosis.get("severity", 0.0)

        summary = {
            "agent": "agent_architect",
            "role": "Chief Systems Architect & Architecture Governance Engine",
            "architecture_diagnosis": diagnosis.get("primary_bottleneck", "LOW_PRIORITY_NO_ACTION"),
            "campaign_type": policy,
            "architecture_health_score": scores.get("architecture_health_score", 0.0),
            "contract_integrity_score": scores.get("contract_integrity_score", 0.0),
            "modularity_score": scores.get("modularity_score", 0.0),
            "observability_score": scores.get("observability_score", 0.0),
            "reliability_score": scores.get("reliability_score", 0.0),
            "technical_debt_score": scores.get("technical_debt_score", 0.0),
            "architecture_change_safety_score": scores.get("architecture_change_safety_score", 0.0),
            "changes_applied": actions_count,
            "unresolved_flags": unresolved[:10],
            "freeze_recommended": policy in ("FREEZE_PENDING_REVIEW", "ESCALATE_STRUCTURAL_REVIEW"),
            "structural_review_needed": severity >= 0.85,
            "next_domain_hint": domain_hint,
            "findings_count": diagnosis.get("findings_count", 0),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "schema_version": "2.0.0",
            "status": "completed",
        }
        return summary

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
