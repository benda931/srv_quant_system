"""
agents/shared/agent_interface.py
==================================
Standard agent interface — defines the contract that all agents must follow.

Every agent in the system should:
  1. Have a clear MANDATE (what it does)
  2. Have defined INPUTS (what it reads)
  3. Have defined OUTPUTS (what it writes)
  4. Be AUDITABLE (log what happened)
  5. Not DUPLICATE core application logic

This module provides:
  - AgentResult: standard return type for all agents
  - AgentManifest: describes an agent's role, inputs, outputs
  - AGENT_REGISTRY: catalog of all registered agents
  - run_agent(): safe wrapper that runs any agent with error handling
  - diagnose_agents(): test all agents and report health

Usage:
    from agents.shared.agent_interface import run_agent, AGENT_REGISTRY
    result = run_agent("auto_improve")
    print(result.status, result.duration_s)
"""
from __future__ import annotations

import logging
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger("agent_interface")


@dataclass
class AgentResult:
    """Standard result returned by every agent run."""
    agent_name: str
    status: str                        # "ok" / "failed" / "skipped"
    duration_s: float
    timestamp: str
    # Output summary
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)  # Files written
    error: str = ""
    # Diagnostic
    import_ok: bool = True
    run_ok: bool = True


@dataclass
class AgentManifest:
    """Describes an agent's role, inputs, and outputs."""
    name: str
    mandate: str                       # One-line description of what it does
    module: str                        # Python module path
    class_name: str                    # Class to instantiate
    # I/O contract
    reads_from: List[str]              # What data it consumes
    writes_to: List[str]               # What artifacts it produces
    # Schedule
    schedule: str                      # When it runs (from orchestrator)
    depends_on: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Agent Registry — single source of truth for all agents
# ─────────────────────────────────────────────────────────────────────────────

AGENT_REGISTRY: Dict[str, AgentManifest] = {
    "auto_improve": AgentManifest(
        name="auto_improve",
        mandate="Test parameter changes via backtest and promote improvements to .env.auto_improve",
        module="agents.auto_improve.engine",
        class_name="AutoImprover",
        reads_from=["prices.parquet", "methodology lab reports"],
        writes_to=[".env.auto_improve", "agents/auto_improve/improvement_log.json", "agents/auto_improve/machine_summary.json"],
        schedule="07:30 weekdays",
        depends_on=["methodology", "optimizer"],
    ),
    "methodology": AgentManifest(
        name="methodology",
        mandate="Evaluate all trading strategies via walk-forward backtest and produce rankings",
        module="agents.methodology.agent_methodology",
        class_name="run",
        reads_from=["prices.parquet", "settings"],
        writes_to=["agents/methodology/reports/*.json"],
        schedule="06:00 weekdays",
        depends_on=["data_refresh"],
    ),
    "optimizer": AgentManifest(
        name="optimizer",
        mandate="Search for better parameters via grid/Bayesian optimization",
        module="agents.optimizer.agent_optimizer",
        class_name="run",
        reads_from=["prices.parquet", "methodology reports"],
        writes_to=["agents/optimizer/optimization_history.json"],
        schedule="07:00 weekdays",
        depends_on=["methodology"],
    ),
    "regime_forecaster": AgentManifest(
        name="regime_forecaster",
        mandate="Predict market regime 5 days ahead using ML ensemble",
        module="agents.regime_forecaster.agent_regime_forecaster",
        class_name="RegimeForecaster",
        reads_from=["prices.parquet"],
        writes_to=["agents/regime_forecaster/regime_forecast.json"],
        schedule="every 2h market hours",
    ),
    "risk_guardian": AgentManifest(
        name="risk_guardian",
        mandate="Pre-trade risk checks — VIX kills, exposure limits, trade vetoes",
        module="agents.risk_guardian.agent_risk_guardian",
        class_name="RiskGuardian",
        reads_from=["paper_portfolio.json", "prices.parquet"],
        writes_to=["agents/risk_guardian/risk_status.json"],
        schedule="16:45 weekdays",
        depends_on=["portfolio_construction"],
    ),
    "data_scout": AgentManifest(
        name="data_scout",
        mandate="Scan external data sources for anomalies and opportunities",
        module="agents.data_scout.agent_data_scout",
        class_name="DataScout",
        reads_from=["prices.parquet", "FMP API"],
        writes_to=["agents/data_scout/scout_report.json"],
        schedule="06:05 weekdays",
    ),
    "alpha_decay": AgentManifest(
        name="alpha_decay",
        mandate="Monitor signal staleness and position aging",
        module="agents.alpha_decay.agent_alpha_decay",
        class_name="StrategyEvidenceAssembler",
        reads_from=["methodology reports", "paper_portfolio.json"],
        writes_to=["agents/alpha_decay/decay_status.json"],
        schedule="08:00 weekdays",
        depends_on=["methodology"],
    ),
    "portfolio_construction": AgentManifest(
        name="portfolio_construction",
        mandate="Assemble portfolio from signals with sector weighting and hedging",
        module="agents.portfolio_construction.agent_portfolio_construction",
        class_name="PortfolioConstructor",
        reads_from=["methodology reports", "regime_forecast.json"],
        writes_to=["agents/portfolio_construction/portfolio_weights.json"],
        schedule="16:30 weekdays",
        depends_on=["methodology", "optimizer"],
    ),
    "math": AgentManifest(
        name="math",
        mandate="Propose mathematical improvements to signal formulas via LLM",
        module="agents.math.agent_math",
        class_name="run",
        reads_from=["methodology reports", "analytics code"],
        writes_to=["agents/math/math_proposals/*.json"],
        schedule="Monday 08:00",
    ),
    "architect": AgentManifest(
        name="architect",
        mandate="Scan codebase for improvement opportunities and code quality issues",
        module="agents.architect.agent_architect",
        class_name="run",
        reads_from=["source code", "test results"],
        writes_to=["agents/architect/improvement_history.json"],
        schedule="09:00 weekdays",
        depends_on=["optimizer"],
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Safe Agent Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_agent(agent_name: str, **kwargs) -> AgentResult:
    """
    Run a single agent with error handling and timing.

    This is the standard way to invoke any agent — handles import errors,
    runtime errors, and produces a consistent AgentResult.
    """
    manifest = AGENT_REGISTRY.get(agent_name)
    if not manifest:
        return AgentResult(
            agent_name=agent_name, status="failed", duration_s=0,
            timestamp=datetime.now(timezone.utc).isoformat(),
            error=f"Unknown agent: {agent_name}",
            import_ok=False, run_ok=False,
        )

    ts = datetime.now(timezone.utc)
    t0 = time.time()

    # Import
    try:
        module = __import__(manifest.module, fromlist=[manifest.class_name])
        agent_class = getattr(module, manifest.class_name)
    except Exception as e:
        return AgentResult(
            agent_name=agent_name, status="failed",
            duration_s=time.time() - t0,
            timestamp=ts.isoformat(),
            error=f"Import failed: {e}",
            import_ok=False, run_ok=False,
        )

    # Run
    try:
        if callable(agent_class) and not isinstance(agent_class, type):
            # It's a function (like run())
            result = agent_class(once=True, **kwargs)
        else:
            # It's a class — instantiate and call run()
            instance = agent_class()
            if hasattr(instance, 'run_full_cycle'):
                result = instance.run_full_cycle()
            elif hasattr(instance, 'run'):
                result = instance.run()
            else:
                result = {"status": "no_run_method"}

        duration = time.time() - t0
        metrics = result if isinstance(result, dict) else {}

        return AgentResult(
            agent_name=agent_name,
            status=metrics.get("status", "ok"),
            duration_s=round(duration, 1),
            timestamp=ts.isoformat(),
            metrics=metrics,
            artifacts=manifest.writes_to,
        )

    except Exception as e:
        return AgentResult(
            agent_name=agent_name, status="failed",
            duration_s=round(time.time() - t0, 1),
            timestamp=ts.isoformat(),
            error=f"{type(e).__name__}: {str(e)[:200]}",
            import_ok=True, run_ok=False,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Agent Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def diagnose_agents(run_test: bool = False) -> Dict[str, Dict]:
    """
    Test all registered agents for importability and optionally run them.

    Returns dict of {agent_name: {import_ok, run_ok, error, duration}}.
    """
    results = {}

    for name, manifest in AGENT_REGISTRY.items():
        diag = {"import_ok": False, "run_ok": False, "error": "", "class": manifest.class_name}

        # Test import
        try:
            module = __import__(manifest.module, fromlist=[manifest.class_name])
            agent_class = getattr(module, manifest.class_name)
            diag["import_ok"] = True
        except Exception as e:
            diag["error"] = f"Import: {e}"
            results[name] = diag
            continue

        # Test run (optional — slow)
        if run_test:
            try:
                result = run_agent(name)
                diag["run_ok"] = result.run_ok
                diag["duration"] = result.duration_s
                diag["status"] = result.status
                if result.error:
                    diag["error"] = result.error
            except Exception as e:
                diag["error"] = f"Run: {e}"
        else:
            diag["run_ok"] = None  # Not tested

        results[name] = diag

    return results
