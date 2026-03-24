"""
agents/architect/scan_domains.py
=================================
8 Scan Domains for the Architect Agent.

Each domain defines:
  - target files to inspect
  - key functions to focus on
  - metrics source (bus / registry / filesystem)
  - health checks with thresholds
  - scanning logic
"""
from __future__ import annotations

import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent

# ─────────────────────────────────────────────────────────────────────────────
# Domain definition
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HealthCheck:
    """Single health check with metric name, operator, and threshold."""
    metric: str
    operator: str   # ">=", "<=", ">", "<", "=="
    threshold: float
    description: str = ""

    def evaluate(self, value: float) -> bool:
        ops = {">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b,
               ">": lambda a, b: a > b, "<": lambda a, b: a < b,
               "==": lambda a, b: abs(a - b) < 1e-9}
        op = ops.get(self.operator, lambda a, b: False)
        try:
            return op(float(value), self.threshold)
        except (ValueError, TypeError):
            return False


@dataclass
class ScanDomain:
    """Definition of a single scan domain."""
    name: str
    priority: int
    description: str
    target_files: List[str]         # relative to ROOT
    key_functions: List[str]
    metrics_source: str             # "bus:agent_name" / "registry" / "tests" / "filesystem"
    health_checks: List[HealthCheck] = field(default_factory=list)


@dataclass
class ScanResult:
    """Result of scanning one domain."""
    domain: str
    file_snippets: Dict[str, str]   # {filename: code_snippet}
    metrics: Dict[str, Any]
    health_status: Dict[str, bool]  # {check_description: pass/fail}
    findings: List[str]             # Human-readable findings


# ─────────────────────────────────────────────────────────────────────────────
# 8 Domains ordered by priority
# ─────────────────────────────────────────────────────────────────────────────

SCAN_DOMAINS: List[ScanDomain] = [
    # ── 1. Signal Quality ────────────────────────────────────────
    ScanDomain(
        name="signal_quality",
        priority=1,
        description="Are signals generating alpha? Check IC, hit rate, regime breakdown.",
        target_files=[
            "analytics/signal_stack.py",
            "analytics/signal_mean_reversion.py",
            "analytics/signal_regime_safety.py",
            "analytics/attribution.py",
        ],
        key_functions=[
            "compute_distortion_score",
            "compute_dislocation_score",
            "compute_mean_reversion_score",
            "compute_regime_safety_score",
            "SignalStackEngine.score_sector_candidates",
        ],
        metrics_source="bus:agent_methodology",
        health_checks=[
            HealthCheck("ic_mean", ">=", 0.02, "IC above minimum"),
            HealthCheck("hit_rate", ">=", 0.52, "Hit rate above 52%"),
            HealthCheck("sharpe", ">=", 0.0, "Sharpe non-negative"),
        ],
    ),

    # ── 2. Risk Management ───────────────────────────────────────
    ScanDomain(
        name="risk_management",
        priority=2,
        description="Are risk controls working? VaR, stress tests, exposure limits.",
        target_files=[
            "analytics/portfolio_risk.py",
            "analytics/tail_risk.py",
            "analytics/stress.py",
            "analytics/position_sizing.py",
        ],
        key_functions=[
            "compute_portfolio_risk",
            "kupiec_var_backtest",
            "compute_expected_shortfall",
            "compute_position_size",
        ],
        metrics_source="bus:agent_methodology",
        health_checks=[
            HealthCheck("max_dd", ">=", -0.15, "MaxDD within limits"),
        ],
    ),

    # ── 3. Trade Execution ───────────────────────────────────────
    ScanDomain(
        name="trade_execution",
        priority=3,
        description="Is trade construction correct? Paper portfolio performance.",
        target_files=[
            "analytics/trade_structure.py",
            "analytics/trade_monitor.py",
            "analytics/paper_trader.py",
            "analytics/options_engine.py",
        ],
        key_functions=[
            "build_sector_rv_trade",
            "build_dispersion_trade",
            "TradeMonitor.evaluate_exit",
            "PaperTrader.process_signals",
        ],
        metrics_source="bus:agent_methodology",
        health_checks=[],
    ),

    # ── 4. Data Quality ──────────────────────────────────────────
    ScanDomain(
        name="data_quality",
        priority=4,
        description="Is data fresh, complete, validated?",
        target_files=[
            "data_ops/quality.py",
            "data_ops/freshness.py",
            "data_ops/validators.py",
            "data/pipeline.py",
        ],
        key_functions=[
            "pre_write_validate",
            "check_freshness",
            "validate_prices",
        ],
        metrics_source="filesystem",
        health_checks=[],
    ),

    # ── 5. Dashboard Accuracy ────────────────────────────────────
    ScanDomain(
        name="dashboard_accuracy",
        priority=5,
        description="Does UI reflect reality? Tab rendering, data freshness display.",
        target_files=[
            "ui/analytics_tabs.py",
            "ui/panels.py",
            "ui/scanner_pro.py",
        ],
        key_functions=[
            "build_dss_tab",
            "build_portfolio_tab",
            "build_methodology_tab",
        ],
        metrics_source="filesystem",
        health_checks=[],
    ),

    # ── 6. Agent Health ──────────────────────────────────────────
    ScanDomain(
        name="agent_health",
        priority=6,
        description="Are other agents working? Check registry for STALE/FAILED.",
        target_files=[
            "agents/shared/agent_registry.py",
            "agents/orchestrator.py",
        ],
        key_functions=[
            "AgentRegistry.all_agents",
            "Orchestrator._self_heal",
        ],
        metrics_source="registry",
        health_checks=[],
    ),

    # ── 7. Code Quality ──────────────────────────────────────────
    ScanDomain(
        name="code_quality",
        priority=7,
        description="Tests passing, coverage, documentation gaps.",
        target_files=[],  # Dynamic: all .py files
        key_functions=[],
        metrics_source="tests",
        health_checks=[],
    ),

    # ── 8. Performance ───────────────────────────────────────────
    ScanDomain(
        name="performance",
        priority=8,
        description="Slow computations, memory issues, startup time.",
        target_files=[
            "analytics/backtest.py",
            "analytics/correlation_engine.py",
            "analytics/stat_arb.py",
        ],
        key_functions=[
            "run_backtest",
            "CorrelationStructureEngine.compute_snapshot",
            "QuantEngine.run",
        ],
        metrics_source="filesystem",
        health_checks=[],
    ),
]

DOMAIN_MAP: Dict[str, ScanDomain] = {d.name: d for d in SCAN_DOMAINS}
DOMAIN_NAMES: List[str] = [d.name for d in SCAN_DOMAINS]


# ─────────────────────────────────────────────────────────────────────────────
# Scanning logic
# ─────────────────────────────────────────────────────────────────────────────

def _read_file_snippet(path: Path, max_chars: int = 3000) -> str:
    """Read a file, truncated to max_chars."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        if len(text) > max_chars:
            return text[:max_chars] + f"\n\n... [truncated, total {len(text)} chars]"
        return text
    except Exception as e:
        return f"[ERROR reading {path}: {e}]"


def _read_function_code(path: Path, func_name: str, max_chars: int = 2000) -> str:
    """Extract a function's source from a file."""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return ""

    # Handle Class.method notation
    if "." in func_name:
        func_name = func_name.split(".")[-1]

    collecting = False
    result = []
    base_indent = 0

    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith(f"def {func_name}(") or stripped.startswith(f"def {func_name} ("):
            collecting = True
            base_indent = len(line) - len(stripped)
            result.append(line)
            continue

        if collecting:
            if stripped == "" or stripped.startswith("#"):
                result.append(line)
                continue
            current_indent = len(line) - len(stripped)
            if current_indent <= base_indent and stripped and not stripped.startswith((")", "]", "}")):
                break
            result.append(line)

    code = "\n".join(result)
    if len(code) > max_chars:
        return code[:max_chars] + "\n... [truncated]"
    return code


def _collect_bus_metrics(agent_name: str) -> Dict[str, Any]:
    """Read latest metrics from AgentBus."""
    try:
        sys.path.insert(0, str(ROOT))
        from scripts.agent_bus import get_bus
        bus = get_bus()
        latest = bus.latest(agent_name)
        return latest if isinstance(latest, dict) else {}
    except Exception as e:
        logger.warning("Cannot read bus for %s: %s", agent_name, e)
        return {}


def _collect_registry_metrics() -> Dict[str, Any]:
    """Read all agent statuses from registry."""
    try:
        sys.path.insert(0, str(ROOT))
        from agents.shared.agent_registry import get_registry
        reg = get_registry()
        agents = reg.all_agents()
        failed = [n for n, info in agents.items() if info.get("status") in ("FAILED", "STALE")]
        return {
            "total_agents": len(agents),
            "failed_agents": failed,
            "n_failed": len(failed),
            "agents": agents,
        }
    except Exception as e:
        return {"error": str(e)}


def _collect_test_metrics() -> Dict[str, Any]:
    """Run pytest and collect results."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(ROOT / "tests"), "-q", "--tb=no"],
            capture_output=True, text=True, timeout=120, cwd=str(ROOT),
        )
        output = result.stdout + result.stderr
        # Parse "126 passed in 1.42s"
        passed = 0
        failed = 0
        for line in output.splitlines():
            if "passed" in line:
                parts = line.split()
                for i, p in enumerate(parts):
                    if p == "passed" and i > 0:
                        try:
                            passed = int(parts[i - 1])
                        except ValueError:
                            pass
                    if p == "failed" and i > 0:
                        try:
                            failed = int(parts[i - 1])
                        except ValueError:
                            pass
        return {"passed": passed, "failed": failed, "all_pass": failed == 0, "output": output[-500:]}
    except Exception as e:
        return {"error": str(e), "all_pass": False}


def _collect_data_freshness() -> Dict[str, Any]:
    """Check data freshness from parquet files."""
    import pandas as pd
    try:
        prices_path = ROOT / "data_lake" / "parquet" / "prices.parquet"
        if prices_path.exists():
            prices = pd.read_parquet(prices_path)
            last_date = str(prices.index[-1].date())
            n_rows = len(prices)
            n_cols = len(prices.columns)
            nan_pct = float(prices.isna().mean().mean())
            return {
                "last_date": last_date,
                "n_rows": n_rows,
                "n_tickers": n_cols,
                "nan_pct": round(nan_pct, 4),
            }
        return {"error": "prices.parquet not found"}
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Main scan function
# ─────────────────────────────────────────────────────────────────────────────

def scan_domain(domain: ScanDomain) -> ScanResult:
    """
    Scan a domain: read files, collect metrics, run health checks.

    Returns ScanResult with everything the GPT analysis needs.
    """
    logger.info("Scanning domain: %s", domain.name)

    # 1. Read file snippets
    file_snippets: Dict[str, str] = {}
    for rel_path in domain.target_files:
        full_path = ROOT / rel_path
        if full_path.exists():
            # If we have key functions, extract them; otherwise snippet the file
            funcs_for_file = [f for f in domain.key_functions
                              if not f.startswith("_") or "." in f]
            func_code = ""
            for func in domain.key_functions:
                code = _read_function_code(full_path, func)
                if code:
                    func_code += f"\n# --- {func} ---\n{code}\n"
            if func_code:
                file_snippets[rel_path] = func_code[:3000]
            else:
                file_snippets[rel_path] = _read_file_snippet(full_path, max_chars=3000)

    # 2. Collect metrics
    metrics: Dict[str, Any] = {}
    if domain.metrics_source.startswith("bus:"):
        agent_name = domain.metrics_source.split(":", 1)[1]
        metrics = _collect_bus_metrics(agent_name)
    elif domain.metrics_source == "registry":
        metrics = _collect_registry_metrics()
    elif domain.metrics_source == "tests":
        metrics = _collect_test_metrics()
    elif domain.metrics_source == "filesystem":
        metrics = _collect_data_freshness()

    # 3. Run health checks
    health_status: Dict[str, bool] = {}
    findings: List[str] = []

    for check in domain.health_checks:
        value = metrics.get(check.metric)
        if value is not None:
            passed = check.evaluate(value)
            health_status[check.description] = passed
            if not passed:
                findings.append(
                    f"FAILED: {check.description} — "
                    f"{check.metric}={value} (need {check.operator} {check.threshold})"
                )
        else:
            health_status[check.description] = False
            findings.append(f"MISSING: {check.description} — metric '{check.metric}' not available")

    # 4. Domain-specific extra checks with concrete quantitative thresholds
    if domain.name == "signal_quality":
        # Concrete signal quality checks
        ic = metrics.get("ic_mean", metrics.get("backtest_ic"))
        if ic is not None:
            ic_f = float(ic)
            if ic_f < 0.01:
                findings.append(f"CRITICAL: IC={ic_f:.4f} barely predictive (need >0.02)")
            elif ic_f < 0.02:
                findings.append(f"WARNING: IC={ic_f:.4f} below target (0.02)")
        sharpe = metrics.get("sharpe", metrics.get("backtest_sharpe"))
        if sharpe is not None:
            s_f = float(sharpe)
            if s_f < 0.3:
                findings.append(f"CRITICAL: Sharpe={s_f:.3f} below minimum viable (0.3)")
        # Regime-specific checks
        regime_bd = metrics.get("regime_breakdown", {})
        for reg, rd in regime_bd.items():
            if isinstance(rd, dict):
                reg_sharpe = rd.get("sharpe", 0)
                if isinstance(reg_sharpe, (int, float)) and float(reg_sharpe) < 0:
                    findings.append(f"NEGATIVE SHARPE in {reg} regime: {float(reg_sharpe):.3f}")

    if domain.name == "risk_management":
        max_dd = metrics.get("max_dd", metrics.get("max_drawdown"))
        if max_dd is not None and abs(float(max_dd)) > 0.05:
            findings.append(f"DRAWDOWN ALERT: MaxDD={float(max_dd):.1%} exceeds 5% limit")
        es_var = metrics.get("es_var_ratio")
        if es_var is not None and float(es_var) > 1.8:
            findings.append(f"FAT TAILS: ES/VaR={float(es_var):.2f} (>1.8 = heavy tails)")

    if domain.name == "agent_health":
        failed = metrics.get("failed_agents", [])
        if failed:
            findings.append(f"FAILED AGENTS: {', '.join(failed)}")

    if domain.name == "code_quality":
        test_metrics = metrics
        if not test_metrics.get("all_pass", True):
            findings.append(f"TESTS FAILING: {test_metrics.get('failed', '?')} failures")

    if domain.name == "data_quality":
        if metrics.get("nan_pct", 0) > 0.05:
            findings.append(f"HIGH NaN: {metrics['nan_pct']:.1%} of data is NaN")
        staleness = metrics.get("data_staleness_days")
        if staleness is not None and int(staleness) > 3:
            findings.append(f"DATA STALE: {staleness} days since last update")

    if domain.name == "trade_execution":
        disp_wr = metrics.get("disp_win_rate")
        if disp_wr is not None and float(disp_wr) < 0.5:
            findings.append(f"DISPERSION WR below 50%: {float(disp_wr):.1%}")
        disp_sharpe = metrics.get("disp_sharpe")
        if disp_sharpe is not None and float(disp_sharpe) < 0.5:
            findings.append(f"DISPERSION Sharpe low: {float(disp_sharpe):.2f}")

    if not findings:
        findings.append(f"All health checks passed for {domain.name}")

    logger.info("Scan complete: %s — %d findings", domain.name, len(findings))
    return ScanResult(
        domain=domain.name,
        file_snippets=file_snippets,
        metrics=metrics,
        health_status=health_status,
        findings=findings,
    )
