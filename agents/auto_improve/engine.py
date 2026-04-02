"""
agents/auto_improve.py
========================
Master Auto-Improvement Script for SRV Quant System

Orchestrates a feedback loop between the 3 agents (Methodology, Optimizer, Math):
  1. Runs methodology evaluation via analytics/methodology_lab.py
  2. Compares results vs last saved results
  3. Identifies weakest regime/strategy
  4. Generates optimization suggestions (rule-based + optional GPT)
  5. Tests suggested parameter changes in sandbox (backtest with new params)
  6. Promotes validated improvements to settings.py via env overrides
  7. Logs everything to agents/auto_improve/improvement_log.json

Usage:
  python agents/auto_improve.py --cycle      # Run one improvement cycle
  python agents/auto_improve.py --status     # Show improvement history
  python agents/auto_improve.py --dry-run    # Evaluate only, don't promote
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Ensure project root on sys.path ─────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent  # srv_quant_system/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

from logging.handlers import RotatingFileHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        RotatingFileHandler(
            LOG_DIR / "auto_improve.log",
            maxBytes=20 * 1024 * 1024, backupCount=3, encoding="utf-8",
        ),
    ],
)
log = logging.getLogger("auto_improve")

# ── Constants ────────────────────────────────────────────────────────────────
IMPROVEMENT_LOG_PATH = ROOT / "agents" / "auto_improve" / "improvement_log.json"
REPORTS_DIR = ROOT / "agents" / "methodology" / "reports"
SHARPE_PROMOTION_THRESHOLD = 0.03
MAX_SUGGESTIONS_PER_CYCLE = 5
MAX_GPT_CALLS_PER_CYCLE = 3

# ── Tunable parameter definitions with safe ranges ──────────────────────────
# Each entry: (settings_field, min, max, step_size, description)
TUNABLE_PARAMS: List[Tuple[str, float, float, float, str]] = [
    ("regime_z_calm",              0.3,  2.0,  0.1,  "Z-score entry threshold in CALM regime"),
    ("regime_z_normal",            0.3,  2.0,  0.1,  "Z-score entry threshold in NORMAL regime"),
    ("regime_z_tension",           0.3,  2.5,  0.1,  "Z-score entry threshold in TENSION regime"),
    ("regime_size_calm",           0.5,  2.0,  0.1,  "Position size multiplier in CALM"),
    ("regime_size_normal",         0.5,  1.5,  0.1,  "Position size multiplier in NORMAL"),
    ("regime_size_tension",        0.1,  1.0,  0.1,  "Position size multiplier in TENSION"),
    ("regime_conviction_scale_calm",   0.5, 2.0, 0.1, "Conviction scale for CALM"),
    ("regime_conviction_scale_normal", 0.5, 1.5, 0.1, "Conviction scale for NORMAL"),
    ("regime_conviction_scale_tension", 0.1, 1.0, 0.1, "Conviction scale for TENSION"),
    ("signal_vix_soft",            15.0, 28.0, 1.0,  "VIX soft threshold"),
    ("signal_vix_hard",            25.0, 40.0, 1.0,  "VIX hard kill threshold"),
    ("signal_entry_threshold",     0.01, 0.20, 0.01, "Signal entry threshold"),
    ("trade_max_holding_days",     10,   40,   5,    "Max holding period in days"),
    ("signal_optimal_hold",        10,   40,   5,    "Optimal hold period"),
    ("monitor_z_compression_exit", 0.4,  0.9,  0.05, "Z compression exit threshold"),
    ("monitor_z_extension_stop",   1.5,  3.0,  0.1,  "Z extension stop loss"),
    ("signal_a1_frob",             0.1,  1.0,  0.1,  "Distortion score Frobenius weight"),
    ("signal_a2_mode",             0.1,  1.0,  0.1,  "Distortion score mode weight"),
    ("signal_a3_coc",              0.1,  1.0,  0.1,  "Distortion score CoC weight"),
    ("non_whitelist_penalty",      0.0,  0.5,  0.1,  "Non-whitelist sector penalty"),
]


def _load_improvement_log() -> Dict:
    """Load the improvement log from disk."""
    IMPROVEMENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if IMPROVEMENT_LOG_PATH.exists():
        try:
            return json.loads(IMPROVEMENT_LOG_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"cycles": []}


def _save_improvement_log(data: Dict) -> None:
    """Save the improvement log to disk."""
    IMPROVEMENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    IMPROVEMENT_LOG_PATH.write_text(
        json.dumps(data, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )


def _load_last_methodology_results() -> Optional[Dict]:
    """Load the most recent methodology lab results from reports dir."""
    if not REPORTS_DIR.exists():
        return None
    files = sorted(REPORTS_DIR.glob("*_methodology_lab.json"), reverse=True)
    if not files:
        return None
    try:
        return json.loads(files[0].read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# AutoImprover
# ─────────────────────────────────────────────────────────────────────────────

class AutoImprover:
    """
    Master auto-improvement engine.

    Runs methodology evaluation, identifies weaknesses, generates parameter
    suggestions (rule-based + optional GPT), tests them in sandbox, and
    promotes validated improvements.
    """

    def __init__(self, settings=None, prices=None, dry_run: bool = False):
        self.dry_run = dry_run
        self._settings = settings
        self._prices = prices
        self.log_data: List[Dict] = []
        self._gpt = None
        self._gpt_calls_used = 0

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def settings(self):
        if self._settings is None:
            from config.settings import get_settings
            self._settings = get_settings()
        return self._settings

    @property
    def prices(self):
        if self._prices is None:
            self._prices = self._load_prices()
        return self._prices

    def _load_prices(self):
        """Load price data from parquet or data lake."""
        parquet = ROOT / "data_lake" / "parquet" / "prices.parquet"
        if parquet.exists():
            import pandas as pd
            return pd.read_parquet(parquet)
        # Fallback: try alternative paths
        for alt in [
            ROOT / "data_lake" / "parquet" / "sector_etf_prices.parquet",
            ROOT / "data" / "prices.parquet",
        ]:
            if alt.exists():
                import pandas as pd
                return pd.read_parquet(alt)
        log.warning("No price data found — cannot run backtest")
        return None

    def current_params(self) -> Dict[str, Any]:
        """Extract current tunable parameters from settings."""
        params = {}
        for field_name, _mn, _mx, _step, _desc in TUNABLE_PARAMS:
            if hasattr(self.settings, field_name):
                params[field_name] = getattr(self.settings, field_name)
        return params

    # ── GPT Bridge ───────────────────────────────────────────────────────

    def _get_gpt(self):
        """Lazy-load GPT conversation bridge. Graceful fallback if unavailable."""
        if self._gpt is not None:
            return self._gpt
        try:
            from agents.shared.gpt_conversation import GPTConversation
            conv = GPTConversation(system_role="quantitative strategy optimizer")
            if conv.available:
                self._gpt = conv
                log.info("GPT bridge loaded successfully")
            else:
                log.info("GPT bridge loaded but no API key — will use rule-based suggestions only")
        except Exception as e:
            log.info("GPT bridge unavailable: %s — using rule-based suggestions only", e)
        return self._gpt

    # ── Institutional Methods ────────────────────────────────────────────

    def check_governance_prerequisites(self) -> Dict:
        """
        Load latest methodology governance report and check strategy approvals.

        Returns dict with:
          loaded: bool — whether a governance report was found
          status: "all_approved" | "some_conditional" | "all_rejected" | "no_report"
          mode: "full" | "cautious" | "skip"
          approved_strategies: list[str]
          conditional_strategies: list[str]
          rejected_strategies: list[str]
          reason: str — human-readable explanation
        """
        result: Dict[str, Any] = {
            "loaded": False,
            "status": "no_report",
            "mode": "full",
            "approved_strategies": [],
            "conditional_strategies": [],
            "rejected_strategies": [],
            "reason": "",
        }

        if not REPORTS_DIR.exists():
            result["reason"] = "Methodology reports directory not found"
            log.warning("Governance: %s", result["reason"])
            return result

        # Find latest report with approval data
        report_data: Optional[Dict] = None
        for fpath in sorted(REPORTS_DIR.glob("*.json"), reverse=True):
            try:
                data = json.loads(fpath.read_text(encoding="utf-8"))
                if isinstance(data, dict) and (
                    "approval_matrix" in data or "approval_matrix_v3" in data
                ):
                    report_data = data
                    log.info("Governance prerequisites: loaded %s", fpath.name)
                    break
            except (json.JSONDecodeError, OSError):
                continue

        if report_data is None:
            result["reason"] = "No methodology report with approval matrix found"
            log.warning("Governance: %s", result["reason"])
            return result

        result["loaded"] = True
        approval_matrix = (
            report_data.get("approval_matrix_v3")
            or report_data.get("approval_matrix", {})
        )

        for strat_name, decision in approval_matrix.items():
            dec = decision if isinstance(decision, str) else decision.get("decision", "REJECTED")
            if dec == "APPROVED":
                result["approved_strategies"].append(strat_name)
            elif dec == "CONDITIONAL":
                result["conditional_strategies"].append(strat_name)
            else:
                result["rejected_strategies"].append(strat_name)

        n_approved = len(result["approved_strategies"])
        n_conditional = len(result["conditional_strategies"])
        n_rejected = len(result["rejected_strategies"])
        total = n_approved + n_conditional + n_rejected

        if total == 0:
            result["status"] = "no_report"
            result["mode"] = "full"
            result["reason"] = "Empty approval matrix"
        elif n_approved == 0 and n_conditional == 0:
            result["status"] = "all_rejected"
            result["mode"] = "skip"
            result["reason"] = (
                f"All {n_rejected} strategies REJECTED — skipping optimization"
            )
        elif n_conditional > 0 and n_approved == 0:
            result["status"] = "some_conditional"
            result["mode"] = "cautious"
            result["reason"] = (
                f"{n_conditional} CONDITIONAL, {n_rejected} REJECTED — cautious mode "
                "(fewer trials, smaller bounds)"
            )
        else:
            result["status"] = "all_approved"
            result["mode"] = "full"
            result["reason"] = (
                f"{n_approved} APPROVED, {n_conditional} CONDITIONAL, {n_rejected} REJECTED"
            )

        log.info("Governance prerequisites: %s — mode=%s", result["reason"], result["mode"])
        return result

    def generate_smart_suggestions(self, report: Optional[Dict] = None) -> List[Dict]:
        """
        Generate targeted suggestions from methodology and alpha_decay machine summaries.

        Reads:
        - methodology machine_summary for optimizer_instructions
        - alpha_decay machine_summary for action_policies
        - Current weaknesses from evaluation

        Each suggestion includes: param, current_value, new_value, rationale, expected_impact.
        """
        suggestions: List[Dict] = []
        params = self.current_params()

        # ── Load methodology machine_summary ───────────────────────────
        meth_ms: Dict = {}
        if report and isinstance(report, dict):
            meth_ms = report.get("machine_summary_v3") or report.get("machine_summary", {})
        else:
            for fpath in sorted(REPORTS_DIR.glob("*.json"), reverse=True):
                try:
                    data = json.loads(fpath.read_text(encoding="utf-8"))
                    if isinstance(data, dict) and (
                        "machine_summary_v3" in data or "machine_summary" in data
                    ):
                        meth_ms = data.get("machine_summary_v3") or data.get("machine_summary", {})
                        break
                except (json.JSONDecodeError, OSError):
                    continue

        # ── Load alpha_decay machine_summary ───────────────────────────
        alpha_ms: Dict = {}
        alpha_dir = ROOT / "agents" / "alpha_decay" / "reports"
        if alpha_dir.exists():
            for fpath in sorted(alpha_dir.glob("*.json"), reverse=True):
                try:
                    data = json.loads(fpath.read_text(encoding="utf-8"))
                    if isinstance(data, dict) and "machine_summary" in data:
                        alpha_ms = data.get("machine_summary", {})
                        break
                except (json.JSONDecodeError, OSError):
                    continue

        # ── Extract optimizer instructions ─────────────────────────────
        optimizer_instructions = meth_ms.get("optimizer_instructions", [])
        if isinstance(optimizer_instructions, list):
            for instr in optimizer_instructions[:3]:
                if isinstance(instr, dict):
                    param = instr.get("param", "")
                    new_val = instr.get("value") or instr.get("new_value")
                    if param in params and new_val is not None:
                        try:
                            new_val = float(new_val)
                            suggestions.append({
                                "param": param,
                                "current_value": params[param],
                                "new_value": new_val,
                                "rationale": instr.get("reason", "methodology optimizer instruction"),
                                "expected_impact": instr.get("expected_impact", "unknown"),
                                "source": "methodology_machine_summary",
                            })
                        except (ValueError, TypeError):
                            pass

        # ── Extract alpha decay action policies ────────────────────────
        action_policies = alpha_ms.get("action_policies", [])
        if isinstance(action_policies, list):
            for policy in action_policies[:2]:
                if isinstance(policy, dict):
                    param = policy.get("param", "")
                    new_val = policy.get("value") or policy.get("new_value")
                    if param in params and new_val is not None:
                        try:
                            new_val = float(new_val)
                            suggestions.append({
                                "param": param,
                                "current_value": params[param],
                                "new_value": new_val,
                                "rationale": policy.get("reason", "alpha decay action policy"),
                                "expected_impact": policy.get("expected_impact", "unknown"),
                                "source": "alpha_decay_machine_summary",
                            })
                        except (ValueError, TypeError):
                            pass

        # ── Weakness-targeted suggestions ──────────────────────────────
        regime_breakdown = meth_ms.get("regime_performance", {})
        if isinstance(regime_breakdown, dict):
            for regime_name, stats in regime_breakdown.items():
                if isinstance(stats, dict) and stats.get("sharpe", 0) < -0.3:
                    size_key = f"regime_size_{regime_name.lower()}"
                    if size_key in params:
                        current = params[size_key]
                        new_val = max(0.1, round(current * 0.7, 2))
                        if new_val != current:
                            suggestions.append({
                                "param": size_key,
                                "current_value": current,
                                "new_value": new_val,
                                "rationale": f"Reduce {regime_name} sizing: Sharpe={stats.get('sharpe', 0):.3f}",
                                "expected_impact": f"reduce {regime_name} losses",
                                "source": "smart_weakness_analysis",
                            })

        # Deduplicate
        seen = set()
        unique = []
        for s in suggestions:
            if s["param"] not in seen:
                seen.add(s["param"])
                unique.append(s)

        log.info("Smart suggestions: %d generated from governance + alpha_decay", len(unique))
        return unique[:MAX_SUGGESTIONS_PER_CYCLE]

    def validate_promotion_safety(
        self,
        param: str,
        old_val: Any,
        new_val: Any,
        delta_sharpe: float,
        test_result: Optional[Dict] = None,
    ) -> bool:
        """
        Multi-check promotion safety gate.

        All of the following must pass:
        1. delta_sharpe >= 0.03
        2. No regime degradation (per-regime Sharpe not worse)
        3. Drawdown didn't worsen by > 2%
        4. Win rate didn't drop by > 3%

        Returns True only if ALL checks pass.
        """
        checks_passed = True
        reasons: List[str] = []

        # Check 1: Minimum Sharpe improvement
        if delta_sharpe < 0.03:
            checks_passed = False
            reasons.append(f"delta_sharpe={delta_sharpe:.4f} < 0.03 minimum")

        if test_result:
            # Check 2: Per-regime degradation
            regime_before = test_result.get("regime_sharpes_before", {})
            regime_after = test_result.get("regime_sharpes_after", {})
            for r_name in regime_before:
                before = regime_before.get(r_name, 0)
                after = regime_after.get(r_name, before)
                if after < before - 0.1:
                    checks_passed = False
                    reasons.append(
                        f"regime {r_name} degraded: {before:.3f} -> {after:.3f}"
                    )

            # Check 3: Drawdown check
            dd_before = abs(test_result.get("max_dd_before", 0))
            dd_after = abs(test_result.get("max_dd_after", 0))
            if dd_after > dd_before + 0.02:
                checks_passed = False
                reasons.append(
                    f"drawdown worsened: {dd_before:.3f} -> {dd_after:.3f} (>{0.02} limit)"
                )

            # Check 4: Win rate check
            wr_before = test_result.get("wr_before", 0)
            wr_after = test_result.get("wr_after", 0)
            if wr_after < wr_before - 0.03:
                checks_passed = False
                reasons.append(
                    f"win_rate dropped: {wr_before:.3f} -> {wr_after:.3f} (>{0.03} limit)"
                )

        if checks_passed:
            log.info("Promotion safety PASSED for %s: %s -> %s (delta_sharpe=%.4f)",
                     param, old_val, new_val, delta_sharpe)
        else:
            log.warning("Promotion safety FAILED for %s: %s", param, "; ".join(reasons))

        return checks_passed

    # ── Core Methods ─────────────────────────────────────────────────────

    def run_evaluation(self) -> Dict:
        """
        Run methodology lab and return current metrics.

        Returns dict with keys: best_name, best_sharpe, best_win_rate,
        best_max_dd, best_trades, regime_breakdown, all_results.
        """
        if self.prices is None:
            log.error("Cannot run evaluation — no price data")
            return {}

        try:
            from analytics.methodology_lab import MethodologyLab
            lab = MethodologyLab(self.prices, self.settings)
            results = lab.run_all()
            lab.save_results()

            # Find best methodology by Sharpe
            if not results:
                return {}

            best_name = max(results, key=lambda k: results[k].sharpe)
            best = results[best_name]

            metrics = {
                "best_name": best_name,
                "best_sharpe": best.sharpe,
                "best_win_rate": best.win_rate,
                "best_max_dd": best.max_drawdown,
                "best_pnl": best.total_pnl,
                "best_trades": best.total_trades,
                "best_avg_hold": best.avg_holding_days,
                "regime_breakdown": best.regime_stats,
                "all_results": {
                    name: {
                        "sharpe": r.sharpe,
                        "win_rate": r.win_rate,
                        "total_pnl": r.total_pnl,
                        "max_drawdown": r.max_drawdown,
                        "total_trades": r.total_trades,
                        "params": r.params,
                    }
                    for name, r in results.items()
                },
            }
            log.info(
                "Evaluation complete: best=%s Sharpe=%.3f WR=%.1f%%",
                best_name, best.sharpe, best.win_rate * 100,
            )
            return metrics
        except Exception as e:
            log.error("Evaluation failed: %s\n%s", e, traceback.format_exc())
            return {}

    def identify_weaknesses(self, metrics: Dict) -> List[Dict]:
        """
        Find worst-performing areas from evaluation metrics.

        Returns list of weakness dicts with keys: type, description, severity, context.
        """
        weaknesses = []
        if not metrics:
            return weaknesses

        # 1. Check overall Sharpe
        best_sharpe = metrics.get("best_sharpe", 0)
        if best_sharpe < 0:
            weaknesses.append({
                "type": "negative_sharpe",
                "description": f"Best strategy {metrics.get('best_name', '?')} has negative Sharpe: {best_sharpe:.3f}",
                "severity": "high",
                "context": {"sharpe": best_sharpe, "strategy": metrics.get("best_name")},
            })

        # 2. Check regime breakdown for weak regimes
        regime_stats = metrics.get("regime_breakdown", {})
        for regime_name, stats in regime_stats.items():
            regime_sharpe = stats.get("sharpe", 0)
            regime_trades = stats.get("trades", 0)
            if regime_trades > 5 and regime_sharpe < -0.3:
                weaknesses.append({
                    "type": "weak_regime",
                    "description": f"{regime_name} regime: Sharpe={regime_sharpe:.3f} across {regime_trades} trades",
                    "severity": "high" if regime_sharpe < -0.5 else "medium",
                    "context": {
                        "regime": regime_name,
                        "sharpe": regime_sharpe,
                        "trades": regime_trades,
                        "win_rate": stats.get("win_rate", 0),
                    },
                })

        # 3. Check for poor win rates (skip for momentum strategies — they naturally have WR < 50%)
        best_wr = metrics.get("best_win_rate", 0)
        best_name = metrics.get("best_name", "")
        is_momentum = "MOMENTUM" in best_name.upper()
        if best_wr < 0.50 and not is_momentum:
            weaknesses.append({
                "type": "low_win_rate",
                "description": f"Win rate below 50%: {best_wr:.1%}",
                "severity": "medium",
                "context": {"win_rate": best_wr},
            })

        # 3b. Momentum optimization opportunity
        if is_momentum and best_sharpe > 0:
            weaknesses.append({
                "type": "momentum_tuning",
                "description": f"Momentum strategy Sharpe={best_sharpe:.3f} — can optimize lookback, rebal, top_n",
                "severity": "low",
                "context": {"sharpe": best_sharpe, "strategy": best_name, "win_rate": best_wr},
            })

        # 4. Check for excessive drawdown
        best_dd = metrics.get("best_max_dd", 0)
        if best_dd < -0.15:
            weaknesses.append({
                "type": "high_drawdown",
                "description": f"Max drawdown too high: {best_dd:.1%}",
                "severity": "high",
                "context": {"max_drawdown": best_dd},
            })

        # 5. Check all strategies — identify if all are negative
        all_results = metrics.get("all_results", {})
        positive_count = sum(1 for r in all_results.values() if r.get("sharpe", 0) > 0)
        if all_results and positive_count == 0:
            weaknesses.append({
                "type": "all_negative",
                "description": f"All {len(all_results)} strategies have negative Sharpe",
                "severity": "critical",
                "context": {"strategies": len(all_results), "positive": 0},
            })

        # 6. Strategy diversity check — are we using the best available?
        all_results = metrics.get("all_results", {})
        if all_results:
            positive_strats = {k: v for k, v in all_results.items() if v.get("sharpe", 0) > 0}
            if len(positive_strats) >= 2:
                best_2 = sorted(positive_strats.items(), key=lambda x: x[1]["sharpe"], reverse=True)[:3]
                weaknesses.append({
                    "type": "strategy_diversification",
                    "description": f"{len(positive_strats)} positive strategies available — consider ensemble allocation",
                    "severity": "low",
                    "context": {
                        "strategies": {k: round(v["sharpe"], 3) for k, v in best_2},
                        "n_positive": len(positive_strats),
                    },
                })

            # Check if any negative strategies should be disabled
            worst_strats = [k for k, v in all_results.items() if v.get("sharpe", 0) < -0.5 and v.get("total_trades", 0) > 50]
            if len(worst_strats) > 5:
                weaknesses.append({
                    "type": "dead_strategies",
                    "description": f"{len(worst_strats)} strategies with Sharpe < -0.5 — consider disabling",
                    "severity": "low",
                    "context": {"dead_count": len(worst_strats), "strategies": worst_strats[:5]},
                })

        # Sort by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        weaknesses.sort(key=lambda w: severity_order.get(w["severity"], 99))

        log.info("Identified %d weaknesses", len(weaknesses))
        for w in weaknesses:
            log.info("  [%s] %s", w["severity"].upper(), w["description"])

        return weaknesses

    def generate_parameter_suggestions(self, weaknesses: List[Dict]) -> List[Dict]:
        """
        Generate parameter change suggestions based on identified weaknesses.

        Each suggestion is a dict with:
          param: settings field name
          current: current value
          proposed: new value
          reason: why this change
          source: "rule" or "gpt"
        """
        suggestions = []
        params = self.current_params()

        for weakness in weaknesses:
            wtype = weakness["type"]
            ctx = weakness.get("context", {})

            if wtype == "weak_regime":
                regime = ctx.get("regime", "").upper()
                suggestions.extend(self._suggest_for_weak_regime(regime, ctx, params))

            elif wtype == "negative_sharpe":
                suggestions.extend(self._suggest_for_negative_sharpe(ctx, params))

            elif wtype == "low_win_rate":
                suggestions.extend(self._suggest_for_low_win_rate(params))

            elif wtype == "high_drawdown":
                suggestions.extend(self._suggest_for_high_drawdown(params))

            elif wtype == "all_negative":
                suggestions.extend(self._suggest_for_all_negative(params))

            elif wtype == "momentum_tuning":
                suggestions.extend(self._suggest_for_momentum(ctx, params))

        # Deduplicate by param name (keep first)
        seen = set()
        unique = []
        for s in suggestions:
            if s["param"] not in seen:
                seen.add(s["param"])
                unique.append(s)
        suggestions = unique

        log.info("Generated %d unique parameter suggestions", len(suggestions))
        return suggestions[:MAX_SUGGESTIONS_PER_CYCLE]

    def _suggest_for_weak_regime(self, regime: str, ctx: Dict, params: Dict) -> List[Dict]:
        """Generate suggestions to fix a weak regime."""
        suggestions = []
        regime_lower = regime.lower()

        # Reduce sizing in weak regime
        size_key = f"regime_size_{regime_lower}"
        if size_key in params:
            current = params[size_key]
            proposed = max(0.0, round(current * 0.6, 2))
            if proposed != current:
                suggestions.append({
                    "param": size_key,
                    "current": current,
                    "proposed": proposed,
                    "reason": f"Reduce {regime} sizing due to Sharpe={ctx.get('sharpe', 0):.3f}",
                    "source": "rule",
                })

        # Increase z-threshold in weak regime (require stronger signals)
        z_key = f"regime_z_{regime_lower}"
        if z_key in params:
            current = params[z_key]
            proposed = min(2.5, round(current + 0.2, 2))
            if proposed != current:
                suggestions.append({
                    "param": z_key,
                    "current": current,
                    "proposed": proposed,
                    "reason": f"Tighten {regime} entry z-threshold due to poor performance",
                    "source": "rule",
                })

        # Reduce conviction scale
        conv_key = f"regime_conviction_scale_{regime_lower}"
        if conv_key in params:
            current = params[conv_key]
            proposed = max(0.0, round(current * 0.7, 2))
            if proposed != current:
                suggestions.append({
                    "param": conv_key,
                    "current": current,
                    "proposed": proposed,
                    "reason": f"Lower conviction in {regime} to reduce losses",
                    "source": "rule",
                })

        # If regime is already fully disabled (no tunable params or all at floor),
        # suggest tightening the VIX hard threshold to prevent leakage
        if not suggestions:
            vix_hard = params.get("signal_vix_hard")
            if vix_hard is not None:
                proposed = max(25.0, round(vix_hard - 2.0, 1))
                if proposed != vix_hard:
                    suggestions.append({
                        "param": "signal_vix_hard",
                        "current": vix_hard,
                        "proposed": proposed,
                        "reason": f"Tighten VIX hard threshold — {regime} regime already disabled but still losing",
                        "source": "rule",
                    })

        return suggestions

    def _suggest_for_negative_sharpe(self, ctx: Dict, params: Dict) -> List[Dict]:
        """Suggestions when overall Sharpe is negative."""
        suggestions = []
        # Tighten entry threshold
        current = params.get("signal_entry_threshold", 0.05)
        proposed = min(0.15, round(current + 0.02, 3))
        if proposed != current:
            suggestions.append({
                "param": "signal_entry_threshold",
                "current": current,
                "proposed": proposed,
                "reason": "Raise entry threshold to filter out low-quality signals",
                "source": "rule",
            })
        return suggestions

    def _suggest_for_low_win_rate(self, params: Dict) -> List[Dict]:
        """Suggestions when win rate is below 50%."""
        suggestions = []
        # Widen z compression exit (hold winners longer)
        current = params.get("monitor_z_compression_exit", 0.75)
        proposed = min(0.90, round(current + 0.05, 3))
        if proposed != current:
            suggestions.append({
                "param": "monitor_z_compression_exit",
                "current": current,
                "proposed": proposed,
                "reason": "Widen z compression exit to hold winners longer",
                "source": "rule",
            })
        return suggestions

    def _suggest_for_high_drawdown(self, params: Dict) -> List[Dict]:
        """Suggestions when max drawdown is too high."""
        suggestions = []
        # Tighten stop loss
        current = params.get("monitor_z_extension_stop", 2.5)
        proposed = max(1.5, round(current - 0.2, 2))
        if proposed != current:
            suggestions.append({
                "param": "monitor_z_extension_stop",
                "current": current,
                "proposed": proposed,
                "reason": "Tighter stop loss to reduce max drawdown",
                "source": "rule",
            })
        # Reduce tension sizing
        current_t = params.get("regime_size_tension", 0.6)
        proposed_t = max(0.1, round(current_t - 0.1, 2))
        if proposed_t != current_t:
            suggestions.append({
                "param": "regime_size_tension",
                "current": current_t,
                "proposed": proposed_t,
                "reason": "Reduce TENSION sizing to limit drawdown",
                "source": "rule",
            })
        return suggestions

    def _suggest_for_all_negative(self, params: Dict) -> List[Dict]:
        """
        Smart suggestions based on alpha research findings:
        - LOWER z-entry = more trades, better WR (proven: z=0.5 → WR 57%)
        - LONGER hold = more time for mean reversion (proven: hold=30 → better)
        - HIGHER regime sizing in CALM = capitalize on best regime (Sharpe 0.66)
        - LOWER tension sizing = avoid TENSION losses
        """
        suggestions = []

        # Lower z-entry in CALM (proven: lower threshold catches more MR opportunities)
        current_z = params.get("regime_z_calm", 0.7)
        if current_z > 0.5:
            suggestions.append({
                "param": "regime_z_calm",
                "current": current_z,
                "proposed": round(max(0.4, current_z - 0.1), 2),
                "reason": "Lower CALM z-entry: more MR trades, proven WR improvement at z=0.5",
                "source": "rule_alpha_research",
            })

        # Increase hold period (MR needs time to revert)
        current_hold = params.get("trade_max_holding_days", 20)
        if current_hold < 30:
            suggestions.append({
                "param": "trade_max_holding_days",
                "current": current_hold,
                "proposed": min(35, current_hold + 5),
                "reason": "Extend hold: MR needs time, proven optimal at 25-30 days",
                "source": "rule_alpha_research",
            })

        # Boost CALM regime sizing (Sharpe 0.66 in CALM)
        current_calm = params.get("regime_size_calm", 1.3)
        if current_calm < 1.5:
            suggestions.append({
                "param": "regime_size_calm",
                "current": current_calm,
                "proposed": min(2.0, round(current_calm + 0.2, 1)),
                "reason": "Increase CALM sizing: best regime for MR (Sharpe=0.66)",
                "source": "rule_alpha_research",
            })

        # Reduce TENSION sizing (Sharpe only 0.23 in NORMAL, MR weaker)
        current_tension = params.get("regime_size_tension", 0.6)
        if current_tension > 0.4:
            suggestions.append({
                "param": "regime_size_tension",
                "current": current_tension,
                "proposed": round(max(0.2, current_tension - 0.1), 1),
                "reason": "Reduce TENSION sizing: MR less reliable under stress",
                "source": "rule_alpha_research",
            })

        return suggestions[:MAX_SUGGESTIONS_PER_CYCLE]

    def _suggest_for_momentum(self, ctx: Dict, params: Dict) -> List[Dict]:
        """
        Momentum-specific tuning suggestions.
        Tests variations of lookback, top_n, rebal_days, max_weight.
        """
        suggestions = []
        import random
        rng = random.Random(42 + hash(str(ctx.get("sharpe", 0))))

        # Momentum lookback: try different windows
        current_lb = params.get("momentum_lookback", 21)
        candidates_lb = [10, 15, 21, 42, 63]
        alt_lb = rng.choice([c for c in candidates_lb if c != current_lb])
        suggestions.append({
            "param": "momentum_lookback",
            "current": current_lb,
            "proposed": alt_lb,
            "reason": f"Test {alt_lb}d momentum lookback (current: {current_lb}d, Sharpe={ctx.get('sharpe', 0):.3f})",
            "source": "rule_momentum",
        })

        # Top N sectors
        current_top = params.get("momentum_top_n", 3)
        alt_top = rng.choice([2, 3, 4])
        if alt_top != current_top:
            suggestions.append({
                "param": "momentum_top_n",
                "current": current_top,
                "proposed": alt_top,
                "reason": f"Test top/bottom {alt_top} sectors (current: {current_top})",
                "source": "rule_momentum",
            })

        # Rebalance frequency
        current_rebal = params.get("momentum_rebal_days", 21)
        alt_rebal = rng.choice([10, 15, 21, 42])
        if alt_rebal != current_rebal:
            suggestions.append({
                "param": "momentum_rebal_days",
                "current": current_rebal,
                "proposed": alt_rebal,
                "reason": f"Test {alt_rebal}d rebalance frequency (current: {current_rebal}d)",
                "source": "rule_momentum",
            })

        # Max weight
        current_w = params.get("momentum_max_weight", 0.10)
        alt_w = rng.choice([0.06, 0.08, 0.10, 0.12, 0.15])
        if abs(alt_w - current_w) > 0.01:
            suggestions.append({
                "param": "momentum_max_weight",
                "current": current_w,
                "proposed": alt_w,
                "reason": f"Test {alt_w:.0%} max weight (current: {current_w:.0%})",
                "source": "rule_momentum",
            })

        return suggestions[:3]  # Max 3 momentum suggestions per cycle

    def ask_gpt_for_ideas(self, weaknesses: List[Dict], current_params: Dict) -> List[Dict]:
        """
        Query GPT for optimization ideas. Short focused prompts, max 3 calls.

        Returns list of suggestion dicts compatible with generate_parameter_suggestions output.
        """
        gpt = self._get_gpt()
        if gpt is None or not gpt.available:
            log.info("GPT unavailable — skipping GPT suggestions")
            return []

        if self._gpt_calls_used >= MAX_GPT_CALLS_PER_CYCLE:
            log.info("GPT call limit reached (%d/%d)", self._gpt_calls_used, MAX_GPT_CALLS_PER_CYCLE)
            return []

        suggestions = []

        # Build compact weakness summary
        weakness_lines = []
        for w in weaknesses[:3]:  # Only top 3 weaknesses
            weakness_lines.append(w["description"])
        weakness_summary = "; ".join(weakness_lines)

        # Build compact params (only relevant ones)
        compact_params = {}
        for key in ["regime_z_calm", "regime_z_normal", "regime_z_tension",
                     "regime_size_calm", "regime_size_tension",
                     "signal_vix_hard", "trade_max_holding_days",
                     "signal_entry_threshold"]:
            if key in current_params:
                compact_params[key] = current_params[key]

        # GPT Call 1: Diagnose
        prompt1 = (
            f"Weaknesses: {weakness_summary}. "
            f"Key params: {json.dumps(compact_params)}. "
            f"Which 2 parameters should change first and to what exact values?"
        )
        self._gpt_calls_used += 1
        response = gpt._query(prompt1)
        if not response:
            return suggestions

        # Parse GPT response for parameter suggestions
        parsed = self._parse_gpt_suggestions(response, current_params)
        suggestions.extend(parsed)

        # GPT Call 2: Refine if we got something
        if parsed and self._gpt_calls_used < MAX_GPT_CALLS_PER_CYCLE:
            prompt2 = (
                f"You suggested: {response[:200]}. "
                f"What is the expected Sharpe improvement and what risk should we watch?"
            )
            self._gpt_calls_used += 1
            gpt._query(prompt2)  # For context building, response logged

        return suggestions

    def _parse_gpt_suggestions(self, response: str, current_params: Dict) -> List[Dict]:
        """
        Parse GPT response text for concrete parameter value suggestions.

        Looks for patterns like "param_name = value" or "param_name to value".
        """
        import re
        suggestions = []

        # Known param names to look for
        param_names = {name for name, *_ in TUNABLE_PARAMS}

        for param_name in param_names:
            # Match patterns like: regime_z_calm = 0.9, regime_z_calm to 0.9,
            # regime_z_calm: 0.9, set regime_z_calm 0.9
            patterns = [
                rf"{param_name}\s*[=:→]\s*([\d.]+)",
                rf"{param_name}\s+to\s+([\d.]+)",
                rf"set\s+{param_name}\s+([\d.]+)",
                rf"{param_name}\s*from\s+[\d.]+\s+to\s+([\d.]+)",
            ]
            for pattern in patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    try:
                        proposed = float(match.group(1))
                        # Validate against known ranges
                        for tname, tmin, tmax, _step, _desc in TUNABLE_PARAMS:
                            if tname == param_name:
                                if tmin <= proposed <= tmax:
                                    current = current_params.get(param_name)
                                    if current is not None and proposed != current:
                                        suggestions.append({
                                            "param": param_name,
                                            "current": current,
                                            "proposed": proposed,
                                            "reason": f"GPT suggestion: {param_name} -> {proposed}",
                                            "source": "gpt",
                                        })
                                break
                    except (ValueError, TypeError):
                        pass
                    break  # Only first match per param

        return suggestions

    def test_suggestion(self, suggestion: Dict) -> Dict:
        """
        Test a parameter change in sandbox by running a backtest with modified params.

        Returns dict with: sharpe_before, sharpe_after, delta, win_rate_after,
        max_dd_after, passed (bool).
        """
        if self.prices is None:
            return {"passed": False, "error": "no price data"}

        param = suggestion["param"]
        proposed = suggestion["proposed"]
        current = suggestion["current"]

        log.info("Testing: %s = %s -> %s", param, current, proposed)

        try:
            # Create modified settings via env override
            os.environ[param.upper()] = str(proposed)

            # Force fresh settings
            from config.settings import Settings
            test_settings = Settings()

            # Run best available strategy with settings-mapped parameters
            from analytics.methodology_lab import MethodologyLab
            lab = MethodologyLab(self.prices, test_settings)

            # Try RelativeMomentum (best strategy), fall back to AlphaWhitelistMR
            try:
                from analytics.methodology_lab import RelativeMomentum
                test_method = RelativeMomentum(
                    lookback=int(getattr(test_settings, "momentum_lookback", 21)),
                    top_n=int(getattr(test_settings, "momentum_top_n", 3)),
                    rebal_days=int(getattr(test_settings, "momentum_rebal_days", 21)),
                    max_weight=float(getattr(test_settings, "momentum_max_weight", 0.10)),
                    vol_scale=bool(getattr(test_settings, "momentum_vol_scale", True)),
                )
            except ImportError:
                from analytics.methodology_lab import AlphaWhitelistMR
                test_method = AlphaWhitelistMR(
                    z_entry=getattr(test_settings, "regime_z_calm", 0.7),
                    max_hold=getattr(test_settings, "trade_max_holding_days", 25),
                    vix_kill=getattr(test_settings, "signal_vix_hard", 32),
                    regime_sizing={
                        "CALM": getattr(test_settings, "regime_size_calm", 1.3),
                        "NORMAL": getattr(test_settings, "regime_size_normal", 1.0),
                        "TENSION": getattr(test_settings, "regime_size_tension", 0.6),
                        "CRISIS": 0.0,
                    },
                )
            result = lab.run_methodology(test_method)

            # Compare vs baseline using WR + Sharpe composite
            baseline_wr = getattr(self, "_last_baseline_wr", 0.50)
            baseline_sh = getattr(self, "_last_baseline_sharpe", 0.0)

            # Composite: 60% WR improvement + 40% Sharpe improvement
            wr_delta = result.win_rate - baseline_wr
            sh_delta = result.sharpe - baseline_sh
            composite_delta = 0.6 * wr_delta + 0.4 * sh_delta

            # Pass if: WR improves or stays same, AND enough trades
            passed = (
                result.win_rate >= baseline_wr - 0.02  # WR doesn't drop much
                and result.total_trades >= 100           # Enough trades
                and composite_delta > -0.02              # Overall improvement
            )

            test_result = {
                "param": param,
                "current": current,
                "proposed": proposed,
                "sharpe_before": baseline_sh,
                "sharpe_after": result.sharpe,
                "wr_before": baseline_wr,
                "wr_after": result.win_rate,
                "delta": round(composite_delta, 4),
                "max_dd_after": result.max_drawdown,
                "trades_after": result.total_trades,
                "passed": passed,
            }

            log.info(
                "  Result: Sharpe %.3f -> %.3f (delta=%.3f) %s",
                self._last_baseline_sharpe, result.sharpe, test_result["delta"],
                "PASS" if test_result["passed"] else "FAIL",
            )
            return test_result

        except Exception as e:
            log.warning("  Test failed for %s: %s", param, e)
            return {"param": param, "passed": False, "error": str(e)}
        finally:
            # Clean up env override
            env_key = param.upper()
            if env_key in os.environ:
                del os.environ[env_key]

    def promote_if_better(self, suggestion: Dict, test_result: Dict) -> bool:
        """
        Promote params to settings if Sharpe improved by threshold.

        Promotion is done by writing to a .env overrides file that settings.py reads.
        """
        if not test_result.get("passed", False):
            return False

        if self.dry_run:
            log.info(
                "DRY RUN: Would promote %s = %s (Sharpe +%.3f)",
                suggestion["param"], suggestion["proposed"], test_result["delta"],
            )
            return False

        param = suggestion["param"]
        proposed = suggestion["proposed"]

        # Write to env overrides file
        env_file = ROOT / ".env.auto_improve"
        existing = {}
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    existing[k.strip()] = v.strip()

        existing[param.upper()] = str(proposed)

        lines = [f"# Auto-improve overrides — {datetime.now(timezone.utc).isoformat()}"]
        for k, v in sorted(existing.items()):
            lines.append(f"{k}={v}")
        env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        log.info(
            "PROMOTED: %s = %s (Sharpe +%.3f) -> %s",
            param, proposed, test_result["delta"], env_file.name,
        )
        return True

    # ── Institutional Master Research Improvement Governor ───────────────

    # Bottleneck taxonomy
    BOTTLENECKS = [
        "NO_EDGE",
        "ZERO_TRADE_PATHOLOGY",
        "REGIME_MISMATCH",
        "IMPLEMENTATION_DRAG",
        "RISK_VETO_BINDING",
        "FORMULA_PATHOLOGY",
        "OPTIMIZER_STUCK",
        "HEALTH_DECAY_DOMINANT",
        "DATA_QUALITY_FAILURE",
        "SYSTEM_HEALTHY_MONITOR_ONLY",
    ]

    # Mode map: bottleneck -> improvement mode
    MODE_MAP = {
        "ZERO_TRADE_PATHOLOGY": "STRUCTURAL_REVIEW",
        "NO_EDGE": "FORMULA_RESEARCH",
        "REGIME_MISMATCH": "REGIME_REPAIR",
        "IMPLEMENTATION_DRAG": "COST_REDUCTION",
        "RISK_VETO_BINDING": "FREEZE_AND_MONITOR",
        "FORMULA_PATHOLOGY": "FORMULA_RESEARCH",
        "OPTIMIZER_STUCK": "STRUCTURAL_REVIEW",
        "HEALTH_DECAY_DOMINANT": "ROBUSTNESS_REPAIR",
        "DATA_QUALITY_FAILURE": "FREEZE_AND_MONITOR",
        "SYSTEM_HEALTHY_MONITOR_ONLY": "DAILY_MAINTENANCE",
    }

    # ── 1. ImprovementInputAssembler ─────────────────────────────────

    def assemble_all_inputs(self) -> Dict:
        """
        Load machine_summary from ALL 9 agents.

        Returns dict keyed by agent name with their latest machine-readable
        output, plus an ``available_agents`` list of those successfully loaded.
        """
        inputs: Dict[str, Any] = {}
        available: List[str] = []

        # Helper: load latest JSON from a directory glob
        def _latest_json(directory: Path, glob_pattern: str = "*.json") -> Optional[Dict]:
            if not directory.exists():
                return None
            candidates = sorted(directory.glob(glob_pattern), reverse=True)
            for fp in candidates:
                try:
                    data = json.loads(fp.read_text(encoding="utf-8"))
                    if isinstance(data, dict):
                        return data
                except (json.JSONDecodeError, OSError):
                    continue
            return None

        # Helper: load a single JSON file
        def _load_json(path: Path) -> Optional[Dict]:
            if not path.exists():
                return None
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return data if isinstance(data, dict) else None
            except (json.JSONDecodeError, OSError):
                return None

        agent_sources = {
            "methodology": lambda: _latest_json(ROOT / "agents" / "methodology" / "reports"),
            "optimizer": lambda: _load_json(ROOT / "agents" / "optimizer" / "optimization_history.json"),
            "alpha_decay": lambda: _load_json(ROOT / "agents" / "alpha_decay" / "decay_status.json"),
            "regime_forecaster": lambda: _load_json(ROOT / "agents" / "regime_forecaster" / "regime_forecast.json"),
            "portfolio_construction": lambda: _load_json(ROOT / "agents" / "portfolio_construction" / "portfolio_weights.json"),
            "risk_guardian": lambda: _load_json(ROOT / "agents" / "risk_guardian" / "risk_status.json"),
            "execution": lambda: _load_json(ROOT / "agents" / "execution" / "execution_state.json"),
            "data_scout": lambda: _load_json(ROOT / "agents" / "data_scout" / "scout_report.json"),
            "math": lambda: _latest_json(ROOT / "agents" / "math" / "math_proposals"),
        }

        for agent_name, loader in agent_sources.items():
            try:
                result = loader()
                if result is not None:
                    # Extract machine_summary if present, else use full dict
                    ms = (
                        result.get("machine_summary_v3")
                        or result.get("machine_summary")
                        or result
                    )
                    inputs[agent_name] = ms
                    available.append(agent_name)
                    log.debug("Assembled input from %s", agent_name)
                else:
                    inputs[agent_name] = None
                    log.debug("No data from %s", agent_name)
            except Exception as exc:
                inputs[agent_name] = None
                log.debug("Failed to load %s: %s", agent_name, exc)

        inputs["available_agents"] = available
        log.info(
            "ImprovementInputAssembler: %d/%d agents available: %s",
            len(available), len(agent_sources), ", ".join(available) or "(none)",
        )
        return inputs

    # ── 2. CycleDiagnosisEngine ──────────────────────────────────────

    def diagnose_bottleneck(self, inputs: Dict) -> Dict:
        """
        Deterministic bottleneck diagnosis from assembled agent inputs.

        Returns:
            {bottleneck, confidence, evidence, recommended_mode}
        """
        evidence: List[str] = []

        # --- Extract key signals from inputs ---
        meth = inputs.get("methodology") or {}
        optimizer = inputs.get("optimizer") or {}
        decay = inputs.get("alpha_decay") or {}
        regime = inputs.get("regime_forecaster") or {}
        risk = inputs.get("risk_guardian") or {}
        math_in = inputs.get("math") or {}
        data_scout = inputs.get("data_scout") or {}
        portfolio = inputs.get("portfolio_construction") or {}

        # Strategy-level signals
        all_sharpes: List[float] = []
        total_trades = 0
        strategy_results = meth.get("strategy_results", meth.get("all_results", {}))
        if isinstance(strategy_results, dict):
            for sname, sdata in strategy_results.items():
                if isinstance(sdata, dict):
                    all_sharpes.append(sdata.get("sharpe", 0))
                    total_trades += sdata.get("total_trades", sdata.get("trades", 0))

        # Risk signals
        can_allocate = risk.get("can_allocate", True)
        risk_veto = risk.get("veto_active", False)

        # Decay signals
        decay_statuses = decay.get("strategy_health", decay.get("statuses", {}))
        non_healthy_count = 0
        if isinstance(decay_statuses, dict):
            for _sname, sdata in decay_statuses.items():
                health = sdata.get("health", sdata.get("status", "healthy")) if isinstance(sdata, dict) else sdata
                if isinstance(health, str) and health.lower() not in ("healthy", "stable", "ok"):
                    non_healthy_count += 1

        # Math signals
        math_priority = math_in.get("priority", math_in.get("severity", ""))
        math_critical = isinstance(math_priority, str) and math_priority.lower() in ("critical", "urgent")

        # Optimizer stuck signals
        opt_failures = optimizer.get("recent_failures", optimizer.get("consecutive_failures", 0))
        if isinstance(opt_failures, list):
            opt_failures = len(opt_failures)

        # Data quality signals
        data_quality = data_scout.get("overall_quality", data_scout.get("quality", "ok"))
        data_stale = isinstance(data_quality, str) and data_quality.lower() in ("stale", "missing", "failed", "critical")

        # Regime signals
        current_regime = regime.get("current_regime", regime.get("regime", ""))
        edge_regimes = meth.get("edge_regimes", [])

        # --- Deterministic rule cascade (order matters) ---

        # Rule 1: Zero trades across all strategies
        if total_trades == 0 and len(all_sharpes) > 0:
            evidence.append(f"0 trades across {len(all_sharpes)} strategies")
            return {
                "bottleneck": "ZERO_TRADE_PATHOLOGY",
                "confidence": 0.95,
                "evidence": evidence,
                "recommended_mode": self.MODE_MAP["ZERO_TRADE_PATHOLOGY"],
            }

        # Rule 2: Risk Guardian blocking allocation
        if not can_allocate or risk_veto:
            evidence.append(f"Risk Guardian: can_allocate={can_allocate}, veto_active={risk_veto}")
            return {
                "bottleneck": "RISK_VETO_BINDING",
                "confidence": 0.90,
                "evidence": evidence,
                "recommended_mode": self.MODE_MAP["RISK_VETO_BINDING"],
            }

        # Rule 3: Data quality failure
        if data_stale:
            evidence.append(f"Data quality: {data_quality}")
            return {
                "bottleneck": "DATA_QUALITY_FAILURE",
                "confidence": 0.85,
                "evidence": evidence,
                "recommended_mode": self.MODE_MAP["DATA_QUALITY_FAILURE"],
            }

        # Rule 4: All Sharpe negative with trades > 0
        if all_sharpes and all(s < 0 for s in all_sharpes) and total_trades > 0:
            evidence.append(f"All {len(all_sharpes)} strategies negative Sharpe, {total_trades} trades")
            return {
                "bottleneck": "NO_EDGE",
                "confidence": 0.85,
                "evidence": evidence,
                "recommended_mode": self.MODE_MAP["NO_EDGE"],
            }

        # Rule 5: Health decay dominant (5+ strategies non-healthy)
        if non_healthy_count >= 5:
            evidence.append(f"{non_healthy_count} strategies non-healthy in decay monitor")
            return {
                "bottleneck": "HEALTH_DECAY_DOMINANT",
                "confidence": 0.80,
                "evidence": evidence,
                "recommended_mode": self.MODE_MAP["HEALTH_DECAY_DOMINANT"],
            }

        # Rule 6: Math critical proposals
        if math_critical:
            evidence.append(f"Math has critical proposals: priority={math_priority}")
            return {
                "bottleneck": "FORMULA_PATHOLOGY",
                "confidence": 0.80,
                "evidence": evidence,
                "recommended_mode": self.MODE_MAP["FORMULA_PATHOLOGY"],
            }

        # Rule 7: Optimizer stuck (repeated failures)
        if isinstance(opt_failures, (int, float)) and opt_failures >= 3:
            evidence.append(f"Optimizer: {opt_failures} consecutive failures")
            return {
                "bottleneck": "OPTIMIZER_STUCK",
                "confidence": 0.75,
                "evidence": evidence,
                "recommended_mode": self.MODE_MAP["OPTIMIZER_STUCK"],
            }

        # Rule 8: Regime mismatch (edge in CALM but current regime different)
        if (
            isinstance(current_regime, str)
            and current_regime.upper() not in ("CALM", "")
            and isinstance(edge_regimes, list)
            and edge_regimes
            and all(r.upper() == "CALM" for r in edge_regimes if isinstance(r, str))
        ):
            evidence.append(
                f"Edge only in {edge_regimes} but current regime is {current_regime}"
            )
            return {
                "bottleneck": "REGIME_MISMATCH",
                "confidence": 0.70,
                "evidence": evidence,
                "recommended_mode": self.MODE_MAP["REGIME_MISMATCH"],
            }

        # Rule 9: Implementation drag (gross OK but net killed by costs)
        gross_sharpe = meth.get("gross_sharpe", None)
        net_sharpe = meth.get("net_sharpe", None)
        if (
            gross_sharpe is not None
            and net_sharpe is not None
            and isinstance(gross_sharpe, (int, float))
            and isinstance(net_sharpe, (int, float))
            and gross_sharpe > 0.3
            and net_sharpe < 0
        ):
            evidence.append(
                f"Gross Sharpe={gross_sharpe:.3f} but Net Sharpe={net_sharpe:.3f} (cost drag)"
            )
            return {
                "bottleneck": "IMPLEMENTATION_DRAG",
                "confidence": 0.75,
                "evidence": evidence,
                "recommended_mode": self.MODE_MAP["IMPLEMENTATION_DRAG"],
            }

        # Rule 10: System healthy
        evidence.append("No critical bottleneck detected")
        return {
            "bottleneck": "SYSTEM_HEALTHY_MONITOR_ONLY",
            "confidence": 0.60,
            "evidence": evidence,
            "recommended_mode": self.MODE_MAP["SYSTEM_HEALTHY_MONITOR_ONLY"],
        }

    # ── 3. ImprovementModeSelector ───────────────────────────────────

    def select_mode(self, diagnosis: Dict, governance: Dict) -> Dict:
        """
        Select improvement mode based on diagnosis and governance state.

        Returns:
            {primary_mode, secondary_mode, allowed_actions, promotion_allowed}
        """
        bottleneck = diagnosis.get("bottleneck", "SYSTEM_HEALTHY_MONITOR_ONLY")
        primary_mode = self.MODE_MAP.get(bottleneck, "DAILY_MAINTENANCE")
        gov_mode = governance.get("mode", "full")

        # Secondary mode: if stuck, add STRUCTURAL_REVIEW; otherwise DAILY_MAINTENANCE
        secondary_mode = "DAILY_MAINTENANCE"
        if bottleneck in ("NO_EDGE", "REGIME_MISMATCH"):
            secondary_mode = "STRUCTURAL_REVIEW"
        elif bottleneck == "HEALTH_DECAY_DOMINANT":
            secondary_mode = "FORMULA_RESEARCH"

        # Allowed actions per mode
        action_map = {
            "STRUCTURAL_REVIEW": [
                "parameter_change", "formula_sandbox", "methodology_revalidation",
                "shadow_validation", "no_action",
            ],
            "FORMULA_RESEARCH": [
                "formula_sandbox", "parameter_change", "shadow_validation", "no_action",
            ],
            "REGIME_REPAIR": [
                "parameter_change", "methodology_revalidation", "shadow_validation", "no_action",
            ],
            "COST_REDUCTION": [
                "parameter_change", "portfolio_haircut", "shadow_validation", "no_action",
            ],
            "FREEZE_AND_MONITOR": ["risk_freeze", "no_action"],
            "ROBUSTNESS_REPAIR": [
                "parameter_change", "methodology_revalidation", "portfolio_haircut",
                "shadow_validation", "no_action",
            ],
            "DAILY_MAINTENANCE": [
                "parameter_change", "shadow_validation", "no_action",
            ],
        }
        allowed_actions = action_map.get(primary_mode, ["no_action"])

        # Promotion allowed only when governance is not skip and mode is not freeze
        promotion_allowed = (
            gov_mode != "skip"
            and primary_mode not in ("FREEZE_AND_MONITOR",)
        )
        # Cautious governance limits promotion
        if gov_mode == "cautious":
            promotion_allowed = True  # but downstream logic limits count

        log.info(
            "ModeSelector: primary=%s secondary=%s promotion_allowed=%s (gov=%s)",
            primary_mode, secondary_mode, promotion_allowed, gov_mode,
        )
        return {
            "primary_mode": primary_mode,
            "secondary_mode": secondary_mode,
            "allowed_actions": allowed_actions,
            "promotion_allowed": promotion_allowed,
        }

    # ── 4. StuckDetectionEngine ──────────────────────────────────────

    def detect_stuck(self) -> Dict:
        """
        Detect if the improvement engine is stuck by analysing history.

        Returns:
            {stuck_state, consecutive_stuck_cycles, exhausted_families,
             escalation_needed}
        """
        history = _load_improvement_log()
        cycles = history.get("cycles", [])

        result: Dict[str, Any] = {
            "stuck_state": "NORMAL_PROGRESS",
            "consecutive_stuck_cycles": 0,
            "exhausted_families": [],
            "escalation_needed": False,
        }

        if len(cycles) < 2:
            return result

        # --- Consecutive same bottleneck ---
        recent_bottlenecks: List[str] = []
        for c in reversed(cycles[-10:]):
            ms = c.get("machine_summary", {})
            bn = ms.get("diagnosed_bottleneck", "")
            if bn:
                recent_bottlenecks.append(bn)
            else:
                break  # stop at first cycle without institutional diagnosis

        consecutive = 1
        if len(recent_bottlenecks) >= 2:
            for i in range(1, len(recent_bottlenecks)):
                if recent_bottlenecks[i] == recent_bottlenecks[0]:
                    consecutive += 1
                else:
                    break

        if consecutive >= 5:
            result["stuck_state"] = "STUCK_STRUCTURAL"
            result["escalation_needed"] = True
        elif consecutive >= 3:
            result["stuck_state"] = "STUCK_LOCAL"

        result["consecutive_stuck_cycles"] = consecutive

        # --- Exhausted parameter families ---
        param_test_counts: Dict[str, int] = {}
        param_improve_counts: Dict[str, int] = {}
        for c in cycles[-20:]:
            for detail in c.get("details", []):
                p = detail.get("param", "")
                if not p:
                    continue
                param_test_counts[p] = param_test_counts.get(p, 0) + 1
                if detail.get("promoted", False):
                    param_improve_counts[p] = param_improve_counts.get(p, 0) + 1

        exhausted = []
        for p, test_count in param_test_counts.items():
            if test_count >= 3 and param_improve_counts.get(p, 0) == 0:
                exhausted.append(p)
        result["exhausted_families"] = exhausted

        # --- No promoted changes in 5+ cycles ---
        recent_promotions = sum(
            c.get("promoted", c.get("suggestions_promoted", 0))
            for c in cycles[-5:]
        )
        if recent_promotions == 0 and len(cycles) >= 5:
            if result["stuck_state"] == "NORMAL_PROGRESS":
                result["stuck_state"] = "LOW_PROGRESS"

        log.info(
            "StuckDetection: state=%s consecutive=%d exhausted=%d escalation=%s",
            result["stuck_state"], result["consecutive_stuck_cycles"],
            len(result["exhausted_families"]), result["escalation_needed"],
        )
        return result

    # ── 5. CandidateActionFactory ────────────────────────────────────

    def generate_candidate_actions(
        self, diagnosis: Dict, mode: Dict, inputs: Dict
    ) -> List[Dict]:
        """
        Generate candidate improvement actions (not just parameter changes).

        Returns list of action dicts with: type, target, rationale, priority.
        """
        actions: List[Dict] = []
        bottleneck = diagnosis.get("bottleneck", "SYSTEM_HEALTHY_MONITOR_ONLY")
        allowed = mode.get("allowed_actions", ["no_action"])
        primary = mode.get("primary_mode", "DAILY_MAINTENANCE")

        meth = inputs.get("methodology") or {}
        math_in = inputs.get("math") or {}
        decay = inputs.get("alpha_decay") or {}
        risk = inputs.get("risk_guardian") or {}

        # --- FREEZE_AND_MONITOR: risk freeze or data freeze ---
        if primary == "FREEZE_AND_MONITOR":
            if "risk_freeze" in allowed:
                actions.append({
                    "type": "risk_freeze",
                    "target": "portfolio",
                    "rationale": f"Bottleneck={bottleneck}: freeze all allocations",
                    "priority": 1,
                })
            actions.append({
                "type": "no_action",
                "target": None,
                "rationale": "Monitor-only mode active",
                "priority": 99,
            })
            return actions

        # --- Formula pathology: sandbox math proposals ---
        if bottleneck in ("FORMULA_PATHOLOGY", "NO_EDGE") and "formula_sandbox" in allowed:
            math_target = math_in.get("target_function", "compute_distortion_score")
            actions.append({
                "type": "formula_sandbox",
                "target": math_target,
                "rationale": f"Math has critical proposals for {math_target}",
                "priority": 2,
            })

        # --- Regime mismatch: methodology revalidation ---
        if bottleneck == "REGIME_MISMATCH" and "methodology_revalidation" in allowed:
            current_regime = (inputs.get("regime_forecaster") or {}).get(
                "current_regime", "TENSION"
            )
            actions.append({
                "type": "methodology_revalidation",
                "target": f"regime_{current_regime}",
                "rationale": f"Edge absent in current regime {current_regime}",
                "priority": 2,
            })

        # --- Health decay: portfolio haircut on decaying strategies ---
        if bottleneck == "HEALTH_DECAY_DOMINANT" and "portfolio_haircut" in allowed:
            decay_statuses = decay.get("strategy_health", decay.get("statuses", {}))
            if isinstance(decay_statuses, dict):
                for sname, sdata in list(decay_statuses.items())[:3]:
                    health = sdata.get("health", "healthy") if isinstance(sdata, dict) else sdata
                    if isinstance(health, str) and health.lower() not in ("healthy", "stable", "ok"):
                        actions.append({
                            "type": "portfolio_haircut",
                            "target": sname,
                            "rationale": f"Structural decay detected: health={health}",
                            "priority": 3,
                        })

        # --- Parameter changes: use existing smart + rule-based generators ---
        if "parameter_change" in allowed:
            # Get parameter suggestions from existing pipeline
            try:
                smart = self.generate_smart_suggestions(report=None)
                for s in smart[:3]:
                    actions.append({
                        "type": "parameter_change",
                        "target": s.get("param", ""),
                        "rationale": s.get("rationale", s.get("reason", "")),
                        "priority": 4,
                        "_suggestion": s,  # carry full suggestion for sandbox
                    })
            except Exception as exc:
                log.debug("Smart suggestion generation failed: %s", exc)

        # --- Shadow validation for promising but unproven ideas ---
        if "shadow_validation" in allowed and bottleneck not in (
            "SYSTEM_HEALTHY_MONITOR_ONLY",
        ):
            # Suggest shadow tracking for risky parameter moves
            for a in actions:
                if a["type"] == "parameter_change":
                    actions.append({
                        "type": "shadow_validation",
                        "target": f"shadow_{a['target']}",
                        "rationale": f"Shadow-track {a['target']} before promotion",
                        "priority": 5,
                    })
                    break  # one shadow candidate is enough

        # --- Fallback: no_action if system healthy ---
        if not actions or bottleneck == "SYSTEM_HEALTHY_MONITOR_ONLY":
            actions.append({
                "type": "no_action",
                "target": None,
                "rationale": "System healthy, monitor only",
                "priority": 99,
            })

        # Sort by priority
        actions.sort(key=lambda a: a.get("priority", 99))
        log.info(
            "CandidateActionFactory: %d actions generated for mode=%s",
            len(actions), primary,
        )
        return actions

    # ── 6. Multi-Stage Sandbox ───────────────────────────────────────

    def run_institutional_sandbox(self, action: Dict) -> Dict:
        """
        Multi-stage institutional sandbox for a candidate action.

        Stages:
            0: eligibility  — governance allows this action type?
            1: fast_validation  — quick backtest (if parameter_change)
            2: institutional_validation  — methodology-compatible?
            3: downstream_impact  — does it worsen regime/tail/cost?
            4: decision  — PROMOTE / SHADOW / REJECT / FREEZE

        Returns:
            {stage_results, final_decision, decision_rationale, test_result}
        """
        stages: Dict[str, Any] = {}
        action_type = action.get("type", "no_action")

        # ── Stage 0: Eligibility ──
        if action_type == "no_action":
            stages["stage_0_eligibility"] = "SKIP"
            return {
                "stage_results": stages,
                "final_decision": "NO_ACTION",
                "decision_rationale": "No action required",
                "test_result": {},
            }

        if action_type == "risk_freeze":
            stages["stage_0_eligibility"] = "PASS"
            return {
                "stage_results": stages,
                "final_decision": "FREEZE",
                "decision_rationale": "Risk freeze activated by governance",
                "test_result": {},
            }

        stages["stage_0_eligibility"] = "PASS"

        # ── Stage 1: Fast validation (parameter changes only) ──
        test_result: Dict = {}
        if action_type == "parameter_change":
            suggestion = action.get("_suggestion")
            if suggestion and suggestion.get("param"):
                try:
                    test_result = self.test_suggestion(suggestion)
                    stages["stage_1_fast_validation"] = (
                        "PASS" if test_result.get("passed", False) else "FAIL"
                    )
                except Exception as exc:
                    stages["stage_1_fast_validation"] = f"ERROR: {exc}"
                    test_result = {"passed": False, "error": str(exc)}
            else:
                stages["stage_1_fast_validation"] = "SKIP_NO_SUGGESTION"
        else:
            # Non-parameter actions pass fast validation by default
            stages["stage_1_fast_validation"] = "PASS_NON_PARAM"

        # ── Stage 2: Institutional validation ──
        # Check that the change is methodology-compatible
        if test_result.get("passed", False) or action_type != "parameter_change":
            stages["stage_2_institutional_validation"] = "PASS"
        else:
            stages["stage_2_institutional_validation"] = "FAIL_FAST_VAL"

        # ── Stage 3: Downstream impact ──
        if stages.get("stage_2_institutional_validation") == "PASS":
            # For parameter changes, check regime degradation via safety gate
            if action_type == "parameter_change" and test_result:
                suggestion = action.get("_suggestion", {})
                delta = test_result.get("delta", 0)
                safe = self.validate_promotion_safety(
                    param=suggestion.get("param", ""),
                    old_val=suggestion.get("current", suggestion.get("current_value")),
                    new_val=suggestion.get("proposed", suggestion.get("new_value")),
                    delta_sharpe=delta,
                    test_result=test_result,
                )
                stages["stage_3_downstream_impact"] = "PASS" if safe else "FAIL_SAFETY"
            else:
                stages["stage_3_downstream_impact"] = "PASS_NON_PARAM"
        else:
            stages["stage_3_downstream_impact"] = "SKIP"

        # ── Stage 4: Decision ──
        if action_type == "parameter_change":
            if (
                stages.get("stage_1_fast_validation", "").startswith("PASS")
                and stages.get("stage_3_downstream_impact", "").startswith("PASS")
            ):
                final_decision = "PROMOTE"
                rationale = "All stages passed"
            elif stages.get("stage_1_fast_validation", "").startswith("PASS"):
                final_decision = "SHADOW"
                rationale = "Fast validation passed but downstream impact check failed"
            else:
                final_decision = "REJECT"
                rationale = "Fast validation failed"
        elif action_type in ("formula_sandbox", "methodology_revalidation"):
            final_decision = "SHADOW"
            rationale = f"{action_type} requires shadow validation period"
        elif action_type == "portfolio_haircut":
            final_decision = "SHADOW"
            rationale = "Portfolio haircut requires shadow validation before promotion"
        elif action_type == "shadow_validation":
            final_decision = "SHADOW"
            rationale = "Explicitly requested shadow tracking"
        else:
            final_decision = "REJECT"
            rationale = f"Unknown action type: {action_type}"

        stages["stage_4_decision"] = final_decision

        log.info(
            "InstitutionalSandbox: action=%s target=%s decision=%s",
            action_type, action.get("target", "?"), final_decision,
        )
        return {
            "stage_results": stages,
            "final_decision": final_decision,
            "decision_rationale": rationale,
            "test_result": test_result,
        }

    # ── 7. System Health Scores ──────────────────────────────────────

    def compute_system_scores(self, inputs: Dict) -> Dict:
        """
        Compute aggregate system health and improvability scores.

        Returns:
            {system_health_score, system_improvability_score,
             structural_bottleneck_score, local_tuning_viable}
        """
        scores: Dict[str, Any] = {
            "system_health_score": 0.50,
            "system_improvability_score": 0.50,
            "structural_bottleneck_score": 0.50,
            "local_tuning_viable": True,
        }

        available = inputs.get("available_agents", [])
        meth = inputs.get("methodology") or {}
        decay = inputs.get("alpha_decay") or {}
        risk = inputs.get("risk_guardian") or {}

        # --- System health: weighted combination ---
        health_signals: List[float] = []

        # Agent availability (0-1)
        agent_ratio = len(available) / 9.0 if available else 0.0
        health_signals.append(agent_ratio)

        # Best Sharpe from methodology (normalize: -1..+2 -> 0..1)
        strategy_results = meth.get("strategy_results", meth.get("all_results", {}))
        best_sharpe = -1.0
        if isinstance(strategy_results, dict):
            for _sname, sdata in strategy_results.items():
                if isinstance(sdata, dict):
                    s = sdata.get("sharpe", -1)
                    if isinstance(s, (int, float)) and s > best_sharpe:
                        best_sharpe = s
        sharpe_score = max(0.0, min(1.0, (best_sharpe + 1.0) / 3.0))
        health_signals.append(sharpe_score)

        # Risk guardian healthy
        can_allocate = risk.get("can_allocate", True)
        health_signals.append(1.0 if can_allocate else 0.0)

        # Decay health
        decay_statuses = decay.get("strategy_health", decay.get("statuses", {}))
        if isinstance(decay_statuses, dict) and decay_statuses:
            healthy_count = 0
            for _s, sd in decay_statuses.items():
                h = sd.get("health", "healthy") if isinstance(sd, dict) else sd
                if isinstance(h, str) and h.lower() in ("healthy", "stable", "ok"):
                    healthy_count += 1
            health_signals.append(healthy_count / max(1, len(decay_statuses)))
        else:
            health_signals.append(0.5)  # unknown

        if health_signals:
            scores["system_health_score"] = round(
                sum(health_signals) / len(health_signals), 2
            )

        # --- Improvability: can local tuning help? ---
        # High if Sharpe is slightly negative (room to improve)
        # Low if Sharpe is deeply negative (structural problem)
        if best_sharpe > 0.3:
            scores["system_improvability_score"] = 0.30  # already good
        elif best_sharpe > 0:
            scores["system_improvability_score"] = 0.70  # some room
        elif best_sharpe > -0.3:
            scores["system_improvability_score"] = 0.80  # most room
        else:
            scores["system_improvability_score"] = 0.40  # structural

        # --- Structural bottleneck ---
        # High if problem is deeper than parameter tuning
        if best_sharpe < -0.5 or not can_allocate:
            scores["structural_bottleneck_score"] = 0.80
            scores["local_tuning_viable"] = False
        elif best_sharpe < 0:
            scores["structural_bottleneck_score"] = 0.50
            scores["local_tuning_viable"] = True
        else:
            scores["structural_bottleneck_score"] = 0.20
            scores["local_tuning_viable"] = True

        log.info(
            "SystemScores: health=%.2f improvability=%.2f structural=%.2f tuning_viable=%s",
            scores["system_health_score"],
            scores["system_improvability_score"],
            scores["structural_bottleneck_score"],
            scores["local_tuning_viable"],
        )
        return scores

    # ── 8. Downstream Contract Builder ───────────────────────────────

    def build_downstream_contract(
        self, mode: Dict, actions: List[Dict], results: List[Dict]
    ) -> Dict:
        """
        Build contract for downstream agents based on improvement mode and results.

        Returns dict keyed by agent name with action directives.
        """
        primary = mode.get("primary_mode", "DAILY_MAINTENANCE")

        # Default contract: all agents continue normal
        contract: Dict[str, Dict[str, Any]] = {
            "methodology": {"action": "continue_normal"},
            "optimizer": {"action": "continue_normal"},
            "math": {"action": "continue_normal"},
            "regime_forecaster": {"action": "continue_normal"},
            "portfolio_construction": {"action": "continue_normal"},
            "risk_guardian": {"action": "continue_normal"},
            "execution": {"action": "continue_normal"},
            "data_scout": {"action": "continue_normal"},
            "alpha_decay": {"action": "continue_normal"},
        }

        # Mode-specific directives
        if primary == "FREEZE_AND_MONITOR":
            contract["portfolio_construction"] = {"action": "freeze_allocation_changes"}
            contract["risk_guardian"] = {"action": "maintain_veto"}
            contract["execution"] = {"action": "caution_mode"}
            contract["optimizer"] = {"action": "pause_optimization"}

        elif primary == "FORMULA_RESEARCH":
            contract["math"] = {"action": "prioritize_distortion_score"}
            contract["methodology"] = {"action": "revalidate_formulas"}
            contract["optimizer"] = {"action": "formula_research_mode"}

        elif primary == "REGIME_REPAIR":
            # Find which regime needs repair from actions
            regime_target = ""
            for a in actions:
                t = a.get("target", "")
                if isinstance(t, str) and t.startswith("regime_"):
                    regime_target = t.replace("regime_", "")
                    break

            contract["methodology"] = {
                "action": f"revalidate_{regime_target}_regime" if regime_target else "revalidate_all_regimes",
            }
            contract["optimizer"] = {
                "action": "regime_repair_mode",
                "params": ["z_entry", "regime_size", "conviction_scale"],
            }
            contract["regime_forecaster"] = {"action": "increase_monitoring"}

        elif primary == "COST_REDUCTION":
            contract["execution"] = {"action": "minimize_costs"}
            contract["optimizer"] = {"action": "cost_reduction_mode"}
            contract["portfolio_construction"] = {"action": "reduce_turnover"}

        elif primary == "STRUCTURAL_REVIEW":
            contract["methodology"] = {"action": "full_revalidation"}
            contract["optimizer"] = {"action": "expanded_search_space"}
            contract["math"] = {"action": "review_all_formulas"}

        elif primary == "ROBUSTNESS_REPAIR":
            contract["alpha_decay"] = {"action": "intensify_monitoring"}
            contract["methodology"] = {"action": "robustness_revalidation"}
            contract["portfolio_construction"] = {"action": "conservative_allocation"}

        elif primary == "DAILY_MAINTENANCE":
            pass  # defaults are fine

        # Enrich with promoted parameter info from results
        promoted_params = []
        for r in results:
            if r.get("final_decision") == "PROMOTE":
                tr = r.get("test_result", {})
                if tr.get("param"):
                    promoted_params.append(tr["param"])
        if promoted_params:
            contract["optimizer"]["promoted_params"] = promoted_params
            contract["methodology"]["action"] = "revalidate_after_promotion"

        log.info("DownstreamContract: mode=%s directives for %d agents", primary, len(contract))
        return contract

    # ── 9. Machine Summary Builder ───────────────────────────────────

    def _build_institutional_machine_summary(
        self,
        diagnosis: Dict,
        stuck: Dict,
        mode: Dict,
        scores: Dict,
        sandbox_results: List[Dict],
        downstream_contract: Dict,
    ) -> Dict:
        """
        Build the institutional machine_summary for consumption by other agents.
        """
        promote_count = sum(1 for r in sandbox_results if r.get("final_decision") == "PROMOTE")
        shadow_count = sum(1 for r in sandbox_results if r.get("final_decision") == "SHADOW")
        freeze_count = sum(1 for r in sandbox_results if r.get("final_decision") == "FREEZE")

        # Cycle result string
        cycle_result = f"{shadow_count}_SHADOWED_{promote_count}_PROMOTED"
        if freeze_count > 0:
            cycle_result += f"_{freeze_count}_FROZEN"

        # Top blockers from diagnosis evidence
        top_blockers = diagnosis.get("evidence", [])[:3]

        # Next recommended action
        bottleneck = diagnosis.get("bottleneck", "SYSTEM_HEALTHY_MONITOR_ONLY")
        next_action_map = {
            "NO_EDGE": "run_formula_research",
            "ZERO_TRADE_PATHOLOGY": "run_structural_review",
            "REGIME_MISMATCH": "run_regime_specific_tuning",
            "IMPLEMENTATION_DRAG": "run_cost_reduction",
            "RISK_VETO_BINDING": "wait_for_risk_clearance",
            "FORMULA_PATHOLOGY": "run_math_review",
            "OPTIMIZER_STUCK": "escalate_to_structural_review",
            "HEALTH_DECAY_DOMINANT": "run_robustness_repair",
            "DATA_QUALITY_FAILURE": "wait_for_data_quality",
            "SYSTEM_HEALTHY_MONITOR_ONLY": "monitor",
        }
        next_action = next_action_map.get(bottleneck, "monitor")

        summary = {
            "cycle_result": cycle_result,
            "primary_mode": mode.get("primary_mode", "DAILY_MAINTENANCE"),
            "diagnosed_bottleneck": bottleneck,
            "stuck_state": stuck.get("stuck_state", "NORMAL_PROGRESS"),
            "promote_count": promote_count,
            "shadow_count": shadow_count,
            "freeze_count": freeze_count,
            "escalation_flag": stuck.get("escalation_needed", False),
            "next_recommended_action": next_action,
            "top_blockers": top_blockers,
            "local_tuning_viable": scores.get("local_tuning_viable", True),
            "structural_review_needed": scores.get("structural_bottleneck_score", 0) > 0.6,
            "system_health_score": scores.get("system_health_score", 0.5),
            "system_improvability_score": scores.get("system_improvability_score", 0.5),
            "downstream_contract": downstream_contract,
        }
        return summary

    # ── Main Cycle ───────────────────────────────────────────────────────

    def run_cycle(self) -> Dict:
        """
        Run one full institutional improvement cycle.

        Master Research Improvement Governor steps:
          1. Assemble ALL agent inputs (institutional)
          2. Diagnose bottleneck (institutional)
          3. Detect stuck state (institutional)
          4. Select improvement mode (institutional)
          5. Compute system scores (institutional)
          6. Check governance prerequisites (existing, enhanced)
          7. Run methodology evaluation (existing)
          8. Identify weaknesses (existing)
          9. Generate candidate actions (institutional, replaces old suggestions)
         10. Run institutional sandbox (institutional, replaces old test loop)
         11. Decide promote/shadow/freeze/escalate (institutional)
         12. Build downstream contract (institutional)
         13. Build machine_summary (institutional)
         14. Save + publish

        All new institutional steps wrapped in try/except to preserve existing
        behavior if any institutional component fails.
        """
        cycle_start = datetime.now(timezone.utc)
        cycle_id = str(uuid.uuid4())[:12]
        log.info("=" * 70)
        log.info("INSTITUTIONAL IMPROVEMENT CYCLE %s START — %s", cycle_id, cycle_start.isoformat())
        log.info("=" * 70)

        cycle_record: Dict[str, Any] = {
            "cycle_id": cycle_id,
            "timestamp": cycle_start.isoformat(),
            "governance_status": "unknown",
            "mode": "full",
            "metrics_before": {},
            "weaknesses": [],
            "suggestions_generated": 0,
            "suggestions_tested": 0,
            "suggestions_promoted": 0,
            "promoted": 0,
            "best_improvement": {},
            "metrics_after": {},
            "gpt_used": False,
            "gpt_suggestions": [],
            "details": [],
            "machine_summary": {},
        }

        # Institutional state containers (safe defaults)
        inst_inputs: Dict = {}
        inst_diagnosis: Dict = {"bottleneck": "SYSTEM_HEALTHY_MONITOR_ONLY", "confidence": 0, "evidence": [], "recommended_mode": "DAILY_MAINTENANCE"}
        inst_stuck: Dict = {"stuck_state": "NORMAL_PROGRESS", "consecutive_stuck_cycles": 0, "exhausted_families": [], "escalation_needed": False}
        inst_mode: Dict = {"primary_mode": "DAILY_MAINTENANCE", "secondary_mode": "DAILY_MAINTENANCE", "allowed_actions": ["parameter_change", "no_action"], "promotion_allowed": True}
        inst_scores: Dict = {"system_health_score": 0.5, "system_improvability_score": 0.5, "structural_bottleneck_score": 0.5, "local_tuning_viable": True}
        inst_candidate_actions: List[Dict] = []
        inst_sandbox_results: List[Dict] = []
        inst_downstream_contract: Dict = {}
        institutional_active = False

        try:
            # ══════════════════════════════════════════════════════════
            # INSTITUTIONAL STEP 1: Assemble ALL agent inputs
            # ══════════════════════════════════════════════════════════
            try:
                log.info("Institutional Step 1/12: Assembling all agent inputs...")
                inst_inputs = self.assemble_all_inputs()
                cycle_record["available_agents"] = inst_inputs.get("available_agents", [])
                institutional_active = True
            except Exception as exc:
                log.warning("Institutional input assembly failed (non-fatal): %s", exc)

            # ══════════════════════════════════════════════════════════
            # INSTITUTIONAL STEP 2: Diagnose bottleneck
            # ══════════════════════════════════════════════════════════
            if institutional_active:
                try:
                    log.info("Institutional Step 2/12: Diagnosing bottleneck...")
                    inst_diagnosis = self.diagnose_bottleneck(inst_inputs)
                    cycle_record["diagnosed_bottleneck"] = inst_diagnosis.get("bottleneck")
                    cycle_record["bottleneck_confidence"] = inst_diagnosis.get("confidence")
                except Exception as exc:
                    log.warning("Institutional diagnosis failed (non-fatal): %s", exc)

            # ══════════════════════════════════════════════════════════
            # INSTITUTIONAL STEP 3: Detect stuck state
            # ══════════════════════════════════════════════════════════
            if institutional_active:
                try:
                    log.info("Institutional Step 3/12: Detecting stuck state...")
                    inst_stuck = self.detect_stuck()
                    cycle_record["stuck_state"] = inst_stuck.get("stuck_state")
                except Exception as exc:
                    log.warning("Institutional stuck detection failed (non-fatal): %s", exc)

            # ══════════════════════════════════════════════════════════
            # STEP 4 (existing enhanced): Check governance prerequisites
            # ══════════════════════════════════════════════════════════
            log.info("Step 4/12: Checking governance prerequisites...")
            gov = self.check_governance_prerequisites()
            cycle_record["governance_status"] = gov["status"]
            cycle_record["mode"] = gov["mode"]

            if gov["mode"] == "skip":
                log.warning("All strategies REJECTED — skipping optimization cycle")
                cycle_record["status"] = "skipped_governance"
                cycle_record["machine_summary"] = {
                    "cycle_result": "SKIPPED_ALL_REJECTED",
                    "next_recommended_action": "review_methodology_governance",
                    "governance_recommendation": gov["reason"],
                    "diagnosed_bottleneck": inst_diagnosis.get("bottleneck", "unknown"),
                    "stuck_state": inst_stuck.get("stuck_state", "NORMAL_PROGRESS"),
                }
                self._save_cycle(cycle_record)
                return cycle_record

            # ══════════════════════════════════════════════════════════
            # INSTITUTIONAL STEP 5: Select improvement mode
            # ══════════════════════════════════════════════════════════
            if institutional_active:
                try:
                    log.info("Institutional Step 5/12: Selecting improvement mode...")
                    inst_mode = self.select_mode(inst_diagnosis, gov)
                    cycle_record["improvement_mode"] = inst_mode.get("primary_mode")
                except Exception as exc:
                    log.warning("Institutional mode selection failed (non-fatal): %s", exc)

            # ══════════════════════════════════════════════════════════
            # INSTITUTIONAL STEP 6: Compute system scores
            # ══════════════════════════════════════════════════════════
            if institutional_active:
                try:
                    log.info("Institutional Step 6/12: Computing system scores...")
                    inst_scores = self.compute_system_scores(inst_inputs)
                    cycle_record["system_scores"] = inst_scores
                except Exception as exc:
                    log.warning("Institutional system scores failed (non-fatal): %s", exc)

            # Cautious mode limits
            max_suggestions = MAX_SUGGESTIONS_PER_CYCLE
            if gov["mode"] == "cautious":
                max_suggestions = min(3, MAX_SUGGESTIONS_PER_CYCLE)
                log.info("Cautious mode: limiting to %d suggestions", max_suggestions)

            # ══════════════════════════════════════════════════════════
            # STEP 7 (existing): Evaluate current state
            # ══════════════════════════════════════════════════════════
            log.info("Step 7/12: Running methodology evaluation...")
            metrics = self.run_evaluation()
            if not metrics:
                log.error("Evaluation returned no results — aborting cycle")
                cycle_record["error"] = "evaluation_failed"
                cycle_record["status"] = "error"
                self._save_cycle(cycle_record)
                return cycle_record

            cycle_record["metrics_before"] = {
                "sharpe": metrics.get("best_sharpe", 0),
                "win_rate": metrics.get("best_win_rate", 0),
                "max_dd": metrics.get("best_max_dd", 0),
                "best_strategy": metrics.get("best_name", ""),
                "trades": metrics.get("best_trades", 0),
            }

            # Cache baseline using AlphaWhitelistMR specifically (not "best" which may be 0-trade)
            all_r = metrics.get("all_results", {})
            # Use best available strategy as baseline (RelativeMomentum > AlphaWhitelistMR)
            best_baseline = all_r.get("RELATIVE_MOMENTUM",
                            all_r.get("ALPHA_WHITELIST_MR",
                            all_r.get("ALPHA_WHITELIST_MR_LOOSE", {})))
            if best_baseline:
                self._last_baseline_sharpe = best_baseline.get("sharpe", 0)
                self._last_baseline_wr = best_baseline.get("win_rate", 0.50)
            else:
                self._last_baseline_sharpe = metrics.get("best_sharpe", 0)
                self._last_baseline_wr = metrics.get("best_win_rate", 0.50)

            # ══════════════════════════════════════════════════════════
            # STEP 8 (existing): Identify weaknesses
            # ══════════════════════════════════════════════════════════
            log.info("Step 8/12: Identifying weaknesses...")
            weaknesses = self.identify_weaknesses(metrics)
            cycle_record["weaknesses"] = [w["description"] for w in weaknesses]

            if not weaknesses:
                log.info("No weaknesses identified — system looks healthy!")
                cycle_record["status"] = "healthy"
                # Build institutional summary even for healthy cycles
                if institutional_active:
                    try:
                        inst_downstream_contract = self.build_downstream_contract(
                            inst_mode, [], []
                        )
                        cycle_record["machine_summary"] = self._build_institutional_machine_summary(
                            inst_diagnosis, inst_stuck, inst_mode, inst_scores,
                            [], inst_downstream_contract,
                        )
                    except Exception:
                        cycle_record["machine_summary"] = {
                            "cycle_result": "HEALTHY_NO_CHANGES",
                            "next_recommended_action": "monitor",
                            "governance_recommendation": "system performing within targets",
                        }
                else:
                    cycle_record["machine_summary"] = {
                        "cycle_result": "HEALTHY_NO_CHANGES",
                        "next_recommended_action": "monitor",
                        "governance_recommendation": "system performing within targets",
                    }
                self._save_cycle(cycle_record)
                return cycle_record

            # ══════════════════════════════════════════════════════════
            # INSTITUTIONAL STEP 9: Generate candidate actions
            # ══════════════════════════════════════════════════════════
            use_institutional_pipeline = False
            if institutional_active:
                try:
                    log.info("Institutional Step 9/12: Generating candidate actions...")
                    inst_candidate_actions = self.generate_candidate_actions(
                        inst_diagnosis, inst_mode, inst_inputs,
                    )
                    use_institutional_pipeline = len(inst_candidate_actions) > 0
                except Exception as exc:
                    log.warning("Institutional candidate generation failed (non-fatal): %s", exc)

            if use_institutional_pipeline:
                # ══════════════════════════════════════════════════════
                # INSTITUTIONAL STEP 10: Run institutional sandbox
                # ══════════════════════════════════════════════════════
                log.info("Institutional Step 10/12: Running institutional sandbox for %d actions...",
                         len(inst_candidate_actions))
                promoted = 0
                best_delta = 0.0
                best_param = ""

                for i, action in enumerate(inst_candidate_actions[:max_suggestions], 1):
                    log.info("  Sandbox action %d/%d: type=%s target=%s",
                             i, min(len(inst_candidate_actions), max_suggestions),
                             action.get("type"), action.get("target"))
                    try:
                        sandbox_result = self.run_institutional_sandbox(action)
                        inst_sandbox_results.append(sandbox_result)

                        # INSTITUTIONAL STEP 11: Decide promote/shadow/freeze/escalate
                        decision = sandbox_result.get("final_decision", "REJECT")
                        test_result = sandbox_result.get("test_result", {})

                        detail: Dict[str, Any] = {
                            "action_type": action.get("type"),
                            "target": action.get("target"),
                            "rationale": action.get("rationale", ""),
                            "decision": decision,
                            "stage_results": sandbox_result.get("stage_results", {}),
                        }

                        if decision == "PROMOTE" and inst_mode.get("promotion_allowed", True):
                            suggestion = action.get("_suggestion")
                            if suggestion and not self.dry_run:
                                if self.promote_if_better(suggestion, test_result):
                                    promoted += 1
                                    detail["promoted"] = True
                                    delta = test_result.get("delta", 0)
                                    if delta > best_delta:
                                        best_delta = delta
                                        best_param = suggestion.get("param", "")
                                else:
                                    detail["promoted"] = False
                            elif suggestion and self.dry_run:
                                log.info("DRY RUN: Would promote %s", suggestion.get("param"))
                                detail["promoted"] = False
                                detail["dry_run"] = True
                            else:
                                detail["promoted"] = False
                        else:
                            detail["promoted"] = False

                        # Copy param-level info for backward compatibility
                        if action.get("_suggestion"):
                            s = action["_suggestion"]
                            detail["param"] = s.get("param", "")
                            detail["current"] = s.get("current", s.get("current_value"))
                            detail["proposed"] = s.get("proposed", s.get("new_value"))
                            detail["test_passed"] = test_result.get("passed", False)
                            detail["sharpe_delta"] = test_result.get("delta", 0)

                        cycle_record["details"].append(detail)

                    except Exception as exc:
                        log.warning("  Sandbox failed for action %d: %s", i, exc)
                        inst_sandbox_results.append({
                            "final_decision": "REJECT",
                            "decision_rationale": f"sandbox error: {exc}",
                            "test_result": {},
                        })

                cycle_record["promoted"] = promoted
                cycle_record["suggestions_promoted"] = promoted
                cycle_record["suggestions_tested"] = len(inst_sandbox_results)
                cycle_record["suggestions_generated"] = len(inst_candidate_actions)
                if best_param:
                    cycle_record["best_improvement"] = {
                        "param": best_param,
                        "delta_sharpe": round(best_delta, 4),
                    }

            else:
                # ══════════════════════════════════════════════════════
                # FALLBACK: Original suggestion pipeline (existing code)
                # ══════════════════════════════════════════════════════
                log.info("Fallback: Using original suggestion pipeline...")

                # Step 3 (original): Generate smart suggestions from governance + alpha_decay
                log.info("Generating smart suggestions from governance...")
                smart_suggestions = self.generate_smart_suggestions(report=None)

                # Also generate rule-based suggestions
                rule_suggestions = self.generate_parameter_suggestions(weaknesses)

                # Merge: smart first, then rule-based (dedup by param)
                suggestions: List[Dict] = []
                seen_params: set = set()
                for s in smart_suggestions + rule_suggestions:
                    param_key = s.get("param", "")
                    if param_key and param_key not in seen_params:
                        seen_params.add(param_key)
                        # Normalize keys for compatibility
                        if "current_value" in s and "current" not in s:
                            s["current"] = s["current_value"]
                        if "new_value" in s and "proposed" not in s:
                            s["proposed"] = s["new_value"]
                        if "rationale" in s and "reason" not in s:
                            s["reason"] = s["rationale"]
                        suggestions.append(s)

                cycle_record["suggestions_generated"] = len(suggestions)

                # Step 4 (original): Ask GPT for ideas
                log.info("Querying GPT for ideas...")
                current_params = self.current_params()
                try:
                    gpt_ideas = self.ask_gpt_for_ideas(weaknesses, current_params)
                    if gpt_ideas:
                        cycle_record["gpt_used"] = True
                        cycle_record["gpt_suggestions"] = [
                            f"{s['param']}={s['proposed']}" for s in gpt_ideas
                        ]
                        # Add GPT suggestions that aren't already covered
                        for idea in gpt_ideas:
                            if idea["param"] not in seen_params:
                                seen_params.add(idea["param"])
                                suggestions.append(idea)
                except Exception as e:
                    log.warning("GPT query failed (non-fatal): %s", e)

                # Limit total suggestions
                suggestions = suggestions[:max_suggestions]
                cycle_record["suggestions_tested"] = len(suggestions)

                # Step 5 (original): Test each suggestion
                log.info("Testing %d suggestions in sandbox...", len(suggestions))
                promoted = 0
                best_delta = 0.0
                best_param = ""
                for i, suggestion in enumerate(suggestions, 1):
                    log.info("  Testing suggestion %d/%d: %s", i, len(suggestions), suggestion["param"])
                    test_result = self.test_suggestion(suggestion)
                    detail = {
                        "param": suggestion["param"],
                        "current": suggestion.get("current", suggestion.get("current_value")),
                        "proposed": suggestion.get("proposed", suggestion.get("new_value")),
                        "source": suggestion.get("source", "rule"),
                        "reason": suggestion.get("reason", suggestion.get("rationale", "")),
                        "test_passed": test_result.get("passed", False),
                        "sharpe_delta": test_result.get("delta", 0),
                    }

                    # Step 6 (original): Validate promotion safety before promoting
                    delta = test_result.get("delta", 0)
                    if test_result.get("passed", False):
                        safe = self.validate_promotion_safety(
                            param=suggestion["param"],
                            old_val=suggestion.get("current", suggestion.get("current_value")),
                            new_val=suggestion.get("proposed", suggestion.get("new_value")),
                            delta_sharpe=delta,
                            test_result=test_result,
                        )
                        if safe and self.promote_if_better(suggestion, test_result):
                            promoted += 1
                            detail["promoted"] = True
                            if delta > best_delta:
                                best_delta = delta
                                best_param = suggestion["param"]
                        else:
                            detail["promoted"] = False
                            if not safe:
                                detail["safety_blocked"] = True
                    else:
                        detail["promoted"] = False

                    cycle_record["details"].append(detail)

                cycle_record["promoted"] = promoted
                cycle_record["suggestions_promoted"] = promoted
                if best_param:
                    cycle_record["best_improvement"] = {
                        "param": best_param,
                        "delta_sharpe": round(best_delta, 4),
                    }

            # ══════════════════════════════════════════════════════════
            # STEP (existing): Final evaluation if anything was promoted
            # ══════════════════════════════════════════════════════════
            promoted = cycle_record.get("promoted", 0)
            if promoted > 0 and not self.dry_run:
                log.info("Step 12/12: Re-evaluating after %d promotions...", promoted)
                new_metrics = self.run_evaluation()
                if new_metrics:
                    cycle_record["metrics_after"] = {
                        "sharpe": new_metrics.get("best_sharpe", 0),
                        "win_rate": new_metrics.get("best_win_rate", 0),
                        "max_dd": new_metrics.get("best_max_dd", 0),
                        "best_strategy": new_metrics.get("best_name", ""),
                    }
            else:
                cycle_record["metrics_after"] = cycle_record["metrics_before"]

            cycle_record["status"] = "completed"

            # ══════════════════════════════════════════════════════════
            # INSTITUTIONAL STEP 12: Build downstream contract
            # ══════════════════════════════════════════════════════════
            if institutional_active:
                try:
                    log.info("Institutional Step 12/12: Building downstream contract + machine_summary...")
                    inst_downstream_contract = self.build_downstream_contract(
                        inst_mode, inst_candidate_actions, inst_sandbox_results,
                    )
                    cycle_record["downstream_contract"] = inst_downstream_contract

                    # INSTITUTIONAL STEP 13: Build institutional machine_summary
                    cycle_record["machine_summary"] = self._build_institutional_machine_summary(
                        inst_diagnosis, inst_stuck, inst_mode, inst_scores,
                        inst_sandbox_results, inst_downstream_contract,
                    )
                except Exception as exc:
                    log.warning("Institutional summary build failed (non-fatal): %s", exc)
                    # Fallback to basic machine_summary
                    cycle_record["machine_summary"] = self._fallback_machine_summary(
                        promoted, weaknesses, gov,
                    )
            else:
                cycle_record["machine_summary"] = self._fallback_machine_summary(
                    promoted, weaknesses, gov,
                )

            # ══════════════════════════════════════════════════════════
            # INSTITUTIONAL: Save machine_summary to disk for other agents
            # ══════════════════════════════════════════════════════════
            try:
                summary_path = ROOT / "agents" / "auto_improve" / "machine_summary.json"
                summary_path.parent.mkdir(parents=True, exist_ok=True)
                summary_path.write_text(
                    json.dumps(cycle_record["machine_summary"], indent=2, default=str),
                    encoding="utf-8",
                )
                log.info("Machine summary saved to %s", summary_path)
            except Exception as exc:
                log.warning("Failed to save machine_summary: %s", exc)

            # ══════════════════════════════════════════════════════════
            # INSTITUTIONAL: Bus publish + registry heartbeat
            # ══════════════════════════════════════════════════════════
            try:
                from agents.shared.agent_bus import get_bus
                get_bus().publish("auto_improve", cycle_record["machine_summary"])
            except Exception:
                pass
            try:
                from agents.shared.agent_registry import get_registry, AgentStatus
                get_registry().heartbeat("auto_improve", AgentStatus.COMPLETED)
            except Exception:
                pass

        except Exception as e:
            log.error("Cycle failed: %s\n%s", e, traceback.format_exc())
            cycle_record["status"] = "error"
            cycle_record["error"] = str(e)
            cycle_record["machine_summary"] = {
                "cycle_result": "ERROR",
                "next_recommended_action": "investigate_failure",
                "governance_recommendation": str(e)[:200],
                "diagnosed_bottleneck": inst_diagnosis.get("bottleneck", "unknown"),
                "stuck_state": inst_stuck.get("stuck_state", "NORMAL_PROGRESS"),
            }

        self._save_cycle(cycle_record)

        elapsed = (datetime.now(timezone.utc) - cycle_start).total_seconds()
        log.info("=" * 70)
        log.info(
            "CYCLE %s COMPLETE in %.0fs — gov=%s mode=%s bottleneck=%s stuck=%s "
            "tested=%d promoted=%d",
            cycle_id, elapsed,
            cycle_record["governance_status"],
            cycle_record.get("improvement_mode", cycle_record["mode"]),
            cycle_record.get("diagnosed_bottleneck", "?"),
            cycle_record.get("stuck_state", "?"),
            cycle_record.get("suggestions_tested", 0),
            cycle_record.get("promoted", 0),
        )
        log.info("=" * 70)

        return cycle_record

    def _fallback_machine_summary(
        self, promoted: int, weaknesses: List[Dict], gov: Dict
    ) -> Dict:
        """Fallback machine_summary when institutional pipeline is unavailable."""
        cycle_result = f"{promoted}_PROMOTED" if promoted > 0 else "0_PROMOTED"
        next_action = "monitor"
        if promoted > 0:
            next_action = "run_regime_specific_tuning"
        elif len(weaknesses) > 2:
            next_action = "expand_search_space"

        governance_rec = "system stable"
        if promoted > 0:
            governance_rec = "re-validate after promotion"
        elif gov.get("mode") == "cautious":
            governance_rec = "wait for APPROVED strategies before full optimization"

        return {
            "cycle_result": cycle_result,
            "next_recommended_action": next_action,
            "governance_recommendation": governance_rec,
        }

    def _save_cycle(self, cycle_record: Dict) -> None:
        """Save a cycle record to the improvement log."""
        data = _load_improvement_log()
        data["cycles"].append(cycle_record)
        # Keep last 100 cycles
        if len(data["cycles"]) > 100:
            data["cycles"] = data["cycles"][-100:]
        _save_improvement_log(data)


# ─────────────────────────────────────────────────────────────────────────────
# Status Display
# ─────────────────────────────────────────────────────────────────────────────

def show_status() -> None:
    """Show improvement history from the log."""
    data = _load_improvement_log()
    cycles = data.get("cycles", [])

    if not cycles:
        print("No improvement cycles recorded yet.")
        print(f"  Log file: {IMPROVEMENT_LOG_PATH}")
        return

    print(f"\n{'='*70}")
    print(f" AUTO-IMPROVE STATUS — {len(cycles)} cycles recorded")
    print(f"{'='*70}\n")

    # Summary of recent cycles
    for cycle in cycles[-10:]:
        ts = cycle.get("timestamp", "?")[:19]
        status = cycle.get("status", "?")
        before = cycle.get("metrics_before", {})
        after = cycle.get("metrics_after", {})
        promoted = cycle.get("promoted", 0)
        tested = cycle.get("suggestions_tested", 0)
        gpt = "GPT" if cycle.get("gpt_used") else "rules"

        sharpe_before = before.get("sharpe", 0)
        sharpe_after = after.get("sharpe", 0)
        delta = sharpe_after - sharpe_before if sharpe_after and sharpe_before else 0

        print(f"  {ts}  {status:>10}  Sharpe {sharpe_before:+.3f} -> {sharpe_after:+.3f} "
              f"(delta={delta:+.3f})  tested={tested} promoted={promoted}  [{gpt}]")

        # Show weaknesses
        for w in cycle.get("weaknesses", [])[:2]:
            print(f"    weakness: {w}")

        # Show promoted changes
        for d in cycle.get("details", []):
            if d.get("promoted"):
                print(f"    PROMOTED: {d['param']} = {d['current']} -> {d['proposed']} "
                      f"(Sharpe +{d.get('sharpe_delta', 0):.3f})")

    # Overall trajectory
    if len(cycles) >= 2:
        first = cycles[0].get("metrics_before", {}).get("sharpe", 0)
        last_after = cycles[-1].get("metrics_after", {}).get("sharpe", 0)
        total_promoted = sum(c.get("promoted", 0) for c in cycles)
        print(f"\n  Total trajectory: Sharpe {first:+.3f} -> {last_after:+.3f}")
        print(f"  Total promotions: {total_promoted}")

    print(f"\n  Log file: {IMPROVEMENT_LOG_PATH}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SRV Auto-Improve — Feedback loop for continuous system improvement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python agents/auto_improve.py --cycle      # Run one improvement cycle
  python agents/auto_improve.py --status     # Show improvement history
  python agents/auto_improve.py --dry-run    # Evaluate only, don't promote
        """,
    )
    parser.add_argument("--cycle", action="store_true", help="Run one improvement cycle")
    parser.add_argument("--status", action="store_true", help="Show improvement history")
    parser.add_argument("--dry-run", action="store_true", help="Evaluate but don't promote changes")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.status:
        show_status()
        return 0

    if args.cycle or args.dry_run:
        improver = AutoImprover(dry_run=args.dry_run)
        result = improver.run_cycle()
        status = result.get("status", "unknown")
        return 0 if status in ("completed", "healthy") else 1

    # Default: show status
    show_status()
    return 0


if __name__ == "__main__":
    sys.exit(main())
