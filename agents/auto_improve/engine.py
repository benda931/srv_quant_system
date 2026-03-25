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

        # 3. Check for poor win rates
        best_wr = metrics.get("best_win_rate", 0)
        if best_wr < 0.50:
            weaknesses.append({
                "type": "low_win_rate",
                "description": f"Win rate below 50%: {best_wr:.1%}",
                "severity": "medium",
                "context": {"win_rate": best_wr},
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
        """Suggestions when all strategies are negative — conservative shift."""
        suggestions = []
        # Shorten hold period
        current_hold = params.get("trade_max_holding_days", 20)
        proposed_hold = max(10, current_hold - 5)
        if proposed_hold != current_hold:
            suggestions.append({
                "param": "trade_max_holding_days",
                "current": current_hold,
                "proposed": proposed_hold,
                "reason": "Shorten max hold to reduce time decay on losing trades",
                "source": "rule",
            })
        # Raise entry bar
        current_z = params.get("regime_z_calm", 0.7)
        proposed_z = min(1.5, round(current_z + 0.15, 2))
        if proposed_z != current_z:
            suggestions.append({
                "param": "regime_z_calm",
                "current": current_z,
                "proposed": proposed_z,
                "reason": "Require stronger signal in CALM regime",
                "source": "rule",
            })
        return suggestions

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

            # Run a quick backtest with best methodology only
            from analytics.methodology_lab import MethodologyLab, PcaZReversal
            lab = MethodologyLab(self.prices, test_settings)

            # Run just the baseline methodology for speed
            baseline = PcaZReversal()
            result = lab.run_methodology(baseline)

            test_result = {
                "param": param,
                "current": current,
                "proposed": proposed,
                "sharpe_before": self._last_baseline_sharpe,
                "sharpe_after": result.sharpe,
                "delta": result.sharpe - self._last_baseline_sharpe,
                "win_rate_after": result.win_rate,
                "max_dd_after": result.max_drawdown,
                "trades_after": result.total_trades,
                "passed": (result.sharpe - self._last_baseline_sharpe) >= SHARPE_PROMOTION_THRESHOLD,
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

    # ── Main Cycle ───────────────────────────────────────────────────────

    def run_cycle(self) -> Dict:
        """
        Run one full improvement cycle.

        Steps:
          1. Run methodology evaluation
          2. Identify weaknesses
          3. Generate rule-based suggestions
          4. Ask GPT for ideas (if available)
          5. Test each suggestion in sandbox
          6. Promote if improvement validated
          7. Log everything
        """
        cycle_start = datetime.now(timezone.utc)
        log.info("=" * 60)
        log.info("AUTO-IMPROVE CYCLE START — %s", cycle_start.isoformat())
        log.info("=" * 60)

        cycle_record = {
            "timestamp": cycle_start.isoformat(),
            "metrics_before": {},
            "weaknesses": [],
            "suggestions_tested": 0,
            "promoted": 0,
            "metrics_after": {},
            "gpt_used": False,
            "gpt_suggestions": [],
            "details": [],
        }

        try:
            # Step 1: Evaluate current state
            log.info("Step 1/6: Running methodology evaluation...")
            metrics = self.run_evaluation()
            if not metrics:
                log.error("Evaluation returned no results — aborting cycle")
                cycle_record["error"] = "evaluation_failed"
                self._save_cycle(cycle_record)
                return cycle_record

            cycle_record["metrics_before"] = {
                "sharpe": metrics.get("best_sharpe", 0),
                "win_rate": metrics.get("best_win_rate", 0),
                "max_dd": metrics.get("best_max_dd", 0),
                "best_strategy": metrics.get("best_name", ""),
                "trades": metrics.get("best_trades", 0),
            }

            # Cache baseline Sharpe for sandbox testing
            self._last_baseline_sharpe = metrics.get("best_sharpe", 0)

            # Step 2: Identify weaknesses
            log.info("Step 2/6: Identifying weaknesses...")
            weaknesses = self.identify_weaknesses(metrics)
            cycle_record["weaknesses"] = [w["description"] for w in weaknesses]

            if not weaknesses:
                log.info("No weaknesses identified — system looks healthy!")
                cycle_record["status"] = "healthy"
                self._save_cycle(cycle_record)
                return cycle_record

            # Step 3: Generate rule-based suggestions
            log.info("Step 3/6: Generating parameter suggestions...")
            suggestions = self.generate_parameter_suggestions(weaknesses)

            # Step 4: Ask GPT for ideas
            log.info("Step 4/6: Querying GPT for ideas...")
            current_params = self.current_params()
            try:
                gpt_ideas = self.ask_gpt_for_ideas(weaknesses, current_params)
                if gpt_ideas:
                    cycle_record["gpt_used"] = True
                    cycle_record["gpt_suggestions"] = [
                        f"{s['param']}={s['proposed']}" for s in gpt_ideas
                    ]
                    # Add GPT suggestions that aren't already covered
                    existing_params = {s["param"] for s in suggestions}
                    for idea in gpt_ideas:
                        if idea["param"] not in existing_params:
                            suggestions.append(idea)
            except Exception as e:
                log.warning("GPT query failed (non-fatal): %s", e)

            # Limit total suggestions
            suggestions = suggestions[:MAX_SUGGESTIONS_PER_CYCLE]
            cycle_record["suggestions_tested"] = len(suggestions)

            # Step 5: Test each suggestion
            log.info("Step 5/6: Testing %d suggestions in sandbox...", len(suggestions))
            promoted = 0
            for i, suggestion in enumerate(suggestions, 1):
                log.info("  Testing suggestion %d/%d: %s", i, len(suggestions), suggestion["param"])
                test_result = self.test_suggestion(suggestion)
                detail = {
                    "param": suggestion["param"],
                    "current": suggestion["current"],
                    "proposed": suggestion["proposed"],
                    "source": suggestion.get("source", "rule"),
                    "reason": suggestion.get("reason", ""),
                    "test_passed": test_result.get("passed", False),
                    "sharpe_delta": test_result.get("delta", 0),
                }

                # Step 6: Promote if better
                if self.promote_if_better(suggestion, test_result):
                    promoted += 1
                    detail["promoted"] = True
                else:
                    detail["promoted"] = False

                cycle_record["details"].append(detail)

            cycle_record["promoted"] = promoted

            # Final evaluation if anything was promoted
            if promoted > 0 and not self.dry_run:
                log.info("Step 6/6: Re-evaluating after %d promotions...", promoted)
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

        except Exception as e:
            log.error("Cycle failed: %s\n%s", e, traceback.format_exc())
            cycle_record["status"] = "error"
            cycle_record["error"] = str(e)

        self._save_cycle(cycle_record)

        elapsed = (datetime.now(timezone.utc) - cycle_start).total_seconds()
        log.info("=" * 60)
        log.info(
            "CYCLE COMPLETE in %.0fs — %d weaknesses, %d tested, %d promoted",
            elapsed, len(cycle_record["weaknesses"]),
            cycle_record["suggestions_tested"], cycle_record["promoted"],
        )
        log.info("=" * 60)

        return cycle_record

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
