"""
agents/methodology/agent_methodology.py
-----------------------------------------
Institutional-Grade Methodology Agent — Research Governance, Walk-Forward
Validation, Anti-Overfitting Controls, Block Bootstrap Monte Carlo,
Regime Attribution, Failure Diagnostics, Strategy Classification.

Runs the full trading methodology pipeline, evaluates backtest quality,
applies promotion gates, and publishes structured results.

CLI:
    python agents/methodology/agent_methodology.py --once
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Root path — 3 levels up: agents/methodology/agent_methodology.py → ROOT ──
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import get_settings, Settings

# ── Logging ──────────────────────────────────────────────────────────────────
_LOG_DIR = ROOT / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

from logging.handlers import RotatingFileHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        RotatingFileHandler(
            _LOG_DIR / "agent_methodology.log",
            maxBytes=50 * 1024 * 1024, backupCount=5, encoding="utf-8",
        ),
    ],
)
log = logging.getLogger("agent_methodology")

# ── System module imports with error handling ──────────────────────────────────
_IMPORTS_OK: Dict[str, bool] = {}

try:
    from scripts.run_all import run_pipeline
    _IMPORTS_OK["run_pipeline"] = True
except ImportError as e:
    log.warning("Could not import run_pipeline: %s", e)
    _IMPORTS_OK["run_pipeline"] = False

try:
    from analytics.stat_arb import QuantEngine
    _IMPORTS_OK["QuantEngine"] = True
except ImportError as e:
    log.warning("Could not import QuantEngine: %s", e)
    _IMPORTS_OK["QuantEngine"] = False

try:
    from analytics.backtest import run_backtest as run_wf_backtest, BacktestResult
    _IMPORTS_OK["backtest"] = True
except ImportError as e:
    log.warning("Could not import backtest: %s", e)
    _IMPORTS_OK["backtest"] = False

try:
    from analytics.signal_stack import SignalStackEngine, signal_stack_summary
    _IMPORTS_OK["signal_stack"] = True
except ImportError as e:
    log.warning("Could not import signal_stack: %s", e)
    _IMPORTS_OK["signal_stack"] = False

try:
    from analytics.signal_regime_safety import compute_regime_safety_score
    _IMPORTS_OK["regime_safety"] = True
except ImportError as e:
    log.warning("Could not import signal_regime_safety: %s", e)
    _IMPORTS_OK["regime_safety"] = False

try:
    from analytics.stress import StressEngine
    _IMPORTS_OK["stress"] = True
except ImportError as e:
    log.warning("Could not import StressEngine: %s", e)
    _IMPORTS_OK["stress"] = False

try:
    from analytics.methodology_lab import MethodologyLab, ALL_METHODOLOGIES
    _IMPORTS_OK["methodology_lab"] = True
except ImportError as e:
    log.warning("Could not import methodology_lab: %s", e)
    _IMPORTS_OK["methodology_lab"] = False

try:
    from analytics.tail_risk import (
        compute_expected_shortfall,
        parametric_correlation_stress,
        tail_correlation_diagnostic,
    )
    _IMPORTS_OK["tail_risk"] = True
except ImportError as e:
    log.warning("Could not import tail_risk: %s", e)
    _IMPORTS_OK["tail_risk"] = False

try:
    from scripts.agent_bus import get_bus
    _IMPORTS_OK["agent_bus"] = True
except ImportError as e:
    log.warning("Could not import agent_bus: %s", e)
    _IMPORTS_OK["agent_bus"] = False

try:
    from agents.shared.agent_registry import get_registry, AgentStatus
    _IMPORTS_OK["registry"] = True
except ImportError as e:
    log.warning("Could not import agent_registry: %s", e)
    _IMPORTS_OK["registry"] = False

try:
    from analytics.performance_metrics import (
        compute_sortino_ratio,
        compute_omega_ratio,
        compute_max_drawdown_duration,
    )
    _IMPORTS_OK["performance_metrics"] = True
except ImportError as e:
    log.warning("Could not import performance_metrics: %s", e)
    _IMPORTS_OK["performance_metrics"] = False

# ── Portfolio Tracker ────────────────────────────────────────────────────────
from agents.methodology.methodology_portfolio import MethodologyPortfolio

# ── Reports directory ─────────────────────────────────────────────────────────
REPORTS_DIR = Path(__file__).resolve().parent / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
METHODOLOGY_VERSION = "2.0.0"
_TRADING_DAYS = 252
_EPSILON = 1e-12
_REGIMES = ["CALM", "NORMAL", "TENSION", "CRISIS"]

# Root-cause failure categories
FAILURE_CATEGORIES = [
    "weak_signal", "unstable_ic", "regime_mismatch", "excessive_dd",
    "overtrading", "cost_drag", "poor_tail", "insufficient_sample",
    "low_robustness", "single_regime", "high_correlation", "low_incremental_alpha",
]

# Strategy classifications
STRATEGY_CLASSIFICATIONS = ["CORE", "DIVERSIFIER", "EXPERIMENTAL", "REDUNDANT", "DISABLE"]


# =============================================================================
# Helper utilities
# =============================================================================

def _sf(x: Any, default: float = 0.0) -> float:
    """Safe float — returns default if value is not finite."""
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _pct_delta(old: float, new: float) -> str:
    """Format percentage change: '+8.3%' / '-2.1%'."""
    if abs(old) < _EPSILON:
        return "N/A"
    delta = (new - old) / abs(old) * 100
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.1f}%"


def _safe_round(val: float, decimals: int = 4) -> float:
    """Round value safely, handling NaN/Inf."""
    try:
        v = float(val)
        return round(v, decimals) if math.isfinite(v) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _compute_data_fingerprint(settings: Settings) -> str:
    """Create a deterministic fingerprint of data-affecting settings."""
    key_params = {
        "pca_window": settings.pca_window,
        "zscore_window": settings.zscore_window,
        "corr_window": settings.corr_window,
        "history_years": settings.history_years,
        "sector_mr_whitelist": settings.sector_mr_whitelist,
    }
    raw = json.dumps(key_params, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _compute_settings_fingerprint(settings: Settings) -> str:
    """Create fingerprint of all trading-relevant settings."""
    key_params = {
        "signal_entry_threshold": settings.signal_entry_threshold,
        "signal_a1_frob": settings.signal_a1_frob,
        "signal_a2_mode": settings.signal_a2_mode,
        "signal_a3_coc": settings.signal_a3_coc,
        "trade_max_holding_days": settings.trade_max_holding_days,
        "regime_conviction_scale_calm": settings.regime_conviction_scale_calm,
        "regime_conviction_scale_crisis": settings.regime_conviction_scale_crisis,
    }
    raw = json.dumps(key_params, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _regime_breakdown_to_dict(breakdown: dict) -> dict:
    """Convert RegimeBreakdown dataclasses to plain dict."""
    result = {}
    for regime, rb in breakdown.items():
        if hasattr(rb, "__dict__"):
            result[regime] = {
                "regime": getattr(rb, "regime", regime),
                "n_walks": getattr(rb, "n_walks", 0),
                "ic_mean": round(_sf(getattr(rb, "ic_mean", 0)), 4),
                "ic_ir": round(_sf(getattr(rb, "ic_ir", 0)), 4),
                "hit_rate": round(_sf(getattr(rb, "hit_rate", 0)), 4),
                "sharpe": round(_sf(getattr(rb, "sharpe", 0)), 4),
            }
        elif isinstance(rb, dict):
            result[regime] = rb
        else:
            result[regime] = {"raw": str(rb)}
    return result


# =============================================================================
# MethodologyScorecard dataclass
# =============================================================================

@dataclass
class MethodologyScorecard:
    """Full institutional scorecard for a single methodology."""
    name: str
    gross_sharpe: float = 0.0
    net_sharpe: float = 0.0
    deflated_sharpe: float = 0.0
    walk_forward_sharpe: float = 0.0
    bootstrap_sharpe_median: float = 0.0
    bootstrap_p_positive: float = 0.0
    hit_rate: float = 0.0
    max_drawdown: float = 0.0
    calmar: float = 0.0
    total_trades: int = 0
    regime_scores: Dict[str, float] = field(default_factory=dict)
    robustness_score: float = 0.0
    stability_score: float = 0.0
    tail_risk_score: float = 0.0
    diversification_score: float = 0.0
    cost_drag_pct: float = 0.0
    classification: str = "EXPERIMENTAL"
    promotion_decision: str = "REJECTED"
    fail_reasons: List[str] = field(default_factory=list)
    final_rank: int = 0

    # Extended metrics
    sortino: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    avg_holding_days: float = 0.0
    total_pnl: float = 0.0
    win_rate_long: float = 0.0
    win_rate_short: float = 0.0

    # Diagnostics
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    failure_tags: List[str] = field(default_factory=list)
    action_candidates: List[str] = field(default_factory=list)
    confidence_score: float = 0.0

    def to_dict(self) -> dict:
        """Serialize to JSON-safe dict."""
        return {k: v for k, v in asdict(self).items()}


# =============================================================================
# ValidationSuite — walk-forward, block bootstrap, cost-aware
# =============================================================================

class ValidationSuite:
    """
    Institutional validation stack: purged walk-forward CV,
    block bootstrap Monte Carlo, cost-aware evaluation.
    """

    def __init__(
        self,
        settings: Settings,
        n_splits: int = 5,
        purge_days: int = 5,
        min_train_days: int = 252,
        n_bootstrap: int = 1000,
        block_size: int = 5,
        cost_bps: float = 5.0,
        slippage_bps: float = 5.0,
    ):
        self.settings = settings
        self.n_splits = n_splits
        self.purge_days = purge_days
        self.min_train_days = min_train_days
        self.n_bootstrap = n_bootstrap
        self.block_size = block_size
        self.cost_bps = cost_bps
        self.slippage_bps = slippage_bps

    def run_walk_forward(
        self,
        strategy_name: str,
        prices,
        target_method,
    ) -> Dict[str, Any]:
        """
        Purged expanding-window walk-forward validation.

        Returns per-split and aggregate metrics: Sharpe, Sortino, Calmar,
        hit_rate, IC, max_dd, skew, kurtosis.
        """
        import pandas as pd

        try:
            if prices is None or prices.empty:
                return {"error": "no price data available"}

            total_days = len(prices)
            split_size = total_days // self.n_splits
            if split_size < 30:
                return {"error": f"insufficient data: {total_days} for {self.n_splits} splits"}

            split_results = []
            for i in range(1, self.n_splits):
                try:
                    train_end_idx = i * split_size - self.purge_days
                    test_start_idx = i * split_size
                    test_end_idx = min((i + 1) * split_size, total_days)

                    if train_end_idx < self.min_train_days or test_start_idx >= total_days:
                        continue

                    test_prices = prices.iloc[test_start_idx:test_end_idx]
                    if len(test_prices) < 20:
                        continue

                    test_lab = MethodologyLab(test_prices, self.settings, step=5)
                    test_result = test_lab.run_methodology(target_method)

                    eq = test_result.equity_curve
                    returns = eq.pct_change().dropna() if len(eq) > 1 else pd.Series([0.0])

                    if len(returns) < 2:
                        continue

                    mean_ret = float(returns.mean())
                    std_ret = float(returns.std(ddof=1))
                    split_sharpe = (mean_ret / std_ret * np.sqrt(_TRADING_DAYS)) if std_ret > _EPSILON else 0.0

                    # Sortino
                    downside = returns[returns < 0]
                    downside_std = float(np.sqrt(np.mean(downside ** 2))) if len(downside) > 0 else _EPSILON
                    split_sortino = (mean_ret / downside_std * np.sqrt(_TRADING_DAYS)) if downside_std > _EPSILON else 0.0

                    # Drawdown
                    cum_ret = (1 + returns).cumprod()
                    running_max = cum_ret.cummax()
                    dd = (cum_ret - running_max) / running_max
                    max_dd = float(dd.min()) if len(dd) > 0 else 0.0

                    # Calmar
                    total_ret = float(cum_ret.iloc[-1] - 1) if len(cum_ret) > 0 else 0.0
                    calmar = abs(total_ret / max_dd) if abs(max_dd) > _EPSILON else 0.0

                    # Higher moments
                    skew_val = float(returns.skew()) if len(returns) > 3 else 0.0
                    kurt_val = float(returns.kurtosis()) if len(returns) > 4 else 0.0

                    split_results.append({
                        "split": i,
                        "train_days": train_end_idx,
                        "test_days": len(test_prices),
                        "purge_days": self.purge_days,
                        "sharpe": _safe_round(split_sharpe),
                        "sortino": _safe_round(split_sortino),
                        "calmar": _safe_round(calmar),
                        "hit_rate": _safe_round(test_result.win_rate),
                        "max_dd": _safe_round(max_dd, 6),
                        "skew": _safe_round(skew_val),
                        "kurtosis": _safe_round(kurt_val),
                        "total_trades": test_result.total_trades,
                        "total_pnl": _safe_round(test_result.total_pnl, 6),
                    })
                except Exception as split_exc:
                    log.debug("WF split %d failed for %s: %s", i, strategy_name, split_exc)
                    split_results.append({"split": i, "error": str(split_exc)})

            valid = [s for s in split_results if "error" not in s]
            if valid:
                agg = self._aggregate_splits(valid)
            else:
                agg = {"error": "no valid splits completed"}

            log.info(
                "Walk-forward %s: %d/%d splits, mean Sharpe=%.3f, consistency=%.2f",
                strategy_name, len(valid), self.n_splits - 1,
                agg.get("mean_sharpe", 0), agg.get("consistency", 0),
            )

            return {
                "strategy_name": strategy_name,
                "n_splits": self.n_splits,
                "purge_days": self.purge_days,
                "min_train_days": self.min_train_days,
                "splits": split_results,
                "aggregate": agg,
            }

        except Exception as exc:
            log.exception("Walk-forward failed for %s", strategy_name)
            return {"error": str(exc), "strategy_name": strategy_name}

    def _aggregate_splits(self, valid: List[dict]) -> dict:
        """Aggregate per-split metrics into summary statistics."""
        n = len(valid)
        sharpes = [s["sharpe"] for s in valid]
        return {
            "mean_sharpe": _safe_round(float(np.mean(sharpes))),
            "std_sharpe": _safe_round(float(np.std(sharpes))),
            "min_sharpe": _safe_round(float(np.min(sharpes))),
            "max_sharpe": _safe_round(float(np.max(sharpes))),
            "mean_sortino": _safe_round(float(np.mean([s["sortino"] for s in valid]))),
            "mean_calmar": _safe_round(float(np.mean([s["calmar"] for s in valid]))),
            "mean_hit_rate": _safe_round(float(np.mean([s["hit_rate"] for s in valid]))),
            "mean_max_dd": _safe_round(float(np.mean([s["max_dd"] for s in valid])), 6),
            "mean_skew": _safe_round(float(np.mean([s["skew"] for s in valid]))),
            "mean_kurtosis": _safe_round(float(np.mean([s["kurtosis"] for s in valid]))),
            "n_valid_splits": n,
            "total_splits": self.n_splits - 1,
            "consistency": _safe_round(
                sum(1 for s in valid if s["sharpe"] > 0) / max(n, 1)
            ),
            "stability_score": _safe_round(self._compute_stability(sharpes)),
        }

    @staticmethod
    def _compute_stability(sharpes: List[float]) -> float:
        """Stability = 1 - CV(Sharpe across folds). Higher = more consistent."""
        if len(sharpes) < 2:
            return 0.0
        mean_s = float(np.mean(sharpes))
        std_s = float(np.std(sharpes, ddof=1))
        if abs(mean_s) < _EPSILON:
            return 0.0
        cv = abs(std_s / mean_s)
        return max(0.0, min(1.0, 1.0 - cv))

    def run_block_bootstrap(
        self,
        strategy_name: str,
        equity_curve,
    ) -> Dict[str, Any]:
        """
        Block bootstrap Monte Carlo for serial dependence.

        Uses block_size=5 to preserve autocorrelation structure.
        Returns confidence intervals for Sharpe, hit_rate, max_dd,
        and P(Sharpe > target) for multiple targets.
        """
        import pandas as pd

        try:
            if equity_curve is None or len(equity_curve) < 20:
                return {"error": "insufficient equity curve data"}

            returns = equity_curve.pct_change().dropna().values
            n_days = len(returns)
            if n_days < 10:
                return {"error": "insufficient return data for bootstrap"}

            rng = np.random.default_rng(42)
            n_blocks = max(1, n_days // self.block_size)

            sharpe_dist = []
            hr_dist = []
            dd_dist = []

            for _ in range(self.n_bootstrap):
                # Block bootstrap: sample blocks with replacement
                block_starts = rng.integers(0, max(1, n_days - self.block_size + 1), size=n_blocks)
                sample = np.concatenate([
                    returns[s:min(s + self.block_size, n_days)]
                    for s in block_starts
                ])[:n_days]

                if len(sample) < 5:
                    continue

                mu = float(np.mean(sample))
                sigma = float(np.std(sample, ddof=1))
                sim_sharpe = (mu / sigma * np.sqrt(_TRADING_DAYS)) if sigma > _EPSILON else 0.0
                sharpe_dist.append(sim_sharpe)

                sim_hr = float(np.mean(sample > 0))
                hr_dist.append(sim_hr)

                cum = np.cumprod(1 + sample)
                running_max = np.maximum.accumulate(cum)
                drawdowns = (cum - running_max) / np.maximum(running_max, _EPSILON)
                sim_dd = float(np.min(drawdowns))
                dd_dist.append(sim_dd)

            sharpe_arr = np.array(sharpe_dist)
            hr_arr = np.array(hr_dist)
            dd_arr = np.array(dd_dist)

            targets = {0.0: "p_sharpe_gt_0", 0.3: "p_sharpe_gt_0_3",
                       0.5: "p_sharpe_gt_0_5", 1.0: "p_sharpe_gt_1_0"}
            p_targets = {}
            for target, key in targets.items():
                p_targets[key] = _safe_round(float(np.mean(sharpe_arr > target)))

            result = {
                "strategy_name": strategy_name,
                "n_bootstrap": self.n_bootstrap,
                "block_size": self.block_size,
                "n_days": n_days,
                "sharpe": {
                    "median": _safe_round(float(np.median(sharpe_arr))),
                    "mean": _safe_round(float(np.mean(sharpe_arr))),
                    "std": _safe_round(float(np.std(sharpe_arr))),
                    "p5": _safe_round(float(np.percentile(sharpe_arr, 5))),
                    "p25": _safe_round(float(np.percentile(sharpe_arr, 25))),
                    "p75": _safe_round(float(np.percentile(sharpe_arr, 75))),
                    "p95": _safe_round(float(np.percentile(sharpe_arr, 95))),
                },
                "hit_rate": {
                    "median": _safe_round(float(np.median(hr_arr))),
                    "p5": _safe_round(float(np.percentile(hr_arr, 5))),
                    "p95": _safe_round(float(np.percentile(hr_arr, 95))),
                },
                "max_dd": {
                    "median": _safe_round(float(np.median(dd_arr)), 6),
                    "p5": _safe_round(float(np.percentile(dd_arr, 5)), 6),
                    "p95": _safe_round(float(np.percentile(dd_arr, 95)), 6),
                },
                "probability_targets": p_targets,
                "confidence_level": (
                    "HIGH" if p_targets.get("p_sharpe_gt_0", 0) > 0.90
                    else "MODERATE" if p_targets.get("p_sharpe_gt_0", 0) > 0.70
                    else "LOW" if p_targets.get("p_sharpe_gt_0", 0) > 0.50
                    else "VERY_LOW"
                ),
            }

            log.info(
                "Block bootstrap %s: median Sharpe=%.3f, P(>0)=%.1f%%, P(>0.3)=%.1f%%",
                strategy_name,
                result["sharpe"]["median"],
                p_targets.get("p_sharpe_gt_0", 0) * 100,
                p_targets.get("p_sharpe_gt_0_3", 0) * 100,
            )

            return result

        except Exception as exc:
            log.exception("Block bootstrap failed for %s", strategy_name)
            return {"error": str(exc), "strategy_name": strategy_name}

    def compute_cost_aware_metrics(
        self,
        strategy_name: str,
        result_obj,
    ) -> Dict[str, Any]:
        """
        Cost-aware evaluation: turnover, slippage, commissions, net Sharpe.

        Rejects strategies where alpha disappears net of costs.
        """
        try:
            trades = getattr(result_obj, "trades", [])
            equity_curve = getattr(result_obj, "equity_curve", None)
            if not trades or equity_curve is None or len(equity_curve) < 10:
                return {"error": "insufficient data for cost analysis"}

            import pandas as pd

            # Gross metrics
            gross_sharpe = _sf(result_obj.sharpe)
            total_pnl_gross = _sf(result_obj.total_pnl)

            # Turnover: sum of absolute weights traded
            total_weight_traded = sum(abs(t.weight) for t in trades)
            n_days = len(equity_curve)
            annualized_turnover = total_weight_traded / max(n_days / _TRADING_DAYS, 0.01)

            # Cost per trade (round-trip)
            cost_per_rt = (self.cost_bps + self.slippage_bps) / 10000.0 * 2
            total_cost = sum(abs(t.weight) * cost_per_rt for t in trades)

            # Net P&L
            net_pnl = total_pnl_gross - total_cost

            # Net Sharpe estimate
            daily_returns = equity_curve.pct_change().dropna()
            daily_cost = total_cost / max(n_days, 1)
            net_returns = daily_returns - daily_cost
            net_std = float(net_returns.std(ddof=1))
            net_sharpe = (float(net_returns.mean()) / net_std * np.sqrt(_TRADING_DAYS)) if net_std > _EPSILON else 0.0

            # Cost drag as percentage of gross alpha
            cost_drag_pct = (total_cost / abs(total_pnl_gross) * 100) if abs(total_pnl_gross) > _EPSILON else 100.0

            # Alpha disappears net of costs?
            alpha_survives = net_sharpe > 0.0 and cost_drag_pct < 80.0

            return {
                "strategy_name": strategy_name,
                "gross_sharpe": _safe_round(gross_sharpe),
                "net_sharpe": _safe_round(net_sharpe),
                "gross_pnl": _safe_round(total_pnl_gross, 6),
                "net_pnl": _safe_round(net_pnl, 6),
                "total_cost": _safe_round(total_cost, 6),
                "cost_bps": self.cost_bps,
                "slippage_bps": self.slippage_bps,
                "annualized_turnover": _safe_round(annualized_turnover),
                "cost_drag_pct": _safe_round(min(cost_drag_pct, 100.0), 2),
                "n_trades": len(trades),
                "alpha_survives_costs": alpha_survives,
            }

        except Exception as exc:
            log.exception("Cost analysis failed for %s", strategy_name)
            return {"error": str(exc), "strategy_name": strategy_name}


# =============================================================================
# OverfittingControl — Deflated Sharpe, ranking stability
# =============================================================================

class OverfittingControl:
    """
    Anti-overfitting controls: Deflated Sharpe Ratio (DSR),
    ranking stability across walk-forward folds.
    """

    @staticmethod
    def deflated_sharpe_ratio(
        observed_sharpe: float,
        n_observations: int,
        n_strategies_tested: int = 19,
        skewness: float = 0.0,
        kurtosis: float = 3.0,
        sharpe_std: float = 1.0,
    ) -> float:
        """
        Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

        Adjusts for multiple testing by computing the expected maximum
        Sharpe from n_strategies_tested independent strategies, then
        testing if observed Sharpe significantly exceeds it.

        Returns the deflated Sharpe (always <= observed).
        """
        try:
            from scipy import stats as sp_stats
        except ImportError:
            # Simplified approximation without scipy
            euler_mascheroni = 0.5772
            expected_max = sharpe_std * (
                (1 - euler_mascheroni) * sp_stats_norm_ppf(1 - 1.0 / n_strategies_tested)
                + euler_mascheroni * sp_stats_norm_ppf(1 - 1.0 / (n_strategies_tested * np.e))
            ) if n_strategies_tested > 1 else 0.0
            return max(0.0, observed_sharpe - expected_max)

        if n_observations < 10 or n_strategies_tested < 1:
            return 0.0

        # Expected maximum Sharpe under null (all strategies are noise)
        euler_mascheroni = 0.5772
        try:
            e_max_sharpe = sharpe_std * (
                (1 - euler_mascheroni) * sp_stats.norm.ppf(1 - 1.0 / n_strategies_tested)
                + euler_mascheroni * sp_stats.norm.ppf(1 - 1.0 / (n_strategies_tested * np.e))
            )
        except Exception:
            e_max_sharpe = 0.0

        # Sharpe standard error adjustment for non-normal returns
        se_sharpe = np.sqrt(
            (1 + 0.5 * observed_sharpe ** 2 - skewness * observed_sharpe
             + ((kurtosis - 3) / 4.0) * observed_sharpe ** 2)
            / max(n_observations - 1, 1)
        )

        if se_sharpe < _EPSILON:
            return 0.0

        # Test statistic: is observed significantly above expected max?
        test_stat = (observed_sharpe - e_max_sharpe) / se_sharpe

        # p-value from standard normal
        try:
            p_value = 1.0 - sp_stats.norm.cdf(test_stat)
        except Exception:
            p_value = 0.5

        # Deflated Sharpe: observed * (1 - 2*p_value), floored at 0
        deflated = observed_sharpe * max(0.0, 1.0 - 2.0 * p_value)
        return _safe_round(deflated)

    @staticmethod
    def ranking_stability(
        walk_forward_results: Dict[str, Any],
        all_wf_results: Dict[str, Dict[str, Any]],
    ) -> float:
        """
        Compute ranking stability across walk-forward folds.

        Measures how consistently a strategy ranks relative to others
        across different time periods. Score 0-1, higher = more stable.
        """
        try:
            target_name = walk_forward_results.get("strategy_name", "")
            splits = walk_forward_results.get("splits", [])
            valid_splits = [s for s in splits if "error" not in s]

            if len(valid_splits) < 2:
                return 0.0

            # Get Sharpe ranks per split
            ranks = []
            for split in valid_splits:
                split_idx = split["split"]
                split_sharpe = split["sharpe"]

                # Collect Sharpes from other strategies for same split
                all_sharpes = [(target_name, split_sharpe)]
                for name, wf in all_wf_results.items():
                    if name == target_name:
                        continue
                    other_splits = wf.get("splits", [])
                    for os in other_splits:
                        if os.get("split") == split_idx and "sharpe" in os:
                            all_sharpes.append((name, os["sharpe"]))
                            break

                # Rank (1 = best)
                sorted_by_sharpe = sorted(all_sharpes, key=lambda x: x[1], reverse=True)
                rank = next(
                    (i + 1 for i, (n, _) in enumerate(sorted_by_sharpe) if n == target_name),
                    len(sorted_by_sharpe),
                )
                ranks.append(rank)

            if len(ranks) < 2:
                return 0.0

            # Stability = 1 - normalized std of ranks
            mean_rank = float(np.mean(ranks))
            std_rank = float(np.std(ranks, ddof=1))
            n_strats = max(len(all_wf_results), 1)
            normalized_std = std_rank / max(n_strats - 1, 1)
            return _safe_round(max(0.0, min(1.0, 1.0 - normalized_std)))

        except Exception as exc:
            log.debug("Ranking stability failed: %s", exc)
            return 0.0

    @staticmethod
    def overfitting_probability(
        is_sharpe: float,
        oos_sharpe: float,
        n_strategies: int = 19,
    ) -> float:
        """
        Estimate probability of overfitting.

        Based on the gap between in-sample and out-of-sample performance
        relative to the number of strategies tested.
        """
        if is_sharpe < _EPSILON:
            return 1.0

        degradation = max(0.0, 1.0 - oos_sharpe / is_sharpe)
        multi_test_penalty = min(1.0, np.log(max(n_strategies, 1)) / 5.0)
        prob = min(1.0, degradation * 0.7 + multi_test_penalty * 0.3)
        return _safe_round(prob, 2)


# =============================================================================
# GovernanceEngine — promotion gates, experiment tracking
# =============================================================================

class GovernanceEngine:
    """
    Research governance: experiment tracking, deterministic promotion gates.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.experiment_id = str(uuid.uuid4())[:12]
        self.data_fingerprint = _compute_data_fingerprint(settings)
        self.settings_fingerprint = _compute_settings_fingerprint(settings)

    def create_governance_record(
        self,
        run_mode: str = "daily",
    ) -> Dict[str, Any]:
        """Create initial governance record for this run."""
        return {
            "experiment_id": self.experiment_id,
            "data_fingerprint": self.data_fingerprint,
            "settings_fingerprint": self.settings_fingerprint,
            "methodology_version": METHODOLOGY_VERSION,
            "run_mode": run_mode,
            "validation_status": "PENDING",
            "promotion_readiness": "PENDING",
            "fail_reasons": [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def evaluate_promotion_gate(
        self,
        scorecard: MethodologyScorecard,
        regime_scores: Dict[str, float],
    ) -> Tuple[str, List[str]]:
        """
        Deterministic promotion gate.

        Returns (decision, fail_reasons) where decision is one of:
        APPROVED / CONDITIONAL / REJECTED.

        Criteria:
        - Minimum net Sharpe > 0.3
        - Max drawdown < 15%
        - Minimum trades > 100
        - Positive in >= 2 regimes
        - Overfitting probability < 50%
        - ES99 proxy (tail_risk_score) acceptable
        - Cost drag < 30% of gross alpha
        - Walk-forward stability > 0.5
        """
        fail_reasons: List[str] = []
        warning_reasons: List[str] = []

        # 1. Minimum net Sharpe
        if scorecard.net_sharpe < 0.3:
            fail_reasons.append(
                f"net_sharpe={scorecard.net_sharpe:.3f} < 0.3 minimum"
            )

        # 2. Max drawdown ceiling
        if abs(scorecard.max_drawdown) > 0.15:
            fail_reasons.append(
                f"max_drawdown={scorecard.max_drawdown:.3f} exceeds 15% ceiling"
            )

        # 3. Minimum trades
        if scorecard.total_trades < 100:
            fail_reasons.append(
                f"total_trades={scorecard.total_trades} < 100 minimum"
            )

        # 4. Regime consistency: positive Sharpe in >= 2 regimes
        positive_regimes = sum(1 for v in regime_scores.values() if v > 0)
        if positive_regimes < 2:
            fail_reasons.append(
                f"positive_in_only_{positive_regimes}_regimes (need >=2)"
            )

        # 5. Overfitting check (via deflated Sharpe proxy)
        if scorecard.deflated_sharpe < 0.0:
            fail_reasons.append(
                f"deflated_sharpe={scorecard.deflated_sharpe:.3f} < 0 (likely overfit)"
            )

        # 6. Tail risk (tail_risk_score 0-1, lower = worse)
        if scorecard.tail_risk_score < 0.3:
            warning_reasons.append(
                f"tail_risk_score={scorecard.tail_risk_score:.2f} indicates poor tail behavior"
            )

        # 7. Cost drag
        if scorecard.cost_drag_pct > 30.0:
            fail_reasons.append(
                f"cost_drag={scorecard.cost_drag_pct:.1f}% > 30% ceiling"
            )

        # 8. Walk-forward stability
        if scorecard.stability_score < 0.5:
            warning_reasons.append(
                f"stability_score={scorecard.stability_score:.2f} < 0.5"
            )

        # Decision
        if fail_reasons:
            decision = "REJECTED"
        elif warning_reasons:
            decision = "CONDITIONAL"
        else:
            decision = "APPROVED"

        return decision, fail_reasons + warning_reasons


# =============================================================================
# RegimeAttributionEngine
# =============================================================================

class RegimeAttributionEngine:
    """
    Per-regime performance attribution, fragility scoring,
    regime diversification, enable/disable recommendations.
    """

    @staticmethod
    def compute_regime_attribution(
        lab_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        For each methodology: performance by regime, fragility, diversification.
        Also identifies best/weakest per regime.
        """
        if not lab_results:
            return {"error": "no lab results"}

        regime_perf: Dict[str, Dict[str, Dict[str, float]]] = {r: {} for r in _REGIMES}
        method_regime_sharpes: Dict[str, Dict[str, float]] = {}

        for name, result in lab_results.items():
            try:
                regime_stats = getattr(result, "regime_stats", {})
                method_regime_sharpes[name] = {}

                for regime in _REGIMES:
                    rs = regime_stats.get(regime) or regime_stats.get(regime.lower()) or {}
                    if isinstance(rs, dict):
                        trades_in_regime = rs.get("n_trades", 0)
                        win_rate = _sf(rs.get("win_rate", 0))
                        avg_pnl = _sf(rs.get("avg_pnl", 0))
                    else:
                        trades_in_regime = getattr(rs, "n_trades", 0)
                        win_rate = _sf(getattr(rs, "win_rate", 0))
                        avg_pnl = _sf(getattr(rs, "avg_pnl", 0))

                    # Approximate regime Sharpe from avg_pnl
                    regime_sharpe = avg_pnl * np.sqrt(_TRADING_DAYS) * 10 if abs(avg_pnl) > _EPSILON else 0.0

                    regime_perf[regime][name] = {
                        "n_trades": trades_in_regime,
                        "win_rate": _safe_round(win_rate),
                        "avg_pnl": _safe_round(avg_pnl, 6),
                        "regime_sharpe": _safe_round(regime_sharpe),
                    }
                    method_regime_sharpes[name][regime] = regime_sharpe

            except Exception as exc:
                log.debug("Regime attribution for %s failed: %s", name, exc)

        # Best/worst per regime
        best_per_regime = {}
        worst_per_regime = {}
        for regime in _REGIMES:
            strats = regime_perf[regime]
            if strats:
                best_name = max(strats, key=lambda s: strats[s]["regime_sharpe"])
                worst_name = min(strats, key=lambda s: strats[s]["regime_sharpe"])
                best_per_regime[regime] = {"strategy": best_name, **strats[best_name]}
                worst_per_regime[regime] = {"strategy": worst_name, **strats[worst_name]}

        # Per-methodology: fragility + diversification
        method_scores = {}
        for name, regime_sharpes in method_regime_sharpes.items():
            values = list(regime_sharpes.values())
            if not values:
                continue

            worst_sharpe = min(values)
            best_sharpe = max(values)
            mean_sharpe = float(np.mean(values))

            # Fragility: how badly it degrades in worst regime
            fragility = abs(worst_sharpe - mean_sharpe) / max(abs(mean_sharpe), _EPSILON)
            fragility = min(fragility, 10.0)

            # Diversification: works across regimes (low std of regime Sharpes)
            std_sharpe = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            diversification = max(0.0, 1.0 - std_sharpe / max(abs(mean_sharpe), _EPSILON))
            diversification = max(0.0, min(1.0, diversification))

            # Positive regime count
            n_positive = sum(1 for v in values if v > 0)

            # Recommended enable/disable
            recommended = {}
            for regime, sharpe in regime_sharpes.items():
                if sharpe > 0.1:
                    recommended[regime] = "ENABLE"
                elif sharpe > -0.1:
                    recommended[regime] = "REDUCE"
                else:
                    recommended[regime] = "DISABLE"

            method_scores[name] = {
                "regime_sharpes": {k: _safe_round(v) for k, v in regime_sharpes.items()},
                "fragility_score": _safe_round(fragility, 2),
                "diversification_score": _safe_round(diversification, 2),
                "n_positive_regimes": n_positive,
                "worst_regime": min(regime_sharpes, key=regime_sharpes.get),
                "best_regime": max(regime_sharpes, key=regime_sharpes.get),
                "recommended_action": recommended,
            }

        return {
            "regime_performance": regime_perf,
            "best_per_regime": best_per_regime,
            "worst_per_regime": worst_per_regime,
            "methodology_scores": method_scores,
        }


# =============================================================================
# MethodologyScorer — builds scorecards, classifies, diagnoses
# =============================================================================

class MethodologyScorer:
    """
    Builds MethodologyScorecard for each strategy, classifies them,
    and provides failure diagnostics.
    """

    def __init__(
        self,
        settings: Settings,
        governance: GovernanceEngine,
        validation: ValidationSuite,
        overfitting: OverfittingControl,
        regime_engine: RegimeAttributionEngine,
    ):
        self.settings = settings
        self.governance = governance
        self.validation = validation
        self.overfitting = overfitting
        self.regime_engine = regime_engine

    def build_scorecards(
        self,
        lab_results: Dict[str, Any],
        prices,
        wf_results: Dict[str, Dict[str, Any]],
        bootstrap_results: Dict[str, Dict[str, Any]],
        cost_results: Dict[str, Dict[str, Any]],
        regime_attribution: Dict[str, Any],
        correlation_matrix: Dict[str, Any],
    ) -> List[MethodologyScorecard]:
        """Build full scorecards for all methodologies."""

        scorecards: List[MethodologyScorecard] = []
        method_scores = regime_attribution.get("methodology_scores", {})
        n_strategies = len(lab_results)

        for name, result in lab_results.items():
            try:
                sc = MethodologyScorecard(name=name)

                # Basic metrics from lab
                sc.gross_sharpe = _sf(result.sharpe)
                sc.hit_rate = _sf(result.win_rate)
                sc.max_drawdown = _sf(result.max_drawdown)
                sc.calmar = _sf(result.calmar)
                sc.total_trades = result.total_trades
                sc.total_pnl = _sf(result.total_pnl)
                sc.avg_holding_days = _sf(result.avg_holding_days)

                # Cost-aware metrics
                cost = cost_results.get(name, {})
                sc.net_sharpe = _sf(cost.get("net_sharpe", sc.gross_sharpe))
                sc.cost_drag_pct = _sf(cost.get("cost_drag_pct", 0))

                # Walk-forward metrics
                wf = wf_results.get(name, {})
                wf_agg = wf.get("aggregate", {})
                sc.walk_forward_sharpe = _sf(wf_agg.get("mean_sharpe", 0))
                sc.stability_score = _sf(wf_agg.get("stability_score", 0))
                sc.sortino = _sf(wf_agg.get("mean_sortino", 0))

                # Bootstrap metrics
                bs = bootstrap_results.get(name, {})
                bs_sharpe = bs.get("sharpe", {})
                sc.bootstrap_sharpe_median = _sf(bs_sharpe.get("median", 0))
                sc.bootstrap_p_positive = _sf(
                    bs.get("probability_targets", {}).get("p_sharpe_gt_0", 0)
                )

                # Deflated Sharpe
                eq = result.equity_curve
                returns = eq.pct_change().dropna() if eq is not None and len(eq) > 1 else None
                skew_val = float(returns.skew()) if returns is not None and len(returns) > 3 else 0.0
                kurt_val = float(returns.kurtosis()) if returns is not None and len(returns) > 4 else 3.0
                n_obs = len(returns) if returns is not None else 0

                sc.deflated_sharpe = self.overfitting.deflated_sharpe_ratio(
                    observed_sharpe=sc.gross_sharpe,
                    n_observations=n_obs,
                    n_strategies_tested=n_strategies,
                    skewness=skew_val,
                    kurtosis=kurt_val,
                )
                sc.skewness = _safe_round(skew_val)
                sc.kurtosis = _safe_round(kurt_val)

                # Regime scores
                ms = method_scores.get(name, {})
                sc.regime_scores = ms.get("regime_sharpes", {})
                sc.diversification_score = _sf(ms.get("diversification_score", 0))

                # Robustness: composite of stability, deflated Sharpe confidence, bootstrap
                sc.robustness_score = _safe_round(
                    0.4 * sc.stability_score
                    + 0.3 * min(1.0, max(0.0, sc.deflated_sharpe / max(sc.gross_sharpe, _EPSILON)))
                    + 0.3 * sc.bootstrap_p_positive
                )

                # Tail risk score (simplified from max_dd and skewness)
                dd_score = max(0.0, 1.0 - abs(sc.max_drawdown) / 0.15)
                skew_score = max(0.0, min(1.0, (skew_val + 1.0) / 2.0))
                sc.tail_risk_score = _safe_round(0.6 * dd_score + 0.4 * skew_score)

                # Classification
                sc.classification = self._classify_strategy(
                    sc, correlation_matrix, lab_results
                )

                # Promotion gate
                decision, fail_reasons = self.governance.evaluate_promotion_gate(
                    sc, sc.regime_scores
                )
                sc.promotion_decision = decision
                sc.fail_reasons = fail_reasons

                # Failure diagnostics
                self._diagnose_failures(sc)

                # Confidence score
                sc.confidence_score = _safe_round(
                    0.3 * sc.robustness_score
                    + 0.3 * sc.bootstrap_p_positive
                    + 0.2 * sc.stability_score
                    + 0.2 * (1.0 if sc.total_trades >= 100 else sc.total_trades / 100.0)
                )

                scorecards.append(sc)

            except Exception as exc:
                log.warning("Scorecard build failed for %s: %s", name, exc)
                scorecards.append(MethodologyScorecard(
                    name=name,
                    fail_reasons=[f"scorecard_build_error: {exc}"],
                    promotion_decision="REJECTED",
                ))

        # Rank by net Sharpe descending
        scorecards.sort(key=lambda s: s.net_sharpe, reverse=True)
        for i, sc in enumerate(scorecards):
            sc.final_rank = i + 1

        return scorecards

    def _classify_strategy(
        self,
        sc: MethodologyScorecard,
        correlation_matrix: Dict[str, Any],
        lab_results: Dict[str, Any],
    ) -> str:
        """
        Classify: CORE / DIVERSIFIER / EXPERIMENTAL / REDUNDANT / DISABLE.

        Based on: correlation to others, incremental alpha, robustness, regime fitness.
        """
        # DISABLE: negative net Sharpe or very poor metrics
        if sc.net_sharpe < 0 or sc.total_trades < 20:
            return "DISABLE"

        # Check correlation to other strategies
        corr_matrix = correlation_matrix.get("correlation_matrix", {})
        avg_corr_to_others = 0.3  # default
        if sc.name in corr_matrix:
            others_corr = [
                abs(v) for k, v in corr_matrix[sc.name].items()
                if k != sc.name and math.isfinite(v)
            ]
            if others_corr:
                avg_corr_to_others = float(np.mean(others_corr))

        # REDUNDANT: high correlation to better strategies
        if avg_corr_to_others > 0.7 and sc.net_sharpe < 0.5:
            return "REDUNDANT"

        # CORE: high robustness + strong Sharpe + diverse regime performance
        if (sc.robustness_score > 0.6 and sc.net_sharpe > 0.5
                and sc.diversification_score > 0.4 and sc.total_trades >= 100):
            return "CORE"

        # DIVERSIFIER: low correlation, adds diversification value
        if avg_corr_to_others < 0.3 and sc.net_sharpe > 0.1:
            return "DIVERSIFIER"

        # EXPERIMENTAL: everything else with some alpha
        return "EXPERIMENTAL"

    def _diagnose_failures(self, sc: MethodologyScorecard) -> None:
        """
        Root-cause analysis: assign failure_tags, strengths, weaknesses, actions.
        """
        strengths = []
        weaknesses = []
        failure_tags = []
        actions = []

        # Strengths
        if sc.net_sharpe > 0.5:
            strengths.append("strong_risk_adjusted_return")
        if sc.hit_rate > 0.55:
            strengths.append("high_hit_rate")
        if sc.stability_score > 0.6:
            strengths.append("walk_forward_stable")
        if sc.diversification_score > 0.5:
            strengths.append("regime_diversified")
        if sc.cost_drag_pct < 10:
            strengths.append("cost_efficient")
        if sc.bootstrap_p_positive > 0.85:
            strengths.append("statistically_significant")

        # Weaknesses / failure tags
        if sc.net_sharpe < 0.1:
            weaknesses.append("weak_alpha_net_of_costs")
            failure_tags.append("weak_signal")
            actions.append("review_signal_generation_logic")

        if sc.stability_score < 0.3:
            weaknesses.append("unstable_across_periods")
            failure_tags.append("low_robustness")
            actions.append("increase_min_train_period")

        if abs(sc.max_drawdown) > 0.10:
            weaknesses.append("excessive_drawdown")
            failure_tags.append("excessive_dd")
            actions.append("reduce_position_sizing_or_add_stops")

        if sc.cost_drag_pct > 30:
            weaknesses.append("high_cost_drag")
            failure_tags.append("cost_drag")
            actions.append("reduce_turnover_or_hold_longer")

        if sc.total_trades < 50:
            weaknesses.append("insufficient_sample")
            failure_tags.append("insufficient_sample")
            actions.append("extend_backtest_period_or_relax_entry")

        positive_regimes = sum(1 for v in sc.regime_scores.values() if v > 0)
        if positive_regimes <= 1:
            weaknesses.append("single_regime_dependency")
            failure_tags.append("single_regime")
            actions.append("add_regime_adaptive_parameters")

        if sc.skewness < -1.0:
            weaknesses.append("negative_skew_tails")
            failure_tags.append("poor_tail")
            actions.append("add_tail_risk_overlay")

        if sc.deflated_sharpe < 0:
            weaknesses.append("likely_overfit")
            failure_tags.append("low_robustness")
            actions.append("reduce_parameter_count_or_regularize")

        if sc.hit_rate < 0.48:
            weaknesses.append("below_breakeven_hit_rate")
            failure_tags.append("weak_signal")

        sc.strengths = strengths
        sc.weaknesses = weaknesses
        sc.failure_tags = list(set(failure_tags))
        sc.action_candidates = actions


# =============================================================================
# ReportAssembler
# =============================================================================

class ReportAssembler:
    """
    Assembles the final JSON report with all institutional sections.
    """

    @staticmethod
    def assemble(
        governance: Dict[str, Any],
        validation_suite: Dict[str, Any],
        scorecards: List[MethodologyScorecard],
        regime_attribution: Dict[str, Any],
        overfitting_control: Dict[str, Any],
        cost_analysis: Dict[str, Any],
        correlation_data: Dict[str, Any],
        base_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build the full institutional report."""

        # Scorecard dicts
        scorecard_dicts = [sc.to_dict() for sc in scorecards]

        # Approval matrix
        approval_matrix = {}
        for sc in scorecards:
            approval_matrix[sc.name] = {
                "decision": sc.promotion_decision,
                "classification": sc.classification,
                "net_sharpe": sc.net_sharpe,
                "rank": sc.final_rank,
                "fail_reasons": sc.fail_reasons,
            }

        # Regime fitness map
        regime_fitness = {}
        ms = regime_attribution.get("methodology_scores", {})
        for name, scores in ms.items():
            regime_fitness[name] = scores.get("recommended_action", {})

        # Ensemble analysis: identify core + diversifier strategies
        core = [sc for sc in scorecards if sc.classification == "CORE"]
        diversifiers = [sc for sc in scorecards if sc.classification == "DIVERSIFIER"]
        ensemble = {
            "core_strategies": [sc.name for sc in core],
            "diversifier_strategies": [sc.name for sc in diversifiers],
            "n_approved": sum(1 for sc in scorecards if sc.promotion_decision == "APPROVED"),
            "n_conditional": sum(1 for sc in scorecards if sc.promotion_decision == "CONDITIONAL"),
            "n_rejected": sum(1 for sc in scorecards if sc.promotion_decision == "REJECTED"),
            "recommended_ensemble_sharpe": _safe_round(
                float(np.mean([sc.net_sharpe for sc in core])) if core else 0.0
            ),
        }

        # Robustness summary
        robustness_summary = {
            "mean_robustness": _safe_round(
                float(np.mean([sc.robustness_score for sc in scorecards])) if scorecards else 0.0
            ),
            "mean_stability": _safe_round(
                float(np.mean([sc.stability_score for sc in scorecards])) if scorecards else 0.0
            ),
            "mean_deflated_sharpe": _safe_round(
                float(np.mean([sc.deflated_sharpe for sc in scorecards])) if scorecards else 0.0
            ),
            "most_robust": scorecards[0].name if scorecards else "N/A",
            "least_robust": scorecards[-1].name if scorecards else "N/A",
        }
        # Sort by robustness for most/least
        by_robust = sorted(scorecards, key=lambda s: s.robustness_score, reverse=True)
        if by_robust:
            robustness_summary["most_robust"] = by_robust[0].name
            robustness_summary["least_robust"] = by_robust[-1].name

        # PM summary (human-readable)
        best = scorecards[0] if scorecards else None
        pm_summary = {
            "best_strategy": best.name if best else "N/A",
            "best_net_sharpe": best.net_sharpe if best else 0.0,
            "best_classification": best.classification if best else "N/A",
            "best_promotion": best.promotion_decision if best else "REJECTED",
            "total_strategies_analyzed": len(scorecards),
            "strategies_approved": sum(1 for sc in scorecards if sc.promotion_decision == "APPROVED"),
            "strategies_conditional": sum(1 for sc in scorecards if sc.promotion_decision == "CONDITIONAL"),
            "strategies_rejected": sum(1 for sc in scorecards if sc.promotion_decision == "REJECTED"),
        }

        # Machine summary — stable contract for downstream agents
        from agents.methodology.report_schema import MachineSummary, REPORT_SCHEMA_VERSION
        best_sc = scorecards[0] if scorecards else None
        approved_list = [sc for sc in scorecards if sc.promotion_decision == "APPROVED"]
        conditional_list = [sc for sc in scorecards if sc.promotion_decision == "CONDITIONAL"]
        rejected_list = [sc for sc in scorecards if sc.promotion_decision == "REJECTED"]
        disable_list = [sc.name for sc in scorecards if sc.classification == "DISABLE"]
        observe_list = [sc.name for sc in scorecards if sc.classification == "EXPERIMENTAL"]

        ms_obj = MachineSummary(
            schema_version=REPORT_SCHEMA_VERSION,
            methodology_version=METHODOLOGY_VERSION,
            experiment_id=governance.get("experiment_id", ""),
            timestamp=governance.get("timestamp", ""),
            n_strategies=len(scorecards),
            n_approved=len(approved_list),
            n_conditional=len(conditional_list),
            n_rejected=len(rejected_list),
            best_strategy_name=best_sc.name if best_sc else "",
            best_strategy_decision=best_sc.promotion_decision if best_sc else "REJECTED",
            best_net_sharpe=best_sc.net_sharpe if best_sc else 0.0,
            best_gross_sharpe=best_sc.gross_sharpe if best_sc else 0.0,
            best_deflated_sharpe=best_sc.deflated_sharpe if best_sc else 0.0,
            best_hit_rate=best_sc.hit_rate if best_sc else 0.0,
            best_max_drawdown=best_sc.max_drawdown if best_sc else 0.0,
            best_total_trades=best_sc.total_trades if best_sc else 0,
            best_classification=best_sc.classification if best_sc else "DISABLE",
            current_regime=regime_attribution.get("current_regime", "UNKNOWN"),
            best_per_regime=regime_attribution.get("best_per_regime", {}),
            validation_complete=bool(validation_suite),
            governance_complete=bool(governance),
            overfitting_flag=any(sc.deflated_sharpe < 0 for sc in scorecards),
            cost_drag_flag=any(sc.cost_drag_pct > 30 for sc in scorecards),
            mean_robustness=_safe_round(float(np.mean([sc.robustness_score for sc in scorecards]))) if scorecards else 0.0,
            mean_stability=_safe_round(float(np.mean([sc.stability_score for sc in scorecards]))) if scorecards else 0.0,
            final_rankings={sc.name: sc.final_rank for sc in scorecards},
            optimizer_should_tune=bool(conditional_list) and not bool(approved_list),
            strategies_to_disable=disable_list,
            strategies_to_observe=observe_list,
        )
        machine_summary = ms_obj.to_dict()

        # Merge into base report
        base_report["governance"] = governance
        base_report["validation_suite"] = validation_suite
        base_report["methodology_scorecards"] = scorecard_dicts
        base_report["approval_matrix"] = approval_matrix
        base_report["regime_fitness_map"] = regime_fitness
        base_report["overfitting_control"] = overfitting_control
        base_report["cost_analysis"] = cost_analysis
        base_report["attribution"] = regime_attribution
        base_report["ensemble_analysis"] = ensemble
        base_report["robustness_summary"] = robustness_summary
        base_report["promotion_gate"] = {
            sc.name: {
                "decision": sc.promotion_decision,
                "fail_reasons": sc.fail_reasons,
            }
            for sc in scorecards
        }
        base_report["pm_summary"] = pm_summary
        base_report["machine_summary"] = machine_summary

        # Preserve existing correlation data
        if correlation_data and "error" not in correlation_data:
            base_report["strategy_correlations"] = correlation_data

        return base_report


# =============================================================================
# MethodologyAgent — main class
# =============================================================================

class MethodologyAgent:
    """
    Institutional-grade methodology analysis engine.

    Orchestrates: governance, walk-forward validation, block bootstrap,
    cost analysis, overfitting controls, regime attribution, strategy
    classification, failure diagnostics, and GPT advisory.
    """

    def __init__(self, settings: Settings = None, engine=None):
        try:
            self.settings = settings or get_settings()
            self.engine = engine
            self._lab = None
            self._lab_results = None
            log.info("MethodologyAgent initialized (v%s)", METHODOLOGY_VERSION)
        except Exception as exc:
            log.error("MethodologyAgent init failed: %s", exc)
            self.settings = get_settings()
            self.engine = None

    # ── Lazy lab access ──────────────────────────────────────────────────────

    def _get_lab(self):
        """Lazy-load MethodologyLab with engine prices."""
        try:
            if self._lab is None and _IMPORTS_OK.get("methodology_lab"):
                prices = getattr(self.engine, "prices", None)
                if prices is not None and not prices.empty:
                    self._lab = MethodologyLab(prices, self.settings, step=10)
            return self._lab
        except Exception as exc:
            log.warning("Failed to create MethodologyLab: %s", exc)
            return None

    def _get_lab_results(self):
        """Run all methodologies and cache results."""
        try:
            if self._lab_results is None:
                lab = self._get_lab()
                if lab is not None:
                    self._lab_results = lab.run_all()
            return self._lab_results or {}
        except Exception as exc:
            log.warning("Failed to run lab: %s", exc)
            return {}

    # ── Strategy correlations ────────────────────────────────────────────────

    def compute_strategy_correlations(self) -> Dict[str, Any]:
        """Compute correlation matrix between strategy P&L series."""
        try:
            import pandas as pd

            results = self._get_lab_results()
            if not results or len(results) < 2:
                return {"error": "need at least 2 strategy results"}

            pnl_dict = {}
            for name, result in results.items():
                try:
                    eq = result.equity_curve
                    if eq is not None and len(eq) > 5:
                        pnl_dict[name] = eq.pct_change().dropna()
                except Exception:
                    continue

            if len(pnl_dict) < 2:
                return {"error": "insufficient equity curves"}

            pnl_df = pd.DataFrame(pnl_dict).fillna(0.0)
            if len(pnl_df) < 10:
                return {"error": "insufficient overlapping data"}

            corr_matrix = pnl_df.corr()

            uncorrelated = []
            highly_correlated = []
            names = list(corr_matrix.columns)

            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    c = float(corr_matrix.loc[names[i], names[j]])
                    if abs(c) < 0.3:
                        uncorrelated.append({"pair": [names[i], names[j]], "correlation": _safe_round(c)})
                    elif abs(c) > 0.7:
                        highly_correlated.append({"pair": [names[i], names[j]], "correlation": _safe_round(c)})

            mean_abs_corr = corr_matrix.abs().mean().sort_values()
            best_diversifiers = [
                {"strategy": n, "mean_abs_corr": _safe_round(float(v))}
                for n, v in mean_abs_corr.head(5).items()
            ]

            corr_dict = {
                s1: {s2: _safe_round(float(corr_matrix.loc[s1, s2])) for s2 in names}
                for s1 in names
            }

            return {
                "correlation_matrix": corr_dict,
                "n_strategies": len(names),
                "uncorrelated_pairs": uncorrelated[:20],
                "highly_correlated": highly_correlated[:20],
                "best_diversifiers": best_diversifiers,
            }

        except Exception as exc:
            log.exception("Strategy correlation analysis failed")
            return {"error": str(exc)}

    # ── GPT advisory (advisory only, cannot modify decisions) ────────────────

    def gpt_advisory(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """
        GPT consultation for advisory commentary.
        Deterministic report is finalized BEFORE GPT receives it.
        GPT output stored under advisory_* fields only.
        """
        try:
            from agents.shared.gpt_conversation import GPTConversation

            gpt = GPTConversation(system_role="senior quant PM")
            if not gpt.available:
                return {"advisory_error": "GPT not available"}

            pm = report.get("pm_summary", {})
            scorecards = report.get("methodology_scorecards", [])
            top3 = scorecards[:3] if scorecards else []

            prompt = (
                f"METHODOLOGY REVIEW SUMMARY:\n"
                f"Best strategy: {pm.get('best_strategy', 'N/A')} "
                f"(net Sharpe={pm.get('best_net_sharpe', 0):.3f}, "
                f"classification={pm.get('best_classification', 'N/A')})\n"
                f"Approved: {pm.get('strategies_approved', 0)}, "
                f"Conditional: {pm.get('strategies_conditional', 0)}, "
                f"Rejected: {pm.get('strategies_rejected', 0)}\n\n"
                f"TOP 3 SCORECARDS:\n"
            )
            for sc_dict in top3:
                prompt += (
                    f"  {sc_dict.get('name', '?')}: net_sharpe={sc_dict.get('net_sharpe', 0):.3f}, "
                    f"robustness={sc_dict.get('robustness_score', 0):.2f}, "
                    f"classification={sc_dict.get('classification', '?')}, "
                    f"decision={sc_dict.get('promotion_decision', '?')}\n"
                    f"    weaknesses: {sc_dict.get('weaknesses', [])}\n"
                )

            prompt += (
                "\nProvide EXACTLY 3 specific actionable improvements. "
                "Focus on the highest-impact change to improve overall portfolio Sharpe. "
                "Be quantitatively specific. Max 200 words."
            )

            gpt.system_prompt = (
                "You are a senior quant PM. Review strategy scorecards and give "
                "SPECIFIC, ACTIONABLE recommendations. Every recommendation must "
                "include what to change, direction, and expected impact."
            )

            response = gpt.query(prompt)
            if not response:
                return {"advisory_error": "empty GPT response"}

            recommendations = []
            for line in response.strip().split("\n"):
                line = line.strip()
                if line and any(line.startswith(f"{i}") for i in range(1, 10)):
                    recommendations.append(line)

            return {
                "advisory_raw_response": response,
                "advisory_recommendations": recommendations[:3],
                "advisory_note": "GPT output is advisory only. Promotion decisions are deterministic.",
            }

        except Exception as exc:
            log.debug("GPT advisory failed: %s", exc)
            return {"advisory_error": str(exc)}

    # ── Full professional analysis orchestration ─────────────────────────────

    def run_professional_analysis(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the complete institutional validation and scoring pipeline.

        Steps:
        1. Governance record
        2. Run all methodologies via lab
        3. Walk-forward validation for ALL strategies
        4. Block bootstrap Monte Carlo for ALL strategies
        5. Cost-aware evaluation for ALL strategies
        6. Strategy correlations
        7. Overfitting controls
        8. Regime attribution
        9. Build scorecards
        10. Assemble report
        11. GPT advisory (after deterministic report is finalized)
        """
        try:
            log.info("Running institutional validation pipeline (v%s)...", METHODOLOGY_VERSION)
            prices = getattr(self.engine, "prices", None)

            # 1. Governance
            log.info("  [INST 1/11] Governance record...")
            gov_engine = GovernanceEngine(self.settings)
            governance = gov_engine.create_governance_record()

            # 2. Run lab
            log.info("  [INST 2/11] Running methodology lab...")
            lab_results = self._get_lab_results()
            if not lab_results:
                governance["validation_status"] = "FAILED"
                governance["fail_reasons"].append("no methodology results")
                report["governance"] = governance
                return report

            # Resolve methodology objects by name
            method_map = {}
            if _IMPORTS_OK.get("methodology_lab"):
                for m in ALL_METHODOLOGIES:
                    method_map[m.name] = m

            # 3. Walk-forward for all strategies
            log.info("  [INST 3/11] Walk-forward validation (all strategies)...")
            validation = ValidationSuite(self.settings)
            wf_results: Dict[str, Dict[str, Any]] = {}
            for name in lab_results:
                target = method_map.get(name)
                if target and prices is not None:
                    wf = validation.run_walk_forward(name, prices, target)
                    wf_results[name] = wf
                else:
                    wf_results[name] = {"error": "method or prices not available"}

            # 4. Block bootstrap for all strategies
            log.info("  [INST 4/11] Block bootstrap Monte Carlo (all strategies)...")
            bootstrap_results: Dict[str, Dict[str, Any]] = {}
            for name, result in lab_results.items():
                eq = getattr(result, "equity_curve", None)
                bs = validation.run_block_bootstrap(name, eq)
                bootstrap_results[name] = bs

            # 5. Cost-aware evaluation
            log.info("  [INST 5/11] Cost-aware evaluation (all strategies)...")
            cost_results: Dict[str, Dict[str, Any]] = {}
            for name, result in lab_results.items():
                cost = validation.compute_cost_aware_metrics(name, result)
                cost_results[name] = cost

            # 6. Strategy correlations
            log.info("  [INST 6/11] Strategy correlation matrix...")
            correlations = self.compute_strategy_correlations()

            # 7. Overfitting controls
            log.info("  [INST 7/11] Overfitting controls...")
            overfitting = OverfittingControl()
            overfitting_summary = {
                "n_strategies_tested": len(lab_results),
                "per_strategy": {},
            }
            for name, result in lab_results.items():
                wf = wf_results.get(name, {})
                wf_agg = wf.get("aggregate", {})
                is_sharpe = _sf(result.sharpe)
                oos_sharpe = _sf(wf_agg.get("mean_sharpe", 0))
                ranking_stab = overfitting.ranking_stability(wf, wf_results)
                overfit_prob = overfitting.overfitting_probability(
                    is_sharpe, oos_sharpe, len(lab_results)
                )
                overfitting_summary["per_strategy"][name] = {
                    "is_sharpe": _safe_round(is_sharpe),
                    "oos_sharpe": _safe_round(oos_sharpe),
                    "ranking_stability": _safe_round(ranking_stab),
                    "overfitting_probability": _safe_round(overfit_prob),
                }

            # 8. Regime attribution
            log.info("  [INST 8/11] Regime attribution...")
            regime_engine = RegimeAttributionEngine()
            regime_attribution = regime_engine.compute_regime_attribution(lab_results)

            # 9. Build scorecards
            log.info("  [INST 9/11] Building methodology scorecards...")
            scorer = MethodologyScorer(
                self.settings, gov_engine, validation, overfitting, regime_engine
            )
            scorecards = scorer.build_scorecards(
                lab_results, prices, wf_results, bootstrap_results,
                cost_results, regime_attribution, correlations,
            )

            # Update governance
            n_approved = sum(1 for sc in scorecards if sc.promotion_decision == "APPROVED")
            governance["validation_status"] = "COMPLETE"
            governance["promotion_readiness"] = (
                "READY" if n_approved > 0 else "NOT_READY"
            )

            # 10. Assemble report
            log.info("  [INST 10/11] Assembling institutional report...")
            validation_suite_data = {
                "walk_forward": {k: v for k, v in wf_results.items()},
                "bootstrap": {k: v for k, v in bootstrap_results.items()},
            }
            assembler = ReportAssembler()
            report = assembler.assemble(
                governance=governance,
                validation_suite=validation_suite_data,
                scorecards=scorecards,
                regime_attribution=regime_attribution,
                overfitting_control=overfitting_summary,
                cost_analysis=cost_results,
                correlation_data=correlations,
                base_report=report,
            )

            # 11. GPT advisory (AFTER deterministic report)
            log.info("  [INST 11/11] GPT advisory (post-finalization)...")
            gpt_advisory = self.gpt_advisory(report)
            report["advisory_gpt"] = gpt_advisory

            log.info("Institutional validation pipeline complete: %d strategies, %d approved",
                     len(scorecards), n_approved)

            return report

        except Exception as exc:
            log.exception("Institutional analysis pipeline failed")
            report["professional_analysis_error"] = str(exc)
            return report


# =============================================================================
# run() entrypoint — backward compatible
# =============================================================================

def run(once: bool = False) -> dict:
    """
    Run the methodology agent: full pipeline, backtest, institutional validation.

    Parameters
    ----------
    once : bool
        True = single run (no loop).

    Returns
    -------
    dict — Full institutional methodology report.
    """
    started_at = datetime.now(timezone.utc)
    today = date.today().isoformat()
    settings = get_settings()
    portfolio = MethodologyPortfolio()

    # Registry
    if _IMPORTS_OK.get("registry"):
        registry = get_registry()
        registry.register("agent_methodology", role="methodology evaluation & benchmarking")
        registry.heartbeat("agent_methodology", AgentStatus.RUNNING)

    log.info("=" * 70)
    log.info("Methodology Agent v%s — %s", METHODOLOGY_VERSION, today)
    log.info("=" * 70)

    report: Dict[str, Any] = {
        "run_date": today,
        "started_at": started_at.isoformat(),
        "methodology_version": METHODOLOGY_VERSION,
        "metrics": {},
        "regime_breakdown": {},
        "signal_stack": {},
        "stress_summary": {},
        "conclusions": [],
        "recommendations": [],
        "errors": [],
        "parameters_snapshot": {},
    }

    # ── Step 1: Full Pipeline ────────────────────────────────────────────────
    pipeline_result = None
    if _IMPORTS_OK.get("run_pipeline"):
        log.info("[1/8] Running full pipeline (with backtest)...")
        try:
            pipeline_result = run_pipeline(
                force_refresh=False,
                run_backtest=True,
                run_ml=True,
                run_optimizer=True,
            )
            log.info(
                "  Pipeline: steps_ok=%s, steps_failed=%s",
                pipeline_result.get("steps_ok", []),
                pipeline_result.get("steps_failed", []),
            )
        except Exception as exc:
            log.exception("Pipeline failed")
            report["errors"].append(f"Pipeline: {exc}")
    else:
        log.warning("[1/8] run_pipeline not available — skipping")
        report["errors"].append("run_pipeline import failed")

    # ── Step 2: QuantEngine → master_df ──────────────────────────────────────
    master_df = None
    engine = None
    if _IMPORTS_OK.get("QuantEngine"):
        log.info("[2/8] Running QuantEngine...")
        try:
            engine = QuantEngine(settings)
            engine.load()
            master_df = engine.calculate_conviction_score()
            n_sectors = len(master_df)
            log.info("  QuantEngine: %d sectors loaded", n_sectors)

            report["parameters_snapshot"] = {
                "pca_window": settings.pca_window,
                "zscore_window": settings.zscore_window,
                "n_sectors": n_sectors,
                "regime": str(getattr(engine, "regime_label", "UNKNOWN")),
            }
        except Exception as exc:
            log.exception("QuantEngine failed")
            report["errors"].append(f"QuantEngine: {exc}")
    else:
        log.warning("[2/8] QuantEngine not available — skipping")
        report["errors"].append("QuantEngine import failed")

    # ── Step 3: Walk-Forward Backtest ────────────────────────────────────────
    backtest_result: Optional[Any] = None
    if _IMPORTS_OK.get("backtest") and engine is not None:
        log.info("[3/8] Running walk-forward backtest...")
        try:
            prices_df = getattr(engine, "prices", None)
            fundamentals_df = getattr(engine, "fundamentals", None)
            weights_df = getattr(engine, "weights", None)

            if prices_df is not None and not prices_df.empty:
                backtest_result = run_wf_backtest(
                    prices_df=prices_df,
                    fundamentals_df=fundamentals_df if fundamentals_df is not None else prices_df.iloc[:0],
                    weights_df=weights_df if weights_df is not None else prices_df.iloc[:0],
                    settings=settings,
                )

                report["metrics"] = {
                    "ic_mean": round(_sf(backtest_result.ic_mean), 4),
                    "ic_ir": round(_sf(backtest_result.ic_ir), 4),
                    "sharpe": round(_sf(backtest_result.sharpe), 4),
                    "hit_rate": round(_sf(backtest_result.hit_rate), 4),
                    "max_dd": round(_sf(backtest_result.max_drawdown), 4),
                    "n_walks": backtest_result.n_walks,
                }

                report["regime_breakdown"] = _regime_breakdown_to_dict(
                    backtest_result.regime_breakdown
                )

                log.info(
                    "  Backtest: IC=%.4f, Sharpe=%.2f, hit=%.1f%%, DD=%.1f%%",
                    _sf(backtest_result.ic_mean),
                    _sf(backtest_result.sharpe),
                    _sf(backtest_result.hit_rate) * 100,
                    _sf(backtest_result.max_drawdown) * 100,
                )
            else:
                report["errors"].append("Backtest: prices_df not available from engine")
        except Exception as exc:
            log.exception("Backtest failed")
            report["errors"].append(f"Backtest: {exc}")
    else:
        log.warning("[3/8] Backtest not available — skipping")
        if not _IMPORTS_OK.get("backtest"):
            report["errors"].append("backtest import failed")

    # ── Step 3.5: Dispersion Backtest ────────────────────────────────────────
    if engine is not None and getattr(engine, "prices", None) is not None:
        log.info("[3.5/8] Running dispersion backtest...")
        try:
            from analytics.dispersion_backtest import DispersionBacktester
            disp_bt = DispersionBacktester(
                engine.prices, hold_period=15, z_entry=0.6, z_exit=0.2,
                max_positions=3, lookback=30,
            )
            disp_result = disp_bt.run()
            report["dispersion_backtest"] = {
                "sharpe": disp_result.sharpe,
                "win_rate": disp_result.win_rate,
                "total_pnl": disp_result.total_pnl,
                "max_drawdown": disp_result.max_drawdown,
                "total_trades": disp_result.total_trades,
                "pnl_by_regime": disp_result.pnl_by_regime,
            }
            log.info(
                "  Dispersion: Sharpe=%.2f, WR=%.1f%%, P&L=%.2f%%",
                disp_result.sharpe,
                disp_result.win_rate * 100,
                disp_result.total_pnl * 100,
            )
        except Exception as e:
            log.warning("  Dispersion backtest failed: %s", e)
            report["errors"].append(f"Dispersion Backtest: {e}")
    else:
        log.warning("[3.5/8] Dispersion backtest skipped — no engine/prices")

    # ── Step 4: Signal Stack ─────────────────────────────────────────────────
    signals: List[Any] = []
    if _IMPORTS_OK.get("signal_stack") and master_df is not None:
        log.info("[4/8] Computing Signal Stack metrics...")
        try:
            stack_engine = SignalStackEngine(settings)
            frob_z = _sf(getattr(engine, "distortion_z", None), 0.0) if engine else 0.0
            mode_share = _sf(getattr(engine, "market_mode_strength", None), 0.3) if engine else 0.3
            coc_z = _sf(getattr(engine, "coc_instability_z", None), 0.0) if engine else 0.0

            signals = stack_engine.score_from_master_df(
                frob_distortion_z=frob_z,
                market_mode_share=mode_share,
                coc_instability_z=coc_z,
                master_df=master_df,
            )

            summary = signal_stack_summary(signals)
            report["signal_stack"] = {
                "n_total": len(signals),
                "n_passing": sum(1 for s in signals if s.passes_entry),
                "top_conviction": round(signals[0].conviction_score, 4) if signals else 0,
                "distortion_score": round(signals[0].distortion_score, 4) if signals else 0,
                "top_candidates": summary.get("top_candidates", [])[:5],
            }

            log.info(
                "  Signal Stack: %d total, %d passing, top=%.4f",
                len(signals),
                report["signal_stack"]["n_passing"],
                report["signal_stack"]["top_conviction"],
            )
        except Exception as exc:
            log.exception("Signal Stack failed")
            report["errors"].append(f"Signal Stack: {exc}")
    else:
        log.warning("[4/8] Signal Stack not available — skipping")

    # ── Step 5: Stress Tests ─────────────────────────────────────────────────
    if _IMPORTS_OK.get("stress") and master_df is not None:
        log.info("[5/8] Running Stress Tests...")
        try:
            stress_results = StressEngine().run_all(master_df, settings)
            if stress_results:
                worst = stress_results[0]
                report["stress_summary"] = {
                    "n_scenarios": len(stress_results),
                    "worst_scenario": worst.scenario_name,
                    "worst_pnl_pct": round(_sf(worst.portfolio_pnl_estimate) * 100, 2),
                    "worst_reliability": round(_sf(getattr(worst, "signal_reliability_score", 0)), 4),
                }
                log.info(
                    "  Stress: worst=%s (%.1f%%)",
                    worst.scenario_name,
                    _sf(worst.portfolio_pnl_estimate) * 100,
                )
        except Exception as exc:
            log.exception("Stress Tests failed")
            report["errors"].append(f"Stress: {exc}")
    else:
        log.warning("[5/8] Stress Tests not available — skipping")

    # ── Step 6: Tail Risk ────────────────────────────────────────────────────
    if _IMPORTS_OK.get("tail_risk") and engine is not None and getattr(engine, "prices", None) is not None:
        log.info("[6/8] Computing tail risk metrics...")
        try:
            log_rets = np.log(engine.prices / engine.prices.shift(1)).dropna()
            _sectors_tr = settings.sector_list()
            eq_weights = {s: 1.0 / len(_sectors_tr) for s in _sectors_tr if s in log_rets.columns}

            es = compute_expected_shortfall(log_rets, eq_weights, confidence=0.975)
            corr_stress = parametric_correlation_stress(log_rets, eq_weights)
            tail_diag = tail_correlation_diagnostic(log_rets, _sectors_tr)

            report["tail_risk"] = {
                "es_97_5_1d": round(es.es_pct, 6),
                "var_97_5_1d": round(es.var_pct, 6),
                "es_var_ratio": round(es.es_to_var_ratio, 4),
                "skewness": es.skewness,
                "kurtosis": es.kurtosis,
                "corr_stress_panic_vol_increase": round(corr_stress[-1].vol_increase_pct, 4) if corr_stress else 0,
                "corr_stress_panic_disp_change": round(corr_stress[-1].dispersion_change_pct, 4) if corr_stress else 0,
                "tail_corr": tail_diag.get("tail_corr", 0),
                "tail_ratio": tail_diag.get("tail_ratio", 1.0),
                "panic_coupling": tail_diag.get("panic_coupling", False),
            }
            log.info(
                "  ES: %.4f%%, ES/VaR=%.2f, skew=%.2f, kurt=%.1f, tail_ratio=%.2f",
                es.es_pct * 100, es.es_to_var_ratio, es.skewness, es.kurtosis,
                tail_diag.get("tail_ratio", 1.0),
            )
        except Exception as exc:
            log.exception("Tail risk computation failed")
            report["errors"].append(f"Tail Risk: {exc}")
    else:
        log.warning("[6/8] Tail risk not available — skipping")

    # ── Step 7: Institutional Validation Pipeline ────────────────────────────
    log.info("[7/8] Running institutional validation pipeline...")
    try:
        pro_agent = MethodologyAgent(settings=settings, engine=engine)
        pro_agent.run_professional_analysis(report)
        log.info("  Institutional analysis: OK")
    except Exception as exc:
        log.warning("Institutional analysis failed: %s", exc)
        report["errors"].append(f"Institutional Analysis: {exc}")

    # ── Step 8: Conclusions & Recommendations ────────────────────────────────
    log.info("[8/8] Generating conclusions and recommendations...")
    try:
        _generate_conclusions(report, portfolio)
    except Exception as exc:
        log.exception("Conclusion generation failed")
        report["errors"].append(f"Conclusions: {exc}")

    # ── Finalize ─────────────────────────────────────────────────────────────
    finished_at = datetime.now(timezone.utc)
    report["finished_at"] = finished_at.isoformat()
    report["elapsed_seconds"] = round(
        (finished_at - started_at).total_seconds(), 1
    )

    # ── Legacy GPT consultation ──────────────────────────────────────────────
    try:
        from agents.shared.gpt_conversation import GPTConversation

        _m = report.get("metrics", {})
        _prev_runs = portfolio.get_history(1)
        _prev_sharpe = "N/A"
        if _prev_runs:
            _prev_sharpe = _prev_runs[-1].get("metrics", {}).get("sharpe", "N/A")

        _regime_label = report.get("parameters_snapshot", {}).get("regime", "UNKNOWN")
        _smart = report.get("smart_conclusions", [])
        _top_issue = _smart[0].get("finding", "none") if _smart else "none"
        _top_action = _smart[0].get("action", "none") if _smart else "none"

        gpt_system = (
            f"You are a senior quant PM reviewing daily strategy performance.\n"
            f"Current metrics: Sharpe={_m.get('sharpe', 'N/A')}, "
            f"IC={_m.get('ic_mean', 'N/A')}, IC_IR={_m.get('ic_ir', 'N/A')}, "
            f"HitRate={_m.get('hit_rate', 'N/A')}, MaxDD={_m.get('max_dd', 'N/A')}.\n"
            f"Previous Sharpe={_prev_sharpe}. Current regime={_regime_label}.\n"
            f"Top issue: {_top_issue}\n"
            f"Suggested action: {_top_action}\n\n"
            f"Focus: What is the SINGLE most impactful change to improve Sharpe by 0.1+?\n"
            f"Be specific: name the parameter, the direction of change, and the exact value.\n"
            f"Max 150 words per response."
        )

        gpt = GPTConversation(system_role="senior quant PM")
        gpt.system_prompt = gpt_system

        if gpt.available and report.get("metrics"):
            log.info("Consulting GPT for focused PM-grade analysis...")
            gpt_results = gpt.full_analysis(
                metrics={
                    "sharpe": _m.get("sharpe", 0),
                    "ic": _m.get("ic_mean", 0),
                    "ic_ir": _m.get("ic_ir", 0),
                    "win_rate": _m.get("hit_rate", 0),
                    "max_dd": _m.get("max_dd", 0),
                    "trades": _m.get("n_walks", 0),
                    "regime_breakdown": report.get("regime_breakdown", {}),
                    "smart_conclusions": [
                        {"priority": sc.get("priority"), "finding": sc.get("finding"),
                         "action": sc.get("action")}
                        for sc in _smart[:3]
                    ],
                },
                params={
                    "pca_window": settings.pca_window,
                    "zscore_window": settings.zscore_window,
                    "signal_entry_threshold": settings.signal_entry_threshold,
                    "signal_a1_frob": settings.signal_a1_frob,
                    "trade_max_holding_days": settings.trade_max_holding_days,
                    "mr_weight_half_life": getattr(settings, "mr_weight_half_life", "N/A"),
                    "mr_weight_adf": getattr(settings, "mr_weight_adf", "N/A"),
                    "mr_weight_hurst": getattr(settings, "mr_weight_hurst", "N/A"),
                },
            )
            report["gpt_analysis"] = gpt_results
            report["gpt_conversation"] = gpt.get_conversation_log()
            log.info("GPT diagnosis: %s", gpt_results.get("diagnosis", "")[:100])
    except Exception as exc:
        log.debug("GPT consultation failed: %s", exc)

    # ── Save JSON report ─────────────────────────────────────────────────────
    report_path = REPORTS_DIR / f"{today}.json"
    try:
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        log.info("Report saved: %s", report_path)
    except Exception as exc:
        log.exception("Failed to save report")

    # ── Publish to AgentBus ──────────────────────────────────────────────────
    if _IMPORTS_OK.get("agent_bus"):
        try:
            bus = get_bus()
            bus_payload = {
                "status": "ok" if not report["errors"] else "partial",
                "run_date": today,
                "metrics": report["metrics"],
                "signal_stack_passing": report.get("signal_stack", {}).get("n_passing", 0),
                "signal_stack_top": report.get("signal_stack", {}).get("top_conviction", 0),
                "conclusions_count": len(report["conclusions"]),
                "errors_count": len(report["errors"]),
                "governance": report.get("governance", {}),
                "pm_summary": report.get("pm_summary", {}),
                "approval_matrix": report.get("approval_matrix", {}),
                "monte_carlo": report.get("validation_suite", {}).get("bootstrap", {}),
                "regime_conditional_ic": report.get("attribution", {}),
                "walk_forward_validation": report.get("validation_suite", {}).get("walk_forward", {}),
                "strategy_correlations_summary": {
                    "n_uncorrelated": len(report.get("strategy_correlations", {}).get("uncorrelated_pairs", [])),
                    "best_diversifiers": report.get("strategy_correlations", {}).get("best_diversifiers", []),
                },
            }
            bus.publish("agent_methodology", bus_payload)
            log.info("Published to AgentBus")
        except Exception as exc:
            log.exception("AgentBus publish failed")

    # ── Update portfolio tracker ─────────────────────────────────────────────
    try:
        portfolio.add_run(report)
        log.info("Portfolio tracker updated")
    except Exception as exc:
        log.exception("Portfolio tracker update failed")

    # ── Registry — finish ────────────────────────────────────────────────────
    if _IMPORTS_OK.get("registry"):
        status = AgentStatus.COMPLETED if not report["errors"] else AgentStatus.FAILED
        registry = get_registry()
        registry.heartbeat("agent_methodology", status,
                           error="; ".join(report["errors"][:3]) if report["errors"] else None)

    log.info(
        "Methodology Agent completed in %.1fs — %d conclusions, %d errors",
        report["elapsed_seconds"],
        len(report["conclusions"]),
        len(report["errors"]),
    )
    return report


# =============================================================================
# Conclusions engine
# =============================================================================

def _generate_conclusions(report: dict, portfolio: MethodologyPortfolio) -> None:
    """
    Generate automated conclusions from current run vs history.
    Adds to report["conclusions"], report["recommendations"],
    and report["smart_conclusions"].
    """
    conclusions: List[str] = []
    recommendations: List[str] = []

    metrics = report.get("metrics", {})

    # PM-grade smart conclusions
    smart_conclusions = _generate_smart_conclusions(metrics, report, portfolio)
    report["smart_conclusions"] = smart_conclusions

    for sc in smart_conclusions:
        priority = sc.get("priority", "INFO")
        finding = sc.get("finding", "")
        action = sc.get("action", "")
        conclusions.append(f"[{priority}] {finding}")
        if action:
            expected = sc.get("expected_impact", "")
            rec = f"{action}"
            if expected:
                rec += f" (expected: {expected})"
            recommendations.append(rec)

    # ── Comparison vs previous run ───────────────────────────────────────────
    comparison = portfolio.compare_vs_previous(report)
    if comparison.get("has_previous"):
        deltas = comparison.get("deltas", {})

        ic_delta = deltas.get("ic_mean", {})
        if ic_delta:
            old_ic = ic_delta["old"]
            new_ic = ic_delta["new"]
            conclusions.append(
                f"IC changed from {old_ic:.4f} to {new_ic:.4f} ({_pct_delta(old_ic, new_ic)})"
            )

        sharpe_delta = deltas.get("sharpe", {})
        if sharpe_delta:
            old_s = sharpe_delta["old"]
            new_s = sharpe_delta["new"]
            conclusions.append(
                f"Sharpe changed from {old_s:.2f} to {new_s:.2f} ({_pct_delta(old_s, new_s)})"
            )
            if new_s < 0.5 and old_s >= 0.5:
                recommendations.append(
                    "Sharpe dropped below 0.5 — consider reducing position sizes"
                )

        hr_delta = deltas.get("hit_rate", {})
        if hr_delta:
            old_hr = hr_delta["old"]
            new_hr = hr_delta["new"]
            conclusions.append(
                f"Hit rate changed from {old_hr:.1%} to {new_hr:.1%} ({_pct_delta(old_hr, new_hr)})"
            )
            if new_hr < 0.50:
                recommendations.append(
                    "Hit rate below 50% — signal quality degraded, review entry thresholds"
                )

        dd_delta = deltas.get("max_dd", {})
        if dd_delta:
            old_dd = dd_delta["old"]
            new_dd = dd_delta["new"]
            if new_dd > old_dd * 1.25:
                conclusions.append(
                    f"Max drawdown increased from {old_dd:.1%} to {new_dd:.1%} — risk elevated"
                )
                recommendations.append(
                    "Drawdown expanding — tighten stop-losses and review regime gates"
                )

    # ── Institutional scorecard conclusions ───────────────────────────────────
    scorecards = report.get("methodology_scorecards", [])
    if scorecards:
        approved = [sc for sc in scorecards if sc.get("promotion_decision") == "APPROVED"]
        rejected = [sc for sc in scorecards if sc.get("promotion_decision") == "REJECTED"]
        if approved:
            conclusions.append(
                f"{len(approved)} strategies APPROVED for promotion: "
                + ", ".join(sc.get("name", "?") for sc in approved[:5])
            )
        if len(rejected) == len(scorecards):
            conclusions.append("ALL strategies REJECTED — system needs recalibration")
            recommendations.append("Review entry thresholds, regime gates, and cost structure")

    # ── Regime analysis ──────────────────────────────────────────────────────
    regime_breakdown = report.get("regime_breakdown", {})
    for regime, data in regime_breakdown.items():
        if isinstance(data, dict):
            hr = data.get("hit_rate", 1.0)
            if hr < 0.45:
                conclusions.append(
                    f"{regime} regime hit_rate={hr:.1%} — below threshold"
                )

    # ── Long-term trends ─────────────────────────────────────────────────────
    for metric_name in ("ic_mean", "sharpe", "hit_rate"):
        trend = portfolio.get_trend(metric_name, n=10)
        direction = trend.get("direction", "insufficient_data")
        if direction == "declining":
            delta_pct = trend.get("delta_pct", 0)
            conclusions.append(
                f"{metric_name} declining over last 10 runs ({delta_pct:+.1f}%)"
            )
            recommendations.append(
                f"Persistent {metric_name} decline — investigate parameter drift or regime shift"
            )
        elif direction == "improving":
            delta_pct = trend.get("delta_pct", 0)
            conclusions.append(
                f"{metric_name} improving over last 10 runs ({delta_pct:+.1f}%)"
            )

    # ── Historical weak regime ───────────────────────────────────────────────
    worst = portfolio.worst_regime()
    if worst and worst.get("consecutive_below_50", 0) >= 3:
        regime = worst["regime"]
        consec = worst["consecutive_below_50"]
        avg_hr = worst["avg_hit_rate"]
        conclusions.append(
            f"{regime} regime consistently weakest: {consec} consecutive runs below 50% "
            f"(avg hit_rate={avg_hr:.1%})"
        )
        recommendations.append(
            f"Consider regime-specific parameter tuning for {regime} or disabling entries"
        )

    # ── Signal Stack ─────────────────────────────────────────────────────────
    signal_stack = report.get("signal_stack", {})
    n_passing = signal_stack.get("n_passing", 0)
    n_total = signal_stack.get("n_total", 0)
    if n_total > 0 and n_passing == 0:
        conclusions.append("No signals passing entry threshold — all gates blocked")
        recommendations.append("Review regime safety score and entry threshold calibration")
    elif n_total > 0:
        pass_rate = n_passing / n_total
        if pass_rate > 0.7:
            conclusions.append(
                f"High signal pass rate ({pass_rate:.0%}) — possible threshold too loose"
            )
            recommendations.append("Consider tightening entry_threshold to improve selectivity")

    # ── Stress ───────────────────────────────────────────────────────────────
    stress = report.get("stress_summary", {})
    worst_pnl = stress.get("worst_pnl_pct", 0)
    if worst_pnl < -5.0:
        worst_name = stress.get("worst_scenario", "?")
        conclusions.append(
            f"Worst stress scenario '{worst_name}' shows {worst_pnl:.1f}% loss"
        )
        recommendations.append("Stress loss exceeds 5% — review tail hedging or position sizing")

    # ── Dispersion backtest ──────────────────────────────────────────────────
    disp = report.get("dispersion_backtest", {})
    if disp:
        disp_sharpe = disp.get("sharpe", 0)
        disp_wr = disp.get("win_rate", 0)
        if disp_sharpe < 0.5:
            conclusions.append(f"Dispersion Sharpe={disp_sharpe:.2f} below target (0.5)")
        if disp_wr < 0.5:
            conclusions.append(f"Dispersion win rate={disp_wr:.1%} below breakeven")
        pnl_by_regime = disp.get("pnl_by_regime", {})
        for reg, pnl in pnl_by_regime.items():
            if isinstance(pnl, (int, float)) and pnl < 0:
                conclusions.append(f"Dispersion loses money in {reg} regime (P&L={pnl:+.2%})")

    # ── Write to report ──────────────────────────────────────────────────────
    if not conclusions:
        conclusions.append("No significant changes detected — methodology performing within norms")

    report["conclusions"] = conclusions
    report["recommendations"] = recommendations


def _generate_smart_conclusions(
    metrics: dict, report: dict, portfolio: MethodologyPortfolio
) -> List[dict]:
    """
    Generate PM-grade conclusions with root cause analysis, specific actions,
    and expected impact. Returns structured list sorted by priority.
    """
    conclusions: List[dict] = []

    sharpe = _sf(metrics.get("sharpe", 0))
    ic = _sf(metrics.get("ic_mean", 0))
    hit_rate = _sf(metrics.get("hit_rate", 0))
    max_dd = _sf(metrics.get("max_dd", 0))
    ic_ir = _sf(metrics.get("ic_ir", 0))

    # 1. Sharpe decomposition
    if sharpe < 0.3:
        if ic < 0.02:
            root = f"IC={ic:.4f} too low — signals barely predictive"
            action = "Run Bayesian optimizer on z-thresholds; consider expanding MR whitelist"
        elif hit_rate < 0.5:
            root = f"Hit rate={hit_rate:.1%} below breakeven"
            action = "Tighten MR whitelist; raise signal_entry_threshold by 0.05"
        elif abs(max_dd) > 0.08:
            root = f"MaxDD={max_dd:.1%} — gains wiped by drawdowns"
            action = "Reduce max_leverage by 20%; activate dd_deleverage at 2%"
        else:
            root = "Low return per unit risk from multiple small inefficiencies"
            action = "Focus on improving IC in weakest regime first"
        conclusions.append({
            "priority": "CRITICAL",
            "finding": f"Sharpe {sharpe:.3f} below minimum viable threshold (0.3)",
            "root_cause": root,
            "action": action,
            "expected_impact": "Sharpe +0.1 to +0.3",
        })
    elif sharpe < 0.8:
        conclusions.append({
            "priority": "WARNING",
            "finding": f"Sharpe {sharpe:.3f} acceptable but below target (0.8)",
            "root_cause": "Room for alpha improvement",
            "action": "Optimize weakest regime parameters; review signal weights",
            "expected_impact": "Sharpe +0.05 to +0.15",
        })

    # 2. Regime-specific
    regime_breakdown = report.get("regime_breakdown", {})
    for regime, rm in regime_breakdown.items():
        if not isinstance(rm, dict):
            continue
        reg_sharpe = _sf(rm.get("sharpe", 0))
        reg_ic = _sf(rm.get("ic_mean", 0))
        reg_hr = _sf(rm.get("hit_rate", 0))

        if reg_sharpe < 0:
            conclusions.append({
                "priority": "HIGH",
                "finding": f"Negative Sharpe in {regime} regime ({reg_sharpe:.3f})",
                "root_cause": f"IC={reg_ic:.4f}, HR={reg_hr:.1%} in {regime}",
                "action": (
                    f"Reduce regime_conviction_scale for {regime.lower()}; "
                    f"raise z-threshold; consider disabling entries in {regime}"
                ),
                "expected_impact": f"Eliminate {regime} losses; improve overall Sharpe",
            })
        elif reg_hr < 0.48:
            conclusions.append({
                "priority": "HIGH",
                "finding": f"{regime} hit rate={reg_hr:.1%} below breakeven",
                "root_cause": f"Signals not predictive in {regime} conditions",
                "action": f"Add regime-specific entry filter for {regime}",
                "expected_impact": f"Improve {regime} hit rate to 52%+",
            })

    # 3. IC quality
    if ic < 0.01:
        conclusions.append({
            "priority": "HIGH",
            "finding": f"IC={ic:.4f} — signals barely predictive (need >0.02)",
            "root_cause": "Scoring function may not capture true alpha signal",
            "action": "Add momentum filter or expand MR whitelist; review distortion weights",
            "expected_impact": "IC improvement to 0.02-0.03",
        })
    elif ic > 0 and ic_ir < 0.3:
        conclusions.append({
            "priority": "WARNING",
            "finding": f"IC={ic:.4f} but IC_IR={ic_ir:.4f} — signal unstable across time",
            "root_cause": "IC varies significantly between walks",
            "action": "Increase zscore_window to smooth signal; review walk-forward stability",
            "expected_impact": "IC_IR improvement to 0.3+",
        })

    # 4. Drawdown
    if abs(max_dd) > 0.05:
        conclusions.append({
            "priority": "HIGH",
            "finding": f"Max drawdown {max_dd:.1%} exceeds 5% limit",
            "root_cause": "Insufficient deleveraging or position sizing too aggressive",
            "action": "Activate dd_deleverage at 2% drawdown; reduce max_leverage",
            "expected_impact": "Cap drawdowns at 3-5%",
        })

    # 5. vs previous run
    prev_runs = portfolio.get_history(1)
    if prev_runs:
        prev_metrics = prev_runs[-1].get("metrics", {})
        prev_sharpe = _sf(prev_metrics.get("sharpe", 0))
        delta = sharpe - prev_sharpe
        if delta < -0.1:
            conclusions.append({
                "priority": "WARNING",
                "finding": f"Sharpe declined {delta:+.3f} vs previous run ({prev_sharpe:.3f} -> {sharpe:.3f})",
                "root_cause": "Possible parameter drift, data quality issue, or regime shift",
                "action": "Check for data staleness; compare parameters vs best_run snapshot",
                "expected_impact": "Restore Sharpe to previous level",
            })
        elif delta > 0.15:
            conclusions.append({
                "priority": "INFO",
                "finding": f"Sharpe improved {delta:+.3f} vs previous run",
                "root_cause": "Recent optimization or favorable regime shift",
                "action": "Snapshot current parameters as new baseline",
                "expected_impact": "Preserve gains",
            })

    # 6. Dispersion cross-check
    disp = report.get("dispersion_backtest", {})
    if disp:
        disp_sharpe = _sf(disp.get("sharpe", 0))
        if disp_sharpe > 2.0 and sharpe < 0.5:
            conclusions.append({
                "priority": "HIGH",
                "finding": (
                    f"Dispersion Sharpe={disp_sharpe:.2f} but WF Sharpe={sharpe:.2f} — "
                    "signal stack not translating dispersion alpha"
                ),
                "root_cause": "Signal stack layers may be filtering out good dispersion trades",
                "action": "Review Layer 4 regime safety gating; lower safety thresholds",
                "expected_impact": "Better pass-through of dispersion alpha",
            })

    # 7. Tail risk
    tail = report.get("tail_risk", {})
    if tail:
        es_var = _sf(tail.get("es_var_ratio", 1.0))
        if es_var > 1.8:
            conclusions.append({
                "priority": "WARNING",
                "finding": f"ES/VaR ratio={es_var:.2f} — fat tails detected",
                "root_cause": "Return distribution has heavier tails than normal",
                "action": "Increase stress buffer; use ES instead of VaR for sizing",
                "expected_impact": "Better tail risk protection",
            })
        if tail.get("panic_coupling", False):
            conclusions.append({
                "priority": "HIGH",
                "finding": "Panic coupling detected — correlations spike in stress",
                "root_cause": "Diversification fails when most needed",
                "action": "Reduce max_positions during high-vol regime; add correlation stress overlay",
                "expected_impact": "Lower tail losses by 20-40%",
            })

    # 8. Institutional scorecard insights
    scorecards = report.get("methodology_scorecards", [])
    if scorecards:
        top_sc = scorecards[0] if scorecards else {}
        overfit = report.get("overfitting_control", {}).get("per_strategy", {})
        for name, of_data in overfit.items():
            prob = _sf(of_data.get("overfitting_probability", 0))
            if prob > 0.7:
                conclusions.append({
                    "priority": "HIGH",
                    "finding": f"{name}: overfitting probability={prob:.0%}",
                    "root_cause": "Large IS vs OOS Sharpe gap with many strategies tested",
                    "action": f"Simplify {name} parameters or increase regularization",
                    "expected_impact": "More robust OOS performance",
                })

    # Sort by priority
    priority_order = {"CRITICAL": 0, "HIGH": 1, "WARNING": 2, "INFO": 3}
    conclusions.sort(key=lambda c: priority_order.get(c.get("priority", "INFO"), 9))

    return conclusions


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SRV Methodology Agent (Institutional)")
    parser.add_argument(
        "--once",
        action="store_true",
        default=True,
        help="Run once (default behavior)",
    )
    args = parser.parse_args()

    result = run(once=args.once)

    # Quick stdout summary
    print("\n" + "=" * 60)
    print(f"Methodology Report v{METHODOLOGY_VERSION} — {result['run_date']}")
    print("=" * 60)

    m = result.get("metrics", {})
    if m:
        print(f"  IC Mean:     {m.get('ic_mean', 'N/A')}")
        print(f"  IC IR:       {m.get('ic_ir', 'N/A')}")
        print(f"  Sharpe:      {m.get('sharpe', 'N/A')}")
        print(f"  Hit Rate:    {m.get('hit_rate', 'N/A')}")
        print(f"  Max DD:      {m.get('max_dd', 'N/A')}")

    pm = result.get("pm_summary", {})
    if pm:
        print(f"\n  Best Strategy: {pm.get('best_strategy', 'N/A')} "
              f"(net Sharpe={pm.get('best_net_sharpe', 0):.3f})")
        print(f"  Approved: {pm.get('strategies_approved', 0)}, "
              f"Conditional: {pm.get('strategies_conditional', 0)}, "
              f"Rejected: {pm.get('strategies_rejected', 0)}")

    ss = result.get("signal_stack", {})
    if ss:
        print(f"\n  Signals:     {ss.get('n_passing', 0)}/{ss.get('n_total', 0)} passing")
        print(f"  Top Conv:    {ss.get('top_conviction', 0)}")

    if result.get("conclusions"):
        print("\n  Conclusions:")
        for c in result["conclusions"][:10]:
            print(f"    - {c}")

    if result.get("recommendations"):
        print("\n  Recommendations:")
        for r in result["recommendations"][:10]:
            print(f"    - {r}")

    if result.get("errors"):
        print(f"\n  Errors ({len(result['errors'])}):")
        for e in result["errors"]:
            print(f"    ! {e}")

    print(f"\n  Elapsed: {result.get('elapsed_seconds', 0)}s")

    sys.exit(0)
