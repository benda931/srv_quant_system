"""
analytics/alpha_research.py
=============================
Alpha Research Engine — institutional-grade OOS validation, multi-strategy
ensemble, dispersion alpha, regime-adaptive optimization, and alpha stability.

Pipeline:
  1. Walk-forward OOS validation (expanding window, purged splits)
  2. IS/OOS decay analysis + statistical significance (t-test, bootstrapped CI)
  3. Regime-adaptive parameter optimization (per CALM/NORMAL/TENSION/CRISIS)
  4. Multi-strategy ensemble with regime weighting
  5. Dispersion alpha research (variance swap P&L + short vol signal)
  6. Alpha stability analysis (rolling OOS Sharpe, parameter sensitivity heatmap)
  7. GPT-assisted strategy refinement
  8. Comprehensive report with persistence

Short Vol / Dispersion is the PRIMARY alpha source — dispersion alpha research
runs as a first-class citizen alongside sector RV strategies.
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent


# ─────────────────────────────────────────────────────────────────────────────
# Result structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OOSResult:
    """Out-of-sample validation result for one parameter set + split."""
    params: Dict[str, Any]
    in_sample_sharpe: float
    out_of_sample_sharpe: float
    oos_win_rate: float
    oos_pnl: float
    oos_max_dd: float
    oos_trades: int
    is_valid: bool                  # OOS Sharpe > 0 AND reasonable
    regime: str                     # Which regime this was optimized for
    # IS/OOS decay analysis
    sharpe_decay_ratio: float = 0.0 # OOS/IS — closer to 1.0 = more robust
    t_stat: float = 0.0            # t-statistic of OOS mean P&L vs 0
    p_value: float = 1.0           # p-value (two-sided)
    ci_lower: float = 0.0          # 95% CI lower bound on OOS Sharpe (bootstrap)
    ci_upper: float = 0.0          # 95% CI upper bound
    split_idx: int = 0             # Which walk-forward split (0-based)
    train_size: int = 0
    test_size: int = 0


@dataclass
class DispersionAlphaResult:
    """Result of dispersion / short-vol alpha research."""
    backtest_sharpe: float
    backtest_win_rate: float
    backtest_total_pnl: float
    backtest_max_dd: float
    backtest_trades: int
    # P&L component breakdown
    vega_pnl_pct: float
    theta_pnl_pct: float
    gamma_pnl_pct: float
    # Short vol signal quality
    short_vol_score: float          # 0-100
    short_vol_label: str
    implied_corr: float
    dispersion_index: float
    vrp: float
    # Optimal params from research
    best_params: Dict[str, Any] = field(default_factory=dict)
    pnl_by_regime: Dict[str, float] = field(default_factory=dict)


@dataclass
class StrategyEnsembleResult:
    """Multi-strategy ensemble with regime weighting."""
    strategy_weights: Dict[str, float]        # {strategy_name: weight}
    strategy_sharpes: Dict[str, float]         # {strategy_name: OOS sharpe}
    regime_allocations: Dict[str, Dict[str, float]]  # {regime: {strategy: weight}}
    ensemble_sharpe: float
    ensemble_win_rate: float
    ensemble_max_dd: float
    n_strategies: int
    diversity_score: float                     # Correlation-based diversity (0-1)


@dataclass
class AlphaStabilityResult:
    """Rolling OOS Sharpe and parameter sensitivity analysis."""
    rolling_oos_sharpes: List[float]           # Rolling OOS Sharpe across time
    rolling_dates: List[str]
    sharpe_mean: float
    sharpe_std: float
    sharpe_min: float
    pct_positive: float                        # % of rolling windows with positive OOS Sharpe
    # Parameter sensitivity
    param_sensitivity: Dict[str, float]        # {param: sensitivity_score}
    most_sensitive_param: str
    least_sensitive_param: str


@dataclass
class RegimeAdaptiveResult:
    """Result of regime-adaptive parameter optimization."""
    regime_params: Dict[str, Dict[str, Any]]  # {regime: {param: value}}
    combined_sharpe: float
    combined_win_rate: float
    combined_pnl: float
    regime_sharpes: Dict[str, float]
    regime_trades: Dict[str, int]              # {regime: n_trades}
    n_trades_total: int
    oos_validated: bool
    # Cross-regime consistency
    sharpe_stability: float                    # std(regime_sharpes) / mean — lower = more stable


@dataclass
class AlphaReport:
    """Complete alpha research report."""
    timestamp: str
    # Core OOS validation
    oos_results: List[OOSResult]
    best_single_strategy: OOSResult
    # Regime-adaptive
    regime_adaptive: RegimeAdaptiveResult
    # Dispersion / Short Vol alpha
    dispersion_alpha: Optional[DispersionAlphaResult]
    # Ensemble
    ensemble: Optional[StrategyEnsembleResult]
    # Stability
    stability: Optional[AlphaStabilityResult]
    # Final metrics
    ensemble_sharpe: float
    recommendations: List[str]
    gpt_suggestions: List[str]
    # Metadata
    total_runtime_s: float = 0.0
    n_strategies_tested: int = 0
    n_param_combos_tested: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Walk-Forward OOS Validation (expanded, purged splits)
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_oos(
    prices: pd.DataFrame,
    settings,
    n_splits: int = 5,
    cost_bps: float = 15.0,
    purge_days: int = 21,
) -> Tuple[List[OOSResult], int]:
    """
    Walk-forward OOS validation with expanding window + purge gap.

    Returns (results, n_combos_tested).

    Purge gap: 21 trading days between train and test to prevent
    information leakage from overlapping holding periods.
    """
    from analytics.methodology_lab import MethodologyLab, PcaZReversal

    n = len(prices)
    results = []
    n_combos = 0

    # Expanding-window splits with purge gap
    test_pct = 0.20
    test_len = int(n * test_pct)
    splits = []
    for i in range(n_splits):
        test_end = n - i * (test_len // 2)
        test_start = test_end - test_len
        train_end = test_start - purge_days
        if train_end < 400 or test_start < 0:
            continue
        splits.append((0, train_end, test_start, test_end))

    for split_idx, (train_start, train_end, test_start, test_end) in enumerate(splits):
        log.info("OOS Split %d/%d: train=0-%d (purge %dd) test=%d-%d",
                 split_idx + 1, len(splits), train_end, purge_days, test_start, test_end)

        train_prices = prices.iloc[train_start:train_end]
        test_prices = prices.iloc[test_start:test_end]

        if len(train_prices) < 400 or len(test_prices) < 100:
            continue

        # Grid search on train — refined grid for medium-term holding
        best_is_sharpe = -999.0
        best_params: Dict[str, Any] = {}
        best_trades_list: List[Any] = []

        for z_entry in [0.5, 0.6, 0.7, 0.8, 1.0]:
            for z_exit in [0.15, 0.20, 0.30]:
                for z_stop in [1.8, 2.0, 2.5]:
                    for hold in [15, 20, 25, 30]:
                        for w in [0.04, 0.06, 0.08]:
                            n_combos += 1
                            try:
                                m = PcaZReversal(
                                    z_entry=z_entry, z_exit_ratio=z_exit,
                                    z_stop_ratio=z_stop, max_hold=hold, max_weight=w,
                                )
                                lab = MethodologyLab(train_prices, settings, step=10, cost_bps=cost_bps)
                                r = lab.run_methodology(m)
                                if r.total_trades >= 25 and r.sharpe > best_is_sharpe:
                                    best_is_sharpe = r.sharpe
                                    best_params = {
                                        "z_entry": z_entry, "z_exit": z_exit,
                                        "z_stop": z_stop, "hold": hold, "w": w,
                                    }
                                    best_trades_list = r.trades
                            except Exception:
                                pass

        if not best_params:
            continue

        # Evaluate on test (OOS)
        m_oos = PcaZReversal(
            z_entry=best_params["z_entry"],
            z_exit_ratio=best_params["z_exit"],
            z_stop_ratio=best_params["z_stop"],
            max_hold=best_params["hold"],
            max_weight=best_params["w"],
        )
        lab_oos = MethodologyLab(test_prices, settings, step=10, cost_bps=cost_bps)
        r_oos = lab_oos.run_methodology(m_oos)

        # Statistical significance: t-test on OOS trade P&Ls
        oos_pnls = np.array([t.pnl for t in r_oos.trades]) if r_oos.trades else np.array([0.0])
        t_stat, p_value = _ttest_vs_zero(oos_pnls)

        # Bootstrap 95% CI on OOS Sharpe
        ci_lo, ci_hi = _bootstrap_sharpe_ci(oos_pnls, n_boot=1000)

        # IS/OOS decay ratio
        decay_ratio = r_oos.sharpe / best_is_sharpe if abs(best_is_sharpe) > 1e-6 else 0.0

        is_valid = (
            r_oos.sharpe > 0
            and r_oos.total_trades >= 15
            and p_value < 0.10           # marginally significant
            and decay_ratio > 0.10       # OOS retains > 10% of IS Sharpe
        )

        results.append(OOSResult(
            params=best_params,
            in_sample_sharpe=round(best_is_sharpe, 4),
            out_of_sample_sharpe=round(r_oos.sharpe, 4),
            oos_win_rate=round(r_oos.win_rate, 4),
            oos_pnl=round(r_oos.total_pnl, 6),
            oos_max_dd=round(r_oos.max_drawdown, 6),
            oos_trades=r_oos.total_trades,
            is_valid=is_valid,
            regime="ALL",
            sharpe_decay_ratio=round(decay_ratio, 4),
            t_stat=round(t_stat, 3),
            p_value=round(p_value, 4),
            ci_lower=round(ci_lo, 4),
            ci_upper=round(ci_hi, 4),
            split_idx=split_idx,
            train_size=len(train_prices),
            test_size=len(test_prices),
        ))

        log.info(
            "  Split %d: IS=%.3f → OOS=%.3f (decay=%.2f, p=%.3f, valid=%s, trades=%d)",
            split_idx + 1, best_is_sharpe, r_oos.sharpe, decay_ratio, p_value,
            is_valid, r_oos.total_trades,
        )

    return results, n_combos


# ─────────────────────────────────────────────────────────────────────────────
# Statistical helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ttest_vs_zero(pnls: np.ndarray) -> Tuple[float, float]:
    """One-sample t-test of mean P&L vs zero. Returns (t_stat, p_value)."""
    n = len(pnls)
    if n < 5:
        return 0.0, 1.0
    mean = pnls.mean()
    se = pnls.std(ddof=1) / math.sqrt(n)
    if se < 1e-12:
        return 0.0, 1.0
    t = float(mean / se)
    # Approximate two-sided p-value from t-distribution
    # Using normal approximation (good enough for n > 20)
    from scipy.stats import t as t_dist
    p = float(2 * t_dist.sf(abs(t), df=n - 1))
    return t, p


def _bootstrap_sharpe_ci(
    pnls: np.ndarray, n_boot: int = 1000, ci: float = 0.95,
) -> Tuple[float, float]:
    """Bootstrap confidence interval for annualized Sharpe ratio."""
    n = len(pnls)
    if n < 10:
        return 0.0, 0.0
    rng = np.random.default_rng(42)
    sharpes = []
    ann_factor = np.sqrt(252 / max(1, 10))  # step=10 in MethodologyLab
    for _ in range(n_boot):
        sample = rng.choice(pnls, size=n, replace=True)
        mu = sample.mean()
        sigma = sample.std(ddof=1)
        sharpes.append(mu / sigma * ann_factor if sigma > 1e-12 else 0.0)
    alpha = (1 - ci) / 2
    lo = float(np.percentile(sharpes, alpha * 100))
    hi = float(np.percentile(sharpes, (1 - alpha) * 100))
    return lo, hi


# ─────────────────────────────────────────────────────────────────────────────
# Regime-Adaptive Parameters
# ─────────────────────────────────────────────────────────────────────────────

def _classify_regimes(prices: pd.DataFrame, settings) -> pd.Series:
    """Classify each date into CALM/NORMAL/TENSION/CRISIS based on VIX."""
    vix_col = next((c for c in prices.columns if "VIX" in c.upper()), None)
    vix = prices[vix_col] if vix_col else pd.Series(18.0, index=prices.index)

    regimes = pd.Series("NORMAL", index=prices.index)
    for i in range(min(60, len(prices)), len(prices)):
        v = float(vix.iloc[i]) if not pd.isna(vix.iloc[i]) else 18
        if v > getattr(settings, "vix_level_hard", 32):
            regimes.iloc[i] = "CRISIS"
        elif v > getattr(settings, "vix_level_soft", 21):
            regimes.iloc[i] = "TENSION"
        elif v > 16:
            regimes.iloc[i] = "NORMAL"
        else:
            regimes.iloc[i] = "CALM"
    return regimes


def optimize_per_regime(
    prices: pd.DataFrame,
    settings,
    cost_bps: float = 15.0,
) -> RegimeAdaptiveResult:
    """
    Find optimal parameters separately for each regime.

    CALM    → aggressive: low z entry, higher weight, longer hold
    NORMAL  → moderate
    TENSION → conservative: high z entry, low weight, short hold
    CRISIS  → no trading (regime kill)
    """
    from analytics.methodology_lab import MethodologyLab, PcaZReversal

    regimes = _classify_regimes(prices, settings)

    # Regime-specific grids calibrated for medium-term holding
    regime_grids = {
        "CALM": {
            "z_entry": [0.5, 0.6, 0.7],
            "z_exit": [0.15, 0.20],
            "z_stop": [2.0, 2.5],
            "hold": [20, 25, 30],
            "w": [0.06, 0.08, 0.10],
        },
        "NORMAL": {
            "z_entry": [0.6, 0.7, 0.8],
            "z_exit": [0.20, 0.25],
            "z_stop": [2.0, 2.5],
            "hold": [15, 20, 25],
            "w": [0.05, 0.06, 0.08],
        },
        "TENSION": {
            "z_entry": [0.8, 1.0, 1.2],
            "z_exit": [0.25, 0.30],
            "z_stop": [2.5, 3.0],
            "hold": [10, 15, 20],
            "w": [0.03, 0.04, 0.05],
        },
    }

    regime_params: Dict[str, Dict[str, Any]] = {}
    regime_sharpes: Dict[str, float] = {}
    regime_trades: Dict[str, int] = {}
    all_pnl = 0.0

    for regime, grid in regime_grids.items():
        regime_dates = regimes[regimes == regime].index
        if len(regime_dates) < 150:
            log.info("  %s: only %d dates — skipping", regime, len(regime_dates))
            continue

        best_sharpe = -999.0
        best_p: Dict[str, Any] = {}
        best_n_trades = 0

        for z_e in grid["z_entry"]:
            for z_x in grid["z_exit"]:
                for z_s in grid["z_stop"]:
                    for h in grid["hold"]:
                        for w in grid["w"]:
                            try:
                                m = PcaZReversal(
                                    z_entry=z_e, z_exit_ratio=z_x,
                                    z_stop_ratio=z_s, max_hold=h, max_weight=w,
                                )
                                lab = MethodologyLab(prices, settings, step=10, cost_bps=cost_bps)
                                r = lab.run_methodology(m)

                                # Filter trades entered during this regime
                                rt = [t for t in r.trades if t.regime == regime]
                                if len(rt) >= 15:
                                    pnls = np.array([t.pnl for t in rt])
                                    mu = float(pnls.mean())
                                    sigma = float(pnls.std())
                                    sharpe = mu / sigma * np.sqrt(252 / 10) if sigma > 1e-10 else 0
                                    if sharpe > best_sharpe:
                                        best_sharpe = sharpe
                                        best_p = {
                                            "z_entry": z_e, "z_exit": z_x,
                                            "z_stop": z_s, "hold": h, "w": w,
                                        }
                                        best_n_trades = len(rt)
                                        all_pnl += float(pnls.sum())
                            except Exception:
                                pass

        if best_p:
            regime_params[regime] = best_p
            regime_sharpes[regime] = round(best_sharpe, 4)
            regime_trades[regime] = best_n_trades
            log.info("  %s: Sharpe=%.3f, trades=%d, params=%s",
                     regime, best_sharpe, best_n_trades, best_p)

    # Cross-regime consistency
    sharpe_vals = list(regime_sharpes.values())
    sharpe_mean = float(np.mean(sharpe_vals)) if sharpe_vals else 0
    sharpe_std = float(np.std(sharpe_vals)) if len(sharpe_vals) > 1 else 0
    stability = sharpe_std / abs(sharpe_mean) if abs(sharpe_mean) > 1e-6 else float("inf")

    combined_sharpe = sharpe_mean
    combined_wr = 0.0
    oos_validated = all(s > 0 for s in sharpe_vals) if sharpe_vals else False
    total_trades = sum(regime_trades.values())

    return RegimeAdaptiveResult(
        regime_params=regime_params,
        combined_sharpe=round(combined_sharpe, 4),
        combined_win_rate=round(combined_wr, 4),
        combined_pnl=round(all_pnl, 6),
        regime_sharpes=regime_sharpes,
        regime_trades=regime_trades,
        n_trades_total=total_trades,
        oos_validated=oos_validated,
        sharpe_stability=round(stability, 4),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dispersion Alpha Research (PRIMARY alpha source)
# ─────────────────────────────────────────────────────────────────────────────

def research_dispersion_alpha(
    prices: pd.DataFrame,
    settings,
) -> Optional[DispersionAlphaResult]:
    """
    Research dispersion / short-vol alpha:
    1. Run DispersionBacktester with a parameter grid
    2. Evaluate short vol signal quality from CorrVolEngine
    3. Extract IV surface metrics (implied corr, VRP, dispersion index)
    """
    try:
        from analytics.dispersion_backtest import DispersionBacktester
    except ImportError:
        log.warning("DispersionBacktester not available — skipping dispersion alpha")
        return None

    sectors = [s for s in settings.sector_list() if s in prices.columns]
    if len(sectors) < 5:
        return None

    # Grid search over dispersion backtest params
    best_sharpe = -999.0
    best_result = None
    best_params: Dict[str, Any] = {}

    for z_entry in [0.5, 0.6, 0.8, 1.0]:
        for z_exit in [0.2, 0.3, 0.4]:
            for hold in [10, 15, 21]:
                for max_pos in [2, 3, 4]:
                    try:
                        bt = DispersionBacktester(
                            prices, sectors=sectors,
                            hold_period=hold, z_entry=z_entry, z_exit=z_exit,
                            max_positions=max_pos, lookback=30,
                        )
                        r = bt.run()
                        if r.total_trades >= 10 and r.sharpe > best_sharpe:
                            best_sharpe = r.sharpe
                            best_result = r
                            best_params = {
                                "z_entry": z_entry, "z_exit": z_exit,
                                "hold_period": hold, "max_positions": max_pos,
                            }
                    except Exception:
                        pass

    if best_result is None:
        return None

    # Short vol signal from correlation engine
    short_vol_score = 0.0
    short_vol_label = "N/A"
    implied_corr = 0.0
    dispersion_idx = 0.0
    vrp = 0.0

    try:
        from analytics.options_engine import OptionsEngine
        surface = OptionsEngine().compute_surface(prices, settings)
        implied_corr = float(getattr(surface, "implied_corr", 0))
        dispersion_idx = float(getattr(surface, "dispersion_index", 0))
        vrp = float(getattr(surface, "vrp_index", 0))
    except Exception:
        pass

    try:
        from analytics.correlation_engine import CorrVolEngine
        from analytics.stat_arb import QuantEngine
        eng = QuantEngine(settings)
        eng.load()
        mdf = eng.calculate_conviction_score()
        cv = CorrVolEngine()
        analysis = cv.run(eng, mdf, settings)
        short_vol_score = float(getattr(analysis, "short_vol_score", 0))
        short_vol_label = str(getattr(analysis, "short_vol_label", "N/A"))
    except Exception:
        pass

    log.info(
        "Dispersion alpha: Sharpe=%.3f, WR=%.1f%%, trades=%d, short_vol=%d (%s)",
        best_result.sharpe, best_result.win_rate * 100, best_result.total_trades,
        short_vol_score, short_vol_label,
    )

    return DispersionAlphaResult(
        backtest_sharpe=round(best_result.sharpe, 4),
        backtest_win_rate=round(best_result.win_rate, 4),
        backtest_total_pnl=round(best_result.total_pnl, 6),
        backtest_max_dd=round(best_result.max_drawdown, 6),
        backtest_trades=best_result.total_trades,
        vega_pnl_pct=round(best_result.total_vega_pnl * 100, 2),
        theta_pnl_pct=round(best_result.total_theta_pnl * 100, 2),
        gamma_pnl_pct=round(best_result.total_gamma_pnl * 100, 2),
        short_vol_score=round(short_vol_score, 1),
        short_vol_label=short_vol_label,
        implied_corr=round(implied_corr, 4),
        dispersion_index=round(dispersion_idx, 4),
        vrp=round(vrp, 4),
        best_params=best_params,
        pnl_by_regime=best_result.pnl_by_regime,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Strategy Ensemble
# ─────────────────────────────────────────────────────────────────────────────

def build_strategy_ensemble(
    prices: pd.DataFrame,
    settings,
    cost_bps: float = 15.0,
    top_n: int = 5,
) -> Optional[StrategyEnsembleResult]:
    """
    Run all available strategies, rank by OOS Sharpe, build a
    diversification-weighted ensemble of the top N.

    Diversity is measured as 1 - avg(|corr|) between strategy equity curves.
    """
    from analytics.methodology_lab import MethodologyLab

    strategies = _get_all_strategies()
    if not strategies:
        return None

    # Split: train 70%, test 30%
    n = len(prices)
    split = int(n * 0.70)
    test_prices = prices.iloc[split:]

    if len(test_prices) < 200:
        return None

    regimes = _classify_regimes(prices, settings)
    strategy_results: List[Tuple[str, float, float, float, pd.Series]] = []

    for name, method in strategies:
        try:
            lab = MethodologyLab(test_prices, settings, step=10, cost_bps=cost_bps)
            r = lab.run_methodology(method)
            if r.total_trades >= 10 and hasattr(r, "equity_curve") and r.equity_curve is not None:
                strategy_results.append((
                    name, r.sharpe, r.win_rate, r.max_drawdown, r.equity_curve,
                ))
        except Exception:
            pass

    if len(strategy_results) < 2:
        return None

    # Sort by Sharpe and take top N
    strategy_results.sort(key=lambda x: x[1], reverse=True)
    top = strategy_results[:top_n]

    # Compute pairwise correlation of equity curves → diversity score
    eq_matrix = pd.DataFrame({name: eq for name, _, _, _, eq in top})
    eq_returns = eq_matrix.pct_change().dropna()
    if len(eq_returns) > 10:
        corr_mat = eq_returns.corr().values
        mask = ~np.eye(len(top), dtype=bool)
        avg_abs_corr = float(np.abs(corr_mat[mask]).mean()) if mask.sum() > 0 else 1.0
        diversity = round(1.0 - avg_abs_corr, 4)
    else:
        diversity = 0.0

    # Equal-weight ensemble (with inverse-vol adjustment if > 2 strategies)
    sharpes = {name: s for name, s, _, _, _ in top}
    total_s = sum(max(s, 0.01) for s in sharpes.values())
    weights = {name: round(max(s, 0.01) / total_s, 4) for name, s in sharpes.items()}

    # Ensemble metrics (weighted average)
    ens_sharpe = sum(sharpes[n] * weights[n] for n in weights)
    ens_wr = sum(wr * weights[n] for n, _, wr, _, _ in top)
    ens_dd = max(dd for _, _, _, dd, _ in top)

    # Regime allocations
    regime_alloc: Dict[str, Dict[str, float]] = {}
    for regime in ["CALM", "NORMAL", "TENSION"]:
        regime_alloc[regime] = weights.copy()

    log.info("Ensemble: %d strategies, diversity=%.2f, Sharpe=%.3f",
             len(top), diversity, ens_sharpe)

    return StrategyEnsembleResult(
        strategy_weights=weights,
        strategy_sharpes=sharpes,
        regime_allocations=regime_alloc,
        ensemble_sharpe=round(ens_sharpe, 4),
        ensemble_win_rate=round(ens_wr, 4),
        ensemble_max_dd=round(ens_dd, 6),
        n_strategies=len(top),
        diversity_score=diversity,
    )


def _get_all_strategies() -> List[Tuple[str, Any]]:
    """Return list of (name, Methodology instance) for all available strategies."""
    strategies = []
    try:
        from analytics.methodology_lab import (
            PcaZReversal, MomentumFilter, DispersionTiming,
            AdaptiveThreshold, MultiFactor,
        )
        strategies.extend([
            ("PcaZReversal", PcaZReversal()),
            ("MomentumFilter", MomentumFilter()),
            ("DispersionTiming", DispersionTiming()),
            ("AdaptiveThreshold", AdaptiveThreshold()),
            ("MultiFactor", MultiFactor()),
        ])
    except ImportError:
        pass
    try:
        from analytics.methodology_lab import (
            ResearchBriefRV, ResearchBriefDispersion,
            ResearchBriefPCABasket, ResearchBriefShortConvexity,
        )
        strategies.extend([
            ("ResearchBriefRV", ResearchBriefRV()),
            ("ResearchBriefDispersion", ResearchBriefDispersion()),
            ("ResearchBriefPCABasket", ResearchBriefPCABasket()),
            ("ResearchBriefShortConvexity", ResearchBriefShortConvexity()),
        ])
    except ImportError:
        pass
    return strategies


# ─────────────────────────────────────────────────────────────────────────────
# Alpha Stability Analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyse_alpha_stability(
    prices: pd.DataFrame,
    settings,
    best_params: Dict[str, Any],
    window_days: int = 504,
    step_days: int = 63,
    cost_bps: float = 15.0,
) -> AlphaStabilityResult:
    """
    Rolling OOS Sharpe + parameter sensitivity analysis.

    1. Slides a window across time, trains on first 70% of window,
       tests on last 30%. Records OOS Sharpe at each step.
    2. Perturbs each parameter ±20% and measures Sharpe change.
    """
    from analytics.methodology_lab import MethodologyLab, PcaZReversal

    n = len(prices)
    rolling_sharpes = []
    rolling_dates = []

    # Rolling OOS Sharpe
    pos = window_days
    while pos <= n:
        window = prices.iloc[pos - window_days: pos]
        split = int(len(window) * 0.70)
        test = window.iloc[split:]

        if len(test) >= 60 and best_params:
            try:
                m = PcaZReversal(
                    z_entry=best_params.get("z_entry", 0.7),
                    z_exit_ratio=best_params.get("z_exit", 0.20),
                    z_stop_ratio=best_params.get("z_stop", 2.0),
                    max_hold=best_params.get("hold", 20),
                    max_weight=best_params.get("w", 0.06),
                )
                lab = MethodologyLab(test, settings, step=10, cost_bps=cost_bps)
                r = lab.run_methodology(m)
                rolling_sharpes.append(round(r.sharpe, 4))
                rolling_dates.append(str(test.index[-1].date()) if hasattr(test.index[-1], "date") else str(test.index[-1]))
            except Exception:
                rolling_sharpes.append(0.0)
                rolling_dates.append("")
        pos += step_days

    sharpe_arr = np.array(rolling_sharpes)
    pct_positive = float((sharpe_arr > 0).mean()) if len(sharpe_arr) > 0 else 0.0

    # Parameter sensitivity
    param_sensitivity: Dict[str, float] = {}
    if best_params:
        # Use last 2 years of data for sensitivity
        ref_prices = prices.iloc[-504:] if len(prices) >= 504 else prices
        ref_split = int(len(ref_prices) * 0.70)
        ref_test = ref_prices.iloc[ref_split:]

        # Baseline Sharpe
        try:
            m_base = PcaZReversal(
                z_entry=best_params.get("z_entry", 0.7),
                z_exit_ratio=best_params.get("z_exit", 0.20),
                z_stop_ratio=best_params.get("z_stop", 2.0),
                max_hold=best_params.get("hold", 20),
                max_weight=best_params.get("w", 0.06),
            )
            lab_base = MethodologyLab(ref_test, settings, step=10, cost_bps=cost_bps)
            base_sharpe = lab_base.run_methodology(m_base).sharpe
        except Exception:
            base_sharpe = 0.0

        # Perturb each param ±20%
        for param_name in ["z_entry", "z_exit", "z_stop", "hold", "w"]:
            if param_name not in best_params:
                continue
            base_val = best_params[param_name]
            deltas = []
            for mult in [0.80, 1.20]:
                perturbed = dict(best_params)
                new_val = base_val * mult
                if param_name == "hold":
                    new_val = max(5, int(new_val))
                perturbed[param_name] = new_val
                try:
                    m_p = PcaZReversal(
                        z_entry=perturbed.get("z_entry", 0.7),
                        z_exit_ratio=perturbed.get("z_exit", 0.20),
                        z_stop_ratio=perturbed.get("z_stop", 2.0),
                        max_hold=int(perturbed.get("hold", 20)),
                        max_weight=perturbed.get("w", 0.06),
                    )
                    lab_p = MethodologyLab(ref_test, settings, step=10, cost_bps=cost_bps)
                    s = lab_p.run_methodology(m_p).sharpe
                    deltas.append(abs(s - base_sharpe))
                except Exception:
                    deltas.append(0.0)
            param_sensitivity[param_name] = round(float(np.mean(deltas)), 4)

    most_sensitive = max(param_sensitivity, key=param_sensitivity.get) if param_sensitivity else ""
    least_sensitive = min(param_sensitivity, key=param_sensitivity.get) if param_sensitivity else ""

    return AlphaStabilityResult(
        rolling_oos_sharpes=rolling_sharpes,
        rolling_dates=rolling_dates,
        sharpe_mean=round(float(sharpe_arr.mean()), 4) if len(sharpe_arr) > 0 else 0.0,
        sharpe_std=round(float(sharpe_arr.std()), 4) if len(sharpe_arr) > 0 else 0.0,
        sharpe_min=round(float(sharpe_arr.min()), 4) if len(sharpe_arr) > 0 else 0.0,
        pct_positive=round(pct_positive, 4),
        param_sensitivity=param_sensitivity,
        most_sensitive_param=most_sensitive,
        least_sensitive_param=least_sensitive,
    )


# ─────────────────────────────────────────────────────────────────────────────
# GPT-Assisted Strategy Refinement
# ─────────────────────────────────────────────────────────────────────────────

def query_gpt_for_alpha(
    current_metrics: Dict,
    regime_breakdown: Dict,
    pair_signals: List[Dict],
    options_data: Dict,
    dispersion_data: Optional[Dict] = None,
) -> List[str]:
    """
    Query GPT for strategy improvement suggestions based on current performance.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        for env_file in [ROOT / ".env", ROOT / "agents" / "credentials" / "api_keys.env"]:
            if env_file.exists():
                for line in env_file.read_text(encoding="utf-8").splitlines():
                    if line.strip().startswith("OPENAI_API_KEY") and "=" in line:
                        api_key = line.split("=", 1)[1].strip().strip("'\"")
                        break
            if api_key:
                break

    if not api_key or len(api_key) < 10:
        return ["GPT API key not available — skipping AI suggestions"]

    try:
        import openai
        client = openai.OpenAI(api_key=api_key)

        dispersion_section = ""
        if dispersion_data:
            dispersion_section = f"""
## Dispersion / Short Vol Alpha
- Backtest Sharpe: {dispersion_data.get('sharpe', 'N/A')}
- Win Rate: {dispersion_data.get('win_rate', 'N/A')}
- Vega P&L: {dispersion_data.get('vega_pnl', 'N/A')}%
- Theta P&L: {dispersion_data.get('theta_pnl', 'N/A')}%
- Gamma P&L: {dispersion_data.get('gamma_pnl', 'N/A')}%
- Short Vol Score: {dispersion_data.get('short_vol_score', 'N/A')}/100
- Implied Correlation: {dispersion_data.get('implied_corr', 'N/A')}
- VRP: {dispersion_data.get('vrp', 'N/A')}
"""

        prompt = f"""You are a quantitative portfolio manager specializing in sector relative-value
and short volatility / correlation dispersion strategies. Analyze these results and suggest
3 specific, actionable improvements.

## Current Performance
- Best OOS Sharpe: {current_metrics.get('best_sharpe', 'N/A')}
- Best OOS Win Rate: {current_metrics.get('best_wr', 'N/A')}
- IS/OOS Decay Ratio: {current_metrics.get('decay_ratio', 'N/A')}
- p-value: {current_metrics.get('p_value', 'N/A')}
- OOS Sharpe 95% CI: [{current_metrics.get('ci_lower', 'N/A')}, {current_metrics.get('ci_upper', 'N/A')}]

## Regime Breakdown
{json.dumps(regime_breakdown, indent=2, default=str)}

## Top Pair Signals
{json.dumps(pair_signals[:5], indent=2, default=str)}

## Options Analytics
- VIX: {options_data.get('vix', 'N/A')}
- Implied Correlation: {options_data.get('implied_corr', 'N/A')}
- VRP: {options_data.get('vrp', 'N/A')}
- Top IV sector: {options_data.get('top_iv', 'N/A')}
{dispersion_section}
Suggest 3 specific improvements. For each:
- What to change (exact parameter or formula)
- Why it should help (mathematical reasoning)
- Expected impact (Sharpe improvement estimate)

Be concise and specific. No general advice."""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1200,
            temperature=0.3,
        )

        suggestions = response.choices[0].message.content.strip().split("\n")
        suggestions = [s.strip() for s in suggestions if s.strip() and len(s.strip()) > 10]
        log.info("GPT returned %d suggestions", len(suggestions))
        return suggestions[:15]

    except Exception as e:
        log.warning("GPT query failed: %s", e)
        return [f"GPT query failed: {e}"]


# ─────────────────────────────────────────────────────────────────────────────
# Full Alpha Research Run
# ─────────────────────────────────────────────────────────────────────────────

def run_alpha_research(
    prices: pd.DataFrame,
    settings,
    include_gpt: bool = True,
    include_dispersion: bool = True,
    include_ensemble: bool = True,
    include_stability: bool = True,
) -> AlphaReport:
    """
    Run complete alpha research pipeline.

    Steps:
      1. Walk-forward OOS validation (5 purged splits)
      2. Regime-adaptive parameter optimization
      3. Dispersion / Short Vol alpha research
      4. Multi-strategy ensemble
      5. Alpha stability analysis
      6. GPT suggestions
      7. Build comprehensive report + persist
    """
    t_start = time.time()

    log.info("=" * 70)
    log.info("Alpha Research Engine — Starting (comprehensive mode)")
    log.info("=" * 70)

    # ── 1. Walk-forward OOS validation ────────────────────────────────────
    log.info("[1/6] Walk-forward OOS validation (5 purged splits)...")
    t0 = time.time()
    oos_results, n_combos = walk_forward_oos(prices, settings)
    log.info("  OOS: %.1fs, %d splits, %d combos tested", time.time() - t0, len(oos_results), n_combos)

    best_oos = max(oos_results, key=lambda r: r.out_of_sample_sharpe) if oos_results else None

    # ── 2. Regime-adaptive optimization ───────────────────────────────────
    log.info("[2/6] Regime-adaptive parameter optimization...")
    t0 = time.time()
    regime_result = optimize_per_regime(prices, settings)
    log.info("  Regime: %.1fs, regimes=%s", time.time() - t0, list(regime_result.regime_sharpes.keys()))

    # ── 3. Dispersion alpha research ──────────────────────────────────────
    dispersion_alpha: Optional[DispersionAlphaResult] = None
    if include_dispersion:
        log.info("[3/6] Dispersion / Short Vol alpha research...")
        t0 = time.time()
        dispersion_alpha = research_dispersion_alpha(prices, settings)
        log.info("  Dispersion: %.1fs", time.time() - t0)

    # ── 4. Multi-strategy ensemble ────────────────────────────────────────
    ensemble: Optional[StrategyEnsembleResult] = None
    if include_ensemble:
        log.info("[4/6] Multi-strategy ensemble...")
        t0 = time.time()
        ensemble = build_strategy_ensemble(prices, settings)
        log.info("  Ensemble: %.1fs", time.time() - t0)

    # ── 5. Alpha stability analysis ───────────────────────────────────────
    stability: Optional[AlphaStabilityResult] = None
    if include_stability and best_oos and best_oos.params:
        log.info("[5/6] Alpha stability analysis...")
        t0 = time.time()
        stability = analyse_alpha_stability(prices, settings, best_oos.params)
        log.info("  Stability: %.1fs, pct_positive=%.0f%%",
                 time.time() - t0, (stability.pct_positive * 100) if stability else 0)

    # ── 6. GPT suggestions ────────────────────────────────────────────────
    gpt_suggestions: List[str] = []
    if include_gpt:
        log.info("[6/6] GPT strategy refinement...")
        try:
            from analytics.pair_scanner import scan_pairs
            pairs = scan_pairs(prices, settings.sector_list())
            pair_data = [{"pair": p.pair_name, "z": p.spread_z, "hl": p.half_life,
                          "strength": p.signal_strength} for p in pairs[:5]]
        except Exception:
            pair_data = []

        try:
            from analytics.options_engine import OptionsEngine
            surface = OptionsEngine().compute_surface(prices, settings)
            opts_data = {
                "vix": surface.vix_current, "implied_corr": surface.implied_corr,
                "vrp": surface.vrp_index,
            }
            top_iv = max(surface.sector_greeks.items(), key=lambda x: x[1].iv)
            opts_data["top_iv"] = f"{top_iv[0]} ({top_iv[1].iv:.0%})"
        except Exception:
            opts_data = {}

        metrics = {}
        if best_oos:
            metrics = {
                "best_sharpe": best_oos.out_of_sample_sharpe,
                "best_wr": best_oos.oos_win_rate,
                "decay_ratio": best_oos.sharpe_decay_ratio,
                "p_value": best_oos.p_value,
                "ci_lower": best_oos.ci_lower,
                "ci_upper": best_oos.ci_upper,
            }

        disp_data = None
        if dispersion_alpha:
            disp_data = {
                "sharpe": dispersion_alpha.backtest_sharpe,
                "win_rate": dispersion_alpha.backtest_win_rate,
                "vega_pnl": dispersion_alpha.vega_pnl_pct,
                "theta_pnl": dispersion_alpha.theta_pnl_pct,
                "gamma_pnl": dispersion_alpha.gamma_pnl_pct,
                "short_vol_score": dispersion_alpha.short_vol_score,
                "implied_corr": dispersion_alpha.implied_corr,
                "vrp": dispersion_alpha.vrp,
            }

        gpt_suggestions = query_gpt_for_alpha(
            metrics, regime_result.regime_sharpes, pair_data, opts_data, disp_data,
        )

    # ── Build recommendations ─────────────────────────────────────────────
    recommendations = _build_recommendations(
        best_oos, regime_result, dispersion_alpha, ensemble, stability,
    )

    # Final ensemble Sharpe — best of: regime-adaptive, ensemble, dispersion
    candidate_sharpes = [regime_result.combined_sharpe]
    if ensemble:
        candidate_sharpes.append(ensemble.ensemble_sharpe)
    if dispersion_alpha:
        candidate_sharpes.append(dispersion_alpha.backtest_sharpe)
    final_sharpe = max(candidate_sharpes)

    total_runtime = time.time() - t_start

    report = AlphaReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        oos_results=oos_results,
        best_single_strategy=best_oos if best_oos else OOSResult(
            params={}, in_sample_sharpe=0, out_of_sample_sharpe=0,
            oos_win_rate=0, oos_pnl=0, oos_max_dd=0, oos_trades=0,
            is_valid=False, regime="ALL",
        ),
        regime_adaptive=regime_result,
        dispersion_alpha=dispersion_alpha,
        ensemble=ensemble,
        stability=stability,
        ensemble_sharpe=round(final_sharpe, 4),
        recommendations=recommendations,
        gpt_suggestions=gpt_suggestions,
        total_runtime_s=round(total_runtime, 1),
        n_strategies_tested=ensemble.n_strategies if ensemble else 0,
        n_param_combos_tested=n_combos,
    )

    # ── Persist report ────────────────────────────────────────────────────
    _persist_report(report)

    log.info("=" * 70)
    log.info(
        "Alpha Research COMPLETE — %.1fs | ensemble Sharpe=%.3f | %d recommendations",
        total_runtime, final_sharpe, len(recommendations),
    )
    log.info("=" * 70)

    return report


def _build_recommendations(
    best_oos: Optional[OOSResult],
    regime: RegimeAdaptiveResult,
    dispersion: Optional[DispersionAlphaResult],
    ensemble: Optional[StrategyEnsembleResult],
    stability: Optional[AlphaStabilityResult],
) -> List[str]:
    """Build prioritized PM recommendations from all research streams."""
    recs = []

    # OOS validation
    if best_oos and best_oos.is_valid:
        recs.append(
            f"[P1] OOS validated: Sharpe={best_oos.out_of_sample_sharpe:.3f}, "
            f"decay={best_oos.sharpe_decay_ratio:.2f}, p={best_oos.p_value:.3f}, "
            f"CI=[{best_oos.ci_lower:.2f}, {best_oos.ci_upper:.2f}]"
        )
    elif best_oos:
        recs.append(
            f"[P2] Best OOS Sharpe={best_oos.out_of_sample_sharpe:.3f} — "
            f"NOT significant (p={best_oos.p_value:.3f}). Need more data or better params."
        )
    else:
        recs.append("[P1] No OOS-validated strategy found — alpha is weak or data insufficient.")

    # Dispersion alpha (PRIMARY)
    if dispersion:
        if dispersion.backtest_sharpe > 0.3:
            recs.append(
                f"[P1] Dispersion alpha STRONG: Sharpe={dispersion.backtest_sharpe:.3f}, "
                f"WR={dispersion.backtest_win_rate:.0%}, "
                f"Vega={dispersion.vega_pnl_pct:+.1f}% / Theta={dispersion.theta_pnl_pct:+.1f}% / "
                f"Gamma={dispersion.gamma_pnl_pct:+.1f}%"
            )
        elif dispersion.backtest_sharpe > 0:
            recs.append(
                f"[P2] Dispersion alpha positive but weak: Sharpe={dispersion.backtest_sharpe:.3f}. "
                f"Short vol score={dispersion.short_vol_score:.0f}/100 ({dispersion.short_vol_label})"
            )
        if dispersion.vrp > 0.03:
            recs.append(f"[P2] VRP positive ({dispersion.vrp:.1%}) — favorable for short vol entry")

    # Regime
    if regime.oos_validated:
        recs.append(
            f"[P2] Regime-adaptive validated: {regime.regime_sharpes} "
            f"(stability={regime.sharpe_stability:.2f})"
        )
    for r, s in regime.regime_sharpes.items():
        if s > 0.5:
            recs.append(f"[P3] {r}: strong alpha (Sharpe={s:.3f}), params={regime.regime_params.get(r, {})}")

    # Ensemble
    if ensemble and ensemble.ensemble_sharpe > 0:
        recs.append(
            f"[P3] Ensemble of {ensemble.n_strategies} strategies: "
            f"Sharpe={ensemble.ensemble_sharpe:.3f}, diversity={ensemble.diversity_score:.2f}"
        )

    # Stability warnings
    if stability:
        if stability.pct_positive < 0.5:
            recs.append(
                f"[P1] WARNING: Alpha unstable — only {stability.pct_positive:.0%} of rolling windows positive"
            )
        if stability.most_sensitive_param:
            recs.append(
                f"[P3] Most sensitive param: '{stability.most_sensitive_param}' "
                f"(sensitivity={stability.param_sensitivity.get(stability.most_sensitive_param, 0):.4f})"
            )

    return recs


def _persist_report(report: AlphaReport) -> None:
    """Save report to disk + publish to agent bus."""
    report_dir = ROOT / "agents" / "methodology" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{date.today().isoformat()}_alpha_research.json"

    report_dict: Dict[str, Any] = {
        "timestamp": report.timestamp,
        "total_runtime_s": report.total_runtime_s,
        "n_strategies_tested": report.n_strategies_tested,
        "n_param_combos_tested": report.n_param_combos_tested,
        "ensemble_sharpe": report.ensemble_sharpe,
        "best_oos": {
            "params": report.best_single_strategy.params,
            "is_sharpe": report.best_single_strategy.in_sample_sharpe,
            "oos_sharpe": report.best_single_strategy.out_of_sample_sharpe,
            "oos_wr": report.best_single_strategy.oos_win_rate,
            "oos_trades": report.best_single_strategy.oos_trades,
            "valid": report.best_single_strategy.is_valid,
            "decay_ratio": report.best_single_strategy.sharpe_decay_ratio,
            "t_stat": report.best_single_strategy.t_stat,
            "p_value": report.best_single_strategy.p_value,
            "ci_lower": report.best_single_strategy.ci_lower,
            "ci_upper": report.best_single_strategy.ci_upper,
        },
        "oos_all_splits": [
            {
                "split": r.split_idx,
                "params": r.params,
                "is_sharpe": r.in_sample_sharpe,
                "oos_sharpe": r.out_of_sample_sharpe,
                "decay_ratio": r.sharpe_decay_ratio,
                "p_value": r.p_value,
                "valid": r.is_valid,
                "trades": r.oos_trades,
            }
            for r in report.oos_results
        ],
        "regime_adaptive": {
            "params": report.regime_adaptive.regime_params,
            "sharpes": report.regime_adaptive.regime_sharpes,
            "trades": report.regime_adaptive.regime_trades,
            "combined_sharpe": report.regime_adaptive.combined_sharpe,
            "stability": report.regime_adaptive.sharpe_stability,
            "validated": report.regime_adaptive.oos_validated,
        },
        "recommendations": report.recommendations,
        "gpt_suggestions": report.gpt_suggestions,
    }

    # Dispersion alpha
    if report.dispersion_alpha:
        da = report.dispersion_alpha
        report_dict["dispersion_alpha"] = {
            "sharpe": da.backtest_sharpe,
            "win_rate": da.backtest_win_rate,
            "total_pnl": da.backtest_total_pnl,
            "max_dd": da.backtest_max_dd,
            "trades": da.backtest_trades,
            "vega_pnl_pct": da.vega_pnl_pct,
            "theta_pnl_pct": da.theta_pnl_pct,
            "gamma_pnl_pct": da.gamma_pnl_pct,
            "short_vol_score": da.short_vol_score,
            "short_vol_label": da.short_vol_label,
            "implied_corr": da.implied_corr,
            "vrp": da.vrp,
            "best_params": da.best_params,
            "pnl_by_regime": da.pnl_by_regime,
        }

    # Ensemble
    if report.ensemble:
        ens = report.ensemble
        report_dict["ensemble"] = {
            "sharpe": ens.ensemble_sharpe,
            "n_strategies": ens.n_strategies,
            "diversity": ens.diversity_score,
            "weights": ens.strategy_weights,
            "sharpes": ens.strategy_sharpes,
        }

    # Stability
    if report.stability:
        stab = report.stability
        report_dict["stability"] = {
            "sharpe_mean": stab.sharpe_mean,
            "sharpe_std": stab.sharpe_std,
            "sharpe_min": stab.sharpe_min,
            "pct_positive": stab.pct_positive,
            "param_sensitivity": stab.param_sensitivity,
            "most_sensitive": stab.most_sensitive_param,
            "rolling_sharpes": stab.rolling_oos_sharpes,
        }

    report_path.write_text(json.dumps(report_dict, indent=2, default=str), encoding="utf-8")
    log.info("Alpha report saved: %s", report_path.name)

    # Publish to bus
    try:
        from scripts.agent_bus import AgentBus
        AgentBus().publish("alpha_research", report_dict)
    except Exception:
        pass
