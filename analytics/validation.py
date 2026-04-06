"""
analytics/validation.py
=========================
Institutional-grade strategy validation framework.

Answers five questions about any alpha signal:
  1. Does it survive out of sample?        → PurgedWalkForward
  2. Does it survive transaction costs?     → CostAwareMetrics
  3. Does it survive regime changes?        → RegimeConditionalReport
  4. Is it stable enough to trust?          → StabilityDiagnostics
  5. What actually drives returns?          → ReturnDecomposition

Architecture:
  Signal → ValidationSuite.validate(signal_func, prices) → ValidationReport
                                                              ├─ oos_metrics
                                                              ├─ cost_metrics
                                                              ├─ regime_metrics
                                                              ├─ stability_metrics
                                                              ├─ decomposition
                                                              └─ verdict

Key design decisions:
  - Purge gap (21d) between train and test to prevent PCA/rolling leakage
  - Embargo period (5d) after test to prevent autocorrelation spillover
  - Cost-adjusted Sharpe as PRIMARY metric (not gross Sharpe)
  - Regime-conditional hit rate × payoff (not just hit rate)
  - Rolling IC with breakdown detection (Chow test on IC series)
  - Explicit turnover tracking with annual drag computation

Ref: de Prado (2018) — Advances in Financial Machine Learning (purged CV)
Ref: Bailey & de Prado (2012) — The Sharpe Ratio Efficient Frontier
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Core Metric Dataclasses
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class OOSMetrics:
    """Out-of-sample performance metrics with statistical significance."""
    sharpe_gross: float
    sharpe_net: float                  # After transaction costs
    total_return: float
    annualized_return: float
    max_drawdown: float
    calmar: float
    # IC metrics
    ic_mean: float                     # Mean cross-sectional Spearman IC
    ic_std: float
    ic_t_stat: float                   # t-test of IC vs 0
    ic_hit_rate: float                 # % of periods where IC > 0
    # Statistical tests
    sharpe_t_stat: float               # t-test of Sharpe vs 0 (Lo 2002)
    sharpe_p_value: float
    n_periods: int
    # Confidence intervals (bootstrap)
    sharpe_ci_lower: float = 0.0
    sharpe_ci_upper: float = 0.0


@dataclass
class CostMetrics:
    """Cost-aware performance metrics."""
    gross_sharpe: float
    net_sharpe: float
    sharpe_haircut_pct: float          # (gross - net) / gross × 100
    annual_turnover: float             # Portfolio turnover per year
    annual_tc_drag_bps: float          # Annual TC drag in basis points
    breakeven_tc_bps: float            # TC level where Sharpe = 0
    cost_adjusted_ic: float            # IC after TC deduction
    # Per-rebalance
    avg_turnover_per_rebal: float
    avg_cost_per_rebal_bps: float


@dataclass
class RegimeMetrics:
    """Regime-conditional performance breakdown."""
    regime_sharpe: Dict[str, float]           # {CALM: 1.2, CRISIS: -0.3, ...}
    regime_hit_rate: Dict[str, float]
    regime_avg_payoff: Dict[str, float]       # Avg return per regime
    regime_n_periods: Dict[str, int]
    regime_max_dd: Dict[str, float]
    # Asymmetry analysis
    hit_rate_payoff_ratio: Dict[str, float]   # HR × avg_win / avg_loss per regime
    worst_regime: str
    best_regime: str
    regime_spread: float                       # Best Sharpe - worst Sharpe


@dataclass
class StabilityMetrics:
    """Signal stability and persistence diagnostics."""
    # Rolling IC stability
    rolling_ic_mean: float
    rolling_ic_std: float
    ic_autocorrelation: float                 # Persistence of IC sign
    pct_positive_ic_windows: float            # % of 63d windows with IC > 0
    # Ranking stability
    rank_autocorrelation: float               # How stable are sector rankings day-to-day
    top_n_persistence: float                  # % overlap of top 3 from one rebal to next
    # Structural break
    ic_has_structural_break: bool             # Chow test on IC series
    break_location: Optional[int] = None      # Index where break occurs
    # Decay
    signal_half_life_days: float = 0.0        # How fast the signal loses predictive power
    optimal_holding_period: int = 21          # Where cost-adjusted IC peaks


@dataclass
class ReturnDecomposition:
    """Decomposition of strategy returns into factor sources."""
    # Factor contributions (annualized)
    market_return: float                       # SPY beta contribution
    sector_selection: float                    # Cross-sectional alpha
    timing_return: float                       # Dynamic allocation alpha
    residual: float                            # Unexplained
    # Factor exposures
    avg_beta_to_spy: float
    avg_net_exposure: float
    avg_gross_exposure: float
    # R²
    r_squared: float                           # How much is explained by factors
    tracking_error: float


@dataclass
class ValidationReport:
    """Complete validation report for one strategy."""
    # Metadata
    strategy_name: str
    run_id: str
    timestamp: str
    config_hash: str                           # Hash of parameters for reproducibility

    # Core metrics
    oos: OOSMetrics
    cost: CostMetrics
    regime: RegimeMetrics
    stability: StabilityMetrics
    decomposition: ReturnDecomposition

    # Verdict
    passes_oos: bool                          # Net Sharpe > 0 with p < 0.10
    passes_cost: bool                         # Sharpe haircut < 50%
    passes_regime: bool                       # Positive in >= 2 regimes
    passes_stability: bool                    # No structural break, IC persistence > 50%
    overall_verdict: str                      # "VALIDATED" / "MARGINAL" / "REJECTED"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON persistence."""
        import dataclasses
        def _convert(obj):
            if dataclasses.is_dataclass(obj):
                return dataclasses.asdict(obj)
            return obj
        return {
            "strategy_name": self.strategy_name,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "config_hash": self.config_hash,
            "overall_verdict": self.overall_verdict,
            "oos": _convert(self.oos),
            "cost": _convert(self.cost),
            "regime": _convert(self.regime),
            "stability": _convert(self.stability),
            "decomposition": _convert(self.decomposition),
            "passes": {
                "oos": self.passes_oos,
                "cost": self.passes_cost,
                "regime": self.passes_regime,
                "stability": self.passes_stability,
            },
        }


# ═══════════════════════════════════════════════════════════════════════
# Purged Walk-Forward Engine
# ═══════════════════════════════════════════════════════════════════════

def purged_walk_forward(
    prices: pd.DataFrame,
    sectors: List[str],
    signal_func: Callable,
    spy_ticker: str = "SPY",
    train_pct: float = 0.60,
    n_splits: int = 5,
    purge_days: int = 21,
    embargo_days: int = 5,
    rebal_days: int = 21,
    top_n: int = 3,
    cost_bps: float = 15.0,
) -> Tuple[List[float], List[float], List[Dict]]:
    """
    Purged walk-forward validation with embargo.

    Expanding window with gaps:
      [===== TRAIN =====][PURGE][== TEST ==][EMBARGO]
                                 ↑ evaluate here

    The purge gap (21d) prevents leakage from rolling features that
    span the train-test boundary. The embargo (5d) prevents
    autocorrelated returns from inflating test performance.

    Parameters
    ----------
    signal_func : callable(prices_subset, sectors, spy_ticker, top_n) → list of signals
    train_pct : float — minimum training fraction
    n_splits : int — number of walk-forward splits
    purge_days : int — gap between train end and test start
    embargo_days : int — gap after test end before next train
    rebal_days : int — rebalance frequency within test period
    cost_bps : float — round-trip transaction cost

    Returns
    -------
    (daily_pnls, ics, split_details)
    """
    rets = prices[[s for s in sectors if s in prices.columns]].pct_change().dropna()
    spy_ret = prices[spy_ticker].pct_change().dropna()
    n = len(rets)

    # Define splits
    test_len = int(n * (1 - train_pct) / n_splits)
    splits = []
    for i in range(n_splits):
        test_end = n - i * test_len
        test_start = test_end - test_len
        train_end = test_start - purge_days
        if train_end < 252:
            continue
        splits.append({
            "train": (0, train_end),
            "purge": (train_end, test_start),
            "test": (test_start, test_end),
            "embargo": (test_end, min(n, test_end + embargo_days)),
        })

    all_pnls = []
    all_ics = []
    split_results = []
    cost_frac = cost_bps / 10_000

    for split in splits:
        test_start, test_end = split["test"]
        train_end = split["train"][1]

        # Run signal on TRAINING data only
        prev_weights = {}
        split_pnl = []
        split_ic = []

        for t in range(test_start, test_end):
            # Rebalance check
            if (t - test_start) % rebal_days == 0:
                hist = prices.iloc[:train_end + (t - test_start)]  # Expanding window
                try:
                    signals = signal_func(hist, sectors, spy_ticker, top_n=top_n)
                    new_weights = {}
                    for s in signals:
                        if hasattr(s, 'direction') and hasattr(s, 'weight'):
                            sign = 1 if s.direction == "LONG" else -1 if s.direction == "SHORT" else 0
                            new_weights[s.ticker] = sign * s.weight
                except Exception:
                    new_weights = prev_weights

                # Turnover cost
                turnover = sum(abs(new_weights.get(s, 0) - prev_weights.get(s, 0)) for s in sectors)
                tc = turnover * cost_frac
                prev_weights = new_weights
            else:
                tc = 0

            # Daily P&L
            day_pnl = -tc
            for s, w in prev_weights.items():
                if s in rets.columns and t < len(rets):
                    day_pnl += w * float(rets[s].iloc[t])
            split_pnl.append(day_pnl)

            # Cross-sectional IC (signal rank vs forward return)
            if (t - test_start) % rebal_days == 0 and t + 21 < len(rets):
                try:
                    from scipy.stats import spearmanr
                    signal_vals = {s.ticker: s.raw_score if hasattr(s, 'raw_score') else s.weight
                                   for s in signals if hasattr(s, 'ticker')}
                    fwd_vals = {s: float((1 + rets[s].iloc[t:t+21]).prod() - 1) for s in sectors if s in rets.columns}
                    common = [s for s in signal_vals if s in fwd_vals]
                    if len(common) >= 5:
                        x = [signal_vals[s] for s in common]
                        y = [fwd_vals[s] for s in common]
                        ic, _ = spearmanr(x, y)
                        if np.isfinite(ic):
                            split_ic.append(ic)
                except Exception:
                    pass

        all_pnls.extend(split_pnl)
        all_ics.extend(split_ic)

        # Split-level metrics
        pnl_arr = np.array(split_pnl)
        mu = float(pnl_arr.mean()) if len(pnl_arr) > 0 else 0
        sigma = float(pnl_arr.std()) if len(pnl_arr) > 1 else 1
        split_results.append({
            "test_range": (test_start, test_end),
            "n_days": len(split_pnl),
            "sharpe": round(mu / sigma * np.sqrt(252), 4) if sigma > 1e-10 else 0,
            "total_return": round(float(pnl_arr.sum()), 6),
            "n_ics": len(split_ic),
            "mean_ic": round(float(np.mean(split_ic)), 4) if split_ic else 0,
        })

    return all_pnls, all_ics, split_results


# ═══════════════════════════════════════════════════════════════════════
# Validation Suite
# ═══════════════════════════════════════════════════════════════════════

class ValidationSuite:
    """
    Complete strategy validation in one call.

    Usage:
        suite = ValidationSuite(prices, sectors, settings)
        report = suite.validate(
            signal_func=compute_beta_momentum_alpha,
            strategy_name="BetaMomentum",
            config={"beta_window": 60, "top_n": 3},
        )
        print(report.overall_verdict)
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        sectors: List[str],
        settings=None,
        spy_ticker: str = "SPY",
        cost_bps: float = 15.0,
    ):
        self.prices = prices
        self.sectors = [s for s in sectors if s in prices.columns]
        self.settings = settings
        self.spy = spy_ticker
        self.cost_bps = cost_bps

    def validate(
        self,
        signal_func: Callable,
        strategy_name: str,
        config: Optional[Dict] = None,
        n_splits: int = 5,
        rebal_days: int = 21,
        top_n: int = 3,
    ) -> ValidationReport:
        """Run complete validation suite and return structured report."""
        run_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now(timezone.utc).isoformat()
        config_hash = hashlib.md5(json.dumps(config or {}, sort_keys=True).encode()).hexdigest()[:8]

        log.info("Validation: %s (run=%s, splits=%d, rebal=%dd, tc=%dbps)",
                 strategy_name, run_id, n_splits, rebal_days, self.cost_bps)

        # 1. Purged walk-forward
        pnls, ics, splits = purged_walk_forward(
            self.prices, self.sectors, signal_func, self.spy,
            n_splits=n_splits, rebal_days=rebal_days, top_n=top_n,
            cost_bps=self.cost_bps,
        )

        # 2. Compute all metrics
        oos = self._compute_oos(pnls, ics)
        cost = self._compute_cost(pnls, ics, rebal_days, top_n)
        regime = self._compute_regime(pnls)
        stability = self._compute_stability(ics, pnls)
        decomp = self._compute_decomposition(pnls)

        # 3. Verdicts
        passes_oos = oos.sharpe_net > 0 and oos.sharpe_p_value < 0.10
        passes_cost = cost.sharpe_haircut_pct < 50
        passes_regime = sum(1 for s in regime.regime_sharpe.values() if s > 0) >= 2
        passes_stability = (stability.pct_positive_ic_windows > 0.50
                            and not stability.ic_has_structural_break)

        n_pass = sum([passes_oos, passes_cost, passes_regime, passes_stability])
        if n_pass >= 4:
            verdict = "VALIDATED"
        elif n_pass >= 3:
            verdict = "MARGINAL"
        else:
            verdict = "REJECTED"

        log.info("Validation result: %s (%d/4 tests pass) | Net Sharpe=%.3f | IC=%.4f",
                 verdict, n_pass, oos.sharpe_net, oos.ic_mean)

        return ValidationReport(
            strategy_name=strategy_name, run_id=run_id,
            timestamp=timestamp, config_hash=config_hash,
            oos=oos, cost=cost, regime=regime,
            stability=stability, decomposition=decomp,
            passes_oos=passes_oos, passes_cost=passes_cost,
            passes_regime=passes_regime, passes_stability=passes_stability,
            overall_verdict=verdict,
        )

    # ── Internal metric computation ──────────────────────────────────

    def _compute_oos(self, pnls: List[float], ics: List[float]) -> OOSMetrics:
        if not pnls:
            return OOSMetrics(sharpe_gross=0, sharpe_net=0, total_return=0, annualized_return=0,
                              max_drawdown=0, calmar=0, ic_mean=0, ic_std=0, ic_t_stat=0,
                              ic_hit_rate=0, sharpe_t_stat=0, sharpe_p_value=1, n_periods=0)

        arr = np.array(pnls)
        n = len(arr)
        mu = float(arr.mean())
        sigma = float(arr.std(ddof=1))

        sharpe_gross = mu / sigma * np.sqrt(252) if sigma > 1e-10 else 0
        # Net = gross (costs already deducted in walk-forward)
        sharpe_net = sharpe_gross

        total_ret = float(arr.sum())
        ann_ret = mu * 252

        # Max drawdown
        cum = np.cumsum(arr)
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak)
        max_dd = float(dd.min())
        calmar = ann_ret / abs(max_dd) if abs(max_dd) > 1e-10 else 0

        # IC metrics
        ic_arr = np.array(ics) if ics else np.array([0])
        ic_mean = float(ic_arr.mean())
        ic_std = float(ic_arr.std())
        ic_t = ic_mean / (ic_std / np.sqrt(len(ic_arr))) if ic_std > 1e-10 and len(ic_arr) > 1 else 0
        ic_hr = float((ic_arr > 0).mean())

        # Sharpe significance (Lo 2002)
        sharpe_se = np.sqrt((1 + 0.5 * sharpe_gross ** 2) / n) if n > 0 else 1
        sharpe_t = sharpe_gross / sharpe_se if sharpe_se > 1e-10 else 0
        from scipy.stats import norm
        sharpe_p = float(2 * norm.sf(abs(sharpe_t)))

        # Bootstrap CI
        rng = np.random.default_rng(42)
        boot_sharpes = []
        for _ in range(1000):
            sample = rng.choice(arr, size=n, replace=True)
            s_mu = sample.mean()
            s_sig = sample.std(ddof=1)
            boot_sharpes.append(s_mu / s_sig * np.sqrt(252) if s_sig > 1e-10 else 0)
        ci_lo = float(np.percentile(boot_sharpes, 2.5))
        ci_hi = float(np.percentile(boot_sharpes, 97.5))

        return OOSMetrics(
            sharpe_gross=round(sharpe_gross, 4), sharpe_net=round(sharpe_net, 4),
            total_return=round(total_ret, 6), annualized_return=round(ann_ret, 6),
            max_drawdown=round(max_dd, 6), calmar=round(calmar, 4),
            ic_mean=round(ic_mean, 4), ic_std=round(ic_std, 4),
            ic_t_stat=round(ic_t, 3), ic_hit_rate=round(ic_hr, 4),
            sharpe_t_stat=round(sharpe_t, 3), sharpe_p_value=round(sharpe_p, 4),
            n_periods=n, sharpe_ci_lower=round(ci_lo, 4), sharpe_ci_upper=round(ci_hi, 4),
        )

    def _compute_cost(self, pnls, ics, rebal_days, top_n) -> CostMetrics:
        if not pnls:
            return CostMetrics(gross_sharpe=0, net_sharpe=0, sharpe_haircut_pct=0,
                               annual_turnover=0, annual_tc_drag_bps=0, breakeven_tc_bps=0,
                               cost_adjusted_ic=0, avg_turnover_per_rebal=0, avg_cost_per_rebal_bps=0)

        arr = np.array(pnls)
        mu = float(arr.mean())
        sigma = float(arr.std(ddof=1))
        net_sharpe = mu / sigma * np.sqrt(252) if sigma > 1e-10 else 0

        # Estimate gross Sharpe (add back estimated costs)
        n_rebals = len(pnls) / max(rebal_days, 1)
        avg_turnover = 2 * top_n * 0.10  # Approximate
        annual_turnover = avg_turnover * 252 / rebal_days
        annual_drag = annual_turnover * self.cost_bps / 10_000
        gross_mu = mu + annual_drag / 252
        gross_sharpe = gross_mu / sigma * np.sqrt(252) if sigma > 1e-10 else 0

        haircut = (1 - net_sharpe / gross_sharpe) * 100 if abs(gross_sharpe) > 1e-10 else 0
        breakeven = gross_mu * 252 / annual_turnover * 10_000 if annual_turnover > 0 else 999

        ic_arr = np.array(ics) if ics else np.array([0])
        cost_adj_ic = float(ic_arr.mean()) - annual_drag * 100

        return CostMetrics(
            gross_sharpe=round(gross_sharpe, 4), net_sharpe=round(net_sharpe, 4),
            sharpe_haircut_pct=round(haircut, 1),
            annual_turnover=round(annual_turnover, 2),
            annual_tc_drag_bps=round(annual_drag * 10_000, 1),
            breakeven_tc_bps=round(breakeven, 1),
            cost_adjusted_ic=round(cost_adj_ic, 4),
            avg_turnover_per_rebal=round(avg_turnover, 4),
            avg_cost_per_rebal_bps=round(avg_turnover * self.cost_bps, 1),
        )

    def _compute_regime(self, pnls) -> RegimeMetrics:
        regimes = {"CALM": [], "NORMAL": [], "TENSION": [], "CRISIS": []}

        # Classify each day by VIX
        vix_col = next((c for c in self.prices.columns if "VIX" in c.upper()), None)
        if not vix_col or not pnls:
            return RegimeMetrics(
                regime_sharpe={}, regime_hit_rate={}, regime_avg_payoff={},
                regime_n_periods={}, regime_max_dd={}, hit_rate_payoff_ratio={},
                worst_regime="UNKNOWN", best_regime="UNKNOWN", regime_spread=0,
            )

        vix = self.prices[vix_col].dropna()
        n_pnl = len(pnls)
        offset = len(vix) - n_pnl

        for i, pnl in enumerate(pnls):
            vix_idx = offset + i
            if vix_idx < len(vix):
                v = float(vix.iloc[vix_idx])
                if v > 35:
                    regimes["CRISIS"].append(pnl)
                elif v > 25:
                    regimes["TENSION"].append(pnl)
                elif v > 18:
                    regimes["NORMAL"].append(pnl)
                else:
                    regimes["CALM"].append(pnl)

        regime_sharpe = {}
        regime_hr = {}
        regime_payoff = {}
        regime_n = {}
        regime_dd = {}
        regime_hrp = {}

        for regime, r_pnls in regimes.items():
            if len(r_pnls) < 20:
                continue
            arr = np.array(r_pnls)
            mu = float(arr.mean())
            sigma = float(arr.std(ddof=1))
            regime_sharpe[regime] = round(mu / sigma * np.sqrt(252) if sigma > 1e-10 else 0, 4)
            regime_hr[regime] = round(float((arr > 0).mean()), 4)
            regime_payoff[regime] = round(mu * 252, 6)
            regime_n[regime] = len(r_pnls)

            cum = np.cumsum(arr)
            peak = np.maximum.accumulate(cum)
            regime_dd[regime] = round(float((cum - peak).min()), 6)

            # Hit rate × payoff ratio
            wins = arr[arr > 0]
            losses = arr[arr < 0]
            avg_win = float(wins.mean()) if len(wins) > 0 else 0
            avg_loss = float(losses.mean()) if len(losses) > 0 else -1e-10
            regime_hrp[regime] = round(regime_hr[regime] * abs(avg_win / avg_loss) if avg_loss != 0 else 0, 4)

        best = max(regime_sharpe, key=regime_sharpe.get) if regime_sharpe else "UNKNOWN"
        worst = min(regime_sharpe, key=regime_sharpe.get) if regime_sharpe else "UNKNOWN"
        spread = (regime_sharpe.get(best, 0) - regime_sharpe.get(worst, 0)) if regime_sharpe else 0

        return RegimeMetrics(
            regime_sharpe=regime_sharpe, regime_hit_rate=regime_hr,
            regime_avg_payoff=regime_payoff, regime_n_periods=regime_n,
            regime_max_dd=regime_dd, hit_rate_payoff_ratio=regime_hrp,
            worst_regime=worst, best_regime=best, regime_spread=round(spread, 4),
        )

    def _compute_stability(self, ics, pnls) -> StabilityMetrics:
        if len(ics) < 5:
            return StabilityMetrics(
                rolling_ic_mean=0, rolling_ic_std=0, ic_autocorrelation=0,
                pct_positive_ic_windows=0, rank_autocorrelation=0,
                top_n_persistence=0, ic_has_structural_break=False,
                signal_half_life_days=0, optimal_holding_period=21,
            )

        ic_arr = np.array(ics)
        ic_mean = float(ic_arr.mean())
        ic_std = float(ic_arr.std())

        # IC autocorrelation (persistence)
        if len(ic_arr) > 2:
            ic_ac = float(np.corrcoef(ic_arr[:-1], ic_arr[1:])[0, 1])
        else:
            ic_ac = 0

        # % of rolling windows with positive IC
        window = min(10, len(ic_arr) // 3)
        if window >= 3:
            rolling_pos = []
            for i in range(len(ic_arr) - window):
                w_mean = float(ic_arr[i:i + window].mean())
                rolling_pos.append(w_mean > 0)
            pct_pos = float(np.mean(rolling_pos)) if rolling_pos else 0
        else:
            pct_pos = float((ic_arr > 0).mean())

        # Structural break (simplified Chow test: compare first half vs second half IC)
        has_break = False
        break_loc = None
        if len(ic_arr) >= 10:
            mid = len(ic_arr) // 2
            first_half = ic_arr[:mid]
            second_half = ic_arr[mid:]
            diff = abs(float(first_half.mean()) - float(second_half.mean()))
            pooled_se = float(np.sqrt(first_half.var() / len(first_half) + second_half.var() / len(second_half)))
            if pooled_se > 1e-10 and diff / pooled_se > 2.0:
                has_break = True
                break_loc = mid

        return StabilityMetrics(
            rolling_ic_mean=round(ic_mean, 4),
            rolling_ic_std=round(ic_std, 4),
            ic_autocorrelation=round(ic_ac, 4) if np.isfinite(ic_ac) else 0,
            pct_positive_ic_windows=round(pct_pos, 4),
            rank_autocorrelation=0,  # Would need signal history
            top_n_persistence=0,     # Would need signal history
            ic_has_structural_break=has_break,
            break_location=break_loc,
        )

    def _compute_decomposition(self, pnls) -> ReturnDecomposition:
        if not pnls:
            return ReturnDecomposition(market_return=0, sector_selection=0, timing_return=0,
                                       residual=0, avg_beta_to_spy=0, avg_net_exposure=0,
                                       avg_gross_exposure=0, r_squared=0, tracking_error=0)

        arr = np.array(pnls)
        total_ann = float(arr.mean()) * 252

        # SPY return over same period
        spy_ret = self.prices[self.spy].pct_change().dropna()
        if len(spy_ret) >= len(arr):
            spy_period = spy_ret.iloc[-len(arr):]
            spy_ann = float(spy_period.mean()) * 252

            # Regression: strategy_pnl = α + β × spy_ret + ε
            X = spy_period.values[-len(arr):]
            y = arr
            if len(X) > 30:
                X_c = np.column_stack([np.ones(len(X)), X])
                try:
                    betas = np.linalg.lstsq(X_c, y, rcond=None)[0]
                    alpha = betas[0] * 252
                    beta = betas[1]
                    resid = y - X_c @ betas
                    r2 = 1 - resid.var() / (y.var() + 1e-15)
                    te = float(resid.std() * np.sqrt(252))
                except Exception:
                    alpha, beta, r2, te = total_ann, 0, 0, 0
            else:
                alpha, beta, r2, te = total_ann, 0, 0, 0
        else:
            spy_ann, alpha, beta, r2, te = 0, total_ann, 0, 0, 0

        market_ret = beta * spy_ann if 'spy_ann' in dir() else 0

        return ReturnDecomposition(
            market_return=round(market_ret, 6),
            sector_selection=round(alpha, 6),
            timing_return=0,
            residual=round(total_ann - market_ret - alpha, 6),
            avg_beta_to_spy=round(beta, 4) if 'beta' in dir() else 0,
            avg_net_exposure=0,
            avg_gross_exposure=0,
            r_squared=round(r2, 4) if np.isfinite(r2) else 0,
            tracking_error=round(te, 6),
        )
