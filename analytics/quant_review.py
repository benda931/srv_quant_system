"""
analytics/quant_review.py
============================
Quantitative methodology fixes for the SRV signal pipeline.

This module implements corrections to the mathematical weaknesses
identified in the alpha methodology review of stat_arb.py.

## Review Summary

### What stat_arb.py Does Well
1. OOS PCA residuals — fits on [t-window:t], transforms t+1 (strict OOS)
2. PCA refit interval — caches PCA model for N days (avoids overfitting noise)
3. Relative returns — subtracts SPY before PCA (removes market factor)
4. EWMA volatility — λ-weighted vol for hedge ratio (responsive to vol regime)
5. Dispersion decomposition — proper variance decomposition with sector weights
6. Regime classification — multi-indicator (VIX + corr + mode + credit + distortion)

### Critical Mathematical Weaknesses
1. Z-SCORE ON CUMULATIVE RESIDUALS — the z-score is computed on cumsum(resid),
   not on the residual itself. cumsum is a random walk if residuals are i.i.d.,
   so the z-score is meaningless for mean-reversion.

   Current: z = (cumsum(resid) - rolling_mean(cumsum)) / rolling_std(cumsum)
   Problem: cumsum of i.i.d. is a random walk → z-score has no predictive power
   Evidence: IC(z→20d return) ≈ 0, half-life = 1 day

2. HEDGE RATIO BY VOL RATIO — uses σ_SPY/σ_sector as hedge ratio.
   This is NOT a proper hedge ratio (should be beta = cov/var).
   Vol ratio ≠ beta unless correlation = 1.

3. NO STATIONARITY GATING — enters trades on z-score without checking
   if the residual is actually stationary. If residual has a unit root,
   z-score mean-reversion is unreliable.

4. CONVICTION SCORE IS MULTIPLICATIVE HEURISTIC —
   conviction = score₁ × score₂ × ... × scoreN
   No probabilistic calibration. Multiplying [0,1] scores pushes
   everything toward zero. Not a probability of profit.

5. DECISION THRESHOLDS ARE HARDCODED —
   0.55, 0.45, 0.30 etc. are magic numbers without statistical basis.

### Fixes Implemented in This Module

Fix 1: Windowed Z-Score on Returns (not cumsum)
  - Z-score the rolling sum of returns over [t-H:t] where H = half_life
  - This captures deviation from expected return over the natural reversion horizon

Fix 2: Beta Hedge Ratio (OLS, not vol ratio)
  - β = Cov(r_sector, r_SPY) / Var(r_SPY) over rolling window
  - Proper risk-neutral hedging

Fix 3: Stationarity Gate (ADF p-value < 0.10)
  - Only trade sectors where residual passes ADF test
  - Prevents trading non-stationary noise

Fix 4: Rank-Based Scoring (cross-sectional, not absolute)
  - Cross-sectional z-score: (score - median) / MAD
  - Robust to outliers, naturally calibrated

Fix 5: Turnover-Penalized Signal
  - Signal includes penalty for high turnover: net_signal = signal - λ × Δsignal
  - Reduces unnecessary trading when conviction barely changes
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Fix 1: Windowed Z-Score on Returns (not cumulative residual)
# ═══════════════════════════════════════════════════════════════════════

def windowed_return_zscore(
    returns: pd.Series,
    half_life: float = 20.0,
    lookback: int = 60,
) -> pd.Series:
    """
    Z-score of rolling sum of returns over a window matched to half-life.

    Instead of z-scoring cumulative residuals (random walk if i.i.d.),
    compute the z-score of the rolling H-day return where H ≈ half-life.

    This measures: "how far has this sector deviated from its expected
    return over the natural reversion horizon?"

    z_t = (Σ_{i=t-H}^{t} r_i - μ_H) / σ_H

    where μ_H, σ_H are the rolling mean and std of H-day returns.
    """
    H = max(5, min(60, int(half_life)))
    rolling_return = returns.rolling(H).sum()
    mu = rolling_return.rolling(lookback).mean()
    sigma = rolling_return.rolling(lookback).std(ddof=1)
    z = (rolling_return - mu) / sigma.replace(0, np.nan)
    return z


# ═══════════════════════════════════════════════════════════════════════
# Fix 2: Beta Hedge Ratio (OLS, not vol ratio)
# ═══════════════════════════════════════════════════════════════════════

def rolling_beta_hedge(
    sector_returns: pd.Series,
    spy_returns: pd.Series,
    window: int = 60,
) -> pd.Series:
    """
    Rolling OLS beta: β = Cov(r_sector, r_SPY) / Var(r_SPY).

    This is the proper hedge ratio for a long-short sector-vs-SPY trade.
    Vol ratio (σ_SPY/σ_sector) is incorrect because it assumes ρ = 1.
    """
    cov = sector_returns.rolling(window).cov(spy_returns)
    var_spy = spy_returns.rolling(window).var(ddof=0)
    beta = cov / var_spy.replace(0, np.nan)
    return beta.clip(-3, 3)  # Cap at ±3 to avoid extreme values


# ═══════════════════════════════════════════════════════════════════════
# Fix 3: Stationarity Gate
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class StationarityResult:
    """Result of stationarity test on a residual series."""
    ticker: str
    adf_stat: float
    adf_pvalue: float
    is_stationary: bool              # True if p < threshold
    half_life: float
    hurst: float
    tradeable: bool                  # Stationary AND half_life in sweet spot


def test_stationarity(
    residual_level: pd.Series,
    ticker: str = "",
    p_threshold: float = 0.10,
    hl_min: float = 3.0,
    hl_max: float = 60.0,
) -> StationarityResult:
    """
    ADF test + half-life + Hurst check for residual stationarity.

    A sector is tradeable for mean-reversion only if:
    1. ADF p-value < threshold (residual is stationary)
    2. Half-life is in [hl_min, hl_max] (reversion is neither too fast nor too slow)
    3. Hurst < 0.5 (anti-persistent / mean-reverting dynamics)
    """
    x = residual_level.dropna().values
    n = len(x)

    if n < 60:
        return StationarityResult(ticker=ticker, adf_stat=0, adf_pvalue=1,
                                   is_stationary=False, half_life=float("inf"),
                                   hurst=0.5, tradeable=False)

    # ADF test
    try:
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(x, maxlag=10, autolag="AIC")
        adf_stat = float(result[0])
        adf_p = float(result[1])
    except ImportError:
        # Fallback: simplified DF regression
        dx = np.diff(x)
        x_lag = x[:-1]
        if len(x_lag) > 10:
            beta = np.polyfit(x_lag, dx, 1)[0]
            adf_stat = beta / (np.std(dx - beta * x_lag) / np.sqrt(len(x_lag)))
            adf_p = 0.05 if adf_stat < -2.86 else 0.5
        else:
            adf_stat, adf_p = 0, 1

    is_stationary = adf_p < p_threshold

    # Half-life
    dx = np.diff(x)
    x_lag = x[:-1]
    if len(x_lag) > 10:
        try:
            phi = np.polyfit(x_lag, dx, 1)[0]
            hl = -np.log(2) / np.log(1 + phi) if -1 < phi < 0 else float("inf")
        except Exception:
            hl = float("inf")
    else:
        hl = float("inf")

    # Hurst exponent (simplified R/S)
    try:
        from analytics.signal_mean_reversion import _hurst_exponent
        hurst = _hurst_exponent(residual_level)
    except Exception:
        hurst = 0.5

    tradeable = (is_stationary
                 and math.isfinite(hl) and hl_min <= hl <= hl_max
                 and hurst < 0.50)

    return StationarityResult(
        ticker=ticker, adf_stat=round(adf_stat, 4), adf_pvalue=round(adf_p, 4),
        is_stationary=is_stationary, half_life=round(hl, 1),
        hurst=round(hurst, 3), tradeable=tradeable,
    )


# ═══════════════════════════════════════════════════════════════════════
# Fix 4: Robust Cross-Sectional Scoring (rank-based)
# ═══════════════════════════════════════════════════════════════════════

def cross_sectional_zscore(
    scores: Dict[str, float],
    use_mad: bool = True,
) -> Dict[str, float]:
    """
    Cross-sectional z-score using median and MAD (robust to outliers).

    Standard z-score: (x - mean) / std
    Robust z-score:   (x - median) / (1.4826 × MAD)

    where MAD = median(|x - median(x)|)

    The 1.4826 factor makes MAD consistent with σ for normal distributions.

    Benefits:
    - Not dominated by a single extreme sector
    - Natural calibration: ±1 = typical deviation from median
    - Stable when one sector has extreme dislocation
    """
    vals = np.array(list(scores.values()), dtype=float)
    names = list(scores.keys())

    if len(vals) < 3:
        return scores

    if use_mad:
        median = float(np.median(vals))
        mad = float(np.median(np.abs(vals - median)))
        scale = 1.4826 * mad if mad > 1e-10 else float(np.std(vals))
        if scale < 1e-10:
            return {n: 0.0 for n in names}
        z_scores = (vals - median) / scale
    else:
        mu = float(vals.mean())
        sigma = float(vals.std(ddof=1))
        if sigma < 1e-10:
            return {n: 0.0 for n in names}
        z_scores = (vals - mu) / sigma

    return {names[i]: round(float(z_scores[i]), 4) for i in range(len(names))}


# ═══════════════════════════════════════════════════════════════════════
# Fix 5: Turnover-Penalized Signal
# ═══════════════════════════════════════════════════════════════════════

def turnover_penalized_signal(
    current_signal: Dict[str, float],
    previous_signal: Dict[str, float],
    penalty_lambda: float = 0.3,
) -> Dict[str, float]:
    """
    Penalize signal changes to reduce unnecessary turnover.

    net_signal_i = signal_i - λ × |signal_i - prev_signal_i|

    If a sector's signal barely changed, the penalty is small.
    If the signal flipped dramatically, the penalty is large,
    requiring a stronger signal to justify the trade.

    λ = 0.3 means a signal must exceed 30% of its change to be net positive.
    This naturally reduces whipsawing.
    """
    if not previous_signal:
        return current_signal

    penalized = {}
    for ticker, signal in current_signal.items():
        prev = previous_signal.get(ticker, 0.0)
        change = abs(signal - prev)
        penalty = penalty_lambda * change
        penalized[ticker] = round(signal - np.sign(signal) * penalty, 6)

    return penalized


# ═══════════════════════════════════════════════════════════════════════
# Comprehensive Signal Quality Report
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SignalQualityReport:
    """Quality assessment of the current signal set."""
    n_sectors: int
    n_stationary: int                  # Sectors passing ADF test
    n_tradeable: int                   # Stationary + HL in sweet spot + Hurst < 0.5
    avg_adf_pvalue: float
    avg_half_life: float
    avg_hurst: float
    stationarity_results: Dict[str, StationarityResult]
    # Recommendations
    recommended_tickers: List[str]     # Tickers that pass all quality gates
    avoid_tickers: List[str]           # Tickers with poor stationarity


def signal_quality_audit(
    residual_levels: pd.DataFrame,
    sectors: List[str],
) -> SignalQualityReport:
    """
    Run stationarity and quality checks on all sector residuals.

    This should be called before trading to verify that the
    mean-reversion assumption holds for each sector.
    """
    results = {}
    for s in sectors:
        if s in residual_levels.columns:
            results[s] = test_stationarity(residual_levels[s], ticker=s)
        else:
            results[s] = StationarityResult(
                ticker=s, adf_stat=0, adf_pvalue=1,
                is_stationary=False, half_life=float("inf"),
                hurst=0.5, tradeable=False,
            )

    n_stat = sum(1 for r in results.values() if r.is_stationary)
    n_trade = sum(1 for r in results.values() if r.tradeable)
    avg_p = float(np.mean([r.adf_pvalue for r in results.values()])) if results else 1.0
    avg_hl = float(np.nanmean([r.half_life for r in results.values() if math.isfinite(r.half_life)])) if results else 0
    avg_h = float(np.mean([r.hurst for r in results.values()])) if results else 0.5

    recommended = [s for s, r in results.items() if r.tradeable]
    avoid = [s for s, r in results.items() if not r.is_stationary]

    return SignalQualityReport(
        n_sectors=len(sectors),
        n_stationary=n_stat,
        n_tradeable=n_trade,
        avg_adf_pvalue=round(avg_p, 4),
        avg_half_life=round(avg_hl, 1),
        avg_hurst=round(avg_h, 3),
        stationarity_results=results,
        recommended_tickers=recommended,
        avoid_tickers=avoid,
    )
