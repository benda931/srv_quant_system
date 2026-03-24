"""
analytics/performance_metrics.py
==================================
Extended performance and risk-adjusted return metrics for the
SRV Quantamental Decision Support System.

Supplements analytics/portfolio_risk.py with institutional-standard
performance measures:
  - Sortino Ratio (downside-deviation adjusted)
  - Information Ratio (active return / tracking error)
  - Tracking Error (annualized)
  - Omega Ratio (gain/loss probability-weighted ratio)
  - Max Drawdown Duration (longest recovery time)

All functions accept pandas Series of returns and are vectorized via numpy.

Ref: Sortino & van der Meer (1991) — Downside risk
Ref: Grinold & Kahn (2000) — Active Portfolio Management
Ref: Keating & Shadwick (2002) — Omega function
"""
from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_TRADING_DAYS: int = 252
_EPSILON: float = 1e-12


# ---------------------------------------------------------------------------
# Sortino Ratio
# ---------------------------------------------------------------------------

def compute_sortino_ratio(
    returns: pd.Series,
    target_return: float = 0.0,
    annualize: bool = True,
) -> float:
    """
    Sortino Ratio = (mean return - target) / downside deviation.

    Unlike Sharpe, only penalizes negative deviations below the target,
    making it more appropriate for asymmetric return distributions
    (common in short-vol strategies).

    Parameters
    ----------
    returns : pd.Series
        Period (typically daily) returns.
    target_return : float
        Minimum acceptable return per period (default 0).
    annualize : bool
        If True, annualizes using sqrt(252).

    Returns
    -------
    float
        Sortino ratio. Returns 0.0 if insufficient data or zero downside deviation.
    """
    if returns is None or len(returns) < 20:
        return 0.0

    clean = returns.dropna()
    if len(clean) < 20:
        return 0.0

    excess = clean - target_return
    downside = excess.clip(upper=0.0)
    downside_std = float(np.sqrt(np.mean(downside ** 2)))

    if downside_std < _EPSILON:
        return 0.0

    mean_excess = float(excess.mean())
    ratio = mean_excess / downside_std

    if annualize:
        ratio *= np.sqrt(_TRADING_DAYS)

    return round(ratio, 4)


# ---------------------------------------------------------------------------
# Information Ratio
# ---------------------------------------------------------------------------

def compute_information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    annualize: bool = True,
) -> float:
    """
    Information Ratio = mean(active return) / std(active return).

    Measures the consistency of excess returns over the benchmark.
    IR > 0.5 is generally considered good; IR > 1.0 is exceptional.

    Parameters
    ----------
    returns : pd.Series
        Strategy returns (daily).
    benchmark_returns : pd.Series
        Benchmark returns (daily), same index alignment expected.
    annualize : bool
        If True, annualizes using sqrt(252).

    Returns
    -------
    float
        Information ratio. Returns 0.0 if insufficient data.
    """
    if returns is None or benchmark_returns is None:
        return 0.0

    # Align indices
    aligned = pd.DataFrame({"strat": returns, "bench": benchmark_returns}).dropna()
    if len(aligned) < 20:
        return 0.0

    active = aligned["strat"] - aligned["bench"]
    te = float(active.std(ddof=1))

    if te < _EPSILON:
        return 0.0

    ratio = float(active.mean()) / te

    if annualize:
        ratio *= np.sqrt(_TRADING_DAYS)

    return round(ratio, 4)


# ---------------------------------------------------------------------------
# Tracking Error
# ---------------------------------------------------------------------------

def compute_tracking_error(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    annualize: bool = True,
) -> float:
    """
    Tracking Error = std(active returns), annualized.

    Measures the volatility of the difference between strategy and benchmark
    returns. Used for risk budgeting and mandate compliance.

    Parameters
    ----------
    returns : pd.Series
        Strategy returns (daily).
    benchmark_returns : pd.Series
        Benchmark returns (daily).
    annualize : bool
        If True, annualizes using sqrt(252).

    Returns
    -------
    float
        Annualized tracking error. Returns 0.0 if insufficient data.
    """
    if returns is None or benchmark_returns is None:
        return 0.0

    aligned = pd.DataFrame({"strat": returns, "bench": benchmark_returns}).dropna()
    if len(aligned) < 20:
        return 0.0

    active = aligned["strat"] - aligned["bench"]
    te = float(active.std(ddof=1))

    if annualize:
        te *= np.sqrt(_TRADING_DAYS)

    return round(te, 6)


# ---------------------------------------------------------------------------
# Omega Ratio
# ---------------------------------------------------------------------------

def compute_omega_ratio(
    returns: pd.Series,
    threshold: float = 0.0,
) -> float:
    """
    Omega Ratio = sum of gains above threshold / sum of losses below threshold.

    A ratio > 1.0 means the strategy generates more gain above the threshold
    than loss below it. Unlike Sharpe/Sortino, Omega captures the full
    distribution shape (all moments).

    Parameters
    ----------
    returns : pd.Series
        Period returns.
    threshold : float
        The return threshold (default 0).

    Returns
    -------
    float
        Omega ratio. Returns 0.0 if insufficient data or no losses.

    Ref: Keating & Shadwick (2002) — A Universal Performance Measure
    """
    if returns is None or len(returns) < 20:
        return 0.0

    clean = returns.dropna()
    if len(clean) < 20:
        return 0.0

    excess = clean - threshold
    gains = float(excess[excess > 0].sum())
    losses = float(-excess[excess <= 0].sum())

    if losses < _EPSILON:
        # No losses — infinite omega, cap at a large number
        return 999.0 if gains > 0 else 0.0

    return round(gains / losses, 4)


# ---------------------------------------------------------------------------
# Max Drawdown Duration
# ---------------------------------------------------------------------------

def compute_max_drawdown_duration(
    equity_curve: pd.Series,
) -> int:
    """
    Maximum drawdown duration in trading days.

    This is the longest period from a peak to recovery back to (or above)
    that peak. If the strategy has not recovered from the deepest drawdown,
    the duration counts from the peak to the end of the series.

    Parameters
    ----------
    equity_curve : pd.Series
        Cumulative equity curve (e.g., cumulative returns + 1, or portfolio value).
        Must be non-negative.

    Returns
    -------
    int
        Maximum drawdown duration in periods (trading days).
        Returns 0 if insufficient data.
    """
    if equity_curve is None or len(equity_curve) < 2:
        return 0

    clean = equity_curve.dropna()
    if len(clean) < 2:
        return 0

    values = clean.values.astype(float)

    # Running maximum
    running_max = np.maximum.accumulate(values)

    max_duration = 0
    current_duration = 0

    for i in range(1, len(values)):
        if values[i] < running_max[i]:
            # In drawdown
            current_duration += 1
        else:
            # Recovered or at new high
            max_duration = max(max_duration, current_duration)
            current_duration = 0

    # Check if still in drawdown at end of series
    max_duration = max(max_duration, current_duration)

    return max_duration


# ---------------------------------------------------------------------------
# Convenience: compute all metrics at once
# ---------------------------------------------------------------------------

def compute_all_performance_metrics(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    target_return: float = 0.0,
) -> dict:
    """
    Compute all performance metrics in a single call.

    Parameters
    ----------
    returns : pd.Series
        Strategy daily returns.
    benchmark_returns : pd.Series or None
        Benchmark daily returns (required for IR and TE).
    target_return : float
        Threshold for Sortino and Omega.

    Returns
    -------
    dict
        Dictionary with all metric values.
    """
    # Build equity curve from returns
    equity = (1 + returns.fillna(0)).cumprod()

    result = {
        "sortino_ratio": compute_sortino_ratio(returns, target_return=target_return),
        "omega_ratio": compute_omega_ratio(returns, threshold=target_return),
        "max_drawdown_duration_days": compute_max_drawdown_duration(equity),
    }

    if benchmark_returns is not None:
        result["information_ratio"] = compute_information_ratio(returns, benchmark_returns)
        result["tracking_error"] = compute_tracking_error(returns, benchmark_returns)
    else:
        result["information_ratio"] = None
        result["tracking_error"] = None

    return result
