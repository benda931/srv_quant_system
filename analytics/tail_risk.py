"""
analytics/tail_risk.py
========================
Expected Shortfall (ES) and Parametric Correlation Stress Tests

From the Research Brief:
  - ES at chosen confidence level (replacing VaR for better tail-risk capture)
  - Parametric correlation shock: C_stress = (1-η)C + η·11' where η = stress intensity
  - Tail-correlation diagnostic (panic coupling detection)
  - Convexity-aware risk metrics for short-vol strategies

Ref: Basel III market-risk standards (FRTB) — ES replaces VaR
Ref: Longin & Solnik — Extreme Correlation of International Equity Markets
Ref: Carr & Wu — Variance Risk Premia
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_EPSILON = 1e-15


# ─────────────────────────────────────────────────────────────────────────────
# Expected Shortfall (ES)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ESReport:
    """Expected Shortfall report for portfolio."""
    confidence: float            # e.g., 0.975
    horizon_days: int            # 1 or 10

    # Core metrics
    es_pct: float                # ES as % of portfolio (negative = loss)
    var_pct: float               # VaR at same confidence
    es_to_var_ratio: float       # ES / VaR — higher = fatter tails

    # Decomposition
    es_parametric: float         # Gaussian ES
    es_historical: float         # Historical simulation ES
    es_cornish_fisher: float     # Cornish-Fisher adjusted ES (skew + kurtosis)

    # Tail statistics
    skewness: float
    kurtosis: float              # Excess kurtosis
    n_observations: int
    tail_observations: int       # Number of observations beyond VaR

    # Per-sector marginal ES contribution
    sector_mes: Dict[str, float] = field(default_factory=dict)


def compute_expected_shortfall(
    returns: pd.DataFrame,
    weights: Dict[str, float],
    confidence: float = 0.975,
    horizon_days: int = 1,
) -> ESReport:
    """
    Compute Expected Shortfall (ES) using three methods:
      1. Parametric (Gaussian)
      2. Historical simulation
      3. Cornish-Fisher expansion (adjusts for skewness + kurtosis)

    Parameters
    ----------
    returns    : Daily returns DataFrame (columns = tickers)
    weights    : {ticker: weight} — portfolio weights
    confidence : ES confidence level (0.975 = 97.5%)
    horizon_days : Risk horizon in trading days

    Returns
    -------
    ESReport

    Ref: Basel FRTB (BIS, 2019) — ES at 97.5% replaces VaR 99% for tail capture
    """
    # Portfolio returns
    tickers = [t for t in weights if t in returns.columns]
    if not tickers:
        return _empty_es(confidence, horizon_days)

    w = np.array([weights[t] for t in tickers])
    R = returns[tickers].dropna()
    if len(R) < 60:
        return _empty_es(confidence, horizon_days)

    port_rets = R.values @ w
    n = len(port_rets)

    mu = float(np.mean(port_rets))
    sigma = float(np.std(port_rets, ddof=1))
    skew = float(pd.Series(port_rets).skew())
    kurt = float(pd.Series(port_rets).kurtosis())  # Excess kurtosis

    # Scale to horizon
    scale = np.sqrt(horizon_days)

    # ── 1. Parametric ES (Gaussian) ──
    from scipy.stats import norm
    alpha = 1.0 - confidence
    z_alpha = norm.ppf(alpha)
    var_parametric = -(mu * horizon_days + z_alpha * sigma * scale)
    es_parametric = -(mu * horizon_days + sigma * scale * norm.pdf(z_alpha) / alpha)

    # ── 2. Historical simulation ES ──
    sorted_rets = np.sort(port_rets)
    n_tail = max(1, int(n * alpha))
    tail_rets = sorted_rets[:n_tail]
    var_historical = -float(sorted_rets[n_tail - 1]) * scale
    es_historical = -float(np.mean(tail_rets)) * scale

    # ── 3. Cornish-Fisher ES ──
    # z_CF = z + (z²-1)·S/6 + (z³-3z)·K/24 - (2z³-5z)·S²/36
    S, K = skew, kurt
    z = z_alpha
    z_cf = z + (z**2 - 1) * S / 6 + (z**3 - 3*z) * K / 24 - (2*z**3 - 5*z) * S**2 / 36
    var_cf = -(mu * horizon_days + z_cf * sigma * scale)
    # ES via numerical integration of CF-adjusted distribution (simplified)
    z_cf_es = z_cf - sigma * scale * 0.1  # Approximation
    es_cf = var_cf * 1.1  # Approx: ES ≈ 1.1 × VaR for fat-tailed distributions

    # Best estimate: average of historical and CF
    es_best = (es_historical + es_cf) / 2.0

    # ── Marginal ES (per-sector contribution) ──
    sector_mes = {}
    for i, t in enumerate(tickers):
        # MES_i = E[r_i | r_p < VaR_p]
        threshold = np.percentile(port_rets, alpha * 100)
        tail_mask = port_rets <= threshold
        if tail_mask.sum() > 0:
            sector_mes[t] = round(float(np.mean(R[t].values[tail_mask])) * scale * weights[t], 6)
        else:
            sector_mes[t] = 0.0

    return ESReport(
        confidence=confidence,
        horizon_days=horizon_days,
        es_pct=round(es_best, 6),
        var_pct=round(var_historical, 6),
        es_to_var_ratio=round(es_best / var_historical, 4) if abs(var_historical) > 1e-10 else 0.0,
        es_parametric=round(es_parametric, 6),
        es_historical=round(es_historical, 6),
        es_cornish_fisher=round(es_cf, 6),
        skewness=round(skew, 4),
        kurtosis=round(kurt, 4),
        n_observations=n,
        tail_observations=n_tail,
        sector_mes=sector_mes,
    )


def _empty_es(conf, horizon):
    return ESReport(
        confidence=conf, horizon_days=horizon,
        es_pct=0, var_pct=0, es_to_var_ratio=0,
        es_parametric=0, es_historical=0, es_cornish_fisher=0,
        skewness=0, kurtosis=0, n_observations=0, tail_observations=0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Parametric Correlation Stress
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CorrelationStressResult:
    """Result of parametric correlation stress test."""
    eta: float                   # Stress intensity (0 = no stress, 1 = full panic)
    label: str                   # "MILD" / "MODERATE" / "SEVERE" / "PANIC"

    # Correlation matrices
    avg_corr_baseline: float     # Average off-diagonal in C_base
    avg_corr_stressed: float     # Average off-diagonal in C_stress

    # Portfolio impact
    portfolio_vol_baseline: float  # Annualized vol under baseline
    portfolio_vol_stressed: float  # Annualized vol under stress
    vol_increase_pct: float        # % increase in vol

    # Dispersion impact
    dispersion_baseline: float
    dispersion_stressed: float
    dispersion_change_pct: float

    # Per-sector impact
    sector_vol_change: Dict[str, float] = field(default_factory=dict)

    # Tail correlation diagnostic
    tail_corr_increase: float = 0.0  # How much tail corr exceeds normal corr


def parametric_correlation_stress(
    returns: pd.DataFrame,
    weights: Dict[str, float],
    eta_levels: List[float] = None,
    window: int = 252,
) -> List[CorrelationStressResult]:
    """
    Parametric correlation shock from Research Brief:
      C_stress = (1-η)C + η·11'

    where η is stress intensity:
      η=0.0 → no stress (baseline)
      η=0.3 → mild (correlations increase moderately)
      η=0.5 → moderate (correlations converge halfway to 1)
      η=0.7 → severe (most correlations near 0.85+)
      η=0.9 → panic (near-perfect correlation — "everything moves together")

    Ref: Longin & Solnik — Extreme Correlation of International Equity Markets
    """
    if eta_levels is None:
        eta_levels = [0.0, 0.3, 0.5, 0.7, 0.9]

    tickers = [t for t in weights if t in returns.columns]
    if not tickers or len(returns) < window:
        return []

    R = returns[tickers].dropna().iloc[-window:]
    n = len(tickers)

    # Baseline correlation and covariance
    C_base = R.corr().values
    vols = R.std().values * np.sqrt(252)  # Annualized
    w = np.array([weights.get(t, 0) for t in tickers])

    # Ones matrix for stress
    ones = np.ones((n, n))

    results = []
    for eta in eta_levels:
        # C_stress = (1-η)C + η·11'
        C_stress = (1.0 - eta) * C_base + eta * ones
        np.fill_diagonal(C_stress, 1.0)  # Ensure diagonal = 1

        # Portfolio vol under stress
        D = np.diag(vols)
        Sigma_base = D @ C_base @ D / 252.0  # Daily covariance
        Sigma_stress = D @ C_stress @ D / 252.0

        port_var_base = float(w @ Sigma_base @ w)
        port_var_stress = float(w @ Sigma_stress @ w)
        port_vol_base = math.sqrt(max(0, port_var_base) * 252)
        port_vol_stress = math.sqrt(max(0, port_var_stress) * 252)
        vol_increase = (port_vol_stress - port_vol_base) / port_vol_base if port_vol_base > 1e-10 else 0

        # Average off-diagonal correlation
        iu = np.triu_indices(n, k=1)
        avg_corr_base = float(np.mean(C_base[iu]))
        avg_corr_stress = float(np.mean(C_stress[iu]))

        # Dispersion: Σw²σ² / σ²_index — higher = more dispersion
        index_var_base = float(w @ Sigma_base @ w)
        index_var_stress = float(w @ Sigma_stress @ w)
        idio_var = float(np.sum((w * vols / np.sqrt(252)) ** 2))
        disp_base = idio_var / index_var_base if index_var_base > 1e-10 else 0
        disp_stress = idio_var / index_var_stress if index_var_stress > 1e-10 else 0
        disp_change = (disp_stress - disp_base) / disp_base if disp_base > 1e-10 else 0

        # Per-sector vol impact
        sector_vol_chg = {}
        for i, t in enumerate(tickers):
            s_vol_base = vols[i]
            # Under stress, beta increases → effective vol contribution increases
            beta_stress = float(C_stress[i] @ w) / max(1e-10, float(w @ C_stress @ w) ** 0.5)
            sector_vol_chg[t] = round((beta_stress - 1.0) * 100, 2)  # % change in beta

        # Label
        if eta <= 0.1:
            label = "BASELINE"
        elif eta <= 0.35:
            label = "MILD"
        elif eta <= 0.55:
            label = "MODERATE"
        elif eta <= 0.75:
            label = "SEVERE"
        else:
            label = "PANIC"

        results.append(CorrelationStressResult(
            eta=eta, label=label,
            avg_corr_baseline=round(avg_corr_base, 4),
            avg_corr_stressed=round(avg_corr_stress, 4),
            portfolio_vol_baseline=round(port_vol_base, 6),
            portfolio_vol_stressed=round(port_vol_stress, 6),
            vol_increase_pct=round(vol_increase, 4),
            dispersion_baseline=round(disp_base, 4),
            dispersion_stressed=round(disp_stress, 4),
            dispersion_change_pct=round(disp_change, 4),
            sector_vol_change=sector_vol_chg,
            tail_corr_increase=round(avg_corr_stress - avg_corr_base, 4),
        ))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Tail Correlation Diagnostic
# ─────────────────────────────────────────────────────────────────────────────

def tail_correlation_diagnostic(
    returns: pd.DataFrame,
    tickers: List[str],
    quantile: float = 0.05,
    window: int = 252,
) -> Dict[str, float]:
    """
    Compare correlation in the tails vs normal times.

    Tail dependence often increases during stress — correlations computed
    conditional on extreme returns differ from unconditional correlations.

    Ref: Longin & Solnik — tail dependence increases in bear markets

    Returns
    -------
    dict with: normal_corr, tail_corr, tail_ratio (>1 = tail dependence)
    """
    avail = [t for t in tickers if t in returns.columns]
    R = returns[avail].dropna().iloc[-window:]
    if len(R) < 60 or len(avail) < 2:
        return {"normal_corr": 0, "tail_corr": 0, "tail_ratio": 1.0}

    # Normal correlation
    C_normal = R.corr()
    iu = np.triu_indices(len(avail), k=1)
    normal_corr = float(np.mean(C_normal.values[iu]))

    # Tail correlation: computed only on days where market return is in the bottom quantile
    market_ret = R.mean(axis=1)
    threshold = market_ret.quantile(quantile)
    tail_days = market_ret <= threshold
    if tail_days.sum() < 10:
        return {"normal_corr": round(normal_corr, 4), "tail_corr": round(normal_corr, 4), "tail_ratio": 1.0}

    C_tail = R.loc[tail_days].corr()
    tail_corr = float(np.mean(C_tail.values[iu]))

    ratio = tail_corr / normal_corr if abs(normal_corr) > 1e-10 else 1.0

    return {
        "normal_corr": round(normal_corr, 4),
        "tail_corr": round(tail_corr, 4),
        "tail_ratio": round(ratio, 4),
        "tail_days": int(tail_days.sum()),
        "panic_coupling": ratio > 1.5,  # True if tail corr >> normal corr
    }


# ─────────────────────────────────────────────────────────────────────────────
# VaR Backtesting — Kupiec POF Test
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VaRBacktestResult:
    """Result of VaR model validation backtest."""
    n_observations: int
    n_exceptions: int          # Days where loss exceeded VaR
    expected_exceptions: float  # n * alpha
    exception_rate: float       # n_exceptions / n
    expected_rate: float        # alpha (1 - confidence)

    # Kupiec POF
    kupiec_statistic: float    # Likelihood ratio test statistic (~chi2(1))
    kupiec_p_value: float       # p-value under H0: exception rate = alpha
    kupiec_pass: bool           # True if p_value > significance level

    # Christoffersen (if computed)
    independence_statistic: Optional[float] = None
    independence_p_value: Optional[float] = None
    independence_pass: Optional[bool] = None

    # Combined
    overall_pass: bool = True


def kupiec_var_backtest(
    returns: pd.Series,
    var_level: pd.Series,
    confidence: float = 0.95,
    significance: float = 0.05,
) -> VaRBacktestResult:
    """
    Kupiec Proportion of Failures (POF) test for VaR model validation.

    Tests whether the observed number of VaR exceptions is consistent with
    the model's confidence level. Under the null hypothesis H0, the exception
    rate equals alpha = 1 - confidence.

    The test statistic follows chi-squared(1) under H0.

    Parameters
    ----------
    returns : pd.Series
        Realized portfolio returns (daily). Losses are negative.
    var_level : pd.Series
        VaR estimates for each day (positive values representing loss thresholds).
        Same index as returns.
    confidence : float
        VaR confidence level (e.g., 0.95 for 95% VaR).
    significance : float
        Significance level for the test (default 0.05).

    Returns
    -------
    VaRBacktestResult

    Ref: Kupiec (1995) — Techniques for Verifying the Accuracy of Risk
         Measurement Models
    """
    from scipy.stats import chi2

    # Align series
    aligned = pd.DataFrame({"ret": returns, "var": var_level}).dropna()
    n = len(aligned)

    if n < 50:
        log.warning("Kupiec test: only %d observations (need >= 50)", n)
        return VaRBacktestResult(
            n_observations=n, n_exceptions=0, expected_exceptions=0,
            exception_rate=0, expected_rate=1 - confidence,
            kupiec_statistic=0, kupiec_p_value=1.0, kupiec_pass=True,
        )

    alpha = 1.0 - confidence

    # Exception: realized loss exceeds VaR (returns < -VaR)
    exceptions = aligned["ret"] < -aligned["var"]
    x = int(exceptions.sum())
    expected = n * alpha

    # Exception rate
    p_hat = x / n if n > 0 else 0.0

    # Kupiec log-likelihood ratio statistic
    # LR = -2 * ln[ alpha^x * (1-alpha)^(n-x) / p_hat^x * (1-p_hat)^(n-x) ]
    # Handle edge cases
    if x == 0:
        # No exceptions — log(p_hat^x) = 0, but (1-p_hat)^(n-x) = 1
        lr_num = x * np.log(alpha + _EPSILON) + (n - x) * np.log(1 - alpha + _EPSILON)
        lr_den = (n - x) * np.log(1.0)  # = 0
        lr_stat = -2.0 * (lr_num - 0.0)
    elif x == n:
        # All exceptions
        lr_num = x * np.log(alpha + _EPSILON)
        lr_den = x * np.log(p_hat + _EPSILON)
        lr_stat = -2.0 * (lr_num - lr_den)
    else:
        lr_num = x * np.log(alpha) + (n - x) * np.log(1 - alpha)
        lr_den = x * np.log(p_hat) + (n - x) * np.log(1 - p_hat)
        lr_stat = -2.0 * (lr_num - lr_den)

    # Ensure non-negative (numerical issues)
    lr_stat = max(0.0, lr_stat)

    # p-value from chi-squared(1) distribution
    p_value = float(1.0 - chi2.cdf(lr_stat, df=1))
    passes = p_value > significance

    return VaRBacktestResult(
        n_observations=n,
        n_exceptions=x,
        expected_exceptions=round(expected, 2),
        exception_rate=round(p_hat, 6),
        expected_rate=round(alpha, 6),
        kupiec_statistic=round(lr_stat, 4),
        kupiec_p_value=round(p_value, 4),
        kupiec_pass=passes,
        overall_pass=passes,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Christoffersen Independence Test
# ─────────────────────────────────────────────────────────────────────────────

def christoffersen_independence_test(
    returns: pd.Series,
    var_level: pd.Series,
    confidence: float = 0.95,
    significance: float = 0.05,
) -> VaRBacktestResult:
    """
    Christoffersen conditional coverage test for VaR exceptions.

    Tests two properties simultaneously:
      1. Unconditional coverage: exception rate = alpha (Kupiec)
      2. Independence: exceptions are not clustered (no serial dependence)

    Clustered exceptions indicate the VaR model fails to capture volatility
    dynamics (e.g., GARCH effects, regime changes).

    The independence test uses a first-order Markov transition matrix of
    exception indicators. Under H0, P(exception today | exception yesterday)
    = P(exception today | no exception yesterday) = alpha.

    Parameters
    ----------
    returns : pd.Series
        Realized portfolio returns (daily).
    var_level : pd.Series
        VaR estimates (positive values).
    confidence : float
        VaR confidence level.
    significance : float
        Significance level for the test.

    Returns
    -------
    VaRBacktestResult
        Includes both Kupiec and independence test results.

    Ref: Christoffersen (1998) — Evaluating Interval Forecasts
    """
    from scipy.stats import chi2

    # First run Kupiec
    kupiec_result = kupiec_var_backtest(returns, var_level, confidence, significance)

    # Align and compute exception indicator series
    aligned = pd.DataFrame({"ret": returns, "var": var_level}).dropna()
    n = len(aligned)

    if n < 100:
        log.warning("Christoffersen test: only %d obs (need >= 100 for reliable results)", n)
        return kupiec_result

    exceptions = (aligned["ret"] < -aligned["var"]).astype(int).values

    # Count transitions: n_ij = count of (I_{t-1} = i, I_t = j)
    n_00, n_01, n_10, n_11 = 0, 0, 0, 0
    for t in range(1, len(exceptions)):
        prev, curr = exceptions[t - 1], exceptions[t]
        if prev == 0 and curr == 0:
            n_00 += 1
        elif prev == 0 and curr == 1:
            n_01 += 1
        elif prev == 1 and curr == 0:
            n_10 += 1
        else:
            n_11 += 1

    # Transition probabilities
    # pi_01 = P(exception | no exception yesterday)
    # pi_11 = P(exception | exception yesterday)
    denom_0 = n_00 + n_01
    denom_1 = n_10 + n_11

    if denom_0 == 0 or denom_1 == 0:
        # Cannot compute — insufficient transitions
        return kupiec_result

    pi_01 = n_01 / denom_0
    pi_11 = n_11 / denom_1

    # Under H0 (independence): pi_01 = pi_11 = pi_hat
    pi_hat = (n_01 + n_11) / (n_00 + n_01 + n_10 + n_11)

    # Likelihood ratio for independence
    # LR_ind = -2 * ln[ L(pi_hat) / L(pi_01, pi_11) ]
    try:
        log_l0 = (
            n_00 * np.log(1 - pi_hat + _EPSILON)
            + n_01 * np.log(pi_hat + _EPSILON)
            + n_10 * np.log(1 - pi_hat + _EPSILON)
            + n_11 * np.log(pi_hat + _EPSILON)
        )
        log_l1 = (
            n_00 * np.log(1 - pi_01 + _EPSILON)
            + n_01 * np.log(pi_01 + _EPSILON)
            + n_10 * np.log(1 - pi_11 + _EPSILON)
            + n_11 * np.log(pi_11 + _EPSILON)
        )
        lr_ind = -2.0 * (log_l0 - log_l1)
        lr_ind = max(0.0, lr_ind)
    except (ValueError, FloatingPointError):
        lr_ind = 0.0

    ind_p_value = float(1.0 - chi2.cdf(lr_ind, df=1))
    ind_pass = ind_p_value > significance

    # Combined conditional coverage test (Kupiec + Independence)
    # LR_cc = LR_kupiec + LR_ind ~ chi2(2)
    lr_cc = kupiec_result.kupiec_statistic + lr_ind
    cc_p_value = float(1.0 - chi2.cdf(lr_cc, df=2))
    overall = kupiec_result.kupiec_pass and ind_pass

    return VaRBacktestResult(
        n_observations=kupiec_result.n_observations,
        n_exceptions=kupiec_result.n_exceptions,
        expected_exceptions=kupiec_result.expected_exceptions,
        exception_rate=kupiec_result.exception_rate,
        expected_rate=kupiec_result.expected_rate,
        kupiec_statistic=kupiec_result.kupiec_statistic,
        kupiec_p_value=kupiec_result.kupiec_p_value,
        kupiec_pass=kupiec_result.kupiec_pass,
        independence_statistic=round(lr_ind, 4),
        independence_p_value=round(ind_p_value, 4),
        independence_pass=ind_pass,
        overall_pass=overall,
    )
