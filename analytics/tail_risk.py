"""
analytics/tail_risk.py
========================
Expected Shortfall (ES), Extreme Value Theory (EVT), and Tail Risk Analytics

Full tail-risk measurement suite for a Short Vol / Dispersion DSS:

  1. Expected Shortfall (ES) — parametric, historical, Cornish-Fisher
  2. Parametric correlation stress: C_stress = (1-η)C + η·11'
  3. Tail-correlation diagnostic (panic coupling detection)
  4. VaR backtesting — Kupiec + Christoffersen conditional coverage
  5. EVT: Peaks-over-Threshold (POT) + Generalized Pareto Distribution (GPD)
  6. Hill tail index estimator (power-law tail fatness)
  7. Regime-conditional tail metrics (per CALM/NORMAL/TENSION/CRISIS)
  8. Short-vol specific tail analysis: convexity P&L, gap risk, vol-of-vol

Ref: Basel III market-risk standards (FRTB) — ES replaces VaR
Ref: Longin & Solnik — Extreme Correlation of International Equity Markets
Ref: Carr & Wu — Variance Risk Premia
Ref: McNeil & Frey (2000) — Estimation of tail-related risk measures (EVT)
Ref: Pickands (1975) — Generalized Pareto Distribution
Ref: Hill (1975) — Tail index estimator for heavy-tailed distributions
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
    # Ref: Cornish & Fisher (1937), Favre & Galeano (2002)
    S, K = skew, kurt
    z = z_alpha
    z_cf = z + (z**2 - 1) * S / 6 + (z**3 - 3*z) * K / 24 - (2*z**3 - 5*z) * S**2 / 36
    var_cf = -(mu * horizon_days + z_cf * sigma * scale)
    # ES via numerical integration over CF-adjusted quantile function
    # ES = (1/α) ∫₀^α VaR(p) dp — approximate with 100-point quadrature
    _n_quad = 100
    _ps = np.linspace(0.001, alpha, _n_quad)
    _es_sum = 0.0
    for _p in _ps:
        _zp = norm.ppf(_p)
        _zcf_p = _zp + (_zp**2 - 1) * S / 6 + (_zp**3 - 3*_zp) * K / 24 - (2*_zp**3 - 5*_zp) * S**2 / 36
        _es_sum += -(mu * horizon_days + _zcf_p * sigma * scale)
    es_cf = _es_sum / _n_quad

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


# ═════════════════════════════════════════════════════════════════════════════
# EXTREME VALUE THEORY — Peaks-over-Threshold + GPD
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class EVTResult:
    """
    Extreme Value Theory result using Peaks-over-Threshold (POT) method
    with Generalized Pareto Distribution (GPD) fitted to exceedances.

    The GPD models the distribution of losses that exceed a high threshold u:
        F_u(y) = 1 - (1 + ξ·y/β)^{-1/ξ}   for ξ ≠ 0
    where:
        ξ (shape/xi) — tail index; ξ > 0 = heavy tail (Pareto), ξ = 0 = exponential, ξ < 0 = finite tail
        β (scale/sigma) — scale parameter
        u — threshold (e.g., 95th percentile of losses)
    """
    # GPD parameters
    xi: float                       # Shape parameter (tail index)
    beta: float                     # Scale parameter
    threshold: float                # Threshold u (absolute value of loss)
    n_exceedances: int              # Number of observations beyond threshold
    n_total: int

    # EVT-derived risk measures
    evt_var_95: float               # VaR at 95% from GPD
    evt_var_99: float               # VaR at 99% from GPD
    evt_es_95: float                # ES at 95% from GPD (if ξ < 1)
    evt_es_99: float                # ES at 99% from GPD (if ξ < 1)

    # Tail classification
    tail_type: str                  # "heavy" (ξ > 0.1), "medium" (0 < ξ ≤ 0.1), "thin" (ξ ≤ 0)
    tail_warning: str               # PM-facing interpretation

    # Goodness-of-fit
    ks_statistic: float = 0.0      # Kolmogorov-Smirnov test statistic
    ks_p_value: float = 0.0        # p-value (> 0.05 = GPD fits well)

    # Per-sector EVT (optional)
    sector_xi: Dict[str, float] = field(default_factory=dict)


def fit_evt_pot(
    returns: pd.DataFrame,
    weights: Dict[str, float],
    threshold_quantile: float = 0.95,
) -> EVTResult:
    """
    Fit Generalized Pareto Distribution to portfolio loss exceedances
    using the Peaks-over-Threshold (POT) method.

    Parameters
    ----------
    returns : pd.DataFrame — daily returns (columns = tickers)
    weights : Dict[str, float] — portfolio weights
    threshold_quantile : float — quantile for threshold selection (default 0.95)

    Returns
    -------
    EVTResult
    """
    # Portfolio returns
    tickers = [t for t in weights if t in returns.columns and abs(weights[t]) > 1e-8]
    if not tickers:
        return _empty_evt()

    w = np.array([weights[t] for t in tickers])
    r = returns[tickers].dropna()
    if len(r) < 100:
        return _empty_evt()

    port_ret = r.values @ w
    n_total = len(port_ret)

    # Work with losses (negate returns)
    losses = -port_ret

    # Threshold: quantile of losses
    u = float(np.quantile(losses, threshold_quantile))
    if u <= 0:
        u = float(np.quantile(losses, 0.90))  # fallback to 90th pct
    if u <= 0:
        return _empty_evt()

    # Exceedances: losses above threshold
    exceedances = losses[losses > u] - u
    n_exc = len(exceedances)

    if n_exc < 15:
        log.warning("EVT: only %d exceedances (need >= 15) — results unreliable", n_exc)
        if n_exc < 5:
            return _empty_evt()

    # Fit GPD via Maximum Likelihood
    xi, beta = _fit_gpd_mle(exceedances)

    # EVT-derived VaR and ES
    # VaR_p = u + (β/ξ) * [(n/N_u * (1-p))^{-ξ} - 1]  for ξ ≠ 0
    exc_rate = n_exc / n_total

    evt_var_95 = _gpd_var(0.95, u, xi, beta, exc_rate)
    evt_var_99 = _gpd_var(0.99, u, xi, beta, exc_rate)

    # ES_p = VaR_p / (1-ξ) + (β - ξ·u) / (1-ξ)   for ξ < 1
    evt_es_95 = _gpd_es(0.95, evt_var_95, xi, beta, u) if xi < 1.0 else float("nan")
    evt_es_99 = _gpd_es(0.99, evt_var_99, xi, beta, u) if xi < 1.0 else float("nan")

    # Tail classification
    if xi > 0.25:
        tail_type = "heavy"
        tail_warning = f"HEAVY tail (ξ={xi:.3f}) — fat-tail risk significantly exceeds Gaussian model. Short-vol positions require wider stops."
    elif xi > 0.10:
        tail_type = "heavy"
        tail_warning = f"Moderately heavy tail (ξ={xi:.3f}) — tail losses will exceed Gaussian estimates by ~{(1+xi)*100-100:.0f}%."
    elif xi > 0:
        tail_type = "medium"
        tail_warning = f"Slightly heavy tail (ξ={xi:.3f}) — near-Gaussian but not exactly. Standard risk models approximately valid."
    else:
        tail_type = "thin"
        tail_warning = f"Thin/bounded tail (ξ={xi:.3f}) — tail risk well-contained. Gaussian model is conservative."

    # KS goodness-of-fit test
    ks_stat, ks_p = _gpd_ks_test(exceedances, xi, beta)

    # Per-sector tail index (Hill estimator on each sector)
    sector_xi: Dict[str, float] = {}
    for t in tickers:
        if t in r.columns:
            sec_losses = -r[t].values
            sec_xi = hill_estimator(sec_losses, k=max(15, int(len(sec_losses) * 0.05)))
            sector_xi[t] = round(sec_xi, 4)

    return EVTResult(
        xi=round(xi, 4),
        beta=round(beta, 6),
        threshold=round(-u, 6),  # Convert back to return space (negative = loss)
        n_exceedances=n_exc,
        n_total=n_total,
        evt_var_95=round(-evt_var_95, 6),  # Negative = loss
        evt_var_99=round(-evt_var_99, 6),
        evt_es_95=round(-evt_es_95, 6) if math.isfinite(evt_es_95) else float("nan"),
        evt_es_99=round(-evt_es_99, 6) if math.isfinite(evt_es_99) else float("nan"),
        tail_type=tail_type,
        tail_warning=tail_warning,
        ks_statistic=round(ks_stat, 4),
        ks_p_value=round(ks_p, 4),
        sector_xi=sector_xi,
    )


def _fit_gpd_mle(exceedances: np.ndarray) -> Tuple[float, float]:
    """
    Fit GPD parameters (ξ, β) via Maximum Likelihood.
    Uses scipy.stats.genpareto if available, else Grimshaw's MLE.
    """
    try:
        from scipy.stats import genpareto
        # genpareto uses (c, loc, scale) where c = ξ
        c, _loc, scale = genpareto.fit(exceedances, floc=0)
        return float(c), float(scale)
    except Exception:
        pass

    # Fallback: method-of-moments (Hosking & Wallis, 1987)
    n = len(exceedances)
    m1 = float(exceedances.mean())
    m2 = float(exceedances.var())
    if m1 <= 0:
        return 0.0, max(m1, 1e-8)
    xi = 0.5 * (m1 ** 2 / m2 - 1)
    beta = m1 * (1 - xi) / 2 if abs(1 - xi) > 1e-10 else m1
    xi = max(-0.5, min(xi, 2.0))
    beta = max(1e-8, beta)
    return float(xi), float(beta)


def _gpd_var(p: float, u: float, xi: float, beta: float, exc_rate: float) -> float:
    """GPD-derived VaR at confidence level p."""
    if exc_rate <= 0 or beta <= 0:
        return u
    if abs(xi) < 1e-10:
        # Exponential case (ξ → 0)
        return u + beta * np.log(exc_rate / (1 - p))
    return u + (beta / xi) * ((exc_rate / (1 - p)) ** xi - 1)


def _gpd_es(p: float, var_p: float, xi: float, beta: float, u: float) -> float:
    """GPD-derived Expected Shortfall at confidence level p (valid for ξ < 1)."""
    if xi >= 1.0:
        return float("nan")
    return var_p / (1 - xi) + (beta - xi * u) / (1 - xi)


def _gpd_ks_test(exceedances: np.ndarray, xi: float, beta: float) -> Tuple[float, float]:
    """Kolmogorov-Smirnov goodness-of-fit test for GPD."""
    try:
        from scipy.stats import genpareto, kstest
        stat, p = kstest(exceedances, genpareto.cdf, args=(xi, 0, beta))
        return float(stat), float(p)
    except Exception:
        return 0.0, 0.0


def _empty_evt() -> EVTResult:
    return EVTResult(
        xi=0.0, beta=0.0, threshold=0.0, n_exceedances=0, n_total=0,
        evt_var_95=0.0, evt_var_99=0.0, evt_es_95=0.0, evt_es_99=0.0,
        tail_type="unknown", tail_warning="Insufficient data for EVT analysis",
    )


# ═════════════════════════════════════════════════════════════════════════════
# Hill Tail Index Estimator
# ═════════════════════════════════════════════════════════════════════════════

def hill_estimator(losses: np.ndarray, k: Optional[int] = None) -> float:
    """
    Hill (1975) tail index estimator for heavy-tailed distributions.

    Estimates α such that P(X > x) ~ x^{-α} for large x.
    Returns ξ = 1/α (GPD shape parameter convention).

    Parameters
    ----------
    losses : np.ndarray — loss values (positive = loss)
    k : int — number of upper order statistics to use (default: 5% of n)

    Returns
    -------
    xi : float — estimated tail index (ξ = 1/α). Higher = heavier tail.
    """
    pos_losses = losses[losses > 0]
    n = len(pos_losses)
    if n < 30:
        return 0.0

    if k is None:
        k = max(15, int(n * 0.05))
    k = min(k, n - 1)

    sorted_losses = np.sort(pos_losses)[::-1]  # Descending
    x_k = sorted_losses[k]  # k-th order statistic
    if x_k <= 0:
        return 0.0

    # Hill estimator: H_k = (1/k) Σ_{i=1}^{k} log(X_{(i)} / X_{(k+1)})
    log_ratios = np.log(sorted_losses[:k] / x_k)
    h_k = float(log_ratios.mean())

    # ξ = H_k (Hill estimate is directly the GPD shape parameter)
    return max(0.0, h_k)


def hill_plot_data(
    losses: np.ndarray, k_range: Optional[Tuple[int, int]] = None,
) -> Tuple[List[int], List[float]]:
    """
    Generate data for Hill plot (ξ estimates vs k).
    Stable plateau indicates reliable ξ estimate.
    """
    pos_losses = losses[losses > 0]
    n = len(pos_losses)
    if n < 50:
        return [], []

    if k_range is None:
        k_range = (max(10, int(n * 0.02)), min(int(n * 0.20), n - 1))

    k_values = list(range(k_range[0], k_range[1], max(1, (k_range[1] - k_range[0]) // 50)))
    xi_values = [hill_estimator(losses, k=k) for k in k_values]
    return k_values, xi_values


# ═════════════════════════════════════════════════════════════════════════════
# Regime-Conditional Tail Metrics
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class RegimeTailResult:
    """Tail metrics computed per-regime."""
    regime: str
    n_days: int
    es_95: float                    # Historical ES at 95% in this regime
    max_daily_loss: float
    skewness: float
    kurtosis: float
    tail_ratio: float               # |ES| / |VaR| — tail concentration
    # EVT
    xi: float                       # GPD shape in this regime
    tail_type: str
    # Conditional metrics
    drawdown_given_tail: float      # Average drawdown following a tail event
    recovery_days: float            # Average days to recover from tail event


@dataclass
class RegimeConditionalTailReport:
    """Complete regime-conditional tail analysis."""
    regime_results: Dict[str, RegimeTailResult]
    worst_regime: str
    regime_tail_spread: float       # Ratio of worst to best regime ES
    warnings: List[str]


def regime_conditional_tails(
    returns: pd.DataFrame,
    weights: Dict[str, float],
    vix: Optional[pd.Series] = None,
    settings=None,
) -> RegimeConditionalTailReport:
    """
    Compute tail metrics for each regime (CALM/NORMAL/TENSION/CRISIS).
    Uses VIX levels to classify dates into regimes.
    """
    tickers = [t for t in weights if t in returns.columns and abs(weights[t]) > 1e-8]
    if not tickers or len(returns) < 200:
        return RegimeConditionalTailReport(
            regime_results={}, worst_regime="N/A",
            regime_tail_spread=0, warnings=["Insufficient data"],
        )

    w = np.array([weights[t] for t in tickers])
    port_ret = returns[tickers].dropna().values @ w

    # Classify regimes from VIX
    if vix is not None and len(vix) >= len(port_ret):
        vix_aligned = vix.iloc[-len(port_ret):].values
    else:
        vix_aligned = np.full(len(port_ret), 18.0)

    vix_soft = getattr(settings, "vix_level_soft", 21.0) if settings else 21.0
    vix_hard = getattr(settings, "vix_level_hard", 32.0) if settings else 32.0

    regimes = np.where(
        vix_aligned > vix_hard, "CRISIS",
        np.where(vix_aligned > vix_soft, "TENSION",
                 np.where(vix_aligned > 16, "NORMAL", "CALM"))
    )

    results: Dict[str, RegimeTailResult] = {}
    warnings: List[str] = []

    for regime in ["CALM", "NORMAL", "TENSION", "CRISIS"]:
        mask = regimes == regime
        r_regime = port_ret[mask]
        n_days = int(mask.sum())

        if n_days < 30:
            continue

        losses = -r_regime
        var_95 = float(np.percentile(losses, 95)) if n_days >= 20 else 0.0
        tail_mask = losses >= var_95 if var_95 > 0 else np.zeros(n_days, dtype=bool)
        es_95 = float(losses[tail_mask].mean()) if tail_mask.sum() > 0 else var_95

        skew = float(pd.Series(r_regime).skew())
        kurt = float(pd.Series(r_regime).kurtosis())
        max_loss = float(r_regime.min())

        tail_ratio = abs(es_95) / abs(var_95) if abs(var_95) > 1e-10 else 1.0

        # Mini EVT on this regime
        xi_regime = hill_estimator(losses, k=max(10, int(n_days * 0.05))) if n_days >= 50 else 0.0
        tail_type = "heavy" if xi_regime > 0.15 else "medium" if xi_regime > 0 else "thin"

        # Conditional drawdown after tail events
        dd_given_tail = 0.0
        recovery = 0.0
        if tail_mask.sum() >= 3:
            tail_indices = np.where(tail_mask)[0]
            dd_list = []
            rec_list = []
            for idx in tail_indices:
                if idx + 5 < n_days:
                    fwd = r_regime[idx + 1: idx + 6]
                    dd_list.append(float(fwd.min()))
                    rec_pos = np.where(np.cumsum(fwd) > 0)[0]
                    rec_list.append(int(rec_pos[0] + 1) if len(rec_pos) > 0 else 5)
            dd_given_tail = float(np.mean(dd_list)) if dd_list else 0.0
            recovery = float(np.mean(rec_list)) if rec_list else 0.0

        results[regime] = RegimeTailResult(
            regime=regime, n_days=n_days,
            es_95=round(-es_95, 6),
            max_daily_loss=round(max_loss, 6),
            skewness=round(skew, 3),
            kurtosis=round(kurt, 3),
            tail_ratio=round(tail_ratio, 3),
            xi=round(xi_regime, 4),
            tail_type=tail_type,
            drawdown_given_tail=round(dd_given_tail, 6),
            recovery_days=round(recovery, 1),
        )

    # Worst regime
    es_by_regime = {r: abs(v.es_95) for r, v in results.items()}
    worst_regime = max(es_by_regime, key=es_by_regime.get) if es_by_regime else "N/A"
    best_es = min(es_by_regime.values()) if es_by_regime else 1.0
    worst_es = max(es_by_regime.values()) if es_by_regime else 1.0
    tail_spread = worst_es / best_es if best_es > 1e-8 else float("inf")

    if "CRISIS" in results and results["CRISIS"].xi > 0.25:
        warnings.append(f"CRISIS regime has very heavy tail (ξ={results['CRISIS'].xi:.3f}) — short-vol positions highly exposed")
    if "TENSION" in results and results["TENSION"].es_95 < -0.03:
        warnings.append(f"TENSION ES 95% = {results['TENSION'].es_95:.2%} — consider reducing gross exposure during elevated VIX")
    if tail_spread > 3.0:
        warnings.append(f"Tail spread {tail_spread:.1f}x across regimes — regime-conditional sizing critical")

    return RegimeConditionalTailReport(
        regime_results=results,
        worst_regime=worst_regime,
        regime_tail_spread=round(tail_spread, 2),
        warnings=warnings,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Short-Vol Specific Tail Analysis
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class ShortVolTailResult:
    """Tail risk metrics specific to short-vol / dispersion strategies."""
    # Convexity P&L analysis
    gamma_pnl_tail: float           # Expected gamma P&L in tail events
    vega_pnl_tail: float            # Expected vega P&L in tail events
    convexity_drag: float           # Average daily convexity drag (negative = cost)

    # Gap risk
    overnight_gap_var95: float      # VaR95 of overnight gaps (close-to-open)
    max_overnight_gap: float        # Worst historical overnight gap
    gap_frequency: float            # Fraction of days with gap > 1%

    # Vol-of-vol metrics
    vvix_proxy: float               # Proxy for VVIX (vol-of-VIX, rolling std of VIX changes)
    vix_mean: float
    vix_convexity: float            # Kurtosis of VIX daily changes (>3 = dangerous for short vol)

    # Dispersion-specific
    corr_spike_severity: float      # Max 5-day increase in avg sector correlation
    corr_spike_pnl_impact: float    # Estimated P&L from worst corr spike

    warnings: List[str] = field(default_factory=list)


def short_vol_tail_analysis(
    prices: pd.DataFrame,
    weights: Dict[str, float],
    settings=None,
) -> ShortVolTailResult:
    """
    Analyse tail risk specific to short volatility / dispersion strategies.

    Covers: convexity drag, overnight gap risk, vol-of-vol, correlation spikes.
    """
    sectors = [t for t in weights if t in prices.columns]
    spy_col = "SPY" if "SPY" in prices.columns else None
    vix_col = next((c for c in prices.columns if "VIX" in c.upper()), None)

    warnings: List[str] = []

    # ── VIX / vol-of-vol metrics ─────────────────────────────────────────
    vix_mean = 18.0
    vvix_proxy = 0.0
    vix_convexity = 3.0

    if vix_col and vix_col in prices.columns:
        vix = prices[vix_col].dropna()
        if len(vix) >= 60:
            vix_mean = float(vix.iloc[-60:].mean())
            vix_changes = vix.diff().dropna()
            vvix_proxy = float(vix_changes.iloc[-60:].std())
            vix_convexity = float(vix_changes.kurtosis()) + 3  # excess + 3 = raw kurtosis
            if vvix_proxy > 3.0:
                warnings.append(f"VVIX proxy elevated ({vvix_proxy:.1f}) — vol-of-vol risk high for short-vol")

    # ── Overnight gap risk ───────────────────────────────────────────────
    overnight_gap_var95 = 0.0
    max_gap = 0.0
    gap_freq = 0.0

    if spy_col and spy_col in prices.columns:
        spy = prices[spy_col].dropna()
        if len(spy) > 2:
            # Proxy: daily open-to-close vs close-to-close gap
            daily_ret = spy.pct_change().dropna()
            # Use absolute value > 1% as "gap" proxy
            gaps = daily_ret[daily_ret.abs() > 0.02]  # >2% moves as gap proxy
            gap_freq = len(gaps) / len(daily_ret) if len(daily_ret) > 0 else 0
            if len(daily_ret) > 20:
                overnight_gap_var95 = float(np.percentile(-daily_ret.values, 95))
                max_gap = float(daily_ret.min())
            if max_gap < -0.05:
                warnings.append(f"Max gap event: {max_gap:.1%} — short-vol can realize multi-sigma loss on gaps")

    # ── Convexity P&L in tail events ─────────────────────────────────────
    gamma_pnl_tail = 0.0
    vega_pnl_tail = 0.0
    convexity_drag = 0.0

    if spy_col and spy_col in prices.columns:
        spy_ret = prices[spy_col].pct_change().dropna()
        if len(spy_ret) >= 100:
            # Convexity drag = E[r²] (realized variance per day)
            convexity_drag = -float((spy_ret ** 2).mean())  # Negative = cost for short gamma

            # Tail events: days where SPY moved > 2 std
            std_spy = float(spy_ret.std())
            tail_days = spy_ret[spy_ret.abs() > 2 * std_spy]
            if len(tail_days) > 0:
                # Gamma P&L in tail: short gamma loses r² on large moves
                gamma_pnl_tail = -float((tail_days ** 2).mean())
                # Vega P&L: VIX spikes on large down-moves → short vega loses
                vega_pnl_tail = float(tail_days[tail_days < 0].mean()) * 0.5  # VIX beta ~50% of SPY move

    # ── Correlation spike analysis ───────────────────────────────────────
    corr_spike_severity = 0.0
    corr_spike_pnl = 0.0

    if len(sectors) >= 3:
        sec_ret = prices[sectors].pct_change().dropna()
        if len(sec_ret) >= 60:
            # Rolling 20-day average pairwise correlation
            rolling_corrs = []
            for i in range(20, len(sec_ret)):
                window = sec_ret.iloc[i - 20: i]
                corr_mat = window.corr().values
                mask = ~np.eye(len(sectors), dtype=bool)
                avg_corr = float(corr_mat[mask].mean())
                rolling_corrs.append(avg_corr)

            if rolling_corrs:
                corr_series = np.array(rolling_corrs)
                # 5-day change in correlation
                corr_changes_5d = corr_series[5:] - corr_series[:-5]
                if len(corr_changes_5d) > 0:
                    corr_spike_severity = float(corr_changes_5d.max())
                    # P&L impact: corr spike of X means dispersion trade loses ~X * gross_notional
                    gross = sum(abs(weights.get(s, 0)) for s in sectors)
                    corr_spike_pnl = -corr_spike_severity * gross * 0.5  # Approximate

                    if corr_spike_severity > 0.15:
                        warnings.append(
                            f"Worst 5-day correlation spike: +{corr_spike_severity:.2f} — "
                            f"dispersion trades face ~{corr_spike_pnl:.2%} P&L impact"
                        )

    return ShortVolTailResult(
        gamma_pnl_tail=round(gamma_pnl_tail, 6),
        vega_pnl_tail=round(vega_pnl_tail, 6),
        convexity_drag=round(convexity_drag, 6),
        overnight_gap_var95=round(overnight_gap_var95, 6),
        max_overnight_gap=round(max_gap, 6),
        gap_frequency=round(gap_freq, 4),
        vvix_proxy=round(vvix_proxy, 3),
        vix_mean=round(vix_mean, 2),
        vix_convexity=round(vix_convexity, 2),
        corr_spike_severity=round(corr_spike_severity, 4),
        corr_spike_pnl_impact=round(corr_spike_pnl, 6),
        warnings=warnings,
    )
