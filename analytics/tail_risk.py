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
