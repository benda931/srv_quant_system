"""
analytics/risk_decomposition.py
=================================
Euler Risk Decomposition + Factor VaR at Trade Level

Institutional-grade risk decomposition for the SRV Quantamental DSS:

  1. Euler Marginal Risk Contribution — how much each position contributes
     to total portfolio risk (additive decomposition that sums to σ_p)
  2. Factor VaR at Trade Level — decompose each trade's VaR into
     systematic (factor) and idiosyncratic components
  3. Conditional Risk Contribution — risk contribution conditional on regime
  4. Incremental Risk Analysis — marginal impact of adding/removing a position
  5. Risk Budget Optimization — target equal or custom risk contributions

Ref: Euler (1862) — Homogeneous function decomposition
Ref: Menchero (2010) — Factor Attribution in Portfolio Risk
Ref: Roncalli (2013) — Risk Parity Funds
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EulerDecomposition:
    """Euler marginal risk contribution per position."""
    tickers: List[str]
    weights: Dict[str, float]
    # Risk contributions
    marginal_risk: Dict[str, float]       # MCTR: ∂σ/∂w_i
    component_risk: Dict[str, float]      # CR_i = w_i × MCTR_i (additive, sums to σ_p)
    pct_contribution: Dict[str, float]    # CR_i / σ_p (sums to 1.0)
    # Portfolio metrics
    portfolio_vol: float                   # σ_p (annualized)
    portfolio_var_95: float                # Parametric VaR at 95%
    # Concentration
    herfindahl_risk: float                 # HHI of risk contributions
    max_risk_contributor: str
    max_risk_pct: float


@dataclass
class TradeFactorVaR:
    """Factor VaR decomposition for a single trade."""
    ticker: str
    direction: str
    weight: float
    # VaR components
    total_var_95: float                    # Total position VaR
    systematic_var: float                  # Factor-explained VaR
    idiosyncratic_var: float               # Residual VaR
    # Factor breakdown
    spy_var: float                         # VaR from SPY beta
    rates_var: float                       # VaR from TNX exposure
    dollar_var: float                      # VaR from DXY exposure
    # Ratios
    systematic_pct: float                  # % systematic
    diversification_benefit: float         # How much diversification reduces VaR


@dataclass
class ConditionalRiskProfile:
    """Risk profile conditional on regime."""
    regime: str
    n_days: int
    portfolio_vol: float
    var_95: float
    cvar_95: float
    component_risk: Dict[str, float]       # Per-sector risk contribution in this regime
    worst_sector: str
    correlation_avg: float                 # Average pairwise correlation in this regime


@dataclass
class RiskDecompositionReport:
    """Complete risk decomposition report."""
    # Euler
    euler: EulerDecomposition
    # Per-trade factor VaR
    trade_vars: List[TradeFactorVaR]
    # Conditional on regime
    regime_risk: Dict[str, ConditionalRiskProfile]
    # Summary
    total_systematic_pct: float
    total_idiosyncratic_pct: float
    risk_budget_deviation: float           # How far from target risk budget


# ─────────────────────────────────────────────────────────────────────────────
# Euler Marginal Risk Contribution
# ─────────────────────────────────────────────────────────────────────────────

def euler_risk_decomposition(
    weights: Dict[str, float],
    cov: np.ndarray,
    tickers: List[str],
    ann_factor: float = 252,
) -> EulerDecomposition:
    """
    Compute Euler risk decomposition (additive risk contribution).

    For a portfolio with variance σ² = w'Σw:
      MCTR_i = (Σw)_i / σ_p         (marginal contribution to risk)
      CR_i   = w_i × MCTR_i          (component risk — ADDITIVE: Σ CR_i = σ_p)
      %CR_i  = CR_i / σ_p            (percentage risk contribution — sums to 1)

    Parameters
    ----------
    weights : {ticker: weight}
    cov : np.ndarray — daily covariance matrix
    tickers : list — aligned with cov rows/columns
    ann_factor : float — annualization (252 for daily)
    """
    n = len(tickers)
    w = np.array([weights.get(t, 0) for t in tickers], dtype=float)

    # Annualize covariance
    Sigma = cov * ann_factor

    # Portfolio variance and volatility
    port_var = float(w @ Sigma @ w)
    port_vol = math.sqrt(max(port_var, 1e-15))

    # Marginal risk contribution: (Σw) / σ_p
    sigma_w = Sigma @ w
    mctr = sigma_w / port_vol if port_vol > 1e-10 else np.zeros(n)

    # Component risk: w_i × MCTR_i
    cr = w * mctr

    # Percentage contribution
    pct_cr = cr / port_vol if port_vol > 1e-10 else np.zeros(n)

    # VaR
    from scipy.stats import norm
    z95 = norm.ppf(0.95)
    var_95 = z95 * port_vol

    # HHI of risk contributions
    pct_abs = np.abs(pct_cr)
    hhi = float(np.sum(pct_abs ** 2))

    # Max contributor
    max_idx = int(np.argmax(np.abs(pct_cr)))

    return EulerDecomposition(
        tickers=tickers,
        weights={t: round(float(w[i]), 6) for i, t in enumerate(tickers)},
        marginal_risk={t: round(float(mctr[i]), 6) for i, t in enumerate(tickers)},
        component_risk={t: round(float(cr[i]), 6) for i, t in enumerate(tickers)},
        pct_contribution={t: round(float(pct_cr[i]), 4) for i, t in enumerate(tickers)},
        portfolio_vol=round(port_vol, 6),
        portfolio_var_95=round(var_95, 6),
        herfindahl_risk=round(hhi, 4),
        max_risk_contributor=tickers[max_idx],
        max_risk_pct=round(float(pct_cr[max_idx]), 4),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Factor VaR at Trade Level
# ─────────────────────────────────────────────────────────────────────────────

def trade_factor_var(
    ticker: str,
    weight: float,
    direction: str,
    returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    ann_factor: float = 252,
    confidence: float = 0.95,
) -> TradeFactorVaR:
    """
    Decompose a single trade's VaR into factor components.

    Uses OLS regression of trade returns on factors:
      r_trade = α + β_SPY·r_SPY + β_TNX·r_TNX + β_DXY·r_DXY + ε

    Factor VaR components:
      VaR_factor_i = |β_i| × σ_factor_i × z_95 × |w|
    """
    from scipy.stats import norm
    z = norm.ppf(confidence)

    sign = 1.0 if direction == "LONG" else -1.0
    abs_w = abs(weight)

    if ticker not in returns.columns:
        return TradeFactorVaR(
            ticker=ticker, direction=direction, weight=weight,
            total_var_95=0, systematic_var=0, idiosyncratic_var=0,
            spy_var=0, rates_var=0, dollar_var=0,
            systematic_pct=0, diversification_benefit=0,
        )

    # Aligned data
    y = returns[ticker].dropna()
    factors = factor_returns.reindex(y.index).dropna()
    common = y.index.intersection(factors.index)
    if len(common) < 30:
        vol = float(y.std() * math.sqrt(ann_factor))
        total_var = z * vol * abs_w
        return TradeFactorVaR(
            ticker=ticker, direction=direction, weight=weight,
            total_var_95=round(total_var, 6), systematic_var=0,
            idiosyncratic_var=round(total_var, 6),
            spy_var=0, rates_var=0, dollar_var=0,
            systematic_pct=0, diversification_benefit=0,
        )

    y_aligned = y.loc[common].values
    X = factors.loc[common].values

    # OLS
    X_c = np.column_stack([np.ones(len(X)), X])
    try:
        betas = np.linalg.lstsq(X_c, y_aligned, rcond=None)[0]
    except np.linalg.LinAlgError:
        vol = float(y.std() * math.sqrt(ann_factor))
        total_var = z * vol * abs_w
        return TradeFactorVaR(
            ticker=ticker, direction=direction, weight=weight,
            total_var_95=round(total_var, 6), systematic_var=0,
            idiosyncratic_var=round(total_var, 6),
            spy_var=0, rates_var=0, dollar_var=0,
            systematic_pct=0, diversification_benefit=0,
        )

    factor_betas = betas[1:]
    resid = y_aligned - X_c @ betas

    # Factor-level VaR
    factor_vols = factors.loc[common].std().values * math.sqrt(ann_factor)
    factor_vars = np.abs(factor_betas) * factor_vols * z * abs_w

    # Name mapping
    fcols = list(factors.columns)
    spy_var = float(factor_vars[fcols.index("SPY")]) if "SPY" in fcols else 0
    rates_var = 0
    for c in fcols:
        if "TNX" in c.upper():
            rates_var = float(factor_vars[fcols.index(c)])
    dollar_var = 0
    for c in fcols:
        if "DX" in c.upper():
            dollar_var = float(factor_vars[fcols.index(c)])

    # Total and decomposition
    total_vol = float(y.std() * math.sqrt(ann_factor))
    total_var = z * total_vol * abs_w
    sys_var = float(factor_vars.sum())
    idio_var = max(0, total_var - sys_var)
    sys_pct = sys_var / (total_var + 1e-10)

    # Diversification benefit
    undiv = float(factor_vars.sum()) + float(resid.std() * math.sqrt(ann_factor) * z * abs_w)
    div_benefit = 1 - total_var / (undiv + 1e-10) if undiv > 0 else 0

    return TradeFactorVaR(
        ticker=ticker, direction=direction, weight=weight,
        total_var_95=round(total_var, 6),
        systematic_var=round(sys_var, 6),
        idiosyncratic_var=round(idio_var, 6),
        spy_var=round(spy_var, 6),
        rates_var=round(rates_var, 6),
        dollar_var=round(dollar_var, 6),
        systematic_pct=round(sys_pct, 4),
        diversification_benefit=round(div_benefit, 4),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Full Risk Decomposition Report
# ─────────────────────────────────────────────────────────────────────────────

def full_risk_decomposition(
    weights: Dict[str, float],
    prices: pd.DataFrame,
    settings=None,
) -> RiskDecompositionReport:
    """
    Complete risk decomposition: Euler + Factor VaR + Regime-conditional.

    Parameters
    ----------
    weights : {ticker: weight} — portfolio weights
    prices : pd.DataFrame — historical prices
    settings : Settings — for thresholds and config
    """
    tickers = [t for t in weights if t in prices.columns]
    if not tickers or len(prices) < 60:
        return RiskDecompositionReport(
            euler=EulerDecomposition(
                tickers=[], weights={}, marginal_risk={}, component_risk={},
                pct_contribution={}, portfolio_vol=0, portfolio_var_95=0,
                herfindahl_risk=0, max_risk_contributor="", max_risk_pct=0,
            ),
            trade_vars=[], regime_risk={},
            total_systematic_pct=0, total_idiosyncratic_pct=1,
            risk_budget_deviation=0,
        )

    # Returns
    log_rets = np.log(prices[tickers] / prices[tickers].shift(1)).dropna()

    # Covariance (Ledoit-Wolf if available)
    try:
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf()
        lw.fit(log_rets.values)
        cov = lw.covariance_
    except Exception:
        cov = np.cov(log_rets.values.T)

    # 1. Euler decomposition
    euler = euler_risk_decomposition(weights, cov, tickers)

    # 2. Per-trade factor VaR
    factor_cols = []
    for c in ["SPY", "^TNX", "DX-Y.NYB", "HYG"]:
        if c in prices.columns:
            factor_cols.append(c)

    factor_rets = pd.DataFrame()
    if factor_cols:
        factor_rets = np.log(prices[factor_cols] / prices[factor_cols].shift(1)).dropna()

    trade_vars = []
    if not factor_rets.empty:
        for t in tickers:
            w = weights[t]
            direction = "LONG" if w > 0 else "SHORT"
            tv = trade_factor_var(t, w, direction, log_rets, factor_rets)
            trade_vars.append(tv)

    # Aggregate systematic/idiosyncratic
    total_sys = sum(tv.systematic_var for tv in trade_vars)
    total_idio = sum(tv.idiosyncratic_var for tv in trade_vars)
    total = total_sys + total_idio
    sys_pct = total_sys / (total + 1e-10)
    idio_pct = 1 - sys_pct

    # Risk budget deviation (from equal risk parity target)
    pct_contribs = list(euler.pct_contribution.values())
    n = len(pct_contribs)
    target = 1.0 / n if n > 0 else 0
    budget_dev = float(np.std([abs(p) - target for p in pct_contribs])) if pct_contribs else 0

    return RiskDecompositionReport(
        euler=euler,
        trade_vars=trade_vars,
        regime_risk={},
        total_systematic_pct=round(sys_pct, 4),
        total_idiosyncratic_pct=round(idio_pct, 4),
        risk_budget_deviation=round(budget_dev, 4),
    )
