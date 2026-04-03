"""
analytics/optimizer.py

Portfolio Optimizer — Risk-Parity, Mean-Variance, and Black-Litterman.

Methods:
  1. Risk-Parity: equal marginal risk contribution per sector
  2. Mean-Variance: maximize Sharpe ratio subject to constraints
  3. Black-Litterman: Bayesian combination of market equilibrium + PM views
  4. Regime-Conditional: weight blending adapted to CALM/NORMAL/TENSION/CRISIS

Black-Litterman (Satchell & Scowcroft, 2000):
  Combines market-implied expected returns (equilibrium) with subjective
  views expressed as E[r_view] = Q ± Ω (uncertainty). The posterior
  mean is a precision-weighted blend of equilibrium and views.

Inputs from master_df: direction, mc_score, w_final (signal-based weights)
Cov matrix: from prices last 252 days (Ledoit-Wolf shrinkage)

Output: opt_weight column added to master_df for UI display.
The optimizer suggestions are advisory — final sizing is PM decision.
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

_EPS = 1e-8


# ──────────────────────────────────────────────────────────────────────────────
# Covariance helpers
# ──────────────────────────────────────────────────────────────────────────────

def _ledoit_wolf_cov(returns: np.ndarray) -> np.ndarray:
    """
    Ledoit-Wolf shrinkage covariance estimator (analytical).

    Shrinks sample covariance toward the scaled identity matrix.
    Falls back to sample covariance if LW is unavailable.
    """
    try:
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf()
        lw.fit(returns)
        return lw.covariance_
    except Exception:
        return np.cov(returns.T)


def _compute_cov_from_prices(
    prices_df: pd.DataFrame,
    tickers: List[str],
    lookback: int = 252,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute annualised covariance matrix from daily price returns.

    Parameters
    ----------
    prices_df : pd.DataFrame
        Wide DataFrame with DatetimeIndex and ticker columns.
    tickers : list[str]
        Sector tickers to include.
    lookback : int
        Number of trading days to use (default 252 = 1 year).

    Returns
    -------
    (cov_matrix, valid_tickers)
        cov_matrix : np.ndarray  (n x n, annualised)
        valid_tickers : list[str]  (subset of tickers present in prices_df)
    """
    # Filter to requested tickers that exist in prices_df
    valid = [t for t in tickers if t in prices_df.columns]
    if not valid:
        return np.eye(len(tickers)), tickers

    px = prices_df[valid].dropna(how="all").tail(lookback + 1)
    rets = px.pct_change().dropna()

    if rets.empty or len(rets) < 20:
        logger.warning("_compute_cov: insufficient return history (%d rows)", len(rets))
        return np.eye(len(valid)) * (0.15 ** 2 / 252), valid

    raw_cov = _ledoit_wolf_cov(rets.values)
    # Annualise
    cov = raw_cov * 252

    # Enforce symmetry and positive semi-definiteness
    cov = (cov + cov.T) / 2.0
    min_eig = np.linalg.eigvalsh(cov).min()
    if min_eig < _EPS:
        cov += (_EPS - min_eig) * np.eye(len(valid))

    return cov, valid


# ──────────────────────────────────────────────────────────────────────────────
# Risk Parity
# ──────────────────────────────────────────────────────────────────────────────

def _risk_contribution(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Marginal risk contribution of each asset (not normalised)."""
    port_var = float(weights @ cov @ weights)
    if port_var < _EPS:
        return np.ones(len(weights)) / len(weights)
    marginal = cov @ weights
    rc = weights * marginal / math.sqrt(port_var)
    return rc


def _risk_parity_objective(weights: np.ndarray, cov: np.ndarray) -> float:
    """
    Sum of squared deviations from equal risk contribution.
    Minimise this → equal marginal risk contributions.
    """
    rc = _risk_contribution(weights, cov)
    target = rc.sum() / len(rc)
    return float(((rc - target) ** 2).sum())


def risk_parity(
    cov_matrix: np.ndarray,
    target_vol: float = 0.12,
    tickers: Optional[List[str]] = None,
    max_single_weight: float = 0.20,
) -> Dict[str, float]:
    """
    Equal marginal risk contribution (risk-parity) optimisation.

    Parameters
    ----------
    cov_matrix : np.ndarray
        Annualised covariance matrix (n x n).
    target_vol : float
        Target annualised portfolio volatility (default 12%).
    tickers : list[str]
        Asset names matching rows/cols of cov_matrix.
    max_single_weight : float
        Upper bound on any single weight (default 0.20 = 20%).

    Returns
    -------
    dict[str, float]
        {ticker: weight} where all weights are positive (long-only risk parity).
        Weights are scaled to hit target_vol.
    """
    n = cov_matrix.shape[0]
    if tickers is None:
        tickers = [f"asset_{i}" for i in range(n)]

    if n == 0:
        return {}

    if n == 1:
        return {tickers[0]: 1.0}

    # Initial guess: equal weight
    w0 = np.ones(n) / n

    constraints = [
        {"type": "eq", "fun": lambda w: w.sum() - 1.0},          # sum = 1
    ]
    bounds = [(0.0, max_single_weight)] * n

    try:
        result = minimize(
            _risk_parity_objective,
            w0,
            args=(cov_matrix,),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-9},
        )
        if result.success or result.fun < 1e-6:
            w = np.clip(result.x, 0.0, max_single_weight)
            w = w / w.sum() if w.sum() > _EPS else w0
        else:
            logger.warning(
                "risk_parity: solver did not converge (%s) — using equal weights",
                result.message,
            )
            w = w0
    except Exception as exc:
        logger.warning("risk_parity: optimisation failed (%s) — equal weights", exc)
        w = w0

    # Scale to target_vol
    port_vol = math.sqrt(float(w @ cov_matrix @ w))
    if port_vol > _EPS and target_vol > 0:
        scale = target_vol / port_vol
        # Cap scaling to avoid extreme leverage (max 3x)
        scale = min(scale, 3.0)
        w = w * scale

    return {tickers[i]: float(w[i]) for i in range(n)}


# ──────────────────────────────────────────────────────────────────────────────
# Mean-Variance (max Sharpe)
# ──────────────────────────────────────────────────────────────────────────────

def mean_variance(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    tickers: Optional[List[str]] = None,
    max_single_weight: float = 0.20,
    min_weight: float = 0.0,
    risk_free_rate: float = 0.05,
) -> Dict[str, float]:
    """
    Maximum Sharpe ratio portfolio (long-only).

    Parameters
    ----------
    expected_returns : np.ndarray
        Expected annualised return per asset.
    cov_matrix : np.ndarray
        Annualised covariance matrix.
    tickers : list[str]
        Asset names matching rows/cols.
    max_single_weight : float
        Upper bound per asset (default 0.20).
    min_weight : float
        Lower bound per asset (default 0 — long-only).
    risk_free_rate : float
        Annual risk-free rate for Sharpe computation (default 5%).

    Returns
    -------
    dict[str, float]
        {ticker: weight} — raw long-only weights (sum may exceed 1 if scaled).
    """
    n = cov_matrix.shape[0]
    if tickers is None:
        tickers = [f"asset_{i}" for i in range(n)]

    if n == 0:
        return {}

    mu = np.asarray(expected_returns, dtype=float)
    rf = float(risk_free_rate)

    def neg_sharpe(w: np.ndarray) -> float:
        port_ret = float(w @ mu)
        port_vol = math.sqrt(max(float(w @ cov_matrix @ w), _EPS))
        return -(port_ret - rf) / port_vol

    w0 = np.ones(n) / n
    bounds = [(min_weight, max_single_weight)] * n
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]

    try:
        result = minimize(
            neg_sharpe,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-9},
        )
        if result.success:
            w = np.clip(result.x, min_weight, max_single_weight)
            w = w / w.sum() if w.sum() > _EPS else w0
        else:
            logger.warning(
                "mean_variance: solver failed (%s) — equal weights", result.message
            )
            w = w0
    except Exception as exc:
        logger.warning("mean_variance: failed (%s) — equal weights", exc)
        w = w0

    return {tickers[i]: float(w[i]) for i in range(n)}


# ──────────────────────────────────────────────────────────────────────────────
# Portfolio Optimizer facade
# ──────────────────────────────────────────────────────────────────────────────

class PortfolioOptimizer:
    """
    High-level optimizer that integrates with master_df from QuantEngine.

    Main entry point: apply_to_master_df(master_df, prices_df, settings).
    """

    def __init__(self) -> None:
        pass

    # ── Public API ────────────────────────────────────────────────────────────

    def apply_to_master_df(
        self,
        master_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        settings,
    ) -> pd.DataFrame:
        """
        Run risk-parity optimisation and add opt_weight + risk_parity_w columns.

        Parameters
        ----------
        master_df : pd.DataFrame
            QuantEngine master DataFrame. Must have 'direction' column and either
            'sector_ticker' column or sector ticker as index.
        prices_df : pd.DataFrame
            Wide price DataFrame (DatetimeIndex x ticker).  Used to estimate cov.
        settings : Settings
            For target_vol and max_single_name_weight.

        Returns
        -------
        pd.DataFrame
            Copy of master_df with 'opt_weight' and 'risk_parity_w' columns added.
        """
        if master_df is None or master_df.empty:
            return master_df

        df = master_df.copy()

        try:
            tickers, directions = _extract_tickers_and_directions(df)
            if not tickers:
                logger.warning("apply_to_master_df: no tickers found — skipping optimisation")
                df["opt_weight"] = 0.0
                df["risk_parity_w"] = 0.0
                return df

            target_vol = float(getattr(settings, "target_portfolio_vol", 0.12))
            max_w = float(getattr(settings, "max_single_name_weight", 0.20))

            # ── Compute covariance from last 252 days ─────────────────────────
            cov, valid_tickers = _compute_cov_from_prices(
                prices_df, tickers, lookback=252
            )

            # ── Risk-parity weights (unsigned — direction applied below) ──────
            rp_weights = risk_parity(
                cov_matrix=cov,
                target_vol=target_vol,
                tickers=valid_tickers,
                max_single_weight=max_w,
            )

            # ── Apply direction sign (long = +, short = −) ────────────────────
            signed_weights = {}
            for t in valid_tickers:
                rp_w = rp_weights.get(t, 0.0)
                direction = directions.get(t, "LONG")
                sign = -1.0 if str(direction).upper() in ("SHORT", "SELL", "-1") else 1.0
                signed_weights[t] = rp_w * sign

            # ── Market-neutral adjustment (sum long ≈ sum short ± 5%) ─────────
            signed_weights = _enforce_market_neutral(signed_weights, tolerance=0.05)

            # ── Write back to DataFrame ────────────────────────────────────────
            ticker_col = "sector_ticker" if "sector_ticker" in df.columns else None
            for idx, row in df.iterrows():
                t = row["sector_ticker"] if ticker_col else str(idx)
                df.at[idx, "risk_parity_w"] = float(rp_weights.get(t, 0.0))
                df.at[idx, "opt_weight"] = float(signed_weights.get(t, 0.0))

            # Fill any tickers missing from prices_df
            if "opt_weight" not in df.columns:
                df["opt_weight"] = 0.0
            if "risk_parity_w" not in df.columns:
                df["risk_parity_w"] = 0.0
            df["opt_weight"] = df["opt_weight"].fillna(0.0)
            df["risk_parity_w"] = df["risk_parity_w"].fillna(0.0)

            logger.info(
                "apply_to_master_df: %d tickers optimised — gross notional %.2f",
                len(valid_tickers),
                sum(abs(v) for v in signed_weights.values()),
            )

        except Exception as exc:
            logger.warning("apply_to_master_df: optimisation failed (%s) — zeroing weights", exc)
            df["opt_weight"] = 0.0
            df["risk_parity_w"] = 0.0

        return df


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _extract_tickers_and_directions(df: pd.DataFrame) -> Tuple[List[str], Dict[str, str]]:
    """Return (tickers, {ticker: direction}) from master_df."""
    if "sector_ticker" in df.columns:
        tickers = list(df["sector_ticker"].dropna().unique())
        if "direction" in df.columns:
            directions = dict(zip(df["sector_ticker"], df["direction"]))
        else:
            directions = {t: "LONG" for t in tickers}
    else:
        tickers = [str(x) for x in df.index.dropna().unique()]
        if "direction" in df.columns:
            directions = dict(zip([str(x) for x in df.index], df["direction"]))
        else:
            directions = {t: "LONG" for t in tickers}
    return tickers, directions


def _enforce_market_neutral(
    weights: Dict[str, float],
    tolerance: float = 0.05,
) -> Dict[str, float]:
    """
    Scale short weights so |sum(longs) - |sum(shorts)|| ≤ tolerance * gross.

    Only adjusts if both long and short positions exist.
    """
    longs = {t: w for t, w in weights.items() if w > 0}
    shorts = {t: w for t, w in weights.items() if w < 0}

    if not longs or not shorts:
        return weights  # one-sided book — no adjustment needed

    sum_long = sum(longs.values())
    sum_short = abs(sum(shorts.values()))

    if sum_long < _EPS or sum_short < _EPS:
        return weights

    gross = sum_long + sum_short
    imbalance = abs(sum_long - sum_short) / (gross + _EPS)

    if imbalance <= tolerance:
        return weights  # already balanced

    # Scale the larger leg down to match the smaller leg
    adjusted = dict(weights)
    if sum_long > sum_short:
        scale = sum_short / sum_long
        for t in longs:
            adjusted[t] = longs[t] * scale
    else:
        scale = sum_long / sum_short
        for t in shorts:
            adjusted[t] = shorts[t] * scale   # shorts[t] is negative

    return adjusted


# ═════════════════════════════════════════════════════════════════════════════
# Black-Litterman Model
# ═════════════════════════════════════════════════════════════════════════════

from dataclasses import dataclass, field


@dataclass
class BlackLittermanResult:
    """Output of Black-Litterman optimization."""
    posterior_returns: Dict[str, float]      # E[r] per sector
    posterior_weights: Dict[str, float]      # Optimal weights
    equilibrium_returns: Dict[str, float]    # Market-implied returns (π)
    view_returns: Dict[str, float]           # PM views (Q)
    view_confidence: Dict[str, float]        # View uncertainty (diagonal of Ω)
    blend_ratio: float                       # How much views influence result (0=all equilibrium, 1=all views)
    sharpe_posterior: float                   # Expected Sharpe of posterior portfolio


def black_litterman(
    cov: np.ndarray,
    tickers: List[str],
    market_weights: Optional[Dict[str, float]] = None,
    views: Optional[Dict[str, float]] = None,
    view_confidence: Optional[Dict[str, float]] = None,
    risk_aversion: float = 2.5,
    tau: float = 0.05,
) -> BlackLittermanResult:
    """
    Black-Litterman model for combining equilibrium returns with PM views.

    Parameters
    ----------
    cov : np.ndarray — N×N covariance matrix (daily, annualized internally)
    tickers : list — sector tickers corresponding to cov rows/columns
    market_weights : dict — equilibrium weights (default: equal weight)
    views : dict — {ticker: expected_annual_return} — PM's views on sectors
    view_confidence : dict — {ticker: confidence_level} — 0 to 1 (1 = fully confident)
    risk_aversion : float — market risk aversion parameter δ (default 2.5)
    tau : float — scalar indicating uncertainty of equilibrium (default 0.05)

    Returns
    -------
    BlackLittermanResult
    """
    n = len(tickers)
    Sigma = cov * 252  # Annualize daily covariance

    # Market weights (default: equal weight)
    if market_weights is None:
        w_eq = np.ones(n) / n
    else:
        w_eq = np.array([market_weights.get(t, 1.0 / n) for t in tickers])
        w_eq = w_eq / (np.sum(np.abs(w_eq)) + _EPS)

    # Equilibrium returns: π = δΣw_eq
    pi = risk_aversion * Sigma @ w_eq

    # If no views, return equilibrium
    eq_returns = {tickers[i]: round(float(pi[i]), 6) for i in range(n)}

    if not views:
        # Mean-variance optimization on equilibrium returns
        opt_w = _mv_optimize(pi, Sigma, tickers)
        return BlackLittermanResult(
            posterior_returns=eq_returns,
            posterior_weights=opt_w,
            equilibrium_returns=eq_returns,
            view_returns={},
            view_confidence={},
            blend_ratio=0.0,
            sharpe_posterior=_portfolio_sharpe(opt_w, pi, Sigma, tickers),
        )

    # Build view matrices
    # P: K×N pick matrix (each row picks one sector)
    # Q: K×1 view returns
    # Ω: K×K view uncertainty (diagonal)
    view_tickers = [t for t in views if t in tickers]
    K = len(view_tickers)
    if K == 0:
        opt_w = _mv_optimize(pi, Sigma, tickers)
        return BlackLittermanResult(
            posterior_returns=eq_returns,
            posterior_weights=opt_w,
            equilibrium_returns=eq_returns,
            view_returns={},
            view_confidence={},
            blend_ratio=0.0,
            sharpe_posterior=_portfolio_sharpe(opt_w, pi, Sigma, tickers),
        )

    P = np.zeros((K, n))
    Q = np.zeros(K)
    omega_diag = np.zeros(K)

    for k, t in enumerate(view_tickers):
        idx = tickers.index(t)
        P[k, idx] = 1.0
        Q[k] = views[t]
        # View uncertainty: higher confidence → lower Ω
        conf = (view_confidence or {}).get(t, 0.5)
        conf = max(0.01, min(0.99, conf))
        omega_diag[k] = (1 - conf) / conf * (tau * Sigma[idx, idx])

    Omega = np.diag(omega_diag)

    # Posterior: E[r] = [(τΣ)^-1 + P'Ω^-1P]^-1 [(τΣ)^-1π + P'Ω^-1Q]
    tau_sigma_inv = np.linalg.inv(tau * Sigma + _EPS * np.eye(n))
    omega_inv = np.linalg.inv(Omega + _EPS * np.eye(K))

    precision_prior = tau_sigma_inv
    precision_views = P.T @ omega_inv @ P
    precision_posterior = precision_prior + precision_views

    try:
        cov_posterior = np.linalg.inv(precision_posterior)
    except np.linalg.LinAlgError:
        cov_posterior = np.linalg.pinv(precision_posterior)

    mean_posterior = cov_posterior @ (precision_prior @ pi + P.T @ omega_inv @ Q)

    posterior_returns = {tickers[i]: round(float(mean_posterior[i]), 6) for i in range(n)}
    view_rets = {view_tickers[k]: round(float(Q[k]), 6) for k in range(K)}
    view_confs = {view_tickers[k]: round(float((view_confidence or {}).get(view_tickers[k], 0.5)), 3) for k in range(K)}

    # Blend ratio: how much views moved the posterior from equilibrium
    diff = np.linalg.norm(mean_posterior - pi)
    total = np.linalg.norm(pi) + _EPS
    blend = min(1.0, diff / total)

    # Optimize on posterior returns
    opt_w = _mv_optimize(mean_posterior, Sigma, tickers)

    return BlackLittermanResult(
        posterior_returns=posterior_returns,
        posterior_weights=opt_w,
        equilibrium_returns=eq_returns,
        view_returns=view_rets,
        view_confidence=view_confs,
        blend_ratio=round(blend, 4),
        sharpe_posterior=_portfolio_sharpe(opt_w, mean_posterior, Sigma, tickers),
    )


def _mv_optimize(
    expected_returns: np.ndarray,
    cov: np.ndarray,
    tickers: List[str],
    max_weight: float = 0.20,
) -> Dict[str, float]:
    """Mean-variance optimize given expected returns + covariance."""
    n = len(tickers)

    def neg_sharpe(w):
        ret = w @ expected_returns
        vol = np.sqrt(w @ cov @ w + _EPS)
        return -ret / vol

    w0 = np.ones(n) / n
    bounds = [(-max_weight, max_weight) for _ in range(n)]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w)}]  # Market-neutral

    try:
        result = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds,
                          constraints=constraints, options={"maxiter": 200})
        w_opt = result.x
    except Exception:
        w_opt = w0

    return {tickers[i]: round(float(w_opt[i]), 6) for i in range(n)}


def _portfolio_sharpe(
    weights: Dict[str, float],
    returns: np.ndarray,
    cov: np.ndarray,
    tickers: List[str],
) -> float:
    """Compute portfolio Sharpe from weights, returns, covariance."""
    w = np.array([weights.get(t, 0) for t in tickers])
    ret = w @ returns
    vol = np.sqrt(w @ cov @ w + _EPS)
    return round(float(ret / vol), 4) if vol > _EPS else 0.0


# ═════════════════════════════════════════════════════════════════════════════
# Regime-Conditional Views for Black-Litterman
# ═════════════════════════════════════════════════════════════════════════════

def regime_views(
    regime: str,
    sectors: List[str],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Generate PM views based on current market regime.

    Returns (views, confidence) dicts for Black-Litterman.

    Regime logic:
      CALM    → favor cyclicals (XLY, XLI, XLF), avoid defensives
      NORMAL  → neutral (no strong views)
      TENSION → favor defensives (XLP, XLV, XLU), reduce tech
      CRISIS  → strong defensive tilt, reduce all cyclicals
    """
    views: Dict[str, float] = {}
    conf: Dict[str, float] = {}

    _CYCLICAL = {"XLY", "XLI", "XLF", "XLB", "XLK", "XLC"}
    _DEFENSIVE = {"XLP", "XLV", "XLU", "XLRE"}

    if regime == "CALM":
        for s in sectors:
            if s in _CYCLICAL:
                views[s] = 0.08   # 8% expected annual return
                conf[s] = 0.4
            elif s in _DEFENSIVE:
                views[s] = 0.03   # 3% expected
                conf[s] = 0.3
    elif regime == "TENSION":
        for s in sectors:
            if s in _DEFENSIVE:
                views[s] = 0.06
                conf[s] = 0.5
            elif s in _CYCLICAL:
                views[s] = -0.02
                conf[s] = 0.4
    elif regime == "CRISIS":
        for s in sectors:
            if s in _DEFENSIVE:
                views[s] = 0.04
                conf[s] = 0.6
            elif s in _CYCLICAL:
                views[s] = -0.10
                conf[s] = 0.7
    # NORMAL → no views (use equilibrium)

    return views, conf
