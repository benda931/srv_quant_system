"""
analytics/optimizer.py

Portfolio Optimizer — Risk-Parity and Mean-Variance.

Risk-Parity: equal marginal risk contribution per sector.
Mean-Variance: maximize Sharpe ratio subject to constraints.

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
