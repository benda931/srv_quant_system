"""
analytics/portfolio_risk.py

PortfolioRiskEngine: parametric and historical risk decomposition for the
SRV Quantamental Decision Support System.

Computes:
- Ledoit-Wolf shrunk covariance matrix (fallback: sample if < 60 obs)
- Parametric (Gaussian) VaR and historical CVaR
- Marginal Contribution to Risk (MCTR) and percentage risk-budget series
- Two-component factor VaR decomposition (systematic vs idiosyncratic)
- Full risk report with breach flags against Settings targets
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.covariance import LedoitWolf

from config.settings import Settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
_COV_WINDOW: int = 252       # rolling window for covariance estimation (trading days)
_MIN_OBS_LW: int = 60        # minimum observations required to apply Ledoit-Wolf
_TRADING_DAYS: int = 252     # annualisation factor
_EPSILON: float = 1e-12      # numerical floor to avoid division by zero


# ---------------------------------------------------------------------------
# RiskReport dataclass
# ---------------------------------------------------------------------------
@dataclass
class RiskReport:
    """Aggregated risk metrics for the current sector portfolio."""

    portfolio_vol_ann: float        # annualised portfolio volatility  σ_p * √252
    var_95_1d: float                # parametric 95 % 1-day VaR  (positive = loss)
    cvar_95_1d: float               # historical 95 % 1-day CVaR (positive = loss)
    mctr_series: pd.Series          # marginal contribution to 1-day σ_p per sector
    risk_budget_series: pd.Series   # % risk contribution per sector (sums to 1.0)
    concentration_hhi: float        # Herfindahl-Hirschman Index of weight vector
    max_sector_weight: float        # largest |w_i| in the portfolio
    vol_target_breach: bool         # True if portfolio_vol_ann > settings.target_portfolio_vol
    max_weight_breach: bool         # True if max_sector_weight > settings.max_single_name_weight


# ---------------------------------------------------------------------------
# Public engine class
# ---------------------------------------------------------------------------
class PortfolioRiskEngine:
    """
    Risk analytics engine for the SRV sector ETF portfolio.

    All public methods accept ``weights`` as a ``dict[str, float]`` mapping
    sector ticker to portfolio weight; values should sum to approximately 1.0.

    Covariance inputs are daily (not pre-annualised); annualisation is
    applied internally where reported figures require it.
    """

    # ------------------------------------------------------------------
    # Covariance estimation
    # ------------------------------------------------------------------

    def compute_cov(self, returns_df: pd.DataFrame) -> np.ndarray:
        """
        Estimate the daily covariance matrix using Ledoit-Wolf shrinkage.

        Uses the most recent ``_COV_WINDOW`` (252) trading-day observations.
        Falls back to the sample covariance when fewer than ``_MIN_OBS_LW``
        (60) clean rows are available after forward-filling.

        Parameters
        ----------
        returns_df : pd.DataFrame
            Daily log-returns (rows = dates, columns = sector tickers).

        Returns
        -------
        np.ndarray
            ``(n, n)`` daily covariance matrix in the same column order as
            ``returns_df``.
        """
        # Drop date rows where every column is NaN, then take the tail window
        clean = returns_df.dropna(how="all").tail(_COV_WINDOW)
        # Forward-fill gaps up to 5 days max (thin markets / holidays only)
        # Then drop remaining NaN rows — never zero-fill, as that creates
        # artificial zero-variance days and suppresses cross-correlations.
        clean = clean.ffill(limit=5).dropna(how="any")
        n_obs, n_assets = clean.shape

        if n_obs < 2:
            raise ValueError(
                f"Insufficient observations for covariance estimation: {n_obs}"
            )

        X = clean.values.astype(float)

        if n_obs >= _MIN_OBS_LW:
            try:
                lw = LedoitWolf(assume_centered=False)
                lw.fit(X)
                logger.debug(
                    "LedoitWolf: n_obs=%d  n_assets=%d  shrinkage=%.4f",
                    n_obs, n_assets, float(lw.shrinkage_),
                )
                return lw.covariance_
            except Exception as exc:
                logger.warning(
                    "LedoitWolf failed (%s); falling back to sample covariance.", exc
                )

        cov = np.cov(X, rowvar=False, ddof=1)
        cov = np.atleast_2d(cov)
        logger.debug(
            "Sample covariance: n_obs=%d  (threshold=%d).", n_obs, _MIN_OBS_LW
        )
        return cov

    # ------------------------------------------------------------------
    # VaR / CVaR
    # ------------------------------------------------------------------

    def compute_var(
        self,
        weights: Dict[str, float],
        cov: np.ndarray,
        confidence: float = 0.95,
        horizon: int = 1,
    ) -> float:
        """
        Parametric (Gaussian) Value-at-Risk.

        ``VaR = z_α × σ_p × √horizon``  where ``σ_p`` is 1-day portfolio
        standard deviation derived from the daily covariance matrix.

        Parameters
        ----------
        weights : dict[str, float]
            Sector ticker → portfolio weight (order must match ``cov`` rows/cols).
        cov : np.ndarray
            Daily covariance matrix.
        confidence : float
            Confidence level (default 0.95).
        horizon : int
            Holding period in days (default 1).

        Returns
        -------
        float
            VaR as a positive number (expected loss at the stated confidence).
        """
        w = _to_vec(weights)
        port_var = float(w @ cov @ w)
        port_std = math.sqrt(max(port_var, 0.0))
        z = float(norm.ppf(confidence))
        return z * port_std * math.sqrt(float(horizon))

    def compute_cvar(
        self,
        weights: Dict[str, float],
        returns_df: pd.DataFrame,
        confidence: float = 0.95,
    ) -> float:
        """
        Historical Conditional Value-at-Risk (Expected Shortfall).

        Reconstructs the portfolio return series from ``returns_df``, then
        returns the mean loss in the left tail beyond the ``1 − confidence``
        quantile.

        Parameters
        ----------
        weights : dict[str, float]
            Sector ticker → portfolio weight.
        returns_df : pd.DataFrame
            Daily log-returns; columns must contain every key in ``weights``.
        confidence : float
            Confidence level (default 0.95).

        Returns
        -------
        float
            CVaR as a positive number (average loss in the tail).
        """
        tickers = [t for t in weights if t in returns_df.columns]
        missing = [t for t in weights if t not in returns_df.columns]
        if missing:
            logger.warning("compute_cvar: dropping missing tickers %s.", missing)
        if not tickers:
            return float("nan")

        w_vec = np.array([weights[t] for t in tickers], dtype=float)
        ret_mat = (
            returns_df[tickers].dropna(how="all").ffill(limit=5).dropna(how="any").values.astype(float)
        )
        port_rets = ret_mat @ w_vec                      # shape (T,)
        cutoff = float(np.quantile(port_rets, 1.0 - confidence))
        tail = port_rets[port_rets <= cutoff]
        if tail.size == 0:
            return float("nan")
        return float(-tail.mean())

    # ------------------------------------------------------------------
    # Risk decomposition
    # ------------------------------------------------------------------

    def compute_mctr(
        self,
        weights: Dict[str, float],
        cov: np.ndarray,
    ) -> pd.Series:
        """
        Marginal Contribution to Risk per sector.

        ``MCTR_i = ∂σ_p / ∂w_i = (Σw)_i / σ_p``

        The result is in the same units as the portfolio standard deviation
        (1-day, unless ``cov`` was annualised before passing in).

        Parameters
        ----------
        weights : dict[str, float]
        cov : np.ndarray

        Returns
        -------
        pd.Series
            Index = sector tickers, values = MCTR_i.
        """
        tickers = list(weights.keys())
        w = _to_vec(weights)
        port_var = float(w @ cov @ w)
        port_std = math.sqrt(max(port_var, _EPSILON))
        mctr = (cov @ w) / port_std
        return pd.Series(mctr, index=tickers, name="mctr")

    def compute_risk_budget(
        self,
        weights: Dict[str, float],
        cov: np.ndarray,
    ) -> pd.Series:
        """
        Percentage risk contribution (risk budget) per sector.

        ``RB_i = w_i × (Σw)_i / (w⊤Σw)``   →  sums to 1.0

        Parameters
        ----------
        weights : dict[str, float]
        cov : np.ndarray

        Returns
        -------
        pd.Series
            Index = sector tickers, values in [0, 1] summing to 1.0.
        """
        tickers = list(weights.keys())
        w = _to_vec(weights)
        port_var = float(w @ cov @ w)
        if port_var < _EPSILON:
            rb = np.full(len(w), 1.0 / max(len(w), 1))
        else:
            cov_w = cov @ w
            rb = w * cov_w / port_var   # element-wise; sums to 1.0 by Euler's theorem
        return pd.Series(rb, index=tickers, name="risk_budget")

    def factor_var_decomp(
        self,
        weights: Dict[str, float],
        factor_returns: pd.DataFrame,
        cov: np.ndarray,
    ) -> dict:
        """
        Decompose portfolio variance into systematic and idiosyncratic components.

        Beta loadings are estimated by OLS regression of each sector return
        series against the pure-factor columns found in ``factor_returns``.
        The sector columns (keys of ``weights``) are used as the dependent
        variables; any remaining columns are treated as factors.

        Parameters
        ----------
        weights : dict[str, float]
            Sector ticker → portfolio weight.
        factor_returns : pd.DataFrame
            Wide returns DataFrame whose columns include the sector tickers
            (needed for beta estimation) **and** one or more factor columns
            (e.g. "SPY", "^TNX", "DX-Y.NYB").  Rows = dates.
        cov : np.ndarray
            Daily sector covariance matrix (Ledoit-Wolf shrunk) aligned to
            the sector order implied by ``weights``.

        Returns
        -------
        dict
            Keys: ``systematic_var_1d``, ``idiosyncratic_var_1d``,
            ``systematic_var_ann``, ``idiosyncratic_var_ann``,
            ``systematic_pct``, ``idiosyncratic_pct``,
            ``factor_betas`` (DataFrame: sectors × factors).
        """
        sector_cols: List[str] = [t for t in weights if t in factor_returns.columns]
        factor_cols: List[str] = [c for c in factor_returns.columns if c not in weights]

        if not sector_cols:
            logger.warning("factor_var_decomp: no sector columns found in factor_returns.")
            return _empty_decomp()

        if not factor_cols:
            logger.warning(
                "factor_var_decomp: no factor columns found; "
                "treating all portfolio variance as systematic."
            )
            w = np.array([weights.get(t, 0.0) for t in sector_cols], dtype=float)
            total_var = float(w @ cov @ w)
            return {
                "systematic_var_1d":    total_var,
                "idiosyncratic_var_1d": 0.0,
                "systematic_var_ann":   total_var * _TRADING_DAYS,
                "idiosyncratic_var_ann": 0.0,
                "systematic_pct":       1.0,
                "idiosyncratic_pct":    0.0,
                "factor_betas":         pd.DataFrame(),
            }

        data = (
            factor_returns[sector_cols + factor_cols]
            .ffill(limit=5)
            .dropna(how="any")
        )
        if len(data) < 30:
            logger.warning(
                "factor_var_decomp: only %d clean rows — decomposition may be unreliable.",
                len(data),
            )

        F = data[factor_cols].values.astype(float)    # (T, n_factors)
        S = data[sector_cols].values.astype(float)    # (T, n_sectors)

        # OLS:  S = intercept + F @ B^T + ε
        F_int = np.hstack([np.ones((len(F), 1)), F])  # (T, n_factors+1)
        try:
            B_full, _, _, _ = np.linalg.lstsq(F_int, S, rcond=None)
        except np.linalg.LinAlgError as exc:
            logger.error("factor_var_decomp OLS failed: %s", exc)
            return _empty_decomp()

        # B shape: (n_sectors, n_factors) — drop intercept row, transpose
        B = B_full[1:, :].T

        # Factor covariance (daily)  — np.cov with 2-D rowvar=True input
        Sigma_F = np.cov(F.T, ddof=1) if F.shape[1] > 1 else np.atleast_2d(float(np.var(F[:, 0], ddof=1)))

        # Systematic covariance: B Σ_F B^T
        Sigma_sys = B @ Sigma_F @ B.T

        w = np.array([weights.get(t, 0.0) for t in sector_cols], dtype=float)
        sys_var = float(w @ Sigma_sys @ w)
        total_var = float(w @ cov @ w)
        idio_var = max(total_var - sys_var, 0.0)

        denom = sys_var + idio_var
        sys_pct = sys_var / denom if denom > _EPSILON else 0.5
        idio_pct = 1.0 - sys_pct

        logger.debug(
            "factor_var_decomp: systematic=%.1f%%  idiosyncratic=%.1f%%",
            sys_pct * 100, idio_pct * 100,
        )

        return {
            "systematic_var_1d":    sys_var,
            "idiosyncratic_var_1d": idio_var,
            "systematic_var_ann":   sys_var * _TRADING_DAYS,
            "idiosyncratic_var_ann": idio_var * _TRADING_DAYS,
            "systematic_pct":       sys_pct,
            "idiosyncratic_pct":    idio_pct,
            "factor_betas":         pd.DataFrame(B, index=sector_cols, columns=factor_cols),
        }

    # ------------------------------------------------------------------
    # Full risk report
    # ------------------------------------------------------------------

    def full_risk_report(
        self,
        weights: Dict[str, float],
        prices_df: pd.DataFrame,
        settings: Settings,
    ) -> RiskReport:
        """
        Compute the full risk report for the current sector portfolio.

        Derives daily log-returns from ``prices_df``, estimates a
        Ledoit-Wolf shrunk covariance over the last 252 trading days, and
        assembles all risk metrics into a :class:`RiskReport`.

        Parameters
        ----------
        weights : dict[str, float]
            Sector ticker → portfolio weight (must sum to approximately 1.0).
        prices_df : pd.DataFrame
            Daily close prices (rows = dates, columns include sector tickers
            and optionally macro / factor tickers such as "SPY", "^TNX").
        settings : Settings
            Validated :class:`~config.settings.Settings` instance; used for
            breach threshold checks.

        Returns
        -------
        RiskReport
        """
        _validate_weights(weights)

        sector_cols = [t for t in weights if t in prices_df.columns]
        if not sector_cols:
            raise ValueError(
                "None of the weight keys appear as columns in prices_df."
            )
        missing_w = [t for t in weights if t not in prices_df.columns]
        if missing_w:
            logger.warning(
                "full_risk_report: tickers absent from prices_df (ignored): %s", missing_w
            )

        # --- log returns ------------------------------------------------------
        log_rets = np.log(prices_df / prices_df.shift(1)).iloc[1:]
        sector_rets = log_rets[sector_cols].copy()

        # Align weights dict to columns present in prices_df
        w_avail: Dict[str, float] = {t: weights[t] for t in sector_cols}

        # --- covariance -------------------------------------------------------
        cov = self.compute_cov(sector_rets)

        # --- core scalar metrics ----------------------------------------------
        w_vec = _to_vec(w_avail)
        port_var_1d = float(w_vec @ cov @ w_vec)
        port_std_1d = math.sqrt(max(port_var_1d, 0.0))
        port_vol_ann = port_std_1d * math.sqrt(_TRADING_DAYS)

        var_95 = self.compute_var(w_avail, cov, confidence=0.95, horizon=1)
        cvar_95 = self.compute_cvar(w_avail, sector_rets, confidence=0.95)

        # --- MCTR & risk budget -----------------------------------------------
        mctr = self.compute_mctr(w_avail, cov)
        risk_budget = self.compute_risk_budget(w_avail, cov)

        # --- concentration & weight stats ------------------------------------
        w_vals = np.array(list(w_avail.values()), dtype=float)
        hhi = float(np.sum(w_vals ** 2))
        max_wt = float(np.max(np.abs(w_vals)))

        # --- breach checks ----------------------------------------------------
        vol_breach = port_vol_ann > settings.target_portfolio_vol
        wt_breach = max_wt > settings.max_single_name_weight

        if vol_breach:
            logger.warning(
                "Vol target breach: portfolio_vol_ann=%.2f%% > target=%.2f%%",
                port_vol_ann * 100,
                settings.target_portfolio_vol * 100,
            )
        if wt_breach:
            logger.warning(
                "Max weight breach: max_sector_weight=%.2f%% > limit=%.2f%%",
                max_wt * 100,
                settings.max_single_name_weight * 100,
            )

        logger.info(
            "Risk report: σ_ann=%.2f%%  VaR_95_1d=%.2f%%  CVaR_95_1d=%.2f%%  "
            "HHI=%.3f  vol_breach=%s  wt_breach=%s",
            port_vol_ann * 100,
            var_95 * 100,
            cvar_95 * 100 if math.isfinite(cvar_95) else float("nan"),
            hhi,
            vol_breach,
            wt_breach,
        )

        return RiskReport(
            portfolio_vol_ann=port_vol_ann,
            var_95_1d=var_95,
            cvar_95_1d=cvar_95,
            mctr_series=mctr,
            risk_budget_series=risk_budget,
            concentration_hhi=hhi,
            max_sector_weight=max_wt,
            vol_target_breach=vol_breach,
            max_weight_breach=wt_breach,
        )


# ---------------------------------------------------------------------------
# Module-level private helpers
# ---------------------------------------------------------------------------

def _to_vec(weights: Dict[str, float]) -> np.ndarray:
    """Convert ordered weight dict to 1-D float array."""
    return np.array(list(weights.values()), dtype=float)


def _validate_weights(weights: Dict[str, float]) -> None:
    """Raise or warn on degenerate weight inputs."""
    if not weights:
        raise ValueError("weights dict is empty.")
    total = sum(weights.values())
    if abs(total - 1.0) > 0.05:
        logger.warning(
            "Portfolio weights sum to %.4f (expected ~1.0).", total
        )


def _empty_decomp() -> dict:
    """Return a NaN-filled decomposition result for error cases."""
    return {
        "systematic_var_1d":    float("nan"),
        "idiosyncratic_var_1d": float("nan"),
        "systematic_var_ann":   float("nan"),
        "idiosyncratic_var_ann": float("nan"),
        "systematic_pct":       float("nan"),
        "idiosyncratic_pct":    float("nan"),
        "factor_betas":         pd.DataFrame(),
    }
