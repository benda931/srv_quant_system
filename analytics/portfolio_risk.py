"""
analytics/portfolio_risk.py

PortfolioRiskEngine: comprehensive risk decomposition for the
SRV Quantamental Decision Support System (Short Vol / Dispersion + Sector RV).

Computes:
- Ledoit-Wolf shrunk covariance matrix (fallback: sample if < 60 obs)
- Parametric (Gaussian) VaR and historical CVaR
- Marginal Contribution to Risk (MCTR) and percentage risk-budget series
- Two-component factor VaR decomposition (systematic vs idiosyncratic)
- Full risk report with breach flags against Settings targets
- Incremental VaR (iVaR) — marginal impact of adding each position
- Component VaR (cVaR) — risk contribution that sums exactly to portfolio VaR
- Conditional VaR/CVaR per regime (CALM/NORMAL/TENSION/CRISIS)
- Drawdown-at-Risk (DaR) — probabilistic drawdown analysis
- Max leverage analysis — how much leverage can the portfolio sustain
- Stress-adjusted risk budgeting — risk budget under stressed covariance
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

def enforce_weight_limits(
    weights: Dict[str, float],
    max_single: float = 0.20,
    max_gross: float = 1.0,
) -> Dict[str, float]:
    """
    Enforce portfolio weight constraints by clipping and re-normalizing.

    1. Clip each |w_i| to max_single
    2. If gross exposure > max_gross, scale down proportionally
    3. Preserve direction (long/short signs)

    Parameters
    ----------
    weights     : {ticker: weight} — can be negative for shorts
    max_single  : Max absolute weight per position (default 20%)
    max_gross   : Max sum of |weights| (default 100%)

    Returns
    -------
    dict : adjusted weights
    """
    if not weights:
        return weights

    # Step 1: clip each position
    clipped = {}
    for k, w in weights.items():
        sign = 1 if w >= 0 else -1
        clipped[k] = sign * min(abs(w), max_single)

    # Step 2: scale if gross exposure exceeds limit
    gross = sum(abs(w) for w in clipped.values())
    if gross > max_gross and gross > 0:
        scale = max_gross / gross
        clipped = {k: w * scale for k, w in clipped.items()}

    return clipped


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


# ═════════════════════════════════════════════════════════════════════════════
# Incremental VaR + Component VaR
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class IncrementalVaRReport:
    """Incremental and Component VaR for each position."""
    # Incremental VaR: change in portfolio VaR from removing each position
    incremental_var: Dict[str, float]   # {ticker: iVaR}
    # Component VaR: risk contribution that sums to portfolio VaR
    component_var: Dict[str, float]     # {ticker: cVaR}
    # Diversification benefit
    undiversified_var: float            # Sum of individual VaRs (no correlation benefit)
    diversified_var: float              # Portfolio VaR (with correlation)
    diversification_ratio: float        # undiversified / diversified (> 1 = benefit)
    # Best trade to reduce risk
    best_reduction_ticker: str
    best_reduction_var: float           # How much VaR drops if removed


def compute_incremental_var(
    weights: Dict[str, float],
    cov: np.ndarray,
    confidence: float = 0.95,
) -> IncrementalVaRReport:
    """
    Compute Incremental VaR (iVaR) and Component VaR (cVaR).

    iVaR_i = VaR(w) - VaR(w without position i)
    cVaR_i = w_i * (Σw)_i / σ_p * VaR_p  (Euler decomposition, sums to VaR_p)
    """
    tickers = list(weights.keys())
    n = len(tickers)
    w = np.array([weights[t] for t in tickers], dtype=float)

    z = norm.ppf(confidence)
    port_var_1d = float(w @ cov @ w)
    port_std = math.sqrt(max(port_var_1d, _EPSILON))
    portfolio_var = z * port_std

    # Component VaR (Euler allocation — additive decomposition)
    mcr = cov @ w  # marginal contribution to variance
    component_var = {}
    for i, t in enumerate(tickers):
        # cVaR_i = w_i * ∂σ_p/∂w_i * z = w_i * (Σw)_i / σ_p * z
        cvar_i = w[i] * mcr[i] / port_std * z if port_std > _EPSILON else 0.0
        component_var[t] = round(float(cvar_i), 8)

    # Incremental VaR (leave-one-out)
    incremental_var = {}
    for i, t in enumerate(tickers):
        w_ex = np.delete(w, i)
        cov_ex = np.delete(np.delete(cov, i, axis=0), i, axis=1)
        var_ex = float(w_ex @ cov_ex @ w_ex)
        std_ex = math.sqrt(max(var_ex, 0.0))
        var_without = z * std_ex
        incremental_var[t] = round(portfolio_var - var_without, 8)

    # Undiversified VaR: sum of individual position VaRs
    undiv_var = sum(abs(w[i]) * math.sqrt(max(float(cov[i, i]), 0.0)) * z for i in range(n))
    div_ratio = undiv_var / portfolio_var if portfolio_var > _EPSILON else 1.0

    # Best trade to remove for risk reduction
    best_t = max(incremental_var, key=incremental_var.get) if incremental_var else "N/A"
    best_v = incremental_var.get(best_t, 0.0)

    return IncrementalVaRReport(
        incremental_var=incremental_var,
        component_var=component_var,
        undiversified_var=round(undiv_var, 8),
        diversified_var=round(portfolio_var, 8),
        diversification_ratio=round(div_ratio, 4),
        best_reduction_ticker=best_t,
        best_reduction_var=round(best_v, 8),
    )


# ═════════════════════════════════════════════════════════════════════════════
# Conditional VaR/CVaR per Regime
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class ConditionalRiskResult:
    """Risk metrics conditional on regime."""
    regime: str
    n_days: int
    vol_ann: float
    var_95: float
    cvar_95: float
    max_daily_loss: float
    avg_daily_return: float
    sharpe_conditional: float        # Annualized Sharpe in this regime


@dataclass
class ConditionalRiskReport:
    """Complete regime-conditional risk breakdown."""
    regime_risk: Dict[str, ConditionalRiskResult]
    worst_regime: str                # Regime with highest CVaR
    risk_regime_spread: float        # Worst/best vol ratio


def compute_conditional_risk(
    weights: Dict[str, float],
    returns_df: pd.DataFrame,
    vix: Optional[pd.Series] = None,
    settings=None,
) -> ConditionalRiskReport:
    """
    Compute VaR/CVaR conditional on each regime (CALM/NORMAL/TENSION/CRISIS).
    """
    tickers = [t for t in weights if t in returns_df.columns]
    if not tickers or len(returns_df) < 100:
        return ConditionalRiskReport(regime_risk={}, worst_regime="N/A", risk_regime_spread=1.0)

    w = np.array([weights[t] for t in tickers], dtype=float)
    rets = returns_df[tickers].dropna()
    port_ret = rets.values @ w

    # Classify regimes
    vix_soft = getattr(settings, "vix_level_soft", 21.0) if settings else 21.0
    vix_hard = getattr(settings, "vix_level_hard", 32.0) if settings else 32.0

    if vix is not None and len(vix) >= len(port_ret):
        vix_vals = vix.iloc[-len(port_ret):].values
    else:
        vix_vals = np.full(len(port_ret), 18.0)

    regimes = np.where(
        vix_vals > vix_hard, "CRISIS",
        np.where(vix_vals > vix_soft, "TENSION",
                 np.where(vix_vals > 16, "NORMAL", "CALM"))
    )

    results: Dict[str, ConditionalRiskResult] = {}
    z95 = norm.ppf(0.95)

    for regime in ["CALM", "NORMAL", "TENSION", "CRISIS"]:
        mask = regimes == regime
        r = port_ret[mask]
        n = int(mask.sum())
        if n < 20:
            continue

        std_d = float(r.std(ddof=1))
        vol_ann = std_d * math.sqrt(_TRADING_DAYS)
        var_95 = z95 * std_d
        losses = -r
        tail = losses[losses >= np.percentile(losses, 95)]
        cvar_95 = float(tail.mean()) if len(tail) > 0 else var_95
        max_loss = float(r.min())
        avg_ret = float(r.mean())
        sharpe = avg_ret / std_d * math.sqrt(_TRADING_DAYS) if std_d > _EPSILON else 0.0

        results[regime] = ConditionalRiskResult(
            regime=regime, n_days=n,
            vol_ann=round(vol_ann, 6),
            var_95=round(-var_95, 6),
            cvar_95=round(-cvar_95, 6),
            max_daily_loss=round(max_loss, 6),
            avg_daily_return=round(avg_ret, 8),
            sharpe_conditional=round(sharpe, 4),
        )

    # Worst regime
    cvar_map = {r: abs(v.cvar_95) for r, v in results.items()}
    worst = max(cvar_map, key=cvar_map.get) if cvar_map else "N/A"
    vol_map = {r: v.vol_ann for r, v in results.items()}
    spread = max(vol_map.values()) / min(vol_map.values()) if len(vol_map) >= 2 and min(vol_map.values()) > _EPSILON else 1.0

    return ConditionalRiskReport(
        regime_risk=results,
        worst_regime=worst,
        risk_regime_spread=round(spread, 2),
    )


# ═════════════════════════════════════════════════════════════════════════════
# Drawdown-at-Risk (DaR)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class DrawdownRiskResult:
    """Probabilistic drawdown analysis."""
    max_historical_dd: float         # Maximum observed drawdown
    dar_95: float                    # 95th percentile of drawdown distribution
    dar_99: float                    # 99th percentile
    avg_dd: float                    # Average drawdown
    avg_dd_duration_days: int        # Average drawdown length
    max_dd_duration_days: int        # Longest drawdown period
    current_dd: float                # Current drawdown from peak
    time_underwater_pct: float       # % of time spent in drawdown
    calmar: float                    # Return / MaxDD


def compute_drawdown_risk(
    weights: Dict[str, float],
    returns_df: pd.DataFrame,
) -> DrawdownRiskResult:
    """Compute drawdown-at-risk and related metrics."""
    tickers = [t for t in weights if t in returns_df.columns]
    if not tickers or len(returns_df) < 60:
        return DrawdownRiskResult(
            max_historical_dd=0, dar_95=0, dar_99=0, avg_dd=0,
            avg_dd_duration_days=0, max_dd_duration_days=0,
            current_dd=0, time_underwater_pct=0, calmar=0,
        )

    w = np.array([weights[t] for t in tickers], dtype=float)
    port_ret = returns_df[tickers].dropna().values @ w

    # Equity curve
    equity = np.cumprod(1 + port_ret)
    running_max = np.maximum.accumulate(equity)
    drawdowns = equity / running_max - 1  # Negative values = in drawdown

    # Drawdown statistics
    max_dd = float(drawdowns.min())
    current_dd = float(drawdowns[-1])
    avg_dd = float(drawdowns[drawdowns < 0].mean()) if (drawdowns < 0).any() else 0.0
    time_underwater = float((drawdowns < -0.001).mean())

    # Drawdown-at-Risk (percentiles of drawdown distribution)
    dar_95 = float(np.percentile(drawdowns, 5))   # 5th pct = 95% DaR
    dar_99 = float(np.percentile(drawdowns, 1))

    # Drawdown durations
    in_dd = drawdowns < -0.001
    dd_lengths = []
    current_length = 0
    for d in in_dd:
        if d:
            current_length += 1
        else:
            if current_length > 0:
                dd_lengths.append(current_length)
            current_length = 0
    if current_length > 0:
        dd_lengths.append(current_length)

    avg_dur = int(np.mean(dd_lengths)) if dd_lengths else 0
    max_dur = int(np.max(dd_lengths)) if dd_lengths else 0

    # Calmar ratio
    ann_ret = float(port_ret.mean()) * _TRADING_DAYS
    calmar = ann_ret / abs(max_dd) if abs(max_dd) > _EPSILON else 0.0

    return DrawdownRiskResult(
        max_historical_dd=round(max_dd, 6),
        dar_95=round(dar_95, 6),
        dar_99=round(dar_99, 6),
        avg_dd=round(avg_dd, 6),
        avg_dd_duration_days=avg_dur,
        max_dd_duration_days=max_dur,
        current_dd=round(current_dd, 6),
        time_underwater_pct=round(time_underwater, 4),
        calmar=round(calmar, 4),
    )


# ═════════════════════════════════════════════════════════════════════════════
# Max Leverage Analysis
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class LeverageAnalysisResult:
    """Maximum sustainable leverage under risk constraints."""
    current_leverage: float
    max_leverage_vol_target: float    # Max leverage s.t. vol < target
    max_leverage_var_limit: float     # Max leverage s.t. VaR95 < limit
    max_leverage_dd_limit: float      # Max leverage s.t. expected DD < limit
    recommended_leverage: float       # min(all constraints)
    headroom_pct: float               # (recommended - current) / current


def analyse_max_leverage(
    weights: Dict[str, float],
    cov: np.ndarray,
    settings=None,
    max_dd_limit: float = 0.15,
) -> LeverageAnalysisResult:
    """
    Determine max sustainable leverage under vol, VaR, and drawdown constraints.
    """
    tickers = list(weights.keys())
    w = np.array([weights[t] for t in tickers], dtype=float)
    current_lev = float(np.abs(w).sum())

    port_var = float(w @ cov @ w)
    port_std_d = math.sqrt(max(port_var, _EPSILON))
    port_vol_ann = port_std_d * math.sqrt(_TRADING_DAYS)

    target_vol = getattr(settings, "target_portfolio_vol", 0.12) if settings else 0.12
    max_lev = getattr(settings, "max_leverage", 5.0) if settings else 5.0

    # Max leverage for vol target
    if port_vol_ann > _EPSILON:
        lev_vol = target_vol / port_vol_ann * current_lev
    else:
        lev_vol = max_lev

    # Max leverage for VaR constraint (VaR95 < 3% daily)
    var_limit = 0.03
    z95 = norm.ppf(0.95)
    var_current = z95 * port_std_d
    if var_current > _EPSILON:
        lev_var = var_limit / var_current * current_lev
    else:
        lev_var = max_lev

    # Max leverage for drawdown limit (approximation: max_dd ≈ 2 * vol * sqrt(T))
    # Using Magdon-Ismail formula approximation
    if port_vol_ann > _EPSILON:
        lev_dd = max_dd_limit / (2 * port_vol_ann) * current_lev
    else:
        lev_dd = max_lev

    recommended = min(lev_vol, lev_var, lev_dd, max_lev)
    headroom = (recommended - current_lev) / current_lev if current_lev > _EPSILON else 0.0

    return LeverageAnalysisResult(
        current_leverage=round(current_lev, 4),
        max_leverage_vol_target=round(min(lev_vol, max_lev), 4),
        max_leverage_var_limit=round(min(lev_var, max_lev), 4),
        max_leverage_dd_limit=round(min(lev_dd, max_lev), 4),
        recommended_leverage=round(recommended, 4),
        headroom_pct=round(headroom, 4),
    )


# ═════════════════════════════════════════════════════════════════════════════
# Stress-Adjusted Risk Budgeting
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class StressRiskBudgetResult:
    """Risk budget under both normal and stressed covariance."""
    normal_risk_budget: Dict[str, float]   # Normal regime risk contribution %
    stressed_risk_budget: Dict[str, float]  # Stressed regime risk contribution %
    budget_shift: Dict[str, float]          # Change in risk budget under stress
    concentration_increase: float           # HHI(stressed) - HHI(normal)
    worst_concentrator: str                 # Ticker with largest stressed risk share


def stress_adjusted_risk_budget(
    weights: Dict[str, float],
    cov_normal: np.ndarray,
    stress_intensity: float = 0.3,
) -> StressRiskBudgetResult:
    """
    Compare risk budgets under normal vs stressed covariance.

    Stressed cov: C_stress = (1-η)C + η·11' × avg_var
    where η = stress intensity (0.3 = 30% weight toward perfect correlation).
    """
    tickers = list(weights.keys())
    n = len(tickers)
    w = np.array([weights[t] for t in tickers], dtype=float)

    # Normal risk budget
    engine = PortfolioRiskEngine()
    rb_normal = engine.compute_risk_budget(weights, cov_normal)
    normal_budget = {t: float(rb_normal.get(t, 0)) for t in tickers}

    # Stressed covariance: blend toward perfect correlation
    avg_var = float(np.diag(cov_normal).mean())
    ones = np.ones((n, n))
    cov_stressed = (1 - stress_intensity) * cov_normal + stress_intensity * ones * avg_var

    rb_stressed = engine.compute_risk_budget(weights, cov_stressed)
    stressed_budget = {t: float(rb_stressed.get(t, 0)) for t in tickers}

    # Budget shift
    budget_shift = {t: round(stressed_budget.get(t, 0) - normal_budget.get(t, 0), 6) for t in tickers}

    # Concentration change
    hhi_normal = sum(v ** 2 for v in normal_budget.values())
    hhi_stressed = sum(v ** 2 for v in stressed_budget.values())
    conc_increase = hhi_stressed - hhi_normal

    worst = max(stressed_budget, key=stressed_budget.get) if stressed_budget else "N/A"

    return StressRiskBudgetResult(
        normal_risk_budget=normal_budget,
        stressed_risk_budget=stressed_budget,
        budget_shift=budget_shift,
        concentration_increase=round(conc_increase, 6),
        worst_concentrator=worst,
    )
