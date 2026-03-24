"""
analytics/signal_mean_reversion.py
===================================
Signal Stack — Layer 3: Mean-Reversion Quality Score (S^mr)

Ensures each trade candidate has genuine mean-reverting dynamics,
not just a large z-score in a random walk.

Three sub-components:
  (a) OU half-life quality — AR(1) estimation, sweet-spot [5, 90] days
  (b) ADF stationarity — Augmented Dickey-Fuller test p-value
  (c) Hurst exponent — regime classification (H < 0.5 = mean-reverting)

  S^mr = w_hl · f_hl(hl) + w_adf · f_adf(p) + w_hurst · f_hurst(H)

Ref: Ornstein-Uhlenbeck calibration (Uhlenbeck & Ornstein, 1930)
Ref: ADF test (Dickey & Fuller, 1979)
Ref: Rescaled-range Hurst exponent (Hurst, 1951; Mandelbrot, 1968)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MeanReversionResult:
    """Layer 3 output per trade candidate."""
    ticker: str

    # Sub-component (a): OU half-life
    half_life_days: float          # Estimated AR(1) half-life
    half_life_quality: float       # f_hl ∈ [0, 1]

    # Sub-component (b): ADF stationarity
    adf_stat: float                # ADF test statistic
    adf_pvalue: float              # ADF p-value
    adf_quality: float             # f_adf ∈ [0, 1]

    # Sub-component (c): Hurst exponent
    hurst_exponent: float          # H ∈ [0, 1]
    hurst_quality: float           # f_hurst ∈ [0, 1]

    # Combined
    mean_reversion_score: float    # S^mr ∈ [0, 1]
    label: str                     # "STRONG_MR" / "MODERATE_MR" / "WEAK_MR" / "NO_MR"
    rationale: str


# ─────────────────────────────────────────────────────────────────────────────
# Sub-component (a): OU half-life
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_half_life(series: pd.Series, min_obs: int = 60) -> float:
    """
    AR(1) half-life: x_t = a + phi * x_{t-1} + eps
    hl = -ln(2) / ln(phi)   for 0 < phi < 1
    """
    x = series.dropna().astype(float)
    if len(x) < min_obs:
        return float("nan")

    x_lag = x.shift(1).dropna()
    x_now = x.loc[x_lag.index]
    if len(x_lag) < min_obs:
        return float("nan")

    X = np.vstack([np.ones(len(x_lag)), x_lag.values]).T
    y = x_now.values
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        phi = float(beta[1])
    except Exception:
        return float("nan")

    if phi <= 0.0 or phi >= 0.999:
        return float("nan")

    return float(-math.log(2.0) / math.log(phi))


def _half_life_quality(hl: float, sweet_lo: float = 5.0, sweet_hi: float = 90.0) -> float:
    """
    Map half-life to quality score:
      - Sweet spot [5, 90] → high quality
      - Very short (< 5) → noisy / micro-structure
      - Very long (> 90) → too slow for practical trade horizon
      - NaN (no convergence) → 0
    """
    if not math.isfinite(hl):
        return 0.0

    if hl < 2.0:
        return 0.15  # Too fast, probably noise
    if hl < sweet_lo:
        return 0.30 + 0.30 * (hl - 2.0) / (sweet_lo - 2.0)
    if hl <= sweet_hi:
        # Peak quality around 20-40 days, tapering
        center = (sweet_lo + sweet_hi) / 2.0
        spread = (sweet_hi - sweet_lo) / 2.0
        return 0.65 + 0.35 * math.exp(-((hl - center) ** 2) / (2 * spread ** 2))
    if hl <= 180:
        return max(0.20, 0.60 * math.exp(-0.01 * (hl - sweet_hi)))
    return 0.10  # Very slow mean reversion


# ─────────────────────────────────────────────────────────────────────────────
# Sub-component (b): ADF stationarity
# ─────────────────────────────────────────────────────────────────────────────

def _adf_test(series: pd.Series, maxlag: int = 12) -> tuple:
    """
    Augmented Dickey-Fuller test.
    Returns (adf_stat, p_value).
    Uses statsmodels if available, else a simplified regression-based approach.
    """
    x = series.dropna().astype(float)
    if len(x) < 30:
        return float("nan"), 1.0

    try:
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(x.values, maxlag=maxlag, autolag="AIC", regression="c")
        return float(result[0]), float(result[1])
    except ImportError:
        pass

    # Fallback: simplified DF regression (no augmentation)
    dx = x.diff().dropna()
    x_lag = x.shift(1).loc[dx.index]
    n = len(dx)
    if n < 30:
        return float("nan"), 1.0

    X = np.vstack([np.ones(n), x_lag.values]).T
    y = dx.values
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X @ beta
        se = np.sqrt(np.sum(resid ** 2) / (n - 2))
        var_beta = se ** 2 * np.linalg.inv(X.T @ X)
        t_stat = beta[1] / max(1e-12, np.sqrt(var_beta[1, 1]))
    except Exception:
        return float("nan"), 1.0

    # Approximate p-value from DF distribution (MacKinnon critical values)
    # Critical values for n > 250 with constant: 1%=-3.43, 5%=-2.86, 10%=-2.57
    if t_stat <= -3.43:
        p_approx = 0.005
    elif t_stat <= -2.86:
        p_approx = 0.03
    elif t_stat <= -2.57:
        p_approx = 0.07
    elif t_stat <= -1.94:
        p_approx = 0.20
    else:
        p_approx = min(1.0, 0.30 + 0.70 * (1.0 / (1.0 + math.exp(-t_stat))))

    return float(t_stat), p_approx


def _adf_quality(p_value: float) -> float:
    """
    Map ADF p-value to quality score:
      p < 0.01 → ~1.0 (strongly stationary)
      p < 0.05 → ~0.75
      p < 0.10 → ~0.50
      p > 0.30 → ~0.10
    """
    if not math.isfinite(p_value):
        return 0.0
    if p_value <= 0.01:
        return 1.0
    if p_value <= 0.05:
        return 0.70 + 0.30 * (0.05 - p_value) / 0.04
    if p_value <= 0.10:
        return 0.45 + 0.25 * (0.10 - p_value) / 0.05
    if p_value <= 0.30:
        return 0.15 + 0.30 * (0.30 - p_value) / 0.20
    return max(0.0, 0.15 * (1.0 - p_value))


# ─────────────────────────────────────────────────────────────────────────────
# Sub-component (c): Hurst exponent (rescaled range)
# ─────────────────────────────────────────────────────────────────────────────

def _hurst_exponent(series: pd.Series, min_window: int = 20, max_window: int = 200) -> float:
    """
    Estimate Hurst exponent via rescaled range (R/S) analysis.
      H < 0.5  → mean-reverting
      H = 0.5  → random walk
      H > 0.5  → trending

    Uses logarithmic regression of R/S vs window size.
    """
    x = series.dropna().values.astype(float)
    n = len(x)
    if n < max(60, min_window * 2):
        return float("nan")

    # Generate window sizes (log-spaced)
    min_w = max(min_window, 10)
    max_w = min(max_window, n // 2)
    if max_w <= min_w:
        return float("nan")

    windows = np.unique(np.logspace(
        np.log10(min_w), np.log10(max_w), num=15
    ).astype(int))
    windows = windows[windows >= min_w]
    if len(windows) < 4:
        return float("nan")

    log_n = []
    log_rs = []

    for w in windows:
        n_blocks = n // w
        if n_blocks < 1:
            continue
        rs_vals = []
        for i in range(n_blocks):
            block = x[i * w: (i + 1) * w]
            mean_block = block.mean()
            deviation = np.cumsum(block - mean_block)
            R = deviation.max() - deviation.min()
            S = block.std(ddof=1)
            if S > 1e-12:
                rs_vals.append(R / S)

        if rs_vals:
            log_n.append(math.log(w))
            log_rs.append(math.log(np.mean(rs_vals)))

    if len(log_n) < 4:
        return float("nan")

    # Linear regression: log(R/S) = H * log(n) + c
    log_n_arr = np.array(log_n)
    log_rs_arr = np.array(log_rs)
    try:
        X = np.vstack([log_n_arr, np.ones(len(log_n_arr))]).T
        beta = np.linalg.lstsq(X, log_rs_arr, rcond=None)[0]
        H = float(beta[0])
    except Exception:
        return float("nan")

    return max(0.0, min(1.0, H))


def _hurst_quality(H: float) -> float:
    """
    Map Hurst exponent to mean-reversion quality:
      H < 0.35 → strong MR → 1.0
      H = 0.40 → moderate MR → 0.75
      H = 0.50 → random walk → 0.30
      H > 0.60 → trending → 0.05
    """
    if not math.isfinite(H):
        return 0.3  # Unknown → neutral assumption

    if H <= 0.30:
        return 1.0
    if H <= 0.40:
        return 0.80 + 0.20 * (0.40 - H) / 0.10
    if H <= 0.50:
        return 0.35 + 0.45 * (0.50 - H) / 0.10
    if H <= 0.60:
        return 0.10 + 0.25 * (0.60 - H) / 0.10
    return max(0.0, 0.10 * (1.0 - H))


# ─────────────────────────────────────────────────────────────────────────────
# Combined Layer 3 score
# ─────────────────────────────────────────────────────────────────────────────

def compute_mean_reversion_score(
    residual_series: pd.Series,
    ticker: str,
    *,
    w_hl: float = 0.35,
    w_adf: float = 0.40,
    w_hurst: float = 0.25,
    hl_sweet_lo: float = 5.0,
    hl_sweet_hi: float = 90.0,
    adf_maxlag: int = 12,
    min_obs: int = 60,
) -> MeanReversionResult:
    """
    Layer 3: Mean-Reversion Quality Score.

    S^mr = w_hl · f_hl(hl) + w_adf · f_adf(p) + w_hurst · f_hurst(H)

    Parameters
    ----------
    residual_series : Residual process x_t^(j) for this candidate
    ticker          : Identifier
    w_hl, w_adf, w_hurst : Sub-component weights (must sum ≈ 1)
    hl_sweet_lo/hi  : Sweet-spot range for half-life
    adf_maxlag      : Max lag for ADF test
    min_obs         : Minimum observations required

    Returns
    -------
    MeanReversionResult
    """
    x = residual_series.dropna()

    # (a) Half-life
    hl = _estimate_half_life(x, min_obs=min_obs)
    hl_q = _half_life_quality(hl, sweet_lo=hl_sweet_lo, sweet_hi=hl_sweet_hi)

    # (b) ADF stationarity
    adf_stat, adf_p = _adf_test(x, maxlag=adf_maxlag)
    adf_q = _adf_quality(adf_p)

    # (c) Hurst exponent — applied to RETURNS (differences), not levels.
    # R/S on levels gives H≈0.9 (false trending) because levels are non-stationary.
    # R/S on returns correctly detects mean-reversion (H<0.5) in the residual process.
    x_returns = x.diff().dropna()
    hurst = _hurst_exponent(x_returns) if len(x_returns) >= 60 else float("nan")
    hurst_q = _hurst_quality(hurst)

    # Weighted combination
    total_w = w_hl + w_adf + w_hurst
    score = (w_hl * hl_q + w_adf * adf_q + w_hurst * hurst_q) / max(total_w, 1e-9)
    score = max(0.0, min(1.0, score))

    # Label
    if score >= 0.65:
        label = "STRONG_MR"
        rationale = (
            f"Strong mean-reversion: hl={hl:.0f}d, ADF p={adf_p:.3f}, H={hurst:.2f}"
        )
    elif score >= 0.40:
        label = "MODERATE_MR"
        rationale = (
            f"Moderate MR quality: hl={hl:.0f}d, ADF p={adf_p:.3f}, H={hurst:.2f}"
        )
    elif score >= 0.20:
        label = "WEAK_MR"
        rationale = (
            f"Weak MR evidence: hl={hl:.0f}d, ADF p={adf_p:.3f}, H={hurst:.2f} — caution"
        )
    else:
        label = "NO_MR"
        rationale = (
            f"No mean-reversion detected: hl={hl:.0f}d, ADF p={adf_p:.3f}, H={hurst:.2f} — avoid"
        )

    return MeanReversionResult(
        ticker=ticker,
        half_life_days=round(hl, 1) if math.isfinite(hl) else float("nan"),
        half_life_quality=round(hl_q, 4),
        adf_stat=round(adf_stat, 4) if math.isfinite(adf_stat) else float("nan"),
        adf_pvalue=round(adf_p, 4),
        adf_quality=round(adf_q, 4),
        hurst_exponent=round(hurst, 4) if math.isfinite(hurst) else float("nan"),
        hurst_quality=round(hurst_q, 4),
        mean_reversion_score=round(score, 4),
        label=label,
        rationale=rationale,
    )


def batch_mean_reversion_scores(
    residuals: dict[str, pd.Series],
    **kwargs,
) -> dict[str, MeanReversionResult]:
    """Compute Layer 3 scores for all candidates. Returns {ticker: MeanReversionResult}."""
    results = {}
    for ticker, series in residuals.items():
        try:
            results[ticker] = compute_mean_reversion_score(series, ticker, **kwargs)
        except Exception as e:
            log.warning("MR score failed for %s: %s", ticker, e)
            results[ticker] = MeanReversionResult(
                ticker=ticker,
                half_life_days=float("nan"),
                half_life_quality=0.0,
                adf_stat=float("nan"),
                adf_pvalue=1.0,
                adf_quality=0.0,
                hurst_exponent=float("nan"),
                hurst_quality=0.3,
                mean_reversion_score=0.0,
                label="ERROR",
                rationale=f"Computation failed: {e}",
            )
    return results
