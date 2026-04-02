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


# ═════════════════════════════════════════════════════════════════════════════
# Ornstein-Uhlenbeck MLE Estimator
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class OUEstimate:
    """Ornstein-Uhlenbeck process parameter estimates via MLE.

    dX_t = θ(μ - X_t)dt + σ dW_t

    θ : speed of mean-reversion (higher = faster)
    μ : long-run mean level
    σ : diffusion volatility
    half_life : ln(2)/θ
    """
    theta: float            # Mean-reversion speed
    mu: float               # Long-run mean
    sigma: float            # Diffusion coefficient
    half_life: float        # ln(2)/θ in same units as dt
    log_likelihood: float   # Maximized log-likelihood
    n_obs: int


def ou_mle(series: pd.Series, dt: float = 1.0) -> OUEstimate:
    """
    Maximum Likelihood Estimation of Ornstein-Uhlenbeck parameters.

    Uses the exact discrete-time transition density:
      X_{t+dt} | X_t ~ N(X_t·e^{-θdt} + μ(1-e^{-θdt}), σ²(1-e^{-2θdt})/(2θ))

    Parameters
    ----------
    series : pd.Series — the time series to fit
    dt : float — time step (1.0 = 1 day)

    Returns
    -------
    OUEstimate with (θ, μ, σ, half_life, log_likelihood)
    """
    x = series.dropna().values.astype(float)
    n = len(x)
    if n < 30:
        return OUEstimate(theta=0, mu=0, sigma=0, half_life=float("inf"),
                          log_likelihood=float("-inf"), n_obs=n)

    # Step 1: AR(1) regression to get initial estimates
    # X_{t+1} = a + b·X_t + ε
    x_lag = x[:-1]
    x_lead = x[1:]

    b = float(np.corrcoef(x_lag, x_lead)[0, 1] * x_lead.std() / (x_lag.std() + 1e-15))
    a = float(x_lead.mean() - b * x_lag.mean())
    residuals = x_lead - (a + b * x_lag)
    sigma_eps = float(residuals.std())

    # Step 2: Convert AR(1) → OU parameters
    if b <= 0 or b >= 1:
        # No mean-reversion
        return OUEstimate(theta=0, mu=float(x.mean()), sigma=float(x.std()),
                          half_life=float("inf"), log_likelihood=float("-inf"), n_obs=n)

    theta = -np.log(b) / dt
    mu = a / (1 - b)
    sigma_ou = sigma_eps * np.sqrt(2 * theta / (1 - b ** 2 + 1e-15))
    half_life = np.log(2) / theta if theta > 1e-10 else float("inf")

    # Step 3: Exact log-likelihood
    e_neg_theta_dt = np.exp(-theta * dt)
    var_transition = sigma_ou ** 2 * (1 - e_neg_theta_dt ** 2) / (2 * theta + 1e-15)
    if var_transition <= 0:
        var_transition = sigma_eps ** 2

    mu_transition = x_lag * e_neg_theta_dt + mu * (1 - e_neg_theta_dt)
    ll = -0.5 * (n - 1) * np.log(2 * np.pi * var_transition) \
         - 0.5 * np.sum((x_lead - mu_transition) ** 2) / var_transition

    return OUEstimate(
        theta=round(float(theta), 6),
        mu=round(float(mu), 6),
        sigma=round(float(sigma_ou), 6),
        half_life=round(float(half_life), 2),
        log_likelihood=round(float(ll), 2),
        n_obs=n,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Variance Ratio Test (Lo & MacKinlay, 1988)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class VarianceRatioResult:
    """Variance ratio test for random walk vs mean-reversion."""
    vr: float                # VR(q) — ratio of long-horizon to short-horizon variance
    z_stat: float            # Heteroskedasticity-adjusted z-statistic
    p_value: float           # Two-sided p-value
    is_mean_reverting: bool  # True if VR < 1 and significant (p < 0.05)
    q: int                   # Horizon parameter
    interpretation: str


def variance_ratio_test(
    series: pd.Series, q: int = 10,
) -> VarianceRatioResult:
    """
    Lo & MacKinlay (1988) variance ratio test.

    VR(q) = Var(r_t + ... + r_{t+q-1}) / (q · Var(r_t))

    VR < 1 → mean-reversion
    VR = 1 → random walk
    VR > 1 → momentum / trending

    Uses heteroskedasticity-robust z-statistic.
    """
    x = series.dropna().values.astype(float)
    n = len(x)
    if n < q * 3:
        return VarianceRatioResult(vr=1.0, z_stat=0.0, p_value=1.0,
                                   is_mean_reverting=False, q=q,
                                   interpretation="Insufficient data")

    # Returns
    r = np.diff(x)
    T = len(r)
    mu = r.mean()

    # Variances
    sigma_1 = float(np.sum((r - mu) ** 2) / (T - 1))
    r_q = np.array([r[i:i + q].sum() for i in range(T - q + 1)])
    sigma_q = float(np.sum((r_q - q * mu) ** 2) / (T - q))

    vr = sigma_q / (q * sigma_1) if sigma_1 > 1e-15 else 1.0

    # Heteroskedasticity-adjusted z-stat (Lo-MacKinlay)
    delta_hat = 0.0
    for j in range(1, q):
        numer = np.sum((r[j:] - mu) ** 2 * (r[:-j] - mu) ** 2)
        denom = (np.sum((r - mu) ** 2)) ** 2
        delta_j = T * numer / denom if denom > 0 else 0
        weight = (2 * (q - j) / q) ** 2
        delta_hat += weight * delta_j

    z_star = (vr - 1) / np.sqrt(max(delta_hat, 1e-15)) if delta_hat > 0 else 0.0

    # Two-sided p-value (normal approximation)
    from scipy.stats import norm
    p_value = float(2 * norm.sf(abs(z_star)))

    is_mr = vr < 1 and p_value < 0.05

    if vr < 0.85:
        interp = f"Strong mean-reversion (VR={vr:.3f})"
    elif vr < 1.0:
        interp = f"Mild mean-reversion (VR={vr:.3f})"
    elif vr > 1.15:
        interp = f"Trending/momentum (VR={vr:.3f})"
    else:
        interp = f"Near random walk (VR={vr:.3f})"

    return VarianceRatioResult(
        vr=round(vr, 4),
        z_stat=round(z_star, 3),
        p_value=round(p_value, 4),
        is_mean_reverting=is_mr,
        q=q,
        interpretation=interp,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Engle-Granger Cointegration Test
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class CointegrationResult:
    """Engle-Granger two-step cointegration test result."""
    is_cointegrated: bool
    adf_stat: float           # ADF on residuals of cointegrating regression
    adf_pvalue: float
    beta: float               # Hedge ratio (cointegrating coefficient)
    half_life: float          # OU half-life of spread
    spread_z: float           # Current z-score of the spread
    spread_std: float         # Standard deviation of spread


def engle_granger_coint(
    y: pd.Series, x: pd.Series, significance: float = 0.05,
) -> CointegrationResult:
    """
    Engle-Granger two-step cointegration test.

    Step 1: OLS regression y = α + β·x + ε
    Step 2: ADF test on residuals ε (must be stationary for cointegration)
    """
    aligned = pd.DataFrame({"y": y, "x": x}).dropna()
    n = len(aligned)
    if n < 60:
        return CointegrationResult(
            is_cointegrated=False, adf_stat=0, adf_pvalue=1,
            beta=0, half_life=float("inf"), spread_z=0, spread_std=0,
        )

    # Step 1: OLS
    y_vals = aligned["y"].values
    x_vals = aligned["x"].values
    x_with_const = np.column_stack([np.ones(n), x_vals])
    coeffs, _, _, _ = np.linalg.lstsq(x_with_const, y_vals, rcond=None)
    alpha, beta = coeffs

    # Residuals (spread)
    spread = y_vals - (alpha + beta * x_vals)
    spread_std = float(spread.std())
    spread_z = float(spread[-1] / spread_std) if spread_std > 1e-10 else 0.0

    # Step 2: ADF on spread
    adf_stat, adf_p = _adf_test(pd.Series(spread))

    # Half-life of spread
    hl = _estimate_half_life(pd.Series(spread))

    return CointegrationResult(
        is_cointegrated=adf_p < significance,
        adf_stat=round(adf_stat, 4),
        adf_pvalue=round(adf_p, 4),
        beta=round(beta, 6),
        half_life=round(hl, 1),
        spread_z=round(spread_z, 3),
        spread_std=round(spread_std, 6),
    )
