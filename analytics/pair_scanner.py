"""
analytics/pair_scanner.py
===========================
Institutional-Grade Pair/Basket Scanner + Position Correlation Tracker

Multi-method cointegration analysis for sector ETF pairs and baskets:

  1. Engle-Granger (1987) — pairwise cointegration (ADF on spread)
  2. Johansen (1991) — multivariate cointegration (VAR-based, 3+ leg trades)
  3. Kalman Filter — real-time adaptive hedge ratios (dynamic beta)
  4. Dynamic z-score thresholds — regime-adjusted entry/exit levels
  5. Spread regime detection — trending vs mean-reverting spread classification
  6. Triangle/basket trades — 3-leg market-neutral constructions

For each pair/basket:
  - Cointegration: ADF p-value + Johansen trace/eigenvalue statistics
  - Spread: z-score with dynamic thresholds (vol-regime adjusted)
  - Hedge ratio: OLS (static) + Kalman (adaptive)
  - Half-life: AR(1) + OU MLE
  - Spread regime: trending/MR/random walk (variance ratio test)

Position correlation tracker:
  - Full correlation matrix of open positions
  - Concentrated risk detection (>0.7 pairwise corr)
  - Diversification ratio and sizing adjustments
  - Eigenvalue decomposition for hidden factor exposure

Ref: Engle & Granger (1987) — Cointegration and Error Correction
Ref: Johansen (1991) — Estimation and Hypothesis Testing of Cointegration
Ref: Kalman (1960) — A New Approach to Linear Filtering
Ref: Gatev, Goetzmann & Rouwenhorst (2006) — Pairs Trading
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class PairSignal:
    """Signal for one sector pair with multi-method analysis."""
    pair_name: str          # e.g., "XLK-XLU"
    ticker_a: str
    ticker_b: str
    spread_z: float         # Z-score of spread
    half_life: float        # Days
    hedge_ratio: float      # h: spread = ln(A) - h·ln(B) (OLS static)
    direction: str          # "LONG_SPREAD" / "SHORT_SPREAD" / "NEUTRAL"
    adf_pvalue: float       # Engle-Granger cointegration test p-value
    correlation: float      # Current 60d correlation
    signal_strength: float  # Combined score [0, 1]

    # Kalman filter hedge ratio (adaptive)
    kalman_hedge_ratio: float = 0.0     # Real-time adaptive beta
    kalman_spread_z: float = 0.0        # Z-score using Kalman-filtered spread

    # Dynamic thresholds (regime-adjusted)
    entry_threshold: float = 2.0        # Z-score entry level (regime-adjusted)
    exit_threshold: float = 0.5         # Z-score exit level
    stop_threshold: float = 4.0         # Z-score stop level

    # Spread regime
    spread_variance_ratio: float = 1.0  # VR(10): <1 = MR, >1 = trending
    spread_regime: str = "UNKNOWN"      # "MEAN_REVERTING" / "TRENDING" / "RANDOM_WALK"

    # Hurst exponent of spread
    hurst: float = 0.5

    # Expected profit (per unit notional)
    expected_profit_bps: float = 0.0    # |z| × spread_vol × MR_probability × 10000


@dataclass
class JohansenResult:
    """Johansen multivariate cointegration test result."""
    n_cointegrating: int              # Number of cointegrating relationships
    trace_stats: List[float]          # Trace test statistics
    trace_crit_95: List[float]        # 95% critical values
    eigen_stats: List[float]          # Eigenvalue test statistics
    eigen_crit_95: List[float]        # 95% critical values
    cointegrating_vector: Optional[np.ndarray] = None  # Weights for the first relationship
    hedge_ratios: Optional[Dict[str, float]] = None    # Normalized hedge ratios


@dataclass
class BasketSignal:
    """Signal for a 3+ leg basket trade (from Johansen)."""
    basket_name: str                  # e.g., "XLK-XLF-XLU"
    tickers: List[str]
    weights: Dict[str, float]         # Normalized weights (sum abs = 1)
    spread_z: float
    half_life: float
    direction: str                    # "LONG_BASKET" / "SHORT_BASKET"
    johansen_n_coint: int             # N cointegrating relationships
    signal_strength: float
    spread_regime: str = "UNKNOWN"


@dataclass
class KalmanState:
    """State of a Kalman filter for adaptive hedge ratio estimation."""
    beta: float                       # Current hedge ratio estimate
    P: float                          # Estimation error variance
    Q: float = 1e-5                   # Process noise (how fast beta changes)
    R: float = 1e-3                   # Measurement noise (spread noise)


@dataclass
class PositionCorrelationReport:
    """Correlation analysis of current open positions."""
    n_positions: int
    avg_pairwise_corr: float
    max_pairwise_corr: float
    max_corr_pair: Tuple[str, str]
    corr_matrix: Dict[str, Dict[str, float]]
    concentrated_pairs: List[Tuple[str, str, float]]  # Pairs with corr > 0.7
    diversification_ratio: float  # 1 = fully diversified, 0 = all same


def scan_pairs(
    prices: pd.DataFrame,
    sectors: List[str],
    window: int = 60,
    z_lookback: int = 60,
    min_half_life: float = 3.0,
    max_half_life: float = 60.0,
) -> List[PairSignal]:
    """
    Scan all sector pairs for cointegrated spreads.

    Returns list sorted by signal_strength descending.
    """
    avail = [s for s in sectors if s in prices.columns]
    log_prices = np.log(prices[avail].dropna())
    n = len(log_prices)

    if n < window + 30:
        return []

    results = []
    for i, a in enumerate(avail):
        for j, b in enumerate(avail):
            if i >= j:
                continue

            pair_name = f"{a}-{b}"

            try:
                # Hedge ratio (OLS on recent window)
                y = log_prices[a].iloc[-window:]
                x = log_prices[b].iloc[-window:]
                beta = float(np.cov(y, x)[0, 1] / np.var(x)) if np.var(x) > 1e-12 else 1.0

                # Spread
                spread = log_prices[a] - beta * log_prices[b]

                # Z-score
                recent = spread.iloc[-z_lookback:]
                mu = float(recent.mean())
                sd = float(recent.std(ddof=1))
                z = (float(spread.iloc[-1]) - mu) / sd if sd > 1e-10 else 0.0

                # Half-life (AR1)
                hl = _estimate_half_life(spread.iloc[-window:])

                # ADF test (simplified)
                adf_p = _simplified_adf_pvalue(spread.iloc[-window:])

                # Correlation
                rets_a = np.log(prices[a] / prices[a].shift(1)).dropna()
                rets_b = np.log(prices[b] / prices[b].shift(1)).dropna()
                corr = float(rets_a.tail(60).corr(rets_b.tail(60)))

                # Direction
                if abs(z) < 0.5:
                    direction = "NEUTRAL"
                elif z < 0:
                    direction = "LONG_SPREAD"   # Spread below mean → buy A, sell B
                else:
                    direction = "SHORT_SPREAD"  # Spread above mean → sell A, buy B

                # Signal strength: combine z, HL quality, ADF
                z_component = min(1.0, abs(z) / 2.5) * 0.4
                hl_component = 0.0
                if math.isfinite(hl) and min_half_life <= hl <= max_half_life:
                    hl_component = 0.3 * math.exp(-abs(hl - 20) / 30)
                adf_component = max(0, (0.10 - adf_p) / 0.10) * 0.3 if adf_p < 0.10 else 0
                strength = z_component + hl_component + adf_component

                results.append(PairSignal(
                    pair_name=pair_name, ticker_a=a, ticker_b=b,
                    spread_z=round(z, 4), half_life=round(hl, 1) if math.isfinite(hl) else float("nan"),
                    hedge_ratio=round(beta, 4), direction=direction,
                    adf_pvalue=round(adf_p, 4), correlation=round(corr, 4),
                    signal_strength=round(strength, 4),
                ))
            except Exception as e:
                log.debug("Pair %s failed: %s", pair_name, e)

    results.sort(key=lambda p: p.signal_strength, reverse=True)
    return results


def compute_position_correlations(
    prices: pd.DataFrame,
    positions: List[Dict],
    window: int = 60,
) -> PositionCorrelationReport:
    """
    Compute correlation matrix between open positions.

    positions: list of dicts with 'ticker' and 'direction' keys
    """
    tickers = [p["ticker"] for p in positions if p.get("ticker") in prices.columns]
    if len(tickers) < 2:
        return PositionCorrelationReport(
            n_positions=len(tickers), avg_pairwise_corr=0, max_pairwise_corr=0,
            max_corr_pair=("", ""), corr_matrix={},
            concentrated_pairs=[], diversification_ratio=1.0,
        )

    log_rets = np.log(prices[tickers] / prices[tickers].shift(1)).dropna().tail(window)
    C = log_rets.corr()

    # Extract upper triangle
    n = len(tickers)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            c = float(C.iloc[i, j])
            pairs.append((tickers[i], tickers[j], c))

    avg_corr = float(np.mean([p[2] for p in pairs])) if pairs else 0
    max_pair = max(pairs, key=lambda p: abs(p[2])) if pairs else ("", "", 0)
    concentrated = [(a, b, c) for a, b, c in pairs if abs(c) > 0.70]

    # Diversification ratio: 1 - (avg_corr)
    div_ratio = max(0, 1.0 - abs(avg_corr))

    # Build matrix dict
    matrix = {}
    for t in tickers:
        matrix[t] = {t2: round(float(C.loc[t, t2]), 4) for t2 in tickers}

    return PositionCorrelationReport(
        n_positions=len(tickers),
        avg_pairwise_corr=round(avg_corr, 4),
        max_pairwise_corr=round(abs(max_pair[2]), 4),
        max_corr_pair=(max_pair[0], max_pair[1]),
        corr_matrix=matrix,
        concentrated_pairs=[(a, b, round(c, 4)) for a, b, c in concentrated],
        diversification_ratio=round(div_ratio, 4),
    )


# ── Helpers ──────────────────────────────────────────────────────────────────

def _estimate_half_life(series: pd.Series, min_obs: int = 30) -> float:
    """AR(1) half-life estimation."""
    x = series.dropna().astype(float)
    if len(x) < min_obs:
        return float("nan")
    x_lag = x.shift(1).dropna()
    x_now = x.loc[x_lag.index]
    X = np.vstack([np.ones(len(x_lag)), x_lag.values]).T
    try:
        beta = np.linalg.lstsq(X, x_now.values, rcond=None)[0]
        phi = float(beta[1])
        if 0 < phi < 0.999:
            return -math.log(2) / math.log(phi)
    except Exception:
        pass
    return float("nan")


def _simplified_adf_pvalue(series: pd.Series) -> float:
    """Simplified ADF p-value via regression t-stat."""
    try:
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(series.dropna().values, maxlag=5, autolag="AIC")
        return float(result[1])
    except ImportError:
        pass
    # Fallback: simple t-stat
    x = series.dropna()
    dx = x.diff().dropna()
    x_lag = x.shift(1).loc[dx.index]
    X = np.vstack([np.ones(len(x_lag)), x_lag.values]).T
    try:
        beta = np.linalg.lstsq(X, dx.values, rcond=None)[0]
        resid = dx.values - X @ beta
        se = np.sqrt(np.sum(resid ** 2) / (len(resid) - 2))
        var_b = se ** 2 * np.linalg.inv(X.T @ X)
        t = beta[1] / np.sqrt(var_b[1, 1])
        if t <= -3.43:
            return 0.005
        if t <= -2.86:
            return 0.03
        if t <= -2.57:
            return 0.07
        return 0.30
    except Exception:
        return 1.0


# ═════════════════════════════════════════════════════════════════════════════
# Kalman Filter for Adaptive Hedge Ratios
# ═════════════════════════════════════════════════════════════════════════════

def kalman_hedge_ratio(
    y: pd.Series, x: pd.Series, Q: float = 1e-5, R: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Kalman filter estimation of time-varying hedge ratio.

    Model: y_t = beta_t · x_t + epsilon_t
           beta_t = beta_{t-1} + eta_t

    Parameters
    ----------
    y : pd.Series — dependent series (log price of asset A)
    x : pd.Series — independent series (log price of asset B)
    Q : float — process noise variance (controls beta adaptation speed)
    R : float — measurement noise variance

    Returns
    -------
    betas : np.ndarray — time series of hedge ratios
    spreads : np.ndarray — time series of Kalman-filtered spreads
    """
    y_vals = y.values.astype(float)
    x_vals = x.values.astype(float)
    n = len(y_vals)

    # Initialize
    beta = np.zeros(n)
    P = np.zeros(n)  # Error covariance
    beta[0] = y_vals[0] / x_vals[0] if abs(x_vals[0]) > 1e-10 else 1.0
    P[0] = 1.0

    for t in range(1, n):
        # Predict
        beta_pred = beta[t - 1]
        P_pred = P[t - 1] + Q

        # Update
        x_t = x_vals[t]
        y_t = y_vals[t]
        innovation = y_t - beta_pred * x_t
        S = x_t ** 2 * P_pred + R  # Innovation variance
        K = P_pred * x_t / S if S > 1e-15 else 0  # Kalman gain

        beta[t] = beta_pred + K * innovation
        P[t] = (1 - K * x_t) * P_pred

    spreads = y_vals - beta * x_vals
    return beta, spreads


def kalman_pair_signal(
    prices: pd.DataFrame, ticker_a: str, ticker_b: str,
    z_lookback: int = 60, Q: float = 1e-5, R: float = 1e-3,
) -> Optional[Dict]:
    """
    Compute Kalman-filtered pair signal.

    Returns dict with: kalman_beta, kalman_spread_z, kalman_spread_std
    """
    if ticker_a not in prices.columns or ticker_b not in prices.columns:
        return None

    log_a = np.log(prices[ticker_a].dropna())
    log_b = np.log(prices[ticker_b].dropna())

    # Align
    common = log_a.index.intersection(log_b.index)
    if len(common) < z_lookback + 30:
        return None

    log_a = log_a.loc[common]
    log_b = log_b.loc[common]

    betas, spreads = kalman_hedge_ratio(log_a, log_b, Q=Q, R=R)

    # Z-score of Kalman spread
    recent = spreads[-z_lookback:]
    mu = float(recent.mean())
    sd = float(recent.std())
    z = (spreads[-1] - mu) / sd if sd > 1e-10 else 0.0

    return {
        "kalman_beta": round(float(betas[-1]), 6),
        "kalman_spread_z": round(z, 4),
        "kalman_spread_std": round(sd, 6),
        "beta_stability": round(float(np.std(betas[-60:])), 6) if len(betas) >= 60 else 0.0,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Johansen Multivariate Cointegration
# ═════════════════════════════════════════════════════════════════════════════

def johansen_test(
    prices: pd.DataFrame, tickers: List[str],
    det_order: int = 0, k_ar_diff: int = 1,
) -> Optional[JohansenResult]:
    """
    Johansen cointegration test for 2+ series.

    Tests for the number of cointegrating relationships among N variables
    using the trace and maximum eigenvalue statistics.

    Parameters
    ----------
    prices : pd.DataFrame — price data (log prices used internally)
    tickers : list — 2+ tickers to test
    det_order : int — deterministic trend (-1=none, 0=constant, 1=trend)
    k_ar_diff : int — number of lagged differences in VAR

    Returns
    -------
    JohansenResult with n_cointegrating, stats, critical values, and hedge ratios
    """
    avail = [t for t in tickers if t in prices.columns]
    if len(avail) < 2:
        return None

    log_p = np.log(prices[avail].dropna())
    if len(log_p) < 60:
        return None

    try:
        from statsmodels.tsa.vector_ar.vecm import coint_johansen
        result = coint_johansen(log_p.values, det_order=det_order, k_ar_diff=k_ar_diff)

        trace_stats = result.lr1.tolist()       # Trace statistics
        trace_crit = result.cvt[:, 1].tolist()  # 95% critical values (column 1)
        eigen_stats = result.lr2.tolist()        # Max eigenvalue statistics
        eigen_crit = result.cvm[:, 1].tolist()   # 95% critical values

        # Count cointegrating relationships (trace test)
        n_coint = sum(1 for ts, cv in zip(trace_stats, trace_crit) if ts > cv)

        # Extract first cointegrating vector
        coint_vec = result.evec[:, 0]

        # Normalize: first element = 1.0
        if abs(coint_vec[0]) > 1e-10:
            coint_vec = coint_vec / coint_vec[0]

        hedge_ratios = {avail[i]: round(float(coint_vec[i]), 6) for i in range(len(avail))}

        return JohansenResult(
            n_cointegrating=n_coint,
            trace_stats=[round(s, 4) for s in trace_stats],
            trace_crit_95=[round(c, 4) for c in trace_crit],
            eigen_stats=[round(s, 4) for s in eigen_stats],
            eigen_crit_95=[round(c, 4) for c in eigen_crit],
            cointegrating_vector=coint_vec,
            hedge_ratios=hedge_ratios,
        )
    except ImportError:
        log.debug("statsmodels not available for Johansen test")
        return None
    except Exception as e:
        log.debug("Johansen test failed: %s", e)
        return None


# ═════════════════════════════════════════════════════════════════════════════
# Dynamic Z-Score Thresholds (Regime-Adjusted)
# ═════════════════════════════════════════════════════════════════════════════

def dynamic_z_thresholds(
    spread: pd.Series,
    vix: Optional[pd.Series] = None,
    base_entry: float = 2.0,
    base_exit: float = 0.5,
    base_stop: float = 4.0,
) -> Tuple[float, float, float]:
    """
    Compute regime-adjusted entry/exit/stop z-score thresholds.

    In high-volatility regimes (VIX > 25):
      - Entry threshold INCREASES (require stronger signal)
      - Stop threshold DECREASES (tighter risk)
    In low-volatility regimes (VIX < 15):
      - Entry threshold DECREASES (smaller z-scores are meaningful)

    Also adjusts based on spread volatility regime:
      - High spread vol → wider thresholds
      - Low spread vol → tighter thresholds
    """
    # Spread volatility regime
    spread_vol = float(spread.iloc[-60:].std()) if len(spread) >= 60 else float(spread.std())
    spread_vol_20d = float(spread.iloc[-20:].std()) if len(spread) >= 20 else spread_vol
    vol_ratio = spread_vol_20d / spread_vol if spread_vol > 1e-10 else 1.0

    # VIX regime adjustment
    vix_mult = 1.0
    if vix is not None and len(vix) > 0:
        current_vix = float(vix.iloc[-1])
        if current_vix > 30:
            vix_mult = 1.4  # Much wider entry in crisis
        elif current_vix > 25:
            vix_mult = 1.2
        elif current_vix < 15:
            vix_mult = 0.8  # Tighter entry in calm

    # Spread vol adjustment
    spread_mult = max(0.7, min(1.5, vol_ratio))

    entry = base_entry * vix_mult * spread_mult
    exit_t = base_exit * (1 / vix_mult)  # Exit tighter in high vol
    stop = base_stop * (1 / vix_mult) * 0.9  # Tighter stop in high vol

    return round(entry, 2), round(exit_t, 2), round(stop, 2)


# ═════════════════════════════════════════════════════════════════════════════
# Spread Regime Detection
# ═════════════════════════════════════════════════════════════════════════════

def classify_spread_regime(spread: pd.Series, q: int = 10) -> Tuple[str, float]:
    """
    Classify whether a spread is mean-reverting, trending, or random walk
    using the variance ratio test.

    Returns (regime, variance_ratio).
    """
    x = spread.dropna().values.astype(float)
    n = len(x)
    if n < q * 3:
        return "UNKNOWN", 1.0

    r = np.diff(x)
    T = len(r)
    mu = r.mean()

    sigma_1 = float(np.sum((r - mu) ** 2) / (T - 1))
    r_q = np.array([r[i:i + q].sum() for i in range(T - q + 1)])
    sigma_q = float(np.sum((r_q - q * mu) ** 2) / (T - q))

    vr = sigma_q / (q * sigma_1) if sigma_1 > 1e-15 else 1.0

    if vr < 0.7:
        regime = "MEAN_REVERTING"
    elif vr > 1.3:
        regime = "TRENDING"
    else:
        regime = "RANDOM_WALK"

    return regime, round(vr, 4)


# ═════════════════════════════════════════════════════════════════════════════
# Basket/Triangle Trade Scanner (3+ legs)
# ═════════════════════════════════════════════════════════════════════════════

def scan_baskets(
    prices: pd.DataFrame,
    sectors: List[str],
    max_legs: int = 3,
    min_coint: int = 1,
    z_lookback: int = 60,
) -> List[BasketSignal]:
    """
    Scan all combinations of 3 sectors for cointegrated baskets.

    Uses Johansen test to find multivariate cointegration.
    Returns baskets sorted by signal strength.
    """
    from itertools import combinations

    avail = [s for s in sectors if s in prices.columns]
    if len(avail) < max_legs:
        return []

    log_prices = np.log(prices[avail].dropna())
    if len(log_prices) < 120:
        return []

    results = []
    for combo in combinations(avail, max_legs):
        tickers = list(combo)
        joh = johansen_test(prices, tickers)
        if joh is None or joh.n_cointegrating < min_coint:
            continue

        # Construct spread using cointegrating vector
        log_p = np.log(prices[tickers].dropna())
        if joh.cointegrating_vector is None:
            continue

        spread = log_p.values @ joh.cointegrating_vector
        spread = pd.Series(spread, index=log_p.index)

        # Z-score
        recent = spread.iloc[-z_lookback:]
        mu = float(recent.mean())
        sd = float(recent.std())
        z = (float(spread.iloc[-1]) - mu) / sd if sd > 1e-10 else 0.0

        # Half-life
        hl = _estimate_half_life(spread.iloc[-120:])

        # Direction
        if abs(z) < 0.5:
            direction = "NEUTRAL"
        elif z < 0:
            direction = "LONG_BASKET"
        else:
            direction = "SHORT_BASKET"

        # Spread regime
        regime, vr = classify_spread_regime(spread)

        # Signal strength
        z_comp = min(1.0, abs(z) / 3.0) * 0.35
        coint_comp = joh.n_cointegrating * 0.25
        hl_comp = 0.0
        if math.isfinite(hl) and 5 <= hl <= 60:
            hl_comp = 0.2 * math.exp(-abs(hl - 20) / 30)
        mr_comp = 0.2 * max(0, 1.0 - vr) if vr < 1.0 else 0.0
        strength = z_comp + coint_comp + hl_comp + mr_comp

        # Normalize weights
        weights = joh.hedge_ratios or {}
        total_abs = sum(abs(v) for v in weights.values()) or 1.0
        norm_weights = {k: round(v / total_abs, 4) for k, v in weights.items()}

        results.append(BasketSignal(
            basket_name="-".join(tickers),
            tickers=tickers,
            weights=norm_weights,
            spread_z=round(z, 4),
            half_life=round(hl, 1) if math.isfinite(hl) else float("nan"),
            direction=direction,
            johansen_n_coint=joh.n_cointegrating,
            signal_strength=round(strength, 4),
            spread_regime=regime,
        ))

    results.sort(key=lambda b: b.signal_strength, reverse=True)
    return results


# ═════════════════════════════════════════════════════════════════════════════
# Enhanced Pair Scanner (combines all methods)
# ═════════════════════════════════════════════════════════════════════════════

def scan_pairs_enhanced(
    prices: pd.DataFrame,
    sectors: List[str],
    vix: Optional[pd.Series] = None,
    window: int = 60,
    z_lookback: int = 60,
) -> List[PairSignal]:
    """
    Enhanced pair scanner using Engle-Granger + Kalman + dynamic thresholds.

    Upgrades over basic scan_pairs:
      1. Kalman-filtered hedge ratios (real-time adaptive beta)
      2. Dynamic z-score thresholds (regime-adjusted)
      3. Spread regime classification (MR vs trending vs RW)
      4. Hurst exponent for spread persistence
      5. Expected profit estimation
    """
    basic_signals = scan_pairs(prices, sectors, window, z_lookback)

    log_prices = np.log(prices[[s for s in sectors if s in prices.columns]].dropna())

    for sig in basic_signals:
        try:
            # 1. Kalman filter
            kalman = kalman_pair_signal(prices, sig.ticker_a, sig.ticker_b,
                                        z_lookback=z_lookback)
            if kalman:
                sig.kalman_hedge_ratio = kalman["kalman_beta"]
                sig.kalman_spread_z = kalman["kalman_spread_z"]

            # 2. Dynamic thresholds
            if sig.ticker_a in log_prices.columns and sig.ticker_b in log_prices.columns:
                spread = log_prices[sig.ticker_a] - sig.hedge_ratio * log_prices[sig.ticker_b]
                entry, exit_t, stop = dynamic_z_thresholds(spread, vix)
                sig.entry_threshold = entry
                sig.exit_threshold = exit_t
                sig.stop_threshold = stop

                # 3. Spread regime
                regime, vr = classify_spread_regime(spread)
                sig.spread_variance_ratio = vr
                sig.spread_regime = regime

                # 4. Hurst exponent
                try:
                    from analytics.signal_mean_reversion import _hurst_exponent
                    sig.hurst = round(_hurst_exponent(spread), 4)
                except Exception:
                    pass

                # 5. Expected profit
                spread_vol = float(spread.iloc[-60:].std()) if len(spread) >= 60 else 0
                mr_prob = max(0, 1.0 - vr) if vr < 1.0 else 0
                sig.expected_profit_bps = round(abs(sig.spread_z) * spread_vol * mr_prob * 10000, 1)

        except Exception as e:
            log.debug("Enhanced analysis failed for %s: %s", sig.pair_name, e)

    # Re-sort by enhanced signal strength
    for sig in basic_signals:
        # Boost signal strength if Kalman confirms and spread is MR
        if sig.spread_regime == "MEAN_REVERTING":
            sig.signal_strength = min(1.0, sig.signal_strength * 1.3)
        if abs(sig.kalman_spread_z) > abs(sig.spread_z):
            sig.signal_strength = min(1.0, sig.signal_strength * 1.1)

    basic_signals.sort(key=lambda s: s.signal_strength, reverse=True)
    return basic_signals
