"""
analytics/pair_scanner.py
===========================
Intra-Sector Pair Scanner + Position Correlation Tracker

Scans all N*(N-1)/2 sector ETF pairs for cointegrated spreads,
ranks by signal strength, and tracks correlation between open positions.

Pairs: XLK-XLRE, XLF-XLU, XLY-XLP, etc. (55 pairs from 11 sectors)

For each pair:
  1. Cointegration test (Engle-Granger)
  2. Spread z-score
  3. Half-life estimation
  4. Hedge ratio (OLS beta)

Position correlation tracker:
  - Computes correlation matrix of open positions
  - Flags concentrated risk (>0.7 pairwise corr between positions)
  - Adjusts sizing for correlated positions

Ref: Engle & Granger (1987) — Cointegration and Error Correction
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
    """Signal for one sector pair."""
    pair_name: str          # e.g., "XLK-XLU"
    ticker_a: str
    ticker_b: str
    spread_z: float         # Z-score of spread
    half_life: float        # Days
    hedge_ratio: float      # h: spread = ln(A) - h·ln(B)
    direction: str          # "LONG_SPREAD" / "SHORT_SPREAD" / "NEUTRAL"
    adf_pvalue: float       # Cointegration test p-value
    correlation: float      # Current 60d correlation
    signal_strength: float  # Combined score [0, 1]


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
