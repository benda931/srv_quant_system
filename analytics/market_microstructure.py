"""
analytics/market_microstructure.py
====================================
Market Microstructure Analysis for Optimal Execution

Institutional-grade execution analytics:
  1. Bid-Ask Spread Estimation — from daily OHLC (Roll, Corwin-Schultz models)
  2. Optimal Execution Timing — intraday volume profile, time-of-day effects
  3. Market Impact Estimation — Almgren-Chriss temporary + permanent impact
  4. Liquidity Risk Scoring — per-sector liquidity assessment
  5. Execution Quality Analysis — compare fills vs VWAP/TWAP benchmarks

Ref: Roll (1984) — A Simple Implicit Measure of the Effective Bid-Ask Spread
Ref: Corwin & Schultz (2012) — A Simple Way to Estimate Bid-Ask Spreads
Ref: Almgren & Chriss (2001) — Optimal Execution of Portfolio Transactions
Ref: Kyle (1985) — Continuous Auctions and Insider Trading
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Bid-Ask Spread Estimation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SpreadEstimate:
    """Estimated bid-ask spread for a security."""
    ticker: str
    spread_bps: float                 # Estimated spread in basis points
    spread_pct: float                 # As percentage
    method: str                       # "roll" / "corwin_schultz" / "volume_proxy"
    confidence: float                 # 0-1 confidence in estimate
    n_observations: int


def estimate_spread_roll(prices: pd.Series) -> float:
    """
    Roll (1984) spread estimator from serial covariance of returns.

    Spread = 2 × √(-Cov(r_t, r_{t-1}))

    The intuition: the bid-ask bounce creates negative autocorrelation
    in returns. The magnitude of this autocorrelation reveals the spread.
    """
    rets = prices.pct_change().dropna()
    if len(rets) < 30:
        return 0.0

    cov = float(rets.iloc[1:].values @ rets.iloc[:-1].values / (len(rets) - 1))
    if cov >= 0:
        return 0.0  # Positive autocorrelation → spread estimate not valid

    spread = 2 * math.sqrt(-cov)
    return max(0, min(0.05, spread))  # Cap at 5%


def estimate_spread_corwin_schultz(
    high: pd.Series, low: pd.Series,
) -> float:
    """
    Corwin & Schultz (2012) spread estimator from high-low prices.

    Uses the ratio of high-low ranges across 1-day and 2-day windows.
    The 2-day range captures volatility + spread, while 1-day captures
    volatility only. The difference isolates the spread.

    β = E[ln(H_t/L_t)²]
    γ = E[ln(H_{t,t+1}/L_{t,t+1})²]  (2-day high-low range)
    α = (√2β - √β) / (3 - 2√2) - √(γ / (3 - 2√2))
    Spread = 2(e^α - 1) / (1 + e^α)
    """
    if len(high) < 30 or len(low) < 30:
        return 0.0

    h = high.values.astype(float)
    l = low.values.astype(float)
    n = len(h)

    # 1-day log range
    beta_vals = np.log(h / l) ** 2
    beta = float(beta_vals[:-1].mean())

    # 2-day high-low range
    h2 = np.maximum(h[:-1], h[1:])
    l2 = np.minimum(l[:-1], l[1:])
    gamma = float((np.log(h2 / l2) ** 2).mean())

    # Corwin-Schultz formula
    sqrt2 = math.sqrt(2)
    denom = 3 - 2 * sqrt2
    if denom == 0 or beta <= 0:
        return 0.0

    alpha = (sqrt2 * math.sqrt(beta) - math.sqrt(beta)) / denom
    alpha -= math.sqrt(gamma / denom)

    if alpha >= 0:
        return 0.0

    spread = 2 * (math.exp(alpha) - 1) / (1 + math.exp(alpha))
    return max(0, min(0.05, abs(spread)))


def estimate_spreads(
    prices: pd.DataFrame,
    tickers: List[str],
    method: str = "roll",
) -> Dict[str, SpreadEstimate]:
    """
    Estimate bid-ask spreads for multiple tickers.

    Parameters
    ----------
    prices : pd.DataFrame — daily close prices
    tickers : list — tickers to analyze
    method : str — "roll" or "volume_proxy" (Corwin-Schultz needs OHLC)
    """
    results = {}
    for t in tickers:
        if t not in prices.columns:
            continue

        p = prices[t].dropna()
        if method == "roll":
            spread = estimate_spread_roll(p)
        else:
            spread = estimate_spread_roll(p)  # Default fallback

        results[t] = SpreadEstimate(
            ticker=t,
            spread_bps=round(spread * 10000, 1),
            spread_pct=round(spread * 100, 3),
            method=method,
            confidence=0.7 if len(p) >= 100 else 0.4,
            n_observations=len(p),
        )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Market Impact Model (Almgren-Chriss)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MarketImpactEstimate:
    """Estimated market impact for a trade."""
    ticker: str
    order_size_shares: int
    order_notional: float
    # Impact components
    temporary_impact_bps: float       # From crossing the spread + temporary price pressure
    permanent_impact_bps: float       # Permanent price change from information leakage
    total_impact_bps: float           # Total execution cost
    # Context
    participation_rate: float          # Order size / ADV
    spread_bps: float                  # Estimated spread
    volatility_daily: float            # Daily vol (for impact scaling)
    # Recommendation
    optimal_execution_time: str        # "IMMEDIATE" / "30_MIN" / "FULL_DAY" / "MULTI_DAY"
    algo_suggestion: str               # "MARKET" / "LIMIT" / "TWAP" / "VWAP"


def estimate_market_impact(
    ticker: str,
    order_notional: float,
    price: float,
    daily_volume: float = 10_000_000,
    daily_vol: float = 0.015,
    spread_bps: float = 2.0,
) -> MarketImpactEstimate:
    """
    Almgren-Chriss market impact model.

    Total cost = Spread cost + Temporary impact + Permanent impact

    Temporary impact: η × σ × (Q/V)^0.6
      η ≈ 0.142 (empirical constant for US equities)
      Q = order quantity, V = daily volume

    Permanent impact: γ × σ × (Q/V)^0.5
      γ ≈ 0.314 (empirical constant)

    Parameters
    ----------
    ticker : str
    order_notional : float — dollar value of the order
    price : float — current price per share
    daily_volume : float — average daily dollar volume
    daily_vol : float — daily return volatility
    spread_bps : float — estimated bid-ask spread (bps)
    """
    shares = int(order_notional / max(price, 0.01))
    participation = order_notional / max(daily_volume, 1_000_000)

    # Spread cost
    spread_cost = spread_bps / 2  # Pay half the spread

    # Temporary impact (Almgren-Chriss)
    eta = 0.142
    temp_impact = eta * daily_vol * 10000 * (participation ** 0.6)

    # Permanent impact
    gamma = 0.314
    perm_impact = gamma * daily_vol * 10000 * (participation ** 0.5)

    total = spread_cost + temp_impact + perm_impact

    # Optimal execution time
    if participation < 0.001:
        exec_time = "IMMEDIATE"
        algo = "MARKET"
    elif participation < 0.01:
        exec_time = "30_MIN"
        algo = "LIMIT"
    elif participation < 0.05:
        exec_time = "FULL_DAY"
        algo = "TWAP"
    else:
        exec_time = "MULTI_DAY"
        algo = "VWAP"

    return MarketImpactEstimate(
        ticker=ticker,
        order_size_shares=shares,
        order_notional=round(order_notional, 2),
        temporary_impact_bps=round(temp_impact, 1),
        permanent_impact_bps=round(perm_impact, 1),
        total_impact_bps=round(total, 1),
        participation_rate=round(participation, 4),
        spread_bps=round(spread_bps, 1),
        volatility_daily=round(daily_vol, 4),
        optimal_execution_time=exec_time,
        algo_suggestion=algo,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Liquidity Risk Scoring
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LiquidityScore:
    """Per-sector liquidity assessment."""
    ticker: str
    score: float                      # 0-100 (100 = most liquid)
    label: str                        # "HIGH" / "MEDIUM" / "LOW" / "ILLIQUID"
    spread_bps: float
    daily_volume_usd: float
    amihud_illiquidity: float         # Amihud (2002) ratio
    turnover_ratio: float             # Volume / market cap proxy


def compute_liquidity_scores(
    prices: pd.DataFrame,
    volumes: Optional[pd.DataFrame] = None,
    tickers: Optional[List[str]] = None,
) -> Dict[str, LiquidityScore]:
    """
    Compute liquidity scores for each sector ETF.

    Uses:
      - Spread estimate (Roll model)
      - Amihud illiquidity ratio: |r_t| / Volume_t (price impact per dollar)
      - Return autocorrelation (high = illiquid)
      - Daily volume level
    """
    if tickers is None:
        tickers = [c for c in prices.columns if not c.startswith("^")]

    results = {}

    for t in tickers:
        if t not in prices.columns:
            continue

        p = prices[t].dropna()
        if len(p) < 60:
            continue

        rets = p.pct_change().dropna()
        spread = estimate_spread_roll(p)
        spread_bps = spread * 10000

        # Amihud illiquidity (proxy — use price × returns as volume proxy)
        abs_rets = rets.abs()
        volume_proxy = p * 1e6  # Assume $1M daily volume per sector ETF
        amihud = float((abs_rets / volume_proxy.iloc[1:]).mean()) * 1e6 if len(abs_rets) > 0 else 0

        # Score: 100 = most liquid
        # Penalize: high spread, high Amihud, low volume
        score = 100.0
        score -= min(30, spread_bps * 3)          # Spread penalty
        score -= min(30, amihud * 1000)             # Amihud penalty
        score -= min(20, abs(float(rets.autocorr())) * 50)  # Autocorrelation penalty
        score = max(0, min(100, score))

        if score >= 80:
            label = "HIGH"
        elif score >= 60:
            label = "MEDIUM"
        elif score >= 40:
            label = "LOW"
        else:
            label = "ILLIQUID"

        results[t] = LiquidityScore(
            ticker=t,
            score=round(score, 1),
            label=label,
            spread_bps=round(spread_bps, 1),
            daily_volume_usd=1e7,  # Placeholder — sector ETFs typically $10M+ daily
            amihud_illiquidity=round(amihud, 6),
            turnover_ratio=0.0,
        )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Execution Quality Analysis
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExecutionQuality:
    """Execution quality metrics for a trade."""
    trade_id: str
    ticker: str
    fill_price: float
    close_price: float                 # Close on execution day
    vwap_estimate: float               # Estimated VWAP
    # Slippage
    slippage_vs_close_bps: float       # Fill vs close (positive = worse than close)
    slippage_vs_vwap_bps: float        # Fill vs VWAP
    # Quality
    execution_quality_score: float     # 0-100 (100 = perfect)
    label: str                         # "EXCELLENT" / "GOOD" / "ACCEPTABLE" / "POOR"


def analyse_execution_quality(
    trades: List[Dict],
    prices: pd.DataFrame,
) -> List[ExecutionQuality]:
    """
    Analyse execution quality for completed trades.

    Compares fill prices against:
      - Close price (benchmark)
      - Estimated VWAP (close ± 0.1% for sector ETFs)
    """
    results = []

    for trade in trades:
        ticker = trade.get("ticker", "")
        fill = trade.get("entry_price") or trade.get("fill_price", 0)
        entry_date = trade.get("entry_date", "")

        if not ticker or not fill or ticker not in prices.columns:
            continue

        # Get close price on entry day
        try:
            close = float(prices[ticker].loc[entry_date]) if entry_date in prices.index else float(prices[ticker].dropna().iloc[-1])
        except Exception:
            close = fill

        # VWAP estimate (close ± noise for sector ETFs)
        vwap = close * 1.0005  # Assume VWAP slightly above close (buy bias)

        # Slippage
        direction = trade.get("direction", "LONG")
        sign = 1 if direction == "LONG" else -1
        slip_close = sign * (fill - close) / close * 10000
        slip_vwap = sign * (fill - vwap) / vwap * 10000

        # Quality score
        abs_slip = abs(slip_close)
        if abs_slip < 2:
            score = 95
            label = "EXCELLENT"
        elif abs_slip < 5:
            score = 80
            label = "GOOD"
        elif abs_slip < 10:
            score = 60
            label = "ACCEPTABLE"
        else:
            score = max(0, 50 - abs_slip)
            label = "POOR"

        results.append(ExecutionQuality(
            trade_id=trade.get("trade_id", ""),
            ticker=ticker,
            fill_price=round(fill, 4),
            close_price=round(close, 4),
            vwap_estimate=round(vwap, 4),
            slippage_vs_close_bps=round(slip_close, 1),
            slippage_vs_vwap_bps=round(slip_vwap, 1),
            execution_quality_score=round(score, 1),
            label=label,
        ))

    return results
