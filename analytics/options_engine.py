"""
analytics/options_engine.py
=============================
Options Analytics Engine — IV surfaces, Greeks, Implied Correlation, Dispersion Index

Derives a complete options analytics layer from:
  1. VIX index (= SPY ATM implied vol, annualized)
  2. Sector realized vol (EWMA)
  3. Sector betas to SPY
  4. Black-Scholes model for Greeks

Provides:
  - Per-sector implied volatility (derived from VIX + beta scaling)
  - Full Greeks profile (delta, gamma, vega, theta) for ATM straddles
  - Cboe-style Implied Correlation Index: ρ_impl = (σ²_I - Σw²σ²) / (2·Σw_i·w_j·σ_i·σ_j)
  - Cboe-style Dispersion Index: DSPX = √(Σw²σ² - σ²_I) (when dispersion > index var)
  - Variance Risk Premium: VRP = IV² - RV²
  - Term structure slope proxy

Ref: Cboe VIX methodology (variance replication)
Ref: Cboe Implied Correlation Index (COR3M)
Ref: Cboe S&P 500 Dispersion Index (DSPX)
Ref: Black-Scholes-Merton (1973)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Black-Scholes primitives
# ─────────────────────────────────────────────────────────────────────────────

def bs_d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes d1."""
    if T <= 0 or sigma <= 0:
        return 0.0
    return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))


def bs_d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    return bs_d1(S, K, T, r, sigma) - sigma * math.sqrt(T)


def bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes call price."""
    if T <= 0:
        return max(0, S - K)
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def bs_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes put price."""
    if T <= 0:
        return max(0, K - S)
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_straddle(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """ATM straddle price = call + put."""
    return bs_call(S, K, T, r, sigma) + bs_put(S, K, T, r, sigma)


# ─────────────────────────────────────────────────────────────────────────────
# Greeks
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GreeksSnapshot:
    """Greeks for a single position."""
    ticker: str
    price: float
    iv: float                    # Implied volatility (annualized)
    rv_20d: float                # 20-day realized vol
    rv_60d: float                # 60-day realized vol

    # Greeks (per $1 notional of ATM straddle)
    delta: float                 # Net delta (straddle ≈ 0)
    gamma: float                 # Gamma per $1 move
    vega: float                  # Vega per 1% IV move
    theta: float                 # Theta per day (negative = pay)

    # Variance risk premium
    vrp: float                   # IV² - RV² (positive = premium to collect)
    vrp_z: float                 # Z-score of VRP vs history

    # IV rank / percentile
    iv_rank_252d: float          # IV percentile over 252 days
    iv_percentile_60d: float     # IV percentile over 60 days


def compute_greeks(
    S: float, sigma: float, T: float = 30/365, r: float = 0.05,
) -> dict:
    """Compute Greeks for ATM straddle (K=S)."""
    K = S
    if T <= 0 or sigma <= 0 or S <= 0:
        return {"delta": 0, "gamma": 0, "vega": 0, "theta": 0}

    d1 = bs_d1(S, K, T, r, sigma)
    sqrt_T = math.sqrt(T)

    # Call delta + put delta = N(d1) + (N(d1) - 1) = 2N(d1) - 1
    delta = 2 * norm.cdf(d1) - 1

    # Gamma (same for call and put) × 2 for straddle
    gamma = 2 * norm.pdf(d1) / (S * sigma * sqrt_T)

    # Vega (per 1% move in sigma) × 2 for straddle
    vega = 2 * S * norm.pdf(d1) * sqrt_T / 100  # Per 1% IV move

    # Theta × 2 for straddle (per calendar day)
    theta_call = -(S * norm.pdf(d1) * sigma) / (2 * sqrt_T) - r * K * math.exp(-r * T) * norm.cdf(d1 - sigma * sqrt_T)
    theta_put = -(S * norm.pdf(d1) * sigma) / (2 * sqrt_T) + r * K * math.exp(-r * T) * norm.cdf(-(d1 - sigma * sqrt_T))
    theta = (theta_call + theta_put) / 365  # Per calendar day

    return {"delta": round(delta, 6), "gamma": round(gamma, 6),
            "vega": round(vega, 6), "theta": round(theta, 6)}


# ─────────────────────────────────────────────────────────────────────────────
# Implied Volatility Surface
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IVSurface:
    """Implied volatility surface for all sectors + index."""
    as_of_date: str

    # Per-ticker IV
    sector_iv: Dict[str, float]          # {ticker: ATM IV annualized}
    index_iv: float                       # SPY IV from VIX

    # Per-ticker Greeks
    sector_greeks: Dict[str, GreeksSnapshot]

    # Implied Correlation (Cboe-style)
    implied_corr: float                   # ρ_impl
    implied_corr_history: Optional[pd.Series] = None

    # Dispersion Index (Cboe DSPX-style)
    dispersion_index: float = 0.0         # √(Σw²σ² - σ²_I) if positive
    dispersion_z: float = 0.0             # Z-score vs history

    # Variance Risk Premium (aggregate)
    vrp_index: float = 0.0                # VIX² - RV²(SPY)
    vrp_sectors_avg: float = 0.0          # Average sector VRP

    # IV term structure proxy
    vix_current: float = 0.0
    vix_20d_avg: float = 0.0
    term_slope: float = 0.0               # Positive = contango, negative = backwardation

    # Vol-of-Vol (VVIX) — timing signal for short vol entry
    vvix_current: float = 0.0             # ^VVIX level (or proxy: 20d rolling std of VIX changes)
    vvix_percentile: float = 0.0          # Percentile vs 252d history (>80% = vol-of-vol elevated)
    vvix_signal: str = ""                 # "SHORT_VOL_FAVORABLE" / "NEUTRAL" / "AVOID"

    # CBOE Skew Index
    skew_current: float = 100.0           # ^SKEW level (100=normal, >130=high tail risk)
    skew_percentile: float = 0.0          # Percentile vs history
    skew_signal: str = ""                 # "TAIL_RISK_LOW" / "NORMAL" / "TAIL_RISK_HIGH"

    # Combined short-vol timing score (0-100)
    short_vol_timing_score: float = 50.0  # Higher = better time to sell vol
    short_vol_timing_label: str = ""      # "SELL VOL" / "NEUTRAL" / "BUY VOL"


class OptionsEngine:
    """
    Derives full options analytics from VIX + sector prices.

    Since we don't have actual options chain data, we derive:
      - Sector IV = VIX × (sector_beta × sector_rv / spy_rv)
      - Greeks via Black-Scholes on derived IV
      - Implied Correlation from index/sector variance decomposition
      - Dispersion Index from variance mismatch

    Usage:
        engine = OptionsEngine()
        surface = engine.compute_surface(prices, settings)
    """

    def compute_surface(
        self,
        prices: pd.DataFrame,
        settings=None,
        risk_free_rate: float = 0.05,
        dte: int = 30,
    ) -> IVSurface:
        """
        Compute full IV surface from prices.

        Parameters
        ----------
        prices    : Daily close prices with sectors + SPY + ^VIX
        settings  : Settings for sector list and weights
        dte       : Days to expiration for Greeks
        """
        from config.settings import get_settings
        if settings is None:
            settings = get_settings()

        sectors = settings.sector_list()
        spy = settings.spy_ticker
        vix_col = "^VIX" if "^VIX" in prices.columns else "VIX"

        avail = [s for s in sectors if s in prices.columns]
        log_rets = np.log(prices / prices.shift(1)).dropna(how="all")

        # Current values
        last_prices = {s: float(prices[s].dropna().iloc[-1]) for s in avail if s in prices.columns}
        last_prices[spy] = float(prices[spy].dropna().iloc[-1]) if spy in prices.columns else 500.0
        vix_current = float(prices[vix_col].dropna().iloc[-1]) / 100.0 if vix_col in prices.columns else 0.20
        vix_20d = float(prices[vix_col].dropna().tail(20).mean()) / 100.0 if vix_col in prices.columns else vix_current

        T = dte / 365.0

        # ── Realized Volatilities ──
        rv_20d = {}
        rv_60d = {}
        for s in avail + [spy]:
            if s in log_rets.columns:
                rv_20d[s] = float(log_rets[s].tail(20).std() * np.sqrt(252))
                rv_60d[s] = float(log_rets[s].tail(60).std() * np.sqrt(252))

        spy_rv = rv_60d.get(spy, 0.15)

        # ── Sector Betas (60d rolling) ──
        betas = {}
        for s in avail:
            if s in log_rets.columns and spy in log_rets.columns:
                r_s = log_rets[s].tail(60)
                r_spy = log_rets[spy].tail(60)
                cov = r_s.cov(r_spy)
                var_spy = r_spy.var()
                betas[s] = cov / var_spy if var_spy > 1e-12 else 1.0

        # ── Derived Implied Volatility per sector ──
        # IV_sector = VIX × (beta × sector_rv / spy_rv)
        # This approximation captures: higher-beta sectors have higher IV
        sector_iv = {}
        for s in avail:
            beta = betas.get(s, 1.0)
            rv_ratio = rv_60d.get(s, spy_rv) / max(spy_rv, 0.01)
            sector_iv[s] = vix_current * abs(beta) * rv_ratio
            # Cap at reasonable range
            sector_iv[s] = max(0.05, min(1.50, sector_iv[s]))

        # ── Greeks per sector (ATM straddle) ──
        sector_greeks = {}
        for s in avail:
            iv = sector_iv[s]
            S = last_prices.get(s, 100)
            g = compute_greeks(S, iv, T, risk_free_rate)

            # VRP = IV² - RV²
            rv = rv_20d.get(s, iv)
            vrp = iv ** 2 - rv ** 2

            # IV rank (simplified: percentile of current IV in last 252 days)
            if vix_col in prices.columns:
                vix_hist = prices[vix_col].dropna().tail(252) / 100.0
                beta_s = betas.get(s, 1.0)
                iv_hist = vix_hist * abs(beta_s) * (rv_60d.get(s, spy_rv) / max(spy_rv, 0.01))
                iv_rank = float((iv_hist <= iv).mean())
                iv_pct_60 = float((prices[vix_col].dropna().tail(60) / 100.0 * abs(beta_s) <= iv).mean())
            else:
                iv_rank = 0.5
                iv_pct_60 = 0.5

            sector_greeks[s] = GreeksSnapshot(
                ticker=s, price=S, iv=round(iv, 4),
                rv_20d=round(rv_20d.get(s, 0), 4),
                rv_60d=round(rv_60d.get(s, 0), 4),
                delta=g["delta"], gamma=g["gamma"], vega=g["vega"], theta=g["theta"],
                vrp=round(vrp, 6), vrp_z=0.0,
                iv_rank_252d=round(iv_rank, 4),
                iv_percentile_60d=round(iv_pct_60, 4),
            )

        # ── Implied Correlation (Cboe-style) ──
        # σ²_I ≈ Σw²σ² + 2·Σw_i·w_j·σ_i·σ_j·ρ_ij
        # ρ_impl = (σ²_I - Σw²σ²) / (2·Σw_i·w_j·σ_i·σ_j)
        n_sec = len(avail)
        w = 1.0 / n_sec  # Equal weight approximation
        index_var = vix_current ** 2

        sum_w2_sigma2 = sum((w * sector_iv.get(s, 0.20)) ** 2 for s in avail)
        cross_term = 0.0
        for i, s1 in enumerate(avail):
            for j, s2 in enumerate(avail):
                if i < j:
                    cross_term += w * w * sector_iv.get(s1, 0.20) * sector_iv.get(s2, 0.20)
        cross_term *= 2

        if cross_term > 1e-10:
            implied_corr = (index_var - sum_w2_sigma2) / cross_term
            implied_corr = max(-1.0, min(1.0, implied_corr))
        else:
            implied_corr = 0.5

        # ── Dispersion Index (DSPX-style) ──
        # DSPX = √(Σw²σ²_i - σ²_I) when positive (dispersion exceeds index var)
        weighted_sector_var = sum((w * sector_iv.get(s, 0.20)) ** 2 for s in avail) * n_sec
        disp_raw = weighted_sector_var - index_var
        dispersion_index = math.sqrt(max(0, disp_raw)) * 100  # Convert to percentage points

        # ── Aggregate VRP ──
        vrp_index = vix_current ** 2 - rv_60d.get(spy, 0.15) ** 2
        vrp_sectors = [sector_greeks[s].vrp for s in avail if s in sector_greeks]
        vrp_avg = float(np.mean(vrp_sectors)) if vrp_sectors else 0.0

        # ── Term structure slope ──
        term_slope = vix_20d - vix_current  # Positive = contango (normal)

        # ── VVIX (vol-of-vol) ──
        vvix_current = 0.0
        vvix_percentile = 0.0
        vvix_signal = "NEUTRAL"

        vvix_col = next((c for c in prices.columns if "VVIX" in c.upper()), None)
        if vvix_col and vvix_col in prices.columns:
            vvix_series = prices[vvix_col].dropna()
            if len(vvix_series) >= 20:
                vvix_current = float(vvix_series.iloc[-1])
                if len(vvix_series) >= 252:
                    vvix_percentile = float((vvix_series.iloc[-252:] <= vvix_current).mean())
                else:
                    vvix_percentile = float((vvix_series <= vvix_current).mean())
        else:
            # Proxy VVIX: 20d rolling std of VIX daily changes × √252
            if vix_col and len(prices[vix_col].dropna()) >= 30:
                _vix_changes = prices[vix_col].diff().dropna()
                vvix_current = float(_vix_changes.iloc[-20:].std() * np.sqrt(252))
                if len(_vix_changes) >= 252:
                    _rolling = _vix_changes.rolling(20).std() * np.sqrt(252)
                    vvix_percentile = float((_rolling.dropna().iloc[-252:] <= vvix_current).mean())

        if vvix_percentile > 0.80:
            vvix_signal = "AVOID"  # VVIX too high — don't sell vol now
        elif vvix_percentile < 0.30:
            vvix_signal = "SHORT_VOL_FAVORABLE"  # Low VVIX = cheap to sell vol
        else:
            vvix_signal = "NEUTRAL"

        # ── CBOE Skew Index ──
        skew_current = 100.0
        skew_percentile = 0.0
        skew_signal = "NORMAL"

        skew_col = next((c for c in prices.columns if c.upper() in ("^SKEW", "SKEW")), None)
        if skew_col and skew_col in prices.columns:
            skew_series = prices[skew_col].dropna()
            if len(skew_series) >= 20:
                skew_current = float(skew_series.iloc[-1])
                if len(skew_series) >= 252:
                    skew_percentile = float((skew_series.iloc[-252:] <= skew_current).mean())
                else:
                    skew_percentile = float((skew_series <= skew_current).mean())

        if skew_current > 130 or skew_percentile > 0.90:
            skew_signal = "TAIL_RISK_HIGH"
        elif skew_current < 115 and skew_percentile < 0.30:
            skew_signal = "TAIL_RISK_LOW"
        else:
            skew_signal = "NORMAL"

        # ── Short-Vol Timing Score (0-100) ──
        # Combines: VRP (30%), VVIX (25%), term structure (20%), implied corr (15%), skew (10%)
        _score_vrp = min(100, max(0, vrp_index * 10000))        # VRP > 0 = favorable
        _score_vvix = max(0, 100 - vvix_percentile * 100)        # Low VVIX = favorable
        _score_term = min(100, max(0, term_slope * 100 * 20 + 50))  # Contango = favorable
        _score_corr = min(100, max(0, (1 - implied_corr) * 100))    # Low impl corr = favorable for dispersion
        _score_skew = max(0, 100 - (skew_current - 100) * 2)        # Low skew = favorable

        short_vol_timing = (
            0.30 * _score_vrp + 0.25 * _score_vvix + 0.20 * _score_term
            + 0.15 * _score_corr + 0.10 * _score_skew
        )
        short_vol_timing = max(0, min(100, short_vol_timing))

        if short_vol_timing >= 65:
            sv_label = "SELL VOL"
        elif short_vol_timing <= 35:
            sv_label = "BUY VOL"
        else:
            sv_label = "NEUTRAL"

        as_of = str(prices.index[-1].date()) if hasattr(prices.index[-1], "date") else str(prices.index[-1])

        return IVSurface(
            as_of_date=as_of,
            sector_iv=sector_iv,
            index_iv=round(vix_current, 4),
            sector_greeks=sector_greeks,
            implied_corr=round(implied_corr, 4),
            dispersion_index=round(dispersion_index, 4),
            vrp_index=round(vrp_index, 6),
            vrp_sectors_avg=round(vrp_avg, 6),
            vix_current=round(vix_current * 100, 2),
            vix_20d_avg=round(vix_20d * 100, 2),
            term_slope=round(term_slope * 100, 4),
            vvix_current=round(vvix_current, 2),
            vvix_percentile=round(vvix_percentile, 4),
            vvix_signal=vvix_signal,
            skew_current=round(skew_current, 2),
            skew_percentile=round(skew_percentile, 4),
            skew_signal=skew_signal,
            short_vol_timing_score=round(short_vol_timing, 1),
            short_vol_timing_label=sv_label,
        )

    def compute_iv_history(
        self,
        prices: pd.DataFrame,
        settings=None,
        window: int = 252,
    ) -> pd.DataFrame:
        """Compute rolling implied correlation + dispersion history."""
        from config.settings import get_settings
        if settings is None:
            settings = get_settings()

        sectors = settings.sector_list()
        spy = settings.spy_ticker
        vix_col = "^VIX" if "^VIX" in prices.columns else "VIX"
        avail = [s for s in sectors if s in prices.columns]

        log_rets = np.log(prices / prices.shift(1)).dropna(how="all")
        n = len(log_rets)
        n_sec = len(avail)
        w = 1.0 / n_sec

        start = max(60, window)
        dates = log_rets.index[start:]

        impl_corr = pd.Series(index=dates, dtype=float)
        disp_idx = pd.Series(index=dates, dtype=float)
        vrp = pd.Series(index=dates, dtype=float)

        spy_rv_col = log_rets[spy] if spy in log_rets.columns else None

        for i, dt in enumerate(dates):
            idx = log_rets.index.get_loc(dt)
            vix_val = float(prices[vix_col].iloc[idx]) / 100.0 if vix_col in prices.columns else 0.20

            # Sector RVs
            sector_rvs = {}
            for s in avail:
                if s in log_rets.columns:
                    rv = float(log_rets[s].iloc[idx-60:idx].std() * np.sqrt(252))
                    sector_rvs[s] = rv

            spy_rv = float(log_rets[spy].iloc[idx-60:idx].std() * np.sqrt(252)) if spy in log_rets.columns else 0.15

            # Sector betas
            sector_ivs = {}
            for s in avail:
                if s in log_rets.columns and spy in log_rets.columns:
                    r_s = log_rets[s].iloc[idx-60:idx]
                    r_spy = log_rets[spy].iloc[idx-60:idx]
                    cov = r_s.cov(r_spy)
                    var_spy = r_spy.var()
                    beta = cov / var_spy if var_spy > 1e-12 else 1.0
                    rv_ratio = sector_rvs.get(s, spy_rv) / max(spy_rv, 0.01)
                    sector_ivs[s] = max(0.05, min(1.5, vix_val * abs(beta) * rv_ratio))

            # Implied correlation
            index_var = vix_val ** 2
            sum_w2_s2 = sum((w * sector_ivs.get(s, 0.20)) ** 2 for s in avail)
            cross = 0.0
            for a in range(n_sec):
                for b in range(a+1, n_sec):
                    cross += w * w * sector_ivs.get(avail[a], 0.20) * sector_ivs.get(avail[b], 0.20)
            cross *= 2

            if cross > 1e-10:
                ic = (index_var - sum_w2_s2) / cross
                impl_corr.iloc[i] = max(-1, min(1, ic))

            # Dispersion
            ws_var = sum((w * sector_ivs.get(s, 0.20)) ** 2 for s in avail) * n_sec
            d = ws_var - index_var
            disp_idx.iloc[i] = math.sqrt(max(0, d)) * 100

            # VRP
            vrp.iloc[i] = (vix_val ** 2 - spy_rv ** 2) * 100

        return pd.DataFrame({
            "implied_corr": impl_corr,
            "dispersion_index": disp_idx,
            "vrp_pct": vrp,
        })


# ─────────────────────────────────────────────────────────────────────────────
# Summary for agents / API
# ─────────────────────────────────────────────────────────────────────────────

def options_summary(surface: IVSurface) -> dict:
    """Compact summary for Claude / AgentBus."""
    return {
        "as_of": surface.as_of_date,
        "vix": surface.vix_current,
        "vix_20d_avg": surface.vix_20d_avg,
        "term_slope": surface.term_slope,
        "implied_corr": surface.implied_corr,
        "dispersion_index": surface.dispersion_index,
        "vrp_index": surface.vrp_index,
        "vrp_sectors_avg": surface.vrp_sectors_avg,
        "n_sectors": len(surface.sector_iv),
        "sector_iv": {k: round(v, 4) for k, v in sorted(surface.sector_iv.items(), key=lambda x: -x[1])[:5]},
        "sector_greeks_summary": {
            s: {"iv": g.iv, "vrp": g.vrp, "iv_rank": g.iv_rank_252d, "theta": g.theta}
            for s, g in list(surface.sector_greeks.items())[:5]
        },
        # VVIX + Skew + timing
        "vvix": surface.vvix_current,
        "vvix_pct": surface.vvix_percentile,
        "vvix_signal": surface.vvix_signal,
        "skew": surface.skew_current,
        "skew_pct": surface.skew_percentile,
        "skew_signal": surface.skew_signal,
        "short_vol_timing": surface.short_vol_timing_score,
        "short_vol_timing_label": surface.short_vol_timing_label,
    }
