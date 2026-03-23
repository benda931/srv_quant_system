"""
analytics/fundamentals_engine.py

Institutional-grade Fundamental Justification Score (FJS) engine.
Implements the full framework from the SRV research document:

  Section 1  — Sector-specific primary multiple mapping
  Section 2  — Composite FJS with justified-multiple residual model
  Section 3  — Rate-adjusted valuation (XLU, XLRE, XLK, XLF)
  Section 4  — Earnings quality + revision momentum
  Section 5  — Holdings aggregation (ratio-of-sums for EV/EBITDA)
  Section 6  — Coverage-adjusted output

Key design decisions:
  - Each sector uses its PRIMARY multiple (P/B for XLF, EV/EBITDA for XLE)
  - Cross-sectional comparison via z-score of residuals (Δ = obs - justified)
  - Sigmoid mapping to [0,1] FJS: low FJS = clean mispricing candidate
  - Fully defensive: any missing data → graceful fallback + coverage flag
  - Rate sensitivity gate: CRISIS regime reduces FJS weight on rate-sensitive sectors
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Sector configuration
# ──────────────────────────────────────────────────────────────

# Primary multiple used per sector (determines aggregation method)
SECTOR_PRIMARY_MULTIPLE: Dict[str, str] = {
    "XLC":  "ev_ebitda",       # Communication — EV/EBITDA
    "XLY":  "forward_pe",      # Discretionary — Forward P/E NTM
    "XLP":  "forward_pe",      # Staples       — Forward P/E + div yield
    "XLE":  "ev_ebitda",       # Energy        — EV/EBITDA + FCF yield
    "XLF":  "pb",              # Financials    — P/B (not P/E)
    "XLV":  "forward_pe",      # Health Care   — Forward P/E
    "XLI":  "ev_ebitda",       # Industrials   — EV/EBITDA
    "XLB":  "ev_ebitda",       # Materials     — EV/EBITDA
    "XLRE": "owner_earnings_yield",  # Real Estate — Owner Earnings Yield / P/FFO proxy
    "XLK":  "forward_pe",      # Technology    — Forward P/E
    "XLU":  "dividend_yield",  # Utilities     — Dividend Yield + forward P/E
}

# Rate-sensitive sectors (need rate-adjusted gap)
RATE_SENSITIVE: Dict[str, float] = {
    "XLU":  1.0,   # full rate sensitivity
    "XLRE": 1.0,
    "XLK":  0.6,   # equity duration
    "XLF":  0.4,   # NIM-driven, partial
    "XLC":  0.3,
}

# Weight matrix: sector × multiple (rows sum to ~1.0)
# Based on research Section 2 weight matrix
WEIGHT_MATRIX: Dict[str, Dict[str, float]] = {
    "XLC":  {"forward_pe": 0.10, "ev_ebitda": 0.45, "fcf_yield": 0.15,
             "pb": 0.00, "dividend_yield": 0.05, "ev_sales": 0.25, "owner_earnings_yield": 0.00},
    "XLY":  {"forward_pe": 0.45, "ev_ebitda": 0.25, "fcf_yield": 0.10,
             "pb": 0.00, "dividend_yield": 0.05, "ev_sales": 0.15, "owner_earnings_yield": 0.00},
    "XLP":  {"forward_pe": 0.50, "ev_ebitda": 0.10, "fcf_yield": 0.10,
             "pb": 0.00, "dividend_yield": 0.30, "ev_sales": 0.00, "owner_earnings_yield": 0.00},
    "XLE":  {"forward_pe": 0.05, "ev_ebitda": 0.45, "fcf_yield": 0.35,
             "pb": 0.00, "dividend_yield": 0.05, "ev_sales": 0.10, "owner_earnings_yield": 0.00},
    "XLF":  {"forward_pe": 0.10, "ev_ebitda": 0.00, "fcf_yield": 0.05,
             "pb": 0.70, "dividend_yield": 0.15, "ev_sales": 0.00, "owner_earnings_yield": 0.00},
    "XLV":  {"forward_pe": 0.45, "ev_ebitda": 0.25, "fcf_yield": 0.15,
             "pb": 0.00, "dividend_yield": 0.05, "ev_sales": 0.10, "owner_earnings_yield": 0.00},
    "XLI":  {"forward_pe": 0.20, "ev_ebitda": 0.45, "fcf_yield": 0.20,
             "pb": 0.00, "dividend_yield": 0.05, "ev_sales": 0.10, "owner_earnings_yield": 0.00},
    "XLB":  {"forward_pe": 0.10, "ev_ebitda": 0.55, "fcf_yield": 0.15,
             "pb": 0.00, "dividend_yield": 0.05, "ev_sales": 0.15, "owner_earnings_yield": 0.00},
    "XLRE": {"forward_pe": 0.05, "ev_ebitda": 0.10, "fcf_yield": 0.10,
             "pb": 0.20, "dividend_yield": 0.25, "ev_sales": 0.00, "owner_earnings_yield": 0.30},
    "XLK":  {"forward_pe": 0.50, "ev_ebitda": 0.10, "fcf_yield": 0.15,
             "pb": 0.00, "dividend_yield": 0.05, "ev_sales": 0.20, "owner_earnings_yield": 0.00},
    "XLU":  {"forward_pe": 0.30, "ev_ebitda": 0.15, "fcf_yield": 0.10,
             "pb": 0.00, "dividend_yield": 0.45, "ev_sales": 0.00, "owner_earnings_yield": 0.00},
}

# Long-run median primary multiple per sector (Damodaran sector data)
# Used as the intercept in compute_justified_multiple instead of a universal 2.80
SECTOR_BASE_LOG_MULTIPLE: Dict[str, float] = {
    "XLC":  2.71,   # EV/EBITDA ~15
    "XLY":  3.00,   # Forward P/E ~20
    "XLP":  3.00,   # Forward P/E ~20
    "XLE":  2.08,   # EV/EBITDA ~8  (cyclical mid-cycle anchor)
    "XLF":  0.41,   # P/B ~1.5
    "XLV":  2.89,   # Forward P/E ~18
    "XLI":  2.64,   # EV/EBITDA ~14
    "XLB":  2.48,   # EV/EBITDA ~12
    "XLRE": 2.71,   # Owner-earnings implied ~15
    "XLK":  3.22,   # Forward P/E ~25
    "XLU":  3.40,   # Dividend yield implied ~30
}

# ──────────────────────────────────────────────────────────────
# Data models
# ──────────────────────────────────────────────────────────────

@dataclass
class SectorFundamentals:
    """Aggregated fundamentals for one sector ETF."""
    etf: str

    # Primary / secondary multiples (aggregated from holdings)
    forward_pe:           float = float("nan")
    ttm_pe:               float = float("nan")
    ev_ebitda:            float = float("nan")
    pb:                   float = float("nan")
    dividend_yield:       float = float("nan")
    fcf_yield:            float = float("nan")
    ev_sales:             float = float("nan")
    owner_earnings_yield: float = float("nan")

    # Growth (NTM consensus)
    eps_growth_ntm:       float = float("nan")
    revenue_growth_ntm:   float = float("nan")
    ebitda_growth_ntm:    float = float("nan")

    # Quality
    roe:                  float = float("nan")
    roic:                 float = float("nan")
    gross_margin:         float = float("nan")
    net_margin:           float = float("nan")
    accrual_ratio:        float = float("nan")   # CFO/NI — income quality
    cfo_ni_ratio:         float = float("nan")

    # Revision momentum
    eps_revision_3m:      float = float("nan")   # % change in consensus EPS (3m)
    beat_rate_4q:         float = float("nan")   # fraction of beats last 4Q

    # Coverage metadata
    coverage_primary:     float = float("nan")   # weight fraction with primary multiple
    coverage_growth:      float = float("nan")
    holdings_count:       int   = 0
    primary_multiple_used: str  = "ttm_pe"       # what was actually used


@dataclass
class FJSResult:
    """Output of compute_sector_fjs() for one sector."""
    etf: str
    fjs: float                    # final score [0,1]
    fjs_raw: float                # before coverage penalty
    mis: float                    # mispricing strength [0,1]
    z_delta: float                # robust z-score of obs-justified gap
    delta: float                  # log(M_obs) - log(M_hat)
    m_obs: float                  # observed composite multiple
    m_hat: float                  # justified multiple (model prediction)
    coverage_primary: float
    rate_gap: float               # rate-adjusted gap (nan if not sensitive)
    earnings_quality_score: float # [0,1], 1=high quality
    revision_score: float         # [0,1], 1=strong upward revisions
    primary_multiple_used: str
    sub_scores: Dict[str, float]  # per-multiple z-scores
    warnings: List[str]


# ──────────────────────────────────────────────────────────────
# Math utilities
# ──────────────────────────────────────────────────────────────

def _safe(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _winsorize_series(s: pd.Series, lo: float = 0.01, hi: float = 0.99) -> pd.Series:
    """Winsorize at p1/p99 — standard practice for multiple data."""
    s = pd.to_numeric(s, errors="coerce")
    q_lo = s.quantile(lo)
    q_hi = s.quantile(hi)
    return s.clip(lower=q_lo, upper=q_hi)


def _robust_zscore(series: pd.Series) -> pd.Series:
    """Median/MAD robust z-score — preferred for skewed multiple distributions."""
    med = series.median()
    mad = (series - med).abs().median()
    if mad < 1e-10:
        mad = series.std()
    if mad < 1e-10:
        return pd.Series(0.0, index=series.index)
    return (series - med) / (1.4826 * mad)   # 1.4826 = consistent scale factor


def _sigmoid(x: float, k: float = 2.0, tau: float = 0.5) -> float:
    """Sigmoid mapping for MIS score. k=steepness, tau=threshold."""
    try:
        return 1.0 / (1.0 + math.exp(-k * (x - tau)))
    except OverflowError:
        return 1.0 if x > tau else 0.0


def _log_safe(x: float) -> float:
    if math.isfinite(x) and x > 0.01:
        return math.log(x)
    return float("nan")


# ──────────────────────────────────────────────────────────────
# Holdings aggregation (Section 5)
# ──────────────────────────────────────────────────────────────

def _aggregate_earnings_yield(
    holdings: pd.DataFrame,
    fund_df: pd.DataFrame,
    pe_col: str = "pe",
) -> Tuple[float, float]:
    """
    Correct institutional aggregation: earnings yield (E/P), then invert.
    Returns (pe_etf, coverage_weight).
    """
    if holdings.empty or fund_df.empty:
        return float("nan"), 0.0

    h = holdings.merge(
        fund_df[["symbol", pe_col, "price"]].dropna(subset=["symbol"]),
        left_on="asset", right_on="symbol", how="left"
    )
    h["weight"] = pd.to_numeric(h["weightPercentage"], errors="coerce") / 100.0
    h[pe_col]   = pd.to_numeric(h[pe_col], errors="coerce")

    # Keep only holdings with valid positive PE
    valid = h[h[pe_col] > 0.5].copy()
    if valid.empty:
        return float("nan"), 0.0

    # Earnings yield = 1/PE
    valid["ey"] = 1.0 / valid[pe_col]
    total_w     = float(valid["weight"].sum())
    if total_w < 1e-6:
        return float("nan"), 0.0

    ey_etf      = float((valid["ey"] * valid["weight"]).sum() / total_w)
    coverage    = min(1.0, total_w)
    pe_etf      = 1.0 / ey_etf if ey_etf > 1e-6 else float("nan")
    return pe_etf, coverage


def _aggregate_ev_ebitda(
    holdings: pd.DataFrame,
    ext_fund_df: pd.DataFrame,
) -> Tuple[float, float]:
    """
    Ratio-of-sums: EV_port / EBITDA_port — the only correct method.
    Avoids the average-of-ratios bias.
    """
    if holdings.empty or ext_fund_df.empty:
        return float("nan"), 0.0

    needed = ["symbol", "enterpriseValue", "ebitda", "price", "sharesOutstanding"]
    available = [c for c in needed if c in ext_fund_df.columns]
    if "symbol" not in available:
        return float("nan"), 0.0

    h = holdings.merge(
        ext_fund_df[available].dropna(subset=["symbol"]),
        left_on="asset", right_on="symbol", how="left"
    )
    h["weight"]  = pd.to_numeric(h["weightPercentage"], errors="coerce") / 100.0
    h["ev"]      = pd.to_numeric(h.get("enterpriseValue", float("nan")), errors="coerce")
    h["ebitda"]  = pd.to_numeric(h.get("ebitda", float("nan")), errors="coerce")
    h["price"]   = pd.to_numeric(h.get("price", float("nan")), errors="coerce")
    h["shares"]  = pd.to_numeric(h.get("sharesOutstanding", float("nan")), errors="coerce")

    valid = h[
        h["ev"].notna() & h["ebitda"].notna() & (h["ebitda"] > 0) &
        h["price"].notna() & h["shares"].notna()
    ].copy()

    if valid.empty:
        return float("nan"), 0.0

    # Ownership fraction via shares held
    valid["shares_held"]    = valid["weight"] / valid["price"].clip(lower=0.01)
    valid["ownership_frac"] = valid["shares_held"] / valid["shares"].clip(lower=1.0)

    ev_port    = float((valid["ownership_frac"] * valid["ev"]).sum())
    ebitda_port= float((valid["ownership_frac"] * valid["ebitda"]).sum())

    if ebitda_port < 1.0:
        return float("nan"), 0.0

    coverage = min(1.0, float(valid["weight"].sum()))
    return ev_port / ebitda_port, coverage


def _aggregate_pb(
    holdings: pd.DataFrame,
    fund_df: pd.DataFrame,
) -> Tuple[float, float]:
    """
    P/B aggregation via book-value yield (BV/P), then invert.
    Correct for XLF.
    """
    if holdings.empty or fund_df.empty:
        return float("nan"), 0.0

    pb_col = next((c for c in ["pb", "priceToBookRatio", "priceToBookRatioTTM"]
                   if c in fund_df.columns), None)
    if pb_col is None:
        return float("nan"), 0.0

    h = holdings.merge(
        fund_df[["symbol", pb_col]].dropna(subset=["symbol"]),
        left_on="asset", right_on="symbol", how="left"
    )
    h["weight"] = pd.to_numeric(h["weightPercentage"], errors="coerce") / 100.0
    h["pb"]     = pd.to_numeric(h[pb_col], errors="coerce")

    valid = h[h["pb"] > 0.1].copy()
    if valid.empty:
        return float("nan"), 0.0

    valid["bv_yield"] = 1.0 / valid["pb"]
    total_w           = float(valid["weight"].sum())
    if total_w < 1e-6:
        return float("nan"), 0.0

    bv_yield_etf = float((valid["bv_yield"] * valid["weight"]).sum() / total_w)
    coverage     = min(1.0, total_w)
    return (1.0 / bv_yield_etf) if bv_yield_etf > 1e-6 else float("nan"), coverage


def _aggregate_dividend_yield(
    holdings: pd.DataFrame,
    fund_df: pd.DataFrame,
) -> Tuple[float, float]:
    """Dividend yield — linear aggregation (correct for yield metrics)."""
    if holdings.empty or fund_df.empty:
        return float("nan"), 0.0

    dy_col = next((c for c in ["dividendYield", "dividendYieldTTM", "dividend_yield"]
                   if c in fund_df.columns), None)
    if dy_col is None:
        return float("nan"), 0.0

    h = holdings.merge(
        fund_df[["symbol", dy_col]].dropna(subset=["symbol"]),
        left_on="asset", right_on="symbol", how="left"
    )
    h["weight"] = pd.to_numeric(h["weightPercentage"], errors="coerce") / 100.0
    h["dy"]     = pd.to_numeric(h[dy_col], errors="coerce")

    valid = h[h["dy"].notna() & (h["dy"] >= 0)].copy()
    if valid.empty:
        return float("nan"), 0.0

    total_w  = float(valid["weight"].sum())
    if total_w < 1e-6:
        return float("nan"), 0.0

    dy_etf   = float((valid["dy"] * valid["weight"]).sum() / total_w)
    coverage = min(1.0, total_w)
    # Normalise: FMP sometimes returns 0.03 or 3.0 for 3%
    if dy_etf > 1.0:
        dy_etf /= 100.0
    return dy_etf, coverage


def _aggregate_fcf_yield(
    holdings: pd.DataFrame,
    ext_fund_df: pd.DataFrame,
) -> Tuple[float, float]:
    """FCF yield = FCF / MarketCap, aggregated as yield (linear)."""
    if holdings.empty or ext_fund_df.empty:
        return float("nan"), 0.0

    fcf_col = next((c for c in ["freeCashFlow", "fcf", "freeCashFlowTTM"]
                    if c in ext_fund_df.columns), None)
    mc_col  = next((c for c in ["marketCap", "mktCap"]
                    if c in ext_fund_df.columns), None)
    if not fcf_col or not mc_col:
        return float("nan"), 0.0

    h = holdings.merge(
        ext_fund_df[["symbol", fcf_col, mc_col]].dropna(subset=["symbol"]),
        left_on="asset", right_on="symbol", how="left"
    )
    h["weight"] = pd.to_numeric(h["weightPercentage"], errors="coerce") / 100.0
    h["fcf"]    = pd.to_numeric(h[fcf_col], errors="coerce")
    h["mc"]     = pd.to_numeric(h[mc_col],  errors="coerce")

    valid = h[h["mc"].notna() & (h["mc"] > 0)].copy()
    if valid.empty:
        return float("nan"), 0.0

    valid["fcf_y"] = valid["fcf"] / valid["mc"]
    valid = valid[valid["fcf_y"].between(-0.5, 0.5)]   # winsorise extremes

    total_w = float(valid["weight"].sum())
    if total_w < 1e-6:
        return float("nan"), 0.0

    fcf_y   = float((valid["fcf_y"] * valid["weight"]).sum() / total_w)
    coverage= min(1.0, total_w)
    return fcf_y, coverage


def _aggregate_owner_earnings_yield(
    holdings: pd.DataFrame,
    ext_fund_df: pd.DataFrame,
) -> Tuple[float, float]:
    """
    Owner Earnings Yield = (OCF - Maintenance Capex) / MarketCap.
    FMP Owner Earnings = OCF + maintenance capex (see FMP docs).
    Fallback: use OCF - 70% of total capex as maintenance proxy.
    """
    if holdings.empty or ext_fund_df.empty:
        return float("nan"), 0.0

    oe_col  = next((c for c in ["ownerEarnings", "owner_earnings"]
                    if c in ext_fund_df.columns), None)
    ocf_col = next((c for c in ["operatingCashFlow", "cashFlowFromOperations"]
                    if c in ext_fund_df.columns), None)
    cap_col = next((c for c in ["capitalExpenditures", "capex", "capitalExpenditure"]
                    if c in ext_fund_df.columns), None)
    mc_col  = next((c for c in ["marketCap", "mktCap"]
                    if c in ext_fund_df.columns), None)

    if not mc_col:
        return float("nan"), 0.0

    cols = ["symbol", mc_col]
    for c in [oe_col, ocf_col, cap_col]:
        if c:
            cols.append(c)

    h = holdings.merge(
        ext_fund_df[cols].dropna(subset=["symbol"]),
        left_on="asset", right_on="symbol", how="left"
    )
    h["weight"] = pd.to_numeric(h["weightPercentage"], errors="coerce") / 100.0
    h["mc"]     = pd.to_numeric(h[mc_col], errors="coerce")

    if oe_col:
        h["oe"] = pd.to_numeric(h[oe_col], errors="coerce")
    elif ocf_col and cap_col:
        ocf = pd.to_numeric(h[ocf_col], errors="coerce")
        cap = pd.to_numeric(h[cap_col], errors="coerce").abs()
        h["oe"] = ocf - 0.70 * cap    # 70% of capex = maintenance proxy
    else:
        return float("nan"), 0.0

    valid = h[h["mc"].notna() & (h["mc"] > 0) & h["oe"].notna()].copy()
    if valid.empty:
        return float("nan"), 0.0

    valid["oey"] = valid["oe"] / valid["mc"]
    valid = valid[valid["oey"].between(-0.5, 0.5)]

    total_w = float(valid["weight"].sum())
    if total_w < 1e-6:
        return float("nan"), 0.0

    oey     = float((valid["oey"] * valid["weight"]).sum() / total_w)
    coverage= min(1.0, total_w)
    return oey, coverage


# ──────────────────────────────────────────────────────────────
# Sector fundamentals aggregation
# ──────────────────────────────────────────────────────────────

def aggregate_sector_fundamentals(
    etf: str,
    holdings_df: pd.DataFrame,
    fund_df: pd.DataFrame,
    ext_fund_df: pd.DataFrame,
) -> SectorFundamentals:
    """
    Aggregate all fundamentals for one sector ETF from holdings data.
    Handles missing data gracefully throughout.
    """
    sf = SectorFundamentals(etf=etf)
    h  = holdings_df[holdings_df["etfSymbol"] == etf].copy() if "etfSymbol" in holdings_df.columns else pd.DataFrame()

    if h.empty:
        logger.warning("No holdings for ETF %s — using ETF-level fallback", etf)
        # ETF-level fallback
        etf_row = fund_df[fund_df["symbol"] == etf] if not fund_df.empty else pd.DataFrame()
        if not etf_row.empty:
            r = etf_row.iloc[0]
            sf.ttm_pe       = _safe(r.get("pe"))
            sf.pb           = _safe(r.get("pb", r.get("priceToBookRatio")))
            sf.dividend_yield= _safe(r.get("dividendYield"))
            sf.roe          = _safe(r.get("roe", r.get("returnOnEquity")))
            sf.primary_multiple_used = "etf_fallback"
        return sf

    # ── P/E (TTM and forward) ─────────────────────────────────
    # Winsorize PE at holdings level before aggregation (Section 5)
    if "pe" in fund_df.columns:
        fund_df = fund_df.copy()
        fund_df["pe"] = _winsorize_series(fund_df["pe"], lo=0.02, hi=0.98)
    sf.ttm_pe, cov_pe = _aggregate_earnings_yield(h, fund_df, "pe")

    if "forwardPE" in fund_df.columns:
        fund_df["forwardPE"] = _winsorize_series(fund_df["forwardPE"], lo=0.02, hi=0.98)
        sf.forward_pe, cov_fpe = _aggregate_earnings_yield(h, fund_df, "forwardPE")
    else:
        sf.forward_pe, cov_fpe = sf.ttm_pe, cov_pe

    # Forward P/E from NTM EPS consensus (analyst estimates — more accurate)
    if not ext_fund_df.empty and "ntmEpsConsensus" in ext_fund_df.columns and "price" in fund_df.columns:
        h_est = h.merge(
            ext_fund_df[["symbol", "ntmEpsConsensus"]].dropna(subset=["symbol"]),
            left_on="asset", right_on="symbol", how="left"
        ).merge(
            fund_df[["symbol", "price"]].dropna(subset=["symbol"]),
            on="symbol", how="left", suffixes=("", "_fund")
        )
        h_est["weight"]  = pd.to_numeric(h_est["weightPercentage"], errors="coerce") / 100.0
        h_est["ntm_eps"] = pd.to_numeric(h_est["ntmEpsConsensus"], errors="coerce")
        h_est["px"]      = pd.to_numeric(h_est.get("price_fund", h_est.get("price")), errors="coerce")
        valid_est = h_est[(h_est["ntm_eps"] > 0.01) & (h_est["px"] > 0.01)].copy()
        if not valid_est.empty:
            valid_est["fpe"] = valid_est["px"] / valid_est["ntm_eps"]
            valid_est["fpe"] = _winsorize_series(valid_est["fpe"], lo=0.02, hi=0.98)
            valid_est = valid_est[valid_est["fpe"] > 0]
            tw = float(valid_est["weight"].sum())
            if tw > 1e-6:
                ey_ntm   = float((valid_est["weight"] / valid_est["fpe"]).sum() / tw)
                sf.forward_pe  = 1.0 / ey_ntm if ey_ntm > 1e-6 else sf.forward_pe
                cov_fpe  = min(1.0, tw)

    # ── EV/EBITDA (ratio-of-sums) ─────────────────────────────
    sf.ev_ebitda, cov_ev = _aggregate_ev_ebitda(h, ext_fund_df)

    # ── P/B ──────────────────────────────────────────────────
    sf.pb, cov_pb = _aggregate_pb(h, fund_df)
    if math.isnan(sf.pb):
        sf.pb, cov_pb = _aggregate_pb(h, ext_fund_df)

    # ── Dividend yield ────────────────────────────────────────
    sf.dividend_yield, cov_dy = _aggregate_dividend_yield(h, fund_df)
    if math.isnan(sf.dividend_yield):
        sf.dividend_yield, cov_dy = _aggregate_dividend_yield(h, ext_fund_df)

    # ── FCF yield ─────────────────────────────────────────────
    sf.fcf_yield, cov_fcf = _aggregate_fcf_yield(h, ext_fund_df)

    # ── Owner Earnings Yield (XLRE) ───────────────────────────
    sf.owner_earnings_yield, cov_oey = _aggregate_owner_earnings_yield(h, ext_fund_df)

    # ── EV/Sales proxy ───────────────────────────────────────
    ev_sales_col = next((c for c in ["priceToSalesRatioTTM", "ps", "ev_sales"]
                         if c in fund_df.columns), None)
    if ev_sales_col:
        ev_s_series = _winsorize_series(
            h.merge(fund_df[["symbol", ev_sales_col]], left_on="asset",
                    right_on="symbol", how="left")[ev_sales_col]
        )
        if ev_s_series.notna().any():
            w = pd.to_numeric(h["weightPercentage"], errors="coerce") / 100.0
            sf.ev_sales = float((ev_s_series * w).sum() / w[ev_s_series.notna()].sum())

    # ── Quality metrics (winsorized at holdings level) ────────
    for col, attr, lo, hi in [
        ("returnOnEquity",          "roe",          -0.30, 0.50),
        ("roe",                     "roe",          -0.30, 0.50),
        ("returnOnInvestedCapital", "roic",         -0.10, 0.40),
        ("roic",                    "roic",         -0.10, 0.40),
        ("grossProfitMargin",       "gross_margin",  0.00, 0.95),
        ("netProfitMargin",         "net_margin",   -0.20, 0.50),
    ]:
        if math.isnan(getattr(sf, attr, float("nan"))) and col in fund_df.columns:
            h_m  = h.merge(fund_df[["symbol", col]], left_on="asset",
                           right_on="symbol", how="left")
            vals = _winsorize_series(pd.to_numeric(h_m[col], errors="coerce"),
                                     lo=0.02, hi=0.98).clip(lo, hi)
            w    = pd.to_numeric(h["weightPercentage"], errors="coerce") / 100.0
            mask = vals.notna()
            if mask.any():
                setattr(sf, attr,
                        float((vals * w)[mask].sum() / w[mask].sum()))

    # ── Growth (NTM consensus) ────────────────────────────────
    if not ext_fund_df.empty:
        for sym_col, growth_col, attr in [
            ("symbol", "epsGrowthNTM",     "eps_growth_ntm"),
            ("symbol", "revenueGrowthNTM", "revenue_growth_ntm"),
            ("symbol", "ebitdaGrowthNTM",  "ebitda_growth_ntm"),
        ]:
            if growth_col in ext_fund_df.columns:
                h_g = h.merge(ext_fund_df[["symbol", growth_col]], left_on="asset",
                              right_on="symbol", how="left")
                vals= pd.to_numeric(h_g[growth_col], errors="coerce")
                w   = pd.to_numeric(h["weightPercentage"], errors="coerce") / 100.0
                if vals.notna().any():
                    setattr(sf, attr,
                            float((vals * w).sum() / w[vals.notna()].sum()))

    # ── ROIC from key-metrics-ttm (preferred over ratios) ────
    if not ext_fund_df.empty and "returnOnInvestedCapital" in ext_fund_df.columns:
        h_roic = h.merge(
            ext_fund_df[["symbol", "returnOnInvestedCapital"]].dropna(subset=["symbol"]),
            left_on="asset", right_on="symbol", how="left"
        )
        roic_vals = _winsorize_series(
            pd.to_numeric(h_roic["returnOnInvestedCapital"], errors="coerce"),
            lo=0.02, hi=0.98
        ).clip(-0.10, 0.40)
        w_r = pd.to_numeric(h["weightPercentage"], errors="coerce") / 100.0
        mask_r = roic_vals.notna()
        if mask_r.any():
            sf.roic = float((roic_vals * w_r)[mask_r].sum() / w_r[mask_r].sum())

    # ── Earnings quality (CFO / NI accrual ratio) ────────────
    # Prefer netIncomeIS (income statement) over netIncome (cashflow statement)
    cfo_col = next((c for c in ["operatingCashFlow", "cashFlowFromOperations"]
                    if c in ext_fund_df.columns), None)
    ni_col  = next((c for c in ["netIncomeIS", "netIncome", "netIncomeRatio"]
                    if c in ext_fund_df.columns), None)
    if cfo_col and ni_col and not ext_fund_df.empty:
        h_q = h.merge(ext_fund_df[["symbol", cfo_col, ni_col]], left_on="asset",
                      right_on="symbol", how="left")
        cfo = pd.to_numeric(h_q[cfo_col], errors="coerce")
        ni  = pd.to_numeric(h_q[ni_col],  errors="coerce")
        valid_q = ni.abs() > 1e3
        if valid_q.any():
            ratios    = (cfo / ni.abs().clip(lower=1.0)).clip(-3.0, 5.0)
            w         = pd.to_numeric(h["weightPercentage"], errors="coerce") / 100.0
            sf.cfo_ni_ratio = float((ratios * w)[valid_q].sum() / w[valid_q].sum())
            # income quality score: CFO/NI > 1 is healthy
            sf.accrual_ratio = max(0.0, min(1.0, (sf.cfo_ni_ratio - 0.5) / 1.5))

    # ── Revision momentum ─────────────────────────────────────
    rev_col = next((c for c in ["epsRevision3m", "eps_revision_3m", "epsEstimateRevision"]
                    if c in ext_fund_df.columns), None)
    if rev_col and not ext_fund_df.empty:
        h_r = h.merge(ext_fund_df[["symbol", rev_col]], left_on="asset",
                      right_on="symbol", how="left")
        revs= pd.to_numeric(h_r[rev_col], errors="coerce")
        w   = pd.to_numeric(h["weightPercentage"], errors="coerce") / 100.0
        if revs.notna().any():
            sf.eps_revision_3m = float((revs * w).sum() / w[revs.notna()].sum())

    # ── Coverage: primary multiple ─────────────────────────────
    primary = SECTOR_PRIMARY_MULTIPLE.get(etf, "ttm_pe")
    sf.primary_multiple_used = primary
    cov_map = {
        "forward_pe":           cov_fpe,
        "ttm_pe":               cov_pe,
        "ev_ebitda":            cov_ev,
        "pb":                   cov_pb,
        "dividend_yield":       cov_dy,
        "fcf_yield":            cov_fcf,
        "owner_earnings_yield": cov_oey,
    }
    sf.coverage_primary = cov_map.get(primary, cov_pe)

    return sf


# ──────────────────────────────────────────────────────────────
# Justified multiple model (Section 2)
# ──────────────────────────────────────────────────────────────

def _get_primary_value(sf: SectorFundamentals) -> float:
    """Return the log of the primary multiple for the sector."""
    primary = sf.primary_multiple_used
    val_map = {
        "forward_pe":           sf.forward_pe if not math.isnan(sf.forward_pe) else sf.ttm_pe,
        "ttm_pe":               sf.ttm_pe,
        "ev_ebitda":            sf.ev_ebitda,
        "pb":                   sf.pb,
        "dividend_yield":       sf.dividend_yield,
        "fcf_yield":            sf.fcf_yield,
        "owner_earnings_yield": sf.owner_earnings_yield,
    }
    v = val_map.get(primary, sf.ttm_pe)
    # For yields: invert so that "more expensive" = higher number
    if primary in ("dividend_yield", "fcf_yield", "owner_earnings_yield"):
        v = 1.0 / v if (math.isfinite(v) and v > 0.001) else float("nan")
    return _log_safe(v)


def compute_justified_multiple(
    sf: SectorFundamentals,
    tnx_10y: float,
    credit_z: float,
) -> float:
    """
    Simple justified-multiple model:
    log(M_hat) = base + b_growth*G + b_quality*Q + b_rate*R + b_credit*C

    Coefficients derived from cross-sectional literature.
    No look-ahead: uses only available inputs.
    """
    # Growth factor G — normalised to ~0 mean, unit std
    g = float("nan")
    if not math.isnan(sf.eps_growth_ntm):
        g = sf.eps_growth_ntm
    elif not math.isnan(sf.revenue_growth_ntm):
        g = sf.revenue_growth_ntm * 0.7   # revenue growth discounted
    g = _safe(g, 0.08)    # fallback to long-run avg ~8%
    g = max(-0.30, min(0.60, g))   # winsorise extreme growth claims

    # Quality factor Q
    q_parts = []
    if not math.isnan(sf.roe):
        q_parts.append(min(max(sf.roe, -0.2), 0.5))
    if not math.isnan(sf.roic):
        q_parts.append(min(max(sf.roic, -0.1), 0.4))
    if not math.isnan(sf.gross_margin):
        q_parts.append(min(max(sf.gross_margin, 0.0), 0.8))
    q = float(np.mean(q_parts)) if q_parts else 0.12   # fallback

    # Rate factor R
    r = _safe(tnx_10y, 4.0) / 100.0   # normalise: rate in decimal
    r = max(0.0, min(0.12, r))

    # Credit factor C
    c = _safe(credit_z, 0.0)
    c = max(-4.0, min(4.0, c))

    # Cross-sectional calibration (approximate, based on literature)
    # b_growth ~0.8 (Damodaran PEG logic), b_quality ~0.4,
    # b_rate ~ -0.5 (higher rates → lower multiples),
    # b_credit ~ -0.15 (stress → discount)
    b_growth  =  0.80
    b_quality =  0.40
    b_rate    = -0.50
    b_credit  = -0.15

    base_log_m = SECTOR_BASE_LOG_MULTIPLE.get(sf.etf, 2.80)
    log_m_hat = (
        base_log_m
        + b_growth  * g
        + b_quality * q
        + b_rate    * r
        + b_credit  * c
    )
    return math.exp(log_m_hat)


# ──────────────────────────────────────────────────────────────
# Earnings quality score (Section 4)
# ──────────────────────────────────────────────────────────────

def compute_earnings_quality_score(sf: SectorFundamentals) -> float:
    """
    Composite earnings quality score [0,1].
    Higher = better quality (strengthens mean reversion case).
    """
    parts: List[float] = []

    # CFO/NI ratio (accrual proxy) — >1 is healthy
    if not math.isnan(sf.cfo_ni_ratio):
        score = min(1.0, max(0.0, (sf.cfo_ni_ratio - 0.3) / 1.2))
        parts.append(score)

    # Gross margin quality (stable high margins = quality)
    if not math.isnan(sf.gross_margin):
        score = min(1.0, max(0.0, (sf.gross_margin - 0.10) / 0.50))
        parts.append(score * 0.5)   # lower weight

    # ROIC (positive ROIC = value creation)
    if not math.isnan(sf.roic):
        score = min(1.0, max(0.0, (sf.roic - 0.05) / 0.20))
        parts.append(score)

    return float(np.mean(parts)) if parts else 0.5   # neutral if unknown


def compute_revision_score(sf: SectorFundamentals) -> float:
    """
    Revision momentum [0,1].
    > 0.5 = upward revisions (weakens FJS — mispricing more likely)
    < 0.5 = downward revisions (strengthens FJS — value trap risk)
    """
    if math.isnan(sf.eps_revision_3m):
        return 0.5   # neutral

    # Map revision % to [0,1]: 0% rev → 0.5, +5% → 0.8, -5% → 0.2
    rev = max(-0.20, min(0.20, sf.eps_revision_3m))
    return min(1.0, max(0.0, 0.5 + rev * 3.0))


# ──────────────────────────────────────────────────────────────
# Rate-adjusted gap (Section 3)
# ──────────────────────────────────────────────────────────────

def compute_rate_gap(
    etf: str,
    log_m_obs: float,
    log_m_hat: float,
    tnx_10y: float,
    tnx_baseline: float = 2.5,   # pre-2022 baseline rate
) -> float:
    """
    Rate-adjusted gap for rate-sensitive sectors.
    If the gap is explained by the rate level change, it's a justified repricing.
    Returns nan for non-rate-sensitive sectors.
    """
    sensitivity = RATE_SENSITIVE.get(etf, 0.0)
    if sensitivity < 0.1:
        return float("nan")

    if not (math.isfinite(log_m_obs) and math.isfinite(log_m_hat)):
        return float("nan")

    raw_gap  = log_m_obs - log_m_hat
    tnx      = _safe(tnx_10y, 4.0)
    rate_adj = sensitivity * (-0.50) * (tnx/100.0 - tnx_baseline/100.0)

    # Gap unexplained by rates:
    return raw_gap - rate_adj


# ──────────────────────────────────────────────────────────────
# Composite multiple (Section 2 weight matrix)
# ──────────────────────────────────────────────────────────────

def compute_composite_multiple(sf: SectorFundamentals) -> Tuple[float, float]:
    """
    Weighted composite of all available multiples for the sector.
    Returns (composite_multiple, coverage_fraction).
    """
    weights = WEIGHT_MATRIX.get(sf.etf, WEIGHT_MATRIX["XLK"])

    # Map multiple name → value (all converted to "price-like" direction)
    mult_values = {
        "forward_pe":           sf.forward_pe if not math.isnan(sf.forward_pe) else sf.ttm_pe,
        "ev_ebitda":            sf.ev_ebitda,
        "fcf_yield":            (1.0 / sf.fcf_yield) if (not math.isnan(sf.fcf_yield) and sf.fcf_yield > 0.005) else float("nan"),
        "pb":                   sf.pb,
        "dividend_yield":       (1.0 / sf.dividend_yield) if (not math.isnan(sf.dividend_yield) and sf.dividend_yield > 0.005) else float("nan"),
        "ev_sales":             sf.ev_sales,
        "owner_earnings_yield": (1.0 / sf.owner_earnings_yield) if (not math.isnan(sf.owner_earnings_yield) and sf.owner_earnings_yield > 0.005) else float("nan"),
    }

    total_w = 0.0
    log_composite = 0.0
    for mult, w in weights.items():
        v = mult_values.get(mult, float("nan"))
        lv = _log_safe(v)
        if math.isfinite(lv) and w > 0:
            log_composite += w * lv
            total_w       += w

    if total_w < 0.05:
        return float("nan"), 0.0

    return math.exp(log_composite / total_w), total_w


# ──────────────────────────────────────────────────────────────
# Main FJS computation
# ──────────────────────────────────────────────────────────────

def compute_sector_fjs(
    sf: SectorFundamentals,
    direction: str,          # "LONG" or "SHORT"
    tnx_10y: float,
    credit_z: float,
    regime: str = "NORMAL",  # CALM / NORMAL / TENSION / CRISIS
    sigmoid_k: float = 2.5,
    sigmoid_tau: float = 0.4,
    cross_sectional_z: Optional[float] = None,  # pre-computed if available
) -> FJSResult:
    """
    Compute the institutional-grade FJS for one sector.

    FJS = 1 - MIS  where MIS = sigmoid(k * (dir_sign * z_delta - tau))
    FJS close to 0: fundamentals do NOT justify dislocation → mispricing candidate
    FJS close to 1: fundamentals JUSTIFY dislocation → repricing, not mean reversion
    """
    warnings: List[str] = []
    dir_sign = -1.0 if direction == "LONG" else 1.0

    # ── Step A: Observed composite multiple ───────────────────
    m_obs, _ = compute_composite_multiple(sf)
    log_m_obs = _log_safe(m_obs)

    # ── Step B: Justified multiple ────────────────────────────
    m_hat = compute_justified_multiple(sf, tnx_10y, credit_z)
    log_m_hat = _log_safe(m_hat)

    # ── Step C: Residual gap ──────────────────────────────────
    if math.isfinite(log_m_obs) and math.isfinite(log_m_hat):
        delta = log_m_obs - log_m_hat
    else:
        delta = float("nan")
        warnings.append("Cannot compute delta: missing obs or justified multiple")

    # ── Step D: Rate-adjusted gap ─────────────────────────────
    rate_gap = compute_rate_gap(sf.etf, log_m_obs, log_m_hat, tnx_10y)

    # Blend raw and rate-adjusted gap for rate-sensitive sectors (Section 3)
    # Z_adj = 0.6·Z_delta + 0.4·Z_gap(rate)
    # We apply the blend at delta level using a normalized rate_gap (÷ 0.25 assumed scale)
    effective_delta = delta
    if math.isfinite(rate_gap) and sf.etf in RATE_SENSITIVE and math.isfinite(delta):
        rate_weight = RATE_SENSITIVE.get(sf.etf, 0.5) * 0.40   # scale by sector sensitivity
        raw_weight  = 1.0 - rate_weight
        effective_delta = raw_weight * delta + rate_weight * rate_gap

    # ── Step E: Cross-sectional z-score ──────────────────────
    # If pre-computed cross-sectional z provided, use it
    # Otherwise use a simple normalisation: assume delta ~ N(0, 0.3)
    z_delta = cross_sectional_z if (cross_sectional_z is not None and math.isfinite(cross_sectional_z))               else (_safe(effective_delta, 0.0) / 0.30)

    # ── Step F: MIS and FJS raw ───────────────────────────────
    if not math.isfinite(z_delta):
        mis    = 0.5   # neutral
        fjs_raw= 0.5
        warnings.append("z_delta not finite — using neutral FJS=0.5")
    else:
        mis     = _sigmoid(dir_sign * z_delta, k=sigmoid_k, tau=sigmoid_tau)
        fjs_raw = 1.0 - mis

    # ── Step G: Direction-aware earnings quality adjustment ───
    eq_score  = compute_earnings_quality_score(sf)
    rev_score = compute_revision_score(sf)

    # LONG: cheap sector with high quality + upward revisions = clean mispricing
    #        → lower FJS (less justified, more tradeable)
    # SHORT: expensive sector with high quality + upward revisions = justified
    #        → raise FJS (more justified, less tradeable as short)
    # The sign flips with direction: dir_adj = +1 for LONG, -1 for SHORT
    dir_adj = 1.0 if direction == "LONG" else -1.0
    eq_adj = 0.0
    if not math.isnan(sf.accrual_ratio):
        eq_adj -= dir_adj * 0.08 * (eq_score - 0.5)
    if not math.isnan(sf.eps_revision_3m):
        eq_adj -= dir_adj * 0.06 * (rev_score - 0.5)

    fjs_adjusted = max(0.0, min(1.0, fjs_raw + eq_adj))

    # ── Step H: Regime gate ───────────────────────────────────
    # In CRISIS, macro dominates — reduce weight on fundamentals
    if regime == "CRISIS":
        fjs_adjusted = 0.5 + 0.4 * (fjs_adjusted - 0.5)  # compress toward 0.5
    elif regime == "TENSION" and sf.etf in RATE_SENSITIVE:
        fjs_adjusted = 0.5 + 0.6 * (fjs_adjusted - 0.5)

    # ── Step I: Coverage penalty ──────────────────────────────
    coverage = _safe(sf.coverage_primary, 0.0)
    fjs_final = max(0.0, min(1.0,
        fjs_adjusted * (0.70 + 0.30 * coverage)
    ))

    # ── Sub-scores ────────────────────────────────────────────
    sub_scores = {
        "forward_pe_z":  (_log_safe(sf.forward_pe if not math.isnan(sf.forward_pe) else sf.ttm_pe) - 2.8) / 0.4
                         if math.isfinite(_log_safe(sf.forward_pe if not math.isnan(sf.forward_pe) else sf.ttm_pe)) else float("nan"),
        "ev_ebitda_z":   (_log_safe(sf.ev_ebitda) - 2.6) / 0.4 if math.isfinite(_log_safe(sf.ev_ebitda)) else float("nan"),
        "pb_z":          (_log_safe(sf.pb) - 0.9) / 0.6 if math.isfinite(_log_safe(sf.pb)) else float("nan"),
        "div_yield_z":   (sf.dividend_yield - 0.025) / 0.015 if math.isfinite(sf.dividend_yield) else float("nan"),
        "fcf_yield_z":   (sf.fcf_yield - 0.04) / 0.025 if math.isfinite(sf.fcf_yield) else float("nan"),
        "eq_score":      eq_score,
        "rev_score":     rev_score,
        "delta_raw":     delta,
        "rate_gap":      rate_gap,
    }

    return FJSResult(
        etf=sf.etf,
        fjs=fjs_final,
        fjs_raw=fjs_raw,
        mis=mis,
        z_delta=z_delta,
        delta=delta,
        m_obs=m_obs if math.isfinite(m_obs) else float("nan"),
        m_hat=m_hat,
        coverage_primary=coverage,
        rate_gap=rate_gap,
        earnings_quality_score=eq_score,
        revision_score=rev_score,
        primary_multiple_used=sf.primary_multiple_used,
        sub_scores=sub_scores,
        warnings=warnings,
    )


# ──────────────────────────────────────────────────────────────
# Cross-sectional scoring (run across all sectors together)
# ──────────────────────────────────────────────────────────────

def compute_all_sectors_fjs(
    sector_fundamentals: Dict[str, SectorFundamentals],
    direction_map: Dict[str, str],       # etf → "LONG" / "SHORT"
    tnx_10y: float,
    credit_z: float,
    regime: str = "NORMAL",
) -> Dict[str, FJSResult]:
    """
    Compute FJS for all sectors cross-sectionally.
    The cross-sectional z-score of delta is computed using robust
    median/MAD normalisation across all 11 sectors.
    """
    # First pass: compute deltas for all sectors
    deltas: Dict[str, float] = {}
    raw_results: Dict[str, tuple] = {}

    for etf, sf in sector_fundamentals.items():
        m_obs, _ = compute_composite_multiple(sf)
        m_hat    = compute_justified_multiple(sf, tnx_10y, credit_z)
        log_obs  = _log_safe(m_obs)
        log_hat  = _log_safe(m_hat)

        delta = log_obs - log_hat if (math.isfinite(log_obs) and math.isfinite(log_hat)) else float("nan")

        # Rate adjustment
        rate_gap = compute_rate_gap(etf, log_obs, log_hat, tnx_10y)
        effective_delta = rate_gap if (math.isfinite(rate_gap) and etf in RATE_SENSITIVE) else delta

        deltas[etf] = effective_delta
        raw_results[etf] = (delta, rate_gap, m_obs, m_hat)

    # Cross-sectional robust z-score (median/MAD across sectors)
    # Winsorize deltas at p5/p95 before z-score (Section 5: winsorize before z)
    valid_deltas = [v for v in deltas.values() if math.isfinite(v)]
    if len(valid_deltas) >= 3:
        delta_arr = np.array(valid_deltas)
        p5, p95   = np.percentile(delta_arr, [5, 95])
        # Winsorize then compute MAD z-score
        winsorized = {etf: float(np.clip(v, p5, p95)) if math.isfinite(v) else float("nan")
                      for etf, v in deltas.items()}
        w_valid    = [v for v in winsorized.values() if math.isfinite(v)]
        med = float(np.median(w_valid))
        mad = float(np.median([abs(v - med) for v in w_valid]))
        mad = max(mad, 1e-6) * 1.4826   # consistent scale factor
        z_scores = {etf: (winsorized[etf] - med) / mad
                    if math.isfinite(winsorized.get(etf, float("nan"))) else float("nan")
                    for etf in deltas}
    else:
        z_scores = {etf: float("nan") for etf in deltas}

    # Second pass: compute final FJS with cross-sectional z-scores
    results: Dict[str, FJSResult] = {}
    for etf, sf in sector_fundamentals.items():
        direction = direction_map.get(etf, "LONG")
        results[etf] = compute_sector_fjs(
            sf=sf,
            direction=direction,
            tnx_10y=tnx_10y,
            credit_z=credit_z,
            regime=regime,
            cross_sectional_z=z_scores.get(etf),
        )

    return results


# ──────────────────────────────────────────────────────────────
# Backward-compatible wrapper (replaces compute_fundamental_justification_score)
# ──────────────────────────────────────────────────────────────

def compute_fjs_from_row(
    row: Dict[str, Any],
    sf: Optional[SectorFundamentals] = None,
    tnx_10y: float = 4.0,
    credit_z: float = 0.0,
    regime: str = "NORMAL",
) -> float:
    """
    Backward-compatible wrapper for attribution.py.
    Falls back to simple P/E logic if SectorFundamentals not available.
    """
    if sf is not None:
        direction = str(row.get("direction", "LONG"))
        result    = compute_sector_fjs(sf, direction, tnx_10y, credit_z, regime)
        return result.fjs

    # Legacy fallback (original single-P/E logic)
    from analytics.attribution import compute_fundamental_justification_score as _legacy
    return _legacy(
        direction=str(row.get("direction", "NEUTRAL")),
        rel_pe_vs_spy=float(row.get("rel_pe_vs_spy", float("nan"))),
        rel_ey_vs_spy=float(row.get("rel_earnings_yield_vs_spy", float("nan"))),
        covered_weight=float(row.get("fund_covered_weight", 0.0)),
        neg_or_missing_weight=float(row.get("neg_or_missing_earnings_weight", 1.0)),
        fund_source=str(row.get("fund_source", "UNKNOWN")),
    )
