"""
analytics/attribution.py
==========================
Multi-layer attribution engine for the SRV Quantamental DSS.

Two attribution frameworks:
  1. DSS Signal Attribution — per-sector conviction scoring (SDS, FJS, MSS, STF, MC)
  2. Brinson Performance Attribution — sector allocation + selection effects
  3. Multi-Factor Return Decomposition — SPY/rates/dollar/credit/alpha breakdown
  4. Risk Attribution — marginal risk contribution per factor

DSS Attribution:
  SDS = Statistical Dislocation Score (PCA residual z-score quality)
  FJS = Fundamental Justification Score (PE, earnings alignment)
  MSS = Macro Shift Score (beta instability, VIX, credit)
  STF = Structural Trend Filter (trend alignment with direction)
  MC  = Mispricing Confidence (composite of all four)

Brinson Attribution (Brinson, Hood, Beebower 1986):
  Total excess return = Allocation Effect + Selection Effect + Interaction
  Allocation: over/underweight sectors that outperformed/underperformed
  Selection: picking better/worse assets within each sector

Ref: Brinson, Hood, Beebower (1986) — Determinants of Portfolio Performance
Ref: Grinold & Kahn (2000) — Active Portfolio Management
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def clip01(x: float) -> float:
    """Clamp x to [0, 1], treating non-finite values as 0."""
    if not math.isfinite(x):
        return 0.0
    return max(0.0, min(1.0, float(x)))


def safe_float(x: Any, default: float = float("nan")) -> float:
    """Convert x to float, returning *default* on any failure."""
    try:
        return float(x)
    except Exception:
        return default


@dataclass(frozen=True)
class AttributionResult:
    # Core scores
    sds: float
    fjs: float
    mss: float
    stf: float
    mc: float

    # Raw structural diagnostics
    trend_ratio_slope_63d: float
    trend_ratio_slope_126d: float
    beta_instability: float
    corr_instability: float
    corr_shift_score: float

    # DSS explanation layer
    dislocation_label: str
    fundamental_label: str
    macro_label: str
    structural_label: str
    mc_label: str
    action_bias: str
    risk_label: str
    interpretation: str
    explanation_tags: List[str]


# ==========================================================
# Core scoring blocks
# ==========================================================
def compute_statistical_dislocation_score(
    z: float,
    disp_z: float,
    half_life: float,
    z_lo: float = 0.75,
    z_hi: float = 2.75,
) -> float:
    """
    Statistical Dislocation Score (SDS) in [0, 1].

    Combines PCA residual z-score strength, dispersion z-score, and
    OU half-life quality into a single dislocation measure.
    """
    z_abs = abs(z) if math.isfinite(z) else 0.0
    z_strength = clip01((z_abs - z_lo) / max(1e-9, z_hi - z_lo))

    disp_strength = 0.0
    if math.isfinite(disp_z):
        disp_strength = clip01((disp_z + 0.5) / 2.5)

    hl_quality = 0.55
    if math.isfinite(half_life):
        if 5 <= half_life <= 90:
            hl_quality = 0.65 + 0.35 * math.exp(-abs(half_life - 35.0) / 45.0)
        elif half_life < 5:
            hl_quality = 0.55
        else:
            hl_quality = 0.50

    return clip01((0.75 * z_strength + 0.25 * disp_strength) * hl_quality)


def compute_fundamental_justification_score(
    direction: str,
    rel_pe_vs_spy: float,
    rel_ey_vs_spy: float,
    covered_weight: float,
    neg_or_missing_weight: float,
    fund_source: str,
) -> float:
    """
    Fundamental Justification Score (FJS) in [0, 1].

    Measures whether the dislocation is explained by relative valuation
    (PE / earnings-yield vs SPY), adjusted for data coverage quality.
    """
    if direction not in {"LONG", "SHORT"}:
        return 0.0

    pe_signal = 0.0
    ey_signal = 0.0

    if direction == "LONG":
        if math.isfinite(rel_pe_vs_spy):
            pe_signal = clip01((rel_pe_vs_spy - 1.00) / 0.30)
        if math.isfinite(rel_ey_vs_spy):
            ey_signal = clip01((1.00 - rel_ey_vs_spy) / 0.25)
    else:
        if math.isfinite(rel_pe_vs_spy):
            pe_signal = clip01((1.00 - rel_pe_vs_spy) / 0.30)
        if math.isfinite(rel_ey_vs_spy):
            ey_signal = clip01((rel_ey_vs_spy - 1.00) / 0.25)

    structural_valuation = max(pe_signal, ey_signal)

    coverage_quality = clip01((covered_weight - 0.20) / 0.60) if math.isfinite(covered_weight) else 0.0
    missing_penalty = clip01(1.0 - (neg_or_missing_weight if math.isfinite(neg_or_missing_weight) else 1.0))
    source_bonus = 1.00 if str(fund_source).upper() == "HOLDINGS" else 0.65

    return clip01(
        structural_valuation
        * (0.65 * coverage_quality + 0.35 * missing_penalty)
        * source_bonus
    )


def compute_macro_shift_score(
    beta_tnx_60d: float,
    beta_dxy_60d: float,
    beta_spy_delta: float,
    corr_to_spy_delta: float,
    credit_z: float,
    vix_level: float,
    vix_soft: float,
    vix_hard: float,
    delta_corr_dist: float,
    beta_mag_norm: float = 1.25,
    beta_delta_norm: float = 0.35,
    corr_delta_norm: float = 0.20,
) -> Dict[str, float]:
    """
    Macro Shift Score (MSS) in [0, 1] plus diagnostic sub-scores.

    Blends rate/FX beta magnitude, beta/correlation instability,
    correlation structure shift, credit stress, and VIX level.
    Returns dict with keys: mss, beta_instability, corr_instability, corr_shift_score.
    """
    beta_mag = math.sqrt(
        (beta_tnx_60d if math.isfinite(beta_tnx_60d) else 0.0) ** 2
        + (beta_dxy_60d if math.isfinite(beta_dxy_60d) else 0.0) ** 2
    )

    beta_level_score = clip01(beta_mag / max(1e-9, beta_mag_norm))
    beta_instability = clip01(abs(beta_spy_delta) / max(1e-9, beta_delta_norm)) if math.isfinite(beta_spy_delta) else 0.0
    corr_instability = clip01(abs(corr_to_spy_delta) / max(1e-9, corr_delta_norm)) if math.isfinite(corr_to_spy_delta) else 0.0
    corr_shift_score = clip01((delta_corr_dist - 0.05) / 0.30) if math.isfinite(delta_corr_dist) else 0.0
    credit_score = clip01((-credit_z - 0.5) / 1.5) if math.isfinite(credit_z) else 0.0

    vix_score = 0.0
    if math.isfinite(vix_level):
        if vix_level <= vix_soft:
            vix_score = 0.0
        elif vix_level >= vix_hard:
            vix_score = 1.0
        else:
            vix_score = clip01((vix_level - vix_soft) / max(1e-9, (vix_hard - vix_soft)))

    mss = clip01(
        0.25 * beta_level_score
        + 0.20 * beta_instability
        + 0.15 * corr_instability
        + 0.20 * corr_shift_score
        + 0.10 * credit_score
        + 0.10 * vix_score
    )

    return {
        "mss": mss,
        "beta_instability": beta_instability,
        "corr_instability": corr_instability,
        "corr_shift_score": corr_shift_score,
    }


def compute_structural_trend_filter(
    slope_63d: float,
    slope_126d: float,
    direction: str,
    slope_norm: float = 0.12,
) -> float:
    """
    Structural Trend Filter (STF) in [0, 1].

    Measures how strongly the trend-ratio slope works *against* the
    proposed trade direction. High STF means the trend opposes the trade.
    """
    if direction not in {"LONG", "SHORT"}:
        return 1.0

    s63 = slope_63d if math.isfinite(slope_63d) else 0.0
    s126 = slope_126d if math.isfinite(slope_126d) else 0.0

    if direction == "LONG":
        adverse = max(0.0, s63) * 1.3 + max(0.0, s126)
    else:
        adverse = max(0.0, -s63) * 1.3 + max(0.0, -s126)

    return clip01(adverse / max(1e-9, slope_norm))


def compute_mispricing_confidence(
    sds: float,
    fjs: float,
    mss: float,
    stf: float,
) -> float:
    """
    Mispricing Confidence (MC) in [0, 1].

    High MC = strong dislocation with low fundamental justification,
    low macro shift risk, and low structural trend opposition.
    """
    base = clip01(sds)
    decay = (1.0 - clip01(fjs)) * (1.0 - clip01(mss)) * (1.0 - clip01(stf))
    return clip01(base * decay)


# ==========================================================
# Explanation helpers
# ==========================================================
def _label_bucket(x: float, lo: float = 0.33, hi: float = 0.66) -> str:
    if not math.isfinite(x):
        return "UNKNOWN"
    if x < lo:
        return "LOW"
    if x < hi:
        return "MEDIUM"
    return "HIGH"


def _dislocation_label(sds: float) -> str:
    b = _label_bucket(sds)
    if b == "LOW":
        return "Weak Statistical Dislocation"
    if b == "MEDIUM":
        return "Moderate Statistical Dislocation"
    return "Strong Statistical Dislocation"


def _fundamental_label(fjs: float) -> str:
    b = _label_bucket(fjs)
    if b == "LOW":
        return "Weak Fundamental Justification"
    if b == "MEDIUM":
        return "Partial Fundamental Justification"
    return "Strong Fundamental Justification"


def _macro_label(mss: float) -> str:
    b = _label_bucket(mss)
    if b == "LOW":
        return "Low Macro Shift Risk"
    if b == "MEDIUM":
        return "Moderate Macro Shift Risk"
    return "Elevated Macro Shift Risk"


def _structural_label(stf: float) -> str:
    b = _label_bucket(stf)
    if b == "LOW":
        return "Low Structural Trend Risk"
    if b == "MEDIUM":
        return "Moderate Structural Trend Risk"
    return "High Structural Trend Risk"


def _mc_label(mc: float) -> str:
    b = _label_bucket(mc, lo=0.25, hi=0.55)
    if b == "LOW":
        return "Weak Mispricing Confidence"
    if b == "MEDIUM":
        return "Moderate Mispricing Confidence"
    return "High Mispricing Confidence"


def _risk_label(mss: float, stf: float, corr_shift_score: float) -> str:
    composite = clip01(0.45 * mss + 0.35 * stf + 0.20 * corr_shift_score)
    b = _label_bucket(composite)
    if b == "LOW":
        return "Contained Risk"
    if b == "MEDIUM":
        return "Moderate Risk"
    return "Elevated Structural / Macro Risk"


def _action_bias(mc: float, mss: float, stf: float) -> str:
    if mc >= 0.60 and mss < 0.35 and stf < 0.35:
        return "LEAN_IN"
    if mc >= 0.35 and mss < 0.60 and stf < 0.60:
        return "SELECTIVE"
    if mc < 0.20 or mss >= 0.70 or stf >= 0.70:
        return "AVOID_OR_REDUCE"
    return "SMALL_SIZE_ONLY"


def _interpretation(
    sds: float,
    fjs: float,
    mss: float,
    stf: float,
    mc: float,
) -> str:
    if mc >= 0.60 and fjs < 0.30 and mss < 0.35 and stf < 0.35:
        return "Likely Clean Mispricing"
    if sds >= 0.55 and fjs >= 0.55:
        return "Dislocation May Be Fundamentally Justified"
    if sds >= 0.55 and mss >= 0.55:
        return "Dislocation May Be Macro Repricing"
    if sds >= 0.55 and stf >= 0.55:
        return "Dislocation May Reflect Structural Trend Shift"
    if mc < 0.20:
        return "Weak Mean-Reversion Setup"
    return "Mixed Signal / Requires PM Judgement"


def _build_explanation_tags(
    sds: float,
    fjs: float,
    mss: float,
    stf: float,
    mc: float,
    beta_instability: float,
    corr_instability: float,
    corr_shift_score: float,
) -> List[str]:
    tags: List[str] = []

    if sds >= 0.60:
        tags.append("strong_dislocation")
    elif sds >= 0.35:
        tags.append("moderate_dislocation")

    if fjs >= 0.60:
        tags.append("fundamentally_justified")
    elif fjs <= 0.20:
        tags.append("weak_fundamental_anchor")

    if mss >= 0.60:
        tags.append("macro_risk_high")
    elif mss >= 0.35:
        tags.append("macro_risk_moderate")

    if stf >= 0.60:
        tags.append("structural_trend_risk_high")
    elif stf >= 0.35:
        tags.append("structural_trend_risk_moderate")

    if beta_instability >= 0.50:
        tags.append("beta_instability")
    if corr_instability >= 0.50:
        tags.append("corr_instability")
    if corr_shift_score >= 0.50:
        tags.append("corr_structure_shift")

    if mc >= 0.60:
        tags.append("high_mc")
    elif mc < 0.20:
        tags.append("low_mc")

    if not tags:
        tags.append("mixed_signal")

    return tags


# ==========================================================
# Main row function
# ==========================================================
def compute_attribution_row(
    row: Dict[str, Any],
    *,
    vix_soft: float,
    vix_hard: float,
    sds_z_lo: float = 0.75,
    sds_z_hi: float = 2.75,
    mss_beta_mag_norm: float = 1.25,
    mss_beta_delta_norm: float = 0.35,
    mss_corr_delta_norm: float = 0.20,
    stf_slope_norm: float = 0.12,
) -> AttributionResult:
    """
    Compute the full DSS attribution for a single sector row.

    Orchestrates SDS, FJS, MSS, STF, and MC scoring, then builds
    human-readable labels, action bias, and explanation tags.
    """
    sds = compute_statistical_dislocation_score(
        z=safe_float(row.get("pca_residual_z")),
        disp_z=safe_float(row.get("market_dispersion_z")),
        half_life=safe_float(row.get("half_life_days_est")),
        z_lo=sds_z_lo,
        z_hi=sds_z_hi,
    )

    fjs = compute_fundamental_justification_score(
        direction=str(row.get("direction", "NEUTRAL")),
        rel_pe_vs_spy=safe_float(row.get("rel_pe_vs_spy")),
        rel_ey_vs_spy=safe_float(row.get("rel_earnings_yield_vs_spy")),
        covered_weight=safe_float(row.get("fund_covered_weight")),
        neg_or_missing_weight=safe_float(row.get("neg_or_missing_earnings_weight")),
        fund_source=str(row.get("fund_source", "UNKNOWN")),
    )

    macro_bits = compute_macro_shift_score(
        beta_tnx_60d=safe_float(row.get("beta_tnx_60d")),
        beta_dxy_60d=safe_float(row.get("beta_dxy_60d")),
        beta_spy_delta=safe_float(row.get("beta_spy_delta")),
        corr_to_spy_delta=safe_float(row.get("corr_to_spy_delta")),
        credit_z=safe_float(row.get("credit_z")),
        vix_level=safe_float(row.get("vix_level")),
        vix_soft=vix_soft,
        vix_hard=vix_hard,
        delta_corr_dist=safe_float(row.get("delta_corr_dist")),
        beta_mag_norm=mss_beta_mag_norm,
        beta_delta_norm=mss_beta_delta_norm,
        corr_delta_norm=mss_corr_delta_norm,
    )

    slope_63d = safe_float(row.get("trend_ratio_slope_63d"))
    slope_126d = safe_float(row.get("trend_ratio_slope_126d"))

    stf = compute_structural_trend_filter(
        slope_63d=slope_63d,
        slope_126d=slope_126d,
        direction=str(row.get("direction", "NEUTRAL")),
        slope_norm=stf_slope_norm,
    )

    mc = compute_mispricing_confidence(
        sds=sds,
        fjs=fjs,
        mss=macro_bits["mss"],
        stf=stf,
    )

    dislocation_label = _dislocation_label(sds)
    fundamental_label = _fundamental_label(fjs)
    macro_label = _macro_label(macro_bits["mss"])
    structural_label = _structural_label(stf)
    mc_label = _mc_label(mc)
    action_bias = _action_bias(mc, macro_bits["mss"], stf)
    risk_label = _risk_label(macro_bits["mss"], stf, macro_bits["corr_shift_score"])
    interpretation = _interpretation(sds, fjs, macro_bits["mss"], stf, mc)

    explanation_tags = _build_explanation_tags(
        sds=sds,
        fjs=fjs,
        mss=macro_bits["mss"],
        stf=stf,
        mc=mc,
        beta_instability=macro_bits["beta_instability"],
        corr_instability=macro_bits["corr_instability"],
        corr_shift_score=macro_bits["corr_shift_score"],
    )

    return AttributionResult(
        sds=sds,
        fjs=fjs,
        mss=macro_bits["mss"],
        stf=stf,
        mc=mc,
        trend_ratio_slope_63d=slope_63d,
        trend_ratio_slope_126d=slope_126d,
        beta_instability=macro_bits["beta_instability"],
        corr_instability=macro_bits["corr_instability"],
        corr_shift_score=macro_bits["corr_shift_score"],
        dislocation_label=dislocation_label,
        fundamental_label=fundamental_label,
        macro_label=macro_label,
        structural_label=structural_label,
        mc_label=mc_label,
        action_bias=action_bias,
        risk_label=risk_label,
        interpretation=interpretation,
        explanation_tags=explanation_tags,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Brinson Performance Attribution
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class BrinsonAttribution:
    """
    Brinson-style performance attribution for sector portfolios.

    Decomposes excess return vs benchmark (SPY) into:
      Allocation: returns from sector weight decisions
      Selection: returns from within-sector stock/ETF selection
      Interaction: cross-effect of allocation × selection
    """
    # Portfolio metrics
    portfolio_return: float
    benchmark_return: float
    excess_return: float

    # Attribution components
    allocation_effect: float          # ΔR from sector weighting decisions
    selection_effect: float           # ΔR from within-sector performance
    interaction_effect: float         # Cross-term

    # Per-sector breakdown
    sector_allocation: Dict[str, float]    # {sector: allocation_contribution}
    sector_selection: Dict[str, float]     # {sector: selection_contribution}

    # Top contributors
    top_allocators: List[Tuple[str, float]]     # Best allocation decisions
    top_selectors: List[Tuple[str, float]]       # Best selection decisions
    worst_allocators: List[Tuple[str, float]]
    worst_selectors: List[Tuple[str, float]]


def brinson_attribution(
    portfolio_weights: Dict[str, float],
    benchmark_weights: Dict[str, float],
    sector_returns: Dict[str, float],
    benchmark_return: float,
) -> BrinsonAttribution:
    """
    Compute Brinson-Fachler attribution.

    Parameters
    ----------
    portfolio_weights : {sector: weight} — portfolio weights (sum ~1)
    benchmark_weights : {sector: weight} — benchmark (SPY) sector weights
    sector_returns : {sector: return} — sector returns for the period
    benchmark_return : float — total benchmark return

    Returns
    -------
    BrinsonAttribution with allocation, selection, interaction effects
    """
    all_sectors = sorted(set(list(portfolio_weights.keys()) + list(benchmark_weights.keys())))

    port_return = sum(portfolio_weights.get(s, 0) * sector_returns.get(s, 0) for s in all_sectors)
    excess = port_return - benchmark_return

    alloc_by_sector: Dict[str, float] = {}
    select_by_sector: Dict[str, float] = {}

    total_alloc = 0.0
    total_select = 0.0
    total_interact = 0.0

    for s in all_sectors:
        wp = portfolio_weights.get(s, 0)
        wb = benchmark_weights.get(s, 0)
        rs = sector_returns.get(s, 0)
        rb = benchmark_return

        # Brinson-Fachler decomposition
        alloc = (wp - wb) * (rs - rb)
        select = wb * (rs - rb)
        interact = (wp - wb) * (rs - rb)

        alloc_by_sector[s] = round(alloc, 8)
        select_by_sector[s] = round(select, 8)

        total_alloc += alloc
        total_select += select

    total_interact = excess - total_alloc - total_select

    # Top/worst contributors
    alloc_sorted = sorted(alloc_by_sector.items(), key=lambda x: x[1], reverse=True)
    select_sorted = sorted(select_by_sector.items(), key=lambda x: x[1], reverse=True)

    return BrinsonAttribution(
        portfolio_return=round(port_return, 8),
        benchmark_return=round(benchmark_return, 8),
        excess_return=round(excess, 8),
        allocation_effect=round(total_alloc, 8),
        selection_effect=round(total_select, 8),
        interaction_effect=round(total_interact, 8),
        sector_allocation=alloc_by_sector,
        sector_selection=select_by_sector,
        top_allocators=alloc_sorted[:3],
        top_selectors=select_sorted[:3],
        worst_allocators=alloc_sorted[-3:],
        worst_selectors=select_sorted[-3:],
    )


# ═════════════════════════════════════════════════════════════════════════════
# Multi-Factor Return Decomposition
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class FactorDecomposition:
    """
    Multi-factor return decomposition.

    Decomposes portfolio returns into factor exposures:
      Market (SPY beta), Rates (TNX), Dollar (DXY), Credit (HYG), Alpha (residual)
    """
    # Factor contributions (daily or period)
    factor_returns: Dict[str, float]    # {factor: return_contribution}
    factor_betas: Dict[str, float]      # {factor: beta/exposure}

    # Summary
    systematic_return: float             # Sum of factor returns
    alpha_return: float                  # Unexplained (residual)
    total_return: float

    # R² and quality
    r_squared: float                     # Explained variance
    tracking_error: float                # Residual volatility (annualized)
    information_ratio: float             # Alpha / tracking_error

    # Per-factor detail
    factor_t_stats: Dict[str, float]     # Statistical significance per factor


def decompose_returns(
    portfolio_returns: pd.Series,
    factor_returns: pd.DataFrame,
    factor_names: Optional[List[str]] = None,
) -> FactorDecomposition:
    """
    OLS regression of portfolio returns on factor returns.

    Parameters
    ----------
    portfolio_returns : pd.Series — daily portfolio returns
    factor_returns : pd.DataFrame — daily factor returns (SPY, ^TNX, DXY, HYG, etc.)
    factor_names : optional — column names to use as factors
    """
    if factor_names:
        factors = factor_returns[factor_names].copy()
    else:
        factors = factor_returns.copy()

    # Align
    aligned = pd.DataFrame({"port": portfolio_returns}).join(factors, how="inner").dropna()
    if len(aligned) < 30:
        return FactorDecomposition(
            factor_returns={}, factor_betas={},
            systematic_return=0, alpha_return=0, total_return=0,
            r_squared=0, tracking_error=0, information_ratio=0,
            factor_t_stats={},
        )

    y = aligned["port"].values
    X = aligned.drop(columns=["port"]).values
    fnames = [c for c in aligned.columns if c != "port"]
    n, k = X.shape

    # OLS with intercept
    X_with_const = np.column_stack([np.ones(n), X])
    try:
        betas, residuals, rank, sv = np.linalg.lstsq(X_with_const, y, rcond=None)
    except np.linalg.LinAlgError:
        return FactorDecomposition(
            factor_returns={}, factor_betas={},
            systematic_return=0, alpha_return=0, total_return=0,
            r_squared=0, tracking_error=0, information_ratio=0,
            factor_t_stats={},
        )

    alpha_daily = betas[0]
    factor_betas_arr = betas[1:]
    y_pred = X_with_const @ betas
    resid = y - y_pred

    # R²
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-15)

    # Factor contributions (cumulative)
    factor_contribs = {}
    for i, fname in enumerate(fnames):
        factor_contribs[fname] = round(float(factor_betas_arr[i] * X[:, i].sum()), 8)

    factor_beta_dict = {fnames[i]: round(float(factor_betas_arr[i]), 6) for i in range(k)}

    systematic = sum(factor_contribs.values())
    alpha_total = float(y.sum()) - systematic
    total = float(y.sum())

    # Tracking error
    te = float(resid.std() * np.sqrt(252))

    # Information ratio
    ir = float(alpha_daily * 252 / te) if te > 1e-10 else 0.0

    # t-statistics
    se = np.sqrt(np.diag(ss_res / (n - k - 1) * np.linalg.pinv(X_with_const.T @ X_with_const)))
    t_stats = {}
    for i, fname in enumerate(fnames):
        t_stats[fname] = round(float(betas[i + 1] / (se[i + 1] + 1e-10)), 3)

    return FactorDecomposition(
        factor_returns=factor_contribs,
        factor_betas=factor_beta_dict,
        systematic_return=round(systematic, 8),
        alpha_return=round(alpha_total, 8),
        total_return=round(total, 8),
        r_squared=round(r2, 4),
        tracking_error=round(te, 6),
        information_ratio=round(ir, 4),
        factor_t_stats=t_stats,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Risk Attribution (Marginal Risk Contribution by Factor)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class RiskAttribution:
    """Per-factor risk contribution."""
    factor_risk_contribution: Dict[str, float]   # {factor: % of portfolio vol}
    systematic_risk_pct: float                    # % from factor exposure
    idiosyncratic_risk_pct: float                 # % from residual
    diversification_benefit: float                # 1 - (undiv_vol / div_vol)


def attribute_risk(
    portfolio_returns: pd.Series,
    factor_returns: pd.DataFrame,
) -> RiskAttribution:
    """
    Decompose portfolio risk into factor and idiosyncratic components.

    Uses variance decomposition: Var(r_p) = β'Σ_F β + σ²_ε
    """
    aligned = pd.DataFrame({"port": portfolio_returns}).join(factor_returns, how="inner").dropna()
    if len(aligned) < 30:
        return RiskAttribution(
            factor_risk_contribution={}, systematic_risk_pct=0,
            idiosyncratic_risk_pct=1, diversification_benefit=0,
        )

    y = aligned["port"].values
    X = aligned.drop(columns=["port"]).values
    fnames = [c for c in aligned.columns if c != "port"]

    # OLS
    X_c = np.column_stack([np.ones(len(X)), X])
    betas = np.linalg.lstsq(X_c, y, rcond=None)[0]
    factor_betas = betas[1:]
    resid = y - X_c @ betas

    # Factor covariance
    Sigma_F = np.cov(X.T) if X.shape[1] > 1 else np.atleast_2d(X.var())

    # Portfolio variance decomposition
    total_var = float(y.var())
    systematic_var = float(factor_betas @ Sigma_F @ factor_betas)
    idio_var = float(resid.var())

    sys_pct = systematic_var / (total_var + 1e-15)
    idio_pct = 1 - sys_pct

    # Per-factor contribution
    factor_contrib = {}
    for i, fname in enumerate(fnames):
        # Marginal contribution: β_i² × σ²_i / total_var
        fc = factor_betas[i] ** 2 * float(Sigma_F[i, i] if Sigma_F.ndim > 1 else Sigma_F) / (total_var + 1e-15)
        factor_contrib[fname] = round(fc, 6)

    # Diversification benefit
    undiv_vol = sum(abs(factor_betas[i]) * float(np.sqrt(Sigma_F[i, i] if Sigma_F.ndim > 1 else Sigma_F))
                     for i in range(len(fnames)))
    div_vol = np.sqrt(systematic_var)
    div_benefit = 1 - div_vol / (undiv_vol + 1e-10) if undiv_vol > 0 else 0

    return RiskAttribution(
        factor_risk_contribution=factor_contrib,
        systematic_risk_pct=round(sys_pct, 4),
        idiosyncratic_risk_pct=round(idio_pct, 4),
        diversification_benefit=round(div_benefit, 4),
    )