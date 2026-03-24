from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List


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