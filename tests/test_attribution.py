"""tests/test_attribution.py — compute_attribution_row synthetic inputs, ranges, labels."""
from __future__ import annotations

import math
from typing import Any, Dict

import pytest

from analytics.attribution import (
    AttributionResult,
    compute_attribution_row,
    compute_mispricing_confidence,
    compute_statistical_dislocation_score,
)

VIX_SOFT = 25.0
VIX_HARD = 35.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _row(**kwargs) -> Dict[str, Any]:
    """Return a minimal row dict with sensible defaults, overridable via kwargs."""
    base = {
        "pca_residual_z":          -2.5,
        "market_dispersion_z":      1.0,
        "half_life_days_est":       30.0,
        "direction":               "LONG",
        "rel_pe_vs_spy":            1.20,
        "rel_earnings_yield_vs_spy": 0.85,
        "fund_covered_weight":      0.70,
        "neg_or_missing_earnings_weight": 0.10,
        "fund_source":             "HOLDINGS",
        "beta_tnx_60d":             0.05,
        "beta_dxy_60d":            -0.02,
        "beta_spy_delta":           0.10,
        "corr_to_spy_delta":        0.05,
        "credit_z":                 0.0,
        "vix_level":               18.0,
        "delta_corr_dist":          0.10,
        "trend_ratio_slope_63d":   -0.02,
        "trend_ratio_slope_126d":  -0.01,
    }
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# Type & structure tests
# ---------------------------------------------------------------------------
def test_returns_attribution_result_instance():
    result = compute_attribution_row(_row(), vix_soft=VIX_SOFT, vix_hard=VIX_HARD)
    assert isinstance(result, AttributionResult)


def test_scores_are_finite_floats():
    result = compute_attribution_row(_row(), vix_soft=VIX_SOFT, vix_hard=VIX_HARD)
    for score in (result.sds, result.fjs, result.mss, result.stf, result.mc):
        assert math.isfinite(score), f"Score not finite: {score}"


def test_all_scores_in_0_1_range():
    result = compute_attribution_row(_row(), vix_soft=VIX_SOFT, vix_hard=VIX_HARD)
    for score in (result.sds, result.fjs, result.mss, result.stf, result.mc):
        assert 0.0 <= score <= 1.0, f"Score out of range: {score}"


def test_explanation_tags_not_empty():
    result = compute_attribution_row(_row(), vix_soft=VIX_SOFT, vix_hard=VIX_HARD)
    assert isinstance(result.explanation_tags, list)
    assert len(result.explanation_tags) >= 1


# ---------------------------------------------------------------------------
# Label validity
# ---------------------------------------------------------------------------
_VALID_ACTION_BIASES = {"LEAN_IN", "SELECTIVE", "SMALL_SIZE_ONLY", "AVOID_OR_REDUCE"}

def test_action_bias_is_valid_enum():
    result = compute_attribution_row(_row(), vix_soft=VIX_SOFT, vix_hard=VIX_HARD)
    assert result.action_bias in _VALID_ACTION_BIASES


def test_labels_are_non_empty_strings():
    result = compute_attribution_row(_row(), vix_soft=VIX_SOFT, vix_hard=VIX_HARD)
    for label in (
        result.dislocation_label, result.fundamental_label, result.macro_label,
        result.structural_label, result.mc_label, result.risk_label,
        result.interpretation,
    ):
        assert isinstance(label, str) and len(label) > 0


# ---------------------------------------------------------------------------
# Business logic
# ---------------------------------------------------------------------------
def test_neutral_direction_gives_zero_fjs():
    result = compute_attribution_row(_row(direction="NEUTRAL"), vix_soft=VIX_SOFT, vix_hard=VIX_HARD)
    assert result.fjs == pytest.approx(0.0)


def test_high_vix_increases_mss():
    low_vix = compute_attribution_row(_row(vix_level=15.0), vix_soft=VIX_SOFT, vix_hard=VIX_HARD)
    high_vix = compute_attribution_row(_row(vix_level=40.0), vix_soft=VIX_SOFT, vix_hard=VIX_HARD)
    assert high_vix.mss > low_vix.mss


def test_lean_in_conditions():
    """High MC, low MSS, low STF → LEAN_IN."""
    mc = compute_mispricing_confidence(sds=0.90, fjs=0.05, mss=0.10, stf=0.10)
    # mc should be high enough to trigger LEAN_IN
    from analytics.attribution import _action_bias
    bias = _action_bias(mc, mss=0.10, stf=0.10)
    assert bias == "LEAN_IN"


def test_all_nan_inputs_returns_valid_result():
    """NaN inputs must not raise; scores should degrade gracefully to 0.0."""
    row = {k: float("nan") for k in [
        "pca_residual_z", "market_dispersion_z", "half_life_days_est",
        "rel_pe_vs_spy", "rel_earnings_yield_vs_spy", "fund_covered_weight",
        "neg_or_missing_earnings_weight", "beta_tnx_60d", "beta_dxy_60d",
        "beta_spy_delta", "corr_to_spy_delta", "credit_z", "vix_level",
        "delta_corr_dist", "trend_ratio_slope_63d", "trend_ratio_slope_126d",
    ]}
    row["direction"] = "LONG"
    row["fund_source"] = "UNKNOWN"
    result = compute_attribution_row(row, vix_soft=VIX_SOFT, vix_hard=VIX_HARD)
    for score in (result.sds, result.fjs, result.mss, result.stf, result.mc):
        assert 0.0 <= score <= 1.0


def test_sds_zero_for_small_z():
    """Residual z near zero → SDS close to 0."""
    sds = compute_statistical_dislocation_score(z=0.1, disp_z=0.0, half_life=30.0)
    assert sds < 0.10


def test_sds_near_one_for_large_z():
    """Very large residual z → SDS approaches 1."""
    sds = compute_statistical_dislocation_score(z=5.0, disp_z=2.0, half_life=35.0)
    assert sds > 0.60
