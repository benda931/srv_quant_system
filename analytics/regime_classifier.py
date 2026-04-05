"""
analytics/regime_classifier.py
=================================
Market regime classification logic extracted from stat_arb.py QuantEngine.

Implements the multi-indicator regime detection used to gate mean-reversion
signals and adjust position sizing. Extracted for testability and clarity.

Classification uses 6 crisis indicators and 5 tension indicators:
  Crisis: VIX > hard, avg_corr > crisis_min, market_mode > crisis_min,
          distortion > crisis_min, delta_distortion > crisis_min, credit_z < stress
  Tension: VIX > soft, avg_corr > tension_min, market_mode > tension_min,
           distortion > tension_min, delta_distortion > tension_min

Regime: CRISIS if crisis_hits >= 3 or crisis_prob >= 0.78
        TENSION if tension_hits >= 2 or transition_score >= danger
        NORMAL if transition/corr/vol scores moderate
        CALM otherwise

Ref: Hamilton (1989) — Regime-switching models
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RegimeClassification:
    """Immutable regime classification output."""
    market_state: str                 # CALM / NORMAL / TENSION / CRISIS
    state_bias: str                   # MEAN_REVERSION_FRIENDLY / etc.
    regime_alert: str                 # LOW_STRESS / ELEVATED / etc.
    mean_reversion_allowed: bool

    # Component scores (0-1)
    vol_score: float
    corr_score: float
    credit_score: float
    transition_score: float
    crisis_probability: float

    # Crisis/tension hit counts
    crisis_hits: int
    tension_hits: int

    # State color for UI
    state_color: str                  # success / info / warning / danger


def classify_regime(
    avg_corr_t: float,
    avg_corr_delta: float,
    distortion_t: float,
    delta_distortion: float,
    market_mode_strength: float,
    vix_level: float,
    vix_percentile: float,
    credit_z: float,
    settings,
) -> RegimeClassification:
    """
    Multi-indicator market regime classification.

    This is the exact logic from stat_arb.py._compute_regime_metrics(),
    extracted for testability without QuantEngine coupling.

    Parameters
    ----------
    avg_corr_t : float — current 20d average pairwise correlation
    avg_corr_delta : float — change in avg correlation (current - baseline)
    distortion_t : float — Frobenius distortion of correlation matrix
    delta_distortion : float — change in Frobenius distortion
    market_mode_strength : float — first eigenvalue share
    vix_level : float — current VIX
    vix_percentile : float — VIX percentile (0-1) vs 252d history
    credit_z : float — credit spread z-score (negative = widening)
    settings : Settings — for threshold parameters
    """
    s = settings

    # ── Component scores (0-1) ────────────────────────────────────
    # VIX score
    if math.isfinite(vix_level):
        vix_range = max(1.0, s.vix_level_hard - s.vix_level_soft)
        vol_score = max(0, min(1, (vix_level - s.vix_level_soft) / vix_range))
    else:
        vol_score = 0.0

    # Correlation score
    if math.isfinite(avg_corr_t):
        corr_range = max(0.01, s.crisis_avg_corr_min - s.calm_avg_corr_max)
        corr_score = max(0, min(1, (avg_corr_t - s.calm_avg_corr_max) / corr_range))
    else:
        corr_score = 0.0

    # Credit score
    credit_score = max(0, min(1, (-credit_z - 0.5) / 2.5)) if math.isfinite(credit_z) else 0.0

    # Transition score (regime change velocity)
    delta_norm = 0.0
    if math.isfinite(avg_corr_delta):
        delta_norm = max(0, min(1, abs(avg_corr_delta) / 0.15))

    mode_chg = 0.0
    if math.isfinite(market_mode_strength):
        mode_chg = max(0, min(1, (market_mode_strength - 0.30) / 0.35))

    dist_chg = 0.0
    if math.isfinite(delta_distortion):
        dist_chg = max(0, min(1, abs(delta_distortion) / 0.15))

    transition_score = max(0, min(1, 0.40 * delta_norm + 0.35 * mode_chg + 0.25 * dist_chg))

    # Crisis probability (composite)
    crisis_probability = max(0, min(1,
        0.20 * vol_score +
        0.30 * corr_score +
        0.15 * credit_score +
        0.15 * transition_score +
        0.10 * (mode_chg if mode_chg > 0.5 else 0) +
        0.10 * (dist_chg if dist_chg > 0.5 else 0)
    ))

    # ── Crisis / Tension hit counting ─────────────────────────────
    crisis_hits = 0
    if math.isfinite(vix_level) and vix_level >= s.vix_level_hard:
        crisis_hits += 1
    if math.isfinite(avg_corr_t) and avg_corr_t >= s.crisis_avg_corr_min:
        crisis_hits += 1
    if math.isfinite(market_mode_strength) and market_mode_strength >= s.crisis_mode_strength_min:
        crisis_hits += 1
    if math.isfinite(distortion_t) and distortion_t >= s.crisis_corr_dist_min:
        crisis_hits += 1
    if math.isfinite(delta_distortion) and delta_distortion >= s.crisis_delta_corr_dist_min:
        crisis_hits += 1
    if math.isfinite(credit_z) and credit_z <= s.credit_stress_z:
        crisis_hits += 1

    tension_hits = 0
    if math.isfinite(vix_level) and vix_level >= s.vix_level_soft:
        tension_hits += 1
    if math.isfinite(avg_corr_t) and avg_corr_t >= s.tension_avg_corr_min:
        tension_hits += 1
    if math.isfinite(market_mode_strength) and market_mode_strength >= s.tension_mode_strength_min:
        tension_hits += 1
    if math.isfinite(distortion_t) and distortion_t >= s.tension_corr_dist_min:
        tension_hits += 1
    if math.isfinite(delta_distortion) and delta_distortion >= s.tension_delta_corr_dist_min:
        tension_hits += 1

    # ── State classification ──────────────────────────────────────
    if crisis_hits >= 3 or crisis_probability >= 0.78:
        market_state = "CRISIS"
    elif tension_hits >= 2 or transition_score >= s.transition_score_danger:
        market_state = "TENSION"
    elif transition_score >= s.transition_score_caution or corr_score >= 0.40 or vol_score >= 0.35:
        market_state = "NORMAL"
    else:
        market_state = "CALM"

    # State properties
    _state_props = {
        "CALM": ("MEAN_REVERSION_FRIENDLY", True, "LOW_STRESS", "success"),
        "NORMAL": ("MODERATE_CAUTION", True, "ELEVATED", "info"),
        "TENSION": ("DEFENSIVE_STANCE", False, "HIGH_ALERT", "warning"),
        "CRISIS": ("RISK_OFF", False, "REGIME_BREAK_RISK", "danger"),
    }
    bias, mr_allowed, alert, color = _state_props.get(market_state, ("UNKNOWN", False, "UNKNOWN", "secondary"))

    return RegimeClassification(
        market_state=market_state,
        state_bias=bias,
        regime_alert=alert,
        mean_reversion_allowed=mr_allowed,
        vol_score=round(vol_score, 4),
        corr_score=round(corr_score, 4),
        credit_score=round(credit_score, 4),
        transition_score=round(transition_score, 4),
        crisis_probability=round(crisis_probability, 4),
        crisis_hits=crisis_hits,
        tension_hits=tension_hits,
        state_color=color,
    )
