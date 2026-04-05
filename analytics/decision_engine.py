"""
analytics/decision_engine.py
================================
Decision and policy logic extracted from stat_arb.py QuantEngine.

Separates the "what to trade" (alpha) from "how to present it" (policy/UI).

stat_arb.py produces raw scores (z-score, MC, conviction).
This module transforms scores into PM-facing decisions:
  - decision_score, decision_label, size_bucket, entry_quality, risk_override
  - pm_note (human-readable explanation)

This separation matters because:
  1. Alpha logic (stat_arb) should be testable without PM formatting
  2. Decision labels can change without touching math
  3. Different execution contexts (paper, live, research) may need different policies
  4. Decision thresholds can be A/B tested independently

All thresholds are documented with their mathematical justification.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class Decision:
    """Immutable trading decision for one sector."""
    decision_score: float              # 0-1 composite tradability score
    decision_label: str                # ENTER / WATCH / REDUCE / AVOID
    size_bucket: str                   # FULL / MEDIUM / SMALL / ZERO
    entry_quality: str                 # HIGH_QUALITY / MEDIUM_QUALITY / SPECULATIVE / POOR_QUALITY
    risk_override: str                 # NONE / CRISIS_OVERRIDE / MACRO_OVERRIDE / etc.
    pm_note: str                       # Human-readable explanation


def compute_decision(
    direction: str,
    conviction_score: float,
    mc_score_raw: float,
    market_state: str,
    regime_transition_score: float,
    crisis_probability: float,
    mss_score: float,
    stf_score: float,
    interpretation: str,
    action_bias: str,
    risk_label: str,
) -> Decision:
    """
    Convert raw model scores into a PM-facing trading decision.

    Decision Score Construction:
      d = 0.45 × MC + 0.35 × conviction_norm
          - regime_penalty
          - 0.15 × transition_score
          - 0.20 × crisis_prob
          - 0.20 × macro_score
          - 0.20 × structural_score

    The weights reflect:
      - MC (mispricing confidence) gets most weight because it's the
        composite quality score
      - Conviction is secondary because it includes MC as a component
      - Penalties are additive (not multiplicative) to avoid pushing
        everything to zero

    Decision Label Logic:
      ENTER:  MC ≥ 0.55, conviction ≥ 0.55, macro < 0.45, structural < 0.45
      WATCH:  MC ≥ 0.30, conviction ≥ 0.40 (interesting but not clean)
      REDUCE: Signal exists but regime/risk suggests smaller size
      AVOID:  Crisis, low MC, or strong macro/structural override
    """
    direction = str(direction or "NEUTRAL").upper().strip()
    market_state = str(market_state or "NORMAL").upper().strip()

    if direction not in {"LONG", "SHORT"}:
        return Decision(
            decision_score=0.0, decision_label="AVOID", size_bucket="ZERO",
            entry_quality="NO_SIGNAL", risk_override="NO_DIRECTION",
            pm_note="No directional signal: residual dislocation not strong enough.",
        )

    # Normalize inputs
    conv_norm = max(0, min(1, float(conviction_score) / 100.0)) if math.isfinite(conviction_score) else 0
    mc_norm = max(0, min(1, float(mc_score_raw))) if math.isfinite(mc_score_raw) else 0

    # Regime penalty (stepped, not proportional — avoids small-number multiplication)
    regime_penalty = {"TENSION": 0.18, "CRISIS": 0.45}.get(market_state, 0.0)

    # Component penalties (capped at 0.20 each)
    _safe = lambda x: max(0, min(1, float(x))) if math.isfinite(x) else 0
    transition_pen = 0.15 * _safe(regime_transition_score)
    crisis_pen = 0.20 * _safe(crisis_probability)
    macro_pen = 0.20 * _safe(mss_score)
    structural_pen = 0.20 * _safe(stf_score)

    # Decision score
    decision_score = max(0, min(1,
        0.45 * mc_norm + 0.35 * conv_norm
        - regime_penalty - transition_pen - crisis_pen - macro_pen - structural_pen
    ))

    # ── Decision label ────────────────────────────────────────────
    if market_state == "CRISIS" or _safe(crisis_probability) >= 0.75:
        label = "AVOID"
    elif mc_norm < 0.15:
        label = "AVOID"
    elif _safe(mss_score) >= 0.75 or _safe(stf_score) >= 0.75:
        label = "AVOID"
    elif market_state == "TENSION":
        if mc_norm >= 0.45 and conv_norm >= 0.50 and _safe(mss_score) < 0.55 and _safe(stf_score) < 0.55:
            label = "REDUCE"
        else:
            label = "AVOID"
    elif mc_norm >= 0.55 and conv_norm >= 0.55 and _safe(mss_score) < 0.45 and _safe(stf_score) < 0.45:
        label = "ENTER"
    elif mc_norm >= 0.30 and conv_norm >= 0.40:
        label = "WATCH"
    else:
        label = "REDUCE"

    # ── Size bucket ───────────────────────────────────────────────
    size_map = {
        "ENTER": "FULL" if decision_score >= 0.65 else "MEDIUM" if decision_score >= 0.50 else "SMALL",
        "WATCH": "SMALL",
        "REDUCE": "SMALL",
    }
    size_bucket = size_map.get(label, "ZERO")

    # ── Entry quality ─────────────────────────────────────────────
    if mc_norm >= 0.65 and _safe(mss_score) < 0.35 and _safe(stf_score) < 0.35:
        entry_quality = "HIGH_QUALITY"
    elif mc_norm >= 0.40 and _safe(mss_score) < 0.55 and _safe(stf_score) < 0.55:
        entry_quality = "MEDIUM_QUALITY"
    elif mc_norm >= 0.20:
        entry_quality = "SPECULATIVE"
    else:
        entry_quality = "POOR_QUALITY"

    # ── Risk override ─────────────────────────────────────────────
    if market_state == "CRISIS":
        risk_override = "CRISIS_OVERRIDE"
    elif _safe(crisis_probability) >= 0.60:
        risk_override = "HIGH_CRISIS_RISK"
    elif _safe(mss_score) >= 0.60:
        risk_override = "MACRO_OVERRIDE"
    elif _safe(stf_score) >= 0.60:
        risk_override = "STRUCTURAL_OVERRIDE"
    elif _safe(regime_transition_score) >= 0.60:
        risk_override = "TRANSITION_OVERRIDE"
    else:
        risk_override = "NONE"

    # ── PM note ───────────────────────────────────────────────────
    note_map = {
        "ENTER": f"{interpretation}. Signal is actionable with {risk_label.lower()} and {action_bias.lower()} bias.",
        "WATCH": f"{interpretation}. Interesting setup, but not yet clean enough for full commitment.",
        "REDUCE": f"{interpretation}. Signal exists, but regime / macro / structural risk suggests smaller size.",
    }
    pm_note = note_map.get(label, f"{interpretation}. Avoid active mean-reversion exposure under current conditions.")

    return Decision(
        decision_score=round(decision_score, 4),
        decision_label=label,
        size_bucket=size_bucket,
        entry_quality=entry_quality,
        risk_override=risk_override,
        pm_note=pm_note,
    )
