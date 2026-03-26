"""
research_sandbox/regime_safety_candidate.py
============================================
EDITABLE — this is the only file you should modify in Target 1 experiments.

Verbatim copy of analytics/signal_regime_safety.py at sandbox creation time
(2026-03-26). The eval harness imports compute_regime_safety_score() from
here — never from the production file.

To run an experiment:
  1. Change one default parameter value in this file.
  2. Run: python research_sandbox/eval_regime_safety.py
  3. Compare output to baseline in results_regime_safety.tsv.
  4. Keep change only if sharpe improves AND all guardrails pass.
  5. Revert if not.

Parameters in scope (default values as of baseline):
  vix_soft=18.0, vix_hard=30.0, vix_kill=35.0
  w_vix=0.30, w_credit=0.25, w_corr=0.25, w_trans=0.20
  credit_stress_soft=-0.5, credit_stress_hard=-2.0, credit_stress_kill=-3.0
  corr_soft=0.55, corr_hard=0.75, corr_z_kill=2.5
  trans_soft=0.30, trans_hard=0.65, crisis_prob_kill=0.70
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RegimeSafetyResult:
    """Layer 4 output: regime safety scoring."""
    # Market state (from existing RegimeMetrics)
    market_state: str              # CALM / NORMAL / TENSION / CRISIS

    # Sub-component penalties ∈ [0, 1]  (0 = no penalty, 1 = max penalty)
    vix_penalty: float
    credit_penalty: float
    corr_penalty: float
    transition_penalty: float

    # Hard kill triggers
    hard_kills: Dict[str, bool]    # Which kill switches fired
    any_hard_kill: bool            # If True → S^safe = 0.0

    # Combined
    regime_safety_score: float     # S^safe ∈ [0, 1]
    size_cap: float                # Max sizing multiplier given regime

    # Explanation
    label: str                     # "SAFE" / "CAUTION" / "DANGER" / "KILLED"
    rationale: str
    alerts: list = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Sub-component (a): VIX regime penalty
# ─────────────────────────────────────────────────────────────────────────────

def _vix_penalty(
    vix_level: float,
    vix_soft: float = 18.0,
    vix_hard: float = 30.0,
    vix_kill: float = 35.0,
) -> tuple:
    """
    VIX-based penalty for short-vol trades.
      VIX < soft → 0.0 (no penalty)
      soft < VIX < hard → linear ramp
      VIX >= hard → ~1.0 (max penalty)
      VIX >= kill → hard kill trigger

    Returns (penalty ∈ [0,1], hard_kill: bool)
    """
    if not math.isfinite(vix_level):
        return 0.2, False  # Unknown → mild penalty

    if vix_level >= vix_kill:
        return 1.0, True
    if vix_level >= vix_hard:
        return min(1.0, 0.85 + 0.15 * (vix_level - vix_hard) / max(1, vix_kill - vix_hard)), False
    if vix_level >= vix_soft:
        return (vix_level - vix_soft) / max(1e-9, vix_hard - vix_soft) * 0.85, False
    return 0.0, False


# ─────────────────────────────────────────────────────────────────────────────
# Sub-component (b): Credit stress penalty
# ─────────────────────────────────────────────────────────────────────────────

def _credit_penalty(
    credit_z: float,
    stress_soft: float = -0.5,
    stress_hard: float = -2.0,
    stress_kill: float = -3.0,
) -> tuple:
    """
    Credit spread z-score penalty.
    Note: credit_z is typically negative when spreads widen (stress).
      z > soft → 0.0 (no stress)
      soft > z > hard → linear ramp
      z <= kill → hard kill

    Returns (penalty ∈ [0,1], hard_kill: bool)
    """
    if not math.isfinite(credit_z):
        return 0.15, False

    if credit_z <= stress_kill:
        return 1.0, True
    if credit_z <= stress_hard:
        return min(1.0, 0.80 + 0.20 * (stress_hard - credit_z) / max(1e-9, stress_hard - stress_kill)), False
    if credit_z <= stress_soft:
        return (stress_soft - credit_z) / max(1e-9, stress_soft - stress_hard) * 0.80, False
    return 0.0, False


# ─────────────────────────────────────────────────────────────────────────────
# Sub-component (c): Correlation spike penalty
# ─────────────────────────────────────────────────────────────────────────────

def _corr_penalty(
    avg_corr: float,
    corr_z: float,
    corr_soft: float = 0.55,
    corr_hard: float = 0.75,
    corr_z_kill: float = 2.5,
) -> tuple:
    """
    Correlation-based penalty for dispersion trades.
    When correlations spike, dispersion collapses → short-vol loses.
      avg_corr < soft → 0.0
      soft < avg_corr < hard → ramp
      corr_z >= kill → hard kill (sudden spike)

    Returns (penalty ∈ [0,1], hard_kill: bool)
    """
    kill = False
    penalty = 0.0

    if math.isfinite(corr_z) and corr_z >= corr_z_kill:
        kill = True
        penalty = 1.0
    elif math.isfinite(avg_corr):
        if avg_corr >= corr_hard:
            penalty = min(1.0, 0.80 + 0.20 * (avg_corr - corr_hard) / 0.15)
        elif avg_corr >= corr_soft:
            penalty = (avg_corr - corr_soft) / max(1e-9, corr_hard - corr_soft) * 0.80

    return penalty, kill


# ─────────────────────────────────────────────────────────────────────────────
# Sub-component (d): Regime transition risk
# ─────────────────────────────────────────────────────────────────────────────

def _transition_penalty(
    transition_score: float,
    crisis_probability: float,
    trans_soft: float = 0.30,
    trans_hard: float = 0.65,
    crisis_prob_kill: float = 0.70,
) -> tuple:
    """
    Penalize fast regime deterioration.
      transition_score < soft → 0.0
      soft < score < hard → ramp
      crisis_probability >= kill → hard kill

    Returns (penalty ∈ [0,1], hard_kill: bool)
    """
    kill = False
    penalty = 0.0

    if math.isfinite(crisis_probability) and crisis_probability >= crisis_prob_kill:
        kill = True
        penalty = 1.0
    elif math.isfinite(transition_score):
        if transition_score >= trans_hard:
            penalty = min(1.0, 0.75 + 0.25 * (transition_score - trans_hard) / (1.0 - trans_hard))
        elif transition_score >= trans_soft:
            penalty = (transition_score - trans_soft) / max(1e-9, trans_hard - trans_soft) * 0.75

    return penalty, kill


# ─────────────────────────────────────────────────────────────────────────────
# Combined Layer 4 score
# ─────────────────────────────────────────────────────────────────────────────

def compute_regime_safety_score(
    market_state: str,
    vix_level: float = float("nan"),
    credit_z: float = float("nan"),
    avg_corr: float = float("nan"),
    corr_z: float = float("nan"),
    transition_score: float = float("nan"),
    crisis_probability: float = float("nan"),
    *,
    # Weights (how much each penalty affects the score)
    w_vix: float = 0.30,
    w_credit: float = 0.25,
    w_corr: float = 0.25,
    w_trans: float = 0.20,
    # VIX thresholds
    vix_soft: float = 18.0,
    vix_hard: float = 30.0,
    vix_kill: float = 35.0,
    # Credit thresholds
    credit_stress_soft: float = -0.5,
    credit_stress_hard: float = -2.0,
    credit_stress_kill: float = -3.0,
    # Correlation thresholds
    corr_soft: float = 0.55,
    corr_hard: float = 0.75,
    corr_z_kill: float = 2.5,
    # Transition thresholds
    trans_soft: float = 0.30,
    trans_hard: float = 0.65,
    crisis_prob_kill: float = 0.70,
) -> RegimeSafetyResult:
    """
    Layer 4: Regime Safety Score.

    S^safe = (1 - w_vix·P_vix) × (1 - w_credit·P_credit)
             × (1 - w_corr·P_corr) × (1 - w_trans·P_trans)

    Hard kill: if ANY kill trigger fires → S^safe = 0.0

    Parameters
    ----------
    market_state        : From RegimeMetrics (CALM/NORMAL/TENSION/CRISIS)
    vix_level           : Current VIX level
    credit_z            : Credit spread z-score
    avg_corr            : Average pairwise correlation
    corr_z              : Correlation z-score (rate of change)
    transition_score    : From RegimeMetrics
    crisis_probability  : From RegimeMetrics

    Returns
    -------
    RegimeSafetyResult
    """
    # Sub-components
    p_vix, kill_vix = _vix_penalty(vix_level, vix_soft, vix_hard, vix_kill)
    p_credit, kill_credit = _credit_penalty(credit_z, credit_stress_soft, credit_stress_hard, credit_stress_kill)
    p_corr, kill_corr = _corr_penalty(avg_corr, corr_z, corr_soft, corr_hard, corr_z_kill)
    p_trans, kill_trans = _transition_penalty(transition_score, crisis_probability, trans_soft, trans_hard, crisis_prob_kill)

    hard_kills = {
        "vix_extreme": kill_vix,
        "credit_blowout": kill_credit,
        "corr_spike": kill_corr,
        "crisis_imminent": kill_trans,
    }
    any_hard_kill = any(hard_kills.values())

    # Also: CRISIS regime from QuantEngine is always a hard kill
    state = str(market_state).upper()
    if state == "CRISIS":
        hard_kills["regime_crisis"] = True
        any_hard_kill = True

    if any_hard_kill:
        score = 0.0
    else:
        # Multiplicative combination: each penalty reduces the score
        score = (
            (1.0 - w_vix * p_vix)
            * (1.0 - w_credit * p_credit)
            * (1.0 - w_corr * p_corr)
            * (1.0 - w_trans * p_trans)
        )
        score = max(0.0, min(1.0, score))

    # Size cap: regime-dependent maximum position size
    size_caps = {
        "CALM": 1.0,
        "NORMAL": 0.75,
        "TENSION": 0.40,
        "CRISIS": 0.0,
    }
    size_cap = size_caps.get(state, 0.50)
    # Further reduce by safety score
    size_cap = size_cap * score

    # Build alerts
    alerts = []
    if kill_vix:
        alerts.append(f"HARD KILL: VIX={vix_level:.1f} >= {vix_kill}")
    elif p_vix >= 0.5:
        alerts.append(f"VIX elevated: {vix_level:.1f}")
    if kill_credit:
        alerts.append(f"HARD KILL: Credit z={credit_z:.2f} <= {credit_stress_kill}")
    elif p_credit >= 0.5:
        alerts.append(f"Credit stress: z={credit_z:.2f}")
    if kill_corr:
        alerts.append(f"HARD KILL: Corr spike z={corr_z:.2f} >= {corr_z_kill}")
    elif p_corr >= 0.5:
        alerts.append(f"Correlations elevated: avg={avg_corr:.2f}")
    if kill_trans:
        alerts.append(f"HARD KILL: Crisis prob={crisis_probability:.0%}")
    elif p_trans >= 0.5:
        alerts.append(f"Regime deteriorating: trans={transition_score:.2f}")
    if state == "CRISIS" and not any(k for k, v in hard_kills.items() if v and k != "regime_crisis"):
        alerts.append("HARD KILL: Regime classified as CRISIS")

    # Labels — consider both score AND market state for accurate labeling
    if any_hard_kill:
        label = "KILLED"
        rationale = f"Trade BLOCKED — hard kill active: {', '.join(a for a in alerts if 'HARD KILL' in a)}"
    elif state in ("TENSION", "CRISIS") or score < 0.85:
        if score >= 0.7:
            label = "CAUTION"
            rationale = f"Regime elevated ({state}), proceed with reduced sizing: S^safe={score:.2f}"
        elif score >= 0.4:
            label = "CAUTION"
            rationale = f"Proceed with reduced size: state={state}, S^safe={score:.2f}"
        else:
            label = "DANGER"
            rationale = f"Elevated risk environment: state={state}, S^safe={score:.2f} — minimal sizing"
    elif score >= 0.85:
        label = "SAFE"
        rationale = f"Regime favorable for short-vol: state={state}, S^safe={score:.2f}"
    elif score >= 0.4:
        label = "CAUTION"
        rationale = f"Proceed with reduced size: state={state}, S^safe={score:.2f}"
    else:
        label = "DANGER"
        rationale = f"Elevated risk environment: state={state}, S^safe={score:.2f} — minimal sizing"

    return RegimeSafetyResult(
        market_state=state,
        vix_penalty=round(p_vix, 4),
        credit_penalty=round(p_credit, 4),
        corr_penalty=round(p_corr, 4),
        transition_penalty=round(p_trans, 4),
        hard_kills=hard_kills,
        any_hard_kill=any_hard_kill,
        regime_safety_score=round(score, 4),
        size_cap=round(size_cap, 4),
        label=label,
        rationale=rationale,
        alerts=alerts,
    )


def compute_regime_safety_from_metrics(
    regime_metrics,
    vix_level: float = float("nan"),
    avg_corr: float = float("nan"),
    corr_z: float = float("nan"),
    **kwargs,
) -> RegimeSafetyResult:
    """
    Convenience: compute S^safe directly from QuantEngine's RegimeMetrics object.
    """
    return compute_regime_safety_score(
        market_state=getattr(regime_metrics, "market_state", "NORMAL"),
        vix_level=vix_level,
        credit_z=getattr(regime_metrics, "regime_credit_score", float("nan")),
        avg_corr=avg_corr,
        corr_z=corr_z,
        transition_score=getattr(regime_metrics, "regime_transition_score", float("nan")),
        crisis_probability=getattr(regime_metrics, "crisis_probability", float("nan")),
        **kwargs,
    )
