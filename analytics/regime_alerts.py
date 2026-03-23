"""
analytics/regime_alerts.py

Regime Transition Alerts for the SRV Quantamental DSS.

Monitors market regime state and generates alerts when:
  1. Regime changes (e.g., CALM → TENSION)
  2. Transition probability rises above warning/danger thresholds
  3. Component scores approach regime boundaries (pre-transition warning)

Also builds a historical regime timeline for dashboard visualisation.

Uses the same classification logic as QuantEngine / WalkForwardBacktester.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Alert severity ───────────────────────────────────────────────────────

class AlertLevel(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


# ── Result containers ────────────────────────────────────────────────────

@dataclass
class RegimeSnapshot:
    """Single-date regime state with all component scores."""
    date: pd.Timestamp
    regime: str                      # CALM / NORMAL / TENSION / CRISIS
    vol_score: float                 # 0-1 VIX component
    credit_score: float              # 0-1 credit spread component
    corr_score: float                # 0-1 correlation composite
    transition_score: float          # 0-1 transition risk
    crisis_probability: float        # 0-1 crisis probability
    vix_level: float
    avg_corr: float
    market_mode_strength: float
    distortion: float


@dataclass
class RegimeTransition:
    """Detected regime change event."""
    date: pd.Timestamp
    from_regime: str
    to_regime: str
    level: AlertLevel
    # Component scores at transition
    vol_score: float
    credit_score: float
    corr_score: float
    transition_score: float
    crisis_probability: float
    message: str                     # Human-readable alert text (Hebrew)


@dataclass
class PreTransitionWarning:
    """Warning when component scores approach regime boundaries."""
    date: pd.Timestamp
    current_regime: str
    approaching_regime: str
    level: AlertLevel
    trigger_component: str           # Which component is near boundary
    current_value: float
    threshold_value: float
    distance_pct: float              # How close (0% = at boundary)
    message: str


@dataclass
class RegimeTimelineEntry:
    """Single entry in the regime timeline."""
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    regime: str
    duration_days: int
    avg_crisis_prob: float
    avg_vix: float


@dataclass
class RegimeAlertResult:
    """Complete regime alert analysis output."""

    # Current state
    current_regime: str
    current_snapshot: RegimeSnapshot

    # Historical transitions
    transitions: List[RegimeTransition]

    # Active pre-transition warnings
    warnings: List[PreTransitionWarning]

    # Full regime timeline
    timeline: List[RegimeTimelineEntry]

    # Full snapshot history (for charts)
    snapshot_history: pd.DataFrame

    # Summary statistics
    regime_stats: Dict[str, Dict[str, float]]  # {regime: {count, avg_duration, pct_time}}

    # Active alerts (transitions + warnings from last N days)
    active_alerts: List[Dict[str, Any]]


# ── Component score calculators (mirrors backtest._classify_regime) ──────

def _clip(x: float, lo: float, hi: float) -> float:
    if not math.isfinite(x):
        return lo
    return max(lo, min(hi, float(x)))


def _upper_triangle_mean(C: pd.DataFrame) -> float:
    if C is None or C.empty:
        return float("nan")
    a = C.values.astype(float)
    n = a.shape[0]
    if n <= 1:
        return float("nan")
    iu = np.triu_indices(n, k=1)
    v = a[iu]
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else float("nan")


def _fro_offdiag(C: pd.DataFrame) -> float:
    if C is None or C.empty:
        return float("nan")
    a = C.values.astype(float).copy()
    np.fill_diagonal(a, 0.0)
    if not np.isfinite(a).any():
        return float("nan")
    return float(np.linalg.norm(a, ord="fro"))


# ── Main engine ──────────────────────────────────────────────────────────

class RegimeAlertEngine:
    """
    Monitors regime transitions and generates alerts.

    Pre-transition warnings fire when any component score is within
    WARNING_DISTANCE (20%) of the next regime boundary.
    """

    WARNING_DISTANCE: float = 0.20   # 20% of boundary → warning
    ALERT_LOOKBACK_DAYS: int = 5     # Active alerts window

    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self._log = logging.getLogger(self.__class__.__name__)

    def analyse(self, prices_df: pd.DataFrame) -> RegimeAlertResult:
        """
        Run full regime analysis on price history.

        Parameters
        ----------
        prices_df : pd.DataFrame
            Wide daily price panel (DatetimeIndex × tickers).

        Returns
        -------
        RegimeAlertResult
        """
        prices = prices_df.copy()
        prices.index = pd.to_datetime(prices.index)
        prices = prices.sort_index()
        prices = prices.apply(pd.to_numeric, errors="coerce").ffill(limit=5)

        s = self.settings
        sectors = s.sector_list()
        log_px = np.log(prices.astype(float))
        returns = log_px.diff()

        corr_w = max(20, s.corr_window)
        base_w = max(corr_w, s.corr_baseline_window)
        macro_w = s.macro_window

        n = len(prices)
        start_idx = max(base_w, 252)  # Need enough history

        if n < start_idx + 1:
            raise ValueError(f"Need at least {start_idx + 1} rows, got {n}")

        # ── Compute regime snapshot for each date ────────────────────────
        snapshots: List[RegimeSnapshot] = []

        # Sample every day for the last 252 trading days, weekly before that
        sample_indices = []
        recent_start = max(start_idx, n - 252)
        # Weekly samples for older history
        for i in range(start_idx, recent_start, 5):
            sample_indices.append(i)
        # Daily for recent 252 days
        for i in range(recent_start, n):
            sample_indices.append(i)

        for t in sample_indices:
            snap = self._compute_snapshot(
                prices=prices.iloc[:t + 1],
                returns=returns.iloc[:t + 1],
                sectors=sectors,
                date=prices.index[t],
            )
            if snap is not None:
                snapshots.append(snap)

        if not snapshots:
            raise RuntimeError("No valid regime snapshots produced.")

        # ── Detect transitions ───────────────────────────────────────────
        transitions: List[RegimeTransition] = []
        for i in range(1, len(snapshots)):
            prev = snapshots[i - 1]
            curr = snapshots[i]
            if prev.regime != curr.regime:
                level = self._transition_severity(prev.regime, curr.regime)
                msg = self._transition_message(prev.regime, curr.regime, curr)
                transitions.append(RegimeTransition(
                    date=curr.date,
                    from_regime=prev.regime,
                    to_regime=curr.regime,
                    level=level,
                    vol_score=curr.vol_score,
                    credit_score=curr.credit_score,
                    corr_score=curr.corr_score,
                    transition_score=curr.transition_score,
                    crisis_probability=curr.crisis_probability,
                    message=msg,
                ))

        # ── Pre-transition warnings (current state) ─────────────────────
        warnings: List[PreTransitionWarning] = []
        if snapshots:
            current = snapshots[-1]
            warnings = self._check_pre_transition(current)

        # ── Build regime timeline ────────────────────────────────────────
        timeline = self._build_timeline(snapshots)

        # ── Build snapshot history DataFrame ─────────────────────────────
        snap_rows = []
        for snap in snapshots:
            snap_rows.append({
                "date": snap.date,
                "regime": snap.regime,
                "vol_score": round(snap.vol_score, 4),
                "credit_score": round(snap.credit_score, 4),
                "corr_score": round(snap.corr_score, 4),
                "transition_score": round(snap.transition_score, 4),
                "crisis_probability": round(snap.crisis_probability, 4),
                "vix_level": round(snap.vix_level, 2) if math.isfinite(snap.vix_level) else None,
                "avg_corr": round(snap.avg_corr, 4) if math.isfinite(snap.avg_corr) else None,
                "market_mode": round(snap.market_mode_strength, 4) if math.isfinite(snap.market_mode_strength) else None,
                "distortion": round(snap.distortion, 4) if math.isfinite(snap.distortion) else None,
            })
        snapshot_df = pd.DataFrame(snap_rows)
        if not snapshot_df.empty:
            snapshot_df["date"] = pd.to_datetime(snapshot_df["date"])

        # ── Regime statistics ────────────────────────────────────────────
        regime_stats = self._compute_regime_stats(timeline, snapshots)

        # ── Active alerts (last N days) ──────────────────────────────────
        active_alerts = self._collect_active_alerts(transitions, warnings)

        self._log.info(
            "Regime analysis: current=%s, transitions=%d, warnings=%d, timeline_periods=%d",
            snapshots[-1].regime if snapshots else "UNKNOWN",
            len(transitions), len(warnings), len(timeline),
        )

        return RegimeAlertResult(
            current_regime=snapshots[-1].regime if snapshots else "UNKNOWN",
            current_snapshot=snapshots[-1] if snapshots else None,
            transitions=transitions,
            warnings=warnings,
            timeline=timeline,
            snapshot_history=snapshot_df,
            regime_stats=regime_stats,
            active_alerts=active_alerts,
        )

    # ── Snapshot computation ─────────────────────────────────────────────

    def _compute_snapshot(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        sectors: List[str],
        date: pd.Timestamp,
    ) -> Optional[RegimeSnapshot]:
        """Compute regime snapshot for a single date."""
        s = self.settings
        corr_w = max(20, s.corr_window)
        base_w = max(corr_w, s.corr_baseline_window)
        macro_w = s.macro_window

        R = returns[sectors].dropna(how="all")
        n = len(R)
        if n < corr_w:
            return None

        # Correlation structure
        C_t = R.iloc[-corr_w:].corr()
        C_b = R.iloc[-base_w:].corr() if n >= base_w else C_t.copy()
        C_delta = C_t - C_b

        avg_corr_t = _upper_triangle_mean(C_t)
        avg_corr_b = _upper_triangle_mean(C_b)
        avg_corr_delta = (
            (avg_corr_t - avg_corr_b)
            if (math.isfinite(avg_corr_t) and math.isfinite(avg_corr_b))
            else float("nan")
        )

        distortion_t = _fro_offdiag(C_delta)
        distortion_prev = float("nan")
        if n >= corr_w + 1:
            C_prev = R.iloc[-(corr_w + 1):-1].corr()
            distortion_prev = _fro_offdiag(C_prev - C_b)
        delta_distortion = (
            float(distortion_t - distortion_prev)
            if (math.isfinite(distortion_t) and math.isfinite(distortion_prev))
            else float("nan")
        )

        # Market mode
        market_mode_strength = float("nan")
        try:
            if C_t.notna().sum().sum() > 0:
                Ct = C_t.fillna(0.0).values.astype(float)
                Ct = 0.5 * (Ct + Ct.T)
                evals, _ = np.linalg.eigh(Ct)
                market_mode_strength = float(np.max(evals)) / len(sectors)
        except Exception:
            pass

        # VIX
        vix_col = s.vol_tickers.get("VIX", "^VIX")
        vix_level = float("nan")
        vix_percentile = float("nan")
        if vix_col in prices.columns:
            vix_s = prices[vix_col].dropna()
            if vix_s.size:
                vix_level = float(vix_s.iloc[-1])
                pct_win = max(252, s.pca_window)
                if len(vix_s) >= pct_win:
                    vix_percentile = float(
                        np.sum(vix_s.values[-pct_win:] <= vix_level) / pct_win
                    )

        # Credit z
        credit_z = float("nan")
        hyg_col = s.credit_tickers.get("HYG", "HYG")
        ief_col = s.credit_tickers.get("IEF", "IEF")
        if hyg_col in prices.columns and ief_col in prices.columns:
            spread = (np.log(prices[hyg_col]) - np.log(prices[ief_col])).dropna()
            if len(spread) >= macro_w:
                mu_s = float(spread.rolling(macro_w, min_periods=macro_w).mean().iloc[-1])
                sd_s = float(spread.rolling(macro_w, min_periods=macro_w).std(ddof=0).iloc[-1])
                if math.isfinite(mu_s) and math.isfinite(sd_s) and sd_s > 1e-12:
                    credit_z = float((spread.iloc[-1] - mu_s) / sd_s)

        # Component scores
        vol_score = 0.0
        if math.isfinite(vix_level):
            if vix_level >= s.vix_level_hard:
                vol_score = 1.0
            elif vix_level > s.vix_level_soft:
                vol_score = (vix_level - s.vix_level_soft) / max(1e-9, s.vix_level_hard - s.vix_level_soft)
        if math.isfinite(vix_percentile):
            vol_score = max(vol_score, _clip((vix_percentile - 0.70) / 0.25, 0.0, 1.0))

        credit_score = _clip((-credit_z - 0.25) / 1.75, 0.0, 1.0) if math.isfinite(credit_z) else 0.0

        corr_level_score = (
            _clip(
                (avg_corr_t - s.calm_avg_corr_max) / max(1e-9, s.crisis_avg_corr_min - s.calm_avg_corr_max),
                0.0, 1.0,
            )
            if math.isfinite(avg_corr_t) else 0.0
        )
        mode_score = (
            _clip(
                (market_mode_strength - s.calm_mode_strength_max) / max(1e-9, s.crisis_mode_strength_min - s.calm_mode_strength_max),
                0.0, 1.0,
            )
            if math.isfinite(market_mode_strength) else 0.0
        )
        dist_score = (
            _clip(
                (distortion_t - s.tension_corr_dist_min) / max(1e-9, s.crisis_corr_dist_min - s.tension_corr_dist_min),
                0.0, 1.0,
            )
            if math.isfinite(distortion_t) else 0.0
        )
        corr_score = _clip(0.45 * corr_level_score + 0.30 * mode_score + 0.25 * dist_score, 0.0, 1.0)

        avg_corr_delta_score = _clip(abs(avg_corr_delta) / 0.18, 0.0, 1.0) if math.isfinite(avg_corr_delta) else 0.0
        delta_dist_score = (
            _clip(
                (delta_distortion - 0.03) / max(1e-9, s.crisis_delta_corr_dist_min - 0.03),
                0.0, 1.0,
            )
            if math.isfinite(delta_distortion) else 0.0
        )
        transition_score = _clip(
            0.45 * delta_dist_score + 0.25 * avg_corr_delta_score + 0.15 * vol_score + 0.15 * credit_score,
            0.0, 1.0,
        )

        crisis_probability = _clip(
            0.40 * corr_score + 0.25 * vol_score + 0.20 * credit_score + 0.15 * transition_score,
            0.0, 1.0,
        )

        # Classification
        crisis_hits = sum([
            math.isfinite(vix_level) and vix_level >= s.vix_level_hard,
            math.isfinite(avg_corr_t) and avg_corr_t >= s.crisis_avg_corr_min,
            math.isfinite(market_mode_strength) and market_mode_strength >= s.crisis_mode_strength_min,
            math.isfinite(distortion_t) and distortion_t >= s.crisis_corr_dist_min,
            math.isfinite(delta_distortion) and delta_distortion >= s.crisis_delta_corr_dist_min,
            math.isfinite(credit_z) and credit_z <= s.credit_stress_z,
        ])
        tension_hits = sum([
            math.isfinite(vix_level) and vix_level >= s.vix_level_soft,
            math.isfinite(avg_corr_t) and avg_corr_t >= s.tension_avg_corr_min,
            math.isfinite(market_mode_strength) and market_mode_strength >= s.tension_mode_strength_min,
            math.isfinite(distortion_t) and distortion_t >= s.tension_corr_dist_min,
            math.isfinite(delta_distortion) and delta_distortion >= s.tension_delta_corr_dist_min,
        ])

        if crisis_hits >= 3 or crisis_probability >= 0.78:
            regime = "CRISIS"
        elif tension_hits >= 2 or transition_score >= s.transition_score_danger:
            regime = "TENSION"
        elif transition_score >= s.transition_score_caution or corr_score >= 0.40 or vol_score >= 0.35:
            regime = "NORMAL"
        else:
            regime = "CALM"

        return RegimeSnapshot(
            date=date,
            regime=regime,
            vol_score=vol_score,
            credit_score=credit_score,
            corr_score=corr_score,
            transition_score=transition_score,
            crisis_probability=crisis_probability,
            vix_level=vix_level,
            avg_corr=avg_corr_t,
            market_mode_strength=market_mode_strength,
            distortion=distortion_t,
        )

    # ── Transition detection ─────────────────────────────────────────────

    _REGIME_ORDER = {"CALM": 0, "NORMAL": 1, "TENSION": 2, "CRISIS": 3}

    def _transition_severity(self, from_r: str, to_r: str) -> AlertLevel:
        """Determine alert severity based on transition direction."""
        f = self._REGIME_ORDER.get(from_r, 0)
        t = self._REGIME_ORDER.get(to_r, 0)

        if to_r == "CRISIS" or (t - f >= 2):
            return AlertLevel.CRITICAL
        elif t > f:
            return AlertLevel.WARNING
        else:
            return AlertLevel.INFO

    def _transition_message(self, from_r: str, to_r: str, snap: RegimeSnapshot) -> str:
        """Generate Hebrew alert message for regime transition."""
        direction = "↑ הסלמה" if self._REGIME_ORDER.get(to_r, 0) > self._REGIME_ORDER.get(from_r, 0) else "↓ הרגעה"

        regime_heb = {
            "CALM": "שוק רגוע",
            "NORMAL": "שוק רגיל",
            "TENSION": "מתח בשוק",
            "CRISIS": "משבר"
        }

        parts = [
            f"{direction}: {regime_heb.get(from_r, from_r)} → {regime_heb.get(to_r, to_r)}",
        ]

        # Add key drivers
        drivers = []
        if snap.vol_score > 0.5:
            vix_str = f"{snap.vix_level:.1f}" if math.isfinite(snap.vix_level) else "—"
            drivers.append(f"VIX={vix_str}")
        if snap.credit_score > 0.3:
            drivers.append("לחץ אשראי")
        if snap.corr_score > 0.4:
            corr_str = f"{snap.avg_corr:.2f}" if math.isfinite(snap.avg_corr) else "—"
            drivers.append(f"קורלציה={corr_str}")
        if snap.transition_score > 0.4:
            drivers.append(f"מעבר={snap.transition_score:.0%}")

        if drivers:
            parts.append(" | ".join(drivers))

        return " — ".join(parts)

    # ── Pre-transition warnings ──────────────────────────────────────────

    def _check_pre_transition(self, snap: RegimeSnapshot) -> List[PreTransitionWarning]:
        """Check if any component is approaching the next regime boundary."""
        warnings: List[PreTransitionWarning] = []
        s = self.settings

        regime_heb = {
            "CALM": "רגוע", "NORMAL": "רגיל",
            "TENSION": "מתח", "CRISIS": "משבר"
        }

        # Define boundary checks per current regime
        checks: List[Tuple[str, str, float, float, str]] = []
        # (component_name, approaching_regime, current_value, threshold, description_heb)

        if snap.regime == "CALM":
            checks = [
                ("VIX", "NORMAL", snap.vol_score, 0.35,
                 f"VIX ({snap.vix_level:.1f}) מתקרב לרמת NORMAL"),
                ("קורלציה", "NORMAL", snap.corr_score, 0.40,
                 f"קורלציה ({snap.avg_corr:.2f}) מתקרבת לרמת NORMAL" if math.isfinite(snap.avg_corr) else ""),
                ("מעבר", "NORMAL", snap.transition_score, s.transition_score_caution,
                 f"ציון מעבר ({snap.transition_score:.0%}) מתקרב לסף"),
            ]
        elif snap.regime == "NORMAL":
            checks = [
                ("VIX", "TENSION", snap.vol_score, 0.60,
                 f"VIX ({snap.vix_level:.1f}) מתקרב לרמת TENSION"),
                ("מעבר", "TENSION", snap.transition_score, s.transition_score_danger,
                 f"ציון מעבר ({snap.transition_score:.0%}) מתקרב לסף סכנה"),
                ("משבר", "CRISIS", snap.crisis_probability, 0.60,
                 f"הסתברות משבר ({snap.crisis_probability:.0%}) עולה"),
            ]
        elif snap.regime == "TENSION":
            checks = [
                ("משבר", "CRISIS", snap.crisis_probability, 0.78,
                 f"הסתברות משבר ({snap.crisis_probability:.0%}) מתקרבת לסף"),
                ("VIX", "CRISIS", snap.vol_score, 0.85,
                 f"VIX ({snap.vix_level:.1f}) מתקרב לרמת משבר"),
            ]

        for comp_name, approaching, current, threshold, msg in checks:
            if not msg:
                continue
            if current < threshold:
                distance = 1.0 - (current / threshold) if threshold > 0 else 1.0
                if distance <= self.WARNING_DISTANCE:
                    level = AlertLevel.WARNING if distance > 0.05 else AlertLevel.CRITICAL
                    warnings.append(PreTransitionWarning(
                        date=snap.date,
                        current_regime=snap.regime,
                        approaching_regime=approaching,
                        level=level,
                        trigger_component=comp_name,
                        current_value=current,
                        threshold_value=threshold,
                        distance_pct=distance,
                        message=msg,
                    ))

        return warnings

    # ── Timeline builder ─────────────────────────────────────────────────

    def _build_timeline(self, snapshots: List[RegimeSnapshot]) -> List[RegimeTimelineEntry]:
        """Build continuous regime timeline from snapshots."""
        if not snapshots:
            return []

        timeline: List[RegimeTimelineEntry] = []
        current_regime = snapshots[0].regime
        period_start = snapshots[0].date
        crisis_probs: List[float] = [snapshots[0].crisis_probability]
        vix_levels: List[float] = [snapshots[0].vix_level]

        for i in range(1, len(snapshots)):
            snap = snapshots[i]
            if snap.regime != current_regime:
                # Close current period
                end_date = snapshots[i - 1].date
                duration = max(1, (end_date - period_start).days)
                cp = [v for v in crisis_probs if math.isfinite(v)]
                vl = [v for v in vix_levels if math.isfinite(v)]
                timeline.append(RegimeTimelineEntry(
                    start_date=period_start,
                    end_date=end_date,
                    regime=current_regime,
                    duration_days=duration,
                    avg_crisis_prob=float(np.mean(cp)) if cp else float("nan"),
                    avg_vix=float(np.mean(vl)) if vl else float("nan"),
                ))
                # Start new period
                current_regime = snap.regime
                period_start = snap.date
                crisis_probs = []
                vix_levels = []

            crisis_probs.append(snap.crisis_probability)
            vix_levels.append(snap.vix_level)

        # Close last period
        end_date = snapshots[-1].date
        duration = max(1, (end_date - period_start).days)
        cp = [v for v in crisis_probs if math.isfinite(v)]
        vl = [v for v in vix_levels if math.isfinite(v)]
        timeline.append(RegimeTimelineEntry(
            start_date=period_start,
            end_date=end_date,
            regime=current_regime,
            duration_days=duration,
            avg_crisis_prob=float(np.mean(cp)) if cp else float("nan"),
            avg_vix=float(np.mean(vl)) if vl else float("nan"),
        ))

        return timeline

    # ── Statistics ────────────────────────────────────────────────────────

    def _compute_regime_stats(
        self,
        timeline: List[RegimeTimelineEntry],
        snapshots: List[RegimeSnapshot],
    ) -> Dict[str, Dict[str, float]]:
        """Compute summary statistics per regime."""
        stats: Dict[str, Dict[str, float]] = {}
        total_days = sum(t.duration_days for t in timeline) or 1

        for regime in ["CALM", "NORMAL", "TENSION", "CRISIS"]:
            periods = [t for t in timeline if t.regime == regime]
            durations = [t.duration_days for t in periods]
            total_in_regime = sum(durations) if durations else 0

            stats[regime] = {
                "count": len(periods),
                "total_days": total_in_regime,
                "pct_time": total_in_regime / total_days,
                "avg_duration": float(np.mean(durations)) if durations else 0.0,
                "max_duration": max(durations) if durations else 0,
                "min_duration": min(durations) if durations else 0,
            }

        return stats

    # ── Active alerts collector ──────────────────────────────────────────

    def _collect_active_alerts(
        self,
        transitions: List[RegimeTransition],
        warnings: List[PreTransitionWarning],
    ) -> List[Dict[str, Any]]:
        """Collect recent transitions + current warnings into active alerts list."""
        alerts: List[Dict[str, Any]] = []

        # Recent transitions (last 5)
        for t in transitions[-5:]:
            alerts.append({
                "type": "transition",
                "date": str(t.date.date()) if hasattr(t.date, "date") else str(t.date),
                "level": t.level.value,
                "message": t.message,
                "from": t.from_regime,
                "to": t.to_regime,
            })

        # Current warnings
        for w in warnings:
            alerts.append({
                "type": "warning",
                "date": str(w.date.date()) if hasattr(w.date, "date") else str(w.date),
                "level": w.level.value,
                "message": w.message,
                "component": w.trigger_component,
                "distance": f"{w.distance_pct:.0%}",
            })

        return alerts

    # ── Persistence (alerts to JSON) ─────────────────────────────────────

    def save_alerts(self, result: RegimeAlertResult, path: Path) -> None:
        """Save active alerts to JSON for external consumers."""
        data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "current_regime": result.current_regime,
            "crisis_probability": round(result.current_snapshot.crisis_probability, 4)
            if result.current_snapshot else None,
            "alerts": result.active_alerts,
            "regime_stats": result.regime_stats,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        self._log.info("Alerts saved to %s", path)


# ── Module-level convenience ─────────────────────────────────────────────

def run_regime_alerts(
    prices_df: pd.DataFrame,
    settings: Any,
) -> RegimeAlertResult:
    """Run regime transition analysis using default parameters."""
    return RegimeAlertEngine(settings).analyse(prices_df)
