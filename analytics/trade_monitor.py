"""
analytics/trade_monitor.py
===========================
Trade Monitoring & Exit Management Engine

Monitors active trades and generates actionable exit/adjustment signals:

  1. Z-Score Compression   — residual converging to mean → profit-take
  2. Z-Score Extension     — residual moving further away → stop-loss
  3. Time Decay            — holding period vs half-life → aging penalty
  4. Regime Deterioration  — safety score drops → forced exit
  5. Hedge Stability       — beta/hedge ratio drift → rebalance signal
  6. P&L Tracking          — synthetic P&L from residual movement

Output: TradeHealthReport per active trade + portfolio-level monitor summary.

Ref: OU process monitoring (Ornstein-Uhlenbeck exit optimization)
Ref: Optimal stopping theory for mean-reverting processes
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
# Exit Signal
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExitSignal:
    """A single exit/adjustment signal for a trade."""
    signal_type: str          # "PROFIT_TAKE" / "STOP_LOSS" / "TIME_EXIT" / "REGIME_EXIT" /
                              # "HEDGE_REBAL" / "PARTIAL_EXIT" / "HOLD"
    urgency: str              # "IMMEDIATE" / "END_OF_DAY" / "NEXT_SESSION" / "MONITOR"
    strength: float           # 0-1 how strong the signal is
    reason: str               # Human-readable explanation
    action: str               # Specific action: "CLOSE_ALL" / "REDUCE_50" / "REBALANCE" / "HOLD"

    @property
    def is_exit(self) -> bool:
        return self.signal_type in ("PROFIT_TAKE", "STOP_LOSS", "TIME_EXIT", "REGIME_EXIT")


# ─────────────────────────────────────────────────────────────────────────────
# Trade Health Report
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TradeHealthReport:
    """Complete health assessment for one active trade."""
    trade_id: str
    ticker: str
    trade_type: str
    direction: str

    # Current state
    current_z: float              # Current z-score
    entry_z: float                # Z at entry
    z_compression_pct: float      # How much z has compressed (0-1, 1 = fully reverted)
    z_distance_to_target: float   # |current_z - z_target|
    z_distance_to_stop: float     # |current_z - z_stop|

    # Time analysis
    days_held: int                # Days since entry
    max_holding_days: int         # From exit conditions
    time_decay_pct: float         # days_held / max_holding_days
    half_lives_elapsed: float     # days_held / half_life

    # Regime check
    current_safety_score: float   # Current S^safe
    safety_at_entry: float        # S^safe at entry time
    safety_deterioration: float   # How much safety dropped (0 = same, 1 = collapsed)

    # P&L proxy
    pnl_proxy_pct: float          # Synthetic P&L from residual movement
    pnl_vs_target: float          # P&L progress toward profit target (0-1)
    pnl_vs_stop: float            # P&L progress toward stop loss (0-1)

    # Health score
    health_score: float           # 0-1 composite health (1 = healthy)
    health_label: str             # "HEALTHY" / "AGING" / "AT_RISK" / "CRITICAL"

    # Exit signals
    signals: List[ExitSignal] = field(default_factory=list)

    # Recommended action
    recommended_action: str = "HOLD"
    pm_note: str = ""

    @property
    def primary_signal(self) -> Optional[ExitSignal]:
        """Strongest exit signal, if any."""
        exits = [s for s in self.signals if s.is_exit]
        if exits:
            return max(exits, key=lambda s: s.strength)
        return None

    @property
    def should_exit(self) -> bool:
        ps = self.primary_signal
        return ps is not None and ps.strength >= 0.7


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio Monitor Summary
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PortfolioMonitorSummary:
    """Portfolio-level monitoring summary."""
    n_trades: int
    n_healthy: int
    n_aging: int
    n_at_risk: int
    n_critical: int
    n_exit_signals: int

    avg_health: float
    avg_z_compression: float
    avg_time_decay: float

    total_pnl_proxy: float
    regime_safety_current: float

    trade_reports: List[TradeHealthReport]
    urgent_exits: List[TradeHealthReport]

    pm_summary: str


# ─────────────────────────────────────────────────────────────────────────────
# Trade Monitor Engine
# ─────────────────────────────────────────────────────────────────────────────

class TradeMonitorEngine:
    """
    Monitors active trades and generates exit signals.

    Usage:
        monitor = TradeMonitorEngine(settings)
        summary = monitor.monitor_portfolio(
            active_tickets,
            current_residuals,      # {ticker: current residual z-score}
            current_safety,         # RegimeSafetyResult
            days_held,              # {trade_id: int}
        )
    """

    def __init__(self, settings=None):
        self.settings = settings

        # Exit thresholds
        self.z_compression_exit = getattr(settings, "monitor_z_compression_exit", 0.75) if settings else 0.75
        self.z_extension_stop = getattr(settings, "monitor_z_extension_stop", 1.50) if settings else 1.50
        self.time_decay_warning = getattr(settings, "monitor_time_decay_warning", 0.60) if settings else 0.60
        self.time_decay_exit = getattr(settings, "monitor_time_decay_exit", 0.90) if settings else 0.90
        self.safety_floor = getattr(settings, "monitor_safety_floor", 0.15) if settings else 0.15
        self.hl_exit_multiple = getattr(settings, "monitor_hl_exit_multiple", 2.5) if settings else 2.5

    def assess_trade(
        self,
        ticket,
        current_z: float,
        current_safety_score: float = 1.0,
        current_safety_label: str = "SAFE",
        days_held: int = 0,
        current_pnl_pct: float = float("nan"),
    ) -> TradeHealthReport:
        """
        Assess health of a single trade and generate exit signals.

        Parameters
        ----------
        ticket             : TradeTicket from trade_structure
        current_z          : Current z-score of the residual
        current_safety_score: Current regime safety score
        current_safety_label: Current safety label
        days_held          : Days since entry
        current_pnl_pct    : Actual P&L if known, else estimated from z-movement
        """
        ec = ticket.exit_conditions
        entry_z = ticket.entry_z

        # ── Z-score analysis ─────────────────────────────────
        z_now = current_z if math.isfinite(current_z) else entry_z
        z_entry_abs = abs(entry_z) if math.isfinite(entry_z) else 1.0

        # Compression: how much has z moved toward target
        if z_entry_abs > 0.1:
            z_compression = max(0.0, 1.0 - abs(z_now) / z_entry_abs)
        else:
            z_compression = 0.0

        z_target = ec.z_target if math.isfinite(ec.z_target) else 0.0
        z_stop = ec.z_stop if math.isfinite(ec.z_stop) else entry_z * 2.0
        z_dist_target = abs(z_now - z_target)
        z_dist_stop = abs(z_now - z_stop)

        # ── Time analysis ────────────────────────────────────
        max_days = max(1, ec.max_holding_days)
        time_decay = min(1.0, days_held / max_days)
        hl = ec.half_life_est if math.isfinite(ec.half_life_est) else 30.0
        hl_elapsed = days_held / max(1.0, hl)

        # ── Safety analysis ──────────────────────────────────
        safety_at_entry = ticket.regime_safety_score
        safety_now = current_safety_score
        if safety_at_entry > 0.01:
            safety_deterioration = max(0.0, 1.0 - safety_now / safety_at_entry)
        else:
            safety_deterioration = 1.0

        # ── P&L proxy ────────────────────────────────────────
        if math.isfinite(current_pnl_pct):
            pnl_proxy = current_pnl_pct
        else:
            # Estimate from z-score compression: positive if z compressed in our direction
            if ticket.direction == "LONG":
                pnl_proxy = (entry_z - z_now) * ticket.final_weight * 0.01
            elif ticket.direction == "SHORT":
                pnl_proxy = (z_now - entry_z) * ticket.final_weight * 0.01
            else:
                pnl_proxy = z_compression * ticket.final_weight * 0.01

        pnl_vs_target = max(0.0, min(1.0, pnl_proxy / ec.profit_target_pct)) if ec.profit_target_pct > 0 else 0.0
        pnl_vs_stop = max(0.0, min(1.0, -pnl_proxy / ec.max_loss_pct)) if ec.max_loss_pct > 0 else 0.0

        # ── Generate Exit Signals ─────────────────────────────
        signals: List[ExitSignal] = []

        # 1. Profit take: z compressed enough
        if z_compression >= self.z_compression_exit:
            signals.append(ExitSignal(
                signal_type="PROFIT_TAKE",
                urgency="END_OF_DAY",
                strength=min(1.0, z_compression),
                reason=f"Z compressed {z_compression:.0%}: {entry_z:+.2f} → {z_now:+.2f}",
                action="CLOSE_ALL",
            ))
        elif z_compression >= 0.50:
            signals.append(ExitSignal(
                signal_type="PARTIAL_EXIT",
                urgency="NEXT_SESSION",
                strength=z_compression * 0.7,
                reason=f"Z partially compressed {z_compression:.0%} — consider partial profit",
                action="REDUCE_50",
            ))

        # 2. Stop loss: z extended beyond stop
        if ticket.direction == "LONG" and z_now < z_stop:
            stop_breach = abs(z_now - z_stop) / max(0.5, abs(z_stop))
            signals.append(ExitSignal(
                signal_type="STOP_LOSS",
                urgency="IMMEDIATE",
                strength=min(1.0, 0.8 + stop_breach * 0.2),
                reason=f"STOP: z={z_now:+.2f} breached stop={z_stop:+.2f}",
                action="CLOSE_ALL",
            ))
        elif ticket.direction == "SHORT" and z_now > z_stop:
            stop_breach = abs(z_now - z_stop) / max(0.5, abs(z_stop))
            signals.append(ExitSignal(
                signal_type="STOP_LOSS",
                urgency="IMMEDIATE",
                strength=min(1.0, 0.8 + stop_breach * 0.2),
                reason=f"STOP: z={z_now:+.2f} breached stop={z_stop:+.2f}",
                action="CLOSE_ALL",
            ))
        elif abs(z_now) > abs(entry_z) * self.z_extension_stop:
            signals.append(ExitSignal(
                signal_type="STOP_LOSS",
                urgency="END_OF_DAY",
                strength=0.7,
                reason=f"Z extended {abs(z_now)/z_entry_abs:.0%} beyond entry — adverse move",
                action="CLOSE_ALL",
            ))

        # 3. Time exit
        if time_decay >= self.time_decay_exit:
            signals.append(ExitSignal(
                signal_type="TIME_EXIT",
                urgency="END_OF_DAY",
                strength=min(1.0, time_decay),
                reason=f"Time limit: {days_held}/{max_days} days ({time_decay:.0%})",
                action="CLOSE_ALL",
            ))
        elif time_decay >= self.time_decay_warning:
            signals.append(ExitSignal(
                signal_type="TIME_EXIT",
                urgency="NEXT_SESSION",
                strength=time_decay * 0.6,
                reason=f"Aging: {days_held}/{max_days} days, {hl_elapsed:.1f} half-lives",
                action="REDUCE_50" if z_compression < 0.3 else "HOLD",
            ))

        # 4. Half-life exceeded
        if hl_elapsed >= self.hl_exit_multiple and z_compression < 0.5:
            signals.append(ExitSignal(
                signal_type="TIME_EXIT",
                urgency="END_OF_DAY",
                strength=min(1.0, 0.6 + 0.2 * (hl_elapsed - self.hl_exit_multiple)),
                reason=f"Exceeded {hl_elapsed:.1f} half-lives with only {z_compression:.0%} compression",
                action="CLOSE_ALL",
            ))

        # 5. Regime deterioration
        if safety_now < self.safety_floor:
            signals.append(ExitSignal(
                signal_type="REGIME_EXIT",
                urgency="IMMEDIATE",
                strength=1.0,
                reason=f"Regime safety collapsed: {safety_now:.2f} < floor {self.safety_floor}",
                action="CLOSE_ALL",
            ))
        elif current_safety_label in ("KILLED", "CRISIS"):
            signals.append(ExitSignal(
                signal_type="REGIME_EXIT",
                urgency="IMMEDIATE",
                strength=1.0,
                reason=f"Regime KILL: {current_safety_label}",
                action="CLOSE_ALL",
            ))
        elif safety_deterioration >= 0.40:
            signals.append(ExitSignal(
                signal_type="REGIME_EXIT",
                urgency="END_OF_DAY",
                strength=0.5 + 0.5 * safety_deterioration,
                reason=f"Safety dropped {safety_deterioration:.0%} since entry",
                action="REDUCE_50",
            ))

        # 6. No signal → HOLD
        if not signals:
            signals.append(ExitSignal(
                signal_type="HOLD",
                urgency="MONITOR",
                strength=0.0,
                reason=f"Trade healthy: z={z_now:+.2f}, {z_compression:.0%} compressed, {days_held}d held",
                action="HOLD",
            ))

        # ── Health Score ─────────────────────────────────────
        # Composite: higher = healthier
        z_health = max(0.0, min(1.0, z_compression))  # More compression = healthier
        time_health = max(0.0, 1.0 - time_decay)       # More time left = healthier
        safety_health = max(0.0, safety_now)            # Higher safety = healthier
        pnl_health = max(0.0, min(1.0, 0.5 + pnl_proxy * 20))  # Positive P&L = healthier

        health = 0.30 * z_health + 0.25 * time_health + 0.25 * safety_health + 0.20 * pnl_health
        health = max(0.0, min(1.0, health))

        if health >= 0.65:
            health_label = "HEALTHY"
        elif health >= 0.40:
            health_label = "AGING"
        elif health >= 0.20:
            health_label = "AT_RISK"
        else:
            health_label = "CRITICAL"

        # ── Recommended action ────────────────────────────────
        exit_signals = [s for s in signals if s.is_exit]
        if exit_signals:
            strongest = max(exit_signals, key=lambda s: s.strength)
            if strongest.strength >= 0.7:
                recommended = strongest.action
            else:
                recommended = "REDUCE_50" if strongest.strength >= 0.4 else "HOLD"
        else:
            recommended = "HOLD"

        # PM note
        parts = [f"{ticket.ticker} ({ticket.direction})"]
        parts.append(f"Z: {entry_z:+.2f}→{z_now:+.2f} ({z_compression:.0%} compressed)")
        parts.append(f"Day {days_held}/{max_days}")
        if exit_signals:
            parts.append(f"EXIT: {exit_signals[0].reason}")
        parts.append(f"Health: {health_label} ({health:.0%})")
        pm_note = " | ".join(parts)

        return TradeHealthReport(
            trade_id=ticket.trade_id,
            ticker=ticker if (ticker := ticket.ticker) else "",
            trade_type=ticket.trade_type,
            direction=ticket.direction,
            current_z=round(z_now, 4),
            entry_z=round(entry_z, 4),
            z_compression_pct=round(z_compression, 4),
            z_distance_to_target=round(z_dist_target, 4),
            z_distance_to_stop=round(z_dist_stop, 4),
            days_held=days_held,
            max_holding_days=max_days,
            time_decay_pct=round(time_decay, 4),
            half_lives_elapsed=round(hl_elapsed, 2),
            current_safety_score=round(safety_now, 4),
            safety_at_entry=round(safety_at_entry, 4),
            safety_deterioration=round(safety_deterioration, 4),
            pnl_proxy_pct=round(pnl_proxy, 6),
            pnl_vs_target=round(pnl_vs_target, 4),
            pnl_vs_stop=round(pnl_vs_stop, 4),
            health_score=round(health, 4),
            health_label=health_label,
            signals=signals,
            recommended_action=recommended,
            pm_note=pm_note,
        )

    def monitor_portfolio(
        self,
        active_tickets: list,
        current_zscores: Dict[str, float],
        current_safety_score: float = 1.0,
        current_safety_label: str = "SAFE",
        days_held_map: Optional[Dict[str, int]] = None,
        pnl_map: Optional[Dict[str, float]] = None,
    ) -> PortfolioMonitorSummary:
        """
        Monitor all active trades and produce a portfolio-level summary.

        Parameters
        ----------
        active_tickets     : List[TradeTicket] — currently active trades
        current_zscores    : {ticker: current_z} — latest z-scores
        current_safety_score: Current regime safety score
        current_safety_label: Current safety label
        days_held_map      : {trade_id: days} — how long each trade has been held
        pnl_map            : {trade_id: pnl_pct} — actual P&L if known
        """
        if days_held_map is None:
            days_held_map = {}
        if pnl_map is None:
            pnl_map = {}

        reports = []
        for ticket in active_tickets:
            if not ticket.is_active:
                continue

            # Get current z-score for this trade
            z = current_zscores.get(ticket.ticker, float("nan"))
            if not math.isfinite(z):
                # Try spread name components
                if "-" in ticket.ticker:
                    parts = ticket.ticker.split("-")
                    z_a = current_zscores.get(parts[0], 0)
                    z_b = current_zscores.get(parts[1], 0)
                    z = z_a - z_b  # Approximate spread z
                else:
                    z = ticket.entry_z  # Fallback to entry

            days = days_held_map.get(ticket.trade_id, 0)
            pnl = pnl_map.get(ticket.trade_id, float("nan"))

            report = self.assess_trade(
                ticket,
                current_z=z,
                current_safety_score=current_safety_score,
                current_safety_label=current_safety_label,
                days_held=days,
                current_pnl_pct=pnl,
            )
            reports.append(report)

        # Aggregate
        n = len(reports)
        n_healthy = sum(1 for r in reports if r.health_label == "HEALTHY")
        n_aging = sum(1 for r in reports if r.health_label == "AGING")
        n_at_risk = sum(1 for r in reports if r.health_label == "AT_RISK")
        n_critical = sum(1 for r in reports if r.health_label == "CRITICAL")
        n_exit = sum(1 for r in reports if r.should_exit)

        avg_health = float(np.mean([r.health_score for r in reports])) if reports else 0.0
        avg_z_comp = float(np.mean([r.z_compression_pct for r in reports])) if reports else 0.0
        avg_time = float(np.mean([r.time_decay_pct for r in reports])) if reports else 0.0
        total_pnl = sum(r.pnl_proxy_pct for r in reports)

        urgent = [r for r in reports if r.should_exit]
        urgent.sort(key=lambda r: r.primary_signal.strength if r.primary_signal else 0, reverse=True)

        # PM summary
        parts = [f"{n} trades monitored"]
        if n_critical:
            parts.append(f"🔴 {n_critical} CRITICAL")
        if n_at_risk:
            parts.append(f"🟠 {n_at_risk} AT RISK")
        if n_exit:
            parts.append(f"⚠️ {n_exit} EXIT SIGNALS")
        if avg_health >= 0.6:
            parts.append(f"📊 Avg health: {avg_health:.0%} (good)")
        else:
            parts.append(f"📊 Avg health: {avg_health:.0%}")
        pm_summary = " | ".join(parts)

        return PortfolioMonitorSummary(
            n_trades=n,
            n_healthy=n_healthy,
            n_aging=n_aging,
            n_at_risk=n_at_risk,
            n_critical=n_critical,
            n_exit_signals=n_exit,
            avg_health=round(avg_health, 4),
            avg_z_compression=round(avg_z_comp, 4),
            avg_time_decay=round(avg_time, 4),
            total_pnl_proxy=round(total_pnl, 6),
            regime_safety_current=current_safety_score,
            trade_reports=reports,
            urgent_exits=urgent,
            pm_summary=pm_summary,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Summary for agents / API
# ─────────────────────────────────────────────────────────────────────────────

    def monitor_all(
        self,
        active_tickets: list,
        master_df: Optional[pd.DataFrame] = None,
    ) -> PortfolioMonitorSummary:
        """
        Convenience method: extract z-scores from master_df and run monitor_portfolio.
        Used by EngineService to avoid manual z-score extraction.
        """
        z_map = {}
        safety_score = 1.0
        safety_label = "SAFE"
        days_map = {}
        pnl_map = {}

        if master_df is not None and not master_df.empty:
            # Extract z-scores from master_df
            for _, row in master_df.iterrows():
                ticker = str(row.get("sector_ticker", ""))
                z = row.get("pca_residual_z", row.get("z_score", float("nan")))
                if ticker and isinstance(z, (int, float)):
                    z_map[ticker] = float(z) if math.isfinite(float(z)) else 0.0

            # Extract safety
            if "market_state" in master_df.columns:
                ms = str(master_df["market_state"].iloc[0])
                safety_label = {"CALM": "SAFE", "NORMAL": "SAFE", "TENSION": "CAUTION",
                                "CRISIS": "DANGER"}.get(ms, "SAFE")

        return self.monitor_portfolio(
            active_tickets=active_tickets,
            current_zscores=z_map,
            current_safety_score=safety_score,
            current_safety_label=safety_label,
            days_held_map=days_map,
            pnl_map=pnl_map,
        )


# ═════════════════════════════════════════════════════════════════════════════
# Trailing Stop Logic
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class TrailingStopState:
    """Tracking state for trailing stop on a position."""
    trade_id: str
    high_water_mark: float = 0.0     # Best P&L % seen
    trailing_stop_level: float = 0.0  # Current stop level (HWM - trail_distance)
    trail_distance_pct: float = 0.015 # Distance from HWM to trigger stop
    is_triggered: bool = False
    trigger_pnl: float = 0.0         # P&L when stop was triggered


def compute_trailing_stop(
    current_pnl_pct: float,
    hwm: float,
    trail_distance: float = 0.015,
    activation_pnl: float = 0.005,
) -> TrailingStopState:
    """
    Compute trailing stop state.

    Trailing stop only activates after position reaches activation_pnl (0.5% profit).
    Once activated, stop follows HWM down by trail_distance (1.5%).

    Parameters
    ----------
    current_pnl_pct : float — current unrealized P&L (% of notional)
    hwm : float — highest P&L % seen since entry
    trail_distance : float — distance from HWM to stop (default 1.5%)
    activation_pnl : float — minimum profit to activate trailing stop (default 0.5%)
    """
    # Update HWM
    new_hwm = max(hwm, current_pnl_pct)

    # Only activate after reaching activation threshold
    if new_hwm < activation_pnl:
        return TrailingStopState(
            trade_id="", high_water_mark=new_hwm,
            trailing_stop_level=float("-inf"),
            trail_distance_pct=trail_distance,
        )

    # Compute stop level
    stop_level = new_hwm - trail_distance
    is_triggered = current_pnl_pct <= stop_level

    return TrailingStopState(
        trade_id="",
        high_water_mark=new_hwm,
        trailing_stop_level=stop_level,
        trail_distance_pct=trail_distance,
        is_triggered=is_triggered,
        trigger_pnl=current_pnl_pct if is_triggered else 0.0,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Greek Exposure Monitoring
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class GreekExposure:
    """Greek exposure for a single position."""
    ticker: str
    direction: str
    notional: float
    # Portfolio-level greeks (synthetic from factor betas)
    delta_spy: float = 0.0       # $ exposure to SPY 1% move
    delta_tnx: float = 0.0       # $ exposure to TNX 10bp move
    delta_dxy: float = 0.0       # $ exposure to DXY 1% move
    # Volatility exposure
    vega_proxy: float = 0.0      # $ exposure to VIX 1-point move
    # Time exposure
    theta_proxy: float = 0.0     # Daily time decay ($)


@dataclass
class PortfolioGreekSummary:
    """Aggregate greek exposure across all positions."""
    net_delta_spy: float = 0.0
    net_delta_tnx: float = 0.0
    net_delta_dxy: float = 0.0
    net_vega: float = 0.0
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    positions: List[GreekExposure] = field(default_factory=list)
    # Risk alerts
    alerts: List[str] = field(default_factory=list)


def compute_portfolio_greeks(
    positions: List[Dict],
    master_df: Optional[pd.DataFrame] = None,
) -> PortfolioGreekSummary:
    """
    Compute portfolio-level greek exposure from positions + factor betas.

    Uses beta_spy, beta_tnx, beta_dxy from master_df to estimate
    synthetic greeks for each position.
    """
    exposures = []
    total_long = 0.0
    total_short = 0.0
    net_spy = 0.0
    net_tnx = 0.0
    net_dxy = 0.0
    net_vega = 0.0

    for pos in positions:
        ticker = pos.get("ticker", "")
        direction = pos.get("direction", "LONG")
        notional = pos.get("notional", 0)
        sign = 1.0 if direction == "LONG" else -1.0

        if direction == "LONG":
            total_long += notional
        else:
            total_short += notional

        # Get factor betas from master_df
        beta_spy = 1.0
        beta_tnx = 0.0
        beta_dxy = 0.0
        if master_df is not None and "sector_ticker" in master_df.columns:
            row = master_df[master_df["sector_ticker"] == ticker]
            if not row.empty:
                beta_spy = float(row.iloc[0].get("beta_spy_delta", 1.0) or 1.0)
                beta_tnx = float(row.iloc[0].get("beta_tnx_60d", 0.0) or 0.0)
                beta_dxy = float(row.iloc[0].get("beta_dxy_60d", 0.0) or 0.0)

        # Dollar exposure per factor
        d_spy = sign * notional * beta_spy * 0.01   # Per 1% SPY move
        d_tnx = sign * notional * beta_tnx * 0.001  # Per 10bp TNX move
        d_dxy = sign * notional * beta_dxy * 0.01    # Per 1% DXY move
        vega = sign * notional * 0.002               # Rough: 20bps per VIX point
        theta = -abs(notional) * 0.0001              # ~1bps/day time decay for sector ETF

        exposures.append(GreekExposure(
            ticker=ticker, direction=direction, notional=notional,
            delta_spy=round(d_spy, 2), delta_tnx=round(d_tnx, 2),
            delta_dxy=round(d_dxy, 2), vega_proxy=round(vega, 2),
            theta_proxy=round(theta, 2),
        ))

        net_spy += d_spy
        net_tnx += d_tnx
        net_dxy += d_dxy
        net_vega += vega

    gross = total_long + total_short
    net = total_long - total_short

    # Risk alerts
    alerts = []
    if abs(net_spy) > gross * 0.3:
        alerts.append(f"High directional SPY exposure: ${net_spy:+,.0f} per 1% move")
    if abs(net_vega) > gross * 0.01:
        alerts.append(f"Significant vol exposure: ${net_vega:+,.0f} per VIX point")
    if abs(net / gross) > 0.3 if gross > 0 else False:
        alerts.append(f"Net exposure {net/gross:.0%} exceeds 30% threshold")

    return PortfolioGreekSummary(
        net_delta_spy=round(net_spy, 2),
        net_delta_tnx=round(net_tnx, 2),
        net_delta_dxy=round(net_dxy, 2),
        net_vega=round(net_vega, 2),
        gross_exposure=round(gross, 2),
        net_exposure=round(net, 2),
        positions=exposures,
        alerts=alerts,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Position Aging Analysis
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class PositionAgingResult:
    """Position aging analysis with severity scoring."""
    trade_id: str
    ticker: str
    days_held: int
    half_life: float                  # Expected MR half-life
    hl_ratio: float                   # days_held / half_life (>2 = overstaying)
    aging_score: float                # 0-1 (1 = fresh, 0 = severely aged)
    aging_label: str                  # "FRESH" / "MATURING" / "AGING" / "EXPIRED"
    alpha_decay_pct: float            # Estimated alpha remaining (from OU decay)
    recommended_action: str           # "HOLD" / "REDUCE" / "EXIT"


def analyse_position_aging(
    trade_id: str,
    ticker: str,
    days_held: int,
    half_life: float,
    current_pnl_pct: float = 0.0,
    max_hold_days: int = 30,
) -> PositionAgingResult:
    """
    Score position aging based on holding period vs expected half-life.

    Alpha decay model: alpha_remaining = exp(-ln(2) * days_held / half_life)
    """
    if not math.isfinite(half_life) or half_life <= 0:
        half_life = 20  # Default assumption

    hl_ratio = days_held / half_life if half_life > 0 else float("inf")
    alpha_remaining = math.exp(-math.log(2) * days_held / half_life)

    # Aging score: 1 = fresh, 0 = expired
    aging_score = max(0, min(1.0, alpha_remaining))

    # Classification
    if hl_ratio < 0.5:
        label = "FRESH"
        action = "HOLD"
    elif hl_ratio < 1.0:
        label = "MATURING"
        action = "HOLD"
    elif hl_ratio < 2.0:
        label = "AGING"
        action = "REDUCE" if current_pnl_pct > 0 else "HOLD"
    else:
        label = "EXPIRED"
        action = "EXIT"

    # Override if past max hold
    if days_held >= max_hold_days:
        label = "EXPIRED"
        action = "EXIT"

    return PositionAgingResult(
        trade_id=trade_id,
        ticker=ticker,
        days_held=days_held,
        half_life=round(half_life, 1),
        hl_ratio=round(hl_ratio, 2),
        aging_score=round(aging_score, 4),
        aging_label=label,
        alpha_decay_pct=round(alpha_remaining * 100, 1),
        recommended_action=action,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Summary for agents / API
# ─────────────────────────────────────────────────────────────────────────────

def monitor_summary_compact(summary: PortfolioMonitorSummary) -> dict:
    """Compact serializable summary for Claude / AgentBus."""
    return {
        "n_trades": summary.n_trades,
        "health": {
            "healthy": summary.n_healthy,
            "aging": summary.n_aging,
            "at_risk": summary.n_at_risk,
            "critical": summary.n_critical,
            "avg_score": summary.avg_health,
        },
        "exits": {
            "n_signals": summary.n_exit_signals,
            "urgent": [
                {
                    "trade_id": r.trade_id,
                    "ticker": r.ticker,
                    "action": r.recommended_action,
                    "health": r.health_label,
                    "reason": r.primary_signal.reason if r.primary_signal else "",
                    "urgency": r.primary_signal.urgency if r.primary_signal else "",
                }
                for r in summary.urgent_exits[:5]
            ],
        },
        "pnl_proxy": summary.total_pnl_proxy,
        "regime_safety": summary.regime_safety_current,
        "pm_summary": summary.pm_summary,
    }
